#pragma once
#include <RLGymCPP/Framework.h>
#include "../FrameworkTorch.h"

#include <torch/nn/modules/activation.h>
#include <torch/nn/modules/container/sequential.h>

#include <torch/optim/adam.h>
#include <torch/optim/adamw.h>
#include <torch/optim/adagrad.h>
#include <torch/optim/rmsprop.h>
#include <torch/optim/sgd.h>

#include "MagSGD.h"
#include "Muon.h"

#include <GigaLearnCPP/PPO/PPOLearnerConfig.h>
#include <GigaLearnCPP/Util/ModelConfig.h>

#include <torch/nn/cloneable.h>

namespace GGL {

	// LayerNorm composed from primitives. The fused native_layer_norm(_backward)
	// produces non-finite gradients on ROCm; building it from mean/var/rsqrt avoids
	// that fused kernel. Normalizes over the last dimension (size normSize), with
	// learnable affine "weight"/"bias" named to match torch::nn::LayerNorm (so the
	// two are checkpoint-compatible). Cloneable so the half-precision seqHalf clone
	// path works.
	struct ManualLayerNormImpl : torch::nn::Cloneable<ManualLayerNormImpl> {
		int64_t normSize;
		double eps;
		torch::Tensor weight, bias;

		ManualLayerNormImpl(int64_t normSize, double eps = 1e-5) : normSize(normSize), eps(eps) {
			reset();
		}

		void reset() override {
			weight = register_parameter("weight", torch::ones({ normSize }));
			bias = register_parameter("bias", torch::zeros({ normSize }));
		}

		torch::Tensor forward(torch::Tensor x) {
			auto mean = x.mean(-1, /*keepdim=*/true);
			auto centered = x - mean;
			auto var = centered.pow(2).mean(-1, /*keepdim=*/true); // biased var (LayerNorm convention)
			auto normed = centered * (var + eps).rsqrt();
			return normed * weight + bias;
		}
	};
	TORCH_MODULE(ManualLayerNorm);

	inline void AddActivationFunc(torch::nn::Sequential& seq, ModelActivationType type) {
		switch (type) {
		case ModelActivationType::RELU:
			seq->push_back(torch::nn::ReLU());
			return;
		case ModelActivationType::LEAKY_RELU:
			seq->push_back(torch::nn::LeakyReLU());
			return;
		case ModelActivationType::SWISH:
			seq->push_back(torch::nn::SiLU());
			return;
		case ModelActivationType::SIGMOID:
			seq->push_back(torch::nn::Sigmoid());
			return;
		case ModelActivationType::TANH:
			seq->push_back(torch::nn::Tanh());
			return;
		}

		RG_ERR_CLOSE("Unknown activation function type: " << (int)type);
	}

	inline torch::optim::Optimizer* MakeOptimizer(ModelOptimType type, const std::vector<torch::Tensor>& parameters, float lr) {
		switch (type) {
		case ModelOptimType::ADAM:
			return new torch::optim::Adam(parameters, lr);
		case ModelOptimType::ADAMW:
			return new torch::optim::AdamW(parameters, lr);
		case ModelOptimType::ADAGRAD:
			return new torch::optim::Adagrad(parameters, lr);
		case ModelOptimType::RMSPROP:
			return new torch::optim::RMSprop(parameters, lr);
		case ModelOptimType::MAGSGD:
			return new MagSGD(parameters, lr);
		case ModelOptimType::MUON:
			return new Muon(parameters, MuonOptions(lr));
		}

		RG_ERR_CLOSE("Unknown optimizer type: " << (int)type);
		return NULL;
	}

	inline void SetOptimizerLR(torch::optim::Optimizer* optimizer, ModelOptimType type, float lr) {
		for (auto& group : optimizer->param_groups()) {
			switch (type) {
			case ModelOptimType::ADAM:
				static_cast<torch::optim::AdamOptions&>(group.options()).lr(lr);
				break;
			case ModelOptimType::ADAMW:
				static_cast<torch::optim::AdamWOptions&>(group.options()).lr(lr);
				break;
			case ModelOptimType::ADAGRAD:
				static_cast<torch::optim::AdagradOptions&>(group.options()).lr(lr);
				break;
			case ModelOptimType::RMSPROP:
				static_cast<torch::optim::RMSpropOptions&>(group.options()).lr(lr);
				break;
			case ModelOptimType::MAGSGD:
				static_cast<MagSGDOptions&>(group.options()).lr(lr);
				break;
			case ModelOptimType::MUON:
				static_cast<MuonOptions&>(group.options()).lr(lr);
				break;
			default:
				RG_ERR_CLOSE("Unknown optimizer type: " << (int)type);
			}
		}
	}

	//////////////////////////

	class Model : public torch::nn::Module {
	public:
		const char* modelName;
		torch::Device device;
		torch::nn::Sequential seq, seqHalf;
		bool _seqHalfOutdated = true;
		ModelConfig config;

		torch::optim::Optimizer* optim;

		Model() : config(PartialModelConfig{}), device({}), modelName(NULL), optim(NULL) {} // Uninitialized init

		Model(
			const char* modelName,
			ModelConfig config,
			torch::Device device
		);

		virtual torch::Tensor Forward(torch::Tensor input, bool halfPrec);
		
		void SetOptimLR(float newLR);

		void StepOptim();

		std::filesystem::path GetSuffixedSavePath(std::filesystem::path folder, std::string suffix) const {
			std::string filename = modelName + suffix ;
			for (char& c : filename)
				c = toupper(c);
			filename += ".lt";
			return folder / filename;
		}

		std::filesystem::path GetSavePath(std::filesystem::path folder) const {
			return GetSuffixedSavePath(folder, "");
		}

		std::filesystem::path GetOptimSavePath(std::filesystem::path folder) const {
			return GetSuffixedSavePath(folder, "_optim");
		}

		virtual void Save(std::filesystem::path folder, bool saveOptim = true);
		virtual void Load(std::filesystem::path folder, bool allowNotExist, bool loadOptim = true);

		virtual torch::Tensor CopyParams() const;

		// NOTE: Resets parameters
		Model* MakeEmptyClone() {
			return new Model(modelName, config, device);
		}

		Model* MakeClone() {
			RG_NO_GRAD;

			Model* clone = MakeEmptyClone();
			auto fromParams = this->parameters();
			auto toParams = clone->parameters();
			for (int i = 0; i < fromParams.size(); i++)
				toParams[i].copy_(fromParams[i], true);
			return clone;
		}

		uint64_t GetParamCount() {
			uint64_t total = 0;
			for (auto& param : this->parameters()) {
				if (!param.requires_grad())
					continue;
				total += param.numel();
			}

			return total;
		}

		virtual ~Model() {
			delete optim;
		}
	};

	class ModelSet {
	public:
		std::map<std::string, Model*> map = {};

		Model* operator[](const std::string& name) {
			auto itr = map.find(name);
			return (itr == map.end()) ? NULL : itr->second; // single lookup (was find + operator[])
		};

		void Add(Model* model) {
			map[model->modelName] = model;
		}

		// NOTE: Automatically zeros grad after
		void StepOptims() {
			for (Model* model : *this) {
				model->StepOptim();
			}
		}

		void Save(std::filesystem::path folder, bool saveOptims = true) {
			for (Model* model : *this)
				model->Save(folder, saveOptims);
		}

		void Load(std::filesystem::path folder, bool allowNotExist, bool loadOptims) {
			for (Model* model : *this)
				model->Load(folder, allowNotExist, loadOptims);
		}

		class ModelIterator : public std::iterator<std::forward_iterator_tag, Model*> {
		public:
			using MapItr = std::map<std::string, Model*>::iterator;
			MapItr _mapItr;

			ModelIterator(MapItr mapItr) : _mapItr(mapItr) {}

			ModelIterator& operator++() { ++_mapItr; return *this; }

			bool operator==(const ModelIterator& other) const { return _mapItr == other._mapItr; }
			bool operator!=(const ModelIterator& other) const { return _mapItr != other._mapItr; }

			Model*& operator*() const { return _mapItr->second; }
		};

		ModelIterator begin() {
			return map.begin();
		}

		ModelIterator end() {
			return map.end();
		}

		ModelSet CloneAll() {
			ModelSet clone = *this;
			for (Model*& model : clone)
				model = model->MakeClone();
			return clone;
		}

		// In-place copy of every model's parameters from src (which must have the same structure). Refreshes
		// a frozen actor clone for overlapped collection each iteration without reallocating the models.
		void CopyParamsFrom(ModelSet& src) {
			RG_NO_GRAD;
			for (auto& kv : map) {
				Model* s = src[kv.first];
				if (!s)
					continue;
				auto dp = kv.second->parameters();
				auto sp = s->parameters();
				for (size_t i = 0; i < dp.size() && i < sp.size(); i++)
					dp[i].copy_(sp[i], true);
			}
		}

		void Free() {
			for (Model* model : *this)
				delete model;
			map.clear();
		}
	};
}
