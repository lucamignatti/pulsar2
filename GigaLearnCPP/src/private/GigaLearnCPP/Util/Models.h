#pragma once
#include <RLGymCPP/Framework.h>
#include "../FrameworkTorch.h"

#include <torch/nn/modules/activation.h>
#include <torch/nn/modules/container/sequential.h>

#include <torch/optim/adam.h>

#include "Muon.h"

#include <GigaLearnCPP/PPO/PPOLearnerConfig.h>
#include <GigaLearnCPP/Util/ModelConfig.h>

namespace GGL {

	inline void AddActivationFunc(torch::nn::Sequential& seq, ModelActivationType type) {
		switch (type) {
		case ModelActivationType::RELU:
			seq->push_back(torch::nn::ReLU());
			return;
		case ModelActivationType::LEAKY_RELU:
			seq->push_back(torch::nn::LeakyReLU());
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
		case ModelOptimType::MUON:
			return new Muon(parameters, MuonOptions(lr).momentum(0.95).nesterov(true));
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
			case ModelOptimType::MUON:
				static_cast<MuonOptions&>(group.options()).lr(lr);
				break;
			default:
				RG_ERR_CLOSE("Unknown optimizer type: " << (int)type);
			}
		}
	}

	//////////////////////////

	// Forward one type-erased module of a Model's flat seq. Mirrors the cast chain in
	// PolicySlots::Forward - both walk the same module list, so both must know the same types.
	torch::Tensor ForwardSeqModule(const std::shared_ptr<torch::nn::Module>& mod, torch::Tensor x);

	class Model : public torch::nn::Module {
	public:
		const char* modelName;
		torch::Device device;
		torch::nn::Sequential seq, seqHalf;
		bool _seqHalfOutdated = true;
		ModelConfig config;

		// Residual block spans over seq's module indices, as (firstModule, lastModule):
		// the input to firstModule is added back to the output of lastModule (which is the
		// block's final LayerNorm, so the add lands BEFORE the block's trailing activation).
		// Empty unless config.addResiduals - and when empty, Forward takes the original
		// straight-through seq->forward() path, so non-residual models are bit-identical.
		std::vector<std::pair<int, int>> residualSpans;

		torch::optim::Optimizer* optim;

		// When true, ModelSet::StepOptims() skips this model — it is stepped
		// independently by its owning module (e.g. ProposerModule::Train()).
		bool groupStepExempt = false;


		Model() : config(PartialModelConfig{}), device({}), modelName(NULL) {} // Uninitialized init

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
			Model* clone = new Model(modelName, config, device);
			return clone;
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

		virtual ~Model() = default;
	};

	class ModelSet {
	public:
		std::map<std::string, Model*> map = {};

		Model* operator[](const std::string& name) { 
			auto itr = map.find(name);
			if (itr == map.end()) {
				return NULL;
			} else {
				return map[name];
			}
		};

		void Add(Model* model) {
			map[model->modelName] = model;
		}

		// NOTE: Automatically zeros grad after
		void StepOptims() {
			for (Model* model : *this) {
				if (model->groupStepExempt)
					continue;
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

		void Free() {
			for (Model* model : *this)
				delete model;
			map.clear();
		}
	};
}