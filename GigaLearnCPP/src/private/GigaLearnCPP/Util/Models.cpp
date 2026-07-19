#include "Models.h"

#include <torch/csrc/api/include/torch/serialize.h>
#include <torch/csrc/api/include/torch/nn/utils/convert_parameters.h>
#include <torch/nn/modules/normalization.h>

GGL::Model::Model(
	const char* modelName,
	ModelConfig config,
	torch::Device device) : 
	modelName(modelName), device(device), seq({}), seqHalf({}), config(config) {

	if (!config.IsValid())
		RG_ERR_CLOSE("Failed to create model \"" << modelName << "\" with invalid config");

	int lastSize = config.numInputs;
	for (int i = 0; i < config.layerSizes.size(); i++) {
		seq->push_back(torch::nn::Linear(lastSize, config.layerSizes[i]));
		if (config.addLayerNorm)
			seq->push_back(torch::nn::LayerNorm(torch::nn::LayerNormOptions({(int64_t)config.layerSizes[i]})));
		lastSize = config.layerSizes[i];
		AddActivationFunc(seq, config.activationType);
	}
	
	if (config.addOutputLayer) {
		seq->push_back(torch::nn::Linear(lastSize, config.numOutputs));
	} else {
		// Write the MEMBER, not the shadowing parameter: the member was already copied in the
		// init list, so assigning `config.numOutputs` here was silently discarded and every
		// no-output-layer model reported numOutputs = 0 to readers of model->config.
		this->config.numOutputs = (int)config.layerSizes.back();
	}

	register_module("seq", seq);
	seq->to(device);
	optim = MakeOptimizer(config.optimType, this->parameters(), 0);
}

torch::Tensor GGL::Model::Forward(torch::Tensor input, bool halfPrec) {

	if (torch::GradMode::is_enabled())
		halfPrec = false;

	if (halfPrec) {

		if (_seqHalfOutdated) {
			_seqHalfOutdated = false;

			if (seqHalf->size() == 0) {
				for (auto& mod : *seq)
					seqHalf->push_back(mod.clone());
				seqHalf->to(RG_HALFPERC_TYPE, true);
			} else {
				auto fromParams = seq->parameters();
				auto toParams = seqHalf->parameters();
				for (int i = 0; i < fromParams.size(); i++) {
					auto scaledParams = fromParams[i].to(RG_HALFPERC_TYPE, true);
					toParams[i].copy_(scaledParams, true);
				}
			}
		}
		
		auto halfInput = input.to(RG_HALFPERC_TYPE);
		auto halfOutput = seqHalf->forward(halfInput);
		return halfOutput.to(torch::kFloat);
	} else {
		return seq->forward(input);
	}
}

// Get sizes of all parameters in a sequence
std::vector<uint64_t> GetSeqSizes(torch::nn::Sequential& seq) {
	std::vector<uint64_t> result = {};

	for (int i = 0; i < seq->size(); i++)
		for (auto param : seq[i]->parameters())
			result.push_back(param.numel());

	return result;
}

void GGL::Model::SetOptimLR(float newLR) {
	SetOptimizerLR(optim, config.optimType, newLR);
}

void GGL::Model::StepOptim() {
	optim->step();
	optim->zero_grad();
	_seqHalfOutdated = true;
}

void GGL::Model::Save(std::filesystem::path folder, bool saveOptim) {
	std::filesystem::path path = GetSavePath(folder);
	auto streamOut = std::ofstream(path, std::ios::binary);
	torch::save(seq, streamOut);

	if (saveOptim) {
		torch::serialize::OutputArchive optimArchive;
		optim->save(optimArchive);
		optimArchive.save_to(GetOptimSavePath(folder).string());
	}
}

void GGL::Model::Load(std::filesystem::path folder, bool allowNotExist, bool loadOptim) {
	std::filesystem::path path = GetSavePath(folder);

	if (!std::filesystem::exists(path)) {
		if (allowNotExist) {
			RG_LOG("Warning: Model \"" << modelName << "\" does not exist in " << folder << " and will be reset");
			return;
		} else {
			RG_ERR_CLOSE("Model \"" << modelName << "\" does not exist in " << folder);
		}
	}

	auto streamIn = std::ifstream(path, std::ios::binary);
	streamIn >> std::noskipws;

	if (!streamIn.good())
		RG_ERR_CLOSE("Failed to load from " << path << ", file does not exist or can't be accessed");

	auto sizesBefore = GetSeqSizes(seq);

	try {
		torch::load(this->seq, streamIn, device);
	} catch (std::exception& e) {
		RG_ERR_CLOSE(
			"Failed to load model \"" << modelName << ", checkpoint may be corrupt or of different model arch.\n" <<
			"Exception: " << e.what()
		);
	}

	// Torch will happily load in a model of a totally different size, then we will crash when we try to use it
	// So we need to manually check if it is the same size
	auto sizesAfter = GetSeqSizes(seq);
	if (!std::equal(sizesBefore.begin(), sizesBefore.end(), sizesAfter.begin(), sizesAfter.end())) {

		// Ladder wire migration: a pre-wire checkpoint differs ONLY in the first
		// Linear's weight (input columns short by exactly allowInputExpand). Zero-pad
		// those columns in place - the padded net is bit-exact with the old one on any
		// input (the new columns multiply into nothing) - and skip the optimizer load
		// below (its state tensors carry the old shapes; one-time momentum reset).
		bool migrated = false;
		if (allowInputExpand > 0) {
			torch::NoGradGuard noGrad;
			auto params = seq->parameters();
			// The first parameter of the first Linear is its weight [out, in]
			if (!params.empty() && params[0].dim() == 2) {
				auto& w = params[0];
				int64_t out = w.size(0), inOld = w.size(1);
				// Verify the ONLY mismatch is that weight's column count
				std::vector<uint64_t> expectedOld = sizesBefore;
				expectedOld[0] = (uint64_t)(out * (inOld + allowInputExpand)) == sizesBefore[0]
					? (uint64_t)(out * inOld) : 0; // 0 = shapes don't line up, fall through
				if (expectedOld[0] != 0
					&& std::equal(sizesAfter.begin(), sizesAfter.end(), expectedOld.begin(), expectedOld.end())) {
					auto padded = torch::cat({ w.to(device),
						torch::zeros({ out, (int64_t)allowInputExpand },
							w.options().device(device)) }, 1);
					w.set_data(padded);
					_seqHalfOutdated = true;
					migrated = true;
					loadOptim = false; // old-shape state; reset (logged below)
					RG_LOG("Model \"" << modelName << "\": MIGRATED from " << path
						<< " - first-layer input " << inOld << " -> " << (inOld + allowInputExpand)
						<< ", new wire columns zero-init (behaviorally exact), optimizer RESET");
				}
			}
		}

		if (!migrated) {
			std::stringstream stream;
			stream << "Saved model has different size than current model, cannot load model from " << path << ":\n";

			for (int i = 0; i < 2; i++) {
				stream << " > " << (i ? "Saved model:   [ " : "Current model: [ ");
				for (uint64_t size : (i ? sizesAfter : sizesBefore))
					stream << size << ' ';

				stream << " ]";
				if (i == 0)
					stream << ",\n";
			}

			RG_ERR_CLOSE(stream.str());
		}
	}

	/////////////////////////////

	if (loadOptim) {
		std::filesystem::path optimPath = GetOptimSavePath(folder);

		if (std::filesystem::exists(optimPath)) {
			std::ifstream testStream = std::ifstream(optimPath, std::istream::ate | std::ios::binary);
			if (testStream.tellg() > 0) {
				torch::serialize::InputArchive optimArchive;
				optimArchive.load_from(optimPath.string(), device);
				optim->load(optimArchive);
			} else {
				RG_LOG("WARNING: Saved optimizer at " << optimPath << " is empty, optimizer will be reset");
			}
		} else {
			RG_LOG("WARNING: No optimizer found at " << optimPath << ", optimizer will be reset");
		}
	}
}

torch::Tensor GGL::Model::CopyParams() const {
	return torch::nn::utils::parameters_to_vector(parameters()).cpu();
}
