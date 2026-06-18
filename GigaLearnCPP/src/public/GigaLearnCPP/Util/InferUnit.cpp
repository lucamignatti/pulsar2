#include "InferUnit.h"

#include <GigaLearnCPP/Util/Models.h>
#include <GigaLearnCPP/PPO/PPOLearner.h>
#include <torch/cuda.h>
#if defined(__APPLE__) && __has_include(<torch/mps.h>)
#include <torch/mps.h>
#define RG_MPS_SUPPORT
#endif

static torch::Device ResolveInferDevice(bool useGPU) {
	if (!useGPU)
		return torch::Device(torch::kCPU);

#ifdef __APPLE__
#ifdef RG_MPS_SUPPORT
	if (torch::mps::is_available())
		return torch::Device(torch::kMPS);
#endif
	if (torch::cuda::is_available())
		return torch::Device(torch::kCUDA);
#else
	if (torch::cuda::is_available())
		return torch::Device(torch::kCUDA);
#ifdef RG_MPS_SUPPORT
	if (torch::mps::is_available())
		return torch::Device(torch::kMPS);
#endif
#endif

	RG_ERR_CLOSE("InferUnit: useGPU was requested but neither MPS nor CUDA is available to libtorch.");
	return torch::Device(torch::kCPU);
}

GGL::InferUnit::InferUnit(
	RLGC::ObsBuilder* obsBuilder, int obsSize, RLGC::ActionParser* actionParser,
	PartialModelConfig sharedHeadConfig, PartialModelConfig policyConfig, 
	std::filesystem::path modelsFolder, bool useGPU) : 
	obsBuilder(obsBuilder), obsSize(obsSize), actionParser(actionParser), useGPU(useGPU) {

	this->models = new ModelSet();
	auto device = ResolveInferDevice(useGPU);

	try {
		PPOLearner::MakeModels(
			false, obsSize, actionParser->GetActionAmount(),
			sharedHeadConfig, policyConfig, {},
			device,
			*this->models
		);
	} catch (std::exception& e) {
		RG_ERR_CLOSE("InferUnit: Exception when trying to construct models: " << e.what());
	}

	try {
		this->models->Load(modelsFolder, false, false);
	} catch (std::exception& e) {
		RG_ERR_CLOSE("InferUnit: Exception when trying to load models: " << e.what());
	}
}

RLGC::Action GGL::InferUnit::InferAction(const RLGC::Player& player, const RLGC::GameState& state, bool deterministic, float temperature) {
	return BatchInferActions({ player }, { state }, deterministic, temperature)[0];
}

std::vector<RLGC::Action> GGL::InferUnit::BatchInferActions(const std::vector<RLGC::Player>& players, const std::vector<RLGC::GameState>& states, bool deterministic, float temperature) {
	RG_ASSERT(players.size() > 0 && states.size() > 0);
	RG_ASSERT(players.size() == states.size());

	int batchSize = players.size();
	std::vector<float> allObs;
	std::vector<uint8_t> allActionMasks;
	for (int i = 0; i < batchSize; i++) {
		FList curObs = obsBuilder->BuildObs(players[i], states[i]);
		if (curObs.size() != obsSize) {
			RG_ERR_CLOSE(
				"InferUnit: Obs builder produced an obs that differs from the provided size (expected: " << obsSize << ", got: " << curObs.size() << ")\n" <<
				"Make sure you provided the correct obs size to the InferUnit constructor.\n" <<
				"Also, make sure there aren't an incorrect number of players (there are " << states[i].players.size() << " in this state)"
			);
		}
		allObs += curObs;

		allActionMasks += actionParser->GetActionMask(players[i], states[i]);
	}
	
	std::vector<RLGC::Action> results = {};

	try {
		RG_NO_GRAD;

		auto device = ResolveInferDevice(useGPU);

		auto tObs = torch::tensor(allObs).reshape({(int64_t)players.size(), obsSize});
		auto tActionMasks = torch::tensor(allActionMasks).reshape({(int64_t)players.size(), this->actionParser->GetActionAmount()});

		tObs = tObs.to(device);
		tActionMasks = tActionMasks.to(device);
		torch::Tensor tActions, tLogProbs;

		PPOLearner::InferActionsFromModels(*models, tObs, tActionMasks, deterministic, temperature, false, &tActions, &tLogProbs);

		auto actionIndices = TENSOR_TO_VEC<int>(tActions);
		
		for (int i = 0; i < batchSize; i++) 
			results.push_back(actionParser->ParseAction(actionIndices[i], players[i], states[i]));

	} catch (std::exception& e) {
		RG_ERR_CLOSE("InferUnit: Exception when inferring model: " << e.what());
	}

	return results;
}
