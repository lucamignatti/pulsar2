#include "InferUnit.h"

#include <GigaLearnCPP/Util/Models.h>
#include <GigaLearnCPP/PPO/PPOLearner.h>

#include <torch/cuda.h>
#include <torch/mps.h>

// useGPU means "best available GPU": CUDA, else MPS (Apple Metal), else CPU
static at::Device BestInferDevice(bool useGPU) {
	if (useGPU) {
		if (torch::cuda::is_available())
			return at::Device(at::kCUDA);
		if (torch::mps::is_available())
			return at::Device(at::kMPS);
	}
	return at::Device(at::kCPU);
}

GGL::InferUnit::InferUnit(
	RLGC::ObsBuilder* obsBuilder, int obsSize, RLGC::ActionParser* actionParser,
	PartialModelConfig sharedHeadConfig, PartialModelConfig policyConfig,
	std::filesystem::path modelsFolder, bool useGPU) :
	obsBuilder(obsBuilder), obsSize(obsSize), actionParser(actionParser), useGPU(useGPU) {

	this->models = new ModelSet();

	try {
		PPOLearner::MakeModels(
			false, obsSize, actionParser->GetActionAmount(),
			sharedHeadConfig, policyConfig, {},
			/*criticTrunkConfig=*/{},   // inference-only: no critics, so no critic trunk
			BestInferDevice(useGPU),
			*this->models);
	} catch (std::exception& e) {
		RG_ERR_CLOSE("InferUnit: Exception when trying to construct models: " << e.what());
	}

	try {
		this->models->Load(modelsFolder, false, false);
	} catch (std::exception& e) {
		RG_ERR_CLOSE("InferUnit: Exception when trying to load models: " << e.what());
	}
}

RLGC::Action GGL::InferUnit::InferAction(const RLGC::Player& player, const RLGC::GameState& state, bool deterministic, float temperature, InferDebug* debugOut) {
	std::vector<InferDebug> debugRows;
	auto result = BatchInferActions({ player }, { state }, deterministic, temperature,
		debugOut ? &debugRows : nullptr)[0];
	if (debugOut && !debugRows.empty())
		*debugOut = std::move(debugRows[0]);
	return result;
}

std::vector<RLGC::Action> GGL::InferUnit::BatchInferActions(const std::vector<RLGC::Player>& players, const std::vector<RLGC::GameState>& states, bool deterministic, float temperature, std::vector<InferDebug>* debugOut) {
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

		auto device = BestInferDevice(useGPU);

		auto tObs = torch::tensor(allObs).reshape({(int64_t)players.size(), obsSize});
		auto tActionMasks = torch::tensor(allActionMasks).reshape({(int64_t)players.size(), this->actionParser->GetActionAmount()});

		tObs = tObs.to(device);
		tActionMasks = tActionMasks.to(device);
		torch::Tensor tActions, tLogProbs;

		PPOLearner::InferActionsFromModels(*models, tObs, tActionMasks, deterministic, temperature, false, &tActions, &tLogProbs);

		auto actionIndices = TENSOR_TO_VEC<int>(tActions);

		for (int i = 0; i < batchSize; i++)
			results.push_back(actionParser->ParseAction(actionIndices[i], players[i], states[i]));

		if (debugOut) {
			debugOut->resize(batchSize);
			const int maskWidth = actionParser->GetActionAmount();
			for (int i = 0; i < batchSize; i++) {
				auto& dbg = (*debugOut)[i];
				dbg.obs.assign(allObs.begin() + (size_t)i * obsSize,
					allObs.begin() + (size_t)(i + 1) * obsSize);
				dbg.actionMask.assign(allActionMasks.begin() + (size_t)i * maskWidth,
					allActionMasks.begin() + (size_t)(i + 1) * maskWidth);
				dbg.actionIndex = actionIndices[i];
			}
		}

	} catch (std::exception& e) {
		RG_ERR_CLOSE("InferUnit: Exception when inferring model: " << e.what());
	}

	return results;
}