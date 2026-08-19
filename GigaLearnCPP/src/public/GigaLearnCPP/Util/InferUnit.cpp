#include "InferUnit.h"

#include <GigaLearnCPP/Util/Models.h>
#include <GigaLearnCPP/PPO/PPOLearner.h>

#include <torch/script.h>
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

GGL::InferUnit::~InferUnit() {
	if (models) {
		models->Free();
		delete models;
		models = nullptr;
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

std::vector<int> GGL::ReadLayerSizesFromModule(const std::string& ltPath, bool dropOutput) {
	std::vector<int> sizes;
	torch::jit::script::Module m = torch::jit::load(ltPath, torch::kCPU);
	for (const auto& p : m.named_parameters()) {
		if (p.value.dim() == 2)
			sizes.push_back((int)p.value.size(0));
	}
	if (dropOutput && !sizes.empty())
		sizes.pop_back();
	return sizes;
}

bool GGL::ReadMoEConfigFromModule(const std::string& ltPath, PartialModelConfig& cfgOut, int topK) {
	// An MoE trunk is not describable by ReadLayerSizesFromModule: the expert stacks are
	// 3-D so it skips them entirely, while each block's 2-D ROUTER weight [E, dim] gets
	// read as an "E-wide layer". That is why every MoE checkpoint died in the deployment
	// path with "addResiduals requires uniform layerSizes, got 64 among 1024-wide layers"
	// — so no MoE bot could be evaluated offline or played in a real match.
	//
	// The geometry is fully recoverable from the weights: expertW1 is [E, hidden, dim]
	// and expertW2 is [E, dim, hidden], one pair per block. Only topK is not stored
	// (it is a routing scalar, not a parameter), so the caller supplies it.
	torch::jit::script::Module m = torch::jit::load(ltPath, torch::kCPU);

	int experts = 0, hidden = 0, dim = 0, expertTensors = 0;
	for (const auto& p : m.named_parameters()) {
		if (p.value.dim() != 3)
			continue;
		expertTensors++;
		const int e = (int)p.value.size(0);
		const int a = (int)p.value.size(1), b = (int)p.value.size(2);
		experts = e;
		// W1 is [E, hidden, dim] and W2 is [E, dim, hidden]; hidden is the larger of the
		// two inner dims for an expanding FFN, but take it from W1 specifically by
		// remembering the pair-min/max rather than assuming which we hit first.
		if (hidden == 0 || dim == 0) { hidden = std::max(a, b); dim = std::min(a, b); }
	}
	if (expertTensors == 0)
		return false; // dense checkpoint, caller keeps its existing path

	if (expertTensors % 2 != 0)
		RG_ERR_CLOSE("ReadMoEConfigFromModule: " << ltPath << " has " << expertTensors
			<< " 3-D expert tensors, expected an even number (W1/W2 per block)");

	cfgOut.layerSizes = { dim };
	cfgOut.addResiduals = false;      // MoE blocks carry their own residual skips
	cfgOut.moeBlocks = expertTensors / 2;
	cfgOut.moeExperts = experts;
	cfgOut.moeHidden = hidden;
	cfgOut.moeTopK = topK;
	RG_LOG("MoE checkpoint detected: " << cfgOut.moeBlocks << " blocks x " << experts
		<< " experts, width " << dim << ", hidden " << hidden << ", top-" << topK);
	return true;
}
