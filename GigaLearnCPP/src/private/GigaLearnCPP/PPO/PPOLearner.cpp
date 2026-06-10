#include "PPOLearner.h"

#include <torch/nn/utils/convert_parameters.h>
#include <torch/nn/utils/clip_grad.h>
#include <torch/csrc/api/include/torch/serialize.h>
#include <public/GigaLearnCPP/Util/AvgTracker.h>
#include <nlohmann/json.hpp>

#include <cmath>
#include <limits>

using namespace torch;

namespace {
	constexpr const char* PPO_STATE_FILE_NAME = "PPO_STATE.json";
	constexpr const char* SORS_STATE_FILE_NAME = "SORS_STATE.json";
	constexpr const char* SORS_REPLAY_STATES_FILE_NAME = "SORS_REPLAY_STATES.pt";
	constexpr const char* SORS_REPLAY_ACTIONS_FILE_NAME = "SORS_REPLAY_ACTIONS.pt";
	constexpr const char* SORS_REPLAY_LENGTHS_FILE_NAME = "SORS_REPLAY_LENGTHS.pt";
	constexpr const char* SORS_REPLAY_LABELS_FILE_NAME = "SORS_REPLAY_LABELS.pt";
}

GGL::PPOLearner::PPOLearner(int obsSize, int numActions, PPOLearnerConfig _config, Device _device) : config(_config), device(_device) {
	curEntropyScale = std::clamp(config.entropyScale, config.minEntropyScale, config.maxEntropyScale);
	curGCRLAdvScale = config.gcrlAdvScale;
	curGCRLRewardGateInfluence = config.gcrlRewardGateInfluence;
	curGCRLAerialRewardGateInfluence = config.gcrlAerialRewardGateInfluence;
	curCurriculumRewardScale = config.curriculumRewardScale;
	curAerialCurriculumRewardScale = config.aerialCurriculumRewardScale;
	curSORSRewardScale = config.sorsRewardScale;

	if (config.useGCRLRewardGate && !config.useGCRL)
		RG_ERR_CLOSE("PPOLearner: useGCRLRewardGate requires useGCRL");

	if (config.miniBatchSize == 0)
		config.miniBatchSize = config.batchSize;

	if (config.batchSize % config.miniBatchSize != 0)
		RG_ERR_CLOSE("PPOLearner: config.batchSize (" << config.batchSize << ") must be a multiple of config.miniBatchSize (" << config.miniBatchSize << ")");

	MakeModels(true, obsSize, numActions, config.sharedHead, config.policy, config.critic, device, models);

	// ── Quasimetric GCRL critics (game-sense channel) ──
	// They score (shared-features, action) -> goal, so their input dim is the shared-head
	// output (or raw obs if there is no shared head). Goals are 6-dim ball pos+vel.
	if (config.useGCRL) {
		int featureDim = config.sharedHead.IsValid() ? config.sharedHead.layerSizes.back() : obsSize;
		const int goalDim = 6;    // ball pos(3)+vel(3): global for goal/anti, car-local for car
		const int actionDim = 8;  // continuous action components

		auto makeGCRLCritic = [&](const char* name) {
			return new QuasimetricCritic(name,
				featureDim, actionDim, goalDim,
				config.gcrlCritic.layerSizes, config.gcrlReprDim,
				config.gcrlCritic.activationType, config.gcrlCritic.addLayerNorm,
				config.gcrlTau, config.gcrlVarReg, config.gcrlInfoNCEPenalty,
				config.gcrlCritic.optimType, device);
		};
		models.Add(makeGCRLCritic("goal_critic"));
		models.Add(makeGCRLCritic("anti_critic"));
		models.Add(makeGCRLCritic("car_critic"));
	}

	if (config.useSORS) {
		const int actionDim = 8;
		models.Add(new SORSRewardModel("sors_reward", obsSize, actionDim, config.sorsReward, device));
	}

	SetLearningRates(config.policyLR, config.criticLR);

	// Print param counts
	RG_LOG("Model parameter counts:");
	uint64_t total = 0;
	for (auto model : this->models) {
		uint64_t count = model->GetParamCount();
		RG_LOG("\t\"" << model->modelName << "\": " << Utils::NumToStr(count));
		total += count;
	}
	RG_LOG("\t[Total]: " << Utils::NumToStr(total));

	if (config.useGuidingPolicy) {
		RG_LOG("Guiding policy enabled, loading from " << config.guidingPolicyPath << "...");
		MakeModels(false, obsSize, numActions, config.sharedHead, config.policy, config.critic, device, guidingPolicyModels);
		guidingPolicyModels.Load(config.guidingPolicyPath, false, false);
	}
}

void GGL::PPOLearner::MakeModels(
	bool makeCritic,
	int obsSize, int numActions, 
	PartialModelConfig sharedHeadConfig, PartialModelConfig policyConfig, PartialModelConfig criticConfig,
	torch::Device device, 
	ModelSet& outModels) {

	ModelConfig fullPolicyConfig = policyConfig;
	fullPolicyConfig.numInputs = obsSize;
	fullPolicyConfig.numOutputs = numActions;

	ModelConfig fullCriticConfig = criticConfig;
	fullCriticConfig.numInputs = obsSize;
	fullCriticConfig.numOutputs = 1;

	if (sharedHeadConfig.IsValid()) {

		ModelConfig fullSharedHeadConfig = sharedHeadConfig;
		fullSharedHeadConfig.numInputs = obsSize;
		fullSharedHeadConfig.numOutputs = 0;

		RG_ASSERT(!sharedHeadConfig.addOutputLayer);

		fullPolicyConfig.numInputs = fullSharedHeadConfig.layerSizes.back();
		fullCriticConfig.numInputs = fullSharedHeadConfig.layerSizes.back();

		outModels.Add(new Model("shared_head", fullSharedHeadConfig, device));
	}

	outModels.Add(new Model("policy", fullPolicyConfig, device));

	if (makeCritic)
		outModels.Add(new Model("critic", fullCriticConfig, device));
}

torch::Tensor GGL::PPOLearner::InferPolicyProbsFromModels(
	ModelSet& models,
	torch::Tensor obs, torch::Tensor actionMasks,
	float temperature, bool halfPrec) {

	actionMasks = actionMasks.to(torch::kBool);

	constexpr float ACTION_MIN_PROB = 1e-11f;
	constexpr float ACTION_DISABLED_LOGIT = -1e10f;

	if (models["shared_head"])
		obs = models["shared_head"]->Forward(obs, halfPrec);

	auto logits = models["policy"]->Forward(obs, halfPrec) / temperature;

	auto result = torch::softmax(logits + ACTION_DISABLED_LOGIT * actionMasks.logical_not(), -1);
	return result.view({ -1, models["policy"]->config.numOutputs }).clamp(ACTION_MIN_PROB, 1);
}

void GGL::PPOLearner::InferActionsFromModels(
	ModelSet& models,
	torch::Tensor obs, torch::Tensor actionMasks, 
	bool deterministic, float temperature, bool halfPrec,
	torch::Tensor* outActions, torch::Tensor* outLogProbs) {

	auto probs = InferPolicyProbsFromModels(models, obs, actionMasks, temperature, halfPrec);

	if (deterministic) {
		auto action = probs.argmax(1);
		if (outActions)
			*outActions = action.flatten();
	} else {
		auto action = torch::multinomial(probs, 1, true);
		auto logProb = torch::log(probs).gather(-1, action);
		if (outActions)
			*outActions = action.flatten();

		if (outLogProbs)
			*outLogProbs = logProb.flatten();
	}
}

void GGL::PPOLearner::InferActions(torch::Tensor obs, torch::Tensor actionMasks, torch::Tensor* outActions, torch::Tensor* outLogProbs, ModelSet* models) {
	InferActionsFromModels(models ? *models : this->models, obs, actionMasks, config.deterministic, config.policyTemperature, config.useHalfPrecision, outActions, outLogProbs);
}

torch::Tensor GGL::PPOLearner::InferCritic(torch::Tensor obs) {

	if (models["shared_head"])
		obs = models["shared_head"]->Forward(obs, config.useHalfPrecision);

	return models["critic"]->Forward(obs, config.useHalfPrecision).flatten();
}

torch::Tensor GGL::PPOLearner::InferGCRLTerminalScores(torch::Tensor obs, torch::Tensor actionComps, torch::Tensor goalTargets, torch::Tensor antiTargets) {
	if (!config.useGCRLRewardGate || !config.useGCRL || !models["goal_critic"] || !models["anti_critic"])
		return {};

	torch::NoGradGuard noGrad;

	if (models["shared_head"])
		obs = models["shared_head"]->Forward(obs, config.useHalfPrecision);

	auto* gcGoal = dynamic_cast<QuasimetricCritic*>(models["goal_critic"]);
	auto* gcAnti = dynamic_cast<QuasimetricCritic*>(models["anti_critic"]);
	RG_ASSERT(gcGoal && gcAnti);

	auto qGoal = gcGoal->score_q(obs, actionComps, goalTargets).flatten();
	auto qAnti = gcAnti->score_q(obs, actionComps, antiTargets).flatten();
	return torch::stack({ qGoal, qAnti }, 1);
}

torch::Tensor GGL::PPOLearner::InferSORSRewards(torch::Tensor obs, torch::Tensor actionComps) {
	if (!config.useSORS || !models["sors_reward"])
		return {};

	auto* sors = dynamic_cast<SORSRewardModel*>(models["sors_reward"]);
	RG_ASSERT(sors);
	return sors->Forward(obs, actionComps, config.useHalfPrecision);
}

void GGL::PPOLearner::AddSORSWindows(std::vector<SORSWindow>&& windows) {
	if (!config.useSORS)
		return;

	for (auto& window : windows) {
		if (window.length <= 0)
			continue;

		sorsReplay.push_back(std::move(window));
		while ((int)sorsReplay.size() > config.sorsMaxReplayWindows)
			sorsReplay.pop_front();
	}
}

void GGL::PPOLearner::TrainSORS(Report& report) {
	if (!config.useSORS || !models["sors_reward"])
		return;

	report["SORS/Replay Windows"] = sorsReplay.size();
	if (sorsReplay.empty())
		return;

	int positives = 0;
	float labelTotal = 0;
	for (auto& window : sorsReplay) {
		positives += window.label > 0;
		labelTotal += window.label;
	}
	report["SORS/Positive Windows"] = positives;
	report["SORS/Avg Label"] = labelTotal / RS_MAX(1, (int)sorsReplay.size());

	if (sorsTrainCalls++ < config.sorsWarmupIters || sorsReplay.size() < 2)
		return;

	auto* sors = dynamic_cast<SORSRewardModel*>(models["sors_reward"]);
	RG_ASSERT(sors);

	// Sample (higher-labeled, lower-labeled) window pairs first, then train them in
	// batched chunks: one forward/backward per chunk with per-window segment sums,
	// instead of one forward/backward (and 4 host->device copies) per pair.
	std::vector<std::pair<const SORSWindow*, const SORSWindow*>> pairs;
	pairs.reserve(config.sorsTrainPairs);
	int attempts = 0;
	int maxAttempts = config.sorsTrainPairs * 8;

	while ((int)pairs.size() < config.sorsTrainPairs && attempts++ < maxAttempts) {
		int idxA = Math::RandInt(0, (int)sorsReplay.size());
		int idxB = Math::RandInt(0, (int)sorsReplay.size());
		if (idxA == idxB)
			continue;

		auto& a = sorsReplay[idxA];
		auto& b = sorsReplay[idxB];
		float delta = a.label - b.label;
		if (fabsf(delta) < config.sorsMinLabelDelta)
			continue;

		pairs.push_back(delta > 0 ? std::make_pair(&a, &b) : std::make_pair(&b, &a));
	}

	if (pairs.empty())
		return;

	MutAvgTracker avgLoss, avgPredReturn, avgAcc;

	constexpr int PAIRS_PER_BATCH = 128;
	for (size_t chunkStart = 0; chunkStart < pairs.size(); chunkStart += PAIRS_PER_BATCH) {
		size_t chunkEnd = RS_MIN(chunkStart + PAIRS_PER_BATCH, pairs.size());
		int numPairs = (int)(chunkEnd - chunkStart);

		// Concatenate windows interleaved [hi0, lo0, hi1, lo1, ...] with per-row window ids
		std::vector<torch::Tensor> statesList, actionsList;
		statesList.reserve((size_t)numPairs * 2);
		actionsList.reserve((size_t)numPairs * 2);
		std::vector<int64_t> windowIds;
		int64_t numWindows = 0;
		for (size_t p = chunkStart; p < chunkEnd; p++) {
			for (const SORSWindow* window : { pairs[p].first, pairs[p].second }) {
				statesList.push_back(window->states);
				actionsList.push_back(window->actionComps);
				windowIds.insert(windowIds.end(), window->length, numWindows);
				numWindows++;
			}
		}

		auto obs = torch::cat(statesList).to(device, RG_H2D_NONBLOCKING(device), true);
		auto acts = torch::cat(actionsList).to(device, RG_H2D_NONBLOCKING(device), true);
		auto tWindowIds = torch::tensor(windowIds, torch::TensorOptions().dtype(torch::kLong)).to(device, RG_H2D_NONBLOCKING(device), true);

		// Per-step rewards -> per-window return sums -> per-pair preference loss
		auto stepRewards = sors->Forward(obs, acts);
		auto windowReturns = torch::zeros({ numWindows }, stepRewards.options())
			.index_add_(0, tWindowIds, stepRewards);
		auto pairReturns = windowReturns.view({ numPairs, 2 });
		auto hiReturns = pairReturns.select(1, 0);
		auto loReturns = pairReturns.select(1, 1);

		auto losses = -torch::log(torch::sigmoid(hiReturns - loReturns) + 1e-8f);
		losses.sum().backward(); // Sum, not mean: matches the old accumulated per-pair backward

		auto lossesCpu = losses.detach().cpu();
		auto hiCpu = hiReturns.detach().cpu();
		auto loCpu = loReturns.detach().cpu();
		const float* lossData = lossesCpu.data_ptr<float>();
		const float* hiData = hiCpu.data_ptr<float>();
		const float* loData = loCpu.data_ptr<float>();
		for (int i = 0; i < numPairs; i++) {
			avgLoss += lossData[i];
			avgPredReturn += hiData[i];
			avgAcc += hiData[i] > loData[i] ? 1.0f : 0.0f;
		}
	}

	nn::utils::clip_grad_norm_(sors->parameters(), 0.5f);
	sors->StepOptim();
	report["SORS/Loss"] = avgLoss.Get();
	report["SORS/Pair Accuracy"] = avgAcc.Get();
	report["SORS/Pred Window Return"] = avgPredReturn.Get();
}

torch::Tensor ComputeEntropy(torch::Tensor probs, torch::Tensor actionMasks, bool maskEntropy) {
	// Compute log probs and entropy
	auto entropy = -(probs.log() * probs).sum(-1);

	if (maskEntropy) {
		// Account for action masking in entropy
		// We will effectively narrow the entropy to the scope of the valid actions
		// This way states with more masked actions don't just have inherently lower entropy
		// Clamp denominator to at least log(2) to avoid division by zero when only one action is valid.
		entropy /= actionMasks.to(torch::kFloat32).sum(-1).clamp_min(2.0f).log();
	} else {
		entropy /= logf(actionMasks.size(-1));
	}

	return entropy.mean();
}

float GGL::PPOLearner::GetEntropyScale() const {
	return config.adaptiveEntropy ? curEntropyScale : config.entropyScale;
}

// Z-score x within each mode group (heatseeker/soccar team sizes have very different
// dynamics, so pooled normalization lets one mode dominate the tails of another).
// Uses biased std so single-sample groups normalize to 0 instead of NaN.
// No host syncs: empty groups produce empty masked_selects and scatter nothing.
static torch::Tensor NormalizePerMode(const torch::Tensor& x, const torch::Tensor& modeIds, int numModes) {
	if (!modeIds.defined() || numModes <= 1)
		return (x - x.mean()) / (x.std(false) + 1e-8f);

	torch::Tensor result = x.clone();
	for (int m = 0; m < numModes; m++) {
		auto mask = (modeIds == m);
		auto vals = x.masked_select(mask);
		auto norm = (vals - vals.mean()) / (vals.std(false) + 1e-8f);
		result = result.masked_scatter(mask, norm);
	}
	return result;
}

namespace {
	// Per-minibatch training metrics, accumulated as detached device scalars and synced
	// to the host once per batch (each .item<float>() in the hot loop is a full GPU sync).
	// Slots that don't apply to a minibatch stay NaN; AvgTracker::Add ignores NaN.
	enum MetricSlot : int {
		M_ENTROPY, M_DIVERGENCE, M_POLICY_LOSS, M_REL_ENTROPY_LOSS, M_CRITIC_LOSS,
		M_GUIDING_LOSS, M_RATIO, M_CLIP,
		M_INFONCE, M_INFONCE_GOAL, M_INFONCE_ANTI, M_INFONCE_CAR,
		M_INFONCE_RAW, M_INFONCE_REG1, M_INFONCE_REG2,
		M_GCRL_ADV, M_GCRL_GOAL_ADV, M_GCRL_ANTI_ADV, M_GCRL_CAR_ADV, M_GCRL_REW_ADV, M_GCRL_FINAL_ADV,
		M_GCRL_GOAL_Q, M_GCRL_ANTI_Q, M_GCRL_CAR_Q,
		M_GCRL_GOAL_QSTD, M_GCRL_ANTI_QSTD, M_GCRL_CAR_QSTD,
		METRIC_SLOT_COUNT
	};
}

void GGL::PPOLearner::Learn(ExperienceBuffer& experience, Report& report, bool isFirstIteration) {
	auto mseLoss = torch::nn::MSELoss();

	AvgTracker avg[METRIC_SLOT_COUNT] = {};

	// Device-side metric staging (see MetricSlot)
	auto nanScalar = torch::full({}, std::numeric_limits<float>::quiet_NaN(),
		torch::TensorOptions().dtype(torch::kFloat32).device(device));
	std::vector<torch::Tensor> mbMetricsList;
	std::vector<torch::Tensor> mbSlots;

	// Save parameters first
	auto policyBefore = models["policy"]->CopyParams();
	auto criticBefore = models["critic"]->CopyParams();
	torch::Tensor sharedHeadBefore;
	if (models["shared_head"]) sharedHeadBefore = models["shared_head"]->CopyParams();
	torch::Tensor gcrlGoalBefore, gcrlAntiBefore, gcrlCarBefore;
	if (models["goal_critic"]) gcrlGoalBefore = models["goal_critic"]->CopyParams();
	if (models["anti_critic"]) gcrlAntiBefore = models["anti_critic"]->CopyParams();
	if (models["car_critic"])  gcrlCarBefore  = models["car_critic"]->CopyParams();

	bool trainPolicy = config.policyLR != 0;
	bool trainCritic = config.criticLR != 0;

	// Cache raw pointers once — avoids repeated std::map lookups inside the hot epoch loop.
	Model* m_sharedHead = models["shared_head"];
	Model* m_policy     = models["policy"];
	Model* m_critic     = models["critic"];

	bool trainSharedHead = m_sharedHead && (trainPolicy || trainCritic);

	bool trainGCRL = config.useGCRL && models["goal_critic"] && models["anti_critic"] && models["car_critic"];
	QuasimetricCritic *gc_goal = nullptr, *gc_anti = nullptr, *gc_car = nullptr;
	if (trainGCRL) {
		gc_goal = dynamic_cast<QuasimetricCritic*>(models["goal_critic"]);
		gc_anti = dynamic_cast<QuasimetricCritic*>(models["anti_critic"]);
		gc_car  = dynamic_cast<QuasimetricCritic*>(models["car_critic"]);
	}

	for (int epoch = 0; epoch < config.epochs; epoch++) {

		// Get randomly-ordered timesteps for PPO
		auto batches = experience.GetAllBatchesShuffled(config.batchSize, config.overbatching);

		for (auto& batch : batches) {
			auto batchActs = batch.actions;
			auto batchOldProbs = batch.logProbs;
			auto batchObs = batch.states;
			auto batchActionMasks = batch.actionMasks;
			auto batchTargetValues = batch.targetValues;
			auto batchAdvantages = batch.advantages;
			auto batchModeIds = batch.modeIds;
			auto batchActionComps = batch.actionComps;
			auto batchFutureGoals = batch.futureGoals;
			auto batchCarFutureGoals = batch.carFutureGoals;

			auto fnRunMinibatch = [&](int start, int stop) {

				// When stepping per minibatch there is no cross-minibatch accumulation to rescale for
				float batchSizeRatio = config.stepPerMiniBatch ? 1.0f : (stop - start) / (float)config.batchSize;

				mbSlots.assign(METRIC_SLOT_COUNT, nanScalar);

				// Send everything to the device and enforce correct shapes
				auto acts = batchActs.slice(0, start, stop).to(device, RG_H2D_NONBLOCKING(device), true);
				auto obs = batchObs.slice(0, start, stop).to(device, RG_H2D_NONBLOCKING(device), true);
				auto actionMasks = batchActionMasks.slice(0, start, stop).to(device, RG_H2D_NONBLOCKING(device), true);

				auto advantages = batchAdvantages.slice(0, start, stop).to(device, RG_H2D_NONBLOCKING(device), true);
				auto oldProbs = batchOldProbs.slice(0, start, stop).to(device, RG_H2D_NONBLOCKING(device), true);
				auto targetValues = batchTargetValues.slice(0, start, stop).to(device, RG_H2D_NONBLOCKING(device), true);

				torch::Tensor modeIds;
				if (batchModeIds.defined())
					modeIds = batchModeIds.slice(0, start, stop).to(device, RG_H2D_NONBLOCKING(device), true);

				torch::Tensor actionComps, futureGoals, carFutureGoals;
				if (trainGCRL) {
					actionComps    = batchActionComps.slice(0, start, stop).to(device, RG_H2D_NONBLOCKING(device), true);
					futureGoals    = batchFutureGoals.slice(0, start, stop).to(device, RG_H2D_NONBLOCKING(device), true);
					carFutureGoals = batchCarFutureGoals.slice(0, start, stop).to(device, RG_H2D_NONBLOCKING(device), true);
				}

				// Shared encoder forward once; policy, value critic and GCRL critics read it
				torch::Tensor features = obs;
				if (m_sharedHead)
					features = m_sharedHead->Forward(obs, false);

				// ── GCRL "game sense" advantage, blended onto the reward-driven advantage ──
				// Both channels are unit-normalized so gcrlAdvScale is a meaningful weight.
				if (trainGCRL) {
					RG_NO_GRAD;

					// The anti critic scores progress toward the own-goal target (conceding danger),
					// matching how the reward gate queries it. Querying it with futureGoals like the
					// goal critic would make it a near-identical twin whose z-scored difference is
					// mostly inter-seed noise.
					torch::Tensor antiGoals;
					if (curGCRLAntiTargetRow.defined())
						antiGoals = curGCRLAntiTargetRow.to(device).expand({ features.size(0), curGCRLAntiTargetRow.size(1) });
					else
						antiGoals = futureGoals;

					auto q_goal = gc_goal->score_q(features, actionComps, futureGoals);
					auto q_anti = gc_anti->score_q(features, actionComps, antiGoals);
					auto q_car  = gc_car->score_q(features, actionComps, carFutureGoals);
					mbSlots[M_GCRL_GOAL_Q] = q_goal.mean();
					mbSlots[M_GCRL_ANTI_Q] = q_anti.mean();
					mbSlots[M_GCRL_CAR_Q]  = q_car.mean();
					mbSlots[M_GCRL_GOAL_QSTD] = q_goal.std();
					mbSlots[M_GCRL_ANTI_QSTD] = q_anti.std();
					mbSlots[M_GCRL_CAR_QSTD]  = q_car.std();

					// Counterfactual action baseline: subtracting the mean Q of shuffled actions
					// turns "this state is near a goal" into "this action beats a typical action
					// from this state", removing state-value leakage from the advantage.
					if (config.gcrlBaselineSamples > 0) {
						int64_t n = features.size(0);
						auto idxOpts = torch::TensorOptions().dtype(torch::kLong).device(device);
						auto bGoal = torch::zeros_like(q_goal);
						auto bAnti = torch::zeros_like(q_anti);
						auto bCar  = torch::zeros_like(q_car);
						for (int k = 0; k < config.gcrlBaselineSamples; k++) {
							auto shuffledActs = actionComps.index_select(0, torch::randperm(n, idxOpts));
							bGoal += gc_goal->score_q(features, shuffledActs, futureGoals);
							bAnti += gc_anti->score_q(features, shuffledActs, antiGoals);
							bCar  += gc_car->score_q(features, shuffledActs, carFutureGoals);
						}
						float invK = 1.0f / config.gcrlBaselineSamples;
						q_goal = q_goal - bGoal * invK;
						q_anti = q_anti - bAnti * invK;
						q_car  = q_car  - bCar  * invK;
					}

					auto adv_goal = NormalizePerMode(q_goal, modeIds, numModes);
					auto adv_anti = NormalizePerMode(q_anti, modeIds, numModes);
					auto adv_car  = NormalizePerMode(q_car,  modeIds, numModes);
					mbSlots[M_GCRL_GOAL_ADV] = adv_goal.abs().mean();
					mbSlots[M_GCRL_ANTI_ADV] = adv_anti.abs().mean();
					mbSlots[M_GCRL_CAR_ADV]  = adv_car.abs().mean();

					// goal pursuit, "anti" own-goal danger, car positioning
					auto adv_gcrl = adv_goal - config.gcrlAntiScale * adv_anti + config.gcrlCarScale * adv_car;
					mbSlots[M_GCRL_ADV] = adv_gcrl.abs().mean();

					auto adv_rew = NormalizePerMode(advantages, modeIds, numModes);
					mbSlots[M_GCRL_REW_ADV] = adv_rew.abs().mean();
					advantages = adv_rew + curGCRLAdvScale * adv_gcrl;
					mbSlots[M_GCRL_FINAL_ADV] = advantages.abs().mean();
				}

				torch::Tensor probs, logProbs, entropy, ratio, clipped, policyLoss, ppoLoss;
				if (trainPolicy) {

					// Get policy log probs and entropy (from the shared features)
					{
						auto maskBool = actionMasks.to(torch::kBool);
						auto logits = m_policy->Forward(features, false) / config.policyTemperature;
						probs = torch::softmax(logits + (-1e10f) * maskBool.logical_not(), -1)
							.view({ -1, m_policy->config.numOutputs }).clamp(1e-11f, 1);
						logProbs = probs.log().gather(-1, acts.unsqueeze(-1));
						entropy = ComputeEntropy(probs, actionMasks, config.maskEntropy);
						mbSlots[M_ENTROPY] = entropy.detach();
					}

					logProbs = logProbs.view_as(oldProbs);
					const float entropyScale = GetEntropyScale();

					// Compute PPO loss
					ratio = exp(logProbs - oldProbs);
					mbSlots[M_RATIO] = ratio.mean().detach();
					clipped = clamp(
						ratio, 1 - config.clipRange, 1 + config.clipRange
					);

					// Compute policy loss
					policyLoss = -min(
						ratio * advantages, clipped * advantages
					).mean();
					mbSlots[M_POLICY_LOSS] = policyLoss.detach();
					mbSlots[M_REL_ENTROPY_LOSS] = (entropy.detach() * entropyScale) / policyLoss.detach();

					ppoLoss = (policyLoss - entropy * entropyScale) * batchSizeRatio;

					// NOTE: The adaptive entropy controller is updated once per batch (after the
					// host metric sync), not here per-minibatch.

					if (config.useGuidingPolicy) {
						torch::Tensor guidingProbs;
						{
							RG_NO_GRAD;
							guidingProbs = InferPolicyProbsFromModels(guidingPolicyModels, obs, actionMasks, config.policyTemperature, config.useHalfPrecision);
						}

						auto guidingLoss = (guidingProbs - probs).abs().mean();
						mbSlots[M_GUIDING_LOSS] = guidingLoss.detach();
						guidingLoss = guidingLoss * config.guidingStrength;
						ppoLoss = ppoLoss + guidingLoss;
					}
				}

				torch::Tensor criticLoss;
				if (trainCritic) {
					auto vals = m_critic->Forward(features, false).flatten();

					// Compute value loss
					vals = vals.view_as(targetValues);
					criticLoss = mseLoss(vals, targetValues) * batchSizeRatio;
					mbSlots[M_CRITIC_LOSS] = criticLoss.detach();
				}

				if (trainPolicy) {
					// Compute KL divergence & clip fraction using SB3 method for reporting;
					{
						RG_NO_GRAD;

						auto logRatio = logProbs - oldProbs;
						auto klTensor = (exp(logRatio) - 1) - logRatio;
						mbSlots[M_DIVERGENCE] = klTensor.mean();

						mbSlots[M_CLIP] = mean((abs(ratio - 1) > config.clipRange).to(kFloat));
					}
				}

				// ── InfoNCE contrastive training of the quasimetric critics ──
				torch::Tensor infoNCELoss;
				if (trainGCRL) {
					int64_t B = features.size(0);
					if (B > 0) {
						int64_t info_n = RS_MIN(B, (int64_t)config.gcrlInfoSubSample);
						auto perm = torch::randperm(B, torch::TensorOptions().device(device).dtype(torch::kLong));
						auto idxs = perm.slice(0, 0, info_n);

						auto sub_obs = features.index_select(0, idxs);
						auto sub_actions = actionComps.index_select(0, idxs);
						auto sub_goals = futureGoals.index_select(0, idxs);
						auto sub_car_goals = carFutureGoals.index_select(0, idxs);

						torch::Tensor rawG, reg1G, reg2G, rawA, reg1A, reg2A, rawC, reg1C, reg2C;
						auto il_goal = gc_goal->infonce_loss(sub_obs, sub_actions, sub_goals, -1, {}, &rawG, &reg1G, &reg2G);
						auto il_anti = gc_anti->infonce_loss(sub_obs, sub_actions, sub_goals, -1, {}, &rawA, &reg1A, &reg2A);
						auto il_car  = gc_car->infonce_loss(sub_obs, sub_actions, sub_car_goals, -1, {}, &rawC, &reg1C, &reg2C);
						infoNCELoss = (il_goal + il_anti + il_car) * batchSizeRatio;
						mbSlots[M_INFONCE] = infoNCELoss.detach();
						mbSlots[M_INFONCE_GOAL] = il_goal.detach();
						mbSlots[M_INFONCE_ANTI] = il_anti.detach();
						mbSlots[M_INFONCE_CAR] = il_car.detach();
						mbSlots[M_INFONCE_RAW]  = rawG + rawA + rawC;
						mbSlots[M_INFONCE_REG1] = reg1G + reg1A + reg1C;
						mbSlots[M_INFONCE_REG2] = reg2G + reg2A + reg2C;
					}
				}

				// ── Combine & backprop (single graph through the shared encoder) ──
				torch::Tensor combinedLoss;
				if (trainPolicy)
					combinedLoss = ppoLoss;
				if (trainCritic)
					combinedLoss = combinedLoss.defined() ? (combinedLoss + criticLoss) : criticLoss;
				if (infoNCELoss.defined()) {
					auto scaled = config.gcrlInfoNCECoef * infoNCELoss;
					combinedLoss = combinedLoss.defined() ? (combinedLoss + scaled) : scaled;
				}
				if (combinedLoss.defined())
					combinedLoss.backward();

				mbMetricsList.push_back(torch::stack(mbSlots));
			};

			auto fnClipAndStep = [&] {
				if (trainPolicy)
					nn::utils::clip_grad_norm_(m_policy->parameters(), 0.5f);
				if (trainCritic)
					nn::utils::clip_grad_norm_(m_critic->parameters(), 0.5f);

				if (trainSharedHead)
					nn::utils::clip_grad_norm_(m_sharedHead->parameters(), 0.5f);

				if (trainGCRL) {
					nn::utils::clip_grad_norm_(gc_goal->parameters(), 0.5f);
					nn::utils::clip_grad_norm_(gc_anti->parameters(), 0.5f);
					nn::utils::clip_grad_norm_(gc_car->parameters(), 0.5f);
				}

				models.StepOptims();
			};

			// Use actual tensor size so overbatched tail samples aren't silently discarded.
			int actualBatchSize = (int)batchObs.size(0);
			if (device.is_cpu()) {
				fnRunMinibatch(0, actualBatchSize);
				if (config.stepPerMiniBatch)
					fnClipAndStep();
			} else {
				for (int mbs = 0; mbs < actualBatchSize; mbs += config.miniBatchSize) {
					int start = mbs;
					int stop = RS_MIN(start + config.miniBatchSize, actualBatchSize);
					fnRunMinibatch(start, stop);
					if (config.stepPerMiniBatch)
						fnClipAndStep();
				}
			}

			if (!config.stepPerMiniBatch)
				fnClipAndStep();

			// Sync this batch's staged metrics to the host in one transfer
			// (NaN slots are "metric not applicable" and are skipped by AvgTracker)
			if (!mbMetricsList.empty()) {
				auto batchMetrics = torch::stack(mbMetricsList).nanmean(0).cpu();
				mbMetricsList.clear();
				const float* m = batchMetrics.data_ptr<float>();
				for (int i = 0; i < METRIC_SLOT_COUNT; i++)
					avg[i] += m[i];

				// Adaptive entropy controller, once per batch from the batch-mean entropy.
				// (Was per-minibatch; with the old accumulate-then-step config there was one
				// minibatch per batch anyway, so the effective cadence is unchanged.)
				if (config.adaptiveEntropy && trainPolicy && !std::isnan(m[M_ENTROPY])) {
					curEntropyScale += config.adaptiveEntropyLR * (config.targetEntropy - m[M_ENTROPY]);
					curEntropyScale = std::clamp(curEntropyScale, config.minEntropyScale, config.maxEntropyScale);
				}
			}
		}
	}

	// Compute magnitude of updates made to the policy and value estimator
	auto policyAfter = models["policy"]->CopyParams();
	auto criticAfter = models["critic"]->CopyParams();

	float policyUpdateMagnitude = (policyBefore - policyAfter).norm().item<float>();
	float criticUpdateMagnitude = (criticBefore - criticAfter).norm().item<float>();
	float sharedHeadUpdateMagnitude = 0;
	if (sharedHeadBefore.defined())
		sharedHeadUpdateMagnitude = (sharedHeadBefore - models["shared_head"]->CopyParams()).norm().item<float>();
	float gcrlGoalUpdateMagnitude = 0;
	float gcrlAntiUpdateMagnitude = 0;
	float gcrlCarUpdateMagnitude = 0;
	if (trainGCRL) {
		gcrlGoalUpdateMagnitude = (gcrlGoalBefore - models["goal_critic"]->CopyParams()).norm().item<float>();
		gcrlAntiUpdateMagnitude = (gcrlAntiBefore - models["anti_critic"]->CopyParams()).norm().item<float>();
		gcrlCarUpdateMagnitude = (gcrlCarBefore - models["car_critic"]->CopyParams()).norm().item<float>();
	}

	// Assemble and return report
	report["Policy Entropy"] = avg[M_ENTROPY].Get();
	report["Entropy Scale"] = GetEntropyScale();
	if (config.adaptiveEntropy)
		report["Target Entropy"] = config.targetEntropy;
	report["Mean KL Divergence"] = avg[M_DIVERGENCE].Get();
	if (!isFirstIteration) {
		// These metrics give bad data on the first iteration, which will mess up graph scaling
		// So we'll just skip them for the first iteration
		report["Policy Loss"] = avg[M_POLICY_LOSS].Get();
		report["Policy Relative Entropy Loss"] = avg[M_REL_ENTROPY_LOSS].Get();
		report["Critic Loss"] = avg[M_CRITIC_LOSS].Get();

		if (trainGCRL) {
			report["InfoNCE Loss"] = avg[M_INFONCE].Get();
			report["GCRL/InfoNCE Goal Loss"] = avg[M_INFONCE_GOAL].Get();
			report["GCRL/InfoNCE Anti Loss"] = avg[M_INFONCE_ANTI].Get();
			report["GCRL/InfoNCE Car Loss"] = avg[M_INFONCE_CAR].Get();
			report["GCRL/InfoNCE Raw"] = avg[M_INFONCE_RAW].Get();
			report["GCRL/InfoNCE LogSumExp Reg"] = avg[M_INFONCE_REG1].Get();
			report["GCRL/InfoNCE Var Reg"] = avg[M_INFONCE_REG2].Get();
			report["GCRL/Avg Advantage"] = avg[M_GCRL_ADV].Get();
			report["GCRL/Goal Advantage"] = avg[M_GCRL_GOAL_ADV].Get();
			report["GCRL/Anti Advantage"] = avg[M_GCRL_ANTI_ADV].Get();
			report["GCRL/Car Advantage"] = avg[M_GCRL_CAR_ADV].Get();
			report["GCRL/Reward Advantage"] = avg[M_GCRL_REW_ADV].Get();
			report["GCRL/Final Advantage"] = avg[M_GCRL_FINAL_ADV].Get();
			report["GCRL/Goal Q Mean"] = avg[M_GCRL_GOAL_Q].Get();
			report["GCRL/Anti Q Mean"] = avg[M_GCRL_ANTI_Q].Get();
			report["GCRL/Car Q Mean"] = avg[M_GCRL_CAR_Q].Get();
			report["GCRL/Goal Q STD"] = avg[M_GCRL_GOAL_QSTD].Get();
			report["GCRL/Anti Q STD"] = avg[M_GCRL_ANTI_QSTD].Get();
			report["GCRL/Car Q STD"] = avg[M_GCRL_CAR_QSTD].Get();
			report["GCRL/Goal Update Magnitude"] = gcrlGoalUpdateMagnitude;
			report["GCRL/Anti Update Magnitude"] = gcrlAntiUpdateMagnitude;
			report["GCRL/Car Update Magnitude"] = gcrlCarUpdateMagnitude;
		}

		if (config.useGuidingPolicy)
			report["Guiding Loss"] = avg[M_GUIDING_LOSS].Get();

		report["SB3 Clip Fraction"] = avg[M_CLIP].Get();
		report["Policy Update Magnitude"] = policyUpdateMagnitude;
		report["Critic Update Magnitude"] = criticUpdateMagnitude;
		if (sharedHeadBefore.defined())
			report["Shared Head Update Magnitude"] = sharedHeadUpdateMagnitude;
	}
}

void GGL::PPOLearner::TransferLearn(
	ModelSet& oldModels,
	torch::Tensor newObs, torch::Tensor oldObs,
	torch::Tensor newActionMasks, torch::Tensor oldActionMasks,
	torch::Tensor actionMaps,
	Report& report,
	const TransferLearnConfig& tlConfig
) {

	torch::Tensor oldProbs;
	{ // No grad for old model inference
		RG_NO_GRAD;
		oldProbs = InferPolicyProbsFromModels(oldModels, oldObs, oldActionMasks, config.policyTemperature, config.useHalfPrecision);
		report["Old Policy Entropy"] = ComputeEntropy(oldProbs, oldActionMasks, config.maskEntropy).detach().cpu().item<float>();

		if (actionMaps.defined())
			oldProbs = oldProbs.gather(1, actionMaps);
	}

	for (auto& model : GetPolicyModels())
		model->SetOptimLR(tlConfig.lr);

	auto policyBefore = models["policy"]->CopyParams();
	
	for (int i = 0; i < tlConfig.epochs; i++) {
		torch::Tensor newProbs = InferPolicyProbsFromModels(models, newObs, newActionMasks, config.policyTemperature, false);

		// Non-summative KL div	loss
		torch::Tensor transferLearnLoss;
		if (tlConfig.useKLDiv) {
			transferLearnLoss = (oldProbs * torch::log(oldProbs / newProbs)).abs();
		} else {
			transferLearnLoss = (oldProbs - newProbs).abs();
		}
		transferLearnLoss = transferLearnLoss.pow(tlConfig.lossExponent);
		transferLearnLoss = transferLearnLoss.mean();
		transferLearnLoss *= tlConfig.lossScale;

		if (i == 0) {
			RG_NO_GRAD;
			torch::Tensor matchingActionsMask = (newProbs.detach().argmax(-1) == oldProbs.detach().argmax(-1));
			report["Transfer Learn Accuracy"] = matchingActionsMask.to(torch::kFloat).mean().cpu().item<float>();
			report["Transfer Learn Loss"] = transferLearnLoss.detach().cpu().item<float>();

			report["Policy Entropy"] = ComputeEntropy(newProbs, newActionMasks, config.maskEntropy).detach().cpu().item<float>();
		}

		transferLearnLoss.backward();

		models.StepOptims();
	}

	auto policyAfter = models["policy"]->CopyParams();
	report["Policy Update Magnitude"] = (policyBefore - policyAfter).norm().item<float>();
}

void GGL::PPOLearner::SaveTo(std::filesystem::path folderPath) {
	models.Save(folderPath);
	SavePPOState(folderPath);
}

void GGL::PPOLearner::LoadFrom(std::filesystem::path folderPath)  {
	if (!std::filesystem::is_directory(folderPath))
		RG_ERR_CLOSE("PPOLearner:LoadFrom(): Path " << folderPath << " is not a valid directory");

	models.Load(folderPath, true, true);
	LoadPPOState(folderPath);

	SetLearningRates(config.policyLR, config.criticLR);
}

void GGL::PPOLearner::SavePPOState(std::filesystem::path folderPath) {
	using namespace nlohmann;

	json j = {};
	j["cur_entropy_scale"] = curEntropyScale;
	j["sors_train_calls"] = sorsTrainCalls;

	auto path = folderPath / PPO_STATE_FILE_NAME;
	std::ofstream fOut(path);
	if (!fOut.good())
		RG_ERR_CLOSE("PPOLearner::SavePPOState(): Can't open file at " << path);

	fOut << j.dump(4);
	SaveSORSState(folderPath);
}

void GGL::PPOLearner::LoadPPOState(std::filesystem::path folderPath) {
	using namespace nlohmann;

	auto path = folderPath / PPO_STATE_FILE_NAME;
	if (std::filesystem::exists(path)) {
		std::ifstream fIn(path);
		if (!fIn.good())
			RG_ERR_CLOSE("PPOLearner::LoadPPOState(): Can't open file at " << path);

		json j = json::parse(fIn);
		if (j.contains("cur_entropy_scale"))
			curEntropyScale = j["cur_entropy_scale"];
		if (j.contains("sors_train_calls"))
			sorsTrainCalls = j["sors_train_calls"];
	} else {
		curEntropyScale = std::clamp(config.entropyScale, config.minEntropyScale, config.maxEntropyScale);
		sorsTrainCalls = 0;
	}

	LoadSORSState(folderPath);
}

void GGL::PPOLearner::SaveSORSState(std::filesystem::path folderPath) {
	using namespace nlohmann;

	if (!config.useSORS)
		return;

	json j = {};
	j["window_count"] = sorsReplay.size();
	j["train_calls"] = sorsTrainCalls;

	auto statePath = folderPath / SORS_STATE_FILE_NAME;
	std::ofstream fOut(statePath);
	if (!fOut.good())
		RG_ERR_CLOSE("PPOLearner::SaveSORSState(): Can't open file at " << statePath);

	if (sorsReplay.empty()) {
		fOut << j.dump(4);
		return;
	}

	size_t totalSteps = 0;
	std::vector<int64_t> lengths = {};
	FList labels = {};
	std::vector<torch::Tensor> statesList = {}, actionsList = {};
	statesList.reserve(sorsReplay.size());
	actionsList.reserve(sorsReplay.size());
	for (const auto& window : sorsReplay) {
		totalSteps += window.length;
		lengths.push_back(window.length);
		labels.push_back(window.label);
		statesList.push_back(window.states.flatten());
		actionsList.push_back(window.actionComps.flatten());
	}

	auto tStates = torch::cat(statesList);
	auto tActions = torch::cat(actionsList);

	torch::save(tStates, (folderPath / SORS_REPLAY_STATES_FILE_NAME).string());
	torch::save(tActions, (folderPath / SORS_REPLAY_ACTIONS_FILE_NAME).string());
	torch::save(torch::tensor(lengths, torch::TensorOptions().dtype(torch::kInt64)), (folderPath / SORS_REPLAY_LENGTHS_FILE_NAME).string());
	torch::save(torch::tensor(labels), (folderPath / SORS_REPLAY_LABELS_FILE_NAME).string());

	j["total_steps"] = totalSteps;
	j["total_state_values"] = (size_t)tStates.numel();
	j["total_action_values"] = (size_t)tActions.numel();
	fOut << j.dump(4);
}

void GGL::PPOLearner::LoadSORSState(std::filesystem::path folderPath) {
	sorsReplay.clear();
	if (!config.useSORS)
		return;

	auto* sors = dynamic_cast<SORSRewardModel*>(models["sors_reward"]);
	RG_ASSERT(sors);

	auto statePath = folderPath / SORS_STATE_FILE_NAME;
	if (!std::filesystem::exists(statePath))
		return;

	std::ifstream fIn(statePath);
	if (!fIn.good())
		RG_ERR_CLOSE("PPOLearner::LoadSORSState(): Can't open file at " << statePath);

	nlohmann::json j = nlohmann::json::parse(fIn);
	if (j.contains("train_calls"))
		sorsTrainCalls = j["train_calls"];

	size_t windowCount = j.value("window_count", (size_t)0);
	if (!windowCount)
		return;

	auto statesPath = folderPath / SORS_REPLAY_STATES_FILE_NAME;
	auto actionsPath = folderPath / SORS_REPLAY_ACTIONS_FILE_NAME;
	auto lengthsPath = folderPath / SORS_REPLAY_LENGTHS_FILE_NAME;
	auto labelsPath = folderPath / SORS_REPLAY_LABELS_FILE_NAME;
	if (!std::filesystem::exists(statesPath) || !std::filesystem::exists(actionsPath) ||
		!std::filesystem::exists(lengthsPath) || !std::filesystem::exists(labelsPath)) {
		RG_ERR_CLOSE("PPOLearner::LoadSORSState(): SORS replay metadata exists but replay tensor files are missing in " << folderPath);
	}

	torch::Tensor tStates, tActions, tLengths, tLabels;
	torch::load(tStates, statesPath.string());
	torch::load(tActions, actionsPath.string());
	torch::load(tLengths, lengthsPath.string());
	torch::load(tLabels, labelsPath.string());

	tStates = tStates.flatten().contiguous().to(torch::kCPU, true, true);
	tActions = tActions.flatten().contiguous().to(torch::kCPU, true, true);
	tLengths = tLengths.flatten().contiguous().to(torch::kCPU, true, true);
	tLabels = tLabels.flatten().contiguous().to(torch::kCPU, true, true);

	if ((size_t)tLengths.numel() != windowCount || (size_t)tLabels.numel() != windowCount)
		RG_ERR_CLOSE("PPOLearner::LoadSORSState(): Corrupt SORS replay metadata in " << folderPath);

	const int64_t* lengthsData = tLengths.data_ptr<int64_t>();
	const float* labelsData = tLabels.data_ptr<float>();

	size_t stateOffset = 0;
	size_t actionOffset = 0;
	sorsReplay.clear();
	sorsReplay.resize(windowCount);
	for (size_t i = 0; i < windowCount; i++) {
		auto& window = sorsReplay[i];
		window.length = (int)lengthsData[i];
		window.label = labelsData[i];
		if (window.length < 0)
			RG_ERR_CLOSE("PPOLearner::LoadSORSState(): Negative SORS window length in " << folderPath);

		size_t stateCount = (size_t)window.length * sors->obs_dim;
		size_t actionCount = (size_t)window.length * sors->action_dim;
		if (stateOffset + stateCount > (size_t)tStates.numel() || actionOffset + actionCount > (size_t)tActions.numel())
			RG_ERR_CLOSE("PPOLearner::LoadSORSState(): Corrupt SORS replay tensor sizes in " << folderPath);

		// Clone so each window owns its memory instead of keeping the big flat tensors alive
		window.states = tStates.slice(0, stateOffset, stateOffset + stateCount)
			.reshape({ (int64_t)window.length, (int64_t)sors->obs_dim }).clone();
		window.actionComps = tActions.slice(0, actionOffset, actionOffset + actionCount)
			.reshape({ (int64_t)window.length, (int64_t)sors->action_dim }).clone();
		stateOffset += stateCount;
		actionOffset += actionCount;
	}

	if (stateOffset != (size_t)tStates.numel() || actionOffset != (size_t)tActions.numel())
		RG_ERR_CLOSE("PPOLearner::LoadSORSState(): Corrupt SORS replay tensor sizes in " << folderPath);
}

void GGL::PPOLearner::SetLearningRates(float policyLR, float criticLR) {
	config.policyLR = policyLR;
	config.criticLR = criticLR;

	models["policy"]->SetOptimLR(policyLR);
	models["critic"]->SetOptimLR(criticLR);

	if (models["shared_head"])
		models["shared_head"]->SetOptimLR(RS_MIN(policyLR, criticLR));

	// GCRL critics are representation learners; default to the policy LR unless gcrlLR is set
	float gcrlLR = config.gcrlLR > 0 ? config.gcrlLR : policyLR;
	if (models["goal_critic"]) models["goal_critic"]->SetOptimLR(gcrlLR);
	if (models["anti_critic"]) models["anti_critic"]->SetOptimLR(gcrlLR);
	if (models["car_critic"])  models["car_critic"]->SetOptimLR(gcrlLR);

	float sorsLR = config.sorsLR > 0 ? config.sorsLR : policyLR;
	if (models["sors_reward"]) models["sors_reward"]->SetOptimLR(sorsLR);

	RG_LOG("PPOLearner: " << RS_STR(std::scientific << "Set learning rate to [" << policyLR << ", " << criticLR << "]"));
}

GGL::ModelSet GGL::PPOLearner::GetPolicyModels() {
	ModelSet result = {};
	for (Model* model : models) {
		std::string name = model->modelName;
		// Exclude value and GCRL critics: only policy + shared head define a playable version
		if (name == "critic" || name == "goal_critic" || name == "anti_critic" || name == "car_critic" || name == "sors_reward")
			continue;

		result.Add(model);
	}
	return result;
}
