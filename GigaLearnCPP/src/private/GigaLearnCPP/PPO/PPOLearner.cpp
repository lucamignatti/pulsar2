#include "PPOLearner.h"

#include <torch/nn/utils/convert_parameters.h>
#include <torch/nn/utils/clip_grad.h>
#include <torch/csrc/api/include/torch/serialize.h>
#include <public/GigaLearnCPP/Util/AvgTracker.h>
#include <cmath>
#include <cfloat>
#include <unordered_map>

using namespace torch;

GGL::PPOLearner::PPOLearner(int obsSize, int numActions, PPOLearnerConfig _config, Device _device) :
	config(_config), device(_device), obsSize(obsSize), numActions(numActions) {

	if (config.miniBatchSize == 0)
		config.miniBatchSize = config.batchSize;

	if (config.batchSize <= 0)
		RG_ERR_CLOSE("PPOLearner: config.batchSize must be positive");
	if (config.miniBatchSize <= 0)
		RG_ERR_CLOSE("PPOLearner: config.miniBatchSize must be positive");
	if (config.policyTemperature <= 0 || !std::isfinite(config.policyTemperature))
		RG_ERR_CLOSE("PPOLearner: config.policyTemperature must be finite and positive");

	if (config.batchSize % config.miniBatchSize != 0)
		RG_ERR_CLOSE("PPOLearner: config.batchSize (" << config.batchSize << ") must be a multiple of config.miniBatchSize (" << config.miniBatchSize << ")");

	if (config.contrastiveGoal.enabled) {
		auto& cfg = config.contrastiveGoal;
		if (cfg.tau <= 0 || !std::isfinite(cfg.tau))
			RG_ERR_CLOSE("PPOLearner: contrastiveGoal.tau must be finite and positive");
		if (cfg.sigmaFloor <= 0 || !std::isfinite(cfg.sigmaFloor))
			RG_ERR_CLOSE("PPOLearner: contrastiveGoal.sigmaFloor must be finite and positive");
		if (cfg.posScaleX == 0 || cfg.posScaleY == 0 || cfg.posScaleZ == 0 || cfg.velScale == 0)
			RG_ERR_CLOSE("PPOLearner: contrastive goal normalization scales must be non-zero");
	}

	MakeModels(true, obsSize, numActions, config.sharedHead, config.policy, config.critic, device, models);

	SetLearningRates(config.policyLR, config.criticLR);

	if (config.contrastiveGoal.enabled) {
		contrastiveGoalLearner = new ContrastiveGoalLearner(obsSize, numActions, config.contrastiveGoal, device);
		RG_LOG("GCRL scoring auxiliary enabled with separate critic encoders");
		if (config.contrastiveGoal.useCarCritic) {
			// Egocentric car-local ball goal; trains on all rows (ignores the ball-moved mask).
			// When useSharedBase, reuse the goal critic's phi encoder (one base, per-head psi).
			Model* sharedPhi = config.contrastiveGoal.useSharedBase ? &contrastiveGoalLearner->stateActionEncoder : nullptr;
			carContrastiveLearner = new ContrastiveGoalLearner(obsSize, numActions, config.contrastiveGoal, device, "gcrl_car", true, false, false, sharedPhi);
			RG_LOG("GCRL car critic enabled (egocentric controllability)" << (sharedPhi ? " [shared phi base]" : ""));
		}
		if (config.contrastiveGoal.useBoostCritic) {
			// Boost goal (own boost level); trains on all rows. Shares phi when useSharedBase.
			Model* sharedPhi = config.contrastiveGoal.useSharedBase ? &contrastiveGoalLearner->stateActionEncoder : nullptr;
			boostContrastiveLearner = new ContrastiveGoalLearner(obsSize, numActions, config.contrastiveGoal, device, "gcrl_boost", false, true, false, sharedPhi);
			RG_LOG("GCRL boost critic enabled (reachability toward full boost)" << (sharedPhi ? " [shared phi base]" : ""));
		}
	}

	if (config.rsNorm.enabled) {
		obsNorm = new RSNorm(obsSize, config.rsNorm.eps, config.rsNorm.initVar, config.rsNorm.initCount, config.rsNorm.clipRange);
		RG_LOG("RSNorm (running observation normalization) enabled over " << obsSize << " dims");
	}

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

GGL::PPOLearner::~PPOLearner() {
	models.Free();
	guidingPolicyModels.Free();
	// Delete the head critics first: under useSharedBase they reference the goal critic's phi encoder.
	delete carContrastiveLearner;
	delete boostContrastiveLearner;
	delete contrastiveGoalLearner;
	delete obsNorm;
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
	float temperature, bool halfPrec, const RSNorm* obsNorm) {

	actionMasks = actionMasks.to(torch::kBool);

	constexpr float ACTION_MIN_PROB = 1e-11f;
	constexpr float ACTION_DISABLED_LOGIT = -1e10f;

	// RSNorm: standardize observations as the FIRST op, before the trunk (stop-grad).
	if (obsNorm)
		obs = obsNorm->Normalize(obs);

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
	torch::Tensor* outActions, torch::Tensor* outLogProbs,
	torch::Tensor* outActionProbs, const RSNorm* obsNorm) {

	auto probs = InferPolicyProbsFromModels(models, obs, actionMasks, temperature, halfPrec, obsNorm);
	if (outActionProbs)
		*outActionProbs = probs;

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

void GGL::PPOLearner::InferActions(torch::Tensor obs, torch::Tensor actionMasks, torch::Tensor* outActions, torch::Tensor* outLogProbs, ModelSet* models, torch::Tensor* outActionProbs) {
	InferActionsFromModels(models ? *models : this->models, obs, actionMasks, config.deterministic, config.policyTemperature, config.useHalfPrecision, outActions, outLogProbs, outActionProbs, obsNorm);
}

torch::Tensor GGL::PPOLearner::InferCritic(torch::Tensor obs) {

	// RSNorm: same shared normalizer as the policy, first op (stop-grad).
	if (obsNorm)
		obs = obsNorm->Normalize(obs);

	if (models["shared_head"])
		obs = models["shared_head"]->Forward(obs, config.useHalfPrecision);

	return models["critic"]->Forward(obs, config.useHalfPrecision).flatten();
}

torch::Tensor ComputeEntropy(torch::Tensor probs, torch::Tensor actionMasks, bool maskEntropy) {
	// Compute log probs and entropy
	auto entropy = -(probs.log() * probs).sum(-1);

	if (maskEntropy) {
		// Account for action masking in entropy
		// We will effectively narrow the entropy to the scope of the valid actions
		// This way states with more masked actions don't just have inherently lower entropy
		auto validActionCounts = actionMasks.to(torch::kFloat32).sum(-1);
		entropy = torch::where(
			validActionCounts > 1,
			entropy / validActionCounts.log(),
			torch::zeros_like(entropy)
		);
	} else {
		float denom = logf(actionMasks.size(-1));
		entropy = denom > 0 ? entropy / denom : torch::zeros_like(entropy);
	}

	return entropy.mean();
}

static torch::Tensor NormalizeAdvantage(torch::Tensor advantage, float floor) {
	auto mean = advantage.mean();
	auto std = advantage.std(false);
	return (advantage - mean) / torch::clamp(std, floor);
}

static float TensorStd(torch::Tensor t) {
	return t.std(false).detach().cpu().item<float>();
}

static float TensorMean(torch::Tensor t) {
	return t.mean().detach().cpu().item<float>();
}

static torch::Tensor OneHotActions(torch::Tensor actions, int64_t numActions) {
	actions = actions.to(torch::kLong);
	torch::Tensor result = torch::zeros({ actions.size(0), numActions }, torch::TensorOptions().dtype(torch::kFloat32).device(actions.device()));
	return result.scatter_(1, actions.unsqueeze(1), 1.f);
}

static torch::Tensor ScoreGCRLChunks(GGL::ContrastiveGoalLearner* learner, torch::Tensor states, torch::Tensor actions, torch::Tensor goals, int64_t numActions, int64_t chunkSize) {
	std::vector<torch::Tensor> chunks;
	int64_t n = states.size(0);
	if (chunkSize <= 0)
		chunkSize = n;

	chunks.reserve((n + chunkSize - 1) / chunkSize);
	for (int64_t start = 0; start < n; start += chunkSize) {
		int64_t stop = RS_MIN(start + chunkSize, n);
		torch::Tensor actionRepresentations = OneHotActions(actions.slice(0, start, stop), numActions);
		chunks.push_back(learner->Score(
			states.slice(0, start, stop),
			actionRepresentations,
			goals.slice(0, start, stop)
		));
	}

	return torch::cat(chunks, 0);
}

static torch::Tensor SampleValidActions(torch::Tensor actionMasks) {
	torch::Tensor weights = actionMasks.to(torch::kFloat32);
	return torch::multinomial(weights, 1, true).flatten().to(torch::kLong);
}

static torch::Tensor ComputeGCRLAdvantageWithBaseline(
	GGL::ContrastiveGoalLearner* learner,
	torch::Tensor states,
	torch::Tensor actions,
	torch::Tensor actionMasks,
	torch::Tensor goals,
	int64_t numActions,
	int64_t chunkSize,
	int baselineSamples,
	torch::Tensor* outTakenScores,
	torch::Tensor* outBaselineScores,
	torch::Tensor* outBaselineSpread
) {
	torch::Tensor takenScores = ScoreGCRLChunks(learner, states, actions, goals, numActions, chunkSize);
	torch::Tensor baselineSum = torch::zeros_like(takenScores);
	torch::Tensor baselineSqSum = torch::zeros_like(takenScores);

	if (baselineSamples > 0) {
		for (int i = 0; i < baselineSamples; i++) {
			torch::Tensor altActions = SampleValidActions(actionMasks);
			torch::Tensor s = ScoreGCRLChunks(learner, states, altActions, goals, numActions, chunkSize);
			baselineSum += s;
			baselineSqSum += s * s;
		}
	}

	torch::Tensor baselineScores = (baselineSamples > 0) ? (baselineSum / baselineSamples) : baselineSum;

	if (outTakenScores)
		*outTakenScores = takenScores;
	if (outBaselineScores)
		*outBaselineScores = baselineScores;
	if (outBaselineSpread) {
		// Within-state spread of the random-action baseline scores: the noise floor
		// the taken-action edge must beat to count as real action-discrimination.
		if (baselineSamples > 1) {
			torch::Tensor var = (baselineSqSum / baselineSamples) - baselineScores * baselineScores;
			*outBaselineSpread = var.clamp_min(0).sqrt();
		} else {
			*outBaselineSpread = torch::zeros_like(takenScores);
		}
	}

	return takenScores - baselineScores;
}

// Phi_head(s) = mean over the K provided action samples of a soft-max over the goal set of the head's
// contrastive reachability score. The samples are drawn from the POLICY by the caller, so Phi is the
// on-policy reachability potential (action-marginalized => a clean state potential). A single-goal set
// (car contact) makes the soft-max a no-op. Chunked to bound memory.
static torch::Tensor ComputeHeadPotential(
	GGL::ContrastiveGoalLearner* critic,
	torch::Tensor states, torch::Tensor sampledActions, torch::Tensor goals,
	float temp, int64_t numActions, int64_t chunkSize
) {
	RG_NO_GRAD;
	int64_t n = states.size(0);
	int64_t numGoals = goals.size(0);
	int64_t K = sampledActions.size(1);
	float safeTemp = RS_MAX(1e-3f, temp);
	if (chunkSize <= 0)
		chunkSize = n;

	torch::Tensor phi = torch::zeros({ n }, torch::TensorOptions().dtype(torch::kFloat32).device(states.device()));
	for (int64_t start = 0; start < n; start += chunkSize) {
		int64_t stop = RS_MIN(start + chunkSize, n);
		int64_t m = stop - start;
		torch::Tensor sChunk = states.slice(0, start, stop);
		torch::Tensor aChunk = sampledActions.slice(0, start, stop); // [m, K]
		torch::Tensor acc = torch::zeros({ m }, phi.options());
		for (int64_t k = 0; k < K; k++) {
			torch::Tensor aRep = OneHotActions(aChunk.select(1, k), numActions);
			torch::Tensor scores = torch::zeros({ m, numGoals }, phi.options());
			for (int64_t g = 0; g < numGoals; g++) {
				torch::Tensor gRow = goals.slice(0, g, g + 1).expand({ m, goals.size(1) });
				scores.select(1, g).copy_(critic->Score(sChunk, aRep, gRow));
			}
			torch::Tensor agg = (numGoals > 1)
				? (safeTemp * torch::logsumexp(scores / safeTemp, 1))
				: scores.select(1, 0);
			acc = acc + agg;
		}
		phi.slice(0, start, stop).copy_(acc / (float)K);
	}
	return phi;
}

static void PrepareGCRLPolicyAdvantages(GGL::PPOLearner* learner, GGL::ExperienceBuffer& experience, GGL::Report& report, uint64_t totalTimesteps) {
	if (!learner->config.contrastiveGoal.enabled)
		return;

	if (!learner->contrastiveGoalLearner)
		RG_ERR_CLOSE("GCRL config is enabled but no contrastive learner exists");
	if (!experience.data.actions.defined() || !experience.data.herGoals.defined() || !experience.data.actionMasks.defined())
		RG_ERR_CLOSE("GCRL config is enabled but rollout goal/action tensors are missing");

	RG_NO_GRAD;

	auto device = learner->device;
	auto& cfg = learner->config.contrastiveGoal;

	torch::Tensor baseAdvRaw = experience.data.advantages.to(device);
	report["PPO Advantage Mean"] = TensorMean(baseAdvRaw);
	report["PPO Advantage Std"] = TensorStd(baseAdvRaw);

	torch::Tensor states = experience.data.states.to(device);
	torch::Tensor actions = experience.data.actions.to(device).to(torch::kLong);
	torch::Tensor actionMasks = experience.data.actionMasks.to(device);
	// Magnitude-blend: each critic's per-row advantage is (taken - mean_baseline) /
	// (baseline_spread + sigmaFloor), clamped, and NOT renormalized to unit std. A
	// critic that can't discriminate the action -> numerator ~0 -> contributes
	// ~nothing automatically; a strong critic contributes in proportion. Differential
	// timing (car early / goal late) emerges from real contribution -- no gate.
	auto criticSep = [&](GGL::ContrastiveGoalLearner* critic, torch::Tensor goals) {
		torch::Tensor taken, baseline, spread;
		torch::Tensor edge = ComputeGCRLAdvantageWithBaseline(
			critic, states, actions, actionMasks, goals,
			learner->numActions, cfg.policyScoreBatchSize, cfg.baselineActionSamples,
			&taken, &baseline, &spread);
		torch::Tensor sep = torch::clamp(edge / (spread + cfg.sigmaFloor), -cfg.gcrlSepClamp, cfg.gcrlSepClamp);
		return std::make_tuple(sep, TensorMean(edge.abs()), TensorMean(spread));
	};

	// Goal critic: scores HER achieved future ball (in-distribution), not the net.
	auto goalRes = criticSep(learner->contrastiveGoalLearner, experience.data.herGoals.to(device));
	torch::Tensor sepSum = std::get<0>(goalRes);
	report["GCRL/Goal Edge Mean"] = std::get<1>(goalRes);
	report["GCRL/Goal Baseline Spread"] = std::get<2>(goalRes);
	report["GCRL/Goal Separation"] = TensorMean(sepSum.abs());

	bool carActive = cfg.useCarCritic && learner->carContrastiveLearner
		&& experience.data.carHerGoals.defined()
		&& experience.data.carHerGoals.size(0) == states.size(0);
	if (carActive) {
		// Car critic: egocentric car-local ball (controllability), short-horizon goals.
		auto carRes = criticSep(learner->carContrastiveLearner, experience.data.carHerGoals.to(device));
		report["GCRL/Car Edge Mean"] = std::get<1>(carRes);
		report["GCRL/Car Baseline Spread"] = std::get<2>(carRes);
		report["GCRL/Car Separation"] = TensorMean(std::get<0>(carRes).abs());
		sepSum = sepSum + std::get<0>(carRes);
	}
	report["GCRL/Car Active"] = carActive ? 1.f : 0.f;

	bool boostActive = cfg.useBoostCritic && learner->boostContrastiveLearner
		&& experience.data.boostHerGoals.defined()
		&& experience.data.boostHerGoals.size(0) == states.size(0);
	if (boostActive) {
		// Boost critic: own boost level (reachability toward full boost), short-horizon goals.
		auto boostRes = criticSep(learner->boostContrastiveLearner, experience.data.boostHerGoals.to(device));
		report["GCRL/Boost Edge Mean"] = std::get<1>(boostRes);
		report["GCRL/Boost Baseline Spread"] = std::get<2>(boostRes);
		report["GCRL/Boost Separation"] = TensorMean(std::get<0>(boostRes).abs());
		sepSum = sepSum + std::get<0>(boostRes);
	}
	report["GCRL/Boost Active"] = boostActive ? 1.f : 0.f;

	// Single global lambda: short bootstrap warmup -> hold. No separation gate; weak
	// critics self-attenuate via their ~0 numerator above.
	float warmupProgress = cfg.gcrlLambdaWarmupSteps == 0 ? 1.f
		: RS_CLAMP(totalTimesteps / (float)cfg.gcrlLambdaWarmupSteps, 0.f, 1.f);
	float lambdaEff = cfg.gcrlLambda * warmupProgress;

	torch::Tensor baseNorm = NormalizeAdvantage(baseAdvRaw, cfg.sigmaFloor);
	torch::Tensor gcrlAdv = lambdaEff * sepSum;
	torch::Tensor policyAdvantage = baseNorm + gcrlAdv;
	experience.data.advantages = policyAdvantage.detach().to(experience.data.advantages.device());
	experience.data.crlAdvantages = gcrlAdv.detach().to(experience.data.advantages.device());

	report["A Policy Mean"] = TensorMean(policyAdvantage);
	report["A Policy Std"] = TensorStd(policyAdvantage);
	report["CRL Advantage Mean"] = TensorMean(gcrlAdv);
	report["CRL Advantage Std"] = TensorStd(gcrlAdv);
	report["CRL Lambda Target"] = cfg.gcrlLambda;
	report["CRL Lambda Effective"] = lambdaEff;
	report["CRL Lambda Warmup Progress"] = warmupProgress;
}

void GGL::PPOLearner::Learn(ExperienceBuffer& experience, Report& report, bool isFirstIteration, uint64_t totalTimesteps) {
	auto mseLoss = torch::nn::MSELoss();

	MutAvgTracker
		avgEntropy,
		avgDivergence,
		avgPolicyLoss,
		avgRelEntropyLoss,
		avgCriticLoss,
		avgGuidingLoss,
		avgRatio,
		avgClip;

	if (config.contrastiveGoal.enabled) {
		ContrastiveGoalStats crlTrainStats = contrastiveGoalLearner->Train(experience.data, experience.rng);

		report["CRL Critic Loss"] = crlTrainStats.loss;
		report["GCRL Row Loss"] = crlTrainStats.rowLoss;
		report["GCRL Column Loss"] = crlTrainStats.columnLoss;
		report["CRL Positive Logit"] = crlTrainStats.positiveLogitMean;
		report["CRL Negative Logit"] = crlTrainStats.negativeLogitMean;
		report["CRL State-Action Embedding Norm"] = crlTrainStats.stateActionEmbeddingNorm;
		report["CRL Goal Embedding Norm"] = crlTrainStats.goalEmbeddingNorm;
		report["GCRL Categorical Accuracy"] = crlTrainStats.categoricalAccuracy;
		report["GCRL LogSumExp"] = crlTrainStats.logsumexpMean;
		report["Horizon Immediate Realized"] = crlTrainStats.realizedImmediate;
		report["Horizon Short Realized"] = crlTrainStats.realizedShort;
		report["Horizon Medium Realized"] = crlTrainStats.realizedMedium;
		report["Horizon Long Realized"] = crlTrainStats.realizedLong;
		report["CRL Anchors Used"] = crlTrainStats.anchorsUsed;
		report["CRL Train Samples Used"] = crlTrainStats.trainSamplesUsed;

		if (carContrastiveLearner && experience.data.carHerGoals.defined()) {
			ContrastiveGoalStats carStats = carContrastiveLearner->Train(experience.data, experience.rng);
			report["GCRL/Car Critic Loss"] = carStats.loss;
			report["GCRL/Car Categorical Accuracy"] = carStats.categoricalAccuracy;
			report["GCRL/Car Train Samples Used"] = (float)carStats.trainSamplesUsed;
		}

		if (boostContrastiveLearner && experience.data.boostHerGoals.defined()) {
			ContrastiveGoalStats boostStats = boostContrastiveLearner->Train(experience.data, experience.rng);
			report["GCRL/Boost Critic Loss"] = boostStats.loss;
			report["GCRL/Boost Categorical Accuracy"] = boostStats.categoricalAccuracy;
			report["GCRL/Boost Train Samples Used"] = (float)boostStats.trainSamplesUsed;
		}

		// In potential-shaping mode GCRL enters via reward shaping (done pre-GAE in the collector),
		// so the advantage-blend is skipped -- the critics are still trained above.
		if (!config.contrastiveGoal.usePotentialShaping)
			PrepareGCRLPolicyAdvantages(this, experience, report, totalTimesteps);
	}

	{
		RG_NO_GRAD;
		if (experience.data.actions.defined()) {
			auto actionCounts = torch::bincount(experience.data.actions.to(torch::kLong).cpu(), {}, numActions).to(torch::kFloat32);
			auto actionFreqs = actionCounts / actionCounts.sum().clamp_min(1);
			for (int i = 0; i < numActions; i++)
				report["Action Frequency " + std::to_string(i)] = actionFreqs[i].item<float>();
		}
	}

	// Save parameters first
	auto policyBefore = models["policy"]->CopyParams();
	auto criticBefore = models["critic"]->CopyParams();

	bool trainPolicy = config.policyLR != 0;
	bool trainCritic = config.criticLR != 0;
	bool trainSharedHead = models["shared_head"] && (trainPolicy || trainCritic);

	// RSNorm: update the running obs stats ONCE per rollout from the freshly
	// collected (raw) observations, BEFORE the K epochs. The stats are then frozen
	// for the entire epoch pass (every minibatch normalizes with these stats inside
	// InferPolicyProbsFromModels / InferCritic). Never update inside the epoch loop.
	if (obsNorm) {
		obsNorm->Update(experience.data.states);
		report["RSNorm/Mean Mu"] = obsNorm->GetMeanMu();
		report["RSNorm/Mean Var"] = obsNorm->GetMeanVar();
		report["RSNorm/Count"] = obsNorm->GetCount();
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
			int64_t actualBatchSize = batchObs.size(0);

			auto fnRunMinibatch = [&](int64_t start, int64_t stop) {

				float batchSizeRatio = (stop - start) / (float)actualBatchSize;

				// Send everything to the device and enforce correct shapes
				auto acts = batchActs.slice(0, start, stop).to(device, true, true);
				auto obs = batchObs.slice(0, start, stop).to(device, true, true);
				auto actionMasks = batchActionMasks.slice(0, start, stop).to(device, true, true);
				
				auto advantages = batchAdvantages.slice(0, start, stop).to(device, true, true);
				auto oldProbs = batchOldProbs.slice(0, start, stop).to(device, true, true);
				auto targetValues = batchTargetValues.slice(0, start, stop).to(device, true, true);

				torch::Tensor probs, logProbs, entropy, ratio, clipped, policyLoss, ppoLoss;
				if (trainPolicy) {

					// Get policy log probs and entropy
					float curEntropy;
					{
						probs = InferPolicyProbsFromModels(models, obs, actionMasks, config.policyTemperature, false, obsNorm);
						logProbs = probs.log().gather(-1, acts.unsqueeze(-1));
						entropy = ComputeEntropy(probs, actionMasks, config.maskEntropy);
						curEntropy = entropy.detach().cpu().item<float>();
						avgEntropy += curEntropy;
					}

					logProbs = logProbs.view_as(oldProbs);

					// Compute PPO loss
					ratio = exp(logProbs - oldProbs);
					avgRatio += ratio.mean().detach().cpu().item<float>();
					clipped = clamp(
						ratio, 1 - config.clipRange, 1 + config.clipRange
					);

					// Compute policy loss
					policyLoss = -min(
						ratio * advantages, clipped * advantages
					).mean();
					float curPolicyLoss = policyLoss.detach().cpu().item<float>();
					avgPolicyLoss += curPolicyLoss;

					if (std::abs(curPolicyLoss) > 1e-12f)
						avgRelEntropyLoss += (curEntropy * config.entropyScale) / curPolicyLoss;

					ppoLoss = (policyLoss - entropy * config.entropyScale) * batchSizeRatio;

					if (config.useGuidingPolicy) {
						torch::Tensor guidingProbs;
						{
							RG_NO_GRAD;
							guidingProbs = InferPolicyProbsFromModels(guidingPolicyModels, obs, actionMasks, config.policyTemperature, config.useHalfPrecision, obsNorm);
						}

						auto guidingLoss = (guidingProbs - probs).abs().mean();
						avgGuidingLoss.Add(guidingLoss.detach().cpu().item<float>());
						guidingLoss = guidingLoss * config.guidingStrength;
						ppoLoss = ppoLoss + guidingLoss;
					}
				}

				torch::Tensor criticLoss;
				if (trainCritic) {
					auto vals = InferCritic(obs);

					// Compute value loss
					vals = vals.view_as(targetValues);
					criticLoss = mseLoss(vals, targetValues) * batchSizeRatio;
					avgCriticLoss += criticLoss.detach().cpu().item<float>();
				}

				if (trainPolicy) {
					// Compute KL divergence & clip fraction using SB3 method for reporting;
					{
						RG_NO_GRAD;

						auto logRatio = logProbs - oldProbs;
						auto klTensor = (exp(logRatio) - 1) - logRatio;
						avgDivergence += klTensor.mean().detach().cpu().item<float>();

						auto clipFraction = mean((abs(ratio - 1) > config.clipRange).to(kFloat));
						avgClip += clipFraction.cpu().item<float>();
					}
				}

				if (trainPolicy && trainCritic) {
					auto combinedLoss = ppoLoss + criticLoss;
					combinedLoss.backward();
				} else {
					if (trainPolicy)
						ppoLoss.backward();
					if (trainCritic)
						criticLoss.backward();
				}
			};

			
			if (device.is_cpu()) {
				// Just run one minibatch
				fnRunMinibatch(0, actualBatchSize);
			} else {
				for (int64_t mbs = 0; mbs < actualBatchSize; mbs += config.miniBatchSize) {
					int64_t start = mbs;
					int64_t stop = std::min<int64_t>(start + config.miniBatchSize, actualBatchSize);
					fnRunMinibatch(start, stop);
				}
			}

			if (trainPolicy)
				nn::utils::clip_grad_norm_(models["policy"]->parameters(), 0.5f);
			if (trainCritic)
				nn::utils::clip_grad_norm_(models["critic"]->parameters(), 0.5f);

			if (trainSharedHead)
				nn::utils::clip_grad_norm_(models["shared_head"]->parameters(), 0.5f);

			models.StepOptims();
		}
	}

	// Compute magnitude of updates made to the policy and value estimator
	auto policyAfter = models["policy"]->CopyParams();
	auto criticAfter = models["critic"]->CopyParams();

	float policyUpdateMagnitude = (policyBefore - policyAfter).norm().item<float>();
	float criticUpdateMagnitude = (criticBefore - criticAfter).norm().item<float>();

	// Assemble and return report
	report["Policy Entropy"] = avgEntropy.Get();
	report["Mean KL Divergence"] = avgDivergence.Get();
	if (!isFirstIteration) {
		// These metrics give bad data on the first iteration, which will mess up graph scaling
		// So we'll just skip them for the first iteration
		report["Policy Loss"] = avgPolicyLoss.Get();
		report["Policy Relative Entropy Loss"] = avgRelEntropyLoss.Get();
		report["Critic Loss"] = avgCriticLoss.Get();

		if (config.useGuidingPolicy)
			report["Guiding Loss"] = avgGuidingLoss.Get();

		report["SB3 Clip Fraction"] = avgClip.Get();
		report["Policy Update Magnitude"] = policyUpdateMagnitude;
		report["Critic Update Magnitude"] = criticUpdateMagnitude;
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
		oldProbs = InferPolicyProbsFromModels(oldModels, oldObs, oldActionMasks, config.policyTemperature, config.useHalfPrecision, obsNorm);
		report["Old Policy Entropy"] = ComputeEntropy(oldProbs, oldActionMasks, config.maskEntropy).detach().cpu().item<float>();

		if (actionMaps.defined())
			oldProbs = oldProbs.gather(1, actionMaps);
	}

	for (auto& model : GetPolicyModels())
		model->SetOptimLR(tlConfig.lr);

	auto policyBefore = models["policy"]->CopyParams();
	
	for (int i = 0; i < tlConfig.epochs; i++) {
		torch::Tensor newProbs = InferPolicyProbsFromModels(models, newObs, newActionMasks, config.policyTemperature, false, obsNorm);

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

// Phi_defense(s) = -(soft-max over the agent's opponents of their goal-reachability Phi_goal). Reuses
// the already-computed goal-head Phi, regrouped by (arena,step) so each row's simultaneous opponents
// are aggregated. A potential -> policy-invariant; agency-correct (the threat is the opponents').
static torch::Tensor ComputeDefensePotential(
	torch::Tensor phiGoal, torch::Tensor groupKeys, torch::Tensor teams, float temp
) {
	torch::Tensor pg = phiGoal.to(torch::kCPU).to(torch::kFloat32).contiguous();
	torch::Tensor gk = groupKeys.to(torch::kCPU).to(torch::kLong).contiguous();
	torch::Tensor tm = teams.to(torch::kCPU).to(torch::kInt8).contiguous();
	int64_t n = pg.size(0);
	const float* pgp = pg.data_ptr<float>();
	const int64_t* gkp = gk.data_ptr<int64_t>();
	const int8_t* tmp = tm.data_ptr<int8_t>();
	float safeTemp = RS_MAX(1e-3f, temp);

	std::unordered_map<int64_t, std::vector<int64_t>> groups;
	groups.reserve((size_t)n);
	for (int64_t i = 0; i < n; i++)
		groups[gkp[i]].push_back(i);

	torch::Tensor phiDef = torch::zeros({ n }, torch::TensorOptions().dtype(torch::kFloat32));
	float* pdp = phiDef.data_ptr<float>();
	for (int64_t i = 0; i < n; i++) {
		const std::vector<int64_t>& grp = groups[gkp[i]];
		float maxV = -FLT_MAX;
		for (int64_t j : grp)
			if (tmp[j] != tmp[i])
				maxV = RS_MAX(maxV, pgp[j]);
		if (maxV == -FLT_MAX) { pdp[i] = 0.f; continue; } // no opponents in this group
		float sum = 0.f;
		for (int64_t j : grp)
			if (tmp[j] != tmp[i])
				sum += expf((pgp[j] - maxV) / safeTemp);
		pdp[i] = -(maxV + safeTemp * logf(sum)); // negated soft-max threat
	}
	return phiDef;
}

torch::Tensor GGL::PPOLearner::ComputePotentialShaping(
	torch::Tensor states, torch::Tensor actionMasks, torch::Tensor segmentIds, torch::Tensor terminals,
	torch::Tensor truncNextStates, float gaeGamma, torch::Tensor contactGoal, torch::Tensor scoringRangeGoals,
	torch::Tensor boostGoal, torch::Tensor defenseGroupKeys, torch::Tensor defenseTeams, Report& report
) {
	RG_NO_GRAD;
	auto& cfg = config.contrastiveGoal;
	torch::Tensor s = states.to(device);
	torch::Tensor masks = actionMasks.to(device);
	int64_t n = s.size(0);
	if (n <= 1)
		return torch::zeros({ n }, torch::TensorOptions().dtype(torch::kFloat32));
	int K = RS_MAX(1, cfg.baselineActionSamples);

	// Sample K actions per state from the POLICY -> on-policy reachability potential.
	torch::Tensor sampledActions = torch::multinomial(
		InferPolicyProbsFromModels(models, s, masks, config.policyTemperature, false, obsNorm), K, /*replacement=*/true);

	// Truncation next-states + their policy action samples, for the Phi bootstrap at TRUNCATED boundaries.
	bool hasTrunc = truncNextStates.defined() && truncNextStates.size(0) > 0;
	torch::Tensor truncS, truncSampled;
	if (hasTrunc) {
		truncS = truncNextStates.to(device);
		int64_t numTruncs = truncS.size(0);
		torch::Tensor truncMasks = torch::ones({ numTruncs, (int64_t)numActions }, masks.options());
		truncSampled = torch::multinomial(
			InferPolicyProbsFromModels(models, truncS, truncMasks, config.policyTemperature, false, obsNorm), K, true);
	}

	// Boundary structure (CPU, built once, shared across heads): per row -> mid-episode next row, or a
	// TRUNCATED forward-index (into truncNextStates), or NORMAL/last (-> Phi(s')=0). The forward trunc
	// counter advances on EVERY truncation so it stays aligned with truncNextStates order.
	torch::Tensor segCpu = segmentIds.to(torch::kCPU).to(torch::kLong).contiguous();
	torch::Tensor termCpu = terminals.to(torch::kCPU).to(torch::kInt8).contiguous();
	const int64_t* segP = segCpu.data_ptr<int64_t>();
	const int8_t* termP = termCpu.data_ptr<int8_t>();
	std::vector<int64_t> nextRow(n, 0), truncGather(n, 0);
	std::vector<float> midSel(n, 0.f), truncSel(n, 0.f);
	int64_t tIdx = 0;
	for (int64_t i = 0; i < n; i++) {
		if (i + 1 < n && segP[i + 1] == segP[i]) {
			nextRow[i] = i + 1; midSel[i] = 1.f;            // mid-episode -> Phi[i+1]
		} else if (termP[i] == RLGC::TerminalType::TRUNCATED) {
			if (hasTrunc) { truncGather[i] = tIdx; truncSel[i] = 1.f; } // TRUNCATED -> Phi(trunc next)
			tIdx++;
		} // else NORMAL terminal / last row -> Phi(s') = 0
	}
	// Contract: one truncation next-state per TRUNCATED boundary, in order. Guard against a mismatch
	// (would otherwise be an out-of-bounds index_select on phiTrunc below).
	if (hasTrunc && tIdx != truncS.size(0))
		RG_ERR_CLOSE("Potential shaping: counted " << tIdx << " TRUNCATED boundaries but have "
			<< truncS.size(0) << " truncation next-states");
	auto opts = torch::TensorOptions().dtype(torch::kLong);
	torch::Tensor nextRowT = torch::tensor(nextRow, opts).to(device);
	torch::Tensor truncGatherT = torch::tensor(truncGather, opts).to(device);
	torch::Tensor midSelT = torch::tensor(midSel).to(device);
	torch::Tensor truncSelT = torch::tensor(truncSel).to(device);

	// Phi(s') with proper boundaries: mid -> Phi[next]; TRUNCATED -> Phi(trunc next); NORMAL/last -> 0.
	auto phiNextOf = [&](torch::Tensor phi, torch::Tensor phiTrunc) -> torch::Tensor {
		torch::Tensor out = phi.index_select(0, nextRowT) * midSelT;
		if (phiTrunc.defined() && phiTrunc.size(0) > 0)
			out = out + phiTrunc.index_select(0, truncGatherT) * truncSelT;
		return out;
	};

	torch::Tensor total = torch::zeros({ n }, torch::TensorOptions().dtype(torch::kFloat32).device(device));

	// Goal head (offense) -- its Phi (states + trunc next-states) is reused for the defense head below.
	torch::Tensor goalGoals = scoringRangeGoals.to(device);
	torch::Tensor phiGoal = ComputeHeadPotential(contrastiveGoalLearner, s, sampledActions, goalGoals, cfg.potentialScoringTemp, numActions, cfg.policyScoreBatchSize);
	torch::Tensor phiGoalTrunc = hasTrunc
		? ComputeHeadPotential(contrastiveGoalLearner, truncS, truncSampled, goalGoals, cfg.potentialScoringTemp, numActions, cfg.policyScoreBatchSize)
		: torch::Tensor();
	{
		torch::Tensor shaping = gaeGamma * phiNextOf(phiGoal, phiGoalTrunc) - phiGoal;
		total = total + shaping;
		report["GCRL/Potential Goal Mean"] = TensorMean(phiGoal);
		report["GCRL/Shaping Goal AbsMean"] = TensorMean(shaping.abs());
	}

	// Car head (offense): contact goal (single -> soft-max is a no-op).
	if (carContrastiveLearner) {
		torch::Tensor carGoal = contactGoal.to(device);
		torch::Tensor phiCar = ComputeHeadPotential(carContrastiveLearner, s, sampledActions, carGoal, 1.0f, numActions, cfg.policyScoreBatchSize);
		torch::Tensor phiCarTrunc = hasTrunc
			? ComputeHeadPotential(carContrastiveLearner, truncS, truncSampled, carGoal, 1.0f, numActions, cfg.policyScoreBatchSize)
			: torch::Tensor();
		torch::Tensor shaping = gaeGamma * phiNextOf(phiCar, phiCarTrunc) - phiCar;
		total = total + shaping;
		report["GCRL/Potential Car Mean"] = TensorMean(phiCar);
		report["GCRL/Shaping Car AbsMean"] = TensorMean(shaping.abs());
	}

	// Boost head (offense): goal = full boost (single -> soft-max is a no-op).
	if (boostContrastiveLearner && boostGoal.defined()) {
		torch::Tensor bGoal = boostGoal.to(device);
		torch::Tensor phiBoost = ComputeHeadPotential(boostContrastiveLearner, s, sampledActions, bGoal, 1.0f, numActions, cfg.policyScoreBatchSize);
		torch::Tensor phiBoostTrunc = hasTrunc
			? ComputeHeadPotential(boostContrastiveLearner, truncS, truncSampled, bGoal, 1.0f, numActions, cfg.policyScoreBatchSize)
			: torch::Tensor();
		torch::Tensor shaping = gaeGamma * phiNextOf(phiBoost, phiBoostTrunc) - phiBoost;
		total = total + shaping;
		report["GCRL/Potential Boost Mean"] = TensorMean(phiBoost);
		report["GCRL/Shaping Boost AbsMean"] = TensorMean(shaping.abs());
	}

	// Defense head: reuse phiGoal regrouped over opponents. No truncation bootstrap (trunc next-states
	// have no opponent grouping), so Phi_defense(s')=0 at every boundary (mid-episode term only).
	if (cfg.potentialDefense && defenseGroupKeys.defined() && defenseGroupKeys.size(0) == n) {
		torch::Tensor phiDef = ComputeDefensePotential(phiGoal, defenseGroupKeys, defenseTeams, cfg.potentialScoringTemp).to(device);
		torch::Tensor shaping = gaeGamma * (phiDef.index_select(0, nextRowT) * midSelT) - phiDef;
		total = total + shaping;
		report["GCRL/Potential Defense Mean"] = TensorMean(phiDef);
		report["GCRL/Shaping Defense AbsMean"] = TensorMean(shaping.abs());
	}

	total = total * cfg.potentialShapingScale;
	report["GCRL/Shaping Total AbsMean"] = TensorMean(total.abs());
	return total.detach().to(torch::kCPU);
}

void GGL::PPOLearner::SaveTo(std::filesystem::path folderPath) {
	models.Save(folderPath);
	if (contrastiveGoalLearner)
		contrastiveGoalLearner->Save(folderPath);
	if (carContrastiveLearner)
		carContrastiveLearner->Save(folderPath);
	if (boostContrastiveLearner)
		boostContrastiveLearner->Save(folderPath);
}

void GGL::PPOLearner::LoadFrom(std::filesystem::path folderPath)  {
	if (!std::filesystem::is_directory(folderPath))
		RG_ERR_CLOSE("PPOLearner:LoadFrom(): Path " << folderPath << " is not a valid directory");

	models.Load(folderPath, true, true);
	if (contrastiveGoalLearner) {
		contrastiveGoalLearner->Load(folderPath, true, true);
	}
	if (carContrastiveLearner) {
		carContrastiveLearner->Load(folderPath, true, true);
	}
	if (boostContrastiveLearner) {
		boostContrastiveLearner->Load(folderPath, true, true);
	}

	SetLearningRates(config.policyLR, config.criticLR);
}

void GGL::PPOLearner::SetLearningRates(float policyLR, float criticLR) {
	config.policyLR = policyLR;
	config.criticLR = criticLR;

	models["policy"]->SetOptimLR(policyLR);
	models["critic"]->SetOptimLR(criticLR);

	if (models["shared_head"])
		models["shared_head"]->SetOptimLR(RS_MIN(policyLR, criticLR));

	if (contrastiveGoalLearner)
		contrastiveGoalLearner->SetLearningRate(config.contrastiveGoal.criticLR);
	if (carContrastiveLearner)
		carContrastiveLearner->SetLearningRate(config.contrastiveGoal.criticLR);
	if (boostContrastiveLearner)
		boostContrastiveLearner->SetLearningRate(config.contrastiveGoal.criticLR);

	RG_LOG("PPOLearner: " << RS_STR(std::scientific << "Set learning rate to [" << policyLR << ", " << criticLR << "]"));
}

GGL::ModelSet GGL::PPOLearner::GetPolicyModels() {
	ModelSet result = {};
	for (Model* model : models) {
		if (std::string(model->modelName) == "critic")
			continue;
		
		result.Add(model);
	}
	return result;
}
