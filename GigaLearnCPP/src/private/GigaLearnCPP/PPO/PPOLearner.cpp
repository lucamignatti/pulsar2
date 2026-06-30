#include "PPOLearner.h"

#include <torch/nn/utils/convert_parameters.h>
#include <torch/nn/utils/clip_grad.h>
#include <torch/csrc/api/include/torch/serialize.h>
#include <torch/cuda.h>
#include <public/GigaLearnCPP/Util/AvgTracker.h>
#include <public/GigaLearnCPP/Util/Timer.h>
#include <cmath>

using namespace torch;

GGL::PPOLearner::PPOLearner(int obsSize, int numActions, PPOLearnerConfig _config, Device _device) :
	config(_config), device(_device), obsSize(obsSize), numActions(numActions) {

	if (config.miniBatchSize == 0)
		config.miniBatchSize = config.batchSize;

	// Adaptive entropy controller starts from the configured fixed scale (clamped
	// to the ceiling); LoadStats overwrites it from the checkpoint on resume.
	curEntropyScale = config.entropyScale;
	if (config.adaptiveEntropy)
		curEntropyScale = RS_CLAMP(curEntropyScale, config.minEntropyScale, config.maxEntropyScale);

	// TRIAD-NATIVE GCRL coupling controller seed (LoadStats overwrites on resume).
	gcrlLambdaEff = config.contrastiveGoal.gcrlLambda;
	gcrlRatioEma = config.contrastiveGoal.gcrlRatioTarget;
	gcrlRenormStd = config.contrastiveGoal.gcrlRenormStdEma; // seed at the target std (~0.7)

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
		if (!config.sharedHead.IsValid())
			RG_ERR_CLOSE("PPOLearner: contrastiveGoal requires a sharedHead (phi = shared_head + phi_tail)");
	}

	MakeModels(true, obsSize, numActions, config.sharedHead, config.policy, config.critic, device, models);

	SetLearningRates(config.policyLR, config.criticLR);

	// RSNorm must be created before GCRL so the pointer passed to the GCRL learners is valid.
	if (config.rsNorm.enabled) {
		obsNorm = new RSNorm(obsSize, config.rsNorm.eps, config.rsNorm.initVar, config.rsNorm.initCount, config.rsNorm.clipRange);
		RG_LOG("RSNorm (running observation normalization) enabled over " << obsSize << " dims");
	}

	if (config.contrastiveGoal.enabled) {
		Model* sh = models["shared_head"];
		contrastiveGoalLearner = new ContrastiveGoalLearner(obsSize, numActions, config.contrastiveGoal, device, sh, obsNorm);
		RG_LOG("GCRL scoring auxiliary enabled (phi = shared_head + phi_tail)");
		if (config.contrastiveGoal.useCarCritic) {
			// CONTROL critic: ball-agnostic car self-state goal; trains on all rows (ignores the ball-moved mask).
			carContrastiveLearner = new ContrastiveGoalLearner(obsSize, numActions, config.contrastiveGoal, device, sh, obsNorm, "gcrl_car", true, false);
			RG_LOG("GCRL control critic enabled (ball-agnostic car self-state)");
		}
		if (config.contrastiveGoal.useApproachCritic) {
			// APPROACH critic (bootstrap): egocentric car-local ball goal (6d); trains on all rows.
			// useCarGoals=false (=> goalInputSize=6), applyTrainMask=false, useApproachGoals=true.
			approachContrastiveLearner = new ContrastiveGoalLearner(obsSize, numActions, config.contrastiveGoal, device, sh, obsNorm, "gcrl_approach", false, false, true);
			RG_LOG("GCRL approach critic enabled (egocentric ball -- cold-start bootstrap)");
		}
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
	delete contrastiveGoalLearner;
	delete carContrastiveLearner;
	delete approachContrastiveLearner;
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

torch::Tensor GGL::PPOLearner::InferCriticFromModels(ModelSet& models, torch::Tensor obs, bool halfPrec, const RSNorm* obsNorm) {

	// RSNorm: same shared normalizer as the policy, first op (stop-grad).
	if (obsNorm)
		obs = obsNorm->Normalize(obs);

	if (models["shared_head"])
		obs = models["shared_head"]->Forward(obs, halfPrec);

	return models["critic"]->Forward(obs, halfPrec).flatten();
}

torch::Tensor GGL::PPOLearner::InferCritic(torch::Tensor obs) {
	return InferCriticFromModels(this->models, obs, config.useHalfPrecision, obsNorm);
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

// embeddings: pre-computed shared_head output [N, embeddingDim], already detached.
static torch::Tensor ScoreGCRLChunks(GGL::ContrastiveGoalLearner* learner, torch::Tensor embeddings, torch::Tensor actions, torch::Tensor goals, int64_t numActions, int64_t chunkSize) {
	std::vector<torch::Tensor> chunks;
	int64_t n = embeddings.size(0);
	if (chunkSize <= 0)
		chunkSize = n;

	chunks.reserve((n + chunkSize - 1) / chunkSize);
	for (int64_t start = 0; start < n; start += chunkSize) {
		int64_t stop = RS_MIN(start + chunkSize, n);
		torch::Tensor actionRepresentations = OneHotActions(actions.slice(0, start, stop), numActions);
		chunks.push_back(learner->Score(
			embeddings.slice(0, start, stop),
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

// embeddings: shared_head output for all states [N, embeddingDim], already detached.
// Baseline samples are batched into a single forward pass: states/goals tiled N_samples
// times, all random action draws stacked, one ScoreGCRLChunks call instead of N_samples.
static torch::Tensor ComputeGCRLAdvantageWithBaseline(
	GGL::ContrastiveGoalLearner* learner,
	torch::Tensor embeddings,
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
	torch::Tensor takenScores = ScoreGCRLChunks(learner, embeddings, actions, goals, numActions, chunkSize);

	if (outTakenScores)
		*outTakenScores = takenScores;

	if (baselineSamples <= 0) {
		torch::Tensor zero = torch::zeros_like(takenScores);
		if (outBaselineScores) *outBaselineScores = zero;
		if (outBaselineSpread) *outBaselineSpread = zero;
		return takenScores;
	}

	// TRIAD-NATIVE memory fix (ROCm OOM): score the K baseline samples ONE AT A TIME and
	// accumulate a running sum + sum-of-squares, instead of tiling embeddings/goals to
	// [n*K, D] up front. The old embeddings.repeat({K,1}) materialized n*K*embeddingDim*4
	// bytes in a SINGLE tensor (22 GiB at K=16, n~1.34M -> HIP OOM on the 16 GB card).
	// ScoreGCRLChunks already chunks internally, so total compute is identical and peak
	// memory is O(n) not O(n*K). The baseline mean and population spread are numerically
	// identical to the old batched form (var = E[x^2] - E[x]^2 == centered.pow(2).mean(0)).
	torch::Tensor baselineSum = torch::zeros_like(takenScores);
	torch::Tensor baselineSqSum = (outBaselineSpread && baselineSamples > 1)
		? torch::zeros_like(takenScores) : torch::Tensor();
	for (int i = 0; i < baselineSamples; i++) {
		torch::Tensor altActions = SampleValidActions(actionMasks);
		torch::Tensor sampleScores = ScoreGCRLChunks(learner, embeddings, altActions, goals, numActions, chunkSize);
		baselineSum = baselineSum + sampleScores;
		if (baselineSqSum.defined())
			baselineSqSum = baselineSqSum + sampleScores.pow(2);
	}
	torch::Tensor baselineScores = baselineSum / (float)baselineSamples;

	if (outBaselineScores)
		*outBaselineScores = baselineScores;
	if (outBaselineSpread) {
		if (baselineSqSum.defined()) {
			torch::Tensor var = baselineSqSum / (float)baselineSamples - baselineScores.pow(2);
			*outBaselineSpread = var.clamp_min(0).sqrt();
		} else {
			*outBaselineSpread = torch::zeros_like(takenScores);
		}
	}

	return takenScores - baselineScores;
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

	torch::Tensor rawStates = experience.data.states.to(device);
	torch::Tensor actions = experience.data.actions.to(device).to(torch::kLong);
	torch::Tensor actionMasks = experience.data.actionMasks.to(device);

	// Compute shared_head embeddings once; both GCRL critics reuse this tensor.
	// Apply RSNorm first if enabled (consistent with how actor/critic use the trunk).
	torch::Tensor normStates = learner->obsNorm ? learner->obsNorm->Normalize(rawStates) : rawStates;
	torch::Tensor embeddings = learner->models["shared_head"]->Forward(normStates, false).detach();

	// Magnitude-blend: each critic's per-row advantage is (taken - mean_baseline) /
	// (baseline_spread + sigmaFloor), clamped, and NOT renormalized to unit std. A
	// critic that can't discriminate the action -> numerator ~0 -> contributes
	// ~nothing automatically; a strong critic contributes in proportion. Differential
	// timing (car early / goal late) emerges from real contribution -- no gate.
	auto criticSep = [&](GGL::ContrastiveGoalLearner* critic, torch::Tensor goals) {
		torch::Tensor taken, baseline, spread;
		torch::Tensor edge = ComputeGCRLAdvantageWithBaseline(
			critic, embeddings, actions, actionMasks, goals,
			learner->numActions, cfg.policyScoreBatchSize, cfg.baselineActionSamples,
			&taken, &baseline, &spread);
		// TRIAD-NATIVE: BATCH-normalize the edge -- the within-state spread DIVISOR is REMOVED
		// (it was 0/0 in the non-discriminating far-field: tiny numerator over a collapsing
		// denominator slammed to the +-clamp, the structural z533fbde 0.27->8 CAR explosion).
		// Then weight by the ALWAYS-ON smooth variance-weight w=sigmoid((crossActionSpread -
		// sigmaMin)/scale): ~1 when the critic discriminates the action, ->0 when it can't
		// (replaces the dead binary gcrlGateRatio gate; per-state suppression is automatic).
		torch::Tensor edgeBN = (edge - edge.mean()) / (edge.std() + 1e-4f);
		torch::Tensor w = torch::sigmoid((spread - cfg.gcrlVarWeightSigmaMin) / cfg.gcrlVarWeightScale);
		// FORK: HARD-suppress a near-zero-spread (non-discriminating) critic so its batch-normed edge cannot
		// reinflate to unit-std noise (the GOALSHORT failure: w's sigmoid floors at ~0.27, not 0, so a dead
		// critic still injected ~30% noise). Linear ramp to 0 as spread->0, complementing the soft sigmoid w.
		// =1 for any healthy critic (spread >= sigmaMin), so CAR is unaffected.
		torch::Tensor spreadGate = torch::clamp(spread / RS_MAX(cfg.gcrlVarWeightSigmaMin, 1e-6f), 0.f, 1.f);
		torch::Tensor wGate = w * spreadGate; // per-row controllability in [0,1] (can the action move reachability-of-goal here) -- this is L(s)
		torch::Tensor sep = torch::clamp(wGate * edgeBN, -cfg.gcrlSepClamp, cfg.gcrlSepClamp);
		return std::make_tuple(sep, TensorMean(edge.abs()), TensorMean(spread), wGate);
	};

	// Competence gate g in [0,1] from the touch-ratio EMA: g~0 cold => the APPROACH bootstrap dominates and
	// CONTROL + POSITIONING are ~off; g~1 once the bot reliably touches the ball => hand off to CONTROL +
	// POSITIONING (+ opponent diversity). smoothstep for a gentle ramp.
	float gLo = cfg.competenceLo, gHi = RS_MAX(cfg.competenceHi, cfg.competenceLo + 1e-6f);
	float gx = RS_CLAMP((learner->touchCompetenceEMA - gLo) / (gHi - gLo), 0.f, 1.f);
	float g = gx * gx * (3.f - 2.f * gx);
	report["GCRL/Competence g"] = g;

	// Goal (world-frame) action-EDGE: action-inert far-field; OFF in production (useGoalCritic=false). Its
	// positioning signal is consumed as the POTENTIAL below, not this edge.
	torch::Tensor sepSum;
	if (cfg.useGoalCritic) {
		auto goalRes = criticSep(learner->contrastiveGoalLearner, experience.data.herGoals.to(device));
		sepSum = std::get<0>(goalRes);
		report["GCRL/Goal Edge Mean"] = std::get<1>(goalRes);
		report["GCRL/Goal Baseline Spread"] = std::get<2>(goalRes);
		report["GCRL/Goal Separation"] = TensorMean(sepSum.abs());
	} else {
		report["GCRL/Goal Edge Mean"] = 0.f;
		report["GCRL/Goal Baseline Spread"] = 0.f;
		report["GCRL/Goal Separation"] = 0.f;
	}

	// APPROACH critic (egocentric ball) -- the cold-start BOOTSTRAP. Weight = (1-g)+approachFloor: dominant
	// cold, annealed to a residual once competent (the reward stack then self-sustains ball contact).
	torch::Tensor ballControllability; // L(s) for the product: the approach (ball) critic's per-row controllability
	bool approachActive = cfg.useApproachCritic && learner->approachContrastiveLearner
		&& experience.data.approachHerGoals.defined()
		&& experience.data.approachHerGoals.size(0) == rawStates.size(0);
	if (approachActive) {
		auto apRes = criticSep(learner->approachContrastiveLearner, experience.data.approachHerGoals.to(device));
		ballControllability = std::get<3>(apRes);
		report["GCRL/Ball Controllability Mean"] = TensorMean(ballControllability);
		float apWeight = (1.f - g) + cfg.approachFloor;
		report["GCRL/Approach Edge Mean"] = std::get<1>(apRes);
		report["GCRL/Approach Baseline Spread"] = std::get<2>(apRes);
		report["GCRL/Approach Separation"] = TensorMean(std::get<0>(apRes).abs());
		report["GCRL/Approach Weight"] = apWeight;
		torch::Tensor apSep = apWeight * std::get<0>(apRes);
		sepSum = sepSum.defined() ? sepSum + apSep : apSep;
	}
	report["GCRL/Approach Active"] = approachActive ? 1.f : 0.f;

	// CONTROL critic (ball-agnostic self-state) -- mechanics. Weight = g: ~0 cold (cannot derail the
	// bootstrap), ramps in post-bootstrap.
	bool carActive = cfg.useCarCritic && learner->carContrastiveLearner
		&& experience.data.carHerGoals.defined()
		&& experience.data.carHerGoals.size(0) == rawStates.size(0);
	if (carActive) {
		auto carRes = criticSep(learner->carContrastiveLearner, experience.data.carHerGoals.to(device));
		report["GCRL/Car Edge Mean"] = std::get<1>(carRes);
		report["GCRL/Car Baseline Spread"] = std::get<2>(carRes);
		report["GCRL/Car Separation"] = TensorMean(std::get<0>(carRes).abs());
		torch::Tensor carSep;
		if (cfg.useProductControl && ballControllability.defined()) {
			// PRODUCT: execute a good car maneuver (selfControlEdge) WHERE that maneuver moves the ball
			// (L(s) = ball controllability), ramped by competence g. Gating the mechanic credit by ball-
			// leverage resolves the self-state critic's orthogonality WITHOUT a proximity magnet -- L gates
			// the credit, it is never maximized, so near-ball-with-sloppy-control pays ~0.
			carSep = g * ballControllability * std::get<0>(carRes);
			report["GCRL/Control Product Mean"] = TensorMean(carSep.abs());
		} else {
			carSep = g * std::get<0>(carRes);
		}
		sepSum = sepSum.defined() ? sepSum + carSep : carSep;
	}
	report["GCRL/Car Active"] = carActive ? 1.f : 0.f;

	// No active GCRL critic: contribute zero so the policy uses the base advantage alone.
	if (!sepSum.defined())
		sepSum = torch::zeros_like(baseAdvRaw);

	float warmupProgress = cfg.gcrlLambdaWarmupSteps == 0 ? 1.f
		: RS_CLAMP(totalTimesteps / (float)cfg.gcrlLambdaWarmupSteps, 0.f, 1.f);

	torch::Tensor baseNorm = NormalizeAdvantage(baseAdvRaw, cfg.sigmaFloor);

	// TRIAD-NATIVE RenormToStd: scale sepSum to a target std (~rho), tracked by an EMA of its
	// own std -- the INNER z533fbde explosion guard (KEPT; the OUTER unit-renorm is DROPPED
	// because it reinflated a weak GCRL signal to unit std exactly in the cold regime).
	float sepStd = sepSum.std().item<float>();
	if (std::isfinite(sepStd))
		learner->gcrlRenormStd = cfg.gcrlRatioEmaDecay * learner->gcrlRenormStd + (1.f - cfg.gcrlRatioEmaDecay) * sepStd;
	float rho = cfg.gcrlRenormStdEma; // target combined std (~0.7)
	torch::Tensor combinedGCRL = sepSum * (rho / (learner->gcrlRenormStd + 1e-6f));

	// Lambda INTEGRAL CONTROLLER: drive std(gcrlAdv)/std(baseNorm) -> gcrlRatioTarget (~1:1, the
	// working-run signature), turning the L3 invariant from a hoped-for property into a set-point.
	// Cold-start protection = the warmup ramp (GCRL ~0 early, ramps to the controlled value).
	float controlledLambda = RS_CLAMP(learner->gcrlLambdaEff, cfg.gcrlLambdaMin, cfg.gcrlLambdaMax);
	float lambdaEff = warmupProgress * controlledLambda;
	torch::Tensor gcrlAdv = lambdaEff * combinedGCRL;

	float stdBase = baseNorm.std().item<float>();
	float stdGcrl = gcrlAdv.std().item<float>();
	float ratioObs = stdGcrl / (stdBase + 1e-6f);
	if (std::isfinite(ratioObs))
		learner->gcrlRatioEma = cfg.gcrlRatioEmaDecay * learner->gcrlRatioEma + (1.f - cfg.gcrlRatioEmaDecay) * ratioObs;
	// Competence-gate the ratio target: strong (gcrlRatioTargetCold) cold so the approach signal reliably
	// drives commitment to the ball (otherwise the cold bootstrap is a marginal stochastic race that the
	// redesign runs lost), annealing to the warm-tuned gcrlRatioTarget as competence rises (the annealed
	// approach WEIGHT then handles the warm ballchase magnet -- the reason the warm target is low).
	float ratioTargetEff = (1.f - g) * cfg.gcrlRatioTargetCold + g * cfg.gcrlRatioTarget;
	report["GCRL/Ratio Target Eff"] = ratioTargetEff;
	learner->gcrlLambdaEff = RS_CLAMP(
		controlledLambda * std::exp(cfg.gcrlLambdaCtrlGain * (ratioTargetEff - learner->gcrlRatioEma)),
		cfg.gcrlLambdaMin, cfg.gcrlLambdaMax);

	// Goal POTENTIAL shaping (positioning). Consume the goal critic as a state-potential Phi(s) =
	// action-marginalized reachability of the per-state SCORING goal (the net), NOT as the one-step
	// action-edge (action-inert far-field -> noise, which is why the edge coupling above stays off).
	// F_t = gamma*Phi(s') - Phi(s) telescopes, so it densely rewards moving toward scoreable states
	// (positioning the egocentric CAR critic can't teach) without being farmable by loitering. Added as a
	// bounded, normalized advantage term -- consistent with this advantage-level blend. Phi is just the
	// action-marginalized baseline score (mean over the K masked-random actions), reused from the helper.
	torch::Tensor goalPotentialAdv = torch::zeros_like(baseAdvRaw);
	bool goalPotentialActive = cfg.useGoalPotential && learner->contrastiveGoalLearner
		&& experience.data.scoringGoals.defined()
		&& experience.data.scoringGoals.size(0) == rawStates.size(0)
		&& experience.data.segmentIds.defined()
		&& experience.data.segmentIds.size(0) == rawStates.size(0);
	// Surface a silently-benched potential (enabled but missing its inputs) instead of just emitting zeros.
	if (cfg.useGoalPotential && !goalPotentialActive) {
		static bool warnedGoalPotential = false;
		if (!warnedGoalPotential) {
			RG_LOG("WARNING: useGoalPotential=true but scoringGoals/segmentIds are undefined or row-misaligned; goal-potential shaping is BENCHED (watch GCRL/Goal Potential Active=0).");
			warnedGoalPotential = true;
		}
	}
	if (goalPotentialActive) {
		torch::Tensor takenG, phi, spreadG;
		ComputeGCRLAdvantageWithBaseline(
			learner->contrastiveGoalLearner, embeddings, actions, actionMasks,
			experience.data.scoringGoals.to(device),
			learner->numActions, cfg.policyScoreBatchSize, cfg.baselineActionSamples,
			&takenG, &phi, &spreadG); // phi = E_a[score(s,a,g*)] = Phi(s)

		int64_t N = phi.size(0);
		torch::Tensor segIds = experience.data.segmentIds.to(device);
		torch::Tensor phiNext = torch::zeros_like(phi);
		torch::Tensor sameSeg = torch::zeros_like(phi);
		if (N > 1) {
			// NB: slice() returns a VIEW; must copy_ through it (operator= would rebind the temporary, a no-op).
			phiNext.slice(0, 0, N - 1).copy_(phi.slice(0, 1, N));                                  // Phi(s_{t+1})
			sameSeg.slice(0, 0, N - 1).copy_((segIds.slice(0, 1, N) == segIds.slice(0, 0, N - 1)).to(torch::kFloat32));
		}
		// At a segment boundary Phi(s')=0 => F = -Phi(s) (standard terminal shaping). The LAST buffer row is
		// likewise treated as terminal-by-construction (phiNext/sameSeg stay 0 there) even if it's a truncation
		// that continues next buffer -- 1 row of ~1.3M, then normalized, so negligible.
		float potGamma = learner->config.gaeGamma;
		torch::Tensor F = potGamma * phiNext * sameSeg - phi;
		// Competence-gated: scale by g so positioning shaping is ~0 cold (when Phi is degenerate noise on the
		// never-moving ball) and ramps to full once the bot moves the ball and Phi becomes informative.
		goalPotentialAdv = NormalizeAdvantage(F, cfg.sigmaFloor) * (g * cfg.gcrlGoalPotentialScale);
		report["GCRL/Goal Potential Mean"] = TensorMean(phi);
		report["GCRL/Goal Potential Shaping Std"] = TensorStd(goalPotentialAdv);
	} else {
		report["GCRL/Goal Potential Mean"] = 0.f;
		report["GCRL/Goal Potential Shaping Std"] = 0.f;
	}
	report["GCRL/Goal Potential Active"] = goalPotentialActive ? 1.f : 0.f;

	// NO outer renorm: baseNorm carries the reward DIRECTION, gcrlAdv the bounded GCRL nudge,
	// goalPotentialAdv the bounded positioning potential.
	torch::Tensor policyAdvantage = baseNorm + gcrlAdv + goalPotentialAdv;
	experience.data.advantages = policyAdvantage.detach().to(experience.data.advantages.device());
	experience.data.crlAdvantages = gcrlAdv.detach().to(experience.data.advantages.device());

	report["A Policy Mean"] = TensorMean(policyAdvantage);
	report["A Policy Std"] = TensorStd(policyAdvantage);
	report["CRL Advantage Mean"] = TensorMean(gcrlAdv);
	report["CRL Advantage Std"] = TensorStd(gcrlAdv);
	report["CRL/Std Ratio (gcrl over base)"] = ratioObs;
	report["CRL/Ratio Ema"] = learner->gcrlRatioEma;
	report["CRL/Renorm Std"] = learner->gcrlRenormStd;
	report["CRL Lambda Effective"] = lambdaEff;
	report["CRL Lambda Controlled"] = learner->gcrlLambdaEff;
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
		// Consumption breakdown timers: GCRL critic training vs the counterfactual-baseline scoring vs the
		// PPO epochs (reported as Consumption/* below). GPU work is async, so each phase is bracketed by a
		// CUDA sync (3/iter, negligible) to attribute wall time honestly instead of to the next sync point.
		if (device.is_cuda()) torch::cuda::synchronize();
		Timer gcrlTrainTimer = {};

		// FORK2: precompute masked policy probs for every buffer state (host, chunked) for the GOALSHORT
		// TD bootstrap's pi-sampled soft value. Only when TD is enabled; CAR/TD-off get an undefined tensor.
		// Kept on CPU (the [N, numActions] tensor is ~0.65GB at N~1.3M — never resident on the GPU).
		torch::Tensor nextPolicyProbs;
		if (config.contrastiveGoal.useTDContrastive
			&& experience.data.states.defined() && experience.data.actionMasks.defined()) {
			RG_NO_GRAD;
			int64_t nRows = experience.data.states.size(0);
			int64_t chunk = std::max<int64_t>(1, config.contrastiveGoal.policyScoreBatchSize);
			std::vector<torch::Tensor> parts;
			parts.reserve((size_t)((nRows + chunk - 1) / chunk));
			for (int64_t s = 0; s < nRows; s += chunk) {
				int64_t e = std::min<int64_t>(s + chunk, nRows);
				torch::Tensor obsChunk = experience.data.states.slice(0, s, e).to(device);
				torch::Tensor maskChunk = experience.data.actionMasks.slice(0, s, e).to(device);
				torch::Tensor probs = InferPolicyProbsFromModels(models, obsChunk, maskChunk, config.policyTemperature, false, obsNorm);
				parts.push_back(probs.to(torch::kCPU));
			}
			nextPolicyProbs = torch::cat(parts, 0);   // [N, numActions], CPU
		}

		// FORK: skip the GOALSHORT critic's training (reclaim its 1024x4 goalEncoder InfoNCE compute) when
		// it is dropped from coupling. Default-constructed stats (zeros) keep the report/Display stable.
		ContrastiveGoalStats crlTrainStats;
		// Train the goal critic when its action-edge coupling is on (useGoalCritic) OR when its POTENTIAL is
		// consumed (useGoalPotential needs a trained Phi). Skipped otherwise to reclaim the InfoNCE compute.
		if (config.contrastiveGoal.useGoalCritic || config.contrastiveGoal.useGoalPotential)
			crlTrainStats = contrastiveGoalLearner->Train(experience.data, experience.rng, totalTimesteps, nextPolicyProbs);

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

		// FORK2 TD-contrastive diagnostics (GOALSHORT). GCRL/ prefix auto-allowlisted.
		report["GCRL/Goal TD Blend R"] = crlTrainStats.tdBlendR;
		report["GCRL/Goal TD Reverted"] = crlTrainStats.tdReverted;
		report["GCRL/Goal TD SoftValue EntropyFrac"] = crlTrainStats.tdSoftValueEntropyFrac;
		report["GCRL/Goal TD Row Loss"] = crlTrainStats.tdRowLoss;
		report["GCRL/Goal TD EMA Drift"] = crlTrainStats.tdEmaDrift;
		report["GCRL/Goal TD Valid Bootstrap Rows"] = crlTrainStats.tdValidBootstrapRows;

		if (carContrastiveLearner && experience.data.carHerGoals.defined()) {
			ContrastiveGoalStats carStats = carContrastiveLearner->Train(experience.data, experience.rng, totalTimesteps, {});
			report["GCRL/Car Critic Loss"] = carStats.loss;
			report["GCRL/Car Categorical Accuracy"] = carStats.categoricalAccuracy;
			report["GCRL/Car Train Samples Used"] = (float)carStats.trainSamplesUsed;
		}

		if (approachContrastiveLearner && experience.data.approachHerGoals.defined()) {
			ContrastiveGoalStats apStats = approachContrastiveLearner->Train(experience.data, experience.rng, totalTimesteps, {});
			report["GCRL/Approach Critic Loss"] = apStats.loss;
			report["GCRL/Approach Categorical Accuracy"] = apStats.categoricalAccuracy;
			report["GCRL/Approach Train Samples Used"] = (float)apStats.trainSamplesUsed;
		}

		if (device.is_cuda()) torch::cuda::synchronize();
		report["Consumption/GCRL Train Time"] = gcrlTrainTimer.Elapsed();

		Timer gcrlScoreTimer = {};
		PrepareGCRLPolicyAdvantages(this, experience, report, totalTimesteps);
		if (device.is_cuda()) torch::cuda::synchronize();
		report["Consumption/GCRL Score Time"] = gcrlScoreTimer.Elapsed();
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

	// BF16 autocast (AMP) and pinned H2D batches are CUDA-only perf levers; both no-op elsewhere.
	bool ampOn = config.useAMP && device.is_cuda();
	bool pinBatches = config.pinBatchMemory && device.is_cuda();

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

	if (device.is_cuda()) torch::cuda::synchronize();
	Timer epochTimer = {};
	for (int epoch = 0; epoch < config.epochs; epoch++) {

		// Get randomly-ordered timesteps for PPO
		auto batches = experience.GetAllBatchesShuffled(config.batchSize, config.overbatching, pinBatches);

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

				// AMP region: the policy/critic forward + loss run under BF16 autocast (matmuls -> BF16
				// tensor cores; softmax/exp/log/layer_norm/losses auto-promote to fp32). backward() runs
				// AFTER RG_AUTOCAST_OFF on fp32 master weights. No-op when useAMP is off / not on CUDA.
				if (ampOn) RG_AUTOCAST_ON();

				// Per-minibatch metric scalars are kept ON-DEVICE here and pulled to the host in ONE hop at the
				// end of the minibatch (the single torch::cat(...).cpu() below) instead of a separate
				// .cpu().item() per metric -- each of those forces a full GPU pipeline drain. The host-side
				// AvgTracker semantics (NaN-skip via Add, the relEntropyLoss guard) are preserved exactly.
				torch::Tensor probs, logProbs, entropy, ratio, clipped, policyLoss, ppoLoss;
				torch::Tensor mEntropy, mRatio, mPolicyLoss, mCriticLoss, mKL, mClip;
				bool didGuiding = false;
				float curGuidingLoss = 0.f;
				if (trainPolicy) {

					// Get policy log probs and entropy
					probs = InferPolicyProbsFromModels(models, obs, actionMasks, config.policyTemperature, false, obsNorm);
					logProbs = probs.log().gather(-1, acts.unsqueeze(-1));
					entropy = ComputeEntropy(probs, actionMasks, config.maskEntropy);
					mEntropy = entropy.detach();

					logProbs = logProbs.view_as(oldProbs);

					// Compute PPO loss
					ratio = exp(logProbs - oldProbs);
					mRatio = ratio.mean().detach();
					clipped = clamp(
						ratio, 1 - config.clipRange, 1 + config.clipRange
					);

					// Compute policy loss
					policyLoss = -min(
						ratio * advantages, clipped * advantages
					).mean();
					mPolicyLoss = policyLoss.detach();

					ppoLoss = (policyLoss - entropy * curEntropyScale) * batchSizeRatio;

					if (config.useGuidingPolicy) {
						torch::Tensor guidingProbs;
						{
							RG_NO_GRAD;
							guidingProbs = InferPolicyProbsFromModels(guidingPolicyModels, obs, actionMasks, config.policyTemperature, config.useHalfPrecision, obsNorm);
						}

						auto guidingLoss = (guidingProbs - probs).abs().mean();
						curGuidingLoss = guidingLoss.detach().cpu().item<float>();
						didGuiding = true;
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
					mCriticLoss = criticLoss.detach();
				}

				// End of AMP region: backward() and the KL/clip reporting below run outside autocast.
				if (ampOn) RG_AUTOCAST_OFF();

				if (trainPolicy) {
					// KL divergence & clip fraction (SB3 method), for reporting. Computed on-device; synced below.
					RG_NO_GRAD;

					auto logRatio = logProbs - oldProbs;
					auto klTensor = (exp(logRatio) - 1) - logRatio;
					mKL = klTensor.mean().detach();
					mClip = mean((abs(ratio - 1) > config.clipRange).to(kFloat)).detach();
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

				// SINGLE device->host sync for all per-minibatch metrics (one torch::cat(...).cpu() in place of
				// ~6 separate .item() syncs). Reading elements from the resident CPU tensor does NOT re-sync.
				{
					std::vector<torch::Tensor> scal;
					if (trainPolicy) {
						scal.push_back(mEntropy.reshape({ 1 }));
						scal.push_back(mRatio.reshape({ 1 }));
						scal.push_back(mPolicyLoss.reshape({ 1 }));
						scal.push_back(mKL.reshape({ 1 }));
						scal.push_back(mClip.reshape({ 1 }));
					}
					if (trainCritic)
						scal.push_back(mCriticLoss.reshape({ 1 }));

					if (!scal.empty()) {
						torch::Tensor host = torch::cat(scal).to(torch::kFloat).cpu();
						const float* h = host.data_ptr<float>();
						int i = 0;
						if (trainPolicy) {
							float curEntropy = h[i++];
							float curRatio = h[i++];
							float curPolicyLoss = h[i++];
							float curKL = h[i++];
							float curClip = h[i++];
							avgEntropy += curEntropy;
							avgRatio += curRatio;
							avgPolicyLoss += curPolicyLoss;
							if (std::abs(curPolicyLoss) > 1e-12f)
								avgRelEntropyLoss += (curEntropy * curEntropyScale) / curPolicyLoss;
							avgDivergence += curKL;
							avgClip += curClip;
						}
						if (trainCritic)
							avgCriticLoss += h[i++];
					}

					if (didGuiding)
						avgGuidingLoss.Add(curGuidingLoss);
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

	if (device.is_cuda()) torch::cuda::synchronize();
	report["Consumption/PPO Epoch Time"] = epochTimer.Elapsed();

	// Compute magnitude of updates made to the policy and value estimator
	auto policyAfter = models["policy"]->CopyParams();
	auto criticAfter = models["critic"]->CopyParams();

	float policyUpdateMagnitude = (policyBefore - policyAfter).norm().item<float>();
	float criticUpdateMagnitude = (criticBefore - criticAfter).norm().item<float>();

	// Entropy PIN (Lagrangian temperature). Drive entropy to config.targetEntropy by
	// adjusting the bonus scale MULTIPLICATIVELY in log-space: scale *= exp(rate*(target -
	// entropy)). Log-space gives wide dynamic range and keeps the scale positive, so the
	// controller can apply WHATEVER bonus is needed to hold entropy at target. The old
	// additive rule clamped to a hard 0.10 ceiling, which let entropy collapse to 0.004
	// once the reward/GCRL gradient overpowered it (runs 5gjwloq2, z533fbde) -- a soft
	// capped bonus cannot pin entropy. maxEntropyScale is now a high sanity bound, not the
	// operating point. Updated AFTER the epochs (sets next iteration's scale). Persisted.
	float curAvgEntropy = avgEntropy.Get();
	if (config.adaptiveEntropy) {
		curEntropyScale *= std::exp(config.entropyScaleAdjustRate * (config.targetEntropy - curAvgEntropy));
		curEntropyScale = RS_CLAMP(curEntropyScale, config.minEntropyScale, config.maxEntropyScale);
	}

	// Assemble and return report
	report["Policy Entropy"] = curAvgEntropy;
	report["Entropy Scale"] = curEntropyScale;
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

void GGL::PPOLearner::SaveTo(std::filesystem::path folderPath) {
	models.Save(folderPath);
	if (contrastiveGoalLearner)
		contrastiveGoalLearner->Save(folderPath);
	if (carContrastiveLearner)
		carContrastiveLearner->Save(folderPath);
	if (approachContrastiveLearner)
		approachContrastiveLearner->Save(folderPath);
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
	if (approachContrastiveLearner) {
		approachContrastiveLearner->Load(folderPath, true, true);
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
	if (approachContrastiveLearner)
		approachContrastiveLearner->SetLearningRate(config.contrastiveGoal.criticLR);

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
