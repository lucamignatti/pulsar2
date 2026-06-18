#include "PPOLearner.h"

#include <torch/nn/utils/convert_parameters.h>
#include <torch/nn/utils/clip_grad.h>
#include <torch/csrc/api/include/torch/serialize.h>
#include <public/GigaLearnCPP/Util/AvgTracker.h>

using namespace torch;

GGL::PPOLearner::PPOLearner(int obsSize, int numActions, PPOLearnerConfig _config, Device _device) :
	config(_config), device(_device), obsSize(obsSize), numActions(numActions) {

	if (config.miniBatchSize == 0)
		config.miniBatchSize = config.batchSize;

	if (config.batchSize % config.miniBatchSize != 0)
		RG_ERR_CLOSE("PPOLearner: config.batchSize (" << config.batchSize << ") must be a multiple of config.miniBatchSize (" << config.miniBatchSize << ")");

	MakeModels(true, obsSize, numActions, config.sharedHead, config.policy, config.critic, device, models);

	SetLearningRates(config.policyLR, config.criticLR);

	if (config.contrastiveGoal.enabled) {
		contrastiveGoalLearner = new ContrastiveGoalLearner(obsSize, 8, config.contrastiveGoal, device);
		RG_LOG("Contrastive scoring auxiliary enabled with separate critic encoders");
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
	torch::Tensor* outActions, torch::Tensor* outLogProbs,
	torch::Tensor* outActionProbs) {

	auto probs = InferPolicyProbsFromModels(models, obs, actionMasks, temperature, halfPrec);
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
	InferActionsFromModels(models ? *models : this->models, obs, actionMasks, config.deterministic, config.policyTemperature, config.useHalfPrecision, outActions, outLogProbs, outActionProbs);
}

torch::Tensor GGL::PPOLearner::InferCritic(torch::Tensor obs) {

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
		entropy /= actionMasks.to(torch::kFloat32).sum(-1).log();
	} else {
		entropy /= logf(actionMasks.size(-1));
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

static torch::Tensor GatherQChunks(GGL::ContrastiveGoalLearner* learner, torch::Tensor states, torch::Tensor actionControls, torch::Tensor goals, int64_t chunkSize) {
	std::vector<torch::Tensor> chunks;
	int64_t n = states.size(0);
	for (int64_t start = 0; start < n; start += chunkSize) {
		int64_t stop = RS_MIN(start + chunkSize, n);
		chunks.push_back(learner->ScoreAllActions(
			states.slice(0, start, stop),
			actionControls,
			goals.slice(0, start, stop)
		));
	}
	return torch::cat(chunks, 0);
}

struct CommandSupportStats {
	float supportFraction = 0;
	float jointDistance = 0;
	float positionDistance = 0;
	float velocityDistance = 0;
};

static CommandSupportStats EstimateCommandSupport(torch::Tensor commandedGoals, torch::Tensor achievedGoals, int64_t maxSamples) {
	RG_NO_GRAD;

	CommandSupportStats result;
	int64_t n = commandedGoals.size(0);
	if (n < 4)
		return result;

	int64_t sampleCount = RS_MIN(n, maxSamples);
	torch::Tensor sampleIndices = torch::randperm(n, torch::TensorOptions().dtype(torch::kLong).device(commandedGoals.device())).slice(0, 0, sampleCount);
	torch::Tensor cmd = commandedGoals.index_select(0, sampleIndices);
	torch::Tensor achieved = achievedGoals.index_select(0, sampleIndices);

	torch::Tensor achievedDists = torch::cdist(achieved, achieved, 2);
	torch::Tensor diag = torch::eye(sampleCount, torch::TensorOptions().dtype(torch::kBool).device(commandedGoals.device()));
	achievedDists = achievedDists.masked_fill(diag, 1e9);
	torch::Tensor achievedNN = std::get<0>(achievedDists.min(1));
	float threshold = achievedNN.mean().item<float>() + 2.f * achievedNN.std(false).item<float>();
	if (threshold <= 0 || !std::isfinite(threshold))
		return result;

	torch::Tensor cmdDists = torch::cdist(cmd, achieved, 2);
	auto nearest = cmdDists.min(1);
	torch::Tensor cmdNN = std::get<0>(nearest);
	torch::Tensor nearestIndices = std::get<1>(nearest);
	torch::Tensor nearestAchieved = achieved.index_select(0, nearestIndices);

	result.supportFraction = cmdNN.le(threshold).to(torch::kFloat32).mean().item<float>();
	result.jointDistance = cmdNN.mean().item<float>();
	result.positionDistance = torch::norm(cmd.slice(1, 0, 3) - nearestAchieved.slice(1, 0, 3), 2, 1).mean().item<float>();
	result.velocityDistance = torch::norm(cmd.slice(1, 3, 6) - nearestAchieved.slice(1, 3, 6), 2, 1).mean().item<float>();
	return result;
}

void GGL::PPOLearner::SetDiscreteActionControls(torch::Tensor controls) {
	discreteActionControls = controls.to(device).to(torch::kFloat32);
	if (config.contrastiveGoal.enabled && (discreteActionControls.size(0) != numActions || discreteActionControls.size(1) != 8))
		RG_ERR_CLOSE("Contrastive scoring auxiliary expected action controls shaped [numActions, 8], got " << discreteActionControls.sizes());
}

static float GetAnnealedContrastiveLambda(const GGL::ContrastiveGoalConfig& cfg, uint64_t totalTimesteps) {
	if (cfg.lambdaAnnealSteps == 0)
		return cfg.lambda;

	float progress = RS_CLAMP(totalTimesteps / (float)cfg.lambdaAnnealSteps, 0.f, 1.f);
	return cfg.lambdaStart + (cfg.lambda - cfg.lambdaStart) * progress;
}

static void PrepareContrastivePolicyAdvantages(GGL::PPOLearner* learner, GGL::ExperienceBuffer& experience, GGL::Report& report, uint64_t totalTimesteps) {
	if (!learner->config.contrastiveGoal.enabled)
		return;

	if (!learner->contrastiveGoalLearner)
		RG_ERR_CLOSE("Contrastive goal config is enabled but no contrastive learner exists");
	if (!learner->discreteActionControls.defined())
		RG_ERR_CLOSE("Contrastive goal config is enabled but decoded discrete action controls were not registered");
	if (!experience.data.oldActionProbs.defined() || !experience.data.commandedGoals.defined() || !experience.data.achievedGoals.defined())
		RG_ERR_CLOSE("Contrastive goal config is enabled but rollout CRL tensors are missing");

	RG_NO_GRAD;

	auto device = learner->device;
	auto& cfg = learner->config.contrastiveGoal;

	torch::Tensor nextoAdvRaw = experience.data.advantages.to(device);
	report["Nexto Advantage Mean"] = TensorMean(nextoAdvRaw);
	report["Nexto Advantage Std"] = TensorStd(nextoAdvRaw);

	torch::Tensor states = experience.data.states.to(device);
	torch::Tensor actions = experience.data.actions.to(device).to(torch::kLong);
	torch::Tensor oldActionProbs = experience.data.oldActionProbs.to(device);
	torch::Tensor commandedGoals = experience.data.commandedGoals.to(device);
	torch::Tensor achievedGoals = experience.data.achievedGoals.to(device);

	torch::Tensor qAll = GatherQChunks(learner->contrastiveGoalLearner, states, learner->discreteActionControls, commandedGoals, 4096);
	torch::Tensor qExecuted = qAll.gather(1, actions.unsqueeze(1)).flatten();
	torch::Tensor vCrl = (oldActionProbs * qAll).sum(1);
	torch::Tensor aCrl = qExecuted - vCrl;

	experience.data.crlAdvantages = aCrl.detach().to(experience.data.advantages.device());

	float crlMean = TensorMean(aCrl);
	float crlStd = TensorStd(aCrl);
	report["CRL Advantage Mean"] = crlMean;
	report["CRL Advantage Std"] = crlStd;

	torch::Tensor actionRange = std::get<0>(qAll.max(1)) - std::get<0>(qAll.min(1));
	float commandActionRange = TensorMean(actionRange);
	float commandShuffleDrop = TensorMean(qExecuted - qAll.mean(1));
	report["Action-Score Range at g_cmd"] = commandActionRange;
	report["Action-Shuffle Drop at g_cmd"] = commandShuffleDrop;

	int64_t sensSampleCount = std::min<int64_t>(states.size(0), 2048);
	if (sensSampleCount > 0) {
		torch::Tensor sensIndices = torch::randperm(states.size(0), torch::TensorOptions().dtype(torch::kLong).device(device)).slice(0, 0, sensSampleCount);
		torch::Tensor qAchieved = learner->contrastiveGoalLearner->ScoreAllActions(
			states.index_select(0, sensIndices),
			learner->discreteActionControls,
			achievedGoals.index_select(0, sensIndices)
		);
		torch::Tensor achievedActions = actions.index_select(0, sensIndices);
		torch::Tensor qAchievedExecuted = qAchieved.gather(1, achievedActions.unsqueeze(1)).flatten();
		report["Action-Score Range at Achieved Goals"] = TensorMean(std::get<0>(qAchieved.max(1)) - std::get<0>(qAchieved.min(1)));
		report["Action-Shuffle Drop at Achieved Goals"] = TensorMean(qAchievedExecuted - qAchieved.mean(1));
	}

	CommandSupportStats support = EstimateCommandSupport(commandedGoals, achievedGoals, 2048);
	float supportFraction = support.supportFraction;
	report["Command Support Fraction"] = support.supportFraction;
	report["Joint Support Distance to g_cmd"] = support.jointDistance;
	report["Position Support Distance to g_cmd"] = support.positionDistance;
	report["Velocity Support Distance to g_cmd"] = support.velocityDistance;

	float lambdaEffective = GetAnnealedContrastiveLambda(cfg, totalTimesteps);
	bool varianceGate = crlStd >= cfg.sigmaMin;
	bool actionGate = commandActionRange >= cfg.minCommandActionRange;
	bool supportGate = supportFraction >= cfg.minCommandSupportFraction;

	torch::Tensor nextoNorm = NormalizeAdvantage(nextoAdvRaw, cfg.sigmaFloor);
	torch::Tensor crlNorm = torch::zeros_like(nextoNorm);
	if (crlStd >= cfg.sigmaMin)
		crlNorm = NormalizeAdvantage(aCrl, cfg.sigmaFloor);

	torch::Tensor policyAdvantage = nextoNorm + lambdaEffective * torch::clamp(crlNorm, -3, 3);
	experience.data.advantages = policyAdvantage.detach().to(experience.data.advantages.device());

	report["A Policy Mean"] = TensorMean(policyAdvantage);
	report["A Policy Std"] = TensorStd(policyAdvantage);
	report["CRL Lambda Start"] = cfg.lambdaStart;
	report["CRL Lambda Target"] = cfg.lambda;
	report["CRL Lambda Effective"] = lambdaEffective;
	report["CRL Lambda Anneal Progress"] = cfg.lambdaAnnealSteps == 0 ? 1 : RS_CLAMP(totalTimesteps / (float)cfg.lambdaAnnealSteps, 0.f, 1.f);
	report["CRL Variance Gate"] = varianceGate ? 1 : 0;
	report["CRL Action Gate"] = actionGate ? 1 : 0;
	report["CRL Support Gate"] = supportGate ? 1 : 0;
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
		PrepareContrastivePolicyAdvantages(this, experience, report, totalTimesteps);

		ContrastiveGoalStats crlTrainStats = contrastiveGoalLearner->Train(experience.data, experience.rng);
		if (crlTrainStats.anchorsUsed > 0)
			contrastiveCriticReady = true;

		report["CRL Critic Loss"] = crlTrainStats.loss;
		report["CRL Positive Logit"] = crlTrainStats.positiveLogitMean;
		report["CRL Negative Logit"] = crlTrainStats.negativeLogitMean;
		report["CRL State-Action Embedding Norm"] = crlTrainStats.stateActionEmbeddingNorm;
		report["CRL Goal Embedding Norm"] = crlTrainStats.goalEmbeddingNorm;
		report["Duplicate-Negative Mask Rate"] = crlTrainStats.duplicateNegativeMaskRate;
		report["Eligible-Future Mask Rate"] = crlTrainStats.eligibleFutureMaskRate;
		report["Horizon Immediate Realized"] = crlTrainStats.realizedImmediate;
		report["Horizon Short Realized"] = crlTrainStats.realizedShort;
		report["Horizon Medium Realized"] = crlTrainStats.realizedMedium;
		report["Horizon Long Realized"] = crlTrainStats.realizedLong;
		report["CRL Anchors Used"] = crlTrainStats.anchorsUsed;
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
						probs = InferPolicyProbsFromModels(models, obs, actionMasks, config.policyTemperature, false);
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

					avgRelEntropyLoss += (curEntropy * config.entropyScale) / curPolicyLoss;

					ppoLoss = (policyLoss - entropy * config.entropyScale) * batchSizeRatio;

					if (config.useGuidingPolicy) {
						torch::Tensor guidingProbs;
						{
							RG_NO_GRAD;
							guidingProbs = InferPolicyProbsFromModels(guidingPolicyModels, obs, actionMasks, config.policyTemperature, config.useHalfPrecision);
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
	if (contrastiveGoalLearner)
		contrastiveGoalLearner->Save(folderPath);
}

void GGL::PPOLearner::LoadFrom(std::filesystem::path folderPath)  {
	if (!std::filesystem::is_directory(folderPath))
		RG_ERR_CLOSE("PPOLearner:LoadFrom(): Path " << folderPath << " is not a valid directory");

	models.Load(folderPath, true, true);
	if (contrastiveGoalLearner) {
		bool hasSavedContrastiveCritic =
			std::filesystem::exists(contrastiveGoalLearner->stateActionEncoder.GetSavePath(folderPath)) &&
			std::filesystem::exists(contrastiveGoalLearner->goalEncoder.GetSavePath(folderPath));
		contrastiveGoalLearner->Load(folderPath, true, true);
		contrastiveCriticReady = hasSavedContrastiveCritic;
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
