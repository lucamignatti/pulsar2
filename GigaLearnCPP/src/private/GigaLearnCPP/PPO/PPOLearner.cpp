#include "PPOLearner.h"

#include <torch/nn/utils/convert_parameters.h>
#include <torch/nn/utils/clip_grad.h>
#include <torch/csrc/api/include/torch/serialize.h>
#include <public/GigaLearnCPP/Util/AvgTracker.h>

using namespace torch;

GGL::PPOLearner::PPOLearner(int obsSize, int numActions, PPOLearnerConfig _config, Device _device) : config(_config), device(_device) {
	curEntropyScale = std::clamp(config.entropyScale, config.minEntropyScale, config.maxEntropyScale);

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

	report["SORS Replay Windows"] = sorsReplay.size();
	if (sorsReplay.empty())
		return;

	int positives = 0;
	float labelTotal = 0;
	for (auto& window : sorsReplay) {
		positives += window.label > 0;
		labelTotal += window.label;
	}
	report["SORS Positive Windows"] = positives;
	report["SORS Avg Label"] = labelTotal / RS_MAX(1, (int)sorsReplay.size());

	if (sorsTrainCalls++ < config.sorsWarmupIters || sorsReplay.size() < 2)
		return;

	auto* sors = dynamic_cast<SORSRewardModel*>(models["sors_reward"]);
	RG_ASSERT(sors);

	MutAvgTracker avgLoss, avgPredReturn, avgAcc;
	int trainedPairs = 0;
	int attempts = 0;
	int maxAttempts = config.sorsTrainPairs * 8;

	while (trainedPairs < config.sorsTrainPairs && attempts++ < maxAttempts) {
		int idxA = Math::RandInt(0, (int)sorsReplay.size());
		int idxB = Math::RandInt(0, (int)sorsReplay.size());
		if (idxA == idxB)
			continue;

		auto& a = sorsReplay[idxA];
		auto& b = sorsReplay[idxB];
		float delta = a.label - b.label;
		if (fabsf(delta) < config.sorsMinLabelDelta)
			continue;

		auto& hi = delta > 0 ? a : b;
		auto& lo = delta > 0 ? b : a;

		auto hiObs = torch::tensor(hi.states).reshape({ hi.length, sors->obs_dim }).to(device, true, true);
		auto hiActs = torch::tensor(hi.actionComps).reshape({ hi.length, sors->action_dim }).to(device, true, true);
		auto loObs = torch::tensor(lo.states).reshape({ lo.length, sors->obs_dim }).to(device, true, true);
		auto loActs = torch::tensor(lo.actionComps).reshape({ lo.length, sors->action_dim }).to(device, true, true);

		auto hiReturn = sors->Forward(hiObs, hiActs).sum();
		auto loReturn = sors->Forward(loObs, loActs).sum();
		auto loss = -torch::log(torch::sigmoid(hiReturn - loReturn) + 1e-8f);
		loss.backward();

		float hiVal = hiReturn.detach().cpu().item<float>();
		float loVal = loReturn.detach().cpu().item<float>();
		avgLoss += loss.detach().cpu().item<float>();
		avgPredReturn += hiVal;
		avgAcc += hiVal > loVal ? 1.0f : 0.0f;
		trainedPairs++;
	}

	if (trainedPairs > 0) {
		nn::utils::clip_grad_norm_(sors->parameters(), 0.5f);
		sors->StepOptim();
		report["SORS Loss"] = avgLoss.Get();
		report["SORS Pair Accuracy"] = avgAcc.Get();
		report["SORS Pred Window Return"] = avgPredReturn.Get();
	}
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

float GGL::PPOLearner::GetEntropyScale() const {
	return config.adaptiveEntropy ? curEntropyScale : config.entropyScale;
}

void GGL::PPOLearner::Learn(ExperienceBuffer& experience, Report& report, bool isFirstIteration) {
	auto mseLoss = torch::nn::MSELoss();

	MutAvgTracker
		avgEntropy,
		avgDivergence,
		avgPolicyLoss,
		avgRelEntropyLoss,
		avgCriticLoss,
		avgGuidingLoss,
		avgRatio,
		avgClip,
		avgInfoNCELoss,
		avgGcrlAdv;

	// Save parameters first
	auto policyBefore = models["policy"]->CopyParams();
	auto criticBefore = models["critic"]->CopyParams();

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
			auto batchActionComps = batch.actionComps;
			auto batchFutureGoals = batch.futureGoals;
			auto batchCarFutureGoals = batch.carFutureGoals;

			auto fnRunMinibatch = [&](int start, int stop) {

				float batchSizeRatio = (stop - start) / (float)config.batchSize;

				// Send everything to the device and enforce correct shapes
				auto acts = batchActs.slice(0, start, stop).to(device, true, true);
				auto obs = batchObs.slice(0, start, stop).to(device, true, true);
				auto actionMasks = batchActionMasks.slice(0, start, stop).to(device, true, true);
				
				auto advantages = batchAdvantages.slice(0, start, stop).to(device, true, true);
				auto oldProbs = batchOldProbs.slice(0, start, stop).to(device, true, true);
				auto targetValues = batchTargetValues.slice(0, start, stop).to(device, true, true);

				torch::Tensor actionComps, futureGoals, carFutureGoals;
				if (trainGCRL) {
					actionComps    = batchActionComps.slice(0, start, stop).to(device, true, true);
					futureGoals    = batchFutureGoals.slice(0, start, stop).to(device, true, true);
					carFutureGoals = batchCarFutureGoals.slice(0, start, stop).to(device, true, true);
				}

				// Shared encoder forward once; policy, value critic and GCRL critics read it
				torch::Tensor features = obs;
				if (m_sharedHead)
					features = m_sharedHead->Forward(obs, false);

				// ── GCRL "game sense" advantage, blended onto the reward-driven advantage ──
				// Both channels are unit-normalized so gcrlAdvScale is a meaningful weight.
				if (trainGCRL) {
					RG_NO_GRAD;
					auto q_goal = gc_goal->score_q(features, actionComps, futureGoals);
					auto q_anti = gc_anti->score_q(features, actionComps, futureGoals);
					auto q_car  = gc_car->score_q(features, actionComps, carFutureGoals);

					auto adv_goal = (q_goal - q_goal.mean()) / (q_goal.std() + 1e-8f);
					auto adv_anti = (q_anti - q_anti.mean()) / (q_anti.std() + 1e-8f);
					auto adv_car  = (q_car  - q_car.mean())  / (q_car.std()  + 1e-8f);

					// goal pursuit, "anti" pessimism (double-critic), car positioning
					auto adv_gcrl = adv_goal - config.gcrlAntiScale * adv_anti + config.gcrlCarScale * adv_car;
					avgGcrlAdv += adv_gcrl.abs().mean().item<float>();

					auto adv_rew = (advantages - advantages.mean()) / (advantages.std() + 1e-8f);
					advantages = adv_rew + config.gcrlAdvScale * adv_gcrl;
				}

				torch::Tensor probs, logProbs, entropy, ratio, clipped, policyLoss, ppoLoss;
				if (trainPolicy) {

					// Get policy log probs and entropy (from the shared features)
					float curEntropy;
					{
						auto maskBool = actionMasks.to(torch::kBool);
						auto logits = m_policy->Forward(features, false) / config.policyTemperature;
						probs = torch::softmax(logits + (-1e10f) * maskBool.logical_not(), -1)
							.view({ -1, m_policy->config.numOutputs }).clamp(1e-11f, 1);
						logProbs = probs.log().gather(-1, acts.unsqueeze(-1));
						entropy = ComputeEntropy(probs, actionMasks, config.maskEntropy);
						curEntropy = entropy.detach().cpu().item<float>();
						avgEntropy += curEntropy;
					}

					logProbs = logProbs.view_as(oldProbs);
					const float entropyScale = GetEntropyScale();

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

					avgRelEntropyLoss += (curEntropy * entropyScale) / curPolicyLoss;

					ppoLoss = (policyLoss - entropy * entropyScale) * batchSizeRatio;

					if (config.adaptiveEntropy) {
						curEntropyScale += config.adaptiveEntropyLR * (config.targetEntropy - curEntropy);
						curEntropyScale = std::clamp(curEntropyScale, config.minEntropyScale, config.maxEntropyScale);
					}

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
					auto vals = m_critic->Forward(features, false).flatten();

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

						auto il_goal = gc_goal->infonce_loss(sub_obs, sub_actions, sub_goals);
						auto il_anti = gc_anti->infonce_loss(sub_obs, sub_actions, sub_goals);
						auto il_car  = gc_car->infonce_loss(sub_obs, sub_actions, sub_car_goals);
						infoNCELoss = (il_goal + il_anti + il_car) * batchSizeRatio;
						avgInfoNCELoss += infoNCELoss.detach().cpu().item<float>();
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
			};

			
			if (device.is_cpu()) {
				// Just run one minibatch
				fnRunMinibatch(0, config.batchSize);
			} else {
				for (int mbs = 0; mbs < config.batchSize; mbs += config.miniBatchSize) {
					int start = mbs;
					int stop = start + config.miniBatchSize;
					fnRunMinibatch(start, stop);
				}
			}

			if (trainPolicy)
				nn::utils::clip_grad_norm_(models["policy"]->parameters(), 0.5f);
			if (trainCritic)
				nn::utils::clip_grad_norm_(models["critic"]->parameters(), 0.5f);

			if (trainSharedHead)
				nn::utils::clip_grad_norm_(models["shared_head"]->parameters(), 0.5f);

			if (trainGCRL) {
				nn::utils::clip_grad_norm_(gc_goal->parameters(), 0.5f);
				nn::utils::clip_grad_norm_(gc_anti->parameters(), 0.5f);
				nn::utils::clip_grad_norm_(gc_car->parameters(), 0.5f);
			}

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
	report["Entropy Scale"] = GetEntropyScale();
	if (config.adaptiveEntropy)
		report["Target Entropy"] = config.targetEntropy;
	report["Mean KL Divergence"] = avgDivergence.Get();
	if (!isFirstIteration) {
		// These metrics give bad data on the first iteration, which will mess up graph scaling
		// So we'll just skip them for the first iteration
		report["Policy Loss"] = avgPolicyLoss.Get();
		report["Policy Relative Entropy Loss"] = avgRelEntropyLoss.Get();
		report["Critic Loss"] = avgCriticLoss.Get();

		if (trainGCRL) {
			report["InfoNCE Loss"] = avgInfoNCELoss.Get();
			report["GCRL/Avg Advantage"] = avgGcrlAdv.Get();
		}

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
}

void GGL::PPOLearner::LoadFrom(std::filesystem::path folderPath)  {
	if (!std::filesystem::is_directory(folderPath))
		RG_ERR_CLOSE("PPOLearner:LoadFrom(): Path " << folderPath << " is not a valid directory");

	models.Load(folderPath, true, true);

	SetLearningRates(config.policyLR, config.criticLR);
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
