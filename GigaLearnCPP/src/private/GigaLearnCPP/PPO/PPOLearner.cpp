#include "PPOLearner.h"

#include <torch/nn/utils/convert_parameters.h>
#include <torch/nn/utils/clip_grad.h>
#include <torch/csrc/api/include/torch/serialize.h>
#include <public/GigaLearnCPP/Util/AvgTracker.h>
#include <RLGymCPP/CommonValues.h>

using namespace torch;

GGL::PPOLearner::PPOLearner(int obsSize, int numActions, PPOLearnerConfig _config, Device _device) : config(_config), device(_device) {

	if (config.miniBatchSize == 0)
		config.miniBatchSize = config.batchSize;

	if (config.batchSize % config.miniBatchSize != 0)
		RG_ERR_CLOSE("PPOLearner: config.batchSize (" << config.batchSize << ") must be a multiple of config.miniBatchSize (" << config.miniBatchSize << ")");

	MakeModels(true, obsSize, numActions, config.sharedHead, config.policy, config.critic, device, models);

	// Secondary goal-only critic: a fully independent net (raw obs in, no shared trunk) so its
	// gradients can't touch the proven policy/critic path. Lives in `models` so it checkpoints and
	// steps with everything else; excluded from GetPolicyModels() like the main critic.
	if (config.goalCritic.enabled) {
		RG_ASSERT(config.goalCritic.model.IsValid());
		RG_ASSERT(config.goalCritic.beta >= 0); // a negative blend would train AWAY from goals
		ModelConfig gcConfig = config.goalCritic.model;
		gcConfig.numInputs = obsSize;
		gcConfig.numOutputs = 1;
		models.Add(new Model("goal_critic", gcConfig, device));
	}

	if (config.reachability.enabled) {
		int trunkOutSize = config.sharedHead.IsValid() ? config.sharedHead.layerSizes.back() : obsSize;
		reach = new ReachabilityModule(trunkOutSize, numActions, config.reachability, device, models,
			/*makeCarStateHead=*/config.proposer.enabled && config.proposer.carEnabled);
	}

	if (config.proposer.enabled) {
		if (!config.reachability.enabled)
			RG_ERR_CLOSE("PPOLearner: config.proposer.enabled requires config.reachability.enabled (the proposer reuses the reachability phi/psiBall goal space)");
		int trunkOutSize = config.sharedHead.IsValid() ? config.sharedHead.layerSizes.back() : obsSize;
		proposer = new ProposerModule(trunkOutSize, config.proposer, device, models);

		// Car proposer = a SECOND ProposerModule instance (same 6D goal space, same A^(N) weights,
		// same unroll/train machinery) with its own delta net, proposing canonical car states.
		if (config.proposer.carEnabled)
			proposerCar = new ProposerModule(trunkOutSize, config.proposer, device, models, "proposer_car_delta");
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
	float temperature, bool halfPrec,
	torch::Tensor steerDelta) {

	actionMasks = actionMasks.to(torch::kBool);

	constexpr float ACTION_MIN_PROB = 1e-11f;
	constexpr float ACTION_DISABLED_LOGIT = -1e10f;

	if (models["shared_head"])
		obs = models["shared_head"]->Forward(obs, halfPrec);

	// Steered-practice collection: shift the trunk output along the commitment direction for
	// the masked rows. Model::Forward returns kFloat even on the halfPrec path, so this add is
	// always fp32. Post-trunk only - steering raw obs would be meaningless.
	if (steerDelta.defined()) {
		RG_ASSERT(models["shared_head"]); // the direction lives in trunk-output space
		obs = obs + steerDelta.to(obs.device());
	}

	auto logits = models["policy"]->Forward(obs, halfPrec) / temperature;

	auto result = torch::softmax(logits + ACTION_DISABLED_LOGIT * actionMasks.logical_not(), -1);
	return result.view({ -1, models["policy"]->config.numOutputs }).clamp(ACTION_MIN_PROB, 1);
}

void GGL::PPOLearner::InferActionsFromModels(
	ModelSet& models,
	torch::Tensor obs, torch::Tensor actionMasks,
	bool deterministic, float temperature, bool halfPrec,
	torch::Tensor* outActions, torch::Tensor* outLogProbs,
	torch::Tensor steerDelta) {

	auto probs = InferPolicyProbsFromModels(models, obs, actionMasks, temperature, halfPrec, steerDelta);

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

void GGL::PPOLearner::SetSteering(torch::Tensor vecCpu, float sigma, float alpha) {
	RG_ASSERT(vecCpu.dim() == 1);
	Model* sharedHead = models["shared_head"];
	RG_ASSERT(sharedHead); // the commitment direction lives in trunk-output space
	// Belt and braces vs Model::config quirks: the trunk's output width is its last hidden
	// layer when it has no output layer (the normal case)
	int64_t trunkOut = sharedHead->config.addOutputLayer
		? (int64_t)sharedHead->config.numOutputs
		: (int64_t)sharedHead->config.layerSizes.back();
	if (vecCpu.size(0) != trunkOut)
		RG_ERR_CLOSE("SetSteering: vector dim " << vecCpu.size(0)
			<< " != trunk output size " << trunkOut
			<< " (direction derived against a different architecture?)");
	// Store unit-norm on the inference device; the delta is alpha * sigma * v per steered row
	steerVec = (vecCpu / vecCpu.norm().clamp_min(1e-8f)).to(device).to(torch::kFloat32);
	steerSigma = sigma;
	steerAlpha = alpha;
}

void GGL::PPOLearner::InferActions(torch::Tensor obs, torch::Tensor actionMasks, torch::Tensor* outActions, torch::Tensor* outLogProbs, ModelSet* models, torch::Tensor steerRowMask) {
	ModelSet& m = models ? *models : this->models;

	torch::Tensor steerDelta = {};
	if (steerRowMask.defined() && steerVec.defined() && steerAlpha != 0) {
		auto maskF = steerRowMask.to(device).to(torch::kFloat32); // [n]

		// Rho-band gate: among the arena-eligible rows, keep only those whose scoring-
		// reachability sits in the batch's middle band - hard-but-plausible plays toward
		// the net, by the bot's own estimate. Uses phi/psiBall from the SAME ModelSet as
		// the policy when present (the pipelined snapshot carries them), so the worker
		// never reads weights that Learn is concurrently updating.
		Model* phi = m["reach_phi"] ? m["reach_phi"] : (reach ? reach->phi : NULL);
		Model* psiB;
		if (steerRhoContact)
			psiB = m["reach_psi_car"] ? m["reach_psi_car"] : (reach ? reach->psiCar : NULL);
		else
			psiB = m["reach_psi_ball"] ? m["reach_psi_ball"] : (reach ? reach->psiBall : NULL);
		if (steerRhoGate && phi && psiB && m["shared_head"]) {
			RG_NO_GRAD;
			auto idx = maskF.nonzero().flatten();
			if (idx.numel() >= 16) {
				auto obsSel = obs.index_select(0, idx);
				auto maskSel = actionMasks.index_select(0, idx).to(torch::kFloat32).clamp_min(1e-9f);
				torch::Tensor trunk = m["shared_head"]->Forward(obsSel, false);

				// K uniform valid actions per row (capability read, not policy read)
				auto acts = torch::multinomial(maskSel, steerRhoK, true);           // [s,K]
				auto trunkRep = trunk.repeat_interleave(steerRhoK, 0);              // [s*K,trunk]
				auto oneHot = torch::one_hot(acts.flatten(), maskSel.size(1)).to(torch::kFloat32);
				auto fnL2 = [](torch::Tensor t) { return t / t.norm(2, -1, true).clamp_min(1e-6f); };
				auto sa = fnL2(phi->Forward(torch::cat({ trunkRep, oneHot }, -1), false)); // [s*K,repr]

				const auto& rc = config.reachability;
				// Contact goal (car head): all-zeros car-local ball = "I am touching it".
				// Scoring goal (ball head): canonical ball entering the net at speed.
				torch::Tensor goal = steerRhoContact
					? torch::zeros({ 1, 6 }).to(device)
					: torch::tensor({
						0.f,
						RLGC::CommonValues::BACK_WALL_Y / rc.posScaleY,
						(RLGC::CommonValues::GOAL_HEIGHT * 0.5f) / rc.posScaleZ,
						0.f,
						rc.scoringGoalSpeed / rc.velScale,
						0.f }).to(device).view({ 1, 6 });
				auto g = fnL2(psiB->Forward(goal, false));                          // [1,repr]

				auto rho = sa.matmul(g.squeeze(0)).view({ idx.numel(), steerRhoK })
					.mean(-1) / rc.tau;                                             // [s]
				auto lo = rho.quantile(steerRhoLo);
				auto hi = rho.quantile(steerRhoHi);
				auto inBand = ((rho >= lo) & (rho <= hi)).to(torch::kFloat32);      // [s]

				auto gated = torch::zeros_like(maskF);
				gated.index_copy_(0, idx, inBand);
				maskF = gated;
				lastRhoGateFrac = inBand.mean().item<float>();
			}
		}

		steerDelta = maskF.unsqueeze(-1) * (steerAlpha * steerSigma) * steerVec.unsqueeze(0); // [n,trunkOut]
	}
	InferActionsFromModels(m, obs, actionMasks, config.deterministic, config.policyTemperature, config.useHalfPrecision, outActions, outLogProbs, steerDelta);
}

torch::Tensor GGL::PPOLearner::InferCritic(torch::Tensor obs) {

	if (models["shared_head"])
		obs = models["shared_head"]->Forward(obs, config.useHalfPrecision);

	return models["critic"]->Forward(obs, config.useHalfPrecision).flatten();
}

torch::Tensor GGL::PPOLearner::InferGoalCritic(torch::Tensor obs) {
	// Independent net: raw obs in, NO shared trunk (by design — zero gradient interference)
	return models["goal_critic"]->Forward(obs, config.useHalfPrecision).flatten();
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

void GGL::PPOLearner::Learn(ExperienceBuffer& experience, Report& report, bool isFirstIteration) {
	auto mseLoss = torch::nn::MSELoss();

	MutAvgTracker
		avgEntropy,
		avgDivergence,
		avgPolicyLoss,
		avgRelEntropyLoss,
		avgCriticLoss,
		avgGoalCriticLoss,
		avgGuidingLoss,
		avgRatio,
		avgClip,
		avgReachCarAcc,
		avgReachBallAcc,
		avgReachCarStateAcc,
		avgReachLoss;

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
			auto batchGoalTargetValues = batch.goalTargetValues; // undefined unless goalCritic.enabled
			auto batchPracticeMask = batch.practiceMask; // undefined unless steering enabled
			auto batchAdvantages = batch.advantages;

			auto fnRunMinibatch = [&](int start, int stop) {

				float batchSizeRatio = (stop - start) / (float)config.batchSize;

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

				// Steered-practice rows: the MAIN critic MUST train on them. Excluding them
				// (the first version of this code) left V(s) pricing frontier states at full
				// match value while practice episodes end true-terminal with only the small
				// attempt pay - GAE then charged ~ -V(s_end) as a phantom penalty smeared over
				// every (short) practice episode, and the policy rapidly unlearned ball
				// engagement (live Elo freefall, 2026-07-12). With practice rows included, V
				// learns the honest practice/match mixture for aliased states; the residual
				// mixture bias on match values is the far smaller error (escalation path if it
				// ever matters: a dedicated practice-value head).
				// The GOAL critic stays excluded: its +/-1 channel is structurally absent in
				// resolution-terminated episodes, so those rows would only teach it "0 here".
				torch::Tensor keepRow = {};
				if (batchPracticeMask.defined())
					keepRow = 1.0f - batchPracticeMask.slice(0, start, stop).to(device, true, true);

				auto fnMaskedMSE = [&](torch::Tensor pred, torch::Tensor target) {
					if (!keepRow.defined())
						return mseLoss(pred, target);
					auto keep = keepRow.view_as(target);
					return ((pred - target).square() * keep).sum() / keep.sum().clamp_min(1);
				};

				torch::Tensor criticLoss;
				if (trainCritic) {
					auto vals = InferCritic(obs);

					// Compute value loss (ALL rows - see comment above)
					vals = vals.view_as(targetValues);
					criticLoss = mseLoss(vals, targetValues) * batchSizeRatio;
					avgCriticLoss += criticLoss.detach().cpu().item<float>();
				}

				// Secondary goal-only critic: plain value regression on its own channel. Fully
				// independent net, so this gradient touches nothing else.
				torch::Tensor goalCriticLoss;
				if (batchGoalTargetValues.defined() && models["goal_critic"]) {
					auto goalTargets = batchGoalTargetValues.slice(0, start, stop).to(device, true, true);
					auto goalVals = InferGoalCritic(obs).view_as(goalTargets);
					goalCriticLoss = fnMaskedMSE(goalVals, goalTargets) * batchSizeRatio;
					avgGoalCriticLoss += goalCriticLoss.detach().cpu().item<float>();
				}

				// Reachability aux losses (InfoNCE on a subsample; gradient flows into the shared head)
				torch::Tensor reachLoss;
				if (reach && batch.carHerGoals.defined()) {
					int64_t mbRows = stop - start;
					int64_t sub = RS_MIN((int64_t)config.reachability.infoSubSample, mbRows);
					if (sub > 1) {
						torch::Tensor subIdx = torch::randperm(mbRows, TensorOptions().dtype(kLong)).slice(0, 0, sub);
						torch::Tensor subIdxDev = subIdx.to(device);

						torch::Tensor subObs = obs.index_select(0, subIdxDev);
						torch::Tensor subActs = acts.index_select(0, subIdxDev);
						torch::Tensor subCarGoals = batch.carHerGoals.slice(0, start, stop)
							.index_select(0, subIdx).to(device, true, true);

						torch::Tensor trunkOut = models["shared_head"]
							? models["shared_head"]->Forward(subObs, false)
							: subObs;
						torch::Tensor sa = reach->EncodeStateAction(trunkOut, subActs);

						auto carRes = reach->ComputeInfoNCELoss(reach->psiCar, sa, subCarGoals);
						if (carRes.loss.defined()) {
							reachLoss = carRes.loss;
							avgReachCarAcc += carRes.categoricalAccuracy;
						}

						// The ball head only trains on rows from episodes where the ball moved
						if (batch.ballHerGoals.defined() && batch.ballMovedMask.defined()) {
							torch::Tensor subMoved = batch.ballMovedMask.slice(0, start, stop).index_select(0, subIdx);
							torch::Tensor movedIdx = subMoved.nonzero().flatten();
							if (movedIdx.size(0) > 1) {
								torch::Tensor subBallGoals = batch.ballHerGoals.slice(0, start, stop)
									.index_select(0, subIdx).index_select(0, movedIdx).to(device, true, true);
								auto ballRes = reach->ComputeInfoNCELoss(
									reach->psiBall, sa.index_select(0, movedIdx.to(device)), subBallGoals);
								if (ballRes.loss.defined()) {
									reachLoss = reachLoss.defined() ? reachLoss + ballRes.loss : ballRes.loss;

									// Tiny subsets give chance-inflated accuracy (chance = 1/rows,
									// e.g. 50% at 2 rows); don't let them feed the gate anneal
									if (movedIdx.size(0) >= 32)
										avgReachBallAcc += ballRes.categoricalAccuracy;
								}
							}
						}

						// Car-state head (car proposer): same InfoNCE on canonical car-state HER goals.
						// Trained on ALL rows (no move-mask - the car always has a state).
						if (reach->psiCarState && batch.carStateHerGoals.defined()) {
							torch::Tensor subCarStateGoals = batch.carStateHerGoals.slice(0, start, stop)
								.index_select(0, subIdx).to(device, true, true);
							auto carStateRes = reach->ComputeInfoNCELoss(reach->psiCarState, sa, subCarStateGoals);
							if (carStateRes.loss.defined()) {
								reachLoss = reachLoss.defined() ? reachLoss + carStateRes.loss : carStateRes.loss;
								avgReachCarStateAcc += carStateRes.categoricalAccuracy;
							}
						}

						if (reachLoss.defined()) {
							reachLoss = (reachLoss + reach->StateActionVarPenalty(sa))
								* config.reachability.auxLossWeight * batchSizeRatio;
							avgReachLoss += reachLoss.detach().cpu().item<float>();
						}
					}
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

				// One combined backward so policy, critic and reachability gradients
				// accumulate into the shared head together
				torch::Tensor totalLoss;
				if (trainPolicy)
					totalLoss = ppoLoss;
				if (trainCritic)
					totalLoss = totalLoss.defined() ? totalLoss + criticLoss : criticLoss;
				if (goalCriticLoss.defined())
					totalLoss = totalLoss.defined() ? totalLoss + goalCriticLoss : goalCriticLoss;
				if (reachLoss.defined())
					totalLoss = totalLoss.defined() ? totalLoss + reachLoss : reachLoss;

				if (totalLoss.defined())
					totalLoss.backward();
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

			if (models["goal_critic"])
				nn::utils::clip_grad_norm_(models["goal_critic"]->parameters(), 0.5f);

			if (reach) {
				nn::utils::clip_grad_norm_(reach->phi->parameters(), 0.5f);
				nn::utils::clip_grad_norm_(reach->psiCar->parameters(), 0.5f);
				nn::utils::clip_grad_norm_(reach->psiBall->parameters(), 0.5f);
				if (reach->psiCarState)
					nn::utils::clip_grad_norm_(reach->psiCarState->parameters(), 0.5f);
			}

			models.StepOptims();
		}
	}

	// Compute magnitude of updates made to the policy and value estimator
	auto policyAfter = models["policy"]->CopyParams();
	auto criticAfter = models["critic"]->CopyParams();

	float policyUpdateMagnitude = (policyBefore - policyAfter).norm().item<float>();
	float criticUpdateMagnitude = (criticBefore - criticAfter).norm().item<float>();

	if (reach) {
		float carAcc = avgReachCarAcc.Get();
		float ballAcc = avgReachBallAcc.Get();
		// The ball head may not have trained at all (no ball movement yet); use the car head alone then
		lastReachAccuracy = (avgReachBallAcc.count > 0) ? RS_MIN(carAcc, ballAcc) : carAcc;
		if (avgReachCarAcc.count > 0)
			lastReachTrained = true;

		report["Reach/Car Accuracy"] = carAcc;
		report["Reach/Ball Accuracy"] = ballAcc;
		if (avgReachCarStateAcc.count > 0)
			report["Reach/Car State Accuracy"] = avgReachCarStateAcc.Get();
		report["Reach/Aux Loss"] = avgReachLoss.Get();
	}

	// Assemble and return report
	report["Policy Entropy"] = avgEntropy.Get();
	report["Mean KL Divergence"] = avgDivergence.Get();
	if (!isFirstIteration) {
		// These metrics give bad data on the first iteration, which will mess up graph scaling
		// So we'll just skip them for the first iteration
		report["Policy Loss"] = avgPolicyLoss.Get();
		report["Policy Relative Entropy Loss"] = avgRelEntropyLoss.Get();
		report["Critic Loss"] = avgCriticLoss.Get();
		if (avgGoalCriticLoss.count > 0)
			report["GoalCritic/Loss"] = avgGoalCriticLoss.Get();

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

	if (models["goal_critic"])
		models["goal_critic"]->SetOptimLR(config.goalCritic.lr);

	RG_LOG("PPOLearner: " << RS_STR(std::scientific << "Set learning rate to [" << policyLR << ", " << criticLR << "]"));
}

GGL::ModelSet GGL::PPOLearner::GetPolicyModels() {
	ModelSet result = {};
	for (Model* model : models) {
		std::string name = model->modelName;
		if (name == "critic" || name == "goal_critic")
			continue;

		// Reachability heads are training-time-only; old policy versions don't carry them
		if (name.rfind("reach_", 0) == 0)
			continue;

		// Same for the deliberate-practice proposer - it's training-time-only and old policy
		// versions are loaded with allowNotExist=false, which would hard-fail on it
		if (name.rfind("proposer", 0) == 0)
			continue;

		result.Add(model);
	}
	return result;
}