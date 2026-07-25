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

	MakeModels(true, obsSize, numActions, config.sharedHead, config.policy, config.critic, device, models,
		config.extraPolicyInputs);

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

	// HEADROOM composition critic: twin V-dagger heads reading the SHARED TRUNK
	// (same input width as the main critic head; gradients flow into the trunk —
	// fresh-run co-adaptation). Mirror the critic head's architecture/optimizer.
	if (config.vdagEnabled) {
		RG_ASSERT(models["critic"]); // V-dagger heads mirror the critic head config
		ModelConfig vc = models["critic"]->config;
		models.Add(new Model("vdag1", vc, device));
		models.Add(new Model("vdag2", vc, device));
	}

	if (config.reachability.enabled) {
		int trunkOutSize = config.sharedHead.IsValid() ? config.sharedHead.layerSizes.back() : obsSize;
		reach = new ReachabilityModule(trunkOutSize, numActions, config.reachability, device, models,
			/*makeCarStateHead=*/(config.proposer.enabled && config.proposer.carEnabled)
				|| config.reachability.carStateHead);
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
		MakeModels(false, obsSize, numActions, config.sharedHead, config.policy, config.critic, device, guidingPolicyModels,
			config.extraPolicyInputs);
		guidingPolicyModels.Load(config.guidingPolicyPath, false, false);
	}
}

void GGL::PPOLearner::MakeModels(
	bool makeCritic,
	int obsSize, int numActions,
	PartialModelConfig sharedHeadConfig, PartialModelConfig policyConfig, PartialModelConfig criticConfig,
	torch::Device device,
	ModelSet& outModels,
	int extraPolicyInputs) {

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

	// Ladder wire: the policy head reads trunk ++ wire; only the POLICY widens
	// (critic/trunk untouched - the wire is policy perception, not value input)
	fullPolicyConfig.numInputs += extraPolicyInputs;
	Model* policyModel = new Model("policy", fullPolicyConfig, device);
	policyModel->allowInputExpand = extraPolicyInputs; // pre-wire checkpoints zero-pad at load
	outModels.Add(policyModel);

	if (makeCritic)
		outModels.Add(new Model("critic", fullCriticConfig, device));
}

torch::Tensor GGL::PPOLearner::MinBankDist(torch::Tensor emb, torch::Tensor bank, float dClamp) {
	// d(x, y) = sum_j relu(x_j - y_j): nonneg, d(x,x)=0, triangle inequality,
	// ASYMMETRIC by construction. The [chunk, bank, dims] broadcast is the wire's
	// dominant memory traffic (it runs per collection STEP), so on GPU it runs in
	// bf16 - the spec's own rule: "bank distances are bf16-safe up to the gamma^d
	// map which should run fp32" (distances are O(100)-O(1000) sums of 32 relu
	// terms; bf16's ~2-3 significant digits shift gamma^d negligibly vs the
	// calibration EMA's own noise floor). fp32 on CPU (smoke path, cheap anyway).
	int64_t n = emb.size(0), k = bank.size(0);
	bool bf16 = emb.is_cuda();
	auto e = bf16 ? emb.to(torch::kBFloat16) : emb;
	auto b = bf16 ? bank.to(torch::kBFloat16) : bank;
	auto out = torch::empty({ n }, emb.options().dtype(torch::kFloat32));
	int64_t chunk = RS_MAX((int64_t)1, (int64_t)(1 << 22) / RS_MAX((int64_t)1, k));
	for (int64_t i = 0; i < n; i += chunk) {
		int64_t end = RS_MIN(i + chunk, n);
		auto d = torch::relu(e.slice(0, i, end).unsqueeze(1) - b.unsqueeze(0)).sum(-1); // [c,k]
		out.slice(0, i, end).copy_(std::get<0>(d.min(1)).to(torch::kFloat32));
	}
	return out.clamp_max(dClamp);
}

torch::Tensor GGL::PPOLearner::ComputeWire(ModelSet& models, const LadderWire& lw,
	torch::Tensor rawObs, torch::Tensor trunkOut, bool halfPrec) {
	// The wire is an OBSERVATION: computed no-grad, detached - no gradient may reach
	// the critic, the sensor, or the map through the policy input (Laws 2/4).
	torch::NoGradGuard noGrad;
	RG_ASSERT(models["critic"]); // V_real reads the same generation as the policy
	// Non-const handles to the shared module impls (ModuleHolder shares, forward()
	// is non-const; the guard above makes these reads mutation-free regardless)
	torch::nn::Sequential exp = lw.exp, mapE = lw.mapE, mapF = lw.mapF;
	auto trunkDet = trunkOut.detach();
	auto vReal = models["critic"]->Forward(trunkDet, halfPrec).flatten().to(torch::kFloat32);
	auto vExp = exp->forward(trunkDet).flatten().to(torch::kFloat32);
	torch::Tensor vMet;
	if (lw.banksReady) {
		auto e = mapF->forward(mapE->forward(rawObs.detach().to(torch::kFloat32)));
		auto dg = MinBankDist(e, lw.bankG, lw.dClamp);
		auto dc = MinBankDist(e, lw.bankC, lw.dClamp);
		// gamma^d in fp32 (the one numerically delicate op in the wire)
		vMet = lw.a * torch::pow(lw.gamma, dg) + lw.a2 * torch::pow(lw.gamma, dc) + lw.b;
	} else {
		vMet = vExp; // neutral: gap_PK = 0 until both banks are seeded
	}
	auto gKD = torch::relu(vExp - vReal);
	auto gPK = torch::relu(vMet - vExp);
	return torch::tanh(torch::stack({ vReal, vExp, gKD, vMet, gPK }, -1) / lw.scale);
}

torch::Tensor GGL::PPOLearner::InferPolicyProbsFromModels(
	ModelSet& models,
	torch::Tensor obs, torch::Tensor actionMasks,
	float temperature, bool halfPrec,
	torch::Tensor steerDelta,
	const LadderWire* ladder) {

	actionMasks = actionMasks.to(torch::kBool);

	constexpr float ACTION_MIN_PROB = 1e-11f;
	constexpr float ACTION_DISABLED_LOGIT = -1e10f;

	torch::Tensor rawObs = obs; // kept for the non-finite diagnostic below
	if (models["shared_head"])
		obs = models["shared_head"]->Forward(obs, halfPrec);

	// Steered-practice collection: shift the trunk output along the commitment direction for
	// the masked rows. Model::Forward returns kFloat even on the halfPrec path, so this add is
	// always fp32. Post-trunk only - steering raw obs would be meaningless.
	if (steerDelta.defined()) {
		RG_ASSERT(models["shared_head"]); // the direction lives in trunk-output space
		obs = obs + steerDelta.to(obs.device());
	}

	// Ladder wire: when the policy head carries extra input columns, fill them -
	// live values on the trained-policy collection/learn paths, exact zeros
	// everywhere else (eval/opponents/render/boot probe: both sides of any eval get
	// the same zeros, and the zero-init migration makes zeros the identity input).
	int64_t wireDims = (int64_t)models["policy"]->config.numInputs - obs.size(-1);
	if (wireDims > 0) {
		torch::Tensor wire;
		if (ladder && ladder->active)
			wire = ComputeWire(models, *ladder, rawObs, obs, halfPrec);
		else
			wire = torch::zeros({ obs.size(0), wireDims }, obs.options());
		obs = torch::cat({ obs, wire.detach() }, -1);
	}

	auto logits = models["policy"]->Forward(obs, halfPrec) / temperature;

	// A non-finite logit row would crash multinomial downstream with an opaque assert
	// ("probability tensor contains inf/nan") - identify the SOURCE here instead. Cheap:
	// one fused reduction per inference batch.
	if (!logits.isfinite().all().item<bool>()) {
		bool trunkOutFinite = obs.isfinite().all().item<bool>(); // post-trunk (+delta)
		auto rawBadRows = (~rawObs.isfinite().all(-1)).nonzero().flatten();
		bool policyWFinite = true, trunkWFinite = true;
		for (auto& p : models["policy"]->parameters())
			policyWFinite &= p.isfinite().all().item<bool>();
		if (models["shared_head"])
			for (auto& p : models["shared_head"]->parameters())
				trunkWFinite &= p.isfinite().all().item<bool>();
		std::ostringstream rows;
		for (int64_t i = 0; i < RS_MIN((int64_t)8, rawBadRows.numel()); i++)
			rows << rawBadRows[i].item<int64_t>() << " ";
		RG_ERR_CLOSE("InferPolicyProbsFromModels: non-finite logits ("
			<< (~logits.isfinite().all(-1)).sum().item<int64_t>() << " of " << logits.size(0)
			<< " rows). RAW obs non-finite rows: " << rawBadRows.numel()
			<< " (first: " << rows.str() << "), trunk out finite: " << trunkOutFinite
			<< ", trunk weights finite: " << trunkWFinite
			<< ", policy weights finite: " << policyWFinite
			<< ", steerDelta " << (steerDelta.defined()
				? (steerDelta.isfinite().all().item<bool>() ? "DEFINED-finite" : "DEFINED-NONFINITE")
				: "none")
			<< ", logits absmax " << logits.abs().max().item<float>()
			<< ", halfPrec " << halfPrec);
	}

	auto result = torch::softmax(logits + ACTION_DISABLED_LOGIT * actionMasks.logical_not(), -1);
	return result.view({ -1, models["policy"]->config.numOutputs }).clamp(ACTION_MIN_PROB, 1);
}

void GGL::PPOLearner::InferActionsFromModels(
	ModelSet& models,
	torch::Tensor obs, torch::Tensor actionMasks,
	bool deterministic, float temperature, bool halfPrec,
	torch::Tensor* outActions, torch::Tensor* outLogProbs,
	torch::Tensor steerDelta,
	const LadderWire* ladder) {

	auto probs = InferPolicyProbsFromModels(models, obs, actionMasks, temperature, halfPrec, steerDelta, ladder);

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

void GGL::PPOLearner::SetSteering(const std::array<torch::Tensor, STEER_MODES>& vecsCpu,
	const std::array<float, STEER_MODES>& sigmas,
	const std::array<float, STEER_MODES>& alphas) {
	Model* sharedHead = models["shared_head"];
	RG_ASSERT(sharedHead); // the commitment directions live in trunk-output space
	// Belt and braces vs Model::config quirks: the trunk's output width is its last hidden
	// layer when it has no output layer (the normal case)
	int64_t trunkOut = sharedHead->config.addOutputLayer
		? (int64_t)sharedHead->config.numOutputs
		: (int64_t)sharedHead->config.layerSizes.back();
	for (int m = 0; m < STEER_MODES; m++) {
		if (!vecsCpu[m].defined()) {
			steerVecs[m] = torch::Tensor();
			steerSigmas[m] = steerAlphas[m] = 0;
			continue;
		}
		RG_ASSERT(vecsCpu[m].dim() == 1);
		if (vecsCpu[m].size(0) != trunkOut)
			RG_ERR_CLOSE("SetSteering: mode " << (m + 1) << "v" << (m + 1) << " vector dim "
				<< vecsCpu[m].size(0) << " != trunk output size " << trunkOut
				<< " (direction derived against a different architecture?)");
		// Store unit-norm on the inference device; the delta is alpha * sigma * v per steered row
		steerVecs[m] = (vecsCpu[m] / vecsCpu[m].norm().clamp_min(1e-8f)).to(device).to(torch::kFloat32);
		steerSigmas[m] = sigmas[m];
		steerAlphas[m] = alphas[m];
	}
}

void GGL::PPOLearner::SetSteerGoal(torch::Tensor goal6Cpu, int head) {
	if (!goal6Cpu.defined()) {
		steerGoalOverride = torch::Tensor();
		return;
	}
	RG_ASSERT(goal6Cpu.numel() == 6);
	RG_ASSERT(head >= 0 && head <= 2);
	steerGoalOverride = goal6Cpu.to(device).to(torch::kFloat32).view({ 1, 6 });
	steerGoalOverrideHead = head;
}

void GGL::PPOLearner::InferActions(torch::Tensor obs, torch::Tensor actionMasks, torch::Tensor* outActions, torch::Tensor* outLogProbs, ModelSet* models, torch::Tensor steerRowMask, torch::Tensor steerRowModes, torch::Tensor styleVec, float styleCoef, const LadderWire* ladder) {
	ModelSet& m = models ? *models : this->models;

	bool anyActive = false;
	int64_t trunkOut = 0;
	for (int md = 0; md < STEER_MODES; md++) {
		if (steerVecs[md].defined()) {
			trunkOut = steerVecs[md].size(0);
			anyActive |= steerAlphas[md] != 0;
		}
	}

	torch::Tensor steerDelta = {};
	if (steerRowMask.defined() && anyActive) {
		RG_ASSERT(steerRowModes.defined()); // per-row mode index selects the direction
		auto maskF = steerRowMask.to(device).to(torch::kFloat32); // [n]
		auto modes = steerRowModes.to(device).to(torch::kLong);   // [n]

		// Per-mode coefficient (alpha*sigma; 0 = mode inactive) and direction matrix
		torch::Tensor coef = torch::zeros({ STEER_MODES }, torch::TensorOptions().device(device));
		torch::Tensor vecMat = torch::zeros({ STEER_MODES, trunkOut }, torch::TensorOptions().device(device));
		for (int md = 0; md < STEER_MODES; md++) {
			if (steerVecs[md].defined()) {
				coef[md] = steerAlphas[md] * steerSigmas[md];
				vecMat[md] = steerVecs[md];
			}
		}
		auto rowCoef = coef.index_select(0, modes);               // [n]
		maskF = maskF * (rowCoef != 0).to(torch::kFloat32);

		// Rho-band gate: among the arena-eligible rows, keep only those whose contact-
		// reachability sits in the middle band of the batch's rho distribution - the
		// coin-flip races, by the bot's own estimate. The band is computed PER MODE
		// (mixed-mode quantiles would skew toward the dominant mode); a mode with < 16
		// eligible rows fails CLOSED (steer nothing there - the gate's contract is
		// "steer ONLY in-band rows"). Uses phi/psi from the SAME ModelSet as the policy
		// when present (the pipelined snapshot carries them), so the worker never reads
		// weights that Learn is concurrently updating.
		Model* phi = m["reach_phi"] ? m["reach_phi"] : (reach ? reach->phi : NULL);
		// META override (see SetSteerGoal) redirects the gate to the active emergent
		// cluster's representative goal on its head; otherwise the fixed default.
		int headSel = steerGoalOverride.defined() ? steerGoalOverrideHead : (steerRhoContact ? 0 : 1);
		Model* psiB;
		if (headSel == 0)
			psiB = m["reach_psi_car"] ? m["reach_psi_car"] : (reach ? reach->psiCar : NULL);
		else if (headSel == 1)
			psiB = m["reach_psi_ball"] ? m["reach_psi_ball"] : (reach ? reach->psiBall : NULL);
		else
			psiB = m["reach_psi_carstate"] ? m["reach_psi_carstate"] : (reach ? reach->psiCarState : NULL);
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
				// META override goal, else the fixed defaults: contact (car head, zeros
				// car-local ball = "I am touching it") or scoring (ball head, canonical
				// ball entering the net at speed).
				torch::Tensor goal;
				if (steerGoalOverride.defined())
					goal = steerGoalOverride;
				else if (steerRhoContact)
					goal = torch::zeros({ 1, 6 }).to(device);
				else
					goal = torch::tensor({
						0.f,
						RLGC::CommonValues::BACK_WALL_Y / rc.posScaleY,
						(RLGC::CommonValues::GOAL_HEIGHT * 0.5f) / rc.posScaleZ,
						0.f,
						rc.scoringGoalSpeed / rc.velScale,
						0.f }).to(device).view({ 1, 6 });
				auto g = fnL2(psiB->Forward(goal, false));                          // [1,repr]

				auto rho = sa.matmul(g.squeeze(0)).view({ idx.numel(), steerRhoK })
					.mean(-1) / rc.tau;                                             // [s]

				// Per-mode quantile band over this batch's eligible rows of that mode
				auto modesSel = modes.index_select(0, idx);                         // [s]
				auto inBand = torch::zeros_like(rho);
				for (int md = 0; md < STEER_MODES; md++) {
					auto mIdx = (modesSel == md).nonzero().flatten();
					if (mIdx.numel() < 16)
						continue; // fail closed for this mode
					auto rhoM = rho.index_select(0, mIdx);
					auto lo = rhoM.quantile(steerRhoLo);
					auto hi = rhoM.quantile(steerRhoHi);
					inBand.index_copy_(0, mIdx, ((rhoM >= lo) & (rhoM <= hi)).to(torch::kFloat32));
				}

				auto gated = torch::zeros_like(maskF);
				gated.index_copy_(0, idx, inBand);
				maskF = gated;
				lastRhoGateFrac = inBand.mean().item<float>();
			} else {
				maskF = torch::zeros_like(maskF);
				lastRhoGateFrac = 0;
			}
		}

		steerDelta = (maskF * rowCoef).unsqueeze(-1) * vecMat.index_select(0, modes); // [n,trunkOut]
	}

	// Opponent-side style delta: uniform over all rows of this call ([1,trunkOut] broadcast)
	if (styleVec.defined() && styleCoef != 0) {
		auto style = (styleVec / styleVec.norm().clamp_min(1e-8f))
			.to(device).to(torch::kFloat32).unsqueeze(0) * styleCoef;
		steerDelta = steerDelta.defined() ? steerDelta + style : style;
	}

	InferActionsFromModels(m, obs, actionMasks, config.deterministic, config.policyTemperature, config.useHalfPrecision, outActions, outLogProbs, steerDelta, ladder);
}

torch::Tensor GGL::PPOLearner::InferCritic(torch::Tensor obs) {

	if (models["shared_head"])
		obs = models["shared_head"]->Forward(obs, config.useHalfPrecision);

	return models["critic"]->Forward(obs, config.useHalfPrecision).flatten();
}

torch::Tensor GGL::PPOLearner::InferVdagMin(torch::Tensor obs) {
	RG_NO_GRAD;
	obs = obs.to(device, true);
	if (models["shared_head"])
		obs = models["shared_head"]->Forward(obs, config.useHalfPrecision);
	auto a = models["vdag1"]->Forward(obs, config.useHalfPrecision).flatten().to(torch::kFloat32);
	auto b = models["vdag2"]->Forward(obs, config.useHalfPrecision).flatten().to(torch::kFloat32);
	return torch::minimum(a, b);
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
		// clamp_min(2): a single-valid-action row would divide by log(1) = 0 (its true
		// entropy is ~0 anyway, so log(2) keeps it finite without changing the semantics)
		entropy /= actionMasks.to(torch::kFloat32).sum(-1).clamp_min(2).log();
	} else {
		entropy /= logf(actionMasks.size(-1));
	}

	return entropy.mean();
}

void GGL::PPOLearner::Learn(ExperienceBuffer& experience, Report& report, bool isFirstIteration) {
	auto mseLoss = torch::nn::MSELoss();

	// Ladder wire: a wired policy may NEVER learn against a silently-defaulted wire
	// (a learn-time input differing from the collection-time input biases the PPO
	// ratio - the spec's one hard rule for the re-derivation path). The Learner must
	// set ladderLearn every iteration; consumed-once so staleness can't hide.
	const bool ladderWired = config.extraPolicyInputs > 0;
	if (ladderWired && !ladderLearnSet)
		RG_ERR_CLOSE("PPOLearner::Learn(): policy carries " << config.extraPolicyInputs
			<< " wire inputs but ladderLearn was not set this iteration - refusing to learn "
			"against a defaulted wire (biased ratio). This is a bug in the Learner's "
			"ladder handoff, not a recoverable state.");

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
		avgReachCarStateLoss,
		avgReachLoss,
		avgVdagLoss,
		avgWireColGrad;

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

			// May exceed config.batchSize: with overbatching (default on) the final batch
			// of each epoch is extended with the whole-episode collection overshoot. The
			// minibatch loop below must cover ALL of it, and batchSizeRatio must divide by
			// the ACTUAL size so the accumulated gradient is the exact mean over the
			// (possibly larger) batch — otherwise the overshoot rows are silently dropped.
			const int64_t curBatchSize = batch.states.size(0);

			auto fnRunMinibatch = [&](int64_t start, int64_t stop) {

				float batchSizeRatio = (stop - start) / (float)curBatchSize;

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
						probs = InferPolicyProbsFromModels(models, obs, actionMasks, config.policyTemperature, false,
							{}, ladderWired ? &ladderLearn : NULL);
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
						// batchSizeRatio keeps gradient accumulation identical to a full-batch
						// pass; without it the accumulated guiding gradient scales with the
						// minibatch count (effective strength would depend on miniBatchSize)
						guidingLoss = guidingLoss * config.guidingStrength * batchSizeRatio;
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

				// HEADROOM twin V-dagger: expectile regression vs one-iteration-frozen TD
				// targets (precomputed at learn-prep, riding the buffer). Gradients FLOW
				// into the shared trunk (fresh-run co-adaptation; see PPOLearnerConfig).
				torch::Tensor vdagLoss;
				if (batch.vdagTargets.defined() && models["vdag1"] && models["vdag2"]) {
					auto yv = batch.vdagTargets.slice(0, start, stop).to(device, true, true).flatten();
					torch::Tensor trunkV = models["shared_head"]
						? models["shared_head"]->Forward(obs, false) : obs;
					for (Model* vh : { models["vdag1"], models["vdag2"] }) {
						auto pred = vh->Forward(trunkV, false).flatten().to(torch::kFloat32);
						auto u = yv - pred;
						auto w = torch::where(u > 0,
							torch::full_like(u, config.vdagTau), torch::full_like(u, 1.f - config.vdagTau));
						auto l = (w * u * u).mean() * batchSizeRatio;
						vdagLoss = vdagLoss.defined() ? vdagLoss + l : l;
					}
					avgVdagLoss += vdagLoss.detach().cpu().item<float>();
				}

				// Reachability aux losses (InfoNCE on a subsample; gradient flows into the shared head)
				torch::Tensor reachLoss, carStateLoss;
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

						// Car-state head: same InfoNCE on canonical car-state HER goals, on ALL rows
						// (no move-mask - the car always has a state). Its state-action side is
						// DETACHED (gradient confined to psi_carstate) unless carStateCouple > 0:
						// on 2026-07-14 this term shipped fully coupled while the fresh head was at
						// chance, and its loss (~2x every other aux term combined) churned the
						// shared trunk - Rating slid ~125 across all modes in ~500 iterations with
						// every behavioral guard green. Detached is exactly the offline-validated
						// frozen-phi regime (calibration monotone at ~7x the ball head's margin),
						// so the head keeps its frontier-detector role at zero policy risk. Kept
						// OUT of reachLoss so Reach/Aux Loss keeps meaning "trunk-coupled aux
						// pressure" (its 0.5 baseline is the fix's verification signal on wandb).
						if (reach->psiCarState && batch.carStateHerGoals.defined()) {
							torch::Tensor subCarStateGoals = batch.carStateHerGoals.slice(0, start, stop)
								.index_select(0, subIdx).to(device, true, true);
							float couple = config.reachability.carStateCouple;
							torch::Tensor saCS = couple >= 1.f ? sa
								: (couple <= 0.f ? sa.detach()
									: sa.detach() + (sa - sa.detach()) * couple);
							auto carStateRes = reach->ComputeInfoNCELoss(reach->psiCarState, saCS, subCarStateGoals);
							if (carStateRes.loss.defined()) {
								carStateLoss = carStateRes.loss
									* config.reachability.auxLossWeight * batchSizeRatio;
								avgReachCarStateLoss += carStateLoss.detach().cpu().item<float>();
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
				if (carStateLoss.defined())
					totalLoss = totalLoss.defined() ? totalLoss + carStateLoss : carStateLoss;
				if (vdagLoss.defined())
					totalLoss = totalLoss.defined() ? totalLoss + vdagLoss : vdagLoss;

				if (totalLoss.defined())
					totalLoss.backward();
			};

			
			if (device.is_cpu()) {
				// Just run one minibatch
				fnRunMinibatch(0, curBatchSize);
			} else {
				for (int64_t mbs = 0; mbs < curBatchSize; mbs += config.miniBatchSize) {
					int64_t start = mbs;
					int64_t stop = RS_MIN(start + config.miniBatchSize, curBatchSize);
					fnRunMinibatch(start, stop);
				}
			}

			// Wire-column gradient watch (Law 6 under Muon: orthogonalizing optimizers
			// can amplify near-null input columns into a random walk - this panel is
			// how that would be seen). Read pre-clip: it's the raw pressure.
			if (ladderWired && trainPolicy) {
				auto params = models["policy"]->seq->parameters();
				if (!params.empty() && params[0].dim() == 2 && params[0].grad().defined()) {
					int64_t in = params[0].size(1);
					avgWireColGrad += params[0].grad()
						.slice(1, in - config.extraPolicyInputs, in).norm().item<float>();
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
		// Reported apart from Aux Loss: with carStateCouple=0 this term never touches
		// the trunk, and folding it in is what masked the 2026-07-14 aux-pressure jump
		if (avgReachCarStateLoss.count > 0)
			report["Reach/Car State Loss"] = avgReachCarStateLoss.Get();
		report["Reach/Aux Loss"] = avgReachLoss.Get();
	if (models["vdag1"])
		report["Headroom/Vdag Loss"] = avgVdagLoss.Get();
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

	if (ladderWired) {
		if (avgWireColGrad.count > 0)
			report["Ladder/Wire Col Grad"] = avgWireColGrad.Get();
		ladderLearnSet = false; // consumed: the Learner must re-arm next iteration
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

		// HEADROOM V-dagger twins mirror the CRITIC head's config, so they are value heads and
		// belong with critic/goal_critic above. Policy versions and the render hot-swap only ever
		// ACT (InferPolicyProbsFromModels reads shared_head + policy; eval paths feed zero wire),
		// so cloning these into every version was pure waste: at the 5x1280 critic sizing they are
		// ~16.1M params = ~64MB of GPU *per version*, and maxOldVersions is 32 -> ~2.06GB of the
		// card held by value twins that are never evaluated. That is what was OOM-crashing the
		// residual cold start (2026-07-25, 3 crashes/9h, all CUDA OOM with <200MB free).
		// Old version dirs keep their now-unread vdag1/vdag2 files; Load() only requires the models
		// the template asks for, so this is backward-compatible.
		if (name.rfind("vdag", 0) == 0)
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