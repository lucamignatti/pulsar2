#include "PPOLearner.h"

#include <torch/nn/utils/convert_parameters.h>
#include <torch/nn/utils/clip_grad.h>
#include <torch/csrc/api/include/torch/serialize.h>
#include <public/GigaLearnCPP/Util/AvgTracker.h>
#include <RLGymCPP/CommonValues.h>
#include "../Util/Plasticity.h"

using namespace torch;

GGL::PPOLearner::PPOLearner(int obsSize, int numActions, PPOLearnerConfig _config, Device _device) : config(_config), device(_device) {

	if (config.miniBatchSize == 0)
		config.miniBatchSize = config.batchSize;

	if (config.batchSize % config.miniBatchSize != 0)
		RG_ERR_CLOSE("PPOLearner: config.batchSize (" << config.batchSize << ") must be a multiple of config.miniBatchSize (" << config.miniBatchSize << ")");

	numActionsCached = numActions; obsSizeCached = obsSize;
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

	// HEADROOM composition critic: twin V-dagger heads reading the SHARED TRUNK
	// (same input width as the main critic head; gradients flow into the trunk —
	// fresh-run co-adaptation). Mirror the critic head's architecture/optimizer.
	if (config.vdagEnabled) {
		RG_ASSERT(models["critic"]); // V-dagger heads mirror the critic head config
		ModelConfig vc = models["critic"]->config;
		models.Add(new Model("vdag1", vc, device));
		models.Add(new Model("vdag2", vc, device));
		// THEORY: twin optimistic reward models (same head shape, own outputs)
		if (config.vdagTheoryEnabled) {
			models.Add(new Model("rhat1", vc, device));
			models.Add(new Model("rhat2", vc, device));
		}
		// IMPLICIT WORLD MODEL. Dynamics live in OBS space (not trunk space) so their
		// targets stay well-defined as the trunk drifts under training, and an imagined
		// state can be pushed straight back through shared_head into the existing heads.
		if (config.vdagWmEnabled && config.vdagTheoryEnabled) {
			ModelConfig dynCfg = models["critic"]->config;
			dynCfg.addResiduals = false;
			dynCfg.layerSizes = config.vdagWmLayers;
			dynCfg.numInputs = obsSize + numActions;   // state + action one-hot
			dynCfg.numOutputs = obsSize;               // predicted DELTA
			ModelConfig imCfg = models["critic"]->config;
			imCfg.addResiduals = false;
			imCfg.layerSizes = config.vdagWmLayers;
			imCfg.numOutputs = 1;
			for (const char* n : { "wm_dyn1", "wm_dyn2" }) {
				Model* m = new Model(n, dynCfg, device);
				m->groupStepExempt = true;   // stepped by TrainWorldModel, not the PPO loop
				models.Add(m);
			}
			for (const char* n : { "wm_v1", "wm_v2" }) {
				Model* m = new Model(n, imCfg, device);
				m->groupStepExempt = true;   // stepped by TrainImagValue
				models.Add(m);
			}
		}
	}

	if (config.reachability.enabled) {
		int trunkOutSize = config.sharedHead.IsValid() ? config.sharedHead.layerSizes.back() : obsSize;
		reach = new ReachabilityModule(trunkOutSize, numActions, config.reachability, device, models,
			/*makeCarStateHead=*/config.reachability.carStateHead);
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

void GGL::PPOLearner::InferActions(torch::Tensor obs, torch::Tensor actionMasks, torch::Tensor* outActions, torch::Tensor* outLogProbs, ModelSet* models) {
	ModelSet& m = models ? *models : this->models;

	// Activation steering was removed 2026-07-25 (it had been inert at alpha = 0 since the
	// Ladder superseded it). The opponent-STYLE delta below is a separate, still-supported
	// path; steerDelta stays as its carrier.
	torch::Tensor steerDelta = {};

	InferActionsFromModels(m, obs, actionMasks, config.deterministic, config.policyTemperature, config.useHalfPrecision, outActions, outLogProbs, steerDelta);
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

torch::Tensor GGL::PPOLearner::InferRhatMax(torch::Tensor obs) {
	RG_NO_GRAD;
	obs = obs.to(device, true);
	if (models["shared_head"])
		obs = models["shared_head"]->Forward(obs, config.useHalfPrecision);
	auto a = models["rhat1"]->Forward(obs, config.useHalfPrecision).flatten().to(torch::kFloat32);
	auto b = models["rhat2"]->Forward(obs, config.useHalfPrecision).flatten().to(torch::kFloat32);
	// the heads regress a NORMALISED reward; scale back to reward units on read
	return torch::maximum(a, b).clamp(0.f, 1.f) * rhatMaxObserved;
}


// ===================== IMPLICIT WORLD MODEL =====================
// One-step twin dynamics in OBS space, trained only on executed transitions, then
// optimistic value iteration over that model read out through the worth theory.
// No rollout, no tree, no buffer: exactly one imagined hop per update, with the
// multi-step composition emerging across updates (amortised background planning).

void GGL::PPOLearner::TrainWorldModel(torch::Tensor states, torch::Tensor actions, torch::Tensor cont) {
	if (!models["wm_dyn1"] || !models["wm_dyn2"])
		return;
	int64_t nR = states.size(0);
	if (nR < 256)
		return;

	torch::Tensor sel;
	{
		RG_NO_GRAD;
		// cont[i] == 1 means row i+1 IS the successor of row i (no episode boundary)
		auto ok = (cont.slice(0, 0, nR - 1) > 0.5f).nonzero().flatten().to(torch::kCPU);
		if (ok.numel() < 128)
			return;
		int64_t take = RS_MIN((int64_t)config.vdagWmRows, ok.numel());
		auto perm = torch::randperm(ok.numel(), torch::TensorOptions().dtype(torch::kLong));
		sel = ok.index_select(0, perm.slice(0, 0, take));
	}

	auto s = states.index_select(0, sel).to(device, true);
	auto s2 = states.index_select(0, sel + 1).to(device, true);
	auto a = actions.index_select(0, sel).to(device, true).to(torch::kLong).flatten();
	auto in = torch::cat({ s, torch::one_hot(a, numActionsCached).to(torch::kFloat32) }, 1);
	torch::Tensor tgt = (s2 - s).detach();

	float lossSum = 0.f;
	for (int it = 0; it < 2; it++) {
		torch::Tensor l;
		for (const char* n : { "wm_dyn1", "wm_dyn2" }) {
			auto pred = models[n]->Forward(in, false).to(torch::kFloat32);
			auto li = (pred - tgt).pow(2).mean();
			// OCCAM ON THE DYNAMICS: among models equally consistent with the observed
			// transitions, prefer the one reading FEWER state features. Physics that does
			// not depend on a feature loses that input entirely, so prediction in a
			// never-visited region is the SAME computation as in a visited one. This is
			// the only measured cure for arbitrary off-support extrapolation, and it is
			// what makes the twins AGREE out there instead of vetoing each other -
			// without it the trust gate switches imagination off exactly where it is
			// needed. (Measured: disagreement 300x under threshold in unvisited regions.)
			auto& w0 = models[n]->seq->named_parameters()["0.weight"];
			li = li + config.vdagWmOccam * w0.norm(2, /*dim=*/0).sum();
			l = l.defined() ? l + li : li;
		}
		l.backward();
		for (const char* n : { "wm_dyn1", "wm_dyn2" })
			models[n]->StepOptim();
		lossSum += l.detach().cpu().item<float>();
	}
	dbgWmDyn = lossSum / 2.f;
}

torch::Tensor GGL::PPOLearner::ImagTargetsFor(torch::Tensor sDev, torch::Tensor* outRing) {
	RG_NO_GRAD;
	int64_t B = sDev.size(0), W = sDev.size(1);
	int64_t K = RS_MAX(1, (int64_t)config.vdagWmActions);

	// Candidate actions sampled UNIFORMLY, never top-k by policy: the conduct we are
	// hunting is by definition improbable under the current policy, so scoring only
	// the likely actions would exclude precisely what we are looking for.
	auto acts = torch::randint(0, numActionsCached, { B * K },
		torch::TensorOptions().dtype(torch::kLong).device(sDev.device()));
	auto sRep = sDev.unsqueeze(1).expand({ B, K, W }).reshape({ B * K, W });
	auto in = torch::cat({ sRep, torch::one_hot(acts, numActionsCached).to(torch::kFloat32) }, 1);

	auto d1 = models["wm_dyn1"]->Forward(in, false).to(torch::kFloat32);
	auto d2 = models["wm_dyn2"]->Forward(in, false).to(torch::kFloat32);
	auto dis = (d1 - d2).pow(2).mean(1).view({ B, K });
	auto imag = sRep + 0.5f * (d1 + d2);

	auto trunkI = models["shared_head"] ? models["shared_head"]->Forward(imag, false) : imag;
	auto ra = models["rhat1"]->Forward(trunkI, false).flatten().to(torch::kFloat32);
	auto rb = models["rhat2"]->Forward(trunkI, false).flatten().to(torch::kFloat32);
	auto rImg = (torch::maximum(ra, rb).clamp(0.f, 1.f) * rhatMaxObserved).view({ B, K });
	auto va = models["wm_v1"]->Forward(trunkI, false).flatten().to(torch::kFloat32);
	auto vb = models["wm_v2"]->Forward(trunkI, false).flatten().to(torch::kFloat32);
	auto vImg = torch::minimum(va, vb).clamp(0.f, rhatMaxObserved).view({ B, K });

	// EVENT-TERMINAL optimistic value iteration: arriving somewhere the worth theory
	// says pays, PAYS and stops. The max over actions is a gamma-contraction, and this
	// head never feeds the honest critic, so it cannot ratchet. Accumulating small
	// step costs instead (no terminal) drives the fixed point to -kappa/(1-gamma) and
	// the whole field collapses to its clamp.
	auto isEv = (rImg > 0.5f * rhatMaxObserved).to(torch::kFloat32);
	auto cand = config.vdagWmGammaIm * (isEv * rImg + (1.f - isEv) * vImg) - config.vdagWmKappa;

	// UNKNOWN IS A PLATEAU, NEVER A CLIFF. An untrusted action makes no new claim and
	// simply keeps the state's own value. Defaulting untrusted candidates to zero (or
	// -inf then clamping) makes moving toward unexplored territory read as
	// catastrophic, which inverts the entire field - measured, and the single most
	// confusing failure in this family.
	auto trunkS = models["shared_head"] ? models["shared_head"]->Forward(sDev, false) : sDev;
	auto oa = models["wm_v1"]->Forward(trunkS, false).flatten().to(torch::kFloat32);
	auto ob = models["wm_v2"]->Forward(trunkS, false).flatten().to(torch::kFloat32);
	auto own = torch::minimum(oa, ob).clamp(0.f, rhatMaxObserved).unsqueeze(1).expand({ B, K });
	auto trust = dis < config.vdagWmDisTh;
	cand = torch::where(trust, cand, own);
	dbgWmTrust = trust.to(torch::kFloat32).mean().item<float>();

	if (outRing) {
		auto keep = trust.reshape({ -1 }).nonzero().flatten();
		if (keep.numel() > 64) {
			int64_t take = RS_MIN(B, keep.numel());
			auto perm = torch::randperm(keep.numel(),
				torch::TensorOptions().dtype(torch::kLong).device(keep.device())).slice(0, 0, take);
			*outRing = imag.index_select(0, keep.index_select(0, perm)).detach();
		}
	}
	return std::get<0>(cand.max(1)).clamp(0.f, rhatMaxObserved);
}

void GGL::PPOLearner::TrainImagValue(torch::Tensor states) {
	if (!models["wm_v1"] || !models["wm_dyn1"] || !models["rhat1"])
		return;
	if (rhatMaxObserved <= 0.f)
		return;   // no reward scale observed yet: nothing to bound a hypothesis with
	int64_t nR = states.size(0);
	if (nR < 256)
		return;

	torch::Tensor S, Y;
	{
		RG_NO_GRAD;
		int64_t take = RS_MIN((int64_t)config.vdagWmRows, nR);
		auto perm = torch::randperm(nR, torch::TensorOptions().dtype(torch::kLong)).slice(0, 0, take);
		auto sBase = states.index_select(0, perm).to(device, true);
		torch::Tensor ring;
		auto yBase = ImagTargetsFor(sBase, &ring);
		// BACKGROUND PLANNING: anchor the iteration on MODEL-GENERATED successors as
		// well, so the solved region advances one ring past the data every update.
		// Fitting only on visited states leaves the field's readings everywhere else as
		// unconstrained network extrapolation rather than computed values (measured: a
		// dead-flat field that carried no gradient at all).
		if (ring.defined() && ring.size(0) > 0) {
			auto yRing = ImagTargetsFor(ring, nullptr);
			S = torch::cat({ sBase, ring }, 0);
			Y = torch::cat({ yBase, yRing }, 0);
		} else {
			S = sBase;
			Y = yBase;
		}
		dbgImag = Y.mean().item<float>();
	}

	float lossSum = 0.f;
	for (int it = 0; it < 2; it++) {
		torch::Tensor trunkS;
		{
			// Trunk DETACHED: the world model is a read-only consumer of the shared
			// representation. V-dagger and r-hat deliberately co-adapt the trunk; this
			// one does not get to reshape it.
			RG_NO_GRAD;
			trunkS = models["shared_head"] ? models["shared_head"]->Forward(S, false) : S;
		}
		torch::Tensor l;
		for (const char* n : { "wm_v1", "wm_v2" }) {
			auto pr = models[n]->Forward(trunkS, false).flatten().to(torch::kFloat32);
			auto li = (pr - Y).pow(2).mean();
			l = l.defined() ? l + li : li;
		}
		l.backward();
		for (const char* n : { "wm_v1", "wm_v2" })
			models[n]->StepOptim();
		lossSum += l.detach().cpu().item<float>();
	}
	dbgWmVi = lossSum / 2.f;
}

torch::Tensor GGL::PPOLearner::InferImagValue(torch::Tensor obs) {
	RG_NO_GRAD;
	obs = obs.to(device, true);
	if (models["shared_head"])
		obs = models["shared_head"]->Forward(obs, config.useHalfPrecision);
	auto a = models["wm_v1"]->Forward(obs, config.useHalfPrecision).flatten().to(torch::kFloat32);
	auto b = models["wm_v2"]->Forward(obs, config.useHalfPrecision).flatten().to(torch::kFloat32);
	return torch::minimum(a, b);
}

void GGL::PPOLearner::BankAscent(torch::Tensor obs, torch::Tensor nextObs, torch::Tensor acts, int cap) {
	// Ring-bank ascent transitions on CPU; replay re-scores them with the CURRENT field.
	int64_t n = obs.size(0);
	if (n <= 0) return;
	if (!archObs.defined()) {
		archObs = torch::zeros({ (int64_t)cap, obs.size(1) }, torch::kFloat32);
		archNextObs = torch::zeros({ (int64_t)cap, obs.size(1) }, torch::kFloat32);
		archAct = torch::zeros({ (int64_t)cap }, torch::kLong);
	}
	auto o = obs.to(torch::kCPU, torch::kFloat32), no = nextObs.to(torch::kCPU, torch::kFloat32);
	auto ac = acts.to(torch::kCPU, torch::kLong);
	for (int64_t i = 0; i < n; i++) {
		archObs[archPtr] = o[i]; archNextObs[archPtr] = no[i]; archAct[archPtr] = ac[i];
		archPtr = (archPtr + 1) % cap; archFill = RS_MIN(archFill + 1, (int64_t)cap);
	}
}

torch::Tensor GGL::PPOLearner::InferGoalCritic(torch::Tensor obs) {
	// Independent net: raw obs in, NO shared trunk (by design — zero gradient interference)
	return models["goal_critic"]->Forward(obs, config.useHalfPrecision).flatten();
}

torch::Tensor ComputeEntropyRows(torch::Tensor probs, torch::Tensor actionMasks, bool maskEntropy) {
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

	return entropy;
}

torch::Tensor ComputeEntropy(torch::Tensor probs, torch::Tensor actionMasks, bool maskEntropy) {
	return ComputeEntropyRows(probs, actionMasks, maskEntropy).mean();
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
		avgReachCarStateLoss,
		avgReachLoss,
		avgVdagLoss,
		avgRhatLoss,
		avgArchLive;

	// Save parameters first
	auto policyBefore = models["policy"]->CopyParams();
	auto criticBefore = models["critic"]->CopyParams();
	// V-dagger: the direct "are these weights actually moving?" signal. Frozen at lr=0 this
	// read exactly 0 for the life of the run, which nothing would have revealed.
	auto vdagBefore = models["vdag1"] ? models["vdag1"]->CopyParams() : torch::Tensor();

	bool trainPolicy = config.policyLR != 0;
	bool trainCritic = config.criticLR != 0;
	bool trainSharedHead = models["shared_head"] && (trainPolicy || trainCritic);

	for (int epoch = 0; epoch < config.epochs; epoch++) {

		// Get randomly-ordered timesteps for PPO
		dbgVdagRows = experience.data.vdagTargets.defined() ? (float)experience.data.vdagTargets.numel() : -2.f;
		dbgRhatEntry = experience.data.rhatTargets.defined() ? (float)experience.data.rhatTargets.numel() : -2.f;
		auto batches = experience.GetAllBatchesShuffled(config.batchSize, config.overbatching);
		if (dbgVdagRows > 0 && !batches.empty())
			dbgRhatRows = batches[0].vdagTargets.defined() ? (float)batches[0].vdagTargets.numel() : -3.f;

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
						probs = InferPolicyProbsFromModels(models, obs, actionMasks, config.policyTemperature, false);
						logProbs = probs.log().gather(-1, acts.unsqueeze(-1));
						auto entRows = ComputeEntropyRows(probs, actionMasks, config.maskEntropy);
						// Report the UNWEIGHTED mean so the panel stays comparable to runs
						// without the gate.
						curEntropy = entRows.mean().detach().cpu().item<float>();
						avgEntropy += curEntropy;
						// H-GATED ENTROPY: sample harder where the critic says there is
						// unrealised value. Only ever ADDS stochasticity -> the entropy
						// floor is strengthened by construction.
						if (batch.entWeights.defined()) {
							auto ew = batch.entWeights.slice(0, start, stop)
								.to(device, true, true).view_as(entRows);
							entropy = (entRows * ew).mean();
							dbgEntGate = ew.mean().detach().cpu().item<float>();
						} else {
							entropy = entRows.mean();
						}
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
				// THEORY (r-hat): twin expectile regression on the reward that landed on
				// each arrival state + group-L1 over INPUT features (Occam). Occam is what
				// makes the theory TRANSFER: among theories equally consistent with the
				// data, prefer the one using fewer features, so "this kind of state pays"
				// generalizes to configurations never paid at.
				torch::Tensor rhatLoss;
				if (batch.rhatTargets.defined() && models["rhat1"] && models["rhat2"]) {
					auto yrRaw = batch.rhatTargets.slice(0, start, stop).to(device, true, true).flatten();
					// DIMENSIONLESS target: regress r/scale, so the Occam coefficient means
					// the same thing at any reward magnitude. Against a raw target it was
					// ~40x too weak here relative to the corridor it was set on, and the
					// theory kept its confounds instead of its signal.
					float rsc = RS_MAX(rhatMaxObserved, 1e-6f);
					auto yr = yrRaw / rsc;
					// EVENT-BALANCED: reward events are a small minority of rows, so an
					// unweighted loss makes "predict nothing" near-optimal and Occam then
					// deletes the very geometry the theory needs. Balancing keeps the
					// signal and prunes the confounds instead.
					auto evM = (yrRaw > 0.25f * rsc).to(torch::kFloat32);
					auto nEv = evM.sum().clamp_min(1.f);
					auto nNo = (1.f - evM).sum().clamp_min(1.f);
					float nAll = (float)yr.numel();
					auto wBal = evM * (0.5f * nAll / nEv) + (1.f - evM) * (0.5f * nAll / nNo);
					torch::Tensor trunkR = models["shared_head"]
						? models["shared_head"]->Forward(obs, false) : obs;
					for (Model* rh : { models["rhat1"], models["rhat2"] }) {
						auto pred = rh->Forward(trunkR, false).flatten().to(torch::kFloat32);
						auto u = yr - pred;
						auto tau = config.vdagTheoryTau;
						auto w = torch::where(u > 0, torch::full_like(u, tau), torch::full_like(u, 1.f - tau));
						auto l = (wBal * w * u * u).mean() * batchSizeRatio;
						auto& firstLin = rh->seq->named_parameters()["0.weight"];
						l = l + config.vdagTheoryL1 * firstLin.norm(2, /*dim=*/0).sum() * batchSizeRatio;
						rhatLoss = rhatLoss.defined() ? rhatLoss + l : l;
					}
					avgRhatLoss += rhatLoss.detach().cpu().item<float>();
				}

				torch::Tensor vdagLoss;
				dbgVdagRows = batch.vdagTargets.defined() ? (float)batch.vdagTargets.numel() : -1.f;
				dbgRhatRows = batch.rhatTargets.defined() ? (float)batch.rhatTargets.numel() : -1.f;
				if (batch.vdagTargets.defined() && models["vdag1"] && models["vdag2"]) {
					auto yv = batch.vdagTargets.slice(0, start, stop).to(device, true, true).flatten();
					torch::Tensor trunkV = models["shared_head"]
						? models["shared_head"]->Forward(obs, false) : obs;
					// PLANT: hypothesis rows from the theory, EVENT-MASKED. r-hat is a
					// REWARD theory, not a VALUE theory - seeding V-dagger with its small
					// interior predictions is ballast that flattens the field and kills the
					// backward relay (measured). Seed only where it predicts a real event.
					torch::Tensor ySeed, mSeed;
					if (models["rhat1"] && models["rhat2"] && rhatMaxObserved > 0) {
						RG_NO_GRAD;
						auto ra = models["rhat1"]->Forward(trunkV.detach(), false).flatten().to(torch::kFloat32);
						auto rb = models["rhat2"]->Forward(trunkV.detach(), false).flatten().to(torch::kFloat32);
						ySeed = torch::maximum(ra, rb).clamp(0.f, 1.f) * rhatMaxObserved;
						mSeed = (ySeed > config.vdagSeedFrac * rhatMaxObserved).to(torch::kFloat32);
					}
					for (Model* vh : { models["vdag1"], models["vdag2"] }) {
						auto pred = vh->Forward(trunkV, false).flatten().to(torch::kFloat32);
						auto u = yv - pred;
						auto w = torch::where(u > 0,
							torch::full_like(u, config.vdagTau), torch::full_like(u, 1.f - config.vdagTau));
						auto l = (w * u * u).mean() * batchSizeRatio;
						if (ySeed.defined() && mSeed.sum().item<float>() > 0) {
							auto us = ySeed - pred;
							auto ws = torch::where(us > 0,
								torch::full_like(us, config.vdagTau), torch::full_like(us, 1.f - config.vdagTau));
							l = l + config.vdagSeedWeight * ((mSeed * ws * us * us).sum()
								/ (mSeed.sum() + 1e-6f)) * batchSizeRatio;
						}
						vdagLoss = vdagLoss.defined() ? vdagLoss + l : l;
					}
					avgVdagLoss += vdagLoss.detach().cpu().item<float>();
					dbgVdagRaw = vdagLoss.detach().cpu().item<float>();
					dbgYvAbs = yv.abs().mean().item<float>();
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
				if (rhatLoss.defined())
					totalLoss = totalLoss.defined() ? totalLoss + rhatLoss : rhatLoss;

				// ARCHIVE replay: convert the field into POLICY. PBRS with a good field is
				// neutral BY THEOREM, so ascent must be imitated directly. Weights are
				// RE-COMPUTED with the CURRENT field, so entries that no longer climb drop
				// out on their own (no eviction heuristic, no stale imitation).
				if (config.vdagArchiveEnabled && trainPolicy && archFill > 64 && models["vdag1"]) {
					int64_t nRep = RS_MIN((int64_t)512, archFill);
					auto ridx = torch::randint(0, archFill, { nRep }, torch::TensorOptions().dtype(torch::kLong));
					auto rs = archObs.index_select(0, ridx).to(device, true);
					auto rsn = archNextObs.index_select(0, ridx).to(device, true);
					auto ra = archAct.index_select(0, ridx).to(device, true);
					torch::Tensor wr;
					{
						RG_NO_GRAD;
						wr = torch::relu(config.gaeGamma * InferVdagMin(rsn) - InferVdagMin(rs));
					}
					auto live = wr > config.vdagArchiveLiveThresh;
					float liveFrac = live.to(torch::kFloat32).mean().item<float>();
					avgArchLive += liveFrac;
					if (live.any().item<bool>()) {
						auto li = live.nonzero().flatten();
						auto sObs = rs.index_select(0, li);
						auto sAct = ra.index_select(0, li);
						auto wSel = wr.index_select(0, li);
						wSel = (wSel / (wSel.mean() + 1e-8f)).clamp(0.f, 10.f);
						auto trunkA = models["shared_head"] ? models["shared_head"]->Forward(sObs, false) : sObs;
						auto logits = models["policy"]->Forward(trunkA, false);
						auto logp = torch::log_softmax(logits.to(torch::kFloat32), -1)
							.gather(1, sAct.view({ -1, 1 })).flatten();
						auto bc = config.vdagArchiveWeight * (wSel * (-logp)).mean() * batchSizeRatio;
						totalLoss = totalLoss.defined() ? totalLoss + bc : bc;
					}
				}

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


			if (trainPolicy)
				nn::utils::clip_grad_norm_(models["policy"]->parameters(), 0.5f);
			if (trainCritic)
				nn::utils::clip_grad_norm_(models["critic"]->parameters(), 0.5f);

			if (trainSharedHead)
				nn::utils::clip_grad_norm_(models["shared_head"]->parameters(), 0.5f);

			if (models["goal_critic"])
				nn::utils::clip_grad_norm_(models["goal_critic"]->parameters(), 0.5f);

			// V-dagger twins clip at the same 0.5 as every other head. They were absent from
			// this list, which was harmless only because their LR was 0 - the moment the LR is
			// wired (same commit) 16.1M params would otherwise be the only unclipped block in
			// the model, on an expectile loss whose targets are the widest-scale ones here.
			for (const char* n : { "vdag1", "vdag2" })
				if (models[n])
					nn::utils::clip_grad_norm_(models[n]->parameters(), 0.5f);

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
	float vdagUpdateMagnitude = vdagBefore.defined()
		? (vdagBefore - models["vdag1"]->CopyParams()).norm().item<float>() : 0.f;

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
	if (models["vdag1"]) {
		report["Headroom/Vdag Loss"] = avgVdagLoss.Get();
		// Must be > 0. It was exactly 0 for the whole run until the LR was wired (2026-07-25);
		// if it reads 0 again, HEADROOM is inert and its 0.15-sigma injection is noise.
		report["Headroom/Vdag Update Magnitude"] = vdagUpdateMagnitude;
		// Row counts: -1 means the targets never reached the batch (broken plumbing),
		// which is indistinguishable from "loss is small" on the loss panel alone.
		report["Headroom/Vdag Rows"] = dbgVdagRows;
		report["Headroom/Rhat Rows"] = dbgRhatRows;
		if (dbgEntGate >= 0.f)
			report["Headroom/Ent Gate"] = dbgEntGate;
		if (config.vdagWmEnabled) {
			report["Headroom/WM Dyn Loss"] = dbgWmDyn;
			report["Headroom/WM VI Loss"] = dbgWmVi;
			report["Headroom/WM Trust Frac"] = dbgWmTrust;
			report["Headroom/Imag Mean"] = dbgImag;
		}
		report["Headroom/Rhat Entry"] = dbgRhatEntry;
		report["Headroom/Vdag Raw"] = dbgVdagRaw;
		report["Headroom/Rhat Entry"] = dbgRhatEntry;
		report["Headroom/Vdag Raw"] = dbgVdagRaw;
		report["Headroom/Yv Abs"] = dbgYvAbs;
		if (models["rhat1"]) {
			report["Headroom/Rhat Loss"] = avgRhatLoss.Get();
			report["Headroom/Rhat Max Obs"] = rhatMaxObserved;
		}
		if (archFill > 0) {
			report["Headroom/Archive Fill"] = (float)archFill;
			report["Headroom/Archive Live Frac"] = avgArchLive.Get();
		}
	}
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

		// PLASTICITY (promoted from the retired PSD subsystem, 2026-07-25). Weights-only, no data
		// batch, so it is cheap enough to run every iteration. Effective-rank decay is the
		// measurement that justified the residual architecture (45a59d5); keeping it means the
		// project can still check whether that change did what it was chosen to do.
		{
			auto lins = Plasticity::LinearLayers(models["policy"]);
			if (!lins.empty())
				report["Plasticity/Policy EffRank"] = Plasticity::EffectiveRank(lins.back()->weight);
			report["Plasticity/Policy Dead Units"] = Plasticity::DeadUnitFraction(models["policy"]);
			if (models["shared_head"]) {
				auto tl = Plasticity::LinearLayers(models["shared_head"]);
				if (!tl.empty())
					report["Plasticity/Trunk EffRank"] = Plasticity::EffectiveRank(tl.back()->weight);
			}
		}
	}

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

	// HEADROOM V-dagger twins. These were MISSING here from the day they were added
	// (9856813), and Model's ctor builds every optimizer at lr=0 (Util/Models.cpp) - under
	// Muon, which they inherit from the critic config, lr=0 is an exact no-op. So 16.1M
	// params (42% of the net) sat frozen at random init while H = relu(min(V1,V2) - V_real)
	// was still injected into advantages at vdagSeekBeta and still backpropagated into the
	// shared trunk. Found by the 2026-07-25 audit. Wiring this turns HEADROOM on for the
	// FIRST time - treat a regression here as a new deployment, not a fix gone wrong.
	// They mirror the critic's architecture and target scale, so they take criticLR.
	for (const char* n : { "vdag1", "vdag2" })
		if (models[n])
			models[n]->SetOptimLR(criticLR);

	// THEORY r-hat twins: same trap (ctor builds optimizers at lr=0 = exact no-op under
	// Muon). They regress a REWARD, not a return, so they take criticLR as well.
	for (const char* n : { "rhat1", "rhat2" })
		if (models[n])
			models[n]->SetOptimLR(criticLR);

	// World-model heads: same ctor lr=0 trap as every other added head.
	for (const char* n : { "wm_dyn1", "wm_dyn2", "wm_v1", "wm_v2" })
		if (models[n])
			models[n]->SetOptimLR(criticLR);

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

		result.Add(model);
	}
	return result;
}