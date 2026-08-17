#include "PPOLearner.h"

#include <cstdlib>
#include <cmath>
#include <string>

#include <torch/nn/utils/convert_parameters.h>
#include <torch/nn/utils/clip_grad.h>
#include <torch/csrc/autograd/autograd.h>   // torch::autograd::grad — the input-gradient the HJB residual needs
#include <torch/csrc/api/include/torch/serialize.h>
#include <ATen/autocast_mode.h>             // learn-pass autocast (BF16 sm_80+ / FP16 V100)
#include <torch/version.h>
#include <torch/cuda.h>                      // GGL_CONSUME_TIMERS synchronize points
#include <public/GigaLearnCPP/Util/AvgTracker.h>
#include <public/GigaLearnCPP/Util/Timer.h>
#include "../Util/MoE.h"
#include <RLGymCPP/CommonValues.h>
#include "../Util/Plasticity.h"

using namespace torch;

// RAII learn-pass autocast; see PPOLearnerConfig::learnAutocastBF16.
// Restores the previous state on scope exit (including on a thrown exception) and clears
// autocast's weight-cast cache, which MUST NOT outlive an optimizer step or the next
// minibatch would matmul against stale casts of the pre-step weights.
// Dtype is GGLHalfPrecType(): BF16 on sm_80+, FP16 on V100. FP16 needs loss scaling
// (done around backward, not here). Geo HJB / InfoNCE / backward stay outside the region.
namespace {
	struct AutocastScope {
		bool active;
		bool prev = false;
		explicit AutocastScope(bool enable) : active(false) {
			if (!enable)
				return;
			active = true;
			auto dt = GGL::GGLHalfPrecType();
			// Libtorch 2.1 uses global GPU autocast; 2.4+ takes a device type.
#if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 4)
			prev = at::autocast::is_autocast_enabled(at::kCUDA);
			at::autocast::set_autocast_dtype(at::kCUDA, dt);
			at::autocast::set_autocast_enabled(at::kCUDA, true);
#else
			prev = at::autocast::is_enabled();
			at::autocast::set_autocast_gpu_dtype(dt);
			at::autocast::set_enabled(true);
#endif
		}
		// End the region early and idempotently. Everything after the call — the geo HJB
		// double-backward, the InfoNCE heads, and backward() itself — runs in fp32.
		void End() {
			if (!active)
				return;
#if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 4)
			at::autocast::set_autocast_enabled(at::kCUDA, prev);
#else
			at::autocast::set_enabled(prev);
#endif
			at::autocast::clear_cache();
			active = false;
		}
		~AutocastScope() { End(); }
	};

	template <typename Fn>
	void AmpForEachGrad(GGL::ModelSet& models, Fn&& fn) {
		for (GGL::Model* m : models) {
			if (!m)
				continue;
			for (auto& p : m->parameters()) {
				auto g = p.grad();
				if (g.defined())
					fn(g);
			}
		}
	}

	void AmpUnscaleGrads(GGL::ModelSet& models, float scale) {
		if (scale == 1.f)
			return;
		AmpForEachGrad(models, [scale](torch::Tensor& g) { g.div_(scale); });
	}

	bool AmpGradsFinite(GGL::ModelSet& models) {
		torch::Tensor acc;
		AmpForEachGrad(models, [&](torch::Tensor& g) {
			auto f = g.isfinite().all();
			acc = acc.defined() ? acc.logical_and(f) : f;
		});
		return !acc.defined() || acc.item<bool>();
	}

	void AmpZeroGrads(GGL::ModelSet& models) {
		for (GGL::Model* m : models) {
			if (!m || m->groupStepExempt || !m->optim)
				continue;
			m->optim->zero_grad();
		}
	}
}

GGL::PPOLearner::PPOLearner(int obsSize, int numActions, PPOLearnerConfig _config, Device _device, Dist::Session* dist) : config(_config), device(_device), dist(dist) {
	// EP (MOE_SPEED.md Stage 2) arming: env GGL_MOE_EP + a multi-rank learner group.
	// Learn-only; the collect fast path is untouched. Safe to call unconditionally —
	// SetMoEExpertParallel no-ops when the env flag is unset.
	GGL::SetMoEExpertParallel(dist);


	if (config.miniBatchSize == 0)
		config.miniBatchSize = config.batchSize;

	if (config.batchSize % config.miniBatchSize != 0)
		RG_ERR_CLOSE("PPOLearner: config.batchSize (" << config.batchSize << ") must be a multiple of config.miniBatchSize (" << config.miniBatchSize << ")");

	numActionsCached = numActions; obsSizeCached = obsSize;
	if (const char* s = std::getenv("GGL_AMP_SCALE"); s && s[0]) {
		float v = std::strtof(s, nullptr);
		if (v >= 1.f)
			ampLossScale = v;
	}
	if (config.learnAutocastBF16 && device.is_cuda()) {
		if (GGL::GGLCudaHasBF16()) {
			RG_LOG("PPOLearner: learn autocast bf16 (no scaler)");
		} else {
			RG_LOG("PPOLearner: learn autocast fp16 loss_scale=" << ampLossScale);
		}
	}
	MakeModels(true, obsSize, numActions, config.sharedHead, config.policy, config.critic,
		config.criticTrunk, device, models);

	// Secondary goal-only critic: a fully independent net (raw obs in, no shared trunk) so its
	// gradients can't touch the proven policy/critic path. Lives in `models` so it checkpoints and
	// steps with everything else; excluded from GetPolicyModels() like the main critic.
	if (config.goalCritic.enabled) {
		RG_ASSERT(config.goalCritic.model.IsValid());
		RG_ASSERT(config.goalCritic.beta >= 0); // a negative blend would train AWAY from goals
		ModelConfig gcConfig = config.goalCritic.model;
		// models["critic"]->config.numInputs is whatever the critic head reads: critic-trunk
		// width, or main-trunk width, or obsSize. Deriving it keeps the two heads in lockstep
		// instead of re-deriving the same width from two places.
		gcConfig.numInputs = models["critic_trunk"] ? models["critic"]->config.numInputs : obsSize;
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

	// GEOMETRY (4th rung). Three INDEPENDENT nets on raw obs — no trunk, like the original
	// goal-critic form — so this rung cannot co-adapt the shared perception path on day one.
	//   geo_sigma : obs -> 2*obs   conditional (mean, log-std) of the one-step displacement.
	//               Only the log-std is used by the HJB; the mean exists so that std is the
	//               spread AROUND the drift (what the action buys) rather than total motion
	//               (which would include the ball falling whether we act or not).
	//   geo_rew   : obs -> 1       r_hat on the ARRIVAL state, fit on the reservoir.
	//   geo_v     : obs -> 1       V_geo, trained to zero the HJB residual.
	if (config.geoEnabled) {
		RG_ASSERT(config.geoModel.IsValid());
		ModelConfig gs = config.geoModel; gs.numInputs = obsSize; gs.numOutputs = obsSize * 2;
		ModelConfig gr = config.geoModel; gr.numInputs = obsSize; gr.numOutputs = 1;
		ModelConfig gv = config.geoModel; gv.numInputs = obsSize; gv.numOutputs = 1;
		models.Add(new Model("geo_sigma", gs, device));
		models.Add(new Model("geo_rew", gr, device));
		models.Add(new Model("geo_v", gv, device));
	}

	// COMPOSITE VALUE CRITIC (PPOLearnerConfig -- the composite value block).
	if (config.valueTwinEnabled) {
		// critic2: EXACTLY the main critic's resolved config (input width depends on the
		// trunk stack, so clone from the built model, not from the partial config).
		// Each head trains on a disjoint half of every minibatch; readout is the mean.
		RG_ASSERT(models["critic"]);
		models.Add(new Model("critic2", models["critic"]->config, device));
	}
	if (config.auxDispEnabled) {
		// one-step displacement (mu, log-sigma) head on the SHARED trunk output --
		// representation pressure, deliberately outside the value head
		int trunkOut = config.sharedHead.IsValid()
			? config.sharedHead.layerSizes.back() : obsSize;
		ModelConfig ad = config.auxDispModel.IsValid()
			? ModelConfig(config.auxDispModel) : ModelConfig(PartialModelConfig{});
		if (!config.auxDispModel.IsValid())
			ad.layerSizes = { 256 };
		ad.numInputs = trunkOut; ad.numOutputs = obsSize * 2;
		models.Add(new Model("aux_disp", ad, device));
	}
	if (config.oppCondEnabled) {
		// zero-init additive opponent embedding into the value body: an exact no-op at
		// init (and therefore on old-checkpoint loads) that learning can grow into
		int vtOut = config.criticTrunk.IsValid() ? config.criticTrunk.layerSizes.back()
			: (config.sharedHead.IsValid() ? config.sharedHead.layerSizes.back() : obsSize);
		ModelConfig oe = PartialModelConfig{};
		oe.layerSizes = { 32 };
		oe.numInputs = config.oppCtxDim; oe.numOutputs = vtOut;
		Model* m = new Model("opp_embed", oe, device);
		{
			RG_NO_GRAD;
			auto params = m->parameters();
			if (!params.empty()) {
				params.back().zero_();                       // final bias
				if (params.size() >= 2) params[params.size() - 2].zero_();  // final weight
			}
		}
		models.Add(m);
		oppCtxLive = torch::zeros({ config.oppCtxDim },
			torch::TensorOptions().dtype(torch::kFloat32).device(device));
	}

	// HULL OPERATOR chart (EPSILON_CRITIC.md section 7). Two small OFF-trunk nets:
	//   hull_proj : obs -> chartDim   the learned dynamics chart (near-linear, L1-sparsified
	//               in the training loss -- it discovers which obs dims CONDITION the local
	//               physics and prunes the rest; inspect its first-layer weights to audit)
	//   hull_head : chartDim -> 2*obs (mu, log-sigma) of the one-step displacement given the
	//               chart cell -- the NLL that trains the chart, and the likelihood reference
	//               for the (optional, estimator-seat) donor gates.
	// The donor BANK is the geo reservoir (obs, next) pairs; deltas = next - obs.
	if (config.hullEnabled) {
		ModelConfig hp = PartialModelConfig{};
		hp.layerSizes = { config.hullChartDim };
		hp.addOutputLayer = false;           // chart coords ARE the layer output
		hp.addLayerNorm = false;             // distances in chart space must be raw
		hp.numInputs = obsSize; hp.numOutputs = config.hullChartDim;
		ModelConfig hh = config.hullHeadModel.IsValid()
			? ModelConfig(config.hullHeadModel) : ModelConfig(PartialModelConfig{});
		if (!config.hullHeadModel.IsValid())
			hh.layerSizes = { 64 };
		hh.numInputs = config.hullChartDim; hh.numOutputs = obsSize * 2;
		models.Add(new Model("hull_proj", hp, device));
		models.Add(new Model("hull_head", hh, device));
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
		// makeCritic=false, so no critic trunk either (it is value-side only)
		MakeModels(false, obsSize, numActions, config.sharedHead, config.policy, config.critic,
			/*criticTrunkConfig=*/{}, device, guidingPolicyModels);
		guidingPolicyModels.Load(config.guidingPolicyPath, false, false);
	}
}

GGL::PPOLearner::~PPOLearner() {
	delete reach;
	reach = nullptr;
	models.Free();
	guidingPolicyModels.Free();
}

void GGL::PPOLearner::MakeModels(
	bool makeCritic,
	int obsSize, int numActions,
	PartialModelConfig sharedHeadConfig, PartialModelConfig policyConfig, PartialModelConfig criticConfig,
	PartialModelConfig criticTrunkConfig,
	torch::Device device,
	ModelSet& outModels) {

	ModelConfig fullPolicyConfig = policyConfig;
	fullPolicyConfig.numInputs = obsSize;
	fullPolicyConfig.numOutputs = numActions;

	ModelConfig fullCriticConfig = criticConfig;
	fullCriticConfig.numInputs = obsSize;
	fullCriticConfig.numOutputs = 1;

	// A configured-but-invalid MoE trunk must fail LOUDLY: IsValid()==false here
	// otherwise means "no shared head", and the trainer runs trunk-less with null
	// map entries until the first unguarded lookup segfaults (2026-08-16).
	if (sharedHeadConfig.moeBlocks > 0 && !sharedHeadConfig.IsValid())
		RG_ERR_CLOSE("sharedHead: moeBlocks=" << sharedHeadConfig.moeBlocks
			<< " but the config is invalid (layerSizes must be exactly 1 entry,"
			<< " addResiduals off, expert/topK/hidden > 0) - refusing to silently"
			<< " build a trunk-less model set");
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

	if (makeCritic) {
		// CRITIC TRUNK: a second shared body between the main trunk and the value heads. Built
		// only alongside the critics, so inference-only model sets (InferUnit, old policy
		// versions) never carry it. Every value head's input width becomes its output width;
		// the goal critic picks that up from models["critic"] in the ctor above.
		if (criticTrunkConfig.IsValid()) {
			RG_ASSERT(!criticTrunkConfig.addOutputLayer); // it is a body, not a head
			ModelConfig fullCriticTrunkConfig = criticTrunkConfig;
			fullCriticTrunkConfig.numInputs = fullCriticConfig.numInputs;
			fullCriticTrunkConfig.numOutputs = 0;
			outModels.Add(new Model("critic_trunk", fullCriticTrunkConfig, device));
			fullCriticConfig.numInputs = criticTrunkConfig.layerSizes.back();
		}

		outModels.Add(new Model("critic", fullCriticConfig, device));
	}
}

torch::Tensor GGL::PPOLearner::InferPolicyProbsFromModels(
	ModelSet& models,
	torch::Tensor obs, torch::Tensor actionMasks,
	float temperature, bool halfPrec,
	torch::Tensor steerDelta, torch::Tensor* outRowOk,
	torch::Tensor precomputedTrunk,
	bool useCudaGraph) {

	if (useCudaGraph)
		return PolicyCudaGraph::InferPolicyProbs(
			models, obs, actionMasks, temperature, halfPrec,
			steerDelta, outRowOk, precomputedTrunk, true);

	actionMasks = actionMasks.to(torch::kBool);

	constexpr float ACTION_MIN_PROB = 1e-11f;
	constexpr float ACTION_DISABLED_LOGIT = -1e10f;

	torch::Tensor rawObs = obs; // kept for the non-finite diagnostic below
	if (precomputedTrunk.defined())
		// The learn pass already forwarded shared_head for the value family; reusing that
		// tensor keeps the graph single-trunk, so autograd accumulates the policy's and the
		// value heads' gradients at one node and traverses the trunk ONCE (see the note by
		// fnValueTrunk in Learn). Header contract: same rows, same model, same order.
		obs = precomputedTrunk;
	else if (models["shared_head"])
		obs = models["shared_head"]->Forward(obs, halfPrec, /*keepHalf=*/halfPrec);

	// Steered-practice: shift trunk output in the commitment direction for masked rows.
	// keepHalf leaves trunk in fp16/bf16, so cast the delta to that dtype.
	if (steerDelta.defined()) {
		RG_ASSERT(models["shared_head"]); // the direction lives in trunk-output space
		obs = obs + steerDelta.to(obs.dtype());
	}

	// temperature == 1 is the live setting and `/ 1.0f` is a full [rows x 90] elementwise kernel
	// launched every collection step for nothing. Inference here is DISPATCH-bound, not
	// arithmetic-bound (the 21-op network was issuing ~55 GPU ops), so an op that computes an
	// identity is a real cost.
	auto logits = models["policy"]->Forward(obs, halfPrec, /*keepHalf=*/halfPrec);
	if (logits.scalar_type() != torch::kFloat)
		logits = logits.to(torch::kFloat);
	if (temperature != 1.f)
		logits = logits / temperature;

	// A non-finite logit row would crash multinomial downstream with a device-side assert that
	// poisons the CUDA context - identify the SOURCE here instead. The reduction is cheap; the
	// .item() is NOT: it is a blocking sync between the forward and the sample, once per
	// collection step (and on a shared stream it drained queued learn kernels too). So the hot
	// path (outRowOk set) does not sync: bad rows are sanitized to uniform logits so sampling
	// stays safe, and the flags are returned for a deferred verdict at a sync the caller pays
	// anyway. On failure the caller re-invokes with outRowOk = null, which lands in the
	// synchronous branch below and dies with the full forensic message.
	auto rowOk = logits.isfinite().all(-1, /*keepdim=*/true);
	if (outRowOk) {
		*outRowOk = rowOk;
		// REVERTED to torch::where (2026-08-04). Replacing this with
		// `logits.masked_fill_(rowOk.logical_not(), 0.f)` looked like a strict op-count win —
		// two ops instead of two, one of them in place — and MEASURED as part of a 40%
		// inference regression (1.790s -> 2.504s median). Op counting is not cost: `where` with
		// a [rows,1] condition against [rows,90] is a well-optimized broadcast ternary, whereas
		// broadcasting a mask through in-place masked_fill_ is not. Do not "optimize" this
		// again without measuring it alone.
		logits = torch::where(rowOk, logits, torch::zeros_like(logits));
	} else if (!rowOk.all().item<bool>()) {
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
	bool useCudaGraph) {

	torch::Tensor rowOk;
	auto probs = InferPolicyProbsFromModels(
		models, obs, actionMasks, temperature, halfPrec,
		steerDelta, &rowOk, {}, useCudaGraph);

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

	// Deferred non-finite verdict. Sanitizer above keeps multinomial safe. The .item() is a
	// blocking CUDA sync; collection immediately .cpu()'s the actions anyway, so this is a
	// second drain. GGL_INFER_FINITE_SYNC: 1 = every call (old), N = every Nth, 0 = skip.
	{
		static int finiteEvery = []() {
			const char* s = std::getenv("GGL_INFER_FINITE_SYNC");
			return (s && *s) ? std::atoi(s) : 1;
		}();
		// atomic: GGL_OPP_PARALLEL runs a second InferActions on a side thread; with
		// GGL_INFER_FINITE_SYNC=0 (production) this counter is never touched.
		static std::atomic<int> finiteCtr = 0;
		if (finiteEvery > 0 && ((++finiteCtr) % finiteEvery) == 0 && !rowOk.all().item<bool>()) {
			InferPolicyProbsFromModels(models, obs, actionMasks, temperature, halfPrec, steerDelta);
			RG_ERR_CLOSE("InferActionsFromModels: non-finite logits, but the diagnostic rerun did not reproduce them");
		}
	}
}

void GGL::PPOLearner::InferActionsLowRankES(ModelSet& models, torch::Tensor obs,
	torch::Tensor actionMasks, const EsLowRankCtx& ctx,
	torch::Tensor* outActions, torch::Tensor* outLogProbs) {

	RG_NO_GRAD;
	const bool hp = config.useHalfPrecision && device.is_cuda();
	constexpr float ACTION_MIN_PROB = 1e-11f;
	constexpr float ACTION_DISABLED_LOGIT = -1e10f;

	torch::Tensor x = obs;
	int li = 0; // Linear cursor across shared_head then policy — MUST match ctx build order
	for (const char* nm : { "shared_head", "policy" }) {
		Model* m = models[nm];
		if (!m)
			continue;
		torch::nn::Sequential& sq = hp ? (m->RefreshHalfCache(), m->seqHalf) : m->seq;
		if (hp) {
			auto ps = sq->parameters();
			if (!ps.empty() && x.scalar_type() != ps[0].scalar_type())
				x = x.to(ps[0].scalar_type());
		}
		std::vector<torch::Tensor> saved(m->residualSpans.size());
		for (int i = 0; i < (int)sq->size(); i++) {
			for (int s = 0; s < (int)m->residualSpans.size(); s++)
				if (m->residualSpans[s].first == i)
					saved[s] = x;
			torch::Tensor xin = x;
			auto modPtr = sq->ptr(i);
			x = GGL::ForwardSeqModule(modPtr, x);
			if (std::dynamic_pointer_cast<torch::nn::LinearImpl>(modPtr)) {
				RG_ASSERT((size_t)li < ctx.A.size());
				// y += sigma_l * (x_in . B_l[m]) * A_l[m], the rank-1 member delta.
				auto Ag = ctx.A[(size_t)li].index_select(0, ctx.rowMember).to(x.scalar_type());
				auto Bg = ctx.B[(size_t)li].index_select(0, ctx.rowMember).to(xin.scalar_type());
				auto u = (xin * Bg).sum(-1, /*keepdim=*/true);
				x = x + (u * Ag) * ctx.sigma[(size_t)li];
				li++;
			}
			for (int s = 0; s < (int)m->residualSpans.size(); s++)
				if (m->residualSpans[s].second == i && saved[s].defined())
					x = x + saved[s];
		}
	}
	RG_ASSERT((size_t)li == ctx.A.size()); // every noise layer consumed exactly once

	torch::Tensor logits = x;
	if (logits.scalar_type() != torch::kFloat)
		logits = logits.to(torch::kFloat);
	// Same non-finite sanitize as the hot InferActions path: bad rows -> uniform logits.
	auto rowOk = logits.isfinite().all(-1, /*keepdim=*/true);
	logits = torch::where(rowOk, logits, torch::zeros_like(logits));
	auto probs = torch::softmax(
		logits + ACTION_DISABLED_LOGIT * actionMasks.to(torch::kBool).logical_not(), -1)
		.clamp(ACTION_MIN_PROB, 1);
	auto action = torch::multinomial(probs, 1, true);
	if (outActions)
		*outActions = action.flatten();
	if (outLogProbs)
		*outLogProbs = torch::log(probs).gather(-1, action).flatten();
}

void GGL::PPOLearner::InferActions(
	torch::Tensor obs, torch::Tensor actionMasks,
	torch::Tensor* outActions, torch::Tensor* outLogProbs,
	ModelSet* models,
	bool allowCudaGraph) {
	ModelSet& m = models ? *models : this->models;

	// Activation steering was removed 2026-07-25 (it had been inert at alpha = 0 since the
	// Ladder superseded it). The opponent-STYLE delta below is a separate, still-supported
	// path; steerDelta stays as its carrier.
	torch::Tensor steerDelta = {};

	InferActionsFromModels(
		m, obs, actionMasks, config.deterministic, config.policyTemperature,
		config.useHalfPrecision, outActions, outLogProbs, steerDelta,
		config.useCudaGraphs && allowCudaGraph);
}

GGL::PolicyCudaGraphStats GGL::PPOLearner::GetCudaGraphStats() {
	return PolicyCudaGraph::GetStats();
}

torch::Tensor GGL::PPOLearner::ValueTrunk(torch::Tensor obs, bool halfPrec) {
	if (models["shared_head"])
		obs = models["shared_head"]->Forward(obs, halfPrec);
	if (models["critic_trunk"])
		obs = models["critic_trunk"]->Forward(obs, halfPrec);
	return obs;
}

torch::Tensor GGL::PPOLearner::InferCritic(torch::Tensor obs) {
	bool hp = config.useHalfPrecision;
	auto vt = ValueTrunk(obs, hp);
	if (config.oppCondEnabled && models["opp_embed"] && oppCtxLive.defined())
		vt = vt + models["opp_embed"]->Forward(
			oppCtxLive.unsqueeze(0), false).expand({ vt.size(0), -1 });
	auto v = models["critic"]->Forward(vt, hp).flatten();
	if (config.valueTwinEnabled && models["critic2"])
		v = 0.5f * (v + models["critic2"]->Forward(vt, hp).flatten());
	return v;
}

torch::Tensor GGL::PPOLearner::InferVdagMin(torch::Tensor obs) {
	RG_NO_GRAD;
	obs = ValueTrunk(obs.to(device, true), config.useHalfPrecision);
	auto a = models["vdag1"]->Forward(obs, config.useHalfPrecision).flatten().to(torch::kFloat32);
	auto b = models["vdag2"]->Forward(obs, config.useHalfPrecision).flatten().to(torch::kFloat32);
	return torch::minimum(a, b);
}

// ONE trunk + critic_trunk forward serving EVERY value head, for the consumption path.
// The Infer* helpers above each rebuild ValueTrunk internally, so calling InferCritic +
// InferGoalCritic + InferVdagMin over the same buffer — which Learner.cpp did, in three
// separate chunk loops — forwarded shared_head 3x and critic_trunk 3x over every row, and
// uploaded the same states 3x. That is 10.26M of 14.97M MAC/row paid three times for one
// result. This is the same duplication that was found on the LEARN side on 2026-08-04
// (fnTrunkVR/fnValueTrunk memoize it there); this is the consumption-side twin.
// Any output tensor may be left undefined to skip that head. geo_v is included because it
// shares the chunk's HOST->DEVICE upload even though it reads raw obs, not the trunk.
void GGL::PPOLearner::InferValueFamily(
	torch::Tensor obs, torch::Tensor* outCritic, torch::Tensor* outGoalCritic,
	torch::Tensor* outVdagMin, torch::Tensor* outGeoV) {

	RG_NO_GRAD;
	bool hp = config.useHalfPrecision;
	auto obsDev = obs;
	if (obsDev.device() != device)
		obsDev = obs.to(device, /*non_blocking=*/true);

	if (outGeoV && models["geo_v"])                    // independent net, raw obs
		*outGeoV = models["geo_v"]->Forward(obsDev, hp).flatten().to(torch::kFloat32);

	bool needTrunk = (outCritic && models["critic"])
		|| (outGoalCritic && models["goal_critic"])
		|| (outVdagMin && models["vdag1"] && models["vdag2"]);
	if (!needTrunk)
		return;

	auto vtBare = ValueTrunk(obsDev, hp);              // THE one trunk + critic_trunk forward
	// Learn and InferVdagMin read V-dagger off the unconditioned trunk. Critic / goal
	// critic get opp_embed on a copy (vtL = trunk + embed). Do not fold embed into
	// vtBare or consume-side min(V1,V2) would disagree with the trained heads.
	torch::Tensor vtCrit = vtBare;
	if (config.oppCondEnabled && models["opp_embed"] && oppCtxLive.defined())
		// privileged opponent conditioning: additive, zero-init at birth (exact no-op
		// until trained); the collection worker sets oppCtxLive for its iteration
		vtCrit = vtBare + models["opp_embed"]->Forward(
			oppCtxLive.unsqueeze(0), false).expand({ vtBare.size(0), -1 });

	if (outCritic && models["critic"]) {
		auto v = models["critic"]->Forward(vtCrit, hp).flatten().to(torch::kFloat32);
		if (config.valueTwinEnabled && models["critic2"])
			v = 0.5f * (v + models["critic2"]->Forward(vtCrit, hp).flatten().to(torch::kFloat32));
		*outCritic = v;
	}
	if (outGoalCritic && models["goal_critic"]) {
		// InferGoalCritic reads the critic trunk only when one exists; without it the head
		// takes raw obs, so preserve that branch exactly.
		*outGoalCritic = models["critic_trunk"]
			? models["goal_critic"]->Forward(vtCrit, hp).flatten().to(torch::kFloat32)
			: models["goal_critic"]->Forward(obsDev, hp).flatten().to(torch::kFloat32);
	}
	if (outVdagMin && models["vdag1"] && models["vdag2"]) {
		auto a = models["vdag1"]->Forward(vtBare, hp).flatten().to(torch::kFloat32);
		auto b = models["vdag2"]->Forward(vtBare, hp).flatten().to(torch::kFloat32);
		*outVdagMin = torch::minimum(a, b);
	}
}

torch::Tensor GGL::PPOLearner::InferRhatMax(torch::Tensor obs) {
	RG_NO_GRAD;
	obs = ValueTrunk(obs.to(device, true), config.useHalfPrecision);
	auto a = models["rhat1"]->Forward(obs, config.useHalfPrecision).flatten().to(torch::kFloat32);
	auto b = models["rhat2"]->Forward(obs, config.useHalfPrecision).flatten().to(torch::kFloat32);
	// the heads regress a NORMALISED reward; scale back to reward units on read
	return torch::maximum(a, b).clamp(0.f, 1.f) * rhatMaxObserved;
}


torch::Tensor GGL::PPOLearner::InferGeoV(torch::Tensor obs) {
	RG_NO_GRAD;
	obs = obs.to(device, true);
	return models["geo_v"]->Forward(obs, config.useHalfPrecision).flatten().to(torch::kFloat32);
}

torch::Tensor GGL::PPOLearner::HullBootstrap(torch::Tensor states) {
	// The hull candidates for each row of `states` (the NEXT states of the buffer): the
	// hullK nearest donor states in chart space lend their eps-scaled witnessed
	// displacement vectors; return max_j InferVdagMin(states + eps * delta_j).
	// OPEN (actuation-seat) configuration deliberately -- no donor-radius or likelihood
	// gates: this feeds training optimism, whose thin-record excess is exploration
	// pressure bounded by SIL's realized-conversion requirement (EPSILON_CRITIC.md s7,
	// the seat theorem). Cost: hullK extra InferVdagMin passes + one cdist per chunk.
	RG_NO_GRAD;
	if (!config.hullEnabled || !models["hull_proj"] || geoResFill < 4096)
		return {};
	int64_t nR = states.size(0);
	int64_t sub = RS_MIN((int64_t)config.hullBankSub, geoResFill);
	auto bidx = torch::randint(0, geoResFill, { sub }, torch::TensorOptions().dtype(torch::kLong));
	auto bankObs = geoResObs.index_select(0, bidx).to(device, true);
	auto bankDelta = (geoResNext.index_select(0, bidx) - geoResObs.index_select(0, bidx))
		.to(device, true);
	// residual teleport guard on top of the feed-side mask: drop the extreme tail of
	// donated displacement norms (a data-driven bound, no domain knowledge)
	auto dNorm = bankDelta.norm(2, 1);
	auto normCap = dNorm.quantile(0.995);
	bankDelta = bankDelta * (dNorm <= normCap).to(torch::kFloat32).unsqueeze(1);
	auto bankCoords = models["hull_proj"]->Forward(bankObs, false).to(torch::kFloat32);

	auto out = torch::empty({ nR }, torch::kFloat32);
	constexpr int64_t HCH = 32768;
	for (int64_t i0 = 0; i0 < nR; i0 += HCH) {
		int64_t i1 = RS_MIN(i0 + HCH, nR);
		auto q = states.slice(0, i0, i1).to(device, true).to(torch::kFloat32);
		auto qc = models["hull_proj"]->Forward(q, false).to(torch::kFloat32);
		auto nnk = std::get<1>(torch::cdist(qc, bankCoords)
			.topk(RS_MIN((int64_t)config.hullK, sub), 1, /*largest=*/false));
		torch::Tensor best;
		for (int64_t j = 0; j < nnk.size(1); j++) {
			auto pert = q + config.hullEps * bankDelta.index_select(0, nnk.select(1, j));
			auto v = InferVdagMin(pert);
			best = best.defined() ? torch::maximum(best, v) : v;
		}
		out.slice(0, i0, i1).copy_(best.to(torch::kCPU, torch::kFloat32));
	}
	return out;
}

void GGL::PPOLearner::GeoReservoirAdd(torch::Tensor obs, torch::Tensor nextObs, torch::Tensor rew, int cap,
	torch::Tensor keepMask, torch::Tensor ret) {
	// Uniform reservoir over the whole run. Vectorised: a per-row loop over a 6144-row buffer
	// dominated runtime in the offline harness.
	// keepMask (optional, float 0/1): rows with 0 are EXCLUDED before insertion. The caller
	// passes the continuation mask so pairs that cross an episode reset (goal -> kickoff
	// teleports) never enter: they are not executed dynamics, and both Sigma (geo) and the
	// hull donor bank would otherwise learn teleport displacements as reachable (measured in
	// the offline testbed: 4x ball-dim slack inflation without the filter).
	RG_NO_GRAD;
	if (keepMask.defined()) {
		auto sel = (keepMask.to(torch::kCPU, torch::kFloat32).flatten() > 0.5f)
			.nonzero().flatten();
		if (sel.numel() == 0) return;
		obs = obs.index_select(0, sel.to(obs.device()));
		nextObs = nextObs.index_select(0, sel.to(nextObs.device()));
		rew = rew.flatten().index_select(0, sel.to(rew.device()));
		if (ret.defined())
			ret = ret.flatten().index_select(0, sel.to(ret.device()));
	}
	int64_t n = obs.size(0);
	if (n <= 0 || cap <= 0) return;
	auto o = obs.to(torch::kCPU, torch::kFloat32);
	auto no = nextObs.to(torch::kCPU, torch::kFloat32);
	auto r = rew.to(torch::kCPU, torch::kFloat32).flatten();
	if (!geoResObs.defined()) {
		geoResObs = torch::zeros({ (int64_t)cap, o.size(1) }, torch::kFloat32);
		geoResNext = torch::zeros({ (int64_t)cap, o.size(1) }, torch::kFloat32);
		geoResRew = torch::zeros({ (int64_t)cap }, torch::kFloat32);
		geoResRet = torch::zeros({ (int64_t)cap }, torch::kFloat32);
	}
	auto rt = ret.defined() ? ret.to(torch::kCPU, torch::kFloat32).flatten()
		: torch::zeros({ n }, torch::kFloat32);
	int64_t take = RS_MIN((int64_t)cap - geoResFill, n);
	if (take > 0) {
		geoResObs.slice(0, geoResFill, geoResFill + take).copy_(o.slice(0, 0, take));
		geoResNext.slice(0, geoResFill, geoResFill + take).copy_(no.slice(0, 0, take));
		geoResRew.slice(0, geoResFill, geoResFill + take).copy_(r.slice(0, 0, take));
		geoResRet.slice(0, geoResFill, geoResFill + take).copy_(rt.slice(0, 0, take));
		geoResFill += take;
	}
	int64_t rest = n - take;
	if (rest > 0) {
		auto idx = torch::randint(0, cap, { rest }, torch::TensorOptions().dtype(torch::kLong));
		auto keep = torch::rand({ rest }) < ((float)cap / (float)(geoResSeen + n));
		auto sel = keep.nonzero().flatten();
		if (sel.numel() > 0) {
			auto dst = idx.index_select(0, sel);
			geoResObs.index_copy_(0, dst, o.slice(0, take, n).index_select(0, sel));
			geoResNext.index_copy_(0, dst, no.slice(0, take, n).index_select(0, sel));
			geoResRew.index_copy_(0, dst, r.slice(0, take, n).index_select(0, sel));
			geoResRet.index_copy_(0, dst, rt.slice(0, take, n).index_select(0, sel));
		}
	}
	geoResSeen += n;
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
	int doWm = (nR >= 256) ? 1 : 0;
	if (dist && dist->distributed())
		dist->min_host(&doWm, 1);
	if (!doWm)
		return;

	torch::Tensor sel;
	{
		RG_NO_GRAD;
		// cont[i] == 1 means row i+1 IS the successor of row i (no episode boundary)
		auto ok = (cont.slice(0, 0, nR - 1) > 0.5f).nonzero().flatten().to(torch::kCPU);
		int haveOk = (ok.numel() >= 128) ? 1 : 0;
		if (dist && dist->distributed())
			dist->min_host(&haveOk, 1);
		if (!haveOk)
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
		if (dist) {
			if (models["wm_dyn1"]) models["wm_dyn1"]->AllReduceGrads(dist);
			if (models["wm_dyn2"]) models["wm_dyn2"]->AllReduceGrads(dist);
		}
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
	int64_t nR = states.size(0);
	int doImag = (rhatMaxObserved > 0.f && nR >= 256) ? 1 : 0;
	if (dist && dist->distributed())
		dist->min_host(&doImag, 1);
	if (!doImag)
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
		if (dist) {
			if (models["wm_v1"]) models["wm_v1"]->AllReduceGrads(dist);
			if (models["wm_v2"]) models["wm_v2"]->AllReduceGrads(dist);
		}
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

torch::Tensor GGL::PPOLearner::InferGoalCritic(torch::Tensor obs) {
	// Reads the critic trunk when one exists (its gradient therefore reaches shared perception —
	// see PPOLearnerConfig::criticTrunk); otherwise the original independent raw-obs path.
	if (models["critic_trunk"])
		obs = ValueTrunk(obs, config.useHalfPrecision);
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
		avgVdagTwinSpread,
		avgRhatLoss;

	// Save parameters first
	auto policyBefore = models["policy"]->CopyParams();
	auto criticBefore = models["critic"]->CopyParams();
	// V-dagger: the direct "are these weights actually moving?" signal. Frozen at lr=0 this
	// read exactly 0 for the life of the run, which nothing would have revealed.
	auto vdagBefore = models["vdag1"] ? models["vdag1"]->CopyParams() : torch::Tensor();

	bool trainPolicy = config.policyLR != 0;
	bool trainCritic = config.criticLR != 0;
	bool trainSharedHead = models["shared_head"] && (trainPolicy || trainCritic);

	// GEOMETRY world-facing fits, ONCE per Learn call. r_hat and Sigma are STATIONARY targets
	// fit on the reservoir, so they neither need nor benefit from a pass per PPO minibatch:
	// the previous form retrained them epochs x minibatches (16x) per iteration, whose
	// reservoir sampling alone moved ~740 MB over PCIe per iteration for two tiny supervised
	// regressions. A fixed 4-chunk budget matches the offline validation's per-iteration dose;
	// Geo/Residual and Geo/Rew Loss are the panels that would show it starving. The HJB
	// residual stays per-minibatch below - the FIELD must track the policy's states; the world
	// does not move. The geo nets touch no shared parameters, so stepping them here is
	// independent of the main backward.
	int doGeoFit = (models["geo_v"] && geoResFill > 1024) ? 1 : 0;
	if (dist && dist->distributed())
		dist->min_host(&doGeoFit, 1);
	if (doGeoFit) {
		torch::Tensor sigAcc, rewAcc;
		int64_t chunk = RS_MIN((int64_t)config.miniBatchSize, geoResFill);
		constexpr int GEO_FIT_CHUNKS = 4;
		for (int k = 0; k < GEO_FIT_CHUNKS; k++) {
			auto ridx = torch::randint(0, geoResFill, { chunk },
				torch::TensorOptions().dtype(torch::kLong));
			auto ro = geoResObs.index_select(0, ridx).to(device, true);
			auto rn = geoResNext.index_select(0, ridx).to(device, true);
			auto rr = geoResRew.index_select(0, ridx).to(device, true).flatten();
			int64_t od = ro.size(1);

			// Sigma: Gaussian NLL of the displacement. Predicting the mean as well makes the
			// std the spread AROUND the drift (what the action buys) rather than total motion
			// (which the world supplies regardless of what we do).
			auto sOut = models["geo_sigma"]->Forward(ro, false).to(torch::kFloat32);
			auto sMu = sOut.slice(1, 0, od);
			auto sLs = sOut.slice(1, od, 2 * od).clamp(-8.f, 2.f);
			auto dlt = rn - ro;
			auto sigLoss = (sLs + 0.5f * ((dlt - sMu) / sLs.exp()).pow(2)).mean();

			auto rPred = models["geo_rew"]->Forward(rn, false).flatten().to(torch::kFloat32);
			auto rewLoss = mseLoss(rPred, rr);

			(sigLoss + rewLoss).backward();
			if (dist) {
				if (models["geo_sigma"]) models["geo_sigma"]->AllReduceGrads(dist);
				if (models["geo_rew"]) models["geo_rew"]->AllReduceGrads(dist);
			}
			nn::utils::clip_grad_norm_(models["geo_sigma"]->parameters(), 1.0f);
			nn::utils::clip_grad_norm_(models["geo_rew"]->parameters(), 1.0f);
			models["geo_sigma"]->StepOptim();
			models["geo_rew"]->StepOptim();

			sigAcc = sigAcc.defined() ? sigAcc + sigLoss.detach() : sigLoss.detach();
			rewAcc = rewAcc.defined() ? rewAcc + rewLoss.detach() : rewLoss.detach();
		}
		// One sync for the panel value, outside the minibatch hot loop.
		dbgGeoRew = (rewAcc / (float)GEO_FIT_CHUNKS).cpu().item<float>();
	}

	// HULL chart fits, ONCE per Learn call (same budget logic as the geo fits above: the
	// chart is a world-facing stationary fit on the reservoir). Loss = displacement NLL
	// through the chart bottleneck + L1 on the projection weights: the NLL forces the chart
	// to keep exactly the obs dims that CONDITION the local dynamics; the L1 deletes the
	// rest. Audit by reading hull_proj's first-layer weights (Hull/Proj L1 panel tracks the
	// surviving mass).
	int doHullFit = (config.hullEnabled && models["hull_proj"] && geoResFill > 1024) ? 1 : 0;
	if (dist && dist->distributed())
		dist->min_host(&doHullFit, 1);
	if (doHullFit) {
		torch::Tensor nllAcc;
		int64_t chunk = RS_MIN((int64_t)config.miniBatchSize, geoResFill);
		constexpr int HULL_FIT_CHUNKS = 4;
		for (int k = 0; k < HULL_FIT_CHUNKS; k++) {
			auto ridx = torch::randint(0, geoResFill, { chunk },
				torch::TensorOptions().dtype(torch::kLong));
			auto ro = geoResObs.index_select(0, ridx).to(device, true);
			auto rn = geoResNext.index_select(0, ridx).to(device, true);
			int64_t od = ro.size(1);
			auto coords = models["hull_proj"]->Forward(ro, false).to(torch::kFloat32);
			auto hOut = models["hull_head"]->Forward(coords, false).to(torch::kFloat32);
			auto hMu = hOut.slice(1, 0, od);
			auto hLs = hOut.slice(1, od, 2 * od).clamp(-8.f, 2.f);
			auto dlt = rn - ro;
			auto nll = (hLs + 0.5f * ((dlt - hMu) / hLs.exp()).pow(2)).mean();
			torch::Tensor l1;
			for (auto& p : models["hull_proj"]->parameters())
				if (p.dim() == 2)
					l1 = l1.defined() ? l1 + p.abs().mean() : p.abs().mean();
			auto loss = l1.defined() ? nll + config.hullChartL1 * l1 : nll;
			loss.backward();
			if (dist) {
				if (models["hull_proj"]) models["hull_proj"]->AllReduceGrads(dist);
				if (models["hull_head"]) models["hull_head"]->AllReduceGrads(dist);
			}
			nn::utils::clip_grad_norm_(models["hull_proj"]->parameters(), 1.0f);
			nn::utils::clip_grad_norm_(models["hull_head"]->parameters(), 1.0f);
			models["hull_proj"]->StepOptim();
			models["hull_head"]->StepOptim();
			nllAcc = nllAcc.defined() ? nllAcc + nll.detach() : nll.detach();
			if (k == HULL_FIT_CHUNKS - 1 && l1.defined())
				dbgHullL1 = l1.detach().cpu().item<float>();
		}
		dbgHullNLL = (nllAcc / (float)HULL_FIT_CHUNKS).cpu().item<float>();
	}
	// HJB debug accumulators: summed as GPU tensors across minibatches, synced ONCE after the
	// epoch loop. The previous form did three .item() syncs per minibatch (48 per iteration)
	// in the middle of the learn pass.
	torch::Tensor geoResidAcc, geoMeanAcc;
	int geoMbCount = 0;

	// Device-side metric sums: one .item() at the end of Learn, not per minibatch
	// (GGL-2). Per-minibatch .cpu().item() was a stream drain in the same class as
	// collect's old tActions.cpu().
	auto metricOpts = torch::TensorOptions().dtype(torch::kFloat32).device(
		device.is_cuda() ? device : torch::kCPU);
	auto zmet = [&]() { return torch::zeros({}, metricOpts); };
	torch::Tensor sumEntropy = zmet(), sumRatio = zmet(), sumPolicyLoss = zmet();
	torch::Tensor sumRelEntropyLoss = zmet(), sumCriticLoss = zmet(), sumGoalCriticLoss = zmet();
	torch::Tensor sumGuidingLoss = zmet(), sumClip = zmet(), sumDivergence = zmet();
	torch::Tensor sumVdagLoss = zmet(), sumVdagTwinSpread = zmet(), sumRhatLoss = zmet();
	torch::Tensor sumReachLoss = zmet(), sumReachCarStateLoss = zmet();
	torch::Tensor lastEntGate, lastSilLoss, lastAuxNLL, lastYvAbs, lastVdagRaw;
	torch::Tensor nRelEntropy = zmet();
	int metricPolicySteps = 0, metricCriticSteps = 0, metricGoalSteps = 0;
	int metricKlSteps = 0, metricClipSteps = 0, metricGuidingSteps = 0;
	int metricVdagSteps = 0, metricTwinSteps = 0, metricRhatSteps = 0;
	int metricReachSteps = 0, metricReachCSSteps = 0;

	// GGL_CONSUME_TIMERS: component profile of the learn pass, for cadence work where
	// Learn is fixed-cost-dominated at small row counts. The boundary synchronize()s
	// perturb the thing being measured — debug flag only, never on in production.
	static const bool consumeTimers = [] {
		const char* e = std::getenv("GGL_CONSUME_TIMERS");
		return e && *e && std::string(e) != "0";
	}();
	double tShuffle = 0, tFwdBwd = 0, tAllReduce = 0, tClip = 0, tOptStep = 0;
	auto fnSyncNow = [&] { if (consumeTimers && device.is_cuda()) torch::cuda::synchronize(); };

	// ADVANTAGE FILTERING AS A ROW SUBSET (config.advFilterSubset). Materialize the kept rows
	// ONCE for the whole Learn call, then run every epoch on them. Built with the buffer's own
	// _GetSamples, which zips fields POSITIONALLY over ExperienceTensors::begin()/end() — so a
	// field added later is carried automatically and cannot be silently left at full length,
	// which is the failure mode a hand-written index_select per field would invite.
	// Materialize the rollout on the training device once. Every epoch then shuffles
	// and slices device tensors instead of re-uploading each minibatch.
	Timer uploadTimer = {};
	experience.UploadToDevice();
	report["PPO/Upload Time"] = uploadTimer.Elapsed();

	ExperienceBuffer* learnExp = &experience;
	ExperienceBuffer filteredExp((int)experience.rng(), device);
	// Fixed-step collect yields floor(tsPerItr/players)*players rows — always slightly
	// UNDER batchSize (200k config: 199,680 rows), so the full-buffer path alone yields
	// ZERO batches from GetAllBatchesShuffled's `startIdx + batchSize <= expSize` test.
	// Historically the advFilterSubset path masked this on every iteration; the first
	// iteration where a rank's subset fell below one minibatch (easy at small
	// GGL_TS_PER_ITR) hit the unclamped fallback and the 0-batch guard killed the fleet
	// (job 4630792). Clamp so the buffer's actual rows form one full batch — the
	// minibatch split below already handles arbitrary sizes, exactly as it does nKept.
	int64_t learnBatchSize = RS_MIN(config.batchSize, experience.data.states.size(0));
	double tOptim = 0, tStepOk = 0; // async-path step timers (APPO merge)
	float dbgSubsetRows = -1.f;
	if (config.advFilterSubset && experience.data.advFilterMask.defined()) {
		auto keepCpu = experience.data.advFilterMask.to(torch::kCPU).flatten();
		auto keptIdx = torch::nonzero(keepCpu > 0.5f).flatten().to(torch::kLong).contiguous();
		int64_t nKept = keptIdx.numel();
		// Async shards can be ~2133 rows; requiring a full miniBatchSize (12500) would
		// never subset. DDP keeps the minibatch floor so a tiny keep-set cannot empty Learn.
		int64_t subsetFloor = (dist && dist->n_collectors() > 0) ? 8 : (int64_t)config.miniBatchSize;
		int useSubset = (nKept >= subsetFloor) ? 1 : 0;
		if (dist && dist->distributed())
			dist->min_host(&useSubset, 1);
		if (useSubset) {
			auto sampleIdx = experience.data.IsOnCUDA()
				? keptIdx.to(experience.data.states.device())
				: keptIdx;
			filteredExp.data = experience._GetSamples(sampleIdx);
			learnExp = &filteredExp;
			// One batch holding every kept row; the minibatch loop below splits it by
			// miniBatchSize exactly as it does the full buffer. Passing config.batchSize here
			// instead would yield ZERO batches whenever nKept < batchSize.
			learnBatchSize = nKept;
			dbgSubsetRows = (float)nKept;
		} else {
			RG_LOG("AdvFilterSubset: only " << nKept << " kept rows (< floor "
				<< subsetFloor << ") - training on the FULL buffer this iteration");
		}
	}

	for (int epoch = 0; epoch < config.epochs; epoch++) {

		// Get randomly-ordered timesteps for PPO
		dbgVdagRows = experience.data.vdagTargets.defined() ? (float)experience.data.vdagTargets.numel() : -2.f;
		dbgRhatEntry = experience.data.rhatTargets.defined() ? (float)experience.data.rhatTargets.numel() : -2.f;
		fnSyncNow();
		Timer shuffleTimer = {};
		auto batches = learnExp->GetAllBatchesShuffled(learnBatchSize, config.overbatching);
		fnSyncNow();
		tShuffle += shuffleTimer.Elapsed();
		if (dist && dist->distributed()) {
			int bc = (int)batches.size();
			dist->min_host(&bc, 1);
			if (bc <= 0)
				RG_ERR_CLOSE("PPOLearner: a rank has 0 batches; NCCL all-reduces would hang");
			if ((int)batches.size() > bc)
				batches.resize((size_t)bc);
		}
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

			// ADVANTAGE FILTERING: rows outside the top advFilterFrac (threshold computed
			// buffer-wide in Learner.cpp) are dropped from the POLICY loss only. The batch-wide
			// kept count is the denominator every minibatch shares, so the accumulated policy
			// gradient is the exact mean over the batch's kept rows - filtering changes WHICH
			// rows teach, not the step size (a per-minibatch mean would instead have scaled the
			// effective policy LR by the kept fraction, i.e. a silent LR cut disguised as a
			// selection rule). The scalar reduction is read once per batch; minibatches keep
			// the mask on the training device with the rest of the experience.
			torch::Tensor batchAdvFilter = batch.advFilterMask;
			float batchKeptRows = (float)curBatchSize;
			if (batchAdvFilter.defined())
				batchKeptRows = RS_MAX(batchAdvFilter.sum().item<float>(), 1.f);

			auto fnRunMinibatch = [&](int64_t start, int64_t stop) {

				float batchSizeRatio = (stop - start) / (float)curBatchSize;

				auto take = [&](torch::Tensor t) {
					if (!t.defined())
						return t;
					auto s = t.slice(0, start, stop);
					if (s.device() == device)
						return s;
					return s.to(device, /*non_blocking=*/true, /*copy=*/true);
				};
				auto acts = take(batchActs);
				auto obs = take(batchObs);
				auto actionMasks = take(batchActionMasks);
				auto advantages = take(batchAdvantages);
				auto oldProbs = take(batchOldProbs);
				auto targetValues = take(batchTargetValues);

				// Advantage filtering (policy only; see the note above the minibatch lambda)
				torch::Tensor advKeep, keepSum;
				if (batchAdvFilter.defined()) {
					advKeep = take(batchAdvFilter);
					keepSum = advKeep.sum().clamp_min(1);
				}

				// Learn autocast covers the dense forwards below (policy, critic, goal critic,
				// V-dagger twins). It is paused around the geo HJB and the InfoNCE heads, and
				// ends before totalLoss.backward(). FP16 on V100; BF16 on sm_80+.
				AutocastScope autocast(config.learnAutocastBF16 && device.is_cuda());

				// THE one main-trunk forward for this minibatch, shared by the policy head and
				// the whole value family (see the note above fnValueTrunk below for why it is
				// declared up here rather than next to its value-side users).
				torch::Tensor trunkVR, valueTrunkVR;
				auto fnTrunkVR = [&]() -> torch::Tensor& {
					if (!trunkVR.defined())
						trunkVR = models["shared_head"]
							? models["shared_head"]->Forward(obs, false) : obs;
					return trunkVR;
				};

				torch::Tensor probs, logProbs, entropy, ratio, clipped, policyLoss, ppoLoss;
				if (trainPolicy) {

					// Get policy log probs and entropy
					torch::Tensor curEntropyT;
					{
						probs = InferPolicyProbsFromModels(models, obs, actionMasks, config.policyTemperature, false,
							{}, nullptr, fnTrunkVR());
						logProbs = probs.log().gather(-1, acts.unsqueeze(-1));
						auto entRows = ComputeEntropyRows(probs, actionMasks, config.maskEntropy);
						// Report the UNWEIGHTED mean so the panel stays comparable to runs
						// without the gate.
						curEntropyT = entRows.mean().detach();
						sumEntropy += curEntropyT;
						metricPolicySteps++;
						// H-GATED ENTROPY: sample harder where the critic says there is
						// unrealised value. Only ever ADDS stochasticity -> the entropy
						// floor is strengthened by construction.
						if (batch.entWeights.defined()) {
							auto ew = take(batch.entWeights).view_as(entRows);
							entropy = (entRows * ew).mean();
							lastEntGate = ew.mean().detach();
						} else {
							entropy = entRows.mean();
						}
					}

					logProbs = logProbs.view_as(oldProbs);

					// Compute PPO loss
					ratio = exp(logProbs - oldProbs);
					sumRatio += ratio.mean().detach();
					clipped = clamp(
						ratio, 1 - config.clipRange, 1 + config.clipRange
					);

					// Compute policy loss (mean over the rows the policy is allowed to train on:
					// all of them, or the advantage-filtered subset)
					auto ppoPerRow = -min(
						ratio * advantages, clipped * advantages
					);
					// policyLossRatio is this minibatch's share of the batch-wide mean; with no
					// filter that is just its row share (the original batchSizeRatio)
					torch::Tensor policyLossScale = torch::full({}, batchSizeRatio, metricOpts);
					if (advKeep.defined()) {
						policyLoss = (ppoPerRow * advKeep.view_as(ppoPerRow)).sum() / keepSum;
						policyLossScale = keepSum / batchKeptRows;
					} else {
						policyLoss = ppoPerRow.mean();
					}
					auto curPolicyLossT = policyLoss.detach();
					sumPolicyLoss += curPolicyLossT;

					// A fully-filtered minibatch (no kept rows) gives policyLoss == 0 exactly, and
					// one inf would poison this tracker for the whole report
					sumRelEntropyLoss += torch::where(
						curPolicyLossT != 0,
						(curEntropyT * config.entropyScale) / curPolicyLossT,
						torch::zeros({}, curPolicyLossT.options()));
					nRelEntropy += (curPolicyLossT != 0).to(torch::kFloat32);

					// Entropy keeps the unfiltered row share: it regularizes the policy over the
					// whole visited state distribution, not just the rows that scored well.
					ppoLoss = policyLoss * policyLossScale
						- entropy * config.entropyScale * batchSizeRatio;

					// SELF-IMITATION: positive-only BC on conversion rows (weights computed at
					// learn-prep; see the silEnabled block in Learner.cpp). Mean over the
					// minibatch's conversion rows, batchSizeRatio-scaled so gradient
					// accumulation matches a full-batch pass (the guiding-loss convention).
					if (batch.silWeights.defined()) {
						auto sw = take(batch.silWeights).view_as(logProbs);
						auto nConv = sw.count_nonzero().to(torch::kFloat32);
						auto silLoss = (-(logProbs.flatten()) * sw.flatten()).sum()
							/ nConv.clamp_min(1) * config.silCoeff * batchSizeRatio;
						silLoss = silLoss * (nConv > 0).to(silLoss.dtype());
						lastSilLoss = silLoss.detach();
						ppoLoss = ppoLoss + silLoss;
					}

					if (config.useGuidingPolicy) {
						torch::Tensor guidingProbs;
						{
							RG_NO_GRAD;
							guidingProbs = InferPolicyProbsFromModels(guidingPolicyModels, obs, actionMasks, config.policyTemperature, config.useHalfPrecision);
						}

						auto guidingLoss = (guidingProbs - probs).abs().mean();
						sumGuidingLoss += guidingLoss.detach();
						metricGuidingSteps++;
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
					keepRow = 1.0f - take(batchPracticeMask);

				auto fnMaskedMSE = [&](torch::Tensor pred, torch::Tensor target) {
					if (!keepRow.defined())
						return mseLoss(pred, target);
					auto keep = keepRow.view_as(target);
					return ((pred - target).square() * keep).sum() / keep.sum().clamp_min(1);
				};

				// ONE forward of the value-side body (main trunk -> critic trunk) serves EVERY
				// value head below: critic, goal critic, V-dagger twins, r-hat twins. Autograd
				// accumulates their trunk gradients identically whether they share a graph or
				// not - but a separate forward per head RETAINS a separate full-minibatch
				// activation set per head, and that product (heads x depth x width x rows) is
				// what sets the learn-pass memory peak and therefore the miniBatchSize ceiling.
				// Before this, InferCritic re-forwarded the main trunk on its own while the
				// V-dagger family used its own copy, so the trunk was materialized twice per
				// minibatch for nothing.
				// 2026-08-04: the SAME duplication survived on the POLICY side and was missed by
				// that pass — InferPolicyProbsFromModels takes RAW obs and runs shared_head
				// itself, so the main trunk was still forwarded (and backpropped) TWICE per
				// minibatch: once for the policy, once for the value family. The policy call
				// above now passes fnTrunkVR() in, so there is exactly one trunk forward and,
				// because autograd accumulates both consumers' gradients at that one tensor
				// before traversing it, exactly one trunk BACKWARD. Worth ~14% of the learn
				// pass (trunk = 3.57M of 25.5M MAC/row when double-counted) and it is
				// mathematically identical — same weights, same input, same graph, just not
				// built twice.
				auto fnValueTrunk = [&]() -> torch::Tensor& {
					if (!valueTrunkVR.defined()) {
						torch::Tensor& t = fnTrunkVR();
						valueTrunkVR = models["critic_trunk"]
							? models["critic_trunk"]->Forward(t, false) : t;
					}
					return valueTrunkVR;
				};

				torch::Tensor criticLoss;
				if (trainCritic) {
					// COMPOSITE VALUE CRITIC (PPOLearnerConfig): optional per-row opponent
					// conditioning, twin heads on disjoint minibatch halves, and a mirrored
					// pass on a subsample. All flag-gated; flags off = the original path.
					torch::Tensor vtL = fnValueTrunk();
					if (config.oppCondEnabled && models["opp_embed"] && batch.oppCtx.defined()) {
						auto ctx = take(batch.oppCtx);
						vtL = vtL + models["opp_embed"]->Forward(ctx, false);
					}
					auto tFlat = targetValues.flatten();
					int64_t nRows = tFlat.size(0);
					if (config.valueTwinEnabled && models["critic2"] && nRows >= 4) {
						// disjoint halves: uncorrelated sample noise; the mean readout
						// (InferValueFamily) is the cancelling operation
						int64_t h = nRows / 2;
						auto v1 = models["critic"]->Forward(vtL.slice(0, 0, h), false).flatten();
						auto v2 = models["critic2"]->Forward(vtL.slice(0, h, nRows), false).flatten();
						criticLoss = 0.5f * (mseLoss(v1, tFlat.slice(0, 0, h))
							+ mseLoss(v2, tFlat.slice(0, h, nRows))) * batchSizeRatio;
					} else {
						auto vals = models["critic"]->Forward(vtL, false).flatten();
						vals = vals.view_as(targetValues);
						criticLoss = mseLoss(vals, targetValues) * batchSizeRatio;
					}
					// mirrored pass (exact game symmetry, same targets): a fresh spine
					// forward on x-mirrored obs for a subsample of rows
					if (config.valueMirrorEnabled && mirrorMap.IsValid()) {
						int64_t mRows = (int64_t)(nRows * config.valueMirrorFrac);
						if (mRows >= 4) {
							auto mObs = ObsMirror::Apply(mirrorMap, obs.slice(0, 0, mRows));
							auto mTrunk = models["shared_head"]
								? models["shared_head"]->Forward(mObs, false) : mObs;
							auto mVt = models["critic_trunk"]
								? models["critic_trunk"]->Forward(mTrunk, false) : mTrunk;
							if (config.oppCondEnabled && models["opp_embed"] && batch.oppCtx.defined()) {
								auto ctx = take(batch.oppCtx).slice(0, 0, mRows);
								mVt = mVt + models["opp_embed"]->Forward(ctx, false);
							}
							auto tM = tFlat.slice(0, 0, mRows);
							torch::Tensor mLoss;
							if (config.valueTwinEnabled && models["critic2"]) {
								int64_t mh = mRows / 2;
								mLoss = 0.5f * (mseLoss(models["critic"]->Forward(
										mVt.slice(0, 0, mh), false).flatten(), tM.slice(0, 0, mh))
									+ mseLoss(models["critic2"]->Forward(
										mVt.slice(0, mh, mRows), false).flatten(), tM.slice(0, mh, mRows)));
							} else {
								mLoss = mseLoss(models["critic"]->Forward(mVt, false).flatten(), tM);
							}
							criticLoss = criticLoss + mLoss * (config.valueMirrorFrac * batchSizeRatio);
						}
					}
					sumCriticLoss += criticLoss.detach();
					metricCriticSteps++;
				}

				// AUX DISPLACEMENT HEAD (composite value critic): one-step displacement
				// NLL on the SHARED trunk output -- representation pressure. Targets are
				// built at learn-prep (boundary rows masked); gradients flow into the trunk,
				// which is the point (the toy's single largest training lever).
				torch::Tensor auxDispLoss;
				if (config.auxDispEnabled && models["aux_disp"] && batch.auxDispTargets.defined()) {
					auto tgt = take(batch.auxDispTargets).to(torch::kFloat32);
					auto mask = take(batch.auxDispMask).to(torch::kFloat32).unsqueeze(1);
					auto out = models["aux_disp"]->Forward(fnTrunkVR(), false).to(torch::kFloat32);
					int64_t od = tgt.size(1);
					auto mu = out.slice(1, 0, od);
					auto ls = out.slice(1, od, 2 * od).clamp(-8.f, 2.f);
					auto nll = ((ls + 0.5f * ((tgt - mu) / ls.exp()).pow(2)) * mask).mean();
					auxDispLoss = nll * (config.auxDispWeight * batchSizeRatio);
					lastAuxNLL = nll.detach();
				}

				// Secondary goal-only critic: plain value regression on its own channel. It reads
				// the same value-side body as the main critic (2026-07-29), so unlike every
				// version before it this gradient DOES reach shared perception - the tradeoff is
				// argued in PPOLearnerConfig::criticTrunk. With no critic trunk configured,
				// InferGoalCritic's raw-obs path keeps the old isolation.
				torch::Tensor goalCriticLoss;
				if (batchGoalTargetValues.defined() && models["goal_critic"]) {
					auto goalTargets = take(batchGoalTargetValues);
					auto goalVals = (models["critic_trunk"]
						? models["goal_critic"]->Forward(fnValueTrunk(), false).flatten()
						: InferGoalCritic(obs)).view_as(goalTargets);
					goalCriticLoss = fnMaskedMSE(goalVals, goalTargets) * batchSizeRatio;
					sumGoalCriticLoss += goalCriticLoss.detach();
					metricGoalSteps++;
				}

				// HEADROOM twin V-dagger: expectile regression vs one-iteration-frozen TD
				// targets (precomputed at learn-prep, riding the buffer). Gradients FLOW
				// into the shared trunk (fresh-run co-adaptation; see PPOLearnerConfig).
				// THEORY (r-hat): twin expectile regression on the reward that landed on
				// each arrival state + group-L1 over INPUT features (Occam). Occam is what
				// makes the theory TRANSFER: among theories equally consistent with the
				// data, prefer the one using fewer features, so "this kind of state pays"
				// generalizes to configurations never paid at.
				// Both families read the SAME memoized value-side body as the critic above
				// (fnValueTrunk) - see the note there for why one shared forward matters.
				torch::Tensor rhatLoss;
				if (batch.rhatTargets.defined() && models["rhat1"] && models["rhat2"]) {
					auto yrRaw = take(batch.rhatTargets).flatten();
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
					torch::Tensor& trunkR = fnValueTrunk();
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
					sumRhatLoss += rhatLoss.detach();
					metricRhatSteps++;
				}

				torch::Tensor vdagLoss;
				dbgVdagRows = batch.vdagTargets.defined() ? (float)batch.vdagTargets.numel() : -1.f;
				dbgRhatRows = batch.rhatTargets.defined() ? (float)batch.rhatTargets.numel() : -1.f;
				if (batch.vdagTargets.defined() && models["vdag1"] && models["vdag2"]) {
					auto yv = take(batch.vdagTargets).flatten();
					torch::Tensor& trunkV = fnValueTrunk();
					// TWIN SPREAD: mean |V1 - V2| on the same rows. The min-in-target anti-ratchet
					// only works while the twins are DIFFERENT functions, and as of 2026-07-29
					// they share a critic_trunk and keep just two private layers each - so their
					// disagreement is now a thing that can quietly go to zero. Compare against
					// Headroom/Yv Abs (the target scale): a spread collapsing toward 0 while H
					// climbs is the 0.3 -> 11.8 inflation shape from COMPOSITION_CRITIC.md 4.4.
					torch::Tensor twinPredA;
					for (Model* vh : { models["vdag1"], models["vdag2"] }) {
						auto pred = vh->Forward(trunkV, false).flatten().to(torch::kFloat32);
						if (!twinPredA.defined()) {
							twinPredA = pred.detach();
						} else {
							sumVdagTwinSpread += (twinPredA - pred.detach()).abs().mean();
							metricTwinSteps++;
						}
						auto u = yv - pred;
						auto w = torch::where(u > 0,
							torch::full_like(u, config.vdagTau), torch::full_like(u, 1.f - config.vdagTau));
						auto l = (w * u * u).mean() * batchSizeRatio;
						vdagLoss = vdagLoss.defined() ? vdagLoss + l : l;
					}
					sumVdagLoss += vdagLoss.detach();
					metricVdagSteps++;
					lastVdagRaw = vdagLoss.detach();
					lastYvAbs = yv.abs().mean().detach();
				}

				// ===================== GEOMETRY: the HJB residual =====================
				// (1 - gamma) V(s) = r_hat(s) + gamma * || grad_s V(s) ||_Sigma(s)
				//
				// The gradient here is w.r.t. the INPUT, not the parameters — that is the whole
				// END OF THE AUTOCAST REGION. Everything below — the geo HJB double-backward, the
				// InfoNCE heads, the loss assembly and backward() — runs in fp32 by design.
				autocast.End();

				// cost of this rung: one extra backward through geo_v per minibatch. create_graph
				// is required so the residual stays differentiable in the parameters.
				//
				// r_hat and Sigma are fit on the RESERVOIR (stationary targets, see config); the
				// residual is solved on the CURRENT states, which is where the field is read.
				torch::Tensor geoLoss;
				if (models["geo_v"] && geoResFill > 1024) {
					// HJB residual ONLY - the world-facing fits (Sigma, r_hat) train once per
					// Learn call before the epoch loop, and are read here frozen (no_grad).
					auto gin = obs.detach().clone().set_requires_grad(true);
					auto vg = models["geo_v"]->Forward(gin, false).flatten().to(torch::kFloat32);
					auto grads = torch::autograd::grad({ vg.sum() }, { gin }, {}, true, true);
					auto gx = grads[0];
					int64_t od = gin.size(1);
					torch::Tensor sig, rh;
					{
						RG_NO_GRAD;
						auto so = models["geo_sigma"]->Forward(obs, false).to(torch::kFloat32);
						sig = so.slice(1, od, 2 * od).clamp(-8.f, 2.f).exp() * config.geoSigmaScale;
						rh = models["geo_rew"]->Forward(obs, false).flatten().to(torch::kFloat32);
					}
					// NO advection term: mu is estimated under the POLICY, so including it
					// re-imports the habit this rung exists to see past (measured: drift-on
					// loses the beats-habit property at every sigma).
					auto gnorm = (gx.pow(2) * sig.pow(2)).sum(-1).clamp_min(1e-12f).sqrt();
					// geoGamma, not gaeGamma: the residual's conditioning is tickSkip-fragile
					// and this gamma defines only the field's fixed point, never a PBRS
					// potential — see PPOLearnerConfig::geoGamma.
					float geoG = (config.geoGamma > 0.f) ? config.geoGamma : config.gaeGamma;
					auto resid = (1.f - geoG) * vg - rh - geoG * gnorm;
					auto hjb = resid.pow(2).mean() * batchSizeRatio;

					geoLoss = hjb;
					// No syncs here: accumulate, item once after the epoch loop.
					geoResidAcc = geoResidAcc.defined() ? geoResidAcc + hjb.detach() : hjb.detach();
					auto vgm = vg.detach().mean();
					geoMeanAcc = geoMeanAcc.defined() ? geoMeanAcc + vgm : vgm;
					geoMbCount++;
				}

				// Reachability aux losses (InfoNCE on a subsample; gradient flows into the shared head)
				torch::Tensor reachLoss, carStateLoss;
				if (reach && batch.carHerGoals.defined()) {
					int64_t mbRows = stop - start;
					int64_t sub = RS_MIN((int64_t)config.reachability.infoSubSample, mbRows);
					if (sub > 1) {
						torch::Tensor subIdx = torch::randperm(
							mbRows, TensorOptions().dtype(kLong).device(obs.device())).slice(0, 0, sub);

						torch::Tensor subObs = obs.index_select(0, subIdx);
						torch::Tensor subActs = acts.index_select(0, subIdx);
						torch::Tensor subCarGoals = take(batch.carHerGoals).index_select(0, subIdx);

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
							torch::Tensor subMoved = take(batch.ballMovedMask).index_select(0, subIdx);
							torch::Tensor movedIdx = subMoved.nonzero().flatten();
							if (movedIdx.size(0) > 1) {
								torch::Tensor subBallGoals = take(batch.ballHerGoals)
									.index_select(0, subIdx).index_select(0, movedIdx);
								auto ballRes = reach->ComputeInfoNCELoss(
									reach->psiBall, sa.index_select(0, movedIdx), subBallGoals);
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
							torch::Tensor subCarStateGoals = take(batch.carStateHerGoals)
								.index_select(0, subIdx);
							float couple = config.reachability.carStateCouple;
							torch::Tensor saCS = couple >= 1.f ? sa
								: (couple <= 0.f ? sa.detach()
									: sa.detach() + (sa - sa.detach()) * couple);
							auto carStateRes = reach->ComputeInfoNCELoss(reach->psiCarState, saCS, subCarStateGoals);
							if (carStateRes.loss.defined()) {
								carStateLoss = carStateRes.loss
									* config.reachability.auxLossWeight * batchSizeRatio;
								sumReachCarStateLoss += carStateLoss.detach();
								metricReachCSSteps++;
								avgReachCarStateAcc += carStateRes.categoricalAccuracy;
							}
						}

						if (reachLoss.defined()) {
							reachLoss = (reachLoss + reach->StateActionVarPenalty(sa))
								* config.reachability.auxLossWeight * batchSizeRatio;
							sumReachLoss += reachLoss.detach();
							metricReachSteps++;
						}
					}
				}

				if (trainPolicy) {
					// Compute KL divergence & clip fraction using SB3 method for reporting;
					{
						RG_NO_GRAD;

						auto logRatio = logProbs - oldProbs;
						auto klTensor = (exp(logRatio) - 1) - logRatio;
						sumDivergence += klTensor.mean().detach();
						metricKlSteps++;

						auto clipFraction = mean((abs(ratio - 1) > config.clipRange).to(kFloat));
						sumClip += clipFraction.detach();
						metricClipSteps++;
					}
				}

				// One combined backward so policy, critic and reachability gradients
				// accumulate into the shared head together
				torch::Tensor totalLoss;
				if (trainPolicy)
					totalLoss = ppoLoss;
				if (trainCritic)
					totalLoss = totalLoss.defined() ? totalLoss + criticLoss : criticLoss;
				if (auxDispLoss.defined())
					totalLoss = totalLoss.defined() ? totalLoss + auxDispLoss : auxDispLoss;
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
				if (geoLoss.defined())
					totalLoss = totalLoss.defined() ? totalLoss + geoLoss : geoLoss;

				if (totalLoss.defined()) {
					const bool fp16Amp = config.learnAutocastBF16 && device.is_cuda()
						&& !GGL::GGLCudaHasBF16();
					if (fp16Amp)
						(totalLoss * ampLossScale).backward();
					else
						totalLoss.backward();
				}
			};

			
			Timer fwdBwdTimer = {};
			if (device.is_cpu()) {
				// Just run one minibatch
				fnRunMinibatch(0, curBatchSize);
				if (onMinibatchEnd)
					onMinibatchEnd();
			} else {
				for (int64_t mbs = 0; mbs < curBatchSize; mbs += config.miniBatchSize) {
					int64_t start = mbs;
					int64_t stop = RS_MIN(start + config.miniBatchSize, curBatchSize);
					fnRunMinibatch(start, stop);
					if (onMinibatchEnd)
						onMinibatchEnd();
				}
			}
			fnSyncNow();
			tFwdBwd += fwdBwdTimer.Elapsed();

			Timer allReduceTimer = {};
			if (dist) {
				models.AllReduceGrads(dist, /*includeExempt=*/false);
				// APPO: lets the async learner overlap the NEXT fragment's H2D with the
				// optimizer step; a no-op (unset) on the synchronous path.
				if (onMinibatchEnd)
					onMinibatchEnd();
			}
			fnSyncNow();
			tAllReduce += allReduceTimer.Elapsed();

			const bool fp16Amp = config.learnAutocastBF16 && device.is_cuda()
				&& !GGL::GGLCudaHasBF16();
			if (fp16Amp)
				AmpUnscaleGrads(models, ampLossScale);

			int stepOk = 1;
			if (fp16Amp && !AmpGradsFinite(models))
				stepOk = 0;
			if (dist && dist->distributed()) {
				Timer soTimer = {};
				dist->min_host(&stepOk, 1);
				tStepOk += soTimer.Elapsed();
			}

			if (!stepOk) {
				ampSkipCount++;
				ampGoodEpochs = 0;
				if (ampLossScale > 1.f)
					ampLossScale *= 0.5f;
				if (ampLossScale < 1.f)
					ampLossScale = 1.f;
				AmpZeroGrads(models);
			} else {
				if (fp16Amp) {
					ampGoodEpochs++;
					if (ampGoodEpochs >= 32 && ampLossScale < 65536.f) {
						ampLossScale *= 2.f;
						ampGoodEpochs = 0;
					}
				}

			Timer clipTimer = {};
			if (trainPolicy)
				nn::utils::clip_grad_norm_(models["policy"]->parameters(), 0.5f);
			if (trainCritic)
				nn::utils::clip_grad_norm_(models["critic"]->parameters(), 0.5f);

			if (trainSharedHead)
				nn::utils::clip_grad_norm_(models["shared_head"]->parameters(), 0.5f);

			// Same 0.5 as every other block. It carries the SUM of four value heads' gradients,
			// so it is the last place an unclipped block would be acceptable.
			if (models["critic_trunk"])
				nn::utils::clip_grad_norm_(models["critic_trunk"]->parameters(), 0.5f);

			if (models["goal_critic"])
				nn::utils::clip_grad_norm_(models["goal_critic"]->parameters(), 0.5f);

			// V-dagger twins clip at the same 0.5 as every other head. They were absent from
			// this list, which was harmless only because their LR was 0 - the moment the LR is
			// wired (same commit) 16.1M params would otherwise be the only unclipped block in
			// the model, on an expectile loss whose targets are the widest-scale ones here.
			for (const char* n : { "vdag1", "vdag2" })
				if (models[n])
					nn::utils::clip_grad_norm_(models[n]->parameters(), 0.5f);

			for (const char* n : { "geo_sigma", "geo_rew", "geo_v" })
				if (models[n])
					nn::utils::clip_grad_norm_(models[n]->parameters(), 1.0f);

			if (reach) {
				nn::utils::clip_grad_norm_(reach->phi->parameters(), 0.5f);
				nn::utils::clip_grad_norm_(reach->psiCar->parameters(), 0.5f);
				nn::utils::clip_grad_norm_(reach->psiBall->parameters(), 0.5f);
				if (reach->psiCarState)
					nn::utils::clip_grad_norm_(reach->psiCarState->parameters(), 0.5f);
			}
			tClip += clipTimer.Elapsed();

			fnSyncNow();
			tClip += clipTimer.Elapsed();

			// APPO hook: the async learner publishes weights right before the step
			// swaps them (double-buffer flip). No-op when unset (synchronous path).
			if (onBeforeStepOptims)
				onBeforeStepOptims();
			Timer optStepTimer = {};
			// GGL_MUON_SHARD: Newton-Schulz sharded across ranks + owner broadcast
			// (bit-identical result; lockstep checksum verifies). Profiled 2026-08-15:
			// the replicated NS step was 0.27s of a 0.59s Learn at 8.3k rows/rank.
			static const bool muonShard = [] {
				const char* e = std::getenv("GGL_MUON_SHARD");
				return e && *e && std::string(e) != "0";
			}();
			if (muonShard && dist && dist->distributed())
				models.StepOptimsSharded(dist);
			else
				models.StepOptims();
			// EP: owners publish their expert slices so every learner holds a complete
			// net for weight-publishing/checkpointing. At the ALLREDUCE LANE (this call
			// site), never inside the optimizer's per-param loop — that is what
			// deadlocked GGL_MUON_SHARD against the collector publish comm.
			models.ReplicateExpertSlices(dist);
			fnSyncNow();
			tOptim += optStepTimer.Elapsed();
			tOptStep += optStepTimer.Elapsed();
			if (onMinibatchEnd)
				onMinibatchEnd();
			}
			if (onEpochEnd)
				onEpochEnd(stepOk);
		}
	}

	if (consumeTimers) {
		RG_LOG("[CONSUME] rank_learn shuffle=" << tShuffle << " fwdbwd=" << tFwdBwd
			<< " allreduce=" << tAllReduce << " clip=" << tClip << " optstep=" << tOptStep);
	}

	// MoE router maintenance (research/reports/MOE_POLICY.md): DSv3 aux-free balancing
	// bias update from this learn pass's accumulated expert load, once per iteration.
	// Identical on every rank (loadAcc comes from identical... NO — loads are rank-local.
	// The bias is a BUFFER, not allreduced; small per-rank drift is acceptable for v1 and
	// the collect-side routing it shapes is rank-local anyway. Panel = mean load entropy
	// (1 = perfectly balanced; collapse toward 0 = router collapse, the classic failure).
	if (models["shared_head"]) {
		float entSum = 0.f;
		int nMoe = 0;
		for (size_t i = 0; i < models["shared_head"]->seq->size(); i++)
			if (auto moe = std::dynamic_pointer_cast<GGL::MoEBlockImpl>(models["shared_head"]->seq->ptr(i))) {
				entSum += moe->UpdateRouterBias(1e-3f);
				nMoe++;
			}
		if (nMoe > 0)
			report["MoE/Load Entropy"] = entSum / (float)nMoe;
	}

	if (metricPolicySteps > 0) {
		avgEntropy.Add(sumEntropy.item<float>(), (uint64_t)metricPolicySteps);
		avgRatio.Add(sumRatio.item<float>(), (uint64_t)metricPolicySteps);
		avgPolicyLoss.Add(sumPolicyLoss.item<float>(), (uint64_t)metricPolicySteps);
	}
	{
		float nRel = nRelEntropy.item<float>();
		if (nRel > 0.5f)
			avgRelEntropyLoss.Add(sumRelEntropyLoss.item<float>(), (uint64_t)(nRel + 0.5f));
	}
	if (metricCriticSteps > 0)
		avgCriticLoss.Add(sumCriticLoss.item<float>(), (uint64_t)metricCriticSteps);
	if (metricGoalSteps > 0)
		avgGoalCriticLoss.Add(sumGoalCriticLoss.item<float>(), (uint64_t)metricGoalSteps);
	if (metricGuidingSteps > 0)
		avgGuidingLoss.Add(sumGuidingLoss.item<float>(), (uint64_t)metricGuidingSteps);
	if (metricClipSteps > 0)
		avgClip.Add(sumClip.item<float>(), (uint64_t)metricClipSteps);
	if (metricKlSteps > 0)
		avgDivergence.Add(sumDivergence.item<float>(), (uint64_t)metricKlSteps);
	if (metricVdagSteps > 0)
		avgVdagLoss.Add(sumVdagLoss.item<float>(), (uint64_t)metricVdagSteps);
	if (metricTwinSteps > 0)
		avgVdagTwinSpread.Add(sumVdagTwinSpread.item<float>(), (uint64_t)metricTwinSteps);
	if (metricRhatSteps > 0)
		avgRhatLoss.Add(sumRhatLoss.item<float>(), (uint64_t)metricRhatSteps);
	if (metricReachSteps > 0)
		avgReachLoss.Add(sumReachLoss.item<float>(), (uint64_t)metricReachSteps);
	if (metricReachCSSteps > 0)
		avgReachCarStateLoss.Add(sumReachCarStateLoss.item<float>(), (uint64_t)metricReachCSSteps);
	if (lastEntGate.defined())
		dbgEntGate = lastEntGate.item<float>();
	if (lastSilLoss.defined())
		dbgSilLoss = lastSilLoss.item<float>();
	if (lastAuxNLL.defined())
		dbgAuxNLL = lastAuxNLL.item<float>();
	if (lastYvAbs.defined())
		dbgYvAbs = lastYvAbs.item<float>();
	if (lastVdagRaw.defined())
		dbgVdagRaw = lastVdagRaw.item<float>();

	// GEOMETRY debug panel values: one sync for the whole learn pass (see accumulators above).
	if (geoMbCount > 0) {
		dbgGeoResid = (geoResidAcc / (float)geoMbCount).cpu().item<float>();
		dbgGeoMean = (geoMeanAcc / (float)geoMbCount).cpu().item<float>();
	}

	// Compute magnitude of updates made to the policy and value estimator
	auto policyAfter = models["policy"]->CopyParams();
	auto criticAfter = models["critic"]->CopyParams();

	if (dist && dist->distributed()) {
		if (const char* chk = std::getenv("GGL_DIST_LOCKSTEP_CHECK"); chk && chk[0] && std::string(chk) != "0") {
			float sums[2] = { policyAfter.sum().item<float>(), criticAfter.sum().item<float>() };
			float root[2] = { sums[0], sums[1] };
			dist->bcast_host(root, sizeof(root), 0);
			if (std::abs(sums[0] - root[0]) > 1e-2f || std::abs(sums[1] - root[1]) > 1e-2f)
				RG_ERR_CLOSE("PPOLearner lockstep checksum mismatch rank " << dist->rank()
					<< " policy " << sums[0] << " vs " << root[0]
					<< " critic " << sums[1] << " vs " << root[1]);
			if (dist->rank() == 0)
				RG_LOG("DIST lockstep ok  policy=" << sums[0] << " critic=" << sums[1]);
		}
	}

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
		// Must stay clearly above 0 - read it against Headroom/Yv Abs (the target scale).
		// 0 means the twins have converged to a single function and min(V1,V2) has stopped
		// being a pessimism operator, which is how the seek term inflates with nothing
		// behind it. Added 2026-07-29 with the critic trunk, which is what made twin
		// collapse plausible: they now share their first three layers.
		if (avgVdagTwinSpread.count > 0)
			report["Headroom/Vdag Twin Spread"] = avgVdagTwinSpread.Get();
		// Row counts: -1 means the targets never reached the batch (broken plumbing),
		// which is indistinguishable from "loss is small" on the loss panel alone.
		// -1 = subset filtering off (or it declined to fire this iteration); otherwise the row
		// count the WHOLE update actually ran on. This is the panel that proves the compute
		// lever engaged — AdvFilter/Kept Frac only reports what was selected, not what was fed.
		report["AdvFilter/Subset Rows"] = dbgSubsetRows;
		report["Headroom/Vdag Rows"] = dbgVdagRows;
		report["Headroom/Rhat Rows"] = dbgRhatRows;
		if (dbgEntGate >= 0.f)
			report["Headroom/Ent Gate"] = dbgEntGate;
		if (dbgSilLoss >= 0.f)
			report["SIL/Loss"] = dbgSilLoss;
		if (dbgHullNLL > -900.f)
			report["Hull/Chart NLL"] = dbgHullNLL;
		if (dbgHullL1 >= 0.f)
			report["Hull/Proj L1"] = dbgHullL1;
		if (dbgAuxNLL > -900.f)
			report["Value/Aux Disp NLL"] = dbgAuxNLL;
		if (dbgTwinDisagree >= 0.f)
			report["Value/Twin Disagree"] = dbgTwinDisagree;
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
		if (models["geo_v"]) {
			// Geo/Residual is the HEALTH GATE for this rung. Offline it sat at ~1e-4 while the
			// mechanism worked and grew 100-250x once r_hat drifted under it — which is exactly
			// when every actuation variant stopped helping. A rising trend means the field is no
			// longer V_geo and the injection has become a persistent orthogonal push. Reservoir
			// Fill should saturate at geoReservoir; if it does not, the stationary fits are
			// effectively still on a sliding window and the residual will run away.
			report["Geo/Residual"] = dbgGeoResid;
			report["Geo/Rew Loss"] = dbgGeoRew;
			report["Geo/V Mean"] = dbgGeoMean;
			report["Geo/Reservoir Fill"] = (float)geoResFill;
		}
	}
	}

	// Assemble and return report
	report["Policy Entropy"] = avgEntropy.Get();
	report["Mean KL Divergence"] = avgDivergence.Get();
	if (config.learnAutocastBF16 && device.is_cuda() && !GGL::GGLCudaHasBF16()) {
		report["AMP/Scale"] = ampLossScale;
		report["AMP/Skip"] = (float)ampSkipCount;
	}
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
		// ^ That "cheap enough to run every iteration" claim was wrong, and measurably so.
		// EffectiveRank is an SVD, and the trunk's last Linear is 1280x1280 — on CUDA that
		// dispatches to the iterative Jacobi driver, estimated 40-150ms PER ITERATION, inside
		// PPO Learn Time, for a panel whose value moves ~0.02 per iteration (897.157 ->
		// 897.171 -> 897.179 on three consecutive iterations). It is a slow-moving structural
		// canary, not a per-step signal, so it runs on a cadence now. Dead Units is a cheap
		// column-norm count and stays every iteration.
		{
			Timer plasTimer = {};
			report["Plasticity/Policy Dead Units"] = Plasticity::DeadUnitFraction(models["policy"]);
			// GGL_NO_PLASTICITY skips the SVD panels — the desktop's CUDA-13/12.4
			// franken-stack dlopen-fails in cusolver (cluster unaffected).
			static const bool noPlast = [] {
				const char* e = std::getenv("GGL_NO_PLASTICITY");
				return e && *e && std::string(e) != "0";
			}();
			constexpr uint64_t EFFRANK_EVERY = 32;
			static uint64_t effRankTick = 0;   // telemetry cadence only; Learn() has no counter
			if ((effRankTick++ % EFFRANK_EVERY) == 0 && !noPlast) {
				auto lins = Plasticity::LinearLayers(models["policy"]);
				if (!lins.empty())
					report["Plasticity/Policy EffRank"] = Plasticity::EffectiveRank(lins.back()->weight);
				if (models["shared_head"]) {
					auto tl = Plasticity::LinearLayers(models["shared_head"]);
					if (!tl.empty())
						report["Plasticity/Trunk EffRank"] = Plasticity::EffectiveRank(tl.back()->weight);
				}
			}
			report["Plasticity Time"] = plasTimer.Elapsed();
		}
	}

	report["PPO/AllReduce Time"] = (float)tAllReduce;
	report["PPO/Clip Time"] = (float)tClip;
	report["PPO/Optim Time"] = (float)tOptim;
	report["PPO/StepOk Min Time"] = (float)tStepOk;

}

void GGL::PPOLearner::SaveTo(std::filesystem::path folderPath) {
	models.Save(folderPath);
}

bool GGL::PPOLearner::VerifySavedWeights(std::filesystem::path folderPath) {
	// Behavior-determining nets only — see the call site in Learner::Save for why.
	// A missing model is not a failure (inference-only sets legitimately lack some);
	// a PRESENT model whose file disagrees with memory is.
	for (const char* n : { "shared_head", "policy" }) {
		Model* m = models[n];
		if (m && !m->VerifySavedWeights(folderPath))
			return false;
	}
	return true;
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

	// The critic trunk is value-side only (nothing but value heads read it), so it takes the
	// critic's LR rather than the shared trunk's min(policy, critic). Same trap as the V-dagger
	// twins below: omit it here and Model's ctor leaves its optimizer at lr = 0, which under Muon
	// is an exact no-op - the entire shared value body would sit frozen at random init while
	// every head trained on its output.
	if (models["critic_trunk"])
		models["critic_trunk"]->SetOptimLR(criticLR);

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

	// GEOMETRY nets. Same lr=0 trap as every head above (Model's ctor builds optimizers at
	// zero, which is an exact no-op under Muon) — the fault that left V-dagger frozen at random
	// init for 3.4B steps while its loss still reshaped the trunk. Named explicitly.
	for (const char* n : { "geo_sigma", "geo_rew", "geo_v" })
		if (models[n])
			models[n]->SetOptimLR(config.geoLR);

	// THEORY r-hat twins: same trap (ctor builds optimizers at lr=0 = exact no-op under
	// Muon). They regress a REWARD, not a return, so they take criticLR as well.
	for (const char* n : { "rhat1", "rhat2" })
		if (models[n])
			models[n]->SetOptimLR(criticLR);

	// World-model heads: same ctor lr=0 trap as every other added head.
	for (const char* n : { "wm_dyn1", "wm_dyn2", "wm_v1", "wm_v2" })
		if (models[n])
			models[n]->SetOptimLR(criticLR);

	// HULL chart nets: same ctor lr=0 trap. World-facing fits (displacement NLL), own LR.
	for (const char* n : { "hull_proj", "hull_head" })
		if (models[n])
			models[n]->SetOptimLR(config.hullChartLR);

	// COMPOSITE VALUE CRITIC heads: same ctor lr=0 trap. All value-side -> criticLR.
	for (const char* n : { "critic2", "aux_disp", "opp_embed" })
		if (models[n])
			models[n]->SetOptimLR(criticLR);

	RG_LOG("PPOLearner: " << RS_STR(std::scientific << "Set learning rate to [" << policyLR << ", " << criticLR << "]"));
}

GGL::ModelSet GGL::PPOLearner::GetPolicyModels() {
	ModelSet result = {};
	for (Model* model : models) {
		std::string name = model->modelName;
		// "critic_trunk" is the value-side shared body - value-only, like the heads that read it.
		// Eval/act paths never touch it (InferPolicyProbsFromModels reads shared_head + policy),
		// and at 4.76M params it would otherwise be cloned into all 32 archived versions.
		if (name == "critic" || name == "goal_critic" || name == "critic_trunk")
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

		// HULL chart nets are training-time-only too (acting reads shared_head + policy).
		// Keeping them here made the render hot-swap's strict Load() reject checkpoints from
		// runs that never trained hull (the AiMOS fleet's), stranding the viewer on its boot
		// checkpoint. Same backward-compatibility argument as vdag: dirs that carry the files
		// keep them unread.
		if (name.rfind("hull_", 0) == 0)
			continue;

		result.Add(model);
	}
	return result;
}
