#pragma once
#include "../FrameworkTorch.h"
#include <torch/nn/cloneable.h>
#include <torch/nn/modules/normalization.h>
#include <atomic>
#include <map>
#include <utility>
#include <vector>

namespace GGL {
	namespace Dist { class Session; }

	// Bumped by Model::RefreshHalfCache after every in-place fp16 weight refresh.
	// The MoE fast path keys its transposed-weight caches on it (rebuild once per
	// refresh, not per tick). Defined in Models.cpp.
	extern std::atomic<uint64_t> g_halfRefreshEpoch;

	// DeepSeek-V3-style MoE block for the policy trunk (research/reports/MOE_POLICY.md):
	//   x + SharedExpert(LN(x)) + TopK-routed fine-grained experts(LN(x))
	// - sigmoid router affinities; gate weights = normalized sigmoid among the selected
	//   experts; aux-loss-FREE load balancing via a selection-only bias buffer, updated
	//   +/-gamma against the running load (UpdateRouterBias, called from the learn tail).
	// - Expert banks are 3D stacked params [E, out, in], executed as ONE bmm pair per
	//   block via capacity-factor bucketing — op count independent of E, which is the
	//   design constraint on this dispatch-taxed platform. Autograd handles the sparse
	//   backward (index ops + bmm are differentiable); only touched experts get grads.
	// - Muon treats dim-3 params as batched 2D (batched Newton-Schulz).
	class MoEBlockImpl : public torch::nn::Cloneable<MoEBlockImpl> {
	public:
		int64_t dim, hidden, numExperts, topK;
		float capacityFactor = 1.25f;

		torch::Tensor routerW;                    // [E, d] param
		torch::Tensor expertW1, expertB1;         // [E, h, d], [E, h] params
		torch::Tensor expertW2, expertB2;         // [E, d, h], [E, d] params
		torch::Tensor sharedW1, sharedB1;         // [h, d], [h] params
		torch::Tensor sharedW2, sharedB2;         // [d, h], [d] params
		torch::Tensor routerBias;                 // [E] BUFFER (selection-only, no grads)
		torch::Tensor loadAcc;                    // [E] BUFFER (running load counts)
		torch::nn::LayerNorm ln{ nullptr };

		MoEBlockImpl(int64_t dim, int64_t hidden, int64_t numExperts, int64_t topK);
		MoEBlockImpl() : dim(0), hidden(0), numExperts(0), topK(0) {}

		void reset() override;
		torch::Tensor forward(torch::Tensor x);

		// DSv3 aux-free balancing: bias moves against the accumulated load, then the
		// accumulator decays. Returns the normalized load entropy (1 = perfectly
		// balanced) for the MoE/* panels.
		float UpdateRouterBias(float gamma);

		// GGL_MOE_KERNELS fast path (research/reports/MOE_SPEED.md Stage 1): fused
		// routing + exact-M CUTLASS grouped GEMMs, fp16 no-grad only — the learn pass
		// always takes the autograd baddbmm path above. Selected in forward() by env
		// GGL_MOE_CUTLASS (fastPathForce overrides for the selftest parity check).
		// Members are NOT registered: Cloneable's clone() default-constructs + reset(),
		// so every clone starts with empty caches and builds its own lazily.
		int fastPathForce = -1;                  // -1 = env, 0 = off, 1 = on
		torch::Tensor ForwardFast(torch::Tensor x);
		// Weight caches for the CUTLASS layout, refreshed by copy_ INTO THE SAME
		// STORAGE each epoch — CUDA-graph capture bakes in device pointers, so a
		// refresh must never reallocate (same law as the flat fp16 buffer).
		uint64_t _wCacheEpoch = ~0ull;
		torch::Tensor _w1T, _w2T;                // fp16 [E, d, h], [E, h, d] contiguous
		torch::Tensor _routerW32;                // fp32 [E, d] (routing is fp32; see .h of kernels)
		// PER-SHAPE scratch: a captured graph for batch shape A must keep its exact
		// buffers when shape B arrives — a shared grow-on-demand buffer would leave
		// shape A's graph reading freed storage after B's realloc.
		struct FastScratch {
			torch::Tensor counts, offsets, rowExp, rowGate, srcRows, expertId, gates;
			torch::Tensor logits32, packed, hid, out;
			uint64_t lastUse = 0;
		};
		// BOUNDED (2026-08-17). This map was unbounded, and an entry costs ~128MB per
		// block at learn batch (hid alone is n*hidden fp16 with n = R*topK). Row counts
		// churn in normal operation — Nexto serves 15% of iterations and drives only some
		// cars, so the policy's inference batch varies — so entries accumulated until the
		// job died of CUDA OOM at ~31GB after ~120 iterations (chain 4631804-07, which
		// took all four hops down the same way). Eviction is skipped when CUDA graphs are
		// on, because a captured graph holds raw pointers into its entry (see above).
		std::map<int64_t, FastScratch> _scratchByRows;
		uint64_t _scratchTick = 0;
		void* _gemmCtx = nullptr;                // ggl_moe_ctx_create, grow-only

		// Stage 1b learn path (GGL_MOE_CUTLASS_LEARN): custom autograd routed-FFN with
		// grouped GEMMs. fp16 weight caches for the LEARN side (fp32 master params),
		// invalidated by the params' tensor version counters (bumped by optimizer
		// in-place updates) — weights change twice per iteration, the fn runs ~6x.
		int learnFastForce = -1;                 // -1 = env, 0 = off, 1 = on (selftest)
		torch::Tensor _lw1, _lw1T, _lw2, _lw2T;  // fp16 [E,h,d],[E,d,h],[E,d,h],[E,h,d]
		torch::Tensor _lb1, _lb2, _lBias;        // fp16 [E,h],[E,d],[E]
		uint64_t _lwVer1 = ~0ull, _lwVer2 = ~0ull;
		void* _gemmCtxLearn = nullptr;
		// EP (Stage 2) per-call owner-side state, handed from forward to backward
		// (these are exchange-plan-shaped, not autograd-tensor-shaped).
		int64_t _epPlanRecvTotal = 0;
		torch::Tensor _epRecvPacked, _epRecvHid, _epRecvOffsets, _epRecvLocal;
		std::vector<int> _epSendRows, _epSendDisp, _epRecvRows, _epRecvDisp;
		int _epOwnStart = 0, _epOwnCount = 0, _epNL = 1;
		// Fixed-capacity EP constants (pure functions of E/cap/nL — built once).
		torch::Tensor _epFixedOffsets, _epFixedLocal;
		int _epFixedCap = -1;
	};

	// Stage 1b custom autograd routed-FFN (research/reports/MOE_SPEED.md). Takes the
	// LN output; LN, shared expert, and the residual stay in ordinary autograd.
	torch::Tensor MoERoutedFFNApply(MoEBlockImpl* blk, torch::Tensor xn);

	// ===================== Stage 2: expert parallelism (EP) =====================
	// Set once at learner init (barrier zone). Non-null + group_world() > 1 + env
	// GGL_MOE_EP arms EP: expert e is owned by rank e / (E / nL), tokens all-to-all
	// to owners at learn time, expert grads complete OWNER-LOCAL (no expert-grad
	// allreduce), optimizer steps only owned slices. Learn-only: the no-grad collect
	// fast path (ForwardFast) is untouched.
	void SetMoEExpertParallel(GGL::Dist::Session* session);
	// Armed state, read by the autograd fn and by the optimizer/allreduce hooks.
	bool MoEExpertParallelOn();
	GGL::Dist::Session* MoEExpertParallelSession();
	// Owned expert range for this rank given E; {start, count}. count == E when off.
	std::pair<int, int> MoEOwnedExpertRange(int numExperts);
	// Every [E,...] expert parameter registered by a live MoEBlock, so AllReduceGrads
	// can skip them under EP (they are already owner-local) and the post-step
	// replication knows exactly what to broadcast.
	const std::vector<torch::Tensor>& MoEExpertParams();
	void RegisterMoEExpertParams(const std::vector<torch::Tensor>& params);
	TORCH_MODULE(MoEBlock);

	int RunMoESelfTest();
}
