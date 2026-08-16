#pragma once
#include "../FrameworkTorch.h"
#include <torch/nn/cloneable.h>
#include <torch/nn/modules/normalization.h>
#include <atomic>
#include <map>

namespace GGL {

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
		};
		std::map<int64_t, FastScratch> _scratchByRows;
		void* _gemmCtx = nullptr;                // ggl_moe_ctx_create, grow-only
	};
	TORCH_MODULE(MoEBlock);

	int RunMoESelfTest();
}
