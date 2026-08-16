#pragma once
#include "../FrameworkTorch.h"
#include <torch/nn/cloneable.h>
#include <torch/nn/modules/normalization.h>

namespace GGL {

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
	};
	TORCH_MODULE(MoEBlock);

	int RunMoESelfTest();
}
