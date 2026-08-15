#pragma once
#include "../FrameworkTorch.h"
#include <torch/optim/sgd.h>

namespace GGL {

	typedef torch::optim::SGDOptions MuonOptions;

	// Muon: MomentUm Orthogonalized by Newton-Schulz (Keller Jordan et al.).
	//
	// 2D weight matrices: nesterov-momentum the gradient, orthogonalize the update with 5
	// Newton-Schulz iterations, then apply with an RMS-MATCHED learning rate: the
	// semi-orthogonal update has per-element RMS ~1/sqrt(max(rows, cols)), so lr is scaled
	// by sqrt(max(rows, cols)) to give per-element steps comparable to Adam at the same lr.
	// Without this the updates are ~17x too small at Adam-tuned LRs on these 256-wide nets —
	// weak enough to freeze whole runs. Do NOT remove; Adam-tuned LRs must transfer as-is.
	//
	// Non-matrix params (biases, LayerNorm affine): plain Adam at the group lr —
	// orthogonalizing a vector is meaningless, and Adam is the canonical fallback.
	//
	// The 2D momentum buffers live in SGD's param state so they serialize with checkpoints;
	// the (tiny) Adam moments for the non-matrix params are process-local and reset on resume.
	class Muon : public torch::optim::SGD {
	public:
		explicit Muon(std::vector<torch::Tensor> params, MuonOptions defaults)
			: SGD(params, defaults) {
		}

		torch::Tensor step(LossClosure closure = nullptr) override;

		// NS-shard spec, set by ModelSet::StepOptimsSharded before each step and reset after.
		// When shardWorld > 1, the momentum update still runs for EVERY 2D param (identical
		// on all ranks — grads are identical post-allreduce — so checkpointed state never
		// diverges), but the Newton-Schulz orthogonalization + param update run only for
		// params this rank owns (shardCounter % shardWorld == shardRank); non-owned params
		// are left stale and MUST be overwritten by the owner's broadcast immediately after.
		// shardCounter advances across models within one StepOptimsSharded pass so ownership
		// spreads evenly over the whole non-exempt family.
		int shardRank = 0, shardWorld = 1;
		int64_t shardCounter = 0;

	private:
		struct AdamState {
			torch::Tensor expAvg, expAvgSq;
			int64_t stepCount = 0;
		};
		std::unordered_map<void*, AdamState> adamStates;

		static torch::Tensor NewtonSchulz5(torch::Tensor g);
	};
}
