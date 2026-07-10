#pragma once
#include "../FrameworkTorch.h"
#include "../Util/Models.h"
#include "Noise.h"

#include <torch/nn/modules/linear.h>
#include <torch/nn/modules/normalization.h>
#include <torch/nn/modules/activation.h>

// PolicySlots — the EGGROLL probe primitive.
//
// A single frozen base policy net, plus S independent rank-r LoRA-style adapters ("slots")
// on its Linear layers. Player rows are partitioned into S contiguous blocks (one per slot);
// a slot-routed forward evaluates the whole population in ONE base matmul + per-layer batched
// low-rank terms (paper §4.3), instead of S separate networks. This is what makes probing a
// large population affordable.
//
// Two lives of the factors:
//   * As NOISE (the ES search direction eps_s): generated deterministically from a round key,
//     used by FoldESUpdate() to fold the fitness-weighted sum into the base weights.
//   * As TRAINABLE parameters (the Baldwinian finetune): initialised to that noise, then
//     gradient-descended for G_probe steps so fitness measures a *descended* basin. Each slot's
//     grads are naturally isolated (bmm batch dim = slot), so one leaf tensor per layer both
//     trains all slots and keeps them independent.
//
// Scope: the policy net only. The shared trunk (read by the critic) is left unperturbed, so the
// critic stays consistent across probes.
namespace GGL::PSD {

	class PolicySlots {
	public:
		Model* policy;        // not owned; the base policy net whose Linear layers we adapt
		int numSlots;         // S (typically 2K for antithetic pairs)
		int rank;             // r
		float sigma;          // perturbation scale (pre per-layer RMS normalisation)
		torch::Device device;

		struct LayerInfo {
			int seqIdx;       // index into policy->seq
			int inDim, outDim;
			float coef;       // sigma * rms(W_l) / sqrt(r): the LayerNorm-invariant scale
		};
		std::vector<LayerInfo> layers;                 // one per Linear in policy->seq

		// Trainable factor stacks, one per Linear layer.
		//   Astack[l]: [S, outDim_l, r]      Bstack[l]: [S, inDim_l, r]
		std::vector<torch::Tensor> Astack, Bstack;

		PolicySlots(Model* policy, int numSlots, int rank, float sigma, torch::Device device);

		// (Re)initialise every slot's factors from deterministic round noise.
		// antithetic: slot 2p and 2p+1 share probe p's noise, with A negated for the odd twin
		// so their perturbations are exact opposites (+eps_p / -eps_p).
		void InitFromNoise(uint64_t roundSeed, bool antithetic);

		// Leaf tensors for the probe-finetune optimizer (fresh each round => fresh optimizer state).
		std::vector<torch::Tensor> Parameters();

		// Slot-routed forward. `trunkOut`: [N, inDim0] with N = numSlots * rowsPerSlot, rows in
		// contiguous per-slot blocks. Returns policy logits [N, numOutputs].
		torch::Tensor Forward(torch::Tensor trunkOut);

		// ES step: regenerate each slot's ORIGINAL (pre-finetune) eps_s from the round noise and
		// fold the fitness-weighted sum into the base weights in place:
		//     W_l <- W_l + alpha * sum_s weights[s] * eps_{s,l}
		// Returns the L2 norm of the total applied delta (for the ES Step Norm metric).
		float FoldESUpdate(uint64_t roundSeed, bool antithetic, const std::vector<float>& weights, float alpha);

		int RowsPerSlot(int totalRows) const { return totalRows / numSlots; }
	};
}
