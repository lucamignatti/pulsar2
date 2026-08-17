#pragma once
#include "../FrameworkTorch.h"
#include "Models.h"

#include <torch/nn/modules/linear.h>
#include <vector>

// Plasticity diagnostics, computed from WEIGHTS ONLY - no data batch, so they are cheap enough to
// log every iteration.
//
// These used to live in PSD/Plasticity.h alongside the ReDo / rank-response INTERVENTIONS that
// acted on them. PSD was retired 2026-07-25, but these two signals were promoted rather than
// deleted: the residual cold start (45a59d5) was justified *by* effective-rank decay, and PSD was
// the only implementation of the measurement. Deleting the subsystem would have removed the
// project's ability to check its own architecture rationale.
//
// They are pure measurement now - nothing acts on them. See Plasticity/* panels.
namespace GGL::Plasticity {

	// Ordered list of the Linear layers in a model's seq (input -> output order).
	inline std::vector<std::shared_ptr<torch::nn::LinearImpl>> LinearLayers(Model* m) {
		std::vector<std::shared_ptr<torch::nn::LinearImpl>> out;
		for (size_t i = 0; i < m->seq->size(); i++)
			if (auto lin = std::dynamic_pointer_cast<torch::nn::LinearImpl>(m->seq->ptr(i)))
				out.push_back(lin);
		return out;
	}

	// Effective rank = exp(entropy of the normalized singular-value spectrum). A full-rank layer
	// approaches min(rows,cols); collapse toward a small number is the feature-rank-collapse alarm.
	inline float EffectiveRank(const torch::Tensor& weight) {
		RG_NO_GRAD;
		// CPU on purpose. The CUDA path goes through cusolver, whose handle creation
		// cudaMalloc's OUTSIDE the caching allocator — at 1B (allocator hoarding ~28GB of
		// a 32GB V100) that malloc failed on the tightest ranks and the uncaught
		// CUSOLVER_STATUS_INTERNAL_ERROR killed a 96-rank fleet (job 4631777, 2026-08-17).
		// This is a slow-moving structural canary on a 32-iteration cadence; a ~100ms CPU
		// SVD is free, and telemetry must never contend with training for device memory.
		torch::Tensor s = std::get<1>(torch::svd(weight.detach().to(torch::kFloat).cpu(), /*some=*/true, /*compute_uv=*/false));
		torch::Tensor p = s / (s.sum() + 1e-12f);
		torch::Tensor ent = -(p * (p + 1e-12f).log()).sum();
		return std::exp(ent.item<float>());
	}

	// Fraction of hidden units in the last hidden layer whose outgoing weight column is ~dead.
	inline float DeadUnitFraction(Model* policy, float thresh = 1e-3f) {
		RG_NO_GRAD;
		auto lins = LinearLayers(policy);
		if (lins.empty()) return 0;
		// Output layer: each column's norm measures how much that last-hidden unit drives the output.
		torch::Tensor colNorm = lins.back()->weight.detach().pow(2).sum(0).sqrt(); // per input unit
		float meanNorm = colNorm.mean().item<float>() + 1e-8f;
		torch::Tensor dead = (colNorm < thresh * meanNorm).to(torch::kFloat);
		return dead.mean().item<float>();
	}
}
