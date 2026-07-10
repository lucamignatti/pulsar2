#pragma once
#include "../FrameworkTorch.h"
#include "../Util/Models.h"
#include <GigaLearnCPP/Util/Report.h>

#include <torch/nn/modules/linear.h>

// Plasticity canaries (handoff §3.6 / §4.2). These are the alarm side of the effective-rank
// signal; the FREEZE side lives in the PSD controller and is competence-conditioned, so a low
// rank never means "freeze" on its own — it is read alongside the rating trend.
//
// Computed from weights only (no data batch needed), so they are cheap enough to log each probe
// round. Interventions (ReDo, distill) are gated on these but default OFF.
namespace GGL::PSD {

	// Effective rank = exp(entropy of the normalized singular-value spectrum). A full-rank layer
	// approaches min(rows,cols); collapse toward a small number is the feature-rank-collapse alarm.
	inline float EffectiveRank(const torch::Tensor& weight) {
		RG_NO_GRAD;
		// singular values via SVD (compute_uv=false path unavailable in this build, so take S only)
		torch::Tensor s = std::get<1>(torch::svd(weight.detach().to(torch::kFloat), /*some=*/true, /*compute_uv=*/false));
		torch::Tensor p = s / (s.sum() + 1e-12f);
		torch::Tensor ent = -(p * (p + 1e-12f).log()).sum();
		return std::exp(ent.item<float>());
	}

	// Fraction of hidden units in the last hidden layer whose outgoing weight row is ~dead.
	inline float DeadUnitFraction(Model* policy, float thresh = 1e-3f) {
		RG_NO_GRAD;
		// Last Linear = output layer; its INPUT dimension is the last hidden layer's width, and
		// each column's norm measures how much that unit drives the output.
		std::shared_ptr<torch::nn::LinearImpl> lastLin;
		for (size_t i = 0; i < policy->seq->size(); i++)
			if (auto lin = std::dynamic_pointer_cast<torch::nn::LinearImpl>(policy->seq->ptr(i)))
				lastLin = lin;
		if (!lastLin) return 0;
		torch::Tensor colNorm = lastLin->weight.detach().pow(2).sum(0).sqrt(); // per input unit
		float meanNorm = colNorm.mean().item<float>() + 1e-8f;
		torch::Tensor dead = (colNorm < thresh * meanNorm).to(torch::kFloat);
		return dead.mean().item<float>();
	}

	inline void LogPlasticity(ModelSet& models, Report& report) {
		if (Model* pol = models["policy"]) {
			// Effective rank of the final policy projection (the policy-head feature collapse metric).
			std::shared_ptr<torch::nn::LinearImpl> lastLin;
			for (size_t i = 0; i < pol->seq->size(); i++)
				if (auto lin = std::dynamic_pointer_cast<torch::nn::LinearImpl>(pol->seq->ptr(i)))
					lastLin = lin;
			if (lastLin)
				report["Plasticity/Policy Head EffRank"] = EffectiveRank(lastLin->weight);
			report["Plasticity/Policy Dead Unit Frac"] = DeadUnitFraction(pol);
		}
	}
}
