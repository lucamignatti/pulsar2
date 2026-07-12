#pragma once
#include "../FrameworkTorch.h"
#include "../Util/Models.h"
#include <GigaLearnCPP/Util/Report.h>

#include <torch/nn/modules/linear.h>
#include <vector>

// Plasticity signals + interventions (handoff §3.6 / §4.2). The two signals below are logged every
// probe round; the interventions in this file are what make them ACT rather than merely alarm.
// Everything is competence-conditioned upstream (the PSD controller only fires these at plateaus and
// gates the deep resets on the rating trend), so a low rank never means "reset" on its own.
//
// Computed from weights only (no data batch needed), so the signals are cheap to log each round.
namespace GGL::PSD {

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
		// singular values via SVD (compute_uv=false path unavailable in this build, so take S only)
		torch::Tensor s = std::get<1>(torch::svd(weight.detach().to(torch::kFloat), /*some=*/true, /*compute_uv=*/false));
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

	// ---- Interventions ------------------------------------------------------------------------
	// Each returns whether it fired and logs what it did. They mutate weights in place under no_grad;
	// the caller marks the model's half-precision cache outdated afterward.

	// (1) ReDo (Sokar et al. 2023). Recycle dead units in the policy's last hidden layer: reinit the
	//     incoming weights (row of the preceding Linear) to a fresh Kaiming draw, zero the incoming
	//     bias, and zero the outgoing weights (column of the output Linear) so the fresh feature grows
	//     from a clean slate without instantly perturbing the output. A recycled unit's incoming Adam
	//     moments are already ~0 (it was dead => ~0 gradient), so it relearns at full effective LR.
	inline bool RecycleDeadUnits(Model* policy, float deadThresh, float trigger, Report& report) {
		RG_NO_GRAD;
		auto lins = LinearLayers(policy);
		if (lins.size() < 2) return false;
		auto& hidLin = lins[lins.size() - 2]; // computes the last hidden activation
		auto& outLin = lins[lins.size() - 1]; // output projection

		torch::Tensor colNorm = outLin->weight.detach().pow(2).sum(0).sqrt(); // [hid]
		float meanNorm = colNorm.mean().item<float>() + 1e-8f;
		torch::Tensor dead = colNorm < (deadThresh * meanNorm);               // [hid] bool
		float frac = dead.to(torch::kFloat).mean().item<float>();
		report["Plasticity/ReDo Dead View"] = frac;
		if (frac < trigger) return false;

		torch::Tensor idx = dead.nonzero().flatten(); // long, [numDead]
		int numDead = (int)idx.numel();
		int fanIn = (int)hidLin->weight.size(1);
		// torch's default Linear init is kaiming_uniform_(a=sqrt(5)) => U(-1/sqrt(fanIn), 1/sqrt(fanIn)).
		float bound = 1.0f / std::sqrt((float)std::max(1, fanIn));

		torch::Tensor newRows = torch::empty({ (int64_t)numDead, (int64_t)fanIn }, hidLin->weight.options()).uniform_(-bound, bound);
		hidLin->weight.index_copy_(0, idx, newRows);
		if (hidLin->bias.defined())
			hidLin->bias.index_copy_(0, idx, torch::zeros({ (int64_t)numDead }, hidLin->bias.options()));
		outLin->weight.index_copy_(1, idx, torch::zeros({ outLin->weight.size(0), (int64_t)numDead }, outLin->weight.options()));

		report["Plasticity/ReDo Recycled Frac"] = frac;
		report["Plasticity/ReDo Fired"] = 1.0f;
		return true;
	}

	// (2) Effective-rank collapse response. Track the running peak of the head's effective rank; when
	//     the current value falls below collapseFrac of that peak, shrink-and-perturb the head weights
	//     (full-rank Gaussian noise re-lifts the collapsed spectrum). effRankMax is reset to the
	//     current value after firing so we don't re-fire on the same dip before the perturb takes.
	inline bool EffRankResponse(Model* policy, float curEffRank, float& effRankMax,
		float collapseFrac, float shrink, float perturbSigma, Report& report) {
		if (curEffRank > effRankMax) effRankMax = curEffRank;
		report["Plasticity/EffRank RollingMax"] = effRankMax;
		if (effRankMax <= 1e-6f) return false;
		float ratio = curEffRank / effRankMax;
		report["Plasticity/EffRank Ratio"] = ratio;
		if (ratio >= collapseFrac) return false;

		RG_NO_GRAD;
		auto lins = LinearLayers(policy);
		if (lins.empty()) return false;
		torch::Tensor W = lins.back()->weight;
		float rms = W.detach().pow(2).mean().sqrt().item<float>() + 1e-8f;
		W.mul_(shrink).add_(torch::randn_like(W) * (perturbSigma * rms));
		effRankMax = curEffRank; // rebuild the peak from here
		report["Plasticity/EffRank Perturb Fired"] = 1.0f;
		return true;
	}

	// (3) Critic partial reset. Interpolate the critic a fraction toward a fresh init:
	//     W <- (1-frac)*W + frac*W_init. The critic loses value plasticity first and tolerates
	//     resets far better than the policy (handoff §4.2: partial, never a cold reset).
	inline bool PartialResetCritic(Model* critic, float frac, Report& report) {
		if (frac <= 0.0f) return false;
		RG_NO_GRAD;
		Model* fresh = critic->MakeEmptyClone(); // fresh-init parameters
		auto cur = critic->parameters();
		auto fp = fresh->parameters();
		for (size_t i = 0; i < cur.size(); i++)
			cur[i].mul_(1.0f - frac).add_(fp[i] * frac);
		delete fresh->optim; // Model dtor doesn't own optim; free it to avoid a leak
		delete fresh;
		report["Plasticity/Critic Partial Reset Frac"] = frac;
		return true;
	}
}
