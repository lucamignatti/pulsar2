#pragma once
#include "../FrameworkTorch.h"
#include <torch/types.h>

// Weight-space evolution operators for the QD league (no gradients). Members are flat parameter
// vectors. Start with the two the handoff names first — full-Gaussian mutation + DARE crossover —
// behind a uniform interface so SVD-structured operators can be added later.
namespace GGL::League {

	// Full-Gaussian mutation: theta' = theta + sigma * rms(theta) * N(0,1), scaled by the parent's
	// own RMS so one sigma is meaningful regardless of layer scale.
	inline torch::Tensor MutateGaussian(const torch::Tensor& theta, float sigma) {
		float rms = std::sqrt(theta.detach().pow(2).mean().item<float>()) + 1e-8f;
		return theta + torch::randn_like(theta) * (sigma * rms);
	}

	// DARE (Drop And REscale, Yu et al.): keep each parameter's delta-from-mean-parent with prob
	// (1-dropRate), rescale survivors by 1/(1-dropRate), around the average of the two parents.
	inline torch::Tensor CrossoverDARE(const torch::Tensor& a, const torch::Tensor& b, float dropRate) {
		torch::Tensor base = (a + b) * 0.5f;
		torch::Tensor da = a - base, db = b - base;
		torch::Tensor keepA = (torch::rand_like(a) >= dropRate).to(torch::kFloat);
		torch::Tensor keepB = (torch::rand_like(b) >= dropRate).to(torch::kFloat);
		float scale = 1.0f / std::max(1e-3f, 1.0f - dropRate);
		return base + (da * keepA + db * keepB) * scale;
	}
}
