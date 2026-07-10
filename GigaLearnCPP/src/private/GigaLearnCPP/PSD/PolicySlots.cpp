#include "PolicySlots.h"

#include <torch/nn/functional/linear.h>
#include <cmath>

using namespace GGL;
using namespace GGL::PSD;

PolicySlots::PolicySlots(Model* policy, int numSlots, int rank, float sigma, torch::Device device)
	: policy(policy), numSlots(numSlots), rank(rank), sigma(sigma), device(device) {

	RG_ASSERT(numSlots > 0 && rank > 0);

	// Walk the policy's Sequential once, recording every Linear layer's shape and the
	// per-layer RMS scale. RMS is read from the base weights so a single global `sigma`
	// produces a comparable perturbation magnitude across layers of different scale.
	RG_NO_GRAD;
	auto& seq = policy->seq;
	for (size_t i = 0; i < seq->size(); i++) {
		auto m = seq->ptr(i);
		if (auto lin = std::dynamic_pointer_cast<torch::nn::LinearImpl>(m)) {
			LayerInfo info;
			info.seqIdx = (int)i;
			info.outDim = (int)lin->weight.size(0);
			info.inDim = (int)lin->weight.size(1);
			float rms = std::sqrt(lin->weight.detach().pow(2).mean().item<float>());
			info.coef = sigma * rms / std::sqrt((float)rank);
			layers.push_back(info);
		}
	}

	// Allocate empty factor stacks; InitFromNoise fills them.
	auto opts = torch::TensorOptions().dtype(torch::kFloat).device(device);
	for (auto& info : layers) {
		Astack.push_back(torch::zeros({ numSlots, info.outDim, rank }, opts));
		Bstack.push_back(torch::zeros({ numSlots, info.inDim, rank }, opts));
	}
}

void PolicySlots::InitFromNoise(uint64_t roundSeed, bool antithetic) {
	RG_NO_GRAD;
	for (int l = 0; l < (int)layers.size(); l++) {
		auto& info = layers[l];
		torch::Tensor A = torch::empty({ numSlots, info.outDim, rank }, torch::TensorOptions().dtype(torch::kFloat).device(device));
		torch::Tensor B = torch::empty({ numSlots, info.inDim, rank }, torch::TensorOptions().dtype(torch::kFloat).device(device));
		for (int s = 0; s < numSlots; s++) {
			int probe = antithetic ? (s / 2) : s;
			float sign = (antithetic && (s % 2 == 1)) ? -1.0f : 1.0f;
			A[s].copy_(GenFactor(roundSeed, probe, l, /*which=*/0, { info.outDim, rank }, device) * sign);
			B[s].copy_(GenFactor(roundSeed, probe, l, /*which=*/1, { info.inDim, rank }, device));
		}
		// Fresh leaf tensors each round => the finetune optimizer starts from zeroed moments.
		Astack[l] = A.clone().set_requires_grad(true);
		Bstack[l] = B.clone().set_requires_grad(true);
	}
}

std::vector<torch::Tensor> PolicySlots::Parameters() {
	std::vector<torch::Tensor> params;
	for (auto& t : Astack) params.push_back(t);
	for (auto& t : Bstack) params.push_back(t);
	return params;
}

torch::Tensor PolicySlots::Forward(torch::Tensor trunkOut) {
	namespace F = torch::nn::functional;
	torch::Tensor x = trunkOut;
	int l = 0;
	auto& seq = policy->seq;
	for (size_t i = 0; i < seq->size(); i++) {
		auto m = seq->ptr(i);
		if (auto lin = std::dynamic_pointer_cast<torch::nn::LinearImpl>(m)) {
			auto& info = layers[l];
			// Base: one dense matmul over the whole population (the bulk of the cost).
			torch::Tensor base = F::linear(x, lin->weight, lin->bias);
			// Low-rank slot term: reshape to [S, rowsPerSlot, in], two batched matmuls.
			//   (x_s @ B_s) @ A_s^T  gives each slot its own rank-r delta.
			torch::Tensor xr = x.view({ numSlots, -1, info.inDim });
			torch::Tensor tmp = torch::bmm(xr, Bstack[l]);                 // [S, rpS, r]
			torch::Tensor delta = torch::bmm(tmp, Astack[l].transpose(1, 2)); // [S, rpS, out]
			delta = delta.reshape({ -1, info.outDim }) * info.coef;
			x = base + delta;
			l++;
		} else if (auto ln = std::dynamic_pointer_cast<torch::nn::LayerNormImpl>(m)) {
			x = ln->forward(x);
		} else if (auto act = std::dynamic_pointer_cast<torch::nn::LeakyReLUImpl>(m)) {
			x = act->forward(x);
		} else if (auto act = std::dynamic_pointer_cast<torch::nn::ReLUImpl>(m)) {
			x = act->forward(x);
		} else if (auto act = std::dynamic_pointer_cast<torch::nn::SigmoidImpl>(m)) {
			x = act->forward(x);
		} else if (auto act = std::dynamic_pointer_cast<torch::nn::TanhImpl>(m)) {
			x = act->forward(x);
		} else {
			RG_ERR_CLOSE("PolicySlots::Forward: unexpected module type in policy net at index " << i);
		}
	}
	return x;
}

float PolicySlots::FoldESUpdate(uint64_t roundSeed, bool antithetic, const std::vector<float>& weights, float alpha) {
	RG_ASSERT((int)weights.size() == numSlots);
	RG_NO_GRAD;
	double totalSq = 0;
	auto& seq = policy->seq;
	int l = 0;
	for (size_t i = 0; i < seq->size(); i++) {
		auto m = seq->ptr(i);
		auto lin = std::dynamic_pointer_cast<torch::nn::LinearImpl>(m);
		if (!lin) continue;
		auto& info = layers[l];
		// Accumulate the weighted sum of ORIGINAL eps directions for this layer.
		torch::Tensor deltaW = torch::zeros({ info.outDim, info.inDim }, torch::TensorOptions().dtype(torch::kFloat).device(device));
		for (int s = 0; s < numSlots; s++) {
			float w = weights[s];
			if (w == 0) continue;
			int probe = antithetic ? (s / 2) : s;
			float sign = (antithetic && (s % 2 == 1)) ? -1.0f : 1.0f;
			torch::Tensor A0 = GenFactor(roundSeed, probe, l, 0, { info.outDim, rank }, device) * sign;
			torch::Tensor B0 = GenFactor(roundSeed, probe, l, 1, { info.inDim, rank }, device);
			// eps = coef * A0 B0^T
			deltaW.add_(torch::matmul(A0, B0.transpose(0, 1)), w * info.coef);
		}
		lin->weight.add_(deltaW, alpha);
		totalSq += (deltaW.mul(alpha)).pow(2).sum().item<double>();
		l++;
	}
	policy->_seqHalfOutdated = true; // base weights changed; half-precision clone is stale
	return (float)std::sqrt(totalSq);
}
