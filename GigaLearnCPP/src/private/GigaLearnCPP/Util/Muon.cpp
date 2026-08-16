#include "Muon.h"
#include <torch/version.h>
#if TORCH_VERSION_MAJOR < 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR < 2)
#include <c10/util/C++17.h>
#endif

using namespace torch;

// Orthogonalize via the quintic Newton-Schulz iteration. The coefficients trade exact
// orthogonality for convergence speed: 5 iterations land every singular value near 1,
// which is all Muon needs (the update's direction structure, not its exact spectrum).
torch::Tensor GGL::Muon::NewtonSchulz5(torch::Tensor g) {
	constexpr float A = 3.4445f, B = -4.7750f, C = 2.0315f;
	constexpr int NUM_ITERS = 5;

	// Batched form: a 3D input [E, m, n] (MoE expert stacks) orthogonalizes every slice
	// in the same ~15 matmuls via bmm — per-slice Frobenius normalization, batched grams.
	// 2D inputs take the identical math with a leading batch of 1 semantics-free.
	const bool batched = g.dim() == 3;
	Tensor x = batched
		? g / (g.norm(2, { -2, -1 }, /*keepdim=*/true) + 1e-7f)
		: g / (g.norm() + 1e-7f);

	// Keep the gram matrix below at the smaller dimension
	bool transposed = false;
	if (x.size(-2) > x.size(-1)) {
		x = x.transpose(-2, -1);
		transposed = true;
	}

	for (int i = 0; i < NUM_ITERS; i++) {
		Tensor a = torch::matmul(x, x.transpose(-2, -1));
		Tensor b = a * B + torch::matmul(a, a) * C;
		x = x * A + torch::matmul(b, x);
	}

	if (transposed)
		x = x.transpose(-2, -1);

	return x;
}

torch::Tensor GGL::Muon::step(LossClosure closure) {
	Tensor loss = {};
	if (closure != nullptr) {
		at::AutoGradMode enableGrad(true);
		loss = closure();
	}

	RG_NO_GRAD;

	for (auto& group : param_groups()) {
		auto& options = static_cast<MuonOptions&>(group.options());
		double lr = options.lr();
		double momentum = options.momentum();
		bool nesterov = options.nesterov();

		for (auto& param : group.params()) {
			if (!param.grad().defined())
				continue;

			Tensor grad = param.grad();

			// dim 3 = MoE expert stacks [E, m, n]: same Muon update, batched NS, RMS-matched
			// lr from the per-slice matrix dims (identical for every slice).
			if ((grad.dim() == 2 && grad.size(0) > 1 && grad.size(1) > 1)
				|| (grad.dim() == 3 && grad.size(1) > 1 && grad.size(2) > 1)) {
				// Momentum buffer in SGD's own param state so it checkpoints with the model.
				// Libtorch 2.1 keys Optimizer::state_ by stringified TensorImpl*; 2.2+
				// (PR 108748) keys by the pointer itself.
#if TORCH_VERSION_MAJOR < 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR < 2)
				const auto stateKey = c10::guts::to_string(param.unsafeGetTensorImpl());
#else
				const auto stateKey = param.unsafeGetTensorImpl();
#endif
				auto stateItr = state_.find(stateKey);
				if (stateItr == state_.end()) {
					auto newState = std::make_unique<optim::SGDParamState>();
					newState->momentum_buffer(torch::zeros_like(param));
					stateItr = state_.insert({ stateKey, std::move(newState) }).first;
				}

				Tensor buf = static_cast<optim::SGDParamState&>(*stateItr->second).momentum_buffer();
				buf.mul_(momentum).add_(grad);

				// NS shard (see header): momentum above always runs; the expensive
				// orthogonalization + apply only on the owning rank.
				const bool nsOwned = shardWorld <= 1
					|| (shardCounter++ % (int64_t)shardWorld) == (int64_t)shardRank;
				if (!nsOwned)
					continue;

				Tensor update = nesterov ? grad.add(buf, momentum) : buf;
				update = NewtonSchulz5(update);

				// The RMS match (see header): per-element step ~= lr, like Adam
				double adjustedLR = lr * std::sqrt((double)std::max(update.size(-2), update.size(-1)));
				param.add_(update, -adjustedLR);
			} else {
				// Non-matrix param: plain Adam at the group lr
				AdamState& adam = adamStates[param.unsafeGetTensorImpl()];
				if (!adam.expAvg.defined()) {
					adam.expAvg = torch::zeros_like(param);
					adam.expAvgSq = torch::zeros_like(param);
				}
				adam.stepCount++;

				constexpr double BETA1 = 0.9, BETA2 = 0.999, EPS = 1e-8;
				adam.expAvg.mul_(BETA1).add_(grad, 1 - BETA1);
				adam.expAvgSq.mul_(BETA2).addcmul_(grad, grad, 1 - BETA2);

				double biasCorr1 = 1 - std::pow(BETA1, (double)adam.stepCount);
				double biasCorr2 = 1 - std::pow(BETA2, (double)adam.stepCount);
				Tensor denom = (adam.expAvgSq / biasCorr2).sqrt_().add_(EPS);
				param.addcdiv_(adam.expAvg / biasCorr1, denom, -lr);
			}
		}
	}

	return loss;
}
