#pragma once
#include "../FrameworkTorch.h"
#include <torch/optim/sgd.h>
#include <torch/csrc/api/include/torch/nn/utils/convert_parameters.h>
#include <cmath>

namespace GGL {

	typedef torch::optim::SGDOptions MagSGDOptions;

	// SGD, but updates the model with a pre-determined update magnitude instead of learning rate
	class MagSGD : public torch::optim::SGD {
	public:
		explicit MagSGD(std::vector<torch::Tensor> params, MagSGDOptions defaults)
			: SGD(params, defaults) {
		}

		torch::Tensor step(LossClosure closure = nullptr) override {
			RG_NO_GRAD;

			torch::Tensor loss = {};
			if (closure != nullptr) {
				at::AutoGradMode enable_grad(true);
				loss = closure();
			}

			// Calculate total update magnitude.
			// Accumulate on-device and sync once, instead of one CPU sync per parameter.
			torch::Tensor gradMagSq;
			for (auto& group : this->param_groups())
				for (auto& param : group.params())
					if (param.grad().defined()) {
						auto contrib = param.grad().detach().square().sum();
						gradMagSq = gradMagSq.defined() ? gradMagSq + contrib : contrib;
					}
			float gradMag = gradMagSq.defined() ? sqrtf(gradMagSq.cpu().item<float>()) : 0.f;

			if (gradMag <= 0 || !std::isfinite(gradMag))
				return loss;

			// Normalize the gradients by dividing them by the update magnitude
			for (auto& group : this->param_groups()) {
				for (auto& param : group.params()) {
					if (!param.grad().defined())
						continue;

					auto& gradSlice = param.mutable_grad();
					gradSlice /= gradMag;
				}
			}

			// Let SGD do the step with our new gradients
			SGD::step(nullptr);
			return loss;
		}
	};
}
