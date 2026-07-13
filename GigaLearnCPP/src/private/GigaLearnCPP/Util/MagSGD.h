#pragma once
#include "../FrameworkTorch.h"
#include <torch/optim/sgd.h>
#include <torch/csrc/api/include/torch/nn/utils/convert_parameters.h>

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

			// Calculate total update magnitude
			float gradMag = 0;
			for (auto& group : this->param_groups())
				for (auto& param : group.params())
					if (param.grad().defined())
						gradMag += param.grad().detach().square().sum().cpu().item<float>();
			gradMag = sqrtf(gradMag);

			// An all-zero gradient (e.g. a head whose loss term was absent this batch) has
			// no direction to normalize; dividing by 0 would write NaN into every param
			if (gradMag < 1e-12f)
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

			// Let SGD do the step with our new gradients (no closure: it was already
			// evaluated above, and passing it again would re-run it a second time)
			SGD::step();
			return loss;
		}
	};
}