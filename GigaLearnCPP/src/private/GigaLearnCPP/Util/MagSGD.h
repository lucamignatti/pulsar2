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
			return SGD::step(closure);
		}
	};
}