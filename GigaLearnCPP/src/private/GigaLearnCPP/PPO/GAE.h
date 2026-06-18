#pragma once
#include <RLGymCPP/BasicTypes/Lists.h>
#include "../FrameworkTorch.h"

namespace GGL {
	// https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/util/torch_functions.py
	namespace GAE {
		void Compute(
			torch::Tensor rews, torch::Tensor terminals, torch::Tensor valPreds, torch::Tensor tTruncValPreds,
			torch::Tensor& outAdvantages, torch::Tensor& outValues, torch::Tensor& outReturns, float& outRewClipPortion,
			float gamma = 0.99f, float lambda = 0.95f, float returnStd = 0, float clipRange = 10
		);
	}
}