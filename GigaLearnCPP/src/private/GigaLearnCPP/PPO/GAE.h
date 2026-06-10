#pragma once
#include <RLGymCPP/BasicTypes/Lists.h>
#include "../FrameworkTorch.h"

namespace GGL {
	// https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/util/torch_functions.py
	namespace GAE {
		// returnStds: per-mode return standard deviations for reward standardization
		//	(pass { 0 } to disable standardization, { std } for the single-mode case).
		// modeIds: optional per-step mode index into returnStds; NULL -> all steps use returnStds[0].
		void Compute(
			torch::Tensor rews, torch::Tensor terminals, torch::Tensor valPreds, torch::Tensor tTruncValPreds,
			torch::Tensor& outAdvantages, torch::Tensor& outValues, torch::Tensor& outReturns, float& outRewClipPortion,
			float gamma = 0.99f, float lambda = 0.95f,
			const FList& returnStds = { 0 }, const int8_t* modeIds = NULL,
			float clipRange = 10
		);
	}
}