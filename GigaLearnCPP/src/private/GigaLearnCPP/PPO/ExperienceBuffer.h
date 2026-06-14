#pragma once
#include "../FrameworkTorch.h"

namespace GGL {

	struct ExperienceTensors {
		torch::Tensor
			states, actions, logProbs, targetValues, actionMasks, advantages,
			// Optional per-transition exploration pressure in [0, 1]. When populated,
			// PPO scales/penalizes entropy per row instead of applying one global bonus.
			explorationWeights,
			// Per-step game-mode id (int64), used for per-mode advantage normalization.
			// Left undefined when only one mode exists (sampler skips undefined fields).
			modeIds,
			// GCRL fields (only populated when useGCRL); left undefined otherwise and
			// skipped by the sampler. actionComps: 8-dim continuous action taken.
			// futureGoals/carFutureGoals: hindsight-relabeled global / car-local ball goal.
			actionComps, futureGoals, carFutureGoals;

		auto begin() { return &states; }
		auto end() { return &carFutureGoals + 1; }
		auto begin() const { return &states; }
		auto end() const { return &carFutureGoals + 1; }
	};

	// https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/ppo/experience_buffer.py
	class ExperienceBuffer {
	public:

		torch::Device device;
		int seed;

		ExperienceTensors data;

		std::default_random_engine rng;

		ExperienceBuffer(int seed, torch::Device device);

		ExperienceTensors _GetSamples(const int64_t* indices, size_t size) const;

		// Not const because it uses our random engine
		std::vector<ExperienceTensors> GetAllBatchesShuffled(int64_t batchSize, bool overbatching);
	};
}
