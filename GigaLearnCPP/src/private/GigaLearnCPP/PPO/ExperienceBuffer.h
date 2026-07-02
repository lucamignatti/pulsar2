#pragma once
#include "../FrameworkTorch.h"

namespace GGL {

	struct ExperienceTensors {
		// NOTE: _GetSamples and any field-zipping code iterate these POSITIONALLY via
		// begin()/end() below — new fields must go between `states` and `advantages`
		// (`advantages` must stay last), and undefined means "feature disabled".
		torch::Tensor
			states, actions, logProbs, targetValues, actionMasks,
			// Reachability aux-training data (undefined when reachability is disabled)
			carHerGoals, ballHerGoals, ballMovedMask,
			advantages;

		auto begin() { return &states; }
		auto end() { return &advantages + 1; }
		auto begin() const { return &states; }
		auto end() const { return &advantages + 1; }
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