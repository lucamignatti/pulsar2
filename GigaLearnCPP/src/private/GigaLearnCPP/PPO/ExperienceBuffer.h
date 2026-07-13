#pragma once
#include "../FrameworkTorch.h"

namespace GGL {

	struct ExperienceTensors {
		// NOTE: _GetSamples and any field-zipping code iterate these POSITIONALLY via
		// begin()/end() below — new fields must go between `states` and `advantages`
		// (`advantages` must stay last), and undefined means "feature disabled".
		torch::Tensor
			states, actions, logProbs, targetValues,
			// Secondary goal-only critic value targets (undefined unless goalCritic.enabled)
			goalTargetValues,
			actionMasks,
			// Reachability aux-training data (undefined when reachability is disabled)
			carHerGoals, ballHerGoals, ballMovedMask,
			// Car-state HER goals for the car proposer's psi head (undefined unless carEnabled)
			carStateHerGoals,
			// Steered-practice rows, float 0/1 (undefined unless steering enabled): still train
			// the policy, but are excluded from critic/goal-critic regression (their episodes
			// end true-terminal at attempt resolution - the returns would alias match returns)
			practiceMask,
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