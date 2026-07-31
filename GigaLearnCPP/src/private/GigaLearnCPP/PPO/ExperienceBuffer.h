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
			// Steered-practice rows, float 0/1 (undefined unless steering enabled). The MAIN
			// critic deliberately trains on these rows too (excluding them caused the phantom
			// -V(s_end) Elo freefall, 2026-07-12 - see the comment in PPOLearner::Learn);
			// only the GOAL critic excludes them, and only under stage-2 resolution
			// termination (their goal channel is structurally absent there)
			practiceMask,
			// HEADROOM: one-iteration-frozen V-dagger TD targets (undefined unless vdagEnabled)
			vdagTargets,
			// THEORY: scaled per-step reward that LANDED on this row's arrival state
			// (r-hat regression target; undefined unless vdagTheoryEnabled)
			rhatTargets,
			// ADVANTAGE FILTERING: float 0/1, 1 = this row ranked in the top advFilterFrac of
			// the iteration - by signed advantage or by |advantage|, per advFilterMode - and the
			// POLICY may train on it (value heads ignore this mask). Undefined unless
			// advFilterFrac < 1.
			advFilterMask,
			// HEADROOM: per-row entropy multiplier from H (undefined unless vdagEntGateEnabled)
			entWeights,
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