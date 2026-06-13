#pragma once

#include "Util/Models.h"
#include <GigaLearnCPP/EvolutionStrategyConfig.h>
#include <GigaLearnCPP/Util/Report.h>
#include <GigaLearnCPP/Util/RenderSender.h>

#include <RLGymCPP/EnvSet/EnvSet.h>

namespace GGL {

	struct BatchedWelfordStat;

	// Periodic EGGROLL-style Evolution Strategies optimizer that runs alongside PPO.
	// Mirrors PolicyVersionManager mechanically (owns its own EnvSet, gated OnIteration hook),
	// but instead of just observing it nudges the live policy weights. Each ES step evaluates a
	// large population of low-rank weight perturbations by playing full games against the current
	// policy (member team vs baseline team, vectorized across arenas), ranks them by goal
	// differential (reward breaks ties), and applies the estimated gradient in-place.
	struct EvolutionStrategy {
		struct {
			EvolutionStrategyConfig config;

			RLGC::EnvSet* envSet = nullptr;

			int iterationsSinceRan = 0;
			int64_t stepCounter = 0; // varies the base seed across ES steps

			float lastMeanGoalDiff = 0;
			float lastScoringRate = 0;

			// Obs standardization (inherited from the trainer; null when standardizeObs is off).
			// ES feeds the same standardized obs the policy was trained on to both members and
			// baseline. Stats are static during an ES step, so the per-dim norm is computed once.
			BatchedWelfordStat* obsStat = nullptr;
			float minObsSTD = 0.1f;
			float maxObsMeanRange = 3.0f;
			std::vector<float> obsOffset, obsInvStd; // scratch, recomputed each ES step
		} es;

		RenderSender* renderSender;

		EvolutionStrategy(
			const EvolutionStrategyConfig& config,
			const RLGC::EnvSetConfig& envSetConfig,
			BatchedWelfordStat* obsStat, float minObsSTD, float maxObsMeanRange,
			RenderSender* renderSender = NULL);

		~EvolutionStrategy();

		void OnIteration(struct PPOLearner* ppo, Report& report, int64_t totalTimesteps, int64_t prevTotalTimesteps);

		void RunESStep(struct PPOLearner* ppo, Report& report);

		// Plays full games for members [chunkOffset, chunkOffset + numMembers) vs the current policy,
		// vectorized one member per arena. Writes per-member goal differential and reward
		// differential (indexed by global member id). P is the effective (even, if antithetic)
		// population size used for perturbation seeding.
		void EvaluateChunk(
			struct PPOLearner* ppo, int64_t baseSeed, int chunkOffset, int numMembers, int P,
			std::vector<int>& outGoalDiff, std::vector<float>& outRewardDiff);

		// Applies the population update to the live policy in-place under no-grad: RANK_GRADIENT
		// (centered-rank weighted sum of all members), CEM_ELITE (top-cemElites mean), or CEM_BEST
		// (re-anchor onto the single best member's perturbation). All regenerate perturbations from
		// seeds. `order` is the member indices sorted best->worst by fitness.
		void ApplyUpdate(struct PPOLearner* ppo, int64_t baseSeed, const std::vector<float>& shapedFitness,
			const std::vector<int>& order, Report& report);
	};
}
