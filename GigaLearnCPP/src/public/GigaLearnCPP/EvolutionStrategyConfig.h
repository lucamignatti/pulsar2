#pragma once

#include "Framework.h"

namespace GGL {
	// Evolution Strategies (EGGROLL-style) optimizer that runs periodically alongside PPO.
	// See arXiv:2511.16652. A large population of low-rank weight perturbations is evaluated
	// by playing full games against the current policy; fitness is goal differential (reward
	// breaks ties), and the estimated gradient nudges the live policy weights to help PPO
	// escape local minima.
	struct EvolutionStrategyConfig {
		bool enabled = false;

		// PPO iterations between ES steps (gate, like SkillTrackerConfig::updateInterval).
		// Large by default because full-game evaluation is expensive.
		int updateInterval = 100;

		// Total population members. With antithetic sampling there are populationSize/2
		// distinct perturbation directions (each evaluated as +/-). Should be even.
		int populationSize = 8192;

		// EGGROLL rank for the low-rank Linear-weight perturbations.
		int lowRankRank = 4;

		// Perturbation scale (std-dev).
		float sigma = 0.02f;

		// ES step size applied to the estimated gradient.
		float learningRate = 0.01f;

		// Full-match game-seconds simulated per chunk. Scoring fitness needs full games, not
		// short rollouts. Tune down to trade fitness fidelity for speed.
		float gameSimTime = 300.0f;
		// Hard cap on simulated seconds per chunk (safety bound).
		float maxSimTime = 360.0f;

		// Antithetic sampling: evaluate +eps and -eps for each direction (variance reduction).
		bool antithetic = true;

		// Centered-rank fitness shaping (OpenAI-ES). Recommended on.
		bool rankNormalize = true;

		// Decoupled L2 pull-to-zero applied during the ES update.
		float weightDecay = 0.005f;

		// Which params ES perturbs/updates. POLICY_ONLY keeps the shared encoder (and hence the
		// critic's input distribution) stable; POLICY_AND_SHARED_HEAD also perturbs the encoder.
		enum class Scope { POLICY_ONLY, POLICY_AND_SHARED_HEAD };
		Scope scope = Scope::POLICY_ONLY;

		// ── Update rule ──────────────────────────────────────────────────────────────
		// How the evaluated population (goal-differential fitness, the TRUE objective) updates the
		// live policy. All three reuse the SAME large-pop low-rank eval (EvaluateChunk); only the
		// step differs:
		//   RANK_GRADIENT -- centered-rank weighted sum of all members (OpenAI-ES gradient). Gentle;
		//                    a small step PPO's own gradient can out-weigh.
		//   CEM_ELITE     -- average only the top-`cemElites` members' perturbations, applied with
		//                    `learningRate` as the step. Harder selection; tolerates a larger `sigma`.
		//   CEM_BEST      -- re-anchor the live policy directly onto the single best member's
		//                    perturbation (a decisive jump PPO doesn't dilute). The most reach with a
		//                    large `sigma`, the most aggressive. Selection picks the winner, so a big
		//                    sigma that would wreck a gradient is fine here.
		// SELECTION (CEM_*) + a large `sigma` is the lever for reaching behaviors PPO is stuck short
		// of, optimizing the true objective rather than the shaped proxy.
		enum class UpdateMode { RANK_GRADIENT, CEM_ELITE, CEM_BEST };
		UpdateMode updateMode = UpdateMode::RANK_GRADIENT;
		// CEM_ELITE: how many top members to average (out of the effective population P).
		int cemElites = 256;
	};
}
