#pragma once
#include <RLGymCPP/BasicTypes/Lists.h>
#include <string>
#include <vector>

namespace GGL {

	// Probe–Step–Descend (Basin-Racing) configuration.
	//
	// PSD alternates two phases on one shared base policy:
	//   DESCEND — ordinary PPO for GExploit iterations (byte-identical to the 2.6 baseline).
	//   PROBE   — K antithetic EGGROLL perturbations get a short PPO finetune of ONLY their
	//             low-rank factors; each is scored on held-out matches; the fitness-weighted
	//             sum of the ORIGINAL perturbation directions is folded into the base weights.
	// Baldwinian: the finetuned factors are discarded, only the direction evidence is kept.
	//
	// The whole subsystem is additive and default-OFF. With enabled=false the Learner runs the
	// proven baseline unchanged.
	struct PSDConfig {
		bool enabled = false;

		// Stay in pure DESCEND until PPO plateaus, then start probing (handoff §4.1: the outer
		// loop only earns its K× cost where PPO alone stalls). Plateau = Rating/1v1 gained less
		// than plateauEloPerGExploit over the last DESCEND phase.
		bool warmupUntilPlateau = true;
		float plateauEloGain = 30.0f;     // Elo/DESCEND-phase below which we call it a plateau

		// Population.
		int K = 16;                       // antithetic pairs => numSlots = 2*K
		int rank = 4;                     // EGGROLL rank r (paper: r=1 already matches full-rank)
		bool antithetic = true;

		// Perturbation scale (the central knob — handoff §4.2). Adapted between rounds within
		// [sigmaMin,sigmaMax] by the clustering/canary rules.
		float sigma = 0.02f;
		float sigmaMin = 0.005f, sigmaMax = 0.05f;
		bool adaptSigma = true;

		// Budgets, in iterations (each iteration = tsPerItr timesteps of the partitioned pool).
		int GProbe = 50;                  // probe-finetune length (tuned live by probe-fitness variance)
		int GExploit = 2500;              // descend length between probe rounds

		// ES step.
		float alpha = 1.0f;               // ES learning rate (fitness weights are centered-rank in [-.5,.5])
		float lambda = 0.995f;            // shrink toward base (the shrink half of shrink-and-perturb)

		// Fitness.
		//   fitnessMode: 0 = post-finetune LEVEL, 1 = end-of-window SLOPE (handoff §4.1 fix; default).
		//   fitnessSource: 0 = dense shaped return (low variance, early), 1 = held-out match outcome.
		int fitnessMode = 1;
		int fitnessSource = 0;
		float evalArenaFrac = 0.25f;      // fraction of the pool reserved for held-out probe eval

		// Probe finetune optimizer (factors only; base + trunk frozen).
		float probeLR = 3e-4f;
		float probeWeightDecay = 1e-4f;   // mild decay on factors (plasticity hygiene)

		// Promotion / distill (deepest plasticity reset; grows over the run).
		int distillPeriod = 0;            // 0 = never distill (start here; raise once stable)

		// Late-game freeze (competence-conditioned; plumbing present, conservative default).
		bool freezeEnabled = false;
		float criticPartialResetFrac = 0.0f; // NEVER cold-reset the critic (handoff §4.2). 0 = warm.

		// Round dumps for backtracking: psd_rounds/<R>/{snapshot, fitness.json, config.json}.
		bool dumpRounds = true;
	};

	// Quality-Diversity league (MAP-Elites over FULL-WEIGHT members — deliberately NOT EGGROLL
	// adapters, so promotion never invalidates members and there is no BC re-anchoring leak).
	// Additive and default-OFF; supersedes trainAgainstOldVersions (which stays off).
	struct LeagueConfig {
		bool enabled = false;

		// Behavior-descriptor axes (progressive). Each name maps to a per-rollout stat the match
		// engine already computes; the grid is binsPerAxis^|axes|.
		std::vector<std::string> gridAxes = { "in_air_ratio", "field_y", "boost_economy" };
		int binsPerAxis = 4;

		float competenceFloor = 0.0f;     // min frontier fitness to be admitted to the archive
		float cullFrac = 0.25f;           // fraction culled per interval (within-cell)
		int exploiterSlots = 2;           // unconstrained (no-BD) exploiters — the completeness audit

		// Evolution operators (weight space, no gradients).
		float mutationSigma = 0.02f;      // full-Gaussian mutation scale (relative to weight RMS)
		float dareDropRate = 0.5f;        // DARE drop-and-rescale probability

		float pfspTemp = 1.0f;            // PFSP matchmaking temperature
		float descendOpponentFrac = 0.25f;// fraction of DESCEND arenas facing league opponents
		int refAnchors = 3;               // frozen reference opponents for BD measurement
		int matchesPerMember = 20;        // eval matches before a member's fitness/BD is trusted
		int maxMembers = 256;
		int evolveEveryIters = 16;        // cadence of the league evolution step
	};
}
