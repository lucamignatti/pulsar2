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
		float evalArenaFrac = 0.25f;      // (Baldwinian mode only) fraction of the pool reserved for eval

		// Probe finetune optimizer (factors only; base + trunk frozen). (Baldwinian mode only.)
		float probeLR = 3e-4f;
		float probeWeightDecay = 1e-4f;   // mild decay on factors (plasticity hygiene)

		// ---- Pure-ES probe (EGGROLL-faithful; arXiv 2511.16652 §6.1: large N is what makes ES work). ----
		// The 32-slot Baldwinian probe measured reliability ~0 (the ranking was noise). Pure-ES drops the
		// per-slot finetune entirely: every arena becomes an eval arena and each of the ~2K slots is
		// scored on LEVEL fitness (mean held-out episodic return) over a long window. The whole round
		// budget then buys population size N and fitness SNR instead of gradient steps. Set K large
		// (e.g. 256 -> S=512) so N sits inside the paper's proven envelope.
		bool pureES = true;               // false = legacy Baldwinian finetune probe
		int  evalWindowSteps = 200;       // eval rollout length; the reallocated GProbe budget -> precision

		// ---- Validation-gated fold. A/B the base policy's held-out return before vs after the fold and
		//      revert the fold if it regressed return by more than valMargin, so a noise-fold never lands
		//      unchecked (the aggregate-level version of the Baldwinian "test the future before committing").
		bool  valGateEnabled = true;
		int   valWindowSteps = 20;        // held-out validation rollout length (pre and post)
		float valMargin = 0.0f;           // revert only if post < pre - margin (>=0; 0 = revert any regression)

		// ---- Plasticity interventions (make the canaries ACT, not just log; handoff §3.6/§4.2). ----
		// Each is individually gated and, under healthy training, is a no-op: they fire only when a
		// plasticity signal actually degrades. Defaults here are the inert 2.6 baseline; the 3.0 run
		// turns the suite on in ExampleMain. All act during probe rounds on the just-folded weights.

		// (1) ReDo — recycle collapsed units in the policy's last hidden layer: reinit their incoming
		//     weights and zero their outgoing weights so a fresh feature can grow (Sokar et al. 2023).
		bool  redoEnabled = false;
		float redoDeadThresh = 1e-2f;        // a unit is "dead" if its output-column norm < thresh*mean
		float redoDeadFracTrigger = 0.10f;   // only recycle when >= this fraction of units are dead

		// (2) Effective-rank collapse response — shrink-and-perturb the policy head to restore rank
		//     when it collapses relative to its running peak (full-rank noise re-lifts the spectrum).
		bool  effRankResponseEnabled = false;
		float effRankCollapseFrac = 0.7f;    // fire when head EffRank < frac * rolling max
		float effRankShrink = 0.99f;         // multiply head weights by this on fire
		float effRankPerturbSigma = 0.01f;   // + Gaussian noise at this fraction of the head weight RMS

		// (3) Critic partial reset toward a fresh init (periodic; the critic tolerates resets and is
		//     the first net to lose value plasticity). Interpolates: W <- (1-frac)*W + frac*W_init.
		int distillPeriod = 0;               // (4) min probe rounds between distills; 0 = never distill
		int criticResetPeriod = 0;           // every N probe rounds; 0 = never
		float criticPartialResetFrac = 0.0f; // interpolation amount toward init. 0 = warm (handoff §4.2).

		// (4) Distill reset (deepest) — reinit the policy sub-net and behavior-distill the current
		//     policy into it over the frozen trunk. Gated on distillPeriod AND a real EffRank collapse,
		//     so a competent policy is only reset when its head has actually degraded.
		float distillTrigger = 0.6f;         // only distill when head EffRank ratio < this
		int   distillIters = 300;            // student KL-distill iterations
		int   distillBatch = 4096;           // observation rows distilled over
		float distillLR = 3e-4f;

		// (5) Competence-conditioned freeze — freeze the shared trunk once the policy is strong, to
		//     stop plasticity loss from eroding a good feature basis. Set the threshold to the
		//     Rating/1v1 you consider "lock it in"; the huge default never fires until you do.
		bool  freezeEnabled = false;
		float freezeRatingThresh = 1e9f;     // freeze trunk when Rating/1v1 >= this (huge = never)

		// Interventions LOG from step 0 but only ACT after this many training iterations. A FRESH net
		// starts at its lifetime-peak effective rank and falls steeply during normal early
		// specialization; without a warmup the rolling-max collapse triggers would fire distill/perturb
		// on that benign drop — bad on an unattended run. 0 = act immediately (fine when resuming a
		// mature checkpoint whose baseline is already established).
		int   interventionWarmupIters = 0;

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
		int binsPerAxis = 6;

		// Quantile-adaptive binning (cheap CVT-MAP-Elites). Uniform bins over the theoretical [0,1]
		// axis ranges are nearly all unreachable: the measured 3.1 population lived in
		// in_air [0.51,0.92], field_y [0.37,0.62], boost [0.016,0.060] — the boost axis never left
		// bin 0 and only 3 of 64 cells were occupied. With quantileBins, bin edges are the running
		// quantiles of the last quantileSampleCap observed BDs (every EvaluateMember contributes), so
		// resolution concentrates where the population actually lives and tracks it as the main
		// improves. Edges refresh on the reseed cadence (plus one bootstrap refresh once
		// quantileMinSamples BDs exist); each refresh re-tokenizes all members and enforces
		// one-elite-per-cell. false = the original fixed uniform [0,1] binning.
		bool quantileBins = true;
		int quantileSampleCap = 512;   // rolling BD sample window (~one reseed era of evals)
		int quantileMinSamples = 32;   // below this, fall back to uniform binning

		// Admission floor on fitness = (member goals - main goals) over an eval. NEGATIVE on purpose:
		// a good sparring partner is allowed to lose by a bit (style diversity matters more than the
		// member out-scoring the current main); only genuinely broken members are culled below this.
		float competenceFloor = -25.0f;
		float cullFrac = 0.25f;           // (reserved) fraction culled per interval
		int exploiterSlots = 2;           // BD-unconstrained members that hill-climb pure fitness-vs-main

		// Evolution operators (weight space, no gradients).
		float mutationSigma = 0.02f;      // full-Gaussian mutation scale (relative to weight RMS)
		float dareDropRate = 0.5f;        // DARE drop-and-rescale probability

		float pfspTemp = 1.0f;            // PFSP matchmaking temperature
		float descendOpponentFrac = 0.25f;// probability a training iteration faces a league opponent

		// === Permanent spaced ANCHOR opponents (research/reports/LEAGUE_ANCHORS.md) ===
		// Measured problem (2026-07-19): the evolved archive collapses to ~3 members / 1
		// occupied cell because fitness (= member goals - main goals) is RE-SCORED against
		// the improving main, so every fixed style ratchets below competenceFloor and is
		// culled; ReseedFromMain only ever adds near-clones. Result: the "diverse" opponent
		// pool is 3 copies of the recent self, and real match-play progress is ~+4 Elo/B.
		//
		// Anchors are full checkpoints archived OUTSIDE the rotation by
		// tools/archive_anchor.sh. They live in their own vector, so they are structurally
		// exempt from Cull/RefreshStalest/TryInsert/MAP-Elites: they are NEVER re-scored and
		// NEVER culled. That exemption is the entire design - a build that keeps scoring
		// them re-collapses in days via the same ratchet.
		float anchorFrac = 0.0f;          // share of ALL training iterations facing an anchor
		                                  // (served as a conditional draw inside the
		                                  //  descendOpponentFrac serve; 0 = feature OFF)
		int anchorMaxServed = 24;         // cap on the SERVING set (spaced-decimation keeps a
		                                  //  log-spaced span, not a recent window); disk keeps all
		float anchorRecencyFloor = 0.15f; // sampling weight of the OLDEST anchor relative to the
		                                  //  newest (linear in rank); never 0 - tail robustness
		std::string anchorDir = "";       // default: "<checkpointFolder>_anchors"
		int matchesPerMember = 20;        // (reserved) eval matches before fitness/BD is trusted
		int maxMembers = 256;
		int evolveEveryIters = 16;        // cadence of the league evolution step
		// Snapshot the CURRENT main policy as a fresh lineage this often (in iterations). Each snapshot
		// founds a new "era" whose style is the main's at that time; 0 = never re-seed (single lineage).
		int reseedEveryIters = 1000;
	};
}
