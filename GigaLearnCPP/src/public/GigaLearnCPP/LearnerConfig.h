#pragma once
#include <RLGymCPP/BasicTypes/Lists.h>
#include "PPO/PPOLearnerConfig.h"
#include "SkillTrackerConfig.h"
#include "PSDConfig.h"

namespace GGL {
	enum class LearnerDeviceType {
		AUTO, // CUDA if available, else MPS (Apple Metal), else CPU
		CPU,
		GPU_CUDA,
		GPU_MPS
	};

	// Steered-practice collection ("optimism surgery"). During COLLECTION ONLY, current-policy
	// rows belonging to the steered practice arenas get alpha*sigma*v added to the shared-trunk
	// output feeding the POLICY head, where v is a behavior-derived "commitment" direction.
	// Never steered: the learn pass, GAE value preds, skill-tracker evals, old-version
	// opponents. Stored logProbs come from the steered distribution, so PPO's clipped
	// importance ratio absorbs the bounded behavior/target divergence - the same mechanism
	// that absorbs pipelinedCollection's one-iteration lag.
	//
	// The direction is derived LIVE, every iteration, from the just-collected buffer (no
	// external file, no sidecar): airborne-ball readings from MATCH arenas are labeled by
	// ball-only landing sims + within-episode lookahead (went to the landing / declined it),
	// matched on (distance, flight time), and the went-declined difference of trunk means is
	// EMA-folded into the active vector. Derivation uses match rows only so the steered data
	// never feeds its own direction (no self-reinforcing loop). Directions go stale FAST as
	// the trunk trains (measured: a vector that gained +7pp engagement was 75M steps later
	// losing 11pp) - which is exactly why this is live instead of a file.
	//
	// Causal auto-gate: a slice of practice arenas runs UNSTEERED as controls (same
	// resolution-termination), and the steered-vs-control landing-engagement delta is EMA'd;
	// if steering stops helping, alpha drops to 0 automatically (derivation continues, and
	// steering re-engages when the delta recovers). This replaces any offline validation.
	//
	// Arena layout: [0, numSteered) steered practice | [numSteered, numPractice) control
	// practice | rest match. Pair ALL practice arenas (steered + control) with
	// AttemptResolutionCondition (ExampleMain wiring): practice episodes end at attempt
	// resolution with a NORMAL terminal (true terminal, NO value bootstrap - bootstrapping
	// V(s_end) would re-inject the counterattack tax through the critic).
	//
	// CRITIC SEMANTICS (learned the hard way): the MAIN critic trains on practice rows too.
	// Excluding them leaves V(s) at full match value while practice returns are truncated, so
	// GAE charges ~ -V(s_end) as a phantom penalty across every practice episode and the
	// policy unlearns ball engagement (live Elo freefall on first deployment, 2026-07-12).
	// Including them makes V the honest practice/match mixture for aliased states - a much
	// smaller, split bias. Practice rows stay excluded from the GOAL critic (its channel is
	// structurally absent in truncated episodes) and from the goal-advantage blend.
	struct CollectSteeringConfig {
		bool enabled = false;
		float alpha = 1.0f;             // strength, in units of sigma (trunk projection std, live-estimated)
		float practiceArenaFrac = 0.2f; // fraction of arenas that are practice (steered + control)
		float controlFracOfPractice = 0.15f; // fraction of practice arenas kept unsteered as gate controls

		// STAGE-1 vs STAGE-2 (see the failure history above): stage 1 runs NORMAL episodes in
		// steered arenas - no AttemptResolutionCondition (user wiring must match this flag), no
		// goal-critic masking, no blend guard; the whiff tax stays and reality does the
		// filtering. Only flip to true (stage 2) together with a dedicated practice-value
		// baseline; the shared critic provably cannot price mid-play truncations.
		bool resolutionTermination = false;

		// Rho-band gate: steer a row only when the ball head rates its state's scoring-
		// reachability inside the middle band of the current inference batch's rho
		// distribution (per-batch quantiles - self-calibrating, no absolute thresholds;
		// offline calibration put the intermediate band at ~40-60% actual conversion).
		// Points the optimism at hard-but-plausible plays toward the net.
		bool rhoGateEnabled = true;
		float rhoGateLo = 0.2f, rhoGateHi = 0.8f;
		int rhoGateActionSamples = 8;   // K uniform valid actions per row for the rho read
		// Gate by CONTACT reachability (car head, "can I reach the ball" - races) instead of
		// scoring reachability (ball head, "can the ball reach the net" - shots). Added after
		// the v2 possession gate kept reading scoring-gated steering as race-LOSING: we told
		// it to commit where the shot was uncertain, then graded it on winning the ball.
		bool rhoGateOnContact = true;

		// Rating drawdown guard: if Rating/1v1 falls more than ratingDrawdownTrip below its
		// slow EMA, steering LATCHES OFF for the rest of the process (loud log; derivation
		// and metrics continue). The one signal that catches update-damage (behavioral gates
		// cannot), made an actuator. Latched = no auto-re-enable; a human decides.
		// Calibration note (learned live): Rating/1v1 wiggles +-30..50 in normal training, so
		// the trip must sit outside that band. The collapse signature this guards against was
		// -130 in ~10 minutes; 75 catches that within a few evals and never fires on noise.
		bool ratingGuardEnabled = true;
		float ratingDrawdownTrip = 75.0f;
		float ratingEmaDecay = 0.995f;  // slow EMA (~140 rating-bearing iters half-life)

		// Live derivation
		float emaDecay = 0.9f;          // per-iteration EMA on the direction and sigma
		int maxReadingsPerIter = 4000;  // airborne readings labeled per iteration (landing sims are ~free)
		int minPairsPerUpdate = 100;    // skip the EMA update when matched pairs are scarcer than this

		// Causal auto-gate (units: absolute engagement fraction, e.g. 0.01 = 1pp).
		// Purpose in stage 1: detect a SIGN-INVERTED direction (steering actively suppressing
		// engagement), NOT "not helping yet" - per-iteration delta se is ~1pp at these arena
		// counts, so a 0.0 threshold trips on noise within minutes (observed live). While
		// tripped, alpha=0 makes steered==control, the delta EMA decays toward 0 and crosses
		// gateReenableAbove -> steering resumes -> re-trips only if genuinely harmful: the
		// thresholds below produce a natural duty-cycled probe with no extra machinery.
		int gateWarmupIters = 150;        // iterations with data before the gate may act
		float gateDisableBelow = -0.03f;  // ~3 sigma of the delta EMA: real inversion only
		float gateReenableAbove = -0.01f; // decay path back to probing
	};

	// https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/learner.py
	struct LearnerConfig {
		int numGames = 300;

		int tickSkip = 8;
		int actionDelay = 7;

		bool renderMode = false;
		// If renderMode, this is the scaling of time for the game
		// 1.0 = Run the game at real time
		// 2.0 = Run the game twice as fast as real time
		float renderTimeScale = 1.0f;

		// If renderMode, poll checkpointFolder every this-many seconds and hot-swap in the newest
		// checkpoint a separate training process has written, so the visualization tracks the model
		// as it learns. Set <= 0 to pin to whatever checkpoint was loaded at startup.
		float renderReloadSecs = 5.0f;

		PPOLearnerConfig ppo = {};

		// Checkpoints are saved here as timestep-numbered subfolders
		//	e.g. a checkpoint at 20,000 steps will save to a subfolder called "20000"
		// Set empty to disable saving
		std::filesystem::path checkpointFolder = "checkpoints"; 

		// Save every timestep
		// Set to zero to just use timestepsPerIteration
		int64_t tsPerSave = 1'000'000;

		int64_t randomSeed = -1; // Set to -1 to use the current time
		int checkpointsToKeep = 8; // Checkpoint storage limit before old checkpoints are deleted, set to -1 to disable

		// Checkpoint robustness (2026-07-13 GPU-lockup incident: the card corrupted
		// device-to-host reads for ~a minute before the driver killed it, poisoning the
		// ENTIRE rotation window - including one checkpoint with finite weights and a
		// behaviorally destroyed policy that structural checks cannot catch).
		// Golden archive: keep the top-N rated checkpoints permanently, outside rotation
		// (dir names "best_r<rating>_<ts>" are non-numeric, so FindNumberedDirs ignores
		// them). Bounds any future loss to "since the last new best". Rate-limited by
		// bestArchiveMinTsSpacing + a small rating margin so a steady climb doesn't copy
		// a checkpoint every save.
		int bestCheckpointsToKeep = 3;
		float bestArchiveRatingMargin = 5.0f;
		int64_t bestArchiveMinTsSpacing = 25'000'000;
		// Boot sanity probe: after loading a checkpoint that CLAIMS competence (rating >=
		// bootSanityMinRating), run a few kickoff episodes on a throwaway arena and require
		// ball touches. A policy that can't touch kickoff balls is treated as corrupt and
		// the loader falls back (healthy bots touch ~100% within ~4s; the incident's
		// scrambled checkpoint managed 1/10). The rating floor keeps young/fresh runs
		// (which legitimately can't kick off yet) exempt.
		bool bootSanityCheckEnabled = true;
		float bootSanityMinRating = 400.0f;
		LearnerDeviceType deviceType = LearnerDeviceType::AUTO; // Auto will use your CUDA GPU if available

		// Allow TF32 tensor-core matmuls on CUDA (Ampere+/Blackwell). libtorch defaults cuBLAS
		// TF32 to OFF, so without this the dense MLPs run strict fp32 and never touch the
		// tensor-core path. Set false for strict-fp32 reproducibility.
		bool allowTF32 = true;

		// Overlap NEXT-iteration experience collection (worker thread, frozen policy snapshot) with
		// THIS iteration's processing + PPO learn. The worker never touches live training weights and
		// its logProbs come from the same snapshot that sampled the actions, so PPO's importance
		// ratio stays exact; the one-update policy lag is standard async-PPO staleness the clip
		// objective absorbs. REVERT to exact sequential behavior by setting this false.
		// Auto-disabled in render mode and with the proposer/practice machinery (unaudited overlap).
		bool pipelinedCollection = false;

		// Standardize the obs values (doesn't seem to help much from my testing)
		bool standardizeObs = false;
		float minObsSTD = 1 / 10.f;
		float maxObsMeanRange = 3;
		int maxObsSamples = 100;

		// Standardize the returns to help the critic (don't disable this unless you know what you're doing)
		bool standardizeReturns = true;
		int maxReturnSamples = 150;

		// Will automatically add the rewards to metrics
		bool addRewardsToMetrics = true;
		int maxRewardSamples = 50; // Maximum reward samples per step for reward metrics
		int rewardSampleRandInterval = 8; // Randomized interval range between sampling rewards (per step)

		// Send metrics to the python metrics receiver
		// The receiver can then log them to wandb or whatever
		bool sendMetrics = true;
		std::string metricsProjectName = "gigalearncpp"; // Project name for the python metrics receiver
		std::string metricsGroupName = "unnamed-runs"; // Group name for the python metrics receiver
		std::string metricsRunName = "gigalearncpp-run"; // Run name for the python metrics receiver

		bool savePolicyVersions = false;
		int64_t tsPerVersion = 25'000'000;
		int maxOldVersions = 32;

		bool trainAgainstOldVersions = false;
		float trainAgainstOldChance = 0.15f; // Chance (from 0 - 1) that an iteration will train against an old version

		SkillTrackerConfig skillTracker = {};

		// Steered-practice collection; additive and default-OFF (see struct comment above)
		CollectSteeringConfig steering = {};

		// Basin-Racing (PSD) + QD league. Both additive and default-OFF; the baseline runs
		// unchanged unless psd.enabled / league.enabled are set.
		PSDConfig psd = {};
		LeagueConfig league = {};
	};
}