#pragma once
#include <RLGymCPP/BasicTypes/Lists.h>
#include <RLGymCPP/StateSetters/AirDrillState.h>
#include "PPO/PPOLearnerConfig.h"
#include "SkillTrackerConfig.h"

#include <memory>

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
	struct GapSensorConfig {
		bool enabled = false;
		float tau = 0.8f;        // expectile: "returns when it goes well"
		float lr = 1e-4f;
		int trainRows = 98304;
	};

	// External fixed opponent (Nexto; NextoOpponent.h has the full rationale):
	// on serveFrac of collection iterations, the non-self team across the whole
	// fleet is played by a frozen external TorchScript bot instead of self/pool/
	// league. Forces contact with its style (aerial play, the point) and gives a
	// FIXED yardstick (Nexto/Goals For/Against) immune to pool inflation. Rows
	// are excluded from training exactly like old-version opponents; eval paths
	// never see it (Rating semantics unchanged). Latch-covered like every other
	// data-distribution intervention.
	struct ExternalOpponentConfig {
		bool enabled = false;
		std::string modelPath = {};  // TorchScript module (Nexto's nexto-model.pt)
		float serveFrac = 0.15f;     // per-iteration serve probability
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

		// RENDER MODE ONLY. Supplies raw controller state for the opponent's cars, so an
		// external agent (an RLBot bot) can play them with continuous controls instead
		// of our discrete action table.
		//
		// Injected as a callback rather than an interface the Learner knows about, on
		// purpose: the RLBot side needs flatbuffers and the RLBot schema, and none of
		// that belongs in GigaLearnCPP. The implementation lives with the executable
		// (src/), which already links RLBotCPP.
		//
		// Called once per decision step with the arena's current state and the set of
		// player rows the opponent owns. Fill `outControls` for those rows and return
		// true; return false to fall back to the action table for this step (bot not
		// connected yet, no fresh input, etc).
		// What the source wants the panel to display about itself.
		struct ExternalControlStatus {
			bool running = false;   // the agent's process is up
			bool connected = false; // ...and it has completed its handshake
			bool controlling = false; // ...and is actually sending inputs we applied
			std::string error;
			std::string name;
		};

		// Called once per decision step with the opponent spec the panel selected (the
		// source owns interpreting it), the arena state, and the player rows the opponent
		// holds. Fill `outControls` for those rows and return true; return false to fall
		// back to the action table for this step (agent not connected, no fresh input,
		// or this spec isn't one the source handles).
		//
		// `outValid` is PER ROW, not all-or-nothing: in a 2v2 one agent can be connected
		// while the other is still starting, and the car without an agent must keep
		// playing on the policy rather than freeze on a zeroed control.
		//
		// Servicing the agent's lifecycle here — rather than through a separate hook —
		// keeps it to one call site: render mode never leaves the collection loop, so
		// this callback is the only place that reliably runs every step.
		std::function<bool(
			const std::string& opponentSpec,
			RocketSim::Team opponentTeam,
			const RLGC::GameState& state,
			const std::vector<int>& opponentPlayerIndices,
			std::vector<RLGC::Action>& outControls,
			std::vector<uint8_t>& outValid,
			ExternalControlStatus& outStatus)> externalControlSource = nullptr;

		// Render only: lists the external agents the panel may offer as opponents. Kept
		// alongside externalControlSource because whoever can serve an agent is also the
		// only thing that knows how to find one.
		std::function<std::vector<std::string>()> vizBotFinder = nullptr;

		// If renderMode, the UDP port the viewer's control panel sends commands to
		// (bound on loopback only). See Util/VizControl.h.
		int vizControlPort = 9276;
		// Rewind history depth, in decision steps. 900 at 15 Hz is a 60-second scrubback;
		// a snapshot is ~1KB at 3v3, so the whole ring is under a megabyte.
		int vizHistoryFrames = 900;

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

		// Train against archived past selves drawn uniformly from the version ring. Since the
		// QD league was removed (2026-07-25) this is the ONLY self-play opponent source, so the
		// realized share matters: the opponent cascade is sequential, and Nexto rolls first, so
		// P(old version) = (1 - externalOpponent.serveFrac) * trainAgainstOldChance.
		// KNOWN PROPERTY, not a bug: the ring is a ROTATING window (maxOldVersions * tsPerVersion
		// = ~800M steps), so as the run matures the pool becomes a narrower slice of recent
		// history - the pool myopia CLAUDE.md documents. Accepted deliberately (user, 2026-07-25)
		// in exchange for having no archive to maintain; the permanent reference set in
		// SkillTrackerConfig is what carries genuinely old styles, and it is never trained against.
		bool trainAgainstOldVersions = false;
		float trainAgainstOldChance = 0.15f; // Chance (from 0 - 1) that an iteration will train against an old version

		// V_exp expectile twin (measurement only - see struct)
		GapSensorConfig gapSensor = {};

		SkillTrackerConfig skillTracker = {};





		// External fixed opponent (Nexto); additive and default-OFF
		ExternalOpponentConfig externalOpponent = {};
	};
}