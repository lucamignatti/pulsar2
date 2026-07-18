#pragma once

#include <RLGymCPP/EnvSet/EnvSet.h>
#include "Util/MetricSender.h"
#include <atomic>
#include "Util/RenderSender.h"
#include "LearnerConfig.h"
#include "PPO/TransferLearnConfig.h"

namespace GGL {

	typedef std::function<void(class Learner*, const std::vector<RLGC::GameState>& states, Report& report)> StepCallbackFn;

	// Called once per training iteration at the tail of the loop, after the skill tracker /
	// guards have written into the report (so Rating/* keys are visible when they refreshed
	// this iteration). May call RequestSaveAndExit (e.g. curriculum phase triggers).
	typedef std::function<void(class Learner*, Report& report)> IterationCallbackFn;

	// https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/learner.py
	class RG_IMEXPORT Learner {
	public:
		LearnerConfig config;

		RLGC::EnvSet* envSet;

		class PPOLearner* ppo;
		class PolicyVersionManager* versionMgr;
		class PSDController* psd = nullptr;   // Basin-Racing outer loop; null unless config.psd.enabled
		class LeagueArchive* league = nullptr; // QD league; null unless config.league.enabled

		RLGC::EnvCreateFn envCreateFn;
		MetricSender* metricSender;
		RenderSender* renderSender;

		int obsSize;
		int numActions;

		struct WelfordStat* returnStat;
		struct BatchedWelfordStat* obsStat;

		std::string runID = {};

		uint64_t
			totalTimesteps = 0,
			totalIterations = 0;

		// Reachability gate anneal state: EMAs of the heads' InfoNCE accuracy and of the
		// touch-prediction agreement (persisted in the checkpoint stats)
		float reachAccEMA = 0, reachAgreeEMA = 0;

		// Last Rating/1v1 seen from the skill tracker (drives the best-checkpoint archive)
		// and the archive's rate limiter
		float lastEvalRating = NAN;
		uint64_t lastBestArchiveTs = 0;

		// Kickoff-touch probe on a throwaway arena; used by Load() to reject checkpoints
		// that load structurally but are behaviorally destroyed (see LearnerConfig)
		bool BootSanityProbe();

		// Steered-practice collection state (config.steering); the direction tensor lives in the
		// PPOLearner (this header stays torch-free) and is derived LIVE from each iteration's
		// buffer inside Start() (see fnSteerUpdate there). These are fixed at startup:
		// [0, numSteeredArenas) steered practice | [numSteeredArenas, numPracticeArenas)
		// unsteered control practice | rest match.
		bool steerLoaded = false;
		int numPracticeArenas = 0;
		int numSteeredArenas = 0;

		// Rating drawdown guard state (persisted in the checkpoint stats): the latch is the
		// ONLY guard that can see update-damage, and the ops wrapper auto-restarts on crashes -
		// process-local latch state would silently re-arm steering on a damaged policy
		float steerRatingEMA = NAN;
		bool steerRatingTripped = false;
		// High-water mark for the peak-drawdown latch (ratingPeakTrip): the slow EMA lags
		// a fresh climb, so a slide off a new peak is invisible to it (2026-07-14). Also
		// persisted - a crash-restart must not forget the peak it was sliding from.
		float steerRatingPeak = NAN;

		// Churn-telemetry archive of the live steering directions (STEERING_ROADMAP
		// "continuous items"), PER MODE (index = playersPerTeam-1): the current applied
		// vector + sigma are snapshotted into every checkpoint's RUNNING_STATS
		// ("steer_vec"/"steer_sigma" for 1v1, "steer_vec_2v2"... for teams, ~5KB each) so
		// offline analysis gets the staleness curve for free and deployment can steer at
		// inference. Save-only, NEVER loaded: directions re-derive within one iteration of
		// any restart by design (a stale vector must not outlive its trunk).
		std::array<std::vector<float>, 3> steerVecSave;
		std::array<float, 3> steerSigmaSave = {};

		// FEAR PANEL (in-trainer census, FEAR_MINE.md): a frozen set of high-
		// disagreement 2v2 decline obs rows, captured ONCE when fear mining first
		// produces a full ranking, then persisted through RUNNING_STATS and re-valued
		// by the CURRENT critic + goal critic every iteration (Steer/Fear Panel zV /
		// zG / Dz panels). zV rising toward 0 across checkpoints = the critic
		// unlearning its fear = the drill deploy working. Unlike steerVecSave this IS
		// loaded: the panel must stay FIXED to be longitudinal (it is measurement
		// state, not an actuator - nothing reads it but the report).
		std::vector<float> fearPanelObs;   // flattened [k, obsSize]
		int64_t fearPanelTimestep = 0;     // totalTimesteps at freeze

		// AirDrill altitude-annealing controller state (AERIAL_GAP.md); persisted in
		// RUNNING_STATS so the curriculum never resets on restart
		float airDrillD = 0.f;             // current difficulty (mirrored into the shared knob)
		float airDrillConvEMA = -1.f;      // aerial-conversion EMA (-1 = unseeded)
		float airDrillConvRef = -1.f;      // reference EMA at the last adjustment
		int64_t airDrillLastAdjustIter = 0;

		// EMERGENCE RC2 observer: rolling sample of the miner's picked obs rows
		// (save-only, refreshed each iteration - offline inspection channel)
		std::vector<float> minerSampleObs;

		// EMERGENCE RC1: RND frontier-optimism state (opaque - torch types stay out
		// of this header; defined in Learner.cpp). Lazily built at first use, loaded
		// from RND_PRED.lt/RND_TARGET.lt when the checkpoint carries them.
		std::shared_ptr<struct RndState> rnd;

		// Introspective frontier drive, Stage-1 gap sensor (opaque; GAP_EXP.lt)
		std::shared_ptr<struct GapState> gapSensor;

		StepCallbackFn stepCallback = NULL;
		IterationCallbackFn iterationCallback = NULL; // optional; assign after construction

		// Programmatic save-and-exit, honored at the same safe point as the Q key (worker
		// joined, checkpoint saved first). A nonzero code makes tools/run_trainer.sh RESTART
		// the trainer (its crash-restart path) - the self-deploy vehicle for boot-time config
		// changes like a curriculum phase flip; 0 stops it like a clean quit.
		std::atomic<bool> exitRequested = false;
		std::atomic<int> requestedExitCode = 0;
		void RequestSaveAndExit(int exitCode) {
			requestedExitCode = exitCode;
			exitRequested = true;
		}

		Learner(RLGC::EnvCreateFn envCreateFunc, LearnerConfig config, StepCallbackFn stepCallback = NULL);
		void Start();

		void StartTransferLearn(const TransferLearnConfig& transferLearnConfig);

		void StartQuitKeyThread(std::atomic<bool>& quitPressed, std::thread& outThread);

		void Save();
		void Load();
		void SaveStats(std::filesystem::path path);
		void LoadStats(std::filesystem::path path);

		// Render-mode live reload: if a numbered checkpoint newer than loadedTimesteps has appeared
		// in config.checkpointFolder and looks fully written, hot-swap its policy weights in and
		// update loadedTimesteps. Returns true on a successful swap. Guarded so a mid-write race
		// against the (uncontrolled) training process can only cost a retry, never the process.
		bool ReloadNewestCheckpointForRender(int64_t& loadedTimesteps);

		RG_NO_COPY(Learner);

		~Learner();
	};
}