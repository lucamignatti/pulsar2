#pragma once

#include <RLGymCPP/EnvSet/EnvSet.h>
#include "Util/MetricSender.h"
#include <atomic>
#include "Util/RenderSender.h"
#include "LearnerConfig.h"
#include "PPO/TransferLearnConfig.h"

namespace GGL {

	typedef std::function<void(class Learner*, const std::vector<RLGC::GameState>& states, Report& report)> StepCallbackFn;

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

		StepCallbackFn stepCallback = NULL;

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