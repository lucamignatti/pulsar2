#pragma once

#include <RLGymCPP/EnvSet/EnvSet.h>
#include "Util/MetricSender.h"
#include "Util/RenderSender.h"
#include "LearnerConfig.h"
#include "PPO/TransferLearnConfig.h"

#include <atomic>

namespace GGL {

	typedef std::function<void(class Learner*, const std::vector<RLGC::GameState>& states, Report& report)> StepCallbackFn;

	// https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/learner.py
	class RG_IMEXPORT Learner {
	public:
		LearnerConfig config;

		RLGC::EnvSet* envSet;

		class PPOLearner* ppo;
		class PolicyVersionManager* versionMgr;
		class EvolutionStrategy* esManager;

		RLGC::EnvCreateFn envCreateFn;
		MetricSender* metricSender;
		RenderSender* renderSender;

		int obsSize;
		int numActions;

		// Game modes: (RocketSim GameMode, playersPerTeam) pairs mapped to dense ids,
		// used for per-mode normalization of returns, GCRL advantages and gate deltas
		int numModes = 1;
		std::vector<int> playerModeIds;
		std::vector<std::string> modeNames;

		// Per-mode running EMA of the GCRL reward gate delta (mean/variance). Using slow
		// running stats instead of per-batch z-scores keeps the gate's semantics stationary
		// so the critic isn't chasing a batch-dependent reward scale.
		std::vector<double> gateDeltaMeanEMA, gateDeltaVarEMA;

		// One return stat per game mode (pointer because WelfordStat is a private type)
		std::vector<struct WelfordStat>* returnStats;
		struct BatchedWelfordStat* obsStat;

		std::string runID = {};

		uint64_t
			totalTimesteps = 0,
			totalIterations = 0;

		uint64_t
			gcrlAdvScaleAnnealStartTS = UINT64_MAX,
			gcrlRewardGateAnnealStartTS = UINT64_MAX,
			gcrlAerialRewardGateAnnealStartTS = UINT64_MAX,
			curriculumRewardAnnealStartTS = UINT64_MAX,
			aerialCurriculumRewardAnnealStartTS = UINT64_MAX,
			sorsRewardScaleAnnealStartTS = UINT64_MAX,
			optWeightAnnealStartTS = UINT64_MAX;

		// Competence-gated anneal state: the progress counters only advance while the matching
		// EMA is at/above its configured gate (see PPOLearnerConfig), so curricula hold full
		// strength — and the GCRL reward gates stay out of the way — until the policy can
		// actually do the thing they teach/filter.
		uint64_t
			curriculumAnnealProgressTS = 0,
			aerialCurriculumAnnealProgressTS = 0,
			gcrlRewardGateAnnealProgressTS = 0,
			gcrlAerialRewardGateAnnealProgressTS = 0,
			optWeightAnnealProgressTS = 0;
		double touchRatioEMA = 0, highAirTouchRatioEMA = 0;

		// Feature C: adaptive (ratcheted-quantile) targets. Seeded in the ctor at today's
		// hand constants (gate target vel from config; StrongTouch floor from the user
		// atomic's initial value, i.e. the reward's constructor threshold), persisted in
		// RUNNING_STATS.json so crash-restarts don't replay the slew climb. Learner-thread
		// only; the floor is published to workers through the config atomic once per iter.
		double adaptiveGateTargetVel = -1, adaptiveStrongTouchFloor = -1;
		struct EMAHistogram* contactBallSpeedHist = nullptr;
		struct EMAHistogram* contactHitForceHist = nullptr;

		// Feature D: rOpt normalizer EMA + burn-in counter (persisted — a restart must not
		// re-run the burn-in or cold-start the normalizer; the bank/target nets refill/
		// re-sync instead).
		double optRewardMeanEMA = 0, optRewardVarEMA = 1;
		int64_t optBurnInItersDone = 0;

		StepCallbackFn stepCallback = NULL;

		Learner(RLGC::EnvCreateFn envCreateFunc, LearnerConfig config, StepCallbackFn stepCallback = NULL);
		void Start();

		void StartTransferLearn(const TransferLearnConfig& transferLearnConfig);

		void StartQuitKeyThread(std::atomic<bool>& quitPressed, std::thread& outThread);

		void Save();
		void Load();
		void SaveStats(std::filesystem::path path);
		void LoadStats(std::filesystem::path path);

		// Feature D offline harness: scores hand-supplied states through the frozen
		// optionality scorer (see GCRLOptionality::DebugScoreStates for the file format).
		// Requires config.ppo.useOptionality.
		void DebugScoreOptionality(const std::string& path);

		RG_NO_COPY(Learner);

		~Learner();
	};
}
