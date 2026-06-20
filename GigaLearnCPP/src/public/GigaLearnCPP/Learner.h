#pragma once

#include <atomic>
#include <RLGymCPP/EnvSet/EnvSet.h>
#include "Util/MetricSink.h"
#include "Util/RenderSink.h"
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

		RLGC::EnvCreateFn envCreateFn;

		// Metric/render side-channels behind ports. Owned by the Learner only when it created the
		// default (wandb / Python) adapter; an injected sink is owned by the caller.
		MetricSink* metricSink;
		RenderSink* renderSink;
		bool ownsMetricSink = false;
		bool ownsRenderSink = false;

		int obsSize;
		int numActions;

		struct WelfordStat* returnStat;
		struct BatchedWelfordStat* obsStat;

		std::string runID = {};

		uint64_t
			totalTimesteps = 0,
			totalIterations = 0;

		StepCallbackFn stepCallback = NULL;

		// metricSink/renderSink default to NULL, in which case the Learner builds the real
		// wandb/Python adapters from `config`. Inject an in-memory sink to run without Python.
		Learner(RLGC::EnvCreateFn envCreateFunc, LearnerConfig config, StepCallbackFn stepCallback = NULL,
			MetricSink* metricSink = NULL, RenderSink* renderSink = NULL);
		void Start();

		void StartTransferLearn(const TransferLearnConfig& transferLearnConfig);

		void StartQuitKeyThread(std::atomic<bool>& quitPressed, std::thread& outThread);

		void Save();
		void Load();
		void SaveStats(std::filesystem::path path);
		void LoadStats(std::filesystem::path path);

		RG_NO_COPY(Learner);

		~Learner();
	};
}