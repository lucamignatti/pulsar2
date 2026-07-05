#pragma once
#include "ExperienceBuffer.h"
#include <GigaLearnCPP/Util/Report.h>
#include <GigaLearnCPP/Util/Timer.h>
#include <GigaLearnCPP/PPO/PPOLearnerConfig.h>
#include <GigaLearnCPP/PPO/TransferLearnConfig.h>

#include "../Util/Models.h"
#include "Reachability.h"
#include "Proposer.h"

#include <torch/optim/adam.h>
#include <torch/nn/modules/loss.h>
#include <torch/nn/modules/container/sequential.h>

#include "ExperienceBuffer.h"

namespace GGL {

	// https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/ppo/ppo_learner.py
	class PPOLearner {
	public:
		ModelSet models = {};
		ModelSet guidingPolicyModels = {};

		// Reachability aux heads (null unless config.reachability.enabled);
		// its models live inside `models` and save/load/step with everything else
		ReachabilityModule* reach = NULL;
		// InfoNCE categorical accuracy of the last Learn() call (min of the two heads),
		// read by the Learner to drive the gate's accuracy EMA. lastReachTrained guards the
		// EMA update: false until the heads have actually trained this process (so a fresh
		// process resuming a checkpoint doesn't blend a meaningless 0 into a healthy EMA).
		float lastReachAccuracy = 0;
		bool lastReachTrained = false;

		// Deliberate-practice goal proposer (null unless config.proposer.enabled); its model lives
		// inside `models` and saves/loads with everything else, but is EXCLUDED from GetPolicyModels()
		// (old policy versions predate it) and is groupStepExempt (steps itself, never via the PPO loop)
		ProposerModule* proposer = NULL;
		// Car proposer: second ProposerModule (canonical car-state goals), null unless proposer.carEnabled
		ProposerModule* proposerCar = NULL;

		PPOLearnerConfig config;
		torch::Device device;

		// 2.2 goal-conditioned worker: extra width appended to the POLICY head's input for the online
		// proposer goal (0 unless proposer.goalCondition). = 6 + (carEnabled ? 6 : 0). The critic head
		// is NOT widened (it stays goal-blind). Computed in the ctor before MakeModels.
		int goalDim = 0;

		PPOLearner(
			int obsSize, int numActions,
			PPOLearnerConfig config, torch::Device device
		);

		static void MakeModels(
			bool makeCritic,
			int obsSize, int numActions,
			PartialModelConfig sharedHeadConfig, PartialModelConfig policyConfig, PartialModelConfig criticConfig,
			torch::Device device,
			ModelSet& outModels,
			int goalDim = 0
		);

		// If models is null, this->models will be used
		void InferActions(torch::Tensor obs, torch::Tensor actionMasks, torch::Tensor* outActions, torch::Tensor* outLogProbs, ModelSet* models = NULL);
		torch::Tensor InferCritic(torch::Tensor obs);

		// Goal-conditioned action inference (2.2). policyIn = cat(trunk, goal), where trunk is
		// precomputedTrunk if defined (reused from the caller's own forward) else shared_head(obs).
		// Runs fp32. The critic path is unaffected (goal-blind). obs may be undefined when
		// precomputedTrunk is supplied.
		void InferActionsGoalConditioned(
			ModelSet& models,
			torch::Tensor obs, torch::Tensor actionMasks, torch::Tensor goal, torch::Tensor precomputedTrunk,
			torch::Tensor* outActions, torch::Tensor* outLogProbs);

		// Perhaps they should be somewhere else? Should probably make an inference interface...
		// goal (optional): concatenated onto the trunk output before the policy head (2.2 goal
		// conditioning). precomputedTrunk (optional): trunk reused verbatim instead of re-forwarding
		// shared_head over obs.
		static torch::Tensor InferPolicyProbsFromModels(
			ModelSet& models,
			torch::Tensor obs, torch::Tensor actionMasks,
			float temperature,
			bool halfPrec,
			torch::Tensor goal = {}, torch::Tensor precomputedTrunk = {}
		);
		static void InferActionsFromModels(
			ModelSet& models,
			torch::Tensor obs, torch::Tensor actionMasks,
			bool deterministic, float temperature, bool halfPrec,
			torch::Tensor* outActions, torch::Tensor* outLogProbs,
			torch::Tensor goal = {}, torch::Tensor precomputedTrunk = {}
		);

		void Learn(ExperienceBuffer& experience, Report& report, bool isFirstIteration);

		void TransferLearn(
			ModelSet& oldModels, 
			torch::Tensor newObs, torch::Tensor oldObs, 
			torch::Tensor newActionMasks, torch::Tensor oldActionMasks, 
			torch::Tensor actionMaps,
			Report& report, 
			const TransferLearnConfig& transferLearnConfig
		);

		void SaveTo(std::filesystem::path folderPath);
		void LoadFrom(std::filesystem::path folderPath);
		void SetLearningRates(float policyLR, float criticLR);

		ModelSet GetPolicyModels();
	};
}