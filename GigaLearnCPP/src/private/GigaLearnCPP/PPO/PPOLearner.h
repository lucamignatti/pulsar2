#pragma once
#include "ExperienceBuffer.h"
#include "ContrastiveGoalLearner.h"
#include <GigaLearnCPP/Util/Report.h>
#include <GigaLearnCPP/Util/Timer.h>
#include <GigaLearnCPP/PPO/PPOLearnerConfig.h>
#include <GigaLearnCPP/PPO/TransferLearnConfig.h>

#include "../Util/Models.h"
#include "../Util/RSNorm.h"

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
		ContrastiveGoalLearner* contrastiveGoalLearner = NULL;
		// Optional second isolated critic: egocentric car-local ball goal (controllability).
		ContrastiveGoalLearner* carContrastiveLearner = NULL;
		// SimBa RSNorm (running obs normalization). NULL unless config.rsNorm.enabled.
		// One shared normalizer for actor & critic; "canonical stats everywhere"
		// (old-version & skill-eval inference use this same normalizer).
		RSNorm* obsNorm = NULL;

		PPOLearnerConfig config;
		torch::Device device;
		int obsSize;
		int numActions;

		PPOLearner(
			int obsSize, int numActions,
			PPOLearnerConfig config, torch::Device device
		);
		~PPOLearner();

		static void MakeModels(
			bool makeCritic, 
			int obsSize, int numActions, 
			PartialModelConfig sharedHeadConfig, PartialModelConfig policyConfig, PartialModelConfig criticConfig,
			torch::Device device,
			ModelSet& outModels
		);
		
		// If models is null, this->models will be used
		void InferActions(torch::Tensor obs, torch::Tensor actionMasks, torch::Tensor* outActions, torch::Tensor* outLogProbs, ModelSet* models = NULL, torch::Tensor* outActionProbs = NULL);
		torch::Tensor InferCritic(torch::Tensor obs);

		// Perhaps they should be somewhere else? Should probably make an inference interface...
		// obsNorm: if non-null, RSNorm is applied as the first op (stop-grad) to obs.
		static torch::Tensor InferPolicyProbsFromModels(
			ModelSet& models,
			torch::Tensor obs, torch::Tensor actionMasks,
			float temperature,
			bool halfPrec,
			const RSNorm* obsNorm = nullptr
		);
		static void InferActionsFromModels(
			ModelSet& models,
			torch::Tensor obs, torch::Tensor actionMasks,
			bool deterministic, float temperature, bool halfPrec,
			torch::Tensor* outActions, torch::Tensor* outLogProbs,
			torch::Tensor* outActionProbs = NULL,
			const RSNorm* obsNorm = nullptr
		);

		void Learn(ExperienceBuffer& experience, Report& report, bool isFirstIteration, uint64_t totalTimesteps);

		// Potential-based GCRL shaping (used when config.contrastiveGoal.usePotentialShaping): returns
		// a per-row reward addon summing gamma*Phi_k(s') - Phi_k(s) over heads, computed from the current
		// (previous-iteration) critics, to be added to the reward stream BEFORE GAE. contactGoal [1,6]
		// is the car head's fixed goal; scoringRangeGoals [K,6] are the goal head's mouth samples.
		torch::Tensor ComputePotentialShaping(
			torch::Tensor states, torch::Tensor actionMasks, torch::Tensor segmentIds, torch::Tensor terminals,
			torch::Tensor truncNextStates, float gaeGamma, torch::Tensor contactGoal, torch::Tensor scoringRangeGoals,
			torch::Tensor defenseGroupKeys, torch::Tensor defenseTeams, Report& report);

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
