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

		// Adaptive entropy-bonus scale (controller state). Initialized from
		// config.entropyScale; when config.adaptiveEntropy, updated each Learn()
		// toward config.targetEntropy and clamped to [0, config.maxEntropyScale].
		// Persisted in the checkpoint (Learner::SaveStats/LoadStats) and re-clamped
		// against current config on resume.
		float curEntropyScale = 0.f;

		// TRIAD-NATIVE GCRL coupling controller state (persisted in Learner::SaveStats/LoadStats):
		// gcrlLambdaEff  = the ratio-controlled lambda target (warmup-ramped before use each iter);
		// gcrlRatioEma   = EMA of std(gcrlAdv)/std(baseNorm), driven toward config gcrlRatioTarget (~1:1);
		// gcrlRenormStd  = EMA of std(sepSum) for RenormToStd (the inner z533fbde explosion guard).
		float gcrlLambdaEff = 0.3f;
		float gcrlRatioEma = 1.f;
		float gcrlRenormStd = 1.f;

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
		// Critic value prediction against an explicit model set + normalizer (so overlapped collection can
		// score the GAE baseline with a FROZEN actor clone + RSNorm snapshot instead of the live, mutating models).
		static torch::Tensor InferCriticFromModels(
			ModelSet& models, torch::Tensor obs, bool halfPrec, const RSNorm* obsNorm = nullptr
		);

		void Learn(ExperienceBuffer& experience, Report& report, bool isFirstIteration, uint64_t totalTimesteps);

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
