#pragma once
#include "ExperienceBuffer.h"
#include <GigaLearnCPP/Util/Report.h>
#include <GigaLearnCPP/Util/Timer.h>
#include <GigaLearnCPP/PPO/PPOLearnerConfig.h>
#include <GigaLearnCPP/PPO/TransferLearnConfig.h>

#include "../Util/Models.h"

#include <torch/optim/adam.h>
#include <torch/nn/modules/loss.h>
#include <torch/nn/modules/container/sequential.h>

#include "ExperienceBuffer.h"
#include <deque>

namespace GGL {

	// https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/ppo/ppo_learner.py
	class PPOLearner {
	public:
		float curEntropyScale;
		float curGCRLAdvScale;
		float curGCRLRewardGateInfluence;
		float curSORSRewardScale;

		ModelSet models = {};
		ModelSet guidingPolicyModels = {};

		struct SORSWindow {
			FList states;
			FList actionComps;
			float label = 0;
			int length = 0;
		};
		std::deque<SORSWindow> sorsReplay;
		int64_t sorsTrainCalls = 0;

		PPOLearnerConfig config;
		torch::Device device;

		PPOLearner(
			int obsSize, int numActions,
			PPOLearnerConfig config, torch::Device device
		);

		static void MakeModels(
			bool makeCritic, 
			int obsSize, int numActions, 
			PartialModelConfig sharedHeadConfig, PartialModelConfig policyConfig, PartialModelConfig criticConfig,
			torch::Device device,
			ModelSet& outModels
		);
		
		// If models is null, this->models will be used
		void InferActions(torch::Tensor obs, torch::Tensor actionMasks, torch::Tensor* outActions, torch::Tensor* outLogProbs, ModelSet* models = NULL);
		torch::Tensor InferCritic(torch::Tensor obs);
		torch::Tensor InferGCRLRewardGate(torch::Tensor obs, torch::Tensor actionComps, torch::Tensor goalTargets, torch::Tensor antiTargets);
		torch::Tensor InferSORSRewards(torch::Tensor obs, torch::Tensor actionComps);
		void AddSORSWindows(std::vector<SORSWindow>&& windows);
		void TrainSORS(Report& report);

		// Perhaps they should be somewhere else? Should probably make an inference interface...
		static torch::Tensor InferPolicyProbsFromModels(
			ModelSet& models, 
			torch::Tensor obs, torch::Tensor actionMasks, 
			float temperature,
			bool halfPrec
		);
		static void InferActionsFromModels(
			ModelSet& models, 
			torch::Tensor obs, torch::Tensor actionMasks, 
			bool deterministic, float temperature, bool halfPrec,
			torch::Tensor* outActions, torch::Tensor* outLogProbs
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
		float GetEntropyScale() const;

		ModelSet GetPolicyModels();

	private:
		void SavePPOState(std::filesystem::path folderPath);
		void LoadPPOState(std::filesystem::path folderPath);
		void SaveSORSState(std::filesystem::path folderPath);
		void LoadSORSState(std::filesystem::path folderPath);
	};
}
