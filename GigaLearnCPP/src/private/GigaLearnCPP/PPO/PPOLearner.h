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
		
		// Steered-practice collection (LearnerConfig::steering): unit commitment direction on
		// this->device + its projection std + strength. Applied ONLY where InferActions gets a
		// row mask - the learn pass, value preds, and old-version inference never pass one.
		torch::Tensor steerVec;
		float steerSigma = 0, steerAlpha = 0;
		void SetSteering(torch::Tensor vecCpu, float sigma, float alpha);

		// Rho-band gate on the steering delta (LearnerConfig::CollectSteeringConfig): steer a
		// row only when its scoring-reachability sits in the middle band of the current
		// batch's rho distribution. Set by the Learner at startup; reads phi/psi from the
		// SAME ModelSet as the policy (the pipelined worker's snapshot includes them, so no
		// concurrent read of live weights).
		bool steerRhoGate = false;
		bool steerRhoContact = true; // gate on the car head's contact reachability (races)
		float steerRhoLo = 0.2f, steerRhoHi = 0.8f;
		int steerRhoK = 8;
		float lastRhoGateFrac = 0; // metric: fraction of eligible rows steered last call

		// If models is null, this->models will be used. steerRowMask (optional, [n] bool, any
		// device): rows to steer with steerAlpha*steerSigma*steerVec added to the trunk output.
		void InferActions(torch::Tensor obs, torch::Tensor actionMasks, torch::Tensor* outActions, torch::Tensor* outLogProbs, ModelSet* models = NULL, torch::Tensor steerRowMask = {});
		torch::Tensor InferCritic(torch::Tensor obs);
		// Secondary goal-only critic (independent net, raw obs). Only valid when goalCritic.enabled.
		torch::Tensor InferGoalCritic(torch::Tensor obs);

		// Perhaps they should be somewhere else? Should probably make an inference interface...
		// steerDelta (optional, [n, trunkOut] or [1, trunkOut]): added to the shared-head output
		// before the policy head. Requires a shared head. Collection-only - see SetSteering.
		static torch::Tensor InferPolicyProbsFromModels(
			ModelSet& models,
			torch::Tensor obs, torch::Tensor actionMasks,
			float temperature,
			bool halfPrec,
			torch::Tensor steerDelta = {}
		);
		static void InferActionsFromModels(
			ModelSet& models,
			torch::Tensor obs, torch::Tensor actionMasks,
			bool deterministic, float temperature, bool halfPrec,
			torch::Tensor* outActions, torch::Tensor* outLogProbs,
			torch::Tensor steerDelta = {}
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