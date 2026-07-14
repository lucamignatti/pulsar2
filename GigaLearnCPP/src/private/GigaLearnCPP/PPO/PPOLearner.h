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
		
		// Steered-practice collection (LearnerConfig::steering), PER-MODE (4.0 team play):
		// one unit commitment direction per team size (index = playersPerTeam-1), each with
		// its own projection std and strength - the commitment frontier is mode-specific
		// (offline: collective declines on feasible balls grow 74%->88%->92% from 1v1->3v3).
		// Applied ONLY where InferActions gets a row mask + row modes - the learn pass, value
		// preds, and old-version inference never pass them.
		static constexpr int STEER_MODES = 3;
		std::array<torch::Tensor, STEER_MODES> steerVecs; // unit, on this->device (undefined = mode off)
		std::array<float, STEER_MODES> steerSigmas = {}, steerAlphas = {};
		void SetSteering(const std::array<torch::Tensor, STEER_MODES>& vecsCpu,
			const std::array<float, STEER_MODES>& sigmas,
			const std::array<float, STEER_MODES>& alphas);

		// Rho-band gate on the steering delta (LearnerConfig::CollectSteeringConfig): steer a
		// row only when its contact-reachability sits in the middle band of the current
		// batch's rho distribution. Bands are computed PER MODE (rho distributions differ
		// across team sizes; a mixed-mode quantile would skew the band toward the dominant
		// mode). Set by the Learner at startup; reads phi/psi from the SAME ModelSet as the
		// policy (the pipelined worker's snapshot includes them, so no concurrent read of
		// live weights).
		bool steerRhoGate = false;
		bool steerRhoContact = true; // gate on the car head's contact reachability (races)
		float steerRhoLo = 0.2f, steerRhoHi = 0.8f;
		int steerRhoK = 8;
		float lastRhoGateFrac = 0; // metric: fraction of eligible rows steered last call

		// META gate-goal override: when defined, the rho band scores reachability toward
		// THIS goal (the active emergent cluster's representative, an achieved state from
		// the agent's own bank) on the given head, replacing the fixed contact/scoring
		// default. Set in the barrier zone only (the worker reads it unsynchronized).
		torch::Tensor steerGoalOverride; // [6], goal-space normalized, on `device`
		bool steerGoalOverrideCar = false;
		void SetSteerGoal(torch::Tensor goal6Cpu, bool carHead); // undefined tensor = clear

		// If models is null, this->models will be used. steerRowMask (optional, [n] bool, any
		// device): rows eligible for steering; steerRowModes ([n] int64, REQUIRED when the
		// mask is passed): each row's mode index (playersPerTeam-1) selecting the direction.
		// styleVec/styleCoef (optional): OPPONENT-side style steering (league phase 1/2) -
		// coef * vec added to the trunk output of EVERY row of this call, ungated (style is
		// a whole-game disposition, not a frontier read). Callers pass it only on the
		// old-version/league-opponent inference call, never on the trained policy's.
		void InferActions(torch::Tensor obs, torch::Tensor actionMasks, torch::Tensor* outActions, torch::Tensor* outLogProbs, ModelSet* models = NULL, torch::Tensor steerRowMask = {}, torch::Tensor steerRowModes = {}, torch::Tensor styleVec = {}, float styleCoef = 0);
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