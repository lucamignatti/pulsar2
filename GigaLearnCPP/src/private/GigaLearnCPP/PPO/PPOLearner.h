#pragma once
#include "ExperienceBuffer.h"
#include <GigaLearnCPP/Util/Report.h>
#include <GigaLearnCPP/Util/Timer.h>
#include <GigaLearnCPP/PPO/PPOLearnerConfig.h>

#include "../Util/Models.h"
#include "Reachability.h"
#include "Proposer.h"

#include <torch/optim/adam.h>
#include <torch/nn/modules/loss.h>
#include <torch/nn/modules/container/sequential.h>

#include "ExperienceBuffer.h"

namespace GGL {

	// Optimistic-Critic Ladder WIRE state (LADDER.md; one generation of everything
	// needed to compute the policy's 5 extra inputs for a batch of rows):
	//   wire = tanh([V_real, V_exp, gap_KD, V_metric, gap_PK] / scale)
	// V_real comes from the SAME ModelSet as the policy forward (the pipelined
	// snapshot carries the critic when the wire is on, so the worker never reads
	// weights Learn is updating). exp/mapE/mapF here are either snapshot copies
	// (collection generation) or the live nets (learn generation); bankG/bankC are
	// the bank obs' map embeddings and a/a2/b the V_metric calibration - at learn
	// time these stay the COLLECTION-time snapshot while the heads are current (the
	// spec's tested re-derivation rule; storing floats is the contingent fallback).
	// Everything is read no-grad and detached: the wire is semantically an
	// observation, never a gradient path (Laws 2/4).
	struct LadderWire {
		bool active = false;       // false = wire feeds exact zeros (pre-warmup / eval)
		bool banksReady = false;   // false = V_metric := V_exp (gap_PK = 0)
		torch::nn::Sequential exp{ nullptr }, mapE{ nullptr }, mapF{ nullptr };
		torch::Tensor bankG, bankC; // [k, 32] map embeddings of the bank obs, on device
		float a = 0, a2 = 0, b = 0; // V_metric calibration snapshot
		float gamma = 1, dClamp = 1, scale = 3;
	};

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
			ModelSet& outModels,
			int extraPolicyInputs = 0 // Ladder wire columns appended to the policy head input
		);

		// Ladder wire generations (owned by the Learner's gap state, assigned each
		// iteration): `ladderCollect` is what the collection forward uses (snapshot
		// nets + snapshot banks; passed explicitly by the collection call site);
		// `ladderLearn` is what Learn()'s re-derivation uses (live nets + the
		// collection generation's banks/calibration). When the policy carries wire
		// columns, Learn() REFUSES to run unless ladderLearn was set this iteration -
		// a silently-zeroed wire biases the PPO ratio (the spec's no-silent-fallback
		// rule; the one hard bug class of the tested configuration).
		LadderWire ladderCollect, ladderLearn;
		bool ladderLearnSet = false;
		
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
		int steerGoalOverrideHead = 0;   // 0 = car-local ball, 1 = canonical ball, 2 = canonical car state
		void SetSteerGoal(torch::Tensor goal6Cpu, int head); // undefined tensor = clear

		// If models is null, this->models will be used. steerRowMask (optional, [n] bool, any
		// device): rows eligible for steering; steerRowModes ([n] int64, REQUIRED when the
		// mask is passed): each row's mode index (playersPerTeam-1) selecting the direction.
		// styleVec/styleCoef (optional): OPPONENT-side style steering (league phase 1/2) -
		// coef * vec added to the trunk output of EVERY row of this call, ungated (style is
		// a whole-game disposition, not a frontier read). Callers pass it only on the
		// old-version/league-opponent inference call, never on the trained policy's.
		void InferActions(torch::Tensor obs, torch::Tensor actionMasks, torch::Tensor* outActions, torch::Tensor* outLogProbs, ModelSet* models = NULL, torch::Tensor steerRowMask = {}, torch::Tensor steerRowModes = {}, torch::Tensor styleVec = {}, float styleCoef = 0, const LadderWire* ladder = NULL);
		torch::Tensor InferCritic(torch::Tensor obs);
		// Secondary goal-only critic (independent net, raw obs). Only valid when goalCritic.enabled.
		torch::Tensor InferGoalCritic(torch::Tensor obs);
		// HEADROOM composition critic: min of the twin V-dagger heads (shared trunk).
		// No-grad; used at learn-prep for one-iteration-frozen TD targets + the H field.
		torch::Tensor InferVdagMin(torch::Tensor obs);

		// Perhaps they should be somewhere else? Should probably make an inference interface...
		// steerDelta (optional, [n, trunkOut] or [1, trunkOut]): added to the shared-head output
		// before the policy head. Requires a shared head. Collection-only - see SetSteering.
		static torch::Tensor InferPolicyProbsFromModels(
			ModelSet& models,
			torch::Tensor obs, torch::Tensor actionMasks,
			float temperature,
			bool halfPrec,
			torch::Tensor steerDelta = {},
			// Wire source when the policy head carries extra input columns. NULL (or
			// inactive) = exact zeros: correct for eval/opponent/render paths (Rating
			// measures the raw policy; both sides of an eval get the same zeros) and
			// for pre-warmup collection. The LEARN path never passes NULL - see
			// ladderLearnSet.
			const LadderWire* ladder = NULL
		);
		static void InferActionsFromModels(
			ModelSet& models,
			torch::Tensor obs, torch::Tensor actionMasks,
			bool deterministic, float temperature, bool halfPrec,
			torch::Tensor* outActions, torch::Tensor* outLogProbs,
			torch::Tensor steerDelta = {},
			const LadderWire* ladder = NULL
		);
		// d(x, bank) = min over bank rows of sum_j relu(x_j - bank_j), clamped;
		// chunked so the [rows, bank, dims] broadcast never materializes at full n.
		static torch::Tensor MinBankDist(torch::Tensor emb, torch::Tensor bank, float dClamp);
		// The 5 wire values for a batch: rawObs feeds the map encoder, trunkOut feeds
		// critic/expectile. Detached, fp32 for the gamma^d map (bf16-safe upstream).
		static torch::Tensor ComputeWire(ModelSet& models, const LadderWire& lw,
			torch::Tensor rawObs, torch::Tensor trunkOut, bool halfPrec);

		void Learn(ExperienceBuffer& experience, Report& report, bool isFirstIteration);

		void SaveTo(std::filesystem::path folderPath);
		void LoadFrom(std::filesystem::path folderPath);
		void SetLearningRates(float policyLR, float criticLR);

		ModelSet GetPolicyModels();
	};
}