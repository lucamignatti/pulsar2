#pragma once
#include "ExperienceBuffer.h"
#include <GigaLearnCPP/Util/Report.h>
#include <GigaLearnCPP/Util/Timer.h>
#include <GigaLearnCPP/PPO/PPOLearnerConfig.h>

#include "../Util/Models.h"
#include "Reachability.h"

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

		// inside `models` and saves/loads with everything else, but is EXCLUDED from GetPolicyModels()
		// (old policy versions predate it) and is groupStepExempt (steps itself, never via the PPO loop)

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
			ModelSet& outModels);

		// Ladder wire generations (owned by the Learner's gap state, assigned each
		// iteration): `ladderCollect` is what the collection forward uses (snapshot
		// nets + snapshot banks; passed explicitly by the collection call site);
		// `ladderLearn` is what Learn()'s re-derivation uses (live nets + the
		// collection generation's banks/calibration). When the policy carries wire
		// columns, Learn() REFUSES to run unless ladderLearn was set this iteration -
		// a silently-zeroed wire biases the PPO ratio (the spec's no-silent-fallback
		// rule; the one hard bug class of the tested configuration).
		
		// Steered-practice collection (LearnerConfig::steering), PER-MODE (4.0 team play):
		// one unit commitment direction per team size (index = playersPerTeam-1), each with
		// its own projection std and strength - the commitment frontier is mode-specific
		// (offline: collective declines on feasible balls grow 74%->88%->92% from 1v1->3v3).
		// Applied ONLY where InferActions gets a row mask + row modes - the learn pass, value
		// preds, and old-version inference never pass them.
		static constexpr int STEER_MODES = 3;

		// Rho-band gate on the steering delta (LearnerConfig::CollectSteeringConfig): steer a
		// row only when its contact-reachability sits in the middle band of the current
		// batch's rho distribution. Bands are computed PER MODE (rho distributions differ
		// across team sizes; a mixed-mode quantile would skew the band toward the dominant
		// mode). Set by the Learner at startup; reads phi/psi from the SAME ModelSet as the
		// policy (the pipelined worker's snapshot includes them, so no concurrent read of
		// live weights).
		float lastRhoGateFrac = 0; // metric: fraction of eligible rows steered last call

		// META gate-goal override: when defined, the rho band scores reachability toward
		// THIS goal (the active emergent cluster's representative, an achieved state from
		// the agent's own bank) on the given head, replacing the fixed contact/scoring
		// default. Set in the barrier zone only (the worker reads it unsynchronized).
		void SetSteerGoal(torch::Tensor goal6Cpu, int head); // undefined tensor = clear

		// If models is null, this->models will be used - which is how the opponent half of a
		// served iteration is inferred (pass the archived version's ModelSet).
		void InferActions(torch::Tensor obs, torch::Tensor actionMasks, torch::Tensor* outActions, torch::Tensor* outLogProbs, ModelSet* models = NULL);
		torch::Tensor InferCritic(torch::Tensor obs);
		// Secondary goal-only critic (independent net, raw obs). Only valid when goalCritic.enabled.
		torch::Tensor InferGoalCritic(torch::Tensor obs);
		// HEADROOM composition critic: min of the twin V-dagger heads (shared trunk).
		// No-grad; used at learn-prep for one-iteration-frozen TD targets + the H field.
		torch::Tensor InferVdagMin(torch::Tensor obs);
		// THEORY: max of the twin reward models (optimism under ambiguity), clamped by
		// the caller to the largest reward actually observed (never invent magnitudes).
		torch::Tensor InferRhatMax(torch::Tensor obs);
		// ARCHIVE of field-ascent transitions (persistent; re-scored at replay).
		torch::Tensor archObs, archNextObs, archAct;
		float rhatMaxObserved = 0.f;
		float dbgVdagRows = -1.f, dbgRhatRows = -1.f, dbgRhatEntry = -9.f, dbgVdagRaw = -9.f, dbgYvAbs = -9.f;
		int64_t learnCalls = 0; // seeding warmup gate   // bound on hypothesis magnitude (never invent)
		int64_t archFill = 0, archPtr = 0;
		void BankAscent(torch::Tensor obs, torch::Tensor nextObs, torch::Tensor acts, int cap);

		// Perhaps they should be somewhere else? Should probably make an inference interface...
		// steerDelta (optional, [n, trunkOut] or [1, trunkOut]): added to the shared-head output
		// before the policy head. Requires a shared head. Collection-only (opponent styles).
		static torch::Tensor InferPolicyProbsFromModels(
			ModelSet& models,
			torch::Tensor obs, torch::Tensor actionMasks,
			float temperature,
			bool halfPrec,
			torch::Tensor steerDelta = {});
		static void InferActionsFromModels(
			ModelSet& models,
			torch::Tensor obs, torch::Tensor actionMasks,
			bool deterministic, float temperature, bool halfPrec,
			torch::Tensor* outActions, torch::Tensor* outLogProbs,
			torch::Tensor steerDelta = {}
		);
		// d(x, bank) = min over bank rows of sum_j relu(x_j - bank_j), clamped;
		// chunked so the [rows, bank, dims] broadcast never materializes at full n.
		// The 5 wire values for a batch: rawObs feeds the map encoder, trunkOut feeds
		// critic/expectile. Detached, fp32 for the gamma^d map (bf16-safe upstream).

		void Learn(ExperienceBuffer& experience, Report& report, bool isFirstIteration);

		void SaveTo(std::filesystem::path folderPath);
		void LoadFrom(std::filesystem::path folderPath);
		void SetLearningRates(float policyLR, float criticLR);

		ModelSet GetPolicyModels();
	};
}