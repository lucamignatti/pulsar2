#pragma once
#include "ExperienceBuffer.h"
#include <GigaLearnCPP/Util/Report.h>
#include <GigaLearnCPP/Util/Timer.h>
#include <GigaLearnCPP/PPO/PPOLearnerConfig.h>

#include "../Util/Models.h"
#include "../Util/ObsMirror.h"
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
		Dist::Session* dist = nullptr;

		PPOLearner(
			int obsSize, int numActions,
			PPOLearnerConfig config, torch::Device device,
			Dist::Session* dist = nullptr
		);
		~PPOLearner();

		static void MakeModels(
			bool makeCritic,
			int obsSize, int numActions,
			PartialModelConfig sharedHeadConfig, PartialModelConfig policyConfig, PartialModelConfig criticConfig,
			PartialModelConfig criticTrunkConfig,
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
		// The value-side body: main trunk, then the critic trunk if one is configured. EVERY value
		// head (critic, goal critic, V-dagger twins, r-hat twins) reads this, so they all see the
		// same features and the trunks are forwarded once per call instead of once per head.
		torch::Tensor ValueTrunk(torch::Tensor obs, bool halfPrec);
		torch::Tensor InferCritic(torch::Tensor obs);
		// Secondary goal-only critic. Reads the CRITIC TRUNK when one is configured; with no critic
		// trunk it keeps its historical independent form (own net, raw obs, no shared body).
		torch::Tensor InferGoalCritic(torch::Tensor obs);
		// HEADROOM composition critic: min of the twin V-dagger heads (shared trunk).
		// No-grad; used at learn-prep for one-iteration-frozen TD targets + the H field.
		torch::Tensor InferVdagMin(torch::Tensor obs);
		// THEORY: max of the twin reward models (optimism under ambiguity), clamped by
		// the caller to the largest reward actually observed (never invent magnitudes).
		torch::Tensor InferRhatMax(torch::Tensor obs);
		// GEOMETRY (4th rung): V_geo, the HJB fixed point. No-grad read for the field.
		torch::Tensor InferGeoV(torch::Tensor obs);
		// FUSED consumption-path read: ONE upload and ONE shared_head+critic_trunk forward
		// feeding critic / goal critic / V-dagger min, plus geo_v off the same upload. Pass
		// nullptr for any head you do not need. Prefer this over calling the singles in
		// separate loops — that forwards the trunk once PER CALL (see the note on the impl).
		// V-dagger (and r-hat) read the UNCONDITIONED trunk, matching InferVdagMin and Learn.
		// opp_embed is added only for critic / goal critic.
		void InferValueFamily(
			torch::Tensor obs, torch::Tensor* outCritic, torch::Tensor* outGoalCritic,
			torch::Tensor* outVdagMin, torch::Tensor* outGeoV);
		// Reservoir over (obs, nextObs, scaledReward) for the STATIONARY world-facing fits
		// (r_hat, Sigma). Reward and one-step displacement spread are properties of the
		// environment; only where we sample them moves as the policy changes. Fitting them on
		// the sliding buffer makes them forget regions the policy left, which is what made the
		// HJB residual run away 100-250x offline.
		torch::Tensor geoResObs, geoResNext, geoResRew;
		int64_t geoResFill = 0, geoResSeen = 0;
		void GeoReservoirAdd(torch::Tensor obs, torch::Tensor nextObs, torch::Tensor rew, int cap,
			torch::Tensor keepMask = {}, torch::Tensor ret = {});
		float dbgGeoResid = -1.f, dbgGeoRew = -1.f, dbgGeoMean = -9.f;

		// ===== HULL OPERATOR (record-licensed relaxed Bellman; EPSILON_CRITIC.md s7) =====
		// For each next-state row: find hullK nearest donor states in the learned dynamics
		// chart, transplant their eps-scaled witnessed displacement vectors, and return the
		// elementwise MAX of InferVdagMin over the perturbed candidates. The donor bank is a
		// per-iteration subsample of the geo reservoir (teleport-filtered at the feed).
		// Returns a CPU float tensor shaped like `states`' first dim; empty tensor when the
		// bank is not ready (caller falls back to the plain bootstrap).
		torch::Tensor HullBootstrap(torch::Tensor states);
		float dbgHullNLL = -999.f, dbgHullL1 = -1.f;   // NLL can be legitimately negative

		// ===== COMPOSITE VALUE CRITIC =====
		// Mirror map built lazily at first Learn (needs the runtime obs size); the
		// opponent context is set by the Learner: oppCtxLive for collection-time
		// inference (the worker knows its iteration's opponent), per-row batch.oppCtx
		// for the learn pass (pipelining means live != batch iteration).
		ObsMirror::Map mirrorMap;
		torch::Tensor oppCtxLive;          // [oppCtxDim] on device; zeros = self-play
		torch::Tensor oppCtxCollected;     // CPU; written by the collect worker at its draw
		torch::Tensor oppCtxForLearn;      // CPU; barrier-copied so learn sees ITS iteration
		float valueEvEma = 0.f;            // explained-variance EMA (drives the epi blend)
		torch::Tensor geoResRet;           // reservoir returns column (episodic blend)
		float dbgAuxNLL = -999.f, dbgTwinDisagree = -1.f;
		// ARCHIVE of field-ascent transitions (persistent; re-scored at replay).
		float rhatMaxObserved = 0.f;
		float dbgVdagRows = -1.f, dbgRhatRows = -1.f, dbgRhatEntry = -9.f, dbgVdagRaw = -9.f, dbgYvAbs = -9.f;   // bound on hypothesis magnitude (never invent)

		// ===== IMPLICIT WORLD MODEL =====
		torch::Tensor InferImagValue(torch::Tensor obs);
		void TrainWorldModel(torch::Tensor states, torch::Tensor actions, torch::Tensor cont);
		void TrainImagValue(torch::Tensor states);
		// One sweep of optimistic value iteration for the given (device) states. When
		// outRing is non-null it receives the TRUSTED imagined successors, which are then
		// trained on as well so the solved region advances one ring beyond the data.
		torch::Tensor ImagTargetsFor(torch::Tensor sDev, torch::Tensor* outRing);
		int numActionsCached = 0, obsSizeCached = 0;
		float dbgWmDyn = -1.f, dbgWmVi = -1.f, dbgWmTrust = -1.f, dbgImag = -1.f;
		float dbgEntGate = -1.f;
		float dbgSilLoss = -1.f;

		// Perhaps they should be somewhere else? Should probably make an inference interface...
		// steerDelta (optional, [n, trunkOut] or [1, trunkOut]): added to the shared-head output
		// before the policy head. Requires a shared head. Collection-only (opponent styles).
		// outRowOk (optional): when non-null, the non-finite-logits guard does NOT sync here.
		// Bad rows are sanitized to uniform (sampling stays safe) and the per-row finite flags
		// are returned for a DEFERRED verdict at the caller's existing sync point. When null,
		// the original synchronous check-and-die with full forensics runs in place.
		// precomputedTrunk (optional): an already-forwarded shared_head output for these exact
		// rows. Pass it from the learn pass, where the value family forwards the trunk anyway —
		// otherwise this function runs shared_head itself and the main trunk is built AND
		// backpropped twice per minibatch. Leave undefined on collection/eval paths, which have
		// no other consumer to share with.
		static torch::Tensor InferPolicyProbsFromModels(
			ModelSet& models,
			torch::Tensor obs, torch::Tensor actionMasks,
			float temperature,
			bool halfPrec,
			torch::Tensor steerDelta = {},
			torch::Tensor* outRowOk = nullptr,
			torch::Tensor precomputedTrunk = {});
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

		// FP16 learn autocast (V100): persist scale across Learn() calls. BF16 leaves this unused.
		float ampLossScale = 4096.f;
		int ampGoodEpochs = 0;
		int ampSkipCount = 0;

		void Learn(ExperienceBuffer& experience, Report& report, bool isFirstIteration);

		void SaveTo(std::filesystem::path folderPath);
		// Re-read the just-written checkpoint and diff it against the live weights.
		// False = the file does not hold what we think it holds; do not publish it.
		bool VerifySavedWeights(std::filesystem::path folderPath);
		void LoadFrom(std::filesystem::path folderPath);
		void SetLearningRates(float policyLR, float criticLR);

		ModelSet GetPolicyModels();
	};
}