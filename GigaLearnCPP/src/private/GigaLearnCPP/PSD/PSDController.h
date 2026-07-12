#pragma once
#include "../FrameworkTorch.h"
#include "PolicySlots.h"
#include <GigaLearnCPP/PSDConfig.h>
#include <GigaLearnCPP/Util/Report.h>

#include <RLGymCPP/EnvSet/EnvSet.h>
#include <nlohmann/json.hpp>

namespace GGL {

	using PSD::PolicySlots;

	class PPOLearner;

	// Owns the Probe–Step–Descend outer loop. Additive: the Learner runs ordinary DESCEND
	// iterations and calls OnDescendIteration() at the end of each. When a plateau is detected
	// (and warmup satisfied), the controller runs a full PROBE round in-line — perturb, finetune,
	// score, fold — then hands control back to DESCEND. All persistent state lives here and is
	// serialised into the checkpoint stats JSON so a crash mid-round resumes cleanly.
	class PSDController {
	public:
		PSDConfig cfg;
		torch::Device device;

		// --- persistent state (checkpointed) ---
		int round = 0;                  // completed probe rounds
		float sigma;                    // current perturbation scale (adapted between rounds)
		uint64_t rngCounter = 0;        // ever-increasing; the round noise seed derives from it
		float ratingAtPhaseStart = 0;   // Rating/1v1 at the start of the current DESCEND phase
		bool havePhaseStartRating = false;
		int descendItersThisPhase = 0;

		// --- plasticity-intervention state (checkpointed) ---
		float effRankMax = 0;           // running peak of the policy-head effective rank
		int roundsSinceDistill = 0;     // probe rounds since the last distill reset
		bool trunkFrozen = false;       // whether the shared trunk is currently frozen
		float lastRating = NAN;         // latest Rating/1v1 seen (for competence conditioning)
		uint64_t curTotalIters = 0;     // latest training-iteration count (gates intervention warmup)

		// --- context (set by the Learner, not owned) ---
		RLGC::EnvSet* envSet = nullptr;
		PPOLearner* ppo = nullptr;
		int numArenas = 0;
		std::filesystem::path roundsDir;    // <checkpointFolder>/psd_rounds

		PSDController(const PSDConfig& cfg, torch::Device device) : cfg(cfg), device(device), sigma(cfg.sigma) {}

		void Init(RLGC::EnvSet* envSet, PPOLearner* ppo, int numArenas, std::filesystem::path checkpointFolder);

		// Called at the END of every DESCEND iteration. `rating` is the latest Rating/1v1
		// (pass NaN if the skill tracker hasn't produced one this iteration). Returns true if a
		// probe round was run this call (so the Learner can finalise/clear its trajectories).
		bool OnDescendIteration(Report& report, float rating, uint64_t totalIterations);

		// Serialise / restore the persistent state.
		void ToJSON(nlohmann::json& j) const;
		void FromJSON(const nlohmann::json& j);

	private:
		// Recorded probe-block transitions for the finetune (rows grouped by slot each step).
		struct ProbeTape {
			std::vector<torch::Tensor> obs;        // per step: [S*rpp, obsSize]
			std::vector<torch::Tensor> masks;      // per step: [S*rpp, numActions] uint8
			std::vector<torch::Tensor> actions;    // per step: [S*rpp] long
			std::vector<torch::Tensor> logProbs;   // per step: [S*rpp] float
			std::vector<torch::Tensor> rewards;    // per step: [S*rpp] float
			int rpp = 0;                           // rows per slot in the probe block
		};

		// The full probe round. Dispatches on cfg.pureES.
		void RunProbeRound(Report& report);
		// Pure-ES probe (default): large population, no finetune, level fitness over a long window,
		// validation-gated fold. This is the EGGROLL-faithful path (arXiv 2511.16652 §6.1).
		void RunProbeRoundPureES(Report& report);
		// Legacy Baldwinian probe: small population, per-slot factor finetune, slope fitness. Kept for
		// the phase-2 "few probes, long finetune" racing experiment; off unless cfg.pureES=false.
		void RunProbeRoundBaldwinian(Report& report);

		// Mean per-player per-step reward of the UNPERTURBED base policy over a held-out window
		// (base-vs-base mirror). The tighten-rule baseline for the pure-ES round.
		float MeasureAggregateReturn(int windowSteps);

		// Pure-ES asymmetric eval step: seat 0 of each eval arena is driven by that arena's slot policy,
		// the opponent seat(s) by the current base policy, so each slot plays a real match against the
		// current self. seat0Idx (len S*perSlotEval, slot-major) lists the seat-0 player rows.
		std::vector<int> StepActionsAsymmetric(PolicySlots& ps, const std::vector<int>& seat0Idx);

		// Head-to-head: play the freshly folded policy (seat 0) vs the pre-fold policy `oldPolicy`
		// (seat 1) across the whole pool; return the mean (seat0 - seat1) reward differential. The
		// validation A/B for option D — dominated by the zero-sum competitive terms, so it tests "did
		// the fold actually beat the policy it replaces", not who farmed more proximity.
		float MeasureHeadToHead(Model* oldPolicy, int windowSteps);

		// Log the plasticity signals and apply the enabled interventions to the just-folded weights.
		void RunInterventions(Report& report);

		// Deepest reset: reinit the policy sub-net and behavior-distill the current policy into it over
		// the frozen trunk, then clear the policy optimizer moments. Called only when gated.
		void DistillReset(Report& report);

		// Build the full per-player action vector for the current env step: base policy for every
		// player, then overwrite the [probe|eval] blocks with slot-routed actions. Records probe
		// transitions into `tape` and accumulates eval per-slot reward into `evalRewardAccum`.
		std::vector<int> StepActions(PolicySlots& ps, int probeArenaStart, int evalArenaStart,
			int perSlotProbeArenas, int perSlotEvalArenas,
			ProbeTape* tape, std::vector<double>* evalRewardAccum);

		// One PPO-clip update of the factors from the recorded probe tape (base + trunk frozen).
		float FactorUpdate(PolicySlots& ps, torch::optim::Optimizer& opt, const ProbeTape& tape);

		void PersistRound(const std::vector<float>& fitness, const std::vector<float>& weights, float stepNorm);
	};
}
