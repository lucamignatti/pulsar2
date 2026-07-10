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
		bool OnDescendIteration(Report& report, float rating);

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

		// The full probe round: partition arenas, perturb, finetune factors, score, ES-fold.
		void RunProbeRound(Report& report);

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
