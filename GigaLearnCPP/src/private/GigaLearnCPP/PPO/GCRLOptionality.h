#pragma once
#include "../Util/Models.h"
#include <GigaLearnCPP/Util/Report.h>

namespace GGL {

	// ── Optionality potential shaping (Feature D) ────────────────────────────
	// Owns the three pieces of the phi_opt machinery:
	//  1. A FROZEN scorer: Polyak-target copies of the goal critic's phi/psi towers
	//     (and the shared head, if one exists). All phi_opt scoring runs through these,
	//     never the live nets — a fast-moving potential breaks the policy-invariance
	//     argument of potential-based shaping. They live outside the ModelSet, so they
	//     get no optimizer steps, no checkpoint files, and cannot leak into training.
	//     No new trained parameters anywhere in this feature.
	//  2. A stratified goal bank of raw 6-dim ball-goal rows (obs space), re-embedded
	//     under the frozen psi after every Polyak update. Each entry is scored together
	//     with its x-flipped twin, which makes phi_opt invariant to the obs x-mirror
	//     frame without any per-state frame bookkeeping (by left/right symmetry the
	//     twin is an equally valid goal).
	//  3. phi_opt(s) = T * logsumexp_g(V(g)-d(s,g)/T) - T * log|G| over the doubled bank,
	//     with action components pinned to ZERO so the potential is a function of
	//     state only (phi takes (s, a); a fixed constant action input keeps it a pure
	//     state function, which the policy-invariance argument requires).
	class GCRLOptionality {
	public:
		PPOLearnerConfig config;
		torch::Device device;

		QuasimetricCritic* targetCritic; // frozen phi/psi clone of goal_critic
		Model* targetSharedHead = nullptr; // frozen shared_head clone (only if live exists)

		// Stratum-partitioned ring: stratum s owns slots [offset, offset+cap), FIFO
		// within the stratum. Starved strata carry their refresh quota over as debt
		// instead of backfilling from another stratum.
		enum StratumIdx { STRATUM_OFFENSIVE = 0, STRATUM_DEFENSIVE = 1, STRATUM_RESOURCE = 2, STRATUM_AMOUNT = 3 };
		struct BankStratum {
			int offset = 0, cap = 0;
			int count = 0, cursor = 0;
			double quotaDebt = 0;
		};
		BankStratum strata[STRATUM_AMOUNT];
		std::vector<float> bankRows;        // [optBankSize * 6], raw goal rows (CPU)
		std::vector<float> bankValues;      // [optBankSize], raw terminal utility values (CPU)
		std::vector<int64_t> bankInsertItr; // insert iteration per slot (age metric)
		torch::Tensor bankPsi;              // [2 * fill, reprDim] frozen psi embeddings (device)
		torch::Tensor bankValueLogits;      // [2 * fill] normalized V(g) logits (device), duplicated for x-twins
		int64_t bankPsiRows = 0;            // bank fill at the last re-embed
		float bankValueMean = 0.0f;
		float bankValueStd = 0.0f;
		float bankValueLogitStd = 0.0f;

		GCRLOptionality(
			const PPOLearnerConfig& config, int featureDim,
			QuasimetricCritic* liveGoalCritic, Model* liveSharedHead,
			torch::Device device
		);

		// targetCritic is new'd in the ctor and targetSharedHead is a MakeClone() — both are
		// owned by this object and alias nothing in the live ModelSet, so they must be freed
		// here. Without this, every PPOLearner teardown leaks the frozen scorer's full nets.
		~GCRLOptionality() {
			delete targetCritic;
			delete targetSharedHead;
		}

		// Once per training iteration: target <- (1-tau)*target + tau*live.
		void PolyakUpdate(QuasimetricCritic* liveGoalCritic, Model* liveSharedHead);

		// Hard copy live -> target. Used after a checkpoint load (the ctor copies the
		// fresh init; the loaded weights arrive later) — the targets are deliberately
		// not checkpointed, so a restart re-syncs them to the loaded live nets.
		void HardSyncFromLive(QuasimetricCritic* liveGoalCritic, Model* liveSharedHead);

		// Once per training iteration. candRows[s] are flat 6-dim candidate goal rows
		// for stratum s (pre-sampled by the Learner); inserts up to this iteration's
		// quota (+ carried debt) per stratum, then re-embeds the bank under frozen psi.
		void RefreshBank(
			const RLGC::FList candRows[STRATUM_AMOUNT],
			const RLGC::FList candValues[STRATUM_AMOUNT],
			int64_t iteration,
			Report& report
		);

		// [N] CPU phi_opt over the batch states; {} while the bank is empty.
		// Frozen nets, zero action components, minibatched on the training device.
		torch::Tensor ComputePhiOpt(torch::Tensor tStatesCpu, torch::Tensor* outReachOnly = nullptr);

		int BankFill() const;
		double BankAgeMean(int64_t iteration) const;

		// Math helpers shared by the production path and the self-tests.
		static torch::Tensor PhiOptFromEmbeddings(
			torch::Tensor phiS,
			torch::Tensor bankPsi,
			float quasiTau,
			float optTemp,
			torch::Tensor bankValueLogits = {}
		);
		// rOptRaw[t] = gamma*phiOpt[t+1] - phiOpt[t], EXACTLY 0 wherever terminals[t] != 0
		// (NORMAL and TRUNCATED both: a reset is not the policy losing options).
		static torch::Tensor MaskedPotentialDelta(torch::Tensor phiOpt, const int8_t* terminals, int64_t n, float gamma);

		// Synthetic-batch asserts (terminal masking, logsumexp normalization identity).
		// RG_ERR_CLOSEs on failure; called at PPOLearner construction when the feature is on.
		static void RunSelfTests();

		// Offline harness: scores hand-supplied states through the frozen scorer.
		// File format, one entry per line:  "g v1 .. v6" adds a goal-bank row,
		// "s v1 .. vObsDim" scores an obs row against the goals supplied so far.
		// Rows must already be in obs space (normalized if standardizeObs is on).
		void DebugScoreStates(const std::string& path, int obsSize);

	private:
		void ReembedBank();
	};
}
