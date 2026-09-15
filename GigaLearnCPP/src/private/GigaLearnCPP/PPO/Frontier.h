#pragma once
#include "../Util/Models.h"

namespace GGL {

	// Port of the AirLine frontier mechanism (research/reports/WORLD_MODEL_W163_VALUE_MAP.md).
	//
	//   value map   V(s) <- expectile_tau over [ r + gamma (1-done) min(Va,Vb)(s') ], BOUNDED
	//   quasimetric d(x->y) = max_i relu(h_i(y)-h_i(x)) + ||g(y)-g(x)||   (asymmetric, triangle-safe)
	//   goals       the reachable candidate with the largest value gain over the current state
	//   sil         executed prefixes weighted by exp(sharpness * progress toward the goal)
	//
	// Instrument only: nothing here enters advantages, returns or the critic, and the trunk is
	// read DETACHED. All labels are real executed actions.
	class FrontierModule {
	public:
		FrontierConfig config;
		torch::Device device;
		int obsSize;

		// Owned by the PPOLearner's ModelSet (saved / loaded / stepped with the rest)
		Model* valueA;   // twin value heads; the target uses min(valueA, valueB) as the anti-ratchet
		Model* valueB;
		Model* quasi;    // encoder -> [latentAsym + latentSym]
		// Lagged copy of valueA, refreshed every progressLagIters. NOT in the ModelSet: it is a
		// measurement device, and a resume that starts it equal to valueA simply reads zero
		// learning progress for one lag period rather than carrying stale state across runs.
		Model* valueSlow = nullptr;
		int sinceLagRefresh = 0;

		// Width of the opponent context appended to every input (0 = unconditioned).
		int ctxDim = 0;
		// Concatenate ctx onto obs. ctx is [ctxDim] (one opponent per iteration, broadcast) or
		// [n, ctxDim]; undefined ctx with ctxDim > 0 is a programming error, not a silent zero.
		torch::Tensor WithCtx(torch::Tensor obs, torch::Tensor ctx);
		void RefreshLag();

		// ===== VISITATION =====
		// Fixed random projection of the latent -> visitBits sign bits -> bucket index, with
		// decaying counts. This is how "we think it is good but have not proven it" is measured:
		// good by V, unproven by how rarely the policy is anywhere near it.
		torch::Tensor visitProj;    // [latent, bits], fixed at construction
		torch::Tensor visitCount;   // [2^bits]
		torch::Tensor VisitKey(torch::Tensor latent);       // [n] int64 bucket ids
		void VisitObserve(torch::Tensor latent);            // accumulate, then decay
		torch::Tensor VisitOf(torch::Tensor latent);        // [n] counts

		// Twin disagreement = epistemic uncertainty = "the map does not know this".
		torch::Tensor Disagreement(torch::Tensor obs, torch::Tensor ctx);

		FrontierModule(int obsSize, const FrontierConfig& config, torch::Device device, ModelSet& outModels);

		// --- value map -------------------------------------------------------------------------
		// min(Va, Vb) over raw obs. Detached by construction: callers never backprop through it.
		torch::Tensor Value(torch::Tensor obs, torch::Tensor ctx);
		// |V - V_lagged|: how much the map has moved here since the last refresh.
		torch::Tensor LearningProgress(torch::Tensor obs, torch::Tensor ctx);
		// One bounded expectile backup. obs/nextObs are [n, obsSize]; reward/done are [n].
		// Returns the scalar loss (already stepped by the caller's optimizer group).
		struct ValueStats { float loss = 0, meanValue = 0, maxAbsTarget = 0; int clamped = 0; };
		ValueStats TrainValue(torch::Tensor obs, torch::Tensor nextObs, torch::Tensor reward, torch::Tensor done, torch::Tensor ctx);

		// --- quasimetric -----------------------------------------------------------------------
		torch::Tensor Encode(torch::Tensor obs, torch::Tensor ctx);
		// d(from -> to) for matched rows of already-encoded latents.
		torch::Tensor Distance(torch::Tensor fromLatent, torch::Tensor toLatent);
		// QRL objective: push distances apart subject to every OBSERVED one-step transition
		// costing at most 1. Triangle inequality is structural, so this converges toward
		// shortest-path decision counts rather than fitting whatever the behaviour produced.
		// done marks rows whose nextObs is the next episode's kickoff, not a successor: those
		// pairs are excluded from the one-step constraint (a goal is not one decision from kickoff).
		struct QuasiStats { float loss = 0, meanLocal = 0, violation = 0, meanSpread = 0; };
		QuasiStats TrainQuasi(torch::Tensor obs, torch::Tensor nextObs, torch::Tensor pairObs, torch::Tensor done, torch::Tensor ctx);

		// --- goals -----------------------------------------------------------------------------
		// For each row of obs, pick the candidate with the largest value gain whose distance falls
		// inside [bandLow, bandHigh]. Returns the chosen candidate rows and a validity mask.
		// Candidates outside the band are rejected because the toy measured that distances to
		// states outside forward reach carry no usable gradient at all.
		struct GoalPick { torch::Tensor goals; torch::Tensor valid; float meanGain = 0, meanDist = 0, meanRarity = 0, meanProgress = 0; };
		// bandLo/bandHi are in METRIC UNITS; the caller converts from decisions via LocalD().
		// mode is FrontierConfig::GoalSelect.
		GoalPick SelectGoals(torch::Tensor obs, torch::Tensor candidates, float bandLo, float bandHi, int mode, torch::Tensor ctx);

		// Live one-step distance, EMA'd over TrainQuasi calls. This is the metric's SCALE: one
		// executed decision costs this much, so a band in decisions becomes a band in units by
		// multiplying. Seeded at 1 (the constraint's own target) so the first iteration is sane.
		float LocalD() const { return localDEma > 1e-3f ? localDEma : 1.0f; }
		float localDEma = 1.0f;

		// Progress of an executed prefix toward its goal, in [0, 1]: 1 - dBest / dStart.
		// Returns the per-row weight exp(sharpness * progress) and the index of the best step.
		// live is [n, T] bool: false for padded steps past the window's episode boundary, which
		// must not be allowed to win the closest-approach argmin.
		struct Progress { torch::Tensor weight; torch::Tensor bestIndex; };
		Progress PrefixProgress(torch::Tensor prefixObs, torch::Tensor goal, torch::Tensor live, torch::Tensor ctx);
	};
}
