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

		FrontierModule(int obsSize, const FrontierConfig& config, torch::Device device, ModelSet& outModels);

		// --- value map -------------------------------------------------------------------------
		// min(Va, Vb) over raw obs. Detached by construction: callers never backprop through it.
		torch::Tensor Value(torch::Tensor obs);
		// One bounded expectile backup. obs/nextObs are [n, obsSize]; reward/done are [n].
		// Returns the scalar loss (already stepped by the caller's optimizer group).
		struct ValueStats { float loss = 0, meanValue = 0, maxAbsTarget = 0; int clamped = 0; };
		ValueStats TrainValue(torch::Tensor obs, torch::Tensor nextObs, torch::Tensor reward, torch::Tensor done);

		// --- quasimetric -----------------------------------------------------------------------
		torch::Tensor Encode(torch::Tensor obs);
		// d(from -> to) for matched rows of already-encoded latents.
		torch::Tensor Distance(torch::Tensor fromLatent, torch::Tensor toLatent);
		// QRL objective: push distances apart subject to every OBSERVED one-step transition
		// costing at most 1. Triangle inequality is structural, so this converges toward
		// shortest-path decision counts rather than fitting whatever the behaviour produced.
		struct QuasiStats { float loss = 0, meanLocal = 0, violation = 0, meanSpread = 0; };
		QuasiStats TrainQuasi(torch::Tensor obs, torch::Tensor nextObs, torch::Tensor pairObs);

		// --- goals -----------------------------------------------------------------------------
		// For each row of obs, pick the candidate with the largest value gain whose distance falls
		// inside [bandLow, bandHigh]. Returns the chosen candidate rows and a validity mask.
		// Candidates outside the band are rejected because the toy measured that distances to
		// states outside forward reach carry no usable gradient at all.
		struct GoalPick { torch::Tensor goals; torch::Tensor valid; float meanGain = 0, meanDist = 0; };
		GoalPick SelectGoals(torch::Tensor obs, torch::Tensor candidates);

		// Progress of an executed prefix toward its goal, in [0, 1]: 1 - dBest / dStart.
		// Returns the per-row weight exp(sharpness * progress) and the index of the best step.
		struct Progress { torch::Tensor weight; torch::Tensor bestIndex; };
		Progress PrefixProgress(torch::Tensor prefixObs, torch::Tensor goal);
	};
}
