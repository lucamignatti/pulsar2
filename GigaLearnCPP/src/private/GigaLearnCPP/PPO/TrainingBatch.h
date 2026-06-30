#pragma once
#include "ExperienceBuffer.h"

namespace GGL {

	// Pure inputs for building one iteration's training batch.
	//
	// The caller has already converted the collected rollout to tensors and run the critic to
	// get value predictions ("critic out": the model-dependent step stays in the caller, the
	// predictions arrive here as plain data). This keeps BuildTrainingBatch a pure function of
	// its inputs, testable with hand-made tensors and no model/env/sockets.
	struct TrainingBatchInputs {
		// Rollout tensors, one row per collected timestep
		torch::Tensor states;       // [N, obsSize]
		torch::Tensor actions;      // [N]
		torch::Tensor logProbs;     // [N]
		torch::Tensor actionMasks;  // [N, numActions]
		torch::Tensor rewards;      // [N]
		torch::Tensor terminals;    // [N] (int8 RLGC::TerminalType)

		// Critic value predictions for the rollout, computed by the caller.
		torch::Tensor valPreds;       // [N]
		torch::Tensor truncValPreds;  // [numTruncations] (may be undefined)

		// Goal-conditioning (GCRL) tensors. Only consulted when gcrlEnabled.
		bool gcrlEnabled = false;
		torch::Tensor achievedGoals, herGoals, carHerGoals, approachHerGoals, scoringGoals, gcrlTrainMask, gcrlScoringMask, segmentIds, segmentSteps;

		// GAE config + the *current* (pre-update) return std. The caller owns the running return
		// statistic; this unit only reads the std and hands back the new returns to update it.
		float gaeGamma = 0.99f;
		float gaeLambda = 0.95f;
		float rewardClipRange = 10.0f;
		float returnStd = 1.0f;
	};

	struct TrainingBatchResult {
		torch::Tensor advantages, targetValues, returns;
		float rewClipPortion = 0;
		float avgReturn = 0, avgAdvantage = 0, avgValTarget = 0;
	};

	// Computes advantages via GAE, validates goal-tensor alignment, and packs `outBuffer` for the
	// PPO update. Pure: it does not run the critic and does not mutate any running statistic.
	// The caller updates its running return stat from `result.returns`.
	TrainingBatchResult BuildTrainingBatch(const TrainingBatchInputs& in, ExperienceBuffer& outBuffer);
}
