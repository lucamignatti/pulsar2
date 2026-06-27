#include "TrainingBatch.h"
#include "GAE.h"

GGL::TrainingBatchResult GGL::BuildTrainingBatch(const TrainingBatchInputs& in, ExperienceBuffer& outBuffer) {
	RG_NO_GRAD;

	TrainingBatchResult result;

	const int64_t expRows = in.states.size(0);

	// Goal tensors must line up row-for-row with the states, or advantages and goals desync.
	if (in.gcrlEnabled) {
		if (in.achievedGoals.size(0) != expRows || in.herGoals.size(0) != expRows || in.scoringGoals.size(0) != expRows
			|| in.gcrlTrainMask.size(0) != expRows || in.segmentIds.size(0) != expRows)
			RG_ERR_CLOSE("GCRL tensor alignment failed: states=" << expRows <<
				", achievedGoals=" << in.achievedGoals.size(0) <<
				", herGoals=" << in.herGoals.size(0) <<
				", scoringGoals=" << in.scoringGoals.size(0) <<
				", gcrlTrainMask=" << in.gcrlTrainMask.size(0) <<
				", segmentIds=" << in.segmentIds.size(0));

		// Car-critic goals are optional (only when useCarCritic + obs builder supports it);
		// validate alignment only when present.
		if (in.carHerGoals.defined() && in.carHerGoals.size(0) != expRows)
			RG_ERR_CLOSE("GCRL car goal alignment failed: states=" << expRows <<
				", carHerGoals=" << in.carHerGoals.size(0));
	}

	// Advantages via GAE. Reads the pre-update return std the caller supplied.
	GAE::Compute(
		in.rewards, in.terminals, in.valPreds, in.truncValPreds,
		result.advantages, result.targetValues, result.returns, result.rewClipPortion,
		in.gaeGamma, in.gaeLambda, in.returnStd, in.rewardClipRange
	);

	result.avgReturn = result.returns.abs().mean().item<float>();
	result.avgAdvantage = result.advantages.abs().mean().item<float>();
	result.avgValTarget = result.targetValues.abs().mean().item<float>();

	// Pack the experience buffer for the PPO update.
	outBuffer.data.actions = in.actions;
	outBuffer.data.logProbs = in.logProbs;
	outBuffer.data.actionMasks = in.actionMasks;
	outBuffer.data.states = in.states;
	outBuffer.data.advantages = result.advantages;
	outBuffer.data.targetValues = result.targetValues;
	if (in.gcrlEnabled) {
		outBuffer.data.achievedGoals = in.achievedGoals;
		outBuffer.data.herGoals = in.herGoals;
		outBuffer.data.carHerGoals = in.carHerGoals;
		outBuffer.data.scoringGoals = in.scoringGoals;
		outBuffer.data.gcrlTrainMask = in.gcrlTrainMask;
		outBuffer.data.gcrlScoringMask = in.gcrlScoringMask;
		outBuffer.data.segmentIds = in.segmentIds;
		outBuffer.data.segmentSteps = in.segmentSteps;
	}

	return result;
}
