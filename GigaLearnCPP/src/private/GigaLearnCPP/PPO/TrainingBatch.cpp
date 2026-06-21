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
		if (in.boostHerGoals.defined() && in.boostHerGoals.size(0) != expRows)
			RG_ERR_CLOSE("GCRL boost goal alignment failed: states=" << expRows <<
				", boostHerGoals=" << in.boostHerGoals.size(0));
	}

	// Value targets + the running-return stat come from the SPARSE rewards: the value critic learns the
	// true task value, not the shaped value V-Phi, so it never chases the nonstationary GCRL potential.
	GAE::Compute(
		in.rewards, in.terminals, in.valPreds, in.truncValPreds,
		result.advantages, result.targetValues, result.returns, result.rewClipPortion,
		in.gaeGamma, in.gaeLambda, in.returnStd, in.rewardClipRange
	);

	// The policy ADVANTAGE uses the shaped reward (sparse + potential shaping) via a second GAE pass
	// sharing the same sparse return std -- potential-based shaping affects the advantage only.
	if (in.shapingF.defined() && in.shapingF.size(0) == expRows) {
		torch::Tensor shapedAdv, shapedTargets, shapedReturns;
		float shapedClip;
		GAE::Compute(
			in.rewards + in.shapingF, in.terminals, in.valPreds, in.truncValPreds,
			shapedAdv, shapedTargets, shapedReturns, shapedClip,
			in.gaeGamma, in.gaeLambda, in.returnStd, in.rewardClipRange
		);
		result.advantages = shapedAdv; // targetValues + returns stay sparse
	}

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
		outBuffer.data.boostHerGoals = in.boostHerGoals;
		outBuffer.data.scoringGoals = in.scoringGoals;
		outBuffer.data.gcrlTrainMask = in.gcrlTrainMask;
		outBuffer.data.segmentIds = in.segmentIds;
		outBuffer.data.segmentSteps = in.segmentSteps;
	}

	return result;
}
