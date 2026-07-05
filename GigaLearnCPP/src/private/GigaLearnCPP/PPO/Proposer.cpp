#include "Proposer.h"

#include <RLGymCPP/TerminalConditions/TerminalCondition.h>
#include <torch/nn/modules/loss.h>

using namespace torch;

GGL::ProposerModule::ProposerModule(
	int trunkOutSize, const ProposerConfig& _config, torch::Device _device, ModelSet& outModels,
	const char* modelName)
	: config(_config), device(_device) {

	ModelConfig deltaConfig = config.delta;
	deltaConfig.numInputs = trunkOutSize + 6;
	deltaConfig.numOutputs = 6;
	deltaConfig.addOutputLayer = true;

	delta = new Model(modelName, deltaConfig, device);
	// The PPO minibatch loop's models.StepOptims() must never step this model - Train() steps
	// it independently, so its optimizer's momentum state can't be perturbed by PPO gradients
	// (which are never even computed for it, since it isn't part of the PPO/critic backward).
	delta->groupStepExempt = true;

	outModels.Add(delta);
	delta->SetOptimLR(config.lr);
}

torch::Tensor GGL::ProposerModule::ComputeFeatures(Model* sharedHead, torch::Tensor obs, int64_t chunkSize) {
	RG_NO_GRAD;

	int64_t n = obs.size(0);
	if (n == 0)
		return torch::zeros({ 0, 0 }, TensorOptions().dtype(kFloat32).device(device));

	if (chunkSize <= 0)
		chunkSize = n;

	std::vector<Tensor> chunks;
	for (int64_t start = 0; start < n; start += chunkSize) {
		int64_t stop = RS_MIN(start + chunkSize, n);
		Tensor obsChunk = obs.slice(0, start, stop).to(device, true);
		Tensor feat = sharedHead ? sharedHead->Forward(obsChunk, false) : obsChunk;
		chunks.push_back(feat.detach());
	}

	return torch::cat(chunks, 0);
}

void GGL::ProposerModule::SegmentEpisodes(
	const std::vector<int8_t>& terminals,
	std::vector<int64_t>& outEpStart, std::vector<int64_t>& outEpEnd) {

	outEpStart.clear();
	outEpEnd.clear();

	int64_t n = (int64_t)terminals.size();
	int64_t curStart = 0;
	for (int64_t i = 0; i < n; i++) {
		if (terminals[i]) {
			outEpStart.push_back(curStart);
			outEpEnd.push_back(i);
			curStart = i + 1;
		}
	}
}

GGL::ProposerModule::UnrollResult GGL::ProposerModule::Unroll(
	torch::Tensor features, torch::Tensor curBall,
	const std::vector<int64_t>& epStart, const std::vector<int64_t>& epEnd) {

	RG_NO_GRAD;

	int64_t n = features.size(0);
	UnrollResult result;

	// Run the whole recurrent unroll on the feature device and copy back to CPU exactly once at
	// the end. The previous version did a device->host .cpu() plus an .item() norm sync inside
	// the per-local-step loop, i.e. ~two GPU round-trips per timestep of the longest episode
	// (hundreds per iteration, times two heads). The goal arithmetic (prevGoal+delta, clamp) is
	// elementwise float32, so on-device results are bit-identical to the old on-CPU compute; the
	// drift-norm accumulator stays in float64, so its logged value is unchanged too.
	torch::Device dev = features.device();
	Tensor goalsDev = torch::zeros({ n, 6 }, TensorOptions().dtype(kFloat32).device(dev));
	Tensor prevGoalsDev = torch::zeros({ n, 6 }, TensorOptions().dtype(kFloat32).device(dev));

	int64_t numEp = (int64_t)epStart.size();
	if (numEp == 0) {
		result.goals = goalsDev.cpu();
		result.prevGoals = prevGoalsDev.cpu();
		result.meanDeltaNorm = 0.f;
		return result;
	}

	RG_ASSERT(curBall.size(0) == n);

	// g_{-1} for each episode = the current (pre-first-action) achieved ball - i.e. "propose
	// staying where we already are" as the neutral, zero-drift starting point
	Tensor curBallDev = curBall.to(kFloat32).to(dev);
	Tensor epStartIdx = torch::tensor(epStart, TensorOptions().dtype(kLong)).to(dev);
	Tensor curGoal = curBallDev.index_select(0, epStartIdx).clone(); // [numEp,6] device, owns storage

	std::vector<int64_t> epLen(numEp);
	int64_t maxLen = 0;
	for (int64_t e = 0; e < numEp; e++) {
		epLen[e] = epEnd[e] - epStart[e] + 1;
		maxLen = RS_MAX(maxLen, epLen[e]);
	}

	Tensor deltaNormSum = torch::zeros({}, TensorOptions().dtype(kFloat64).device(dev));
	int64_t deltaNormCount = 0;
	float clampVal = RS_MAX(1e-3f, config.goalClamp);

	// Ragged-batched by local step L: one batched Delta forward per L, over whichever episodes
	// are still alive at that L (shrinks as shorter episodes finish) - cost is one forward per
	// timestep of the LONGEST episode this iteration, not one recurrent forward per row.
	for (int64_t L = 0; L < maxLen; L++) {
		std::vector<int64_t> aliveEp, aliveRow;
		aliveEp.reserve(numEp);
		aliveRow.reserve(numEp);
		for (int64_t e = 0; e < numEp; e++) {
			if (L < epLen[e]) {
				aliveEp.push_back(e);
				aliveRow.push_back(epStart[e] + L);
			}
		}
		if (aliveEp.empty())
			continue;

		Tensor epIdx = torch::tensor(aliveEp, TensorOptions().dtype(kLong)).to(dev);
		Tensor rowIdx = torch::tensor(aliveRow, TensorOptions().dtype(kLong)).to(dev);

		Tensor prevGoalAlive = curGoal.index_select(0, epIdx); // [m,6] device
		prevGoalsDev.index_copy_(0, rowIdx, prevGoalAlive);

		Tensor featAlive = features.index_select(0, rowIdx); // [m,trunk] device
		Tensor deltaIn = torch::cat({ featAlive, prevGoalAlive }, -1);
		Tensor deltaOut = delta->Forward(deltaIn, false).to(kFloat32); // [m,6] device

		deltaNormSum += deltaOut.norm(2, -1).sum().to(kFloat64);
		deltaNormCount += (int64_t)aliveEp.size();

		Tensor newGoal = (prevGoalAlive + deltaOut).clamp(-clampVal, clampVal);
		goalsDev.index_copy_(0, rowIdx, newGoal);
		curGoal.index_copy_(0, epIdx, newGoal);
	}

	result.goals = goalsDev.cpu();
	result.prevGoals = prevGoalsDev.cpu();
	result.meanDeltaNorm = deltaNormCount > 0 ? (float)(deltaNormSum.item<double>() / deltaNormCount) : 0.f;
	return result;
}

torch::Tensor GGL::ProposerModule::ComputeNStepAdvantages(
	torch::Tensor rews, const std::vector<int8_t>& terminals,
	torch::Tensor valPreds, torch::Tensor truncValPreds,
	const std::vector<int64_t>& epStart, const std::vector<int64_t>& epEnd,
	int horizonSteps, float gamma, float returnStd, float clipRange) {

	int64_t n = rews.size(0);
	RG_ASSERT(valPreds.size(0) == n);
	RG_ASSERT((int64_t)terminals.size() == n);

	rews = rews.contiguous();
	valPreds = valPreds.contiguous();
	auto _rews = rews.const_data_ptr<float>();
	auto _valPreds = valPreds.const_data_ptr<float>();

	bool hasTrunc = truncValPreds.defined();
	const float* _truncValPreds = NULL;
	int64_t numTruncs = 0;
	if (hasTrunc) {
		truncValPreds = truncValPreds.contiguous();
		_truncValPreds = truncValPreds.const_data_ptr<float>();
		numTruncs = truncValPreds.size(0);
	}

	// Same standardize+clip transform GAE applies to rewards before differencing against V,
	// so A^(N) is expressed in the same units the critic was actually trained against
	std::vector<float> rHat(n);
	for (int64_t i = 0; i < n; i++) {
		float r = _rews[i];
		if (returnStd != 0) {
			r = r / returnStd;
			if (clipRange > 0)
				r = RS_CLAMP(r, -clipRange, clipRange);
		}
		rHat[i] = r;
	}

	int N = RS_MAX(1, horizonSteps);
	std::vector<float> outAN(n, 0.f);

	int64_t truncCursor = 0; // consumed front-to-back = chronological order, matching storage order
	int64_t numEp = (int64_t)epStart.size();
	for (int64_t e = 0; e < numEp; e++) {
		int64_t s = epStart[e], en = epEnd[e];
		bool endIsTrunc = (terminals[en] == RLGC::TerminalType::TRUNCATED);

		float truncVal = 0;
		if (endIsTrunc) {
			if (!hasTrunc || truncCursor >= numTruncs)
				RG_ERR_CLOSE(
					"ProposerModule::ComputeNStepAdvantages: truncated episode but not enough "
					"truncated value predictions (have " << numTruncs << ")");
			truncVal = _truncValPreds[truncCursor];
			truncCursor++;
		}

		for (int64_t t = s; t <= en; t++) {
			int64_t K = RS_MIN((int64_t)N, en - t + 1);

			double acc = 0, discount = 1;
			for (int64_t k = 0; k < K; k++) {
				acc += discount * rHat[t + k];
				discount *= gamma;
			}

			// discount == gamma^N here (K==N, window stayed inside the episode) if t+N<=en,
			// else discount == gamma^K (window ran off the end at en) - either way it's already
			// the correct multiplier for whichever bootstrap branch we take below
			double bootstrap = (t + N <= en) ?
				(discount * (double)_valPreds[t + N]) :
				(discount * (endIsTrunc ? (double)truncVal : 0.0));

			outAN[t] = (float)(acc + bootstrap - (double)_valPreds[t]);
		}
	}

	if (hasTrunc && truncCursor != numTruncs)
		RG_ERR_CLOSE(
			"ProposerModule::ComputeNStepAdvantages: consumed " << truncCursor << "/" << numTruncs
			<< " truncated value predictions (episode segmentation doesn't match GAE's)");

	return torch::tensor(outAN);
}

GGL::ProposerModule::TrainResult GGL::ProposerModule::Train(
	torch::Tensor features, torch::Tensor prevGoals, torch::Tensor targets, torch::Tensor weights) {

	TrainResult result = {};

	int64_t n = features.size(0);
	if (n == 0)
		return result;

	RG_ASSERT(prevGoals.size(0) == n && targets.size(0) == n && weights.size(0) == n);

	Tensor prevGoalsDev = prevGoals.to(kFloat32).to(device, true);
	Tensor targetsDev = targets.to(kFloat32).to(device, true);
	Tensor weightsDev = weights.to(kFloat32).to(device, true);

	auto smoothL1 = torch::nn::SmoothL1Loss(torch::nn::SmoothL1LossOptions().reduction(torch::kNone));

	double lossSum = 0;
	int64_t lossCount = 0;

	int epochs = RS_MAX(1, config.trainEpochs);
	int64_t mbSize = RS_MAX((int64_t)1, (int64_t)config.trainMinibatchSize);

	for (int epoch = 0; epoch < epochs; epoch++) {
		// Matches the reachability aux loss's own subsampling idiom: unseeded torch::randperm
		Tensor order = torch::randperm(n, TensorOptions().dtype(kLong));

		for (int64_t start = 0; start < n; start += mbSize) {
			int64_t stop = RS_MIN(start + mbSize, n);
			Tensor idx = order.slice(0, start, stop).to(device);

			Tensor featBatch = features.index_select(0, idx);
			Tensor prevGoalBatch = prevGoalsDev.index_select(0, idx);
			Tensor targetBatch = targetsDev.index_select(0, idx);
			Tensor weightBatch = weightsDev.index_select(0, idx);

			Tensor deltaIn = torch::cat({ featBatch, prevGoalBatch }, -1);
			Tensor pred = prevGoalBatch + delta->Forward(deltaIn, false);

			Tensor perRow = smoothL1(pred, targetBatch).sum(-1); // [m]
			Tensor loss = (perRow * weightBatch).mean();

			loss.backward();
			delta->StepOptim();

			int64_t m = stop - start;
			lossSum += loss.detach().cpu().item<double>() * m;
			lossCount += m;
		}
	}

	result.loss = lossCount > 0 ? (float)(lossSum / lossCount) : 0.f;
	return result;
}
