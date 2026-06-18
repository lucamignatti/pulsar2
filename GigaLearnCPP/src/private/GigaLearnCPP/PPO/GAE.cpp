#include "GAE.h"

void GGL::GAE::Compute(
	torch::Tensor rews, torch::Tensor terminals, torch::Tensor valPreds, torch::Tensor truncValPreds,
	torch::Tensor& outAdvantages, torch::Tensor& outTargetValues, torch::Tensor& outReturns, float& outRewClipPortion,
	float gamma, float lambda, float returnStd, float clipRange
) {

	bool hasTruncValPreds = truncValPreds.defined();

	float prevLambda = 0;
	int numReturns = rews.size(0);
	outAdvantages = torch::zeros(numReturns);
	outReturns = torch::zeros(numReturns);
	float prevRet = 0;
	int truncCount = 0;

	float totalRew = 0, totalClippedRew = 0;

	// Make sure all tensors are contiguous first
	rews = rews.contiguous();
	terminals = terminals.contiguous();
	valPreds = valPreds.contiguous();
	if (hasTruncValPreds)
		truncValPreds = truncValPreds.contiguous();

	// Accessing the raw pointers makes this all like 10x faster
	auto _terminals = terminals.const_data_ptr<int8_t>();
	auto _rews = rews.const_data_ptr<float>();
	auto _valPreds = valPreds.const_data_ptr<float>();

	const float* _truncValPreds;
	int numTruncs;
	if (hasTruncValPreds) {
		_truncValPreds = truncValPreds.const_data_ptr<float>();
		numTruncs = truncValPreds.size(0);
	} else {
		_truncValPreds = NULL;
		numTruncs = 0;
	}

	auto _outReturns = std::vector<float>(numReturns, 0);
	auto _outAdvantages = std::vector<float>(numReturns, 0);

	for (int step = numReturns - 1; step >= 0; step--) {
		uint8_t terminal = _terminals[step];
		float done = terminal == RLGC::TerminalType::NORMAL;
		float trunc = terminal == RLGC::TerminalType::TRUNCATED;

		float curReward;
		if (returnStd != 0) {
			curReward = _rews[step] / returnStd;

			totalRew += abs(curReward);

			// We only clip if returns are standardized
			if (clipRange > 0)
				curReward = RS_CLAMP(curReward, -clipRange, clipRange);

			totalClippedRew += abs(curReward);
		} else {
			curReward = _rews[step];
			totalRew += abs(curReward);
		}

		float nextValPred;
		if (terminal == RLGC::TerminalType::TRUNCATED) {
			// We've encountered a truncation
			// Pull the next truncated value

			if (!hasTruncValPreds)
				RG_ERR_CLOSE("GAE encountered a truncated terminal, but has no truncated val pred");

			if (truncCount >= numTruncs)
				RG_ERR_CLOSE("GAE encountered too many truncated terminals, not enough val preds (max: " << numTruncs << ")")

			nextValPred = _truncValPreds[truncCount];
			truncCount++;
		} else {
			nextValPred = _valPreds[step + 1];
		}

		float predReturn = curReward + gamma * nextValPred * (1 - done);
		float delta = predReturn - _valPreds[step];
		float curReturn = _rews[step] + prevRet * gamma * (1 - done) * (1 - trunc);
		_outReturns[step] = curReturn;
		
		prevLambda = delta + gamma * lambda * (1 - done) * (1 - trunc) * prevLambda;
		_outAdvantages[step] = prevLambda;

		prevRet = curReturn;
	}
	
	if (hasTruncValPreds)
		if (truncCount != truncValPreds.size(0))
			RG_ERR_CLOSE("GAE didn't receive expected truncation count (only " << truncCount << "/" << truncValPreds.size(0) << ")");

	outReturns = torch::tensor(_outReturns);
	outAdvantages = torch::tensor(_outAdvantages);
	outTargetValues = valPreds.slice(0, 0, numReturns) + outAdvantages;
	outRewClipPortion = (totalRew - totalClippedRew) / RS_MAX(totalRew, 1e-7f);
}