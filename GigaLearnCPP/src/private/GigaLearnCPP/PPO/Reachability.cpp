#include "Reachability.h"

#include <torch/nn/modules/loss.h>

using namespace torch;

static Tensor L2Normalize(Tensor t) {
	return t / t.norm(2, -1, true).clamp_min(1e-6f);
}

GGL::ReachabilityModule::ReachabilityModule(
	int trunkOutSize, int numActions, const ReachabilityConfig& _config, torch::Device _device, ModelSet& outModels)
	: config(_config), device(_device), numActions(numActions) {

	ModelConfig phiConfig = config.phi;
	phiConfig.numInputs = trunkOutSize + numActions;
	phiConfig.numOutputs = config.representationSize;
	phiConfig.addOutputLayer = true;

	ModelConfig psiConfig = config.psi;
	psiConfig.numInputs = 6;
	psiConfig.numOutputs = config.representationSize;
	psiConfig.addOutputLayer = true;

	phi = new Model("reach_phi", phiConfig, device);
	psiCar = new Model("reach_psi_car", psiConfig, device);
	psiBall = new Model("reach_psi_ball", psiConfig, device);

	outModels.Add(phi);
	outModels.Add(psiCar);
	outModels.Add(psiBall);

	phi->SetOptimLR(config.lr);
	psiCar->SetOptimLR(config.lr);
	psiBall->SetOptimLR(config.lr);
}

torch::Tensor GGL::ReachabilityModule::EncodeStateAction(torch::Tensor trunkOut, torch::Tensor actions) {
	// one_hot (unlike scatter_) bounds-checks the action indices
	Tensor oneHot = torch::one_hot(actions.to(kLong), numActions).to(kFloat32);
	return L2Normalize(phi->Forward(torch::cat({ trunkOut, oneHot }, -1), false));
}

GGL::ReachabilityModule::InfoNCEResult GGL::ReachabilityModule::ComputeInfoNCELoss(
	Model* psiHead, torch::Tensor sa, torch::Tensor goals) {

	InfoNCEResult result = {};

	int64_t b = sa.size(0);
	if (b < 2 || goals.size(0) != b)
		return result;

	Tensor g = L2Normalize(psiHead->Forward(goals.to(kFloat32), false));

	Tensor logits = torch::matmul(sa, g.transpose(0, 1)) / config.tau;
	Tensor labels = torch::arange(b, TensorOptions().dtype(kLong).device(sa.device()));

	Tensor rowLoss = torch::nn::CrossEntropyLoss()(logits, labels);
	Tensor columnLoss = torch::nn::CrossEntropyLoss()(logits.transpose(0, 1), labels);

	Tensor logsumexpRows = torch::logsumexp(logits, 1);
	Tensor logsumexpColumns = torch::logsumexp(logits, 0);
	Tensor logsumexpPenalty = config.logsumexpPenaltyCoeff * (logsumexpRows.pow(2).mean() + logsumexpColumns.pow(2).mean());

	// Keep the goal embeddings from collapsing to a point (which would make the logits degenerate)
	float stdNorm = sqrtf((float)config.representationSize);
	Tensor goalVarPenalty = config.varReg * (1.0f / (g.std(0, false).mean() * stdNorm + 1e-4f));

	result.loss = rowLoss + columnLoss + logsumexpPenalty + goalVarPenalty;

	{
		RG_NO_GRAD;
		Tensor rowCorrect = logits.argmax(1).eq(labels).to(kFloat);
		Tensor columnCorrect = logits.argmax(0).eq(labels).to(kFloat);
		result.categoricalAccuracy = (0.5f * (rowCorrect.mean() + columnCorrect.mean())).cpu().item<float>();
		result.rawLoss = (rowLoss + columnLoss).detach().cpu().item<float>();
	}

	return result;
}

torch::Tensor GGL::ReachabilityModule::StateActionVarPenalty(torch::Tensor sa) {
	float stdNorm = sqrtf((float)config.representationSize);
	return config.varReg * (1.0f / (sa.std(0, false).mean() * stdNorm + 1e-4f));
}

std::vector<torch::Tensor> GGL::ReachabilityModule::EvalRho(
	Model* sharedHead, const std::vector<GoalQuery>& queries, torch::Tensor obs, torch::Tensor actionMasks) {

	RG_NO_GRAD;

	int64_t n = obs.size(0);
	int64_t k = RS_MAX(1, config.numActionSamples);
	int64_t chunkSize = (config.scoreChunkSize > 0) ? config.scoreChunkSize : n;
	float tau = RS_MAX(1e-6f, config.tau);

	// Goals are fixed for the whole call, encode each psi once
	std::vector<Tensor> goalEmbeds;
	for (auto& query : queries)
		goalEmbeds.push_back(L2Normalize(
			query.psiHead->Forward(query.goal6.to(kFloat32).to(device).view({ 1, 6 }), false))); // [1, repr]

	std::vector<Tensor> rhos;
	for (size_t q = 0; q < queries.size(); q++)
		rhos.push_back(torch::zeros({ n }, TensorOptions().dtype(kFloat32)));

	for (int64_t start = 0; start < n; start += chunkSize) {
		int64_t stop = RS_MIN(start + chunkSize, n);
		int64_t m = stop - start;

		Tensor obsChunk = obs.slice(0, start, stop).to(device, true);
		Tensor trunkOut = sharedHead ? sharedHead->Forward(obsChunk, false) : obsChunk;

		// Uniform over valid actions; the tiny epsilon guards a (never-expected) all-zero mask row
		Tensor maskChunk = actionMasks.slice(0, start, stop).to(device).to(kFloat32) + 1e-6f;
		Tensor sampled = torch::multinomial(maskChunk, k, true); // [m, k]

		// One batched phi forward over all (row, sample) pairs instead of k sequential ones
		Tensor trunkRep = trunkOut.repeat_interleave(k, 0);              // [m*k, trunkOut]
		Tensor sa = EncodeStateAction(trunkRep, sampled.reshape({ -1 })); // [m*k, repr]

		for (size_t q = 0; q < queries.size(); q++) {
			Tensor scores = torch::matmul(sa, goalEmbeds[q].transpose(0, 1)).view({ m, k }) / tau;
			rhos[q].slice(0, start, stop).copy_(scores.mean(1).cpu());
		}
	}

	return rhos;
}
