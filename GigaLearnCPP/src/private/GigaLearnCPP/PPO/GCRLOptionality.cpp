#include "GCRLOptionality.h"

#include <torch/nn/functional/normalization.h>

#include <cmath>
#include <fstream>
#include <sstream>

using namespace torch;

namespace {
	constexpr int GOAL_DIM = 6;

	// Copy params pairwise (same architecture by construction)
	void CopyParamsPairwise(std::vector<torch::Tensor>& dst, const std::vector<torch::Tensor>& src) {
		RG_ASSERT(dst.size() == src.size());
		for (size_t i = 0; i < dst.size(); i++)
			dst[i].copy_(src[i], false);
	}

	// target <- (1-tau)*target + tau*live
	void PolyakPairwise(std::vector<torch::Tensor>& dst, const std::vector<torch::Tensor>& src, float tau) {
		RG_ASSERT(dst.size() == src.size());
		for (size_t i = 0; i < dst.size(); i++)
			dst[i].mul_(1.0f - tau).add_(src[i], tau);
	}

	// Goal rows + their x-flipped twins (negate x-pos and x-vel), as one [2n, 6] tensor
	torch::Tensor RowsWithFlippedTwins(torch::Tensor rows) {
		torch::Tensor flipped = rows.clone();
		flipped.select(1, 0).neg_();
		flipped.select(1, 3).neg_();
		return torch::cat({ rows, flipped }, 0);
	}
}

GGL::GCRLOptionality::GCRLOptionality(
	const PPOLearnerConfig& _config, int featureDim,
	QuasimetricCritic* liveGoalCritic, Model* liveSharedHead,
	torch::Device _device) : config(_config), device(_device) {

	RG_ASSERT(liveGoalCritic);

	RG_NO_GRAD;

	// Fresh net with the live goal critic's exact architecture, then a hard param copy.
	// NOT built via Model::MakeEmptyClone()/MakeClone() — that path doesn't reconstruct
	// the phi/psi towers of a QuasimetricCritic.
	targetCritic = new QuasimetricCritic("opt_target",
		featureDim, 8, GOAL_DIM,
		config.gcrlCritic.layerSizes, config.gcrlReprDim,
		config.gcrlCritic.activationType, config.gcrlCritic.addLayerNorm,
		config.gcrlTau, config.gcrlVarReg, config.gcrlInfoNCEPenalty,
		config.gcrlCritic.optimType, device);
	auto dstParams = targetCritic->parameters();
	auto srcParams = liveGoalCritic->parameters();
	CopyParamsPairwise(dstParams, srcParams);
	for (torch::Tensor& p : dstParams)
		p.set_requires_grad(false);

	if (liveSharedHead) {
		targetSharedHead = liveSharedHead->MakeClone();
		auto headParams = targetSharedHead->parameters();
		for (torch::Tensor& p : headParams)
			p.set_requires_grad(false);
	}

	// Stratum slot partition (last stratum takes the rounding remainder)
	int size = RS_MAX(config.optBankSize, STRATUM_AMOUNT);
	int capOff = RS_MAX(1, (int)(config.optBankOffensiveFrac * size));
	int capDef = RS_MAX(1, (int)(config.optBankDefensiveFrac * size));
	int capRes = RS_MAX(1, size - capOff - capDef);
	strata[STRATUM_OFFENSIVE] = { 0, capOff };
	strata[STRATUM_DEFENSIVE] = { capOff, capDef };
	strata[STRATUM_RESOURCE] = { capOff + capDef, capRes };

	bankRows.assign((size_t)size * GOAL_DIM, 0.0f);
	bankValues.assign(size, 0.0f);
	bankInsertItr.assign(size, 0);
}

void GGL::GCRLOptionality::PolyakUpdate(QuasimetricCritic* liveGoalCritic, Model* liveSharedHead) {
	RG_NO_GRAD;
	float tau = config.optTargetTau;

	auto dstParams = targetCritic->parameters();
	auto srcParams = liveGoalCritic->parameters();
	PolyakPairwise(dstParams, srcParams, tau);

	if (targetSharedHead && liveSharedHead) {
		auto dstHead = targetSharedHead->parameters();
		auto srcHead = liveSharedHead->parameters();
		PolyakPairwise(dstHead, srcHead, tau);
	}
}

void GGL::GCRLOptionality::HardSyncFromLive(QuasimetricCritic* liveGoalCritic, Model* liveSharedHead) {
	RG_NO_GRAD;

	auto dstParams = targetCritic->parameters();
	auto srcParams = liveGoalCritic->parameters();
	CopyParamsPairwise(dstParams, srcParams);

	if (targetSharedHead && liveSharedHead) {
		auto dstHead = targetSharedHead->parameters();
		auto srcHead = liveSharedHead->parameters();
		CopyParamsPairwise(dstHead, srcHead);
	}
}

int GGL::GCRLOptionality::BankFill() const {
	int fill = 0;
	for (const BankStratum& s : strata)
		fill += s.count;
	return fill;
}

double GGL::GCRLOptionality::BankAgeMean(int64_t iteration) const {
	int64_t totalAge = 0;
	int fill = 0;
	for (const BankStratum& s : strata) {
		for (int i = 0; i < s.count; i++)
			totalAge += iteration - bankInsertItr[s.offset + i];
		fill += s.count;
	}
	return fill > 0 ? (double)totalAge / fill : 0.0;
}

void GGL::GCRLOptionality::RefreshBank(
	const RLGC::FList candRows[STRATUM_AMOUNT],
	const RLGC::FList candValues[STRATUM_AMOUNT],
	int64_t iteration,
	Report& report
) {
	static const char* STRATUM_NAMES[STRATUM_AMOUNT] = { "Offensive", "Defensive", "Resource" };

	int insertedTotal = 0;
	double insertedValueSum = 0.0;
	for (int s = 0; s < STRATUM_AMOUNT; s++) {
		BankStratum& stratum = strata[s];
		int numCands = (int)(candRows[s].size() / GOAL_DIM);

		// This iteration's quota plus carried debt; debt capped at the stratum capacity
		// so months of starvation can't build an instant full-stratum flush.
		double quota = (double)config.optBankRefreshFrac * stratum.cap + stratum.quotaDebt;
		int toInsert = RS_MIN((int)quota, numCands);
		stratum.quotaDebt = RS_MIN(quota - toInsert, (double)stratum.cap);

		for (int i = 0; i < toInsert; i++) {
			int slot = stratum.offset + stratum.cursor;
			memcpy(bankRows.data() + (size_t)slot * GOAL_DIM, candRows[s].data() + (size_t)i * GOAL_DIM, GOAL_DIM * sizeof(float));
			float value = i < (int)candValues[s].size() ? candValues[s][i] : 0.0f;
			if (!std::isfinite(value))
				value = 0.0f;
			bankValues[slot] = value;
			bankInsertItr[slot] = iteration;
			stratum.cursor = (stratum.cursor + 1) % stratum.cap;
			stratum.count = RS_MIN(stratum.count + 1, stratum.cap);
			insertedValueSum += value;
		}
		insertedTotal += toInsert;

		int fill = BankFill();
		report["Opt/Bank Mix " + std::string(STRATUM_NAMES[s])] = fill > 0 ? (double)stratum.count / fill : 0.0;
	}

	report["Opt/Bank Fill"] = BankFill();
	report["Opt/Bank Inserted"] = insertedTotal;
	report["Opt/Bank Inserted Value Mean"] = insertedTotal > 0 ? insertedValueSum / insertedTotal : 0.0;
	report["Opt/Bank Age Mean"] = BankAgeMean(iteration);

	// The frozen psi just moved (Polyak), so the whole bank re-embeds regardless of
	// whether anything was inserted — one [2*fill, 6] forward, negligible.
	ReembedBank();
	report["Opt/Bank Value Mean"] = bankValueMean;
	report["Opt/Bank Value Std"] = bankValueStd;
	report["Opt/Bank Value Logit Std"] = bankValueLogitStd;
	report["Opt/Value Weight"] = config.optValueWeight;
}

void GGL::GCRLOptionality::ReembedBank() {
	RG_NO_GRAD;

	int fill = BankFill();
	bankPsiRows = fill;
	if (fill == 0) {
		bankGoalRows = torch::Tensor();
		bankPsi = torch::Tensor();
		bankValueLogits = torch::Tensor();
		bankValueMean = bankValueStd = bankValueLogitStd = 0.0f;
		return;
	}

	// Gather filled rows (strata are contiguous slot ranges)
	torch::Tensor rows = torch::empty({ fill, GOAL_DIM });
	torch::Tensor values = torch::empty({ fill }, torch::TensorOptions().dtype(torch::kFloat32));
	int outIdx = 0;
	for (const BankStratum& s : strata) {
		if (s.count > 0) {
			rows.slice(0, outIdx, outIdx + s.count).copy_(
				torch::from_blob((void*)(bankRows.data() + (size_t)s.offset * GOAL_DIM),
					{ s.count, GOAL_DIM }, torch::TensorOptions().dtype(torch::kFloat32)));
			values.slice(0, outIdx, outIdx + s.count).copy_(
				torch::from_blob((void*)(bankValues.data() + (size_t)s.offset),
					{ s.count }, torch::TensorOptions().dtype(torch::kFloat32)));
			outIdx += s.count;
		}
	}

	torch::Tensor doubled = RowsWithFlippedTwins(rows).to(device, RG_H2D_NONBLOCKING(device), true);
	bankGoalRows = doubled;
	bankPsi = targetCritic->embed_psi(bankGoalRows);

	bankValueMean = values.mean().item<float>();
	bankValueStd = fill > 1 ? values.std(false).item<float>() : 0.0f;
	if (config.optValueWeight > 0.0f && bankValueStd > 1e-6f) {
		float clip = RS_MAX(0.0f, config.optValueClip);
		torch::Tensor norm = ((values - bankValueMean) / bankValueStd)
			.clamp(-clip, clip) * config.optValueWeight;
		bankValueLogits = torch::cat({ norm, norm }, 0).to(device, RG_H2D_NONBLOCKING(device), true);
		bankValueLogitStd = bankValueLogits.std(false).item<float>();
	} else {
		bankValueLogits = torch::Tensor();
		bankValueLogitStd = 0.0f;
	}
}

torch::Tensor GGL::GCRLOptionality::PhiOptFromEmbeddings(
	torch::Tensor phiS,
	torch::Tensor bankPsi,
	float quasiTau,
	float optTemp,
	torch::Tensor bankValueLogits
) {
	// Exact scoring-time distance composition: d = -cos(phi, psi) / tau over the
	// L2-normalized embeddings, soft-min'd over the (doubled) bank. Optional value
	// logits rerank real bank states; they never create off-manifold targets.
	torch::Tensor D = -torch::matmul(phiS, bankPsi.transpose(0, 1)) / quasiTau;
	torch::Tensor score = -D;
	if (bankValueLogits.defined() && bankValueLogits.numel() == bankPsi.size(0))
		score = score + bankValueLogits.to(score.device()).unsqueeze(0);
	return optTemp * (torch::logsumexp(score / optTemp, 1) - std::log((double)bankPsi.size(0)));
}

torch::Tensor GGL::GCRLOptionality::MaskedPotentialDelta(torch::Tensor phiOpt, const int8_t* terminals, int64_t n, float gamma) {
	torch::Tensor out = torch::zeros({ n }, torch::TensorOptions().dtype(torch::kFloat32));
	torch::Tensor phiCont = phiOpt.contiguous();
	const float* phi = phiCont.data_ptr<float>();
	float* dst = out.data_ptr<float>();
	for (int64_t t = 0; t < n; t++) {
		// t == n-1 is defensively treated as a boundary even though every combined
		// batch ends with a nonzero terminal.
		if (t + 1 >= n || terminals[t] != 0)
			continue;
		dst[t] = gamma * phi[t + 1] - phi[t];
	}
	return out;
}

torch::Tensor GGL::GCRLOptionality::ComputePhiOpt(torch::Tensor tStatesCpu, torch::Tensor* outReachOnly) {
	if (bankPsiRows == 0 || !bankPsi.defined())
		return {};

	RG_NO_GRAD;

	int64_t N = tStatesCpu.size(0);
	torch::Tensor out = torch::empty({ N }, torch::TensorOptions().dtype(torch::kFloat32));
	torch::Tensor reachOnly;
	if (outReachOnly)
		reachOnly = torch::empty({ N }, torch::TensorOptions().dtype(torch::kFloat32));
	int64_t step = device.is_cpu() ? N : (int64_t)config.miniBatchSize;
	lastRefineAcceptedFraction = 0.0f;
	lastRefineGoalDeltaNorm = 0.0f;
	lastRefineScoreGainMean = 0.0f;
	lastRefinePhiLiftMean = 0.0f;
	lastRefineStateFraction = 0.0f;
	double refineAccepted = 0.0, refineDelta = 0.0, refineGain = 0.0, refinePhiLift = 0.0;
	int64_t refineCandCount = 0, refineStateCount = 0;

	for (int64_t i = 0; i < N; i += step) {
		int64_t start = i, end = RS_MIN(i + step, N);
		torch::Tensor obs = tStatesCpu.slice(0, start, end).to(device, RG_H2D_NONBLOCKING(device), true);
		if (targetSharedHead)
			obs = targetSharedHead->Forward(obs, false);
		// Zero action components: the potential must be a function of state only
		torch::Tensor zeroActs = torch::zeros({ end - start, 8 }, torch::TensorOptions().dtype(torch::kFloat32).device(device));
		torch::Tensor phi = targetCritic->embed_phi(obs, zeroActs);
		torch::Tensor scoreReach = torch::matmul(phi, bankPsi.transpose(0, 1)) / config.gcrlTau;
		torch::Tensor score = scoreReach;
		if (bankValueLogits.defined() && bankValueLogits.numel() == bankPsi.size(0))
			score = score + bankValueLogits.to(score.device()).unsqueeze(0);
		int64_t bankCount = bankPsi.size(0);
		torch::Tensor phiBase = config.optTemp *
			(torch::logsumexp(score / config.optTemp, 1) - std::log((double)bankCount));
		torch::Tensor phiScored = phiBase;

		bool doRefine = config.optRefineGoals && bankGoalRows.defined() &&
			config.optRefineTopK > 0 && config.optRefineSteps > 0 &&
			config.optRefineStepSize > 0.0f && config.optRefineMaxDelta > 0.0f;
		if (doRefine) {
			int64_t B = score.size(0);
			int64_t M = score.size(1);
			int64_t RB = B;
			torch::Tensor refineIdx;
			if (config.optRefineMaxStates > 0 && B > config.optRefineMaxStates) {
				RB = config.optRefineMaxStates;
				refineIdx = (torch::arange(RB, torch::TensorOptions().dtype(torch::kLong).device(device)) * B / RB).to(torch::kLong);
			}
			torch::Tensor refineScore = refineIdx.defined() ? score.index_select(0, refineIdx) : score;
			torch::Tensor refinePhi = refineIdx.defined() ? phi.index_select(0, refineIdx) : phi;

			int64_t K = RS_MIN((int64_t)config.optRefineTopK, M);
			auto top = refineScore.topk(K, 1, true, true);
			torch::Tensor topVals = std::get<0>(top);
			torch::Tensor topIdx = std::get<1>(top);
			torch::Tensor flatIdx = topIdx.reshape({ -1 });
			torch::Tensor goals0 = bankGoalRows.index_select(0, flatIdx).detach();
			torch::Tensor goals = goals0.clone();
			torch::Tensor phiRep = refinePhi.unsqueeze(1).expand({ RB, K, refinePhi.size(1) }).reshape({ RB * K, refinePhi.size(1) }).detach();
			torch::Tensor valueFlat;
			if (bankValueLogits.defined() && bankValueLogits.numel() == M)
				valueFlat = bankValueLogits.index_select(0, flatIdx).detach();
			else
				valueFlat = torch::zeros({ RB * K }, torch::TensorOptions().dtype(torch::kFloat32).device(device));

			{
				at::AutoGradMode enableGrad(true);
				float maxDelta = RS_MAX(1e-6f, config.optRefineMaxDelta);
				float stepSize = RS_MAX(1e-6f, config.optRefineStepSize);
				float trustPenalty = RS_MAX(0.0f, config.optRefineTrustPenalty);
				for (int r = 0; r < config.optRefineSteps; r++) {
					goals = goals.detach();
					goals.set_requires_grad(true);
					torch::Tensor psi = targetCritic->embed_psi(goals);
					torch::Tensor delta = goals - goals0;
					torch::Tensor candidateScore =
						(phiRep * psi).sum(1) / config.gcrlTau + valueFlat -
						trustPenalty * delta.square().sum(1);
					torch::Tensor objective = candidateScore.mean();
					objective.backward();
					torch::Tensor grad = goals.grad();
					torch::Tensor gradNorm = grad.norm(2, 1, true).clamp_min(1e-6f);
					torch::Tensor next = goals + grad / gradNorm * stepSize;
					torch::Tensor nextDelta = next - goals0;
					torch::Tensor deltaNorm = nextDelta.norm(2, 1, true).clamp_min(1e-6f);
					torch::Tensor scale = (maxDelta / deltaNorm).clamp_max(1.0f);
					goals = (goals0 + nextDelta * scale).detach();
				}
			}

			torch::Tensor finalPsi = targetCritic->embed_psi(goals);
			torch::Tensor finalScore = (phiRep * finalPsi).sum(1) / config.gcrlTau + valueFlat;
			torch::Tensor origScore = topVals.reshape({ -1 });
			torch::Tensor accepted = finalScore > origScore;
			torch::Tensor refinedScore = torch::where(accepted, finalScore, origScore).reshape({ RB, K });

			torch::Tensor tempTopOrig = topVals / config.optTemp;
			torch::Tensor tempTopRefined = refinedScore / config.optTemp;
			torch::Tensor scoreTemp = refineScore / config.optTemp;
			torch::Tensor baseLSE = torch::logsumexp(scoreTemp, 1);
			torch::Tensor topOrigLSE = torch::logsumexp(tempTopOrig, 1);
			torch::Tensor topRefinedLSE = torch::logsumexp(tempTopRefined, 1);
			torch::Tensor remFrac = (topOrigLSE - baseLSE).exp().clamp_max(1.0f - 1e-7f);
			torch::Tensor remLSE = baseLSE + torch::log1p(-remFrac);
			torch::Tensor refinedLSE = torch::logaddexp(remLSE, topRefinedLSE);
			torch::Tensor phiRefined = config.optTemp * (refinedLSE - std::log((double)M));
			if (refineIdx.defined()) {
				phiScored = phiBase.clone();
				phiScored.index_copy_(0, refineIdx, phiRefined);
			} else {
				phiScored = phiRefined;
			}

			torch::Tensor gain = (refinedScore.reshape({ -1 }) - origScore).clamp_min(0.0f);
			torch::Tensor deltaNorm = (goals - goals0).norm(2, 1);
			refineAccepted += accepted.to(torch::kFloat32).sum().item<double>();
			refineDelta += deltaNorm.sum().item<double>();
			refineGain += gain.sum().item<double>();
			refinePhiLift += (phiRefined - (refineIdx.defined() ? phiBase.index_select(0, refineIdx) : phiBase)).sum().item<double>();
			refineCandCount += RB * K;
			refineStateCount += RB;
		}

		out.slice(0, start, end).copy_(phiScored.cpu(), true);
		if (outReachOnly) {
			torch::Tensor phiReach = config.optTemp *
				(torch::logsumexp(scoreReach / config.optTemp, 1) - std::log((double)bankCount));
			reachOnly.slice(0, start, end).copy_(phiReach.cpu(), true);
		}
	}

	if (refineCandCount > 0) {
		lastRefineAcceptedFraction = (float)(refineAccepted / refineCandCount);
		lastRefineGoalDeltaNorm = (float)(refineDelta / refineCandCount);
		lastRefineScoreGainMean = (float)(refineGain / refineCandCount);
		lastRefinePhiLiftMean = refineStateCount > 0 ? (float)(refinePhiLift / refineStateCount) : 0.0f;
		lastRefineStateFraction = (float)((double)refineStateCount / (double)RS_MAX((int64_t)1, N));
	}
	if (outReachOnly)
		*outReachOnly = reachOnly;
	return out;
}

torch::Tensor GGL::GCRLOptionality::ComputeOptionBreadth(
	torch::Tensor tStatesCpu,
	torch::Tensor* outPhiOpt,
	torch::Tensor* outReachOnly
) {
	if (bankPsiRows == 0 || !bankPsi.defined())
		return {};

	RG_NO_GRAD;

	int64_t N = tStatesCpu.size(0);
	torch::Tensor out = torch::empty({ N }, torch::TensorOptions().dtype(torch::kFloat32));
	torch::Tensor phiOpt;
	if (outPhiOpt)
		phiOpt = torch::empty({ N }, torch::TensorOptions().dtype(torch::kFloat32));
	torch::Tensor reachOnly;
	if (outReachOnly)
		reachOnly = torch::empty({ N }, torch::TensorOptions().dtype(torch::kFloat32));

	int64_t step = device.is_cpu() ? N : (int64_t)config.miniBatchSize;
	float breadthTemp = RS_MAX(1e-4f, config.optionEntropyBreadthTemp);

	for (int64_t i = 0; i < N; i += step) {
		int64_t start = i, end = RS_MIN(i + step, N);
		torch::Tensor obs = tStatesCpu.slice(0, start, end).to(device, RG_H2D_NONBLOCKING(device), true);
		if (targetSharedHead)
			obs = targetSharedHead->Forward(obs, false);

		torch::Tensor zeroActs = torch::zeros({ end - start, 8 }, torch::TensorOptions().dtype(torch::kFloat32).device(device));
		torch::Tensor phi = targetCritic->embed_phi(obs, zeroActs);
		torch::Tensor scoreReach = torch::matmul(phi, bankPsi.transpose(0, 1)) / config.gcrlTau;
		torch::Tensor score = scoreReach;
		if (bankValueLogits.defined() && bankValueLogits.numel() == bankPsi.size(0))
			score = score + bankValueLogits.to(score.device()).unsqueeze(0);

		int64_t bankCount = bankPsi.size(0);
		torch::Tensor probs = torch::softmax(score / breadthTemp, 1);
		torch::Tensor optionEntropy = -(probs * probs.clamp_min(1e-12f).log()).sum(1);
		torch::Tensor breadth;
		if (bankCount > 1) {
			torch::Tensor effectiveCount = optionEntropy.exp();
			breadth = ((effectiveCount - 1.0f) / (float)(bankCount - 1)).clamp(0.0f, 1.0f);
		} else {
			breadth = torch::zeros_like(optionEntropy);
		}
		out.slice(0, start, end).copy_(breadth.cpu(), true);

		if (outPhiOpt) {
			torch::Tensor phiPart = config.optTemp *
				(torch::logsumexp(score / config.optTemp, 1) - std::log((double)bankCount));
			phiOpt.slice(0, start, end).copy_(phiPart.cpu(), true);
		}
		if (outReachOnly) {
			torch::Tensor phiReach = config.optTemp *
				(torch::logsumexp(scoreReach / config.optTemp, 1) - std::log((double)bankCount));
			reachOnly.slice(0, start, end).copy_(phiReach.cpu(), true);
		}
	}

	if (outPhiOpt)
		*outPhiOpt = phiOpt;
	if (outReachOnly)
		*outReachOnly = reachOnly;
	return out;
}

void GGL::GCRLOptionality::RunSelfTests() {
	RG_NO_GRAD;
	constexpr const char* ERROR_PREFIX = "GCRLOptionality::RunSelfTests(): ";

	{ // Terminal masking: rOptRaw must be zero exactly at boundary transitions
		const float gamma = 0.9f;
		torch::Tensor phi = torch::tensor({ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f });
		int8_t terminals[6] = { 0, 0, 1, 0, 0, 1 };
		torch::Tensor delta = MaskedPotentialDelta(phi, terminals, 6, gamma);
		float expected[6] = {
			gamma * 2 - 1, gamma * 3 - 2, 0,
			gamma * 5 - 4, gamma * 6 - 5, 0
		};
		for (int t = 0; t < 6; t++) {
			float got = delta[t].item<float>();
			if (std::abs(got - expected[t]) > 1e-6f)
				RG_ERR_CLOSE(ERROR_PREFIX << "terminal masking failed at t=" << t << " (got " << got << ", expected " << expected[t] << ")");
		}
	}

	{ // Logsumexp normalization: with a bank of identical goals, phi_opt == -d(s, g)
	  // exactly (the -log|G| term cancels the count).
		const float quasiTau = 0.05f, optTemp = 1.0f;
		torch::Tensor phiS = torch::nn::functional::normalize(torch::randn({ 4, 8 }),
			torch::nn::functional::NormalizeFuncOptions().p(2).dim(-1));
		torch::Tensor g = torch::nn::functional::normalize(torch::randn({ 1, 8 }),
			torch::nn::functional::NormalizeFuncOptions().p(2).dim(-1));
		torch::Tensor bank = g.expand({ 16, 8 });
		torch::Tensor phiOpt = PhiOptFromEmbeddings(phiS, bank, quasiTau, optTemp);
		torch::Tensor expected = torch::matmul(phiS, g.transpose(0, 1)).flatten() / quasiTau; // == -d(s, g)
		float maxErr = (phiOpt - expected).abs().max().item<float>();
		if (maxErr > 1e-4f)
			RG_ERR_CLOSE(ERROR_PREFIX << "logsumexp identity failed (max err " << maxErr << ")");
	}

	{ // Value logits rerank real bank entries inside the same logsumexp.
		const float quasiTau = 0.05f, optTemp = 1.0f;
		torch::Tensor phiS = torch::nn::functional::normalize(torch::randn({ 3, 8 }),
			torch::nn::functional::NormalizeFuncOptions().p(2).dim(-1));
		torch::Tensor bank = torch::nn::functional::normalize(torch::randn({ 5, 8 }),
			torch::nn::functional::NormalizeFuncOptions().p(2).dim(-1));
		torch::Tensor vals = torch::tensor({ -1.0f, 0.5f, 0.0f, 1.25f, -0.25f });
		torch::Tensor scored = PhiOptFromEmbeddings(phiS, bank, quasiTau, optTemp, vals);
		torch::Tensor D = -torch::matmul(phiS, bank.transpose(0, 1)) / quasiTau;
		torch::Tensor expected = optTemp * (torch::logsumexp((-D + vals.unsqueeze(0)) / optTemp, 1) - std::log(5.0));
		float maxErr = (scored - expected).abs().max().item<float>();
		if (maxErr > 1e-4f)
			RG_ERR_CLOSE(ERROR_PREFIX << "value-logit composition failed (max err " << maxErr << ")");
	}

	RG_LOG("GCRLOptionality::RunSelfTests(): All passed");
}

void GGL::GCRLOptionality::DebugScoreStates(const std::string& path, int obsSize) {
	std::ifstream fIn(path);
	if (!fIn.good())
		RG_ERR_CLOSE("GCRLOptionality::DebugScoreStates(): Can't open file at " << path);

	RG_NO_GRAD;

	std::vector<float> goalRows;
	std::string line;
	int lineNum = 0;
	while (std::getline(fIn, line)) {
		lineNum++;
		std::istringstream ss(line);
		char kind;
		if (!(ss >> kind))
			continue;

		std::vector<float> vals;
		float v;
		while (ss >> v)
			vals.push_back(v);

		if (kind == 'g') {
			if (vals.size() != GOAL_DIM)
				RG_ERR_CLOSE("DebugScoreStates: line " << lineNum << ": goal needs " << GOAL_DIM << " values, got " << vals.size());
			goalRows += vals;
		} else if (kind == 's') {
			if ((int)vals.size() != obsSize)
				RG_ERR_CLOSE("DebugScoreStates: line " << lineNum << ": state needs " << obsSize << " values, got " << vals.size());
			if (goalRows.empty())
				RG_ERR_CLOSE("DebugScoreStates: line " << lineNum << ": no goals supplied before the first state");

			torch::Tensor rows = torch::from_blob(goalRows.data(),
				{ (int64_t)(goalRows.size() / GOAL_DIM), GOAL_DIM }, torch::TensorOptions().dtype(torch::kFloat32)).clone();
			torch::Tensor psi = targetCritic->embed_psi(RowsWithFlippedTwins(rows).to(device, RG_H2D_NONBLOCKING(device), true));

			torch::Tensor obs = torch::from_blob(vals.data(), { 1, obsSize }, torch::TensorOptions().dtype(torch::kFloat32))
				.clone().to(device, RG_H2D_NONBLOCKING(device), true);
			if (targetSharedHead)
				obs = targetSharedHead->Forward(obs, false);
			torch::Tensor zeroActs = torch::zeros({ 1, 8 }, torch::TensorOptions().dtype(torch::kFloat32).device(device));
			torch::Tensor phi = targetCritic->embed_phi(obs, zeroActs);
			float phiOpt = PhiOptFromEmbeddings(phi, psi, config.gcrlTau, config.optTemp).cpu().item<float>();
			RG_LOG("phi_opt[line " << lineNum << "] = " << phiOpt);
		}
	}
}
