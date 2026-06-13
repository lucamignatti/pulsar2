#include "GCRLOptionality.h"

#include <torch/nn/functional/normalization.h>

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

	if (liveSharedHead)
		targetSharedHead = liveSharedHead->MakeClone();

	// Stratum slot partition (last stratum takes the rounding remainder)
	int size = RS_MAX(config.optBankSize, STRATUM_AMOUNT);
	int capOff = RS_MAX(1, (int)(config.optBankOffensiveFrac * size));
	int capDef = RS_MAX(1, (int)(config.optBankDefensiveFrac * size));
	int capRes = RS_MAX(1, size - capOff - capDef);
	strata[STRATUM_OFFENSIVE] = { 0, capOff };
	strata[STRATUM_DEFENSIVE] = { capOff, capDef };
	strata[STRATUM_RESOURCE] = { capOff + capDef, capRes };

	bankRows.assign((size_t)size * GOAL_DIM, 0.0f);
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

void GGL::GCRLOptionality::RefreshBank(const RLGC::FList candRows[STRATUM_AMOUNT], int64_t iteration, Report& report) {
	static const char* STRATUM_NAMES[STRATUM_AMOUNT] = { "Offensive", "Defensive", "Resource" };

	int insertedTotal = 0;
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
			bankInsertItr[slot] = iteration;
			stratum.cursor = (stratum.cursor + 1) % stratum.cap;
			stratum.count = RS_MIN(stratum.count + 1, stratum.cap);
		}
		insertedTotal += toInsert;

		int fill = BankFill();
		report["Opt/Bank Mix " + std::string(STRATUM_NAMES[s])] = fill > 0 ? (double)stratum.count / fill : 0.0;
	}

	report["Opt/Bank Fill"] = BankFill();
	report["Opt/Bank Inserted"] = insertedTotal;
	report["Opt/Bank Age Mean"] = BankAgeMean(iteration);

	// The frozen psi just moved (Polyak), so the whole bank re-embeds regardless of
	// whether anything was inserted — one [2*fill, 6] forward, negligible.
	ReembedBank();
}

void GGL::GCRLOptionality::ReembedBank() {
	RG_NO_GRAD;

	int fill = BankFill();
	bankPsiRows = fill;
	if (fill == 0) {
		bankPsi = torch::Tensor();
		return;
	}

	// Gather filled rows (strata are contiguous slot ranges)
	torch::Tensor rows = torch::empty({ fill, GOAL_DIM });
	int outIdx = 0;
	for (const BankStratum& s : strata) {
		if (s.count > 0) {
			rows.slice(0, outIdx, outIdx + s.count).copy_(
				torch::from_blob((void*)(bankRows.data() + (size_t)s.offset * GOAL_DIM),
					{ s.count, GOAL_DIM }, torch::TensorOptions().dtype(torch::kFloat32)));
			outIdx += s.count;
		}
	}

	torch::Tensor doubled = RowsWithFlippedTwins(rows).to(device, RG_H2D_NONBLOCKING(device), true);
	bankPsi = targetCritic->embed_psi(doubled);
}

torch::Tensor GGL::GCRLOptionality::PhiOptFromEmbeddings(torch::Tensor phiS, torch::Tensor bankPsi, float quasiTau, float optTemp) {
	// Exact scoring-time distance composition: d = -cos(phi, psi) / tau over the
	// L2-normalized embeddings, soft-min'd over the (doubled) bank.
	torch::Tensor D = -torch::matmul(phiS, bankPsi.transpose(0, 1)) / quasiTau;
	return optTemp * (torch::logsumexp(-D / optTemp, 1) - std::log((double)bankPsi.size(0)));
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

torch::Tensor GGL::GCRLOptionality::ComputePhiOpt(torch::Tensor tStatesCpu) {
	if (bankPsiRows == 0 || !bankPsi.defined())
		return {};

	RG_NO_GRAD;

	int64_t N = tStatesCpu.size(0);
	torch::Tensor out = torch::empty({ N }, torch::TensorOptions().dtype(torch::kFloat32));
	int64_t step = device.is_cpu() ? N : (int64_t)config.miniBatchSize;

	for (int64_t i = 0; i < N; i += step) {
		int64_t start = i, end = RS_MIN(i + step, N);
		torch::Tensor obs = tStatesCpu.slice(0, start, end).to(device, RG_H2D_NONBLOCKING(device), true);
		if (targetSharedHead)
			obs = targetSharedHead->Forward(obs, false);
		// Zero action components: the potential must be a function of state only
		torch::Tensor zeroActs = torch::zeros({ end - start, 8 }, torch::TensorOptions().dtype(torch::kFloat32).device(device));
		torch::Tensor phi = targetCritic->embed_phi(obs, zeroActs);
		out.slice(0, start, end).copy_(
			PhiOptFromEmbeddings(phi, bankPsi, config.gcrlTau, config.optTemp).cpu(), true);
	}

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
