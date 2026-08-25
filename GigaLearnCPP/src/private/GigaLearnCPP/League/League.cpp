#include "League.h"
#include "../PPO/PPOLearner.h"
#include <public/GigaLearnCPP/Util/Report.h>

#include <torch/csrc/api/include/torch/serialize.h>
#include <torch/nn/modules/loss.h>
#include <cstring>
#ifdef RG_CUDA_SUPPORT
#include <c10/cuda/CUDAStream.h>
#endif

using namespace torch;

// Same helper as Models.cpp's (static there): the NCCL stream for a device tensor.
static GGL::Dist::Session::Stream LeagueCudaStream(const torch::Tensor& t) {
#ifdef RG_CUDA_SUPPORT
	if (t.defined() && t.is_cuda())
		return (GGL::Dist::Session::Stream)c10::cuda::getCurrentCUDAStream(
			t.device().index()).stream();
#endif
	return nullptr;
}

// Masking constants mirror InferActionsFromModels exactly - variant sampling must obey
// the same action-legality semantics as the main policy.
static constexpr float ACTION_MIN_PROB = 1e-11f;
static constexpr float ACTION_DISABLED_LOGIT = -1e10f;

// The same normalized-entropy form as PPOLearner's ComputeEntropyRows (maskEntropy
// branch): entropy narrowed to the valid-action scope so heavily-masked states don't
// read as inherently low-entropy.
static torch::Tensor LeagueEntropyRows(torch::Tensor probs, torch::Tensor actionMasks) {
	auto entropy = -(probs.log() * probs).sum(-1);
	entropy /= actionMasks.to(torch::kFloat32).sum(-1).clamp_min(2).log();
	return entropy;
}

static std::vector<GGL::Model*> PolicyChain(GGL::ModelSet& models) {
	std::vector<GGL::Model*> chain;
	if (models["shared_head"])
		chain.push_back(models["shared_head"]);
	RG_ASSERT(models["policy"]);
	chain.push_back(models["policy"]);
	return chain;
}

void GGL::LeagueModule::MakeAdapters(const std::vector<Model*>& chain, AdapterStack& live, AdapterStack& snap) {
	RG_NO_GRAD;
	const int V = cfg.Total();
	const int r = cfg.rank;
	for (Model* m : chain) {
		for (int i = 0; i < (int)m->seq->size(); i++) {
			auto lin = std::dynamic_pointer_cast<torch::nn::LinearImpl>(m->seq->ptr(i));
			if (!lin)
				continue;
			int64_t out = lin->weight.size(0), in = lin->weight.size(1);
			// LoRA init: B = 0 (delta identically zero => variants ARE the main at birth,
			// which is also what makes a warm start with no LEAGUE files correct), A random
			// at 1/sqrt(r) so per-rank contributions stay O(1) once B grows.
			auto A = torch::randn({ V, r, out },
				TensorOptions().dtype(kFloat32).device(device)) * (1.f / sqrtf((float)r));
			auto B = torch::zeros({ V, r, in },
				TensorOptions().dtype(kFloat32).device(device));
			A.set_requires_grad(true);
			B.set_requires_grad(true);
			live.A.push_back(A);
			live.B.push_back(B);
			snap.A.push_back(A.detach().clone());
			snap.B.push_back(B.detach().clone());
		}
	}
}

GGL::LeagueModule::LeagueModule(ModelSet& baseModels, LeagueConfig config, torch::Device device, int numPlayers)
	: cfg(config), device(device) {

	RG_ASSERT(cfg.numDiverse >= 2); // the discriminator needs >= 2 classes
	RG_ASSERT(cfg.rank >= 1 && cfg.lagShort < cfg.lagLong);

	MakeAdapters(PolicyChain(baseModels), pol, polSnap);
	RG_ASSERT(baseModels["critic"]);
	MakeAdapters({ baseModels["critic"] }, cri, criSnap);

	// Discriminator MLPs. Deliberately standalone on raw descriptors: putting this on
	// the trunk would train shared perception to encode "which bot am I" (the
	// carStateHead incident / probes-never-reshape-the-trunk law).
	auto mkDisc = [&](int inDim) {
		return torch::nn::Sequential(
			torch::nn::Linear(inDim, 128), torch::nn::LeakyReLU(),
			torch::nn::Linear(128, 128), torch::nn::LeakyReLU(),
			torch::nn::Linear(128, cfg.numDiverse));
	};
	discPair = mkDisc(DESC_DIM * 2 + NUM_LAGS);
	discMarg = mkDisc(DESC_DIM);
	discPair->to(device);
	discMarg->to(device);
	discPairSnap = mkDisc(DESC_DIM * 2 + NUM_LAGS);
	discMargSnap = mkDisc(DESC_DIM); // CPU copies for the collect thread

	std::vector<torch::Tensor> adapterParams;
	for (auto* s : { &pol, &cri })
		for (auto* v : { &s->A, &s->B })
			for (auto& t : *v)
				adapterParams.push_back(t);
	// Adam, not Muon: low-rank factors and the reachability precedent (embedding-like
	// params train poorly under orthogonalized updates). LR set HERE at construction -
	// this module must never depend on being remembered in SetLearningRates (the vdag
	// lr=0 trap froze 42% of the net for weeks).
	adapterOptim = new torch::optim::Adam(adapterParams, cfg.adapterLR);
	std::vector<torch::Tensor> discParams;
	for (auto& p : discPair->parameters()) discParams.push_back(p);
	for (auto& p : discMarg->parameters()) discParams.push_back(p);
	discOptim = new torch::optim::Adam(discParams, cfg.discLR);

	rings.resize(numPlayers);
	for (auto& r2 : rings)
		r2.buf.resize((size_t)(cfg.lagLong + 1) * DESC_DIM);
	rdivMean.assign(cfg.Total(), 0.f);
	rdivVar.assign(cfg.Total(), 1.f);
	goalsFor.assign(cfg.Total(), 0);
	goalsAgainst.assign(cfg.Total(), 0);

	SyncCollectSnapshot();

	uint64_t nParams = 0;
	for (auto& t : LiveParams())
		nParams += t.numel();
	RG_LOG("LeagueModule: " << cfg.numDiverse << " diverse + " << cfg.numExploiters
		<< " exploiters, rank " << cfg.rank << ", " << pol.A.size() << " policy-path + "
		<< cri.A.size() << " critic-path Linears, " << Utils::NumToStr(nParams) << " params");
}

GGL::LeagueModule::~LeagueModule() {
	delete adapterOptim;
	delete discOptim;
}

std::vector<torch::Tensor> GGL::LeagueModule::LiveParams() {
	std::vector<torch::Tensor> out;
	for (auto* s : { &pol, &cri })
		for (auto* v : { &s->A, &s->B })
			for (auto& t : *v)
				out.push_back(t);
	for (auto& p : discPair->parameters()) out.push_back(p);
	for (auto& p : discMarg->parameters()) out.push_back(p);
	return out;
}

void GGL::LeagueModule::SyncCollectSnapshot() {
	RG_NO_GRAD;
	for (auto pr : { std::make_pair(&pol, &polSnap), std::make_pair(&cri, &criSnap) }) {
		for (size_t i = 0; i < pr.first->A.size(); i++) {
			pr.second->A[i].copy_(pr.first->A[i], true);
			pr.second->B[i].copy_(pr.first->B[i], true);
		}
	}
	auto copySeq = [](torch::nn::Sequential& from, torch::nn::Sequential& to) {
		auto fp = from->parameters();
		auto tp = to->parameters();
		for (size_t i = 0; i < fp.size(); i++)
			tp[i].copy_(fp[i].cpu());
	};
	copySeq(discPair, discPairSnap);
	copySeq(discMarg, discMargSnap);
}

void GGL::LeagueModule::BuildDescriptor(const RLGC::GameState& gs, const RLGC::Player& self,
	const RLGC::Player* opp, float* out) {
	// Canonical mirror: ORANGE sees itself attacking +y, like AdvancedObsPadded.
	const float m = (self.team == Team::ORANGE) ? -1.f : 1.f;
	auto putPos = [&](int i, const Vec& v) {
		out[i + 0] = m * v.x / 4096.f;
		out[i + 1] = m * v.y / 5120.f;
		out[i + 2] = v.z / 2044.f;
	};
	auto putVel = [&](int i, const Vec& v) {
		out[i + 0] = m * v.x / 2300.f;
		out[i + 1] = m * v.y / 2300.f;
		out[i + 2] = v.z / 2300.f;
	};
	putPos(0, gs.ball.pos);
	putVel(3, gs.ball.vel);
	putPos(6, self.pos);
	putVel(9, self.vel);
	const Vec f = self.rotMat.forward;
	out[12] = m * f.x; out[13] = m * f.y; out[14] = f.z;
	out[15] = self.rotMat.up.z;
	out[16] = self.boost / 100.f;
	out[17] = self.isOnGround ? 1.f : 0.f;
	out[18] = self.vel.Length() / 2300.f;
	putPos(19, gs.ball.pos - self.pos);
	if (opp) {
		putPos(22, opp->pos);
		putVel(25, opp->vel);
	} else {
		for (int i = 22; i < 28; i++)
			out[i] = 0.f;
	}
	static_assert(DESC_DIM == 28, "descriptor layout is hand-indexed");
}

torch::Tensor GGL::LeagueModule::ForwardLora(Model* m, torch::Tensor x, const AdapterStack& stack,
	torch::Tensor rowVariant, int& li) {
	// Mirrors InferActionsLowRankES's walk (residual spans honored), generalized to
	// rank r and usable WITH grad (the caller decides via NoGradGuard).
	const float scale = 1.f; // alpha/r folded into A's init scale
	std::vector<torch::Tensor> saved(m->residualSpans.size());
	for (int i = 0; i < (int)m->seq->size(); i++) {
		for (int s = 0; s < (int)m->residualSpans.size(); s++)
			if (m->residualSpans[s].first == i)
				saved[s] = x;
		torch::Tensor xin = x;
		auto modPtr = m->seq->ptr(i);
		x = GGL::ForwardSeqModule(modPtr, x);
		if (std::dynamic_pointer_cast<torch::nn::LinearImpl>(modPtr)) {
			RG_ASSERT((size_t)li < stack.A.size());
			auto Ag = stack.A[(size_t)li].index_select(0, rowVariant); // [rows, r, out]
			auto Bg = stack.B[(size_t)li].index_select(0, rowVariant); // [rows, r, in]
			auto u = (Bg * xin.unsqueeze(1)).sum(-1);                  // [rows, r]
			x = x + (u.unsqueeze(-1) * Ag).sum(1) * scale;             // [rows, out]
			li++;
		}
		for (int s = 0; s < (int)m->residualSpans.size(); s++)
			if (m->residualSpans[s].second == i && saved[s].defined())
				x = x + saved[s];
	}
	return x;
}

torch::Tensor GGL::LeagueModule::PolicyLogitsLora(ModelSet& base, torch::Tensor obs,
	const AdapterStack& stack, torch::Tensor rowVariant) {
	torch::Tensor x = obs;
	int li = 0;
	for (Model* m : PolicyChain(base))
		x = ForwardLora(m, x, stack, rowVariant, li);
	RG_ASSERT((size_t)li == stack.A.size());
	return x;
}

void GGL::LeagueModule::InferActions(ModelSet& base, torch::Tensor obs, torch::Tensor actionMasks,
	torch::Tensor rowVariant, torch::Tensor* outActions, torch::Tensor* outLogProbs) {
	RG_NO_GRAD;
	torch::Tensor logits = PolicyLogitsLora(base, obs, polSnap, rowVariant);
	if (logits.scalar_type() != torch::kFloat)
		logits = logits.to(torch::kFloat);
	auto rowOk = logits.isfinite().all(-1, true);
	logits = torch::where(rowOk, logits, torch::zeros_like(logits));
	auto probs = torch::softmax(
		logits + ACTION_DISABLED_LOGIT * actionMasks.to(torch::kBool).logical_not(), -1)
		.clamp(ACTION_MIN_PROB, 1);
	auto action = torch::multinomial(probs, 1, true);
	if (outActions)
		*outActions = action.flatten();
	if (outLogProbs)
		*outLogProbs = torch::log(probs).gather(-1, action).flatten();
}

std::vector<float> GGL::LeagueModule::StepRdivAndPush(const std::vector<int>& players,
	const std::vector<int>& variants, const std::vector<float>& descs) {
	RG_NO_GRAD;
	const int n = (int)players.size();
	std::vector<float> rdiv((size_t)n, 0.f);
	if (n == 0)
		return rdiv;
	RG_ASSERT(descs.size() == (size_t)n * DESC_DIM);

	const int lags[NUM_LAGS] = { cfg.lagShort, cfg.lagLong };
	const int ringCap = cfg.lagLong + 1;

	// Gather valid (d_prev, d_now) pairs per lag BEFORE pushing d_now (ring holds
	// strictly-previous steps at this point).
	struct PairBatch { std::vector<float> d0, d1; std::vector<int> row; };
	PairBatch pb[NUM_LAGS];
	for (int k = 0; k < n; k++) {
		auto& ring = rings[players[k]];
		for (int L = 0; L < NUM_LAGS; L++) {
			int lag = lags[L];
			if (ring.count < lag)
				continue;
			// d_prev = the descriptor pushed `lag` steps ago
			int idx = (ring.head - lag + ringCap * 2) % ringCap;
			const float* dPrev = &ring.buf[(size_t)idx * DESC_DIM];
			const float* dNow = &descs[(size_t)k * DESC_DIM];
			pb[L].d0.insert(pb[L].d0.end(), dPrev, dPrev + DESC_DIM);
			pb[L].d1.insert(pb[L].d1.end(), dNow, dNow + DESC_DIM);
			pb[L].row.push_back(k);
		}
	}

	// Push d_now into rings + harvest reservoir tuples (diverse variants only; the
	// exploiters are not identity classes, their whole point is convergence on holes).
	for (int k = 0; k < n; k++) {
		auto& ring = rings[players[k]];
		std::memcpy(&ring.buf[(size_t)ring.head * DESC_DIM], &descs[(size_t)k * DESC_DIM],
			DESC_DIM * sizeof(float));
		ring.head = (ring.head + 1) % ringCap;
		ring.count = RS_MIN(ring.count + 1, ringCap);
	}
	for (int L = 0; L < NUM_LAGS; L++) {
		for (size_t j = 0; j < pb[L].row.size(); j++) {
			int k = pb[L].row[j];
			if (variants[k] >= cfg.numDiverse)
				continue;
			auto& res = reservoir;
			res.seen++;
			if (res.Size() < res.cap) {
				res.d0.insert(res.d0.end(), &pb[L].d0[j * DESC_DIM], &pb[L].d0[j * DESC_DIM] + DESC_DIM);
				res.d1.insert(res.d1.end(), &pb[L].d1[j * DESC_DIM], &pb[L].d1[j * DESC_DIM] + DESC_DIM);
				res.lag.push_back((int8_t)L);
				res.z.push_back((int8_t)variants[k]);
			} else {
				// reservoir sampling: uniform over everything seen
				uint64_t slot = rng() % (uint64_t)res.seen;
				if (slot < (uint64_t)res.cap) {
					std::memcpy(&res.d0[slot * DESC_DIM], &pb[L].d0[j * DESC_DIM], DESC_DIM * sizeof(float));
					std::memcpy(&res.d1[slot * DESC_DIM], &pb[L].d1[j * DESC_DIM], DESC_DIM * sizeof(float));
					res.lag[slot] = (int8_t)L;
					res.z[slot] = (int8_t)variants[k];
				}
			}
		}
	}

	// r_div forward (CPU snapshots). Warmup: an untrained discriminator's log-ratios
	// are noise; injecting them would be a random reward.
	if (discUpdates < cfg.discWarmupUpdates)
		return rdiv;

	std::vector<float> raw((size_t)n, 0.f);
	std::vector<uint8_t> any((size_t)n, 0);
	for (int L = 0; L < NUM_LAGS; L++) {
		int64_t rowsL = (int64_t)pb[L].row.size();
		if (!rowsL)
			continue;
		auto d0 = torch::from_blob(pb[L].d0.data(), { rowsL, DESC_DIM }, kFloat32);
		auto d1 = torch::from_blob(pb[L].d1.data(), { rowsL, DESC_DIM }, kFloat32);
		auto lagOne = torch::zeros({ rowsL, NUM_LAGS }, kFloat32);
		lagOne.narrow(1, L, 1).fill_(1.f);
		auto logqPair = torch::log_softmax(discPairSnap->forward(torch::cat({ d0, d1, lagOne }, 1)), -1);
		auto logqMarg = torch::log_softmax(discMargSnap->forward(d0), -1);
		auto pairAcc = logqPair.accessor<float, 2>();
		auto margAcc = logqMarg.accessor<float, 2>();
		for (int64_t j = 0; j < rowsL; j++) {
			int k = pb[L].row[j];
			int z = variants[k];
			if (z >= cfg.numDiverse)
				continue;
			raw[k] += pairAcc[j][z] - margAcc[j][z];
			any[k] = 1;
		}
	}
	for (int k = 0; k < n; k++) {
		if (!any[k])
			continue;
		int z = variants[k];
		// EMA center + clamp at 3 sigma; beta-scaled. Centering removes the "always
		// identifiable" baseline so only above-typical identifiability pays.
		float mean = rdivMean[z], var = rdivVar[z];
		float c = raw[k] - mean;
		float sd = sqrtf(RS_MAX(var, 1e-6f));
		c = RS_CLAMP(c, -3.f * sd, 3.f * sd);
		rdiv[k] = cfg.divBeta * c;
		rdivMean[z] = 0.999f * mean + 0.001f * raw[k];
		rdivVar[z] = 0.999f * var + 0.001f * (raw[k] - mean) * (raw[k] - mean);
		rdivSum += rdiv[k]; rdivSqSum += (double)rdiv[k] * rdiv[k]; rdivCnt++;
	}
	return rdiv;
}

void GGL::LeagueModule::PrepareLearnData() {
	RG_NO_GRAD;
	// Harvest collect-side telemetry while the worker is provably idle.
	statRdivCnt = rdivCnt;
	if (rdivCnt) {
		double m = rdivSum / rdivCnt;
		statRdivMean = (float)m;
		statRdivStd = (float)sqrt(RS_MAX(0.0, rdivSqSum / rdivCnt - m * m));
		rdivSum = rdivSqSum = 0;
		rdivCnt = 0;
	}
	statGoalsFor = goalsFor;
	statGoalsAgainst = goalsAgainst;
	statReservoirFill = reservoir.Size();

	discD0 = discD1 = discLag = discZ = torch::Tensor();
	int64_t sz = reservoir.Size();
	if (sz < 256) // too few tuples to train on meaningfully
		return;
	int64_t b = RS_MIN((int64_t)cfg.discBatch, sz);
	auto idx = torch::randint(0, sz, { b }, TensorOptions().dtype(kLong));
	auto d0 = torch::from_blob(reservoir.d0.data(), { sz, DESC_DIM }, kFloat32).index_select(0, idx);
	auto d1 = torch::from_blob(reservoir.d1.data(), { sz, DESC_DIM }, kFloat32).index_select(0, idx);
	auto lag = torch::from_blob(reservoir.lag.data(), { sz }, kInt8).index_select(0, idx);
	auto z = torch::from_blob(reservoir.z.data(), { sz }, kInt8).index_select(0, idx);
	discD0 = d0.to(device);
	discD1 = d1.to(device);
	discLag = lag.to(device, kLong);
	discZ = z.to(device, kLong);
}

torch::Tensor GGL::LeagueModule::InferValues(PPOLearner* ppo, torch::Tensor obs, torch::Tensor rowVariant) {
	RG_NO_GRAD;
	torch::Tensor trunk = ppo->ValueTrunk(obs, false);
	int li = 0;
	torch::Tensor v = ForwardLora(ppo->models["critic"], trunk, cri, rowVariant, li);
	return v.flatten();
}

void GGL::LeagueModule::Learn(PPOLearner* ppo, torch::Tensor states, torch::Tensor actionMasks,
	torch::Tensor actions, torch::Tensor logProbs, torch::Tensor advantages,
	torch::Tensor targetValues, torch::Tensor rowVariant,
	Dist::Session* dist, Report& report) {

	const int64_t n = states.defined() ? states.size(0) : 0;
	const int V = cfg.Total();

	// Pre-step copy for the update-magnitude panel (the "is it actually training"
	// tripwire this project keeps re-learning the need for).
	torch::Tensor preFlat;
	{
		RG_NO_GRAD;
		std::vector<torch::Tensor> flats;
		for (auto* s : { &pol, &cri })
			for (auto* v2 : { &s->A, &s->B })
				for (auto& t : *v2)
					flats.push_back(t.detach().flatten());
		preFlat = torch::cat(flats).clone();
	}

	// RAW advantages, exactly like the main PPO pass. The first cluster deploy
	// normalized them to unit std per variant - under GCO a 24-step fragment almost
	// never contains a goal, so that amplified near-pure value noise to unit scale
	// (a 10-100x hotter policy gradient of noise than the main's calibration) and
	// walked every variant into conceding 25:1 within an hour. Advantage SCALE is
	// part of the main run's tuned economy (returnStd scaling, entropyScale balance);
	// the league must inherit it, not invent its own.
	double avgPolicyLoss = 0, avgCriticLoss = 0, avgEntropy = 0, avgDiscLoss = 0;
	double avgClipFrac = 0;
	int64_t updates = 0;
	(void)V;

	for (int epoch = 0; epoch < cfg.epochs; epoch++) {
		// PPO on variant rows (single pass; league buffers are small - minibatch only
		// if the row count ever exceeds cfg.miniBatch).
		for (int64_t start = 0; start < n; start += cfg.miniBatch) {
			int64_t stop = RS_MIN(start + cfg.miniBatch, n);
			auto sl = [&](torch::Tensor t) { return t.slice(0, start, stop); };

			torch::Tensor logits = PolicyLogitsLora(ppo->models, sl(states), pol, sl(rowVariant));
			auto masks = sl(actionMasks);
			auto probs = torch::softmax(
				logits + ACTION_DISABLED_LOGIT * masks.to(torch::kBool).logical_not(), -1)
				.clamp(ACTION_MIN_PROB, 1);
			auto newLogProbs = torch::log(probs).gather(-1, sl(actions).unsqueeze(-1)).flatten();
			auto entropy = LeagueEntropyRows(probs, masks).mean();

			auto ratio = torch::exp(newLogProbs - sl(logProbs));
			auto adv = sl(advantages);
			auto clipped = torch::clamp(ratio, 1.f - cfg.clipRange, 1.f + cfg.clipRange);
			auto policyLoss = -torch::min(ratio * adv, clipped * adv).mean();

			// Per-variant critic on the DETACHED value trunk: variant value error must
			// not reshape shared perception; only the critic-head LoRA (and, unavoidably,
			// the base critic head - zeroed below) receive gradient.
			torch::Tensor trunk;
			{
				RG_NO_GRAD;
				trunk = ppo->ValueTrunk(sl(states), false);
			}
			int li = 0;
			auto vPred = ForwardLora(ppo->models["critic"], trunk.detach(), cri, sl(rowVariant), li).flatten();
			auto criticLoss = (vPred - sl(targetValues)).pow(2).mean();

			torch::Tensor loss = policyLoss - entropy * cfg.entropyScale + criticLoss * 0.5f;
			loss.backward();

			avgPolicyLoss += policyLoss.item<float>();
			avgCriticLoss += criticLoss.item<float>();
			avgEntropy += entropy.item<float>();
			avgClipFrac += ((ratio - 1).abs() > cfg.clipRange).to(kFloat32).mean().item<float>();
			updates++;
		}

		// Discriminator CE (pair + marginal heads), once per epoch on the barrier-
		// sampled batch. Trained even when no PPO rows exist this iteration.
		if (discD0.defined()) {
			auto lagOne = torch::one_hot(discLag, NUM_LAGS).to(kFloat32);
			auto pairLogits = discPair->forward(torch::cat({ discD0, discD1, lagOne }, 1));
			auto margLogits = discMarg->forward(discD0);
			auto discLoss = torch::nn::CrossEntropyLoss()(pairLogits, discZ)
				+ torch::nn::CrossEntropyLoss()(margLogits, discZ);
			discLoss.backward();
			avgDiscLoss += discLoss.item<float>();
			{
				RG_NO_GRAD;
				auto acc = (pairLogits.argmax(-1) == discZ).to(kFloat32).mean().item<float>();
				auto lagMask = discLag == (NUM_LAGS - 1);
				if (lagMask.any().item<bool>()) {
					auto accLong = (pairLogits.argmax(-1) == discZ).masked_select(lagMask)
						.to(kFloat32).mean().item<float>();
					report["League/Disc Acc Long"] = accLong;
				}
				report["League/Disc Acc"] = acc;
			}
			discUpdates++;
		}

		// Grad sync + step: league params only. Every rank hosts every variant, so the
		// allreduce keeps adapters bit-identical across ranks like the main models.
		if (dist && dist->distributed()) {
			std::vector<Dist::Session::GradRef> refs;
			for (auto& p : LiveParams()) {
				auto g = p.mutable_grad();
				if (!g.defined()) {
					g = torch::zeros_like(p);
					p.mutable_grad() = g;
				}
				if (!g.is_contiguous()) {
					p.mutable_grad() = g.contiguous();
					g = p.grad();
				}
				refs.push_back({ g.data_ptr<float>(), (size_t)g.numel() });
			}
			dist->allreduce_avg_grads(refs, LeagueCudaStream(LiveParams()[0].grad()));
		}
		adapterOptim->step();
		adapterOptim->zero_grad();
		if (discD0.defined()) {
			discOptim->step();
			discOptim->zero_grad();
		}
	}

	// The league backward necessarily deposited gradients on the BASE weights
	// (shared_head/policy/critic are part of the graph). They must never leak into the
	// next main optimizer step - zero every base grad we could have touched.
	for (Model* m : ppo->models)
		if (m->optim)
			m->optim->zero_grad();

	{
		RG_NO_GRAD;
		std::vector<torch::Tensor> flats;
		for (auto* s : { &pol, &cri })
			for (auto* v2 : { &s->A, &s->B })
				for (auto& t : *v2)
					flats.push_back(t.detach().flatten());
		auto postFlat = torch::cat(flats);
		report["League/Adapter Update Magnitude"] = (postFlat - preFlat).abs().mean().item<float>();
		report["League/Adapter Norm"] = postFlat.norm().item<float>();
	}
	if (updates) {
		report["League/Policy Loss"] = (float)(avgPolicyLoss / updates);
		report["League/Critic Loss"] = (float)(avgCriticLoss / updates);
		report["League/Entropy"] = (float)(avgEntropy / updates);
		report["League/Clip Frac"] = (float)(avgClipFrac / updates);
	}
	// One compact stdout status line (rank 0 only - the 192-rank ungated-log lesson).
	// The stdout log is what gets tailed on the cluster; a league that silently went
	// inert must be visible there, not only in wandb.
	if (!dist || dist->rank() == 0) {
		// Mean goal share vs the main per role (criteria 3/4 readable from the log
		// alone - this lineage runs with wandb disabled).
		auto share = [&](int lo, int hi) {
			int64_t gf = 0, ga = 0;
			for (int v = lo; v < hi && v < (int)statGoalsFor.size(); v++) {
				gf += statGoalsFor[v];
				ga += statGoalsAgainst[v];
			}
			return (gf + ga > 0) ? (double)gf / (double)(gf + ga) : -1.0;
		};
		RG_LOG("League: rows=" << n << " updMag=" << report["League/Adapter Update Magnitude"]
			<< " adNorm=" << report["League/Adapter Norm"]
			<< " ent=" << (updates ? avgEntropy / updates : -1.)
			<< " discAcc=" << (report.Has("League/Disc Acc") ? report["League/Disc Acc"] : -1.)
			<< " res=" << statReservoirFill
			<< " rdivMean=" << statRdivMean
			<< " gsDiv=" << share(0, cfg.numDiverse)
			<< " gsExp=" << share(cfg.numDiverse, cfg.Total())
			<< " (warmup " << discUpdates << "/" << cfg.discWarmupUpdates << ")");
	}
	if (avgDiscLoss != 0)
		report["League/Disc Loss"] = (float)(avgDiscLoss / cfg.epochs);
	report["League/Rows"] = (float)n;
	report["League/Reservoir Fill"] = (float)statReservoirFill;
	if (statRdivCnt) {
		report["League/Rdiv Mean"] = statRdivMean;
		report["League/Rdiv Std"] = statRdivStd;
	}
	for (int v = 0; v < (int)statGoalsFor.size(); v++) {
		int64_t gf = statGoalsFor[v], ga = statGoalsAgainst[v];
		if (gf + ga > 0)
			report[(v < cfg.numDiverse ? "League/GoalShare Div " : "League/GoalShare Exp ")
				+ std::to_string(v)] = (float)gf / (float)(gf + ga);
	}
}

void GGL::LeagueModule::Save(std::filesystem::path folder) {
	RG_NO_GRAD;
	torch::serialize::OutputArchive ar;
	auto put = [&](const std::string& name, const std::vector<torch::Tensor>& ts) {
		for (size_t i = 0; i < ts.size(); i++)
			ar.write(name + std::to_string(i), ts[i].detach().cpu());
	};
	put("polA", pol.A); put("polB", pol.B);
	put("criA", cri.A); put("criB", cri.B);
	auto dp = discPair->parameters(), dm = discMarg->parameters();
	put("discPair", { dp.begin(), dp.end() });
	put("discMarg", { dm.begin(), dm.end() });
	ar.write("discUpdates", torch::tensor((int64_t)discUpdates));
	ar.save_to((folder / "LEAGUE.lt").string());
}

void GGL::LeagueModule::Load(std::filesystem::path folder) {
	auto path = folder / "LEAGUE.lt";
	if (!std::filesystem::exists(path)) {
		RG_LOG("LeagueModule: no LEAGUE.lt in " << folder << " - fresh adapters (variants start as the main)");
		return;
	}
	RG_NO_GRAD;
	torch::serialize::InputArchive ar;
	ar.load_from(path.string());
	auto get = [&](const std::string& name, std::vector<torch::Tensor>& ts) {
		for (size_t i = 0; i < ts.size(); i++) {
			torch::Tensor t;
			ar.read(name + std::to_string(i), t);
			RG_ASSERT(t.sizes() == ts[i].sizes());
			ts[i].copy_(t.to(ts[i].device()));
		}
	};
	get("polA", pol.A); get("polB", pol.B);
	get("criA", cri.A); get("criB", cri.B);
	auto loadSeq = [&](const std::string& name, torch::nn::Sequential& seq) {
		auto ps = seq->parameters();
		for (size_t i = 0; i < ps.size(); i++) {
			torch::Tensor t;
			ar.read(name + std::to_string(i), t);
			ps[i].copy_(t.to(ps[i].device()));
		}
	};
	loadSeq("discPair", discPair);
	loadSeq("discMarg", discMarg);
	torch::Tensor du;
	ar.read("discUpdates", du);
	discUpdates = du.item<int64_t>();
	SyncCollectSnapshot();
	RG_LOG("LeagueModule: loaded LEAGUE.lt (discUpdates=" << discUpdates << ")");
}

void GGL::LeagueModule::BroadcastParams(Dist::Session* dist) {
	if (!dist || !dist->distributed())
		return;
	RG_NO_GRAD;
	for (auto& p : LiveParams()) {
		if (p.is_cuda() && p.scalar_type() == torch::kFloat) {
			auto t = p.detach().contiguous();
			dist->bcast_device(t.data_ptr<float>(), (size_t)t.numel(), 0, LeagueCudaStream(t));
			if (!p.is_same(t))
				p.copy_(t);
		} else {
			auto cpu = p.detach().contiguous().cpu();
			dist->bcast_host(cpu.data_ptr(), (size_t)cpu.numel() * cpu.element_size(), 0);
			p.copy_(cpu);
		}
	}
	SyncCollectSnapshot();
}
