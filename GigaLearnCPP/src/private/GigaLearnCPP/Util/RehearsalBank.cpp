#include "RehearsalBank.h"
#include <RLGymCPP/Framework.h>
#include <private/GigaLearnCPP/FrameworkTorch.h>
#include <public/GigaLearnCPP/Util/Report.h>
#include <torch/serialize.h>
#include <algorithm>
#include <cmath>

void GGL::RehearsalBank::Admit(const torch::Tensor& states, const torch::Tensor& actions, const torch::Tensor& masks,
	const torch::Tensor& targets, const torch::Tensor& terminals, const torch::Tensor& conv,
	const torch::Tensor& mainProb, const torch::Tensor& headroom,
	const torch::Tensor& intentIds, const torch::Tensor& intentFeat,
	const torch::Tensor& stateNovelty, const torch::Tensor& playerIds,
	int64_t iteration, float pThresh, float targetThresh) {
	RG_NO_GRAD;
	const int64_t n = actions.size(0);
	carriedRows = 0; carriedSegments = 0;
	if (n == 0) return;
	const bool carry = cfg.carryTails && playerIds.defined();
	torch::Tensor pidA = carry ? playerIds.to(torch::kCPU).to(torch::kInt32).contiguous() : torch::Tensor();
	const int32_t* pid = carry ? pidA.data_ptr<int32_t>() : nullptr;
	if (carry) RG_ASSERT(pidA.numel() == n);
	// First pass: which segment is each player's FIRST and LAST in this batch (segments are
	// terminal-delimited and never span players: every appended trajectory ends in a terminal).
	std::unordered_map<int32_t, int64_t> firstSeg, lastSeg;
	if (carry) {
		int64_t segStart = 0, segIdx = 0;
		auto termT = terminals.to(torch::kCPU).to(torch::kInt8).contiguous();
		const int8_t* tt = termT.data_ptr<int8_t>();
		for (int64_t i = 0; i < n; i++) {
			if (tt[i] == 0 && i != n - 1) continue;
			int32_t p = pid[segStart];
			if (!firstSeg.count(p)) firstSeg[p] = segIdx;
			lastSeg[p] = segIdx;
			segIdx++; segStart = i + 1;
		}
	}
	std::unordered_map<int32_t, Prefix> nextPrefixes;
	RG_ASSERT(states.size(0) == n && masks.size(0) == n && targets.numel() == n
		&& terminals.numel() == n && conv.numel() == n && mainProb.numel() == n && headroom.numel() == n);
	const bool withIntent = intentIds.defined() && intentFeat.defined();
	if (withIntent) RG_ASSERT(intentIds.size(0) == n && intentFeat.size(0) == n);
	auto termA = terminals.to(torch::kCPU).to(torch::kInt8).contiguous();
	auto convA = conv.to(torch::kCPU).to(torch::kBool).contiguous();
	auto probA = mainProb.to(torch::kCPU).to(torch::kFloat32).contiguous();
	auto hA = headroom.to(torch::kCPU).to(torch::kFloat32).contiguous();
	auto tgA = targets.to(torch::kCPU).to(torch::kFloat32).contiguous();
	const float* tgp = tgA.data_ptr<float>();
	const int8_t* tp = termA.data_ptr<int8_t>();
	const bool* cp = convA.data_ptr<bool>();
	const float* pp = probA.data_ptr<float>();
	const float* hp = hA.data_ptr<float>();
	torch::Tensor novA = stateNovelty.defined() ? stateNovelty.to(torch::kCPU).to(torch::kFloat32).contiguous() : torch::zeros({ n }, torch::kFloat32);
	RG_ASSERT(novA.numel() == n);
	const float* np_ = novA.data_ptr<float>();

	int64_t start = 0, segIdx = 0;
	for (int64_t i = 0; i < n; i++) {
		if (tp[i] == 0 && i != n - 1) continue;
		const int64_t s0 = start, e0 = i; // inclusive segment
		const int64_t mySeg = segIdx++;
		start = i + 1;
		// Carried tail: the player's first segment continues its previous iteration's truncated
		// segment; gate, rank and store the CHAIN prefix + segment.
		const Prefix* pre = nullptr; int32_t p = -1;
		if (carry) {
			p = pid[s0];
			if (firstSeg[p] == mySeg) { auto it = prefixes.find(p); if (it != prefixes.end()) pre = &it->second; }
		}
		const int64_t pn = pre ? pre->actions.size(0) : 0;
		bool anyConv = false, anyRare = false; double hSum = 0, sSum = 0, sMax = 0, nMax = 0, hMax = -1e30; float tgMax = -1e30f;
		if (pre) {
			const float* ppp = pre->prob.data_ptr<float>(); const float* ptg = pre->targets.data_ptr<float>(); const float* pnv = pre->novelty.data_ptr<float>();
			const float* phd = pre->headroom.defined() ? pre->headroom.data_ptr<float>() : nullptr;
			for (int64_t j = 0; j < pn; j++) {
				anyRare |= (ppp[j] < pThresh);
				const double sj = -std::log(RS_MAX(ppp[j], 1e-12f));
				sSum += sj; sMax = RS_MAX(sMax, sj); nMax = RS_MAX(nMax, (double)pnv[j]);
				if (phd) hMax = RS_MAX(hMax, (double)phd[j]);
				tgMax = RS_MAX(tgMax, ptg[j]);
			}
			carriedSegments++; carriedRows += pn;
		}
		for (int64_t j = s0; j <= e0; j++) {
			anyConv |= cp[j];
			anyRare |= (pp[j] < pThresh);
			hSum += hp[j];
			const double sj = -std::log(RS_MAX(pp[j], 1e-12f));
			sSum += sj; sMax = RS_MAX(sMax, sj); nMax = RS_MAX(nMax, (double)np_[j]);
			hMax = RS_MAX(hMax, (double)hp[j]);
			tgMax = RS_MAX(tgMax, tgp[j]);
		}
		// Next iteration's carried tail for this player: the last <= maxRows rows of the chain
		// if this is the player's LAST segment here and it was TRUNCATED (episode continues).
		auto fnChain = [&](int64_t keepFrom /*chain-relative start*/) {
			// chain-relative index k in [0, pn + L): k < pn -> prefix row k; else batch row s0 + (k - pn)
			const int64_t L = e0 - s0 + 1, total = pn + L;
			auto cat2 = [&](const torch::Tensor& preT, const torch::Tensor& batchT, torch::ScalarType dt) {
				torch::Tensor b = batchT.slice(0, s0, e0 + 1).to(torch::kCPU, dt);
				if (pn == 0) return b.slice(0, keepFrom, L).clone();
				torch::Tensor c = torch::cat({ preT.to(dt), b }, 0);
				return c.slice(0, keepFrom, total).clone();
			};
			Prefix out;
			out.states = cat2(pre ? pre->states : torch::Tensor(), states, torch::kFloat32);
			out.actions = cat2(pre ? pre->actions : torch::Tensor(), actions, torch::kInt64);
			out.masks = cat2(pre ? pre->masks : torch::Tensor(), masks, torch::kUInt8);
			out.targets = cat2(pre ? pre->targets : torch::Tensor(), targets, torch::kFloat32);
			out.prob = cat2(pre ? pre->prob : torch::Tensor(), mainProb, torch::kFloat32);
			out.novelty = cat2(pre ? pre->novelty : torch::Tensor(), novA, torch::kFloat32);
			out.headroom = cat2(pre ? pre->headroom : torch::Tensor(), hA, torch::kFloat32);
			if (withIntent) {
				out.intentIds = cat2(pre ? pre->intentIds : torch::Tensor(), intentIds, torch::kInt64);
				out.intentFeat = cat2(pre ? pre->intentFeat : torch::Tensor(), intentFeat, torch::kFloat32);
			}
			return out;
		};
		if (carry && lastSeg[p] == mySeg && tp[e0] == 2 /*TRUNCATED*/) {
			const int64_t total = pn + (e0 - s0 + 1);
			nextPrefixes[p] = fnChain(RS_MAX((int64_t)0, total - cfg.maxRowsPerEntry));
		}
		candidates++;
		if (!(tgMax > cfg.minTarget) || !(tgMax > targetThresh)) { refusedNegative++; continue; }
		if (cfg.requireConv && !anyConv) { refusedNoConv++; continue; }
		if (!anyRare) { refusedCommon++; continue; }
		// keep at most maxRowsPerEntry rows of the CHAIN, the ones that END at the success
		const int64_t total = pn + (e0 - s0 + 1);
		Prefix chain = fnChain(RS_MAX((int64_t)0, total - cfg.maxRowsPerEntry));
		auto a = chain.actions;
		auto st = chain.states;
		int64_t resetEdgesChain = 0;
		if (st.size(1) > 77 && st.size(0) > 1) {
			auto hf = st.select(1, 77) > 0.5f; auto og = st.select(1, 76) > 0.5f; const int64_t L = st.size(0);
			resetEdgesChain = ((hf.slice(0, 1, L) & ~hf.slice(0, 0, L - 1)) & ~og.slice(0, 1, L)).sum().item<int64_t>();
		}
		if (cfg.requireResetEdge && resetEdgesChain == 0) { refusedNoEvent++; continue; }
		bool dup = false;
		for (auto& x : entries) {
			if (x.actions.numel() != a.numel() || !torch::equal(x.actions, a)) continue;
			if ((x.states[0] - st[0]).norm().item<float>() < 1e-3f) { dup = true; break; }
		}
		if (dup) { refusedDuplicate++; continue; }
		Entry e;
		e.score = (float)(cfg.rankMode == 1 ? sSum : cfg.rankMode == 0 ? sMax : cfg.rankMode == 3 ? hMax : nMax);
		e.maxSurprisal = (float)sMax; e.maxNovelty = (float)nMax; e.maxHeadroom = (float)hMax;
		e.states = st;
		e.actions = a;
		e.masks = chain.masks;
		e.targets = chain.targets;
		if (withIntent) { e.intentIds = chain.intentIds; e.intentFeat = chain.intentFeat; }
		e.carried = RS_MAX((int64_t)0, pn - RS_MAX((int64_t)0, total - cfg.maxRowsPerEntry));
		e.resetEdges = resetEdgesChain;
		e.headroom = (float)(hSum / (double)(e0 - s0 + 1));
		e.bestTarget = tgMax;
		e.born = iteration;
		if ((int)entries.size() >= cfg.capacity) {
			// COMPETITIVE admission: at capacity a candidate must beat the weakest entry's
			// score, else it is refused and the bank keeps what it has (an
			// unconditional evict-then-push made the gco-2.1 bank a window of the last 8
			// goals). Success is the GATE, rarity is the RANKING.
			size_t victim = 0;
			for (size_t k = 1; k < entries.size(); k++)
				if (entries[k].score < entries[victim].score
					|| (entries[k].score == entries[victim].score && entries[k].born < entries[victim].born))
					victim = k;
			if (!(e.score > entries[victim].score)) { refusedWeak++; continue; }
			entries.erase(entries.begin() + (ptrdiff_t)victim);
			evicted++;
		}
		entries.push_back(std::move(e));
		admitted++;
	}
	if (carry) prefixes.swap(nextPrefixes); // players not truncated this iteration drop their tail
}

void GGL::RehearsalBank::Expire(int64_t iteration) {
	size_t before = entries.size();
	entries.erase(std::remove_if(entries.begin(), entries.end(),
		[&](const Entry& e) { return iteration - e.born > cfg.life; }), entries.end());
	expired += (int64_t)(before - entries.size());
}

GGL::RehearsalBank::Rows GGL::RehearsalBank::GetRows(bool dropLast) const {
	Rows r;
	std::vector<torch::Tensor> st, ac, mk, tg, ii, ifeat;
	bool allIntent = !entries.empty();
	for (auto& e : entries) allIntent &= e.intentIds.defined() && e.intentFeat.defined();
	for (auto& e : entries) {
		int64_t L = e.actions.size(0);
		if (dropLast) L -= 1;
		if (L <= 0) continue;
		st.push_back(e.states.slice(0, 0, L));
		ac.push_back(e.actions.slice(0, 0, L));
		mk.push_back(e.masks.slice(0, 0, L));
		tg.push_back(e.targets.slice(0, 0, L));
		if (allIntent) { ii.push_back(e.intentIds.slice(0, 0, L)); ifeat.push_back(e.intentFeat.slice(0, 0, L)); }
	}
	if (st.empty()) return r;
	r.states = torch::cat(st, 0); r.actions = torch::cat(ac, 0); r.masks = torch::cat(mk, 0); r.targets = torch::cat(tg, 0);
	if (allIntent) { r.intentIds = torch::cat(ii, 0); r.intentFeat = torch::cat(ifeat, 0); }
	r.n = r.actions.size(0);
	return r;
}

int64_t GGL::RehearsalBank::RowCount() const {
	int64_t n = 0;
	for (auto& e : entries) n += e.actions.size(0);
	return n;
}

size_t GGL::RehearsalBank::Bytes() const {
	size_t b = 0;
	for (auto& e : entries)
		for (auto* t : { &e.states, &e.actions, &e.masks, &e.targets, &e.intentIds, &e.intentFeat })
			if (t->defined()) b += (size_t)t->numel() * t->element_size();
	return b;
}

void GGL::RehearsalBank::Save(const std::filesystem::path& folder) const {
	torch::serialize::OutputArchive ar;
	ar.write("count", torch::tensor((int64_t)entries.size()));
	for (size_t i = 0; i < entries.size(); i++) {
		auto& e = entries[i]; std::string p = "e" + std::to_string(i) + "_";
		ar.write(p + "states", e.states); ar.write(p + "actions", e.actions);
		ar.write(p + "masks", e.masks); ar.write(p + "targets", e.targets);
		ar.write(p + "has_intent", torch::tensor((int64_t)(e.intentIds.defined() ? 1 : 0)));
		if (e.intentIds.defined()) { ar.write(p + "intent_ids", e.intentIds); ar.write(p + "intent_feat", e.intentFeat); }
		ar.write(p + "meta", torch::tensor({ (double)e.headroom, (double)e.born, (double)e.bestTarget, (double)e.score, (double)e.maxSurprisal, (double)e.maxNovelty, (double)e.carried, (double)e.resetEdges, (double)e.maxHeadroom }, torch::kFloat64));
	}
	ar.save_to((folder / "REHEARSAL_BANK.lt").string());
}

void GGL::RehearsalBank::Load(const std::filesystem::path& folder) {
	auto path = folder / "REHEARSAL_BANK.lt";
	entries.clear();
	if (!std::filesystem::exists(path)) return;
	try {
		torch::serialize::InputArchive ar;
		ar.load_from(path.string());
		torch::Tensor cnt; ar.read("count", cnt);
		int64_t c = cnt.item<int64_t>();
		for (int64_t i = 0; i < c; i++) {
			std::string p = "e" + std::to_string(i) + "_";
			Entry e; torch::Tensor meta, hasIntent;
			ar.read(p + "states", e.states); ar.read(p + "actions", e.actions);
			ar.read(p + "masks", e.masks); ar.read(p + "targets", e.targets); ar.read(p + "meta", meta);
			ar.read(p + "has_intent", hasIntent);
			if (hasIntent.item<int64_t>() != 0) { ar.read(p + "intent_ids", e.intentIds); ar.read(p + "intent_feat", e.intentFeat); }
			e.headroom = (float)meta[0].item<double>(); e.born = (int64_t)meta[1].item<double>();
			e.bestTarget = (float)meta[2].item<double>(); e.score = (float)meta[3].item<double>();
			if (meta.numel() >= 6) { e.maxSurprisal = (float)meta[4].item<double>(); e.maxNovelty = (float)meta[5].item<double>(); }
			if (meta.numel() >= 7) e.carried = (int64_t)meta[6].item<double>();
			if (meta.numel() >= 8) e.resetEdges = (int64_t)meta[7].item<double>();
			if (meta.numel() >= 9) e.maxHeadroom = (float)meta[8].item<double>();
			entries.push_back(std::move(e));
		}
		RG_LOG("RehearsalBank: loaded " << entries.size() << " entries (" << RowCount() << " rows) from " << path);
	} catch (const std::exception& ex) {
		entries.clear();
		RG_LOG("RehearsalBank: load failed (" << ex.what() << ") - starting empty");
	}
}

void GGL::RehearsalBank::ReportTo(Report& report, const std::string& prefix) const {
	report[prefix + "Entries"] = (float)entries.size();
	report[prefix + "Rows"] = (float)RowCount();
	report[prefix + "KB"] = (float)Bytes() / 1024.f;
	report[prefix + "Admitted"] = (float)admitted;
	report[prefix + "Candidates"] = (float)candidates;
	report[prefix + "Refused Common"] = (float)refusedCommon;
	report[prefix + "Refused Negative"] = (float)refusedNegative;
	report[prefix + "Refused NoConv"] = (float)refusedNoConv;
	report[prefix + "Refused NoEvent"] = (float)refusedNoEvent;
	report[prefix + "Refused Weak"] = (float)refusedWeak;
	report[prefix + "Refused Dup"] = (float)refusedDuplicate;
	report[prefix + "Evicted"] = (float)evicted;
	report[prefix + "Expired"] = (float)expired;
	float hMean = 0, sMean = 0, sMin = 1e30f, tMean = 0, surMean = 0, novMean = 0;
	for (auto& e : entries) { hMean += e.headroom; sMean += e.score; sMin = std::min(sMin, e.score); tMean += e.bestTarget; surMean += e.maxSurprisal; novMean += e.maxNovelty; }
	report[prefix + "Mean Score"] = entries.empty() ? 0.f : sMean / (float)entries.size();
	report[prefix + "Min Score"] = entries.empty() ? 0.f : sMin;
	report[prefix + "Mean Max Surprisal"] = entries.empty() ? 0.f : surMean / (float)entries.size();
	report[prefix + "Mean Max Novelty"] = entries.empty() ? 0.f : novMean / (float)entries.size();
	float hmMean = 0; for (auto& e : entries) hmMean += e.maxHeadroom;
	report[prefix + "Mean Max Headroom"] = entries.empty() ? 0.f : hmMean / (float)entries.size();
	float carriedMean = 0; for (auto& e : entries) carriedMean += (float)e.carried;
	report[prefix + "Mean Carried Rows"] = entries.empty() ? 0.f : carriedMean / (float)entries.size();
	report[prefix + "Carried Rows"] = (float)carriedRows;
	report[prefix + "Carried Segments"] = (float)carriedSegments;
	report[prefix + "Prefix Players"] = (float)prefixes.size();
	float withReset = 0; for (auto& e : entries) withReset += e.resetEdges > 0 ? 1.f : 0.f;
	report[prefix + "Entries With Reset"] = withReset;
	report[prefix + "Mean H"] = entries.empty() ? 0.f : hMean / (float)entries.size();
	report[prefix + "Mean Best Target"] = entries.empty() ? 0.f : tMean / (float)entries.size();
}
