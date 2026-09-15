#include "OpponentMap.h"
#include <cmath>
#include <algorithm>
#include <random>

namespace GGL {

	static constexpr int DY = 6;   // opponent displacement dims (pos 3 + vel 3)

	struct OppMember : torch::nn::Module {
		torch::nn::GRU gru{ nullptr };
		torch::nn::Linear zh{ nullptr };
		torch::nn::Sequential head{ nullptr };
		OppMember(int obsSize, int numActions, int hidden, int zdim) {
			gru = register_module("gru", torch::nn::GRU(torch::nn::GRUOptions(obsSize + DY, hidden).batch_first(true)));
			zh = register_module("zh", torch::nn::Linear(hidden, zdim));
			head = register_module("head", torch::nn::Sequential(
				torch::nn::Linear(obsSize + numActions + zdim, hidden), torch::nn::Tanh(),
				torch::nn::Linear(hidden, hidden), torch::nn::Tanh(),
				torch::nn::Linear(hidden, 2 * DY)));
		}
		// x [B,T,obs+DY] -> z [B,T,zdim]
		torch::Tensor Encode(torch::Tensor x) {
			auto out = std::get<0>(gru->forward(x));
			return zh->forward(out);
		}
		// obs [.., obs], actOneHot [.., numActions], z [.., zdim] -> (mean [.., DY], logvar [.., DY])
		std::pair<torch::Tensor, torch::Tensor> Predict(torch::Tensor obs, torch::Tensor actOneHot, torch::Tensor z) {
			auto o = head->forward(torch::cat({ obs, actOneHot, z }, -1));
			auto mean = o.narrow(-1, 0, DY);
			auto logvar = o.narrow(-1, DY, DY).clamp(-6.f, 4.f);
			return { mean, logvar };
		}
	};

	struct OppOutcome : torch::nn::Module {
		torch::nn::Sequential net{ nullptr };
		torch::nn::Embedding ident{ nullptr };
		int intentDim; bool useContext;
		OppOutcome(int obsSize, int intentDim, int zdim, int hidden, int identitySlots, bool useContext)
			: intentDim(intentDim), useContext(useContext) {
			ident = register_module("ident", torch::nn::Embedding(identitySlots, zdim));
			net = register_module("net", torch::nn::Sequential(
				torch::nn::Linear(obsSize + RS_MAX(intentDim, 1) + 2 * zdim, hidden), torch::nn::Tanh(),
				torch::nn::Linear(hidden, hidden), torch::nn::Tanh(),
				torch::nn::Linear(hidden, 1)));
		}
		// obs [.., obs], intent onehot [.., max(intentDim,1)], z [.., zdim], oppId [..] long -> [..]
		// context = inferred z ++ identity embedding; the ablation zeroes both
		torch::Tensor Forward(torch::Tensor obs, torch::Tensor intentOneHot, torch::Tensor z, torch::Tensor oppId) {
			auto e = ident->forward(oppId);
			if (!useContext) { z = torch::zeros_like(z); e = torch::zeros_like(e); }
			return net->forward(torch::cat({ obs, intentOneHot, z, e }, -1)).squeeze(-1);
		}
	};

	OpponentMap::OpponentMap(const OppMapConfig& cfg, int obsSize, int intentDim, int numActions, torch::Device device)
		: cfg(cfg), obsSize(obsSize), intentDim(intentDim), numActions(numActions), device(device) {
		for (int e = 0; e < RS_MAX(1, cfg.ensemble); e++) {
			auto m = std::make_shared<OppMember>(obsSize, numActions, cfg.hidden, cfg.zdim);
			m->to(device);
			members.push_back(m);
			memberOpt.push_back(std::make_shared<torch::optim::Adam>(m->parameters(), cfg.lr));
		}
		outcome = std::make_shared<OppOutcome>(obsSize, intentDim, cfg.zdim, cfg.hidden, cfg.identitySlots, cfg.useContext);
		outcome->to(device);
		outcomeOpt = std::make_shared<torch::optim::Adam>(outcome->parameters(), cfg.lr);
		planHead = std::make_shared<OppOutcome>(obsSize, intentDim, cfg.zdim, cfg.hidden, cfg.identitySlots, cfg.useContext);
		planHead->to(device);
		tgtStd = torch::ones({ DY }, torch::TensorOptions().device(device));
		RG_LOG("OpponentMap: " << members.size() << " members, zdim " << cfg.zdim << ", hidden " << cfg.hidden
			<< ", planFrac " << cfg.planFrac << ", warmup " << cfg.planWarmup << " (intentDim " << intentDim << ", context " << (cfg.useContext ? "on" : "OFF") << ")");
	}

	torch::Tensor OpponentMap::OppFeat(torch::Tensor states) const {
		// states [..., obs] -> [..., DY]: pos+vel of the present opponent slot (1v1: exactly one).
		auto flags = states.narrow(-1, cfg.presenceBase, cfg.oppSlots);              // [..., slots]
		auto slot = flags.argmax(-1, true);                                           // [..., 1]
		auto base = slot * cfg.playerStride + cfg.oppSlotBase;                         // [..., 1]
		auto idxPos = base + torch::arange(3, torch::TensorOptions().dtype(torch::kLong).device(states.device()));
		auto idxVel = base + cfg.velOff + torch::arange(3, torch::TensorOptions().dtype(torch::kLong).device(states.device()));
		auto idx = torch::cat({ idxPos, idxVel }, -1);                                // [..., 6]
		return states.gather(-1, idx);
	}

	OppMapLearnResult OpponentMap::Learn(torch::Tensor states, torch::Tensor targetVals, torch::Tensor intentIds,
		torch::Tensor actions, const std::vector<int32_t>& playerIds, const std::vector<int8_t>& terminals,
		const std::vector<int32_t>& oppIds, const std::vector<uint8_t>& boundary) {
		OppMapLearnResult res;
		torch::AutoGradMode gradOn(true);   // learn-prep runs under RG_NO_GRAD; the training passes below need autograd
		const int64_t N = states.size(0);
		if (N < 2 || (int64_t)playerIds.size() != N || (int64_t)terminals.size() != N || (int64_t)oppIds.size() != N)
			return res;
		torch::Tensor oppIdT = torch::from_blob((void*)oppIds.data(), { N }, torch::kInt32).to(torch::kLong).clamp(0, cfg.identitySlots - 1);
		torch::Tensor bndT = boundary.size() == (size_t)N
			? torch::from_blob((void*)boundary.data(), { N }, torch::kUInt8).to(torch::kFloat32)
			: torch::ones({ N });
		// ---- segments: contiguous rows of one player, cut at terminals
		struct Seg { int64_t start, len; int player; };
		std::vector<Seg> segs;
		int64_t s0 = 0;
		for (int64_t i = 1; i <= N; i++) {
			bool cut = (i == N) || playerIds[i] != playerIds[s0] || terminals[i - 1] != 0;
			if (cut) {
				if (i - s0 >= 2) segs.push_back({ s0, i - s0, (int)playerIds[s0] });
				s0 = i;
			}
		}
		if (segs.empty()) return res;
		// subsample segments for training; keep ALL for the z table (cheap: one forward each)
		std::mt19937 rng((unsigned)(updates * 7919 + 13));
		std::vector<int> order(segs.size());
		for (size_t i = 0; i < order.size(); i++) order[i] = (int)i;
		std::shuffle(order.begin(), order.end(), rng);
		const int B = (int)RS_MIN((int64_t)cfg.trainEpisodes, (int64_t)segs.size());
		// pad each chunk only to its longest segment (a rank's iteration holds ~48 rows per
		// player; padding everything to maxLen cost 2.5x throughput on the first hops)
		int64_t longest = 2;
		for (auto& sg : segs) longest = RS_MAX(longest, RS_MIN(sg.len, (int64_t)cfg.maxLen));
		const int T = (int)longest;
		// ---- build padded batches (chunks of 128 segments)
		auto build = [&](const std::vector<int>& ids, torch::Tensor& X, torch::Tensor& Y, torch::Tensor& M,
			torch::Tensor& OBS, torch::Tensor& VAL, torch::Tensor& INT, torch::Tensor& ACT, torch::Tensor& OPP, torch::Tensor& BND,
			std::vector<int>& lastT) {
			const int b = (int)ids.size();
			OBS = torch::zeros({ b, T, (int64_t)obsSize });
			VAL = torch::zeros({ b, T });
			INT = torch::zeros({ b, T }, torch::kLong);
			ACT = torch::zeros({ b, T }, torch::kLong);
			OPP = torch::zeros({ b, T }, torch::kLong);
			BND = torch::zeros({ b, T });
			M = torch::zeros({ b, T });
			lastT.assign(b, 0);
			for (int k = 0; k < b; k++) {
				const Seg& sg = segs[(size_t)ids[k]];
				int64_t st = sg.start, ln = sg.len;
				if (ln > T) { st += ln - T; ln = T; }
				OBS[k].narrow(0, 0, ln).copy_(states.narrow(0, st, ln));
				VAL[k].narrow(0, 0, ln).copy_(targetVals.narrow(0, st, ln));
				if (intentIds.defined()) INT[k].narrow(0, 0, ln).copy_(intentIds.narrow(0, st, ln));
				ACT[k].narrow(0, 0, ln).copy_(actions.narrow(0, st, ln));
				OPP[k].narrow(0, 0, ln).copy_(oppIdT.narrow(0, st, ln));
				BND[k].narrow(0, 0, ln).copy_(bndT.narrow(0, st, ln));
				M[k].narrow(0, 0, ln - 1).fill_(1.f);   // target valid where t+1 is in-segment
				lastT[k] = (int)ln - 1;
			}
			auto feat = OppFeat(OBS);                                   // [b,T,6]
			auto dy = feat.narrow(1, 1, T - 1) - feat.narrow(1, 0, T - 1);
			Y = torch::cat({ dy, torch::zeros({ b, 1, DY }) }, 1);      // y_t = feat_{t+1} - feat_t
			auto yPrev = torch::cat({ torch::zeros({ b, 1, DY }), Y.narrow(1, 0, T - 1) }, 1);
			X = torch::cat({ OBS, yPrev }, -1);                          // encoder input: obs_t ++ y_{t-1}
		};
		// running target std (from the first chunk, EMA after)
		std::vector<float> excessAll; double nllSum = 0, entSum = 0, disSum = 0, outSum = 0; int64_t rowsScored = 0; int outN = 0;
		double zNormSum = 0; int zN = 0;
		res.stale = torch::zeros({ N });
		float* staleP = res.stale.data_ptr<float>();
		const int chunk = 128;
		// DISTRIBUTED: every chunk issues one gradient all-reduce per member (+1 for the outcome
		// head), so all ranks must run the SAME number of chunks or the collectives deadlock
		// (the map arms hung at iteration 1 on 2026-09-12 before this). Agree on the minimum.
		int nChunks = (B + chunk - 1) / chunk;
		if (dist && dist->distributed())
			dist->min_host(&nChunks, 1);
		if (nChunks <= 0) return res;
		for (int ci = 0; ci < nChunks; ci++) {
			const int c0 = ci * chunk;
			std::vector<int> ids;
			for (int i = c0; i < RS_MIN(c0 + chunk, B); i++) ids.push_back(order[(size_t)i]);
			if (ids.empty()) ids.push_back(order[0]);   // cannot happen after min_host, defensive
			const bool train = true;
			torch::Tensor X, Y, M, OBS, VAL, INT, ACT, OPP, BND; std::vector<int> lastT;
			build(ids, X, Y, M, OBS, VAL, INT, ACT, OPP, BND, lastT);
			auto Xd = X.to(device), Yd = Y.to(device), Md = M.to(device), OBSd = OBS.to(device);
			auto ACTd = torch::nn::functional::one_hot(ACT.to(device).clamp(0, numActions - 1), numActions).to(torch::kFloat32);
			auto OPPd = OPP.to(device);
			{
				torch::NoGradGuard ng;
				auto ystd = (Yd * Md.unsqueeze(-1)).pow(2).sum(0).sum(0).div(Md.sum().clamp_min(1.f)).sqrt().clamp_min(1e-4f);
				if (!statsInit) { tgtStd = ystd; statsInit = true; }
				else tgtStd = tgtStd * 0.98f + ystd * 0.02f;
			}
			auto Yn = Yd / tgtStd;
			auto Xn = torch::cat({ Xd.narrow(-1, 0, obsSize), Xd.narrow(-1, obsSize, DY) / tgtStd }, -1);
			// ---- prequential scoring (before the update) on the first member set
			std::vector<torch::Tensor> means;
			torch::Tensor nllRow, entRow, z0;
			{
				torch::NoGradGuard ng;
				std::vector<torch::Tensor> nlls, ents, vars;
				for (size_t e = 0; e < members.size(); e++) {
					auto z = members[e]->Encode(Xn);
					auto [mean, logvar] = members[e]->Predict(OBSd, ACTd, z);
					auto nll = 0.5f * ((Yn - mean).pow(2) * torch::exp(-logvar) + logvar + std::log(2 * M_PI)).sum(-1);
					auto ent = 0.5f * (logvar + 1.f + std::log(2 * M_PI)).sum(-1);
					nlls.push_back(nll); ents.push_back(ent); means.push_back(mean); vars.push_back(torch::exp(logvar));
					if (e == 0) z0 = z;
				}
				nllRow = torch::stack(nlls, 0).mean(0); entRow = torch::stack(ents, 0).mean(0);
				auto ex = (nllRow - entRow).masked_select(Md > 0.5f);
				auto exC = ex.cpu();
				const float* p = exC.data_ptr<float>();
				for (int64_t i = 0; i < exC.numel(); i++) excessAll.push_back(p[i]);
				nllSum += nllRow.masked_select(Md > 0.5f).sum().item<double>();
				entSum += entRow.masked_select(Md > 0.5f).sum().item<double>();
				rowsScored += exC.numel();
				if (means.size() > 1) {
					auto st = torch::stack(means, 0);                          // [E,b,T,6]
					auto pv = torch::stack(vars, 0).mean(0).clamp_min(1e-4f);  // predicted variance (aleatoric)
					auto dis = (st.var(0, false) / pv).mean(-1);               // [b,T] normalised disagreement
					disSum += dis.masked_select(Md > 0.5f).sum().item<double>();
					auto disC = (dis * Md).cpu();
					for (size_t k = 0; k < ids.size(); k++) {
						const Seg& sg = segs[(size_t)ids[k]];
						int64_t st0 = sg.start, ln = sg.len;
						if (ln > T) { st0 += ln - T; ln = T; }
						const float* row = disC[(int64_t)k].data_ptr<float>();
						for (int64_t tt = 0; tt < ln; tt++) staleP[st0 + tt] = row[tt];
					}
				}
				// z table: final z per segment (member 0)
				auto zc = z0.cpu();
				for (size_t k = 0; k < ids.size(); k++) {
					auto row = zc[(int64_t)k][lastT[k]];
					std::vector<float> v(row.data_ptr<float>(), row.data_ptr<float>() + cfg.zdim);
					zPending[segs[(size_t)ids[k]].player] = v;
					zNormSum += row.norm().item<double>(); zN++;
				}
			}
			if (!train) continue;
			// ---- training pass: every member on this chunk
			for (size_t e = 0; e < members.size(); e++) {
				auto z = members[e]->Encode(Xn);
				auto [mean, logvar] = members[e]->Predict(OBSd, ACTd, z);
				auto nll = 0.5f * ((Yn - mean).pow(2) * torch::exp(-logvar) + logvar).sum(-1);
				auto loss = (nll * Md).sum() / Md.sum().clamp_min(1.f);
				memberOpt[e]->zero_grad();
				loss.backward();
				if (dist && dist->distributed()) {
					std::vector<Dist::Session::GradRef> refs;
					for (auto& p : members[e]->parameters()) {
						if (!p.grad().defined()) p.mutable_grad() = torch::zeros_like(p);
						auto g = p.grad();
						if (g.is_cuda() && g.scalar_type() == torch::kFloat) refs.push_back({ g.data_ptr<float>(), (size_t)g.numel() });
					}
					dist->allreduce_avg_grads(refs);
				}
				memberOpt[e]->step();
			}
			// ---- outcome head on all rows of the chunk with member-0 z (detached)
			{
				auto zd = z0.detach();
				torch::Tensor oneHot;
				if (intentDim > 0)
					oneHot = torch::nn::functional::one_hot(INT.to(device).clamp(0, intentDim - 1), intentDim).to(torch::kFloat32);
				else
					oneHot = torch::zeros({ (int64_t)ids.size(), T, 1 }, torch::TensorOptions().device(device));
				// outcome rows = intent-interval STARTS inside the segment ("execute this intent for one
				// interval, then continue"); Md marks in-segment rows (t+1 in-segment) - add the last row back
				auto inSeg = Md.clone();
				for (size_t k = 0; k < ids.size(); k++) inSeg[(int64_t)k][lastT[k]] = 1.f;
				auto valid = inSeg * BND.to(device);
				auto pred = outcome->Forward(OBSd, oneHot, zd, OPPd);
				auto tgt = VAL.to(device);
				auto loss = ((pred - tgt).pow(2) * valid).sum() / valid.sum().clamp_min(1.f);
				outcomeOpt->zero_grad();
				loss.backward();
				if (dist && dist->distributed()) {
					std::vector<Dist::Session::GradRef> refs;
					for (auto& p : outcome->parameters()) {
						if (!p.grad().defined()) p.mutable_grad() = torch::zeros_like(p);
						auto g = p.grad();
						if (g.is_cuda() && g.scalar_type() == torch::kFloat) refs.push_back({ g.data_ptr<float>(), (size_t)g.numel() });
					}
					dist->allreduce_avg_grads(refs);
				}
				outcomeOpt->step();
				outSum += loss.item<double>(); outN++;
			}
		}
		updates++;
		res.segments = (int)segs.size(); res.rows = (int)rowsScored;
		if (rowsScored > 0) {
			res.nll = (float)(nllSum / rowsScored); res.entropy = (float)(entSum / rowsScored);
			res.excess = res.nll - res.entropy;
			std::vector<float> ex = excessAll;
			std::nth_element(ex.begin(), ex.begin() + (ptrdiff_t)(ex.size() * 9 / 10), ex.end());
			res.excessP90 = ex[ex.size() * 9 / 10];
			res.disagree = (float)(disSum / rowsScored);
		}
		if (outN) res.outcomeLoss = (float)(outSum / outN);
		if (zN) res.zNorm = (float)(zNormSum / zN);
		return res;
	}

	void OpponentMap::SyncCollectSnapshot() {
		// barrier zone: publish z table + frozen outcome head
		for (auto& kv : zPending) zLive[kv.first] = kv.second;
		torch::NoGradGuard ng;
		auto src = outcome->parameters(); auto dst = planHead->parameters();
		for (size_t i = 0; i < src.size() && i < dst.size(); i++) dst[i].copy_(src[i]);
		planReady = intentDim > 0 && updates >= (int64_t)cfg.planWarmup && !zLive.empty();
	}

	std::vector<int> OpponentMap::Plan(const float* obsRows, const std::vector<int>& players, int M, int oppId) {
		std::vector<int> out;
		if (!planReady || M <= 0) return out;
		torch::NoGradGuard ng;
		auto obs = torch::from_blob((void*)obsRows, { (int64_t)M, (int64_t)obsSize }, torch::kFloat32).to(device);
		torch::Tensor z = torch::zeros({ (int64_t)M, (int64_t)cfg.zdim });
		for (int k = 0; k < M; k++) {
			auto it = zLive.find(players[(size_t)k]);
			if (it != zLive.end())
				std::copy(it->second.begin(), it->second.end(), z[k].data_ptr<float>());
		}
		auto zd = z.to(device);
		// value of every intent: [M, intentDim]
		auto obsR = obs.unsqueeze(1).expand({ (int64_t)M, (int64_t)intentDim, (int64_t)obsSize });
		auto zR = zd.unsqueeze(1).expand({ (int64_t)M, (int64_t)intentDim, (int64_t)cfg.zdim });
		auto eye = torch::eye(intentDim, torch::TensorOptions().device(device)).unsqueeze(0).expand({ (int64_t)M, (int64_t)intentDim, (int64_t)intentDim });
		auto idR = torch::full({ (int64_t)M, (int64_t)intentDim }, (int64_t)RS_CLAMP(oppId, 0, cfg.identitySlots - 1), torch::TensorOptions().dtype(torch::kLong).device(device));
		auto v = planHead->Forward(obsR, eye, zR, idR);                 // [M, intentDim]
		auto best = v.argmax(1).to(torch::kCPU);
		out.assign(best.data_ptr<int64_t>(), best.data_ptr<int64_t>() + M);
		return out;
	}

	void OpponentMap::Save(const std::filesystem::path& folder) const {
		for (size_t e = 0; e < members.size(); e++)
			torch::save(members[e], (folder / RS_STR("OPPMAP_M" << e << ".lt")).string());
		torch::save(outcome, (folder / "OPPMAP_HEAD.lt").string());
		std::vector<torch::Tensor> st = { tgtStd.cpu() };
		torch::save(st, (folder / "OPPMAP_STATS.lt").string());
	}

	void OpponentMap::Load(const std::filesystem::path& folder) {
		if (!std::filesystem::exists(folder / "OPPMAP_M0.lt")) return;
		try {
			for (size_t e = 0; e < members.size(); e++) {
				auto p = folder / RS_STR("OPPMAP_M" << e << ".lt");
				if (std::filesystem::exists(p)) torch::load(members[e], p.string());
			}
			torch::load(outcome, (folder / "OPPMAP_HEAD.lt").string());
			std::vector<torch::Tensor> st;
			torch::load(st, (folder / "OPPMAP_STATS.lt").string());
			if (!st.empty()) { tgtStd = st[0].to(device); statsInit = true; }
			updates = cfg.planWarmup;   // a loaded map is trusted immediately
			for (auto& m : members) m->to(device);
			outcome->to(device);
			RG_LOG("OpponentMap loaded from " << folder);
		} catch (const std::exception& e) {
			RG_LOG("OpponentMap load failed (" << e.what() << ") - starting fresh");
		}
	}

	void OpponentMap::BroadcastParameters() {
		if (!dist || !dist->distributed()) return;
		torch::NoGradGuard ng;
		auto bc = [&](std::vector<torch::Tensor> ps) {
			for (auto& p : ps) {
				if (p.is_cuda() && p.scalar_type() == torch::kFloat) {
					auto t = p.contiguous();
					dist->bcast_device(t.data_ptr<float>(), (size_t)t.numel(), 0);
					if (!p.is_same(t)) p.copy_(t);
				}
			}
		};
		for (auto& m : members) bc(m->parameters());
		bc(outcome->parameters());
	}
}
