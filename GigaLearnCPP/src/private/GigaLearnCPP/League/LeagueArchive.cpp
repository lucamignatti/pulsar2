#include "LeagueArchive.h"
#include "Operators.h"
#include "../PPO/PPOLearner.h"

#include <torch/nn/utils/convert_parameters.h>
#include <torch/csrc/api/include/torch/serialize.h>
#include <RLGymCPP/StateSetters/FuzzedKickoffState.h>
#include <RLGymCPP/TerminalConditions/GoalScoreCondition.h>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <set>

using namespace GGL;
using namespace torch;

static const char* SCRATCH_MODELS[] = { "shared_head", "policy" };

LeagueArchive::LeagueArchive(const LeagueConfig& cfg, PPOLearner* ppo, RLGC::EnvSet* trainEnv,
	torch::Device device, std::filesystem::path checkpointFolder)
	: cfg(cfg), device(device), ppo(ppo) {

	// Isolated match arenas (never touches training arenas), goal-terminated like the skill tracker.
	RLGC::EnvSetConfig mc = trainEnv->config;
	mc.numArenas = 16;
	matchEnv = new RLGC::EnvSet(mc);
	for (int i = 0; i < (int)matchEnv->arenas.size(); i++) {
		matchEnv->stateSetters[i] = { new RLGC::FuzzedKickoffState() };
		matchEnv->terminalConditions[i] = { new RLGC::GoalScoreCondition() };
	}

	// Two model sets: `scratch` for evaluation, `oppServe` for the training loop to borrow as its
	// opponent (kept separate so serving an opponent during collection can't race an evolve eval).
	for (const char* name : SCRATCH_MODELS)
		if (ppo->models[name]) {
			scratch.Add(ppo->models[name]->MakeClone());
			oppServe.Add(ppo->models[name]->MakeClone());
		}

	if (!checkpointFolder.empty())
		leagueDir = checkpointFolder / "league";
}

LeagueArchive::~LeagueArchive() {
	delete matchEnv;
	scratch.Free();
	oppServe.Free();
}

std::vector<torch::Tensor> LeagueArchive::SnapshotMain() const {
	RG_NO_GRAD;
	std::vector<torch::Tensor> out;
	for (const char* name : SCRATCH_MODELS)
		if (ppo->models[name])
			out.push_back(nn::utils::parameters_to_vector(ppo->models[name]->parameters()).detach().cpu().clone());
	return out;
}

// Ladder wire migration for archived flat vectors: elites stored before the
// policy head gained its wire columns come up short by exactly out x expand
// elements of the FIRST Linear's weight. Zero-pad those columns row-wise (flat
// layout is row-major: weight rows first) - behaviorally exact, same rule as
// Model::Load. Returns the input unchanged when shapes already match (or don't
// match the migration signature). MUST run at STORAGE time (FromJSON), not just
// at LoadInto: EvolveStep's crossover/mutation mixes stored member vectors with
// fresh SnapshotMain() vectors arithmetically - a stale-shape member crashes the
// add (live incident, 2026-07-18: 839770 vs 837210 = 512*5 wire columns).
static torch::Tensor MigrateFlatVec(GGL::Model* mdl, torch::Tensor p) {
	auto cur = mdl->parameters();
	int64_t curTotal = 0;
	for (auto& c : cur)
		curTotal += c.numel();
	if (p.numel() == curTotal || mdl->allowInputExpand <= 0
		|| cur.empty() || cur[0].dim() != 2)
		return p;
	int64_t out = cur[0].size(0), inNew = cur[0].size(1);
	int64_t inOld = inNew - mdl->allowInputExpand;
	if (inOld <= 0 || p.numel() != curTotal - out * mdl->allowInputExpand)
		return p;
	auto w = p.slice(0, 0, out * inOld).view({ out, inOld });
	auto wPad = torch::cat({ w,
		torch::zeros({ out, (int64_t)mdl->allowInputExpand }, w.options()) }, 1)
		.contiguous().flatten();
	static bool loggedOnce = false;
	if (!loggedOnce) {
		loggedOnce = true;
		RG_LOG("League: migrating pre-wire member vectors ("
			<< inOld << " -> " << inNew << " policy inputs, zero-pad; "
			"logged once, applies to every archived elite)");
	}
	return torch::cat({ wPad, p.slice(0, out * inOld) });
}

void LeagueArchive::LoadInto(ModelSet& set, const std::vector<torch::Tensor>& params) {
	RG_NO_GRAD;
	int i = 0;
	for (const char* name : SCRATCH_MODELS) {
		if (!set[name]) continue;
		Model* mdl = set[name];
		torch::Tensor p = MigrateFlatVec(mdl, params[i]); // defensive; storage migrates at FromJSON
		nn::utils::vector_to_parameters(p.to(device), mdl->parameters());
		mdl->_seqHalfOutdated = true;
		i++;
	}
}

long LeagueArchive::CellIndex(const std::vector<float>& bd) const {
	long idx = 0;
	for (size_t a = 0; a < bd.size(); a++) {
		int b;
		if (a < binEdges.size() && !binEdges[a].empty()) {
			// Quantile bins: bin = number of edges <= v. Edges are ascending (possibly tied when an
			// axis has near-zero variance; ties just merge those bins, which is honest).
			b = (int)(std::upper_bound(binEdges[a].begin(), binEdges[a].end(), bd[a]) - binEdges[a].begin());
		} else {
			// Uniform [0,1] fallback (quantileBins off, or edges not bootstrapped yet).
			b = std::clamp((int)(bd[a] * cfg.binsPerAxis), 0, cfg.binsPerAxis - 1);
		}
		idx = idx * cfg.binsPerAxis + b;
	}
	return idx;
}

void LeagueArchive::RecordBDSample(const std::vector<float>& bd) {
	if ((int)bdSamples.size() < cfg.quantileSampleCap) {
		bdSamples.push_back(bd);
	} else {
		bdSamples[bdSampleCursor] = bd;
		bdSampleCursor = (bdSampleCursor + 1) % cfg.quantileSampleCap;
	}
}

void LeagueArchive::RefreshBinEdges() {
	int axes = (int)cfg.gridAxes.size();
	if ((int)bdSamples.size() < cfg.quantileMinSamples) return;
	binEdges.assign(axes, {});
	for (int a = 0; a < axes; a++) {
		std::vector<float> v;
		v.reserve(bdSamples.size());
		for (auto& s : bdSamples)
			if (a < (int)s.size()) v.push_back(s[a]);
		if (v.empty()) continue;
		std::sort(v.begin(), v.end());
		for (int k = 1; k < cfg.binsPerAxis; k++) {
			size_t i = std::min((size_t)((double)k / cfg.binsPerAxis * v.size()), v.size() - 1);
			binEdges[a].push_back(v[i]);
		}
	}
	// The coordinate system moved: re-tokenize every member under the new edges, then restore the
	// one-elite-per-cell invariant.
	for (auto& m : members)
		if (!m.exploiter) m.cell = (int)CellIndex(m.bd);
	DedupCells();
}

void LeagueArchive::DedupCells() {
	// One elite per cell. RefreshStalest and bin-edge refreshes reassign EXISTING members' cells,
	// which (unlike TryInsert) can stack several members in one cell; drop the weaker duplicates.
	std::map<long, int> best;
	for (int i = 0; i < (int)members.size(); i++) {
		if (members[i].exploiter || members[i].cell < 0) continue;
		auto it = best.find(members[i].cell);
		if (it == best.end() || members[i].fitness > members[it->second].fitness)
			best[members[i].cell] = i;
	}
	std::vector<Member> kept;
	kept.reserve(members.size());
	for (int i = 0; i < (int)members.size(); i++)
		if (members[i].exploiter || members[i].cell < 0 || best[members[i].cell] == i)
			kept.push_back(std::move(members[i]));
	members = std::move(kept);
	RebuildCellMap();
}

float LeagueArchive::EvaluateMember(const std::vector<torch::Tensor>& memberParams, std::vector<float>& outBD) {
	RG_NO_GRAD;
	LoadInto(scratch, memberParams);
	matchEnv->Reset();

	// Fixed team split (member = team A, main = team B) computed from the initial state.
	std::vector<int> aPlayers, bPlayers;
	Team aTeam = Team::BLUE;
	for (int i = 0; i < (int)matchEnv->arenas.size(); i++) {
		auto& st = matchEnv->state.gameStates[i];
		for (int j = 0; j < (int)st.players.size(); j++) {
			int pIdx = matchEnv->state.arenaPlayerStartIdx[i] + j;
			(st.players[j].team == aTeam ? aPlayers : bPlayers).push_back(pIdx);
		}
	}
	Tensor tA = torch::tensor(aPlayers), tB = torch::tensor(bPlayers);

	double inAir = 0, fieldY = 0, boost = 0;
	long samples = 0;
	int aGoals = 0, bGoals = 0;
	const int matchSteps = 300;
	for (int step = 0; step < matchSteps; step++) {
		matchEnv->Reset();
		Tensor obs = DIMLIST2_TO_TENSOR<float>(matchEnv->state.obs);
		Tensor masks = DIMLIST2_TO_TENSOR<uint8_t>(matchEnv->state.actionMasks);

		matchEnv->StepFirstHalf(true);

		Tensor aAct, bAct, lp;
		PPOLearner::InferActionsFromModels(scratch,
			obs.index_select(0, tA).to(device, true), masks.index_select(0, tA).to(device, true),
			false, ppo->config.policyTemperature, ppo->config.useHalfPrecision, &aAct, &lp);
		PPOLearner::InferActionsFromModels(ppo->models,
			obs.index_select(0, tB).to(device, true), masks.index_select(0, tB).to(device, true),
			false, ppo->config.policyTemperature, ppo->config.useHalfPrecision, &bAct, &lp);

		auto av = TENSOR_TO_VEC<int>(aAct), bv = TENSOR_TO_VEC<int>(bAct);
		std::vector<int> actions(matchEnv->state.numPlayers, 0);
		for (int i = 0; i < (int)aPlayers.size(); i++) actions[aPlayers[i]] = av[i];
		for (int i = 0; i < (int)bPlayers.size(); i++) actions[bPlayers[i]] = bv[i];

		matchEnv->Sync();
		matchEnv->StepSecondHalf(actions, false);

		// Behavior descriptors over team-A cars.
		for (int i = 0; i < (int)matchEnv->arenas.size(); i++) {
			auto& st = matchEnv->state.gameStates[i];
			for (auto& p : st.players) {
				if (p.team != aTeam) continue;
				inAir += p.isOnGround ? 0.0 : 1.0;
				fieldY += std::min(1.0, std::abs((double)p.pos.y) / 5120.0);
				boost += std::clamp((double)p.boost / 100.0, 0.0, 1.0);
				samples++;
			}
			if (st.goalScored) {
				// RS_TEAM_FROM_Y returns the team whose NET the ball is in = the team that got
				// scored ON. The SCORER is the opposite team (matches GoalReward's convention).
				if (RS_TEAM_FROM_Y(st.ball.pos.y) != aTeam) aGoals++; else bGoals++;
			}
		}
	}

	outBD = {
		(float)(inAir / std::max(1L, samples)),
		(float)(fieldY / std::max(1L, samples)),
		(float)(boost / std::max(1L, samples))
	};
	// Trim/pad BD to the configured axis count.
	outBD.resize(cfg.gridAxes.size(), 0.0f);
	if (cfg.quantileBins)
		RecordBDSample(outBD); // every eval feeds the quantile-edge window
	return (float)(aGoals - bGoals); // > 0 => member BEATS the current main
}

void LeagueArchive::RebuildCellMap() {
	cellToMember.clear();
	for (int i = 0; i < (int)members.size(); i++) {
		if (members[i].exploiter || members[i].cell < 0) continue;
		auto it = cellToMember.find(members[i].cell);
		if (it == cellToMember.end() || members[i].fitness > members[it->second].fitness)
			cellToMember[members[i].cell] = i;
	}
}

void LeagueArchive::TryInsert(Member&& m) {
	if (m.fitness < cfg.competenceFloor) return;

	m.cell = (int)CellIndex(m.bd);
	auto it = cellToMember.find(m.cell);
	if (it == cellToMember.end()) {
		members.push_back(std::move(m));
		cellToMember[members.back().cell] = (int)members.size() - 1;
	} else if (m.fitness > members[it->second].fitness) {
		members[it->second] = std::move(m); // within-cell replacement (style-preserving elitism)
	}
}

int LeagueArchive::SampleOpponent() const {
	if (members.empty()) return -1;
	// PFSP: prefer members near the frontier (even matches), softmax over -|fitness| / temp.
	std::vector<double> logits(members.size());
	double mx = -1e30;
	for (size_t i = 0; i < members.size(); i++) {
		logits[i] = -std::abs((double)members[i].fitness) / std::max(1e-3, (double)cfg.pfspTemp);
		mx = std::max(mx, logits[i]);
	}
	double sum = 0; for (double& l : logits) { l = std::exp(l - mx); sum += l; }
	double r = ((double)Math::RandInt(0, 100000) / 100000.0) * sum;
	double acc = 0;
	for (size_t i = 0; i < members.size(); i++) { acc += logits[i]; if (r <= acc) return (int)i; }
	return (int)members.size() - 1;
}

ModelSet* LeagueArchive::LoadPFSPOpponentModels() {
	if (members.empty()) return nullptr;
	int idx = SampleOpponent();
	if (idx < 0) return nullptr;
	LoadInto(oppServe, members[idx].params);
	return &oppServe;
}

void LeagueArchive::ReseedFromMain() {
	RG_NO_GRAD;
	std::vector<torch::Tensor> mainW = SnapshotMain();
	std::vector<float> bd;
	float q = EvaluateMember(mainW, bd); // ~0 vs itself; drifts negative as the main improves
	Member seed;
	seed.params = mainW; seed.bd = bd; seed.fitness = q; seed.matches = 1;
	seed.lineage = nextLineage++;
	TryInsert(std::move(seed));
}

void LeagueArchive::EvolveExploiters() {
	RG_NO_GRAD;
	if (cfg.exploiterSlots <= 0 || members.empty()) return;

	int have = 0;
	for (auto& m : members) if (m.exploiter) have++;

	// Fill empty exploiter slots by mutating the strongest current member.
	while (have < cfg.exploiterSlots) {
		int best = 0;
		for (int i = 1; i < (int)members.size(); i++)
			if (members[i].fitness > members[best].fitness) best = i;
		std::vector<torch::Tensor> childW;
		for (auto& t : members[best].params)
			childW.push_back(League::MutateGaussian(t, cfg.mutationSigma));
		std::vector<float> bd;
		float q = EvaluateMember(childW, bd);
		Member e; e.params = childW; e.bd = bd; e.fitness = q; e.matches = 1;
		e.exploiter = true; e.cell = -1; e.lineage = members[best].lineage;
		members.push_back(std::move(e));
		have++;
	}

	// Hill-climb one exploiter this step: mutate it, keep the mutation only if it beats the main by
	// more. Exploiters chase pure fitness-vs-current-main, so they keep pressure on your weaknesses.
	std::vector<int> expIdx;
	for (int i = 0; i < (int)members.size(); i++) if (members[i].exploiter) expIdx.push_back(i);
	if (expIdx.empty()) return;
	int pick = expIdx[Math::RandInt(0, (int)expIdx.size())];
	std::vector<torch::Tensor> mutW;
	for (auto& t : members[pick].params)
		mutW.push_back(League::MutateGaussian(t, cfg.mutationSigma));
	std::vector<float> bd;
	float q = EvaluateMember(mutW, bd);
	if (q > members[pick].fitness) {
		members[pick].params = std::move(mutW);
		members[pick].bd = bd;
		members[pick].fitness = q;
	}
	members[pick].age = 0;
}

void LeagueArchive::RefreshStalest() {
	RG_NO_GRAD;
	// Round-robin re-evaluation of non-exploiter members: as the MAIN improves, an old member's
	// fitness (and its style relative to the new main) drifts, so a stale archive lies. Re-scoring
	// keeps PFSP honest and lets a member that has fallen below the floor get culled.
	if (members.empty()) return;
	int n = (int)members.size();
	for (int tries = 0; tries < n; tries++) {
		int i = refreshCursor % n;
		refreshCursor = (refreshCursor + 1) % n;
		if (members[i].exploiter) continue; // exploiters refresh themselves in EvolveExploiters
		std::vector<float> bd;
		float q = EvaluateMember(members[i].params, bd);
		members[i].fitness = q;
		members[i].bd = bd;
		members[i].cell = (int)CellIndex(bd);
		members[i].age = 0;
		return;
	}
}

void LeagueArchive::Cull() {
	// Cap total members. Never drop an exploiter or a member that is the sole elite of its cell;
	// among the rest, drop the lowest fitness first. Also drop anyone below the competence floor.
	DedupCells(); // RefreshStalest may have drifted a member into an occupied cell
	RebuildCellMap();
	std::set<int> protectedIdx;
	for (auto& kv : cellToMember) protectedIdx.insert(kv.second);

	// Below-floor removal (stale refresh may have pushed a member under the floor).
	for (int i = (int)members.size() - 1; i >= 0; i--)
		if (!members[i].exploiter && !protectedIdx.count(i) && members[i].fitness < cfg.competenceFloor) {
			members.erase(members.begin() + i);
			protectedIdx.clear();
			RebuildCellMap();
			for (auto& kv : cellToMember) protectedIdx.insert(kv.second);
		}

	while ((int)members.size() > cfg.maxMembers) {
		int worst = -1; float wf = 1e30f;
		for (int i = 0; i < (int)members.size(); i++) {
			if (members[i].exploiter || protectedIdx.count(i)) continue;
			if (members[i].fitness < wf) { wf = members[i].fitness; worst = i; }
		}
		if (worst < 0) break; // everything left is protected
		members.erase(members.begin() + worst);
		protectedIdx.clear();
		RebuildCellMap();
		for (auto& kv : cellToMember) protectedIdx.insert(kv.second);
	}
	RebuildCellMap();
}

void LeagueArchive::EvolveStep(Report& report) {
	RG_NO_GRAD;

	// Seed the archive from the main agent when empty (founds lineage 0).
	if (members.empty()) {
		ReseedFromMain();
		return;
	}

	// Age everyone one step (drives the staleness refresh).
	for (auto& m : members) m.age++;

	// Produce a batch of MAP-Elites candidates via mutation + crossover on existing members.
	int nCandidates = 4;
	for (int c = 0; c < nCandidates; c++) {
		int pa = SampleOpponent();
		std::vector<torch::Tensor> childW;
		int childLineage = members[pa].lineage;
		if ((int)members.size() >= 2 && (Math::RandInt(0, 2) == 0)) {
			int pb = SampleOpponent();
			for (size_t k = 0; k < members[pa].params.size(); k++)
				childW.push_back(League::CrossoverDARE(members[pa].params[k], members[pb].params[k], cfg.dareDropRate));
		} else {
			for (auto& t : members[pa].params)
				childW.push_back(League::MutateGaussian(t, cfg.mutationSigma));
		}

		std::vector<float> bd;
		float q = EvaluateMember(childW, bd);

		Member m; m.params = childW; m.bd = bd; m.fitness = q; m.matches = 1; m.lineage = childLineage;

		// Exploiter audit: a strong member whose cell is already held is evidence the descriptor
		// basis is missing an axis.
		long cell = CellIndex(bd);
		auto occ = cellToMember.find((int)cell);
		if (q > 0 && occ != cellToMember.end())
			exploiterUnmappedWins++;

		TryInsert(std::move(m));
	}

	EvolveExploiters();
	RefreshStalest();
	Cull();
}

void LeagueArchive::LogMetrics(Report& report) const {
	long totalCells = 1;
	for (size_t i = 0; i < cfg.gridAxes.size(); i++) totalCells *= cfg.binsPerAxis;
	report["League/Cell Count"] = (float)cellToMember.size();
	report["League/Member Count"] = (float)members.size();
	report["League/Coverage"] = (float)cellToMember.size() / (float)std::max(1L, totalCells);
	report["League/Exploiter Unmapped Wins"] = (float)exploiterUnmappedWins;
	report["League/BD Samples"] = (float)bdSamples.size();
	report["League/Quantile Bins Live"] = binEdges.empty() ? 0.0f : 1.0f;
	// Per-axis span of the live quantile edges (how tightly the grid has zoomed onto the population).
	for (size_t a = 0; a < binEdges.size() && a < cfg.gridAxes.size(); a++)
		if (!binEdges[a].empty())
			report["League/Edge Span " + cfg.gridAxes[a]] = binEdges[a].back() - binEdges[a].front();

	int exploiters = 0; std::set<int> lineages;
	float bestExploiterFit = -1e30f, meanFit = 0;
	for (auto& m : members) {
		if (m.exploiter) { exploiters++; bestExploiterFit = std::max(bestExploiterFit, m.fitness); }
		lineages.insert(m.lineage);
		meanFit += m.fitness;
	}
	report["League/Exploiter Count"] = (float)exploiters;
	report["League/Lineage Count"] = (float)lineages.size();
	if (!members.empty()) report["League/Mean Fitness"] = meanFit / members.size();
	if (exploiters > 0) report["League/Best Exploiter Fitness"] = bestExploiterFit;

	float worst = 1e30f, best = -1e30f;
	for (auto& kv : cellToMember) { worst = std::min(worst, members[kv.second].fitness); best = std::max(best, members[kv.second].fitness); }
	if (!cellToMember.empty()) {
		report["League/Worst Cell Fitness"] = worst;
		report["League/Best Cell Fitness"] = best;
	}

	// Mean pairwise behavioral distance (diversity health).
	double dsum = 0; long dn = 0;
	for (size_t i = 0; i < members.size(); i++)
		for (size_t j = i + 1; j < members.size(); j++) {
			double d = 0;
			for (size_t a = 0; a < members[i].bd.size() && a < members[j].bd.size(); a++)
				d += (members[i].bd[a] - members[j].bd[a]) * (members[i].bd[a] - members[j].bd[a]);
			dsum += std::sqrt(d); dn++;
		}
	if (dn > 0) report["League/Pairwise Behavioral Distance"] = (float)(dsum / dn);
}

void LeagueArchive::OnIteration(Report& report, uint64_t totalIterations) {
	if (!cfg.enabled) return;

	// Quantile-edge maintenance: bootstrap the edges as soon as enough BD samples exist (until then
	// binning is uniform), then refresh on the reseed cadence so the grid tracks the population as
	// the main improves. Runs BEFORE the reseed so a new seed is tokenized under fresh edges.
	if (cfg.quantileBins) {
		bool bootstrap = binEdges.empty() && (int)bdSamples.size() >= cfg.quantileMinSamples;
		bool periodic = cfg.reseedEveryIters > 0 && totalIterations > 0
			&& (totalIterations % cfg.reseedEveryIters) == 0 && !bdSamples.empty();
		if (bootstrap || periodic)
			RefreshBinEdges();
	}

	// Found a new lineage from the current main on the re-seed cadence (a fresh "era").
	if (cfg.reseedEveryIters > 0 && totalIterations > 0 && (totalIterations % cfg.reseedEveryIters) == 0
		&& !members.empty()) {
		ReseedFromMain();
		Cull();
	}

	if (cfg.evolveEveryIters > 0 && (totalIterations % cfg.evolveEveryIters) == 0)
		EvolveStep(report);
	LogMetrics(report);
}

void LeagueArchive::ToJSON(nlohmann::json& j) const {
	nlohmann::json l;
	l["member_count"] = members.size();
	l["exploiter_unmapped_wins"] = exploiterUnmappedWins;
	l["next_lineage"] = nextLineage;
	l["bin_edges"] = binEdges;
	l["bd_samples"] = bdSamples;
	l["bd_sample_cursor"] = bdSampleCursor;
	// Persist member metadata; weights are saved to the league dir keyed by index.
	nlohmann::json arr = nlohmann::json::array();
	for (size_t i = 0; i < members.size(); i++) {
		nlohmann::json m;
		m["bd"] = members[i].bd;
		m["fitness"] = members[i].fitness;
		m["cell"] = members[i].cell;
		m["exploiter"] = members[i].exploiter;
		m["lineage"] = members[i].lineage;
		arr.push_back(m);
		if (!leagueDir.empty()) {
			std::filesystem::create_directories(leagueDir / "members");
			// tmp + rename: this dir is SHARED by every checkpoint (and the golden archive's
			// stats reference it too) - a crash mid-write would otherwise leave a truncated
			// .pt that makes every checkpoint's FromJSON throw at boot
			auto finalPath = leagueDir / "members" / (std::to_string(i) + ".pt");
			auto tmpPath = leagueDir / "members" / (std::to_string(i) + ".pt.tmp");
			torch::save(members[i].params, tmpPath.string());
			std::error_code ec;
			std::filesystem::rename(tmpPath, finalPath, ec);
			if (ec)
				RG_LOG("LeagueArchive: failed to finalize member " << i << " save: " << ec.message());
		}
	}
	l["members"] = arr;
	j["league"] = l;
}

void LeagueArchive::FromJSON(const nlohmann::json& j) {
	if (!j.contains("league")) return;
	auto& l = j["league"];
	exploiterUnmappedWins = l.value("exploiter_unmapped_wins", 0L);
	nextLineage = l.value("next_lineage", 0);
	binEdges = l.value("bin_edges", std::vector<std::vector<float>>{});
	bdSamples = l.value("bd_samples", std::vector<std::vector<float>>{});
	bdSampleCursor = l.value("bd_sample_cursor", 0);
	if (bdSampleCursor < 0 || bdSampleCursor >= std::max(1, cfg.quantileSampleCap))
		bdSampleCursor = 0; // cap may have changed between runs
	if ((int)bdSamples.size() > cfg.quantileSampleCap)
		bdSamples.resize(cfg.quantileSampleCap);
	members.clear();
	cellToMember.clear();
	if (leagueDir.empty() || !l.contains("members")) return;
	int nModels = 0; for (const char* n : SCRATCH_MODELS) if (scratch[n]) nModels++;
	int i = 0;
	for (auto& m : l["members"]) {
		Member mem;
		mem.bd = m.value("bd", std::vector<float>{});
		mem.fitness = m.value("fitness", 0.0f);
		mem.cell = m.value("cell", -1);
		mem.exploiter = m.value("exploiter", false);
		mem.lineage = m.value("lineage", 0);
		bool ok = true;
		auto path = leagueDir / "members" / (std::to_string(i) + ".pt");
		if (std::filesystem::exists(path)) {
			try {
				torch::load(mem.params, path.string());
			} catch (std::exception& e) {
				// Drop just this member: throwing here would make the checkpoint loader
				// quarantine the (healthy) checkpoint dir - and since the member files are
				// shared across all checkpoints, it would then destroy the ENTIRE rotation
				// and fall through the golden archive for one bad league file
				RG_LOG("LeagueArchive: dropping member " << i << " (unreadable weights: "
					<< e.what() << ")");
				ok = false;
			}
		} else
			ok = false;
		if (ok && (int)mem.params.size() == nModels) {
			// Migrate stale-shape flat vectors AT STORAGE: crossover/mutation do
			// arithmetic directly on these against fresh SnapshotMain() vectors
			{
				RG_NO_GRAD;
				int k = 0;
				for (const char* n : SCRATCH_MODELS) {
					if (!scratch[n]) continue;
					mem.params[k] = MigrateFlatVec(scratch[n], mem.params[k]);
					k++;
				}
			}
			members.push_back(std::move(mem));
		}
		i++;
	}
	// Re-tokenize from the stored RAW BDs rather than trusting stored cells: the binning scheme
	// (binsPerAxis / quantile edges) may have changed since this archive was saved. Old checkpoints
	// without edges stay uniform-binned until the bootstrap refresh fires.
	for (auto& mem : members)
		if (!mem.exploiter) mem.cell = (int)CellIndex(mem.bd);
	DedupCells();
}
