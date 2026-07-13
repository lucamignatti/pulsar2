#include "PSDController.h"
#include "Plasticity.h"
#include "../PPO/PPOLearner.h"

#include <torch/optim/adam.h>
#include <torch/nn/functional/activation.h>
#include <fstream>
#include <algorithm>
#include <numeric>

using namespace GGL;
using namespace torch;

void PSDController::Init(RLGC::EnvSet* envSet, PPOLearner* ppo, int numArenas, std::filesystem::path checkpointFolder) {
	this->envSet = envSet;
	this->ppo = ppo;
	this->numArenas = numArenas;
	if (!checkpointFolder.empty())
		roundsDir = checkpointFolder / "psd_rounds";
}

void PSDController::ToJSON(nlohmann::json& j) const {
	nlohmann::json p;
	p["round"] = round;
	p["sigma"] = sigma;
	p["rng_counter"] = rngCounter;
	p["rating_at_phase_start"] = ratingAtPhaseStart;
	p["have_phase_start_rating"] = havePhaseStartRating;
	p["descend_iters_this_phase"] = descendItersThisPhase;
	p["eff_rank_max"] = effRankMax;
	p["rounds_since_distill"] = roundsSinceDistill;
	p["trunk_frozen"] = trunkFrozen;
	j["psd"] = p;
}

void PSDController::FromJSON(const nlohmann::json& j) {
	if (!j.contains("psd")) return;
	auto& p = j["psd"];
	round = p.value("round", 0);
	sigma = p.value("sigma", cfg.sigma);
	rngCounter = p.value("rng_counter", (uint64_t)0);
	ratingAtPhaseStart = p.value("rating_at_phase_start", 0.0f);
	havePhaseStartRating = p.value("have_phase_start_rating", false);
	descendItersThisPhase = p.value("descend_iters_this_phase", 0);
	effRankMax = p.value("eff_rank_max", 0.0f);
	roundsSinceDistill = p.value("rounds_since_distill", 0);
	trunkFrozen = p.value("trunk_frozen", false);
}

// Centered-rank transform (Salimans et al. 2017): rank ascending, map to [-0.5, 0.5].
static std::vector<float> CenteredRank(const std::vector<float>& f) {
	int n = (int)f.size();
	std::vector<int> idx(n);
	std::iota(idx.begin(), idx.end(), 0);
	std::sort(idx.begin(), idx.end(), [&](int a, int b) { return f[a] < f[b]; });
	std::vector<float> w(n);
	for (int r = 0; r < n; r++)
		w[idx[r]] = (n > 1) ? ((float)r / (n - 1) - 0.5f) : 0.0f;
	return w;
}

bool PSDController::OnDescendIteration(Report& report, float rating, uint64_t totalIterations) {
	if (!cfg.enabled) return false;

	curTotalIters = totalIterations;
	descendItersThisPhase++;
	if (!std::isnan(rating)) {
		lastRating = rating; // latest competence signal, read by the interventions at the probe round
		if (!havePhaseStartRating) {
			ratingAtPhaseStart = rating;
			havePhaseStartRating = true;
		}
	}

	report["PSD/Round"] = (float)round;
	report["PSD/Sigma"] = sigma;
	report["PSD/Phase Descend Iters"] = (float)descendItersThisPhase;

	// Decide whether to launch a probe round.
	bool budgetReached = descendItersThisPhase >= cfg.GExploit;
	bool plateaued = true;
	if (cfg.warmupUntilPlateau) {
		// Judge on lastRating (the most recent sample), not this iteration's `rating`:
		// rating lands only every skill-eval interval, so requiring a sample to coincide
		// with the budget-hit iteration made the gate FAIL OPEN - probe rounds fired
		// ungated on nearly every budget hit, regardless of whether Rating had stalled.
		if (havePhaseStartRating && !std::isnan(lastRating)) {
			float gain = lastRating - ratingAtPhaseStart;
			report["PSD/Phase Elo Gain"] = gain;
			// Plateau only judged once we've spent the exploit budget; before that, keep descending.
			plateaued = gain < cfg.plateauEloGain;
		} else {
			// No competence signal this phase (e.g. skill tracker off or no eval yet):
			// fail SAFE - keep descending rather than probe with an unjudgeable plateau.
			plateaued = false;
		}
	}

	if (!budgetReached) return false;
	if (cfg.warmupUntilPlateau && !plateaued) {
		// Made strong progress this phase — extend DESCEND, re-baseline the phase.
		havePhaseStartRating = false;
		descendItersThisPhase = 0;
		return false;
	}

	RunProbeRound(report);

	// New DESCEND phase begins.
	havePhaseStartRating = false;
	descendItersThisPhase = 0;
	return true;
}

std::vector<int> PSDController::StepActions(
	PolicySlots& ps, int probeArenaStart, int evalArenaStart,
	int perSlotProbeArenas, int perSlotEvalArenas,
	ProbeTape* tape, std::vector<double>* evalRewardAccum) {

	int S = ps.numSlots;
	auto& st = envSet->state;
	int numPlayers = st.numPlayers;

	// Current obs / masks as materialized CPU tensors (cloned before any stepping mutates them).
	Tensor obsAll = torch::from_blob(st.obs.data.data(),
		{ (int64_t)st.obs.size[0], (int64_t)st.obs.size[1] }, torch::kFloat32).clone();
	Tensor maskAll = torch::from_blob(st.actionMasks.data.data(),
		{ (int64_t)st.actionMasks.size[0], (int64_t)st.actionMasks.size[1] }, torch::kUInt8).clone();

	// Base policy actions for every player (valid actions everywhere; overwritten on the blocks).
	Tensor baseActions;
	{
		RG_NO_GRAD;
		Tensor obsDev = obsAll.to(ppo->device, true);
		Tensor maskDev = maskAll.to(ppo->device, true);
		ppo->InferActions(obsDev, maskDev, &baseActions, nullptr);
		baseActions = baseActions.to(torch::kCPU);
	}
	std::vector<int> actions = TENSOR_TO_VEC<int>(baseActions.to(torch::kInt));

	auto fnBlockRange = [&](int arenaStart, int arenaCount, int& pStart, int& pCount) {
		pStart = st.arenaPlayerStartIdx[arenaStart];
		int endArena = arenaStart + arenaCount;
		int pEnd = (endArena < (int)st.arenaPlayerStartIdx.size())
			? st.arenaPlayerStartIdx[endArena] : numPlayers;
		pCount = pEnd - pStart;
	};

	// Route a contiguous, slot-grouped block through PolicySlots and write its actions.
	auto fnRouteBlock = [&](int arenaStart, int perSlotArenas, bool record, std::vector<double>* evalAccum) {
		int arenaCount = perSlotArenas * S;
		if (arenaCount <= 0) return;
		int pStart, pCount;
		fnBlockRange(arenaStart, arenaCount, pStart, pCount);
		int rpp = pCount / S; // rows per slot (grouped: slot 0's arenas first, etc.)

		Tensor obsBlk = obsAll.slice(0, pStart, pStart + pCount);
		Tensor maskBlk = maskAll.slice(0, pStart, pStart + pCount);

		Tensor sampled, logp, rewardRows;
		{
			RG_NO_GRAD;
			Tensor trunk = ppo->models["shared_head"]
				? ppo->models["shared_head"]->Forward(obsBlk.to(ppo->device, true), false)
				: obsBlk.to(ppo->device, true);
			Tensor logits = ps.Forward(trunk);
			constexpr float DISABLED = -1e10f;
			Tensor maskDev = maskBlk.to(ppo->device, torch::kBool, true, true);
			Tensor probs = torch::softmax(logits + DISABLED * maskDev.logical_not(), -1).clamp(1e-11f, 1);
			sampled = torch::multinomial(probs, 1, true);           // [pCount,1]
			logp = torch::log(probs).gather(-1, sampled).flatten(); // [pCount]
			sampled = sampled.flatten().to(torch::kCPU);
		}
		std::vector<int> blkActs = TENSOR_TO_VEC<int>(sampled.to(torch::kInt));
		for (int i = 0; i < pCount; i++)
			actions[pStart + i] = blkActs[i];

		if (record && tape) {
			tape->obs.push_back(obsBlk.clone());
			tape->masks.push_back(maskBlk.clone());
			tape->actions.push_back(sampled.to(torch::kLong));
			tape->logProbs.push_back(logp.to(torch::kCPU));
			tape->rpp = rpp;
		}
		// Reward is read AFTER the step by the caller; stash the block player range via a marker.
		(void)evalAccum;
	};

	// Route probe + eval blocks (record only the probe block).
	fnRouteBlock(probeArenaStart, perSlotProbeArenas, /*record=*/true, nullptr);
	fnRouteBlock(evalArenaStart, perSlotEvalArenas, /*record=*/false, evalRewardAccum);

	return actions;
}

float PSDController::FactorUpdate(PolicySlots& ps, torch::optim::Optimizer& opt, const ProbeTape& tape) {
	if (tape.obs.empty()) return 0;
	int steps = (int)tape.obs.size();
	int S = ps.numSlots;
	int rpp = tape.rpp;
	int rows = S * rpp;
	float gamma = 0.99f;

	// Monte-Carlo reward-to-go per row across the rollout (lightweight, self-contained baseline;
	// no value bootstrap). Advantage = per-slot-centered return. This is a genuine policy-gradient
	// finetune of the factors — the point is only to select gradient-friendly basins, not to be
	// a full GAE critic path.
	std::vector<Tensor> returns(steps);
	Tensor running = torch::zeros({ rows }, torch::TensorOptions().dtype(torch::kFloat));
	for (int t = steps - 1; t >= 0; t--) {
		running = tape.rewards[t] + gamma * running;
		returns[t] = running.clone();
	}

	float clip = ppo->config.clipRange;
	float lastLoss = 0;
	Tensor obsCat = torch::cat(tape.obs, 0).to(ppo->device, true);        // [steps*rows, obs]
	Tensor maskCat = torch::cat(tape.masks, 0).to(ppo->device, true);
	Tensor actCat = torch::cat(tape.actions, 0).to(ppo->device, true);     // [steps*rows]
	Tensor oldLogpCat = torch::cat(tape.logProbs, 0).to(ppo->device, true);
	Tensor retCat = torch::cat(returns, 0).to(ppo->device, true);          // [steps*rows]

	// Center advantages per slot (rows within each step are grouped by slot; center over all
	// steps per slot position is a fine approximation for a short rollout).
	Tensor adv = retCat.view({ steps, S, rpp });
	adv = adv - adv.mean(at::IntArrayRef({ 0, 2 }), /*keepdim=*/true);
	adv = adv.reshape({ -1 });
	adv = (adv - adv.mean()) / (adv.std() + 1e-6f);

	// Trunk is frozen (detached) so gradients reach ONLY the factors.
	Tensor trunk;
	{
		RG_NO_GRAD;
		trunk = ppo->models["shared_head"]
			? ppo->models["shared_head"]->Forward(obsCat, false)
			: obsCat;
	}
	trunk = trunk.detach();

	// One PPO-clip epoch over the whole tape.
	{
		// ps.Forward reshapes to [numSlots, -1, in]; the concatenated tape rows are grouped
		// [step][slot][row], so slot is NOT the outer dim. Re-view so slot is outermost.
		// Easiest: process per step to keep the [S, rpp] grouping ps.Forward expects.
		Tensor totalLoss = torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat).device(ppo->device));
		int off = 0;
		for (int t = 0; t < steps; t++) {
			Tensor trunkT = trunk.slice(0, off, off + rows);
			Tensor logits = ps.Forward(trunkT);
			constexpr float DISABLED = -1e10f;
			Tensor maskT = maskCat.slice(0, off, off + rows).to(torch::kBool);
			Tensor probs = torch::softmax(logits + DISABLED * maskT.logical_not(), -1).clamp(1e-11f, 1);
			Tensor logp = torch::log(probs).gather(-1, actCat.slice(0, off, off + rows).unsqueeze(-1)).flatten();
			Tensor oldLogp = oldLogpCat.slice(0, off, off + rows);
			Tensor a = adv.slice(0, off, off + rows);
			Tensor ratio = torch::exp(logp - oldLogp);
			Tensor loss = -torch::min(ratio * a, torch::clamp(ratio, 1 - clip, 1 + clip) * a).mean();
			totalLoss = totalLoss + loss;
			off += rows;
		}
		totalLoss = totalLoss / steps;
		opt.zero_grad();
		totalLoss.backward();
		opt.step();
		lastLoss = totalLoss.detach().cpu().item<float>();
	}
	return lastLoss;
}

void PSDController::RunProbeRound(Report& report) {
	if (cfg.pureES)
		RunProbeRoundPureES(report);
	else
		RunProbeRoundBaldwinian(report);
}

// Mean per-player per-step reward of the unperturbed base policy over a held-out window. Used both
// as the validation A/B signal (option D) and as the tighten-rule baseline in the pure-ES round.
float PSDController::MeasureAggregateReturn(int windowSteps) {
	double total = 0; long count = 0;
	int W = std::max(1, windowSteps);
	for (int step = 0; step < W; step++) {
		envSet->Reset(); // resets only arenas that actually terminated -> continuous episodes
		Tensor obsAll = torch::from_blob(envSet->state.obs.data.data(),
			{ (int64_t)envSet->state.obs.size[0], (int64_t)envSet->state.obs.size[1] }, torch::kFloat32).clone();
		Tensor maskAll = torch::from_blob(envSet->state.actionMasks.data.data(),
			{ (int64_t)envSet->state.actionMasks.size[0], (int64_t)envSet->state.actionMasks.size[1] }, torch::kUInt8).clone();
		std::vector<int> actions;
		{
			RG_NO_GRAD;
			Tensor a;
			ppo->InferActions(obsAll.to(ppo->device, true), maskAll.to(ppo->device, true), &a, nullptr);
			actions = TENSOR_TO_VEC<int>(a.to(torch::kCPU).to(torch::kInt));
		}
		envSet->StepFirstHalf(true);
		envSet->Sync();
		envSet->StepSecondHalf(actions, false);
		for (int i = 0; i < envSet->state.numPlayers; i++) { total += envSet->state.rewards[i]; count++; }
	}
	return count ? (float)(total / count) : 0.0f;
}

// Pure-ES asymmetric eval step: seat 0 of each eval arena runs that arena's slot policy, the opponent
// seat(s) run the current base policy, so each slot plays a real match against the current self. The
// zero-sum reward terms (which the old mirror eval cancelled to zero) now dominate fitness.
std::vector<int> PSDController::StepActionsAsymmetric(PolicySlots& ps, const std::vector<int>& seat0Idx) {
	Tensor obsAll = torch::from_blob(envSet->state.obs.data.data(),
		{ (int64_t)envSet->state.obs.size[0], (int64_t)envSet->state.obs.size[1] }, torch::kFloat32).clone();
	Tensor maskAll = torch::from_blob(envSet->state.actionMasks.data.data(),
		{ (int64_t)envSet->state.actionMasks.size[0], (int64_t)envSet->state.actionMasks.size[1] }, torch::kUInt8).clone();

	// Base actions for EVERY player (covers the opponent seats + any non-eval arenas).
	std::vector<int> actions;
	{
		RG_NO_GRAD;
		Tensor a;
		ppo->InferActions(obsAll.to(ppo->device, true), maskAll.to(ppo->device, true), &a, nullptr);
		actions = TENSOR_TO_VEC<int>(a.to(torch::kCPU).to(torch::kInt));
	}

	// Route the seat-0 rows through the slots (gathered slot-major -> rowsPerSlot = perSlotEval).
	std::vector<int64_t> idx64(seat0Idx.begin(), seat0Idx.end());
	std::vector<int> seat0Acts;
	{
		RG_NO_GRAD;
		Tensor idxDev = torch::from_blob(idx64.data(), { (int64_t)idx64.size() }, torch::kLong).to(ppo->device);
		Tensor obsSeat0 = obsAll.to(ppo->device, true).index_select(0, idxDev);
		Tensor maskSeat0 = maskAll.to(ppo->device, torch::kBool, true, true).index_select(0, idxDev);
		Tensor trunk = ppo->models["shared_head"]
			? ppo->models["shared_head"]->Forward(obsSeat0, false)
			: obsSeat0;
		Tensor logits = ps.Forward(trunk);
		constexpr float DISABLED = -1e10f;
		Tensor probs = torch::softmax(logits + DISABLED * maskSeat0.logical_not(), -1).clamp(1e-11f, 1);
		seat0Acts = TENSOR_TO_VEC<int>(torch::multinomial(probs, 1, true).flatten().to(torch::kCPU).to(torch::kInt));
	}
	for (size_t k = 0; k < seat0Idx.size(); k++)
		actions[seat0Idx[k]] = seat0Acts[k];
	return actions;
}

// Play the freshly folded policy (one full team) vs the pre-fold `oldPolicy` (the other team)
// across the whole pool and return the mean per-player (new-team - old-team) per-step reward
// differential. Both share the current (unfolded) trunk; only the policy heads differ. Dominated
// by the zero-sum competitive terms (goals ±150, ball-to-goal ±75) since the per-player farmable
// terms are tiny by weight -> a genuine win/edge test. Team membership is read from the live game
// states, so this is valid at any team size (2v2/3v3), not just 1v1.
float PSDController::MeasureHeadToHead(Model* oldPolicy, int windowSteps) {
	std::vector<int64_t> s0, s1;
	s0.reserve(numArenas); s1.reserve(numArenas);
	for (int a = 0; a < numArenas; a++) {
		int p0 = envSet->state.arenaPlayerStartIdx[a];
		auto& players = envSet->state.gameStates[a].players;
		Team newTeam = players[0].team; // new (folded) policy drives this whole team
		for (int i = 0; i < (int)players.size(); i++)
			((players[i].team == newTeam) ? s0 : s1).push_back(p0 + i);
	}
	Tensor s0Dev = torch::from_blob(s0.data(), { (int64_t)s0.size() }, torch::kLong).to(ppo->device);
	Tensor s1Dev = torch::from_blob(s1.data(), { (int64_t)s1.size() }, torch::kLong).to(ppo->device);

	auto fnSeatActions = [&](const Tensor& obsDev, const Tensor& maskBoolDev, const Tensor& idxDev, Model* head) {
		RG_NO_GRAD;
		Tensor obsS = obsDev.index_select(0, idxDev);
		Tensor maskS = maskBoolDev.index_select(0, idxDev);
		Tensor trunk = ppo->models["shared_head"] ? ppo->models["shared_head"]->Forward(obsS, false) : obsS;
		Tensor logits = head->Forward(trunk, false);
		constexpr float DISABLED = -1e10f;
		Tensor probs = torch::softmax(logits + DISABLED * maskS.logical_not(), -1).clamp(1e-11f, 1);
		return TENSOR_TO_VEC<int>(torch::multinomial(probs, 1, true).flatten().to(torch::kCPU).to(torch::kInt));
	};

	double sum0 = 0, sum1 = 0; long n0 = 0, n1 = 0;
	int W = std::max(1, windowSteps);
	for (int step = 0; step < W; step++) {
		envSet->Reset();
		Tensor obsDev = torch::from_blob(envSet->state.obs.data.data(),
			{ (int64_t)envSet->state.obs.size[0], (int64_t)envSet->state.obs.size[1] }, torch::kFloat32).clone().to(ppo->device, true);
		Tensor maskBoolDev = torch::from_blob(envSet->state.actionMasks.data.data(),
			{ (int64_t)envSet->state.actionMasks.size[0], (int64_t)envSet->state.actionMasks.size[1] }, torch::kUInt8).clone().to(ppo->device, torch::kBool, true, true);
		std::vector<int> actions;
		{
			RG_NO_GRAD;
			Tensor a;
			ppo->InferActions(obsDev, maskBoolDev, &a, nullptr);
			actions = TENSOR_TO_VEC<int>(a.to(torch::kCPU).to(torch::kInt));
		}
		std::vector<int> a0 = fnSeatActions(obsDev, maskBoolDev, s0Dev, ppo->models["policy"]);
		std::vector<int> a1 = fnSeatActions(obsDev, maskBoolDev, s1Dev, oldPolicy);
		for (size_t k = 0; k < s0.size(); k++) actions[(size_t)s0[k]] = a0[k];
		for (size_t k = 0; k < s1.size(); k++) actions[(size_t)s1[k]] = a1[k];
		envSet->StepFirstHalf(true);
		envSet->Sync();
		envSet->StepSecondHalf(actions, false);
		for (int64_t idx : s0) { sum0 += envSet->state.rewards[idx]; n0++; }
		for (int64_t idx : s1) { sum1 += envSet->state.rewards[idx]; n1++; }
	}
	// Per-player means so unequal team sizes can't bias the differential
	return (n0 && n1) ? (float)(sum0 / n0 - sum1 / n1) : 0.0f;
}

// Pure-ES probe (EGGROLL-faithful; arXiv 2511.16652 §6.1: large populations are what make ES work).
// No Baldwinian finetune: every arena is an eval arena and each of the S slots is scored on LEVEL
// fitness (mean held-out episodic return) over a long window, so the whole round budget buys
// population size N and fitness SNR instead of per-slot gradient steps. The fold is validation-gated.
void PSDController::RunProbeRoundPureES(Report& report) {
	RG_LOG("PSD: launching pure-ES probe round " << round << " (sigma=" << sigma << ", K=" << cfg.K << ")");
	Timer roundTimer = {};

	int S = cfg.antithetic ? 2 * cfg.K : cfg.K;
	if (S > numArenas)
		RG_ERR_CLOSE("PSD: pure-ES needs S=" << S << " slots <= numArenas=" << numArenas
			<< " - each slot drives seat 0 of its own eval arenas (the seat-0 index walk"
			" below would read past arenaPlayerStartIdx, silently corrupting fitness)."
			" Reduce psd.K or raise numGames.");

	int perSlotEval = std::max(1, numArenas / S);
	int evalArenaStart = 0;
	RG_LOG("PSD:   pure-ES slots=" << S << " arenas/slot=" << perSlotEval
		<< " evalWindow=" << cfg.evalWindowSteps << " (asymmetric: slot vs current base)");

	// Seat-0 player of every eval arena, slot-major (rowsPerSlot = perSlotEval). Each slot drives seat 0
	// against the current base policy in seat 1, so the zero-sum competitive terms — which the old mirror
	// eval cancelled to zero — now dominate fitness. Fixed across the window.
	std::vector<int> seat0Idx;
	seat0Idx.reserve((size_t)S * perSlotEval);
	for (int s = 0; s < S; s++)
		for (int j = 0; j < perSlotEval; j++)
			seat0Idx.push_back(envSet->state.arenaPlayerStartIdx[evalArenaStart + s * perSlotEval + j]);

	// Build the perturbation set for this round.
	uint64_t roundSeed = PSD::NoiseKey(rngCounter, round, 0, 0);
	PSD::PolicySlots ps(ppo->models["policy"], S, cfg.rank, sigma, ppo->device);
	ps.InitFromNoise(roundSeed, cfg.antithetic);

	// Tighten-rule baseline: base-vs-base mirror return (symmetric, so seat-0 mean == all-player mean),
	// on the same scale as each slot's seat-0 fitness -> "hurt" = slot scores below the unperturbed self.
	float baseReturn = MeasureAggregateReturn(cfg.valWindowSteps);

	// --- Score every slot on LEVEL fitness (seat-0 competitive return vs the base opponent) over the
	//     eval window. Split even/odd steps for a direct split-half reliability estimate. ---
	int W = std::max(2, cfg.evalWindowSteps);
	std::vector<double> accEven(S, 0.0), accOdd(S, 0.0);
	for (int step = 0; step < W; step++) {
		envSet->Reset(); // resets only arenas that actually terminated -> continuous episodes
		std::vector<int> actions = StepActionsAsymmetric(ps, seat0Idx);
		envSet->StepFirstHalf(true);
		envSet->Sync();
		envSet->StepSecondHalf(actions, false);
		std::vector<double>& acc = (step & 1) ? accOdd : accEven;
		for (int s = 0; s < S; s++)
			for (int j = 0; j < perSlotEval; j++)
				acc[s] += envSet->state.rewards[seat0Idx[(size_t)s * perSlotEval + j]];
	}
	int evenSteps = (W + 1) / 2, oddSteps = W / 2;
	std::vector<float> fitness(S), fitEven(S), fitOdd(S);
	for (int s = 0; s < S; s++) {
		fitEven[s] = (float)(accEven[s] / std::max(1, perSlotEval * evenSteps));
		fitOdd[s] = (float)(accOdd[s] / std::max(1, perSlotEval * oddSteps));
		fitness[s] = (float)((accEven[s] + accOdd[s]) / std::max(1, perSlotEval * W));
	}

	// Split-half reliability: Spearman between the even- and odd-step rankings.
	std::vector<float> ra = CenteredRank(fitEven), rb = CenteredRank(fitOdd);
	double rnum = 0, rda = 0, rdb = 0;
	for (int s = 0; s < S; s++) { rnum += ra[s] * rb[s]; rda += ra[s] * ra[s]; rdb += rb[s] * rb[s]; }
	float reliability = (rda > 0 && rdb > 0) ? (float)(rnum / std::sqrt(rda * rdb)) : 0;

	// --- metrics ---
	float fmin = *std::min_element(fitness.begin(), fitness.end());
	float fmax = *std::max_element(fitness.begin(), fitness.end());
	float fmean = std::accumulate(fitness.begin(), fitness.end(), 0.0f) / S;
	float fstd = 0; for (float v : fitness) fstd += (v - fmean) * (v - fmean); fstd = std::sqrt(fstd / S);
	std::vector<float> sortedF = fitness; std::sort(sortedF.begin(), sortedF.end());
	float fmed = sortedF[S / 2];
	int hurt = 0; for (float v : fitness) if (v < baseReturn) hurt++;
	float fracHurt = (float)hurt / S;

	report["PSD/Probe Fitness Mean"] = fmean;
	report["PSD/Probe Fitness Std"] = fstd;
	report["PSD/Probe Fitness Max"] = fmax;
	report["PSD/Probe Fitness Min"] = fmin;
	report["PSD/Top Minus Median Fitness"] = fmax - fmed;
	report["PSD/Fitness Reliability"] = reliability;
	report["PSD/Base Return"] = baseReturn;
	report["PSD/Frac Probes Hurt"] = fracHurt;
	report["PSD/Population S"] = (float)S;
	report["PSD/Round Time"] = roundTimer.Elapsed();

	// --- sigma adaptation (handoff §4.2: the central knob). Rules that measure what they claim:
	//   WIDEN   when the ranking is noise (split-half reliability low) -> perturbations too small.
	//   TIGHTEN when most slots (both antithetic twins) fall below the base return -> too big. ---
	if (cfg.adaptSigma) {
		if (fracHurt > 0.75f)        sigma *= 0.85f;
		else if (reliability < 0.2f) sigma *= 1.15f;
		sigma = std::clamp(sigma, cfg.sigmaMin, cfg.sigmaMax);
	}

	// --- Validation-gated ES fold (option D). Snapshot the policy, fold the fitness-weighted ORIGINAL
	//     directions into the base weights, then A/B the base policy's held-out return; revert if the
	//     fold regressed return by more than valMargin, so a noise-fold never lands unchecked. ---
	std::vector<float> weights = CenteredRank(fitness);

	// --- Head-to-head validation-gated ES fold (option D, competitive form). Clone the pre-fold policy,
	//     fold the fitness-weighted ORIGINAL directions into the base weights, then play NEW (seat 0) vs
	//     OLD (seat 1) across the whole pool. Keep the fold only if the new policy is at least as strong
	//     as the one it replaces; else revert. Zero-sum terms make this a true win/edge test, not a
	//     farmable-return A/B. `oldPolicy` doubles as the opponent and the revert source. ---
	Model* oldPolicy = cfg.valGateEnabled ? ppo->models["policy"]->MakeClone() : nullptr;
	float stepNorm = ps.FoldESUpdate(roundSeed, cfg.antithetic, weights, cfg.alpha);
	ppo->models["policy"]->_seqHalfOutdated = true;

	bool reverted = false;
	if (cfg.valGateEnabled) {
		float h2h = MeasureHeadToHead(oldPolicy, cfg.valWindowSteps); // (new seat0) - (old seat1) reward
		report["PSD/Val H2H Diff"] = h2h;
		if (h2h < -cfg.valMargin) {
			RG_NO_GRAD;
			auto live = ppo->models["policy"]->parameters();
			auto old = oldPolicy->parameters();
			for (size_t i = 0; i < live.size(); i++)
				live[i].copy_(old[i]);
			ppo->models["policy"]->_seqHalfOutdated = true;
			reverted = true;
			stepNorm = 0;
		}
		delete oldPolicy->optim;
		delete oldPolicy;
	}
	report["PSD/Fold Reverted"] = reverted ? 1.0f : 0.0f;
	report["PSD/ES Step Norm"] = stepNorm;

	// Plasticity signals + interventions (log, then act on the — possibly reverted — base weights).
	RunInterventions(report);

	if (cfg.dumpRounds)
		PersistRound(fitness, weights, stepNorm);

	round++;
	rngCounter++;
	RG_LOG("PSD: round done. reliability=" << reliability << " fracHurt=" << fracHurt
		<< " stepNorm=" << stepNorm << (reverted ? " [REVERTED]" : "") << " -> sigma=" << sigma);
}

void PSDController::RunProbeRoundBaldwinian(Report& report) {
	RG_LOG("PSD: launching probe round " << round << " (sigma=" << sigma << ", K=" << cfg.K << ")");
	Timer roundTimer = {};

	int S = cfg.antithetic ? 2 * cfg.K : cfg.K;

	// Partition arenas: [probe block | eval block | idle remainder]. Both blocks are split into
	// S equal per-slot sub-blocks, so the player rows land grouped by slot for PolicySlots.
	int evalArenasTarget = (int)std::round(cfg.evalArenaFrac * numArenas);
	int perSlotEval = std::max(1, evalArenasTarget / S);
	int perSlotProbe = std::max(1, (numArenas - perSlotEval * S) / S);
	int probeArenaStart = 0;
	int evalArenaStart = perSlotProbe * S;
	if (evalArenaStart + perSlotEval * S > numArenas) {
		// Not enough arenas — shrink probe block.
		perSlotProbe = std::max(1, (numArenas / S) - perSlotEval);
		evalArenaStart = perSlotProbe * S;
	}
	RG_LOG("PSD:   arenas/slot probe=" << perSlotProbe << " eval=" << perSlotEval << " slots=" << S);

	// Build the perturbation set for this round.
	uint64_t roundSeed = PSD::NoiseKey(rngCounter, round, 0, 0);
	PSD::PolicySlots ps(ppo->models["policy"], S, cfg.rank, sigma, ppo->device);
	ps.InitFromNoise(roundSeed, cfg.antithetic);

	torch::optim::Adam opt(ps.Parameters(),
		torch::optim::AdamOptions(cfg.probeLR).weight_decay(cfg.probeWeightDecay));

	int rolloutSteps = 8; // short rollout per finetune iteration
	// eval fitness trajectory: [GProbe][S]
	std::vector<std::vector<float>> evalTraj;
	float lastProbeLoss = 0;

	for (int it = 0; it < cfg.GProbe; it++) {
		ProbeTape tape;
		std::vector<double> dummy;
		// Collect a short rollout, recording probe transitions and reading rewards each step.
		for (int step = 0; step < rolloutSteps; step++) {
			envSet->Reset();
			std::vector<int> actions = StepActions(ps, probeArenaStart, evalArenaStart, perSlotProbe, perSlotEval, &tape, &dummy);
			envSet->StepFirstHalf(true);
			envSet->Sync();
			envSet->StepSecondHalf(actions, false);

			// Record this step's probe-block rewards into the tape (aligns with the pushed obs).
			int pStart = envSet->state.arenaPlayerStartIdx[probeArenaStart];
			int pCount = perSlotProbe * S * (envSet->state.arenaPlayerStartIdx.size() > 1
				? (envSet->state.arenaPlayerStartIdx[1] - envSet->state.arenaPlayerStartIdx[0]) : 2);
			Tensor rew = torch::from_blob(envSet->state.rewards.data() + pStart, { (int64_t)pCount }, torch::kFloat32).clone();
			tape.rewards.push_back(rew);
		}

		lastProbeLoss = FactorUpdate(ps, opt, tape);

		// Held-out eval: mean reward per slot on the eval block over one short rollout.
		std::vector<double> evalAccum(S, 0.0);
		int evalPlayersPerSlot = perSlotEval * (envSet->state.arenaPlayerStartIdx.size() > 1
			? (envSet->state.arenaPlayerStartIdx[1] - envSet->state.arenaPlayerStartIdx[0]) : 2);
		for (int step = 0; step < rolloutSteps; step++) {
			envSet->Reset();
			std::vector<int> actions = StepActions(ps, probeArenaStart, evalArenaStart, perSlotProbe, perSlotEval, nullptr, nullptr);
			envSet->StepFirstHalf(true);
			envSet->Sync();
			envSet->StepSecondHalf(actions, false);
			int evStartP = envSet->state.arenaPlayerStartIdx[evalArenaStart];
			for (int s = 0; s < S; s++)
				for (int r = 0; r < evalPlayersPerSlot; r++)
					evalAccum[s] += envSet->state.rewards[evStartP + s * evalPlayersPerSlot + r];
		}
		std::vector<float> fit(S);
		for (int s = 0; s < S; s++)
			fit[s] = (float)(evalAccum[s] / std::max(1, evalPlayersPerSlot * rolloutSteps));
		evalTraj.push_back(fit);
	}

	// --- Fitness per slot: level (last) or end-of-window slope (handoff §4.1). ---
	std::vector<float> fitness(S, 0);
	int T = (int)evalTraj.size();
	if (T > 0) {
		if (cfg.fitnessMode == 0 || T < 3) {
			fitness = evalTraj.back();
		} else {
			// Slope over the last half of the finetune window via least squares.
			int w0 = T / 2;
			int n = T - w0;
			float xmean = 0; for (int t = w0; t < T; t++) xmean += t; xmean /= n;
			float xvar = 0; for (int t = w0; t < T; t++) xvar += (t - xmean) * (t - xmean);
			for (int s = 0; s < S; s++) {
				float cov = 0;
				float ymean = 0; for (int t = w0; t < T; t++) ymean += evalTraj[t][s]; ymean /= n;
				for (int t = w0; t < T; t++) cov += (t - xmean) * (evalTraj[t][s] - ymean);
				fitness[s] = (xvar > 1e-9f) ? cov / xvar : 0;
			}
		}
	}

	// Level-vs-slope rank correlation (the §4.1 confound canary): does short-horizon level
	// select the same probes as the trajectory signal?
	if (T >= 3) {
		std::vector<float> lvl = evalTraj.back();
		std::vector<float> wl = CenteredRank(lvl), ws = CenteredRank(fitness);
		double num = 0, dl = 0, ds = 0;
		for (int s = 0; s < S; s++) { num += wl[s] * ws[s]; dl += wl[s] * wl[s]; ds += ws[s] * ws[s]; }
		report["PSD/Fitness Level-Slope RankCorr"] = (float)((dl > 0 && ds > 0) ? num / std::sqrt(dl * ds) : 0);
	}

	// --- ES step: fold the fitness-weighted ORIGINAL directions into the base weights. ---
	std::vector<float> weights = CenteredRank(fitness);
	float stepNorm = ps.FoldESUpdate(roundSeed, cfg.antithetic, weights, cfg.alpha * (1.0f)); // alpha
	// Apply the shrink half of shrink-and-perturb toward the pre-round base is implicit via lambda:
	// (lambda<1 pulls the standing delta down; here delta is folded each round (M=1), so we apply
	// a mild global shrink of the just-updated weights toward their pre-round value is skipped —
	// M=1 keeps the standing delta zero, so lambda acts as a no-op unless distillPeriod>1.)

	// --- metrics ---
	float fmin = *std::min_element(fitness.begin(), fitness.end());
	float fmax = *std::max_element(fitness.begin(), fitness.end());
	float fmean = std::accumulate(fitness.begin(), fitness.end(), 0.0f) / S;
	float fstd = 0; for (float v : fitness) fstd += (v - fmean) * (v - fmean); fstd = std::sqrt(fstd / S);
	std::vector<float> sortedF = fitness; std::sort(sortedF.begin(), sortedF.end());
	float fmed = sortedF[S / 2];

	report["PSD/Probe Fitness Mean"] = fmean;
	report["PSD/Probe Fitness Std"] = fstd;
	report["PSD/Probe Fitness Max"] = fmax;
	report["PSD/Probe Fitness Min"] = fmin;
	report["PSD/Top Minus Median Fitness"] = fmax - fmed;
	report["PSD/ES Step Norm"] = stepNorm;
	report["PSD/Probe Finetune Loss"] = lastProbeLoss;
	report["PSD/GProbe"] = (float)cfg.GProbe;
	report["PSD/Fitness Mode"] = (float)cfg.fitnessMode;
	report["PSD/Round Time"] = roundTimer.Elapsed();

	// --- sigma adaptation (handoff §4.2: the central knob) ---
	// The original rules were inert by construction: "spread" was normalized by |fmed| (~1e-4 under
	// slope fitness, so the widen test could never trip) and "hurt" counted probes below the MEDIAN
	// (~50% by definition, never > 75%). Replaced with rules that measure what they claim:
	//   WIDEN   when the probe ranking is NOISE — split-half reliability: recompute per-slot fitness
	//           from only the even- vs only the odd-indexed eval steps of the scoring window and
	//           Spearman-correlate the two rankings. Same underlying trend, independent noise, so a
	//           low correlation means perturbations are too small to create real differences.
	//   TIGHTEN when most probes actively degrade during their finetune (slope fitness < 0):
	//           perturbations are big enough to knock probes off the policy's basin.
	if (cfg.adaptSigma && T >= 6) {
		int w0 = T / 2;
		// Per-slot fitness recomputed from one parity of the scoring window's eval steps.
		auto fnFitnessOverParity = [&](int parity) {
			std::vector<float> out(S, 0.0f);
			std::vector<int> ts;
			for (int t = w0; t < T; t++)
				if ((t & 1) == parity) ts.push_back(t);
			if (ts.size() < 2) return out;
			if (cfg.fitnessMode == 0) {
				for (int s = 0; s < S; s++) {
					float ym = 0; for (int t : ts) ym += evalTraj[t][s];
					out[s] = ym / ts.size();
				}
			} else {
				float xm = 0; for (int t : ts) xm += t; xm /= ts.size();
				float xv = 0; for (int t : ts) xv += (t - xm) * (t - xm);
				for (int s = 0; s < S; s++) {
					float ym = 0; for (int t : ts) ym += evalTraj[t][s]; ym /= ts.size();
					float cov = 0; for (int t : ts) cov += (t - xm) * (evalTraj[t][s] - ym);
					out[s] = (xv > 1e-9f) ? cov / xv : 0;
				}
			}
			return out;
		};
		std::vector<float> ra = CenteredRank(fnFitnessOverParity(0)), rb = CenteredRank(fnFitnessOverParity(1));
		double num = 0, da = 0, db = 0;
		for (int s = 0; s < S; s++) { num += ra[s] * rb[s]; da += ra[s] * ra[s]; db += rb[s] * rb[s]; }
		float reliability = (da > 0 && db > 0) ? (float)(num / std::sqrt(da * db)) : 0;
		report["PSD/Fitness Reliability"] = reliability;

		// Only meaningful under slope fitness (level of a shaped return has no natural zero).
		float fracHurt = 0;
		if (cfg.fitnessMode == 1) {
			int hurt = 0; for (float v : fitness) if (v < 0) hurt++;
			fracHurt = (float)hurt / S;
			report["PSD/Frac Probes Hurt"] = fracHurt;
		}

		if (cfg.fitnessMode == 1 && fracHurt > 0.75f)
			sigma *= 0.85f;                      // probes knocked off the basin -> tighten
		else if (reliability < 0.2f)
			sigma *= 1.15f;                      // ranking indistinguishable from noise -> widen
		sigma = std::clamp(sigma, cfg.sigmaMin, cfg.sigmaMax);
	}

	// Plasticity signals + interventions (log, then act on the just-folded weights).
	RunInterventions(report);

	if (cfg.dumpRounds)
		PersistRound(fitness, weights, stepNorm);

	round++;
	rngCounter++;
	RG_LOG("PSD: round done. fitness mean=" << fmean << " std=" << fstd << " stepNorm=" << stepNorm << " -> sigma=" << sigma);
}

// Log the two plasticity signals, then apply whichever interventions are enabled to the freshly
// folded weights. Under healthy training every branch is a no-op; each fires only on real degradation
// (or, for the critic reset, on its cadence). NOTE: no function-level no_grad here — DistillReset
// needs autograd for the student; the pure-weight helpers each guard themselves.
void PSDController::RunInterventions(Report& report) {
	Model* pol = ppo->models["policy"];
	if (!pol) return;
	Model* critic = ppo->models["critic"];
	Model* trunk = ppo->models["shared_head"];

	// --- signals (kept: this is the logging the canaries always did) ---
	auto lins = PSD::LinearLayers(pol);
	float effRank = lins.empty() ? 0.0f : PSD::EffectiveRank(lins.back()->weight);
	float deadFrac = PSD::DeadUnitFraction(pol);
	report["Plasticity/Policy Head EffRank"] = effRank;
	report["Plasticity/Policy Dead Unit Frac"] = deadFrac;

	// Warmup gate: on a FRESH run the effective rank starts at its lifetime peak and falls during
	// benign early specialization; the collapse-triggered interventions would read that drop as
	// pathology and fire distill/perturb spuriously. During warmup we LOG the signals and keep the
	// rolling peak up to date, but take NO action, so interventions later arm against an established
	// baseline. (0 warmup = act immediately, correct when resuming a mature checkpoint.)
	bool warming = curTotalIters < (uint64_t)cfg.interventionWarmupIters;
	report["Plasticity/Intervention Warmup"] = warming ? 1.0f : 0.0f;
	if (warming) {
		if (effRank > effRankMax) effRankMax = effRank;
		report["Plasticity/EffRank RollingMax"] = effRankMax;
		return;
	}

	bool policyChanged = false;

	// --- (1) ReDo: recycle dead units in the last hidden layer ---
	if (cfg.redoEnabled)
		policyChanged |= PSD::RecycleDeadUnits(pol, cfg.redoDeadThresh, cfg.redoDeadFracTrigger, report);

	// --- (2) EffRank collapse response: shrink-and-perturb the head (also maintains effRankMax) ---
	if (cfg.effRankResponseEnabled)
		policyChanged |= PSD::EffRankResponse(pol, effRank, effRankMax,
			cfg.effRankCollapseFrac, cfg.effRankShrink, cfg.effRankPerturbSigma, report);
	else if (effRank > effRankMax)
		effRankMax = effRank; // keep tracking the peak even when the response is off

	// --- (3) Critic partial reset toward init (periodic; critic tolerates it) ---
	if (critic && cfg.criticResetPeriod > 0 && round > 0 && (round % cfg.criticResetPeriod) == 0) {
		if (PSD::PartialResetCritic(critic, cfg.criticPartialResetFrac, report))
			critic->_seqHalfOutdated = true;
	}

	// --- (4) Distill reset (deepest): gated on cadence AND a real head-rank collapse ---
	roundsSinceDistill++;
	float effRatio = (effRankMax > 1e-6f) ? (effRank / effRankMax) : 1.0f;
	report["Plasticity/Rounds Since Distill"] = (float)roundsSinceDistill;
	if (cfg.distillPeriod > 0 && roundsSinceDistill >= cfg.distillPeriod && effRatio < cfg.distillTrigger) {
		DistillReset(report);
		roundsSinceDistill = 0;
		policyChanged = true;
	}

	// --- (5) Competence-conditioned freeze of the shared trunk ---
	if (cfg.freezeEnabled && trunk && !std::isnan(lastRating)) {
		bool wantFrozen = lastRating >= cfg.freezeRatingThresh;
		// Apply idempotently: a resumed checkpoint restores trunkFrozen but reloads params with
		// requires_grad=true, so re-asserting the flag every round self-heals that mismatch.
		for (auto& p : trunk->parameters())
			p.set_requires_grad(!wantFrozen);
		if (wantFrozen != trunkFrozen) {
			trunkFrozen = wantFrozen;
			RG_LOG("PSD: trunk " << (trunkFrozen ? "FROZEN" : "unfrozen") << " (rating=" << lastRating << ")");
		}
	}
	if (cfg.freezeEnabled)
		report["Plasticity/Trunk Frozen"] = trunkFrozen ? 1.0f : 0.0f;

	if (policyChanged)
		pol->_seqHalfOutdated = true; // half-precision inference cache must rebuild from new weights
}

// Deepest plasticity reset: reinit the policy sub-net and behavior-distill the CURRENT policy into it
// over the frozen trunk, so plasticity is restored while behavior is preserved. Only the policy is
// reset (the trunk and critic are untouched); the policy optimizer moments are cleared afterward.
void PSDController::DistillReset(Report& report) {
	RG_LOG("PSD: distill reset (policy plasticity reset) at round " << round);
	Model* pol = ppo->models["policy"];
	Model* trunk = ppo->models["shared_head"];

	// 1) Gather an observation batch by rolling the base policy forward a few resets (decorrelated).
	std::vector<Tensor> obsChunks, maskChunks;
	int gathered = 0;
	while (gathered < cfg.distillBatch && (int)obsChunks.size() < 64) {
		envSet->Reset();
		auto& st = envSet->state;
		Tensor obs = torch::from_blob(st.obs.data.data(),
			{ (int64_t)st.obs.size[0], (int64_t)st.obs.size[1] }, torch::kFloat32).clone();
		Tensor mask = torch::from_blob(st.actionMasks.data.data(),
			{ (int64_t)st.actionMasks.size[0], (int64_t)st.actionMasks.size[1] }, torch::kUInt8).clone();
		obsChunks.push_back(obs);
		maskChunks.push_back(mask);
		gathered += (int)obs.size(0);

		std::vector<int> actions;
		{
			RG_NO_GRAD;
			Tensor od = obs.to(ppo->device, true), md = mask.to(ppo->device, true), a;
			ppo->InferActions(od, md, &a, nullptr);
			actions = TENSOR_TO_VEC<int>(a.to(torch::kCPU).to(torch::kInt));
		}
		envSet->StepFirstHalf(true);
		envSet->Sync();
		envSet->StepSecondHalf(actions, false);
	}
	Tensor obsAll = torch::cat(obsChunks, 0).to(ppo->device);
	Tensor maskAll = torch::cat(maskChunks, 0).to(ppo->device);

	// 2) Teacher probabilities from the current policy (snapshot before reinit).
	Tensor teacherProbs, trunkOut;
	{
		RG_NO_GRAD;
		teacherProbs = PPOLearner::InferPolicyProbsFromModels(ppo->models, obsAll, maskAll, 1.0f, false).detach();
		trunkOut = trunk ? trunk->Forward(obsAll, false).detach() : obsAll;
	}
	Tensor maskNot = maskAll.to(torch::kBool).logical_not();
	constexpr float DISABLED = -1e10f;

	// 3) Fresh student policy; distill teacher -> student over the (frozen) trunk features.
	Model* student = pol->MakeEmptyClone();
	torch::optim::Adam sopt(student->parameters(), torch::optim::AdamOptions(cfg.distillLR));
	float finalKL = 0;
	for (int it = 0; it < cfg.distillIters; it++) {
		Tensor logits = student->Forward(trunkOut, false);
		Tensor logp = torch::log_softmax(logits + DISABLED * maskNot, -1);
		Tensor loss = (teacherProbs * (torch::log(teacherProbs) - logp)).sum(-1).mean(); // KL(teacher||student)
		sopt.zero_grad();
		loss.backward();
		sopt.step();
		finalKL = loss.detach().cpu().item<float>();
	}

	// 4) Copy the distilled weights into the live policy in place (keeps tensor identity/optim keys).
	{
		RG_NO_GRAD;
		auto to = pol->parameters();
		auto from = student->parameters();
		for (size_t i = 0; i < to.size(); i++)
			to[i].copy_(from[i]);
	}
	// 5) Deepest reset: drop the policy optimizer moments so it restarts fresh on the new basin.
	pol->optim->state().clear();
	pol->_seqHalfOutdated = true;

	delete student->optim;
	delete student;

	report["Plasticity/Distill Fired"] = 1.0f;
	report["Plasticity/Distill Final KL"] = finalKL;
	RG_LOG("PSD:   distill done, final KL=" << finalKL);
}

void PSDController::PersistRound(const std::vector<float>& fitness, const std::vector<float>& weights, float stepNorm) {
	if (roundsDir.empty()) return;
	std::filesystem::path dir = roundsDir / std::to_string(round);
	std::filesystem::create_directories(dir);

	// Snapshot the just-folded policy weights (the resumable rollback state for this round).
	ppo->models["policy"]->Save(dir, false);
	if (ppo->models["shared_head"]) ppo->models["shared_head"]->Save(dir, false);
	if (ppo->models["critic"]) ppo->models["critic"]->Save(dir, false);

	nlohmann::json j;
	j["round"] = round;
	j["sigma"] = sigma;
	j["K"] = cfg.K;
	j["rank"] = cfg.rank;
	j["alpha"] = cfg.alpha;
	j["lambda"] = cfg.lambda;
	j["fitness_mode"] = cfg.fitnessMode;
	j["step_norm"] = stepNorm;
	j["fitness"] = fitness;
	j["weights"] = weights;
	std::ofstream(dir / "fitness.json") << j.dump(2);
}
