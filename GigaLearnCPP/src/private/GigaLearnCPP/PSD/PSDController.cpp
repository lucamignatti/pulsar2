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

bool PSDController::OnDescendIteration(Report& report, float rating) {
	if (!cfg.enabled) return false;

	descendItersThisPhase++;
	if (!std::isnan(rating) && !havePhaseStartRating) {
		ratingAtPhaseStart = rating;
		havePhaseStartRating = true;
	}

	report["PSD/Round"] = (float)round;
	report["PSD/Sigma"] = sigma;
	report["PSD/Phase Descend Iters"] = (float)descendItersThisPhase;

	// Decide whether to launch a probe round.
	bool budgetReached = descendItersThisPhase >= cfg.GExploit;
	bool plateaued = true;
	if (cfg.warmupUntilPlateau && havePhaseStartRating && !std::isnan(rating)) {
		float gain = rating - ratingAtPhaseStart;
		report["PSD/Phase Elo Gain"] = gain;
		// Plateau only judged once we've spent the exploit budget; before that, keep descending.
		plateaued = gain < cfg.plateauEloGain;
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
	if (cfg.adaptSigma) {
		float spread = (fmax - fmed) / (std::abs(fmed) + 1e-6f);
		int hurt = 0; for (float v : fitness) if (v < fmed) hurt++;
		float fracHurt = (float)hurt / S;
		if (spread < 0.05f) sigma *= 1.15f;          // probes cluster at current perf -> widen
		else if (fracHurt > 0.75f) sigma *= 0.85f;   // most perturbations hurt -> tighten
		sigma = std::clamp(sigma, cfg.sigmaMin, cfg.sigmaMax);
	}

	// Plasticity canaries (read alongside the rating trend — collapse alone never means "freeze").
	PSD::LogPlasticity(ppo->models, report);

	if (cfg.dumpRounds)
		PersistRound(fitness, weights, stepNorm);

	round++;
	rngCounter++;
	RG_LOG("PSD: round done. fitness mean=" << fmean << " std=" << fstd << " stepNorm=" << stepNorm << " -> sigma=" << sigma);
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
