#include <GigaLearnCPP/NextoEval.h>

#include "NextoOpponent.h"
#include "PPO/PPOLearner.h"

#include <GigaLearnCPP/Util/InferUnit.h>

#include <RLGymCPP/EnvSet/EnvSet.h>
#include <RLGymCPP/ObsBuilders/AdvancedObsPadded.h>
#include <RLGymCPP/ActionParsers/DefaultAction.h>
#include <RLGymCPP/StateSetters/KickoffState.h>
#include <RLGymCPP/TerminalConditions/GoalScoreCondition.h>
#include <RLGymCPP/TerminalConditions/NoTouchCondition.h>

#include <ATen/Parallel.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

using namespace RLGC;

// Match-flow parity notes (this measures the QUALIFIER's distribution, so every
// divergence from a real RLBot match vs Nexto is deliberate and listed):
//  - Kickoff-only resets, goal-only terminals: the real match structure. NoTouch(30)
//    plus a hard episode cap only break dead/degenerate episodes; those episodes
//    count as neither side's goal and are reported separately.
//  - Pulsar acts through InferUnit exactly as GigaLearnRLBot does (its own obs
//    build per decision, argmax unless GGL_SAMPLE_ACTIONS=1).
//  - Nexto acts through the trainer's NextoOpponent with the terminals vector
//    passed exactly as Learner.cpp passes it (all-zero after EnvSet::Reset()), so
//    its prev-action memory persists across goals - which is also how its real
//    RLBot process behaves within a continuous match.
//  - Sides alternate per arena (even arenas: Nexto on ORANGE) to cancel any side bias.

namespace {

	int EnvInt(const char* name, int def) {
		const char* v = std::getenv(name);
		return v && *v ? std::atoi(v) : def;
	}

	bool EnvOn(const char* name) {
		const char* v = std::getenv(name);
		return v && *v && std::string(v) != "0";
	}

	std::string EnvStr(const char* name, const std::string& def) {
		const char* v = std::getenv(name);
		return v && *v ? std::string(v) : def;
	}

	// P(we reach `need` goals before the opponent reaches `oppNeed`) at per-goal win
	// probability p: the race is decided within the first need+oppNeed-1 goals, so
	// it is P(Binomial(need+oppNeed-1, p) >= need). Summed in log space.
	double QualifyProb(double p, int need, int oppNeed) {
		if (p <= 0) return 0;
		if (p >= 1) return 1;
		int n = need + oppNeed - 1;
		double sum = 0;
		for (int k = need; k <= n; k++) {
			double logC = std::lgamma(n + 1.0) - std::lgamma(k + 1.0) - std::lgamma(n - k + 1.0);
			sum += std::exp(logC + k * std::log(p) + (n - k) * std::log(1 - p));
		}
		return sum;
	}

	// Wilson 95% interval on a binomial share (goals are near-independent Bernoulli
	// trials here - each starts from a fresh kickoff, unlike per-step metrics and
	// their episode-cluster variance)
	void Wilson95(int successes, int n, double& lo, double& hi) {
		if (n == 0) { lo = 0; hi = 1; return; }
		const double z = 1.96, z2 = z * z;
		double phat = (double)successes / n;
		double denom = 1 + z2 / n;
		double center = (phat + z2 / (2 * n)) / denom;
		double half = z * std::sqrt(phat * (1 - phat) / n + z2 / (4.0 * n * n)) / denom;
		lo = center - half; hi = center + half;
	}

	// Hard per-episode sim-time cap: two competent bots can juggle past NoTouch
	// forever without scoring; a capped episode is data about neither side's goal
	// rate, so it just resets (counted and reported as "capped").
	class EpisodeTimeCondition : public TerminalCondition {
	public:
		float elapsed = 0, maxTime;
		EpisodeTimeCondition(float maxTime) : maxTime(maxTime) {}
		virtual void Reset(const GameState& initialState) override { elapsed = 0; }
		virtual bool IsTerminal(const GameState& currentState) override {
			elapsed += currentState.deltaTime;
			return elapsed >= maxTime;
		}
		virtual bool IsTruncation() override { return true; }
	};

	// Same trick as src/RLBotMain.cpp's ComputeObsSize: measure a representative 1v1 obs
	int ComputeObsSize(ObsBuilder* obs) {
		GameState gs;
		Player a = {}; a.carId = 1; a.team = Team::BLUE;   a.index = 0;
		Player b = {}; b.carId = 2; b.team = Team::ORANGE; b.index = 1;
		gs.players = { a, b };
		return (int)obs->BuildObs(gs.players[0], gs).size();
	}
}

int GGL::RunNextoEval() {
	// This runs beside the LIVE trainer: CPU only, and polite about threads. The env
	// thread pool still sizes itself to the machine, but with this few arenas its
	// workers are mostly parked; `nice -n 19` on the process covers the rest.
	at::set_num_threads(EnvInt("GGL_EVAL_TORCH_THREADS", 4));

	std::string checkpoint = EnvStr("GGL_CHECKPOINT", "");
	if (checkpoint.empty())
		RG_ERR_CLOSE("GGL_NEXTO_EVAL needs GGL_CHECKPOINT=<saved checkpoint dir> "
			"(copy it out of the live rotation first - checkpoints rotate)");
	std::string nextoPath = EnvStr("GGL_NEXTO_MODEL", "rlbot-run/nexto/nexto-model.pt");
	int numArenas = EnvInt("GGL_EVAL_ARENAS", 8);
	int targetGoals = EnvInt("GGL_EVAL_GOALS", 200);
	float epCapS = (float)EnvInt("GGL_EVAL_EP_CAP_S", 300);
	bool sample = EnvOn("GGL_SAMPLE_ACTIONS");
	int tickSkip = EnvInt("GGL_TICK_SKIP", 8);

	// The qualifier race parameters ("score NEED before Nexto scores OPP_NEED")
	const int QUAL_NEED = 42, QUAL_OPP = 28;

	RG_LOG("=== Nexto goal-share eval ===");
	RG_LOG("  checkpoint: " << checkpoint);
	RG_LOG("  nexto:      " << nextoPath);
	RG_LOG("  arenas=" << numArenas << " targetGoals=" << targetGoals
		<< " tickSkip=" << tickSkip << " pulsar=" << (sample ? "SAMPLED" : "argmax (deployment)"));

	// ---- Pulsar via the deployment path (mirrors src/RLBotMain.cpp) ----
	auto* obsBuilder = new AdvancedObsPadded(3);
	auto* actionParser = new DefaultAction();
	int obsSize = ComputeObsSize(obsBuilder);

	std::vector<int> sharedSizes, policySizes;
	try {
		sharedSizes = ReadLayerSizesFromModule(checkpoint + "/SHARED_HEAD.lt", false);
		policySizes = ReadLayerSizesFromModule(checkpoint + "/POLICY.lt", true);
	} catch (const std::exception& e) {
		RG_ERR_CLOSE("Could not read layer sizes from checkpoint \"" << checkpoint << "\": " << e.what());
	}

	PartialModelConfig sharedHeadConfig;
	sharedHeadConfig.layerSizes = sharedSizes;
	sharedHeadConfig.addResiduals = true; // current dense lineage; same as RLBotMain.cpp
	sharedHeadConfig.activationType = ModelActivationType::LEAKY_RELU;
	sharedHeadConfig.addLayerNorm = true;
	sharedHeadConfig.addOutputLayer = false;

	PartialModelConfig policyConfig;
	policyConfig.layerSizes = policySizes;
	policyConfig.addResiduals = true;
	policyConfig.activationType = ModelActivationType::LEAKY_RELU;
	policyConfig.addLayerNorm = true;

	// MoE trunks override the dense assumptions above (geometry read from the weights).
	// Under GGL_MOE the trainer also shrinks the policy head to a single dense layer with
	// no residuals, so mirror that too — otherwise the head shape will not match.
	if (ReadMoEConfigFromModule(checkpoint + "/SHARED_HEAD.lt", sharedHeadConfig,
			EnvInt("GGL_MOE_TOPK", 4))) {
		// ONLY the trunk's residual flag is an MoE consequence (MoE blocks carry their
		// own skips). Do NOT touch the POLICY head: addResiduals is load-invisible and
		// behaviour-critical, so forcing it false made the 402M moe2 checkpoint - which
		// has a 3-layer residual head - load shape-clean and then play at 0-250 vs Nexto.
		// A single-layer head (this run's GGL_MOE config) is unaffected either way,
		// since {W} is unchanged by the flag.
	}

	// CPU by default so this can run beside the live trainer without touching its GPU.
	// GGL_EVAL_GPU=1 for large models: a 1B MoE (78M active params) on CPU is far too
	// slow to reach a useful goal count — the first attempt on this checkpoint made no
	// scoring progress in 10 minutes, while the GPU path is interactive.
	const bool evalGPU = EnvOn("GGL_EVAL_GPU");
	RG_LOG("  device:     " << (evalGPU ? "cuda" : "cpu"));
	InferUnit inferUnit(obsBuilder, obsSize, actionParser,
		sharedHeadConfig, policyConfig, checkpoint, /*useGPU=*/evalGPU);

	// ---- Nexto via the trainer's adapter, on CPU (the GPU belongs to the trainer) ----
	// Nexto follows the same device: leaving it on CPU while Pulsar is on GPU would make
	// the opponent's forward the new bottleneck (it is small, but it runs every step).
	NextoOpponent nexto(nextoPath, torch::Device(evalGPU ? torch::kCUDA : torch::kCPU));

	// ---- Match-flow env ----
	EnvSetConfig cfg = {};
	cfg.numArenas = numArenas;
	cfg.tickSkip = tickSkip;
	cfg.actionDelay = 0;
	cfg.saveRewards = false;
	cfg.envCreateFn = [epCapS](int idx) {
		auto arena = Arena::Create(GameMode::SOCCAR);
		arena->AddCar(Team::BLUE);
		arena->AddCar(Team::ORANGE);
		EnvCreateResult r = {};
		r.arena = arena;
		r.actionParser = new DefaultAction();
		r.obsBuilder = new AdvancedObsPadded(3);
		r.stateSetter = new KickoffState();
		r.terminalConditions = {
			new GoalScoreCondition(),
			new NoTouchCondition(30),
			new EpisodeTimeCondition(epCapS),
		};
		r.rewards = {}; // eval only: no reward stack
		return r;
	};
	EnvSet envSet(cfg);

	const int numPlayers = envSet.state.numPlayers;
	nexto.BeginServe(numPlayers);

	// Alternate Nexto's side per arena to cancel side bias
	std::vector<Team> nextoTeam(numArenas);
	for (int a = 0; a < numArenas; a++)
		nextoTeam[a] = (a & 1) ? Team::BLUE : Team::ORANGE;

	// ---- MPC lookahead search (GGL_SEARCH=1) ----
	// v0, goal-aware only: for each of the policy's top-K candidate actions, roll the
	// world forward `horizon` decisions in a scratch arena - Pulsar continued by its own
	// argmax, Nexto simulated by a rollout COPY of the adapter (shared frozen weights,
	// separate prev-action memory seeded from the live one each search) - and score
	// LEXICOGRAPHICALLY: scoring within the horizon dominates (sooner is better),
	// conceding dominates negatively (later is better), and when no candidate's rollout
	// sees any goal (most decisions) the ranking degenerates to the policy's own argmax,
	// i.e. exactly the baseline actor. Any goal-share delta vs the no-search baseline is
	// therefore attributable to ~1s of exact goal foresight against a known opponent.
	// No critic leaf value in v0 - that is the planned v2 lever if this one moves p.
	const bool searchOn = EnvOn("GGL_SEARCH");
	const int topK = EnvInt("GGL_SEARCH_TOPK", 12);
	const int horizon = EnvInt("GGL_SEARCH_HORIZON", 15);
	const bool searchGate = EnvOn("GGL_SEARCH_GATE"); // optional speed gate, default OFF
	const float SEARCH_GAMMA = 0.9969f; // per-decision at ts8 (= TRAIN_GAMMA), tie-orders goal timing

	const int numSlots = searchOn ? numArenas * topK : 0;
	std::vector<Arena*> scratch(numSlots);
	std::vector<DefaultAction*> slotParsers(numSlots);
	std::vector<GameState> scratchGs(numSlots), scratchPrevGs(numSlots);
	std::vector<uint8_t> slotActive(numSlots, 0);
	std::vector<float> slotOutcome(numSlots, 0);   // +g^t scored, -g^t conceded, 0 none
	std::vector<int> slotCandidate(numSlots, 0);   // the candidate action this slot tests
	std::vector<int> slotActionIdx(numSlots * 2, 0);
	std::vector<uint8_t> zeroTermSlots(numSlots, 0);
	std::optional<NextoOpponent> rolloutNexto;
	if (searchOn) {
		for (int s = 0; s < numSlots; s++) {
			scratch[s] = Arena::Create(GameMode::SOCCAR);
			scratch[s]->AddCar(Team::BLUE);
			scratch[s]->AddCar(Team::ORANGE);
			slotParsers[s] = new DefaultAction();
		}
		rolloutNexto.emplace(nexto); // shares the frozen module; separate prev-action memory
		rolloutNexto->BeginServe(numSlots * 2);
		RG_LOG("  search: ON  topK=" << topK << " horizon=" << horizon << " decisions ("
			<< horizon * tickSkip / 120.0 << "s), gate=" << (searchGate ? "on" : "off"));
	}

	int64_t searchDecisions = 0, searchOverrides = 0, searchGoalSighted = 0;

	// Copy live arena physics into a scratch arena (cars/ball/pads by construction order)
	auto fnCopyArenaState = [](Arena* src, Arena* dst) {
		for (size_t i = 0; i < src->_cars.size(); i++)
			dst->_cars[i]->SetState(src->_cars[i]->GetState());
		dst->ball->SetState(src->ball->GetState());
		for (size_t i = 0; i < src->_boostPads.size(); i++)
			dst->_boostPads[i]->SetState(src->_boostPads[i]->GetState());
	};

	// Pick actions for every live Pulsar row via the lookahead. Fills chosen[a] for
	// every live arena a (the action-table index for the Pulsar player there).
	auto fnSearchActions = [&](std::vector<int>& chosen) {
		// One batched policy forward over the live Pulsar rows gives the candidate
		// sets, the tie-break order, and the baseline argmax in one place.
		std::vector<float> obsFlat;
		std::vector<uint8_t> maskFlat;
		std::vector<int> jPulsar(numArenas), jNexto(numArenas);
		for (int a = 0; a < numArenas; a++) {
			auto& gs = envSet.state.gameStates[a];
			jNexto[a] = (gs.players[0].team == nextoTeam[a]) ? 0 : 1;
			jPulsar[a] = 1 - jNexto[a];
			obsFlat += obsBuilder->BuildObs(gs.players[jPulsar[a]], gs);
			maskFlat += actionParser->GetActionMask(gs.players[jPulsar[a]], gs);
		}
		const int numActions = actionParser->GetActionAmount();
		torch::Tensor tProbs;
		{
			RG_NO_GRAD;
			auto tObs = torch::tensor(obsFlat).reshape({ numArenas, obsSize });
			auto tMasks = torch::tensor(maskFlat).reshape({ numArenas, numActions });
			tProbs = PPOLearner::InferPolicyProbsFromModels(
				*inferUnit.models, tObs, tMasks, 1.0f, false).cpu();
		}
		auto probs = tProbs.accessor<float, 2>();

		// Candidate sets: top-K by policy probability (probs are already masked)
		std::vector<std::vector<int>> cands(numArenas);
		std::vector<uint8_t> arenaSearched(numArenas, 0);
		for (int a = 0; a < numArenas; a++) {
			std::vector<int> order(numActions);
			for (int i = 0; i < numActions; i++) order[i] = i;
			int k = RS_MIN(topK, numActions);
			std::partial_sort(order.begin(), order.begin() + k, order.end(),
				[&](int x, int y) { return probs[a][x] > probs[a][y]; });
			for (int i = 0; i < k; i++)
				if (probs[a][order[i]] > 0)
					cands[a].push_back(order[i]);

			// Optional speed gate: only search when a goal within the horizon is
			// plausible (ball near either net or moving fast toward one). Ungated
			// arenas act on the policy argmax directly.
			bool doSearch = true;
			if (searchGate) {
				auto& ball = envSet.state.gameStates[a].ball;
				float dNear = RS_MIN(
					(ball.pos - Vec(0, -5120, 320)).Length(),
					(ball.pos - Vec(0, 5120, 320)).Length());
				doSearch = (dNear < 4000) || (std::abs(ball.vel.y) > 1300);
			}
			arenaSearched[a] = doSearch && cands[a].size() > 1;
			chosen[a] = cands[a][0]; // policy argmax = default / fallback
		}

		// Seed the scratch slots
		int activeCount = 0;
		for (int a = 0; a < numArenas; a++) {
			auto& liveGs = envSet.state.gameStates[a];
			for (int c = 0; c < topK; c++) {
				int s = a * topK + c;
				if (!arenaSearched[a] || c >= (int)cands[a].size()) {
					slotActive[s] = 0;
					continue;
				}
				fnCopyArenaState(envSet.arenas[a], scratch[s]);
				scratchGs[s] = GameState(scratch[s]);
				for (int j = 0; j < 2; j++)
					scratchGs[s].players[j].prevAction = liveGs.players[j].prevAction;
				// Rollout Nexto continues the LIVE prev-action memory of this arena's row
				for (int j = 0; j < 2; j++)
					rolloutNexto->prevActions[s * 2 + j] = nexto.prevActions[a * 2 + j];
				slotActive[s] = 1;
				slotOutcome[s] = 0;
				slotCandidate[s] = cands[a][c];
				activeCount++;
			}
		}
		if (!activeCount)
			return;
		searchDecisions++;

		std::vector<bool> rolloutMask(numSlots * 2, false);
		for (int t = 0; t < horizon && activeCount; t++) {
			// Nexto side, batched over active slots
			for (int s = 0; s < numSlots; s++)
				rolloutMask[s * 2 + jNexto[s / topK]] = slotActive[s] != 0;
			std::vector<int> extRollout(numSlots * 2, 0);
			rolloutNexto->Act(scratchGs, rolloutMask, zeroTermSlots, extRollout);

			// Pulsar side: candidate action at t=0, then its own argmax continuation
			std::vector<int> activeSlots;
			if (t == 0) {
				for (int s = 0; s < numSlots; s++)
					if (slotActive[s])
						slotActionIdx[s * 2 + jPulsar[s / topK]] = slotCandidate[s];
			} else {
				std::vector<float> rObs;
				std::vector<uint8_t> rMask;
				for (int s = 0; s < numSlots; s++) {
					if (!slotActive[s]) continue;
					int jp = jPulsar[s / topK];
					rObs += obsBuilder->BuildObs(scratchGs[s].players[jp], scratchGs[s]);
					rMask += actionParser->GetActionMask(scratchGs[s].players[jp], scratchGs[s]);
					activeSlots.push_back(s);
				}
				RG_NO_GRAD;
				auto tObs = torch::tensor(rObs).reshape({ (int64_t)activeSlots.size(), obsSize });
				auto tMasks = torch::tensor(rMask).reshape({ (int64_t)activeSlots.size(), numActions });
				torch::Tensor tActs;
				PPOLearner::InferActionsFromModels(*inferUnit.models, tObs, tMasks,
					/*deterministic=*/true, 1, false, &tActs, nullptr);
				auto acts = TENSOR_TO_VEC<int>(tActs);
				for (size_t i = 0; i < activeSlots.size(); i++)
					slotActionIdx[activeSlots[i] * 2 + jPulsar[activeSlots[i] / topK]] = acts[i];
			}
			for (int s = 0; s < numSlots; s++)
				if (slotActive[s])
					slotActionIdx[s * 2 + jNexto[s / topK]] = extRollout[s * 2 + jNexto[s / topK]];

			// Physics + gamestate update, parallel over slots
			auto fnRolloutStep = [&](int s) {
				if (!slotActive[s]) return;
				Arena* ar = scratch[s];
				auto& gs = scratchGs[s];
				scratchPrevGs[s] = gs;
				gs.ResetBeforeStep();
				std::vector<Action> acts(2);
				auto carItr = ar->_cars.begin();
				for (int j = 0; j < 2; j++, carItr++) {
					acts[j] = slotParsers[s]->ParseAction(slotActionIdx[s * 2 + j], gs.players[j], gs);
					(*carItr)->controls = (CarControls)acts[j];
				}
				ar->Step(tickSkip);
				gs.UpdateFromArena(ar, acts, &scratchPrevGs[s]);
			};
			g_ThreadPool.StartBatchedJobsChunked(fnRolloutStep, numSlots, false);

			// Resolve goals; a resolved slot stops stepping (its score is frozen)
			for (int s = 0; s < numSlots; s++) {
				if (!slotActive[s] || !scratchGs[s].goalScored) continue;
				int a = s / topK;
				bool weScored = RS_TEAM_FROM_Y(scratchGs[s].ball.pos.y) == nextoTeam[a];
				slotOutcome[s] = (weScored ? 1.f : -1.f) * std::pow(SEARCH_GAMMA, t + 1);
				slotActive[s] = 0;
				activeCount--;
			}
		}

		// Choose: best goal outcome wins; all-zero (no goals sighted) keeps candidate 0,
		// the policy argmax - i.e. baseline behavior. Ties in outcome keep the earlier
		// (higher-probability) candidate.
		for (int a = 0; a < numArenas; a++) {
			if (!arenaSearched[a]) continue;
			float bestScore = 0; int bestIdx = -1; bool anyGoal = false;
			for (int c = 0; c < (int)cands[a].size(); c++) {
				float sc = slotOutcome[a * topK + c];
				if (sc != 0) anyGoal = true;
				if (bestIdx < 0 || sc > bestScore) { bestScore = sc; bestIdx = c; }
			}
			if (anyGoal) {
				searchGoalSighted++;
				if (cands[a][bestIdx] != cands[a][0])
					searchOverrides++;
				chosen[a] = cands[a][bestIdx];
			}
		}
	};

	int64_t pulsarGoals = 0, nextoGoals = 0, cappedEpisodes = 0, steps = 0;
	auto tStart = std::chrono::steady_clock::now();

	std::vector<bool> nextoMask(numPlayers, false);
	std::vector<int> allActions(numPlayers, 0), extActions(numPlayers, 0);
	std::vector<Player> pulsarPlayers;
	std::vector<GameState> pulsarStates;
	std::vector<int> pulsarGlobalIdx;
	std::vector<InferUnit::InferDebug> dbg;

	auto fnReport = [&](bool final) {
		int n = (int)(pulsarGoals + nextoGoals);
		double share = n ? (double)pulsarGoals / n : 0;
		double lo, hi;
		Wilson95((int)pulsarGoals, n, lo, hi);
		double wallS = std::chrono::duration<double>(std::chrono::steady_clock::now() - tStart).count();
		double simS = (double)steps * tickSkip / 120.0 * numArenas;
		RG_LOG((final ? "FINAL: " : "") << "goals " << pulsarGoals << "-" << nextoGoals
			<< " (share " << share * 100 << "%, 95% CI [" << lo * 100 << ", " << hi * 100 << "])"
			<< " capped=" << cappedEpisodes
			<< " | P(qualify " << QUAL_NEED << "-before-" << QUAL_OPP << " per attempt): "
			<< QualifyProb(share, QUAL_NEED, QUAL_OPP) * 100 << "%"
			<< " [CI " << QualifyProb(lo, QUAL_NEED, QUAL_OPP) * 100 << "% .. "
			<< QualifyProb(hi, QUAL_NEED, QUAL_OPP) * 100 << "%]"
			<< " | sim " << simS << "s in wall " << wallS << "s (" << simS / RS_MAX(wallS, 1e-9) << "x)"
			<< (searchOn ? RS_STR(" | search: goalSighted=" << searchGoalSighted
				<< " overrides=" << searchOverrides << " of " << searchDecisions << " searched") : ""));
	};

	while (pulsarGoals + nextoGoals < targetGoals) {
		envSet.Reset();
		envSet.StepFirstHalf(false); // actionDelay 0: a no-op, kept for protocol parity

		// Split rows: Nexto's mask by team, Pulsar rows batched for InferUnit
		pulsarPlayers.clear(); pulsarStates.clear(); pulsarGlobalIdx.clear();
		int gi = 0;
		for (int a = 0; a < numArenas; a++) {
			for (auto& pl : envSet.state.gameStates[a].players) {
				bool isNexto = (pl.team == nextoTeam[a]);
				nextoMask[gi] = isNexto;
				if (!isNexto) {
					pulsarPlayers.push_back(pl);
					pulsarStates.push_back(envSet.state.gameStates[a]);
					pulsarGlobalIdx.push_back(gi);
				}
				gi++;
			}
		}

		nexto.Act(envSet.state.gameStates, nextoMask, envSet.state.terminals, extActions);
		for (int i = 0; i < numPlayers; i++)
			if (nextoMask[i])
				allActions[i] = extActions[i];

		if (searchOn) {
			// 1v1: exactly one Pulsar row per arena, in arena order, so
			// pulsarGlobalIdx[a] is arena a's Pulsar player
			std::vector<int> chosen(numArenas, 0);
			fnSearchActions(chosen);
			for (int a = 0; a < numArenas; a++)
				allActions[pulsarGlobalIdx[a]] = chosen[a];
		} else {
			inferUnit.BatchInferActions(pulsarPlayers, pulsarStates, !sample, 1, &dbg);
			for (size_t i = 0; i < pulsarGlobalIdx.size(); i++)
				allActions[pulsarGlobalIdx[i]] = dbg[i].actionIndex;
		}

		envSet.StepSecondHalf(allActions, false);
		steps++;

		for (int a = 0; a < numArenas; a++) {
			if (!envSet.state.terminals[a])
				continue;
			auto& gs = envSet.state.gameStates[a];
			if (gs.goalScored) {
				// RS_TEAM_FROM_Y(ball y) = the CONCEDING team (same idiom as the
				// trainer's Nexto yardstick counters in Learner.cpp)
				if (RS_TEAM_FROM_Y(gs.ball.pos.y) == nextoTeam[a])
					pulsarGoals++;
				else
					nextoGoals++;
				if ((pulsarGoals + nextoGoals) % 10 == 0)
					fnReport(false);
			} else {
				cappedEpisodes++;
			}
		}
	}

	fnReport(true);
	return 0;
}
