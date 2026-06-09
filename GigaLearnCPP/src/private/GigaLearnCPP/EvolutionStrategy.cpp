#include "EvolutionStrategy.h"

#include <GigaLearnCPP/Util/Utils.h>
#include <GigaLearnCPP/Util/Timer.h>

#include <RLGymCPP/CommonValues.h>
#include <RLGymCPP/StateSetters/FuzzedKickoffState.h>
#include <RLGymCPP/TerminalConditions/GoalScoreCondition.h>

#include <private/GigaLearnCPP/PPO/PPOLearner.h>
#include <private/GigaLearnCPP/Util/WelfordStat.h>

#include <torch/nn/modules/linear.h>
#include <torch/nn/modules/normalization.h>

#include <random>
#include <algorithm>
#include <numeric>
#include <climits>
#include <cstdint>

using namespace GGL;

// ── EGGROLL low-rank perturbation ────────────────────────────────────────────

namespace {

	// Deterministic seed for a (direction, model, layer). The antithetic sign is applied
	// separately, so a +/- pair shares this base seed (hence the same base noise).
	inline uint64_t SeedFor(int64_t baseSeed, int direction, int modelSalt, int layerIdx) {
		uint64_t h = (uint64_t)baseSeed;
		auto mix = [&](uint64_t x) { h ^= x + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2); };
		mix((uint64_t)(uint32_t)direction);
		mix((uint64_t)(uint32_t)modelSalt);
		mix((uint64_t)(uint32_t)layerIdx);
		return h;
	}

	struct DirSign { int dir; float sign; };
	inline DirSign GetDirSign(int globalMember, int P, bool antithetic) {
		if (antithetic) {
			int half = P / 2;
			if (globalMember < half)
				return { globalMember, +1.0f };
			return { globalMember - half, -1.0f };
		}
		return { globalMember, +1.0f };
	}

	// Fill one member's CPU noise buffers for a Linear layer.
	// The antithetic sign goes on A and nbias only (NOT B), so ΔW = (σ/√r)(sign·A)Bᵀ flips sign
	// for the negative member (negating both A and B would cancel).
	void GenLayerNoise(
		int64_t baseSeed, int globalMember, int modelSalt, int layerIdx,
		int outDim, int inDim, int r, int P, bool antithetic,
		float* Adst, float* Bdst, float* nbDst) {

		DirSign ds = GetDirSign(globalMember, P, antithetic);
		std::mt19937_64 rng(SeedFor(baseSeed, ds.dir, modelSalt, layerIdx));
		std::normal_distribution<float> nd(0.0f, 1.0f);

		for (int i = 0; i < outDim * r; i++) Adst[i] = nd(rng) * ds.sign;
		for (int i = 0; i < inDim * r; i++)  Bdst[i] = nd(rng);
		for (int i = 0; i < outDim; i++)     nbDst[i] = nd(rng) * ds.sign;
	}

	torch::Tensor ApplyActivation(torch::Tensor x, ModelActivationType type) {
		switch (type) {
		case ModelActivationType::RELU:       return torch::relu(x);
		case ModelActivationType::LEAKY_RELU: return torch::leaky_relu(x);
		case ModelActivationType::SIGMOID:    return torch::sigmoid(x);
		case ModelActivationType::TANH:       return torch::tanh(x);
		}
		return x;
	}

	// A model's forward graph flattened into ops, with every tensor pre-cast to the working dtype
	// (bf16 when half precision) and per-member low-rank factors pre-generated. Built ONCE per
	// chunk so the per-step rollout has no dynamic_cast and no re-casting.
	enum class OpType { LINEAR, LAYERNORM, ACTIVATION };
	struct PreparedOp {
		OpType type;

		// LINEAR
		torch::Tensor W, b;            // W = weightᵀ [in, out]; b [out]
		bool hasFactors = false;
		torch::Tensor A, B, nbias;     // [M, out, r], [M, in, r], [M, out]

		// LAYERNORM
		torch::Tensor lnW, lnB;
		std::vector<int64_t> lnShape;
		double lnEps = 1e-5;

		// ACTIVATION
		ModelActivationType act = ModelActivationType::RELU;
	};
	struct PreparedModel {
		std::vector<PreparedOp> ops;
	};

	constexpr int SALT_POLICY = 1;
	constexpr int SALT_SHARED = 2;

	// Build the prepared op list for a model. If lowRank, every Linear layer gets per-member
	// EGGROLL factors keyed by (baseSeed, chunkOffset+member, salt, layerIdx); otherwise it is a
	// plain shared-weight Linear. layerIdx counts Linear layers only and must match ApplyUpdate.
	PreparedModel BuildPreparedModel(
		Model* model, bool lowRank, int modelSalt,
		int64_t baseSeed, int chunkOffset, int M, int r, int P, bool antithetic,
		torch::Device device, torch::ScalarType wdt) {

		PreparedModel pm;
		auto wopt = torch::TensorOptions().device(device).dtype(wdt);

		int layerIdx = 0;
		for (auto& child : model->seq->children()) {
			if (auto lin = std::dynamic_pointer_cast<torch::nn::LinearImpl>(child)) {
				PreparedOp op;
				op.type = OpType::LINEAR;
				int outDim = (int)lin->weight.size(0);
				int inDim = (int)lin->weight.size(1);

				op.W = lin->weight.t().contiguous().to(wopt); // [in, out]
				op.b = lin->bias.to(wopt);                    // [out]

				if (lowRank) {
					op.hasFactors = true;
					std::vector<float> A((size_t)M * outDim * r), B((size_t)M * inDim * r), nb((size_t)M * outDim);
					for (int a = 0; a < M; a++) {
						GenLayerNoise(
							baseSeed, chunkOffset + a, modelSalt, layerIdx, outDim, inDim, r, P, antithetic,
							A.data() + (size_t)a * outDim * r,
							B.data() + (size_t)a * inDim * r,
							nb.data() + (size_t)a * outDim);
					}
					op.A = torch::from_blob(A.data(), { M, outDim, r }).clone().to(wopt);
					op.B = torch::from_blob(B.data(), { M, inDim, r }).clone().to(wopt);
					op.nbias = torch::from_blob(nb.data(), { M, outDim }).clone().to(wopt);
				}

				pm.ops.push_back(std::move(op));
				layerIdx++;
			} else if (auto ln = std::dynamic_pointer_cast<torch::nn::LayerNormImpl>(child)) {
				PreparedOp op;
				op.type = OpType::LAYERNORM;
				op.lnW = ln->weight.to(wopt);
				op.lnB = ln->bias.to(wopt);
				op.lnShape = ln->options.normalized_shape();
				op.lnEps = ln->options.eps();
				pm.ops.push_back(std::move(op));
			} else {
				PreparedOp op;
				op.type = OpType::ACTIVATION;
				op.act = model->config.activationType;
				pm.ops.push_back(std::move(op));
			}
		}
		return pm;
	}

	// Forward a flat batch [N, in] (working dtype) through a prepared model, applying each new
	// player's member-specific low-rank delta to every Linear layer. memberIds [N] indexes the
	// chunk's factor stacks. Ragged team sizes are handled per player; no padding.
	torch::Tensor LowRankForward(
		const PreparedModel& pm, torch::Tensor h, const torch::Tensor& memberIds,
		float sigma, float scale) {

		for (const PreparedOp& op : pm.ops) {
			switch (op.type) {
			case OpType::LINEAR: {
				torch::Tensor base = torch::addmm(op.b, h, op.W); // [N, out]
				if (op.hasFactors) {
					torch::Tensor A_sel = op.A.index_select(0, memberIds);          // [N, out, r]
					torch::Tensor B_sel = op.B.index_select(0, memberIds);          // [N, in, r]
					torch::Tensor nb_sel = op.nbias.index_select(0, memberIds);     // [N, out]
					torch::Tensor t = torch::einsum("ni,nir->nr", { h, B_sel });    // [N, r]
					torch::Tensor low = torch::einsum("nr,nor->no", { t, A_sel });  // [N, out]
					h = base + low * scale + nb_sel * sigma;
				} else {
					h = base;
				}
				break;
			}
			case OpType::LAYERNORM:
				h = torch::layer_norm(h, op.lnShape, op.lnW, op.lnB, op.lnEps);
				break;
			case OpType::ACTIVATION:
				h = ApplyActivation(h, op.act);
				break;
			}
		}
		return h;
	}

	// Everything the per-step member rollout needs; assembled once per chunk.
	struct MemberForwardCtx {
		PreparedModel policyPrepared;
		PreparedModel sharedPrepared; // populated only when sharedLowRank
		Model* sharedHead = nullptr;  // for the unperturbed shared path
		bool sharedLowRank = false;
		float sigma = 0, scale = 0, temperature = 1;
		bool halfPrec = false;
		torch::ScalarType wdt = torch::kFloat32;
	};

	// Run the perturbed members' policy (low-rank), mask + sample. Mirrors
	// InferPolicyProbsFromModels: shared_head feeds the policy; softmax is in float for stability.
	torch::Tensor RunMemberForward(
		const MemberForwardCtx& ctx, torch::Tensor obs, torch::Tensor masks, const torch::Tensor& memberIds) {

		constexpr float ACTION_MIN_PROB = 1e-11f;
		constexpr float ACTION_DISABLED_LOGIT = -1e10f;

		torch::Tensor features;
		if (ctx.sharedLowRank)
			features = LowRankForward(ctx.sharedPrepared, obs.to(ctx.wdt), memberIds, ctx.sigma, ctx.scale);
		else if (ctx.sharedHead)
			features = ctx.sharedHead->Forward(obs, ctx.halfPrec).to(ctx.wdt);
		else
			features = obs.to(ctx.wdt);

		torch::Tensor logits = LowRankForward(ctx.policyPrepared, features, memberIds, ctx.sigma, ctx.scale)
			.to(torch::kFloat32) / ctx.temperature;

		torch::Tensor masksBool = masks.to(torch::kBool);
		torch::Tensor probs = torch::softmax(logits + ACTION_DISABLED_LOGIT * masksBool.logical_not(), -1).clamp(ACTION_MIN_PROB, 1);
		return torch::multinomial(probs, 1, true).flatten();
	}
}

// ── EvolutionStrategy ────────────────────────────────────────────────────────

GGL::EvolutionStrategy::EvolutionStrategy(
	const EvolutionStrategyConfig& config, const RLGC::EnvSetConfig& envSetConfig,
	BatchedWelfordStat* obsStat, float minObsSTD, float maxObsMeanRange, RenderSender* renderSender)
	: renderSender(renderSender) {

	es.config = config;
	es.obsStat = obsStat;
	es.minObsSTD = minObsSTD;
	es.maxObsMeanRange = maxObsMeanRange;

	if (config.enabled) {
		// Inherit the tuned env config (arena count, reward fns, tickSkip, obs builder...).
		RLGC::EnvSetConfig esEnvSetConfig = envSetConfig;
		es.envSet = new RLGC::EnvSet(esEnvSetConfig);

		// Full-game scoring: kickoff + goal terminal (like SkillTracker) so each arena plays
		// repeated kickoff->goal cycles. KEEP the reward functions (do NOT clear) -- they feed
		// the fitness tie-breaker.
		for (int i = 0; i < (int)es.envSet->arenas.size(); i++) {
			es.envSet->stateSetters[i] = { new RLGC::FuzzedKickoffState() };
			es.envSet->terminalConditions[i] = { new RLGC::GoalScoreCondition() };
		}
	} else {
		es.envSet = NULL;
	}
}

GGL::EvolutionStrategy::~EvolutionStrategy() {
	delete es.envSet;
}

void GGL::EvolutionStrategy::OnIteration(PPOLearner* ppo, Report& report, int64_t totalTimesteps, int64_t prevTotalTimesteps) {
	if (!es.config.enabled)
		return;

	es.iterationsSinceRan++;
	if (es.iterationsSinceRan >= es.config.updateInterval) {
		es.iterationsSinceRan = 0;
		RunESStep(ppo, report);
	}
}

void GGL::EvolutionStrategy::RunESStep(PPOLearner* ppo, Report& report) {
	RG_NO_GRAD;

	Timer esTimer = {};

	int P = es.config.populationSize;
	if (es.config.antithetic)
		P &= ~1; // force even so +/- pairs align
	if (P <= 0)
		return;

	int M = (int)es.envSet->arenas.size();
	int numChunks = (P + M - 1) / M;

	int64_t baseSeed = (int64_t)Math::RandInt(0, INT32_MAX) * 2654435761ll + es.stepCounter * 0x9E3779B1ll + 1;
	es.stepCounter++;

	// Obs stats are static during an ES step -> compute the per-dim normalization once.
	if (es.obsStat)
		ComputeObsNorm(es.obsStat, es.minObsSTD, es.maxObsMeanRange, es.obsOffset, es.obsInvStd);

	std::vector<int> goalDiff(P, 0);
	std::vector<float> rewardDiff(P, 0.0f);

	RG_LOG("Running ES step (pop=" << P << ", chunks=" << numChunks << ", gameSimTime=" << es.config.gameSimTime << ")...");

	for (int c = 0; c < numChunks; c++) {
		int chunkOffset = c * M;
		int numMembers = RS_MIN(M, P - chunkOffset);
		EvaluateChunk(ppo, baseSeed, chunkOffset, numMembers, P, goalDiff, rewardDiff);
	}

	// Lexicographic order: goal differential first, reward differential breaks ties.
	std::vector<int> order(P);
	std::iota(order.begin(), order.end(), 0);
	std::sort(order.begin(), order.end(), [&](int a, int b) {
		if (goalDiff[a] != goalDiff[b])
			return goalDiff[a] > goalDiff[b];
		return rewardDiff[a] > rewardDiff[b];
	});

	std::vector<float> shaped(P, 0.0f);
	if (es.config.rankNormalize) {
		// Centered rank: best -> +0.5, worst -> -0.5.
		for (int rankPos = 0; rankPos < P; rankPos++) {
			int m = order[rankPos];
			float frac = (P > 1) ? (float)(P - 1 - rankPos) / (float)(P - 1) : 0.5f;
			shaped[m] = frac - 0.5f;
		}
	} else {
		double mean = 0;
		for (int g : goalDiff) mean += g;
		mean /= P;
		double var = 0;
		for (int g : goalDiff) var += (g - mean) * (g - mean);
		var /= RS_MAX(1, P);
		double stdv = std::sqrt(var) + 1e-8;
		for (int m = 0; m < P; m++)
			shaped[m] = (float)((goalDiff[m] - mean) / stdv);
	}

	ApplyUpdate(ppo, baseSeed, shaped, report);

	double meanGoalDiff = 0;
	int scoring = 0, bestGoalDiff = INT_MIN;
	for (int m = 0; m < P; m++) {
		meanGoalDiff += goalDiff[m];
		if (goalDiff[m] != 0) scoring++;
		bestGoalDiff = RS_MAX(bestGoalDiff, goalDiff[m]);
	}
	meanGoalDiff /= P;
	es.lastMeanGoalDiff = (float)meanGoalDiff;
	es.lastScoringRate = (float)scoring / P;

	report["ES/MeanGoalDiff"] = meanGoalDiff;
	report["ES/BestGoalDiff"] = bestGoalDiff;
	report["ES/ScoringRate"] = (double)scoring / P;
	report["ES/Population"] = P;
	report["ES/ChunkCount"] = numChunks;
	report["ES/StepTime"] = esTimer.Elapsed();

	RG_LOG(" > ES done: meanGoalDiff=" << meanGoalDiff << ", scoringRate=" << es.lastScoringRate << ", time=" << esTimer.Elapsed() << "s");
}

void GGL::EvolutionStrategy::EvaluateChunk(
	PPOLearner* ppo, int64_t baseSeed, int chunkOffset, int numMembers, int P,
	std::vector<int>& outGoalDiff, std::vector<float>& outRewardDiff) {

	RG_NO_GRAD;

	RLGC::EnvSet* envSet = es.envSet;
	int M = (int)envSet->arenas.size();
	int numPlayers = envSet->state.numPlayers;
	int obsSize = envSet->obsSize;
	torch::Device device = ppo->device;

	envSet->Reset();

	// Per-arena member team + player partition (member team = "new", baseline = "old").
	std::vector<Team> memberTeam(M);
	std::vector<int> newPlayerIndices, oldPlayerIndices, newPlayerMemberId;
	std::vector<std::vector<int>> arenaNewPlayers(M), arenaOldPlayers(M);

	for (int a = 0; a < M; a++) {
		memberTeam[a] = (Team)Math::RandInt(0, 2);
		auto& gs = envSet->state.gameStates[a];
		int startIdx = envSet->state.arenaPlayerStartIdx[a];
		for (int j = 0; j < (int)gs.players.size(); j++) {
			int playerIdx = startIdx + j;
			if (gs.players[j].team == memberTeam[a]) {
				newPlayerIndices.push_back(playerIdx);
				newPlayerMemberId.push_back(a);
				arenaNewPlayers[a].push_back(playerIdx);
			} else {
				oldPlayerIndices.push_back(playerIdx);
				arenaOldPlayers[a].push_back(playerIdx);
			}
		}
	}

	torch::Tensor tNewPlayerIndices = torch::tensor(newPlayerIndices);
	torch::Tensor tOldPlayerIndices = torch::tensor(oldPlayerIndices);
	torch::Tensor tMemberIds = torch::tensor(newPlayerMemberId, torch::TensorOptions().dtype(torch::kLong)).to(device);

	// Assemble the per-step member forward context once (prepared low-rank models in working dtype).
	bool sharedLowRank = (es.config.scope == EvolutionStrategyConfig::Scope::POLICY_AND_SHARED_HEAD);
	Model* sharedHead = ppo->models["shared_head"];
	Model* policy = ppo->models["policy"];

	int r = es.config.lowRankRank;
	torch::ScalarType wdt = ppo->config.useHalfPrecision ? RG_HALFPERC_TYPE : torch::kFloat32;

	MemberForwardCtx ctx;
	ctx.sharedHead = sharedHead;
	ctx.sharedLowRank = (sharedHead && sharedLowRank);
	ctx.sigma = es.config.sigma;
	ctx.scale = es.config.sigma / std::sqrt((float)r);
	ctx.temperature = ppo->config.policyTemperature;
	ctx.halfPrec = ppo->config.useHalfPrecision;
	ctx.wdt = wdt;
	ctx.policyPrepared = BuildPreparedModel(policy, true, SALT_POLICY, baseSeed, chunkOffset, M, r, P, es.config.antithetic, device, wdt);
	if (ctx.sharedLowRank)
		ctx.sharedPrepared = BuildPreparedModel(sharedHead, true, SALT_SHARED, baseSeed, chunkOffset, M, r, P, es.config.antithetic, device, wdt);

	std::vector<int> goalsFor(M, 0), goalsAgainst(M, 0);
	std::vector<double> newRewardSum(M, 0.0), oldRewardSum(M, 0.0);

	float stepTime = envSet->config.tickSkip * RLGC::CommonValues::TICK_TIME;
	float totalSimTime = 0;
	for (float t = 0; t < es.config.gameSimTime && totalSimTime < es.config.maxSimTime; t += stepTime, totalSimTime += stepTime) {
		envSet->Reset(); // reset arenas that terminated last step (e.g. after a goal) to a fresh kickoff

		// Standardize obs exactly the way the policy was trained (no-op when standardizeObs is off).
		if (es.obsStat)
			ApplyObsNorm(envSet->state.obs.data.data(), numPlayers, obsSize, es.obsOffset, es.obsInvStd);

		torch::Tensor tStates = DIMLIST2_TO_TENSOR<float>(envSet->state.obs);
		torch::Tensor tActionMasks = DIMLIST2_TO_TENSOR<uint8_t>(envSet->state.actionMasks);

		torch::Tensor tNewStates = tStates.index_select(0, tNewPlayerIndices).to(device, true);
		torch::Tensor tOldStates = tStates.index_select(0, tOldPlayerIndices).to(device, true);
		torch::Tensor tNewMasks = tActionMasks.index_select(0, tNewPlayerIndices).to(device, true);
		torch::Tensor tOldMasks = tActionMasks.index_select(0, tOldPlayerIndices).to(device, true);

		envSet->StepFirstHalf(true);

		// Members (perturbed, low-rank) vs the current policy baseline.
		torch::Tensor tNewActions = RunMemberForward(ctx, tNewStates, tNewMasks, tMemberIds);

		torch::Tensor tOldActions, _tLogProbs;
		ppo->InferActions(tOldStates, tOldMasks, &tOldActions, &_tLogProbs); // current policy (this->models)

		torch::Tensor tActions = torch::zeros({ numPlayers }, torch::kLong);
		tActions.index_copy_(0, tNewPlayerIndices, tNewActions.cpu());
		tActions.index_copy_(0, tOldPlayerIndices, tOldActions.cpu());

		auto curActions = TENSOR_TO_VEC<int>(tActions);

		envSet->Sync();
		envSet->StepSecondHalf(curActions, false);

		// Goal attribution.
		// RS_TEAM_FROM_Y returns the team whose goal the ball is IN (i.e. the team that conceded),
		// matching the convention used in GoalReward: scored = (player.team != RS_TEAM_FROM_Y(...))
		for (int a = 0; a < M; a++) {
			auto& gs = envSet->state.gameStates[a];
			if (gs.goalScored) {
				Team concededTeam = RS_TEAM_FROM_Y(gs.ball.pos.y);
				if (concededTeam == memberTeam[a])
					goalsAgainst[a]++;
				else
					goalsFor[a]++;
			}
		}

		// Reward tie-breaker (mean per-team reward this step).
		for (int a = 0; a < M; a++) {
			if (!arenaNewPlayers[a].empty()) {
				double s = 0;
				for (int p : arenaNewPlayers[a]) s += envSet->state.rewards[p];
				newRewardSum[a] += s / arenaNewPlayers[a].size();
			}
			if (!arenaOldPlayers[a].empty()) {
				double s = 0;
				for (int p : arenaOldPlayers[a]) s += envSet->state.rewards[p];
				oldRewardSum[a] += s / arenaOldPlayers[a].size();
			}
		}

		if (renderSender)
			renderSender->Send(envSet->state.gameStates[0]);
	}

	for (int a = 0; a < numMembers; a++) {
		outGoalDiff[chunkOffset + a] = goalsFor[a] - goalsAgainst[a];
		outRewardDiff[chunkOffset + a] = (float)(newRewardSum[a] - oldRewardSum[a]);
	}
}

void GGL::EvolutionStrategy::ApplyUpdate(PPOLearner* ppo, int64_t baseSeed, const std::vector<float>& shaped, Report& report) {
	RG_NO_GRAD;

	int P = (int)shaped.size();
	int r = es.config.lowRankRank;
	float lr = es.config.learningRate;
	float wd = es.config.weightDecay;
	float scaleW = 1.0f / ((float)P * std::sqrt((float)r));
	float scaleB = 1.0f / (float)P;
	torch::Device device = ppo->device;

	bool sharedLowRank = (es.config.scope == EvolutionStrategyConfig::Scope::POLICY_AND_SHARED_HEAD);

	double updateNormSq = 0;

	// The gradient uses the nominal (float32) perturbation directions, which is correct even when
	// the fitness forward ran in bf16: that lower-precision forward is just a faster, slightly
	// noisier fitness evaluator -- we still want to step along the intended direction.
	auto fnUpdateModel = [&](Model* model, int modelSalt) {
		if (!model)
			return;

		int layerIdx = 0;
		for (auto& child : model->seq->children()) {
			auto lin = std::dynamic_pointer_cast<torch::nn::LinearImpl>(child);
			if (!lin)
				continue;

			int outDim = (int)lin->weight.size(0);
			int inDim = (int)lin->weight.size(1);

			torch::Tensor gW = torch::zeros({ outDim, inDim });
			torch::Tensor gB = torch::zeros({ outDim });

			std::vector<float> A((size_t)outDim * r), B((size_t)inDim * r), nb((size_t)outDim);
			for (int m = 0; m < P; m++) {
				float s = shaped[m];
				if (s == 0.0f)
					continue;

				GenLayerNoise(baseSeed, m, modelSalt, layerIdx, outDim, inDim, r, P, es.config.antithetic,
					A.data(), B.data(), nb.data());

				torch::Tensor At = torch::from_blob(A.data(), { outDim, r });
				torch::Tensor Bt = torch::from_blob(B.data(), { inDim, r });
				torch::Tensor nbt = torch::from_blob(nb.data(), { outDim });

				gW.addmm_(At, Bt.t(), 1.0, s); // gW += s * (A @ Bᵀ)
				gB.add_(nbt, s);               // gB += s * n
			}

			gW = gW.to(device) * scaleW;
			gB = gB.to(device) * scaleB;

			updateNormSq += (gW.square().sum().item<double>() + gB.square().sum().item<double>()) * (double)(lr * lr);

			// θ += lr·g, then decoupled weight decay.
			lin->weight.add_(gW, lr);
			lin->weight.mul_(1.0f - lr * wd);
			lin->bias.add_(gB, lr);
			lin->bias.mul_(1.0f - lr * wd);

			layerIdx++;
		}
	};

	fnUpdateModel(ppo->models["policy"], SALT_POLICY);
	if (sharedLowRank)
		fnUpdateModel(ppo->models["shared_head"], SALT_SHARED);

	report["ES/UpdateNorm"] = std::sqrt(updateNormSq);
}
