#include "Learner.h"

#include <GigaLearnCPP/PPO/PPOLearner.h>
#include <GigaLearnCPP/PPO/ExperienceBuffer.h>
#include <RLGymCPP/CommonValues.h>
#include <RLGymCPP/ObsBuilders/AdvancedObs.h>
#include <RLGymCPP/StateSetters/FrontierState.h>

#include <torch/cuda.h>
#include <nlohmann/json.hpp>
#include <pybind11/embed.h>

#include <optional>
#include <limits>
#include <deque>
#ifdef RG_CUDA_SUPPORT
#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)
#include <c10/hip/HIPCachingAllocator.h>
#include <c10/hip/HIPStream.h>
#include <c10/hip/HIPGuard.h>
#else
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#endif
// libtorch's ROCm build masquerades as CUDA: even the c10/hip/* headers declare
// these under c10::cuda (matching the c10::cuda::CUDACachingAllocator call below),
// so the same spelling works on both backends.
namespace c10_gpu = c10::cuda;
using GpuStream = c10::cuda::CUDAStream;
using GpuStreamGuard = c10::cuda::CUDAStreamGuard;
#endif
#include <private/GigaLearnCPP/PPO/ExperienceBuffer.h>
#include <private/GigaLearnCPP/PPO/GAE.h>
#include <private/GigaLearnCPP/PolicyVersionManager.h>
#include <private/GigaLearnCPP/EvolutionStrategy.h>

#include "Util/KeyPressDetector.h"
#include <private/GigaLearnCPP/Util/WelfordStat.h>
#include <private/GigaLearnCPP/Util/EMAHistogram.h>
#include <private/GigaLearnCPP/PPO/GCRLOptionality.h>
#include "Util/AvgTracker.h"

using namespace RLGC;

namespace {
	constexpr int GCRL_GATE_GOAL_DIM = 6;

	// targetVel is passed in (not read from config) because Feature C can adapt it at
	// runtime; when useAdaptiveGateTargetVel is off the Learner's member never moves
	// from the config seed, so the rows are bit-identical to the old behavior.
	torch::Tensor MakeGCRLTerminalTargetRow(
		bool ownGoal,
		const GGL::LearnerConfig& config,
		GGL::BatchedWelfordStat* obsStat,
		float targetVel
	) {
		// AdvancedObs inverts orange-player states into blue perspective, so these
		// opponent/own targets are already correct for every player's obs row.
		Vec pos = ownGoal ? CommonValues::BLUE_GOAL_BACK : CommonValues::ORANGE_GOAL_BACK;
		float yVel = ownGoal ? -targetVel : targetVel;
		float vals[GCRL_GATE_GOAL_DIM] = {
			pos.x * AdvancedObs::POS_COEF,
			pos.y * AdvancedObs::POS_COEF,
			pos.z * AdvancedObs::POS_COEF,
			0.0f,
			yVel * AdvancedObs::VEL_COEF,
			0.0f
		};

		if (config.standardizeObs && obsStat) {
			const auto& meanVec = obsStat->GetMean();
			auto stdVec = obsStat->GetSTD();
			for (int d = 0; d < GCRL_GATE_GOAL_DIM; d++) {
				float offset = RS_CLAMP((float)meanVec[d], -config.maxObsMeanRange, config.maxObsMeanRange);
				float invStd = 1.0f / RS_MAX((float)stdVec[d], config.minObsSTD);
				vals[d] = (vals[d] - offset) * invStd;
			}
		}

		auto targets = torch::empty({ 1, GCRL_GATE_GOAL_DIM }, torch::TensorOptions().dtype(torch::kFloat32));
		for (int d = 0; d < GCRL_GATE_GOAL_DIM; d++)
			targets.slice(1, d, d + 1).fill_(vals[d]);
		return targets;
	}

	torch::Tensor InferGCRLTerminalScoresBatched(
		GGL::PPOLearner* ppo,
		torch::Tensor states,
		torch::Tensor actionComps,
		torch::Tensor goalTargetRow,
		torch::Tensor antiTargetRow,
		int length
	) {
		if (ppo->device.is_cpu()) {
			return ppo->InferGCRLTerminalScores(
				states.to(ppo->device, RG_H2D_NONBLOCKING(ppo->device), true),
				actionComps.to(ppo->device, RG_H2D_NONBLOCKING(ppo->device), true),
				goalTargetRow.expand({ length, GCRL_GATE_GOAL_DIM }).to(ppo->device, RG_H2D_NONBLOCKING(ppo->device), true),
				antiTargetRow.expand({ length, GCRL_GATE_GOAL_DIM }).to(ppo->device, RG_H2D_NONBLOCKING(ppo->device), true)
			).cpu();
		}

		torch::Tensor goalTargetDev = goalTargetRow.to(ppo->device, RG_H2D_NONBLOCKING(ppo->device), true);
		torch::Tensor antiTargetDev = antiTargetRow.to(ppo->device, RG_H2D_NONBLOCKING(ppo->device), true);
		torch::Tensor terminalScores = torch::zeros({ (int64_t)length, 2 });
		for (int i = 0; i < length; i += ppo->config.miniBatchSize) {
			int start = i;
			int end = RS_MIN(i + ppo->config.miniBatchSize, length);
			int batchSize = end - start;
			auto scorePart = ppo->InferGCRLTerminalScores(
				states.slice(0, start, end).to(ppo->device, RG_H2D_NONBLOCKING(ppo->device), true),
				actionComps.slice(0, start, end).to(ppo->device, RG_H2D_NONBLOCKING(ppo->device), true),
				goalTargetDev.expand({ batchSize, GCRL_GATE_GOAL_DIM }),
				antiTargetDev.expand({ batchSize, GCRL_GATE_GOAL_DIM })
			).cpu();
			terminalScores.slice(0, start, end).copy_(scorePart, true);
		}
		return terminalScores;
	}

	// ── Feature A: frontier resets ──
	// Consistency probe step: d(s_t, g_t) vs d(s_{t+k}, g_t) along the trajectory. Small
	// so the comparison stays within the same play; the magnitude of violations is what
	// gets percentile-thresholded, so the exact k is not load-bearing.
	constexpr int FRONTIER_CONSISTENCY_STEP = 4;
	// Per-iteration insert cap so one pathological batch can't flush the whole ring
	// (ring 4096 / cap 1024 -> worst-case full turnover every 4 iterations).
	constexpr int FRONTIER_MAX_INSERTS_PER_ITER = 1024;

	// Full-physics reset candidate from a live game state; null if any car is demoed
	// (cheap pre-filter; the geometric sanity filter runs at harvest time).
	std::shared_ptr<const RLGC::FrontierSnapshot> MakeFrontierSnapshot(const RLGC::GameState& gs, int gameMode) {
		for (auto& player : gs.players)
			if (player.isDemoed)
				return nullptr;

		auto snap = std::make_shared<RLGC::FrontierSnapshot>();
		snap->ball = gs.ball;
		snap->cars.reserve(gs.players.size());
		for (auto& player : gs.players) {
			RLGC::FrontierCarSnapshot cs = {};
			cs.pos = player.pos;
			cs.vel = player.vel;
			cs.angVel = player.angVel;
			cs.rotMat = player.rotMat;
			cs.boost = player.boost;
			cs.team = player.team;
			snap->cars.push_back(cs);
		}
		snap->padsActive = gs.boostPads;
		snap->padTimers = gs.boostPadTimers;
		snap->gameMode = gameMode;
		return snap;
	}

	// ── Feature C: adaptive target histograms ──
	// 50 bins over [0, 6000] (ball speeds and hit forces both live well inside this),
	// decay 0.999/iter ≈ a ~1000-iteration estimation horizon — much slower than the
	// policy, as the slew-limited targets require.
	constexpr int ADAPTIVE_HIST_BINS = 50;
	constexpr float ADAPTIVE_HIST_MAX = 6000.0f;
	constexpr double ADAPTIVE_HIST_DECAY = 0.999;

	// ── Feature D: optionality stratum flags (per step, collector side) ──
	constexpr uint8_t OPT_FLAG_DEFENSIVE = 1, OPT_FLAG_RESOURCE = 2, OPT_FLAG_TOUCH = 4;
	// Defensive-region geometry: ball deeper toward own net than the car AND in the
	// defensive third, or car within this radius of its own net.
	constexpr float OPT_DEFENSIVE_THIRD_Y = 2500.0f, OPT_OWN_NET_RADIUS = 1500.0f;
	// Per-stratum candidate cap handed to the bank each iteration (the bank only takes
	// ~refreshFrac*cap anyway; this just bounds the sampling work).
	constexpr int OPT_MAX_CANDS_PER_STRATUM = 256;

	// World-space ball state -> obs-space 6-dim goal row (same POS/VEL_COEF +
	// standardization path as MakeGCRLTerminalTargetRow). invert = orange perspective
	// (AdvancedObs rotates orange players' world 180°: x and y negate).
	void MakeObsSpaceBallRow(
		const BallState& bs, bool invert,
		const GGL::LearnerConfig& config, GGL::BatchedWelfordStat* obsStat,
		float* out6
	) {
		float sgn = invert ? -1.0f : 1.0f;
		out6[0] = sgn * bs.pos.x * AdvancedObs::POS_COEF;
		out6[1] = sgn * bs.pos.y * AdvancedObs::POS_COEF;
		out6[2] = bs.pos.z * AdvancedObs::POS_COEF;
		out6[3] = sgn * bs.vel.x * AdvancedObs::VEL_COEF;
		out6[4] = sgn * bs.vel.y * AdvancedObs::VEL_COEF;
		out6[5] = bs.vel.z * AdvancedObs::VEL_COEF;

		if (config.standardizeObs && obsStat) {
			const auto& meanVec = obsStat->GetMean();
			auto stdVec = obsStat->GetSTD();
			for (int d = 0; d < 6; d++) {
				float offset = RS_CLAMP((float)meanVec[d], -config.maxObsMeanRange, config.maxObsMeanRange);
				float invStd = 1.0f / RS_MAX((float)stdVec[d], config.minObsSTD);
				out6[d] = (out6[d] - offset) * invStd;
			}
		}
	}
}

GGL::Learner::Learner(EnvCreateFn envCreateFn, LearnerConfig config, StepCallbackFn stepCallback) :
	envCreateFn(envCreateFn), config(config), stepCallback(stepCallback)
{
	pybind11::initialize_interpreter();

#ifndef NDEBUG
	RG_LOG("===========================");
	RG_LOG("WARNING: GigaLearn runs extremely slowly in debug, and there are often bizzare issues with debug-mode torch.");
	RG_LOG("It is recommended that you compile in release mode without optimization for debugging.");
	RG_SLEEP(1000);
#endif

	if (config.tsPerSave == 0)
		config.tsPerSave = config.ppo.tsPerItr;

	RG_LOG("Learner::Learner():");

	if (config.randomSeed == -1)
		config.randomSeed = RS_CUR_MS();

	RG_LOG("\tCheckpoint Save/Load Dir: " << config.checkpointFolder);

	torch::manual_seed(config.randomSeed);

	at::Device device = at::Device(at::kCPU);
	if (
		config.deviceType == LearnerDeviceType::GPU_CUDA ||
		(config.deviceType == LearnerDeviceType::AUTO && torch::cuda::is_available())
		) {
		RG_LOG("\tUsing CUDA GPU device...");

		// Test out moving a tensor to GPU and back to make sure the device is working
		torch::Tensor t;
		bool deviceTestFailed = false;
		try {
			t = torch::tensor(0);
			t = t.to(at::Device(at::kCUDA));
			t = t.cpu();
		} catch (...) {
			deviceTestFailed = true;
		}

		if (!torch::cuda::is_available() || deviceTestFailed)
			RG_ERR_CLOSE(
				"Learner::Learner(): Can't use CUDA GPU because " <<
				(at::hasMPS() ? "libtorch cannot access the GPU" : "CUDA is not available to libtorch") << ".\n" <<
				"Make sure your libtorch comes with CUDA support, and that CUDA is installed properly."
			)
		device = at::Device(at::kCUDA);
	} else if (
		config.deviceType == LearnerDeviceType::GPU_MPS ||
		(config.deviceType == LearnerDeviceType::AUTO && at::hasMPS())
		) {
		RG_LOG("\tUsing MPS GPU device...");

		// Test out moving a tensor to GPU and back to make sure the device is working
		torch::Tensor t;
		bool deviceTestFailed = false;
		try {
			t = torch::tensor(0);
			t = t.to(at::Device(at::kMPS));
			t = t.cpu();
		} catch (...) {
			deviceTestFailed = true;
		}

		if (!at::hasMPS() || deviceTestFailed)
			RG_ERR_CLOSE(
				"Learner::Learner(): Can't use MPS GPU because " <<
				(at::hasMPS() ? "libtorch cannot access the GPU" : "MPS is not available to libtorch") << ".\n" <<
				"Make sure your libtorch comes with MPS support, and that MPS is installed properly."
			)
		device = at::Device(at::kMPS);
	} else {
		RG_LOG("\tUsing CPU device...");
		device = at::Device(at::kCPU);
	}

	if (RocketSim::GetStage() != RocketSimStage::INITIALIZED) {
		RG_LOG("\tInitializing RocketSim...");
		RocketSim::Init("collision_meshes", true);
	}

	{
		RG_LOG("\tCreating envs...");
		EnvSetConfig envSetConfig = {};
		envSetConfig.envCreateFn = envCreateFn;
		envSetConfig.numArenas = config.renderMode ? 1 : config.numGames;
		envSetConfig.tickSkip = config.tickSkip;
		envSetConfig.actionDelay = config.actionDelay;
		envSetConfig.saveRewards = config.addRewardsToMetrics;
		envSet = new RLGC::EnvSet(envSetConfig);
		obsSize = envSet->state.obs.size[1];
		numActions = envSet->actionParsers[0]->GetActionAmount();
	}

	{
		// Map each arena's (gameMode, playersPerTeam) to a dense mode id; per-mode
		// normalization (returns, GCRL advantages, gate deltas) groups by these ids
		std::map<std::pair<int, int>, int> modeKeyToId = {};
		playerModeIds.resize(envSet->state.numPlayers);
		for (int arenaIdx = 0; arenaIdx < envSet->arenas.size(); arenaIdx++) {
			Arena* arena = envSet->arenas[arenaIdx];
			auto& gs = envSet->state.gameStates[arenaIdx];

			int teamCounts[2] = { 0, 0 };
			for (auto& player : gs.players)
				teamCounts[(int)player.team]++;

			auto key = std::make_pair((int)arena->gameMode, RS_MAX(teamCounts[0], teamCounts[1]));
			auto itr = modeKeyToId.find(key);
			int modeId;
			if (itr != modeKeyToId.end()) {
				modeId = itr->second;
			} else {
				modeId = (int)modeNames.size();
				modeKeyToId[key] = modeId;
				const char* modeStr = (key.first < 4) ? GAMEMODE_STRS[key.first] : "unknown";
				modeNames.push_back(RS_STR(modeStr << "_" << teamCounts[0] << "v" << teamCounts[1]));
			}

			int startIdx = envSet->state.arenaPlayerStartIdx[arenaIdx];
			for (int i = 0; i < (int)gs.players.size(); i++)
				playerModeIds[startIdx + i] = modeId;
		}
		numModes = (int)modeNames.size();

		RG_LOG("\tGame modes (" << numModes << "):");
		for (int m = 0; m < numModes; m++)
			RG_LOG("\t > [" << m << "] " << modeNames[m]);

		gateDeltaMeanEMA.assign(numModes, 0.0);
		gateDeltaVarEMA.assign(numModes, 1.0);
	}

	{
		if (config.standardizeReturns) {
			this->returnStats = new std::vector<WelfordStat>(numModes);
		} else {
			this->returnStats = NULL;
		}

		if (config.standardizeObs) {
			this->obsStat = new BatchedWelfordStat(obsSize);
		} else {
			this->obsStat = NULL;
		}
	}

	try {
		RG_LOG("\tMaking PPO learner...");
		ppo = new PPOLearner(obsSize, numActions, config.ppo, device);
	} catch (std::exception& e) {
		RG_ERR_CLOSE("Failed to create PPO learner: " << e.what());
	}
	ppo->numModes = numModes;

	// Feature C: seed the adaptive targets at today's hand-tuned constants. The seed is
	// the initial condition, not a permanent constant — but a cold or sparse histogram
	// can never move it (min-samples hold), and LoadStats() below restores the ratcheted
	// values across restarts. The StrongTouch floor seeds from the user atomic's initial
	// value, which IS the reward's constructor threshold.
	adaptiveGateTargetVel = config.ppo.gcrlRewardGateTargetVel;
	// KPHToVel(20): RLGC::Math isn't usable in this TU (it aliases RocketSim::Math via the
	// namespace using-directive), so apply the conversion constant (250/9 uu/s per kph).
	adaptiveStrongTouchFloor = config.ppo.adaptiveStrongTouchFloorAtomic
		? (double)config.ppo.adaptiveStrongTouchFloorAtomic->load(std::memory_order_relaxed)
		: (20.0 * (250.0 / 9.0));
	if (config.ppo.useAdaptiveGateTargetVel)
		contactBallSpeedHist = new EMAHistogram(ADAPTIVE_HIST_BINS, ADAPTIVE_HIST_MAX, ADAPTIVE_HIST_DECAY);
	if (config.ppo.useAdaptiveStrongTouchFloor)
		contactHitForceHist = new EMAHistogram(ADAPTIVE_HIST_BINS, ADAPTIVE_HIST_MAX, ADAPTIVE_HIST_DECAY);

	if (config.renderMode) {
		renderSender = new RenderSender(config.renderTimeScale);
	} else {
		renderSender = NULL;
	}

	if (config.skillTracker.enabled || config.trainAgainstOldVersions)
		config.savePolicyVersions = true;

	if (config.savePolicyVersions && !config.renderMode) {
		if (config.checkpointFolder.empty())
			RG_ERR_CLOSE("Cannot save/load old policy versions with no checkpoint save folder");
		versionMgr = new PolicyVersionManager(
			config.checkpointFolder / "policy_versions", config.maxOldVersions, config.tsPerVersion,
			config.skillTracker, envSet->config
		);
	} else {
		versionMgr = NULL;
	}

	if (config.evolutionStrategy.enabled && !config.renderMode) {
		esManager = new EvolutionStrategy(
			config.evolutionStrategy, envSet->config,
			obsStat, config.minObsSTD, config.maxObsMeanRange);
	} else {
		esManager = NULL;
	}

	if (!config.checkpointFolder.empty())
		Load();

	if (config.savePolicyVersions && !config.renderMode) {
		if (config.checkpointFolder.empty())
			RG_ERR_CLOSE("Cannot save/load old policy versions with no checkpoint save folder");
		auto models = ppo->GetPolicyModels();
		versionMgr->LoadVersions(models, totalTimesteps);
	}

	if (config.sendMetrics && !config.renderMode) {
		if (!runID.empty())
			RG_LOG("\tRun ID: " << runID);
		metricSender = new MetricSender(config.metricsProjectName, config.metricsGroupName, config.metricsRunName, runID);
	} else {
		metricSender = NULL;
	}

	RG_LOG(RG_DIVIDER);
}

void GGL::Learner::SaveStats(std::filesystem::path path) {
	using namespace nlohmann;

	constexpr const char* ERROR_PREFIX = "Learner::SaveStats(): ";

	std::ofstream fOut(path);
	if (!fOut.good())
		RG_ERR_CLOSE(ERROR_PREFIX << "Can't open file at " << path);

	json j = {};
	j["total_timesteps"] = totalTimesteps;
	j["total_iterations"] = totalIterations;
	if (gcrlAdvScaleAnnealStartTS != UINT64_MAX)
		j["gcrl_adv_scale_anneal_start_ts"] = gcrlAdvScaleAnnealStartTS;
	if (gcrlRewardGateAnnealStartTS != UINT64_MAX)
		j["gcrl_reward_gate_anneal_start_ts"] = gcrlRewardGateAnnealStartTS;
	if (gcrlAerialRewardGateAnnealStartTS != UINT64_MAX)
		j["gcrl_aerial_reward_gate_anneal_start_ts"] = gcrlAerialRewardGateAnnealStartTS;
	if (curriculumRewardAnnealStartTS != UINT64_MAX)
		j["curriculum_reward_anneal_start_ts"] = curriculumRewardAnnealStartTS;
	if (aerialCurriculumRewardAnnealStartTS != UINT64_MAX)
		j["aerial_curriculum_reward_anneal_start_ts"] = aerialCurriculumRewardAnnealStartTS;
	if (sorsRewardScaleAnnealStartTS != UINT64_MAX)
		j["sors_reward_scale_anneal_start_ts"] = sorsRewardScaleAnnealStartTS;
	j["curriculum_anneal_progress_ts"] = curriculumAnnealProgressTS;
	j["aerial_curriculum_anneal_progress_ts"] = aerialCurriculumAnnealProgressTS;
	j["gcrl_reward_gate_anneal_progress_ts"] = gcrlRewardGateAnnealProgressTS;
	j["gcrl_aerial_reward_gate_anneal_progress_ts"] = gcrlAerialRewardGateAnnealProgressTS;
	j["touch_ratio_ema"] = touchRatioEMA;
	j["high_air_touch_ratio_ema"] = highAirTouchRatioEMA;

	// Feature C: persist the ratcheted targets + histograms so the crash-restart loop
	// doesn't re-seed to the hand value and replay the slew climb (~35 iters per 2x at
	// 2%/iter). Flag-off runs write nothing new.
	if (config.ppo.useAdaptiveGateTargetVel || config.ppo.useAdaptiveStrongTouchFloor) {
		j["adaptive_gate_target_vel"] = adaptiveGateTargetVel;
		j["adaptive_strong_touch_floor"] = adaptiveStrongTouchFloor;
		if (contactBallSpeedHist)
			j["contact_ball_speed_hist"] = contactBallSpeedHist->ToJSON();
		if (contactHitForceHist)
			j["contact_hit_force_hist"] = contactHitForceHist->ToJSON();
	}

	// Feature D: persist the rOpt normalizer + burn-in + anneal progress (the goal bank
	// and target nets are deliberately NOT checkpointed — they refill/re-sync on restart).
	if (config.ppo.useOptionality) {
		j["opt_reward_mean_ema"] = optRewardMeanEMA;
		j["opt_reward_var_ema"] = optRewardVarEMA;
		j["opt_burn_in_iters_done"] = optBurnInItersDone;
		if (optWeightAnnealStartTS != UINT64_MAX)
			j["opt_weight_anneal_start_ts"] = optWeightAnnealStartTS;
		j["opt_weight_anneal_progress_ts"] = optWeightAnnealProgressTS;
	}

	if (config.sendMetrics)
		j["run_id"] = metricSender->curRunID;

	if (returnStats) {
		json returnStatsJ = {};
		for (int m = 0; m < numModes; m++)
			returnStatsJ[modeNames[m]] = (*returnStats)[m].ToJSON();
		j["return_stats_per_mode"] = returnStatsJ;
	}
	if (obsStat)
		j["obs_stat"] = obsStat->ToJSON();

	{
		json gateEMAJ = {};
		for (int m = 0; m < numModes; m++)
			gateEMAJ[modeNames[m]] = { { "mean", gateDeltaMeanEMA[m] }, { "var", gateDeltaVarEMA[m] } };
		j["gcrl_gate_delta_ema"] = gateEMAJ;
	}

	if (versionMgr)
		versionMgr->AddRunningStatsToJSON(j);

	std::string jStr = j.dump(4);
	fOut << jStr;
}

void GGL::Learner::LoadStats(std::filesystem::path path) {
	// TODO: Repetitive code, merge repeated code into one function called from both SaveStats() and LoadStats()

	using namespace nlohmann;
	constexpr const char* ERROR_PREFIX = "Learner::LoadStats(): ";

	std::ifstream fIn(path);
	if (!fIn.good())
		RG_ERR_CLOSE(ERROR_PREFIX << "Can't open file at " << path);

	json j = json::parse(fIn);
	totalTimesteps = j["total_timesteps"];
	totalIterations = j["total_iterations"];
	if (j.contains("gcrl_adv_scale_anneal_start_ts"))
		gcrlAdvScaleAnnealStartTS = j["gcrl_adv_scale_anneal_start_ts"];
	if (j.contains("gcrl_reward_gate_anneal_start_ts"))
		gcrlRewardGateAnnealStartTS = j["gcrl_reward_gate_anneal_start_ts"];
	if (j.contains("gcrl_aerial_reward_gate_anneal_start_ts"))
		gcrlAerialRewardGateAnnealStartTS = j["gcrl_aerial_reward_gate_anneal_start_ts"];
	if (j.contains("curriculum_reward_anneal_start_ts"))
		curriculumRewardAnnealStartTS = j["curriculum_reward_anneal_start_ts"];
	if (j.contains("aerial_curriculum_reward_anneal_start_ts"))
		aerialCurriculumRewardAnnealStartTS = j["aerial_curriculum_reward_anneal_start_ts"];
	if (j.contains("sors_reward_scale_anneal_start_ts"))
		sorsRewardScaleAnnealStartTS = j["sors_reward_scale_anneal_start_ts"];
	if (j.contains("curriculum_anneal_progress_ts"))
		curriculumAnnealProgressTS = j["curriculum_anneal_progress_ts"];
	if (j.contains("aerial_curriculum_anneal_progress_ts"))
		aerialCurriculumAnnealProgressTS = j["aerial_curriculum_anneal_progress_ts"];
	if (j.contains("gcrl_reward_gate_anneal_progress_ts"))
		gcrlRewardGateAnnealProgressTS = j["gcrl_reward_gate_anneal_progress_ts"];
	if (j.contains("gcrl_aerial_reward_gate_anneal_progress_ts"))
		gcrlAerialRewardGateAnnealProgressTS = j["gcrl_aerial_reward_gate_anneal_progress_ts"];
	if (j.contains("touch_ratio_ema"))
		touchRatioEMA = j["touch_ratio_ema"];
	if (j.contains("high_air_touch_ratio_ema"))
		highAirTouchRatioEMA = j["high_air_touch_ratio_ema"];

	if (j.contains("adaptive_gate_target_vel"))
		adaptiveGateTargetVel = j["adaptive_gate_target_vel"];
	if (j.contains("adaptive_strong_touch_floor")) {
		adaptiveStrongTouchFloor = j["adaptive_strong_touch_floor"];
		// Re-publish immediately so workers don't run the first batch on the seed value
		if (config.ppo.useAdaptiveStrongTouchFloor && config.ppo.adaptiveStrongTouchFloorAtomic)
			config.ppo.adaptiveStrongTouchFloorAtomic->store((float)adaptiveStrongTouchFloor, std::memory_order_relaxed);
	}
	if (contactBallSpeedHist && j.contains("contact_ball_speed_hist"))
		contactBallSpeedHist->ReadFromJSON(j["contact_ball_speed_hist"]);
	if (contactHitForceHist && j.contains("contact_hit_force_hist"))
		contactHitForceHist->ReadFromJSON(j["contact_hit_force_hist"]);

	if (j.contains("opt_reward_mean_ema"))
		optRewardMeanEMA = j["opt_reward_mean_ema"];
	if (j.contains("opt_reward_var_ema"))
		optRewardVarEMA = j["opt_reward_var_ema"];
	if (j.contains("opt_burn_in_iters_done"))
		optBurnInItersDone = j["opt_burn_in_iters_done"];
	if (j.contains("opt_weight_anneal_start_ts"))
		optWeightAnnealStartTS = j["opt_weight_anneal_start_ts"];
	if (j.contains("opt_weight_anneal_progress_ts"))
		optWeightAnnealProgressTS = j["opt_weight_anneal_progress_ts"];

	if (j.contains("run_id"))
		runID = j["run_id"];

	if (returnStats && j.contains("return_stats_per_mode")) {
		const json& returnStatsJ = j["return_stats_per_mode"];
		for (int m = 0; m < numModes; m++)
			if (returnStatsJ.contains(modeNames[m]))
				(*returnStats)[m].ReadFromJSON(returnStatsJ[modeNames[m]]);
	}
	if (obsStat)
		obsStat->ReadFromJSON(j["obs_stat"]);

	if (j.contains("gcrl_gate_delta_ema")) {
		const json& gateEMAJ = j["gcrl_gate_delta_ema"];
		for (int m = 0; m < numModes; m++) {
			if (gateEMAJ.contains(modeNames[m])) {
				gateDeltaMeanEMA[m] = gateEMAJ[modeNames[m]]["mean"];
				gateDeltaVarEMA[m] = gateEMAJ[modeNames[m]]["var"];
			}
		}
	}

	if (versionMgr)
		versionMgr->LoadRunningStatsFromJSON(j);
}

void GGL::Learner::DebugScoreOptionality(const std::string& path) {
	if (!config.ppo.useOptionality || !ppo->optionality)
		RG_ERR_CLOSE("Learner::DebugScoreOptionality(): requires config.ppo.useOptionality");
	ppo->optionality->DebugScoreStates(path, obsSize);
}

// Different than RLGym-PPO to show that they are not compatible
constexpr const char* STATS_FILE_NAME = "RUNNING_STATS.json";

void GGL::Learner::Save() {
	if (config.checkpointFolder.empty())
		RG_ERR_CLOSE("Learner::Save(): Cannot save because config.checkpointSaveFolder is not set");

	std::filesystem::path saveFolder = config.checkpointFolder / std::to_string(totalTimesteps);
	std::filesystem::create_directories(saveFolder);

	RG_LOG("Saving to folder " << saveFolder << "...");
	SaveStats(saveFolder / STATS_FILE_NAME);
	ppo->SaveTo(saveFolder);

	// Remove old checkpoints
	if (config.checkpointsToKeep != -1) {
		std::set<int64_t> allSavedTimesteps = Utils::FindNumberedDirs(config.checkpointFolder);
		while (allSavedTimesteps.size() > config.checkpointsToKeep) {
			int64_t lowestCheckpointTS = INT64_MAX;
			for (int64_t savedTimesteps : allSavedTimesteps)
				lowestCheckpointTS = RS_MIN(lowestCheckpointTS, savedTimesteps);

			std::filesystem::path removePath = config.checkpointFolder / std::to_string(lowestCheckpointTS);
			try {
				std::filesystem::remove_all(removePath);
			} catch (std::exception& e) {
				RG_ERR_CLOSE("Failed to remove old checkpoint from " << removePath << ", exception: " << e.what());
			}
			allSavedTimesteps.erase(lowestCheckpointTS);
		}
	}

	if (versionMgr)
		versionMgr->SaveVersions();

	RG_LOG(" > Done.");
}

void GGL::Learner::Load() {
	if (config.checkpointFolder.empty())
		RG_ERR_CLOSE("Learner::Load(): Cannot load because config.checkpointLoadFolder is not set");

	RG_LOG("Loading most recent checkpoint in " << config.checkpointFolder << "...");

	int64_t highest = -1;
	std::set<int64_t> allSavedTimesteps = Utils::FindNumberedDirs(config.checkpointFolder);
	for (int64_t timesteps : allSavedTimesteps)
		highest = RS_MAX(timesteps, highest);

	if (highest != -1) {
		std::filesystem::path loadFolder = config.checkpointFolder / std::to_string(highest);
		RG_LOG(" > Loading checkpoint " << loadFolder << "...");
		LoadStats(loadFolder / STATS_FILE_NAME);
		ppo->LoadFrom(loadFolder);
		RG_LOG(" > Done.");
	} else {
		RG_LOG(" > No checkpoints found, starting new model.")
	}
}

void GGL::Learner::StartQuitKeyThread(std::atomic<bool>& quitPressed, std::thread& outThread) {
	quitPressed = false;

	RG_LOG("Press 'Q' to save and quit!");
	outThread = std::thread(
		[&] {
			while (true) {
				char c = toupper(KeyPressDetector::GetPressedChar());
				if (c == 'Q') {
					RG_LOG("Save queued, will save and exit next iteration.");
					quitPressed = true;
				}
			}
		}
	);

	outThread.detach();
}
void GGL::Learner::StartTransferLearn(const TransferLearnConfig& tlConfig) {

	RG_LOG("Starting transfer learning...");

	// TODO: Lots of manual obs builder stuff going on which is quite volatile
	//	Although I can't really think another way to do this

	std::vector<ObsBuilder*> oldObsBuilders = {};
	for (int i = 0; i < envSet->arenas.size(); i++)
		oldObsBuilders.push_back(tlConfig.makeOldObsFn());

	// Reset all obs builders initially
	for (int i = 0; i < envSet->arenas.size(); i++)
		oldObsBuilders[i]->Reset(envSet->state.gameStates[i]);

	std::vector<ActionParser*> oldActionParsers = {};
	for (int i = 0; i < envSet->arenas.size(); i++)
		oldActionParsers.push_back(tlConfig.makeOldActFn());

	int oldNumActions = oldActionParsers[0]->GetActionAmount();

	if (oldNumActions != numActions) {
		if (!tlConfig.mapActsFn) {
			RG_ERR_CLOSE(
				"StartTransferLearn: Old and new action parsers have a different number of actions, but tlConfig.mapActsFn is NULL.\n" <<
				"You must implement this function to translate the action indices."
			);
		};
	}

	// Determine old obs size
	int oldObsSize;
	{
		GameState testState = envSet->state.gameStates[0];
		oldObsSize = oldObsBuilders[0]->BuildObs(testState.players[0], testState).size();
	}

	ModelSet oldModels = {};
	{
		RG_NO_GRAD;
		PPOLearner::MakeModels(false, oldObsSize, oldNumActions, tlConfig.oldSharedHeadConfig, tlConfig.oldPolicyConfig, {}, ppo->device, oldModels);

		oldModels.Load(tlConfig.oldModelsPath, false, false);
	}

	try {
		std::atomic<bool> saveQueued;
		std::thread keyPressThread;
		StartQuitKeyThread(saveQueued, keyPressThread);

		while (true) {
			Report report = {};

			// Collect obs
			std::vector<float> allNewObs = {};
			std::vector<float> allOldObs = {};
			std::vector<uint8_t> allNewActionMasks = {};
			std::vector<uint8_t> allOldActionMasks = {};
			std::vector<int> allActionMaps = {};
			int stepsCollected;
			{
				RG_NO_GRAD;
				for (stepsCollected = 0; stepsCollected < tlConfig.batchSize; stepsCollected += envSet->state.numPlayers) {

					auto terminals = envSet->state.terminals; // Backup
					envSet->Reset();
					for (int i = 0; i < envSet->arenas.size(); i++) // Manually reset old obs builders
						if (terminals[i])
							oldObsBuilders[i]->Reset(envSet->state.gameStates[i]);

					torch::Tensor tActions, tLogProbs;
					torch::Tensor tStates = DIMLIST2_TO_TENSOR<float>(envSet->state.obs);
					torch::Tensor tActionMasks = DIMLIST2_TO_TENSOR<uint8_t>(envSet->state.actionMasks);

					// Collect new and old obs BEFORE StepFirstHalf: it mutates the game states
					// on worker threads (ResetBeforeStep + arena step), so reading them after
					// would be a data race.
					allNewObs += envSet->state.obs.data;
					allNewActionMasks += envSet->state.actionMasks.data;

					// Run all old obs and old action parser on each player
					// TODO: Could be multithreaded
					for (int arenaIdx = 0; arenaIdx < envSet->arenas.size(); arenaIdx++) {
						auto& gs = envSet->state.gameStates[arenaIdx];
						for (auto& player : gs.players) {
							allOldObs += oldObsBuilders[arenaIdx]->BuildObs(player, gs);
							allOldActionMasks += oldActionParsers[arenaIdx]->GetActionMask(player, gs);

							if (tlConfig.mapActsFn) {
								auto curMap = tlConfig.mapActsFn(player, gs);
								if (curMap.size() != numActions)
									RG_ERR_CLOSE("StartTransferLearn: Your action map must have the same size as the new action parser's actions");
								allActionMaps += curMap;
							}
						}
					}

					envSet->StepFirstHalf(true);

					ppo->InferActions(
						tStates.to(ppo->device, RG_H2D_NONBLOCKING(ppo->device)), tActionMasks.to(ppo->device, RG_H2D_NONBLOCKING(ppo->device)),
						&tActions, &tLogProbs
					);

					auto curActions = TENSOR_TO_VEC<int>(tActions);

					envSet->Sync();
					envSet->StepSecondHalf(curActions, false);

					if (stepCallback)
						stepCallback(this, envSet->state.gameStates, report);
				}
			}

			uint64_t prevTimesteps = totalTimesteps;
			totalTimesteps += stepsCollected;
			report["Total Timesteps"] = totalTimesteps;
			report["Collected Timesteps"] = stepsCollected;
			totalIterations++;
			report["Total Iterations"] = totalIterations;

			// Make tensors
			torch::Tensor tNewObs = torch::tensor(allNewObs).reshape({ -1, obsSize }).to(ppo->device);
			torch::Tensor tOldObs = torch::tensor(allOldObs).reshape({ -1, oldObsSize }).to(ppo->device);
			torch::Tensor tNewActionMasks = torch::tensor(allNewActionMasks).reshape({ -1, numActions }).to(ppo->device);
			torch::Tensor tOldActionMasks = torch::tensor(allOldActionMasks).reshape({ -1, oldNumActions }).to(ppo->device);

			torch::Tensor tActionMaps = {};
			if (!allActionMaps.empty())
				tActionMaps = torch::tensor(allActionMaps).reshape({ -1, numActions }).to(ppo->device);

			// Transfer learn
			ppo->TransferLearn(oldModels, tNewObs, tOldObs, tNewActionMasks, tOldActionMasks, tActionMaps, report, tlConfig);

			if (versionMgr)
				versionMgr->OnIteration(ppo, report, totalTimesteps, prevTimesteps);

			if (saveQueued) {
				if (!config.checkpointFolder.empty())
					Save();
				exit(0);
			}

			if (!config.checkpointFolder.empty()) {
				if (totalTimesteps / config.tsPerSave > prevTimesteps / config.tsPerSave) {
					// Auto-save
					Save();
				}
			}

			report.Finish();

			if (metricSender)
				metricSender->Send(report);

			report.Display(
				{
					"Transfer Learn Accuracy",
					"Transfer Learn Loss",
					"",
					"Policy Entropy",
					"Old Policy Entropy",
					"Policy Update Magnitude",
					"",
					"Collected Timesteps",
					"Total Timesteps",
					"Total Iterations"
				}
			);
		}

	} catch (std::exception& e) {
		RG_ERR_CLOSE("Exception thrown during transfer learn loop: " << e.what());
	}
}

void GGL::Learner::Start() {

	bool render = config.renderMode;

	RG_LOG("Learner::Start():");
	RG_LOG("\tObs size: " << obsSize);
	RG_LOG("\tAction amount: " << numActions);

	if (render)
		RG_LOG("\t(Render mode enabled)");
	if (!render && config.ppo.deterministic)
		RG_ERR_CLOSE("Learner::Start(): PPO deterministic mode cannot be used for training because sampled action log-probs are required.");

	try {
		std::atomic<bool> saveQueued;
		std::thread keyPressThread;
		StartQuitKeyThread(saveQueued, keyPressThread);

		ExperienceBuffer experience = ExperienceBuffer(config.randomSeed, torch::kCPU);

		int numPlayers = envSet->state.numPlayers;
		bool useActionComps = config.ppo.useGCRL || config.ppo.useSORS || config.ppo.useGCRLRewardGate;

		// Car-local ball offset for GCRL carFutureGoals — depends on obs builder layout.
		int carLocalBallOffset = envSet->obsBuilders[0]->GetCarLocalBallOffset();

		std::vector<int> playerArenaIdx(numPlayers), playerLocalIdx(numPlayers);
		for (int arenaIdx = 0; arenaIdx < envSet->arenas.size(); arenaIdx++) {
			int start = envSet->state.arenaPlayerStartIdx[arenaIdx];
			int count = envSet->state.gameStates[arenaIdx].players.size();
			for (int i = 0; i < count; i++) {
				playerArenaIdx[start + i] = arenaIdx;
				playerLocalIdx[start + i] = i;
			}
		}

		struct Trajectory {
			FList states, nextStates, rewards, gcrlGatedRewards, curriculumRewards, aerialGCRLGatedRewards, aerialCurriculumRewards, logProbs;
			// GCRL: 8-dim action components (per step) and hindsight-relabeled future
			// ball goals (global / car-local), filled at episode end.
			FList actionComps, futureGoals, carFutureGoals;
			std::vector<SORSStep> sorsSteps;
			std::vector<uint8_t> actionMasks;
			std::vector<int8_t> terminals;
			std::vector<int32_t> actions;
			// Per-step game mode id (for per-mode normalization) and obs x-mirror flag
			// (for mirror-consistent hindsight goal relabeling).
			std::vector<int8_t> modeIds;
			std::vector<uint8_t> mirrored;
			// Frontier curriculum (Feature A): per-step shared physics snapshot — null on
			// non-sampled steps, shared between all players of an arena. Empty when off.
			std::vector<std::shared_ptr<const RLGC::FrontierSnapshot>> snapshots;
			// Optionality (Feature D): per-step stratum flags (OPT_FLAG_*). Empty when off.
			std::vector<uint8_t> stratumFlags;

			void Clear() {
				states.clear();
				nextStates.clear();
				rewards.clear();
				gcrlGatedRewards.clear();
				curriculumRewards.clear();
				aerialGCRLGatedRewards.clear();
				aerialCurriculumRewards.clear();
				logProbs.clear();
				actionComps.clear();
				futureGoals.clear();
				carFutureGoals.clear();
				sorsSteps.clear();
				actionMasks.clear();
				terminals.clear();
				actions.clear();
				modeIds.clear();
				mirrored.clear();
				snapshots.clear();
				stratumFlags.clear();
			}

			void Append(const Trajectory& other) {
				states += other.states;
				nextStates += other.nextStates;
				rewards += other.rewards;
				gcrlGatedRewards += other.gcrlGatedRewards;
				curriculumRewards += other.curriculumRewards;
				aerialGCRLGatedRewards += other.aerialGCRLGatedRewards;
				aerialCurriculumRewards += other.aerialCurriculumRewards;
				logProbs += other.logProbs;
				actionComps += other.actionComps;
				futureGoals += other.futureGoals;
				carFutureGoals += other.carFutureGoals;
				sorsSteps += other.sorsSteps;
				actionMasks += other.actionMasks;
				terminals += other.terminals;
				actions += other.actions;
				modeIds += other.modeIds;
				mirrored += other.mirrored;
				snapshots += other.snapshots;
				stratumFlags += other.stratumFlags;
			}

			size_t Length() const {
				return actions.size();
			}
		};

		// Persistent per-player trajectory state — spans episode boundaries across batches.
		// Owned exclusively by the collector (main thread in sync mode, collector thread in async).
		auto trajectories = std::vector<Trajectory>(numPlayers, Trajectory{});
		int maxEpisodeLength = (int)(config.ppo.maxEpisodeDuration * (120.f / config.tickSkip));

		// Pre-reserve to avoid repeated realloc during episodes.
		for (auto& traj : trajectories) {
			traj.states.reserve((size_t)maxEpisodeLength * obsSize);
			traj.rewards.reserve(maxEpisodeLength);
			traj.gcrlGatedRewards.reserve(maxEpisodeLength);
			traj.curriculumRewards.reserve(maxEpisodeLength);
			traj.aerialGCRLGatedRewards.reserve(maxEpisodeLength);
			traj.aerialCurriculumRewards.reserve(maxEpisodeLength);
			traj.logProbs.reserve(maxEpisodeLength);
			traj.actions.reserve(maxEpisodeLength);
			traj.terminals.reserve(maxEpisodeLength);
			traj.modeIds.reserve(maxEpisodeLength);
			traj.mirrored.reserve(maxEpisodeLength);
			traj.actionMasks.reserve((size_t)maxEpisodeLength * numActions);
			if (useActionComps)
				traj.actionComps.reserve((size_t)maxEpisodeLength * 8);
			if (config.ppo.useSORS)
				traj.sorsSteps.reserve(maxEpisodeLength);
			if (config.ppo.useFrontierResets && config.ppo.frontierBuffer)
				traj.snapshots.reserve(maxEpisodeLength);
			if (config.ppo.useOptionality)
				traj.stratumFlags.reserve(maxEpisodeLength);
		}

		// Scratch buffers reused across batches to avoid per-call realloc (owned by collector).
		std::vector<float> _normOffset(obsSize, 0.0f), _normInvStd(obsSize, 1.0f);
		FList _curActionComps;
		if (useActionComps)
			_curActionComps.resize(numPlayers * 8);

		auto fnBuildSORSWindows = [&](const Trajectory& traj) {
			std::vector<PPOLearner::SORSWindow> windows;
			if (!config.ppo.useSORS)
				return windows;

			int N = (int)traj.Length();
			if (N <= 0 || traj.sorsSteps.size() != N || traj.actionComps.size() != (size_t)N * 8)
				return windows;

			auto fnMakeWindow = [&](int center, float label) {
				int start = RS_MAX(0, center - config.ppo.sorsWindowBefore);
				int stop = RS_MIN(N, center + config.ppo.sorsWindowAfter + 1);
				int len = stop - start;
				if (len <= 0)
					return;

				PPOLearner::SORSWindow window;
				window.length = len;
				window.label = label;
				// Clone the from_blob views since the trajectory buffers are cleared at episode end
				window.states = torch::from_blob(
					(void*)(traj.states.data() + (size_t)start * obsSize),
					{ (int64_t)len, (int64_t)obsSize }, torch::TensorOptions().dtype(torch::kFloat32)).clone();
				window.actionComps = torch::from_blob(
					(void*)(traj.actionComps.data() + (size_t)start * 8),
					{ (int64_t)len, 8 }, torch::TensorOptions().dtype(torch::kFloat32)).clone();
				windows.push_back(std::move(window));
			};

			for (int t = 0; t < N; t++) {
				bool isTrigger = false;
				float label = 0;
				for (const WeightedSORSLabel& weighted : config.ppo.sorsLabels) {
					if (!weighted.label)
						continue;

					float curLabel = weighted.label->GetLabel(traj.sorsSteps, t);
					label += weighted.weight * curLabel;
					isTrigger = isTrigger || (weighted.label->IsTrigger() && curLabel != 0);
				}

				if (!isTrigger)
					continue;

				fnMakeWindow(t, label);

				int negCenter = t - config.ppo.sorsWindowBefore - config.ppo.sorsWindowAfter - 1;
				if (negCenter >= 0)
					fnMakeWindow(negCenter, 0);
			}

			return windows;
		};

		// ── Batch output ─────────────────────────────────────────────────────────
		struct CollectedBatch {
			Trajectory combinedTraj;
			Report report;
			int stepsCollected = 0;
			// Live-player touch counts for the curriculum anneal competence gates
			int touchCount = 0, highAirTouchCount = 0;
			// Per-touch achieved contact stats for the adaptive targets (Feature C):
			// resulting ball speed (gate-target analog) and ball-vel delta (StrongTouch's
			// exact hitForce formula). Empty when both flags are off.
			FList touchBallSpeeds, touchHitForces;
			// Arena resets this batch (denominator for Frontier/Reset Fraction Actual)
			int arenaResets = 0;
			float collectionTime = 0, inferTime = 0, envStepTime = 0;
		};

		// ── Collection lambda ─────────────────────────────────────────────────────
		// Fills one iteration's worth of experience into `out`.
		// Uses `inferModels` for new-policy InferActions; old-version uses its own models.
		// Takes shared_lock on versionsMutex for the duration so AddVersion/erase can't dangle the pointer.
		// ── Old-version stint state (persists across batches, owned by the collector) ──
		// The assignment is re-rolled every trainAgainstOldStintBatches batches instead of
		// every batch: each re-roll that changes a player's controller interrupts their
		// in-flight episode (the partial trajectory must be truncated at the handoff), so
		// per-batch re-rolls were interrupting half the population every ~1/chance batches
		// -- flooding training with truncated tails and switching opponents mid-episode.
		int oldStintBatchesLeft = 0;
		bool oldStintActive = false;
		uint64_t oldStintVersionTS = 0;
		Team oldStintTeam = Team::BLUE;

		// Trajectories truncated at an old-version handoff, queued for consumption.
		// Drained oldest-first into each batch under a per-batch cap, so one stint
		// boundary's flush (potentially millions of steps) is spread over several
		// iterations instead of forming a single mega-batch of stale tails.
		std::deque<Trajectory> preservedTrajs;
		size_t preservedTrajSteps = 0;

		auto fnCollectBatch = [&](ModelSet& inferModels, CollectedBatch& out) {

			GGL::PolicyVersion* oldVersion = NULL;
			std::vector<bool> oldVersionPlayerMask;
			std::vector<int> newPlayerIndices = {}, oldPlayerIndices = {};
			torch::Tensor tNewPlayerIndices, tOldPlayerIndices;

			for (int i = 0; i < numPlayers; i++)
				newPlayerIndices.push_back(i);

			// Hold shared_lock for the lifetime of oldVersion pointer (AddVersion's unique_lock will wait).
			std::shared_lock<std::shared_mutex> versionLock;
			if (config.trainAgainstOldVersions && versionMgr) {
				versionLock = std::shared_lock<std::shared_mutex>(versionMgr->versionsMutex);
				RG_ASSERT(config.trainAgainstOldChance >= 0 && config.trainAgainstOldChance <= 1);

				if (oldStintBatchesLeft <= 0) {
					// Re-roll the stint assignment
					oldStintBatchesLeft = RS_MAX(1, config.trainAgainstOldStintBatches);
					oldStintActive =
						(RocketSim::Math::RandFloat() < config.trainAgainstOldChance)
						&& !versionMgr->versions.empty()
						&& !render;
					if (oldStintActive) {
						int oldVersionIdx = RocketSim::Math::RandInt(0, versionMgr->versions.size());
						oldStintVersionTS = versionMgr->versions[oldVersionIdx].timesteps;
						oldStintTeam = Team(RocketSim::Math::RandInt(0, 2));
					}
				}
				oldStintBatchesLeft--;

				if (oldStintActive) {
					// Re-resolve the stint's version by timesteps each batch: the versions
					// vector can grow/prune between batches, so a pointer or index cached
					// from the previous batch would dangle.
					for (auto& version : versionMgr->versions) {
						if (version.timesteps == oldStintVersionTS) {
							oldVersion = &version;
							break;
						}
					}
					if (!oldVersion && !versionMgr->versions.empty()) {
						// The stint's version was pruned mid-stint; fall back to the oldest
						// remaining (versions are sorted by timesteps ascending).
						oldVersion = &versionMgr->versions.front();
						oldStintVersionTS = oldVersion->timesteps;
					}
				}

				if (oldVersion) {
					Team oldVersionTeam = oldStintTeam;

					newPlayerIndices.clear();
					oldVersionPlayerMask.resize(numPlayers);
					int i = 0;
					for (auto& state : envSet->state.gameStates) {
						for (auto& player : state.players) {
							if (player.team == oldVersionTeam) {
								oldVersionPlayerMask[i] = true;
								oldPlayerIndices.push_back(i);
							} else {
								oldVersionPlayerMask[i] = false;
								newPlayerIndices.push_back(i);
							}
							i++;
						}
					}

					tNewPlayerIndices = torch::tensor(newPlayerIndices);
					tOldPlayerIndices = torch::tensor(oldPlayerIndices);
				}
			}

			int numRealPlayers = oldVersion ? (int)newPlayerIndices.size() : envSet->state.numPlayers;
			out.stepsCollected = 0;
			out.touchCount = 0;
			out.highAirTouchCount = 0;
			out.touchBallSpeeds.clear();
			out.touchHitForces.clear();
			out.arenaResets = 0;
			out.inferTime = 0;
			out.envStepTime = 0;

			// Self-tuning curriculum collector hooks (all dead branches when off)
			bool doFrontier = config.ppo.useFrontierResets && config.ppo.frontierBuffer && config.ppo.useGCRL;
			bool doAdaptiveTargets = config.ppo.useAdaptiveGateTargetVel || config.ppo.useAdaptiveStrongTouchFloor;
			bool doOptionality = config.ppo.useOptionality && config.ppo.useGCRL;

			// Reset and pre-reserve the combined trajectory.
			auto& combinedTraj = out.combinedTraj;
			combinedTraj.Clear();
			{
				size_t expTs = config.ppo.tsPerItr + config.ppo.tsPerItr / 4;
				combinedTraj.states.reserve(expTs * obsSize);
				combinedTraj.rewards.reserve(expTs);
				combinedTraj.gcrlGatedRewards.reserve(expTs);
				combinedTraj.curriculumRewards.reserve(expTs);
				combinedTraj.aerialGCRLGatedRewards.reserve(expTs);
				combinedTraj.aerialCurriculumRewards.reserve(expTs);
				combinedTraj.logProbs.reserve(expTs);
				combinedTraj.actions.reserve(expTs);
				combinedTraj.terminals.reserve(expTs);
				combinedTraj.modeIds.reserve(expTs);
				combinedTraj.mirrored.reserve(expTs);
				combinedTraj.actionMasks.reserve(expTs * numActions);
				if (useActionComps)
					combinedTraj.actionComps.reserve(expTs * 8);
				if (config.ppo.useGCRL) {
					combinedTraj.futureGoals.reserve(expTs * 6);
					combinedTraj.carFutureGoals.reserve(expTs * 6);
				}
				if (config.ppo.useSORS)
					combinedTraj.sorsSteps.reserve(expTs);
				if (config.ppo.useFrontierResets && config.ppo.frontierBuffer)
					combinedTraj.snapshots.reserve(expTs);
				if (config.ppo.useOptionality)
					combinedTraj.stratumFlags.reserve(expTs);
			}

			auto& report = out.report;
			report = Report{};

			// Relabel a completed (or just-truncated) trajectory in place: hindsight-relabel
			// GCRL goals and build SORS windows. Callers must have pushed the final terminal
			// type (and, for truncations, the bootstrap next-state) before calling.
			auto fnRelabelTrajectory = [&](Trajectory& traj) {
				// ── GCRL hindsight relabeling: per timestep pick a future ball state as ──
				// the goal. goal/anti critics use the global ball (obs[0:6]); the car
				// critic uses the car-local ball at carLocalBallOffset (layout-dependent).
				// With useDifficultyHER (Feature B) relabeling is DEFERRED to one batched
				// learner-thread pass over the combined batch (futureGoals stay empty here;
				// they're only consumed at experience tensor creation and in Learn()).
				if (config.ppo.useGCRL && !config.ppo.useDifficultyHER) {
					int N = traj.Length();
					int H = config.ppo.gcrlHorizon;
					int minH = config.ppo.gcrlMinHorizon;
					bool hasMirrorFlags = traj.mirrored.size() == (size_t)N;
					traj.futureGoals.resize(N * 6);
					traj.carFutureGoals.resize(N * 6);
					for (int t = 0; t < N; t++) {
						// Goal is a future ball state sampled within [minH, H] steps ahead,
						// capped at the episode end -- so gcrlHorizon bounds the goal window.
						int lo = t + minH;
						int hi = RS_MIN(t + H, N - 1);
						int t_target;
						if (config.ppo.gcrlUseVariableHER && hi > lo)
							t_target = Math::RandInt(lo, hi + 1); // seeded; inclusive of hi
						else
							t_target = hi;

						// The obs x-mirror flips whenever the player crosses x=0, so a goal
						// copied from t_target's obs may be in the opposite mirror frame to
						// step t's features. Flip its lateral components back when the flags
						// differ. (With standardizeObs this negates the *normalized* value;
						// exact only if the running mean of these dims is ~0, which holds by
						// the game's left/right symmetry.)
						bool flip = hasMirrorFlags && (traj.mirrored[t] != traj.mirrored[t_target]);

						for (int d = 0; d < 6; d++)
							traj.futureGoals[t * 6 + d] = traj.states[t_target * obsSize + d];
						if (flip) {
							// Global ball: negate x of pos and vel
							traj.futureGoals[t * 6 + 0] = -traj.futureGoals[t * 6 + 0];
							traj.futureGoals[t * 6 + 3] = -traj.futureGoals[t * 6 + 3];
						}
						if (carLocalBallOffset >= 0) {
							for (int d = 0; d < 6; d++)
								traj.carFutureGoals[t * 6 + d] = traj.states[t_target * obsSize + carLocalBallOffset + d];
							if (flip) {
								// Car-local ball: negate the right-axis components
								// (RotMat::Dot returns (forward, right, up) projections)
								traj.carFutureGoals[t * 6 + 1] = -traj.carFutureGoals[t * 6 + 1];
								traj.carFutureGoals[t * 6 + 4] = -traj.carFutureGoals[t * 6 + 4];
							}
						}
					}
				}

				if (config.ppo.useSORS)
					ppo->AddSORSWindows(fnBuildSORSWindows(traj));
			};

			// Relabel, move into the combined batch, and reset (the normal episode-end path).
			auto fnFinalizeTrajectory = [&](Trajectory& traj) {
				fnRelabelTrajectory(traj);
				combinedTraj.Append(traj);
				traj.Clear();
			};

			// Players being handed to an old version mid-episode: finalize their in-flight
			// partial episodes as TRUNCATED (bootstrapping from the current obs) instead of
			// discarding them. The old behavior (Clear() at batch end) silently destroyed
			// >half of all collected experience once episodes outlived the ~8-batch mean
			// gap between old-version stints (collected/iter ran 2-2.5x tsPerItr on every
			// run since trainAgainstOldVersions landed; the good run g7jf6cwc consumed ~all
			// of it), and it biased the surviving data toward episode TAILS -- early-episode
			// play (kickoff positioning, buildup, defense) almost never reached training.
			// state.obs still holds the obs built after the previous batch's last step, which
			// is exactly the next-state these trajectories' last actions led to; players whose
			// arena terminated on that step have already-finalized (empty) trajectories.
			size_t preservedSteps = 0;
			for (int oldPlayerIdx : oldPlayerIndices) {
				auto& traj = trajectories[oldPlayerIdx];
				if (traj.Length() == 0)
					continue;

				preservedSteps += traj.Length();
				traj.terminals.back() = RLGC::TerminalType::TRUNCATED;
				envSet->state.obs.AppendRow(oldPlayerIdx, traj.nextStates);
				// Normalize the just-appended next-state with the previous iteration's stats,
				// matching the normalization its trajectory states were stored with.
				if (obsStat && !_normOffset.empty()) {
					float* ptr = traj.nextStates.data() + traj.nextStates.size() - obsSize;
					ApplyObsNorm(ptr, 1, obsSize, _normOffset, _normInvStd);
				}
				// Relabel now (the trajectory is final), then queue it for capped draining
				// below instead of dumping it all into this batch.
				fnRelabelTrajectory(traj);
				preservedTrajSteps += traj.Length();
				preservedTrajs.push_back(std::move(traj));
				traj.Clear();
			}

			// Drain the preserved queue into this batch, oldest-first (bounds staleness),
			// capped per batch so one stint boundary's flush is spread over several
			// iterations: an uncapped flush formed mega-batches of stale tails followed by
			// starved batches of short post-flush fragments, oscillating the composition of
			// every per-batch metric (gate rewards, avg reward, batch size) in a sawtooth.
			size_t drainedSteps = 0;
			{
				size_t drainBudget = (size_t)(config.ppo.tsPerItr / 2);
				while (!preservedTrajs.empty() && drainedSteps < drainBudget) {
					Trajectory& pres = preservedTrajs.front();
					drainedSteps += pres.Length();
					preservedTrajSteps -= pres.Length();
					combinedTraj.Append(pres);
					preservedTrajs.pop_front();
				}
			}

			Timer collectionTimer = {};
			{
				RG_NO_GRAD;

				// step == 0: always take at least one env step, even when the truncated
				// trajectories preserved from an old-version handoff already exceed tsPerItr
				// (otherwise stepsCollected would be 0 and total timesteps would stall).
				for (int step = 0; step == 0 || combinedTraj.Length() < config.ppo.tsPerItr || render; step++, out.stepsCollected += numRealPlayers) {
					Timer stepTimer = {};
					envSet->Reset();
					out.envStepTime += stepTimer.Elapsed();

#ifndef NDEBUG
					for (float f : envSet->state.obs.data)
						if (isnan(f) || isinf(f))
							RG_ERR_CLOSE("Obs builder produced a NaN/inf value");
#endif

					if (!render && obsStat) {
						// TODO: This samples from old versions too
						// NOTE: obsStat is also read by main thread (MakeGCRLTerminalTargetRow) — benign torn-read for approximate normalization stats.
						int numSamples = RS_MIN(envSet->state.numPlayers, config.maxObsSamples);
						for (int i = 0; i < numSamples; i++) {
							int idx = Math::RandInt(0, envSet->state.numPlayers);
							obsStat->IncrementRow(&envSet->state.obs.At(idx, 0));
						}

						// Precompute per-dim float scale/offset once (not per player), then apply.
						// Shared with the ES rollouts via ComputeObsNorm/ApplyObsNorm.
						ComputeObsNorm(obsStat, config.minObsSTD, config.maxObsMeanRange, _normOffset, _normInvStd);
						ApplyObsNorm(envSet->state.obs.data.data(), envSet->state.numPlayers, obsSize, _normOffset, _normInvStd);
					}

					torch::Tensor tActions, tLogProbs;
					torch::Tensor tStates = DIMLIST2_TO_TENSOR<float>(envSet->state.obs);
					torch::Tensor tActionMasks = DIMLIST2_TO_TENSOR<uint8_t>(envSet->state.actionMasks);

					if (!render) {
						// Frontier snapshot capture (Feature A): at most one snapshot per arena
						// per sampled step, shared between all of the arena's players. Taken
						// here — before StepFirstHalf mutates the game states — so it matches
						// the state this step's obs rows were built from (same reasoning as
						// the mirror flag below).
						std::vector<std::shared_ptr<const RLGC::FrontierSnapshot>> curArenaSnaps;
						if (doFrontier && (step % RS_MAX(1, config.ppo.frontierSnapshotInterval) == 0)) {
							curArenaSnaps.resize(envSet->arenas.size());
							for (int arenaIdx = 0; arenaIdx < (int)envSet->arenas.size(); arenaIdx++)
								curArenaSnaps[arenaIdx] = MakeFrontierSnapshot(
									envSet->state.gameStates[arenaIdx], (int)envSet->arenas[arenaIdx]->gameMode);
						}

						for (int newPlayerIdx : newPlayerIndices) {
							envSet->state.obs.AppendRow(newPlayerIdx, trajectories[newPlayerIdx].states);
							envSet->state.actionMasks.AppendRow(newPlayerIdx, trajectories[newPlayerIdx].actionMasks);
							trajectories[newPlayerIdx].modeIds.push_back((int8_t)playerModeIds[newPlayerIdx]);

							// Mirror flag must be read here, before StepFirstHalf mutates the game
							// states, so it matches the state this obs row was built from.
							int arenaIdx = playerArenaIdx[newPlayerIdx];
							const Player& player = envSet->state.gameStates[arenaIdx].players[playerLocalIdx[newPlayerIdx]];
							trajectories[newPlayerIdx].mirrored.push_back(
								envSet->obsBuilders[arenaIdx]->IsObsMirroredX(player) ? 1 : 0);

							if (doFrontier)
								trajectories[newPlayerIdx].snapshots.push_back(
									curArenaSnaps.empty() ? nullptr : curArenaSnaps[arenaIdx]);
						}
					}

					envSet->StepFirstHalf(true);

					Timer inferTimer = {};

					if (oldVersion) {
						torch::Tensor tdNewStates = tStates.index_select(0, tNewPlayerIndices).to(ppo->device, RG_H2D_NONBLOCKING(ppo->device));
						torch::Tensor tdOldStates = tStates.index_select(0, tOldPlayerIndices).to(ppo->device, RG_H2D_NONBLOCKING(ppo->device));
						torch::Tensor tdNewActionMasks = tActionMasks.index_select(0, tNewPlayerIndices).to(ppo->device, RG_H2D_NONBLOCKING(ppo->device));
						torch::Tensor tdOldActionMasks = tActionMasks.index_select(0, tOldPlayerIndices).to(ppo->device, RG_H2D_NONBLOCKING(ppo->device));

						torch::Tensor tNewActions;
						torch::Tensor tOldActions;

						ppo->InferActions(tdNewStates, tdNewActionMasks, &tNewActions, &tLogProbs, &inferModels);
						ppo->InferActions(tdOldStates, tdOldActionMasks, &tOldActions, NULL, &oldVersion->models);

						tActions = torch::zeros(numPlayers, tNewActions.dtype());
						tActions.index_copy_(0, tNewPlayerIndices, tNewActions.cpu());
						tActions.index_copy_(0, tOldPlayerIndices, tOldActions.cpu());
					} else {
						torch::Tensor tdStates = tStates.to(ppo->device, RG_H2D_NONBLOCKING(ppo->device));
						torch::Tensor tdActionMasks = tActionMasks.to(ppo->device, RG_H2D_NONBLOCKING(ppo->device));
						ppo->InferActions(tdStates, tdActionMasks, &tActions, &tLogProbs, &inferModels);
						tActions = tActions.cpu();
					}
					out.inferTime += inferTimer.Elapsed();

					auto curActions = TENSOR_TO_VEC<int>(tActions);

					FList newLogProbs;
					if (tLogProbs.defined() && !render)
						newLogProbs = TENSOR_TO_VEC<float>(tLogProbs);

					stepTimer.Reset();
					envSet->Sync(); // Make sure the first half is done

					// Decode flat action indices -> 8-dim continuous components for auxiliary critics/reward models.
					// Placed after Sync() to avoid a data race: StepFirstHalf calls ResetBeforeStep()
					// on worker threads which writes player.eventState concurrently with this read.
					if (useActionComps) {
						for (int arenaIdx = 0; arenaIdx < envSet->arenas.size(); arenaIdx++) {
							auto* actionParser = envSet->actionParsers[arenaIdx];
							auto& gs = envSet->state.gameStates[arenaIdx];
							int playerStartIdx = envSet->state.arenaPlayerStartIdx[arenaIdx];
							for (int playerIdx = 0; playerIdx < (int)gs.players.size(); playerIdx++) {
								int flatIdx = playerStartIdx + playerIdx;
								Action act = actionParser->ParseAction(curActions[flatIdx], gs.players[playerIdx], gs);
								for (int d = 0; d < 8; d++)
									_curActionComps[flatIdx * 8 + d] = act[d];
							}
						}
					}

					envSet->StepSecondHalf(curActions, false);
					out.envStepTime += stepTimer.Elapsed();

					if (stepCallback)
						stepCallback(this, envSet->state.gameStates, report);

					if (render) {
						renderSender->Send(envSet->state.gameStates[0]);
						continue;
					}

					// Count live-policy touches for the curriculum anneal competence gates
					for (int newPlayerIdx : newPlayerIndices) {
						int arenaIdx = playerArenaIdx[newPlayerIdx];
						const GameState& gs = envSet->state.gameStates[arenaIdx];
						const Player& player = gs.players[playerLocalIdx[newPlayerIdx]];
						if (player.ballTouchedStep) {
							out.touchCount++;
							if (!player.isOnGround && gs.ball.pos.z >= 500)
								out.highAirTouchCount++;

							// Achieved contact stats for the adaptive targets (Feature C).
							// gs.prev is null only on the first step after an arena reset;
							// the hit force is StrongTouchReward's exact formula.
							if (doAdaptiveTargets && gs.prev) {
								out.touchBallSpeeds.push_back(gs.ball.vel.Length());
								out.touchHitForces.push_back((gs.ball.vel - gs.prev->ball.vel).Length());
							}
						}
					}

					// Calc average rewards
					if (config.addRewardsToMetrics && (Math::RandInt(0, config.rewardSampleRandInterval) == 0)) {
						int numSamples = RS_MIN(envSet->arenas.size(), config.maxRewardSamples);
						std::unordered_map<std::string, AvgTracker> avgRewards = {};
						std::unordered_map<std::string, AvgTracker> avgGatedRewards = {};
						std::unordered_map<std::string, AvgTracker> avgCurriculumRewards = {};
						std::unordered_map<std::string, AvgTracker> avgAerialGatedRewards = {};
						std::unordered_map<std::string, AvgTracker> avgAerialCurriculumRewards = {};
						for (int i = 0; i < numSamples; i++) {
							int arenaIdx = Math::RandInt(0, envSet->arenas.size());
							auto& prevRewards = envSet->state.lastRewards[arenaIdx];
							auto& prevGatedRewards = envSet->state.lastGCRLGatedRewards[arenaIdx];
							auto& prevCurriculumRewards = envSet->state.lastCurriculumRewards[arenaIdx];
							auto& prevAerialGatedRewards = envSet->state.lastAerialGCRLGatedRewards[arenaIdx];
							auto& prevAerialCurriculumRewards = envSet->state.lastAerialCurriculumRewards[arenaIdx];

							for (int j = 0; j < envSet->rewards[arenaIdx].size(); j++) {
								std::string rewardName = envSet->rewards[arenaIdx][j].reward->GetName();
								avgRewards[rewardName] += prevRewards[j];
							}
							for (int j = 0; j < envSet->gcrlGatedRewards[arenaIdx].size(); j++) {
								std::string rewardName = envSet->gcrlGatedRewards[arenaIdx][j].reward->GetName();
								avgGatedRewards[rewardName] += prevGatedRewards[j];
							}
							for (int j = 0; j < envSet->curriculumRewards[arenaIdx].size(); j++) {
								std::string rewardName = envSet->curriculumRewards[arenaIdx][j].reward->GetName();
								avgCurriculumRewards[rewardName] += prevCurriculumRewards[j];
							}
							for (int j = 0; j < envSet->aerialGCRLGatedRewards[arenaIdx].size(); j++) {
								std::string rewardName = envSet->aerialGCRLGatedRewards[arenaIdx][j].reward->GetName();
								avgAerialGatedRewards[rewardName] += prevAerialGatedRewards[j];
							}
							for (int j = 0; j < envSet->aerialCurriculumRewards[arenaIdx].size(); j++) {
								std::string rewardName = envSet->aerialCurriculumRewards[arenaIdx][j].reward->GetName();
								avgAerialCurriculumRewards[rewardName] += prevAerialCurriculumRewards[j];
							}
						}

						for (auto& pair : avgRewards)
							report.AddAvg("Rewards/" + pair.first, pair.second.Get());
						for (auto& pair : avgGatedRewards)
							report.AddAvg("Rewards/Gated/" + pair.first, pair.second.Get());
						for (auto& pair : avgCurriculumRewards)
							report.AddAvg("Rewards/Curriculum/" + pair.first, pair.second.Get());
						for (auto& pair : avgAerialGatedRewards)
							report.AddAvg("Rewards/AerialGated/" + pair.first, pair.second.Get());
						for (auto& pair : avgAerialCurriculumRewards)
							report.AddAvg("Rewards/AerialCurriculum/" + pair.first, pair.second.Get());
					}

					// Now that we've inferred and stepped the env, we can add that stuff to the trajectories
					int i = 0;
					for (int newPlayerIdx : newPlayerIndices) {
						trajectories[newPlayerIdx].actions.push_back(curActions[newPlayerIdx]);
						trajectories[newPlayerIdx].rewards += envSet->state.rewards[newPlayerIdx];
						trajectories[newPlayerIdx].gcrlGatedRewards += envSet->state.gcrlGatedRewards[newPlayerIdx];
						trajectories[newPlayerIdx].curriculumRewards += envSet->state.curriculumRewards[newPlayerIdx];
						trajectories[newPlayerIdx].aerialGCRLGatedRewards += envSet->state.aerialGCRLGatedRewards[newPlayerIdx];
						trajectories[newPlayerIdx].aerialCurriculumRewards += envSet->state.aerialCurriculumRewards[newPlayerIdx];
						trajectories[newPlayerIdx].logProbs += newLogProbs[i];
						if (useActionComps) {
							const float* comp = _curActionComps.data() + newPlayerIdx * 8;
							trajectories[newPlayerIdx].actionComps.insert(
								trajectories[newPlayerIdx].actionComps.end(), comp, comp + 8);
						}
						if (config.ppo.useSORS) {
							int arenaIdx = playerArenaIdx[newPlayerIdx];
							int localIdx = playerLocalIdx[newPlayerIdx];
							const GameState& gs = envSet->state.gameStates[arenaIdx];
							const Player& player = gs.players[localIdx];
							const Player* prevPlayer = player.prev;

							SORSStep sorsStep = {};
							sorsStep.ballPos = gs.ball.pos;
							sorsStep.ballVel = gs.ball.vel;
							sorsStep.playerPos = player.pos;
							sorsStep.playerVel = player.vel;
							sorsStep.team = player.team;
							sorsStep.touch = player.ballTouchedStep;
							sorsStep.shot = player.eventState.shot;
							if (gs.goalScored) {
								bool scored = (player.team != RS_TEAM_FROM_Y(gs.ball.pos.y));
								sorsStep.goalFor = scored;
								sorsStep.goalAgainst = !scored;
							}
							sorsStep.playerOnGround = player.isOnGround;
							sorsStep.playerDemoed = player.isDemoed;
							sorsStep.gotFlipReset = player.GotFlipReset();
							sorsStep.prevValid = prevPlayer;
							sorsStep.prevGotFlipReset = prevPlayer && prevPlayer->GotFlipReset();
							sorsStep.prevFlipping = prevPlayer && prevPlayer->isFlipping;
							sorsStep.prevOnGround = prevPlayer && prevPlayer->isOnGround;
							sorsStep.playerUpZ = player.rotMat.up.z;

							float ownDist = player.pos.Dist(gs.ball.pos);
							float bestOppDist = 1e30f;
							for (const Player& other : gs.players)
								if (other.team != player.team)
									bestOppDist = RS_MIN(bestOppDist, other.pos.Dist(gs.ball.pos));
							sorsStep.firstToBall = ownDist <= bestOppDist;
							trajectories[newPlayerIdx].sorsSteps.push_back(sorsStep);
						}

						// Optionality stratum flags (Feature D): cheap CPU tags that mark which
						// steps' achieved ball states qualify for the defensive/resource bank
						// strata (plus the touch mask for Opt/Phi At Touch). Flags describe the
						// post-step state, one step (~33ms) after the obs row they select — an
						// acceptable approximation for goal-bank purposes.
						if (doOptionality) {
							int arenaIdx = playerArenaIdx[newPlayerIdx];
							const GameState& gs = envSet->state.gameStates[arenaIdx];
							const Player& player = gs.players[playerLocalIdx[newPlayerIdx]];
							const Player* prevPlayer = player.prev;

							uint8_t flags = 0;
							if (player.ballTouchedStep)
								flags |= OPT_FLAG_TOUCH;
							if (!player.isDemoed) {
								// Defensive: ball deeper toward own net than the car and in the
								// defensive third, or car parked near its own net.
								float s = (player.team == Team::BLUE) ? -1.0f : 1.0f;
								Vec ownNet = (player.team == Team::BLUE) ?
									CommonValues::BLUE_GOAL_CENTER : CommonValues::ORANGE_GOAL_CENTER;
								bool ballBehind = (gs.ball.pos.y * s > player.pos.y * s) && (gs.ball.pos.y * s > OPT_DEFENSIVE_THIRD_Y);
								if (ballBehind || player.pos.Dist(ownNet) < OPT_OWN_NET_RADIUS)
									flags |= OPT_FLAG_DEFENSIVE;

								// Resource: big-pad pickup (small pads give exactly 12) or
								// supersonic with a healthy boost reserve.
								bool bigPadPickup = prevPlayer && (player.boost - prevPlayer->boost > 12.5f);
								if (bigPadPickup || (player.isSupersonic && player.boost >= 50))
									flags |= OPT_FLAG_RESOURCE;
							}
							trajectories[newPlayerIdx].stratumFlags.push_back(flags);
						}
						i++;
					}

					auto curTerminals = std::vector<uint8_t>(numPlayers, 0);
					for (int idx = 0; idx < envSet->arenas.size(); idx++) {
						uint8_t terminalType = envSet->state.terminals[idx];
						if (!terminalType)
							continue;

						// These are exactly the arenas envSet->Reset() will reset next step
						out.arenaResets++;

						auto playerStartIdx = envSet->state.arenaPlayerStartIdx[idx];
						int playersInArena = envSet->state.gameStates[idx].players.size();
						for (int i = 0; i < playersInArena; i++)
							curTerminals[playerStartIdx + i] = terminalType;
					}

					for (int newPlayerIdx : newPlayerIndices) {
						int8_t terminalType = curTerminals[newPlayerIdx];
						auto& traj = trajectories[newPlayerIdx];

						if (!terminalType && traj.Length() >= maxEpisodeLength) {
							// Episode is too long, truncate it here
							// This won't actually reset the env, but rather will just add it to experience buffer as truncated
							terminalType = RLGC::TerminalType::TRUNCATED;
						}

						traj.terminals.push_back(terminalType);
						if (terminalType) {

							if (terminalType == RLGC::TerminalType::TRUNCATED) {
								// Truncation requires an additional next state for the critic
								envSet->state.obs.AppendRow(newPlayerIdx, traj.nextStates);
								// Normalize the just-appended next-state using the current iteration's stats
								// (regular trajectory states are already normalized; nextStates come raw from the env).
								if (obsStat && !_normOffset.empty()) {
									float* ptr = traj.nextStates.data() + traj.nextStates.size() - obsSize;
									ApplyObsNorm(ptr, 1, obsSize, _normOffset, _normInvStd);
								}
							}

							// Frontier episode-return comparison (Feature A): on a real
							// episode end (not a max-length truncation), bucket the episode
							// return by the reset source that started it. A max-length
							// truncation doesn't reset the arena, so the next episode keeps
							// the previous tag — acceptable approximation.
							if (doFrontier && terminalType != RLGC::TerminalType::TRUNCATED) {
								int arenaIdx = playerArenaIdx[newPlayerIdx];
								uint8_t tag = config.ppo.frontierBuffer->GetArenaTag(envSet->arenas[arenaIdx]);
								double epReturn = 0;
								for (float r : traj.rewards)
									epReturn += r;
								if (tag == RLGC::FrontierStateBuffer::TAG_FRONTIER)
									report.AddAvg("Frontier/Episode Return (Frontier Resets)", epReturn);
								else if (tag == RLGC::FrontierStateBuffer::TAG_KICKOFF)
									report.AddAvg("Frontier/Episode Return (Kickoff Resets)", epReturn);
							}

							fnFinalizeTrajectory(traj);
						}
					}
				}

				report["Inference Time"] = out.inferTime;
				report["Env Step Time"] = out.envStepTime;

				// ── Data-pipeline conservation audit ──
				// Across the run: sum(Collected) == sum(Consumed) + delta(In-Flight) +
				// delta(Preserved Queue). If these ever drift apart, experience is being
				// silently destroyed somewhere (this is exactly how the old-version
				// trajectory-discard bug stayed invisible: every run since
				// trainAgainstOldVersions landed collected 2-2.5x what it consumed).
				size_t inFlightSteps = 0;
				for (auto& traj : trajectories)
					inFlightSteps += traj.Length();
				report["Data/Consumed Timesteps"] = combinedTraj.Length();
				report["Data/In-Flight Timesteps"] = inFlightSteps;
				report["Data/Preserved Truncated Timesteps"] = preservedSteps;
				report["Data/Preserved Queue Timesteps"] = preservedTrajSteps;
				report["Data/Drained Timesteps"] = drainedSteps;
			}

			// Safety net: old-version players' trajectories were finalized as TRUNCATED at the
			// start of this batch and receive no appends during it, so these should already be
			// empty. Clearing again guarantees no old-version-era data is ever stitched to
			// new-policy steps if that invariant is ever broken.
			for (int oldPlayerIdx : oldPlayerIndices)
				trajectories[oldPlayerIdx].Clear();

			out.collectionTime = collectionTimer.Elapsed();
		}; // fnCollectBatch

		// ── Async collection setup ────────────────────────────────────────────────
		// When asyncCollection is on, a background thread fills pendingBatch while
		// the main thread processes and learns from the previous batch (1 update off-policy).
		// When off, fnCollectBatch is called inline on the main thread.
		bool doAsync = config.ppo.asyncCollection && !render;
		ModelSet inferenceModels;
		if (doAsync)
			inferenceModels = ppo->GetPolicyModels().CloneAll();

		// Dedicated GPU stream for the async collector. The collector thread and the main
		// (process+learn) thread both issue libtorch GPU ops; driving the *same* default
		// HIP/CUDA stream from two threads is what produced the HSA queue faults
		// (HSA_STATUS_ERROR_EXCEPTION 0x1016): one thread's caching-allocator free recycled
		// a block the other thread's in-flight copy/kernel was still reading. Pinning the
		// collector to its own non-default stream makes the caching allocator insert the
		// proper cross-stream events whenever a block is reused between the two threads, so
		// the overlap stays correct without serializing kernels.
#ifdef RG_CUDA_SUPPORT
		std::optional<GpuStream> collectorStream;
		if (doAsync && ppo->device.is_cuda())
			collectorStream = c10_gpu::getStreamFromPool(/*isHighPriority=*/false, ppo->device.index());
#endif

		struct {
			std::mutex m;
			std::condition_variable cvReady, cvGo;
			bool batchReady = false, collectGo = false, stop = false;
		} handoff;

		CollectedBatch pendingBatch;
		std::thread collectorThread;

		if (doAsync) {
			collectorThread = std::thread([&] {
#ifdef RG_CUDA_SUPPORT
				// Make `collectorStream` the current stream for this thread's entire
				// lifetime, so every device tensor the collector allocates is homed on it.
				std::optional<GpuStreamGuard> streamGuard;
				if (collectorStream.has_value())
					streamGuard.emplace(*collectorStream);
#endif
				while (true) {
					{
						std::unique_lock<std::mutex> lk(handoff.m);
						handoff.cvGo.wait(lk, [&] { return handoff.collectGo || handoff.stop; });
						if (handoff.stop) return;
						handoff.collectGo = false;
					}
					fnCollectBatch(inferenceModels, pendingBatch);
#ifdef RG_CUDA_SUPPORT
					// Flush all collector GPU work before publishing the batch, so every
					// device block this batch touched is fully retired before the main
					// thread takes over and the next CopyParamsFrom/kick happens.
					if (collectorStream.has_value())
						collectorStream->synchronize();
#endif
					{
						std::lock_guard<std::mutex> lk(handoff.m);
						handoff.batchReady = true;
					}
					handoff.cvReady.notify_one();
				}
			});

			// Prime: kick first batch immediately
			{
				std::lock_guard<std::mutex> lk(handoff.m);
				handoff.collectGo = true;
			}
			handoff.cvGo.notify_one();
		}

		while (true) {
			bool isFirstIteration = (totalTimesteps == 0);
			Timer iterationTimer = {};

			// ── Get a batch ───────────────────────────────────────────────────────
			CollectedBatch batch;
			if (doAsync) {
				// Wait for collector to finish the batch it was kicked with last turn
				{
					std::unique_lock<std::mutex> lk(handoff.m);
					handoff.cvReady.wait(lk, [&] { return handoff.batchReady; });
					handoff.batchReady = false;
				}
				batch = std::move(pendingBatch);

				// Sync policy snapshot to the latest weights (ES + Learn from previous turn).
				// Collector is idle between handoff and kick, so no lock is needed here.
				inferenceModels.CopyParamsFrom(ppo->GetPolicyModels());

				// Kick next batch collection — runs concurrently with process+learn below
				{
					std::lock_guard<std::mutex> lk(handoff.m);
					handoff.collectGo = true;
				}
				handoff.cvGo.notify_one();
			} else {
				fnCollectBatch(ppo->models, batch);
			}

			Report report = std::move(batch.report);
			int stepsCollected = batch.stepsCollected;
			auto& combinedTraj = batch.combinedTraj;

			auto fnGetAnnealedScale = [&](float targetScale, int64_t configStart, int64_t annealSteps, uint64_t& startTS) {
				if (annealSteps <= 0)
					return targetScale;

				if (startTS == UINT64_MAX)
					startTS = configStart >= 0 ? (uint64_t)configStart : totalTimesteps;

				if (totalTimesteps <= startTS)
					return 0.0f;

				float progress = (float)((double)(totalTimesteps - startTS) / (double)annealSteps);
				return targetScale * RS_CLAMP(progress, 0.0f, 1.0f);
			};
			// Update the competence EMAs that gate the curriculum anneals
			{
				double d = RS_CLAMP(config.ppo.competenceEMADecay, 0.0, 1.0);
				double touchRatio = stepsCollected > 0 ? (double)batch.touchCount / stepsCollected : 0.0;
				double highAirTouchRatio = stepsCollected > 0 ? (double)batch.highAirTouchCount / stepsCollected : 0.0;
				touchRatioEMA = touchRatioEMA * d + touchRatio * (1.0 - d);
				highAirTouchRatioEMA = highAirTouchRatioEMA * d + highAirTouchRatio * (1.0 - d);
				report["Curriculum/Touch Ratio EMA"] = (float)touchRatioEMA;
				report["Curriculum/High Air Touch Ratio EMA"] = (float)highAirTouchRatioEMA;
			}

			// ── Feature C: adaptive (ratcheted-quantile) targets ──
			// Update the EMA histograms from this iteration's touch events, then ratchet each
			// enabled target toward the configured quantile under the three stability rules:
			// hold below min-samples (cold buffer can't move the seed), rise at most
			// maxSlewPerIter, fall at most decayPerIter (default 0 = pure ratchet). The gate
			// target is learner-side only; the StrongTouch floor is published to workers via
			// the config atomic.
			if (config.ppo.useAdaptiveGateTargetVel || config.ppo.useAdaptiveStrongTouchFloor) {
				int nTouchSamples = (int)batch.touchHitForces.size(); // == touchBallSpeeds size
				report["Adaptive/Touch Samples Per Iter"] = nTouchSamples;

				// Decay+accumulate every iteration (sparse iterations decay correctly); the
				// min-samples gate only controls whether the TARGET moves, not the histogram.
				if (contactBallSpeedHist)
					contactBallSpeedHist->Update(batch.touchBallSpeeds);
				if (contactHitForceHist)
					contactHitForceHist->Update(batch.touchHitForces);

				auto fnRatchet = [&](double& cur, EMAHistogram* hist, float q) {
					if (!hist || nTouchSamples < config.ppo.adaptiveTargetMinSamples)
						return; // hold
					double raw = hist->Quantile(q);
					if (raw > cur)
						cur = RS_MIN(raw, cur * (1.0 + config.ppo.adaptiveTargetMaxSlewPerIter));
					else if (raw < cur)
						cur = RS_MAX(raw, cur * (1.0 - config.ppo.adaptiveTargetDecayPerIter));
				};

				if (config.ppo.useAdaptiveGateTargetVel) {
					fnRatchet(adaptiveGateTargetVel, contactBallSpeedHist, config.ppo.adaptiveGateTargetVelQuantile);
					report["Adaptive/Gate Target Vel"] = (float)adaptiveGateTargetVel;
					report["Adaptive/Contact Vel P70 (Raw)"] = contactBallSpeedHist ?
						(float)contactBallSpeedHist->Quantile(config.ppo.adaptiveGateTargetVelQuantile) : 0.0f;
				}
				if (config.ppo.useAdaptiveStrongTouchFloor) {
					fnRatchet(adaptiveStrongTouchFloor, contactHitForceHist, config.ppo.adaptiveStrongTouchFloorQuantile);
					if (config.ppo.adaptiveStrongTouchFloorAtomic)
						config.ppo.adaptiveStrongTouchFloorAtomic->store((float)adaptiveStrongTouchFloor, std::memory_order_relaxed);
					report["Adaptive/StrongTouch Floor"] = (float)adaptiveStrongTouchFloor;
					report["Adaptive/StrongTouch Floor KPH"] = (float)(adaptiveStrongTouchFloor / (250.0 / 9.0));
					report["Adaptive/Hit Force P40 (Raw)"] = contactHitForceHist ?
						(float)contactHitForceHist->Quantile(config.ppo.adaptiveStrongTouchFloorQuantile) : 0.0f;
				}
			}

			// Like fnGetAnnealedRange, but progress is an accumulated counter that only advances
			// while the competence EMA is at/above its gate (gate <= 0 -> always advances, which
			// matches the old wall-clock behavior). Keeps curricula at full strength however long
			// the policy stays below the gate.
			auto fnGetGatedAnnealedRange = [&](float startScale, float targetScale, int64_t configStart, int64_t annealSteps,
				uint64_t& startTS, uint64_t& progressTS, float competenceGate, double competenceEMA) {
				if (annealSteps <= 0)
					return targetScale;

				if (startTS == UINT64_MAX)
					startTS = configStart >= 0 ? (uint64_t)configStart : totalTimesteps;

				if (totalTimesteps > startTS && (competenceGate <= 0 || competenceEMA >= competenceGate))
					progressTS += (uint64_t)stepsCollected;

				float progress = (float)((double)progressTS / (double)annealSteps);
				progress = RS_CLAMP(progress, 0.0f, 1.0f);
				return startScale + (targetScale - startScale) * progress;
			};

			ppo->curGCRLAdvScale = fnGetAnnealedScale(
				config.ppo.gcrlAdvScale,
				config.ppo.gcrlAdvScaleAnnealStart,
				config.ppo.gcrlAdvScaleAnnealSteps,
				gcrlAdvScaleAnnealStartTS
			);
			// Gate influence ramps are competence-gated on the touch EMA: the gate filters
			// rewards by GCRL terminal progress, and the critics only know anything about
			// terminal progress once the policy actually touches the ball. On a wall clock
			// (run tkpk0780) influence hit 1.0 over a touchless world and the curriculum/
			// gated rewards spent ~3B steps multiplied by sigmoid(noise) ≈ 0.5 ± 0.2.
			ppo->curGCRLRewardGateInfluence = config.ppo.useGCRLRewardGate ? fnGetGatedAnnealedRange(
				0.0f,
				config.ppo.gcrlRewardGateInfluence,
				config.ppo.gcrlRewardGateAnnealStart,
				config.ppo.gcrlRewardGateAnnealSteps,
				gcrlRewardGateAnnealStartTS,
				gcrlRewardGateAnnealProgressTS,
				config.ppo.curriculumAnnealTouchRatioGate,
				touchRatioEMA
			) : 0.0f;
			ppo->curGCRLAerialRewardGateInfluence = config.ppo.useGCRLRewardGate ? fnGetGatedAnnealedRange(
				config.ppo.gcrlAerialRewardGateStartInfluence,
				config.ppo.gcrlAerialRewardGateInfluence,
				config.ppo.gcrlAerialRewardGateAnnealStart,
				config.ppo.gcrlAerialRewardGateAnnealSteps,
				gcrlAerialRewardGateAnnealStartTS,
				gcrlAerialRewardGateAnnealProgressTS,
				config.ppo.curriculumAnnealTouchRatioGate,
				touchRatioEMA
			) : 0.0f;
			ppo->curCurriculumRewardScale = fnGetGatedAnnealedRange(
				config.ppo.curriculumRewardScale,
				0.0f,
				config.ppo.curriculumRewardAnnealStart,
				config.ppo.curriculumRewardAnnealSteps,
				curriculumRewardAnnealStartTS,
				curriculumAnnealProgressTS,
				config.ppo.curriculumAnnealTouchRatioGate,
				touchRatioEMA
			);
			ppo->curAerialCurriculumRewardScale = fnGetGatedAnnealedRange(
				config.ppo.aerialCurriculumRewardScale,
				0.0f,
				config.ppo.aerialCurriculumRewardAnnealStart,
				config.ppo.aerialCurriculumRewardAnnealSteps,
				aerialCurriculumRewardAnnealStartTS,
				aerialCurriculumAnnealProgressTS,
				config.ppo.aerialCurriculumAnnealAirTouchRatioGate,
				highAirTouchRatioEMA
			);
			ppo->curSORSRewardScale = config.ppo.useSORS ? fnGetAnnealedScale(
				config.ppo.sorsRewardScale,
				config.ppo.sorsRewardScaleAnnealStart,
				config.ppo.sorsRewardScaleAnnealSteps,
				sorsRewardScaleAnnealStartTS
			) : 0.0f;
			report["GCRL/Adv Scale"] = ppo->curGCRLAdvScale;
			if (config.ppo.useGCRLRewardGate) {
				report["GCRL Gate/Influence"] = ppo->curGCRLRewardGateInfluence;
				report["GCRL Aerial Gate/Influence"] = ppo->curGCRLAerialRewardGateInfluence;
			}
			report["Curriculum/Scale"] = ppo->curCurriculumRewardScale;
			report["Aerial Curriculum/Scale"] = ppo->curAerialCurriculumRewardScale;
			if (config.ppo.useSORS)
				report["SORS/Reward Scale"] = ppo->curSORSRewardScale;

			if (config.ppo.useSORS)
				ppo->TrainSORS(report);

			// Refresh the own-goal target the anti critic's advantage queries
			// (depends on obs standardization stats when standardizeObs is on)
			if (config.ppo.useGCRL)
				ppo->curGCRLAntiTargetRow = MakeGCRLTerminalTargetRow(true, config, obsStat, (float)adaptiveGateTargetVel);

			Timer consumptionTimer = {};
			{ // Process timesteps
					RG_NO_GRAD;

					// Make and transpose tensors
					torch::Tensor tStates = torch::tensor(combinedTraj.states).reshape({ -1, obsSize });
					torch::Tensor tActionMasks = torch::tensor(combinedTraj.actionMasks).reshape({ -1, numActions });
					torch::Tensor tActions = torch::tensor(combinedTraj.actions);
					torch::Tensor tLogProbs = torch::tensor(combinedTraj.logProbs);
					torch::Tensor tRewards = torch::tensor(combinedTraj.rewards);
					torch::Tensor tTerminals = torch::tensor(combinedTraj.terminals);

					// Built once, shared by the GCRL gate, SORS inference and the experience buffer
					torch::Tensor tActionComps;
					if (!combinedTraj.actionComps.empty())
						tActionComps = torch::tensor(combinedTraj.actionComps).reshape({ -1, 8 });

					// Per-step game-mode ids (kLong, CPU) for per-mode normalization/reporting
					torch::Tensor tModeIds = torch::from_blob(
						(void*)combinedTraj.modeIds.data(), { (int64_t)combinedTraj.modeIds.size() },
						torch::TensorOptions().dtype(torch::kChar)).to(torch::kLong);

					// LIVE goal-critic phi(s,a) over the whole batch — computed once, lazily,
					// and shared by difficulty-HER (Feature B) and frontier scoring (Feature A).
					torch::Tensor tSharedPhi;
					auto fnGetSharedPhi = [&]() -> torch::Tensor {
						if (!tSharedPhi.defined() && tActionComps.defined())
							tSharedPhi = ppo->InferGCRLPhiEmbeddings(tStates, tActionComps);
						return tSharedPhi;
					};

					// Per-step goal-critic terminal score [N], shared between the gate block
					// (which already computes it) and the optionality offensive-stratum filter.
					torch::Tensor tGoalScoreShared;
					// Raw terminal utility [N] for value-weighted optionality:
					// opponent-goal reachability minus own-goal danger, normalized later inside the bank.
					torch::Tensor tOptionalityValueShared;

					// Episode segment end per index (terminals[epEnd] != 0), shared by the
					// HER pass and the frontier scan. Built only when a feature needs it.
					int trajN = (int)combinedTraj.Length();
					std::vector<int> epEndOf, epStartOf;
					auto fnBuildEpBounds = [&]() {
						if (!epEndOf.empty() || trajN == 0)
							return;
						epEndOf.resize(trajN);
						epStartOf.resize(trajN);
						for (int epStart = 0; epStart < trajN;) {
							int epEnd = epStart;
							while (epEnd < trajN - 1 && combinedTraj.terminals[epEnd] == 0)
								epEnd++;
							for (int i = epStart; i <= epEnd; i++) {
								epEndOf[i] = epEnd;
								epStartOf[i] = epStart;
							}
							epStart = epEnd + 1;
						}
					};

					// ── Feature B: difficulty-aware HER (deferred batched relabel) ──
					// Relabeling was skipped on the collector when useDifficultyHER is on; do it
					// here in one batched pass. The uniform-floor branch reproduces today's
					// sampling byte-for-byte; the candidate branch prefers goals near the middle
					// of the batch distance distribution.
					if (config.ppo.useGCRL && config.ppo.useDifficultyHER && trajN > 0 && tActionComps.defined()) {
						RG_NO_GRAD;
						fnBuildEpBounds();
						const float* states = combinedTraj.states.data();
						int H = config.ppo.gcrlHorizon, minH = config.ppo.gcrlMinHorizon;
						int K = RS_MAX(1, config.ppo.herCandidates);
						bool hasMirror = combinedTraj.mirrored.size() == (size_t)trajN;

						std::vector<int> tTarget(trajN);
						std::vector<int> candAnchors; candAnchors.reserve(trajN);
						int uniformCount = 0;
						for (int t = 0; t < trajN; t++) {
							int lo = t + minH;
							int hi = RS_MIN(t + H, epEndOf[t]);
							if (hi <= lo) { tTarget[t] = hi; continue; } // degenerate (as today)
							if (RocketSim::Math::RandFloat() < config.ppo.herUniformFraction) {
								tTarget[t] = Math::RandInt(lo, hi + 1); // byte-identical uniform floor
								uniformCount++;
							} else {
								tTarget[t] = -1;
								candAnchors.push_back(t);
							}
						}

						int Nsel = (int)candAnchors.size();
						if (Nsel > 0) {
							torch::Tensor tPhi = fnGetSharedPhi();
							std::vector<int> candOffsets((size_t)Nsel * K);
							FList candGoals((size_t)Nsel * K * 6);
							for (int s = 0; s < Nsel; s++) {
								int t = candAnchors[s];
								int lo = t + minH;
								int hi = RS_MIN(t + H, epEndOf[t]);
								for (int k = 0; k < K; k++) {
									int kt = Math::RandInt(lo, hi + 1);
									candOffsets[(size_t)s * K + k] = kt;
									bool flip = hasMirror && (combinedTraj.mirrored[t] != combinedTraj.mirrored[kt]);
									float* dst = candGoals.data() + ((size_t)s * K + k) * 6;
									for (int d = 0; d < 6; d++)
										dst[d] = states[(size_t)kt * obsSize + d];
									if (flip) { dst[0] = -dst[0]; dst[3] = -dst[3]; }
								}
							}
							torch::Tensor tCandGoals = torch::tensor(candGoals).reshape({ (int64_t)Nsel * K, 6 });
							torch::Tensor tCandPsi = ppo->InferGCRLPsiEmbeddings(tCandGoals);

							torch::Tensor tAnchorIdx = torch::tensor(candAnchors, torch::TensorOptions().dtype(torch::kLong));
							torch::Tensor phiSel = tPhi.index_select(0, tAnchorIdx);
							torch::Tensor phiRep = phiSel.unsqueeze(1).expand({ Nsel, K, phiSel.size(1) }).reshape({ (int64_t)Nsel * K, -1 });
							torch::Tensor d = -(phiRep * tCandPsi).sum(1) / config.ppo.gcrlTau; // [Nsel*K]

							// Self-normalization: percentile within the batch's distance distribution
							torch::Tensor sortedD = std::get<0>(d.sort());
							torch::Tensor p = torch::searchsorted(sortedD, d).to(torch::kFloat32) / (float)(Nsel * K);
							float sigma = RS_MAX(1e-3f, config.ppo.herDifficultySigma);
							torch::Tensor w = torch::exp(-(p - 0.5f).square() / (2.0f * sigma * sigma)).reshape({ Nsel, K });
							torch::Tensor chosen = torch::multinomial(w, 1).flatten(); // [Nsel] in [0, K)

							torch::Tensor pSel = p.reshape({ Nsel, K }).gather(1, chosen.unsqueeze(1)).flatten();
							report["HER/Selected Distance Percentile Mean"] = pSel.mean().item<float>();

							torch::Tensor chosenCpu = chosen.to(torch::kCPU).contiguous();
							const int64_t* chosenData = chosenCpu.data_ptr<int64_t>();
							for (int s = 0; s < Nsel; s++)
								tTarget[candAnchors[s]] = candOffsets[(size_t)s * K + (int)chosenData[s]];
						}

						// Write back futureGoals / carFutureGoals — verbatim port of the
						// collector relabel flip logic; one t_target drives both arrays.
						combinedTraj.futureGoals.resize((size_t)trajN * 6);
						combinedTraj.carFutureGoals.resize((size_t)trajN * 6);
						FList selectedOffsets; selectedOffsets.reserve(trajN);
						for (int t = 0; t < trajN; t++) {
							int tt = tTarget[t];
							bool flip = hasMirror && (combinedTraj.mirrored[t] != combinedTraj.mirrored[tt]);
							for (int d = 0; d < 6; d++)
								combinedTraj.futureGoals[(size_t)t * 6 + d] = states[(size_t)tt * obsSize + d];
							if (flip) {
								combinedTraj.futureGoals[(size_t)t * 6 + 0] = -combinedTraj.futureGoals[(size_t)t * 6 + 0];
								combinedTraj.futureGoals[(size_t)t * 6 + 3] = -combinedTraj.futureGoals[(size_t)t * 6 + 3];
							}
							if (carLocalBallOffset >= 0) {
								for (int d = 0; d < 6; d++)
									combinedTraj.carFutureGoals[(size_t)t * 6 + d] = states[(size_t)tt * obsSize + carLocalBallOffset + d];
								if (flip) {
									combinedTraj.carFutureGoals[(size_t)t * 6 + 1] = -combinedTraj.carFutureGoals[(size_t)t * 6 + 1];
									combinedTraj.carFutureGoals[(size_t)t * 6 + 4] = -combinedTraj.carFutureGoals[(size_t)t * 6 + 4];
								}
							}
							selectedOffsets.push_back((float)(tt - t));
						}

						torch::Tensor tOff = torch::tensor(selectedOffsets);
						report["HER/Selected Offset Mean"] = tOff.mean().item<float>();
						report["HER/Selected Offset P10"] = tOff.quantile(0.1).item<float>();
						report["HER/Selected Offset P50"] = tOff.quantile(0.5).item<float>();
						report["HER/Selected Offset P90"] = tOff.quantile(0.9).item<float>();
						report["HER/Uniform Fraction Actual"] = trajN > 0 ? (float)uniformCount / trajN : 0.0f;
					}

					bool hasGatedRewards = !combinedTraj.gcrlGatedRewards.empty();
					bool hasCurriculumRewards = !combinedTraj.curriculumRewards.empty();
					bool hasAerialGatedRewards = !combinedTraj.aerialGCRLGatedRewards.empty();
					bool hasAerialCurriculumRewards = !combinedTraj.aerialCurriculumRewards.empty();
					bool needsGCRLRewardGate =
						config.ppo.useGCRLRewardGate &&
						(ppo->curGCRLRewardGateInfluence > 0 || ppo->curGCRLAerialRewardGateInfluence > 0) &&
						(hasGatedRewards || hasCurriculumRewards || hasAerialGatedRewards || hasAerialCurriculumRewards) &&
						!combinedTraj.actionComps.empty();

					if (needsGCRLRewardGate) {
						torch::Tensor tGatedRewards = hasGatedRewards ?
							torch::tensor(combinedTraj.gcrlGatedRewards) :
							torch::zeros_like(tRewards);
						torch::Tensor tCurriculumRewards = hasCurriculumRewards ?
							torch::tensor(combinedTraj.curriculumRewards) :
							torch::zeros_like(tRewards);
						torch::Tensor tAerialGatedRewards = hasAerialGatedRewards ?
							torch::tensor(combinedTraj.aerialGCRLGatedRewards) :
							torch::zeros_like(tRewards);
						torch::Tensor tAerialCurriculumRewards = hasAerialCurriculumRewards ?
							torch::tensor(combinedTraj.aerialCurriculumRewards) :
							torch::zeros_like(tRewards);
						torch::Tensor tScaledCurriculumRewards = tCurriculumRewards * ppo->curCurriculumRewardScale;
						torch::Tensor tScaledAerialCurriculumRewards = tAerialCurriculumRewards * ppo->curAerialCurriculumRewardScale;
						torch::Tensor tNormalRewards = tGatedRewards + tScaledCurriculumRewards;
						torch::Tensor tAerialRewards = tAerialGatedRewards + tScaledAerialCurriculumRewards;
						torch::Tensor tBaseRewards = tRewards - tGatedRewards - tCurriculumRewards - tAerialGatedRewards - tAerialCurriculumRewards;
						torch::Tensor tGoalTargetRow = MakeGCRLTerminalTargetRow(false, config, obsStat, (float)adaptiveGateTargetVel);
						torch::Tensor tAntiTargetRow = MakeGCRLTerminalTargetRow(true, config, obsStat, (float)adaptiveGateTargetVel);
						torch::Tensor tTerminalScores = InferGCRLTerminalScoresBatched(
							ppo,
							tStates,
							tActionComps,
							tGoalTargetRow,
							tAntiTargetRow,
							combinedTraj.Length()
						);

						if (tTerminalScores.defined()) {
							torch::Tensor tGoalScore = tTerminalScores.select(1, 0);
							torch::Tensor tAntiScore = tTerminalScores.select(1, 1);
							tGoalScoreShared = tGoalScore; // reused by the optionality offensive stratum
							tOptionalityValueShared = tGoalScore - config.ppo.gcrlRewardGateAntiScale * tAntiScore;
							torch::Tensor tNormGoal = (tGoalScore - tGoalScore.mean()) / (tGoalScore.std(false) + 1e-8f);
							torch::Tensor tNormAnti = (tAntiScore - tAntiScore.mean()) / (tAntiScore.std(false) + 1e-8f);
							torch::Tensor tTerminalAdv = tNormGoal - config.ppo.gcrlRewardGateAntiScale * tNormAnti;

							int trajLength = combinedTraj.Length();
							int lookahead = RS_MAX(1, config.ppo.gcrlRewardGateLookahead);
							std::vector<int64_t> futureIdxs(trajLength);
							for (int epStart = 0; epStart < trajLength;) {
								int epEnd = epStart;
								while (epEnd < trajLength - 1 && combinedTraj.terminals[epEnd] == 0)
									epEnd++;
								for (int i = epStart; i <= epEnd; i++)
									futureIdxs[i] = RS_MIN(i + lookahead, epEnd);
								epStart = epEnd + 1;
							}

							torch::Tensor tFutureIdxs = torch::tensor(futureIdxs, torch::TensorOptions().dtype(torch::kLong));
							torch::Tensor tGateDelta = tTerminalAdv.index_select(0, tFutureIdxs) - tTerminalAdv;

							// Normalize the gate delta with slow per-mode EMA stats instead of a
							// per-batch z-score: the gate's meaning then drifts over ~1/(1-decay)
							// iterations instead of jumping with each batch's composition, and one
							// mode's dynamics (e.g. heatseeker ball speeds) can't reprice another's.
							constexpr double GATE_EMA_DECAY = 0.99;
							torch::Tensor tNormGateDelta = torch::empty_like(tGateDelta);
							for (int m = 0; m < numModes; m++) {
								torch::Tensor mask = (tModeIds == m);
								torch::Tensor vals = tGateDelta.masked_select(mask);
								if (vals.numel() == 0)
									continue;

								gateDeltaMeanEMA[m] = GATE_EMA_DECAY * gateDeltaMeanEMA[m] + (1 - GATE_EMA_DECAY) * vals.mean().item<double>();
								gateDeltaVarEMA[m] = GATE_EMA_DECAY * gateDeltaVarEMA[m] + (1 - GATE_EMA_DECAY) * vals.var(false).item<double>();

								torch::Tensor norm = (vals - gateDeltaMeanEMA[m]) / std::sqrt(gateDeltaVarEMA[m] + 1e-8);
								tNormGateDelta = tNormGateDelta.masked_scatter(mask, norm.to(torch::kFloat32));
							}

							torch::Tensor tGate = torch::sigmoid(config.ppo.gcrlRewardGateSharpness * tNormGateDelta);
							torch::Tensor tEffectiveGate = 1.0f + (tGate - 1.0f) * ppo->curGCRLRewardGateInfluence;
							torch::Tensor tEffectiveAerialGate = 1.0f + (tGate - 1.0f) * ppo->curGCRLAerialRewardGateInfluence;
							torch::Tensor tAppliedGatedRewards = tNormalRewards * tEffectiveGate;
							torch::Tensor tAppliedAerialRewards = tAerialRewards * tEffectiveAerialGate;
							tRewards = tBaseRewards + tAppliedGatedRewards + tAppliedAerialRewards;

							report["GCRL Gate/Mean"] = tEffectiveGate.mean().item<float>();
							report["GCRL Gate/STD"] = tEffectiveGate.std(false).item<float>();
							if (numModes > 1) {
								for (int m = 0; m < numModes; m++) {
									torch::Tensor modeGate = tEffectiveGate.masked_select(tModeIds == m);
									if (modeGate.numel() > 0)
										report["GCRL Gate/Mean/" + modeNames[m]] = modeGate.mean().item<float>();
								}
							}
							report["GCRL Aerial Gate/Mean"] = tEffectiveAerialGate.mean().item<float>();
							report["GCRL Aerial Gate/STD"] = tEffectiveAerialGate.std(false).item<float>();
							report["GCRL Gate/Delta Mean"] = tGateDelta.mean().item<float>();
							report["GCRL Gate/Delta STD"] = tGateDelta.std(false).item<float>();
							report["GCRL Gate/Base Reward"] = tBaseRewards.mean().item<float>();
							report["GCRL Gate/Gated Reward Raw"] = tGatedRewards.mean().item<float>();
							report["GCRL Gate/Curriculum Reward Raw"] = tCurriculumRewards.mean().item<float>();
							report["GCRL Gate/Curriculum Reward Scaled"] = tScaledCurriculumRewards.mean().item<float>();
							report["GCRL Gate/Gated Reward Applied"] = tAppliedGatedRewards.mean().item<float>();
							report["GCRL Aerial Gate/Aerial Reward Raw"] = tAerialGatedRewards.mean().item<float>();
							report["GCRL Aerial Gate/Curriculum Reward Raw"] = tAerialCurriculumRewards.mean().item<float>();
							report["GCRL Aerial Gate/Curriculum Reward Scaled"] = tScaledAerialCurriculumRewards.mean().item<float>();
							report["GCRL Aerial Gate/Aerial Reward Applied"] = tAppliedAerialRewards.mean().item<float>();
							report["GCRL Gate/Final Reward"] = tRewards.mean().item<float>();
						}
					}

					// ── Feature A: frontier reset curriculum (consistency scoring) ──
					// Find timesteps where the goal critic contradicts itself (moving a few
					// steps along the trajectory toward an achieved goal increases d), backtrack
					// ~1s, and harvest the physics snapshot there. Runs after the gate (and after
					// Feature B fills futureGoals). Snapshots must align 1:1 with states; if a
					// just-enabled run drained pre-snapshot preserved trajectories the vectors
					// would be shorter, so guard on the exact size.
					if (config.ppo.useFrontierResets && config.ppo.frontierBuffer && config.ppo.useGCRL
						&& tActionComps.defined() && trajN > 0
						&& (int)combinedTraj.futureGoals.size() == trajN * 6
						&& combinedTraj.snapshots.size() == (size_t)trajN) {
						RG_NO_GRAD;
						fnBuildEpBounds();

						torch::Tensor tPhi = fnGetSharedPhi();
						torch::Tensor tGoals = torch::tensor(combinedTraj.futureGoals).reshape({ trajN, 6 });
						torch::Tensor tGoalsFlip = tGoals.clone();
						tGoalsFlip.select(1, 0).neg_();
						tGoalsFlip.select(1, 3).neg_();
						torch::Tensor tPsiGoal = ppo->InferGCRLPsiEmbeddings(tGoals);
						torch::Tensor tPsiGoalFlip = ppo->InferGCRLPsiEmbeddings(tGoalsFlip);

						std::vector<int64_t> jIdx(trajN);
						for (int t = 0; t < trajN; t++)
							jIdx[t] = RS_MIN(t + FRONTIER_CONSISTENCY_STEP, epEndOf[t]);
						torch::Tensor tJIdx = torch::tensor(jIdx, torch::TensorOptions().dtype(torch::kLong));

						// Anchor-t's goal re-expressed in step-j's mirror frame (same
						// normalized-x-negation argument as the relabeler).
						bool hasMirror = combinedTraj.mirrored.size() == (size_t)trajN;
						torch::Tensor psiG_j = tPsiGoal;
						if (hasMirror) {
							std::vector<uint8_t> fm(trajN);
							for (int t = 0; t < trajN; t++)
								fm[t] = (combinedTraj.mirrored[t] != combinedTraj.mirrored[jIdx[t]]) ? 1 : 0;
							torch::Tensor flipMask = torch::from_blob(fm.data(), { trajN },
								torch::TensorOptions().dtype(torch::kUInt8)).to(torch::kBool).unsqueeze(1);
							psiG_j = torch::where(flipMask, tPsiGoalFlip, tPsiGoal);
						}

						float tau = config.ppo.gcrlTau;
						torch::Tensor d_t = -(tPhi * tPsiGoal).sum(1) / tau;
						torch::Tensor phi_j = tPhi.index_select(0, tJIdx);
						torch::Tensor d_j = -(phi_j * psiG_j).sum(1) / tau;
						torch::Tensor score = torch::relu(d_j - d_t); // [N]

						float thresh = score.quantile(config.ppo.frontierSpikePercentile).item<float>();
						report["Frontier/Mean Uncertainty Score"] = score.mean().item<float>();
						report["Frontier/Spike Threshold"] = thresh;

						torch::Tensor scoreCpu = score.to(torch::kCPU).contiguous();
						const float* scoreData = scoreCpu.data_ptr<float>();
						int inserts = 0;
						for (int t = 0; t < trajN && inserts < FRONTIER_MAX_INSERTS_PER_ITER; t++) {
							if (!(scoreData[t] > thresh) || scoreData[t] <= 0)
								continue;
							int target = RS_MAX(epStartOf[t], t - config.ppo.frontierBacktrackSteps);
							int floor = RS_MAX(epStartOf[t], target - 2 * RS_MAX(1, config.ppo.frontierSnapshotInterval));
							for (int idx = target; idx >= floor; idx--) {
								const auto& snap = combinedTraj.snapshots[idx];
								if (!snap)
									continue;
								if (RLGC::FrontierStateBuffer::PassesSanityFilter(*snap)) {
									config.ppo.frontierBuffer->Insert(*snap);
									inserts++;
								}
								break; // first non-null snapshot at/below target
							}
						}

						report["Frontier/Candidates Per Iter"] = inserts;
						report["Frontier/Buffer Fill"] = (double)config.ppo.frontierBuffer->Size();
						uint64_t fRes = config.ppo.frontierBuffer->frontierResets.exchange(0, std::memory_order_relaxed);
						uint64_t fbRes = config.ppo.frontierBuffer->fallbackResets.exchange(0, std::memory_order_relaxed);
						report["Frontier/Fallback Resets"] = (double)fbRes;
						report["Frontier/Reset Fraction Actual"] = batch.arenaResets > 0 ? (double)fRes / batch.arenaResets : 0.0;
					}

					if (config.ppo.useSORS && tActionComps.defined()) {
						torch::Tensor tSORSRewards;
						if (ppo->device.is_cpu()) {
							tSORSRewards = ppo->InferSORSRewards(tStates.to(ppo->device, RG_H2D_NONBLOCKING(ppo->device), true), tActionComps.to(ppo->device, RG_H2D_NONBLOCKING(ppo->device), true)).cpu();
						} else {
							tSORSRewards = torch::zeros({ (int64_t)combinedTraj.Length() });
							for (int i = 0; i < combinedTraj.Length(); i += ppo->config.miniBatchSize) {
								int start = i;
								int end = RS_MIN(i + ppo->config.miniBatchSize, combinedTraj.Length());
								auto rewardPart = ppo->InferSORSRewards(
									tStates.slice(0, start, end).to(ppo->device, RG_H2D_NONBLOCKING(ppo->device), true),
									tActionComps.slice(0, start, end).to(ppo->device, RG_H2D_NONBLOCKING(ppo->device), true)
								).cpu();
								tSORSRewards.slice(0, start, end).copy_(rewardPart, true);
							}
						}

						if (tSORSRewards.defined()) {
							tSORSRewards = tSORSRewards.clamp(-config.ppo.sorsRewardClipRange, config.ppo.sorsRewardClipRange);
							report["SORS/Avg Reward"] = tSORSRewards.mean().item<float>();
							report["SORS/Avg Abs Reward"] = tSORSRewards.abs().mean().item<float>();
							tRewards = tRewards + tSORSRewards * ppo->curSORSRewardScale;
						}
					}

					// ── Feature D: optionality potential shaping ──
					// Polyak the frozen scorer, refresh the stratified goal bank, compute
					// phi_opt over the batch, and inject the masked potential delta into
					// tRewards — AFTER all multiplicative gating (it is never gated) and
					// BEFORE GAE (it flows through the normal value/GAE path, not as an
					// advantage stream). Burn-in injects zero while the normalizer seeds.
					if (config.ppo.useOptionality && config.ppo.useGCRL && ppo->optionality && trajN > 0 && tActionComps.defined()) {
						RG_NO_GRAD;
						GCRLOptionality* opt = ppo->optionality;

						// 1. Polyak the frozen scorer toward the live nets
						opt->PolyakUpdate(
							dynamic_cast<QuasimetricCritic*>(ppo->models["goal_critic"]), ppo->models["shared_head"]);

						// 2. Build stratified candidate ball-goal rows (obs space, canonicalized
						// to the unmirrored frame so phi_opt's frame-symmetry holds).
						auto fnPushBankRow = [&](RLGC::FList& dstRows, RLGC::FList& dstValues, int t, float value) {
							const float* src = combinedTraj.states.data() + (size_t)t * obsSize;
							float row[6];
							for (int d = 0; d < 6; d++)
								row[d] = src[d];
							if (combinedTraj.mirrored.size() == (size_t)trajN && combinedTraj.mirrored[t]) {
								row[0] = -row[0];
								row[3] = -row[3];
							}
							dstRows.insert(dstRows.end(), row, row + 6);
							dstValues.push_back(value);
						};

						RLGC::FList candRows[GCRLOptionality::STRATUM_AMOUNT];
						RLGC::FList candValues[GCRLOptionality::STRATUM_AMOUNT];

						// Offensive: achieved ball goals in the top half of goal-critic terminal
						// score (commitment bias / hover guard). Reuse the gate's score or
						// compute it standalone when the gate didn't run.
						torch::Tensor goalScore = tGoalScoreShared;
						torch::Tensor optValue = tOptionalityValueShared;
						if (!goalScore.defined() || !optValue.defined()) {
							torch::Tensor gRow = MakeGCRLTerminalTargetRow(false, config, obsStat, (float)adaptiveGateTargetVel);
							torch::Tensor aRow = MakeGCRLTerminalTargetRow(true, config, obsStat, (float)adaptiveGateTargetVel);
							torch::Tensor ts = InferGCRLTerminalScoresBatched(ppo, tStates, tActionComps, gRow, aRow, trajN);
							if (ts.defined()) {
								goalScore = ts.select(1, 0);
								torch::Tensor antiScore = ts.select(1, 1);
								optValue = goalScore - config.ppo.gcrlRewardGateAntiScale * antiScore;
							}
						}

						// Reservoir sampler so each stratum draws an unbiased subset under its cap
						auto fnReservoir = [&](std::vector<int>& res, int& seen, int t, int cap) {
							seen++;
							if ((int)res.size() < cap)
								res.push_back(t);
							else {
								int j = Math::RandInt(0, seen);
								if (j < cap)
									res[j] = t;
							}
						};

						std::vector<int> offSel, defSel, resSel;
						int offSeen = 0, defSeen = 0, resSeen = 0;
						float goalMedian = goalScore.defined() ? goalScore.median().item<float>() : 0.0f;
						torch::Tensor goalScoreCpu = goalScore.defined() ? goalScore.to(torch::kCPU).contiguous() : torch::Tensor();
						const float* goalScoreData = goalScore.defined() ? goalScoreCpu.data_ptr<float>() : nullptr;
						torch::Tensor optValueCpu = optValue.defined() ? optValue.to(torch::kCPU).contiguous() : torch::Tensor();
						const float* optValueData = optValue.defined() ? optValueCpu.data_ptr<float>() : nullptr;
						bool hasFlags = combinedTraj.stratumFlags.size() == (size_t)trajN;
						for (int t = 0; t < trajN; t++) {
							if (goalScoreData && goalScoreData[t] >= goalMedian)
								fnReservoir(offSel, offSeen, t, OPT_MAX_CANDS_PER_STRATUM);
							if (hasFlags) {
								if (combinedTraj.stratumFlags[t] & OPT_FLAG_DEFENSIVE)
									fnReservoir(defSel, defSeen, t, OPT_MAX_CANDS_PER_STRATUM);
								if (combinedTraj.stratumFlags[t] & OPT_FLAG_RESOURCE)
									fnReservoir(resSel, resSeen, t, OPT_MAX_CANDS_PER_STRATUM);
							}
						}
						auto fnValueAt = [&](int t) {
							return optValueData ? optValueData[t] : 0.0f;
						};
						for (int t : offSel) fnPushBankRow(
							candRows[GCRLOptionality::STRATUM_OFFENSIVE],
							candValues[GCRLOptionality::STRATUM_OFFENSIVE], t, fnValueAt(t));
						for (int t : defSel) fnPushBankRow(
							candRows[GCRLOptionality::STRATUM_DEFENSIVE],
							candValues[GCRLOptionality::STRATUM_DEFENSIVE], t, fnValueAt(t));
						for (int t : resSel) fnPushBankRow(
							candRows[GCRLOptionality::STRATUM_RESOURCE],
							candValues[GCRLOptionality::STRATUM_RESOURCE], t, fnValueAt(t));

						// Optionally top up the defensive stratum from the frontier buffer
						// (states already at the competence frontier in own-net regions).
						if (config.ppo.frontierBuffer && (int)config.ppo.frontierBuffer->Size() >= config.ppo.frontierBufferMinFill) {
							int extra = OPT_MAX_CANDS_PER_STRATUM / 2;
							for (int i = 0; i < extra; i++) {
								RLGC::FrontierSnapshot snap;
								if (!config.ppo.frontierBuffer->Sample(snap))
									break;
								// Defensive third in the canonical (blue) frame: ball deep at y < 0
								if (snap.ball.pos.y < -OPT_DEFENSIVE_THIRD_Y) {
									float row[6];
									MakeObsSpaceBallRow(snap.ball, false, config, obsStat, row);
									candRows[GCRLOptionality::STRATUM_DEFENSIVE].insert(
										candRows[GCRLOptionality::STRATUM_DEFENSIVE].end(), row, row + 6);
									candValues[GCRLOptionality::STRATUM_DEFENSIVE].push_back(0.0f);
								}
							}
						}

						// 3. Refresh the bank (FIFO per stratum) + re-embed under the frozen psi
						opt->RefreshBank(candRows, candValues, (int64_t)totalIterations, report);

						// 4. phi_opt over the batch
						torch::Tensor phiReachOnly;
						torch::Tensor phiOpt = opt->ComputePhiOpt(tStates, &phiReachOnly); // [N] cpu, or {} if bank empty
						if (phiOpt.defined()) {
							report["Opt/Phi Mean"] = phiOpt.mean().item<float>();
							report["Opt/Phi Std"] = phiOpt.std(false).item<float>();
							if (phiReachOnly.defined()) {
								torch::Tensor phiLift = phiOpt - phiReachOnly;
								report["Opt/Phi ReachOnly Mean"] = phiReachOnly.mean().item<float>();
								report["Opt/Phi Value Lift Mean"] = phiLift.mean().item<float>();
								report["Opt/Phi Value Lift Std"] = phiLift.std(false).item<float>();
							}
							report["OptRefine/Enabled"] = config.ppo.optRefineGoals ? 1.0 : 0.0;
							report["OptRefine/Accepted Fraction"] = opt->lastRefineAcceptedFraction;
							report["OptRefine/State Fraction"] = opt->lastRefineStateFraction;
							report["OptRefine/Goal Delta Norm"] = opt->lastRefineGoalDeltaNorm;
							report["OptRefine/Score Gain Mean"] = opt->lastRefineScoreGainMean;
							report["OptRefine/Phi Lift Mean"] = opt->lastRefinePhiLiftMean;
							if (hasFlags) {
								std::vector<uint8_t> touchMask(trajN);
								for (int t = 0; t < trajN; t++)
									touchMask[t] = (combinedTraj.stratumFlags[t] & OPT_FLAG_TOUCH) ? 1 : 0;
								torch::Tensor tTouch = torch::from_blob(touchMask.data(), { trajN },
									torch::TensorOptions().dtype(torch::kUInt8)).to(torch::kBool);
								if (tTouch.any().item<bool>())
									report["Opt/Phi At Touch"] = phiOpt.masked_select(tTouch).mean().item<float>();
							}

							// 5. Masked potential delta + running normalizer
							torch::Tensor rOptRaw = GCRLOptionality::MaskedPotentialDelta(
								phiOpt, combinedTraj.terminals.data(), trajN, config.ppo.gaeGamma);
							double batchMean = rOptRaw.mean().item<double>();
							double batchVar = rOptRaw.var(false).item<double>();
							constexpr double OPT_NORM_DECAY = 0.999;
							optRewardMeanEMA = OPT_NORM_DECAY * optRewardMeanEMA + (1 - OPT_NORM_DECAY) * batchMean;
							optRewardVarEMA = OPT_NORM_DECAY * optRewardVarEMA + (1 - OPT_NORM_DECAY) * batchVar;
							double normStd = std::sqrt(optRewardVarEMA + 1e-8);

							// 6. Weight schedule + interlock
							float effectiveOptWeight = fnGetGatedAnnealedRange(
								config.ppo.optWeight, config.ppo.optWeightFinal,
								-1, config.ppo.optAnnealSteps,
								optWeightAnnealStartTS, optWeightAnnealProgressTS,
								config.ppo.curriculumAnnealTouchRatioGate, touchRatioEMA);
							bool interlock = ppo->curGCRLAdvScale >= 0.5f;
							if (interlock)
								effectiveOptWeight *= 0.5f;
							report["Opt/Interlock Active"] = interlock ? 1.0 : 0.0;

							bool burnIn = optBurnInItersDone < config.ppo.optBurnInIters;
							optBurnInItersDone++;

							torch::Tensor injected = rOptRaw * (effectiveOptWeight / (float)normStd);
							report["Opt/Reward Std (pre-norm)"] = (float)std::sqrt(RS_MAX(batchVar, 0.0));
							report["Opt/Effective Weight"] = burnIn ? 0.0f : effectiveOptWeight;
							report["Opt/Reward Mean"] = burnIn ? 0.0f : injected.mean().item<float>();
							float meanAbsTotal = tRewards.abs().mean().item<float>();
							report["Opt/Reward Share"] = (!burnIn && meanAbsTotal > 1e-8f)
								? injected.abs().mean().item<float>() / meanAbsTotal : 0.0f;

							// Hover monitor: Pearson(rOpt, curriculum reward — dominant term is the chase reward)
							if (combinedTraj.curriculumRewards.size() == (size_t)trajN) {
								torch::Tensor tChase = torch::tensor(combinedTraj.curriculumRewards);
								torch::Tensor a = rOptRaw - rOptRaw.mean();
								torch::Tensor b = tChase - tChase.mean();
								float denom = (a.norm() * b.norm()).item<float>();
								report["Opt/Corr With Chase"] = denom > 1e-8f ? (a * b).sum().item<float>() / denom : 0.0f;
							}

							if (!burnIn)
								tRewards = tRewards + injected;
						}
					}

					// States we truncated at (there could be none)
					torch::Tensor tNextTruncStates;
					if (!combinedTraj.nextStates.empty())
						tNextTruncStates = torch::tensor(combinedTraj.nextStates).reshape({ -1, obsSize });

					report["Average Step Reward"] = tRewards.mean().item<float>();
					report["Collected Timesteps"] = stepsCollected;

					torch::Tensor tValPreds;
					torch::Tensor tTruncValPreds;

					if (ppo->device.is_cpu()) {
						// Predict values all at once
						tValPreds = ppo->InferCritic(tStates.to(ppo->device, RG_H2D_NONBLOCKING(ppo->device), true)).cpu();
						if (tNextTruncStates.defined())
							tTruncValPreds = ppo->InferCritic(tNextTruncStates.to(ppo->device, RG_H2D_NONBLOCKING(ppo->device), true)).cpu();
					} else {
						// Predict values using minibatching
						tValPreds = torch::zeros({ (int64_t)combinedTraj.Length() });
						for (int i = 0; i < combinedTraj.Length(); i += ppo->config.miniBatchSize) {
							int start = i;
							int end = RS_MIN(i + ppo->config.miniBatchSize, combinedTraj.Length());
							torch::Tensor tStatesPart = tStates.slice(0, start, end);

							auto valPredsPart = ppo->InferCritic(tStatesPart.to(ppo->device, RG_H2D_NONBLOCKING(ppo->device), true)).cpu();
							RG_ASSERT(valPredsPart.size(0) == (end - start));
							tValPreds.slice(0, start, end).copy_(valPredsPart, true);
						}

						if (tNextTruncStates.defined()) {
							// This really just should never happen
							// If this is ever actually a real problem in a legitimate use case, ping Zealan in the dead of night
							RG_ASSERT(tNextTruncStates.size(0) <= ppo->config.miniBatchSize);

							tTruncValPreds = ppo->InferCritic(tNextTruncStates.to(ppo->device, RG_H2D_NONBLOCKING(ppo->device), true)).cpu();
						}
					}

					// Skip when no episode fully ended this batch (would divide by zero -> inf on the graph)
					float terminalPortion = (tTerminals == 1).to(torch::kFloat32).mean().item<float>();
					if (terminalPortion > 0)
						report["Episode Length"] = 1.f / terminalPortion;

					Timer gaeTimer = {};
					// Run GAE (rewards standardized by each step's own mode's return STD)
					// Default 1 (not 0) preserves the old standardizeReturns=false behavior of
					// clipping raw rewards at rewardClipRange.
					FList returnStds(numModes, 1.0f);
					if (returnStats)
						for (int m = 0; m < numModes; m++)
							returnStds[m] = (float)(*returnStats)[m].GetSTD();

					torch::Tensor tAdvantages, tTargetVals, tReturns;
					float rewClipPortion;
					GAE::Compute(
						tRewards, tTerminals, tValPreds, tTruncValPreds,
						tAdvantages, tTargetVals, tReturns, rewClipPortion,
						config.ppo.gaeGamma, config.ppo.gaeLambda,
						returnStds, combinedTraj.modeIds.data(),
						config.ppo.rewardClipRange
					);
					report["GAE Time"] = gaeTimer.Elapsed();
					report["Clipped Reward Portion"] = rewClipPortion;

					if (returnStats) {
						for (int m = 0; m < numModes; m++)
							report[numModes > 1 ? ("GAE/Returns STD/" + modeNames[m]) : "GAE/Returns STD"] = returnStds[m];

						int numToIncrement = RS_MIN(config.maxReturnSamples, tReturns.size(0));
						const float* returnsData = tReturns.contiguous().data_ptr<float>();
						for (int i = 0; i < numToIncrement; i++) {
							int idx = Math::RandInt(0, (int)tReturns.size(0));
							(*returnStats)[combinedTraj.modeIds[idx]].Increment(returnsData[idx]);
						}
					}
					report["GAE/Avg Return"] = tReturns.abs().mean().item<float>();
					report["GAE/Avg Advantage"] = tAdvantages.abs().mean().item<float>();
					report["GAE/Avg Val Target"] = tTargetVals.abs().mean().item<float>();

					// Set experience buffer
					experience.data.actions = tActions;
					experience.data.logProbs = tLogProbs;
					experience.data.actionMasks = tActionMasks;
					experience.data.states = tStates;
					experience.data.advantages = tAdvantages;
					experience.data.targetValues = tTargetVals;
					if (numModes > 1)
						experience.data.modeIds = tModeIds;

					if (config.ppo.useGCRL) {
						experience.data.actionComps    = tActionComps;
						experience.data.futureGoals    = torch::tensor(combinedTraj.futureGoals).reshape({ -1, 6 });
						experience.data.carFutureGoals = torch::tensor(combinedTraj.carFutureGoals).reshape({ -1, 6 });
					}
				}

				// Free CUDA cache.
				// emptyCache() does a device-wide hipFree sync that stalls the concurrent
				// collector overlap and largely defeats the caching allocator, so don't run
				// it every iteration -- only periodically to keep fragmentation in check.
				// Bump kEmptyCacheInterval up if you still see throughput dips, down (or to 1)
				// if you hit OOM from cached-block fragmentation.
#ifdef RG_CUDA_SUPPORT
				constexpr uint64_t kEmptyCacheInterval = 32;
				if (ppo->device.is_cuda() && (totalIterations % kEmptyCacheInterval == 0))
					c10::cuda::CUDACachingAllocator::emptyCache();
#endif

				// Learn
				Timer learnTimer = {};
				ppo->Learn(experience, report, isFirstIteration);
				report["PPO Learn Time"] = learnTimer.Elapsed();

				// Set metrics
				float consumptionTime = consumptionTimer.Elapsed();
				report["Collection Time"] = batch.collectionTime;
				report["Consumption Time"] = consumptionTime;
				report["Collection Steps/Second"] = stepsCollected / batch.collectionTime;
				report["Consumption Steps/Second"] = stepsCollected / consumptionTime;
				report["Overall Steps/Second"] = stepsCollected / iterationTimer.Elapsed();

				uint64_t prevTimesteps = totalTimesteps;
				totalTimesteps += stepsCollected;
				report["Total Timesteps"] = totalTimesteps;
				totalIterations++;
				report["Total Iterations"] = totalIterations;

				if (versionMgr)
					versionMgr->OnIteration(ppo, report, totalTimesteps, prevTimesteps);

				if (esManager)
					esManager->OnIteration(ppo, report, totalTimesteps, prevTimesteps);

				if (saveQueued) {
					if (doAsync) {
						// Join the collector before saving so no torch op is mid-flight during teardown.
						{
							std::lock_guard<std::mutex> lk(handoff.m);
							handoff.stop = true;
						}
						handoff.cvGo.notify_one();
						collectorThread.join();
					}
					if (!config.checkpointFolder.empty())
						Save();
					exit(0);
				}

				if (!config.checkpointFolder.empty()) {
					if (totalTimesteps / config.tsPerSave > prevTimesteps / config.tsPerSave) {
						// Auto-save
						Save();
					}
				}

				report.Finish();

				if (metricSender)
					metricSender->Send(report);

				report.Display(
					{
						"Average Step Reward",
						"Policy Entropy",
						"Mean KL Divergence",
						"",
						"Policy Update Magnitude",
						"Critic Update Magnitude",
						"Shared Head Update Magnitude",
						"",
						"Collection Steps/Second",
						"Consumption Steps/Second",
						"Overall Steps/Second",
						"",
						"Collection Time",
						"-Inference Time",
						"-Env Step Time",
						"Consumption Time",
						"-GAE Time",
						"-PPO Learn Time",
						"",
						"GCRL/Adv Scale",
						"GCRL Gate/Influence",
						"GCRL Gate/Mean",
						"GCRL Gate/Delta Mean",
						"GCRL Gate/Gated Reward Raw",
						"GCRL Gate/Gated Reward Applied",
						"",
						"Curriculum/Scale",
						"Curriculum/Touch Ratio EMA",
						"Aerial Curriculum/Scale",
						"Curriculum/High Air Touch Ratio EMA",
						"",
						"Frontier/Buffer Fill",
						"Frontier/Candidates Per Iter",
						"Frontier/Reset Fraction Actual",
						"HER/Selected Offset P50",
						"HER/Uniform Fraction Actual",
						"Adaptive/Gate Target Vel",
						"Adaptive/StrongTouch Floor",
						"Adaptive/Touch Samples Per Iter",
						"Opt/Phi Mean",
						"Opt/Phi ReachOnly Mean",
						"Opt/Phi Value Lift Mean",
						"Opt/Bank Value Std",
						"Opt/Value Weight",
						"Opt/Reward Share",
						"Opt/Effective Weight",
						"Opt/Interlock Active",
						"OptRefine/Enabled",
						"OptRefine/Accepted Fraction",
						"OptRefine/State Fraction",
						"OptRefine/Goal Delta Norm",
						"OptRefine/Score Gain Mean",
						"OptRefine/Phi Lift Mean",
						"SORS/Reward Scale",
						"SORS/Loss",
						"SORS/Pair Accuracy",
						"SORS/Replay Windows",
						"SORS/Positive Windows",
						"SORS/Avg Label",
						"SORS/Avg Reward",
						"SORS/Avg Abs Reward",
						"SORS/Pred Window Return",
						"",
						"Collected Timesteps",
						"Total Timesteps",
						"Total Iterations"
					}
				);
		}

	} catch (std::exception& e) {
		RG_ERR_CLOSE("Exception thrown during main learner loop: " << e.what());
	}
}

GGL::Learner::~Learner() {
	delete ppo;
	delete versionMgr;
	delete esManager;
	delete metricSender;
	delete renderSender;
	delete envSet;
	delete returnStats;
	delete obsStat;
	delete contactBallSpeedHist;
	delete contactHitForceHist;
	pybind11::finalize_interpreter();
}
