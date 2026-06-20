#include "Learner.h"

#include <GigaLearnCPP/PPO/PPOLearner.h>
#include <GigaLearnCPP/PPO/ExperienceBuffer.h>

#include <torch/cuda.h>
#if defined(__APPLE__) && __has_include(<torch/mps.h>)
#include <torch/mps.h>
#define RG_MPS_SUPPORT
#endif
#include <nlohmann/json.hpp>
#include <pybind11/embed.h>
#include <cmath>

#ifdef RG_CUDA_SUPPORT
#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)
#include <c10/hip/HIPCachingAllocator.h>
#else
#include <c10/cuda/CUDACachingAllocator.h>
#endif
#endif
#include <private/GigaLearnCPP/PPO/ExperienceBuffer.h>
#include <private/GigaLearnCPP/PPO/GAE.h>
#include <private/GigaLearnCPP/PolicyVersionManager.h>

#include "Util/KeyPressDetector.h"
#include <private/GigaLearnCPP/Util/WelfordStat.h>
#include "Util/AvgTracker.h"
#include <RLGymCPP/CommonValues.h>
#include <RLGymCPP/Gamestates/StateUtil.h>

using namespace RLGC;

static void AppendNormalizedGoal(FList& out, const Vec& pos, const Vec& vel, const GGL::ContrastiveGoalConfig& cfg) {
	out += pos.x / cfg.posScaleX;
	out += pos.y / cfg.posScaleY;
	out += pos.z / cfg.posScaleZ;
	out += vel.x / cfg.velScale;
	out += vel.y / cfg.velScale;
	out += vel.z / cfg.velScale;
}

static void AppendAchievedGoal(FList& out, const GameState& state, const Player& player, const GGL::ContrastiveGoalConfig& cfg) {
	PhysState ball = InvertPhys(state.ball, player.team == Team::ORANGE);
	AppendNormalizedGoal(out, ball.pos, ball.vel, cfg);
}

static void AppendScoringGoal(FList& out, const GameState& state, const Player& player, const GGL::ContrastiveGoalConfig& cfg) {
	// Canonical frame: the player's attacking goal is always at +y.
	PhysState ball = InvertPhys(state.ball, player.team == Team::ORANGE);

	float targetXRange = CommonValues::GOAL_WIDTH_FROM_CENTER - CommonValues::BALL_RADIUS;
	float minTargetZ = CommonValues::BALL_RADIUS;
	float maxTargetZ = CommonValues::GOAL_HEIGHT - CommonValues::BALL_RADIUS;

	// (2A) Least-resistance score point: lead the ball along its own velocity,
	// then clamp into the goal mouth. A ball already flying at the far post leads
	// to the far post (where it would cross), not the perpendicular-nearest
	// point. The lead is scaled by how much the velocity points at the goal (so
	// a ball moving away/sideways doesn't over-lead) and capped, and it vanishes
	// as the ball slows — degrading continuously to the closest mouth point with
	// no piecewise switch.
	Vec goalCenter = Vec(0, CommonValues::BACK_WALL_Y, CommonValues::GOAL_HEIGHT * 0.5f);
	Vec toGoal = goalCenter - ball.pos;
	Vec dirToGoal = (toGoal.Length() > 1e-3f) ? toGoal.Normalized() : Vec(0, 1, 0);

	float speed = ball.vel.Length();
	float align = (speed > 1e-3f) ? (ball.vel / speed).Dot(dirToGoal) : 0.f;
	float leadScale = RS_CLAMP(align, 0.f, 1.f);
	Vec lead = ball.vel * (cfg.scoringGoalLeadTime * leadScale);
	float leadLen = lead.Length();
	if (cfg.scoringGoalMaxLead > 0 && leadLen > cfg.scoringGoalMaxLead)
		lead = lead * (cfg.scoringGoalMaxLead / leadLen);

	Vec anchor = ball.pos + lead;

	Vec targetPos = {};
	targetPos.x = RS_CLAMP(anchor.x, -targetXRange, targetXRange);
	targetPos.y = CommonValues::BACK_WALL_Y;
	targetPos.z = RS_CLAMP(anchor.z, minTargetZ, maxTargetZ);

	Vec direction = targetPos - ball.pos;
	if (direction.Length() < 1e-3f)
		direction = Vec(0, 1, 0);
	else
		direction = direction.Normalized();

	float minSpeed = RS_MAX(0.f, cfg.targetSpeed - cfg.targetSpeedJitter);
	float maxSpeed = RS_MAX(minSpeed, cfg.targetSpeed + cfg.targetSpeedJitter);
	float targetSpeed = (minSpeed == maxSpeed) ? minSpeed : RocketSim::Math::RandFloat(minSpeed, maxSpeed);
	Vec targetVel = direction * targetSpeed;
	AppendNormalizedGoal(out, targetPos, targetVel, cfg);
}

static int GetArenaIdxForPlayer(const EnvSet& envSet, int playerIdx) {
	for (int arenaIdx = 0; arenaIdx < envSet.state.arenaPlayerStartIdx.size(); arenaIdx++) {
		int start = envSet.state.arenaPlayerStartIdx[arenaIdx];
		int stop = (arenaIdx + 1 < envSet.state.arenaPlayerStartIdx.size()) ?
			envSet.state.arenaPlayerStartIdx[arenaIdx + 1] : envSet.state.numPlayers;
		if (playerIdx >= start && playerIdx < stop)
			return arenaIdx;
	}
	RG_ERR_CLOSE("Could not find arena for player index " << playerIdx);
	return -1;
}

static bool CanMoveTensorToDevice(at::Device device) {
	try {
		torch::Tensor t = torch::tensor(0);
		t = t.to(device);
		t = t.cpu();
		return true;
	} catch (...) {
		return false;
	}
}

static bool IsMPSAvailable() {
#ifdef RG_MPS_SUPPORT
	return torch::mps::is_available();
#else
	return false;
#endif
}

static at::Device UseCudaDevice() {
	RG_LOG("\tUsing CUDA GPU device...");

	bool deviceTestFailed = !CanMoveTensorToDevice(at::Device(at::kCUDA));
	if (!torch::cuda::is_available() || deviceTestFailed)
		RG_ERR_CLOSE(
			"Learner::Learner(): Can't use CUDA GPU because " <<
			(torch::cuda::is_available() ? "libtorch cannot access the GPU" : "CUDA is not available to libtorch") << ".\n" <<
			"Make sure your libtorch comes with CUDA support, and that CUDA is installed properly."
		)

	return at::Device(at::kCUDA);
}

static at::Device UseMPSDevice() {
#ifdef RG_MPS_SUPPORT
	RG_LOG("\tUsing MPS GPU device...");

	bool deviceTestFailed = !CanMoveTensorToDevice(at::Device(at::kMPS));
	if (!torch::mps::is_available() || deviceTestFailed)
		RG_ERR_CLOSE(
			"Learner::Learner(): Can't use MPS GPU because " <<
			(torch::mps::is_available() ? "libtorch cannot access the GPU" : "MPS is not available to libtorch") << ".\n" <<
			"Make sure your libtorch comes with MPS support and that you are running on a supported macOS device."
		)

	return at::Device(at::kMPS);
#else
	RG_ERR_CLOSE(
		"Learner::Learner(): Can't use MPS GPU because this build of libtorch does not provide torch/mps.h.\n" <<
		"Use a macOS libtorch build with MPS support or choose AUTO/CPU/CUDA."
	)
	return at::Device(at::kCPU);
#endif
}

static at::Device ResolveLearnerDevice(GGL::LearnerDeviceType deviceType) {
	if (deviceType == GGL::LearnerDeviceType::CPU) {
		RG_LOG("\tUsing CPU device...");
		return at::Device(at::kCPU);
	}

	if (deviceType == GGL::LearnerDeviceType::GPU_CUDA)
		return UseCudaDevice();

	if (deviceType == GGL::LearnerDeviceType::GPU_MPS)
		return UseMPSDevice();

#ifdef __APPLE__
	if (IsMPSAvailable())
		return UseMPSDevice();
	if (torch::cuda::is_available())
		return UseCudaDevice();
#else
	if (torch::cuda::is_available())
		return UseCudaDevice();
	if (IsMPSAvailable())
		return UseMPSDevice();
#endif

	RG_LOG("\tUsing CPU device...");
	return at::Device(at::kCPU);
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

	if (config.numGames <= 0)
		RG_ERR_CLOSE("LearnerConfig::numGames must be positive");
	if (config.tickSkip <= 0)
		RG_ERR_CLOSE("LearnerConfig::tickSkip must be positive");
	if (config.tsPerSave <= 0)
		RG_ERR_CLOSE("LearnerConfig::tsPerSave must be positive after defaulting");
	if (config.ppo.tsPerItr <= 0)
		RG_ERR_CLOSE("PPOLearnerConfig::tsPerItr must be positive");
	if (config.addRewardsToMetrics && config.rewardSampleRandInterval <= 0)
		RG_ERR_CLOSE("LearnerConfig::rewardSampleRandInterval must be positive when reward metrics are enabled");
	if (config.renderMode && (config.renderTimeScale <= 0 || !std::isfinite(config.renderTimeScale)))
		RG_ERR_CLOSE("LearnerConfig::renderTimeScale must be positive and finite in render mode");
	if (config.savePolicyVersions || config.skillTracker.enabled || config.trainAgainstOldVersions) {
		if (config.tsPerVersion <= 0)
			RG_ERR_CLOSE("LearnerConfig::tsPerVersion must be positive when policy versions are enabled");
		if (config.maxOldVersions <= 0)
			RG_ERR_CLOSE("LearnerConfig::maxOldVersions must be positive when policy versions are enabled");
	}

	RG_LOG("Learner::Learner():");

	if (config.randomSeed == -1)
		config.randomSeed = RS_CUR_MS();

	RG_LOG("\tCheckpoint Save/Load Dir: " << config.checkpointFolder);

	torch::manual_seed(config.randomSeed);

	at::Device device = ResolveLearnerDevice(config.deviceType);

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
		if (config.standardizeReturns) {
			this->returnStat = new WelfordStat();
		} else {
			this->returnStat = NULL;
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

	if (config.sendMetrics)
		j["run_id"] = metricSender->curRunID;

	if (returnStat)
		j["return_stat"] = returnStat->ToJSON();
	if (obsStat)
		j["obs_stat"] = obsStat->ToJSON();

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

	if (j.contains("run_id"))
		runID = j["run_id"];

	if (returnStat)
		returnStat->ReadFromJSON(j["return_stat"]);
	if (obsStat)
		obsStat->ReadFromJSON(j["obs_stat"]);

	if (versionMgr)
		versionMgr->LoadRunningStatsFromJSON(j);
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

	if (!KeyPressDetector::HasTerminalInput()) {
		RG_LOG("No interactive stdin; Q-to-save key listener disabled. Use the runner stop command or SIGTERM to stop.");
		return;
	}

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
		std::atomic<bool> saveQueued{false};
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

					envSet->StepFirstHalf(true);

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

					ppo->InferActions(
						tStates.to(ppo->device, true), tActionMasks.to(ppo->device, true), 
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
		std::atomic<bool> saveQueued{false};
		std::thread keyPressThread;
		StartQuitKeyThread(saveQueued, keyPressThread);

		ExperienceBuffer experience = ExperienceBuffer(config.randomSeed, torch::kCPU);

		int numPlayers = envSet->state.numPlayers;

		struct Trajectory {
			FList states, nextStates, rewards, logProbs;
			FList achievedGoals, herGoals, scoringGoals;
			std::vector<uint8_t> actionMasks;
			std::vector<uint8_t> gcrlTrainMask; // (2B) per-row: 1 = use this row to train the contrastive critic
			std::vector<int8_t> terminals;
			std::vector<int32_t> actions;
			std::vector<int64_t> segmentIds, segmentSteps;

			void Clear() {
				*this = Trajectory();
			}

			void Append(const Trajectory& other) {
				states += other.states;
				nextStates += other.nextStates;
				rewards += other.rewards;
				logProbs += other.logProbs;
				achievedGoals += other.achievedGoals;
				herGoals += other.herGoals;
				scoringGoals += other.scoringGoals;
				gcrlTrainMask += other.gcrlTrainMask;
				actionMasks += other.actionMasks;
				terminals += other.terminals;
				actions += other.actions;
				segmentIds += other.segmentIds;
				segmentSteps += other.segmentSteps;
			}

			size_t Length() const {
				return actions.size();
			}
		};

		auto trajectories = std::vector<Trajectory>(numPlayers, Trajectory{});
		std::vector<int64_t> curSegmentIds(numPlayers);
		std::vector<int64_t> curSegmentSteps(numPlayers, 0);
		int64_t nextSegmentId = 1;
		for (int i = 0; i < numPlayers; i++)
			curSegmentIds[i] = nextSegmentId++;

		int maxEpisodeLength = (int)(config.ppo.maxEpisodeDuration * (120.f / config.tickSkip));

		while (true) {
			Report report = {};

			bool isFirstIteration = (totalTimesteps == 0);

			// TODO: Old version switching messes up the gameplay potentially
			GGL::PolicyVersion* oldVersion = NULL;
			std::vector<bool> oldVersionPlayerMask;
			std::vector<int> newPlayerIndices = {}, oldPlayerIndices = {};
			torch::Tensor tNewPlayerIndices, tOldPlayerIndices;

			for (int i = 0; i < numPlayers; i++)
				newPlayerIndices.push_back(i);

			if (config.trainAgainstOldVersions) {
				RG_ASSERT(config.trainAgainstOldChance >= 0 && config.trainAgainstOldChance <= 1);
				bool shouldTrainAgainstOld =
					(RocketSim::Math::RandFloat() < config.trainAgainstOldChance)
					&& !versionMgr->versions.empty()
					&& !render;

				if (shouldTrainAgainstOld) {
					// Set up training against old versions

					int oldVersionIdx = RocketSim::Math::RandInt(0, versionMgr->versions.size());
					oldVersion = &versionMgr->versions[oldVersionIdx];

					Team oldVersionTeam = Team(RocketSim::Math::RandInt(0, 2)); 
					
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

			// When training against an old version, the players controlled by the old policy are not
			// collected this iteration, but their arenas keep stepping and resetting. If we kept their
			// partial trajectories around, the next time they're collected we'd glue pre-freeze data
			// onto new data across one or more env resets with no terminal in between, corrupting GAE.
			// So discard any partial trajectory for a player we won't collect this iteration.
			if (oldVersion) {
				std::vector<bool> willCollect(numPlayers, false);
				for (int idx : newPlayerIndices)
					willCollect[idx] = true;
				for (int p = 0; p < numPlayers; p++) {
					if (!willCollect[p] && trajectories[p].Length() > 0) {
						trajectories[p].Clear();
						curSegmentIds[p] = nextSegmentId++;
						curSegmentSteps[p] = 0;
					}
				}
			}

			int numRealPlayers = oldVersion ? newPlayerIndices.size() : envSet->state.numPlayers;

			int stepsCollected = 0;
			{ // Generate experience

				// Only contains complete episodes
				auto combinedTraj = Trajectory();
				FList herSelectedOffsets;
				int herTotalRows = 0;

				auto relabelHERGoals = [&](Trajectory& traj) {
					if (!config.ppo.contrastiveGoal.enabled)
						return;

					int n = (int)traj.Length();
					if (n <= 0)
						return;

					int minOffset = RS_MAX(1, config.ppo.contrastiveGoal.herMinOffset);
					int maxOffsetCfg = RS_MAX(minOffset, config.ppo.contrastiveGoal.herMaxOffset);
					float shortBiasPower = RS_MAX(1e-3f, config.ppo.contrastiveGoal.herShortBiasPower);
					float goalwardBias = RS_CLAMP(config.ppo.contrastiveGoal.herGoalwardBias, 0.f, 1.f);

					if ((int)traj.achievedGoals.size() != n * 6)
						RG_ERR_CLOSE("GCRL HER relabeling expected " << n << " achieved goal rows, got " << (traj.achievedGoals.size() / 6));

					// (2B) Admit this match to contrastive training only if the ball
					// actually moved during it; a dead kickoff->timeout match would just
					// reteach the degenerate stationary-ball manifold. achievedGoals hold
					// the ball's canonical (pos, vel) normalized by posScale*/velScale, so
					// compare the normalized speed against the configured threshold.
					float moveThreshNorm = config.ppo.contrastiveGoal.gcrlMinBallMoveSpeed / RS_MAX(1e-6f, config.ppo.contrastiveGoal.velScale);
					float moveThreshNormSq = moveThreshNorm * moveThreshNorm;
					bool ballMoved = false;
					for (int t = 0; t < n && !ballMoved; t++) {
						float vx = traj.achievedGoals[(size_t)t * 6 + 3];
						float vy = traj.achievedGoals[(size_t)t * 6 + 4];
						float vz = traj.achievedGoals[(size_t)t * 6 + 5];
						if (vx * vx + vy * vy + vz * vz >= moveThreshNormSq)
							ballMoved = true;
					}
					traj.gcrlTrainMask.assign((size_t)n, ballMoved ? 1 : 0);

					traj.herGoals.resize((size_t)n * 6);
					for (int t = 0; t < n; t++) {
						herTotalRows++;
						int remaining = n - t - 1;
						int selectedOffset = 0;
						if (remaining > 0) {
							int maxOffset = RS_MIN(maxOffsetCfg, remaining);
							if (maxOffset >= minOffset) {
								if (goalwardBias > 0.f && RocketSim::Math::RandFloat() < goalwardBias) {
									// (2C) Target the most-goalward (max canonical +y) achieved
									// state within the offset window, so real near-net states
									// populate the goal space and the (2A) scoring goal is
									// in-distribution. Still a true achieved state (valid positive).
									int bestOffset = minOffset;
									float bestY = traj.achievedGoals[(size_t)(t + minOffset) * 6 + 1];
									for (int o = minOffset + 1; o <= maxOffset; o++) {
										float y = traj.achievedGoals[(size_t)(t + o) * 6 + 1];
										if (y > bestY) {
											bestY = y;
											bestOffset = o;
										}
									}
									selectedOffset = bestOffset;
								} else {
									float u = RocketSim::Math::RandFloat();
									float biased = powf(u, shortBiasPower);
									int span = maxOffset - minOffset + 1;
									selectedOffset = minOffset + RS_MIN(span - 1, (int)floorf(biased * span));
								}
							} else {
								selectedOffset = remaining;
							}
						}

						int target = t + selectedOffset;
						for (int d = 0; d < 6; d++)
							traj.herGoals[(size_t)t * 6 + d] = traj.achievedGoals[(size_t)target * 6 + d];
						if (selectedOffset > 0)
							herSelectedOffsets += (float)selectedOffset;
					}
				};

				Timer collectionTimer = {};
				{ // Collect timesteps
					RG_NO_GRAD;

					float inferTime = 0;
					float envStepTime = 0;

					for (int step = 0; combinedTraj.Length() < config.ppo.tsPerItr || render; step++, stepsCollected += numRealPlayers) {
						Timer stepTimer = {};
						envSet->Reset();
						envStepTime += stepTimer.Elapsed();

						for (float f : envSet->state.obs.data)
							if (isnan(f) || isinf(f))
								RG_ERR_CLOSE("Obs builder produced a NaN/inf value");

						// Snapshot of this step's obs standardization stats (empty when disabled).
						// Reused below for the truncation next-state so it gets the SAME normalization
						// the stored states received (otherwise the critic would be bootstrapped on raw obs).
						std::vector<double> obsMean, obsStd;
						auto standardizeObsRow = [&](float* row) {
							if (obsMean.empty())
								return;
							for (int j = 0; j < obsSize; j++)
								row[j] = (float)((row[j] - obsMean[j]) / obsStd[j]);
						};

						if (!render && obsStat) {
							// TODO: This samples from old versions too
							int numSamples = RS_MIN(envSet->state.numPlayers, config.maxObsSamples);
							for (int i = 0; i < numSamples; i++) {
								int idx = Math::RandInt(0, envSet->state.numPlayers);
								obsStat->IncrementRow(&envSet->state.obs.At(idx, 0));
							}

							obsMean = obsStat->GetMean();
							obsStd = obsStat->GetSTD();
							for (double& f : obsMean)
								f = RS_CLAMP(f, -config.maxObsMeanRange, config.maxObsMeanRange);
							for (double& f : obsStd)
								f = RS_MAX(f, config.minObsSTD);
							for (int i = 0; i < envSet->state.numPlayers; i++)
								standardizeObsRow(&envSet->state.obs.At(i, 0));
						}

						torch::Tensor tActions, tLogProbs;
						torch::Tensor tStates = DIMLIST2_TO_TENSOR<float>(envSet->state.obs);
						torch::Tensor tActionMasks = DIMLIST2_TO_TENSOR<uint8_t>(envSet->state.actionMasks);

						if (!render) {
							for (int newPlayerIdx : newPlayerIndices) {
								trajectories[newPlayerIdx].states += envSet->state.obs.GetRow(newPlayerIdx);
								trajectories[newPlayerIdx].actionMasks += envSet->state.actionMasks.GetRow(newPlayerIdx);

								if (config.ppo.contrastiveGoal.enabled) {
									int arenaIdx = GetArenaIdxForPlayer(*envSet, newPlayerIdx);
									int localPlayerIdx = newPlayerIdx - envSet->state.arenaPlayerStartIdx[arenaIdx];
									const GameState& state = envSet->state.gameStates[arenaIdx];
									const Player& player = state.players[localPlayerIdx];

									AppendScoringGoal(trajectories[newPlayerIdx].scoringGoals, state, player, config.ppo.contrastiveGoal);
									trajectories[newPlayerIdx].segmentIds.push_back(curSegmentIds[newPlayerIdx]);
									trajectories[newPlayerIdx].segmentSteps.push_back(curSegmentSteps[newPlayerIdx]);
								}
							}
						}

						envSet->StepFirstHalf(true);

						Timer inferTimer = {};

						if (oldVersion) {
							torch::Tensor tdNewStates = tStates.index_select(0, tNewPlayerIndices).to(ppo->device, true);
							torch::Tensor tdOldStates = tStates.index_select(0, tOldPlayerIndices).to(ppo->device, true);
							torch::Tensor tdNewActionMasks = tActionMasks.index_select(0, tNewPlayerIndices).to(ppo->device, true);
							torch::Tensor tdOldActionMasks = tActionMasks.index_select(0, tOldPlayerIndices).to(ppo->device, true);

							torch::Tensor tNewActions;
							torch::Tensor tOldActions;

							ppo->InferActions(tdNewStates, tdNewActionMasks, &tNewActions, &tLogProbs);
							ppo->InferActions(tdOldStates, tdOldActionMasks, &tOldActions, NULL, &oldVersion->models);

							tActions = torch::zeros(numPlayers, tNewActions.dtype());
							tActions.index_copy_(0, tNewPlayerIndices, tNewActions.cpu());
							tActions.index_copy_(0, tOldPlayerIndices, tOldActions.cpu());
						} else {
							torch::Tensor tdStates = tStates.to(ppo->device, true);
							torch::Tensor tdActionMasks = tActionMasks.to(ppo->device, true);
							ppo->InferActions(tdStates, tdActionMasks, &tActions, &tLogProbs);
							tActions = tActions.cpu();
						}
						inferTime += inferTimer.Elapsed();

						auto curActions = TENSOR_TO_VEC<int>(tActions);
						FList newLogProbs;
						if (tLogProbs.defined() && !render)
							newLogProbs = TENSOR_TO_VEC<float>(tLogProbs);	

						stepTimer.Reset();
						envSet->Sync(); // Make sure the first half is done

						envSet->StepSecondHalf(curActions, false);
						envStepTime += stepTimer.Elapsed();

						if (stepCallback)
							stepCallback(this, envSet->state.gameStates, report);

						if (render) {
							renderSender->Send(envSet->state.gameStates[0]);
							continue;
						}

						// Calc average rewards
						if (config.addRewardsToMetrics && (Math::RandInt(0, config.rewardSampleRandInterval) == 0)) {
							int numSamples = RS_MIN(envSet->arenas.size(), config.maxRewardSamples);
							std::unordered_map<std::string, AvgTracker> avgRewards = {};
							for (int i = 0; i < numSamples; i++) {
								int arenaIdx = Math::RandInt(0, envSet->arenas.size());
								auto& prevRewards = envSet->state.lastRewards[arenaIdx];

								for (int j = 0; j < envSet->rewards[arenaIdx].size(); j++) {
									std::string rewardName = envSet->rewards[arenaIdx][j].reward->GetName();
									avgRewards[rewardName] += prevRewards[j];
								}
							}

							for (auto& pair : avgRewards)
								report.AddAvg("Rewards/" + pair.first, pair.second.Get());
						}

						// Now that we've inferred and stepped the env, we can add that stuff to the trajectories
						int i = 0;
						for (int newPlayerIdx : newPlayerIndices) {
							trajectories[newPlayerIdx].actions.push_back(curActions[newPlayerIdx]);
							trajectories[newPlayerIdx].rewards += envSet->state.rewards[newPlayerIdx];
							trajectories[newPlayerIdx].logProbs += newLogProbs[i];

							if (config.ppo.contrastiveGoal.enabled) {
								int arenaIdx = GetArenaIdxForPlayer(*envSet, newPlayerIdx);
								int localPlayerIdx = newPlayerIdx - envSet->state.arenaPlayerStartIdx[arenaIdx];
								const GameState& state = envSet->state.gameStates[arenaIdx];
								const Player& player = state.players[localPlayerIdx];
								AppendAchievedGoal(trajectories[newPlayerIdx].achievedGoals, state, player, config.ppo.contrastiveGoal);
							}
							i++;
						}

						auto curTerminals = std::vector<uint8_t>(numPlayers, 0);
						for (int idx = 0; idx < envSet->arenas.size(); idx++) {
							uint8_t terminalType = envSet->state.terminals[idx];
							if (!terminalType)
								continue;

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
									// Truncation requires an additional next state for the critic.
									// StepSecondHalf rebuilt obs as RAW values, so apply the same
									// standardization the stored states got this step before bootstrapping.
									FList nextStateRow = envSet->state.obs.GetRow(newPlayerIdx);
									standardizeObsRow(nextStateRow.data());
									traj.nextStates += nextStateRow;
								}

								relabelHERGoals(traj);
								combinedTraj.Append(traj);
								traj.Clear();
								curSegmentIds[newPlayerIdx] = nextSegmentId++;
								curSegmentSteps[newPlayerIdx] = 0;
							} else {
								curSegmentSteps[newPlayerIdx]++;
							}
						}
					}

					report["Inference Time"] = inferTime;
					report["Env Step Time"] = envStepTime;
				}
				float collectionTime = collectionTimer.Elapsed();

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
					torch::Tensor tAchievedGoals, tHERGoals, tScoringGoals, tGcrlTrainMask, tSegmentIds, tSegmentSteps;
					if (config.ppo.contrastiveGoal.enabled) {
						tAchievedGoals = torch::tensor(combinedTraj.achievedGoals).reshape({ -1, 6 });
						tHERGoals = torch::tensor(combinedTraj.herGoals).reshape({ -1, 6 });
						tScoringGoals = torch::tensor(combinedTraj.scoringGoals).reshape({ -1, 6 });
						tGcrlTrainMask = torch::tensor(combinedTraj.gcrlTrainMask, torch::TensorOptions().dtype(torch::kUInt8));
						tSegmentIds = torch::tensor(combinedTraj.segmentIds, torch::TensorOptions().dtype(torch::kInt64));
						tSegmentSteps = torch::tensor(combinedTraj.segmentSteps, torch::TensorOptions().dtype(torch::kInt64));

						int64_t expRows = tStates.size(0);
						if (tAchievedGoals.size(0) != expRows || tHERGoals.size(0) != expRows || tScoringGoals.size(0) != expRows || tGcrlTrainMask.size(0) != expRows || tSegmentIds.size(0) != expRows)
							RG_ERR_CLOSE("GCRL tensor alignment failed: states=" << expRows <<
								", achievedGoals=" << tAchievedGoals.size(0) <<
								", herGoals=" << tHERGoals.size(0) <<
								", scoringGoals=" << tScoringGoals.size(0) <<
								", gcrlTrainMask=" << tGcrlTrainMask.size(0) <<
								", segmentIds=" << tSegmentIds.size(0));

						if (!herSelectedOffsets.empty()) {
							torch::Tensor tOffsets = torch::tensor(herSelectedOffsets);
							report["HER Selected Offset Mean"] = tOffsets.mean().item<float>();
							report["HER Selected Offset P10"] = tOffsets.quantile(0.1).item<float>();
							report["HER Selected Offset P50"] = tOffsets.quantile(0.5).item<float>();
							report["HER Selected Offset P90"] = tOffsets.quantile(0.9).item<float>();
							report["HER Valid Relabel Count"] = (float)herSelectedOffsets.size();
						}
						report["HER Total Relabel Rows"] = (float)herTotalRows;
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
						tValPreds = ppo->InferCritic(tStates.to(ppo->device, true, true)).cpu();
						if (tNextTruncStates.defined())
							tTruncValPreds = ppo->InferCritic(tNextTruncStates.to(ppo->device, true, true)).cpu();
					} else {
						// Predict values using minibatching
						tValPreds = torch::zeros({ (int64_t)combinedTraj.Length() });
						for (int i = 0; i < combinedTraj.Length(); i += ppo->config.miniBatchSize) {
							int start = i;
							int end = RS_MIN(i + ppo->config.miniBatchSize, combinedTraj.Length());
							torch::Tensor tStatesPart = tStates.slice(0, start, end);

							auto valPredsPart = ppo->InferCritic(tStatesPart.to(ppo->device, true, true)).cpu();
							RG_ASSERT(valPredsPart.size(0) == (end - start));
							tValPreds.slice(0, start, end).copy_(valPredsPart, true);
						}

						if (tNextTruncStates.defined()) {
							// This really just should never happen
							// If this is ever actually a real problem in a legitimate use case, ping Zealan in the dead of night
							RG_ASSERT(tNextTruncStates.size(0) <= ppo->config.miniBatchSize);

							tTruncValPreds = ppo->InferCritic(tNextTruncStates.to(ppo->device, true, true)).cpu();
						}
					}

					float normalTerminalRate = (tTerminals == RLGC::TerminalType::NORMAL).to(torch::kFloat32).mean().item<float>();
					report["Episode Length"] = normalTerminalRate > 0 ? 1.f / normalTerminalRate : 0;

					Timer gaeTimer = {};
					// Run GAE
					torch::Tensor tAdvantages, tTargetVals, tReturns;
					float rewClipPortion;
					GAE::Compute(
						tRewards, tTerminals, tValPreds, tTruncValPreds,
						tAdvantages, tTargetVals, tReturns, rewClipPortion,
						config.ppo.gaeGamma, config.ppo.gaeLambda, returnStat ? returnStat->GetSTD() : 1, config.ppo.rewardClipRange
					);
					report["GAE Time"] = gaeTimer.Elapsed();
					report["Clipped Reward Portion"] = rewClipPortion;

					if (returnStat) {
						report["GAE/Returns STD"] = returnStat->GetSTD();

						int numToIncrement = RS_MIN(config.maxReturnSamples, tReturns.size(0));
						if (numToIncrement > 0) {
							auto selectedReturns = tReturns.index_select(0, torch::randint(tReturns.size(0), { (int64_t)numToIncrement }));
							returnStat->Increment(TENSOR_TO_VEC<float>(selectedReturns));
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
					if (config.ppo.contrastiveGoal.enabled) {
						experience.data.achievedGoals = tAchievedGoals;
						experience.data.herGoals = tHERGoals;
						experience.data.scoringGoals = tScoringGoals;
						experience.data.gcrlTrainMask = tGcrlTrainMask;
						experience.data.segmentIds = tSegmentIds;
						experience.data.segmentSteps = tSegmentSteps;
					}
				}

				// Free CUDA cache
#ifdef RG_CUDA_SUPPORT
				if (ppo->device.is_cuda())
					c10::cuda::CUDACachingAllocator::emptyCache();
#endif

				// Learn
				Timer learnTimer = {};
				ppo->Learn(experience, report, isFirstIteration, totalTimesteps);
				report["PPO Learn Time"] = learnTimer.Elapsed();

				// Set metrics
				float consumptionTime = consumptionTimer.Elapsed();
				report["Collection Time"] = collectionTime;
				report["Consumption Time"] = consumptionTime;

				auto calcStepsPerSecond = [](int steps, float seconds) {
					return seconds > 0 ? steps / seconds : 0.f;
				};
				report["Collection Steps/Second"] = calcStepsPerSecond(stepsCollected, collectionTime);
				report["Consumption Steps/Second"] = calcStepsPerSecond(stepsCollected, consumptionTime);
				report["Overall Steps/Second"] = calcStepsPerSecond(stepsCollected, collectionTime + consumptionTime);

				uint64_t prevTimesteps = totalTimesteps;
				totalTimesteps += stepsCollected;
				report["Total Timesteps"] = totalTimesteps;
				totalIterations++;
				report["Total Iterations"] = totalIterations;

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
						"Average Step Reward",
						"Policy Entropy",
						"KL Div Loss",
						"First Accuracy",
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
						"Collected Timesteps",
						"Total Timesteps",
						"Total Iterations"
					}
				);
			}
		}
		
	} catch (std::exception& e) {
		RG_ERR_CLOSE("Exception thrown during main learner loop: " << e.what());
	}
}

GGL::Learner::~Learner() {
	delete ppo;
	delete versionMgr;
	delete metricSender;
	delete renderSender;
	pybind11::finalize_interpreter();
}
