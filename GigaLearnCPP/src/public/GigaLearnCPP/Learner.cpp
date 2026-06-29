#include "Learner.h"

#include <GigaLearnCPP/PPO/PPOLearner.h>
#include <GigaLearnCPP/PPO/ExperienceBuffer.h>

#include <torch/cuda.h>
#include <ATen/Context.h>
#if defined(__APPLE__) && __has_include(<torch/mps.h>)
#include <torch/mps.h>
#define RG_MPS_SUPPORT
#endif
#include <nlohmann/json.hpp>
#include <pybind11/embed.h>
#include <cmath>
#include <future>
#include <private/GigaLearnCPP/Util/RSNorm.h>

#ifdef RG_CUDA_SUPPORT
#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)
#include <c10/hip/HIPCachingAllocator.h>
#else
#include <c10/cuda/CUDACachingAllocator.h>
#endif
#endif
#include <private/GigaLearnCPP/PPO/ExperienceBuffer.h>
#include <private/GigaLearnCPP/PPO/GAE.h>
#include <private/GigaLearnCPP/PPO/TrainingBatch.h>
#include <private/GigaLearnCPP/PolicyVersionManager.h>

#include "Util/KeyPressDetector.h"
#include "Util/MetricSender.h"
#include "Util/RenderSender.h"
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

// Ball-AGNOSTIC car-CONTROL goal: the car's own kinematic + mechanic state = 15 floats --
// velocity(3) + forward(3) + up(3) + angVel(3) + air-control flags(3) -- team-canonicalized so +y is always
// the attacking direction. This makes the CAR critic an empowerment/motor-skill signal (speedflips,
// wavedashes, aerial orientation, recoveries): it learns which actions reach diverse, controllable SELF
// states, with zero reason to drive at the ball. forward+up fully encode attitude (yaw/pitch/roll) and angVel
// the rotation RATES, so no Euler angles are needed; the air-control flags add the jump/flip mechanic state
// that kinematics can't express. Deliberately EXCLUDES world position (positioning is the goal critic's job;
// including it would re-add a location magnet AND bind the skill to a spot instead of generalizing the
// motion) and boost (a boost-in-goal term invites hoarding). Layout MUST match carGoalInputSize.
static void AppendCarStateGoal(FList& out, const Player& player, const GGL::ContrastiveGoalConfig& cfg) {
	PhysState car = InvertPhys(player, player.team == Team::ORANGE);
	out += car.vel.x / CommonValues::CAR_MAX_SPEED;
	out += car.vel.y / CommonValues::CAR_MAX_SPEED;
	out += car.vel.z / CommonValues::CAR_MAX_SPEED;
	out += car.rotMat.forward.x;
	out += car.rotMat.forward.y;
	out += car.rotMat.forward.z;
	out += car.rotMat.up.x;
	out += car.rotMat.up.y;
	out += car.rotMat.up.z;
	out += car.angVel.x / CommonValues::CAR_MAX_ANG_VEL;
	out += car.angVel.y / CommonValues::CAR_MAX_ANG_VEL;
	out += car.angVel.z / CommonValues::CAR_MAX_ANG_VEL;
	// Air-control / mechanic state -- frame-invariant booleans, read from the un-canonicalized player.
	// These are the jump/flip resources that govern aerials, wavedashes and flip resets (not derivable
	// from pos/vel/orientation), so the critic can learn to reach "airborne with my flip still available".
	out += player.isOnGround ? 1.f : 0.f;
	out += player.HasFlipOrJump() ? 1.f : 0.f; // a flip / double-jump is still available
	out += player.isFlipping ? 1.f : 0.f;      // currently mid-dodge/flip
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

GGL::Learner::Learner(EnvCreateFn envCreateFn, LearnerConfig config, StepCallbackFn stepCallback,
	MetricSink* injectedMetricSink, RenderSink* injectedRenderSink) :
	envCreateFn(envCreateFn), config(config), stepCallback(stepCallback)
{
	// No interpreter init here — the pybind adapters (MetricSender/RenderSender) own the Python
	// runtime via a refcounted guard, so a Learner with only injected in-memory sinks needs no Python.

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

	// Apply the TF32 matmul policy on CUDA. libtorch defaults cuBLAS-matmul TF32 to OFF, so without this the
	// dense MLP forward/backward run as strict fp32 and never touch the Ampere+/Blackwell TF32 tensor-core
	// path -- config.allowTF32 (default true) is otherwise inert. Set it false for strict-fp32 reproducibility.
	// cuDNN TF32 is already on by default and irrelevant to this conv-free MLP; we set it for symmetry. The
	// setters exist on every build but only bite on CUDA, so this is gated to a CUDA device.
	if (device.is_cuda()) {
		at::globalContext().setAllowTF32CuBLAS(config.allowTF32);
		at::globalContext().setAllowTF32CuDNN(config.allowTF32);
		RG_LOG("\tTF32 matmuls (CUDA tensor cores): " << (config.allowTF32 ? "enabled" : "disabled"));
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
		if (config.standardizeReturns) {
			this->returnStat = new WelfordStat();
		} else {
			this->returnStat = NULL;
		}

	}

	try {
		RG_LOG("\tMaking PPO learner...");
		ppo = new PPOLearner(obsSize, numActions, config.ppo, device);
	} catch (std::exception& e) {
		RG_ERR_CLOSE("Failed to create PPO learner: " << e.what());
	}

	if (injectedRenderSink) {
		renderSink = injectedRenderSink;
		ownsRenderSink = false;
	} else if (config.renderMode) {
		renderSink = new RenderSender(config.renderTimeScale);
		ownsRenderSink = true;
	} else {
		renderSink = NULL;
		ownsRenderSink = false;
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
		versionMgr->maxAnchors = config.maxAnchorVersions;
		versionMgr->anchorSelectChance = config.anchorSelectChance;
		versionMgr->anchorPromoteMargin = config.anchorPromoteMargin;
		versionMgr->anchorMinTsSpacing = config.anchorMinTsSpacing;
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

	if (injectedMetricSink) {
		metricSink = injectedMetricSink;
		ownsMetricSink = false;
	} else if (config.sendMetrics && !config.renderMode) {
		if (!runID.empty())
			RG_LOG("\tRun ID: " << runID);
		metricSink = new MetricSender(config.metricsProjectName, config.metricsGroupName, config.metricsRunName, runID);
		ownsMetricSink = true;
	} else {
		metricSink = NULL;
		ownsMetricSink = false;
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

	if (config.sendMetrics && metricSink)
		j["run_id"] = metricSink->GetRunID();

	if (returnStat)
		j["return_stat"] = returnStat->ToJSON();
	// RSNorm stats travel WITH the weights (part of the policy state).
	if (ppo && ppo->obsNorm)
		j["rsnorm_stat"] = ppo->obsNorm->ToJSON();

	// Adaptive entropy controller state (so a crash-resume doesn't re-ramp from
	// the fixed seed and dip entropy each restart).
	if (ppo) {
		j["entropy_scale"] = ppo->curEntropyScale;
		// TRIAD-NATIVE GCRL coupling controller state (so resume doesn't re-seed the ratio
		// controller / RenormToStd EMA and re-ramp the GCRL blend each restart).
		j["gcrl_lambda_eff"] = ppo->gcrlLambdaEff;
		j["gcrl_ratio_ema"] = ppo->gcrlRatioEma;
		j["gcrl_renorm_std"] = ppo->gcrlRenormStd;
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

	if (j.contains("run_id"))
		runID = j["run_id"];

	if (returnStat)
		returnStat->ReadFromJSON(j["return_stat"]);
	// contains-guard so checkpoints predating RSNorm still load (stats stay at init).
	if (ppo && ppo->obsNorm && j.contains("rsnorm_stat"))
		ppo->obsNorm->ReadFromJSON(j["rsnorm_stat"]);

	// Restore the adaptive entropy scale, re-clamped against current config (a
	// checkpoint predating it keeps the ctor seed). contains-guarded.
	if (ppo && j.contains("entropy_scale")) {
		ppo->curEntropyScale = j["entropy_scale"];
		if (ppo->config.adaptiveEntropy)
			ppo->curEntropyScale = RS_CLAMP(ppo->curEntropyScale, ppo->config.minEntropyScale, ppo->config.maxEntropyScale);
	}
	// TRIAD-NATIVE GCRL coupling controller state (contains-guarded; pre-TRIAD checkpoints keep ctor seeds).
	if (ppo) {
		if (j.contains("gcrl_lambda_eff")) ppo->gcrlLambdaEff = j["gcrl_lambda_eff"];
		if (j.contains("gcrl_ratio_ema")) ppo->gcrlRatioEma = j["gcrl_ratio_ema"];
		if (j.contains("gcrl_renorm_std")) ppo->gcrlRenormStd = j["gcrl_renorm_std"];
	}

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

			if (metricSink)
				metricSink->Send(report);

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

		// Double-buffered rollout production. The producer (collection + value-pred + GAE) fills one slot
		// using a FROZEN actor clone + RSNorm snapshot; the consumer (PPO update) reads the other. With
		// overlapCollection on, the producer runs on a background thread while the main thread runs Learn.
		ExperienceBuffer expBuf[2] = {
			ExperienceBuffer(config.randomSeed, torch::kCPU),
			ExperienceBuffer(config.randomSeed, torch::kCPU)
		};
		Report repBuf[2];
		int stepsBuf[2] = { 0, 0 };

		const bool overlap = config.overlapCollection && !config.renderMode;
		int curBuf = 0;
		bool overlapPrimed = false;

		// Frozen actor copy the producer reads while Learn mutates the live ppo->models. CloneAll deep-copies
		// shared_head/policy/critic; CopyParamsFrom refreshes it each iteration. RSNorm snapshot likewise.
		ModelSet actorClone = ppo->models.CloneAll();
		RSNorm rsnormSnap(obsSize);
		const RSNorm* snapPtr = ppo->obsNorm ? &rsnormSnap : nullptr;
		auto fnRefreshActor = [&]() {
			actorClone.CopyParamsFrom(ppo->models);
			if (ppo->obsNorm)
				rsnormSnap.CopyStatsFrom(*ppo->obsNorm);
#ifdef RG_CUDA_SUPPORT
			if (ppo->device.is_cuda()) torch::cuda::synchronize();
#endif
#ifdef RG_MPS_SUPPORT
			if (ppo->device.is_mps()) torch::mps::synchronize();
#endif
		};

		int numPlayers = envSet->state.numPlayers;

		struct Trajectory {
			FList states, nextStates, rewards, logProbs;
			FList achievedGoals, herGoals, carHerGoals, scoringGoals;
			FList carStates; // per-step car self-state (12 floats); the CAR critic's HER goal source
			std::vector<uint8_t> actionMasks;
			std::vector<uint8_t> gcrlTrainMask; // (2B) per-row: 1 = use this row to train the contrastive critic
			std::vector<uint8_t> gcrlScoringMask; // FORK2 Part C: 1 = this row's herGoal is a SYNTHETIC scoring goal (excluded from TD bootstrap)
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
				carHerGoals += other.carHerGoals;
				scoringGoals += other.scoringGoals;
				carStates += other.carStates;
				gcrlTrainMask += other.gcrlTrainMask;
				gcrlScoringMask += other.gcrlScoringMask;
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
			Timer iterTimer = {};

			// Producer: collect a rollout, value-predict, and run GAE into buffer slot outIdx using the FROZEN
			// actor clone + RSNorm snapshot (so the live models/obsNorm can be updated by Learn concurrently).
			// Runs on a background thread when overlapping; otherwise called inline.
			auto fnProduce = [&](int outIdx) {
			Report& report = repBuf[outIdx];
			ExperienceBuffer& experience = expBuf[outIdx];
			int& stepsCollected = stepsBuf[outIdx];
			report = Report{};
			stepsCollected = 0;

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

					// Anchor-aware draw: with probability anchorSelectChance this returns a "gold" anchor
					// (a strong, spaced past self) rather than a near-duplicate rolling-recent snapshot.
					oldVersion = versionMgr->PickOldVersion();

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

			{ // Generate experience

				// Only contains complete episodes
				auto combinedTraj = Trajectory();
				FList herSelectedOffsets;
				int herTotalRows = 0;

				// The CAR critic is now ball-agnostic (its goal is the car's own future kinematic state,
				// collected per-step into traj.carStates via AppendCarStateGoal), so it no longer needs a
				// car-local ball offset from the obs builder.

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
					float scoringGoalMixFrac = RS_CLAMP(config.ppo.contrastiveGoal.scoringGoalMixFrac, 0.f, 1.f);

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
					traj.gcrlScoringMask.assign((size_t)n, 0); // FORK2 Part C

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
						// FORK2 Part C: with prob scoringGoalMixFrac, draw a SYNTHETIC net-directed
						// scoring goal (scoringGoals[t]) instead of the achieved future. Flagged so the
						// TD bootstrap excludes it (V^-(s',g_synthetic) would be an OOD extrapolation).
						// CAR (carHerGoals) is NEVER mixed. Synthetic rows don't add offset stats.
						bool useScoring = scoringGoalMixFrac > 0.f
							&& (size_t)(t + 1) * 6 <= traj.scoringGoals.size()
							&& RocketSim::Math::RandFloat() < scoringGoalMixFrac;
						if (useScoring) {
							for (int d = 0; d < 6; d++)
								traj.herGoals[(size_t)t * 6 + d] = traj.scoringGoals[(size_t)t * 6 + d];
							traj.gcrlScoringMask[(size_t)t] = 1;
						} else {
							for (int d = 0; d < 6; d++)
								traj.herGoals[(size_t)t * 6 + d] = traj.achievedGoals[(size_t)target * 6 + d];
							if (selectedOffset > 0)
								herSelectedOffsets += (float)selectedOffset;
						}
					}

					// Car-CONTROL critic: its OWN short, near-term HER window over the car's own future
					// kinematic SELF-state (carStates: velocity + orientation + angular velocity), sampled
					// short because car control plays out on a ~1-3s horizon. Ball-agnostic, so -- unlike the
					// ball goal -- there is no goalward bias and no synthetic scoring-goal mix; the target is
					// always a genuine achieved future self-state (a valid InfoNCE positive).
					const int carGoalSize = config.ppo.contrastiveGoal.carGoalInputSize;
					if (config.ppo.contrastiveGoal.useCarCritic && carGoalSize > 0
						&& (int)traj.carStates.size() == n * carGoalSize) {
						int carMin = RS_MAX(1, config.ppo.contrastiveGoal.carHerMinOffset);
						int carMaxCfg = RS_MAX(carMin, config.ppo.contrastiveGoal.carHerMaxOffset);
						float carShortBiasPower = RS_MAX(1e-3f, config.ppo.contrastiveGoal.carHerShortBiasPower);
						traj.carHerGoals.resize((size_t)n * carGoalSize);
						for (int t = 0; t < n; t++) {
							int remaining = n - t - 1;
							int carOffset = 0;
							if (remaining > 0) {
								int carMax = RS_MIN(carMaxCfg, remaining);
								if (carMax >= carMin) {
									float u = RocketSim::Math::RandFloat();
									float biased = powf(u, carShortBiasPower);
									int span = carMax - carMin + 1;
									carOffset = carMin + RS_MIN(span - 1, (int)floorf(biased * span));
								} else {
									carOffset = remaining;
								}
							}
							int carTarget = t + carOffset;
							for (int d = 0; d < carGoalSize; d++)
								traj.carHerGoals[(size_t)t * carGoalSize + d] = traj.carStates[(size_t)carTarget * carGoalSize + d];
						}
					}
				};

				Timer collectionTimer = {};
				{ // Collect timesteps
					RG_NO_GRAD;

					float inferTime = 0;
					float envStepTime = 0;
					float prepTime = 0;
					float recordTime = 0;

					// Chunked parallel-for over [0,n): ~4 chunks per worker on the global pool, blocking until done.
					// Parallelizes the per-player prep/record loops. INVARIANT: g_ThreadPool must be quiescent when
					// called -- wait_for_tasks() is pool-global, not per-batch -- so call only from the (single)
					// producer thread, sequentially, never nested in a pool job and never while async arena jobs run.
					auto fnParallelFor = [](int n, const std::function<void(int)>& body) {
						if (n <= 0) return;
						int nChunks = RS_MIN(n, RS_MAX(1, RLGC::g_ThreadPool.GetNumThreads() * 4));
						int chunkSize = (n + nChunks - 1) / nChunks;
						RLGC::g_ThreadPool.StartBatchedJobs([&body, n, chunkSize](int c) {
							int start = c * chunkSize;
							int end = RS_MIN(start + chunkSize, n);
							for (int k = start; k < end; k++)
								body(k);
						}, nChunks, false);
					};
					// Reused per-step scratch: player->arena map (replaces the per-player O(numArenas) scan) and
					// terminalType-per-player (parallel terminal pass -> serial finalize).
					std::vector<int> playerArena(numPlayers, 0);
					std::vector<int8_t> finalTerminals(numPlayers, 0);

					for (int step = 0; combinedTraj.Length() < config.ppo.tsPerItr || render; step++, stepsCollected += numRealPlayers) {
						Timer stepTimer = {};
						envSet->Reset();
						envStepTime += stepTimer.Elapsed();

						// Serial main-thread prep: obs NaN-check + obs->tensor build + pre-step trajectory setup.
						Timer prepTimer = {};
						for (float f : envSet->state.obs.data)
							if (isnan(f) || isinf(f))
								RG_ERR_CLOSE("Obs builder produced a NaN/inf value");

						// Observations are stored RAW; normalization (if enabled) is RSNorm,
						// applied inside the actor/critic forward (PPOLearner), not here.

						torch::Tensor tActions, tLogProbs;
						torch::Tensor tStates = DIMLIST2_TO_TENSOR<float>(envSet->state.obs);
						torch::Tensor tActionMasks = DIMLIST2_TO_TENSOR<uint8_t>(envSet->state.actionMasks);

						// Player->arena lookup for this step: replaces the per-player O(numArenas) linear scan
						// (GetArenaIdxForPlayer) with an O(1) lookup. Rebuilt each step since resets can change layout.
						{
							int nA = (int)envSet->state.arenaPlayerStartIdx.size();
							for (int a = 0; a < nA; a++) {
								int start = envSet->state.arenaPlayerStartIdx[a];
								int stop = (a + 1 < nA) ? (int)envSet->state.arenaPlayerStartIdx[a + 1] : numPlayers;
								for (int p = start; p < stop; p++)
									playerArena[p] = a;
							}
						}

						if (!render) {
							// Parallel per-player: each task writes only its own trajectories[newPlayerIdx], race-free.
							fnParallelFor((int)newPlayerIndices.size(), [&](int k) {
								int newPlayerIdx = newPlayerIndices[k];
								trajectories[newPlayerIdx].states += envSet->state.obs.GetRow(newPlayerIdx);
								trajectories[newPlayerIdx].actionMasks += envSet->state.actionMasks.GetRow(newPlayerIdx);

								if (config.ppo.contrastiveGoal.enabled) {
									int arenaIdx = playerArena[newPlayerIdx];
									int localPlayerIdx = newPlayerIdx - envSet->state.arenaPlayerStartIdx[arenaIdx];
									const GameState& state = envSet->state.gameStates[arenaIdx];
									const Player& player = state.players[localPlayerIdx];

									AppendScoringGoal(trajectories[newPlayerIdx].scoringGoals, state, player, config.ppo.contrastiveGoal);
									trajectories[newPlayerIdx].segmentIds.push_back(curSegmentIds[newPlayerIdx]);
									trajectories[newPlayerIdx].segmentSteps.push_back(curSegmentSteps[newPlayerIdx]);
								}
							});
						}

						prepTime += prepTimer.Elapsed();
						envSet->StepFirstHalf(true);

						Timer inferTimer = {};

						if (oldVersion) {
							torch::Tensor tdNewStates = tStates.index_select(0, tNewPlayerIndices).to(ppo->device, true);
							torch::Tensor tdOldStates = tStates.index_select(0, tOldPlayerIndices).to(ppo->device, true);
							torch::Tensor tdNewActionMasks = tActionMasks.index_select(0, tNewPlayerIndices).to(ppo->device, true);
							torch::Tensor tdOldActionMasks = tActionMasks.index_select(0, tOldPlayerIndices).to(ppo->device, true);

							torch::Tensor tNewActions;
							torch::Tensor tOldActions;

							PPOLearner::InferActionsFromModels(actorClone, tdNewStates, tdNewActionMasks, ppo->config.deterministic, ppo->config.policyTemperature, ppo->config.useHalfPrecision, &tNewActions, &tLogProbs, nullptr, snapPtr);
							PPOLearner::InferActionsFromModels(oldVersion->models, tdOldStates, tdOldActionMasks, ppo->config.deterministic, ppo->config.policyTemperature, ppo->config.useHalfPrecision, &tOldActions, nullptr, nullptr, snapPtr);

							tActions = torch::zeros(numPlayers, tNewActions.dtype());
							tActions.index_copy_(0, tNewPlayerIndices, tNewActions.cpu());
							tActions.index_copy_(0, tOldPlayerIndices, tOldActions.cpu());
						} else {
							torch::Tensor tdStates = tStates.to(ppo->device, true);
							torch::Tensor tdActionMasks = tActionMasks.to(ppo->device, true);
							PPOLearner::InferActionsFromModels(actorClone, tdStates, tdActionMasks, ppo->config.deterministic, ppo->config.policyTemperature, ppo->config.useHalfPrecision, &tActions, &tLogProbs, nullptr, snapPtr);
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

						// Serial main-thread record: reward sampling + per-player trajectory append + terminals/HER.
						Timer recordTimer = {};
						if (stepCallback)
							stepCallback(this, envSet->state.gameStates, report);

						if (render) {
							renderSink->Send(envSet->state.gameStates[0]);
							continue;
						}

						// Calc average rewards
						if (config.addRewardsToMetrics && (Math::RandInt(0, config.rewardSampleRandInterval) == 0)) {
							int numSamples = RS_MIN(envSet->arenas.size(), config.maxRewardSamples);
							std::unordered_map<std::string, AvgTracker> avgRewards = {};
							for (int i = 0; i < numSamples; i++) {
								int arenaIdx = Math::RandInt(0, envSet->arenas.size());
								auto& prevRewards = envSet->state.lastRewards[arenaIdx];

								const auto& rewardNames = envSet->GetRewardNames(arenaIdx);
								int numRewards = RS_MIN((int)rewardNames.size(), (int)prevRewards.size());
								for (int j = 0; j < numRewards; j++)
									avgRewards[rewardNames[j]] += prevRewards[j];
							}

							for (auto& pair : avgRewards)
								report.AddAvg("Rewards/" + pair.first, pair.second.Get());
						}

						// Now that we've inferred and stepped the env, add that stuff to the trajectories.
						// Parallel per-player: writes only trajectories[newPlayerIdx]; logProbs indexed by ordinal k.
						fnParallelFor((int)newPlayerIndices.size(), [&](int k) {
							int newPlayerIdx = newPlayerIndices[k];
							trajectories[newPlayerIdx].actions.push_back(curActions[newPlayerIdx]);
							trajectories[newPlayerIdx].rewards += envSet->state.rewards[newPlayerIdx];
							trajectories[newPlayerIdx].logProbs += newLogProbs[k];

							if (config.ppo.contrastiveGoal.enabled) {
								int arenaIdx = playerArena[newPlayerIdx];
								int localPlayerIdx = newPlayerIdx - envSet->state.arenaPlayerStartIdx[arenaIdx];
								const GameState& state = envSet->state.gameStates[arenaIdx];
								const Player& player = state.players[localPlayerIdx];
								AppendAchievedGoal(trajectories[newPlayerIdx].achievedGoals, state, player, config.ppo.contrastiveGoal);
								if (config.ppo.contrastiveGoal.useCarCritic)
									AppendCarStateGoal(trajectories[newPlayerIdx].carStates, player, config.ppo.contrastiveGoal);
							}
						});

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

						// Parallel per-player: terminal type, append it, truncation next-state, advance live-episode steps.
						// Writes only per-player-distinct slots (traj[idx], curSegmentSteps[idx], finalTerminals[idx]).
						fnParallelFor((int)newPlayerIndices.size(), [&](int k) {
							int newPlayerIdx = newPlayerIndices[k];
							int8_t terminalType = curTerminals[newPlayerIdx];
							auto& traj = trajectories[newPlayerIdx];

							if (!terminalType && traj.Length() >= maxEpisodeLength) {
								// Episode too long: truncate here (added to buffer as truncated, no env reset).
								terminalType = RLGC::TerminalType::TRUNCATED;
							}

							traj.terminals.push_back(terminalType);
							if (terminalType == RLGC::TerminalType::TRUNCATED) {
								// Truncation requires an additional next state for the critic (stored RAW).
								traj.nextStates += envSet->state.obs.GetRow(newPlayerIdx);
							}
							if (!terminalType)
								curSegmentSteps[newPlayerIdx]++;

							finalTerminals[newPlayerIdx] = terminalType;
						});

						// Serial finalize (deterministic newPlayerIndices order == original row order): HER relabel + append
						// completed episodes into the shared combinedTraj + bump the shared segment counter. Kept serial
						// because relabelHERGoals writes shared HER metrics and combinedTraj.Append/nextSegmentId++ are
						// shared mutations -- as serial as the original, while the per-player bulk above is now parallel.
						for (int newPlayerIdx : newPlayerIndices) {
							if (!finalTerminals[newPlayerIdx])
								continue;
							auto& traj = trajectories[newPlayerIdx];
							relabelHERGoals(traj);
							combinedTraj.Append(traj);
							traj.Clear();
							curSegmentIds[newPlayerIdx] = nextSegmentId++;
							curSegmentSteps[newPlayerIdx] = 0;
						}

						recordTime += recordTimer.Elapsed();
					}

					report["Collection/Prep Time"] = prepTime;
					report["Collection/Inference Time"] = inferTime;
					report["Collection/Env Step Time"] = envStepTime;
					report["Collection/Record Time"] = recordTime;
				}
				float collectionTime = collectionTimer.Elapsed();
				report["Collection Time"] = collectionTime;

				{ // Process timesteps
					RG_NO_GRAD;

					// Make and transpose tensors
					torch::Tensor tStates = torch::tensor(combinedTraj.states).reshape({ -1, obsSize });
					torch::Tensor tActionMasks = torch::tensor(combinedTraj.actionMasks).reshape({ -1, numActions });
					torch::Tensor tActions = torch::tensor(combinedTraj.actions);
					torch::Tensor tLogProbs = torch::tensor(combinedTraj.logProbs);
					torch::Tensor tRewards = torch::tensor(combinedTraj.rewards);
					torch::Tensor tTerminals = torch::tensor(combinedTraj.terminals);
					torch::Tensor tAchievedGoals, tHERGoals, tCarHERGoals, tScoringGoals, tGcrlTrainMask, tGcrlScoringMask, tSegmentIds, tSegmentSteps;
					if (config.ppo.contrastiveGoal.enabled) {
						tAchievedGoals = torch::tensor(combinedTraj.achievedGoals).reshape({ -1, 6 });
						tHERGoals = torch::tensor(combinedTraj.herGoals).reshape({ -1, 6 });
						if (config.ppo.contrastiveGoal.useCarCritic && !combinedTraj.carHerGoals.empty())
							tCarHERGoals = torch::tensor(combinedTraj.carHerGoals).reshape({ -1, config.ppo.contrastiveGoal.carGoalInputSize });
						tScoringGoals = torch::tensor(combinedTraj.scoringGoals).reshape({ -1, 6 });
						tGcrlTrainMask = torch::tensor(combinedTraj.gcrlTrainMask, torch::TensorOptions().dtype(torch::kUInt8));
						tGcrlScoringMask = torch::tensor(combinedTraj.gcrlScoringMask, torch::TensorOptions().dtype(torch::kUInt8));
						tSegmentIds = torch::tensor(combinedTraj.segmentIds, torch::TensorOptions().dtype(torch::kInt64));
						tSegmentSteps = torch::tensor(combinedTraj.segmentSteps, torch::TensorOptions().dtype(torch::kInt64));

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
						tValPreds = PPOLearner::InferCriticFromModels(actorClone, tStates.to(ppo->device, true, true), ppo->config.useHalfPrecision, snapPtr).cpu();
						if (tNextTruncStates.defined())
							tTruncValPreds = PPOLearner::InferCriticFromModels(actorClone, tNextTruncStates.to(ppo->device, true, true), ppo->config.useHalfPrecision, snapPtr).cpu();
					} else {
						// Predict values using minibatching
						tValPreds = torch::zeros({ (int64_t)combinedTraj.Length() });
						for (int i = 0; i < combinedTraj.Length(); i += ppo->config.miniBatchSize) {
							int start = i;
							int end = RS_MIN(i + ppo->config.miniBatchSize, combinedTraj.Length());
							torch::Tensor tStatesPart = tStates.slice(0, start, end);

							auto valPredsPart = PPOLearner::InferCriticFromModels(actorClone, tStatesPart.to(ppo->device, true, true), ppo->config.useHalfPrecision, snapPtr).cpu();
							RG_ASSERT(valPredsPart.size(0) == (end - start));
							tValPreds.slice(0, start, end).copy_(valPredsPart, true);
						}

						if (tNextTruncStates.defined()) {
							// This really just should never happen
							// If this is ever actually a real problem in a legitimate use case, ping Zealan in the dead of night
							RG_ASSERT(tNextTruncStates.size(0) <= ppo->config.miniBatchSize);

							tTruncValPreds = PPOLearner::InferCriticFromModels(actorClone, tNextTruncStates.to(ppo->device, true, true), ppo->config.useHalfPrecision, snapPtr).cpu();
						}
					}

					float normalTerminalRate = (tTerminals == RLGC::TerminalType::NORMAL).to(torch::kFloat32).mean().item<float>();
					report["Episode Length"] = normalTerminalRate > 0 ? 1.f / normalTerminalRate : 0;

					// Build advantages + training batch: GAE, the goal-tensor alignment check, and buffer
					// packing now live in one pure, tested unit (BuildTrainingBatch). The critic
					// predictions above are passed in as data; the running return stat is read before
					// the call and updated after it, so that ordering stays visible here.
					TrainingBatchInputs batchIn;
					batchIn.states = tStates;
					batchIn.actions = tActions;
					batchIn.logProbs = tLogProbs;
					batchIn.actionMasks = tActionMasks;
					batchIn.rewards = tRewards;
					batchIn.terminals = tTerminals;
					batchIn.valPreds = tValPreds;
					batchIn.truncValPreds = tTruncValPreds;
					batchIn.gcrlEnabled = config.ppo.contrastiveGoal.enabled;
					if (batchIn.gcrlEnabled) {
						batchIn.achievedGoals = tAchievedGoals;
						batchIn.herGoals = tHERGoals;
						if (tCarHERGoals.defined())
							batchIn.carHerGoals = tCarHERGoals;
						batchIn.scoringGoals = tScoringGoals;
						batchIn.gcrlTrainMask = tGcrlTrainMask;
						batchIn.gcrlScoringMask = tGcrlScoringMask;
						batchIn.segmentIds = tSegmentIds;
						batchIn.segmentSteps = tSegmentSteps;
					}
					batchIn.gaeGamma = config.ppo.gaeGamma;
					batchIn.gaeLambda = config.ppo.gaeLambda;
					batchIn.rewardClipRange = config.ppo.rewardClipRange;
					batchIn.returnStd = returnStat ? returnStat->GetSTD() : 1;

					Timer gaeTimer = {};
					TrainingBatchResult batch = BuildTrainingBatch(batchIn, experience);
					report["GAE Time"] = gaeTimer.Elapsed();
					report["Clipped Reward Portion"] = batch.rewClipPortion;

					// Caller owns the running return stat: read old std (above) -> GAE -> update now.
					if (returnStat) {
						report["GAE/Returns STD"] = returnStat->GetSTD();

						int numToIncrement = RS_MIN(config.maxReturnSamples, batch.returns.size(0));
						if (numToIncrement > 0) {
							auto selectedReturns = batch.returns.index_select(0, torch::randint(batch.returns.size(0), { (int64_t)numToIncrement }));
							returnStat->Increment(TENSOR_TO_VEC<float>(selectedReturns));
						}
					}
					report["GAE/Avg Return"] = batch.avgReturn;
					report["GAE/Avg Advantage"] = batch.avgAdvantage;
					report["GAE/Avg Val Target"] = batch.avgValTarget;
				}
			} // end "Generate experience" (producer scope)
			}; // end fnProduce

			// ===== Pipeline =====
			// Overlap: collect rollout N+1 on a background thread (frozen actor clone + RSNorm snapshot) while
			// the main thread runs Learn on rollout N. Join BEFORE versionMgr->OnIteration so version-save /
			// skill-eval / checkpoint never run concurrently with collection. Depth-1 staleness, PPO-ratio-corrected.
			std::future<void> prodFut;
			int consumeIdx;
			if (overlap) {
				if (!overlapPrimed) {
					fnRefreshActor();
					fnProduce(curBuf);
					overlapPrimed = true;
				}
				int nextBuf = 1 - curBuf;
				fnRefreshActor();
				prodFut = std::async(std::launch::async, [&, nextBuf]() { fnProduce(nextBuf); });
				consumeIdx = curBuf;
				curBuf = nextBuf;
			} else {
				fnProduce(curBuf);
				consumeIdx = curBuf;
			}

			// ===== Consume buffer consumeIdx =====
			Report& report = repBuf[consumeIdx];
			ExperienceBuffer& experience = expBuf[consumeIdx];
			int stepsCollected = stepsBuf[consumeIdx];
			bool isFirstIteration = (totalTimesteps == 0);

				// Free CUDA cache
#ifdef RG_CUDA_SUPPORT
				if (ppo->device.is_cuda())
					c10::cuda::CUDACachingAllocator::emptyCache();
#endif

				// Learn
				Timer learnTimer = {};
				ppo->Learn(experience, report, isFirstIteration, totalTimesteps);
				float consumptionTime = learnTimer.Elapsed();
				report["PPO Learn Time"] = consumptionTime;

				// Join the background producer before the version manager / checkpoint touch shared state
				// (the producer reads the version manager + steps the env; OnIteration mutates them).
				if (prodFut.valid())
					prodFut.get();

				// Set metrics. With overlap the iteration wall-clock is ~max(produce, learn), so Overall SPS is
				// measured from the real iteration time (iterTimer), not collection+consumption summed.
				report["Consumption Time"] = consumptionTime;
				float collTime = report.Has("Collection Time") ? (float)report["Collection Time"] : 0.f;
				float iterTime = (float)iterTimer.Elapsed();

				auto calcStepsPerSecond = [](int steps, float seconds) {
					return seconds > 0 ? steps / seconds : 0.f;
				};
				report["Collection Steps/Second"] = calcStepsPerSecond(stepsCollected, collTime);
				report["Consumption Steps/Second"] = calcStepsPerSecond(stepsCollected, consumptionTime);
				report["Overall Steps/Second"] = calcStepsPerSecond(stepsCollected, iterTime);

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

				// FORK2 Part D: derived striking-quality metrics. report.Finish() has materialized all
				// AddAvg means into report.data, so these reads/divisions are valid here.
				// NOTE on what is/ isn't usable: GoalReward logs ~0 (zero-sum 1v1 cancellation), and
				// Save/Shot log under mangled "PlayerDataEventReward<...>" keys -> a "scoring" bucket
				// from them is unreliable. The RELIABLE, diagnostic decomposition is among the DENSE
				// terms: FINISHING (GoalwardImpact, the channel we boosted) vs PRESENCE (TouchBall +
				// StrongTouch + AerialTouch, the channel we trimmed) vs APPROACH (VPB). The key metric is
				// "Striking/Finishing Share": it should climb as the rebalance shifts play toward power
				// strikes (the wucwxpfx diagnosis had finishing ~6% of weighted dense income). Rewards/*
				// are RAW (pre-weight) -> multiply each by its ExampleMain weight (keep in sync).
				{
					const double eps = 1e-6;
					if (report.Has("Rewards/GoalwardImpactReward") && report.Has("Player/Ball Touch Ratio"))
						report["Striking/GoalwardImpact Per Touch"] =
							report["Rewards/GoalwardImpactReward"] / (report["Player/Ball Touch Ratio"] + eps);

					auto wget = [&](const char* k, double w) -> double {
						std::string key = std::string("Rewards/") + k;
						return report.Has(key) ? w * report[key] : 0.0;
					};
					double finishing = wget("GoalwardImpactReward", 100.0);
					double presence  = wget("StrongTouchReward", 40.0) + wget("TouchBallReward", 15.0)
						+ wget("AerialTouchReward", 12.0);
					double approach  = wget("VelocityPlayerToBallReward", 0.5);
					report["Striking/Finishing Weighted"] = finishing;
					report["Striking/Presence Weighted"] = presence;
					report["Striking/Approach Weighted"] = approach;
					double denom = std::abs(finishing) + std::abs(presence) + std::abs(approach) + eps;
					report["Striking/Finishing Share"] = finishing / denom;
					report["Striking/Presence Share"] = presence / denom;
				}

				if (metricSink)
					metricSink->Send(report);

				report.Display(
					{
						"Average Step Reward",
						"Policy Entropy",
						"Entropy Scale",
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
						"-Collection/Prep Time",
						"-Collection/Inference Time",
						"-Collection/Env Step Time",
						"-Collection/Record Time",
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

	} catch (std::exception& e) {
		RG_ERR_CLOSE("Exception thrown during main learner loop: " << e.what());
	}
}

GGL::Learner::~Learner() {
	delete ppo;
	delete versionMgr;
	// Only delete sinks we created; injected sinks are owned by the caller. The interpreter is
	// finalized by each pybind adapter's PythonRuntime guard as it is destroyed here.
	if (ownsMetricSink)
		delete metricSink;
	if (ownsRenderSink)
		delete renderSink;
}
