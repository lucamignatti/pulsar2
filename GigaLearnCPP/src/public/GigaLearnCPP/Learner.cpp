#include "Learner.h"

#include <GigaLearnCPP/PPO/PPOLearner.h>
#include <GigaLearnCPP/PPO/ExperienceBuffer.h>

#include <torch/cuda.h>
#include <nlohmann/json.hpp>
#include <pybind11/embed.h>

#ifdef RG_CUDA_SUPPORT
#include <c10/cuda/CUDACachingAllocator.h>
#endif
#include <private/GigaLearnCPP/PPO/ExperienceBuffer.h>
#include <private/GigaLearnCPP/PPO/GAE.h>
#include <private/GigaLearnCPP/PolicyVersionManager.h>
#include <private/GigaLearnCPP/PSD/PSDController.h>
#include <private/GigaLearnCPP/League/LeagueArchive.h>

#include "Util/KeyPressDetector.h"
#include <private/GigaLearnCPP/Util/WelfordStat.h>
#include "Util/AvgTracker.h"
#include <RLGymCPP/StateSetters/DrillBank.h>

using namespace RLGC;

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
				(torch::cuda::is_available() ? "libtorch cannot access the GPU" : "CUDA is not available to libtorch") << ".\n" <<
				"Make sure your libtorch comes with CUDA support, and that CUDA is installed properly."
			)
		device = at::Device(at::kCUDA);
	} else {
		RG_LOG("\tUsing CPU device...");
		device = at::Device(at::kCPU);
	}

	// Apply the TF32 matmul policy on CUDA (inert elsewhere) — without this the dense MLPs
	// run strict fp32 and never touch the Ampere+/Blackwell tensor-core path
	if (device.is_cuda()) {
		at::globalContext().setAllowTF32CuBLAS(config.allowTF32);
		at::globalContext().setAllowTF32CuDNN(config.allowTF32);
		RG_LOG("\tTF32 matmuls (CUDA tensor cores): " << (config.allowTF32 ? "enabled" : "disabled"));

		// On the GPU path the heavy math runs on-device; libtorch's default intra-op pool
		// (one thread per core) otherwise oversubscribes the machine against RLGymCPP's own
		// collection thread pool for the many small CPU-side tensor ops (index_select, blob
		// conversions, .cpu() copies) in the consumption phase. Cap it low.
		at::set_num_threads(2);
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
		envSetConfig.collectGatedBuckets = config.ppo.reachability.enabled && !config.renderMode;
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

	// Basin-Racing (PSD) + QD league — additive, created only when enabled. Built BEFORE Load()
	// so LoadStats can restore their persistent state from the checkpoint.
	if (config.psd.enabled && !config.renderMode) {
		psd = new PSDController(config.psd, device);
		psd->Init(envSet, ppo, (int)envSet->arenas.size(), config.checkpointFolder);
	}
	if (config.league.enabled && !config.renderMode) {
		league = new LeagueArchive(config.league, ppo, envSet, device, config.checkpointFolder);
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

	j["reach_acc_ema"] = reachAccEMA;
	j["reach_agree_ema"] = reachAgreeEMA;

	if (versionMgr)
		versionMgr->AddRunningStatsToJSON(j);

	if (psd)
		psd->ToJSON(j);
	if (league)
		league->ToJSON(j);

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

	if (j.contains("reach_acc_ema"))
		reachAccEMA = RS_MAX(0.f, (float)j["reach_acc_ema"]);
	if (j.contains("reach_agree_ema"))
		reachAgreeEMA = RS_MAX(0.f, (float)j["reach_agree_ema"]);

	if (versionMgr)
		versionMgr->LoadRunningStatsFromJSON(j);

	if (psd)
		psd->FromJSON(j);
	if (league)
		league->FromJSON(j);
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

bool GGL::Learner::ReloadNewestCheckpointForRender(int64_t& loadedTimesteps) {
	if (config.checkpointFolder.empty())
		return false;

	// Newest fully-numbered checkpoint dir. FindNumberedDirs already skips the sibling
	// policy_versions / league / psd_rounds folders (their names aren't all-digits).
	std::set<int64_t> saved = Utils::FindNumberedDirs(config.checkpointFolder);
	if (saved.empty())
		return false;
	int64_t newest = *saved.rbegin();
	if (newest <= loadedTimesteps)
		return false; // nothing newer than what's already on screen

	std::filesystem::path dir = config.checkpointFolder / std::to_string(newest);

	// The trainer writes a checkpoint's files one-by-one (POLICY.lt, CRITIC.lt, ... + optims), not
	// atomically, and it isn't built to drop a completion marker. So only swap once the directory
	// has *settled* -- nothing in it written for a moment -- to avoid reading a half-written .lt.
	constexpr double SETTLE_SECS = 2.0;
	try {
		if (!std::filesystem::exists(dir / "POLICY.lt"))
			return false;
		auto newestWrite = std::filesystem::file_time_type::min();
		for (const auto& entry : std::filesystem::directory_iterator(dir)) {
			auto t = entry.last_write_time();
			if (t > newestWrite)
				newestWrite = t;
		}
		auto age = std::filesystem::file_time_type::clock::now() - newestWrite;
		if (std::chrono::duration<double>(age).count() < SETTLE_SECS)
			return false; // still being written; retry on the next poll
	} catch (std::exception&) {
		return false; // dir raced (checkpoint rotation / partial write); retry next poll
	}

	// A viewer only runs the policy forward (obs -> shared_head -> policy -> action), so reload just
	// those two nets and skip the critic, reachability heads, and ALL optimizer state. That's
	// strictly lighter than the startup LoadFrom() -- which matters when this runs live alongside a
	// busy trainer -- so it can't regress the viewer's ability to coexist.
	//
	// Defense in depth: even settled, guard the load. Model::Load() throws (RG_ERR_CLOSE) on a
	// truncated/corrupt file, and that throw would otherwise be caught by the Start() loop's handler
	// and take the whole viewer down. Leave loadedTimesteps unchanged on failure so we re-attempt
	// this same dir on the next poll.
	try {
		ModelSet policyModels = ppo->GetPolicyModels();
		policyModels.Load(dir, /*allowNotExist=*/false, /*loadOptims=*/false);
	} catch (std::exception& e) {
		RG_LOG("[render] Checkpoint " << newest << " not ready, will retry: " << e.what());
		return false;
	}

	loadedTimesteps = newest;
	totalTimesteps = newest; // keep logging in sync with what's being shown
	RG_LOG("[render] Now visualizing checkpoint " << newest);
	return true;
}

void GGL::Learner::StartQuitKeyThread(std::atomic<bool>& quitPressed, std::thread& outThread) {
	quitPressed = false;

	// Detached runs (run_trainer.sh points stdin at /dev/null) have no terminal: the raw-mode
	// reader would spin on EOF and spam termios errors. Stop via the runner or SIGTERM instead.
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
		oldObsBuilders[i]->Reset(envSet->state.gameStates[0]);

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
		std::atomic<bool> saveQueued = false;
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

	// Render-mode live reload: follow the newest checkpoint a separate training process writes, so
	// the viewer tracks the model as it learns. Starts at the checkpoint loaded in the constructor
	// (0 if none); the timer throttles how often we poll the checkpoint folder.
	int64_t renderLoadedTS = (int64_t)totalTimesteps;
	Timer renderReloadTimer = {};
	if (render && config.renderReloadSecs > 0)
		RG_LOG("\t(Live reload: polling " << config.checkpointFolder << " every "
			<< config.renderReloadSecs << "s for newer checkpoints)");

	try {
		std::atomic<bool> saveQueued = false;
		std::thread keyPressThread;
		StartQuitKeyThread(saveQueued, keyPressThread);

		ExperienceBuffer experience = ExperienceBuffer(config.randomSeed, torch::kCPU);

		int numPlayers = envSet->state.numPlayers;

		struct Trajectory {
			FList states, nextStates, rewards, logProbs;
			// Goal-only reward channel for the secondary goal critic (only filled when
			// config.ppo.goalCritic.enabled): +1 own team scored / -1 conceded / 0 otherwise,
			// computed straight from game outcomes — independent of the reward stack.
			FList goalRews;
			std::vector<uint8_t> actionMasks;
			std::vector<int8_t> terminals;
			std::vector<int32_t> actions;

			// Reachability extras (only filled when config.ppo.reachability.enabled):
			FList oppStates;                    // Opponent's obs row (rho_opp for the control read)
			std::vector<uint8_t> oppActionMasks;
			// Achieved-state SCRATCH for the HER relabel: n rows appended per step + 1 extra
			// terminal-outcome row appended at finalize. Consumed by fnRelabelReachGoals and
			// dead afterwards, so deliberately NOT carried through Append().
			FList achievedBall;                 // Canonical normalized ball pos+vel, 6/row
			FList achievedCarBall;              // Car-local normalized ball pos+vel, 6/row
			FList achievedCarState;             // Canonical normalized CAR pos+vel, 6/row (car proposer)
			std::vector<uint8_t> touched, oppTouched; // Per-step touch flags (gate validity metric)
			FList gatedPos;                     // Positive parts of the gated reward components
			// Built at episode finalize by the HER relabel:
			FList carHerGoals, ballHerGoals;
			FList carStateHerGoals;             // future canonical car states (car proposer psi head)
			std::vector<uint8_t> ballMoved;

			// Deliberate-practice proposer extras (only filled when config.ppo.proposer.enabled):
			// achieved ball at row t and at row t+horizonSteps (clamped to the episode's terminal
			// outcome row), 6/row each - the (current, hindsight-target) pair the proposer trains
			// on. Unlike achievedBall/achievedCarBall these ARE carried through Append(), since
			// they're per-row training data, not per-episode scratch.
			FList proposerCurBall, proposerTargetBall;
			// Car proposer training pairs (canonical car state at t and t+horizonSteps), 6/row each.
			FList proposerCarCur, proposerCarTarget;

			// Deliberate-practice DRILL extras (only filled when propCfg.practiceEnabled and a
			// bank is configured): whether row t falls inside an active practice window, the
			// committed goal for that window (zeros if not practicing), which bank drill it came
			// from, and provenance (which trajectories[] slot / local step) so the learn-prep pass
			// can look up that step's banked ArenaSnapshot for a fresh Phi-drop detection.
			std::vector<uint8_t> practiceMask;
			FList practiceGoals;
			std::vector<int64_t> drillIds;
			std::vector<int32_t> srcPlayer, srcStep;

			void Clear() {
				*this = Trajectory();
			}

			// For the long-lived combinedTraj: drop contents but KEEP allocations, so the
			// ~100MB of per-iteration append targets aren't re-allocated every iteration
			void ClearKeepCapacity() {
				states.clear();
				nextStates.clear();
				rewards.clear();
				goalRews.clear();
				logProbs.clear();
				actionMasks.clear();
				terminals.clear();
				actions.clear();
				oppStates.clear();
				oppActionMasks.clear();
				achievedBall.clear();
				achievedCarBall.clear();
				achievedCarState.clear();
				touched.clear();
				oppTouched.clear();
				gatedPos.clear();
				carHerGoals.clear();
				ballHerGoals.clear();
				carStateHerGoals.clear();
				ballMoved.clear();
				proposerCurBall.clear();
				proposerTargetBall.clear();
				proposerCarCur.clear();
				proposerCarTarget.clear();
				practiceMask.clear();
				practiceGoals.clear();
				drillIds.clear();
				srcPlayer.clear();
				srcStep.clear();
			}

			void Reserve(size_t rows, int obsSize, int numActions, bool reach, bool proposer, bool practice) {
				states.reserve(rows * obsSize);
				actionMasks.reserve(rows * numActions);
				rewards.reserve(rows);
				goalRews.reserve(rows);
				logProbs.reserve(rows);
				terminals.reserve(rows);
				actions.reserve(rows);

				if (reach) {
					oppStates.reserve(rows * obsSize);
					oppActionMasks.reserve(rows * numActions);
					touched.reserve(rows);
					oppTouched.reserve(rows);
					gatedPos.reserve(rows);
					carHerGoals.reserve(rows * 6);
					ballHerGoals.reserve(rows * 6);
					ballMoved.reserve(rows);
				}

				if (proposer) {
					proposerCurBall.reserve(rows * 6);
					proposerTargetBall.reserve(rows * 6);
					proposerCarCur.reserve(rows * 6);
					proposerCarTarget.reserve(rows * 6);
					carStateHerGoals.reserve(rows * 6);
				}

				if (practice) {
					practiceMask.reserve(rows);
					practiceGoals.reserve(rows * 6);
					drillIds.reserve(rows);
					srcPlayer.reserve(rows);
					srcStep.reserve(rows);
				}
			}

			void Append(const Trajectory& other) {
				other.AssertAligned();

				states += other.states;
				nextStates += other.nextStates;
				rewards += other.rewards;
				goalRews += other.goalRews;
				logProbs += other.logProbs;
				actionMasks += other.actionMasks;
				terminals += other.terminals;
				actions += other.actions;

				oppStates += other.oppStates;
				oppActionMasks += other.oppActionMasks;
				touched += other.touched;
				oppTouched += other.oppTouched;
				gatedPos += other.gatedPos;
				carHerGoals += other.carHerGoals;
				ballHerGoals += other.ballHerGoals;
				carStateHerGoals += other.carStateHerGoals;
				ballMoved += other.ballMoved;

				proposerCurBall += other.proposerCurBall;
				proposerTargetBall += other.proposerTargetBall;
				proposerCarCur += other.proposerCarCur;
				proposerCarTarget += other.proposerCarTarget;

				practiceMask += other.practiceMask;
				practiceGoals += other.practiceGoals;
				drillIds += other.drillIds;
				srcPlayer += other.srcPlayer;
				srcStep += other.srcStep;
			}

			// Every per-row column must have exactly one entry per action row; a missed append
			// would silently pair rows with the wrong steps (goals/gate trained on garbage)
			void AssertAligned() const {
				size_t n = actions.size();
				RG_ASSERT(rewards.size() == n && logProbs.size() == n && terminals.size() == n);
				if (!goalRews.empty())
					RG_ASSERT(goalRews.size() == n);
				if (!touched.empty()) {
					RG_ASSERT(touched.size() == n && oppTouched.size() == n && gatedPos.size() == n);
					RG_ASSERT(carHerGoals.size() == n * 6 && ballHerGoals.size() == n * 6 && ballMoved.size() == n);
				}
				if (!proposerCurBall.empty())
					RG_ASSERT(proposerCurBall.size() == n * 6 && proposerTargetBall.size() == n * 6);
				if (!proposerCarCur.empty())
					RG_ASSERT(proposerCarCur.size() == n * 6 && proposerCarTarget.size() == n * 6);
				if (!carStateHerGoals.empty())
					RG_ASSERT(carStateHerGoals.size() == n * 6);
				if (!practiceMask.empty()) {
					RG_ASSERT(practiceMask.size() == n && practiceGoals.size() == n * 6 && drillIds.size() == n);
					RG_ASSERT(srcPlayer.size() == n && srcStep.size() == n);
				}
			}

			size_t Length() const {
				return actions.size();
			}
		};

		auto trajectories = std::vector<Trajectory>(numPlayers, Trajectory{});
		int maxEpisodeLength = (int)(config.ppo.maxEpisodeDuration * (120.f / config.tickSkip));

		// Reachability collection extras
		const auto& reachCfg = config.ppo.reachability;
		const bool reachOn = reachCfg.enabled && !render;

		// Deliberate-practice proposer collection extras (requires reach's achieved-ball buffers)
		const auto& propCfg = config.ppo.proposer;
		const bool proposerOn = propCfg.enabled && reachOn;
		const bool proposerCarOn = proposerOn && propCfg.carEnabled;

		// Deliberate-practice DRILLS (Stage 3): off unless explicitly enabled with a bank attached
		const bool practiceOn = proposerOn && propCfg.practiceEnabled && propCfg.drillBank != NULL;
		const bool goalCriticOn = config.ppo.goalCritic.enabled && !render;
		if (practiceOn) {
			propCfg.drillBank->Configure(
				(int)envSet->arenas.size(), propCfg.practiceWindowSteps, propCfg.maxDrillBankSize,
				propCfg.drillMaxTries, propCfg.drillMinTriesForRetire, propCfg.drillRetireSuccessRate);
		}

		// Static per-player maps: owning arena, slot within it, and the first opposing player
		// (whose obs row supplies rho_opp for the control read); teams never change mid-run
		std::vector<int> playerArenaIdx(numPlayers), playerSlotIdx(numPlayers), playerPartnerIdx(numPlayers, -1);
		// Per-arena "did team X touch this step" scratch, indexed [team][arena] (hoisted out of the step loop)
		std::vector<uint8_t> arenaTeamTouched[2];

		// Chunked parallel-for over [0, n) on the global pool, blocking until done. Used for the
		// per-player prep/record loops, whose serial cost dominates collection at high player
		// counts; each body writes only its own player's trajectory, so the work is race-free by
		// construction. INVARIANT: the pool must be quiescent when this is called (WaitUntilDone
		// is pool-global) — only call from the main collection thread, never while async arena
		// jobs are in flight.
		auto fnParallelFor = [](int n, const std::function<void(int)>& body) {
			if (n <= 0)
				return;

			int numChunks = RS_MIN(n, RS_MAX(1, RLGC::g_ThreadPool.GetNumThreads() * 4));
			int chunkSize = (n + numChunks - 1) / numChunks;
			RLGC::g_ThreadPool.StartBatchedJobs(
				[&body, n, chunkSize](int chunk) {
					int start = chunk * chunkSize;
					int stop = RS_MIN(start + chunkSize, n);
					for (int i = start; i < stop; i++)
						body(i);
				},
				numChunks, false
			);
		};

		// Terminal types decided by the parallel per-player pass, consumed by the serial finalize
		std::vector<int8_t> finalTerminals(numPlayers, 0);

		// Long-lived across iterations (capacity reused); only contains complete episodes.
		// Slack covers overbatching: collection finishes the step (and its whole episodes)
		// after crossing tsPerItr.
		auto combinedTraj = Trajectory();
		combinedTraj.Reserve((size_t)config.ppo.tsPerItr + numPlayers * 4, obsSize, numActions, reachOn, proposerOn, practiceOn);

		// Deliberate-practice DRILL snapshot scratch (Stage 3): one entry per (step, arena) this
		// iteration where a snapshot was captured, keyed by step*numArenas+arenaIdx. Cleared with
		// combinedTraj each iteration (step always restarts at 0) - the learn-prep Phi-drop pass
		// looks these up via each row's srcStep/srcPlayer provenance before they're dropped.
		std::unordered_map<int64_t, RLGC::ArenaSnapshot> stepSnapshots;
		if (reachOn) {
			for (int arenaIdx = 0; arenaIdx < envSet->arenas.size(); arenaIdx++) {
				int startIdx = envSet->state.arenaPlayerStartIdx[arenaIdx];
				auto& players = envSet->state.gameStates[arenaIdx].players;
				for (int i = 0; i < players.size(); i++) {
					playerArenaIdx[startIdx + i] = arenaIdx;
					playerSlotIdx[startIdx + i] = i;
					for (int j = 0; j < players.size(); j++) {
						if (players[j].team != players[i].team) {
							playerPartnerIdx[startIdx + i] = startIdx + j;
							break;
						}
					}
				}
			}
		}

		// Appends one achieved-state row (canonical ball + car-local ball, normalized) for
		// this player from the given game state. Used per step during collection AND once at
		// episode finalize for the terminal outcome state.
		auto fnAppendAchieved = [&](Trajectory& traj, const GameState& gs, const Player& player) {
			bool inv = (player.team == Team::ORANGE);
			float sign = inv ? -1.f : 1.f;

			// Canonical frame: +y is always the attacked net (negate x,y for ORANGE — the
			// same convention as StateUtil's InvertPhys, which the obs builders use)
			traj.achievedBall += sign * gs.ball.pos.x / reachCfg.posScaleX;
			traj.achievedBall += sign * gs.ball.pos.y / reachCfg.posScaleY;
			traj.achievedBall += gs.ball.pos.z / reachCfg.posScaleZ;
			traj.achievedBall += sign * gs.ball.vel.x / reachCfg.velScale;
			traj.achievedBall += sign * gs.ball.vel.y / reachCfg.velScale;
			traj.achievedBall += gs.ball.vel.z / reachCfg.velScale;

			// Car-local ball (~zeros = contact)
			Vec relPos = gs.ball.pos - player.pos;
			Vec relVel = gs.ball.vel - player.vel;
			traj.achievedCarBall += relPos.Dot(player.rotMat.forward) / reachCfg.carLocalScale;
			traj.achievedCarBall += relPos.Dot(player.rotMat.right) / reachCfg.carLocalScale;
			traj.achievedCarBall += relPos.Dot(player.rotMat.up) / reachCfg.carLocalScale;
			traj.achievedCarBall += relVel.Dot(player.rotMat.forward) / reachCfg.carLocalScale;
			traj.achievedCarBall += relVel.Dot(player.rotMat.right) / reachCfg.carLocalScale;
			traj.achievedCarBall += relVel.Dot(player.rotMat.up) / reachCfg.carLocalScale;

			// Canonical CAR state (the car proposer's goal space: where the CAR itself goes,
			// NOT ball-relative — this is what fixes the old car-critic's ball-chasing magnet).
			// Same y-flipped canonical frame + normalization as achievedBall.
			traj.achievedCarState += sign * player.pos.x / reachCfg.posScaleX;
			traj.achievedCarState += sign * player.pos.y / reachCfg.posScaleY;
			traj.achievedCarState += player.pos.z / reachCfg.posScaleZ;
			traj.achievedCarState += sign * player.vel.x / reachCfg.velScale;
			traj.achievedCarState += sign * player.vel.y / reachCfg.velScale;
			traj.achievedCarState += player.vel.z / reachCfg.velScale;
		};

		// HER relabeling for the reachability heads, run once per finalized episode.
		// Goals are FUTURE ACHIEVED states (true InfoNCE positives), short-biased. First
		// appends the terminal outcome state (post-step) as achieved row n, so scoring
		// states enter the goal distribution and no row ever trains on its own state as
		// the goal (offsets are always >= 1).
		auto fnRelabelReachGoals = [&](Trajectory& traj, int playerIdx) {
			if (!reachOn)
				return;

			int n = (int)traj.Length();
			if (n <= 0)
				return;

			{ // Terminal outcome state -> achieved row n
				auto& gs = envSet->state.gameStates[playerArenaIdx[playerIdx]];
				fnAppendAchieved(traj, gs, gs.players[playerSlotIdx[playerIdx]]);
			}
			RG_ASSERT((int)traj.achievedBall.size() == (n + 1) * 6);
			RG_ASSERT((int)traj.achievedCarBall.size() == (n + 1) * 6);

			// Short-biased draw from the window [lo, hi]
			auto fnDrawOffset = [](int lo, int hi, float biasPow) {
				float u = powf(RocketSim::Math::RandFloat(), biasPow);
				int span = hi - lo + 1;
				return lo + RS_MIN(span - 1, (int)(u * span));
			};

			// Picks the achieved-row offset for row t: off in [1, n - t] (achieved has n+1
			// rows, so t + off is always a strictly-future state, terminal row included)
			auto fnPickOffset = [&](int t, int minOffCfg, int maxOffCfg, float biasPow, float goalwardBias) {
				int remaining = n - t; // >= 1
				int hi = RS_MIN(RS_MAX(minOffCfg, maxOffCfg), remaining);
				int lo = RS_MIN(RS_MAX(1, minOffCfg), remaining);
				if (hi < lo)
					return remaining;

				if (goalwardBias > 0 && RocketSim::Math::RandFloat() < goalwardBias) {
					// Most-goalward (max canonical +y) achieved state in the window, so real
					// near-net states populate the goal space (keeps the fixed scoring query
					// goal in-distribution)
					int best = lo;
					float bestY = traj.achievedBall[(size_t)(t + lo) * 6 + 1];
					for (int o = lo + 1; o <= hi; o++) {
						float y = traj.achievedBall[(size_t)(t + o) * 6 + 1];
						if (y > bestY) {
							bestY = y;
							best = o;
						}
					}
					return best;
				}

				return fnDrawOffset(lo, hi, biasPow);
			};

			float ballBiasPow = RS_MAX(1e-3f, reachCfg.ballHerShortBiasPower);
			float ballGoalwardBias = RS_CLAMP(reachCfg.ballHerGoalwardBias, 0.f, 1.f);
			float carBiasPow = RS_MAX(1e-3f, reachCfg.carHerShortBiasPower);

			traj.ballHerGoals.resize((size_t)n * 6);
			traj.carHerGoals.resize((size_t)n * 6);
			if (proposerCarOn)
				traj.carStateHerGoals.resize((size_t)n * 6);
			for (int t = 0; t < n; t++) {
				int ballOff = fnPickOffset(t, reachCfg.ballHerMinOffset, reachCfg.ballHerMaxOffset, ballBiasPow, ballGoalwardBias);
				// Controllability is local: short window, no goalward bias
				int carOff = fnPickOffset(t, reachCfg.carHerMinOffset, reachCfg.carHerMaxOffset, carBiasPow, 0);

				for (int d = 0; d < 6; d++) {
					traj.ballHerGoals[(size_t)t * 6 + d] = traj.achievedBall[(size_t)(t + ballOff) * 6 + d];
					traj.carHerGoals[(size_t)t * 6 + d] = traj.achievedCarBall[(size_t)(t + carOff) * 6 + d];
				}

				// Car-STATE HER goal (for the car proposer's psi head): future canonical car state,
				// broad window (~proposer horizon), no goalward bias - car states have no +y analog
				if (proposerCarOn) {
					int carStateOff = fnPickOffset(t, reachCfg.ballHerMinOffset, reachCfg.ballHerMaxOffset, ballBiasPow, 0);
					for (int d = 0; d < 6; d++)
						traj.carStateHerGoals[(size_t)t * 6 + d] = traj.achievedCarState[(size_t)(t + carStateOff) * 6 + d];
				}
			}

			// Ball-moved mask: the ball head only trains on episodes where the ball actually moved
			// (a dead never-touched episode would just reteach the stationary-ball manifold)
			{
				float thresh = reachCfg.minBallMoveSpeed / RS_MAX(1e-6f, reachCfg.velScale);
				float threshSq = thresh * thresh;
				bool moved = false;
				for (int t = 0; t <= n && !moved; t++) {
					float vx = traj.achievedBall[(size_t)t * 6 + 3];
					float vy = traj.achievedBall[(size_t)t * 6 + 4];
					float vz = traj.achievedBall[(size_t)t * 6 + 5];
					if (vx * vx + vy * vy + vz * vz >= threshSq)
						moved = true;
				}
				traj.ballMoved.assign((size_t)n, moved ? 1 : 0);
			}
		};

		// Deliberate-practice proposer training targets: for each row t, records the achieved ball
		// at t (proposerCurBall) and at t+horizonSteps, clamped to the episode's terminal outcome
		// row (proposerTargetBall) - the (current, hindsight-target) pair the proposer regresses
		// onto. Must run AFTER fnRelabelReachGoals (needs the n+1-row achievedBall it just built,
		// before that scratch is dropped for the next episode).
		auto fnAppendProposerTargets = [&](Trajectory& traj) {
			if (!proposerOn)
				return;

			int n = (int)traj.Length();
			if (n <= 0)
				return;

			RG_ASSERT((int)traj.achievedBall.size() == (n + 1) * 6);

			int horizon = RS_MAX(1, propCfg.horizonSteps);
			traj.proposerCurBall.resize((size_t)n * 6);
			traj.proposerTargetBall.resize((size_t)n * 6);
			bool carOn = proposerCarOn;
			if (carOn) {
				RG_ASSERT((int)traj.achievedCarState.size() == (n + 1) * 6);
				traj.proposerCarCur.resize((size_t)n * 6);
				traj.proposerCarTarget.resize((size_t)n * 6);
			}
			for (int t = 0; t < n; t++) {
				int targetRow = RS_MIN(t + horizon, n);
				for (int d = 0; d < 6; d++) {
					traj.proposerCurBall[(size_t)t * 6 + d] = traj.achievedBall[(size_t)t * 6 + d];
					traj.proposerTargetBall[(size_t)t * 6 + d] = traj.achievedBall[(size_t)targetRow * 6 + d];
					if (carOn) {
						traj.proposerCarCur[(size_t)t * 6 + d] = traj.achievedCarState[(size_t)t * 6 + d];
						traj.proposerCarTarget[(size_t)t * 6 + d] = traj.achievedCarState[(size_t)targetRow * 6 + d];
					}
				}
			}
		};

		while (true) {
			Report report = {};

			bool isFirstIteration = (totalTimesteps == 0);

			// This iteration's opponent for the non-self team: an old policy version, a PFSP-sampled
			// league member, or (default) the current self. `oppModels` is the opponent's network
			// (null = mirror self-play); the player masking, trajectory exclusion, and split inference
			// below are shared by all three sources. The sources are mutually exclusive and the choice
			// is made once per iteration.
			ModelSet* oppModels = nullptr;
			std::vector<bool> oldVersionPlayerMask;
			std::vector<int> newPlayerIndices = {}, oldPlayerIndices = {};
			torch::Tensor tNewPlayerIndices, tOldPlayerIndices;

			for (int i = 0; i < numPlayers; i++)
				newPlayerIndices.push_back(i);

			if (!render) {
				RG_ASSERT(config.trainAgainstOldChance >= 0 && config.trainAgainstOldChance <= 1);
				if (config.trainAgainstOldVersions && versionMgr && !versionMgr->versions.empty()
					&& RocketSim::Math::RandFloat() < config.trainAgainstOldChance) {
					int oldVersionIdx = RocketSim::Math::RandInt(0, versionMgr->versions.size());
					oppModels = &versionMgr->versions[oldVersionIdx].models;
				} else if (league && RocketSim::Math::RandFloat() < config.league.descendOpponentFrac) {
					// PFSP-sampled league member -> the exposure to non-self styles the league is for
					// (returns null while the archive is still empty, falling back to self-play).
					oppModels = league->LoadPFSPOpponentModels();
				}
			}

			if (oppModels) {
				Team oppTeam = Team(RocketSim::Math::RandInt(0, 2));

				newPlayerIndices.clear();
				oldVersionPlayerMask.resize(numPlayers);
				int i = 0;
				for (auto& state : envSet->state.gameStates) {
					for (auto& player : state.players) {
						if (player.team == oppTeam) {
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

			int numRealPlayers = oppModels ? newPlayerIndices.size() : envSet->state.numPlayers;

			int stepsCollected = 0;
			{ // Generate experience

				combinedTraj.ClearKeepCapacity();
				stepSnapshots.clear();

				// Players handed to the opponent this iteration stop being collected; their in-flight
				// partial episode must not silently SPLICE with a later episode when they return to
				// collecting (HER goals and gate windows would cross a hidden reset). DISCARD the
				// partial (Clear) rather than finalize-and-append it: appending up to half the players'
				// in-flight episodes into this iteration's freshly-cleared buffer can exceed tsPerItr
				// and starve fresh collection entirely — the loop condition is `Length() < tsPerItr`,
				// so a pre-filled buffer collects ZERO steps (Collection SPS -> 0, no progress). The
				// discarded tails are minor, slightly-off-policy truncated data PPO doesn't want; the
				// collection loop still gathers a full tsPerItr of fresh on-policy experience.
				if (oppModels) {
					for (int oldPlayerIdx : oldPlayerIndices)
						trajectories[oldPlayerIdx].Clear();
				}

				Timer collectionTimer = {};
				{ // Collect timesteps
					RG_NO_GRAD;

					float inferTime = 0;
					float envStepTime = 0;
					float prepTime = 0;
					float recordTime = 0;

					for (int step = 0; combinedTraj.Length() < config.ppo.tsPerItr || render; step++, stepsCollected += numRealPlayers) {
						Timer stepTimer = {};
						// Drop any practice window whose arena is about to reset for a reason
						// OTHER than the drill itself (terminals still hold the PREVIOUS step's
						// flags here - Reset() only zeroes them for arenas that actually reset)
						if (practiceOn)
							propCfg.drillBank->ClearWindowsForResets(envSet->state.terminals);
						envSet->Reset();
						envStepTime += stepTimer.Elapsed();

						// Sampled every 4th step: this is a hard-stop debug guard, and any obs-NaN
						// source persists across steps, so sampling keeps the guarantee while
						// dropping ~75% of its (measurable) cost
						if ((step & 3) == 0) {
							torch::Tensor tObsCheck = torch::from_blob(
								envSet->state.obs.data.data(), { (int64_t)envSet->state.obs.data.size() }, torch::kFloat32);
							if (!torch::isfinite(tObsCheck).all().item<bool>())
								RG_ERR_CLOSE("Obs builder produced a NaN/inf value");
						}

						if (!render && obsStat) {
							// TODO: This samples from old versions too
							int numSamples = RS_MAX(envSet->state.numPlayers, config.maxObsSamples);
							for (int i = 0; i < numSamples; i++) {
								int idx = Math::RandInt(0, envSet->state.numPlayers);
								obsStat->IncrementRow(&envSet->state.obs.At(idx, 0));
							}

							std::vector<double> mean = obsStat->GetMean();
							std::vector<double> std = obsStat->GetSTD();
							for (double& f : mean)
								f = RS_CLAMP(f, -config.maxObsMeanRange, config.maxObsMeanRange);
							for (double& f : std)
								f = RS_MAX(f, config.minObsSTD);
							for (int i = 0; i < envSet->state.numPlayers; i++) {
								for (int j = 0; j < obsSize; j++) {
									float& obsVal = envSet->state.obs.At(i, j);
									obsVal = (obsVal - mean[j]) / std[j];
								}
							}
						}

						// Zero-copy views over the env's storage (safe: they're consumed within this
						// step — device transfer / index_select materialize immediately, and the
						// underlying buffers only mutate on the next env step)
						torch::Tensor tActions, tLogProbs;
						torch::Tensor tStates = torch::from_blob(
							envSet->state.obs.data.data(),
							{ (int64_t)envSet->state.obs.size[0], (int64_t)envSet->state.obs.size[1] },
							torch::kFloat32);
						torch::Tensor tActionMasks = torch::from_blob(
							envSet->state.actionMasks.data.data(),
							{ (int64_t)envSet->state.actionMasks.size[0], (int64_t)envSet->state.actionMasks.size[1] },
							torch::kUInt8);

						if (!render) {
							// Parallel per-player: each body writes only trajectories[newPlayerIdx]
							Timer prepTimer = {};
							fnParallelFor((int)newPlayerIndices.size(), [&](int k) {
								int newPlayerIdx = newPlayerIndices[k];
								trajectories[newPlayerIdx].states += envSet->state.obs.GetRow(newPlayerIdx);
								trajectories[newPlayerIdx].actionMasks += envSet->state.actionMasks.GetRow(newPlayerIdx);

								if (reachOn) {
									auto& traj = trajectories[newPlayerIdx];

									// Opponent's view of the same moment (rho_opp for the control read);
									// if the mode somehow has no opponent, reuse our own row (control reads 0.5)
									int partner = playerPartnerIdx[newPlayerIdx];
									int oppIdx = (partner >= 0) ? partner : newPlayerIdx;
									traj.oppStates += envSet->state.obs.GetRow(oppIdx);
									traj.oppActionMasks += envSet->state.actionMasks.GetRow(oppIdx);

									auto& gs = envSet->state.gameStates[playerArenaIdx[newPlayerIdx]];
									fnAppendAchieved(traj, gs, gs.players[playerSlotIdx[newPlayerIdx]]);
								}
							});
							prepTime += prepTimer.Elapsed();
						}

						envSet->StepFirstHalf(true);

						Timer inferTimer = {};

						if (oppModels) {
							torch::Tensor tdNewStates = tStates.index_select(0, tNewPlayerIndices).to(ppo->device, true);
							torch::Tensor tdOldStates = tStates.index_select(0, tOldPlayerIndices).to(ppo->device, true);
							torch::Tensor tdNewActionMasks = tActionMasks.index_select(0, tNewPlayerIndices).to(ppo->device, true);
							torch::Tensor tdOldActionMasks = tActionMasks.index_select(0, tOldPlayerIndices).to(ppo->device, true);

							torch::Tensor tNewActions;
							torch::Tensor tOldActions;

							ppo->InferActions(tdNewStates, tdNewActionMasks, &tNewActions, &tLogProbs);
							ppo->InferActions(tdOldStates, tdOldActionMasks, &tOldActions, NULL, oppModels);

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
							if (config.renderReloadSecs > 0 && renderReloadTimer.Elapsed() >= config.renderReloadSecs) {
								renderReloadTimer.Reset();
								ReloadNewestCheckpointForRender(renderLoadedTS);
							}
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

						Timer recordTimer = {};

						// Per-arena, per-team touch flags for this step (avoids an O(players^2) scan);
						// serial: writes shared vectors, and it's O(players) cheap
						if (reachOn) {
							arenaTeamTouched[0].assign(envSet->arenas.size(), 0);
							arenaTeamTouched[1].assign(envSet->arenas.size(), 0);
							for (int arenaIdx = 0; arenaIdx < envSet->arenas.size(); arenaIdx++)
								for (auto& player : envSet->state.gameStates[arenaIdx].players)
									if (player.ballTouchedStep)
										arenaTeamTouched[(int)player.team][arenaIdx] = 1;
						}

						// Deliberate-practice DRILL snapshot capture (Stage 3): every snapshotEveryK
						// steps, record enough of each arena's physics state to restore play from
						// exactly here later. Serial - O(numArenas * playersPerArena), same order
						// as the arenaTeamTouched scan just above.
						if (practiceOn && (step % RS_MAX(1, propCfg.snapshotEveryK)) == 0) {
							for (int arenaIdx = 0; arenaIdx < (int)envSet->arenas.size(); arenaIdx++) {
								RLGC::ArenaSnapshot snap;
								auto& gs = envSet->state.gameStates[arenaIdx];
								snap.ball = gs.ball;
								for (auto& player : gs.players) {
									RLGC::ArenaSnapshot::CarSnap cs;
									cs.team = player.team;
									cs.state = (CarState)player;
									snap.cars.push_back(cs);
								}
								for (BoostPad* pad : envSet->arenas[arenaIdx]->GetBoostPads())
									snap.pads.push_back(pad->GetState());

								stepSnapshots[(int64_t)step * (int64_t)envSet->arenas.size() + (int64_t)arenaIdx] = std::move(snap);
							}
						}

						// Now that we've inferred and stepped the env, we can add that stuff to the
						// trajectories. Parallel per-player; logProbs is indexed by the ordinal k
						// (its rows follow newPlayerIndices order, not global player order).
						fnParallelFor((int)newPlayerIndices.size(), [&](int k) {
							int newPlayerIdx = newPlayerIndices[k];
							trajectories[newPlayerIdx].actions.push_back(curActions[newPlayerIdx]);
							trajectories[newPlayerIdx].rewards += envSet->state.rewards[newPlayerIdx];
							trajectories[newPlayerIdx].logProbs += newLogProbs[k];

							if (goalCriticOn) {
								// Goal-only channel, straight from game outcomes (same convention as
								// GoalReward: RS_TEAM_FROM_Y(ball.y) = the team whose net the ball is
								// in = the CONCEDING team; the scorer is the other one).
								auto& gs = envSet->state.gameStates[playerArenaIdx[newPlayerIdx]];
								float gr = 0;
								if (gs.goalScored) {
									auto& player = gs.players[playerSlotIdx[newPlayerIdx]];
									gr = (player.team != RS_TEAM_FROM_Y(gs.ball.pos.y)) ? 1.f : -1.f;
								}
								trajectories[newPlayerIdx].goalRews += gr;
							}

							if (reachOn) {
								auto& traj = trajectories[newPlayerIdx];
								traj.gatedPos += envSet->state.gatedPosRewards[newPlayerIdx];

								// Post-step touch flags (which team touched during this step)
								int arenaIdx = playerArenaIdx[newPlayerIdx];
								auto& player = envSet->state.gameStates[arenaIdx].players[playerSlotIdx[newPlayerIdx]];
								traj.touched.push_back(player.ballTouchedStep);
								traj.oppTouched.push_back(arenaTeamTouched[player.team == Team::BLUE ? 1 : 0][arenaIdx]);

								// Deliberate-practice DRILL per-row tagging (Stage 3): only the
								// PRACTICING team's rows get tagged - the opponent trains normally
								if (practiceOn) {
									auto win = propCfg.drillBank->GetWindow(arenaIdx);
									bool practicing = win.active && player.team == win.sourceTeam;
									traj.practiceMask.push_back(practicing ? 1 : 0);
									for (int d = 0; d < 6; d++)
										traj.practiceGoals.push_back(practicing ? win.goal[d] : 0.f);
									traj.drillIds.push_back(practicing ? (int64_t)win.drillId : 0);
									traj.srcPlayer.push_back(newPlayerIdx);
									traj.srcStep.push_back(step);
								}
							}
						});

						if (practiceOn)
							propCfg.drillBank->DecrementWindows();

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

						// Parallel per-player: decide the terminal type, record it, store the
						// truncation next-state. Finalization stays serial below — the HER relabel
						// draws from the shared RNG and combinedTraj is shared.
						fnParallelFor((int)newPlayerIndices.size(), [&](int k) {
							int newPlayerIdx = newPlayerIndices[k];
							int8_t terminalType = curTerminals[newPlayerIdx];
							auto& traj = trajectories[newPlayerIdx];

							if (!terminalType && traj.Length() >= maxEpisodeLength) {
								// Episode is too long, truncate it here
								// This won't actually reset the env, but rather will just add it to experience buffer as truncated
								terminalType = RLGC::TerminalType::TRUNCATED;
							}

							traj.terminals.push_back(terminalType);
							if (terminalType == RLGC::TerminalType::TRUNCATED) {
								// Truncation requires an additional next state for the critic
								traj.nextStates += envSet->state.obs.GetRow(newPlayerIdx);
							}

							finalTerminals[newPlayerIdx] = terminalType;
						});

						// Serial finalize, in the original player order (deterministic episode
						// order in combinedTraj, same as the old fully-serial loop)
						for (int newPlayerIdx : newPlayerIndices) {
							if (!finalTerminals[newPlayerIdx])
								continue;

							auto& traj = trajectories[newPlayerIdx];
							fnRelabelReachGoals(traj, newPlayerIdx);
							fnAppendProposerTargets(traj);
							combinedTraj.Append(traj);
							traj.Clear();
						}

						recordTime += recordTimer.Elapsed();
					}

					report["Inference Time"] = inferTime;
					report["Env Step Time"] = envStepTime;
					report["Prep Time"] = prepTime;
					report["Record Time"] = recordTime;
				}
				float collectionTime = collectionTimer.Elapsed();

				Timer consumptionTimer = {};

				// Deliberate-practice proposer: Train() needs gradients, but the whole "Process
				// timesteps" block below runs under RG_NO_GRAD (like the reachability gate's own
				// no-grad rho reads) - so its inputs are captured here and Train() is called AFTER
				// the block closes, mirroring how ppo->Learn() itself (which also needs gradients)
				// is deferred to after this block.
				torch::Tensor tPropFeatures, tPropPrevGoals, tPropTargets, tPropWeights;
				// Car head shares tPropFeatures + tPropWeights (same trunk features + aspiration
				// weights); only its regression pair differs.
				torch::Tensor tPropCarPrevGoals, tPropCarTargets;

				{ // Process timesteps
					RG_NO_GRAD;

					// Make and transpose tensors
					torch::Tensor tStates = torch::tensor(combinedTraj.states).reshape({ -1, obsSize });
					torch::Tensor tActionMasks = torch::tensor(combinedTraj.actionMasks).reshape({ -1, numActions });
					torch::Tensor tActions = torch::tensor(combinedTraj.actions);
					torch::Tensor tLogProbs = torch::tensor(combinedTraj.logProbs);
					torch::Tensor tRewards = torch::tensor(combinedTraj.rewards);
					torch::Tensor tTerminals = torch::tensor(combinedTraj.terminals);

					// States we truncated at (there could be none)
					torch::Tensor tNextTruncStates;
					if (!combinedTraj.nextStates.empty())
						tNextTruncStates = torch::tensor(combinedTraj.nextStates).reshape({ -1, obsSize });

					report["Average Step Reward"] = tRewards.mean().item<float>();
					report["Collected Timesteps"] = stepsCollected;
					
					Timer valPredTimer = {};
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

					// Secondary goal-critic value predictions (same minibatching pattern).
					torch::Tensor tGoalValPreds, tGoalTruncValPreds;
					if (goalCriticOn) {
						if (ppo->device.is_cpu()) {
							tGoalValPreds = ppo->InferGoalCritic(tStates.to(ppo->device, true, true)).cpu();
						} else {
							tGoalValPreds = torch::zeros({ (int64_t)combinedTraj.Length() });
							for (int i = 0; i < combinedTraj.Length(); i += ppo->config.miniBatchSize) {
								int start = i;
								int end = RS_MIN(i + ppo->config.miniBatchSize, combinedTraj.Length());
								auto part = ppo->InferGoalCritic(tStates.slice(0, start, end).to(ppo->device, true, true)).cpu();
								tGoalValPreds.slice(0, start, end).copy_(part, true);
							}
						}
						if (tNextTruncStates.defined())
							tGoalTruncValPreds = ppo->InferGoalCritic(tNextTruncStates.to(ppo->device, true, true)).cpu();
					}
					report["Value Pred Time"] = valPredTimer.Elapsed();

					// Only report when at least one NORMAL terminal occurred; an all-truncated
					// iteration (every episode hit maxEpisodeLength / handoff) makes the fraction
					// 0 and the reciprocal +inf, which poisons the logged series.
					float normalTermFrac = (tTerminals == 1).to(torch::kFloat32).mean().item<float>();
					if (normalTermFrac > 0)
						report["Episode Length"] = 1.f / normalTermFrac;

					// Shared FP32 trunk over the batch states, computed ONCE and reused by the
					// reachability gate reads, the per-row rho reads, and the proposer unroll — each
					// of which used to re-forward sharedHead over all n rows independently (3-4
					// full-batch trunk passes per iteration). Detached, on device. The critic is
					// deliberately NOT folded in: it forwards the trunk at useHalfPrecision, a
					// different dtype, so its trunk isn't interchangeable with this FP32 one. Left
					// undefined when there's no shared head (the rho reads then feed raw obs to phi,
					// exactly as before) or when neither consumer is active.
					// Gate-off cadence: when the gate is disabled the whole rho-read block below (3
					// full-buffer model passes + CPU smoothing) feeds nothing but Reach/* dashboard
					// panels, so refresh them every diagEveryIters instead of every iteration. When
					// the gate is ON, rewards depend on the reads — run every iteration as before.
					const bool reachReadsThisIter = reachOn && ppo->reach &&
						(reachCfg.gateEnabled ||
						 (totalIterations % (uint64_t)RS_MAX(1, reachCfg.diagEveryIters)) == 0);

					Timer reachReadTimer = {};
					torch::Tensor tTrunkFeatures;
					if ((reachReadsThisIter || proposerOn) && combinedTraj.Length() > 0) {
						Model* trunkModel = ppo->models["shared_head"];
						if (trunkModel) {
							int64_t nRows = (int64_t)combinedTraj.Length();
							int64_t chunk = proposerOn ? (int64_t)propCfg.featureChunkSize : (int64_t)reachCfg.scoreChunkSize;
							if (chunk <= 0)
								chunk = nRows;
							std::vector<torch::Tensor> parts;
							for (int64_t s = 0; s < nRows; s += chunk) {
								int64_t e = RS_MIN(s + chunk, nRows);
								torch::Tensor obsChunk = tStates.slice(0, s, e).to(ppo->device, true);
								parts.push_back(trunkModel->Forward(obsChunk, false).detach());
							}
							tTrunkFeatures = parts.size() == 1 ? parts[0] : torch::cat(parts, 0);
						}
					}

					// Reachability: rho reads -> level x delta gate multiplier + validity metrics.
					// The gate scales REWARDS only (never advantages/values); with beta=0 or
					// gateEnabled=false the rewards are untouched.
					if (reachReadsThisIter && combinedTraj.Length() > 0) {
						int64_t n = (int64_t)combinedTraj.Length();

						torch::Tensor tOppStates = torch::tensor(combinedTraj.oppStates).reshape({ -1, obsSize });
						torch::Tensor tOppMasks = torch::tensor(combinedTraj.oppActionMasks).reshape({ -1, numActions });

						// Fixed query goals, normalized like the HER goals (canonical frame)
						torch::Tensor contactGoal = torch::zeros({ 6 });
						torch::Tensor scoringGoal = torch::tensor({
							0.f,
							CommonValues::BACK_WALL_Y / reachCfg.posScaleY,
							(CommonValues::GOAL_HEIGHT * 0.5f) / reachCfg.posScaleZ,
							0.f,
							reachCfg.scoringGoalSpeed / reachCfg.velScale,
							0.f
						});

						// One pass over our states answers both heads (the trunk/phi work is
						// goal-independent); the opponent states need their own pass
						Model* sharedHead = ppo->models["shared_head"];
						auto rhoOwn = ppo->reach->EvalRho(sharedHead,
							{ { ppo->reach->psiCar, contactGoal }, { ppo->reach->psiBall, scoringGoal } },
							tStates, tActionMasks, tTrunkFeatures);
						auto rhoOpp = ppo->reach->EvalRho(sharedHead,
							{ { ppo->reach->psiCar, contactGoal } },
							tOppStates, tOppMasks);
						FList rhoCarUs = TENSOR_TO_VEC<float>(rhoOwn[0]);
						FList rhoScore = TENSOR_TO_VEC<float>(rhoOwn[1]);
						FList rhoCarOpp = TENSOR_TO_VEC<float>(rhoOpp[0]);

						auto& terminals = combinedTraj.terminals;

						// Segment starts (combinedTraj = complete episodes back-to-back; a nonzero
						// terminal marks the LAST row of its episode)
						std::vector<int64_t> segStart(n);
						{
							int64_t curStart = 0;
							for (int64_t i = 0; i < n; i++) {
								segStart[i] = curStart;
								if (terminals[i])
									curStart = i + 1;
							}
						}

						// Smoothed reads (boxcar, clipped at the segment start) - the raw per-step
						// delta of a slowly-moving learned logit is noise
						int smooth = RS_MAX(1, reachCfg.deltaSmooth);
						FList relSmooth(n), scoSmooth(n);
						for (int64_t i = 0; i < n; i++) {
							int64_t lo = RS_MAX(segStart[i], i - smooth + 1);
							float accR = 0, accS = 0;
							for (int64_t j = lo; j <= i; j++) {
								accR += rhoCarUs[j] - rhoCarOpp[j];
								accS += rhoScore[j];
							}
							int cnt = (int)(i - lo + 1);
							relSmooth[i] = accR / cnt;
							scoSmooth[i] = accS / cnt;
						}

						// Windowed progress deltas; zero where the window would cross an episode start
						int window = RS_MAX(1, reachCfg.deltaWindow);
						FList deltaRel(n, 0.f), deltaSco(n, 0.f);
						for (int64_t i = 0; i < n; i++) {
							if (i - window >= segStart[i]) {
								deltaRel[i] = relSmooth[i] - relSmooth[i - window];
								deltaSco[i] = scoSmooth[i] - scoSmooth[i - window];
							}
						}

						auto fnSigmoid = [](float x) { return 1.f / (1.f + expf(-x)); };

						// Validity: does sign(deltaControl) predict which team touches the ball next?
						// (Measured on realized touches; this is what lets the gate open.)
						float agreement = -1;
						{
							int64_t agreeCount = 0, labelCount = 0;
							int horizon = RS_MAX(1, reachCfg.touchPredHorizon);
							auto& touched = combinedTraj.touched;
							auto& oppTouched = combinedTraj.oppTouched;
							for (int64_t i = 0; i < n; i++) {
								// A terminal row has no future in its own episode — scanning
								// onward would label it with the NEXT episode's touches
								if (terminals[i])
									continue;

								// The first deltaWindow rows of each episode have their delta
								// structurally zeroed; scoring sign(0) would systematically
								// count them as "predicts opponent" and bias the metric down
								if (i - window < segStart[i])
									continue;

								for (int64_t j = i + 1; j <= RS_MIN(n - 1, i + horizon); j++) {
									bool own = touched[j], opp = oppTouched[j];
									if (own && !opp) {
										labelCount++;
										agreeCount += (deltaRel[i] > 0);
										break;
									}
									if (opp && !own) {
										labelCount++;
										agreeCount += (deltaRel[i] <= 0);
										break;
									}
									if ((own && opp) || terminals[j])
										break; // Ambiguous, or the episode ended without a touch
								}
							}
							if (labelCount > 0)
								agreement = (float)agreeCount / labelCount;
						}

						// Anneal: the gate must EARN its way in (head accuracy AND touch-prediction validity)
						float beta;
						{
							float decay = RS_CLAMP(reachCfg.emaDecay, 0.f, 0.999999f);
							// Only blend once the heads have actually trained this process —
							// otherwise iteration 1 (and every checkpoint resume) would knock
							// a healthy persisted EMA toward the meaningless initial 0
							if (ppo->lastReachTrained)
								reachAccEMA = decay * reachAccEMA + (1 - decay) * ppo->lastReachAccuracy;
							if (agreement >= 0)
								reachAgreeEMA = decay * reachAgreeEMA + (1 - decay) * agreement;

							auto fnSmoothstep = [](float x, float lo, float hi) {
								float t = RS_CLAMP((x - lo) / RS_MAX(1e-6f, hi - lo), 0.f, 1.f);
								return t * t * (3 - 2 * t);
							};
							beta = RS_MIN(
								fnSmoothstep(reachAccEMA, reachCfg.accLo, reachCfg.accHi),
								fnSmoothstep(reachAgreeEMA, reachCfg.aucLo, reachCfg.aucHi)
							);
							if (reachCfg.betaOverride >= 0)
								beta = RS_CLAMP(reachCfg.betaOverride, 0.f, 1.f);
						}

						// level x delta gate: level = anti-farm ("does this state matter"),
						// delta = progress credit ("is this helping"); delta=0 => exactly the level gate
						FList mult(n);
						float alpha = RS_CLAMP(reachCfg.gateAlpha, 0.f, 1.f);
						double multSum = 0, levelSum = 0, controlSum = 0, scoringSum = 0;
						double rhoUsSum = 0, rhoOppSum = 0, rhoScoreSum = 0;
						double dRelMean = 0, dRelVar = 0;
						for (int64_t i = 0; i < n; i++)
							dRelMean += deltaRel[i];
						dRelMean /= n;
						for (int64_t i = 0; i < n; i++) {
							float control = fnSigmoid((rhoCarUs[i] - rhoCarOpp[i]) / RS_MAX(1e-6f, reachCfg.controlTemp));
							float scoring = fnSigmoid((rhoScore[i] - reachCfg.scoringBias) / RS_MAX(1e-6f, reachCfg.scoringTemp));
							float level = powf(control, alpha) * powf(scoring, 1 - alpha);
							float d = fnSigmoid(reachCfg.deltaControlWeight * deltaRel[i] + reachCfg.deltaScoringWeight * deltaSco[i]);
							float gRaw = RS_CLAMP(level * 2 * d, 0.f, 1.f);
							float gEff = reachCfg.gateFloor + (1 - reachCfg.gateFloor) * gRaw;
							mult[i] = 1 - beta * (1 - gEff);

							multSum += mult[i];
							levelSum += level;
							controlSum += control;
							scoringSum += scoring;
							rhoUsSum += rhoCarUs[i];
							rhoOppSum += rhoCarOpp[i];
							rhoScoreSum += rhoScore[i];
							dRelVar += (deltaRel[i] - dRelMean) * (deltaRel[i] - dRelMean);
						}

						report["Reach/Beta"] = beta;
						report["Reach/Rho Car Us Mean"] = (float)(rhoUsSum / n);
						report["Reach/Rho Car Opp Mean"] = (float)(rhoOppSum / n);
						report["Reach/Rho Score Mean"] = (float)(rhoScoreSum / n);
						report["Reach/Gate Mult Mean"] = (float)(multSum / n);
						report["Reach/Level Mean"] = (float)(levelSum / n);
						report["Reach/Control Mean"] = (float)(controlSum / n);
						report["Reach/Scoring Mean"] = (float)(scoringSum / n);
						report["Reach/Delta Control Std"] = sqrtf((float)(dRelVar / n));
						report["Reach/Acc EMA"] = reachAccEMA;
						report["Reach/Agree EMA"] = reachAgreeEMA;
						if (agreement >= 0)
							report["Reach/Touch Pred Agreement"] = agreement;
						{
							FList multSorted = mult;
							std::nth_element(multSorted.begin(), multSorted.begin() + n / 10, multSorted.end());
							report["Reach/Gate Mult P10"] = multSorted[n / 10];
						}

						if (reachCfg.gateEnabled) {
							// Positive-part apply: only the gated components' POSITIVE parts get
							// scaled (reward = total + (mult - 1)*gatedPos); everything else —
							// including the gated components' penalties — passes through in full.
							torch::Tensor tGatedPos = torch::tensor(combinedTraj.gatedPos);
							torch::Tensor tMult = torch::tensor(mult);
							tRewards = tRewards + (tMult - 1) * tGatedPos;
							report["Reach/Gated Reward Removed"] = ((1 - tMult) * tGatedPos).mean().item<float>();
							// The reward PPO actually optimizes (Average Step Reward is pre-gate)
							report["Reach/Gated Avg Step Reward"] = tRewards.mean().item<float>();
						}

						report["Reach Read Time"] = reachReadTimer.Elapsed();
					}

					// Deliberate-practice proposer: advantage-weighted-hindsight goal proposal +
					// (Stage 2, off by default) advantage-only shaping. Stage 1 is PASSIVE - this
					// block only READS tRewards/tValPreds/tTerminals (which are already gate-mutated
					// above, i.e. the SAME rewards the policy is actually trained on) and never
					// writes to them; tPropAInt (if shapingBeta > 0) is the only thing it hands to
					// GAE's advantages below, and it's centered + explicitly zeroed at terminals.
					torch::Tensor tPropAInt; // Stage 2 shaping term, added into tAdvantages after GAE
						torch::Tensor tPropCarAInt; // car shaping term (carShapingBeta), added after GAE
					if (proposerOn && combinedTraj.Length() > 0 && ppo->proposer) {
						Timer propTimer = {};
						int64_t n = (int64_t)combinedTraj.Length();

						std::vector<int64_t> epStart, epEnd;
						ProposerModule::SegmentEpisodes(combinedTraj.terminals, epStart, epEnd);

						Model* sharedHead = ppo->models["shared_head"];
						// Reuse the trunk features computed once above; only recompute in the (config
						// edge) case where no shared head exists so nothing precomputed them.
						torch::Tensor tFeatures = tTrunkFeatures.defined()
							? tTrunkFeatures
							: ppo->proposer->ComputeFeatures(sharedHead, tStates, propCfg.featureChunkSize);

						torch::Tensor tCurBall = torch::tensor(combinedTraj.proposerCurBall).reshape({ -1, 6 });
						torch::Tensor tTargetBall = torch::tensor(combinedTraj.proposerTargetBall).reshape({ -1, 6 });

						auto unroll = ppo->proposer->Unroll(tFeatures, tCurBall, epStart, epEnd);
						torch::Tensor tGoals = unroll.goals;         // [n,6] CPU
						torch::Tensor tPrevGoals = unroll.prevGoals; // [n,6] CPU

						// Stage 3: committed-goal override - a row inside an active practice window
						// uses the FIXED goal that was in play when the drill was banked, not the
						// freshly unrolled one (the point of a drill is repeated attempts at the
						// SAME target). Done before goalQueries/tGoalsShift below are built from
						// tGoals, so both the shaping term and the logging reflect the committed goal.
						if (practiceOn && !combinedTraj.practiceMask.empty()) {
							torch::Tensor tPracticeMask = torch::tensor(combinedTraj.practiceMask).to(torch::kBool);
							torch::Tensor tPracticeGoals = torch::tensor(combinedTraj.practiceGoals).reshape({ -1, 6 });
							tGoals = torch::where(tPracticeMask.unsqueeze(-1), tPracticeGoals, tGoals);
						}

						// N-step advantage (in GAE's own standardized/clipped reward units) drives
						// the CRR-binary aspiration weight: rows beating the batch's aspiration
						// percentile get weight 1, the rest get a small nonzero floor - diluted,
						// never repelled, so a costly-but-aggressive miss never gets trained AWAY
						// from, only outweighed by the better outcomes.
						torch::Tensor tAN = ProposerModule::ComputeNStepAdvantages(
							tRewards, combinedTraj.terminals, tValPreds, tTruncValPreds,
							epStart, epEnd, propCfg.horizonSteps, config.ppo.gaeGamma,
							returnStat ? returnStat->GetSTD() : 1, config.ppo.rewardClipRange);

						float aspirationThresh;
						{
							FList anSorted = TENSOR_TO_VEC<float>(tAN);
							int64_t rank = RS_CLAMP(
								(int64_t)(anSorted.size() * RS_CLAMP(propCfg.aspirationPercentile, 0.f, 1.f)),
								(int64_t)0, (int64_t)anSorted.size() - 1);
							std::nth_element(anSorted.begin(), anSorted.begin() + rank, anSorted.end());
							aspirationThresh = anSorted[rank];
						}
						torch::Tensor tWeights = torch::where(
							tAN >= aspirationThresh,
							torch::ones_like(tAN),
							torch::full_like(tAN, propCfg.belowAspirationWeight));

						// rho(s_t -> g_t); with shaping on, also rho(s_t -> g_{t-1}) via each row's
						// goal shifted one step forward within its episode, so ONE EvalRhoRowwise
						// call (shared sampled actions) answers both rho(s,g) and rho(s',g) terms.
						bool shapingOn = propCfg.shapingBeta > 0;
						std::vector<torch::Tensor> goalQueries = { tGoals };
						torch::Tensor tGoalsShift;
						if (shapingOn) {
							tGoalsShift = tGoals.clone();
							for (int64_t e = 0; e < (int64_t)epStart.size(); e++) {
								int64_t s = epStart[e], en = epEnd[e];
								if (en > s)
									tGoalsShift.slice(0, s + 1, en + 1).copy_(tGoals.slice(0, s, en));
								// Row s keeps its cloned value (tGoals[s]) - there's no g_{t-1} to
								// shift in at an episode's first row, and the corresponding rhoNext
								// read (at row s-1, a different episode or out of bounds) is zeroed
								// by the terminal mask below regardless.
							}
							goalQueries.push_back(tGoalsShift);
						}

						auto rhos = ppo->reach->EvalRhoRowwise(sharedHead, ppo->reach->psiBall, goalQueries, tStates, tActionMasks, c10::nullopt, tTrunkFeatures);
						torch::Tensor tRhoGoal = rhos[0]; // rho(s_t, g_t)

						if (shapingOn) {
							torch::Tensor tRhoGoalPrev = rhos[1]; // rho(s_t, g_{t-1})
							// rhoNext[t] = rho(s_{t+1}, g_t) = tRhoGoalPrev[t+1] (tGoalsShift[t+1] == tGoals[t])
							torch::Tensor tRhoNext = torch::zeros_like(tRhoGoal);
							if (n > 1)
								tRhoNext.slice(0, 0, n - 1).copy_(tRhoGoalPrev.slice(0, 1, n));

							// gamma*rho(s',g) - rho(s,g), using the SAME g on both sides: the
							// goal-motion component is excluded BY CONSTRUCTION (no difference is
							// ever taken across the goal update), so this only ever charges the
							// policy for progress toward a goal that stood.
							torch::Tensor tAIntRaw = config.ppo.gaeGamma * tRhoNext - tRhoGoal;

							// Zero at terminal rows: no real s_{t+1} to credit, and it happens to
							// also absorb the cross-episode boundary noise in tRhoNext above
							torch::Tensor tTerminalMask = (tTerminals == 0).to(torch::kFloat32);
							tPropAInt = tAIntRaw * tTerminalMask;

							// Stage 3: amplify shaping on practice-tagged rows (repeated at-bats
							// should count for more while the policy is drilling a specific miss)
							if (practiceOn && !combinedTraj.practiceMask.empty()) {
								FList betaScaleVec(n, 1.f);
								for (int64_t t = 0; t < n; t++)
									if (combinedTraj.practiceMask[t])
										betaScaleVec[t] = propCfg.practiceBetaScale;
								tPropAInt = tPropAInt * torch::tensor(betaScaleVec);
							}
						}

						// Stage 3: Phi-drop detection (banks new drills from non-practice
						// near-misses) + drill success evaluation (retires solved practice
						// windows). Both read tRhoGoal, which already reflects any committed-goal
						// override above.
						if (practiceOn && n > 0) {
							torch::Tensor tPhi = torch::sigmoid(tRhoGoal / RS_MAX(1e-3f, propCfg.phiSquashTemp));
							FList phi = TENSOR_TO_VEC<float>(tPhi);
							int numArenas = (int)envSet->arenas.size();
							int dropWindow = RS_MAX(1, propCfg.phiDropWindow);
							int snapK = RS_MAX(1, propCfg.snapshotEveryK);

							// Detector thresholds: self-calibrated from THIS batch's non-practice Phi
							// distribution (default; tracks the drifting rho scale so the detector
							// neither floods nor starves) or the fixed absolutes as a fallback.
							float phiHigh = propCfg.phiHighThresh;
							float phiDropMag = propCfg.phiDropThresh;
							if (propCfg.phiCalibratePerIter) {
								FList phiSorted;
								phiSorted.reserve(n);
								for (int64_t t = 0; t < n; t++)
									if (combinedTraj.practiceMask.empty() || !combinedTraj.practiceMask[t])
										phiSorted.push_back(phi[t]);
								if (phiSorted.size() >= 8) {
									auto pctl = [&](float p) {
										int64_t k = RS_CLAMP((int64_t)(phiSorted.size() * RS_CLAMP(p, 0.f, 1.f)),
											(int64_t)0, (int64_t)phiSorted.size() - 1);
										std::nth_element(phiSorted.begin(), phiSorted.begin() + k, phiSorted.end());
										return phiSorted[k];
									};
									float pHigh = pctl(propCfg.phiHighPercentile);
									float pLow = pctl(propCfg.phiDropPercentile);
									phiHigh = pHigh;
									phiDropMag = RS_MAX(1e-4f, pHigh - pLow);
								}
							}
							report["Proposer/Phi High Thresh"] = phiHigh;
							report["Proposer/Phi Drop Mag"] = phiDropMag;

							// Collect ALL drop candidates first, then bank the LARGEST-drop
							// maxNewDrillsPerItr - "the worst misses this iteration", not the
							// first-in-scan-order (which biased toward early-in-the-batch episodes).
							struct DrillCand { float drop; int64_t highRow; int64_t epStartRow; };
							std::vector<DrillCand> cands;
							for (int64_t e = 0; e < (int64_t)epStart.size(); e++) {
								int64_t s = epStart[e], en = epEnd[e];
								float runningHigh = phi[s];
								int64_t highRow = s;
								for (int64_t t = s + 1; t <= en; t++) {
									if (!combinedTraj.practiceMask.empty() && combinedTraj.practiceMask[t])
										continue; // mid-drill mistakes aren't fresh near-misses to bank

									if (phi[t] > runningHigh) {
										runningHigh = phi[t];
										highRow = t;
										continue;
									}

									float drop = runningHigh - phi[t];
									if (runningHigh >= phiHigh && drop >= phiDropMag && (t - highRow) <= dropWindow) {
										cands.push_back({ drop, highRow, s });
										// Don't re-trigger repeatedly on the same decline
										runningHigh = phi[t];
										highRow = t;
									}
								}
							}

							std::sort(cands.begin(), cands.end(),
								[](const DrillCand& a, const DrillCand& b) { return a.drop > b.drop; });
							int newDrills = 0;
							for (const DrillCand& c : cands) {
								if (newDrills >= propCfg.maxNewDrillsPerItr)
									break;
								int player = combinedTraj.srcPlayer[c.highRow];
								int localStep = combinedTraj.srcStep[c.highRow];
								int snapStep = (localStep / snapK) * snapK;
									// snapStep rounds the near-miss step DOWN to the nearest snapshot
									// multiple. Episode resets land at arbitrary steps (not multiples),
									// so for a peak early in its episode snapStep can fall before the
									// episode began - pointing at a stale prior-episode snapshot of the
									// same arena. Bank only when snapStep is still inside this episode;
									// otherwise no valid in-episode snapshot exists at/before the peak.
									int epStartStep = combinedTraj.srcStep[c.epStartRow];
									if (snapStep < epStartStep)
										continue;
								int arenaIdx = playerArenaIdx[player];

								auto it = stepSnapshots.find((int64_t)snapStep * numArenas + arenaIdx);
								if (it != stepSnapshots.end()) {
									FList goalVec = TENSOR_TO_VEC<float>(tGoals[c.highRow]);
									float goal6[6];
									for (int d = 0; d < 6; d++)
										goal6[d] = goalVec[d];
									Team sourceTeam = envSet->state.gameStates[arenaIdx].players[playerSlotIdx[player]].team;
									propCfg.drillBank->AddDrill(it->second, goal6, sourceTeam, c.drop);
									newDrills++;
								}
							}
							report["Proposer/Drills Added"] = (float)newDrills;
							report["Proposer/Drill Candidates"] = (float)cands.size();

							// Drill success: a practice-tagged drillId succeeds if Phi recovered to
							// the high threshold anywhere within its tagged rows
							if (!combinedTraj.practiceMask.empty()) {
								std::unordered_map<int64_t, bool> drillSuccess;
								for (int64_t t = 0; t < n; t++) {
									if (!combinedTraj.practiceMask[t])
										continue;
									int64_t drillId = combinedTraj.drillIds[t];
									bool success = phi[t] >= phiHigh;
									auto res = drillSuccess.emplace(drillId, success);
									if (!res.second && success)
										res.first->second = true;
								}
								for (auto& pair : drillSuccess)
									propCfg.drillBank->ReportResult((uint64_t)pair.first, pair.second);

								int64_t practiceCount = 0;
								for (uint8_t m : combinedTraj.practiceMask)
									practiceCount += m;
								report["Proposer/Practice Step Fraction"] = (float)practiceCount / (float)RS_MAX((int64_t)1, n);
							}

							report["Proposer/Drill Bank Size"] = (float)propCfg.drillBank->Size();
							report["Proposer/Drill Success Rate"] = propCfg.drillBank->AvgSuccessRate();

							// JSONL dump of the bank's current contents for offline eyeballing -
							// the "are these real near-misses (aerial whiffs, blown saves) or junk
							// (kickoff chaos, opponent bounces)?" check before replay is ever enabled.
							if (propCfg.drillDumpEveryNItrs > 0 && !config.checkpointFolder.empty() &&
								(totalIterations % propCfg.drillDumpEveryNItrs == 0)) {
								auto dumpRows = propCfg.drillBank->SnapshotForDump(propCfg.drillDumpMaxRows);
								if (!dumpRows.empty()) {
									std::filesystem::path dumpDir = config.checkpointFolder / "drill_dumps";
									std::filesystem::create_directories(dumpDir);
									std::filesystem::path dumpPath = dumpDir / ("itr_" + std::to_string(totalIterations) + ".jsonl");
									std::ofstream dumpOut(dumpPath);
									if (dumpOut.good()) {
										for (auto& dr : dumpRows) {
											using namespace nlohmann;
											json j = {};
											j["itr"] = totalIterations;
											j["id"] = dr.id;
											j["ball_pos"] = { dr.ballPos[0], dr.ballPos[1], dr.ballPos[2] };
											j["ball_vel"] = { dr.ballVel[0], dr.ballVel[1], dr.ballVel[2] };
											j["goal"] = std::vector<float>(dr.goal, dr.goal + 6);
											j["team"] = dr.sourceTeam;
											j["drop"] = dr.drop;
											j["tries"] = dr.tries;
											j["successes"] = dr.successes;
											dumpOut << j.dump() << "\n";
										}
									}
								}
							}
						}

						// Hand off to the deferred (gradient-requiring) Train() call after this
						// no-grad block closes
						tPropFeatures = tFeatures;
						tPropPrevGoals = tPrevGoals;
						tPropTargets = tTargetBall;
						tPropWeights = tWeights;

						report["Proposer/Aspiration Threshold"] = aspirationThresh;
						report["Proposer/AN Mean"] = tAN.mean().item<float>();
						report["Proposer/AN Std"] = tAN.std().item<float>();
						report["Proposer/Weight Fraction"] = (tWeights >= 1.f).to(torch::kFloat32).mean().item<float>();
						report["Proposer/Goal Target Dist"] = (tGoals - tTargetBall).norm(2, -1).mean().item<float>();
						report["Proposer/Goal Drift"] = unroll.meanDeltaNorm;
						report["Proposer/Rho Goal Mean"] = tRhoGoal.mean().item<float>();
						if (shapingOn) {
							report["Proposer/A Int Mean"] = tPropAInt.mean().item<float>();
							report["Proposer/A Int Std"] = tPropAInt.std().item<float>();
						}

						// JSONL calibration dump (local disk only - metrics are scalars-only to
						// wandb): sampled (current, proposed goal, hindsight target, A^(N), weight,
						// rho) rows, so proposals can be sanity-checked (e.g. against a known setup
						// like a backboard save) without any live-run risk.
						if (propCfg.dumpEveryNItrs > 0 && !config.checkpointFolder.empty() &&
							(totalIterations % propCfg.dumpEveryNItrs == 0) && n > 0) {

							std::filesystem::path dumpDir = config.checkpointFolder / "proposer_dumps";
							std::filesystem::create_directories(dumpDir);
							std::filesystem::path dumpPath = dumpDir / ("itr_" + std::to_string(totalIterations) + ".jsonl");
							std::ofstream dumpOut(dumpPath);
							if (dumpOut.good()) {
								int64_t numRows = RS_MIN((int64_t)RS_MAX(0, propCfg.dumpMaxRows), n);
								torch::Tensor sampleIdx = torch::randperm(n, torch::TensorOptions().dtype(torch::kLong)).slice(0, 0, numRows);
								auto _sampleIdx = sampleIdx.const_data_ptr<int64_t>();
								for (int64_t i = 0; i < numRows; i++) {
									int64_t row = _sampleIdx[i];
									using namespace nlohmann;
									json j = {};
									j["itr"] = totalIterations;
									j["ts"] = totalTimesteps;
									j["row"] = row;
									j["cur"] = std::vector<float>(
										combinedTraj.proposerCurBall.begin() + row * 6, combinedTraj.proposerCurBall.begin() + row * 6 + 6);
									j["goal"] = TENSOR_TO_VEC<float>(tGoals[row]);
									j["ach"] = std::vector<float>(
										combinedTraj.proposerTargetBall.begin() + row * 6, combinedTraj.proposerTargetBall.begin() + row * 6 + 6);
									j["aN"] = tAN[row].item<float>();
									j["w"] = tWeights[row].item<float>();
									j["rho"] = tRhoGoal[row].item<float>();
									j["practice"] = 0;
									dumpOut << j.dump() << "\n";
								}
							}
						}

						// ---- Car proposer head: second unroll on canonical car-state goals, reusing
						// the SAME trunk features + A^(N) aspiration weights (aspiration is goal-space
						// agnostic). Passive unless carShapingBeta > 0. rho_car uses the psiCarState
						// head. Mirrors the ball block above.
						if (proposerCarOn && ppo->proposerCar) {
							torch::Tensor tCarCur = torch::tensor(combinedTraj.proposerCarCur).reshape({ -1, 6 });
							torch::Tensor tCarTarget = torch::tensor(combinedTraj.proposerCarTarget).reshape({ -1, 6 });

							auto carUnroll = ppo->proposerCar->Unroll(tFeatures, tCarCur, epStart, epEnd);
							torch::Tensor tCarGoals = carUnroll.goals;
							torch::Tensor tCarPrevGoals = carUnroll.prevGoals;

							bool carShapingOn = propCfg.carShapingBeta > 0 && ppo->reach->psiCarState;
							torch::Tensor tRhoCarGoal;
							if (carShapingOn) {
								std::vector<torch::Tensor> carQ = { tCarGoals };
								torch::Tensor tCarGoalsShift = tCarGoals.clone();
								for (int64_t e = 0; e < (int64_t)epStart.size(); e++) {
									int64_t s = epStart[e], en = epEnd[e];
									if (en > s)
										tCarGoalsShift.slice(0, s + 1, en + 1).copy_(tCarGoals.slice(0, s, en));
								}
								carQ.push_back(tCarGoalsShift);
								auto carRhos = ppo->reach->EvalRhoRowwise(sharedHead, ppo->reach->psiCarState, carQ, tStates, tActionMasks, c10::nullopt, tTrunkFeatures);
								tRhoCarGoal = carRhos[0];
								torch::Tensor tRhoCarNext = torch::zeros_like(tRhoCarGoal);
								if (n > 1)
									tRhoCarNext.slice(0, 0, n - 1).copy_(carRhos[1].slice(0, 1, n));
								torch::Tensor tCarAIntRaw = config.ppo.gaeGamma * tRhoCarNext - tRhoCarGoal;
								torch::Tensor tTerminalMask = (tTerminals == 0).to(torch::kFloat32);
								tPropCarAInt = tCarAIntRaw * tTerminalMask;
							}

							tPropCarPrevGoals = tCarPrevGoals;
							tPropCarTargets = tCarTarget;

							report["Proposer/Car Goal Target Dist"] = (tCarGoals - tCarTarget).norm(2, -1).mean().item<float>();
							report["Proposer/Car Goal Drift"] = carUnroll.meanDeltaNorm;
							if (carShapingOn)
								report["Proposer/Car Rho Goal Mean"] = tRhoCarGoal.mean().item<float>();

							// Car JSONL dump for the tilt/separability report (same cadence as ball)
							if (propCfg.dumpEveryNItrs > 0 && !config.checkpointFolder.empty() &&
								(totalIterations % propCfg.dumpEveryNItrs == 0) && n > 0) {
								std::filesystem::path dumpDir = config.checkpointFolder / "proposer_car_dumps";
								std::filesystem::create_directories(dumpDir);
								std::filesystem::path dumpPath = dumpDir / ("itr_" + std::to_string(totalIterations) + ".jsonl");
								std::ofstream dumpOut(dumpPath);
								if (dumpOut.good()) {
									int64_t numRows = RS_MIN((int64_t)RS_MAX(0, propCfg.dumpMaxRows), n);
									torch::Tensor sampleIdx = torch::randperm(n, torch::TensorOptions().dtype(torch::kLong)).slice(0, 0, numRows);
									auto _sampleIdx = sampleIdx.const_data_ptr<int64_t>();
									for (int64_t i = 0; i < numRows; i++) {
										int64_t row = _sampleIdx[i];
										using namespace nlohmann;
										json j = {};
										j["itr"] = totalIterations;
										j["row"] = row;
										j["cur"] = std::vector<float>(
											combinedTraj.proposerCarCur.begin() + row * 6, combinedTraj.proposerCarCur.begin() + row * 6 + 6);
										j["goal"] = TENSOR_TO_VEC<float>(tCarGoals[row]);
										j["ach"] = std::vector<float>(
											combinedTraj.proposerCarTarget.begin() + row * 6, combinedTraj.proposerCarTarget.begin() + row * 6 + 6);
										// ball pos at this row (for the distance-to-ball separability bucket)
										j["ball"] = std::vector<float>(
											combinedTraj.proposerCurBall.begin() + row * 6, combinedTraj.proposerCurBall.begin() + row * 6 + 6);
										j["aN"] = tAN[row].item<float>();
										j["w"] = tWeights[row].item<float>();
										dumpOut << j.dump() << "\n";
									}
								}
							}
						}

						report["Proposer/Time"] = propTimer.Elapsed();
					}

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

					// --- Secondary goal-only critic: its own GAE pass at the long-horizon gamma, then
					//     std-matched advantage blend. The goal channel is ±1 at goal terminals and 0
					//     elsewhere, so returnStd=0/clip=0 (no standardization — it's already bounded).
					torch::Tensor tGoalTargetVals;
					if (goalCriticOn) {
						torch::Tensor tGoalRews = torch::tensor(combinedTraj.goalRews);
						RG_ASSERT(tGoalRews.size(0) == (int64_t)combinedTraj.Length());

						torch::Tensor tGoalAdvantages, tGoalReturns;
						float goalClipPortion;
						GAE::Compute(
							tGoalRews, tTerminals, tGoalValPreds, tGoalTruncValPreds,
							tGoalAdvantages, tGoalTargetVals, tGoalReturns, goalClipPortion,
							config.ppo.goalCritic.gamma, config.ppo.gaeLambda, /*returnStd=*/0, /*clipRange=*/0
						);

						// VALIDATION METRICS (the panels that prove correctness):
						// V_goal and A_goal must correlate POSITIVELY with realized outcomes over rows
						// whose episode actually ended in a goal (sign of the goal-channel MC return).
						// A channel or sign bug reads NEGATIVE here within minutes on a goal-dense run.
						torch::Tensor outcomeMask = tGoalReturns.abs() > 1e-6f;
						long nOutcome = outcomeMask.sum().item<long>();
						report["GoalCritic/Outcome Rows Frac"] = (float)nOutcome / RS_MAX(1, (int)combinedTraj.Length());
						if (nOutcome > 100) {
							auto fnCorr = [&](const torch::Tensor& a, const torch::Tensor& b) {
								auto ac = a - a.mean(), bc = b - b.mean();
								return ((ac * bc).mean() /
									(a.std(false) * b.std(false) + 1e-8f)).item<float>();
							};
							torch::Tensor outcome = tGoalReturns.sign().masked_select(outcomeMask);
							report["GoalCritic/Value-Outcome Corr"] = fnCorr(tGoalValPreds.masked_select(outcomeMask), outcome);
							report["GoalCritic/Adv-Outcome Corr"] = fnCorr(tGoalAdvantages.masked_select(outcomeMask), outcome);
						}
						report["GoalCritic/Mean Val"] = tGoalValPreds.mean().item<float>();
						report["GoalCritic/Val Abs Mean"] = tGoalValPreds.abs().mean().item<float>();

						// Std-matched blend: beta is the FRACTION of dense-advantage scale contributed.
						// Centered so the blend shifts relative preferences, never the average gradient.
						float advStd = tAdvantages.std().item<float>();
						float goalAdvStd = tGoalAdvantages.std().item<float>();
						if (config.ppo.goalCritic.beta > 0 && goalAdvStd > 1e-8f && advStd > 1e-8f) {
							float betaEff = config.ppo.goalCritic.beta * advStd / goalAdvStd;
							torch::Tensor injected = betaEff * (tGoalAdvantages - tGoalAdvantages.mean());
							tAdvantages = tAdvantages + injected;
							report["GoalCritic/Blend BetaEff"] = betaEff;
							report["GoalCritic/Injected Abs Mean"] = injected.abs().mean().item<float>();
						}
					}

					// Stage 2 (deliberate-practice shaping): added AFTER GAE has already derived
					// tTargetVals from the UNshaped advantages (GAE.cpp: targetValues = valPreds +
					// advantages) - so the critic's targets never see this term, only the policy's
					// advantages do. Centered (pushes toward above-average-progress goals, not just
					// "more"), scaled so its std matches shapingBeta fraction of the extrinsic std,
					// then added in. shapingBeta == 0 (the default) makes this exactly a no-op.
					if (tPropAInt.defined() && config.ppo.proposer.shapingBeta > 0) {
						tPropAInt = tPropAInt - tPropAInt.mean();
						float stdExt = tAdvantages.std().item<float>();
						float stdInt = RS_MAX(1e-6f, tPropAInt.std().item<float>());
						float betaEff = config.ppo.proposer.shapingBeta * stdExt / stdInt;
						tAdvantages = tAdvantages + betaEff * tPropAInt;
						report["Proposer/Shaping BetaEff"] = betaEff;
						report["Proposer/Shaping Injected Abs Mean"] = (betaEff * tPropAInt).abs().mean().item<float>();
					}

					// Car shaping term: identical centered/std-matched injection, own beta. stdExt
					// is recomputed against the (possibly ball-shaped) advantages so the two terms
					// compose to roughly shapingBeta + carShapingBeta of the extrinsic std.
					if (tPropCarAInt.defined() && config.ppo.proposer.carShapingBeta > 0) {
						tPropCarAInt = tPropCarAInt - tPropCarAInt.mean();
						float stdExt = tAdvantages.std().item<float>();
						float stdInt = RS_MAX(1e-6f, tPropCarAInt.std().item<float>());
						float betaEff = config.ppo.proposer.carShapingBeta * stdExt / stdInt;
						tAdvantages = tAdvantages + betaEff * tPropCarAInt;
						report["Proposer/Car Shaping BetaEff"] = betaEff;
						report["Proposer/Car Shaping Injected Abs Mean"] = (betaEff * tPropCarAInt).abs().mean().item<float>();
					}

					// Set experience buffer
					experience.data.actions = tActions;
					experience.data.logProbs = tLogProbs;
					experience.data.actionMasks = tActionMasks;
					experience.data.states = tStates;
					experience.data.advantages = tAdvantages;
					experience.data.targetValues = tTargetVals;
					if (goalCriticOn)
						experience.data.goalTargetValues = tGoalTargetVals;

					if (reachOn) {
						experience.data.carHerGoals = torch::tensor(combinedTraj.carHerGoals).reshape({ -1, 6 });
						experience.data.ballHerGoals = torch::tensor(combinedTraj.ballHerGoals).reshape({ -1, 6 });
						experience.data.ballMovedMask = torch::tensor(combinedTraj.ballMoved);
						if (proposerCarOn && !combinedTraj.carStateHerGoals.empty())
							experience.data.carStateHerGoals = torch::tensor(combinedTraj.carStateHerGoals).reshape({ -1, 6 });
					}
				}

				// Free CUDA cache
#ifdef RG_CUDA_SUPPORT
				if (ppo->device.is_cuda())
					c10::cuda::CUDACachingAllocator::emptyCache();
#endif

				// Deliberate-practice proposer training: deferred until here (outside the
				// RG_NO_GRAD "Process timesteps" block above, same reason ppo->Learn() itself is
				// deferred) so Delta's own backward pass can actually build a graph. Uses the
				// pre-update goals/rho computed above for this iteration's shaping/logging, then
				// updates - so what got logged/shaped this iteration reflects the OLD proposer.
				if (proposerOn && ppo->proposer && tPropFeatures.defined()) {
					Timer propTrainTimer = {};
					auto propTrainResult = ppo->proposer->Train(tPropFeatures, tPropPrevGoals, tPropTargets, tPropWeights);
					report["Proposer/Loss"] = propTrainResult.loss;
					report["Proposer/Train Time"] = propTrainTimer.Elapsed();

					// Car head trains on the SAME features + weights, its own regression pair
					if (proposerCarOn && ppo->proposerCar && tPropCarTargets.defined()) {
						auto carTrainResult = ppo->proposerCar->Train(tPropFeatures, tPropCarPrevGoals, tPropCarTargets, tPropWeights);
						report["Proposer/Car Loss"] = carTrainResult.loss;
					}
				}

				// Learn
				Timer learnTimer = {};
				ppo->Learn(experience, report, isFirstIteration);
				report["PPO Learn Time"] = learnTimer.Elapsed();

				// Set metrics
				float consumptionTime = consumptionTimer.Elapsed();
				report["Collection Time"] = collectionTime;
				report["Consumption Time"] = consumptionTime;
				report["Collection Steps/Second"] = stepsCollected / collectionTime;
				report["Consumption Steps/Second"] = stepsCollected / consumptionTime;
				report["Overall Steps/Second"] = stepsCollected / (collectionTime + consumptionTime);

				uint64_t prevTimesteps = totalTimesteps;
				totalTimesteps += stepsCollected;
				report["Total Timesteps"] = totalTimesteps;
				totalIterations++;
				report["Total Iterations"] = totalIterations;

				if (versionMgr)
					versionMgr->OnIteration(ppo, report, totalTimesteps, prevTimesteps);

				// QD league: evolve/evaluate members between iterations (additive, off by default).
				if (league)
					league->OnIteration(report, totalIterations);

				// Basin-Racing (PSD): may run a full probe round in-line. If it does, the arenas
				// were stepped out from under the in-flight trajectories, so clear them — the next
				// iteration starts fresh episodes (a probe boundary is like a checkpoint boundary).
				if (psd) {
					float rating = report.Has("Rating/1v1") ? (float)report["Rating/1v1"] : NAN;
					bool probed = psd->OnDescendIteration(report, rating, totalIterations);
					if (probed) {
						// Clear the persistent per-player trajectories so post-probe collection
						// starts fresh episodes (combinedTraj is rebuilt from these each iteration).
						for (auto& traj : trajectories)
							traj.Clear();
					}
				}

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
						"Reach/Beta",
						"Reach/Gate Mult Mean",
						"Reach/Gate Mult P10",
						"Reach/Delta Control Std",
						"Reach/Car Accuracy",
						"Reach/Ball Accuracy",
						"Reach/Touch Pred Agreement",
						"",
						"Proposer/Loss",
						"Proposer/Rho Goal Mean",
						"Proposer/Weight Fraction",
						"Proposer/Shaping BetaEff",
						"Proposer/Drill Bank Size",
						"Proposer/Drills Added",
						"Proposer/Practice Step Fraction",
						"Proposer/Car Loss",
						"Proposer/Car Goal Drift",
						"Proposer/Car Shaping BetaEff",
						"",
						"Policy Update Magnitude",
						"Critic Update Magnitude",
						"Shared Head Update Magnitude",
						"",
						"GoalCritic/Loss",
						"GoalCritic/Value-Outcome Corr",
						"GoalCritic/Adv-Outcome Corr",
						"GoalCritic/Outcome Rows Frac",
						"GoalCritic/Mean Val",
						"GoalCritic/Val Abs Mean",
						"GoalCritic/Blend BetaEff",
						"GoalCritic/Injected Abs Mean",
						"",
						"Collection Steps/Second",
						"Consumption Steps/Second",
						"Overall Steps/Second",
						"",
						"Collection Time",
						"-Inference Time",
						"-Env Step Time",
						"-Prep Time",
						"-Record Time",
						"Consumption Time",
						"-Value Pred Time",
						"-Reach Read Time",
						"-GAE Time",
						"-PPO Learn Time"
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
	delete psd;
	delete league;
	delete ppo;
	delete versionMgr;
	delete metricSender;
	delete renderSender;
	pybind11::finalize_interpreter();
}