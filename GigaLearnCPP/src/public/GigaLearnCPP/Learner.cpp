#include "Learner.h"

#include <GigaLearnCPP/PPO/PPOLearner.h>
#include <GigaLearnCPP/PPO/ExperienceBuffer.h>

#include <torch/cuda.h>
#include <nlohmann/json.hpp>
#include <pybind11/embed.h>

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
	if (gcrlAdvScaleAnnealStartTS != UINT64_MAX)
		j["gcrl_adv_scale_anneal_start_ts"] = gcrlAdvScaleAnnealStartTS;
	if (sorsRewardScaleAnnealStartTS != UINT64_MAX)
		j["sors_reward_scale_anneal_start_ts"] = sorsRewardScaleAnnealStartTS;

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
	if (j.contains("gcrl_adv_scale_anneal_start_ts"))
		gcrlAdvScaleAnnealStartTS = j["gcrl_adv_scale_anneal_start_ts"];
	if (j.contains("sors_reward_scale_anneal_start_ts"))
		sorsRewardScaleAnnealStartTS = j["sors_reward_scale_anneal_start_ts"];

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

void GGL::Learner::StartQuitKeyThread(bool& quitPressed, std::thread& outThread) {
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
		bool saveQueued;
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

	try {
		bool saveQueued;
		std::thread keyPressThread;
		StartQuitKeyThread(saveQueued, keyPressThread);

		ExperienceBuffer experience = ExperienceBuffer(config.randomSeed, torch::kCPU);

		int numPlayers = envSet->state.numPlayers;
		bool useActionComps = config.ppo.useGCRL || config.ppo.useSORS;

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
			FList states, nextStates, rewards, logProbs;
			// GCRL: 8-dim action components (per step) and hindsight-relabeled future
			// ball goals (global / car-local), filled at episode end.
			FList actionComps, futureGoals, carFutureGoals;
			std::vector<SORSStep> sorsSteps;
			std::vector<uint8_t> actionMasks;
			std::vector<int8_t> terminals;
			std::vector<int32_t> actions;

			void Clear() {
				states.clear();
				nextStates.clear();
				rewards.clear();
				logProbs.clear();
				actionComps.clear();
				futureGoals.clear();
				carFutureGoals.clear();
				sorsSteps.clear();
				actionMasks.clear();
				terminals.clear();
				actions.clear();
			}

			void Append(const Trajectory& other) {
				states += other.states;
				nextStates += other.nextStates;
				rewards += other.rewards;
				logProbs += other.logProbs;
				actionComps += other.actionComps;
				futureGoals += other.futureGoals;
				carFutureGoals += other.carFutureGoals;
				sorsSteps += other.sorsSteps;
				actionMasks += other.actionMasks;
				terminals += other.terminals;
				actions += other.actions;
			}

			size_t Length() const {
				return actions.size();
			}
		};

		auto trajectories = std::vector<Trajectory>(numPlayers, Trajectory{});
		int maxEpisodeLength = (int)(config.ppo.maxEpisodeDuration * (120.f / config.tickSkip));

		// Pre-reserve to avoid repeated realloc during episodes.
		for (auto& traj : trajectories) {
			traj.states.reserve((size_t)maxEpisodeLength * obsSize);
			traj.rewards.reserve(maxEpisodeLength);
			traj.logProbs.reserve(maxEpisodeLength);
			traj.actions.reserve(maxEpisodeLength);
			traj.terminals.reserve(maxEpisodeLength);
			traj.actionMasks.reserve((size_t)maxEpisodeLength * numActions);
			if (useActionComps) {
				traj.actionComps.reserve((size_t)maxEpisodeLength * 8);
			}
			if (config.ppo.useSORS)
				traj.sorsSteps.reserve(maxEpisodeLength);
		}

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
				window.states.insert(window.states.end(),
					traj.states.begin() + (size_t)start * obsSize,
					traj.states.begin() + (size_t)stop * obsSize);
				window.actionComps.insert(window.actionComps.end(),
					traj.actionComps.begin() + (size_t)start * 8,
					traj.actionComps.begin() + (size_t)stop * 8);
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

		while (true) {
			Report report = {};

			bool isFirstIteration = (totalTimesteps == 0);

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

			ppo->curGCRLAdvScale = fnGetAnnealedScale(
				config.ppo.gcrlAdvScale,
				config.ppo.gcrlAdvScaleAnnealStart,
				config.ppo.gcrlAdvScaleAnnealSteps,
				gcrlAdvScaleAnnealStartTS
			);
			ppo->curSORSRewardScale = fnGetAnnealedScale(
				config.ppo.sorsRewardScale,
				config.ppo.sorsRewardScaleAnnealStart,
				config.ppo.sorsRewardScaleAnnealSteps,
				sorsRewardScaleAnnealStartTS
			);
			report["GCRL/Adv Scale"] = ppo->curGCRLAdvScale;
			report["SORS/Reward Scale"] = ppo->curSORSRewardScale;

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

			int numRealPlayers = oldVersion ? newPlayerIndices.size() : envSet->state.numPlayers;

			int stepsCollected = 0;
			{ // Generate experience

				// Only contains complete episodes
				auto combinedTraj = Trajectory();
				{
					// Pre-reserve combined trajectory to avoid realloc as episodes accumulate.
					size_t expTs = config.ppo.tsPerItr + config.ppo.tsPerItr / 4;
					combinedTraj.states.reserve(expTs * obsSize);
					combinedTraj.rewards.reserve(expTs);
					combinedTraj.logProbs.reserve(expTs);
					combinedTraj.actions.reserve(expTs);
					combinedTraj.terminals.reserve(expTs);
					combinedTraj.actionMasks.reserve(expTs * numActions);
					if (useActionComps) {
						combinedTraj.actionComps.reserve(expTs * 8);
					}
					if (config.ppo.useGCRL) {
						combinedTraj.futureGoals.reserve(expTs * 6);
						combinedTraj.carFutureGoals.reserve(expTs * 6);
					}
					if (config.ppo.useSORS)
						combinedTraj.sorsSteps.reserve(expTs);
				}

				Timer collectionTimer = {};
				{ // Collect timesteps
					RG_NO_GRAD;

					float inferTime = 0;
					float envStepTime = 0;

					// Scratch buffers declared outside the loop to avoid per-step allocation.
					std::vector<float> _normOffset(obsSize, 0.0f), _normInvStd(obsSize, 1.0f);
					FList _curActionComps;
					if (useActionComps)
						_curActionComps.resize(numPlayers * 8);

					for (int step = 0; combinedTraj.Length() < config.ppo.tsPerItr || render; step++, stepsCollected += numRealPlayers) {
						Timer stepTimer = {};
						envSet->Reset();
						envStepTime += stepTimer.Elapsed();

#ifndef NDEBUG
						for (float f : envSet->state.obs.data)
							if (isnan(f) || isinf(f))
								RG_ERR_CLOSE("Obs builder produced a NaN/inf value");
#endif

						if (!render && obsStat) {
							// TODO: This samples from old versions too
							int numSamples = RS_MAX(envSet->state.numPlayers, config.maxObsSamples);
							for (int i = 0; i < numSamples; i++) {
								int idx = Math::RandInt(0, envSet->state.numPlayers);
								obsStat->IncrementRow(&envSet->state.obs.At(idx, 0));
							}

							// Precompute per-dim float scale/offset once (not per player).
							// Turns inner-loop division into multiplication and avoids double→float
							// conversion inside the hot path.
							const auto& meanVec = obsStat->GetMean();
							auto stdVec = obsStat->GetSTD();
							for (int j = 0; j < obsSize; j++) {
								_normOffset[j] = RS_CLAMP((float)meanVec[j], -config.maxObsMeanRange, config.maxObsMeanRange);
								_normInvStd[j] = 1.0f / RS_MAX((float)stdVec[j], config.minObsSTD);
							}
							float* obsPtr = envSet->state.obs.data.data();
							for (int i = 0; i < envSet->state.numPlayers; i++) {
								float* row = obsPtr + i * obsSize;
								for (int j = 0; j < obsSize; j++)
									row[j] = (row[j] - _normOffset[j]) * _normInvStd[j];
							}
						}

						torch::Tensor tActions, tLogProbs;
						torch::Tensor tStates = DIMLIST2_TO_TENSOR<float>(envSet->state.obs);
						torch::Tensor tActionMasks = DIMLIST2_TO_TENSOR<uint8_t>(envSet->state.actionMasks);

						if (!render) {
							for (int newPlayerIdx : newPlayerIndices) {
								envSet->state.obs.AppendRow(newPlayerIdx, trajectories[newPlayerIdx].states);
								envSet->state.actionMasks.AppendRow(newPlayerIdx, trajectories[newPlayerIdx].actionMasks);
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

						// Decode flat action indices -> 8-dim continuous components for auxiliary critics/reward models.
						if (useActionComps) {
							auto* actionParser = envSet->actionParsers[0];
							for (int p = 0; p < (int)curActions.size(); p++) {
								Action act = actionParser->ParseAction(curActions[p], envSet->state.gameStates[0].players[0], envSet->state.gameStates[0]);
								for (int d = 0; d < 8; d++)
									_curActionComps[p * 8 + d] = act[d];
							}
						}
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
								auto& prevRewards = envSet->state.lastRewards[i];

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
									// Truncation requires an additional next state for the critic
									envSet->state.obs.AppendRow(newPlayerIdx, traj.nextStates);
								}

								// ── GCRL hindsight relabeling: per timestep pick a future ball state as ──
								// the goal. goal/anti critics use the global ball (obs[0:6]); the car
								// critic uses the car-local ball (obs[69:75]).
								if (config.ppo.useGCRL) {
									int N = traj.Length();
									int H = config.ppo.gcrlHorizon;
									int minH = config.ppo.gcrlMinHorizon;
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
										for (int d = 0; d < 6; d++)
											traj.futureGoals[t * 6 + d] = traj.states[t_target * obsSize + d];
										for (int d = 0; d < 6; d++)
											traj.carFutureGoals[t * 6 + d] = traj.states[t_target * obsSize + 69 + d];
									}
								}

								if (config.ppo.useSORS)
									ppo->AddSORSWindows(fnBuildSORSWindows(traj));

								combinedTraj.Append(traj);
								traj.Clear();
							}
						}
					}

					report["Inference Time"] = inferTime;
					report["Env Step Time"] = envStepTime;
				}
				float collectionTime = collectionTimer.Elapsed();

				if (config.ppo.useSORS)
					ppo->TrainSORS(report);

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

					if (config.ppo.useSORS && !combinedTraj.actionComps.empty()) {
						torch::Tensor tActionComps = torch::tensor(combinedTraj.actionComps).reshape({ -1, 8 });
						torch::Tensor tSORSRewards;
						if (ppo->device.is_cpu()) {
							tSORSRewards = ppo->InferSORSRewards(tStates.to(ppo->device, true, true), tActionComps.to(ppo->device, true, true)).cpu();
						} else {
							tSORSRewards = torch::zeros({ (int64_t)combinedTraj.Length() });
							for (int i = 0; i < combinedTraj.Length(); i += ppo->config.miniBatchSize) {
								int start = i;
								int end = RS_MIN(i + ppo->config.miniBatchSize, combinedTraj.Length());
								auto rewardPart = ppo->InferSORSRewards(
									tStates.slice(0, start, end).to(ppo->device, true, true),
									tActionComps.slice(0, start, end).to(ppo->device, true, true)
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

					report["Episode Length"] = 1.f / (tTerminals == 1).to(torch::kFloat32).mean().item<float>();

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

					if (config.ppo.useGCRL) {
						experience.data.actionComps    = torch::tensor(combinedTraj.actionComps).reshape({ -1, 8 });
						experience.data.futureGoals    = torch::tensor(combinedTraj.futureGoals).reshape({ -1, 6 });
						experience.data.carFutureGoals = torch::tensor(combinedTraj.carFutureGoals).reshape({ -1, 6 });
					}
				}

				// Free CUDA cache
#ifdef RG_CUDA_SUPPORT
				if (ppo->device.is_cuda())
					c10::cuda::CUDACachingAllocator::emptyCache();
#endif

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
						"GCRL/Adv Scale",
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
