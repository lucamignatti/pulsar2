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
				(torch::cuda::is_available() ? "libtorch cannot access the GPU" : "CUDA is not available to libtorch") << ".\n" <<
				"Make sure your libtorch comes with CUDA support, and that CUDA is installed properly."
			)
		device = at::Device(at::kCUDA);
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

		struct Trajectory {
			FList states, nextStates, rewards, logProbs;
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
			std::vector<uint8_t> touched, oppTouched; // Per-step touch flags (gate validity metric)
			FList gatedPos;                     // Positive parts of the gated reward components
			// Built at episode finalize by the HER relabel:
			FList carHerGoals, ballHerGoals;
			std::vector<uint8_t> ballMoved;

			void Clear() {
				*this = Trajectory();
			}

			void Append(const Trajectory& other) {
				other.AssertAligned();

				states += other.states;
				nextStates += other.nextStates;
				rewards += other.rewards;
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
				ballMoved += other.ballMoved;
			}

			// Every per-row column must have exactly one entry per action row; a missed append
			// would silently pair rows with the wrong steps (goals/gate trained on garbage)
			void AssertAligned() const {
				size_t n = actions.size();
				RG_ASSERT(rewards.size() == n && logProbs.size() == n && terminals.size() == n);
				if (!touched.empty()) {
					RG_ASSERT(touched.size() == n && oppTouched.size() == n && gatedPos.size() == n);
					RG_ASSERT(carHerGoals.size() == n * 6 && ballHerGoals.size() == n * 6 && ballMoved.size() == n);
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

		// Static per-player maps: owning arena, slot within it, and the first opposing player
		// (whose obs row supplies rho_opp for the control read); teams never change mid-run
		std::vector<int> playerArenaIdx(numPlayers), playerSlotIdx(numPlayers), playerPartnerIdx(numPlayers, -1);
		// Per-arena "did team X touch this step" scratch, indexed [team][arena] (hoisted out of the step loop)
		std::vector<uint8_t> arenaTeamTouched[2];
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
			for (int t = 0; t < n; t++) {
				int ballOff = fnPickOffset(t, reachCfg.ballHerMinOffset, reachCfg.ballHerMaxOffset, ballBiasPow, ballGoalwardBias);
				// Controllability is local: short window, no goalward bias
				int carOff = fnPickOffset(t, reachCfg.carHerMinOffset, reachCfg.carHerMaxOffset, carBiasPow, 0);

				for (int d = 0; d < 6; d++) {
					traj.ballHerGoals[(size_t)t * 6 + d] = traj.achievedBall[(size_t)(t + ballOff) * 6 + d];
					traj.carHerGoals[(size_t)t * 6 + d] = traj.achievedCarBall[(size_t)(t + carOff) * 6 + d];
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

			int numRealPlayers = oldVersion ? newPlayerIndices.size() : envSet->state.numPlayers;

			int stepsCollected = 0;
			{ // Generate experience

				// Only contains complete episodes
				auto combinedTraj = Trajectory();

				// Players handed to an old version stop being collected this iteration; their
				// in-flight partial episodes would otherwise silently SPLICE with a later
				// episode when they return (HER goals and gate windows would cross a hidden
				// reset). Finalize them as truncations instead — the current obs is exactly
				// the next-state the critic should bootstrap from.
				if (oldVersion) {
					for (int oldPlayerIdx : oldPlayerIndices) {
						auto& traj = trajectories[oldPlayerIdx];
						if (traj.Length() == 0)
							continue;

						traj.terminals.back() = RLGC::TerminalType::TRUNCATED;
						traj.nextStates += envSet->state.obs.GetRow(oldPlayerIdx);
						fnRelabelReachGoals(traj, oldPlayerIdx);
						combinedTraj.Append(traj);
						traj.Clear();
					}
				}

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

						torch::Tensor tActions, tLogProbs;
						torch::Tensor tStates = DIMLIST2_TO_TENSOR<float>(envSet->state.obs);
						torch::Tensor tActionMasks = DIMLIST2_TO_TENSOR<uint8_t>(envSet->state.actionMasks);

						if (!render) {
							for (int newPlayerIdx : newPlayerIndices) {
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
								auto& prevRewards = envSet->state.lastRewards[i];

								for (int j = 0; j < envSet->rewards[arenaIdx].size(); j++) {
									std::string rewardName = envSet->rewards[arenaIdx][j].reward->GetName();
									avgRewards[rewardName] += prevRewards[j];
								}
							}

							for (auto& pair : avgRewards)
								report.AddAvg("Rewards/" + pair.first, pair.second.Get());
						}

						// Per-arena, per-team touch flags for this step (avoids an O(players^2) scan)
						if (reachOn) {
							arenaTeamTouched[0].assign(envSet->arenas.size(), 0);
							arenaTeamTouched[1].assign(envSet->arenas.size(), 0);
							for (int arenaIdx = 0; arenaIdx < envSet->arenas.size(); arenaIdx++)
								for (auto& player : envSet->state.gameStates[arenaIdx].players)
									if (player.ballTouchedStep)
										arenaTeamTouched[(int)player.team][arenaIdx] = 1;
						}

						// Now that we've inferred and stepped the env, we can add that stuff to the trajectories
						int i = 0;
						for (int newPlayerIdx : newPlayerIndices) {
							trajectories[newPlayerIdx].actions.push_back(curActions[newPlayerIdx]);
							trajectories[newPlayerIdx].rewards += envSet->state.rewards[newPlayerIdx];
							trajectories[newPlayerIdx].logProbs += newLogProbs[i];

							if (reachOn) {
								auto& traj = trajectories[newPlayerIdx];
								traj.gatedPos += envSet->state.gatedPosRewards[newPlayerIdx];

								// Post-step touch flags (which team touched during this step)
								int arenaIdx = playerArenaIdx[newPlayerIdx];
								auto& player = envSet->state.gameStates[arenaIdx].players[playerSlotIdx[newPlayerIdx]];
								traj.touched.push_back(player.ballTouchedStep);
								traj.oppTouched.push_back(arenaTeamTouched[player.team == Team::BLUE ? 1 : 0][arenaIdx]);
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
									traj.nextStates += envSet->state.obs.GetRow(newPlayerIdx);
								}

								fnRelabelReachGoals(traj, newPlayerIdx);
								combinedTraj.Append(traj);
								traj.Clear();
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

					// Reachability: rho reads -> level x delta gate multiplier + validity metrics.
					// The gate scales REWARDS only (never advantages/values); with beta=0 or
					// gateEnabled=false the rewards are untouched.
					if (reachOn && combinedTraj.Length() > 0 && ppo->reach) {
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
							tStates, tActionMasks);
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
							dRelVar += (deltaRel[i] - dRelMean) * (deltaRel[i] - dRelMean);
						}

						report["Reach/Beta"] = beta;
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

					// Set experience buffer
					experience.data.actions = tActions;
					experience.data.logProbs = tLogProbs;
					experience.data.actionMasks = tActionMasks;
					experience.data.states = tStates;
					experience.data.advantages = tAdvantages;
					experience.data.targetValues = tTargetVals;

					if (reachOn) {
						experience.data.carHerGoals = torch::tensor(combinedTraj.carHerGoals).reshape({ -1, 6 });
						experience.data.ballHerGoals = torch::tensor(combinedTraj.ballHerGoals).reshape({ -1, 6 });
						experience.data.ballMovedMask = torch::tensor(combinedTraj.ballMoved);
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
						"Reach/Beta",
						"Reach/Gate Mult Mean",
						"Reach/Gate Mult P10",
						"Reach/Delta Control Std",
						"Reach/Car Accuracy",
						"Reach/Ball Accuracy",
						"Reach/Touch Pred Agreement",
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
	delete ppo;
	delete versionMgr;
	delete metricSender;
	delete renderSender;
	pybind11::finalize_interpreter();
}