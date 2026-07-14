#include "Learner.h"

#include <GigaLearnCPP/PPO/PPOLearner.h>
#include <GigaLearnCPP/PPO/ExperienceBuffer.h>

#include <torch/cuda.h>
#include <torch/mps.h>
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

#include <algorithm>
#include <map>
#include <random>
#include <thread>
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
	} else if (
		config.deviceType == LearnerDeviceType::GPU_MPS ||
		(config.deviceType == LearnerDeviceType::AUTO && torch::mps::is_available())
		) {
		RG_LOG("\tUsing MPS (Apple Metal) device...");

		bool deviceTestFailed = false;
		try {
			torch::Tensor t = torch::tensor(0);
			t = t.to(at::Device(at::kMPS));
			t = t.cpu();
		} catch (...) {
			deviceTestFailed = true;
		}

		if (!torch::mps::is_available() || deviceTestFailed)
			RG_ERR_CLOSE(
				"Learner::Learner(): Can't use MPS because " <<
				(torch::mps::is_available() ? "libtorch cannot access the Metal device" : "MPS is not available to libtorch") << ".\n" <<
				"MPS needs an Apple-silicon Mac and a libtorch built with Metal support."
			)
		device = at::Device(at::kMPS);

		// The halfPrec inference path clones weights to BF16 on-device (Models.cpp);
		// probe it here so an unsupported macOS/libtorch combo degrades loudly to fp32
		// at boot instead of throwing mid-collection
		if (config.ppo.useHalfPrecision) {
			try {
				auto t = torch::ones({ 4, 4 },
					torch::TensorOptions().device(device).dtype(torch::kBFloat16));
				(void)t.matmul(t).to(torch::kFloat).cpu();
			} catch (std::exception& e) {
				RG_LOG("\tWARNING: BF16 matmul unsupported on this MPS device (" << e.what()
					<< ") - disabling useHalfPrecision (fp32 inference)");
				config.ppo.useHalfPrecision = false;
			}
		}
	} else {
		RG_LOG("\tUsing CPU device...");
		device = at::Device(at::kCPU);
	}

	// The ctor body mutates its `config` PARAMETER (tsPerSave/randomSeed fix-ups above, the
	// MPS halfPrec fallback) but the member was copy-initialized before the body ran - sync
	// it here or Start()/Save() read the un-fixed values (e.g. tsPerSave=0 saving every
	// iteration; the same shadowing family as the Model ctor bug fixed in 155c2da).
	this->config = config;

	// Apply the TF32 matmul policy on CUDA (inert elsewhere) — without this the dense MLPs
	// run strict fp32 and never touch the Ampere+/Blackwell tensor-core path
	if (device.is_cuda()) {
		at::globalContext().setAllowTF32CuBLAS(config.allowTF32);
		at::globalContext().setAllowTF32CuDNN(config.allowTF32);
		RG_LOG("\tTF32 matmuls (CUDA tensor cores): " << (config.allowTF32 ? "enabled" : "disabled"));
	}

	if (!device.is_cpu()) {
		// On the GPU paths (CUDA/MPS) the heavy math runs on-device; libtorch's default
		// intra-op pool (one thread per core) otherwise oversubscribes the machine against
		// RLGymCPP's own collection thread pool for the many small CPU-side tensor ops
		// (index_select, blob conversions, .cpu() copies) in the consumption phase. Cap it low.
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
			// Only the main collection loop standardizes obs. Every other inference surface
			// (steering derivation's obs decoding, PSD/league/skill-tracker eval rollouts,
			// and Save() racing the worker's stat updates under pipelining) reads RAW obs
			// and would silently measure garbage - fail loudly instead of training on it.
			if (config.steering.enabled || config.psd.enabled || config.league.enabled
				|| config.skillTracker.enabled || config.pipelinedCollection)
				RG_ERR_CLOSE("Learner::Learner(): standardizeObs is only supported by the plain "
					"sequential PPO path - steering/PSD/league/skillTracker/pipelinedCollection "
					"all feed raw obs to the models and would break silently");
			this->obsStat = new BatchedWelfordStat(obsSize);
		} else {
			this->obsStat = NULL;
		}
	}

	// Deterministic sampling records no logprobs: training would read empty logprob rows
	// (garbage IS ratios / OOB) long before PPO's own learn-time check could throw
	if (config.ppo.deterministic && !config.renderMode)
		RG_ERR_CLOSE("Learner::Learner(): config.ppo.deterministic is for render/eval only, "
			"it cannot collect trainable experience");

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

	// Steering rating-guard state: the latch must survive the wrapper's automatic
	// crash-restarts (NaN EMA is skipped - JSON has no NaN and it just means "unseeded")
	if (!std::isnan(steerRatingEMA))
		j["steer_rating_ema"] = steerRatingEMA;
	j["steer_rating_tripped"] = steerRatingTripped;

	// Churn-telemetry vector archive, per mode (save-only; see Learner.h steerVecSave).
	// 1v1 keeps the legacy un-suffixed keys.
	for (int md = 0; md < 3; md++) {
		if (steerVecSave[md].empty())
			continue;
		std::string suffix = md == 0 ? "" : "_" + std::to_string(md + 1) + "v" + std::to_string(md + 1);
		j["steer_vec" + suffix] = steerVecSave[md];
		j["steer_sigma" + suffix] = steerSigmaSave[md];
	}

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

	if (j.contains("steer_rating_ema"))
		steerRatingEMA = (float)j["steer_rating_ema"];
	if (j.contains("steer_rating_tripped")) {
		steerRatingTripped = (bool)j["steer_rating_tripped"];
		if (steerRatingTripped)
			RG_LOG("NOTE: steering rating guard was TRIPPED in this checkpoint - steering stays "
				"latched OFF (a human decides; clear steer_rating_tripped in the checkpoint's "
				"running-stats JSON or restore an untripped checkpoint to re-enable)");
	}

	if (versionMgr)
		versionMgr->LoadRunningStatsFromJSON(j);

	if (psd)
		psd->FromJSON(j);
	if (league)
		league->FromJSON(j);
}

// Different than RLGym-PPO to show that they are not compatible
constexpr const char* STATS_FILE_NAME = "RUNNING_STATS.json";

bool GGL::Learner::BootSanityProbe() {
	RG_NO_GRAD;

	// Throwaway single arena driven manually (the main EnvSet is reserved for training).
	// 3 kickoff episodes x 8s; a healthy policy touches the ball nearly every episode
	// (median ~3.4s across every checkpoint ever tested); the 2026-07-13 scrambled
	// checkpoint managed 1/10. Pass = touches in >= 2 of 3 episodes.
	EnvCreateResult res = envCreateFn(0);
	Arena* arena = res.arena;

	int touchedEpisodes = 0;
	int stepsPerEp = (int)(8 * 120 / RS_MAX(1, config.tickSkip));
	int numPlayers = (int)arena->_cars.size();

	for (int ep = 0; ep < 3; ep++) {
		arena->ResetToRandomKickoff(ep);
		GameState gs = GameState(arena);
		res.obsBuilder->Reset(gs);
		auto actions = std::vector<Action>(numPlayers);

		for (int step = 0; step < stepsPerEp; step++) {
			FList obsAll;
			std::vector<uint8_t> masksAll;
			int obsSizeLocal = 0;
			for (int i = 0; i < numPlayers; i++) {
				FList obs = res.obsBuilder->BuildObs(gs.players[i], gs);
				obsSizeLocal = (int)obs.size();
				obsAll += obs;
				auto mask = res.actionParser->GetActionMask(gs.players[i], gs);
				masksAll.insert(masksAll.end(), mask.begin(), mask.end());
			}
			torch::Tensor tObs = torch::tensor(obsAll).reshape({ numPlayers, obsSizeLocal }).to(ppo->device);
			torch::Tensor tMasks = torch::tensor(masksAll).reshape({ numPlayers, -1 }).to(ppo->device);
			torch::Tensor tActs;
			ppo->InferActions(tObs, tMasks, &tActs, NULL); // no steer mask: raw policy
			auto acts = TENSOR_TO_VEC<int>(tActs.cpu());

			auto carItr = arena->_cars.begin();
			for (int i = 0; i < numPlayers; i++, carItr++) {
				actions[i] = res.actionParser->ParseAction(acts[i], gs.players[i], gs);
				(*carItr)->controls = (CarControls)actions[i];
			}
			arena->Step(config.tickSkip);
			gs.UpdateFromArena(arena, actions, NULL);

			bool touched = false;
			for (auto& player : gs.players)
				touched |= player.ballTouchedStep;
			if (touched) {
				touchedEpisodes++;
				break;
			}
		}
	}

	// Best-effort cleanup of the throwaway env pieces
	delete res.obsBuilder;
	delete res.actionParser;
	delete res.stateSetter;
	for (auto* c : res.terminalConditions)
		delete c;
	for (auto& wr : res.rewards)
		delete wr.reward;
	delete arena;

	RG_LOG(" > Boot sanity probe: kickoff touches in " << touchedEpisodes << "/3 episodes");
	return touchedEpisodes >= 2;
}


void GGL::Learner::Save() {
	if (config.checkpointFolder.empty())
		RG_ERR_CLOSE("Learner::Save(): Cannot save because config.checkpointSaveFolder is not set");

	// ATOMIC save (2026-07-13): a CUDA-watchdog crash mid-save left a truncated
	// checkpoint (0-byte RUNNING_STATS.json); the loader aborted on it every restart and
	// the wrapper crash-looped into giving up overnight. Write into "<ts>.tmp" (ignored
	// by FindNumberedDirs - not all-digits) and rename into place at the end: a crash
	// mid-save now leaves no numbered dir at all, and rename() is atomic on POSIX.
	std::filesystem::path finalFolder = config.checkpointFolder / std::to_string(totalTimesteps);
	std::filesystem::path saveFolder = config.checkpointFolder / (std::to_string(totalTimesteps) + ".tmp");
	std::filesystem::remove_all(saveFolder); // stale tmp from a prior mid-save crash
	std::filesystem::create_directories(saveFolder);

	RG_LOG("Saving to folder " << finalFolder << "...");
	SaveStats(saveFolder / STATS_FILE_NAME);
	ppo->SaveTo(saveFolder);
	std::filesystem::remove_all(finalFolder); // paranoia: re-save at an identical timestep
	std::filesystem::rename(saveFolder, finalFolder);

	// Golden archive: keep the top-N rated checkpoints permanently (see LearnerConfig).
	// Rate-limited (spacing + rating margin) so a steady climb doesn't copy every save.
	if (config.bestCheckpointsToKeep > 0 && !std::isnan(lastEvalRating)
		&& totalTimesteps - lastBestArchiveTs >= (uint64_t)config.bestArchiveMinTsSpacing) {
		try {
			std::vector<std::pair<float, std::filesystem::path>> bests;
			for (auto& e : std::filesystem::directory_iterator(config.checkpointFolder)) {
				std::string name = e.path().filename().string();
				if (name.rfind("best_r", 0) == 0) {
					size_t us = name.find('_', 6);
					bests.push_back({ std::stof(name.substr(6, us - 6)), e.path() });
				}
			}
			std::sort(bests.begin(), bests.end());
			bool qualifies = (int)bests.size() < config.bestCheckpointsToKeep
				|| lastEvalRating > bests.front().first + config.bestArchiveRatingMargin;
			if (qualifies) {
				std::string bestName = "best_r" + std::to_string((int)lastEvalRating)
					+ "_" + std::to_string(totalTimesteps);
				std::filesystem::copy(finalFolder, config.checkpointFolder / bestName,
					std::filesystem::copy_options::recursive);
				bests.push_back({ lastEvalRating, config.checkpointFolder / bestName });
				std::sort(bests.begin(), bests.end());
				while ((int)bests.size() > config.bestCheckpointsToKeep) {
					std::filesystem::remove_all(bests.front().second);
					bests.erase(bests.begin());
				}
				lastBestArchiveTs = totalTimesteps;
				RG_LOG(" > Archived best-rated checkpoint (" << (int)lastEvalRating << ")");
			}
		} catch (std::exception& e) {
			RG_LOG("WARNING: best-checkpoint archive maintenance failed: " << e.what());
		}
	}

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

	// Fallback across corrupt checkpoints (2026-07-13): a crash that interrupts a save
	// (pre-atomic-rename era, disk-full, etc.) must not strand an unattended run - abort
	// here meant the wrapper crash-looped on the same poisoned dir until it gave up.
	// Try newest -> oldest; a candidate that throws is renamed to "corrupt_<ts>" (ignored
	// by FindNumberedDirs) so future boots and the rotation never see it again. Retrying
	// is safe: LoadStats parses before assigning, and LoadFrom re-loads every model.
	std::set<int64_t> allSavedTimesteps = Utils::FindNumberedDirs(config.checkpointFolder);

	// One attempt: stats + models + (for checkpoints that claim competence) a behavioral
	// probe - a scrambled-but-loadable checkpoint (finite weights, destroyed policy;
	// 2026-07-13 GPU-lockup incident) throws here so the fallback can skip past it.
	auto fnTryLoad = [&](const std::filesystem::path& loadFolder) {
		LoadStats(loadFolder / STATS_FILE_NAME);
		ppo->LoadFrom(loadFolder);

		if (config.bootSanityCheckEnabled) {
			float claimedRating = 0;
			try {
				std::ifstream fIn(loadFolder / STATS_FILE_NAME);
				auto j = nlohmann::json::parse(fIn);
				if (j.contains("skill_ratings") && j["skill_ratings"].contains("1v1"))
					claimedRating = j["skill_ratings"]["1v1"].get<float>();
			} catch (...) {}
			if (claimedRating >= config.bootSanityMinRating && !BootSanityProbe())
				throw std::runtime_error("boot sanity probe failed: a policy rated "
					+ std::to_string((int)claimedRating) + " cannot touch kickoff balls - "
					"weights are likely scrambled (GPU-fault-era save)");
		}
	};

	bool loaded = false;
	for (auto itr = allSavedTimesteps.rbegin(); itr != allSavedTimesteps.rend(); itr++) {
		std::filesystem::path loadFolder = config.checkpointFolder / std::to_string(*itr);
		RG_LOG(" > Loading checkpoint " << loadFolder << "...");
		try {
			fnTryLoad(loadFolder);
			loaded = true;
			RG_LOG(" > Done.");
			break;
		} catch (std::exception& e) {
			RG_LOG(" > CORRUPT/UNREADABLE checkpoint " << loadFolder << " (" << e.what()
				<< ") - quarantining and falling back to the previous one");
			std::error_code ec;
			std::filesystem::rename(loadFolder,
				config.checkpointFolder / ("corrupt_" + std::to_string(*itr)), ec);
			if (ec)
				RG_LOG(" > (quarantine rename failed: " << ec.message() << " - skipping in place)");
		}
	}

	// Last resort: the golden best-rated archive (full checkpoints, highest rating first).
	// Loaded IN PLACE (not renamed) - archive entries are precious even if one is bad.
	// exists() guard: on a fresh run the folder may not exist yet and directory_iterator
	// would throw out of the constructor; a malformed best_r* name must skip, not abort -
	// this is exactly the unattended-recovery path.
	if (!loaded && std::filesystem::exists(config.checkpointFolder)) {
		std::vector<std::pair<float, std::filesystem::path>> bests;
		for (auto& e : std::filesystem::directory_iterator(config.checkpointFolder)) {
			std::string name = e.path().filename().string();
			if (name.rfind("best_r", 0) == 0) {
				try {
					bests.push_back({ std::stof(name.substr(6, name.find('_', 6) - 6)), e.path() });
				} catch (...) {
					RG_LOG(" > (ignoring unparseable archive dir name: " << name << ")");
				}
			}
		}
		std::sort(bests.rbegin(), bests.rend());
		for (auto& [rating, path] : bests) {
			RG_LOG(" > FALLING BACK TO BEST-RATED ARCHIVE: " << path << "...");
			try {
				fnTryLoad(path);
				loaded = true;
				RG_LOG(" > Done (resumed from golden archive, rating " << (int)rating << ").");
				break;
			} catch (std::exception& e) {
				RG_LOG(" > archive entry failed too (" << e.what() << "), trying next");
			}
		}
	}

	if (!loaded)
		RG_LOG(" > No (loadable) checkpoints found, starting new model.")
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
			// Own-TEAM touch (self OR teammate) - lets the steering labeler score team-arena
			// (group 3) readings by TEAM possession outcome, where self-only `touched` would
			// mislabel a teammate's race win as "nobody got it"
			std::vector<uint8_t> teamTouched;
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

			// Steered-practice rows (LearnerConfig::steering) - independent of the proposer's
			// practiceMask above (that one is drill-window machinery; this one just marks rows
			// from steered, resolution-terminated arenas for critic exclusion).
			// steerPractice = ROLE (0 match, 1 steered, 2 control); steerMode = the row's
			// arena team size - 1 (per-mode derivation pools, gates, and panels)
			std::vector<uint8_t> steerPractice;
			std::vector<uint8_t> steerMode;

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
				teamTouched.clear();
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
				steerPractice.clear();
				steerMode.clear();
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
					teamTouched.reserve(rows);
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

				steerPractice.reserve(rows); // cheap; populated only when steering is on
				steerMode.reserve(rows);
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
				teamTouched += other.teamTouched;
				gatedPos += other.gatedPos;
				carHerGoals += other.carHerGoals;
				ballHerGoals += other.ballHerGoals;
				steerPractice += other.steerPractice;
				steerMode += other.steerMode;
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
					RG_ASSERT(teamTouched.size() == n);
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
				if (!steerPractice.empty())
					RG_ASSERT(steerPractice.size() == n && steerMode.size() == n);
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

		// The report key the skill tracker writes for the TRAINING arenas' team size
		// ("Rating/1v1", "Rating/2v2", ...). Every in-loop rating consumer (steering rating
		// guard, golden-archive feed, PSD plateau/freeze feed) tracks this key, so the guards
		// keep working when the fleet moves to 2v2/3v3 instead of silently going dark on a
		// stale "Rating/1v1" literal. With a mixed-mode fleet, arena 0's mode is the one
		// tracked - keep the primary mode at index 0.
		const std::string ratingKey = "Rating/" + SkillRating::GetModeName(envSet->state.gameStates[0]);

		// Steered-practice collection (LearnerConfig::steering), PER MODE (4.0 team play).
		// The commitment frontier is mode-specific - offline census: collective declines on
		// feasible balls grow 74% -> 88% -> 92% from 1v1 -> 2v2 -> 3v3 - so each team size
		// gets its OWN direction, sigma, causal gate, and steered/control arena slices.
		// Modes are CONTIGUOUS arena blocks by the ExampleMain layout (1v1 leading, team
		// modes trailing, asserted below); each block's leading arenas become its practice
		// slice, so every mode has a within-mode treatment and control population.
		const bool steerOn = config.steering.enabled && !render;
		constexpr int STEER_MODES = 3; // index = playersPerTeam - 1
		auto fnModeName = [](int md) { return std::to_string(md + 1) + "v" + std::to_string(md + 1); };
		struct SteerModeBlock { int first = -1, count = 0, numPractice = 0, numSteered = 0; };
		std::array<SteerModeBlock, STEER_MODES> steerBlocks;
		std::vector<uint8_t> arenaMode(envSet->arenas.size(), 0);   // playersPerTeam - 1
		std::vector<uint8_t> arenaSteerRole(envSet->arenas.size(), 0); // 0 match, 1 steered, 2 control
		for (int a = 0; a < (int)envSet->arenas.size(); a++) {
			int ppt = (int)envSet->arenas[a]->_cars.size() / 2;
			RG_ASSERT(ppt >= 1 && ppt <= STEER_MODES);
			arenaMode[a] = (uint8_t)(ppt - 1);
			auto& blk = steerBlocks[ppt - 1];
			if (blk.first == -1)
				blk.first = a;
			else if (a != blk.first + blk.count)
				RG_ERR_CLOSE("Steering: arenas of mode " << fnModeName(ppt - 1)
					<< " are not a contiguous index block (arena " << a << " vs block ["
					<< blk.first << ", " << blk.first + blk.count << ")) - keep each team "
					"size contiguous in EnvCreateFunc");
			blk.count++;
		}
		if (steerOn) {
			for (int md = 0; md < STEER_MODES; md++) {
				auto& blk = steerBlocks[md];
				if (blk.count == 0)
					continue;
				if (md > 0 && !config.steering.steerTeamModes)
					continue;
				blk.numPractice = RS_CLAMP(
					(int)(blk.count * config.steering.practiceArenaFrac), 0, blk.count);
				if (blk.numPractice < 2)
					continue; // too small for a steered + control split
				int numControl = RS_CLAMP(
					(int)(blk.numPractice * config.steering.controlFracOfPractice),
					1, blk.numPractice - 1);
				blk.numSteered = blk.numPractice - numControl;
				for (int a = blk.first; a < blk.first + blk.numPractice; a++)
					arenaSteerRole[a] = (a < blk.first + blk.numSteered) ? 1 : 2;
				RG_LOG("Steered-practice collection [" << fnModeName(md) << "]: "
					<< blk.numSteered << " steered + " << numControl << " control of "
					<< blk.count << " arenas");
			}
			// Legacy aggregate members (kept for the stage-2 wiring/logs)
			numSteeredArenas = steerBlocks[0].numSteered;
			numPracticeArenas = steerBlocks[0].numPractice;
			RG_LOG("Steering: alpha " << config.steering.alpha
				<< (config.steering.resolutionTermination ? ", RESOLUTION-TERMINATED" : ", normal episodes (stage 1)")
				<< (config.steering.rhoGateEnabled ? ", rho-band gated (per-mode bands)" : "")
				<< (config.steering.steerTeamModes ? ", team modes steered" : ", team modes measurement-only")
				<< " (directions derived live per iteration; first iteration runs unsteered)");

			// Rho-band gate parameters live on the PPOLearner (it applies them at inference)
			ppo->steerRhoGate = config.steering.rhoGateEnabled && config.ppo.reachability.enabled;
			ppo->steerRhoContact = config.steering.rhoGateOnContact;
			ppo->steerRhoLo = config.steering.rhoGateLo;
			ppo->steerRhoHi = config.steering.rhoGateHi;
			ppo->steerRhoK = RS_MAX(1, config.steering.rhoGateActionSamples);
		}

		// Opponent style library (config.steering.opponentStylesFile, roadmap phase 1):
		// offline-validated trunk directions the OPPONENT side occasionally plays with.
		// Loaded once and immutable afterwards - the pipelined collect worker reads it
		// without synchronization. A malformed file is a config error and fails loud.
		struct OppStyle { std::string name; torch::Tensor vec; float sigma, aLo, aHi; };
		std::vector<OppStyle> oppStyles;
		if (steerOn && !config.steering.opponentStylesFile.empty()) {
			std::ifstream sf(config.steering.opponentStylesFile);
			if (!sf.good()) {
				RG_LOG("Opponent styles: " << config.steering.opponentStylesFile
					<< " not found - opponent styling off");
			} else {
				try {
					nlohmann::json js = nlohmann::json::parse(sf);
					for (auto& e : js) {
						OppStyle s;
						s.name = e["name"];
						std::vector<float> v = e["vec"].get<std::vector<float>>();
						s.vec = torch::tensor(v);
						s.sigma = (float)e["sigma"];
						s.aLo = (float)e["alpha_lo"];
						s.aHi = (float)e["alpha_hi"];
						oppStyles.push_back(std::move(s));
					}
				} catch (std::exception& e) {
					RG_ERR_CLOSE("Opponent styles: failed to parse "
						<< config.steering.opponentStylesFile << ": " << e.what());
				}
				RG_LOG("Opponent styles: " << oppStyles.size() << " loaded from "
					<< config.steering.opponentStylesFile << " (chance "
					<< config.steering.opponentStyleChance << " per opponent iteration)");
			}
		}
		std::atomic<int> oppStyleIters = 0, oppIters = 0; // worker increments, report reads

		// Live per-mode steering state (derived in fnSteerUpdate during learn-prep, applied
		// in the barrier zone where no collect worker is in flight). CPU tensors.
		std::array<torch::Tensor, STEER_MODES> steerVecEMA;
		std::array<float, STEER_MODES> steerSigmaEMA = {};
		std::array<float, STEER_MODES> steerGateDeltaEMA = {}; // steered-minus-control team-possession, EMA
		std::array<int, STEER_MODES> steerGateIters = {};      // iterations that contributed gate data
		std::array<bool, STEER_MODES> steerGateActive;         // alpha drops to 0 when the causal gate trips
		steerGateActive.fill(true);
		bool steerPendingApply = false;
		// Rating drawdown guard state (steerRatingEMA / steerRatingTripped) lives on the
		// Learner and is persisted in the checkpoint stats - a crash-restart must not
		// silently un-latch steering (the wrapper restarts automatically and unattended)
		// Car-free arenas for ball-landing sims, one per ad-hoc sim thread (lazy, reused).
		// Deliberately NOT on the shared thread pool: in pipelined mode learn-prep overlaps the
		// collect worker, which owns the pool.
		constexpr int STEER_SIM_THREADS = 4;
		std::vector<Arena*> steerSimArenas;
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
		// goalCriticOn: the goal-channel recorder below indexes these maps too - without it,
		// a goal-critic-only config would read arena 0 / slot 0 for every player and train
		// the goal critic on garbage credit
		if (reachOn || steerOn || goalCriticOn) {
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


		// ================= Pipelined collection (config.pipelinedCollection) =================
		// Overlaps NEXT-iteration experience collection with THIS iteration's processing + Learn().
		// Collapse-safety design (this exact failure mode has killed runs before):
		//   * The worker NEVER infers from live training weights — it uses `collectSnapshot`, a frozen
		//     copy synced at the barrier, so a forward can never see half-updated (torn) parameters.
		//   * logProbs are recorded from the SAME snapshot that sampled the actions, so PPO's ratio
		//     pi_new/pi_behavior is exact; the one-update policy lag is standard async-PPO staleness
		//     that the clip objective is built to absorb.
		//   * Everything that steps the training EnvSet (PSD probes), submits to the global thread
		//     pool (league/skill-eval match envs), or evaluates live weights (league, version manager)
		//     runs in the BARRIER ZONE between join and kick — never concurrent with the worker.
		// REVERT: set config.pipelinedCollection = false — the flag-off path is the exact sequential
		// order and call pattern (inline collect, live models, original tail call sites).
		const bool pipelineOn = config.pipelinedCollection && !render && !practiceOn && !proposerOn;
		// The pipelined worker's frozen model set. With the steering rho-gate on, the reach
		// heads are snapshotted too - the worker must never read weights Learn is updating.
		std::vector<const char*> snapshotNames = { "shared_head", "policy" };
		if (steerOn && config.steering.rhoGateEnabled) {
			snapshotNames.push_back("reach_phi");
			snapshotNames.push_back("reach_psi_ball");
			snapshotNames.push_back("reach_psi_car");
		}
		ModelSet collectSnapshot;
		if (pipelineOn)
			for (const char* nm : snapshotNames)
				if (ppo->models[nm])
					collectSnapshot.Add(ppo->models[nm]->MakeClone());
		ModelSet* const collectModelsPtr = pipelineOn ? &collectSnapshot : NULL;
		auto fnSyncSnapshot = [&]() {
			RG_NO_GRAD;
			for (const char* nm : snapshotNames) {
				Model* dst = collectSnapshot[nm];
				if (!dst)
					continue;
				auto to = dst->parameters();
				auto from = ppo->models[nm]->parameters();
				for (size_t i = 0; i < to.size(); i++)
					to[i].copy_(from[i], true);
				dst->_seqHalfOutdated = true;
			}
		};
		Trajectory combinedTrajNext;  // the worker fills this; swapped into combinedTraj at the join
		Report collectReport;         // worker-owned between barriers; merged into the iteration report
		int collectSteps = 0;
		float collectWallTime = 0;
		uint64_t prevVersionTimesteps = totalTimesteps;
		// jthread, NOT thread: if anything throws in the main loop while the worker is collecting,
		// unwinding to the catch below destroys this object — a joinable std::thread would call
		// std::terminate there, aborting BEFORE the real error is ever printed (found the hard way:
		// the abort masked the underlying exception entirely). jthread joins on destruction instead,
		// so the worker drains and the actual exception reaches RG_ERR_CLOSE.
		// ===== Steered-practice live derivation (LearnerConfig::CollectSteeringConfig) =====
		// Runs inside learn-prep (grad-free, main thread) on the just-collected buffer:
		//   1. label airborne-ball readings via ball-only landing sims + within-episode lookahead
		//   2. engagement per arena group -> causal auto-gate (steered vs control practice)
		//   3. matched went/declined difference of trunk means from MATCH rows only -> EMA
		// The updated direction is applied in the barrier zone (fnApplySteering), never while a
		// collect worker is in flight. Rows of one player's episode are CONTIGUOUS in
		// combinedTraj (episodes are appended whole at finalize), so lookahead is row + k.
		auto fnSteerUpdate = [&](Report& report) {
			const auto& cfgS = config.steering;
			const auto& states = combinedTraj.states;
			const auto& groups = combinedTraj.steerPractice;
			int64_t n = (int64_t)combinedTraj.Length();
			if (n == 0 || groups.size() != (size_t)n || combinedTraj.steerMode.size() != (size_t)n)
				return;
			// v2 possession labels need the reach buffers' touch flags
			if (combinedTraj.touched.size() != (size_t)n || combinedTraj.oppTouched.size() != (size_t)n
				|| combinedTraj.teamTouched.size() != (size_t)n)
				return;

			// Obs layout (AdvancedObs, 1v1), team-canonical frame - the field is symmetric under
			// the canonical flip, so landing sims run directly in it. Constants mirror the
			// validated offline pipeline (analysis/probes/steer_test.py).
			constexpr int BALL_POS = 0, BALL_VEL = 3, BALL_ANGVEL = 6, SELF_POS = 51;
			constexpr float POS_SCALE = 5000, VEL_SCALE = 2300, ANGVEL_SCALE = 3;
			constexpr float ARM_Z = 300, LANDING_Z = 111.25f;
			constexpr float FEASIBLE_SPEED = 1300;
			constexpr int SIM_CAP_TICKS = 720;
			const float stepsPerSec = 120.f / config.tickSkip;

			auto fnObs = [&](int64_t row, int off) { return states[row * (int64_t)obsSize + off]; };

			// 1) Candidate readings: airborne-ball rows, strided to the per-iteration cap
			struct Reading {
				int64_t row, epEnd;
				uint8_t role, mode;
				float land[3] = {}; // x, y, t
				bool ok = false;
			};
			const auto& rowModes = combinedTraj.steerMode;
			std::vector<Reading> readings;
			{
				std::vector<std::pair<int64_t, int64_t>> cand; // (row, epEnd)
				int64_t epStart = 0;
				for (int64_t r = 0; r < n; r++) {
					if (!combinedTraj.terminals[r])
						continue;
					for (int64_t i = epStart; i <= r; i++)
						if (fnObs(i, BALL_POS + 2) * POS_SCALE > ARM_Z)
							cand.push_back({ i, r });
					epStart = r + 1;
				}
				size_t stride = RS_MAX((size_t)1, cand.size() / (size_t)RS_MAX(1, cfgS.maxReadingsPerIter));
				for (size_t i = 0; i < cand.size(); i += stride)
					readings.push_back({ cand[i].first, cand[i].second,
						groups[cand[i].first], rowModes[cand[i].first] });
			}
			if (readings.empty())
				return;

			// 2) Ball-only landing sims. Ad-hoc threads with their own car-free arenas: the
			// shared pool belongs to the (possibly in-flight) collect worker in pipelined mode.
			while ((int)steerSimArenas.size() < STEER_SIM_THREADS)
				steerSimArenas.push_back(Arena::Create(GameMode::SOCCAR));
			{
				std::atomic<size_t> nextIdx = 0;
				std::vector<std::thread> simThreads;
				for (int t = 0; t < STEER_SIM_THREADS; t++) {
					simThreads.emplace_back([&, t]() {
						Arena* arena = steerSimArenas[t];
						size_t i;
						while ((i = nextIdx++) < readings.size()) {
							auto& rd = readings[i];
							BallState bs = {};
							bs.pos = Vec(fnObs(rd.row, BALL_POS) * POS_SCALE,
								fnObs(rd.row, BALL_POS + 1) * POS_SCALE,
								fnObs(rd.row, BALL_POS + 2) * POS_SCALE);
							bs.vel = Vec(fnObs(rd.row, BALL_VEL) * VEL_SCALE,
								fnObs(rd.row, BALL_VEL + 1) * VEL_SCALE,
								fnObs(rd.row, BALL_VEL + 2) * VEL_SCALE);
							bs.angVel = Vec(fnObs(rd.row, BALL_ANGVEL) * ANGVEL_SCALE,
								fnObs(rd.row, BALL_ANGVEL + 1) * ANGVEL_SCALE,
								fnObs(rd.row, BALL_ANGVEL + 2) * ANGVEL_SCALE);
							arena->ball->SetState(bs);
							for (int tick = 1; tick <= SIM_CAP_TICKS; tick++) {
								arena->Step(1);
								auto b = arena->ball->GetState();
								if (b.pos.z <= LANDING_Z) {
									rd.land[0] = b.pos.x;
									rd.land[1] = b.pos.y;
									rd.land[2] = tick / 120.f;
									rd.ok = true;
									break;
								}
							}
						}
					});
				}
				for (auto& th : simThreads)
					th.join();
			}

			// 3) Label feasible readings by POSSESSION OUTCOME (v2), per mode: scan the same
			// player's subsequent rows (episodes are row-contiguous per player) from the
			// reading to shortly past touchdown for the FIRST touch - self = WON, teammate =
			// TEAMMATE (a resolved race, not a decline), opponent = LOST, nobody = NONE (in
			// team modes: a COLLECTIVE decline - everyone assumed someone else would go; the
			// offline census reads 74%/88%/92% NONE for 1v1/2v2/3v3, so this is exactly the
			// team-mode frontier). Possession is style-proof: winning the ball first is good
			// at every level and unfakeable by empty flight.
			struct Labeled { int64_t row; bool won; float dNow, tLand; };
			// Per-mode direction pools: WON vs NONE. LOST (tried, got beaten) and TEAMMATE
			// (someone went) are excluded - punishing lost races trains hesitation back in,
			// and teammate-resolved races carry no self-commitment signal.
			std::array<std::vector<Labeled>, STEER_MODES> pools;
			// Frontier rows for the phase-3 reset pool: feasible + collectively declined,
			// MATCH rows only (practice arenas must never feed their own reset pool)
			std::array<std::vector<int64_t>, STEER_MODES> frontierRows;
			int feas[STEER_MODES][3] = {}, possTeamWon[STEER_MODES][3] = {};
			const float raceMarginRows = 0.5f * stepsPerSec; // grace past touchdown for the race
			for (auto& rd : readings) {
				if (!rd.ok)
					continue;
				int64_t tdRow = rd.row + (int64_t)roundf(rd.land[2] * stepsPerSec);
				if (tdRow > rd.epEnd)
					continue; // censored: episode ended before touchdown
				float sx = fnObs(rd.row, SELF_POS) * POS_SCALE;
				float sy = fnObs(rd.row, SELF_POS + 1) * POS_SCALE;
				float dNow = sqrtf((sx - rd.land[0]) * (sx - rd.land[0]) + (sy - rd.land[1]) * (sy - rd.land[1]));
				if (dNow / RS_MAX(rd.land[2], 1e-6f) >= FEASIBLE_SPEED)
					continue;
				feas[rd.mode][rd.role]++;

				int outcome = 0; // 0 none, 1 self won, 2 lost, 3 teammate won
				int64_t scanEnd = RS_MIN(rd.epEnd, tdRow + (int64_t)raceMarginRows);
				for (int64_t rr = rd.row + 1; rr <= scanEnd; rr++) {
					if (combinedTraj.touched[rr]) { outcome = 1; break; }
					if (combinedTraj.teamTouched[rr]) { outcome = 3; break; }
					if (combinedTraj.oppTouched[rr]) { outcome = 2; break; }
				}

				// Gate/panel metric = TEAM possession (identical to self-won in 1v1):
				// style-proof at team level, and the safety net against steered double-commits
				if (outcome == 1 || outcome == 3)
					possTeamWon[rd.mode][rd.role]++;
				if (rd.role == 0 && (outcome == 0 || outcome == 1))
					pools[rd.mode].push_back({ rd.row, outcome == 1, dNow, rd.land[2] });
				if (rd.role == 0 && outcome == 0)
					frontierRows[rd.mode].push_back(rd.row);
			}

			// Panels: legacy un-suffixed keys stay 1v1 (wandb continuity); team modes suffixed
			const char* roleNames[3] = { "Match", "Steered", "Control" };
			auto fnModeKey = [&](const char* base, int md) {
				return md == 0 ? std::string(base) : std::string(base) + " " + fnModeName(md);
			};
			for (int md = 0; md < STEER_MODES; md++)
				for (int g = 0; g < 3; g++)
					if (feas[md][g] >= 50)
						report[fnModeKey((std::string("Steer/PossWin ") + roleNames[g]).c_str(), md)] =
							(float)possTeamWon[md][g] / feas[md][g];

			// 4) Causal auto-gates, one per mode: that mode's steered arenas must WIN THE BALL
			// (team possession) on feasible aerial situations at least as often as its controls
			// (identical arenas, only steering differs)
			for (int md = 0; md < STEER_MODES; md++) {
				if (steerBlocks[md].numSteered == 0)
					continue;
				if (steerLoaded && feas[md][1] >= 50 && feas[md][2] >= 50) {
					float delta = (float)possTeamWon[md][1] / feas[md][1]
						- (float)possTeamWon[md][2] / feas[md][2];
					steerGateDeltaEMA[md] = (steerGateIters[md] == 0)
						? delta : 0.9f * steerGateDeltaEMA[md] + 0.1f * delta;
					steerGateIters[md]++;
					report[fnModeKey("Steer/Gate Delta EMA", md)] = steerGateDeltaEMA[md];
					if (steerGateIters[md] >= cfgS.gateWarmupIters) {
						if (steerGateActive[md] && steerGateDeltaEMA[md] < cfgS.gateDisableBelow) {
							steerGateActive[md] = false;
							steerPendingApply = true;
							RG_LOG("Steering causal gate [" << fnModeName(md) << "] TRIPPED "
								"(team-possession delta EMA " << steerGateDeltaEMA[md]
								<< ") - alpha -> 0, derivation continues");
						} else if (!steerGateActive[md] && steerGateDeltaEMA[md] > cfgS.gateReenableAbove) {
							steerGateActive[md] = true;
							steerPendingApply = true;
							RG_LOG("Steering causal gate [" << fnModeName(md) << "] RE-ENGAGED "
								"(team-possession delta EMA " << steerGateDeltaEMA[md] << ")");
						}
					}
				}
				report[fnModeKey("Steer/Gate Active", md)] = (float)steerGateActive[md];
			}

			// 5) Matched difference of trunk means -> per-mode EMA direction + sigma.
			// A mode whose pool is too thin keeps its own EMA untouched; fnApplySteering
			// falls back to the 1v1 direction for it (validated offline: the 1v1 direction
			// TRANSFERS to 2v2 - teamWon +3.4pp at +0.5 with sign-correct air/touch shifts -
			// while directions derived from thin weak-team pools were causally dead).
			auto fnGatherTrunk = [&](const std::vector<int64_t>& rows) {
				FList buf;
				buf.reserve(rows.size() * obsSize);
				for (int64_t r : rows)
					for (int c = 0; c < obsSize; c++)
						buf.push_back(states[r * (int64_t)obsSize + c]);
				torch::Tensor obs = torch::tensor(buf).reshape({ (int64_t)rows.size(), obsSize }).to(ppo->device);
				return ppo->models["shared_head"]->Forward(obs, false).cpu();
			};
			for (int md = 0; md < STEER_MODES; md++) {
				if (steerBlocks[md].count == 0 || (md > 0 && !cfgS.steerTeamModes))
					continue;
				auto& pool = pools[md];
				std::vector<int64_t> selWent, selDecl;
				if ((int)pool.size() >= 2 * cfgS.minPairsPerUpdate) {
					// quantile bin edges over the pool (5 distance bins x 3 flight-time bins)
					auto fnEdges = [&](auto getter, int bins) {
						FList vals;
						for (auto& p : pool)
							vals.push_back(getter(p));
						std::sort(vals.begin(), vals.end());
						FList edges;
						for (int b = 1; b < bins; b++)
							edges.push_back(vals[vals.size() * b / bins]);
						return edges;
					};
					FList dEdges = fnEdges([](const Labeled& p) { return p.dNow; }, 5);
					FList tEdges = fnEdges([](const Labeled& p) { return p.tLand; }, 3);
					auto fnBin = [&](const Labeled& p) {
						int db = 0, tb = 0;
						while (db < (int)dEdges.size() && p.dNow > dEdges[db]) db++;
						while (tb < (int)tEdges.size() && p.tLand > tEdges[tb]) tb++;
						return db * 8 + tb;
					};
					std::map<int, std::pair<std::vector<int64_t>, std::vector<int64_t>>> byBin;
					for (auto& p : pool)
						(p.won ? byBin[fnBin(p)].first : byBin[fnBin(p)].second).push_back(p.row);
					std::mt19937 shuffleRng((unsigned)(totalIterations + md * 7919));
					for (auto& kv : byBin) {
						auto& w = kv.second.first;
						auto& d = kv.second.second;
						std::shuffle(w.begin(), w.end(), shuffleRng);
						std::shuffle(d.begin(), d.end(), shuffleRng);
						size_t m = RS_MIN(w.size(), d.size());
						selWent.insert(selWent.end(), w.begin(), w.begin() + m);
						selDecl.insert(selDecl.end(), d.begin(), d.begin() + m);
					}
				}

				if ((int)selWent.size() >= cfgS.minPairsPerUpdate) {
					torch::Tensor vNew = (fnGatherTrunk(selWent).mean(0) - fnGatherTrunk(selDecl).mean(0));
					vNew = vNew / vNew.norm().clamp_min(1e-8f);
					if (steerVecEMA[md].defined()) {
						report[fnModeKey("Steer/Dir Drift", md)] =
							1.f - torch::dot(steerVecEMA[md], vNew).item<float>();
						steerVecEMA[md] = cfgS.emaDecay * steerVecEMA[md] + (1.f - cfgS.emaDecay) * vNew;
						steerVecEMA[md] = steerVecEMA[md] / steerVecEMA[md].norm().clamp_min(1e-8f);
					} else {
						steerVecEMA[md] = vNew;
					}
					report[fnModeKey("Steer/Pairs", md)] = (float)selWent.size();
				}
				report[fnModeKey("Steer/Dir Source Own", md)] = (float)steerVecEMA[md].defined();

				// sigma of THIS mode's match-row projections onto the vector this mode will
				// APPLY (its own EMA, or the 1v1 fallback) - so the dose alpha*sigma is
				// calibrated to the mode's own trunk distribution either way
				torch::Tensor applied = steerVecEMA[md].defined() ? steerVecEMA[md] : steerVecEMA[0];
				if (applied.defined()) {
					std::vector<int64_t> sampleRows;
					int64_t sampleStride = RS_MAX((int64_t)1, n / 4096);
					for (int64_t r = 0; r < n; r += sampleStride)
						if (groups[r] == 0 && rowModes[r] == md)
							sampleRows.push_back(r);
					if (sampleRows.size() >= 64) {
						float sigNew = fnGatherTrunk(sampleRows).matmul(applied).std().item<float>();
						steerSigmaEMA[md] = (steerSigmaEMA[md] == 0) ? sigNew
							: cfgS.emaDecay * steerSigmaEMA[md] + (1.f - cfgS.emaDecay) * sigNew;
						steerPendingApply = true;
					}
					report[fnModeKey("Steer/Sigma", md)] = steerSigmaEMA[md];
				}
			}

			// 6) Frontier reset pool (roadmap phase 3): bank the collectively-declined
			// readings as canonical-frame reset entries reconstructed from obs-visible
			// quantities. Layout support: AdvancedObsPadded(3) (230) and plain AdvancedObs
			// 1v1 (109); anything else skips (loudly, once).
			if (cfgS.frontierPool) {
				auto& fpool = cfgS.frontierPool;
				fpool->Advance(totalIterations);
				const bool padded = obsSize == 230, plain = obsSize == 109;
				static bool warnedLayout = false;
				if (!padded && !plain) {
					if (!warnedLayout) {
						RG_LOG("Frontier pool: unsupported obs layout (size " << obsSize
							<< ") - pool stays empty");
						warnedLayout = true;
					}
				} else {
					// Per-player block field offsets (AddPlayerToObs): pos 0, forward 3,
					// up 6, vel 9, angVel 12, boost 24, isDemoed 27
					auto fnCar = [&](int64_t row, int base) {
						RLGC::FrontierPool::CarSpawn c;
						c.pos = Vec(fnObs(row, base + 0), fnObs(row, base + 1), fnObs(row, base + 2)) * POS_SCALE;
						c.forward = Vec(fnObs(row, base + 3), fnObs(row, base + 4), fnObs(row, base + 5));
						c.up = Vec(fnObs(row, base + 6), fnObs(row, base + 7), fnObs(row, base + 8));
						c.vel = Vec(fnObs(row, base + 9), fnObs(row, base + 10), fnObs(row, base + 11)) * VEL_SCALE;
						c.angVel = Vec(fnObs(row, base + 12), fnObs(row, base + 13), fnObs(row, base + 14)) * ANGVEL_SCALE;
						c.boost = fnObs(row, base + 24) * 100.f;
						return c;
					};
					constexpr int SELF_BASE = 51, TM_SLOT0 = 80, OPP_SLOT0 = 138, PRESENCE0 = 225;
					auto fnEntry = [&](int64_t row, int md, RLGC::FrontierPool::Entry& e) {
						e.ball = BallState{};
						e.ball.pos = Vec(fnObs(row, BALL_POS), fnObs(row, BALL_POS + 1), fnObs(row, BALL_POS + 2)) * POS_SCALE;
						e.ball.vel = Vec(fnObs(row, BALL_VEL), fnObs(row, BALL_VEL + 1), fnObs(row, BALL_VEL + 2)) * VEL_SCALE;
						e.ball.angVel = Vec(fnObs(row, BALL_ANGVEL), fnObs(row, BALL_ANGVEL + 1), fnObs(row, BALL_ANGVEL + 2)) * ANGVEL_SCALE;
						if (fnObs(row, SELF_BASE + 27) > 0.5f)
							return false; // self demoed: no sane spawn
						e.blueCars.push_back(fnCar(row, SELF_BASE));
						if (padded) {
							for (int s = 0; s < 2; s++) {
								if (fnObs(row, PRESENCE0 + s) <= 0.5f)
									continue;
								if (fnObs(row, TM_SLOT0 + s * 29 + 27) > 0.5f)
									return false;
								e.blueCars.push_back(fnCar(row, TM_SLOT0 + s * 29));
							}
							for (int s = 0; s < 3; s++) {
								if (fnObs(row, PRESENCE0 + 2 + s) <= 0.5f)
									continue;
								if (fnObs(row, OPP_SLOT0 + s * 29 + 27) > 0.5f)
									return false;
								e.orangeCars.push_back(fnCar(row, OPP_SLOT0 + s * 29));
							}
						} else {
							if (fnObs(row, 80 + 27) > 0.5f)
								return false;
							e.orangeCars.push_back(fnCar(row, 80));
						}
						return (int)e.blueCars.size() == md + 1 && (int)e.orangeCars.size() == md + 1;
					};
					for (int md = 0; md < STEER_MODES; md++) {
						auto& rows = frontierRows[md];
						if (rows.empty())
							continue;
						std::vector<RLGC::FrontierPool::Entry> entries;
						entries.reserve(RS_MIN((int)rows.size(), cfgS.frontierPoolPerMode));
						size_t stride = RS_MAX((size_t)1, rows.size() / (size_t)RS_MAX(1, cfgS.frontierPoolPerMode));
						for (size_t i = 0; i < rows.size() && (int)entries.size() < cfgS.frontierPoolPerMode; i += stride) {
							RLGC::FrontierPool::Entry e;
							if (fnEntry(rows[i], md, e))
								entries.push_back(std::move(e));
						}
						report[fnModeKey("Steer/Frontier Pool", md)] = (float)entries.size();
						fpool->Fill(md, std::move(entries));
					}
				}
			}
		};

		// ===== META frontier steering (roadmap phase 4, prior-free) =====
		// Generalizes the commitment mechanism from ONE hand-derived contrast to the
		// agent's whole capability frontier, with NO human priors: goals are sampled from
		// the agent's own achieved-state bank; the frontier is what its own self-model
		// rates coin-flip; structure is emergent (k-means in its own psi geometry, no
		// cluster ever named); outcomes are model-free continuous ATTAINMENT (how close
		// future achieved states got to the goal within the head's own HER horizon),
		// compared only through within-population quantiles. Offline validation
		// (analysis/probes/meta_frontier_validate.py, STEERING_META_40.md): the ball
		// head's calibration is monotone (a real frontier detector), the car head's is
		// not for arbitrary goals (it self-disables here while keeping its contact-gate
		// role), and at least one emergent cluster shows a monotone causal attainment
		// uplift under its derived direction with clean canaries - with heterogeneity
		// across clusters, which is exactly the scheduler's reason to exist.
		struct MetaClusterState {
			torch::Tensor centroid;  // [repr] psi space, slot-matched + EMA'd across iters
			torch::Tensor repGoal;   // [6] representative achieved goal (raw goal space)
			torch::Tensor dirEMA;    // [trunk] the cluster's steering direction
			float effectEMA = 0;     // normalized causal effect (steered-vs-control attain)
			int effectIters = 0;
			bool benched = false;
			int64_t lastDwellIdx = -1;
			int lastPairs = 0;
		};
		std::array<std::vector<MetaClusterState>, 2> metaSlots; // [0]=car head, [1]=ball head
		std::array<std::array<float, 3>, 2> metaCalibEMA = {};  // median attain below/in/above
		std::array<int, 2> metaCalibIters = {};
		std::array<bool, 2> metaHeadValid = { false, false };   // monotone calibration gate
		int metaActiveHead = -1, metaActiveCluster = -1;
		int64_t metaDwellIdx = 0;
		int metaDwellLeft = 0;
		std::array<float, STEER_MODES> metaSigma = {};
		bool metaPendingApply = false;
		const bool metaOn = steerOn && config.steering.meta && reachOn;

		auto fnMetaUpdate = [&](Report& report) {
			if (!metaOn)
				return;
			RG_NO_GRAD;
			const auto& cfgS = config.steering;
			const auto& states = combinedTraj.states;
			const auto& groups = combinedTraj.steerPractice;
			const auto& rowModes = combinedTraj.steerMode;
			int64_t n = (int64_t)combinedTraj.Length();
			if (n == 0 || groups.size() != (size_t)n || rowModes.size() != (size_t)n)
				return;

			auto fnObs = [&](int64_t row, int off) { return states[row * (int64_t)obsSize + off]; };
			const bool paddedObs = obsSize == 230, plainObs = obsSize == 109;
			if (!paddedObs && !plainObs)
				return; // unsupported layout (loud enough via the frontier-pool warning)

			// Episode ids + episode end per row (rows of one player's episode are contiguous)
			std::vector<int32_t> epId(n);
			std::vector<int64_t> epEnd(n);
			{
				int32_t cur = 0;
				int64_t start = 0;
				for (int64_t r = 0; r < n; r++) {
					epId[r] = cur;
					if (combinedTraj.terminals[r]) {
						for (int64_t i = start; i <= r; i++)
							epEnd[i] = r;
						cur++;
						start = r + 1;
					}
				}
				for (int64_t i = start; i < n; i++)
					epEnd[i] = n - 1; // trailing partial episode
			}

			// Achieved-goal vectors per row per head, from the canonical obs the policy
			// consumed (fnAppendAchieved parity; obs coefs 1/5000 pos, 1/2300 vel)
			const auto& rc = config.ppo.reachability;
			auto fnAch = [&](int64_t r, int head, float* out) {
				if (head == 1) { // ball: canonical pos/vel -> reach scales
					out[0] = fnObs(r, 0) * 5000.f / rc.posScaleX;
					out[1] = fnObs(r, 1) * 5000.f / rc.posScaleY;
					out[2] = fnObs(r, 2) * 5000.f / rc.posScaleZ;
					out[3] = fnObs(r, 3) * 2300.f / rc.velScale;
					out[4] = fnObs(r, 4) * 2300.f / rc.velScale;
					out[5] = fnObs(r, 5) * 2300.f / rc.velScale;
				} else { // car: self-block local ball pos/vel -> carLocalScale
					constexpr int SELF = 51;
					out[0] = fnObs(r, SELF + 18) * 5000.f / 2300.f;
					out[1] = fnObs(r, SELF + 19) * 5000.f / 2300.f;
					out[2] = fnObs(r, SELF + 20) * 5000.f / 2300.f;
					out[3] = fnObs(r, SELF + 21);
					out[4] = fnObs(r, SELF + 22);
					out[5] = fnObs(r, SELF + 23);
				}
			};

			auto fnGatherTrunkRows = [&](const std::vector<int64_t>& rows) {
				FList buf;
				buf.reserve(rows.size() * obsSize);
				for (int64_t r : rows)
					for (int c = 0; c < obsSize; c++)
						buf.push_back(states[r * (int64_t)obsSize + c]);
				torch::Tensor obs = torch::tensor(buf).reshape({ (int64_t)rows.size(), obsSize }).to(ppo->device);
				return ppo->models["shared_head"]->Forward(obs, false);
			};

			auto fnKMeans = [&](torch::Tensor emb, int k, int iters) {
				auto perm = torch::randperm(emb.size(0)).slice(0, 0, k);
				torch::Tensor cent = emb.index_select(0, perm).clone();
				torch::Tensor labels = torch::zeros({ emb.size(0) }, torch::kLong);
				for (int it = 0; it < iters; it++) {
					labels = emb.matmul(cent.t()).argmax(1);
					for (int c = 0; c < k; c++) {
						auto sel = (labels == c).nonzero().flatten();
						if (sel.numel() > 0) {
							auto m = emb.index_select(0, sel).mean(0);
							cent[c] = m / m.norm().clamp_min(1e-6f);
						}
					}
				}
				return std::make_pair(cent, labels);
			};

			// ---- per head: bank -> clusters -> mined pairs -> calibration/dirs/effect
			for (int head = 0; head < 2; head++) {
				const char* headName = head == 0 ? "car" : "ball";
				Model* psi = ppo->models[head == 0 ? "reach_psi_car" : "reach_psi_ball"];
				Model* phi = ppo->models["reach_phi"];
				if (!psi || !phi)
					continue;
				int W = head == 0 ? rc.carHerMaxOffset : rc.ballHerMaxOffset;

				// Bank: the agent's own achieved goals, from MATCH rows with a full window
				std::vector<int64_t> bankRows;
				{
					int64_t tries = 0;
					while ((int)bankRows.size() < cfgS.metaBankSize && tries++ < cfgS.metaBankSize * 8) {
						int64_t r = (int64_t)RocketSim::Math::RandInt(0, (int)n);
						if (groups[r] == 0 && r + W <= epEnd[r])
							bankRows.push_back(r);
					}
				}
				if ((int)bankRows.size() < cfgS.metaClusters * 8)
					continue;
				torch::Tensor bankGoals = torch::zeros({ (int64_t)bankRows.size(), 6 });
				for (size_t i = 0; i < bankRows.size(); i++)
					fnAch(bankRows[i], head, bankGoals[i].data_ptr<float>());
				torch::Tensor bankEmb = psi->Forward(bankGoals.to(ppo->device), false).cpu();
				bankEmb = bankEmb / bankEmb.norm(2, -1, true).clamp_min(1e-6f);

				auto [cent, bankLabels] = fnKMeans(bankEmb, cfgS.metaClusters, 8);

				// Slot-match new centroids to persistent slots (greedy max cosine) + EMA,
				// so per-cluster state keeps its identity across iterations
				auto& slots = metaSlots[head];
				if ((int)slots.size() != cfgS.metaClusters) {
					slots.assign(cfgS.metaClusters, MetaClusterState{});
					for (int c = 0; c < cfgS.metaClusters; c++)
						slots[c].centroid = cent[c].clone();
				}
				std::vector<int> newToSlot(cfgS.metaClusters, -1);
				{
					std::vector<bool> slotUsed(cfgS.metaClusters, false);
					torch::Tensor slotCent = torch::zeros({ cfgS.metaClusters, cent.size(1) });
					for (int c = 0; c < cfgS.metaClusters; c++)
						slotCent[c] = slots[c].centroid;
					torch::Tensor sim = cent.matmul(slotCent.t()); // [new, slot]
					for (int step = 0; step < cfgS.metaClusters; step++) {
						auto flat = sim.argmax().item<int64_t>();
						int nc = (int)(flat / cfgS.metaClusters), sc = (int)(flat % cfgS.metaClusters);
						newToSlot[nc] = sc;
						sim.index_put_({ nc, torch::indexing::Slice() }, -2.f);
						sim.index_put_({ torch::indexing::Slice(), sc }, -2.f);
					}
					for (int nc = 0; nc < cfgS.metaClusters; nc++) {
						auto& s = slots[newToSlot[nc]];
						s.centroid = cfgS.metaCentroidEma * s.centroid + (1 - cfgS.metaCentroidEma) * cent[nc];
						s.centroid = s.centroid / s.centroid.norm().clamp_min(1e-6f);
					}
				}
				// Representative goal per slot = bank goal nearest the slot centroid
				{
					torch::Tensor slotCent = torch::zeros({ cfgS.metaClusters, bankEmb.size(1) });
					for (int c = 0; c < cfgS.metaClusters; c++)
						slotCent[c] = slots[c].centroid;
					torch::Tensor sims = bankEmb.matmul(slotCent.t()); // [bank, slot]
					auto best = sims.argmax(0);                        // [slot]
					for (int c = 0; c < cfgS.metaClusters; c++)
						slots[c].repGoal = bankGoals[best[c].item<int64_t>()].clone();
				}

				// Mined pairs: strided rows (all roles - roles 1/2 feed the causal effect,
				// role 0 feeds the direction pools), goals sampled cross-episode
				std::vector<int64_t> mineRows;
				int64_t stride = RS_MAX((int64_t)1, n / RS_MAX(1, cfgS.metaMaxRows));
				for (int64_t r = 0; r < n; r += stride)
					if (r + W <= epEnd[r])
						mineRows.push_back(r);
				if ((int)mineRows.size() < 64)
					continue;

				torch::Tensor trunk = fnGatherTrunkRows(mineRows); // [rows, trunk] on device
				// K uniform valid actions per row -> phi pieces
				torch::Tensor masksT;
				{
					FList mbuf;
					mbuf.reserve(mineRows.size() * numActions);
					for (int64_t r : mineRows)
						for (int a = 0; a < numActions; a++)
							mbuf.push_back((float)combinedTraj.actionMasks[r * (int64_t)numActions + a]);
					masksT = torch::tensor(mbuf).reshape({ (int64_t)mineRows.size(), numActions })
						.clamp_min(1e-9f).to(ppo->device);
				}
				int K = RS_MAX(1, cfgS.rhoGateActionSamples);
				auto acts = torch::multinomial(masksT, K, true);
				auto trunkRep = trunk.repeat_interleave(K, 0);
				auto oneHot = torch::one_hot(acts.flatten(), numActions).to(torch::kFloat32);
				torch::Tensor phiT = phi->Forward(torch::cat({ trunkRep, oneHot }, -1), false);
				phiT = (phiT / phiT.norm(2, -1, true).clamp_min(1e-6f))
					.view({ (int64_t)mineRows.size(), K, -1 }).cpu();
				trunk = trunk.cpu();

				// Pair assembly
				struct MetaPair { int rowIdx; int bankIdx; float rho, attain; uint8_t bandPos, role; };
				std::vector<MetaPair> pairs;
				pairs.reserve(mineRows.size() * cfgS.metaGoalsPerRow);
				for (size_t i = 0; i < mineRows.size(); i++) {
					int64_t r = mineRows[i];
					for (int gsel = 0; gsel < cfgS.metaGoalsPerRow; gsel++) {
						int bi = -1;
						for (int t = 0; t < 8; t++) {
							int c = RocketSim::Math::RandInt(0, (int)bankRows.size());
							if (epId[bankRows[c]] != epId[r]) { bi = c; break; }
						}
						if (bi < 0)
							continue;
						float rho = phiT[i].matmul(bankEmb[bi]).mean().item<float>();
						pairs.push_back({ (int)i, bi, rho, 0.f, 0, groups[r] });
					}
				}
				if ((int)pairs.size() < 256)
					continue;

				// Band quantiles over this head's pairs
				{
					std::vector<float> rhos(pairs.size());
					for (size_t j = 0; j < pairs.size(); j++)
						rhos[j] = pairs[j].rho;
					std::sort(rhos.begin(), rhos.end());
					float lo = rhos[(size_t)(rhos.size() * cfgS.rhoGateLo)];
					float hi = rhos[(size_t)(rhos.size() * cfgS.rhoGateHi)];
					for (auto& p : pairs)
						p.bandPos = p.rho < lo ? 0 : (p.rho <= hi ? 1 : 2);
				}

				// Attainment (model-free): -min goal-space distance over the head's window
				for (auto& p : pairs) {
					int64_t r = mineRows[p.rowIdx];
					const float* g = bankGoals[p.bankIdx].data_ptr<float>();
					float best = 1e30f;
					float a[6];
					for (int64_t f = r + 1; f <= RS_MIN(epEnd[r], r + W); f++) {
						fnAch(f, head, a);
						float d2 = 0;
						for (int d = 0; d < 6; d++)
							d2 += (a[d] - g[d]) * (a[d] - g[d]);
						best = RS_MIN(best, d2);
					}
					p.attain = -sqrtf(best);
				}

				// Calibration: median attain below/in/above the band; the head drives
				// steering ONLY while this is monotone (it disables its own bad senses)
				{
					std::array<std::vector<float>, 3> byBand;
					for (auto& p : pairs)
						byBand[p.bandPos].push_back(p.attain);
					auto fnMedian = [](std::vector<float>& v) {
						if (v.empty()) return NAN;
						std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
						return v[v.size() / 2];
					};
					for (int b = 0; b < 3; b++) {
						float m = fnMedian(byBand[b]);
						if (!std::isnan(m))
							metaCalibEMA[head][b] = metaCalibIters[head] == 0 ? m
								: 0.9f * metaCalibEMA[head][b] + 0.1f * m;
					}
					metaCalibIters[head]++;
					metaHeadValid[head] = metaCalibIters[head] >= 3
						&& metaCalibEMA[head][0] <= metaCalibEMA[head][1]
						&& metaCalibEMA[head][1] <= metaCalibEMA[head][2];
					report[std::string("Meta/Calib Valid ") + headName] = (float)metaHeadValid[head];
				}

				// Per-cluster: direction pools (role 0 only - practice arenas never feed
				// their own direction) + the ACTIVE cluster's causal effect (roles 1 vs 2)
				for (int c = 0; c < cfgS.metaClusters; c++) {
					auto& slot = slots[c];
					std::vector<const MetaPair*> pool;
					std::vector<float> steered, control;
					// pair cluster = its bank goal's slot-matched label
					for (auto& p : pairs) {
						int slotOfPair = newToSlot[(int)bankLabels[p.bankIdx].item<int64_t>()];
						if (slotOfPair != c || p.bandPos != 1)
							continue;
						if (p.role == 0)
							pool.push_back(&p);
						else if (p.role == 1)
							steered.push_back(p.attain);
						else
							control.push_back(p.attain);
					}
					slot.lastPairs = (int)pool.size();

					// Direction: matched top-vs-bottom attain terciles within rho quintiles
					if ((int)pool.size() >= 2 * cfgS.minPairsPerUpdate) {
						std::sort(pool.begin(), pool.end(),
							[](const MetaPair* a, const MetaPair* b) { return a->rho < b->rho; });
						std::vector<int64_t> selTop, selBot;
						size_t binSize = pool.size() / 5;
						for (int b = 0; b < 5 && binSize >= 6; b++) {
							auto first = pool.begin() + b * binSize;
							auto last = b == 4 ? pool.end() : first + binSize;
							std::vector<const MetaPair*> bin(first, last);
							std::sort(bin.begin(), bin.end(),
								[](const MetaPair* a, const MetaPair* b) { return a->attain < b->attain; });
							size_t third = bin.size() / 3;
							for (size_t i = 0; i < third; i++) {
								selBot.push_back(mineRows[bin[i]->rowIdx]);
								selTop.push_back(mineRows[bin[bin.size() - 1 - i]->rowIdx]);
							}
						}
						if ((int)selTop.size() >= cfgS.minPairsPerUpdate) {
							torch::Tensor vNew = (fnGatherTrunkRows(selTop).mean(0)
								- fnGatherTrunkRows(selBot).mean(0)).cpu();
							vNew = vNew / vNew.norm().clamp_min(1e-8f);
							slot.dirEMA = slot.dirEMA.defined()
								? cfgS.emaDecay * slot.dirEMA + (1 - cfgS.emaDecay) * vNew
								: vNew;
							slot.dirEMA = slot.dirEMA / slot.dirEMA.norm().clamp_min(1e-8f);
						}
					}

					// Causal effect for the ACTIVE cluster: normalized attain shift
					if (head == metaActiveHead && c == metaActiveCluster
						&& steered.size() >= 30 && control.size() >= 30) {
						auto fnMedIqr = [](std::vector<float>& v, float& med, float& iqr) {
							std::sort(v.begin(), v.end());
							med = v[v.size() / 2];
							iqr = v[(size_t)(v.size() * 0.75)] - v[(size_t)(v.size() * 0.25)];
						};
						float ms, is, mc, ic;
						fnMedIqr(steered, ms, is);
						fnMedIqr(control, mc, ic);
						float eff = (ms - mc) / RS_MAX(ic, 1e-3f);
						slot.effectEMA = slot.effectIters == 0 ? eff
							: 0.95f * slot.effectEMA + 0.05f * eff;
						slot.effectIters++;
						report["Meta/Effect EMA"] = slot.effectEMA;
						report["Meta/Attain Steered"] = ms;
						report["Meta/Attain Control"] = mc;
						if (slot.effectIters >= cfgS.metaWarmupIters) {
							if (!slot.benched && slot.effectEMA < cfgS.metaEffectTrip) {
								slot.benched = true;
								RG_LOG("Meta steering: cluster " << headName << "/" << c
									<< " BENCHED (effect EMA " << slot.effectEMA << ")");
							} else if (slot.benched && slot.effectEMA > cfgS.metaEffectReenable) {
								slot.benched = false;
								RG_LOG("Meta steering: cluster " << headName << "/" << c
									<< " UNBENCHED (effect EMA " << slot.effectEMA << ")");
							}
						}
					}
				}
			}

			// ---- Scheduler: one active (head, cluster); dwell + periodic exploration
			if (metaDwellLeft > 0)
				metaDwellLeft--;
			if (metaDwellLeft == 0) {
				metaDwellIdx++;
				bool explore = (metaDwellIdx % RS_MAX(1, (int64_t)cfgS.metaExploreEvery)) == 0;
				int bestH = -1, bestC = -1;
				float bestScore = -1e30f;
				for (int h = 0; h < 2; h++) {
					if (!metaHeadValid[h])
						continue;
					for (int c = 0; c < (int)metaSlots[h].size(); c++) {
						auto& s = metaSlots[h][c];
						if (!s.dirEMA.defined() || s.lastPairs < cfgS.minPairsPerUpdate)
							continue;
						if (!explore && s.benched)
							continue;
						float score = explore
							? -(float)s.lastDwellIdx            // stalest first
							: (s.effectIters == 0 ? 1e6f        // unmeasured: probe it
								: s.effectEMA);
						if (score > bestScore) {
							bestScore = score;
							bestH = h;
							bestC = c;
						}
					}
				}
				if (bestH >= 0) {
					if (bestH != metaActiveHead || bestC != metaActiveCluster)
						RG_LOG("Meta steering: active cluster -> "
							<< (bestH == 0 ? "car" : "ball") << "/" << bestC
							<< (explore ? " (exploration dwell)" : ""));
					metaActiveHead = bestH;
					metaActiveCluster = bestC;
					metaSlots[bestH][bestC].lastDwellIdx = metaDwellIdx;
				}
				metaDwellLeft = RS_MAX(1, cfgS.metaDwellIters);
			}
			report["Meta/Active Head"] = (float)metaActiveHead;
			report["Meta/Active Cluster"] = (float)metaActiveCluster;
			if (metaActiveHead >= 0 && metaActiveCluster >= 0)
				report["Meta/Active Pairs"] = (float)metaSlots[metaActiveHead][metaActiveCluster].lastPairs;

			// Per-mode sigma of match-row projections onto the ACTIVE direction
			if (metaActiveHead >= 0 && metaActiveCluster >= 0) {
				auto& act = metaSlots[metaActiveHead][metaActiveCluster];
				if (act.dirEMA.defined()) {
					for (int md = 0; md < STEER_MODES; md++) {
						if (steerBlocks[md].count == 0)
							continue;
						std::vector<int64_t> sampleRows;
						int64_t sampleStride = RS_MAX((int64_t)1, n / 4096);
						for (int64_t r = 0; r < n; r += sampleStride)
							if (groups[r] == 0 && rowModes[r] == md)
								sampleRows.push_back(r);
						if (sampleRows.size() >= 64) {
							float sig = fnGatherTrunkRows(sampleRows).cpu()
								.matmul(act.dirEMA).std().item<float>();
							metaSigma[md] = metaSigma[md] == 0 ? sig
								: cfgS.emaDecay * metaSigma[md] + (1 - cfgS.emaDecay) * sig;
						}
					}
					metaPendingApply = true;
				}
			}
		};

		// Rating drawdown guard: called wherever the skill tracker may have just written
		// the training mode's rating into the report. A drop of more than ratingDrawdownTrip
		// below the slow EMA latches steering OFF for the rest of the process - no
		// auto-re-enable, a human decides (both live collapses tonight were visible on this
		// signal within minutes while every behavioral gate stayed green).
		auto fnRatingGuard = [&](Report& report) {
			if (!steerOn || !config.steering.ratingGuardEnabled || !report.Has(ratingKey))
				return;
			float rating = (float)report[ratingKey];
			if (std::isnan(steerRatingEMA)) {
				steerRatingEMA = rating;
				return;
			}
			if (!steerRatingTripped && rating < steerRatingEMA - config.steering.ratingDrawdownTrip) {
				steerRatingTripped = true;
				steerPendingApply = true; // push alpha=0 at the next barrier
				RG_LOG("STEERING RATING GUARD TRIPPED: " << ratingKey << " " << rating
					<< " vs EMA " << steerRatingEMA << " (drawdown > "
					<< config.steering.ratingDrawdownTrip << ") - steering latched OFF");
			}
			steerRatingEMA = config.steering.ratingEmaDecay * steerRatingEMA
				+ (1.f - config.steering.ratingEmaDecay) * rating;
		};

		// Applies the latest derived direction/gate state to the PPOLearner. ONLY call when no
		// collect worker is in flight (barrier zone / sequential mode) - the worker reads
		// ppo->steerVec without synchronization.
		// Applies the META system's active cluster: its direction on all modes (dosed by
		// each mode's own sigma), its representative goal as the rho-gate target, alphas
		// gated by benching + the rating latch. ONLY call in the barrier zone.
		auto fnApplyMeta = [&]() {
			if (!metaOn || !metaPendingApply || metaActiveHead < 0 || metaActiveCluster < 0)
				return;
			auto& act = metaSlots[metaActiveHead][metaActiveCluster];
			if (!act.dirEMA.defined())
				return;
			std::array<torch::Tensor, PPOLearner::STEER_MODES> vecs = {};
			std::array<float, PPOLearner::STEER_MODES> sigmas = {}, alphas = {};
			bool any = false;
			for (int md = 0; md < PPOLearner::STEER_MODES; md++) {
				if (steerBlocks[md].numSteered == 0 || metaSigma[md] == 0)
					continue;
				vecs[md] = act.dirEMA;
				sigmas[md] = metaSigma[md];
				alphas[md] = (!act.benched && !steerRatingTripped) ? config.steering.alpha : 0.f;
				any = true;
			}
			if (!any)
				return;
			ppo->SetSteering(vecs, sigmas, alphas);
			ppo->SetSteerGoal(act.repGoal, metaActiveHead == 0);
			steerLoaded = true;
			metaPendingApply = false;

			// Churn-telemetry archive: under meta, the archived vector is the ACTIVE
			// cluster's direction (what collection actually steers with)
			for (int md = 0; md < PPOLearner::STEER_MODES; md++) {
				if (vecs[md].defined()) {
					auto vec = vecs[md].contiguous();
					steerVecSave[md].assign(vec.data_ptr<float>(), vec.data_ptr<float>() + vec.numel());
					steerSigmaSave[md] = sigmas[md];
				}
			}
		};

		auto fnApplySteering = [&]() {
			// Under META, the incumbent commitment derivation keeps running for its
			// panels/gates but never actuates - fnApplyMeta owns the applied vectors
			if (metaOn) {
				fnApplyMeta();
				return;
			}
			if (!steerOn || !steerPendingApply)
				return;
			// Per-mode vectors: a mode without its own direction falls back to the 1v1 one
			// (offline-validated transfer), dosed by ITS OWN sigma. Modes stay undefined
			// (unsteered) until a vector + sigma exist for them.
			std::array<torch::Tensor, PPOLearner::STEER_MODES> vecs = {};
			std::array<float, PPOLearner::STEER_MODES> sigmas = {}, alphas = {};
			bool any = false;
			for (int md = 0; md < PPOLearner::STEER_MODES; md++) {
				torch::Tensor v = steerVecEMA[md].defined() ? steerVecEMA[md]
					: (md > 0 && config.steering.steerTeamModes ? steerVecEMA[0] : torch::Tensor());
				if (!v.defined() || steerSigmaEMA[md] == 0 || steerBlocks[md].numSteered == 0)
					continue;
				vecs[md] = v;
				sigmas[md] = steerSigmaEMA[md];
				alphas[md] = (steerGateActive[md] && !steerRatingTripped) ? config.steering.alpha : 0.f;
				any = true;
			}
			if (!any)
				return;
			ppo->SetSteering(vecs, sigmas, alphas);
			steerLoaded = true;
			steerPendingApply = false;

			// Snapshot for the checkpoint's churn-telemetry archive (Save() runs on this
			// thread at the same safe point, so no synchronization is needed)
			for (int md = 0; md < PPOLearner::STEER_MODES; md++) {
				if (vecs[md].defined()) {
					auto vec = vecs[md].contiguous();
					steerVecSave[md].assign(vec.data_ptr<float>(), vec.data_ptr<float>() + vec.numel());
					steerSigmaSave[md] = sigmas[md];
				}
			}
		};

		std::jthread collectThread;
		// The whole per-iteration collection (opponent selection -> env stepping -> episode finalize).
		// Defined OUTSIDE the iteration loop on purpose: it must not capture any loop-local (the
		// compiler enforces this — loop locals aren't in scope here), because in pipelined mode it
		// executes concurrently with the NEXT iteration's locals.
		auto fnCollectIteration = [&]() {
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

			// Opponent style draw (roadmap phase 1): with opponentStyleChance, this
			// iteration's opponent additionally plays a validated style direction at an
			// alpha from its dose window. Applied on the OPPONENT inference call only.
			torch::Tensor oppStyleVec = {};
			float oppStyleCoef = 0;
			if (oppModels) {
				oppIters++;
				if (!oppStyles.empty()
					&& RocketSim::Math::RandFloat() < config.steering.opponentStyleChance) {
					auto& st = oppStyles[RocketSim::Math::RandInt(0, (int)oppStyles.size())];
					oppStyleVec = st.vec;
					oppStyleCoef = RocketSim::Math::RandFloat(st.aLo, st.aHi) * st.sigma;
					oppStyleIters++;
				}
			}

			// Steered-practice row mask + per-row mode for the CURRENT-POLICY inference calls
			// below (the mode index selects that row's direction on the PPOLearner). Control
			// practice arenas, old-version and league opponents are never steered. Static
			// across the iteration: arena assignment and the practice split don't change
			// mid-collect.
			torch::Tensor tSteerMask = {}, tSteerModes = {};
			if (steerOn && steerLoaded) {
				auto steerRows = std::vector<uint8_t>(numPlayers);
				auto steerModes = std::vector<int64_t>(numPlayers);
				for (int i = 0; i < numPlayers; i++) {
					steerRows[i] = arenaSteerRole[playerArenaIdx[i]] == 1;
					steerModes[i] = arenaMode[playerArenaIdx[i]];
				}
				tSteerMask = torch::tensor(steerRows).to(torch::kBool);
				tSteerModes = torch::tensor(steerModes);
				if (oppModels) {
					tSteerMask = tSteerMask.index_select(0, tNewPlayerIndices);
					tSteerModes = tSteerModes.index_select(0, tNewPlayerIndices);
				}
			}

			collectSteps = 0;
			// -- Generate experience (scope brace removed: body now lives in the collect fn) --

				combinedTrajNext.ClearKeepCapacity();
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

					// Obs-normalization stats, fetched/clamped ONCE per collect call (hoisted out of the
					// step loop): the running stat moves negligibly within a single iteration.
					std::vector<double> obsNormMean, obsNormStd;
					if (!render && obsStat) {
						obsNormMean = obsStat->GetMean();
						obsNormStd = obsStat->GetSTD();
						for (double& f : obsNormMean)
							f = RS_CLAMP(f, -config.maxObsMeanRange, config.maxObsMeanRange);
						for (double& f : obsNormStd)
							f = RS_MAX(f, config.minObsSTD);
					}

					for (int step = 0; combinedTrajNext.Length() < config.ppo.tsPerItr || render; step++, collectSteps += numRealPlayers) {
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
							int numSamples = RS_MIN(envSet->state.numPlayers, config.maxObsSamples);
							for (int i = 0; i < numSamples; i++) {
								int idx = RocketSim::Math::RandInt(0, envSet->state.numPlayers);
								obsStat->IncrementRow(&envSet->state.obs.At(idx, 0));
							}

							// mean/std hoisted: refreshed once per iteration below (obsNormMean/Std) — the
							// running stat drifts negligibly within one iteration and per-step GetMean/GetSTD
							// re-fetch + clamp was pure overhead
							for (int i = 0; i < envSet->state.numPlayers; i++) {
								for (int j = 0; j < obsSize; j++) {
									float& obsVal = envSet->state.obs.At(i, j);
									obsVal = (obsVal - obsNormMean[j]) / obsNormStd[j];
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

							ppo->InferActions(tdNewStates, tdNewActionMasks, &tNewActions, &tLogProbs, collectModelsPtr, tSteerMask, tSteerModes);
							ppo->InferActions(tdOldStates, tdOldActionMasks, &tOldActions, NULL, oppModels,
								{}, {}, oppStyleVec, oppStyleCoef);

							tActions = torch::zeros(numPlayers, tNewActions.dtype());
							tActions.index_copy_(0, tNewPlayerIndices, tNewActions.cpu());
							tActions.index_copy_(0, tOldPlayerIndices, tOldActions.cpu());
						} else {
							torch::Tensor tdStates = tStates.to(ppo->device, true);
							torch::Tensor tdActionMasks = tActionMasks.to(ppo->device, true);
							ppo->InferActions(tdStates, tdActionMasks, &tActions, &tLogProbs, collectModelsPtr, tSteerMask, tSteerModes);
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
							stepCallback(this, envSet->state.gameStates, collectReport);

						if (render) {
							renderSender->Send(envSet->state.gameStates[0]);
							if (config.renderReloadSecs > 0 && renderReloadTimer.Elapsed() >= config.renderReloadSecs) {
								renderReloadTimer.Reset();
								ReloadNewestCheckpointForRender(renderLoadedTS);
							}
							continue;
						}

						// Calc average rewards
						if (config.addRewardsToMetrics && (RocketSim::Math::RandInt(0, config.rewardSampleRandInterval) == 0)) {
							int numSamples = RS_MIN(envSet->arenas.size(), config.maxRewardSamples);
							std::unordered_map<std::string, AvgTracker> avgRewards = {};
							for (int i = 0; i < numSamples; i++) {
								int arenaIdx = RocketSim::Math::RandInt(0, envSet->arenas.size());
								auto& prevRewards = envSet->state.lastRewards[arenaIdx];

								for (int j = 0; j < envSet->rewards[arenaIdx].size(); j++) {
									std::string rewardName = envSet->rewards[arenaIdx][j].reward->GetName();
									avgRewards[rewardName] += prevRewards[j];
								}
							}

							for (auto& pair : avgRewards)
								collectReport.AddAvg("Rewards/" + pair.first, pair.second.Get());
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

							// Steered-practice row tags: ROLE (0 match, 1 steered, 2 control)
							// + MODE (team size - 1). Tagged by ARENA, not by whether a vector
							// is active: the resolution-terminated returns are what poison the
							// critic, steered or not; the 1-vs-2 split feeds the causal gates.
							if (steerOn) {
								int arena = playerArenaIdx[newPlayerIdx];
								trajectories[newPlayerIdx].steerPractice.push_back(arenaSteerRole[arena]);
								trajectories[newPlayerIdx].steerMode.push_back(arenaMode[arena]);
							}

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
								traj.teamTouched.push_back(arenaTeamTouched[player.team == Team::BLUE ? 0 : 1][arenaIdx]);

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
						// draws from the shared RNG and combinedTrajNext is shared.
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
								if (!render && obsStat) {
									// This row is post-step, captured BEFORE the next loop-top
									// standardization pass - normalize it here or the critic
									// (trained on standardized obs) bootstraps from a raw row
									size_t rowStart = traj.nextStates.size() - obsSize;
									for (int j = 0; j < obsSize; j++)
										traj.nextStates[rowStart + j] =
											(traj.nextStates[rowStart + j] - (float)obsNormMean[j]) / (float)obsNormStd[j];
								}
							}

							finalTerminals[newPlayerIdx] = terminalType;
						});

						// Serial finalize, in the original player order (deterministic episode
						// order in combinedTrajNext, same as the old fully-serial loop)
						for (int newPlayerIdx : newPlayerIndices) {
							if (!finalTerminals[newPlayerIdx])
								continue;

							auto& traj = trajectories[newPlayerIdx];
							fnRelabelReachGoals(traj, newPlayerIdx);
							fnAppendProposerTargets(traj);
							combinedTrajNext.Append(traj);
							traj.Clear();
						}

						recordTime += recordTimer.Elapsed();
					}

					collectReport["Inference Time"] = inferTime;
					collectReport["Env Step Time"] = envStepTime;
					collectReport["Prep Time"] = prepTime;
					collectReport["Record Time"] = recordTime;
				}
				collectWallTime = collectionTimer.Elapsed();
				collectReport["Collection Time"] = collectWallTime;
		};

		while (true) {
			Report report = {};

			bool isFirstIteration = (totalTimesteps == 0);

			// ================= Collection (pipelined orchestration) =================
			// Sequential mode: collect runs inline right here (worker thread never started).
			// Pipelined mode: the worker collected THIS iteration's data during the previous
			// iteration's processing+Learn; join it, swap buffers, run the barrier-zone work, then
			// kick the worker for the NEXT iteration with a freshly-frozen policy snapshot.
			Timer iterTimer = {};
			if (collectThread.joinable()) {
				collectThread.join();
			} else {
				fnApplySteering(); // no worker in flight - safe to swap the direction
				fnCollectIteration(); // first iteration, or sequential mode
			}
			int stepsCollected = collectSteps;
			std::swap(combinedTraj, combinedTrajNext);
			collectReport.Finish();
			for (auto& kv : collectReport.data)
				report.data[kv.first] = kv.second;
			collectReport.Clear();
			float collectionTime = collectWallTime;

			// ---- BARRIER ZONE (worker idle): shared-resource consumers. In sequential mode these
			// stay at their original tail call sites; exactly one site is active per mode. Running
			// them here (top of iteration N) is the same program point as the tail of iteration N-1.
			if (pipelineOn) {
				if (versionMgr)
					versionMgr->OnIteration(ppo, report, totalTimesteps, prevVersionTimesteps);
				prevVersionTimesteps = totalTimesteps;
				if (report.Has(ratingKey))
					lastEvalRating = (float)report[ratingKey]; // feeds the best-checkpoint archive
				fnRatingGuard(report); // may latch steering off; applied by fnApplySteering below
				if (league)
					league->OnIteration(report, totalIterations);
				if (psd) {
					float rating = report.Has(ratingKey) ? (float)report[ratingKey] : NAN;
					bool probed = psd->OnDescendIteration(report, rating, totalIterations);
					if (probed) {
						// Probe rounds stepped the arenas out from under the in-flight episodes
						for (auto& traj : trajectories)
							traj.Clear();
					}
				}
				// Freeze the current policy for the worker, then collect the next iteration
				// concurrently with this iteration's processing + Learn.
				fnApplySteering(); // barrier zone: the worker is joined, no reader in flight
				fnSyncSnapshot();
				collectThread = std::jthread([&]() { fnCollectIteration(); });
			}


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
						int64_t nOutcome = outcomeMask.sum().item<int64_t>();
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
							// STAGE-2 only: resolution-terminated practice rows never receive
							// goal-critic credit (their goal channel is structurally 0). Stage 1
							// episodes are normal, so the blend applies everywhere. steerPractice
							// is the ROLE tag (0 = match in any mode).
							if (config.steering.resolutionTermination && !combinedTraj.steerPractice.empty()) {
								torch::Tensor tGroups = torch::tensor(combinedTraj.steerPractice);
								injected = injected * (tGroups == 0).to(torch::kFloat32);
							}
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

					// Steered-practice bookkeeping. The goal-critic exclusion mask is a
					// STAGE-2 (resolutionTermination) semantic - stage 1 runs normal episodes,
					// so every row is an ordinary row for both critics.
					if (steerOn && !combinedTraj.steerPractice.empty()) {
						torch::Tensor tGroups = torch::tensor(combinedTraj.steerPractice);
						// Practice = roles 1 (steered) and 2 (control), any mode
						torch::Tensor tPractice = (tGroups > 0).to(torch::kFloat32);
						if (config.steering.resolutionTermination)
							experience.data.practiceMask = tPractice;
						report["Steer/Practice Row Frac"] = tPractice.mean().item<float>();
						report["Steer/Vector Active"] = (float)steerLoaded;
						for (int md = 0; md < PPOLearner::STEER_MODES; md++) {
							if (steerBlocks[md].numSteered == 0)
								continue;
							// Under META, the applied alpha is governed by the active
							// cluster's bench state, not the incumbent possession gate
							bool alphaOn;
							if (metaOn) {
								alphaOn = steerLoaded && !steerRatingTripped
									&& metaActiveHead >= 0 && metaActiveCluster >= 0
									&& !metaSlots[metaActiveHead][metaActiveCluster].benched
									&& metaSigma[md] != 0;
							} else {
								alphaOn = steerLoaded && steerGateActive[md] && !steerRatingTripped
									&& (steerVecEMA[md].defined() || (md > 0 && steerVecEMA[0].defined()));
							}
							std::string key = md == 0 ? "Steer/Alpha" : "Steer/Alpha " + fnModeName(md);
							report[key] = alphaOn ? config.steering.alpha : 0.f;
						}
						report["Steer/RhoGate In-Band Frac"] = ppo->lastRhoGateFrac;
						report["Steer/Rating Guard Tripped"] = (float)steerRatingTripped;
						if (!std::isnan(steerRatingEMA))
							report["Steer/Rating EMA"] = steerRatingEMA;
						if (!oppStyles.empty() && oppIters > 0)
							report["Steer/Opp Style Frac"] = (float)oppStyleIters / (float)oppIters;
					}

					// Derive/refresh the steering direction from THIS buffer's match-arena rows,
					// and feed the causal gate from the steered-vs-control practice split
					if (steerOn)
						fnSteerUpdate(report);
						fnMetaUpdate(report);
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
				// Pipelined: collection overlaps consumption, so summing the two double-counts —
				// the iteration wall clock is the honest denominator.
				report["Overall Steps/Second"] = stepsCollected /
					(pipelineOn ? RS_MAX(1e-6f, iterTimer.Elapsed()) : (collectionTime + consumptionTime));

				uint64_t prevTimesteps = totalTimesteps;
				totalTimesteps += stepsCollected;
				report["Total Timesteps"] = totalTimesteps;
				totalIterations++;
				report["Total Iterations"] = totalIterations;

				// In pipelined mode these three run in the BARRIER ZONE at the top of the next
				// iteration (same program point: tail of N == top of N+1), where the collection
				// worker is guaranteed idle — they step EnvSets on the shared thread pool and/or
				// evaluate live model weights, so they must never overlap a collecting worker.
				if (!pipelineOn) {
					if (versionMgr)
						versionMgr->OnIteration(ppo, report, totalTimesteps, prevTimesteps);
					if (report.Has(ratingKey))
						lastEvalRating = (float)report[ratingKey]; // feeds the best-checkpoint archive
					fnRatingGuard(report); // may latch steering off before the next apply

					// QD league: evolve/evaluate members between iterations (additive, off by default).
					if (league)
						league->OnIteration(report, totalIterations);

					// Basin-Racing (PSD): may run a full probe round in-line. If it does, the arenas
					// were stepped out from under the in-flight trajectories, so clear them — the next
					// iteration starts fresh episodes (a probe boundary is like a checkpoint boundary).
					if (psd) {
						float rating = report.Has(ratingKey) ? (float)report[ratingKey] : NAN;
						bool probed = psd->OnDescendIteration(report, rating, totalIterations);
						if (probed) {
							// Clear the persistent per-player trajectories so post-probe collection
							// starts fresh episodes (combinedTraj is rebuilt from these each iteration).
							for (auto& traj : trajectories)
								traj.Clear();
						}
					}
				}

				// User per-iteration hook (curriculum triggers etc.): sees the finished
				// iteration's report and may call RequestSaveAndExit, which the check just
				// below honors this same iteration.
				if (iterationCallback)
					iterationCallback(this, report);

				if (saveQueued || exitRequested) {
					// Never exit with a collection worker in flight
					if (collectThread.joinable())
						collectThread.join();
					if (!config.checkpointFolder.empty())
						Save();
					int exitCode = exitRequested ? requestedExitCode.load() : 0;
					if (exitCode != 0)
						RG_LOG("Learner: exiting with code " << exitCode
							<< " (programmatic restart request - the ops wrapper relaunches onto the saved checkpoint)");
					exit(exitCode);
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
						"League/Member Count",
						"League/Cell Count",
						"League/Lineage Count",
						"League/Quantile Bins Live",
						"League/BD Samples",
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
	delete psd;
	delete league;
	delete ppo;
	delete versionMgr;
	delete metricSender;
	delete renderSender;
	pybind11::finalize_interpreter();
}