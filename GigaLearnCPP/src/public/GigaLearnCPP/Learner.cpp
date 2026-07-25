#include "Learner.h"

#include <GigaLearnCPP/PPO/PPOLearner.h>
#include <GigaLearnCPP/PPO/ExperienceBuffer.h>

#include <torch/cuda.h>
#include <torch/mps.h>
#include <torch/serialize.h>
#include <torch/nn/utils/clip_grad.h>
#include <nlohmann/json.hpp>
#include <pybind11/embed.h>

#ifdef RG_CUDA_SUPPORT
#include <c10/cuda/CUDACachingAllocator.h>
#endif
#include <private/GigaLearnCPP/PPO/ExperienceBuffer.h>
#include <private/GigaLearnCPP/PPO/GAE.h>
#include <private/GigaLearnCPP/PolicyVersionManager.h>
#include <private/GigaLearnCPP/NextoOpponent.h>
#include <private/GigaLearnCPP/Util/Plasticity.h>
#include <private/GigaLearnCPP/League/LeagueArchive.h>

#include "Util/KeyPressDetector.h"
#include <private/GigaLearnCPP/Util/WelfordStat.h>

#include <algorithm>
#include <map>
#include <random>
#include <thread>
#include "Util/AvgTracker.h"

using namespace RLGC;

// V_exp - the return-level expectile twin of the value critic (COMPOSITION_CRITIC.md section 8,
// ladder rung 2: "what I sometimes do"). Trained on the SAME extrinsic GAE targets as the
// critic but read through a DETACHED trunk, so it measures without reshaping what it measures.
// Persisted as GAP_EXP.lt.
//
// MEASUREMENT ONLY as of 2026-07-25. The paper's mechanism actuates from the COMPOSITION
// critic alone (a single seek term, Phi = +H). Removed with the rest of the Ladder in that
// conformance pass: the quasimetric map, the goal/concede banks, V_metric calibration,
// gap_PK, the closure drive Phi = -(gap_KD + gap_PK), the 5-column policy wire, the
// impossible-control falsification family, and RND novelty.
// History: git log -- docs/LADDER.md docs/EMERGENCE.md
struct GGL::GapState {
	torch::nn::Sequential exp{ nullptr };
	std::shared_ptr<torch::optim::Adam> optim;
	std::filesystem::path loadFrom;
	int64_t updates = 0;

	void Build(int64_t featIn, torch::Device device, float lr) {
		exp = torch::nn::Sequential(
			torch::nn::Linear(featIn, 256), torch::nn::LeakyReLU(),
			torch::nn::Linear(256, 256), torch::nn::LeakyReLU(),
			torch::nn::Linear(256, 1));
		if (!loadFrom.empty() && std::filesystem::exists(loadFrom / "GAP_EXP.lt")) {
			try {
				torch::load(exp, (loadFrom / "GAP_EXP.lt").string());
				updates = 1000;
				RG_LOG("Gap sensor loaded from " << loadFrom);
			} catch (const std::exception& e) {
				RG_LOG("Gap sensor load failed (" << e.what() << ") - starting fresh");
			}
		}
		loadFrom.clear();
		exp->to(device);
		optim = std::make_shared<torch::optim::Adam>(exp->parameters(), lr);
	}
};
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

	// State shell exists from boot (net lazily built at first use so the feature width
	// comes from the real trunk output, not a config guess)
	if (config.gapSensor.enabled)
		gapSensor = std::make_shared<GapState>();

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
			if (config.league.enabled
				|| config.skillTracker.enabled || config.pipelinedCollection)
				RG_ERR_CLOSE("Learner::Learner(): standardizeObs is only supported by the plain "
					"sequential PPO path - steering/league/skillTracker/pipelinedCollection "
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

	// External fixed opponent (Nexto): fail LOUD at boot if the model is missing -
	// a mid-run lazy failure would silently turn serve iterations into self-play
	if (config.externalOpponent.enabled && !config.renderMode)
		nexto = std::make_shared<NextoOpponent>(config.externalOpponent.modelPath, device);

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

	// Rating-guard state: the latch must survive the wrapper's automatic crash-restarts
	// (NaN EMA is skipped - JSON has no NaN and it just means "unseeded").
	// Keys renamed steer_rating_* -> rating_guard_* on 2026-07-25 when the guard moved out of
	// the steering config. LoadStats reads BOTH, so a checkpoint written by either version
	// restores correctly - important because this is the latch, and losing a tripped state
	// silently re-arms six live mechanisms on a policy that already failed.
	if (!std::isnan(ratingGuardEMA))
		j["rating_guard_ema"] = ratingGuardEMA;
	if (!std::isnan(ratingGuardPeak))
		j["rating_guard_peak"] = ratingGuardPeak;

	// Churn-telemetry vector archive, per mode (save-only; see Learner.h steerVecSave).
	// 1v1 keeps the legacy un-suffixed keys.
	for (int md = 0; md < 3; md++) {
		if (steerVecSave[md].empty())
			continue;
		std::string suffix = md == 0 ? "" : "_" + std::to_string(md + 1) + "v" + std::to_string(md + 1);
		j["steer_vec" + suffix] = steerVecSave[md];
		j["steer_sigma" + suffix] = steerSigmaSave[md];
	}


	// Nexto yardstick: cumulative series must survive restarts to stay longitudinal
	if (nexto) {
		j["nexto_goals_for"] = (int64_t)nextoGoalsFor;
		j["nexto_goals_against"] = (int64_t)nextoGoalsAgainst;
		j["nexto_serve_iters"] = (int64_t)nextoServeIters;
	}

	if (versionMgr)
		versionMgr->AddRunningStatsToJSON(j);

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

	// Read the current keys, falling back to the pre-2026-07-25 steer_rating_* names so a
	// checkpoint from before the guard was re-homed still restores its latch.
	if (j.contains("rating_guard_ema"))
		ratingGuardEMA = (float)j["rating_guard_ema"];
	else if (j.contains("steer_rating_ema"))
		ratingGuardEMA = (float)j["steer_rating_ema"];

	if (j.contains("rating_guard_peak"))
		ratingGuardPeak = (float)j["rating_guard_peak"];
	else if (j.contains("steer_rating_peak"))
		ratingGuardPeak = (float)j["steer_rating_peak"];

	if (j.contains("fear_panel_obs")) {
		fearPanelObs = j["fear_panel_obs"].get<std::vector<float>>();
		fearPanelTimestep = j.value("fear_panel_timestep", (int64_t)0);
	}

	if (versionMgr)
		versionMgr->LoadRunningStatsFromJSON(j);

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
	if (gapSensor && gapSensor->exp)
		torch::save(gapSensor->exp, (saveFolder / "GAP_EXP.lt").string());
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
		if (config.gapSensor.enabled && gapSensor)
			gapSensor->loadFrom = loadFolder;

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

			// Deliberate-practice DRILL extras (proposer removed 2026-07-25; fields kept in
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
			// steerPractice = ROLE (0 match, 1 steered, 2 control, 3 ladder-impossible -
			// role 3 rows are excluded from every steering pool by the equality-keyed
			// selections); steerMode = the row's arena team size - 1
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
				}
				if (proposer || reach)
					carStateHerGoals.reserve(rows * 6); // proposer-era buffer, also fed by the car-state reach head

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
		// Car-state reachability head (META third goal space) - trains the proposer-era
		// psiCarState machinery independent of the proposer
		const bool reachCarStateOn = reachOn && config.ppo.reachability.carStateHead;

		// Deliberate-practice DRILLS (Stage 3): off unless explicitly enabled with a bank attached
		const bool goalCriticOn = config.ppo.goalCritic.enabled && !render;

		// The report key the skill tracker writes for the TRAINING arenas' team size
		// ("Rating/1v1", "Rating/2v2", ...). Every in-loop rating consumer (steering rating
		// guard, golden-archive feed, PSD plateau/freeze feed) tracks this key, so the guards
		// keep working when the fleet moves to 2v2/3v3 instead of silently going dark on a
		// stale "Rating/1v1" literal. With a mixed-mode fleet, arena 0's mode is the one
		// tracked - keep the primary mode at index 0.
		const std::string ratingKey = "Rating/" + SkillRating::GetModeName(envSet->state.gameStates[0]);

		// Arena modes (playersPerTeam - 1). The practice/control SPLIT was removed
		// 2026-07-25: the composition critic's constraint set requires homogeneous
		// environment instances (C1) and forbids banked reset states (C2), and the split
		// existed only to serve the steering treatment/control populations and the
		// frontier drill pool. Every arena is now identical.
		constexpr int STEER_MODES = 3; // index = playersPerTeam - 1
		auto fnModeName = [](int md) { return std::to_string(md + 1) + "v" + std::to_string(md + 1); };
		std::vector<uint8_t> arenaMode(envSet->arenas.size(), 0);
		for (int a = 0; a < (int)envSet->arenas.size(); a++) {
			int ppt = (int)envSet->arenas[a]->_cars.size() / 2;
			RG_ASSERT(ppt >= 1 && ppt <= STEER_MODES);
			arenaMode[a] = (uint8_t)(ppt - 1);
		}


		// Opponent style library (roadmap phase 1), synthesized LIVE - nothing on disk.
		// hesitant/overcommit are the commitment direction at negative / mild positive
		// dose; shadow is a live challenge-vs-shadow contrast (derived in fnSteerUpdate).
		// The frozen steering_styles.json era ended 2026-07-15: pinned vectors rot
		// (measured +7pp -> -11pp within ~75M steps for the commitment direction), so a
		// checkpoint-stale file is diversity in name only. Dose windows keep their
		// offline-validated values (STEERING_PHASE0_40: hesitant suppresses race-winning
		// ~2.7 sigma at -2..-1; shadow cuts challenge rate 44%->24% at -2..-1; overcommit
		// has clean canaries at +0.5..+1) and are in units of the LIVE projection sigma,
		// so they self-calibrate as the trunk drifts. The vectors the draw reads are
		// barrier-swapped copies (same discipline as ppo->steerVec): written only in
		// the collect worker joined.

		// Live per-mode steering state (derived in fnSteerUpdate during learn-prep, applied
		// in the barrier zone where no collect worker is in flight). CPU tensors.
		// Challenge-vs-shadow contrast for the live "shadow" opponent style (1v1 match
		// rows; same EMA discipline as the commitment direction)
		torch::Tensor challengeVecEMA;
		float challengeSigmaEMA = 0;
		std::array<float, STEER_MODES> steerGateDeltaEMA = {}; // steered-minus-control team-possession, EMA
		std::array<int, STEER_MODES> steerGateIters = {};      // iterations that contributed gate data
		bool steerPendingApply = false;
		// Rating watch state (ratingGuardEMA / ratingGuardPeak) lives on the
		// Learner and is persisted in the checkpoint stats - a crash-restart must not
		// silently un-latch steering (the wrapper restarts automatically and unattended)

		// META slot-ownership attribution (declared here, above fnSteerUpdate, so the
		// incumbent gate can read it). The steering slot is time-multiplexed between the
		// incumbent commitment direction and meta cluster probes (see the scheduler in
		// fnMetaUpdate); measurements must attribute each buffer to whoever actually
		// steered it: applied* = the owner set at the LAST barrier apply, measured* = the
		// owner that steered the buffer currently being processed (one apply behind in
		// pipelined mode, the same apply in sequential mode). -1 = the incumbent owns.
		bool metaOwnsActuation = false;
		int appliedMetaHead = -1, appliedMetaCluster = -1;
		int measuredMetaHead = -1, measuredMetaCluster = -1;
		// Car-free arenas for ball-landing sims, one per ad-hoc sim thread (lazy, reused).
		// Deliberately NOT on the shared thread pool: in pipelined mode learn-prep overlaps the
		// collect worker, which owns the pool.
		constexpr int STEER_SIM_THREADS = 4;
		std::vector<Arena*> steerSimArenas;

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
		combinedTraj.Reserve((size_t)config.ppo.tsPerItr + numPlayers * 4, obsSize, numActions, reachOn, false, false);

		// goalCriticOn: the goal-channel recorder below indexes these maps too - without it,
		// a goal-critic-only config would read arena 0 / slot 0 for every player and train
		// indexes playerArenaIdx as well.
		if (reachOn || goalCriticOn) {
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
			if (reachCarStateOn)
				traj.carStateHerGoals.resize((size_t)n * 6);
			for (int t = 0; t < n; t++) {
				int ballOff = fnPickOffset(t, reachCfg.ballHerMinOffset, reachCfg.ballHerMaxOffset, ballBiasPow, ballGoalwardBias);
				// Controllability is local: short window, no goalward bias
				int carOff = fnPickOffset(t, reachCfg.carHerMinOffset, reachCfg.carHerMaxOffset, carBiasPow, 0);

				for (int d = 0; d < 6; d++) {
					traj.ballHerGoals[(size_t)t * 6 + d] = traj.achievedBall[(size_t)(t + ballOff) * 6 + d];
					traj.carHerGoals[(size_t)t * 6 + d] = traj.achievedCarBall[(size_t)(t + carOff) * 6 + d];
				}

				// Car-STATE HER goal: future canonical car state, no goalward bias (car
				// states have no +y analog). Window: the head's own calibration-chosen
				// horizon (carStateHerMaxOffset; the proposer-era code used the ball
				// window - the proposer is off, this head's semantics own the choice now).
				if (reachCarStateOn) {
					int carStateOff = fnPickOffset(t, reachCfg.carStateHerMinOffset, reachCfg.carStateHerMaxOffset, ballBiasPow, 0);
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
		const bool pipelineOn = config.pipelinedCollection && !render;
		// The pipelined worker's frozen model set. With the steering rho-gate on, the reach
		// heads are snapshotted too - the worker must never read weights Learn is updating.
		std::vector<const char*> snapshotNames = { "shared_head", "policy" };
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

		// fnSyncLadderCollect freezes one full wire generation for the NEXT collection
		// (deep-cloned aux nets + bank embeddings + calibration - the pipelined worker
		// must never read nets learn-prep is training). fnArmLadderLearn hands the
		// JOINED buffer's generation to Learn(): banks/calibration stay the collection
		// snapshot, the nets switch to the LIVE handles (the spec's tested
		// re-derivation rule: current predictions against collection-time anchors).
		// Both run only in the barrier zone / sequential program points.
		auto fnCloneSeq = [](const torch::nn::Sequential& src) {
			return torch::nn::Sequential(
				std::dynamic_pointer_cast<torch::nn::SequentialImpl>(src->clone()));
		};
		// RATING WATCH - measurement only. Tracks the training mode's rating against a slow EMA
		// and a decaying high-water mark and publishes both drawdowns. There is no latch: it was
		// removed 2026-07-25 after repeatedly false-tripping and taking six live mechanisms dark.
		// This is the only signal that sees update damage when behavioral gates read green, so it
		// is kept as the operator's instrument. See RatingWatchConfig.
		auto fnRatingWatch = [&](Report& report) {
			if (!report.Has(ratingKey))
				return;
			float rating = (float)report[ratingKey];
			if (std::isnan(ratingGuardEMA)) {
				ratingGuardEMA = rating;
				ratingGuardPeak = rating;
				return;
			}
			if (std::isnan(ratingGuardPeak))
				ratingGuardPeak = rating; // resumed from a checkpoint saved before peak tracking
			// Positive = how far below the reference the rating currently sits.
			report["RatingWatch/Drawdown From EMA"] = ratingGuardEMA - rating;
			report["RatingWatch/Drawdown From Peak"] = ratingGuardPeak - rating;
			ratingGuardEMA = config.ratingWatch.emaDecay * ratingGuardEMA
				+ (1.f - config.ratingWatch.emaDecay) * rating;
			// The high-water mark decays so a stale old spike cannot dominate the panel forever
			ratingGuardPeak = RS_MAX(rating, ratingGuardPeak - config.ratingWatch.peakDecay);
		};

		Trajectory combinedTrajNext;  // the worker fills this; swapped into combinedTraj at the join
		Report collectReport;         // worker-owned between barriers; merged into the iteration report
		int collectSteps = 0;
		float collectWallTime = 0;
		uint64_t prevVersionTimesteps = totalTimesteps;


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
			// External fixed opponent (Nexto): mutually exclusive with the sources
			// below, highest-priority roll. Rows are excluded from training via the
			// same old-player split; inference routes through the adapter instead of
			// InferActions (Nexto reads GameStates, not our obs).
			bool oppExternal = false;
			std::vector<bool> oldVersionPlayerMask;
			std::vector<int> newPlayerIndices = {}, oldPlayerIndices = {};
			torch::Tensor tNewPlayerIndices, tOldPlayerIndices;
			Team oppTeam = Team::BLUE;

			for (int i = 0; i < numPlayers; i++)
				newPlayerIndices.push_back(i);

			if (!render) {
				RG_ASSERT(config.trainAgainstOldChance >= 0 && config.trainAgainstOldChance <= 1);
				if (nexto
					&& RocketSim::Math::RandFloat() < config.externalOpponent.serveFrac) {
					// covers this data-distribution intervention too
					oppExternal = true;
					nexto->BeginServe(numPlayers);
					nextoServeIters++;
				} else if (config.trainAgainstOldVersions && versionMgr && !versionMgr->versions.empty()
					&& RocketSim::Math::RandFloat() < config.trainAgainstOldChance) {
					int oldVersionIdx = RocketSim::Math::RandInt(0, versionMgr->versions.size());
					oppModels = &versionMgr->versions[oldVersionIdx].models;
				} else if (league && RocketSim::Math::RandFloat() < config.league.descendOpponentFrac) {
					// PFSP-sampled league member -> the exposure to non-self styles the league is for
					// (returns null while the archive is still empty, falling back to self-play).
					// anchors are the new intervention, so they answer to the same latch that kills
					// steering / the Ladder drive / RND. Same write-in-barrier, read-on-collect
					// discipline as ppo->steerVec.
					oppModels = league->LoadPFSPOpponentModels(true);
				}
			}

			const bool oppServed = oppModels || oppExternal;
			if (oppServed) {
				oppTeam = Team(RocketSim::Math::RandInt(0, 2));

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

			int numRealPlayers = oppServed ? newPlayerIndices.size() : envSet->state.numPlayers;

			collectSteps = 0;
			// -- Generate experience (scope brace removed: body now lives in the collect fn) --

				combinedTrajNext.ClearKeepCapacity();

				// Players handed to the opponent this iteration stop being collected; their in-flight
				// partial episode must not silently SPLICE with a later episode when they return to
				// collecting (HER goals and gate windows would cross a hidden reset). DISCARD the
				// partial (Clear) rather than finalize-and-append it: appending up to half the players'
				// in-flight episodes into this iteration's freshly-cleared buffer can exceed tsPerItr
				// and starve fresh collection entirely — the loop condition is `Length() < tsPerItr`,
				// so a pre-filled buffer collects ZERO steps (Collection SPS -> 0, no progress). The
				// discarded tails are minor, slightly-off-policy truncated data PPO doesn't want; the
				// collection loop still gathers a full tsPerItr of fresh on-policy experience.
				if (oppServed) {
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

						if (oppServed) {
							torch::Tensor tdNewStates = tStates.index_select(0, tNewPlayerIndices).to(ppo->device, true);
							torch::Tensor tdNewActionMasks = tActionMasks.index_select(0, tNewPlayerIndices).to(ppo->device, true);

							torch::Tensor tNewActions;
							torch::Tensor tOldActions;

							ppo->InferActions(tdNewStates, tdNewActionMasks, &tNewActions, &tLogProbs, collectModelsPtr);
							if (oppExternal) {
								// Nexto reads GameStates directly (its own obs builder), and
								// returns OUR action-table indices via the checked map. The
								// terminals vector still holds the PREVIOUS step's flags -
								// exactly the fresh-episode signal its prev-action memory needs.
								std::vector<int> extActions(numPlayers, 0);
								nexto->Act(envSet->state.gameStates, oldVersionPlayerMask,
									envSet->state.terminals, extActions);
								auto tExt = torch::tensor(extActions);
								tOldActions = tExt.index_select(0, tOldPlayerIndices);
							} else {
								torch::Tensor tdOldStates = tStates.index_select(0, tOldPlayerIndices).to(ppo->device, true);
								torch::Tensor tdOldActionMasks = tActionMasks.index_select(0, tOldPlayerIndices).to(ppo->device, true);
								ppo->InferActions(tdOldStates, tdOldActionMasks, &tOldActions, NULL, oppModels);
							}

							tActions = torch::zeros(numPlayers, tNewActions.dtype());
							tActions.index_copy_(0, tNewPlayerIndices, tNewActions.cpu());
							tActions.index_copy_(0, tOldPlayerIndices, tOldActions.cpu().to(tNewActions.dtype()));
						} else {
							torch::Tensor tdStates = tStates.to(ppo->device, true);
							torch::Tensor tdActionMasks = tActionMasks.to(ppo->device, true);
							ppo->InferActions(tdStates, tdActionMasks, &tActions, &tLogProbs, collectModelsPtr);
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


						// Nexto yardstick: cumulative goals for/against the external opponent
						// (RS_TEAM_FROM_Y = the CONCEDING team; conceder == Nexto -> we scored).
						// SKIP the impossible-control arenas: they spawn certified-unreachable
						// ballistic intercepts, so whatever the ball does there is a property of
						// the falsification fixture, not of play against Nexto. This series is the
						// only pool-inflation-proof yardstick in the run and must not be diluted
						// by arenas that exist to be unwinnable.
						if (oppExternal) {
							int nArenas = (int)envSet->state.gameStates.size();
							for (int arenaIdx = 0; arenaIdx < nArenas; arenaIdx++) {
								auto& gs = envSet->state.gameStates[arenaIdx];
								if (gs.goalScored) {
									if (RS_TEAM_FROM_Y(gs.ball.pos.y) == oppTeam)
										nextoGoalsFor++;
									else
										nextoGoalsAgainst++;
								}
							}
						}

						// Deliberate-practice DRILL snapshot capture (Stage 3): every snapshotEveryK
						// steps, record enough of each arena's physics state to restore play from
						// exactly here later. Serial - O(numArenas * playersPerArena), same order
						// as the arenaTeamTouched scan just above.

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
							// Ladder impossible-control rows: tagged by arena (the drive
							// injection masks them; the probe panels read them)

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
				fnRatingWatch(report); // measurement only - publishes RatingWatch/* panels
				if (league)
					league->OnIteration(report, totalIterations);
				// Freeze the current policy for the worker, then collect the next iteration
				// concurrently with this iteration's processing + Learn.
				fnSyncSnapshot();
				collectThread = std::jthread([&]() { fnCollectIteration(); });
			}


				Timer consumptionTimer = {};

				// Deliberate-practice proposer: Train() needs gradients, but the whole "Process
				// timesteps" block below runs under RG_NO_GRAD (like the reachability gate's own
				// no-grad rho reads) - so its inputs are captured here and Train() is called AFTER
				// the block closes, mirroring how ppo->Learn() itself (which also needs gradients)
				// is deferred to after this block.
				// Car head shares tPropFeatures + tPropWeights (same trunk features + aspiration
				// weights); only its regression pair differs.

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
					if (reachReadsThisIter && combinedTraj.Length() > 0) {
						Model* trunkModel = ppo->models["shared_head"];
						if (trunkModel) {
							int64_t nRows = (int64_t)combinedTraj.Length();
							int64_t chunk = (int64_t)reachCfg.scoreChunkSize;
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

					Timer gaeTimer = {};
					// Run GAE
					torch::Tensor tAdvantages, tTargetVals, tReturns, tVdagTargets;
					float rewClipPortion;
					GAE::Compute(
						tRewards, tTerminals, tValPreds, tTruncValPreds,
						tAdvantages, tTargetVals, tReturns, rewClipPortion,
						config.ppo.gaeGamma, config.ppo.gaeLambda, returnStat ? returnStat->GetSTD() : 1, config.ppo.rewardClipRange
					);
					report["GAE Time"] = gaeTimer.Elapsed();
					report["Clipped Reward Portion"] = rewClipPortion;

					// ===== HEADROOM (composition critic; PPOLearnerConfig::vdagEnabled) =====
					// TD targets in the critic's own units WITHOUT touching GAE internals:
					// scaled_r_i = A_i - g*lam*(1-d_i)*A_{i+1} - g*(1-d_i)*V_{i+1} + V_i (GAE
					// identity, pre-injection A). y_i = scaled_r_i + g*(1-d_i)*Vdag(s_{i+1}),
					// one-iteration-frozen heads = the implicit target net. Nonzero terminal
					// codes (incl. truncations) zero the bootstrap, so simple i+1 indexing is
					// boundary-safe. SEEK injection: Phi = +H, std-matched, clamped, latch-
					// covered (the closure sign is the measured avoidance pathology).
					if (config.ppo.vdagEnabled && (int64_t)combinedTraj.Length() > 1) {
						RG_NO_GRAD;
						int64_t nR = (int64_t)combinedTraj.Length();
						float g = config.ppo.gaeGamma, lmb = config.ppo.gaeLambda;
						auto advF = tAdvantages.to(torch::kFloat32).flatten();
						auto vpF = tValPreds.to(torch::kFloat32).flatten();
						auto termF = tTerminals.to(torch::kFloat32).flatten();
						auto cont = (termF == 0).to(torch::kFloat32);
						auto z1 = torch::zeros({ 1 }, advF.options());
						auto advN = torch::cat({ advF.slice(0, 1, nR), z1 });
						auto vpN = torch::cat({ vpF.slice(0, 1, nR), z1 });
						auto scaledR = advF - g * lmb * cont * advN - g * cont * vpN + vpF;
						// V-dagger on all rows (chunked trunk+head forwards)
						auto vdag = torch::empty({ nR }, torch::kFloat32);
						constexpr int64_t VCH = 32768;
						for (int64_t i0 = 0; i0 < nR; i0 += VCH) {
							int64_t i1 = RS_MIN(i0 + VCH, nR);
							vdag.slice(0, i0, i1).copy_(
								ppo->InferVdagMin(tStates.slice(0, i0, i1)).to(torch::kCPU, torch::kFloat32));
						}
						auto vdagN = torch::cat({ vdag.slice(0, 1, nR), z1 });
						float vScale = tTargetVals.abs().to(torch::kFloat32).quantile(0.99).item<float>();
						tVdagTargets = (scaledR + g * cont * vdagN)
							.clamp(-2.f * RS_MAX(vScale, 1.f), 2.f * RS_MAX(vScale, 1.f));
						// H field + seek injection
						auto tH = torch::relu(vdag - vpF);
						auto tHN = torch::cat({ tH.slice(0, 1, nR), z1 });
						auto aInt = g * cont * tHN - tH;
						aInt = aInt - aInt.mean();
						float sExt = advF.std().item<float>();
						float sInt = RS_MAX(0.05f * sExt, aInt.std().item<float>());
						auto inj = ((config.ppo.vdagSeekBeta * sExt / sInt) * aInt)
							.clamp(-3.f * sExt, 3.f * sExt);
						tAdvantages = tAdvantages + inj.view_as(tAdvantages);
						report["Headroom/Vdag Mean"] = vdag.mean().item<float>();
						report["Headroom/H Mean"] = tH.mean().item<float>();
						report["Headroom/H P90"] = tH.quantile(0.9).item<float>();
						report["Headroom/Inj Abs Mean"] = inj.abs().mean().item<float>();
					}

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

					// Car shaping term: identical centered/std-matched injection, own beta. stdExt
					// is recomputed against the (possibly ball-shaped) advantages so the two terms
					// compose to roughly shapingBeta + carShapingBeta of the extrinsic std.

					// Nexto yardstick panels: cumulative counters (persisted in stats), the
					// fixed external benchmark immune to version-pool inflation
					if (nexto) {
						report["Nexto/Goals For"] = (float)(int64_t)nextoGoalsFor;
						report["Nexto/Goals Against"] = (float)(int64_t)nextoGoalsAgainst;
						report["Nexto/Serve Iters"] = (float)(int64_t)nextoServeIters;
					}

					// ===== OPTIMISTIC-CRITIC LADDER (LADDER.md; full build 2026-07-18) =====
					// Stage 1: expectile sensor (detached twin, GAE targets) -> gap_KD.
					// Rung 3: quasimetric map (own optimizer, QRL local/spread + dual
					// lambda) + goal/concede banks -> V_metric -> gap_PK. Drive: potential
					// Phi = -(gap_KD + gap_PK) into the advantages, masked (terminals,
					// truncations, impossible-control rows), centered OVER UNMASKED ROWS
					// ONLY and re-masked (the spec's corrected form - masked rows pay
					// literally zero), std-floored beta_eff, clamped +-3 sigma_ext.
					// Value/expectile targets were computed BEFORE any injection (the
					// sensor never measures its own payments).
					if (gapSensor && (int64_t)combinedTraj.Length() > 0 && tTargetVals.defined()) {
						const auto& gc = config.gapSensor;
						Timer gapTimer = {};
						int64_t nAll = combinedTraj.Length();
						int64_t chunk = RS_MAX((int64_t)8192, (int64_t)ppo->config.miniBatchSize);
						auto tTgt = tTargetVals.to(torch::kFloat32).flatten();

						// train one subsample pass (grad locally re-enabled; expectile loss)
						{
							torch::AutoGradMode _gapGradOn(true);
							auto perm = torch::randperm(nAll,
								torch::TensorOptions().dtype(torch::kLong))
								.slice(0, 0, RS_MIN((int64_t)gc.trainRows, nAll));
							float lossSum = 0; int lossN = 0;
							for (int64_t i = 0; i < perm.size(0); i += chunk) {
								auto idx = perm.slice(0, i, RS_MIN(i + chunk, perm.size(0)));
								torch::Tensor h2;
								{
									RG_NO_GRAD;
									h2 = ppo->models["shared_head"]->Forward(
										tStates.index_select(0, idx).to(ppo->device, true), false);
								}
								if (!gapSensor->exp)
									gapSensor->Build(h2.size(1), ppo->device, gc.lr);
								gapSensor->optim->zero_grad();
								auto pred = gapSensor->exp->forward(h2).flatten();
								auto u = tTgt.index_select(0, idx).to(ppo->device) - pred;
								auto w = torch::where(u > 0,
									torch::full_like(u, gc.tau), torch::full_like(u, 1.f - gc.tau));
								auto loss = (w * u * u).mean();
								loss.backward();
								gapSensor->optim->step();
								lossSum += loss.item<float>(); lossN++;
							}
							gapSensor->updates++;
							if (lossN > 0)
								report["Gap/Loss"] = lossSum / lossN;
						}

						// buffer-wide gap panels (no-grad, chunked, subsampled)
						if (gapSensor->exp && gapSensor->updates >= 5) {
							RG_NO_GRAD;
							int64_t sample = RS_MIN((int64_t)65536, nAll);
							auto idx = torch::randperm(nAll,
								torch::TensorOptions().dtype(torch::kLong)).slice(0, 0, sample);
							auto h2 = ppo->models["shared_head"]->Forward(
								tStates.index_select(0, idx).to(ppo->device, true), false);
							auto vExp = gapSensor->exp->forward(h2).flatten().cpu();
							auto vReal = tValPreds.to(torch::kFloat32).flatten().index_select(0, idx);
							auto gap = torch::relu(vExp - vReal);
							report["Gap/Mean"] = gap.mean().item<float>();
							report["Gap/P90"] = gap.quantile(0.9).item<float>();
							report["Gap/VExp Mean"] = vExp.mean().item<float>();
							report["Gap/VReal Mean"] = vReal.mean().item<float>();

							// the bridge: gap on the frozen fear-panel states (+ the ladder's
							// gap_PK on the same feasible-frontier states - the comparator the
							// impossible family's acceptance criterion is judged against)
							if (!fearPanelObs.empty() && obsSize > 0
								&& fearPanelObs.size() % (size_t)obsSize == 0) {
								int64_t k = (int64_t)(fearPanelObs.size() / (size_t)obsSize);
								torch::Tensor pObs = torch::from_blob(fearPanelObs.data(),
									{ k, (int64_t)obsSize }, torch::kFloat32).to(ppo->device);
								auto ph2 = ppo->models["shared_head"]->Forward(pObs, false);
								auto pExp = gapSensor->exp->forward(ph2).flatten();
								auto pReal = ppo->InferCritic(pObs).to(torch::kFloat32).flatten();
								report["Gap/Fear Panel"] =
									torch::relu(pExp - pReal).mean().item<float>();
							}
						}

						report["Gap/Time"] = gapTimer.Elapsed();
					}

					// Set experience buffer
					experience.data.actions = tActions;
					experience.data.logProbs = tLogProbs;
					experience.data.actionMasks = tActionMasks;
					experience.data.states = tStates;
					experience.data.advantages = tAdvantages;
					experience.data.targetValues = tTargetVals;
					if (tVdagTargets.defined())
						experience.data.vdagTargets = tVdagTargets;
					if (goalCriticOn)
						experience.data.goalTargetValues = tGoalTargetVals;

					if (reachOn) {
						experience.data.carHerGoals = torch::tensor(combinedTraj.carHerGoals).reshape({ -1, 6 });
						experience.data.ballHerGoals = torch::tensor(combinedTraj.ballHerGoals).reshape({ -1, 6 });
						experience.data.ballMovedMask = torch::tensor(combinedTraj.ballMoved);
						if (reachCarStateOn && !combinedTraj.carStateHerGoals.empty())
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
					fnRatingWatch(report); // measurement only - publishes RatingWatch/* panels

					// QD league: evolve/evaluate members between iterations (additive, off by default).
					if (league)
						league->OnIteration(report, totalIterations);

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
						// HEADROOM had NO console presence, despite the boot banner promising
						// "Headroom/* panels" and the seek term injecting ~0.15 sigma into
						// advantages - which is how 16.1M frozen params went unnoticed for the
						// life of the run. Update Magnitude is the load-bearing one: 0 means the
						// twins are not training and the injection is a random projection.
						"Headroom/Vdag Update Magnitude",
						"Headroom/Vdag Loss",
						"Headroom/H Mean",
						"Headroom/Inj Abs Mean",
						"",
						// The update-damage signal. Nothing acts on these any more (the latch was
						// removed 2026-07-25) - they are here so a drawdown is visible to the
						// operator on the console, not just in wandb.
						"RatingWatch/Drawdown From EMA",
						"RatingWatch/Drawdown From Peak",
						"",
						"Plasticity/Trunk EffRank",
						"Plasticity/Policy EffRank",
						"Plasticity/Policy Dead Units",
						"",
						"Gap/Loss",
						"Gap/Mean",
						"Gap/Fear Panel",
						"Gap/Drive Inj Abs Mean",
						"Ladder/Map Local Loss",
						"Ladder/Lambda",
						"Ladder/Bank Goal Fill",
						"Ladder/Bank Concede Fill",
						"Ladder/Calib A",
						"Ladder/Calib A2",
						"Ladder/GapPK Mean",
						"Ladder/Fear GapPK",
						"Ladder/Imp GapPK Spawn",
						"Ladder/Imp Touches",
						"Ladder/Wire Active",
						"Ladder/Wire Col Grad",
						"Ladder/Retention Viol",
						"",
						"Nexto/Goals For",
						"Nexto/Goals Against",
						"Nexto/Serve Iters",
						"",
						"League/Member Count",
						"League/Cell Count",
						"League/Lineage Count",
						// Anchor opponents (LEAGUE_ANCHORS.md): Anchor Serves rising is the
						// only proof the reallocated 0.05 slice is actually being played;
						// Anchor Count 0 means the anchor dir is missing/empty (feature inert).
						"League/Anchor Count",
						"League/Anchor Serves",
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
	delete league;
	delete ppo;
	delete versionMgr;
	delete metricSender;
	delete renderSender;
	pybind11::finalize_interpreter();
}