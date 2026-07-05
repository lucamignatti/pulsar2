#include "PolicyVersionManager.h"
#include <nlohmann/json.hpp>

#include <GigaLearnCPP/Util/Utils.h>

#include <RLGymCPP/StateSetters/FuzzedKickoffState.h>
#include <RLGymCPP/TerminalConditions/GoalScoreCondition.h>

#include <private/GigaLearnCPP/PPO/PPOLearner.h>

using namespace nlohmann;

GGL::PolicyVersionManager::PolicyVersionManager(
	std::filesystem::path saveFolder, int maxVersions, uint64_t tsPerVersion, 
	const SkillTrackerConfig& skillTrackerConfig, const RLGC::EnvSetConfig& envSetConfig, RenderSender* renderSender) : 
	saveFolder(saveFolder), maxVersions(maxVersions), tsPerVersion(tsPerVersion), 
	renderSender(renderSender) {

	skill.config = skillTrackerConfig;

	if (!std::filesystem::exists(saveFolder))
		std::filesystem::create_directories(saveFolder);

	if (skill.config.enabled) {
		RLGC::EnvSetConfig skillEnvSetConfig = envSetConfig;
		skillEnvSetConfig.numArenas = skill.config.numArenas;
		skill.envSet = new RLGC::EnvSet(skillEnvSetConfig);
		for (int i = 0; i < skill.envSet->arenas.size(); i++) {
			skill.envSet->rewards[i].clear();
			skill.envSet->stateSetters[i] = { new RLGC::FuzzedKickoffState() };
			skill.envSet->terminalConditions[i] = { new RLGC::GoalScoreCondition() };
		}
	} else {
		skill.envSet = NULL;
	}
}

GGL::PolicyVersion& GGL::PolicyVersionManager::AddVersion(ModelSet modelsToClone, uint64_t timesteps) {
	RG_NO_GRAD;

	auto models = modelsToClone.CloneAll();

	auto newVersion = PolicyVersion{
		timesteps,
		models
	};

	newVersion.ratings = skill.curRatings;

	versions.push_back(newVersion);

	SortVersions();

	// Remove old versions
	while (versions.size() > maxVersions) {
		auto& toRemove = versions[0];
		toRemove.models.Free();
		versions.erase(versions.begin());
	}

	return versions.back();
}

void GGL::PolicyVersionManager::SaveVersions() {
	RG_NO_GRAD;

	// Remove old saved versions
	std::set<int64_t> allSavedTimesteps = Utils::FindNumberedDirs(saveFolder);

	for (int64_t savedTimesteps : allSavedTimesteps) {
		bool matchesVersion = false;
		for (auto& version : versions)
			matchesVersion |= (savedTimesteps == version.timesteps);

		if (matchesVersion) {
			// We want to keep this
			allSavedTimesteps.insert(savedTimesteps);
		} else {
			// Get rid of it
			std::filesystem::remove_all(saveFolder / std::to_string(savedTimesteps));
		}
	}

	for (auto& version : versions) {
		if (allSavedTimesteps.contains(version.timesteps))
			continue;
		auto versionSaveFolder = saveFolder / std::to_string(version.timesteps);
		std::filesystem::create_directories(versionSaveFolder);

		version.models.Save(versionSaveFolder, false);

		{ // Save JSON
			auto jsonPath = versionSaveFolder / "STATS.json";

			std::ofstream fOut(jsonPath);
			RG_ASSERT(fOut.good());

			json j = {};
			j["skill_ratings"] = version.ratings.ToJSON();
			std::string jStr = j.dump(4);
			fOut << jStr;
		}
	}
}

void GGL::PolicyVersionManager::LoadVersions(ModelSet modelsTemplate, uint64_t curTimesteps) {

	RG_NO_GRAD;

	RG_LOG("PolicyVersionManager::LoadVersions():");

	for (auto& version : versions)
		version.models.Free();
	versions.clear();

	std::set<int64_t> allSavedTimesteps = Utils::FindNumberedDirs(saveFolder);

	for (int64_t savedTimesteps : allSavedTimesteps) {

		if (savedTimesteps > curTimesteps) {
			RG_ERR_CLOSE(
				"Tried to load saved policy version that is newer than our current model (" << savedTimesteps << " > " << curTimesteps << ")!\n" <<
				"If you deleted some checkpoints, make sure to delete that far back in the saved policy versions as well");
		}
		auto path = saveFolder / std::to_string(savedTimesteps);
		PolicyVersion& version = AddVersion(modelsTemplate, savedTimesteps);
		version.models.Load(path, false, false);

		{ // Load JSON
			// TODO: Repetitive
			auto jsonPath = path / "STATS.json";
			std::ifstream fIn(jsonPath);
			RG_ASSERT(fIn.good());

			json j = json::parse(fIn);
			if (j.contains("skill_ratings"))
				version.ratings.ReadFromJSON(j["skill_ratings"]);
		}
	}

	SortVersions();

	RG_LOG(" > Loaded " << versions.size() << " versions(s)");
}

void GGL::PolicyVersionManager::SortVersions() {
	auto fnCompareVersions = [](const PolicyVersion& a, const PolicyVersion& b) {
		return a.timesteps < b.timesteps;
	};

	std::sort(versions.begin(), versions.end(), fnCompareVersions);
}

/////////////////////////////////////////////////////////////////////

void GGL::PolicyVersionManager::RunSkillMatches(PPOLearner* ppo, Report& report) {
	RG_NO_GRAD;
	
	auto fnUpdateRatings = [this](SkillRating& winner, SkillRating& loser, RLGC::GameState& state) {
		float& winnerRating = winner.GetRating(state, skill.config.initialRating);
		float& loserRating = loser.GetRating(state, skill.config.initialRating);
		
		// Update according to ELO math
		float expDelta = (loserRating - winnerRating) / 400;
		float expected = 1 / (powf(10, expDelta) + 1);

		winnerRating += skill.config.ratingInc * (1 - expected);
		loserRating += skill.config.ratingInc * (expected - 1);
	};

	
	Team newTeam;
	int oldVersionIndex;
	float totalSimTime;
	if (skill.doContinuation) {
		RG_ASSERT(skill.prevOldVersionIndex < versions.size());
		oldVersionIndex = skill.prevOldVersionIndex;
		newTeam = skill.prevNewTeam;
		totalSimTime = skill.prevSimTime;
	} else {
		oldVersionIndex = Math::RandInt(0, versions.size());
		newTeam = (Team)Math::RandInt(0, 2);
		totalSimTime = 0;

		skill.envSet->Reset();
	}
	skill.doContinuation = false;

	auto& oldVersion = versions[oldVersionIndex];

	// Find which players are on which teams
	std::vector<int>
		newPlayers = {}, 
		oldPlayers = {};
	for (int i = 0; i < skill.envSet->arenas.size(); i++) {
		auto& state = skill.envSet->state.gameStates[i];
		for (int j = 0; j < state.players.size(); j++) {
			int playerIdx = skill.envSet->state.arenaPlayerStartIdx[i] + j;
			bool isNew = (state.players[j].team == newTeam);
			(isNew ? newPlayers : oldPlayers).push_back(playerIdx);
		}
	}

	torch::Tensor
		tNewPlayers = torch::tensor(newPlayers),
		tOldPlayers = torch::tensor(oldPlayers);

	// 2.2 goal-conditioned worker: the banked policies were saved with a goalDim-widened policy head,
	// so the skill matches must run the SAME live per-step proposer goal walk and feed each side its
	// goal (else the head is forwarded goalless -> shape crash). Uses the LIVE proposer (versions
	// don't carry it) + live trunk, exactly like collection; both new and old policy consume the same
	// per-player goal. goalDim==0 (2.1) => all skipped, original goalless path.
	const bool goalModel = ppo->goalDim > 0;
	const bool goalHasCar = ppo->goalDim > 6;
	const auto& reachCfg = ppo->config.reachability;
	const int numSkillPlayers = skill.envSet->state.numPlayers;
	std::vector<float> skillGoalBall(goalModel ? numSkillPlayers * 6 : 0, 0.f);
	std::vector<float> skillGoalCar(goalHasCar ? numSkillPlayers * 6 : 0, 0.f);
	std::vector<uint8_t> skillGoalAtEpStart(goalModel ? numSkillPlayers : 0, 1);

	// Seed g_{-1} = current achieved state (canonical ball + car), same frame/normalization as the
	// Learner's fnSeedGoal, written straight into the per-player goal buffers by global player index.
	auto fnSeedSkillGoal = [&](int arenaIdx, int slot, int globalIdx) {
		const auto& gs = skill.envSet->state.gameStates[arenaIdx];
		const auto& player = gs.players[slot];
		float sign = (player.team == Team::ORANGE) ? -1.f : 1.f;
		float* b = &skillGoalBall[(size_t)globalIdx * 6];
		b[0] = sign * gs.ball.pos.x / reachCfg.posScaleX;
		b[1] = sign * gs.ball.pos.y / reachCfg.posScaleY;
		b[2] = gs.ball.pos.z / reachCfg.posScaleZ;
		b[3] = sign * gs.ball.vel.x / reachCfg.velScale;
		b[4] = sign * gs.ball.vel.y / reachCfg.velScale;
		b[5] = gs.ball.vel.z / reachCfg.velScale;
		if (goalHasCar) {
			float* c = &skillGoalCar[(size_t)globalIdx * 6];
			c[0] = sign * player.pos.x / reachCfg.posScaleX;
			c[1] = sign * player.pos.y / reachCfg.posScaleY;
			c[2] = player.pos.z / reachCfg.posScaleZ;
			c[3] = sign * player.vel.x / reachCfg.velScale;
			c[4] = sign * player.vel.y / reachCfg.velScale;
			c[5] = player.vel.z / reachCfg.velScale;
		}
	};

	int newGoals = 0, oldGoals = 0;

	RG_LOG("Running skill matches (simTime=" << skill.config.simTime << ")...");

	SkillRating prevCurRatings = skill.curRatings;

	float stepTime = skill.envSet->config.tickSkip * RLGC::CommonValues::TICK_TIME;
	for (float t = 0; 
		t < skill.config.simTime && totalSimTime < skill.config.maxSimTime && skill.curGoals < skill.envSet->arenas.size();
		t += stepTime, totalSimTime += stepTime) {

		skill.envSet->Reset();

		torch::Tensor tStates = DIMLIST2_TO_TENSOR<float>(skill.envSet->state.obs);
		torch::Tensor tActionMasks = DIMLIST2_TO_TENSOR<uint8_t>(skill.envSet->state.actionMasks);

		torch::Tensor tNewStates = tStates.index_select(0, tNewPlayers);
		torch::Tensor tOldStates = tStates.index_select(0, tOldPlayers);

		torch::Tensor tNewActionMasks = tActionMasks.index_select(0, tNewPlayers);
		torch::Tensor tOldActionMasks = tActionMasks.index_select(0, tOldPlayers);

		skill.envSet->StepFirstHalf(true);

		// 2.2: step the live proposer goal for every player, split to the two sides. Both policies
		// consume the same live goal (the old version's own trunk feeds its head; the goal is shared).
		torch::Tensor goalCatNew, goalCatOld;
		if (goalModel) {
			for (int a = 0; a < skill.envSet->arenas.size(); a++) {
				auto& gs = skill.envSet->state.gameStates[a];
				for (int j = 0; j < (int)gs.players.size(); j++) {
					int gidx = skill.envSet->state.arenaPlayerStartIdx[a] + j;
					if (skillGoalAtEpStart[gidx])
						fnSeedSkillGoal(a, j, gidx);
				}
			}
			torch::Tensor liveTrunk = ppo->models["shared_head"]
				? ppo->models["shared_head"]->Forward(tStates.to(ppo->device, true), false)
				: tStates.to(ppo->device, true);
			torch::Tensor gBall = ppo->proposer->StepGoal(liveTrunk,
				torch::from_blob(skillGoalBall.data(), { (int64_t)numSkillPlayers, 6 }, torch::kFloat32).to(ppo->device, true));
			torch::Tensor gCar;
			if (goalHasCar)
				gCar = ppo->proposerCar->StepGoal(liveTrunk,
					torch::from_blob(skillGoalCar.data(), { (int64_t)numSkillPlayers, 6 }, torch::kFloat32).to(ppo->device, true));
			torch::Tensor goalCat = goalHasCar ? torch::cat({ gBall, gCar }, -1) : gBall;
			goalCatNew = goalCat.index_select(0, tNewPlayers.to(ppo->device, true));
			goalCatOld = goalCat.index_select(0, tOldPlayers.to(ppo->device, true));

			// Advance the recurrence (g_{t-1} <- g_t) for the next step
			skillGoalBall = TENSOR_TO_VEC<float>(gBall.reshape({ -1 }).cpu());
			if (goalHasCar)
				skillGoalCar = TENSOR_TO_VEC<float>(gCar.reshape({ -1 }).cpu());
		}

		torch::Tensor tNewActions, tOldActions;
		torch::Tensor _tLogProbs;

		// fp32 when goal-conditioned (goalCat is fp32); else keep the configured half precision.
		bool skillHalfPrec = goalModel ? false : ppo->config.useHalfPrecision;
		PPOLearner::InferActionsFromModels(
			ppo->models, tNewStates.to(ppo->device, true), tNewActionMasks.to(ppo->device, true),
			skill.config.deterministic, ppo->config.policyTemperature, skillHalfPrec,
			&tNewActions, &_tLogProbs, goalCatNew);
		PPOLearner::InferActionsFromModels(
			oldVersion.models, tOldStates.to(ppo->device, true), tOldActionMasks.to(ppo->device, true),
			skill.config.deterministic, ppo->config.policyTemperature, skillHalfPrec,
			&tOldActions, &_tLogProbs, goalCatOld);

		auto newActions = TENSOR_TO_VEC<int>(tNewActions);
		auto oldActions = TENSOR_TO_VEC<int>(tOldActions);

		auto combinedActions = std::vector<int>(skill.envSet->state.numPlayers, -1);
		for (int i = 0; i < newActions.size(); i++)
			combinedActions[newPlayers[i]] = newActions[i];
		for (int i = 0; i < oldActions.size(); i++)
			combinedActions[oldPlayers[i]] = oldActions[i];

		skill.envSet->Sync();
		skill.envSet->StepSecondHalf(combinedActions, false);

		for (int i = 0; i < skill.envSet->arenas.size(); i++) {
			auto& gs = skill.envSet->state.gameStates[i];
			if (gs.goalScored) {
				std::string modeName = SkillRating::GetModeName(gs);

				if (RS_TEAM_FROM_Y(gs.ball.pos.y) != newTeam) {
					fnUpdateRatings(skill.curRatings, oldVersion.ratings, gs);
				} else {
					fnUpdateRatings(oldVersion.ratings, skill.curRatings, gs);
				}

				skill.curGoals++;
			}
		}

		// 2.2: flag players whose arena ended so they reseed g_{-1} after the next-iteration Reset()
		if (goalModel)
			for (int a = 0; a < skill.envSet->arenas.size(); a++) {
				uint8_t term = skill.envSet->state.terminals[a];
				int start = skill.envSet->state.arenaPlayerStartIdx[a];
				int n = (int)skill.envSet->state.gameStates[a].players.size();
				for (int j = 0; j < n; j++)
					skillGoalAtEpStart[start + j] = term != 0;
			}

		if (renderSender)
			renderSender->Send(skill.envSet->state.gameStates[0]);
	}

	for (auto& pair : skill.curRatings.data) {
		float prevRating = prevCurRatings.GetRating(pair.first, skill.config.initialRating);
		float delta = pair.second - prevRating;

		std::stringstream ratingLine;
		ratingLine << " > " << pair.first << " = " << prevRating;
		if (delta != 0)
			ratingLine << " (" << (delta >= 0 ? '+' : '-') << abs(delta) << ")";

		RG_LOG(" > " << ratingLine.str());

		report["Rating/" + pair.first] = pair.second;
	}

	if (skill.curGoals < skill.envSet->arenas.size() && totalSimTime < skill.config.maxSimTime) {
		// Not enough goals were scored, we will force a continuation where the same models keep playing from the end position
		// This COULD switch versions, but it wouldn't be a big deal as it would just be the immediate next version
		RG_LOG(" > Forcing continuation (" << skill.curGoals <<  "/" << skill.envSet->arenas.size() << ")");
		skill.doContinuation = true;
		skill.prevOldVersionIndex = oldVersionIndex;
		skill.prevNewTeam = newTeam;
		skill.prevSimTime = totalSimTime;
	} else {
		skill.curGoals = 0;
	}
}

void GGL::PolicyVersionManager::OnIteration(struct PPOLearner* ppo, Report& report, int64_t totalTimesteps, int64_t prevTotalTimesteps) {
	if ((totalTimesteps / tsPerVersion > prevTotalTimesteps / tsPerVersion) || (prevTotalTimesteps == 0)) {
		// Save version
		AddVersion(ppo->GetPolicyModels(), totalTimesteps);
	}

	if (skill.config.enabled) {
		skill.iterationsSinceRan++;
		if (skill.iterationsSinceRan >= skill.config.updateInterval && !versions.empty()) {
			skill.iterationsSinceRan = 0;
			RunSkillMatches(ppo, report);
		}
	}
}

void GGL::PolicyVersionManager::AddRunningStatsToJSON(nlohmann::json& json) {
	if (skill.config.enabled)
		json["skill_ratings"] = skill.curRatings.ToJSON();
}

void GGL::PolicyVersionManager::LoadRunningStatsFromJSON(const nlohmann::json& json) {
	if (skill.config.enabled)
		if (json.contains("skill_ratings"))
			skill.curRatings.ReadFromJSON(json["skill_ratings"]);
}