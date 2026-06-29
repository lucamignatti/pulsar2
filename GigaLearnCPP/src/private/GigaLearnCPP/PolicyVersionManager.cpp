#include "PolicyVersionManager.h"
#include <nlohmann/json.hpp>
#include <cmath>

#include <GigaLearnCPP/Util/Utils.h>

#include <RLGymCPP/StateSetters/FuzzedKickoffState.h>
#include <RLGymCPP/TerminalConditions/GoalScoreCondition.h>

#include <private/GigaLearnCPP/PPO/PPOLearner.h>

using namespace nlohmann;

GGL::PolicyVersionManager::PolicyVersionManager(
	std::filesystem::path saveFolder, int maxVersions, uint64_t tsPerVersion, 
	const SkillTrackerConfig& skillTrackerConfig, const RLGC::EnvSetConfig& envSetConfig, RenderSink* renderSender) :
	saveFolder(saveFolder), maxVersions(maxVersions), tsPerVersion(tsPerVersion), 
	renderSender(renderSender) {

	skill.config = skillTrackerConfig;

	if (maxVersions <= 0)
		RG_ERR_CLOSE("PolicyVersionManager maxVersions must be positive");
	if (tsPerVersion == 0)
		RG_ERR_CLOSE("PolicyVersionManager tsPerVersion must be positive");

	if (!std::filesystem::exists(saveFolder))
		std::filesystem::create_directories(saveFolder);

	if (skill.config.enabled) {
		if (skill.config.numArenas <= 0)
			RG_ERR_CLOSE("SkillTrackerConfig::numArenas must be positive");
		if (skill.config.simTime <= 0 || !std::isfinite(skill.config.simTime))
			RG_ERR_CLOSE("SkillTrackerConfig::simTime must be positive and finite");
		if (skill.config.maxSimTime <= 0 || !std::isfinite(skill.config.maxSimTime))
			RG_ERR_CLOSE("SkillTrackerConfig::maxSimTime must be positive and finite");
		if (skill.config.updateInterval <= 0)
			RG_ERR_CLOSE("SkillTrackerConfig::updateInterval must be positive");
		if (!std::isfinite(skill.config.ratingInc))
			RG_ERR_CLOSE("SkillTrackerConfig::ratingInc must be finite");
		if (!std::isfinite(skill.config.initialRating))
			RG_ERR_CLOSE("SkillTrackerConfig::initialRating must be finite");

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

GGL::PolicyVersion& GGL::PolicyVersionManager::AddVersion(ModelSet modelsToClone, uint64_t timesteps, bool allowPrune) {
	RG_NO_GRAD;

	auto models = modelsToClone.CloneAll();

	auto newVersion = PolicyVersion{
		timesteps,
		models
	};

	newVersion.ratings = skill.curRatings;

	versions.push_back(newVersion);

	SortVersions();

	if (allowPrune)
		PruneVersions();

	// SortVersions keeps versions ascending by timesteps, so the just-added newest is at the back
	// (it cannot have been pruned: PruneVersions only evicts the OLDEST non-anchors).
	return versions.back();
}

float GGL::PolicyVersionManager::VersionRating(const PolicyVersion& v) const {
	if (v.ratings.data.empty())
		return skill.config.initialRating;
	float sum = 0;
	for (const auto& pair : v.ratings.data)
		sum += pair.second;
	return sum / (float)v.ratings.data.size();
}

void GGL::PolicyVersionManager::PruneVersions() {
	// Anchors are exempt from the rolling prune; cap only the NON-anchor (rolling-recent) population
	// at maxVersions, always evicting the oldest non-anchor first (versions stay sorted ascending by ts).
	RG_NO_GRAD;
	while (true) {
		int nonAnchorCount = 0;
		for (auto& v : versions)
			if (!v.isAnchor)
				nonAnchorCount++;
		if (nonAnchorCount <= maxVersions)
			break;

		for (auto it = versions.begin(); it != versions.end(); ++it) {
			if (!it->isAnchor) {
				it->models.Free();
				versions.erase(it);
				break;
			}
		}
	}
}

void GGL::PolicyVersionManager::MaybeUpdateAnchors(uint64_t totalTimesteps) {
	if (maxAnchors <= 0 || versions.empty())
		return;

	// Candidate = the newest version (just added; == current policy strength via inherited ratings).
	// Don't anchor the near-random early policy: require enough training before the first anchor.
	PolicyVersion& cand = versions.back();
	if (cand.isAnchor || cand.timesteps < anchorMinTsSpacing)
		return;

	// Must be temporally spaced from the most-recent existing anchor (keeps anchors behaviorally distinct).
	uint64_t newestAnchorTs = 0;
	int anchorCount = 0, weakestIdx = -1;
	float weakestRating = 0;
	for (int i = 0; i < (int)versions.size(); i++) {
		if (!versions[i].isAnchor)
			continue;
		anchorCount++;
		newestAnchorTs = RS_MAX(newestAnchorTs, versions[i].timesteps);
		float r = VersionRating(versions[i]);
		if (weakestIdx < 0 || r < weakestRating) {
			weakestRating = r;
			weakestIdx = i;
		}
	}

	if (anchorCount > 0 && cand.timesteps < newestAnchorTs + anchorMinTsSpacing)
		return;

	if (anchorCount < maxAnchors) {
		cand.isAnchor = true;
		RG_LOG("PolicyVersionManager: promoted version @" << cand.timesteps << " ts to anchor (" << (anchorCount + 1) << "/" << maxAnchors << ", rating " << VersionRating(cand) << ")");
	} else if (VersionRating(cand) > weakestRating + anchorPromoteMargin) {
		versions[weakestIdx].isAnchor = false; // demoted anchor rejoins the rolling pool (may be pruned next add)
		cand.isAnchor = true;
		RG_LOG("PolicyVersionManager: anchor swap -- version @" << cand.timesteps << " (rating " << VersionRating(cand) << ") replaces weakest anchor (rating " << weakestRating << ")");
	}
}

GGL::PolicyVersion* GGL::PolicyVersionManager::PickOldVersion() {
	if (versions.empty())
		return nullptr;

	std::vector<int> anchorIdx, rollingIdx;
	for (int i = 0; i < (int)versions.size(); i++)
		(versions[i].isAnchor ? anchorIdx : rollingIdx).push_back(i);

	bool useAnchor = !anchorIdx.empty() && (rollingIdx.empty() || Math::RandFloat() < anchorSelectChance);
	auto& pool = useAnchor ? anchorIdx : rollingIdx;
	int idx = pool[Math::RandInt(0, (int)pool.size())];
	return &versions[idx];
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
			j["is_anchor"] = version.isAnchor;
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
		// Defer pruning until all versions are loaded and their anchor flags are restored, so an
		// anchor checkpoint can't be evicted by the rolling prune before we know it's an anchor.
		PolicyVersion& version = AddVersion(modelsTemplate, savedTimesteps, false);
		version.models.Load(path, false, false);

		{ // Load JSON
			// TODO: Repetitive
			auto jsonPath = path / "STATS.json";
			std::ifstream fIn(jsonPath);
			RG_ASSERT(fIn.good());

			json j = json::parse(fIn);
			if (j.contains("skill_ratings"))
				version.ratings.ReadFromJSON(j["skill_ratings"]);
			version.isAnchor = j.value("is_anchor", false);
		}
	}

	SortVersions();
	PruneVersions();

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

		torch::Tensor tNewActions, tOldActions;
		torch::Tensor _tLogProbs;

		// RSNorm "canonical stats everywhere": normalize both sides with the live
		// normalizer (the obs here are raw env observations).
		PPOLearner::InferActionsFromModels(
			ppo->models, tNewStates.to(ppo->device, true), tNewActionMasks.to(ppo->device, true),
			skill.config.deterministic, ppo->config.policyTemperature, ppo->config.useHalfPrecision,
			&tNewActions, &_tLogProbs, NULL, ppo->obsNorm);
		PPOLearner::InferActionsFromModels(
			oldVersion.models, tOldStates.to(ppo->device, true), tOldActionMasks.to(ppo->device, true),
			skill.config.deterministic, ppo->config.policyTemperature, ppo->config.useHalfPrecision,
			&tOldActions, &_tLogProbs, NULL, ppo->obsNorm);

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
		MaybeUpdateAnchors(totalTimesteps);
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
