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
		if (skill.config.envCreateFn)
			skillEnvSetConfig.envCreateFn = skill.config.envCreateFn;
		skill.envSet = new RLGC::EnvSet(skillEnvSetConfig);
		for (int i = 0; i < skill.envSet->arenas.size(); i++) {
			skill.envSet->rewards[i].clear();
			skill.envSet->stateSetters[i] = { new RLGC::FuzzedKickoffState() };
			skill.envSet->terminalConditions[i] = { new RLGC::GoalScoreCondition() };
		}

		// The reference battery gets its OWN arenas so its matches cannot stomp the rating
		// matches' continuation bookkeeping (which is stateful across calls). Not new cost:
		// this is the 16-arena eval fleet the deleted league match env used to own.
		if (skill.config.maxReferences > 0) {
			refBattery.envSet = new RLGC::EnvSet(skillEnvSetConfig);
			for (int i = 0; i < refBattery.envSet->arenas.size(); i++) {
				refBattery.envSet->rewards[i].clear();
				refBattery.envSet->stateSetters[i] = { new RLGC::FuzzedKickoffState() };
				refBattery.envSet->terminalConditions[i] = { new RLGC::GoalScoreCondition() };
			}
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
	if (dist && dist->rank() != 0)
		return;
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
		if (allSavedTimesteps.contains(version.timesteps)) {
			// Models are immutable but the version's Elo evolves with every skill eval:
			// re-persist STATS.json (tmp + rename) or every restart snaps the opponent
			// pool back to creation-time ratings, stepping Rating/1v1 - which feeds the
			// steering latch, the PSD plateau gate, and the golden archive.
			auto dir = saveFolder / std::to_string(version.timesteps);
			auto jsonTmp = dir / "STATS.json.tmp";
			std::ofstream fOut(jsonTmp);
			if (fOut.good()) {
				json j = {};
				j["skill_ratings"] = version.ratings.ToJSON();
				fOut << j.dump(4);
				fOut.close();
				std::error_code ec;
				std::filesystem::rename(jsonTmp, dir / "STATS.json", ec);
			}
			continue;
		}
		// Atomic like Learner::Save: write to .tmp, rename into place, so a crash
		// mid-save can never leave a truncated version dir for LoadVersions to abort on
		auto versionFinalFolder = saveFolder / std::to_string(version.timesteps);
		auto versionSaveFolder = saveFolder / (std::to_string(version.timesteps) + ".tmp");
		std::filesystem::remove_all(versionSaveFolder);
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

		std::filesystem::remove_all(versionFinalFolder);
		std::filesystem::rename(versionSaveFolder, versionFinalFolder);
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

		auto path = saveFolder / std::to_string(savedTimesteps);

		// A version NEWER than the current model happens legitimately now: the checkpoint
		// loader falls back across corrupt checkpoints (2026-07-13), which moves time
		// backwards. Aborting here would strand an unattended run - quarantine instead.
		if (savedTimesteps > curTimesteps) {
			RG_LOG(" > Version " << savedTimesteps << " is newer than the loaded checkpoint ("
				<< curTimesteps << ") - quarantining (checkpoint fallback moved time backwards)");
			std::error_code ec;
			std::filesystem::rename(path, saveFolder / ("stale_" + std::to_string(savedTimesteps)), ec);
			continue;
		}

		// Corrupt/truncated version dirs (pre-atomic-save crashes) must not abort the boot
		try {
			PolicyVersion& version = AddVersion(modelsTemplate, savedTimesteps);
			try {
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
			} catch (...) {
				version.models.Free();
				versions.pop_back();
				throw;
			}
		} catch (std::exception& e) {
			RG_LOG(" > CORRUPT/UNREADABLE version " << path << " (" << e.what()
				<< ") - quarantining and continuing");
			std::error_code ec;
			std::filesystem::rename(path, saveFolder / ("corrupt_" + std::to_string(savedTimesteps)), ec);
		}
	}

	SortVersions();

	RG_LOG(" > Loaded " << versions.size() << " versions(s)");
}

// Play `env` for up to `simSecs` of sim time: the live main on `mainTeam`, `oppModels` on the
// other. Calls onGoal(state, mainScored) for every goal and returns how many were scored in this
// segment. Extracted so the Elo rating matches and the reference battery cannot drift apart in
// how they actually play a game - the only differences between them are what they do with a goal
// and whether they continue an unfinished match.
namespace GGL {
static int PlayEvalSegment(
	PPOLearner* ppo, RLGC::EnvSet* env, ModelSet& oppModels, Team mainTeam,
	float simSecs, float* totalSimTime, float maxSimTime, int goalCap, int goalsSoFar,
	bool deterministic, RenderSender* renderSender,
	const std::function<void(const RLGC::GameState&, bool mainScored)>& onGoal) {

	// Team split is computed once: Reset() only resets arenas already flagged terminal, so
	// player-to-team assignment is stable for the whole segment.
	std::vector<int> mainPlayers = {}, oppPlayers = {};
	for (int i = 0; i < env->arenas.size(); i++) {
		auto& state = env->state.gameStates[i];
		for (int j = 0; j < state.players.size(); j++) {
			int playerIdx = env->state.arenaPlayerStartIdx[i] + j;
			(state.players[j].team == mainTeam ? mainPlayers : oppPlayers).push_back(playerIdx);
		}
	}

	torch::Tensor
		tMainPlayers = torch::tensor(mainPlayers),
		tOppPlayers = torch::tensor(oppPlayers);

	int goals = 0;
	float stepTime = env->config.tickSkip * RLGC::CommonValues::TICK_TIME;
	for (float t = 0;
		t < simSecs && *totalSimTime < maxSimTime && (goalsSoFar + goals) < goalCap;
		t += stepTime, *totalSimTime += stepTime) {

		env->Reset();

		torch::Tensor tStates = DIMLIST2_TO_TENSOR<float>(env->state.obs);
		torch::Tensor tActionMasks = DIMLIST2_TO_TENSOR<uint8_t>(env->state.actionMasks);

		torch::Tensor tMainStates = tStates.index_select(0, tMainPlayers);
		torch::Tensor tOppStates = tStates.index_select(0, tOppPlayers);
		torch::Tensor tMainMasks = tActionMasks.index_select(0, tMainPlayers);
		torch::Tensor tOppMasks = tActionMasks.index_select(0, tOppPlayers);

		env->StepFirstHalf(true);

		torch::Tensor tMainActions, tOppActions, _tLogProbs;
		PPOLearner::InferActionsFromModels(
			ppo->models, tMainStates.to(ppo->device, true), tMainMasks.to(ppo->device, true),
			deterministic, ppo->config.policyTemperature, ppo->config.useHalfPrecision,
			&tMainActions, &_tLogProbs);
		PPOLearner::InferActionsFromModels(
			oppModels, tOppStates.to(ppo->device, true), tOppMasks.to(ppo->device, true),
			deterministic, ppo->config.policyTemperature, ppo->config.useHalfPrecision,
			&tOppActions, &_tLogProbs);

		auto mainActions = TENSOR_TO_VEC<int>(tMainActions);
		auto oppActions = TENSOR_TO_VEC<int>(tOppActions);

		auto combinedActions = std::vector<int>(env->state.numPlayers, -1);
		for (int i = 0; i < mainActions.size(); i++)
			combinedActions[mainPlayers[i]] = mainActions[i];
		for (int i = 0; i < oppActions.size(); i++)
			combinedActions[oppPlayers[i]] = oppActions[i];

		env->Sync();
		env->StepSecondHalf(combinedActions, false);

		for (int i = 0; i < env->arenas.size(); i++) {
			auto& gs = env->state.gameStates[i];
			if (gs.goalScored) {
				onGoal(gs, RS_TEAM_FROM_Y(gs.ball.pos.y) != mainTeam);
				goals++;
			}
		}

		if (renderSender)
			renderSender->Send(env->state.gameStates[0]);
	}
	return goals;
}
} // namespace GGL

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

	RG_LOG("Running skill matches (simTime=" << skill.config.simTime << ")...");

	SkillRating prevCurRatings = skill.curRatings;

	skill.curGoals += PlayEvalSegment(
		ppo, skill.envSet, oldVersion.models, newTeam,
		skill.config.simTime, &totalSimTime, skill.config.maxSimTime,
		(int)skill.envSet->arenas.size(), skill.curGoals,
		skill.config.deterministic, renderSender,
		[&](const RLGC::GameState& gs, bool mainScored) {
			if (mainScored)
				fnUpdateRatings(skill.curRatings, oldVersion.ratings, const_cast<RLGC::GameState&>(gs));
			else
				fnUpdateRatings(oldVersion.ratings, skill.curRatings, const_cast<RLGC::GameState&>(gs));
		});

	if (dist && dist->distributed()) {
		auto avgR = [&](SkillRating& r) {
			const char* modes[] = { "1v1", "2v2", "3v3" };
			float vals[3];
			for (int i = 0; i < 3; i++)
				vals[i] = r.data.count(modes[i]) ? r.data[modes[i]] : skill.config.initialRating;
			dist->avg_host(vals, 3);
			for (int i = 0; i < 3; i++)
				r.data[modes[i]] = vals[i];
		};
		avgR(skill.curRatings);
		for (auto& v : versions)
			avgR(v.ratings);
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

// ==================== PERMANENT REFERENCE SET ====================
// The honest yardstick. See SkillTrackerConfig for why Rating/1v1 cannot be one.

void GGL::PolicyVersionManager::ConsiderReference(ModelSet modelsToClone, uint64_t timesteps) {
	RG_NO_GRAD;
	if (skill.config.maxReferences <= 0)
		return;
	for (auto& r : references)
		if (r.timesteps == timesteps)
			return; // already have it (restart replay)

	PolicyVersion ref{ timesteps, modelsToClone.CloneAll() };
	references.push_back(ref);
	std::sort(references.begin(), references.end(),
		[](const PolicyVersion& a, const PolicyVersion& b) { return a.timesteps < b.timesteps; });

	// Re-decimate to a log-spaced span. This DOES evict, but never the oldest or the newest -
	// so Ref/Oldest Share is measured against a permanently fixed opponent, which is the whole
	// point. FIFO here would recreate the recent-window myopia the set exists to cure.
	std::vector<uint64_t> ts;
	for (auto& r : references) ts.push_back(r.timesteps);
	std::vector<size_t> keep = DecimateSpaced(ts, (size_t)skill.config.maxReferences);

	std::set<size_t> keepSet(keep.begin(), keep.end());
	std::vector<PolicyVersion> kept;
	for (size_t i = 0; i < references.size(); i++) {
		if (keepSet.count(i)) {
			kept.push_back(references[i]);
		} else {
			RG_LOG("Reference set: dropping " << references[i].timesteps << " (re-spacing)");
			references[i].models.Free();
			refGoals.erase(references[i].timesteps);
		}
	}
	references = kept;
}

void GGL::PolicyVersionManager::SaveReferences() {
	if (dist && dist->rank() != 0)
		return;
	RG_NO_GRAD;
	if (skill.config.maxReferences <= 0)
		return;

	std::set<uint64_t> live;
	for (auto& r : references) live.insert(r.timesteps);

	// Drop dirs for references we no longer hold (decimated away).
	for (auto& e : std::filesystem::directory_iterator(saveFolder)) {
		if (!e.is_directory()) continue;
		std::string n = e.path().filename().string();
		if (n.rfind("ref_", 0) != 0) continue;
		std::string tsStr = n.substr(4);
		if (tsStr.empty() || !std::all_of(tsStr.begin(), tsStr.end(), ::isdigit)) continue;
		if (!live.count((uint64_t)std::stoull(tsStr)))
			std::filesystem::remove_all(e.path());
	}

	for (auto& ref : references) {
		auto finalDir = saveFolder / ("ref_" + std::to_string(ref.timesteps));
		if (std::filesystem::exists(finalDir))
			continue; // reference weights are immutable and carry no drifting rating to re-persist
		auto tmpDir = saveFolder / ("ref_" + std::to_string(ref.timesteps) + ".tmp");
		std::filesystem::remove_all(tmpDir);
		std::filesystem::create_directories(tmpDir);
		ref.models.Save(tmpDir, false);
		std::filesystem::rename(tmpDir, finalDir);
	}
}

void GGL::PolicyVersionManager::LoadReferences(ModelSet modelsTemplate, uint64_t curTimesteps) {
	RG_NO_GRAD;
	if (skill.config.maxReferences <= 0)
		return;

	for (auto& r : references) r.models.Free();
	references.clear();

	std::vector<uint64_t> found;
	for (auto& e : std::filesystem::directory_iterator(saveFolder)) {
		if (!e.is_directory()) continue;
		std::string n = e.path().filename().string();
		if (n.rfind("ref_", 0) != 0) continue;
		std::string tsStr = n.substr(4);
		if (tsStr.empty() || !std::all_of(tsStr.begin(), tsStr.end(), ::isdigit)) continue;
		found.push_back((uint64_t)std::stoull(tsStr));
	}
	std::sort(found.begin(), found.end());

	for (uint64_t ts : found) {
		auto path = saveFolder / ("ref_" + std::to_string(ts));
		// Same fallback discipline as versions: a reference is an OPTIONAL input and a bad one
		// must never take down the run.
		if (ts > curTimesteps) {
			RG_LOG(" > Reference " << ts << " is newer than the loaded checkpoint ("
				<< curTimesteps << ") - quarantining");
			std::error_code ec;
			std::filesystem::rename(path, saveFolder / ("stale_ref_" + std::to_string(ts)), ec);
			continue;
		}
		try {
			PolicyVersion ref{ ts, modelsTemplate.CloneAll() };
			try {
				ref.models.Load(path, false, false);
			} catch (...) {
				ref.models.Free();
				throw;
			}
			references.push_back(ref);
		} catch (std::exception& e) {
			RG_LOG(" > CORRUPT/UNREADABLE reference " << path << " (" << e.what()
				<< ") - quarantining and continuing");
			std::error_code ec;
			std::filesystem::rename(path, saveFolder / ("corrupt_ref_" + std::to_string(ts)), ec);
		}
	}
	RG_LOG(" > Loaded " << references.size() << " reference(s)");
}

void GGL::PolicyVersionManager::RunReferenceMatches(PPOLearner* ppo, Report& report) {
	RG_NO_GRAD;
	if (references.empty() || !refBattery.envSet)
		return;

	// Round-robin: every reference gets sampled evenly, so no single one dominates the
	// aggregate. No continuation bookkeeping - an inconclusive segment simply contributes
	// nothing, and the counters are cumulative so the slope is unaffected.
	refBattery.nextRefIdx %= (int)references.size();
	PolicyVersion& ref = references[refBattery.nextRefIdx];
	refBattery.nextRefIdx = (refBattery.nextRefIdx + 1) % (int)references.size();

	Team mainTeam = (Team)Math::RandInt(0, 2); // swap sides to cancel any first-move bias
	refBattery.envSet->Reset();

	int64_t mainGoals = 0, oppGoals = 0;
	float totalSimTime = 0;
	PlayEvalSegment(
		ppo, refBattery.envSet, ref.models, mainTeam,
		skill.config.simTime, &totalSimTime, skill.config.simTime,
		(int)refBattery.envSet->arenas.size(), 0,
		skill.config.deterministic, nullptr,
		[&](const RLGC::GameState&, bool mainScored) {
			(mainScored ? mainGoals : oppGoals)++;
		});

	int64_t scored[2] = { mainGoals, oppGoals };
	if (dist && dist->distributed())
		dist->sum_host(scored, 2);
	auto& tally = refGoals[ref.timesteps];
	tally.first += scored[0];
	tally.second += scored[1];

	// Cumulative counters: as with Nexto/Goals *, the SLOPE is the signal and the level is not
	// directly comparable across runs.
	int64_t aggFor = 0, aggAgainst = 0;
	for (auto& r : references) {
		auto it = refGoals.find(r.timesteps);
		if (it == refGoals.end()) continue;
		aggFor += it->second.first;
		aggAgainst += it->second.second;
	}
	report["Ref/Count"] = (float)references.size();
	report["Ref/Oldest Ts"] = (float)references.front().timesteps;
	report["Ref/Goals For"] = (float)aggFor;
	report["Ref/Goals Against"] = (float)aggAgainst;
	if (aggFor + aggAgainst > 0)
		report["Ref/Share"] = (float)aggFor / (float)(aggFor + aggAgainst);

	// The headline. The oldest reference is never evicted, so this series alone is measured
	// against a genuinely fixed opponent and cannot drift the way Rating/1v1 does.
	auto oldest = refGoals.find(references.front().timesteps);
	if (oldest != refGoals.end() && oldest->second.first + oldest->second.second > 0)
		report["Ref/Oldest Share"] =
			(float)oldest->second.first / (float)(oldest->second.first + oldest->second.second);

	RG_LOG("Reference battery vs " << ref.timesteps << ": " << mainGoals << "-" << oppGoals
		<< " (cumulative share " << (aggFor + aggAgainst > 0
			? (float)aggFor / (float)(aggFor + aggAgainst) : 0.f) << ")");
}

void GGL::PolicyVersionManager::OnIteration(struct PPOLearner* ppo, Report& report, int64_t totalTimesteps, int64_t prevTotalTimesteps) {
	if ((totalTimesteps / tsPerVersion > prevTotalTimesteps / tsPerVersion) || (prevTotalTimesteps == 0)) {
		// Save version
		AddVersion(ppo->GetPolicyModels(), totalTimesteps);
		// Every archived version is also a reference CANDIDATE; the log-spacing decides which
		// ones stay. Versions rotate out at ~800M steps, references do not.
		ConsiderReference(ppo->GetPolicyModels(), totalTimesteps);
	}

	if (skill.config.enabled) {
		skill.iterationsSinceRan++;
		if (skill.iterationsSinceRan >= skill.config.updateInterval && !versions.empty()) {
			skill.iterationsSinceRan = 0;
			RunSkillMatches(ppo, report);
		}

		refBattery.iterationsSinceRan++;
		if (refBattery.iterationsSinceRan >= skill.config.referenceUpdateInterval
			&& !references.empty()) {
			refBattery.iterationsSinceRan = 0;
			RunReferenceMatches(ppo, report);
		}
	}
}

void GGL::PolicyVersionManager::AddRunningStatsToJSON(nlohmann::json& json) {
	if (skill.config.enabled) {
		json["skill_ratings"] = skill.curRatings.ToJSON();

		// Persist the reference tallies keyed by timestep. Cumulative counters are only a
		// yardstick if they survive restarts - forgetting the matching read is exactly how the
		// Nexto counters were silently zeroing on every restart (4f25b1c).
		nlohmann::json refJ = nlohmann::json::object();
		for (auto& kv : refGoals)
			refJ[std::to_string(kv.first)] = { kv.second.first, kv.second.second };
		json["reference_goals"] = refJ;
	}
}

void GGL::PolicyVersionManager::LoadRunningStatsFromJSON(const nlohmann::json& json) {
	if (skill.config.enabled) {
		if (json.contains("skill_ratings"))
			skill.curRatings.ReadFromJSON(json["skill_ratings"]);

		refGoals.clear();
		if (json.contains("reference_goals")) {
			for (auto& item : json["reference_goals"].items()) {
				auto v = item.value();
				if (v.is_array() && v.size() == 2)
					refGoals[(uint64_t)std::stoull(item.key())] =
						{ v[0].get<int64_t>(), v[1].get<int64_t>() };
			}
		}
	}
}