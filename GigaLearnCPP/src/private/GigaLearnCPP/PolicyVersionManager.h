#pragma once

#include "Util/Models.h"
#include <GigaLearnCPP/SkillTrackerConfig.h>
#include <GigaLearnCPP/Util/Report.h>
#include <GigaLearnCPP/Util/RenderSender.h>

#include <nlohmann/json.hpp>
#include <limits>
#include <shared_mutex>

namespace GGL {

	struct SkillRating {
		std::map<std::string, float> data;

		static std::string GetModeName(const RLGC::GameState& state) {
			int playersOnTeams[2] = { 0, 0 };
			for (auto& player : state.players)
				playersOnTeams[(int)player.team]++;

			int minPlayersOnTeam = RS_MIN(playersOnTeams[0], playersOnTeams[1]);
			int maxPlayersOnTeam = RS_MAX(playersOnTeams[0], playersOnTeams[1]);

			std::string name = RS_STR(minPlayersOnTeam << "v" << maxPlayersOnTeam);
			return name;
		}

		float& GetRating(std::string name, float defaultRating) {
			if (data.contains(name)) {
				return data[name];
			} else {
				data[name] = defaultRating;
				return data[name];
			}
		}

		float& GetRating(const RLGC::GameState& state, float defaultRating) {
			return GetRating(GetModeName(state), defaultRating);
		}

		nlohmann::json ToJSON() {
			nlohmann::json j = {};
			for (auto pair : data)
				j[pair.first] = pair.second;
			return j;
		}

		void ReadFromJSON(const nlohmann::json& j) {
			data = {};
			for (auto& pair : j.items())
				data[pair.key()] = pair.value();
		}
	};

	struct PolicyVersion {
		uint64_t timesteps;
		ModelSet models;
		SkillRating ratings;
	};

	struct PolicyVersionManager {
		std::vector<PolicyVersion> versions;
		mutable std::shared_mutex versionsMutex; // shared: collectors reading; unique: OnIteration mutating
		std::filesystem::path saveFolder;
		std::filesystem::path bestAnchorFolder;
		int maxVersions;
		uint64_t tsPerVersion;

		PolicyVersion bestAnchor = {};
		bool hasBestAnchor = false;
		float bestAnchorRating = -std::numeric_limits<float>::infinity();
		std::string bestAnchorRatingMode = "";

		//////////////////

		struct {
			SkillTrackerConfig config;

			RLGC::EnvSet* envSet;
			int curGoals = 0;

			bool doContinuation = false;
			int prevOldVersionIndex;
			bool prevOldVersionIsBestAnchor = false;
			Team prevNewTeam;
			float prevSimTime;

			int iterationsSinceRan = 0;

			SkillRating curRatings = {};
		} skill;

		RenderSender* renderSender;

		PolicyVersionManager(
			std::filesystem::path saveFolder, int maxVersions, uint64_t tsPerVersion,
			const SkillTrackerConfig& skillTrackerConfig, const RLGC::EnvSetConfig& envSetConfig,
			RenderSender* renderSender = NULL);

		// NOTE: Passed models should not be already cloned
		PolicyVersion& AddVersion(ModelSet modelsToClone, uint64_t timesteps);

		void SaveVersions();
		void LoadVersions(ModelSet modelsTemplate, uint64_t curTimesteps);
		void SaveBestAnchor();
		void LoadBestAnchor(ModelSet modelsTemplate, uint64_t curTimesteps);

		void SortVersions();
		int GetVersionCandidateCount() const;
		PolicyVersion& GetVersionCandidate(int index);
		float GetPeakRating(const SkillRating& ratings, std::string* outMode = NULL) const;
		void UpdateBestAnchor(struct PPOLearner* ppo, Report& report, uint64_t totalTimesteps);

		void RunSkillMatches(struct PPOLearner* ppo, Report& report, uint64_t totalTimesteps);

		void OnIteration(struct PPOLearner* ppo, Report& report, int64_t totalTimesteps, int64_t prevTotalTimesteps);

		void AddRunningStatsToJSON(nlohmann::json& json);
		void LoadRunningStatsFromJSON(const nlohmann::json& json);

		// Each stored PolicyVersion owns a cloned ModelSet (AddVersion -> CloneAll), and the
		// skill tracker owns a new'd EnvSet. ModelSet has no destructor, so without this every
		// version's cloned models + the skill EnvSet leaked on teardown. Defined in the .cpp
		// where RLGC::EnvSet is a complete type.
		~PolicyVersionManager();
	};
}
