#pragma once

#include "Util/Models.h"
#include <GigaLearnCPP/SkillTrackerConfig.h>
#include <GigaLearnCPP/Util/Report.h>
#include <GigaLearnCPP/Util/RenderSink.h>

#include <nlohmann/json.hpp>

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
		bool isAnchor = false; // "gold" anchors are exempt from the rolling prune (see LearnerConfig anchor fields)
	};

	struct PolicyVersionManager {
		std::vector<PolicyVersion> versions;
		std::filesystem::path saveFolder;
		int maxVersions;
		uint64_t tsPerVersion;

		// Anchor-pool config (mirrored from LearnerConfig; set by the Learner after construction).
		int maxAnchors = 0;
		float anchorSelectChance = 0.5f;
		float anchorPromoteMargin = 25.0f;
		uint64_t anchorMinTsSpacing = 100'000'000;

		//////////////////

		struct {
			SkillTrackerConfig config;

			RLGC::EnvSet* envSet;
			int curGoals = 0;

			bool doContinuation = false;
			int prevOldVersionIndex;
			Team prevNewTeam;
			float prevSimTime;

			int iterationsSinceRan = 0;

			SkillRating curRatings = {};
		} skill;

		RenderSink* renderSender;

		PolicyVersionManager(
			std::filesystem::path saveFolder, int maxVersions, uint64_t tsPerVersion,
			const SkillTrackerConfig& skillTrackerConfig, const RLGC::EnvSetConfig& envSetConfig,
			RenderSink* renderSender = NULL);

		// NOTE: Passed models should not be already cloned
		PolicyVersion& AddVersion(ModelSet modelsToClone, uint64_t timesteps, bool allowPrune = true);

		void SaveVersions();
		void LoadVersions(ModelSet modelsTemplate, uint64_t curTimesteps);

		void SortVersions();

		// Anchor-pool helpers.
		float VersionRating(const PolicyVersion& v) const;   // mean ELO across modes (initialRating if unrated)
		void PruneVersions();                                // evict oldest non-anchors down to maxVersions
		void MaybeUpdateAnchors(uint64_t totalTimesteps);    // consider promoting the newest version to an anchor
		PolicyVersion* PickOldVersion();                     // anchor (prob anchorSelectChance) or rolling recent

		void RunSkillMatches(struct PPOLearner* ppo, Report& report);

		void OnIteration(struct PPOLearner* ppo, Report& report, int64_t totalTimesteps, int64_t prevTotalTimesteps);

		void AddRunningStatsToJSON(nlohmann::json& json);
		void LoadRunningStatsFromJSON(const nlohmann::json& json);

		// TODO: Add deconstructor
	};
}