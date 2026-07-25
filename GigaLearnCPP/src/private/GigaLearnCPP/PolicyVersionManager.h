#pragma once

#include "Util/Models.h"
#include <GigaLearnCPP/SkillTrackerConfig.h>
#include <GigaLearnCPP/Util/Report.h>
#include <GigaLearnCPP/Util/LogSpaced.h>
#include <GigaLearnCPP/Util/RenderSender.h>

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
	};

	struct PolicyVersionManager {
		std::vector<PolicyVersion> versions;

		// PERMANENT REFERENCE SET (2026-07-25). A separate vector on purpose, exactly as the
		// league's anchors were: everything that rotates, re-rates or trains walks `versions`
		// only, so references are structurally exempt rather than exempt-by-if-check. They are
		// the measuring stick - never drawn as a training opponent, never assigned a drifting
		// rating. Persisted to saveFolder/ref_<ts>/ using the same self-contained-dir pattern as
		// versions (NOT the league's index-keyed shared blob store, which desynced in practice:
		// 141 weight files for a 121-member archive).
		std::vector<PolicyVersion> references;
		// Cumulative goal tallies against the reference set, keyed by reference timestep. Kept
		// keyed rather than positional so decimation cannot silently re-point a series, and
		// persisted so a restart does not zero the yardstick (the exact bug that silently killed
		// the Nexto counters in 4f25b1c).
		std::map<uint64_t, std::pair<int64_t, int64_t>> refGoals; // ts -> {mainGoals, refGoals}

		std::filesystem::path saveFolder;
		int maxVersions;
		uint64_t tsPerVersion;

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

		// Reference battery state. Its own EnvSet so its matches cannot stomp the rating
		// matches' continuation bookkeeping; this is a REALLOCATION of the 16 eval arenas the
		// deleted league match env used, not new cost.
		struct {
			RLGC::EnvSet* envSet = nullptr;
			int iterationsSinceRan = 0;
			int nextRefIdx = 0;         // round-robin so every reference is sampled evenly
		} refBattery;

		RenderSender* renderSender;

		PolicyVersionManager(
			std::filesystem::path saveFolder, int maxVersions, uint64_t tsPerVersion,
			const SkillTrackerConfig& skillTrackerConfig, const RLGC::EnvSetConfig& envSetConfig,
			RenderSender* renderSender = NULL);

		// NOTE: Passed models should not be already cloned
		PolicyVersion& AddVersion(ModelSet modelsToClone, uint64_t timesteps);

		void SaveVersions();
		void LoadVersions(ModelSet modelsTemplate, uint64_t curTimesteps);

		void SortVersions();

		// Consider promoting a freshly-archived version into the permanent reference set, then
		// re-decimate so the kept set stays log-spaced and capped at config.maxReferences.
		void ConsiderReference(ModelSet modelsToClone, uint64_t timesteps);
		void SaveReferences();
		void LoadReferences(ModelSet modelsTemplate, uint64_t curTimesteps);

		// One reference battery segment: the current main vs a round-robin reference, tallied as
		// goal share. No rating is computed or updated for either side - that is the point.
		void RunReferenceMatches(struct PPOLearner* ppo, Report& report);

		void RunSkillMatches(struct PPOLearner* ppo, Report& report);

		void OnIteration(struct PPOLearner* ppo, Report& report, int64_t totalTimesteps, int64_t prevTotalTimesteps);

		void AddRunningStatsToJSON(nlohmann::json& json);
		void LoadRunningStatsFromJSON(const nlohmann::json& json);

		// TODO: Add deconstructor
	};
}