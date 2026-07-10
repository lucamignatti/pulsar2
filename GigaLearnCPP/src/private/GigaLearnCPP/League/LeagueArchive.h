#pragma once
#include "../FrameworkTorch.h"
#include "../Util/Models.h"
#include <GigaLearnCPP/PSDConfig.h>
#include <GigaLearnCPP/Util/Report.h>

#include <RLGymCPP/EnvSet/EnvSet.h>
#include <nlohmann/json.hpp>
#include <map>

namespace GGL {

	class PPOLearner;

	// Quality-Diversity league: a MAP-Elites archive of FULL-WEIGHT, gradient-free members that
	// serve as a diverse opponent population. Members are snapshots of the main agent and their
	// evolved offspring (Gaussian mutation + DARE crossover). Behavior descriptors place each in a
	// grid cell; within a cell the higher-quality member wins; unconstrained "exploiter" slots
	// audit for behaviors the descriptor basis misses (handoff §4.2 completeness gap).
	//
	// Runs on its OWN isolated match EnvSet (like the skill tracker), so it never disturbs the
	// training arenas. Additive; created only when config.league.enabled.
	class LeagueArchive {
	public:
		struct Member {
			std::vector<torch::Tensor> params; // flat param vectors, one per scratch model (CPU)
			std::vector<float> bd;             // behavior descriptor (one per config axis)
			float fitness = 0;                 // quality (goal differential vs the main agent)
			int cell = -1;                     // MAP-Elites cell index (-1 = unconstrained exploiter)
			bool exploiter = false;
			int matches = 0;
		};

		LeagueConfig cfg;
		torch::Device device;
		PPOLearner* ppo;                       // main agent (opponent + weight template); not owned
		RLGC::EnvSet* matchEnv = nullptr;      // isolated arenas for league matches
		std::filesystem::path leagueDir;

		std::vector<Member> members;
		std::map<long, int> cellToMember;      // occupied cell -> index into members
		long exploiterUnmappedWins = 0;

		LeagueArchive(const LeagueConfig& cfg, PPOLearner* ppo, RLGC::EnvSet* trainEnv,
			torch::Device device, std::filesystem::path checkpointFolder);
		~LeagueArchive();

		// Called each Learner iteration; evolves/evaluates on the configured cadence.
		void OnIteration(Report& report, uint64_t totalIterations);

		// PFSP opponent sample (softmax over difficulty). Returns index or -1 if empty. Exposed for
		// future DESCEND-arena opponent injection; not wired into the hot collection loop yet.
		int SampleOpponent() const;
		const Member* GetMember(int i) const { return (i >= 0 && i < (int)members.size()) ? &members[i] : nullptr; }

		void ToJSON(nlohmann::json& j) const;
		void FromJSON(const nlohmann::json& j);

	private:
		ModelSet scratch;                      // policy + shared_head to load member weights into

		std::vector<torch::Tensor> SnapshotMain() const;              // current main policy weights
		void LoadInto(ModelSet& set, const std::vector<torch::Tensor>& params);
		long CellIndex(const std::vector<float>& bd) const;

		// Play `steps` of match env with `memberParams` (team A) vs the main agent (team B); return
		// quality (A goals - B goals) and fill `outBD`.
		float EvaluateMember(const std::vector<torch::Tensor>& memberParams, std::vector<float>& outBD);

		void EvolveStep(Report& report);
		void TryInsert(Member&& m);
		void LogMetrics(Report& report) const;
	};
}
