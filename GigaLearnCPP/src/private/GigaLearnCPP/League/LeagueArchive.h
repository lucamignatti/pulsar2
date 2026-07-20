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
			float fitness = 0;                 // quality = (member goals - main goals) over an eval
			int cell = -1;                     // MAP-Elites cell index (-1 = unconstrained exploiter)
			bool exploiter = false;
			int matches = 0;
			int lineage = 0;                   // which re-seed era this member descends from
			int age = 0;                       // evolve steps since last (re)evaluation; drives refresh
		};

		LeagueConfig cfg;
		torch::Device device;
		PPOLearner* ppo;                       // main agent (opponent + weight template); not owned
		RLGC::EnvSet* matchEnv = nullptr;      // isolated arenas for league matches
		std::filesystem::path leagueDir;

		std::vector<Member> members;
		std::map<long, int> cellToMember;      // occupied cell -> index into members
		long exploiterUnmappedWins = 0;
		int nextLineage = 0;                   // monotonic era counter for re-seeds
		int refreshCursor = 0;                 // round-robin over members for staleness refresh

		// Permanent spaced ANCHOR opponents (LEAGUE_ANCHORS.md). A SEPARATE vector on
		// purpose: every collapse mechanism (Cull, RefreshStalest, TryInsert, the MAP-Elites
		// cell map, DedupCells) walks `members` only, so anchors are structurally exempt
		// from re-scoring and culling rather than relying on scattered if-checks. They are
		// also excluded from the diversity/cell metrics, and NOT persisted into the league
		// dir - they are re-loaded from their checkpoint dirs at every boot.
		std::vector<Member> anchors;           // fitness/cell/bd unused; params only
		std::vector<long long> anchorTs;       // timestep of each anchor, ascending
		long anchorServes = 0;                 // telemetry: anchor opponent serves this process

		// Quantile-adaptive binning state (cfg.quantileBins): a rolling window of every BD ever
		// measured, and the per-axis ascending bin edges derived from its quantiles. binEdges empty
		// (or flag off) = the original uniform [0,1] binning.
		std::vector<std::vector<float>> bdSamples;
		int bdSampleCursor = 0;
		std::vector<std::vector<float>> binEdges;

		LeagueArchive(const LeagueConfig& cfg, PPOLearner* ppo, RLGC::EnvSet* trainEnv,
			torch::Device device, std::filesystem::path checkpointFolder);
		~LeagueArchive();

		// Called each Learner iteration; evolves/evaluates on the configured cadence.
		void OnIteration(Report& report, uint64_t totalIterations);

		// PFSP opponent sample (softmax preferring even matches). Returns index or -1 if empty.
		int SampleOpponent() const;
		const Member* GetMember(int i) const { return (i >= 0 && i < (int)members.size()) ? &members[i] : nullptr; }

		// Load a PFSP-sampled member's weights into the internal opponent ModelSet and return it for
		// the training collection loop to use as this iteration's opponent; nullptr if the archive is
		// empty. This is what finally gives DESCEND arenas exposure to non-self styles.
		// allowAnchors=false disables the anchor slice only (evolved members still serve) - the
		// caller passes !steerRatingTripped so the rating latch kills the anchor intervention
		// exactly the way it kills steering/Ladder/RND (LEAGUE_ANCHORS.md guard).
		ModelSet* LoadPFSPOpponentModels(bool allowAnchors = true);

		void ToJSON(nlohmann::json& j) const;
		void FromJSON(const nlohmann::json& j);

	private:
		ModelSet scratch;                      // policy + shared_head to load member weights into (eval)
		ModelSet oppServe;                     // separate set the training loop borrows as its opponent

		std::vector<torch::Tensor> SnapshotMain() const;              // current main policy weights
		void LoadInto(ModelSet& set, const std::vector<torch::Tensor>& params);
		long CellIndex(const std::vector<float>& bd) const;

		// Anchor support (all no-ops when cfg.anchorFrac <= 0).
		void LoadAnchors();                    // scan anchorDir, decimate, load flat vectors
		int SampleAnchor() const;              // recency-spaced draw with a floor; -1 if none

		// Play a fixed window of match env with `memberParams` (team A) vs the main agent (team B);
		// return quality = (A goals - B goals) and fill `outBD`.
		float EvaluateMember(const std::vector<torch::Tensor>& memberParams, std::vector<float>& outBD);

		void EvolveStep(Report& report);
		void ReseedFromMain();                 // snapshot the current main as a new lineage
		void EvolveExploiters();               // hill-climb the BD-unconstrained exploiter slots
		void RefreshStalest();                 // re-evaluate the stalest member (fitness + BD drift)
		void TryInsert(Member&& m);            // MAP-Elites cell insert (one elite per cell)
		void RebuildCellMap();                 // recompute cellToMember from members
		void Cull();                           // cap member count, preserving cell elites + exploiters
		void RecordBDSample(const std::vector<float>& bd); // rolling window for quantile edges
		void RefreshBinEdges();                // recompute quantile edges + re-tokenize all members
		void DedupCells();                     // enforce one elite per cell (drops weaker duplicates)
		void LogMetrics(Report& report) const;
	};
}
