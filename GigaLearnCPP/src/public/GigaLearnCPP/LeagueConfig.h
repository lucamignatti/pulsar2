#pragma once
#include <RLGymCPP/BasicTypes/Lists.h>
#include <string>
#include <vector>

// Extracted from PSDConfig.h on 2026-07-25 when PSD was retired. LeagueConfig had been sharing
// that header purely by historical accident - the QD league is LIVE and has nothing to do with
// Basin-Racing, and leaving it in a file named after a deleted subsystem would have been the
// reason PSDConfig.h could never be removed.
namespace GGL {

	// Quality-Diversity league (MAP-Elites over FULL-WEIGHT members — deliberately NOT EGGROLL
	// adapters, so promotion never invalidates members and there is no BC re-anchoring leak).
	// Additive and default-OFF; supersedes trainAgainstOldVersions (which stays off).
	struct LeagueConfig {
		bool enabled = false;

		// Behavior-descriptor axes (progressive). Each name maps to a per-rollout stat the match
		// engine already computes; the grid is binsPerAxis^|axes|.
		std::vector<std::string> gridAxes = { "in_air_ratio", "field_y", "boost_economy" };
		int binsPerAxis = 6;

		// Quantile-adaptive binning (cheap CVT-MAP-Elites). Uniform bins over the theoretical [0,1]
		// axis ranges are nearly all unreachable: the measured 3.1 population lived in
		// in_air [0.51,0.92], field_y [0.37,0.62], boost [0.016,0.060] — the boost axis never left
		// bin 0 and only 3 of 64 cells were occupied. With quantileBins, bin edges are the running
		// quantiles of the last quantileSampleCap observed BDs (every EvaluateMember contributes), so
		// resolution concentrates where the population actually lives and tracks it as the main
		// improves. Edges refresh on the reseed cadence (plus one bootstrap refresh once
		// quantileMinSamples BDs exist); each refresh re-tokenizes all members and enforces
		// one-elite-per-cell. false = the original fixed uniform [0,1] binning.
		bool quantileBins = true;
		int quantileSampleCap = 512;   // rolling BD sample window (~one reseed era of evals)
		int quantileMinSamples = 32;   // below this, fall back to uniform binning

		// Admission floor on fitness = (member goals - main goals) over an eval. NEGATIVE on purpose:
		// a good sparring partner is allowed to lose by a bit (style diversity matters more than the
		// member out-scoring the current main); only genuinely broken members are culled below this.
		float competenceFloor = -25.0f;
		float cullFrac = 0.25f;           // (reserved) fraction culled per interval
		int exploiterSlots = 2;           // BD-unconstrained members that hill-climb pure fitness-vs-main

		// Evolution operators (weight space, no gradients).
		float mutationSigma = 0.02f;      // full-Gaussian mutation scale (relative to weight RMS)
		float dareDropRate = 0.5f;        // DARE drop-and-rescale probability

		float pfspTemp = 1.0f;            // PFSP matchmaking temperature
		float descendOpponentFrac = 0.25f;// probability a training iteration faces a league opponent

		// === Permanent spaced ANCHOR opponents (research/reports/LEAGUE_ANCHORS.md) ===
		// Measured problem (2026-07-19): the evolved archive collapses to ~3 members / 1
		// occupied cell because fitness (= member goals - main goals) is RE-SCORED against
		// the improving main, so every fixed style ratchets below competenceFloor and is
		// culled; ReseedFromMain only ever adds near-clones. Result: the "diverse" opponent
		// pool is 3 copies of the recent self, and real match-play progress is ~+4 Elo/B.
		//
		// Anchors are full checkpoints archived OUTSIDE the rotation by
		// tools/archive_anchor.sh. They live in their own vector, so they are structurally
		// exempt from Cull/RefreshStalest/TryInsert/MAP-Elites: they are NEVER re-scored and
		// NEVER culled. That exemption is the entire design - a build that keeps scoring
		// them re-collapses in days via the same ratchet.
		float anchorFrac = 0.0f;          // share of ALL training iterations facing an anchor
		                                  // (served as a conditional draw inside the
		                                  //  descendOpponentFrac serve; 0 = feature OFF)
		int anchorMaxServed = 24;         // cap on the SERVING set (spaced-decimation keeps a
		                                  //  log-spaced span, not a recent window); disk keeps all
		float anchorRecencyFloor = 0.15f; // sampling weight of the OLDEST anchor relative to the
		                                  //  newest (linear in rank); never 0 - tail robustness
		std::string anchorDir = "";       // default: "<checkpointFolder>_anchors"
		int matchesPerMember = 20;        // (reserved) eval matches before fitness/BD is trusted
		int maxMembers = 256;
		int evolveEveryIters = 16;        // cadence of the league evolution step
		// Snapshot the CURRENT main policy as a fresh lineage this often (in iterations). Each snapshot
		// founds a new "era" whose style is the main's at that time; 0 = never re-seed (single lineage).
		int reseedEveryIters = 1000;
	};
}
