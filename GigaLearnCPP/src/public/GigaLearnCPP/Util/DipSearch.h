#pragma once
// DIP SEARCH: search-and-imitate from value dips (research/reports/HEADROOM_SEARCH.md).
//
// The 2026-09-03 study asked, for a banked physics state, whether a short action prefix
// exists that beats what the policy realises from that state. Search from the composition
// critic's high-H states found nothing (pre-registered P1 failed on two seeds). Search from
// VALUE DIPS - rows whose k-step realised advantage sits in the bottom few percent while play
// continues - found a fat tail of full-goal-sized fixes (P2 passed twice), and the single
// best predictor of a fixable state was the critic's overestimate of it. So the trigger is
// the realised advantage the trainer already computes, not H.
//
// Mechanism (all default-OFF, LearnerConfig::dipSearch / GGL_DIPSEARCH):
//   collect   : a fixed subset of arenas captures an ArenaSnapshot every decision step
//               (the exact v3 restore the viewer's rewind uses) into a rolling bank; every
//               row of a subset arena carries its snapshot id.
//   learn-prep: after GAE, rows with k steps of play left are scored by their realised
//               k-step advantage; the bottom `dipQuantile` are dip candidates, de-duplicated
//               to one per k-row window, capped at maxStates. Each is restored into a
//               private arena pool and searched: baseline = fresh closed-loop rollouts of
//               the policy; candidates = open-loop macro-action prefixes (uniform-valid and
//               policy-at-temperature) with the policy closing the loop and playing every
//               other car; finalists are re-evaluated fresh; the winner is re-evaluated
//               HELD-OUT. gain = held-out mean - baseline mean, in critic (GAE) units.
//   learn     : states with gain >= minGain contribute their prefix steps (obs, action,
//               mask) as extra imitation rows, weight = min(gain, weightCap), consumed by
//               PPOLearner as a SIL-shaped term (-log pi(a|s) * w, silCoeff-scaled). They
//               never touch the critic, the advantages, or the PPO ratio.
//
// Constraint note: this banks retrospective states and searches from them, which is what
// COMPOSITION_CRITIC.md's C2 forbids for the composition-critic claim. That is a recorded
// decision, not an oversight: the measurement that motivates it is the one that retired H
// as a trigger.
//
// Never uses RLGC::g_ThreadPool (it belongs to the collect worker during learn-prep); the
// pool steps on ad-hoc std::threads, the steering-landing-sim pattern.

#include "../Framework.h"
#include "VizControl.h"          // ArenaSnapshot
#include "DipSearchConfig.h"
#include <RLGymCPP/EnvSet/EnvSet.h>

#include <deque>
#include <mutex>
#include <vector>

namespace GGL {


	// Rolling, id-addressed store of ArenaSnapshots. Push from the collect worker (any
	// thread), Get from learn-prep, Trim in the barrier zone. Ids are global and monotonic,
	// so a row can name its snapshot across the iteration boundary; an evicted id simply
	// fails Get (the episode outlived the bank - not searchable, not an error).
	class DipSnapshotBank {
	public:
		explicit DipSnapshotBank(size_t capacity) : capacity(capacity) {}
		int64_t Push(ArenaSnapshot&& snap);
		bool Get(int64_t id, ArenaSnapshot& out) const;
		void Trim();            // drop oldest beyond capacity (call only when no pushes are in flight)
		size_t Size() const;
		size_t capacity;
	private:
		mutable std::mutex mtx;
		std::deque<ArenaSnapshot> frames;
		int64_t firstId = 0;
	};

	struct DipSearchState {
		ArenaSnapshot snap;
		int slot = 0;            // the player (car index in the arena) whose row dipped
		int64_t row = -1;        // combinedTraj row, telemetry only
		float aK = 0;            // realised k-step advantage that flagged it
	};

	struct DipSearchResult {
		bool searched = false;   // false = budget ran out before this state
		bool accepted = false;   // gain >= minGain
		float baseMean = 0, baseStd = 0, bestMean = 0, gain = 0;
		float baseGoal = 0, bestGoal = 0, baseConcede = 0, bestConcede = 0;
		int bestL = 0;           // winning prefix length in macros (0 = the policy itself won)
		std::vector<int> prefix;
		// Imitation rows (prefix steps of the recorded held-out rollouts), obs already in the
		// policy's input space (normalised if the learner normalises)
		int nRows = 0;
		std::vector<float> obs;
		std::vector<int32_t> actions;
		std::vector<uint8_t> masks;
	};

	class PPOLearner;

	class DipSearch {
	public:
		DipSearch(const DipSearchConfig& cfg, RLGC::EnvCreateFn envCreateFn,
			int tickSkip, int actionDelay, int obsSize, int numActions, float gamma);
		~DipSearch();
		RG_NO_COPY(DipSearch);

		// Search every state (in order) until the budget runs out. rewardScale multiplies the
		// env reward into critic units (1/returnStd, GAE parity); rewardClip is the GAE clip
		// on the scaled reward (0 = none). normMean/normStd (may be null) are the learner's
		// clamped obs statistics.
		void Run(std::vector<DipSearchState>& states, PPOLearner* ppo,
			const std::vector<double>* normMean, const std::vector<double>* normStd,
			float rewardScale, float rewardClip, std::vector<DipSearchResult>& out,
			float budgetSecs);

		// telemetry from the last Run
		float lastWaveSecs = 0;
		int lastWaves = 0;

	private:
		struct Slot;
		struct Job;
		DipSearchConfig cfg;
		int tickSkip, actionDelay, obsSize, numActions, numPlayers = 0;
		float gamma;
		std::vector<Slot*> pool;

		void Restore(Slot& s, const ArenaSnapshot& snap);
		void ParallelFor(int n, const std::function<void(int)>& fn);
		void RunWave(std::vector<Job>& jobs, const std::vector<DipSearchState>& states,
			PPOLearner* ppo, const std::vector<double>* normMean,
			const std::vector<double>* normStd, float rewardScale, float rewardClip);
	};
}
