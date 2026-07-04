#pragma once
#include "../Gamestates/GameState.h"

#include <mutex>
#include <vector>
#include <cstring>
#include <algorithm>

namespace RLGC {

	// A restorable mid-episode snapshot: enough of the arena's physics state to resume play from
	// exactly where a drill was banked. curLockedCar pointers inside BoostPadState are NOT
	// portable across arenas and are nulled out on restore (accepted v1 fidelity loss).
	struct ArenaSnapshot {
		struct CarSnap {
			Team team;
			CarState state;
		};
		std::vector<CarSnap> cars;
		BallState ball;
		std::vector<BoostPadState> pads;
	};

	// Deliberate-practice drill bank (Stage 3): a shared, mutex-guarded pool of banked near-miss
	// snapshots + the committed goal that was in play when the mistake happened. One bank is
	// shared by all per-arena DrillSetters AND the Learner's collection/learn-prep code.
	//
	// Thread-safety is by PHASE DISCIPLINE, not just the mutex: DrillSetters call TryBeginDrill()
	// from inside EnvSet::ResetArena(), which runs on the thread pool but is fenced (WaitUntilDone)
	// before the Learner's serial record/learn-prep code runs; the Learner only reads/writes
	// (ClearWindowsForResets/GetWindow/DecrementWindows/AddDrill/ReportResult) from its own serial
	// collection loop. The mutex exists to protect the bank's own containers, not to serialize
	// otherwise-concurrent readers/writers of the SAME window.
	//
	// In-memory only for v1: the bank is not checkpointed and simply starts empty on process
	// restart, re-populating from live Phi-drop detections.
	class DrillBank {
	public:
		struct Drill {
			uint64_t id = 0;
			ArenaSnapshot snap;
			float goal[6] = {};
			Team sourceTeam = Team::BLUE;
			int tries = 0, successes = 0;
		};

		struct PracticeWindow {
			bool active = false;
			uint64_t drillId = 0;
			float goal[6] = {};
			Team sourceTeam = Team::BLUE;
			int stepsLeft = 0;
		};

		void Configure(int numArenas, int windowSteps, int maxSize, int maxTries, int minTriesForRetire, float retireSuccessRate) {
			std::lock_guard<std::mutex> lock(mtx);
			windows.assign(std::max(0, numArenas), PracticeWindow{});
			windowStepsCfg = std::max(1, windowSteps);
			maxSizeCfg = std::max(1, maxSize);
			maxTriesCfg = std::max(1, maxTries);
			minTriesForRetireCfg = std::max(1, minTriesForRetire);
			retireSuccessRateCfg = retireSuccessRate;
		}

		// Setter side (called from a thread-pool worker, inside EnvSet::ResetArena): samples a
		// drill uniformly, copies its snapshot out, increments its try count, and arms this
		// arena's practice window. Returns false (bank empty, or arenaIdx not configured) so the
		// caller can fall back to its normal setter.
		bool TryBeginDrill(int arenaIdx, ArenaSnapshot& outSnap, PracticeWindow& outWin) {
			std::lock_guard<std::mutex> lock(mtx);
			if (drills.empty() || arenaIdx < 0 || arenaIdx >= (int)windows.size())
				return false;

			int idx = RocketSim::Math::RandInt(0, (int)drills.size());
			Drill& d = drills[idx];
			d.tries++;

			outSnap = d.snap;
			outWin = PracticeWindow{};
			outWin.active = true;
			outWin.drillId = d.id;
			memcpy(outWin.goal, d.goal, sizeof(outWin.goal));
			outWin.sourceTeam = d.sourceTeam;
			outWin.stepsLeft = windowStepsCfg;

			windows[arenaIdx] = outWin;
			return true;
		}

		// Learner side, called BEFORE envSet->Reset(): terminals still hold the PREVIOUS step's
		// flags (Reset() only zeroes them for arenas that actually reset this step), so this drops
		// any window whose arena is about to reset for a reason OTHER than the drill itself (a
		// goal, a timeout, an old-version handoff) - a drill-triggered reset re-arms its own
		// window from inside TryBeginDrill() right after this runs.
		void ClearWindowsForResets(const std::vector<uint8_t>& terminals) {
			std::lock_guard<std::mutex> lock(mtx);
			for (int i = 0; i < (int)terminals.size() && i < (int)windows.size(); i++)
				if (terminals[i])
					windows[i] = PracticeWindow{};
		}

		PracticeWindow GetWindow(int arenaIdx) const {
			std::lock_guard<std::mutex> lock(mtx);
			if (arenaIdx < 0 || arenaIdx >= (int)windows.size())
				return PracticeWindow{};
			return windows[arenaIdx];
		}

		// Serial, once per collected step.
		void DecrementWindows() {
			std::lock_guard<std::mutex> lock(mtx);
			for (auto& w : windows) {
				if (w.active && --w.stepsLeft <= 0)
					w = PracticeWindow{};
			}
		}

		void AddDrill(const ArenaSnapshot& snap, const float goal[6], Team sourceTeam) {
			std::lock_guard<std::mutex> lock(mtx);
			Drill d;
			d.id = nextId++;
			d.snap = snap;
			memcpy(d.goal, goal, sizeof(d.goal));
			d.sourceTeam = sourceTeam;
			drills.push_back(std::move(d));
			if ((int)drills.size() > maxSizeCfg)
				drills.erase(drills.begin()); // drop oldest
		}

		// Retires a drill once it's been tried enough AND is either reliably solved (success rate
		// at/above the retire threshold) or has exhausted its try budget (a drill nobody ever
		// solves shouldn't occupy the bank forever either).
		void ReportResult(uint64_t drillId, bool success) {
			std::lock_guard<std::mutex> lock(mtx);
			for (auto it = drills.begin(); it != drills.end(); ++it) {
				if (it->id == drillId) {
					if (success)
						it->successes++;
					if (it->tries >= minTriesForRetireCfg) {
						float rate = (float)it->successes / std::max(1, it->tries);
						if (rate >= retireSuccessRateCfg || it->tries >= maxTriesCfg)
							drills.erase(it);
					}
					return;
				}
			}
		}

		int Size() const {
			std::lock_guard<std::mutex> lock(mtx);
			return (int)drills.size();
		}

		float AvgSuccessRate() const {
			std::lock_guard<std::mutex> lock(mtx);
			if (drills.empty())
				return 0;
			double sum = 0;
			for (auto& d : drills)
				sum += d.tries > 0 ? (double)d.successes / d.tries : 0.0;
			return (float)(sum / drills.size());
		}

	private:
		mutable std::mutex mtx;
		std::vector<Drill> drills;
		std::vector<PracticeWindow> windows;
		uint64_t nextId = 1;
		int windowStepsCfg = 90, maxSizeCfg = 512, maxTriesCfg = 20, minTriesForRetireCfg = 5;
		float retireSuccessRateCfg = 0.7f;
	};
}
