#pragma once

#include "StateSetter.h"

#include <atomic>
#include <mutex>
#include <unordered_map>

namespace RLGC {

	// Minimal car physics for a frontier reset (everything RandomState-style setters
	// reconstruct; jump/flip/contact state is deliberately reset like RandomState does).
	struct FrontierCarSnapshot {
		Vec pos, vel, angVel;
		RotMat rotMat;
		float boost = 0;
		Team team = Team::BLUE;
	};

	// One harvested arena state: ball + cars (in GameState player order) + boost pads
	// (in CommonValues::BOOST_LOCATIONS order, matching GameState.boostPads).
	struct FrontierSnapshot {
		BallState ball = {};
		std::vector<FrontierCarSnapshot> cars;
		std::vector<bool> padsActive;
		std::vector<float> padTimers;
		int gameMode = 0; // (int)Arena::gameMode, checked at sample time so a snapshot
		                  // can't be replayed into an incompatible arena
	};

	// Fixed-capacity FIFO ring of frontier reset candidates.
	// Written by the GigaLearn learner thread (batched, once per training iteration),
	// read by FrontierState on g_ThreadPool worker threads during arena resets. A single
	// mutex is plenty: resets sample tens of ~400B snapshots per env step and the learner
	// inserts at most ~1k once per multi-second iteration.
	class FrontierStateBuffer {
	public:
		// Reset-source tags for the episode-return comparison metric
		// (Frontier/Episode Return ...). Written by the state setters at reset time,
		// read by the collector when an episode ends.
		enum : uint8_t {
			TAG_KICKOFF = 0,
			TAG_RANDOM = 1,
			TAG_FRONTIER = 2,
			TAG_FALLBACK = 3, // FrontierState fell back to its child setter
			TAG_NONE = 255
		};

		// Reset-source counters, read+zeroed by the learner each iteration
		// (Frontier/Reset Fraction Actual).
		std::atomic<uint64_t> frontierResets = 0, fallbackResets = 0;

		explicit FrontierStateBuffer(int capacity);

		void Insert(const FrontierSnapshot& snap); // ring write, FIFO eviction
		bool Sample(FrontierSnapshot& out) const;  // uniform random copy; false if empty
		size_t Size() const;

		// Ball in play, cars not demoed (filtered at capture), everything within arena
		// bounds at sane velocity. High critic uncertainty also flags garbage states;
		// this keeps them out of the reset pool.
		static bool PassesSanityFilter(const FrontierSnapshot& snap);

		void TagArenaReset(const Arena* arena, uint8_t tag);
		uint8_t GetArenaTag(const Arena* arena) const; // TAG_NONE if never tagged

	private:
		mutable std::mutex mutex;
		std::vector<FrontierSnapshot> ring;
		size_t capacity;
		size_t writeIdx = 0, count = 0;
		std::unordered_map<const Arena*, uint8_t> arenaTags;
	};

	// Samples uniformly from a FrontierStateBuffer; falls back to `fallback` (e.g.
	// RandomState) while the buffer is below minFill or when no sampled snapshot is
	// compatible with the arena (game mode / per-team car counts).
	class FrontierState : public StateSetter {
	public:
		FrontierStateBuffer* buffer;
		StateSetter* fallback;
		int minFill;

		FrontierState(FrontierStateBuffer* buffer, StateSetter* fallback, int minFill) :
			buffer(buffer), fallback(fallback), minFill(minFill) {
		}

		virtual void ResetArena(Arena* arena) override;
	};
}
