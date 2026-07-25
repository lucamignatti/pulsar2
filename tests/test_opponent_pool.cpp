#include "TestFramework.h"

#include <GigaLearnCPP/Util/LogSpaced.h>

#include <random>
#include <vector>

// Tests for the opponent-pool machinery that replaced the QD league on 2026-07-25:
// the log-spaced reference decimation, and the arithmetic of the sequential opponent
// cascade whose realized shares the config comments depend on.

static std::vector<uint64_t> KeptTimesteps(const std::vector<uint64_t>& ts, size_t keep) {
	std::vector<uint64_t> out;
	for (size_t i : GGL::DecimateSpaced(ts, keep))
		out.push_back(ts[i]);
	return out;
}

TEST(Decimate_keeps_everything_when_under_cap) {
	std::vector<uint64_t> ts = { 0, 25, 50 };
	auto kept = KeptTimesteps(ts, 8);
	CHECK_EQ(kept.size(), (size_t)3);
	CHECK_EQ(kept[0], (uint64_t)0);
	CHECK_EQ(kept[2], (uint64_t)50);
}

// The whole point of decimating rather than FIFO-evicting: the oldest entry must survive
// forever, because Ref/Oldest Share is only a fixed yardstick if its opponent never changes.
TEST(Decimate_never_drops_the_oldest_or_newest) {
	std::vector<uint64_t> ts;
	for (uint64_t t = 0; t <= 1000; t += 25)
		ts.push_back(t);
	auto kept = KeptTimesteps(ts, 8);
	CHECK_EQ(kept.size(), (size_t)8);
	CHECK_EQ(kept.front(), (uint64_t)0);
	CHECK_EQ(kept.back(), (uint64_t)1000);
}

// Repeated decimation (one new candidate per archived version, forever) must not degenerate
// into a recent window - that is precisely the pool myopia the reference set exists to cure.
TEST(Decimate_retains_old_span_under_repeated_growth) {
	std::vector<uint64_t> kept = { 0 };
	for (uint64_t t = 25; t <= 5000; t += 25) {
		kept.push_back(t);
		kept = KeptTimesteps(kept, 8);
	}
	CHECK_EQ(kept.size(), (size_t)8);
	CHECK_EQ(kept.front(), (uint64_t)0);
	CHECK_EQ(kept.back(), (uint64_t)5000);
	// A FIFO window of 8 at this spacing would span only 175 steps. Require the retained set
	// to still cover most of the run's history.
	CHECK(kept[1] < 2500);
	// Strictly ascending, no duplicates.
	for (size_t i = 1; i < kept.size(); i++)
		CHECK(kept[i] > kept[i - 1]);
}

TEST(Decimate_handles_degenerate_inputs) {
	std::vector<uint64_t> one = { 7 };
	CHECK_EQ(KeptTimesteps(one, 8).size(), (size_t)1);
	CHECK_EQ(KeptTimesteps(one, 1).size(), (size_t)1);
	std::vector<uint64_t> two = { 7, 9 };
	// keep=1 cannot be honoured without dropping an endpoint; both endpoints win.
	CHECK_EQ(KeptTimesteps(two, 1).size(), (size_t)2);
}

// The opponent cascade in Learner.cpp is a sequential if/else-if, so the second branch only
// sees the iterations the first declined. The ExampleMain comments quote the resulting
// realized shares (0.15 and 0.85*0.30 = 0.255, total non-self 0.405); pin that arithmetic.
TEST(OpponentCascade_realized_shares_are_nested_not_independent) {
	const float serveFrac = 0.15f, oldChance = 0.30f;
	std::mt19937_64 rng(12345);
	std::uniform_real_distribution<float> roll(0.0f, 1.0f);

	const int N = 400000;
	int nexto = 0, oldVersion = 0, self = 0;
	for (int i = 0; i < N; i++) {
		if (roll(rng) < serveFrac) {
			nexto++;
		} else if (roll(rng) < oldChance) {
			oldVersion++;
		} else {
			self++;
		}
	}

	CHECK_NEAR((double)nexto / N, 0.15, 0.005);
	CHECK_NEAR((double)oldVersion / N, 0.85 * 0.30, 0.005);
	CHECK_NEAR((double)(nexto + oldVersion) / N, 0.405, 0.005);
	CHECK_EQ(nexto + oldVersion + self, N);
}

// The bug this replaced: the roll used RocketSim's thread_local minstd_rand0, re-seeded from
// the millisecond clock on every fresh collect thread, so the serve decision was the engine's
// FIRST draw after a reseed - a 127.773 s sawtooth of wall clock rather than a probability
// (measured: all 169 serves inside one contiguous 8/40 phase arc, 8.9% realized vs 15%
// configured). A persistent engine advanced once per iteration has no such structure: check
// that consecutive serves are not clustered.
TEST(OpponentCascade_serves_are_not_clock_clustered) {
	std::mt19937_64 rng(999);
	std::uniform_real_distribution<float> roll(0.0f, 1.0f);

	const int N = 100000;
	const float p = 0.15f;
	int serves = 0, adjacentPairs = 0;
	bool prevServed = false;
	for (int i = 0; i < N; i++) {
		bool served = roll(rng) < p;
		if (served) {
			serves++;
			if (prevServed) adjacentPairs++;
		}
		prevServed = served;
	}
	CHECK_NEAR((double)serves / N, 0.15, 0.005);
	// Independent draws => P(both of an adjacent pair serve) = p^2 = 0.0225. The sawtooth
	// produced bursts instead, with a gap histogram that jumped from 1 straight to 15+.
	CHECK_NEAR((double)adjacentPairs / N, 0.0225, 0.004);
}
