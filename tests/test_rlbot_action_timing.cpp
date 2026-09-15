#include "TestFramework.h"
#include "../src/RLBotActionTiming.h"

namespace {
struct Client {
	RLBotActionTiming timing;
	int action = 0, controls = 0, prevAction = 0;
	bool Packet(int elapsed, int skip, int delay) {
		prevAction = controls;
		const bool infer = timing.Advance(elapsed, skip);
		if (infer) ++action;
		if (timing.ShouldApply(delay)) controls = action;
		return infer;
	}
};
}

TEST(RLBot_zero_delay_sends_on_decision_packet) {
	for (int skip : {1, 2, 8}) {
		Client c;
		for (int frame = 0; frame <= 3 * skip; ++frame) {
			CHECK_EQ(c.Packet(frame == 0 ? 12000 : 1, skip, 0), frame % skip == 0);
			CHECK_EQ(c.controls, 1 + frame / skip);
			CHECK_EQ(c.prevAction, frame == 0 ? 0 : 1 + (frame - 1) / skip);
		}
	}
}

TEST(RLBot_delay_is_measured_from_inference) {
	for (int delay : {1, 2, 7}) {
		Client c;
		for (int frame = 0; frame < 24; ++frame) {
			CHECK_EQ(c.Packet(frame == 0 ? 0 : 1, 8, delay), frame % 8 == 0);
			CHECK_EQ(c.controls, frame < delay ? 0 : 1 + (frame - delay) / 8);
		}
	}
}

TEST(RLBot_skipped_packets_decide_immediately_without_backdating_delay) {
	Client c;
	CHECK(c.Packet(0, 8, 2));
	CHECK(!c.Packet(3, 8, 2));
	CHECK_EQ(c.controls, 1);
	CHECK(c.Packet(8, 8, 2)); // Crossed the next decision boundary.
	CHECK_EQ(c.controls, 1); // New decision has not waited two ticks yet.
	CHECK(!c.Packet(1, 8, 2));
	CHECK_EQ(c.controls, 1);
	CHECK(!c.Packet(1, 8, 2));
	CHECK_EQ(c.controls, 2);
	CHECK(c.Packet(24, 8, 0)); // No catch-up decisions on unavailable states.
	CHECK_EQ(c.controls, 3);
}

TEST(RLBot_duplicate_packets_do_not_repeat_inference) {
	Client c;
	CHECK(c.Packet(0, 1, 0));
	CHECK(!c.Packet(0, 1, 0));
	CHECK(!c.Packet(-1, 1, 0));
	CHECK_EQ(c.controls, 1);
	CHECK(c.Packet(1, 1, 0));
	CHECK_EQ(c.controls, 2);
}

TEST(RLBot_variable_holds_expire_on_the_boundary_packet) {
	Client c;
	const int holds[] = {2, 6, 1, 8};
	const int decisions[] = {0, 2, 8, 9, 17};
	int decision = 0, hold = 8;
	for (int frame = 0; frame <= 17; ++frame) {
		const bool infer = c.Packet(frame == 0 ? 0 : 1, hold, 0);
		CHECK_EQ(infer, frame == decisions[decision]);
		if (infer) {
			if (decision < 4) hold = holds[decision];
			++decision;
		}
		CHECK_EQ(c.controls, decision);
	}
	CHECK_EQ(decision, 5);
}
