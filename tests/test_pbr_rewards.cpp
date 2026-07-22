#include "TestFramework.h"

#include <RLGymCPP/Rewards/CommonRewards.h>
#include <RLGymCPP/Rewards/ZeroSumReward.h>
#include <RLGymCPP/Gamestates/GameState.h>

#include <cmath>

using namespace RLGC;

// PBRS semantics for the two 2026-07-21 aerial rewards (ConsecutiveAirTouchReward,
// WallJumpToBallReward). The point of these tests is the UNFARMABLE guarantee the
// user asked for: with gamma == 1 an exact potential telescopes to Phi(end) -
// Phi(start), so any trajectory that returns to the starting (augmented) state nets
// EXACTLY 0 - no grind can extract free reward. The gamma < 1 cases pin the local
// credit shape (positive on skill progress, refunded on release).

// One-player frame; drives a stateful reward step by step. Addresses are kept stable
// by reserving the states vector up front, so prev pointers stay valid.
struct Frame {
	Vec carPos = Vec(0, 0, 17);
	bool onGround = false;
	bool hasContact = false;
	Vec normal = Vec(0, 0, 1);
	Vec ballPos = Vec(0, 0, 93);
	bool touched = false;
	bool demoed = false;
};

// Runs a single-player sequence through `r` and returns the per-frame reward. Frame 0
// is the initial (post-Reset) state and always sees a null prev, matching the live loop.
static std::vector<float> RunSingle(Reward& r, const std::vector<Frame>& frames) {
	std::vector<GameState> states;
	states.reserve(frames.size());
	for (const Frame& f : frames) {
		GameState gs;
		Player p = {};
		p.index = 0;
		p.carId = 1;
		p.team = Team::BLUE;
		p.pos = f.carPos;
		p.isOnGround = f.onGround;
		p.worldContact.hasContact = f.hasContact;
		p.worldContact.contactNormal = f.normal;
		p.ballTouchedStep = f.touched;
		p.isDemoed = f.demoed;
		gs.players.push_back(p);
		gs.ball.pos = f.ballPos;
		states.push_back(std::move(gs));
	}
	for (size_t i = 1; i < states.size(); i++) {
		states[i].prev = &states[i - 1];
		states[i].players[0].prev = &states[i - 1].players[0];
	}
	r.Reset(states[0]);
	std::vector<float> out;
	for (size_t i = 0; i < states.size(); i++)
		out.push_back(r.GetReward(states[i].players[0], states[i], false));
	return out;
}

static float Sum(const std::vector<float>& v) {
	float s = 0;
	for (float x : v)
		s += x;
	return s;
}

// ---------------------------------------------------------------------------
// ConsecutiveAirTouchReward
// ---------------------------------------------------------------------------

TEST(ConsecAir_phi_shape) {
	// Streak 0 and 1 pay nothing (a lone air touch is AerialTouch's job); credit
	// begins on the 2nd chained touch and saturates toward 1.
	CHECK_NEAR(ConsecutiveAirTouchReward::Phi(0), 0.f, 1e-9f);
	CHECK_NEAR(ConsecutiveAirTouchReward::Phi(1), 0.f, 1e-9f);
	CHECK_NEAR(ConsecutiveAirTouchReward::Phi(2), 1.f - expf(-0.5f), 1e-6f);
	CHECK_NEAR(ConsecutiveAirTouchReward::Phi(3), 1.f - expf(-1.0f), 1e-6f);
	CHECK(ConsecutiveAirTouchReward::Phi(3) > ConsecutiveAirTouchReward::Phi(2));
	CHECK(ConsecutiveAirTouchReward::Phi(10) < 1.f);
	// Diminishing marginal credit (saturating): 3rd touch pays less than the 2nd
	float d2 = ConsecutiveAirTouchReward::Phi(2) - ConsecutiveAirTouchReward::Phi(1);
	float d3 = ConsecutiveAirTouchReward::Phi(3) - ConsecutiveAirTouchReward::Phi(2);
	CHECK(d3 < d2);
}

TEST(ConsecAir_local_credit_shape) {
	ConsecutiveAirTouchReward r(0.99f);
	auto air = [](bool touch) { Frame f; f.onGround = false; f.touched = touch; return f; };
	Frame ground; ground.onGround = true;

	std::vector<Frame> frames = {
		air(true),   // 1: streak 0->1, first air touch pays 0
		air(true),   // 2: streak 1->2, first CHAIN credit, positive
		air(true),   // 3: streak 2->3, more (but smaller) credit
		air(false),  // 4: airborne, no touch -> tiny negative bleed
		ground,      // 5: land -> full refund, negative
	};
	auto rew = RunSingle(r, frames);

	CHECK_NEAR(rew[0], 0.f, 1e-6f);                          // lone air touch: 0
	CHECK(rew[1] > 0.2f);                                    // 2nd consecutive: real positive credit
	CHECK(rew[2] > 0.f);                                     // 3rd: still positive
	CHECK(rew[2] < rew[1]);                                  // ...but smaller (saturating)
	CHECK(rew[3] < 0.f && rew[3] > -0.01f);                 // holding a banked chain slowly bleeds
	CHECK(rew[4] < -0.5f);                                   // landing refunds the whole chain
}

TEST(ConsecAir_unfarmable_cycle_nets_zero) {
	// With gamma == 1 an exact potential telescopes: build any chain, land, and the
	// undiscounted sum is Phi(start) - Phi(end) == 0. Grinding cannot extract reward.
	ConsecutiveAirTouchReward r(1.0f);
	auto air = [](bool touch) { Frame f; f.onGround = false; f.touched = touch; return f; };
	Frame ground; ground.onGround = true;

	std::vector<Frame> frames = {
		air(true), air(true), air(true), air(true), air(false), air(true), ground,
	};
	auto rew = RunSingle(r, frames);
	CHECK_NEAR(Sum(rew), 0.f, 1e-5f);

	// Two full grind cycles also net zero (no per-cycle accumulation)
	std::vector<Frame> twoCycles = {
		air(true), air(true), air(true), ground,
		air(true), air(true), air(true), ground,
	};
	CHECK_NEAR(Sum(RunSingle(r, twoCycles)), 0.f, 1e-5f);
}

TEST(ConsecAir_ground_and_wall_break_chain) {
	// isOnGround is true on walls too, so landing on either resets the streak.
	ConsecutiveAirTouchReward r(1.0f);
	auto air = [](bool touch) { Frame f; f.onGround = false; f.touched = touch; return f; };
	Frame wall; wall.onGround = true; wall.hasContact = true; wall.normal = Vec(1, 0, 0);

	std::vector<Frame> frames = { air(true), air(true), air(true), wall };
	auto rew = RunSingle(r, frames);
	// The wall frame refunds the chain exactly (streak -> 0), so the whole thing nets 0
	CHECK_NEAR(Sum(rew), 0.f, 1e-5f);
	CHECK(rew[3] < -0.5f); // the wall landing is the refund
}

TEST(ConsecAir_demo_guard) {
	// A demoed step pays nothing and drops the chain (teleport, not the player's action).
	ConsecutiveAirTouchReward r(0.99f);
	auto air = [](bool touch) { Frame f; f.onGround = false; f.touched = touch; return f; };
	Frame demoed; demoed.onGround = false; demoed.demoed = true;

	std::vector<Frame> frames = { air(true), air(true), demoed, air(true) };
	auto rew = RunSingle(r, frames);
	CHECK_NEAR(rew[2], 0.f, 1e-9f);          // demoed step: exactly 0, no refund charged
	// After respawn the chain has been reset: the next lone air touch pays 0 again
	CHECK_NEAR(rew[3], 0.f, 1e-6f);
}

// ---------------------------------------------------------------------------
// WallJumpToBallReward
// ---------------------------------------------------------------------------

TEST(WallJump_onwall_detection) {
	auto mk = [](bool onGround, bool hasContact, Vec normal) {
		Player p = {};
		p.isOnGround = onGround;
		p.worldContact.hasContact = hasContact;
		p.worldContact.contactNormal = normal;
		return p;
	};
	// Flat ground: normal up -> not a wall
	CHECK(!WallJumpToBallReward::OnWall(mk(true, true, Vec(0, 0, 1))));
	// Side wall: horizontal normal -> wall
	CHECK(WallJumpToBallReward::OnWall(mk(true, true, Vec(1, 0, 0))));
	CHECK(WallJumpToBallReward::OnWall(mk(true, true, Vec(0, 1, 0))));
	// Ceiling: normal down -> not a wall
	CHECK(!WallJumpToBallReward::OnWall(mk(true, true, Vec(0, 0, -1))));
	// Airborne (no world contact) -> not a wall even if a stale normal lingers
	CHECK(!WallJumpToBallReward::OnWall(mk(false, false, Vec(1, 0, 0))));
}

TEST(WallJump_local_credit_and_unfarmable) {
	// Wall launch toward the ball, close in, land. gamma == 1 so the full climb-and-
	// land cycle telescopes to exactly 0 (unfarmable), with positive credit while
	// closing and a full refund on landing.
	WallJumpToBallReward r(1.0f);
	Vec ball(2000, 0, 500);
	Frame onWall;  onWall.onGround = true; onWall.hasContact = true; onWall.normal = Vec(1, 0, 0);
	onWall.carPos = Vec(4096, 0, 500); onWall.ballPos = ball;
	auto air = [&](Vec pos, bool touch) { Frame f; f.onGround = false; f.carPos = pos; f.ballPos = ball; f.touched = touch; return f; };
	Frame land; land.onGround = true; land.hasContact = true; land.normal = Vec(0, 0, 1);
	land.carPos = Vec(2000, 0, 17); land.ballPos = ball;

	std::vector<Frame> frames = {
		onWall,                       // 0: on wall (prev null) -> 0
		air(Vec(3500, 0, 550), false),// 1: launched off wall, still far
		air(Vec(2900, 0, 520), false),// 2: closing
		air(Vec(2200, 0, 505), true), // 3: strike, very close
		land,                         // 4: land -> refund
	};
	auto rew = RunSingle(r, frames);

	CHECK_NEAR(rew[0], 0.f, 1e-9f);   // on-wall / first frame: no potential diff yet
	CHECK(rew[1] > 0.f);              // launching toward the ball pays as the latch opens near it
	CHECK(rew[2] > 0.f);              // closing further pays more
	CHECK(rew[3] > 0.f);             // final approach to the strike still positive
	CHECK(rew[4] < 0.f);             // landing clears the latch: full refund
	CHECK_NEAR(Sum(rew), 0.f, 1e-5f);// unfarmable: the whole cycle nets zero
}

TEST(WallJump_no_wall_no_reward) {
	// A plain jump off the FLAT ground (prev normal up) never opens the latch, so the
	// identical aerial approach pays nothing here.
	WallJumpToBallReward r(1.0f);
	Vec ball(2000, 0, 500);
	Frame ground; ground.onGround = true; ground.hasContact = true; ground.normal = Vec(0, 0, 1);
	ground.carPos = Vec(4096, 0, 17); ground.ballPos = ball;
	auto air = [&](Vec pos) { Frame f; f.onGround = false; f.carPos = pos; f.ballPos = ball; return f; };

	std::vector<Frame> frames = { ground, air(Vec(3500, 0, 400)), air(Vec(2900, 0, 450)), air(Vec(2200, 0, 500)) };
	auto rew = RunSingle(r, frames);
	for (float x : rew)
		CHECK_NEAR(x, 0.f, 1e-9f);
}

TEST(WallJump_leaving_wall_away_from_ball_pays_negligible) {
	// Launching off a wall AWAY from a distant ball pays ~0 (proximity term ~ exp(-far)).
	WallJumpToBallReward r(1.0f);
	Vec ball(-3000, 0, 100); // opposite side of the field from the wall exit
	Frame onWall; onWall.onGround = true; onWall.hasContact = true; onWall.normal = Vec(1, 0, 0);
	onWall.carPos = Vec(4096, 0, 500); onWall.ballPos = ball;
	auto air = [&](Vec pos) { Frame f; f.onGround = false; f.carPos = pos; f.ballPos = ball; return f; };

	std::vector<Frame> frames = { onWall, air(Vec(3800, 0, 600)), air(Vec(3500, 0, 700)) };
	auto rew = RunSingle(r, frames);
	for (float x : rew)
		CHECK(std::abs(x) < 0.05f);
}

TEST(WallJump_demo_guard) {
	// A demo mid-flight drops the latch and is not charged.
	WallJumpToBallReward r(0.99f);
	Vec ball(2000, 0, 500);
	Frame onWall; onWall.onGround = true; onWall.hasContact = true; onWall.normal = Vec(1, 0, 0);
	onWall.carPos = Vec(4096, 0, 500); onWall.ballPos = ball;
	auto air = [&](Vec pos, bool demo) { Frame f; f.onGround = false; f.carPos = pos; f.ballPos = ball; f.demoed = demo; return f; };

	std::vector<Frame> frames = { onWall, air(Vec(3500, 0, 550), false), air(Vec(2900, 0, 520), true) };
	auto rew = RunSingle(r, frames);
	CHECK_NEAR(rew[2], 0.f, 1e-9f); // demoed step: exactly 0
}

// ---------------------------------------------------------------------------
// Zero-sum wrapping (whole-stack invariant) — both new terms sum to 0 across a 1v1.
// ---------------------------------------------------------------------------

TEST(PBR_zero_sum_1v1_sums_to_zero) {
	// Build a 1v1 where blue (idx 0) is chaining air touches and orange (idx 1) is idle.
	// A per-player potential wrapped in ZeroSumReward is antisymmetric in 1v1 -> sums to 0.
	GameState prev, cur;
	auto mk = [](int idx, Team t, Vec pos, bool onGround, bool touched) {
		Player p = {};
		p.index = idx; p.carId = idx + 1; p.team = t;
		p.pos = pos; p.isOnGround = onGround; p.ballTouchedStep = touched;
		return p;
	};
	prev.players = { mk(0, Team::BLUE, Vec(0, 0, 600), false, true), mk(1, Team::ORANGE, Vec(0, 500, 17), true, false) };
	cur.players  = { mk(0, Team::BLUE, Vec(0, 0, 600), false, true), mk(1, Team::ORANGE, Vec(0, 500, 17), true, false) };
	prev.ball.pos = Vec(0, 0, 600);
	cur.ball.pos = Vec(0, 0, 600);
	cur.prev = &prev;
	cur.players[0].prev = &prev.players[0];
	cur.players[1].prev = &prev.players[1];

	for (Reward* child : { (Reward*)new ConsecutiveAirTouchReward(0.99f), (Reward*)new WallJumpToBallReward(0.99f) }) {
		Reward* zs = new ZeroSumReward(child, 0.3f);
		zs->Reset(prev);
		zs->PreStep(cur);
		auto rews = zs->GetAllRewards(cur, false);
		CHECK_NEAR(rews[0] + rews[1], 0.f, 1e-5f);
		delete zs;
	}
}
