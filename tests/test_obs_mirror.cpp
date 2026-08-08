#include "TestFramework.h"

#include <RLGymCPP/ObsBuilders/AdvancedObsPadded.h>
#include <RLGymCPP/Gamestates/GameState.h>
#include "../GigaLearnCPP/src/private/GigaLearnCPP/Util/ObsMirror.h"

using namespace RLGC;

// PHYSICAL equivalence test for the composite value critic's mirror augmentation:
//   ObsMirror::Apply(map, obs(state)) == obs(mirror(state))
// for hand-mirrored GameStates (x -> -x), including nontrivial rotations, angular
// velocities (pseudovectors), steer/yaw/roll in prevAction, asymmetric boost-pad
// state, and both team canonical frames. A wrong mirror is SILENT training
// corruption; this test is the only thing standing between the sign conventions
// and the value loss.

static Player MirrorPlayer(const Player& p) {
	Player m = p;
	m.pos = Vec(-p.pos.x, p.pos.y, p.pos.z);
	m.vel = Vec(-p.vel.x, p.vel.y, p.vel.z);
	m.angVel = Vec(p.angVel.x, -p.angVel.y, -p.angVel.z);       // pseudovector
	m.rotMat.forward = Vec(-p.rotMat.forward.x, p.rotMat.forward.y, p.rotMat.forward.z);
	m.rotMat.up = Vec(-p.rotMat.up.x, p.rotMat.up.y, p.rotMat.up.z);
	m.rotMat.right = Vec(p.rotMat.right.x, -p.rotMat.right.y, -p.rotMat.right.z); // -M*r
	m.prevAction.steer = -p.prevAction.steer;
	m.prevAction.yaw = -p.prevAction.yaw;
	m.prevAction.roll = -p.prevAction.roll;
	return m;
}

static int MirrorPadIndex(int i) {
	const Vec& p = CommonValues::BOOST_LOCATIONS[i];
	int best = -1; float bestD = 10.f;
	for (int j = 0; j < CommonValues::BOOST_LOCATIONS_AMOUNT; j++) {
		const Vec& q = CommonValues::BOOST_LOCATIONS[j];
		float d = std::abs(q.x + p.x) + std::abs(q.y - p.y) + std::abs(q.z - p.z);
		if (d < bestD) { bestD = d; best = j; }
	}
	return best;
}

static GameState MirrorState(const GameState& gs) {
	GameState m = gs;
	m.ball.pos = Vec(-gs.ball.pos.x, gs.ball.pos.y, gs.ball.pos.z);
	m.ball.vel = Vec(-gs.ball.vel.x, gs.ball.vel.y, gs.ball.vel.z);
	m.ball.angVel = Vec(gs.ball.angVel.x, -gs.ball.angVel.y, -gs.ball.angVel.z);
	for (auto& p : m.players)
		p = MirrorPlayer(p);
	// pads move to their x-mirrored partner (both frames)
	for (int i = 0; i < CommonValues::BOOST_LOCATIONS_AMOUNT; i++) {
		int j = MirrorPadIndex(i);
		m.boostPads[j] = gs.boostPads[i];
		m.boostPadsInv[j] = gs.boostPadsInv[i];
		m.boostPadTimers[j] = gs.boostPadTimers[i];
		m.boostPadTimersInv[j] = gs.boostPadTimersInv[i];
	}
	return m;
}

static Player MakeRichPlayer(uint32_t carId, Team team, float yawAngle, Vec pos) {
	Player p = {};
	p.carId = carId;
	p.team = team;
	p.pos = pos;
	p.vel = Vec(320, -410, 55);
	p.angVel = Vec(0.5f, -0.3f, 0.8f);
	float c = cosf(yawAngle), s = sinf(yawAngle);
	p.rotMat.forward = Vec(c, s, 0);
	p.rotMat.right = Vec(-s, c, 0);
	p.rotMat.up = Vec(0, 0, 1);
	p.boost = 63;
	p.isOnGround = false;
	p.hasJumped = true;
	p.prevAction.throttle = 0.5f;
	p.prevAction.steer = 0.7f;
	p.prevAction.pitch = -0.2f;
	p.prevAction.yaw = 0.4f;
	p.prevAction.roll = -0.6f;
	p.prevAction.jump = 1.f;
	p.prevAction.boost = 0.f;
	p.prevAction.handbrake = 1.f;
	return p;
}

TEST(ObsMirror_PhysicalEquivalence) {
	constexpr int MAX_PER_TEAM = 3;
	// shuffleSlots OFF: slot assignment must be deterministic for the comparison
	AdvancedObsPadded builder(MAX_PER_TEAM, /*shuffleSlots=*/false);

	GameState gs = {};
	gs.players.push_back(MakeRichPlayer(1, Team::BLUE, 0.6f, Vec(-900, -2800, 120)));
	gs.players.push_back(MakeRichPlayer(2, Team::BLUE, -1.2f, Vec(1400, -500, 17)));
	gs.players.push_back(MakeRichPlayer(101, Team::ORANGE, 2.1f, Vec(300, 2600, 400)));
	gs.players.push_back(MakeRichPlayer(102, Team::ORANGE, -0.4f, Vec(-2200, 1800, 17)));
	for (int i = 0; i < (int)gs.players.size(); i++)
		gs.players[i].index = i;
	gs.ball.pos = Vec(520, 310, 210);
	gs.ball.vel = Vec(-640, 850, 120);
	gs.ball.angVel = Vec(1.2f, -0.7f, 2.3f);
	gs.boostPads = std::vector<bool>(CommonValues::BOOST_LOCATIONS_AMOUNT, true);
	gs.boostPadsInv = gs.boostPads;
	gs.boostPadTimers.assign(CommonValues::BOOST_LOCATIONS_AMOUNT, 0.f);
	gs.boostPadTimersInv.assign(CommonValues::BOOST_LOCATIONS_AMOUNT, 0.f);
	// asymmetric pad state so the permutation is actually exercised
	gs.boostPads[3] = false;  gs.boostPadTimers[3] = 4.f;
	gs.boostPads[18] = false; gs.boostPadTimers[18] = 1.5f;
	gs.boostPadsInv[7] = false; gs.boostPadTimersInv[7] = 2.5f;

	GameState mg = MirrorState(gs);

	for (int pIdx : { 0, 2 }) {  // one BLUE (identity frame), one ORANGE (inverted frame)
		FList a = builder.BuildObs(gs.players[pIdx], gs);
		FList b = builder.BuildObs(mg.players[pIdx], mg);
		CHECK((int)a.size() == (int)b.size());

		auto map = GGL::ObsMirror::Build(MAX_PER_TEAM, (int)a.size());
		auto ta = torch::tensor(std::vector<float>(a.begin(), a.end())).unsqueeze(0);
		auto mirrored = GGL::ObsMirror::Apply(map, ta).squeeze(0);

		for (int i = 0; i < (int)b.size(); i++)
			CHECK_NEAR(mirrored[i].item<float>(), b[i], 1e-4f);
	}
}
