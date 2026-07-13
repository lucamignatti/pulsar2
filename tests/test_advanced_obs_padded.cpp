#include "TestFramework.h"

#include <RLGymCPP/ObsBuilders/AdvancedObs.h>
#include <RLGymCPP/ObsBuilders/AdvancedObsPadded.h>
#include <RLGymCPP/Gamestates/GameState.h>
#include <RLGymCPP/Gamestates/StateUtil.h>

using namespace RLGC;

// Layout constants under test. HEADER must stay in sync with AdvancedObs's header
// (ball 9 + prevAction + boost pads); PLAYER_ELEMS with AddPlayerToObs.
static const int HEADER = 9 + (int)Action::ELEM_AMOUNT + (int)CommonValues::BOOST_LOCATIONS_AMOUNT;
static const int PLAYER_ELEMS = 29;

static Player MakePlayer(uint32_t carId, Team team, Vec pos, Vec vel, RotMat rot, float boost = 50) {
	Player p = {};
	p.carId = carId;
	p.index = (int)carId - 1;
	p.team = team;
	p.pos = pos;
	p.vel = vel;
	p.angVel = Vec(0, 0, 0);
	p.rotMat = rot;
	p.boost = boost;
	p.isOnGround = true;
	return p;
}

static GameState MakeState(std::vector<Player> players, Vec ballPos, Vec ballVel) {
	GameState gs = {};
	for (int i = 0; i < (int)players.size(); i++)
		players[i].index = i;
	gs.players = std::move(players);
	gs.ball.pos = ballPos;
	gs.ball.vel = ballVel;
	gs.ball.angVel = Vec(0, 0, 0);
	return gs;
}

static GameState MakeStateNvN(int playersPerTeam) {
	std::vector<Player> players;
	for (int i = 0; i < playersPerTeam; i++) {
		players.push_back(MakePlayer(1 + i, Team::BLUE,
			Vec(-1000.f + 500.f * i, -3000.f, 17.f), Vec(100.f * i, 200.f, 0), RotMat::GetIdentity(), 30 + 10.f * i));
		players.push_back(MakePlayer(101 + i, Team::ORANGE,
			Vec(800.f - 400.f * i, 2500.f, 17.f), Vec(-50.f * i, -300.f, 0), RotMat::GetIdentity(), 80 - 10.f * i));
	}
	return MakeState(std::move(players), Vec(0, 500, 93), Vec(300, -400, 0));
}

static bool BlockIsZero(const FList& obs, int start, int count) {
	for (int i = start; i < start + count; i++)
		if (obs[i] != 0)
			return false;
	return true;
}

static bool BlocksEqual(const FList& obs, int start, const FList& expected) {
	for (int i = 0; i < (int)expected.size(); i++)
		if (obs[start + i] != expected[i])
			return false;
	return true;
}

TEST(test_padded_obs_fixed_size_across_team_sizes) {
	AdvancedObs unpadded = {};
	AdvancedObsPadded padded(3);

	auto gs1 = MakeStateNvN(1);
	auto gs2 = MakeStateNvN(2);
	auto gs3 = MakeStateNvN(3);

	CHECK_EQ((int)unpadded.BuildObs(gs1.players[0], gs1).size(), HEADER + 2 * PLAYER_ELEMS); // 109 today

	const int expectedPadded = HEADER + 6 * PLAYER_ELEMS + 5; // self + 2 tm + 3 opp slots + 5 flags
	CHECK_EQ((int)padded.BuildObs(gs1.players[0], gs1).size(), expectedPadded);
	CHECK_EQ((int)padded.BuildObs(gs2.players[0], gs2).size(), expectedPadded);
	CHECK_EQ((int)padded.BuildObs(gs3.players[0], gs3).size(), expectedPadded);
	// Every seat, every team size, same width (what EnvSet's single obsSize requires)
	for (auto* gs : { &gs1, &gs2, &gs3 })
		for (auto& p : gs->players)
			CHECK_EQ((int)padded.BuildObs(p, *gs).size(), expectedPadded);

	AdvancedObsPadded padded2(2);
	CHECK_EQ((int)padded2.BuildObs(gs2.players[0], gs2).size(), HEADER + 4 * PLAYER_ELEMS + 3);
}

TEST(test_padded_header_and_self_match_unpadded) {
	// The steering landing sims and analysis tooling read ball@0 and self right after the
	// header — the padded builder must keep that prefix byte-identical to AdvancedObs.
	AdvancedObs unpadded = {};
	AdvancedObsPadded padded(3);

	auto gs = MakeStateNvN(1);
	for (auto& p : gs.players) {
		FList u = unpadded.BuildObs(p, gs);
		FList pd = padded.BuildObs(p, gs);
		for (int i = 0; i < HEADER + PLAYER_ELEMS; i++)
			CHECK_EQ(pd[i], u[i]);
	}
}

TEST(test_padded_1v1_slots_and_presence_flags) {
	AdvancedObs unpadded = {};
	AdvancedObsPadded padded(3);

	auto gs = MakeStateNvN(1);
	auto& self = gs.players[0];

	FList u = unpadded.BuildObs(self, gs);
	FList pd = padded.BuildObs(self, gs);

	// Expected opponent block = the single other-player block of the unpadded obs
	FList oppBlock(u.begin() + HEADER + PLAYER_ELEMS, u.begin() + HEADER + 2 * PLAYER_ELEMS);

	const int tmStart = HEADER + PLAYER_ELEMS;
	const int oppStart = tmStart + 2 * PLAYER_ELEMS;
	const int flagStart = oppStart + 3 * PLAYER_ELEMS;

	// No teammates: both teammate slots zero, flags 0
	CHECK(BlockIsZero(pd, tmStart, 2 * PLAYER_ELEMS));
	CHECK_EQ(pd[flagStart + 0], 0.f);
	CHECK_EQ(pd[flagStart + 1], 0.f);

	// Exactly one opponent slot holds the real opponent, flag agrees, others zero
	int found = -1;
	for (int slot = 0; slot < 3; slot++) {
		int start = oppStart + slot * PLAYER_ELEMS;
		float flag = pd[flagStart + 2 + slot];
		if (BlocksEqual(pd, start, oppBlock) && flag == 1.f) {
			CHECK(found == -1);
			found = slot;
		} else {
			CHECK(BlockIsZero(pd, start, PLAYER_ELEMS));
			CHECK_EQ(flag, 0.f);
		}
	}
	CHECK(found != -1);
}

TEST(test_padded_2v2_slot_contents) {
	AdvancedObsPadded padded(3);

	auto gs = MakeStateNvN(2);
	auto& self = gs.players[0]; // BLUE, so inv = false
	CHECK(self.team == Team::BLUE);

	// Expected blocks straight from AddPlayerToObs (the padded builder must place these
	// verbatim into slots)
	FList tmBlock = {}, opp1Block = {}, opp2Block = {};
	std::vector<FList*> expectedOpps = { &opp1Block, &opp2Block };
	int oppSeen = 0;
	for (auto& p : gs.players) {
		if (p.carId == self.carId)
			continue;
		if (p.team == self.team)
			padded.AddPlayerToObs(tmBlock, p, false, gs.ball);
		else
			padded.AddPlayerToObs(*expectedOpps[oppSeen++], p, false, gs.ball);
	}
	CHECK_EQ(oppSeen, 2);

	FList pd = padded.BuildObs(self, gs);

	const int tmStart = HEADER + PLAYER_ELEMS;
	const int oppStart = tmStart + 2 * PLAYER_ELEMS;
	const int flagStart = oppStart + 3 * PLAYER_ELEMS;

	// Teammate appears in exactly one of the 2 teammate slots; the other is zero
	int tmFound = 0;
	for (int slot = 0; slot < 2; slot++) {
		int start = tmStart + slot * PLAYER_ELEMS;
		float flag = pd[flagStart + slot];
		if (BlocksEqual(pd, start, tmBlock) && flag == 1.f) {
			tmFound++;
		} else {
			CHECK(BlockIsZero(pd, start, PLAYER_ELEMS));
			CHECK_EQ(flag, 0.f);
		}
	}
	CHECK_EQ(tmFound, 1);

	// Both opponents appear in 2 of the 3 opponent slots; the third is zero
	int oppFound = 0;
	for (int slot = 0; slot < 3; slot++) {
		int start = oppStart + slot * PLAYER_ELEMS;
		float flag = pd[flagStart + 2 + slot];
		if ((BlocksEqual(pd, start, opp1Block) || BlocksEqual(pd, start, opp2Block)) && flag == 1.f) {
			oppFound++;
		} else {
			CHECK(BlockIsZero(pd, start, PLAYER_ELEMS));
			CHECK_EQ(flag, 0.f);
		}
	}
	CHECK_EQ(oppFound, 2);
}

TEST(test_padded_team_canonicalization) {
	// Perfectly mirror-symmetric 1v1 state: the ORANGE player's canonical obs must equal
	// the BLUE player's. This is THE regression test for team canonicalization (probes
	// silently die when it breaks — see CLAUDE.md).
	RotMat reflected = RotMat(Vec(-1, 0, 0), Vec(0, -1, 0), Vec(0, 0, 1));
	auto blue = MakePlayer(1, Team::BLUE, Vec(1000, -2000, 17), Vec(500, 300, 0), RotMat::GetIdentity(), 64);
	auto orange = MakePlayer(2, Team::ORANGE, Vec(-1000, 2000, 17), Vec(-500, -300, 0), reflected, 64);
	auto gs = MakeState({ blue, orange }, Vec(0, 0, 93), Vec(0, 0, 0));

	AdvancedObs unpadded = {};
	AdvancedObsPadded padded(3);

	// The unpadded base builder must already hold this invariant
	FList ub = unpadded.BuildObs(gs.players[0], gs);
	FList uo = unpadded.BuildObs(gs.players[1], gs);
	CHECK_EQ(ub.size(), uo.size());
	for (int i = 0; i < (int)ub.size(); i++)
		CHECK_NEAR(ub[i], uo[i], 1e-5f);

	// Padded: seed the shared rand engine before each build so slot shuffles agree
	::Math::GetRandEngine().seed(1234);
	FList pb = padded.BuildObs(gs.players[0], gs);
	::Math::GetRandEngine().seed(1234);
	FList po = padded.BuildObs(gs.players[1], gs);
	CHECK_EQ(pb.size(), po.size());
	for (int i = 0; i < (int)pb.size(); i++)
		CHECK_NEAR(pb[i], po[i], 1e-5f);
}
