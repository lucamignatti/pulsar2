#include "TestFramework.h"

#include <RLGymCPP/StateSetters/BallNearCarState.h>
#include <RLGymCPP/StateSetters/AirDrillState.h>

#include <filesystem>

using namespace RLGC;

// Team-aware state setters (4.0): with >1 car per team, one contester/climber per team,
// remaining cars in goal-side support positions. 1v1 keeps the original behavior.
//
// These tests need real arenas -> RocketSim::Init with collision meshes. Run the test
// binary from the repo root (or build/), where collision_meshes/ lives.

static void EnsureRocketSim() {
	static bool inited = false;
	if (inited)
		return;
	for (const char* path : { "collision_meshes", "../collision_meshes", "build/collision_meshes" }) {
		if (std::filesystem::exists(std::filesystem::path(path) / "soccar")) {
			RocketSim::Init(path);
			inited = true;
			return;
		}
	}
	throw TestFailure{ "collision_meshes/ not found - run the test binary from the repo root" };
}

static Arena* MakeArena(int playersPerTeam) {
	EnsureRocketSim();
	Arena* arena = Arena::Create(GameMode::SOCCAR);
	for (int i = 0; i < playersPerTeam; i++) {
		arena->AddCar(Team::BLUE);
		arena->AddCar(Team::ORANGE);
	}
	return arena;
}

static float HorizDist(Vec a, Vec b) {
	float dx = a.x - b.x, dy = a.y - b.y;
	return sqrtf(dx * dx + dy * dy);
}

TEST(BallNearCar_1v1_both_cars_in_contest_ring) {
	Arena* arena = MakeArena(1);
	BallNearCarState setter(600, 900);
	for (int reset = 0; reset < 20; reset++) {
		setter.ResetArena(arena);
		Vec ballPos = arena->ball->GetState().pos;
		for (Car* car : arena->_cars) {
			CarState cs = car->GetState();
			float d = HorizDist(cs.pos, ballPos);
			CHECK(d >= 600 - 1 && d <= 900 + 1);
			CHECK_NEAR(cs.pos.z, 17.f, 1.f);
			// Facing the ball (ground yaw)
			Vec toBall = (ballPos - cs.pos);
			toBall.z = 0;
			CHECK(cs.rotMat.forward.Dot(toBall.Normalized()) > 0.95f);
		}
	}
	delete arena;
}

TEST(BallNearCar_3v3_one_contester_goal_side_supports) {
	Arena* arena = MakeArena(3);
	BallNearCarState setter(600, 900);
	for (int reset = 0; reset < 20; reset++) {
		setter.ResetArena(arena);
		Vec ballPos = arena->ball->GetState().pos;

		int contesters[2] = { 0, 0 };
		for (Car* car : arena->_cars) {
			CarState cs = car->GetState();
			float d = HorizDist(cs.pos, ballPos);
			CHECK_NEAR(cs.pos.z, 17.f, 1.f);

			if (d <= 900 + 1) {
				// Contester: in the contest ring
				CHECK(d >= 600 - 1);
				contesters[(int)car->team]++;
			} else {
				// Support: in the support band, goal-side of the ball
				CHECK(d >= 1800 - 1 && d <= 3200 + 1);
				Vec ownGoal = (car->team == Team::BLUE)
					? CommonValues::BLUE_GOAL_BACK : CommonValues::ORANGE_GOAL_BACK;
				Vec toGoal = ownGoal - ballPos;
				toGoal.z = 0;
				Vec off = cs.pos - ballPos;
				off.z = 0;
				CHECK(off.Normalized().Dot(toGoal.Normalized()) >= -1e-3f);
			}

			// Facing the ball
			Vec toBall = ballPos - cs.pos;
			toBall.z = 0;
			CHECK(cs.rotMat.forward.Dot(toBall.Normalized()) > 0.95f);
		}
		CHECK_EQ(contesters[0], 1);
		CHECK_EQ(contesters[1], 1);

		// No two cars intersect at spawn
		std::vector<Vec> positions;
		for (Car* car : arena->_cars)
			positions.push_back(car->GetState().pos);
		for (size_t i = 0; i < positions.size(); i++)
			for (size_t j = i + 1; j < positions.size(); j++)
				CHECK((positions[i] - positions[j]).Length() > 150);
	}
	delete arena;
}

TEST(AirDrill_1v1_both_cars_climbing) {
	Arena* arena = MakeArena(1);
	AirDrillState setter;
	// Pin the CONTESTED path. The solo-completion variant (2026-07-17) makes one team a
	// grounded support with probability soloFrac (live default 0.5), which made this test
	// fail on ~half the resets from 07-17 until 07-25. soloFrac is covered separately below.
	setter.soloFrac = 0.f;
	for (int reset = 0; reset < 20; reset++) {
		setter.ResetArena(arena);
		Vec ballPos = arena->ball->GetState().pos;
		for (Car* car : arena->_cars) {
			CarState cs = car->GetState();
			CHECK(cs.pos.z >= 249); // airborne
			CHECK(cs.pos.z < ballPos.z); // below the ball
			float speed = cs.vel.Length();
			CHECK(speed >= 700 - 1 && speed <= 1400 + 1);
			// Climbing toward the ball
			Vec toBall = (ballPos - cs.pos).Normalized();
			CHECK(cs.vel.Normalized().Dot(toBall) > 0.99f);
			CHECK(cs.boost >= 45 - 1e-3f);
		}
	}
	delete arena;
}

TEST(AirDrill_2v2_one_climber_grounded_supports) {
	Arena* arena = MakeArena(2);
	AirDrillState setter;
	setter.soloFrac = 0.f; // contested path: exactly one climber PER TEAM (see note above)
	for (int reset = 0; reset < 20; reset++) {
		setter.ResetArena(arena);
		Vec ballPos = arena->ball->GetState().pos;
		Vec ballShadow = Vec(ballPos.x, ballPos.y, 0);

		int climbers[2] = { 0, 0 };
		for (Car* car : arena->_cars) {
			CarState cs = car->GetState();
			if (cs.pos.z >= 249) {
				// Climber: airborne, moving at the ball
				climbers[(int)car->team]++;
				Vec toBall = (ballPos - cs.pos).Normalized();
				CHECK(cs.vel.Normalized().Dot(toBall) > 0.99f);
			} else {
				// Support: grounded, at rest, in the support band, goal-side
				CHECK_NEAR(cs.pos.z, 17.f, 1.f);
				CHECK_NEAR(cs.vel.Length(), 0.f, 1e-3f);
				float d = HorizDist(cs.pos, ballShadow);
				CHECK(d >= 1500 - 1 && d <= 3000 + 1);
				Vec ownGoal = (car->team == Team::BLUE)
					? CommonValues::BLUE_GOAL_BACK : CommonValues::ORANGE_GOAL_BACK;
				Vec toGoal = ownGoal - ballShadow;
				toGoal.z = 0;
				Vec off = cs.pos - ballShadow;
				off.z = 0;
				CHECK(off.Normalized().Dot(toGoal.Normalized()) >= -1e-3f);
				// Facing the ball
				Vec toBall = ballPos - cs.pos;
				toBall.z = 0;
				CHECK(cs.rotMat.forward.Dot(toBall.Normalized()) > 0.9f);
			}
			CHECK(cs.boost >= 45 - 1e-3f);
		}
		CHECK_EQ(climbers[0], 1);
		CHECK_EQ(climbers[1], 1);
	}
	delete arena;
}

// SOLO-COMPLETION VARIANT (AirDrillState.h:196-212, 2026-07-17). Live soloFrac is 0.5, so
// ~half of all air drills spawn UNCONTESTED - roughly 8% of all resets. It shipped with no
// test and silently broke the two contested tests above; pin it here so the next change to
// the branch is caught. 1v1 is the live case (PHASE A).
TEST(AirDrill_1v1_solo_gives_exactly_one_climber) {
	Arena* arena = MakeArena(1);
	AirDrillState setter;
	setter.soloFrac = 1.f; // always uncontested
	for (int reset = 0; reset < 20; reset++) {
		setter.ResetArena(arena);
		Vec ballPos = arena->ball->GetState().pos;

		int climbers = 0, supports = 0;
		for (Car* car : arena->_cars) {
			CarState cs = car->GetState();
			if (cs.pos.z >= 249) {
				climbers++;
				// Same climber contract as the contested path
				CHECK(cs.pos.z < ballPos.z);
				Vec toBall = (ballPos - cs.pos).Normalized();
				CHECK(cs.vel.Normalized().Dot(toBall) > 0.99f);
			} else {
				supports++;
				CHECK_NEAR(cs.pos.z, 17.f, 1.f);
				CHECK_NEAR(cs.vel.Length(), 0.f, 1e-3f);
			}
			CHECK(cs.boost >= 45 - 1e-3f);
		}
		// Exactly one team contests; the other's designated climber becomes a support
		CHECK_EQ(climbers, 1);
		CHECK_EQ(supports, 1);
	}
	delete arena;
}
