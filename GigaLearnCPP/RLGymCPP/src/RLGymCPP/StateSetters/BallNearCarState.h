#pragma once
#include "StateSetter.h"
#include "../Math.h"

namespace RLGC {
	// Places cars BALL-RELATIVE: the CONTESTER = ballPos + horizontalDir * dist, dist in
	// [minDist, maxDist], on-ground, at rest, facing the ball. By default the ball is static
	// on the ground in the field interior; the optional height/drift/boost params turn the
	// same placement into an aerial drill (ball hanging overhead, drifting down, cars
	// guaranteed fuel to go get it). Unlike RandomState, positions are sampled AROUND the
	// ball, so a cold policy is always within driving reach of a touch. minDist should stay
	// well above BALL_RADIUS + car length so a car can never spawn already touching.
	//
	// TEAM-AWARE (4.0 team play, 2026-07-14): with >1 car per team, exactly ONE random car
	// per team spawns in the contest ring; the remaining cars spawn as SUPPORT — goal-side
	// of the ball (within +-90 degrees of the ball->own-goal direction) at supportMinDist..
	// supportMaxDist, facing the ball. The old behavior (every car in the ring) spawned a
	// 4-6 car scrum in 2v2/3v3, teaching everyone to converge on the ball; this gives the
	// reset the 1st-man/2nd-man structure real team play has. With one car per team the
	// contester path is exactly the old placement, so 1v1 arenas are unaffected.
	class BallNearCarState : public StateSetter {
	public:
		float minDist, maxDist;
		float minBallZ, maxBallZ; // Ball spawn height range (default: static on the ground)
		float maxBallDrift;       // Ball speed: horizontal in [0, drift], vertical in [-drift, 0] (descending)
		float minBoost;           // Guarantee fuel (aerial drills must never be boost-starved)
		float supportMinDist, supportMaxDist; // Team play: non-contester spawn ring (goal-side arc)

		BallNearCarState(float minDist = 600, float maxDist = 900,
			float minBallZ = CommonValues::BALL_RADIUS, float maxBallZ = CommonValues::BALL_RADIUS,
			float maxBallDrift = 0, float minBoost = 0,
			float supportMinDist = 1800, float supportMaxDist = 3200)
			: minDist(minDist), maxDist(maxDist), minBallZ(minBallZ), maxBallZ(maxBallZ),
			maxBallDrift(maxBallDrift), minBoost(minBoost),
			supportMinDist(supportMinDist), supportMaxDist(supportMaxDist) {
		}

		virtual void ResetArena(Arena* arena) {
			using RocketSim::Math::RandFloat;
			using RocketSim::Math::RandInt;

			// Reset boost pads and everything
			arena->ResetToRandomKickoff();

			BallState bs = {};
			// Keep the ball far enough inside the field that the car's offset ring fits without
			// hitting a wall (interior box + maxDist stays inside the wall clamp below)
			bs.pos = Math::RandVec(
				Vec(-2800, -3800, minBallZ),
				Vec(2800, 3800, maxBallZ)
			);
			if (maxBallDrift > 0) {
				float driftTheta = RandFloat(0, M_PI * 2);
				bs.vel = Vec(cosf(driftTheta), sinf(driftTheta), 0) * RandFloat(0, maxBallDrift);
				bs.vel.z = RandFloat(-maxBallDrift, 0);
			}
			arena->ball->SetState(bs);

			constexpr float WALL_MARGIN = 300;
			constexpr float CAR_SEPARATION = 300; // ~2.5 car lengths between spawn positions
			const float clampX = CommonValues::SIDE_WALL_X - WALL_MARGIN;
			const float clampY = CommonValues::BACK_WALL_Y - WALL_MARGIN;

			std::vector<Vec> placedCars = {};

			// Rejection-sample a ground position on a ring around the ball, restricted to
			// [thetaCenter - thetaHalfRange, thetaCenter + thetaHalfRange], in-bounds and clear
			// of other cars; clamping instead would silently break the distance ring (and two
			// cars spawning intersecting get a violent solver impulse on tick 1). With the
			// default radii the ball's interior box makes full-circle bounds always pass.
			auto fnPlaceOnRing = [&](float ringMin, float ringMax, float thetaCenter, float thetaHalfRange) {
				Vec carPos = {};
				bool placed = false;
				for (int attempt = 0; attempt < 16 && !placed; attempt++) {
					// Uniform angle avoids the diagonal bias of sampling a box then normalizing
					float theta = thetaCenter + RandFloat(-thetaHalfRange, thetaHalfRange);
					Vec dir = Vec(cosf(theta), sinf(theta), 0);
					float dist = RandFloat(ringMin, ringMax);
					carPos = bs.pos + dir * dist;
					carPos.z = 17;

					if (fabsf(carPos.x) > clampX || fabsf(carPos.y) > clampY)
						continue;

					bool overlaps = false;
					for (Vec& other : placedCars) {
						if ((carPos - other).Length() < CAR_SEPARATION) {
							overlaps = true;
							break;
						}
					}
					placed = !overlaps;
				}

				if (!placed) {
					// Extreme configs only (huge radii near a wall): best effort — clamp into
					// bounds, then push back out to the distance floor toward the interior
					carPos.x = RS_CLAMP(carPos.x, -clampX, clampX);
					carPos.y = RS_CLAMP(carPos.y, -clampY, clampY);
					Vec off = carPos - bs.pos;
					off.z = 0;
					float offLen = off.Length();
					if (offLen < ringMin && offLen > 1)
						carPos = bs.pos + (off / offLen) * ringMin;
					carPos.x = RS_CLAMP(carPos.x, -clampX, clampX);
					carPos.y = RS_CLAMP(carPos.y, -clampY, clampY);
					carPos.z = 17;
				}

				placedCars.push_back(carPos);
				return carPos;
			};

			auto fnSetCar = [&](Car* car, Vec carPos) {
				CarState cs = {};
				cs.pos = carPos;

				// Face the ball (ground yaw only), at rest
				Vec toBall = bs.pos - carPos;
				float yaw = atan2f(toBall.y, toBall.x);
				cs.rotMat = Angle(yaw, 0, 0).ToRotMat();

				cs.boost = RandFloat(minBoost, 100);

				car->SetState(cs);
			};

			// Pick one contester per team (the only car, in 1v1)
			std::vector<Car*> teamCars[2];
			for (Car* car : arena->_cars)
				teamCars[(int)car->team].push_back(car);
			int contesterIdx[2] = {};
			for (int t = 0; t < 2; t++)
				if (teamCars[t].size() > 1)
					contesterIdx[t] = RandInt(0, (int)teamCars[t].size());

			// Contesters first so their ring is never crowded out by supports
			for (int t = 0; t < 2; t++)
				if (!teamCars[t].empty())
					fnSetCar(teamCars[t][contesterIdx[t]], fnPlaceOnRing(minDist, maxDist, RandFloat(0, M_PI * 2), M_PI));

			// Supports: goal-side of the ball, within +-90 degrees of the ball->own-goal direction
			for (int t = 0; t < 2; t++) {
				if (teamCars[t].size() <= 1)
					continue;
				Vec ownGoal = ((Team)t == Team::BLUE) ? CommonValues::BLUE_GOAL_BACK : CommonValues::ORANGE_GOAL_BACK;
				Vec toGoal = ownGoal - bs.pos;
				float goalTheta = atan2f(toGoal.y, toGoal.x);
				for (int i = 0; i < (int)teamCars[t].size(); i++) {
					if (i == contesterIdx[t])
						continue;
					fnSetCar(teamCars[t][i], fnPlaceOnRing(supportMinDist, supportMaxDist, goalTheta, M_PI / 2));
				}
			}
		}
	};
}
