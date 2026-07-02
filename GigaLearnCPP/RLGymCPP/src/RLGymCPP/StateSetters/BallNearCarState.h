#pragma once
#include "StateSetter.h"
#include "../Math.h"

namespace RLGC {
	// Places each car BALL-RELATIVE: car = ballPos + horizontalDir * dist, dist in [minDist, maxDist],
	// on-ground, at rest, facing the ball. The ball is static in the field interior.
	// Unlike RandomState, the car's position is sampled AROUND the ball, so a cold policy is always
	// within driving reach of a touch. minDist should stay well above BALL_RADIUS + car length so the
	// car can never spawn already touching.
	class BallNearCarState : public StateSetter {
	public:
		float minDist, maxDist;

		BallNearCarState(float minDist = 600, float maxDist = 900)
			: minDist(minDist), maxDist(maxDist) {
		}

		virtual void ResetArena(Arena* arena) {
			using RocketSim::Math::RandFloat;

			// Reset boost pads and everything
			arena->ResetToRandomKickoff();

			BallState bs = {};
			// Keep the ball far enough inside the field that the car's offset ring fits without
			// hitting a wall (interior box + maxDist stays inside the wall clamp below)
			bs.pos = Math::RandVec(
				Vec(-2800, -3800, CommonValues::BALL_RADIUS),
				Vec(2800, 3800, CommonValues::BALL_RADIUS)
			);
			arena->ball->SetState(bs);

			constexpr float WALL_MARGIN = 300;
			constexpr float CAR_SEPARATION = 300; // ~2.5 car lengths between spawn positions
			const float clampX = CommonValues::SIDE_WALL_X - WALL_MARGIN;
			const float clampY = CommonValues::BACK_WALL_Y - WALL_MARGIN;

			std::vector<Vec> placedCars = {};
			for (Car* car : arena->_cars) {
				CarState cs = {};

				// Rejection-sample an angle that lands in-bounds and clear of other cars;
				// clamping instead would silently break the [minDist, maxDist] ring (and
				// two cars spawning intersecting get a violent solver impulse on tick 1).
				// With the default radii the ball's interior box makes bounds always pass.
				Vec carPos = {};
				bool placed = false;
				for (int attempt = 0; attempt < 16 && !placed; attempt++) {
					// Uniform angle avoids the diagonal bias of sampling a box then normalizing
					float theta = RandFloat(0, M_PI * 2);
					Vec dir = Vec(cosf(theta), sinf(theta), 0);
					float dist = RandFloat(minDist, maxDist);
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
					if (offLen < minDist && offLen > 1)
						carPos = bs.pos + (off / offLen) * minDist;
					carPos.x = RS_CLAMP(carPos.x, -clampX, clampX);
					carPos.y = RS_CLAMP(carPos.y, -clampY, clampY);
					carPos.z = 17;
				}

				placedCars.push_back(carPos);
				cs.pos = carPos;

				// Face the ball (ground yaw only), at rest
				Vec toBall = bs.pos - carPos;
				float yaw = atan2f(toBall.y, toBall.x);
				cs.rotMat = Angle(yaw, 0, 0).ToRotMat();

				cs.boost = RandFloat(0, 100);

				car->SetState(cs);
			}
		}
	};
}
