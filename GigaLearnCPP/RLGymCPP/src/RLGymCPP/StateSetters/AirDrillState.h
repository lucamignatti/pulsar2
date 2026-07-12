#pragma once
#include "StateSetter.h"
#include "../Math.h"

namespace RLGC {
	// Reverse-curriculum aerial drill.
	//
	// The old aerial drill (BallNearCarState with a height/drift) spawned the ball overhead but BOTH
	// cars ON THE GROUND with the ball DESCENDING, so the optimal play was to wait for the ball to
	// fall and hit it on the ground — no aerial was ever required, and 17B+ steps produced none.
	//
	// This setter instead spawns the car ALREADY AIRBORNE and CLIMBING toward an overhead ball, nose
	// pointed at it, wheels roughly down, boost-fed: the car starts committed to the aerial and only
	// has to COMPLETE the touch (an achievable sub-skill). Reverse curriculum — start at the goal
	// state and let PPO extend it backward into initiating the takeoff itself, now that the reward
	// (AerialTouch / AirInterceptPotential) actually pays for the airborne touch.
	class AirDrillState : public StateSetter {
	public:
		float minBallZ, maxBallZ;   // overhead ball height
		float minDrop, maxDrop;     // how far BELOW the ball the car spawns (vertical gap)
		float minHoriz, maxHoriz;   // horizontal offset of the car from under the ball
		float minSpeed, maxSpeed;   // car climb speed toward the ball
		float minBoost;

		AirDrillState(
			float minBallZ = 900, float maxBallZ = 1500,
			float minDrop = 400, float maxDrop = 900,
			float minHoriz = 350, float maxHoriz = 750,
			float minSpeed = 700, float maxSpeed = 1400,
			float minBoost = 45)
			: minBallZ(minBallZ), maxBallZ(maxBallZ), minDrop(minDrop), maxDrop(maxDrop),
			  minHoriz(minHoriz), maxHoriz(maxHoriz), minSpeed(minSpeed), maxSpeed(maxSpeed),
			  minBoost(minBoost) {}

		virtual void ResetArena(Arena* arena) {
			using RocketSim::Math::RandFloat;

			arena->ResetToRandomKickoff(); // reset boost pads etc.

			// Ball hangs overhead in the field interior with a gentle drift (variety, not a fast fall).
			BallState bs = {};
			bs.pos = Math::RandVec(Vec(-2600, -3400, minBallZ), Vec(2600, 3400, maxBallZ));
			bs.vel = Vec(RandFloat(-200, 200), RandFloat(-200, 200), RandFloat(-100, 100));
			arena->ball->SetState(bs);

			constexpr float WALL_MARGIN = 300;
			constexpr float CAR_SEPARATION = 350;
			const float clampX = CommonValues::SIDE_WALL_X - WALL_MARGIN;
			const float clampY = CommonValues::BACK_WALL_Y - WALL_MARGIN;

			std::vector<Vec> placed = {};
			for (Car* car : arena->_cars) {
				CarState cs = {};

				// Place the car airborne, below and horizontally offset from the ball, in-bounds and
				// clear of the other car. The vertical gap + horizontal offset keep the car->ball
				// direction well off vertical, so LookAt below is never degenerate.
				Vec carPos = {};
				bool ok = false;
				for (int attempt = 0; attempt < 16 && !ok; attempt++) {
					float theta = RandFloat(0, M_PI * 2);
					float horiz = RandFloat(minHoriz, maxHoriz);
					carPos = bs.pos + Vec(cosf(theta) * horiz, sinf(theta) * horiz, 0);
					carPos.z = bs.pos.z - RandFloat(minDrop, maxDrop);
					if (carPos.z < 250) carPos.z = 250; // stay clearly airborne

					if (fabsf(carPos.x) > clampX || fabsf(carPos.y) > clampY)
						continue;
					bool overlaps = false;
					for (Vec& o : placed)
						if ((carPos - o).Length() < CAR_SEPARATION) { overlaps = true; break; }
					ok = !overlaps;
				}
				if (!ok) {
					carPos.x = RS_CLAMP(carPos.x, -clampX, clampX);
					carPos.y = RS_CLAMP(carPos.y, -clampY, clampY);
				}
				placed.push_back(carPos);

				cs.pos = carPos;

				// Nose pointed at the ball, roof toward world-up (wheels down). LookAt orthogonalizes
				// the up hint, so a steep-but-not-vertical climb angle stays well-defined.
				Vec toBall = bs.pos - carPos;
				cs.rotMat = RotMat::LookAt(toBall.Normalized(), Vec(0, 0, 1));

				// Climbing toward the ball; boost-fed so the aerial is completable.
				cs.vel = toBall.Normalized() * RandFloat(minSpeed, maxSpeed);
				cs.boost = RandFloat(minBoost, 100);

				car->SetState(cs);
			}
		}
	};
}
