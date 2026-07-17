#pragma once
#include "StateSetter.h"
#include "../Math.h"

#include <atomic>
#include <memory>

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
	//
	// TEAM-AWARE (4.0 team play, 2026-07-14): with >1 car per team, exactly ONE random car per
	// team is the CLIMBER (the airborne placement above — both teams' climbers contest the same
	// ball, keeping the drill a symmetric 50/50 like it always was in 1v1); the remaining cars
	// spawn GROUNDED goal-side of the ball's shadow at supportMinDist..supportMaxDist, facing
	// the ball, at rest. The old behavior launched all 4-6 cars at one ball — a midair scrum
	// teaching simultaneous full-team commitment. With one car per team the climber path is
	// exactly the old placement, so 1v1 arenas are unaffected.
	// ALTITUDE ANNEALING (AERIAL_GAP.md, 2026-07-15): shared difficulty knob written by
	// the learner's controller (learn-prep) and read by every AirDrillState at reset
	// (env threads) - the FrontierPool sharing pattern. difficulty 0 = the classic
	// airborne-climbing spawn; 1 = grounded takeoff (the measured missing skill:
	// jump-1s 98% but car z>500 only 9%, aerial touch 0/300 from the ground).
	struct AirDrillCurriculum {
		std::atomic<float> difficulty{ 0.f };
	};

	class AirDrillState : public StateSetter {
	public:
		float minBallZ, maxBallZ;   // overhead ball height
		float minDrop, maxDrop;     // how far BELOW the ball the car spawns (vertical gap)
		float minHoriz, maxHoriz;   // horizontal offset of the car from under the ball
		float minSpeed, maxSpeed;   // car climb speed toward the ball
		float minBoost;
		float supportMinDist, supportMaxDist; // Team play: grounded non-climber spawn ring (goal-side arc)
		float soloFrac = 0.5f; // chance the drill is UNCONTESTED (see solo-completion comment below)
		std::shared_ptr<AirDrillCurriculum> curriculum; // NULL = fixed classic behavior

		AirDrillState(
			float minBallZ = 900, float maxBallZ = 1500,
			float minDrop = 400, float maxDrop = 900,
			float minHoriz = 350, float maxHoriz = 750,
			float minSpeed = 700, float maxSpeed = 1400,
			float minBoost = 45,
			float supportMinDist = 1500, float supportMaxDist = 3000)
			: minBallZ(minBallZ), maxBallZ(maxBallZ), minDrop(minDrop), maxDrop(maxDrop),
			  minHoriz(minHoriz), maxHoriz(maxHoriz), minSpeed(minSpeed), maxSpeed(maxSpeed),
			  minBoost(minBoost), supportMinDist(supportMinDist), supportMaxDist(supportMaxDist) {}

		virtual void ResetArena(Arena* arena) {
			using RocketSim::Math::RandFloat;
			using RocketSim::Math::RandInt;

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

			auto fnOverlaps = [&](const Vec& carPos) {
				for (Vec& o : placed)
					if ((carPos - o).Length() < CAR_SEPARATION)
						return true;
				return false;
			};

			// The drill placement: airborne, below and horizontally offset from the ball, climbing at it
			auto fnSetClimber = [&](Car* car) {
				CarState cs = {};

				// Place the car airborne, in-bounds and clear of the other cars. The vertical
				// gap + horizontal offset keep the car->ball direction well off vertical, so
				// LookAt below is never degenerate.
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
					ok = !fnOverlaps(carPos);
				}
				if (!ok) {
					carPos.x = RS_CLAMP(carPos.x, -clampX, clampX);
					carPos.y = RS_CLAMP(carPos.y, -clampY, clampY);
				}
				placed.push_back(carPos);

				// ALTITUDE ANNEALING: difficulty D lerps the spawn from the classic
				// airborne-climbing placement (D=0) down to a grounded takeoff (D=1) -
				// the measured missing skill (AERIAL_GAP.md M4: 0/300 aerial touches
				// from the ground). The controller in the Learner only raises D while
				// the aerial-conversion metric stays healthy.
				float D = curriculum ? RS_CLAMP(curriculum->difficulty.load(), 0.f, 1.f) : 0.f;
				carPos.z = carPos.z + (17.f - carPos.z) * D;

				cs.pos = carPos;

				if (carPos.z > 60) {
					// Airborne: nose pointed at the ball, roof toward world-up (wheels
					// down). LookAt orthogonalizes the up hint, so a steep-but-not-
					// vertical climb angle stays well-defined.
					Vec toBall = bs.pos - carPos;
					cs.rotMat = RotMat::LookAt(toBall.Normalized(), Vec(0, 0, 1));
					// Climbing toward the ball; boost-fed so the aerial is completable.
					// The climb speed eases off with D so late-curriculum airborne spawns
					// don't hand over the whole approach for free.
					cs.vel = toBall.Normalized() * (RandFloat(minSpeed, maxSpeed) * (1.f - 0.3f * D));
					cs.isOnGround = false;
				} else {
					// Grounded takeoff: on wheels under/near the ball, facing its shadow,
					// rolling at it - the M4 probe geometry. The jump, boost-pitch and
					// climb are all the policy's to produce.
					cs.pos.z = 17;
					Vec toShadow = Vec(bs.pos.x, bs.pos.y, 0) - Vec(carPos.x, carPos.y, 0);
					float yaw = atan2f(toShadow.y, toShadow.x);
					cs.rotMat = Angle(yaw, 0, 0).ToRotMat();
					cs.vel = toShadow.Normalized() * RandFloat(600, 1000);
					cs.isOnGround = true;
				}
				cs.boost = RandFloat(minBoost, 100);

				car->SetState(cs);
			};

			// Grounded support: goal-side of the ball's shadow, facing the ball, at rest —
			// positioned to play the follow-up (clear, rebound, cover) rather than joining the aerial
			auto fnSetSupport = [&](Car* car, Team team) {
				Vec ownGoal = (team == Team::BLUE) ? CommonValues::BLUE_GOAL_BACK : CommonValues::ORANGE_GOAL_BACK;
				Vec ballShadow = Vec(bs.pos.x, bs.pos.y, 0);
				Vec toGoal = ownGoal - ballShadow;
				float goalTheta = atan2f(toGoal.y, toGoal.x);

				Vec carPos = {};
				bool ok = false;
				for (int attempt = 0; attempt < 16 && !ok; attempt++) {
					float theta = goalTheta + RandFloat(-M_PI / 2, M_PI / 2);
					float dist = RandFloat(supportMinDist, supportMaxDist);
					carPos = ballShadow + Vec(cosf(theta) * dist, sinf(theta) * dist, 0);
					carPos.z = 17;

					if (fabsf(carPos.x) > clampX || fabsf(carPos.y) > clampY)
						continue;
					ok = !fnOverlaps(carPos);
				}
				if (!ok) {
					carPos.x = RS_CLAMP(carPos.x, -clampX, clampX);
					carPos.y = RS_CLAMP(carPos.y, -clampY, clampY);
					carPos.z = 17;
				}
				placed.push_back(carPos);

				CarState cs = {};
				cs.pos = carPos;

				Vec toBall = bs.pos - carPos;
				float yaw = atan2f(toBall.y, toBall.x);
				cs.rotMat = Angle(yaw, 0, 0).ToRotMat();

				cs.boost = RandFloat(minBoost, 100);

				car->SetState(cs);
			};

			// Pick one climber per team (the only car, in 1v1)
			std::vector<Car*> teamCars[2];
			for (Car* car : arena->_cars)
				teamCars[(int)car->team].push_back(car);
			int climberIdx[2] = {};
			for (int t = 0; t < 2; t++)
				if (teamCars[t].size() > 1)
					climberIdx[t] = RandInt(0, (int)teamCars[t].size());

			// SOLO-COMPLETION VARIANT (2026-07-17, break-even seeding): with soloFrac
			// chance, only ONE team's climber contests - the other's spawns as a far
			// grounded support. Measured mechanism (aerial_gap census 575M->4B on
			// 5.0v3): contested drills reach a MUTUAL-BAIL equilibrium (completion
			// 33%->12% while ground skill grows; zero-sum self-play never punishes
			// symmetric bailing - the collective-decline structure inside the drill).
			// A free completion always pays (TouchAccel + AerialTouch, no race lost),
			// putting attempts unconditionally above break-even until the skill exists
			// to win contested versions.
			bool solo = RandFloat(0, 1) < soloFrac;
			int soloTeam = RandInt(0, 2);

			// Climbers first so the aerial contest is never crowded out by support placement
			for (int t = 0; t < 2; t++)
				if (!teamCars[t].empty()) {
					if (solo && t != soloTeam)
						fnSetSupport(teamCars[t][climberIdx[t]], (Team)t);
					else
						fnSetClimber(teamCars[t][climberIdx[t]]);
				}

			for (int t = 0; t < 2; t++)
				for (int i = 0; i < (int)teamCars[t].size(); i++)
					if (i != climberIdx[t])
						fnSetSupport(teamCars[t][i], (Team)t);
		}
	};
}
