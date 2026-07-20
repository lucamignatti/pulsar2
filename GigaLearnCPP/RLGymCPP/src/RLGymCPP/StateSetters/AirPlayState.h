#pragma once
#include "StateSetter.h"
#include "../Math.h"
#include "../CommonValues.h"

namespace RLGC {
	// ADVANCED-AIR SEEDING (2026-07-20, user-directed). Flip resets, air dribbles
	// and wall play are ZERO-RATE: the policy never enters the precursor states, so
	// no reward can shape them (reward only shapes visited states) and no
	// amplification can find a fragment that is never sampled. This setter drops the
	// car straight INTO those setups so the rest of the stack (FlipResetReward,
	// AerialTouchReward, AirInterceptPotential, energy, B2G, Goal) can extend them
	// backward - the same reverse-curriculum logic as AirDrillState, one rung higher.
	//
	// Two sub-modes, chosen per reset:
	//   FLIP_RESET_READY  car airborne below a high ball, ROOF toward it (wheels
	//                     first), flip already spent (hasJumped) so a wheel touch
	//                     REGISTERS as a reset, drifting up into contact.
	//   AIR_CARRY         a ball moving toward the opponent net and rising, car just
	//                     behind/under it at matched velocity, nose on it - the
	//                     air-dribble carry, ready to keep nudging it goalward.
	//
	// TEAM-AWARE (mirrors AirDrillState): exactly one random car per team is the
	// air-player (both teams' air-players share the same ball, a symmetric contest);
	// the rest spawn grounded goal-side as support. 1v1 = just the air-player.
	class AirPlayState : public StateSetter {
	public:
		float carrySpeed;   // ball speed for the AIR_CARRY mode
		float resetFrac;    // fraction of resets that are FLIP_RESET_READY (rest AIR_CARRY)
		float supportMinDist, supportMaxDist;

		AirPlayState(float carrySpeed = 900, float resetFrac = 0.5f,
			float supportMinDist = 1500, float supportMaxDist = 3000)
			: carrySpeed(carrySpeed), resetFrac(resetFrac),
			  supportMinDist(supportMinDist), supportMaxDist(supportMaxDist) {}

		virtual void ResetArena(Arena* arena) {
			using RocketSim::Math::RandFloat;
			using RocketSim::Math::RandInt;
			using namespace CommonValues;

			arena->ResetToRandomKickoff();

			bool flipReset = RandFloat(0, 1) < resetFrac;

			// Attack direction is shared by the setup; pick a random attacker frame so
			// the ball heads toward ONE net (BLUE attacks +y, ORANGE attacks -y). For
			// the reset mode direction is cosmetic; for carry it sets the goalward line.
			float attackSign = RandInt(0, 2) ? 1.f : -1.f;

			BallState bs = {};
			if (flipReset) {
				// High, slow ball in the field interior.
				bs.pos = Math::RandVec(Vec(-2200, -3000, 1200), Vec(2200, 3000, 1700));
				bs.vel = Vec(RandFloat(-150, 150), RandFloat(-150, 150), RandFloat(-250, 50));
			} else {
				// Mid-height ball moving toward the attacked net and rising - a carry
				// in progress.
				bs.pos = Vec(RandFloat(-2000, 2000), attackSign * RandFloat(-500, 1500),
					RandFloat(350, 800));
				bs.vel = Vec(RandFloat(-200, 200), attackSign * carrySpeed, RandFloat(150, 450));
			}
			arena->ball->SetState(bs);

			constexpr float WALL_MARGIN = 300;
			constexpr float CAR_SEPARATION = 350;
			const float clampX = SIDE_WALL_X - WALL_MARGIN;
			const float clampY = BACK_WALL_Y - WALL_MARGIN;

			std::vector<Vec> placed = {};
			auto fnOverlaps = [&](const Vec& p) {
				for (Vec& o : placed)
					if ((p - o).Length() < CAR_SEPARATION) return true;
				return false;
			};

			// The air-player: placed into the mode's setup.
			auto fnSetAir = [&](Car* car) {
				CarState cs = {};
				Vec carPos = {};

				if (flipReset) {
					// Directly below the ball so "up" points at it; a small horizontal
					// jitter keeps LookAt well-defined.
					float drop = RandFloat(300, 520);
					carPos = bs.pos + Vec(RandFloat(-200, 200), RandFloat(-200, 200), -drop);
				} else {
					// Behind (goal-side of the ball's travel) and just below it.
					carPos = bs.pos - Vec(0, attackSign * RandFloat(250, 500), RandFloat(60, 200));
				}
				carPos.x = RS_CLAMP(carPos.x, -clampX, clampX);
				carPos.y = RS_CLAMP(carPos.y, -clampY, clampY);
				carPos.z = RS_MAX(300.f, carPos.z); // clearly airborne
				if (fnOverlaps(carPos))
					carPos.z += CAR_SEPARATION;
				placed.push_back(carPos);
				cs.pos = carPos;

				Vec toBall = (bs.pos - carPos);
				Vec toBallN = toBall.Normalized();
				if (flipReset) {
					// ROOF toward the ball (wheels make contact = a reset). Forward is
					// any in-plane direction; LookAt orthogonalizes against the up hint,
					// so pass a horizontal forward and up = toBall.
					Vec fwd = Vec(toBallN.y, -toBallN.x, 0);
					if (fwd.Length() < 0.1f) fwd = Vec(1, 0, 0);
					cs.rotMat = RotMat::LookAt(fwd.Normalized(), toBallN);
					// Flip already SPENT so a wheel touch registers as a reset.
					cs.hasJumped = true;
					cs.hasFlipped = true;
					cs.airTime = 0.6f;
					cs.airTimeSinceJump = 0.6f;
					// Drift up into the ball.
					cs.vel = toBallN * RandFloat(300, 650);
				} else {
					// Nose on the ball, roof up, velocity matched to the ball (carry).
					cs.rotMat = RotMat::LookAt(toBallN, Vec(0, 0, 1));
					cs.vel = bs.vel + toBallN * RandFloat(50, 200);
				}
				cs.isOnGround = false;
				cs.boost = RandFloat(50, 100);
				car->SetState(cs);
			};

			auto fnSetSupport = [&](Car* car, Team team) {
				Vec ownGoal = (team == Team::BLUE) ? BLUE_GOAL_BACK : ORANGE_GOAL_BACK;
				Vec ballShadow = Vec(bs.pos.x, bs.pos.y, 0);
				Vec toGoal = ownGoal - ballShadow;
				float goalTheta = atan2f(toGoal.y, toGoal.x);
				Vec carPos = {};
				bool ok = false;
				for (int a = 0; a < 16 && !ok; a++) {
					float theta = goalTheta + RandFloat(-M_PI / 2, M_PI / 2);
					float dist = RandFloat(supportMinDist, supportMaxDist);
					carPos = ballShadow + Vec(cosf(theta) * dist, sinf(theta) * dist, 0);
					carPos.z = 17;
					if (fabsf(carPos.x) > clampX || fabsf(carPos.y) > clampY) continue;
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
				cs.rotMat = Angle(atan2f(toBall.y, toBall.x), 0, 0).ToRotMat();
				cs.boost = RandFloat(50, 100);
				car->SetState(cs);
			};

			std::vector<Car*> teamCars[2];
			for (Car* car : arena->_cars)
				teamCars[(int)car->team].push_back(car);
			int airIdx[2] = {};
			for (int t = 0; t < 2; t++)
				if (teamCars[t].size() > 1)
					airIdx[t] = RandInt(0, (int)teamCars[t].size());

			for (int t = 0; t < 2; t++)
				if (!teamCars[t].empty())
					fnSetAir(teamCars[t][airIdx[t]]);
			for (int t = 0; t < 2; t++)
				for (int i = 0; i < (int)teamCars[t].size(); i++)
					if (i != airIdx[t])
						fnSetSupport(teamCars[t][i], (Team)t);
		}
	};
}
