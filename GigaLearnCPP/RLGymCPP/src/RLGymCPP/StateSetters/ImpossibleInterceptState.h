#pragma once
#include "StateSetter.h"
#include "../Math.h"
#include "../CommonValues.h"

namespace RLGC {
	// OPTIMISTIC-CRITIC LADDER: the standing-falsification drill family (LADDER.md
	// section 3.2). Every spawn is CERTIFIED UNACHIEVABLE on the analytic axis: the
	// ball is launched on a fully-airborne sub-second ballistic line into a goal
	// (no bounce, so its flight time is exact projectile math; drag is a few percent
	// and covered by the margin), and every car is placed so the average speed it
	// would need to touch the ball before it scores exceeds SPEED_MARGIN x the hard
	// 2300uu/s physics cap. The check runs per jittered spawn, per car, against the
	// closest point of the whole flight segment; a failed draw is re-rolled.
	//
	// Purpose: a permanent regression against self-serving optimism. The ladder's
	// acceptance criteria at these spawns (checked continuously by the Ladder/Imp*
	// panels): zero ball touches EVER (a single touch voids the certificate, not the
	// run), V_exp deflating toward V_real with exposure, gap_PK below the feasible
	// families' peaks (the concede attractor is how the map learns doom: d_concede
	// small, d_goal at clamp). Rows from these arenas are masked out of the drive
	// injection; the free-play-occurrence axis holds by construction (all cars at
	// rest in the far corners while an unreachable rocket flies at their net is not
	// a state 1v1 play produces).
	//
	// Team-vs-team accounting stays exactly zero-sum: the target goal alternates
	// randomly, so in expectation both teams concede these free goals equally, and
	// the critic learns the states are doomed (which is the point) while advantages
	// stay ~0 (the -150 is fully predicted).
	class ImpossibleInterceptState : public StateSetter {
	public:
		static constexpr float SPEED_MARGIN = 1.6f;   // x CAR_MAX_SPEED, the certificate
		static constexpr float CLEARANCE_PAD = 500;   // car length / jump reach buffer, uu

		virtual void ResetArena(Arena* arena) {
			using RocketSim::Math::RandFloat;
			using RocketSim::Math::RandInt;
			using namespace CommonValues;

			arena->ResetToRandomKickoff(); // boost pads etc.; all cars re-placed below

			BallState bs = {};
			std::vector<Vec> carSpots = {};
			bool committed = false;

			for (int attempt = 0; attempt < 32 && !committed; attempt++) {
				// Target: a point inside the goal mouth (comfortably within the posts
				// and under the bar) on a random side.
				float side = RandInt(0, 2) ? 1.f : -1.f;
				Vec target(RandFloat(-650, 650), side * BACK_WALL_Y, RandFloat(150, 480));

				// Ball spawn: same half, 2000-3500uu out, airborne.
				Vec bpos(RandFloat(-2300, 2300), side * RandFloat(1600, 3100), RandFloat(300, 600));
				Vec to = target - bpos;
				float dist = to.Length();
				float T = RandFloat(0.75f, 1.05f); // flight time to the goal plane
				if (dist / T > 5600)               // stay clearly under BALL_MAX_SPEED
					continue;

				// Exact projectile velocity to hit the target at time T (gravity
				// compensated); flight stays airborne the whole way (spawn z >= 300,
				// target z >= 150, downward parabola: the minimum is at an endpoint),
				// so the certificate's time bound is pure kinematics.
				Vec vel = to * (1.f / T);
				vel.z += 0.5f * (-GRAVITY_Z) * T;
				if (vel.Length() > 5800)
					continue;

				// Cars: at rest along the back wall of the OPPOSITE half, spread out.
				// Certificate per car: distance to the closest point of the flight
				// segment must exceed what SPEED_MARGIN x max speed covers in T.
				float required = SPEED_MARGIN * CAR_MAX_SPEED * T + CLEARANCE_PAD;
				carSpots.clear();
				int numCars = (int)arena->_cars.size();
				bool ok = true;
				for (int i = 0; i < numCars && ok; i++) {
					Vec spot(
						(i % 2 ? 1.f : -1.f) * (SIDE_WALL_X - 500 - 400 * (i / 2)) + RandFloat(-120, 120),
						-side * (BACK_WALL_Y - 450),
						17);
					// point-to-segment distance vs the ball's flight chord (the true
					// curved path sags BELOW the chord only in z; xy is exact, and the
					// z sag only moves the path further from a grounded car)
					Vec ab = target - bpos;
					float tSeg = RS_CLAMP((spot - bpos).Dot(ab) / ab.Dot(ab), 0.f, 1.f);
					float dMin = (spot - (bpos + ab * tSeg)).Length();
					if (dMin < required)
						ok = false;
					else
						carSpots.push_back(spot);
				}
				if (!ok)
					continue;

				bs.pos = bpos;
				bs.vel = vel;
				committed = true;
			}

			// A 32x rejection failure is geometrically impossible with these ranges
			// (corner-to-path distance is ~7000+ vs required ~4400), but never leave
			// the arena half-set: fall back to a fixed certified geometry.
			if (!committed) {
				bs.pos = Vec(0, 2400, 400);
				Vec target(0, BACK_WALL_Y, 300);
				float T = 0.9f;
				bs.vel = (target - bs.pos) * (1.f / T);
				bs.vel.z += 0.5f * (-GRAVITY_Z) * T;
				carSpots.clear();
				for (int i = 0; i < (int)arena->_cars.size(); i++)
					carSpots.push_back(Vec((i % 2 ? 1.f : -1.f) * (SIDE_WALL_X - 500 - 400 * (i / 2)),
						-(BACK_WALL_Y - 450), 17));
			}

			arena->ball->SetState(bs);

			int i = 0;
			for (Car* car : arena->_cars) {
				CarState cs = {};
				cs.pos = carSpots[i++];
				cs.rotMat = Angle(RandFloat(0, (float)(M_PI * 2)), 0, 0).ToRotMat();
				cs.vel = Vec(0, 0, 0);
				cs.boost = RandFloat(0, 100); // boost is irrelevant: 2300 is a hard cap
				cs.isOnGround = true;
				car->SetState(cs);
			}
		}
	};
}
