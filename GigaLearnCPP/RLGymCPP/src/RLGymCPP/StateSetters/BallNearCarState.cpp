#include "BallNearCarState.h"
#include "../Math.h"

using RocketSim::Math::RandFloat;

void RLGC::BallNearCarState::ResetArena(Arena* arena) {

	// Reset boost pads + everything to a kickoff baseline, then override the ball and cars.
	arena->ResetToRandomKickoff();

	// Ball: a random field position (kept well inside the field so the car offset stays in-bounds), low and
	// STATIONARY -- a still ball on/near the ground gives the cleanest impulse credit on the first strike.
	constexpr float
		BALL_BOX_X = 2500,
		BALL_BOX_Y = 3000,
		BALL_Z_MAX = 250; // low: mostly groundable, occasional shallow bounce
	Vec ballPos = Vec(
		RandFloat(-BALL_BOX_X, BALL_BOX_X),
		RandFloat(-BALL_BOX_Y, BALL_BOX_Y),
		RandFloat(CommonValues::BALL_RADIUS, BALL_Z_MAX));
	{
		BallState bs = {};
		bs.pos = ballPos;
		// vel / angVel default to 0 -> stationary ball.
		arena->ball->SetState(bs);
	}

	// Cars: each placed on the ground within [radiusMin, radiusMax] HORIZONTAL of the ball, at rest, facing a
	// random heading (it must turn toward the ball to strike it). Independent angle per car so the two 1v1
	// cars are spread around the ball rather than stacked.
	float rMin = radiusMin < 0 ? 0 : radiusMin;
	float rMax = radiusMax < rMin ? rMin : radiusMax;
	for (Car* car : arena->_cars) {
		float ang = RandFloat(-M_PI, M_PI);
		float dist = RandFloat(rMin, rMax);

		CarState cs = {};
		cs.pos = Vec(ballPos.x + cosf(ang) * dist, ballPos.y + sinf(ang) * dist, 17.f);

		Angle angle = Angle(RandFloat(-M_PI, M_PI), 0, 0); // random yaw, level pitch/roll
		cs.rotMat = angle.ToRotMat();

		cs.boost = RandFloat(0, 100);
		// vel / angVel default to 0 -> car at rest; it must throttle to reach the ball.
		car->SetState(cs);
	}
}
