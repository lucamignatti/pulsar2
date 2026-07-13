#include "RandomState.h"
#include "../Math.h"

using RocketSim::Math::RandFloat;
using RLGC::Math::RandVec;

Vec RandNormVec() {
	return RandVec(Vec(-1, -1, -1), Vec(1, 1, 1)).Normalized();
}

void RLGC::RandomState::ResetArena(Arena* arena) {
	
	// Reset boost pads and everything
	arena->ResetToRandomKickoff();

	// Wide enough that corner/goal-line states (saves, clears, goalmouth scrambles) occur;
	// ball radius still clears the walls (3900+93 < 4096 side, 4800+93 < 5120 back).
	// The box corners (|x|+|y| up to 8700) poke through the 45-degree corner walls
	// (planes at |x|+|y| ~= 8064), so corner draws are rejection-resampled below —
	// otherwise ~1% of resets spawned an entity inside/behind the corner mesh (an
	// untouchable ball = a full NoTouch-truncated junk episode).
	constexpr float
		X_MAX = 3900,
		Y_MAX = 4800,
		Z_MAX = 1820,
		CAR_Z_MIN = 150,
		CORNER_XY_MAX = 7900, // |x|+|y| cap: 8064 corner plane minus ball/car clearance
		PITCH_MAX = M_PI / 2,
		YAW_MAX = M_PI,
		ROLL_MAX = M_PI,
		ANGVEL_MAX = 5.5f;

	auto fnRandPosInField = [&](float zMin, float zMax) {
		Vec p;
		do {
			p = Math::RandVec(Vec(-X_MAX, -Y_MAX, zMin), Vec(X_MAX, Y_MAX, zMax));
		} while (fabsf(p.x) + fabsf(p.y) > CORNER_XY_MAX);
		return p;
	};

	{ // Randomize ball
		BallState bs = {};
		bs.pos = fnRandPosInField(CommonValues::BALL_RADIUS, Z_MAX);
		if (randBallSpeed) {
			bs.vel = RandNormVec() * RandFloat(0, 4000);
			bs.angVel = Math::RandVec(Vec(-4, -4, -4), Vec(4, 4, 4));
		}
		arena->ball->SetState(bs);
	}

	for (Car* car : arena->_cars) { // Randomize cars
		CarState cs = {};
		cs.pos = fnRandPosInField(CAR_Z_MIN, Z_MAX);

		if (randCarSpeed) {
			Vec randVelDir = Math::RandVec(Vec(-1, -1, -1), Vec(1, 1, 1)).Normalized();
			cs.vel = RandNormVec() * RandFloat(0, RLConst::CAR_MAX_SPEED);
			cs.angVel = RandNormVec() * ANGVEL_MAX;
		}

		Angle angle = Angle(RandFloat(-YAW_MAX, YAW_MAX), RandFloat(-PITCH_MAX, PITCH_MAX), RandFloat(-ROLL_MAX, ROLL_MAX));

		bool onGround = carsOnGround ? true : (RandFloat() > 0.5);
		if (onGround) {
			cs.pos.z = 17;
			angle.pitch = angle.roll = 0;
			cs.vel.z = 0;
			cs.angVel = {};
		}

		cs.rotMat = angle.ToRotMat();

		cs.boost = RandFloat(0, 100);

		car->SetState(cs);
	}
}