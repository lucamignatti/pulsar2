#include "RandomState.h"
#include "../Math.h"
#include "../Util/CrashDebug.h"

using RocketSim::Math::RandFloat;
using RLGC::Math::RandVec;

Vec RandNormVec() {
	return RandVec(Vec(-1, -1, -1), Vec(1, 1, 1)).Normalized();
}

void RLGC::RandomState::ResetArena(Arena* arena) {
	RG_CRASH_LOG(
		"RandomState::ResetArena begin arena=" << arena <<
		" randBallSpeed=" << randBallSpeed <<
		" randCarSpeed=" << randCarSpeed <<
		" carsOnGround=" << carsOnGround <<
		" cars=" << arena->_cars.size()
	);
	
	// Reset boost pads and everything
	arena->ResetToRandomKickoff();
	RG_CRASH_LOG("RandomState::ResetArena after ResetToRandomKickoff arena=" << arena << " tick=" << arena->tickCount);

	constexpr float
		X_MAX = 3500,
		Y_MAX = 4000,
		Z_MAX = 1820,
		CAR_Z_MIN = 150,
		PITCH_MAX = M_PI / 2,
		YAW_MAX = M_PI,
		ROLL_MAX = M_PI,
		ANGVEL_MAX = 5.5f;

	{ // Randomize ball
		BallState bs = {};
		bs.pos = Math::RandVec(Vec(-X_MAX, -Y_MAX, CommonValues::BALL_RADIUS), Vec(X_MAX, Y_MAX, Z_MAX));
		if (randBallSpeed) {
			bs.vel = RandNormVec() * RandFloat(0, 4000);
			bs.angVel = Math::RandVec(Vec(-4, -4, -4), Vec(4, 4, 4));
		}
		RG_CRASH_LOG(
			"RandomState ball arena=" << arena <<
			" pos=(" << bs.pos.x << "," << bs.pos.y << "," << bs.pos.z << ")" <<
			" vel=(" << bs.vel.x << "," << bs.vel.y << "," << bs.vel.z << ")" <<
			" ang=(" << bs.angVel.x << "," << bs.angVel.y << "," << bs.angVel.z << ")"
		);
		arena->ball->SetState(bs);
	}

	int carIdx = 0;
	for (Car* car : arena->_cars) { // Randomize cars
		CarState cs = {};
		cs.pos = Math::RandVec(Vec(-X_MAX, -Y_MAX, CAR_Z_MIN), Vec(X_MAX, Y_MAX, Z_MAX));

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

		RG_CRASH_LOG(
			"RandomState car arena=" << arena <<
			" carIdx=" << carIdx <<
			" car=" << car <<
			" onGround=" << onGround <<
			" pos=(" << cs.pos.x << "," << cs.pos.y << "," << cs.pos.z << ")" <<
			" vel=(" << cs.vel.x << "," << cs.vel.y << "," << cs.vel.z << ")" <<
			" ang=(" << cs.angVel.x << "," << cs.angVel.y << "," << cs.angVel.z << ")" <<
			" boost=" << cs.boost
		);
		car->SetState(cs);
		carIdx++;
	}
	RG_CRASH_LOG("RandomState::ResetArena end arena=" << arena);
}
