#include "RocketSim.h"

#include <cfloat>

// State-mapping notes (v2 field <- v3 field):
//  - lastControls <- prev_controls (same semantics: controls used last sim tick)
//  - supersonicTime: v2 counts time spent supersonic, v3 keeps a grace timer
//    since dropping below threshold. No consumer reads or meaningfully writes
//    either (state setters leave the default 0), so both directions map to 0.
//  - timeSpentBoosting <-> boosting_time (v3 splits is_boosting out; v2 infers
//    boosting from timeSpentBoosting > 0).
//  - carContact / updateCounter: no v3 counterpart, inert (no consumer reads).
//  - ballHitInfo: v3 reports touches as step events, not car state; the compat
//    Arena accumulates them per-car into Car::_ballHitInfo (persists across
//    steps like v2), and SetState overwrites it (setters reset it via {}).

RS_NS_START

static RocketSimStage g_Stage = RocketSimStage::UNINITIALIZED;

RocketSimStage GetStage() {
	return g_Stage;
}

void Init(std::filesystem::path collisionMeshesFolder, bool silent) {
	g_Stage = RocketSimStage::INITIALIZING;
	int result = rsf_init(collisionMeshesFolder.string().c_str(), silent ? 1 : 0);
	if (result != 0)
		RS_ERR_CLOSE("RocketSim(v3)::Init() failed to load collision meshes from " << collisionMeshesFolder << " (code " << result << ")");
	g_Stage = RocketSimStage::INITIALIZED;
}

/////////////// conversions ///////////////

static Vec FromRsf(const RsfVec3& v) { return Vec(v.x, v.y, v.z); }
static RsfVec3 ToRsf(const Vec& v) { return RsfVec3{ v.x, v.y, v.z }; }

static RotMat FromRsf(const RsfRotMat& m) { return RotMat(FromRsf(m.forward), FromRsf(m.right), FromRsf(m.up)); }
static RsfRotMat ToRsf(const RotMat& m) { return RsfRotMat{ ToRsf(m.forward), ToRsf(m.right), ToRsf(m.up) }; }

static CarControls FromRsf(const RsfCarControls& c) {
	CarControls out = {};
	out.throttle = c.throttle;
	out.steer = c.steer;
	out.pitch = c.pitch;
	out.yaw = c.yaw;
	out.roll = c.roll;
	out.jump = c.jump;
	out.boost = c.boost;
	out.handbrake = c.handbrake;
	return out;
}
static RsfCarControls ToRsf(const CarControls& c) {
	return RsfCarControls{
		c.throttle, c.steer, c.pitch, c.yaw, c.roll,
		(uint8_t)c.jump, (uint8_t)c.boost, (uint8_t)c.handbrake
	};
}

static void PhysFromRsf(PhysState& out, const RsfPhysState& p) {
	out.pos = FromRsf(p.pos);
	out.rotMat = FromRsf(p.rotMat);
	out.vel = FromRsf(p.vel);
	out.angVel = FromRsf(p.angVel);
}
static RsfPhysState PhysToRsf(const PhysState& p) {
	return RsfPhysState{ ToRsf(p.pos), ToRsf(p.rotMat), ToRsf(p.vel), ToRsf(p.angVel) };
}

/////////////// Car ///////////////

CarState Car::GetState() {
	RsfCarState rs;
	rsf_arena_get_car_state(_arena->_rs, _idx, &rs);

	CarState s = {};
	PhysFromRsf(s, rs.phys);
	s.updateCounter = _arena->tickCount;
	s.isOnGround = rs.isOnGround;
	for (int i = 0; i < 4; i++)
		s.wheelsWithContact[i] = rs.wheelsWithContact[i];
	s.hasJumped = rs.hasJumped;
	s.hasDoubleJumped = rs.hasDoubleJumped;
	s.hasFlipped = rs.hasFlipped;
	s.flipRelTorque = FromRsf(rs.flipRelTorque);
	s.jumpTime = rs.jumpTime;
	s.flipTime = rs.flipTime;
	s.isFlipping = rs.isFlipping;
	s.isJumping = rs.isJumping;
	s.airTime = rs.airTime;
	s.airTimeSinceJump = rs.airTimeSinceJump;
	s.boost = rs.boost;
	s.timeSpentBoosting = rs.boostingTime;
	s.isSupersonic = rs.isSupersonic;
	s.supersonicTime = 0;
	s.handbrakeVal = rs.handbrakeVal;
	s.isAutoFlipping = rs.isAutoFlipping;
	s.autoFlipTimer = rs.autoFlipTimer;
	s.autoFlipTorqueScale = rs.autoFlipTorqueScale;
	s.worldContact.hasContact = rs.hasWorldContact;
	s.worldContact.contactNormal = FromRsf(rs.worldContactNormal);
	s.isDemoed = rs.isDemoed;
	s.demoRespawnTimer = rs.demoRespawnTimer;
	s.ballHitInfo = _ballHitInfo;
	s.lastControls = FromRsf(rs.prevControls);
	return s;
}

void Car::SetState(const CarState& state) {
	RsfCarState rs = {};
	rs.phys = PhysToRsf(state);
	rs.controls = ToRsf(controls);
	rs.prevControls = ToRsf(state.lastControls);
	rs.isOnGround = state.isOnGround;
	for (int i = 0; i < 4; i++)
		rs.wheelsWithContact[i] = state.wheelsWithContact[i];
	rs.hasJumped = state.hasJumped;
	rs.hasDoubleJumped = state.hasDoubleJumped;
	rs.hasFlipped = state.hasFlipped;
	rs.flipRelTorque = ToRsf(state.flipRelTorque);
	rs.jumpTime = state.jumpTime;
	rs.flipTime = state.flipTime;
	rs.isFlipping = state.isFlipping;
	rs.isJumping = state.isJumping;
	rs.airTime = state.airTime;
	rs.airTimeSinceJump = state.airTimeSinceJump;
	rs.boost = state.boost;
	rs.timeSinceBoosted = 0;
	rs.isBoosting = state.timeSpentBoosting > 0;
	rs.boostingTime = state.timeSpentBoosting;
	rs.isSupersonic = state.isSupersonic;
	rs.supersonicGraceTimer = 0;
	rs.handbrakeVal = state.handbrakeVal;
	rs.isAutoFlipping = state.isAutoFlipping;
	rs.autoFlipTimer = state.autoFlipTimer;
	rs.autoFlipTorqueScale = state.autoFlipTorqueScale;
	rs.bumpCooldownTimer = state.carContact.cooldownTimer;
	rs.hasWorldContact = state.worldContact.hasContact;
	rs.worldContactNormal = ToRsf(state.worldContact.contactNormal);
	rs.isDemoed = state.isDemoed;
	rs.demoRespawnTimer = state.demoRespawnTimer;
	rsf_arena_set_car_state(_arena->_rs, _idx, &rs);

	_ballHitInfo = state.ballHitInfo;
}

/////////////// Ball ///////////////

BallState Ball::GetState() {
	RsfBallState rs;
	rsf_arena_get_ball_state(_arena->_rs, &rs);
	BallState s = {};
	PhysFromRsf(s, rs.phys);
	return s;
}

void Ball::SetState(const BallState& state) {
	RsfBallState rs = {};
	rs.phys = PhysToRsf(state);
	rsf_arena_set_ball_state(_arena->_rs, &rs);
	// v2 stateset-detection contract: the update counter resets on SetState so
	// GameEventTracker::Update sees a decrease and clears persistent info.
	_updateCounter = 0;
}

/////////////// BoostPad ///////////////

BoostPadState BoostPad::GetState() {
	RsfPadState rs;
	rsf_arena_get_pad_state(_arena->_rs, _idx, &rs);
	BoostPadState s = {};
	s.isActive = rs.isActive;
	s.cooldown = rs.cooldown;
	s.curLockedCar = NULL;
	return s;
}

void BoostPad::SetState(const BoostPadState& state) {
	// v3 derives isActive from cooldown <= 0; keep the v2 pair consistent.
	RsfPadState rs = {};
	rs.cooldown = state.isActive ? 0.0f : RS_MAX(state.cooldown, 1 / 120.f);
	rs.isActive = state.isActive;
	rsf_arena_set_pad_state(_arena->_rs, _idx, &rs);
}

/////////////// Arena ///////////////

static uint32_t GameModeToRsf(GameMode mode) {
	switch (mode) {
	case GameMode::SOCCAR: return 0;
	case GameMode::HOOPS: return 1;
	case GameMode::HEATSEEKER: return 2;
	case GameMode::SNOWDAY: return 3;
	case GameMode::THE_VOID: return 5;
	default:
		RS_ERR_CLOSE("RocketSim(v3) compat: unsupported GameMode " << (int)mode);
		return 0;
	}
}

Arena* Arena::Create(GameMode gameMode, float tickRate) {
	if (g_Stage != RocketSimStage::INITIALIZED)
		RS_ERR_CLOSE("RocketSim(v3) compat: Arena::Create() before Init()");
	if (tickRate != 120)
		RS_ERR_CLOSE("RocketSim(v3) compat: only 120Hz tick rate is supported (got " << tickRate << ")");

	Arena* arena = new Arena();
	arena->gameMode = gameMode;
	arena->tickTime = 1 / tickRate;
	arena->_rs = rsf_arena_new(GameModeToRsf(gameMode), -1);
	arena->tickCount = rsf_arena_tick_count(arena->_rs);

	arena->ball = new Ball();
	arena->ball->_arena = arena;

	uint32_t numPads = rsf_arena_num_pads(arena->_rs);
	arena->_boostPads.reserve(numPads);
	for (uint32_t i = 0; i < numPads; i++) {
		RsfPadConfig cfg;
		rsf_arena_get_pad_config(arena->_rs, i, &cfg);
		BoostPad* pad = new BoostPad();
		pad->_arena = arena;
		pad->_idx = i;
		pad->config.pos = FromRsf(cfg.pos);
		pad->config.isBig = cfg.isBig;
		arena->_boostPads.push_back(pad);
	}

	return arena;
}

Arena::~Arena() {
	for (Car* car : _cars)
		delete car;
	for (BoostPad* pad : _boostPads)
		delete pad;
	delete ball;
	if (_rs)
		rsf_arena_free(_rs);
}

Car* Arena::AddCar(Team team) {
	uint32_t idx = rsf_arena_add_car(_rs, (uint32_t)team);

	Car* car = new Car();
	car->_arena = this;
	car->_idx = idx;
	car->id = idx + 1; // v2 ids are unique and > 0
	car->team = team;
	if (idx != _cars.size())
		RS_ERR_CLOSE("RocketSim(v3) compat: engine car index " << idx << " != wrapper count " << _cars.size());
	_cars.push_back(car);
	return car;
}

void Arena::Step(int ticksToSimulate) {
	if (ticksToSimulate <= 0)
		return;

	for (Car* car : _cars) {
		RsfCarControls c = ToRsf(car->controls);
		rsf_arena_set_car_controls(_rs, car->_idx, &c);
	}

	rsf_arena_step(_rs, (uint32_t)ticksToSimulate);
	tickCount = rsf_arena_tick_count(_rs);
	ball->_updateCounter += ticksToSimulate;

	// Drain the step's events: synthesize v2 ballHitInfo + bump callbacks
	uint32_t total = 0;
	rsf_arena_get_events(_rs, NULL, 0, &total);
	if (total > 0) {
		static thread_local std::vector<RsfEvent> events;
		events.resize(total);
		rsf_arena_get_events(_rs, events.data(), total, NULL);

		for (const RsfEvent& ev : events) {
			switch (ev.kind) {
			case RSF_EVENT_CAR_HIT_BALL: {
				Car* car = _cars[ev.a];
				car->_ballHitInfo.isValid = true;
				car->_ballHitInfo.tickCountWhenHit = ev.tick;
				car->_ballHitInfo.ballPos = FromRsf(ev.pos);
				car->_ballHitInfo.extraHitVel = FromRsf(ev.extra);
				car->_ballHitInfo.relativePosOnBall = Vec();
				break;
			}
			case RSF_EVENT_CAR_HIT_CAR: {
				if (_bumpCallback.func)
					_bumpCallback.func(this, _cars[ev.a], _cars[ev.b], ev.isDemo, _bumpCallback.userInfo);
				break;
			}
			default:
				break;
			}
		}
	}
}

void Arena::ResetToRandomKickoff(int seed) {
	rsf_arena_reset_kickoff(_rs, seed);
	tickCount = rsf_arena_tick_count(_rs);
	ball->_updateCounter = 0;
	for (Car* car : _cars)
		car->_ballHitInfo = {};
}

bool Arena::IsBallScored() const {
	return rsf_arena_is_ball_scored(_rs) != 0;
}

// Ported verbatim from v2 Arena.cpp:827 (soccar/snowday path); uses default
// mutator values (goalBaseThresholdY, gravity, ballRadius) since the compat
// layer exposes no mutator config.
bool Arena::IsBallProbablyGoingIn(float maxTime, float extraMargin, Team* goalTeamOut) const {
	RsfBallState rs;
	rsf_arena_get_ball_state(_rs, &rs);
	Vec ballPos = FromRsf(rs.phys.pos);
	Vec ballVel = FromRsf(rs.phys.vel);

	if (gameMode == GameMode::SOCCAR || gameMode == GameMode::SNOWDAY) {
		if (abs(ballVel.y) < FLT_EPSILON)
			return false;

		float scoreDirSgn = RS_SGN(ballVel.y);
		float goalY = RLConst::SOCCAR_GOAL_SCORE_BASE_THRESHOLD_Y * scoreDirSgn;
		float distToGoal = abs(ballPos.y - goalY);

		float timeToGoal = distToGoal / abs(ballVel.y);

		if (timeToGoal > maxTime)
			return false;

		Vec gravity = Vec(0, 0, RLConst::GRAVITY_Z);
		Vec extrapPosWhenScore = ballPos + (ballVel * timeToGoal) + (gravity * timeToGoal * timeToGoal) / 2;

		constexpr float
			APPROX_GOAL_HALF_WIDTH = 892.755f,
			APPROX_GOAL_HEIGHT = 642.775;

		float scoreMargin = RLConst::BALL_COLLISION_RADIUS_SOCCAR * 0.1f + extraMargin;

		if (extrapPosWhenScore.z > APPROX_GOAL_HEIGHT + scoreMargin)
			return false;

		if (abs(extrapPosWhenScore.x) > APPROX_GOAL_HALF_WIDTH + scoreMargin)
			return false;

		if (goalTeamOut)
			*goalTeamOut = RS_TEAM_FROM_Y(scoreDirSgn);

		return true;
	} else {
		RS_ERR_CLOSE("RocketSim(v3) compat: IsBallProbablyGoingIn() unsupported for gamemode " << (int)gameMode);
		return false;
	}
}

RS_NS_END
