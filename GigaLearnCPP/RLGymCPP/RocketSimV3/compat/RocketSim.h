#pragma once
// v2-API-compatible facade over the vendored v3 Rust engine (see ../README.md).
// Consumers (RLGymCPP/GigaLearnCPP) compile against this header unchanged; the
// physics itself runs in Rust behind the C ABI in RocketSimV3FFI.h.
//
// Engine-independent v2 headers are reused directly (math types, constants,
// PhysState, controls, game modes) so consumer code sees byte-identical types;
// only the engine classes (Arena/Car/Ball/BoostPad) are reimplemented here.
// The API surface is exactly the consumer inventory of 2026-07-17 — if a new
// consumer needs a v2 symbol this facade lacks, add it here, don't include v2
// engine headers.

#include "../../RocketSim/src/BaseInc.h"
#include "../../RocketSim/src/RLConst.h"
#include "../../RocketSim/src/Sim/GameMode.h"
#include "../../RocketSim/src/Sim/CarControls.h"
#include "../../RocketSim/src/Sim/PhysState/PhysState.h"
#include "../../RocketSim/src/Sim/BallHitInfo/BallHitInfo.h"
#include "../../RocketSim/src/Math/Math.h"

#include "RocketSimV3FFI.h"

#include <vector>
#include <filesystem>
#include <functional>

RS_NS_START

enum class RocketSimStage : byte {
	UNINITIALIZED,
	INITIALIZING,
	INITIALIZED
};
RocketSimStage GetStage();
void Init(std::filesystem::path collisionMeshesFolder, bool silent = false);

enum class Team : byte {
	BLUE = 0,
	ORANGE = 1
};

#define RS_OPPOSITE_TEAM(team) ((team) == Team::BLUE ? Team::ORANGE : Team::BLUE)
#define RS_TEAM_FROM_Y(y) ((y) < 0 ? Team::BLUE : Team::ORANGE)

// Field-for-field copy of v2's CarState (Sim/Car/Car.h) minus serialization.
// Fields with no v3 counterpart (updateCounter) are carried but inert — see
// the mapping notes in RocketSimCompat.cpp.
struct CarState : public PhysState {
	uint64_t updateCounter = 0;
	bool isOnGround = true;
	bool wheelsWithContact[4] = {};
	// [0,1] compressed→extended; wheel order FR, FL, BR, BL (same as wheelsWithContact)
	float wheelsSuspension[4] = {};
	bool hasJumped = false;
	bool hasDoubleJumped = false;
	bool hasFlipped = false;
	Vec flipRelTorque = { 0, 0, 0 };
	float jumpTime = 0;
	float flipTime = 0;
	bool isFlipping = false;
	bool isJumping = false;
	float airTime = 0;
	float airTimeSinceJump = 0;
	float boost = RLConst::BOOST_SPAWN_AMOUNT;
	float timeSpentBoosting = 0;
	bool isSupersonic = false;
	// Compat name. Value is v3 supersonic_grace_timer: seconds spent in the
	// [2100, 2200) band after dropping below START_SPEED (resets if you go back
	// above 2200). Not "time since first became supersonic."
	float supersonicTime = 0;
	float handbrakeVal = 0;
	bool isAutoFlipping = false;
	float autoFlipTimer = 0;
	float autoFlipTorqueScale = 0;

	struct {
		bool hasContact = false;
		Vec contactNormal;
	} worldContact;

	struct {
		uint32_t otherCarID = 0;
		float cooldownTimer = 0;
	} carContact;

	bool isDemoed = false;
	float demoRespawnTimer = 0;

	BallHitInfo ballHitInfo = BallHitInfo();
	// v3 ball_extra_impulse_tick. ~0 = never. Next on_hit is gated if
	// tickCount <= last+1 (unless last > tickCount).
	uint64_t ballExtraImpulseTick = ~0ULL;

	CarControls lastControls = CarControls();

	CarState() : PhysState() {
		pos.z = RLConst::CAR_SPAWN_REST_Z;
	}

	bool HasFlipOrJump() const {
		return isOnGround ||
			(!hasFlipped && !hasDoubleJumped && airTimeSinceJump < RLConst::DOUBLEJUMP_MAX_DELAY);
	}
	bool HasFlipReset() const { return !isOnGround && HasFlipOrJump() && !hasJumped; }
	bool GotFlipReset() const { return !isOnGround && !hasJumped; }
};

struct BallState : public PhysState {
	// v2 keeps the ball at rest height by default; heatseeker/dropshot info is
	// engine-side in v3 and preserved across SetState there.
	BallState() : PhysState() {
		pos.z = RLConst::BALL_REST_Z;
	}
};

class Arena;

class Car {
public:
	Team team;
	uint32_t id;   // 1-based, = v3 car index + 1
	CarControls controls;

	RSAPI CarState GetState();
	RSAPI void SetState(const CarState& state);

	// Compat internals (public like v2's _-prefixed members; not part of the v2 API)
	Arena* _arena = NULL;
	uint32_t _idx = 0;
	BallHitInfo _ballHitInfo = {}; // synthesized from CarHitBall events, persists across steps
};

class Ball {
public:
	RSAPI BallState GetState();
	RSAPI void SetState(const BallState& state);

	Arena* _arena = NULL;
	// v2 stateset-detection semantics for GameEventTracker: increments each
	// simulated tick, resets on SetState.
	uint64_t _updateCounter = 0;
};

struct BoostPadConfig {
	Vec pos;
	bool isBig = false;
};

struct BoostPadState {
	bool isActive = true;
	float cooldown = 0;
	Car* curLockedCar = NULL;
};

class BoostPad {
public:
	BoostPadConfig config;

	RSAPI BoostPadState GetState();
	RSAPI void SetState(const BoostPadState& state);

	Arena* _arena = NULL;
	uint32_t _idx = 0;
};

typedef std::function<void(Arena* arena, Car* bumper, Car* victim, bool isDemo, void* userInfo)> CarBumpEventFn;

class Arena {
public:
	GameMode gameMode;
	uint64_t tickCount = 0;
	float tickTime = 1 / 120.f;
	Ball* ball = NULL;

	std::vector<Car*> _cars;
	std::vector<BoostPad*> _boostPads;

	RSAPI static Arena* Create(GameMode gameMode, float tickRate = 120);
	RSAPI ~Arena();

	std::vector<Car*>& GetCars() { return _cars; }
	std::vector<BoostPad*>& GetBoostPads() { return _boostPads; }
	// v3 only supports 120Hz. Do not return 1/tickTime: 1/(1/120.f) is not 120.f.
	float GetTickRate() const { return 120.f; }

	RSAPI Car* AddCar(Team team);
	// True if this car (Car::id) is the pending claimant of any pad.
	RSAPI bool CarHasPendingPadGrant(uint32_t carId) const;
	RSAPI void Step(int ticksToSimulate = 1);
	RSAPI void ResetToRandomKickoff(int seed = -1);
	RSAPI bool IsBallScored() const;

	// Closed-form ballistic score prediction, ported verbatim from v2
	// Arena.cpp (soccar path only) for GameEventTracker.
	RSAPI bool IsBallProbablyGoingIn(float maxTime = 0.2f, float extraMargin = 0, Team* goalTeamOut = NULL) const;

	// Closest static mesh (plus the floor plane) to a world-UU query.
	// hit=false if nothing is inside maxDist; dist is then maxDist.
	struct ClosestSurfaceHit {
		Vec point;
		Vec normal;
		float dist = 0.f;
		bool hit = false;
	};
	RSAPI void QueryClosestSurface(Vec query, float maxDist, ClosestSurfaceHit& out) const;
	RSAPI void QueryClosestSurfaceN(
		const Vec* queries, int n, float maxDist, ClosestSurfaceHit* out) const;

	void SetCarBumpCallback(CarBumpEventFn callbackFn, void* userInfo = NULL) {
		_bumpCallback.func = callbackFn;
		_bumpCallback.userInfo = userInfo;
	}

	// Compat internals
	RsfArena* _rs = NULL;
	struct {
		CarBumpEventFn func = NULL;
		void* userInfo = NULL;
	} _bumpCallback;

	Arena() = default;
	Arena(const Arena&) = delete;
	Arena& operator=(const Arena&) = delete;
};

RS_NS_END
