#pragma once
// C ABI of rocketsim_ffi (../rocketsim_ffi/src/lib.rs). Hand-maintained mirror —
// any struct/field change must be made in BOTH files. All structs are plain
// f32/u8/u32/u64 (no SIMD alignment crosses the boundary).
#include <cstdint>

extern "C" {

struct RsfVec3 { float x, y, z; };
struct RsfRotMat { RsfVec3 forward, right, up; };
struct RsfPhysState { RsfVec3 pos; RsfRotMat rotMat; RsfVec3 vel, angVel; };

struct RsfCarControls {
	float throttle, steer, pitch, yaw, roll;
	uint8_t jump, boost, handbrake;
};

struct RsfCarState {
	RsfPhysState phys;
	RsfCarControls controls, prevControls;
	uint8_t isOnGround;
	uint8_t wheelsWithContact[4];
	uint8_t hasJumped, hasDoubleJumped, hasFlipped;
	RsfVec3 flipRelTorque;
	float jumpTime, flipTime;
	uint8_t isFlipping, isJumping;
	float airTime, airTimeSinceJump;
	float boost, timeSinceBoosted;
	uint8_t isBoosting;
	float boostingTime;
	uint8_t isSupersonic;
	float supersonicGraceTimer, handbrakeVal;
	uint8_t isAutoFlipping;
	float autoFlipTimer, autoFlipTorqueScale, bumpCooldownTimer;
	uint8_t hasWorldContact;
	RsfVec3 worldContactNormal;
	uint8_t isDemoed;
	float demoRespawnTimer;
};

struct RsfBallState { RsfPhysState phys; };
struct RsfPadConfig { RsfVec3 pos; uint8_t isBig; };
struct RsfPadState { float cooldown; uint8_t isActive; };

enum : uint32_t {
	RSF_EVENT_CAR_HIT_BALL = 0,
	RSF_EVENT_CAR_HIT_CAR = 1,
	RSF_EVENT_CAR_PICKUP_BOOST = 2,
	RSF_EVENT_BALL_HIT_WORLD = 3,
	RSF_EVENT_CAR_HIT_WORLD = 4,
};

struct RsfEvent {
	uint32_t kind, a, b;
	uint8_t isDemo;
	uint64_t tick;
	RsfVec3 pos, extra;
};

struct RsfArena; // opaque

int rsf_init(const char* path, int silent);
int rsf_is_initialized();
RsfArena* rsf_arena_new(uint32_t gameMode, int64_t seed);
void rsf_arena_free(RsfArena* arena);
uint32_t rsf_arena_add_car(RsfArena* arena, uint32_t team);
uint32_t rsf_arena_num_cars(RsfArena* arena);
void rsf_arena_step(RsfArena* arena, uint32_t ticks);
uint32_t rsf_arena_get_events(RsfArena* arena, RsfEvent* out, uint32_t cap, uint32_t* total);
void rsf_arena_get_car_state(RsfArena* arena, uint32_t idx, RsfCarState* out);
void rsf_arena_set_car_state(RsfArena* arena, uint32_t idx, const RsfCarState* s);

// Suspension / pending-bump state that RsfCarState cannot express, moved as an OPAQUE
// block: ask for the size, keep that many bytes, hand them back. Nothing on this side
// names a field, so the Rust struct can grow without a header change here — unlike the
// rest of this file, these two cannot drift out of sync. Needed only for exact state
// restore (the viz control panel's rewind); ordinary stepping never calls them.
uint32_t rsf_car_extra_state_size(void);
void rsf_arena_get_car_extra_state(RsfArena* arena, uint32_t idx, void* out);
void rsf_arena_set_car_extra_state(RsfArena* arena, uint32_t idx, const void* s);

// Arena RNG state (demo respawn spawn-point selection is its only consumer).
uint64_t rsf_arena_get_rng_state(RsfArena* arena);
void rsf_arena_set_rng_state(RsfArena* arena, uint64_t state);
void rsf_arena_set_car_controls(RsfArena* arena, uint32_t idx, const RsfCarControls* c);
void rsf_arena_respawn_car(RsfArena* arena, uint32_t idx);
void rsf_arena_get_ball_state(RsfArena* arena, RsfBallState* out);
void rsf_arena_set_ball_state(RsfArena* arena, const RsfBallState* s);
void rsf_arena_reset_kickoff(RsfArena* arena, int64_t seed);
uint64_t rsf_arena_tick_count(RsfArena* arena);
int rsf_arena_is_ball_scored(RsfArena* arena);
uint32_t rsf_arena_num_pads(RsfArena* arena);
void rsf_arena_get_pad_config(RsfArena* arena, uint32_t idx, RsfPadConfig* out);
void rsf_arena_get_pad_state(RsfArena* arena, uint32_t idx, RsfPadState* out);
void rsf_arena_set_pad_state(RsfArena* arena, uint32_t idx, const RsfPadState* s);

} // extern "C"
