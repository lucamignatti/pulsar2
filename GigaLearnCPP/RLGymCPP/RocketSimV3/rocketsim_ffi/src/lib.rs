//! C FFI over the vendored v3 Rust engine, consumed by the C++ compat layer in
//! ../compat/. Layout contract: every struct here is #[repr(C)] with plain
//! f32/u8/u32 fields (glam's Vec3A is 16-byte SIMD-aligned and must never cross
//! the boundary). The C++ side mirrors these in RocketSimV3FFI.h — keep the two
//! in sync BY HAND; there is no bindgen step.
//!
//! Threading contract: one arena is only ever touched by one thread at a time
//! (the EnvSet thread-pool model). Arenas share no mutable global state after
//! rsf_init (mesh shapes are Arc'd immutable), asserted Send below.

use std::ffi::CStr;
use std::os::raw::{c_char, c_int};

use glam::{Mat3A, Vec3A};
use rocketsim::{
    Arena, ArenaConfig, ArenaEvent, BallState, BoostPadState, CarBodyConfig, CarControls,
    CarExtraState, CarState, GameMode, Team,
};

// A Rust panic aborts (panic="abort") rather than unwinding into C++; arena
// handles must therefore only be used with valid indices — same fail-fast
// contract as v2's asserts.
const _: () = {
    const fn assert_send<T: Send>() {}
    assert_send::<Arena>()
};

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct RsfVec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl From<Vec3A> for RsfVec3 {
    fn from(v: Vec3A) -> Self {
        Self { x: v.x, y: v.y, z: v.z }
    }
}
impl From<RsfVec3> for Vec3A {
    fn from(v: RsfVec3) -> Self {
        Vec3A::new(v.x, v.y, v.z)
    }
}

/// Axes follow v2's RotMat convention: forward/right/up.
/// v3 stores Mat3A with x_axis=forward, y_axis=right, z_axis=up.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct RsfRotMat {
    pub forward: RsfVec3,
    pub right: RsfVec3,
    pub up: RsfVec3,
}

impl From<Mat3A> for RsfRotMat {
    fn from(m: Mat3A) -> Self {
        Self {
            forward: m.x_axis.into(),
            right: m.y_axis.into(),
            up: m.z_axis.into(),
        }
    }
}
impl From<RsfRotMat> for Mat3A {
    fn from(m: RsfRotMat) -> Self {
        Mat3A::from_cols(m.forward.into(), m.right.into(), m.up.into())
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct RsfPhysState {
    pub pos: RsfVec3,
    pub rot_mat: RsfRotMat,
    pub vel: RsfVec3,
    pub ang_vel: RsfVec3,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct RsfCarControls {
    pub throttle: f32,
    pub steer: f32,
    pub pitch: f32,
    pub yaw: f32,
    pub roll: f32,
    pub jump: u8,
    pub boost: u8,
    pub handbrake: u8,
}

impl From<CarControls> for RsfCarControls {
    fn from(c: CarControls) -> Self {
        Self {
            throttle: c.throttle,
            steer: c.steer,
            pitch: c.pitch,
            yaw: c.yaw,
            roll: c.roll,
            jump: c.jump as u8,
            boost: c.boost as u8,
            handbrake: c.handbrake as u8,
        }
    }
}
impl From<RsfCarControls> for CarControls {
    fn from(c: RsfCarControls) -> Self {
        Self {
            throttle: c.throttle,
            steer: c.steer,
            pitch: c.pitch,
            yaw: c.yaw,
            roll: c.roll,
            jump: c.jump != 0,
            boost: c.boost != 0,
            handbrake: c.handbrake != 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RsfCarState {
    pub phys: RsfPhysState,
    pub controls: RsfCarControls,
    pub prev_controls: RsfCarControls,
    pub is_on_ground: u8,
    pub wheels_with_contact: [u8; 4],
    pub has_jumped: u8,
    pub has_double_jumped: u8,
    pub has_flipped: u8,
    pub flip_rel_torque: RsfVec3,
    pub jump_time: f32,
    pub flip_time: f32,
    pub is_flipping: u8,
    pub is_jumping: u8,
    pub air_time: f32,
    pub air_time_since_jump: f32,
    pub boost: f32,
    pub time_since_boosted: f32,
    pub is_boosting: u8,
    pub boosting_time: f32,
    pub is_supersonic: u8,
    pub supersonic_grace_timer: f32,
    pub handbrake_val: f32,
    pub is_auto_flipping: u8,
    pub auto_flip_timer: f32,
    pub auto_flip_torque_scale: f32,
    pub bump_cooldown_timer: f32,
    pub has_world_contact: u8,
    pub world_contact_normal: RsfVec3,
    pub is_demoed: u8,
    pub demo_respawn_timer: f32,
}

impl From<&CarState> for RsfCarState {
    fn from(s: &CarState) -> Self {
        Self {
            phys: RsfPhysState {
                pos: s.phys.pos.into(),
                rot_mat: s.phys.rot_mat.into(),
                vel: s.phys.vel.into(),
                ang_vel: s.phys.ang_vel.into(),
            },
            controls: s.controls.into(),
            prev_controls: s.prev_controls.into(),
            is_on_ground: s.is_on_ground as u8,
            wheels_with_contact: s.wheels_with_contact.map(|b| b as u8),
            has_jumped: s.has_jumped as u8,
            has_double_jumped: s.has_double_jumped as u8,
            has_flipped: s.has_flipped as u8,
            flip_rel_torque: s.flip_rel_torque.into(),
            jump_time: s.jump_time,
            flip_time: s.flip_time,
            is_flipping: s.is_flipping as u8,
            is_jumping: s.is_jumping as u8,
            air_time: s.air_time,
            air_time_since_jump: s.air_time_since_jump,
            boost: s.boost,
            time_since_boosted: s.time_since_boosted,
            is_boosting: s.is_boosting as u8,
            boosting_time: s.boosting_time,
            is_supersonic: s.is_supersonic as u8,
            supersonic_grace_timer: s.supersonic_grace_timer,
            handbrake_val: s.handbrake_val,
            is_auto_flipping: s.is_auto_flipping as u8,
            auto_flip_timer: s.auto_flip_timer,
            auto_flip_torque_scale: s.auto_flip_torque_scale,
            bump_cooldown_timer: s.bump_cooldown_timer,
            has_world_contact: s.world_contact_normal.is_some() as u8,
            world_contact_normal: s.world_contact_normal.unwrap_or(Vec3A::ZERO).into(),
            is_demoed: s.is_demoed as u8,
            demo_respawn_timer: s.demo_respawn_timer,
        }
    }
}

impl From<&RsfCarState> for CarState {
    fn from(s: &RsfCarState) -> Self {
        CarState {
            phys: rocketsim::PhysState {
                pos: s.phys.pos.into(),
                rot_mat: s.phys.rot_mat.into(),
                vel: s.phys.vel.into(),
                ang_vel: s.phys.ang_vel.into(),
            },
            controls: s.controls.into(),
            prev_controls: s.prev_controls.into(),
            is_on_ground: s.is_on_ground != 0,
            wheels_with_contact: s.wheels_with_contact.map(|b| b != 0),
            has_jumped: s.has_jumped != 0,
            has_double_jumped: s.has_double_jumped != 0,
            has_flipped: s.has_flipped != 0,
            flip_rel_torque: s.flip_rel_torque.into(),
            jump_time: s.jump_time,
            flip_time: s.flip_time,
            is_flipping: s.is_flipping != 0,
            is_jumping: s.is_jumping != 0,
            air_time: s.air_time,
            air_time_since_jump: s.air_time_since_jump,
            boost: s.boost,
            time_since_boosted: s.time_since_boosted,
            is_boosting: s.is_boosting != 0,
            boosting_time: s.boosting_time,
            is_supersonic: s.is_supersonic != 0,
            supersonic_grace_timer: s.supersonic_grace_timer,
            handbrake_val: s.handbrake_val,
            is_auto_flipping: s.is_auto_flipping != 0,
            auto_flip_timer: s.auto_flip_timer,
            auto_flip_torque_scale: s.auto_flip_torque_scale,
            bump_cooldown_timer: s.bump_cooldown_timer,
            world_contact_normal: (s.has_world_contact != 0)
                .then(|| s.world_contact_normal.into()),
            is_demoed: s.is_demoed != 0,
            demo_respawn_timer: s.demo_respawn_timer,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct RsfBallState {
    pub phys: RsfPhysState,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct RsfPadConfig {
    pub pos: RsfVec3,
    pub is_big: u8,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct RsfPadState {
    pub cooldown: f32,
    pub is_active: u8,
}

pub const RSF_EVENT_CAR_HIT_BALL: u32 = 0;
pub const RSF_EVENT_CAR_HIT_CAR: u32 = 1;
pub const RSF_EVENT_CAR_PICKUP_BOOST: u32 = 2;
pub const RSF_EVENT_BALL_HIT_WORLD: u32 = 3;
pub const RSF_EVENT_CAR_HIT_WORLD: u32 = 4;

/// kind-dependent fields:
///  CAR_HIT_BALL:     a=car_idx,            pos=contact_point, extra=extra_hit_vel
///  CAR_HIT_CAR:      a=bumper, b=victim,   pos=contact_point, is_demo
///  CAR_PICKUP_BOOST: a=car_idx, b=pad_idx
///  BALL_HIT_WORLD:   pos=contact_point,    extra=contact_normal
///  CAR_HIT_WORLD:    a=car_idx, pos=contact_point, extra=contact_normal
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RsfEvent {
    pub kind: u32,
    pub a: u32,
    pub b: u32,
    pub is_demo: u8,
    pub tick: u64,
    pub pos: RsfVec3,
    pub extra: RsfVec3,
}

/// Owns the arena plus the event log accumulated across the ticks of the most
/// recent rsf_arena_step call (v3's get_last_step_events only covers ONE tick).
pub struct FfiArena {
    arena: Arena,
    events: Vec<RsfEvent>,
}

fn game_mode_from(v: u32) -> GameMode {
    match v {
        0 => GameMode::Soccar,
        1 => GameMode::Hoops,
        2 => GameMode::Heatseeker,
        3 => GameMode::Snowday,
        4 => GameMode::Dropshot,
        _ => GameMode::TheVoid,
    }
}

/// # Safety
/// `path` must be a valid NUL-terminated UTF-8 path.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rsf_init(path: *const c_char, silent: c_int) -> c_int {
    let path = unsafe { CStr::from_ptr(path) };
    let Ok(path) = path.to_str() else {
        return -2;
    };
    match rocketsim::init(path, silent != 0) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rsf_is_initialized() -> c_int {
    rocketsim::is_initialized() as c_int
}

#[unsafe(no_mangle)]
pub extern "C" fn rsf_arena_new(game_mode: u32, seed: i64) -> *mut FfiArena {
    let config = ArenaConfig {
        rng_seed: (seed >= 0).then_some(seed as u64),
        ..ArenaConfig::new(game_mode_from(game_mode))
    };
    Box::into_raw(Box::new(FfiArena {
        arena: Arena::new_with_config(config),
        events: Vec::with_capacity(64),
    }))
}

/// # Safety
/// `arena` must be a live pointer from rsf_arena_new; not used after this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rsf_arena_free(arena: *mut FfiArena) {
    if !arena.is_null() {
        drop(unsafe { Box::from_raw(arena) });
    }
}

unsafe fn arena<'a>(p: *mut FfiArena) -> &'a mut FfiArena {
    unsafe { &mut *p }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rsf_arena_add_car(p: *mut FfiArena, team: u32) -> u32 {
    let team = if team == 0 { Team::Blue } else { Team::Orange };
    unsafe { arena(p) }.arena.add_car(team, CarBodyConfig::OCTANE) as u32
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rsf_arena_num_cars(p: *mut FfiArena) -> u32 {
    unsafe { arena(p) }.arena.num_cars() as u32
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rsf_arena_step(p: *mut FfiArena, ticks: u32) {
    // ticks == 0 must be an exact no-op (v2's Step(0) semantics: EnvSet calls
    // Step(actionDelay) which is 0 under tickSkip-8/delay-0 configs).
    let fa = unsafe { arena(p) };
    fa.events.clear();
    for _ in 0..ticks {
        let tick = fa.arena.tick_count();
        for ev in fa.arena.step_tick() {
            fa.events.push(convert_event(ev, tick));
        }
    }
}

fn convert_event(ev: &ArenaEvent, tick: u64) -> RsfEvent {
    let mut out = RsfEvent {
        kind: 0,
        a: 0,
        b: 0,
        is_demo: 0,
        tick,
        pos: RsfVec3::default(),
        extra: RsfVec3::default(),
    };
    match ev {
        ArenaEvent::CarHitBall(e) => {
            out.kind = RSF_EVENT_CAR_HIT_BALL;
            out.a = e.car_idx as u32;
            out.pos = e.contact_point.into();
            out.extra = e.extra_hit_vel.into();
        }
        ArenaEvent::CarHitCar(e) => {
            out.kind = RSF_EVENT_CAR_HIT_CAR;
            out.a = e.bumper_car_idx as u32;
            out.b = e.victim_car_idx as u32;
            out.is_demo = e.is_demo as u8;
            out.pos = e.contact_point.into();
        }
        ArenaEvent::CarPickupBoost(e) => {
            out.kind = RSF_EVENT_CAR_PICKUP_BOOST;
            out.a = e.car_idx as u32;
            out.b = e.boost_pad_idx as u32;
        }
        ArenaEvent::BallHitWorld(e) => {
            out.kind = RSF_EVENT_BALL_HIT_WORLD;
            out.pos = e.contact_point.into();
            out.extra = e.contact_normal.into();
        }
        ArenaEvent::CarHitWorld(e) => {
            out.kind = RSF_EVENT_CAR_HIT_WORLD;
            out.a = e.car_idx as u32;
            out.pos = e.contact_point.into();
            out.extra = e.contact_normal.into();
        }
    }
    out
}

/// Returns number of events written (<= cap); total available regardless of cap
/// is written to *total if non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rsf_arena_get_events(
    p: *mut FfiArena,
    out: *mut RsfEvent,
    cap: u32,
    total: *mut u32,
) -> u32 {
    let fa = unsafe { arena(p) };
    if !total.is_null() {
        unsafe { *total = fa.events.len() as u32 };
    }
    let n = fa.events.len().min(cap as usize);
    if !out.is_null() {
        unsafe { std::ptr::copy_nonoverlapping(fa.events.as_ptr(), out, n) };
    }
    n as u32
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rsf_arena_get_car_state(
    p: *mut FfiArena,
    idx: u32,
    out: *mut RsfCarState,
) {
    let s = unsafe { arena(p) }.arena.get_car_state(idx as usize);
    unsafe { *out = s.into() };
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rsf_arena_set_car_state(
    p: *mut FfiArena,
    idx: u32,
    s: *const RsfCarState,
) {
    let state: CarState = unsafe { &*s }.into();
    unsafe { arena(p) }.arena.set_car_state(idx as usize, state);
}

// -- Exact-restore extras ----------------------------------------------------
//
// RsfCarState covers everything CarState expresses, but a car also carries raycast
// suspension state and a pending bump impulse that CarState cannot represent. Without
// those, set_car_state produces a restore that looks right and then diverges (measured
// ~23uu after one second of identical inputs).
//
// This pair is deliberately OPAQUE: C++ asks for the size and moves that many bytes
// around without naming a single field. That breaks the hand-mirroring contract in the
// best way — CarExtraState can gain fields with no C++ header change and no chance of
// the two drifting apart, which is exactly the failure mode that produced the bug.

#[unsafe(no_mangle)]
pub extern "C" fn rsf_car_extra_state_size() -> u32 {
    size_of::<CarExtraState>() as u32
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rsf_arena_get_car_extra_state(
    p: *mut FfiArena,
    idx: u32,
    out: *mut std::ffi::c_void,
) {
    let s = unsafe { arena(p) }.arena.get_car_extra_state(idx as usize);
    unsafe { std::ptr::write_unaligned(out.cast::<CarExtraState>(), s) };
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rsf_arena_set_car_extra_state(
    p: *mut FfiArena,
    idx: u32,
    s: *const std::ffi::c_void,
) {
    let state = unsafe { std::ptr::read_unaligned(s.cast::<CarExtraState>()) };
    unsafe { arena(p) }
        .arena
        .set_car_extra_state(idx as usize, &state);
}

// The arena RNG. Only demo respawns draw from it (Car::respawn picks a spawn location),
// but that is enough to make a rewind across a demo land the victim somewhere else.
// fastrand's WyRand state is a single u64, so the seed IS the state.

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rsf_arena_get_rng_state(p: *mut FfiArena) -> u64 {
    unsafe { arena(p) }.arena.rng.get_seed()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rsf_arena_set_rng_state(p: *mut FfiArena, state: u64) {
    unsafe { arena(p) }.arena.rng.seed(state);
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rsf_arena_set_car_controls(
    p: *mut FfiArena,
    idx: u32,
    c: *const RsfCarControls,
) {
    let controls: CarControls = unsafe { *c }.into();
    unsafe { arena(p) }.arena.set_car_controls(idx as usize, controls);
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rsf_arena_respawn_car(p: *mut FfiArena, idx: u32) {
    unsafe { arena(p) }.arena.respawn_car(idx as usize);
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rsf_arena_get_ball_state(p: *mut FfiArena, out: *mut RsfBallState) {
    let s = unsafe { arena(p) }.arena.get_ball_state();
    unsafe {
        *out = RsfBallState {
            phys: RsfPhysState {
                pos: s.phys.pos.into(),
                rot_mat: s.phys.rot_mat.into(),
                vel: s.phys.vel.into(),
                ang_vel: s.phys.ang_vel.into(),
            },
        }
    };
}

/// Patches only the physical state; heatseeker/dropshot sub-state is preserved.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rsf_arena_set_ball_state(p: *mut FfiArena, s: *const RsfBallState) {
    let fa = unsafe { arena(p) };
    let src = unsafe { &*s };
    let mut bs: BallState = *fa.arena.get_ball_state();
    bs.phys.pos = src.phys.pos.into();
    bs.phys.rot_mat = src.phys.rot_mat.into();
    bs.phys.vel = src.phys.vel.into();
    bs.phys.ang_vel = src.phys.ang_vel.into();
    fa.arena.set_ball_state(bs);
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rsf_arena_reset_kickoff(p: *mut FfiArena, seed: i64) {
    unsafe { arena(p) }
        .arena
        .reset_to_random_kickoff((seed >= 0).then_some(seed as u64));
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rsf_arena_tick_count(p: *mut FfiArena) -> u64 {
    unsafe { arena(p) }.arena.tick_count()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rsf_arena_is_ball_scored(p: *mut FfiArena) -> c_int {
    unsafe { arena(p) }.arena.is_ball_scored() as c_int
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rsf_arena_num_pads(p: *mut FfiArena) -> u32 {
    unsafe { arena(p) }.arena.num_boost_pads() as u32
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rsf_arena_get_pad_config(
    p: *mut FfiArena,
    idx: u32,
    out: *mut RsfPadConfig,
) {
    let cfg = unsafe { arena(p) }.arena.get_boost_pad_config(idx as usize);
    unsafe {
        *out = RsfPadConfig {
            pos: cfg.pos.into(),
            is_big: cfg.is_big as u8,
        }
    };
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rsf_arena_get_pad_state(
    p: *mut FfiArena,
    idx: u32,
    out: *mut RsfPadState,
) {
    let s = unsafe { arena(p) }.arena.get_boost_pad_state(idx as usize);
    unsafe {
        *out = RsfPadState {
            cooldown: s.cooldown,
            is_active: s.is_active() as u8,
        }
    };
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rsf_arena_set_pad_state(
    p: *mut FfiArena,
    idx: u32,
    s: *const RsfPadState,
) {
    let src = unsafe { &*s };
    unsafe { arena(p) }
        .arena
        .set_boost_pad_state(idx as usize, BoostPadState { cooldown: src.cooldown });
}
