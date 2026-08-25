//! Scenario tests for jump and boost-pad semantics (SIM2REAL_AUDIT.md S42/S44b/S48).
//!
//! RULE (user-set 2026-08-13, binding): **every assertion here must trace to an
//! EXTERNAL truth source** -- a real-game capture, the maneuver battery, the
//! decompiled game code, or a CDO constant. Never to this sim's own output.
//!
//! A test that asserts "the sim does what the sim does" cannot detect that we are
//! wrong about Rocket League, only that someone changed our answer -- and it is
//! actively harmful, because pinning an unverified value makes the eventual CORRECT
//! fix look like a regression. Three such tests were written and deleted the same
//! day; what they were guarding (minimum-jump lift-off tick, full-boost pad
//! consumption) is genuinely unknown, and lives as DORMANT battery segments
//! (`jump_min_from_rest`, `pad_full_boost_deny`) awaiting a real-game recording.

use glam::{Mat3A, Vec3A};
use rocketsim::{Arena, CarBodyConfig, CarControls, GameMode, Team};

const MESHES: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../../../build/collision_meshes"
);

fn setup() -> Arena {
    let mut paths = vec![std::path::PathBuf::from(MESHES)];
    if let Ok(p) = std::env::var("RLPR_MESHES") {
        paths.insert(0, std::path::PathBuf::from(p));
    }
    if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
        let root = std::path::PathBuf::from(manifest);
        paths.push(root.join("collision_meshes"));
        paths.push(root.join("../../../../collision_meshes"));
        paths.push(root.join("../../../../build/collision_meshes"));
        paths.push(root.join("../../../../build-npl/collision_meshes"));
    }
    let mut inited = false;
    for p in &paths {
        if p.is_dir() && rocketsim::init(p, true).is_ok() {
            inited = true;
            break;
        }
    }
    assert!(inited, "no collision_meshes found; tried {paths:?}");
    let mut arena = Arena::new(GameMode::Soccar);
    // Park the ball far away so it can never interfere.
    let mut bs = *arena.get_ball_state();
    bs.phys.pos = Vec3A::new(-3500.0, 4800.0, 93.0);
    bs.phys.vel = Vec3A::ZERO;
    arena.set_ball_state(bs);
    arena
}

fn park(arena: &mut Arena, car: usize, pos: Vec3A, boost: f32) {
    let mut cs = *arena.get_car_state(car);
    cs.phys.pos = pos;
    cs.phys.rot_mat = Mat3A::IDENTITY;
    cs.phys.vel = Vec3A::ZERO;
    cs.phys.ang_vel = Vec3A::ZERO;
    cs.is_on_ground = true;
    cs.boost = boost;
    arena.set_car_state(car, cs);
}

/// Settle on the ground so wheel contacts and jump flags are physically real
/// (poking `is_on_ground` does not ground the car -- see SIM2REAL_AUDIT S12).
fn settle(arena: &mut Arena, car: usize, ticks: usize) {
    arena.set_car_controls(car, CarControls::default());
    for _ in 0..ticks {
        arena.step_tick();
    }
}

fn airborne(arena: &Arena, car: usize) -> bool {
    arena
        .get_car_state(car)
        .wheels_with_contact
        .iter()
        .all(|c| !*c)
}

/// Run a jump of `hold` ticks from rest and return the tick index (relative to the
/// first tick the jump input is applied) at which every wheel loses contact.
fn jump_liftoff_tick(hold: usize) -> usize {
    let mut arena = setup();
    let car = arena.add_car(Team::Blue, CarBodyConfig::OCTANE);
    park(&mut arena, car, Vec3A::new(0.0, -2000.0, 17.0), 100.0);
    settle(&mut arena, car, 30);

    for k in 0..40 {
        let mut c = CarControls::default();
        c.jump = k < hold;
        arena.set_car_controls(car, c);
        arena.step_tick();
        if airborne(&arena, car) {
            return k;
        }
    }
    panic!("car never left the ground");
}

#[test]
fn one_tick_tap_matches_minimum_jump() {
    // SOURCE: jump::MIN_TIME, an RL constant. 0.025 s = exactly 3 ticks at 120 Hz, so a 1-tick tap must be
    // indistinguishable from a 3-tick hold: the hold force keeps firing until MIN_TIME
    // regardless of the input. If these ever diverge, the MIN_TIME floor broke.
    assert_eq!(
        jump_liftoff_tick(1),
        jump_liftoff_tick(3),
        "a 1-tick tap must lift off identically to a 3-tick hold (MIN_TIME floor)"
    );
}

/// Index of the nearest boost pad to `pos`, plus its config.
fn nearest_pad(arena: &Arena, pos: Vec3A, big: bool) -> (usize, Vec3A) {
    let cfgs = arena.get_all_boost_pad_configs();
    let (mut best, mut best_d) = (usize::MAX, f32::INFINITY);
    for (i, c) in cfgs.iter().enumerate() {
        if c.is_big != big {
            continue;
        }
        let d = c.pos.distance_squared(pos);
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    (best, cfgs[best].pos)
}

#[test]
fn pad_grant_lands_two_ticks_after_overlap() {
    // S42: Rocket League routes pickups through the UE3 touch-event pipeline, so the
    // boost (and the pad cooldown) land ~2 ticks after first overlap, not at overlap.
    // Measured on the capture: exact-tick pickup matches 14 -> 55.
    let mut arena = setup();
    let car = arena.add_car(Team::Blue, CarBodyConfig::OCTANE);
    let (pad_idx, pad_pos) = nearest_pad(&arena, Vec3A::new(0.0, -1000.0, 0.0), false);
    park(&mut arena, car, Vec3A::new(pad_pos.x, pad_pos.y, 17.0), 0.0);
    settle(&mut arena, car, 1);

    let mut granted_at = None;
    for k in 0..8 {
        arena.step_tick();
        if arena.get_car_state(car).boost > 0.0 && granted_at.is_none() {
            granted_at = Some(k);
        }
    }
    assert_eq!(
        granted_at,
        Some(1),
        "pad grant must land GRANT_DELAY_TICKS after the overlap tick"
    );
    assert!(
        arena.get_boost_pad_state(pad_idx).cooldown > 0.0,
        "the pad must be on cooldown once it has granted"
    );
}

#[test]
fn pickup_box_is_centred_on_the_hitbox_not_the_origin() {
    // S44b: the trigger box is centred at the car's true collision centre
    // (hitbox_pos_offset: Octane +13.88 fwd, +20.75 up), not the rigid-body origin.
    // Two real big-pad pickups in the capture sit 13-17 uu outside the origin-centred
    // box but inside the offset one. Approach a big pad from behind so the offset moves
    // the box FORWARD onto the pad: the origin-centred test would miss this.
    let mut arena = setup();
    let car = arena.add_car(Team::Blue, CarBodyConfig::OCTANE);
    let (_, pad_pos) = nearest_pad(&arena, Vec3A::new(3584.0, 0.0, 0.0), true);

    // Facing +x (identity rotation), sat so the pad centre is just beyond the reach of
    // an origin-centred box (half-length ~59 + 160 radius) but inside the offset one.
    let gap = 59.0 + 160.0 + 8.0;
    park(
        &mut arena,
        car,
        Vec3A::new(pad_pos.x - gap, pad_pos.y, 17.0),
        0.0,
    );
    for _ in 0..6 {
        arena.step_tick();
    }
    assert!(
        arena.get_car_state(car).boost > 0.0,
        "the hitbox-offset trigger box must reach this pad; the origin-centred box did not"
    );
}

/// PR66 sink repro: hold a downward force on the ball. Unfixed special contacts
/// use contact radius as "penetration", so positional recovery never fires and
/// the ball goes through the floor. Floor rest height is ~93.15 uu.
#[test]
fn ball_floor_sink_under_sustained_force() {
    fn run(label: &str, start_z: f32, force_z: f32, ticks: u32) -> f32 {
        let mut arena = setup();
        let mut bs = *arena.get_ball_state();
        bs.phys.pos = Vec3A::new(0.0, 0.0, start_z);
        bs.phys.vel = Vec3A::ZERO;
        arena.set_ball_state(bs);
        let mut min_z = start_z;
        for _ in 0..ticks {
            let mut bs = *arena.get_ball_state();
            bs.phys.vel.z += force_z;
            arena.set_ball_state(bs);
            arena.step_tick();
            min_z = min_z.min(arena.get_ball_state().phys.pos.z);
        }
        let end_z = arena.get_ball_state().phys.pos.z;
        eprintln!(
            "SINK {label} pen_fix={} start={start_z} force={force_z} min_z={min_z:.2} end_z={end_z:.2}",
            std::env::var("GGL_BALL_PEN_FIX").unwrap_or_default()
        );
        min_z
    }
    run("rest+press", 93.15, -2000.0, 120);
    run("already-under", 40.0, -200.0, 120);
}
