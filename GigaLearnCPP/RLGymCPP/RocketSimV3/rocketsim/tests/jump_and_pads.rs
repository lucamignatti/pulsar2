//! Scenario tests for jump lift-off timing and boost-pad pickup semantics
//! (SIM2REAL_AUDIT.md S42/S44b/S45c/S47).
//!
//! Several of these are TRIPWIRES rather than proofs: they pin behaviour whose
//! real-game value is still open, so that changing it has to be deliberate. Each
//! such test says so, and names the experiment that would settle it.

use glam::{Mat3A, Vec3A};
use rocketsim::{Arena, CarBodyConfig, CarControls, GameMode, Team};

const MESHES: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../../../build/collision_meshes"
);

fn setup() -> Arena {
    rocketsim::init(MESHES, true).ok(); // idempotent across tests
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
fn minimum_jump_from_rest_liftoff_tick() {
    // TRIPWIRE (S47). RocketSim v2 was reported to leave the ground one tick EARLY
    // versus RLBot on a minimum jump from rest. The 120Hz match capture cannot settle
    // it -- strong bots are essentially never stationary, so it holds exactly one
    // near-rest jump (which matched at offset 0) -- and over 126 grounded jumps of all
    // speeds the sim-vs-real lift-off offset is modal at exactly 0, so there is no
    // systematic early lift-off to fix. This pins the current value; the real number
    // comes from the dormant `jump_min_from_rest` battery segment once a fresh
    // real-game capture includes it.
    let three = jump_liftoff_tick(3);
    assert_eq!(
        three, 5,
        "minimum (3-tick) jump from rest changed lift-off tick: {three}"
    );
}

#[test]
fn one_tick_tap_matches_minimum_jump() {
    // jump::MIN_TIME = 0.025 s = exactly 3 ticks at 120 Hz, so a 1-tick tap must be
    // indistinguishable from a 3-tick hold: the hold force keeps firing until MIN_TIME
    // regardless of the input. If these ever diverge, the MIN_TIME floor broke.
    assert_eq!(
        jump_liftoff_tick(1),
        jump_liftoff_tick(3),
        "a 1-tick tap must lift off identically to a 3-tick hold (MIN_TIME floor)"
    );
}

#[test]
fn no_jump_impulse_on_the_release_tick() {
    // S45c: the hold force used to be applied BEFORE deciding whether the jump was
    // still held, so the release tick got one extra ~12 uu/s impulse. A minimum jump
    // must therefore carry exactly MIN_TIME worth of hold force -- no more. Measured
    // as apex height, which is what the extra impulse inflated.
    let mut arena = setup();
    let car = arena.add_car(Team::Blue, CarBodyConfig::OCTANE);
    park(&mut arena, car, Vec3A::new(0.0, -2000.0, 17.0), 100.0);
    settle(&mut arena, car, 30);

    let mut apex: f32 = 0.0;
    for k in 0..120 {
        let mut c = CarControls::default();
        c.jump = k < 3;
        arena.set_car_controls(car, c);
        arena.step_tick();
        apex = apex.max(arena.get_car_state(car).phys.pos.z);
    }
    // Both values MEASURED, not guessed: the shipped build apexes at 94.55, the
    // pre-S45c build (hold force applied before deciding) at 100.29. The tolerance
    // pins the former and excludes the latter, so re-adding a release-tick impulse
    // fails here.
    assert!(
        (apex - 94.55).abs() < 1.5,
        "minimum-jump apex moved: {apex:.2} (expected ~94.55; pre-S45c regression reads ~100.29)"
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
fn full_boost_car_does_not_consume_the_pad() {
    // TRIPWIRE, and the one genuinely OPEN question in the pad model (S45/S45b).
    // We skip the pad entirely at max boost (v2 behaviour); the "tuned" fork consumes
    // it and denies the boost, claiming real RL does the same. The 120Hz capture cannot
    // decide it: a full-boost pickup produces NO boost increase, so it is invisible
    // except through a LATER denied pickup, and the capture holds only 2 full-boost pad
    // crossings, neither followed by a discriminating pickup.
    //
    // The decisive experiment is the `pad_full_boost_deny` battery segment: cross a big
    // pad at 100 boost, burn some boost, re-cross the SAME pad inside its 10 s cooldown.
    // If the second crossing grants +100 the pad was never consumed (this model); if it
    // grants nothing, RL consumes pads at full and this test should be inverted.
    let mut arena = setup();
    let car = arena.add_car(Team::Blue, CarBodyConfig::OCTANE);
    let (pad_idx, pad_pos) = nearest_pad(&arena, Vec3A::new(0.0, -1000.0, 0.0), false);
    park(&mut arena, car, Vec3A::new(pad_pos.x, pad_pos.y, 17.0), 100.0);

    for _ in 0..10 {
        arena.step_tick();
    }
    assert_eq!(
        arena.get_boost_pad_state(pad_idx).cooldown,
        0.0,
        "a car sitting on a pad at MAX boost must not put it on cooldown (v2 model)"
    );

    // ...and the pad is still there to be taken the moment the tank is not full.
    let mut cs = *arena.get_car_state(car);
    cs.boost = 50.0;
    arena.set_car_state(car, cs);
    for _ in 0..4 {
        arena.step_tick();
    }
    assert!(
        arena.get_car_state(car).boost > 50.0,
        "an un-consumed pad must still grant once the car drops below max"
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
