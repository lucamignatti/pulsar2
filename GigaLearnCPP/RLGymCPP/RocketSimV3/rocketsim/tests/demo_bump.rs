//! Scenario tests for the car-car contact pipeline rebuilt from the decompiled
//! Car_TA::ApplyCarImpactForces / ShouldDemolish / GetBumpImpulse
//! (SIM2REAL_AUDIT.md S36). Each scenario builds a fresh Soccar arena, stages two
//! cars, steps until they touch, and asserts on the resulting events/impulses.

use glam::{Mat3A, Vec3A};
use rocketsim::{Arena, ArenaEvent, CarBodyConfig, GameMode, Team};

const MESHES: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../../../build/collision_meshes"
);

fn setup() -> (Arena, usize, usize) {
    rocketsim::init(MESHES, true).ok(); // idempotent across tests
    let mut arena = Arena::new(GameMode::Soccar);
    let attacker = arena.add_car(Team::Blue, CarBodyConfig::OCTANE);
    let victim = arena.add_car(Team::Orange, CarBodyConfig::OCTANE);
    // Park the ball far away so it can't interfere.
    let mut bs = *arena.get_ball_state();
    bs.phys.pos = Vec3A::new(-3500.0, 4800.0, 93.0);
    bs.phys.vel = Vec3A::ZERO;
    arena.set_ball_state(bs);
    (arena, attacker, victim)
}

fn yaw_mat(yaw: f32) -> Mat3A {
    Mat3A::from_cols(
        Vec3A::new(yaw.cos(), yaw.sin(), 0.0),
        Vec3A::new(-yaw.sin(), yaw.cos(), 0.0),
        Vec3A::Z,
    )
}

/// Stage attacker/victim, run until first CarHitCar event (or `max_ticks`).
/// Returns (was_demo, victim_vel_before_hit, victim_vel_after_hit).
fn run_contact(
    arena: &mut Arena,
    attacker: usize,
    victim: usize,
    attacker_yaw: f32,
    attacker_vel: Vec3A,
    victim_pos: Vec3A,
    max_ticks: usize,
) -> Option<(bool, Vec3A, Vec3A)> {
    let mut asrc = *arena.get_car_state(attacker);
    asrc.phys.pos = Vec3A::new(0.0, 0.0, 17.0);
    asrc.phys.rot_mat = yaw_mat(attacker_yaw);
    asrc.phys.vel = attacker_vel;
    asrc.phys.ang_vel = Vec3A::ZERO;
    asrc.is_on_ground = true;
    arena.set_car_state(attacker, asrc);

    let mut vs = *arena.get_car_state(victim);
    vs.phys.pos = victim_pos;
    vs.phys.rot_mat = Mat3A::IDENTITY;
    vs.phys.vel = Vec3A::ZERO;
    vs.phys.ang_vel = Vec3A::ZERO;
    vs.is_on_ground = victim_pos.z < 30.0;
    arena.set_car_state(victim, vs);

    for _ in 0..max_ticks {
        let pre_vel = arena.get_car_state(victim).phys.vel;
        let events: Vec<ArenaEvent> = arena.step_tick().to_vec();
        for ev in events {
            if let ArenaEvent::CarHitCar(e) = ev
                && e.bumper_car_idx == attacker
                && e.victim_car_idx == victim
            {
                let post_vel = arena.get_car_state(victim).phys.vel;
                return Some((e.is_demo, pre_vel, post_vel));
            }
        }
    }
    None
}

#[test]
fn supersonic_head_on_demos() {
    let (mut arena, a, v) = setup();
    let hit = run_contact(
        &mut arena,
        a,
        v,
        0.0,
        Vec3A::new(2300.0, 0.0, 0.0),
        Vec3A::new(400.0, 0.0, 17.0),
        60,
    )
    .expect("cars never touched");
    assert!(hit.0, "supersonic head-on front hit must demolish");
}

#[test]
fn supersonic_sideways_slide_does_not_demo() {
    let (mut arena, a, v) = setup();
    // Facing +y, sliding +x at supersonic speed: forward-projected speed ~0.
    let hit = run_contact(
        &mut arena,
        a,
        v,
        std::f32::consts::FRAC_PI_2,
        Vec3A::new(2300.0, 0.0, 0.0),
        Vec3A::new(400.0, 0.0, 17.0),
        60,
    );
    if let Some((is_demo, _, _)) = hit {
        assert!(!is_demo, "sideways-sliding supersonic car must NOT demolish");
    }
    assert!(
        !arena.get_car_state(v).is_demoed,
        "victim demoed by a sideways slide"
    );
}

#[test]
fn supersonic_reversing_rear_hit_demos() {
    let (mut arena, a, v) = setup();
    // Facing -x (yaw pi), moving +x: rear-first supersonic contact.
    let hit = run_contact(
        &mut arena,
        a,
        v,
        std::f32::consts::PI,
        Vec3A::new(2300.0, 0.0, 0.0),
        Vec3A::new(400.0, 0.0, 17.0),
        60,
    )
    .expect("cars never touched");
    assert!(
        hit.0,
        "bAllowBackwardsDemolitions: reversing supersonic rear hit must demolish"
    );
}

#[test]
fn ground_bump_pushes_up_air_bump_does_not() {
    // Grounded victim: impulse includes a push along the victim's up axis.
    let (mut arena, a, v) = setup();
    let (is_demo, pre, post) = run_contact(
        &mut arena,
        a,
        v,
        0.0,
        Vec3A::new(1500.0, 0.0, 0.0),
        Vec3A::new(400.0, 0.0, 17.0),
        60,
    )
    .expect("ground bump never landed");
    assert!(!is_demo, "1500 uu/s must not demo");
    let dvz_ground = post.z - pre.z;

    // Airborne victim: no vertical component from the script impulse. The victim
    // falls before contact, so aim the attacker where the victim will be and give
    // the drop only a few ticks.
    let (mut arena, a, v) = setup();
    let (is_demo, pre, post) = run_contact(
        &mut arena,
        a,
        v,
        0.0,
        Vec3A::new(2000.0, 0.0, 0.0),
        Vec3A::new(300.0, 0.0, 45.0),
        30,
    )
    .expect("air bump never landed");
    assert!(!is_demo);
    let dvz_air = post.z - pre.z;

    // The grounded bump gets the scripted up-push; the airborne one only whatever
    // the rigid-body contact itself produced. Grounded must be clearly larger.
    assert!(
        dvz_ground > 100.0,
        "grounded bump up-push missing: dvz={dvz_ground}"
    );
    assert!(
        dvz_air < dvz_ground * 0.5,
        "airborne victim should get no scripted up-push: air {dvz_air} vs ground {dvz_ground}"
    );
}

#[test]
fn mutual_supersonic_head_on_demos_both() {
    let (mut arena, a, v) = setup();

    let mut asrc = *arena.get_car_state(a);
    asrc.phys.pos = Vec3A::new(0.0, 0.0, 17.0);
    asrc.phys.rot_mat = yaw_mat(0.0);
    asrc.phys.vel = Vec3A::new(2300.0, 0.0, 0.0);
    asrc.is_on_ground = true;
    arena.set_car_state(a, asrc);

    let mut vs = *arena.get_car_state(v);
    vs.phys.pos = Vec3A::new(600.0, 0.0, 17.0);
    vs.phys.rot_mat = yaw_mat(std::f32::consts::PI);
    vs.phys.vel = Vec3A::new(-2300.0, 0.0, 0.0);
    vs.is_on_ground = true;
    arena.set_car_state(v, vs);

    for _ in 0..60 {
        arena.step_tick();
        if arena.get_car_state(a).is_demoed || arena.get_car_state(v).is_demoed {
            break;
        }
    }
    assert!(
        arena.get_car_state(a).is_demoed && arena.get_car_state(v).is_demoed,
        "mutual supersonic head-on must demolish BOTH cars (deferred actions): a={} v={}",
        arena.get_car_state(a).is_demoed,
        arena.get_car_state(v).is_demoed
    );
}

#[test]
fn quantized_rest_height_response() {
    // The 120 Hz capture stores z quantized to 0.01 uu (bQuantizePhysics): a resting
    // car records 17.01 while the true equilibrium is ~17.012. Restoring at the
    // quantized height starts the suspension compressed by ~2 milli-uu; this test
    // measures the one-tick vertical response, i.e. the REPLAY noise floor that
    // quantization imposes on all ground-regime error stats (S38).
    let (mut arena, a, _v) = setup();
    for (z, label) in [(17.0121f32, "equilibrium"), (17.01, "quantized")] {
        let mut cs = *arena.get_car_state(a);
        cs.phys.pos = Vec3A::new(0.0, 0.0, z);
        cs.phys.rot_mat = Mat3A::IDENTITY;
        cs.phys.vel = Vec3A::ZERO;
        cs.phys.ang_vel = Vec3A::ZERO;
        cs.is_on_ground = true;
        arena.set_car_state(a, cs);
        arena.step_tick();
        let vz = arena.get_car_state(a).phys.vel.z;
        eprintln!("restore z={z:.4} ({label}) -> one-tick vz = {vz:+.3}");
    }
}

#[test]
fn supersonic_grace_counts_from_band_entry_not_first_start() {
    // RL semantics (CDO: TurnoffSpeedBuffer=100, TurnoffTime=1): the 1s grace applies
    // whenever speed sits in [2100, 2200), timed from ENTERING the band -- not from
    // when supersonic first started. Upstream C++ v2 accumulates supersonicTime while
    // above 2200 too, so a car supersonic for >1s loses its grace band entirely; this
    // test fails under those semantics.
    let (mut arena, a, _v) = setup();
    let mut set_speed = |arena: &mut Arena, v: f32| {
        let mut cs = *arena.get_car_state(a);
        cs.phys.pos = Vec3A::new(0.0, 0.0, 17.0);
        cs.phys.rot_mat = Mat3A::IDENTITY;
        cs.phys.vel = Vec3A::new(v, 0.0, 0.0);
        cs.phys.ang_vel = Vec3A::ZERO;
        cs.is_on_ground = true;
        arena.set_car_state(a, cs);
    };

    // 2 seconds continuously above start speed (240 ticks).
    for _ in 0..240 {
        set_speed(&mut arena, 2250.0);
        arena.step_tick();
    }
    assert!(arena.get_car_state(a).is_supersonic, "supersonic while at 2250");

    // Drop into the maintain band: must KEEP supersonic for ~1s from band entry.
    for _ in 0..100 {
        set_speed(&mut arena, 2150.0);
        arena.step_tick();
    }
    assert!(
        arena.get_car_state(a).is_supersonic,
        "grace must time from band entry (v2-flaw semantics would have dropped it)"
    );

    // Briefly back above 2200 resets the grace...
    for _ in 0..5 {
        set_speed(&mut arena, 2250.0);
        arena.step_tick();
    }
    // ...so another ~0.9s in the band still holds.
    for _ in 0..110 {
        set_speed(&mut arena, 2150.0);
        arena.step_tick();
    }
    assert!(
        arena.get_car_state(a).is_supersonic,
        "grace must RESET on re-exceeding start speed"
    );

    // Staying in the band past 1s total loses it.
    for _ in 0..15 {
        set_speed(&mut arena, 2150.0);
        arena.step_tick();
    }
    assert!(
        !arena.get_car_state(a).is_supersonic,
        "grace must expire after 1s continuously in the band"
    );

    // And below the maintain floor it drops instantly.
    for _ in 0..240 {
        set_speed(&mut arena, 2250.0);
        arena.step_tick();
    }
    set_speed(&mut arena, 2050.0);
    arena.step_tick();
    assert!(
        !arena.get_car_state(a).is_supersonic,
        "below 2100 supersonic must drop immediately"
    );
}
