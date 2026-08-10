//! Whole-recording error analysis for real-game `.rlpr` captures (RLRecord2 /
//! BakkesMod). Unlike the pass/fail fixtures, this never panics: it replays every
//! clean transition, aggregates per-regime error statistics, and prints the worst
//! offenders for manual attribution. Run with:
//!
//!   RLPR_PATH=/path/to/capture.rlpr cargo test --release --test mod analyze_rlpr -- --nocapture
//!
//! Control pairing: BakkesMod-semantics files (detected by the zeroed hitbox in the
//! header, per rlpr_format.h) store the inputs consumed DURING frame F in record F
//! itself; sim-generated fixtures store them in record F+1's `prev_controls`.

use glam::Vec3A;
use rocketsim::{Arena, CarBodyConfig, CarControls, GameMode, Team};

use super::recording::Recording;
use super::set_state_to_record_tick;

#[derive(Default, Clone)]
struct Stat {
    vals: Vec<f32>,
}

impl Stat {
    fn push(&mut self, v: f32) {
        self.vals.push(v);
    }
    fn summary(&mut self) -> String {
        if self.vals.is_empty() {
            return "n=0".into();
        }
        self.vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = self.vals.len();
        let q = |p: f32| self.vals[((n as f32 * p) as usize).min(n - 1)];
        format!(
            "n={n:6}  p50={:8.3}  p90={:8.3}  p99={:8.3}  max={:8.3}",
            q(0.5),
            q(0.9),
            q(0.99),
            self.vals[n - 1]
        )
    }
}

#[test]
fn analyze_rlpr() {
    let Ok(path) = std::env::var("RLPR_PATH") else {
        eprintln!("analyze_rlpr: RLPR_PATH not set, skipping");
        return;
    };
    if !rocketsim::is_initialized() {
        rocketsim::init_from_default(true).unwrap();
    }
    let bytes = std::fs::read(&path).expect("read RLPR_PATH");
    let recording = Recording::from_bytes(&path, &bytes).expect("parse rlpr");

    let hb = &recording.info.hitbox_rel_min_bt;
    let bakkes_semantics = hb.x == 0.0 && hb.y == 0.0 && hb.z == 0.0;
    let num_cars = recording.info.num_cars as usize;
    eprintln!(
        "recording: {} ticks, {num_cars} cars, {} semantics",
        recording.ticks.len(),
        if bakkes_semantics { "BakkesMod" } else { "sim-fixture" }
    );

    let mut arena = Arena::new(GameMode::Soccar);
    let car_idcs: Vec<usize> = (0..num_cars)
        .map(|i| {
            let team = if i % 2 == 0 { Team::Blue } else { Team::Orange };
            arena.add_car(team, CarBodyConfig::OCTANE)
        })
        .collect();

    // regime name -> (vel_err, pos_err, ang_vel_err) stats
    let mut regimes: std::collections::BTreeMap<&'static str, [Stat; 3]> =
        std::collections::BTreeMap::new();
    // (vel_err, tick_index, car, regime) worst list
    let mut worst: Vec<(f32, usize, usize, &'static str)> = Vec::new();
    let mut skipped_gaps = 0usize;

    for i in 0..recording.ticks.len() - 1 {
        let from_tick = &recording.ticks[i];
        let to_tick = &recording.ticks[i + 1];

        // Skip transitions across recorder gaps (game paused / frames missed).
        let clean = (0..num_cars).all(|c| {
            to_tick.car_records[c].phys.physics_frame
                == from_tick.car_records[c].phys.physics_frame + 1
        });
        if !clean {
            skipped_gaps += 1;
            continue;
        }

        let controls: Vec<CarControls> = (0..num_cars)
            .map(|c| {
                if bakkes_semantics {
                    from_tick.car_records[c].prev_controls.into()
                } else {
                    to_tick.car_records[c].prev_controls.into()
                }
            })
            .collect();

        set_state_to_record_tick(&mut arena, &car_idcs, from_tick, &controls);
        // Boost amount gates whether a boost input actually fires; the shared
        // state-setter does not carry it.
        for (c, &car_idx) in car_idcs.iter().enumerate() {
            let mut cs = *arena.get_car_state(car_idx);
            cs.boost = from_tick.car_records[c].boost_amount;
            arena.set_car_state(car_idx, cs);
        }

        arena.step_tick();

        for (c, &car_idx) in car_idcs.iter().enumerate() {
            let pred = arena.get_car_state(car_idx);
            let real = &to_tick.car_records[c];
            let real_pos: Vec3A = real.phys.pos.into();
            let real_vel: Vec3A = real.phys.lin_vel.into();
            let real_ang: Vec3A = real.phys.ang_vel.into();

            let pos_err = (pred.phys.pos - real_pos).length();
            let vel_err = (pred.phys.vel - real_vel).length();
            let ang_err = (pred.phys.ang_vel - real_ang).length();

            let fr = &from_tick.car_records[c];
            let wheels_touching = fr.wheels.iter().filter(|w| w.has_contact).count();
            let on_wall = fr.phys.has_world_contact
                && Vec3A::from(fr.phys.world_contact_normal).z.abs() < 0.7;
            let regime: &'static str = if fr.is_flipping {
                if wheels_touching > 0 {
                    "flip+wheels"
                } else {
                    "flip_air"
                }
            } else if fr.is_on_ground {
                if on_wall {
                    "wall_drive"
                } else if fr.prev_controls.boost {
                    "ground_boost"
                } else {
                    "ground"
                }
            } else if wheels_touching > 0 {
                "air+wheels"
            } else if real.is_touching_ball || fr.is_touching_ball {
                "ball_contact"
            } else if fr.prev_controls.boost {
                "air_boost"
            } else {
                "air_free"
            };

            let e = regimes.entry(regime).or_default();
            e[0].push(vel_err);
            e[1].push(pos_err);
            e[2].push(ang_err);
            worst.push((vel_err, i, c, regime));
        }
    }

    eprintln!("skipped {skipped_gaps} gap transitions");
    eprintln!("\n=== per-regime single-tick VELOCITY error (uu/s) ===");
    for (name, stats) in &mut regimes {
        eprintln!("{name:14} {}", stats[0].summary());
    }
    eprintln!("\n=== per-regime single-tick ANG VEL error (rad/s) ===");
    for (name, stats) in &mut regimes {
        eprintln!("{name:14} {}", stats[2].summary());
    }

    worst.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    eprintln!("\n=== 25 worst transitions by velocity error ===");
    for (err, i, c, regime) in worst.iter().take(25) {
        let fr = &recording.ticks[*i].car_records[*c];
        let pos: Vec3A = fr.phys.pos.into();
        let vel: Vec3A = fr.phys.lin_vel.into();
        eprintln!(
            "  tick {i:6} car {c} {regime:12} verr={err:8.2}  pos=({:7.0},{:7.0},{:6.0}) |v|={:6.0} ctl[j{} b{} t{:+.1} p{:+.1}]",
            pos.x,
            pos.y,
            pos.z,
            vel.length(),
            u8::from(fr.prev_controls.jump),
            u8::from(fr.prev_controls.boost),
            fr.prev_controls.throttle,
            fr.prev_controls.pitch,
        );
    }
}
