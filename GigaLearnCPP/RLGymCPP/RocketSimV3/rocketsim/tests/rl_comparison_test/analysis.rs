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

trait XyLen {
    fn xy_len_sq(&self) -> f32;
}
impl XyLen for Vec3A {
    fn xy_len_sq(&self) -> f32 {
        self.x * self.x + self.y * self.y
    }
}

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

        // Skip transitions across recorder gaps (game paused / frames missed),
        // around demo respawns (short roster ticks: slot assignment is ambiguous),
        // and across TELEPORTS (goal resets / demolitions): no legal physics moves a
        // car more than ~|v_max| * dt + margin in one tick, so a larger real position
        // delta is a game event, not dynamics.
        let clean = from_tick.car_records.len() == num_cars
            && to_tick.car_records.len() == num_cars
            && (0..num_cars).all(|c| {
                to_tick.car_records[c].phys.physics_frame
                    == from_tick.car_records[c].phys.physics_frame + 1
            })
            && (0..num_cars).all(|c| {
                let fp: Vec3A = from_tick.car_records[c].phys.pos.into();
                let tp: Vec3A = to_tick.car_records[c].phys.pos.into();
                (tp - fp).length() < 40.0
            });
        if !clean {
            skipped_gaps += 1;
            continue;
        }

        // KICKOFF COUNTDOWN: after a goal reset RL runs physics but IGNORES all car
        // inputs for ~3 s while the ball sits frozen at the kickoff spot. Replaying
        // the recorded (spammed) inputs there manufactures phantom jumps/throttle.
        // Skip while the ball is frozen at origin and neither car is driving yet.
        {
            let bp: Vec3A = from_tick.ball_record.pos.into();
            let bv: Vec3A = from_tick.ball_record.lin_vel.into();
            let ball_frozen = bp.x.abs() < 1.0 && bp.y.abs() < 1.0 && bv.length_squared() == 0.0;
            let cars_staged = (0..num_cars).all(|c| {
                let v: Vec3A = from_tick.car_records[c].phys.lin_vel.into();
                v.xy_len_sq() < 50.0 * 50.0
            });
            if ball_frozen && cars_staged {
                skipped_gaps += 1;
                continue;
            }
        }

        // CONTROL PAIRING (measured, 2026-08-10): BakkesMod records are END-of-frame
        // state, and the input delivered via SetVehicleInput at frame R takes effect
        // in frame R+2 (verified on jump activations: press at 285 -> jump at 287,
        // press at 597 -> jump at 599). Simulating frame F+1 (record F -> F+1)
        // therefore uses the controls recorded at F-1. GGL_CTRL_LAG overrides for
        // A/B tests (0 = record F+1, 1 = record F, 2 = record F-1 [default]).
        let lag: usize = std::env::var("GGL_CTRL_LAG")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(if bakkes_semantics { 2 } else { 0 });
        if i + 1 < lag {
            continue;
        }
        let ctrl_tick = &recording.ticks[i + 1 - lag];
        if ctrl_tick.car_records.len() != num_cars {
            continue;
        }
        let controls: Vec<CarControls> = (0..num_cars)
            .map(|c| ctrl_tick.car_records[c].prev_controls.into())
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
            // Wall = WHEEL contact whose normal is closer to horizontal than vertical;
            // chassis world-contact alone misses ordinary wheel-on-wall driving.
            let on_wall = fr
                .wheels
                .iter()
                .any(|w| w.has_contact && Vec3A::from(w.contact_normal).z.abs() < 0.7);
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

            // GGL_DUMP_FLIP: print structure of high-ang-error flipping ticks.
            if std::env::var("GGL_DUMP_FLIP").is_ok()
                && fr.is_flipping
                && wheels_touching == 0
                && ang_err > 1.0
            {
                let pred_d = pred.phys.ang_vel - Vec3A::from(fr.phys.ang_vel);
                let real_d = real_ang - Vec3A::from(fr.phys.ang_vel);
                let frt: Vec3A = fr.flip_rel_torque.into();
                eprintln!(
                    "FLIP tick {i} car {c} ft={:.4} |av|={:.3} aerr={ang_err:.3} pred_d=({:+.3},{:+.3},{:+.3}) real_d=({:+.3},{:+.3},{:+.3}) frt=({:+.2},{:+.2},{:+.2}) pitch={:+.2} jump={}",
                    fr.flip_time,
                    Vec3A::from(fr.phys.ang_vel).length(),
                    pred_d.x, pred_d.y, pred_d.z,
                    real_d.x, real_d.y, real_d.z,
                    frt.x, frt.y, frt.z,
                    fr.prev_controls.pitch,
                    u8::from(fr.prev_controls.jump),
                );
            }

            // GGL_DUMP_LAND: landing/impact transitions -- any tick where a wheel is
            // (or becomes) loaded while the car carries downward velocity, with the
            // real vs predicted vertical response.
            if std::env::var("GGL_DUMP_LAND").is_ok() {
                let from_vz = Vec3A::from(fr.phys.lin_vel).z;
                let to_wheels = real.wheels.iter().filter(|w| w.has_contact).count();
                if from_vz < -50.0 && (wheels_touching > 0 || to_wheels > 0) {
                    let pred_dvz = pred.phys.vel.z - from_vz;
                    let real_dvz = real_vel.z - from_vz;
                    let min_susp = fr
                        .wheels
                        .iter()
                        .filter(|w| w.has_contact)
                        .map(|w| w.susp_length)
                        .fold(f32::INFINITY, f32::min);
                    eprintln!(
                        "LAND {i} c{c} vz={from_vz:8.1} wh={wheels_touching}->{to_wheels} susp={min_susp:7.3} pred_dvz={pred_dvz:8.2} real_dvz={real_dvz:8.2} err={:8.2} flip={} upz={:.2}",
                        pred_dvz - real_dvz,
                        u8::from(fr.is_flipping),
                        Vec3A::from(fr.phys.rot.rows[2]).x.mul_add(0.0, {
                            let r = &fr.phys.rot;
                            r.rows[2].z
                        }),
                    );
                }
            }

            // GGL_GROUND_DECOMP: signed error components for grounded ticks.
            if std::env::var("GGL_GROUND_DECOMP").is_ok() && regime == "ground" {
                let dv = pred.phys.vel - real_vel;
                let fwd: Vec3A = {
                    let r = &fr.phys.rot;
                    Vec3A::new(r.rows[0].x, r.rows[1].x, r.rows[2].x)
                };
                let up: Vec3A = {
                    let r = &fr.phys.rot;
                    Vec3A::new(r.rows[0].z, r.rows[1].z, r.rows[2].z)
                };
                let lat = up.cross(fwd);
                let c = &fr.prev_controls;
                let mode = if c.handbrake { "handbrake" }
                    else if c.boost { "boost" }
                    else if c.throttle > 0.5 { if c.steer.abs() > 0.5 { "throttle+steer" } else { "throttle" } }
                    else if c.throttle < -0.5 { "reverse" }
                    else { "coast" };
                eprintln!("GDECOMP {mode} {:.4} {:.4} {:.4} {i}", dv.dot(up), dv.dot(fwd), dv.dot(lat));
            }

            let e = regimes.entry(regime).or_default();
            e[0].push(vel_err);
            e[1].push(pos_err);
            e[2].push(ang_err);
            worst.push((vel_err, i, c, regime));
        }
    }

    // ---- REAL impulse census: the game's own per-tick force trace ----
    // lin impulses are velocity deltas in BT units (x50 -> uu/s); ang in rad/s.
    let mut imp_lin: std::collections::BTreeMap<String, Stat> = Default::default();
    let mut imp_ang: std::collections::BTreeMap<String, Stat> = Default::default();
    for tick in &recording.ticks {
        for cr in &tick.car_records {
            for imp in cr.phys.impulse_records() {
                let name = format!("{:?}", imp.impulse_type);
                let lin: Vec3A = imp.lin_impulse.into();
                let ang: Vec3A = imp.ang_impulse.into();
                imp_lin.entry(name.clone()).or_default().push(lin.length() * 50.0);
                imp_ang.entry(name).or_default().push(ang.length());
            }
        }
    }
    eprintln!("\n=== REAL per-tick impulse magnitudes (lin uu/s | ang rad/s) ===");
    for (name, stat) in &mut imp_lin {
        eprintln!("{name:14} lin {}", stat.summary());
    }
    for (name, stat) in &mut imp_ang {
        eprintln!("{name:14} ang {}", stat.summary());
    }

    eprintln!("skipped {skipped_gaps} gap transitions");
    eprintln!("\n=== per-regime single-tick VELOCITY error (uu/s) ===");
    for (name, stats) in &mut regimes {
        let summary = stats[0].summary();
        let v = &stats[0].vals;
        let n = v.len().max(1) as f32;
        let frac = |t: f32| 100.0 * v.iter().filter(|&&x| x < t).count() as f32 / n;
        let (f1, f5, f23) = (frac(1.0), frac(5.0), frac(23.0));
        eprintln!("{name:14} {summary}  <1uu/s:{f1:5.1}%  <5:{f5:5.1}%  <23(1%vmax):{f23:5.1}%");
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
