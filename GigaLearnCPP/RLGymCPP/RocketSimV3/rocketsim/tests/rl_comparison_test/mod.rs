use crate::rl_comparison_test::recording::Recording;
use crate::rl_comparison_test::recording::tick_record::TickRecord;
use glam::Vec3A;
use rocketsim::consts::BT_TO_UU;
use rocketsim::{Arena, CarBodyConfig, CarControls, CarState, GameMode, PhysState, Team};

mod analysis;
mod compare;
mod recording;

pub fn set_state_to_record_tick(
    arena: &mut Arena,
    car_idcs: &[usize],
    tick: &TickRecord,
    car_controls: &[CarControls],
    bakkes_semantics: bool,
) {
    for (i, &car_idx) in car_idcs.iter().enumerate() {
        let mut cs = *arena.get_car_state(car_idx);
        let rec = &tick.car_records[i];
        let rep_cs: CarState = (*rec).into();
        cs.phys = rep_cs.phys;
        cs.is_jumping = rep_cs.is_jumping;
        cs.is_flipping = rep_cs.is_flipping;
        cs.set_jump_time(rep_cs.jump_time());
        cs.set_flip_time(rep_cs.flip_time());
        cs.has_jumped = rep_cs.has_jumped;
        cs.prev_controls = rec.prev_controls.into();
        cs.controls = car_controls[i];
        cs.boost = rec.boost_amount;
        cs.is_demoed = false;
        cs.demo_respawn_ticks = 0;
        // Bakkes stores already-scaled dodge torque; CarState wants a unit direction.
        let rec_torque = Vec3A::from(rec.flip_rel_torque);
        cs.flip_rel_torque = if bakkes_semantics {
            rec_torque / Vec3A::new(260.0, 224.0, 1.0)
        } else {
            rec_torque
        };

        if bakkes_semantics {
            // Bakkes `has_flip` is "flip still available" — inverse of `has_flipped`.
            let flip_spent = !rec.has_flip && !rec.is_on_ground;
            let was_flip = rec_torque.length_squared() > 0.0;
            cs.has_flipped = rec.is_flipping || (flip_spent && was_flip);
            cs.has_double_jumped = flip_spent && !was_flip;
        } else if cs.has_flip_or_jump() {
            cs.has_double_jumped = rep_cs.has_double_jumped;
        }

        // GGL_RESTORE_WORLD=1: teacher-force chassis-world contact (autoflip / autoroll).
        if std::env::var("GGL_RESTORE_WORLD").is_ok_and(|s| s != "0") {
            cs.world_contact_normal = if rec.phys.has_world_contact {
                Some(rec.phys.world_contact_normal.into())
            } else {
                None
            };
        }

        arena.set_car_state(car_idx, cs);

        // GGL_RESTORE_WHEELS: overlay recorded per-wheel extra after refresh_contact.
        // Unset = off. `1`/`bt` = susp as Bullet units (tuned dump). `uu` = convert UU→BT.
        if let Ok(mode) = std::env::var("GGL_RESTORE_WHEELS") {
            if mode != "0" {
                let mut extra = arena.get_car_extra_state(car_idx);
                for (dst, src) in extra.wheels.iter_mut().zip(rec.wheels.iter()) {
                    dst.engine_force = src.engine_force;
                    dst.brake = src.brake;
                    dst.steer_angle = src.steer_amount;
                    dst.lat_friction = src.lat_friction;
                    dst.long_friction = src.long_friction;
                    dst.extra_pushback = src.extra_pushback;
                    // Bakkes susp_length is ~-2 while our raycast is ~+0.5 BT.
                    // Do not overlay length/relvel; they are a different convention.
                    if src.has_contact {
                        dst.has_raycast_info = 1;
                        dst.is_in_contact_with_world = 1;
                        dst.contact_normal = [
                            src.contact_normal.x,
                            src.contact_normal.y,
                            src.contact_normal.z,
                        ];
                    } else {
                        dst.has_raycast_info = 0;
                        dst.is_in_contact_with_world = 0;
                    }
                }
                arena.set_car_extra_state(car_idx, &extra);
            }
        }

        if std::env::var("GGL_DUMP_SUSP").is_ok() && rec.is_on_ground {
            static DUMP_N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            let n = DUMP_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n < 12 {
                let extra = arena.get_car_extra_state(car_idx);
                eprint!("SUSP rec=[");
                for w in &rec.wheels {
                    eprint!(" {:.4}", w.susp_length);
                }
                eprint!("] sim=[");
                for w in &extra.wheels {
                    eprint!(" {:.4}", w.suspension_length);
                }
                eprintln!(
                    "] vel=[{:.3} {:.3} {:.3} {:.3}]",
                    rec.wheels[0].susp_rel_vel,
                    rec.wheels[1].susp_rel_vel,
                    rec.wheels[2].susp_rel_vel,
                    rec.wheels[3].susp_rel_vel,
                );
            }
        }
    }

    let rep_bs_phys: PhysState = tick.ball_record.into();
    let mut bs = *arena.get_ball_state();
    bs.phys = rep_bs_phys;
    arena.set_ball_state(bs);
}

fn test_recording(recording: &Recording) {
    let mut arena = Arena::new(GameMode::Soccar);
    let num_cars = recording.info.num_cars as usize;
    let car_idcs: Vec<usize> = (0..num_cars)
        .map(|i| {
            let team = if (i % 2) == 0 {
                Team::Blue
            } else {
                Team::Orange
            };
            arena.add_car(team, CarBodyConfig::OCTANE)
        })
        .collect();

    for i in 0..(recording.ticks.len() - 1) {
        let from_tick = &recording.ticks[i];
        let to_tick = &recording.ticks[i + 1];
        let controls_during: Vec<CarControls> = to_tick
            .car_records
            .iter()
            .map(|car_record| car_record.prev_controls.into())
            .collect();
        let hb = &recording.info.hitbox_rel_min_bt;
        let bakkes = hb.x == 0.0 && hb.y == 0.0 && hb.z == 0.0;
        set_state_to_record_tick(&mut arena, &car_idcs, from_tick, &controls_during, bakkes);
        arena.step_tick();

        let ball_state = arena.get_ball_state();
        let car_states: Vec<CarState> = car_idcs
            .iter()
            .map(|&car_idx| *arena.get_car_state(car_idx))
            .collect();

        let comparison = compare::compare_states_to_tick(&car_states, ball_state, to_tick);
        let norm_error = comparison.calc_norm_error();
        if norm_error >= 1.0 {
            let recording_name = &recording.name;
            let mut lines: Vec<String> = vec![
                "=".repeat(50),
                format!(
                    "COMPARISON TEST \"{recording_name}\" FAILED at i={i}, norm_error={norm_error}"
                ),
            ];

            lines.push(format!(
                "Notable state fields (of {}):",
                comparison.map().len()
            ));
            for (k, v) in comparison.map() {
                let rel_error = v.rel_error();
                if rel_error >= 0.25 {
                    lines.push(format!(" > (e={rel_error}) \"{k}\": {v:#?}"));
                }
            }
            lines.push("CAR CONTROLS DURING TICK:".to_string());
            for (j, car_state) in car_states.iter().enumerate().take(num_cars) {
                lines.push(format!(" > Car [{j}]: {:?}", car_state.controls));
            }
            lines.push("CAR IMPULSES DURING TICK:".to_string());
            #[cfg(debug_assertions)]
            for j in 0..num_cars {
                lines.push(format!(" > Car [{j}]:"));
                let pred_impulses = arena.get_car_impulse_history(j);
                let real_impulses = to_tick.car_records[j].phys.impulse_records();

                if pred_impulses.len() + real_impulses.len() > 0 {
                    for ((name, is_accum), (lin_impulse, ang_impulse)) in pred_impulses {
                        let lin_impulse = lin_impulse * BT_TO_UU;
                        let ang_impulse = ang_impulse * BT_TO_UU;
                        lines.push(format!(
                            "\tPRED: {{ \
                            name: {name}, lin_impulse: {lin_impulse}, ang_impulse: {ang_impulse}, is_accum: {is_accum} \
                            }}"
                        ));
                    }
                    lines.push(String::new());
                    for real_impulse in real_impulses {
                        let impulse_type = real_impulse.impulse_type;
                        let lin_impulse: Vec3A = real_impulse.lin_impulse.into();
                        let ang_impulse: Vec3A = real_impulse.ang_impulse.into();
                        let lin_impulse = lin_impulse * BT_TO_UU;
                        let ang_impulse = ang_impulse * BT_TO_UU;
                        let is_accum = real_impulse.is_accum;
                        lines.push(format!(
                            "\tREAL: {{ \
                            name: {impulse_type:?}, lin_impulse: {lin_impulse}, ang_impulse: {ang_impulse}, is_accum: {is_accum} \
                            }}"
                        ));
                    }
                } else {
                    lines.push("\t(None)".to_string());
                }
            }

            panic!("{}", lines.join("\n"));
        }
    }
}

// This will be called from each test file
#[allow(unused)]
fn run_comparison_test(name: &str, recording_bytes: &[u8]) {
    if !rocketsim::is_initialized() {
        rocketsim::init_from_default(true).unwrap();
    }
    let recording = Recording::from_bytes(name, recording_bytes).unwrap();
    test_recording(&recording);
}
include!(concat!(env!("OUT_DIR"), "/gen_comparison_tests.rs"));
