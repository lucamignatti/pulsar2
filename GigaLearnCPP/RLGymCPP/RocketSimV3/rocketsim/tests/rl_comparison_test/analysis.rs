//! Whole-recording error analysis for real-game `.rlpr` captures (RLRecord2 /
//! BakkesMod). Unlike the pass/fail fixtures, this never panics: it replays every
//! clean transition, aggregates per-regime error statistics, and prints the worst
//! offenders for manual attribution. Run with:
//!
//!   RLPR_PATH=/path/to/capture.rlpr cargo test --release --test mod analyze_rlpr -- --nocapture
//!
//! **120 Hz only.** Rocket League physics is 120 ticks/s. Captures taken at 60 fps
//! (`UncappedFramerate=False`, RLBot packet rate, the old maneuver TSVs) are not
//! a valid physics tape: either every other tick is missing (`physics_frame` Δ=2)
//! or the engine itself ran at 60 Hz (Δvz per recorded step ≈ 10.8 instead of
//! 5.42). This test refuses those files. Do not "fix" that by intersecting
//! every other sim tick.
//!
//! Control pairing: BakkesMod-semantics files (detected by the zeroed hitbox in the
//! header, per rlpr_format.h) store the inputs consumed DURING frame F in record F
//! itself; sim-generated fixtures store them in record F+1's `prev_controls`.

use glam::{Mat3A, Vec3A};
use rocketsim::consts::{arena, ball, goal};
use rocketsim::{Arena, ArenaEvent, CarBodyConfig, CarControls, GameMode, Team};
use super::recording::cpp_records::{CarRecord, ImpulseRecordType, WheelRecord};

trait XyLen {
    fn xy_len_sq(&self) -> f32;
}
impl XyLen for Vec3A {
    fn xy_len_sq(&self) -> f32 {
        self.x * self.x + self.y * self.y
    }
}

fn extra_vec(a: [f32; 3]) -> Vec3A {
    Vec3A::new(a[0], a[1], a[2])
}

fn unit_angle_deg(a: Vec3A, b: Vec3A) -> Option<f32> {
    let a = a.try_normalize()?;
    let b = b.try_normalize()?;
    Some(a.dot(b).clamp(-1.0, 1.0).acos().to_degrees())
}

/// Angle ignoring a 180° sign flip (axle / long dir).
fn unsigned_angle_deg(a: Vec3A, b: Vec3A) -> Option<f32> {
    let a = a.try_normalize()?;
    let b = b.try_normalize()?;
    Some(a.dot(b).abs().clamp(0.0, 1.0).acos().to_degrees())
}

use super::recording::cpp_records::ControlsRecord;
use super::recording::Recording;
use super::recording::tick_record::TickRecord;
use super::set_state_to_record_tick;

fn ggl_tape_n() -> bool {
    std::env::var("GGL_TAPE_N").is_ok_and(|s| s != "0")
}

fn car_is_packed_fillet(fr: &super::recording::cpp_records::CarRecord) -> bool {
    if fr.is_flipping || !fr.is_on_ground {
        return false;
    }
    let wheels_touching = fr.wheels.iter().filter(|w| w.has_contact).count();
    let on_wall = fr
        .wheels
        .iter()
        .any(|w| w.has_contact && Vec3A::from(w.contact_normal).z.abs() < 0.7);
    let up_z = Vec3A::new(fr.phys.rot.rows[0].z, fr.phys.rot.rows[1].z, fr.phys.rot.rows[2].z)
        .z
        .abs();
    let min_susp = fr
        .wheels
        .iter()
        .filter(|w| w.has_contact)
        .map(|w| w.susp_length)
        .fold(f32::INFINITY, f32::min);
    on_wall && wheels_touching == 4 && min_susp < -6.0 && up_z >= 0.25
}

fn inject_tape_n(arena: &mut Arena, car_idcs: &[usize], tick: &TickRecord) {
    if !ggl_tape_n() {
        return;
    }
    let packed_only = std::env::var("GGL_TAPE_N").is_ok_and(|s| s != "all");
    for (c, &car_idx) in car_idcs.iter().enumerate() {
        let Some(fr) = tick.car_records.get(c) else {
            continue;
        };
        if packed_only && !car_is_packed_fillet(fr) {
            arena.set_susp_n_override(car_idx, [None; 4]);
            continue;
        }
        let mut ns = [None; 4];
        for w in 0..4 {
            if !fr.wheels[w].has_contact {
                continue;
            }
            ns[w] = Vec3A::from(fr.wheels[w].contact_normal).try_normalize();
        }
        arena.set_susp_n_override(car_idx, ns);
    }
}

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

/// Ball-accuracy surfaces. Scoreboard order: ground, air, wall (fillets
/// included), crossbar, post. Derived from `consts::{arena,ball,goal}` (wiki /
/// RLBot useful-game-values). Do not invent extra ball buckets.
const BALL_SURFACES: [&str; 5] = ["ground", "air", "wall", "crossbar", "post"];

fn dist_point_segment(p: Vec3A, a: Vec3A, b: Vec3A) -> f32 {
    let ab = b - a;
    let denom = ab.length_squared();
    if denom < 1e-8 {
        return (p - a).length();
    }
    let t = (p - a).dot(ab).clamp(0.0, denom) / denom;
    (p - (a + ab * t)).length()
}

/// Classify a ball *center* into a sim2real surface. Post/crossbar win over
/// wall; floor-wall and wall-ceiling ramps are **wall**, not ground.
fn classify_ball_surface(p: Vec3A) -> &'static str {
    let r = ball::get_radius(GameMode::Soccar);
    let slack = 20.0;
    let frame_r = goal::SOCCAR_GOAL_FRAME_RADIUS;
    let hw = goal::SOCCAR_GOAL_HALF_WIDTH;
    let gh = goal::SOCCAR_GOAL_HEIGHT;
    let ext_x = arena::SOCCAR_EXTENT_X;
    let ext_y = arena::SOCCAR_EXTENT_Y;
    let ceil = arena::SOCCAR_CEILING_Z;
    let ramp = arena::SOCCAR_WALL_BOTTOM_RAMP;
    let rest = ball::REST_Z;

    let mut d_post = f32::INFINITY;
    let mut d_bar = f32::INFINITY;
    for sy in [-1.0, 1.0] {
        let y = sy * ext_y;
        d_bar = d_bar.min(dist_point_segment(
            p,
            Vec3A::new(-hw, y, gh),
            Vec3A::new(hw, y, gh),
        ));
        for sx in [-1.0, 1.0] {
            let x = sx * hw;
            d_post = d_post.min(dist_point_segment(
                p,
                Vec3A::new(x, y, 0.0),
                Vec3A::new(x, y, gh),
            ));
        }
    }
    let frame_hit = r + frame_r + slack;
    if d_post <= frame_hit || d_bar <= frame_hit {
        return if d_post <= d_bar {
            "post"
        } else {
            "crossbar"
        };
    }

    let side = ext_x - p.x.abs();
    let back = ext_y - p.y.abs();
    let in_mouth = p.x.abs() < hw - frame_r && p.z < gh - frame_r;
    let back_wall = if in_mouth { f32::INFINITY } else { back };
    let corner = (arena::SOCCAR_CORNER_INTERCEPT - p.x.abs() - p.y.abs()) / 2.0f32.sqrt();
    let wall_dist = side.min(back_wall).min(ceil - p.z).min(corner);
    let plane_hit = r + slack;
    let fillet = wall_dist < ramp + r && p.z < ramp + rest + slack;
    let ceil_fillet = wall_dist < ramp + r && p.z > ceil - ramp - r - slack;
    if wall_dist < plane_hit || fillet || ceil_fillet {
        return "wall";
    }
    if p.z < rest + slack {
        return "ground";
    }
    "air"
}

/// Closest Octane/Plank/… hitbox feature for a ball center in world space.
/// Face is in car axes: nose=+x, tail=−x, left=+y, right=−y, roof=+z, underside=−z.
fn classify_hitbox_feature(
    ball_world: Vec3A,
    car_pos: Vec3A,
    car_rot: Mat3A,
    cfg: CarBodyConfig,
) -> (Vec3A, &'static str, &'static str, f32) {
    let local = car_rot.transpose() * (ball_world - car_pos);
    let p = local - cfg.hitbox_pos_offset;
    let half = cfg.hitbox_size * 0.5;
    let ax = p.x.abs() - half.x;
    let ay = p.y.abs() - half.y;
    let az = p.z.abs() - half.z;
    let out_x = ax > 0.0;
    let out_y = ay > 0.0;
    let out_z = az > 0.0;
    let n_out = u8::from(out_x) + u8::from(out_y) + u8::from(out_z);
    let feat = match n_out {
        0 | 1 => "face",
        2 => "edge",
        _ => "corner",
    };
    let pick = |x: f32, y: f32, z: f32| {
        if x >= y && x >= z {
            if p.x >= 0.0 {
                "nose"
            } else {
                "tail"
            }
        } else if y >= z {
            if p.y >= 0.0 {
                "left"
            } else {
                "right"
            }
        } else if p.z >= 0.0 {
            "roof"
        } else {
            "underside"
        }
    };
    let face = if n_out == 0 {
        pick(-ax, -ay, -az)
    } else {
        pick(
            if out_x { ax } else { -1.0 },
            if out_y { ay } else { -1.0 },
            if out_z { az } else { -1.0 },
        )
    };
    let q = Vec3A::new(
        p.x.clamp(-half.x, half.x),
        p.y.clamp(-half.y, half.y),
        p.z.clamp(-half.z, half.z),
    );
    let r = ball::get_radius(GameMode::Soccar);
    let dist_surface = (p - q).length() - r;
    (p, face, feat, dist_surface)
}

fn car_rot_mat(rec: &CarRecord) -> Mat3A {
    Mat3A::from_cols(
        rec.phys.rot.column(0).into(),
        rec.phys.rot.column(1).into(),
        rec.phys.rot.column(2).into(),
    )
}

/// `ref` / `ray` / `rest` are car-local (fwd/right/up). `contact_location` is world.
fn wheel_local_to_world(rec: &CarRecord, local: Vec3A) -> Vec3A {
    Vec3A::from(rec.phys.pos) + car_rot_mat(rec) * local
}

/// Wheel sphere at rest, then drop by `susp_length` along chassis up, then −r n.
fn tape_contact_from_rest(rec: &CarRecord, w: &WheelRecord) -> Option<Vec3A> {
    if w.wheel_radius < 1.0 {
        return None;
    }
    let rest_w = wheel_local_to_world(rec, Vec3A::from(w.preset_rest_position));
    let up = Vec3A::from(rec.phys.rot.up());
    let n = Vec3A::from(w.contact_normal).try_normalize()?;
    Some(rest_w - up * w.susp_length - n * w.wheel_radius)
}

fn tape_lin_imp_uu(cr: &CarRecord, ty: ImpulseRecordType) -> Vec3A {
    cr.phys
        .impulse_records()
        .iter()
        .filter(|i| i.impulse_type == ty)
        .fold(Vec3A::ZERO, |acc, i| acc + Vec3A::from(i.lin_impulse) * 50.0)
}

fn tape_has_imp(cr: &CarRecord, ty: ImpulseRecordType) -> bool {
    cr.phys
        .impulse_records()
        .iter()
        .any(|i| i.impulse_type == ty)
}

fn init_collision_meshes() {
    if rocketsim::is_initialized() {
        return;
    }
    let mut paths = Vec::new();
    if let Ok(p) = std::env::var("RLPR_MESHES") {
        paths.push(std::path::PathBuf::from(p));
    }
    paths.push(std::path::PathBuf::from("./collision_meshes"));
    paths.push(std::path::PathBuf::from("./collision_meshes"));
    if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
        let root = std::path::PathBuf::from(manifest);
        paths.push(root.join("collision_meshes"));
        paths.push(root.join("../../../../collision_meshes"));
        paths.push(root.join("../../../../build/collision_meshes"));
        paths.push(root.join("../../../../build/collision_meshes"));
        paths.push(root.join("../../../../build-npl/collision_meshes"));
    }
    let mut last_err = None;
    for p in &paths {
        if p.is_dir() {
            match rocketsim::init(p, true) {
                Ok(()) => {
                    eprintln!("collision meshes: {}", p.display());
                    return;
                }
                Err(e) => last_err = Some(e),
            }
        }
    }
    panic!(
        "analyze_rlpr: no collision meshes (set RLPR_MESHES). last_err={last_err:?}"
    );
}

/// Median physics_frame delta and median airborne |Δvz|. 120 Hz tape: Δframe=1
/// and free-fall Δvz ≈ g/120 ≈ 5.42. 60 fps tapes: Δframe=2 and/or Δvz ≈ 10.8.
fn tape_rate(recording: &Recording, num_cars: usize) -> (u32, f32, f32, usize) {
    let mut dfs = Vec::new();
    let mut dvz = Vec::new();
    for w in recording.ticks.windows(2) {
        if w[0].car_records.len() != num_cars || w[1].car_records.len() != num_cars {
            continue;
        }
        for c in 0..num_cars {
            let a = &w[0].car_records[c];
            let b = &w[1].car_records[c];
            let df = b.phys.physics_frame.saturating_sub(a.phys.physics_frame);
            if df > 0 && df < 20 {
                dfs.push(df);
            }
            if !a.is_on_ground && !b.is_on_ground && !a.prev_controls.jump && !a.prev_controls.boost
            {
                let vz0: Vec3A = a.phys.lin_vel.into();
                let vz1: Vec3A = b.phys.lin_vel.into();
                dvz.push((vz1.z - vz0.z).abs());
            }
        }
    }
    dfs.sort_unstable();
    dvz.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med_df = if dfs.is_empty() { 0 } else { dfs[dfs.len() / 2] };
    let frac1 = if dfs.is_empty() {
        0.0
    } else {
        dfs.iter().filter(|&&d| d == 1).count() as f32 / dfs.len() as f32
    };
    let med_dvz = if dvz.is_empty() {
        0.0
    } else {
        dvz[dvz.len() / 2]
    };
    (med_df, frac1, med_dvz, dfs.len())
}

fn parse_body_name(s: &str) -> Option<CarBodyConfig> {
    Some(match s.trim().to_ascii_lowercase().as_str() {
        "octane" | "23" => CarBodyConfig::OCTANE,
        "dominus" => CarBodyConfig::DOMINUS,
        "plank" | "batmobile" | "1919" => CarBodyConfig::PLANK,
        "breakout" => CarBodyConfig::BREAKOUT,
        "hybrid" => CarBodyConfig::HYBRID,
        "merc" => CarBodyConfig::MERC,
        "psyclops" => CarBodyConfig::PSYCLOPS,
        _ => return None,
    })
}

/// Bakkes `.rlpr` headers zero the hitbox, so the tape cannot name the body.
/// `GGL_CAR_BODIES=octane,plank` overrides; else `{RLPR_PATH}.meta.txt` `body_id=`
/// (23 = Octane, 1919 = Plank / Batmobile on the 2026-08-23 tape).
fn car_bodies_for_tape(rlpr_path: &str, num_cars: usize) -> Vec<CarBodyConfig> {
    let mut bodies = vec![CarBodyConfig::OCTANE; num_cars];
    if let Ok(spec) = std::env::var("GGL_CAR_BODIES") {
        for (i, part) in spec.split(',').enumerate() {
            if i >= num_cars {
                break;
            }
            if let Some(cfg) = parse_body_name(part) {
                bodies[i] = cfg;
            } else {
                eprintln!("GGL_CAR_BODIES: unknown token {part:?}, slot {i} stays Octane");
            }
        }
        return bodies;
    }
    let meta_path = std::path::Path::new(rlpr_path).with_extension("meta.txt");
    if let Ok(text) = std::fs::read_to_string(&meta_path) {
        for line in text.lines() {
            let Some(slot_s) = line.strip_prefix("slot ") else {
                continue;
            };
            let Some((idx_s, rest)) = slot_s.split_once(':') else {
                continue;
            };
            let Ok(idx) = idx_s.trim().parse::<usize>() else {
                continue;
            };
            if idx >= num_cars {
                continue;
            }
            if let Some(id_s) = rest.split("body_id=").nth(1) {
                let id = id_s.split_whitespace().next().unwrap_or("");
                if let Some(cfg) = parse_body_name(id) {
                    bodies[idx] = cfg;
                } else {
                    eprintln!("meta body_id={id} slot {idx}: no mapping, Octane");
                }
            }
        }
    }
    bodies
}

#[test]
fn analyze_rlpr() {
    let Ok(path) = std::env::var("RLPR_PATH") else {
        eprintln!("analyze_rlpr: RLPR_PATH not set, skipping");
        return;
    };
    init_collision_meshes();
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

    let (med_df, frac1, med_dvz, n_df) = tape_rate(&recording, num_cars);
    eprintln!(
        "tape rate: physics_frame Δ median={med_df}  frac_Δ1={:.4}  n={n_df}  airborne |Δvz| p50={med_dvz:.3} (120Hz free-fall ≈5.42, 60Hz ≈10.83)",
        frac1
    );
    if med_df != 1 || frac1 < 0.95 {
        panic!(
            "analyze_rlpr: refused 60 fps / skipped-tick tape (physics_frame Δ median={med_df}, frac_Δ1={frac1:.4}). Need a 120 Hz capture (Δ=1). Do not intersect every other tick."
        );
    }
    // Δvz gate: skip if we have too few airborne samples, but refuse a clear 60 Hz dt.
    if med_dvz > 8.0 {
        panic!(
            "analyze_rlpr: refused tape whose airborne |Δvz| p50={med_dvz:.3} matches 60 Hz physics dt, not 120 Hz (≈5.42)."
        );
    }

    let mut arena = {
        let mut cfg = rocketsim::ArenaConfig::new(GameMode::Soccar);
        // GGL_BALL_SCALE: experiment knob on the ball-hit extra impulse (S43).
        if let Ok(v) = std::env::var("GGL_BALL_SCALE")
            && let Ok(f) = v.parse::<f32>()
        {
            cfg.mutators.ball_hit_extra_force_scale = f;
            eprintln!("GGL_BALL_SCALE applied: {f}");
        }
        Arena::new_with_config(cfg)
    };
    let bodies = car_bodies_for_tape(&path, num_cars);
    let body_label = |c: CarBodyConfig| {
        if c == CarBodyConfig::OCTANE {
            "octane"
        } else if c == CarBodyConfig::PLANK {
            "plank"
        } else if c == CarBodyConfig::DOMINUS {
            "dominus"
        } else if c == CarBodyConfig::BREAKOUT {
            "breakout"
        } else if c == CarBodyConfig::HYBRID {
            "hybrid"
        } else if c == CarBodyConfig::MERC {
            "merc"
        } else {
            "other"
        }
    };
    eprintln!(
        "car bodies: {}",
        bodies
            .iter()
            .enumerate()
            .map(|(i, c)| format!("c{i}={}", body_label(*c)))
            .collect::<Vec<_>>()
            .join(" ")
    );
    let car_idcs: Vec<usize> = (0..num_cars)
        .map(|i| {
            let team = if i % 2 == 0 { Team::Blue } else { Team::Orange };
            arena.add_car(team, bodies[i])
        })
        .collect();

    let pad_configs = arena.get_all_boost_pad_configs();

    // regime name -> (vel_err, pos_err, ang_vel_err) stats
    let mut regimes: std::collections::BTreeMap<&'static str, [Stat; 3]> =
        std::collections::BTreeMap::new();
    // (car, regime) -> vel_err — tape 2 c1 is a Plank (body_id 1919), not skip-4 Octane.
    let mut regimes_by_car: std::collections::BTreeMap<(usize, &'static str), Stat> =
        std::collections::BTreeMap::new();
    let mut wall_by_nw: std::collections::BTreeMap<(usize, usize), Stat> =
        std::collections::BTreeMap::new();
    let mut wall_by_nz: std::collections::BTreeMap<(usize, &'static str), Stat> =
        std::collections::BTreeMap::new();
    let mut aw_by_nw: std::collections::BTreeMap<(usize, usize), Stat> =
        Default::default();
    let mut ball_regimes: std::collections::BTreeMap<&'static str, [Stat; 2]> =
        Default::default();
    let mut ball_touch_rows: Vec<(
        f32,
        &'static str,
        &'static str,
        bool,
        bool,
        bool,
        bool,
        f32,
        usize,
    )> = Vec::new();
    // (vel_err, tick_index, car, regime) worst list
    let mut worst: Vec<(f32, usize, usize, &'static str)> = Vec::new();
    let mut skipped_gaps = 0usize;
    // flip+wheels restore quantization vs raw tape flip_time (1-step gates).
    let mut fw_n = 0u32;
    let mut fw_pitch = 0u32;
    let mut fw_zstart = 0u32;
    let mut fw_zend = 0u32;
    let mut fw_torque = 0u32;
    let mut fw_plock = 0u32;
    let mut fw_no_has_flip = 0u32;
    let mut fw_err_cross: Vec<f32> = Vec::new();
    let mut fw_err_same: Vec<f32> = Vec::new();
    let mut fw_by_wheels: [Vec<f32>; 5] = Default::default();
    let mut fw_floor: Vec<f32> = Vec::new();
    let mut fw_wall: Vec<f32> = Vec::new();
    let mut packed_vel = Stat::default();
    let mut packed_d_fci = Stat::default();
    let mut packed_d_push = Stat::default();
    let mut packed_d_susp = Stat::default();
    let mut packed_tape_drive = Stat::default();
    let mut packed_tape_fci = Stat::default();
    let mut packed_sim_fci = Stat::default();
    let mut packed_n = 0usize;
    let mut packed_n_ang = Stat::default();
    let mut packed_lat_ang = Stat::default();
    let mut packed_long_ang = Stat::default();
    let mut packed_pt_dist = Stat::default();
    let mut packed_rest_to_hit = Stat::default();
    let mut packed_pred_contact = Stat::default();
    let mut packed_sim_vs_pred = Stat::default();
    let mut flat_sim_vs_pred = Stat::default();
    let mut flat_sim_vs_tape = Stat::default();
    let mut car_imp_vel = Stat::default();
    let mut ball_imp_vel = Stat::default();
    let mut car_imp_applied = Stat::default();
    let mut car_imp_missed = Stat::default();
    let mut ball_imp_applied = Stat::default();
    let mut ball_imp_missed = Stat::default();
    let mut ball_extra_vs_tape = Stat::default();
    let mut car_imp_tape_n = 0u32;
    let mut car_imp_sim_n = 0u32;
    let mut car_imp_both = 0u32;
    let mut car_imp_tape_only = 0u32;
    let mut car_imp_sim_only = 0u32;
    let mut ball_imp_tape_n = 0u32;
    let mut ball_imp_sim_n = 0u32;
    let mut ball_imp_both = 0u32;
    let mut ball_imp_tape_only = 0u32;
    let mut ball_imp_sim_only = 0u32;
    let mut packed_d_steer = Stat::default();
    let mut packed_tape_spin = Stat::default();
    let mut packed_tape_wvel = Stat::default();
    let mut packed_d_wvel = Stat::default();
    let mut packed_spin_r_vs_wvel = Stat::default();
    let mut packed_tape_brake = Stat::default();
    let mut packed_d_out_thr = Stat::default();
    let mut packed_d_out_steer = Stat::default();
    let mut packed_d_out_brake = Stat::default();
    let mut packed_d_out_hb = Stat::default();
    let mut packed_d_sim_thr = Stat::default();
    let mut packed_d_sim_steer = Stat::default();
    let mut packed_d_sim_hb = Stat::default();
    let mut fillet_samples: Vec<String> = Vec::new();
    let mut aw_d_thr = Stat::default();
    let mut aw_d_steer = Stat::default();
    let mut aw_output_n = 0usize;

    let tick_lo: usize = std::env::var("GGL_TICK_BEGIN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let tick_hi: usize = std::env::var("GGL_TICK_END")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(usize::MAX);
    eprintln!("tick window: [{tick_lo}, {tick_hi})  (GGL_TICK_BEGIN/GGL_TICK_END)");
    if ggl_tape_n() {
        let mode = std::env::var("GGL_TAPE_N").unwrap_or_default();
        eprintln!("GGL_TAPE_N={mode}  (packed-fillet tape contact_normal → susp/pushback; friction stays on sim ray)");
    }

    {
        fn ctrl_changed(a: &ControlsRecord, b: &ControlsRecord) -> bool {
            (a.throttle - b.throttle).abs() > 0.02
                || (a.steer - b.steer).abs() > 0.02
                || (a.pitch - b.pitch).abs() > 0.02
                || (a.yaw - b.yaw).abs() > 0.02
                || (a.roll - b.roll).abs() > 0.02
                || a.jump != b.jump
                || a.boost != b.boost
                || a.handbrake != b.handbrake
        }
        eprintln!("=== control-hold (prev_controls run length) ===");
        for c in 0..num_cars {
            let mut holds = Vec::new();
            let mut run = 1u32;
            let mut prev: Option<ControlsRecord> = None;
            for tick in &recording.ticks {
                if tick.car_records.len() != num_cars {
                    prev = None;
                    run = 1;
                    continue;
                }
                let cur = tick.car_records[c].prev_controls;
                if let Some(p) = prev {
                    if ctrl_changed(&p, &cur) {
                        holds.push(run);
                        run = 1;
                    } else {
                        run = run.saturating_add(1);
                    }
                }
                prev = Some(cur);
            }
            holds.push(run);
            holds.sort_unstable();
            let n = holds.len();
            let med = if n == 0 { 0 } else { holds[n / 2] };
            let p90 = if n == 0 { 0 } else { holds[(n * 9) / 10] };
            let ge4 = holds.iter().filter(|&&h| h >= 4).count();
            let ge4f = if n == 0 { 0.0 } else { ge4 as f32 / n as f32 };
            eprintln!(
                "  car {c}: n_runs={n}  med_hold={med}  p90_hold={p90}  frac_hold>=4={ge4f:.3}"
            );
        }
    }
    let horizon: usize = std::env::var("GGL_FREERUN_TICKS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1)
        .max(1);
    eprintln!("free-run horizon: {horizon} tick(s)  (GGL_FREERUN_TICKS, 1 = restore-then-1-step)");
    if let Ok(s) = std::env::var("GGL_SAT_SLACK") {
        eprintln!("GGL_SAT_SLACK={s} uu (closest-axis miss→hit; not box inflate)");
    }

    for i in 0..recording.ticks.len().saturating_sub(horizon) {
        if i < tick_lo || i + horizon > tick_hi {
            continue;
        }
        // PR74 ce694cd: skip recording tick 0 (SetPhysicsState settle). No-op
        // under GGL_CTRL_LAG=2, which already drops i=0/1. GGL_PR74_SKIP0=1.
        if i == 0 && std::env::var("GGL_PR74_SKIP0").is_ok_and(|s| s != "0") {
            continue;
        }
        let from_tick = &recording.ticks[i];
        let to_tick = &recording.ticks[i + horizon];

        // Skip transitions across recorder gaps (game paused / frames missed),
        // around demo respawns (short roster ticks: slot assignment is ambiguous),
        // and across TELEPORTS (goal resets / demolitions): no legal physics moves a
        // car more than ~|v_max| * dt + margin in one tick, so a larger real position
        // delta is a game event, not dynamics.
        // A roster SHRINK is a real demolition during this transition: still step it
        // (with the same lagged controls) so the demo pipeline can be validated, but
        // skip the error stats (the vanished car has no ground truth).
        let pair_clean = |a: &TickRecord, b: &TickRecord| -> bool {
            a.car_records.len() == num_cars
                && b.car_records.len() == num_cars
                && (0..num_cars).all(|c| {
                    b.car_records[c].phys.physics_frame
                        == a.car_records[c].phys.physics_frame + 1
                })
                && (0..num_cars).all(|c| {
                    let fp: Vec3A = a.car_records[c].phys.pos.into();
                    let tp: Vec3A = b.car_records[c].phys.pos.into();
                    (tp - fp).length() < 40.0
                })
        };
        let demo_transition = from_tick.car_records.len() == num_cars
            && to_tick.car_records.len() < num_cars
            && horizon == 1;
        let clean = (0..horizon).all(|k| {
            pair_clean(&recording.ticks[i + k], &recording.ticks[i + k + 1])
        });
        if !clean && demo_transition {
            let ctrl_i = (i + 1).saturating_sub(2);
            if recording.ticks[ctrl_i].car_records.len() == num_cars {
                let controls: Vec<CarControls> = (0..num_cars)
                    .map(|c| recording.ticks[ctrl_i].car_records[c].prev_controls.into())
                    .collect();
                set_state_to_record_tick(&mut arena, &car_idcs, from_tick, &controls, bakkes_semantics);
                for ev in arena.step_tick().to_vec() {
                    if let rocketsim::ArenaEvent::CarHitCar(e) = ev {
                        eprintln!(
                            "REALDEMO tick {i}: sim CarHitCar attacker={} victim={} is_demo={}",
                            e.bumper_car_idx, e.victim_car_idx, e.is_demo
                        );
                    }
                }
                eprintln!("REALDEMO tick {i}: transition stepped (roster {} -> {})",
                    from_tick.car_records.len(), to_tick.car_records.len());
            }
            skipped_gaps += 1;
            continue;
        }
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
        if i + horizon < lag {
            continue;
        }
        let mut skip_ctrl = false;
        for k in 0..horizon {
            let ctrl_i = i + k + 1 - lag;
            if recording.ticks[ctrl_i].car_records.len() != num_cars {
                skip_ctrl = true;
                break;
            }
        }
        if skip_ctrl {
            continue;
        }
        // GGL_BOOST_ALIGN (default on; =0 disables): the recorded control stream places
        // the boost press on a tick the fixed CTRL_LAG cannot always reproduce. Measured
        // 2026-08-23: the all-regime p90 error band is EXACTLY one tick of boost --
        // 8.264 = 991.667/120 on thr=0 ticks, 7.708 = (991.667-66.667)/120 on thr=+1
        // ticks (sim applied air throttle, real applied boost) -- 97% sim-slow, pure
        // forward-axis. The tape's boost_amount telemetry shows the actual burn, so
        // teacher-force controls.boost from consumption: burned => boosting, unchanged
        // => not boosting, gained (pad pickup can mask a concurrent burn) => keep the
        // lagged control bit.
        let boost_align = std::env::var("GGL_BOOST_ALIGN").map_or(true, |s| s != "0");
        let opp_filter: f32 = std::env::var("GGL_OPP_FILTER")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0);
        let align_boost = |k: usize, c: usize, ctrl: &mut CarControls| {
            if !boost_align {
                return;
            }
            let fb = recording.ticks[i + k].car_records[c].boost_amount;
            let tb = recording.ticks[i + k + 1].car_records[c].boost_amount;
            if tb < fb - 1e-4 {
                ctrl.boost = true;
            } else if tb <= fb + 1e-4 && fb < 0.999 {
                // Unchanged tank => not boosting -- UNLESS the tank is pinned full
                // (freeplay unlimited-boost mutator burns nothing); then consumption
                // carries no signal and the lagged control bit stands. Without this
                // guard the aligner strips boost on every tick of such tapes
                // (ground_boost p50 read exactly one boost quantum, 8.258).
                ctrl.boost = false;
            }
        };
        let mut controls0: Vec<CarControls> = (0..num_cars)
            .map(|c| recording.ticks[i + 1 - lag].car_records[c].prev_controls.into())
            .collect();
        for (c, ctrl) in controls0.iter_mut().enumerate() {
            align_boost(0, c, ctrl);
        }

        set_state_to_record_tick(&mut arena, &car_idcs, from_tick, &controls0, bakkes_semantics);
        for (c, &car_idx) in car_idcs.iter().enumerate() {
            let mut cs = *arena.get_car_state(car_idx);
            cs.boost = from_tick.car_records[c].boost_amount * 100.0;
            arena.set_car_state(car_idx, cs);
        }
        // The restore's refresh_contact re-raycasts at the restored pose, erasing the
        // one-tick contact history the sticky prev-gate depends on. Seed it from the
        // RECORDED previous tick's wheel flags so a falling grazing wheel correctly
        // gets no sticky on its first contact tick (the -2.53 uu/s T1 graze bias).
        if i > 0 && recording.ticks[i - 1].car_records.len() == num_cars {
            for (c, &car_idx) in car_idcs.iter().enumerate() {
                let had = recording.ticks[i - 1].car_records[c]
                    .wheels
                    .iter()
                    .any(|w| w.has_contact);
                arena.seed_sticky_gate_prev(car_idx, had);
            }
        }
        inject_tape_n(&mut arena, &car_idcs, from_tick);
        let pre_flip: Vec<(bool, bool, u32)> = car_idcs
            .iter()
            .map(|&car_idx| {
                let cs = arena.get_car_state(car_idx);
                (cs.has_flipped, cs.is_flipping, cs.flip_ticks)
            })
            .collect();
        // Sim's post-restore wheel-contact mask (refresh_contact ray decisions at the
        // GT pose) for contact-agreement analysis against the recorded flags.
        let sim_wheel_masks: Vec<u8> = car_idcs
            .iter()
            .map(|&car_idx| {
                arena
                    .get_car_state(car_idx)
                    .wheels_with_contact
                    .iter()
                    .enumerate()
                    .fold(0u8, |m, (k, &c)| m | (u8::from(c) << k))
            })
            .collect();
        let wheel_car_masks: Vec<u8> = car_idcs
            .iter()
            .map(|&car_idx| arena.wheel_hit_car_mask(car_idx))
            .collect();

        // GGL_IMP_TRACE="lo:hi": for from-ticks i in [lo, hi], print every labeled
        // impulse the engine applies during the step (IMP lines, velocity units
        // in BT). Used to attribute a per-tick velocity error to a specific term.
        let imp_trace = {
            static V: std::sync::OnceLock<Option<(usize, usize)>> = std::sync::OnceLock::new();
            *V.get_or_init(|| {
                let s = std::env::var("GGL_IMP_TRACE").ok()?;
                let (a, b) = s.split_once(':')?;
                Some((a.parse().ok()?, b.parse().ok()?))
            })
        };
        let tracing = imp_trace.is_some_and(|(lo, hi)| i >= lo && i <= hi);
        if tracing {
            eprintln!("IMPTICK i={i}");
            rocketsim::DBG_IMPULSE_TRACE.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        rocketsim::reset_last_sat_gap();
        let mut step_events: Vec<rocketsim::ArenaEvent> = Vec::new();
        for k in 0..horizon {
            if k > 0 {
                let ctrl_i = i + k + 1 - lag;
                for (c, &car_idx) in car_idcs.iter().enumerate() {
                    let mut cs = *arena.get_car_state(car_idx);
                    cs.controls = recording.ticks[ctrl_i].car_records[c].prev_controls.into();
                    align_boost(k, c, &mut cs.controls);
                    arena.set_car_state(car_idx, cs);
                }
            }
            step_events.extend(arena.step_tick().to_vec());
        }
        if tracing {
            rocketsim::DBG_IMPULSE_TRACE.store(false, std::sync::atomic::Ordering::Relaxed);
        }
        for ev in &step_events {
            if let ArenaEvent::CarHitCar(e) = ev
                && e.is_demo
            {
                eprintln!(
                    "DEMOEVT tick {i} attacker={} victim={}",
                    e.bumper_car_idx, e.victim_car_idx
                );
            }
        }

        // Ball 1-tick error, always scored on ground/air/wall/crossbar/post.
        // The ball state was restored from from_tick. GGL_BALL dumps per-tick lines.
        {
            let bs = arena.get_ball_state();
            let real_bp: Vec3A = to_tick.ball_record.pos.into();
            let real_bv: Vec3A = to_tick.ball_record.lin_vel.into();
            let real_bav: Vec3A = to_tick.ball_record.ang_vel.into();
            let from_bp: Vec3A = from_tick.ball_record.pos.into();
            let from_bv: Vec3A = from_tick.ball_record.lin_vel.into();
            // Skip ball teleports (goal resets). 60 uu is a 1-tick bound; scale
            // with free-run horizon or moving balls are dropped as "teleports".
            if (real_bp - from_bp).length() < 60.0 * horizon as f32 {
                let verr = (bs.phys.vel - real_bv).length();
                let averr = (bs.phys.ang_vel - real_bav).length();
                let surf = classify_ball_surface(real_bp);
                let slot = ball_regimes.entry(surf).or_default();
                slot[0].push(verr);
                slot[1].push(averr);
                let touching = (0..num_cars).any(|c| {
                    from_tick.car_records[c].is_touching_ball
                        || to_tick.car_records[c].is_touching_ball
                });
                let dv_real = (real_bv - from_bv).length();
                let sim_hit = step_events
                    .iter()
                    .any(|e| matches!(e, ArenaEvent::CarHitBall(_)));
                let mut best: Option<(f32, usize, Vec3A, &'static str, &'static str, f32)> =
                    None;
                for c in 0..num_cars {
                    if from_tick.car_records.len() != num_cars {
                        break;
                    }
                    let cr = &from_tick.car_records[c];
                    let (local, face, feat, dist) = classify_hitbox_feature(
                        from_bp,
                        cr.phys.pos.into(),
                        car_rot_mat(cr),
                        bodies[c],
                    );
                    if best.is_none_or(|(d, ..)| dist < d) {
                        best = Some((dist, c, local, face, feat, dist));
                    }
                }
                if let Some((dist, c, local, face, feat, _)) = best {
                    let near = dist < 8.0;
                    if touching || (near && (verr >= 1.0 || dv_real >= 50.0)) {
                        let cr = &from_tick.car_records[c];
                        ball_touch_rows.push((
                            verr,
                            face,
                            feat,
                            cr.is_flipping,
                            cr.is_on_ground,
                            sim_hit,
                            touching,
                            dist,
                            c,
                        ));
                        if std::env::var("GGL_BALL").is_ok() {
                            let cv: Vec3A = cr.phys.lin_vel.into();
                            let rel = (from_bv - cv).length();
                            eprintln!(
                                "BALLTOUCH {i} c{c} {surf} face={face} feat={feat} flip={} gnd={} wheels={} flag={} simhit={} verr={verr:8.2} dist={dist:7.1} rel={rel:7.1} dv={dv_real:7.1} local=({:.0},{:.0},{:.0})",
                                u8::from(cr.is_flipping),
                                u8::from(cr.is_on_ground),
                                cr.wheels.iter().filter(|w| w.has_contact).count(),
                                u8::from(touching),
                                u8::from(sim_hit),
                                local.x, local.y, local.z,
                            );
                        }
                    }
                }
                if std::env::var("GGL_BALL").is_ok() {
                    eprintln!(
                        "BALLERR {i} {surf} touch={} verr={verr:8.2} averr={averr:6.3} |rv|={:7.1} |pv|={:7.1} z={:6.0} dv_real={dv_real:7.1}",
                        u8::from(touching),
                        real_bv.length(),
                        bs.phys.vel.length(),
                        real_bp.z,
                    );
                }
            }
        }

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
            let from_vel: Vec3A = fr.phys.lin_vel.into();
            let sim_dv = pred.phys.vel - from_vel;
            let resid_uu = Vec3A::from(fr.residual_lin) * 50.0;
            let tape_car_imp = tape_lin_imp_uu(fr, ImpulseRecordType::CarImpact);
            let tape_ball_imp = tape_lin_imp_uu(fr, ImpulseRecordType::BallImpact);
            let tape_car = tape_has_imp(fr, ImpulseRecordType::CarImpact);
            let tape_ball = tape_has_imp(fr, ImpulseRecordType::BallImpact);
            let sim_car = step_events.iter().any(|e| {
                matches!(
                    e,
                    ArenaEvent::CarHitCar(h)
                        if h.bumper_car_idx == car_idx || h.victim_car_idx == car_idx
                )
            });
            let sim_ball = step_events.iter().any(|e| {
                matches!(e, ArenaEvent::CarHitBall(h) if h.car_idx == car_idx)
            });
            if tape_car {
                car_imp_tape_n += 1;
                car_imp_vel.push(vel_err);
                car_imp_applied.push((sim_dv - (tape_car_imp + resid_uu)).length());
                car_imp_missed.push((sim_dv - resid_uu).length());
            }
            if sim_car {
                car_imp_sim_n += 1;
            }
            if tape_car && sim_car {
                car_imp_both += 1;
            } else if tape_car {
                car_imp_tape_only += 1;
            } else if sim_car {
                car_imp_sim_only += 1;
            }
            if tape_ball {
                ball_imp_tape_n += 1;
                ball_imp_vel.push(vel_err);
                ball_imp_applied.push((sim_dv - (tape_ball_imp + resid_uu)).length());
                ball_imp_missed.push((sim_dv - resid_uu).length());
                if let Some(extra) = step_events.iter().find_map(|e| match e {
                    ArenaEvent::CarHitBall(h) if h.car_idx == car_idx => Some(h.extra_hit_vel),
                    _ => None,
                }) {
                    ball_extra_vs_tape.push((extra.length() - tape_ball_imp.length()).abs());
                }
            }
            if sim_ball {
                ball_imp_sim_n += 1;
            }
            if tape_ball && sim_ball {
                ball_imp_both += 1;
            } else if tape_ball {
                ball_imp_tape_only += 1;
            } else if sim_ball {
                ball_imp_sim_only += 1;
            }
            // GGL_OPP_FILTER=<uu>: drop this car's tick from ALL metrics when an
            // opponent is within the given distance. Car-car scrape/push contact
            // measured 2026-08-24 as the dominant partial-contact error tail (big-err
            // ticks med opp-dist 168 vs 1793 for small-err); the filter isolates the
            // solo-physics scoreboard while that surface goes unattacked. Default off.
            let opp_dist = {
                let p: Vec3A = fr.phys.pos.into();
                from_tick
                    .car_records
                    .iter()
                    .enumerate()
                    .filter(|&(oc, _)| oc != c)
                    .map(|(_, o)| (Vec3A::from(o.phys.pos) - p).length())
                    .fold(f32::INFINITY, f32::min)
            };
            if opp_dist < opp_filter {
                continue;
            }
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

            let extra = arena.get_car_extra_state(car_idx);
            let up_z = {
                let r = &fr.phys.rot;
                Vec3A::new(r.rows[0].z, r.rows[1].z, r.rows[2].z).z.abs()
            };
            let min_susp = fr
                .wheels
                .iter()
                .filter(|w| w.has_contact)
                .map(|w| w.susp_length)
                .fold(f32::INFINITY, f32::min);
            let packed = regime == "wall_drive"
                && wheels_touching == 4
                && min_susp < -6.0
                && up_z >= 0.25;
            if packed {
                packed_n += 1;
                packed_vel.push(vel_err);
                packed_tape_drive.push(fr.drive_torque);
                packed_tape_brake.push(fr.brake_torque);
                packed_d_out_thr.push(fr.output_throttle - fr.prev_controls.throttle);
                packed_d_out_steer.push(fr.output_steer - fr.prev_controls.steer);
                packed_d_out_brake.push(fr.output_brake - 0.0);
                packed_d_out_hb.push(fr.output_handbrake - f32::from(fr.prev_controls.handbrake));
                packed_d_sim_thr.push(fr.output_throttle - pred.controls.throttle);
                packed_d_sim_steer.push(fr.output_steer - pred.controls.steer);
                packed_d_sim_hb.push(fr.output_handbrake - pred.handbrake_val);
                let mut sum_fci_t = 0.0f32;
                let mut sum_fci_s = 0.0f32;
                let mut sum_push_d = 0.0f32;
                let mut sum_susp_d = 0.0f32;
                let mut nw = 0u32;
                let mut sample_n_ang = 0.0f32;
                for w in 0..4 {
                    if !fr.wheels[w].has_contact {
                        continue;
                    }
                    nw += 1;
                    let tw = &fr.wheels[w];
                    let sw = &extra.wheels[w];
                    sum_fci_t += tw.friction_curve_input;
                    sum_fci_s += sw.last_friction_curve_input;
                    sum_push_d += sw.extra_pushback - tw.extra_pushback;
                    let sim_susp = sw.suspension_length * rocketsim::consts::BT_TO_UU;
                    sum_susp_d += sim_susp - tw.susp_length;
                    packed_d_steer.push(sw.steer_angle - tw.steer_amount);
                    packed_tape_spin.push(tw.spin_speed.abs());
                    let tape_wvel = Vec3A::from(tw.wheel_linear_velocity);
                    packed_tape_wvel.push(tape_wvel.length());
                    if sw.has_raycast_info != 0 {
                        let sim_n = extra_vec(sw.contact_normal);
                        let tape_n = Vec3A::from(tw.contact_normal);
                        if let Some(deg) = unit_angle_deg(sim_n, tape_n) {
                            packed_n_ang.push(deg);
                            sample_n_ang = sample_n_ang.max(deg);
                        }
                        let sim_axle = extra_vec(sw.axle_dir);
                        let tape_lat = Vec3A::from(tw.lat_direction);
                        let tape_long = Vec3A::from(tw.long_direction);
                        if let Some(deg) = unsigned_angle_deg(sim_axle, tape_lat) {
                            packed_lat_ang.push(deg);
                        }
                        let sim_long = sim_axle.cross(sim_n);
                        if let Some(deg) = unsigned_angle_deg(sim_long, tape_long) {
                            packed_long_ang.push(deg);
                        }
                        let sim_pt = extra_vec(sw.contact_point) * rocketsim::consts::BT_TO_UU;
                        let tape_pt = Vec3A::from(tw.contact_location);
                        if tape_pt.length_squared() > 1.0 {
                            packed_pt_dist.push((sim_pt - tape_pt).length());
                            let rest_w =
                                wheel_local_to_world(fr, Vec3A::from(tw.preset_rest_position));
                            packed_rest_to_hit.push((rest_w - tape_pt).length());
                            if let Some(pred_pt) = tape_contact_from_rest(fr, tw) {
                                packed_pred_contact.push((pred_pt - tape_pt).length());
                                packed_sim_vs_pred.push((sim_pt - pred_pt).length());
                            }
                        }
                        let sim_wvel = extra_vec(sw.vel_at_contact_point) * rocketsim::consts::BT_TO_UU;
                        packed_d_wvel.push((tape_wvel - sim_wvel).length());
                        let r_uu = sw.wheels_radius * rocketsim::consts::BT_TO_UU;
                        packed_spin_r_vs_wvel.push((tw.spin_speed.abs() * r_uu - tape_wvel.length()).abs());
                    }
                }
                let nw = nw.max(1) as f32;
                packed_tape_fci.push(sum_fci_t / nw);
                packed_sim_fci.push(sum_fci_s / nw);
                packed_d_fci.push((sum_fci_s - sum_fci_t) / nw);
                packed_d_push.push(sum_push_d / nw);
                packed_d_susp.push(sum_susp_d / nw);
                if fillet_samples.len() < 8 {
                    fillet_samples.push(format!(
                        "FILLET i={i} c{c} verr={vel_err:.3} uz={up_z:.3} n_ang={sample_n_ang:.1} drv_t={:.0} brk_t={:.0} fci_t={:.3} fci_s={:.3} out_t={:+.2} out_s={:+.2} out_hb={:.2} thr={:+.1} st={:+.1}",
                        fr.drive_torque,
                        fr.brake_torque,
                        sum_fci_t / nw,
                        sum_fci_s / nw,
                        fr.output_throttle,
                        fr.output_steer,
                        fr.output_handbrake,
                        fr.prev_controls.throttle,
                        fr.prev_controls.steer,
                    ));
                }
            }
            if !packed
                && fr.is_on_ground
                && !on_wall
                && wheels_touching == 4
            {
                for w in 0..4 {
                    let tw = &fr.wheels[w];
                    let sw = &extra.wheels[w];
                    if !tw.has_contact || sw.has_raycast_info == 0 {
                        continue;
                    }
                    let sim_pt = extra_vec(sw.contact_point) * rocketsim::consts::BT_TO_UU;
                    let tape_pt = Vec3A::from(tw.contact_location);
                    if tape_pt.length_squared() <= 1.0 {
                        continue;
                    }
                    flat_sim_vs_tape.push((sim_pt - tape_pt).length());
                    if let Some(pred_pt) = tape_contact_from_rest(fr, tw) {
                        flat_sim_vs_pred.push((sim_pt - pred_pt).length());
                    }
                }
            }
            if regime == "air+wheels" && opp_dist >= 200.0 {
                aw_output_n += 1;
                aw_d_thr.push(fr.output_throttle - fr.prev_controls.throttle);
                aw_d_steer.push(fr.output_steer - fr.prev_controls.steer);
            }

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

            // GGL_PARTIAL_DECOMP: same signed axes for wall / air+wheels / flip+wheels.
            if std::env::var("GGL_PARTIAL_DECOMP").is_ok()
                && matches!(regime, "wall_drive" | "air+wheels" | "flip+wheels")
            {
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
                let min_nz = fr
                    .wheels
                    .iter()
                    .filter(|w| w.has_contact)
                    .map(|w| Vec3A::from(w.contact_normal).z.abs())
                    .fold(f32::INFINITY, f32::min);
                let min_susp = fr
                    .wheels
                    .iter()
                    .filter(|w| w.has_contact)
                    .map(|w| w.susp_length)
                    .fold(f32::INFINITY, f32::min);
                let v: Vec3A = fr.phys.lin_vel.into();
                let pc = &fr.prev_controls;
                let rf_mask = fr
                    .wheels
                    .iter()
                    .enumerate()
                    .fold(0u8, |m, (k, w)| m | (u8::from(w.has_contact) << k));
                let rt_mask = real
                    .wheels
                    .iter()
                    .enumerate()
                    .fold(0u8, |m, (k, w)| m | (u8::from(w.has_contact) << k));
                // Real contact mask one tick BEFORE the from-tick (gate-history
                // reconstruction; 0b1111+1 sentinel when unavailable).
                let pf_mask = if i > 0 && recording.ticks[i - 1].car_records.len() == num_cars {
                    recording.ticks[i - 1].car_records[c]
                        .wheels
                        .iter()
                        .enumerate()
                        .fold(0u8, |m, (k, w)| m | (u8::from(w.has_contact) << k))
                } else {
                    16
                };
                let odist = opp_dist;
                eprintln!(
                    "PDECOMP {regime} c{c} n={wheels_touching} up={:.4} fwd={:.4} lat={:.4} verr={vel_err:.4} spd={:.0} vz={:.0} nz={min_nz:.2} susp={min_susp:.3} thr={:+.1} st={:+.1} j={} b={} hb={} ft={:.3} sm={:04b} rf={rf_mask:04b} rt={rt_mask:04b} pf={pf_mask:05b} od={odist:.0} uz={:.3} i={i}",
                    dv.dot(up),
                    dv.dot(fwd),
                    dv.dot(lat),
                    v.length(),
                    v.z,
                    pc.throttle,
                    pc.steer,
                    u8::from(pc.jump),
                    u8::from(pc.boost),
                    u8::from(pc.handbrake),
                    fr.flip_time,
                    sim_wheel_masks[c],
                    up.z,
                );
            }

            // GGL_CARCAR=<uu>: opponent-frame error dump for ticks with an opponent
            // inside the threshold. Errors are decomposed along the car->opponent
            // axis (ax, + = sim closes on the opponent faster than real), the
            // horizontal perpendicular (pp), and world z (ez). cls = real closing
            // speed at the from-tick (+ = approaching), dz = opponent height minus
            // car height, bump = sim CarHitCar fired during this step.
            // n = tape wheel contacts; sn = sim wheel contacts at restore;
            // tnz = min |contact_normal.z| of tape wheels in contact (1 = floor);
            // wc/wm = sim wheels whose ray hit another car.
            // sg/sa/sk = last car-car SAT closest-axis gap (uu), axis, kind
            // (0 none, 1 miss, 2 hit, 3 clip-empty, 4 AABB miss, 5 slack-promoted).
            let carcar_thresh = {
                static V: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
                *V.get_or_init(|| {
                    std::env::var("GGL_CARCAR")
                        .ok()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.0)
                })
            };
            if carcar_thresh > 0.0 && opp_dist < carcar_thresh {
                let p: Vec3A = fr.phys.pos.into();
                let (oc, orec) = from_tick
                    .car_records
                    .iter()
                    .enumerate()
                    .filter(|&(ocx, _)| ocx != c)
                    .min_by(|a, b| {
                        let da = (Vec3A::from(a.1.phys.pos) - p).length();
                        let db = (Vec3A::from(b.1.phys.pos) - p).length();
                        da.total_cmp(&db)
                    })
                    .unwrap();
                let opos: Vec3A = orec.phys.pos.into();
                let ovel: Vec3A = orec.phys.lin_vel.into();
                let fvel: Vec3A = fr.phys.lin_vel.into();
                let axis3 = opos - p;
                let axis = axis3.normalize_or_zero();
                let dvv = pred.phys.vel - real_vel;
                let pp_dir = Vec3A::new(-axis.y, axis.x, 0.0).normalize_or_zero();
                let cls = (fvel - ovel).dot(axis);
                let bump = step_events.iter().any(|ev| {
                    matches!(ev, ArenaEvent::CarHitCar(e)
                        if e.bumper_car_idx as usize == car_idcs[c] || e.victim_car_idx as usize == car_idcs[c])
                });
                let tnz = fr
                    .wheels
                    .iter()
                    .filter(|w| w.has_contact)
                    .map(|w| Vec3A::from(w.contact_normal).z.abs())
                    .fold(f32::INFINITY, f32::min);
                let _ = oc;
                let sg = f32::from_bits(
                    rocketsim::LAST_SAT_GAP_UU.load(std::sync::atomic::Ordering::Relaxed),
                );
                let sa = rocketsim::LAST_SAT_AXIS.load(std::sync::atomic::Ordering::Relaxed);
                let sk = rocketsim::LAST_SAT_KIND.load(std::sync::atomic::Ordering::Relaxed);
                eprintln!(
                    "CARCAR c{c} od={opp_dist:.0} dz={:.0} ax={:+.3} pp={:+.3} ez={:+.3} verr={vel_err:.3} cls={cls:+.0} spd={:.0} n={wheels_touching} sn={} tnz={:.2} thr={:+.1} hb={} bump={} wc={} wm={:04b} sg={sg:+.2} sa={sa} sk={sk} i={i}",
                    opos.z - p.z,
                    dvv.dot(axis),
                    dvv.dot(pp_dir),
                    dvv.z,
                    fvel.length(),
                    sim_wheel_masks[c].count_ones(),
                    if tnz.is_finite() { tnz } else { -1.0 },
                    fr.prev_controls.throttle,
                    u8::from(fr.prev_controls.handbrake),
                    u8::from(bump),
                    wheel_car_masks[c].count_ones(),
                    wheel_car_masks[c],
                );
            }

            // GGL_BOOST: pad-pickup comparison. A real pickup is a boost INCREASE
            // between records; compare with whether the replay's arena granted one.
            if std::env::var("GGL_BOOST").is_ok() {
                let real_gain = (real.boost_amount - fr.boost_amount) * 100.0;
                let pred_gain = pred.boost - fr.boost_amount * 100.0;
                let real_pick = real_gain > 6.0;
                let sim_pick = pred_gain > 6.0;
                if real_pick || sim_pick {
                    let pos: Vec3A = fr.phys.pos.into();
                    eprintln!(
                        "BOOSTPICK {i} c{c} real={} sim={} gain_r={real_gain:6.1} gain_s={pred_gain:6.1} pos=({:7.1},{:7.1},{:5.1}) grounded={}",
                        u8::from(real_pick),
                        u8::from(sim_pick),
                        pos.x, pos.y, pos.z,
                        u8::from(fr.is_on_ground),
                    );
                }
            }

            // GGL_PADGEO: near-pad geometry dump (S44 boost audit). For every tick a
            // car's REAL position is within 400uu (2D) of a pad center, print the
            // trigger geometry exactly as boost_pad_grid computes it (closest point on
            // the ORIGIN-centred oriented box; offset variant too) plus the pad's sim
            // cooldown, real boost and both gains -- lets offline analysis classify
            // every real/sim pickup mismatch as geometry vs cooldown-desync vs timing.
            if std::env::var("GGL_PADGEO").is_ok() {
                let real_gain = (real.boost_amount - fr.boost_amount) * 100.0;
                let pred_gain = pred.boost - fr.boost_amount * 100.0;
                let cfg = rocketsim::CarBodyConfig::OCTANE;
                let half = cfg.hitbox_size * 0.5;
                let r = &fr.phys.rot;
                let fwd = Vec3A::new(r.rows[0].x, r.rows[1].x, r.rows[2].x);
                let rightax = Vec3A::new(r.rows[0].y, r.rows[1].y, r.rows[2].y);
                let up = Vec3A::new(r.rows[0].z, r.rows[1].z, r.rows[2].z);
                let cpos: Vec3A = fr.phys.pos.into();
                for (pi, pcfg) in pad_configs.iter().enumerate() {
                    let pp = pcfg.pos;
                    if pp.truncate().distance_squared(cpos.truncate()) > 400.0 * 400.0 {
                        continue;
                    }
                    let closest = |center: Vec3A| -> Vec3A {
                        let rel = pp - center;
                        let local =
                            Vec3A::new(rel.dot(fwd), rel.dot(rightax), rel.dot(up));
                        let cl = local.clamp(-half, half);
                        center + fwd * cl.x + rightax * cl.y + up * cl.z
                    };
                    // as-implemented (origin-centred) and offset-corrected variants
                    let c0 = closest(cpos);
                    let off = cfg.hitbox_pos_offset;
                    let c1 = closest(cpos + fwd * off.x + rightax * off.y + up * off.z);
                    let cd = arena.get_boost_pad_state(pi).cooldown;
                    eprintln!(
                        "PADGEO {i} c{c} p{pi} big={} d0={:6.1} z0={:6.1} d1={:6.1} z1={:6.1} cd={cd:5.2} boost={:5.1} gr={real_gain:5.1} gs={pred_gain:5.1}",
                        u8::from(pcfg.is_big),
                        pp.truncate().distance(c0.truncate()),
                        c0.z - pp.z,
                        pp.truncate().distance(c1.truncate()),
                        c1.z - pp.z,
                        fr.boost_amount * 100.0,
                    );
                }
            }

            // GGL_CARCAR: velocity error on car-proximity ticks (bump/demo pipeline).
            if std::env::var("GGL_CARCAR").is_ok() && num_cars == 2 {
                let other: Vec3A = from_tick.car_records[1 - c].phys.pos.into();
                let dist = (Vec3A::from(fr.phys.pos) - other).length();
                if dist < 200.0 {
                    eprintln!("CARCAR {i} c{c} dist={dist:6.1} verr={vel_err:8.2} aerr={ang_err:6.2}");
                }
            }

            if regime == "flip+wheels" {
                fw_n += 1;
                let nw = wheels_touching.min(4);
                fw_by_wheels[nw].push(vel_err);
                if on_wall {
                    fw_wall.push(vel_err);
                } else {
                    fw_floor.push(vel_err);
                }
                let ft = fr.flip_time;
                let (has_f, is_f, qt) = pre_flip[c];
                if is_f && !has_f {
                    fw_no_has_flip += 1;
                }
                let mut crossed = false;
                use rocketsim::consts::car::flip as flip_c;
                let pitch_old = ft >= 0.041;
                let pitch_new = qt >= flip_c::PITCH_CANCEL_MIN_TICKS;
                if pitch_old != pitch_new {
                    fw_pitch += 1;
                    crossed = true;
                }
                let zs_old = ft >= 0.15;
                let zs_new = qt >= flip_c::Z_DAMP_START_TICKS;
                if zs_old != zs_new {
                    fw_zstart += 1;
                    crossed = true;
                }
                let ze_old = ft < 0.21;
                let ze_new = qt < flip_c::Z_DAMP_END_TICKS;
                if ze_old != ze_new {
                    fw_zend += 1;
                    crossed = true;
                }
                let tq_old = ft < 0.65;
                let tq_new = qt < flip_c::TORQUE_TICKS;
                if tq_old != tq_new {
                    fw_torque += 1;
                    crossed = true;
                }
                let pl_old = ft < 0.95;
                let pl_new = qt < flip_c::PITCHLOCK_TICKS;
                if pl_old != pl_new {
                    fw_plock += 1;
                    crossed = true;
                }
                if crossed {
                    fw_err_cross.push(vel_err);
                    if fw_err_cross.len() <= 12 {
                        eprintln!(
                            "FLIPW_Q tick {i} c{c} ft={ft:.6} qticks={qt} verr={vel_err:.3} has_flipped={has_f} is_flipping={is_f} pitch {pitch_old}->{pitch_new} zend {ze_old}->{ze_new} torque {tq_old}->{tq_new}"
                        );
                    }
                } else {
                    fw_err_same.push(vel_err);
                }
            }

            let e = regimes.entry(regime).or_default();
            e[0].push(vel_err);
            e[1].push(pos_err);
            e[2].push(ang_err);
            regimes_by_car.entry((c, regime)).or_default().push(vel_err);
            if regime == "wall_drive" {
                wall_by_nw
                    .entry((c, wheels_touching))
                    .or_default()
                    .push(vel_err);
                let nz = fr
                    .wheels
                    .iter()
                    .filter(|w| w.has_contact)
                    .map(|w| Vec3A::from(w.contact_normal).z.abs())
                    .fold(1.0f32, f32::min);
                let nz_bin: &'static str = if nz < 0.2 {
                    "nz<0.2"
                } else if nz < 0.5 {
                    "nz<0.5"
                } else {
                    "nz<0.7"
                };
                wall_by_nz.entry((c, nz_bin)).or_default().push(vel_err);
            }
            if regime == "air+wheels" {
                aw_by_nw
                    .entry((c, wheels_touching))
                    .or_default()
                    .push(vel_err);
            }
            worst.push((vel_err, i, c, regime));
        }
    }

    {
        let mut med = |v: &mut Vec<f32>| -> f32 {
            if v.is_empty() {
                return f32::NAN;
            }
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        eprintln!(
            "=== flip+wheels restore quantization n={fw_n} pitch_cross={fw_pitch} zstart_cross={fw_zstart} zend_cross={fw_zend} torque_cross={fw_torque} plock_cross={fw_plock} is_flipping_without_has_flipped={fw_no_has_flip}"
        );
        eprintln!(
            "    floor n={} p50={}  wall n={} p50={}  wheels1 n={} p50={}  wheels2 n={} p50={}  wheels3+ n={} p50={}",
            fw_floor.len(),
            med(&mut fw_floor),
            fw_wall.len(),
            med(&mut fw_wall),
            fw_by_wheels[1].len(),
            med(&mut fw_by_wheels[1]),
            fw_by_wheels[2].len(),
            med(&mut fw_by_wheels[2]),
            fw_by_wheels[3].len() + fw_by_wheels[4].len(),
            {
                let mut v = fw_by_wheels[3].clone();
                v.extend_from_slice(&fw_by_wheels[4]);
                med(&mut v)
            },
        );
        eprintln!(
            "    vel p50 same-gate={} n={}  cross-gate={} n={}",
            med(&mut fw_err_same),
            fw_err_same.len(),
            med(&mut fw_err_cross),
            fw_err_cross.len(),
        );
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

    // GGL_BALL_EVENTS: phase-robust hit-power measurement. For each touch EVENT
    // (rising edge of any is_touching_ball), restore the full state 2 ticks before,
    // roll the sim through the event WITHOUT resyncing, and compare the outgoing ball
    // velocity 4 ticks after the touch flag clears. Reports speed ratio + direction.
    if std::env::var("GGL_BALL_EVENTS").is_ok() {
        let touching_at = |i: usize| {
            recording.ticks[i].car_records.len() == num_cars
                && (0..num_cars).any(|c| recording.ticks[i].car_records[c].is_touching_ball)
        };
        let mut i = 3;
        while i + 8 < recording.ticks.len() {
            if !(touching_at(i) && !touching_at(i - 1)) {
                i += 1;
                continue;
            }
            // find event end (touch flag clears for both cars)
            let mut end = i;
            while end + 1 < recording.ticks.len() && touching_at(end + 1) && end - i < 60 {
                end += 1;
            }
            let start = i - 2;
            let fin = (end + 4).min(recording.ticks.len() - 1);
            // roster/frames must be clean across the whole window
            let mut ok = true;
            for k in start..=fin {
                if recording.ticks[k].car_records.len() != num_cars {
                    ok = false;
                    break;
                }
            }
            if ok {
                // restore at `start`, then step to `fin` with lagged controls
                let lag = 2usize;
                let c0: Vec<CarControls> = (0..num_cars)
                    .map(|c| recording.ticks[start.saturating_sub(lag) + 1]
                        .car_records[c].prev_controls.into())
                    .collect();
                set_state_to_record_tick(&mut arena, &car_idcs, &recording.ticks[start], &c0, bakkes_semantics);
                for (c, &car_idx) in car_idcs.iter().enumerate() {
                    let mut cs = *arena.get_car_state(car_idx);
                    cs.boost = recording.ticks[start].car_records[c].boost_amount * 100.0;
                    arena.set_car_state(car_idx, cs);
                }
                let mut extra_sum = Vec3A::ZERO;
                for k in start..fin {
                    let ci = (k + 1).saturating_sub(lag);
                    if recording.ticks[ci].car_records.len() == num_cars {
                        for (c, &car_idx) in car_idcs.iter().enumerate() {
                            let mut cs = *arena.get_car_state(car_idx);
                            cs.controls =
                                recording.ticks[ci].car_records[c].prev_controls.into();
                            arena.set_car_state(car_idx, cs);
                        }
                    }
                    for ev in arena.step_tick().to_vec() {
                        if let ArenaEvent::CarHitBall(e) = ev {
                            extra_sum += e.extra_hit_vel;
                        }
                    }
                }
                let sim_v = arena.get_ball_state().phys.vel;
                let real_v: Vec3A = recording.ticks[fin].ball_record.lin_vel.into();
                let pre_v: Vec3A = recording.ticks[start].ball_record.lin_vel.into();
                let real_dv = (real_v - pre_v).length();
                if real_dv > 150.0 && real_v.length() > 100.0 {
                    let ratio = sim_v.length() / real_v.length();
                    let dir = sim_v
                        .normalize_or_zero()
                        .dot(real_v.normalize_or_zero())
                        .clamp(-1.0, 1.0)
                        .acos()
                        .to_degrees();
                    let bz = recording.ticks[i].ball_record.pos.z;
                    let bp: Vec3A = recording.ticks[i].ball_record.pos.into();
                    let surf = classify_ball_surface(bp);
                    // relative car-ball speed just before contact (max over cars)
                    let bv0: Vec3A = recording.ticks[i - 1].ball_record.lin_vel.into();
                    let rel = (0..num_cars)
                        .map(|c| {
                            let cv: Vec3A =
                                recording.ticks[i - 1].car_records[c].phys.lin_vel.into();
                            (bv0 - cv).length()
                        })
                        .fold(0.0f32, f32::max);
                    eprintln!(
                        "BALLEVT {i} surf={surf} len={} ratio={ratio:6.3} dir={dir:6.2} |real|={:7.1} |sim|={:7.1} ballz={bz:6.0} dv={real_dv:6.0} rel={rel:6.0} extra=({:.1},{:.1},{:.1}) sim=({:.1},{:.1},{:.1}) realv=({:.1},{:.1},{:.1}) prev=({:.1},{:.1},{:.1})",
                        end - i + 1,
                        real_v.length(),
                        sim_v.length(),
                        extra_sum.x, extra_sum.y, extra_sum.z,
                        sim_v.x, sim_v.y, sim_v.z,
                        real_v.x, real_v.y, real_v.z,
                        pre_v.x, pre_v.y, pre_v.z,
                    );
                }
            }
            i = end + 1;
        }
    }

    // GGL_JUMPLIFT: when does the car actually LEAVE THE GROUND after a jump press?
    // Known RocketSim v2 defect: a minimum (3-tick) jump from rest lifted off one tick
    // early vs the real game. Find real jump presses from a grounded, near-stationary
    // car, restore the full state there, roll the sim forward WITHOUT resyncing, and
    // compare the first airborne tick (0 wheels in contact) sim vs real.
    if std::env::var("GGL_JUMPLIFT").is_ok() {
        const ROLL: usize = 24;
        let lag = 2usize;
        let mut rows: Vec<(i32, i32, usize, usize)> = Vec::new(); // (sim_lift, real_lift, hold_ticks, tick)
        for i in 4..recording.ticks.len().saturating_sub(ROLL + 4) {
            let ok = (i - 4..=i + ROLL + 2)
                .all(|k| recording.ticks[k].car_records.len() == num_cars);
            if !ok {
                continue;
            }
            for c in 0..num_cars {
                let jr = |k: usize| recording.ticks[k].car_records[c].prev_controls.jump;
                // rising edge of the jump input in the control stream
                if !(jr(i) && !jr(i - 1)) {
                    continue;
                }
                // Reject the kickoff countdown: RL ignores inputs there, so the recorded
                // jump spam produces phantom "presses" whose real lift-off is just the
                // countdown ending (this fired 205 bogus events before the filter).
                let bp: Vec3A = recording.ticks[i].ball_record.pos.into();
                let bv: Vec3A = recording.ticks[i].ball_record.lin_vel.into();
                if bp.x.abs() < 1.0 && bp.y.abs() < 1.0 && bv.length_squared() == 0.0 {
                    continue;
                }
                // Require a CLEAN edge: no jump input for the preceding 8 ticks.
                if (1..=8).any(|b| jr(i - b.min(i))) {
                    continue;
                }
                let cr = &recording.ticks[i].car_records[c];
                let v: Vec3A = cr.phys.lin_vel.into();
                let grounded = cr.wheels.iter().filter(|w| w.has_contact).count() == 4;
                let vmax: f32 = std::env::var("GGL_JUMPLIFT_VMAX").ok().and_then(|x| x.parse().ok()).unwrap_or(150.0);
                if !grounded || v.length() > vmax {
                    continue;
                }
                // how many ticks is the jump held?
                let mut hold = 1usize;
                while hold < 12 && jr(i + hold) {
                    hold += 1;
                }
                // restore at the tick the press first ACTS (press + lag) minus 1
                let start = i + lag - 1;
                if start + ROLL + 1 >= recording.ticks.len() {
                    continue;
                }
                let c0: Vec<CarControls> = (0..num_cars)
                    .map(|q| recording.ticks[start - lag + 1].car_records[q].prev_controls.into())
                    .collect();
                set_state_to_record_tick(&mut arena, &car_idcs, &recording.ticks[start], &c0, bakkes_semantics);
                for (q, &ci) in car_idcs.iter().enumerate() {
                    let mut cs = *arena.get_car_state(ci);
                    cs.boost = recording.ticks[start].car_records[q].boost_amount * 100.0;
                    arena.set_car_state(ci, cs);
                }
                let (mut sim_lift, mut real_lift) = (-1i32, -1i32);
                for k in 0..ROLL {
                    let ct = (start + k + 1).saturating_sub(lag);
                    for (q, &ci) in car_idcs.iter().enumerate() {
                        let mut cs = *arena.get_car_state(ci);
                        cs.controls = recording.ticks[ct].car_records[q].prev_controls.into();
                        arena.set_car_state(ci, cs);
                    }
                    arena.step_tick();
                    let sim_air = arena.get_car_state(car_idcs[c]).wheels_with_contact
                        .iter().filter(|w| **w).count() == 0;
                    let real_air = recording.ticks[start + k + 1].car_records[c]
                        .wheels.iter().filter(|w| w.has_contact).count() == 0;
                    if sim_air && sim_lift < 0 {
                        sim_lift = k as i32;
                    }
                    if real_air && real_lift < 0 {
                        real_lift = k as i32;
                    }
                }
                // The real car must actually leave the ground promptly; otherwise the
                // input was ignored or overridden and this is not a jump event.
                if sim_lift >= 0 && (0..=12).contains(&real_lift) {
                    rows.push((sim_lift, real_lift, hold, i));
                }
            }
        }
        let mut hist: std::collections::BTreeMap<i32, usize> = std::collections::BTreeMap::new();
        let mut hist3: std::collections::BTreeMap<i32, usize> = std::collections::BTreeMap::new();
        for &(s, r, hold, _) in &rows {
            *hist.entry(s - r).or_default() += 1;
            if hold <= 3 {
                *hist3.entry(s - r).or_default() += 1;
            }
        }
        eprintln!("\n=== JUMPLIFT: sim-minus-real first-airborne tick ===");
        eprintln!("events: {} (all holds)  offset histogram: {hist:?}", rows.len());
        let n3: usize = hist3.values().sum();
        eprintln!("minimum jumps (hold <= 3 ticks): {n3}  offset histogram: {hist3:?}");
        for &(s, r, hold, t) in rows.iter().filter(|r| r.2 <= 3).take(12) {
            eprintln!("  tick {t:6} hold={hold} sim_lift={s} real_lift={r} diff={}", s - r);
        }
    }

    eprintln!("skipped {skipped_gaps} gap transitions");

    // Tape-only flip-reset hold: 3+ wheels with contact and NOT world geometry.
    {
        let mut holds: std::collections::BTreeMap<u32, usize> = Default::default();
        let mut events = 0usize;
        let mut tick_reset = 0usize;
        let mut n_on = 0u32;
        for tick in recording.ticks[tick_lo..recording.ticks.len().min(tick_hi)].iter() {
            for cr in &tick.car_records {
                let n_ball = cr
                    .wheels
                    .iter()
                    .filter(|w| w.has_contact && !w.has_world_geometry_contact)
                    .count() as u32;
                if n_ball >= 3 {
                    tick_reset += 1;
                    n_on += 1;
                } else if n_on > 0 {
                    *holds.entry(n_on).or_default() += 1;
                    events += 1;
                    n_on = 0;
                }
            }
        }
        if n_on > 0 {
            *holds.entry(n_on).or_default() += 1;
            events += 1;
        }
        eprintln!("\n=== flip-reset tape hold (3+ wheels on ball) ===");
        eprintln!("events={events} ticks_in_reset={tick_reset} hold_hist={holds:?}");
    }

    eprintln!("\n=== packed fillet (wall 4wh susp<-6 |uz|>=0.25) ===");
    if packed_n == 0 {
        eprintln!("n=0 (no packed-fillet ticks in window)");
    } else {
        eprintln!("n={packed_n}");
        eprintln!("vel          {}", packed_vel.summary());
        eprintln!("tape fci     {}", packed_tape_fci.summary());
        eprintln!("sim fci      {}", packed_sim_fci.summary());
        eprintln!("sim-tape fci {}", packed_d_fci.summary());
        eprintln!("n angle deg  {}  (sim vs tape contact_normal)", packed_n_ang.summary());
        eprintln!("lat ang deg  {}  (sim axle vs tape lat_dir, unsigned)", packed_lat_ang.summary());
        eprintln!("long ang deg {}  (sim axle×n vs tape long_dir, unsigned)", packed_long_ang.summary());
        eprintln!("contact dist {}  (sim hit×50 vs tape contact_location, uu)", packed_pt_dist.summary());
        eprintln!("rest→hit     {}  (world(preset_rest) vs contact_location; ≈ radius on flat)", packed_rest_to_hit.summary());
        eprintln!("pred contact {}  (world(rest) − susp·up − r n vs tape contact_location)", packed_pred_contact.summary());
        eprintln!("sim vs pred  {}  (sim hit×50 vs reconstructed tape contact)", packed_sim_vs_pred.summary());
        eprintln!("d steer      {}  (sim steer_angle − tape steer_amount)", packed_d_steer.summary());
        eprintln!("tape |spin|  {}  (no sim wheel ω)", packed_tape_spin.summary());
        eprintln!("tape |wvel|  {}", packed_tape_wvel.summary());
        eprintln!("|wvel-sim v| {}  (tape wheel linvel vs chassis vel at contact ×50)", packed_d_wvel.summary());
        eprintln!("|spin·r−|v|| {}  (tape; r from sim wheel radius)", packed_spin_r_vs_wvel.summary());
        eprintln!("tape drive   {}  (GetDriveTorque; not sim engine_force)", packed_tape_drive.summary());
        eprintln!("tape brake   {}  (GetBrakeTorque)", packed_tape_brake.summary());
        eprintln!("out-press thr{}", packed_d_out_thr.summary());
        eprintln!("out-press st {}", packed_d_out_steer.summary());
        eprintln!("output brake {}", packed_d_out_brake.summary());
        eprintln!("out-press hb {}", packed_d_out_hb.summary());
        eprintln!("out-sim thr  {}", packed_d_sim_thr.summary());
        eprintln!("out-sim st   {}", packed_d_sim_steer.summary());
        eprintln!("out-sim hb   {}", packed_d_sim_hb.summary());
        eprintln!("sim-tape susp{}  (units; not a spring fit)", packed_d_susp.summary());
        eprintln!("sim-tape push{}  (tape extra_pushback is always 0)", packed_d_push.summary());
        for s in &fillet_samples {
            eprintln!("{s}");
        }
    }
    eprintln!("\n=== flat ground contact convention (4wh, not wall) ===");
    eprintln!("sim vs tape loc {}", flat_sim_vs_tape.summary());
    eprintln!("sim vs pred     {}  (same reconstruction as packed)", flat_sim_vs_pred.summary());

    eprintln!("\n=== tape vs sim hit events (restore-1-step, per car-tick) ===");
    eprintln!(
        "CarImpact  tape={car_imp_tape_n} sim_CarHitCar={car_imp_sim_n} both={car_imp_both} tape_only={car_imp_tape_only} sim_only={car_imp_sim_only}"
    );
    eprintln!(
        "BallImpact tape={ball_imp_tape_n} sim_CarHitBall={ball_imp_sim_n} both={ball_imp_both} tape_only={ball_imp_tape_only} sim_only={ball_imp_sim_only}"
    );
    eprintln!("car vel_err on CarImpact  {}", car_imp_vel.summary());
    eprintln!("car vel_err on BallImpact {}", ball_imp_vel.summary());
    eprintln!(
        "CarImpact |sim_dv − (tape_imp+resid)| {}  (small ⇒ sim applied a bump-like Δv)",
        car_imp_applied.summary()
    );
    eprintln!(
        "CarImpact |sim_dv − resid|            {}  (small ⇒ sim looks like it missed the bump)",
        car_imp_missed.summary()
    );
    eprintln!(
        "BallImpact |sim_dv − (tape_imp+resid)| {}",
        ball_imp_applied.summary()
    );
    eprintln!(
        "BallImpact |sim_dv − resid|            {}",
        ball_imp_missed.summary()
    );
    eprintln!(
        "|sim ball extra_hit − tape BallImpact| {}  (ball extra vs car Δv; different bodies)",
        ball_extra_vs_tape.summary()
    );

    eprintln!("\n=== air+wheels output-vs-press (od>=200) ===");
    if aw_output_n == 0 {
        eprintln!("n=0");
    } else {
        eprintln!("n={aw_output_n}");
        eprintln!("d throttle   {}", aw_d_thr.summary());
        eprintln!("d steer      {}", aw_d_steer.summary());
    }

    eprintln!("\n=== ball surface 1-tick VELOCITY error (uu/s) ===");
    for name in BALL_SURFACES {
        let stats = ball_regimes.entry(name).or_default();
        let summary = stats[0].summary();
        let v = &stats[0].vals;
        let n = v.len().max(1) as f32;
        let frac = |t: f32| 100.0 * v.iter().filter(|&&x| x < t).count() as f32 / n;
        let (f1, f5, f23) = (frac(1.0), frac(5.0), frac(23.0));
        eprintln!("{name:14} {summary}  <1uu/s:{f1:5.1}%  <5:{f5:5.1}%  <23(1%vmax):{f23:5.1}%");
    }
    eprintln!("=== ball surface 1-tick ANG VEL error (rad/s) ===");
    for name in BALL_SURFACES {
        let stats = ball_regimes.entry(name).or_default();
        eprintln!("{name:14} {}", stats[1].summary());
    }
    if !ball_touch_rows.is_empty() {
        let n = ball_touch_rows.len();
        let mass: f32 = ball_touch_rows.iter().map(|r| r.0).sum();
        eprintln!("\n=== car-ball ticks (flag or near+impulse) n={n} sum(verr)={mass:.1} ===");
        let dump = |title: &str, key: &dyn Fn(&(f32, &str, &str, bool, bool, bool, bool, f32, usize)) -> String| {
            let mut g: std::collections::BTreeMap<String, (usize, f32, f32)> =
                std::collections::BTreeMap::new();
            for r in &ball_touch_rows {
                let e = g.entry(key(r)).or_default();
                e.0 += 1;
                e.1 += r.0;
                e.2 = e.2.max(r.0);
            }
            let mut rows: Vec<_> = g.into_iter().collect();
            rows.sort_by(|a, b| b.1.1.partial_cmp(&a.1.1).unwrap());
            eprintln!("--- {title} ---");
            eprintln!(
                "{:22} {:>7} {:>6} {:>10} {:>7} {:>8} {:>9}",
                "key", "n", "n%", "mass", "mass%", "mean", "max"
            );
            for (k, (cn, cm, mx)) in rows {
                eprintln!(
                    "{k:22} {cn:7} {:>5.1}% {cm:10.1} {:>6.1}% {:>8.1} {mx:9.1}",
                    100.0 * cn as f32 / n as f32,
                    100.0 * cm / mass,
                    cm / cn as f32,
                );
            }
        };
        dump("face", &|r| r.1.to_string());
        dump("feature", &|r| r.2.to_string());
        dump("flip", &|r| if r.3 { "flipping".into() } else { "not-flip".into() });
        dump("grounded", &|r| if r.4 { "on_ground".into() } else { "airborne".into() });
        dump("sim CarHitBall", &|r| if r.5 { "sim-hit".into() } else { "sim-miss".into() });
        dump("tape flag", &|r| if r.6 { "flag".into() } else { "no-flag".into() });
        dump("face x flip", &|r| {
            format!("{}|{}", r.1, if r.3 { "flip" } else { "nof" })
        });
        dump("face x simhit", &|r| {
            format!("{}|{}", r.1, if r.5 { "hit" } else { "miss" })
        });
        dump("flip x simhit", &|r| {
            format!(
                "{}|{}",
                if r.3 { "flip" } else { "nof" },
                if r.5 { "hit" } else { "miss" }
            )
        });
        dump("flag x simhit", &|r| {
            format!(
                "{}|{}",
                if r.6 { "flag" } else { "noflag" },
                if r.5 { "hit" } else { "miss" }
            )
        });
    }
    eprintln!("\n=== per-regime single-tick VELOCITY error (uu/s) ===");
    for (name, stats) in &mut regimes {
        let summary = stats[0].summary();
        let v = &stats[0].vals;
        let n = v.len().max(1) as f32;
        let frac = |t: f32| 100.0 * v.iter().filter(|&&x| x < t).count() as f32 / n;
        let (f1, f5, f23) = (frac(1.0), frac(5.0), frac(23.0));
        eprintln!("{name:14} {summary}  <1uu/s:{f1:5.1}%  <5:{f5:5.1}%  <23(1%vmax):{f23:5.1}%");
    }
    eprintln!("\n=== VELOCITY error by car (uu/s) ===");
    for ((c, name), stat) in &mut regimes_by_car {
        eprintln!("c{c} {name:14} {}", stat.summary());
    }
    eprintln!("\n=== wall_drive by car, n_wheels (uu/s) ===");
    for ((c, nw), stat) in &mut wall_by_nw {
        eprintln!("c{c} nw={nw} {}", stat.summary());
    }
    eprintln!("\n=== wall_drive by car, min |n.z| (uu/s) ===");
    for ((c, bin), stat) in &mut wall_by_nz {
        eprintln!("c{c} {bin} {}", stat.summary());
    }
    eprintln!("\n=== air+wheels by car, n_wheels (uu/s) ===");
    for ((c, nw), stat) in &mut aw_by_nw {
        eprintln!("c{c} nw={nw} {}", stat.summary());
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

#[test]
fn ball_surface_classifies_wiki_geometry() {
    assert_eq!(
        classify_ball_surface(Vec3A::new(0.0, 0.0, ball::REST_Z)),
        "ground"
    );
    assert_eq!(classify_ball_surface(Vec3A::new(0.0, 0.0, 500.0)), "air");
    // Floor-wall fillet: near side wall on the ramp, not ground.
    assert_eq!(
        classify_ball_surface(Vec3A::new(4000.0, 0.0, ball::REST_Z)),
        "wall"
    );
    assert_eq!(
        classify_ball_surface(Vec3A::new(4005.0, 0.0, 400.0)),
        "wall"
    );
    assert_eq!(
        classify_ball_surface(Vec3A::new(
            goal::SOCCAR_GOAL_HALF_WIDTH,
            arena::SOCCAR_EXTENT_Y,
            300.0
        )),
        "post"
    );
    assert_eq!(
        classify_ball_surface(Vec3A::new(
            0.0,
            arena::SOCCAR_EXTENT_Y,
            goal::SOCCAR_GOAL_HEIGHT
        )),
        "crossbar"
    );
}
