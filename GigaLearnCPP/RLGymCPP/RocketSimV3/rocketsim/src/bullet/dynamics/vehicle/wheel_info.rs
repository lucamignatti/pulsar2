use std::sync::OnceLock;

use glam::{Affine3A, Quat, Vec3A};

/// Tuned apply-time vehicle: friction, pushback, and suspension rel-vel are taken
/// when impulses are applied (after `update_wheels` wrote this tick's engine force),
/// not when the ray was cast. Combines with `GGL_WHEELS_PRE` to also apply before
/// the Bullet step (tuned's full order).
/// DEFAULT ON since 2026-08-23 (SIM2REAL_LOOP.md winning stack); `=0` disables.
pub(crate) fn ggl_apply_time() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("GGL_APPLY_TIME").map_or(true, |s| s != "0"))
}

/// Apply wheel suspension/friction/pushback as a Newton pair when the ray hits
/// another car. T5 (2026-08-24): roof-ride ticks ARE wheel-on-car (wc=2) but
/// enabling this was a no-op/slight regression (od<130 p90 15.07→16.43);
/// grind 4182 is NOT wheel-on-car (tnz=1, wc=0, SAT miss). Default OFF.
pub(crate) fn ggl_wheel_on_car() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("GGL_WHEEL_ON_CAR").is_ok_and(|s| s != "0"))
}

fn ggl_no_pushback() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("GGL_NO_PUSHBACK").is_ok_and(|s| s != "0"))
}

/// Partial-contact (<3 wheels) force-scaling mode for `GGL_LAT_PARTIAL` /
/// `GGL_BRAKE_PARTIAL` -- see the comment in `calc_friction_impulses`.
#[derive(Clone, Copy)]
pub(crate) enum PartialMode {
    Off,
    N4,
    N3,
    Fixed(f32),
}

fn parse_partial(var: &str) -> PartialMode {
    match std::env::var(var).as_deref() {
        Ok("n4") | Ok("norm") => PartialMode::N4,
        Ok("n3") => PartialMode::N3,
        Ok(s) => s.parse().map_or(PartialMode::Off, PartialMode::Fixed),
        Err(_) => PartialMode::Off,
    }
}

pub(crate) fn ggl_lat_partial() -> PartialMode {
    static V: OnceLock<PartialMode> = OnceLock::new();
    *V.get_or_init(|| parse_partial("GGL_LAT_PARTIAL"))
}

pub(crate) fn ggl_brake_partial() -> PartialMode {
    static V: OnceLock<PartialMode> = OnceLock::new();
    *V.get_or_init(|| parse_partial("GGL_BRAKE_PARTIAL"))
}

/// Friction-cone mu for the lateral load cap (see `calc_friction_impulses`). 0 = off.
pub(crate) fn ggl_lat_loadcap() -> f32 {
    static V: OnceLock<f32> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("GGL_LAT_LOADCAP")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0)
    })
}

/// 0 = uncapped (tuned). DEFAULT UNCAPPED since 2026-08-23: the S39 cap (48) was fit
/// on the post-step stack; on the apply-time stack the resolve sees post-suspension
/// velocities and the cap was CAUSING the fillet/ceiling tails (SIM2REAL_LOOP.md
/// 22:00). Set `GGL_PUSHBACK_CAP=48` to restore the old S39 behavior.
fn ggl_pushback_cap() -> f32 {
    static V: OnceLock<f32> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("GGL_PUSHBACK_CAP")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0)
    })
}

/// Skip the apply-time extra_pushback *recompute* unless the DEEPEST contact
/// is at least this many UU inside rest+radius. All-or-nothing per car
/// (per-wheel gating lopsided T6). Default 0 = off: zero-then-recompute is
/// the shipped T6 fix. `=0.5` taxes T1/T7 wall p90. `GGL_PUSHBACK_MINPEN`.
pub(crate) fn ggl_pushback_minpen_bt() -> f32 {
    static V: OnceLock<f32> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("GGL_PUSHBACK_MINPEN")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0)
            * UU_TO_BT
    })
}

/// Chassis `|up.z|` below this counts as "on a wall" for `GGL_PUSHBACK_MINPEN`.
/// Default 0.5: floor/ceiling landings never skip. `GGL_PUSHBACK_MINPEN_UZ`.
pub(crate) fn ggl_pushback_minpen_uz() -> f32 {
    static V: OnceLock<f32> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("GGL_PUSHBACK_MINPEN_UZ")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.5)
    })
}

/// DEFAULT ON since 2026-08-23: the graze-damp fade (S39 companion) hurt air/flip on
/// the pre-step apply-time stack. `GGL_NO_DAMP_FADE=0` restores the fade.
fn ggl_no_damp_fade() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("GGL_NO_DAMP_FADE").map_or(true, |s| s != "0"))
}

fn ggl_no_detect_extra() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("GGL_NO_DETECT_EXTRA").is_ok_and(|s| s != "0"))
}

/// Tuned contact model: ray is exactly rest+travel+radius (no SUSPENSION_SUBTRACTION,
/// no adhesion-only DETECT_EXTRA band), and every hit produces full suspension and
/// friction forces. Tuned measured 317 vs 353 failing ticks in favor of no subtraction.
/// DEFAULT ON since 2026-08-23; `=0` restores the adhesion-band model.
pub(crate) fn ggl_tuned_ray() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("GGL_TUNED_RAY").map_or(true, |s| s != "0"))
}

/// Tuned RS_TUNE_STEER_AT_APPLY (ships on): recompute the friction axle from THIS
/// tick's steer_angle at apply time instead of the raycast-time axle (1 tick stale).
/// DEFAULT ON since 2026-08-23; `=0` restores the stale-axle behavior.
fn ggl_steer_at_apply() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("GGL_STEER_AT_APPLY").map_or(true, |s| s != "0"))
}

/// Project the friction axle onto the contact plane (Bullet default). Shipped
/// PR73 uses the raw axle, which on a fillet has a normal component so the
/// bilateral lat solve leaks into/out of the surface. Floor and true walls
/// are n ⟂ axle already (no-op). `GGL_PROJ_AXLE=1` enables.
/// Rejected 2026-08-24: T7 packed p90 3.39→3.04, air+wheels 0.91→7.85.
fn ggl_proj_axle() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("GGL_PROJ_AXLE").is_ok_and(|s| s != "0"))
}

/// Blend suspension/pushback direction from contact-normal toward car-up.
/// 0 = shipped (along n). 1 = along chassis up. Fillet mesh n is faceted
/// (T7 packed nz bins 0.10/0.30/0.48/0.64) so spring-along-n kicks car-lat
/// every triangle; true walls are n ≈ up (near no-op).
/// Rejected 2026-08-24: T7 packed 0.89/3.39 → 5.63/17.6 (along-n is load-bearing).
fn ggl_susp_up() -> f32 {
    static V: OnceLock<f32> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("GGL_SUSP_UP")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0)
    })
}

/// Packed-fillet leftover: 4-wheel `|up.z|∈[0.25,0.90]` ticks have faceted mesh
/// normals. Blend each wheel's contact normal toward the 4-wheel mean for
/// *suspension and extra_pushback only* (friction stays per-facet).
/// `GGL_FILLET_MEAN_N=0` off (default); `=1` full mean; `=0.25` lerp.
pub(crate) fn ggl_fillet_mean_n() -> f32 {
    static V: OnceLock<f32> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("GGL_FILLET_MEAN_N")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0)
    })
}

/// Cap on `1 / (contact_normal · car_up)` applied to spring and damper.
/// Bullet default is 10 (denom > 0.1). `GGL_INV_DOT_CAP=1` kills the 1/cos
/// boost that leaks spring into car-lat on a slightly rolled wall. Default 10
/// is a no-op vs current.
fn ggl_inv_dot_cap() -> f32 {
    static V: OnceLock<f32> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("GGL_INV_DOT_CAP")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(10.0)
    })
}

/// Divide hard-contact pushback by this (default 4 = NUM_WHEELS).
fn ggl_pushback_div() -> f32 {
    static V: OnceLock<f32> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("GGL_PUSHBACK_DIV")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(NUM_WHEELS as f32)
    })
}

fn ggl_pushback_ncontact() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("GGL_PUSHBACK_NCONTACT").is_ok_and(|s| s != "0"))
}

fn ggl_pushback_min_wheels() -> usize {
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("GGL_PUSHBACK_MIN_WHEELS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0)
    })
}

/// Packed fillet (`n_contact==4`, `|up.z|≥0.25`): scale extra_pushback. Default 1.
fn ggl_packed_push_scale() -> f32 {
    static V: OnceLock<f32> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("GGL_PACKED_PUSH_SCALE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0)
    })
}

/// Packed fillet: cap `1/(n·up)` used by spring/damper. 0 = off (keep ray inv_dot).
fn ggl_packed_inv_dot_cap() -> f32 {
    static V: OnceLock<f32> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("GGL_PACKED_INV_DOT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0)
    })
}

/// Packed fillet: scale spring+damper force (not extra_pushback). Default 1.
fn ggl_packed_susp_scale() -> f32 {
    static V: OnceLock<f32> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("GGL_PACKED_SUSP_SCALE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0)
    })
}

fn pushback_divisor(n_contact: usize) -> f32 {
    if ggl_pushback_ncontact() {
        n_contact.max(1) as f32
    } else {
        ggl_pushback_div().max(1.0)
    }
}

use super::{NUM_WHEELS, raycaster::VehicleRaycasterResult};
use crate::bullet::dynamics::rigid_body::Impulse;
use crate::{
    bullet::{
        dynamics::{
            constraint_solver::contact_constraint::{
                resolve_single_bilateral, resolve_single_collision,
            },
            rigid_body::RigidBody,
        },
        linear_math::QuatExt,
    },
    consts::{UU_TO_BT, bullet_vehicle},
    sim::UserInfoTypes,
};

fn susp_impulse_dir(chassis: &RigidBody, contact_normal: Vec3A) -> Vec3A {
    let t = ggl_susp_up();
    if t <= 0.0 {
        contact_normal
    } else if t >= 1.0 {
        chassis.get_world_trans().matrix3.z_axis
    } else {
        let up = chassis.get_world_trans().matrix3.z_axis;
        (contact_normal * (1.0 - t) + up * t).normalize_or_zero()
    }
}

pub struct RaycastInfo {
    pub contact_normal: Vec3A,
    pub contact_point: Vec3A,
    pub suspension_length: f32,
    pub impulse: Vec3A,
    pub is_in_contact_with_world: bool,
    pub clipped_inv_contact_dot_suspension: f32,
    pub suspension_relative_vel: f32,
    pub ground_body_idx: usize,
    pub wheel_trace_len: f32,
}

pub struct WheelInfo {
    pub raycast_info: Option<RaycastInfo>,
    /// Contact as of the previous tick's pre-pass (touchdown-exception bookkeeping).
    pub was_in_contact_prev_tick: bool,
    /// A surface is within the EXTENDED probe but outside the suspension's working range.
    /// Such a wheel feeds the sticky-force gate (adhesion) but gets no `raycast_info`, so
    /// it generates no friction, drive or suspension force. See SIM2REAL_AUDIT.md S29/S30.
    pub adhesion_contact: bool,
    pub hard_point: Vec3A,
    pub axle_dir: Vec3A,
    pub chassis_connection_point_cs: Vec3A,
    pub suspension_rest_length_1: f32,
    pub wheels_radius: f32,
    pub engine_force: f32,
    pub brake: f32,
    pub steer_angle: f32,
    pub vel_at_contact_point: Vec3A,
    pub lat_friction: f32,
    pub long_friction: f32,
    pub suspension_force_scale: f32,
    pub extra_pushback: f32,
    pub real_ray_length: f32,
    /// Last `friction_curve_input` written by `apply_friction_curves` (0 if no contact).
    pub last_friction_curve_input: f32,
    /// One-tick suspension/pushback normal override (analyze `GGL_TAPE_N`). Friction
    /// still uses the raycast normal. Cleared at the end of `update_vehicle_second`.
    pub susp_n_override: Option<Vec3A>,
}

impl WheelInfo {
    pub const DEFAULT: Self = Self {
        raycast_info: None,
        was_in_contact_prev_tick: false,
        adhesion_contact: false,
        hard_point: Vec3A::ZERO,
        axle_dir: Vec3A::ZERO,
        chassis_connection_point_cs: Vec3A::ZERO,
        suspension_rest_length_1: 0.0,
        wheels_radius: 0.0,
        engine_force: 0.0,
        brake: 0.0,
        steer_angle: 0.0,
        vel_at_contact_point: Vec3A::ZERO,
        lat_friction: 1.0,
        long_friction: 1.0,
        suspension_force_scale: 1.0,
        extra_pushback: 0.0,
        real_ray_length: 0.0,
        last_friction_curve_input: 0.0,
        susp_n_override: None,
    };

    pub(crate) fn susp_n(&self, mean_n: Option<Vec3A>) -> Option<Vec3A> {
        let ray = self.raycast_info.as_ref()?;
        if let Some(tape) = self.susp_n_override {
            return Some(tape.normalize_or_zero());
        }
        let t = ggl_fillet_mean_n().clamp(0.0, 1.0);
        if t > 0.0 {
            if let Some(mean) = mean_n {
                return Some((ray.contact_normal * (1.0 - t) + mean * t).normalize_or_zero());
            }
        }
        Some(ray.contact_normal)
    }

    pub const fn set_params(
        &mut self,
        chassis_connection_cs: Vec3A,
        suspension_rest_length: f32,
        wheel_radius: f32,
        suspsension_force_scale: f32,
    ) {
        self.chassis_connection_point_cs = chassis_connection_cs;
        self.suspension_rest_length_1 = suspension_rest_length;
        self.wheels_radius = wheel_radius;
        self.suspension_force_scale = suspsension_force_scale;

        let suspension_travel = bullet_vehicle::MAX_SUSPENSION_TRAVEL * UU_TO_BT;
        self.real_ray_length =
            self.suspension_rest_length_1 + suspension_travel + self.wheels_radius
                - bullet_vehicle::SUSPENSION_SUBTRACTION
                + bullet_vehicle::SUSPENSION_DETECT_EXTRA * UU_TO_BT;
    }

    pub fn prepare_for_raycast(&mut self, chassis_trans: &Affine3A) -> (Vec3A, Vec3A) {
        self.hard_point = chassis_trans.transform_point3a(self.chassis_connection_point_cs);
        let ray_len = if ggl_tuned_ray() {
            let travel = bullet_vehicle::MAX_SUSPENSION_TRAVEL * UU_TO_BT;
            self.suspension_rest_length_1 + travel + self.wheels_radius
        } else {
            let extra = if ggl_no_detect_extra() {
                0.0
            } else {
                bullet_vehicle::SUSPENSION_DETECT_EXTRA * UU_TO_BT
            };
            self.real_ray_length - bullet_vehicle::SUSPENSION_DETECT_EXTRA * UU_TO_BT + extra
        };
        let target = self.hard_point - (chassis_trans.matrix3.z_axis * ray_len);

        (self.hard_point, target)
    }

    pub fn reset_wheel_suspension(&mut self) {
        self.extra_pushback = 0.0;
        self.raycast_info = None;
        self.adhesion_contact = false;
    }

    /// How far the ray is inside the pushback threshold, BT (>=0).
    pub fn pushback_pen_bt(&self) -> f32 {
        let Some(info) = self.raycast_info.as_ref() else {
            return 0.0;
        };
        let thresh = self.suspension_rest_length_1 + self.wheels_radius
            - bullet_vehicle::SUSPENSION_SUBTRACTION;
        (thresh - info.wheel_trace_len).max(0.0)
    }

    pub fn apply_ray_cast(
        &mut self,
        chassis: &RigidBody,
        ray_results: VehicleRaycasterResult,
        time_step: f32,
        front: bool,
    ) {
        let contact_point = ray_results.hit_point_in_world;
        let contact_normal = ray_results.hit_normal_in_world;
        let is_in_contact_with_world = ray_results.rigid_body.is_static_obj();

        let chassis_trans = chassis.get_world_trans();
        self.axle_dir = if front {
            Quat::from_axis_angle_simd(chassis_trans.matrix3.z_axis, self.steer_angle)
                * chassis_trans.matrix3.y_axis
        } else {
            chassis_trans.matrix3.y_axis
        };

        let up = chassis_trans.matrix3.z_axis;
        let wheel_trace_len_sq = (self.hard_point - contact_point).dot(up);
        let mut suspension_length = wheel_trace_len_sq - self.wheels_radius;

        let suspension_travel = bullet_vehicle::MAX_SUSPENSION_TRAVEL * UU_TO_BT;
        let min_suspension_len = self.suspension_rest_length_1 - suspension_travel;
        let max_suspension_len = self.suspension_rest_length_1 + suspension_travel;

        // ADHESION-ONLY BAND. The extended probe reaches past the suspension's working
        // range so a car can re-acquire a wall it has drifted off (RL holds a car to a wall
        // with WheelSuspension and has no sticky force at all -- S28). But a wheel out
        // there is NOT on the ground: letting it generate friction and drive is what made a
        // jumping car behave as if still on the floor (half_flip 6.2 -> 334.6 uu, and that
        // regression is independent of the sticky force -- measured at 335.1 with sticky
        // disabled). So flag it for the sticky gate and take no other force from it.
        self.adhesion_contact = false;
        // GGL_TUNED_RAY: the ray already ends at the suspension's working range, so every
        // hit is real contact with full forces (tuned's model, no adhesion-only band).
        if !ggl_tuned_ray() && suspension_length > max_suspension_len {
            self.adhesion_contact = is_in_contact_with_world;
            self.raycast_info = None;
            return;
        }
        suspension_length = suspension_length.clamp(min_suspension_len, max_suspension_len);

        let rel_pos = contact_point - chassis_trans.translation;
        self.vel_at_contact_point = chassis.get_vel_in_local_point(rel_pos);

        let proj_vel = contact_normal.dot(self.vel_at_contact_point);
        let denom = contact_normal.dot(up);

        // PROGRESSIVE DAMPER ENGAGEMENT past rest length (SIM2REAL_AUDIT.md S34). The
        // damper reads contact-POINT velocity, so a fast-rotating car (a flip, or being
        // pitched by the floor-to-wall fillet at speed) sweeping a barely-touching wheel
        // along a surface reads omega x r (~250+ uu/s) as approach velocity, and the
        // full damper yanks the chassis off the surface -- measured -7..-22 uu/s per
        // tick in transition_flip_off while the REAL car in the same window shows no
        // such force (vx constant while its wheels graze the wall mid-flip). Fading the
        // damper linearly from full at rest length to zero at max extension keeps the
        // approach cushioning that slow fillet transits need (removing extended-range
        // damping entirely regressed no_jump_control 12.5 -> 37.0 uu) while removing
        // most of the graze-fling. Validated on two independent captures: total error
        // 1157.5 -> 1126.0 (fit) and 1181.9 -> 1151.5 (holdout); transition_land_tilted
        // 62.1 -> 4.8, transition_supersonic_into 35.6 -> 9.3; all steady-state guards
        // (drive/brake/steer/wall/ceiling/dodge/wavedash) bit-identical.
        let damp_scale = if ggl_no_damp_fade() {
            1.0
        } else if suspension_length > self.suspension_rest_length_1 {
            let span = max_suspension_len - self.suspension_rest_length_1;
            if span > 0.0 {
                ((max_suspension_len - suspension_length) / span).clamp(0.0, 1.0)
            } else {
                1.0
            }
        } else {
            1.0
        };
        let inv_dot_cap = ggl_inv_dot_cap();
        let (suspension_relative_vel, clipped_inv_contact_dot_suspension) = if denom > 0.1 {
            let inv = (1.0 / denom).min(inv_dot_cap);
            (proj_vel * inv * damp_scale, inv)
        } else {
            (0.0, inv_dot_cap.min(10.0))
        };

        if is_in_contact_with_world && !ggl_apply_time() && !ggl_no_pushback() {
            let ray_pushback_thresh = self.suspension_rest_length_1 + self.wheels_radius
                - bullet_vehicle::SUSPENSION_SUBTRACTION;
            if wheel_trace_len_sq < ray_pushback_thresh {
                let wheel_trace_dist_delta = wheel_trace_len_sq - ray_pushback_thresh;
                let collision_result = resolve_single_collision(
                    chassis,
                    ray_results.rigid_body,
                    ray_results.hit_point_in_world,
                    ray_results.hit_normal_in_world,
                    time_step,
                    wheel_trace_dist_delta,
                );

                // CAPPED (SIM2REAL_AUDIT.md S39). The hard resolve kills the FULL
                // normal approach velocity in one tick. At fillet-transit speeds that
                // is the support the real game also provides; at landing speeds the
                // real game lets the suspension absorb the impact over several ticks
                // (120 Hz capture: sim over-pushed +15..+62 uu/s per tick, monotone in
                // compression depth, while the real response stayed spring-shaped).
                // Capping the per-tick resolve at PUSHBACK_MAX_IMPULSE reconciles both
                // instruments: deep-compression landing bias +15.4 -> +0.5 uu/s, and
                // the maneuver battery total 1126.0 -> 1035.2 (jump_after_wall_launch
                // 153.9 -> 21.2, no_jump_control 15.0 -> 8.4; honest cost: the
                // teleport-settling tilt_* family +40 total, whose state-set penetration
                // recovery is not a real-play situation).
                let cap = ggl_pushback_cap();
                self.extra_pushback = if cap <= 0.0 {
                    collision_result
                } else {
                    collision_result.min(cap)
                } / pushback_divisor(1);
            }
        }

        let impulse = if ggl_apply_time() {
            Vec3A::ZERO
        } else {
            // Legacy raycast-time path: n_contact is unknowable mid-raycast; 4 keeps
            // the partial-contact scales neutral (this path is GGL_APPLY_TIME=0 only).
            self.calc_friction_impulses(
                chassis,
                ray_results.rigid_body,
                contact_normal,
                contact_point,
                time_step,
                front,
                4,
            )
        };

        self.raycast_info = Some(RaycastInfo {
            contact_normal,
            contact_point,
            suspension_length,
            impulse,
            is_in_contact_with_world,
            clipped_inv_contact_dot_suspension,
            suspension_relative_vel,
            ground_body_idx: ray_results.body_idx,
            wheel_trace_len: wheel_trace_len_sq,
        });
    }

    pub fn calc_friction_impulses(
        &self,
        chassis: &RigidBody,
        ground_rb: &RigidBody,
        contact_normal: Vec3A,
        contact_point: Vec3A,
        time_step: f32,
        front: bool,
        n_contact: usize,
    ) -> Vec3A {
        let friction_scale = chassis.get_mass() / 3.0;
        // Partial-contact force scaling (2026-08-23 identification pass, T1):
        // - GGL_LAT_PARTIAL: with <3 wheels each wheel's bilateral solve still kills
        //   the FULL body's lateral slip -- a single grazing wheel steers like a
        //   planted one (nw=1+steer bin: med|lat| 8.2 uu/s, sign matches steer 31/36).
        //   Modes: "n4" = n/4, "n3" = min(1, n/3), numeric = fixed factor for n<3.
        // - GGL_BRAKE_PARTIAL: per-wheel brake clamp is full-strength but only n of 4
        //   wheels contribute, so coasting on partial contact under-brakes (coast bin
        //   med_fwd +2.60). "norm" = scale brake by 4/n when n<3, numeric = fixed.
        let (lat_scale, brake_scale) = if n_contact >= 3 || n_contact == 0 {
            (1.0, 1.0)
        } else {
            let n = n_contact as f32;
            let lat = match ggl_lat_partial() {
                PartialMode::Off => 1.0,
                PartialMode::N4 => n / 4.0,
                PartialMode::N3 => (n / 3.0).min(1.0),
                PartialMode::Fixed(f) => f,
            };
            let brake = match ggl_brake_partial() {
                PartialMode::Off => 1.0,
                PartialMode::N4 | PartialMode::N3 => 4.0 / n,
                PartialMode::Fixed(f) => f,
            };
            (lat, brake)
        };
        // Isolated A/B: raw axle (no contact-plane projection). PR73 / tuned
        // RS_TUNE_RAW_LATERAL_AXLE. Revert if holdout p50/p90 regress.
        let mut axle_dir = if ggl_steer_at_apply() && front {
            // Tuned: friction direction uses THIS tick's steer, not the raycast-time axle.
            let trans = chassis.get_world_trans();
            (Quat::from_axis_angle_simd(trans.matrix3.z_axis, self.steer_angle)
                * trans.matrix3.y_axis)
                .normalize_or_zero()
        } else {
            self.axle_dir.normalize_or_zero()
        };
        if ggl_proj_axle() {
            axle_dir = (axle_dir - contact_normal * axle_dir.dot(contact_normal)).normalize_or_zero();
        }

        let forward_dir = contact_normal.cross(axle_dir).normalize_or_zero();

        let side_impulse =
            resolve_single_bilateral(chassis, ground_rb, contact_point, contact_point, axle_dir);

        let rolling_friction = if self.engine_force == 0.0 {
            if self.brake == 0.0 {
                0.0
            } else {
                const ROLLING_FRICTION_SCALE: f32 = 113.73963;

                let car_rel_contact_point = contact_point - chassis.get_world_trans().translation;

                let contact_vel = chassis.get_vel_in_local_point(car_rel_contact_point)
                    - ground_rb.get_vel_in_local_point(car_rel_contact_point);
                let mut rel_vel = contact_vel.dot(forward_dir);

                if time_step > 1.0 / 80.0 {
                    let threshold = 0.8 - (1.0 / (time_step * 150.0));
                    if rel_vel.abs() < threshold {
                        rel_vel = 0.0;
                    }
                }

                let b = self.brake * brake_scale;
                (-rel_vel * ROLLING_FRICTION_SCALE).clamp(-b, b)
            }
        } else {
            -self.engine_force / friction_scale
        };

        let mut lat_force = side_impulse * self.lat_friction * lat_scale * friction_scale;
        // GGL_LAT_LOADCAP=mu: friction-cone cap on the lateral impulse, |F_lat| <=
        // mu * N with N = this wheel's suspension force. The bilateral solve kills
        // the body's lateral slip through ANY contact regardless of load; a car
        // spinning on one corner wheel (T1 i=4648-4725: sustained -15.8 uu/s per
        // tick, steer-aligned) gets grip a real saturated wheel cannot deliver.
        // A cap only binds when demand exceeds load, so planted driving is
        // untouched (blanket lat scaling measured worse everywhere). 0 = off.
        let mu = ggl_lat_loadcap();
        if mu > 0.0 {
            let n_force = self.suspension_force_scalar(chassis);
            let max_lat = mu * n_force;
            lat_force = lat_force.clamp(-max_lat, max_lat);
        }
        forward_dir * rolling_friction * self.long_friction * friction_scale + axle_dir * lat_force
    }

    pub fn update_suspension(&mut self, cb: &mut RigidBody, delta_time: f32) {
        let Some(raycast_info) = self.raycast_info.as_ref() else {
            return;
        };

        let force = (self.suspension_rest_length_1 - raycast_info.suspension_length)
            * bullet_vehicle::SUSPENSION_STIFFNESS
            * raycast_info.clipped_inv_contact_dot_suspension;

        let damping_vel_scale = if raycast_info.suspension_relative_vel < 0.0 {
            bullet_vehicle::WHEELS_DAMPING_COMPRESSION
        } else {
            bullet_vehicle::WHEELS_DAMPING_RELAXATION
        };

        let mut wheels_suspension_force =
            force - (damping_vel_scale * raycast_info.suspension_relative_vel);
        wheels_suspension_force *= self.suspension_force_scale;
        if wheels_suspension_force <= 0.0 {
            return;
        }
        let base_force_scale = wheels_suspension_force * delta_time + self.extra_pushback;
        let contact_point_offset = raycast_info.contact_point - cb.get_world_trans().translation;

        let force = susp_impulse_dir(cb, raycast_info.contact_normal) * base_force_scale;
        cb.add_impulse(
            Some("WheelsSuspension"),
            Impulse::LinearRelPos(force, contact_point_offset),
            true,
            false,
        );
    }

    pub fn apply_friction_impulses(&self, cb: &mut RigidBody, time_step: f32) {
        let Some(raycast_info) = self.raycast_info.as_ref() else {
            return;
        };

        let trans = cb.get_world_trans();
        let wheel_contact_offset = raycast_info.contact_point - trans.translation;
        let contact_up_dot = trans.matrix3.z_axis.dot(wheel_contact_offset);
        let wheel_rel_pos = wheel_contact_offset - trans.matrix3.z_axis * contact_up_dot;
        cb.add_impulse(
            Some("WheelsFriction"),
            Impulse::LinearRelPos(raycast_info.impulse * time_step, wheel_rel_pos),
            true,
            false,
        );
    }

    pub fn update_extra_pushback(
        &mut self,
        chassis: &RigidBody,
        ground_rb: &RigidBody,
        time_step: f32,
        n_contact: usize,
        contact_n_override: Option<Vec3A>,
        packed: bool,
    ) {
        let Some(raycast_info) = self.raycast_info.as_ref() else {
            return;
        };
        if !raycast_info.is_in_contact_with_world
            && !(ggl_wheel_on_car() && ground_rb.user_idx == UserInfoTypes::Car)
        {
            return;
        }
        if n_contact < ggl_pushback_min_wheels() {
            return;
        }
        let ray_pushback_thresh = self.suspension_rest_length_1 + self.wheels_radius
            - bullet_vehicle::SUSPENSION_SUBTRACTION;
        if raycast_info.wheel_trace_len < ray_pushback_thresh {
            let wheel_trace_dist_delta = raycast_info.wheel_trace_len - ray_pushback_thresh;
            let n = self
                .susp_n(contact_n_override)
                .unwrap_or(raycast_info.contact_normal);
            let collision_result = resolve_single_collision(
                chassis,
                ground_rb,
                raycast_info.contact_point,
                n,
                time_step,
                wheel_trace_dist_delta,
            );
            if ggl_no_pushback() {
                return;
            }
            let cap = ggl_pushback_cap();
            self.extra_pushback = if cap <= 0.0 {
                collision_result
            } else {
                collision_result.min(cap)
            } / pushback_divisor(n_contact);
            if packed {
                self.extra_pushback *= ggl_packed_push_scale();
            }
        }
    }

    /// This wheel's current suspension force magnitude (spring + damper, clamped to
    /// >= 0), same math as `compute_suspension_impulse` without the impulse assembly.
    /// Used as the normal load N for the `GGL_LAT_LOADCAP` friction cone.
    pub fn suspension_force_scalar(&self, cb: &RigidBody) -> f32 {
        let Some(raycast_info) = self.raycast_info.as_ref() else {
            return 0.0;
        };
        let inv_dot = raycast_info.clipped_inv_contact_dot_suspension;
        let suspension_relative_vel = if inv_dot >= 10.0 {
            0.0
        } else {
            let rel_pos = raycast_info.contact_point - cb.get_world_trans().translation;
            let proj_vel = raycast_info
                .contact_normal
                .dot(cb.get_vel_in_local_point(rel_pos));
            let damp_scale = if ggl_no_damp_fade() {
                1.0
            } else if raycast_info.suspension_length > self.suspension_rest_length_1 {
                let travel = bullet_vehicle::MAX_SUSPENSION_TRAVEL * UU_TO_BT;
                let max_len = self.suspension_rest_length_1 + travel;
                let span = max_len - self.suspension_rest_length_1;
                if span > 0.0 {
                    ((max_len - raycast_info.suspension_length) / span).clamp(0.0, 1.0)
                } else {
                    1.0
                }
            } else {
                1.0
            };
            proj_vel * inv_dot * damp_scale
        };
        let force = (self.suspension_rest_length_1 - raycast_info.suspension_length)
            * bullet_vehicle::SUSPENSION_STIFFNESS
            * inv_dot;
        let damping_vel_scale = if suspension_relative_vel < 0.0 {
            bullet_vehicle::WHEELS_DAMPING_COMPRESSION
        } else {
            bullet_vehicle::WHEELS_DAMPING_RELAXATION
        };
        ((force - damping_vel_scale * suspension_relative_vel) * self.suspension_force_scale)
            .max(0.0)
    }

    /// Snapshot suspension impulse from the chassis's CURRENT velocity (all wheels
    /// before any apply, so later wheels don't see earlier wheels' impulses).
    pub fn compute_suspension_impulse(
        &self,
        cb: &RigidBody,
        delta_time: f32,
        contact_n_override: Option<Vec3A>,
        packed: bool,
    ) -> Option<Vec3A> {
        let raycast_info = self.raycast_info.as_ref()?;
        let n = self
            .susp_n(contact_n_override)
            .unwrap_or(raycast_info.contact_normal);
        let tape_n = self.susp_n_override.is_some();
        let mut inv_dot = if tape_n {
            let denom = n.dot(cb.get_world_trans().matrix3.z_axis);
            if denom > 0.1 {
                (1.0 / denom).min(ggl_inv_dot_cap())
            } else {
                ggl_inv_dot_cap().min(10.0)
            }
        } else {
            raycast_info.clipped_inv_contact_dot_suspension
        };
        if packed {
            let cap = ggl_packed_inv_dot_cap();
            if cap > 0.0 {
                inv_dot = inv_dot.min(cap);
            }
        }
        let suspension_relative_vel = if inv_dot >= 10.0 {
            0.0
        } else {
            let rel_pos = raycast_info.contact_point - cb.get_world_trans().translation;
            let n_damp = if tape_n {
                n
            } else {
                raycast_info.contact_normal
            };
            let proj_vel = n_damp.dot(cb.get_vel_in_local_point(rel_pos));
            let damp_scale = if ggl_no_damp_fade() {
                1.0
            } else if raycast_info.suspension_length > self.suspension_rest_length_1 {
                let travel = bullet_vehicle::MAX_SUSPENSION_TRAVEL * UU_TO_BT;
                let max_len = self.suspension_rest_length_1 + travel;
                let span = max_len - self.suspension_rest_length_1;
                if span > 0.0 {
                    ((max_len - raycast_info.suspension_length) / span).clamp(0.0, 1.0)
                } else {
                    1.0
                }
            } else {
                1.0
            };
            proj_vel * inv_dot * damp_scale
        };

        let force = (self.suspension_rest_length_1 - raycast_info.suspension_length)
            * bullet_vehicle::SUSPENSION_STIFFNESS
            * inv_dot;
        let damping_vel_scale = if suspension_relative_vel < 0.0 {
            bullet_vehicle::WHEELS_DAMPING_COMPRESSION
        } else {
            bullet_vehicle::WHEELS_DAMPING_RELAXATION
        };
        let mut wheels_suspension_force =
            force - (damping_vel_scale * suspension_relative_vel);
        wheels_suspension_force *= self.suspension_force_scale;
        if packed {
            wheels_suspension_force *= ggl_packed_susp_scale();
        }
        if wheels_suspension_force <= 0.0 {
            return None;
        }
        let base_force_scale = wheels_suspension_force * delta_time + self.extra_pushback;
        Some(susp_impulse_dir(cb, n) * base_force_scale)
    }

    pub fn apply_suspension_impulse(&self, cb: &mut RigidBody, force: Vec3A) {
        let Some(raycast_info) = self.raycast_info.as_ref() else {
            return;
        };
        let contact_point_offset = raycast_info.contact_point - cb.get_world_trans().translation;
        cb.add_impulse(
            Some("WheelsSuspension"),
            Impulse::LinearRelPos(force, contact_point_offset),
            true,
            false,
        );
    }

    pub fn apply_friction_impulse(&self, cb: &mut RigidBody, impulse: Vec3A, time_step: f32) {
        let Some(raycast_info) = self.raycast_info.as_ref() else {
            return;
        };
        let trans = cb.get_world_trans();
        let wheel_contact_offset = raycast_info.contact_point - trans.translation;
        let contact_up_dot = trans.matrix3.z_axis.dot(wheel_contact_offset);
        let wheel_rel_pos = wheel_contact_offset - trans.matrix3.z_axis * contact_up_dot;
        cb.add_impulse(
            Some("WheelsFriction"),
            Impulse::LinearRelPos(impulse * time_step, wheel_rel_pos),
            true,
            false,
        );
    }
}
