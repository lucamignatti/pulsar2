//! Everything about a car that `CarState` does not carry.
//!
//! `Arena::set_car_state` restores the rigid body's transform and velocities and
//! replaces `Car::state`, but a car also owns a raycast vehicle (`VehicleRL`) whose
//! per-wheel suspension and contact state persists across ticks, plus a pending bump
//! impulse (`vel_impulse_cache`, which `set_state` zeroes rather than restores).
//! Leaving those behind makes a state restore *look* exact while quietly resuming on
//! someone else's suspension: replaying identical inputs from a restored grounded car
//! diverged ~23uu within a second (measured; see `tests/test_viz_snapshot.cpp` in the
//! consuming C++ tree).
//!
//! This exists so a restore can be exact. It is a plain `#[repr(C)]` POD so the C FFI
//! can hand it to C++ as an opaque byte block — C++ never names a field, so this
//! struct can gain members without touching the hand-mirrored FFI header.
//!
//! Deliberately NOT included, because `Arena::step_tick` reconstructs them before use:
//! the rigid body's `accum_lin_vel`/`accum_ang_vel` (cleared by `clear_accum_forces`
//! at the top of every tick) and `interp_world_trans` (recomputed by
//! `predict_unconstraint_motion`).

use glam::Vec3A;

use crate::bullet::dynamics::vehicle::{NUM_WHEELS, RaycastInfo, WheelInfo};

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct WheelExtraState {
    /// 0 when `WheelInfo::raycast_info` is `None`; the raycast fields below are then
    /// meaningless and are not applied.
    pub has_raycast_info: u32,
    pub contact_normal: [f32; 3],
    pub contact_point: [f32; 3],
    pub suspension_length: f32,
    pub impulse: [f32; 3],
    pub is_in_contact_with_world: u32,
    pub clipped_inv_contact_dot_suspension: f32,
    pub suspension_relative_vel: f32,

    pub hard_point: [f32; 3],
    pub axle_dir: [f32; 3],
    pub chassis_connection_point_cs: [f32; 3],
    pub suspension_rest_length_1: f32,
    pub wheels_radius: f32,
    pub engine_force: f32,
    pub brake: f32,
    pub steer_angle: f32,
    pub vel_at_contact_point: [f32; 3],
    pub lat_friction: f32,
    pub long_friction: f32,
    pub suspension_force_scale: f32,
    pub extra_pushback: f32,
    pub real_ray_length: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CarExtraState {
    pub wheels: [WheelExtraState; NUM_WHEELS],
    pub vel_impulse_cache: [f32; 3],
}

fn to_arr(v: Vec3A) -> [f32; 3] {
    [v.x, v.y, v.z]
}
fn from_arr(a: [f32; 3]) -> Vec3A {
    Vec3A::new(a[0], a[1], a[2])
}

impl WheelExtraState {
    pub(crate) fn capture(wheel: &WheelInfo) -> Self {
        let mut out = Self {
            hard_point: to_arr(wheel.hard_point),
            axle_dir: to_arr(wheel.axle_dir),
            chassis_connection_point_cs: to_arr(wheel.chassis_connection_point_cs),
            suspension_rest_length_1: wheel.suspension_rest_length_1,
            wheels_radius: wheel.wheels_radius,
            engine_force: wheel.engine_force,
            brake: wheel.brake,
            steer_angle: wheel.steer_angle,
            vel_at_contact_point: to_arr(wheel.vel_at_contact_point),
            lat_friction: wheel.lat_friction,
            long_friction: wheel.long_friction,
            suspension_force_scale: wheel.suspension_force_scale,
            extra_pushback: wheel.extra_pushback,
            real_ray_length: wheel.real_ray_length,
            ..Default::default()
        };

        if let Some(ray) = wheel.raycast_info.as_ref() {
            out.has_raycast_info = 1;
            out.contact_normal = to_arr(ray.contact_normal);
            out.contact_point = to_arr(ray.contact_point);
            out.suspension_length = ray.suspension_length;
            out.impulse = to_arr(ray.impulse);
            out.is_in_contact_with_world = u32::from(ray.is_in_contact_with_world);
            out.clipped_inv_contact_dot_suspension = ray.clipped_inv_contact_dot_suspension;
            out.suspension_relative_vel = ray.suspension_relative_vel;
        }

        out
    }

    pub(crate) fn apply(&self, wheel: &mut WheelInfo) {
        wheel.hard_point = from_arr(self.hard_point);
        wheel.axle_dir = from_arr(self.axle_dir);
        wheel.chassis_connection_point_cs = from_arr(self.chassis_connection_point_cs);
        wheel.suspension_rest_length_1 = self.suspension_rest_length_1;
        wheel.wheels_radius = self.wheels_radius;
        wheel.engine_force = self.engine_force;
        wheel.brake = self.brake;
        wheel.steer_angle = self.steer_angle;
        wheel.vel_at_contact_point = from_arr(self.vel_at_contact_point);
        wheel.lat_friction = self.lat_friction;
        wheel.long_friction = self.long_friction;
        wheel.suspension_force_scale = self.suspension_force_scale;
        wheel.extra_pushback = self.extra_pushback;
        wheel.real_ray_length = self.real_ray_length;

        wheel.raycast_info = if self.has_raycast_info == 0 {
            None
        } else {
            Some(RaycastInfo {
                contact_normal: from_arr(self.contact_normal),
                contact_point: from_arr(self.contact_point),
                suspension_length: self.suspension_length,
                impulse: from_arr(self.impulse),
                is_in_contact_with_world: self.is_in_contact_with_world != 0,
                clipped_inv_contact_dot_suspension: self.clipped_inv_contact_dot_suspension,
                suspension_relative_vel: self.suspension_relative_vel,
            })
        };
    }
}
