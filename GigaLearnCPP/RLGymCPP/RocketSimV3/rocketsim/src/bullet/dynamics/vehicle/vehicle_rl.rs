use std::sync::OnceLock;

use glam::Vec3A;

use super::{NUM_WHEELS, raycaster::VehicleRaycaster, wheel_info::WheelInfo};
use crate::bullet::{
    collision::broadphase::CollisionFilterGroups,
    dynamics::{
        discrete_dynamics_world::DiscreteDynamicsWorld,
        rigid_body::{Impulse, RigidBody},
    },
};
use crate::sim::UserInfoTypes;

fn ggl_wheel_sweep() -> f32 {
    static V: OnceLock<f32> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("GGL_WHEEL_SWEEP")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(0.0)
            .clamp(0.0, 0.95)
    })
}

fn ggl_wheel_ball_reaction() -> f32 {
    static V: OnceLock<f32> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("GGL_WHEEL_BALL_REACTION")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0)
    })
}

pub struct VehicleRL {
    raycaster: VehicleRaycaster,
    chassis_body_idx: usize,
    pub wheels: [WheelInfo; NUM_WHEELS],
}

impl VehicleRL {
    pub const fn new(chassis_body_idx: usize, wheels: [WheelInfo; NUM_WHEELS]) -> Self {
        Self {
            raycaster: VehicleRaycaster::new(CollisionFilterGroups::DropshotFloor as u8),
            chassis_body_idx,
            wheels,
        }
    }

    fn is_packed_fillet(&self, up_z: f32, n_contact: usize) -> bool {
        // Floor is |up.z|≈1; true wall ≈0. Packed fillet is the in-between, with
        // at least one wheel normal closer to horizontal (same as tape wall_drive).
        let az = up_z.abs();
        if n_contact != NUM_WHEELS || !(0.25..=0.90).contains(&az) {
            return false;
        }
        let mut any_wallish = false;
        for wheel in &self.wheels {
            let Some(info) = wheel.raycast_info.as_ref() else {
                return false;
            };
            if !info.is_in_contact_with_world {
                return false;
            }
            if info.contact_normal.z.abs() < 0.7 {
                any_wallish = true;
            }
        }
        any_wallish
    }

    fn packed_fillet_mean_n(&self, up_z: f32, n_contact: usize) -> Option<Vec3A> {
        if super::wheel_info::ggl_fillet_mean_n() <= 0.0 || !self.is_packed_fillet(up_z, n_contact) {
            return None;
        }
        let mut sum = Vec3A::ZERO;
        for wheel in &self.wheels {
            let info = wheel.raycast_info.as_ref()?;
            if !info.is_in_contact_with_world {
                return None;
            }
            sum += info.contact_normal;
        }
        sum.try_normalize()
    }

    pub fn get_upwards_dir_from_wheel_contacts(&self, cb: &RigidBody) -> Vec3A {
        let mut sum_contact_dir = Vec3A::ZERO;
        for wheel in &self.wheels {
            if let Some(raycast_info) = wheel.raycast_info.as_ref() {
                sum_contact_dir += raycast_info.contact_normal;
            }
        }

        sum_contact_dir
            .try_normalize()
            .unwrap_or_else(|| cb.get_up_vector())
    }

    pub const fn get_num_wheels(&self) -> usize {
        self.wheels.len()
    }

    pub fn update_vehicle_first(
        &mut self,
        collision_world: &DiscreteDynamicsWorld,
        time_step: f32,
    ) {
        let chassis = &collision_world.bodies()[self.chassis_body_idx];
        let chassis_trans = chassis.get_world_trans();

        let mut sources = [Vec3A::ZERO; 4];
        let mut targets = [Vec3A::ZERO; 4];

        for (i, wheel) in self.wheels.iter_mut().enumerate() {
            (sources[i], targets[i]) = wheel.prepare_for_raycast(chassis_trans);
        }

        let ray_results = self
            .raycaster
            .cast_rays(collision_world, &sources, &targets, chassis);

        let sweep = ggl_wheel_sweep();
        if sweep > 0.0 {
            let up = chassis_trans.matrix3.z_axis;
            let fwd = chassis_trans.matrix3.x_axis;
            let mut best: [Option<(f32, Vec3A, Vec3A, usize)>; 4] = [None; 4];
            for (i, res) in ray_results.iter().enumerate() {
                if let Some(r) = res {
                    let dist = (sources[i] - r.hit_point_in_world).dot(up);
                    best[i] = Some((
                        dist,
                        r.hit_point_in_world,
                        r.hit_normal_in_world,
                        r.body_idx,
                    ));
                }
            }
            for sign in [-1.0f32, 1.0] {
                let mut off_src = [Vec3A::ZERO; 4];
                let mut off_dst = [Vec3A::ZERO; 4];
                let mut bulge = [0.0f32; 4];
                for (i, wheel) in self.wheels.iter().enumerate() {
                    let r = wheel.wheels_radius;
                    let d = sign * sweep * r;
                    bulge[i] = r - (r * r - d * d).max(0.0).sqrt();
                    off_src[i] = sources[i] + fwd * d;
                    off_dst[i] = off_src[i] - up * (wheel.real_ray_length - bulge[i]);
                }
                let extra = self
                    .raycaster
                    .cast_rays(collision_world, &off_src, &off_dst, chassis);
                for (i, res) in extra.iter().enumerate() {
                    let Some(r) = res else { continue };
                    let dist = (off_src[i] - r.hit_point_in_world).dot(up) + bulge[i];
                    if best[i].is_none_or(|(d0, ..)| dist < d0) {
                        best[i] = Some((
                            dist,
                            r.hit_point_in_world,
                            r.hit_normal_in_world,
                            r.body_idx,
                        ));
                    }
                }
            }
            let bodies = collision_world.bodies();
            let chassis = &bodies[self.chassis_body_idx];
            for (i, wheel) in self.wheels.iter_mut().enumerate() {
                match best[i] {
                    Some((_, point, normal, body_idx)) => {
                        wheel.apply_ray_cast(
                            chassis,
                            super::raycaster::VehicleRaycasterResult {
                                rigid_body: &bodies[body_idx],
                                body_idx,
                                hit_point_in_world: point,
                                hit_normal_in_world: normal,
                            },
                            time_step,
                            i < 2,
                        );
                    }
                    None => wheel.reset_wheel_suspension(),
                }
            }
        } else {
            for (i, wheel) in self.wheels.iter_mut().enumerate() {
                if let Some(ray_result) = ray_results[i] {
                    wheel.apply_ray_cast(chassis, ray_result, time_step, i < 2);
                } else {
                    wheel.reset_wheel_suspension();
                }
            }
        }
    }

    pub fn update_vehicle_second(
        &mut self,
        collision_world: &mut DiscreteDynamicsWorld,
        step: f32,
    ) {
        if super::wheel_info::ggl_apply_time() {
            let mut friction_impulses = [Vec3A::ZERO; NUM_WHEELS];
            let mut mean_n: Option<Vec3A> = None;
            let mut packed = false;
            {
                let bodies = collision_world.bodies();
                let wheel_on_car = super::wheel_info::ggl_wheel_on_car();
                let n_contact = self
                    .wheels
                    .iter()
                    .filter(|w| {
                        w.raycast_info.as_ref().is_some_and(|info| {
                            info.is_in_contact_with_world
                                || (wheel_on_car
                                    && bodies.get(info.ground_body_idx).is_some_and(|b| {
                                        b.user_idx == UserInfoTypes::Car
                                    }))
                        })
                    })
                    .count();
                // Apply-time default: this tick's resolve or none. extra_pushback
                // used to persist when the ray was not inside the threshold, so a
                // leftover yaw-into-wall kick rode for hundreds of T6 ticks
                // (+6.84 car-up). Zero, then recompute. `GGL_PUSHBACK_MINPEN`
                // can skip the recompute (taxes T1/T7; default 0 = off).
                let minpen = super::wheel_info::ggl_pushback_minpen_bt();
                let skip_recompute = if minpen > 0.0 {
                    let max_pen = self
                        .wheels
                        .iter()
                        .map(WheelInfo::pushback_pen_bt)
                        .fold(0.0_f32, f32::max);
                    let on_wall = bodies[self.chassis_body_idx]
                        .get_world_trans()
                        .matrix3
                        .z_axis
                        .z
                        .abs()
                        < super::wheel_info::ggl_pushback_minpen_uz();
                    on_wall && max_pen < minpen
                } else {
                    false
                };
                let up_z = bodies[self.chassis_body_idx]
                    .get_world_trans()
                    .matrix3
                    .z_axis
                    .z;
                packed = self.is_packed_fillet(up_z, n_contact);
                mean_n = self.packed_fillet_mean_n(up_z, n_contact);
                for i in 0..NUM_WHEELS {
                    self.wheels[i].extra_pushback = 0.0;
                    if skip_recompute {
                        continue;
                    }
                    let Some(ground_idx) = self.wheels[i]
                        .raycast_info
                        .as_ref()
                        .map(|info| info.ground_body_idx)
                    else {
                        continue;
                    };
                    self.wheels[i].update_extra_pushback(
                        &bodies[self.chassis_body_idx],
                        &bodies[ground_idx],
                        step,
                        n_contact,
                        mean_n,
                        packed,
                    );
                }
                let chassis = &bodies[self.chassis_body_idx];
                for (i, wheel) in self.wheels.iter().enumerate() {
                    let Some(info) = wheel.raycast_info.as_ref() else {
                        continue;
                    };
                    friction_impulses[i] = wheel.calc_friction_impulses(
                        chassis,
                        &bodies[info.ground_body_idx],
                        info.contact_normal,
                        info.contact_point,
                        step,
                        i < 2,
                        n_contact,
                    );
                }
            }

            let mut suspension_impulses = [None; NUM_WHEELS];
            {
                let chassis = &mut collision_world.bodies_mut()[self.chassis_body_idx];
                for (i, wheel) in self.wheels.iter().enumerate() {
                    suspension_impulses[i] =
                        wheel.compute_suspension_impulse(chassis, step, mean_n, packed);
                }
                for (wheel, impulse) in self.wheels.iter().zip(suspension_impulses) {
                    if let Some(force) = impulse {
                        wheel.apply_suspension_impulse(chassis, force);
                    }
                }
                for (wheel, impulse) in self.wheels.iter().zip(friction_impulses) {
                    wheel.apply_friction_impulse(chassis, impulse, step);
                }
            }
            for wheel in &mut self.wheels {
                wheel.susp_n_override = None;
            }
            if super::wheel_info::ggl_wheel_on_car() {
                let mut reactions: [(usize, Vec3A, Vec3A); NUM_WHEELS] =
                    [(0, Vec3A::ZERO, Vec3A::ZERO); NUM_WHEELS];
                let mut n_react = 0;
                {
                    let bodies = collision_world.bodies();
                    for (i, wheel) in self.wheels.iter().enumerate() {
                        let Some(info) = wheel.raycast_info.as_ref() else {
                            continue;
                        };
                        if info.ground_body_idx == self.chassis_body_idx {
                            continue;
                        }
                        if !bodies
                            .get(info.ground_body_idx)
                            .is_some_and(|b| b.user_idx == UserInfoTypes::Car)
                        {
                            continue;
                        }
                        let mut force = friction_impulses[i] * step;
                        if let Some(s) = suspension_impulses[i] {
                            force += s;
                        }
                        if force.length_squared() <= f32::EPSILON {
                            continue;
                        }
                        reactions[n_react] = (info.ground_body_idx, -force, info.contact_point);
                        n_react += 1;
                    }
                }
                for &(idx, force, pt) in reactions.iter().take(n_react) {
                    let ground = &mut collision_world.bodies_mut()[idx];
                    let rel = pt - ground.get_world_trans().translation;
                    ground.add_impulse(
                        Some("WheelOnCar"),
                        Impulse::LinearRelPos(force, rel),
                        true,
                        false,
                    );
                }
            }
            let reaction_scale = ggl_wheel_ball_reaction();
            if reaction_scale != 0.0 {
                let mut reactions: [(usize, Vec3A, Vec3A); NUM_WHEELS] =
                    [(0, Vec3A::ZERO, Vec3A::ZERO); NUM_WHEELS];
                let mut n_react = 0;
                {
                    let bodies = collision_world.bodies();
                    for (i, wheel) in self.wheels.iter().enumerate() {
                        let (Some(force), Some(info)) =
                            (suspension_impulses[i], wheel.raycast_info.as_ref())
                        else {
                            continue;
                        };
                        if !bodies
                            .get(info.ground_body_idx)
                            .is_some_and(|b| !b.is_static_obj())
                        {
                            continue;
                        }
                        reactions[n_react] =
                            (info.ground_body_idx, -force * reaction_scale, info.contact_point);
                        n_react += 1;
                    }
                }
                for &(idx, force, pt) in reactions.iter().take(n_react) {
                    let ground = &mut collision_world.bodies_mut()[idx];
                    let rel = pt - ground.get_world_trans().translation;
                    ground.add_impulse(
                        Some("WheelOnBall"),
                        Impulse::LinearRelPos(force, rel),
                        true,
                        false,
                    );
                }
            }
            return;
        }

        let chassis = &mut collision_world.bodies_mut()[self.chassis_body_idx];
        for wheel in &mut self.wheels {
            wheel.update_suspension(chassis, step);
        }

        // note: all suspension MUST be updated before impulses are applied
        for wheel in &mut self.wheels {
            wheel.apply_friction_impulses(chassis, step);
        }
    }
}
