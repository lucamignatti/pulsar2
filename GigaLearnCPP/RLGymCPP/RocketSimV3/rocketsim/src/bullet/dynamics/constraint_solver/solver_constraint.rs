use glam::Vec3A;

use super::{contact_solver_info, solver_body::SolverBody};
use crate::bullet::{
    collision::narrowphase::manifold_point::ManifoldPoint, dynamics::rigid_body::RigidBody,
};

fn bullet_dot(vec0: Vec3A, vec1: Vec3A) -> f32 {
    let result = vec0 * vec1;
    result.x + (result.y + result.z)
}

#[derive(Default)]
pub struct SolverConstraint {
    pub rel_pos1_cross_normal: Vec3A,
    pub contact_normal_1: Vec3A,
    pub rel_pos2_cross_normal: Vec3A,
    pub contact_normal_2: Vec3A,
    pub angular_component_a: Vec3A,
    pub angular_component_b: Vec3A,
    pub applied_push_impulse: f32,
    pub applied_impulse: f32,
    pub friction: f32,
    pub jac_diag_ab_inv: f32,
    pub rhs: f32,
    pub lower_limit: f32,
    pub upper_limit: f32,
    pub rhs_penetration: f32,
    pub friction_idx: usize,
    pub solver_body_id_a: usize,
    pub solver_body_id_b: usize,
    pub is_special: bool,
    /// (manifold index, point index) for `write_back_contacts` warmstart.
    pub origin: Option<(usize, usize)>,
}

impl SolverConstraint {
    pub fn get_friction_constraint(
        (solver_body_id_a, solver_body_id_b): (usize, usize),
        (solver_body_a, solver_body_b): (&mut SolverBody, &mut SolverBody),
        (rb0, rb1): (Option<&RigidBody>, Option<&RigidBody>),
        (rel_pos1, rel_pos2): (Vec3A, Vec3A),
        cp: &ManifoldPoint,
        friction_idx: usize,
    ) -> Self {
        let mut constraint = Self {
            friction_idx,
            solver_body_id_a,
            solver_body_id_b,
            lower_limit: -cp.combined_friction,
            upper_limit: cp.combined_friction,
            friction: cp.combined_friction,
            ..Default::default()
        };

        constraint.setup_friction_constraint(
            (solver_body_a, solver_body_b),
            (rb0, rb1),
            (rel_pos1, rel_pos2),
            cp.lateral_friction_dir_1,
        );

        constraint
    }

    pub fn get_contact_constraint(
        (solver_body_id_a, solver_body_id_b): (usize, usize),
        (solver_body_a, solver_body_b): (&mut SolverBody, &mut SolverBody),
        (rb0, rb1): (Option<&RigidBody>, Option<&RigidBody>),
        (rel_pos1, rel_pos2): (Vec3A, Vec3A),
        cp: &ManifoldPoint,
        friction_idx: usize,
        time_step: f32,
    ) -> Self {
        let mut constraint = Self {
            solver_body_id_a,
            solver_body_id_b,
            friction_idx,
            friction: cp.combined_friction,
            is_special: cp.is_special,
            lower_limit: 0.0,
            upper_limit: 1e10,
            ..Default::default()
        };

        constraint.setup_contact_constraint(
            (solver_body_a, solver_body_b),
            (rb0, rb1),
            (rel_pos1, rel_pos2),
            cp,
            time_step,
        );

        constraint
    }

    pub fn restitution_curve(rel_vel: f32, restitution: f32) -> f32 {
        if rel_vel.abs() < contact_solver_info::restitution_velocity_threshold() {
            0.0
        } else {
            restitution * -rel_vel
        }
    }

    fn setup_contact_constraint(
        &mut self,
        (solver_body_a, solver_body_b): (&mut SolverBody, &mut SolverBody),
        (rb0, rb1): (Option<&RigidBody>, Option<&RigidBody>),
        (rel_pos1, rel_pos2): (Vec3A, Vec3A),
        cp: &ManifoldPoint,
        time_step: f32,
    ) {
        let inv_time_step = 1.0 / time_step;
        let erp = contact_solver_info::ERP_2;
        self.applied_impulse = cp.applied_impulse * contact_solver_info::WARMSTARTING_FACTOR;

        let (denom0, vel0, vel_1_dot_n) = rb0.map_or((0.0, Vec3A::ZERO, 0.0), |rb| {
            let torque_axis = rel_pos1.cross(cp.normal_world_on_b);
            self.angular_component_a = rb.inv_inertia_tensor_world.mul_transpose_vec3a(torque_axis);
            let vec = self.angular_component_a.cross(rel_pos1);
            let denom = rb.inv_mass + cp.normal_world_on_b.dot(vec);
            let vel = rb.get_vel_in_local_point(rel_pos1);

            self.contact_normal_1 = cp.normal_world_on_b;
            self.rel_pos1_cross_normal = torque_axis;

            solver_body_a.internal_apply_impulse(
                cp.normal_world_on_b * solver_body_a.inv_mass,
                self.angular_component_a,
                self.applied_impulse,
            );

            let vel_dot_n = self
                .contact_normal_1
                .dot(solver_body_a.lin_vel + solver_body_a.external_force_impulse)
                + self
                    .rel_pos1_cross_normal
                    .dot(solver_body_a.ang_vel + solver_body_a.external_torque_impulse);

            (denom, vel, vel_dot_n)
        });

        let (denom1, vel1, vel_2_dot_n) = rb1.map_or((0.0, Vec3A::ZERO, 0.0), |rb| {
            let torque_axis = rel_pos2.cross(cp.normal_world_on_b);
            self.angular_component_b = rb
                .inv_inertia_tensor_world
                .mul_transpose_vec3a(-torque_axis);
            let vec = (-self.angular_component_b).cross(rel_pos2);
            let denom = rb.inv_mass + cp.normal_world_on_b.dot(vec);
            let vel = rb.get_vel_in_local_point(rel_pos2);
            self.contact_normal_2 = -cp.normal_world_on_b;
            self.rel_pos2_cross_normal = -torque_axis;

            solver_body_b.internal_apply_impulse(
                cp.normal_world_on_b * solver_body_b.inv_mass,
                -self.angular_component_b,
                -self.applied_impulse,
            );

            let vel_dot_n = self
                .contact_normal_2
                .dot(solver_body_b.lin_vel + solver_body_b.external_force_impulse)
                + self
                    .rel_pos2_cross_normal
                    .dot(solver_body_b.ang_vel + solver_body_b.external_torque_impulse);

            (denom, vel, vel_dot_n)
        });

        self.jac_diag_ab_inv = contact_solver_info::SOR / (denom0 + denom1);

        let vel = vel0 - vel1;
        let rel_vel = cp.normal_world_on_b.dot(vel);
        let restitution = Self::restitution_curve(rel_vel, cp.combined_restitution).max(0.0);

        let penetration = cp.distance_1;
        let positional_error = if penetration > 0.0 {
            0.0
        } else {
            -penetration * erp * inv_time_step
        };

        let rel_vel = vel_1_dot_n + vel_2_dot_n;
        let vel_error = restitution - rel_vel;

        let penetration_impulse = positional_error * self.jac_diag_ab_inv;
        let vel_impulse = vel_error * self.jac_diag_ab_inv;

        // GGL_CC_BAUMGARTE=<erp>: for CAR-CAR contacts, route the penetration
        // correction into the VELOCITY rhs (old-Bullet `m_splitImpulse = false`
        // path) instead of the position-only split impulse. RL ships a 2.6x-era
        // Bullet where split impulse defaulted OFF, so its 1e30 threshold is
        // inert and penetrating cars gain real separation VELOCITY each tick --
        // the T1 grind (i=4648+) shows mutual momentum-conserving separation
        // bursts of ~16 uu/s decaying over ~5 ticks that the split path can
        // never produce (push velocity is discarded, never recorded). v2 set the
        // threshold but inherited the modern `m_splitImpulse = true` default
        // silently. 0 = off (split path, v2/v3 behavior).
        let cc_erp = {
            use std::sync::OnceLock;
            static V: OnceLock<f32> = OnceLock::new();
            *V.get_or_init(|| {
                std::env::var("GGL_CC_BAUMGARTE")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.0)
            })
        };
        // GGL_CC_DEPTH=<uu>: deepen the SOLVER's view of car-car penetration by
        // this many uu (only when already penetrating -- never creates contact).
        // RL's 2.6x Bullet measures box-box depth against margin-inflated boxes,
        // so its restoring impulses run ~1-2 uu deeper than nominal geometry.
        let cc_depth = {
            use std::sync::OnceLock;
            static V: OnceLock<f32> = OnceLock::new();
            *V.get_or_init(|| {
                std::env::var("GGL_CC_DEPTH")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.0)
                    * crate::consts::UU_TO_BT
            })
        };
        let is_car_car = (cc_erp > 0.0 || cc_depth > 0.0)
            && rb0.is_some_and(|r| r.user_idx == crate::sim::UserInfoTypes::Car)
            && rb1.is_some_and(|r| r.user_idx == crate::sim::UserInfoTypes::Car);
        let penetration = if is_car_car && penetration < 0.0 {
            penetration - cc_depth
        } else {
            penetration
        };
        if (rb0.is_some_and(|r| r.user_idx == crate::sim::UserInfoTypes::Car)
            && rb1.is_some_and(|r| r.user_idx == crate::sim::UserInfoTypes::Car))
            && crate::DBG_IMPULSE_TRACE.load(std::sync::atomic::Ordering::Relaxed)
        {
            eprintln!(
                "CCSOLVE pen={penetration:+.5} rel_vel={rel_vel:+.4} vel_imp={vel_impulse:+.4} pen_imp={penetration_impulse:+.4} n=({:+.2},{:+.2},{:+.2}) fric={:.3}",
                cp.normal_world_on_b.x,
                cp.normal_world_on_b.y,
                cp.normal_world_on_b.z,
                cp.combined_friction,
            );
        }
        (self.rhs, self.rhs_penetration) = if is_car_car && cc_erp > 0.0 {
            let pos_err = if penetration > 0.0 {
                0.0
            } else {
                -penetration * cc_erp * inv_time_step
            };
            (pos_err * self.jac_diag_ab_inv + vel_impulse, 0.0)
        } else {
            // Depth-only car-car mode still deepens the split-impulse pushout.
            let pen_imp = if is_car_car {
                let pos_err = if penetration > 0.0 {
                    0.0
                } else {
                    -penetration * erp * inv_time_step
                };
                pos_err * self.jac_diag_ab_inv
            } else {
                penetration_impulse
            };
            if penetration > contact_solver_info::SPLIT_IMPULSE_PENETRATION_THRESHOLD {
                (pen_imp + vel_impulse, 0.0)
            } else {
                (vel_impulse, pen_imp)
            }
        };
    }

    fn setup_friction_constraint(
        &mut self,
        (solver_body_a, solver_body_b): (&SolverBody, &SolverBody),
        (rb0, rb1): (Option<&RigidBody>, Option<&RigidBody>),
        (rel_pos1, rel_pos2): (Vec3A, Vec3A),
        normal_axis: Vec3A,
    ) {
        let (vel_1_dot_n, denom1) = rb0.map_or((0.0, 0.0), |rb| {
            self.contact_normal_1 = normal_axis;
            self.rel_pos1_cross_normal = rel_pos1.cross(self.contact_normal_1);
            self.angular_component_a = rb
                .inv_inertia_tensor_world
                .mul_transpose_vec3a(self.rel_pos1_cross_normal);

            let vec = self.angular_component_a.cross(rel_pos1);
            let denom = rb.inv_mass + normal_axis.dot(vec);

            let vel_dot_n = self
                .contact_normal_1
                .dot(solver_body_a.lin_vel + solver_body_a.external_force_impulse)
                + self.rel_pos1_cross_normal.dot(solver_body_a.ang_vel);

            (vel_dot_n, denom)
        });

        let (vel_2_dot_n, denom2) = rb1.map_or((0.0, 0.0), |rb| {
            self.contact_normal_2 = -normal_axis;
            self.rel_pos2_cross_normal = rel_pos2.cross(self.contact_normal_2);
            self.angular_component_b = rb
                .inv_inertia_tensor_world
                .mul_transpose_vec3a(self.rel_pos2_cross_normal);

            let vec = (-self.angular_component_b).cross(rel_pos2);
            let denom = rb.inv_mass + normal_axis.dot(vec);

            let vel_dot_n = self
                .contact_normal_2
                .dot(solver_body_b.lin_vel + solver_body_b.external_force_impulse)
                + self.rel_pos2_cross_normal.dot(solver_body_b.ang_vel);

            (vel_dot_n, denom)
        });

        self.jac_diag_ab_inv = contact_solver_info::SOR / (denom1 + denom2);

        let rel_vel = vel_1_dot_n + vel_2_dot_n;
        let vel_error = -rel_vel;
        self.rhs = vel_error * self.jac_diag_ab_inv;
    }

    pub fn resolve_single_constraint_row_generic(
        &mut self,
        body_a: &mut SolverBody,
        body_b: &mut SolverBody,
    ) -> f32 {
        let mut delta_impulse = self.rhs;

        let delta_vel_1_dot_n = bullet_dot(self.contact_normal_1, body_a.delta_lin_vel)
            + bullet_dot(self.rel_pos1_cross_normal, body_a.delta_ang_vel);
        let delta_vel_2_dot_n = bullet_dot(self.contact_normal_2, body_b.delta_lin_vel)
            + bullet_dot(self.rel_pos2_cross_normal, body_b.delta_ang_vel);

        delta_impulse -= delta_vel_1_dot_n * self.jac_diag_ab_inv;
        delta_impulse -= delta_vel_2_dot_n * self.jac_diag_ab_inv;

        let sum = self.applied_impulse + delta_impulse;
        if sum < self.lower_limit {
            delta_impulse = self.lower_limit - self.applied_impulse;
            self.applied_impulse = self.lower_limit;
        } else if sum > self.upper_limit {
            delta_impulse = self.upper_limit - self.applied_impulse;
            self.applied_impulse = self.upper_limit;
        } else {
            self.applied_impulse = sum;
        }

        body_a.delta_lin_vel += self.contact_normal_1 * body_a.inv_mass * delta_impulse;
        body_a.delta_ang_vel += self.angular_component_a * delta_impulse;

        body_b.delta_lin_vel += self.contact_normal_2 * body_b.inv_mass * delta_impulse;
        body_b.delta_ang_vel += self.angular_component_b * delta_impulse;

        delta_impulse / self.jac_diag_ab_inv
    }

    pub fn resolve_single_constraint_row_lower_limit(
        &mut self,
        body_a: &mut SolverBody,
        body_b: &mut SolverBody,
    ) -> f32 {
        let mut delta_impulse = self.rhs;

        let delta_vel_1_dot_n = bullet_dot(self.contact_normal_1, body_a.delta_lin_vel)
            + bullet_dot(self.rel_pos1_cross_normal, body_a.delta_ang_vel);
        let delta_vel_2_dot_n = bullet_dot(self.contact_normal_2, body_b.delta_lin_vel)
            + bullet_dot(self.rel_pos2_cross_normal, body_b.delta_ang_vel);

        delta_impulse -= delta_vel_1_dot_n * self.jac_diag_ab_inv;
        delta_impulse -= delta_vel_2_dot_n * self.jac_diag_ab_inv;

        let sum = self.applied_impulse + delta_impulse;
        if sum < self.lower_limit {
            delta_impulse = self.lower_limit - self.applied_impulse;
            self.applied_impulse = self.lower_limit;
        } else {
            self.applied_impulse = sum;
        }

        body_a.delta_lin_vel += self.contact_normal_1 * body_a.inv_mass * delta_impulse;
        body_a.delta_ang_vel += self.angular_component_a * delta_impulse;

        body_b.delta_lin_vel += self.contact_normal_2 * body_b.inv_mass * delta_impulse;
        body_b.delta_ang_vel += self.angular_component_b * delta_impulse;

        delta_impulse / self.jac_diag_ab_inv
    }

    pub fn resolve_split_penetration_impulse(
        &mut self,
        body_a: &mut SolverBody,
        body_b: &mut SolverBody,
    ) -> f32 {
        if self.rhs_penetration == 0.0 {
            return 0.0;
        }

        let mut delta_impulse = self.rhs_penetration;

        let delta_vel_1_dot_n = bullet_dot(self.contact_normal_1, body_a.push_vel)
            + bullet_dot(self.rel_pos1_cross_normal, body_a.turn_vel);
        let delta_vel_2_dot_n = bullet_dot(self.contact_normal_2, body_b.push_vel)
            + bullet_dot(self.rel_pos2_cross_normal, body_b.turn_vel);

        delta_impulse -= delta_vel_1_dot_n * self.jac_diag_ab_inv;
        delta_impulse -= delta_vel_2_dot_n * self.jac_diag_ab_inv;

        let sum = self.applied_push_impulse + delta_impulse;
        if sum < self.lower_limit {
            delta_impulse = self.lower_limit - self.applied_push_impulse;
            self.applied_push_impulse = self.lower_limit;
        } else {
            self.applied_push_impulse = sum;
        }

        body_a.push_vel += self.contact_normal_1 * body_a.inv_mass * delta_impulse;
        body_a.turn_vel += self.angular_component_a * delta_impulse;

        body_b.push_vel += self.contact_normal_2 * body_b.inv_mass * delta_impulse;
        body_b.turn_vel += self.angular_component_b * delta_impulse;

        delta_impulse / self.jac_diag_ab_inv
    }
}
