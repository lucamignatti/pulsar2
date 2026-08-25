use std::mem;

use crate::{
    bullet::{
        collision::{
            dispatch::internal_edge_utility::adjust_internal_edge_contacts,
            narrowphase::{
                manifold_point::ManifoldPoint, persistent_manifold::ContactAddedCallback,
            },
        },
        dynamics::rigid_body::RigidBody,
    },
    consts,
    sim::UserInfoTypes,
};

// An instance of a contact event
#[derive(Debug, Copy, Clone)]
pub(crate) struct ContactRecord {
    pub is_swap: bool,
    pub rb_idx_a: usize,
    pub rb_idx_b: usize,
    pub manifold_point: ManifoldPoint,
}

// A struct to be accessed through the bullet contact callbacks
pub(crate) struct ArenaContactTracker {
    collision_records: Vec<ContactRecord>,
}

impl ArenaContactTracker {
    pub fn new() -> Self {
        Self {
            collision_records: Vec::with_capacity(4), // Rarely exceeded
        }
    }

    pub const fn num_records(&self) -> usize {
        self.collision_records.len()
    }

    pub fn get_record(&self, idx: usize) -> &ContactRecord {
        &self.collision_records[idx]
    }

    pub fn clear_records(&mut self) {
        self.collision_records.clear();
    }
}

impl ContactAddedCallback for ArenaContactTracker {
    fn callback<'a>(
        &mut self,
        manifold_point: &mut ManifoldPoint,
        mut body_a: &'a RigidBody,
        mut body_b: &'a RigidBody,
        idx: Option<usize>,
    ) {
        debug_assert!(body_a.has_contact_response() || body_b.has_contact_response());

        let should_swap =
            if body_a.user_idx != UserInfoTypes::None && body_b.user_idx != UserInfoTypes::None {
                body_a.user_idx > body_b.user_idx
            } else {
                body_b.user_idx != UserInfoTypes::None
            };

        if should_swap {
            mem::swap(&mut body_a, &mut body_b);
        }

        let user_idx_a = body_a.user_idx;
        let user_idx_b = body_b.user_idx;

        if user_idx_a == UserInfoTypes::Car {
            let hit_coefs = match user_idx_b {
                UserInfoTypes::Ball => consts::car::HIT_BALL_COEFS,
                UserInfoTypes::Car => consts::car::HIT_CAR_COEFS,
                // CHASSIS-vs-world friction is suppressed while the wheels carry the car.
                //
                // The floor/wall fillet is CONCAVE, so a long box hitbox digs into it and
                // the chassis scrapes even though the car is driving normally on its
                // wheels. Measured against the real capture: the sim bled ~257 uu/s
                // crossing the fillet at supersonic that the real game does not, and the
                // deficit then stayed pinned at exactly -248.8 uu/s forever after -- i.e.
                // a one-off energy loss in the curve, not a force error (boost, gravity
                // and drive all track afterwards).
                //
                // Zeroing this friction outright fixed the whole fillet cluster but
                // wrecked every tilted-landing segment (tilt_inverted 3.4 -> 171.2 uu),
                // because a car sliding on its shell genuinely needs it. Gating on wheel
                // contact separates the two cases. See SIM2REAL_AUDIT.md S25.
                _ if body_a.wheels_grounded
                    && !std::env::var("GGL_NO_CHASSIS_SUPPRESS").is_ok_and(|s| s != "0") =>
                {
                    let mut coefs = consts::car::HIT_WORLD_WHEELS_DOWN_COEFS;
                    // Wheels-down still used restitution 0.3; that can bounce the
                    // chassis on a 1-tick GT pose even with friction already zeroed.
                    if std::env::var("GGL_CHASSIS_REST0").is_ok_and(|s| s != "0") {
                        coefs.restitution = 0.0;
                    }
                    coefs
                }
                _ => {
                    let mut coefs = consts::car::HIT_WORLD_COEFS;
                    if std::env::var("GGL_WORLD_REST0").is_ok_and(|s| s != "0") {
                        coefs.restitution = 0.0;
                    }
                    coefs
                }
            };
            manifold_point.combined_friction = hit_coefs.friction;
            manifold_point.combined_restitution = hit_coefs.restitution;
        } else if user_idx_a == UserInfoTypes::Ball
            && user_idx_b == UserInfoTypes::None
            && (!std::env::var("GGL_PR74").is_ok_and(|s| s != "0")
                && !std::env::var("GGL_PR74_DEDUP").is_ok_and(|s| s != "0")
                || body_b.is_static_obj())
        {
            manifold_point.is_special = true;
        }

        // NOTE: Push *before* the manifold is mutated by adjust_internal_edge_contacts()
        self.collision_records.push(ContactRecord {
            is_swap: should_swap,
            rb_idx_a: body_a.world_array_idx,
            rb_idx_b: body_b.world_array_idx,
            manifold_point: *manifold_point,
        });

        if let Some(idx) = idx {
            adjust_internal_edge_contacts(manifold_point, body_b, idx);
        }
    }
}
