use std::collections::{HashMap, HashSet};

use super::{
    collision_obj_wrapper::RigidBodyWrapper, compound_collision_alg, convex_concave_collision_alg,
    convex_plane_collision_alg, obb_obb_collision_alg, sphere_concave_collision_alg,
    sphere_obb_collision_alg,
};
use crate::bullet::{
    collision::{
        broadphase::{BroadphaseProxy, GridBroadphase},
        dispatch::convex_convex_collision_alg,
        narrowphase::persistent_manifold::{ContactAddedCallback, PersistentManifold},
        shapes::collision_shape::CollisionShapes,
    },
    dynamics::rigid_body::RigidBody,
};

pub struct CollisionDispatcher {
    pub manifolds: Vec<PersistentManifold>,
    /// CAR-CAR persistent manifold cache (GGL_CC_PERSIST). Real Bullet keeps a
    /// manifold per broadphase pair across ticks: once a contact point exists it
    /// survives (and keeps entering the solver) until it drifts past the
    /// breaking threshold (~1 uu) along or orthogonal to its normal. This port
    /// rebuilds manifolds from the detector every tick, so a grazing box-box
    /// miss instantly drops the contact -- measured on the T1 grind (i=4648+)
    /// as flickering contact and ~16 uu/s of missing mutual separation, while
    /// blanket box inflation over-fires on fly-by ticks. Entries are evicted
    /// when the broadphase pair stops overlapping (near_callback not reached).
    cc_cache: Vec<(usize, usize, PersistentManifold, bool)>,
    /// Chassis-vs-plane/mesh 4-point cache (v3-tuned `RS_TUNE_PERSISTENT_MANIFOLDS`).
    /// Separate from car-car `GGL_CC_PERSIST`. Off unless `GGL_PLANE_PERSIST` is set.
    plane_cache: HashMap<(usize, usize), PersistentManifold>,
    plane_touched: HashSet<(usize, usize)>,
}

impl Default for CollisionDispatcher {
    fn default() -> Self {
        Self {
            manifolds: Vec::with_capacity(8),
            cc_cache: Vec::new(),
            plane_cache: HashMap::new(),
            plane_touched: HashSet::new(),
        }
    }
}

fn ggl_cc_persist() -> bool {
    use std::sync::OnceLock;
    static V: OnceLock<bool> = OnceLock::new();
    // Default ON (2026-08-24): T1 od<130 verr p90 12.39 -> 10.92, grind-event
    // axis bias +6.21 -> +2.70, no collateral anywhere (fly-bys untouched --
    // persistence needs a real seed contact, unlike box inflation which was
    // tested and rejected). GGL_CC_PERSIST=0 restores per-tick manifolds.
    *V.get_or_init(|| !std::env::var("GGL_CC_PERSIST").is_ok_and(|s| s == "0"))
}

/// Bitmask: 1 = convex-vs-static-plane, 2 = other (mesh). Default 0 (A/B).
/// Tuned ships 3. Unrelated to `GGL_CC_PERSIST`.
fn ggl_plane_persist_mask() -> u32 {
    use std::sync::OnceLock;
    static V: OnceLock<u32> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("GGL_PLANE_PERSIST")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0)
    })
}

impl CollisionDispatcher {
    fn process_collision<'a, T: ContactAddedCallback>(
        col_obj_a: &'a RigidBody,
        col_obj_b: &'a RigidBody,
        contact_added_callback: &'a mut T,
    ) -> Option<PersistentManifold> {
        match col_obj_a.get_collision_shape() {
            CollisionShapes::StaticPlane(plane) => match col_obj_b.get_collision_shape() {
                CollisionShapes::Sphere(_) | CollisionShapes::ConvexHull(_) => {
                    convex_plane_collision_alg::process_collision(
                        &RigidBodyWrapper {
                            obj: col_obj_b,
                            world_trans: *col_obj_b.get_world_trans(),
                            child_shape_override: None,
                        },
                        col_obj_a,
                        plane,
                        contact_added_callback,
                    )
                }
                CollisionShapes::Compound(compound) => compound_collision_alg::process_collision(
                    col_obj_b,
                    compound,
                    col_obj_a,
                    contact_added_callback,
                ),
                _ => unreachable!(),
            },
            CollisionShapes::Sphere(sphere) => match col_obj_b.get_collision_shape() {
                CollisionShapes::StaticPlane(plane) => {
                    convex_plane_collision_alg::process_collision(
                        &RigidBodyWrapper {
                            obj: col_obj_a,
                            world_trans: *col_obj_a.get_world_trans(),
                            child_shape_override: None,
                        },
                        col_obj_b,
                        plane,
                        contact_added_callback,
                    )
                }
                CollisionShapes::TriangleMesh(mesh) => {
                    sphere_concave_collision_alg::process_collision(
                        col_obj_a,
                        sphere,
                        col_obj_b,
                        mesh,
                        contact_added_callback,
                    )
                }
                CollisionShapes::Compound(compound) => sphere_obb_collision_alg::process_collision(
                    col_obj_a,
                    sphere,
                    col_obj_b,
                    compound,
                    contact_added_callback,
                ),
                CollisionShapes::ConvexHull(_) => convex_convex_collision_alg::process_collision(
                    &RigidBodyWrapper {
                        obj: col_obj_a,
                        world_trans: *col_obj_a.get_world_trans(),
                        child_shape_override: None,
                    },
                    col_obj_b,
                    contact_added_callback,
                ),
                _ => unreachable!(),
            },
            CollisionShapes::TriangleMesh(mesh) => match col_obj_b.get_collision_shape() {
                CollisionShapes::Sphere(sphere) => sphere_concave_collision_alg::process_collision(
                    col_obj_b,
                    sphere,
                    col_obj_a,
                    mesh,
                    contact_added_callback,
                ),
                CollisionShapes::Compound(compound) => compound_collision_alg::process_collision(
                    col_obj_b,
                    compound,
                    col_obj_a,
                    contact_added_callback,
                ),
                CollisionShapes::ConvexHull(_) => convex_concave_collision_alg::process_collision(
                    col_obj_b,
                    col_obj_a,
                    mesh,
                    contact_added_callback,
                ),
                _ => unreachable!(),
            },
            CollisionShapes::Compound(compound_a) => match col_obj_b.get_collision_shape() {
                CollisionShapes::StaticPlane(_) | CollisionShapes::TriangleMesh(_) => {
                    compound_collision_alg::process_collision(
                        col_obj_a,
                        compound_a,
                        col_obj_b,
                        contact_added_callback,
                    )
                }
                CollisionShapes::Sphere(sphere) => sphere_obb_collision_alg::process_collision(
                    col_obj_b,
                    sphere,
                    col_obj_a,
                    compound_a,
                    contact_added_callback,
                ),
                CollisionShapes::Compound(compound_b) => obb_obb_collision_alg::process_collision(
                    col_obj_a,
                    compound_a,
                    col_obj_b,
                    compound_b,
                    contact_added_callback,
                ),
                CollisionShapes::ConvexHull(_) => compound_collision_alg::process_collision(
                    col_obj_a,
                    compound_a,
                    col_obj_b,
                    contact_added_callback,
                ),
                CollisionShapes::Triangle(_) => unreachable!(),
            },
            CollisionShapes::ConvexHull(_) => match col_obj_b.get_collision_shape() {
                CollisionShapes::StaticPlane(plane) => {
                    convex_plane_collision_alg::process_collision(
                        &RigidBodyWrapper {
                            obj: col_obj_a,
                            world_trans: *col_obj_a.get_world_trans(),
                            child_shape_override: None,
                        },
                        col_obj_b,
                        plane,
                        contact_added_callback,
                    )
                }
                CollisionShapes::Compound(compound) => compound_collision_alg::process_collision(
                    col_obj_b,
                    compound,
                    col_obj_a,
                    contact_added_callback,
                ),
                CollisionShapes::TriangleMesh(mesh) => {
                    convex_concave_collision_alg::process_collision(
                        col_obj_a,
                        col_obj_b,
                        mesh,
                        contact_added_callback,
                    )
                }
                CollisionShapes::Sphere(_) => convex_convex_collision_alg::process_collision(
                    &RigidBodyWrapper {
                        obj: col_obj_a,
                        world_trans: *col_obj_a.get_world_trans(),
                        child_shape_override: None,
                    },
                    col_obj_b,
                    contact_added_callback,
                ),
                _ => unreachable!(),
            },
            CollisionShapes::Triangle(_) => unreachable!(),
        }
    }

    pub fn near_callback<T: ContactAddedCallback>(
        &mut self,
        collision_objs: &[RigidBody],
        proxy0: &BroadphaseProxy,
        proxy1: &BroadphaseProxy,
        contact_added_callback: &mut T,
    ) {
        let rb0 = &collision_objs[proxy0.client_obj_idx as usize];
        let rb1 = &collision_objs[proxy1.client_obj_idx as usize];

        if !rb0.is_active() && !rb1.is_active()
            || !rb0.has_contact_response()
            || !rb1.has_contact_response()
        {
            return;
        }

        let fresh = Self::process_collision(rb0, rb1, contact_added_callback);

        let is_car_car = ggl_cc_persist()
            && rb0.user_idx == crate::sim::UserInfoTypes::Car
            && rb1.user_idx == crate::sim::UserInfoTypes::Car;
        if is_car_car {
            let key = if rb0.world_array_idx < rb1.world_array_idx {
                (rb0.world_array_idx, rb1.world_array_idx)
            } else {
                (rb1.world_array_idx, rb0.world_array_idx)
            };
            let slot = self
                .cc_cache
                .iter_mut()
                .find(|(a, b, ..)| (*a, *b) == key);
            if let Some(manifold) = fresh {
                match slot {
                    Some(entry) => {
                        entry.2 = manifold.clone();
                        entry.3 = true;
                    }
                    None => self.cc_cache.push((key.0, key.1, manifold.clone(), true)),
                }
                self.manifolds.push(manifold);
            } else if let Some(entry) = slot {
                entry.3 = true;
                // Detector miss with live cached points: refresh against the
                // current transforms (drops points past the breaking threshold)
                // and, if any survive, hand the cached manifold to the solver.
                let (b0, b1) = if entry.2.body0_idx == rb0.world_array_idx {
                    (rb0, rb1)
                } else {
                    (rb1, rb0)
                };
                entry.2.refresh_contact_points(b0, b1);
                if entry.2.point_cache.is_empty() {
                    entry.3 = false;
                } else {
                    self.manifolds.push(entry.2.clone());
                }
            }
        } else if let Some(manifold) = fresh {
            let is_plane_pair = matches!(rb0.get_collision_shape(), CollisionShapes::StaticPlane(_))
                || matches!(rb1.get_collision_shape(), CollisionShapes::StaticPlane(_));
            let mask = ggl_plane_persist_mask();
            let keep = if is_plane_pair {
                mask & 1 != 0
            } else {
                mask & 2 != 0
            };
            if !keep {
                self.manifolds.push(manifold);
            } else {
                let key = (manifold.body0_idx, manifold.body1_idx);
                let body0 = &collision_objs[manifold.body0_idx];
                let body1 = &collision_objs[manifold.body1_idx];
                let cached = self
                    .plane_cache
                    .entry(key)
                    .or_insert_with(|| PersistentManifold::new(body0, body1));
                cached.merge_new_points(&manifold, body0, body1);
                self.plane_touched.insert(key);
                if !cached.point_cache.is_empty() {
                    self.manifolds.push(cached.clone());
                }
            }
        }
    }

    fn sync_applied_impulses(&mut self) {
        if self.plane_cache.is_empty() {
            return;
        }
        for manifold in &self.manifolds {
            let Some(cached) = self.plane_cache.get_mut(&(manifold.body0_idx, manifold.body1_idx))
            else {
                continue;
            };
            if cached.point_cache.len() != manifold.point_cache.len() {
                continue;
            }
            for (cached_point, solved) in cached.point_cache.iter_mut().zip(&manifold.point_cache) {
                cached_point.applied_impulse = solved.applied_impulse;
            }
        }
    }

    pub fn dispatch_all_collision_pairs<T: ContactAddedCallback>(
        &mut self,
        collision_objs: &[RigidBody],
        pair_cache: &mut GridBroadphase,
        contact_added_callback: &mut T,
    ) {
        // Solver leaves last tick's manifolds so impulses can warm-start the cache.
        self.sync_applied_impulses();
        self.manifolds.clear();
        for entry in &mut self.cc_cache {
            entry.3 = false;
        }
        self.plane_touched.clear();
        pair_cache.process_all_overlapping_pairs(collision_objs, self, contact_added_callback);
        self.cc_cache.retain(|(.., touched)| *touched);
        if !self.plane_cache.is_empty() {
            let touched = &self.plane_touched;
            self.plane_cache.retain(|key, _| touched.contains(key));
        }
    }
}
