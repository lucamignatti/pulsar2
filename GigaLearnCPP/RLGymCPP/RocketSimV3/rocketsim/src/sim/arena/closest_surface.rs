use super::Arena;
use crate::{
    bullet::collision::shapes::collision_shape::CollisionShapes,
    consts::{BT_TO_UU, UU_TO_BT},
    shared::Aabb,
};
use glam::Vec3A;

/// Closest arena surface to a query point. Distances and points are UU.
/// `hit` is false when nothing is inside `max_dist` (then `dist` is `max_dist`).
#[derive(Clone, Copy, Debug)]
pub struct ClosestSurface {
    pub point: Vec3A,
    pub normal: Vec3A,
    pub dist: f32,
    pub hit: bool,
}

impl ClosestSurface {
    fn miss(max_dist_uu: f32) -> Self {
        Self {
            point: Vec3A::ZERO,
            normal: Vec3A::ZERO,
            dist: max_dist_uu,
            hit: false,
        }
    }
}

struct BestHit {
    point_bt: Vec3A,
    normal: Vec3A,
    dist_sq_bt: f32,
    hit: bool,
}

impl BestHit {
    fn consider(&mut self, point_bt: Vec3A, normal: Vec3A, query_bt: Vec3A) {
        let delta = query_bt - point_bt;
        let d2 = delta.length_squared();
        if d2 < self.dist_sq_bt {
            self.dist_sq_bt = d2;
            self.point_bt = point_bt;
            self.normal = delta.try_normalize().unwrap_or(normal);
            self.hit = true;
        }
    }
}

impl Arena {
    #[must_use]
    pub fn closest_surface(&self, query_uu: Vec3A, max_dist_uu: f32) -> ClosestSurface {
        if !(max_dist_uu > 0.0) {
            return ClosestSurface::miss(0.0);
        }

        let query_bt = query_uu * UU_TO_BT;
        let max_bt = max_dist_uu * UU_TO_BT;
        let max_dist_sq = max_bt * max_bt;
        let query_aabb = Aabb {
            min: query_bt - Vec3A::splat(max_bt),
            max: query_bt + Vec3A::splat(max_bt),
        };

        let mut best = BestHit {
            point_bt: Vec3A::ZERO,
            normal: Vec3A::Z,
            dist_sq_bt: max_dist_sq,
            hit: false,
        };

        for body in self.bullet_world.bodies() {
            match body.get_collision_shape() {
                CollisionShapes::TriangleMesh(mesh) => {
                    let trans = *body.get_world_trans();
                    let world_aabb = body.get_collision_shape().get_aabb(&trans);
                    if !world_aabb.intersects(&query_aabb) {
                        continue;
                    }
                    let query_local = trans
                        .matrix3
                        .mul_transpose_vec3a(query_bt - trans.translation);
                    let Some((pt_local, n_local, _)) =
                        mesh.closest_point_to(query_local, best.dist_sq_bt)
                    else {
                        continue;
                    };
                    let pt_world = trans.transform_point3a(pt_local);
                    let n_world = trans
                        .transform_vector3a(n_local)
                        .try_normalize()
                        .unwrap_or(n_local);
                    best.consider(pt_world, n_world, query_bt);
                }
                CollisionShapes::StaticPlane(plane) => {
                    // Floor only. Wall/ceiling planes are the analytic stadium.
                    let n_local = plane.get_plane_normal();
                    if n_local.z < 0.9 {
                        continue;
                    }
                    let trans = *body.get_world_trans();
                    let n_world = trans
                        .transform_vector3a(n_local)
                        .try_normalize()
                        .unwrap_or(n_local);
                    let origin = trans.translation;
                    let signed = (query_bt - origin).dot(n_world);
                    let pt_world = query_bt - n_world * signed;
                    best.consider(pt_world, n_world, query_bt);
                }
                _ => {}
            }
        }

        if !best.hit {
            return ClosestSurface::miss(max_dist_uu);
        }
        ClosestSurface {
            point: best.point_bt * BT_TO_UU,
            normal: best.normal,
            dist: best.dist_sq_bt.sqrt() * BT_TO_UU,
            hit: true,
        }
    }

    pub fn closest_surface_n(
        &self,
        queries_uu: &[Vec3A],
        max_dist_uu: f32,
        out: &mut [ClosestSurface],
    ) {
        let n = queries_uu.len().min(out.len());
        if n == 0 {
            return;
        }
        if n == 1 {
            out[0] = self.closest_surface(queries_uu[0], max_dist_uu);
            return;
        }
        if !(max_dist_uu > 0.0) {
            for slot in out.iter_mut().take(n) {
                *slot = ClosestSurface::miss(0.0);
            }
            return;
        }

        const MAX_N: usize = 8;
        if n > MAX_N {
            for i in 0..n {
                out[i] = self.closest_surface(queries_uu[i], max_dist_uu);
            }
            return;
        }

        let max_bt = max_dist_uu * UU_TO_BT;
        let max_dist_sq = max_bt * max_bt;
        let mut queries_bt = [Vec3A::ZERO; MAX_N];
        let mut union = Aabb {
            min: Vec3A::splat(f32::MAX),
            max: Vec3A::splat(f32::MIN),
        };
        let r = Vec3A::splat(max_bt);
        for i in 0..n {
            let q = queries_uu[i] * UU_TO_BT;
            queries_bt[i] = q;
            union.min = union.min.min(q - r);
            union.max = union.max.max(q + r);
        }

        let mut best_d2 = [max_dist_sq; MAX_N];
        let mut best_pt = [Vec3A::ZERO; MAX_N];
        let mut best_n = [Vec3A::Z; MAX_N];
        let mut hit = [false; MAX_N];

        for body in self.bullet_world.bodies() {
            match body.get_collision_shape() {
                CollisionShapes::TriangleMesh(mesh) => {
                    let trans = *body.get_world_trans();
                    let world_aabb = body.get_collision_shape().get_aabb(&trans);
                    if !world_aabb.intersects(&union) {
                        continue;
                    }
                    let mut queries_local = [Vec3A::ZERO; MAX_N];
                    for i in 0..n {
                        queries_local[i] = trans
                            .matrix3
                            .mul_transpose_vec3a(queries_bt[i] - trans.translation);
                    }
                    let mut local_d2 = [0.0f32; MAX_N];
                    let mut local_pt = [Vec3A::ZERO; MAX_N];
                    let mut local_n = [Vec3A::Z; MAX_N];
                    let mut local_hit = [false; MAX_N];
                    local_d2[..n].copy_from_slice(&best_d2[..n]);
                    mesh.closest_point_to_n(
                        &queries_local[..n],
                        &mut local_d2[..n],
                        &mut local_pt[..n],
                        &mut local_n[..n],
                        &mut local_hit[..n],
                    );
                    for i in 0..n {
                        if !local_hit[i] {
                            continue;
                        }
                        let pt_world = trans.transform_point3a(local_pt[i]);
                        let n_world = trans
                            .transform_vector3a(local_n[i])
                            .try_normalize()
                            .unwrap_or(local_n[i]);
                        let delta = queries_bt[i] - pt_world;
                        let d2 = delta.length_squared();
                        if d2 < best_d2[i] {
                            best_d2[i] = d2;
                            best_pt[i] = pt_world;
                            best_n[i] = n_world;
                            hit[i] = true;
                        }
                    }
                }
                CollisionShapes::StaticPlane(plane) => {
                    let n_local = plane.get_plane_normal();
                    if n_local.z < 0.9 {
                        continue;
                    }
                    let trans = *body.get_world_trans();
                    let n_world = trans
                        .transform_vector3a(n_local)
                        .try_normalize()
                        .unwrap_or(n_local);
                    let origin = trans.translation;
                    for i in 0..n {
                        let signed = (queries_bt[i] - origin).dot(n_world);
                        let pt_world = queries_bt[i] - n_world * signed;
                        let delta = queries_bt[i] - pt_world;
                        let d2 = delta.length_squared();
                        if d2 < best_d2[i] {
                            best_d2[i] = d2;
                            best_pt[i] = pt_world;
                            best_n[i] = n_world;
                            hit[i] = true;
                        }
                    }
                }
                _ => {}
            }
        }

        for i in 0..n {
            out[i] = if hit[i] {
                ClosestSurface {
                    point: best_pt[i] * BT_TO_UU,
                    normal: best_n[i],
                    dist: best_d2[i].sqrt() * BT_TO_UU,
                    hit: true,
                }
            } else {
                ClosestSurface::miss(max_dist_uu)
            };
        }
    }
}
