use super::{
    triangle_callback::ProcessTriangle, triangle_info_map::TriangleInfoMap,
    triangle_mesh::TriangleMesh, triangle_mesh_shape::TriangleMeshShape,
    triangle_shape::TriangleShape,
};
use crate::{
    bullet::collision::{
        dispatch::{
            internal_edge_utility::generate_internal_edge_info,
            tri_bvh_util::{NodeOverlapCallback, QuadRayNodeOverlapCallback},
        },
        shapes::{optimized_bvh::create_bvh, triangle_callback::ProcessQuadRayTriangle},
    },
    shared::{Aabb, QuadRayInfo, bvh::Tree},
};
use glam::Vec3A;

pub struct BvhTriangleMeshShape {
    bvh: Tree,
    mesh_interface: TriangleMesh,
    triangle_info_map: TriangleInfoMap,
    pub aabb_ident_cache: Aabb,
}

impl BvhTriangleMeshShape {
    pub fn new(mesh_interface: TriangleMesh) -> Self {
        let triangle_mesh_shape = TriangleMeshShape::new(&mesh_interface);

        // pre-calculate the aabb
        let aabb_ident_cache = triangle_mesh_shape.get_ident_aabb();

        let bvh = create_bvh(&mesh_interface, triangle_mesh_shape.local_aabb);
        let triangle_info_map = generate_internal_edge_info(&bvh, &mesh_interface);

        Self {
            bvh,
            mesh_interface,
            triangle_info_map,
            aabb_ident_cache,
        }
    }

    pub const fn get_triangle_info_map(&self) -> &TriangleInfoMap {
        &self.triangle_info_map
    }

    pub const fn get_mesh_interface(&self) -> &TriangleMesh {
        &self.mesh_interface
    }

    pub fn process_all_triangles<T: ProcessTriangle>(&self, callback: &mut T, aabb: &Aabb) {
        let mut my_node_callback = NodeOverlapCallback::new(self.get_mesh_interface(), callback);
        self.bvh
            .report_aabb_overlapping_node(&mut my_node_callback, aabb);
    }

    pub fn check_overlap_with(&self, aabb: &Aabb) -> bool {
        self.bvh.check_overlap_with(aabb)
    }

    pub fn perform_quad_raycast<T: ProcessQuadRayTriangle>(
        &self,
        callback: &mut T,
        ray_info: &mut QuadRayInfo,
    ) {
        let mut my_node_callback =
            QuadRayNodeOverlapCallback::new(self.get_mesh_interface(), callback);
        self.bvh
            .report_quad_ray_overlapping_node(&mut my_node_callback, ray_info);
    }

    /// Closest point on this mesh to `query` (BVH / BT space).
    /// `max_dist_sq` is the search radius squared; None if nothing is inside it.
    /// Returns (point, unit normal facing the query, dist_sq).
    pub fn closest_point_to(
        &self,
        query: Vec3A,
        max_dist_sq: f32,
    ) -> Option<(Vec3A, Vec3A, f32)> {
        if !(max_dist_sq > 0.0) {
            return None;
        }
        let max_dist = max_dist_sq.sqrt();
        let r = Vec3A::splat(max_dist);
        let aabb = Aabb {
            min: query - r,
            max: query + r,
        };
        if !self.check_overlap_with(&aabb) {
            return None;
        }
        let mut cb = ClosestTriCallback {
            query,
            best_d2: max_dist_sq,
            best_pt: Vec3A::ZERO,
            best_n: Vec3A::Z,
            hit: false,
        };
        self.process_all_triangles(&mut cb, &aabb);
        cb.hit
            .then_some((cb.best_pt, cb.best_n, cb.best_d2))
    }

    /// One BVH walk for N nearby queries (Octane wheels). `best_d2` is both
    /// the search radius² in and the per-query result out. glam Vec3A is the
    /// SIMD unit; the win is visiting each triangle once.
    pub fn closest_point_to_n(
        &self,
        queries: &[Vec3A],
        best_d2: &mut [f32],
        best_pt: &mut [Vec3A],
        best_n: &mut [Vec3A],
        hit: &mut [bool],
    ) {
        let n = queries
            .len()
            .min(best_d2.len())
            .min(best_pt.len())
            .min(best_n.len())
            .min(hit.len());
        if n == 0 {
            return;
        }
        let mut max_d2 = 0.0f32;
        for i in 0..n {
            max_d2 = max_d2.max(best_d2[i]);
        }
        if !(max_d2 > 0.0) {
            return;
        }
        let max_dist = max_d2.sqrt();
        let r = Vec3A::splat(max_dist);
        let mut aabb = Aabb {
            min: queries[0] - r,
            max: queries[0] + r,
        };
        for q in &queries[1..n] {
            aabb.min = aabb.min.min(*q - r);
            aabb.max = aabb.max.max(*q + r);
        }
        if !self.check_overlap_with(&aabb) {
            return;
        }
        let mut cb = ClosestTriCallbackN {
            queries,
            best_d2,
            best_pt,
            best_n,
            hit,
            n,
        };
        self.process_all_triangles(&mut cb, &aabb);
    }
}

struct ClosestTriCallback {
    query: Vec3A,
    best_d2: f32,
    best_pt: Vec3A,
    best_n: Vec3A,
    hit: bool,
}

impl ProcessTriangle for ClosestTriCallback {
    fn process_triangle(&mut self, triangle: &TriangleShape, _triangle_idx: usize) {
        let obj_to_points = [
            self.query - triangle.points[0],
            self.query - triangle.points[1],
            self.query - triangle.points[2],
        ];
        let pt = triangle.closest_point(&obj_to_points);
        let delta = self.query - pt;
        let d2 = delta.length_squared();
        if d2 < self.best_d2 {
            self.best_d2 = d2;
            self.best_pt = pt;
            self.best_n = delta.try_normalize().unwrap_or(triangle.normal);
            self.hit = true;
        }
    }
}

struct ClosestTriCallbackN<'a> {
    queries: &'a [Vec3A],
    best_d2: &'a mut [f32],
    best_pt: &'a mut [Vec3A],
    best_n: &'a mut [Vec3A],
    hit: &'a mut [bool],
    n: usize,
}

impl ProcessTriangle for ClosestTriCallbackN<'_> {
    fn process_triangle(&mut self, triangle: &TriangleShape, _triangle_idx: usize) {
        let p0 = triangle.points[0];
        let p1 = triangle.points[1];
        let p2 = triangle.points[2];
        for i in 0..self.n {
            let q = self.queries[i];
            let obj_to_points = [q - p0, q - p1, q - p2];
            let pt = triangle.closest_point(&obj_to_points);
            let delta = q - pt;
            let d2 = delta.length_squared();
            if d2 < self.best_d2[i] {
                self.best_d2[i] = d2;
                self.best_pt[i] = pt;
                self.best_n[i] = delta.try_normalize().unwrap_or(triangle.normal);
                self.hit[i] = true;
            }
        }
    }
}
