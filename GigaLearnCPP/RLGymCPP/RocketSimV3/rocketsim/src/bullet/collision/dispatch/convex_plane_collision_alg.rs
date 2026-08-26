use glam::{Affine3A, Mat3A, Quat, Vec3};

use super::collision_obj_wrapper::RigidBodyWrapper;
use crate::bullet::{
    collision::{
        narrowphase::persistent_manifold::{
            CONTACT_BREAKING_THRESHOLD, ContactAddedCallback, PersistentManifold,
        },
        shapes::static_plane_shape::StaticPlaneShape,
    },
    dynamics::rigid_body::RigidBody,
    linear_math::plane_space_2,
};

/// Bullet's `m_minimumPointsPerturbationThreshold` where the perturbation is enabled.
const MIN_POINTS_PERTURBATION_THRESHOLD: usize = 3;

fn ggl_plane_perturb_iters() -> u32 {
    use std::sync::OnceLock;
    static V: OnceLock<u32> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("GGL_PLANE_PERTURB")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(0)
    })
}

pub fn process_collision<T: ContactAddedCallback>(
    convex_obj: &RigidBodyWrapper,
    plane_obj: &RigidBody,
    plane_shape: &StaticPlaneShape,
    contact_added_callback: &mut T,
) -> Option<PersistentManifold> {
    let convex_aabb = convex_obj.get_aabb();
    if !convex_aabb.intersects(&plane_shape.aabb_cache) {
        return None;
    }

    let plane_normal = plane_shape.get_plane_normal();

    let plane_trans = plane_obj.get_world_trans();
    let plane_in_convex = convex_obj.world_trans.matrix3.transpose() * plane_trans.matrix3;
    let convex_in_plane_trans = Affine3A {
        matrix3: plane_trans.matrix3.transpose() * convex_obj.world_trans.matrix3,
        translation: plane_trans.matrix3 * convex_obj.world_trans.translation
            - plane_trans.translation,
    };

    let vtx = convex_obj.local_get_supporting_vertex(plane_in_convex * -plane_normal);
    let vtx_in_plane = convex_in_plane_trans.transform_point3a(vtx);
    let distance = plane_normal.dot(vtx_in_plane);

    let mut manifold = PersistentManifold::new(convex_obj.obj, plane_obj);
    if distance >= manifold.contact_breaking_threshold {
        return None;
    }

    let vtx_in_plane_projected = vtx_in_plane - distance * plane_normal;
    let vtx_in_plane_world = plane_obj
        .get_world_trans()
        .transform_point3a(vtx_in_plane_projected);
    let normal_on_surface_b = plane_obj.get_world_trans().matrix3 * plane_normal;

    manifold.add_contact_point(
        convex_obj.obj,
        plane_obj,
        normal_on_surface_b,
        vtx_in_plane_world,
        distance,
        None,
        contact_added_callback,
    );

    // v3-tuned `RS_TUNE_PLANE_PERTURBATION`. Default 0 (their official 30→62).
    // `GGL_PLANE_PERTURB=3` is Bullet's default-config count.
    let iters = ggl_plane_perturb_iters();
    if iters > 0 && manifold.point_cache.len() < MIN_POINTS_PERTURBATION_THRESHOLD {
        let (v0, _) = plane_space_2(plane_normal);
        let radius = convex_obj.get_collision_shape().get_angular_motion_disc();
        let angle_limit = 0.125 * std::f32::consts::PI;
        let perturbe_angle = (CONTACT_BREAKING_THRESHOLD / radius).min(angle_limit);
        let perturbe_rot = Quat::from_axis_angle(Vec3::from(v0).normalize_or_zero(), perturbe_angle);
        let normal_axis = Vec3::from(plane_normal).normalize_or_zero();
        for i in 0..iters {
            let iteration_angle = i as f32 * (std::f32::consts::TAU / iters as f32);
            let rotq = Quat::from_axis_angle(normal_axis, iteration_angle);
            let perturbed = rotq.inverse() * perturbe_rot * rotq;

            let tilted_basis = convex_obj.world_trans.matrix3 * Mat3A::from_quat(perturbed);
            let plane_in_tilted = tilted_basis.transpose() * plane_trans.matrix3;
            let vtx = convex_obj.local_get_supporting_vertex(plane_in_tilted * -plane_normal);
            let vtx_in_plane = convex_in_plane_trans.transform_point3a(vtx);
            let distance = plane_normal.dot(vtx_in_plane);
            if distance >= manifold.contact_breaking_threshold {
                continue;
            }
            let vtx_in_plane_projected = vtx_in_plane - distance * plane_normal;
            let vtx_in_plane_world = plane_obj
                .get_world_trans()
                .transform_point3a(vtx_in_plane_projected);
            manifold.add_contact_point(
                convex_obj.obj,
                plane_obj,
                normal_on_surface_b,
                vtx_in_plane_world,
                distance,
                None,
                contact_added_callback,
            );
        }
    }

    manifold.refresh_contact_points(convex_obj.obj, plane_obj);
    Some(manifold)
}
