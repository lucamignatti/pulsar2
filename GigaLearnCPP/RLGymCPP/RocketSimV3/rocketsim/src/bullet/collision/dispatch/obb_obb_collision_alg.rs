use super::box_box_detector::BoxBoxDetector;
use crate::bullet::{
    collision::{
        narrowphase::persistent_manifold::{ContactAddedCallback, PersistentManifold},
        shapes::compound_shape::CompoundShape,
    },
    dynamics::rigid_body::RigidBody,
};

pub fn process_collision<T: ContactAddedCallback>(
    compound_a_obj: &RigidBody,
    compound_a_shape: &CompoundShape,
    compound_b_obj: &RigidBody,
    compound_b_shape: &CompoundShape,
    contact_added_callback: &mut T,
) -> Option<PersistentManifold> {
    let trace = crate::DBG_IMPULSE_TRACE.load(std::sync::atomic::Ordering::Relaxed);
    let org_0_trans = compound_a_obj.get_world_trans();
    let aabb_0 = compound_a_shape.get_aabb(org_0_trans);
    let org_1_trans = compound_b_obj.get_world_trans();
    let aabb_1 = compound_b_shape.get_aabb(org_1_trans);
    if !aabb_0.intersects(&aabb_1) {
        let dx = (aabb_0.min.x - aabb_1.max.x).max(aabb_1.min.x - aabb_0.max.x);
        let dy = (aabb_0.min.y - aabb_1.max.y).max(aabb_1.min.y - aabb_0.max.y);
        let dz = (aabb_0.min.z - aabb_1.max.z).max(aabb_1.min.z - aabb_0.max.z);
        let gap_uu = dx.max(dy).max(dz) * crate::consts::BT_TO_UU;
        let d_uu = (org_0_trans.translation - org_1_trans.translation).length()
            * crate::consts::BT_TO_UU;
        super::box_box_detector::store_last_sat(gap_uu, 0, 4);
        if trace || super::box_box_detector::sat_gap_dump() {
            eprintln!("SATGAP hit=0 gap={gap_uu:+.2} axis=0 aabb=1 d={d_uu:.1}");
        }
        return None;
    }

    let child_a_trans = &compound_a_shape.child_trans;
    let child_a_world_trans = org_0_trans * child_a_trans;

    let child_b_trans = &compound_b_shape.child_trans;
    let child_b_world_trans = org_1_trans * child_b_trans;

    let mut detector = BoxBoxDetector {
        box1: &compound_a_shape.child_shape,
        col1: compound_a_obj,
        box2: &compound_b_shape.child_shape,
        col2: compound_b_obj,
        contact_added_callback,
    };

    let res = detector.get_closest_points(child_a_world_trans, child_b_world_trans);
    if trace {
        let d = (org_0_trans.translation - org_1_trans.translation).length()
            * crate::consts::BT_TO_UU;
        match &res {
            Some(m) => eprintln!(
                "OBBOBB hit npts={} d={d:.1} pens={:?}",
                m.point_cache.len(),
                m.point_cache
                    .iter()
                    .map(|p| (p.distance_1 * crate::consts::BT_TO_UU * 100.0).round() / 100.0)
                    .collect::<Vec<_>>()
            ),
            None => eprintln!("OBBOBB detector-miss d={d:.1}"),
        }
    }
    res
}
