use glam::Vec3A;

use crate::{
    BoostPad, BoostPadConfig, CarState, MutatorConfig,
    consts::{TICK_TIME, boost_pads},
    shared::{Aabb, bvh},
};

pub struct BoostPadProcessor<'a> {
    all_pads: &'a mut [BoostPad],
    car_state: &'a mut CarState,
    mutator_config: &'a MutatorConfig,
    tick_count: u64,
    pad_idx: Option<usize>,
    /// VENDOR PATCH (pulsar 2026-08-01): full hitbox size of the car body.
    car_hitbox_size: Vec3A,
    /// Hitbox centre offset in car-local space (S44): the collision box is NOT
    /// centred on the rigid-body origin (Octane: +13.88 fwd, +20.75 up).
    car_hitbox_offset: Vec3A,
    /// Arena index of the car being tested, recorded into `pending_grant`.
    car_idx: usize,
}

impl bvh::ProcessNode for BoostPadProcessor<'_> {
    fn process_node(&mut self, pad_idx: usize) {
        if self.pad_idx.is_some() {
            return; // Already found a pad to give boost from, no need to check more
        }

        let pad = &mut self.all_pads[pad_idx];

        if pad.pending_grant.is_some() {
            return; // Already claimed by a touch awaiting its grant
        }
        if let Some(last_give_tick_count) = pad.gave_boost_tick_count
            && ((self.tick_count as i64 - last_give_tick_count) as f32 * TICK_TIME)
                < pad.max_cooldown
        {
            return;
        }

        let pad_pos = pad.config().pos;

        // VENDOR PATCH (pulsar 2026-08-01): test the pad cylinder against the CAR BODY,
        // not the car's origin point.
        //
        // Rocket League decides pickups from UE3's `Touching` array
        // (`AVehiclePickup_TA::IsTouchingAVehicle` -> RocketLeague.exe FUN_140f0d590),
        // i.e. a collision-primitive overlap between the pad volume and the car's
        // collision body. Testing only the origin misses pickups where the car body
        // covers the pad: measured on real telemetry, RocketSim missed 10 of 19 real
        // pickups, at closest origin approaches of 146-170uu against a 144uu radius.
        //
        // The car's ORIENTED box is used deliberately -- an axis-aligned box of a rotated
        // car is much larger than the car and over-triggers badly (a tested AABB variant
        // produced 116 false pickups in 1508 frames).
        // See research/reports/SIM2REAL_AUDIT.md S14.
        let half = self.car_hitbox_size * 0.5;
        let rot = self.car_state.rot_mat;
        // Box CENTRE = rigid-body origin + local hitbox offset (S44). Testing an
        // origin-centred box missed real big-pad pickups where the car flew over the
        // pad edge (2/153 in the 120Hz capture, 13-17uu outside the origin box but
        // inside the offset box; the only events the offset box adds sit <1uu inside
        // the radius -- quantization-level boundary noise).
        let center = self.car_state.pos
            + rot.x_axis * self.car_hitbox_offset.x
            + rot.y_axis * self.car_hitbox_offset.y
            + rot.z_axis * self.car_hitbox_offset.z;
        let rel = pad_pos - center;
        // pad centre in car-local space, then clamped onto the box => closest point
        let local = Vec3A::new(rel.dot(rot.x_axis), rel.dot(rot.y_axis), rel.dot(rot.z_axis));
        let clamped = local.clamp(-half, half);
        let closest =
            center + rot.x_axis * clamped.x + rot.y_axis * clamped.y + rot.z_axis * clamped.z;

        let dist_sq_2d = pad_pos.truncate().distance_squared(closest.truncate());
        // Radius is BOX_RAD (120 small / 160 big), not CYL_RAD (144/208): the larger
        // cylinder radii are origin-test approximations that already bake in typical car
        // body reach, so pairing them with a body test double-counts it. Using the box
        // radii -- previously dead constants -- against the car body reproduces the game.
        let overlapping = dist_sq_2d < pad.box_radius.powi(2)
            && (closest.z - pad_pos.z).abs() <= boost_pads::CYL_HEIGHT;
        if overlapping {
            // Touch: the grant lands GRANT_DELAY_TICKS later (S42) -- Rocket League
            // routes pickups through the UE3 touch-event pipeline, so the boost (and
            // the cooldown) start ~2 ticks after first overlap, not at overlap.
            pad.pending_grant =
                Some((self.tick_count + boost_pads::GRANT_DELAY_TICKS, self.car_idx));
            self.pad_idx = Some(pad_idx);
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct BoostPadGrid {
    pub bvh_tree: bvh::Tree,
    pub all_pads: Vec<BoostPad>,
    pub max_pad_z: f32,
}

impl BoostPadGrid {
    pub fn new(pad_configs: &[BoostPadConfig], mutator_config: &MutatorConfig) -> Self {
        assert!(!pad_configs.is_empty());

        let mut all_pads: Vec<BoostPad> = pad_configs
            .iter()
            .map(|&pad_config| BoostPad::new(pad_config, mutator_config))
            .collect();

        // Sort them to match RLBot/RLGym ordering
        all_pads.sort_by(|a, b| {
            let a_pos = a.config.pos;
            let b_pos = b.config.pos;
            match a_pos.y.total_cmp(&b_pos.y) {
                std::cmp::Ordering::Equal => a_pos.x.total_cmp(&b_pos.x),
                other => other,
            }
        });

        let all_aabb = {
            let mut all_aabb_accum: Option<Aabb> = None;

            for pad in &all_pads {
                let pad_aabb = pad.aabb();
                if let Some(all_aabb) = all_aabb_accum {
                    all_aabb_accum = Some(all_aabb.combine(&pad_aabb));
                } else {
                    all_aabb_accum = Some(pad_aabb);
                }
            }
            all_aabb_accum.unwrap()
        };

        let mut aabb_nodes = Vec::new();
        for (i, pad) in all_pads.iter().enumerate() {
            let node = bvh::Node {
                aabb: pad.aabb(),
                node_type: bvh::BvhNodeType::Leaf { leaf_idx: i },
            };
            aabb_nodes.push(node);
        }

        let bvh_tree = bvh::Tree::build(all_aabb, &mut aabb_nodes);

        Self {
            bvh_tree,
            all_pads,
            max_pad_z: all_aabb.max.z,
        }
    }

    pub fn reset(&mut self) {
        for pad in &mut self.all_pads {
            pad.reset();
        }
    }

    /// If boost was collected, returns the pad index
    pub(crate) fn maybe_give_car_boost(
        &mut self,
        car_state: &mut CarState,
        mutator_config: &MutatorConfig,
        tick_count: u64,
        car_hitbox_size: Vec3A,
        car_hitbox_offset: Vec3A,
        car_idx: usize,
    ) -> Option<usize> {
        if car_state.boost >= mutator_config.car_max_boost_amount {
            return None; // Already full on boost
        }

        if car_state.pos.z > self.max_pad_z {
            return None; // Can't possibly overlap with a boost pad
        }

        // VENDOR PATCH: broad-phase must cover the car body, matching the test below.
        // Padded by the hitbox offset magnitude so the offset-centred box (S44) is
        // always inside the queried AABB regardless of orientation.
        let body_reach = car_hitbox_size * 0.5 + Vec3A::splat(car_hitbox_offset.length());
        let car_center_aabb = Aabb::new(car_state.pos - body_reach, car_state.pos + body_reach);

        let mut pad_processor = BoostPadProcessor {
            all_pads: &mut self.all_pads,
            car_state,
            mutator_config,
            tick_count,
            pad_idx: None,
            car_hitbox_size,
            car_hitbox_offset,
            car_idx,
        };
        self.bvh_tree
            .report_aabb_overlapping_node(&mut pad_processor, &car_center_aabb);

        pad_processor.pad_idx
    }
}
