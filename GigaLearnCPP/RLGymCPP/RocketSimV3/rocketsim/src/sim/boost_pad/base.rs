use glam::Vec3A;

use crate::{BoostPadConfig, MutatorConfig, consts::boost_pads, shared::Aabb};

#[derive(Debug, Copy, Clone)]
pub(crate) struct BoostPad {
    pub config: BoostPadConfig,
    pub box_radius: f32,
    pub cyl_radius: f32,
    pub max_cooldown: f32,
    pub boost_amount: f32,
    pub aabb: Aabb,
    pub gave_boost_tick_count: Option<i64>,
}

impl BoostPad {
    pub fn new(config: BoostPadConfig, mutator_config: &MutatorConfig) -> Self {
        let box_radius = if config.is_big {
            boost_pads::BOX_RAD_BIG
        } else {
            boost_pads::BOX_RAD_SMALL
        };

        let cyl_radius = if config.is_big {
            boost_pads::CYL_RAD_BIG
        } else {
            boost_pads::CYL_RAD_SMALL
        };

        let max_cooldown = if config.is_big {
            mutator_config.boost_pad_cooldown_big
        } else {
            mutator_config.boost_pad_cooldown_small
        };

        let boost_amount = if config.is_big {
            mutator_config.boost_pad_amount_big
        } else {
            mutator_config.boost_pad_amount_small
        };

        // VENDOR PATCH (pulsar 2026-08-01): the BVH broad-phase AABB must cover the
        // CYLINDER used by the actual pickup test, not the (unused) box radius.
        // It was built with box_radius (120 small / 160 big) while
        // BoostPadGrid::process_node tests dist_2d < cyl_radius (144 / 208), so any pad
        // between those radii was culled by the broad phase and never tested. Measured on
        // real match telemetry: RocketSim missed 10 of 19 real pickups (53%), all on small
        // pads, at closest approaches of 123-170uu -- several INSIDE the 144uu cylinder.
        // See research/reports/SIM2REAL_AUDIT.md S14.
        let _ = box_radius;
        let extent = Vec3A::new(cyl_radius, cyl_radius, boost_pads::CYL_HEIGHT);
        let aabb = Aabb::new(config.pos - extent, config.pos + extent);

        Self {
            config,
            box_radius,
            cyl_radius,
            max_cooldown,
            boost_amount,
            aabb,
            gave_boost_tick_count: None,
        }
    }

    pub const fn reset(&mut self) {
        self.gave_boost_tick_count = None;
    }

    pub const fn config(&self) -> &BoostPadConfig {
        &self.config
    }

    pub const fn aabb(&self) -> Aabb {
        self.aabb
    }
}
