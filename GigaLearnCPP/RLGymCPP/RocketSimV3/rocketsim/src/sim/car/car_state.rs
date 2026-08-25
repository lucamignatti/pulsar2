use std::ops::{Deref, DerefMut};

use glam::{Mat3A, Vec3A};

use crate::{CarControls, PhysState, consts};

#[derive(Clone, Copy, Debug)]
pub struct CarState {
    pub phys: PhysState,
    /// Controls to simulate the car with
    pub controls: CarControls,
    /// Controls from the last time this car was simulated (equals `controls` after step)
    pub prev_controls: CarControls,
    /// True if 3 or more wheels have contact
    pub is_on_ground: bool,
    /// Whether each of the 4 wheels have contact
    /// First two are front
    /// If your car has 3 wheels, the 4th bool will always be false
    pub wheels_with_contact: [bool; 4],
    /// Per-wheel suspension compression in [0, 1] (0 = fully compressed, 1 = fully extended).
    /// Wheel order matches `wheels_with_contact` (FR, FL, BR, BL).
    pub wheels_suspension: [f32; 4],
    /// Whether we jumped to get into the air
    ///
    /// Can be false while airborne, if we left the ground with a flip reset
    pub has_jumped: bool,
    /// True if we have double jumped and are still in the air
    pub has_double_jumped: bool,
    /// True if we are in the air, and (have flipped or are currently flipping)
    pub has_flipped: bool,
    /// Relative torque direction of the flip
    ///
    /// Forward flip will have positive Y
    pub flip_rel_torque: Vec3A,
    /// Ticks since the current jump started (0 if not in a jump)
    pub jump_ticks: u32,
    /// Ticks since the current flip started (0 if not flipping)
    pub flip_ticks: u32,
    /// True during a flip (not an auto-flip, and not after a flip)
    pub is_flipping: bool,
    /// True during a jump
    pub is_jumping: bool,
    /// Total ticks spent in the air
    pub air_ticks: u32,
    /// Ticks spent in the air once `!is_jumping`
    ///
    /// If we never jumped, it is 0
    pub air_ticks_since_jump: u32,
    /// Goes from 0 to 100
    pub boost: f32,
    /// Used for recharge boost, counts up from 0 on spawn
    pub ticks_since_boosted: u32,
    /// True if we boosted that tick
    ///
    /// There exists a minimum boosting time, thus why we must track boosting time
    pub is_boosting: bool,
    pub boosting_ticks: u32,
    pub is_supersonic: bool,
    /// Ticks since the car's speed dropped below `START_SPEED` while still supersonic,
    /// used for the supersonic maintain grace period
    pub supersonic_grace_ticks: u32,
    /// This is a state variable due to the rise/fall rate of handbrake inputs
    pub handbrake_val: f32,
    pub is_auto_flipping: bool,
    /// Remaining auto-flip ticks (counts down)
    pub auto_flip_ticks: u32,
    pub auto_flip_torque_scale: f32,
    pub bump_cooldown_ticks: u32,
    /// RL's bump rate limit is PER VICTIM (`CarInteraction.LastHitCar` + `BumpInterval`
    /// in the ShouldDemolish decompile): hitting a DIFFERENT car is never blocked by
    /// the cooldown. Stores `1 + victim arena index` of the last bumped car; 0 = none.
    pub bump_last_victim: u32,
    /// Tick when this car last applied the psyonix ball-hit extra impulse. The 1-tick
    /// repeat gate is PER CAR (v2 keeps it in `car->_internalState.ballHitInfo`), NOT
    /// global to the ball: two cars striking the same tick BOTH apply, and their
    /// impulses sum -- a ball-global gate silently dropped the second car's impulse in
    /// every kickoff pinch / 50-50 (SIM2REAL_AUDIT.md S44).
    pub ball_extra_impulse_tick: Option<u64>,
    /// If in contact with a static mesh/body, this is the collision normal of that contact on said body
    pub world_contact_normal: Option<Vec3A>,
    pub is_demoed: bool,
    pub demo_respawn_ticks: u32,
}

impl Default for CarState {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl CarState {
    pub const DEFAULT: Self = Self {
        phys: PhysState {
            pos: Vec3A::new(0.0, 0.0, consts::car::spawn::REST_Z),
            rot_mat: Mat3A::IDENTITY,
            vel: Vec3A::ZERO,
            ang_vel: Vec3A::ZERO,
        },
        controls: CarControls::DEFAULT,
        prev_controls: CarControls::DEFAULT,
        is_on_ground: true,
        wheels_with_contact: [false; 4],
        wheels_suspension: [0.0; 4],
        has_jumped: false,
        has_double_jumped: false,
        has_flipped: false,
        flip_rel_torque: Vec3A::ZERO,
        jump_ticks: 0,
        flip_ticks: 0,
        is_flipping: false,
        is_jumping: false,
        air_ticks: 0,
        air_ticks_since_jump: 0,
        boost: consts::car::boost::SPAWN_AMOUNT,
        ticks_since_boosted: 0,
        is_boosting: false,
        boosting_ticks: 0,
        is_supersonic: false,
        supersonic_grace_ticks: 0,
        handbrake_val: 0.0,
        is_auto_flipping: false,
        world_contact_normal: None,
        bump_cooldown_ticks: 0,
        bump_last_victim: 0,
        ball_extra_impulse_tick: None,
        auto_flip_ticks: 0,
        auto_flip_torque_scale: 0.0,
        is_demoed: false,
        demo_respawn_ticks: 0,
    };

    #[must_use]
    pub const fn has_flip_or_jump(&self) -> bool {
        self.is_on_ground
            || (!self.has_flipped
                && !self.has_double_jumped
                && self.air_ticks_since_jump < consts::car::jump::DOUBLEJUMP_MAX_TICKS)
    }

    #[must_use]
    pub const fn has_flip_reset(&self) -> bool {
        !self.is_on_ground && self.has_flip_or_jump() && !self.has_jumped
    }

    #[must_use]
    pub const fn got_flip_reset(&self) -> bool {
        !self.is_on_ground && !self.has_jumped
    }

    #[inline]
    #[must_use]
    pub const fn jump_time(&self) -> f32 {
        consts::ticks_to_secs(self.jump_ticks)
    }

    #[inline]
    pub const fn set_jump_time(&mut self, secs: f32) {
        self.jump_ticks = consts::secs_to_ticks(secs);
    }

    #[inline]
    #[must_use]
    pub const fn flip_time(&self) -> f32 {
        consts::ticks_to_secs(self.flip_ticks)
    }

    #[inline]
    pub const fn set_flip_time(&mut self, secs: f32) {
        self.flip_ticks = consts::secs_to_ticks(secs);
    }

    #[inline]
    #[must_use]
    pub const fn air_time(&self) -> f32 {
        consts::ticks_to_secs(self.air_ticks)
    }

    #[inline]
    #[must_use]
    pub const fn air_time_since_jump(&self) -> f32 {
        consts::ticks_to_secs(self.air_ticks_since_jump)
    }

    #[inline]
    #[must_use]
    pub const fn time_since_boosted(&self) -> f32 {
        consts::ticks_to_secs(self.ticks_since_boosted)
    }

    #[inline]
    #[must_use]
    pub const fn boosting_time(&self) -> f32 {
        consts::ticks_to_secs(self.boosting_ticks)
    }

    #[inline]
    #[must_use]
    pub const fn supersonic_grace_timer(&self) -> f32 {
        consts::ticks_to_secs(self.supersonic_grace_ticks)
    }

    #[inline]
    #[must_use]
    pub const fn auto_flip_timer(&self) -> f32 {
        consts::ticks_to_secs(self.auto_flip_ticks)
    }

    #[inline]
    #[must_use]
    pub const fn bump_cooldown_timer(&self) -> f32 {
        consts::ticks_to_secs(self.bump_cooldown_ticks)
    }

    #[inline]
    #[must_use]
    pub const fn demo_respawn_timer(&self) -> f32 {
        consts::ticks_to_secs(self.demo_respawn_ticks)
    }

    #[must_use]
    pub fn num_wheels_in_contact(&self) -> usize {
        let mut result = 0;
        for b in self.wheels_with_contact {
            if b {
                result += 1;
            }
        }
        result
    }
}

impl Deref for CarState {
    type Target = PhysState;
    fn deref(&self) -> &Self::Target {
        &self.phys
    }
}

impl DerefMut for CarState {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.phys
    }
}
