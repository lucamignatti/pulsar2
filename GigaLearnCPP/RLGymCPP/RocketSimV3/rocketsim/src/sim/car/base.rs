use std::{
    env,
    f32::consts::PI,
    ops::{Deref, DerefMut},
    sync::OnceLock,
};

fn env_on(name: &'static str) -> bool {
    static LOCKS: OnceLock<std::sync::Mutex<Vec<(String, bool)>>> = OnceLock::new();
    let cache = LOCKS.get_or_init(|| std::sync::Mutex::new(Vec::new()));
    let mut g = cache.lock().unwrap();
    if let Some((_, v)) = g.iter().find(|(n, _)| n == name) {
        return *v;
    }
    let v = env::var(name).is_ok_and(|s| s != "0");
    g.push((name.to_string(), v));
    v
}

fn env_f32(name: &'static str, default: f32) -> f32 {
    static LOCKS: OnceLock<std::sync::Mutex<Vec<(String, f32)>>> = OnceLock::new();
    let cache = LOCKS.get_or_init(|| std::sync::Mutex::new(Vec::new()));
    let mut g = cache.lock().unwrap();
    if let Some((_, v)) = g.iter().find(|(n, _)| n == name) {
        return *v;
    }
    let v = env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default);
    g.push((name.to_string(), v));
    v
}

/// Tuned wheel order: suspension/friction impulses at this tick's pose BEFORE the
/// Bullet step (with the end-of-tick raycast reused next tick). Part of the winning
/// stack (SIM2REAL_LOOP.md 2026-08-23 22:00) together with apply-time forces,
/// no graze-damp fade, tuned ray, steer-at-apply, and uncapped pushback.
/// DEFAULT ON since 2026-08-23; `GGL_WHEELS_PRE=0` restores post-step wheels.
fn ggl_legacy_dodge_gate() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| env::var("GGL_LEGACY_DODGE_GATE").is_ok_and(|s| s != "0"))
}

fn ggl_no_touchdown_exc() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| env::var("GGL_NO_TOUCHDOWN_EXC").is_ok_and(|s| s != "0"))
}

/// Extra suspension extension (uu) beyond rest length within which a ray-touching
/// wheel still counts as CONTACT for the DODGE PRESS GATE. DEFAULT -1 = DISABLED:
/// the press gate uses is_on_ground, and that is MEASURED CORRECT - replaying 625
/// real press edges, is_on_ground matches the real fire-rate-by-z curve at 91.8%
/// per-event agreement (mean bin error 3.1%); every shorter-reach variant broke it
/// (research/tools/dodge_gate_sweep.py). The z~31-33 over-eating seen in the
/// flat-ground smoke test does not occur on real poses: tilted cars drop below
/// 3-wheel contact exactly where the real game frees the dodge. Knob kept for
/// future refits only.
fn ggl_dodge_contact_ext_uu() -> f32 {
    static V: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        env::var("GGL_DODGE_CONTACT_EXT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(-1.0)
    })
}

/// Minimum wheels in contact for the DODGE PRESS gate (not is_on_ground, which stays
/// `>= 3` for air control / throttle - routing ascent-contact ticks elsewhere doubled
/// replay error once). 3 reproduces legacy behaviour; 1 = any wheel still touching eats
/// the press. Fit against the real press population, see the gate site below.
/// Restrict gate v3's takeoff guard so a genuine landing still restores the flip.
/// See the guard site for the measurement. GGL_FLIP_RESET_FIX=0 restores the v3 guard.
fn ggl_flip_reset_fix() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| env::var("GGL_FLIP_RESET_FIX").map_or(true, |s| s != "0"))
}

/// Eat the dodge press for N ticks after is_on_ground breaks (0 = off). DEFAULT 1: Fit on the FREE-RUNNING 120 Hz maneuver battery (restore-based replay is not
/// valid for this knob - the restored car's ray state differs from a free-running one,
/// which is what falsified the first 1-tick attempt). The real game keeps eating
/// presses 1-2 ticks after this engine's contact breaks on level fast takeoffs
/// (speed_flip: real ate a press the engine converts, 573 uu divergence; boundary
/// population real fires 18% vs engine 92%). Suspension-ray reach CANNOT express this:
/// at those press ticks no wheel ray reaches the ground at all (GGL_DODGE_CONTACT_EXT
/// 2/4/6 byte-identical on the battery), so the hold is tick-based.
/// Attitude gate for the contact hold: apply only while the car's up-vector z is at
/// least this (level takeoffs). Tilted poses keep the plain contact gate, which already
/// matches the real z-curve at 92.5%+.
fn ggl_dodge_hold_min_upz() -> f32 {
    static V: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        env::var("GGL_DODGE_HOLD_MIN_UPZ")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.95)
    })
}

fn ggl_dodge_ground_hold_ticks() -> u8 {
    static V: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        env::var("GGL_DODGE_GROUND_HOLD")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1)
    })
}

fn ggl_dodge_contact_wheels() -> u32 {
    static V: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        env::var("GGL_DODGE_CONTACT_WHEELS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(3)
    })
}

/// How many ticks after touchdown keep the post-step wheel pass. Fit on real-game
/// full-pose captures (see research/tools landing tests); default from that fit.
fn ggl_touchdown_exc_ticks() -> u8 {
    static V: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    // Default 0 since 2026-08-26: the =5 ship was fit on the 60 Hz misaligned captures
    // (the 1-tick comparison-harness offset found today inflated every landing number it
    // was fit against). On the alignment-corrected 120 Hz battery exc=0 is better or
    // equal on EVERY landing segment (corner_land_steep p90 122->8.8, tilt_nose_down
    // 53->2.0, plain drops unchanged). Titan-appo reached the same verdict independently.
    *V.get_or_init(|| env::var("GGL_TOUCHDOWN_EXC_TICKS").ok().and_then(|s| s.parse().ok()).unwrap_or(0))
}

fn ggl_wheels_pre() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| env::var("GGL_WHEELS_PRE").map_or(true, |s| s != "0"))
}

// (titan-appo carried duplicates of ggl_no_touchdown_exc / ggl_touchdown_exc_ticks here
//  with default 0; ours above, default 5, is the capture-fit ship. Port 2026-08-26.)

/// Where car speed caps run. Default `both` = arena tick head + publish (S37).
/// `GGL_VEL_CLAMP=head` = tuned default (start only). `end` = publish only.
fn ggl_vel_clamp_mode() -> &'static str {
    static V: OnceLock<String> = OnceLock::new();
    V.get_or_init(|| match env::var("GGL_VEL_CLAMP").ok().as_deref() {
        Some("head") => "head".to_string(),
        Some("end") => "end".to_string(),
        _ => "both".to_string(),
    })
    .as_str()
}

pub(crate) fn ggl_vel_clamp_head() -> bool {
    ggl_vel_clamp_mode() != "end"
}

pub(crate) fn ggl_vel_clamp_end() -> bool {
    ggl_vel_clamp_mode() != "head"
}

use fastrand::Rng;
use glam::{Affine3A, EulerRot, Mat3A, Vec3A};

use crate::bullet::dynamics::rigid_body::Impulse;
use crate::consts::GRAVITY_Z;
use crate::{
    CarBodyConfig, CarControls, CarState, GameMode, MutatorConfig, PhysState, Team,
    bullet::{
        collision::{
            broadphase::CollisionFilterGroups,
            shapes::{
                box_shape::BoxShape, collision_shape::CollisionShapes,
                compound_shape::CompoundShape,
            },
        },
        dynamics::{
            discrete_dynamics_world::DiscreteDynamicsWorld,
            rigid_body::{ActivationState, CollisionFlags, RigidBody, RigidBodyConstructionInfo},
            vehicle::{NUM_WHEELS, VehicleRL, WheelInfo},
        },
    },
    consts::{
        BT_TO_UU, TICK_TIME, UU_TO_BT, bullet_vehicle as vehicle_consts,
        car::{self as car_consts, drive as drive_consts},
        curves, secs_to_ticks, ticks_gt, ticks_until,
    },
    sim::{UserInfoTypes, car::car_info::CarInfo},
};

pub struct Car {
    /// Touchdown exception: run the wheel pass post-step for THIS tick only.
    pub(crate) wheels_post_this_tick: bool,
    /// Countdown: remaining ticks of post-step wheel pass after a touchdown.
    pub(crate) touchdown_post_ticks: u8,
    pub(crate) info: CarInfo,
    pub(crate) bullet_vehicle: VehicleRL,
    pub(crate) rigid_body_idx: usize,
    pub(crate) vel_impulse_cache: Vec3A,
    pub(crate) state: CarState,
    /// Last tick's world-contact sticky gate (PR73). Default false.
    pub(crate) sticky_gate_prev: bool,
    /// Ticks since is_on_ground last read true (saturating), for the dodge-press
    /// contact hold (see the gate site). Lives on Car, NOT CarState: within-tick
    /// bookkeeping only, so no serialized state / checkpoint / FFI layout change.
    pub(crate) ticks_since_ground: u8,
    /// Last tick's chassis world contact (GGL_PLANE_SLACK scrape gate).
    /// Persistence law: promote only if there was contact before. Default false;
    pub(crate) chassis_scrape_prev: bool,
}

impl Deref for Car {
    type Target = CarInfo;
    fn deref(&self) -> &Self::Target {
        &self.info
    }
}
impl DerefMut for Car {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.info
    }
}

impl Car {
    pub(crate) fn new(
        idx: usize,
        team: Team,
        bullet_world: &mut DiscreteDynamicsWorld,
        mutator_config: &MutatorConfig,
        config: CarBodyConfig,
    ) -> Self {
        let child_hitbox_shape = BoxShape::new(config.hitbox_size * UU_TO_BT * 0.5);
        let local_inertia = child_hitbox_shape.calculate_local_intertia(mutator_config.car_mass);

        let hitbox_offset = Affine3A {
            matrix3: Mat3A::IDENTITY,
            translation: config.hitbox_pos_offset * UU_TO_BT,
        };
        let compound_shape = CompoundShape::new(child_hitbox_shape, hitbox_offset);

        let collision_shape = CollisionShapes::Compound(compound_shape);
        let mut rb_info = RigidBodyConstructionInfo::new(mutator_config.car_mass, collision_shape);
        rb_info.friction = car_consts::BASE_COEFS.friction;
        rb_info.restitution = car_consts::BASE_COEFS.restitution;
        rb_info.start_world_trans = Affine3A::IDENTITY;
        rb_info.local_inertia = local_inertia;

        let mut body = RigidBody::new(rb_info);
        body.user_idx = UserInfoTypes::Car;
        body.collision_flags |= CollisionFlags::CustomMaterialCallback;

        let rigid_body_idx = bullet_world.add_rigid_body(
            body,
            CollisionFilterGroups::Default | CollisionFilterGroups::DropshotFloor,
            CollisionFilterGroups::ALL,
        );

        let mut wheels = [WheelInfo::DEFAULT; NUM_WHEELS];
        for (i, wheel) in wheels.iter_mut().enumerate() {
            let front = i < 2;
            let left = i % 2 != 0;

            let (wheel_config, suspension_force_scale) = if front {
                (
                    &config.front_wheels,
                    vehicle_consts::SUSPENSION_FORCE_SCALE_FRONT,
                )
            } else {
                (
                    &config.back_wheels,
                    vehicle_consts::SUSPENSION_FORCE_SCALE_BACK,
                )
            };

            let mut wheel_ray_start_offset = wheel_config.connection_point_offset;
            if left {
                wheel_ray_start_offset.y *= -1.0;
            }

            let suspension_rest_length =
                wheel_config.suspension_rest_length - vehicle_consts::MAX_SUSPENSION_TRAVEL;

            wheel.set_params(
                wheel_ray_start_offset * UU_TO_BT,
                suspension_rest_length * UU_TO_BT,
                wheel_config.wheel_radius * UU_TO_BT,
                suspension_force_scale,
            );
        }

        Self {
            wheels_post_this_tick: false,
            touchdown_post_ticks: 0,
            info: CarInfo { idx, team, config },
            rigid_body_idx,
            bullet_vehicle: VehicleRL::new(rigid_body_idx, wheels),
            vel_impulse_cache: Vec3A::ZERO,
            state: CarState {
                boost: mutator_config.car_spawn_boost_amount,
                ..Default::default()
            },
            sticky_gate_prev: false,
            ticks_since_ground: u8::MAX,
            chassis_scrape_prev: false,
        }
    }

    /// - `respawn_delay` by default is `rocketsim::consts::DEMO_RESPAWN_TIME`
    pub(crate) const fn demolish(&mut self, respawn_delay: f32) {
        self.state.is_demoed = true;
        self.state.demo_respawn_ticks = secs_to_ticks(respawn_delay);
    }

    /// - `boost_amount` by default is `rocketsim::consts::BOOST_RESPAWN_AMOUNT`
    ///
    /// Respawn the car, called after we have been demolished and waited for the respawn timer
    pub(crate) fn respawn(
        &mut self,
        rb: &mut RigidBody,
        rng: &mut Rng,
        game_mode: GameMode,
        boost_amount: f32,
    ) {
        let respawn_locations = car_consts::spawn::get_respawn_locations(game_mode);
        let spawn_pos_idx = rng.usize(0..respawn_locations.len());
        let spawn_pos = respawn_locations[spawn_pos_idx];

        let new_state = CarState {
            phys: PhysState {
                pos: Vec3A::new(
                    spawn_pos.x,
                    spawn_pos.y * -self.team.get_y_dir(),
                    car_consts::spawn::RESPAWN_Z,
                ),
                rot_mat: Mat3A::from_euler(
                    EulerRot::ZYX,
                    spawn_pos.yaw_ang + if self.team == Team::Blue { 0.0 } else { PI },
                    0.0,
                    0.0,
                ),
                vel: Vec3A::ZERO,
                ang_vel: Vec3A::ZERO,
            },
            boost: boost_amount,
            ..Default::default()
        };

        self.set_state(rb, &new_state);
    }

    #[must_use]
    pub const fn get_state(&self) -> &CarState {
        &self.state
    }

    #[must_use]
    pub const fn get_config(&self) -> &CarBodyConfig {
        &self.info.config
    }

    pub const fn set_controls(&mut self, new_controls: CarControls) {
        self.state.controls = new_controls;
    }

    pub(crate) fn set_state(&mut self, rb: &mut RigidBody, state: &CarState) {
        debug_assert_eq!(rb.user_idx, UserInfoTypes::Car);
        debug_assert_eq!(rb.world_array_idx, self.rigid_body_idx);

        rb.set_world_trans(Affine3A {
            matrix3: state.phys.rot_mat,
            translation: state.phys.pos * UU_TO_BT,
        });

        rb.lin_vel = state.phys.vel * UU_TO_BT;
        rb.ang_vel = state.phys.ang_vel;
        rb.update_inertia_tensor();

        self.vel_impulse_cache = Vec3A::ZERO;
        self.state = *state;
        self.sticky_gate_prev = self.state.wheels_with_contact.iter().any(|&c| c);
        // A restored car has no prior tick; seed from the restored flag so a SetState
        // never manufactures a spurious contact-break (replay probes restore mid-jump).
        self.ticks_since_ground = if self.state.is_on_ground { 0 } else { u8::MAX };
        self.wheels_post_this_tick = false;
        self.touchdown_post_ticks = 0;
    }

    /// Re-raycast wheels at the car's current pose and refresh contact flags.
    ///
    /// `From<CarRecord>` leaves `wheels_with_contact` false. Without this, the
    /// next tick treats a grounded car as 0 wheels in contact (drive torque /4,
    /// no sticky, chassis-world friction on). Tuned calls this from every
    /// `set_car_state`. Disable with `GGL_NO_REFRESH_CONTACT=1`.
    pub(crate) fn refresh_contact_state(
        &mut self,
        collision_world: &mut DiscreteDynamicsWorld,
    ) {
        if self.state.is_demoed {
            return;
        }
        self.bullet_vehicle
            .update_vehicle_first(collision_world, TICK_TIME);
        let mut n = 0u8;
        for (i, wheel) in self.bullet_vehicle.wheels.iter().enumerate() {
            let hit = wheel.raycast_info.is_some();
            self.state.wheels_with_contact[i] = hit;
            n += u8::from(hit);
        }
        self.state.is_on_ground = n >= 3;
        collision_world.bodies_mut()[self.rigid_body_idx].wheels_grounded = n >= 3;
        self.sticky_gate_prev = self.bullet_vehicle.wheels.iter().any(|wheel| {
            wheel.adhesion_contact
                || wheel
                    .raycast_info
                    .as_ref()
                    .is_some_and(|info| info.is_in_contact_with_world)
        });
    }

    /////////////////////////////

    fn update_wheels(
        &mut self,
        rb: &mut RigidBody,
        num_wheels_in_contact: usize,
        forward_speed_uu: f32,
    ) {
        let handbrake_delta = if self.state.controls.handbrake {
            drive_consts::POWERSLIDE_RISE_RATE
        } else {
            -drive_consts::POWERSLIDE_FALL_RATE
        } * TICK_TIME;
        self.state.handbrake_val = (self.state.handbrake_val + handbrake_delta).clamp(0.0, 1.0);

        let mut real_brake = 0.0;
        let real_throttle = if self.state.controls.boost && self.state.boost > 0.0 {
            1.0
        } else {
            self.state.controls.throttle
        };

        let abs_forward_speed_uu = forward_speed_uu.abs();
        let mut engine_throttle = real_throttle;
        if !self.state.controls.handbrake {
            if real_throttle.abs() >= drive_consts::THROTTLE_DEADZONE {
                if abs_forward_speed_uu > drive_consts::STOPPING_FORWARD_VEL
                    && real_throttle.signum() != forward_speed_uu.signum()
                {
                    real_brake = 1.0;

                    if abs_forward_speed_uu > drive_consts::BRAKING_NO_THROTTLE_SPEED_THRESH {
                        engine_throttle = 0.0;
                    }
                }
            } else {
                engine_throttle = 0.0;
                real_brake = if abs_forward_speed_uu < drive_consts::STOPPING_FORWARD_VEL {
                    1.0
                } else {
                    drive_consts::COASTING_BRAKE_FACTOR
                };
            }
        }

        let mut drive_speed_scale =
            curves::DRIVE_SPEED_TORQUE_FACTOR.get_output(abs_forward_speed_uu);
        if num_wheels_in_contact < 3 {
            // Shipped RL: 1/4 drive with <3 wheels. GGL_DRIVE_PARTIAL overrides
            // the divisor (1 = full torque, 2 = half). Air+wheels 1-tick error
            // is forward-short; this is the first A/B for that bucket.
            drive_speed_scale /= env_f32("GGL_DRIVE_PARTIAL", 4.0);
        }
        // Packed+tilted 4-wheel fillets never hit DRIVE_PARTIAL. 3-step T7
        // packed thr+ accumulates +fwd (sim too fast). GGL_FILLET_DRIVE>1
        // divides engine force when last-tick extra_pushback fired and
        // |up.z| is in the fillet band (not floor ~1, not true wall ~0).
        // Rejected 2026-08-24: true walls bit-identical, packed thr+ 1.23→2.03,
        // uncomp fillet p90 0.89→4.12 (pushback also fires there). 1-step is
        // not over-drive.
        let fillet_div = env_f32("GGL_FILLET_DRIVE", 1.0);
        if fillet_div > 1.0 {
            let uz = self.state.get_up_dir().z.abs();
            let packed = self
                .bullet_vehicle
                .wheels
                .iter()
                .any(|w| w.extra_pushback.abs() > 0.0);
            if packed && (0.25..0.90).contains(&uz) {
                drive_speed_scale /= fillet_div;
            }
        }

        let drive_engine_force = engine_throttle
            * const { drive_consts::THROTTLE_TORQUE_AMOUNT * UU_TO_BT }
            * drive_speed_scale;
        let drive_brake_force = real_brake * const { drive_consts::BRAKE_TORQUE_AMOUNT * UU_TO_BT };
        for wheel in &mut self.bullet_vehicle.wheels {
            wheel.engine_force = drive_engine_force;
            wheel.brake = drive_brake_force;
        }

        let mut steer_angle = if self.config.three_wheels {
            curves::STEER_ANGLE_FROM_SPEED_THREEWHEEL.get_output(abs_forward_speed_uu)
        } else {
            curves::STEER_ANGLE_FROM_SPEED.get_output(abs_forward_speed_uu)
        };
        if self.state.handbrake_val != 0.0 {
            steer_angle += (curves::POWERSLIDE_STEER_ANGLE_FROM_SPEED
                .get_output(abs_forward_speed_uu)
                - steer_angle)
                * self.state.handbrake_val;
        }

        steer_angle *= self.state.controls.steer;
        self.bullet_vehicle.wheels[0].steer_angle = steer_angle;
        self.bullet_vehicle.wheels[1].steer_angle = steer_angle;

        let car_pos = rb.get_world_pos();
        let car_vel = rb.lin_vel;
        let car_ang_vel = rb.ang_vel;

        for wheel in &mut self.bullet_vehicle.wheels {
            let Some(raycast_info) = wheel.raycast_info.as_ref() else {
                continue;
            };

            if !raycast_info.is_in_contact_with_world {
                continue;
            }

            let lat_dir = wheel.axle_dir;
            let long_dir = lat_dir.cross(raycast_info.contact_normal);

            let wheel_delta = if env_on("GGL_FRIC_LEVER") {
                raycast_info.contact_point - car_pos
            } else {
                wheel.hard_point - car_pos
            };
            let cross_vec = (car_ang_vel.cross(wheel_delta) + car_vel) * BT_TO_UU;

            let base_friction = cross_vec.dot(lat_dir).abs();
            let friction_curve_input = if base_friction > 5.0 {
                base_friction / (cross_vec.dot(long_dir).abs() + base_friction)
            } else {
                0.0
            };
            wheel.last_friction_curve_input = friction_curve_input;

            let mut lat_friction = if self.info.config.three_wheels {
                curves::LAT_FRICTION_THREEWHEEL
            } else {
                curves::LAT_FRICTION
            }
            .get_output(friction_curve_input);
            let mut long_friction = 1.0;

            if self.state.handbrake_val != 0.0 {
                let handbrake_amount = self.state.handbrake_val;
                lat_friction *= 1.0
                    + (curves::HANDBRAKE_LAT_FRICTION_FACTOR.get_output(friction_curve_input)
                        - 1.0)
                        * handbrake_amount;
                long_friction *= 1.0
                    + (curves::HANDBRAKE_LONG_FRICTION_FACTOR.get_output(friction_curve_input)
                        - 1.0)
                        * handbrake_amount;
            }

            if real_throttle == 0.0 {
                // Contact is not sticky
                let non_sticky_scale =
                    curves::NON_STICKY_FRICTION_FACTOR.get_output(raycast_info.contact_normal.z);
                lat_friction *= non_sticky_scale;
                long_friction *= non_sticky_scale;
            }

            if !env_on("GGL_FRIC_APPLY") {
                wheel.lat_friction = lat_friction;
                wheel.long_friction = long_friction;
            }
        }

        let n_sticky_wheels = self
            .bullet_vehicle
            .wheels
            .iter()
            .filter(|wheel| {
                wheel.adhesion_contact
                    || wheel
                        .raycast_info
                        .as_ref()
                        .is_some_and(|info| info.is_in_contact_with_world)
            })
            .count();
        // GGL_STICKY_FADE: SUPERSEDED, default off. Fades sticky over the last
        // stretch of suspension travel. This was a proxy fit to the T1
        // extension-vs-bias curve before the true variable was found: the "fade"
        // was really P(no contact one tick earlier | extension) -- the prev-tick
        // gate above beats every fade shape on both tapes. Kept for A/B only.
        // Modes: unset/0 = off, "cut" = hard zero above 0.875 travel, "lin" =
        // linear fade over [0.875, 1.0], "quad" = that squared.
        let sticky_fade = {
            static V: OnceLock<u8> = OnceLock::new();
            *V.get_or_init(|| match env::var("GGL_STICKY_FADE").as_deref() {
                Ok("cut") => 1,
                Ok("lin") => 2,
                Ok("quad") => 3,
                _ => 0,
            })
        };
        // Flip exemption: the same extension bins measured NO bias during flips
        // (flip+wheels ext 11.5-12 med_up -0.032 vs air+wheels -2.61) -- the real
        // game keeps full sticky on a grazing wheel while dodging.
        let sticky_fade_scale = if sticky_fade == 0 || self.state.is_flipping {
            1.0
        } else {
            let travel = const { vehicle_consts::MAX_SUSPENSION_TRAVEL * UU_TO_BT };
            let mut min_ext_frac = f32::INFINITY;
            for wheel in &self.bullet_vehicle.wheels {
                if let Some(info) = wheel.raycast_info.as_ref()
                    && info.is_in_contact_with_world
                {
                    let ext = (info.suspension_length - wheel.suspension_rest_length_1) / travel;
                    min_ext_frac = min_ext_frac.min(ext);
                }
            }
            if min_ext_frac.is_infinite() {
                1.0
            } else {
                let lin = ((1.0 - min_ext_frac) / 0.125).clamp(0.0, 1.0);
                match sticky_fade {
                    1 => f32::from(min_ext_frac <= 0.875),
                    2 => lin,
                    _ => lin * lin,
                }
            }
        };
        // GGL_STICKY_PREV: DEFAULT ON since 2026-08-23 (`=0` restores fresh-ray gate).
        // Sticky requires contact on the PREVIOUS tick too (upstream PR73: "a car
        // doesn't stick on its spawn tick"). Independently derived from T1: single
        // grazing wheels carry the full 0.5*g*dt sticky bias when FALLING (no contact
        // one tick earlier, med_up -2.53) and none when RISING (had contact, +0.005)
        // -- the extension-fade model this replaces was a proxy for exactly that.
        // Conjunction with current contact keeps the upwards-dir well-defined.
        let sticky_prev_gate = {
            static V: OnceLock<bool> = OnceLock::new();
            *V.get_or_init(|| env::var("GGL_STICKY_PREV").map_or(true, |s| s != "0"))
        };
        let sticky_ok = if sticky_prev_gate {
            self.sticky_gate_prev
                && n_sticky_wheels >= env_f32("GGL_STICKY_MIN_WHEELS", 1.0) as usize
        } else {
            n_sticky_wheels >= env_f32("GGL_STICKY_MIN_WHEELS", 1.0) as usize
        };
        if sticky_ok {
            let upwards_dir = self.bullet_vehicle.get_upwards_dir_from_wheel_contacts(rb);

            let full_stick = real_throttle != 0.0
                || abs_forward_speed_uu > car_consts::drive::STOPPING_FORWARD_VEL;
            let mut sticky_force_scale = f32::from(!self.config.three_wheels) * 0.5;
            // Extra 1-|up.z| term is ~+1.0 on walls. GGL_NO_STICKY_TILT=1 drops it.
            if full_stick && !env_on("GGL_NO_STICKY_TILT") {
                sticky_force_scale += 1.0 - upwards_dir.z.abs();
            }

            // GGL_STICKY_SCALE: 1 = shipped 0.5·g on a level Octane. Ground 1-tick
            // error is ~+3 uu/s along up (we float); 0.5·g·dt ≈ 2.7 uu/s, so 2.0 is
            // the first A/B for "full gravity-sticky on flat".
            // GGL_STICKY_FLAT: replace SCALE when |up.z| > 0.9 (floor only). Default 1.
            let mut sticky_scale = env_f32("GGL_STICKY_SCALE", 1.0);
            let sticky_flat = env_f32("GGL_STICKY_FLAT", 1.0);
            if (sticky_flat - 1.0).abs() > 1e-6 && upwards_dir.z.abs() > 0.9 {
                sticky_scale = sticky_flat;
            }
            rb.add_impulse(
                Some("StickyForce"),
                Impulse::Linear(
                    upwards_dir
                        * sticky_force_scale
                        * sticky_scale
                        * sticky_fade_scale
                        * const { GRAVITY_Z * TICK_TIME * UU_TO_BT },
                ),
                false,
                true,
            );
        }
    }

    fn apply_friction_curves(&mut self, rb: &RigidBody) {
        let real_throttle = if self.state.controls.boost && self.state.boost > 0.0 {
            1.0
        } else {
            self.state.controls.throttle
        };
        let car_pos = rb.get_world_pos();
        let car_vel = rb.lin_vel;
        let car_ang_vel = rb.ang_vel;
        for wheel in &mut self.bullet_vehicle.wheels {
            wheel.last_friction_curve_input = 0.0;
            let Some(raycast_info) = wheel.raycast_info.as_ref() else {
                continue;
            };
            if !raycast_info.is_in_contact_with_world {
                continue;
            }
            let lat_dir = wheel.axle_dir;
            let long_dir = lat_dir.cross(raycast_info.contact_normal);
            let wheel_delta = if env_on("GGL_FRIC_LEVER") {
                raycast_info.contact_point - car_pos
            } else {
                wheel.hard_point - car_pos
            };
            let cross_vec = (car_ang_vel.cross(wheel_delta) + car_vel) * BT_TO_UU;
            let base_friction = cross_vec.dot(lat_dir).abs();
            let friction_curve_input = if base_friction > 5.0 {
                base_friction / (cross_vec.dot(long_dir).abs() + base_friction)
            } else {
                0.0
            };
            wheel.last_friction_curve_input = friction_curve_input;
            let mut lat_friction = if self.info.config.three_wheels {
                curves::LAT_FRICTION_THREEWHEEL
            } else {
                curves::LAT_FRICTION
            }
            .get_output(friction_curve_input);
            let mut long_friction = 1.0;
            if self.state.handbrake_val != 0.0 {
                let handbrake_amount = self.state.handbrake_val;
                lat_friction *= 1.0
                    + (curves::HANDBRAKE_LAT_FRICTION_FACTOR.get_output(friction_curve_input)
                        - 1.0)
                        * handbrake_amount;
                long_friction *= 1.0
                    + (curves::HANDBRAKE_LONG_FRICTION_FACTOR.get_output(friction_curve_input)
                        - 1.0)
                        * handbrake_amount;
            }
            if real_throttle == 0.0 {
                let non_sticky_scale =
                    curves::NON_STICKY_FRICTION_FACTOR.get_output(raycast_info.contact_normal.z);
                lat_friction *= non_sticky_scale;
                long_friction *= non_sticky_scale;
            }
            wheel.lat_friction = lat_friction;
            wheel.long_friction = long_friction;
        }
    }

    fn update_air_torque(
        &mut self,
        rb: &mut RigidBody,
        update_air_control: bool,
        num_wheels_in_contact: usize,
    ) {
        let forward_dir = self.state.get_forward_dir();
        let right_dir = self.state.get_right_dir();
        let up_dir = self.state.get_up_dir();

        let dir_pitch = -right_dir;
        let dir_yaw = up_dir;
        let dir_roll = -forward_dir;

        if self.state.is_flipping {
            self.state.is_flipping =
                self.state.has_flipped
                    && self.state.flip_ticks < car_consts::flip::TORQUE_TICKS;
        }

        let mut do_air_control = false;
        if self.state.is_flipping {
            if self.state.flip_rel_torque == Vec3A::ZERO {
                do_air_control = true;
            } else {
                let mut rel_dodge_torque = self.state.flip_rel_torque;

                let mut pitch_scale = 1.0;
                if rel_dodge_torque.y != 0.0
                    && self.state.controls.pitch != 0.0
                    && rel_dodge_torque.y.signum() == self.state.controls.pitch.signum()
                    // The cancel only engages once the flip is PITCH_CANCEL_MIN_TIME
                    // old -- the first ~5 ticks always get full torque (measured; see
                    // the constant's comment). Without this gate a 1ts policy's brief
                    // into-flip pitch taps killed torque the real game applies.
                    && self.state.flip_ticks >= car_consts::flip::PITCH_CANCEL_MIN_TICKS
                {
                    pitch_scale = 1.0 - self.state.controls.pitch.abs().min(1.0);
                    do_air_control = true;
                }

                rel_dodge_torque.y *= pitch_scale;
                // GGL_DODGE_NEEDS_AIR=1: skip dodge torque once any wheel is down (PR72).
                let apply_dodge = update_air_control || !env_on("GGL_DODGE_NEEDS_AIR");
                let dodge_torque = rel_dodge_torque
                    * Vec3A::new(car_consts::flip::TORQUE_X, car_consts::flip::TORQUE_Y, 0.0)
                    * env_f32("GGL_FLIP_TORQUE", 1.0)
                    * TICK_TIME;

                // REVERTED to upstream (pulsar 2026-08-02). A vendor patch here divided
                // this torque by the inverse inertia tensor, reasoning that RL's angular
                // primitive (RocketLeague.exe FUN_140981560, mode 5) is Bullet's
                // applyTorqueImpulse. The Ghidra reading was right about what mode 5 does;
                // the mistake was assuming RocketSim's TORQUE_X/Y are in the same units as
                // the torque RL feeds that primitive. They are not.
                //
                // The replay validation that appeared to confirm it (angVel error
                // 2.048 -> 0.051 rad/s) was circular: it fed the sim the game's REPLICATED
                // DodgeTorque, which carries its own ~1/100 scale factor, so a torque that
                // was also divided by inertia matched a target that was already small.
                //
                // The scripted maneuver test (research/maneuvers) is decisive -- identical
                // inputs from identical states, dodge orientation error vs the real game:
                //     WITH the inertia division:  fwd 94.6 / 98.6 / 121.7 / 99.4 deg
                //     WITHOUT (this code):        fwd  6.0 /  4.4 /   5.1 /  2.5 deg
                // for dodge forward/backward/side/diagonal. See SIM2REAL_AUDIT.md S21.
                // Re-checked 2026-08-23 on the 120 Hz RLPR tape: inv_inertia * torque
                // raises flip+wheels vel p50/p90 and flip_air ang p50/p90. Keep this.
                // GGL_DODGE_NEEDS_AIR=1: drop dodge torque as soon as any wheel is down.
                if apply_dodge {
                    rb.add_impulse(
                        None,
                        Impulse::Angular(rb.get_world_trans().matrix3 * dodge_torque),
                        false,
                        true,
                    );
                }
                if env_on("GGL_DODGE_DAMP") {
                    let damp_pitch = dir_pitch.dot(rb.ang_vel) * car_consts::air_control::DAMPING.x;
                    let damp_yaw = dir_yaw.dot(rb.ang_vel) * car_consts::air_control::DAMPING.y;
                    let damp_roll = dir_roll.dot(rb.ang_vel) * car_consts::air_control::DAMPING.z;
                    let damping = dir_yaw * damp_yaw + dir_pitch * damp_pitch + dir_roll * damp_roll;
                    rb.add_impulse(
                        None,
                        Impulse::Angular(
                            damping * const { car_consts::air_control::TORQUE_APPLY_SCALE * TICK_TIME },
                        ),
                        false,
                        true,
                    );
                }
                if env_on("GGL_AXIS_SPIN") {
                    const CAP_X: f32 = 7.4396;
                    const CAP_Y: f32 = 7.2348;
                    let proj_x = rb.ang_vel.x + rb.accum_ang_vel.x;
                    if proj_x > CAP_X {
                        rb.accum_ang_vel.x -= proj_x - CAP_X;
                    } else if proj_x < -CAP_X {
                        rb.accum_ang_vel.x -= proj_x + CAP_X;
                    }
                    let proj_y = rb.ang_vel.y + rb.accum_ang_vel.y;
                    if proj_y > CAP_Y {
                        rb.accum_ang_vel.y -= proj_y - CAP_Y;
                    } else if proj_y < -CAP_Y {
                        rb.accum_ang_vel.y -= proj_y + CAP_Y;
                    }
                }
            }
        } else {
            do_air_control = true;
        }

        do_air_control &= !self.state.is_auto_flipping;
        do_air_control &= update_air_control;
        if do_air_control {
            let mut pitch_torque_scale = 1.0;
            let torque = if self.state.controls.pitch != 0.0
                || self.state.controls.yaw != 0.0
                || self.state.controls.roll != 0.0
            {
                if self.state.is_flipping
                    || self.state.has_flipped
                        && self.state.flip_ticks < car_consts::flip::PITCHLOCK_TICKS
                {
                    pitch_torque_scale = 0.0;
                }

                self.state.controls.pitch
                    * dir_pitch
                    * pitch_torque_scale
                    * car_consts::air_control::TORQUE.x
                    + self.state.controls.yaw * dir_yaw * car_consts::air_control::TORQUE.y
                    + self.state.controls.roll * dir_roll * car_consts::air_control::TORQUE.z
            } else {
                Vec3A::ZERO
            };

            let ang_vel = rb.ang_vel;

            let damp_pitch = dir_pitch.dot(ang_vel)
                * car_consts::air_control::DAMPING.x
                * (1.0 - (self.state.controls.pitch * pitch_torque_scale).abs());
            let damp_yaw = dir_yaw.dot(ang_vel)
                * car_consts::air_control::DAMPING.y
                * (1.0 - self.state.controls.yaw.abs());
            let damp_roll = dir_roll.dot(ang_vel) * car_consts::air_control::DAMPING.z;

            let damping = dir_yaw * damp_yaw + dir_pitch * damp_pitch + dir_roll * damp_roll;

            let rb_torque = (torque - damping)
                * const { car_consts::air_control::TORQUE_APPLY_SCALE * TICK_TIME };

            rb.add_impulse(None, Impulse::Angular(rb_torque), false, true);
        }

        // VENDOR PATCH (pulsar 2026-08-01): do not apply air throttle while boosting.
        // RocketSim added THROTTLE_AIR_ACCEL (66.67 uu/s^2) on top of boost whenever
        // throttle != 0, and `DefaultAction` sets throttle = boost on every aerial
        // action, so it applied on EVERY boosted aerial. Verified against real match
        // telemetry: predicted excess forward dV +4.444 uu/s per 8-tick window,
        // measured +4.459 (0.3%). Was upstream's "TODO: Fix air-throttle not
        // respecting boost". See research/reports/SIM2REAL_AUDIT.md S5.
        // VENDOR PATCH (pulsar 2026-08-24): no air throttle while ANY wheel is in
        // contact. T4 (solo autoroll/powerslide tape): every thr=+/-1 partial-
        // contact tick carried a forward bias of exactly +/-0.556 uu/s =
        // THROTTLE_AIR_ACCEL (200/3) * dt, uniform across speed 0-2300 (including
        // >1400 where ground drive is zero), tilt, and wheel count 1-2 -- the sim
        // stacked air throttle on top of wheel drive; the real game does not.
        // GGL_AIR_THROTTLE_WHEELS=1 restores the old stacking behavior.
        let throttle_scale = if num_wheels_in_contact > 0 && !env_on("GGL_AIR_THROTTLE_WHEELS") {
            0.0
        } else if env_on("GGL_BOOST_THROTTLE") && self.state.controls.boost {
            1.0
        } else if self.state.controls.boost {
            0.0
        } else {
            self.state.controls.throttle
        };
        if throttle_scale != 0.0 {
            let throttle_force = forward_dir
                * throttle_scale
                * const { car_consts::drive::THROTTLE_AIR_ACCEL * UU_TO_BT * TICK_TIME };
            rb.add_impulse(None, Impulse::Linear(throttle_force), false, true);
        }
    }

    fn update_jump(
        &mut self,
        rb: &mut RigidBody,
        mutator_config: &MutatorConfig,
        jump_pressed: bool,
    ) {
        let up_dir = self.state.get_up_dir();

        // Check jump activation
        if !self.state.has_jumped && self.state.is_on_ground && jump_pressed {
            self.state.is_jumping = true;
            self.state.has_jumped = true;
            self.state.jump_ticks = 0;
        }

        // Apply forces
        if self.state.is_jumping {
            // Jump started, apply initial boost force
            if self.state.jump_ticks == 0 {
                let jump_start_force = up_dir * mutator_config.jump_immediate_force * UU_TO_BT;
                rb.add_impulse(
                    Some("Jump"),
                    Impulse::Linear(jump_start_force),
                    false,
                    false,
                );
            }

            // Decide whether the jump is still HELD before applying the hold force
            // (SIM2REAL_AUDIT S45c). Applying first and deciding after fired one extra
            // hold impulse on the release tick -- a free ~12 uu/s that corrupted exactly
            // the jump->double-jump/dodge transition. Measured: battery fit 1027.8 ->
            // 1017.9, holdout 1057.3 -> 1048.2, moving 7 of 80 segments (all jump/dodge),
            // with double_jump -5.14 and dodge_diagonal -4.89 carrying it.
            let still_held = self.state.jump_ticks < car_consts::jump::MIN_TICKS
                || (self.state.controls.jump && self.state.jump_ticks < car_consts::jump::MAX_TICKS);
            if still_held {
                let jump_force =
                    up_dir * mutator_config.jump_accel * const { UU_TO_BT * TICK_TIME };
                rb.add_impulse(Some("Jump"), Impulse::Linear(jump_force), false, true);
            }

            self.state.jump_ticks += 1;
            self.state.is_jumping = self.state.jump_ticks < car_consts::jump::MIN_TICKS
                || (self.state.controls.jump && self.state.jump_ticks < car_consts::jump::MAX_TICKS);
        }

        // Update jump state
        if self.state.has_jumped {
            if !self.state.is_jumping {
                self.state.jump_ticks += 1;
            }

            // Possibly reset `has_jumped`
            if !env_on("GGL_JUMP_SETTLE")
                && self.state.is_on_ground
                && ticks_gt(
                    self.state.jump_ticks,
                    const { car_consts::jump::MIN_TIME + car_consts::jump::RESET_TIME_PAD },
                )
            {
                // Don't reset the jump just yet, we might still be leaving the ground
                // This fixes the bug where jump is reset before we actually leave the ground after a minimum-time jump
                // TODO: RL does something similar to this time-pad, but not exactly the same
                self.state.has_jumped = false;
                self.state.is_jumping = false;
                self.state.jump_ticks = 0;
            }
        }
    }

    fn update_auto_flip(&mut self, rb: &mut RigidBody, jump_pressed: bool) {
        if jump_pressed
            && self
                .state
                .world_contact_normal
                .is_some_and(|world_contact_normal| {
                    world_contact_normal.z > car_consts::autoflip::NORM_Z_THRESH
                })
        {
            let (_, _, roll) = self.state.phys.rot_mat.to_euler(EulerRot::ZYX);
            let abs_roll = roll.abs();
            if abs_roll > car_consts::autoflip::ROLL_THRESH {
                self.state.auto_flip_ticks =
                    secs_to_ticks(car_consts::autoflip::TIME * (abs_roll / PI));
                self.state.auto_flip_torque_scale = roll.signum();
                self.state.is_auto_flipping = true;

                let force =
                    -self.state.get_up_dir() * const { car_consts::autoflip::IMPULSE * UU_TO_BT };
                rb.add_impulse(None, Impulse::Linear(force), false, false);
            }
        }

        if self.state.is_auto_flipping {
            if self.state.auto_flip_ticks == 0 {
                self.state.is_auto_flipping = false;
            } else {
                rb.ang_vel += self.state.get_forward_dir()
                    * car_consts::autoflip::TORQUE
                    * self.state.auto_flip_torque_scale
                    * TICK_TIME;
                self.state.auto_flip_ticks -= 1;
            }
        }
    }

    fn update_double_jump_or_flip(
        &mut self,
        rb: &mut RigidBody,
        mutator_config: &MutatorConfig,
        jump_pressed: bool,
        forward_speed_uu: f32,
    ) {
        // RL-PARITY DODGE GATE v3 (2026-08-25). Per-tick real-game traces
        // (research/tools/wavedash_trace.py + earlypress_probe.py on
        // pulsar-gco-ts1-bot/debug.3914029) show the game's wheel contact has THREE
        // decoupled effects that the legacy single ground early-return bundled:
        //
        //   1. contact EATS dodge presses - always, jump state does NOT override
        //      (fire-tick alignment: a during-jump press escape fired on
        //      masked-ground presses the real game ate; the clean-trace mid-Jumping
        //      dodge fired because its contact had already broken). Legacy already
        //      had this right via the contact gate.
        //   2. contact does NOT reset the dodge window or flip state during takeoff:
        //      dodge_timeout kept counting straight through an AirState
        //      Jumping->OnGround->InAir flicker. State resets only with the landing
        //      reset that clears has_jumped (update_jump). Legacy reset both on
        //      every contact tick.
        //   3. contact cancels active flip torque only on the way DOWN (that cancel
        //      is the wavedash mechanic itself); a takeoff dodge keeps its dodge
        //      state through ascent contact flicker. Legacy cancelled on any contact.
        //
        // History: gate v1 (reverted) escaped press-eating during is_jumping AND
        // re-based the window on jump start - both wrong, sim wavedash fell 23->6%.
        // v2 rev B kept the is_jumping press escape - fire agreement 18/29 vs
        // legacy 21/29. v3 keeps legacy's press gating (contact only) and fixes
        // only #2/#3. The residual real-vs-sim gap is contact-model persistence
        // (sim wheel contact outlives RL's by 1-3 ticks during takeoff), not gating.
        // GGL_LEGACY_DODGE_GATE=1 restores the full legacy bundle.
        let legacy_gate = ggl_legacy_dodge_gate();
        // Effect #2: an unsettled jump keeps the machinery live through contact.
        // has_jumped clears via update_jump's landing reset (runs earlier this tick),
        // so a true landing still resets flip state on the landing tick itself.
        // FLIP-RESET HOLE (2026-08-26, v5). The takeoff guard above was `has_jumped`
        // alone, on the stated assumption that "has_jumped clears via update_jump's
        // landing reset, so a true landing still resets flip state on the landing tick".
        // That assumption fails: update_jump only clears has_jumped once
        // jump_ticks > MIN_TIME + RESET_TIME_PAD, so after a SHORT jump the car lands
        // with has_jumped still set, the guard suppresses the flip reset, and
        // has_flipped is never cleared - the next press finds no dodge available.
        // Singles rarely notice; a CHAIN is exactly a rapid land -> re-jump -> dodge
        // cycle, so it hits at every link. Measured on 1635 unbiased real press edges:
        // gate agreement 88.1%, errors 2:1 asymmetric toward "sim eats a press the real
        // game converts", and 71.5% of those misses have the sim holding has_flipped
        // while the real game grants the dodge. At 88% per-press, a 5-link chain
        // survives only 68% of the time - which is the measured 32% link-2 chain
        // mismatch, and the user-visible "chained wavedashes misfire in game".
        // Restrict the guard to an ACTUAL takeoff (the jump is still live, or we have
        // not left the ground yet), which keeps gate v3's Jumping->OnGround->InAir
        // flicker fix while letting a genuine landing restore the flip.
        let takeoff = !legacy_gate
            && self.state.has_jumped
            && (!ggl_flip_reset_fix() || self.state.is_jumping || self.state.air_ticks == 0);
        if self.state.is_on_ground && !takeoff {
            self.state.has_double_jumped = false;
            self.state.has_flipped = false;
            self.state.air_ticks = 0;
            self.state.air_ticks_since_jump = 0;
            self.state.flip_ticks = 0;
            return;
        }

        self.state.air_ticks += 1;

        // Window counts from jump-state END in both modes (matches the packet's
        // dodge_timeout: at the first tick after a flicker it read 1.25 minus the
        // time since jump end, not since takeoff).
        let counts = self.state.has_jumped && !self.state.is_jumping;
        if counts {
            self.state.air_ticks_since_jump += 1;
        } else {
            self.state.air_ticks_since_jump = 0;
        }
        // Effect #1: presses are eaten by wheel contact, ALWAYS - the jump state does
        // NOT override. Fire-tick alignment vs real (29 clean events): a press escape
        // during is_jumping fired on masked-ground presses the real game ate (v2 rev B
        // agreement 18/29 vs legacy's 21/29). The real gate is the contact query at
        // the press tick, full stop; the clean-trace dodge that fired mid-Jumping did
        // so because ITS contact had already broken (z=26.8 at the press).
        //
        // BUT it is a SHORTER contact query than is_on_ground (v4, 2026-08-26):
        // 661 real press edges give fire-rate 0% below z~21, 50% at z~27.5, ~100%
        // from z~31 (level Octane, rest 17), while is_on_ground's ray reach kept
        // eating presses to z~31-33. The policy had learned those presses are FREE
        // in sim; in the real game each one fires a dodge 1-3 ticks early - full
        // flips instead of wavedashes, the back lifting on forward wavedashes.
        // Gate the press on per-wheel rays with a short reach past rest length
        // (GGL_DODGE_CONTACT_EXT uu, fit against the real fire curve; <0 restores
        // the is_on_ground gate).
        // WHEEL-COUNT GATE (2026-08-26, v5). is_on_ground is `>= 3 wheels in contact`, but
        // Ghidra shows RL keeps THREE separate contact queries (IsOnGround /
        // GetNumWheelContacts / GetNumWheelWorldContacts), so the press gate need not be
        // the is_on_ground one. Measured on the real FAILURE population - 182
        // availability-filtered directional presses where the sim read airborne while the
        // packet still read OnGround - the real game fires a dodge on 18.1% of them while
        // this engine fires on 91.8% (per-event agreement 23.1%). The engine frees the
        // dodge the moment the car drops below THREE wheels, which during takeoff is
        // several ticks before the last wheel leaves; the policy learned those presses
        // dodge, so in game they return as plain jumps ("jumping instead of flipping") or
        // fire early enough to become a full flip instead of a wavedash. Requiring FEWER
        // wheels keeps the press eaten longer, which is the real behaviour.
        // GGL_DODGE_CONTACT_WHEELS: 3 = legacy is_on_ground semantics; 1 = any wheel.
        let dodge_contact = {
            let ext_uu = ggl_dodge_contact_ext_uu();
            let min_wheels = ggl_dodge_contact_wheels();
            if legacy_gate {
                self.state.is_on_ground
            } else if self.ticks_since_ground <= ggl_dodge_ground_hold_ticks()
                && self.state.phys.rot_mat.z_axis.z >= ggl_dodge_hold_min_upz()
            {
                // CONTACT HOLD (2026-08-26, v6): eat the press for one tick after this
                // engine's contact breaks, LEVEL CARS ONLY. Fit on the free-running
                // 120 Hz instruments (restore-based replay distorts this knob):
                //   - speed_flip scripted tape: real eats the level-takeoff press this
                //     engine converts; a 1-tick hold takes the divergence 1127 -> 24 uu
                //     while leaving all 82 other battery segments byte-identical.
                //     hold=2 breaks half_flip (18 -> 538), so N=1 is sharply identified.
                //   - the attitude condition: on TILTED match-play poses the real game
                //     frees the dodge exactly where our contact breaks (the 92.5%
                //     z-curve fit), and an unconditional hold costs 2.2pp per-event
                //     agreement there (88.2 -> 86.0 at PRE=12). Level fast takeoffs are
                //     where the real gate outlives ours; tilt is where it does not.
                // Chained wavedashes are strings of level fast takeoffs - this is the
                // "chains misfire in game" fix. GGL_DODGE_GROUND_HOLD=0 disables.
                true
            } else if min_wheels < 3 {
                let n = self
                    .bullet_vehicle
                    .wheels
                    .iter()
                    .filter(|w| w.raycast_info.is_some())
                    .count() as u32;
                n >= min_wheels
            } else if ext_uu < 0.0 {
                self.state.is_on_ground
            } else {
                let ext_bt = ext_uu * UU_TO_BT;
                self.bullet_vehicle.wheels.iter().any(|w| {
                    w.raycast_info
                        .as_ref()
                        .is_some_and(|r| r.suspension_length < w.suspension_rest_length_1 + ext_bt)
                })
            }
        };
        // GATE v4 (2026-08-26, scripted-chain tape): DURING the jump state a press
        // fires from GGL_DODGE_JUMP_MIN_TICKS ticks after activation regardless of
        // wheel contact. The deterministic chain replays (real_maneuvers.tsv, 8 real
        // turning-wavedash chains re-executed in-game) show the real car airborne and
        // dodging 2-3 ticks after the jump press on FLAT ground while sim contact
        // persisted 4 more ticks and ate the dodge - every chain link degenerated to
        // a plain jump (the user-visible "chained wavedashes don't work in game";
        // flat level takeoff is exactly where match-play poses are tilted enough that
        // the earlier 91.8%-agreement sweep never sampled the regime). The bare
        // during-jump escape (v2 rev B, effectively K=1) overfired: real EATS presses
        // 1 tick after activation (fire alignment 18/29 vs legacy 21/29). K>=2
        // reconciles the chain tape, the mid-jump clean-trace fire at z=26.8, and the
        // eaten +1-tick mash presses. Post-jump-state grounded presses stay eaten.
        let k_min = {
            static V: OnceLock<u32> = OnceLock::new();
            *V.get_or_init(|| {
                env::var("GGL_DODGE_JUMP_MIN_TICKS")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0)
            })
        };
        // k_min == 0 DISABLES the during-jump press escape (measured verdict: any
        // small K overfires - K=2 fired 57% of z 17-21 during-jump grounded presses
        // the real game eats; 625-edge agreement 87.7% vs 90.0% disabled. The knob
        // stays for refits against future scripted tapes.
        let in_jump_past_k = !legacy_gate
            && k_min > 0
            && self.state.is_jumping
            && self.state.jump_ticks > k_min;
        let press_air_ok = legacy_gate || in_jump_past_k || !dodge_contact;

        // The post-jump lockout only means anything if a jump actually happened:
        // air_ticks_since_jump is pinned at 0 for a car that never jumped, so gating on it
        // unconditionally re-imposes the has_jumped block that the capture already
        // disproved (corner_flip_into shows the real car taking a +280.8 uu/s jump impulse
        // in mid-air having never grounded). Measured: without this guard stall regresses
        // 9.3 -> 240.6 uu and corner_flip_into 11.1 -> 106.5 uu.
        // RL gate v2 addition: a rising-edge press DURING the jump state dodges
        // immediately (real: press 17ms after a ground jump -> AirState Dodging one
        // packet-tick later, mid-Jumping). The edge itself guarantees >=1 tick since
        // jump activation, so the FLIP_MIN_DELAY intent is preserved; atsj is pinned
        // 0 while is_jumping, hence the explicit is_jumping escape.
        // A press during the jump state with contact already broken fires in the real
        // game (clean trace: press 2 ticks after a ground jump at z=26.8, AirState
        // Dodging one packet-tick later). atsj is pinned 0 while is_jumping, so that
        // case needs an explicit escape; press_air_ok above keeps it airborne-only
        // (grounded during-jump presses stay eaten, which the mash traces demand).
        let flip_delay_ok = !self.state.has_jumped
            || (!legacy_gate && self.state.is_jumping)
            || in_jump_past_k
            || self.state.air_ticks_since_jump >= car_consts::jump::FLIP_MIN_DELAY_TICKS;
        if jump_pressed
            && press_air_ok
            && flip_delay_ok
            && self.state.air_ticks_since_jump < car_consts::jump::DOUBLEJUMP_MAX_TICKS
        {
            let input_magnitude = self.state.controls.yaw.abs()
                + self.state.controls.pitch.abs()
                + self.state.controls.roll.abs();
            let is_flip_input = input_magnitude >= self.config.dodge_deadzone;

            // NOT PATCHED. It is tempting to add `self.state.has_jumped &&` here: the
            // gate checks has_double_jumped and has_flipped but not has_jumped, and
            // air_time_since_jump is reset to 0 every tick while has_jumped is false, so
            // the DOUBLEJUMP_MAX_DELAY window is vacuous. Adding it made `stall` go
            // 249.9 -> 8.1 uu against the real game and looked like a clear win.
            //
            // It was measurement error. The real game CARRIES jump/flip state across a
            // state set; the maneuver runner reset the car every segment. `stall` follows
            // `speed_flip`, which ends mid-flip, so the real car had already spent its
            // flip -- nothing to do with has_jumped. The paired segment `corner_flip_into`
            // (which follows a segment ending on the ground) shows the real car taking a
            // +280.8 uu/s jump impulse while airborne and never grounded, i.e. RL DOES
            // allow it there, and the patch regressed that segment 20.3 -> 103.5 uu.
            //
            // Decide this only from segments whose jump state is controlled. See
            // SIM2REAL_AUDIT.md S23.
            let can_use = !self.state.is_auto_flipping
                && !self.state.has_double_jumped
                && !self.state.has_flipped
                || if is_flip_input {
                    mutator_config.unlimited_flips
                } else {
                    mutator_config.unlimited_double_jumps
                };

            if can_use {
                if is_flip_input {
                    self.state.flip_ticks = 0;
                    self.state.has_flipped = true;
                    self.state.is_flipping = true;

                    let forward_speed_ratio = forward_speed_uu.abs() / car_consts::MAX_SPEED;
                    let mut dodge_dir = Vec3A::new(
                        -self.state.controls.pitch,
                        self.state.controls.yaw + self.state.controls.roll,
                        0.0,
                    );

                    if dodge_dir.x.abs() < 0.1 && dodge_dir.y.abs() < 0.1 {
                        dodge_dir = Vec3A::ZERO;
                    } else {
                        dodge_dir = dodge_dir.normalize_or_zero();
                    }

                    self.state.flip_rel_torque = Vec3A::new(-dodge_dir.y, dodge_dir.x, 0.0);

                    // v3-tuned `RS_TUNE_DODGE_TORQUE_AT_START`: `update_air_torque`
                    // already ran with `is_flipping` still false, so the first tick
                    // of dodge torque is otherwise lost. Shipped on titan-appo 2026-08-25
                    // (T1/211647 flip_air tie; air+wheels ang p90 drop).
                    //
                    // DEFAULT OFF for us since 2026-08-26 (`GGL_DODGE_TORQUE_START=1`
                    // restores it). Ported ON, then bisected out: with it ON the 528B
                    // policy scores 10.7% vs BonkDaddy, OFF it scores 30.8% (n=150-200)
                    // - a 3x swing from this one change, isolated by A/B while every
                    // other ported default was ruled out (contact hold 8.2%, touchdown
                    // exc=5 7.5%). Accuracy against the HELD-OUT real tape does not pay
                    // for that: medians tie (p90 11.54 ON vs 11.80 OFF), OFF wins 8
                    // segments to 4, and OFF has 17% fewer ground-flag mismatches
                    // (345 vs 418). It genuinely helps half_flip / diag_flip /
                    // land_during_flip_rotation and genuinely hurts flip_into /
                    // flip_early / flip_into_wall / backwall_flip - net neutral on
                    // position, worse on contact, catastrophic on play. Revisit only
                    // with an angular-rate measurement against real captures, which is
                    // the evidence titan fit it on and which we have never reproduced.
                    if env::var("GGL_DODGE_TORQUE_START").is_ok_and(|s| s != "0") {
                        let mut rel_dodge_torque = self.state.flip_rel_torque;
                        let mut pitch_scale = 1.0;
                        if rel_dodge_torque.y != 0.0
                            && self.state.controls.pitch != 0.0
                            && rel_dodge_torque.y.signum() == self.state.controls.pitch.signum()
                            && self.state.flip_ticks >= car_consts::flip::PITCH_CANCEL_MIN_TICKS
                        {
                            pitch_scale = 1.0 - self.state.controls.pitch.abs().min(1.0);
                        }
                        rel_dodge_torque.y *= pitch_scale;
                        let dodge_torque = rel_dodge_torque
                            * Vec3A::new(car_consts::flip::TORQUE_X, car_consts::flip::TORQUE_Y, 0.0)
                            * env_f32("GGL_FLIP_TORQUE", 1.0)
                            * TICK_TIME;
                        rb.add_impulse(
                            None,
                            Impulse::Angular(rb.get_world_trans().matrix3 * dodge_torque),
                            false,
                            true,
                        );
                    }

                    if dodge_dir.x.abs() < 0.1 {
                        dodge_dir.x = 0.0;
                    }

                    if dodge_dir.y.abs() < 0.1 {
                        dodge_dir.y = 0.0;
                    }

                    if dodge_dir.length_squared() > const { f32::EPSILON * f32::EPSILON } {
                        let should_dodge_backwards = if forward_speed_uu.abs() < 100. {
                            dodge_dir.x.is_sign_negative()
                        } else {
                            dodge_dir.x.signum() != forward_speed_uu.signum()
                        };

                        let max_speed_scale_x = if should_dodge_backwards {
                            car_consts::flip::BACKWARD_IMPULSE_MAX_SPEED_SCALE
                        } else {
                            car_consts::flip::FORWARD_IMPULSE_MAX_SPEED_SCALE
                        };

                        let mut initial_dodge_vel = dodge_dir
                            * car_consts::flip::INITIAL_VEL_SCALE
                            * env_f32("GGL_DODGE_VEL", 1.0);
                        initial_dodge_vel.x *=
                            ((max_speed_scale_x - 1.) * forward_speed_ratio) + 1.0;
                        initial_dodge_vel.y *= ((car_consts::flip::SIDE_IMPULSE_MAX_SPEED_SCALE
                            - 1.)
                            * forward_speed_ratio)
                            + 1.0;
                        if should_dodge_backwards {
                            initial_dodge_vel.x *= car_consts::flip::BACKWARD_IMPULSE_SCALE_X;
                        }

                        let forward_dir_2d =
                            self.state.get_forward_dir().with_z(0.0).normalize_or_zero();
                        let right_dir_2d = Vec3A::new(-forward_dir_2d.y, forward_dir_2d.x, 0.0);
                        let final_delta_vel = initial_dodge_vel.x * forward_dir_2d
                            + initial_dodge_vel.y * right_dir_2d;

                        rb.add_impulse(
                            None,
                            Impulse::Linear(final_delta_vel * UU_TO_BT),
                            false,
                            false,
                        );
                    }
                } else {
                    let jump_start_force =
                        self.state.get_up_dir() * mutator_config.jump_immediate_force * UU_TO_BT;
                    rb.add_impulse(
                        Some("double_jump"),
                        Impulse::Linear(jump_start_force),
                        false,
                        false,
                    );
                    self.state.has_double_jumped = true;
                }
            }
        }

        if self.state.is_flipping {
            // Gate Z-damp on pre-increment flip_ticks (Zealan PR73 / v3-tuned
            // RS_TUNE_FLIP_ZDAMP_PRE_INCREMENT). Post-increment fires one tick early.
            if !env_on("GGL_NO_FLIP_ZDAMP")
                && self.state.flip_ticks < car_consts::flip::TORQUE_TICKS
                && self.state.flip_ticks >= car_consts::flip::Z_DAMP_START_TICKS
                && (rb.lin_vel.z < 0.0
                    || self.state.flip_ticks < car_consts::flip::Z_DAMP_END_TICKS)
            {
                rb.lin_vel.z *= 1.0 - car_consts::flip::Z_DAMP_120;
            }
            self.state.flip_ticks += 1;
        } else if self.state.has_flipped {
            self.state.flip_ticks += 1;
        }
    }

    fn update_auto_roll(&self, rb: &mut RigidBody, num_wheels_in_contact: usize) {
        let ground_up_dir = if num_wheels_in_contact > 0 {
            self.bullet_vehicle.get_upwards_dir_from_wheel_contacts(rb)
        } else {
            self.state.world_contact_normal.unwrap()
        };

        let ground_down_dir = -ground_up_dir;

        let forward_dir = self.state.get_forward_dir();
        let right_dir = self.state.get_right_dir();

        let cross_right_dir = ground_up_dir.cross(forward_dir);
        let cross_forward_dir = ground_down_dir.cross(cross_right_dir);

        let right_torque_factor = 1.0 - right_dir.dot(cross_right_dir).clamp(0.0, 1.0);
        let forward_torque_factor = 1.0 - forward_dir.dot(cross_forward_dir).clamp(0.0, 1.0);

        let torque_dir_right = forward_dir * -right_dir.dot(ground_up_dir).signum();
        let torque_dir_forward = right_dir * forward_dir.dot(ground_up_dir).signum();

        let torque_right = torque_dir_right * right_torque_factor;
        let torque_forward = torque_dir_forward * forward_torque_factor;

        rb.add_impulse(
            None,
            Impulse::Linear(
                ground_down_dir * const { car_consts::autoroll::FORCE * UU_TO_BT * TICK_TIME },
            ),
            false,
            true,
        );

        rb.add_impulse(
            None,
            Impulse::Angular(
                (torque_forward + torque_right)
                    * const { car_consts::autoroll::TORQUE * TICK_TIME },
            ),
            false,
            true,
        );
    }

    fn update_boost(&mut self, rb: &mut RigidBody, mutator_config: &MutatorConfig) {
        self.state.is_boosting = if self.state.boost > 0.0 {
            self.state.controls.boost
                || (self.state.is_boosting
                    && self.state.boosting_ticks < car_consts::boost::MIN_TICKS)
        } else {
            false
        };

        if self.state.is_boosting {
            self.state.boosting_ticks += 1;
            self.state.ticks_since_boosted = 0;
            self.state.boost -= mutator_config.boost_used_per_second * TICK_TIME;

            // GGL_BOOST_AIR_WHEELS: DEFAULT OFF since 2026-08-24. Ground boost
            // accel whenever any wheel is down. is_on_ground is n>=3, so 1-2-wheel
            // ticks were taking ACCEL_AIR (3175/3) instead of ACCEL_GROUND (2975/3).
            // Difference = THROTTLE_AIR_ACCEL (200/3) = +0.556 uu/s/tick. Dual of
            // the wheel-gated air-throttle patch. =1 restores air accel on
            // partial contact.
            let accel = if self.state.is_on_ground
                || (self.state.num_wheels_in_contact() > 0 && !env_on("GGL_BOOST_AIR_WHEELS"))
            {
                mutator_config.boost_accel_ground
            } else {
                mutator_config.boost_accel_air
            } * if self.state.is_flipping {
                env_f32("GGL_BOOST_FLIP_SCALE", 1.0)
            } else {
                1.0
            };

            rb.add_impulse(
                Some("Boost"),
                Impulse::Linear(accel * self.state.get_forward_dir() * (UU_TO_BT * TICK_TIME)),
                false,
                true,
            );
        } else {
            self.state.boosting_ticks = 0;
            self.state.ticks_since_boosted += 1;

            if mutator_config.recharge_boost_enabled
                && self.state.ticks_since_boosted
                    >= ticks_until(mutator_config.recharge_boost_delay)
            {
                self.state.boost += mutator_config.recharge_boost_per_second * TICK_TIME;
            }
        }

        self.state.boost = self.state.boost.clamp(0.0, car_consts::boost::MAX);
    }

    pub(crate) fn pre_tick_update(
        &mut self,
        collision_world: &mut DiscreteDynamicsWorld,
        rng: &mut Rng,
        game_mode: GameMode,
        mutator_config: &MutatorConfig,
    ) {
        debug_assert!(
            self.bullet_vehicle.get_num_wheels() == 4 || self.bullet_vehicle.get_num_wheels() == 3
        );

        // Capture dodge before `is_on_ground` wipes `is_flipping` (sim ≥3 wheels,
        // including a graze the real dodge still rides). Used by autoroll "land".
        let flipping_at_entry = self.state.is_flipping;

        {
            let rb = &mut collision_world.bodies_mut()[self.rigid_body_idx];
            if self.state.is_demoed {
                self.state.demo_respawn_ticks = self.state.demo_respawn_ticks.saturating_sub(1);
                if self.state.demo_respawn_ticks == 0 {
                    self.respawn(rb, rng, game_mode, mutator_config.car_spawn_boost_amount);
                }

                rb.set_activation_state(ActivationState::DisableSimulation);
                rb.collision_flags |= CollisionFlags::NoContactResponse as u8;
                return;
            }

            rb.force_activate();
            rb.collision_flags &= !(CollisionFlags::NoContactResponse as u8);
            self.state.controls = self.state.controls.clamp();
        }

        let forward_speed_uu =
            collision_world.bodies()[self.rigid_body_idx].get_forward_speed() * BT_TO_UU;

        let jump_pressed = self.state.controls.jump && !self.state.prev_controls.jump;

        {
            let rb = &mut collision_world.bodies_mut()[self.rigid_body_idx];

            // TODO: Refactor and move
            let num_wheels_in_contact = self.state.num_wheels_in_contact();
            rb.wheels_grounded = num_wheels_in_contact >= 3;
            rb.chassis_scrape_ok = num_wheels_in_contact == 0
                && self.state.is_flipping
                && self.chassis_scrape_prev;

            self.update_wheels(rb, num_wheels_in_contact, forward_speed_uu);

            // RL gate v2 effect #3: contact cancels a flip only on the way DOWN (the
            // wavedash flatten). During a jump's ascent the wheels still register
            // contact for a few ticks, but the real game keeps the dodge alive
            // (trace: instant takeoff dodge held AirState Dodging for 300+ms through
            // contact heights). Direction, not contact, is the landing discriminator.
            // NOTE: only the CANCEL is skipped; air control/throttle stay gated by
            // is_on_ground exactly as legacy. Routing ascent-contact ticks into
            // update_air_torque applied air control + air throttle on the 1-2 sticky
            // ticks of EVERY jump and doubled the early-press probe's t+5 error
            // (2.4u -> 4.9u) - the real game, like legacy, applies none there.
            let takeoff_rise = !ggl_legacy_dodge_gate()
                && self.state.has_jumped
                && rb.lin_vel.dot(self.state.get_up_dir()) > 0.0;
            if self.state.is_on_ground {
                if !takeoff_rise {
                    self.state.is_flipping = false;
                }
            } else if !env_on("GGL_FLIP_THEN_AIR") {
                self.update_air_torque(
                    rb,
                    num_wheels_in_contact == 0 || env_on("GGL_AIR_WITH_WHEELS"),
                    num_wheels_in_contact,
                );
            }

            self.update_jump(rb, mutator_config, jump_pressed);
            self.update_auto_flip(rb, jump_pressed);
            self.update_double_jump_or_flip(rb, mutator_config, jump_pressed, forward_speed_uu);

            if !self.state.is_on_ground && env_on("GGL_FLIP_THEN_AIR") {
                self.update_air_torque(
                    rb,
                    num_wheels_in_contact == 0 || env_on("GGL_AIR_WITH_WHEELS"),
                    num_wheels_in_contact,
                );
            }

            // GGL_AUTOROLL: "1" = v2 always-on, "0"/"off" = never, "noflip" = v2
            // gate minus jump/flip ticks, "strict" = noflip and wheels-only (no
            // chassis world-contact arm). Default "land" (5) = noflip plus skip
            // assist on ticks that *entered* mid-dodge even if sim wheels cleared
            // `is_flipping` this tick (lv-flipwheels 2026-08-26: T1 fw p90
            // 1.055→0.922, other regimes bit-identical). `GGL_AUTOROLL=noflip`
            // restores the previous default.
            let autoroll_mode = {
                static V: OnceLock<u8> = OnceLock::new();
                *V.get_or_init(|| match env::var("GGL_AUTOROLL").as_deref() {
                    Ok("1") => 1,
                    Ok("0") | Ok("off") => 0,
                    Ok("strict") => 3,
                    Ok("nofliplanded") => 4,
                    Ok("noflip") => 2,
                    Ok("land") => 5,
                    _ => 5, // land default
                })
            };
            let autoroll_ok = match autoroll_mode {
                0 => false,
                1 => true,
                _ => {
                    !self.state.is_flipping
                        && !self.state.is_jumping
                        && (autoroll_mode != 4 || !self.state.has_flipped)
                        && (autoroll_mode != 5 || !flipping_at_entry)
                        && (autoroll_mode == 2 || autoroll_mode == 5 || num_wheels_in_contact > 0)
                }
            };
            if autoroll_ok
                && self.state.controls.throttle != 0.0
                && ((0 < num_wheels_in_contact && num_wheels_in_contact < 4)
                    || (autoroll_mode != 3 && self.state.world_contact_normal.is_some()))
            {
                self.update_auto_roll(rb, num_wheels_in_contact);
            }

            self.state.world_contact_normal = None;

            if !ggl_wheels_pre() {
                self.update_boost(rb, mutator_config);
            }
        }

        if ggl_wheels_pre() {
            // TOUCHDOWN EXCEPTION (2026-08-25, measured on two real-game full-pose
            // captures): the pre-step wheel pass is exact for wheels ALREADY in contact
            // (steady ground |vz| residual 0.01 uu/s) but over-damps the first contact
            // ticks of a landing: sim under-rebounds by -13.5 uu/s vertical at t+2 vs
            // the real game, while the post-step path lands at -1.6 (and conversely
            // costs 2-9 uu/s continuously when used for steady rolling). So gate
            // per tick: any wheel newly in contact this tick (touchdown) -> post-step
            // path for this tick; all persisting contacts -> pre-step path.
            // GGL_NO_TOUCHDOWN_EXC=1 restores unconditional pre-step.
            let newly_touching = !ggl_no_touchdown_exc()
                && self.bullet_vehicle.wheels.iter().any(|w| {
                    w.raycast_info.is_some() && !w.was_in_contact_prev_tick
                });
            for w in self.bullet_vehicle.wheels.iter_mut() {
                w.was_in_contact_prev_tick = w.raycast_info.is_some();
            }
            if newly_touching {
                self.touchdown_post_ticks = ggl_touchdown_exc_ticks();
            }
            if self.touchdown_post_ticks > 0 {
                self.touchdown_post_ticks -= 1;
                // post-step semantics for the touchdown tick: boost now, wheels after
                // the Bullet step (post_tick path invoked by the caller flag below).
                let rb = &mut collision_world.bodies_mut()[self.rigid_body_idx];
                self.update_boost(rb, mutator_config);
                self.wheels_post_this_tick = true;
            } else {
                // Tuned order: suspension/friction at this pose, then Bullet step.
                self.bullet_vehicle
                    .update_vehicle_second(collision_world, TICK_TIME);
                let rb = &mut collision_world.bodies_mut()[self.rigid_body_idx];
                self.update_boost(rb, mutator_config);
                self.wheels_post_this_tick = false;
            }
        }
    }

    pub(crate) fn post_tick_update(&mut self, collision_world: &mut DiscreteDynamicsWorld) {
        const START_SPEED_SQ: f32 =
            car_consts::supersonic::START_SPEED * car_consts::supersonic::START_SPEED;
        const MAINTAIN_MIN_SPEED_SQ: f32 =
            car_consts::supersonic::MAINTAIN_MIN_SPEED * car_consts::supersonic::MAINTAIN_MIN_SPEED;

        let rb = &collision_world.bodies()[self.rigid_body_idx];
        if self.state.is_demoed {
            return;
        }

        // Snapshot the ground flag BEFORE this tick's contact refresh below. The dodge
        // gate runs in pre_tick_update and therefore reads is_on_ground as computed at
        // the end of the PREVIOUS tick; saving the pre-refresh value here leaves
        // the counter exactly one tick behind it, which is the pair the hold needs.
        self.ticks_since_ground = if self.state.is_on_ground {
            0
        } else {
            self.ticks_since_ground.saturating_add(1)
        };

        self.state.phys.rot_mat = rb.get_world_trans().matrix3;

        let speed_squared = (rb.lin_vel * BT_TO_UU).length_squared();
        if self.state.is_supersonic {
            if speed_squared >= START_SPEED_SQ {
                // Back above start speed: reset the maintain timer.
                self.state.supersonic_grace_ticks = 0;
            } else if speed_squared < MAINTAIN_MIN_SPEED_SQ {
                // Dropped below the minimum maintain speed: lose supersonic immediately.
                self.state.is_supersonic = false;
                self.state.supersonic_grace_ticks = 0;
            } else {
                // Between maintain min and start speed: keep supersonic for the grace period.
                self.state.supersonic_grace_ticks += 1;
                if self.state.supersonic_grace_ticks >= car_consts::supersonic::MAINTAIN_MAX_TICKS
                {
                    self.state.is_supersonic = false;
                    self.state.supersonic_grace_ticks = 0;
                }
            }
        } else if speed_squared >= START_SPEED_SQ {
            // REVERTED S32's grounded-start gate (SIM2REAL_AUDIT.md S40): the 120 Hz
            // capture's one real demolition has the attacker crossing 2200 while
            // AIRBORNE (z=72, boost surge 2001 -> 2300 on the contact tick) and the
            // real game demolishes -- a direct counter-example to "supersonic can only
            // start while grounded", which had been adopted on external evidence and
            // was flagged unvalidated. The CDO's SuperSonicSettings carries no ground
            // condition either. Whatever removed Moonwatcher's phantom demos, it was
            // not this rule; the hit-angle cones + forward-speed gate (S36) are the
            // real phantom filters.
            self.state.is_supersonic = true;
            self.state.supersonic_grace_ticks = 0;
        } else {
            self.state.supersonic_grace_ticks = 0;
        }

        self.bullet_vehicle
            .update_vehicle_first(collision_world, TICK_TIME);
        if env_on("GGL_FRIC_APPLY") {
            let chassis = &collision_world.bodies()[self.rigid_body_idx];
            self.apply_friction_curves(chassis);
        }
        if !ggl_wheels_pre() || self.wheels_post_this_tick {
            self.bullet_vehicle
                .update_vehicle_second(collision_world, TICK_TIME);
            self.wheels_post_this_tick = false;
        }
        let mut num_wheels_in_contact = 0u8;
        let travel_bt = vehicle_consts::MAX_SUSPENSION_TRAVEL * UU_TO_BT;
        for (i, wheel) in self.bullet_vehicle.wheels.iter().enumerate() {
            let in_contact = wheel.raycast_info.is_some();
            self.state.wheels_with_contact[i] = in_contact;
            num_wheels_in_contact += u8::from(in_contact);

            let rest = wheel.suspension_rest_length_1;
            let len = wheel
                .raycast_info
                .as_ref()
                .map(|r| r.suspension_length)
                .unwrap_or(rest + travel_bt);
            let denom = 2.0 * travel_bt;
            self.state.wheels_suspension[i] = if denom > 0.0 {
                ((len - (rest - travel_bt)) / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };
        }

        self.state.is_on_ground = num_wheels_in_contact >= 3;

        self.sticky_gate_prev = self.bullet_vehicle.wheels.iter().any(|wheel| {
            wheel
                .raycast_info
                .as_ref()
                .is_some_and(|info| info.is_in_contact_with_world)
        });
        // world_contact_normal still holds THIS step's contact here (cleared in
        // the next pre_tick_update).
        self.chassis_scrape_prev = self.state.world_contact_normal.is_some();
        if env_on("GGL_JUMP_SETTLE")
            && self.state.has_jumped
            && !self.state.is_jumping
            && self.state.is_on_ground
        {
            let extending = self.bullet_vehicle.wheels.iter().any(|w| {
                w.raycast_info
                    .as_ref()
                    .is_some_and(|ri| ri.suspension_relative_vel > 1.0)
            });
            if !extending {
                self.state.has_jumped = false;
            }
        }

        self.state.bump_cooldown_ticks = self.state.bump_cooldown_ticks.saturating_sub(1);
        self.state.prev_controls = self.state.controls;
    }

    pub(crate) fn finish_physics_tick(&mut self, rb: &mut RigidBody) {
        debug_assert_eq!(rb.world_array_idx, self.rigid_body_idx);

        if self.state.is_demoed {
            return;
        }

        if self.vel_impulse_cache != Vec3A::ZERO {
            rb.lin_vel += self.vel_impulse_cache;
            self.vel_impulse_cache = Vec3A::ZERO;
        }

        // Clamp BEFORE publishing, matching RL's observable state (SIM2REAL_AUDIT.md
        // S37). `GGL_VEL_CLAMP=head` skips this (tuned default: head of tick only).
        if ggl_vel_clamp_end() {
            rb.limit_vels(
                const { car_consts::MAX_SPEED * UU_TO_BT },
                car_consts::MAX_ANG_SPEED,
            );
        }

        self.state.phys.pos = rb.get_world_trans().translation * BT_TO_UU;
        self.state.phys.vel = rb.lin_vel * BT_TO_UU;
        self.state.phys.ang_vel = rb.ang_vel;
    }
}
