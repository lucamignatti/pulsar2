#![allow(dead_code)]

use std::mem::size_of;
use std::ops::{Index, IndexMut};

use glam::{Mat3A, Vec3A};
use rocketsim::{CarControls, CarState, PhysState};
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct VecRecord {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}
impl VecRecord {
    pub fn new(x: f32, y: f32, z: f32) -> VecRecord {
        VecRecord { x, y, z }
    }
    pub fn length(&self) -> f32 {
        (self.x.powi(2) + self.y.powi(2) + self.z.powi(2)).sqrt()
    }
}
impl From<VecRecord> for Vec3A {
    fn from(val: VecRecord) -> Self {
        Vec3A::new(val.x, val.y, val.z)
    }
}
impl Index<usize> for VecRecord {
    type Output = f32;
    fn index(&self, index: usize) -> &f32 {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => unreachable!(),
        }
    }
}
impl IndexMut<usize> for VecRecord {
    fn index_mut(&mut self, index: usize) -> &mut f32 {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => unreachable!(),
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Mat3Record {
    pub rows: [VecRecord; 3],
}
impl Mat3Record {
    pub fn column(&self, idx: usize) -> VecRecord {
        VecRecord::new(self.rows[0][idx], self.rows[1][idx], self.rows[2][idx])
    }
    pub fn forward(&self) -> VecRecord {
        self.column(0)
    }
    pub fn right(&self) -> VecRecord {
        self.column(1)
    }
    pub fn up(&self) -> VecRecord {
        self.column(2)
    }
}

/// v2 on-disk wheel (48 bytes).
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct WheelRecordV2 {
    pub susp_length: f32,
    pub susp_rel_vel: f32,

    pub has_contact: bool,
    pub contact_normal: VecRecord,

    pub steer_amount: f32,
    pub engine_force: f32,
    pub brake: f32,

    pub lat_friction: f32,
    pub long_friction: f32,
    pub extra_pushback: f32,
}

/// v3 on-disk wheel (88 bytes).
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct WheelRecordV3 {
    pub susp_length: f32,
    pub susp_rel_vel: f32,

    pub has_contact: bool,
    pub contact_normal: VecRecord,

    pub steer_amount: f32,
    pub engine_force: f32,
    pub brake: f32,

    pub lat_friction: f32,
    pub long_friction: f32,
    pub extra_pushback: f32,

    pub has_world_geometry_contact: bool,
    pub contact_location: VecRecord,
    pub lat_direction: VecRecord,
    pub long_direction: VecRecord,
}

/// v4 wheel (112 bytes). `had_contact` is the previous tick; v2/v3 files widen with zeros.
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct WheelRecordV4 {
    pub susp_length: f32,
    pub susp_rel_vel: f32,

    pub has_contact: bool,
    pub contact_normal: VecRecord,

    pub steer_amount: f32,
    pub engine_force: f32,
    pub brake: f32,

    pub lat_friction: f32,
    pub long_friction: f32,
    pub extra_pushback: f32,

    pub has_world_geometry_contact: bool,
    pub contact_location: VecRecord,
    pub lat_direction: VecRecord,
    pub long_direction: VecRecord,

    pub friction_curve_input: f32,
    pub spin_speed: f32,
    pub had_contact: bool,
    pub wheel_linear_velocity: VecRecord,
}

/// v5 wheel (184 bytes).
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct WheelRecord {
    pub susp_length: f32,
    pub susp_rel_vel: f32,
    pub has_contact: bool,
    pub contact_normal: VecRecord,
    pub steer_amount: f32,
    pub engine_force: f32,
    pub brake: f32,
    pub lat_friction: f32,
    pub long_friction: f32,
    pub extra_pushback: f32,
    pub has_world_geometry_contact: bool,
    pub contact_location: VecRecord,
    pub lat_direction: VecRecord,
    pub long_direction: VecRecord,
    pub friction_curve_input: f32,
    pub spin_speed: f32,
    pub had_contact: bool,
    pub wheel_linear_velocity: VecRecord,
    pub contact_force_distance: f32,
    pub has_contact_change_time: f32,
    pub wheel_radius: f32,
    pub ref_wheel_location: VecRecord,
    pub local_suspension_ray_start: VecRecord,
    pub preset_rest_position: VecRecord,
    pub contact_actor: u64,
    pub contact_component: u64,
    pub contact_phys_mat: u64,
}
const ZERO_VEC: VecRecord = VecRecord {
    x: 0.0,
    y: 0.0,
    z: 0.0,
};
impl From<WheelRecordV3> for WheelRecord {
    fn from(w: WheelRecordV3) -> Self {
        Self {
            susp_length: w.susp_length,
            susp_rel_vel: w.susp_rel_vel,
            has_contact: w.has_contact,
            contact_normal: w.contact_normal,
            steer_amount: w.steer_amount,
            engine_force: w.engine_force,
            brake: w.brake,
            lat_friction: w.lat_friction,
            long_friction: w.long_friction,
            extra_pushback: w.extra_pushback,
            has_world_geometry_contact: w.has_world_geometry_contact,
            contact_location: w.contact_location,
            lat_direction: w.lat_direction,
            long_direction: w.long_direction,
            friction_curve_input: 0.0,
            spin_speed: 0.0,
            had_contact: false,
            wheel_linear_velocity: ZERO_VEC,
            contact_force_distance: 0.0,
            has_contact_change_time: 0.0,
            wheel_radius: 0.0,
            ref_wheel_location: ZERO_VEC,
            local_suspension_ray_start: ZERO_VEC,
            preset_rest_position: ZERO_VEC,
            contact_actor: 0,
            contact_component: 0,
            contact_phys_mat: 0,
        }
    }
}
impl From<WheelRecordV4> for WheelRecord {
    fn from(w: WheelRecordV4) -> Self {
        Self {
            susp_length: w.susp_length,
            susp_rel_vel: w.susp_rel_vel,
            has_contact: w.has_contact,
            contact_normal: w.contact_normal,
            steer_amount: w.steer_amount,
            engine_force: w.engine_force,
            brake: w.brake,
            lat_friction: w.lat_friction,
            long_friction: w.long_friction,
            extra_pushback: w.extra_pushback,
            has_world_geometry_contact: w.has_world_geometry_contact,
            contact_location: w.contact_location,
            lat_direction: w.lat_direction,
            long_direction: w.long_direction,
            friction_curve_input: w.friction_curve_input,
            spin_speed: w.spin_speed,
            had_contact: w.had_contact,
            wheel_linear_velocity: w.wheel_linear_velocity,
            contact_force_distance: 0.0,
            has_contact_change_time: 0.0,
            wheel_radius: 0.0,
            ref_wheel_location: ZERO_VEC,
            local_suspension_ray_start: ZERO_VEC,
            preset_rest_position: ZERO_VEC,
            contact_actor: 0,
            contact_component: 0,
            contact_phys_mat: 0,
        }
    }
}
impl From<WheelRecordV2> for WheelRecord {
    fn from(w: WheelRecordV2) -> Self {
        Self {
            susp_length: w.susp_length,
            susp_rel_vel: w.susp_rel_vel,
            has_contact: w.has_contact,
            contact_normal: w.contact_normal,
            steer_amount: w.steer_amount,
            engine_force: w.engine_force,
            brake: w.brake,
            lat_friction: w.lat_friction,
            long_friction: w.long_friction,
            extra_pushback: w.extra_pushback,
            has_world_geometry_contact: false,
            contact_location: ZERO_VEC,
            lat_direction: ZERO_VEC,
            long_direction: ZERO_VEC,
            friction_curve_input: 0.0,
            spin_speed: 0.0,
            had_contact: false,
            wheel_linear_velocity: ZERO_VEC,
            contact_force_distance: 0.0,
            has_contact_change_time: 0.0,
            wheel_radius: 0.0,
            ref_wheel_location: ZERO_VEC,
            local_suspension_ray_start: ZERO_VEC,
            preset_rest_position: ZERO_VEC,
            contact_actor: 0,
            contact_component: 0,
            contact_phys_mat: 0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct ControlsRecord {
    pub throttle: f32,
    pub steer: f32,
    pub pitch: f32,
    pub yaw: f32,
    pub roll: f32,
    pub jump: bool,
    pub boost: bool,
    pub handbrake: bool,
}
impl From<ControlsRecord> for CarControls {
    fn from(val: ControlsRecord) -> Self {
        CarControls {
            throttle: val.throttle,
            steer: val.steer,
            pitch: val.pitch,
            yaw: val.yaw,
            roll: val.roll,
            jump: val.jump,
            boost: val.boost,
            handbrake: val.handbrake,
        }
    }
}

#[repr(u8)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum ImpulseRecordType {
    WheelsSuspension,
    WheelsFriction,
    StickyForce,
    Jump,
    DoubleJump,
    DodgeImpulse,
    DodgeTorque,
    AirControl,
    Boost,
    FlipCar,
    BallImpact,
    CarImpact,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct ImpulseRecord {
    pub lin_impulse: VecRecord,
    pub ang_impulse: VecRecord,
    pub impulse_type: ImpulseRecordType,
    pub is_accum: bool,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct PhysRecordV4 {
    pub physics_frame: u32,
    pub pos: VecRecord,
    pub rot: Mat3Record,
    pub lin_vel: VecRecord,
    pub ang_vel: VecRecord,
    pub has_world_contact: bool,
    pub world_contact_point: VecRecord,
    pub world_contact_normal: VecRecord,
    impulse_records_data: [ImpulseRecord; 8],
    num_impulse_records: u32,
}

/// v5 phys (344 bytes): v4 plus world-contact velocity.
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct PhysRecord {
    pub physics_frame: u32,

    pub pos: VecRecord,
    pub rot: Mat3Record,
    pub lin_vel: VecRecord,
    pub ang_vel: VecRecord,

    pub has_world_contact: bool,
    pub world_contact_point: VecRecord,
    pub world_contact_normal: VecRecord,

    impulse_records_data: [ImpulseRecord; 8],
    num_impulse_records: u32,
    pub world_contact_velocity: VecRecord,
}
impl From<PhysRecordV4> for PhysRecord {
    fn from(p: PhysRecordV4) -> Self {
        Self {
            physics_frame: p.physics_frame,
            pos: p.pos,
            rot: p.rot,
            lin_vel: p.lin_vel,
            ang_vel: p.ang_vel,
            has_world_contact: p.has_world_contact,
            world_contact_point: p.world_contact_point,
            world_contact_normal: p.world_contact_normal,
            impulse_records_data: p.impulse_records_data,
            num_impulse_records: p.num_impulse_records,
            world_contact_velocity: ZERO_VEC,
        }
    }
}
impl PhysRecord {
    pub fn impulse_records(&self) -> &[ImpulseRecord] {
        &self.impulse_records_data[..self.num_impulse_records as usize]
    }
}
impl From<PhysRecord> for PhysState {
    fn from(phys_record: PhysRecord) -> Self {
        // Verify rotation matrix is sane
        {
            for i in 0..3 {
                let c_len = phys_record.rot.rows[i].length();
                assert!((1.0 - c_len).abs() < 1e-6);

                // Dirs should be 90 degrees from other row dirs
                assert!(
                    Vec3A::dot(
                        phys_record.rot.rows[i].into(),
                        phys_record.rot.rows[(i + 1) % 3].into()
                    )
                    .abs()
                        < 1e-6
                );
            }
        }

        PhysState {
            pos: phys_record.pos.into(),
            vel: phys_record.lin_vel.into(),
            ang_vel: phys_record.ang_vel.into(),
            rot_mat: Mat3A::from_cols(
                phys_record.rot.column(0).into(),
                phys_record.rot.column(1).into(),
                phys_record.rot.column(2).into(),
            ),
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct CarRecord {
    pub phys: PhysRecord,

    pub is_on_ground: bool,
    pub is_jumping: bool,
    pub is_flipping: bool,
    pub jump_time: f32,
    pub flip_time: f32,
    pub has_jumped: bool,
    pub double_jumped_or_flipped: bool,
    pub has_flip: bool,
    pub flip_rel_torque: VecRecord,

    pub boost_amount: f32,

    pub is_touching_ball: bool,

    pub prev_controls: ControlsRecord,

    pub wheels: [WheelRecord; 4],

    pub drive_torque: f32,
    pub brake_torque: f32,
    pub output_throttle: f32,
    pub output_steer: f32,
    pub output_brake: f32,
    pub output_handbrake: f32,
    pub residual_lin: VecRecord,
    pub residual_ang: VecRecord,
    pub ground_normal: VecRecord,
    pub sticky_ground: f32,
    pub sticky_wall: f32,
    pub time_on_ground: f32,
    pub time_off_ground: f32,
    pub num_wheel_contacts: i32,
    pub num_wheel_world_contacts: i32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct CarRecordV4 {
    pub phys: PhysRecordV4,
    pub is_on_ground: bool,
    pub is_jumping: bool,
    pub is_flipping: bool,
    pub jump_time: f32,
    pub flip_time: f32,
    pub has_jumped: bool,
    pub double_jumped_or_flipped: bool,
    pub has_flip: bool,
    pub flip_rel_torque: VecRecord,
    pub boost_amount: f32,
    pub is_touching_ball: bool,
    pub prev_controls: ControlsRecord,
    pub wheels: [WheelRecordV4; 4],
    pub drive_torque: f32,
    pub brake_torque: f32,
    pub output_throttle: f32,
    pub output_steer: f32,
    pub output_brake: f32,
    pub output_handbrake: f32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct CarRecordV3 {
    pub phys: PhysRecordV4,
    pub is_on_ground: bool,
    pub is_jumping: bool,
    pub is_flipping: bool,
    pub jump_time: f32,
    pub flip_time: f32,
    pub has_jumped: bool,
    pub double_jumped_or_flipped: bool,
    pub has_flip: bool,
    pub flip_rel_torque: VecRecord,
    pub boost_amount: f32,
    pub is_touching_ball: bool,
    pub prev_controls: ControlsRecord,
    pub wheels: [WheelRecordV3; 4],
}
impl From<CarRecordV3> for CarRecord {
    fn from(c: CarRecordV3) -> Self {
        Self {
            phys: c.phys.into(),
            is_on_ground: c.is_on_ground,
            is_jumping: c.is_jumping,
            is_flipping: c.is_flipping,
            jump_time: c.jump_time,
            flip_time: c.flip_time,
            has_jumped: c.has_jumped,
            double_jumped_or_flipped: c.double_jumped_or_flipped,
            has_flip: c.has_flip,
            flip_rel_torque: c.flip_rel_torque,
            boost_amount: c.boost_amount,
            is_touching_ball: c.is_touching_ball,
            prev_controls: c.prev_controls,
            wheels: c.wheels.map(WheelRecord::from),
            drive_torque: 0.0,
            brake_torque: 0.0,
            output_throttle: 0.0,
            output_steer: 0.0,
            output_brake: 0.0,
            output_handbrake: 0.0,
            residual_lin: ZERO_VEC,
            residual_ang: ZERO_VEC,
            ground_normal: ZERO_VEC,
            sticky_ground: 0.0,
            sticky_wall: 0.0,
            time_on_ground: 0.0,
            time_off_ground: 0.0,
            num_wheel_contacts: 0,
            num_wheel_world_contacts: 0,
        }
    }
}

impl From<CarRecordV4> for CarRecord {
    fn from(c: CarRecordV4) -> Self {
        Self {
            phys: c.phys.into(),
            is_on_ground: c.is_on_ground,
            is_jumping: c.is_jumping,
            is_flipping: c.is_flipping,
            jump_time: c.jump_time,
            flip_time: c.flip_time,
            has_jumped: c.has_jumped,
            double_jumped_or_flipped: c.double_jumped_or_flipped,
            has_flip: c.has_flip,
            flip_rel_torque: c.flip_rel_torque,
            boost_amount: c.boost_amount,
            is_touching_ball: c.is_touching_ball,
            prev_controls: c.prev_controls,
            wheels: c.wheels.map(WheelRecord::from),
            drive_torque: c.drive_torque,
            brake_torque: c.brake_torque,
            output_throttle: c.output_throttle,
            output_steer: c.output_steer,
            output_brake: c.output_brake,
            output_handbrake: c.output_handbrake,
            residual_lin: ZERO_VEC,
            residual_ang: ZERO_VEC,
            ground_normal: ZERO_VEC,
            sticky_ground: 0.0,
            sticky_wall: 0.0,
            time_on_ground: 0.0,
            time_off_ground: 0.0,
            num_wheel_contacts: 0,
            num_wheel_world_contacts: 0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct CarRecordV2 {
    pub phys: PhysRecordV4,
    pub is_on_ground: bool,
    pub is_jumping: bool,
    pub is_flipping: bool,
    pub jump_time: f32,
    pub flip_time: f32,
    pub has_jumped: bool,
    pub double_jumped_or_flipped: bool,
    pub has_flip: bool,
    pub flip_rel_torque: VecRecord,
    pub boost_amount: f32,
    pub is_touching_ball: bool,
    pub prev_controls: ControlsRecord,
    pub wheels: [WheelRecordV2; 4],
}
impl From<CarRecordV2> for CarRecord {
    fn from(c: CarRecordV2) -> Self {
        Self {
            phys: c.phys.into(),
            is_on_ground: c.is_on_ground,
            is_jumping: c.is_jumping,
            is_flipping: c.is_flipping,
            jump_time: c.jump_time,
            flip_time: c.flip_time,
            has_jumped: c.has_jumped,
            double_jumped_or_flipped: c.double_jumped_or_flipped,
            has_flip: c.has_flip,
            flip_rel_torque: c.flip_rel_torque,
            boost_amount: c.boost_amount,
            is_touching_ball: c.is_touching_ball,
            prev_controls: c.prev_controls,
            wheels: c.wheels.map(WheelRecord::from),
            drive_torque: 0.0,
            brake_torque: 0.0,
            output_throttle: 0.0,
            output_steer: 0.0,
            output_brake: 0.0,
            output_handbrake: 0.0,
            residual_lin: ZERO_VEC,
            residual_ang: ZERO_VEC,
            ground_normal: ZERO_VEC,
            sticky_ground: 0.0,
            sticky_wall: 0.0,
            time_on_ground: 0.0,
            time_off_ground: 0.0,
            num_wheel_contacts: 0,
            num_wheel_world_contacts: 0,
        }
    }
}

impl From<CarRecord> for CarState {
    fn from(phys_record: CarRecord) -> Self {
        Self {
            phys: phys_record.phys.into(),
            boost: phys_record.boost_amount,
            controls: phys_record.prev_controls.into(),
            is_on_ground: phys_record.is_on_ground,
            is_jumping: phys_record.is_jumping,
            is_flipping: phys_record.is_flipping,
            flip_rel_torque: phys_record.flip_rel_torque.into(),
            jump_ticks: rocketsim::consts::secs_to_ticks(phys_record.jump_time),
            flip_ticks: rocketsim::consts::secs_to_ticks(phys_record.flip_time),
            has_jumped: phys_record.has_jumped,
            ..Default::default()
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct RecordingInfo {
    pub num_cars: u32,
    pub hitbox_rel_min_bt: VecRecord,
    pub hitbox_rel_max_bt: VecRecord,
}

const _: () = {
    assert!(size_of::<WheelRecordV2>() == 48);
    assert!(size_of::<WheelRecordV3>() == 88);
    assert!(size_of::<WheelRecordV4>() == 112);
    assert!(size_of::<WheelRecord>() == 184);
    assert!(size_of::<CarRecordV2>() == 584);
    assert!(size_of::<CarRecordV3>() == 744);
    assert!(size_of::<CarRecordV4>() == 864);
    assert!(size_of::<CarRecord>() == 1232);
    assert!(size_of::<PhysRecordV4>() == 332);
    assert!(size_of::<PhysRecord>() == 344);
    assert!(size_of::<RecordingInfo>() == 28);
};
