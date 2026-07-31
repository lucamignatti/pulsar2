mod raycaster;
mod vehicle_rl;
mod wheel_info;

pub const NUM_WHEELS: usize = 4;

pub use vehicle_rl::VehicleRL;
// RaycastInfo is exported so a car's suspension state can be captured and restored
// exactly (see sim::car::car_extra_state).
pub use wheel_info::{RaycastInfo, WheelInfo};
