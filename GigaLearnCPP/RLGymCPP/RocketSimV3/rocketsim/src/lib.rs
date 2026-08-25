#![allow(
    clippy::suboptimal_flops,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation
)]

mod base;
mod bullet;
mod glam_inc;
mod logging;
pub mod shared;
mod sim;
///////////

pub use base::*;
pub use bullet::collision::dispatch::{
    LAST_SAT_AXIS, LAST_SAT_GAP_UU, LAST_SAT_KIND, reset_last_sat_gap,
};
pub use bullet::dynamics::rigid_body::DBG_IMPULSE_TRACE;
pub use glam_inc::*;

pub use crate::sim::*;
