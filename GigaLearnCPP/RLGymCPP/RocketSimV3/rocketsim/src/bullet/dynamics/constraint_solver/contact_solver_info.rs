pub const NUM_ITERATIONS: usize = 10;
pub const SOR: f32 = 1.0;
/// Positional-error ERP for the wheel hard-contact resolve (`resolve_single_collision`).
/// MEASURED, not inherited: swept 0.0/0.05/0.10/0.15/0.20/0.30 against both real captures
/// (SIM2REAL_AUDIT S45b). The maneuver battery has a clean interior minimum at 0.10 on the
/// fit AND holdout captures independently (fit 1046.5/1035.2/**1027.8**/1029.4/1035.2/1035.9,
/// holdout 1076.4/1065.4/**1057.3**/1059.2/1064.7/1065.4), so this is a real optimum rather
/// than a monotone slide to "no positional correction". Bullet's stock 0.2 was never fit to
/// Rocket League. Only valid TOGETHER WITH `PUSHBACK_MAX_IMPULSE` (S39): uncapped, 0.1 is
/// much WORSE (fit 1186.0) than the shipped pair.
pub const ERP: f32 = 0.1;
pub const ERP_2: f32 = 0.8;
pub const SPLIT_IMPULSE_PENETRATION_THRESHOLD: f32 = 1e30;
pub const SPLIT_IMPULSE_TURN_ERP: f32 = 0.1;
pub const WARMSTARTING_FACTOR: f32 = 0.85;
pub const RESTITUTION_VELOCITY_THRESHOLD: f32 = 0.2;
