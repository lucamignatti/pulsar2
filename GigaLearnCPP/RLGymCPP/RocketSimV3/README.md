# RocketSimV3 — vendored Rust engine + C FFI

Vendored from https://github.com/ZealanL/RocketSim branch `v3-rust`
at commit `b1c2ece647417b672bf73129b6f1415fc68b544f`
("Fix double jump impulse scaling", 2026-07-11), written by VirxEC/ZealanL.
`rocketsim/` and `rocketsim_derive/` are UNMODIFIED upstream — keep it that way;
carry any local need in `rocketsim_ffi/` (ours) or the C++ compat layer
(`compat/`), so upstream bumps stay a clean directory swap.

Why v3 (2026-07-17, user decision): measured ~3x single-thread physics speedup
(same protocol/meshes/machine: 267k → 815k TPS @ 2 cars) and an upstream
accuracy harness that compares per-tick impulses against real recorded game
data. Known accepted risks at the pinned commit: upstream's own
`case_simple_jump_land` comparison test FAILS (missing sticky-force impulse,
suspension impulse ~14% off) and boost-pad box-hitbox pickup is a TODO
(cylinder-only). Physics differs from v2 → checkpoints trained on v2 dynamics
are behaviorally stale; engine swap requires a fresh run.

Collision meshes: the STANDARD 16-mesh soccar dump (all hashes on the whitelist
both v2 and v3 embed) — sourced from the `rlgym_rocket_league` pip package
(`site-packages/rlgym/rocket_league/sim/collision_meshes/`). v3 hard-rejects
non-whitelisted meshes (v2 only warned); the repo's previous 10-mesh dump was
nonstandard and is parked at `build/collision_meshes/soccar_nonstandard_backup`.

Build: CMake drives `cargo build --release -p rocketsim_ffi` (staticlib) via
`rust-toolchain.toml` (auto-installs the pinned nightly through rustup; the
training box needs rustup once: `curl https://sh.rustup.rs | sh`).
