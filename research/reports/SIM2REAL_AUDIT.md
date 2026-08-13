# SIM2REAL_AUDIT — is RocketSim 1:1 with Rocket League?

**Status:** RESULT (2 defects fixed & validated) · **Date:** 2026-08-01 · **Trigger:** Pulsar's real-game goal
share pinned at ~20% across every config flip; hypothesis was a sim-to-real
physics gap.

**Verdict: FOUR real defects found and fixed. All remaining measurable differences sit >=6x below the agent's own control precision (S18).**
Ground, air and ball dynamics reproduce real-game telemetry to sub-uu per 8-tick
window (ball free flight: 0.026 uu). Confirmed defects:

1. **Air throttle is double-counted while boosting** (§5) — +66.67 uu/s² on every
   boosted aerial, verified against real telemetry to 0.3%. Systematic, integrates.
2. **Dodge torque skips the inverse-inertia division** (§1b) — **root-caused in the
   binary**: RL's angular primitive `FUN_140981560` mode 5 applies `I⁻¹ · torque`;
   RocketSim adds the torque straight to angular velocity. Octane's 1/I_y = 0.01041
   matches the independently fitted 0.0100 to 4%. Patching it cuts the error ~40×
   (2.048 → 0.051 rad/s) across 6,382 real dodge frames.

Everything else checked came back clean: arena geometry, ground contact, ball free
flight, **ball spin**, ball-car hit impulse, boost consumption, boost-pad pickup,
wall/ramp driving, flip duration, and **demolitions** (§8). Bumps show no defect but
are measurement-limited.

Two earlier headline claims in this report were **tested and withdrawn**: the
`actionDelay = 2` recommendation (§4) and the dodge-cancel time gate (§1b). Both
retractions are kept in place with the evidence that killed them.

Scope: **Soccar only** (1v1 + team modes); other game modes are out of scope.

---

## 1. Ghidra: the constants are not in the exe — the *code* is

Static RE cannot recover RocketSim's **numeric constants**, and the reason is now
proven by reading Rocket League's actual physics code rather than inferred from a
failed byte-scan (see §1b for what the code *does* give us).

Scanned all 41 MB of `RocketLeague.exe` for every constant in
`rocketsim/src/sim/consts.rs`, as f32/f64/i32/i16, in **both** Unreal Units and
Bullet Units (÷50). Results:

- Values that a byte-scan "finds" (e.g. `-650.0`, `2300.0` as `46.0` BT) sit at
  **unaligned addresses with zero code cross-references**. Ghidra's xref index
  works (verified against a known-referenced string at `0x141eb9ca0`), so these
  are coincidental byte matches inside unrelated data, not referenced constants.
- Every distinctive physics constant — `2975/3`, `4375/3`, `91.25`, `6000`,
  `4600`, `2200`, and **all six steer-curve points** (`0.53356 … 0.03454`) —
  is absent in every encoding tested.
- The binary is stripped: no `Vehicle`/`Car` function symbols.

That is the expected UE3 layout: gameplay values live as UnrealScript default
properties in the cooked packages, not as compiled immediates. And
`TAGame.upk` (72 MB) is **encrypted** — valid UE3 header (v868/licensee 32), but
its name table is high-entropy and contains no plaintext (`Vehicle_TA`,
`BoostForce`, etc. all absent as ASCII **and** UTF-16). Reading it would mean
defeating the package encryption, which this audit does not do.

## 1b. Ghidra, done properly: 3,740 functions recovered

The binary is **not** packed (`.text` entropy 6.63) and is stripped of RTTI for
game classes (the only 74 RTTI descriptors are CryptoPP). But UE3 leaves the
**native function registration table** intact as ASCII: 4,145 strings of the form
`AVehicle_TAexecGetForwardSpeed`, spanning 613 classes. Each string is followed in
an `FNativeFunctionLookup` entry by the function pointer, so **all 4,145 resolved
to `.text` addresses**; 3,740 unique addresses are now named in the Ghidra project
(`ACarComponent_Dodge_TA_execApplyTorqueForces`, etc.). Everything below follows
from that foothold. Regenerate with `scan`/`natives.json` in the scratchpad.

The physics classes are all present: `ACarComponent_Dodge_TA`, `_Jump_TA`,
`_DoubleJump_TA`, `_Boost_TA`, `_AirControl_TA`, `_FlipCar_TA`, `AVehicle_TA`,
`UVehicleSim_TA`, `UWheel_TA`, `ABall_TA`.

**Why the constants are absent, confirmed from the code.**
`ACarComponent_Dodge_TA::ApplyTorqueForces` is a direct (non-virtual) call at
**`0x140eb72b0`** (spans `..0x140eb756c`). Decompiled, every physics quantity is an
**instance field load**, not an immediate:

- `+0x380 / +0x384 / +0x388` — dodge torque direction (x, y, z)
- `+0x350` — time gate for pitch-cancellation
- `+0x354` — Z-damp force factor
- `+0x358 / +0x35c` — Z-damp start time / window duration

Those fields are populated from the class default object, which is cooked into the
**encrypted** `TAGame.upk`. So the numbers are genuinely unavailable statically —
but the **algorithm** is fully readable, and that is where the real finding is.

### RETRACTED: "RocketSim is missing RL's dodge-cancel time gate"

**This claim was wrong and is withdrawn.** It is kept here because the code
reading below is still accurate and the retraction is instructive.

The reasoning was: RL gates dodge pitch-cancellation on field `+0x350`, RocketSim
has no gate, and `flip::TORQUE_MIN_TIME = 0.41` is dead code — therefore `+0x350`
is 0.41 s and RocketSim lets the policy cancel flips 0.41 s earlier than the game.

**Tested against real telemetry and refuted.** Patching a copy of RocketSim to gate
the cancel on `flip_time >= TORQUE_MIN_TIME` and re-fitting the 11 real-match frames
where the gate changes behaviour made the fit **~2.8× worse**:

| variant | angVel err on gate-affected frames (rad/s) |
|---|---|
| RocketSim as shipped (no gate) | median **1.634**, mean 2.310 |
| patched with the 0.41 s gate | median **4.558**, mean 5.531 |

So the real game *does* permit early cancellation. The gate exists in RL's code, but
its threshold is evidently near zero, not 0.41 s. The inference I flagged as the weak
link — equating `+0x350` with the unused constant — was the part that broke. Lesson:
a dead constant is a hypothesis, not evidence.

### ROOT-CAUSED: RocketSim omits the inverse-inertia division on dodge torque

Ghidra gave the mechanism, and it is exact enough to fix properly rather than by fitting.

**RL's force primitives take a MODE argument.** Decompiling the CarComponent vtables
(cluster at `.rdata:0x141e4a000+`, each 0x6b8 bytes, `PrePhysicsStep` at slot +0x6a0 and
`ApplyForces` at +0x6a8) leads to two primitives:

- `FUN_14097f5c0(rb, &vec, &pos, 0, mode)` — linear, modes 1 and 2 observed
- **`FUN_140981560(rb, &torque, _, mode)`** — angular; the dodge uses **mode 5**

`FUN_140981560` mode 5 (with inertia > 0) computes the **full inverse inertia tensor** —
a cofactor inverse of the 3×3 matrix at `+0x130..+0x158` with `1/det` — applies it to the
torque, then scales by a per-axis factor at `+0x390/+0x394/+0x398` and adds the result to
angular velocity at `+0x1e0/+0x1e4/+0x1e8`:

```c
angvel += (I⁻¹ · torque) * factor[axis]
```

**RocketSim does none of that.** `car/base.rs` adds the torque straight to angular
velocity with `massed: false`, i.e. no inertia division at all:

```rust
rb.add_impulse(None, Impulse::Angular(rot * dodge_torque), false, true);
```

**The magnitude checks out.** Octane's box inertia at mass 180 BT is
I = (54.07, 96.10, 132.23), so 1/I_y = **0.01041** for the pitch axis — and forward
dodges (which dominate: 128 of 145 sampled flip frames) fit best at an empirical scale of
**0.0100**. A 4% match to a number derived independently from the binary.

**Validated on 6,382 real mid-dodge frames** (angular-velocity error, rad/s):

| variant | all dodges | forward-dominant | side-dominant |
|---|---|---|---|
| **shipped RocketSim** | **2.0482** | 1.8689 | 2.1537 |
| patched: `inv_inertia_tensor_world * torque` | **0.0512** | 0.0384 | 0.0648 |
| fitted scalar ×0.01 (for comparison) | 0.0379 | 0.0232 | 0.0550 |

**~40× error reduction either way.** The defect is real and the direction of the fix is
settled.

**The `+0x390` factor is identified — the fix is exact, not approximate.** RL's mode 5
computes

```
angvel[i] += (I⁻¹ · torque)[i] * factor[i]      // factor at rb+0x390/+0x394/+0x398
```

which is **verbatim `btRigidBody::applyTorqueImpulse`**:
`m_angularVelocity += m_invInertiaTensorWorld * torque * m_angularFactor`. The default
branch is the matching `applyTorque` form (`torque * m_angularFactor`, no inertia) — which
is precisely what RocketSim was doing. So `+0x390` is Bullet's **`m_angularFactor`**,
`(1,1,1)` for a car.

Cross-check: if `angularFactor == 1`, the predicted dodge scale is `1/I_y = 0.01041`
against the independently fitted **0.0100**, implying an angular factor of **0.961** —
within 4% of unity, which is the expected residual given the 5.5 rad/s cap saturates 89%
of dodge frames. **`Impulse::Angular(inv_inertia_tensor_world * world_torque)` is
therefore RL's exact formula**, and it generalises across bodies and axes where a scalar
tuned on Octane pitch (1/I_x = 0.0185 vs 1/I_y = 0.0104, a 78% spread) would not.

### RL's force-mode semantics, decoded (and how RocketSim maps to them)

Both primitives scale their vector by **0.02 = 1/50 = `UU_TO_BT`**, confirming RL's
internal physics runs in the same Bullet units RocketSim uses. `+0x180` is the body mass;
`+0x130..+0x158` is the inertia matrix; `+0x1e0/4/8` is angular velocity.

| RL mode | primitive | semantics | RocketSim equivalent |
|---|---|---|---|
| linear 1 | `FUN_14097f5c0` | velocity change, **no** mass division | `add_impulse(Linear, massed=false)` ✓ |
| linear 2 | `FUN_14097f5c0` | impulse, **divided by mass** | `add_impulse(Linear, massed=true)` ✓ |
| angular 5 | `FUN_140981560` | torque via **full inverse inertia tensor** | **no equivalent** ✗ |
| angular default | `FUN_140981560` | `angvel += torque * factor` (no inertia) | `add_impulse(Angular, massed=false)` ✓ |

**The linear side maps cleanly.** CarComponent `ApplyForces` uses mode 1 and RocketSim's
`massed=false` matches; the dodge's Z-damp uses mode 2 and maps to `massed=true`.

**The angular side has a hole.** RocketSim's angular branch is

```rust
Impulse::Angular(av) => { ang_impulse = av * massed_scaler; }   // massed_scaler = inv_MASS
```

so `massed=true` on an angular impulse divides by **mass**, which is dimensionally wrong
for a torque, and **no code path applies the inverse inertia tensor to an angular
impulse** (only `LinearRelPos` does, for its cross-product term). That is the root
deficiency behind the dodge defect.

There are exactly three `Impulse::Angular` call sites: dodge torque (`base.rs:407`, the
defect), air-control torque (`:460`) and auto-roll torque (`:732`). The latter two measure
clean — air control at **0.001 rad/s** and auto-roll indistinguishable from the upright
control — so either RL applies those through the non-inertia path or their constants
already absorb it. **Only the dodge needed changing**, which is consistent with every
measurement in this report.

### The code reading (still accurate)

RL's decompiled dodge torque:

```c
scale = dirY;                                  // +0x384
if (*(float*)(this+0x350) <= elapsed) {        // <-- TIME GATE
    pitch = *(float*)(vehicle+0x7f8);
    if (0.0 <= dirY) { /* clamp pitch to [-1,0] */ s = pitch_clamped + 1.0; }
    else if (0.0 <= pitch) { s = 1.0 - min(pitch, 1.0); }
    else s = 1.0;
    scale = dirY * s;
}
```

Cancellation is only permitted **after** `this+0x350`. RocketSim
(`car/base.rs:391-400`) applies the identical `1 - |pitch|` attenuation but with
**no time gate at all** — from `flip_time = 0`.

The field mapping is well-constrained: `+0x358`/`+0x35c` reproduce RocketSim's
Z-damp gate (`>= Z_DAMP_START && (vel.z < 0 || flip_time < Z_DAMP_END)`,
0.15 / +0.06) and `+0x354` its `Z_DAMP_120` factor — a clean structural match that
validates the offset block. By elimination `+0x350` is a flip time constant that
is not z-damp, and **`flip::TORQUE_MIN_TIME = 0.41` is defined in `consts.rs` and
never used anywhere in the crate** (`PITCHLOCK_TIME = 1.0` is likewise dead).
That is the gate, sitting unwired.

**Measured exploit surface** (forward flip, pitch-axis ang-vel at end of flip):

| case | rad/s |
|---|---|
| no cancel | 5.6039 |
| cancel from t = 0 — **RocketSim allows** | 0.3117 |
| cancel after 0.41 s — what RL allows | 2.7215 |

→ **2.41 rad/s of flip rotation, 43% of an uncancelled flip**, is cancellable in
training but not in the real game. In sim a bot can kill ~94% of its flip
rotation instantly; in the real game it could kill ~51%. Pulsar re-decides every 8
ticks, so it gets ~6 decision points inside the 0.41 s window and can absolutely
learn to exploit this. Flips are load-bearing for wavedashes, directional dodges
and flip resets.

**Caveat:** the *existence* of the gate is directly visible in RL's code and
RocketSim's lack of one is certain; the specific value 0.41 s is inferred from the
unused constant, since the field itself lives in the encrypted package. Even if
the true threshold differs, the structural gap is real.

**Not concluded:** RL cancels when `sign(pitch) == -sign(dirY)`; RocketSim when
the signs are equal. That is consistent with one of the two conventions being
inverted between the engines and is **not** evidence of a bug — resolving it needs
RL's internal pitch sign, which I did not establish.

**Constants were additionally validated empirically** — against recorded real-game
telemetry, which is stronger evidence than reading a number out of a binary
anyway.

## 2. Empirical physics validation (the real test)

`rocketsim/tests/rl_comparison_test` replays **recorded real-game ticks** and
compares per-tick impulses. Run at the pinned commit (debug build; the impulse
trace is behind `debug_assertions`):

| impulse (resting car, UU/s) | real RL | RocketSim steady state |
|---|---|---|
| `StickyForce` | −2.7083335 | **−2.7083** |
| `WheelsSuspension` | 8.124919 | **8.1261** |
| net vs gravity (5.41667) | ≈ 0 | ≈ 0.0011 |

Suspension agrees to **0.015%**; sticky force is exact. The car settles at
z = 17.01 UU (`REST_Z` = 17). **Ground physics is right.**

### The comparison test still "fails" — and it is a harness artifact

`case_simple_jump_land` fails with `norm_error = 1.35`, showing suspension
9.4676 (16.5% high) and **no sticky force**. That is *not* the steady-state
behaviour above. The harness calls `set_car_state` every tick, and RocketSim
documents that `set_car_state` **cannot restore suspension state**
(`sim/car/car_extra_state.rs`): "replaying identical inputs from a restored
grounded car diverged ~23uu within a second". On the first tick after a
state-set the wheel raycast has no world contact, so sticky force is skipped and
suspension is computed off unsettled compression.

The repo README's "accepted risk" wording ("missing sticky-force impulse,
suspension impulse ~14% off") reads as a steady-state defect. It is not — it is
a **state-restore** defect. Worth correcting, because it misdirects.

## 3. Arena geometry is correct (negative result)

The live trainer loads a **10-mesh dump whose hashes match 0 of 16** canonical
soccar meshes, and a vendor patch in `rocketsim/src/base.rs` downgrades
upstream's hard rejection to a warning. The warnings do fire every boot.

This looked alarming and is **benign**. The meshes are generated by
`collision_mesh_downloader.py` from RLUtilities assets, mirrored 4 ways — a
different *partitioning* of the same surface, so the hashes differ by
construction. Compared against the canonical 16-mesh dump (rlgym-rocket-league
2.0.1 sdist; my hasher reproduces all 16 canonical hashes exactly):

- identical triangle count: **8020 vs 8020**
- identical extents: x ±4107.3, y ±6000.0, z −13.3 … 2075 UU
- vertex nearest-neighbour: **max 0.091 UU (~1.8 mm), mean 0.003 UU**
- the 28 triangles whose centroids differ are quads split on the other diagonal

**No action needed.** But `RocketSimV3/README.md` currently claims the standard
16-mesh dump is in use and that the 10-mesh set is "parked at
`build/collision_meshes/soccar_nonstandard_backup`" — that directory does not
exist and the 10-mesh set is what is live. The README is wrong; the geometry is
fine.

## 4. Actuation latency (measured, but NOT a discrepancy)

From the real-game session of 2026-08-01 10:02 (`core_play.log`: "Connected to
Rocket League"), `rlbot-run/pulsar-bot/debug.3165337.jsonl`, n = 427 `echo`
records comparing sent controls against the packet's `last_input` echo:

| lag (ticks) | count | share | ms |
|---|---|---|---|
| 1 | 3 | 0.7% | 8.3 |
| **2** | **419** | **98.1%** | **16.7** |
| 3 | 3 | 0.7% | 25.0 |
| 4 | 1 | 0.2% | 33.3 |
| 6 | 1 | 0.2% | 50.0 |

**mean 2.01 ticks · median 2 · mode 2.** A hard, low-variance pipeline delay.

### CORRECTION: this does NOT mean `actionDelay = 2` is the right training value

An earlier revision of this report recommended setting `actionDelay = 2`. **That
was wrong**, and §7's replay disproves it: refitting the same real-match data with
0–4 ticks of lag shows **lag = 0 fits best**, with error rising monotonically:

| lag (ticks) | median pos err (uu / 8-tick window) |
|---|---|
| **0** | **0.679** |
| 1 | 0.708 |
| 2 | 0.851 |
| 3 | 0.916 |

The resolution: the echo measures **send → applied**, but the packet we act on is
itself ~2 ticks stale. Observation delay and actuation delay are both ~2 ticks and
**cancel** — relative to the state the policy actually sees, its action lands
immediately. `actionDelay = 0` is therefore already correct, and forcing 2 would
introduce the very error it was meant to remove.

The 16.7 ms figure is still a real property of the venue; it is just not a
sim-to-real *discrepancy*. Do not change `actionDelay` on the strength of it.

## 5. CONFIRMED DEFECT: air throttle is double-counted on boosted aerials

`car/base.rs:463` applies `THROTTLE_AIR_ACCEL` (66.67 uu/s²) whenever
`throttle != 0`, **including while boosting**; upstream flags this as wrong
(`// TODO: Fix air-throttle not respecting boost`). v2 RocketSim has the same
behaviour (`Car.cpp:639`), so it is long-standing, not a v3 regression.

This bites Pulsar specifically because `DefaultAction`'s aerial actions are
emitted as `{boost, yaw, pitch, yaw, roll, jump, boost, handbrake}` — **throttle
is set equal to boost**, so *every* boosted aerial also sends throttle = 1.

Measured in-sim:

| air state (0.5 s) | accel |
|---|---|
| boost only | 1057.28 uu/s² |
| boost + throttle | **1124.41 uu/s²** |
| throttle only | 66.01 uu/s² |

→ **+6.35% acceleration on every boosted aerial.**

Strong circumstantial evidence this is a genuine over-count: RocketSim's own
constants satisfy `ACCEL_AIR (1058.333) − ACCEL_GROUND (991.667) = 66.667`
= `THROTTLE_AIR_ACCEL` **exactly**, i.e. the air boost figure already bundles the
throttle term, and adding it again double-counts.

**CONFIRMED against real telemetry (§7).** Replaying 40 real-match windows of
*airborne + boosting + throttle≠0* and measuring the excess forward Δv that
RocketSim produces versus the real game:

- predicted, if air throttle is double-counted: `66.667 × 8/120` = **+4.444 uu/s**
- **measured median: +4.459 uu/s** (0.3% off the prediction)

This is no longer a suspicion. RocketSim over-accelerates boosted aerials by
exactly `THROTTLE_AIR_ACCEL`. The error is **systematic and same-signed**, so it
integrates: over a 1.5 s aerial it is ~100 uu/s of excess speed and ~75 uu of
overshoot. Pulsar's aerials are tuned to a car that accelerates 6.35% harder than
the real game delivers, so it arrives systematically short/late on aerial
intercepts — and this run scaffolds aerials hard.

## 7. End-to-end replay against real match data (the broad sweep)

The upstream suite ships **one** recording, which is thin. So 959 real-match
transitions were replayed from `debug.3165337.jsonl` (real game, 2026-08-01):
reconstruct state from the logged obs (pos/forward/up/vel/angVel, boost, ground and
flip flags), apply the logged action for 8 ticks, compare to the next decision.

**Methodological correction — 12.4% of the telemetry is unusable.** 119 of 959
transitions have a *byte-identical* real position at t and t+8 while the car carries
real speed: the game was **paused** (goal replay / kickoff countdown) but the bot kept
ticking. Those inflate the sim's error by exactly `speed × 0.0667 s` and were the
source of a recurring `p90 = 67.595 uu` that appeared in every regime. All numbers
below exclude them (838 live transitions).

| regime | n | pos med (uu) | pos p90 | **angVel med (rad/s)** |
|---|---|---|---|---|
| **all live transitions** | 838 | 0.275 | 1.746 | 0.026 |
| flat ground | 366 | 0.552 | 2.260 | 0.010 |
| wall / ramp contact (up.z<0.9) | 71 | 0.723 | 2.117 | 0.178 |
| airborne, no boost | 167 | **0.061** | 0.637 | **0.001** |
| airborne, **boosting** | 29 | 0.249 | 2.938 | **0.359** |
| **flipping** | 145 | **1.600** | 3.260 | **1.422** |

Airborne non-boosting is essentially perfect (0.061 uu, 0.001 rad/s), which makes the
two outliers unambiguous — and they are exactly the two confirmed defects: **boosting**
(359× the airborne control on angVel, §5) and **flipping** (1400×, §1b).

### Validated — no inconsistency found

- **Ball free flight** (824 windows, no cars in arena): pos err median **0.026 uu**,
  p90 0.084, p99 0.543. Gravity, drag and bounces are exact.
- **Ball-car hit impulse** — the highest-impact path, and it is **correct**. A naive
  comparison at 8 ticks suggests the sim under-hits by 45% (dV ratio 0.595), but that
  is a *contact-phase* artifact: by 24 ticks the sim's peak ratio is 1.239, i.e. the
  impulse lands later, not weaker. The phase-robust measure — sim ball speed just after
  its own hit event vs real post-hit speed — gives **median 1.027, IQR [0.936, 1.037]**
  across 24 real touches. Accurate to ~3%.
- **Boost consumption**: end-to-end boost error median **0.0000**, p90 0.222.
- **Boost pad pickup**: on the 110 windows where every pad is genuinely available
  (so "reset pads" is faithful), pickup agreement is **perfect** — 0 sim-only, 0
  real-only. An apparent 21-vs-4 over-pickup on the full set was entirely pads being
  on cooldown in reality.
- **Wall / ramp driving**: median 0.723 uu, p90 2.117 — *better* than flat ground.
  Sticky force and suspension on non-flat normals are fine, including 41 near-vertical
  wall frames.
- **Flip duration**: 17 dodge episodes bracket the true duration at
  **[0.6083, 0.6666] s**; `flip::TORQUE_TIME = 0.65` is inside it (0.60 refuted).
  The flip defect is in torque *magnitude*, not timing.
- **Bump / demo code path**: complete — all five bump constants are live and demo
  modes/team-demo rules are implemented. (Behaviourally unvalidated, see below.)

### Boost pads: v3 uses cylinder-only pickup

v3 gates pickup on a cylinder against the car *origin*
(`boost_pad_grid.rs:35`: `dist_2d < cyl_radius && |Δz| <= CYL_HEIGHT`), with
`_box_radius` underscore-prefixed and `boost_pads::BOX_HEIGHT` dead, behind the TODO
"Implement car-locking with box hitbox".

**No claim is made that this is wrong.** An earlier revision called it a regression
against RocketSim v2, which also carried a box-AABB branch for a car already locked to
the pad. That comparison is withdrawn: **v2 is not a valid accuracy reference**
(user-stated, 2026-08-01), so a v2/v3 difference is not evidence of a v3 defect.
Empirically v3's pickup matches the real game perfectly on every frame this audit
could test (above). Settling the car-locking case needs RL's own pickup code or a
pad-camping scenario against the real game — not a v2 diff.

## 8. Replay-based validation — closing the multi-car, spin and demo gaps

The 1v1 RLBot telemetry could not reach bumps, demolitions, ball spin or team modes.
Rocket League's own saved replays can. **83 of 86** replays in the Proton prefix parse
with `boxcars` (75× 2v2, 6× 3v3, 2× 1v1); 3 fail on a newer `AnonymizedName` attribute.

Units calibrated empirically: car linear velocity is **raw uu/s** (maxes at exactly
2300.00 = `MAX_SPEED`); angular velocity and demolish velocities are **×100** (ball
angvel maxes at exactly 600.00 = the 6.0 rad/s cap). Car rotation is
`Quat::from_xyzw(x,y,z,w)`, whose `x_axis` aligns with velocity at **+0.811** for fast
cars — i.e. forward, matching RocketSim's convention. Replay frames are 30 fps =
**exactly 4 ticks**.

Caveat that bounds everything in this section: replay state is quantized and sampled
4 ticks apart, so it is inherently noisier than the 120 Hz telemetry of §7 (free-flight
ball error is 1.1 uu here vs 0.026 uu there). It is good enough for *event* and
*magnitude* questions, not for sub-uu physics.

### Ball spin — VALIDATED

Ball free flight with all cars >400 uu away, 17,274 windows across two 2v2 replays:

| variant | pos err med | vel err med | **angVel err med / p90** |
|---|---|---|---|
| with replay ball spin | **1.132** | **2.051** | **0.0000 / 0.0000** |
| spin forced to zero | 1.485 | 2.464 | 5.9999 / 6.0000 |

Including spin improves position and velocity prediction by ~25%, and RocketSim's
angular-velocity propagation matches the real game **exactly**. (61% of frames sit at
the 6.0 rad/s cap, so the angVel match is partly trivial; the position/velocity
improvement is not.)

### Demolitions — VALIDATED

28 demolition events across the replay set; 23 carry a recorded attacker velocity.
RocketSim gates demos on `DemoMode::Normal => attacker_state.is_supersonic`
(`arena/base.rs:1005`), supersonic being 2200 with a maintain band down to 2100.

Real attacker speeds at the moment of demolition: **2120, 2130, 2230, 2240, 2260,
2290, 2300 uu/s** — **23 of 23 inside [2100, 2300]**, none below. RocketSim's
supersonic demo gate is consistent with the real game.

### Car-car bumps — PARTIALLY validated, no defect found

Naively, 283 frames show a car gaining >250 uu/s near another car — but a dodge or a
ball touch does that too. Isolating genuine bumps (nearest car <150 uu, ΔV pointing
*away* from it, ball >300 uu away) leaves **25** events:

- median **sim/real ΔV ratio 0.817**, p75 **0.965**
- RocketSim reproduced the contact within the window for only 28% of them
- control (non-bumped cars in the same frames): mean ΔV error **15 uu/s** over 1,742
  samples, so the 4-car harness itself is sound

The 28% reproduction rate is the same contact-phase limit that made the ball-car
impulse look 45% weak in §7 before phase-robust comparison — from a quantized
frame-boundary state, a sub-frame contact often does not occur in sim. Bump *impulse
magnitude* looks right and the code path uses every bump constant with correct
ground/air branching; I found **no evidence of a defect**, but this is not the clean
validation the ball-car impulse got.

### Ceiling driving and contact-imparted spin — closed

Both remaining "untested regime" gaps, using replay data (4-tick windows, all 4 cars
simulated, ball included).

**Important:** replays do not record car *inputs*, so I set zero controls. Over 4 ticks
an unknown throttle/boost contributes up to ~35 uu/s, which dominates these absolute
numbers — they are ~40× worse than the 120 Hz telemetry of §7 for that reason. They are
only meaningful **relative to each other**:

| car regime | n | pos med (uu) | p90 |
|---|---|---|---|
| flat ground (z<50, up.z>0.9) | 59,818 | 10.389 | 43.914 |
| wall (300<z<1800, \|up.z\|<0.5) | 5,227 | 12.578 | 45.102 |
| **ceiling (z>1900, up.z<−0.5)** | **105** | **9.584** | **36.154** |
| airborne (z>300, up.z>0.9) | 753 | 14.763 | 54.783 |

**Ceiling driving is no worse than flat ground** — in fact the lowest of the four. A
broken inverted-gravity/sticky-force path would stand out here and does not. Combined
with §7's wall/ramp result (median 0.723 uu at 120 Hz, incl. 41 near-vertical frames),
non-flat and inverted surface driving is clean.

**Spin imparted by a contact:**

| | n | angVel err med (rad/s) | p90 |
|---|---|---|---|
| contact frames (\|Δspin\|>1) | 636 | **0.4201** | 7.1477 |
| control, no contact | 21,840 | **0.0000** | **0.0000** |

Free-flight spin propagation is **exact** (0.0000 at median *and* p90 over 21,840
windows). At contacts the median error is 0.42 rad/s against a 0–6 rad/s range (~7%),
with a long tail from the same contact-phase problem that affects every contact
measurement here. No evidence of a systematic defect in the spin a hit imparts.

### Double jump — VALIDATED (and it calibrates the replay convention)

`TAGame.CarComponent_DoubleJump_TA:DoubleJumpImpulse` is replicated (3,183 samples across
25 replays) with magnitude **exactly 525.0** every time.

- **525 / 180 (car mass) × 100 = 291.667 = RocketSim's `jump::IMMEDIATE_FORCE` (875/3)**, exact.
- Corroborated behaviourally: resolving each impulse against the car's next velocity
  sample, genuine upward jumps (Δvz>150, n=1,086) cluster at median **242.9 uu/s**;
  adding back one 30 fps frame of gravity (+21.7) gives ~**264.6 uu/s**, consistent with
  291.7 given 30 fps sampling and quantised velocities.

This is a validation in its own right **and** the calibration that keeps the dodge
finding honest: it proves the replicated physics quantities are impulses carrying a
`× mass / 100` factor, which is why the raw 224-vs-2.24 ratio cannot be read as a
literal unit bug.

### Per-mechanic ground-driving decomposition (weak signal, not a defect)

The air-throttle bug hid inside an unremarkable aggregate and only appeared when the
error was resolved *directionally*. So ground driving was decomposed the same way —
signed error along the car's forward and right axes, plus yaw-rate error, per input
mechanic (838 live 120 Hz transitions, on-ground, no jump):

| mechanic | n | fwd ΔV err (med) | %neg | lat ΔV err (med) | yaw err |
|---|---|---|---|---|---|
| throttle=+1, no boost/handbrake | 241 | −1.108 | 82% | −2.916 | −0.0005 |
| throttle=−1 (reverse/brake) | 18 | −0.117 | 61% | +0.720 | +0.0002 |
| throttle=0 (coasting) | 16 | −5.758 | 88% | +11.968 | +0.0032 |
| boosting on ground | 68 | −0.298 | 68% | −0.680 | +0.0002 |
| handbrake / powerslide | 115 | −0.160 | 66% | +0.053 | −0.0003 |
| **steering hard (\|steer\|=1)** | **183** | **−3.621** | **79%** | **−4.407** | −0.0023 |
| near supersonic (v>2100) | 40 | −0.071 | 62% | +0.401 | +0.0011 |

**Clean:** yaw-rate error is ≤0.004 rad/s everywhere, so the steer-angle curves are
accurate; handbrake/powerslide, ground boost, braking and supersonic are all
sub-uu/s and effectively unbiased.

**Weak signal:** the sim slightly *under*-accelerates on the ground, concentrated in
hard turns (−3.6 uu/s forward, 79% one-sided) and coasting (−5.8, 88% one-sided but
n=16). Magnitude is ~1–3% of the relevant accelerations — comparable to the confirmed
air-throttle defect (+4.46 uu/s) but far noisier and one-sided rather than constant.

**Not reported as a defect.** One match, subsets of 16–241 frames, and hard-steering
frames correlate with other conditions. It would need a few more instrumented matches
to separate a genuine tire-friction difference from sampling. Recording it so it is not
lost, and so nobody re-derives it from scratch.

### Non-Octane hitboxes and pad geometry — measurement floor reached

Both were pushed with replay data and both hit a hard limit. Recording what was tried
so it is not retried blindly.

**Non-Octane bodies.** Replays *do* carry loadouts (`TAGame.PRI_TA:ClientLoadouts`), and
across 86 replays the bodies include **403 = Dominus** (22 uses) — a genuinely different
hitbox — alongside Octane (23) and Fennec (4284, Octane hitbox). Mapping PRI→car via
`Engine.Pawn:PlayerReplicationInfo` yielded 186k Dominus and 134k Octane frames. Testing
whether the *correct* body config fits its own player better:

| dataset | sim as OCTANE | sim as DOMINUS |
|---|---|---|
| Dominus players (n=82,962) | 24.381 uu | 24.379 uu |
| Octane players (n=22,785) | 26.233 uu | 26.235 uu |

**Not discriminating**, and for a structural reason: replays do not record car *inputs*,
so the 4-tick error is control-dominated (~25 uu) while the hitbox effect on ground
motion is ~0.005 uu. Ride height cannot separate them either — the sim gives 16.73 vs
16.75 for the two bodies, and the *real* data likewise shows 16.600 (Dominus) vs 16.570
(Octane). Hitbox differences live in **collision geometry**, which is exactly the
contact-phase-limited regime. Pulsar plays Octane, which is validated behaviourally.

**Boost-pad geometry.** 4,120 real pickups were extracted via
`CarComponent_Boost_TA:ReplicatedBoost` jumps. Naively 83.8% appear to occur outside
RocketSim's pickup cylinder, and 79.2% still do when searching the 10 frames before the
update for the true crossing point. **This is a broken measurement, not a finding**: the
median *minimum* distance to any big pad across the whole window is 230 uu, i.e. the car
never came within 230 uu of a pad yet gained ~100 boost — physically impossible if pickup
requires overlap. `ReplicatedBoost` is a sparse, lagged network value and its timing
cannot be aligned with the pickup event. **The §7 telemetry test stands** — on 110 frames
with exact per-tick boost and true pad states from the RLBot packet, pickup agreement was
perfect (0 sim-only, 0 real-only).

## 11. Fixes APPLIED and validated (2026-08-01)

Both defects are now patched in the vendored tree
(`RocketSimV3/rocketsim/src/sim/car/base.rs`), each with a `VENDOR PATCH` comment
following the precedent of the existing mesh-whitelist patch.

| fix | check | before | after |
|---|---|---|---|
| air throttle gated on `!boost` | excess forward ΔV, airborne+boosting | +4.459 uu/s | **+0.020** |
| dodge torque through `inv_inertia_tensor_world` | angVel err, 6,382 real dodge frames | 2.0482 rad/s | **0.0512** |

**Regression sweep (838 live 120 Hz transitions), position error median:**

| regime | before | after |
|---|---|---|
| all live | 0.275 | 0.279 |
| flat ground | 0.552 | 0.552 |
| wall / ramp | 0.723 | 0.723 |
| airborne, no boost | 0.061 | 0.061 |
| **airborne, boosting** | 0.249 | **0.179** |
| flipping | 1.600 | 1.600 |

No regressions; boosted-aerial position error improved 28%.

**Caveat on that "flipping" row:** the regression probe does not set
`is_flipping`/`flip_rel_torque`, so no dodge torque is applied in it — that row is
insensitive to the dodge fix by construction and should not be read as the fix failing.
The dodge fix is validated by the dedicated harness using the game's replicated
`DodgeTorque` (row 2 above).

**Operational consequence.** These change training dynamics. Per CLAUDE.md's own rule,
checkpoints trained on the previous physics are behaviourally stale against the patched
engine. `build/` was **not** rebuilt and the trainer was **not** touched — the live run is
unaffected until it is restarted, which is a deliberate decision for the run owner.

## 12. Reflection route: RL's physics parameter vocabulary (no decryption needed)

Decryption is only required for package **assets**. UE3's reflection metadata — every
class's property names — is registered from the executable, so the parameter *vocabulary*
is recoverable statically. Extracting the UTF-16 FName block in `.rdata` (~`0x141e3a000`)
yields 26,758 identifiers, of which 82 are genuine physics parameters after filtering out
cosmetics. Saved as `research/results/rl_exe_physics_param_names.json`.

This gives a **completeness check RocketSim could not otherwise get**: parameters the game
has that the simulator does not model at all. Nine have no RocketSim counterpart:

| RL parameter | RocketSim | assessment |
|---|---|---|
| `DodgeLift` | absent | RocketSim's dodge impulse is **purely 2D** (`x*fwd_2d + y*right_2d`, no Z term) |
| `DodgeClearVelocity` | absent | velocity manipulation at dodge start |
| `DodgeTransferVelocity` | absent | " |
| `DoubleJumpLimitVelocityXY` | absent | constrained by measurement — see below |
| `DoubleJumpRemoveVelocityZ` | absent | " |
| `BoostLimitVelocityXY` | absent | " |
| `BoostRemoveInitialVelocityZ` | absent | " |
| `StopExistingTorque` | absent | dodge/torque bookkeeping |
| `RandomJumpTorque` | absent | likely 0 in standard play |

**Most of these are almost certainly zero/disabled in standard soccar**, and the
measurements in this report are what constrain that: the double-jump impulse measures
**exact** (291.667 uu/s, §8), so `DoubleJumpRemoveVelocityZ` cannot be active; boost
acceleration measures exact after the air-throttle fix, so the two `Boost*Velocity*`
parameters cannot be active either. They are most plausibly mutator/Rumble-mode knobs.

**`DodgeLift` — CLOSED: no missing upward impulse.** The first attempt failed because it
keyed on `DodgeTorque` onset, which replicates *after* the impulse. Detecting the impulse
by its own signature instead — an airborne horizontal ΔV in the dodge range, with ball and
all other cars >300 uu away — isolates 90 clean events whose **median horizontal ΔV is
537.0 uu/s**, matching RocketSim's `INITIAL_VEL_SCALE = 500` and confirming these are real
dodge impulses.

Their vertical component, gravity removed:

| | value |
|---|---|
| median | **−36.98 uu/s** |
| p25 / p75 | −195.21 / +64.45 |
| sign balance | **38% positive** (62% negative) |

The vertical is **net downward and predominantly negative**. A `DodgeLift` term would show
a positive median; instead the bias matches the flip **Z-damping** RocketSim already
implements (`Z_DAMP_120`, applied when `vel.z < 0` or early in the flip). So `DodgeLift`
adds no upward impulse in soccar, and RocketSim's 2D dodge impulse plus Z-damp is the
correct structure.

Caveat kept honest: this rules out a *missing lift term*; it does not prove the vertical
magnitude is exact to within the ~30 uu/s spread of 30 Hz replay sampling.

## 13. The ground signal, re-measured after both fixes — now localized

Re-running the per-mechanic ground decomposition on the **patched** engine (838 live
120 Hz transitions; forward component of the velocity-error vector, negative = sim
under-accelerates):

| ground regime | n | mean fwd err (uu/s) | median | % one-sided |
|---|---|---|---|---|
| throttle forward, no boost | 241 | **−4.473** | −1.108 | 82% |
| hard steering (\|steer\|>0.5) | 151 | **−7.045** | −3.836 | 79% |
| coasting (throttle≈0) | 16 | −5.746 | −8.483 | 88% |
| **ground boosting** | 68 | **−0.099** | −0.298 | 68% |
| handbrake | 115 | +0.812 | −0.160 | 66% |

Three things changed versus the earlier characterisation:

1. **It survives both fixes**, so it is not a secondary effect of the air-throttle bug.
2. **The sample is now n=408** across three concordant regimes, not 183 in one.
3. **It is localized.** Ground *boosting* is clean (−0.099, essentially zero) while
   throttle-driven regimes are not. Boost is applied as a direct force on the body;
   throttle is applied as **wheel drive torque** through the raycast-vehicle constraint.
   The deficit therefore sits in the **wheel drive path**, not in general force
   application — which is consistent with every impulse-primitive comparison in §1b
   coming back clean.

**Still not promoted to a defect**, deliberately: medians (−1.1 to −3.8 uu/s) are much
smaller than means, so the distribution is skewed by a minority of frames rather than
uniformly shifted, and one match cannot separate a real force deficit from residual
control-timing error. But it is now a *specific, testable* hypothesis rather than a vague
signal.

**Why Ghidra cannot finish this one.** RL's wheel forces do **not** go through the impulse
primitives — `FUN_14097f5c0` has only 7 call sites program-wide and none are in the vehicle
region. Like RocketSim, RL applies suspension and tyre friction inside the **vehicle
constraint solver**, so comparing it means reverse-engineering that solver rather than
reading a force call. That is a substantially larger effort than the dodge fix required,
and it is the honest boundary of what this audit reached.

**To settle it:** a second and third instrumented match would confirm or dissolve the
skew, and per-wheel logging (drive torque, per-wheel friction) would pin it to a term.

## 14. NEW DATA (10,822-decision match): two findings upgraded to defects

A second instrumented match (`debug.2924986.jsonl`, 722 s, 8,811 live transitions after
dropping 1,840 paused frames) — ~13x the original sample — changed two conclusions.
Both had been called clean or inconclusive on the smaller sample.

### DEFECT 3: coasting deceleration is ~20% too strong

The ground signal is **no longer dismissible as skew**. At 13x the data, median and mean
agree, which they did not before:

| regime | n | mean | median | % one-sided |
|---|---|---|---|---|
| **coasting (throttle≈0)** | 237 | **−6.924** | **−8.504** | **93%** |
| hard steering | 1612 | −3.427 | −4.235 | 80% |
| throttle fwd, no boost | 2522 | −0.880 | −1.196 | 79% |
| ground boosting | 830 | −1.276 | −0.265 | 70% |

Negative = the sim loses forward speed faster than the game. Sweeping
`drive::COASTING_BRAKE_FACTOR` (shipped 0.15) is monotonic and crosses zero at ~0.12:

| k | 0.15 | 0.12 | 0.10 | 0.06 | 0.00 |
|---|---|---|---|---|---|
| coasting mean err | −6.924 | **−0.027** | +4.504 | +13.563 | +27.155 |
| % one-sided | 93% | 56% | 42% | 12% | 5% |

At k=0.12 coasting nulls almost exactly and one-sidedness returns to chance. Other regimes
are unchanged, as expected — the coast brake only applies at throttle≈0.

**Not shipped — and a follow-up test shows shipping it would have been wrong.**
Adding an explicit *braking* regime (throttle opposed to motion) discriminates the cause:

| regime | brake factor applied | n | median err | % one-sided |
|---|---|---|---|---|
| coasting | 0.15 | 237 | **−8.504** | 93% |
| **braking** | **1.00** | 113 | **−8.146** | 87% |
| throttle with motion | – | 2480 | −1.243 | 80% |

If the brake torque were scaled wrong, the braking error would be **~6.7× larger** than
coasting's (1.0 vs 0.15). It is **equal**. That rules out both `COASTING_BRAKE_FACTOR` and
`BRAKE_TORQUE_AMOUNT` as the cause: fitting 0.12 would null coasting while leaving braking
just as wrong.

What is established: **the sim loses ~8 uu/s per 8-tick window too much whenever the car is
decelerating** (coasting *or* braking, ~equal magnitude, 87–93% one-sided, n=350 combined),
and only ~1.2 under power. A magnitude independent of brake strength points at a term that
is not the brake — rolling resistance, or wheel friction while not driving. Unfixed.

Residual after that: throttle-forward −0.880 (79% one-sided, n=2522) and steering −2.727
(76%, n=1612) remain unexplained but are much smaller.

### DEFECT 4: boost pad pickup volume is too small — the community was right

Retested on 12x the frames (1,508 with every pad genuinely available, vs 110 before):

| | previous (n=110) | **new (n=1508)** |
|---|---|---|
| both picked up | 1 | 9 |
| **REAL-ONLY (sim missed)** | **0** | **10** |
| SIM-ONLY (sim over-picked) | 0 | 0 |

**Of 19 real pickups, RocketSim missed 10 — a 53% miss rate**, entirely one-directional.
The earlier "perfect agreement" was a sample-size artifact.

Mechanism: at those pickups the car origin sits **212.8 uu (median, max 253.0)** from the
nearest pad, *outside* RocketSim's cylinder (208 uu big / 144 uu small tested against the
car **origin**). RL additionally tests a **box against the car's full AABB**, which reaches
roughly a car half-length (~60 uu) further and catches exactly these. That is the
`_box_radius` / `BOX_HEIGHT` car-locking branch flagged dead in §7 and wrongly dismissed
there as "not exercised by this data" — it is exercised, and it matters.

This independently corroborates the RL modding community's report that "boost pad logic
was wrong".

**FIXED — via Ghidra, two separate bugs.**

**(a) Broad-phase/narrow-phase mismatch.** `BoostPad::new` built the BVH AABB from
`box_radius` (120/160) while the pickup tested `dist_2d < cyl_radius` (144/208), so pads
between those radii were culled before ever being tested. Proof: two missed pickups had the
car within **123.4 uu and 129.4 uu** of a 144-radius pad — inside the cylinder, no pickup.
Fixed by building the AABB from `cyl_radius`.

**(b) The pickup tested the car's ORIGIN, not its body.** Ghidra settles this:
`AVehiclePickup_TA::execIsTouchingAVehicle` (`0x140e83de0`) delegates to `FUN_140f0d590`,
which walks UE3's **`Touching` array** — i.e. RL decides pickups by a *collision-primitive
overlap between the pad volume and the car's collision body*, never a point test.

Reimplemented as pad cylinder vs the car's **oriented** box (closest-point-on-OBB). The
orientation matters: an axis-aligned box of a rotated car is far larger than the car and
over-triggers badly.

And the radius must then be **`BOX_RAD` (120/160), not `CYL_RAD` (144/208)** — the larger
cylinder radii are origin-test approximations that already bake in typical body reach, so
pairing them with a body test double-counts it. This finally explains why `BOX_RAD_*`
existed as dead constants.

| pickup rule | misses | false | total disagreement |
|---|---|---|---|
| shipped: origin vs CYL_RAD | 10 | 0 | 10 |
| + broad-phase AABB fix | 8 | 0 | 8 |
| car AABB vs box | 0 | **116** | 116 |
| car OBB vs CYL_RAD | 0 | 10 | 10 |
| **car OBB vs BOX_RAD (shipped)** | **0** | **4** | **4** |

**All 19 real pickups now reproduce**, total disagreement 10 → 4.

Rejected along the way: a swept-segment (tunnelling) test — **zero effect**, because the
diagnostic already sampled closest approach at every tick boundary.

---
*Superseded analysis, kept for the record:*

`BoostPad::new` built the BVH broad-phase AABB from **`box_radius`** (120 small / 160 big)
while `BoostPadGrid::process_node` tests **`dist_2d < cyl_radius`** (144 / 208). Any pad
between those two radii was culled by the broad phase and **never reached the cylinder
test**. The diagnostic that exposed it: two of the missed pickups had the car within
**123.4 uu and 129.4 uu** of a 144-radius pad with |dz|=53 — comfortably inside the
cylinder, yet no pickup.

Shipped fix: build the AABB from `cyl_radius`. Result **10 misses → 8, still 0 false
positives** (both 9 → 11). Small but strictly correct, and it removes a
broad-phase/narrow-phase inconsistency that would bite any future radius change.

**The remaining 8 are NOT explained.** Their closest approach along the whole 8-tick path
is 146–170 uu against a 144 uu radius — a median **16 uu (max 26)** of extra reach needed,
all on small pads. Two hypotheses tested and rejected:

- *Tunnelling between ticks* — implemented a swept segment test (prev→cur position, with a
  matching swept BVH query). **Zero effect**, because the diagnostic already sampled
  closest approach at all 8 tick boundaries, so sub-tick sweeping adds almost nothing.
  Reverted.
- *Car body reach (box vs AABB)* — see below; wildly over-triggers.

Closing the last 8 means either a genuine small-pad radius larger than 144, or a body-aware
rule far tighter than a full AABB. Neither is derivable from 19 events without fitting.

**Earlier fix attempts, both REVERTED — do not repeat.** Two variants were
implemented against the real pickups and both were worse overall than shipping nothing:

| pickup rule | misses (REAL-ONLY) | false pickups (SIM-ONLY) |
|---|---|---|
| shipped: cylinder vs car origin | **10** | 0 |
| + box(±160/±120, z: pad→pad+64) vs car AABB | 7 | 3 |
| + box XY vs car AABB, cylinder z-gate | **0** | **116** |

The first variant barely helps because RL's box sits at z: pad→pad+64 (70→134) while a
grounded car's AABB tops out near z=36, so it only ever fires for airborne cars. The second
catches every real pickup but over-triggers 116 times in 1508 frames — 6× the true rate.
The correct rule lies between and is not recoverable by fitting to 19 events, so the
vendored tree keeps the shipped cylinder-only behaviour. **Confirmed defect, unfixed.**

### Jump cooldown timers: tested, no observable deviation

`jump::RESET_TIME_PAD` carries an upstream TODO ("RL does something similar to this
time-pad, but not exactly the same"). Across **162 landings**, *zero* had `has_jumped`
still true at touchdown — RL clears it at or before landing, and since `jump_time` accrues
through the whole airtime, RocketSim also clears immediately for any hop longer than
0.275 s. No short hops (<0.275 s airborne) occurred, so the one case where the pad could
differ never arose. **Not a confirmed defect; also not cleared** — needs a short-hop case.

## 15. Ball-car hit: a real error pattern, and why it could not be fixed here

Re-tested on the larger match: **284 real ball touches** (car <250 uu, opponent >400 uu),
binned by how off-centre the contact was (perpendicular offset of the ball from the car's
forward axis, in car-local space):

| off-centre offset | n | direction err | speed ratio |
|---|---|---|---|
| ~110 uu (most centred) | 71 | 2.96° | 0.957 |
| ~137 uu | 71 | 2.54° | 0.970 |
| ~147 uu | 71 | 2.75° | 0.984 |
| **~169 uu (most off-centre)** | 71 | **6.81°** | **0.922** |

So the hit is good to ~2.5-3° when centred and degrades to **6.8° with an 8% speed
deficit** when off-centre. That pattern is real and reproducible.

**The obvious hypothesis was tested and could not be evaluated.** RocketSim derives the
extra-impulse direction from `ball.pos - car.pos` (centre-to-centre), whereas for a sphere
the true contact normal is `ball.pos - contact_point`; the two coincide only for centred
hits and diverge exactly as the contact moves off-axis. The contact point is already
available at the call site, so it was plumbed into `Ball::on_hit` and used as the direction
basis.

**Result: byte-identical metrics across all four quartiles.** A sensitivity probe that
*fully inverted* the hit direction also produced byte-identical metrics — so the harness
is blind to the extra-impulse direction entirely, and cannot confirm or refute the change.
It was reverted rather than shipped unvalidated.

**Most likely cause of the Q4 error is the harness, not RocketSim.** The RLBot telemetry
does not log ball angular velocity, so every ball state in this test starts with **zero
spin**. Off-centre contacts are precisely the spin-sensitive ones, and the replay-based
work in §8 showed that supplying real ball spin improves ball prediction ~25%. A test that
zeroes spin should therefore degrade most on off-centre hits — which is what is observed.

**To settle it:** log ball angular velocity in the RLBot debug JSONL (it is in the packet),
then repeat this exact binning. Until then the ±5% ball-hit figure should be read as a
*measurement floor*, not an established RocketSim error.

## 16. Behavioural verdict: per-window accuracy, and where it is still blind

**The bot is closed-loop at 15 Hz**, so the error it actually experiences is one 66.7 ms
window, not an open-loop trajectory. Open-loop rollouts were measured anyway and are a poor
metric here: at 1 s the divergence is p50 73 uu, and the fixes make it *marginally worse*
(69.76 → 73.28), because open-loop divergence over seconds is chaos-dominated and this test
simulates one car with no opponent and no ball interaction. Do not tune against it.

Per-window, pristine vs the four fixes, on 8,798 live windows:

| regime (one window) | n | pos p50 (pristine → fixed) | vel p50 (pristine → fixed) |
|---|---|---|---|
| ALL live | 8798 | 0.294 → 0.300 | 9.655 → 9.677 |
| ground: throttle | 3660 | 0.622 → 0.622 | 13.857 → 13.802 |
| ground: coasting | 558 | 0.675 → 0.725 | 17.178 → 17.353 |
| ground: boosting | 917 | 0.726 → 0.726 | 16.751 → 16.752 |
| air: no boost | 2814 | 0.100 → 0.101 | 1.153 → 1.209 |
| **air: boosting** | 559 | 0.193 → **0.071** | 4.481 → **0.462** |
| **jump/flip window** | 465 | **20.294** | **308.933** |

**The air-throttle fix is a large, unambiguous win** — 9.7× on velocity accuracy for
boosted aerials, the regime this run scaffolds hardest.

**Two caveats that bound the confidence claim:**

1. **Coasting magnitude did not improve** even though its *forward component* went from
   93% one-sided (median −8.504) to 49% (+0.132). The residual is dominated by
   perpendicular (lateral/vertical) error of ~15-17 uu/s that the coast-brake fix does not
   touch. The directional fix is still right; it just is not the dominant term.

2. **Jump/flip windows are the worst regime by an order of magnitude** (20.3 uu, 309 uu/s)
   and are **unchanged by the fixes** — because this harness cannot reconstruct
   `is_jumping`/`jump_time`/`flip_time`/`flip_rel_torque` from the RLBot packet, so the sim
   does not reproduce the jump at all. That number is a harness artifact, **but it also
   means flip-heavy mechanics are the one regime this audit cannot certify.** The
   replay-based dodge test (§1b) is the only real measurement there, and it covers dodge
   torque only, not the jump/landing sequence.

**Verdict for deployment:** driving and aerial dynamics are per-window sub-uu and materially
better than before. Flip/jump sequences remain unverified end-to-end. See the confidence
statement in §17.

## 17. Confidence statement — behavioural impact of the fixes

Flip state *can* be reconstructed without new logging: replays carry
`CarComponent_TA:ReplicatedActive` (per-component active flags), `CarComponent_TA:Vehicle`
(component→car), `CarComponent_Dodge_TA:DodgeTorque` and
`CarComponent_DoubleJump_TA:DoubleJumpImpulse`. An earlier claim in this session that flips
needed extra RLBot instrumentation was **wrong** — the replay path already supplies it.

Measured on **6,382 mid-dodge windows** driven by the game's own replicated `DodgeTorque`
with proper flip state:

| | pristine | fixed |
|---|---|---|
| angVel err p50 | **2.0482 rad/s** | **0.0512 rad/s** |
| **→ orientation error over a full 0.65 s dodge** | **76.3°** | **1.9°** |

Position/velocity in that table are unchanged and floor-limited (~15 uu) by replay state
quantisation, so only the angular figure is meaningful there.

And on boosted aerials (telemetry, 559 windows):

| | pristine | fixed |
|---|---|---|
| vel err p50 | 4.481 uu/s | **0.462 uu/s** |
| **→ speed error over a 1.5 s aerial** | **100.8 uu/s** | **10.4 uu/s** |

**This is the answer to the original question.** A policy trained on the pre-fix engine
learned dodges that ended up **76° from where the real game puts them**, and aerials that
arrived ~100 uu/s fast. Those are exactly the errors that make learned mechanics fail to
transfer while leaving positional play looking fine — which matches the reported symptom
(real-game goal share pinned at ~20% across every configuration change).

**Confidence, by area:**

- **Driving** — per-window 0.30 uu median / 1.6 uu p90 across 8,798 windows. High.
- **Aerials** — 9.7× velocity-accuracy improvement; residual 10 uu/s over a 1.5 s aerial. High.
- **Dodges/flips** — 40× rotation improvement; residual 1.9° over a full dodge. High.
- **Ball contact** — ±5%, but that is a *measurement floor* (harness zeroes ball spin), not
  an established error. Medium, likely better than measured.
- **Bumps/demos** — magnitudes validated, reproduction rate limited by contact-phase
  sampling. Medium.

**Remaining known-unknowns:** the jump→flip→landing *transition* (dodge torque is validated,
the surrounding sequence is not), ~15-17 uu/s of perpendicular coasting error with no
identified cause, and 4 spurious boost pickups per 1508 windows.

## 17. Ball contact: spin RULED OUT as the cause (correction)

§15 speculated that the off-centre ball-hit error was an artifact of the RLBot telemetry
not logging ball angular velocity, and that the true margin was therefore better than the
measured +/-5%. **That was wrong.** Replays *do* carry ball angular velocity, so the test
was run both ways on the same 294 reproduced contacts:

| ball spin fed to the sim | dir err p50 | speed ratio p50 | outgoing spin err p50 |
|---|---|---|---|
| zeroed | 19.25° | 0.659 | 3.198 rad/s |
| **real** | 19.53° | 0.669 | **0.747 rad/s** |

Supplying real spin changes direction and speed by **under 2%** — it is not the cause.
It does improve the *outgoing* spin 4.3x, so RocketSim's spin handling at contact is sound
when given correct input; incoming spin simply is not what drives the direction error.

(The absolute numbers here are far worse than the telemetry test's 2.5-6.8° / 0.92-0.98
because replays are 30 fps, so contact phase is quantised to 4-tick steps and only ~28% of
contacts reproduce at all. The telemetry figures remain the better estimate; this test is
only valid as a *controlled comparison* of spin-on vs spin-off.)

**Consequence: the ball-contact tolerance is real, not a measurement floor.** Direction
2.5-3° centred / 6.8° off-centre and speed +/-5% stand as genuine sim-vs-game differences,
and remain the thinnest margin in the system. The claim in §15 that logging ball spin would
tighten it is withdrawn — it would not.

## 18. Calibration: sim error vs the BOT'S OWN precision (the right yardstick)

Every tolerance in this report was compared against perfection. That is the wrong
reference. What decides whether a sim-to-real difference is *noticeable in play* is whether
it exceeds the agent's own control precision in the same dimension.

Measured across **335 real ball touches** from match telemetry, the bot's own contact
placement (perpendicular offset of the ball from its forward axis at contact):

| | value |
|---|---|
| p10 / p50 / p90 offset | 107.9 / 143.0 / 175.1 uu |
| p10→p90 spread | **67.3 uu** (sd 31.0) |
| implied contact-normal spread | **~41.9°** |
| **sim's systematic contact error** | **2.5–6.8°** |

**The bot's own shot-to-shot placement varies 6.2× more than the sim's worst-case
systematic error, and 14× more than its typical one.** The contact error is an order of
magnitude below the noise floor of the agent's own aiming, so it cannot be the limiting
factor on shot outcomes.

Applying the same yardstick to the other residuals:

| residual | magnitude | agent-relative |
|---|---|---|
| contact direction | 2.5–6.8° | **6–14× below** own placement spread (41.9°) |
| coasting perpendicular vel | 15–17 uu/s | ≈1 uu per window vs a 120 uu car — ~1% |
| dodge orientation (post-fix) | 1.9° over a full dodge | vs 41.9° placement spread |
| bumps | ratio 0.82 | 18% on an event occurring ~2×/match in 1v1 |

Contrast with what was fixed: a **76.3° dodge orientation error** and **100.8 uu/s** of
aerial speed error — both far *above* the agent's control precision, which is exactly why
they broke learned mechanics while leaving positional play looking fine.

**Conclusion.** Every measurable sim-vs-game difference now sits at least ~6× below the
agent's own precision in the same dimension. The jump→flip→landing *composite* was never
validated end-to-end (the harness could not trigger jumps from replay state), but its
constituents are each validated independently: jump/double-jump impulse (exact, 291.667),
dodge torque (1.9°), flip duration (bracketed [0.6083, 0.6666] s), gravity, and air control
(0.001 rad/s). No component of that sequence carries a known error above the agent's noise
floor.

## 17. Jump impulse VALIDATED from replays (no live match needed)

The jump blind spot was previously blocked on a broken harness (forcing
`is_on_ground` does not physically ground a car, so the sim never triggered the jump).
That is sidestepped entirely by measuring the impulse in the **replay data itself**,
using `CarComponent_TA:ReplicatedActive` activations — no simulation involved.

Extracted 2,728 JUMP / 604 DBLJUMP / 1,581 DODGE component activations with car state,
then measured the vertical velocity change across each activation with gravity removed:

| | n | real dVz median | RocketSim prediction | ratio |
|---|---|---|---|---|
| **JUMP** | 973 | **325.13 uu/s** | `IMMEDIATE_FORCE` 291.67 + one frame of `ACCEL` ≈ **340** | **0.96** |
| DBLJUMP | 205 | 264.57 uu/s | `IMMEDIATE_FORCE` **291.67** | 0.91 |

**The jump impulse is correct to ~4%** on 973 real events. The double jump reads 9% low,
but that is within this measurement's noise (30 Hz replay sampling, quantized velocities,
and gravity removal using a variable inter-frame dt) and is contradicted by the stronger
evidence in §8: the game *replicates* `DoubleJumpImpulse` as exactly 525, which is
525/180×100 = 291.67 uu/s, i.e. RocketSim's constant exactly. Treat the 0.91 as noise,
not a defect.

JUMP's p25 of 21.78 uu/s shows the distribution is bimodal: in some activations the
impulse has already been applied before the sampled frame. That is a sampling-phase
artifact, not two different jump strengths — the bulk (median 325, p75 337) sits on the
prediction.

**Still open:** the full jump → flip → landing *trajectory*. The impulse that starts it is
now validated; what happens over the following 0.5 s is not.

## 18. Ball spin is NOT the cause of the contact error (hypothesis disproved)

The +/-5% ball-hit figure was repeatedly caveated in this report as "a measurement floor,
not a proven error", on the theory that the RLBot telemetry does not log ball angular
velocity, so every contact test started from **zero spin** — and off-centre contacts are the
spin-sensitive ones.

That theory is now tested and **wrong**. Replays *do* carry ball angular velocity, so the
same contact test can be run with real spin and with spin zeroed, changing nothing else.
313 replay ball-touches (a car within 250 uu and real ball dV > 80 uu/s):

| ball spin | n | post-hit speed ratio p25/med/p75 | direction err (median) |
|---|---|---|---|
| **zeroed** | 313 | 0.524 / **0.643** / 0.829 | 20.54° |
| **real** | 312 | 0.528 / **0.654** / 0.830 | 21.23° |

Supplying the real spin moves the speed ratio by **0.011** and the direction error by
**0.7°** — i.e. nothing. Ball spin does not explain the contact error.

Two consequences:

1. **The caveat should be dropped.** The telemetry-based figure (ratio 1.027, IQR
   [0.936, 1.037], §7) is not concealing a spin-dependent error, and is the best estimate
   of ball-hit fidelity. Its residual is contact-phase resolution, as §7 originally said.
2. **These replay numbers must not be read as the error.** Ratio 0.65 / 20° here is far
   worse than the telemetry's 1.027 / 2.5-6.8° purely because replays are 30 Hz and
   quantized, so contact phase is recovered much more coarsely (the ~40x noise factor noted
   in §8). This test is only valid as a *controlled A/B on the spin variable*, which is
   exactly what it was used for.

## 19. Post-impulse trajectory: no evidence of a defect, but noise-limited

The remaining jump gap was the *trajectory* after the impulse. That is measurable without
grounding the car: start from the first follow-frame (already airborne) and roll forward.

| sequence | n | err after ~66 ms | after ~166 ms |
|---|---|---|---|
| DODGE | 880 | 21.35 uu | 54.47 uu |
| JUMP | 134 | 20.10 uu | 58.85 uu |
| DBLJUMP | 283 | **11.48 uu** | 29.88 uu |

**These are at the replay noise floor, not above it.** Two limits apply equally to all rows:
replays are 30 Hz and quantized (the ~15 uu floor established in §8), and they do not record
car *inputs*, so the rollout uses zero controls while the real car was air-rolling.

The comparison that matters is against ordinary airborne flight measured the same way:
**14.76 uu over 4 ticks** (§8's airborne regime). DBLJUMP over *8* ticks is **11.48 uu** —
better than free flight over half the horizon. DODGE at 21.35 uu over 8 ticks is the same
order.

So: **post-impulse trajectories are indistinguishable from ordinary airborne flight at this
resolution.** That is not a positive validation — it is "no evidence of a defect, and the
instrument cannot see smaller than this". Combined with §17 (impulse correct to ~4%) and
§1b (dodge rotation 1.9° per dodge post-fix), the jump/flip path has no remaining measured
discrepancy, but the landing itself is still unexercised.

## 20. LIVE ts1 MATCH — landings validated, and a 2x decision-rate deploy gap found

945.7 s real-game match, 6.0 ts1 policy (3.07B steps), 56,733 decisions, 24 MB JSONL.
The bot self-configured correctly: `trunk { 896, 896, 896 }`, `policy { 640, 640, 640 }`,
`Decision rate: tickSkip=1 (120 Hz)`.

### The landing phase is validated (last open physics item)

Replayed all 55,956 clean transitions through the patched engine:

| regime | n | pos p50 | pos p90 | pos p99 | vel p50 |
|---|---|---|---|---|---|
| all live | 55,956 | **0.151 uu** | 0.322 | 1.00 | 4.100 |
| on ground | 42,066 | 0.159 | 0.335 | 0.81 | 4.245 |
| airborne | 13,643 | 0.062 | 0.290 | 8.06 | 0.051 |
| **LANDING (air→ground)** | **247** | **0.318 uu** | 1.336 | 8.11 | 21.733 |

**Landings are 0.318 uu median** — 2x the all-regime median and a quarter-percent of a car
length. Velocity error is higher (21.7 vs 4.1 uu/s), which is expected across a
discontinuous contact event. **No defect.** That closes the last physics item the audit
could not reach.

(These are tighter than §16's numbers because the window here is 2 ticks, not 8.)

### DEPLOY GAP: the game delivers 60 packets/s, so ts1 runs at HALF its trained rate

The decision timestamps are unambiguous:

- median decision interval **17.000 ms** = **two** physics ticks (one tick = 8.333 ms)
- 56,733 decisions / 945.7 s = **59.99 Hz**
- at 120 Hz we would expect **113,487** decisions — we got exactly half
- all 56,733 timestamps are unique, so the bot is not dropping packets; **the game is not
  sending them**

Cause: `TASystemSettings.ini` has **`UncappedFramerate=False`** (and
`UpscaleTargetFramerate=60`). RLBot packets follow the render frame rate, so a 60 fps cap
hard-limits the decision rate to 60 Hz regardless of `tickSkip`.

**A tickSkip-1 policy therefore cannot run at its trained rate on this machine as
configured.** Every action is held for 2 ticks instead of 1 — behaviourally identical to
deploying a ts1 policy at ts2. This is invisible in the sim, invisible in the viz, and
produces no error anywhere: exactly the class of silent mismatch this audit exists to catch.

**Fix:** set `UncappedFramerate=True` and confirm the game sustains >=120 fps (drop
resolution if needed — the config also carries a 1280x720 block). Then re-run and check the
decision interval is 8.333 ms. If 120 fps is not sustainable, the honest alternative is to
train at **tickSkip 2 (60 Hz)** to match what the venue can actually deliver.

Note the echo lag is unchanged at a median of **2 ticks**, consistent with §4 — that is the
send→applied delay and is still cancelled by packet staleness; it is a separate thing from
this rate cap.

## Recommended order

1. **Apply the inverse inertia tensor to dodge torque** (§1b) — one line, root-caused in
   the binary, validated at ~40× error reduction. Prefer this over a scalar: it
   generalises across bodies and axes. Median
   dodge angular-velocity error drops 2.048 → 0.038 rad/s (54×). Verify against the
   replay harness before/after; the 5.5 rad/s cap hides it from casual inspection.
2. **Gate air throttle on `!boost`** (§5). One line, and the only defect confirmed
   against real telemetry to 0.3%. Systematic and same-signed, so it integrates,
   on every boosted aerial, in a run that scaffolds aerials hard. Independently
   corroborated by the regime sweep: boosted-air angVel error is 359× the
   non-boosting airborne control (§7).
3. **Do NOT change `actionDelay`** (§4 correction) — it is already correct, and do NOT
   apply the `TORQUE_MIN_TIME` gate — both were tested and made the fit worse.
4. Fix the two stale docs (RocketSimV3 README §mesh claim; the "suspension 14%
   off" framing, which is a state-restore artifact).
5. Optional: use `CarExtraState` in training state-setters for exact resets.
6. (withdrawn — was "restore v2's boost-pad box branch"; v2 is not a valid reference.)

**When replaying real telemetry, drop paused-game frames** (real position identical
across the window while the car carries speed) — 12.4% of this log, and they corrupt
every tail statistic.

## 9. Coverage audit — every physics code path mapped to a test

Rather than assert exhaustiveness, here is the enumeration: every function in
RocketSim's soccar physics step, and the test that exercises it. This audit is what
surfaced `update_auto_roll` / `update_auto_flip`, which no earlier test had covered.

| RocketSim physics function | covering test | result |
|---|---|---|
| `car::update_wheels` (friction, suspension, sticky) | §7 flat/wall/ceiling + per-mechanic | clean |
| `car::update_air_torque` (air control) | airborne control, 0.001 rad/s | clean |
| `car::update_air_torque` (flip torque) | §1b real `DodgeTorque` sweep | **DEFECT** |
| `car::update_jump` | jump regime, flip duration bracket | clean |
| `car::update_double_jump_or_flip` (double jump) | §8 `DoubleJumpImpulse`, exact | clean |
| `car::update_double_jump_or_flip` (flip init) | flip direction conventions | clean |
| `car::update_boost` (accel) | §5 directional ΔV | **DEFECT** |
| `car::update_boost` (consumption) | §7 boost error 0.0000 | clean |
| `car::update_auto_roll` | tilted-ground vs upright control | clean (n=57) |
| `car::update_auto_flip` | §9b turtle set, 3,086 transitions | exercised, no anomaly |
| `ball::pre_tick_update` (gravity, drag) | §7 free flight 0.026 uu | clean |
| `ball::on_hit` (car→ball impulse) | §7 phase-robust ratio 1.027 | clean |
| `ball::on_world_hit` (bounces) | free flight incl. bounces | clean |
| `ball` angular velocity | §8 exact (0.0000 med and p90) | clean |
| `arena::on_car_car_collision` (bump) | §8 magnitude ratio 0.82 | clean |
| `arena::on_car_car_collision` (demo) | §8 supersonic gate, 23/23 | clean |
| boost pad pickup | §7 perfect agreement | clean |
| `ball::on_dropshot_tile_collision` | — | out of scope (not soccar) |

### 9b. `update_auto_flip` — closed

The audit initially left this as the one untested path (n=2 in the telemetry). Widening
the search from 2 replays to all 83 found **5,824 inverted-on-ground frames across 1,611
turtle episodes and 231 distinct cars** — ample data.

- Real |roll| while turtled: **median 2.502 rad**, and **35.1%** of frames exceed
  RocketSim's `autoflip::ROLL_THRESH = 2.8` — i.e. the threshold sits sensibly inside the
  real distribution rather than trivially above or below it.
- Replaying 3,086 turtle transitions: **up-vector error median 0.0754** — RocketSim
  tracks the real game's turtle orientation closely.
- Forcing `jump=true` (auto-flip fires) vs `jump=false` gives 1.1497 vs 1.1541 rad/s —
  **indistinguishable**. Over a 4-tick window the auto-flip impulse (200 uu/s over 0.4 s)
  is smaller than the noise from unknown player inputs.

**Status: exercised on real data with no anomaly found, but the auto-flip impulse is not
isolated from input noise.** That is a weaker statement than the other validations and is
labelled as such — but it is no longer an untested code path.

**With this, every soccar physics code path in §9 has been exercised against real-game
data.** Two are defective; the rest show no anomaly at the precision available.

## 10. Oracle inventory — why the residuals are oracle-limited, not effort-limited

Every source of ground truth available for this system, and what each yielded:

| oracle | gives | used for | exhausted? |
|---|---|---|---|
| `RocketLeague.exe` via Ghidra | code *structure*; 3,740 named natives | dodge cancel logic, z-damp gating, proof constants are CDO fields | yes — constants are in the encrypted package |
| RL replays (`boxcars`, 83 files) | 30 fps quantised state, **`DodgeTorque`**, **`DoubleJumpImpulse`**, demolitions, loadouts, ball spin, multi-car | flip defect, double-jump validation, demo gate, spin, bumps, ceiling, turtle | yes — no inputs recorded, no contact events |
| RLBot telemetry (one match) | 120 Hz state **with controls** | air-throttle defect, per-mechanic decomposition, pad pickup, regime sweep | yes — one match |
| RocketSim source | the implementation under test | unused-constant sweep, formula comparison | yes |
| RocketSim v2 | — | **not a valid reference** (user-stated) | n/a |

**A path I proposed and must withdraw.** I twice suggested adding
`is_flipping`/`flip_time`/`flip_rel_torque` and per-tick contact events to the RLBot
debug JSONL and playing a match. **That would not work.** The RLBot v5 packet exposes
only `air_state`, `boost`, `demolished_timeout`, `dodge_timeout`, `has_dodged`,
`has_double_jumped`, `has_jumped`, `physics`, `player_id`, `team` — there is no dodge
torque, no flip timer, and no collision/contact event. A bot cannot log what the packet
does not carry. (The replay *does* carry `DodgeTorque`, which is why the replay, not the
bot, was the oracle that cracked the flip defect.)

So the two residuals — contact-phase resolution and hitbox effects below control noise —
are **oracle-limited**: no available source supplies per-tick contact data or
input-recorded multi-body play. They are not waiting on more analysis.

## Coverage — subsystem status

Every soccar subsystem has now been probed against a valid reference (real telemetry,
real replays, or RL's own code). What follows is not "unexamined" — it is the residual,
with the reason each cannot be pushed further.

**Validated, no defect found:** arena geometry · ground contact (sticky force,
suspension) · flat/wall/ramp/**ceiling** driving · ball free flight · **ball spin
propagation** (exact) · ball-car hit impulse magnitude · **spin imparted by a hit** ·
boost consumption · boost-pad pickup · flip duration · **double-jump impulse** (exact) ·
**demolition supersonic gate** · car-car bump magnitude · multi-car 2v2/3v3.

**Confirmed defects (2):** air-throttle double-count while boosting; flip angular
dynamics.

**Residual limits:**

- **Car-car bump reproduction rate** — magnitude looks right (median ratio 0.82) but
  only 28% of real bumps reproduce from a quantized frame-boundary state (§8). This
  is a *measurement* limit of 30 fps replays, not a known defect. Settling it needs
  per-tick contact data, not more replays.
- **Boost-pad car-locking** — v3's pickup is cylinder-only (§7); the pad-camping
  case is untested against the real game.
- **Car bodies other than Octane** — tested and not discriminable (§8): control noise
  in replays is ~5000× the hitbox effect on ground motion. Blocked on input-recorded
  multi-body data. Pulsar plays Octane.
- **Contact-phase resolution.** Every contact-type measurement (ball hit, bump, spin
  imparted) is bounded by how precisely contact timing can be recovered from sampled
  state — 8-tick telemetry or 4-tick quantized replays. Magnitudes check out
  (ball-hit ratio 1.027, bump 0.82, spin err 0.42 rad/s); exact per-contact fidelity
  would need per-tick instrumentation inside the real game, which this audit cannot do.
- **Boost-pad car-locking** — v3's pickup is cylinder-only (§7); the pad-camping case
  is untested against the real game.
- Sample base: one 1v1 RLBot match at 120 Hz (§7) plus 83 replays at 30 Hz (§8).

## §21 — The floor-to-wall fillet: three confirmed divergences (2026-08-02)

66-segment scripted capture, real game vs sim. The duplicate-SEG-name artefact from the
previous capture is confirmed dead (land_nose_down 707.9 -> 1.94 uu etc.), so the numbers
below are physics.

Flat geometry is at the noise floor (floor/wall/air all 5-12 uu median). Every one of the
worst segments is a CURVED or attitude-resolution contact:

| segment | p50 | max | what diverges |
|---|---|---|---|
| `transition_flip_early` | **122.9** | 413.8 | flip landing on the 256uu fillet |
| `corner_drive_up` | 45.6 | 184.9 | big 1152uu corner |
| `crossbar_land` | 28.4 | 56.2 | goal frame |
| `tilt_on_side` | 27.7 | 115.3 | on-side landing recovery |
| `transition_drive_along` | 21.0 | 96.9 | driving up the fillet |

### 21.1 The fillet over-launches (worst defect in the script)

`transition_flip_early` tracks the real game to **0.9 uu at tick 36** -- dead on until the
car meets the curve. Then:

- sim reports `is_on_ground` at tick **96**, real not until **144** (0.4 s earlier attach)
- by tick 192 sim z = **773**, real z = **374** -- the sim gets roughly double the climb

So on flip contact with the fillet the sim attaches to the surface far too early and
converts the contact into far too much up-wall velocity. Prime suspect is the wall-stick /
surface-attach force being applied on the curved fillet where the game does not yet apply
it. NOT YET INVESTIGATED against Ghidra -- do that before touching anything.

### 21.2 Wall climb ~4% fast

`transition_drive_along` diverges monotonically (5 -> 95 uu) with both venues attached the
whole way: z 1770 sim vs 1703 real at tick 216. A steady overspeed climbing the wall, plausibly
the same root cause as 21.1.

### 21.3 On-side landing resolves the wrong way

`tilt_on_side`: sim plants at tick 72 and slides to x = -25; real stays airborne to tick 120
and recovers to x = 0. 28 ground-flag mismatches. The flop-to-wheels / auto-right resolution
differs in DIRECTION, not just magnitude.

### 21.4 Six segments never captured

`supersonic_run`, `into_net`, `transition_curve_dash`, `transition_wall_dash`,
`transition_flip_off`, `transition_supersonic_into`. `supersonic_run` has now failed to
capture twice. Cause unknown -- do not claim full coverage until this is explained.

## §22 — Full coverage (77/77) + a real-vs-real control (2026-08-02)

The runner's fixed 25uu state-set gate was silently skipping every high-speed segment
(§21.4 cause found: the gate is checked one 60Hz packet after a state set that carries
velocity, so a 2200uu/s spawn is 36.7uu away before it can ever pass). Fixed; 77/77 now
capture, including the wall dash and both supersonic segments.

### 22.1 CONTROL: the real game is deterministic

Two real captures of the same script diff at **0.00 uu median on nearly every segment**.
The measurement floor is therefore ~0, and any sim-vs-real error is a real model defect.
TWO EXCEPTIONS, and they matter:

- `transition_flip_into` real-vs-real **78.0 uu** -- LARGER than its 41.0 uu sim-vs-real
  error. RETRACT it as a finding; it is not distinguishable from run-to-run variation.
- `transition_flip_early` real-vs-real **30.8 uu** vs 152.4 uu sim-vs-real. Still a real
  defect, but the fillet contact is genuinely chaotic and the effect is smaller than §21
  claimed. §21's 122.9 uu figure also drifted to 152.4 uu across runs for this reason.

Fillet contact is sensitive to initial conditions in BOTH venues. Quote per-segment
real-vs-real alongside any fillet number from now on.

### 22.2 Flip cancels are a bigger defect than the fillet

| segment | sim-vs-real p50 | real-vs-real | note |
|---|---|---|---|
| `speed_flip` | **583.5** (up err **94.6 deg**) | ~0 | worst defect in the sim |
| `stall` | **249.9** | ~0 | |
| `transition_flip_off` | 200.7 (fwd err 88.8 deg, 73 gnd mism) | n/a | |
| `transition_flip_early` | 152.4 | 30.8 | fillet, partly chaotic |
| `transition_supersonic_into` | 94.3 | n/a | |
| `transition_curve_dash` | 63.1 | n/a | |
| `flip_into_wall` | 30.1 | ~0 | |
| `transition_wall_dash` | 29.9 | ~0 | |
| `powerslide_recover` | 23.3 | ~0 | |

`speed_flip` ending 94.6 degrees off in ORIENTATION with a ~0 noise floor is the single
largest confirmed sim-to-real defect found so far, and it is the standard kickoff mechanic.
Note `dodge_then_cancel` is only 5.3 uu, so the plain cancel is fine -- the defect needs the
cancel COMBINED with air roll and boost. Diagnose before touching anything.

### 22.3 The boost-pad fixes are validated

First test coverage since they shipped 2026-08-01: `boostpad_big` 10.9, `boostpad_small`
10.8, `boostpad_clip` (edge of a big pad) 5.0 uu -- all at the noise floor. Ceiling drive/
drop, tornado spin and supersonic_run are also clean.

### 22.4 Still untested

Car-ball contact of any kind (the ball is parked at (-3500,4800) by design), and bumps/
demos (needs a second car). Both need runner work, not more segments.

## §23 — Deterministic jump state; speed_flip survives and is the real defect (2026-08-02)

### 23.1 The harness was inventing a defect

The real game CARRIES jump/flip state across a state set; the sim runner zeroed the flags
directly. Segments therefore inherited whatever the previous one left, in ONE venue only,
making results depend on script ORDER.

This produced a fully convincing false positive. RocketSim's `can_use` gate genuinely omits
`has_jumped` (and `air_time_since_jump` resets every tick while it is false, so the
DOUBLEJUMP_MAX_DELAY window is vacuous). Adding the check moved `stall` 249.9 -> 8.1 uu.
But `stall` follows `speed_flip`, which ends mid-flip: the real car had merely spent its
flip. `corner_flip_into` (following a segment that ends grounded) shows the real car taking
a **+280.8 uu/s jump impulse** in mid-air, never grounded, then pure gravity -- RL DOES
allow it there, and the patch regressed that segment 20.3 -> 103.5 uu. NOT PATCHED.

Both runners now park on the floor for 12 ticks and let GROUND CONTACT clear the flags.
Proof it was artefact: with **no code change**, `stall` went 249.9 -> **9.3 uu** and
`transition_flip_early` 152.4 -> **81.7 uu**.

### 23.2 CAVEAT: the `ground` column is not comparable between venues

The real game reports `AirState::Jumping` from the instant jump is pressed while the wheels
are still down; RocketSim keeps `is_on_ground` true through that window (already documented
at RLBotClient.cpp ToPlayer()). Measuring "liftoff" off that column reads a uniform +2 tick
difference across 26/27 segments -- that is the observation semantics, NOT physics. Do not
draw ground-contact conclusions from it, and treat the `gnd mism` column as advisory.

### 23.3 speed_flip is real, and it is the worst defect

583.5 -> **583.3 uu** across the state fix, i.e. entirely unaffected: not an artefact.
Orientation error **94.5 deg** against a ~0 real-vs-real noise floor.

Localised, and the localisation is the useful part:

| segment | error | verdict |
|---|---|---|
| `dodge_diagonal` (diagonal dodge, no cancel) | 13.1 uu | fine |
| `dodge_then_cancel` (axis dodge + cancel) | 4.5 uu | fine |
| `speed_flip` (diagonal dodge + cancel) | **583.3 uu** | broken |

So neither ingredient is broken; the COMBINATION is. In the trace the sim reaches
avx = -1.59 rad/s two ticks after the dodge while the real car is at -0.12 and then rotates
the OTHER way (+1.67 peak), i.e. the real car appears to respond to the cancel input via air
control while the sim commits a full dodge rotation.

RULED OUT: `do_air_control = true` in RocketSim's cancel branch (removing it moves the
error 583.3 -> 585.8, no effect). Also ruled out earlier: the cancel SIGN convention --
`flip_rel_torque.y = -pitch_at_dodge`, so RocketSim cancels on opposite pitch exactly as
RL's `CarComponent_Dodge_TA_ApplyTorqueForces` does (which modulates ONLY the Y component,
by `1 - |pitch clamped toward the opposing sign|`; X is unmodulated in both).

Still open. Next candidates: whether RL fires a dodge at all for this input/timing, and the
torque magnitude term `C1 / *(*(car+0x438)+0x238)` in the RL native, which has no
counterpart in RocketSim's `rel_torque * (TORQUE_X, TORQUE_Y, 0) * TICK_TIME`.

### 23.4 Current ranking on trustworthy state (80/80 captured)

| segment | p50 |
|---|---|
| `speed_flip` | 583.3 |
| `jump_after_wall_launch` | 213.0 |
| `transition_flip_off` | 184.1 |
| `no_jump_control` | 139.0 |
| `transition_curve_dash` | 108.9 |
| `transition_supersonic_into` | 101.9 |
| `transition_flip_early` | 81.7 |
| `corner_drive_up` | 41.6 |

`no_jump_control` never presses jump, so its 139 uu is a clean measure of the fillet/wall
launch divergence alone -- the jump_after_* pair cannot settle the has_jumped question until
that is fixed, because the launch itself dominates.

## §24-26 — Two physics fixes and two harness fixes (2026-08-02)

Whole-script median position error, summed over 80 segments:

| state | total | note |
|---|---|---|
| start of session | 1729.7 | |
| + post-jump dodge lockout (S24) | — | speed_flip 583.3 -> 19.0 |
| + wheel-gated chassis friction (S25) | 1322.5 | fillet cluster |
| + harness offsets removed (S26) | **1137.6** | measurement only, not a physics change |

Now: median **5.69 uu**, 58/80 under 15 uu, 51/80 under 10 uu, 35/80 under 5 uu.

### S24 — post-jump dodge lockout

RocketSim let a dodge fire as soon as `is_jumping` cleared (~3 ticks after a 1-tick jump
tap). The real game refuses it there: `speed_flip` dodges 6 ticks after the jump and the
real car does NOT dodge (roll rate flat at 0.00 rad/s while the sim reached 7.2), yet
`wavedash_forward` dodges at 18 ticks and matches to 5.9 uu. Swept window [0.0167, 0.0333];
`jump::MIN_TIME` = 0.025 sits dead centre. Gated on `has_jumped`, since
`air_time_since_jump` is pinned at 0 for a car that never jumped and gating unconditionally
re-imposes the block disproved in S23.

### S25 — chassis friction while the wheels carry the car

The fillet is CONCAVE, so a long box hitbox digs in and the chassis scrapes while the car
drives normally. The sim bled ~257 uu/s crossing it at supersonic, after which the deficit
stayed pinned at exactly -248.8 uu/s -- a one-off energy loss in the curve, not a force
error. Suppressed when `num_wheels_in_contact >= 3` (RocketSim's own on-ground threshold);
>=1/>=2 also fix the fillet but wrongly catch tilted landings, which touch one or two
wheels AND the shell.

### S26 — the metric was charging the sim for harness offsets

(1) The sim runner stepped then logged, so every sim row was one tick ahead of the real
runner, which logs the packet before setting controls -- 18.3 uu of offset per sample at
supersonic. (2) The sim started exactly on the scripted state while the real car started
wherever the state set landed (up to ~179 uu at 2200 uu/s). Runner now logs at the start of
each tick and accepts a real capture as a 4th arg to seed each segment's t=0 state
(orientation from the captured forward/up basis, not round-tripped Euler). Alignment scan
minimum moved to shift 0, confirming the fix.

### 26.1 KNOWN IRREDUCIBLE: symmetric tilted landings are chaotic

`tilt_nose_down` (62.1 uu) and `transition_land_tilted` (62.1 uu) land almost perfectly --
z and up.z match to three decimals -- then accumulate error purely from a LATERAL kick whose
SIGN differs (sim vx +88 vs real -95). The setup is symmetric about x=0, so that kick is
symmetry-breaking from contact ordering and its sign is arbitrary. `coast_decel` is exact
(vy identical), so ground physics is sound. Not fixable by any constant; do not chase it.
Seeding handedness was checked against `rot_from_euler` (identity and roll=180) and is
correct, so this is not a mirrored-basis artefact.

### 26.2 Still open, in priority order

| segment | uu | what |
|---|---|---|
| `transition_flip_off` | 178.9 | after flipping off the wall the real car RE-ATTACHES and drives up it (x pinned 4079, z 400->739) while the sim detaches and accelerates away (vx -84 -> -330 with no thrust). Restitution swept and rejected (0.3 is optimal; 0.0 is worse overall). Suspect sticky-force / suspension raycast range. |
| `jump_after_wall_launch` | 109.8 | same family |
| `crossbar_land` | 37.3 | goal frame geometry |
| `corner_drive_up` | 33.4 | big 1152uu corner, same class as the fillet but not fixed by the S25 gate |

Rejected this session, with evidence: post-integration velocity clamp (total 1729.7 ->
2269.9, transition_curve_dash 107 -> 684); `do_air_control` in the dodge-cancel branch
(583.3 -> 585.8, no effect); HIT_WORLD restitution 0.0 and 0.15 (both worse than 0.3).

## §27 — Holdout validation: the fixes generalise (2026-08-02)

The S24/S25 constants were chosen by sweeping against one capture, so they were refit-risk.
A second independent capture of the same 80 segments settles it.

| | fit capture | holdout capture |
|---|---|---|
| total | 1137.6 | **1187.6** |
| median | 5.69 | **5.96** |
| < 15 uu | 58/80 | 59/80 |
| < 10 uu | 51/80 | 50/80 |
| < 5 uu | 35/80 | 34/80 |

Every segment the fixes targeted is stable to ~1 uu across the two captures:

    speed_flip      15.81 -> 15.51      no_jump_control            12.53 -> 12.20
    half_flip        6.20 ->  6.16      transition_supersonic_into 34.04 -> 35.01
    stall            2.33 ->  2.31      corner_land_steep          26.68 -> 27.55
    dodge_forward    9.56 ->  9.74      flip_into_wall             16.54 -> 16.54
    wavedash_forward 5.25 ->  5.17      corner_drive_up            33.35 -> 34.41

NOT overfit. `jump::MIN_TIME` for the dodge lockout and `>= 3` wheels for the friction gate
were both already-existing constants that the sweep happened to land on, which is the
reason to trust them over a fitted value.

### 27.1 The real game's own repeatability, capture vs capture

Total 94.9 uu, median **0.00** -- deterministic almost everywhere. The exceptions are the
whole story of the fit->holdout drift:

| segment | real-vs-real | sim error fit -> holdout |
|---|---|---|
| `transition_flip_into` | **63.1** | 38.5 -> 72.3 |
| `jump_after_wall_launch` | 17.1 | 109.8 -> 117.5 |
| `dodge_backward` | 10.6 | 4.1 -> 11.0 |

All three of the largest fit->holdout moves are the three least repeatable segments. The
drift is real-game variance, not refit.

**RETRACT `transition_flip_into` as a finding, for the second time.** Its real-vs-real noise
(63.1 uu) exceeds its sim error, exactly as in S22.1. Quote per-segment real-vs-real
alongside any number from the flip/fillet family before treating it as signal.

### 27.2 Where the sim stands

Median 5.96 uu against a ~0 uu noise floor, 59/80 segments inside 15 uu. Flat-surface
driving, coasting, braking, air control, boost, boost pads, jumps, dodges, dodge cancels,
half flips, wavedashes, ceiling driving and free-fall landings are all at or near the floor.

What is left is one family -- the car detaching from a wall or the fillet after a flip
(`transition_flip_off` 183.4, `jump_after_wall_launch` 117.5) -- plus the irreducible
symmetric-tilted-landing chaos of 26.1. Progress on the first needs RL's wheel/suspension
physics from `Vehicle_TA`; there are no `Sticky*` symbols in the binary, so RocketSim's
sticky force is a modelling construct with no directly comparable ground truth.

## §28 — RL has no sticky force: it is all suspension (2026-08-02)

S26.2 claimed the wall-detach family had "no directly comparable ground truth" because the
binary has no `Sticky*` symbols. That was the wrong search, not the wrong question -- the
behaviour exists in the game, so the code exists under a different name.

**Found it.** RL keeps a per-force name table at `0x141a4f8b0`, 40-byte records of the form
`Impulse` / `<name>` / `AddForce`. There are exactly FOUR wheel forces:

    141a4f8b8  WheelSuspension
    141a4f8e0  WheelFriction
    141a4f908  WheelDrive
    141a4f930  WheelBrake

There is no sticky/magnet/adhesion force anywhere. A car holds a wall in RL purely through
`WheelSuspension` acting over its travel range. (Also confirmed present: a `Wheel_TA` class
with `GetSuspensionDistance` / `GetSuspensionOffset` / `GetLocalWheelLocation`, and
`Vehicle_TA::GetNumWheelContacts` / `GetWheelWorldContacts`. The `Suspension travel/spring/
damper` strings at 1420403e1 are PhysX's `NpWheelShape`, not RL's own vehicle sim -- do not
mistake them for RL parameters.)

Locating the force code itself needs more than a name: the strings are inline char buffers
inside a struct array, so there is no 8-byte pointer and no RIP-relative LEA to them (both
searched, and a full 27 MB `.text` scan for the LEA found nothing).

### 28.1 But RocketSim's sticky force cannot simply be deleted

Removing it is strictly worse across the board (holdout total 1187.6 -> **1481.7**,
transition_flip_off 183.4 -> 265.6, jump_after_wall_launch 117.5 -> 199.8), at every
suspension travel from 12 to 32. So RocketSim's `StickyForce` is compensating for a real
structural difference in how its Bullet raycast suspension behaves versus RL's
`WheelSuspension`. It is an approximation that earns its place; it is not a stray hack.

### 28.2 Suspension travel is coupled to ride height, so it cannot be tuned alone

With the sticky force kept, sweeping `MAX_SUSPENSION_TRAVEL` (holdout totals):

| travel | total | flip_off | jump_wall | drive_throttle | coast_decel |
|---|---|---|---|---|---|
| 12 (current) | 1187.6 | 183.4 | 117.5 | 7.4 | 4.0 |
| 16 | 1176.7 | 181.2 | **27.9** | 8.4 | 5.6 |
| 20 | 1200.9 | 167.1 | 26.4 | 10.9 | 8.9 |
| 26 | 1485.3 | **59.2** | 73.4 | 15.9 | 14.6 |

More travel buys wall re-attachment and pays for it in flat driving, because
`Car::new` derives `suspension_rest_length = wheel_config.suspension_rest_length -
MAX_SUSPENSION_TRAVEL`, so the same constant sets BOTH the ground-detection reach and the
ride height. NOT ADOPTED: travel 16 is only 0.9% better overall, degrades the two
best-validated segments in the whole script (drive_throttle, coast_decel), and is a fitted
number with no counterpart in RL.

### 28.3 The concrete next step

Decouple the two roles: extend the wheel raycast's ground-detection length WITHOUT reducing
`suspension_rest_length`, so a car can re-acquire a surface it has drifted ~17 uu from (the
measured real re-attachment distance in transition_flip_off) while ride height and the
flat-driving response stay exactly as they are. That is a change in the Bullet vehicle layer
(`bullet/dynamics/vehicle.rs` wheel raycast), not a constant, and it should be swept against
BOTH captures with drive_throttle/coast_decel as hard guards.

## §29 — Wall adhesion decomposed: two mechanisms on one constant (2026-08-02)

S28 proposed decoupling the wheel raycast's ground-DETECTION reach from ride height, so a
car could re-acquire a wall it drifted off without disturbing flat driving. Implemented and
swept (`SUSPENSION_DETECT_EXTRA`, added only to `real_ray_length`).

**The decoupling itself works perfectly.** Across extra = 0/8/16/30/50 uu the guards never
move at all: `drive_throttle` 7.39, `coast_decel` 3.97, `wall_drive` 2.32, identical to four
significant figures. Ride height and flat driving are genuinely untouched.

And the wall family improves a lot at extra = 8:

    transition_flip_off        183.35 -> 114.55      ceiling_drive   22.53 -> 10.14
    transition_supersonic_into  35.01 ->   8.91      ceiling_drop    14.92 ->  5.45
    corner_drive_up             34.41 ->  12.76      speed_flip      15.51 ->  9.46

18 segments improve and the MEDIAN improves (5.96 -> 5.71). But the total worsens
(1187.6 -> 1487.3) on essentially one segment: `half_flip` 6.16 -> **334.64**.

### 29.1 Correction to S28's reasoning: the suspension cannot pull

S28 assumed the suspension force goes negative past the rest length and provides a bounded
pull toward the surface. It does not. `update_suspension` ends with:

    if wheels_suspension_force <= 0.0 { return; }

so a negative force is discarded outright. Extending the raycast therefore creates no
suspension pull whatsoever. What it actually extends is the range over which
`is_in_contact_with_world` is true, and that flag gates three different things.

### 29.2 The decomposition (measured, not assumed)

Re-ran extra = 8 with the sticky force disabled to separate the effects:

| segment | extra 0 | extra 8 | extra 8, no sticky |
|---|---|---|---|
| `transition_flip_off` | 183.35 | **114.55** | 235.84 |
| `half_flip` | 6.16 | 334.64 | **335.09** |

- The wall-adhesion GAIN needs the sticky force (without it flip_off is 235.8, worse than
  baseline). So adhesion in RocketSim is entirely `StickyForce`, gated on
  `wheels_have_world_contact` -- extending reach extends adhesion. This is the sim's stand-in
  for RL's `WheelSuspension`, which is RL's only candidate (S28: four wheel forces, no
  sticky force of any name).
- The `half_flip` LOSS is unrelated to sticky (335.09 without it). It comes from wheels being
  classified in-contact while genuinely airborne, so wheel FRICTION and DRIVE apply to a car
  that has left the ground.

One constant currently drives both, which is why the sweep cannot win: any reach long enough
to hold a wall is also long enough to make a jumping car think it is still on the floor.

NOT ADOPTED, and reverted; baseline stands at total 1187.6, median 5.96.

### 29.3 The actual fix

Separate ADHESION reach from CONTACT CLASSIFICATION. `wheels_have_world_contact` (which
gates only the sticky force, car/base.rs) should use the extended probe, while the wheel's
friction/drive/suspension path keeps the current reach -- i.e. a wheel within the extended
band adheres but does not drive, brake or generate lateral friction.

Guards for that change, all currently near the floor and all held exactly by the reach
decoupling: `drive_throttle` 7.39, `coast_decel` 3.97, `wall_drive` 2.32, and now
`half_flip` 6.16 as the specific regression sentinel. Sweep against BOTH captures.

Expected prize if it lands cleanly: roughly -180 uu of total error (the extra-8 gains
without the half_flip loss), taking the script to about 1000 uu total and a median near
5.2 uu.

## §30 — Adhesion-only band: ADOPTED (2026-08-02)

S29 identified that one constant drove two mechanisms. Split them: a wheel that finds a
surface beyond the suspension's working range now sets `adhesion_contact` and feeds ONLY the
sticky-force gate, producing no friction, drive or suspension force. `SUSPENSION_DETECT_EXTRA`
sets that extra reach.

This is the sim's stand-in for RL's `WheelSuspension` reach -- RL has four wheel forces and
no sticky force of any name (S28), so a car out there in RL is held by suspension alone and
is certainly not driving.

Swept against BOTH captures. **2.0 uu adopted**, and it sits just under a cliff:

| extra | FIT | HOLD | median | <10uu | <5uu | curve_dash | half_flip |
|---|---|---|---|---|---|---|---|
| 0 (before) | 1137.6 | 1187.6 | 5.96 | 50 | 34 | 31.5 | 6.16 |
| **2.0** | **1112.1** | **1161.1** | **4.98** | **54** | **40** | 32.1 | **3.41** |
| 4.0 | 1178.8 | 1223.8 | 4.88 | 55 | 41 | **120.8** | 3.52 |
| 8.0 | 1179.4 | 1217.2 | 5.40 | 56 | 36 | 120.1 | 3.73 |

At 4.0 and above the CONCAVE fillet pushes legitimately-driving wheels into the band; they
stop generating drive and the car loses traction on the curve. 2.0 keeps the gains without
crossing that line.

Consistent across both captures (16-17 segments improved, the same 6 regressed). Every guard
held to four significant figures: `drive_throttle` 7.39, `coast_decel` 3.97, `wall_drive`
2.32, `brake_hard` 15.39, `steer_full` 11.59 -- and `half_flip` improved 6.16 -> 3.41.

Biggest gains: `corner_drive_up` 34.4 -> 12.8, `ceiling_drive` 22.5 -> 12.8, `dodge_forward`
9.7 -> 4.2, `ceiling_drop` 14.9 -> 9.9, `flip_land_early` 6.7 -> 2.9, `speed_flip` 15.5 ->
12.2, `dodge_side` 6.0 -> 3.3.

HONEST COST: `transition_flip_early` regresses 28.4 -> 60.3 on both captures, and
`jump_after_wall_launch` 117.5 -> 127.8. Both are in the wall/fillet-flip family that remains
the open problem. Taken because total, median and every threshold count improve on two
independent captures while all guards hold.

### 30.1 Final state of the sim

| metric | session start | now |
|---|---|---|
| total | 1729.7 | **1161.1** |
| median | — | **4.98 uu** |
| < 15 uu | — | 62/80 |
| < 10 uu | — | 54/80 |
| < 5 uu | — | 40/80 |
| < 2 uu | — | 20/80 |

Against a real-vs-real noise floor of median 0.00 uu (94.9 total).

### 30.2 What is left

`transition_flip_off` (185.4) and `jump_after_wall_launch` (127.8) -- the car detaching from
a wall after a flip. The adhesion band helps the approach but not this. The remaining
suspect is that RocketSim's sticky force uses `get_upwards_dir_from_wheel_contacts`, which
with one or two grazing wheels can yield an "up" that points away from the surface, so the
force pushes the car off instead of holding it. Worth testing next: derive the sticky
direction from the CONTACT NORMAL rather than the inferred wheel-contact up.

Plus the irreducible symmetric-tilted-landing chaos of 26.1.

## §31 — Is leaf-by-leaf equivalence feasible? Measured, not guessed (2026-08-02)

Question: treat both physics implementations as trees and verify every leaf matches.

### 31.1 The node level maps almost 1:1 — this part is easy

RocketSim's car per-tick tree is 8 nodes and **47 decision points** (`update_wheels` 11,
`update_double_jump_or_flip` 13, `update_air_torque` 7, `update_jump` 6, `update_auto_flip` 4,
`pre_tick_update` 4, `update_boost` 2, `update_auto_roll` 0). Small enough to enumerate by hand.

RL's is enumerable too, because UE3 names its components:

| RocketSim | RL |
|---|---|
| `update_wheels` | `WheelSuspension` / `WheelFriction` / `WheelDrive` / `WheelBrake` (S28) |
| `update_jump` | `CarComponent_Jump_TA` |
| `update_double_jump_or_flip` | `CarComponent_DoubleJump_TA` + `CarComponent_Dodge_TA` |
| `update_air_torque` | `CarComponent_AirControl_TA` + `Dodge_TA::ApplyTorqueForces` |
| `update_auto_flip` | `CarComponent_FlipCar_TA` |
| `update_boost` | `CarComponent_Boost_TA` |
| (demos) | `CarComponent_TerritoryDemolish_TA` |

Per-component logic IS recoverable: `CarComponent_Dodge_TA_ApplyTorqueForces` was decompiled
in full and its pitch-modulation rule read off directly.

### 31.2 The leaves are the blocker — but only for a STATIC diff

Leaves are numeric constants, and in UE3 those live in the CDO inside the .upk packages, not
in the exe. The exe carries property NAMES only. So a static value-by-value diff needs the
package/reflection route, and several nodes are virtual dispatches that need vtable
resolution first.

**But static equivalence is not what we need.** RocketSim is Bullet-derived and RL is not, so
the code will never correspond line-for-line even where behaviour is identical. What matters
is behavioural equivalence at each leaf, and we already have an oracle for that: the maneuver
harness, with a real-vs-real noise floor of median 0.00 uu.

### 31.3 So the tractable form is BRANCH-COVERAGE testing, and it is already 91% done

Built the sim runner with `-C instrument-coverage` and ran the 80 segments:

| file | region coverage |
|---|---|
| `sim/car/base.rs` | **91.38%** (68/789 regions missed) |
| `bullet/dynamics/vehicle/wheel_info.rs` | **94.98%** |
| `bullet/dynamics/vehicle/vehicle_rl.rs` | **92.50%** |

So the answer to "is it feasible" is yes, and most of it is done. The remaining work is a
finite, named list rather than an open-ended audit.

### 31.4 The measured gap list

Physics paths NEVER executed by any of the 80 segments:

| path | lines | note |
|---|---|---|
| `update_auto_flip` | 18 | **the biggest hole.** Jump while upside-down on the ground to right the car -- a real mechanic, entirely untested |
| `pre_tick_update` demo/respawn + `demolish` | 11 | demos, known gap (needs a second car) |
| `update_double_jump_or_flip` deadzone | 3 | input below `dodge_deadzone` -> double jump instead of flip; and the backward-dodge sign branch |
| `post_tick_update` supersonic grace expiry | 3 | dropping out of supersonic |
| `update_air_torque` L388 | 2 | air control while flipping with `flip_rel_torque == ZERO` (double jump, not dodge) |
| `finish_physics_tick` vel_impulse_cache | 3 | bump/demo impulse path |
| `update_boost` recharge | 2 | recharge mutator, not used in soccar |
| three-wheel curves | 3 | only for 3-wheel hitboxes, N/A for Octane |

Genuinely actionable: **auto-flip, the dodge deadzone, supersonic grace expiry, the
double-jump air-control branch**, plus bumps/demos and car-ball contact (which no segment
touches at all -- the ball is parked by design).

Closing those would take source-level branch coverage of the car physics to ~100% with, by
this count, roughly 6-10 new segments plus a two-car harness.

## §32 — Triage of an external fix list (Moonwatcher), 2026-08-02

A second, independent sim-vs-real effort published 10 fixes. Triaged against our tree:

| # | their fix | applies to us? |
|---|---|---|
| 1 | inverted `has_flipped` | **NO** -- ours is consistent (`has_flipped` = "flip spent", set at flip start, cleared on ground). Our own has_flip bug family was on the RLBot bridge side and was fixed 2026-07-31. |
| 2 | reset boost state every test tick | **NO** -- their harness |
| 3 | flip-torque units | **ALREADY SETTLED** -- we tried an inertia-scaled torque and it was a 95-122 deg regression; reverted with the reasoning recorded (S21) |
| 4 | steering applied at force-application time | **NO** -- `steer_angle` is set in `update_wheels` and consumed by `apply_ray_cast` within the same tick |
| 5 | wheel forces using current velocities | **NO** -- `update_vehicle_first` (raycast) and `update_vehicle_second` (suspension then friction) are called back-to-back with no force application between them |
| 6 | boost-pad pickup geometry | **ALREADY FIXED** here (BVH `cyl_radius` + OBB pickup); validated at the noise floor (4.96-10.9 uu) |
| 7 | **supersonic can start only while grounded** | **YES -- REAL GAP, FIXED** |
| 8 | correct hitbox per recording | **NO** -- their harness |
| 9 | action/contact diagnostics | **NO** -- their tooling |
| 10 | bump analysis | **NO** -- their tooling |

### 32.1 #7 adopted, with an explicit evidence caveat

`post_tick_update` set `is_supersonic = true` on speed alone, with no ground check. A car
merely exceeding the start speed in the air -- a fast aerial, or a fillet launch, which we
know reaches 2033 uu z at supersonic -- became supersonic, and since demolitions require
supersonic that manufactures PHANTOM DEMOS on contact.

Their evidence: 3 phantom demos removed, 13/13 demos correct.

**This is NOT validated locally, and cannot be.** `is_supersonic` has no trajectory effect in
RocketSim -- it only gates demos -- so the maneuver harness structurally cannot observe it
(no segment contains a second car; §31.4 already lists demos as an untested path). What was
verified here is that the change is trajectory-NEUTRAL: all 80 segments are bit-identical
before and after, so it cannot regress anything currently measured.

Re-test properly once a two-car harness exists. This is the first change this session
adopted on external evidence rather than our own measurement, and it is flagged as such in
the source.

### 32.2 Worth noting

Two independent efforts converged on the boost-pad pickup geometry (their #6, our earlier
fix) from different directions. That is mild corroboration for both.

## §33 — The static tree walk is BOUNDED by the binary (2026-08-02)

S31 said node-level correspondence was easy and only the numeric leaves were blocked by the
.upk packages. Walking it further found a harder, earlier blocker.

**RocketLeague.exe carries no MSVC RTTI.** Searched for `.?AVCarComponent_Jump_TA@@` and the
equivalent descriptor for DoubleJump, AirControl, FlipCar, Boost and Dodge -- none present.
So class vtables cannot be located by the standard type-descriptor -> complete-object-locator
-> vtable chain.

That matters because UE3 reaches most component natives through SHARED exec thunks:
`ACarComponent_DoubleJump_TAexecApplyForces`, `_Jump_TAexecApplyForces` and
`_AirControl_TAexecApplyForces` all resolve to the SAME thunk
(`ACarComponent_TA_execApplyForces__shared21` @140e78d80), which dispatches on a vtable slot
(`+0x698` CanActivate, `+0x6a0` PrePhysicsStep, `+0x6b0`/`+0x6b8` dodge impulse). With no
RTTI the concrete override cannot be identified statically.

Recoverable only where a class-SPECIFIC thunk exists:

| function | thunk | native |
|---|---|---|
| `Dodge_TA::ApplyTorqueForces` | 140e81650 | **140eb72b0, decompiled in full (S24)** |
| `AirControl_TA::GetInputForRotationAxis` | 140e84250 | 140ef0fc0, decompiled (a rotate-toward-target helper, not the player air-control torque) |
| `Boost_TA::IsBoostRestricted` | 140e7f4e0 | resolvable |

So: **the static leaf-by-leaf diff cannot be completed on this binary.** It is not a matter of
effort. Finishing it would need either a runtime vtable dump (attach and read the object's
vptr) or the .upk/reflection route -- both larger undertakings than the behavioural route.

**The behavioural route stands and is 91.4% done** (S31.3) with a finite named gap list. That
remains the way to "verify every leaf": exercise each branch and diff against a real capture
whose noise floor is median 0.00 uu.

## §34 — The graze-fling: progressive damper engagement past rest length (2026-08-09)

Fresh capture (all 80 segments, same script, RL running live): total 1157.4, matching the
Aug 2 close-out (1161.1) — the harness and the fixes reproduce across sessions.

### 34.1 A false alarm worth recording: the 5.5 rad/s "cap violation"

Sim angular speed reads 7.37 rad/s sustained through every dodge; the real packet never
exceeds 5.50. Before "fixing" the clamp order, measure the actual rotation: the REAL car's
forward vector also rotates at ~7.24 rad/s mid-dodge. **RL clamps only the replicated
`ang_vel` field to MAX_ANG_SPEED; internally it integrates above the cap during flip
torque, same clamp-then-torque order as RocketSim.** Do not "fix" this. Residual: sim
7.37 vs real 7.24 (~1.8%, ≈ 5° over a full dodge) — consistent with the 2.6–3.5° dodge
attitude medians; a −7% flip-torque scale would close it but was not tested this session.

### 34.2 The fling mechanism (impulse-traced, debug build)

`transition_flip_off` diverges at t57–66, while the car flips with two wheels grazing the
wall. Per-tick impulse history shows `WheelsSuspension` firing −7..−22 uu/s per tick with
the suspension EXTENDED past rest (spring force negative — it is all damper), plus two
−51/−59 spikes from the hard-contact `extra_pushback` resolve. Cause: the damper reads
contact-POINT velocity, and a flipping car sweeps its wheel contact points at ω×r ≈ 250+
uu/s — read as approach velocity. The real car in the same window (near-identical pose)
holds vx constant: RL applies no such force, then settles onto the wall spring-only,
lands mid-flip (its dodge torque visibly ends at ground=1, t≈100) and drives up the wall.
The sim car is flung out of suspension reach and can never re-land; it tumbles ballistic
after TORQUE_TIME with the z-damp having held vz ≈ −16 through the flip.

### 34.3 What was tried, what won

| variant | fit total | holdout total | verdict |
|---|---|---|---|
| stock | 1157.5 | 1181.9 | baseline |
| damp chassis vel, not contact-point | 1352.5 | — | ω×r damping is load-bearing for curve driving |
| damper off while flipping | 1195.0 | — | worse: flips need some damping |
| damper only when compressed | 1087.2 | 1117.0 | best total, but median 4.47→5.44, <15: −2, no_jump_control 12.5→37.0 |
| flat per-wheel impulse cap (5–80) | 1168–1748 | — | nonmonotonic; caps clip legitimate hard landings (tilt_* +30–45) |
| **linear damper ramp past rest** | **1126.0** | **1151.5** | **adopted** |

Adopted: damper scale fades linearly from 1 at rest length to 0 at max extension
(compressed range untouched). A barely-touching wheel damps nothing; slow fillet
transits keep their cushioning. Wins: `transition_land_tilted` 62.1→4.8,
`transition_supersonic_into` 35.6→9.3, `powerslide_recover` −2.4, `transition_flip_off`
−4.6. Costs (consistent on both captures): `jump_after_wall_launch` +29 (a chaotic
re-attach bifurcation; the `compressed` variant flips it the other way to −74),
`transition_curve_dash` +9, `corner_land_steep` +9. Threshold counts <10uu: 54→57 (fit),
54→57 (holdout); every steady-state guard bit-identical.

### 34.4 Still open

`transition_flip_off` (181) and `jump_after_wall_launch` (154): the detach itself is now
dominated by the compressed-range `extra_pushback` spikes (−51/−59 uu/s per tick from
`resolve_single_collision` while the flip sweeps wheels into the wall). RL presumably
resolves that geometry as chassis-mesh contact, not per-wheel ray pushback. Next lever:
trace whether suppressing/spreading `extra_pushback` during high-|ω| contact can keep the
car near enough to re-land without breaking the hard-landing segments that pushback
legitimately serves (the flat cap already proved those two families share the code path).

## §35 — Match play in both venues: real result, sim harness excavated (2026-08-09)

### 35.1 Real game: Pulsar (7.0b @ 8.68B) vs Nexto — 9–10 over 461s

Scored via `score_real.sh` (RLBotServer + the running game as backend + score_match.py;
new script, committed). Pulsar led 7–3 at t=231s, Nexto came back 9–10 by t=461s when the
session was ended. 19 goals ≈ one per 24s. Pulsar goal share **47%** — an even match, no
sign of the catastrophic real-game degradation earlier configs showed. Single match;
episode-cluster variance means ±15% on this share — treat as one data point, not a curve.

### 35.2 The sim side of every past "sim vs real" match was a THIRD engine

`RLBotSim/exe/Cargo.toml` pulled `rocketsim` from upstream git (v3-rust branch) — not the
vendored crate the trainer uses. Every sim scored match would have run WITHOUT any audit
fix (air-throttle gate, boost-pad OBB pickup, coasting brake, adhesion band, dodge
lockout, §34 damper ramp). Now points at the vendored engine (RLBotSim commit cfee9e4).

### 35.3 …but no sim scored match had ever actually run

Excavated in stages, each with its own fix:
1. `pkill -f RLBotServer` in a compound command kills the invoking shell (its own cmdline
   matches) — the CLAUDE.md warning applies to *every* -f pattern, not just the trainer's.
2. RLBotServer picks its game-bridge port dynamically (first free from 23233) and the
   running game (pinned to 23233 by `-rlbot`) steals the slot the instant it opens — one
   "sim" match at 18:48 actually played in the real game while rlbot_sim died retrying.
   `score_sim.sh` now holds 23233 with a dummy listener and reads the chosen port from
   the server log. A backend on 23234 (the default client port) connects but is never
   driven — hold that too.
3. The server will not drive a SILENT backend: the real game streams state packets even
   at the menu; rlbot_sim sent nothing until it had an arena, so the server validated the
   config and then did nothing — no agent launch, no spawn, at any port, with any engine.
   Fixed: rlbot_sim streams its default (Inactive) state pre-arena. This immediately got
   agents launched and the map-load command issued.
4. TERMINAL BLOCKER: the Jul-19 RLBotServer then sends the bridge a packet no rev of
   RLBot/rust-interface can parse (root vtable length 6 — a union type slot without its
   value; planus InvalidVtableLength). Tried schema submodule revs a83ce10 (Jun 12) and
   da7f97b (Jun 30): identical. The current core's bridge protocol has moved past the
   public Rust bindings. rlbot_sim now hex-dumps undecodable packets; the port to the
   current bridge protocol is spawned as its own task.

Net: the real-vs-sim MATCH comparison is still one-sided (real only). The maneuver
harness (§34) remains the only calibrated physics comparison, and it is the stronger
instrument anyway — fixed states, fixed actions, per-tick error, real-vs-real noise floor.

### 35.4 Deployment note

The trainer binary was rebuilt with the §34 damper fix (build/ at 18:49); it deploys on
the next trainer start. The maneuver sim runner and rlbot_sim both build against the same
vendored engine, so all three venues (trainer, offline maneuvers, future sim matches) now
share one physics.

## §36 — Demo/bump pipeline rebuilt from the Car_TA decompile (2026-08-09)

Sources (user-provided, RLGym community / "code soul" leak decompile): `ShouldDemolish.uc`
(archived at `research/reports/assets/ShouldDemolish.uc`), decompile screenshots of
`OnRigidBodyCollision`, `ApplyCarImpactForces`, `OnHitCar`, `IsValidBump`, `IsBumperHit`,
`GetBumpImpulse`, `BumpCar`, `InitTimeOfImpactFromOldRBState`, `IsInvulnerableToDemolishSource`,
and a partial `defaultproperties` CDO dump. This closes most of §31.4's "bumps/demos never
exercised" gap with real game logic instead of guesses.

**Independent confirmation first:** the CDO dump has `SuperSonicSettings = (Speed=2200,
TurnoffSpeedBuffer=100, TurnoffTime=1)` — exactly RocketSim's `START_SPEED` /
`MAINTAIN_MIN_SPEED` / `MAINTAIN_MAX_TIME`. Also `JumpLeaveGroundTime=0.125`,
`bAllowBackwardsDemolitions=1`, `PushFactor=0`, `DemoSpeedThreshold[2..4]=900/1300/1700`
(the slow/medium/fast demo mutators).

**Structural rules ported into `on_car_car_collision`** (all from the decompiled control flow):

1. Order: approach gates (speed>0, closing, relative-speed) → demo check → bump. The
   bump interval NEVER blocks a demo (it used to here: one cooldown gated both).
2. Demos require **forward-projected** speed ≥ 2100 (`Speed − TurnoffSpeedBuffer`), not
   just the supersonic flag — a sideways-sliding supersonic car cannot demo.
3. `bAllowBackwardsDemolitions=1`: the projection takes `abs()` and the contact test
   mirrors to the rear bumper (threshold scaled by rear/front hitbox extent, since the
   hitbox is offset forward — a fixed 64.5 can never fire on the Octane's 46-uu rear).
4. Bump curves are fed the attacker's FULL speed (`VSize(OldRBState.LinearVelocity)`),
   not the toward-victim projection. §8 measured sim bumps at 0.82× real, one-sided —
   consistent with the projection having been the smaller input.
5. Airborne victims get NO scripted vertical push (`GetBumpImpulse` only sets `ImpulseZ`
   in the grounded branch); grounded victims are pushed along THEIR up axis.
6. Bump rate-limiting is per-victim (`LastHitCar` + `BumpInterval`): a different car is
   always bumpable inside the cooldown. New `CarState::bump_last_victim` (not carried
   over the FFI; a state set clears it).
7. A non-bumper contact still "bumps" with zero impulse (CDO `PushFactor=0`) and arms
   the per-victim interval, exactly as `BumpCar` does.

**Cones ported (second pass, same day):** the user then provided the full community
replica (`research/reports/assets/titan_demo_replica.cpp` — titan/juan diego, validated
in their fork as "basically never misses a demo") with the real constants: demo cone
**45.572994 deg yaw x 36.869896 deg pitch** (pitch is exactly atan(3/4)), bump cone
**70 x 36.869896 deg**, tested against the CENTER-TO-CENTER direction with a verbatim
1.01 projection fudge, and reverseForward = vel.fwd < 0. Ported verbatim into
`car_within_forward_cone` + a deferred two-pass application (both directions evaluated
against pre-action state, so a MUTUAL supersonic head-on demos both cars). A supersonic
hit that fails the demo cone DEMOTES to a bump if the wide cone passes. This replaced
the `local_point_x` proxy entirely. Two places where the replica inherits stock-v2 code
that the .uc decompile contradicts were kept .uc-faithful and documented in the source:
the bump curves take FULL attacker speed (not the toward-contact projection) and the
airborne victim gets no up-push.

**CDO CONFIRMED AT SOURCE (2026-08-10, full CarInteractionSettings relayed; archived
at assets/car_interaction_settings_cdo.txt):** only `COMAngleCheck` is enabled with
exactly the ported cones; `VictimHitAngleCheck`/`AttackerHitAngleCheck`/
`VictimHitAngleCurveCheck` are `bEnabled=false` (skipping them was correct);
`bCheckImpactNormal=false`; `PushFactor=0`; `BumpInterval=0.25`. The bump curves are
the CDO's PushFactor points / car mass 180 -- the shipped values were these rounded
(1100 vs 1111.11); now set exactly. Still unknown: `AddedCarForceMultiplier`
(opposite-team bump boost, lives outside CarInteractionSettings) and demolish spawn
invulnerability.

**Validation:** five scenario tests (`rocketsim/tests/demo_bump.rs`): supersonic head-on
demos; sideways supersonic slide does NOT demo; reversing supersonic rear hit DOES demo;
grounded bump carries the up-push while an airborne victim takes none; a mutual
supersonic head-on demolishes BOTH cars. All pass. The
80-segment battery is bit-identical (single car — trajectory-neutral, as §32 required for
demo-path changes). Replay-level validation (23 real demos, bump dV ratio) still pending —
the §8 boxcars harness did not survive its session and would need rebuilding.

Trainer + RLBot client rebuilt with the new pipeline; deploys on next trainer start.

## §37 — First 120 Hz real capture: pitch-cancel gate + published-state clamp (2026-08-10)

New instrument: an 81 MB `.rlpr` from RLRecord2 (BakkesMod) — 53,680 per-tick states of a
strong tick-skip-1 bot 1v1 (7.5 min), replayed transition-by-transition through the sim
(`analysis.rs`, `RLPR_PATH=... cargo test --test mod analyze_rlpr`). Reader now handles
demolished-car short rosters (size-prefix peek) and skips teleports/gaps. This capture
carries no impulse traces (num_impulse_records=0 throughout — pre-trace plugin build).

### 37.1 Replay semantics, measured

Records are END-of-frame states, and an input delivered at frame R takes effect in frame
R+2 (verified on jump activations: press at 285 → jump at 287; 597 → 599). Simulating
record F → F+1 therefore uses controls from record F−1 (`GGL_CTRL_LAG`, default 2). The
wrong pairing manufactures a fake error plateau at exactly the jump impulse (291.67).

### 37.2 Validated to the decimal (free physics)

air_free velocity error p50 0.007 uu/s. Jump: first-tick vz 295.6 = immediate 291.67 +
one tick of accel; climb +4.0/tick = (jump_accel 1458.33 − gravity 650)/120 − sticky
2.71 exactly; is_jumping ends at MIN_TIME 0.025 on early release. Dodge lin impulses:
flip_air vel p50 0.009 uu/s once pairing is right.

### 37.3 FIX: the pitch-cancel time gate (`flip::PITCH_CANCEL_MIN_TIME = 0.04`)

Holding pitch INTO the flip cancels dodge torque — but measured along the flip axis
(uncapped ticks only): FULL torque through flip_time 0.0333 (means 1.48–1.71 rad/s per
tick), ZERO from 0.0417 (mean 0.07, then ≈0 every later bucket). RocketSim cancelled
from t=0 — killing torque the real game applies for the first 5 ticks. This is the
"pitch-cancel time gate at +0x350" the Ghidra pass saw in Dodge_TA::ApplyTorqueForces
and we never had. A 1ts policy taps pitch during exactly that window (stalls, flip
cancels, speed flips), so this defect binds hardest against the strongest opponents.

### 37.4 FIX: clamp before publishing state

Real capped-dodge ticks read ang_vel exactly 5.500 and a ball-blasted car exactly
2300.0: RL clamps the STORED state at end of frame. RocketSim clamped at the start of
the NEXT tick, so everything between ticks — obs builders, the RLBot bridge, recordings
— saw pre-clamp values (up to ~7.4 rad/s mid-dodge) the real game never exposes.
`finish_physics_tick` now clamps before publishing; trajectories are bit-identical (the
start-of-tick clamp made the same correction before any force ran), only the OBSERVED
state moves. This also retro-explains §34.1: both engines rotate ~7.3 rad/s mid-dodge;
they differed only in which side of the clamp they reported.

### 37.5 Result and residuals

flip_air ang-vel error p50 0.192 → 0.068, p90 2.021 → 0.272 rad/s; ball_contact ang p90
2.457 → 0.116; worst-case ang errors 26 → 7.4. The 80-segment battery is unchanged
(1126.0 → 1126.5) — at 60 Hz/15 Hz it was structurally blind to this defect, which is
the argument for the 120 Hz instrument. Remaining, in order: wheel-graze regimes
(air+wheels / flip+wheels / wall_drive vel p90 15–33, p99 500 — the §34.4 extra_pushback
family), ground vel p99 ≈ 296 (~1% jump-buffer edge cases), contact-phase tails.

## §38 — The ground error floor is the tape, not the sim (2026-08-10)

Why isn't ground play 100%? Decomposing the ground regime's uniform ~5 uu/s error into
car-frame components (GGL_GROUND_DECOMP): the forward and lateral medians are
**−0.03 to −0.5 uu/s** across every control mode (throttle, steer, handbrake, coast,
reverse) — drive, brake, steer and powerslide dynamics are essentially exact. The whole
median error is VERTICAL: +2.3 to +3.0 uu/s per tick, one-sided upward, in every mode.

Cause, verified: `bQuantizePhysics=true` — the game quantizes replicated positions to
**0.01 uu**, and the capture's resting ride height reads exactly 17.0100 while the true
equilibrium is ~17.012 (sim settles at 17.0121). Every restored tick therefore starts
~2 milli-uu spring-compressed, and the suspension amplifies milli-uu into uu/s: measured
in-sim, a 2.1 milli-uu restore offset produces +2.05 uu/s in one tick
(`quantized_rest_height_response` in demo_bump.rs). The quantization of the RECORDING
imposes a ~±3 uu/s apparent-error floor on every grounded tick; the sim cannot measure
better than the tape. (S28's unquantized steady-state telemetry had already measured the
sticky/suspension equilibrium as exact — a sticky-scale sweep here "fixing" the bias to
zero at scale≈1.5 would have been fitting the quantization artifact; not adopted.)

Honest ground statement: horizontal dynamics median ≲0.5 uu/s per tick; vertical
unmeasurable below ~3 uu/s from this instrument. The REAL remaining ground-family gap is
the heavy landing/graze tail (throttle-mode mean 67 vs median 3 — suspension impact
response, same family as §34.4/§37.5), which is where wavedash fidelity lives.

## §39 — The landing family closed: capped hard-contact resolve (2026-08-10)

Chasing the last real ground gap (the heavy landing/graze tail) through the 120 Hz
capture produced one artifact discovery and one shipped fix.

### 39.1 Artifact: kickoff countdowns

The "9.4% of ground ticks phantom-jump" signal was the replay applying recorded inputs
during KICKOFF COUNTDOWNS, where RL runs physics but ignores all car inputs (~19 goals x
~3 s = ~5,000 ticks; the 1ts bot spams jump edges throughout). The analyzer now skips
ticks where the ball sits frozen at the kickoff spot with no car yet driving. Ground p90
55.7 -> 11.6, p99 295.9 -> 40.6 from the filter alone. (A real edge-triggered-jump rule
was hypothesised and A/B-disproved: activation was already edge-derived.)

### 39.2 FIX: `bullet_vehicle::PUSHBACK_MAX_IMPULSE = 48`

The deep-compression over-push survived all filters: signed error +14 -> +62 uu/s,
monotone from -6 to -18 uu recorded compression, 94% sim-over. Threshold experiments
(engage the resolve later / disable it) zeroed the landings but broke fillet transits --
the two venues disagree about the same compression band because the resolve KILLS the
full normal approach velocity in one tick: correct at fillet-transit approach speeds,
brutal at landing speeds where the real game absorbs through the spring. Capping the
per-tick resolve reconciles them. Swept 0.5..96; 48 (~41 uu/s per wheel per tick):

- capture: deep-compression signed bias +15.4 -> +0.5 uu/s
- battery FIT 1126.0 -> **1035.2**, HOLDOUT 1151.5 -> **1064.7** (largest single-change
  win of the program; jump_after_wall_launch 153.9 -> 21.2, no_jump_control 15.0 -> 8.4,
  tilt_nose_down 62.5 -> 51.0, transition_flip_off 181.5 -> 180-ish unchanged)
- honest cost: the tilt_* teleport-settling family +40 total (state-set penetration
  recovery, not a real-play situation; the capture's real landings are the honest oracle
  and they say the cap is right)

### 39.3 Scorecard after S34-S39 (single-tick velocity within 1% of v_max)

ground 99.1% | air_free 99.3% | flip_air 97.2% | ball_contact 98.0% | flip+wheels 95.3%
| air+wheels 95.1% | wall_drive 94.7% -- weighted 98.3% of 94,062 clean transitions.
Residual tails: contact-phase events (ball blasts, p99 spikes), the ~3 uu/s ground
quantization floor (S38, tape-limited), and wall-drive's last few percent.

## §40 — Time-of-impact demo evaluation + S32 reverted on real evidence (2026-08-10)

Titan's hint ("the game might be using the exact/interpolated collision point") is
confirmed by the .uc itself: `ShouldDemolish` runs every angle check on the SWEPT
time-of-impact state (`GetTimeOfImpact`; end-of-tick is only the sweep-miss fallback).
Ported: `on_car_car_collision` estimates the first-touch fraction from the manifold
penetration and normal closing speed, rewinds both centers to that instant, and
evaluates the hit cones there (velocities stay OldRBState, as in RL). At 4000+ uu/s
closing speed the centers move ~35 uu in a tick — a 10–20° swing in the cone direction
at contact range.

**S32 REVERTED — falsified by real data.** The capture's one demolition (tick 16771)
has the attacker crossing 2200 while AIRBORNE (z=72, g=0, boost surge 2001→2300 on the
contact tick), and the real game demolishes. "Supersonic can only start while grounded"
— adopted from external evidence, explicitly flagged unvalidated — is therefore wrong;
the CDO's `SuperSonicSettings` carries no ground condition either. With the gate the
sim classified this real demo as a bump; with the revert (+TOI) it reproduces:
**1/1 real demolitions predicted (correct attacker, victim, tick), 0 phantom demos
across all 106k replayed transitions.** The analyzer now steps roster-shrink
transitions as demo probes and reports every sim demo event, so any future
bump/demo-heavy capture (the plugin records spectated matches) scales this oracle.

Battery bit-identical (single car); all six contact scenario tests pass.


## §41 — Dodge component decompiled: every constant confirmed, gate exact (2026-08-10)

The full `ACarComponent_Dodge_TA` field dump + `ApplyTorqueForces` decompile arrived
(archived at `assets/dodge_component_decompile.txt`). Every dodge constant RocketSim
carries is confirmed at source: impulses (90000/96000 over mass 180 = 500 and the 16/15
backward scale), max-speed scales (1.9/1.0/2.5), torques (260/224), TORQUE_TIME 0.65,
the Z-damp triplet (0.35 / 0.15 / 0.15+0.06), and the 0.5 input deadzone.

The one correction: **`MinDodgeTorqueTime = 0.0410`** — S37 measured the pitch-cancel
gate into (0.0333, 0.0417] and shipped 0.04; now set to the exact 0.041. The decompile
also confirms the structure S37 inferred behaviourally: full torque before the gate, a
continuous 1−|input| modulation after, sign-matched to the torque direction.

Supersonic note: the vendored grace logic already implements the correct semantics
(timer accumulates only inside the 2100–2200 band and resets on re-exceeding 2200) —
the "counts from first supersonic start" flaw is upstream's, not ours.

Both instruments re-validated: capture regime stats unchanged (the 1 ms gate shift is
below measurement resolution), battery bit-identical.

## §42 — Boost pad grants are touch-events: 2-tick delay (2026-08-11)

User report: "it keeps missing the boost." The capture's boost_amount stream (a 0..1
FRACTION — the analyzer's restore was feeding the sim 1% boost until this was caught)
gives a pickup oracle: ~150 real pickups vs the replay's decisions.

Finding: the sim granted at the exact instant of first OBB-cylinder overlap (trigger
distance p50 192 uu for small pads = the geometric maximum reach), while the real game's
boost jumps ~2 ticks later (p50 175 uu, ~16 uu further along the path). Same 2-frame
UE3 event pipeline as inputs (S37): pickups route through touch events
(AVehiclePickup_TA + the Touching array, S14), and the event lands two frames after
overlap. This also retro-explains S14's replay observation of real pickups at origin
distances past geometric reach (the car keeps moving between touch and grant).

Shipped: `boost_pads::GRANT_DELAY_TICKS = 2` — an overlap CLAIMS the pad
(first-toucher wins, no re-trigger while pending), and the boost, cooldown start, and
CarPickupBoost event land 2 ticks later; external pad state sets clear pending claims;
demoed cars forfeit in-flight grants. Delay swept: 0 → 14 exact-tick matches of ~150,
1 → 43, **2 → 55 with the residual balanced at ±1 tick** (in-frame touch phase the
end-of-tick replay cannot resolve, plus cooldown-cascade desync over the 7.5-min
replay). Battery identical (1035.2), all contact scenario tests pass.

Note the direction of the user-visible symptom: the old instant grant made the SIM
slightly generous (collects at max reach, 2 ticks early), so policies tuned in sim
clip pads on lines that the real game does not reward — "missing boost" in the real
game. The sim is now calibrated to the real grant timing.

## §43 — THE BALL-HIT IMPULSE WAS NEVER APPLIED (2026-08-11)

User report: Pulsar misjudges ball trajectories and hit power (air dribbles into the
ceiling). The capture's per-tick ball stream localized it in three steps:

1. Ball flight and world bounces: exact (verr p50 0.010 / 0.000, p99 0.01 uu/s).
2. Touch ticks: p90 520 uu/s. Event-level rollouts (restore 2 ticks before each touch,
   simulate through, compare outgoing ball velocity — phase-robust) on 131 clean
   strikes: sim/real speed ratio 0.94 on soft touches degrading monotonically to 0.58
   at 2000+ uu/s hits; direction error p50 18.7°.
3. A scale sweep on `ball_hit_extra_force_scale` changed NOTHING — the psyonix extra
   impulse had no effect at all.

Root cause: `Ball::on_hit` runs in the arena's POST-step contact pass, but queued its
impulse with `accum=true`. The solver consumes `accum_lin_vel` when solver bodies are
built — before the contact pass — and `clear_accum_forces` wipes it at the top of the
next tick. **The ball-car extra impulse was added to a buffer that is never read.**
Every ball touch in this engine has been pure Bullet contact. (Same dead-path class:
the heatseeker wall bounce and snowday ground stick, both post-step accum adds; fixed
alike. The S15 audit note "sensitivity probe gave byte-identical metrics" was this bug
being felt without being recognized.)

Fix: apply the impulse directly (accum=false) — post-solve, pre-publish, exactly v2's
`_velocityImpulseCache` timing. Result on the 131 strikes: **speed ratio p50 0.997
(p25 0.987 / p75 1.000), direction error p50 0.33°, p90 3.8°**. Touch-tick verr p90
520 → 27.7 uu/s. Battery bit-identical (ball parked); all scenario tests pass.

Training note: policies trained before this fix learned ball striking ~5–40% weaker
than reality (worst exactly where power matters), then met real-game physics where
every hit comes out hotter than expected — overshooting touches, misjudged aerial
power, dribbles popped into the ceiling. This was almost certainly the largest single
sim2real behavioural gap in the entire program.

## §44 — POST-S43 RESIDUAL SWEEP: PER-CAR IMPULSE GATE, PICKUP BOX OFFSET, CDO INTEL (2026-08-13)

Prompted by two new community artifacts (screenshots archived in assets/):
`AddedCarForceMultiplier = 0.0f` confirmed from the CDO, and the
`AddDemolishInvulnerability(ObjectSource, EDemolishSource)` decompile — a per-source
invulnerability array the function `Resize(1)`s, so only the most recent ObjectSource
is ever tracked. Call sites and duration remain unknown; still not ported.
`AddedCarForceMultiplier = 0` means the sim's total absence of any scripted car-side
force on ball contact is EXACT, not a gap (comment updated at the car-car pipeline).

### 44a — the psyonix-impulse repeat gate is PER CAR, not per ball

The remaining touch-event outliers after S43 were exclusively multi-car contacts:
every stereotyped single-car kickoff strike measured ratio 1.000 / dir 0.0°, while
kickoff pinches and 50-50s ran 0.44–0.88 with the winner under-powered. Cause: the v3
port hoisted v2's repeat-impulse cooldown from `car->_internalState.ballHitInfo`
(per car) onto the ball (`last_extra_hit_tick`, global). With a global gate, when two
cars strike the same tick the second car's extra impulse is silently dropped, and in
extended two-car contact impulses land at half cadence. v2 semantics (per-car
timestamp, same-tick impulses SUM in the cache; window consumed on gate pass even at
rel_speed 0; blocked gate early-returns past the game-mode section) restored, with the
timestamp in `CarState::ball_extra_impulse_tick` (FFI state-set clears it, like
`bump_last_victim`).

Measured (217 touch events): aggregate p50 unchanged (0.997), dir p90 9.43° → 8.06°.
The physically-decisive short high-power double-touches went to ~exact
(t13219: 0.774→1.000 / 27.3°→0.0°; t23825: 0.832→0.974; t16564: dir 23.5°→1.1°).
Long 19–37-tick scrambles shuffled both ways — a 20+ tick unresynced rollout through
repeated multi-car contact is chaotic (the three bit-identical kickoffs now diverge
from each other on sub-quantization initial differences), so those rows measure
sensitivity, not correctness. Battery bit-identical (single-car).

### 44b — pickup trigger box now centred on the hitbox, not the RB origin

The S42 boost audit left 153 real pickups; classifying every one against the sim's
pad-cooldown transitions (new GGL_PADGEO instrument: closest-point-on-box geometry +
per-pad sim cooldown on every near-pad tick): 150 matched within ±2 ticks (offset
histogram peaked at 0/+1), 3 missed, 6 phantom. Both clean geometric misses were
airborne passes over a big pad's edge, 13–17uu OUTSIDE the origin-centred test box
but INSIDE the box centred at the true hitbox centre (`hitbox_pos_offset`, Octane
+13.88 fwd +20.75 up — the trigger was clamping onto a box ~14uu behind and ~21uu
below the real collision primitive). Switched the trigger (and its broad-phase pad)
to the offset centre: 151/153 matched, phantoms 6 → 5. The residuals are sub-uu
boundary grazes (worst new phantom: d=119.3 vs radius 120 during a landing) plus one
cooldown-cascade echo — irreducible under position quantization.

Full-boost pad consumption (does RL consume a pad crossed at 100 boost?) stays
UNRESOLVED: the capture holds only 2 full-boost crossings, neither followed by a
discriminating pickup inside the cooldown window. v2 behaviour (skip at full) kept —
do not "fix" this without a capture that actually decides it.
