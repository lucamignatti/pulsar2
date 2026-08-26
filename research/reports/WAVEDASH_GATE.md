# WAVEDASH_GATE — the dodge-gate arc, its measurement artifacts, and gate v3

**Date**: 2026-08-25/26. **Status**: RESULT (gate v3 shipped; meter v2 shipped).
**Data**: real-game full-pose captures `pulsar-gco-ts1-bot/debug.3914029.jsonl` (67MB,
120Hz, post-mirror-fix, GCO ts1 @470.25B) and `debug.2377799.jsonl`; CrossPlay sim JSONL
(same checkpoint, vs BonkDaddy V4.5); replay probes against the tuned vendored RocketSimV3.

## Executive verdict

**There was never a large sim-vs-real wavedash gap.** With a corrected meter, on identical
weights: real 42.3% wavedash success (n=111), sim legacy-gate 41.3% (n=426). The
"real 36% vs sim 23% vs new-gate 6%" table that drove two engine deployments (one reverted
in production) was built from three stacked measurement artifacts. The genuinely-real
discrepancies found along the way are small and are fixed by **gate v3** (this commit),
which beats legacy on every alignment metric while keeping every battery guard green.

The real-game wavedash degradation the run saw earlier is attributed to the RLBot
client-side state bugs fixed previously (masked ground flag -> mirror; has_flip bug #3;
timers), not to engine physics: the post-mirror-fix capture wavedashes at sim parity.

## The three meter artifacts (wavedash_meter v1)

1. **Logging-phase sp0.** The sim (CrossPlay) JSONL press row's velocity already contains
   the ~500 uu/s dodge impulse; the real capture's press row (packet state) does not.
   v1 measured speed gain from the press row, nulling the criterion for every sim attempt:
   a perfect sim wavedash read dsp ~ -11 while the identical real one read +421.
   Sim success 41.3% measured as 6.7%. Fix: baseline speed 3 rows before the press.
2. **Gate-dependent attempt denominator.** v1 defined attempts by `atsj > 0.02`; atsj
   semantics are exactly what the engine builds under test changed, so each build was
   scored on a different attempt population (64 vs 90 attempts on the same play). This
   manufactured the "new gate made it worse" 23.4% -> 5.6% verdict. Fix: attempts are
   input-edge-defined (ground jump press -> airborne jump press), engine-independent;
   compare worlds binned by press-to-press hop duration.
3. **Speed-cap confound + episode splicing.** Wavedashes executed at 2250+ uu/s cannot
   gain 50 uu/s (counted fail); CrossPlay JSONL restarts t per episode and v1 paired
   presses across resets (negative hop durations). Both fixed in meter v2.

Add to the trap list next to denominator drift: **compare like-phased logs** — know at
which point in the tick each world's velocity was sampled before differencing them.

## What is genuinely different (per-tick real traces), and gate v3

Real-game per-tick traces (wavedash_trace.py) establish the game's rules; wheel contact
has three decoupled effects, where legacy RocketSim bundled all three into one ground
early-return:

1. **Presses**: eaten by wheel contact, always — the jump state does NOT override.
   (A during-jump press at z=26.8 fired because contact had broken; masked-ground mash
   presses were eaten. Fire-tick alignment on 29 clean events: legacy 21/29, a
   during-jump press escape 18/29 — the escape is wrong.) Legacy had this right.
2. **Window/state**: contact during takeoff does NOT reset the dodge window or flip
   state — the packet's dodge_timeout counts straight through an AirState
   Jumping->OnGround->InAir flicker. Reset happens only with the landing reset that
   clears has_jumped. Legacy reset on every contact tick (wrong).
3. **Flip cancel**: contact cancels active flip torque only on descent (the wavedash
   flatten); ascent contact keeps the dodge alive. Legacy cancelled on any contact
   (wrong). NOTE: only the cancel is direction-gated; air control/throttle stay gated
   by is_on_ground exactly as legacy — routing ascent-contact ticks into
   update_air_torque applied air throttle on every jump's sticky ticks and doubled
   replay error (measured, 2.4u -> 4.9u at t+5).

Gate v3 = legacy press gating + fixes for #2/#3 + an airborne-only during-jump escape in
flip_delay (atsj is pinned 0 while is_jumping; press_air_ok keeps it airborne-only).
`GGL_LEGACY_DODGE_GATE=1` restores the full legacy bundle.

**v3 vs legacy, all measured** (earlypress_probe.py, fidelity_battery.py):

| metric | legacy | v3 |
|---|---|---|
| fire-tick agreement ±1 (29 clean events) | 21 | **22** |
| missed real fires | 2 | **1** |
| spurious sim fires | 0 | 0 |
| early-press fire coverage (100 events) | 80 both / 9 real-only | **86 / 3** |
| probe pos err t+5 | 2.4u | **1.9u** |
| battery JUMP t+2 / t+20 | 1.0u / 9.1u | 1.0u / 9.1u |
| battery CANCEL, HBRAKE | — | bit-identical |
| corrected-meter sim wavedash success | 41.3% | (v3 arm: see below) |

Gate v1's post-mortem (deployed to the fleet 2026-08-25, reverted same day): its two
premises — "96% of early presses are free no-ops" and "the window counts from jump start"
— were both artifacts of the harness (hardcoded isJumping=0 restores; atsj-defined
attempts). In live sim play the policy's re-press edges land after contact breaks, so
legacy already fired 94.7% of them.

## Residual known gap (small, unfixed, documented)

Sim wheel-contact persists 1-3 ticks longer than the game's dodge-gating contact query
during takeoff (sim grounded to z~31-33 vs real dodge-free by z~26.8; Ghidra shows RL
keeps three distinct contact queries — IsOnGround / GetNumWheelContacts /
GetNumWheelWorldContacts — where RocketSim has one). Effect: a press in that 1-3 tick
boundary is eaten in sim, fires in real (1 missed fire of 29). Fixing it means a
second, shorter-reach contact criterion for the press gate; boundary evidence is n=2
events, too thin to calibrate. Revisit only with more boundary captures.

## Dark domains: closed at the capture level

The two never-measured regimes (ball-touch physics, car-car contact) were dark because
the capture lacked ball angVel and opponent pose. Both now logged per decision
(`bav`, `op/ov/of/ou/oav` — nearest opponent full pose) in RLBotClient. The next long
real-game session makes both regimes scoreable.

## Addendum 2026-08-26: the boundary hypothesis refuted; full parity table

The residual-gap section above suspected sim wheel contact outlives the real dodge
query by 1-3 ticks. A 661-press real calibration curve (fire-rate 0% below z~21, 50%
at z~27.5, ~100% from z~31) appeared to confirm it - but replaying all 625 directional
press edges through the engine (dodge_gate_sweep.py), the EXISTING is_on_ground gate
matches the real curve at **91.8% per-event agreement** (mean bin error 3.1). Tilted
real poses drop below 3-wheel contact exactly where the real game frees the dodge;
the flat-ground level-car smoke test that motivated the hypothesis is the pathological
case, not the typical one. GGL_DODGE_CONTACT_EXT ships default -1 (disabled).

Also closed the same day:
- **DODGE_roll impulse deficit** (battery: sim right-dv 163.9 vs real 254.4) was a
  fire-tick misalignment in the battery harness. Timing-aligned per event: sim matches
  real to ~2% (fwd 337.1/336.6, right 435.5/443.4, n=81).
- **Dodge direction**: median angle error 0.0deg in every input class (n=717; an
  apparent 177.6deg roll inversion was a cross-product order bug in the analysis).
- **Meter v3**: zmax/rotation measured only until reground; the 0.8s window had been
  misclassifying successful wavedashes whose follow-up left the ground.

**Final parity, identical weights (528B ts1, meter v3):**

| metric | real | sim (gate v3) |
|---|---|---|
| wavedash success | 57.7 / 58.8 / 60.1% (3 captures) | 51.2% |
| FULLFLIP failures | 8.7-11.8% | 6.5% |
| HIGH-hop failures | 17.6-21.6% | 27.3% |
| NOREGROUND / NOSPEED | ~5% / ~6-7% | 8.4% / 6.6% |
| press-gate fire curve | - | 91.8% per-event agreement |
| dodge direction | exact | exact |
| dodge impulse (rolled) | 336.6 / 443.4 | 337.1 / 435.5 |

The failure modes the run's operator observed in-game (full flips instead of
wavedashes, the back lifting on forward wavedashes, weak diagonals) occur in sim at
comparable rates: they are POLICY skill, visible in viz, and trainable. Remaining
open: DBLJUMP deep-horizon tail (n=3 clean events, not attributable until the
ball-touch/car-car dark-domain captures land).
