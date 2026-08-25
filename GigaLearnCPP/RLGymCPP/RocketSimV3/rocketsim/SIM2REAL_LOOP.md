# 1-tick RLPR vs rocketsim-v3-tuned

Metric: restore-then-1-step, `GGL_CTRL_LAG=2`, 120 Hz only. Fit `[0,25000)`.
Ignore kilo-uu/s demo/teleport maxes. Score tape 2 **by car**.

## Tapes

All under Bakkes `data/rlrecord3/` (v4 recorder, contact resampled one tick later +
non-local input fix). 120 Hz only (`physics_frame` Δ median=1, airborne `|Δvz|`
p50 ≈ 5.42). Bakkes zeroed hitbox ⇒ `GGL_CTRL_LAG=2`.

Old v2 captures remain in `data/rlrecord2/` as archive. Do not score them as
ground truth anymore.

- **T1 live** `rlrecord3/autosave_20260824_174851.rlpr` — v4 recorder, two Octanes, both skip-1, spectating
- **T1 archive** `rlrecord2/autosave_20260809_212407.rlpr` — same job, v2 recorder (no contact resample). Keep for tape-vs-tape A/B; not the live GT.
- T6 `rlrecord3/autosave_20260824_175720.rlpr` — solo Octane wall-drive (local). Order: floor→wall fillet; wall circles, drift circles, boost+drift circles; ceiling fillet; straight wall; wall-backboard fillet; corner fillet drive+drift; air→wall contacts. Replaces archive T6+T7.
- T2–T5, T7 — **not recaptured yet** (old files in `rlrecord2/`)

Ignore 60 fps RLBot/UncappedFramerate tapes. See `.cursor/rules/rl-120hz-tapes.mdc`.

## Current shipped defaults (2026-08-24)

Compiled-in, no env vars. Opt-out knobs in parentheses. Live T1 numbers are
the **2026-08-24 19:00** table below (`WHEELS_PRE` on). The 3.39 table further
down is the historical post-step / PRE-off floor.

**On (training physics):**
- Wheel stack: `WHEELS_PRE`, `APPLY_TIME`, `NO_DAMP_FADE`, `TUNED_RAY`,
  `STEER_AT_APPLY`; uncapped pushback (`PUSHBACK_CAP=0`)
- Sticky prev-tick gate (`STICKY_PREV`; `=0` restores fresh-ray)
- Auto-roll `noflip` (`GGL_AUTOROLL=1` restores v2, `0` kills it)
- Air throttle off while any wheel is in contact (`GGL_AIR_THROTTLE_WHEELS=1` restores stacking)
- Ground boost accel while any wheel is in contact (`GGL_BOOST_AIR_WHEELS=1` restores air accel on 1–2 wheels)
- Car-car manifold persist (`GGL_CC_PERSIST=0` rebuilds every tick)
- SAT closest-axis miss→hit **0.2 uu** (`GGL_SAT_SLACK`; `=0` disables). Slack-only
  manifolds skip the bump/demo cone (real overlap still bumps).
- Apply-time extra_pushback clear each tick, then recompute (T6 +6.84 was stale persist)
- Ball-world special contact normal is unit-length (`(avg).normalize()`, PR74 `22cf396`; `GGL_NO_NORM=1` restores the unnormalized average)

**Off (tested, do not enable for training):**
- `GGL_WHEEL_ON_CAR` — T5 roof-ride, p90 worse
- `GGL_SAT_SLACK=0.3` or ≥1 — T5 fly-by bump spikes; 0.2 + no-cone is the ship
- `GGL_CC_CLIP_FB` / `GGL_SLACK_CD` / `GGL_CC_INFLATE` — clip fallback and inflate A/Bs
- `GGL_CC_INFLATE` / `GGL_CC_BAUMGARTE` / `GGL_CC_DEPTH` / `GGL_CC_GEN` — inflate/solver A/Bs
- `GGL_WHEELS_PRE=0` etc. — undoes the 22:40 winning stack
- `GGL_PUSHBACK_MINPEN` — skips the live recompute on shallow walls; T1/T7 p90 tax.
  The T6 win is the clear above, not this gate.
- `GGL_PROJ_AXLE` — project friction axle onto contact plane; air+wheels wrecked
- `GGL_SUSP_UP` — spring/pushback along car-up; packed fillets need force along *n*
- `GGL_FILLET_DRIVE` — scale engine force on packed+tilted extra_pushback; 1-step
  is not over-drive (packed thr+ worse, uncompressed fillet p90 0.89→4.12)
- `GGL_VEL_CLAMP=head` — tuned head-only speed cap; live T1 flip_air ang 0.00→0.15.
  Default remains `both` (tick head + publish). `end` is 1-tick-identical to `both`.

**Leftovers (open):**
1. Car-car SAT generation — 0.2 uu slack + bump-cone skip shipped 2026-08-24
   (v5 1v1 CarImpact p90 26→14). Remaining: wrong-direction real hits, clip-empty
   promotions, resid-dominant (not car-car). `GGL_TAPE_N` off. Packed ω/dropout
   and post/crossbar contact-n vs mesh are still open.
2. Roof-ride — wheel-on-car rays fire (`wc>0`) before chassis SAT; Newton pair rejected
3. Persist warmstart (`applied_impulse` carry) — thin slice of T5 error
4. Packed+tilted fillet transit — 4-wheel, `susp<-6` and `|uz|≳0.25`. T7 **0.89 / 3.39**,
   T1 opp-200 **0.51 / 3.80** (12% of T1 wall ticks, ~42% of wall error mass).
   Uncompressed true wall is already ~0.02 / 0.78. Contact-law A/Bs all reject;
   typical 1-tick leftover is `WheelsSuspension` along faceted fillet *n*, not drive.
5. Ball-world **floor-fillet bounce** — quiet ground/air/true-wall are ~0.01. T5
   world-only mass is 44% of that tape and ~80% of it is wall with `z≲370`
   (`verr≥1` n=153, max 38). T1/T2 `sum(verr)` is ~95% car hits; do not fit
   those air maxes.
6. Car–ball **aerial roof** — hitbox +z, not world-up. T1: `roof|flip` is 4% of
   touch ticks / **52% of touch mass** (mean 158); sim *does* emit `CarHitBall`
   (`flip|hit` 61%) so extra/impulse is wrong. T2: roof 85% of touch mass, mostly
   not-flipping, **sim-miss 59%**. Underside is ~0.
7. Car–ball **ground side-edge 50/50s** — T5: left+right 67% of touch mass,
   edge+corner 99.9%, grounded, tape-touch, **sim-miss 69%**. Nose dribbles are
   already cheap (mean 2.5).

## Live baseline 2026-08-24 19:00 (current crate, `WHEELS_PRE` on)

Same harness both files: `analyze_rlpr`, `GGL_CTRL_LAG=2`, window `[0,25000)`,
1-tick restore, **no** `GGL_OPP_FILTER`, **no** `GGL_WHEELS_PRE=0`. Ignore
kilo-uu/s demo/teleport maxes. p50 / p90, unfiltered pooled.

| car vel (uu/s) | T1 archive (v2) | T1 live (v4) |
|---|---:|---:|
| ground | 0.019 / 0.435 | 0.024 / 0.487 |
| wall_drive | 0.052 / 0.874 | 0.105 / 1.232 |
| air+wheels | 0.082 / 2.225 | 0.080 / 1.013 |
| flip+wheels | 0.075 / 1.301 | 0.062 / 1.068 |
| air_free | 0.007 / 0.014 | 0.007 / 0.013 |
| flip_air | 0.007 / 0.136 | 0.007 / 0.493 |

Air+wheels 0.082 on the archive file is the 09:45 unfiltered number (not 0.540).
v4 vs v2 is not a p50 win on ground; wall p50 is worse on the recapture (0.052 →
0.105). Air+wheels p90 drops 2.23 → 1.01.

Ball surfaces (vel p50 / p90; maxes are car hits):

| ball | T1 archive | T1 live |
|---|---:|---:|
| ground | 0.001 / 0.010 | 0.000 / 0.009 |
| air | 0.008 / 0.011 | 0.008 / 0.012 |
| wall | 0.008 / 0.012 | 0.007 / 0.010 |
| post | 0.006 / 0.009 (n=23) | 0.000 / 0.009 |
| crossbar | n=0 | 0.000 / 0.009 |

Car–ball mass split differs (same engine, different session): archive
`roof\|flip` 52% of touch mass / sim-hit 71%; live `nose\|flip` miss 44% /
sim-miss 78%. Do not mix those leftovers.

**Fit target going forward:** T1 live (`174851`). Archive T1 is the A/B for
recorder alignment only.

## Iteration 2026-08-24 19:15 (vel clamp head vs end, live T1)

Tuned default is clamp at tick **head** and skip publish. We ship **both** (S37
publish + existing arena head). Knob: `GGL_VEL_CLAMP=head|end|both` (default both).

Live T1 `[0,25000)`, PRE on, lag 2. p50 / p90:

| | both (shipped) | head (tuned) | end |
|---|---:|---:|---:|
| ground vel | 0.024 / 0.487 | 0.024 / 0.511 | **same as both** |
| wall vel | 0.105 / 1.232 | 0.105 / 1.232 | same |
| air+wheels vel | 0.080 / 1.013 | 0.083 / **1.481** | same |
| flip+wheels vel | 0.062 / 1.068 | 0.077 / **2.228** | same |
| flip_air **ang** | 0.000 / 0.110 | **0.151 / 1.991** | same |
| air_free ang | 0.000 / 0.144 | 0.068 / 0.183 | same |

**Reject `head`.** Publish clamp is S37: dodges read 5.5 in the tape; head-only
leaves sim ω above the cap for the compared tick. `end` vs `both` is a no-op on
1-tick GT restore (tape vel is already capped, so the head clamp never fires).
Keep `both`. Do not pull tuned's head-only default.

## Historical scoreboard (keep-stack, no WHEELS_PRE)

Post-step floor from early A/Bs. `GGL_WHEELS_PRE=0` on live T1 still returns
here (~2.5 ground / ~3.5 wall). Not the training default.

| | T1 pooled | T2 c0 (skip-1) | T2 c1 (skip-4) | Tuned T1 |
|---|---:|---:|---:|---:|
| ground | 3.39 | 3.38 | 4.81 | **0.032** |
| wall | 3.79 | 4.44 | 7.13 | **0.081** |
| air+wheels | 3.25 | 2.73 | 3.58 | **0.57** |
| flip+wheels | 3.09 | 3.70 | 8.79 | **0.12** |
| air/flip_air | ~0 | ~0 | ~0 | ~0 |

Tuned T2 **ground ~6.8** (worse than us) while owning `<1` pile — T1-specialized, same 1-tick GT-pose apply.

## Mechanisms

1. **Pre-step `update_vehicle_second` + end-of-tick raycast reuse** (tuned default).
   1-tick teacher-force applies wheels at the GT pose. Recreates T1 `<1` pile (`GGL_WHEELS_PRE`). **Hurts T2 pooled p50**. Do not ship as default. Fair skip-1 skill is post-step.
2. **Sticky 0.5 g on flat** → ~2.7 uu/s up, matches T1 ground median (~3.4, up-dominated). Scale 2 / last-contact sticky: T1 `<1` up, T2 bimodal — reject.
3. **Skip-4 car** inflates T2 wall/flip pooled. Do not fit physics to c1.
4. **Tuned autoroll default 0/0** (v2 scales 100/80 off). Ours still runs autoroll. Prior T2 A/B pooled: off **hurts** air+wheels/flip. Re-check T1 + T2-c0.
5. **Drive /4 with <3 wheels** — T2 air+wheels is fwd-short; divisor 2/1 **hurts**. Keep 4.
6. **Apply-time friction** without PRE: no-op (we already write engine_force before post-step second).

## Keep (honest)

Refresh contact after `set_car_state`, restore `prev_controls`, Bakkes `has_flip` invert, integer flip clocks, `FLIP_MIN_DELAY = TICK_TIME`, pre-inc flip Z-damp, raw lateral axle, boost restore, Bakkes flip dir.

## Reject

Sticky-from-last, fric lever-at-contact, chassis-suppress off, PR72 skip dodge on wheels, PR66 ball pen, restore susp overlay, sticky×2, autoroll-off (pooled T2), skip wall sticky tilt, WHEELS_PRE default, DRIVE_PARTIAL 1/2, PUSHBACK DIV/CAP/ERP, INV_DOT_CAP, LAT_LOADCAP, PROJ_AXLE, SUSP_UP, FILLET_DRIVE, NO_STICKY_TILT, STICKY_SCALE=0.5.

## Iteration 2026-08-23 20:30

**Autoroll off** (tuned default 0/0): T1 air+wheels 3.25→3.30 worse; T2-c0 air+wheels 2.73→3.02 worse. **Reject.** Our v2 autoroll is better on these 120 Hz tapes than tuned’s zeroed scales.

**WHEELS_PRE by car (T2):** skip-1 **benefits**, skip-4 **pays**. That is tuned’s T1 win.

| T2 | keep c0 | PRE c0 | keep c1 | PRE c1 |
|---|---:|---:|---:|---:|
| ground | 3.38 | **0.49** | 4.81 | 6.84 |
| wall | 4.44 | **1.84** | 7.13 | 9.79 |
| air+wheels | 2.73 | 2.57 | 3.58 | 3.69 |
| flip+wheels | 3.70 | 3.60 | 8.79 | **11.84** |

PRE pooled T2 ground 6.82 ≈ tuned T2 ground ~6.8. Same 1-tick GT-pose apply, same T2 skip-4 tax.

**Do not ship PRE.** Score skip-1 post-step. Honest leftover: ground ~3.4 (up/sticky), wall ~4.4 (up tails).

## Iteration 2026-08-23 20:32 (loop tick 1)

Skip-1 leftover is still ground ~3.4 / wall ~4.4. Sticky family A/B (no PRE):

| knob | T1 ground | T1 wall | T1 air+w | T2-c0 ground | T2-c0 wall | T2-c0 air+w |
|---|---:|---:|---:|---:|---:|---:|
| keep | 3.39 | 3.79 | 3.25 | 3.38 | 4.44 | 2.73 |
| STICKY×2 | **2.62** | 3.29* | **4.25** | **2.88** | **6.06** | **3.45** |
| NO_TILT | 3.39 | **6.11** | 3.73 | 3.40 | **6.37** | 3.21 |
| STICKY_PREV | 3.39 | 3.79 | 3.25 | (same as keep if T2 matches) | | |

\*T1 wall pooled drop is c1 only (3.65→2.09); **c0 wall 4.27→6.88**. Skip-1 wall gets worse.

**Reject all three.** Extra suction helps flat ground a bit and wrecks partial-contact / skip-1 walls. Prev-tick gate is a no-op after `refresh_contact` on 1-tick restore.

## Iteration 2026-08-23 20:36 (loop tick 2)

Wall skip-1 is **not** a 3-wheel problem. Mass is 4 wheels on a true wall; **fillet/ceiling** is the fat tail.

T2-c0 wall keep: pooled 4.44. `nw=3` 4.37 (n=79) vs `nw=4` 4.46 (n=1139).
`|n.z|<0.2` **3.66** (n=825) / `<0.5` **7.67** (n=303) / `<0.7` **9.33** (n=90).

T1-c0: `nw=4` 4.49 vs `nw=3` 2.06; `|n.z|<0.7` **8.94**. Same fillet tax.

T2-c1 (skip-4): `nw=3` **9.51** vs `nw=4` 6.26 — partial contact + held inputs, not a skip-1 physics miss.

`GGL_APPLY_TIME=1` T2: bit-identical to keep on c0 wall/nz bins. Post-step batch susp is a no-op.

Skip-1 air+wheels: 1-wheel slightly worse than 2-wheel (T2-c0 3.38 vs 2.71) — not the /4 drive story (already rejected).

## Iteration 2026-08-23 20:40 (loop tick 3)

Fillet/ceiling is **under-supported if we drop pushback, over-kicked if we uncap**. Keep S39 cap 48.

T2-c0 vs keep (wall 4.44, `|n.z|` 3.66 / 7.67 / 9.33, ground 3.38):

| knob | wall | nz<0.2 | fillet | ceiling | ground |
|---|---:|---:|---:|---:|---:|
| NO_PUSHBACK | 4.83 | 3.33 | **8.64** | **11.62** | **2.87** |
| UNCAP (tuned) | **5.39** | 4.20 | **10.87** | **12.51** | 3.44 |
| NO_DAMP_FADE | 4.51 | 3.71 | 7.85 | 9.33 | 3.54 |

T1-c0 same pattern: no-pushback ground **2.42** but wall/ceiling worse; uncap wall **5.58**, ceiling **14.4**.

**Reject all three for skip-1 walls.** Pushback cap is load-bearing on fillets. Ground “win” without pushback is the same suction/support trade as sticky×2.

Tuned’s uncapped pushback is **not** their 1-tick wall win; that remains PRE-step apply.

## Iteration 2026-08-23 20:46 (loop tick 4)

Fillet **geometry** vs tuned: they have an optional 3-ray wheel sweep (default **off**) and no 2uu detect extra.

T2-c0 keep: wall 4.44, fillet 7.67, ceiling 9.33.

| knob | wall | fillet | ceiling | ground |
|---|---:|---:|---:|---:|
| WHEEL_SWEEP=0.4 | 4.70 | 8.00 | **10.86** | 3.41 |
| NO_DETECT_EXTRA | 4.85 | 8.00 | 9.73 | 3.54 |

**Reject both.** Thin-ray + 2uu extra is better on this 120 Hz tape than disc-sweep or tuned’s shorter ray. Fillet ~7.7 is not “we miss the curve with one ray.”

Stuck on skip-1 fillet after sticky / pushback / damp / ray shape. Next: leave wall; try skip-1 **ground up** (Bullet chassis contact / world friction) or skip-1 flip+wheels 3.7.

Sweep stays **off** (env only).

## Iteration 2026-08-23 21:00 (plank, not skip-4)

T2 `.meta.txt`: c0 `body_id=23` Octane, c1 `body_id=1919` **Plank (Batmobile)**. Harness had spawned Octane for both. Bakkes header zeros hitbox so the `.rlpr` cannot name the body.

`analyze_rlpr` now reads sibling `.meta.txt` (`23`→Octane, `1919`→Plank) or `GGL_CAR_BODIES=octane,plank`. T1 both 23, unchanged.

T2 fit, c1 Octane vs Plank (c0 stays Octane; skip-1 score is c0):

| T2 c1 | octane (wrong) | **plank** |
|---|---:|---:|
| ground | 4.81 | **3.42** |
| wall | 7.13 | **5.90** |
| air+wheels | 3.58 | **2.36** |
| flip+wheels | 8.79 | 8.28 |
| wall nz<0.2 | 5.73 | **4.50** |
| ball_contact | 2.70 | **0.80** |

c1 ground now matches skip-1 Octane (~3.4). The old T2 pooled ground 4.35 was the Octane-on-Plank tax, not skip-4 physics.

T2-c0 unchanged (ground 3.38, wall 4.44, flip+wheels 3.70).

Loop chassis/dodge A/Bs on Octane (no PRE): `NO_CHASSIS_SUPPRESS` no-op on skip-1 p50; `DODGE_NEEDS_AIR` hurts T1 flip+wheels 3.09→3.48; `FLIP_THEN_AIR` hurts air+wheels; `STICKY=0` wrecks skip-1 wall 3.79→8.04. Keep stack.

New Octane tapes: score those cars only. T2-c1 is a Plank opponent, not an Octane holdout.

## Iteration 2026-08-23 21:10 (floor-only sticky)

Skip-1 leftover is **up / under-sticky**: T1 ground signed-up p50 **+2.43** (~0.5·g·dt); throttle fwd ~0.06.

`GGL_STICKY_SCALE=1.25` (global): T1 ground 3.39→3.09, wall 3.79→3.24, but air/flip pay (3.25→3.34 / 3.09→3.35). 1.5 taxes air/flip more.

`GGL_STICKY_FLAT` (SCALE only when `|up.z|>0.9`; default 1, not shipped): walls **bit-identical**. Ground matches global 1.25/1.5. Air/flip **still** pay (T1 FLAT=1.25 air 3.25→3.36, flip 3.09→3.34) — extra suction on wheels-down floor during those regimes, not a wall leak.

**Keep SCALE=1.** Floor extra is not free. Next skip-1 lever is flip+wheels (~3.1–3.7), not more sticky.

## Iteration 2026-08-23 21:13 (flip+wheels)

T1 skip-1 flip+wheels p50 **3.09** (c0 3.24). Decomp mixed (not sticky-up).

| knob | flip+wheels | notes |
|---|---:|---|
| keep | 3.09 | Bakkes torque 260/224 |
| TORQUE=0.85 | **3.40** | c0 3.78 |
| TORQUE=1.15 | 3.15 | c0 3.39 |
| NO_FLIP_ZDAMP | **4.86** | Z-damp is load-bearing |
| DODGE_VEL ±15% | 3.09 | 1-tick restore already has the kick; **hurts air+wheels** 3.25→3.75 |
| DODGE_DAMP | 3.11 | no-op |
| AXIS_SPIN | 3.09 | bit-identical |

**Reject all.** Flip+wheels 1-tick error is wheel/contact interaction at the GT pose, same family as PRE — do not ship PRE to “win” it. Keep torque/zdamp/impulse.

Skip-1 Octane leftover stays ground ~3.4 (up/sticky), wall ~4.4 (fillet), flip+wheels ~3.1. New Octane tapes are the holdout.

## Iteration 2026-08-23 21:24 (loop re-arm)

Highest leftover is still skip-1 **linear** vs tuned T1 `<1` pile. Angular is already ~0.02.

Mechanisms A/B'd (no PRE):

| knob | T1 ground | T1 wall | T1 air+w | T1 flip+w |
|---|---:|---:|---:|---:|
| keep | 3.39 | 3.79 | 3.30 | 3.09 |
| CHASSIS_REST0 | 3.39 | 3.79 | 3.30 | 3.09 |
| WORLD_REST0 | 3.39 | 3.79 | 3.32 | **3.16** worse |
| SOLVER_ITERS=4 | 3.39 | 3.79 | 3.30 | 3.09 |
| RESTORE_WORLD | 3.39 | 3.79 | 3.30 | 3.09 |

**Reject** chassis/world restitution, solver-4, restore-world. Ground float is **not** a chassis bounce. Tuned's 4 solver iters are a no-op on this 1-tick metric.

## Iteration 2026-08-23 21:32 (tuned wheel *order* + apply-time)

Tuned's remaining T1 win is the discrete vehicle scheme, not sticky/ERP/solver.

| T1 stack | ground | wall | air+w | flip+w | ground `<1` |
|---|---:|---:|---:|---:|---:|
| keep | 3.39 | 3.79 | 3.30 | 3.09 | 2.1% |
| PRE | 0.331 | 0.521 | **4.29** | 2.58 | 63% |
| PRE+APPLY_TIME | **0.063** | **0.157** | 3.84 | 2.03 | 72.5% |
| PRE+STICKY_PREV | 0.331 | 0.521 | 4.29 | 2.58 | (= PRE) |
| PRE+NO_DAMP (ray-time) | 0.321 | 0.497 | **2.55** | **1.58** | 64.5% |
| **PRE+APPLY+NO_DAMP** (damp gated at apply) | **0.059** | **0.139** | **1.49** | **0.366** | **75.6%** |
| tuned T1 | **0.032** | **0.081** | **0.57** | **0.12** | ~pile |

APPLY_TIME without PRE was previously a no-op. With PRE it is most of the leftover 0.33→0.06. Graze-damp fade (S39 companion) was **hurting** PRE air/flip; it was a no-op on the APPLY_TIME path until `compute_suspension_impulse` respected `GGL_NO_DAMP_FADE`.

**Do not ship PRE as default yet** (T2 skip-4 tax still). Experimental stack: `GGL_WHEELS_PRE=1 GGL_APPLY_TIME=1 GGL_NO_DAMP_FADE=1`.

Honest skip-1 vs tuned T1: ground 0.059 vs 0.032, wall 0.139 vs 0.081, air 1.49 vs 0.57, flip 0.37 vs 0.12. Same scheme, we are close on ground/wall; air/flip still ~2–3×.

T2 with the same stack (c0 skip-1 / c1 Plank skip-4):

| | keep c0 | combo c0 | keep c1 | combo c1 |
|---|---:|---:|---:|---:|
| ground | 3.38 | **0.069** | 3.42 | **0.177** |
| wall | 4.44 | **0.675** | 5.90 | 7.52 |
| air+wheels | 2.73 | **0.99** | 2.36 | **0.80** |
| flip+wheels | 3.70 | **0.64** | 8.28 | 7.63 |

PRE-only skip-4 ground tax (6.84) **disappears** once apply-time + no graze-fade are on. Remaining skip-4 cost is wall. Pooled T2 ground **0.135** vs tuned T2 ~6.8.

`GGL_STICKY_PREV` + PRE is bit-identical to PRE. Reject (dup section).

## Iteration 2026-08-23 21:38 (air leftover on combo)

Combo T1 leftover is **air+wheels 1.49** (c0 0.995 / **c1 2.33**) vs tuned 0.57. Flip 0.37 vs 0.12.

On top of combo (`PRE+APPLY+NO_DAMP`):

| knob | ground | wall | air+w | flip+w |
|---|---:|---:|---:|---:|
| combo | 0.059 | 0.139 | 1.49 | 0.366 |
| NO_AUTOROLL | 0.059 | 0.139 | 1.49 | 0.366 |
| STICKY=0 | **2.72** | **7.91** | **2.86** | **2.72** |
| DRIVE_PARTIAL=1 | 0.058 | 0.128 | **1.74** | 0.355 |
| AIR_WITH_WHEELS | 0.058 | 0.128 | **1.64** | **1.50** |

Reject all four. Tuned autoroll-off is a **no-op** once wheels apply pre-step. Sticky is load-bearing on the combo (not the air miss). Full drive on 1–2 wheels and air-torque-while-wheels both **hurt** air/flip.

Air+wheels split on combo: **1-wheel** is the leftover (c0 nw=1 p50 **2.02**, c1 **2.85**); 2-wheel is already near tuned (c0 **0.61**, c1 **1.19**).

| knob | ground | wall | air+w | flip+w | c1 nw=1 |
|---|---:|---:|---:|---:|---:|
| combo | 0.059 | 0.139 | 1.49 | 0.366 | 2.85 |
| DRIVE_PARTIAL=8 | 0.058 | 0.128 | **1.42** | 0.355 | 2.24 |
| STICKY_MIN_WHEELS=2 | 0.058 | 0.128 | **2.71** | **2.70** | **4.13** |

Weaker 1–2 wheel drive is a tiny air win; not worth the extra knob vs tuned. Dropping sticky on 1-wheel **hurts** that bin — 1-wheel needs suction, it is not over-stuck.

## Iteration 2026-08-23 21:47 (1-wheel apply-time pushback)

Hypothesis: `extra_pushback / 4` under-supports a single contact. Combo T1 + apply-time:

| knob | ground | wall | air+w | flip+w | c1 nw=1 |
|---|---:|---:|---:|---:|---:|
| combo (/4 always) | 0.059 | 0.139 | 1.49 | 0.366 | 2.85 |
| PUSHBACK_NCONTACT (/n) | 0.058 | 0.129 | **2.13** | 0.410 | **3.04** |
| PUSHBACK_MIN_WHEELS=3 (0 if <3) | 0.059 | 0.128 | **2.71** | 0.597 | **3.03** |

**Reject both.** 1-wheel *needs* the /4 share. More pushback or dropping it both inflate air. Ground/wall unchanged (still 4-wheel /4). Leave `/ NUM_WHEELS`. 1-wheel leftover is not hard-contact resolve; next is 1-wheel **friction / suspension spring** (lat/long or 1/cos), not pushback.

| | keep c0 | combo c0 | keep c1 | combo c1 |
|---|---:|---:|---:|---:|
| ground | 3.38 | **0.069** | 3.42 | **0.177** |
| wall | 4.44 | **0.675** | 5.90 | 7.52 |
| air+wheels | 2.73 | **0.99** | 2.36 | **0.80** |
| flip+wheels | 3.70 | **0.64** | 8.28 | 7.63 |

PRE-only skip-4 ground tax (6.84) **disappears** once apply-time + no graze-fade are on. Remaining skip-4 cost is wall. Pooled T2 ground **0.135** vs tuned T2 ~6.8.

`GGL_STICKY_PREV` + PRE is bit-identical to PRE. Reject.

## Iteration 2026-08-23 22:00 (read tuned's source directly — three ports, board swept)

Tuned is Rust too. Its *shipping* defaults vs ours exposed three structural diffs:

1. **`GGL_TUNED_RAY`** — tuned's wheel ray is exactly `rest+travel+radius` (susp_subtract
   ships 0, no 2uu detect-extra) and **every hit gives full suspension+friction**; our
   adhesion-only band gives grazing wheels zero forces. (Old `NO_DETECT_EXTRA` reject was
   only the *shorter* ray, not the full-forces model.)
2. **`GGL_STEER_AT_APPLY`** — tuned recomputes the friction axle from THIS tick's
   `steer_angle` at apply time (ships on, "+1.42pp, zero regressions" per their notes);
   ours used the raycast-time axle = steering lagged friction by 1 tick.
3. **Pushback uncap** (`GGL_PUSHBACK_CAP=0`) — tuned has no S39-style cap. Uncap was
   rejected on the post-step stack (over-kick), but on the apply-time stack the resolve
   sees post-suspension velocities and the cap was *causing* the tails.

T1 stacking on combo (`WHEELS_PRE=1 APPLY_TIME=1 NO_DAMP_FADE=1`):

| stack | ground | wall | air+w | flip+w | c1 nw=1 |
|---|---:|---:|---:|---:|---:|
| combo | 0.059 | 0.139 | 1.49 | 0.366 | 2.85 |
| +TUNED_RAY | 0.057 | 0.125 | 1.07 | 0.263 | 2.70 |
| +STEER only | 0.031 | 0.087 | 0.702 | 0.210 | 2.71 |
| +both | 0.031 | 0.085 | 0.601 | 0.173 | 1.99 |
| +both+UNCAP | **0.031** | **0.076** | **0.540** | **0.115** | **1.71** |
| tuned T1 | 0.032 | 0.081 | 0.57 | 0.12 | — |

**We now beat tuned on every T1 regime**, including tails (wall p99 45.7→11.9, ground
p99 21→8.6, air p99 150→77.7).

T2 same stack: c0 (skip-1) ground **0.043** / wall **0.269** / air+w **0.457** / flip+w
**0.173**; c1 Plank skip-4 wall 4.54→**0.697**. Tuned T2 ground ~6.8. Across the board.

Also checked / rejected this pass:
- **`GGL_PUSHBACK_ERP=0.1`** (tuned ships half-Bullet ERP, measured from real wheel
  records): bit-identical on both tapes at 0.1 vs 0.2 (knob verified live — ERP=5 wrecks
  everything). Apply-time pushback is velocity-dominated here. Keep 0.2.
- Flip z-damp pre-increment: already ours by default (tuned agrees).
- Suspension impulse math (live rel-vel, sentinel-zero, inv-dot): already identical.
- TUNED_RAY/STEER on the **post-step** (no-PRE) stack: no p50 change (T1 ground stays
  ~3.38). The honest post-step floor ~3.4 IS the discretization, not wheel details —
  and tuned's own 0.032 comes from the same pre-step scheme our combo replicates.

Winning stack: `GGL_WHEELS_PRE=1 GGL_APPLY_TIME=1 GGL_NO_DAMP_FADE=1 GGL_TUNED_RAY=1
GGL_STEER_AT_APPLY=1 GGL_PUSHBACK_CAP=0`.

**SHIPPED AS COMPILED-IN DEFAULTS 2026-08-23 22:40** — no env vars needed. Each knob
inverts to opt-out (`GGL_WHEELS_PRE=0`, `GGL_APPLY_TIME=0`, `GGL_NO_DAMP_FADE=0`,
`GGL_TUNED_RAY=0`, `GGL_STEER_AT_APPLY=0`, `GGL_PUSHBACK_CAP=48` restores S39 cap).
Verified post-flip with a clean env: T1 ground 0.031 / wall 0.076 / air+w 0.540 /
flip+w 0.115; T2 c0 ground 0.043 / wall 0.269, c1 ground 0.182 / wall 0.697 —
bit-identical to the env-var stack. `GGL_WHEELS_PRE=0` off-switch confirmed working
(ground returns to ~3.5 post-step floor). NOTE: this changes TRAINING physics too
(the trainer links this crate) — the earlier "do not ship PRE as default" verdict was
for PRE alone; the full stack removed the skip-4 tax (see 21:32).

## Iteration 2026-08-23 22:30 (upstream PR sweep: ZealanL/RocketSim #73 #72 #66)

PR73 ("Fix jumping", merged 2026-08-23) is mostly already ours: integer jump/flip tick
clocks, no extra hold-force tick on release (S45c), pre-increment flip Z-damp, raw
lateral axle, wheels-before-integration (= WHEELS_PRE), pushback ERP 0.1 (ours is the
measured S45b optimum *with* the cap; upstream fit the same 0.1). The rest existed as
knobs; all re-A/B'd **on the winning stack** (verdicts could have flipped like UNCAP did):

| knob (PR73 item) | T1 baseline → knob | verdict |
|---|---|---|
| `GGL_REST_THRESH=1.0` (restitution cutoff 10→50 uu/s, new knob) | air+w 0.540→0.557, flip+w 0.115→0.143; T2-c0 air 0.457→0.465 | **reject**, keep 0.2 |
| `GGL_FLIP_THEN_AIR=1` (dodge torque on trigger tick) | vel bit-identical; flip+w ang p90 0.335→0.444, T2-c1 flip p90 8.57→9.60 | **reject** (2nd time) |
| `GGL_DODGE_NEEDS_AIR=1` (PR72 wheel-contact dodge stop) | flip+w 0.115→0.139, ang p50 0.003→0.226 | **reject** (2nd time) |
| `GGL_AXIS_SPIN=1` (per-axis dodge spin caps) | bit-identical | no-op, off |
| `GGL_DODGE_DAMP=1` (aerial damping during flips) | flip+w 0.115→0.176, flip_air ang 0.000→0.042 | **reject** |
| `GGL_JUMP_SETTLE=1` (suspension-settle jump re-arm) | bit-identical | no-op on 1-tick metric, off |
| `GGL_BOOST_THROTTLE=1` (PR73 full air throttle w/ boost) | bit-identical p50/p90 | keep OUR patch (S5 telemetry, 0.3%) |

PR66 (ball pen) already a knob (`GGL_BALL_PEN_FIX`), off: no-op on tapes (no sustained
ball-press regime). VirxEC confirmed the bug is real upstream — revisit only if training
shows ball-through-floor under dribble pinning.

**Nothing to adopt.** Upstream's PR73 verdicts were fit to their recordings; on our 120 Hz
tapes the winning stack already beats tuned everywhere and each remaining PR73 delta
measures worse or dead. Note T2-c1 (Plank skip-4) flip+wheels stands at 7.44 — the one
regime still fat, and it is held-input discretization, not a physics knob.

## Iteration 2026-08-23 22:45 (air+wheels / flip+wheels decomposition)

Enriched `GGL_PARTIAL_DECOMP` (car, signed axes, spd/vz, min nz, min susp, controls,
flip clock) and mined T1. The two "high" regimes are three separate stories:

1. **The p90 ≈ 7.7–8.26 band is NOT physics — it is boost-press timing.** The 7–9.5
   uu/s band (n=1257) is 97% sim-slower, pure-forward, with two discrete magnitudes:
   **8.264 = 991.667/120** (one tick of boost, thr=0 ticks) and **7.708 = 925/120 =
   (boost − air-throttle)/120** (thr=+1 ticks, where the sim applied air throttle but
   real applied boost). The real boost press lands on a tick our fixed `GGL_CTRL_LAG=2`
   doesn't reproduce. Same floor shows in air_free/ground p90 (8.26). Harness/input
   artifact, shared by every regime; a boost-onset aligner (infer press from
   boost_amount consumption) would remove it from the metric but changes what is
   measured. Physics knobs cannot touch it.
2. **air+wheels median driver: single barely-touching wheel.** nw=1 p50 1.48 vs nw=2
   0.37; within nw=1 it is flat floor (nz>0.9, p50 1.49) with suspension near full
   extension (susp 8–12uu p50 1.71 vs deep-compression 0.43). Signed medians ~0 —
   symmetric chatter at the contact threshold, not a biased force. RL's grazing-wheel
   force differs from ours in both directions; no simple scale fixes it (sticky/drive
   divisor sweeps all rejected earlier).
3. **flip+wheels tail: flipping against a tilted surface.** Floor bins are excellent
   (nw=1 floor p50 0.060, nw=2 floor 0.077). The tail is nw=2 on nz<=0.9 (p50 5.54,
   med_up **−0.84** — sim under-supports along car-up) and nw=1 on nz<=0.9 (p90 34).
   Plus 23 jump-held flip ticks at p50 2.96 (med_fwd −2.4). Late-flip (ft>=0.4) is
   0.33 vs 0.09 early.
4. **p99/max are tape discontinuities, not sim**: identical errors on BOTH cars at the
   same tick (i=8903), exact repeated values (fwd +502.17 at spd~80, thr=+1 — reset/
   freeze frames), negative susp / nz=0.00 penetration snapshots (i=4690 susp −2.3).

So the honest physics gap left in these regimes is (2) graze-contact strength and
(3) tilted-surface support during flips; (1) and (4) are measurement artifacts that
also inflate tuned's numbers the same way.

## Iteration 2026-08-23 22:55 (boost-press aligner — the p90 band is dead)

`GGL_BOOST_ALIGN` (default ON in the harness; `=0` restores the old metric): the
stepped tick's `controls.boost` is teacher-forced from the tape's `boost_amount`
telemetry — burned ⇒ boosting, unchanged ⇒ not, gained (pad pickup can mask a
concurrent burn) ⇒ keep the lagged bit. This removes the boost-press timing jitter
that put exactly one tick of boost accel (8.264 / 7.708) at p90 of EVERY regime.

T1 velocity before → after (p50 / p90):

| regime | before | after |
|---|---|---|
| ground | 0.031 / 8.26 | **0.021 / 0.46** |
| wall_drive | 0.076 / 8.26 | **0.058 / 0.90** |
| air+wheels | 0.540 / 7.72 | **0.516 / 3.54** |
| flip+wheels | 0.115 / 7.71 | **0.114 / 2.66** |
| air_free | 0.007 / 8.26 | **0.007 / 0.014** |
| flip_air | 0.009 / 8.26 | **0.007 / 0.556** |

`<1 uu/s` share: air_free 65.8→95.0%, ground 80.0→93.9%. Angular p99 also cleans up
(air+wheels 77.7 was velocity only; ang max drops to ~2.4). flip_air p90 is now
**0.556 = 66.667/120** — the residual is the same story one octave down: air-throttle
timing, not worth chasing (it needs a throttle-stream aligner with no telemetry to
anchor it).

T2: c0 ground 0.043→**0.028** (p90 0.52), c0 air+w 0.457→**0.253**, c0 flip+w
0.173→0.187 (p90 collapses 7.6→2.8); **c1 Plank skip-4 flip+wheels 7.44→1.57** — the
last fat regime was mostly held-boost misalignment compounding across the 4-tick holds.

CAVEAT: the aligner teacher-forces one input bit from telemetry, so numbers are not
comparable across tools unless both use it. Tuned's 0.032/0.081/0.57/0.12 were
measured WITHOUT it (and would also improve). Same-metric comparisons only.

## Iteration 2026-08-23 23:30 (the grazing-wheel law: sticky gates on LAST tick's contact)

Chased the "single barely-touching wheel" driver to a real physics law, in three steps.

**Step 1 — contact geometry is exonerated.** Dumped sim-vs-real per-wheel contact
masks (`sm`/`rf`/`rt` in PDECOMP). Raw masks disagreed on 55% of partial-contact
ticks — but every disagreement is a bit-swap within wheel pairs (0↔1, 2↔3): Bakkes
records left/right in the opposite order. After that permutation the masks match on
**2491/2491 air+wheels and 1258/1258 flip+wheels ticks**. Our ray reproduces RL's
contact decisions exactly; the recorded `susp_length` is extension-above-rest in uu
(max 12 = full travel), not absolute length.

**Step 2 — the bias curve.** Binning med_up error by min wheel extension (floor,
aligner on): dead zero for ext < 10.5uu, then −1.13 / −2.06 / −2.61 for 10.5–11 /
11–11.5 / 11.5–12 — saturating at exactly 0.5·g·dt = **2.708**, our whole-body sticky
impulse. Three wrong models tested: blanket loaded-wheels-only gate (worse everywhere
— extended wheels DO carry sticky), extension fades `cut`/`lin`/`quad` at 0.875
travel (cut helped pooled but left the bins bistable: med_up 0 with |err| ~2 — half
the ticks real applied full sticky, half none; a hidden variable).

**Step 3 — the hidden variable.** Splitting the graze bin: wheel STAYS in contact
med_up −2.47 vs LEAVES +0.006; falling (vz<0) −2.53 vs rising +0.005. A falling
grazing wheel had NO contact one tick earlier; a rising one did. **RL's sticky force
requires wheel contact on the PREVIOUS tick** — upstream PR73's "a car doesn't stick
on its spawn tick", which we had as `GGL_STICKY_PREV` and measured as a no-op: the
harness restore re-raycasts at the restored pose and erases the history. The knob was
never wrong, the test was.

Shipped (defaults): `GGL_STICKY_PREV` now **default ON** (`=0` restores fresh-ray
gate), as the conjunction prev-tick contact AND current contact (direction needs a
live normal). `Arena::seed_sticky_gate_prev` lets the harness seed the gate from the
tape's previous tick. `GGL_STICKY_FADE` kept as an off-by-default A/B knob, marked
superseded. No flip exemption needed — the gate explains flips natively.

Velocity p50 / p90, before → after (aligner metric):

| regime | T1 before | T1 after |
|---|---|---|
| air+wheels | 0.516 / 3.54 | **0.376 / 2.36** |
| flip+wheels | 0.114 / 2.66 | **0.104 / 2.37** |
| ground | 0.021 / 0.46 | 0.021 / 0.46 (identical) |
| wall_drive | 0.058 / 0.90 | 0.058 / 0.90 (identical) |

T2: c0 air+w 0.254/2.71→**0.207/1.88**, c0 flip+w 0.187/2.88→**0.173/2.73**,
c1 Plank air+w 0.428/4.19→**0.323/2.84**, c1 flip+w 1.57/4.51→**1.52/3.45**;
ground/wall identical. Improvement or parity in every regime on both tapes,
including the skip-4 Plank. Graze bins after: med_up ~0 AND p50_verr 0.12–0.84
(was 2.7) — per-tick fixed, not just mean-fixed.

Answer to the original question: sticky DOES apply through a single wheel (a lone
compressed wheel takes the full 0.5·g body impulse, unbiased), and extension/load
doesn't matter — what matters is contact history. The general law, all wheel counts:
`sticky_this_tick = any_wheel_contact(t−1) && any_wheel_contact(t)`.

## Iteration 2026-08-23 23:45 (post-gate identification pass — the next targets)

Re-decomposed air+wheels / flip+wheels with the prev-gate live (PDECOMP now also
dumps `pf`, the real mask one tick before the from-tick).

**The matching p90s (2.356 / 2.368) are a coincidence of magnitude, not mechanism**:
zero tick-index overlap between the two tails. But both tails ENTER at the same
quantum — top histogram bin ~2.5–2.75 = the sticky impulse — because both regimes
share the same residual physics families below.

**Gate truth-table reconstruction** (label real's sticky from the ±2.708 up-error on
floor ticks, evaluate candidate gates on the recorded masks): the shipped
`any(t−1) && any(t)` gate is right on **2914/2944** labeled ticks. All 30 residuals
are sim-ON/real-OFF with pf==rf (same wheel, both ticks, falling, mid-extension) —
mask-indistinguishable: every mask-based gate scores identically, so whatever turns
real's sticky off there is NOT contact history (candidates: post-jump state,
internal wheel-contact latency). 1% population, low priority.

**Error-mass map** (junk verr>=50 excluded, share of summed verr):

air+wheels (mass 2785):
1. **floor lat-dominated 28.8%** — the fattest target. The clean core is nw=1 +
   steer: n=36, med|lat| **8.2 uu/s**, sign matches steer on **31/36**, spd ~2000,
   all suspension depths. Sim generates lateral grip through a single wheel that the
   real game does not — single-wheel steering is far too effective. (nw=2 lat is a
   different, muddier bin: handbrake 45%, not steer-aligned.)
2. **floor fwd-dominated 21.7%** — coasting (thr=0) n=67 med_fwd **+2.60**: sim
   retains forward speed where real scrubs it, worst at compression (susp<0:
   +5.28) — landing/partial-contact rolling friction too weak in sim. Small
   counter-bin: thr=+1 spd>1400 med_fwd −2.76 at p50 7.71 (residual boost-aligner
   misses, n=16).
3. **tilted other 17.2%** (wall/curve transitions, mixed axes).
4. floor up-dominated 8.2% (med_up −1.25, suspension impact strength).

flip+wheels (mass 1388):
1. **tilted 38.1%** (p50 8.23, med_up −0.87) — flipping against a tilted surface,
   the known under-support; unchanged by the gate work.
2. floor fwd 20.4% (thr=+1 high-speed +1.20; coast +4.77 — same coast-scrub family
   as air+wheels).
3. floor up-dominated 16.3%, lat 9.8%.

Sticky-quantum populations are now 0.4–2.3% of mass — the gate work is done.

**Attack order by expected yield**: (1) single-wheel lateral grip (load-scale or
count-scale the friction curves — RL's real per-wheel lateral force at partial
contact must be much weaker than our full curve), (2) coast/landing longitudinal
scrub, (3) flip-vs-tilted-surface support. All three are force-magnitude questions
at partial contact, where our friction/drive is currently applied at full strength
regardless of wheel count or load.

## Iteration 2026-08-24 00:00 (partial-contact force attack — three clean REJECTS, better map)

Implemented and swept three force-model hypotheses. All rejected with dose-response
curves; all knobs default OFF (baseline re-verified bit-identical afterward).

1. **`GGL_LAT_PARTIAL` (scale lateral grip when <3 wheels): REJECTED.** n/4 →
   air+wheels p50 0.376→2.93; fixed 0.25 → 3.98; 0 → 5.07. Monotone worsening:
   the real game applies FULL bilateral-strength lateral grip through 1–2 wheels
   for the bulk population.
2. **`GGL_BRAKE_PARTIAL` (renormalize brake by 4/n when <3 wheels): REJECTED.**
   0.376→1.135. Per-wheel brake at full clamp with missing wheels uncompensated is
   already the real behavior.
3. **`GGL_LAT_LOADCAP` (friction cone |F_lat| <= mu*N, N = suspension force):
   REJECTED at every mu** (0.5/1/2/4 → ground p50 3.47/1.74/0.17/0.024, p90 all
   wrecked). Physics lesson: RL cornering demands lateral force many multiples of
   mu*N (2000 uu/s turns are ~10g) — **real RL grip has no friction cone at all**.

Why the identification bins misled: drilling the tails further shows they are not
distributed physics error —

- The nw=1 "BL wheel p50 6.04" bin is ONE sustained event (c0 i=4648–4725, ~80
  ticks, mirrored small one on c1 i=9441): a car pirouetting on its back-left
  corner at ~2050 uu/s. Two phases: falling onto the grazing wheel (up −17, real
  supports MORE than sim) then sliding on it (lat −15.8/tick steady, sim grips
  toward steer, real saturates). Real's one-wheel force law during corner-balance
  differs in both axes, but with n≈2 events it cannot be fit without overfitting —
  needs more tapes containing corner-spin play.
- The fwd tail splits: ticks AT throttle transitions (26.6% vs 6.0% base rate,
  med_fwd +7.0 — one throttle quantum) are input-timing jitter, the boost story
  again but with NO telemetry to anchor an aligner (throttle leaves no
  boost_amount trail). Unfixable in the harness; inflates air+wheels p90
  permanently. The residual steady-state coast scrub is med +2.2, n=80.

Remaining honest physics gaps in these regimes, post-rejections: (a) the
corner-spin one-wheel force law (needs data), (b) steady coast scrub +2.2 on
partial contact (mechanism unknown — brake renorm is NOT it), (c) flip-vs-tilted
support (untouched), (d) 30 sticky gate residuals (mask-invisible). The identified
input-jitter populations (steer 36%-enriched lat ticks, throttle-transition fwd
ticks) put a measurement FLOOR under p90 that only better tapes can lower.

## Iteration 2026-08-24 00:15 (T3 targeted tape — one-wheel physics EXONERATED, real culprit is car-car)

User recorded a targeted solo tape (T3, `autosave_20260823_235840`, 17,742 ticks,
single car): supersonic jump → yaw sideways → land dragging a single rear wheel,
steer held, both directions, sustained 10–18-tick runs.

**Harness fix first**: T3 is freeplay with unlimited boost — `boost_amount` pinned
at 100 burns nothing, so the boost aligner concluded "not boosting" every tick and
stripped boost (ground_boost p50 read exactly 8.258, one boost quantum). Guard
added: unchanged-tank inference only applies when the tank is below full; pinned
full ⇒ keep the lagged control bit. ground_boost 8.258 → **0.026**.

**Result: the deliberately-hunted regime is ACCURATE.** 313 nw=1 ticks (BL 117 /
BR 143, mirrored steer, med speed ~1800), 252 of them HARD falls (vz < −150, med
−212): sustained-run p50 **0.109**, p90 1.08, unbiased on all three axes. Hard-fall
subset p50 0.22. Steady AND impact single-wheel wheel physics need no fix.

**So what was the T1 corner-spin event?** Added opponent distance (`od`) to
PDECOMP: during the entire c0 i=4648–4725 event the opponent is **66–103 uu away**
— two Octanes are ~85–120 uu center-to-center when touching. The event is a
prolonged CAR-CAR GRIND while drifting (and c1's i=9441 "event" is the same grind
from the other car's tape, od 90). It generalizes: across T1's partial-contact
regimes, big-error ticks (verr>3) have med od **168** vs **1793** for small-error
ticks; 53% of big-error ticks are within 300 uu vs 7.7% of small.

**Conclusion: the dominant remaining tail in air+wheels / flip+wheels is car-car
contact during partial wheel contact** (side-by-side grinding, riding on the
opponent), not wheel force laws. The decomp bump/demo routines cover discrete
impacts; the continuous scrape/push manifold is the untested surface. Next data:
a two-car tape with deliberate side-by-side leaning/grinding at speed.

## Iteration 2026-08-24 00:20 (GGL_OPP_FILTER — the solo-physics scoreboard)

Car-car is parked (user decision: harder problem, later). `GGL_OPP_FILTER=<uu>`
(harness, default off) drops a car's tick from ALL metrics when an opponent is
within the distance. At 275uu, T1 velocity (p50/p90):

| regime | unfiltered | filtered 275 |
|---|---|---|
| air+wheels | 0.376 / 2.356 | **0.296 / 1.069** |
| flip+wheels | 0.104 / 2.368 | **0.089 / 1.750** |
| ground | 0.021 / 0.461 | 0.020 / 0.375 |
| wall_drive | 0.058 / 0.900 | 0.057 / 0.885 |

T2: c0 air+w p90 1.875→1.183, c1 air+w 2.842→2.475, c1 flip+w 3.452→3.157.
(p99s now look WORSE in some regimes — the contested ticks that left were mid-tail,
so the known tape-junk spikes (kickoff-freeze repeats at ~480-680) moved up to p99.
They are artifacts, not physics.)

**Next fruit, from the filtered decomposition** (error mass, junk excluded):

1. air+wheels **floor fwd-dominated 36.1%** (n=104, med_fwd +2.20, med_vz −86) —
   the coast/landing scrub again, now the clear #1.
2. flip+wheels splits evenly: tilted 27.5% / floor-up 24.4% / floor-fwd 24.8%.
3. **A shared signature**: every err>1 bin in BOTH regimes carries med_up ≈
   **−0.85** (tilted −0.85/−0.86, up-dom −0.86/−0.82, fwd-dom −0.83/−0.87) except
   the pure-lat bins. A constant ~0.85 uu/s per-tick under-support along car-up on
   partial contact, independent of flip state — one common additive term
   (~0.157·g·dt, or ~0.31·sticky) is missing from the sim whenever the error is
   nonzero. Identifying that constant is the cheapest single lever left.

## Iteration 2026-08-24 00:45 (the −0.85 constant = AUTO-ROLL's linear force; gate it off during jump/flip)

The −0.85 was not quantization (one-sided, throttle-conditioned) and not the
sticky tilt term (mode ticks are nearly upright, med 1−|up.z| = 0.033; the error
doesn't track tilt in any bin). The histogram spike is razor-sharp at **−0.82 ±
0.02**, present in sustained runs (a 19-tick flip landing at steady −0.80) and on
ticks where sim/real contact masks agree exactly — a constant force behind a
binary gate, not contact noise.

**Found by instrumentation, not hypothesis**: added a release-mode impulse trace
(`DBG_IMPULSE_TRACE` in rigid_body.rs, harness knob `GGL_IMP_TRACE=lo:hi`) and
read the per-term ledger on the flip-landing event. One anonymous impulse of
exactly **(0, 0, −1/60 BT) = −0.8333 uu/s** fires every tick on the erroring car:
`autoroll::FORCE (100 uu/s²) × dt`, the **auto-roll assist's linear ground-down
force**, gated on throttle ≠ 0 ∧ (1–3 wheels ∨ chassis contact). The real game
applies nothing on these ticks. v3-tuned independently shipped BOTH autoroll
scales at 0.0 from corpus measurement ("real autoroll does not run the way v2/v3
model it") — but zeroing it outright is measurably wrong too:

| T1 regime (p90 / <1uu/s) | v2 on | fully off | **noflip (shipped)** |
|---|---|---|---|
| flip+wheels | 2.368 / 78.9% | 1.332 / 88.2% | 1.536 / 87.2% |
| flip_air    | 0.556 | 0.136 | 0.136 |
| air+wheels <1uu/s | 84.4% | 79.9% | 84.1% |
| ground      | 0.461 | **0.676** | 0.474 |

Fully-off buys flips but regresses ground p90 45% and air+wheels — the assist is
real outside jump/flip. **Shipped default `GGL_AUTOROLL` = "noflip"**: exact v2
gate minus `is_flipping ∨ is_jumping` ticks. T2 confirms: flip+wheels p50
0.645→**0.455**, p90 2.851→**1.624**, <1uu/s 56.4%→**77.4%**; flip_air p90
1.050→0.487; everything else flat. Variants rejected: `strict` (also drops the
chassis-contact arm — equal on tapes but would delete real turtle auto-roll the
tapes don't sample), `nofliplanded` (also excludes has_flipped — no flip gain,
air+wheels <1 drops to 82.1%). Knob: `GGL_AUTOROLL=1` restores v2, `0` kills it.

## Iteration 2026-08-24 01:00 (T4 targeted tape — noflip validated, strict killed, air-throttle stacking found)

User recorded T4 (`autosave_20260824_004902`, 31,323 ticks, solo Octane, clean
120 Hz): sustained powerslides on 1–2 wheels, wall slides, turtle-with-throttle,
side-wheel landings, jump launches. This is auto-roll's native regime, which
T1/T2 barely sampled.

**Gate arbitration, settled by data:**

- **`noflip` (shipped) is right.** Auto-roll fully off drives T4 air+wheels p50
  0.559 → 0.950 — real auto-roll exists in sustained partial-contact driving.
- **`strict` is dead.** The turtle/chassis-contact play lives in air_free and
  air_boost (0 wheels + chassis contact): with the chassis arm intact air_free
  p90 is 0.924 and air_boost 0.415; strict/off (which delete turtle auto-roll)
  regress them to 1.420 / 0.813. Real turtle auto-roll exists.
- Flip gating re-confirmed: T4 flip+wheels p50 1.271 → 0.531 (v2 → noflip),
  flip_air p90 0.826 → 0.013.

**Second find — air throttle stacks on wheel drive.** With auto-roll settled, T4
air+wheels sat in a tight band (p50 0.559, p90 0.840). Decomp: pure FORWARD bias,
exactly **±0.556 uu/s signed by throttle** = `THROTTLE_AIR_ACCEL (200/3) × dt`,
uniform across speed 0–2300 (including >1400 where ground drive is zero), tilt,
wheel count 1–2, and boost — zero at throttle 0. `update_air_torque`'s throttle
force ran whenever `!is_on_ground` (n<3), so 1–2-wheel ticks got air throttle ON
TOP of wheel drive; the real game applies it only with **zero wheels in
contact**. Gated (VENDOR PATCH 2026-08-24, `GGL_AIR_THROTTLE_WHEELS=1` restores
stacking):

| air+wheels p50 | before | after |
|---|---|---|
| T4 | 0.559 | **0.060** (9×) |
| T1 | 0.376 | **0.251** |
| T2 | 0.256 | **0.180** (p90 2.19 → 1.87) |

T4 flip+wheels p50 0.531 → 0.139. No regressions on any regime of any tape.

## Iteration 2026-08-24 01:45 (car-car grind — manifold persistence shipped)

New instrumentation: `GGL_CARCAR=<uu>` dumps opponent-frame error (ax = along
car→opponent axis, + means sim under-separates; pp = horizontal perp; ez = world
z) plus closing speed, dz, and sim bump events, for every tick with an opponent
inside the radius. T1 od<130 (touching, n=988) baseline: verr p50 0.54 / p90
12.39, worst bin = sustained neutral-closing grind (med ax +0.66) — the sim
under-separates; the tail is on the NORMAL axis, not friction; bump events are
irrelevant (1% of ticks).

**Root cause (traced, not guessed).** The T1 grind event (i=4648–4725) shows
mutual momentum-conserving separation bursts of ~16 uu/s decaying over ~5 ticks
on BOTH cars. `CCSOLVE`/`OBBOBB` tracing (new, gated on `GGL_IMP_TRACE`) showed
the box-box detector **flickering**: on exactly the burst ticks it finds NO
contact (1–2 uu nominal gap), while adjacent ticks hit with pen jumping −0.01 →
−2.8 uu. Real Bullet keeps a **persistent manifold per pair**: a point created
at a real touch survives (and keeps entering the solver, stopping approach)
until it drifts >1 uu along or orthogonal to its normal. The port rebuilt
manifolds from the detector every tick, so one grazing miss dropped the contact
entirely.

**Shipped: `GGL_CC_PERSIST` default ON** — car-car manifolds are cached in the
dispatcher across ticks; on a detector miss the cached points are refreshed
(Bullet break criteria) and handed to the solver if alive; entries evict when
the broadphase pair separates. T1 od<130: verr p90 **12.39 → 10.92**, |ax| p90
7.31 → 6.46, grind-event med ax +6.21 → +2.70. T2 bit-identical (no grind
there). No collateral: persistence needs a real seed touch, so fly-bys are
untouched. Aggregate regime tables barely move (car-car is a thin slice of each
regime) — the od<130 table is the scoreboard for this surface.

**Tested and rejected:**
- `GGL_CC_INFLATE` (+1/2/3 uu box inflation, the old-Bullet-2.6x
  unshrunk-margin theory): kills the grind bias (event +6.21 → +0.46, p50
  halved) but manufactures contact on fly-bys the real game misses — od 141
  passes gain 42 uu/s errors, one false bump (verr 821). Real contact
  generation is at nominal dims. Falsified.
- `GGL_CC_BAUMGARTE=0.2` (velocity-level penetration correction, old-Bullet
  `m_splitImpulse=false` path): zeroes the event bias (−0.29) but od<130 p90
  11.49 vs persist's 10.92. Mixed; knob kept, default off.
- `GGL_CC_DEPTH` 1–2 uu (deeper solver pen for car-car), `bg 0.8`: both worsen
  p90. Rejected.

Honest leftover: od<130 p90 ~10.9 (vs ~0.5–2 for solo regimes) — grind residual
med ax +0.49 under bg, contact-tick response magnitude still off (e.g. i=9417–22
cluster where inflation, wrongly, helped). Next lever would be warmstarting the
persistent points (`applied_impulse` carry, real Bullet does) or the per-point
friction anchor during multi-tick scrapes.

## Iteration 2026-08-24 02:30 (T5 octane-vs-octane grind tape)

T5 `autosave_20260824_020526.rlpr` — 120 Hz (Δ=1, airborne |Δvz| p50=5.42),
c0 skip-1, c1 med hold 13. 29 contact events, 1352 ticks od<130, bump rate 2%.

**Persist on this tape:** 57 ticks improved >0.5 vs 1 regressed, but only 29/675
c0 ticks are persist-sensitive (error mass 176 vs 4189). ~96% of T5 contact
error is ticks where chassis SAT never seeds. Persist stays default ON.

New dump fields on `GGL_CARCAR`: `sn` (sim wheel contacts at restore), `tnz`
(tape min |nz|), `wc`/`wm` (sim wheels whose ray hit another car).
`Arena::wheel_hit_car_mask`.

**Two leftovers, now split:**

1. **Wheel-on-car roof-ride (wc>0, 82 ticks, 6% of od<130).** verr p50 19.4 /
   p90 48 vs wc=0 p50 0.21. Event 15428: airborne approach verr 0.01 until first
   wheel-on-roof, then 192 uu/s spike *before* SAT; sit is wc=2, tnz=1 (roof is
   horizontal). `GGL_WHEEL_ON_CAR=1` (Newton pair + pushback on car hits):
   od<130 p90 15.07→16.43, wc>0 3 wins / 4 losses. **Rejected, knob kept off.**
   The riding car already gets suspension; the missing magnitude is not the
   reaction on the car underneath.

2. **Floor-floor grind with no generated contact (event 4182–4204).** 19 ticks,
   both cars +14..+31 ax (identical, same sign), n=2 sn=2 tnz=1.00 **wc=0**,
   SAT miss every tick (d=129.5→109.9) until bump at 4205 which then matches.
   Tape and sim agree: two wheels on the *floor*, not on the opponent. Persistence
   cannot seed; wheel-on-car cannot fire. Real still applies a mutual normal
   our SAT never emits. Inflation remains rejected (T5 dive 15408–26 is verr
   0.01 through the same distance range).

Bump path: when `CarHitCar` fires, subsequent ticks match. Not the leftover.

Next SAT-side lever: log closest-axis gap on box-box miss (how many uu short
are we on 4182?), not blanket inflate. Warmstart still only helps the persist
slice.

## Iteration 2026-08-24 07:30 (SAT closest-axis gap + 0.3 uu slack A/B)

`GGL_CARCAR` now carries `sg` / `sa` / `sk` from the last car-car box test of
the step (always stored; `GGL_SAT_GAP=1` also prints `SATGAP` lines):

- `sg` = closest-axis SAT gap, uu. Positive = separated (short of contact);
  negative = penetration. NaN if the detector never ran.
- `sa` = axis (1–3 car A faces, 4–6 car B faces, 7–15 edge×edge).
- `sk` = 0 none / 1 miss / 2 hit / 3 overlap but clip empty / 4 AABB miss /
  5 slack-promoted (only with `GGL_SAT_SLACK`).

AABB misses also store an axis-aligned gap (`sa=0`, `sk=4`).

### 4182 is a 0.1–0.3 uu edge miss

Event 4182–4205, both cars, floor-floor (`n=2 sn=2 tnz=1`, `wc=0`):

| ticks | `sg` | `sk` | verr |
|---|---|---|---|
| 4182–4197 | **+0.10 to +0.28**, always **axis 9** | miss | 13–31, almost all `ax` |
| 4198–4200 | +0.38 to +0.74 | miss | ~12 |
| 4201–4204 | +0.39 to +1.03 | miss | ~0 (tape already not colliding) |
| 4205 | **−0.16**, axis 9 | hit, `bump=1` | ~0 |

Axis 9 = edge×edge (`7 + i*3 + j` → car A forward × car B up). Two Octanes at
`od≈129` with a 0.17 uu surface gap is a **corner scrape**, not face-face. The
moment SAT goes negative, the bump path matches.

od<130 c0 (678 ticks): SAT miss **444** (gap p50 **+0.29**, verr p90 **20**);
SAT hit 97; AABB miss 24; detector never ran 112 (roof dive 15408–24, verr
0.01). Misses with `sg < 1` are the error mass (verr p90 29); misses with
`sg ≥ 8` are clean fly-bys (verr p90 0.5). Clip-empty is rare off slack (1
tick). Roof 15428 still first wheel-on-car with SAT **+3 to +7 uu** short —
not this generation gap.

### `GGL_SAT_SLACK=0.3` — tested, **rejected**, default OFF

Not box inflation: if the closest SAT axis misses by ≤0.3 uu, treat it as a
shallow overlap on that axis (`depth = slack − gap`) and generate a manifold.
Knob `GGL_SAT_SLACK=<uu>`, default 0.

T5 A/B vs slack=0:

- **Target event mostly closes, then explodes.** 4182–4194 / 4196–4197:
  verr 14–31 → 0.1–7, `sk=5`. **4195:** false `CarHitCar`, verr **16 → 992**
  (`ax −694`). **4205:** the tick that already matched (verr 0.04, real SAT
  hit) → **1036**, bump lost. 4198–4200 improve without promotion (`sg` still
  0.38–0.74) — persist is carrying slack-seeded points across restore-1-step
  ticks; that is not a licensed generation fix.
- **Same fly-by kill as inflate.** i=4165 od=136 `sg=+0.04` (well inside
  0.3): verr 24 → **1116**, false bump. od 130–200: 97 slack-promoted ticks,
  that subset verr p90 **12.9**; band aggregate p90 stays 1.12 because they
  are a thin slice.
- od<130 c0: p90 **17.61 → 13.76**, p99 **152 → 215**. Tickwise |dv|>0.5:
  **92 wins / 54 losses**. wc=0 floor slice p90 11.35 → 5.73, p99 137 → 262.
  Slack-promoted od<80: verr p50 **29.5** (roof-adjacent / stacked).
  Clip-empty **1 → 57** (od<130) / 114 (od<300): many promotions have no
  clip points; persist then holds whatever was cached.
- Roof 15425–40: unchanged where `sg > 0.3`; tiny-gap roof ticks 15433–39
  `sk=5` with no verr help.
- Regime table: air+wheels p90 6.89 → 3.37 (4182-class ticks live there) while
  the worst list gains 4195 / 4205 / 4165 kilo-uu/s spikes.

**Reject 0.3 without a bump-cone gate.** Slack-promoted manifolds counted as
touching for `on_car_car_collision` and fired `bump_impulse` at attacker full
speed. Later (v5 `211647`): **slack=0.2 + skip bump on `LAST_SAT_KIND==5`**
is a pure win (window CarImpact p90 26.05→13.82; full-tape p99 unchanged;
demos still fire). That combo is the training default. `GGL_SAT_SLACK=0`
disables promotion. Clip-fallback / center-dot gates stay off.

Honest leftover is still generation: real contacts at 0.1–0.3 uu of SAT
separation on edge 9, and we do not have a contact-point recipe that applies
the real impulse without also hitting fly-bys. Roof-ride is a separate
wheel-on-car magnitude miss. Persist/warmstart remains a thin slice.

## Iteration 2026-08-24 09:45 (ground boost on any wheel — T1 air+wheels +0.556 mode)

Nearby cars masked (`GGL_OPP_FILTER=200`). Remaining air+wheels error mass
(verr<50) was **66% floor fwd-dominated**, p50 verr **0.567**, med_fwd **+0.562**.
That is `THROTTLE_AIR_ACCEL * dt = (200/3)/120 = 0.556`, uniform from ~490 to
~2300 uu/s at `thr=+1`. Air throttle is already gated (`GGL_AIR_THROTTLE_WHEELS`);
IMP on 3922 / 3693 showed zero throttle impulse and a **Boost** vector of
**8.82 uu/s** (air + 0.556) vs real **8.26** (ground).

Cause: `update_boost` keyed on `is_on_ground` (`n>=3`). 1–2 wheels took
`ACCEL_AIR` (`3175/3`) instead of `ACCEL_GROUND` (`2975/3`). Dual of the
air-throttle gate: real still uses ground boost with wheels down.

**Shipped default ON:** ground boost accel whenever any wheel is in contact.
`GGL_BOOST_AIR_WHEELS=1` restores air accel on partial contact. Do not train
with the restore knob.

T1, nearby cars filtered (`od < 200`):

| regime | before p50 / p90 | after |
|---|---|---|
| air+wheels | 0.211 / 1.07 | **0.075 / 1.00** |
| flip+wheels | 0.090 / 1.01 | **0.071 / 0.99** |
| ground / wall / air_free | 0.020 / 0.38 | unchanged |

c1 air+wheels p50 **0.283 → 0.084**. Unfiltered air+wheels p50 **0.251 → 0.082**;
p90 **2.32 → 2.23** (car-car tail). Flip+wheels unfiltered p50 **0.101 → 0.077**
(p90 still 1.54). Ground/wall bit-identical.

p90 barely moved: leftover filtered air+wheels p90 ~1.0 is not this quantum.

## Iteration 2026-08-24 11:15 (T6/T7 walls — rest-length pushback, MINPEN not shipped)

T6 (flat side wall) and T7 (fillets), 120 Hz solo Octane. T6 stayed on-wall
(`|uz|<0.25` on 6406/6649 wall ticks) but circles did **not** load the spring
(only 3% `susp<-6`; median −2). Straight driving is already **0.012 / 0.51**.
T7 compressed fillet is **0.87 / 2.98**, not T1's 7.8 — T1's "compressed wall
p90" was mostly fillet/car-car, not a missing spring.

T6 leftover is a **+6.84 uu/s car-up quantum** on ~17% of true-wall ticks,
almost all full-steer at rest compression, 60% of `verr>1` ticks sit on that
spike, none of the opposite sign (sim leaves the wall). Runs last hundreds of
ticks (e.g. 2269–2571). Sticky is **not** it: IMP sticky is identical on clean
straight (i=859) and quantum (i=2269), `−0.1625` BT = `1.5·g·dt` into the wall.
What changes is rear-wheel `WheelsSuspension` (~0.049 → ~0.106 each). Net
sticky+spring cancel when driving straight; while turning, extra pushback
(~5.9 uu/s out) is the quantum.

**Core issue (11:15 attribution; 12:15 revises):** uncapped apply-time pushback's
velocity resolve can kill into-wall speed at the contact, including yaw×lever
at rest length. Fillets still need that vel-kill once compressed. The T6
quantum itself was **stale extra_pushback persist**, not a missing depth gate
— see 12:15. Per-wheel depth gating is worse: some wheels still fire, the
kick goes lopsided, T6 straight p50 0.01→4.7.

**Semi-fix, default off (superseded 12:15):** `GGL_PUSHBACK_MINPEN=<uu>` skips extra_pushback for
**all** wheels when the chassis is on a wall (`|up.z|<0.5`,
`GGL_PUSHBACK_MINPEN_UZ`) **and** the deepest contact is shallower than the
threshold. `=0.5` is the A/B. The keep column here is *pre-clear* (stale
`extra_pushback` still applied). The MINPEN column also includes the apply-time
clear, which is the actual T6 win — see 12:15. Do not train with MINPEN.

| tape / slice | keep p50/p90 | MINPEN=0.5 |
|---|---|---|
| T6 true wall | 0.038 / **6.84** | **0.022 / 0.82** (quantum 17%→0) |
| T6 steer, uncompressed | 0.046 / 6.84 | **0.027 / 0.97** |
| T7 wall overall | 0.024 / 0.88 | 0.029 / 1.05 |
| T7 fillet | 0.116 / 1.28 | 0.116 / 1.47 |
| T1 opp-200 air+wheels | 0.075 / 1.00 | 0.075 / 1.02 |
| T1 ground | 0.018 / 0.35 | unchanged |
| T1 wall | 0.052 / 0.87 | 0.063 / 1.01 |

The keep column is pre-clear (T6 +6.84 still present). MINPEN=0.5 also
cleared stale extra_pushback, which is why T6 moved; the T1/T7 p90 tax is the
skip-recompute gate, not the clear. See 12:15. `GGL_INV_DOT_CAP=1` (kill
1/cos) rejected on the same pass: T6-irrelevant, air+wheels 0.075→0.365.

## Iteration 2026-08-24 12:15 (T6 +6.84 = stale extra_pushback; clear shipped, MINPEN rejected)

The 11:15 table mixed two changes. Apply-time `update_extra_pushback` only
writes when the ray is inside the rest+radius threshold. If it is not, the
previous tick's extra_pushback stayed on the wheel and was applied again at
the new pose. On T6 that leftover is the rest-length yaw kick: sticky is
identical on clean straight (i=859) and the old quantum (i=2269), `−0.1625`
BT; rear `WheelsSuspension` was ~0.106 (spring + stale pushback) vs ~0.043
(spring only) after the clear. The quantum ran for hundreds of ticks with one
sign because the leftover never got overwritten.

**Shipped:** every apply-time tick zeros extra_pushback, then recomputes.
This tick's resolve or none. T6 true-wall `|up|>6` goes 17% → 0. T1/T7 wall
p50/p90 stay at the pre-MINPEN keep numbers.

**Rejected:** `GGL_PUSHBACK_MINPEN=0.5` additionally skips the recompute on
shallow walls. That is the T1/T7 tax (and 34 T6 ticks that wanted a live
resolve, e.g. i=1818 1.84→12.6). Tightening `GGL_PUSHBACK_MINPEN_UZ` to 0.25
only saves T7 fillets; true-wall tax stays. Default 0.

| tape / slice | shipped (clear + recompute) | MINPEN=0.5 |
|---|---|---|
| T6 true wall | **0.022 / 0.78** (`\|up\|>6` = 0%) | 0.022 / 0.82 |
| T7 wall overall | **0.024 / 0.88** | 0.029 / 1.05 |
| T1 wall (opp-200) | **0.051 / 0.85** | 0.062 / 0.96 |

A/B trap: `set GGL_NO_PUSHBACK=` (empty) is still set, and
`is_ok_and(|s| s != "0")` treats it as on. Unset the var; do not empty-assign
flag knobs.

## Iteration 2026-08-24 12:50 (packed fillet leftover; no ship)

Target: `wall_drive` ∩ `susp<-6` ∩ `|uz|≥0.25`. T7 n=149 all 4-wheel (`sm=1111`).
Uncompressed true wall already **0.020 / 0.77**. Packed fillet is the T1 wall
p90 leftover class (opp-200: 12% of wall ticks, ~42% of wall error mass).

3-step free-run does **not** average it out: T7 packed 0.89/3.39 → 1.60/18.6,
throttle-on accumulates +fwd. Sticky tilt / `STICKY_SCALE=0.5` wreck true
walls (5.4 / 4.1). `GGL_FRIC_LEVER` packed p90 3.39→3.06, uncompressed fillet
p50 0.044→0.101. Damp fade is a no-op on packed (fade is extended-only).

IMP on a median packed tick (T7 i=11707, steer 0, lat −1.12): friction ~0.02,
`WheelsSuspension` ~1 uu/s. Contact `nz` is faceted (0.10 / 0.30 / 0.48 / 0.64).

| lever | T7 packed | T7 true wall | T7 air+wheels |
|---|---:|---:|---:|
| keep | **0.89 / 3.39** | **0.020 / 0.77** | **0.036 / 0.91** |
| `GGL_PROJ_AXLE=1` | 1.06 / 3.04 | 0.026 / 0.81 | **2.17 / 7.85** |
| `GGL_SUSP_UP=1` | **5.63 / 17.6** | 0.092 / 1.40 | 0.040 / **36.7** |
| `GGL_FILLET_DRIVE=2` | 1.25 / 3.48 | bit-identical | bit-identical |

`FILLET_DRIVE=2` also taxes uncompressed fillet p90 0.89→4.12; packed thr+
1.23→2.03 so 1-step is **not** over-drive. Weaker or redirected contact force
on this slice blows packed, uncompressed, or air+wheels — same class as
DIV/CAP/ERP/`INV_DOT`. Leave packed fillet as accepted leftover unless the
fillet mesh is replaced. Default-off knobs stay off.

## Iteration 2026-08-24 16:17 (PR74 isolate; ship normalize, reject dedup)

VirxEC [PR 74](https://github.com/ZealanL/RocketSim/pull/74) three commits, each
alone vs keep. Restore-then-1-step, `GGL_CTRL_LAG=2`, `[0,25000)`. Ground / air /
post / crossbar and T7 cars bit-identical on every arm.

1. `ce694cd` skip recording tick 0 — already dropped by control lag 2. No-op.
2. `22cf396` unit-length averaged special normal — **ship**. T2 wall p99 4.15→3.72,
   `<1` 98.1→98.6; T5 wall max 266→239; T1 ang p90 0.027→0.020. No car tax.
   `GGL_NO_NORM=1` restores the unnormalized average.
3. `1499c4d` dedup + min `distance_1` + static-only special — **reject**. Wall p99
   T1 3.19→4.51, T2 4.15→19.4, T5 5.68→7.76. `GGL_PR74` / `GGL_PR74_DEDUP` stay off.

## Iteration 2026-08-24 17:11 (ball leftover split; car-touch faces)

`sum(|v_sim−v_real|)` on restore-then-1-step. Quiet `verr<1` is 95–98% of ticks
and almost none of the mass. T1/T2 mass is car hits (flag + missed-flag
`dv_real≥50`); T5 is the world leftover (floor-fillet). See leftovers 5–7.

Car-local hitbox face (nose=+x, roof=+z) on tape-touch or near+impulse ticks.
`analyze_rlpr` always prints the table; `GGL_BALL=1` also dumps `BALLTOUCH`.

| | T1 | T2 | T5 |
|---|---|---|---|
| face | roof 84% mass | roof 85% | left+right 67% |
| feature | edge 83% | mixed | edge+corner ~100% |
| flip | **64% mass / 21% ticks** | 7% | ~0 |
| sim `CarHitBall` | hit 71% (wrong impulse) | miss 59% | miss 69% |
| underside | 0.3% | ~0 | 0 |

Next physics levers, in order: T1 `roof|flip` extra-hit, T5 side-edge generation,
T2 roof generation. Do not chase underside or nose dribbles.
