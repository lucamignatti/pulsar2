# AERIAL_GAP — where exactly does the aerial break down? (measurement)

**Pre-registered 2026-07-15 BEFORE running** (user observation: "the bot still
isn't going for aerials really"). Script `aerial_gap.py`, newest checkpoint.
Measurement only; the lever (likely a takeoff curriculum) gets its own
pre-registration if convicted.

## Context

- AirDrillState (0.20 of resets) spawns the car ALREADY AIRBORNE climbing at
  the ball — deliberately, because the old grounded drill let the bot wait the
  ball down. Consequence: the drill teaches CLIMB→TOUCH, never TAKEOFF.
- MOVEMENT_PHASO0: execution is a capability gap (curriculum lever, not
  steering). FEAR_DECOMP: the critic taxes attempts made on LOW BOOST —
  aerials are maximally boost-hungry.

## Registered measurements (1v1 400k rows + 2v2 500k rows, newest checkpoint)

1. **Touch-height histogram**: ball z at every touch (buckets: <300 ground/
   dribble, 300-642 jumpable, 642-1200 low aerial, >1200 true aerial). The
   headline "is it going for aerials" number.
2. **Aerial-opportunity conversion**: readings with ball z > 642 at the
   reading, feasible (required speed < 1300): fraction where self (a) leaves
   the ground meaningfully (car z > 300 within the window), (b) touches the
   ball above 642, (c) average boost held at the opportunity (fear link).
3. **Drill-context completion** (does the trained skill exist at all?):
   300 episodes reset with `set_team_air_drill` (the live drill's python
   port, airborne spawn): touch rate within 3s and touch-height distribution.
4. **Takeoff probe** (the skill the drill never taught): 300 episodes spawned
   GROUNDED, boosted (70), under/near a high ball (z 900-1500, mild fall),
   opponent far: fraction that (a) jump within 1s, (b) reach car z > 500,
   (c) touch the ball airborne before it falls below 500.

## Registered interpretation map

- Drill completion HIGH + takeoff probe LOW → the gap is TAKEOFF initiation →
  the lever is a reverse curriculum on SPAWN HEIGHT (anneal the AirDrill spawn
  from airborne toward grounded as completion holds), implemented in C++ with
  an automatic difficulty controller (own pre-registration).
- Drill completion LOW → the drill itself stopped converting (regression) →
  investigate drill parameters before anything new.
- Opportunity conversion low WITH low boost held → the fear/resource link
  extends to aerials → boost-economy interaction becomes part of the story
  (still curriculum-first per the can't-vs-won't frame).
- Touch histogram already healthy (>642 share above ~15%) → the user's
  impression may be render-window sampling; report and re-check viz.

## Results (2026-07-15, checkpoint 30300098676; raw: `results/aerial_gap_30300098676.json`)

| measurement | result |
|---|---|
| touch heights 1v1 | ground 86%, jumpable 1%, low-aerial 9.4%, true-aerial 3.4% |
| touch heights 2v2 | ground 92%, low-aerial 3.9%, true-aerial 2.3% |
| opportunity conversion | 13.0% (1v1) / 8.1% (2v2); left-ground 74%/46%; boost held 46/40 |
| M3 drill, airborne spawn | touch 44% in 3s; of touches, 98% above goal height (median z 1098) |
| **M4 takeoff, grounded spawn** | **jump-1s 98%, car z>500 only 9%, aerial touch 0% (0/300)** |

**VERDICT (per the registered map): the gap is TAKEOFF→CLIMB, precisely.** The
bot reacts (98% jumps), can complete from mid-air (44%, touches above goal
height), holds adequate boost — but cannot climb from the ground even with 70
boost and zero pressure. The reverse curriculum stopped one stage early: the
jump→boost-climb transition has never been in the training distribution.

## The lever (pre-registered 2026-07-15): AirDrill ALTITUDE ANNEALING (C++, in-trainer)

One difficulty scalar D ∈ [0,1] interpolates the climber spawn: D=0 = today's
drill (airborne, climbing); D=1 = grounded takeoff (z=17, on wheels, facing the
ball's shadow, rolling at it — exactly the M4 probe). Spawn z lerps between the
two; below ~60uu the orientation/velocity switch to the grounded form.

Controller (all C++, in the learn-prep census section; state persisted in
RUNNING_STATS so it survives restarts): the aerial-conversion metric — high
(>goal-height) feasible readings converted by ANY above-goal-height touch,
style-proof — is EMA'd each iteration. Every 50 iterations: conversion not
degraded >20% relative to the last adjustment's reference → D += 0.05;
degraded → D −= 0.05 (automatic backoff). D starts at 0 = deploy is
behaviorally a NO-OP that ramps only while the metric stays healthy — which is
also why stacking it with the hours-old FEAR_MINE deploy is acceptable
one-lever-wise (separate proximal metrics, gradual onset).

Panels: `Curriculum/AirDrill D`, `Curriculum/Aerial Conv EMA`. Revert = flag
off (setter reverts to the fixed airborne spawn). Note: the controller runs
inside fnSteerUpdate (where readings live) — if steering is ever disabled, D
freezes in place and the drill keeps its last difficulty (safe degradation).

Pre-registered success bars (checked offline, ~3 days): M4 takeoff probe
car z>500 ≥ 40% and aerial touch ≥ 15% at whatever D the controller reached
(≥0.5 expected); live aerial-conversion EMA not below its deploy value. Fail →
flag off, record, rethink (candidate: dedicated takeoff reward term).

## Results — curriculum deployment

*(post-deploy tracking; bars above are frozen)*
