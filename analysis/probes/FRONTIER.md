# FRONTIER — the Potential Frontier: one axis (quasimetric potential) for the drill curriculum

**Pre-registered 2026-07-23 BEFORE trainer code, per repo doctrine (measurement
convicting the problem, pre-registered criteria, an automatic guard, a revert path,
one lever). Directive (user): the practice/drill curriculum should be driven by a
SINGLE axis — the quasimetric potential — not by hand-named difficulty axes or a
multi-cluster scheduler. "I only want one axis, quasimetric potential… general about
the game, encompassing aerials and more."**

## The one axis

For any candidate game state `s`, the frontier coordinate is

    d_goal(s) = min over the goal bank of  d( Embed(s), Embed(g) )
    d(x,y) = Σ_j relu( f(E(x))_j − f(E(y))_j )        # the shipped quasimetric map

i.e. the quasimetric distance from `s` to the nearest state from which the agent has
actually scored. This is ALREADY computed live (`GapState::Embed` +
`PPOLearner::MinBankDist(·, bankGEmb, dClamp)`; it is the same `d_goal` that feeds
`V_metric`/`gap_PK`). Small `d_goal` = a situation the agent routinely converts;
large `d_goal` = a situation it cannot yet convert. Aerials are not a special case —
a high ball the agent can't reach simply HAS a large `d_goal`, on the same ruler as
every other unconverted situation (positioning, team spacing, ceiling shots, …).

## The mechanism (replaces the frontier DEFINITION, not the drill machinery)

The drill infrastructure stays exactly as shipped: `fnSteerUpdate` mines candidate
match-play states each iteration, banks a per-mode pool (`FrontierPool::Fill`), and
`FrontierDrillState` resets a `practiceArenaFrac` slice of arenas into jittered
copies of pooled states. **We change only which candidates become pool entries.**

- **Incumbent (`frontierFearMining`)**: candidates = feasible-but-declined possession
  races, ranked by `Dz` (goal-critic vs critic disagreement).
- **Potential Frontier (`frontierPotential`)**: candidates = all mined match rows;
  select the ones whose `d_goal` sits in a moving quantile band `[θ, θ+w]` of the
  per-iteration candidate `d_goal` distribution, plus a fixed retention fraction
  drawn from below θ. A θ-controller (one scalar) walks θ outward as the drilled
  band is mastered. One axis; one scalar of state.

θ-controller (outcome-based, self-play-robust — drill outcomes are stationary even as
the league drifts):

- **Advance**: when the resolution rate on the drilled (practice-arena) rows —
  generic positive resolution: possession-win / net-positive extrinsic return within
  the drill horizon, NOT any conduct-specific detector — sustains above `advHi`,
  θ += `thetaStep`.
- **Retreat**: handled structurally. A too-hard band produces low resolution → its
  `d_goal` region stops shrinking → the shell stops being "mastered" → θ does not
  advance; combined with the retention fraction the mass stays learnable. The global
  peak-drawdown rating latch (below) is the hard backstop.

Monotone by construction: θ never decreases on noise (advance-only + retention band);
the success bank only grows, so the `d_goal` ruler only extends outward — the shell
follows the bank. No cluster k, no dwell scheduler, no per-cluster gates, no
hand altitude axis.

## Why this design (evidence + rationale)

Validated in an offline RocketSim testbed (`~/Projects/experiments/rltest`, aerials
from cold ground spawns, generic touch reward, n≤4 — HONEST scope: single-task, small
n, not self-play): a curriculum whose difficulty ORDER is the quasimetric distance to
a banked success discovers the difficulty axis with no human input (it reconstructs
"high ball = hard", `d` monotone with height) and drives base 0.000 → high-aerial
success, matching a hand-picked height axis. A θ-controller with one scalar of state
was measurably ~2× more step-efficient and more robust to critic drift than a
stateless practice-density variant — the reason θ is kept. The composition principle
that survived every failed variant: **use each signal only in its trustworthy domain**
— the quasimetric everywhere (geometry), the real critic on-support (mastery),
outcomes always (bank + advance). This spec is that principle wired into the shipped
`FrontierPool`.

## Integration map (file:line, current tree)

- Candidate mining + banking: `fnSteerUpdate` (`Learner.cpp:1926`); mining
  `~2072–2140` (`frontierRows`/`frontierDz`), banking `~2389–2514`
  (`FrontierPool::Fill` at 2511). fnSteerUpdate is called at `Learner.cpp:5297` —
  AFTER the ladder-map block (`~4830–5235`), so `gapSensor->bankGEmb` and `dClamp`
  are FRESH at fill time (no staleness).
- `d_goal`: `GapState::Embed` (`Learner.cpp:166`), `PPOLearner::MinBankDist`
  (`PPOLearner.cpp:114`), goal-bank embeddings `bankGEmb` (`Learner.cpp:5007`),
  `dClamp = dClampMult * gapSensor->dClampEma` (`Learner.cpp:4886`).
- Pool: `FrontierPool` / `FrontierDrillState`
  (`RLGymCPP/.../StateSetters/FrontierDrillState.h`). Selection = uniform `Sample`,
  so all ranking is at fill time (unchanged contract).
- Config: `CollectSteeringConfig` (`LearnerConfig.h:54`). NEW fields (below).
- θ persistence: `Learner::SaveStats`/`LoadStats` (`Learner.cpp:429/526`), init-if-
  absent alongside `air_drill_d` / `ladder_*`.

## New config (all default OFF/incumbent)

    // CollectSteeringConfig
    bool  frontierPotential   = false;   // master flag; when true, d_goal quantile selection replaces Dz
    float frontierThetaW      = 0.15f;   // quantile band width [θ, θ+w]
    float frontierRetainFrac  = 0.30f;   // fraction of pool drawn from below θ (retention)
    float frontierThetaStep   = 0.02f;   // θ advance per mastered check
    float frontierAdvHi       = 0.55f;   // drilled-row resolution rate to advance
    int   frontierThetaAdjustEvery = 25; // iterations between θ adjust checks

Runtime state on the Learner (persisted, init 0): `float frontierTheta = 0.f;`,
plus a resolution EMA. Boot override: `GGL_FRONTIER_POTENTIAL` (env, "1" → on),
read in `ExampleMain.cpp` with the existing `GGL_*` idiom, so a restart flips it
with no rebuild.

## Boot / restart / checkpoint contract (user requirement)

- **Toggle by restart, no rebuild**: set `GGL_FRONTIER_POTENTIAL=1` and restart
  (exit-99 planned restart or manual). Flag off = incumbent fear-mining.
- **Checkpoint-compatible with the live `checkpoints_6M` lineage**: reuses the
  already-trained map (`GAP_MAP_E/F.lt`) and banks; adds NO network and NO shape
  change; the only new persisted state is `frontier_theta` (+ resolution EMA) in
  `RUNNING_STATS.json`, loaded init-if-absent. A dashboard/wrapper restart RESUMES
  the current checkpoint with the mechanism on. **No checkpoint-folder change is
  required.**
- **New-folder fallback (unused here, stated for completeness)**: only a network
  shape change would force it; this spec introduces none. If one ever is,
  `cfg.checkpointFolder` (`ExampleMain.cpp:740`) is the one-line switch to a fresh
  lineage — a new/empty folder cold-starts cleanly.
- **Revert**: flag off → incumbent, same checkpoint. Rollback anchor:
  `checkpoints_6M` golden archive (`best_r*`), unchanged by this feature.

## Phase 0 — SENSOR ONLY (actuate nothing; the entry gate)

With `frontierPotential` on, compute `d_goal` for every mined candidate row but keep
banking by the incumbent `Dz`/uniform path. Log, per iteration:

- `Frontier/Dgoal {P10,P50,P90}` over candidates; `Frontier/Dgoal Std`.
- `Frontier/Dgoal Dz Spearman` — rank-correlation of `d_goal` vs the incumbent `Dz`
  on the same rows (do the two frontier definitions agree at all?).
- `Frontier/Dgoal vs BallZ Spearman` and `vs BallDistToNet` — sanity that `d_goal`
  ranks a physical difficulty proxy monotonically (the testbed's "discovers height"
  check, on real match states).
- `Frontier/Map Ready` (mapUpdates ≥ warmup AND banks filled).

**Phase-0 pass gate (on record, decide before Phase 1)**: over a stable window with
`Map Ready`, `Dgoal vs BallZ Spearman` ≥ 0.3 (d_goal ranks obvious difficulty
sensibly) AND `Dgoal Std` bounded away from 0 (non-degenerate axis) AND `Dgoal Dz
Spearman` reported (any value — this quantifies how different the new frontier is).
STOP and diagnose if `d_goal` is degenerate or anti-correlated with the difficulty
proxy.

## Phase 1 — ACTUATION (behind the flag, after Phase 0 passes)

Swap the fill selection to the `d_goal`-quantile band + θ-controller. Guards:

- **Rating latch (inherited, automatic)**: `fnRatingGuard` peak-drawdown
  (`ratingPeakTrip=110`) already halts all `FrontierPool::Fill` when tripped
  (`!steerRatingTripped` at 2398) — the exact incident backstop, for free.
- **Match-play is the transfer meter**: only `practiceArenaFrac=0.30` of arenas
  drill; the other 70% is ordinary self-play, and Rating is measured on the raw
  policy across the pool. Judge the feature on match Rating slope, never on drilled
  reps.
- **θ bounds**: θ ∈ [0, 0.9]; never advance past 0.9 (keeps a hard-ceiling tail out
  of the drill band).
- **Kickoff boot probe (inherited)**: `BootSanityProbe` already gates a resumed
  policy on touching kickoff balls.

**Phase-1 pre-registered success criterion**: match Rating slope over a matched
window NON-INFERIOR to the `frontierFearMining` incumbent (the repo's standard bar),
AND the drilled-band resolution rate rises as θ advances (the wave moves). Else
revert (flag off). Report `Frontier/Theta`, `Frontier/Drilled Resolve`,
`Frontier/Pool Dgoal {P50}`.

## Deferred (not in this change)

- Subsuming the separate `AirDrillCurriculum` altitude axis into the same `d_goal`
  ranking (the "one axis truly for everything" end state) — after Phase 1 proves the
  `d_goal` frontier already covers aerial states in the pool.
- Unifying the k-cluster activation-steering direction onto the same axis — a
  separate system (`fnMetaUpdate`), already OFF (`steering.meta=false`).

## Method note

Everything above reuses shipped, battle-tested machinery (the map, the banks, the
pool, the drill setter, the rating latch). The Potential Frontier is a change of the
frontier's DEFINITION (from feasible-decline+Dz to the single quasimetric axis) plus
one scalar θ-controller — not a new subsystem.
