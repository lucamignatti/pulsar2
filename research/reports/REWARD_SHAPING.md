# Pulsar reward shaping — canonical report

Merges `REWARD_SHAPING_V2.md` (the proposal) and `REWARD_SHAPING_ADDENDUM.md` (the measurements
that ran). **The addendum wins every conflict.** Where V2 proposed and the addendum measured, only
the measured result is recorded here. v1 (`REWARD_SHAPING_PROPOSAL.md`) is superseded entirely;
its surviving content is in the graveyard (§10) with kill reasons.

---

## 0. STATUS BOX — read this first

**Nothing was touched.** No code, no config, no reward term, no checkpoint, no trainer process, no
GPU work, no build. `git status` shows only the two pre-existing working-tree modifications
(`cpp-interface`, `tools/trainerctl`) that were there before this analysis began. **This document
is the only artifact produced.** Nothing in it has been approved, staged, or shipped. Every stage
below is a proposal awaiting your decision.

**The live run changed underneath this analysis.** Commit `45a59d5` "Residual blocks + asymmetric
policy/value sizing (cold start)" went live **2026-07-24 23:36** (parallel session, user-directed):

| | before | now |
|---|---|---|
| checkpoint dir | `build/checkpoints_6M` | **`build/checkpoints_resid`** (`src/ExampleMain.cpp:778`) [M] |
| wandb | `6M-1152` | `resid-768p-1280v` |
| state | **FROZEN at 3.44 B, not overwritten** | **cold start at step 0**, ~137 M steps / 489 iters, `Rating/1v1` ≈ **21** (golden archive `best_r21_100157440`) [M] |
| arch | 1152 trunk + 1152 policy + 1152 critic | residual blocks everywhere (`addResiduals`, `:714-721`); trunk {1152×3} `:723`; **policy SHRUNK {768×3}** `:724`; critic {1280×5} `:725`; goal-critic {1280×5} `:673`; reach φ/ψ 384 `:731-732`. ~37.96 M params [M] |

**The reward stack is bit-identical across that commit.** `git diff 9856813 45a59d5 -- src/ExampleMain.cpp`
touches exactly two hunks, both inside `main()` at `:670+` — no reward term, no `TEAM_SPIRIT`, no
gamma, no `serveFrac` [M]. Everything this document says about *reward semantics* still applies to
the live binary. Everything it says about *numbers* was measured on the frozen run.

**Two consequences, and they pull in opposite directions:**

- **Every quantitative gate constant here is historical.** They were measured on 6M-1152. They are
  marked **[6M]** and **do not transfer**. S0 (instrumentation) is now *doubly* blocking: there is
  no baseline on the resid run at all.
- **The dominant risk in the whole analysis just evaporated.** S1–S4 needed guards, staged windows
  and branch-point backups because changing rewards under a mature policy is a distribution shock.
  **A run at 137 M steps has no mature policy to damage.** This project's own doctrine
  (`PULSAR5.md`) says the formative window is exactly when scaffold weights do their work. The case
  for shipping the *correctness fixes* now is stronger than V2 implied — letting a fresh policy
  spend 3 B steps learning around a verified boost sawtooth is worse than the 6M situation, not
  better.

**PHASE_B: not imminent. Demoted to a standing trap.** The trigger reads
`cfg.checkpointFolder / PHASE_B_MARKER` (`src/ExampleMain.cpp:1254-1276`), which is now
`checkpoints_resid` — a fresh folder with **no marker** (verified: markers exist in
`checkpoints_4.0`, `checkpoints_5.0`, `checkpoints_5.0v3`,
`checkpoints_6M_prehr_backup_20260724_084117`; **absent** in both `checkpoints_6M` and
`checkpoints_resid`) [M]. `PHASE_B_RATING_TRIGGER = 1200` (`:81`), `PHASE_B_TRIGGER_STREAK = 3`
(`:82`), Rating ≈ 21. It will matter again in a few days, not tonight. **When it does fire it
means 33 % 2v2 + 33 % 3v3, `TEAM_SPIRIT` 0.3 → 0.6 (`:1063`), and `exit(99)` + relaunch — every
gate in this programme that straddles the flip becomes uninterpretable**, because a Nexto "serve"
stops meaning the same thing, every ZeroSum-wrapped term's credit is rewritten, and in-process
guard state resets. Note also that **1200 was written before `Rating/1v1` was known to be
pool-inflated** (memory `rating-pool-inflation-5.0v3`: 570 → 1125 on 5.0v3 while the external
yardstick moved far less; the comment at `:78-80` justifies 1200 against a "measured 1v1 plateau
~1250-1300 on the 3.1 lineage"). The threshold is a *pool-relative* number being used as an
*absolute* competence gate. Re-derive it before the resid run approaches it.

**Restarts on 2026-07-24 at 23:01 and 23:36 were user-initiated (exit 143), not phase-B exits.**
The 23:36 one is the resid cold start.

---

## 1. Bottom line

1. **Reward shaping is not the top lever, and the honest expected effect of the whole reward
   programme is small.** Four non-reward levers rank above every reward change (§5). This is not
   a re-weighting of the objective — it is **three correctness fixes and a pile of read-only
   instrumentation**, moving Goal's real credit-density share by **< 1 pp** (32.3 % → ~32.4 %)
   [D, 6M].
2. **The shippable core is S0 + S1 + S3 = 1.7 B ≈ 7 h.** The full programme as scoped is 3.2 B
   ≈ 13 h, but **S2 and S4 are BLOCKED** (§6) and should not be counted as plan.
3. **Nothing reward-side ships before the guard exists** (§8). The existing rating latch is blind
   to any decline slower than **627 Elo/B** on the peak arm and **336 Elo/B** on the EMA arm inside
   a 0.5 B stage [D] — far slower than a run's own improvement rate. Doctrine requirement (3) is
   currently unmet for every reward change anyone has proposed.
4. **Highest-conviction change: S1**, muting `GuardedPickupBoostReward` — a positive-part-only
   payment on the *identical* `sqrt(boost/100)` function `CarEnergyPotentialReward` already carries
   two-sidedly. One line, revertible in-process, and its own source comment argues for it
   backwards (§7.1).
5. **Almost everything else was killed by measurement, not opinion.** The boost-term family, the
   threat/offensive-potential family, and goal-speed scaling all returned nulls or opposite signs
   when the deferrals' own re-open conditions were actually run (§10). Six of v1's seven reward
   items do not ship; two of V2's three deferrals converted to kills.
6. **What is actually between Pulsar and the best bots is not the reward stack** (§12): a 90-action
   table with no intermediate stick deflections, 0 % replay-derived reset states, a ~1.26 s credit
   window, and a yardstick served on 4.89 % of iterations.

---

## 2. Provenance convention

| mark | meaning |
|---|---|
| **[M]** | measured, or verified in source at the cited `file:line` |
| **[D]** | derived by arithmetic from [M] quantities |
| **[E]** | estimated; band given |
| **[U]** | unverified / unmeasured |
| **[6M]** | **measured on the now-frozen `6M-1152` run.** Qualitative finding transfers (the reward stack is unchanged); **the number does not.** Re-measure on `resid` before use as a gate constant. |

Nothing is asserted without one of these. Weights are given as **nominal → post-ZeroSum credit
density** in reward-units/player-step. Line numbers verified against HEAD `45a59d5` on 2026-07-25.
**Note:** V2's citations above `src/ExampleMain.cpp:670` are stale by **+39 lines** (the resid
commit's two hunks land at `:670` and `:775`); e.g. `serveFrac` is now `:1221`, not `:1182`.
Citations below `:670` — the entire reward block — are unchanged. `Learner.cpp`,
`CommonRewards.h`, `EnvSet.cpp`, `Reward.h` were not touched by the commit.

---

## 3. Corrected measurement baseline — all [6M]

### 3.1 The trend: deceleration, not plateau

**v1's headline claim ("the run has PLATEAUED, slope from 1.7 B = −0.28 ± 2.95 pp/B") is
FALSIFIED.** It used overlapping 150 M windows, which understate the SE and, fitted through the
1.8 B excursion, manufacture a flat line.

Nexto goal share by **non-overlapping** 0.3 B block, segment-clean (the naive parse double-counts
the 2.9259 B and 3.3255 B restart replays) [M, 6M]:

| block (B) | 0.0–0.3 | 0.3–0.6 | 0.6–0.9 | 0.9–1.2 | 1.2–1.5 | 1.5–1.8 | 1.8–2.1 | 2.1–2.4 | 2.4–2.7 | 2.7–3.0 | 3.0–3.3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| share % | 12.17 | 18.06 | 21.35 | 22.61 | 25.45 | 27.84 | **40.07** | 35.29 | 36.04 | 37.79 | 39.24 |

Non-overlapping OLS: from 0.0 B **+9.17 ± 0.70 pp/B** (t = 13.0); from 1.5 B **+5.62 ± 1.76**; from
2.1 B **+4.23**. The correct characterisation is **deceleration**, with the last four blocks rising
monotonically.

**Consequences that bound how any gate may be built** (these are methodological and *do* transfer):

- **The null for any slope criterion is the run's own improvement rate, not zero.** v1 sized
  "+5 pp/B ≈ 3.6σ" against a zero null; against the real null it is **0.47σ** [D]. Every v1 slope
  criterion passes on a null result.
- **Slope-vs-frozen-baseline gating is invalid under deceleration** — the null itself is moving.
  Use **level tests against a frozen contemporaneous baseline**.
- On 6M, parity was (50 − 39.24)/4.23 = **2.54 B ≈ 11 h** [D]. **This benchmark is dead** — that
  run is frozen (§5).

### 3.2 The ~1.8 B discontinuity — REAL, CAUSE UNRESOLVED [6M]

27.84 % → 40.07 % in one block, settling to 35–39 %.

- **Affirmatively excluded** [M]: no restart (0 → 2.945 B is one uninterrupted process); no
  checkpoint reload/quarantine/boot probe; no serve-rate change (48 → 49 serves/block); no
  denominator-composition change; no PHASE flip; no guard latch; no warmup threshold; no mechanism
  toggle (`Ladder/Wire Active` → 1 at 0.0011 B and never changes).
- **Statistically real** [M]: burst-level permutation p = 0.0001 (20 k permutations); +3.70 sd off a
  log-odds trend fitted without it; leave-one-burst-out moves it only 39.36–40.86 %.
- **Shape**: a ~0.1 B ramp with overshoot, not a step (rolling 15-burst 24.53 @1.702 B → 45.48
  @1.872 → 35.10 @2.001). `Rating/1v1` traces the same rise-and-partial-retrace [M].
- **Best-supported cause**: a genuine, partly transient policy excursion — both halves of the ledger
  move together at constant pace. **Not established.** One untestable candidate remains: `oppTeam =
  RandInt(0,2)` (`Learner.cpp:3452`) is the *second* LCG draw off the same seed as the serve
  decision, and Nexto has a replicated orange-observer asymmetry (`NextoOpponent.cpp:151-152`,
  `:195`) — simulation bounds its block contribution at 1–2 pp, not 12 [D].

**Binding consequence: no comparison may straddle it,** and **no stage baseline may be frozen
across a discontinuity of this kind.** That rule transfers even though the specific window does not.

### 3.3 The yardstick is broken in three ways — mechanism transfers, numbers are [6M]

1. **Realized Nexto serve rate 4.89 %** (581/11,884) against `serveFrac = 0.15f`
   (`src/ExampleMain.cpp:1221`) — a **3.07× shortfall** [M, 6M]. Fully diagnosed as a **closed-loop
   wall-clock resonance**: `Math::GetRandEngine()` re-seeds a `thread_local minstd_rand0` from
   `RS_CUR_MS() + hash(tid)` (`RocketSim/src/Math/Math.cpp:58-63`) on the fresh `std::jthread`
   created every iteration (`Learner.cpp:3972`), and `std::hash<thread::id>` is constant across
   recycled threads (measured 9869274154721403212 for 6/6), so the roll at `Learner.cpp:3429` is a
   **deterministic sawtooth of the millisecond clock** — period 2147483647/16807 ms = **127.773 s**,
   i.e. a **19.17 s serve gate every 127.77 s**. A served iteration costs **19.1 s** vs **3.1 s**
   unserved because **Nexto runs on CPU** (the boot CUDA probe fails on a baked-in `device("cpu")`
   constant inside the traced `emb_convertor` graph). **One serve consumes its own gate.**
   Closed-loop simulation reproduces **4.75 % [4.58, 4.93]** vs a measured 4.90 %; with the serve
   cost removed it gives **15.00 %** [M]. **Free detector for this bug class: serve-gap sd/mean =
   0.10** (a Bernoulli process gives 1.0).
2. **The dose drifts monotonically with training**: 7.00 % over the first 1 k iterations → 4.10 %
   over the last, because served-iteration duration grew 12.6 s → 20–29 s while unserved stayed at
   ~3.1 s [M, 6M]. This confounds *every* before/after comparison.
3. **The unit of observation is the serve BURST, not the goal.** Serves fire in bursts of 1–3
   separated by gaps of 26–34; a 0.3 B block holds **~35 independent clusters**, not ~48,000 goals.
   Cluster-bootstrap SE per block is **1.8–2.7 pp, ~4× the binomial SE** [M, 6M]. **Every SE v1
   quotes is wrong by ~4×**, and v1's "−14.1σ" figure is **void** (it assumes independent Bernoulli
   draws on a near-periodic process). Burst-share sd = **0.1251**, 114.8 bursts/B [M, 6M].
4. **The share is measured on the training reset mix** — `BallNearCarState(600,900)` 0.30,
   `AirDrillState` 0.20, `AirPlayState` 0.15, `KickoffState` 0.10, `RandomState` 0.25
   (`src/ExampleMain.cpp:398-409`) — only **10 % kickoffs**. **It is not a match-play scoreline.**

### 3.4 The magnitude currency — post-ZeroSum, like-for-like

**This is the single most important methodological correction in the document, and it is
architecture-independent.**

`ZeroSumReward::GetAllRewards` sets `_lastRewards = rewards;` at **`ZeroSumReward.cpp:5`**, i.e.
**before** the transform loop at `:19-28` (verified at source) [M]. `EnvSet.cpp:259-260` then
deliberately overwrites the already-post-transform `output` from `:227` with that pre-transform
child. So:

- **wrapped terms log the raw one-sided child**, while
- **unwrapped antisymmetric terms** (e.g. `GoalReward`, `CommonRewards.h:37-43`) **cancel to ≈ 0**.

**The naive per-term panel table is therefore apples-to-oranges.** In 1v1 the post-transform value
is exactly `r_i − r_j` (TEAM_SPIRIT cancels algebraically at n = 1, `ZeroSumReward.cpp:19-28`),
whose **mean is identically zero** for all 15 zero-sum/antisymmetric terms. Of v1's 16 rows: 2 are
structurally zero, 6 measure drift not magnitude, 7 are understated by exactly 2×, 1 is exact.

The correct currency is a **dispersion** statistic. Two, answering different questions:

- **A — credit density**: `w · E|r_i − r_j|` per player-step. What shapes the local gradient.
- **C — optimum-relocating magnitude** per episode. For exact PBRS the discounted episodic sum
  telescopes, so C is only the **unrefunded terminal residual**.

For 8 terms (7 non-negative mutually-exclusive event terms + TimeCost) the panel maps exactly
`A = 2·w·panel` — validated by Demo panel 1.295e-5 vs `Player/Demo Rate` 1.281e-5 (1.1 %) [M]. For
the 5 state-function potentials the joint `(r_i, r_j)` was measured offline on
`research/data/dataset.npz` — validated by CarEnergy computed `E[r]` −2.874e-3 vs live panel
−2.891e-3 (0.6 %) [M]. Goal's rate is exact and free: `Episode Length` is literally
`1/normalTermFrac` (`Learner.cpp:4060-4062`) and `GoalScoreCondition` is the only NORMAL terminal
(`src/ExampleMain.cpp:359-361`), so `p_goal = 1/228.49 = 4.376e-3`/player-step [M, 6M].

**Window 2.5–3.0 B, 1v1, post-ZeroSum, weighted. All [6M].**

| # | term | w (nominal) | A (density) | A share | C (/episode) | C share | source |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | **GoalReward** | 150 | **0.6565** | **32.34 %** | **150.00** | **61.2 %** | [M] exact, 150 × p_goal |
| 2 | CarEnergyPotential | 15 | 0.5362 | 26.41 % | 6.34 | 2.6 % | [M] offline replay, 0.6 % validated |
| 3 | ConsecutiveAirTouch | 30 | 0.25 [0.18–0.49] | 12.31 % | ≥5.53 | 2.3 % | [E] widest band |
| 4 | BallToGoalPotential | 75 | 0.1871 | 9.22 % | 26.35 | 10.8 % | [M] offline |
| 5 | AerialTouch | 120 | 0.1181 | 5.82 % | 26.97 | 11.0 % | [M] exact 2·w·4.91881e-4 |
| 6 | AirInterceptPotential | 40 | 0.1019 | 5.02 % | 0.34 | 0.1 % | [M] offline replay |
| 7 | BallProximityPotential | 4 | 0.0596 | 2.94 % | 1.25 | 0.5 % | [M] offline replay |
| 8 | TouchAccel | 10 | 0.0538 | 2.65 % | 12.29 | 5.0 % | [M] exact 2·w·2.68857e-3 |
| 9 | TeamPressure | 0.15 | 0.0277 | 1.36 % | 6.33 | 2.6 % | [M] exact formula |
| 10 | **GuardedPickupBoost** | 6 | **0.0266** | **1.31 %** | 6.07 | 2.5 % | [M] exact 2·w·2.21311e-3 |
| 11 | TimeCost | 0.01 | 0.0100 | 0.49 % | 3.23 | 1.3 % | [M] exact; **differential contribution 0** |
| 12 | WallJumpToBall | 30 | **0.0013** | **0.064 %** | ~0 | 0.0 % | [M] live-panel zero-fraction |
| 13 | Demo | 37.5 | 9.71e-4 | 0.048 % | 0.22 | 0.09 % | [M] exact |
| 14 | OpposedSave | 25 | 3.79e-4 | 0.019 % | 0.09 | 0.04 % | [M] exact |
| 15 | KickoffRace | 25 | 1.28e-4 | 0.006 % | 0.03 | 0.01 % | [M] exact |
| 16 | FlipReset | 40 | 1.5e-5 | 0.0007 % | ~0 | 0.0 % | [M] nonzero on 24/20,421 points |
| | **TOTAL** | **607.66** | **2.030** | | **245.0** | | |

Two corrections to figures circulating earlier in this workflow's record:

- **WallJumpToBall is 0.064 %, not 3.7 %.** The 3.7 % came from an offline proxy on a
  5.0v3-lineage checkpoint. The live panel reads **exactly 0.0 on 1,865 of 3,185 logged points**;
  calibrating the estimator against `OpposedSaveReward` (payout exactly 1.0, so its panel mean *is*
  its rate) gives **p = 4.66e-4/player-step** [M, 6M].
- **FlipReset is not exactly zero** (24 nonzero points of 20,421, max 9.96e-4) [M]. "Provably zero
  distribution shift" is not literally true for it.

Key ratios [D, 6M]: **Goal / AerialTouch = 5.56×** on credit density (v1 said 9× and derived
"inverted by ~50×"; both are wrong — v1 compared Goal's *post*-ZeroSum value against AerialTouch's
*pre*-ZS child mean). The aerial family (AerialTouch + ConsecAir + AirIntercept + WallJump +
FlipReset) is **42.8 % nominal but 23.2 % of credit density and 13.4 % of relocating magnitude**.

**Nominal-weight shares are not the operative quantity.** `GAE.cpp:50-53` divides every reward by a
running `returnStd` before GAE, so uniformly scaling the whole stack is an exact no-op and changing
its **composition** rescales every other term.

**But the fixed-budget premise does not bind at stage timescale** (addendum C1, and this reverses
V2's strongest rhetorical argument): `returnStat` is a never-forgetting `WelfordStat`
(`Util/WelfordStat.h`) incremented `min(maxReturnSamples = 150, batch)` samples/iteration
(`Learner.cpp:4771-4775`) and persisted (`:445`/`:543`). At `tsPerItr = 200'000`
(`src/ExampleMain.cpp:568`), 3.44 B steps ≈ 2.58 M samples; a 0.5 B stage contributes 12.7 % of the
new total, so a **+3.1 % density change moves `returnStd` by +0.40 %, not 3.1 %** [D]. So:
**additions genuinely add at this horizon, and deletions are not the bargain V2 claimed.** V2's
"~26 % of a stage's variance stays baked in after a revert" is **backwards** — ~99 % of the
pre-stage variance is retained and the stage barely registers. *(On the resid run at 137 M steps
this cuts the other way: the Welford has far fewer samples, so a stage is a much larger share of
it. Re-derive before quoting.)*

### 3.5 Measurement defects that must NOT be used as criteria

- **`Average Step Reward` (`Learner.cpp:4004`)** must equal exactly **−0.0100** in a symmetric 1v1
  buffer, since `TimeCost` is the sole non-zero-sum term. It reads **+0.2946 ± 0.0152** over
  2.5–2.95 B [M, 6M]; 61 % of iterations sit in a near-zero mode, ~8 % below −0.011 (Nexto), ~31 %
  above +0.3 (league). The residual is **unexplained [U]**. v1's Stage-1 criterion (b) is literally
  "still reads exactly −0.0100" — **a criterion the current code already fails.**
- **The whole-stack zero-sum invariant is false as stated** (addendum C2). `TimeCostReward` is
  unwrapped and non-antisymmetric (`src/ExampleMain.cpp:252`): the stack sums to a **constant
  −0.02/pair-step**, not 0. Benign for PSD ranking (`PSDController.cpp:439-459` is a **mean
  per-step** competitive reward, so a constant shifts every candidate equally). **Not benign** for
  `Average Step Reward` as an identity check whenever episode length moves.
- **`GAE/Returns STD` is retired as an identity check** (addendum C1). A 0.5 B stage moves it
  **0.40 %**, so "no step > 1 %" cannot fail even for a genuinely non-identity deploy. Report it
  (`Guard/Returns STD Jump`); **do not reset it** (§10).

---

## 4. Diagnosis — binding constraints with their convicting measurements

| # | constraint | convicting measurement | status |
|---|---|---|---|
| **D1** | **Defense is the stalled axis** | Log-odds decomposition 2.1→3.3 B: total +0.1565, of which `d ln(GF)` = **+0.1686** and `−d ln(GA)` = **−0.0121**. GF/serve-iter 358.9 → 424.8 (+18.4 %); **GA/serve-iter 657.9 → 665.9 (+1.2 %)**, flat since 1.8 B | **[M, 6M]** |
| **D2** | **Nothing in the stack prices defender position** | `BallToGoalPotentialReward::Phi` is a pure function of **ball** position (`CommonRewards.h:246-252`). `OpposedSaveReward` is 0.019 % of density and fires only after the engine's shot detector arms (`:704-705`) | **[M]** |
| **D3** | **The boost economy has a positive-part-only ratchet on a two-sided axis** | `GuardedPickupBoostReward` (`CommonRewards.h:642-646`) pays `sqrt(b/100) − sqrt(b_prev/100)` **positive-part only**; `CarEnergyPotentialReward::Phi` (`:191`) contains the identical `sqrtf(0.01f*p.boost)` **two-sided**. Marginal pickup income is **5.6× higher at boost 0 (2.079) than at boost 88 (0.372)** [D]. Live `Player/Boost = 15.78 ± 0.064` with a **median of 0** (60.1 % of player-steps below 1 boost, 70.8 % below 12) [M, 6M] | **[M]** algebra; attribution **[E]** — closed by the `Diag/Boost At Pickup` pre-gate |
| **D4** | **`AerialTouchReward`'s documented AND has collapsed** | `float airFrac = RS_MIN(1.0f, player.airTime / MAX_CREDIT_AIR_TIME);` (`CommonRewards.h:523`), `MAX_CREDIT_AIR_TIME = 1.75f` (`:501`); the engine clears `air_time` only under `is_on_ground` (`RocketSimV3/rocketsim/src/sim/car/base.rs:569-572`). After 1.75 s of one flight every subsequent touch scores `airFrac = 1.0`, so `heightFrac = RS_MIN(ballFrac, airFrac)` (`:524`) degrades to a pure ball-height gate on the stack's largest scaffold | **[M]** structural. **The farm ceiling is NOT convicted**: realized payout is 0.088 reward/s/player against a ~121 reward/s ceiling, **1,375× below it** [D, 6M] |
| **D5** | **`TouchAccelReward` is direction-blind** | `CommonRewards.h:341-356`: pays `(curSpeedFrac − prevSpeedFrac)⁺` on any `player.ballTouchedStep` with **no direction term**. A power touch into our own net scores identically to one at theirs | **[M]** structural; the *frequency* is **[U]** — `Guard/OwnwardTouch Frac` closes it |
| **D6** | **The yardstick's dose is 1/3 of nominal and drifting** | §3.3 | **[M, 6M]** |
| **D7** | **No reward change has an automatic actuator** | `EnvSet.cpp:229` reads `weightedReward.weight` directly; `WeightedReward` (`Reward.h:62-75`) has **no scale member** (verified at source — the struct carries only `reward`, `weight`, `gated`, `zeroSumPtr`); `steerRatingTripped` gates only the advantage injections and opponent draws | **[M]** — §8 |
| **D8** | **The only guard that can see update damage is structurally blind** | Rating evals arrive every ~2.24 M steps. `ratingPeakDecay = 0.5`/eval = 223 Elo/B, so the peak arm needs a decline **> 627 Elo/B** to reach `ratingPeakTrip = 200` (`src/ExampleMain.cpp:992`) inside 0.5 B; the EMA arm (`decay = 0.995`, lag 199 evals) needs **> 336 Elo/B** to reach `ratingDrawdownTrip = 150` (`:995`) [D] | **[M]** algebra + **[D]** simulation |

**Two v1 diagnoses are struck:**

- **"Goals against are ~63–74 % of the Nexto deficit."** A **category error**: 22,503/36,127 =
  62.3 % is the concede share of *goals*, not a decomposition of the gap to parity. On the metric
  the programme is graded against, `d(share)/dGF = 1.4353e-5` vs `|d(share)/dGA| = 9.156e-6` —
  **one extra goal scored is worth 1.57 goals prevented** [D, 6M]. The defensive case must rest on
  D1 (headroom), not on a leverage claim pointing the other way.
- **"The stack cannot express defense at all."** False. `BallToGoalPotential` at w = 75 is a
  symmetric threat differential paying **0.0106 reward/uu** at `dOwn = 1000` — moving the ball
  2000 uu out of danger pays ~21 units, 14 % of a Goal [D]. A concede is already priced at **−176**
  (−150 Goal, −26.35 unrefunded B2G terminal residual) [D]. The real gap is narrower: nothing
  prices *where the defender was*.

---

## 5. The lever ranking

"Risk" weights probability × size of trajectory damage by revertibility and by whether an
**automatic actuator** exists.

**Row 0 has been rewritten.** V2's benchmark was *"do nothing for 11 hours (+4.23 pp/B)"*, and that
was the 6M run's trajectory. **That run is frozen.** The resid run is 137 M steps into its own
formative window and **its trajectory is unmeasured**. There is currently **no null to beat**,
which is a reason to build the instrument, not a licence to skip it.

| # | lever | cost | risk | auto guard? | expected effect | verdict |
|---:|---|---|---|---|---|---|
| **0** | **The do-nothing null** | 0 | 0 | n/a | **UNMEASURED on `resid`** [U] | Was +4.23 pp/B on 6M [M, 6M] and beat most of the programme. **It must be re-measured before any stage can be graded.** This is now an argument *for* S0, not a rival to it. |
| **1** | **Re-trace Nexto's `emb_convertor` for CUDA** | offline Python, no restart of its own | **zero** (no trainer contact) | n/a | recovers **18.6 % of wall clock** (8,920 s of 48,024 s) [M, 6M] **and** removes the resonance so realized dose rises toward 15 % | **SHIP FIRST.** Larger and more certain than any reward stage, and it repairs the instrument every gate depends on. The new run needs that instrument re-baselined from scratch anyway. |
| **2** | **Offline goal-critic λ sweep** on a copied checkpoint (`research/tools/goal_critic_audit.py` already computes `A_goal` by GAE at `lam=0.95`; add `lam ∈ {0.95, 0.98, 0.99, 1.0}`) | ~4-line edit + one niced CPU rollout | **zero** | n/a | converts the λ question from opinion to number | **SHIP SECOND.** λ = 0.95 truncates the goal channel by 84× (γλ = 0.949430 → 19.77 steps = **1.32 s** against a γ_goal horizon of 111 s; `GOAL_CRITIC_AUDIT.md:20-24`). **But** the existing audit shows `D_G` scores 0.678 vs `A_goal` 0.680 — the authority is in `V_goal`, not the λ sum. **Sweep, do not change.** |
| **3** | **S0 — instrumentation restart** (§6) | one restart, ~0.7 B | low | *is* the guard | 0 by construction | **PREREQUISITE for everything reward-side, and now doubly so** — the resid run has no baseline at all. |
| **4** | **Add the missing `!steerRatingTripped` to the goal-critic blend** (`Learner.cpp:4821`) | 1 line | ~zero | *is* the guard | 0 expected | Pure correctness. β = 0.25 is the **largest** of the four advantage injections and the **only** one unguarded (`:4759` headroom guarded, `:4900` RND guarded, `:5227` Ladder guarded, **`:4821` not**) [M]. Free rider on S0. |
| **5** | **`GGL_FRONTIER_POTENTIAL=1`** (`src/ExampleMain.cpp:1115`) | one env var, **no rebuild** | ~zero (pure telemetry) | inherits the rating latch | free information on the curriculum axis | Free rider on S0. Returns FRONTIER.md's own pre-registered Phase-0 gate. Phase 1 (the θ-controller) is **not implemented**. |
| **6** | **S1 — mute `GuardedPickupBoostReward`** | 1 line + rebuild + restart | low | needs #3 | **[E] 0 to +1 pp/B**, plus a testable `Player/Boost` discontinuity | **Highest-conviction reward change**, and **stronger now than V2 implied**: a formative policy has not yet built a style around the ratchet. |
| **7** | **S3 — direction factor on `TouchAccelReward`** | ~15 lines | low-medium | needs #3 | **[E] 0 to +1 pp/B** | Removes a genuine mispricing at a bounded, whiff-tax-free, surface-rebound-immune construction (§7.2). |
| **8** | **S2 — `AerialTouchReward` airFrac AND-restore** | ~6 lines | medium | needs #3 | **[E] −0.5 to +0.5 pp/B, sign ambiguous** | **BLOCKED** (§6). Correctness fix on the largest scaffold that also deflates the one measurably improving axis. |
| **9** | **S4 — `ConcedeAccountabilityReward`** | new term | medium | needs #3 | **[E] 0 to +1.5 pp/B, LOW confidence** | **BLOCKED** (§6). The only addition and the only term aimed at D1. Last. Conditional. |
| 10 | `goalCritic.lambda` as its own field, raised **only if #2 convicts** | ~5 lines, `Learner.cpp:4794` | low | free: 2-line latch add; the blend is std-matched (`betaEff = β·advStd/goalAdvStd`, `:4822`) so λ_goal cannot change the effective LR | unknown until #2 | Best *code* lever after the free ones. One-literal revert, nothing checkpointed. |
| 11 | Global `gaeLambda` 0.95 → 0.97 | 1 literal | **medium-high** | none | — | **There is no advantage normalisation anywhere** (`PPOLearner.cpp:572` uses raw `advantages`; `Learner.cpp:5370` passes `tAdvantages` straight through) [M], so λ 0.95→0.97 raises `std(A)` ≈ **1.26×** = an uncontrolled effective-policy-LR bump. Two levers in one. Strictly dominated by #10. |
| 12 | FRONTIER.md Phase 1 (θ-controller) | real implementation | medium | inherits the latch | — | Doctrine-perfect design, zero of it built. Queue behind #5's readout. |
| 13 | Entropy / LR schedule | 1 literal | high, **effectively irreversible** | none | — | On 6M: `Policy Entropy` 0.617 normalized ⇒ ~**16 of 90 actions** effective support, flat since 1.2 B, at `entropyScale = 0.035` [M, 6M]. Endgame lever. |
| 14 | **Raise `serveFrac` 0.15 → 0.35** (`:1221`) | 1 literal | high | latch-covered | **NEGATIVE: −1.14 pp/B** [D, 6M] | **Do not ship.** It would realize ~12 %, not 35 % (§3.3); forcing a true 15 % at the current CPU cost is +37.5 % wall clock (SPS ×0.73). Under the zero-sum invariant every wrapped term charges us the opponent's income, against an opponent outscoring us 1.57:1. **Replaced by #1.** |
| 15 | Replay-derived `StateSetter` | **days** of offline pipeline | high, hard to A/B | none | — | The in-trainer half is nearly free (`FrontierPool::Entry` is already the exact schema), and **30 % of arenas already draw 60 % of resets from `FrontierPool`** (`src/ExampleMain.cpp:448-451`) ≈ 18 % of resets are already curriculum drills. **Exhaust the shipped, idle, guarded machinery first** (#5). A replay setter also changes state distribution AND effective opponent competence at once — no attribution path. |
| 16 | Net capacity | new cold start | catastrophic | n/a | — | **Just happened** (`45a59d5`). Off the table for a long while. |

**Falsifiers for this ranking, pre-registered:** (a) if the λ sweep shows `A_goal`'s held-out AUC
flat or falling in λ, #10 collapses to the bottom and reward work moves up; (b) if
`Frontier/Dgoal vs BallZ Spearman` < 0.3 or `Dgoal Std` ≈ 0, FRONTIER.md's own Phase-0 gate fails,
#12 is dead and #15 rises materially; (c) if `Diag/Boost At Pickup` mass sits near 45+ rather than
near 0, S1's mechanism is not live and S1 drops to the bottom of the reward list.

---

## 6. The staged plan — FINAL scope

**Global gate doctrine.** Slope gates are invalid under deceleration (§3.1) and goal-level SEs are
optimistic by ~4× (§3.3). **Nexto share is a HARM detector only** — its measured MDE for a level
contrast at a 0.5 B stage (57 vs 40 serves, σ = 12.38 pp) is **6.34 pp** [M, 6M], against programme
effects of 0–0.75 pp per stage: **a factor of 8 below resolution.** Pretending otherwise is exactly
the failure the doctrine was written against. **Efficacy moves to per-iteration panels, tested as a
local-linear intercept discontinuity** (§9.1).

**Every stage:** branch-point backup first (`cp -a build/checkpoints_resid
build/checkpoints_resid_s<N>_backup_<ts>`), edit, `cmake --build build -j8` (safe while the trainer
runs — the process keeps its inode), restart. **Reward terms are not checkpointed** —
`BuildRewards` constructs fresh per-arena objects at every boot via `MakeEnv`
(`src/ExampleMain.cpp:412`) — so every revert is resume-compatible with any checkpoint.

### Scope table

| stage | what | status | steps | wall clock @70 k SPS |
|---|---|---|---:|---:|
| **S0** | instrumentation + actuator + CUSUM + panels. **No reward change.** | **BLOCKING** | 0.7 B | 2.8 h |
| **S1** | mute `GuardedPickupBoostReward` (6 nominal → −0.0266, −1.31 % of density) | ship, after its pre-gate | 0.5 B | 2.0 h |
| **S3** | `GoalDirectedTouchAccel` A/B swap for `TouchAccel` (10 nominal, 0.0538 = 2.65 %) | ship | 0.5 B | 2.0 h |
| **S2** | `AerialTouchReward` airFrac AND-restore (120 nominal) | **BLOCKED** | 0.5 B | 2.0 h |
| **S4** | `ConcedeAccountabilityReward` (+0.0613, +3.1 %) | **BLOCKED** | 1.0 B | 4.0 h |
| | restarts (S2/S3/S4 rebuilds; S1 needs none under the actuator) | | — | 0.25 h |
| | **full programme** | | **3.2 B** | **≈ 13.0 h** |
| | **SHIPPABLE CORE = S0 + S1 + S3** | | **1.7 B** | **≈ 7.0 h** |

**S0 extended from 0.35 B to 0.7 B** because K = 80 serve groups halves the baseline-error
false-alarm inflation from 24.5× to 5.0× (§8.3), and because σ̂ must be measured on the in-process
statistic rather than inherited.

**Why S2 and S4 are BLOCKED (addendum C5).** Neither has a **written, farm-audited reward body**.
V2 carried a draft sketch for each (a 6-edit patch for S2, a full class for S4), and the addendum's
audit judged neither sufficient to ship: in particular the **~0.8 s `AerialTouchReward` refire
cooldown is the obvious cooldown-gaming surface and S2 edits exactly that term**, and S4 must ship
ZeroSum-wrapped or it cannot be latched without breaking antisymmetry. **Nothing ships until each
body is written out and farm-audited.** Do not count them as plan.

S4 carries a further **structural limitation**: in mirror self-play concede rate ≡ score rate by
symmetry, so "we concede less" is measurable **only against a fixed external opponent** — i.e. the
low-power Nexto instrument. That is why its outcome bar had to be raised to ≥ 20 % (§9.6).

**Programme-level criterion.** **At least ONE of the shipped stages must clear its OUTCOME clause
at the stated σ.** If all land inside their null bands, the programme's premise — that 1–3 %
credit-density surgery moves this run — is **falsified**, and the correct conclusion is to stop
shaping and reallocate the steps (§12.3).

### Explicitly NOT in the programme

`serveFrac` raise (§5 #14, negative EV); un-gating the five demo-gated potentials (0.005 % of
density); the uniform terminal potential refund and its Goal compensation (the algebra inverts);
deleting `WallJumpToBallReward` or `FlipResetReward` (zero upside, negative option value); the
TeamPressure 0.15 → 0.05 cut (1.36 % of density, criterion passes on a null);
`GoalsideEngagementPotential` (two independent kills); the global `gaeLambda` change; **any boost
term at all** (§7.3); **any threat / offensive-potential term** (§7.4); **goal-speed scaling**
(§7.4). All in the graveyard (§10) with reasons.

---

## 7. Term specifications — what survived

Line numbers re-verified against HEAD `45a59d5`.

### 7.1 S1 — mute `GuardedPickupBoostReward` (6 nominal → 0.0266 density, 1.31 %)

**Ships as `ship = 0 / safe = 1` on a term that stays instantiated** (§8.1) — *not* as a deleted
line. That is what makes it revertible in-process with no rebuild and no restart.

```cpp
// src/ExampleMain.cpp:174 — the line STAYS. Only its actScale changes.
//   { new ZeroSumReward(new GuardedPickupBoostReward(), TEAM_SPIRIT), 6.f },
```

**The mechanism** (verified at source, `CommonRewards.h:626-648`):

```cpp
// GuardedPickupBoostReward::GetReward, payout at :642-646
if (player.boost > player.prev->boost) {
    return sqrtf(player.boost / 100.f) - sqrtf(player.prev->boost / 100.f);
} else { return 0; }                       // <-- POSITIVE PART ONLY
```

against `CarEnergyPotentialReward::Phi` (`:191`), which carries the identical function
**two-sidedly**:

```cpp
float be = sqrtf(0.01f * p.boost) * KE_NORM;   // and :199-201 returns gamma*Phi(cur) - Phi(prev)
```

**The source comment at `CommonRewards.h:189-190` argues for the overlap, and its reasoning is
backwards. Quote it so nobody re-adds the term:**

> `// Pickup credit intentionally overlaps GuardedPickupBoost - this term adds`
> `// the SPEND side of the ledger, which a pickup reward cannot express.`

**Correction.** `CarEnergyPotentialReward` is **exact PBRS** (`:199-201`,
`return gamma * Phi(player) - Phi(*player.prev);`), so it *already* carries both sides: it pays
`+15·Δ√` on the pickup **and** charges `−15·Δ√` on the drain. A positive-part-only pickup reward on
the **same** `sqrt(boost/100)` function is therefore not a complement — it is a one-sided duplicate
whose total income is `6 ×` the total **positive variation** of `√(b/100)`, maximised by oscillating
boost as fast as pad cooldowns allow. The comment should be replaced with a warning, not a
rationale.

**Conviction, RESTATED and weakened (addendum C3).** V2's headline — *"a 5.7× / 1.13 units-per-second
boost farm"* — is a **PRE-ZeroSum figure and it cancels in mirror self-play.** The term is
registered wrapped (`src/ExampleMain.cpp:174`); post-transform 1v1 reward is `r_i − r_j`
(`ZeroSumReward.cpp:19-28`), so two copies of the same policy running the same pad circuit give
`E[r_i − r_j] = 0` **identically**, and 95.1 % of iterations are mirror self-play [M, 6M]. The
honest claim is:

> **S1 removes a positive-part-only duplication of a two-sided axis — a differential ratchet worth
> 0.0266 units/player-step (1.31 % of credit density).** Not free income.

Supporting arithmetic that survives [D, 6M]: marginal pickup income at boost 0/12/33/50/88 =
**2.079 / 0.861 / 0.578 / 0.482 / 0.372** — a **5.6× tilt toward running empty**; live
`Player/Boost` mean 15.78 with **median 0** [M, 6M].

**NEW PRE-GATE (evaluated on S0 data, before S1 ships).** `Diag/Boost At Pickup` (mean
`player.prev->boost` on steps where boost increases, §7.3) must read **< 45** over S0. **≥ 45 ⇒ the
low-tank sawtooth is not live, S1 is unconvicted, and it does not ship.**

**Why mute, not repair.** The honest repair is to make it two-sided — at which point it is
*literally* CarEnergy's `be` at a different weight. The boost axis stays covered, two-sidedly, at
w = 15. Muting removes the **broken duplicate**, not the axis.

**What is genuinely lost.** Today the opponent is *charged* for our pickup, so **pad denial is
paid** — the design comment's stated rationale (`:170-173`). Counted honestly: 1.31 % of credit
density, and the same denial is still priced two-sidedly by CarEnergy at w = 15 (denying a pad keeps
the opponent's Φ low, which the wrapper transfers to us). Net loss: [E] ~40 % of a 1.3 % axis.

**Demo guard — verified, nothing stranded.** `CarEnergyPotentialReward` carries its own at `:199`
(`player.isDemoed || player.isDemoed != player.prev->isDemoed`). It is **equivalent** to
`GuardedPickupBoost`'s `:639-640` form, *not* "strictly wider" as drafted — identical truth tables
over all four `(isDemoed, prev->isDemoed)` states [M]. **Requirement to record beside S1:** any
future boost term MUST use the two-condition form. A level-difference term checking only
`prev->isDemoed` still books the respawn jump — `BOOST_SPAWN_AMOUNT = 33.33` (`RLConst.h:52`) is
**2.15×** the live mean, and under a one-sided pickup reward would pay **+0.577** per demo,
**15.9× a small pad** [D].

**FREE MEASUREMENT that settles C3.** Because the term stays instantiated at `actScale = 0`, its
`Rewards/GuardedPickupBoost` panel keeps reporting the **unscaled pre-ZeroSum child** (§3.4) — a
**live counterfactual on a muted arm**. If the ratchet was being farmed, it falls ≥ 20 % relative as
the sawtooth decays; if it stays flat, the ratchet was never live and S1's conviction was thin.
Record either way; not a ship gate.

### 7.2 S3 — `GoalDirectedTouchAccelReward` (10 nominal, 0.0538 density, 2.65 %)

**This is NOT v1's design.** v1's form (`push = dv·n̂ > 0` gate, credit `push·(n̂·ĝ)`, absolute-speed
clamp removed) was killed on six independent grounds (§10). The design below **keeps the
incumbent's exact magnitude** — which is what made it safe — and multiplies by a **non-negative**
direction factor.

```cpp
// CommonRewards.h — INSERT immediately after TouchAccelReward's closing brace at :357.
// KEEP TouchAccelReward in the file: S3 ships as an A/B PAIR (§8.1), both instantiated.
//
// r = (curSpeedFrac - prevSpeedFrac)^+  *  max(0, cos(v_ball_new, ball -> their goal))
class GoalDirectedTouchAccelReward : public Reward {
public:
    constexpr static float MAX_REWARDED_BALL_SPEED = RLGC::Math::KPHToVel(110); // 3055.6

    virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
        if (!state.prev || !player.ballTouchedStep) return 0;
        float prevSpeedFrac = RS_MIN(1.f, state.prev->ball.vel.Length() / MAX_REWARDED_BALL_SPEED);
        float curSpeedFrac  = RS_MIN(1.f, state.ball.vel.Length()       / MAX_REWARDED_BALL_SPEED);
        if (curSpeedFrac <= prevSpeedFrac) return 0;

        float speed = state.ball.vel.Length();
        if (speed < 1e-3f) return 0;
        Vec vhat = state.ball.vel * (1.f / speed);

        Vec oppGoal = (player.team == Team::BLUE)
            ? CommonValues::ORANGE_GOAL_BACK : CommonValues::BLUE_GOAL_BACK;
        Vec toGoal = oppGoal - state.ball.pos;
        float gl = toGoal.Length();
        if (gl < 1e-3f) return 0;
        Vec ghat = toGoal * (1.f / gl);

        float align = RS_MAX(0.f, vhat.Dot(ghat));   // 1 = at their net, 0 = lateral or worse
        return (curSpeedFrac - prevSpeedFrac) * align;
    }
    virtual std::string GetName() override { return "TouchAccelReward"; }
};
```

**Every safety property of the incumbent survives, because the magnitude is byte-identical.** This
is the entire design argument — each of these is also a closed farm surface:

| property | why it holds | what breaks without it |
|---|---|---|
| **BOUNDED INCOME** | double clamp: total pay for accelerating a ball from rest ≤ 1.0, and above 3055.6 uu/s the term pays **0 forever** | v1's clamp-free design became a **~50 reward/s annuity** against a 31.6 reward/s whole-stack density |
| **SURFACE-REBOUND IMMUNE** | a wall/backboard/floor rebound is inelastic against a static surface ⇒ **reduces** `\|v_ball\|` ⇒ positive part pays 0 | a Δv-projection form needs an energy gate, and the gate proposed was dead code (`MIN_SEP = 60 uu` < `BALL_RADIUS = 92.75`) |
| **NO WHIFF TAX** | `r ≥ 0` always ⇒ `E[attempt]` weakly higher than not attempting, every geometry | v1's signed form **inverted** `E[attempt]` on the chase-down clear — the structure `STEERED_PRACTICE.md` convicts |
| **NO CARRY ANNUITY** | max ground Δv/step = `(400 + 991.67) × 0.06667 = 92.78 uu/s` [D] = 0.030 of the clamp ⇒ 33 maximal steps for one unit, after which the term is dead | — |
| **NO GRAVITY LEAK** | reads speeds, not impulses ⇒ the −43.3 uu/s/step gravitational Δv never enters | — |

**What changes:** an own-goal-ward power touch pays **0** instead of full credit; a lateral clear
pays ~0. The term never goes negative — "unpriced", not "punished". It deliberately reads **no**
`PlayerEventState` field: `GameEventTracker`'s shot attribution is the source-verified phantom farm
that keeps `ShotReward` banned. Opponent branch: **new vs the incumbent**, an opponent bashing the
ball toward *their own* net no longer transfers us a penalty — a small, correct-direction change.
Demo exposure: none.

**Weight, set by measurement not by ratio.** `w = 10 / E[max(0, cos)]`, where `E[max(0,cos)]` is
`Guard/Touch Goal Alignment` measured in S0, **clamped to [10, 25]** ([E] lands 18–22) — making the
swap **density-neutral by construction** and cleanly attributable. V2 asserted net-zero weight
without deriving it.

**Panel continuity.** `Reward::GetName()` (`Reward.h:32-56`) returns the Itanium-mangled typeid (the
trim keys `"::"` and `" "` do not match a mangled name), which is why every panel reads
`N4RLGC17TouchAccelRewardE`. **Pin the panel key to the mangled incumbent string** so `Rewards/*` is
continuous across the swap and the incumbent's own arm is a valid control.

**Fields, all verified** [M]: `player.ballTouchedStep` (`Gamestates/Player.h:25` — **not `:26`,
which is `ballTouchedTick`**; computed at `Player.cpp:18-20`); `player.team` (`Player.h:21`);
`state.prev` (`GameState.h:22`, NULL on the first step after every reset);
`CommonValues::ORANGE_GOAL_BACK`/`BLUE_GOAL_BACK` (`CommonValues.h:29-30`, both at x = 0);
`RLGC::Math::KPHToVel` is `constexpr` (`RLGymCPP/Math.h:12`); `RS_TEAM_FROM_Y`
(`RocketSimV3/compat/RocketSim.h:43`) confirms BLUE attacks +y.

**Known limitation, stated rather than hidden:** on airborne touches `AerialTouchReward` (max raw
1.0 at w = 120, post-ZS swing 240) outweighs this term (max raw 1.0 at w ≈ 20, swing 40) **6:1** and
reads no goal direction, so an own-goal-ward *aerial* still nets ≈ +200 post-ZS. If directional
pricing of aerials is wanted, the factor belongs **inside** `AerialTouchReward` — a separate, later
lever, explicitly not bundled here.

### 7.3 Boost — SHIP THE PANELS, SHIP NO TERM

**Verdict: B0 ships (read-only). B1 defer. B2 armed, not shipped. B3 KILLED. B4 verified.**

**Why no term.** `CarEnergyPotentialReward`'s `be` is **37.3 % of E[Φ]**, **30.0 % of per-step
|ΔΦ|**, and carries **0.2340 units/player-step** post-ZeroSum at w = 15 [M, 6M] — the **4th-largest
single signal in the stack, 8.8× GuardedPickupBoost's 0.0266**, and nearly orthogonal to the
kinematic channel (`corr(Δbe, Δke+Δpe) = −0.054`). **S1 removes 10.2 % of the axis's shaping mass,
not the axis.** It removes 100 % of its *optimum-relocating* mass, but that mass pointed at a pickup
argmax **at empty**. Post-S1 the axis retains **89.8 %** of its density and 0 relocating mass.
**Ceiling if a term is ever admitted: ≤ 0.03 density (≈ 1.5 %)** — no larger than what S1 frees.

**External yardstick** [M, 6M]: from 12,412 live iterations (ts ≥ 0.9 B), served `Player/Boost` =
17.677 ± 0.527 (n = 456) vs unserved 15.269 ⇒ **implied Nexto tank ≈ 20.1**, falling 21.1 → 18.9
across the run while ours rose 14.83 → 15.68 — **the gap is closing, 6.2 → 3.2**. **Re-open trigger:**
B0 convicts, *or* the implied gap re-widens to ≥ 8 points over any 0.6 B block.

**B0 — the six read-only panels (0 nominal, 0.0000 density).** No live panel can convict or refute
the boost thesis: `Player/Boost` is a mean hiding a **median of 0**. Insert inside the existing
`doExpensiveMetrics` per-player loop (`src/ExampleMain.cpp:485-505`):

```cpp
report.AddAvg("Diag/Boost Empty Frac",   player.boost < 1.f);
report.AddAvg("Diag/Boost Under12 Frac", player.boost < 12.f);   // BOOST_AMOUNT_SMALL, RLConst.h:208
if (player.prev && player.boost > player.prev->boost)
    report.AddAvg("Diag/Boost At Pickup", player.prev->boost);   // the sawtooth test — S1's PRE-GATE
if (state.ball.pos.z > 300.f && (state.ball.pos - player.pos).Length() < 2000.f)
    report.AddAvg("Diag/Boost At Aerial Opp", player.boost);     // the census slice
if (player.ballTouchedStep && !player.isOnGround && state.ball.pos.z > 400.f)
    report.AddAvg("Diag/Boost At Aerial Touch", player.boost);
```

and — **braced, because the existing `if (state.goalScored)` at `:507-508` has no braces**:

```cpp
if (state.goalScored) {
    report.AddAvg("Game/Goal Speed", state.ball.vel.Length());   // existing line :508
    Team conceder = RS_TEAM_FROM_Y(state.ball.pos.y);            // compat/RocketSim.h:43
    for (auto& p : state.players)
        report.AddAvg(p.team == conceder ? "Diag/Boost At Concede" : "Diag/Boost At Score", p.boost);
}
```

Scope verified [M]: `Team` and `RS_TEAM_FROM_Y` reach `ExampleMain.cpp` transitively via
`CommonRewards.h`; `Team::BLUE` is already used at `:372`; `player.prev` is one level only
(`Player.cpp:11-12` nulls the grandparent); `report.AddAvg(name, bool)` has precedent at `:487`. The
goal branch sits **outside** the 1-in-4 `doExpensiveMetrics` gate (`:481`), so it sees every goal.

**CRITERION — the registered prediction is the REFUTING branch.** Over 200 M steps the boost thesis
is CONVICTED only if **both** (a) `Diag/Boost At Score − Diag/Boost At Concede ≥ 3.0` **and**
(b) `Diag/Boost At Aerial Opp ≤ Player/Boost − 3.0`. **Offline predicts the opposite on (b)**:
opportunity-slice boost is **15.46 vs 13.98 overall, i.e. +1.5 ABOVE baseline** [M, 6M]. **A null
closes the boost line for the run.** Bias directions are conservative: Nexto occupies a whole team
on served iterations (biasing the score/concede difference by < 0.5 pt) and a respawned car sits at
33.33, inflating the **conceding** side.

**B2 — ARMED, not shipped: the `(c, p)` shape knob inside `be`** (15 nominal, unchanged). If B0
convicts, the doctrine-correct response is a shape change **inside the potential that already owns
the axis**: `be = c·powf(0.01·boost, p)·KE_NORM`, with `(c, p) = (1, 0.5)` bit-identical today.
Exact PBRS preserved at every `(c,p)` ⇒ unfarmable; diff is two constants. Break-even speed for
spending fuel is `v* = 889·c·p·(0.01b)^(p−1)` ⇒ 445 uu/s at a full tank, **1119 at the live mean
15.8**, 1988 at b = 5; 43.3 % of offline steps sit below it [D, 6M]. Worked takeoff tax: 0.6 s climb,
20 boost from empty-ish, v 600→1100 ⇒ +3.06 kinematic credit − 6.71 fuel = **−3.65**, against an
`AerialTouch` payout of ≈ 48 — a **13× margin**, so the current shape does **not** deter aerials.
Target band: keep `be` within **0.15–0.30** density. **Do not touch it while the aerial scaffold is
at formative weights** (AerialTouch 120, AirIntercept 40, ConsecAir 30) or while PHASE_B is pending.
Criterion if it ever fires: over 0.5 B, `Diag/Boost Empty Frac` falls ≥ 5 pp absolute **and**
`Diag/Boost At Aerial Opp` rises ≥ 2.0 points. **< 5 pp ⇒ the knob is refuted**, not "a partial win".

**B3 — KILLED: the Nexto-style height-tapered boost-spend penalty.** Full reasons in §10; the
headline is that **the taper reinforces the very pathology it targets** — it zeroes the penalty at
height, so the cheapest way to satisfy it is *get high first, spend there* = **takeoff on an empty
tank**.

### 7.4 Threat / offensive potentials and goal speed — NOTHING SHIPS but a two-line tripwire

**All four killed.** V2 deferred two of these with explicit re-open conditions. **Both conditions
were run. Both returned the opposite sign.** Deferrals converted to kills.

**T-GV — solid-angle `GoalView` replacing B2G's Φ: KILL.** V2's re-open condition was *"an offline
replay showing B2G's Φ mis-ranks scoring probability at fixed |ball − goal|"*. Ran it on **47,774
goal-terminated rows** [M, 6M]:

| | AUC(→ own team scores) | stratified within the other's deciles |
|---|---:|---:|
| **Φ_B2G (incumbent)** | **0.6009** | **0.5452** within Φ_SA deciles |
| Φ_SA (challenger) | 0.5972 | 0.5198 within Φ_B2G deciles |

`corr = 0.9327`. **The incumbent carries strictly more independent goal-outcome information.** The
challenger is also **5.19× hotter per step** (E|γΦ′−Φ| = 0.012943 vs 0.002495) and **9.3× fatter in
the tail** (p99.9 0.1772 vs 0.0190) — more gradient volatility for less information. At weight 75 it
would be **48 % of the entire stack**; density-neutral is **14.5**. **Re-open bar (pre-registered,
currently FAILED):** ≥ 50,000 goal-terminated rows from the live lineage with **both**
AUC(Φ_SA) ≥ AUC(Φ_B2G) + 0.02 **and** stratified AUC ≥ 0.56. Measured **−0.0037** and **0.5198**.

Two sub-findings worth keeping: (i) the v1 lateral-exp kill is **correct and strengthened** —
`exp(−|b_x|/1200)` is a common multiplicative gain on both exp terms because both goal backs are at
x = 0 (`CommonValues.h:29-30`), so `Φ_v1 ≡ L(|b_x|)·Φ_B2G` exactly, zero new axis; (ii) the
solid-angle form genuinely does **not** factor out (R² = 0.951, within-bin sd 0.129 vs total 0.587)
— it is simply not *useful*.

**T-OP — Lucy-SKG Offensive Potential (KRC of align × distance × closing velocity): KILL.** Lucy's
highest-weighted shaping term measures **worse than Pulsar's 4-weight `BallProximityPotential`** on
Pulsar's own data. On 82,528 rows labelled by the possession race (the style-robust metric the
steering programme converged on) [M, 6M]:

| | AUC(possession race) | stratified within BallProximity deciles |
|---|---:|---:|
| **Φ_BallProximity (incumbent, w = 4)** | **0.5260** | — |
| Φ_op (challenger) | 0.5107 | **0.5026 ≈ nothing** |

A geometric mean is bounded by its weakest factor, and here the strongest factor already *equals*
the incumbent: φ_dp2b = 0.5260 (it **is** a distance term), φ_up2b = 0.5164, **φ_ab2g = 0.4955 —
BELOW CHANCE**. Component definitions are **[U]** (the paper never gives them; reconstructed from
the Necto commit Lucy cites), but the conclusion is robust to that. Lucy's gains were measured
against Necto, whose stack has none of Pulsar's BallProximity / TouchAccel / AirIntercept /
KickoffRace. **Re-open bar (currently FAILED):** stratified AUC ≥ 0.55 on ≥ 50,000 live-lineage rows
(measured **0.5026**); secondary hard gate even on a pass: p99.9 |γΦ′−Φ| < 0.10 (measured
**0.7994**, a 42× fatter tail than B2G, from the KRC's `sgn()` flips). Full kill reasoning: §10.2.

**GS-1 (goal-speed-scaled objective) and SP-1 (raising `MAX_REWARDED_BALL_SPEED`): both KILLED.**
Reasoning in full at §10.2. The three load-bearing numbers:

- `Game/Goal Speed` (`src/ExampleMain.cpp:507-508`) is **1742 uu/s, flat since ~2.0 B**, while the
  mature 5.0v3 lineage (**~700 Elo stronger**, same TouchAccel 10 / Goal 150 core) reads **~1760**:
  **+53 B steps and +700 Elo buy +1 % goal speed** [M, 6M]. Not on the improvement path.
- Speed is **already priced twice** — at the touch by TouchAccel, and at the finish by discounting
  (`γ¹⁵ = 0.95450` ⇒ a goal 1 s sooner is worth **7.0 units/s**, farm-free).
- The **stall farm is exact**: break-even `t* = w·(1−0.290)/7.0` ⇒ **0.51/1.01/1.52 s** at w = 5/10/15
  separate, **7.6 s** multiplicative at k = 0.5, **15.2 s** fully multiplicative [D] — with **no
  terminal to stop it**.

**Re-open test for GS-1 (on the existing series, can fail):** falsified if `Game/Goal Speed` 0.25 B
block means fall **below 1550 uu/s for 3 consecutive blocks** while burst-standardised Nexto share
is flat or down. Confirmed by the null: blocks stay in [1650, 1850].

**SP-1's exceedance rate is UNMEASURED [U]** — which is the whole point of TEL-1.

**TEL-1 — SHIP (0 nominal, 0.0000 density): the two lines that make SP-1 falsifiable.**

```cpp
// inside the existing doExpensiveMetrics per-player loop, after :498
if (player.ballTouchedStep && state.prev) {
    constexpr float C = TouchAccelReward::MAX_REWARDED_BALL_SPEED;  // 3055.6, CommonRewards.h:337
    float pre = state.prev->ball.vel.Length(), post = state.ball.vel.Length();
    if (post > pre && pre > 0.f)     // pre > 0 excludes the post-reset step: EnvSet.cpp:331 MakeEmpty()
        report.AddAvg("Game/Touch Unpriced Frac", RS_MAX(0.f, post - RS_MAX(pre, C)) / (post - pre));
}
// INSIDE the goalScored branch, WHICH MUST BE BRACED FIRST (:507-508 is an unbraced single-statement if)
report.AddAvg("Game/Goal Speed Over Cap",
    state.ball.vel.Length() > TouchAccelReward::MAX_REWARDED_BALL_SPEED);
```

`MAX_REWARDED_BALL_SPEED` is `public constexpr static` (`CommonRewards.h:337`) and the header is
included at `ExampleMain.cpp:7` [M].

**CRITERION (both directions live).** Over the first 0.25 B after deploy: SP-1's kill is
**CONFIRMED** if `Game/Touch Unpriced Frac` < 0.05 **and** `Game/Goal Speed Over Cap` < 0.10. It is
**FALSIFIED** if either exceeds its threshold, reopening the change as the **paired, rate-preserving**
edit `C → 6000` **with** `w 10 → 19.6`, as its own 0.5 B CUSUM-gated stage. **TEL-1 itself fails** if
`Touch Unpriced Frac` reads exactly 0.0000 for all blocks while `Goal Speed Over Cap` is non-zero
(mutually inconsistent ⇒ the touch counter is not wired to the strike step), or if either series is
absent; then revert the two lines and treat SP-1 as **unresolved**, not killed.

### 7.5 S3's convicting panels (S0, read-only)

Inserted immediately after `src/ExampleMain.cpp:503`, **inside** the per-player loop that opens at
`:486` and closes at `:504`. An earlier draft placed one at `:507` — that is outside the loop,
`player` is out of scope, and it does not compile.

```cpp
if (state.prev && player.ballTouchedStep) {
    float sp = state.ball.vel.Length(), spPrev = state.prev->ball.vel.Length();
    if (sp > spPrev && sp > 1e-3f) {
        Vec oppGoal = (player.team == Team::BLUE)
            ? CommonValues::ORANGE_GOAL_BACK : CommonValues::BLUE_GOAL_BACK;
        Vec toGoal = oppGoal - state.ball.pos;
        float gl = toGoal.Length();
        if (gl > 1e-3f) {
            float c = (state.ball.vel * (1.f / sp)).Dot(toGoal * (1.f / gl));
            report.AddAvg("Guard/Touch Goal Alignment", RS_MAX(0.f, c));   // S3's WEIGHT instrument
            report.AddAvg("Guard/OwnwardTouch Frac", c < -0.3f);           // S3's CONVICTING measurement
        }
    }
}
// wall-play census: reuses the reward's own test verbatim (CommonRewards.h:817-820) so panel and
// reward agree by construction. Measures the PRECURSOR state the reward cannot shape.
report.AddAvg("Player/Wall Drive Frac", RLGC::WallJumpToBallReward::OnWall(player));
```

**`Guard/OwnwardTouch Frac` is the convicting measurement for S3, not its manipulation check** — v1
had this inverted. **S3's pre-gate:** it must read **≥ 0.02** over S0. Below 0.02 the mispricing is
< 2 % of touches, S3 is unconvicted, **and it does not ship.**

---

## 8. The automatic guard — full mechanism spec

**Nothing reward-side ships without this.** Doctrine requirement (3) is an *actuator*, not a
dashboard. Today the reward stack has none: `EnvSet.cpp:229` reads `weightedReward.weight` directly
and `WeightedReward` (`Reward.h:62-75`) has no scale member — verified at source, the struct carries
only `reward`, `weight`, `gated`, `zeroSumPtr` [M].

### 8.0 Why none of the three existing candidates works

| candidate | why it fails |
|---|---|
| `steerRatingTripped` | (a) **Structurally blind** — needs > 627 Elo/B (peak arm) or > 336 Elo/B (EMA arm) to trip inside 0.5 B (D8). (b) **Blast radius of six subsystems**: it gates the Headroom injection (`Learner.cpp:4759`), RND (`:4900`), the Ladder drive (`:5227`), `FrontierPool::Fill` (`:2398`), the league anchor slice (`:3446`), opponent styles (`:3488`) **and Nexto serving (`:3428-3429`)** — so tripping it **stops measuring the yardstick at exactly the moment you need it** and makes the post-trip run structurally different from the baseline, destroying attribution even after a revert [M]. |
| the reachability gated channel (`WeightedReward::gated` → `gatedPos`) | **Positive-part only by design** (`EnvSet.cpp:232-235`: *"a low gate must not mute penalties"*). For a ZeroSum-wrapped one-sided term, latching would mute the scorer's `+x` and leave the conceder's `−x` — **breaking the zero-sum invariant exactly when the guard fires** [M]. Also `gateEnabled = false` (`src/ExampleMain.cpp:633`), so the channel is live-inert. |
| `iterationCallback` (`Learner.cpp:5510`) as the write site | **Data race.** No `join()` between `Learner.cpp:3972` (re-kick) and `:5511`, so in pipelined mode 1024 arena threads are reading `weightedReward.weight` while it is written [M]. This is where an implementer would naturally put it (it is the documented per-iteration hook and where the PHASE-B trigger lives) — **and it is behind two early returns and only runs on eval iterations (~1 in 27)**, so the scales would frequently never be applied at all. |

### 8.1 Actuator — `WeightedReward::actScale`, keyed by INDEX

**The competing `StageScaled` wrapper design is DROPPED.** Placed *outside* `ZeroSumReward` it makes
`EnvSet.cpp:111`'s cached `dynamic_cast<ZeroSumReward*>(weighted.reward)` return **null**, so the
logging path at `EnvSet.cpp:259-260` falls through to `rewardToSave = output[playerSampleIndex]` —
the **post-ZeroSum, post-scale** value. **All 16 `Rewards/*` panels silently change meaning**; at
scale 1.0 the "bit-identical" no-op wrapper makes `Rewards/GuardedPickupBoost` jump from ~+0.0022 to
a mean-≈0 series, **failing its own S0 identity test**. This is a rediscovery of V2's own kill
*"Wrapping a scheduler/latch outside `ZeroSumReward`"*. It also leaked `child` (no destructor) and
had no way to key an A/B pair of one class. **Do not ship a wrapper.**

The correct actuator is a scale member on `WeightedReward`, which sits *outside* the cast entirely:

```cpp
// [A] RLGymCPP/src/RLGymCPP/Rewards/Reward.h — struct WeightedReward (:62-75)
    float actScale = 1.f;   // reward-guard actuator. barrier-written, collect-read.

// [B] EnvSet.cpp:229 — the ONLY hot-path change (one multiply, no branch).
//     Downstream of zeroSumPtr resolution (:111), so Rewards/<name> keeps reporting the RAW
//     pre-ZeroSum child, UNSCALED — which is what makes the muted arm a live counterfactual.
-   float weighted = output[i] * weightedReward.weight;
+   float weighted = output[i] * weightedReward.weight * weightedReward.actScale;
//     The gated bucket at :234 correctly inherits the scale via `weighted`.
```

**Keyed by index, not name.** `RewardWrapper::GetName()` forwards the child (`RewardWrapper.h:32-33`)
and `Reward::GetName()` is `typeid`-derived, so **two instances of one class are indistinguishable** —
and the shipping convention below *requires* exactly that (S3 ships as two `TouchAccel`-family
instances with complementary scales). A name-keyed whitelist would set both arms to the same scale:
**a silent no-op.** `rewardAct` is therefore
`std::vector<RewardActEntry{ int idx; float ship, safe; }>` indexing into the per-arena
`std::vector<WeightedReward>` built by `BuildRewards` (`src/ExampleMain.cpp:142`), which is
constructed identically for every arena (`EnvSet.cpp:96` copies from `EnvCreateResult::rewards`).
**Index stability is asserted at boot** against `reward->GetName()`.

**Shipping convention — this is what makes deletions revertible. NEVER delete or edit a reward
line.**

- a **deletion** ships as `ship = 0 / safe = 1` on a term that stays instantiated;
- a **structural swap** ships as an **A/B pair** with complementary `(0,1)/(1,0)` scales.

Then one float per `WeightedReward` reverts every stage **in-process, no rebuild, no restart**. V2
§7.1's admitted hole — *"cannot restore a deleted term (S1) or undo a structural change (S2/S3)"* —
**closes**. That is precisely what makes S1 and S3 shippable.

Pairing is safe only for terms whose `GetAllRewards` is a pure function of `GameState`. Per-episode
consumable state is held **per instance** (`KickoffRaceReward`'s fire-once flag,
`CommonRewards.h:494`), so an A/B pair is independent; `Demo`/`Goal` are event *queries* on
`GameEventTracker`, not consuming reads, so a shadow instance neither steals nor duplicates events.
**Each stage must source-check its own pair before shipping.**

**Zero-sum safety.** `actScale` multiplies the term's output identically for every player in the
arena (`EnvSet.cpp:228-230` loops all players with the same `weightedReward`), so a ZeroSum-wrapped
term stays exactly zero-sum (`c·Σ = 0`) and an antisymmetric term stays antisymmetric. Both signs
scale together — unlike `gatedPos`.

**PBRS.** `c·(γΦ(s′) − Φ(s))` is exact PBRS with potential `c·Φ`. A mid-episode scale change breaks
telescoping for exactly one step. State the magnitude correctly: for a *wrapped* potential the break
is `c₂γ(Φᵢ′ − Φⱼ′) − c₁(Φᵢ − Φⱼ)` — of order the **cross-player potential level difference**
(≈ 15 × 0.3 ≈ 4.5 units for CarEnergy, ≈ 8× a typical 0.29 step reward), **not** the per-step delta.
Bounded, ZeroSum-damped, one-off, and strictly preferable to waiting for episode boundaries. **Log
it.**

### 8.2 Write site — `Learner.cpp:3944`, plus a MANDATORY boot apply

Barrier ordering verified at source: `collectThread.join()` **:3928** → report merge **:3941-3943**
→ `float collectionTime = collectWallTime;` **:3944** → `if (pipelineOn) {` **:3949** →
`fnRatingGuard(report)` **:3955** → `fnSyncSnapshot()` **:3970** → re-kick `std::jthread` **:3972**
[M].

**Line 3944 is inside the barrier, before every shared-state consumer, and executes in both
pipelined and sequential mode** — which is why it beats `:3955`.

**A boot apply immediately after `envSet` is constructed is also mandatory.** The first iteration
takes the `else` branch at **`Learner.cpp:3932-3936`** and calls `fnCollectIteration()` **inline**,
before `:3944` ever runs — so **~614 k timesteps per restart would otherwise collect at un-actuated
weights** [M].

Consequence to accept and document: `steerRatingTripped` is read **one iteration stale** at `:3944`
(the rating guard runs at `:3955`). For a one-way latch a ~3 s lag is immaterial.

### 8.3 Trip statistic — serve-indexed CUSUM on Nexto logit share, σ̂ and h CALIBRATED LIVE

The yardstick's atomic sample is the **serve group** (one iteration where the whole non-self fleet is
Nexto), not a step window. `Nexto/Serve Iters` increments at collection start (`Learner.cpp:3434`)
while goals land at `:3751/:3753`; the panels are written during the **learn** phase at
`:4953-4955`, i.e. *after* the next worker was kicked at `:3972` **and while it is concurrently
incrementing the same atomics**. Every wandb row is a random mid-collection snapshot straddling two
generations — that, not step-window boundaries, is the source of the measured 8–13× overdispersion
[M].

**Therefore the σ that sized `h = 8` is invalid for the shipped statistic.** The in-process
statistic reads *deltas of the atomic counters after the join at `:3944`*, so serve increments and
their goals are in the same iteration **by construction**, and its σ is smaller and unknown.

```
per serve group g (closed when a served iteration is followed by an unserved one):
    F_g = ΔNexto/Goals For,  A_g = ΔNexto/Goals Against,  N_g = F_g + A_g   (drop N_g == 0)
    y_g = logit(F_g / N_g)
baseline  μ₀ = mean(y) over the K serve groups immediately preceding the stage edit, FROZEN
CUSUM     S₀ = 0 ;  S_g = min(0, S_{g-1} + (y_g − μ₀)/σ̂ + k),  k = 0.5
TRIP      S_g < −h        (one-way, never auto-re-enables)
```

**`σ̂` and `h` are DERIVED LIVE over S0 from the in-process series, NOT hard-coded:**

- **σ must be re-measured** (above). The wandb-reconstructed **σ = 0.5252 logit / 12.38 pp** and its
  `[0.40, 0.66]` sanity band are **artifacts of the smeared series**; a hard-coded band would likely
  sit permanently outside its own alarm.
- **Baseline estimation error inflates false alarms convexly.** Wald FA multiplier is
  `exp(2h²σ_δ²)` with `σ_δ = σ/√K`. At K = 40, `σ_δ = 0.158σ` ⇒ **24.5×** — a drafted 0.88 %/stage
  becomes ~22 %. At K = 80, 5.0×. At K = 160, 2.2× [D]. **This is why S0 is 0.7 B, not 0.35 B.**
- **The drafted dual-rolling rule (R56 ≤ b0 − 0.05 ‖ R14 ≤ b0 − 0.10) understates false alarms
  ~25×.** It omits the frozen baseline's own variance *and* multiplicity. Correctly,
  `sd(r56 − b0) = 0.02364` ⇒ **0.05 is 2.12σ, not 3.00σ**; `sd(r14 − b0) = 0.03738` ⇒ 0.10 is 2.68σ;
  with ~1.5 and ~4 quasi-independent looks per stage that is **≈ 4 %/stage, ≈ 15 % programme-wide**
  against a claimed 0.13 % [D]. **A CUSUM is sequential by construction and is the right
  instrument.**

**Calibration procedure (S0, blocking):** run the false-positive replay of the *shipped* rule over
the S0 in-process series **with μ₀ re-estimated from a finite K-window inside each trial** (the error
the drafted bootstrap omitted), and pick the smallest `h` giving ≤ 1 % FA per 57-serve stage.
Expected `h ∈ [10, 14]`. Planning values at `h = 12`, K = 80:

| true shift | Δ in σ | median detection | wall clock |
|---|---:|---:|---:|
| −10 pp | 0.81 σ | 39 serves = 0.34 B | **82 min** |
| −15 pp | 1.22 σ | 17 serves = 0.15 B | 36 min |
| −20 pp | 1.62 σ | 11 serves = 0.10 B | **23 min** |
| −25 pp | 2.01 σ | 8 serves = 0.070 B | 17 min |

(`dp/dlogit = p(1−p) = 0.2357` at p = 0.3803; delay ≈ `h/(δ − k)`.) All [D, 6M] — **re-derive on
`resid`.**

**Serve-starvation watchdog.** Serve groups arrive every ~8.79 M steps ≈ 44 iterations at
`tsPerItr = 200'000`. If no group closes within **400 iterations** (9× the mean gap, geometric
P ≈ 1.1e-4 per opportunity, ≈ 0.036 expected false fires programme-wide [D]) **the sensor is dead:
abort the programme and restore every scale to safe.** This closes the otherwise-real hole where a
regression hides behind a frozen statistic.

### 8.4 Durability — marker file + forced save

`rewardGuardTripped` in the stats JSON alone is **not durable**. `tsPerSave = 25'000'000`
(`src/ExampleMain.cpp:790`) ≈ 6 min at 70 k SPS, and the corrupt-checkpoint fallback walks
newest→oldest across up to `checkpointsToKeep = 8` × 25 M = **200 M steps**, which **can silently
un-revert a stage a guard already convicted**. Therefore, on trip:

1. write `<checkpointFolder>/REWARD_GUARD_TRIPPED` (first line: timestep, statistic, reason),
2. force an immediate `SaveStats`,
3. `RG_LOG` fail-loud in the `steerRatingTripped` style,
4. apply `safe` scales at the very next barrier (≤ 1 iteration).

Boot reads the **marker first, JSON second**. Persisted state: `reward_guard_tripped`,
`reward_guard_baseline` (μ₀), `reward_guard_sigma` (σ̂), `reward_guard_h`, the serve-group `y`
history, and the in-flight accumulator. **NaN-guard on save and `j.contains()` on load** — the
codebase idiom is explicit at `Learner.cpp:454-458` / `:550-565`; nlohmann serializes NaN as `null`
and `(float)j[...]` on `null` **throws**, so a naive port breaks `LoadStats` on the first boot.

### 8.5 The rating latch — leave it alone; a trip is a programme ABORT

- **Blind, by a wide margin** (D8): > 627 Elo/B on the peak arm, > 336 Elo/B on the EMA arm, to trip
  inside 0.5 B [D].
- **Tightening it would destroy the yardstick.** `Learner.cpp:3428-3429` gates the Nexto serve roll
  on `!steerRatingTripped`. A latch trip stops serving, so the outcome measurement **ceases to
  exist**.

So: **do not touch `ratingPeakTrip = 200` (`:992`) / `ratingDrawdownTrip = 150` (`:995`) during the
programme**, give the CUSUM the fast arm, and treat a latch trip as a hard **programme abort** —
which also removes the perverse path where a regression hides behind a deliberately-killed sensor.

**Free consolation prize, post-programme.** Simulating the *tight* 110/75 bands over the full 3 B
history yields **zero trips** — max observed drawdown from the running peak is **71.6** [D, 6M].
CLAUDE.md's standing instruction to "walk both back down toward 110/75 as the Elo trajectory
flattens" was **satisfiable at no measured false-positive cost** on 6M. **On `resid` at Rating ≈ 21
this must be re-derived from scratch** — a young run's volatility is exactly what forced 110/75 →
200/150 in the first place.

### 8.6 Guard hygiene — what prior incidents demand

- **One-way, never duty-cycled.** The causal gate (`LearnerConfig.h:253-255`) toggled **83 full duty
  cycles** on a healthy run (one toggle per ~40 M steps) [M, 6M]. Tolerable for a behaviour policy,
  **catastrophic for a reward weight**: each toggle changes `V^π` itself and simultaneously
  invalidates the critic, the expectile twin `V_exp`, the goal critic (γ = 0.9994) and the headroom
  V-dagger head.
- **Thresholds must not sit at the healthy operating point.** Both prior false-trip incidents were
  exactly that error (a 0.0-threshold possession gate; `ratingPeakTrip` 110 → 200 after three
  volatility trips). Every threshold here is derived from a measured population sd and stated with
  its σ.

### 8.7 PHASE_B hold — the exact minimal edit (keep on file, not urgent now)

Not needed today (Rating ≈ 21 vs a 1200 trigger, §0) — **but if a programme is still running when
the resid run approaches 1200, this is the edit.** Firing mid-programme destroys the experiment
three ways: 33 % 2v2 + 33 % 3v3 changes what a Nexto serve *is* (every frozen μ₀ and the CUSUM state
become meaningless); `TEAM_SPIRIT` 0.3 → 0.6 (`:1063`) rewrites credit in every ZeroSum-wrapped term
under test; and `exit(99)` + relaunch resets in-process guard state while `ratingKey` is rebuilt from
arena 0's mode (`Learner.cpp:1400`), so `Rating/1v1` may stop appearing.

```cpp
// src/ExampleMain.cpp:1257-1258, inside learner->iterationCallback. VERBATIM before:
//     if (g_PhaseB || !report.Has("Rating/1v1"))
//         return;
// after — PHASE_B stays armed in code; deleting the marker re-arms it with no rebuild:
    if (g_PhaseB
        || std::filesystem::exists(learner->config.checkpointFolder / "REWARD_PROGRAMME_ACTIVE")
        || !report.Has("Rating/1v1")) {
        if (!g_PhaseB) g_PhaseBStreak = 0;   // never carry a stale streak across the hold
        return;
    }
```

Compiles as written [M]: the lambda captures `[]` and takes `Learner* learner` (`:1254`),
`<filesystem>` is included (`:4`), `checkpointFolder` already uses `operator/` at `:1264`, and
`g_PhaseBStreak` is a file-scope global that would otherwise carry across the hold. Fail-safe
direction: forgetting to delete the marker leaves the run in PHASE A (a calibrated regime). Log
`Guard/PhaseB Held For` and RG_LOG loudly every 0.5 B past the marker's intended end timestep.

**Note the ordering hazard this design avoids.** Had the actuator been written from
`iterationCallback` (the wrapper design), it would sit *behind* this new early return and behind
`!report.Has("Rating/1v1")` (true on ~26 of 27 iterations) — **the stage scales would never be
applied for the entire programme.** Because the actuator lives at `Learner.cpp:3944`, the PHASE_B
gate is a standalone three-line edit with no interaction.

### 8.8 The one-line goal-critic guard fix (independent; ship regardless)

The four advantage injections: `Learner.cpp:4759` (headroom, guarded), **`:4821` (goal-critic,
UNGUARDED)**, `:4900` (RND, guarded), `:5227` (Ladder drive, guarded). Verbatim at `:4821`,
confirmed at source: `if (config.ppo.goalCritic.beta > 0 && goalAdvStd > 1e-8f && advStd > 1e-8f) {`

```cpp
-   if (config.ppo.goalCritic.beta > 0 && goalAdvStd > 1e-8f && advStd > 1e-8f) {
+   if (config.ppo.goalCritic.beta > 0 && !steerRatingTripped
+       && goalAdvStd > 1e-8f && advStd > 1e-8f) {
        ...
+   } else {
+       report["GoalCritic/Blend BetaEff"] = 0.f;   // panel is written only inside the branch (:4833)
```

**This is the largest of the four injections and the only one the rating latch cannot reach.** It is
orthogonal to the reward programme — ship it with S0.

### 8.9 Per-stage instantiation

All stages: same CUSUM, μ₀ **re-frozen** over the K serve groups immediately preceding that stage's
edit, **never inherited**.

| stage | ships as | trip action |
|---|---|---|
| **S0** | all `actScale = 1`, no reward change | no CUSUM trip (S0 *is* the baseline); integrity gate only |
| **S1** | `GuardedPickupBoost` stays instantiated at `ship = 0 / safe = 1` | `actScale → 1` + delete stage marker + halt programme |
| **S2** | A/B pair of `AerialTouchReward` (incumbent, airFrac-fixed) at `(1,0)/(0,1)` | crossfade back to `(1,0)` + halt |
| **S3** | A/B pair `TouchAccel` (incumbent) / `GoalDirectedTouchAccel` at `(1,0)/(0,1)` | crossfade back + halt |
| **S4** | `ConcedeAccountability` at `ship = 1 / safe = 0`, ZeroSum-wrapped | `actScale → 0` + halt |

**Secondary automatic arm, S3 and S4 only** (their named failure mode is ball-tethering):
`Player/Ball Touch Ratio` rolling-200-iteration mean outside **[0.048, 0.144]** (mean 0.09616,
population sd 0.02392 over 2.4–2.94 B; ±2 population sd) [M, 6M]. Rolling-mean SE at 200 iterations
with lag-1 0.70 is 0.00292, so the band sits **16 SE from centre** — it cannot false-trip on noise
and only catches gross behavioural collapse. *(v1's proposed [0.075, 0.120] band was labelled "±4
population sd" but is **±0.95 sd** — it substituted the SE of the mean for the population sd — and
**33.3 % of healthy iterations already sit outside it**. Under the literal "200 consecutive" reading
it is unfireable: longest healthy consecutive excursion = 14.)*

**S0 must DEMONSTRATE the actuator, not describe it.** On a `GGL_SMOKE` + **`GGL_DEVICE=cpu`**
sandbox (never on the GPU — **two OOM crashes of the live trainer are on record from smoke runs
without it**): a forced trip must (a) flip `Guard/Act Scale S1` from 0.0 → 1.0 within 1 iteration,
(b) write `REWARD_GUARD_TRIPPED` and force a save, (c) survive a process restart with the scale still
at 1.0, and (d) survive a restart onto a checkpoint 200 M steps older than the trip.

### 8.10 Telemetry

`Guard/Act Scale <term>` (one series per `rewardAct` entry — **the single most important panel: it
is the proof the actuator actuated**), `Guard/Tripped`, `Guard/Trip Reason` (0 none, 1 CUSUM,
2 rating latch, 3 serve starvation), `Guard/CUSUM S` (σ units), `Guard/Baseline Share` (μ₀, pp),
`Guard/Sigma Serve` (σ̂), `Guard/H`, `Guard/Serve Share`, `Guard/Serve Count`,
`Guard/Serves Since Last`, `Guard/Rating Latch`, `Guard/PhaseB Held`, `Guard/Edit Timestep`,
`Guard/Returns STD Jump`. Existing series that become guard readouts: `Rewards/<term>` (unscaled
pre-ZeroSum child — **a live counterfactual on a muted arm**), `GoalCritic/Blend BetaEff`,
`Steer/Rating Guard Tripped` (`:5415`), `Curriculum/Phase B Streak` (**must read 0 all programme**).

Plus the burst accumulator, which is the reason S0 exists at all: **today the only in-run Nexto
signal is three cumulative counters** (`Learner.cpp:4953-4955`) emitted **after** the next collect
thread is launched (`:3972` precedes `:4952`), so serve increments and their goals land on different
report rows — directly visible as `inc = 1` rows carrying a median of only 29 goals while the bulk
arrives on `inc = 0` rows [M].

**Burst-level standardisation is mandatory.** Single-serve bursts score systematically ~9 pp lower
than multi-serve bursts within every block (6.19 vs 12.68 at 0–0.3 B; 34.50 vs 44.31 at 2.7–3.0 B),
and the goal-weighted short-burst fraction drifts monotonically **0.079 → 0.847** across the run —
purely an artifact of iteration wall-clock period feeding the LCG [M, 6M]. **So the guard must
standardise by burst length** or it fires on composition drift. Direct standardisation moves the
block series to 9.76 / 14.85 / 18.18 / 22.04 / 26.69 / 29.15 / 39.91 / 38.70 / 36.95 / 39.89 /
**43.37** — the crude series **understates** the current level by ~4 pp and the recent slope by
~1 pp/B [M, 6M].

---

## 9. Pre-registered criteria — every one can FAIL on a null

Non-inferiority phrasing ("slope ≥ 0", "does not fall", "not worse than the previous stage") is a
**release** condition and **never an outcome**.

### 9.1 The two instruments, and what each may claim

**Nexto share is a HARM detector only** (§6). MDE 6.34 pp per 0.5 B stage against expected effects
of 0–0.75 pp/stage.

**Efficacy moves to per-iteration panels, tested as a discontinuity — but NOT as a difference of
window means.** The drafted RD design is **trend-biased**, worst exactly where it was leaned on
hardest: at `Player/Boost`'s +2.142/B, ±0.1 B windows manufacture a spurious **+0.214** with no
discontinuity present, and the *lagged* confirmation window (post 0.15–0.35 B) — the one introduced
to defeat mechanical rescale — manufactures **+0.643 against its own 2σ bar of 0.79, i.e. 81 % of
threshold under a pure trend** [D, 6M].

**Replace with a local-linear intercept discontinuity:** fit `y = a + b·t` by OLS on each side of
the edit over ±0.25 B, test `a_post − a_pre` at the boundary. **Trend enters `b`, not `a`; a pure
trend gives jump = 0 exactly.** Cost: the boundary intercept has 4× the variance of a window mean,
so SE/side doubles; widening to ±0.25 B recovers most of it.

Re-derived bars (`n_eff = n(1−α)/(1+α)`), all [M, 6M] over 2.4–2.94 B:

| panel | mean | sd | lag-1 α | n_eff/side @±0.25 B | SE_jump | bar |
|---|---:|---:|---:|---:|---:|---:|
| `Player/Boost` | 15.632 | 3.642 | 0.62 | 425 | **0.50** | 3σ = **1.50** |
| `Player/Aerial Touch Ratio` | 0.003543 | 0.001300 | 0.50 | 605 | **1.50e-4** | 3σ = **4.5e-4** (12.7 % of mean) |
| `Player/Ball Touch Ratio` | 0.09616 | 0.02392 | 0.70 | 320 | **3.78e-3** | 2σ = **7.6e-3** |
| `Average Step Reward` | 0.2944 | 0.4446 | 0.62 | 425 | **0.061** | 3σ = **0.183** |

**Nexto-served iterations must be excluded from behavioural RD windows** — they change the opponent
distribution and shift every `Player/*` panel; 4.89 % contamination inflates SE ~5 %.

**Three specs report three different slopes for `Player/Boost`** ("FLAT since 0.9 B"; +0.381/B over
0.9–3.44 B; +2.142/B over 2.4–2.94 B). **Reconcile from the raw series before S1 freezes its
pre-window.** The local-linear estimator makes the criterion insensitive to which is right; the
sizing table above uses the steepest.

### 9.2 S0 — infrastructure, 0.7 B (~2.8 h). No reward change. BLOCKING.

Ships: the actuator at `ship = safe = 1` everywhere; the boot apply; the serve accumulator + CUSUM +
all `Guard/*` panels; the goal-critic guard fix (§8.8); the B0 boost panels (§7.3); TEL-1 (§7.4);
S3's alignment panels (§7.5); and **every manipulation panel S1–S4 will later read** — panels must be
**baselined before the stage that reads them** or the stage has no pre-period.

**OUTCOME (identity deployment, falsifiable on a null):**
- every `Rewards/<term>` window mean within **±2 %** of its 0.2 B pre-deploy mean (these panels are
  pre-scale but **behaviourally** sensitive, so drift ⇒ the deploy is not a no-op);
- `Average Step Reward` local-linear jump within **±0.183** (3σ);
- `Guard/Act Scale <term>` = 1.0000 for every entry, `Guard/Tripped` = 0 throughout;
- **integrity identity**: Σ over reconstructed serve groups of `N_g` equals the cumulative counter
  delta **exactly**.

**CALIBRATION (blocking; all five must return a usable answer, and each can return a null):**
1. **σ̂ measured live** on the in-process serve series (n ≈ 80).
2. **False-positive replay** of the shipped rule over the S0 series, μ₀ re-estimated inside each
   trial, sliding the freeze point over every index ≥ K: choose the smallest `h` with **≤ 1 % FA per
   57-serve stage**. ≥ 2 clean-data fires at the chosen `h` ⇒ recalibrate.
3. **Fault injection**: a synthetic −20 pp step injected at a random index of a replay of the last 60
   real groups, 200 seeds — must be caught within **10 groups in ≥ 95 % of seeds**. **A guard that
   never fires under injection FAILS.**
4. **Actuator demonstration** on CPU smoke (§8.9), all four clauses.
5. **RD SE self-check**: the four SEs measured live over S0 must agree with §9.1 within a factor of
   1.5. Any panel beyond that ⇒ its autocorrelation changed, every downstream criterion is
   mis-sized, and **the programme stops at S0 for re-sizing**.

**Additionally on `resid`: S0 must establish the do-nothing null itself** — the burst-standardised
share trajectory over its own 0.7 B — because §5 row 0 no longer has a value.

### 9.3 S1 — mute `GuardedPickupBoostReward`. 0.5 B.

**PRE-GATE (on S0 data, before S1 ships):** `Diag/Boost At Pickup` **< 45**. ≥ 45 ⇒ **does not
ship** (§7.1).

**OUTCOME:** local-linear RD intercept jump in `Player/Boost` ≥ **+1.50** (3σ), **and** a jump in
`Diag/Boost Under12 Frac` of the corresponding sign at ≥ 2σ (a second, differently-mechanical
channel; this replaces the drafted lagged window, which is 81 % trend-corrupted). **Null jump = 0 ⇒
fails with P = 0.9987.**

*The drafted `Player/Boost ≥ 17.0` bar is **DELETED as null-passing**: the measured trend alone
reaches 16.85 in 0.5 B. It rewarded inaction.*

**MANIPULATION:** `Guard/Act Scale GuardedPickupBoost` = 0.0000; applied density falls by
`2 × 6 × E[child]` = **0.0266 ± 0.003** units/player-step, computed from the unscaled panel.
*(The drafted check "`RewardMag` → exactly 0.0" is **wrong under this actuator**: `zeroSumPtr`
survives, so the panel keeps reporting the raw child and does **not** go to zero.)*

**FREE MEASUREMENT (settles C3):** §7.1.

### 9.4 S2 — `AerialTouchReward` airFrac AND-restore. 0.5 B. **BLOCKED**

**Precondition (C5): the reward body does not exist in writing and cannot be farm-audited.** The
~0.8 s refire cooldown is the obvious cooldown-gaming surface and S2 edits exactly that term.

For the record, when it is written: the reset **must** be placed **before** the early return at
`CommonRewards.h:517` (`if (!state.prev || !player.ballTouchedStep || player.isOnGround) return 0;`)
— **an `isOnGround` reset placed after it never executes**, and the term goes permanently dead after
the first landing. The mechanism is a per-player `airTimeAtLastPay` so `creditedAir =
max(0, airTime − airTimeAtLastPay)` and each payout re-earns 1.75 s. Engine premise verified
directly in the Rust: `air_time = 0.0` under `is_on_ground` (`RocketSimV3/rocketsim/src/sim/car/base.rs:569-572`)
[M]. Constants: `MAX_CREDIT_AIR_TIME = 1.75f` at `:501`, `REFIRE_COOLDOWN_STEPS = 12` at `:503`
(**not** `:500`/`:501` as an earlier draft said).

**OUTCOME (when unblocked):** post-edit slope of `Player/Aerial Touch Ratio` ≥ **2.9e-3/B** against
the pre-stage 1.68e-3/B — a +71 % increase, 2σ on the combined slope SE 5.9e-4/B. **Null (slope
unchanged) fails with P = 0.977.** *(The "> −3.6e-4 jump" clause is a release condition, not the
outcome.)*

**MANIPULATION:** `Rewards/AerialTouch` falls **10–50 %** relative. Below 10 % the AND never landed;
above 50 % the formative scaffold is gutted and **the stage reverts on that clause alone**.

**Density risk, stated:** [E] −0.02 to −0.06 units/player-step, **unbounded by its own outcome
criterion. This is the stage's real risk** — it is a deflation of the only measurably improving axis
dressed as a correctness fix. Farm ceiling before: ~121 reward/s; after: ~55 reward/s (a 2.2× cut);
**realized today: 0.088 reward/s/player, 1,375× below the old ceiling** [D, 6M].

### 9.5 S3 — `GoalDirectedTouchAccel` A/B swap. 0.5 B.

**PRE-GATE:** `Guard/OwnwardTouch Frac` ≥ **0.02** over S0 (§7.5).

**OUTCOME:** `Guard/OwnwardTouch Frac` (S0-baselined) falls ≥ **40 % relative**, **and**
`Player/Ball Touch Ratio` local-linear jump within **±7.6e-3** (2σ, the anti-tethering release).

**MANIPULATION:** applied density of the B arm within **±10 %** of the incumbent's 0.0538
(neutrality landed), and `Guard/Act Scale` reads (0,1) — the crossfade actually crossfaded.

### 9.6 S4 — `ConcedeAccountabilityReward`. 1.0 B. **BLOCKED**

**Precondition (C5): body unwritten, un-auditable.** Must ship ZeroSum-wrapped — a positive-part-only
concede term cannot be latched without breaking antisymmetry.

**Structural limitation (§6):** measurable only against a fixed external opponent.

**OUTCOME (when unblocked):** Nexto **goals-against per serve group falls ≥ 20 % relative** vs a
re-frozen 57-group baseline (log-sd of GA/serve = 0.4479 ⇒ SE of a 114-vs-57 contrast = 8.3 %
relative ⇒ 2.4σ; **null fails with P ≈ 0.99**). *(The drafted ≥ 8 % bar has ~25 % power —
unreachable. The drafted "share rises ≥ 5.0 pp" bar sits at the instrument's MDE; keep it as a
**secondary report**, not a gate.)*

**MANIPULATION:** `Guard/Concede Dist` (mean conceder→play distance at the goal instant) falls
≥ 20 %; `Guard/Concede Acct Imp Share` ≈ 0.

**Note C1:** the +3.1 % density is *not* mostly reallocated — at these horizons `returnStd` moves
+0.40 %, so this addition genuinely adds. **That makes S4 cheaper than V2 claimed and its farm audit
correspondingly more important.**

If it is ever written, two design constraints are load-bearing and were correctly identified: **the
ring buffer** (at the goal step the ball is past a back wall by construction, so a naive version is a
distance-to-own-net penalty, i.e. it teaches **net-camping**; key to the ball ~1 s earlier), and
**the alive counter** (a demoed/just-respawned conceder's position is a respawn point, not a
decision — and post-ZeroSum the wrapper hands the **scorer** `+x`, so without it the stack pays extra
for **"demo, then score while they respawn"**, worth up to 20 + 37.5 = 57.5). **Impossible-arena
inertness** is required but the exposure is **0.78 % (8 of 1024, `src/ExampleMain.cpp:1196`), not
~11 %** as an earlier draft claimed [M] — and reward scaling cannot void the Ladder certificate
regardless (the counter keys on `ballTouchedStep`, `Learner.cpp:3741-3746`).

---

## 10. The graveyard

Every rejected idea with its kill reason. **Nothing is silently dropped.** `(v1)` = carried forward
from the first proposal; `(V2-defer→KILL)` = V2 banked it as a deferral with an explicit re-open
condition, **the condition was run, and it failed**.

### 10.1 Falsified claims

| claim | kill reason |
|---|---|
| **"The run has plateaued"** (v1) | **FALSIFIED.** Overlapping 150 M windows understate the SE and, fitted through the 1.8 B excursion, manufacture a flat line. Non-overlapping blocks give **+9.17 ± 0.70 pp/B** from 0.0 B and **+4.23** from 2.1 B, last four blocks rising monotonically. §3.1 |
| **"−14.1σ serve deficit"** (v1) | The **level is right** (4.89 % vs 15 %) and the **statistic is void**: it assumes independent Bernoulli draws on a process whose serve gaps have **sd/mean = 0.10** (Bernoulli gives 1.0). Argue from the 3× level gap and the reproduced closed-loop model. §3.3 |
| **"Goals against are 63–74 % of the deficit"** (v1) | **Category error.** 22,503/36,127 = 62.3 % is the concede share of *goals*, not a decomposition of the gap. Marginal leverage: `d(share)/dGF = 1.4353e-5` vs `|d/dGA| = 9.156e-6` — **offense has 1.57× the per-goal leverage.** The defensive case rests on D1 instead. |
| **"The objective owns 4.6 % of the gradient, inverted by ~50×"** (v1) | Apples-to-oranges **twice**: Goal's 0.55 is post-ZeroSum, AerialTouch's 0.0585 is the pre-ZS child mean (`ZeroSumReward.cpp:5`). In one currency Goal owns **32.3 % of credit density and 61.2 % of relocating magnitude**; Goal/AerialTouch = **5.56×**. §3.4 |
| **"The whole stack sums to 0 across players"** | **False.** `TimeCostReward` is unwrapped and non-antisymmetric (`src/ExampleMain.cpp:252`): **−0.02/pair-step**. Benign for PSD ranking (a constant, and PSD fitness is **mean per-step**, not episodic — correcting a further mis-statement); **not** benign for `Average Step Reward` as an identity check. |
| **"S1 deletes a 5.7× / 1.13 units-per-second boost farm"** | **Pre-ZeroSum figure.** Wrapped (`:174`), mirror self-play gives `E[r_i − r_j] = 0` identically, and 95.1 % of iterations are mirror. Restated as a **differential ratchet** worth 0.0266/player-step; conviction now rests on the `Diag/Boost At Pickup` pre-gate. §7.1 |
| **"CarEnergy's demo guard is strictly wider than GuardedPickupBoost's"** | **Identical truth tables** over all four `(isDemoed, prev->isDemoed)` states (`CommonRewards.h:199` vs `:639-640`). B4's conclusion survives; its reason does not. |
| **"AerialTouch's collapsed AND is being farmed"** | **NOT convicted.** Realized payout is **1,375× below the ceiling** [D]. S2's claim is only that the largest scaffold's documented AND is not implemented. |
| **"~26 % of a stage's variance stays baked in after a revert"** | **Backwards.** `returnStat` is a 2.58 M-sample cumulative Welford; **~99 % of the pre-stage variance is retained** and the stage barely registers. §3.4 (C1) |
| **"Impossible-intercept exposure is ~11 %"** | **0.78 %** — `impossibleArenas = 8` of 1024 (`src/ExampleMain.cpp:1196`). |
| **"WallJumpToBall is 3.7 % of density"** | **0.064 %.** The 3.7 % came from an offline proxy on a 5.0v3 checkpoint; the live panel reads exactly 0.0 on 1,865 of 3,185 points. §3.4 |

### 10.2 Killed reward changes

| idea | kill reason |
|---|---|
| **Stage A — raise `serveFrac` 0.15 → 0.35** (v1) | Would realize ~12 %, not 35 %, because the gate is **resonant, not Bernoulli**. Forcing a true 15 % at the current CPU cost is +37.5 % wall clock (SPS ×0.73) = **−1.14 pp/B**. And it more than doubles exposure to an opponent outscoring us 1.57:1 under a zero-sum invariant where every wrapped term charges us their income. **Replaced by: re-trace Nexto for CUDA.** |
| **Stage 1a — un-gate the demo branch on the potentials** (v1) | There are **five** demo-gated potentials, not v1's three (BallProximity `:128-133`, CarEnergy `:199-200`, AirIntercept `:619-621`, ConsecutiveAirTouch `:765-769`, WallJumpToBall `:832-837`). Total leak: **1.1e-4 units/player-step = 0.005 % of density** — the entire defect is the size of `KickoffRace`, the smallest live term. Unconvicted. *(One free finding: `AirInterceptPotentialReward`'s own `:600` comment says "NEVER gate" and `:619-621` gates it — a **one-word comment fix**, not a code change.)* |
| **Stage 1b — uniform terminal potential refund** (v1) | **The algebra inverts the argument.** The exact per-step mean is `[γ·Φ(s_N) − Φ(s_0) − (1−γ)·Σ Φ_t]/N`; v1 dropped `+γ·Φ(s_N)/N`. At this run's parameters `γ/N = 4.363e-3` vs `(1−γ) = 3.1e-3`, so the terminal term is **1.41× the discount drag** and a **positive panel is the correct-implementation expectation** below `N* = 321.6` steps (episodes are 228.5). Empirically decisive: 8 windows with N > N* have mean ConsecAir panel **−5.7e-5** (3/8 positive), the 66 with N < N* have **+5.6e-4** (61/66 positive), **Fisher exact p = 7.3e-4** [M, 6M]. The sign flips exactly where the algebra says it must. Additionally its dominant component is B2G's +26.35 at a goal, so its main real effect is a **15 % CUT to the effective goal price** (176 → 150) — the opposite of v1's own diagnosis. |
| **Stage 1c — Goal 150 + measured compensation** (v1) | Exists only to undo 1b. With 1b dropped it is a no-op. If the objective deserves more weight, **raise the literal** — one literal in, one literal out — rather than a state-dependent path sized by an unmeasured quantity. |
| **Stage 1d — delete `WallJumpToBallReward` (30)** (v1) | **Reversed: KEEP.** Live rate 4.66e-4/player-step ⇒ **A = 0.0013 (0.064 %)**, so deletion frees nothing, and it removes the gradient any future `WallDriveState` seeding would need — zero upside, negative option value, costs a rebuild. v1's supporting citation ("Opti's single largest shaping weight is wall_touch = 15") **does not survive the Opti source**: `wall_touch_w` defaults to **0** (commented 0.25) and the term fires on a *grounded* touch of a high ball in a non-zero-sum self-reward. |
| **Stage 1d — delete `FlipResetReward` (40)** (v1) | **Reversed: KEEP.** 0.0007 % of density; deleting it changes nothing measurable and breaks the deliberate reward+seeding pairing with `AirPlayState`'s FLIP_RESET_READY slice (`src/ExampleMain.cpp:226`: *"a zero-rate mechanic needs both"*). |
| **Stage 2 — TeamPressure 0.15 → 0.05** (v1) | The convicting measurement is real (in PHASE A the `p.team == player.team` loop is self-only, so it is a per-player ball-tether penalty firing on 14.85 % of player-steps), but the magnitude is **1.36 % of density** and its criterion ("Nexto slope ≥ 0") **passes on a null**. It is also the only per-step anti-disengagement pressure in the stack. Deferred, not killed. |
| **Stage 4 as v1 specified it** (`push = dv·n̂ > 0` gate, credit `push·(n̂·ĝ)`, no speed clamp) | **Six independent kills.** (1) It **inverts the sign of E[attempt] on the chase-down clear**: a successful corner clear has `push < 0` and pays 0, while a failed touch that deepens the ball has `push > 0` and pays negative — the whiff-tax structure `STEERED_PRACTICE.md` convicts. (2) `MIN_SEP = 60 uu` is **dead code**: `BALL_RADIUS = 92.75` alone exceeds it. (3) Dropping the absolute-speed clamp converts a term bounded at ≤ 1.0 per acceleration-from-rest into a **~50 reward/s annuity against a 31.6 reward/s whole-stack density**. (4) `MIN_IMPULSE = 100` survives a rolling carry by only 7 % (true max ground Δv is **92.78** uu/s/step, not the "~67" inherited from `TouchHeightReward`'s comment) and **opens on a wall carry** (√(92.78² + 43.33²) = 102.4). (5) `dv` carries a fixed **−43.3 uu/s gravity term** never modelled. (6) The gate rejects wall rebounds *behind* the ball but **credits floor rebounds under it**, because on the ground n̂.z > 0. **Replaced by §7.2.** |
| **Stage 5 — `GoalsideEngagementPotential`** (v1) | **Two independent kills.** (a) Its Φ **does not measure goalside position**: `along = (p.pos − ball)·û` with û pointing ball→own-goal, banded at 900 uu, so a car 900 uu own-goal-side of a ball parked in the **opponent's** net scores depth = 1.0 — a trail-the-ball potential, a near-duplicate of `BallProximityPotentialReward`. (b) Even repaired, **an exact PBRS provably cannot do what Stage 5 is asked to do**: with matched γ its total episodic contribution is `−Φ(s_0)`, a constant. It buys sample efficiency, not a new optimum. And its only honest outcome bar needs a ≥ 4 B window. |
| **Seer's anti-dribble decay envelope (λ) on `AerialTouchReward`** | Wrong repair for the named defect, and **unobservable**. It damps *repeats*; the defect (D4) is a **collapsed AND**. λ has an effective memory of ~69 steps = 4.6 s — **3.7× the 18.9-step GAE window** — and appears nowhere in the 230-dim obs, so the critic must marginalize it away: the identical objection that killed `FlipResetConversion`. The "provably inert" framing is **withdrawn**: as a bolt-on *beside* the 12-step cooldown it is inert (fixed point 2.86 → clamped to 1.0), but **as a replacement for the cooldown it is worse** — Seer decays per *touch* and `ballTouchedStep` fires every step of sustained contact, so λ pins at its 0.1 floor during a carry while payouts arrive ~12× more often, netting ≈ **1.2× the income**. |
| **rlgym-tools `AerialDistanceReward` replacing `ConsecutiveAirTouchReward`** | A **non-potential event reward with unbounded per-episode income**. Replacing an exact PBRS with it reverses the property the term was explicitly built for (the user's "unfarmable by making them pbrs" requirement, `src/ExampleMain.cpp:203-209`) and its **argmax is TRAVEL, not control**: at w = 30 one cross-field air carry pays 30 × 6000/5120 = **35 units = 23 % of a Goal for one touch, unrefunded**. And there is no defect left to fix — the incumbent telescopes to `−Φ(s_0) = 0` on every non-terminal path. |
| **Nexto-style height-tapered boost-spend penalty (B3)** | Nexto needs `boost_lose` because its stack has **no energy potential at all**; Pulsar prices spend two-sidedly at **11.5 % of density**. Porting it would (a) **double-price ground spend** while CarEnergy's `ke` term simultaneously *pays* for it above `v* = 444.6/√(0.01b)` — two live terms with **opposite signs on the identical transition**, flipping at 1119 uu/s with 43.3 % of steps below; (b) be one-sided and non-PBRS — the mirror image of the ratchet S1 mutes — with a convicted **hoarding** failure mode, against a run whose `Player/Aerial Touch Ratio` is its healthiest trend (**4.8× over 3.4 B**); (c) **reinforce the very pathology it targets**, because the taper zeroes the penalty at height, so the cheapest way to satisfy it is *get high first, spend there* = **takeoff on an empty tank**; (d) open a demo laundering channel (die full, respawn at 33.33 untaxed). The physics taper is also the **wrong sign**: air boost is already **6.30 % more efficient** than ground (implied taper 0.937, not a height ramp). Sizing it to be even visible needs `w_s ≈ 1.4` — noise. **Reconsider only if** B0 convicts on **both** clauses **and** B2 is measured insufficient at the band limits **and** the scaffold has annealed to AerialTouch ≤ 50 / AirIntercept ≤ 20. All three. §7.3 |
| **A dedicated boost term after S1 (B1)** | The axis retains **89.8 %** of its density (0.2340 of 0.2606) and is the **4th-largest signal in the stack**. Every non-PBRS form is convicted (one-sided pickup = the ratchet S1 mutes; one-sided spend = hoarding); every PBRS form is optimum-neutral, so the cheap lever is the `(c,p)` shape knob **inside** the existing Φ. **Ceiling if ever admitted: ≤ 0.03 density.** §7.3 |
| **Adding a dedicated boost spend-side term** (v1) | The axis **is** covered, two-sidedly, by CarEnergy's `sqrtf(0.01f*p.boost)` at w = 15. The measured pathology is not a missing spend side — it is a **positive-part-only pickup payment**. |
| **rlgym-tools solid-angle `GoalView`** *(V2-defer→KILL)* | V2's own re-open condition ran and returned the **opposite sign**: AUC **0.5972 vs the incumbent's 0.6009**; stratified **0.5198 vs a 0.56 bar**. Also 5.19× hotter per step, 9.3× fatter tail, density-neutral weight 14.5. §7.4 |
| **Lucy-SKG Offensive Potential / KRC** *(V2-defer→KILL)* | Measures **worse** than Pulsar's 4-weight BallProximity on Pulsar's data: AUC **0.5107 vs 0.5260**, stratified increment **0.5026**, and the alignment factor alone reads **0.4955 — below chance**. Density-neutral weight **1.55**; p99.9 \|ΔΦ\| = 0.7994 (**42× B2G's tail**) from `sgn()` flips. §7.4 |
| **Goal-speed-scaled objective (GS-1), both shapes** | Goal speed is **flat at 1742 uu/s since 2.0 B** and moves **+1 % across +700 Elo** of the mature lineage. Speed is already priced at the touch and at **7.0 units/s by discounting**. Exact **stall farm**: t\* = 1.52 s at w = 15, **7.6–15.2 s multiplicative**, with **no terminal to stop it**, plus a conceder-side incentive to **slow the ball at the line**. Multiplicative additionally redenominates PSD fitness. §7.4 |
| **Raising/removing `MAX_REWARDED_BALL_SPEED` (SP-1)** | The cap is a **divisor**: `C → 6000` at w = 10 cuts every ordinary touch's rate **49 %**; rate-preserving needs w → 19.6 (2.6 % → 5.1 % of density). Removing it opens a pump farm and destroys the `prevFrac = 1` defence. Killed pending TEL-1; falsifiable at 0.05/0.10. §7.4 |
| **Goal-speed-scaled goal / aerial-goal bonus** (Nexto, Opti, Seer, Lucy all ship one) | Makes the **objective** state-dependent, on a term that already owns 61 % of the relocating budget. If finishing speed matters it should be priced on the **touch** — which is exactly what S3 does. |
| **`ContestedSteal` at weight 8** (v1) | ~40 % of the post-deploy stack; a soft poke that *returns* possession re-arms every 8 steps for **~15 reward/s (≈4× the whole honest stack)**. Only the *responder* is paid, so touching first is a 16-unit liability → hesitation gradient and a **mutual-abstention equilibrium**. |
| **`PadAccessMargin`** (v1) | **Sign-inverted**: `need()` sits inside the agent's **own** access term, so it pays for burning your own boost **and for the opponent refuelling**, partly cancelling CarEnergy. |
| **Possession potential with `ctrl = exp(−\|v_ball\|/2300)`** (v1) | A **shot-power tax**: cancels ~70 % of the touch reward for a hard strike, and above 3055 uu/s the stack's marginal incentive on shot power goes **strictly negative**. |
| **Territory potential `R_own·u − R_opp·(1−u)`** (v1) | `u → 0` at your own goal line **annihilates your own reach term exactly where the defensive conviction lives** — at the goal line, winning a free ball pays 0.000. |
| **Goalside `cover = clamp(along/(0.35·L))`** (v1) | `depth → 0` at contact ⇒ **charges the defender for making the save**; argmax is inside the net. |
| **`OnTargetPotential` with a lateral `exp(−\|b.x\|/1200)` factor** (v1) | Both goal backs are at **x = 0** (`CommonValues.h:29-30`), so the lateral factor **factors out of the difference** — the term collapses to a context-dependent reweight of B2G, not a new axis. *(Correct for that form only; the solid-angle form does not factor out — it is killed separately, on measurement.)* |
| **Removing `TimeCost` and `TeamPressure` together** (v1) | Leaves an exactly-zero-sum stack in which a 20 s mutual stall returns exactly 0.0000 to both players, and `NoTouchCondition` is a bootstrapped truncation so the boundary costs nothing either. **Mutual disengagement becomes a stable fixed point. Keep `TimeCost`.** |
| **`FlipResetConversion` (mechanic-then-goal)** (v1) | The conversion window is cleared on landing, but the car lands ~1.2–2.1 s after the reset — *before* the ball crosses — so **the conversion branch never fires**. Its 75-step credit also sits far outside the 18.9-step GAE window, and `convWindow` is not in the obs. |
| **Aggression bias (asymmetric goal/concede)** (v1) | Destroys the whole-stack zero-sum invariant; PSD probe fitness is literally mean per-step reward (`PSDController.cpp:452-459`), so it becomes **farmable per-player income**. Architectural cost, not a tuning knob. |
| **Scoreboard win-probability potential** (v1) | `ScoreLine` is declared but **never instantiated as a `GameState` member**; episodes are goal-terminated with no clock. Obs + episode-structure redesign, not a reward change. |
| **Gating a new potential on demo** (v1) | Refunds an uncharged Φ drop. Stacked on the five existing gated potentials, a sixth at weight 25 refunds **~61 % of `DemoReward`'s 37.5**. |

### 10.3 Killed guard / methodology designs

| idea | kill reason |
|---|---|
| **`StageScaled` wrapper as the reward actuator** | Outside `ZeroSumReward` it **nulls `EnvSet.cpp:111`'s cached `dynamic_cast`**, so `:259-260` logs the post-ZeroSum, post-scale value and **all 16 `Rewards/*` panels silently change meaning** — the ×1.0 "no-op" fails its own S0 identity test. Rediscovery of V2's own *"wrapping outside ZeroSumReward"* kill. Also leaked `child` and could not key an A/B pair. **Replaced by `WeightedReward::actScale`.** §8.1 |
| **Name-keyed actuator whitelist** | `RewardWrapper::GetName()` forwards the child and `Reward::GetName()` is `typeid`-derived, so **the two arms of an A/B pair are indistinguishable** and the crossfade would set both to the same scale — **a silent no-op.** Key by **index**. §8.1 |
| **Reusing `steerRatingTripped` as the actuator** | Structurally blind; blast radius of six subsystems; **stops the Nexto yardstick at the moment you need it**. §8.0 |
| **Reusing the reachability gated channel** | Positive-part only by design (`EnvSet.cpp:232-235`) — latching a one-sided ZeroSum-wrapped term mutes the scorer's `+x` and leaves the conceder's `−x`, **breaking the invariant exactly when the guard fires**. §8.0 |
| **Writing the actuator from `iterationCallback`** | **Data race** (no `join()` between `Learner.cpp:3972` and `:5511`), *and* it sits behind two early returns on ~1-in-27 iterations. §8.0 |
| **Dual-rolling (R56/R14) burst-share trip rule** | False-positive arithmetic understated **~25×**: omits the frozen baseline's own variance (`sd(r56−b0) = 0.02364` ⇒ **0.05 is 2.12σ, not 3.00σ**) and treats a rolling sequence as one test. **≈ 4 %/stage, ≈ 15 % programme-wide** vs a claimed 0.13 %. Cannot fire before serve 14. **Replaced by a serve-indexed CUSUM.** §8.3 |
| **Hard-coded CUSUM `h = 8σ` with `σ = 0.5252` logit** | σ was measured on the **wandb** series, whose rows are mid-collection snapshots written during learn **while the worker mutates the same atomics**. The shipped statistic reads post-join deltas at `:3944` and is exact by construction, with a different, smaller σ. The `[0.40, 0.66]` sanity band would likely sit **permanently outside its own alarm**. **Derive σ̂ and h live over S0.** §8.3 |
| **Trip state in the stats JSON only** | `tsPerSave = 25 M` ≈ 6 min, and the corrupt-checkpoint fallback walks up to **200 M steps** back — **silently un-reverting a convicted stage.** Marker file + forced save. Also: NaN fields must use the `isnan`-guard/`j.contains` idiom or `LoadStats` **throws**. §8.4 |
| **RD as a difference of window means** | **Trend-biased**: at `Player/Boost`'s +2.142/B, ±0.1 B windows manufacture **+0.214** with no discontinuity, and the *lagged* confirmation window manufactures **+0.643 against its own 0.79 bar (81 %)**. Replaced by a local-linear intercept discontinuity. §9.1 |
| **`Player/Boost ≥ 17.0` as S1's outcome** | **Passed by the null**: the measured trend alone reaches 16.85 in 0.5 B. **It rewarded inaction.** Replaced by an RD intercept jump ≥ 1.50 (3σ). §9.3 |
| **`GAE/Returns STD` continuity as the S0 identity check** | A 0.5 B stage moves the divisor **+0.40 %**, so "no step > 1 %" **cannot fail** even for a genuinely non-identity deploy. Replaced by per-term panel identity + `Average Step Reward` RD. §3.4 |
| **Reset `returnStd` / `RUNNING_STATS.json` on each deploy** (v1) | `WelfordStat::GetSTD()` returns **1** when `count < 2`, and GAE divides rewards by it *before* the iteration's `Increment`. The first learn pass after a reset standardizes by **1 instead of 113**: goals clamp at the ±50 rail, ordinary steps inflate ~113×, critic and both trunk-coupled V-dagger heads take ~100× targets, **all five Ladder wire columns saturate**. Also a hand-edit of a rotating checkpoint's sidecar, which the **recovery doctrine forbids**. |
| **S4 outcome "Nexto GA/serve falls ≥ 8 %"** | Log-sd of GA/serve is 0.4479 ⇒ SE of a 114-vs-57 contrast is 8.3 % relative ⇒ **~25 % power**. Raised to ≥ 20 % (2.4σ). §9.6 |
| **Per-stage efficacy on the Nexto share** | **MDE 6.34 pp per 0.5 B stage against claimed effects of 0–0.75 pp/stage — a factor of 8.** Nexto demoted to **harm detection only**; efficacy moves to per-iteration panels. §6 |
| **Slope-vs-frozen-baseline Nexto gates** (v1's global rule) | **Invalid under deceleration** — the null itself is moving. Every v1 slope criterion was sized against a **zero** null, making "+5 pp/B ≈ 3.6σ" actually **0.47σ**. Replaced by level tests. §3.1 |
| **Absolute-level Nexto gates (≥ 55/60/65 %)** (v1) | A level bar against a *moving* series cannot distinguish an intervention effect from window scatter. A level bar against a **frozen contemporaneous baseline** is a different and valid construction. |
| **`Diag/SelfPlay Avg Step Reward` reads exactly −0.0100 as a criterion** (v1) | **The current code already fails it**: the panel reads **+0.2946 ± 0.0152** and 0 % of 1,595 points sit at −0.0100. The residual is unexplained. §3.5 |
| **`Player/Ball Touch Ratio ∈ [0.075, 0.120]` as a "±4 population sd" guard** (v1) | It is **±0.95 sd** — v1 substituted the SE of the mean (0.00042) for the population sd (0.02373) — and **33.3 % of healthy iterations already sit outside it**. Under "200 consecutive" it is unfireable. Replaced by a rolling-200 mean in [0.048, 0.144]. §8.9 |
| **25 % arena-slice "control"** (v1) | There is **one** policy and the 230-dim obs carries **no arena tag**, so treated and control arenas cannot diverge behaviorally — **no contrast exists**. The proposed leading slice also sits entirely inside the FrontierDrill block, so it would compare two different *reset distributions*. |
| **`\|mean(Steal Net)\| < 1e-3` / "Cover Deficit ∈ [0,1]" as falsification counters** (v1) | Both **true by construction** — antisymmetric nets are pinned at 0 by symmetry, a clamped product is range-bound by definition. |
| **Deflate the aerial scaffold (AerialTouch 120 → 55/60/70)** (v1) | Convicting evidence was the §3.4 currency artifact. Independently, **aerial is the only measurably improving axis**: aerial touch share 1.83 % → 3.80 % across six 0.5 B bins **while `Player/In Air Ratio` stayed flat at 0.59** — conversion efficiency improving, the opposite of "air-warped style". **Fix the shape (S2), not the weight.** |

### 10.4 Deferrals — banked, not killed

| idea | why deferred | what would reopen it |
|---|---|---|
| **`WallDriveState` seeding + keeping `WallJumpToBallReward`** | Wall driving is a genuine zero-rate mechanic (4.66e-4/player-step) and **no state setter puts a car on a wall**: `BallNearCarState` forces `carPos.z = 17`, `RandomState` caps \|x\| at 3900 < SIDE_WALL_X 4096 and forces z = 17 on its grounded half. **Reward cannot shape unvisited states.** | `Player/Wall Drive Frac` (§7.5) ≥ 0.020 over 200 M steps **falsifies** the zero-rate premise (then it is a weight question); < 0.005 **confirms** it and licenses the setter — gated additionally on an offline census attributing ≥ 3 pp of the Nexto deficit to wall/backboard sequences. |
| **`KickoffRaceReward` — `WINDOW_STEPS` 30 → 45** | The term is **broken, not merely small**: measured payout rate 2.19e-6/player-step, of which the `BallNearCarState(600,900)` false-fire geometry predicts **~93 % of its entire realized income** [D]. `WINDOW_STEPS = 30 = 2.00 s` at tickSkip 8 against a nearest kickoff spawn ~3278 uu away needing ≥ 2.1 s — **it essentially never fires on a real kickoff.** | A `Kickoff/Untouched Ball Frac` + kickoff-outcome panel, then the one-integer change as its own stage. Turning a dead term at w = 25 live is a real distribution change with no telemetry to gate it. |
| **TeamPressure 0.15 → 0.05** | 1.36 % of density; criterion passes on a null; only per-step anti-disengagement pressure. | A measurement that goalside retreat is specifically what triggers it. |
| **Repaired goalside Φ** = `−threat(ball)·(1 − cover(car, ball))` | Numerically verified **non-duplicate**: at identical BallProximity Φ = 0.5654 it takes three distinct values (0.000 goalside / −0.759 beaten / −0.442 lateral). **But it is still exact PBRS and therefore cannot relocate the optimum**, and its honest outcome bar needs ≥ 4 B. | Only as a *sample-efficiency* lever, after the programme, with no behavioural bar pre-registered. |
| **Possession / turnover / passing axis** | Two specific designs killed, but **the axis itself is not dispositioned** — nothing reads `lastTouchCarID` (`GameState.h:27`) for a possession change. | A turnover-rate census showing Pulsar gives the ball up materially more than Nexto per possession. **Does not exist.** |
| **Directional pricing inside `AerialTouchReward`** | S3's known limitation: on airborne touches AerialTouch outweighs the directional term **6:1** and reads no goal direction. | After S3 lands and `Guard/OwnwardTouch Frac` shows the residual is concentrated in aerials. |
| **`goalCritic.lambda` as its own config field** | Ranked #10 in §5; blocked on the offline λ sweep (#2). | The sweep showing `A_goal` held-out AUC **rising** in λ. |

---

## 11. Standing traps — record, do not rediscover

- **`src/ExampleMain.cpp:507-508` is an unbraced single-statement `if (state.goalScored)`.** Any
  "insert after line 508" instruction silently places the new panel **outside** the branch, where it
  averages over every player-step and **reads ≈ 0 by construction**. Brace before appending.
- **`EnvSet.cpp:331` calls `prevGameStates[index].MakeEmpty()` on reset**, so on the first
  post-reset step `state.prev->ball.vel` is **0**. A `state.prev &&` guard is not enough; any
  velocity-difference telemetry needs `pre > 0.f`.
- **`player.ballHitInfo.relativePosOnBall` is hard-zeroed on the v3 engine**
  (`RocketSimV3/compat/RocketSimCompat.cpp:285`, `relativePosOnBall = Vec()` unconditionally). Any
  term reading it reads (0,0,0), **silently, forever**. `ballHitInfo.ballPos` **is** populated
  (`:283` from `ev.pos`) but `ev.pos`'s semantics (ball centre vs contact point) are **[U]** — read
  the Rust emitter before using it.
- **`isFinal` is NOT "goal"**: `EnvSet.cpp:227` passes `terminalType` (a `uint8_t` 0/1/2) into a
  `bool isFinal` slot (`Reward.h:22`), so it is **TRUE at NoTouch truncations too**. Any
  terminal-keyed logic must use `state.goalScored` (`Gamestates/GameState.h:26` — **not `:25`**).
- **`Player::UpdateFromCar` nulls the grandparent** (`Player.cpp:11-12`) — **only one level of
  history exists.**
- **`EnvSet.cpp:259-260` overwrites the already-post-ZeroSum `output` from `:227` with the
  pre-transform child** (`ZeroSumReward.cpp:5` captures `_lastRewards` **before** the transform loop
  at `:19-28`). **Every `Rewards/*` panel is a pre-transform, one-sided value** — which is why
  wrapped terms show a nonzero mean and unwrapped antisymmetric ones (`GoalReward`,
  `CommonRewards.h:37-43`) cancel to ≈ 0.
- **`GAE.cpp:50-53` divides rewards by a cumulative `returnStd` that never forgets.** Composition
  changes rescale every other term, permanently and un-instrumented — but the effect at a 0.5 B
  stage on a mature run is only ~0.4 %, **and much larger on a young one**. Re-derive per run age.
- **Anything written from `iterationCallback` runs behind two (soon three) early returns** and only
  on eval iterations (~1 in 27). **It is not a general per-iteration hook.**
- **`tools/trainerctl`'s `CKPT_DIR` default was NOT bumped by commit `45a59d5`**, so `status`,
  `checkpoints`, `golden` and `restore-golden` silently pointed at the stale `checkpoints_6M`. It
  currently reads `checkpoints_resid` **as an uncommitted working-tree change** (`tools/trainerctl:29`)
  [M] — verify `TRAINERCTL_CKPT_DIR` before trusting any trainerctl output, and expect this to break
  again at the next lineage change. **Same bug class as the render-mode arch-flag parity issue
  (`ba4f33a`)**: a checkpoint-folder or architecture change must be propagated to *every* independent
  consumer, and each one fails silently.

---

## 12. What this does not fix, and the open questions

### 12.1 The reward stack is not what is between Pulsar and the best bots

Stated plainly because the ask was "beat the best bots out there". **Beating Nexto is necessary, not
sufficient** — the 2023-25 RLBot ML winners are Seer, LeBot+ and Slater. Four constraints outrank
everything in §6, and **none of them is a reward term**:

| # | constraint | evidence | why no reward term fixes it |
|---|---|---|---|
| **12.1a** | **Action-space ceiling.** 90 discrete actions (24 ground + 66 air), no intermediate stick deflections, at 15 Hz. Opti runs ~373 actions at 30 Hz | `DefaultAction` table; `cfg.tickSkip = 8`, `cfg.actionDelay = 0` (`src/ExampleMain.cpp:552-553`) [M] | Reward shapes *which* action is chosen, never *which actions exist*. A car that cannot deflect 40 % right cannot make some shots at any weight. |
| **12.1b** | **State distribution.** 0 % replay-derived reset states; the mix is `BallNearCarState` 0.30, `AirDrillState` 0.20, `AirPlayState` 0.15, `KickoffState` 0.10, `RandomState` 0.25 (`:398-409`) [M]. The project's own record: ground takeoff was 0/300 for 30 B steps **with the reward live** and cracked on **seeding**, not weight | [M] | **Reward only shapes visited states.** This is the single largest structural difference from Nexto's training, and it is a **curriculum** lever. FRONTIER.md's Phase-0 telemetry is implemented and **has never been switched on** (§5 #5). |
| **12.1c** | **Credit window.** γλ = 0.947055 ⇒ 1/(1−γλ) = 18.888 steps = **1.259 s**; 90 % of GAE weight within 2.82 s; weight on a residual 5 s ahead = 0.017 [D]. On the goal channel λ truncates a 111 s horizon to **1.32 s** — an **84× cut** (`GOAL_CRITIC_AUDIT.md:20-24`) | [M] | Rotation, boost routing and multi-second setup play cannot be credited **as sequences** by any reward term. **Caveat:** the existing audit shows the goal channel's authority is in `V_goal`, not in the terminal reward λ truncates (`D_G` 0.678 vs `A_goal` 0.680), so this may be a non-problem. That is what the sweep (§5 #2) is for. |
| **12.1d** | **Entropy floor.** `Policy Entropy` 0.617 normalized ⇒ ~e^2.78 ≈ **16 of 90 actions** effective support, flat since 1.2 B, at `entropyScale = 0.035` | [M, 6M] | The policy explored **18 % of its action table**. No reward term widens that; it is an entropy/LR lever, and effectively irreversible. **Re-measure on `resid`** — a 137 M-step policy has not yet collapsed its support, and this is the one number most likely to read differently. |

**Honest expected effect of the whole reward programme:** [E] **0 to +2 pp/B**. On the 6M run that
was set against a do-nothing null of +4.23 pp/B that cost nothing; on `resid` **the null is
unmeasured**, so the comparison cannot currently be made at all. **The strongest thing that can be
said for the programme is that S1 and S3 remove two verified defects — a positive-part-only boost
ratchet and a direction-blind touch reward — and that removing defects early, while a policy is
formative and the cost is one float, is worth doing even when the effect is not resolvable.** That
is a modest claim and it should stay modest.

### 12.2 Open questions, ranked

1. **The `resid` run has no baseline for anything in this document.** Every gate constant is [6M].
   S0 is the only way to fix this and it is the only fully blocking item. **This is question 1 and
   it did not exist yesterday.**
2. **The ~1.8 B discontinuity is unexplained** (§3.2). Every mechanical candidate is affirmatively
   excluded and the excursion is +3.70 sd. **Zero-risk next step:** log per-iteration
   `Nexto/Serve OppTeam` alongside per-burst goals — `oppTeam = RandInt(0,2)`
   (`Learner.cpp:3452`) is the second LCG draw off the same seed as the serve decision, and Nexto has
   a real orange-side observation asymmetry (`NextoOpponent.cpp:151-152`, `:195`). Simulation bounds
   its contribution at 1–2 pp, so it is unlikely to be *the* cause — but it is an **uninstrumented
   variance source in the one metric the whole programme is graded on**.
3. **`Average Step Reward` reads +0.2946 where the algebra demands exactly −0.0100** (§3.5), and
   ~31 % of iterations sit above +0.3. **[U].** **Resolve before any zero-sum tripwire is trusted as
   an actuator.**
4. **`League/Anchor Count` = 0 for the entire 3.4 B 6M run** [M, 6M] — the league anchor slice has
   never served, while `descendOpponentFrac = 0.35` (`src/ExampleMain.cpp:914`) still draws from the
   archive. **Anything anyone assumes about anchor exposure is not happening.** One-line fix, another
   confound. Check whether it reproduces on `resid`.
5. **The Nexto share is measured on the training reset mix, which is 10 % kickoffs** (§3.3). **It is
   not a match-play scoreline.** A fixed kickoff-only eval fleet against a frozen anchor is a
   prerequisite for any "will it beat the best bots" claim, needs **no reward change and no live
   risk**, and is cheaper than any stage in §6. **This should probably be item 0 of the next
   programme.**
6. **`Rating/1v1` overstates real progress ~6×** (memory `rating-pool-inflation-5.0v3`) and the two
   yardsticks **disagreed in sign** at 2.1–2.4 B on the 6M run (Rating 969.8 → 1095.3 while the Nexto
   share fell 40.07 → 35.29) [M, 6M]. Keep Nexto (burst-standardised) as primary. **This is also why
   `PHASE_B_RATING_TRIGGER = 1200` deserves re-derivation** (§0).
7. **The offline probe toolchain is 512-wide; the live trunk is 1152-wide and the policy head is now
   768.** `research/tools/load_checkpoint.py` hard-codes expected shapes and `dataset.npz` stores
   `(n, 512)` h1/h2, so every offline measurement needs a width parameter first (derive `trunk_w`
   from `head['0.weight'].shape[0]`), plus widened buffers in `collect_dataset.py` and
   `checkpoints_resid` added to `_default_root()`. **The residual blocks are a second, new problem**:
   the eager rebuild must reproduce skip connections or every tap is wrong. Also **4 columns are
   missing** from the `phys` block (`collect_dataset.py:283-289`) — `ballTouchedStep` and
   `worldContact.contactNormal.z` among them — which is why `ConsecutiveAirTouch` (0.18–0.49) and
   `WallJumpToBall` are the two widest cells in §3.4. **Budget this; it is not "run two scripts".**
8. **Five offline-replay rows in §3.4 are ±20–30 %.** `ConsecutiveAirTouchReward` is 12.3 % of credit
   density with a **2.7× error band**, and `CarEnergyPotentialReward` is 26.4 % measured on a
   5.0v3-lineage checkpoint (behavioural drift: mean ball z 454 vs 291.7, car speed 1217 vs 972).
   Closed by item 7.
9. **The code comments' claim that exact-PBRS terms are "UNFARMABLE BY CONSTRUCTION … the weight is a
   pure credit-density knob"** (`CommonRewards.h:713-739`, `src/ExampleMain.cpp:203-209`) is **false
   as implemented** — the unrefunded terminal bank scales **linearly with weight** (≥ 5.53
   units/episode at w = 30 for ConsecutiveAirTouch; 26.35 at w = 75 for B2G) [D]. Not a reason to
   change the reward today — **a precondition on any future weight increase** to the Φ(s₀) = 0 aerial
   potentials.
10. **PHASE B remains unvalidated.** `TEAM_SPIRIT` is an algebraic no-op in 1v1, and no run of this
    lineage has trained under 0.6. Any team-aware term's behaviour there is **[U]**.

### 12.3 If the programme fails

If the shipped stages all land cleanly and the burst-standardised Nexto share is still on its own
pre-programme trajectory, the correct conclusion is **not** "shape harder". It is that **the reward
stack was never the binding constraint**, and the next programme should be 12.1b (a replay-derived
or Frontier-selected state distribution) and 12.1a (the action table), in that order — both larger,
slower, and outside the scope of anything called reward shaping.

---

## 13. If you only do three things

1. **Re-trace Nexto's `emb_convertor` for CUDA.** Offline Python, **zero trainer contact**, no
   restart of its own. It recovered **18.6 % of wall clock** on the 6M run [M, 6M] and it **repairs
   the instrument every gate in this document depends on** — the 3.07× serve shortfall, the
   monotonic dose drift, and the burst clustering are all downstream of Nexto running on CPU. It is
   larger and more certain than any reward change here, and the new run needs the yardstick
   re-baselined from scratch anyway.

2. **Ship S0 — instrumentation only, no reward change.** The `WeightedReward::actScale` actuator
   (`Reward.h:62-75`, multiplied at `EnvSet.cpp:229`, written in the barrier at `Learner.cpp:3944`,
   **with the mandatory boot apply** because iteration 1 collects inline at `:3932-3936`), keyed by
   **index**; the serve-indexed CUSUM with **σ̂ and h calibrated live**; the six boost panels; the two
   S3 alignment panels; TEL-1; and the one-line `!steerRatingTripped` fix on the goal-critic blend
   (`Learner.cpp:4821` — **the largest of the four advantage injections and the only unguarded
   one**). This is now **doubly blocking**: the resid run has no baseline for anything, including its
   own do-nothing null. Adopt the shipping convention on day one — **never delete a reward line**; a
   deletion ships as `ship=0/safe=1`, a swap as an A/B pair with complementary scales. That single
   convention is what makes S1 and S3 revertible in-process, with no rebuild and no restart.

3. **Then S1 — mute `GuardedPickupBoostReward`, if and only if `Diag/Boost At Pickup` reads < 45.**
   One float. A verified positive-part-only duplication of the two-sided `sqrt(boost/100)` axis
   `CarEnergyPotentialReward` already owns at w = 15 — and **its own source comment
   (`CommonRewards.h:189-190`) argues for the overlap backwards** and should be replaced with a
   warning. The case is *stronger* now than when it was drafted: a 137 M-step policy has not yet
   built a style around the ratchet, and the whole reason this needed guards and staged windows was
   the risk of shocking a mature policy that no longer exists.

**And one thing not to do:** do not let the size of this document imply the size of the prize. The
programme moves Goal's real credit-density share by **under 1 pp**; four non-reward levers rank
above all of it; two of the three deferrals died when their own re-open conditions were finally run;
and on the run where it could be measured, **doing nothing for eleven hours beat most of the
programme**. Ship the correctness fixes because they are cheap, verified and now revertible in
process — not because they are expected to move the curve.
