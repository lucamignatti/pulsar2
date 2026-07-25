> **Status: SUPERSEDED — archived.** Part of the 4.0-lineage activation-steering
> campaign. Steering is numerically inert on HEAD (`steering.alpha = 0`); the
> Optimistic-Critic Ladder ([LADDER.md](../LADDER.md)) replaced it as the optimism
> mechanism. Kept as provenance for how the program was tested and why it was
> parked. **Do not cite this to justify a new change.**
>
> Offline numbers here predate the 2026-07-19 h2-truncation fix and were computed
> against **pre-activation `h2`** — see [H2_TRUNCATION.md](../H2_TRUNCATION.md).

---

# META frontier steering — prior-free phase 4, validation and build record

2026-07-14 (third program of the day), against pinned checkpoint **12425039422**.
Directive: the FULL meta steering system — the point of steering is attempts at the
ability frontier, generalized beyond the commitment frontier — under a HARD
constraint: **no human priors**. A first draft with a hand-designed task registry
(shot / aerial / clear slices, windows, outcome labels) was discarded for exactly
that reason. Scripts: `frontier_goals.py` (the system's python mirror),
`meta_frontier_validate.py` (M1 census / M2 sweeps). Raw tables: `results/meta_*`.

## The prior-free construction

Everything the system steers toward is agent-derived:

| element | source |
|---|---|
| goals | sampled from the agent's OWN achieved-state bank (both self-model goal spaces) |
| frontier | (state, goal) pairs its OWN self-model rates coin-flip (rho band, per-batch quantiles) |
| structure | emergent k-means clusters in its OWN psi-embedding geometry — never named |
| outcome | CONTINUOUS ATTAINMENT: −min goal-space distance of future achieved states within the head's OWN HER horizon (car 20 steps, ball 90) — model-free, threshold-free (all comparisons are within-population quantiles) |
| matching | the direction contrast is matched on rho itself (the confound IS "already likely") |
| head validity | a head drives steering ONLY while its own calibration curve is monotone — the system disables its own unreliable senses |

Remaining knobs are doses, not task content: band [0.2, 0.8], alpha, k=6 clusters,
dwell/exploration cadence, EMA decays. Two labeling designs were tried and
rejected on the way: psi-cosine achievement with consecutive-step calibration
(degenerate — the embedding is so smooth the threshold rounds to 1.0 and nothing
counts) and bank-NN-radius binary achievement (0.1% in-band positives — thin-pool
death, the E2/E4 lesson again). Continuous attainment has no threshold to fail.

## M1 — census (120k rows, 1v1 and 2v2)

- **Ball head calibration: MONOTONE** — median attain below/in/above band =
  −0.880 / −0.833 / −0.787 (1v1); also monotone in 2v2 (−0.974/−0.882/−0.750).
  The self-model genuinely ranks reachability for arbitrary sampled goals.
- **Car head calibration: NOT monotone** (−1.210/−1.117/−1.558) — unreliable for
  arbitrary cross-episode goals at its 0.67s horizon (it keeps its validated role:
  the incumbent contact-rho gate). The head-validity rule turns this finding into
  a permanent self-check instead of a hand-coded exclusion.
- Emergent ball clusters span the outcome space (own-half regions, attacking-half,
  a high-ball region at z≈800, near-kickoff rest states) with in-band n ≈ 400–1900
  and attain IQR 0.35–0.72 per cluster.

## M2 — causal sweeps (120k rows/alpha, paired seeds, cluster-goal rho gate)

| emergent cluster (descriptive only) | α=0 | α=0.5 | α=1.0 | verdict |
|---|---|---|---|---|
| deep-own-half HIGH ball (z≈728) | −1.002 | −0.946 | −0.901 | **monotone uplift**, canaries clean |
| own-corner ground ball | −0.870 | −0.890 | −0.797 | positive at 1.0 |
| mid-own-half slow ball | −0.936 | −0.963 | −0.899 | flat/noise |

Attainment of NON-steered clusters stayed flat-to-slightly-better in all sweeps;
touch/goals/kickoff canaries clean throughout. **Cluster heterogeneity is real —
the scheduling signal exists.** (Effects are ~0.1–0.15 IQR at offline n; the live
system re-measures with ~8x rows/iteration and EMAs over hundreds of iterations.)

## What shipped in the trainer (`cfg.steering.meta`, ON)

Per iteration in learn-prep (`fnMetaUpdate`), all sim-free (cheaper than the
incumbent's landing sims): achieved-goal vectors from the buffer's own obs → bank
(512/head) → psi k-means with slot-matched EMA'd centroids (cluster identity
persists across iterations) → mined cross-episode (state, goal) pairs (4000 rows ×
3 goals) → band quantiles → attainment scan → per-head calibration EMA + validity
→ per-cluster directions (role-0 pools only; matched attain terciles within rho
quintiles) → ACTIVE cluster's causal effect (steered-vs-control attain shift,
IQR-normalized, roles 1 vs 2) with bench/unbench thresholds → scheduler (dwell 10
iters; every 4th dwell explores the stalest cluster; otherwise argmax effect EMA;
unmeasured clusters probed first) → per-mode sigmas → barrier apply: the active
cluster's direction on all modes + its representative goal as the rho-gate target
(`PPOLearner::SetSteerGoal` override).

Guards: per-cluster benching (duty-cycle probing), head-validity self-disable,
the global rating latch unchanged, and `meta=false` = the pinned incumbent —
which is also the **pre-registered baseline: meta must beat pinned-commitment on
Elo slope over a matched window, else it reverts** (the roadmap's own criterion).
The incumbent derivation keeps running for its Steer/* panels either way.

Panels: `Meta/Calib Valid car|ball`, `Meta/Active Head|Cluster|Pairs`,
`Meta/Effect EMA`, `Meta/Attain Steered|Control`.

## Extension: the car-state head (third goal space, same day)

Canonical CAR pos+vel — movement/positioning capability, decoupled from the ball
(the achieved stream + HER machinery predate this from the proposer era; only the
psi head was untrained). Offline test (`carstate_head_validate.py`), deliberately
conservative — psi trained against the FROZEN phi, which was never given gradients
toward car-state information:

- **Detector gate: PASS, decisively.** Calibration monotone at every candidate
  window with margin ~0.65 — ~7x the ball head's. Window 45 chosen BY margin
  across {20, 45, 90}, not by hand.
- **Offline steerability: not demonstrated.** 0/3 sampled clusters show a clean
  monotone attainment uplift (best: +0.058 @ α=0.5, non-monotone). Note the
  structural pessimism: frozen phi (InfoNCE acc plateaued ~0.2 offline vs the
  live heads' 0.67–0.83), fresh directions, single seeds.

**Decision: shipped as a SENSE** (`reachability.carStateHead`, window 45): the
head trains its InfoNCE aux live (phi co-trains toward car-state discrimination),
initializes fresh on resume (`allowNotExist`), and enters the meta registry as
head index 2 — where the head-validity gate and per-cluster effect gates hold it
out of actuation until its LIVE calibration is monotone and clusters earn effect.
In this architecture steering rights are never granted by offline fiat anyway;
the offline detector pass is the entry criterion, and it passed. Watch:
`Meta/Calib Valid carstate` (expect 0 for a while — a fresh head over a mostly
ground-dwelling movement distribution needs training before its band means
anything), then whether the scheduler ever dwells there. Revert = flag false.

## INCIDENT 2026-07-14 — the first live deployment tanked Elo (~125, all modes)

Wandb (run q8kfp6q0): the deploy restart landed at step ~61280 with Rating/1v1
~1435 (recent peak 1462; the incumbent per-mode steering had just driven
1380 → 1462). Within ~500 iterations Rating slid to ~1336, **2v2 and 3v3 fell in
lockstep** (412→386, 246→227), KL/entropy normal — the shared-trunk-churn
signature. `Reach/Aux Loss` jumped from its **0.47–0.52** multi-thousand-iteration
baseline to **1.36 → 0.88 sustained (~2x)**. No guard fired: the slow-EMA rating
latch was still below the fresh peak (its structural blind spot), and no guard
watched the aux-gradient path at all.

Two mechanisms, both shipped in that restart:

1. **Carstate InfoNCE churned the trunk.** The fresh chance-level head's loss
   (~2x every other aux term combined) flowed through `sa → trunkOut →
   shared_head` from iteration 1. The policy played through a shifting
   representation; Elo fell in every mode at once.
2. **Meta took the steering slot permanently.** `fnApplySteering` delegated
   wholly to `fnApplyMeta`: the proven commitment direction stopped actuating
   and rotating unproven cluster directions replaced it, while benching was
   mathematically inert (150-iter per-cluster warmup ÷ 10-iter dwells, accrual
   only while active).

**Fixes (same day):**

- `reachability.carStateCouple = 0` (new): the carstate term trains **detached**
  (gradient confined to psi_carstate — exactly the offline-validated frozen-phi
  regime its detector gate passed in). Reported separately as `Reach/Car State
  Loss`; `Reach/Aux Loss` regains its 0.5 baseline = live verification.
- **Time-multiplexed slot ownership** (`metaProbeEvery=3`, `metaPromoteMin=0.05`,
  `metaWarmupIters` 150→30): the incumbent is the default actuator; every 3rd
  dwell probes one cluster (unmeasured first, then stalest — benched included,
  that re-probe is the unbench path); a cluster owns exploit dwells only after
  its measured effect EMA clears the promotion bar. Actuation is earned.
- **Measurement attribution follows the slot owner** (measured*/applied* state,
  one apply behind in pipelined mode): the incumbent possession gate no longer
  grades meta-steered buffers and cluster effects only accrue from buffers that
  cluster actually steered.
- **Peak-drawdown rating latch** (`ratingPeakTrip=110`, decaying high-water
  mark, persisted): catches exactly the slide-off-a-fresh-peak shape the slow
  EMA missed. Panel: `Steer/Rating Peak`.

**Recovery**: doctrine says restore a known-good FULL checkpoint. The golden
archive's top-3 (best_r ~1450+) all predate the restart; quarantine the
post-restart lineage (numbered checkpoints + policy versions newer than the
restore point auto-quarantine at boot).

## Honest limitations

- Offline effect sizes are small at this checkpoint (the 1v1 positive side has
  been saturated all day); the meta system's value case is (a) team modes, where
  the frontier census is far from saturated, and (b) tracking the frontier as it
  MOVES with training — neither measurable offline.
- Cluster count k=6 and the dwell/exploration cadence are structural knobs; the
  scheduler's own live criterion exists precisely to adjudicate them.
- The goal spaces are the self-model's existing two heads; extending the meta
  system to new goal spaces (e.g. canonical CAR state — the car-proposer head's
  space) is a natural next step and requires no new mechanism.
