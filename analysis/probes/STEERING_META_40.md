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
