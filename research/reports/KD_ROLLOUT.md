# KD_ROLLOUT — a rollout-only knowing–doing gap, and what to do with it

> **Status: RESULT (NEGATIVE) 2026-08-16 — Q1 FAILED its pre-registered gate on
> both snapshots; Q2/Q3 are NOT licensed. No machinery was built.** The
> pre-registration below is unchanged; results at the bottom.
>
> **Original status: PRE-REGISTERED 2026-08-16 (before any measurement ran).**
> Target lineage: the AiMOS fleet run (wandb `7.0b-aimos`, dense full-size ts8
> arch), measured offline on desktop CPU against the local mirror
> `build/checkpoints_aimos/`. Interventions (Q2/Q3), if licensed, run on AiMOS
> as separate small jobs — never on the live fleet.

## The three questions (user-posed, 2026-08-16)

1. **Q1** — can the knowing–doing gap of [KNOWING_DOING.md](KNOWING_DOING.md)
   be recovered *more elegantly and generally from rollouts alone* — no ridge
   probes, no landing labels, no offline supervision?
2. **Q2** — if yes, can that statistic be used as an intrinsic reward pushing
   *doing → knowing*?
3. **Q3** — can it replace advantage filtering in SIL — i.e. filter out
   conversions on behavior the bot already knows *and does*, keeping only
   frontier conversions?

## The statistic

The reachability module already trains, from rollouts alone (HER + InfoNCE), a
self-model whose defining property is **capability, not policy**: the published
read `rho(s→g)` averages over *uniform* valid actions ("can we"), deliberately
not policy-sampled ("would we"). The knowing–doing gap is exactly the
difference between those two readings of the same embedding space:

```
c(s,a,g)   = cos( φ(h2(s), a), ψ_car(g) ) / τ            τ = 0.02 (EvalRho parity)
ρ_can(s|G) = max_{a ∈ valid(s)}  score(s,a,G)             "an action exists"
ρ_do(s|G)  = Σ_a π(a|s) · score(s,a,G)                    "the policy takes it"

KD(s) = ρ_can(s|G) − ρ_do(s|G)   ≥ 0
```

where `G` is a **witnessed touch-goal bank**: M car-local ball-relative configs
(`achievedCarBall` encoding, 6-dim, /2300) harvested at actual ball-touch
moments of the *same rollout batch* — goals are configurations the policy has
demonstrably achieved somewhere, in the composition-critic spirit of witnessed
pieces. `score(s,a,G)` is `max_g c(s,a,g)` (primary — "some witnessed way of
having the ball") with `mean_g` as secondary.

- "Knowing": ρ_can high — the self-model asserts an action heading to
  possession exists from here, within its HER window (carHerMaxOffset = 10
  steps ≈ 0.67 s at 15 Hz).
- "Doing": ρ_do — the identical read weighted by the policy's actual action
  distribution.
- KD high ⇔ *can, but won't*. Zero on mastered behavior (does what it knows)
  and on genuinely unreachable states (knows it can't). Per-frame, per-goal-
  family, computable in-trainer at learn-prep cost ~ one φ pass per (row ×
  valid action) on a subsample. No probes, no labels, no sims.

This is the action-level complement of the return-level headroom
`H = relu(min(V†₁,V†₂) − V_real)` already live: H says *this state is worth
more than I realize*; KD says *this state contains a known-reachable outcome my
action distribution avoids*. S3 below tests that they are not the same signal.

## Q1 ground-truth protocol

`kd_curve.py`'s attendance protocol, ported to the 7.0 arch: self-play rollouts
(offline harness, ts8/aD0 parity, live-style reset mix), ballistic landing
labels (`simulate_landing`), feasibility = straight-line arrival budget
< 1300 uu/s, attendance = car within 500 uu of the landing point at touchdown,
free = ball lands uncontested (ball within 300 uu of predicted landing, z<200).

Primary slice: readings with `t_land ≤ 1.33 s` (2–20 steps — the horizon where
the car head's HER window makes "knowing" well-posed). Full-range secondary.

The 2×2 prediction (per-cell mean KD):

| | attended | unattended |
|---|---|---|
| **feasible** | low (does what it knows) | **HIGH — the frontier** |
| **infeasible** | (rare, noise) | low (knows it can't) |

## Pre-registered success criteria (Q1 verdict)

- **S1 (knowing content — the anti-tautology control).** AUC(KD :
  feasible-unattended vs infeasible-unattended) **≥ 0.70**, AND this AUC
  exceeds the same AUC for `−ρ_do` alone by **≥ 0.05**. Rationale: "policy
  isn't heading there" (−ρ_do) is high for *both* cells — near-tautological
  with unattendance; only the capability term ρ_can can separate can-but-won't
  from can't. If −ρ_do alone matches KD, the statistic is inverted propensity
  in disguise → FAIL.
- **S2 (doing content).** AUC(KD : feasible-unattended vs feasible-attended)
  **≥ 0.60**.
- **S3 (non-redundancy with live headroom).** |Spearman(KD, H)| **< 0.5** on
  the airborne analysis slice, H read from the same checkpoint's V†/V heads
  (V = mean of twin critics). If ≥ 0.5, KD is H in disguise; Q2/Q3 would add
  nothing over the existing headroom-gated machinery → FAIL-redundant.
- Both sampled snapshots (93B, 529B) must individually pass S1/S2; S3 on both.

PASS = S1 ∧ S2 ∧ S3 → Q2/Q3 arms licensed. Anything else: record, stop, no
machinery (the 9uz761ua doctrine).

Controls reported alongside (not gates): AUCs for ρ_can alone, −ρ_do alone,
distance-to-landing, t_land; bank-mean vs bank-max variants; the per-cell 2×2
table; KD of the *taken* action's stored φ (needed later by Q3's filter).

## Estimators (fixed before the run)

Exact enumeration over all 90 actions masked to valid (no K-sample MC): ρ_do is
the full π-weighted sum, ρ_can the exact max. Bank M = 64 touch configs sampled
from distinct episodes (touched-flag frames, |rel_p| sanity < 300 uu). Analysis
anchors: up to 12k airborne-ball frames (ball z > 300) per snapshot from ~120k
collected player-frames, 16 arenas, seeded. Two snapshots: oldest (93B) and
newest (529B) in the mirror. AUCs are Mann-Whitney; cells with n < 50 void the
criterion touching them (re-run bigger before verdict).

## Q2 design (CONTINGENT on Q1 PASS — not yet licensed)

KD as *doing→knowing* pressure, in order of least machinery, one lever at a
time, each as a separate AiMOS arm vs a config-identical control:

- **Q2a — KD-gated entropy** (first): extend the existing `vdagEntGate`
  channel with KD in place of / mixed with H — per-row entropy multiplier
  `1 + k·KD/mean(KD)`, clamped as today. Only ever raises entropy (the
  measured-safe channel class); directly makes π sample the known-but-untaken
  actions where they exist. No new loss, no reward change.
- **Q2b — seek potential** Φ = **+KD** (σ-matched, clamped, PBRS form). Sign
  is load-bearing per the composition-critic record: closure (−KD) pays the
  policy to *destroy its own ρ_can* — retreat to states where it can't do
  anything — the same failure the −H form showed. Seek attracts to the
  frontier; conversion is then SIL/entropy's job.

## Q3 design (CONTINGENT on Q1 PASS)

Current SIL row filter (Learner.cpp): `(R > V_exp) & (hMix ≥ q_{0.70})`,
weight `(R − V)+` capped — return-level luck-vs-skill filtering. KD variant:

```
conv_KD = (R > V_exp) & (KD ≥ q_{0.70}(KD)) & (c(s, a_taken, G) ≥ median ρ_can slice)
```

— imitate only rows where the bot *did the thing it knows but rarely does*:
high-KD state, and the executed action was itself a capability action. Mastered
behavior is excluded by construction (KD ≈ 0 there), which is precisely
"filter stuff the bot knows [and already does]". Run as arm vs the hMix gate at
matched silCoeff; judge on ignition/conduct metrics + Nexto/Ref shares.

## Caveats pre-declared

- Horizon mismatch: ρ speaks ~0.67 s ahead; attendance decisions span ~3 s.
  The primary slice restriction absorbs this; a full-range miss does not fail
  Q1.
- ρ is a coarse compass (calibration study: bin it, never trust a single
  value); all criteria are distributional (AUC), never per-frame.
- One goal family (ball possession via ψ_car). Generality to ψ_ball (shots)
  and ψ_carstate (poses) is claimed by construction but only measured for
  possession here.
- Offline reset mix approximates the live trainer mix (5.3-era port); the 2×2
  test is within-slice, so mix drift shifts cell sizes, not the comparison.
- The touch bank is policy-conditional (touches the current policy achieves);
  a policy that never touches would have an empty bank. Not a concern at
  93B+ steps (touch rates are healthy); noted for cold-start use.

## Results (2026-08-16, `research/tools/kd_rollout.py`, seed 20260816)

120k player-frames / 12k anchors / 180 episodes per snapshot; full JSON in
`research/results/kd_rollout_<ts>.json`.

| criterion | required | 93B | 529B | verdict |
|---|---|---|---|---|
| S1 AUC(KD; feas-unatt vs infeas-unatt), primary | ≥ 0.70 | 0.555 | 0.490 | **FAIL** |
| S1 margin over −ρ_do | ≥ +0.05 | +0.14 | −0.05 | FAIL (529B) |
| S2 AUC(KD; feas-unatt vs feas-att), primary | ≥ 0.60 | 0.446 | 0.425 | **FAIL (inverted)** |
| S3 \|spearman(KD, H)\| | < 0.5 | 0.051 | 0.013 | pass (vacuous) |

KD reads chance-level everywhere; it is *higher* on attended than unattended
landings (cell means 0.84 vs 0.70 at 93B, 0.75 vs 0.61 at 529B) — the opposite
of the frontier prediction. The controls behaved exactly as pre-declared:
−ρ_do alone is near-tautological for attendance (S2 0.77/0.83) and ρ_can alone
is a strong *state* signal (S2 0.17/0.23 — attended states read reachable).

**Why it failed (diagnostic, 529B):** the φ(s,a) embedding is ~10× more
state-driven than action-driven — within-state action spread of the bank-max
cosine is 0.30 τ-units (p90 0.61) vs 2.96 τ-units between states; in embedding
space, mean distance to the per-state action-centroid is 0.074 vs 0.78 across
states. ρ_can − ρ_do is a one-step action counterfactual, and at 15 Hz a
single 8-tick action almost never flips will-I-have-the-ball-in-0.67s — so HER
gives φ's action channel almost no gradient, the c(s,a) surface is flat in a,
and KD is noise riding on ~2 % of the read's variance. This is structural
(physics + horizon), not a sampling artifact: both snapshots, both slices,
bank-max and bank-mean all agree.

**What this licenses: nothing.** Per doctrine the Q2/Q3 arms are dead as
specified. Any revival needs a *multi-step* capability object on the knowing
side — either the return-level one that already exists (H = V†−V_real, whose
attendance S2 0.54–0.59 is weak but right-signed, and which SIL already
actuates), or a goal-conditioned/quasimetric read (the machinery class that was
deliberately removed 2026-07-25; reintroducing it needs its own conviction
first). A one-step action contrast on the reach head is not it.

**Replication bonus (behavioral side, load-bearing for the program):** the
knowing–doing *phenomenon* itself replicated on the AiMOS lineage and is not
closing with scale. Four-point curve (primary slice, unattendance among
feasible-free landings; 180–226 episodes per point):

| steps | 93B | 401B | 454B | 529B |
|---|---|---|---|---|
| unattendance | 54.9 % (n=1149) | 51.6 % (n=728) | 61.8 % (n=1048) | 68.2 % (n=984) |
| overall touch rate | — | 3.74 % | 2.72 % | 2.02 % |

Flat ~52–55 % through 401B, then rising sharply in the last ~130B steps
*together with a falling touch rate*. **Confound before reading that slope as
organic gap growth:** the fleet's ExampleMain now has AerialTouch at weight
**15** (annealed from the 120-era scaffold after the juggle-farm regression);
the anneal landed mid-window, so reduced ball-pursuit in the 454B/529B points
is at least partly the *intended* incentive change. The defensible claims are:
(1) the gap is large and persistent at every point — it does not close on its
own under headroom-gated SIL at fleet scale; (2) the recent slope is
reward-schedule-confounded and should not be quoted as degradation. The
protocol is cheap (~3 min/snapshot) — rerun it across future snapshots at
fixed reward config before trusting any trend.
