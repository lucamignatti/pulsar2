# EMERGENCE — fix the learning system, let skills emerge (the root-cause program)

**Pre-registered 2026-07-15/16, authorized by the user** ("skills should emerge
naturally from a simple reward... fix the root cause... apply in the live
trainer, give it time, rollback to golden if needed"). Rollback anchor:
`build/checkpoints_4.0_branch_backup/32350074916` (+ golden archive best_r1677).

## Doctrine for this program

No skill is ever named in any mechanism. Every fix targets a measured pathology
of the LEARNER (MECHANICS.md elimination record). Deploy order = least
invasive first, observer before actuator, one lever per attribution window
(~1 day), full ritual each (compile-check tree, CPU smoke, branch backup,
panels, latch coverage, flag revert). The user accepts rollback risk.

## RC2 — learning-progress mining (data concentrates where learning happens)

*Pathology*: refinement starves at SNR × density (wavedash: 40% attempt rate,
+4σ value signal, 11% success flat for billions of steps). The match
distribution under-samples exactly the states where the gradient is alive.

*Fix, staged*:
- **Stage O (observer, deploys first)**: in learn-prep, mine top-|z-scored
  GAE-advantage| rows (both signs, per-episode capped) — the general
  "surprise" statistic; NO outcome definitions. Characterization panels
  (`Miner/*`) + a rolling 64-row obs sample in RUNNING_STATS for offline
  inspection. Actuates nothing.
- **Pre-registered rediscovery bars** (checked after ~1 day of observer data):
  the miner must rediscover UNPROMPTED ≥2 of the 3 hand-found state families —
  (a) pre-landing states (car airborne <300uu, descending) at ≥3× their
  base-rate share, (b) fear-tail overlap (mean Δz of banked declines > +0.5),
  (c) aerial-attempt states (grounded, high ball near). FAIL → the statistic
  is not general enough to trust with resets; iterate the statistic, do NOT
  actuate.
- **Stage A (actuator)**: only after bars pass — a modest practice-reset slice
  draws from the general pool (separate from FrontierPool; its own useFrac;
  staleness-bounded; latch-covered). Own mini-registration at flip time.

## RC3 — practice-value head (pricing isolation)

*Pathology*: the shared critic imports match-scale consequences into practice
contexts (CREDIT_PROBE: declines priced above pursuits; FEAR_MINE: +2σ
disagreement states are 53%-winnable). The stage-2 post-mortem already names
the sound fix.

*Fix*: a second value head on the trunk, selected per-row by the PRACTICE TAG
(never the obs — the tag is what kills aliasing) as the GAE baseline for
practice rows; each head trains only on its own rows. New panels: practice-vs-
match value on drill states = the contamination, measured live (also serves
RC1 diagnostics). *Risks*: GAE plumbing is collapse-class surgery — the two
historical stage-2 collapses were exactly here; extensive smoke + the rating
latch + anchor are the containment. Success signal (pre-registered): Fear
Panel zV closure rate accelerates vs its pre-deploy trend; no rating
drawdown beyond the noise band attributable to the deploy window.

## RC1 — frontier optimism (acquisition is not punished)

*Pathology*: below a success break-even, E[advantage|attempt] < 0 and PPO
actively unlearns infant skills (aerials at 0%; historically commitment). The
single realized-mean baseline prices uncertainty as failure.

*Fix*: novelty-priced optimism on the ADVANTAGE side — an RND head (small
predictor of a frozen random projection of trunk⊕action; error = novelty,
self-annealing with familiarity) contributing a small std-matched advantage
bonus. Explicitly NOT a reward term: the zero-sum stack invariant and
PSD/league fitness accounting stay untouched (precedent: the goal-critic β
blend is already a std-matched advantage adjustment). Knobs: weight (start
0.1), std-matching window, decay on RND learning rate. Success signal
(pre-registered): mechanic-fragment success rates (census re-run) move off
their plateaus within ~3-5 days; aerial conversion EMA > 0 sustained; Rating
within noise band. Deploys LAST, after RC3's isolation exists.

## What was deliberately NOT done

Per-skill drills/rewards (user doctrine: skills must emerge; the approved
aerial drill + height-scaled AerialTouch are SUPERSEDED by this program and
intentionally not built). Reward-stack changes (CREDIT_PROBE located the
pathology in pricing, not rewards).

## Rollback

Any deploy: flag off + restart reverts machinery. Lineage damage: restore the
anchor (32350074916) or golden best_r*, per the recovery doctrine (full
checkpoint or nothing).

## Results

### RC2 Stage O v1 (|z-advantage|) — FAIL, observer worked (2026-07-16, ~33.6B)

Miner enrichment ratios (picked share ÷ same-iteration base share), read off
`Miner/*` panels over ~800M steps of observation:

| family | picked | base | enrichment |
|---|---|---|---|
| pre-landing | 0.315 | 0.428 | **0.74× (de-enriched)** |
| grounded-high-ball | 0.018 | 0.021 | **0.85× (de-enriched)** |
| fear-tail (Δz mean) | −0.04 | — | no concentration |

**|advantage| magnitude = OUTCOME VARIANCE, not learning-progress.** It
concentrates on high-swing outcome states (goals, bounces) and actively AVOIDS
moderate-variance skill states — the same failure that retracted rare-event
replay, now measured directly. The observer-before-actuator staging caught it
with zero resets drawn. (Bar-spec erratum: the "≥3× base" pre-landing bar was
numerically impossible at a 43% base rate; enrichment-ratio > 1 is the correct
test and also fails.) Incidental win: `Miner/Boost Mean` ≈ 7 — even the wrong
statistic rediscovered the low-boost fear frontier (FEAR_DECOMP).

### RC2 Stage O v2 (positive-advantage × action-surprise) — deployed 2026-07-16

Frozen rule on v1 fail = iterate the statistic, do not actuate. v2 ranks by
`relu(z(advantage)) × clamp(−logProb, 0, 10)` — "an UNLIKELY action that PAID
OFF", the acquisition-frontier signal (targets skill DISCOVERY, not outcome
swing; cheap — logProbs already stored). Same rediscovery bars, now read as
enrichment ratio > 1 on ≥2 of {pre-landing, grounded-high-ball, proto-dribble}.
Still observer-only; still actuates nothing.
