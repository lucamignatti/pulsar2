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

### RC2 Stage O v2 — FAIL on live data (2026-07-16, ~34.9B); scalar mining RETIRED

4,100 iterations of live observation. Enrichment (picked ÷ base):
pre-landing **0.67×**, grounded-high-ball **0.68×** — BOTH de-enriched; fear-tail
Δz +0.41 (< +0.5). The acquisition statistic robustly AVOIDS mechanic-attempt
states, same as v1. Persistent robust signal across both statistics:
`Boost Mean ~10`, `Ball Z ~230` — the advantage/surprise frontier is low-boost
GROUND play, not mechanics.

**Decision (delegated by user): RC2 scalar reset-mining is DEAD; the actuator
(Stage A) will NOT be built.** General mechanistic reason: mechanic-acquisition
states are LOW-BASE-RATE with MODEST per-attempt advantage, so any scalar built
on advantage/surprise is dominated by high-variance ground scrambles. You
cannot LOCATE the acquisition frontier by mining outcome statistics — it is
defined by low SUCCESS, not high SURPRISE. (This is why FEAR_MINE works — it
mines the high-value DECLINE frontier — and this cannot.) The observer,
deployed before any actuator, caught this with zero resets drawn. The Miner
observer stays on as free low-cost telemetry; nothing reads its picks.

Implication for the program: don't try to FIND the frontier by mining — PRICE
optimism onto it by NOVELTY/DENSITY, which is orthogonal to advantage-surprise
(mechanic states are low-DENSITY even though low-ADVANTAGE). That is RC1, now
the load-bearing fix. Gated offline first (RC1_NOVELTY below) per the RC2
lesson.

## Overnight deploy record (2026-07-16, user-delegated)

- **RC1 LIVE at 35.1B** (2fbe806). Smoke caught + fixed a no-grad bug in the
  predictor pass. First live hour: RND/Loss 0.021 → 0.0006, novelty std
  annealing 0.0074 → 0.0002, injection ~15% of advantage scale, rating in-band,
  no trips. Persistence proven on the next restart ("RND self-model loaded").
- **Energy PBRS LIVE at 35.25B** (def0661, weight 6): CarEnergyPotentialReward,
  total mechanical energy as exact PBRS (loop-farm impossible by telescoping,
  ZeroSum-wrapped, gamma-matched, PE included so aerials untaxed; ball half =
  TouchAccel). Smoke clean; boot clean.
- **Trust steering: FAILED its bars, NOT deployed** (TRUST_PAIR.md): pairing
  +1.1pp (0.4σ) vs the 4pp/2σ bar; back-fill inverted — the axis produces
  upfield drift, not rotation. Dead-candidate list.
- Backups: 35025088660 (pre-RC1), 35250141800-adjacent (pre-energy), golden
  best_r1753. Morning watch list: RND/* (loss falling, injection stable),
  Rating vs the drawdown floor, Steer/Fear Panel zV, Census NONE 2v2, and the
  first pace census (energy term's proximate metric) after ~a day.

## RC1 offline novelty gate (before any live advantage change)

`rnd_novelty_probe.py`: RND (predictor trained toward a frozen random projection
of trunk⊕action over a rollout) → novelty = prediction error. Bar (frozen):
novelty ENRICHES (> 1.3×) on ≥2 of {pre-landing, grounded-high-ball,
proto-dribble} — i.e. the bonus RC1 would inject lands ON mechanic states, not
on ground scrambles. PASS → RC1 has a green light for a careful, latch-covered,
conservative-weight live deploy. FAIL → RC1's premise is also wrong; report and
rethink before any actuation.
