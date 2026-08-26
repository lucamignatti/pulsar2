# LEAGUE_LORA — a league of LoRA variants with a discriminator diversity reward

**Status: RESULT (NEGATIVE ON DIVERSITY, POSITIVE ON MECHANISM) — 2026-08-26.**
One hypothesis (symmetry breaking) is specified and built but UNTESTED, blocked on an
AiMOS partition outage. Implementation: commits `1da004f`..`6775b29` on `private`,
all behind `GGL_LEAGUE=1` (default off).

## VERDICT

A league of rank-4 LoRA variants riding the live main policy trains **stably and
strongly** (0.42-0.47 goal share vs the main, criterion >= 0.40) but **does not become
behaviourally diverse**. Across FIVE configurations — league size 8 and 3, diversity
weight spanning 1% to 128% of the advantage scale, and two descriptor designs — the
discriminator's separability settled at **kappa 0.089-0.108** against a 0.286 target,
with no configuration distinguishable from any other. Exploiters never cleared parity
(0.40-0.53 vs a >0.55 criterion) despite a pure zero-sum objective.

The mechanism (batched multi-variant collection, adapters on a live base, per-variant
critics, exploiters, measurement harness) is built and validated. The diversity payload
is not delivered. Best current explanation, specified but not yet tested: **the variants
are initialised bit-identical (B=0), so initial diversity is exactly zero and the
discriminator-based reward has nothing to bootstrap from.**

## Question

Can a bot be given diverse-but-strong sparring partners cheaply, by training N low-rank
adapters on the live main policy, with distinctness supplied by a discriminator over
state pairs n seconds apart?

Design (as deployed):
- **Variants = rank-r LoRA adapters on `shared_head` + `policy`, riding the LIVE main.**
  No snapshot and no periodic re-basing: the adapter tracks the main's improvements for
  free, which is what makes variants "at the main's skill level" structurally rather
  than by scheduling.
- **Diversity reward** `r_div = log q(z | d_t, d_{t+n}) - log q(z | d_t)`, a variational
  estimate of `I(z ; s_{t+n} | s_t)`. The marginal subtraction is load-bearing: without
  it a variant is paid for merely *being* in an identifiable state (the DIAYN camping
  failure) rather than for making the *transition* identifiable.
- **Credited at the LATER endpoint.** A reward landing at `t` never enters the advantage
  of `a_t..a_{t+n}`, which are the actions that produced the transition.
- **Exploiters**: same adapters, no diversity term, plain zero-sum reward vs the live
  main = AlphaStar's main-exploiter dynamic for free.

## What had to be true before any of it measured anything

The single most valuable artifact was the **frozen control**: `GGL_LEAGUE_FRESH=1` +
LR/ENT/beta/SIL all zero, so `B` stays 0 and every variant is bit-identical to the main
at every instant (the adapter rides the live base, so it cannot fall behind either).
It measured **goal share 0.51 with discriminator accuracy AT CHANCE** — simultaneously
validating the play path, the goal accounting, and the discriminator's honesty (it
invents no signal from identical policies). It also **calibrated metric noise**: its
windowed goal share swung 0.25-0.67 while its true value was provably 0.5, which is how
we learned to read cumulative shares only.

**This control was built after four rounds of patching. It should have been first.**

## Results

Settled values at matched displacement (`||B||` ~ 1.8), diverse variants only:

| configuration | disc acc | chance | kappa | goal share |
|---|---|---|---|---|
| 8 variants, beta 0.1 | .194 | .125 | **.079** | .413 |
| 8 variants, beta 1.0 | .192-.207 | .125 | **.077-.094** | .465 |
| 8 variants, beta 3.0 | .209-.215 | .125 | **.096-.103** | .419 |
| 3 variants, beta 1.0 | .393-.405 | .333 | **.089-.108** | .448 |
| 3 variants, beta 1.0, window-aggregate descriptor | .397 | .333 | **.096** | .440 |
| frozen control (B=0) | at chance | — | **~0** | .511 |

Use **kappa = (acc - chance)/(1 - chance)**, not "x chance": the pre-registered
">= 3x chance" bar is only coherent at 8-way (3 x .125 = .375); at 3-way it demands
100% accuracy. Target kappa 0.286 is that bar restated.

### The unifying observation

**Kappa DECAYS as `||B||` grows, in every arm** (0.22 at `bPol` 0.33 -> 0.089 at 1.84).
Variants get *less* distinguishable the more they train, because the shared extrinsic
objective pulls all of them onto the same optimum and the diversity term has no signal
to oppose it.

The decisive evidence that the diversity term carried no signal (rather than too little
weight): at beta 3.0 the diversity reward was **118% of the advantage scale** — it
dominated the objective — and still produced no separation, while costing the worst
competence of any arm (0.419). That is what a large reward pointing nowhere looks like.

## Why each lever failed (all falsified, in order)

1. **Reward magnitude (beta).** 1% -> 128% of advantage scale moved kappa .079 -> .10.
   The reward is a discriminator LOG-RATIO; a discriminator stuck near chance emits a
   weak noisy log-ratio, so beta scales NOISE. The discriminator was not underfit —
   accuracy flat across whole runs on a full 65k reservoir, i.e. signal-limited.
2. **League size.** 8 -> 3 (chance .125 -> .333, ~2x rows/variant) changed nothing once
   settled. Its early kappa 0.202 was an artifact; it decayed to 0.089.
3. **Descriptor.** Replacing/augmenting instantaneous snapshots with window aggregates
   (mean boost, air fraction, speed, height, ball-relative offsets over `[t, t+lag]`)
   moved settled kappa 0.089 -> 0.096. The hypothesis — that snapshots are dominated by
   shared context, since ball and opponent state against the same opponent are set by
   the interaction rather than by variant identity — was reasonable and wrong.
4. **Rank.** Nearly spent a run on rank 4 -> 8 before measuring `||B||` and finding
   displacement was never the constraint (`B` climbs 0.23 -> 0.34 in 16 iterations, and
   to 2.5 over a run). Cancelled on the measurement.

## The untested hypothesis (built, blocked)

`B = 0` at init makes every variant **bit-identical at birth**, so initial diversity is
*exactly* zero: nothing to discriminate -> no signal -> the diversity gradient is noise
-> nothing separates -> still nothing to discriminate. The mechanism cannot bootstrap
out of perfect symmetry, and beta / league size / descriptor are all downstream of that.

`GGL_LEAGUE_BINIT` seeds each *diverse* variant's `B` with its own small noise
(exploiters stay at 0 — competitive objective, not identity-based). **Scale trap**:
`||B|| ~ sigma*sqrt(N)` over ~80k elements, so 1e-2 gives `||B||` 2.8 (past where
training ever reaches) while 1e-3 gives 0.32 (the regime where kappa was highest).

Two submissions (4647155, 4647161) died with `launch_failure_limit_exceeded_requeued_held`
during an el8 outage (36 of ~159 nodes down/drain/inval). The script never executed —
`diff` vs the working arm shows only the intended changes and `bash -n` passes — so the
hypothesis is genuinely unevaluated, not implicated.

## If symmetry breaking also fails

Then across six configurations a rank-4 adapter on a converged 442B-step policy produces
play that differs measurably in WEIGHTS but not in BEHAVIOUR, because the shared
extrinsic gradient dominates anything the adapter can express. The response would be
**architectural** — full policy copies (as the retired QD league used), or a rank high
enough to be a different model rather than a perturbation — not another sweep. Note the
cost model then changes completely: the entire economic argument for adapters was that N
variants cost ~one batched forward.

An independent finding worth keeping either way: **the exploiters never beat parity**
(0.40-0.53) despite a pure zero-sum objective and no diversity term. That is evidence
the GCO main at 442B has no *cheap* exploitable hole reachable by a low-rank delta.

## Method lessons (paid for)

- **Build the null-result control FIRST.** It validates the harness, proves the metric
  is honest, and calibrates its noise band, all at once.
- **Never infer a quantity you can cheaply measure.** Two panels (`||B||`, and
  `rdivStd` vs `advStd`) overturned a wrong lever choice that would have cost a run.
  `Adapter Norm` was useless for displacement — dominated by the random-init `A`.
  `r_div` is centred so its MEAN says nothing; only its spread vs the advantage spread
  is informative.
- **Cumulative shares mislead for a long time.** Early readings lied three separate
  times in one day, always optimistically (0.458 -> 0.413; a beta ordering that
  vanished; 3-way kappa 0.202 -> 0.089). Never conclude from a league arm under ~1h.
- **Compare arms matched on `||B||`, not wall-clock.** Different arms train at different
  rates; comparing a young arm's opening against an old arm's settled value is how the
  above errors happened.
- **A variant must inherit the main's WHOLE learning economy**, not just its objective:
  raw advantages (normalising them amplified pure value noise to unit scale and
  collapsed every variant within an hour), the main-critic baseline read-only, the
  advantage filter, and SIL. See `[[league-lora-variants]]` memory for the four-stage
  post-mortem.
- **`sbatch` copies the script at submission.** Editing an sbatch does not reach queued
  jobs, so with a long `afterany` chain every later env-var edit silently applies to
  nothing while code rebuilds DO land. Put tunables in code as defaults.
