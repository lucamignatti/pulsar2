# LEAGUE_LORA — a league of LoRA variants with a discriminator diversity reward

**Status: RESULT (SOLVED) — 2026-08-26.** Implementation `1da004f`..`5c4793f` on
`private`, behind `GGL_LEAGUE=1` (default off).

## VERDICT (FINAL)

**Works.** Verified on the CONVERGED 442B GCO policy (548B viz mirror, local GPU) with
the shipped defaults and no league tuning env vars. Settled equilibrium:

| criterion | bar | measured |
|---|---|---|
| variants distinguishable (kappa) | >= 0.286 | **0.342** |
| diverse variants competitive (windowed goal share) | >= 0.40 | **0.427** |
| exploiters beat the main | > 0.55 | **0.561** |

Two things got there, after six configurations that did not:

1. **The diversity signal had to CREATE difference, not merely detect it.** A
   Variant-KL panel showed variant-to-base KL at ~0.05 nats on a policy carrying ~3.4
   nats of entropy — variants were playing almost identically, so every knob on the
   *discriminator's* evidence was amplifying a difference that did not exist. Pairwise
   repulsion (all variants evaluated on the SAME states, paid while their mean pairwise
   KL is under target) makes them differ directly, with no bootstrap.
2. **The league LR had to drop an order of magnitude below the main's** (1.5e-4 ->
   2e-5). Most competence loss under repulsion was never repulsion; it was PPO damage.
   A variant trains on a fraction of the rows the main gets, so at the main's LR its
   updates are mostly noise. Repulsion is a hinge with its own gradient, so a low LR
   only slows its approach to target and then holds. Measured: wDiv .39-.42 -> .49 with
   kappa UP, and exploiters .27 -> .63. The exploiters are the clean tell — they get no
   repulsion and no KL floor, so their entire gain is the removal of noise damage.

The operating window is narrow and the frontier steep: repelTarget 0.15 leaves variants
indistinguishable (kappa .24), 0.5+ destroys them (wDiv .04 at repel 1.78). 0.28 with a
tight collar (max 0.35) is the measured point.

## DOES IT MAKE THE MAIN BETTER? (A/B, 2026-08-27)

Both arms start from the SAME converged 563B checkpoint, run identical configs for equal
wall clock, and are scored on goal share against the SAME frozen copy of that start
(`policy_versions/ref_563659776000`). That anchor is the only yardstick CLAUDE.md trusts —
`Rating/1v1` inflates because its pool tracks the agent, and **Nexto is unusable here: the
policy beats it 40-0, a total ceiling**. The comparison is honest about cost: with the
league on the main gets only ~75% of the arenas.

| main LR | control (self-play) | league | |
|---|---|---|---|
| 1.5e-4 (production) | **0.190** (203-864) | **0.291** (256-624) | league clearly better |
| 1.5e-5 (1/10) | 0.491 (479-497) | 0.491 (135-140) | neither moves; no harm |

**At production LR the league-trained main is substantially stronger than the
self-play-trained main** — a 10-point gap on ~1000 goals per arm, z ~ 2.3 even after
inflating the SE 5x for episode-cluster variance, and it wins while training on a quarter
less data.

**Caveat, stated plainly:** both arms scored BELOW 0.5, i.e. both degraded and the league
merely degraded far less. That is a harness artifact, not the league — resuming without
optimizer state restarts Adam cold, and at production LR its first updates wreck a mature
policy. Confirmed by dropping the LR 10x, which restored the control to ~0.49 (stable).
So the measured claim is **the league makes training markedly more robust**, which is what
diverse opponents should do, and it is a real difference between two policies trained
identically apart from the league. The low-LR arm is a null control only: 25 minutes at
1/10 LR cannot move a 563B policy, and both arms sit at 0.491.

Loading the real optimizer state locally FAILS (shape mismatch 256 vs 1280 — those tensors
carry the cluster's net config, not the local build's), so the clean version of this
experiment wants a cluster run where training resumes genuinely. `GGL_MAIN_LR` exists for
offline A/Bs of this shape and must never be set in production.

## VERDICT (superseded, kept for the reasoning trail)

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

## THE ROOT CAUSE, and the mechanism that addresses it (2026-08-26, `d0e7d71`)

Adding a **Variant-KL** panel — mean KL(variant || base) over real rows — produced the
number that reframed the whole study: **~0.05 nats**, on a 90-action policy carrying
~3.4 nats of entropy. The variants were playing *almost identically* to the main.

That is why all six levers were inert. Beta, league size, descriptor, symmetry breaking
and rank were every one of them trying to amplify *evidence* of a difference that did
not behaviourally exist. The discriminator was not failing; it was reporting the truth.
**||B|| (parameter distance) was the wrong instrument all along** — post-LN blocks can
make large parameter movement behaviourally nearly free, so an adapter can travel a long
way in weight space and barely change the policy.

Two direct, dense, differentiable terms now CREATE the difference, leaving the
discriminator only to MEASURE it:

- **`repelCoeff`** — evaluate every diverse variant on the SAME sampled states (one extra
  batched per-row LoRA forward) and pay while their mean pairwise KL is below target.
  This makes variants differ from EACH OTHER, and needs no bootstrap: unlike the
  discriminator reward, it does not require variants to already be different in order to
  make them different.
- **`klCoeff`** — a hinge floor on KL(variant || base), applied to EXPLOITERS TOO. An
  exploiter at KL 0.05 effectively IS the main, and a mirror match is 0.5 by
  construction — which is exactly where exploiter goal share sat (0.40-0.53) in every
  arm. The hinge is a floor on deviation, not a direction; the zero-sum objective still
  chooses where that deviation goes.

Both are two-sided bands (push below target, pull back above max), so the objective is
"be different but remain a policy" rather than a race away from each other.

**Measured locally (CPU smoke, corrected units):**

| repelCoeff | KL(var‖base) | mean pairwise | entropy |
|---|---|---|---|
| 0 | .044 -> .062 | .038 -> .097 | .763 |
| 0.5 | .110 -> **.244** | .061 -> **.471** | .736 |
| 2.0 | .119 -> **.242** | .065 -> **.422** | .736 |

Separation reaches the target and HOLDS there without overshoot; 0.5 and 2.0 converge to
the same place, which is the saturation the hinge is designed to produce. Entropy cost
~0.03. Combined repel+KL run: KL .355, pairwise .384, no crashes.

**STILL UNVALIDATED — the competence question.** Every number above comes from a smoke
whose base policy is randomly initialised. Whether forcing KL ~0.3-0.5 on a CONVERGED
442B policy preserves goal share is the one thing that decides whether this works, and it
has not been run. It needs either the cluster or a local run against a full checkpoint.

**Two instrumentation bugs found, each of which had already produced a wrong conclusion:**
- The repel panel divided by `V*(V-1)` while summing over `V*V*S`, under-dividing by
  S=128. A healthy 0.29 nats read as a 37-nat runaway and prompted a "fix" for a problem
  that did not exist.
- The `postLN` hypothesis (delta after LayerNorm rather than at the Linear) measured
  NEUTRAL: KL .045-.079 vs .032-.072. Falsified; the option defaults off.

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

## Symmetry breaking — TESTED, ALSO NEGATIVE (job 4647161)

`GGL_LEAGUE_BINIT=1e-3`, 3 variants, beta 1.0. Birth displacement verified numerically
(first-update `||B||` 0.282 vs 0.053 for a B=0 arm) rather than from the log — the
`GGL_LEAGUE_FRESH` banner claimed "B=0" unconditionally and could not distinguish the
two configurations (fixed in `8753534`).

Settled at `||B||` ~1.9-2.0: **kappa 0.105-0.113, goal share 0.454**. Nominally the best
of the six arms and still 2.5x short of target, inside the spread of everything else.
Trajectory 0.202 (bPol .49) -> 0.165 (.84) -> 0.111 (1.44) -> ~0.11 settled: the same
decay curve as every other arm, and BELOW baseline at matched displacement.

**The result that matters most here is incidental**: `||B||` = 0.28 of *random*
perturbation produced no more discriminability than a *trained* variant of the same
magnitude. Random and learned low-rank deltas are equally invisible to the
discriminator. That argues the limit is **the low-rank family itself**, not what the
diversity objective does inside it — and it undercuts the bootstrap story this arm was
built on (the variants were never stuck for lack of an initial difference).

## The original framing of that hypothesis (kept for the reasoning trail)

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
