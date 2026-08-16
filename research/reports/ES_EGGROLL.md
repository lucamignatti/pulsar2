# ES_EGGROLL — low-rank Evolution Strategies vs PPO at fleet scale

**Status: RESULT (CLEAN NEGATIVE) — 2026-08-16 noon. Pre-registered before any run.**

## VERDICT

EGGROLL as specified does not extract match-play fitness signal on this task at
paper-scale populations. Evidence across two fitness definitions:
- Shaped fitness, 176,640 members/gen, ~2,650 generations (~465M member-episodes,
  ~470B env steps): Update Norm pinned at the 1/sqrt(N) noise floor (0.0033) for
  the entire run; rating flat (0 +/- 1 Elo over 100+ evals).
- Goal-diff fitness, 39s windows, 900+ generations (~320M member-episodes):
  identical floor, identical flat rating.
- Dose-response sanity held throughout (sigma scales update norm and member
  fitness penalty as expected; member forward plays at mu level), so the
  machinery measured what it claims to measure.
Population-scaling baselines (6 and 1,536 members) were also flat, so the
negative is not population starvation inside the tested range 6 -> 1.8e5.
Mechanistic reading: a rank-1 weight perturbation at sigma achievable without
destroying play changes 13-39s match outcomes by less than the kickoff/bounce
lottery noise, and averaging 1.8e5 members per generation still leaves
signal-to-noise below extraction threshold. PPO's per-step credit assignment is
doing load-bearing work that episode-level ES cannot replace on this task at
this compute scale. The forward-only/fleet-linear infrastructure thesis was
CONFIRMED (23.5M SPS on 690 GPUs, 2us consume) — the learning thesis was not.

## Question

Does EGGROLL-style low-rank ES (arXiv 2511.16652) train a Rocket League policy
better than PPO **per unit wall-clock at fleet scale**? The paper's pitch:
forward-only, no optimizer, no critic, population parallelism that scales
linearly with workers because members are reconstructed from counter-based RNG
seeds and only fitness scalars are communicated.

Why our fleet is the right test bed: the 2026-08-15 cadence work established
that (a) this ppc64le platform is CPU-dispatch-bound (~0.15-0.5ms/tensor-op),
so deleting the consume phase is worth ~40% of the wall outright, and (b) PPO's
per-iteration trust region makes ranks beyond ~4 nodes worthless (they only
inflate the batch). ES has no such ceiling: N ranks = N members/generation.

Honest priors AGAINST: the paper's tabula-rasa RL used 256-wide 3-layer nets
(~100x smaller than our 34M policy); adversarial self-play fitness has
episode-cluster variance ~5x binomial (measured, CLAUDE.md); PPO here is a
strong, tuned baseline. This experiment can fail and that is a result.

## Method (v1, deliberately minimal)

- Mode `GGL_ES=1` inside the existing trainer. Default off; PPO path untouched.
- Per generation (= one collect iteration): each rank perturbs its **worker
  snapshot** of shared_head+policy 2D weights with rank-1 noise
  `W += sigma_l * (1/sqrt(r)) A B^T`, `sigma_l = GGL_ES_SIGMA * std(W_l)`,
  A/B drawn on CPU from a generator seeded `hash(generation, rank)` —
  reconstructible by every rank, zero perturbation communication.
- mu (the main ModelSet) is NEVER perturbed: saves, eval, boot probe, viz all
  see the clean policy at all times.
- Opponent every iteration = frozen copy of current mu via the existing
  opponent-split path (oppModels). Members play mu; fitness = mean per-player
  total shaped return of the member's rows in the iteration's trajectory
  buffer (the zero-sum stack makes unperturbed-vs-mu fitness 0-mean by
  construction).
- Update at the barrier: allreduce the N fitness scalars, z-score them
  (guard sigma_f ~ 0), reconstruct all N perturbations from seeds, apply
  `W += (GGL_ES_ALPHA / N) * sum_i f_i * sigma_l * (1/sqrt(r)) A_i B_i^T`
  identically on every rank (one [m,N]x[N,n] GEMM per param via the stacked
  trick). Existing lockstep checksum verifies rank agreement for free.
- All PPO processing (GAE, value pred, Learn, aux) skipped in ES mode.
  Pipeline off (no consume to hide collect under).
- LN/bias/1-D params unperturbed in v1. Reachability/vdag/critic heads inert.

## Pre-registered success criteria

Baseline: the live 7.1-cadence PPO run (checkpoints_cadence_luca), which at
matched wall-clock T hours from ITS cold start provides Nexto goal share and
kickoff-probe numbers.

1. **Sanity (24h)**: fitness slope > 0 over the first 24h AND kickoff probe
   reaches >= 5/10 touches (PPO reached competence well within this window).
   Fail -> one sigma/alpha resweep (parallel jobs, idle nodes), then verdict.
2. **Primary (72h, only if sanity passes)**: Nexto goal share and anchor Elo
   at matched wall-clock >= the PPO run's trajectory at the same age, using
   the same eval (GGL_NEXTO_EVAL / match_play_eval vs fixed anchors), same
   node count (4). Scale-up to >16 nodes only AFTER passing at 4 — the paper's
   scaling claim is tested second, competence first.
3. **Kill criteria**: fitness slope <= 0 for 12h after resweep, or NaN/lockstep
   divergence, or entropy collapse of mu (action distribution degenerate).

## Revert path

Separate lineage (checkpoints_es_*), separate jobs on idle nodes; the PPO run
is never touched. Killing the experiment = scancel. The code is env-gated and
default-off; the shared binary keeps byte-identical PPO behavior with GGL_ES
unset (compile-checked, and the live chain's hop restarts exercise it).

## AMENDMENT 2026-08-16 (~1h into the sweep, BEFORE any verdict)

Observed: all three sigma jobs mechanically clean (1,300+ generations each, zero
errors, update norms scale with sigma, member fitness slightly negative = perturbations
behaviorally meaningful) but Rating/1v1 flat after ~800M steps each. Identified
confound, declared before the 24h gate matures: **population starvation** — 6
members/generation is far below anything in the ES literature for policies this size,
so a small-scale sanity failure would not distinguish "EGGROLL doesn't work here" from
"population too small". Amendment: add a population axis — one 24-node job (144
members/gen, sigma=0.03) launched alongside the unchanged 1-node sweep. The sanity
gate (fitness/rating slope, kickoff competence, 24h) applies to the LARGEST population
tested; the 1-node jobs become the population-scaling baseline rather than the gate.
Also added: an ES-update lockstep checksum (the PPO one lives in Learn(), which ES
skips) — divergent ranks would otherwise silently corrupt the population.

## AMENDMENT 2 2026-08-16 — paper-scale populations (user-directed)

The paper's scaling figure separates from backprop only at populations 10^4-10^6;
one-member-per-rank caps at the rank count (6-690), i.e. the figure's lightest
curves. v2 (`GGL_ES_PER_ARENA`, default on, commit 7181597) makes every ARENA a
member via the batched low-rank forward (InferActionsLowRankES — the paper's
"91% of batch inference" trick): population = arenas x ranks. The primary run is
now 115 nodes x 690 ranks x 256 arenas = **176,640 members/generation** (10^5.2,
inside the paper's regime). The v1 jobs' flat ratings are recorded as the
population-scaling baseline, not a verdict on the method. Update math v2: local
per-layer GEMM over each rank's members, dW allreduce-summed — no cross-rank
noise determinism needed; lockstep checksum retained.

## AMENDMENT 3 2026-08-16 evening — overnight decision rule (pre-registered)

Paper-scale run (176,640 members) at gen ~620: Update Norm pinned at the 1/sqrt(N)
noise floor (0.0033), fitness mean stable, rating flat — machinery verified, no
extracted signal. Deadline: verdict by noon 2026-08-17. Automated decision at
generation >= 14,000 (~12h): KEEP the current config iff last-200 Update Norm >
0.005 (1.5x floor) OR rating > +30; otherwise the remaining hops are replaced by
the strongest single fitness lever: GGL_ES_FIT_GOAL=1 (goal-diff-only fitness —
the window-mean shaped fitness telescopes its PBRS terms to endpoint lottery) with
GGL_TS_PER_ITR=300000 (39s windows), same mu lineage. If the goal-fitness hops are
also floored by noon, the verdict is a CLEAN NEGATIVE at paper scale: EGGROLL as
specified does not extract match-play fitness signal on this task at 10^5 members,
across two fitness definitions and sigma dose-response evidence.

## Sweep plan

sigma_rel in {0.01, 0.03, 0.1} as three parallel 1-node jobs (alpha=1 fixed,
r=1, N=6/gen at 1 node), a few hours each; promote the best slope to the
4-node primary run. Population scaling (more members/rank via the batched
low-rank forward trick, the paper's real throughput engine) is v2 — only worth
building if v1 shows a pulse.
