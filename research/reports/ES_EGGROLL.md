# ES_EGGROLL — low-rank Evolution Strategies vs PPO at fleet scale

**Status: PRE-REGISTERED 2026-08-16 (before any ES run). User-directed experiment.**

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

## Sweep plan

sigma_rel in {0.01, 0.03, 0.1} as three parallel 1-node jobs (alpha=1 fixed,
r=1, N=6/gen at 1 node), a few hours each; promote the best slope to the
4-node primary run. Population scaling (more members/rank via the batched
low-rank forward trick, the paper's real throughput engine) is v2 — only worth
building if v1 shows a pulse.
