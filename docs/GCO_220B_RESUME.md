# GCO "220b" resume on `22b-compat` — 16 nodes, launched 2026-09-01 night

*Status doc for the morning check-in. Companion to `docs/GCO_RESUME.md` (the earlier
VTS/league plan, NOT what is running) and the bundle's `GCO_CONFIG.md` (the contract).*

## What is running

| | |
|---|---|
| Seed | `~/Desktop/pulsar2-gco-FULL-235.7B/checkpoint/235726848000` — the **only** FULL (resumable) GCO bundle on the Desktop; the `pulsar2-gco-220.8B` folder is policy-only and cannot be resumed (recovery doctrine: full checkpoint or nothing). Cluster copy `barn-shared/pulsar2-gco/RESTORE_ts8_235726848000` is md5-identical to the Desktop manifest; it was copied to the run folder, the RESTORE dir stays as the pinned restore point. |
| Branch | `22b-compat` = `private` HEAD `133169b` + 2 commits (below). **No VTS, no league, no titan.** |
| Cluster tree | `~/scratch/pulsar2-private-luca` (branch `22b-compat`, own `build/`). The `pulsar2-gco` tree stays on the VTS branch, untouched. |
| Run folder | `~/scratch-shared/checkpoints_gco_220b` (keep 5, save every 500 iters); policy milestones in `checkpoints_gco_220b_archive/<ts>` once per hop |
| Logs | `~/scratch-shared/logs/gco_220b_<jobid>.out` |
| wandb | run `7.9-gco-220b-16n`, group `gco-220b`, online via the proxy (rank 0) |
| sbatch | `~/scratch/pulsar2-private-luca/pulsar2_gco_220b.sbatch`, 6h hops chained with `afterany`, partitions `el8,dcs-2024`, elastic 16–24 nodes, preflight + banned list, NEED=16 (degrades to 12 with the same effective batch) |

## The two commits on top of `private`

1. **Optimizer-state shape guard** (`Models.cpp` `Load`). The checkpoint's `*_OPTIM.lt`
   were written by the cluster's older libtorch (2.1-style hex TensorImpl keys). The
   desktop's libtorch 2.9 parses those keys with `stoull` → every state collapses onto
   one param → scrambled moments → `Adam::step` size error on iteration 1 (this is the
   "256 vs 1280" crash the previous session attributed to branch drift; it reproduces on
   plain `private` and is a desktop-only artifact). The cluster's own libtorch remaps
   correctly — the original run resumed these same archives at every hop. The guard
   validates each mapped state against its param and resets **that model's** state only
   on mismatch, logging either `Optimizer state for "X" loaded (N param states verified)`
   or `WARNING: optimizer state for "X" ... RESET`. Grep the first hop log for which.
2. `[DIST][COLLECT]` per-iteration log line on rank 0 only (96 ranks × 6h ≈ 500MB otherwise).

## Config decisions (all in the sbatch header too)

- **Semantic six unchanged**: `GGL_GCO=1 GGL_NO_REACH=1 GGL_HULL=0 GGL_NO_VERSIONS=1
  GGL_NEXTO_SERVE_FRAC=0 GGL_TRAIN_AGAINST_OLD_CHANCE=0`, plus `GGL_LEAGUE=0` explicit.
- **Effective batch held at 200,448** (what the LR was tuned at): 96 ranks × 2088 mb,
  8352 rows/iter/rank, 174 arenas/rank. Same 801,792 fleet steps/iter, 8 updates/iter,
  24-step GAE window as the 32-node original. Cost: ~2× per-iteration collect wall.
  Alternative not taken: keep 87 arenas/rank and run at half the effective batch.
- **Engine = HEAD defaults** (corrected physics, dodge torque off, obs-flag noise off).
  Per `GCO_RESUME.md` D1: every engine commit post-dates the checkpoint, ≈12% relative
  eval dip expected to adapt out; not reverted because the corrected physics was
  validated against real-game traces and real matches are the objective.
- `GGL_FRESH_OPTIM=0` (state loads correctly on the cluster; the guard is the backstop).
- tickSkip 8 / γ 0.9969 / λ 0.95 / goal γ 0.9994 are compiled in on `private` (verified).

## Local smoke (desktop, `GGL_SMOKE=1`, CUDA, `GGL_FRESH_OPTIM=0`)

Booted with **zero unexpected "will be reset"** lines, the guard reset all 10 optimizer
states (desktop key-collision, expected), then 9+ iterations from `Total Timesteps
2.357269e+11 / Total Iterations 294,001` with saves rotating every 2 iters, no NaN.

## Morning checklist

```bash
ssh aimos 'squeue -u $USER -o "%.10i %.12P %.10j %.3t %.10M %.6D %R"'
ssh aimos 'ls -t ~/scratch-shared/logs/gco_220b_*.out | head -1 | xargs grep -E "PREFLIGHT|optimizer state|will be reset|Total Timesteps|Episode Length|FATAL|NaN" | tail -40'
ssh aimos 'ls ~/scratch-shared/checkpoints_gco_220b'
```
Healthy = timesteps climbing past 235,726,848,000, new numbered dirs appearing, Episode
Length trending down (inverse goal rate; mean reward is 0 by construction under GCO).
The pinned restore point is untouched at `barn-shared/pulsar2-gco/RESTORE_ts8_235726848000`
and on the Desktop.

Known leftovers: job `4677876` (`gco-build`, 1 node, pending since 14:01) is the previous
session's VTS build job in the `pulsar2-gco` tree; it does not touch this run.

## 00:20 update — cluster blocked, desktop stopgap running

SLURM's own estimate (`squeue --start`, confirmed with `sbatch --test-only` at 4/6/8/16
nodes and a 1h wall) puts **every** job of ours at 2026-09-03 afternoon/evening: the 15
idle nodes are held for higher-priority jobs invisible to `squeue`. No cluster training
was going to happen by morning at any size, so the chain `4677970-73` stays queued and
untouched, and a **desktop stopgap** of the same resume was started at 00:24:

| | |
|---|---|
| binary / branch | `pulsar2-22b/build/GigaLearnBot`, `22b-compat` (same as the cluster) |
| folder | `pulsar2-22b/build/checkpoints_gco_220b_local` (seeded from the bundle, keep 8, save every 40 iters ≈ 8M steps) |
| semantics | identical env contract; 1024 arenas; 200k rows/iter, mb 12.5k grad-accum → 2 updates/iter = 1 update per 100k steps at ~200k effective batch, the same update density and effective batch as the cluster split; GAE window 98 steps vs the cluster's 24 |
| optimizer | all 10 states reset by the guard (desktop libtorch cannot read the cluster-written archives) — i.e. a cold-momentum restart, the same thing `GGL_FRESH_OPTIM=1` would do |
| speed | ~80–100k steps/s → ~2.5–3B steps overnight |
| wandb | `7.9-gco-220b-desktop`; note it **resumes wandb run id `739468ux`** (from RUNNING_STATS), the original lineage's run — the cluster run will append to the same id when it starts |
| wrapper | `tools/run_trainer.sh` (nohup fallback, crash-restart); log `pulsar2-22b/run_logs/latest.log`; stop with `RUN_TRAINER_LOG_DIR=/home/luca/Projects/pulsar2-22b/run_logs tools/run_trainer.sh --stop` |

**The fork to resolve in the morning (your call, nothing automated):** if the cluster
chain starts while the desktop run is alive, both continue from 235.7B independently and
both write to wandb id `739468ux`. Options: (a) stop the desktop run and let the cluster
proceed from the bundle (discard the desktop steps); (b) stop the desktop run, copy its
newest complete numbered dir into `~/scratch-shared/checkpoints_gco_220b/` before hop 1
starts (the loader takes the newest numbered dir), so the cluster continues the desktop
progress; (c) scancel the chain and keep training locally. Desktop-written `*_OPTIM.lt`
load on the cluster (string keys remap by order) but carry the cold-restarted moments.
