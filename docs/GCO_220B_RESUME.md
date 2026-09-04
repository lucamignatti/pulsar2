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

## 04:35 update — cluster hop 1 running, desktop stopgap stopped

Hop `4677970` got a dcs-2024 allocation at 04:31 (24 nodes drawn, 7 banned excluded,
16 used). NCCL self-tests passed, config banner correct (174 arenas / 8352 rows / mb 2088),
and **all 10 optimizer states loaded with shapes verified on every rank, zero resets, zero
"will be reset"** — confirming the libtorch-key diagnosis. Throughput ~890–910k steps/s,
0.9 s/iteration. The desktop stopgap was stopped at 04:34 at 236,926,730,240 steps; its
folder `pulsar2-22b/build/checkpoints_gco_220b_local` is kept but is now a dead branch
(the cluster resumed from the 235.7B bundle and overtakes it within ~25 min). Harmless log
oddity: the node-hygiene `pkill -f build/GigaLearnBot` sweep matches its own srun shell and
prints `Killed`; the `|| true` absorbs it (fix later: `pkill -f "[b]uild/GigaLearnBot"`).

## 10:00 update — v2 config from hop 2 (user-directed "learn faster": items 1,2,3,4,6)

Hop 1 (`4677970`, v1 config) runs to its wall at ~10:31; hops 2–5 were re-queued on
`pulsar2_gco_220b_v2.sbatch` (copies of both sbatch versions in `tools/aimos/`). Measured
v1 split: collect 0.52 s (env 0.47) + learn 0.37 s, sequential, 0.90 s/iter.

| # | change | how |
|---|---|---|
| 1 | pipelined collection ON | `GGL_PIPELINED_COLLECTION=1` (iteration → max of the two phases) |
| 2 | up to 32 nodes | `--nodes=16-34`; preflight NEED table {32,24,16,12} recomputes arenas/rows/mb per rank so the fleet is always 16,704 arenas / 801,792 steps/iter / 24-step GAE window |
| 3 | effective batch halved to **100,224** | mb/rank 522/696/1044/1392 by N (was 200,448) — 2× updates per step; LR unchanged (tuned at 200k: the one deliberate deviation) |
| 4 | epochs 2 → 3 | `GGL_PPO_EPOCHS=3` → 24 updates/iter (was 8) |
| 6 | version ring ON | `GGL_NO_VERSIONS=0`, train vs old versions 0.30 (Nexto stays 0), `GGL_TS_PER_VERSION=1e9` (new env override; the 25M default is a desktop cadence and would give a ~15-minute ring at cluster speed) |

Declined by the user: seek injection (7) and any reward shaping (8). Also fixed: the
node-hygiene `pkill -f` now uses `"[b]uild/GigaLearnBot"` so it cannot match its own shell.

Gates to watch from hop 2 on (compare against hop 1's 236–252B stretch): Episode Length
slope, Policy Entropy (0.56 at hop 1), Policy/Critic Update Magnitude (~0.4), KL/clip, and
Bonk crossplay on synced checkpoints. `Rating/1v1` is inflated by construction; read
`Ref/Oldest Share` for the ring. If the halved batch destabilises, the revert is v1's
per-rank split (mb 2088 at 16 nodes) with everything else kept.

## 10:35 update — hop 2 (v2) measured; from-scratch true-VTS run launched

**Hop 2 (`4678207`, v2)** started 10:25 on 16 healthy of a 28-node dcs-2024 draw. Banners
confirm pipelined ON, version ring ON (`GGL_TS_PER_VERSION: 1000000000`, skill fleet 16
arenas), mb 1044, epochs 3. Measured: collection 0.53 s now fully overlapped (join 0), PPO
learn 0.89 s (24 updates vs 8 — dispatch-bound per the platform law), iteration 1.06 s,
~760k steps/s. So ~15% fewer steps/s than v1 but ~2.5× more optimizer updates per second.
A 32-node draw will cut rows per minibatch in half and shrink the learn pass.

**From-scratch true-VTS run** (user: "private, not titan; GCO; vts and everything"):
branch `vts-true-skip` + `private` HEAD merged (`26753b0`, clean merge), cluster tree
`~/scratch/pulsar2-vts` (git worktree of pulsar2-private-luca, own build), launcher
`pulsar2_gco_truevts_cs.sbatch` (copy in `tools/aimos/`), chain `4678218 → 19 → 20 → 21`
(el8,dcs-2024, 16–34 nodes, NEED = all healthy ≤ 32, learners = 1/3 of ranks). Lineage
`~/scratch-shared/checkpoints_gco_truevts_cs` (fresh, verified absent), wandb
`7.9-gco-truevts-cs`, logs `~/scratch-shared/logs/gco_truevts_<jobid>.out`. Config: GCO
sparse, APPO (global batch 800k decision rows, lag 64, epochs 1, 128 arenas/collector),
true VTS factored head with buckets {1,2,4,8,16} ticks and per-tick gammas, no advantage
skip bias, entropy on the control marginal only, LoRA league (theta-run settings, repulsion
0), reach/SIL/vdag/gap/goal-critic from birth, all three team modes pinned on, version ring
at 1B steps/version. No pooling exists on this branch. Watchdog kills a hop that never
iterates (20 min) or stalls (25 min). This branch had never run on the cluster before
this job; the first hop's boot is the smoke test.

## 11:00 update — true-VTS first contact: three bugs, all fixed on `vts-true-skip`

The branch had never run on the cluster. Three failures in sequence, each found by a
2-node 20-minute smoke (`truevts_smoke.sbatch` in `~/scratch/pulsar2-vts`):

1. `356bf42` **League KL push term**: base policy chain forward ends at the 95-wide factored
   layer, variant logits are the 450 table → `sub` size error on every learner.
2. `72cb4dc` **Ring column assert ordering** (`LearnerAsync.cpp` `ErasePrefix`): the
   `isDec/ticks` length check ran AFTER the main columns were trimmed, comparing pre- and
   post-erase lengths; fired on every collector at the first tick-budget cut (~25 s). Data was
   consistent. Check moved before the erases (with a sizes message).
3. `37c7fbb` **League Variant-KL telemetry**: the same unexpanded base forward as (1).
   All `PolicyChain` forwards now go through `ExpandFactoredLogits`.

Also `be42ee9`: a 3-shot minibatch shape log in `LeagueModule::Learn` (note: RG_LOG goes to
block-buffered stdout under mpirun, so it is lost on abort — stderr `what()` is what you see).

## 15:40 update — true-VTS skip collapse diagnosed; per-decision cost deployed (user: "do 2")

Diagnosis (details in `vts-true-skip:research/reports/TRUE_VTS_DECISION_COST.md`): at
2.9B rows the run's learning meters sat where the ts2-mm APPO baseline sat at the same
stage (flat through its first ~7B steps), so "not learning" was premature; but the factored
skip head was collapsing toward 1-tick holds (mean chosen ticks 3.9 → 2.7, 1-tick share
0.33 → 0.47, ~86 decisions per game-second, 55–67% filler rows in the batch, ~3× less
game-time per wall-hour than ts2). Cause: goal-only reward puts no price on a decision, so
the shortest hold is weakly optimal; the toy study only got a skip optimum with a per-decision
cost. Chosen lever (option 2): `GGL_VTS_DECISION_COST=0.001`, subtracted from every DECISION
row's stored reward in the collector (`bcc66e8` on `vts-true-skip`), pre-registered with
gates at 5B/10B rows (mean ticks ≥ 4.5, 1-tick share < 0.25, sim-seconds/s ≥ 2×, touches/s
not below 0.010–0.013). Learner split left at NP/3 on purpose (one lever at a time).
Deployment: old chain cancelled at the 3.6B save; new chain `4683363 → 64 → 66 → 67 → 68 →
71` on the cost launcher (copies in `tools/aimos/`). The 2-node smoke could not get nodes
in 30 min and was skipped: the change is three lines on a reward scalar and the hop has its
own retry/watchdog/chain guard; the boot banner `PER-DECISION COST: 0.001` and first
iterations are verified directly instead.

## 18:20 update — two experiments branched off the 220b lineage (user: "2ts and mm")

Seed: the newest full 220b save `270604800000` copied into two new lineages on
scratch-shared. Main 220b chain untouched (16–32 nodes). Both experiments run the same
`22b-compat` sync trainer, v2 config (pipelined, half batch 100,224, epochs 3, version ring),
elastic 8–16 nodes, 4 hops each; launchers in `tools/aimos/`.

| | 2ts (`checkpoints_gco_220b_ts2`, wandb `7.9-gco-220b-ts2`) | mm (`checkpoints_gco_220b_mm`, wandb `7.9-gco-220b-mm`) |
|---|---|---|
| change | `GGL_TRAIN_TICK_SKIP=2`: tickSkip 2 (60 Hz) with gaeGamma 0.9969^(1/4), gaeLambda re-derived to keep the γλ horizon, goalCritic γ 0.9994^(1/4), entropyScale 0.14/4 — wall-clock horizons preserved (commit `ada5c20`) | `GGL_MULTI_MODE=1`: 1v1/2v2/3v3 arenas 1/3 each from the first iteration (PHASE_B_ENABLED is compiled false, so the marker alone cannot; the env pins g_PhaseB) |
| per-rank split @16 nodes | 48 arenas → 96 players × 87 steps = 8352 rows (GAE window ≈1.45 s of game, vs 1.6 s at ts8) | 87 arenas → 29×(2+4+6) = 348 players × 24 steps = 8352 rows (same window as main) |
| what it tests | the 8→2 curriculum step on a mature policy: expect a competence dip then recovery (the 08-23 8→1 jump recovered in ~8 h); read goals per game-second and Bonk crossplay at matched game-time | transfer of a 1v1-only policy to team play; read per-mode Rating/Ref shares and Episode Length by mode |
| not changed | reset mix, GCO reward, NoTouch(20 s), reach off, hull off | same |

## 2026-09-03 04:12-04:45 — ts2 mid-hop segfaults; launchers now relaunch within the allocation

- **ts2 hop 2 (`4685597`) died at 04:12** after 40 min: rank 34 on dcs214, SIGSEGV with a
  null function pointer inside `at::isfinite` (TensorIterator build) called from
  libGigaLearnCPP on a `std::thread` (the collect worker). Last save `284636160000`.
- **ts2 hop 3 (`4685598`) died at 04:24** after 12 min, before any save: rank 14 on dcs180,
  SIGSEGV inside `posix_memalign` (corrupted malloc arena) under `empty_cpu` from a
  comparison op. Same 16-node set as every ts2 hop; two different nodes.
- Both stacks are heap-corruption-shaped, on CPU tensor ops in the collect thread. Only the
  tickSkip-2 runs have crashed so far: ts2 2× in ~3h, ts2mm 1× (the 03:06 SVD/LAPACK abort
  — that exception is caught hundreds of times per hop on every run, it escaped once),
  main/mm 0× in ~15h. Not root-caused. Episode buffers DO scale with tickSkip
  (`maxEpisodeLength = maxEpisodeDuration * 120/tickSkip`, Learner.cpp:1375), so the
  "ts8-sized buffer overflows at ts2" hypothesis is out.
- **Launcher bug exposed:** the 3-attempt loop only retried deaths inside the first 300 s of
  the hop, so each mid-hop crash ended the hop and burned a chain link; ts2 went from hop 2
  to hop 4 (`4685599`) in 13 min and had NO queued hops left behind it.
- **Fix (`586f595`):** `_ts2/_mm/_ts2mm.sbatch` relaunch from the newest save while >= 15 min
  of allocation remains (wall end from `squeue -o %e`, fallback HOP_T0+6h); 3 consecutive
  fast failures still hold 30 min. Queued mm/ts2mm hops keep the old script (spool trap)
  but have not crashed; new ts2 hops `4685675 -> 4685676 -> 4685677` (6h each, 6-18 nodes)
  chained after `4685599`, which covers 19:00 with margin. A crash now costs ~1 min plus
  the progress since the last save (~7 min at 400M/save).

## 2026-09-03 08:15 — mm crashed too; the crash site is the collect worker's obs finite check

- **mm hop 3 (`4685573`) died at 08:15** (SIGABRT, rank 45 on dcs247) after ~4.7h, last save
  `284515200000`. This time the trainer's own handler printed the cause:
  `FATAL: collect worker exception: Expected all tensors to be on the same device, but found
  at least two devices, cuda:3 and cpu!`, raised from `TensorIterator::compute_types` under
  `at::isfinite` ← `Learner::Start()::{lambda#18}` on the pipelined-collection thread
  (`{lambda#22}`, symbolized with addr2line on the cluster .so).
- That is the **obs NaN guard** at `Learner.cpp:2628-2634`: every 4th step the collect worker
  wraps `envSet->state.obs.data` with `torch::from_blob` (CPU, float32) and calls
  `isfinite(...).all()`. Nothing in that expression can legitimately be on `cuda:3` — a
  fresh CPU `from_blob` tensor and two CPU bool intermediates — so a device mismatch there
  means a **TensorImpl with garbage metadata**, i.e. heap corruption in the collect process.
  The ts2 hop-2 segfault (null function pointer inside the same `isfinite` → `mul` →
  `build_borrowing_binary_op`, same two libGigaLearnCPP frame offsets `+0x2504d4/+0x2574c8`)
  and the ts2 hop-3 `posix_memalign` arena corruption are the same corruption seen from
  three angles. The guard is the most frequent allocator-heavy CPU op on that thread, so it
  is where the damage surfaces, not necessarily where it originates.
- Tally: ts2 2 crashes / ~5h, mm 1 / ~9h, ts2mm 0 segfaults / ~8h (its 03:06 abort was the
  SVD/LAPACK exception escaping — different), **main 0 / ~30h**. So it is not tickSkip-2
  specific; the main run's config (PHASE A, 174 arenas/rank, 20 nodes) has not shown it.
- **Not root-caused; no code change deployed** — a fix means rebuilding the shared
  `libGigaLearnCPP.so` that the LIVE main trainer has mapped, which the user said not to
  touch. Next diagnostic step (needs approval): an ASan/`-g` build in a separate cluster tree
  under a 2-node smoke with `GGL_MULTI_MODE=1`, and audit the collect-thread writers indexed
  by player/arena (`ksSuppress/ksArenaCar`, ring-opponent rows) for out-of-bounds on
  multi-mode arenas.
- **Chains hardened instead:** fixed-launcher hops appended — mm `4685700` (after `4685600`),
  ts2mm `4685701` (after `4685601`); ts2 already had `4685675-77`. Queued old-launcher hops
  (`4685574`, `4685600`, `4685589`, `4685601`) still end on a crash but the chain continues.
- **10:10 — mm hop 4 (`4685574`) died again after 1h** (last save `285707520000`), same
  handler, a different libtorch complaint: `collect worker exception: Index is supposed to
  be an empty tensor or a vector`, from CUDA `index_select` in the same collect lambda
  (`+0x24de8c`, `{lambda#18}`) — the `tdStates.index_select(0, idxNew/idxOld)` opponent
  split at `Learner.cpp:2763-2797`. A 1-D index tensor built every iteration reading as
  non-1-D is the same "TensorImpl metadata is garbage" symptom. mm's interval is now ~2h.
- Ruled out from the launchers: `GGL_OPP_PARALLEL` (the known-broken side-thread opponent
  inference) is set nowhere; the mm/ts2 environments differ from the crash-free main
  launcher only in `GGL_MULTI_MODE` / `GGL_TRAIN_TICK_SKIP` / `GGL_TS_PER_VERSION` and the
  per-node-count arena rows. **Stopping log forensics here** — four distinct libtorch
  symptoms from one thread is a heap-corruption signature that only an ASan/`-g` build will
  localize. Recommendation for the user: approve a 2-node ASan smoke of the private tree
  with `GGL_MULTI_MODE=1` (that reproduces fastest, ~2h) in a separate cluster build dir.

## 2026-09-03 18:xx — all three experiments log into the MAIN run's wandb id (739468ux)

Found while pulling telemetry for the mm-vs-1v1 analysis: `gco_220b_mm/ts2/ts2mm` were seeded
by copying a main checkpoint folder, and `RUNNING_STATS.json` carries `"run_id": "739468ux"`,
so every experiment resumes the main run's wandb run. Four processes write interleaved rows
into one history; wandb's upload appears to have stalled at ~13:45 for all of them. The rows
are separable offline (each run's exact time->steps curve from the hop logs is in
`research/results/mm_slow_learning_20260903/ref_series.txt`; the split history is
`wandb_739468ux_split_by_run.json`), but the live dashboard for the main run is polluted.
Fix (NOT applied — user said change nothing): delete/replace `run_id` in each experiment's
newest checkpoint `RUNNING_STATS.json` (or set a fresh `GGL_RUN_NAME` + clear the id) before
their next hop; each would then start its own wandb run.
