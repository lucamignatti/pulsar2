# MOE_SPEED — making the 402M/15M-active MoE run at dense speed on the V100 fleet

Status: **STAGE 1 IN PROGRESS** (CUTLASS + fused routing kernels).
Owner run: 7.3-moe-b3 (`checkpoints_moe2_luca`, wandb `7.3-moe-b3`).
Companion docs: `MOE_POLICY.md` (architecture + bring-up), memory notes
`cuda-graphs-cluster-verdict`, `moe-bench-numbers`.

## The problem, quantified

The ppc64le platform bills **CPU op dispatch** (~0.15–0.5ms per tensor op), not GPU
time. Measured per collect tick:

| config | launches/tick | infer ms/tick | fleet SPS |
|---|---|---|---|
| dense trunk | ~20 | ~1.6 | ~260k |
| MoE 6 blocks × 128 experts | ~190 | 24–37 | ~52k |
| MoE 3 blocks × 512-hidden (same params/FLOPs) | ~95 | ~12–18 | ~89k |

Same active FLOPs as dense, 3× slower — every remaining millisecond is launch count
(glue ops: argsort/cummax/index dance ~40 torch ops per block) plus padded-GEMM
waste (capacity buffer + trash row).

## What was tried and settled

1. **CUDA graphs (GGL_CUDA_GRAPH, commit a3248f1)** — capture the whole collect
   forward as one replayable launch. **Desktop: 5× smoke speedup, kept.**
   **Cluster: DEAD** — torch's `capture_begin` segfaults in `libcuda
   cuStreamGetCaptureInfo` (SIGSEGV, uncatchable, crash-looped 3 chain hops on
   2026-08-16). A raw-CUDA probe passes all three capture modes on the same nodes:
   the driver is fine single-threaded; torch's graph path on CUDA **11.2** under
   concurrent multi-thread traffic is the failure (torch graphs are only solid on
   ≥11.4). Fleet sbatch pins `GGL_CUDA_GRAPH=0`. Do not retry without a torch
   rebuild ≥11.4 — each experiment costs crash-looped hops on a live lineage.

2. **moe-bench** (`~/scratch-shared/moe-bench`, 2026-08-13) — measured on-node:
   - CUTLASS 2.x sm_70 grouped GEMM (`kDeviceOnly` schedule): fp16 ds200_t1024
     expert FFN **3.65ms vs 7.14ms torch_padded** (= our baddbmm approach) —
     ~2× GEMM win at our many-small-experts shape, 4.2× vs per-expert mm.
   - NCCL intra-node all-to-all: **0.09–0.36ms** against 1.7–5.3ms GEMMs — expert
     parallelism is bandwidth-viable on the 6×V100 nodes.
   - Liftable kernels: gather/permute/scatter_add(+scale) fp16/fp32; grouped GEMM
     with fused-SiLU epilogue (host-side offsets — needs a device-fill variant).

## Stage 1 — fused routing + CUTLASS grouped GEMM (collect forward)

Target: MoE block glue ~40 torch ops → ~13 launches; exact-M GEMMs (no capacity
padding, no trash row, **no capacity drops at all**). Expected ≥2–3× collect.

Design (`Util/MoEKernels.{h,cu}`, CMake option `GGL_MOE_KERNELS`, env
`GGL_MOE_CUTLASS=1`, fp16 no-grad path only — the learn pass keeps the
autograd baddbmm path unchanged):

```
xn = LN(x)                                   torch
logits = xn @ routerW.T                      torch (fp16 mm, fp32 accum)
plan kernel(s):                              CUSTOM (2 tiny launches)
    fp32 sigmoid(logits) + routerBias, top-k(4/128) per row,
    gate renorm, per-expert counts -> offsets (excl. scan),
    src_rows / gate / expert_id per assignment in expert-sorted order,
    fill d_problems (GemmCoord{m_e, N, K}) + A/B/C pointer & ld arrays
packed = gather(xn, src_rows)                lifted kernel
h  = groupedGEMM(packed, W1T)                CUTLASS, device-only schedule
h  = leaky(h + b1[expert])                   CUSTOM fused bias+activation
y  = groupedGEMM(h, W2T)                     CUTLASS
out += scatter_add((y + b2[e]) * gate)       lifted kernel + bias
shared expert + residual                     torch (4 ops)
```

Key mechanics:
- **Device-side problem fill** — CUTLASS `GemmGrouped` with `kDeviceOnly` is a
  persistent kernel: fixed grid, walks `d_problems` on device. problem_count = E
  constant; m=0 experts contribute zero tiles. No host sync of routing counts
  anywhere (a D2H sync would re-pay the dispatch tax the whole exercise removes).
- **Transposed weight caches** — CUTLASS wants B row-major [K,N] per expert:
  `W1T [E,d,h]`, `W2T [E,h,d]` fp16 contiguous, rebuilt when the half-cache
  refresh epoch bumps (once per iteration, not per tick).
- **Parity harness** — `GGL_MOE_SELFTEST` extended to run both forwards on the
  same inputs and compare (rel-RMS ≤ 1e-2 fp16). Gate deployment on parity, not
  eyeballs.
- Desktop builds keep `GGL_MOE_KERNELS=OFF` (CUDA 13 cannot even target sm_70);
  the baddbmm path remains the only desktop path.

## Stage 1b — CUTLASS autograd learn path (approved 2026-08-16, task #2)

Make learner compute scale with ACTIVE params: a custom `torch::autograd::Function`
replacing the eager padded-bmm routed-FFN (LN, shared expert, residual stay in
torch autograd around it). Forward reuses the collect kernels (route plan, gather,
grouped GEMMs, bias+leaky, gated scatter), saving packed X, post-activation H,
offsets/srcRows/expertId/gates. Backward:
- dY = gather(dOut, srcRows) * gate  (1 kernel)
- dB2 = per-expert segment sum of dY (atomic fp32 [E,d] kernel); dB1 likewise
- dH = grouped GEMM(dY, expertW2 as-is [E,d,h] — already the right layout);
  leaky mask from sign(H) (post-activation sign == pre-activation sign)
- dX_packed = grouped GEMM(dHpre, expertW1 as-is [E,h,d]); scatter-add to dxn
- wgrads NEED ONE NEW INSTANTIATION: GemmGrouped with LayoutA=ColumnMajor —
  row-major X [m,h] read as column-major IS X^T with lda=h, so
  dW1[e] = dHpre_e^T @ X_e lands directly in expertW1's [h,d] layout
  (M=h, N=d, K=m_e), same for dW2. No transposes materialized.
- gate grad: per-assignment dot(dOut[src], y+b2) kernel; the tiny [R,k]
  gate->logit chain + router wgrad stay in torch (3 small ops).
- Precision: fp32 params cast to fp16 inside forward (~50ms/iter total at 6
  calls), backward returns fp32 grads so accumulation/AMP-unscale is unchanged.
- GATE: selftest grad-parity vs eager autograd (grad relRMS per param family)
  before any fleet flip — same doctrine as the inference parity gate.

STATUS 2026-08-16 night: implemented + cluster-compiling; inference gates still
green (3.2e-4, 3.35x). Grad-parity run 4631402 FAILED with "CUDA error:
misaligned address" — root-caused: the wgrad grouped GEMM slices the transposed
assignment buffer at `T + offsets[e]` (arbitrary token counts) and uses
per-problem K = m_e, but the fp16 tensor-op template requires 8-ELEMENT
alignment on A pointers and K. (Forward GEMMs are immune: their offsets scale
by d/h which are ÷8; that is why inference parity passes.) Also fixed en route:
ColumnMajor-A has no legal sm_70 thread map at our tile — wgrad now transposes
once globally and uses the proven RowMajor template with lda = n.
NEXT STEP (exact): learn-path routing plan emits 8-ALIGNED per-expert segments —
scan kernel rounds each expert's segment start up to 8, pad slots get
srcRows = -1 / gates = 0, gather kernels write zero rows for -1, scatter/segsum
kernels skip -1; collect fast path keeps the unpadded layout (flag on
ggl_moe_route_plan). Pad rows are zero through the whole chain so wgrad picks up
exact zeros. Then re-run the grad-parity gate (4631402's config).

Muon-shard-under-APPO VERDICT (task #1, trial 4631380): DEADLOCK — 25 min,
ver=0 on all collectors, zero updates. The per-param learner-group bcasts in
StepOptimsSharded interleave against the collector weight-publish collectives
on a different NCCL comm (multi-comm ordering hazard). GGL_MUON_SHARD stays
OFF in async mode (annotated in pulsar2_asyncmoe_luca.sbatch); owner-local NS
via EP (task #3) is the correct optimizer-cost fix under APPO — it needs no
mid-step collective at all.

## Stage 2 — NCCL expert parallelism (user-directed)

Each of a node's 6 V100s owns E/6 = ~21 experts per block (fixed assignment);
collect and learn route tokens to expert owners by NCCL all-to-all, results
return the same way. Bench says intra-node comm is ~free. Design points:

- **Collect**: after the plan kernel, counts per (expert→rank) are known on
  device; a2a token exchange (fp16 rows), owner runs its experts' grouped GEMM,
  a2a back. Per-rank expert weights shrink 6× → the 32GB ceiling lifts ~6× on
  expert params: the **1B-total model fits** (a 1B/6 ≈ 170M expert slice + dense
  trunk + optimizer per GPU).
- **Learn (APPO-shaped, approved 2026-08-16, task #3)**: EP lives on the LEARNER
  GROUP. Each learner owns E/nL experts; the MoE learn forward does an NCCL
  all-to-all on the learner group (tokens to owners, expert outputs back; the
  backward reverses it), so an owned expert's grads complete owner-local —
  NO expert-grad allreduce exists at all, and NS/optimizer state shrink to the
  owned slice. The learner-group allreduce shrinks to dense trunk + router +
  heads (~4M params). Lift the a2a mechanics from moe-bench `moe_ep.cpp`.
  Session already routes collectives to the learner group under async routing
  (group_rank/group_world added 2026-08-16 — global-rank sharding would have
  assigned owners to collector ranks that never step).
  Sequencing: AFTER Stage 1b (a2a wraps the grouped-GEMM path, not the eager one).
- **Publish (task #4)**: expert-delta publishing — per-expert version counters in
  WeightDoubleBuffer, collectors pull only changed experts + periodic full-net
  resync. Cuts the 800MB/publish; prerequisite for Stage 3 per-expert async.
- **Inter-node EP** is an experiment, not a default: bench measured intra-node
  NVLink; inter-node IB latency per a2a needs measuring first (pre-register a
  threshold: EP pays if a2a < 20% of the expert GEMM time it parallelizes).
- Muon note: expert stacks become per-rank-local 3D [E/6,m,n]; the batched-NS
  path already handles arbitrary E — the shard predicate must treat owned expert
  stacks as always-local (no bcast of expert params).

## Stage 3 — async per-expert learning + expert version ring (user-directed)

Once experts live on fixed GPUs, their updates need not be synchronous with the
trunk step. Direction (needs its own pre-registration before build):

- Owners apply expert updates as token batches arrive, decoupled from the
  iteration barrier; router/trunk stay on the synchronous PPO clock.
- **Reproducibility ring**: keep the last N versions of each expert's weights
  in host RAM (an expert slice is small: 402M/128 ≈ 3M params ≈ 6MB fp16 —
  N=8 versions × 384 experts ≈ 18GB host RAM, fine on these nodes). Every
  trajectory records the (expert, version) pair it was served by; reproducing a
  result = swapping that version back in. This is the MoE analogue of the
  policy-version snapshot invariant ("logProbs come from the snapshot that
  acted") — the IS ratio stays exact per-expert.
- Risks to pre-register: router/expert staleness interaction (router learns
  against moving experts), per-expert learning-rate fairness under load skew,
  and the anti-ratchet concern if asymmetric staleness correlates with
  advantage sign. Measurement before machinery: A/B a 2-expert-async pilot
  against the sync run before fleet-wide.

## Amendment 2026-08-16: graphs are BACK on the table (titan-graphs merge)

The "cluster-dead" verdict above was half right: the segfault is ATen's
`capture_begin` passing a null pId into `cudaStreamGetCaptureInfo` (driver 460
writes through it). The `titan-graphs` branch (w451/jcooley972) root-caused it
and captures with RAW driver APIs while keeping torch's graph mempool
(`beginAllocateStreamToPool`), graphs only the deterministic probs forward
(multinomial stays eager — raw capture cannot advance philox), syncs after
replay on drivers < 11.4, and evicts graphs from Model destructors. Merged
(commits 8aeae82 + 882e2ac + compat fixes), desktop-validated pipelined
(66 iters, captures + replays, stable entropy). Env: `GGL_CUDA_GRAPHS`.
CUTLASS scratch/caches were made pointer-stable (per-shape scratch, copy_-in-
place weight caches) so graphs and the CUTLASS path compose.

## Parity + speed result (Stage 1 gate, V100, 2026-08-16)

- First run FAILED honestly at relRMS 0.114: the fast path routed on fp16-rounded
  logits (eager routes fp32) and the test's x100 router boost saturated sigmoids
  into topk ties. Fixed: fp32 router matmul (cached fp32 router weights).
- **PASS: relRMS 3.16e-4** (fp16 noise). **Block forward 777 rows: eager 2.82ms →
  fast 0.83ms (3.41×)**.
- Also fixed en route: the config-order trap had silently clobbered
  GGL_NO_VERSIONS / GGL_NO_REACH / GGL_NO_HEADROOM (see ExampleMain's LATE
  OVERRIDES section) — the clobbered version ring reloading on resume (+6.2GB)
  was what OOM'd every resumed hop and exhausted two chains; the saved ring is
  quarantined at `checkpoints_moe2_luca/policy_versions_quarantine_20260816`.

## Execution state

- [x] Bench numbers read into the record (this doc + memory).
- [x] Stage 1 build: MoEKernels.{h,cu}, CMake gate (`GGL_MOE_KERNELS`), fast path,
      parity gate PASSED (3e-4), 3.41× block forward.
- [x] titan-graphs merge (raw capture) + pointer-stability for composition.
- [x] **Fleet flip: GGL_MOE_CUTLASS=1 live on chain 4631318+** (7.3-moe-b3).
- [x] Fleet measurements (2026-08-16 evening, jobs 4631318/26/34/42):
      CUTLASS collect infer 8.0→5.6ms/tick, SPS 89k→99k — then the run became
      LEARN-BOUND (collect 0.9s hides under learn 1.5s; iteration ~2.0s), so
      graphs bought zero SPS (and the <11.4 post-replay sync worsened collect
      latency 0.55→0.76s) → GGL_CUDA_GRAPHS parked OFF, code stays ready.
      Learn breakdown (GGL_CONSUME_TIMERS): fwdbwd 0.77 / optstep 0.39 /
      allreduce 0.32. GGL_MINIBATCH 926→2778 halved fwdbwd (0.37s), learn
      1.50→1.28s, **SPS ~105k**, mem 10.4GB. Now the top items are
      allreduce 0.49 + optstep 0.39 (incl. the 1.6GB owner-bcast) — BOTH are
      exactly what Stage 2 EP eliminates by construction. EP is next.
- [x] Stage 1b SHIPPED (2026-08-17): grad-parity green all 6 families incl. an
      UNDER-AUTOCAST pass (the fleet's AMP intercepted the router mm → fp16
      logits read as fp32 → illegal access, jobs 4631411-17; AutocastOffScope
      now pins precision inside the custom fn). Fleet: fwdbwd 0.37→0.32s,
      SPS ~105k→115-128k, entropy healthy. MoE-APPO remeasured: 171k→190k
      (2 nodes; dense 600k — gap 3.2x, all in optstep 0.39 + allreduce 0.33 +
      publish, ALL total-param costs).
- [ ] Stage 2 (EP) IN PROGRESS — build order, each gated:
      A. DONE (d9089b9): Session::allgather_host_group +
         alltoall_rows_f16_group on the learner group (lifted from moe-bench;
         self-traffic = local copy).
      B. EP mode in MoERoutedFFN (env GGL_MOE_EP): after the route plan, host-
         allgather per-expert counts; owner o owns experts [o*E/nL,(o+1)*E/nL);
         a2a packed rows (+expertId localized to the owner's slice) to owners;
         owners run the SAME grouped-GEMM chain on their slice for ALL learners'
         tokens; a2a y back. Backward mirrors (dY a2a to owners; owners compute
         dgrad/wgrad — wgrad for owned experts completes LOCALLY; dX a2a back).
         nL=1 must degenerate to the current path bit-for-bit (selftest gate).
      C. Owner-sliced optimizer: Muon steps only param.narrow(0, own0, ownN) of
         3D stacks (per-param epSlice fields); AllReduceGrads EXCLUDES dim-3
         expert params under EP; post-step fp32 slice bcast per owner AT THE
         ALLREDUCE CALL SITE (that lane provably coexists with publishes — the
         per-param mid-step bcasts are what deadlocked Muon-shard; v2 = task #4
         removes the replication entirely).
      D. 2-node MoE-APPO trial: target learner SPS >> 190k; then scale nL.
- [ ] Stage 3 (async experts) pre-registration — also the lever that cuts
      per-expert NS FREQUENCY (parity with dense optimizer cost at ~16 sync
      learners, or fewer with async expert cadence).

## THE 1B MODEL — LANDED 2026-08-17 (run 7.5-moe-1b)

**Config**: `GGL_MOE_EXPERTS=320` at the existing 3 blocks x hidden 512 x width
1024 = **1.006B total / ~15M active** — same active compute as the 402M run, 2.5x
the capacity. NO architecture change was needed; 1B was purely a memory problem.

**The memory find (this is the whole story).** 1B OOM'd at 29.7GB/31.75GB on BOTH
the sync and async paths. Ledger at 1B: fp32 params 4.0 + Muon momentum 4.0 +
grads 4.0 + fp16 learn caches 4.0 + seqHalf mirror 2.0 ~= 18GB, and then the
**pipelined collect snapshot** adds a WHOLE SECOND COPY of the trunk (fp32 clone
+ its own fp16 mirror) — at 1B that is the difference between 29.7GB and 15.9GB.
`GGL_PIPELINED_COLLECTION=0` was the single change that made 1B fit. Cost: collect
no longer hides under learn (~35% throughput), which is the right trade to exist
at all. Also landed: EP owner-slices the fp16 learn caches (4GB -> 1GB at nL=4).

**Live**: 8 nodes / 48 GPUs, 7.7GB per GPU (abundant headroom — the minibatch can
grow), chain 4631491-99 (9 x 2h hops), `checkpoints_1b_luca`, wandb `7.5-moe-1b`.

**Path back to pipelining at 1B** (untried): the snapshot only needs the POLICY
half in fp16 for collection — a fp16-only snapshot would cost ~2GB instead of ~14GB.

## EP CONVERGED + THE 1B COST LEDGER (2026-08-17)

EP works on the SYNC path. The async (APPO) deadlock was never diagnosed; the
sync path is in strict rank lockstep so the collectives match by construction.
Bugs fixed to get here, each real: EP armed on collectors (needs `is_learner()`),
world-comm broadcast in the replication (`bcast_device_group`), GEMM ctx sized E
instead of nL*ownCount, and the remainder-on-last-rank ownership split (rank 23
held 21 experts vs 13 — one straggler paced all ~72 exchanges/iteration).

**Measured at 1B, 4 nodes / 24 ranks, per learn pass:**

| | non-EP (8 nodes) | EP | EP + expert-major |
|---|---|---|---|
| fwdbwd | 1.73 | 3.77 | **2.87** |
| allreduce | 2.38 | **0.028** | **0.028** |
| optstep | 1.77 | 1.49 | 1.49-2.0 |

EP does exactly what it promised — the 4GB expert-grad allreduce is GONE
(2.38s -> 0.028s) — and it made 1B memory-stable (7.7GB vs the 29.7GB OOM).
But the token exchange initially gave the saving back in fwdbwd.

**THE SHAPE FIND (bisect probe, decisive).** `GGL_MOE_LEARN_BISECT=1` (skip the
wgrad GEMMs) took fwdbwd 3.77 -> 1.30s: **wgrad was 65% of it.** Not volume —
SHAPE. NCCL needs per-peer contiguity, so rows arrive RANK-major and every wgrad
problem had K = cap = 48: 312 tiny problems that are almost entirely tile
overhead. One permute to EXPERT-major after the exchange gives ownCount problems
of K = nL*cap = 1152, and each expert's wgrad then spans all ranks' tokens
directly (the per-rank block sum disappears). fwdbwd 3.77 -> 2.87.

**Still open, in order of expected value:**
1. optstep 1.49-2.0s — Muon NS still runs over the FULL 1B on every rank.
   Owner-slicing it (grads are already exact zeros off-slice) is the next big cut.
2. The exchange itself. Hierarchical EP (expert-parallel WITHIN a node over
   NVLink, data-parallel across nodes) would cut both participant count and
   latency; the flat 24-rank a2a is the current design's weak point.
3. cap is 1.25x mean (38% padding at E=320) — lowering capacityFactor for EP
   trades a few dropped tokens for proportionally less GEMM work.

## ASYNC (APPO) EP MAY NOW WORK — HIGHEST-VALUE UNTESTED LEVER

The async deadlock's prime suspect was the blocking MPI_Allgather of routing
counts in EpBuildPlan: APPO learners are NOT in lockstep, so any path where one
learner runs a different number of MoE forwards desyncs a host collective
permanently. The fixed-capacity rewrite (EpBuildPlanFixed) removed that
allgather entirely — every exchange size is now a pure function of (E, cap, nL),
computed with ZERO host collectives. Verified by inspection: the only remaining
allgather is in the dead dynamic builder.

So the async path should be retried: sbatch `pulsar2_asyncmoe_luca.sbatch` with
GGL_MOE_EP=1. This matters because DENSE'S 600k CAME FROM APPO — dedicated
collectors with no barrier. Sync-path MoE is structurally capped well below it
no matter how much kernel work is done; the remaining multiplier is the
architecture, not the arithmetic. Success signal: "Async/Learner SPS" lines
appear at all (the old failure was fragments flowing with ver=0 forever).

## EP STATUS — earlier async record (superseded by the above)

Stages A-C are implemented, compile, and pass the nL=1 degeneracy gate (all six
grad-parity families + autocast unchanged). Multi-rank it still produces ZERO
optimizer updates. Two deadlocks were found and fixed along the way, both real:
1. Collectors also construct a PPOLearner, so EP armed on them and they entered
   learner-only collectives ("collectors must not enter Learners collectives").
   Fixed: `is_learner()` in the arming predicate.
2. `ReplicateExpertSlices` used `bcast_device`, which broadcasts on the WORLD
   comm — collectors are members but never call it. Fixed: `bcast_device_group`.
After both fixes the fleet still hangs with fragments flowing and `ver=0`.
Remaining suspects, in order: (a) a per-learn-step count mismatch in the
MPI_Allgather (host, blocking) across learners — any code path where one learner
runs a different number of MoE forwards desyncs it permanently; (b) MPI-collective
vs NCCL-stream interleaving on the learner comm; (c) a zero-row send/recv pairing
mismatch in the a2a. NEXT DEBUG STEP: log a per-rank counter of
allgather_host_group calls + a barrier_learners() immediately before the first
allgather — if the barrier hangs, the desync is upstream of EP entirely.

## POST-MORTEM 2026-08-17: the 16-node "production" failure was WANDB, not MoE

The 4× production hops (4631580-83) that ran 77-90 min each and retained 0 steps
were not slow at training at all. Autopsy of hop 4631583 (no timestamps in logs,
reconstructed from timer semantics + wandb debug logs):

- Every instrumented phase was healthy: collect ~4s (hidden by pipeline), learn
  ~9s (fwdbwd 4.48 + optstep 4.16 × 2 updates), `Overall Steps/Second` 85-88k.
- `Overall` is measured at the TOP of the epilogue. Everything after it —
  exit-flag collectives, save check, `MetricSender::Send`, `Display` — was
  unmeasured. Rank 0 stalled ~10-25 min per iteration inside **`wandb.log()`**
  (the compute nodes' network path to the wandb backend is pathological: hop
  4631580's `wandb.init` round-trip took 57 s; once the SDK's buffer fills,
  `.log()` blocks on backpressure with 90 s HTTP retries). The other 95 ranks
  waited at the next iteration's obs-stat sync — whose timer is only ever
  *displayed* for rank 0, where it reads ~0 because everyone else is already
  waiting. A perfect stealth stall: fleet at ~500 real SPS, log claiming 85k.
- Corroboration: hops 81-83 created no wandb run dir at all (init returned a
  degenerate run) and were slowest; hop 80 (working online wandb) was ~2.5×
  faster but still wandb-bound. `[HALFREFRESH]` cost growth (0.3 → 944 ms) was
  a symptom (queue backpressure), not a cause.

Fixes landed (commit 01416a7):
1. **MetricSender is now async**: `Send()` enqueues plain C++ data into a
   bounded (8, drop-oldest, logged) queue; a dedicated worker thread owns all
   Python/GIL. wandb can now be arbitrarily broken and cost zero train time.
   `add_metrics` failures log instead of killing the run; shutdown drain is
   bounded at 20 s then detaches.
2. **`True Iter Time` / `True Steps/Second`**: wall period between consecutive
   iterations measured across the WHOLE loop including the epilogue, displayed
   and sent. `[ITERWALL]` stderr stamp per iteration with time-of-day on rank 0;
   non-zero ranks print their obs-sync wait when > 5 s (GGL_CONSUME_TIMERS).
   The class of "timers say fast, wall says slow" can no longer hide.

METHOD LESSON (goes with "measurement before machinery"): a throughput metric
computed from phase timers is a claim about the phases, not the run. Gate
production decisions on wall-clock-derived rates only. The four configs pushed
to production on the strength of `Overall Steps/Second` were pushed on a number
that could not, even in principle, see the failure they died of.

## PARKED BUG 2026-08-17: pipelined + EP wedges at the first save boundary

With the wandb stall fixed, a fresh 16-node run (job 4631779) did 26 iterations
at **5-6.5s true / 30-40k SPS honest**, wrote checkpoint #1 (8.1GB, ~12s,
verify-passed) — then wedged permanently in iteration 27, the first POST-SAVE
iteration. Evidence: ranks 0/4/8/19 have 27 `[DIST][COLLECT]` lines vs 28 on the
other 92 (their iteration-28 pipelined collect never started → they never reached
the worker kick); all 96 main threads spin (the 4 in `cuCtxSynchronize`, the rest
in MPI/NCCL busy-poll); `DIST lockstep ok` count = 26 → learn #27 never completed
anywhere. The save path is the one structural novelty: on a save iteration the
collect worker is joined and NOT re-kicked, so the next iteration collects INLINE
on the main thread — this run is the first time 1B+EP crossed a save boundary in
pipelined mode at all (the 31.9M-step 4-node run had pipelining OFF). Suspects:
EP's NCCL-stream kernels from learn #26 still pending when the inline-collect /
snapshot-sync path device-syncs, interleaved with the obs-stat MPI collective —
the report's known trap (b) shape. NOT diagnosed further; fleet time was burning.

Decision: run 1B with `GGL_PIPELINED_COLLECTION=0` (chain 4631783-86) — the
configuration the 1B memory ledger wants anyway (kills the ~14GB snapshot copy;
reserved was 28.5/32GB and that pressure caused the cusolver telemetry crash of
job 4631777). Cost: collect (~2s at 2084 ts/rank) is exposed instead of hidden.
Anyone re-enabling pipelining at 1B+EP must first survive TWO save boundaries on
a throwaway arena.

## THE ROOT CAUSE, MEASURED 2026-08-17: the MoE forward is per-CALL bound, and
## production was running the smallest possible batch

Six single-arm probes (1 node, 6 ranks, 24 forward passes per iteration, identical
everything except the swept variable). `infer_s` covers exactly 24 forwards:

| arm | players/rank | geometry | ms/forward | collect SPS/rank | learn fwdbwd |
|---|---|---|---|---|---|
| A | 84   | 320 exp x 512  | **85.8** |     979 | 1.386s @ 2,016 rows |
| B | 336  | 320 exp x 512  |   69.2 |   4,858 | 2.224s @ 8,064 rows |
| C | 1344 | 320 exp x 512  | **84.6** |  15,890 | 3.505s @ 32,256 rows |
| D | 84   | 64 exp x 2560  | **35.2** |   2,389 | 1.408s @ 2,016 rows |
| E | 84   | dense          | **1.25** |  67,425 | 0.071s @ 2,016 rows |
| F | 1344 | dense          |   1.70 | 792,472 | 0.424s @ 32,256 rows |

**Finding 1 — the MoE forward cost is FLAT in batch size.** 84 tokens and 1344
tokens both cost ~85ms (A vs C: 16x the work, 0.99x the time). There is no token
term at all at these sizes; it is entirely per-call (launch + per-expert problem
overhead). Dense over the same range goes 1.25 -> 1.70ms, i.e. it actually does
work proportional to input.

**Finding 2 — that is the whole dense gap.** At the SAME shape (84 players) dense
is 1.25ms and MoE is 85.8ms: **69x**. Not FLOPs (MoE has ~4.2M active params vs
dense's ~34M), not the network, not wandb. Per-call overhead.

**Finding 3 — the production config was pessimal.** `GGL_NUM_GAMES=42` (84
players/rank) was chosen for DENSE, where it trades collect wall for GAE window
length. For a MoE whose forward is batch-free it is the worst possible choice: it
pays the 85ms 24 times to produce 2,016 rows. Collect wall is proportional to
TICKS, not players — so the right shape is few ticks x many players, the exact
opposite of the dense tuning.

**Finding 4 — fewer, BIGGER experts win at iso-parameter.** 64 experts x hidden
2560 is the same 1.006B total as 320 x 512 (both 335M/block x 3) but has 5x the
ACTIVE params (21M vs 4.2M) and is **2.4x FASTER** (35.2 vs 85.8ms). More FLOPs,
fewer problems, less time — the signature of an overhead-bound regime. 320 experts
at top-4 also means ~1 token per expert at batch 84, i.e. the grouped GEMM was
being asked to do 320 single-row problems.

Consequences for configuration (all iso-parameter, no architecture change):
- Maximize players/rank; minimize ticks/iteration subject to the GAE window.
- Prefer 64 x 2560 (or 32 x 5120) over 320 x 512.
- Global batch = ranks x players x ticks, so a big per-rank batch means FEWER
  RANKS for the same global batch. The 1B MoE cannot usefully occupy 96 GPUs at a
  200k-800k global batch; ~24 ranks is the right order.
- Remaining kernel-side headroom after that: 35ms/3 blocks = 11.7ms per MoE block
  vs dense's ~0.4ms per layer, so the grouped-GEMM/routing path is still ~30x off
  dense per call. CUDA graphs (merged, currently gated behind pipelining) are the
  next lever.

## LEVER RESULTS 2026-08-17 night (all four measured, two are duds)

**Lever 1 — pipelined collect: FIXED but WORTHLESS here.** The save-boundary wedge
is real and is fixed (`94d894d`): auto-save now runs in the barrier zone, where the
worker is already joined, instead of joining mid-iteration and suppressing the
re-kick. Verified at 8 nodes: saves complete, no wedge. But pipelining does NOT
speed this workload up — 5.4-6.5 s/iter with vs 5.2-5.9 s/iter without. The reason
is structural and should have been predicted: collect here is ~98% GPU inference
(`kern_s ~= infer_s` in every COLLECT line), and learn is GPU training, on the SAME
device. Overlapping two GPU-bound phases on one GPU cannot help. Pipelining only
pays when collect is CPU/env-bound. **Run with GGL_PIPELINED_COLLECTION=0.**

**Lever 2 — grouped-GEMM overhead: confirmed as the wall, NOT reachable by flag.**
`GroupScheduleMode::kDeviceOnly` makes CUTLASS's visitor linear-scan the problem
list per tile, which is the measured `0.2ms x experts` term. Turning the CUTLASS
LEARN path off drops fwdbwd 1.53 -> 0.40s (3.8x!) — but that number is a MIRAGE:
the EP token all-to-all lives only inside `MoERoutedFFNApply`, so the eager
fallback trains each owner's experts on 1/nL of the batch with no allreduce. Now
hard-failed at startup (`83efbf1`). The honest alternative (EP off + eager + full
4GB allreduce) measured 7.60 s/iter vs 4.95 — worse. So EP + CUTLASS-learn stays,
and the real fix is a visitor/scheduling rewrite. Still the biggest lever left.

**Lever 3 — CUDA graphs: a PESSIMIZATION on this platform.** 10.26 s/iter with
graphs vs 6.48 s without, same shape. Capture cost is not amortized (per-shape
scratch + shape churn). `GGL_CUDA_GRAPHS=0`.

**Lever 4 — consolidate onto fewer nodes: BLOCKED by memory.** Fewer nodes at a
fixed global batch means more rows/rank, and 8,352 rows/rank OOMs at ~30GB on a
32GB V100. 4,176 is at the practical limit, so the per-GPU efficiency win is not
available at 1B.

**THE ACTUAL BUG behind the idle fleet:** `MoEBlockImpl::_scratchByRows` was an
unbounded map keyed by row count, ~128MB/entry/block at learn batch. Nexto serves
15% of iterations and drives only some cars, so the inference batch size churns
and entries accumulated until CUDA OOM at ~31GB after ~120 iterations — which
killed all four hops of chain 4631804-07 identically. LRU-capped at 4 shapes
(`9e9be59`); verified allocated memory now FLAT at 20.63GB across a long run.
