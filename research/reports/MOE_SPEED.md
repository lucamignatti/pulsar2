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

## Stage 2 — NCCL expert parallelism (user-directed)

Each of a node's 6 V100s owns E/6 = ~21 experts per block (fixed assignment);
collect and learn route tokens to expert owners by NCCL all-to-all, results
return the same way. Bench says intra-node comm is ~free. Design points:

- **Collect**: after the plan kernel, counts per (expert→rank) are known on
  device; a2a token exchange (fp16 rows), owner runs its experts' grouped GEMM,
  a2a back. Per-rank expert weights shrink 6× → the 32GB ceiling lifts ~6× on
  expert params: the **1B-total model fits** (a 1B/6 ≈ 170M expert slice + dense
  trunk + optimizer per GPU).
- **Learn**: dispatch stays the same in backward — each owner accumulates grads
  ONLY for its experts (no allreduce over expert params at all; router/dense
  stay data-parallel). This replaces the current "every rank holds every expert +
  full allreduce" scheme; expert-grad traffic drops to the token a2a itself.
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

## Execution state

- [x] Bench numbers read into the record (this doc + memory).
- [x] Graphs desktop-kept / cluster-dead verdict, fleet pinned =0.
- [ ] Stage 1 build: MoEKernels.{h,cu}, CMake gate, MoE.cpp fast path, selftest
      parity, 1-node speed measure, fleet flip.
- [ ] Stage 2 design review + a2a microbench inter-node.
- [ ] Stage 3 pre-registration.
