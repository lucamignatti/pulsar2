# MOE_POLICY — DeepSeek-V3-style sparse policy (1B total / ~18M active)

**Status: DESIGN + BUILD LOG, started 2026-08-16 evening (user-directed).
Deliverable: architecture validated + throughput measured + early training curves
by 2026-08-17 noon.**

## Goal

A 1B-total-parameter policy with dense-net active compute (~18M params/token),
trained by the existing PPO recipe on AiMOS V100s. Capacity scales 60x while
per-step FLOPs stay at today's dense trunk level.

## Architecture (decisions locked for v1)

```
obs(230) -> dense embed 230->1024
  -> 6 x MoE block:  x + MoE(LN(x))
       router: sigmoid affinity, top-4 of 320 experts + 1 shared expert
               (DSv3 aux-loss-FREE balancing: selection bias b_e, +/-gamma by load)
       expert: 1024 -> 256 -> 1024 FFN (LeakyReLU), fine-grained
  -> LN -> policy head 1024 -> 90 logits
critic/value family: UNCHANGED dense (capacity goes to acting, not valuation)
```

- Params: experts 6x320x(2x1024x256) = 1.007B; shared 6x0.52M; routers ~2M;
  embed+head ~0.3M. Total ~1.013B.
- Active/row: (4 routed + 1 shared) x 0.52M x 6 + embed + head + routers read
  ~= 18M. In the requested 10-20M band, matching today's dense active compute.
- Router weights = normalized sigmoid among selected (DSv3); bias b_e used for
  SELECTION only, never weighting. gamma = 1e-3 per update.

## Why no custom CUDA kernels in v1 (measured-lens decision)

This platform's binding constraint is ~0.15-0.5ms CPU dispatch per tensor op
(measured, aimos memory), NOT GEMM throughput. So the dispatch design is
capacity-factor batched routing: per block, top-k assignments are bucketed to
[E, cap, d] and the whole expert bank runs as ONE bmm pair (+gather/scatter),
~18 ops/block regardless of E=320. Six blocks ~= 110 ops/forward vs ~50 dense —
expected <=2x collect wall, and every op is a tensor-core bmm. If profiling
convicts this path, the v2 kernel is a fused gather-grouped-GEMM-scatter
(CUTLASS 2.x grouped GEMM supports sm_70). Kernels only after conviction.

## Parallelizing the backward/optimizer across experts

- Forward/backward: pure differentiable ops (index_select / bmm / scatter_add)
  -> autograd handles the sparse backward exactly; only touched experts get
  nonzero grads. Data-parallel replication: 1B fp32 + Muon momentum + grads
  ~= 12GB, fits 32GB V100 (NOT the 16GB desktop 5080 — cluster-only at full
  size; smokes use a tiny config).
- Expert weights are 3D stacked params [E, out, in]. Muon extended: dim==3 =>
  BATCHED Newton-Schulz (the NS iteration is matmul-only, so the whole expert
  bank orthogonalizes in ~15 bmm ops per stack). This is the "parallelize the
  backpass through each expert" ask realized as batched math + the existing
  validated GGL_MUON_SHARD round-robin for multi-rank sharding of the stacks.
- Grad allreduce: existing bucketed path (12 big 3D tensors + small ones).

## Training plan (v1)

- GGL_MOE=1 gates the trunk swap in ExampleMain; everything else (rewards, PPO,
  opponent mix, cadence config) = the validated 7.1-cadence recipe.
- Bring-up ladder: CPU smoke (tiny: 4 blocks x 8 experts) -> 1-node throughput
  job vs dense baseline numbers -> 4-node cold start overnight.
- Panels: MoE/Expert Load Entropy, MoE/Drop Frac (capacity overflow), MoE/Bias
  Range. Load entropy collapse = router collapse (the classic failure).
- LR: Muon RMS-matching transfers Adam-tuned LRs across width; entropy
  coefficient does NOT transfer across net sizes (6.1 lesson) — watch Policy
  Entropy from birth; expect one entropyScale correction.

## Risks (honest)

Router collapse at RL reward noise; fp16 router scores on V100 (compute
affinities fp32); capacity drops under bursty routing (shared expert is the
safety net); 1B-param checkpoint size (~8GB with optim state) x rotation — set
checkpointsToKeep low; offline tooling (load_checkpoint.py) will not parse MoE
checkpoints without work (explicitly out of scope for noon).

## Build log — the segfault hunt (2026-08-16 overnight)

Symptom: trainer segfaults on iteration 1 with GGL_MOE=1, at the gap-sensor's
`models["shared_head"]` lookup (Learner.cpp:4134) — CONFIRMED PORTABLE (desktop
sm_120 stack AND cluster V100/torch-2.1, job 4630856). Hard-established facts:
- The MoE block itself is clean: fwd/bwd/clone/bias-update selftests pass at 96
  rows, 25k-row 3-block stacks, and GGL::Model+Sequential GPU phase (30 reps,
  both contiguous and gathered inputs). `GGL_MOE_SELFTEST=1`.
- The defensive forward rewrite (static shapes, no bincount/masked_select/
  index_copy; scatter+trash-row bucketing) did NOT change the crash → the
  forward's op choice was never the cause.
- gdb w/ RelWithDebInfo: fault at the map lookup itself; map HEADER healthy
  (15 nodes), $rdi = the "shared_head" key string → crash walking the rb-tree →
  a map NODE is stomped: heap corruption with delayed detonation, planted
  between the last good lookup (value-pred) and the gap block.
- Dense control on same binary: passes the whole region (then hits the known
  desktop-only cusolver SVD abort in Plasticity::EffectiveRank — separate,
  cluster unaffected).
- PolicySlots/LinearLayers layer-count theory: dead (only Plasticity survives,
  Learn-side, not reached).
Current instrument: ASan build (build-asan) to catch the stomping write.
