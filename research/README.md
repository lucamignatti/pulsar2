# research/ — the offline measurement program

Everything here is **offline, CPU-only analysis**. Nothing in this tree touches
the GPU, the live trainer process, or `build/checkpoints_*` except by copying a
checkpoint directory out first. That separation is load-bearing: the trainer
usually runs live on the box's only GPU and rotates checkpoint dirs at any
moment.

```
research/
  reports/     the experimental record — one .md per study, with a status index
    archive/   superseded studies (steering-era planning + parked mechanisms)
  tools/       the Python toolkit: loaders, rollout harnesses, probes, evals
  results/     JSON outputs, keyed by checkpoint timestep
  data/        datasets + checkpoint caches (untracked, ~1GB, regenerable)
  .venv/       the analysis environment (untracked, ~1.3GB)
```

Start with [reports/README.md](reports/README.md) for *what has been measured*
and [tools/README.md](tools/README.md) for *how to run things*.

## Two traps to know before you trust any number here

**1. The offline loader currently reads a frozen run.** `tools/load_checkpoint.py`
resolves its default checkpoint root by walking a hardcoded candidate list that
does **not** include `checkpoints_resid` — the live run — so it falls through to
`build/checkpoints_5.0v3`, frozen at ~11.75B steps. Anything run without an
explicit `PULSAR_CKPT_ROOT` is analyzing a dead lineage. The loader also asserts
a 512-wide / 2-layer non-residual trunk, which the residual run does not match.
Both are recorded in [reports/DEAD_CODE_AUDIT.md](reports/DEAD_CODE_AUDIT.md) §5
and §15; neither is fixed. Fixing the shape assertion naively — rebuilding
residual nets as skip-free MLPs — would replay the h2-truncation bug class, so it
needs the same oracle verification that fix got.

**2. Offline results dated before 2026-07-19 used pre-activation `h2`.** The
loader dropped the trunk's trailing LeakyReLU, so every offline *behavioral*
number computed before the fix is invalid. In-trainer telemetry is unaffected —
the bug was in this toolkit, not the C++. Full incident record and the oracle
that verified the fix: [reports/H2_TRUNCATION.md](reports/H2_TRUNCATION.md).
Affected reports carry a banner.

## House rules (paid for in Elo, per the repo doctrine)

- **Measurement before machinery.** Every intervention needs a measurement
  convicting the problem, pre-registered success criteria, an automatic guard,
  and a revert path. Most reports here open with a pre-registration block dated
  *before* the run — keep that habit; it is what makes the record trustworthy.
- **Never starve the trainer.** Cap threads and nice everything:
  `OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 nice -n 19 ...`
  (sklearn ignores `torch.set_num_threads`).
- **Copy checkpoints before reading.** They rotate on ~10-minute windows; retry
  against the next-newest if files vanish mid-copy.
- **Canonicalize per row.** The obs is team-canonical (x,y negated for ORANGE).
  World-frame spatial labels silently ruin probes — the diagnostic is an
  x/y-vs-time asymmetry and R² ≈ −0.5.
- **Episode-grouped CV always.** Adjacent frames are near-duplicates.
- **Gate on causal/behavioral deltas, not decodability.** In-sample AUC of a
  512-d difference-of-means reads 1.0 at n≈100/class.
