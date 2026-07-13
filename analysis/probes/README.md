# Phase 0 probes: linear detectors on frozen checkpoints

Offline CPU-only analysis. Never touches the GPU or the live trainer; checkpoints are
copied out of `build/checkpoints_3.1/` before reading (the trainer rotates dirs at any
moment).

## Environment

Python venv (torch-cpu, numpy, scikit-learn, matplotlib, RocketSim pip bindings
v2.2.1 — see `requirements.txt`). The run used a scratch venv:

```
python3 -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
.venv/bin/pip install numpy scikit-learn matplotlib RocketSim
```

RocketSim needs collision meshes; scripts point it at `../../build/collision_meshes`.

## Pipeline (run in order)

1. `load_checkpoint.py` — copy newest checkpoint, jit-load the `.lt` files, rebuild
   as eager `nn.Sequential`, shape-verify against the architecture. Also a module
   (`load_latest()`, `PulsarPolicy` with activation taps).
2. `collect_dataset.py` — 100k player-frames of self-play in RocketSim
   (`PROBE_FRAMES` env to override). Ports AdvancedObs + DefaultAction + the
   tickSkip-4/actionDelay-3 step split. → `data/dataset.npz`
3. `label_landing.py` — ball-landing ground truth for airborne frames (car-free
   arena, step until touchdown, 6 s cap). → `data/labels.npz`
4. `train_probes.py` — ridge probes (episode-grouped 5-fold CV) on raw obs /
   trunk h1 / trunk h2 / reach_phi + controls (shuffled labels, MLP-on-obs,
   ballistic formula, ball-z passthrough). → `results/metrics.json`, `results/plots/`

Run everything thread-capped and niced so the live trainer keeps the box, e.g.
`OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 nice -n 19 .venv/bin/python -u train_probes.py`
(sklearn ignores `torch.set_num_threads`; without the env caps the MLP control
grabs every core).

Findings and caveats: `REPORT.md`.
