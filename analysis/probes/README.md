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

## Honest progress measurement (2026-07-19)

**Do not read `Rating/1v1` as absolute progress.** It is measured against a
rolling 800M-step pool of recent selves that drifts with the policy; measured
overstatement is ~6x (claimed +187 Elo over 8.6B steps; real match-play gain
+31). Use the fixed-anchor battery instead:

```
tools/archive_anchor.sh [--list|--seed DIR]   # permanent spaced checkpoint anchors
                                              # (systemd --user timer: pulsar-anchor.timer)
analysis/probes/anchor_battery.py             # real match-play Elo vs those anchors
analysis/probes/match_play_eval.py A B        # one pairing, match rules (kickoff->goal)
```

Protocol note that matters: `compare_checkpoints.py` cross-play runs on the
TRAINER reset mix (drill/random spawns) and biases AGAINST the current policy —
it read 48% where match play read 54% for the same pair. For "is it actually
better", use match play.

Incident record: `H2_TRUNCATION.md` (the offline loader dropped the trunk's
trailing LeakyReLU — every pre-2026-07-19 offline behavioral result used
pre-activation h2; fixed and oracle-verified). League fix spec awaiting review:
`LEAGUE_ANCHORS.md`.
