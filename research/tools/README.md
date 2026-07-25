# tools/ — the offline analysis toolkit

Offline CPU-only. Never touches the GPU or the live trainer; checkpoints are
**copied out** of `build/checkpoints_<run>/` before reading, because the trainer
rotates dirs on a ~10-minute window.

Reports live in [`../reports/`](../reports/); JSON output lands in
[`../results/`](../results/); datasets and checkpoint caches in `../data/`
(untracked).

## Environment

```bash
cd research
python3 -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
.venv/bin/pip install -r tools/requirements.txt      # numpy, sklearn, matplotlib, RocketSim==2.2.1
```

Run everything from `research/`, thread-capped and niced so the live trainer keeps
the box (sklearn ignores `torch.set_num_threads`; without the caps the MLP control
grabs every core):

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  nice -n 19 .venv/bin/python -u tools/train_probes.py
```

RocketSim needs collision meshes; scripts resolve them at `<repo>/build/collision_meshes`.

## ⚠ The loader points at a frozen run

`load_checkpoint.py` picks its default checkpoint root from a hardcoded candidate
list that does **not** include `checkpoints_resid` (the live run), so it falls
through to `build/checkpoints_5.0v3` — frozen at ~11.75B steps. It also asserts a
512-wide / 2-layer **non-residual** trunk, which the residual run does not match.

Until that is fixed, pass the root explicitly:

```bash
PULSAR_CKPT_ROOT=/path/to/copied/checkpoint_dir .venv/bin/python tools/<script>.py
```

Both issues are written up in [`../reports/DEAD_CODE_AUDIT.md`](../reports/DEAD_CODE_AUDIT.md)
§5 and §15. A naive shape fix that rebuilds residual nets as skip-free MLPs would
replay the h2-truncation bug class — it needs the same oracle verification that
fix got ([`../reports/H2_TRUNCATION.md`](../reports/H2_TRUNCATION.md)).

Two scripts additionally hardcode dead lineages: `kd_curve.py` → `checkpoints_3.1`,
`rot_dynamics.py` → `checkpoints_4.0`.

## The pipeline (run in order)

1. `load_checkpoint.py` — copy newest checkpoint, jit-load the `.lt` files, rebuild
   as eager `nn.Sequential`, shape-verify. Also a module: `load_latest()`,
   `PulsarPolicy` with h1/h2/phi activation taps.
2. `collect_dataset.py` — 100k player-frames of self-play (`PROBE_FRAMES` to
   override). Ports AdvancedObsPadded + DefaultAction and the **tickSkip 8 /
   actionDelay 0** step split (15 Hz — the 5.0 dynamics). → `../data/dataset.npz`
3. `label_landing.py` — ball-landing ground truth for airborne frames (car-free
   arena, step to touchdown, 6 s cap). → `../data/labels.npz`
4. `train_probes.py` — ridge probes, episode-grouped 5-fold CV, on raw obs / trunk
   h1 / h2 / reach_phi, plus controls (shuffled labels, MLP-on-obs, ballistic
   formula, ball-z passthrough). → `../results/metrics.json`, `../results/plots/`

Findings: [`../reports/REPORT.md`](../reports/REPORT.md).

> Offline datasets interleave 2 players per step; in-trainer trajectories append
> each player's episode row-contiguously. This indexing mismatch has caused bugs —
> check which layout you are in.

## Measuring progress honestly

**Do not read `Rating/1v1` as absolute progress.** It is measured against a
rolling pool of recent selves that drifts with the policy; measured overstatement
is ~6x (claimed +187 Elo over 8.6B steps; real match-play gain +31). Use the
fixed-anchor battery:

```bash
../tools/archive_anchor.sh [--list|--seed DIR]   # permanent spaced anchors (systemd: pulsar-anchor.timer)
.venv/bin/python tools/anchor_battery.py         # real match-play Elo vs those anchors
.venv/bin/python tools/match_play_eval.py A B    # one pairing, match rules (kickoff -> goal)
```

Protocol note that matters: `compare_checkpoints.py` cross-play runs on the
**trainer reset mix** (drill/random spawns) and biases *against* the current
policy — it read 48% where match play read 54% on the same pair. For "is it
actually better", use match play.

## Layout

The toolkit is a flat module directory: scripts import each other directly
(`from load_checkpoint import ...`), which works because Python puts a script's
own directory on `sys.path`. Load-bearing modules, by importer count:

| Module | Importers | Role |
|---|---|---|
| `load_checkpoint.py` | 43 | checkpoint load + eager rebuild + activation taps |
| `steer_team.py` | 27 | N-player rollout harness (1v1/2v2/3v3), trainer-parity reset mix |
| `team_decline_probe.py` | 14 | shared decline-reading + clustering utilities |
| `collect_dataset.py` | 14 | self-play collection |
| `steer_test.py` | 10 | 1v1 rollout harness |
| `advanced_obs.py` | 8 | AdvancedObs / AdvancedObsPadded, action table + masking |
| `fear_decomp.py` | 7 | goal-side landing sims, bank utilities |
| `label_landing.py` | 6 | ball-landing ground truth |
| `steer_v2.py` | 5 | possession derivation |
| `team_gate_validate.py` | 5 | `TeamGatedPolicy` |
| `frontier_validate.py` | 3 | `roll_and_judge`, reconstruction |

Everything else is a single-experiment driver, most with a matching report in
[`../reports/`](../reports/). Steering-era drivers still import the harnesses
above, so they are kept even though `steering.alpha = 0` on HEAD.
