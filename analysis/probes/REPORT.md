# Phase 0, Detector #1: ball-landing linear probes on the live 3.1 checkpoint

**TL;DR.** The landing concept is decodable everywhere, but the trunk does NOT
expose it linearly much beyond what raw obs already give: ridge on raw obs reads
landing (x, y) at R² 0.92/0.94 by itself. The network's added value is confined to
exactly the frames where the concept is genuinely nonlinear — wall/ceiling bounces
(17.7 % of airborne frames), where trunk layer 1 beats raw obs by ~20 % (median
1007 vs 1257 uu) — and to time-to-land, which grows steadily more decodable with
depth (R² 0.62 → 0.66 → 0.70). Meanwhile raw ball height is progressively
*discarded* (passthrough R² 1.00 → 0.975 → 0.924). Verdict for the workspace
program: this looks like automatic processing that computes just enough
task-relevant physics inline, not a consolidated, many-consumer-readable "where
will the ball land" variable. That is the Phase 1 motivation, now with a baseline
to beat.

## Setup

- **Checkpoint**: `checkpoints_3.1/1461076992` (~1.46 B steps, iteration 6558,
  Rating/1v1 ≈ 688, reach InfoNCE acc EMA ≈ 0.63), copied out of the live folder
  before reading. The C++ `torch::save` `.lt` archives load directly with
  `torch.jit.load`; each Sequential is rebuilt eagerly and shape-verified against
  the architecture (trunk 109→512→512, policy →90, phi 602→…→128 — all exact).
  Obs are raw (standardizeObs=false in this run), so activations correspond to
  unnormalized AdvancedObs.
- **Data**: 100 k player-frames of self-play (stochastic sampling, DefaultAction
  masking, all 90 actions exercised), 16 arenas, 105 episodes, reset mix 25 %
  kickoff / 75 % RandomState(randBall, randCar, air) — the training distribution
  minus drill setters. tickSkip 4 / actionDelay 3 replicated exactly (3 ticks old
  controls → set new controls → 1 tick → build obs).
- **Obs parity**: AdvancedObs ported float-for-float (coefficients, pad-timer
  blending, canonical pad order, team inversion, local frames). End-to-end
  behavioral check: 20/20 kickoffs reach the ball (median 3.1 s, closing speed
  ≈ 1150 uu/s) — a scrambled obs port would not drive a competent kickoff.
- **Labels**: for the 24,694 frames with ball z > 300 uu, the ball (pos, vel,
  angVel) is cloned into a car-free arena and stepped to touchdown (z ≤ 111 uu,
  cap 6 s). 100 % landed; median flight 1.23 s. Labels are **team-canonicalized**
  (x, y negated for orange rows) to match the obs frame — see Surprises.
- **Probes**: RidgeCV (α ∈ 10⁻²…10⁵, chosen per fold on train data), 5-fold CV
  **grouped by episode** (adjacent frames are near-duplicates; ungrouped folds
  would leak). Identical folds for every source; targets standardized per fold and
  predictions inverse-transformed.

## Results (out-of-fold, 24,694 airborne frames)

| source | dim | R² x_land | R² y_land | R² t_land | median err (uu) | bounce | direct |
|---|---|---|---|---|---|---|---|
| ridge raw obs      | 109 | 0.917 | 0.938 | 0.619 | **555** | 1257 | **499** |
| ridge trunk h1     | 512 | **0.921** | **0.956** | 0.663 | 618 | **1007** | 564 |
| ridge trunk h2     | 512 | 0.904 | 0.934 | **0.698** | 810 | 1163 | 753 |
| ridge reach_phi    | 128 | 0.908 | 0.930 | 0.663 | 772 | 1091 | 714 |
| MLP raw obs (ctrl) | 109 | 0.878 | 0.905 | 0.649 | 882 | 1187 | — |
| ballistic formula (ctrl) | — | — | — | — | 35 | 1925 | 24 |

("bounce" = the 17.7 % of frames where the closed-form no-bounce ballistic
extrapolation misses by > 500 uu — i.e. where the concept is genuinely nonlinear;
"direct" = the rest. Full values in `results/metrics.json`.)

Controls:

- **Shuffled labels**: R² ∈ [−0.0014, −0.0001] on every source — no leakage. ✓
- **Passthrough (current ball z)**: raw obs 1.0000 (by construction), trunk h1
  0.9752 (extraction verified ✓), trunk h2 0.9242 (a finding, see below).
- **Ballistic closed-form** (physics, no learning): median 35 uu overall — on
  direct frames the concept is essentially solvable; every learned reader is far
  from that ceiling.
- **MLP on raw obs**: 882 uu median — *underfit* (12 k train rows, 300 iters,
  early stopping); it is a weak reference, not an upper bound. The ballistic
  control is the honest "computable at all" witness.

## Reading

1. **The headline gap (trunk ≫ raw-obs-linear) did not materialize.** Raw
   AdvancedObs already supports R² ~0.93 linear decoding of the landing point,
   because for 82 % of airborne frames landing ≈ pos + vel·t(quadratic), which a
   109-feature linear map approximates well within one episode distribution. Any
   claim that "the network computes landing" has to be made on the bounce slice.

2. **On the bounce slice, the trunk does add something.** Trunk h1's linear
   readout (1007 uu) beats every obs-side reader there: raw-obs ridge (1257), the
   MLP control (1187), and the no-bounce physics formula itself (1925). The
   network carries bounce-adjusted landing information that is not linearly
   available in the obs — real, but modest: the concept is partially computed,
   not cleanly exposed.

3. **Depth trades raw state for predicted dynamics.** Down the trunk, current
   ball z fades (passthrough 1.000 → 0.975 → 0.924) while time-to-land sharpens
   (0.619 → 0.663 → 0.698). Landing *position* peaks at h1; landing *time* peaks
   at h2. The trunk is transforming toward behaviorally-proximal variables
   ("when do I need to be there") rather than maintaining a global physics
   forecast — exactly the "automatic processing" profile from the workspace
   paper: task-consumed information, not a report-ready representation.

4. **reach_phi is surprisingly dense.** At 128-d (4× smaller than the trunk taps,
   L2-normalized, trained only on InfoNCE reachability) it decodes landing nearly
   as well as h2. The reachability objective apparently concentrates
   ball-trajectory information — worth remembering when Phase 2 looks for a
   compact concept space.

5. **Phase 1 has a concrete target.** A many-consumer aux head predicting
   (x, y, t)_land should, if the workspace hypothesis is right, push the bounce-
   slice linear readability of the trunk well below the current 1007 uu while the
   ballistic control (35 uu) shows how much headroom exists. Measurement first:
   this table is the pre-intervention baseline.

## Surprises / gotchas (read before building on this)

- **Team-frame labels.** AdvancedObs inverts x, y for the orange player, so the
  network's world IS the team-canonical frame. World-frame labels silently
  destroy every probe (first run: R² ≈ −0.5 on x/y for all sources, t_land fine —
  that asymmetry is the diagnostic). Every future detector with spatial targets
  must canonicalize per-row.
- **Trunk h2 passthrough is only 0.92.** Not an extraction bug (h1 = 0.975 from
  the same forward pass); layer 2 genuinely discards ~8 % of current-ball-height
  variance. Sanity gates for future detectors should gate on h1.
- **Checkpoint dirs vanish mid-copy** (trainer rotation); `load_checkpoint.py`
  retries on next-newest. Happened twice during this session.
- **sklearn ignores torch thread caps** — the MLP control grabbed every core
  until re-run with `OMP_NUM_THREADS=4` etc. Keep the env caps or the probe run
  competes with the trainer's env workers.
- **Multi-output target scales.** Without per-fold target standardization the
  squared loss ignores t_land (σ 0.9 s vs 3000 uu) — the MLP scored R² −0.14 on t
  until fixed.

## Caveats / divergences from the trainer

- pip RocketSim 2.2.1 vs in-repo 2.1.1 (minor physics deltas possible); fp32
  inference vs the trainer's bf16 collection; 3 post-reset ticks run zeroed
  controls where the C++ EnvSet carries pre-reset controls.
- reach_phi is probed with the sampled action's one-hot (it is a state-action
  embedding); a state-only view would marginalize over valid actions.
- Labels are ball-only physics: the probed concept is "unimpeded landing point";
  a real opponent may intercept first.
- Rating context: this checkpoint is early in the 3.1 run (~1.46 B steps,
  Rating ≈ 688 vs the lineage's ~1190 peak). Re-running the pipeline on later
  checkpoints (it is fully scripted) turns this into a concept-emergence curve —
  arguably the most interesting follow-up plot.

## Artifacts

- `results/metrics.json` — all numbers above
- `results/plots/r2_by_source.png`, `pred_vs_true.png`, `landing_error_hist.png`
- `data/dataset.npz` (100 k frames: obs, h1, h2, phi, action, physics, episode),
  `data/labels.npz` (landing x, y, t + validity)
- Pipeline: `load_checkpoint.py` → `collect_dataset.py` → `label_landing.py` →
  `train_probes.py` (see README.md for env + thread-cap invocation)
