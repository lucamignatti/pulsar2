# inject2d — injection-technique testbed

Fast offline testbed for comparing **actuation channels** for the ladder's
optimism signal (the critic side is taken as validated; what is under test is
how its signal reaches the policy). Motivated by the injection critique of
`research/reports/GEOMETRIC_CRITIC.md`: the production channel (potential
difference → advantage add, σ-matched, clamped) has a narrow therapeutic
window, injects the slope of a near-flat field, is priced by no baseline, and
is state-level where the wall is action-level.

## Environment: Aerial2D (`env.py`)

2-D car-and-ball distillation of the acquisition wall. Ground play (drive,
jump-touch low balls) is easy and shaped toward; high balls need jump → an
unbroken chained boost run (boost is ineffective once upward momentum is lost)
→ x-drift alignment, against boost cost + crash penalty + time cost. Reward has
**no aerial-specific term**. Metrics logging-only: `touch/air/hi` per 1k steps.

Calibration (measured):
- random-valid policy: touch 1.9, air 1.3, hi **0.072** per 1k steps
- scripted perfect vertical commit: touches a bz=7.0 ball (reach ≈ 7.2)
- vanilla PPO 3 seeds @ 2M: 2 seeds wall cleanly (touch plateaus ~3.5, hi
  collapses to ~0.006 — *below random*, i.e. avoidance is learned), 1 seed
  ignites via entropy luck. The graded outcome mirrors production's
  serendipity-gated ignition; arms are therefore scored on **ignition rate
  across 8 seeds** plus bucket rates.

## Ladder (`ladder.py`)

Testbed-scale replica of the four rungs: V (PPO critic), V_exp (return
expectile τ=0.8), V† twins (one-step expectile TD τ=0.75, min-in-target,
target nets, clamp), V_geo (σ-net Gaussian-NLL + r̂-net + HJB field, uniform
reservoir over all data, σ-scale 0.5). Optional Q†(s,a) per-action head
(`aprior` arm). Reservoir feed uses true pre-reset next-obs, so no
boundary-teleport pairs enter Σ (the bug flagged against production is done
right here).

## Arms (`train.py`)

| arm | channel | idea |
|---|---|---|
| ppo | — | wall baseline |
| pbrs | advantage add | production replica: unit-normed Φ_mix, β=0.04, clamp ±3σ |
| pbrs_raw | advantage add | same, raw un-normalized fields at nominal β — dose-confound regime |
| servo_raw | advantage add | pbrs_raw + closed-loop β on the **delivered** dose ratio |
| entgate | entropy coeff | per-state entropy raised by headroom (only up; C5) |
| temp | collection dist | temperature bump in high-H states |
| intr | own GAE stream | potential-diff as intrinsic reward, own critic, γ_int=0.95, std-matched blend |
| sil | extra loss | headroom-gated self-imitation on realized conversions (R > V_exp, high H) |
| aprior | collection dist | Q† per-action optimism prior as logit bias |
| align | own GAE stream | cos(Δs, Σ²∇V_geo) alignment intrinsic |
| kstep | advantage add | k=8-step chunked potential differencing (SNR fix) |
| sparse | advantage add | injection only in top-20% headroom rows (SNR fix) |

## Run

```bash
python3 sweep.py --arms ppo pbrs entgate intr sil aprior --seeds 0 1 2 3 4 5 6 7 --steps 3000000
python3 analyze.py runs
```

~30–50 s per 3M-step run on an M4 Pro (12 threads, 6 jobs × 2 torch threads).

## Caveats / what does NOT transfer

This tests the *relative* merit of channels on a wall-shaped task, at toy
scale, single-agent, γ=0.99. It cannot see: opponent nonstationarity, the
γ=0.9994 discount amplification, trunk co-adaptation (all toy nets are
independent), or Elo-level costs of injection (no opponent). Winners here are
candidates for the RocketSim testbed of the paper, not for direct deploy.
