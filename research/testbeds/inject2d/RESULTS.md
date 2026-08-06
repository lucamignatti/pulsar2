# inject2d results — 2026-08-05

All numbers: 8 seeds/arm, M4 Pro, ~50–70 s/run. Compare only within a batch
(the obs gained a dim between batch 1 and 2, which reshuffles seed luck).

## Batch 1 — injection channels (3M steps, static balls)

| arm | ignited | ign step | hi q4 | rew_f |
|---|---|---|---|---|
| sil | 8/8 | 0.99M | 5.15 | 2.83 |
| sil_ent | 8/8 | 1.25M | 5.39 | 2.88 |
| entgate | 8/8 | 1.41M | 4.68 | 2.55 |
| sil_nogate | 8/8 | 1.65M | 5.00 | 2.76 |
| align | 8/8 | 2.20M | 4.12 | 2.28 |
| kstep | 6/8 | 1.20M | 3.37 | 2.03 |
| ppo | 6/8 | 2.18M | 2.96 | 1.78 |
| sparse | 4/8 | 1.25M | 3.03 | 1.86 |
| sil_pbrs | 7/8 | 1.10M | 1.45 | 1.28 |
| pbrs (prod replica) | 5/8 | 1.11M | 1.04 | 1.08 |
| servo_raw | 5/8 | — | 0.88 | 0.97 |
| intr | 2/8 | — | 0.59 | 0.84 |
| pbrs_raw | 3/8 | — | 0.58 | 0.82 |
| decay | 4/8 | — | 0.46 | 0.64 |
| temp | 0/8 | — | 0.14 | −0.03 |

Findings:
1. **The potential/advantage channel ignites fast then CAPS** (per-seed: pbrs
   ignites earlier than ppo on every shared seed, converges to ~1/3 its hi).
   Adding SIL on top does not rescue it (sil_pbrs 1.45): swap, don't add.
2. **SIL and entgate win**; they compose (sil_ent) with no interference.
   The ladder's H gate = ignition speed (sil 0.99M vs sil_nogate 1.65M).
3. **Pricing a potential-diff neutralizes it** (intr): a potential difference is
   predictable from state, so its advantage under an accurate intrinsic critic
   → 0. "Priced" and "potential-based" are mutually canceling for seek.
4. **Sampler-side modification fails under PPO IS** (temp 0/8, aprior mixed).
5. Dose servo works both directions (raw fields under-delivered 4x here);
   kstep/sparse (SNR fixes) salvage some of the channel but don't beat sil.

## Batch 2 — gap closure (max-vs-avg = relu(V_exp − V), 3M steps)

| arm | ignited | ign step | hi q4 | rew_f |
|---|---|---|---|---|
| sil | 8/8 | 1.34M | 5.54 | 3.02 |
| gap_sil (elevation) | 8/8 | 1.38M | 5.03 | 2.77 |
| gap_pre (priced potential) | 6/8 | 1.20M | 4.08 | 2.67 |
| gap_post (advantage add) | 8/8 | 1.52M | 4.02 | 2.31 |
| ppo | 6/8 | 1.50M | 3.85 | 2.27 |
| gap_rew (plain penalty) | 0/8 | — | 0.07 | **−1712** |

Findings: the plain penalty is a self-referential doom loop (penalty lowers
returns → V falls → gap widens → penalty grows). The same quantity works
through non-self-referential channels; closure-by-elevation ≫ closure-by-
avoidance; nothing beats headroom-gated SIL.

## Batch 3 — meta-transfer (4M steps, frontier shift at 2.4M)

Phase A: static balls. Phase B: high-only falling balls (timed intercept).
silobs = policy obs carries tanh(V†−V_exp), tanh(V†/5); silobs_lp adds
learning-progress gating (relu(H_old − H_now), ladder snapshots every 10 it).

| arm | A-ign (M) | shift dip | recover-80% (iters) | B air final |
|---|---|---|---|---|
| sil (scaffold only) | 1.13 | 51% kept | 71 | 10.25 |
| silobs | 1.11 | 67% kept | **9** | 10.50 (1 seed collapsed: 0.3) |
| silobs_lp | 1.44 | 63% kept | 21 | **12.55** (worst seed 8.8) |

Findings: **the meta-transfer signature is real** — self-observing policies
lose less skill at the shift and recover 3–8× faster; LP-gating trades a
slower FIRST ignition for the best and most uniform post-shift performance
(no collapsed seeds). The silobs s6 collapse is the nonstationary-self-input
hazard surfacing exactly when self-estimates go stale (at the shift).
Caveats: the fully-new conduct (high intercept) ignited in no arm within the
1.6M post-shift budget — the differential is on skill recovery/transfer, not
from-scratch second-frontier acquisition; needs a longer phase B to test that.

## What transfers to production (claims, not proofs — RocketSim testbed next)

- Replace the Φ_mix advantage injection with headroom-gated SIL + the entropy
  gate (both hot-swappable: no new params, learn-pass only).
- The self-obs + LP design is a fresh-lineage feature (obs width) with a
  deployment story: prev-step features (prevAction idiom), trunk-coupled
  heads, mode-dropout for a clamped exploit mode at eval/deploy.
- Do NOT ship: plain gap penalties (self-referential), collection-time
  temperature, priced seek potentials, adding SIL on top of the live mix.
