# TEAM_GATE — turning the team-whiff-tax finding into a lever (offline validation)

**Pre-registered 2026-07-15, BEFORE the sweep ran.** Checkpoint **27550023244**
(frozen copy, `build/checkpoints_4.0-interp2/`). Script: `team_gate_validate.py`.

## The finding being converted (INTERP_SWEEP2 Stage B)

In 2v2 the trunk decodes "I am the best-placed teammate" (AUC 0.95 vs obs 0.70)
and "my teammate is not going" (belief 0.14 on best-placed decline rows), yet
56% of feasible balls go collectively unpursued → team whiff tax (incentive
mis-pricing, the 1v1 story at team scale).

## Lever selection rationale (why steering-targeting, not the alternatives)

- **Frontier drills already run** (per-mode collective-decline mining →
  FrontierDrillState on the practice slices, useFrac 0.35): the reps lever is
  in place; no evidence yet that its dose is the binding constraint.
- **Credit-side surgery is banned/forbidden**: stage-2 termination collapsed
  twice (critic aliasing); gated PBRS potentials break telescoping; a dedicated
  practice-value head is a big update-side change with no offline testbed.
- **Team steering runs UNTARGETED**: the 1v1-transfer direction steers any
  steered-arena row inside the rho band — including the NOT-best-placed
  teammate. Steering the wrong player toward commitment trains double-commits
  (the exact failure the per-mode possession gate exists to catch) and dilutes
  the dose where it pays. B1 says the needed targeting signal is computable
  from information already available at collection time. This is the smallest
  intervention that converts the new finding into expected performance, and it
  is fully offline-testable with the existing harness.

## Arms (2v2, 400k rows each, same eval seed; direction = 1v1-derived commitment
transferred, the E3 recipe, derived fresh at this checkpoint)

| arm | alpha | per-row team gate (on top of the live rho band) |
|---|---|---|
| A0 | 0 | — (baseline) |
| A1 | +0.5 | none (live config's mechanism) |
| A2 | +1.0 | none (dose headroom, measurement only) |
| G1 | +0.5 | proximity: steer only the teammate strictly closer to the ball |
| G2 | +0.5 | rho-gap: steer only if own contact-rho ≥ teammate's |
| G3 | +1.0 | proximity (does targeting make the higher dose safe?) |

Primary metric: **teamWon** (self or teammate first touch) on feasible airborne
readings, episode-cluster bootstrap SE. Secondary: NONE (collective-decline)
rate. Canaries: goals/ep and touch ratio within −10%/−20% relative of A0,
kickoff first-touch < 5s, in-air ratio not exploding (> A0 + 15pp absolute).

## Pre-registered decision rules

1. **Transfer-alive bar**: teamWon(A1) − teamWon(A0) ≥ +2pp with the gap
   ≥ 1.5× the cluster SE of the difference. FAIL → team steering is
   saturated/dead at this checkpoint; NO deploy of anything from this sweep;
   record and stop (drills/credit become future candidates on new evidence).
2. **Gating bar (the deploy decision)**: best of {G1, G2} beats A1 by ≥ +1.5pp
   teamWon at ≥ 1.5× SE_diff, with canaries clean and NONE not higher than A1.
   PASS → implement THAT gate in the trainer for team modes only, α unchanged
   at 0.5, existing guards (per-mode possession gates, rating latch, branch
   backup ritual) covering it. One lever: the gate, nothing else.
3. **Dose arms (A2, G3) are measurement only.** Offline rollouts cannot see the
   update-side clipping ratchet that made α=1σ bleed live at matched
   offline-positive readings; no α change is licensed by this experiment.
4. **Underpowered but sign-correct** (bars missed, point estimates right):
   one repeat of the two deciding arms at 800k rows; still short → no deploy.

## Results (2026-07-15; raw: `results/team_gate_27550023244.json`)

Direction: 223 matched 1v1 pairs, sigma 6.16. All arms 400k rows, same eval seed.

| arm | teamWon ± SE | NONE | notes |
|---|---|---|---|
| A0 baseline | 12.1% ± 2.0% | 79.8% | |
| A1 +0.5 ungated | 12.7% ± 1.7% | 79.5% | the live mechanism |
| A2 +1.0 ungated | 12.1% ± 1.6% | 79.9% | |
| G1 +0.5 prox-gated | 12.1% ± 1.8% | 78.5% | |
| G2 +0.5 rho-gap-gated | **6.7% ± 1.7%** | 83.4% | actively harmful, ~2.3σ below A1 |
| G3 +1.0 prox-gated | 12.3% ± 1.7% | 78.9% | |

**VERDICT: Bar 1 (transfer-alive) FAILS — NOTHING FROM THIS SWEEP DEPLOYS.**
A1 − A0 = +0.6pp (SE_diff ≈ 2.6pp) against a ≥ +2pp bar; not a power miss —
the point estimate itself is under the bar. The transfer effect measured
+3.4pp at 12.4B has decayed to ~0 at 27.5B (rating 2v2 ~290 → ~410 in
between): the same positive-side saturation that hit 1v1 (STEERING_PHASE0_40
0a), one era later. Canaries fine everywhere; kickoff/touch/goals flat.

Additional recorded findings:
- **Rho-gap gating is anti-correlated with useful targets** (G2): steering only
  the teammate with the higher contact-rho HALVES teamWon. The uniform-action
  capability read evidently selects the already-committed/closer player whose
  behavior steering then distorts. Do not revisit without new evidence.
- Proximity gating is behaviorally neutral (G1≈A0≈A1): targeting can't rescue
  a dose that no longer moves anything.
- NONE stays ~79% across ALL arms: the residual collective decline is NOT
  movable along the 1v1-commitment axis at this checkpoint, despite the whiff
  tax provably persisting (INTERP_SWEEP2 B3). The residual is a different axis
  (team "who-goes") or not linearly steerable at all.

## Post-hoc supplement (exploratory, licenses nothing)

`team_gate_posthoc.py`, same eval seed: S1 = the trainer's live per-mode
2v2-OWN derived direction @ +0.5 ungated (is live team steering doing anything
acutely?); S2 = best-placed-conditioned WON-vs-NONE direction @ +0.5
prox-gated (the INTERP_SWEEP2-shaped candidate).

Results (`results/team_gate_posthoc_27550023244.json`):

- **S1: 12.3% ± 2.9%, NONE 78.2%** — indistinguishable from A0/A1. The live
  2v2-own direction is ALSO acutely dead: live team-mode steering is currently
  cosmetic (its possession gate presumably duty-cycles near zero — consistent
  with nothing being lost when it re-derives).
- **S2: 19.7% ± 4.0%, NONE 73.9%** (+7.6pp over A0, ~1.7σ; canaries clean;
  414 matched pairs vs 175 for the unconditioned pool — conditioning on
  best-placed CLEANS the contrast: unconditioned NONE rows include "correctly
  deferred to the better-placed teammate", which is not a decline at all and
  dilutes/poisons the direction). cos(v_bp, v_own) = +0.91 — a small angular
  correction with a large behavioral difference, same pattern as the v1→v2
  derivation fix in 1v1.

S2 is exploratory (one seed, direction+gate confounded, 1.7σ). It licenses a
CONFIRMATION, not a deploy.

## Confirmation round (pre-registered 2026-07-15 BEFORE running; script
`team_gate_confirm.py`)

Fresh derivation rollout (new seed — this tests the RECIPE, not one lucky
vector), fresh eval seed, 800k rows/arm:

| arm | direction | gate |
|---|---|---|
| B0 | — | — |
| B1 | 2v2-own (live recipe) +0.5 | prox |
| B2 | best-placed-conditioned +0.5 | none |
| B3 | best-placed-conditioned +0.5 | prox |

Frozen bars:

1. **Deploy bar**: teamWon(B3) − teamWon(B0) ≥ +4pp at ≥ 2× SE_diff, AND
   NONE(B3) ≤ NONE(B0) − 2pp, AND canaries clean (touch/goals within
   −10%/−20% relative of B0, kickoff < 5s, in-air ≤ B0 + 15pp).
2. **Attribution** (what deploys if bar 1 passes): if B2 is within 1 SE_diff of
   B3 → deploy the DERIVATION CONDITIONING only (pool rows must be best-placed;
   smallest diff). If B1 ≈ B3 (gate alone suffices) → deploy the collect-time
   prox gate only. Otherwise both, as one semantic lever ("the best-placed
   player goes"): condition the team-mode derivation pools on best-placed AND
   prox-gate the team-mode steering application.
3. FAIL → record and stop. Next lever class (frontier-drill dose, credit-side)
   requires new evidence and its own pre-registration.

Live implementation notes (verified feasible before running): both pieces read
only the row's OWN padded obs — teammate slots at 80/109 (+29 each), presence
flags at 225+, pos at slot+0..2 (coef 1/5000, canonical frame preserves
distances); derivation conditioning = require reqSelf ≤ min over present
teammates' required speed in fnSteerUpdate's labeling loop; application gate =
per-step multiply into tSteerMask before InferActions (team rows only, 1v1
unaffected). Guards unchanged: per-mode possession gates, rating latch, branch
backup ritual.

### Confirmation results

*(to be filled after the run; bars above are frozen)*
