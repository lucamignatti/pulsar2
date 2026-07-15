# DECISION_GATE — steer undecided moments, not contested races (steering limits #3b)

**Pre-registered 2026-07-15 BEFORE running.** Checkpoint **27550023244** (frozen
copy). Script `decision_gate.py`.

## Hypothesis (from INTERP_SWEEP2 A3)

The trunk broadcasts its own intention with near-perfect fidelity
(premeditation: "self reaches ball within 1s" decodes at AUC 0.96 from >500uu
away). Every steering deployment so far gates on the RHO band — where the
RACE is a coin-flip. But if commitment is *internally decided* well before the
reading, rho-band steering fires after the die is cast — which would explain
why every positive-side dose response is saturated (1v1 0a, TEAM_GATE) while
suppressive doses stay strong (it is always possible to break a made decision,
rarely possible to re-make it later). The test: gate steering on the
PREMEDITATION readout being uncertain — steer decision-boundary moments.

## Mechanism (live-parity, no hardcoding)

Premeditation direction w = difference of h2 means, label = same-player touch
within 1s (the trainer has this label for free in `combinedTraj.touched`; fully
live-derivable per iteration like every other vector). Gate = per-batch
quantile band [0.2, 0.8] on the w-projection — the same band width as the rho
gate, so the treated-row budget is matched and the comparison isolates gate
SELECTION, not dose volume (parity check registered: in-band fractions within
±10pp).

## Arms (same direction within a mode; only the gate differs)

1v1 (600k rows/arm, direction = v2 commitment) and 2v2 (800k rows/arm,
direction = best-placed-conditioned — strongest known for the mode):

| arm | alpha | gate |
|---|---|---|
| D0 | 0 | — |
| Drho | +0.5 | rho band (the live gate) |
| Ddec | +0.5 | premeditation band (rho off) |

## Frozen bars

Per mode, primary = possession on feasible readings (1v1: possWin; 2v2:
teamWon), episode-cluster SEs:

1. **Gate-swap bar**: Ddec − Drho ≥ +2pp at ≥1.5× SE_diff, AND Ddec − D0 > 0,
   AND canaries clean (touch/goals within −10%/−20% of D0, kickoff < 5s,
   in-air ≤ D0 + 15pp). PASS in a mode → licenses a live one-lever trial of
   swapping that mode's gate (own deployment pre-registration + ritual;
   direction/gate both remain live-derived per iteration).
2. Sanity: premeditation direction must actually read intention — projection
   AUC vs the 1s-touch label ≥ 0.80 on held-out rows; below that the gate is
   noise and the experiment is void (record, fix derivation, rerun once).
3. FAIL both modes → decision-boundary steering joins the dead-candidate list;
   the saturation explanation shifts to "the trap is not at the gate" (i.e.,
   positive-side saturation is dose- or axis-limited, not targeting-limited).

## Results (2026-07-15; raw: `results/decision_gate_27550023244.json`)

Sanity bar PASSED in both modes: premeditation projection AUC 0.868 (1v1) /
0.886 (2v2) from a plain difference-of-means — the live-derivable recipe reads
intention. In-band fractions matched (0.58/0.60 both gates).

| mode | D0 | Drho (+0.5, rho band) | Ddec (+0.5, premed band) | Ddec−Drho | bar |
|---|---|---|---|---|---|
| 1v1 | 12.5% ± 1.1% | 12.0% ± 1.3% | 13.7% ± 1.9% | +1.7pp (0.74σ) | FAIL |
| 2v2 | 15.4% ± 2.3% | 13.4% ± 1.2% | 15.3% ± 2.3% | +1.9pp (0.73σ), Ddec−D0 ≈ 0 | FAIL |

**VERDICT: FAIL in both modes — decision-boundary steering joins the
dead-candidate list.** Sign-consistent over the rho gate in both modes but far
under the bar, and indistinguishable from no-steering baselines. Per the
registered rule 3 the interpretation shifts: positive-side saturation is NOT a
targeting artifact — the trap is not at the gate. The acute steering channel
itself is exhausted at this checkpoint (across the day: 3 directions × 4 gates
× 2 doses, every arm within [−6, +4.5]pp, nothing clears an honest bar), while
CREDIT_PROBE places the persisting mis-pricing in the main critic with a
demonstrated horizon component. The frontier of this research program has
moved from the trunk to the credit assignment.
