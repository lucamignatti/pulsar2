# Headroom realizability by search — does H point at fixable mistakes?

*2026-09-03. Run 7.9-gco ts1 (`build/checkpoints_7.9gco_aimos/611200512000`, full checkpoint
with CRITIC/CRITIC2/VDAG1/VDAG2/GAP_EXP/OPP_EMBED). Tool: `research/tools/headroom_search.py`.
Results: `research/results/headroom_search_611200512000_{full,rep}.json`. Status: **RESULT**.*

## 0. The proposal under test

"Track the value of the bot's actions, detect obvious mistakes (value dips) or states the
headroom critic says have big unrealized upside, and run a tree search from those states in
the simulator to manufacture the improvement." Two triggers were on the table: **H** =
relu(min(V†₁,V†₂) − V_real), which this lineage already computes and gates SIL with, and the
**value dip**, a large negative realized k-step advantage.

This study asks the prerequisite question: from which states does search actually find
something? If the trigger does not concentrate realizable improvement, building the actuator
on it measures nothing.

## 1. Pre-registration (written before the run, verbatim in the script docstring)

For a banked state `s` (full physics snapshot: ball, both cars, pads, prevAction):

- **baseline** = mean over K fresh closed-loop policy rollouts from `s` of
  `score = Σ γ^t r_t + γ^T V_real(s_T)` (critic units; only goals are rewarded — GCO).
- **search** = evolutionary prefix search over the self player's actions: macro-actions held
  3 steps (0.2 s), prefixes of 2–12 macros from uniform-valid / policy-at-temperature /
  mutation generators, the policy closing the loop afterwards, opponent always the policy.
  96 candidates × 3 rounds, elites re-evaluated, a finalist stage re-evaluates the top
  prefixes fresh, then the winner is **re-evaluated held-out** on K fresh rollouts.
- **gain** = held-out mean(best) − baseline. Horizon 60 steps (4 s).

Strata (n=120 each per run): `H_top` (H ≥ q95), `H_top_neutral` (H ≥ q90 and |V| < 2, i.e.
not already decided), `H_mid` (q45–q55), `H_zero` (H = 0), `dip` (15-step realized
advantage ≤ q03 with 15 steps of play remaining; searched from the state *before* the drop),
`uniform` (control: the base rate at which this search finds anything anywhere).

Criteria (1 unit ≈ 1/7.5 goal at Returns-STD 20.06):

- **P1** H is a usable trigger iff gain(H_top) − gain(uniform) ≥ 0.5 **and** Spearman(H, gain) on uniform ≥ 0.2.
- **P2** dip is a usable trigger iff gain(dip) − gain(uniform) ≥ 0.5.
- Power: if gain(uniform) ≈ 0 and P(goal) is unchanged everywhere, the search is
  underpowered and nothing here is evidence about H.

Two runs: `full` (seed 11, K=24, finalists 8×8) and `rep` (seed 777, K=32, finalists 6×24).
Sim: pip RocketSim 2.2.1 at 5.0 dynamics (tickSkip 8 / actionDelay 0). Search and baseline
share the sim and the snapshot, so the contrast is internally consistent; absolute numbers
carry the v2/v3 engine gap. Restore parity was checked exactly (snapshot→restore reproduces
`Arena.clone` to 0.0 uu over 60 steps of random actions).

## 2. Results

Per stratum, both runs (critic units). `Pconc`/`Pgoal` = concede/goal share inside the
4 s horizon, baseline → search.

| stratum | run | H | baseline | gain | se | frac gain > 0.5 | frac gain > 3 | Pconc b→s | Pgoal b→s |
|---|---|---|---|---|---|---|---|---|---|
| H_top | full | 5.64 | −5.94 | **+0.02** | 0.06 | 0.10 | 0.00 | 0.67→0.65 | 0.01→0.00 |
| H_top | rep | 5.61 | −5.68 | +0.27 | 0.12 | 0.16 | — | 0.70→0.66 | 0.01→0.01 |
| H_top_neutral | full | 3.59 | −0.34 | +0.13 | 0.16 | 0.23 | 0.04 | 0.00→0.00 | 0.13→0.14 |
| H_top_neutral | rep | 3.58 | −0.37 | −0.19 | 0.17 | 0.29 | — | 0.00→0.00 | 0.12→0.09 |
| H_mid | full | 2.08 | −0.35 | −0.16 | 0.07 | 0.13 | — | 0.03→0.04 | 0.02→0.01 |
| H_mid | rep | 2.07 | −0.26 | −0.11 | 0.08 | 0.08 | — | 0.02→0.02 | 0.01→0.00 |
| H_zero | full | 0.00 | +5.97 | −0.03 | 0.10 | 0.11 | — | 0.00→0.00 | 0.68→0.68 |
| H_zero | rep | 0.00 | +5.80 | −0.15 | 0.09 | 0.11 | — | 0.01→0.01 | 0.69→0.69 |
| **dip** | full | 2.23 | −1.23 | **+0.59** | 0.22 | 0.41 | **0.15** | 0.20→0.17 | 0.10→0.13 |
| **dip** | rep | 2.00 | −1.00 | **+0.30** | 0.22 | 0.38 | **0.12** | 0.18→0.16 | 0.11→0.10 |
| uniform | full | 2.17 | −0.37 | −0.32 | 0.12 | 0.14 | 0.00 | 0.10→0.11 | 0.07→0.06 |
| uniform | rep | 2.20 | −0.20 | −0.23 | 0.09 | 0.17 | 0.01 | 0.05→0.06 | 0.05→0.05 |

Contrasts and correlations:

| | full | rep |
|---|---|---|
| gain(dip) − gain(uniform) | **+0.92** (Welch t 3.59, p 4e-4) | **+0.53** (t 2.24, p 0.03) |
| gain(H_top) − gain(uniform) | +0.34 | +0.49 |
| gain(H_top_neutral) − gain(uniform) | +0.46 (p 0.02) | +0.04 (p 0.85) |
| Spearman(H, gain), uniform | −0.10 | −0.19 |
| Spearman(H, gain), all 720 | +0.03 | +0.02 |
| Spearman(H, MC q75 − mean), uniform | −0.04 | +0.24 |
| mean H vs mean MC(q75 − mean), uniform | 2.17 vs 0.91 | 2.20 vs 0.78 |
| Spearman(V_real, MC baseline), uniform | +0.84 | +0.89 |
| mean(V_real − MC), uniform / dip | −0.00 / **+1.18** | −0.10 / **+1.36** |
| **Spearman(V_real − MC, gain), all** | **+0.37** | **+0.37** |
| min V† − MC, uniform / H_zero | +2.04 / −1.89 | — |
| winner's curse (in-sample − held-out), uniform | +0.73 | +0.21 |

### Verdicts

- **P1 fails, both runs.** H's top-5 % contrast is below 0.5 in both runs and Spearman(H, gain)
  is zero-to-negative. The neutral-H stratum (high H, game not yet decided) is +0.46 in one run
  and +0.04 in the other. H is **not a trigger for realizable improvement**.
- **P2 passes, both runs** (+0.92 and +0.53 against a 0.5 bar). The dip stratum is the only one
  with a fat tail: 12–15 % of dip states yield a gain > 3 units (≥ 0.4 goal) against 0–1 % of
  uniform states.
- **Power is adequate**: the search finds the fat tail where it exists, the uniform gain is
  slightly negative because the finalist stage still picks a noisy candidate over the policy
  (winner's curse +0.2–0.7; every held-out gain here is conservative by about that much, and
  the contrasts are fair because every stratum shares the procedure).

## 3. What the search found

The top dip fixes are short — 2–5 macros, 0.4–1.0 s — and are exactly "obvious mistakes":

- A **deterministic concede** (baseline −6.96 ± 0.0) undone by a 3-macro aerial correction
  (boost + pitch/roll toward the ball): gain +8.0, concede 1.00 → 0.00.
- Air-drill and near-ball states where a 2–5 macro prefix turns goal odds 0.12 → 0.71,
  0.00 → 0.88, 0.29 → 1.00 (gains +5.5 to +7.0).
- Kickoff: a jump-then-tilt prefix, goal 0.00 → 1.00 (+7.0).

By reset kind within the dip stratum (gain, full / rep): **air +1.90 / +0.77**,
random +0.67 / +0.30, kickoff +0.39 / +0.51, near −0.21 / −0.02. The realizable
mistakes are aerial. Near-ball dips are not fixable by a short prefix — those are
opponent-driven, not self-inflicted.

## 4. What actually predicts a fixable state

Not H. The best single predictor in both runs is **V_real − MC**: the critic's estimate of
the state minus what fresh rollouts of the policy actually realize from it (ρ = +0.37 twice;
+0.52 inside the dip stratum). The critic is well calibrated on average (ρ 0.84–0.89 against
Monte Carlo, bias ≈ 0) but on dip states it overestimates by 1.2–1.4 units — it prices the
state as if the policy will play it correctly, and the policy then does not. That over-
estimate is where the search cashes in.

H measures none of this. It does not even track the policy's own within-state upside
(ρ ≈ 0 against the Monte Carlo 75th-percentile spread, and its magnitude is 2.5× that
spread), and min V† sits about +2 units above Monte Carlo on every stratum except H_zero,
where it sits −1.9 below — a roughly constant offset, not a signal. This is the same
finding as `MM_SLOW_LEARNING.md` (H's top decile is where the team concedes) from the
other direction: search from those states finds nothing because the concede is already
locked in.

The online-computable version of V_real − MC is the **negative realized advantage** — the
GAE arrays the trainer already holds. The `dip` stratum is that quantity at k = 15.

## 5. Implications for the actuator

1. **Trigger on realized-advantage dips, not H.** The SIL gate's `unit(H) ≥ q70` is gating on
   a quantity with zero correlation to realizable improvement.
2. **Search is worth ~0.5–0.9 units per triggered state** (7–12 % of a goal), concentrated in a
   12–15 % fat tail of full-goal-sized fixes, and the fixes are 0.4–1.0 s aerial corrections.
   Cost on the desktop: ~4–5 s per state at 400 rollouts, six workers on the 5080 → ~700
   states in 10 min. Dip states are 3 % of rows by construction, so an online deployment
   would search a few thousand states per iteration-equivalent — not free, but a batch
   search per iteration over the worst 1 % of the batch is feasible.
3. **Consumer**: the found prefix + the state is a positive-advantage off-policy transition set;
   SIL is the existing consumer with the right shape. This is a C2 conflict with
   `COMPOSITION_CRITIC.md` (banked retrospective states) — a decision to record, not hide.
4. **Not measured here**: whether imitating search-found prefixes generalises (the search fixes
   *this* state against *this* opponent sample; held-out re-evaluation used fresh opponent
   samples, so the fixes are robust to opponent noise, but not to state perturbation), and the
   v3-engine numbers (this ran on pip RocketSim v2).

## 6. Reproduce

```bash
cd build && ../research/.venv/bin/python ../research/tools/headroom_search.py \
  --ckpt checkpoints_7.9gco_aimos/611200512000 --out out.json --workers 6 \
  --arenas 16 --steps 1500 --per-stratum 20 --n-cand 96 --rounds 3 --k-eval 24
```
