> **Status: HISTORICAL.** The activation-steering program this belongs to was
> superseded by the Optimistic-Critic Ladder ([LADDER.md](LADDER.md)). Retained
> because its argument is still cited as standing rationale — see
> [README.md](README.md).
>
> **`h2` caveat.** Offline behavioural numbers below were computed before the
> 2026-07-19 h2-truncation fix, against **pre-activation `h2`** (the loader dropped
> the trunk's trailing LeakyReLU). Treat the numbers as invalid and the reasoning
> as live. See [H2_TRUNCATION.md](H2_TRUNCATION.md).

---

# Is rho a trustworthy frontier compass? Mostly yes — as a coarse one.

**Verdict: usable but noisy.** Both reachability heads rank goals monotonically by
actual achievement probability, with ~5x success enrichment from bottom to top
rho-decile — but discrimination is soft (AUC 0.70 car / 0.74 ball), so rho supports
*binned* goal selection ("sample from the intermediate band"), not fine per-goal
fitness weighting. Notably, the intermediate rho deciles land right in the
goals-of-intermediate-difficulty band (~35–55 % success) that self-curriculum
methods target — the frontier the optimism agenda wants is directly addressable
with the machinery the bot already trains.

Runtime: 13 s (60 k rows self-play + ~100 k scored goal pairs), checkpoint
2670070784 (~2.67 B steps — note: the trainer has advanced well past the probe
study's 1.46 B checkpoint).

## Protocol

rho(s → g) = mean over 16 uniform *valid* actions of cos(phi(trunk(s), a), psi(g))/tau
(EvalRho parity). The heads train InfoNCE + HER where positives are future achieved
states within a window (car: car-local ball pos+vel /2300, 20 steps ≈ 0.67 s;
ball: canonical ball pos+vel, 90 steps = 3 s). Calibration ground truth is the same
event the training defines: does ordinary play actually pass within ε of g (normalized
6-D) in the next window? Goal menu per anchor state: 3 same-player future achieved
states (short-biased offsets, reachable by construction), 3 jittered futures
(σ ∈ {0.08, 0.2, 0.5} — intermediate), 3 cross-episode achieved states (mostly
unreachable). ~11 k anchors, ~100 k pairs, ε swept {0.075, 0.15, 0.3}.

## Results (ε = 0.15; ε-sweep in results/rho_calibration.json)

| | car head (20-step window) | ball head (90-step window) |
|---|---|---|
| ROC AUC | 0.701 | 0.737 |
| Spearman (rho vs goal proximity) | 0.358 | 0.481 |
| success, bottom rho decile | 14 % | 2 % |
| success, top rho decile | 66 % | 56–60 % |
| rho separation future vs cross | −10.1 ± 4.4 vs −13.9 ± 4.3 | −6.4 ± 2.8 vs −12.2 ± 4.0 |

Decile success curves are monotone for both heads (plot:
`results/plots/rho_calibration.png`); the ball head dips slightly at the very top
decile — plausibly window truncation by goal-scored episode ends near the net; not
diagnosed here.

## What this means for the self-curriculum idea

- **Green light for coarse use.** Sampling drill/replay goals from the
  intermediate-rho band is meaningful: the band genuinely sits at ~40 % success,
  and the tails genuinely differ 5x. That is enough signal to *enrich* the
  training distribution with frontier situations — no human skill taxonomy
  involved.
- **Red light for fine-grained use.** Per-goal rho ranking is noisy (AUC 0.7);
  any mechanism that weights individual proposals by rho inherits that noise.
  This rhymes with the PSD probe-fitness reliability ≈ 0 finding and is a
  candidate ingredient in why fitness-weighted proposals underperformed in the
  9uz761ua era. Aggregate, bin, and validation-gate — never trust a single rho.
- **Known blind spot.** This calibrates "will ordinary play reach g" (the HER
  event). The policy is goal-blind, so "can it reach g when *trying*" is not
  measurable without goal-conditioning — capability at the boundary is likely
  *underestimated* for goal types the policy currently avoids (exactly the
  optimism-relevant ones). rho-selected drills would tighten this gap over time
  by construction, since drilled goals enter the HER data.

## Repro

`calibrate_rho.py` (env `RHO_ROWS` overrides the 60 k default; thread-cap + nice
as in README). Outputs `results/rho_calibration.json` + the calibration plot.
