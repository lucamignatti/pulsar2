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

# Optimism surgery, offline validation: it works, with a dose limit

**Verdict: (1) works.** A "commitment" direction exists in the trunk, is strongly
decodable, and steering it during rollouts causally moves engagement the intended
way without breaking basic competence — at moderate strength. At +2σ the behavior
shift is large but signs of cost appear. The mechanism is validated as a
collection-time lever; whether steered collection improves *learning* is a trainer
A/B question by construction, not answerable offline.

## The direction (read access)

From one 60 k-row self-play pass per pinned checkpoint: feasible free landings are
labeled went/declined by touchdown lookahead (same definitions as KNOWING_DOING.md),
matched over (distance-to-landing × flight-time) bins, direction = difference of
class means in h2.

- Matched went-vs-declined projection AUC: **0.82–0.89 in-sample** (ckpts
  3025003520 / 3047354368, ~720–810 readings/class). CORRECTION (found while
  building the sidecar): in-sample AUC of a 512-d difference-of-means is
  overfit-inflated at these sample sizes (a 96/class fit reads 1.000 in-sample).
  Honest held-out numbers: **0.76** at 750/class (ckpt 4089133056); a 516/class
  derivation read 0.55, so small-n estimates swing hard. The premeditation signal
  is real but moderate on a per-frame basis.
- Still a finding: matched for geometry, the trunk carries a linearly-readable
  *premeditation* signal — and per below, a causally-steerable one. But the
  quality metric that matters for the trainer is the behavioral dose-response,
  which `derive_steering.py` now gates on directly (α=0 vs +2 rollouts, ≥ +3 pp
  engagement; measured +7.2 pp at ckpt 4089133056).

## Steering (write access), ckpt 3047354368, 80 k rows/α, paired seeds

h2' = h2 + α·σ_proj·v fed to the policy head only. Engagement = P(contested in
flight OR present at touchdown | feasible reading, contested included) — immune to
the denominator drift that plagues attendance-of-free alone.

| α | engagement | in-air | touch rate | goals/ep | kickoff 1st touch |
|---|---|---|---|---|---|
| −2 | 11.2 % | 28.1 % | 0.14 % | 0.13 | 3.97 s |
| −1 | 18.3 % | 28.4 % | 0.21 % | 0.14 | 3.33 s |
| 0 | 17.6 % | 28.5 % | 0.21 % | 0.18 | 3.17 s |
| +1 | 19.3 % | 30.0 % | 0.14 % | 0.10 | 3.23 s |
| +2 | **26.2 %** | 31.2 % | 0.11 % | 0.11 | 3.57 s |

- Engagement: +49 % relative at +2σ (~6σ significant at n≈1.2–1.4 k readings/α),
  collapse to 11 % at −2σ. The previous run (ckpt 3025003520, 40 k rows/α) showed
  the same engagement pattern and a dramatic monotone kickoff effect (8.1 s at −2σ
  → 2.97 s at +2σ); kickoff medians are noisy across runs but never break at
  positive α.
- **Dose limit**: at +2σ touch rate halves and goals/ep drops — overcommitment
  cost and/or off-manifold activation damage. +1σ is nearly free but weak. The
  usable window is roughly α ∈ [+1, +2), and rho-band gating (steer only where the
  capability model rates outcomes intermediate) is the obvious refinement before
  raising the dose.

## Caveats

- Trainer advanced between runs — each run pins its own checkpoint (direction is
  re-derived per checkpoint from its own rollouts; that re-derivation being cheap
  and automatic is a feature the trainer-side version inherits).
- Goals/episode is underpowered at this scale and flipped sign between runs; no
  claim about match-level outcomes.
- Steering is constant-everywhere; no state gating yet.
- Behavior policy only: PPO-learning effects (clipped-IS off-policyness at these α,
  avgRatio/KL drift, and whether optimistic data actually speeds skill acquisition)
  are the trainer A/B, gated on the current run finishing.

## Trainer-side shape (when the time comes)

Steer only the collect worker's policy snapshot (h2 + α·σ·v before the policy
head), α ≈ +1 to start, direction re-extracted periodically from recent rollouts
(the labeling is pure physics + lookahead — no human input), watched by the
existing avgRatio/KL logs and Rating canary, behind the validation-gate pattern.
Learn pass untouched: the target policy still optimizes pure zero-sum returns.

Repro: `steer_test.py` (~50 s; thread-cap + nice per README).
