# CRITIC_DUEL — who is wrong on the frontier, the critic or the goal critic?

**Pre-registered 2026-07-15 BEFORE running** (after the FEAR_MINE deploy — this
is the user's challenge made testable, with a revert bar). Checkpoint
**27550023244** (frozen copy). Script `critic_duel.py`.

## The challenge (user)

"The goal critic when perfect is a source of truth, but it should lag since
the signal is so sparse. The shaped critic should ideally align with a perfect
goal critic. Since neither is perfect, I'm not sure about this [Δz mining]."
Correct: Δz disagreement is SYMMETRIC evidence. CREDIT_PROBE convicted the
shaped critic partly by trusting the goal critic — unearned. Two additional
disagreement sources neither net is guilty of: the deliberate γ mismatch
(~15s vs ~77s half-life — even perfect critics disagree on delayed payoffs)
and PBRS potential terms Φ(s) baked into V's targets.

## Discriminating tests (model-free ground truth only)

Data: fresh 2v2 rollout (900k rows, h2+obs+state bank), V/G/Δz per row,
`decline_readings` labels.

1. **Prediction duel on resolved races**: population = readings where a race
   actually resolved (outcome ≠ NONE). y = reading player's team touched first.
   Compare held-out AUC of zV vs zG (episode-grouped folds), and a joint
   2-feature logistic — does zG carry outcome information BEYOND zV
   (coefficient sign stable at ≥2σ under episode bootstrap)? Under the user's
   hypothesis (goal critic = optimistic noise on these states), zG adds
   nothing given zV.
2. **Forced-contest calibration of Δz** (the mining criterion tested directly):
   from declined feasible best-placed readings, top-100 Δz (HI) vs bottom-100
   Δz (LO); reconstruct with SMALL noise (100uu/100uu/s — preserve state
   identity, unlike the 250uu drill noise) and roll the frozen policy 5s × 2.
   Under the fear hypothesis, HI states are systematically more winnable when
   actually contested: reader-team first-touch share HI − LO ≥ +10pp at
   ≥1.5× the two-proportion SE. Under goal-critic-noise, HI ≈ LO (or inverted).

## Frozen decision rule for the live flag

- KEEP `frontierFearMining` if test 1 shows zG adds outcome information beyond
  zV (≥2σ) **or** test 2 shows HI − LO ≥ +10pp at ≥1.5σ.
- REVERT the flag (flip false, rebuild, restart — the branch backup and one-line
  revert exist) if BOTH fail: then Δz is not demonstrated to select winnable
  fear over critic-correct declines, and the uniform decline sample returns
  until better evidence exists.
- Either way, record which disagreement source dominates for the reward-quality
  program (the user's framing: this diagnostic tells us when the shaped rewards
  suck).

## Results (2026-07-15; raw: `results/critic_duel_27550023244.json`)

384 episodes, 8,190 readings.

**Test 1 — FAILS (the user's skepticism vindicated):** on 1,767 resolved races,
held-out AUC of zG = **0.410** (worse than chance; zV = 0.554), joint-logistic
zG coefficient +0.22 ± 0.34 (nothing), same on the best-placed subset. The
goal critic carries NO local outcome authority — sparse-signal lag is real.
CREDIT_PROBE's horizon-disagreement result stands only in its weak form
("V is internally inconsistent on the frontier"), NOT as "G knows better".

**Test 2 — PASSES at 3.7σ (the selector works anyway):** forced contests from
near-exact reconstructions of declined best-placed readings:

| tail | median Δz | reader-team win share |
|---|---|---|
| HI (top-100 Δz) | +2.39 | **53.2%** (n=62 resolved) |
| LO (bottom-100 Δz) | −2.81 | **23.0%** (n=61 resolved) |

HI − LO = **+30.3pp ± 8.3pp**. High-disagreement declines are genuine
coin-flips the bot walked away from; low-disagreement declines are lost causes
the critic was RIGHT to decline. (Both tails: ~69% of rolls stay unresolved at
100uu noise — the production drill's 250uu perturbation is load-bearing for
eliciting contests.)

**Verdict per the frozen rule: KEEP `frontierFearMining`** — bar 2 passed.
The corrected mechanism story: Δz works not because G is a source of truth
(it is nearly outcome-blind locally) but because subtracting V from a roughly
unbiased long-horizon signal surfaces where V is ANOMALOUSLY PESSIMISTIC —
the fear lives in V; G merely has to not share it. Standing rule earned here:
the goal critic must never gate, steer, or judge anything on its own; it is
usable only inside relative/differential constructions validated against
behavioral ground truth.
