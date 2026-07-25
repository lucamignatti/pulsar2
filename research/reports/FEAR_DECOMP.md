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

# FEAR_DECOMP — what makes the critic scared? (reward-diagnostic study)

**Pre-registered 2026-07-15 BEFORE running.** Script `fear_decomp.py`, newest
live checkpoint (selective copy). Measurement only — no intervention is
licensed by this study.

## Question

CRITIC_DUEL validated a fear population: team-mode declines the shaped critic
prices pessimistically that are empirically ~53% winnable when forced. This
study asks WHERE the fear lives in state space — because if the pessimism
concentrates along the structure of specific shaped-reward terms, that
implicates those terms (the user's program: critic/goal-critic disagreement as
the "our rewards suck HERE" diagnostic).

## Design

2v2 rollout (900k rows) at the newest checkpoint; V/G/Δz per row;
`decline_readings`. Population: feasible best-placed DECLINED readings. HI =
top Δz quartile (validated-fear states), LO = bottom quartile
(critic-correct declines, 23% winnable per CRITIC_DUEL).

Per-reading features (team-canonical where directional): flight time, distance
to landing, required speed, teammate margin, minimum OPPONENT required speed
(contestedness), self boost, canonical self y (own-half depth), self distance
to own goal, canonical landing y, landing distance to own goal, canonical ball
velocity y (toward own goal = danger), ball speed, self speed, on-ground flag.

Analyses:
1. **HI vs LO standardized mean difference** per feature with episode-cluster
   bootstrap CIs — the primary, assumption-light readout.
2. Ridge Δz ~ features (episode-grouped CV, held-out R²) — how much of the
   fear score is explained by these interpretable features at all.

## Registered interpretation guide (descriptive, not a deploy decision)

- Fear concentrated at LOW BOOST → boost-economy reward terms under-price
  attempts made on empty tanks (or the critic over-prices boost itself).
- Fear concentrated DEEP IN OWN HALF / near own goal / ball moving toward own
  goal → the counterattack tax: defensive-exposure pricing is the culprit
  (the original 1v1 whiff-tax mechanism, still alive in team modes).
- Fear concentrated at HIGH landing / long flight time → aerial execution
  pricing (interacts with AirDrill/AirIntercept terms).
- Low ridge R² (< 0.15) → the fear is not explained by these interpretable
  features; it lives in subtler state structure → SAE-lite decomposition of
  the critic pathway becomes the follow-up.

Any reward-term change motivated by this study needs its own pre-registration
(zero-sum invariant, PBRS gamma rule, one lever).

## Results (2026-07-15, checkpoint 29950113508; raw: `results/fear_decomp_29950113508.json`)

431 episodes, 6,996 readings, 2,074 declined best-placed (HI/LO = 519 each).
Standardized HI−LO mean differences (* = |smd| > 2 SE):

| feature | smd | | feature | smd |
|---|---|---|---|---|
| landing dist own goal | **+0.88*** | | boost | **−0.58*** |
| self dist own goal | **+0.77*** | | ball z | **−0.55*** |
| landing y (canonical) | **+0.77*** | | min opp req (landing) | **+0.50*** |
| ball y (canonical) | **+0.67*** | | ball vy (canonical) | **+0.36*** |
| self y (canonical) | **+0.59*** | | everything else | n.s. |

Ridge Δz ~ features held-out R² = **0.27** (> the 0.15 bar — the fear is
substantially explained by interpretable features; no SAE needed yet).

**The fear profile is NOT what the registered guide guessed.** The critic's
pessimism concentrates on **attacking-half, low, UNCONTESTED landings reached
on a LOW boost tank** — not on defensive exposure at the ball. Reading: the
whiff tax in team modes is priced onto *offense without resources* — being the
committed forward with an empty tank means no recovery if the play breaks,
and the critic taxes the commitment itself rather than the (empirically ~53%
winnable, opponents-far) ball. This coheres with CRITIC_DUEL: these are
exactly the states forced contests win at coin-flip rates.

Reward-terms implicated (descriptive only; any change needs its own
pre-registration under the zero-sum + PBRS-gamma rules): the boost-economy /
recovery pricing interaction with deep offensive positioning. A concrete
follow-up candidate: check whether the shaped stack's potentials make
boost-at-depth look more valuable than the attempt it enables.
