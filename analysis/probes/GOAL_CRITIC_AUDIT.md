# GOAL_CRITIC_AUDIT — is the goal critic driving learning, coasting, or hindering?

**Pre-registered 2026-07-20 BEFORE running.** Checkpoint **42200430819** (live
5.0v3 lineage, run `bfl8mbw4`, frozen copy). Script `goal_critic_audit.py`.

## The question (user)

"Is the shaped critic or the goal critic driving learning, or is the goal critic
hindering it?"

## What the trainer actually does (Learner.cpp:4679-4733)

```
A = A_dense + betaEff*(A_goal - mean(A_goal))  [+ RND + ladder drive]
betaEff = 0.25 * std(A_dense)/std(A_goal)        <- std-matched, so the goal
                                                    channel is EXACTLY 25% of
                                                    dense-advantage scale
```
Both advantages are GAE with **lambda = 0.95**. Effective credit window is
`1/(1-gamma*lambda)`: dense `1/(1-0.9969*0.95)` = **18.9 steps**, goal
`1/(1-0.9994*0.95)` = **19.8 steps**. At 15 Hz that is 1.26 s vs 1.32 s — the
gamma separation the design was built on (15 s vs 77 s half-life) does NOT
survive into the advantages; lambda truncates both to the same ~1.3 s window.
Gamma survives only inside the two value functions. So the goal channel is not
a long-horizon channel in the gradient — it is a **second opinion at the same
time-scale, sourced from outcomes instead of shaping**. That is what gets tested.

## Prior (CRITIC_DUEL, ckpt 27550023244, different lineage)

zG had NO local outcome authority on resolved races (held-out AUC 0.410 vs
zV 0.554; joint coef +0.22 +- 0.34). The goal critic's LEVEL was outcome-blind
locally. That test never examined the quantity that actually enters the
gradient, which is the ~1.3 s **derivative** of V_goal, not its level.

## Tests (model-free ground truth only)

Fresh 2v2 rollout (`rollout_team`, trainer reset mix), plus a 1v1 pass for the
non-race tests. Per player-contiguous sequence: `V = CRITIC(h2)`,
`G = GOAL_CRITIC(obs)`, goal channel = +-1 at goal terminals, 0 elsewhere.

- `A_goal` = GAE(goal channel, G, gamma 0.9994, lambda 0.95) — **the exact
  quantity the trainer injects**.
- `D_G`   = same with the terminal reward removed (pure value drift).
- `D_V`   = GAE(0, V, gamma 0.9969, lambda 0.95) — the shaped critic's local
  drift, the directly comparable half of `A_dense`.

**T1 Level authority.** Episode-grouped held-out AUC of G and of V predicting
the episode's realized goal sign. Reproduces the in-run
`GoalCritic/Value-Outcome Corr` panel and asks whether V already knows it.

**T2 Redundancy.** corr(zV, zG); joint logistic `outcome ~ zV + zG` with
episode-bootstrap CIs. If zG adds nothing over zV, the channel is duplicating
information the dense path already carries.

**T3 Derivative authority (the new core test).** On resolved possession races
(`decline_readings`, outcome != NONE), episode-grouped held-out AUC of `A_goal`
vs `D_V` for "reader's team wins the ball", plus the joint-logistic coefficient
on `A_goal` given `D_V`.

**T4 Agreement.** corr(A_goal, D_V) and corr(D_G, D_V) — do the two critics'
local derivatives agree, ignore each other, or fight?

**T5 Interference.** Distribution shape of A_goal (std, kurtosis, top-1% mass
share) and, under a std-matched Gaussian A_dense, the rate at which the
injected term FLIPS the sign of the advantage and the rate at which it exceeds
|A_dense|. This is how often the goal channel overrules the shaped critic.

## Pre-registered reading of the outcome

- **Driving / earning its 25%**: T3 shows `A_goal` AUC > 0.5 at >=2 sigma AND the
  joint coefficient on `A_goal` is positive at >=2 sigma given `D_V`.
- **Coasting (harmless redundancy)**: T3 near chance but T4 shows strong
  positive agreement with `D_V` — the injection mostly re-states the dense
  signal, so the 25% is wasted scale, not damage.
- **Hindering**: T3 near or below chance AND T4 agreement near zero or negative
  AND T5 sign-flip rate materially above zero — an independent, uninformative
  term is overturning the shaped critic's ranking on a measurable share of rows.

Whatever the verdict, it is a MEASUREMENT, not a deployment. No live-config
change follows from this document.
