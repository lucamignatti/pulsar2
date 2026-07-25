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

# CREDIT_PROBE — the whiff tax in the critic's own valuations (interp sweep #3a)

**Pre-registered 2026-07-15 BEFORE running.** Checkpoint **27550023244** (frozen
copy). Script `credit_probe.py`. New interpretability direction: every probe so
far read the TRUNK (what the policy knows); this reads the CRITICS (what PPO
optimizes against). The whiff tax was convicted behaviorally (1v1 era),
belief-level (INTERP_SWEEP2 B3), and steering-resistant (TEAM_GATE). If it is
real, it must be visible in the value function itself — and if the LONG-horizon
goal critic (γ=0.9997, ~77s) disagrees with the main critic about frontier
states, the tax is partly a HORIZON artifact, which would make the β blend a
cheap measured lever.

## Data

Fresh 2v2 rollout (900k rows, want_h2 + want_obs), `decline_readings` labels
(feasible, best-placed, pursued, decline). V = CRITIC(h2) (the critic head runs
on the trunk output; h2 is recorded), G = GOAL_CRITIC(obs) (independent net on
raw obs; loaded ad hoc via the generic rebuild). Both computed post-hoc — no
rollout changes. All comparisons matched on (req_self tercile × t_land tercile
× margin tercile), episode-cluster bootstrap SEs.

## Registered analyses and interpretations

1. **Pricing at the decision** — among feasible AND best-placed readings, mean
   V at the reading row for subsequently-PURSUED vs subsequently-DECLINED
   (matched). If V(decline) ≥ V(pursue) (gap ≥ 0 at ≥2σ, or indistinguishable
   from 0): the critic prices declining at-or-above attempting at states where
   the bot is the right player to go — the mis-pricing convicted IN the value
   function. If V(pursue) exceeds V(decline) clearly: the critic already
   prefers attempts and the bottleneck is elsewhere (weakens the credit-lever
   case; record and stop).
2. **The tax trajectory** — mean V over the 3s after the reading (same-slot
   rows), matched pursued vs declined. A sharper post-attempt V drop is the
   counterattack tax as the critic sees it.
3. **Horizon disagreement** — repeat 1-2 with the goal critic; index
   Δz = z(G) − z(V) (per-rollout z-scoring puts the two value scales on one
   axis). Registered: Δz(pursue) − Δz(decline) > 0 at ≥2σ means the long-horizon
   critic RELATIVELY favors attempting where the short-horizon critic does not →
   horizon-induced component → a goal-critic-weighted advantage blend at
   frontier states becomes the next lever candidate (own pre-registration; β is
   currently a global 0.25 std-matched blend).

Caveat registered up front: behavior is not randomized — matching covers only
the labeled difficulty variables, so results are directional evidence, not a
causal estimate. The value of the probe is that it reads the exact quantity PPO
uses; a mis-pricing visible here is the mechanism, not a proxy for it.

## Results (2026-07-15; raw: `results/credit_probe_27550023244.json`)

382 episodes, 7,833 readings, 3,975 feasible+best-placed, 1,248 matched pairs
per class. V mean 0.456 std 0.278; G mean 0.071 std 0.265 (z-scored for
comparisons).

| measure | value | read |
|---|---|---|
| pricing V: pursue − decline | **−0.053 ± 0.022** (−0.19z, ~2.3σ) | critic prices DECLINING higher |
| pricing G: pursue − decline | **+0.031 ± 0.013** (+0.12z, ~2.4σ) | goal critic prices PURSUING higher |
| horizon index Δz(p) − Δz(d) | **+0.305 ± 0.107 (2.9σ)** | ≥2σ bar MET |
| traj V 2-3s drop: pursue vs decline | −0.188 vs **−0.319** (diff +0.130 ± 0.059) | value erodes FASTER after declines |
| traj G 2-3s drop | −0.121 vs −0.107 (≈0) | long horizon indifferent |

**All three registered analyses land on the same conviction:**

1. The mis-pricing is real and lives in the main critic: at matched
   best-placed frontier states it values the about-to-decline branch above the
   about-to-pursue branch (analysis 1) — while its own subsequent valuations
   show declining leads to worse states (analysis 2). The critic is
   internally inconsistent on the frontier in exactly the whiff-tax direction.
2. The registered horizon test (analysis 3) PASSES at 2.9σ: the γ=0.9997 goal
   critic — same observations, same experience, longer horizon — disagrees
   with the γ-short critic in favor of attempts. A substantial component of
   the team whiff tax is HORIZON-INDUCED, not information-induced.

Registered consequence: **the goal-critic advantage blend (β, currently a
global std-matched 0.25) is the next lever candidate** — the first credit-side
lever with direct measurement conviction behind it. It is an UPDATE-side,
training-time change (offline rollouts cannot validate it); a trial needs its
own pre-registration with live success criteria (Rating/2v2 slope, Steer
PossWin/NONE panels, drawdown latch armed) and user sign-off. Options in
ascending machinery order: (a) modest global β raise, (b) per-mode β (team
modes only), (c) frontier-conditional β (most targeted, most machinery —
stage-2-adjacent caution applies to anything row-conditional near the critic).

Correlational caveat (registered up front) stands: matching covers labeled
difficulty only. The internal-inconsistency finding (analysis 1 vs 2) partly
defuses it — both branches share whatever confounds the matching misses.

**CORRECTION (2026-07-15, post CRITIC_DUEL):** the user challenged the implicit
"goal critic as arbiter" framing, and the follow-up test proved them right —
the goal critic is nearly outcome-blind locally (AUC 0.41 on resolved races,
zero incremental information beyond V). Analysis 3's conclusion survives ONLY
in its weak form: V is internally inconsistent on the frontier, and Δz is a
behaviorally-calibrated fear detector (forced-contest validation, +30pp).
Read CRITIC_DUEL.md before building anything else on the goal critic.
