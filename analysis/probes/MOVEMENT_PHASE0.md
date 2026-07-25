# MOVEMENT_PHASE0 — car-movement (execution) frontier: offline validation

**Pre-registered 2026-07-15, BEFORE any sweep ran.** Per the measurement doctrine:
criteria first, data second, no post-hoc bar-moving.

## Question

Commitment steering moves *initiation* (go for the ball at all). This asks whether a
second, distinct trunk direction moves *execution*: among races the bot already
commits to, does steering raise the fraction it actually wins? The frontier
hypothesis (user, 2026-07-15): the frontiers worth steering are commitment, car
movement, and possibly positioning — not the arbitrary state-space frontiers the meta
system chased (convicted 2026-07-14: only mis-priced frontiers pay; meta's
attainment-quantile outcomes are style-fakeable, possession/completion outcomes are not).

## Labels (from 1v1 self-play rollouts, trainer-parity reset mix)

- Readings: airborne ball (z > 300), ball-only landing sim, FEASIBLE for the reading's
  player (required speed < 1300 uu/s), uncensored (touchdown inside the episode) —
  identical to the in-trainer v2 derivation.
- Outcome: first touch between reading and touchdown + 0.5s — WON / LOST / NONE
  (steer_v2 parity).
- **Pursued** (the new label): the player meaningfully chased this landing —
  WON, or within 500 uu of the landing point at touchdown, or net approach speed
  ≥ 60% of the required speed.
- **COMPLETED** = pursued ∧ WON.  **WHIFFED** = pursued ∧ ¬WON (arrived late, lost
  the race, or missed the touch). Both classes committed, so the contrast isolates
  execution quality with commitment held fixed.

## Direction

v_move = matched (5 distance-quantile × 3 flight-time-quantile bins) difference of
trunk-output means, COMPLETED − WHIFFED, unit-normalized; dose scale σ = std of
all-row projections (same recipe that produced the causally-validated commitment
direction). Report cos(v_move, v_commit) with v_commit derived from the same dataset.

## Sweep

α ∈ {−1, −0.5, 0, +0.25, +0.5, +1} (±2 excluded: known-harmful doses), paired arena
seeds, ≥100k rows per α, episode-cluster bootstrap SEs (readings within an episode
are near-duplicates; naive binomial is ~5× optimistic).

## Pre-registered bars (ALL must hold to enable in the trainer)

1. **Causal, right sign**: completion = P(WON | pursued) at α=+0.5 exceeds α=0 by
   ≥ +3pp with the gap ≥ 2× the cluster SE of the difference.
2. **Dose-response shape**: completion(−0.5) < completion(0) < completion(+0.5)
   (strict ordering; the ±1 endpoints may saturate or roll off).
3. **Not commitment in disguise**: pursue-rate = P(pursued | feasible) at +0.5 within
   ±3pp of α=0, AND |cos(v_move, v_commit)| < 0.6.
4. **Canaries at +0.5**: kickoff median first-touch < 5s; touch_ratio and
   goals/episode each within −10% / −20% relative of α=0; overall poss_win not lower
   than α=0 by more than its 2σ.

Fail any bar → do NOT enable; record the result and stop (a capability gap here feeds
curriculum design instead, per the alpha-sweep "can't vs won't" logic).

## If it passes — trainer-side plan (one lever)

Live per-iteration derivation at the existing fnSteerUpdate site (COMPLETED vs
WHIFFED from MATCH-arena rows only, EMA'd), applied at α=0.5σ to a dedicated
sub-slice of the steered practice arenas (commitment keeps its slice; separate
attribution), with its own steered-vs-control **completion gate** and the existing
rating latch covering everything. 1v1 derivation only at first (team pools are thin —
same reason commitment uses the transfer direction there).

## Results (2026-07-15, checkpoint 26025052368, ~26B steps, Rating ~1640)

**VERDICT: all four bars FAIL — movement steering is NOT enabled.**

Two derivations were tested; both are recorded:

1. **Naive contrast** (quarter-scale smoke, checkpoint 26000075780): COMPLETED−WHIFFED
   matched on distance × time only came out at **cos +0.92 vs the commitment
   direction** — completion differences are mostly pursuit-intensity differences;
   dosing it just re-ran the commitment dose-response (pursue-rate doubled at both
   dose signs). Commitment in disguise.
2. **Refined contrast** (full scale: 300k base rows, 200k/α; approach-intensity added
   to the matching bins, result orthogonalized against v_commit before dosing —
   raw cos +0.58, applied cos 0.00, 167 matched pairs):
   - Completion across α = [−1, −0.5, 0, +0.25, +0.5, +1]:
     0.30, 0.42, 0.38, 0.43, 0.43, 0.32 (±0.08–0.10 cluster SE) — **no dose-response
     shape**; ΔCompletion@+0.5 = +4.3pp ± 13.7pp (bar demanded ≥3pp at ≥2σ).
   - **Pursue-rate still rose +10pp at +0.5 (0.22 → 0.32)** even with the commitment
     component projected out — the residual causal effect of this direction is still
     commitment-flavored, not execution-flavored.

## Interpretation (feeds the roadmap)

The trunk appears to have ONE strong causally-steerable behavioral axis — commitment
— and mechanical execution quality is not linearly steerable at trunk granularity.
This is consistent with three independent prior results: team-derived directions are
causally dead (STEERING_TEAM_40 E2), the car reach head self-disables for arbitrary
goals (STEERING_META_40), and the meta system's harmful fold (2026-07-14 incident).

In the alpha-sweep "can't vs won't" frame: commitment was a **won't** (steering moved
it → disposition gap, steer it). Movement is a **can't** (steering cannot move it →
capability gap). Per the pre-registration, capability gaps feed **curriculum**, not
steering — the existing reps machinery (AirDrillState reverse curriculum,
FrontierDrillState resets at collectively-declined states) is the correct lever for
execution, and it already runs. If execution needs more pressure later, the
measurement-backed options are widening frontier-drill useFrac/noise or a dedicated
completion-outcome drill population — not a trunk direction.
