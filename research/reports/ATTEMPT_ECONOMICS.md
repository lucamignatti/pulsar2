# ATTEMPT_ECONOMICS — is skipping the landing rational under current returns?

> **Status: PRE-REGISTERED 2026-08-16 (before the measurement ran).** Follow-on to
> [KD_ROLLOUT.md](KD_ROLLOUT.md). User directive: close the knowing–doing gap;
> whiff-and-learn during training is acceptable; the converged policy must both be
> able to and actually execute what it knows. PBRS attempt-payment is rejected
> (toy-measured inferior to SIL: 5/8 vs 8/8 ignition, ~5× conduct rate). The
> remaining lever choice hinges on ONE fact this measurement decides.

## The question

Among feasible, free ball landings (the KD_ROLLOUT ground-truth population): does
*attending* actually produce better outcomes than *skipping*, under the current
game and reward structure?

- If **skip ≥ attend** (matched): avoidance is return-rational. No amount of
  exploration or success-cloning closes the gap — the return structure must
  change (the user has licensed this). → **Lever B: return-side.**
- If **attend > skip** but the policy skips anyway: the policy leaves measured
  return on the table — attempt *generation* is the bottleneck, and
  exploration/consolidation-side pressure can close it without touching returns.
  → **Lever A: attempt-generation.**
- Whiff share among attempts quantifies the skill deficit: high whiff + attend
  still better ⇒ attempts compound (acquisition wall, worth the transient
  bleed); low whiff ⇒ pure priority failure.

## Protocol

Same offline harness as KD_ROLLOUT (120k player-frames, 16 arenas, live-style
reset mix, ts8 dynamics), snapshots 401B (pre-anneal) and 529B (current) of the
AiMOS fleet lineage. Readings: feasible-free landings, **deduplicated to one per
(episode, player, touchdown-step) event**. Outcomes over the window after
predicted touchdown:

1. **First touch attribution** within 3 s (nearer-car heuristic at the touched
   frame): self / opponent / none.
2. **Canonical ball progress**: team-frame Δ(ball y) from touchdown to +3 s
   (+ = toward opponent goal).
3. **Signed goal** within 7.5 s of touchdown (+1 for, −1 against, 0 none).
4. **Whiff**: attended (≤500 uu at touchdown) but no self-touch within 1.33 s.
5. Critic reads at the anchor (V_real, V_exp, H) for attend vs skip.

Confound control: attendance is endogenous (the bot attends when close), so all
attend-vs-skip comparisons are **matched within required-speed quartiles**
(d_now / t_land) and reported per-quartile plus pooled.

## Pre-registered decision rule

Lever B (return-side) iff, pooled over matched quartiles on the 529B snapshot,
skip ≥ attend on BOTH ball progress and first-touch-is-self. Otherwise Lever A
(attempt-generation), with the whiff share deciding whether to expect a
transient rating cost. Minimum 150 events per arm per snapshot or the
comparison is void (collect more frames). Nothing ships to the cluster from
this report; it only picks which arm gets designed next.

## Results

*(to be filled by `research/tools/kd_economics.py`)*
