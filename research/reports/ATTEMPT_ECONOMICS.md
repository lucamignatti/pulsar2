# ATTEMPT_ECONOMICS — is skipping the landing rational under current returns?

> **Status: RESULT 2026-08-16 — verdict LEVER A (attempt-generation). Attending
> beats skipping on every metric in every matched quartile on both snapshots;
> the policy leaves large measured return on the table. Additional mechanism
> finding: V and H are ~flat across attend/skip anchors — the headroom gate is
> BLIND to this frontier, which is why H-gated SIL/entropy never targeted it.**
>
> **Original status: PRE-REGISTERED 2026-08-16 (before the measurement ran).** Follow-on to
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

## Results (2026-08-16, `research/tools/kd_economics.py`, 300k frames/snapshot)

Feasible-free landing EVENTS (deduped): 529B — 1348 (263 attend / 1085 skip);
401B — 1200 (460 / 740). Both arms far above the 150 minimum.

| metric | 529B attend | 529B skip | 401B attend | 401B skip |
|---|---|---|---|---|
| first touch = self | 55.5 % | 19.8 % | 64.8 % | 32.0 % |
| first touch = opponent | 14.1 % | 25.1 % | 25.2 % | 49.1 % |
| ball progress @+3 s | +591 uu | −140 uu | +305 uu | −200 uu |
| signed goal @+7.5 s | +0.103 | −0.019 | +0.050 | 0.000 |
| V_real at anchor | −0.77 | −0.79 | +0.54 | +0.40 |
| H at anchor | 2.92 | 3.00 | 2.98 | 3.09 |

Matched (required-speed quartiles, pooled): attend − skip = **+794 uu / +36.6 %
first-self** (529B), **+500 uu / +32.5 %** (401B). The advantage holds in all
eight quartile cells including the easiest (q0). Whiff among attempts: 54.4 % /
45.9 % — half the attempts still win first touch inside 1.33 s, and the
*average* attempt is strongly net-positive despite the whiffs.

**Pre-registered verdict: LEVER A.** Skipping is not return-rational; attempt
generation is the bottleneck. (Causal caveat: observational, matched only on
required speed — residual endogeneity is possible, but the effect is huge,
uniform across difficulty, and in the same direction on both snapshots.)

**Mechanism finding (unregistered, load-bearing):** V_real and H barely
separate attend from skip anchors. V flat is *correct* on-policy valuation —
the critic prices what the policy will do, and the policy will skip. But H
flat means **the composition critic is not pricing this achievable-but-
unrealized conduct**, despite the pieces (and even the whole, at 20–38 %
attendance) being in the data. Since every live actuation channel (SIL gate,
entropy gate) keys on H, the machinery that exists to close exactly this kind
of gap never fires on it. The gap isn't stuck because SIL is weak — it's stuck
because the gate is blind.

## Licensed next step (user directive: close the gap; whiffs acceptable)

Two arms for an AiMOS A/B vs config-identical control (never the fleet), to be
pre-registered separately before build:

- **Arm SIGHT (general fix):** aux landing-prediction head on the shared trunk
  (the long-deferred "Phase 1" workspace lever) — self-supervised from
  rollouts (realized touchdown, HER-style), forcing the trunk/critic side to
  expose the opportunity so H starts separating these states and the EXISTING
  actuation targets them. Gate: H attend/skip AUC > 0.6 within the arm budget,
  then attendance rises.
- **Arm SEEK (targeted proof):** entropy-gate keyed on an engine-computed
  in-flight signal (ball-only landing sim + feasibility + no-attender check in
  learn-prep — the steering-era landing-sim pattern; no resets, no reward
  change, entropy-only, the measured-safe channel class). Gate: attendance
  among feasible-free +15 pts vs control at matched steps, anchor-Elo cost
  within a pre-set budget.

Success for the program = the ATTEMPT_ECONOMICS attend share rising and the
KD_ROLLOUT unattendance curve bending down at fixed reward config.
