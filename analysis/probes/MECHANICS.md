# MECHANICS — why the bot has none: the elimination record (2026-07-15)

The question (user): the bot plays pure positional Rocket League — no aerials,
flicks, wave dashes, speed flips. Mechanistic explanation demanded, not
machinery. This doc records the theory ELIMINATIONS (each measured, same-day)
and what survives. Checkpoints ~30.3-32.0B.

## Eliminated, in order

1. **"Exploration never samples mechanics"** — KILLED by the mechanic census
   (`mechanic_census.py`): fragments are abundant per 100k steps — flip-cancel
   patterns 1261, wavedash-shaped landings-with-speed-gain 82, grounded jumps
   at high balls 19, proto-dribble hood touches 7. Nothing is at zero.
2. **"The policy is too peaked to explore"** — KILLED by the temperature probe
   (`temp_probe.py`): in takeoff states the policy already samples at 3.4 nats
   (the global 0.71 is ground-state peakedness), and T=3 sampling produces
   0/200 aerial touches. Cluelessness, not confidence.
3. **"Below break-even, PPO anti-teaches attempts" (the skill-acquisition-tax
   law)** — SURVIVES for aerials, KILLED as the general law by the break-even
   probe (`breakeven_probe.py`): flip-at-landing is attempted on ~40% OF ALL
   LANDINGS (2,970/rollout), success 11.2%, and attempt-conditioned ΔV(+2s) is
   POSITIVE at 4σ (+0.095 ± 0.023 vs matched controls; correlational caveat:
   contexts self-selected). The sloppy wavedash is fully consolidated; it has
   been stuck at 11% precision for billions of steps.

## What survives: THREE different blockers by mechanic class (no single cause)

| class | blocker | evidence | targeted fix |
|---|---|---|---|
| aerial takeoff (long chain, 0% success) | circuit absent; below break-even; drills never contained the chain start | temp probe 0/300 at any T; AERIAL_GAP M4 vs M3 | approved: takeoff drill v2 (break-even-gated stages) + enlarged height-scaled non-PBRS AerialTouch; practice-value head fixes the pricing side |
| short precision mechanics (wavedash, flip timing) | REFINEMENT plateau at high attempt rate — suspected INTERFACE CEILING: tickSkip 4 + actionDelay 3 gives ±4-tick timing precision | 11.2% success flat for ~billions of steps despite +4σ value signal | OPEN: oracle executor (`oracle_wavedash.py`) must first measure the ceiling; v1 is self-refuting (0.5% < bot's 11.2% — dodge-state setup bug, needs tick-level debugging). If ceiling ≈ bot's rate → structural (tickSkip/actionDelay), a FRESH-RUN parameter, untrainable. If ceiling high → refinement curriculum. |
| flicks (prerequisite-gated) | dribble-state density ~7/100k — the prerequisite never gets practiced | census | dribble-state density (drill or replay) BEFORE any flick work |

## Standing conclusions

- There is no general-purpose mechanism that fixes all three; the blockers are
  heterogeneous. Rare-event precursor replay (proposed earlier) was retracted:
  it addresses none of the three cleanly (aerials have no successes to replay;
  wavedash doesn't lack density; flicks lack a prerequisite, not replay).
- The practice-value head (stage-2 post-mortem's sound retry) remains the
  cross-cutting PRICING fix and is independently motivated by FEAR_MINE's
  measured critic contamination — but it is not sufficient for any mechanic on
  its own.
- Priority order implied by the evidence: (1) aerial package (approved by
  user: drill v2 + non-PBRS height-scaled touch reward; drill is load-bearing —
  height-scaled touch pay alone already failed for 17B steps as TouchHeight),
  (2) oracle v2 to settle the interface-ceiling question BEFORE spending
  anything on precision-mechanic training, (3) dribble density, later.
