# Knowing–doing gap: the bot knows where the ball lands and mostly doesn't go

**Verdict: the gap is on the DOING side.** Among free landings the bot could
physically reach in time, unattendance is nearly flat in how well its trunk knows
the landing point — 64–69 % across the top three knowledge quartiles (66.0 % when
the probe reads the landing within 500 uu, 71.8 % when it doesn't; only the worst
quartile, which is also the aleatorically-hard bounce slice, rises to 77 %). If
ignorance caused the passivity, unattendance would climb steeply across quartiles.
It doesn't. **Knowledge is not the bottleneck — incentive is.** This is the
measurement that says the frontier push should come from optimism/exploration
levers, not (only) from representation pressure.

## Numbers (checkpoint 1461076992, 24,694 airborne readings)

- 522 censored (episode ended first), **1.4 % contested in flight** — almost no
  aerial interception exists in this policy's play, consistent with the known
  ~0.1 % aerial touch ratio of the lineage.
- 6,772 feasible free landings (straight-line arrival budget < 1300 uu/s).

| trunk knowledge (OOF probe err quartile) | P(not at landing spot) | median dist at touchdown | median approach speed |
|---|---|---|---|
| Q1 (best, <~330 uu) | 64.1 % | 676 uu | 100 uu/s |
| Q2 | 68.7 % | 736 uu | 100 uu/s |
| Q3 | 66.8 % | 764 uu | 168 uu/s |
| Q4 (worst — bounce-heavy) | 77.2 % | 1191 uu | 205 uu/s |

- **Danger slice**: free landings within the defensive box area near the bot's own
  goal, feasible to attend: 388 readings, **78.6 % unattended** by the defender —
  worse than baseline, not better; the bot does not prioritize dangerous landings.
- **Aerial passivity**: high balls (>0.75 s flight) the trunk tracked well, within
  2,500 uu: 3,589 readings — in **60.1 % the bot never leaves the ground at all**
  before touchdown. It waits under the ball rather than meeting it — the
  quantified version of "plays like it's afraid of the air".

Params: attend radius 500 uu, contest deviation 300 uu; full sweep-able constants
at the top of `knowing_doing.py`; outputs in `results/knowing_doing.json`,
plot `results/plots/knowing_doing.png`.

## Caveats

- "Go to the landing spot" is not always correct play (shadowing, net coverage,
  boost economy) — the absolute 66 % is soft; the **flatness across knowledge**
  and the danger slice are the load-bearing findings.
- The knowledge proxy is per-frame probe error, which mixes model ignorance with
  aleatoric difficulty (Q4 is mostly bounce frames). That confound *strengthens*
  the conclusion: even ignoring Q4, Q1–Q3 are flat.
- Feasibility is a straight-line speed model; a proper time-to-arrive (turning,
  boost) would shrink the feasible set but not plausibly bend a flat curve.
- Early checkpoint (~1.46 B steps, Rating ≈ 688). Worth re-measuring on later
  checkpoints — if the gap *closes* with rating, PPO fixes it on its own; if it
  persists, the incentive story is confirmed. The pipeline is scripted for this.

## What this buys the "highly optimistic bot" agenda

The bot already computes the information optimism needs. Making it act on that
information is a training-signal problem, in order of least machinery (the
9uz761ua lesson applies — measure, add one lever, re-measure with THIS script):

1. **Outcome-terminated landing/interception drills** (state setter + terminal
   condition, no new learning machinery): spawn ball trajectories with known
   landings, terminate the episode when the attempt resolves — whiffs never enter
   the return, so trying is free by construction. The AirDrillState pattern
   already does the spawn half; it lacks the terminate-on-resolution half.
2. **Attempt-paying PBRS**: a potential on distance-to-(unimpeded landing point)
   during ball flight — exact PBRS (telescopes, refundable, unfarmable), pays for
   *going*, not for arriving. Pairs with the existing AirInterceptPotential which
   pays for climbing.
3. **Optimism-in-the-face-of-uncertainty, later**: once Phase 1 aux heads exist,
   their online prediction error is a principled curiosity bonus — reward play in
   states the bot's own world-prediction finds hard. Requires trainer changes;
   hold until (1)/(2) are measured.

Re-run this analysis after any lever lands: success = the quartile table's
unattendance dropping (especially the danger slice), aerial passivity < 60 %, and
contested-in-flight rising above 1.4 %.
