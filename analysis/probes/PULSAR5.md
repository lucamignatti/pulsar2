# PULSAR 5.0 — cold-start design (draft 1, 2026-07-16)

**Decision (user): 4.0 is end-of-life.** ESCALATE-1 runs as its final experiment
while this design matures. The motivating diagnosis, all measured: the 4.0
lineage is PLASTIC (0% dormant units, stable rank, +Elo to 1793) but sits in a
matured equilibrium whose opponent pool prices out learning-phase play, and it
passed its formative high-entropy window before any aerial/mechanic pressure
existed. A cold start is not an escape from bad luck — it is putting the
learned recipe into the formative window where it compounds.

## Structural changes vs 4.0 (each carries its evidence)

1. **tickSkip 8** (4.0: tickSkip 4 + actionDelay 3). Halves every action chain
   in decision-space and doubles per-decision consequence mass (SNR) — the
   measured wavedash plateau (11% success, refinement slower than churn, SNR
   ~0.18/instance) and the Nexto counter-example both point here. Open
   sub-decision: actionDelay 0 (max learnability) vs 2 (sim-to-real fidelity
   for RLBot deployment). Default: 0; revisit only if deployment matters.
2. **RND frontier optimism from step 0** (advantage-side, mean-zero,
   std-matched, w=0.1; persisted nets). In 4.0 it arrived at 35B — after the
   equilibrium. In the formative window, novelty pressure and high entropy
   compound: skills get sampled while opponents are too weak to punish
   experiments. Keep the latch coverage + warmup.
3. **Reward stack (FRONTIER-9 heritage + this week's terms), from birth**:
   - BallToGoalPotential 75, TouchAccel 10, Demo 37.5, Save 25, Goal 150 (the
     proven core, untouched).
   - **AerialTouch 50 / AirIntercept 20** (REBALANCE-1 weights — success must
     clear the acquisition valley while the policy is young).
   - **CarEnergyPotential 15** (tempo credit; farm-proof by telescoping).
   - **TimeCost 0.01** (urgency; uniform constant cancels in margins).
   - **Proximity potential with a far-field linear term**: Phi = exp(-d/1410)
     + eps*(1 - d/12000) — the measured gradient desert (exp saturates beyond
     ~2000uu; ~9% dead frames) gets a nonzero slope everywhere. Still an exact
     potential; still team-closest in team modes.
   - TEAM_SPIRIT schedule: 0.3 in phase A -> 0.6 at phase B (shared-fate trust,
     free-rider bounded while 1v1 dominates).
4. **Practice-value head (RC3) built BEFORE drills scale** — the tag-keyed
   baseline from the stage-2 post-mortem. In 4.0 it was never built; CREDIT_PROBE
   measured the shared critic importing match fear into practice pricing. With
   drills at real dose from the start, pricing isolation is a prerequisite,
   not an option.
5. **Drills at effective dose from the start**: AirDrill (airborne spawns) in
   the mix as in 4.0 AND grounded-takeoff windows via a curriculum controller
   meeting the AERIAL_GAP v2 requirements (drill-outcome attribution via reset
   tags, non-refreshing baseline floor, eval-fleet exclusion, latch coverage,
   days-scale anneal). FEAR machinery (Δz mining, census, frozen panel) armed
   from PHASE B onset with practiceArenaFrac 0.30 / useFrac 0.60 — the 4.0
   dose (6% of team resets) measurably did nothing in 10B steps.
6. **Eval integrity**: skill-tracker fleet excluded from ALL curriculum
   objects (the D=0.9 incident's second lesson); Rating stays a fixed
   instrument for the run's whole life.
7. **Pool-myopia mitigation**: permanently seed the league archive with
   spaced anchors (not just recent versions) so the ecosystem never fully
   forgets styles; keep opponent styles live-synthesized (no files, ever).
8. **Obs/actions unchanged** (AdvancedObsPadded 230, 90-action table): every
   analysis tool, probe, and census carries over — the measurement continuity
   is worth more than any obs tweak we can't justify with data.

## What 5.0 inherits unchanged

Atomic saves + fallback loader + golden archive + boot sanity probe; rating
latch and possession gates; pipelined collection; league/PFSP + PSD (off until
plateau); live-derived steering (commitment) with per-mode slices; the
in-trainer census and Miner observer panels from step 0 — the longitudinal
instruments are born WITH the run this time.

## Success criteria (measurable, with 4.0 as the baseline curve)

The 4.0 record gives waypoint curves we never had before. 5.0 targets, checked
by the same census scripts: aerial-touch share > 4.0's at matched steps by 5B (SCAFFOLD weights: AerialTouch 120 / AirIntercept 40 for the formative window - anneal toward 50/20 once established);
takeoff-probe conversion > 0 by 10B (4.0: 0/300 at 32B); wavedash success
> 20% by 15B (4.0: 11% plateau); decline census NONE < 4.0's matched-step
value through PHASE B. Rating comparisons are lineage-internal only.

Cross-lineage counter caveat: emergence_check.py's per-100k-STEP mechanic
counters (takeoff_attempts, airborne_jump, proto_dribble) count decision
steps, and a 5.0 step (tickSkip 8) spans 2x the sim-time of a 4.0 step
(tickSkip 4 + actionDelay 3) — divide 5.0 counters by 2 (or double the 4.0
curve) before comparing against 4.0 waypoints. Fractions (aerial_touch_frac
and the other *_frac metrics) are per-step-invariant and safe.

## Launch sequence

1. ESCALATE-1 window concludes on the 4.0 run (final data + any surprises).
2. Config freeze of this doc -> ExampleMain 5.0 edits behind a fresh
   checkpoint dir (checkpoints_5.0), PHASE markers reset.
3. 4.0 lineage archived (golden best_r* + final checkpoint + backups dir
   quarantine-copied, never deleted).
4. Boot 5.0; the first fear-panel freeze and census baselines happen in the
   formative window, giving the full longitudinal record from birth.
