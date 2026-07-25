# FEAR_MINE — disagreement-mined frontier drills (dataset quality, the main lever)

**Pre-registered 2026-07-15 BEFORE running.** Checkpoint **27550023244** (frozen
copy). Script `fear_mine.py`.

## The spec (user, 2026-07-15)

"The next lever is dataset quality... force it into suboptimal states that it
thinks could be good but is too scared to commit to." CREDIT_PROBE supplies the
selector for exactly that population: readings where the bot is feasible AND
best-placed AND declines, ranked by Δz = z(goalCritic) − z(critic) — the
long-horizon evaluator says promising, the short-horizon one (which PPO
baselines against) says no. Forcing these states happens through the EXISTING
FrontierPool reset machinery (post-mortem-sound: the critic sees episode
starts), so the live change, if validated, is only the Fill mining criterion.
Both value tensors are already computed every iteration in learn-prep — the
selector is free.

## Design

Phase 1 (mine, one 900k-row 2v2 rollout with state banking): compute V, G, Δz
per row; `decline_readings` labels; build two pools of K=150 states each
(deduped per arena-step block):

- **CURRENT**: live Fill parity — self-feasible readings with outcome NONE
  (nobody touched), stride-sampled (what the trainer banks today).
- **FEAR**: self-feasible AND best-placed AND collectively unpursued, ranked by
  Δz descending (the "thinks it could be good but too scared" tail).

Phase 2 (drill test): reconstruct each state with the validated C++-parity
noise (250uu/250uu/s), roll the frozen policy 5s, 2 repeats — the same
protocol that validated the original frontier drills (`frontier_validate.py`).

## Frozen bars (FEAR must be a BETTER drill source than CURRENT, not just different)

1. **Playability**: ≥95% of FEAR reconstructions finite/playable.
2. **Elicitation**: resolution rate (any touch within 5s) of FEAR ≥ CURRENT − 5pp.
   Forcing scared states must produce outcome-labeled experience, not frozen
   re-declines.
3. **Balance**: reader-team first-touch share among resolved FEAR rolls in
   [30%, 70%] (coin-flip practice — the property that made the original drills
   good training data).
4. **Distinctness**: FEAR pool median Δz exceeds CURRENT pool median Δz by
   ≥ 0.5z (the selector selects something the current criterion doesn't).

PASS all four → deploy ONE lever: the live Fill ranking switches to
Δz-priority over best-placed declined readings (flag-gated, useFrac/noise/dose
untouched, existing latch + staleness bounds cover it, branch backup ritual).
FAIL any → record, no deploy; the mining criterion stays as is.

## 1v1 extension (2026-07-15, same frozen bars): FAIL — 1v1 stays on the original criterion

`FEAR_PPT=1`, raw: `results/fear_mine_1v1_27550023244.json`. Elicitation FAILS
(FEAR resolution 9.3% vs CURRENT 16.7% — the bot re-declines fear states even
when reset into them) and balance FAILS (reader-first 75% when resolved — easy
ignored balls, not coin-flips). Root difference: 1v1 shows NO decline-pessimism
split (median Δz pursued −0.20 vs declined −0.10, against 2v2's +0.09/−0.47).
The critic's fear is a TEAM-MODE phenomenon — the 1v1 disposition gap was
largely closed by the steering era; remaining 1v1 declines are correct or
capability-limited. The deployed team-only scope is confirmed as the right one.

## The census (in-trainer, C++ — user rule: no python automation)

The standing instrument judging this deploy lives in the trainer itself
(fnSteerUpdate section 7; ~zero cost, everything already in hand per
iteration): `Steer/Census NONE Frac` per mode, `Steer/Census Scared Tail`
2v2/3v3 (fraction of best-placed declines with Δz above the WON-readings
median), and the **longitudinal fear panel** — 128 top-Δz 2v2 decline obs rows
frozen ONCE at first ranking after enablement, persisted through
RUNNING_STATS, re-valued by the current critics every iteration
(`Steer/Fear Panel zV / zG / Dz / Age Bsteps`). zV rising toward 0 across
checkpoints = the critic unlearning its fear = this deploy working.
`analysis/probes/fear_census.py` remains as a MANUAL research tool (its
forced-contest selector-calibration check is the one piece too heavy to run
in-trainer); it is not scheduled anywhere.

## Characterization (reported regardless, feeds the reward-diagnostic program)

Median Δz of pursued vs declined best-placed readings and the size of the
high-Δz declined tail — the standing measurement of "how much frontier the
critic is scared of", worth re-running at future checkpoints as the
reward-quality diagnostic (per the user's framing: critic/goal-critic
disagreement identifies when the shaped rewards suck).

## Results

*(to be filled after the run; bars above are frozen)*
