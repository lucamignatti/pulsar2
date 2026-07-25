> **Status: SUPERSEDED — archived.** Part of the 4.0-lineage activation-steering
> campaign. Steering is numerically inert on HEAD (`steering.alpha = 0`); the
> Optimistic-Critic Ladder ([LADDER.md](../LADDER.md)) replaced it as the optimism
> mechanism. Kept as provenance for how the program was tested and why it was
> parked. **Do not cite this to justify a new change.**
>
> Offline numbers here predate the 2026-07-19 h2-truncation fix and were computed
> against **pre-activation `h2`** — see [H2_TRUNCATION.md](../H2_TRUNCATION.md).

---

# INTERP_SWEEP2 — interpretability battery #2 (post-steering era)

**Pre-registered 2026-07-15, BEFORE any sweep ran**, per the measurement doctrine.
Checkpoint: **27550023244** (~27.55B steps, Rating/1v1 ~1640), frozen copy at
`build/checkpoints_4.0-interp2/`. All offline CPU, niced + thread-capped.

## Why now

The Phase-0 probe program (rating ~688) found the knowing-doing gap and produced
steering. Since then: commitment steering validated and live (the ONE steerable
axis), movement convicted as a capability gap (MOVEMENT_PHASE0), meta steering
convicted as harmful, team-derived directions causally dead, and the team frontier
identified as **collective decline** (74/88/92% of feasible airborne balls unclaimed
in 1v1/2v2/3v3 at the 12.4B census). This sweep asks: *what else does the trunk
know, and where is the next mis-priced frontier?* Measurement only — no trainer
changes are licensed by this doc; any intervention gets its own pre-registration.

## Stage A — core-probe refresh (1v1, existing pipeline)

Rerun `collect_dataset.py` (150k player-frames, trainer-parity reset mix) →
`label_landing.py` → `train_probes.py` → `knowing_doing.py` against 27550023244.

Registered questions and interpretations:

- **A1 landing representation**: has 15B more steps improved trunk landing R²
  (especially the ~18% bounce slice) relative to the raw-obs ridge and MLP-obs
  controls? If the h2−obs gap on the bounce slice is still ≈ 0, the trunk has NOT
  grown a better ball model on its own → the Phase-1 aux landing-prediction head
  stays the queued representation lever. If it has clearly improved (ΔR² ≥ +0.1
  over obs-ridge on the bounce slice), deprioritize the aux head.
- **A2 knowing-doing gap**: is the gap still knowledge-COUPLED (skip rate graded
  by read quality, as measured post-retune) or has it reverted to knowledge-flat?
  Flat again → incentive-side pressure (steering channel saturated or latched)
  is the binding constraint; coupled → representation-side pressure is.
- **A3 premeditation**: report decodability of upcoming commitment for continuity
  with the Phase-0 record (no decision hangs on it alone).

## Stage B — team collective-decline interpretability (2v2, NEW)

The census number (88% of feasible balls unclaimed in 2v2) is a behavior. This
stage asks what the trunk *believes* during those declines. Rollouts: 2v2 via
`steer_team.rollout_team` machinery with h2 taps, trainer-parity team reset mix,
target ≥ 200k player-rows / ≥ 300 episodes.

Labels (all team-canonical, per-row):

- **Reading**: airborne ball (z > 300), ball-only landing sim, uncensored;
  feasibility per player = required speed < 1300 uu/s (steer_v2 parity).
- **Best-placed**: within a team, the feasible player with the lowest required
  speed; margin = reqSpeed(teammate) − reqSpeed(self) (positive = self better).
- **Pursued** (per player): MOVEMENT_PHASE0 definition — WON the race, or within
  500 uu of the landing at touchdown, or net approach speed ≥ 60% of required.
- **Collective decline**: reading feasible for ≥1 team player, NO team player
  pursued.

Registered probes (ridge/logistic on h2, episode-grouped 5-fold CV, shuffled-label
and obs-ridge controls; decodability bar = held-out AUC ≥ 0.65 or R² ≥ 0.1 above
obs control where the obs already contains the raw ingredients):

- **B1 self-best-placed**: can h2 decode "I am the best-placed teammate for this
  ball" (binary + margin regression)? The obs contains teammate pos/vel, so the
  obs-ridge control is the real bar: does the trunk *compute* the comparison or
  merely carry the ingredients?
- **B2 knowledge-coupling of decline**: among readings where self is feasible AND
  best-placed, is pursue-rate graded by probe-readout quality (the KD analysis
  transplanted to 2v2)?
- **B3 bystander belief** (the novel contrast): probe h2 for "a teammate will
  pursue this reading within 1.5s" (ground truth from the rollout). Evaluate the
  trained probe on collective-decline rows. Pre-registered readings of the result:
  - Probe reads HIGH (teammate-will-go belief) during mutual declines → the
    decline is a **coordination belief error** (each expects the other) →
    who-goes/role-signal lever (own pre-registration required).
  - Probe reads LOW (bot "knows" nobody is going) and still declines → **team
    whiff tax** (mis-priced frontier, the 1v1 story at team scale) → credit-side
    lever, NOT team steering (team directions are causally dead — E2).
  - Teammate-pursuit not decodable at all (< 0.65 held-out AUC) → **representation
    gap** → aux teammate-intent head becomes the Phase-1 candidate alongside the
    landing head.

## Stage C — opponent-model probes (1v1, piggybacks Stage A dataset)

- **C1 opponent future state**: probe h2 for opponent canonical position at
  +0.5s and +1.0s. Control: obs-ridge (which can learn pos+vel·t extrapolation).
  Report ΔR² = h2 − obs at each horizon.
- **C2 race prediction**: "opponent touches the ball first within 2s" — AUC on
  h2 vs obs-ridge.
- Interpretation: ΔR² ≥ +0.1 (or ΔAUC ≥ +0.05) beyond obs at the 1.0s horizon →
  the trunk carries a genuine opponent model → future exploit-style direction
  candidates become plausible (own pre-registration). Otherwise: no opponent
  model beyond passthrough; do not build opponent-conditioned machinery on
  representation grounds.

## Methodology (carried traps — binding)

Team-canonical labels per-row; episode-grouped CV always; per-fold target
standardization for multi-output regression; cluster (episode) bootstrap for
behavioral rates, ~100+ episodes minimum; decodability alone licenses NOTHING —
interventions gate on causal/behavioral deltas in their own pre-registrations;
copy-first checkpoint access; `OMP_NUM_THREADS=4 ... nice -n 19` for everything.

## Results (2026-07-15, all against 27550023244)

Scripts: existing Stage-A pipeline; `team_decline_probe.py` (Stage B, new);
`opp_model_probe.py` (Stage C + A3, new). Raw: `results/metrics.json`,
`results/knowing_doing.json`, `results/team_decline_probe_27550023244.json`
(1.1M-row run; the 700k-row run's numbers are quoted below from its log),
`results/opp_model_probe_27550023244.json`.

### A1 — landing representation: IMPROVED ORGANICALLY (bar met, aux head deprioritized)

150k frames, 26,934 airborne, bounce slice 18.8%. Trunk-over-obs gap GREW since
the 688-era Phase 0 (cross-era caveat: different obs width and lineage; the
comparison is directional):

| | obs ridge | trunk h1 | h1 − obs |
|---|---|---|---|
| R² t_land (now) | 0.613 | **0.817** | **+0.204** (was +0.044 at 688) |
| bounce median err (now) | 995 | **800** | −20% (was 1257→1007, −20%) |
| R² x/y (now) | 0.926/0.962 | 0.951/0.974 | small, as before |

ΔR² t_land +0.20 ≥ the +0.1 bar → per registration the trunk grew a
substantially better ball model without an aux head; **the Phase-1 aux
landing-prediction head is DEPRIORITIZED**. Controls clean (shuffles ≈ 0,
ball-z passthrough 0.95+).

### A2 — knowing-doing (1v1): FLAT by the registered metric, but the metric is stale

P(unattended | trunk knows landing) = 81.9% = P(unattended | doesn't know);
flat across know-err quartiles (83.5/80.4/80.3/83.5). Registered reading:
knowledge-flat → incentive-side. HOWEVER the kd script scores *landing
attendance*, the definition already outgrown once (v1 steering post-mortem);
at this rating the bot converts airborne balls before touchdown. Do NOT act on
A2 until a possession-outcome KD variant confirms it. (The 1v1 pursue-rate from
MOVEMENT_PHASE0 at 26B was 0.22 on feasible readings — decline is real; its
knowledge-coupling is what needs the better metric.)

### A3 — premeditation: stronger than ever

"Self reaches ball within 1s, from >500uu" — AUC h2 **0.961** vs obs 0.720
(shuffle 0.487, n=49k, base 4%).

### B — team collective decline (2v2): the trunk KNOWS and still declines

1.1M rows, 467 episodes; 7,460 readings in 360 episodes (registration ≥300 met);
decline 56.2%, self-pursue 24.9%, teammate-pursues-1.5s base 30.3%.

- **B1 PASS — the trunk computes best-placed**: AUC h2 **0.951** vs obs 0.696
  (shuffle 0.505); margin regression R² h2 **0.459** vs obs 0.073 (both-feasible
  subset). Not an ingredients artifact — the obs carries teammate pos/vel and
  still can't linearly produce the comparison.
- **B2 — coupling weak-moderate and seed-noisy**: pursue(hi-read) − pursue(lo-read)
  at fixed true margin = +0.045 ± 0.059 (700k run) and +0.131 ± 0.044 (1.1M run);
  inverse-variance pool ≈ **+0.10 ± 0.035**. Suggestive positive coupling — 2v2
  decline is not purely incentive-flat — but treat as provisional.
- **B3 — bystander belief reads LOW**: teammate-will-pursue decodes at AUC
  **0.785** (obs 0.629, shuffle 0.488). On collective-decline rows the held-out
  belief is 0.247 vs 0.303 base, and **0.137 on the self-feasible-and-best-placed
  subset** (n=2,109). Pre-registered reading: NOT a coordination belief error
  ("I thought my teammate had it") and NOT a representation gap — the bot knows
  it is best-placed, knows the teammate is not going, and declines anyway →
  **team whiff tax** (the 1v1 mis-priced frontier at team scale).

### C — opponent model: outcome-level YES, kinematic NO

- **C1 FAIL**: opponent future position — obs ridge beats h2 at both horizons
  (+1.0s R² z: obs 0.777, h2 0.682, kinematic extrapolation 0.229). The trunk
  compresses away opponent kinematics; no model beyond passthrough.
- **C2 PASS decisively**: "opponent reaches ball first within 2s" (positional
  proxy, 250uu — dataset has no touch events) — AUC h2 **0.890** vs obs 0.588
  (shuffle 0.483, n=17.7k, base 0.49). ΔAUC +0.30 ≥ the +0.05 bar.

The trunk carries a RACE-OUTCOME opponent model without a kinematic one —
consistent with the reachability InfoNCE pressure (the contact head predicts
exactly this quantity). Per registration this licenses opponent-race-conditioned
*measurements*; any exploit-style machinery needs its own pre-registration.

### Ranked intervention candidates (each needs its own pre-registration)

1. **Team whiff tax** (B1+B3): the exact structural signature that licensed 1v1
   steering, now convicted in 2v2 with belief-level evidence. Team-derived
   steering directions are causally dead (E2) and 1v1-transfer commitment is
   already live — the untried levers are credit-side (team practice value /
   reward shaping within the zero-sum constraint) or transfer-dose retune
   measured on teamWon.
2. **Possession-based KD re-measurement** (A2 follow-up): cheap, blocks/licenses
   any further 1v1 incentive work.
3. **Opponent-race-conditioned analyses** (C2): e.g. does the policy already
   exploit predicted-lost races defensively? Measurement only.
4. Aux landing head: DEPRIORITIZED (A1 bar met by organic improvement).
