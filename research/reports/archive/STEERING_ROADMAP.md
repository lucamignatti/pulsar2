> **Status: SUPERSEDED — archived.** A planning document for the 4.0-lineage
> steering program, explicitly flagged stale in `CLAUDE.md`. It plans phases that
> no longer exist: steering is inert on HEAD (`steering.alpha = 0`) and the
> Optimistic-Critic Ladder ([LADDER.md](../LADDER.md)) is the current lever.
> **Superseded in full — do not treat any item here as a live plan.**
>
> Offline numbers here predate the 2026-07-19 h2-truncation fix and were computed
> against **pre-activation `h2`** — see [H2_TRUNCATION.md](../H2_TRUNCATION.md).

---

# Steering roadmap: agreed program, priorities, and rejected branches

> STATUS 2026-07-14: **Phase 0 COMPLETE** on the 4.0 lineage against checkpoint
> 12425039422 — full results + decisions in STEERING_PHASE0_40.md. Headlines:
> the commitment direction is causally real but its POSITIVE side is saturated
> at this checkpoint (0a); random ~ derived at positive dose (0b, outcome (a));
> challenge/shadow style VALIDATED with a dose table, depth a documented
> negative, aerial too thin (0c). Landed in the trainer (resume-compatible):
> team-row exclusion from derivation + team-labeled Steer/PossWin TeamMatch
> panel (the live PHASE B gate duty-cycling was traced to teammate-blind
> possession labels), and the "continuous item" churn-telemetry vector archive
> (steer_vec/steer_sigma in every checkpoint's RUNNING_STATS). 0a's wandb
> ritual run id: alpha-sweep-4-0.
>
> STATUS 2026-07-14 (later, team program — record in STEERING_TEAM_40.md):
> directive changed to "steering must work in all modes; get the whole roadmap,
> python-validated first." E1: the team frontier is COLLECTIVE DECLINE (74/88/92%
> of feasible balls unclaimed for 1v1/2v2/3v3). E2: mode-derived directions are
> causally dead (thin WON pools). E3: the 1v1 direction TRANSFERS to 2v2
> (teamWon +3.4pp @ +0.5, sign-correct panel) → SHIPPED per-mode steering with
> 1v1-direction fallback + per-mode gates/sigmas/panels/slices/archives.
> E4: exploiter steering FAILED its pre-registered bar (32-pair contrast,
> unstable dose) — dropped; styles INFRASTRUCTURE shipped (phase 1:
> steering_styles.json with shadow/hesitant/overcommit, opponent-side, 0.25
> chance, dose windows). E5: frontier-reset reconstruction 100% playable,
> 250/250 perturbation = coin-flip races → SHIPPED phase 3 (FrontierPool +
> FrontierDrillState on practice arenas, useFrac 0.35, staleness-bounded).
> Phase 4 = the per-mode registry structure is its seed; scheduler/battery
> beyond commitment deferred (needs more validated player-side contrasts).
> Render-mode style knob deferred. GGL_SMOKE=1 sandbox smoke affordance added.
>
> STATUS 2026-07-14 (third program - record in STEERING_META_40.md): PHASE 4
> SHIPPED, prior-free per hard requirement (no human priors; a hand-task
> registry draft was discarded). Goals from the agent's own achieved bank;
> frontier by its own self-model; emergent psi-space clusters (never named);
> model-free continuous-attainment outcomes; head-validity self-check (ball
> head passes calibration, car head self-disables for arbitrary goals);
> per-cluster causal gates + dwell scheduler with exploration. Offline: one
> emergent cluster shows monotone causal attainment uplift; heterogeneity
> across clusters = the scheduling signal. cfg.steering.meta=false is the
> pinned incumbent AND the pre-registered live baseline (Elo slope, matched
> window). Commit 86fefb7.

> Drafted 2026-07-13 from a design discussion while the trainer ran unattended.
> Context at time of writing: stage-1 steered collection (terminationless, v2
> possession derivation, contact-rho gate) ENABLED and credited with roughly
> doubling the Elo slope after bootstrap. NOTHING in this file is started —
> every item below requires weights access to test and refine offline BEFORE
> any trainer code is written. This is the plan for that work, in order.
>
> Doctrine applies to every item: measurement convicting the problem,
> pre-registered success criteria, an automatic guard, a revert path, one
> lever at a time.

## Prior-art scan (2026-07-13, for the record / a possible later writeup)

Nothing found doing interpretability-derived activation steering as an
exploration mechanism in embodied/self-play RL. Nearest neighbors:

- **VSPO** (arXiv 2605.15604, May 2026, LLMs) — the same skeleton in
  language models: steering vectors applied to rollout generation so GRPO
  sees rare behaviors ("sparse behavioral reward bottleneck"), unsteered
  policy internalizes via "on-policy latent self-distillation." Differences
  from ours: teacher-specified contrast pairs (not self-derived labels),
  static vector (no staleness handling), no causal control arm, text domain.
- **Maze cheese vector** (Turner/Mini et al. 2023) — established linear
  behavioral directions in a deep RL policy, steered causally; never closed
  the training loop. (Activation steering arguably *started* in RL.)
- **Parameter Space Noise** (Plappert et al. 2017) / NoisyNets — coherent
  whole-episode exploration via weight perturbation; isotropic, not
  semantically targeted.
- **DSRL / latent-noise steering of diffusion policies** (2025-26,
  robotics) — RL over a frozen generative policy's latent space; steering as
  the optimization variable, not as an exploration prior.

What appears novel here: direction discovered from the agent's own
experience with physics-derived labels; live re-derivation (staleness is
real and measured: +7pp -> -11pp in 75M steps); causal control-arm gating;
on-policy PPO with the clipped ratio as the soundness mechanism.

## Phase 0 — ground day: offline groundwork (no trainer changes, CPU only)

All of this runs against copied checkpoints with the usual nice/thread-cap
discipline. Order within the phase is flexible; everything later depends on
it.

**0a. Alpha-sweep dose-response report (accepted item: plateau diagnosis).**
Extend `steer_test.py` into a per-checkpoint diagnostic: sweep alpha over
[-2, +2], report engagement/possession-win/touch/goals vs alpha, and log the
table to wandb (same project, checkpoint timestep as x-axis) so dose-response
history accumulates next to the training panels. Decision value: causally
separates "can't" (steering moves nothing -> capability gap, feed curriculum)
from "won't" (steering moves it -> disposition gap, keep steering). Invocation
stays MANUAL per checkpoint-of-interest — the sidecar-cron history says no
parked Python processes.

**0b. Random-direction baseline (exploratory item 10, feasibility probe).**
Sample N (~8-16) random unit vectors in trunk space plus a few least-squares
orthogonalized variants of the derived direction; run each through the same
paired-seed causal validation as 0a at the production alpha. Three outcomes:
random ~ derived (the effect is partly "any coherent perturbation," i.e.
PSN-in-representation-space — important negative, changes the story and
suggests cheap direction mining); random << derived (hardens the headline
claim); some random direction >> derived (jackpot: gate-filtered direction
search becomes a real method). Cheap, scientifically necessary either way.

**0c. Style-contrast prototyping (seed for league steering AND the meta-loop).**
Define 2-3 style contrasts with pure physics labels, derive directions
offline, and validate each causally with the 0a harness. Candidate contrasts,
all labelable from existing buffer machinery (touch flags, landing sims,
positions):
  - challenge-vs-shadow: on opponent possession, closed distance to ball vs
    held net-side position, outcome-matched;
  - ground-vs-aerial conversion: possession wins via carry/bounce vs via
    flight, geometry-matched;
  - depth: mean net-side offset during neutral play, quantile-contrasted.
Success bar per direction: measurable, sign-correct behavior shift at
alpha in [1, 2] WITHOUT competence collapse (touch rate, goals/ep within
noise of alpha=0). Directions that pass become league styles (phase 1) and
meta-loop battery entries (phase 4). Expect some to fail — style is likely
less linear than commitment; that attrition rate is itself a finding.

Exit criteria for phase 0: dose-response report exists in wandb for the
current checkpoint; random-vs-derived verdict recorded; >= 1 style direction
causally validated (or a documented negative).

## Phase 1 — steered league opponents (accepted)

Style variants of archived checkpoints as PFSP opponents: sample
(member, direction, alpha) instead of member alone. Opponents are
inference-only, so styles cost nothing at runtime and nothing to store
(512 floats each).

- Convicting measurement (already on the books): style nontransitivity —
  the 12-19 loss to the 4.16B self; `descendOpponentFrac` 0.35 is the
  current, compute-expensive patch.
- Implementation sketch: opponent inference path gets an optional
  (vec, sigma, alpha) — `SetSteering` semantics already exist; league
  archive entries gain a style list; PFSP treats (member, style) as the
  sampling atom. Barrier-zone only, same as all opponent swaps.
- Pre-registered success criteria: (i) main-agent counter-play measurably
  differentiates across styles of the same member (per-style behavior
  descriptors from the existing league machinery — if all styles get the
  same answer, the diversity is cosmetic); (ii) exploitability by old styles
  (the anchor battery) shrinks vs the 0.35-descent baseline, allowing
  `descendOpponentFrac` back down.
- Guards/revert: styles are opponent-side only — the rating latch and
  possession gate don't apply; the risk is training-distribution skew, so
  cap steered-opponent share and watch Rating/1v1 slope. Revert = empty the
  style list (config).
- Known limitation to record now: behavioral diversity, not parametric —
  shared-weight perception exploits transfer across all styles of a member.

## Phase 2 — exploiter steering (accepted; strictly after phase 1)

Reuses phase 1 wholesale, opposite intent: steer league opponents along
directions that historically beat the main agent (PFSP already identifies
those members; the derivation contrast is "rows where the archived style
scored/won possession vs main"). Pre-registered bar: a steered exploiter must
raise the main agent's loss rate vs the unsteered same member by a margin
outside episode-cluster noise, else it's a caricature and gets dropped.
Watch for overfitting-to-caricature: anchor battery must not degrade.

## Phase 3 — frontier state setters (accepted, enthusiastic)

Reset-based practice at the frontier: mine each iteration's buffer for
feasible-but-declined readings (already labeled by `fnSteerUpdate`) and
coin-flip-contact-rho states (already scored by the gate), bank a reset pool,
and have a `FrontierDrillState` setter (AirDrillState pattern) reset practice
arenas into perturbed copies.

- Why this is the sound successor to stage 2: reset-based episodes are
  post-mortem option (c) — the critic sees episode starts at resets, learns
  the drill value profile, and NO termination surgery is needed. The whiff
  tax on practice reps shrinks because the episode starts at the attempt
  rather than paying the approach, without touching GAE semantics.
- Composes with steering: reset to the declined state AND steer commitment.
  Deploy separately first (setter alone, steering alone already live), then
  combined — three measurable configurations.
- Pre-registered success criteria: possession-win on frontier situations
  (the `Steer/PossWin` panels, which transfer unchanged) rises faster than
  the phase-0 trend; Rating slope non-inferior.
- Guards: existing rating latch covers it; add a staleness bound on banked
  states (a reset pool older than ~1 iteration re-imports the stale-direction
  problem in state space). Revert = `practiceArenaFrac` of drill arenas to 0.
- Design questions to settle offline first: perturbation magnitude (exact
  replays overfit), pool size/refresh, whether opponent placement is
  replayed or re-sampled, and boost-state handling (don't hand the drill a
  boost prior the real distribution lacks).

## Phase 4 — automatic bottleneck discovery, the meta-loop (accepted: "really what we want")

The destination: generalize from "steer the direction one study found" to
"continuously find and steer the current knowing-doing bottleneck." Highest
effort item; do not start until phases 0-1 have produced 2-3 individually
validated directions — the meta-loop multiplies every per-direction risk by
the battery size.

Architecture sketch (all shapes reuse existing machinery):

- **Contrast registry**: each entry = a physics-derived labeling function
  over the buffer (the possession-outcome pattern generalized), producing
  matched pairs per iteration. Labels must be style-proof by construction —
  the v1 landing-attendance aging is the cautionary tale; possession-outcome
  survived because "winning the ball first is good at every level."
  Initial battery = the phase-0c survivors + the current commitment/
  possession contrast.
- **Per-direction state**: own EMA vector, own sigma, own PossWin-style
  outcome metric, own steered-vs-control causal gate with the calibrated
  thresholds (warmup 150, trip -3pp, re-enable -1pp). The control-arena
  split already exists; per-direction gating means arena groups per active
  direction.
- **Scheduler**: steer ONE direction at a time (doctrine), chosen by largest
  *causal* gap — gate on write, never on read/AUC (the read!=write lesson,
  measured: AUC 0.68 read, -10.7pp write). Hysteresis on switching (a
  direction holds the slot for >= K iterations) or the loop thrashes on
  noise; per-iteration delta se is ~1pp.
- **Costs**: labeling N contrasts per iteration is cheap (landing sims ~free,
  touch scans linear); N trunk-gather passes are the GPU cost — batch them
  into one gather.
- **Guards**: rating latch is global and unchanged (it's the only
  update-damage detector); per-direction gates handle harmful directions;
  add a global "no direction is helping" state where alpha=0 everywhere and
  the loop keeps measuring (the duty-cycle pattern, generalized).
- **Pre-registered success criterion**: the meta-loop must beat the
  single-direction baseline on Elo slope over a matched window, else it's
  complexity for nothing and reverts to phase-1 state.

## Continuous / free items (do alongside anything)

- **Churn telemetry (accepted)**: keep `Steer/Dir Drift` and `Steer/Sigma`
  first-class in wandb; annotate PSD folds and league changes so
  representation-churn events are readable against drift spikes. Archive the
  EMA vector at each checkpoint save (tiny) — this also serves open question
  1 (inference-time steering) and phase-4 forensics, and gives the staleness
  curve for free: validate archived vectors against later checkpoints in 0a.
- **Alpha-sweep-to-wandb** from 0a becomes a per-checkpoint ritual (manual).

## Deployment track — runtime style/difficulty knobs (accepted, wanted)

Strictly a deployment feature, gated on phase 0c producing validated style
directions. Steps: export vectors at checkpoint save (shared with churn
telemetry above); un-disable steering in render mode behind an explicit
`GGL_STEER_STYLE=<name>:<alpha>` env (render currently forces steering off —
keep that default); expose the dose window per direction from its offline
validation. The alpha range that changes style without competence collapse
is narrow (dose tables), so knobs ship with per-direction bounds, not a free
slider.

## Rejected branches (recorded for the possible later writeup, not for reopening)

- **Skill-retention vectors** (regression -> re-steer old direction):
  rejected. Defining "regression" imports human priors or a metric registry
  that ages (v1 metric history); plasticity should be allowed to discard
  unneeded skills; the league IS the retention mechanism with the right
  filter — PFSP resurfaces a style exactly when it wins again, i.e. exactly
  when it's still competitively relevant. Archived vectors also likely rot
  (staleness measurement).
- **Behavioral checkpoint-health probe** (direction projections as corruption
  detector): rejected — kickoff sanity probe + golden archive already close
  the gap; unproven that corruption scrambles linear structure.
- **Steered-teacher distillation** (explicit BC toward steered successes):
  rejected — extra loss term and selection bias solving a problem (slow
  internalization) not yet observed. Trigger to reconsider was noted as: the
  plain-vs-steered-self head-to-head gap failing to shrink across
  checkpoints; even then, not expected to be worth it.
- **Non-Pulsar transfer / paper-first framing**: deferred, not the goal.
  This file and the `*.md` experimental record are the memory if a writeup
  happens later.
