# Steered-practice collection: design, implementation, A/B protocol

> **STATUS UPDATE (2026-07-12, late): STAGE 1 (terminationless) ENABLED.**
> Normal episodes everywhere; steering only changes where experience comes from.
> New since the failures: rho-band gating (steer only in states the ball head
> rates as hard-but-plausible for scoring — per-batch quantile band, ~156 steered
> + ~27 control arenas), and a **rating drawdown guard** (Rating/1v1 more than 25
> below its slow EMA latches steering off for the process — the collapse signal
> made an actuator). Branch backup: `checkpoints_3.1_branch_backup/4703169536`.
> Stage 2 (termination + practice-value head) remains gated behind
> `resolutionTermination`, which must never be enabled without that head.
>
> **Stage-1 first-night findings (2026-07-12, ~116M treated steps):**
> - Head-to-head, treated **beats its branch point 42–24** (p≈0.02) and kicks off
>   faster (3.08 vs 3.63 s), but **loses 12–19 to the older 4.16B self** — pool-Elo
>   drift is (at least partly) style NONTRANSITIVITY, not weakness; the league's
>   old-style exposure is the natural corrective.
> - The engagement gate duty-cycles (real −3pp trips): steering along the current
>   direction acutely suppresses "landing attendance" — but the bot wins anyway.
>   The METRIC IS AGING: the improving bot converts via early pressure/bounce play,
>   not touchdown camping. Both the direction derivation and the gate need a v2
>   definition based on POSSESSION OUTCOMES (who wins the next touch and what it
>   leads to) before the steering loop optimizes a stale concept.
> - Guard calibration lessons: rating trip must sit outside the ±30–50 normal
>   wiggle band (75, slow EMA); engagement gate must trip only on inversion
>   (−3pp ≈ 3σ, 150-iter warmup) — the naive 0.0/20-iter settings tripped on
>   noise within minutes, twice.
>
> **v2 (same night): possession-outcome retarget.** User kept steering on (the
> qualitative + head-to-head gains were real; pool-Elo still drifted down under
> the stale v1 metric). The direction derivation and the gate now run on
> POSSESSION outcomes: first-touch scan from each feasible reading to shortly
> past touchdown — self touch = WON, opponent = LOST, neither = NONE. Direction
> contrast = WON vs NONE (LOST excluded: punishing lost races trains hesitation
> back in); gate = possession-win rate steered-vs-control (`Steer/PossWin *`
> panels). Unfakeable by empty flight, doesn't age with style. Also
> `league.descendOpponentFrac` 0.25 → 0.35 to patch the measured old-style
> exploitability (PFSP targets exactly the members that beat the main).
>
> **Plateau + contact-gate retune (2026-07-12, late).** After the v2 recovery
> uptick, head-to-head showed REAL stagnation (dead even 12-13 vs the self from
> 800M steps prior; anchor lead shrunk 29-15 -> 16-13) - not Elo-pool saturation.
> The steering gate had idled all evening on a semantic mismatch: gated by
> SCORING-reachability coin-flips, judged on RACE-winning. Retuned to the car
> head's CONTACT-reachability band (`rhoGateOnContact`, default true): commit
> where the race for the ball is a coin-flip, judged on winning that race -
> aim and ruler finally aligned. Staged in build/; picked up at next restart.
> PSD's warmupUntilPlateau machinery is expected to engage on its own now.
>
> **Original STATUS (2026-07-12) after two live failures — read this first.**
> Both failures came from the RESOLUTION-TERMINATION half; the steering half never
> got a clean test (failure 1's damage flows from the termination alone — the
> unsteered control arenas produce it too).
> Failure 1: critic excluded from practice rows → GAE charges ~ −V(s_end) as a
> phantom penalty over every truncated episode → policy unlearns ball engagement →
> Elo freefall in minutes. Failure 2 (the "fix"): mixture critic on all rows —
> insufficient, because the critic sees identical obs for practice and match rows
> and can only learn the blend, leaving ~(1 − practiceFrac) of the bias in place →
> still tanking. The aliasing is STRUCTURAL: no shared critic can price truncated
> and full episodes of the same state.
> A sound retry requires ONE of: (a) a dedicated practice-value head keyed on the
> row tag (not the obs) used as the baseline for practice rows; (b) steering
> WITHOUT termination — pure behavior-policy exploration, PPO-sound, keeps the
> whiff tax; (c) practice as separate reset-based episodes (AirDrill pattern),
> where the critic learns the drill value profile because episodes start at
> resets, not mid-play cuts.
> Rating was restored from the branch backup after each failure; tanked
> checkpoints are in `build/checkpoints_3.1_quarantine/`. Everything below is the
> original design + first post-mortem, kept for the retry.
>
> **Quantified aftermath** (`compare_checkpoints.py`, baseline 4163149824 vs
> treated 4247282688; results/checkpoint_comparison.json): the treated bot flies
> +10 pp more (38.7 → 48.1 % in-air — the "looked better in viz" impression) but
> engages LESS (landings 29.2 → 19.4 %, contests 0.8 → 0.4 %, touches 0.24 →
> 0.18 %, mirror goals 0.21 → 0.12/ep); head-to-head a wash (22–18 over 82 eps).
> Mechanism, precisely as diagnosed: the phantom penalty concentrates at
> RESOLUTION states (episodes end there) while the approach is still paid — the
> gradient taught "climb toward it, but don't be there when it arrives." The
> shaping channel is powerful and placement-precise; the retry must fix
> last-step credit and validate against exactly this table.

The trainer-side implementation of "optimism surgery": optimistic behavior during
collection, sober learning target, punishment-free practice reps at the bot's own
frontier. Everything is flag-gated and OFF by default — with `cfg.steering.enabled =
false` the trainer is byte-identical to the 3.1 baseline. **Do not enable while the
current run is live**; this is the A/B for the next run.

## Why (one paragraph)

Measured chain, all in this directory: the trunk knows more than the bot acts on
(KNOWING_DOING.md — declines are knowledge-independent); its capability model can
locate the frontier (RHO_CALIBRATION.md — monotone, coarse); and a read-write
"commitment" intention variable exists in trunk-output space (STEERING.md — matched
AUC 0.82–0.89, causal behavior shift ±, dose window α ∈ [+1, +2)). The remaining
blocker is credit assignment: below break-even success, every attempt's counterattack
tax teaches avoidance. This design fixes the data half (steering generates coherent
frontier attempts) and the gradient half (resolution-terminated practice episodes
delete the tax) while matches stay untouched as the disciplinarian.

## Mechanism (what the code does)

- **Arena split**: the first `numGames * practiceArenaFrac` arenas are practice
  arenas; the rest are match arenas. Both sides of the split feed the same PPO
  learn pass.
- **Steering** (collection only): practice-arena current-policy rows get
  `alpha * sigma * v` added to the shared-trunk output feeding the POLICY head.
  Never steered: the learn pass, GAE value preds (separate `InferCritic` pass),
  skill-tracker evals, old-version/league opponents. Stored logProbs are the
  steered policy's, so PPO's clipped importance ratio absorbs the divergence — the
  same mechanism absorbing pipelinedCollection's one-iteration lag.
- **Termination** (`AttemptResolutionCondition`, practice arenas only): arms when
  the ball rises above 300 uu, ends the episode as a **NORMAL terminal** when any
  player touches it, it lands untouched, or 6 s pass armed. True terminal = GAE
  does not bootstrap V at the cut. This is load-bearing: a truncation-bootstrap
  would re-inject the punishment through the critic's prior and make the whole
  mechanism a no-op. Attempt quality is already priced by the existing stack
  (TouchAccel / AerialTouch / AirInterceptPotential) — no reward changes.
- **Critic semantics** (POST-MORTEM, first deployment 2026-07-12): v1 excluded
  practice rows from BOTH critics to avoid aliasing truncated practice returns
  with match returns. That was the wrong trade and caused an immediate Elo
  freefall: with the critic pricing frontier states at full match value, GAE
  charged ~ −V(s_end) as a phantom penalty over every truncated practice episode
  (λ smears the terminal delta across the whole short episode), so ~20 % of the
  data actively taught the policy to disengage from the ball — a manufactured
  punishment in place of the deleted one. Fix: the MAIN critic trains on all rows
  (V learns the honest 80/20 mixture; the residual match-value bias at frontier
  states is the far smaller error, escalation path = dedicated practice-value
  head). The GOAL critic stays excluded (its ±1 channel is structurally absent in
  resolution-terminated episodes) and practice rows get no goal-advantage blend.
- **Direction supply — LIVE, in-trainer** (no file, no sidecar): every iteration,
  during learn-prep, the trainer labels airborne-ball readings from its own
  just-collected buffer (ball-only landing sims on ad-hoc threads + within-episode
  lookahead — episodes are row-contiguous in the buffer), builds matched
  went/declined pairs from MATCH-arena rows only (steered data never feeds its own
  direction), and EMA-folds the trunk-mean difference into the active vector
  (`fnSteerUpdate` in Learner.cpp; applied in the barrier zone, race-free with the
  pipelined collect worker). Iteration 1 runs unsteered; steering engages from
  iteration 2. This exists because directions go stale FAST: a vector measured at
  +7 pp engagement was −11 pp only 75M steps later.
- **Causal auto-gate**: a slice of practice arenas (controlFracOfPractice) runs
  UNSTEERED with the same resolution-termination. Steered-vs-control engagement is
  the pure steering effect, EMA'd each iteration; if it goes non-positive the gate
  drops alpha to 0 (derivation continues) and re-engages when it recovers. This
  replaced the offline sidecar gate — per-frame AUC proved a weak proxy (in-sample
  values overfit-inflate in 512-d; held-out values swing with sample size), and a
  parked Python cron was an operational liability. The behavioral delta is what
  the trainer consumes, so the trainer measures it itself, continuously.

Known v1 scope limits, deliberate: the resolution rule is aerial-only (ground
dribble/flick chains that pop the ball above 300 uu will end practice episodes at
the next touch — the rule is a single swappable header,
`AttemptResolutionCondition.h`); steering is spatially uniform (no rho-band gating
yet — measured dose +1σ is safe uniform); both practice-arena players are steered.

## Files touched

- `GigaLearnCPP/src/public/GigaLearnCPP/LearnerConfig.h` — `CollectSteeringConfig`
  (arena split, EMA/gate knobs)
- `GigaLearnCPP/src/public/GigaLearnCPP/Learner.{h,cpp}` — live derivation
  (`fnSteerUpdate`), barrier-zone apply, steered/control/match arena split + row
  tagging, goal-blend guard, `Steer/*` metrics
- `GigaLearnCPP/src/private/GigaLearnCPP/PPO/PPOLearner.{h,cpp}` — `SetSteering`,
  `steerDelta` through the inference path, masked critic losses
- `GigaLearnCPP/src/private/GigaLearnCPP/PPO/ExperienceBuffer.h` — `practiceMask`
- `GigaLearnCPP/RLGymCPP/src/RLGymCPP/TerminalConditions/AttemptResolutionCondition.h`
- `src/ExampleMain.cpp` — wiring, ENABLED (practice arenas get the terminal condition)
- `analysis/probes/derive_steering.py` — now an OFFLINE analysis tool only (the
  trainer no longer reads any vector file)

## Resume protocol (ENABLED as of 2026-07-12; takes effect on next trainer restart)

The comparison is before/after on the same 3.1 lineage (single-GPU reality — no
parallel arm). State at enablement:

- `cfg.steering.enabled = true`, alpha 1.0, practice frac 0.2 (15 % of practice
  arenas as unsteered controls); `build/` rebuilt. Nothing else to run: the
  direction derives itself from iteration 1's buffer and steering engages from
  iteration 2.
- Operating point justified offline at the branch checkpoint: **+4.7 pp
  engagement at +1σ**, while +2σ measured **−10.7 pp at the same checkpoint** —
  the dose window is checkpoint-dependent, which is exactly what the continuous
  in-trainer gate handles (and why alpha should not be raised casually).
- Branch-point backup: `build/checkpoints_3.1_branch_backup/4163149824` (full
  checkpoint incl. optimizers + RUNNING_STATS, outside the rotation window;
  also the run's exact final state — the stop landed on this checkpoint).
  REVERT = set `enabled = false`, rebuild, copy the backup into `checkpoints_3.1/`,
  restart.

Pre-registered success criteria (all instrumented, zero manual steps):
- `Steer/Engagement Steered` > `Steer/Engagement Control` (the pure steering
  effect; `Steer/Gate Delta EMA` is its smoothed version, and `Steer/Gate Active`
  flips to 0 automatically if it goes non-positive).
- Player/Aerial Touch Ratio and in-flight contest rate rise within ~a day
  (baselines: ~0.001 % and 1.4 %); knowing-doing unattendance (66 % @
  feasible+known) falls — re-measure with `knowing_doing.py` per checkpoint.
- `Steer/Dir Drift` and `Steer/Pairs` sane (drift spikes flag representation
  churn, e.g. around PSD folds); GAE avg ratio / KL in the pipelined-collection
  envelope.
- Rating/1v1 slope vs the pre-switch trend; sustained deficit = revert.

Known interaction wrinkles (accepted, unaudited): PSD probe rounds inherit
practice-arena termination semantics (~20 % of probe arenas → probe-fitness noise);
league iterations get fewer goal events from practice arenas (mild exploiter-signal
dilution). Set `psd.enabled = false` for the cleanest first read if desired.

Anneal: as contest metrics rise, step `practiceArenaFrac` down (0.2 → 0.1 → 0.05);
the scaffold should retire itself.

## Failure containment

Worst case (assumptions 4/5 of the mechanism fail): ~20 % of collection spent on
uninformative practice data, visible within hours on the Steer/* + Rating panels,
reverted by one config flag. No new networks, no learned goal machinery, no reward
edits, no mid-match modifications — the two 9uz761ua failure patterns are absent by
construction.
