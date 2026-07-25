> **Status: SUPERSEDED — archived.** Part of the 4.0-lineage activation-steering
> campaign. Steering is numerically inert on HEAD (`steering.alpha = 0`); the
> Optimistic-Critic Ladder ([LADDER.md](../LADDER.md)) replaced it as the optimism
> mechanism. Kept as provenance for how the program was tested and why it was
> parked. **Do not cite this to justify a new change.**
>
> Offline numbers here predate the 2026-07-19 h2-truncation fix and were computed
> against **pre-activation `h2`** — see [H2_TRUNCATION.md](../H2_TRUNCATION.md).

---

# Steering Phase 0 on the 4.0 lineage — dose-response, random baseline, style contrasts

2026-07-14, against pinned checkpoint **12425039422** (best_r1386, PHASE B,
Rating/1v1 ~1360 / 2v2 ~292 / 3v3 ~270), copied to `build/checkpoints_4.0-copy/`.
Executes STEERING_ROADMAP.md Phase 0 (0a/0b/0c). Scripts: `alpha_sweep.py`,
`random_dirs.py`, `style_contrasts.py`, shared machinery `steer_v2.py` (v2
possession labels, in-trainer `fnSteerUpdate` parity — NOT the stale v1
landing-attendance metric). Raw tables: `results/alpha_sweep_12425039422.json`,
`results/random_dirs_12425039422.json`, `results/style_contrasts_12425039422.json`;
the 0a table is also on wandb (run `alpha-sweep-4-0`, x = checkpoint timestep).

## Tooling ported to 4.0 first (all verified)

- `load_checkpoint.py` auto-detects the lineage obs width from SHARED_HEAD (109 = 3.1,
  230 = 4.0 padded); checkpoint root prefers `checkpoints_4.0-copy`.
- `advanced_obs.py` gained `build_obs_padded` — **numerically exact** vs the C++
  `AdvancedObsPadded` (golden-dump comparison, max |diff| ~6e-8; slot assignment
  differs only by the independent shuffle draws, presence flags track it).
- `collect_dataset.py` reset mix moved to TRAINER PARITY (35% near-ball / 20% air
  drill / 15% kickoff / 30% random). The old kickoff+random-only mix starved ball
  interactions ~10x vs the live trainer (touch ratio 0.001 vs 0.006), leaving the
  possession labeler with almost no resolved races.
- Behavioral port check: kickoff probe 7/10 touches, median 2.6s (the 3.1-era doc
  says 20/20; part of the gap is real 4.0 behavior — see the kickoff-stall note in
  the caveats — and the obs port is numerically exact, so the port is trusted).

## The live-run finding that frames everything (wandb q8kfp6q0)

Pre-PHASE-B (steps ~10k-24.4k): causal gate solidly ACTIVE, delta EMA mostly +0.01..+0.05.
Since the PHASE B flip (step 24467): the gate **duty-cycles around zero** (trip → decay →
re-engage → re-trip, ~10 cycles), `Steer/Alpha` spends much of the time at 0. Steering has
been effectively dead through PHASE B. Mechanism identified in code: the derivation pool
took rows from ALL match arenas, and in team arenas the self-only possession labels score
a teammate's race win as NONE ("nobody got it") — good second-man play entered the
WON-vs-NONE contrast as "declined". Fixed in the trainer (see Decisions).

## 0a — alpha-sweep dose-response (150k rows/alpha, paired seeds, ~110 episodes/point)

Direction: v2 possession contrast, 146 matched pairs/class, sigma_proj 4.78.

| alpha | possWin ± SE (cluster bootstrap) | in-air | touch | goals/ep | kickoff |
|------:|----------------------------------|-------:|------:|---------:|--------:|
| -2.0  | 5.6% ± 1.6%                      | 85.7%  | 0.12% | 0.09     | 3.08s   |
| -1.0  | 10.0% ± 1.7%                     | 73.5%  | 0.19% | 0.12     | 2.67s   |
| -0.5  | 10.8% ± 1.8%                     | 68.2%  | 0.24% | 0.18     | 3.17s   |
|  0.0  | 11.5% ± 2.2%                     | 59.2%  | 0.24% | 0.18     | 3.13s   |
| +0.5  | 9.7% ± 2.3%                      | 52.5%  | 0.26% | 0.23     | 2.87s   |
| +1.0  | 11.7% ± 2.8%                     | 48.2%  | 0.23% | 0.17     | 2.73s   |
| +2.0  | 14.1% ± 2.1%                     | 41.6%  | 0.24% | 0.24     | 2.82s   |

**Reading (the 0a decision value):** the direction is causally REAL — negative dose
destroys race-winning (~2.7σ at -2) and in-air ratio falls monotonically 86%→42%
across the sweep (clean read-write dose-response). But the POSITIVE side is
saturated: +2σ buys only +2.6pp possession (~1.2σ). At this checkpoint the 1v1
disposition gap has largely closed — "won't" is mostly solved; the remaining
frontier is capability/representation and the TEAM modes (2v2/3v3 races are
nowhere near saturated at Rating ~290). Production alpha +0.5: all canaries clean.

## 0b — random-direction baseline (80k rows/point, production +0.5 and +1.0)

Baseline possWin 7.7% ± 2.3%. Derived: +2.2pp (@+0.5), +4.9pp (@+1.0).
8 random unit directions: deltas -3.7..+7.4pp (mean ≈ +1.6pp). 3 orthogonalized
variants: -0.4..+5.3pp — indistinguishable from derived.

**Verdict: random ~ derived at positive dose** (roadmap outcome (a)) — at THIS
mature checkpoint, the possession effect of the derived direction is not
separable from "any coherent perturbation" at these sample sizes. The derived
direction's unique causal signature lives on the SUPPRESSIVE side (0a: -2σ
halves possession-win; no random direction was observed doing that, but the
negative-dose random arm was not run — noted as the sharper follow-up design).
Consistent with 0a saturation; also a caution against reading the live gate's
positive deltas as direction-specific at plateau.

## 0c — style contrasts (100k rows/point)

- **challenge-vs-shadow** (268/352 readings, 201 matched pairs): baseline
  challenge-rate 0.43. At -2σ → 0.24 (-19pp, sign-correct, strong); -1σ → 0.38;
  positive doses DON'T raise it (0.39/0.34 — saturation again). Canaries at -1σ
  intact (touch 0.23% = baseline), at -2σ degraded but functional (touch 0.14%,
  goals 0.07 vs 0.15). **VALIDATED as a league style** ("shadow-heavy") with a
  usable dose window around -1..-2; ships with its dose table per the roadmap's
  deployment rule.
- **ground-vs-aerial** (only 19 aerial touches → 16 pairs): trend is monotone-ish
  (+2σ → 16.7% aerial share vs 12.7% baseline) but +2σ collapses touch/goals, and
  n is far too thin. **INCONCLUSIVE** — rerun with a drill-seeded or 5-10x larger
  base before judging.
- **depth** (1272 pairs): FAILS causal validation — BOTH signs increase measured
  depth (202 → 318..422). Neutral-play depth is not a linear trunk direction here
  (or any perturbation reads as "hangs back more"). **Documented negative**, as
  the roadmap anticipated ("style is likely less linear than commitment").

Phase 0 exit criteria: dose-response report exists (wandb + results/) ✓;
random-vs-derived verdict recorded ✓; ≥1 style direction causally validated
(challenge) plus a documented negative (depth) ✓.

## Decisions taken (trainer changes, all resume-compatible)

1. **Team-row exclusion from derivation** (`steerPractice` group 3 = team match;
   pool + sigma stay 1v1-only, exactly matched to the steered arenas, which are
   all 1v1 by layout — now enforced with a loud boot error). Targets the measured
   PHASE B gate degradation directly.
2. **Team-labeled possession measurement**: new `teamTouched` buffer; group-3
   readings are scored by TEAM outcome (teammate touch = WON) and reported as
   `Steer/PossWin TeamMatch`. Measurement first — team-mode steering itself stays
   OFF pending that panel's baseline.
3. **Churn-telemetry vector archive**: the live EMA direction + sigma are
   snapshotted into every checkpoint's RUNNING_STATS (`steer_vec`/`steer_sigma`,
   save-only, never loaded). Gives the staleness curve for free and unblocks
   offline inference-time steering studies (roadmap open question 1).

## Deliberately NOT implemented (not "fully ready")

- **Phase 1 steered league opponents**: its convicting measurement (anchor-battery
  exploitability) does not exist on the 4.0 lineage yet, only ONE style validated,
  and this restart already carries the team-play reward/setter batch — bundling
  another lever would confound attribution. Next lever after this batch settles.
- Phase 2 exploiters / phase 3 frontier setters / phase 4 meta-loop: gated on
  phase 1 and on 2-3 validated directions per the roadmap.
- Stage-2 resolution termination: stays off (unchanged post-mortem status).

## Caveats

- Offline stack is RocketSim pip 2.2.1 vs vendored 2.1.1 (known, accepted); fp32
  vs bf16 collection; effects are read as within-harness contrasts (paired seeds),
  never as absolute parity with trainer panels.
- Offline kickoff probe shows occasional 10s kickoff stalls (3/10) the trainer's
  boot probe doesn't; plausibly a mix of version drift and a genuine
  kickoff-stall equilibrium at this Elo — worth a render-mode look sometime.
- possWin SEs are episode-cluster bootstrap; naive binomial would be ~5x smaller
  and wrong (README trap list).
- numpy-on-Accelerate emits spurious FP-exception warnings in big matmuls on this
  Mac; outputs are finite-asserted (`steer_v2.py`).
