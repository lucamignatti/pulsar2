> **Status: SUPERSEDED — archived.** Part of the 4.0-lineage activation-steering
> campaign. Steering is numerically inert on HEAD (`steering.alpha = 0`); the
> Optimistic-Critic Ladder ([LADDER.md](../LADDER.md)) replaced it as the optimism
> mechanism. Kept as provenance for how the program was tested and why it was
> parked. **Do not cite this to justify a new change.**
>
> Offline numbers here predate the 2026-07-19 h2-truncation fix and were computed
> against **pre-activation `h2`** — see [H2_TRUNCATION.md](../H2_TRUNCATION.md).

---

# Steering across modes + roadmap phases 1-3 on 4.0 — validation and build record

2026-07-14 (same day as STEERING_PHASE0_40.md, follow-on program), against pinned
checkpoint **12425039422**. Directive: steering should force commitment to frontier
plays in ALL modes (the frontier differs per mode), and the remaining roadmap phases
should ship where python-first validation supports them. Scripts: `steer_team.py`
(team rollouts + live rho-gate port + team possession labels), `team_steer_validate.py`
(E1-E3), `exploiter_validate.py` (E4), `frontier_validate.py` (E5),
`export_styles.py` (style library). Raw tables in `results/team_*`, `results/exploiter_*`,
`results/frontier_*`.

## E1 — the team frontier exists and dwarfs the 1v1 one (census, 120k rows/mode)

Collective declines (feasible airborne ball, NOBODY on either team goes) on the
trainer-parity reset mix:

| mode | selfWon | mateWon | lost | **NONE** | touch | in-air |
|-----:|--------:|--------:|-----:|---------:|------:|-------:|
| 1v1  | 13.6%   | -       | 12.3%| **74.1%**| 0.26% | 60%    |
| 2v2  | 5.4%    | 1.7%    | 4.6% | **88.3%**| 0.10% | 80%    |
| 3v3  | 2.0%    | 2.0%    | 3.9% | **92.1%**| 0.08% | 87%    |

The "everyone assumes someone else goes" failure is the dominant team-mode state.
Commitment steering has far more to buy in 2v2/3v3 than in (saturated) 1v1.

## E2 — mode-derived directions are causally DEAD (thin pools)

2v2's own WON-vs-NONE direction (149 matched pairs), swept with the live rho-band
gate port: no monotone dose-response (teamWon 5.3/10.4/11.9/4.7/3.1/10.6% across
-2..+2). 3v3 (65 pairs): same noise. Weak team policies produce too few self-won
races to carve a direction — derivation quality follows data quality.

## E3 — the 1v1 direction TRANSFERS to 2v2 (the design-deciding result)

Applying the 1v1-derived commitment direction (314 pairs) in 2v2, rho-gated:

| alpha | teamWon | mateWon | NONE | in-air | touch |
|------:|--------:|--------:|-----:|-------:|------:|
| 0.0   | 11.9%   | 3.3%    | 79.5%| 79%    | 0.11% |
| +0.5  | **15.3%** | **7.5%** | 77.8%| 77%  | 0.14% |
| +1.0  | 12.9%   | 4.2%    | 82.9%| 74%    | 0.14% |
| +2.0  | 12.7%   | 4.8%    | 77.6%| 71%    | 0.14% |

Sign-correct on the correlated panel (teamWon up, in-air monotone down - the
direction's signature - touch up, goals/kickoff clean). Point estimates are ~1σ
each (SE ~4pp); the live per-mode gates adjudicate with ~40x more data per day.

**Design consequence (implemented):** per-mode steering state everywhere (arena
slices, gates, sigmas, panels), but a mode APPLIES the 1v1 direction — dosed by the
mode's OWN sigma — until its own WON-vs-NONE pool clears `minPairsPerUpdate`.
Self-annealing: as team play improves, per-mode directions phase in automatically.

## E4 — exploiter steering FAILS its pre-registered bar (documented, not shipped)

Cross-play main vs the 0.8B-older archived version: races between them are rare
(old side teamWon 3.3%, 32 matched WON/LOST pairs). Steered old side: +4.0pp @ +0.5
but -0.8pp @ +1.0 — unstable, thin. Roadmap's own rule: "else it's a caricature and
gets dropped." Dropped. Re-derive with 10x+ cross-play rows before adding exploiter
styles; the styles INFRASTRUCTURE ships regardless (below).

## E5 — frontier reset reconstruction is sound (phase 3 unblocked)

120 mined frontier states (85% of feasible readings are declines), reconstructed
purely from obs-visible quantities (pos/vel/angVel/forward/up/boost — the same
information the trainer has per row, so no ArenaSnapshot banking needed):

| perturbation | playable | touch-in-4s | first-touch split |
|-------------:|---------:|------------:|-------------------|
| exact        | 100%     | 22%         | 32/20             |
| small (100)  | 100%     | 19%         | 35/10             |
| medium (250) | 100%     | 18%         | **22/21**         |

Medium (250uu pos / 250uu/s vel Gaussian) decorrelates replays into balanced
coin-flip races. Recommendation implemented as the FrontierDrillState default.

## What shipped in the trainer (all resume-compatible)

1. **Per-mode steering** (`steerTeamModes`): per-mode contiguous-block arena slices
   (leading `practiceArenaFrac` of each mode's block = steered + control), per-mode
   rho-band quantiles at inference, per-mode causal gates on TEAM-possession
   deltas, per-mode sigma/EMA/drift/panels (1v1 keeps legacy wandb keys; team keys
   suffixed " 2v2"/" 3v3"), per-mode vector archive in RUNNING_STATS
   (`steer_vec`, `steer_vec_2v2`, ...). Direction fallback per E3. Labels: first
   touch → WON / TEAMMATE / LOST / NONE; pool contrast = WON vs NONE (teammate-
   resolved races excluded); gate metric = team possession.
2. **Opponent styles (phase 1)**: `steering_styles.json` at the repo root (3
   validated styles: shadow, hesitant, overcommit — `export_styles.py`), sampled on
   `opponentStyleChance` (0.25) of opponent iterations, applied ungated to the
   OPPONENT inference call only, alpha drawn from each style's validated dose
   window. Panel: `Steer/Opp Style Frac`. Revert = delete the file.
3. **Frontier resets (phase 3)**: `FrontierPool` (staleness-bounded, per-mode) +
   `FrontierDrillState` wrapping the PRACTICE arenas' setter (`useFrac` 0.35,
   noise 250/250); the learner banks collectively-declined MATCH readings each
   iteration (`Steer/Frontier Pool <mode>` panels). Practice arenas keep 65% of
   the normal mix; empty/stale pool = silent fallback. Revert = don't create the
   pool in ExampleMain.
4. **GGL_SMOKE=1**: permanent sandbox-smoke affordance (128 arenas, 25k ts/iter,
   everything else production) — offline end-to-end validation on the Mac now takes
   minutes instead of hours.

## Deliberately not shipped

- Exploiter styles (E4 fail — see above).
- Phase 4 meta-loop beyond the per-mode registry: the roadmap's own gate is 2-3
  validated directions; the registry structure (per-direction state arrays) now
  exists, and the commitment contrast per mode is its first battery. Style
  contrasts as PLAYER-side steered contrasts remain future work.
- Render-mode style knob (deployment track): vectors are archived per checkpoint
  and the styles file exists, so offline/inference use is unblocked; the
  `GGL_STEER_STYLE` render path is deferred (small, isolated).

## End-to-end sandbox smoke (GGL_SMOKE, resuming the REAL 12.4B PHASE B checkpoint)

55+ iterations on the Mac (CPU, sequential, offline wandb), zero errors, with every
mechanism live and reporting: per-mode alphas 0.5 in all three modes, per-mode gates
active, per-mode sigmas (2.04 / 1.64 / 1.47), own directions derived in every mode
(Pairs 115/120/155), frontier pools banked 174/177/128 entries per mode, opponent
styles engaging (Opp Style Frac 0.17), all three Rating/<mode> evals running, and
`Player/Ball Touch Ratio` 0.0055 in-smoke vs 0.0058 on the live box (behavioral
parity). Boot: PHASE B fleet from the real marker, checkpoint + 32 versions + 170
league members loaded, 3/3 kickoff boot probe.

## Mac smoke ops findings (recorded for the next sandbox session)

- **torch-cpu + Accelerate concurrency bug (macOS)**: with pipelined collection, the
  first learn pass overlapping the collect worker's inference produced NON-FINITE
  trunk outputs from provably finite weights and obs (100% of rows, deterministic
  timing = the moment concurrency starts). MPS separately dies in bf16 multinomial.
  Neither exists on the Linux/CUDA box. GGL_SMOKE therefore forces fp32 + SEQUENTIAL
  collection (also makes smokes deterministic). The diagnostic that pinned this — a
  finite-check on logits with source attribution (raw obs / trunk out / weights /
  delta) in `InferPolicyProbsFromModels` — is permanent; it turns any future NaN
  crash from an opaque multinomial assert into a one-line diagnosis.
- The embedded metric sender needs a python3.14 wandb (PYTHONPATH to a venv's
  site-packages) and `WANDB_MODE=offline`; strip `run_id` from the sandbox
  checkpoint copy's RUNNING_STATS or the smoke resumes the LIVE wandb run.

## Caveats

- All offline caveats from STEERING_PHASE0_40.md apply (RocketSim 2.2.1 vs 2.1.1,
  fp32-vs-bf16, one-shot samples vs live-gate statistics).
- E3's effect sizes are ~1σ point estimates; the shipped mechanism's real
  verdict comes from the live per-mode gate deltas and Rating/2v2, Rating/3v3
  slopes — the gates fail closed (alpha → 0 per mode) if the transfer doesn't
  reproduce live.
- Opponent styles shift the training distribution; watch `League/*` +
  Rating panels after enabling; the styles file is data, so reverting or
  re-deriving styles requires no rebuild.
