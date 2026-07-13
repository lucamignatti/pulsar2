# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Pulsar: a Rocket League 1v1 bot trained by PPO self-play. The trainer is C++
(libtorch + a vendored RocketSim physics sim); on top of it sits a research
program testing whether the bot can be made to *explore its capability frontier*
("optimism") — inspired by Anthropic's global-workspace paper — using linear
probes, a learned self-model of reachability, and collection-time activation
steering. This is not just a codebase; it is a running experiment with a history
of measured successes and instructive collapses. **Read the design rationale
sections below before changing training semantics — most of the non-obvious
constraints here were paid for in Elo.**

The operating doctrine, learned from run 9uz761ua (a proposer/drill mechanism
enabled without evidence decelerated the best run 10x): **measurement before
machinery**. Every intervention needs (1) a measurement convicting the problem,
(2) pre-registered success criteria, (3) an automatic guard, (4) a revert path.
One lever at a time.

## Operations

The trainer usually runs LIVE on the only GPU (RTX 5080, 16GB). Treat the
training process, `build/checkpoints_3.1/`, and the GPU as off-limits for
experiments; all analysis is offline CPU.

```bash
tools/trainerctl status|start|stop|restart   # trainer lifecycle (wraps run_trainer.sh)
tools/trainerctl follow | logs [N]           # live / recent trainer log
tools/trainerctl update                      # git pull --ff-only + rebuild + restart
tools/trainerctl checkpoints | gpu | doctor  # inspection
cmake --build build -j8                      # rebuild main binary (Release, CUDA)
cmake --build build-steer -j8                # compile-check tree (never executed)
```

Key operational facts:
- **Rebuilding `build/` while the trainer runs is safe** (the process keeps its
  inode); the new binary takes effect at the next restart — including the
  wrapper's automatic crash-restart, which is how staged fixes self-deploy.
- The wrapper (`tools/run_trainer.sh`, transient systemd --user service)
  restarts on crashes but NOT on clean exits or `--stop`; on a fast crash-loop
  it backs off `CRASH_LOOP_BACKOFF_SECS` (900) and retries forever — it NEVER
  gives up (a corrupt-checkpoint loop once stranded an unattended run for
  hours). Logs: `run_logs/train-<ts>.log` (`latest.log` symlink; this box's
  Rust-coreutils `tail -f` freezes on symlinks — `readlink -f` first).
- **Checkpoint saves are atomic** (written to `<ts>.tmp`, renamed into place)
  and the loader **falls back newest→oldest across corrupt checkpoints**,
  renaming them `corrupt_<ts>`; the version manager quarantines corrupt/stale
  version dirs the same way. A crash can therefore cost at most one save
  interval, never the run. Root-cause note: CUDA launch-timeout crashes
  (display-attached GPU watchdog) and OOMs under desktop-graphics memory
  pressure are environmental and expected occasionally — the system is designed
  to make them cheap, not impossible.
- **Checkpoints rotate** (`checkpointsToKeep=8`, ~1M steps apart at full speed —
  a ~10-minute window). Always copy a checkpoint dir out before reading it, and
  retry on next-newest if files vanish mid-copy.
- **Never check for the trainer with `pgrep -f`** — it matches its own command
  line. Use: `for p in /proc/[0-9]*/exe; do readlink $p | grep -q build/GigaLearnBot && ...`.
- Offline analysis must be niced and thread-capped or it starves the trainer's
  1024 env threads: `OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
  nice -n 19 ...` (sklearn ignores `torch.set_num_threads`).
- **GPU memory budget**: the trainer needs ~8.5GB steady + learn-pass bursts.
  Desktop graphics (viz viewer, browser) can hold 6GB+; two OOM crashes happened
  under that squeeze. `miniBatchSize` is the burst-peak lever (mathematically
  identical via gradient accumulation); currently 50k for this reason.
- Render/viz: `GGL_RENDER=1` (optionally `GGL_DEVICE=cpu`) runs a single-arena
  live viewer that hot-swaps newer checkpoints. Render mode disables steering.
- Python analysis env: `analysis/probes/requirements.txt` (torch-cpu, sklearn,
  matplotlib, pip `RocketSim==2.2.1`; the vendored engine is 2.1.1 — close
  enough that the policy plays competently, verified by kickoff behavior).

## System architecture

Four layers, bottom-up:

1. **RocketSim** (`GigaLearnCPP/RLGymCPP/RocketSim/`, vendored v2.1.1) — the
   physics engine. 120Hz ticks, deterministic. Needs `collision_meshes/`
   (present in `build/`).
2. **RLGymCPP** (`GigaLearnCPP/RLGymCPP/src/`) — gym layer. `EnvSet` steps 1024
   arenas on a shared thread pool with pluggable ObsBuilders / ActionParsers /
   Rewards / TerminalConditions / StateSetters. The split-step protocol
   implements action delay: `StepFirstHalf` runs `actionDelay` (3) ticks with
   the OLD controls, then new actions are set and `StepSecondHalf` runs the
   remaining tick — so with tickSkip 4 the bot decides at 30Hz with 3 ticks of
   actuation latency, and the obs built after the step carries `prevAction` =
   the action just set.
3. **GigaLearnCPP** (`GigaLearnCPP/src/`) — the learner. `Learner` orchestrates
   collection/processing/PPO; `PPOLearner` owns the models; aux modules:
   Reachability (InfoNCE self-model), PSD (Basin-Racing outer loop), League
   (QD MAP-Elites archive), PolicyVersionManager (Elo skill tracker),
   Proposer/DrillBank (present, disabled — the 9uz761ua regression machinery).
4. **`src/ExampleMain.cpp`** — THE configuration. Everything (rewards, arenas,
   net sizes, all feature flags) is code here, heavily commented with the
   rationale and history of each value. Config changes = edit + rebuild +
   restart. Read this file first; it is the run's lab notebook.

### Networks (1v1, current run "3.1")

```
obs(109, RAW - standardizeObs=false)            AdvancedObs, team-canonical frame
  └─ shared trunk: 2×[Linear512, LayerNorm, LeakyReLU]      "the trunk", h1/h2 taps
       ├─ policy head: 3×[512,LN,LReLU] → 90 logits          DefaultAction table, masked softmax
       ├─ critic head:  3×[512,LN,LReLU] → 1                 GAE value
       └─ reach_phi(trunk ⊕ onehot(action)=602 → 128)        InfoNCE state-action embedding
            vs reach_psi_car(goal6→128)   "can I reach the ball"  (car-local ball pos+vel /2300)
            vs reach_psi_ball(goal6→128)  "can the ball reach the net" (canonical ball pos+vel)
goal_critic: independent net, raw obs → 1                    γ=0.9997 (~77s), β=0.25 std-matched
                                                             advantage blend; long-horizon credit
```

- **AdvancedObs is team-canonical**: x,y are negated for ORANGE so both players
  see themselves attacking +y. Every spatial label/analysis MUST canonicalize
  per-row or silently die (probes read R²≈−0.5 on x/y when this is missed — the
  x/y-vs-time asymmetry is the diagnostic).
- Obs layout: ball 9 | prevAction 8 | 34 boost pads (canonical order, timer-
  blended) | self 29 | opp 29. Pads reverse-order under inversion.
- `DefaultAction`: 24 ground + 66 air = 90 actions; masks by ground/air state,
  boost, flip availability.
- **Checkpoints** (`.lt`, C++ `torch::save`): load directly in Python with
  `torch.jit.load(path, map_location="cpu")`; params named `<seqIdx>.weight`;
  2-D=Linear, 1-D=LayerNorm, index gaps=activations. No forward method — rebuild
  eagerly (see `analysis/probes/load_checkpoint.py`).
- Reachability trains by HER: positives are FUTURE ACHIEVED states within a
  window (car head 20 steps ≈0.67s, ball head 90 steps = 3s), InfoNCE on
  in-batch negatives; rho(s→g) = mean over K uniform VALID actions of
  cos(φ,ψ)/τ — uniform, not policy-sampled: capability ("can we"), not policy
  ("would we"). Calibration (measured): monotone in real achievement, ~5x
  enrichment tail-to-tail, AUC ~0.7 — a COARSE compass; bin it, never trust a
  single rho.

### The collection/learn loop (Learner.cpp — one large function, by design)

`Learner::Start()` contains the whole loop with function-local `Trajectory`
struct and lambdas. Key invariants:

- **Pipelined collection**: a worker thread collects iteration N+1 on a frozen
  model snapshot while the main thread processes+learns N. The clipped IS ratio
  absorbs the one-iteration staleness (stored logprobs come from the snapshot
  that acted). **Barrier zone**: the top of each iteration, worker joined — the
  ONLY place shared state (model weights read by the worker, steering vectors,
  version manager, league, PSD) may be mutated. The shared thread pool belongs
  to the worker during learn; never use `fnParallelFor` from learn-prep (ad-hoc
  `std::thread`s are the pattern there, e.g. the steering landing sims).
- **Episodes are appended whole** to `combinedTraj` at finalize — one player's
  episode is row-contiguous. In-trainer analyses (steering derivation) rely on
  `row+k` = same player k steps later. (Offline datasets from
  `analysis/probes/collect_dataset.py` interleave 2 players per step instead —
  different indexing; this mismatch has caused bugs, check which layout you're in.)
- **GAE terminal semantics** (GAE.cpp): `NORMAL` = true terminal, no bootstrap;
  `TRUNCATED` = bootstrap from a stored next-state value. THE central hard-won
  rule: **whoever changes episode boundaries must let the critic learn the new
  return structure** (see steering post-mortems below).
- The stdout log prints only a curated summary block; the full `Report` (all
  `Steer/*`, `Reach/*`, `GoalCritic/*` panels) goes to wandb via the embedded
  Python metric sender.

### Reward stack (FRONTIER-9, ExampleMain)

Whole-stack invariant: every component is exactly zero-sum or antisymmetric →
the stack sums to 0 across players, so PSD/league fitness is pure competitive
margin. Rules encoded in comments there: PBRS potentials MUST use the learner's
`gaeGamma` (or telescoping breaks); never gate a potential; `ZeroSum(PBRS)` is
still exact PBRS; ShotReward is deliberately absent (its shot-attribution is
phantom-farmable — source-verified exploit in GameEventTracker).

### Outer loops

- **Skill tracker**: Elo (`Rating/1v1`) from eval matches vs
  `checkpoints_3.1/policy_versions/` (32 versions × 25M steps ≈ last 800M steps
  only). **Pool myopia caveat**: the pool cannot measure improvement against
  styles older than its window, and a style shift can read as an Elo dip while
  head-to-head vs recent selves improves (measured: nontransitivity is real
  here). Rating noise band is ±30–50; any guard thresholds must sit outside it.
- **QD League**: MAP-Elites archive over behavior descriptors (quantile-adaptive
  bins), PFSP-sampled opponents on `descendOpponentFrac` of iterations (0.35 —
  raised from 0.25 after measuring exploitability by archived styles).
- **PSD (Basin-Racing)**: ES-style probe rounds gated by `warmupUntilPlateau` —
  designed to engage exactly when Rating stalls. Probe rounds repurpose arenas
  and step them out from under in-flight episodes.
- Eval paths (skill tracker, league, PSD, render) are all UNSTEERED — verified;
  Rating always measures the raw policy.

## The steering system ("optimism surgery") — mechanism and rationale

**Problem (measured, not assumed)**: knowing-doing gap. Linear probes showed the
trunk computes ball-landing information; behavioral lookahead showed the bot
declines feasible plays *independent of how well its trunk reads them* (66%
skip rate at knowledge-flat quartiles, rating ~688). So the bottleneck was
incentive/credit, not representation: a whiffed attempt's counterattack enters
the return (γ half-life ~15s), so E[advantage of attempting] < 0 below a
success break-even — PPO actively teaches avoidance, which starves the data
that could improve success. A one-way trap.

**Mechanism (stage 1, live)** — `CollectSteeringConfig` in LearnerConfig.h;
implementation in Learner.cpp (`fnSteerUpdate`/`fnApplySteering`) and
PPOLearner (`SetSteering`, `InferActions` rho gate):

1. Arena split: ~157 steered + ~27 unsteered *control* arenas + match arenas.
   All run NORMAL episodes (see stage-2 warning). Controls exist purely so the
   treatment effect is measurable in-run.
2. **Direction, derived live every iteration** from the just-collected buffer
   (never from a file — directions go stale within ~75M steps, measured +7pp →
   −11pp): label airborne-ball "readings" via ball-only landing sims (car-free
   arenas on ad-hoc threads) + within-episode lookahead; outcome = first touch
   in the window (self=WON / opp=LOST / none=NONE); direction = matched
   (distance × flight-time bins) difference of trunk means, **WON vs NONE from
   MATCH-arena rows only** (steered data must never feed its own direction;
   LOST is excluded because punishing lost races trains hesitation back in);
   EMA'd (decay 0.9) and applied in the barrier zone.
3. **Application**: steered current-policy rows get `α·σ·v` added to the trunk
   output feeding the POLICY head only (α=0.5, σ = live projection std). The
   learn pass, value preds, and all eval paths never see the delta. Stored
   logprobs are the steered policy's → IS ratios exact.
4. **Rho-band gate**: steer a row only when the CAR head's contact-reachability
   (goal = zeros = "touching the ball") sits in the middle quantile band
   [0.2, 0.8] of the current inference batch — i.e., commit where the RACE for
   the ball is a coin-flip. Per-batch quantiles = self-calibrating across
   checkpoints. (History: v1 gated on the BALL head's scoring goal — "commit
   where the *shot* is uncertain, graded on winning the *ball*" — a semantic
   mismatch that kept the acute effect negative.) The pipelined snapshot
   includes phi/psi so the worker never reads weights mid-update.
5. **Guards (automatic actuators, not dashboards)**:
   - *Possession gate*: steered arenas must win the race on feasible readings
     at least as often as controls. Trips only on ~3σ inversion (−3pp, 150-iter
     warmup — a 0.0-threshold version tripped on noise in minutes, twice);
     while tripped α=0 and the delta EMA decays back across the re-enable line,
     yielding a natural duty-cycled probe. Detects a harmful/inverted direction.
   - *Rating latch*: `Rating/1v1` > 75 below its slow EMA (decay 0.995) latches
     steering OFF for the process, no auto-re-enable. 75 sits outside the
     ±30–50 noise band and inside the −130 collapse signature. This is the ONLY
     guard that can see update-damage; behavioral gates cannot.
   - Ritual: branch-point checkpoint backup (`build/checkpoints_3.1_branch_backup/`)
     before every enablement; quarantine (move, never delete) on revert
     (`build/checkpoints_3.1_quarantine/` holds the two collapse periods).

**Why the specific dose (α=0.5σ) — the clipping ratchet**: for actions steering
makes much likelier than the base policy, the learn-pass ratio r = π/π_steered
≪ 1−ε. PPO's pessimistic min then keeps the gradient for POSITIVE advantages
(unclipped branch) but ZEROES it for NEGATIVE ones (clipped branch) —
**successes reinforce, punished failures are discarded**. At α=1σ this one-way
ratchet compounded lucky overcommits into an Elo bleed while viz looked
"better". Smaller α keeps induced ratios mostly inside the clip window so both
outcome signs teach. Raising α is NOT a free aggression knob; the dose window is
also checkpoint-dependent (offline: +1σ helped, +2σ hurt at the same checkpoint).

**Stage 2 exists but must stay off** (`resolutionTermination=false`;
`AttemptResolutionCondition` + goal-critic row masking are implemented): the
idea was to delete the whiff tax by ending practice episodes at attempt
resolution as true terminals. Two live deployments collapsed Elo within
minutes, for the same structural reason in two guises: GAE baselines every step
against V(s), and a shared critic **cannot price truncated and full episodes of
the same observation** — exclusion left a phantom `−V(s_end)` penalty smeared
over every practice episode ("climb toward it but don't be there when it
arrives": +10pp air time, −10pp engagement, measured); training on all rows
only re-splits the bias (critic learns the blend). A sound retry requires a
dedicated practice-value head keyed on the ROW TAG (not the obs), or
reset-based practice episodes (the AirDrillState pattern, where the critic sees
the boundary coming). Full post-mortems: `analysis/probes/STEERED_PRACTICE.md`.

**Open questions and their planned fixes** (in priority order):
1. *Inference-time steering*: the trained policy beat its own unsteered self
   25–17 when given the vector at inference — an equilibrium of the data
   distribution (skills tuned under shifted activations), not a train/test bug
   (the learn pass optimizes the unsteered function). Fixes: export the live
   EMA vector at checkpoint save + optional steered render/eval mode; track
   plain-vs-steered-self convergence per checkpoint (gap should shrink as the
   policy internalizes the behavior).
2. *Anchor-lead erosion*: lead over the fixed 4.16B anchor read 29–15 → 16–13 →
   13–11 across one evening (small-n each; could be noise, could be
   meta-overfitting to the recent pool). Fix: large-n anchor battery; if
   confirmed, widen the version pool (`tsPerVersion`/`maxOldVersions`) or seed
   the league archive with old anchors.
3. The steering direction is not checkpointed (re-derives within 1 iteration of
   any restart — first iteration always runs unsteered; by design, but relevant
   to 1).
4. `analysis/probes/derive_steering.py` still uses the v1 landing-attendance
   metric — offline analysis only now; the in-trainer v2 possession derivation
   is authoritative.
5. Metric aging is a live risk pattern: the v1 "landing attendance" definition
   was outgrown by the improving bot (it converts via early pressure/bounce
   play). Possession outcomes were chosen because they cannot be satisfied by
   style ("winning the ball first" is good at every level) — but audit any
   behavioral metric against head-to-head results periodically.

## analysis/probes — the measurement toolkit

Pipeline scripts (see its README.md for env + invocation; all offline CPU,
copy-first checkpoint access, deterministic seeds):
`load_checkpoint.py` (jit-load + eager rebuild + `PulsarPolicy` with h1/h2/phi
taps) → `collect_dataset.py` (self-play with exact obs/action/step parity) →
`label_landing.py` (ball-only touchdown sims) → `train_probes.py` (ridge probes,
episode-grouped CV, controls) — plus `knowing_doing.py` (the gap analysis),
`calibrate_rho.py`, `steer_test.py` (offline steering validation),
`derive_steering.py`, `compare_checkpoints.py` (mirror style panels +
head-to-head cross-play with side swap), `kd_curve.py` (gap closure across
policy_versions). The five `*.md` reports there are the experimental record;
`STEERED_PRACTICE.md` is the canonical history/status of the steering system.

Methodological traps already paid for (do not rediscover):
- Team-canonical labels (see above). World-frame spatial targets ruin probes.
- **Episode-grouped CV always** — adjacent frames are near-duplicates.
- In-sample AUC of a 512-d difference-of-means is overfit-inflated (reads 1.0
  at n≈100/class); held-out AUC swings with n. **Gate on causal/behavioral
  deltas, not decodability.**
- Multi-output regression needs per-fold target standardization or the squared
  loss ignores small-scale targets (t_land vs x/y).
- Behavioral metrics have episode-cluster variance ~5x binomial (attendance
  correlates within episodes); ~100 episodes minimum per point.
- Denominators drift under interventions (e.g. "attendance of free landings"
  when steering changes what stays free) — define rates over pre-intervention-
  comparable populations (all feasible readings, contested included).
- Ballistic closed-form lands median 35uu (direct-fall frames are ~solvable);
  the bounce slice (~18%) is where nonlinearity/representation questions live.

## History in one paragraph (why the code looks like this)

Phase 0 probes (rating ~688) showed the trunk linearly exposes modest
bounce-adjusted landing info and a strong *premeditation* signal; the
knowing-doing gap was knowledge-flat → incentive-limited. Rho calibration
licensed coarse frontier selection. The commitment direction proved read-write
(steering engagement causally, dose-dependent). Deployment: two termination
collapses (critic aliasing), reverted via backups; terminationless v1 steering
seeded a genuinely stronger style (beat its branch point 42–24) plus a ratchet
overcommit bleed; v2 re-aimed derivation+gate at possession outcomes, α→0.5,
league 0.35; a real plateau was then measured (dead-even vs self −800M steps),
and the contact-reachability gate retune re-engaged the channel — after which
improvement resumed. The knowing-doing gap has since become knowledge-COUPLED
(graded by read quality), which means representation-side pressure (an aux
landing-prediction head — "Phase 1" of the workspace program) is the next
measurement-backed lever when the current mechanisms saturate.
