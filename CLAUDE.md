# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Pulsar: a Rocket League self-play bot trained by PPO. Core play is 1v1; the
current run ("5.0v3") also opens 2v2/3v3 team-play phases (PHASE B). The trainer
is C++ (libtorch + a vendored RocketSim physics sim); on top of it sits a
research program testing whether the bot can be made to *explore its capability
frontier* ("optimism") — inspired by Anthropic's global-workspace paper — using
linear probes, a learned self-model of reachability, RND novelty pressure, and —
currently — an **Optimistic-Critic Ladder** that turns the bot's own
knowing-doing gap into training signal (it superseded the earlier
activation-steering mechanism, which is parked but still in the code). This is
not just a codebase; it is a running experiment with a history of measured
successes and instructive collapses. **Read the design rationale sections below
before changing training semantics — most of the non-obvious constraints here
were paid for in Elo.**

> **Run identity note.** Much of the file below was written for the older "3.1"
> 1v1 run. Where it still describes hard-won *rationale* (recovery doctrine,
> measurement-before-machinery, critic-aliasing post-mortems, methodological
> traps) it remains authoritative. Where it states *current facts* (engine
> version, tickSkip, obs/net dims, reward weights, checkpoint dirs) the truth is
> now the 5.0v3 run: RocketSim **v3** (Rust FFI), **tickSkip 8 / actionDelay 0**
> (15 Hz), **230-dim padded obs**, **517-wide policy head**, `checkpoints_5.0v3`.
> Sections are updated inline; `research/reports/PULSAR5.md` (cold-start design)
> and `research/reports/LADDER.md` (Ladder deploy record) are the canonical
> current-run docs.

The operating doctrine, learned from run 9uz761ua (a proposer/drill mechanism
enabled without evidence decelerated the best run 10x): **measurement before
machinery**. Every intervention needs (1) a measurement convicting the problem,
(2) pre-registered success criteria, (3) an automatic guard, (4) a revert path.
One lever at a time.

## Operations

The trainer usually runs LIVE on the only GPU (RTX 5080, 16GB). Treat the
training process, `build/checkpoints_5.0v3/`, and the GPU as off-limits for
experiments; all analysis is offline CPU. (`trainerctl` reads
`checkpoints_5.0v3` by default — `TRAINERCTL_CKPT_DIR` overrides.)

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
- **KNOWN RESIDUAL GAP — corrupt-but-loadable checkpoints** (2026-07-13
  incident): a GPU lockup (kernel `Xid 8`, "GPU is probably locked") corrupted
  device-to-host reads during its ~50s onset (saves are ~6s apart at tsPerSave
  1M); most checkpoints from that window were truncated (caught by the
  fallback) but at least one **loaded fine with finite, sane-magnitude weights and a
  behaviorally destroyed policy** (1/10 kickoff touches, 89% aimless air time).
  No structural check can catch these. Detection: the in-run rating guard trips
  (it did), and offline a 2-minute kickoff test
  (`research/tools` — healthy ≈ 10/10 touches, median ~3.4s) is definitive.
  Automatic defenses (same day): the boot sanity probe (3 kickoff episodes on a
  throwaway arena, >= 2 must have touches, only for checkpoints claiming rating
  >= 400), a golden archive keeping the top-3 rated checkpoints outside rotation
  as the loader's last resort ("best_r<rating>_<ts>"), and tsPerSave raised to
  25M so the rotation window spans ~20 minutes instead of ~50 seconds.
- **RECOVERY DOCTRINE (user-set, binding): NO recovery machinery.** If a
  checkpoint is corrupt/damaged, restore a KNOWN-GOOD FULL checkpoint (golden
  archive or `checkpoints_5.0v3_branch_backup/`), however old; if none exists,
  start a fresh run. NEVER hand-assemble hybrid checkpoints (healthy policy +
  borrowed critic/optims): it was tried once and 550M steps of training against
  the mismatched value baseline made the policy WORSE than its restore point
  (measured 16-31 head-to-head) while every guard stayed green. Partial
  restores create subtle damage that no warmup machinery should exist to
  compensate for. Full checkpoint or fresh start, nothing in between.
  Before any restore: quarantine (never delete) the entire damaged lineage —
  numbered checkpoints, best_r* entries, and policy versions descended from it —
  so the loader cannot prefer them; versions newer than the restored timestep
  auto-quarantine at boot.
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
- Render/viz: `GGL_RENDER=1` (optionally `GGL_DEVICE=cpu`, `GGL_RENDER_TEAM_SIZE`)
  runs a single-arena live viewer that hot-swaps newer checkpoints. Render mode
  feeds zero steering / zero Ladder-wire values and never runs a learn pass, but
  it MUST still build the model with the Ladder architecture flags
  (`gapSensor.wireEnabled`) or it loads a 512-wide policy head against 517-wide
  checkpoints — the exact bug fixed in `ba4f33a`.
- Python analysis env: `research/tools/requirements.txt` (torch-cpu, sklearn,
  matplotlib, pip `RocketSim==2.2.1`). The vendored training engine is now
  RocketSim **v3** (Rust); the offline harness was migrated to 5.0 dynamics
  (tickSkip 8, actionDelay 0, DT auto-derived) so rollout parity holds — the
  policy plays competently, verified by kickoff behavior.

## System architecture

Four layers, bottom-up:

1. **RocketSim** — the physics engine. 120Hz ticks, deterministic. Needs
   `collision_meshes/` (present in `build/`). The 5.0 run switched to **v3**
   (`RocketSimV3/`, commit `b1e1e9a`): a vendored Rust `rocketsim_ffi` staticlib
   (plain-scalar C ABI) behind a `compat/` facade that reproduces the v2 C++
   class surface (`Arena`, `Car`, `Step()`, `ballHitInfo`, bump/demo callbacks
   synthesized from FFI events). The swap is a physics-backend change *behind the
   facade* — it touched exactly one `RLGymCPP` file (`Framework.h`, an `#ifdef`
   include switch); no gym-consumer code changed.
2. **RLGymCPP** (`GigaLearnCPP/RLGymCPP/src/`) — gym layer. `EnvSet` steps 1024
   arenas on a shared thread pool with pluggable ObsBuilders / ActionParsers /
   Rewards / TerminalConditions / StateSetters. The split-step protocol still
   exists: `StepFirstHalf` runs `actionDelay` ticks with the OLD controls, then
   new actions are set and `StepSecondHalf` runs `tickSkip - actionDelay`. **5.0
   set actionDelay 0 and tickSkip 8** (was 3/4), so `StepFirstHalf` is a
   confirmed no-op (`Arena::Step(0)` returns immediately) — the chosen action
   now governs the whole 8-tick window from tick 1 (zero actuation latency),
   traded for a coarser **15 Hz** decision rate (was 30 Hz + 3-tick lag). The
   two-phase scaffold is kept generic for any future non-zero delay. The obs
   built after the step still carries `prevAction` = the action just set.
3. **GigaLearnCPP** (`GigaLearnCPP/src/`) — the learner. `Learner` orchestrates
   collection/processing/PPO; `PPOLearner` owns the models; aux modules:
   Reachability (InfoNCE self-model), **the Optimistic-Critic Ladder**
   (`GapState` — expectile sensor + quasimetric map + goal/concede banks +
   advantage drive + policy-head wire; see its own section below), the HEADROOM
   composition critic (twin V-dagger heads), League (QD MAP-Elites archive) and
   PolicyVersionManager (Elo skill tracker).
   **REMOVED 2026-07-25** (strip; restore point tag `pre-strip-20260725`): PSD
   (Basin-Racing), Proposer/DrillBank (the 9uz761ua regression machinery), and
   TransferLearn. All three had been disabled by pre-registered verdicts and had
   drifted incompatible with the live net. Their post-mortems remain in
   `research/reports/*.md`; the code is one `git log -- <path>` away. PSD's two
   useful signals were PROMOTED, not deleted — see `Util/Plasticity.h`.
4. **`src/ExampleMain.cpp`** — THE configuration. Everything (rewards, arenas,
   net sizes, all feature flags) is code here, heavily commented with the
   rationale and history of each value. Config changes = edit + rebuild +
   restart. Read this file first; it is the run's lab notebook.

### Networks (current run "5.0v3")

```
obs(230, RAW - standardizeObs=false)            AdvancedObsPadded(3), team-canonical frame
  └─ shared trunk: 2×[Linear512, LayerNorm, LeakyReLU]      "the trunk", h1/h2 taps
       ├─ policy head: 3×[512,LN,LReLU] → 90 logits          DefaultAction table, masked softmax
       │     INPUT widened 512→517: +5 Ladder "wire" columns [V_real,V_exp,gap_KD,V_metric,gap_PK]
       ├─ critic head:  3×[512,LN,LReLU] → 1                 GAE value
       ├─ reach_phi(trunk ⊕ onehot(action) → embed)          InfoNCE state-action embedding, φ/ψ hidden 256
       │     vs reach_psi_car   "can I reach the ball"  (car-local ball pos+vel /2300)
       │     vs reach_psi_ball  "can the ball reach the net" (canonical ball pos+vel)
       │     vs carStateHead    "can I reach this car pose" (canonical car pos+vel), DETACHED
       └─ gapSensor / Optimistic-Critic Ladder (see its own section) — expectile V_exp twin,
             quasimetric map (own optimizer), goal/concede banks, V_metric, gap_KD+gap_PK drive
goal_critic: independent net, raw obs → 1                    γ=0.9994 (~77s at 15Hz), std-matched
                                                             advantage blend; long-horizon credit
```
`TRAIN_GAMMA = gaeGamma = 0.9969` (~15s half-life at 15 Hz), re-derived for
tickSkip 8; `goalCritic.gamma = 0.9994` (~77s). If you change tickSkip, re-derive
both: `half-life_s = ln2 / (-ln γ) / (120/tickSkip)`.

- **carStateHead** (added 2026-07-14) is a THIRD reachability goal-head beyond
  psi_car/psi_ball, over canonical car pos+vel. It is trained fully **detached**
  (`carStateCouple=0`) after an incident where undetached co-training crashed
  Rating ~125 points — the same "probes never reshape the trunk" law the Ladder
  formalizes (Law 2/4).
- **Policy head is 517-wide, not 512** — the extra 5 columns are the Ladder's
  self-conditioning "wire". Zero-init at migration, so pre-wire checkpoints are
  behaviorally identical until trained; **one-way migration** — checkpoints saved
  after go 517-wide and will not load on a 512-wide model. Every model-build call
  site (train, league, render, boot) must independently carry the width flag or
  crash/silently-mismatch (two bugs paid for this: `dccba19`, `ba4f33a`).
- **AdvancedObsPadded is team-canonical AND fixed-width**: x,y are negated for
  ORANGE so both players see themselves attacking +y (every spatial
  label/analysis MUST canonicalize per-row or silently die — probes read
  R²≈−0.5 on x/y when missed, the x/y-vs-time asymmetry is the diagnostic). Width
  is padded to `MAX_PLAYERS_PER_TEAM=3` (230-dim: ball + prevAction + boost pads +
  self + up to 3 teammates + up to 3 opponents, zero-padded slots + presence
  flags), so one obs serves 1v1/2v2/3v3. Ball@0/self offsets stay
  AdvancedObs-compatible — NOT weight-compatible with 3.1's 109-dim checkpoints.
  Exact layout: `RLGymCPP/.../ObsBuilders/AdvancedObsPadded.h`.
- `DefaultAction`: 24 ground + 66 air = 90 actions (unchanged); masks by
  ground/air state, boost, flip availability.
- **Checkpoints** (`.lt`, C++ `torch::save`): load directly in Python with
  `torch.jit.load(path, map_location="cpu")`; params named `<seqIdx>.weight`;
  2-D=Linear, 1-D=LayerNorm, index gaps=activations. No forward method — rebuild
  eagerly (see `research/tools/load_checkpoint.py`).
- Reachability trains by HER: positives are FUTURE ACHIEVED states within a
  step-count window (car head vs ball head; verify current window lengths in
  code — note a step is now 15 Hz / ~0.067s under tickSkip 8, so any
  step→seconds conversion doubled vs the old 30 Hz numbers), InfoNCE on
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
  version manager, league) may be mutated. The shared thread pool belongs
  to the worker during learn; never use `fnParallelFor` from learn-prep (ad-hoc
  `std::thread`s are the pattern there, e.g. the steering landing sims).
- **Episodes are appended whole** to `combinedTraj` at finalize — one player's
  episode is row-contiguous. In-trainer analyses (steering derivation) rely on
  `row+k` = same player k steps later. (Offline datasets from
  `research/tools/collect_dataset.py` interleave 2 players per step instead —
  different indexing; this mismatch has caused bugs, check which layout you're in.)
- **GAE terminal semantics** (GAE.cpp): `NORMAL` = true terminal, no bootstrap;
  `TRUNCATED` = bootstrap from a stored next-state value. THE central hard-won
  rule: **whoever changes episode boundaries must let the critic learn the new
  return structure** (see steering post-mortems below).
- The stdout log prints only a curated summary block; the full `Report` (all
  `Steer/*`, `Reach/*`, `GoalCritic/*` panels) goes to wandb via the embedded
  Python metric sender.

### Reward stack (`BuildRewards()`, ExampleMain)

Whole-stack invariant: every component is exactly zero-sum or antisymmetric →
the stack sums to 0 across players, so league fitness is pure competitive
margin. Rules encoded in comments there: PBRS potentials MUST use `TRAIN_GAMMA`
(== `gaeGamma`, or telescoping breaks); never gate a potential; `ZeroSum(PBRS)`
is still exact PBRS; ShotReward is deliberately absent (its shot-attribution is
phantom-farmable — source-verified exploit in GameEventTracker).

Current live weights (HEAD): BallToGoalPotential **75** (already antisymmetric —
no ZeroSum wrapper, that would double it), TouchAccel **10**, Demo **37.5**,
BallProximityPotential **4** (team-closest, far-field linear term added in 5.0),
GuardedPickupBoost **6**, AerialTouch **120**, AirInterceptPotential **75**,
ConsecutiveAirTouch **30**, WallJumpToBall **30**, FlipReset **40**, AirReward
**0.45**, OpposedSave **25**, CarEnergyPotential **75** (tempo credit), TimeCost
**0.01** (not PBRS, not zero-sum-wrapped; ~3% of a goal per 30s episode),
TeamPressure **0.15**, KickoffRace **25**, Goal **150** (the objective, exactly
±150). All the ZeroSum-wrapped terms carry `TEAM_SPIRIT`.
(Weights re-read from ExampleMain 2026-07-25 — the previous list here had drifted:
it understated AirIntercept 40→75 and CarEnergy 15→75 and omitted five terms.)

- **AerialTouch 120 / AirIntercept 75 are SCAFFOLD weights** for 5.0's formative
  aerial window, flagged to anneal back down once aerial-touch share establishes.
  Don't read them as steady-state values. **Their stated anneal trigger does not
  exist**: six SCAFFOLD weights totalling ~260 (vs Goal 150) name `mechanic_census`,
  which measures none of the relevant mechanics and cannot load the live
  checkpoints — so "anneal later" currently means "never". Either implement it or
  delete the promise.
- **CarEnergyPotential 75** rode uncommitted through two cold starts and is
  recorded but UNMEASURED on this lineage; extrapolation from `REWARD_SHAPING.md`
  puts it near 60% of per-step credit density. Re-measure before trusting it.
- **`TEAM_SPIRIT` = 0.3 (PHASE A / 1v1-dominant) → 0.6 (PHASE B onward)**, set
  from the `PHASE_B` marker file in `main()`. It's an algebraic no-op in 1v1
  (teamMean == own); it only shapes 2v2/3v3 credit.
- **`NoTouchCondition(20)`** (raised 10→20s in 5.0) so dead-play trajectories
  survive long enough to recover and TimeCost's idle penalty accumulates into
  real signal. Viz/render drops NoTouch entirely (goal-only terminals) so the
  viewer shows unbounded real games.
- **RND / EMERGENCE novelty** (`rndOptimism`, w=**0.1** from step 0 in 5.0 —
  vs 4.0's end-of-life 0.3) is applied as an advantage-space injection, NOT a
  reward term: mean-zero, std-matched, rating-latch-covered, self-annealing as
  the RND predictor learns. It rides alongside the Ladder drive.

### Outer loops

- **Skill tracker** (`PolicyVersionManager`): Elo (`Rating/1v1`) from eval
  matches vs `checkpoints_5.0v3/policy_versions/` (a ring of versions ~25M steps
  apart). **Pool myopia caveat**: the pool cannot measure improvement against
  styles older than its window, and a style shift can read as an Elo dip while
  head-to-head vs recent selves improves (measured: nontransitivity is real
  here). Mature Rating noise band is ±30–50 — but **this run is young and
  steeply climbing, so Rating legitimately swings ±80 around the trend**. The
  two auto-kill latches were loosened for that (110→200, 75→150) after 3 false
  trips on pure volatility — and then **the latch was REMOVED entirely
  (2026-07-25, user-directed)** after it fired a fourth time, on a spike-and-settle
  where Rating was still +213 ABOVE its own EMA, taking six unrelated live
  mechanisms dark. What remains is measurement only: `RatingWatch/Drawdown From
  EMA` and `/From Peak`. **There is no automatic update-damage guard any more** —
  the boot sanity probe and the impossible-arena certificate are now the only
  automatic safety checks, and neither sees gradual update damage.
- **QD League**: MAP-Elites archive over behavior descriptors (quantile-adaptive
  bins), PFSP-sampled opponents on `descendOpponentFrac` of iterations (0.35 —
  raised from 0.25 after measuring exploitability by archived styles). Members
  are stored as flat param vectors; **the 512→517 policy-head change made stored
  vectors stale-shaped**, crashing crossover arithmetic in `EvolveStep`. Fixed
  (`dccba19`) by migrating (zero-padding the first Linear's input columns) at
  `FromJSON` load time, not just `LoadInto` — any archive saved pre-517
  self-heals on next checkpoint load. Watch this pattern on any future net-width
  change.
- **Plasticity telemetry** (`Util/Plasticity.h`, promoted out of PSD when it was
  retired): `Plasticity/Trunk EffRank`, `/Policy EffRank`, `/Policy Dead Units`.
  Weights-only, every iteration, no actuation. Kept because the residual
  architecture (`45a59d5`) was justified BY effective-rank decay — deleting PSD
  wholesale would have removed the ability to check that rationale.
- Eval paths (skill tracker, league, render) are all UNSTEERED and feed
  ZERO Ladder-wire values — verified; Rating always measures the raw policy,
  symmetric across the pool.

## The Optimistic-Critic Ladder — CURRENT optimism mechanism (live)

This is the mechanism now actuating the "explore your frontier" program; it
**superseded activation steering** (next section, parked). Live on 5.0v3 since
~18.88B steps (built across `fc34dd3` → `36f1e01` → `15eabda`). Canonical record:
`research/reports/LADDER.md`. Implementation: `GapState` in Learner.cpp; wire in
PPOLearner; config `GapSensorConfig` / `cfg.gapSensor.*` in ExampleMain.cpp.

**Why it exists**: steering nudged *behavior* toward feasible plays via an
activation push. The Ladder instead gives the network a measurable, *trainable*
self-model of its own optimism deficit, turns closing that deficit into an
advantage-shaping signal, AND lets the policy directly perceive the deficit as
input — a representation+credit lever, not just a behavioral one. Same
knowing-doing-gap target, moved from activations to advantages and observations.

Two stacked "gaps", combined into one closing drive:

1. **`gap_KD` (rung 2, the sensor)** — `gapSensor->exp` is an expectile-τ=0.8
   twin of the critic, trained on the SAME extrinsic GAE targets but reading the
   trunk through `.detach()` (pure probe — can't reshape what it measures).
   `gap_KD = relu(V_exp − V_real)`: "the optimistic estimate exceeds the honest
   one." This alone was Stage 1/2a.
2. **`gap_PK` (rung 3, the geometry)** — a learned **quasimetric map**:
   `E: obs→256→256→64`, `f: 64→128→32`, distance `d(x,y)=Σ relu(f(E(x))−f(E(y)))`
   (asymmetric, triangle-respecting), trained QRL-style each iteration (`L_local`
   on consecutive same-agent pairs, `L_spread` on random pairs, dual-ascent λ)
   with its **own Adam + own clip group** (Law 1) and **never touching the
   trunk** (Law 2). **Goal/concede banks** (256-per-side ring buffers, live;
   spec default 1024) hold raw obs from the final ~1s (~15 rows) before each
   scored/conceded goal — landmark states. **`V_metric` = a·γ^d_goal +
   a2·γ^d_concede + b**, fit each iteration by fp64 OLS against the critic's own
   extrinsic targets (EMA'd, clamped), gated until banks≥32 AND mapUpdates≥50.
   `gap_PK = relu(V_metric − V_exp)`: what the geometry claims *above* the sensor.

**The drive**: `Φ = −(gap_KD + gap_PK)`, injected as an **undiscounted closing
delta** into raw advantages — it pays for *shrinking* the gap, not for having one
(the loitering fix). Masked at terminals AND truncations AND impossible rows;
centered OVER UNMASKED ROWS ONLY then re-masked (Law 8b); std-matched at
`driveBeta=0.05` (≈5% of extrinsic-advantage scale — the knob analogous to
steering's old α, but on advantages); ±3σ clamped; covered by the same rating
latch. There is no advantage normalization in this codebase, so the inject-before-
handoff ordering is exact.

**The 5-wire (why the policy head is 517)**: the five scalars
`[V_real, V_exp, gap_KD, V_metric, gap_PK]`, tanh-squashed, appended as extra
input columns to the **policy head only** (trunk/critic untouched). The policy
literally sees its own critic-vs-optimist disagreement when choosing actions.
Zero-init migration; the Muon policy optimizer is reset once at migration
(logged transient). **Collection** fuses the wire from the pipelined snapshot
generation (exp/mapE/mapF cloned, bank embeddings + calibration frozen per
generation at the barrier); **Learn** re-derives it per-minibatch from CURRENT
heads but the COLLECTION generation's bank embeddings/calibration (spec 2.5), and
**hard-refuses to run if that hand-off wasn't armed** (no silent zero-fallback).
Eval/opponent/render/boot feed explicit zeros.

**The impossible-control family (standing falsification test)**:
`ImpossibleInterceptState` on the last 8 arenas of the 1v1 block spawns
certified-unreachable ballistic intercepts (required speed > 1.6× the hard cap).
Rows are masked from injection; a cumulative **`Ladder/Imp Touches` counter must
stay 0 for the life of the run** (one touch voids the certificate), and
`Ladder/Imp GapPK Spawn` must fall BELOW `Ladder/Fear GapPK` as the map's doom
geometry converges — if it never crosses, the system is manufacturing
self-serving optimism.

**Design laws (institutionalized by prior incidents, enforced in code)**: (1)
separate gradient economies — the map/sensor get their own optimizer+clip so
their losses can't crush the policy gradient; (2) map never touches the trunk;
(4) probes are detached; (6) wire and drive ship together; (7) off ==
bit-identical (`mapEnabled=false` degrades to gap_KD-only, `driveBeta=0` kills
drive+wire); (8b) center over unmasked rows only. **Rollback**: the 517 head
reverts only via backup — `checkpoints_5.0v3_branch_backup/18875158586` (pre-wire,
512-head) + golden archive.

## The steering system ("optimism surgery") — PARKED (superseded by the Ladder)

**Status**: activation steering is numerically inert on HEAD — `steering.alpha =
0`, `opponentStyleChance = 0` (set per the Stage-2 protocol, `36f1e01`).
`steering.enabled` stays true only so fear-drill/census/miner telemetry keep
running; `SetSteering` executes but adds no `α·σ·v` anywhere. The Ladder above
replaced it (actuating both would double-dose the same axis). Reverting is a
one-line flip back to α=0.5. **The mechanism and its hard-won lessons below are
retained as history** — the critic-aliasing post-mortems in particular still
govern any future episode-boundary change.

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
   - Ritual: branch-point checkpoint backup (`build/checkpoints_5.0v3_branch_backup/`)
     before every enablement; quarantine (move, never delete) on revert
     (`build/<ckptdir>_quarantine_<ts>/`; the 3.1-era quarantine held the two
     collapse periods).

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
the boundary coming). Full post-mortems: `research/reports/STEERED_PRACTICE.md`.

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
4. `research/tools/derive_steering.py` still uses the v1 landing-attendance
   metric — offline analysis only now; the in-trainer v2 possession derivation
   is authoritative.
5. Metric aging is a live risk pattern: the v1 "landing attendance" definition
   was outgrown by the improving bot (it converts via early pressure/bounce
   play). Possession outcomes were chosen because they cannot be satisfied by
   style ("winning the ball first" is good at every level) — but audit any
   behavioral metric against head-to-head results periodically.

## research/ — the measurement program

Everything offline lives under `research/` (it was `analysis/probes/` until
2026-07-25; the old name was a fossil of the Phase 0 linear-probe study it grew
out of). Four parts, each with its own README:

- **`research/reports/`** — the experimental record, one `.md` per study, most
  opening with a pre-registration block dated before the run. Its `README.md` is
  the status index (CANONICAL / RESULT / HISTORICAL / SUPERSEDED) and is the
  right entry point. Current-run canonical: `PULSAR5.md` (cold-start design),
  `LADDER.md` (Ladder deploy record), `FRONTIER.md`, `REWARD_SHAPING.md`,
  `DEAD_CODE_AUDIT.md`, `LEAGUE_ANCHORS.md`. `STEERED_PRACTICE.md` is the
  canonical history of the parked steering system and its episode-boundary rule
  still binds. **`reports/archive/`** holds nine superseded 4.0-lineage steering
  studies (including `STEERING_ROADMAP.md`) — provenance only, never a
  justification for a new change.
- **`research/tools/`** — the Python toolkit. Pipeline: `load_checkpoint.py`
  (jit-load + eager rebuild + `PulsarPolicy` with h1/h2/phi taps) →
  `collect_dataset.py` (self-play at exact obs/action/step parity) →
  `label_landing.py` (ball-only touchdown sims) → `train_probes.py` (ridge
  probes, episode-grouped CV, controls). Plus `anchor_battery.py` /
  `match_play_eval.py` (real match-play Elo vs fixed anchors — use these, not
  `Rating/1v1`), `compare_checkpoints.py`, `knowing_doing.py`, `kd_curve.py`,
  `calibrate_rho.py`, `steer_test.py` / `steer_team.py` (rollout harnesses that
  the steering-era drivers still import).
- **`research/results/`** — JSON output keyed by checkpoint timestep.
- **`research/data/`, `research/.venv/`** — untracked, large, regenerable.

**Two live traps** (both written up in `reports/DEAD_CODE_AUDIT.md` §5/§15,
neither fixed): `load_checkpoint.py`'s default checkpoint root does not list
`checkpoints_resid`, so it silently falls through to the frozen
`checkpoints_5.0v3` — pass `PULSAR_CKPT_ROOT` explicitly; and it asserts a
512-wide 2-layer non-residual trunk, which the residual run does not match. A
naive shape fix would replay the h2-truncation bug class.

Offline behavioral numbers dated before 2026-07-19 used pre-activation `h2`
(`reports/H2_TRUNCATION.md`); affected reports carry a banner. In-trainer
telemetry is unaffected — that bug was in the toolkit, not the C++.

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
improvement resumed. The knowing-doing gap became knowledge-COUPLED (graded by
read quality). The **4.0 lineage** then extended the program to team play
(230-dim padded obs, 2v2/3v3 PHASE B, energy/tempo PBRS, RND novelty) and reached
a plastic-but-matured equilibrium (Elo ~1793, 0% dormant units) whose opponent
pool priced out learning-phase play — it had passed its formative high-entropy
window before aerial/mechanic pressure existed. So **5.0 was a user-directed cold
start** (`PULSAR5.md`): the same recipe placed into the formative window, on the
RocketSim **v3** engine, at **tickSkip 8 / actionDelay 0** (15 Hz — chosen
because a wavedash plateau at 11% convicted learnability, not a control ceiling,
as the binding constraint), with scaffolded aerial rewards and RND from birth.
On that run the optimism program itself moved off activations: the
**Optimistic-Critic Ladder** (expectile sensor → quasimetric map → combined
gap-closing advantage drive + policy-head self-conditioning wire) went live at
18.88B steps and is the current lever. Representation-side pressure (an aux
landing-prediction head — "Phase 1" of the workspace program) remains the next
measurement-backed option when the current mechanisms saturate.
