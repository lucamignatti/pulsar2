# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Pulsar: a Rocket League self-play bot trained by PPO. Core play is 1v1; the
current run ("5.0v3") also opens 2v2/3v3 team-play phases (PHASE B). The trainer
is C++ (libtorch + a vendored RocketSim physics sim); on top of it sits a
research program testing whether the bot can be made to *explore its capability
frontier* ("optimism") — inspired by Anthropic's global-workspace paper — using
linear probes, a learned self-model of reachability, and — currently — a
**composition critic** that credits conducts the bot has never performed as
wholes, from pieces it has performed separately (see its section below; the
canonical spec is `research/reports/COMPOSITION_CRITIC.md`). This is
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
> (15 Hz), **230-dim padded obs**, a policy head at plain trunk width (the Ladder
> wire was removed 2026-07-25), `checkpoints_resid`.
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
  never runs a learn pass, but it MUST build the SAME architecture as the trainer
  or it loads mismatched shapes — the bug fixed in `ba4f33a`. (The wire that
  caused that specific 512-vs-517 incident is gone; the parity requirement is not.)
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
   (`GapState` — the `V_exp` expectile twin, measurement only), the **composition
   critic** (twin V-dagger heads + the seek term; see its own section below), League (QD MAP-Elites archive) and
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
       ├─ critic head:  3×[512,LN,LReLU] → 1                 GAE value
       ├─ reach_phi(trunk ⊕ onehot(action) → embed)          InfoNCE state-action embedding, φ/ψ hidden 256
       │     vs reach_psi_car   "can I reach the ball"  (car-local ball pos+vel /2300)
       │     vs reach_psi_ball  "can the ball reach the net" (canonical ball pos+vel)
       │     vs carStateHead    "can I reach this car pose" (canonical car pos+vel), DETACHED
       └─ gapSensor / Optimistic-Critic Ladder (see its own section) — expectile V_exp twin,
             V_exp expectile twin (measurement); V-dagger twins + Phi=+H seek (the actuator)
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
  are stored as flat param vectors, so **any net-width change makes stored vectors
  stale-shaped** and crashes crossover arithmetic in `EvolveStep`. That happened
  once (`dccba19`, the 512→517 wire migration). The migration path was removed with
  the wire on 2026-07-25 — if you change net width again, the archive needs one.
- **Plasticity telemetry** (`Util/Plasticity.h`, promoted out of PSD when it was
  retired): `Plasticity/Trunk EffRank`, `/Policy EffRank`, `/Policy Dead Units`.
  Weights-only, every iteration, no actuation. Kept because the residual
  architecture (`45a59d5`) was justified BY effective-rank decay — deleting PSD
  wholesale would have removed the ability to check that rationale.
- Eval paths (skill tracker, league, render) never receive any intervention —
  Rating always measures the raw policy, symmetric across the pool.

## The composition critic — CURRENT optimism mechanism (live)

Canonical spec: **`research/reports/COMPOSITION_CRITIC.md`** (the paper). The trainer was
conformed to it on 2026-07-25; read the paper, not this summary, when the details matter.

The problem it addresses is the **acquisition wall**, not sample efficiency: a conduct whose
first success has near-zero probability under current behaviour produces no reward, no gradient,
and is never learned. A whiffed aerial also hands the opponent a counterattack that enters the
return, so below a break-even success rate PPO actively teaches avoidance — which starves the
data that would raise the success rate.

The escape is optimism about **compositions**. A conduct never performed as a whole may have
every *piece* performed somewhere: the bot has jumped, has boosted while tilted, has touched low
balls — in different lives. Optimism over such chains is falsifiable piecewise.

Three critics, one actuator:

1. **`V_real`** — the ordinary critic. "What I reliably do."
2. **`V_exp`** — a return-level expectile twin (τ=0.8) trained on the SAME extrinsic GAE targets
   through a **detached** trunk read. "What I sometimes do." **Measurement only** — publishes
   `Gap/*`; nothing injects from it.
3. **`V†₁, V†₂`** — the composition critic: twin heads on the shared trunk, expectile τ=0.75, on
   one-step TD targets over **executed transitions only**, with the target taking
   `min(V†₁, V†₂)`. That min is the anti-ratchet: online asymmetric TD otherwise self-amplifies
   through its own bootstrap (the paper's §4.4 measured H inflating 0.3 → 11.8 with no conversion
   behind it). Gradients DO flow into the trunk — deliberate co-adaptation from step zero.

**Actuation, and there is only one:** `Φ = +H` with `H = relu(min(V†₁,V†₂) − V_real)`, injected
as `γ(1−d)H(s') − H(s)`, centred, σ-matched at `vdagSeekBeta = 0.15`, clamped ±3σ. `d` is nonzero
at terminal **and** truncation. Potential-based, so optimal policies are preserved.

**The sign is load-bearing.** The *closure* form `Φ = −H` — which the retired Ladder used — is
catastrophic on a strong peaked field: the cheapest way to reduce the potential along a
trajectory is to leave the peak, so the policy is paid to walk away from its own frontier
(measured: ball interaction collapsed ~10×). Seek attracts; closure repels. Closure remains fine
for weak diffuse gaps.

**Deployment note (`Learner.cpp`)**: production GAE normalizes rewards, so the TD target is
reconstructed in the critic's units from GAE outputs alone —
`r_scaled = A_i − γλ(1−d)A_{i+1} − γ(1−d)V_{i+1} + V_i`.

**Watch `Headroom/Vdag Update Magnitude`.** It read exactly **0 for the life of the run** until
2026-07-25: `SetLearningRates` never named `vdag1`/`vdag2`, and `Model`'s ctor builds every
optimizer at `lr=0` (an exact no-op under Muon). 42% of the net sat at random init while its
seek term still injected and its loss still reshaped the trunk. If that panel returns to 0, the
mechanism is inert and the injection is a random projection.

**REMOVED 2026-07-25** in the conformance pass (restore tag `pre-strip-20260725`): the
quasimetric map, goal/concede banks, `V_metric`, `gap_PK`, the closure drive
`Φ = −(gap_KD + gap_PK)`, the **5-column policy-head wire** (the paper is explicit that the
policy never consumes `H`, so the head is plain trunk width again), RND novelty, and the
impossible-control falsification family. Deploy records for the removed machinery live in
`docs/LADDER.md` and `docs/EMERGENCE.md` — provenance only, not a description of the system.

## Steering ("optimism surgery") — actuation REMOVED, derivation retained

Activation steering was superseded by the optimism work and had been numerically inert (α=0)
long before it was removed on 2026-07-25. Gone: `fnApplySteering`, `SetSteering`, the causal
auto-gate, the matched trunk-mean direction/sigma derivation, META in full, the rho-band gate,
and 27 dead config fields.

**`config.steering.enabled` is still true and no longer means steering.** What it now gates:
airborne-reading collection, ball-only landing sims, POSSESSION-OUTCOME labelling, the frontier
reset pool that drives `FrontierDrillState` on ~30% of arena resets, the in-trainer census, and
the emergence miner. The name is misleading and worth changing.

**Two lessons from that program still bind, and are cited from the C++ source:**

- **The clipping ratchet.** For actions a push makes much likelier than the base policy, the
  learn-pass ratio falls outside the clip window, and PPO's pessimistic min keeps the gradient
  for POSITIVE advantages while zeroing it for NEGATIVE ones — successes reinforce, punished
  failures are discarded. At α=1σ this compounded lucky overcommits into an Elo bleed while the
  viewer looked better. Any future activation-space intervention inherits this.
- **Critic aliasing at episode boundaries** (`resolutionTermination`, still `false`, two Elo
  collapses): a shared critic cannot price truncated and full episodes of the same observation.
  Whoever changes episode boundaries must let the critic learn the new return structure. Full
  post-mortem: `research/reports/STEERED_PRACTICE.md`.


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
On that run the optimism program moved off activations: the **Optimistic-Critic
Ladder** (expectile sensor → quasimetric map → gap-closing drive + a
policy-head wire) went live at 18.88B steps. It in turn was **superseded on
2026-07-25** by the **composition critic**, and the Ladder's map/banks/`gap_PK`/
wire were removed with it — the current lever is `Φ = +H` from the V-dagger twins,
specified in `research/reports/COMPOSITION_CRITIC.md`. Two audits that same day
found the twins had been frozen at `lr=0` since introduction, and stripped ~6.5k
lines of parked machinery (PSD, proposer/drills, TransferLearn, RND, steering
actuation, the rating latch). Representation-side pressure (an aux
landing-prediction head — "Phase 1" of the workspace program) remains the next
measurement-backed option when the current mechanisms saturate.
