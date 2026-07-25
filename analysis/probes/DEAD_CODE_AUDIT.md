# DEAD CODE / INERT FEATURE AUDIT — 2026-07-25

> **OUTCOME (acted on the same day).** Net: **~4,270 deletions**. Restore point: tag
> `pre-strip-20260725`.
>
> | Item | Outcome |
> |---|---|
> | PSD / Basin-Racing (~1,880 LOC) | **STRIPPED**; `EffectiveRank`/`DeadUnitFraction` promoted to `Util/Plasticity.h` as unconditional telemetry; `LeagueConfig` extracted to its own header |
> | Proposer / drill bank (~1,220 LOC) | **STRIPPED**; carStateHead disjunct, carStateHerGoals/achievedCarState and pipelineOn semantics all verified preserved |
> | TransferLearn chain (~280 LOC) | **STRIPPED**; `StartQuitKeyThread` preserved (it is live) |
> | Dead types + optimizers (~480 LOC) | **STRIPPED** (PlayerReward, DrillSetter, Quat, DefaultObs pair, MagSGD, 3 unused optimizer arms) |
> | Dead `Report` API (~45 LOC) | **STRIPPED** |
> | HEADROOM vdag frozen at random init | **FIXED** — LR wired, twins added to the gradient-clip list, `Headroom/*` given a console block |
> | Rating latch | **REMOVED ENTIRELY** (user-directed) — measurement kept as `RatingWatch/*`; no automatic update-damage guard exists any more |
> | Nexto counters / league reward stack | **FIXED** (both one-liners) |
> | Test suite | **REPAIRED** — it failed 6/30 on a clean tree; now 31/31 with new `soloFrac` coverage |
> | Steering actuation (~340 LOC) | **NOT STRIPPED** — entangled with the live FrontierPool; deferred as its own change |
> | `userInfo`, 109-dim branches | **NOT STRIPPED** — poor risk/reward |
> | Offline toolkit (38 scripts) | **NOT FIXED** — still cannot load this run's checkpoints |
>
> **Two claims in this audit were WRONG**, caught during execution: `Report::SingleToString` is
> live (`Display` calls it unqualified, so a `.SingleToString(` grep misses it), and
> `digitCommas` *is* passed `true` — the dead part was the unread parameter, not the function.
>
> **What remains enabled, and the decisions still open: `ENABLED_INVENTORY.md`.**


Scope: every feature surface in the repo, resolved against the **live config**
(`src/ExampleMain.cpp`, not header defaults) and the **live process**
(PID 364530 → `build/GigaLearnBot`, launched 09:57:57, run `checkpoints_resid`).
Read-only audit; nothing was edited, built, or run against the GPU.

**Audited**: 16,676 LOC first-party C++ (`GigaLearnCPP/src`, `src`, `tests`) +
4,242 LOC gym layer (`RLGymCPP/src`) + 9,787 LOC Python (`analysis/probes`) +
2,486 LOC tooling + 5,115 lines of `.md` research record. Vendored trees
(RocketSim v2 68k, RocketSimV3 17k, cpp-interface 4.4k) audited at tree/target
level only.

## Bucket counts

| Status | Count | Meaning |
|---|---:|---|
| LIVE | 41 | Enabled, materially affects training. Protected. |
| INERT | 27 | **Flag on, effect zero.** Burns compute/complexity for nothing. |
| DISABLED_FLAG | 34 | Cleanly gated off, code intact, one-line revive. |
| DEAD_CODE | 22 | No call site / never instantiated. |
| DRIFTED | 26 | Still running; semantics no longer match this run. |
| SCAFFOLD | 7 | Live but explicitly temporary, with a stated exit condition. |

## Headline findings

1. **HEADROOM V-dagger twins are frozen at random init and still driving 15% of
   advantage std.** `Model` ctor builds every optimizer at `lr=0`
   (`Util/Models.cpp:63`); `PPOLearner::SetLearningRates` (`PPOLearner.cpp:944-957`)
   sets policy/critic/shared_head/goal_critic and **not** vdag1/vdag2. No
   `SetOptimLR` call reaches them anywhere (12-hit grep). Under Muon, `lr=0` is an
   exact no-op. So 16.1M params (42% of the net) never move, `H = relu(min(V1,V2) − V_real)`
   is a fixed random projection of the trunk, and its seek term is still injected
   at `vdagSeekBeta=0.15` (`Learner.cpp:4759-4760`) **unclipped into the shared
   trunk** (`PPOLearner.cpp:643-657`, not detached, absent from the clip list at
   `:797-806`). NewtonSchulz5 runs on every vdag matrix every minibatch and is then
   multiplied by zero. This is the single most consequential finding in the audit.
2. **~30% of the actor's advantage std is now non-extrinsic, and no knob owns the
   total.** Four std-matched injectors chain: HEADROOM 0.15 (vs *pre*-injection
   std), goal-critic 0.25, RND 0.10, Ladder drive 0.05 — the last three each
   re-read `tAdvantages.std()` *after* the earlier ones landed
   (`Learner.cpp:4755, 4819, 4902, 5286`). Composite ≈ 0.32 σ_ext. No panel
   reports it. `GAE/Avg Advantage` (`Learner.cpp:4778`) is computed *after* the
   HEADROOM injection, so the panel named for raw GAE is a hybrid. Masking is
   inconsistent: only the Ladder drive masks terminals/truncations/impossible rows.
3. **The one safety latch that can see update-damage is reachable only through
   `steering.enabled`.** `fnRatingGuard` early-returns on `!steerOn`
   (`Learner.cpp:3241`). `steerRatingTripped` gates six *live* mechanisms — frontier
   drills (`:2398`), Nexto serve (`:3428`), league anchors (`:3446`), HEADROOM seek
   (`:4759`), RND injection (`:4900`), Ladder drive (`:5227`). The documented
   "revert steering" move (`steering.enabled = false`) would silently disarm all
   six with no log line, and its trip message still says only "steering, opponent
   styles and frontier drills latched OFF".
4. **The Ladder's standing falsification test is unscoreable, and currently reads
   bad.** `Ladder/Imp GapPK Spawn` runs 0.83–1.19 against a buffer-wide
   `Ladder/GapPK Mean` of 0.09–0.35 — the map assigns 4–8× more claimed headroom at
   *certified-unreachable* spawns. Its acceptance comparator `Ladder/Fear GapPK`
   (`Learner.cpp:5218`) requires `fearPanelObs`, frozen **only** under `md == 1`
   (2v2) at `Learner.cpp:2482`. This run is PHASE A. `grep -c Fear` over the live
   log = 0. Numerator publishes, denominator never does.
5. **The entire offline measurement toolkit cannot load a single checkpoint of the
   live run, and silently analyzes a frozen lineage instead.**
   `load_checkpoint.py:57-68` hardcodes 512-wide/2-layer/non-residual shapes and
   raises on mismatch; `_default_root()` (`:43-51`) does not list `checkpoints_resid`
   and falls through to `build/checkpoints_5.0v3`, which still exists. 38 scripts
   route through it, including `match_play_eval.py`/`anchor_battery.py` — the only
   pool-inflation-proof yardstick. Worse: `residualSpans` is never serialized
   (`Models.h:105`, applied at forward time only), so a naive shape fix would
   silently rebuild the residual net as a skip-free MLP — an exact replay of the
   2026-07-19 h2-truncation class of bug.

---

# INERT: enabled but doing nothing

The owner's priority category. Ordered by cost, not by LOC.

| # | Feature | Anchor | Why it's zero | Cost while inert |
|---|---|---|---|---|
| I1 | Live commitment-direction + sigma derivation | `Learner.cpp:2188-2280` | Output multiplied by `steering.alpha = 0` (`ExampleMain.cpp:1005`) | **3 GPU trunk forwards/iter** + matched-bin selection, every iteration |
| I2 | Steered/control possession causal gate | `Learner.cpp:2153-2185` | Selects between `alpha` and 0 — both 0 | Noise generator; **logs "TRIPPED … alpha → 0"** on a zero treatment |
| I3 | Rho-band gate snapshot cost | `Learner.cpp:1817-1822` | Gate body never runs (`anyActive==false`, `PPOLearner.cpp:317`) | 4 extra `MakeClone()` + **4 full param copies per iteration in the barrier zone** |
| I4 | Steering actuation (α·σ·v) | `PPOLearner.cpp:309-428` | `anyActive` short-circuit at `:322` | ~0 GPU (clean short-circuit); complexity only |
| I5 | Reachability rho diagnostic read block | `Learner.cpp:4099-4297` | Gate off (`gateEnabled=false`, `ExampleMain.cpp:633`); output is telemetry | **3 full-buffer model passes** every 16 iters; opponent-obs rows collected *every* iter, read 1-in-16 |
| I6 | `reach_psi_carstate` head | `Reachability.cpp:37-41` | Trained detached, **zero rho consumers** (all 3 readers meta/proposer-gated) | fwd+bwd+Adam step per minibatch, per-minibatch `.item()` D2H sync, 4.2MB/checkpoint |
| I7 | Dz z-scoring pass | `Learner.cpp:1938-1953` | All consumers require `md > 0`; PHASE A is 1v1-only | 2 full-buffer z-normalizations per iteration, discarded |
| I8 | League match env computes the 16-term reward stack | `LeagueArchive.cpp:23-30` | Fitness is `goalScored` only; `rewards` never cleared or read | ~57.6k player-steps of discarded reward eval per evolve, **in the barrier zone** |
| I9 | Reachability reward gate (gated buckets) | `Learner.cpp:4286-4296` | Doubly dead: flag off **and** `gatedPos` identically 0 (no reward passes `gated=true`) | per-row float accumulate ×1024 arenas ×every step |
| I10 | Nexto CUDA device probe | `NextoOpponent.cpp:51-91` | Probe deterministically throws; always serves CPU | Full jit::load onto CUDA + dummy fwd at boot, on a 16GB display-attached card |
| I11 | Fear-panel comparator | `Learner.cpp:5199-5222` | `fearPanelObs` only freezes under `md==1` | ~0 compute; **kills the Ladder acceptance test** |
| I12 | `steer_vec`/`steer_sigma` checkpoint payload | `Learner.cpp:461-469` | Save-only by design; no reader exists any more | ~20KB/save, forever |
| I13 | Three phantom stdout rows | `Learner.cpp:5542,5543,5566` | Zero producers repo-wide (exhaustive 75-key audit) | 3 hash misses/iter; **misleads the operator** |
| I14 | `Steer/RhoGate In-Band Frac` panel | `Learner.cpp:5414` | `lastRhoGateFrac` written only inside the dead gate; pinned at initializer 0 | Reads as "gate rejecting 100%", not "gate not running" |
| I15 | GameEventTracker: 7 of 9 `PlayerEventState` flags | `EnvSet.cpp:4-25` | Only `save` and `demo` have readers | per-event linear scan; **detector itself is load-bearing** |
| I16 | `driveWarmupIters` / `mapWarmupIters` | `LearnerConfig.h:296,316` | `GapState::Build` force-sets `updates=1000` on load (`Learner.cpp:125`); `mapUpdates` persisted | Gate can never re-close after the first checkpoint |
| I17 | TEAM_SPIRIT 0.3→0.6 schedule | `ExampleMain.cpp:1063` | Exact algebraic identity at n=1 (`ZeroSumReward.cpp:24-27`) | Free; but 13 reward lines *look* team-tuned |
| I18 | Team-mode branches in setters/rewards | `AirDrillState.h:150-220`, `CommonRewards.h:109-118,214-240` | PHASE A ⇒ 1 car/team | Free; dormant-by-phase |
| I19 | `EnvSet::ResetArena` control clear | `EnvSet.cpp:295-301` | Hazard requires `actionDelay > 0`; live value 0 | ~2 stores/reset; **keep** (guard for future delay) |
| I20 | `ObsBuilder::Reset` virtual | `ObsBuilder.h:10` | No subclass overrides it | 1 vtable dispatch/arena/reset |
| I21 | Ladder wire zero-pad migration | `Models.cpp:208-239` | Cold start born wire-wide; no short vector can exist | Branch test per Load; **reachable from RLBot path** |
| I22 | Render-mode gapSensor | `ExampleMain.cpp:1241-1242` | Wire fed exact zeros (`PPOLearner.cpp:202-208`) | **Deliberate** — the `ba4f33a` architecture-parity fix |
| I23 | Nexto per-episode prevAction reset | `NextoOpponent.cpp:156-164` | `terminals` already zeroed by `EnvSet::Reset` before `Act` | **Latent bug**: stale action leaks 1 frame per reset |
| I24 | `steerTeamModes` | `ExampleMain.cpp:1017` | md>0 blocks have `count==0` | Only observable effect is a **false boot banner** ("team modes steered") |
| I25 | `frontierFearMining` ranked path | `ExampleMain.cpp:1101` | `ranked` requires `md > 0` | Pool falls back to uniform stride; `Steer/Frontier Dz` never emits |
| I26 | Proposer/* console panel block | `Learner.cpp:5552-5563` | 10 keys, all producers proposer-gated | Two adjacent blank lines every iteration |
| I27 | Proposer delta-net arch config | `ExampleMain.cpp:721,733` | Configures a net never constructed | Comment reasons about "drill quality" that cannot exist |

**Aggregate**: the recurring per-iteration inert cost is I1 + I3 + I5 + I7 + I8 —
roughly 3 trunk forwards, 4 model param copies in the barrier zone, 2 full-buffer
z-scorings, 3 full-buffer model passes 1-in-16, and ~58k discarded reward
evaluations per evolve. Individually small; collectively it is the reason
consumption sits near collection time.

---

# Subsystem breakdown

## 1. Steering (parked mechanism, live satellites)

The most misleading subsystem in the codebase: `steering.enabled = true`
(`ExampleMain.cpp:985`) with `steering.alpha = 0.0f` (`:1005`) means the flag no
longer gates *steering* — it gates the **frontier drill pipeline**, the **rating
latch**, the **census/miner telemetry**, and the **arena role tagging the Ladder
depends on**.

| Feature | Status | Gating (live) | Strip? | Notes |
|---|---|---|---|---|
| Rating drawdown+peak latch | LIVE | `steerOn && ratingGuardEnabled` = true | **NO** | Gates 6 live mechanisms; must be lifted out of `steering.*` |
| Ball-landing-sim reading pipeline | LIVE | `if (steerOn)` `:5426` | **NO** | Feeds `FrontierPool` → 60% of practice-arena resets |
| `practiceArenaFrac`+`frontierPool`+`FrontierDrillState` | LIVE | `ExampleMain.cpp:1009,1089-1092` | **NO** | ~18% of all resets are mined frontier states |
| Guard band 200/150 | SCAFFOLD | `ExampleMain.cpp:992,995` | keep | Exit condition written in-file: "tighten toward 110/75 at maturity" |
| Direction/sigma derivation | INERT | α=0 | after decision | I1 |
| Possession causal gate | INERT | α=0 | after decision | I2 — writes scary TRIPPED lines |
| Rho-band gate + 4 reach snapshots | INERT | `anyActive==false` | after decision | I3 + I14 |
| Steering actuation | INERT | α=0 | keep | The documented 1-line revert |
| Per-mode/FEAR_MINE/fear panel | INERT | `md > 0` | keep | I7, I11, I25 |
| `steerTeamModes` | INERT | 0 team arenas | keep | I24 |
| `steer_vec`/`steer_sigma` payload | INERT | write-only | after decision | I12 |
| Steered/control arena split | DRIFTED | `practiceArenaFrac=0.30` | **NO** | See below |
| Steering delta applied **before** Ladder wire | INERT | α=0 | investigate | See below |
| `opponentStyleChance` | DISABLED_FLAG | `= 0.0f` `:1053` | keep | Cleanest parked feature — zero cost |
| `steering.meta` (~470 LOC) | DISABLED_FLAG | `= false` `:1045` | after decision | Largest dead block in the subsystem |
| `resolutionTermination` / `AttemptResolutionCondition` | DISABLED_FLAG | `= false` `:1010` | keep | Two Elo collapses; keep the lesson visible |
| `frontierPotential` | DISABLED_FLAG | `getenv(GGL_FRONTIER_POTENTIAL)` unset | keep | Phase-0 telemetry only |
| 5 Phase-1 frontier knobs | DEAD_CODE | none | **STRIP** | `LearnerConfig.h:162-166`, zero readers |
| `emergenceMiner` | INERT | flag on, observer-only | keep | `EMERGENCE.md` already retired the mechanism |

**DRIFTED — steered/control arena split.** Its stated purpose
(`ExampleMain.cpp:954-955`: "a slice runs unsteered as controls so a causal
auto-gate can drop alpha to 0") has been false since **2026-07-18 / commit
`36f1e01`**, when alpha went to 0. The 261 "steered" and 46 "control" arenas are
now pixel-identical — both are `IsPracticeArena` (`ExampleMain.cpp:318-330`) and
both draw `FrontierDrillState` resets. Its real current function is "which 30% of
arenas draw frontier drills". Separately `practiceArenaFrac` was escalated
0.18→0.30 with an explicit "*this lineage is END-OF-LIFE, one full-dose final
experiment*" rationale — that lineage ended two cold starts ago and the dose
carried in unreviewed.

**INERT — steering delta before the wire.** `PPOLearner.cpp:194` mutates `obs`
with `steerDelta`, then `:205` passes that same mutated tensor into
`ComputeWire`. Restoring `alpha = 0.5` would make collection derive `V_real`,
`V_exp`, `gap_KD` and (via `V_exp`) `gap_PK` from the **steered** trunk while the
learn pass re-derives them from the unsteered trunk. `36f1e01` (α→0, 14:29) and
`15eabda` (the 5-wire, 20:20) landed the same day, so **α>0 and `wireEnabled`
have never coexisted in-tree**. The documented one-line revert is no longer
one-line safe.

**Latent brace bug (three verifiers found it independently)**:
`Learner.cpp:5426-5428` is `if (steerOn) fnSteerUpdate(...); fnMetaUpdate(report);`
— `fnMetaUpdate` runs unconditionally, including in render mode. Harmless today
only because `metaOn ⊆ steerOn`.

## 2. Proposer / drills (the 9uz761ua regression machinery)

| Feature | Status | Gating (live) | Strip? | Notes |
|---|---|---|---|---|
| `FrontierDrillState`/`FrontierPool` | LIVE | `steering.enabled && !render` | **NO** | Name collision with DrillBank — different mechanism |
| Goal proposer (~1000 LOC) | DISABLED_FLAG | `proposer.enabled = false` `:657` | after decision | Nothing constructed, nothing checkpointed |
| Proposer Stage-2 shaping + car head | DISABLED_FLAG | `shapingBeta=0`, `carEnabled=false` | after decision | Unreachable twice over |
| Drill bank Stage 3 (~350 LOC) | DISABLED_FLAG | `practiceOn` false **3 ways** | after decision | `drillBank` is never constructed anywhere |
| `DrillSetter` (124 LOC) | DEAD_CODE | none — never `#include`d | **STRIP** | Not in the compile graph since the 2.6 revert |
| Proposer/* panels (12 literals) | INERT | producers all gated | **STRIP** | I26 |
| `proposer.delta.*` arch writes | INERT | net never built | after decision | I27 |
| AirDrill altitude curriculum | DISABLED_FLAG | ctor **commented out** `:1130-1131` | keep | Full post-mortem + 5 re-enable conditions in-file |
| `tools/proposer_report.py`, `drill_report.py` | DISABLED_FLAG | dumps only under `proposerOn` | after decision | Runnable today against sibling-repo dumps |
| `TransferLearn` (~280 LOC) | DEAD_CODE | none | **STRIP** | 4 self-references, zero callers |

`proposer.enabled = false` is also load-bearing for pipelined collection:
`Learner.cpp:1811` `pipelineOn = ... && !practiceOn && !proposerOn`. Re-enabling
the proposer silently halves throughput.

`AirDrillState` itself is **LIVE** at 0.20 of the reset mix — only the
`curriculum` pointer is dead. The persisted `air_drill_d` is written unguarded
(`Learner.cpp:485`) and restored unguarded (`:571-574`); re-enabling against an
old lineage would resurrect the 2026-07-15 D≈0.9 at boot. That is re-enable
condition (5), still unmet.

## 3. PSD / Basin-Racing (disabled by pre-registered verdict)

| Feature | Status | Gating (live) | Strip? | Notes |
|---|---|---|---|---|
| `PSDController` (~1040 LOC) | DISABLED_FLAG | `psd.enabled = false` `:842` | after decision | 6 null-guarded call sites; compiled into the binary |
| 42 PSDConfig fields (16 set true in ExampleMain) | DISABLED_FLAG | same | after decision | Six TRUEs directly under a false master flag |
| Plasticity canaries + interventions | DISABLED_FLAG | same | after decision | **No effective-rank / dead-unit telemetry has been emitted since 2026-07-13** |
| `PolicySlots` (EGGROLL) | DISABLED_FLAG | same | after decision | `45a59d5` *maintained* it for residual spans |
| `fitnessSource` | DEAD_CODE | none | **STRIP** | Documents a switch that was never implemented |
| `MagSGD` + ADAMW/ADAGRAD/RMSPROP arms | DISABLED_FLAG | `optimType` never selects them | **STRIP** | `e513a82` fixed a NaN bug in code nothing calls |
| `psd_selftest.cpp` | DEAD_CODE | in no build target | **STRIP** | Its build script `tools/build_psd_selftest.sh` does not exist |
| `psd_report.py` | DISABLED_FLAG | reads `psd_rounds/` | keep | Has real input under `checkpoints_3.1` |

**Re-enable is not one line.** PSD probe rollouts bypass `PPOLearner::InferActions`
and feed the bare trunk output into the policy head
(`PSDController.cpp:163-166, 330-333, 368-370`) — against a 1157-input head that
would shape-mismatch. `RunProbeRoundPureES` also scores over **all 1024 arenas**,
which now include 307 frontier-drill and 8 impossible-intercept arenas. And
`PartialResetCritic` touches only `models["critic"]`, desynchronising the vdag
twins and goal critic. `psd.enabled = true` today would abort or produce garbage.

`PSDConfig.h` **cannot be deleted** — it also declares `LeagueConfig`, which is
live.

## 4. League / QD archive

| Feature | Status | Gating (live) | Strip? | Notes |
|---|---|---|---|---|
| Member weights + evolution | LIVE | `league.enabled = true` `:899` | **NO** | Serves ~30% of iterations |
| Quantile MAP-Elites binning | LIVE | `quantileBins` default true | **NO** | Metric drift — see below |
| Evolution operators (Mutate/CrossoverDARE) | LIVE | `evolveEveryIters=16` | **NO** | Parents come from a degenerate PFSP draw |
| PFSP sampling (`pfspTemp = 1.0`) | DRIFTED | never overridden | investigate | **Units mismatch** — see below |
| Archive as opponent source | DRIFTED | same | investigate | 33 elites, P(served) ≈ 2.3e-5 |
| Exploiter fitness high-water mark | DRIFTED | `LeagueArchive.cpp:494` skips exploiters | investigate | Root cause of the PFSP collapse |
| League ANCHOR opponents | **INERT** | `anchorFrac=0.05` ON, dir missing | investigate | The anti-collapse fix is not running |
| `competenceFloor` cull path | INERT | every elite is `protectedIdx` | investigate | Cull() removes nothing at all |
| `gridAxes` names | DRIFTED | only `.size()` is consumed | keep | Header claims a name→stat mapping that was never implemented |
| Per-folder index-keyed weight store | LIVE (defect) | unconditional | investigate | Older-checkpoint restore pairs stale metadata with new weights |
| `MigrateFlatVec` 512→517 | INERT | no short vector can exist | keep | Rearms on any future width change |
| `cullFrac`, `matchesPerMember`, `Member::matches` | DEAD_CODE | none | **STRIP** | Self-labelled "(reserved)" |

**DRIFTED — PFSP.** `SampleOpponent` (`LeagueArchive.cpp:389-403`) weights
`softmax(−|fitness| / pfspTemp)` where fitness is a **raw goal difference** over a
16-arena × 300-step eval (`:363`), not a win probability. With `pfspTemp = 1.0`
(a value correct for probabilities in [0,1]) and live fitnesses of
{+2.0, +2.0, −12 … −49}, the two exploiters take **99.998%** of draws. Verified on
three independent checkpoints. Both the opponent serve *and* `EvolveStep`'s parent
selection (`:552,556`) use this, so the archive is repopulated from two members.

**Root cause.** `RefreshStalest` skips exploiters (`:494`); `EvolveExploiters` only
writes fitness on improvement (`:476-480`); `Cull` never removes them. So exploiter
fitness is a monotone high-water mark, frozen at +2.0 while every re-scored elite
ratchets to −40. Both live exploiters carry `lineage: 0` with **byte-identical
descriptors across 175M steps** — they are fossils of the first ~10 minutes of the
cold start.

**INERT — anchors.** `anchorFrac = 0.05` is set (`ExampleMain.cpp:935`) but
`anchorDir` defaults to `checkpoints_resid_anchors`, which does not exist. Live log:
`League anchors: no anchor dir … anchors DISABLED`. The `pulsar-anchor.timer` is
active every 30 min but `tools/archive_anchor.sh:26` still defaults to
`build/checkpoints_5.0v3` and self-skips forever. **`tools/trainerctl:29` was
updated for the cold start; the satellite tools were not.** Consequence: the
mechanism built specifically to cure archive collapse is off, while the collapse it
cures is present.

## 5. Optimistic-Critic Ladder

| Feature | Status | Gating (live) | Strip? | Notes |
|---|---|---|---|---|
| Sensor + quasimetric map + drive | LIVE | `gapSensor.enabled/mapEnabled/wireEnabled` true, `driveBeta=0.05` | **NO** | The run's current optimism lever |
| Impossible-control family | LIVE | `impossibleArenas = 8` | **NO — falsification control** | Currently reading BAD |
| `Ladder/Imp Touches` counter | LIVE | `ladderImpOn` | **NO** | Panel, not actuator — by design (`ImpossibleInterceptState.h:18-22`) |
| Retention gauge | LIVE | in-memory freeze | keep | Contingency is a documented *manual* response |
| λ dual ascent | LIVE | header defaults | keep | Saturates at `lambdaMax=100` given enough steps |
| `bankCapacity = 256` | LIVE | `ExampleMain.cpp:1204` | keep | **`LADDER.md:70` still says 1024** |
| `preGoalWindowSteps`, `Wire Col Grad` | LIVE | auto/derived | keep | Sentinel-0 = "auto", not "off" |
| Fear-panel comparator | INERT | `md == 1` | investigate | I11 |
| Imp falsification family reading BAD | DRIFTED | drive gate | investigate | See below |
| `driveWarmupIters`, `mapWarmupIters` | INERT | `updates` force-set 1000 on load | keep | I16 |
| Wire migration path | INERT | no short vectors | keep | I21 |
| Render-mode gapSensor | INERT | zeros | **NO** | I22 — the `ba4f33a` fix |

**DRIFTED — the falsification family.** `Ladder/Imp GapPK Spawn` = 0.83–1.19 vs
`Ladder/GapPK Mean` = 0.09–0.35, with no downward trend. `Ladder/Imp Touches = 0`
(certificate holds). But: (a) the comparator does not exist in PHASE A (I11), and
(b) all the Imp diagnostics are nested **inside** the drive gate
(`Learner.cpp:5226-5357`, which includes `!steerRatingTripped`), so if the rating
latch ever fires, every panel that would explain the trip goes dark in the same
instant — only the cumulative touch counter survives.

Also: the Ladder's own Stage-2 bridge criterion (`ExampleMain.cpp:1158`,
`LearnerConfig.h:275-283`) — "Gap/Fear Panel vs Gap/Mean, two independent frontier
detectors agreeing" — is structurally unmeasurable on this run, while `driveBeta`
and the wire are live.

## 6. Reachability

| Feature | Status | Gating (live) | Strip? | Notes |
|---|---|---|---|---|
| `reach_phi` (InfoNCE encoder) | LIVE | `reachability.enabled = true` `:632` | **NO** | Its aux loss reshapes the trunk every minibatch |
| `reach_psi_car`, `reach_psi_ball` | LIVE | same | **NO** | Feed `lastReachAccuracy` → checkpointed EMAs |
| `reach_psi_carstate` | INERT | detached, no rho reader | after decision | I6 — plus 4 clones/iter via I3 |
| Reachability reward gate + 12 knobs | DISABLED_FLAG / INERT | `gateEnabled=false` **and** `gatedPos≡0` | **STRIP** | Retired by design — gating breaks the zero-sum invariant |
| rho diagnostic read block | INERT | 1-in-16, telemetry only | investigate | I5 |
| Steering rho-band gate | INERT | α=0 | keep | I3/I14 |
| `EvalRhoRowwise` + META rho mining | DISABLED_FLAG | proposer / meta off | keep | `EvalRhoRowwise` has **zero live call sites** |

Reachability's documented purpose (`Reachability.h:10-11`, `PPOLearnerConfig.h:12-18`:
"the input of the reward gate") is stale. Its only *live* effect today is the trunk
aux gradient; every rho read is telemetry or dead-flagged. `carStateHerMaxOffset = 45`
is self-documented at `PPOLearnerConfig.h:66` as "ts4-calibrated (void at ts8);
re-calibration pending".

## 7. Aux value heads (HEADROOM / goal critic / RND)

| Feature | Status | Gating (live) | Strip? | Notes |
|---|---|---|---|---|
| Goal critic | LIVE | `goalCritic.enabled=true` `:669` | **NO** | Also drives FEAR_MINE Dz — data selection, not just advantage |
| RND novelty | LIVE | `rndOptimism.enabled=true`, w=0.1 | **NO** | Evidence drift only (see below) |
| **HEADROOM vdag twins** | **DRIFTED** | `vdagEnabled` default true, LR never set | **investigate — headline** | Frozen at random init |
| HEADROOM seek at TRUNCATED rows | LIVE (defect) | `cont = (termF==0)` | investigate | Charges `−H` on truncations; V-dagger target bootstraps from the **main** critic there |
| Four chained injectors | DRIFTED | 4 independent betas | investigate | See headline 2 |
| `Headroom/*` + `RND/*` panels | LIVE | wandb-only | investigate | `Headroom/Vdag Loss` emitted from **inside `if (reach)`** (`PPOLearner.cpp:827-846`) |

**vdag detail.** `ModelSet::StepOptims` skips only `groupStepExempt` models (set
once, on the proposer delta), so vdag1/vdag2 *are* stepped every minibatch — Muon
runs NewtonSchulz5 and then multiplies by `lr=0`. The failure is invisible in
telemetry because `InferVdagMin` forwards through `shared_head` first, so
`Headroom/Vdag Loss` falls as the *trunk* trains. Sign consequence: `min` of two
random heads makes `H` fire preferentially where `V_real` is low, so the seek
potential pays for moving toward states the honest critic rates badly — worse than
noise-neutral. The offline validation cited at `PPOLearnerConfig.h:290-291`
(`rltest/OVERNIGHT_LOG.md`) does not exist on this box.

**RND drift (evidence, not code).** The 0.1 dose was licensed by
`rnd_novelty_probe.py`, which hardcodes `FEAT = 512 + 90` (`:26`), was run on a
4.0-lineage checkpoint at tickSkip 4, and predates the 2026-07-19 h2-truncation
fix — so its enrichment ratios (2.77×/3.42×, quoted verbatim in `LearnerConfig.h:266-269`
and `ExampleMain.cpp:1143-1145`) were measured on the pre-activation trunk tap.

**Latch asymmetry**: `!steerRatingTripped` gates HEADROOM, RND and the Ladder
drive but **not** the goal-critic blend (`Learner.cpp:4821`). On a trip the goal
critic becomes the only non-extrinsic channel and its share jumps 79% → 100%.

## 8. Gym layer — obs / actions / terminals

| Feature | Status | Gating (live) | Strip? | Notes |
|---|---|---|---|---|
| `AdvancedObsPadded` (230-dim) | LIVE | unconditional | **NO** | 26% of columns are permanently-zero teammate slots (SCAFFOLD for PHASE B) |
| `DefaultAction` (90 actions) | LIVE | unconditional | **NO** | Nexto mapping asserts exact tuple equality |
| `GoalScoreCondition` + `NoTouchCondition(20)` | LIVE | unconditional | **NO** | Tick-rate safe (accumulates `deltaTime`) |
| `DefaultObs` / `DefaultObsPadded` (165 LOC) | DEAD_CODE | never constructed | **STRIP** | Single commit, never touched; stale `#include` at `ExampleMain.cpp:12` |
| `AdvancedObs::BuildObs` | DISABLED_FLAG | tests only (`GGL_BUILD_TESTS=OFF`) | keep | Parity **oracle** for the padded builder's byte-identical prefix |
| 109-dim compat branch in frontier pool | DEAD_CODE | `obsSize` always 230 | **STRIP** | Two instances — `Learner.cpp:2400` and `:2811` |
| `AttemptResolutionCondition` + `g_NumPracticeArenas` | DISABLED_FLAG | `resolutionTermination=false` | keep | See §1 |
| `ObsBuilder::Reset` | INERT | no overrides | keep | I20 — but InferUnit never calls it (parity asymmetry) |
| `MirrorPhysX` | DEAD_CODE | never called | keep | 17 LOC; correct, hard-to-rederive mirror primitive |
| `TransferLearnConfig::MakeObsFn/MakeActFn` | DEAD_CODE | dead chain | **STRIP** | Only reason obs/action factories are polymorphic |
| RLBot client (`GigaLearnRLBot`) | DRIFTED | separate target, always built | investigate | See below |

**DRIFTED — RLBot client.** `src/RLBotMain.cpp:68,77` hardcode `{512,512}` trunk
and `{512,512,512}` policy and never set `addResiduals`. The live net is
`{1152,1152,1152}` / `{768,768,768}` with residuals (`ExampleMain.cpp:714-725`).
It works today only because `rlbot-run/pulsar-bot/checkpoint/` is a frozen
2026-07-19 5.0v3 snapshot (gitignored, never auto-synced). **Danger**: residual
blocks are parameter-free and `residualSpans` is not serialized, so a partial fix
that updates only the widths would **load cleanly and silently compute the wrong
function**.

## 9. Gym layer — state setters / rewards

| Feature | Status | Gating (live) | Strip? | Notes |
|---|---|---|---|---|
| 9-setter reset mix + drill/impossible wrappers | LIVE | unconditional | **NO** | Impossible slice is the Ladder's control |
| `RewardWrapper`/`ZeroSumReward` | LIVE | 13 of 16 terms | **NO** | Missing `GetAllRewards` override is a trap for a 2nd subclass |
| `GuardedPickupBoostReward` (w=6) | LIVE | unconditional | investigate | One-sided ratchet duplicating CarEnergy's `be`; weight predates that overlap |
| `KickoffRaceReward` (w=25) | LIVE | unconditional | **NO** | Only hand-named `GetName()` in the tree |
| `FlipReset` 40 / `WallJumpToBall` 30 / `ConsecutiveAirTouch` 30 | SCAFFOLD | unconditional | keep | Exit conditions cite `mechanic_census` — see §17 |
| `DrillBank` (219 LOC) | DISABLED_FLAG | `practiceOn` false ×3 | after decision | Replay half is unreachable regardless (no `DrillSetter`) |
| `DrillSetter` (124 LOC) | DEAD_CODE | not in compile graph | **STRIP** | |
| `PlayerReward<T>` (47 LOC) | DEAD_CODE | never included | **STRIP** | **Does not compile** — derives from nonexistent `RewardFunction`, malformed `for`, wrong member name |
| 8 unused event typedefs | DEAD_CODE | never constructed | keep | `ShotReward`/`SaveReward` carry the phantom-shot exploit note |
| 10 legacy scalar rewards | DEAD_CODE | never constructed | keep | Upstream palette; `TouchBallReward` is a test fixture |
| `TouchHeightReward`, `PickupBoostReward` | DEAD_CODE | superseded | keep | **`PickupBoostReward` differs from the live one by a single guard line** — a live trap |
| AirDrill curriculum | DISABLED_FLAG | ctor commented out | keep | |
| `AirDrillState::soloFrac = 0.5` | SCAFFOLD | header default, invisible from ExampleMain | investigate | ~8% of all resets are uncontested aerials |
| Reachability gated buckets | INERT | `gated≡false` | **STRIP** | I9 |
| Team-mode reward/setter branches | INERT | PHASE A | keep | I18 |

**`soloFrac` detail.** `AirDrillState.h:45` — 2 references repo-wide, both inside
that header, **no assignment anywhere in `ExampleMain.cpp`**. So the run's lab
notebook shows only `{ airDrill, 0.20f }`. It was licensed by a 5.0v3-era census
(575M→4B, contested completion 33%→12%) and inherited verbatim through two cold
starts. Two unit tests (`tests/test_team_state_setters.cpp:113,134`) still assert
`soloFrac == 0` semantics and would fail — they are never compiled
(`GGL_BUILD_TESTS=OFF`).

## 10. Config surface

`src/ExampleMain.cpp` is designated THE configuration. These are the places where
the live value comes from somewhere else.

| Feature | Status | Gating (live) | Strip? | Notes |
|---|---|---|---|---|
| `vdagEnabled` (16.1M params, 3× critic cost) | LIVE | **header default only** | **NO** | Its only ExampleMain reference is a log line nested in the *steering* block |
| `SkillTrackerConfig` — 7 of 9 fields | LIVE | header defaults | **NO** | `ratingInc=5` alone defines the units of 4 automatic thresholds |
| `standardizeReturns` + `maxReturnSamples` | LIVE | header default true | **NO** | Divides **every reward**; see below |
| Boot sanity probe | LIVE | `bootSanityMinRating=400` default | **NO — guard** | Gate is open (rating ~935) |
| `maskEntropy` | DISABLED_FLAG | header default false | keep | Entropy normalized by `log(90)` regardless of valid-action count (18–72) |
| PSDConfig (42 fields) | DISABLED_FLAG | `psd.enabled=false` | after decision | |
| ProposerConfig (36 fields) | DISABLED_FLAG | `proposer.enabled=false` | after decision | |
| META (12 fields) | DISABLED_FLAG | `steering.meta=false` | after decision | |
| Guiding policy + `trainAgainstOldVersions` | DISABLED_FLAG | header defaults false | keep | 4 fields, ~1 branch |
| `standardizeObs` + 3 params | DISABLED_FLAG | default false | keep | Flipping it **aborts at boot** (`Learner.cpp:336-340` rejects it with steering/league/skillTracker/pipelining on) |
| `frontierTheta*` ×5 | DEAD_CODE | none | **STRIP** | |
| `drillJitterPos/Vel`, `successProximity` | DEAD_CODE | none | **STRIP** | |
| `fitnessSource`, `cullFrac`, `matchesPerMember` | DEAD_CODE | none | **STRIP** | |
| `TransferLearnConfig` (11 fields) | DEAD_CODE | none | **STRIP** | |
| Reachability gate (12 knobs) | INERT | gate off | **STRIP** | I5/I9 |
| Steering control-split + gate thresholds | DRIFTED | α=0 | investigate | |
| AirDrill curriculum (5 fields) | DEAD_CODE | pointer never assigned | keep | |

**`standardizeReturns` detail (LIVE, worth knowing).** Every raw reward is divided
by a **run-lifetime cumulative Welford std** (`GAE.cpp:51-52`) with no decay, no
window, no reset — and it is persisted. Measured: `count == 150 × iterations`
exactly; on `checkpoints_resid` σ=106.5 at 5,978 iterations (+1.4% over 594
iterations); on the mature 5.0v3 lineage σ=109.9 moved in the **6th decimal** over
113 iterations. The divisor asymptotically freezes. Consequences nobody has
recorded: (a) the SCAFFOLD reward anneals (AerialTouch 120→50, AirIntercept 40→20)
will change the raw reward scale while the divisor does not follow; (b) at σ≈106
the `rewardClipRange = 50` is inert by design (Goal 150 → 1.41 standardized).
Note the goal critic trains in **raw** units (`Learner.cpp:4791` passes
`returnStd=0`), so the two value heads live in different unit systems — safe only
because the blend is std-matched.

## 11. Util / infra

| Feature | Status | Gating (live) | Strip? | Notes |
|---|---|---|---|---|
| Muon optimizer | LIVE | `optimType = MUON` `:743-746` | **NO** | Momentum buffers are checkpoint-entangled (also vdag1/vdag2) |
| GAE, ExperienceBuffer, Welford, Report, Models, ThreadPool, Timer, GameState | LIVE | unconditional | **NO** | Foundational |
| `MagSGD` + 3 unused optimizer arms | DISABLED_FLAG | `optimType` never selects them | **STRIP** | `MagSGD.cpp` is a 19-byte TU |
| `BatchedWelfordStat` / obs standardization | DISABLED_FLAG | `standardizeObs=false` | keep | Re-enable **aborts at boot** |
| `KeyPressDetector` | DISABLED_FLAG | `isatty(STDIN)` false under systemd | keep | Correct env gate, logs itself |
| `RenderSender` | DISABLED_FLAG | `renderMode` | **NO** | Live in the separate viewer process |
| `PolicyVersionManager::renderSender` | DEAD_CODE | ctor default NULL, never passed | **STRIP** | 4 LOC + 2 stale comments claiming eval arena 0 "feeds the render sender" |
| `Model::Load` wire zero-pad | INERT | no short vectors | keep | I21 |
| `ExperienceTensors::practiceMask` lane | DISABLED_FLAG | `resolutionTermination=false` | keep | `fnMaskedMSE` degrades to plain `mseLoss` |
| `Learner::StartTransferLearn` (~172 LOC) | DEAD_CODE | no caller | **STRIP** | Drags `PPOLearner::TransferLearn` (~50 LOC) with it |
| `RG_AUTOCAST_ON/OFF` macros | DEAD_CODE | zero expansions | **STRIP** | bf16 is done manually via `seqHalf` |
| `Report::Add/FinishAvg/ToString/operator+`, `digitCommas` | DEAD_CODE | no callers | **STRIP** | **`Report::Clear` is LIVE** (`Learner.cpp:3943`) — do not sweep it |
| `RLGC::Quat` (39 LOC) | DEAD_CODE | never included | **STRIP** | `Angle::ToRotMat` calls are the vendored engine's, not this |
| `EnvCreateResult::userInfo` lane | DEAD_CODE | written, never read | **STRIP** | |
| `EnvSet::ResetArena` control clear | INERT | `actionDelay=0` | keep | I19 |
| `InferUnit` + `GigaLearnRLBot` | DRIFTED | separate target | investigate | See §8 |

## 12. Outer loops (Elo / golden archive / Nexto / PHASE B)

| Feature | Status | Gating (live) | Strip? | Notes |
|---|---|---|---|---|
| Skill tracker + policy version ring | LIVE | `skillTracker.enabled=true` `:769` | **NO** | Forces `savePolicyVersions` |
| `GetPolicyModels` vdag filter | LIVE (**uncommitted**) | unconditional | **NO** | Fixes ~2.06GB VRAM; a `git checkout` reverts it |
| Rating peak arm + thresholds | SCAFFOLD | 200/150 vs header 110/75 | keep | Exit written in-file |
| Rating EMA drawdown arm | DRIFTED | 150 vs EMA 570 at rating 935 | investigate | Needs a ~55% collapse to fire; cannot arm on a climb |
| Elo output as actuator input | DRIFTED | `updateInterval=16` | investigate | ~6× overstatement; 4 automatic actuators consume it |
| PHASE A→B trigger (1200) | DRIFTED | marker file absent | investigate | Same bar tripped at 3.20B/5.69B/8.06B/9.35B across four lineages |
| Golden archive (top-3 by Rating) | DRIFTED | `bestArchiveMinTsSpacing=25M` | investigate | **All 3 entries inside the 200M rotation window — zero depth** |
| `trainAgainstOldVersions` | DISABLED_FLAG | default false | keep | 3 LOC; `pipelineOn` depends on it being false |
| Per-mode Elo fleet (2v2/3v3) | DISABLED_FLAG | PHASE A | keep | Proven on 5.0v3 |
| Nexto multi-mode batching | DISABLED_FLAG | PHASE A | keep | Untested-in-anger for the flip |
| Nexto goal counters | DRIFTED | unfiltered arena loop | investigate | ~14% of counted goals are free impossible-arena goals |
| Nexto CUDA probe | INERT | probe always throws | keep | I10 |
| Nexto `freshEpisode` reset | INERT | terminals pre-zeroed | investigate | I23 |
| `PolicyVersionManager::renderSender` | DEAD_CODE | never passed | **STRIP** | |

**Golden archive detail.** `bestArchiveMinTsSpacing = 25M` was chosen when
`tsPerSave` was 1M (25 saves apart). `tsPerSave` is now 25M
(`ExampleMain.cpp:790`), so the spacing gate is a **no-op** and the archive's
maximum reach (3 × 25M = 75M) is *narrower* than the rotation window it backstops
(8 × 25M = 200M). Live: `best_r919_1675172864`, `best_r911_1750145024`,
`best_r935_1775291392` — all inside rotation. `lastBestArchiveTs` is not persisted,
so restarts reset the rate limit too.

**Nexto counters detail.** `Learner.cpp:3745-3754` iterates **all** `gameStates`
with no arena filter, five lines after `Ladder/Imp Touches` does exactly that
filtering (`:3739-3743`). The 8 impossible arenas score a free goal roughly every
14 steps, contributing ~111 goals/served iteration ≈ 14% of all counted goals and
~23% of `Nexto/Goals For`. Symmetric (target side randomized) so the sign is
unbiased, but it dilutes the ratio toward 1 — and the counters are **persisted**, so
the longitudinal series cannot be retro-corrected. Fix is a one-line index range.

## 13. Phases / curriculum / env vars

| Feature | Status | Gating (live) | Strip? | Notes |
|---|---|---|---|---|
| Render env family (`GGL_RENDER*`, `GGL_DEVICE`) | LIVE | systemd viz unit | **NO** | Architecture-parity `else` branch is the `ba4f33a` scar |
| `MAX_PLAYERS_PER_TEAM=3` padding | SCAFFOLD | unconditional | **NO** | 60 of 230 columns permanently zero; the price of phase switching |
| Team arena block allocation | DISABLED_FLAG | no PHASE_B marker | keep | Runtime contiguity assert makes mis-layout fail loudly |
| PHASE B auto trigger | DRIFTED | Rating ≥1200 × 3 | investigate | Self-arming, irreversible, restarts the process |
| TEAM_SPIRIT schedule | INERT | 1v1 identity | keep | I17 |
| `GGL_SMOKE` (3 blocks) | DISABLED_FLAG | env unset | keep | Actively prescribed by `REWARD_SHAPING.md:1059` |
| `GGL_FRONTIER_POTENTIAL` | DISABLED_FLAG | env unset | keep | Verified unset in `/proc/<pid>/environ` |
| `g_NumPracticeArenas` plumbing | DISABLED_FLAG | `resolutionTermination=false` | keep | **Name collision** with the live `g_PracticeArenaFrac` |
| `dashboard.py` checkpoint dir | DRIFTED | `checkpoints_5.0v3` default | investigate | Remote dashboard reports on a dead lineage |
| Anchor timer / `archive_anchor.sh` | DRIFTED | `checkpoints_5.0v3` default | investigate | See §4 |

**PHASE B latent landmine**: `g_NumPracticeArenas` uses a *flat* `numGames × frac`
rule while the Learner tags practice rows with a *per-mode* rule
(`Learner.cpp:1440-1447`). They coincide exactly in PHASE A and diverge in PHASE B —
reproducing the critic-aliasing failure the post-mortems blame for two Elo
collapses. Harmless only because the flag is off.

## 14. Aux trees / build

| Feature | Status | Gating (live) | Strip? | Notes |
|---|---|---|---|---|
| RocketSimV3 compat facade | LIVE | `GGL_ROCKETSIM_V3=ON` | **NO** | This IS the physics engine |
| `build-viz/` (149MB) | LIVE | `trainerctl` manages it | **NO** | Rebuilt in lockstep; checkpoints are symlinks |
| `cpp-interface` uncommitted patches | LIVE (**at risk**) | working tree only | **NO** | See below |
| `rlbot-run/` | LIVE | manual | **NO** | Trainer depends on `nexto/nexto-model.pt` at an absolute, **gitignored** path |
| RocketSim v2 tree (68k LOC) | DISABLED_FLAG | `GGL_ROCKETSIM_V3=OFF` | keep | v3 compat **reuses its math/PhysState headers** — cannot delete |
| `cpp-interface` FetchContent at configure time | LIVE (risk) | unconditional | after decision | A fresh `cmake ..` needs **network + a liburing build** for a target the trainer never links |
| `build-steer/` (2.0GB) | DEAD_CODE | referenced by 1 stale doc line | **STRIP** | CLAUDE.md says "never executed" — 28 wandb offline runs say otherwise |
| ATBA / ExampleBot / launcher | DEAD_CODE | unconditional `add_subdirectory` | **STRIP** (guard) | 3 LTO-linked binaries per rebuild, incl. on the training box |
| `rlbot/` (v4 Python harness) | DEAD_CODE | none | **STRIP** | Its C++ peer was deleted by `34e6995` |
| RLBotCPP benchmark | DISABLED_FLAG | `BUILD_BENCHMARK=OFF` | keep | Would FetchContent a **second** RocketSim |
| `GigaLearnTests` | DISABLED_FLAG | `GGL_BUILD_TESTS=OFF` | keep | Maintained; `test_team_envset_smoke.cpp` is 4.0-era drifted |
| `tools/checkpoint_converter.py` | DEAD_CODE | `import rlgym_ppo` unavailable | **STRIP** | Targets `PPO_POLICY.pt` — a format that no longer exists |

**`cpp-interface` — the single highest-risk item in the repo.** Two load-bearing
patches exist **only** in the submodule working tree, in no commit anywhere:
(1) `CMakeLists.txt:23-31` bumps the flatbuffers-schema tag to the one RLBotServer
v5.0.0-rc16 requires (older cores drop the game link 1s after map load);
(2) `library/Client.cpp:27-32` cuts `PREALLOCATED_BUFFERS` 32→4 because at 32 each
bot pins 4MB of io_uring memory against an un-raisable 8MB `RLIMIT_MEMLOCK`, so
only one bot could register — which broke 2v2/3v3. A routine
`git submodule update` or `git -C cpp-interface checkout .` silently discards both,
and the resulting failures look like RLBot/game problems.

## 15. analysis/probes (offline toolkit)

| Feature | Status | Gating (live) | Strip? | Notes |
|---|---|---|---|---|
| `.venv`, `data/`, `__pycache__` | LIVE | documented invocation | **NO** | `data/` is hardcoded in 25+ scripts |
| `load_checkpoint.py` architecture lock | DRIFTED | hardcoded 512/2-layer | investigate | **Blocks the whole toolkit** |
| `rebuild_sequential` drops residual skips | DRIFTED | no span input possible | investigate | Silent-wrong-numbers hazard; 6 scripts rebuild `GOAL_CRITIC` raw |
| `_default_root()` → `checkpoints_5.0v3` | DRIFTED | env override only | investigate | Silently analyzes a frozen run |
| `collect_dataset.py` reset mix | DRIFTED | module constants | investigate | Tick dynamics **correct** (8/0/20s); mix missing `AirPlayState` |
| Steering dose/validation family | DRIFTED | α=0 | keep | `steer_test.py`/`steer_team.py` are **shared infrastructure** — 10 and 27 importers |
| Team-mode probe family | DRIFTED | PHASE A | keep | `steer_team.py` is the general N-player harness |
| Meta/frontier validators | DRIFTED | meta off | keep | `frontier_validate.py` is imported by 3 live-mechanism scripts |
| `derive_steering.py`, `export_styles.py` | DRIFTED | sinks deleted from C++ | after decision | `derive_steering.py` never calls `set_obs_size` → 109-dim shape mismatch |
| 21 of 30 `.md` pre-2026-07-19 | DRIFTED | n/a | keep | h2-truncation invalidated their numbers |
| `README.md` pipeline section | DRIFTED | n/a | **fix** | Says `checkpoints_3.1` and "tickSkip-4/actionDelay-3" |
| `REWARD_SHAPING.md` (128KB) | LIVE | n/a | **`git add`** | The newest canonical report is **untracked** |

The `.md` corpus stopped tracking the run at 2026-07-21. `REWARD_SHAPING.md`
(2026-07-25) does carry a cold-start status box, but it is untracked. Separately:
`src/ExampleMain.cpp:684-686` licenses the entire residual architecture by citing
`~/Projects/experiments/arch/RESULTS.md` — **that directory is empty**. The sole
empirical justification for the live net topology is unreachable.

## 16. Telemetry integrity

Grouped because these are all "the panel does not measure what its name says".

| Panel | Problem | Anchor |
|---|---|---|
| `Rewards/<term>` | Unweighted, pre-ZeroSum child, one random player **per term**, 50 arenas with replacement, 1-in-9 steps | `EnvSet.cpp:238-266`, `Learner.cpp:3706-3721` |
| `Rewards/<term>` keys | GCC-mangled (`Rewards/N4RLGC17TouchAccelRewardE`) — the trim logic only works for MSVC | `Reward.h:32-56` |
| `GAE/Avg Advantage` | Computed after the HEADROOM injection | `Learner.cpp:4778` vs `:4760` |
| `Steer/RhoGate In-Band Frac` | Frozen initializer 0 forever | `Learner.cpp:5414` |
| `Steer/Alpha`, `Steer/Gate *` | Report a constant 0 / a zero-treatment contrast | `Learner.cpp:5411-5412` |
| `Ladder/Fear GapPK`, `Gap/Fear Panel` | Never emitted in PHASE A | `Learner.cpp:5207,5218` |
| `Curriculum/AirDrill D`, `Aerial Conv EMA` | Never emitted (curriculum pointer NULL) — yet `aerialHighN`/`aerialConvN` are computed every iteration | `Learner.cpp:2623-2645` |
| `Headroom/*`, `RND/*`, `Frontier/*`, `Player/*`, `Rating/*` | wandb-only; **absent from the curated stdout block** | `Learner.cpp:5538-5628` |
| 3 phantom rows + 12 gated rows | ~14 of 75 advertised rows can never render | `Learner.cpp:5539-5628` |
| `Nexto/Goals For/Against` | ~14% free goals from the impossible-drill arenas | `Learner.cpp:3745-3754` |

## 17. Scaffolds with an unreachable exit condition

Six reward weights (260 nominal, vs Goal 150) carry "anneal once
`mechanic_census` shows…" comments: AerialTouch 120, AirIntercept 40, FlipReset
40, WallJumpToBall 30, ConsecutiveAirTouch 30 (+ `soloFrac` 0.5).
`analysis/probes/mechanic_census.py` counts `wavedash_like`, `flip_cancel_ish`,
`dribble`, `takeoff_attempt` — **`grep -i reset` returns nothing**, and it routes
through the architecture-locked loader anyway. The named anneal trigger cannot
fire for any of them.

The in-trainer alternative is also dead: `aerialHighN`/`aerialConvN`
(`Learner.cpp:2068,2099-2101`) are accumulated every iteration and read **only**
inside `if (config.steering.airDrillCurriculum)` — a pointer whose assignment is
commented out.

---

# Entanglement map

**Independent — strip alone, zero coupling:**
`PlayerReward<T>` · `DrillSetter` · `RLGC::Quat` · `RG_AUTOCAST_*` ·
`MirrorPhysX` (if chosen) · `fitnessSource` · `cullFrac`/`matchesPerMember`/`Member::matches` ·
`frontierTheta*` ×5 · `drillJitter*`/`successProximity` · `checkpoint_converter.py` ·
`rlbot/` · `build-steer/` · the 3 phantom stdout rows · `userInfo` lane ·
109-dim compat branches.

**Ordered / coupled:**

1. `DefaultObs` + `DefaultObsPadded` → also delete the stale `#include` at
   `ExampleMain.cpp:12`. Do **not** touch `AdvancedObs::BuildObs` (parity oracle)
   or `InvertPhys` (live in the padded builder).
2. `TransferLearn` chain: `Learner::StartTransferLearn` (172 LOC) +
   `PPOLearner::TransferLearn` (50) + `TransferLearnConfig` (51) + 2 `#include`s.
   **Keep `StartQuitKeyThread`** — also called from the live `Start()` (`:1148`).
3. `MagSGD` removal requires deleting the `MAGSGD` enumerator **and** both switch
   arms, or `SetOptimizerLR`'s `default:` fires `RG_ERR_CLOSE`.
4. Proposer bundle (proposer + Stage-2 shaping + car head + DrillBank +
   Proposer/* panels + delta config + report tools) is one unit. **Preserve**:
   the `|| config.reachability.carStateHead` disjunct at `PPOLearner.cpp:47`
   (keeps `psi_carstate` alive), `carStateHerGoals`/`achievedCarState`, and the
   `!proposerOn` term in `pipelineOn` (`Learner.cpp:1811`).
5. PSD bundle: `PSDController` + `PolicySlots` + `Plasticity` + 42 config fields +
   `psd_selftest.cpp`. **Cannot delete `PSDConfig.h`** (declares the live
   `LeagueConfig`). Removal also frees the model builder from the flat-`seq`
   constraint documented at `ModelConfig.h:38-40` — and **would delete the only
   `EffectiveRank`/`DeadUnitFraction` implementation**, which the residual cold
   start (`45a59d5`, motivated by effective-rank decay) has no substitute for.
6. Steering bundle: stripping the derivation (I1) **kills `FrontierPool`**, which
   kills `FrontierDrillState` on 30% of arenas. Must be split: keep sections 1–4
   (readings/landing sims/possession labels) and 6–8 (frontier fill/census/miner);
   remove only sections 5/5b (trunk contrast, sigma EMA, challenge contrast) and
   `fnApplySteering`/`SetSteering`.
7. **`fnRatingGuard` must be lifted out of `steering.*` BEFORE any steering
   cleanup.** Otherwise `steering.enabled = false` disarms RND, the Ladder drive,
   HEADROOM seek, Nexto serving, league anchors and frontier drills at once.
8. `arenaSteerRole` is shared: role 3 tags the Ladder's impossible arenas
   (`Learner.cpp:1485-1486`) and the overlap assert (`:1481-1484`) reads
   `steerBlocks[0].numPractice`. Do not strip the role vector or change
   `practiceArenaFrac` casually — both can crash the boot.
9. `IsPracticeArena` (`ExampleMain.cpp:318-330`) must mirror `steerBlocks`
   arithmetic (`Learner.cpp:1438-1447`) — two independent sites, no shared code.
10. `frontierFearMining` supplies `dzOk`/`fnDz` to `Miner/Dz Mean`, `censusWonDz`
    and the `frontierPotential` telemetry. Flipping it false silently blanks
    those too. It is also the sole writer of `fearPanelObs` → `Ladder/Fear GapPK`.
11. Reachability reward gate: remove `WeightedReward::gated`,
    `EnvState::gatedPosRewards`, `EnvSetConfig::collectGatedBuckets`,
    `Trajectory::gatedPos`, **and** the `gatedPos.size() == n` term in
    `AssertAligned` (`Learner.cpp:1350`) together, or the assert fires.
12. `reach_psi_carstate` removal must **keep** `carStateHerGoals` and
    `achievedCarState` (shared with the disabled car proposer) and the
    `models["reach_psi_carstate"]` fallback at `PPOLearner.cpp:357`.
13. `Report::Add/FinishAvg/ToString/operator+` are strippable; **`Report::Clear`
    is LIVE** (`Learner.cpp:3943`) and prevents stale `Rewards/*` keys from
    re-merging every iteration.
14. ATBA/ExampleBot: prefer `EXCLUDE_FROM_ALL` over deletion — `cpp-interface` is
    an upstream submodule and deleting them diverges from it. Removing them
    leaves `RLBotCPP-static` with only the OFF-gated benchmark as a consumer.
15. RocketSim v2: only the 16 engine `.cpp` + non-LinearMath bullet3 are
    removable. `Math.cpp`/`MathTypes.cpp`/`PhysState.cpp` are **compiled into
    RocketSimV3**, and 20 headers (7 of them bullet3 `LinearMath`) are in the live
    dependency graph. Pruning also forfeits the `GGL_ROCKETSIM_V3=OFF` A/B path.
16. `load_checkpoint.py`: fixing `_default_root()` alone converts ~30 scripts from
    *silently stale* to *hard crash*. `expected_shapes`, `rebuild_sequential`
    (residual spans), `PulsarPolicy.wire_pad`/`trunk_block1|2`, and `MODEL_FILES`
    (missing `VDAG1/2`, `GAP_*`, `RND_*`) must change in the same commit.

---

# Recommended strip order

## Tier 1 — safe / mechanical (no behavioral risk)

Verified zero call sites or zero-effect literals. Nothing here can change training.

| Item | Payoff |
|---|---|
| `PlayerReward<T>` — does not even compile | 47 LOC |
| `DrillSetter` — not in the compile graph | 124 LOC |
| `RLGC::Quat` | 39 LOC |
| `RG_AUTOCAST_ON/OFF` + `<ATen/autocast_mode.h>` in 13 TUs | 12 LOC, 1 include |
| `DefaultObs` + `DefaultObsPadded` + stale include | 165 LOC |
| `TransferLearn` chain (`Learner` + `PPOLearner` + config + includes) | ~280 LOC |
| `MagSGD` + 3 unused optimizer arms + enum members | ~75 LOC |
| `PolicyVersionManager::renderSender` + 2 false comments | 4 LOC |
| `Report::Add/FinishAvg/ToString/operator+/digitCommas` (**not `Clear`**) | ~45 LOC |
| Dead config fields: `frontierTheta*` ×5, `drillJitter*`+`successProximity`, `fitnessSource`, `cullFrac`, `matchesPerMember`, `Member::matches` | 11 fields |
| 3 phantom stdout rows + 10 Proposer/* rows | 13 literals |
| 109-dim compat branches (`Learner.cpp:2400`, `:2811`) | ~8 LOC |
| `EnvCreateResult::userInfo` lane | ~4 LOC |
| `MirrorPhysX` (optional — 17 LOC of correct mirror math) | 17 LOC |
| `tools/checkpoint_converter.py`, `rlbot/` (9 files) | ~170 LOC |
| `build-steer/` | **2.0 GB disk** |
| `git add analysis/probes/REWARD_SHAPING.md` | prevents loss of the newest canonical report |
| Fix `analysis/probes/README.md` (3.1 → resid, tickSkip 4 → 8) | doc |
| Fix `CLAUDE.md` live weights (CarEnergy 15 → 75; add the 5 missing terms) | doc |
| One-line: exclude impossible arenas from the Nexto goal counters | fixes the only external yardstick |
| One-line: `matchEnv->rewards[i].clear()` (mirrors `PolicyVersionManager.cpp:31`) | removes I8 |
| ATBA/ExampleBot → `EXCLUDE_FROM_ALL` | 3 LTO links per rebuild |

**Tier 1 total: ~1,000 LOC + 2.0 GB, plus 4 doc corrections and 2 one-line bug fixes.**

## Tier 2 — safe but needs an owner decision

| Item | Payoff | Decision needed |
|---|---|---|
| Proposer bundle (proposer + Stage 2 + car head + DrillBank + panels + delta config + 2 report tools) | ~1,500 LOC + 36 config fields + 474 LOC Python | Is the proposer line permanently retired? |
| PSD bundle (controller + PolicySlots + Plasticity + 42 fields + selftest) | ~1,300 LOC | Same — but keep `EffectiveRank`/`DeadUnitFraction` (see below) |
| META steering (~470 LOC + 12 fields) | ~500 LOC | Same |
| Steering derivation sections 5/5b + `fnApplySteering`/`SetSteering` + `steer_vec` payload | ~250 LOC + **3 GPU forwards/iter** | Is steering paused or dead? |
| Steering causal gate (I2) + rho-gate snapshot (I3) | ~90 LOC + **4 param copies/iter in the barrier** | Same |
| `reach_psi_carstate` (I6) | 1 head, 4.2MB/ckpt, per-minibatch fwd+bwd+sync | Is META ever coming back? |
| Reachability reward gate (12 knobs + ~140 LOC + gated-bucket plumbing) | ~180 LOC + per-row accumulate | Retired by the zero-sum invariant — safe to confirm |
| `cpp-interface` FetchContent → option-guarded | removes network+liburing from a fresh `cmake ..` | Is always-ready RLBot worth the offline-rebuild risk? |
| RocketSim v2 engine `.cpp` + non-LinearMath bullet3 | ~65k LOC, 3 MB | Forfeits the `GGL_ROCKETSIM_V3=OFF` A/B path |

**Tier 2 total: ~4,300 LOC C++ + 474 LOC Python + ~65k LOC vendored, plus the
largest recurring inert compute in the run.**

## Tier 3 — do NOT strip

**Protected — LIVE:** the Ladder core, RND, goal critic, reachability phi/psi_car/psi_ball,
Muon, GAE/ExperienceBuffer/Welford, `AdvancedObsPadded`, `DefaultAction`,
`GoalScoreCondition`/`NoTouchCondition`, the 9-setter reset mix, `FrontierPool`/
`FrontierDrillState`, the league archive/evolution, the skill tracker, `RocketSimV3`,
`build-viz`, `rlbot-run`, the `cpp-interface` patches.

**Protected — falsification controls and guards** (these exist to *catch* the
project being wrong; removing them removes the ability to be refuted):
- `ImpossibleInterceptState` + `Ladder/Imp *` (`ExampleMain.cpp:457-459`,
  `Learner.cpp:5316-5357`) — the run's only standing falsification test.
- `fnRatingGuard` (`Learner.cpp:3240-3272`) — the only actuator that can see
  update damage across six live mechanisms.
- Boot sanity probe (`Learner.cpp:611-677`) — the 2026-07-13 corrupt-but-loadable
  defense; the only check that catches finite-weight/destroyed-policy checkpoints.
- Golden archive (`Learner.cpp:712-745`) — needs *repair*, not removal.
- The steered/control **role vector** (`arenaSteerRole`) — the Ladder's impossible
  arenas ride on it.

**Protected — deliberately parked with recorded post-mortems** (the code IS the
institutional memory; deleting it deletes the lesson):
- `resolutionTermination` / `AttemptResolutionCondition` — two Elo collapses.
- AirDrill altitude curriculum — reverted 2.5h after deploy, Rating 1657→1546, with
  5 written re-enable conditions.
- `opponentStyleChance` — parked by the Stage-2 protocol, zero cost.
- `ShotReward` / `SaveReward` / `PickupBoostReward` / `TouchHeightReward` typedefs —
  each annotated with the exploit or the measured failure that retired it.
- The 21 pre-2026-07-19 `.md` reports — the paid-for experimental record.
- `GGL_SMOKE` and `GigaLearnTests` — actively prescribed by the newest audit doc.
- `PSD::EffectiveRank` / `DeadUnitFraction` — cheap weights-only diagnostics the
  *current* residual cold start would benefit from. If PSD goes, **promote these
  two to unconditional Learner telemetry** rather than deleting them.

---

# Open questions for the owner

These are judgement calls that depend on research intent, not on code.

1. **Is the activation-steering line dead or paused?** Everything in §1 hinges on
   it. If dead: Tier 2 releases ~250 LOC and 3 GPU trunk forwards per iteration,
   and the rating latch must be re-homed first. If paused: the α=0.5 revert is no
   longer one line (the wire-ordering hazard) and needs a documented fix first.
2. **Is the proposer/drill line permanently retired?** ~1,500 LOC + the Python
   report tools. `proposer.enabled = false` is currently load-bearing for pipelined
   collection, so re-enabling costs throughput independent of the merits.
3. **Is PSD permanently retired?** The disable was a pre-registered verdict, but
   the code is now architecturally incompatible with the live net (wire columns,
   arena pool, critic family). If it stays, add a boot-time assert that
   `psd.enabled` is incompatible with `gapSensor.wireEnabled`, so nobody mistakes
   the sub-flags for a working lever.
4. **HEADROOM: fix the LR, or turn it off?** Adding
   `models["vdag1"/"vdag2"]->SetOptimLR(criticLR)` **turns HEADROOM on for the
   first time** — a new deployment needing pre-registration and latch coverage, not
   a bug fix. Setting `vdagEnabled = false` reclaims 42% of parameters and removes
   an adverse 0.15σ injection. Do not silently "fix" it mid-run.
5. **What should the composite injection budget be?** Four betas were each chosen
   against a world with fewer injectors. Is ~0.32 σ_ext the intended total?
   Should every injector std-match against the *same* pre-injection extrinsic std?
6. **Should the impossible-control arenas be masked from HEADROOM / goal-critic /
   RND injection**, as they already are from the Ladder drive? Right now the
   falsification family receives three of the four injections.
7. **How is the Ladder's acceptance criterion evaluated in PHASE A?** Either
   re-key the fear-panel freeze to `md == 0`, or formally retire the
   `Imp GapPK < Fear GapPK` bar in `LADDER.md`. As written, the standing
   falsification test is unscoreable while the drive is live.
8. **Is the PFSP/league opponent pool worth repairing?** Fixing `pfspTemp` units
   and the exploiter high-water mark would immediately start serving 33 elites
   that have never been served — a live data-distribution change. Alternatively,
   seed `checkpoints_resid_anchors` and let anchors carry the diversity load.
9. **Re-anchor the PHASE-B trigger?** 1200 was calibrated on the 3.1 lineage and
   has tripped at 3.20B–9.35B across four runs on a metric measured to overstate
   progress ~6×. Options: raise it, key it to Nexto goal share / `anchor_battery`,
   or make PHASE B manual.
10. **When do the SCAFFOLD reward weights anneal?** Six weights totalling 260 (vs
    Goal 150) name `mechanic_census` as their trigger; it measures none of the
    relevant mechanics and cannot load the live checkpoints. Either implement the
    metric or delete the "anneal later" comments so they stop reading as temporary.
11. **`CarEnergyPotential` 15 → 75 is uncommitted and live.** A 5× on the run's
    tempo term, with a stale rationale comment above it and no commit. Commit with
    rationale, or revert. (Static extrapolation from `REWARD_SHAPING.md`'s measured
    table puts it near 60% of per-step credit density — needs re-measurement, not
    assumption.)
12. **`soloFrac = 0.5`** — ~8% of all resets are uncontested aerials, licensed by a
    5.0v3 census, invisible from the config file. Surface it into ExampleMain with
    a dated rationale, or re-measure the mutual-bail rate on this lineage.
13. **Repair the offline toolkit, or accept it is 5.0v3-only?** Nothing can be
    measured against the live run today, including the honest-progress yardstick.
    A partial fix (widths without residual spans) is worse than none.
14. **Where should the anchor lineage live?** `checkpoints_5.0v3_anchors` is
    architecturally unloadable by the resid net, so the fixed-yardstick Elo history
    is permanently discontinuous. Repoint the timer + seed, or accept the gap.

---

# Confidence and human-check flags

High confidence (direct code + live-process verification): all gating resolutions,
all call-site counts, the INERT table, the HEADROOM LR finding, the PFSP
arithmetic, the anchor/dashboard lineage pins, the load_checkpoint lock.

**Needs a human check** — asserted but not verified by running anything:

- **HEADROOM magnitude.** That `lr=0` freezes the twins is proven from code; the
  *behavioral* consequence of a 0.15σ random-projection injection is inferred, not
  measured. A wandb check of `Headroom/Inj Abs Mean` vs `Headroom/Vdag Loss`
  trajectories would confirm.
- **Ladder `Imp GapPK` interpretation.** The elevated value is real; whether it is
  self-serving optimism or an artifact of unusual spawn states (car at rest in the
  far corner, ball airborne) cannot be decided without the missing comparator.
- **`standardizeReturns` freeze consequences.** The arithmetic is verified; the
  claim that SCAFFOLD anneals will shift effective advantage magnitude is a
  prediction.
- **Composite injection ≈ 0.32 σ.** Computed from the four betas assuming
  independence; the HEADROOM floor (`sInt = max(0.05·sExt, aInt.std())`) and the
  Ladder's masked-row zeros make it an upper bound.
- **`GGL_SMOKE` dose adequacy after the residual widening.** `miniBatchSize ==
  tsPerItr == 25k` at a 4.5× larger per-row activation footprint was flagged, then
  argued down (CPU sandbox, 42 GB free, `numGames` also drops 8×). Not tested.
- **`GGL_ROCKETSIM_V3=OFF` still compiles.** Inferred from zero consumer churn
  against the facade (`b1e1e9a` touched exactly one gym file), not built.
- **`AdvancedObs::BuildObs` / `GigaLearnTests` still compile.** `git log` shows no
  gym-layer change since the tests were last updated, but the target has not been
  built.
- **`build-steer` execution history.** 28 wandb offline-run dirs prove it was run;
  the checkpoint copies under it share one mtime, so they are a `cp -r`, not a
  rotation. The "never executed" doc line is wrong either way.
- **RLBot deployment.** The client works against its frozen 5.0v3 checkpoint
  (bot logs show clean loads); that it would abort against `checkpoints_resid` is
  inferred from `Models.cpp:243-255`, not observed.
