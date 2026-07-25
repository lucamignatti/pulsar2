# LEAGUE RECON — 2026-07-25

Reconnaissance on the QD League before a rework. Engineering record, not a research
report: it answers *what the trainer is currently doing*, not *what is true about the
agent*. No design or rework plan here by request.

**Method.** Eight parallel readers over the C++ sources and the wandb history
(entity `lucamignatti-personal`, project `gigalearncpp`), then every defect claim was
handed to an independent adversarial verifier told to refute it: **63 claims → 59
survived, 4 refuted**. Three critics then swept for gaps, contradictions and whether
the subsystem earns its compute. Numbers are from the live run **434etlix**
(`resid-768p-1280v`) unless marked.

**Companion docs:** `ENABLED_INVENTORY.md` (what is live), `LEAGUE_ANCHORS.md`
(the anchor pre-registration — note §3.7 below, its premise is wrong).

---

# The QD League: what we're working with

*All numbers from run **434etlix** (`resid-768p-1280v`, live) unless marked. Live-log/checkpoint figures re-read at iteration 2,046–2,070 / 650–659M steps while writing.*

---

## 1. What the league is

A MAP-Elites archive of frozen policy snapshots, used as sparring opponents. A **member** is two flat CPU tensors — `shared_head` and `policy` only, no critic, no vdag, no reach heads (`LeagueArchive.cpp:17`) — plus a 3-float behaviour descriptor, a scalar fitness, and a cell index (`LeagueArchive.h:25-34`). Both the BD and the fitness come from one shared measurement: `EvaluateMember` (`LeagueArchive.cpp:261-333`) plays the member (always BLUE, `:268`) against the **live main** for 300 decision steps on 16 isolated arenas with the reward stack cleared and `GoalScoreCondition` as the only terminal (`:25`, `:33-35`, `:281`). Fitness is the raw integer `aGoals - bGoals` (`:332`); the BD is three hardcoded per-step means over the member's cars (`:306-327`): fraction of steps not on ground, `min(1,|y|/5120)`, `boost/100`. Cells are a base-6 mixed-radix index over those three axes — `binsPerAxis^|gridAxes|` = 6³ = **216 cells** (`:554-555`) — with quantile edges fitted to a rolling 512-BD window and refreshed only on the `reseedEveryIters`=1000 cadence (`:611-618`). Every `evolveEveryIters`=16 iterations, `EvolveStep` (`:519-550`) breeds 4 candidates by Gaussian mutation (σ=0.02) or DARE crossover (`Operators.h:12-26`), hill-climbs one of 2 exploiter slots, re-scores **exactly one** archived member round-robin, and calls `Cull` — 6 `EvaluateMember` calls in steady state, all serialised in the barrier zone. Serving: on `descendOpponentFrac`=0.35 of the iterations Nexto did not pre-empt, one member is drawn by `softmax(-|fitness| / pfspTemp=1.0)` (`:358-372`) and installed as the opponent for **all 1024 arenas at once** (`Learner.cpp:1346`).

### The three pools, and why there are three

| Pool | Purpose | Storage | Refresh | Live? |
|---|---|---|---|---|
| **1. Policy versions** (`PolicyVersionManager`) | **Measurement only** — produces `Rating/1v1` | ModelSet clones in VRAM + `policy_versions/<ts>/` | one every 25M steps, ring of 32 → ~800M window | yes (24 dirs on disk) |
| **2. League members** | **Training** — behavioural diversity via MAP-Elites | flat param vectors, `league/members/<i>.pt` | evolved q16 iters, re-scored 1/evolve, reseeded q1000 | yes, 121–125 members |
| **3. Anchors** | **Fixed yardstick + anti-forgetting** | full checkpoint dirs outside rotation, loaded once at boot | none — read-only, never re-scored, never culled | **no** (0 for the whole run) |

Pools 1 and 2 are structurally independent: `trainAgainstOldVersions` is `false` (`LearnerConfig.h:157`) and is never assigned anywhere in the tree, so the version ring is never trained against, and `PolicyVersionManager.cpp` contains no reference to the league. Pool 3 exists because pool 2's fitness is re-scored against a *moving* main, so nothing fixed can survive in it — the anchor vector's exemption from `Cull`/`RefreshStalest`/`TryInsert`/`cellToMember`/`DedupCells` is structural, not an if-check (`LeagueArchive.h:48-56`), and `LeagueConfig.h:49-64` says that exemption "is the entire design."

**Nexto is a fourth source, outside all three**: the highest-priority branch of a mutually-exclusive if/else cascade (`Learner.cpp:1324-1341`), with no BD, no fitness, no archive entry, and its own goal counters.

---

## 2. What it is actually doing right now

### Live archive, iteration 2,046 (`checkpoints_resid/650253312/RUNNING_STATS.json`)

- **121 members = 119 cell elites + 2 exploiters**, 119 distinct cells of 216 → Coverage 0.551.
- **All 121 members are lineage 0.** `next_lineage = 3` — two post-founding reseeds were attempted (iterations 1000 and 2000) and **both were rejected**. `League/Lineage Count` has read 1 for all 2,070 iterations.
- Fitness histogram: `0: 33, +1: 26, +2: 8, +3: 2, -1: 9, ... -17: 2`. Mean drifting negative; `League/Mean Fitness` went +0.41 (iter 881) → 0-crossing at iter ~1415 / 0.418G → −2.19 (iter 1810). `Worst Cell Fitness` −1 (iter 823) → −17 now.
- **PFSP concentration**: 69.4% of all draws land on the 33 members at fitness exactly 0; ESS 59.6 of 121; the two exploiters get **0.39% combined**, ~4× *below* uniform.
- **BD**: `in_air` spans 0.0215 (2.15 pp) across the whole population, `field_y` 0.3248, `boost` 0.0725. Live quantile edge spans (q⅙–q⅚): 0.00896 / 0.11094 / 0.02548. Raw BD variance is 94.4% `field_y`, 5.1% `boost`, 0.5% `in_air`; axes mutually uncorrelated (|r| ≤ 0.042).

### The rollover happened this afternoon

`League/Member Count` peaked at **141 (139 cells, Coverage 0.6435)** and held from ~iter 1735 to 1999. Then, at the `reseedEveryIters`=1000 boundary:

```
1,999  members=141 cells=139
2,000  members=116 cells=114     <- 25 members lost in one iteration
2,016  members=119 cells=117
2,069  members=125 cells=123
```

The iteration-1000 event was the same shape, smaller: 110 → 98. This is `RefreshBinEdges` re-tokenising every member's cell and `DedupCells` deleting the collisions (`LeagueArchive.cpp:219-240`) — see §3.3. Every ≥5-member drop on the sibling run wsd2oclp lands on the same boundaries (999→1000: 111→93; 1999→2000: 128→101; 2999→3000: 99→80; 4006→4007: 65→57).

### Serve budget, measured

| source | configured | realised | cost per iteration |
|---|---|---|---|
| Nexto | 0.15 | **~0.089** | 15.31 s wall vs 5.15 s self-play (2.97×); 96% of it CPU inference |
| old versions | — | **0.000** (dead branch) | — |
| league members | 0.35 (of *all* iterations, per comments) | **~0.319** (= 0.35 × 0.911) | collection 5.57–5.72 s vs 4.19–5.15 s unserved |
| self-play | — | ~0.59 | baseline |

Nexto yardstick, cumulative: **Goals For 14,373 / Against 146,755 = 8.9% goal share** (was 3.9% at 0.5B). No league-member serve counter exists anywhere — `LogMetrics` publishes 18 `League/*` keys and none of them counts a serve (`LeagueArchive.cpp:552-604`); the only serve counters in the tree are `League/Anchor Serves` and `Nexto/Serve Iters`.

### Cost

- **Evolve barrier**: per-iteration wall time by `Total Iterations mod 16` — residue 1 median **9.007 s** vs 5.247 s for the other 15 residues (n≈105 each) → +3.76 s per 16 iterations = **3.55% of run wall clock**, fully serialised with the collect worker joined (`Learner.cpp:1750-1751`).
- **Serving**: served iterations discard half the simulated rows. ~42–43% of iterations are served; mean `Collected Timesteps` 315,549 vs `tsPerItr` 200,000 → ~115,549 rows/iteration (36.6%) discarded at `Learner.cpp:1385-1386`, and `totalTimesteps` counts them (`:2381`).
- **Storage**: `build/checkpoints_resid/league` is **2.7 GB** (20,297,633 B × members) against **329 MB** for an entire checkpoint, fully rewritten on every save (~every 470 s), outside `checkpointsToKeep` rotation.

### Young vs matured

| | 434etlix (0.65B) | wsd2oclp (2.08B, same config) | 17o01ku8 (11.3B) | bfl8mbw4 (57B, 4.0) |
|---|---|---|---|---|
| Member Count | 121–141 | peak 130 → 33 → 48 | peak 94 → **3** | **3** in 93.9% of blocks |
| Cell Count / 216 | 119–139 | peak 128 → 32 | 1 from 4.85B on | **1**; max 7 in a 20.5B window |
| Coverage | 0.55–0.64 | 0.593 → 0.213 | 0.426 → 0.0046 | **0.0046** |
| Mean Fitness | −2.2 | min **−41.8** | min **−42.7** | min −39.6 |
| Worst Cell Fitness | −17 | **−54** | −56 | **−57** |
| Best Exploiter Fitness | 3.0 constant since iter 199 | 2.0 constant | 2.0 constant since iter 257 | **1.0 in all 3000 samples across 57B** |
| Anchors | 0 / 0 | 0 / 0 | 0 / 0 | Count 7→14, **4.8% of iterations** |
| Pairwise Behavioural Distance | 0.072 | — | 0.193 | **0.316** |

The terminal state is **1 cell elite + 2 exploiters**. In that state the elite sits at fitness −11 to −12 while both exploiters are frozen at +1, so `softmax(-|f|)` gives the elite ~1.2e−5 of the draw: **the 0.35 budget becomes two weight-frozen fossils.** Note that `Pairwise Behavioural Distance` *rises* as the archive dies (it is a raw-unit Euclidean over all members including the 2 BD-unconstrained exploiters, `:594-603`) — it is an anti-metric. And league health is *anti*-correlated with skill: corr(`Rating/1v1`, `Cell Count`) = **−0.919** on 17o01ku8, −0.524 on wsd2oclp.

---

## 3. What is broken or vestigial

### 3.1 — CRITICAL, latent, armed right now: the next rebuild silently turns all 121 members into random bots

The live binary is **pre-strip** (built 12:12:48 from `0d64e56`, launched 12:12:51; the strip commits land 13:23–14:52). It builds a 1157-wide policy input and is writing **2,144,346-element** policy vectors into `league/members/*.pt` as we speak. HEAD builds 1152 (`PPOLearner.cpp:91-92`: `fullPolicyConfig.numInputs = fullSharedHeadConfig.layerSizes.back()`), i.e. **2,140,506** — a 3,840-element (= 768×5) surplus in every stored member.

`FromJSON`'s only guard is a tensor **count** check, `mem.params.size() == nModels` → 2 == 2 (`LeagueArchive.cpp:708/717`). Its migration block is the literal self-assignment `mem.params[k] = mem.params[k];` (`:716`) under a comment claiming it migrates. And libtorch's `vector_to_parameters` slices `vec.slice(0, pointer, pointer+num_param)` from a running pointer with **no total-size assertion** — so an *oversized* vector loads with no error (a short one throws).

Measured directly on the real `members/0.pt` reinterpreted against HEAD's shape list: the three LayerNorm gains read **0.0007 / 0.0004 / 0.0007** instead of ~1.0, logit bias 0.0038. A LayerNorm with gain ≈ 0 emits a constant, so the head emits **constant logits — a masked-uniform random policy on a correct trunk**. Exactly the "finite, sane-magnitude weights, behaviourally destroyed" class CLAUDE.md says no structural check can catch.

Then it gets worse. Stored *fitness* is restored from JSON, so PFSP keeps preferring the 33 members at 0. `RefreshStalest` needs ~2,250 iterations (~700M steps, ~4 h) before the telemetry even notices. And `ReseedFromMain` injects a fresh 2,140,506-element member every 1000 iterations; the next `CrossoverDARE` that draws one stale and one fresh parent computes `(a+b)*0.5` on non-broadcastable 1-D tensors and **throws from `league->OnIteration` in the barrier zone on the main thread with no try/catch** (`Learner.cpp:1750-1751`, `:2397-2398`) → process dies → wrapper restarts into identical on-disk state → the 900 s-backoff-forever crash loop.

**Trigger: `trainerctl update`, or any `cmake --build` followed by a crash-restart.** No edit required.

### 3.2 — HIGH, live: the archive is 121 noisy copies of the untrained birth network

`SnapshotMain()` has **exactly one call site** — inside `ReseedFromMain`, on the 1000-iteration cadence (`LeagueArchive.cpp:402`). Its output goes through `TryInsert`, which needs a free cell or a **strictly** greater fitness than the incumbent (`:353`). A fresh main snapshot scores ~0 against itself (`:404`), and 33 incumbents sit at exactly 0 with 36 above — so it loses. Both attempted reseeds were rejected (`next_lineage = 3`, all members lineage 0). Every other generation path mutates or crosses *existing members*.

The founding seed therefore dates from the first `EvolveStep`, ~iteration 16. Measured in weight space (members 0/47/100/133 loaded off disk): each sits **0.039–0.064 relative-L2 from `policy_versions/0`** — the birth network, recorded `skill_ratings.1v1 = −19.166` — and **0.73–0.78 from the current main** (rating 340–392). Mean pairwise distance across 10 sampled members 0.0568; the archive's entire internal spread is **4.9% of its distance to the main**. Weight norms 87.91–88.02 vs policy_versions/0's 87.84 and the main's 111.87.

**Consequence: ~32% of the main's training rollouts — and therefore ~32% of what trains the critic and the V†twins — are collected against noisy copies of the untrained init.** The zero-sum stack makes it worse: competitive margin against a non-participant carries almost no signal about competitive play.

There is a perverse feedback here worth naming: reseeds only succeed when the archive is **sparse**. 17o01ku8's lineage advanced at 77 / 47 / 26 / 14 / 8 / 7 / 4 members; wsd2oclp's first success was at 101 members (its iteration-1000 attempt failed at 111); 434etlix failed at 110 and again at 141. **High Coverage causally predicts a sealed archive.** Anyone reading Coverage as league health is reading it backwards.

*Caveat, flagged honestly:* refreshed members currently score −6 to −17, not the −40 to −57 seen on the mature runs. Either the 1-per-16-iterations refresh lag hasn't caught up, or the weight distance overstates the behavioural gap. See §5.2.

### 3.3 — HIGH: `competenceFloor` never culls anything; the documented collapse mechanism does not exist

`Cull()` (`LeagueArchive.cpp:474-503`) calls `DedupCells()` (`:477`) then `RebuildCellMap()` (`:478`), then builds `protectedIdx` from every entry of `cellToMember` (`:479-480`). Since dedup guarantees one non-exploiter per cell and `CellIndex` always returns ≥ 0, **`protectedIdx` contains every non-exploiter index**. The below-floor loop (`:483-489`) and the `maxMembers` loop (`:491-502`) both skip exploiters *and* protected indices; the latter hits `if (worst < 0) break;` immediately. Neither can ever match anybody. `maxMembers` is unenforceable at any value (and moot: 216 cells + 2 exploiters caps the archive at 218 < 256).

Positive confirmation: cell elites persist far below the −25 floor and are not removed — **2,253 of 3,000 sampled wsd2oclp rows** carry `Worst Cell Fitness` < −25, reaching **−54**; bfl8mbw4 reaches **−57**.

`competenceFloor`'s only live effect is the **admission** gate at `TryInsert:346`. So the actual dynamic is a one-way ratchet: as the main improves, candidates score below −25 and the entrance closes, while `DedupCells` at each `RefreshBinEdges` keeps removing. Measured — after Mean Fitness passes −25, member-count increases:decreases run **13:42** (wsd2oclp) and **7:55** (17o01ku8).

**Consequence:** `docs/LEAGUE_ANCHORS.md`, `src/ExampleMain.cpp:811-817`, `LeagueConfig.h:52-55`, and the comments at `LeagueArchive.cpp:457` and `:475-476` all describe a cull that does not exist — and that wrong story is the premise the entire anchor workstream was built on. Tuning `competenceFloor` or `maxMembers` changes nothing. The live levers are the rebin cadence and the admission gate.

### 3.4 — HIGH: fitness is one noisy integer against a moving target, and PFSP rewards not measuring it

- **One sample, never averaged.** `matchesPerMember = 20` is declared (`LeagueConfig.h:70`) and read nowhere; `Member::matches` is written at `:136/:406/:428/:536` and read nowhere; `cullFrac` likewise. Every insert/cull/serve decision keys off a single 320-arena-second goal differential.
- **Measurement sd ≈ 2.7–3.9 goals.** From 6,233 consecutive re-scores of an unchanged elite on bfl8mbw4's collapsed 1-cell archive: mean drift +0.005, **sd 2.72**; a wider fetch gives 3.38–3.85. `pfspTemp = 1.0` is in the *same units*, so 1σ of pure noise swings a member's serve weight by **e^2.7 ≈ 15×**.
- **Refresh is 1 member per evolve step, round-robin, not staleness-ordered.** `RefreshStalest` returns after the first eval (`:470`) and picks by `refreshCursor % n` (`:461-462`). Full sweep = 121×16 = 1,936 iterations ≈ 590M steps — comparable to the run's entire life. `Member::age` ("drives refresh", `LeagueArchive.h:33`) is written at `:450/:469/:516` and **read nowhere**. `refreshCursor` is not persisted (absent from `ToJSON:632-666`), so it restarts at 0 every boot, and `Cull`/`DedupCells` erases shift indices under it.
- **Measured staleness bias:** 86 of 138 members were byte-identical in *both* `bd` and `fitness` across 175M steps. Those fossils average **+0.30** and take **87.4%** of PFSP draws; the 52 freshly re-scored members average **−6.21** and take 12.6%.
- **Exploiters are a monotone high-water ratchet.** `RefreshStalest` skips them (`:463`, on a comment that is false); `EvolveExploiters` writes fitness only when `q > members[pick].fitness` (`:445`), comparing a fresh score against a stale one; `Cull` can never remove them. `League/Best Exploiter Fitness` = 3.0 constant on 434etlix since iteration 199 (~1,870 iterations, ~570M steps) while Rating went −4 → +392. On bfl8mbw4 it is **1.0 in all 3,000 samples across 57B steps** — i.e. no mutation was ever accepted and both slots are weight-frozen fossils carrying a fictional "level with the main" record. In the collapsed regime those two fossils take ≥99.99% of member draws.
- The same softmax picks **evolution parents** (`:521`, `:525`), so reproduction inherits "be indistinguishable from the main" instead of MAP-Elites' uniform-from-archive draw.

### 3.5 — HIGH, operational: the Nexto dose is a wall-clock artifact, not a probability

`Learner.cpp:1755` creates a **fresh `std::jthread` every iteration**. RocketSim's `RandFloat` uses a `thread_local` `minstd_rand0` seeded `RS_CUR_MS() + hash(thread_id)` (`Math.cpp:54-64`), and `hash(tid)` is constant under glibc stack reuse. The Nexto roll is that engine's **first** draw — a pure sawtooth of the millisecond clock with period (2³¹−1)/16807 = **127.773 s**.

Evidence: phase-folding all 169 serves on 434etlix puts **every one inside a contiguous 8/40 arc**, with 0 serves among the 1,575 iterations outside it (p ~ e⁻¹⁴⁸). Gap histogram: 66–69 gaps of exactly 1 (within-burst), then **nothing until 15**, then 15–21; bursts of length 1–3. Bernoulli(0.15) would put ~90% of gaps below 15.

The *level* deficit is a self-limiting feedback loop nobody had pinned down: the accept window is 19.17 s of **wall clock** per 127.77 s cycle, but the rate is counted per **iteration**, and serving Nexto lengthens the iteration 2.5–3× (12.0–15.3 s vs 5.2–5.3 s), so ~1.72 iterations fit inside the window instead of ~2.89 → predicted 0.090 vs measured 0.0897. It reproduces the cross-run ordering exactly, monotone in mean iteration time:

| run | mean s/iter | realised Nexto rate |
|---|---|---|
| 17o01ku8 (6M net) | 3.61 | 3.64% |
| tpgyf8de | 4.40 | 4.83% |
| bfl8mbw4 | — | ~4.03% (segment-wise) |
| wsd2oclp | 6.14 | 7.05% |
| **434etlix** | **6.62** | **8.9%** |

**Consequence:** the exposure dose is a function of machine speed and net size, not of `serveFrac`. The league's realised share is 0.319, not the 0.35 stated at `ExampleMain.cpp:806` and in CLAUDE.md. The **second** draw (the league roll) advances 16807² mod M = 282,475,249 per ms → wraps every 7.6 ms → genuinely uniform (measured 0.32–0.36). Any new RNG call added at the top of `fnCollectIteration` would shift the draw indices and silently move both.

### 3.6 — MEDIUM-HIGH: the league dir is index-keyed, shared, and already desynced

`leagueDir = checkpointFolder/"league"` (`:47`); weights live at `league/members/<i>.pt` keyed purely by **position in the members vector** (`:650-663`, `:693`); the metadata (bd/fitness/cell/lineage) lives in each *checkpoint's own* `RUNNING_STATS.json`. Positions churn four ways: `Cull` erase (`:485`, `:498`), `DedupCells` compaction (`:253-257`), `TryInsert` within-cell replacement (`:353`), exploiter in-place mutation (`:448-451`).

Live evidence, right now: **141 `.pt` files on disk for a 121-member archive.** 121 files carry mtime 15:49:10–11 (the newest save); **20 carry 15:41:45** — orphans from the pre-drop 141-member archive that nothing will ever clean up or reconcile. Measured earlier: restoring one save back mispairs **28 of 134** members; the oldest dir in rotation mispairs **44 of 115**; between 450174976 and 625233920, 41 of 130 common indices changed BD.

**Consequence:** the recovery doctrine assumes a checkpoint dir is self-contained. It is not — restoring *any* non-latest checkpoint (numbered, `best_r*`, or a backup without its own `league/`) pairs stale descriptors with whatever weights the newest save left in those slots, with no checksum, timestep or shape tag to catch it. Blast radius is the sparring pool, not the trained policy, and `RefreshStalest` self-corrects over ~2,000 iterations — silently.

### 3.7 — MEDIUM: anchors are inert on this lineage and cannot self-heal

`anchorFrac = 0.05` is set (`ExampleMain.cpp:829`), but `anchorDir` derives to `checkpoints_resid_anchors` (`LeagueArchive.cpp:53-56`), which does not exist → `LoadAnchors` early-returns with "anchors DISABLED" (`train-20260725-121250.log:54`), and the serve branch short-circuits on `!anchors.empty()` (`:380`), falling through to an ordinary PFSP draw. `League/Anchor Count` and `Anchor Serves` read **0 for every iteration** of 434etlix, wsd2oclp, tpgyf8de and 17o01ku8.

This is *not* a broken feature — on bfl8mbw4 it worked exactly to spec: Anchor Count 7→14, serves **4.854%** of iterations against a prediction of (1−0.0403)×0.05 = 4.80%. Two independent blockers keep it dark here:

1. **Archival is pointed at the dead lineage.** `tools/archive_anchor.sh:26` still defaults `CKPT_DIR` to `build/checkpoints_5.0v3` while `tools/trainerctl:30` was migrated to `checkpoints_resid`, and `pulsar-anchor.service` sets no `Environment=`. The timer is enabled and firing every 30 min and has logged `skip` every time since the cold start ("newest checkpoint 56975427085 is only 0 steps past anchor 56975427085"). Last real archive: **2026-07-22 20:59:53**.
2. **The 17 existing anchors are shape-incompatible.** 5.0v3: `POLICY.lt` 3.37 MB, first layer 512×517. Live: 8.59 MB, 768×1157. The zero-pad migration went with the wire on 2026-07-25, and `ExampleMain.cpp:822-823` records that archived anchors must now match the current net width. They would be caught and skipped at `:126-131`.

**Consequence:** no permanent spaced history is accumulating for this lineage — `checkpointsToKeep=8` is deleting it as we go (the run's first ~350M steps are gone), and the offline `anchor_battery.py` yardstick has no resid opponents *and* can't load resid checkpoints anyway (`load_checkpoint.py:57-64`, `:216` hardcode a 512-wide 2-layer trunk).

Also worth knowing before reusing that design: **the anchor intervention was never judged.** `research/results/anchor_battery_history.jsonl` has exactly one line — the 2026-07-19 pre-change baseline. The anchor era on bfl8mbw4 ran 28.7B steps, 3.3× the baseline window, so there was ample opportunity. The one diagnostic that *was* watched failed: Member Count == 3 in **90.4%** of the anchor era, against the 92% floor occupancy cited as the evidence of collapse. The pre-registered guard is also gone (`Learner.cpp:1340` passes a literal `true`; the latch was removed 2026-07-25), and the secondary criterion's "held-out anchor subset excluded from league serving" has **no implementation anywhere** (repo-wide grep: 0 hits).

### 3.8 — MEDIUM: the descriptor basis

- **`gridAxes` is decorative.** Only `.size()` is read (`:220`, `:329`, `:555`); the string is used once, to build the wandb key at `:565`. The BD is a hardcoded positional triple at `:306-327`. Adding a 4th name appends a constant column → `totalCells` 216→1296 and Coverage silently divides by 6; dropping to 2 truncates the boost axis; changing the count across a resume mixes indexing schemes, because `FromJSON` re-tokenises via `CellIndex` looping over `bd.size()` (`:195`, `:727-728`).
- **`field_y` takes `abs()`** (`:311`), folding "camps in own net" and "camps in opponent net" onto one coordinate — gratuitously, since `aTeam` is hardcoded BLUE (`:268`) and BLUE attacks +y, so signed y is already canonical.
- **`League/Coverage` is arithmetically the elite count.** `TryInsert` enforces one member per cell, so distinct-cells ≡ elite-count. Verified with **zero violations in 20,793 sampled rows across four runs**. It carries no diversity information at any maturity.
- **`Exploiter Unmapped Wins`** — the one panel that would convict a missing axis — requires `q > 0`, so it freezes the moment the main outgrows the archive: stuck at **33 since iteration 1351** on 434etlix (486 iterations, ~121 further candidates, p≈4e−6 under the prior rate); stuck at 22 for 6,403 iterations / 1.83G on wsd2oclp.
- **The coordinate system is frozen for long stretches.** The bootstrap edges were fitted to **33 BD samples at iteration 82** and governed the grid for 918 iterations / 258M steps / 1.71 h. Only two refreshes have ever fired. `ExampleMain.cpp:797-798`'s "track it as the bot improves all week" is false.
- **Arenas are never reset between member evaluations.** `EnvSet::Reset()` resets only arenas already flagged terminal (`EnvSet.cpp:334-340`), and `GoalScoreCondition` is the only terminal with no timeout. At ~0.9 goals/arena/window, the expected number of arenas terminal at eval end is ~0.05 of 16 — so a member typically plays its **entire** 300-step window inside an episode its predecessor started. Measured symptom: league boost BD median 0.029 (2.9 boost) vs training `Player/Boost` median 7.9–11.3. Since team A is always BLUE, an inherited BLUE-favourable state always favours the member under test.
- **The eval is always 1v1**, even after PHASE B — `matchEnv` clones the training create-func with `numArenas=16` and team arenas sit at the end of the index range (`:24-26`, `ExampleMain.cpp:41-45`).

### 3.9 — MEDIUM: episode boundaries nobody registered

On a served iteration the opponent half's in-flight partial episodes are `Clear()`ed, not finalised (`Learner.cpp:1385-1386`), and `numRealPlayers` halves (`:1368`). Mean episode length is 687 steps against ~189 steps of episode-time per iteration, so **essentially every trained episode spans two or more different opponents**, with no arena reset at the switch. CLAUDE.md's episode-boundary rule ("whoever changes episode boundaries must let the critic learn the new return structure") was paid for in two Elo collapses; this is a boundary change that was never registered as one. Head-truncation itself is safe — GAE accumulates strictly backwards (`GAE.cpp:45`) — but the opponent-identity discontinuity inside an episode is not the same thing.

### 3.10 — LOW: dead and drifted surface

- Never read: `Member::matches`, `Member::age`, `LeagueConfig::cullFrac`, `LeagueConfig::matchesPerMember`, `LeagueArchive::GetMember()`.
- `LeagueArchive.cpp:716`: `mem.params[k] = mem.params[k];` under a comment claiming it migrates stale-shape vectors.
- Three comments in the serving path describe the rating latch and the impossible-arena skip, both deleted 2026-07-25: `Learner.cpp:1326`, `:1337-1339` (above a literal `true`), `:1580` (above an unfiltered loop at `:1585-1596`), plus `LeagueArchive.h:80-81`.
- `docs/LEAGUE_ANCHORS.md:40,42,78,128-130` still specifies `MigrateFlatVec`/`allowInputExpand`, deleted by `6953718` — the same commit that moved the doc without editing a byte.
- **`nexto_goals_for/against/serve_iters` are written by `SaveStats` (`Learner.cpp:340-342`) under a comment saying they "must survive restarts," but `4f25b1c` deleted the matching `LoadStats` read as collateral inside a steering-removal hunk.** The running binary predates it, so today's series is intact; the next rebuild+restart zeroes the panels. wandb history survives (`run_id` is persisted, `:370` → `:288`), so first-differences survive everywhere except the restart interval.
- Nexto's per-episode prev-action reset is dead: `prevArenaTerminals` is `envSet->state.terminals`, unconditionally zeroed by `Reset()` at `Learner.cpp:1415` before `nexto->Act` at `:1504`, and the only writer is `StepSecondHalf` at `:1531`. One stale action frame per episode start — 0.2–1.5% of Nexto frames, and `BeginServe` already zeroes all 2048 slots mid-episode at every serve, which is 3–25× larger and deliberate.
- **The live binary is not what `ExampleMain.cpp` at HEAD says.** It still prints `Ladder/*` panels, runs RND (`RND/Injected Abs Mean` 0.0221), the miner, the 30% practice/control arena split and 8 impossible-control arenas. For the league specifically this doesn't matter — `cfg.league.*` and the opponent cascade are byte-identical between `0d64e56` and HEAD. (And the live Nexto counters are **not** contaminated by impossible arenas: `5e6c661` added the filter at 11:54:57, before the 12:12:48 build.)

---

## 4. The structural question for folding Nexto in

**What fits.** Nexto is already a first-class opponent source with a bespoke serve branch (`Learner.cpp:1324-1329` roll, `:1497-1506` `Act`, `:1587-1596` goal counting). Structurally it is *exactly what an anchor is*: a frozen, fitness-exempt, cull-exempt, serve-only opponent. `anchors` is already a separate `std::vector<Member>` outside `members`, exempt from `Cull`/`RefreshStalest`/`TryInsert`/`cellToMember`/`DedupCells` **by construction** (`LeagueArchive.h:48-56`). That is the only slot shape in the archive that can hold a thing like Nexto. The observation that keeps recurring across every reader: **anchors and Nexto are the same category implemented as two independent budgets rolled in sequence.**

**What does not fit.**

1. **Representation.** A member is two flat CPU tensors matching the live net's `shared_head`+`policy` shapes, moved via `parameters_to_vector`/`vector_to_parameters` (`:17`, `:171-191`) — 5,073,882 floats / 20.3 MB. Nexto is a **444,032-param opaque TorchScript graph** (1.85 MB) that reads `GameState` directly through its own 37×24 entity obs + 32-dim query, argmaxes over its own 90-row lookup remapped to `DefaultAction` by checked tuple equality, keeps its own prev-action memory, and ignores our action masks. There is no param vector.
2. **Operators.** `MutateGaussian`/`CrossoverDARE` are elementwise on identically-shaped flat tensors (`Operators.h:12-26`). Nothing to mutate, nothing to cross with, and mutating a frozen external reference is meaningless anyway.
3. **The fitness scale actively ejects it.** Fitness is `memberGoals − mainGoals` against the *current* main, so any fixed opponent's fitness necessarily drifts as the main improves — which for Nexto **is** the signal. But PFSP is `softmax(-|fitness|/1.0)`, maximised at 0. Nexto currently takes 91% of the goals against us; a fitness of +20 to +40 gives it weight e⁻²⁰…e⁻⁴⁰ ≈ 0. **Putting Nexto under PFSP makes it stop serving.** The same softmax is why the two exploiters at +3/+2 already get 0.39% of draws.
4. **Cost cannot be priced in one budget.** A Nexto iteration is 2.97× a self-play iteration (15.31 s vs 5.15 s median), 96% of it CPU inference (13.43 s of 14.03 s) because the CUDA probe throws at every boot and it runs under `at::set_num_threads(2)` (`Learner.cpp:185`, `NextoOpponent.cpp:41,57-91`). A league member serves on GPU at +11%. At 8.1% of iterations Nexto already costs **14.6–20.8% of wall clock**; repairing the dose to the configured 15% roughly doubles that.
5. **The two budgets already interact silently.** Nexto pre-empts the league roll, so raising `serveFrac` shrinks the league share *and* the anchor slice with it — and the anchor conversion at `:381-382` (`pAnchor = anchorFrac / descendOpponentFrac`) assumes `anchorFrac` is a share of *all* iterations, which it isn't. It happens to be self-consistent (both sub-draws scale by the same factor; measured 4.854% against a predicted 4.80% on bfl8mbw4), but only by accident.

**What is genuinely available.** The BD axes are model-agnostic — `EvaluateMember` computes them from `GameState` (`:306-320`); the *only* ModelSet-shaped thing in that function is the actor call at `:290-295`. So Nexto could be given a BD and a fitness on the same scale with one generalisation of that call. The obstacle is not measurement, it is that a frozen entry occupying a cell either blocks it (one-elite-per-cell) or needs the anchor-style exemption anyway — which lands back on "one unified frozen-external-opponent vector with an opaque serve handle," and the archive's flat-param `Member` is what currently forbids that.

**One thing that must follow it.** Nexto's `Goals For/Against` counters are the run's only pool-inflation-proof yardstick *in the trainer*. Anchor games today produce **no outcome telemetry at all** — `:576-583` publishes Count / Serves / Span Steps / Oldest Ts and nothing else, eight lines away from the goal accounting that exists for Nexto. Folding Nexto into a serve budget without carrying the counters loses the yardstick.

---

## 5. What we do not know

**5.1 — The per-evaluation noise floor, on any axis or on fitness.** This is the single biggest hole: *everything* in the archive keys off statistics whose variance has never been measured. Fitness is the best-established — **sd 2.72 goals** from 6,233 consecutive re-scores of an unchanged elite on bfl8mbw4's collapsed 1-cell archive (mean drift +0.005; a wider fetch gives 3.38–3.85). The BD axes are contested: an iid-binomial *lower bound* on `in_air` gives SE 0.0039 against a between-member sd of 0.0045 (≥74% noise), while a direct test-retest via LCS-aligned `RefreshStalest` re-evaluations (n=8 pairs) gives SE 0.00133 (≈9% noise) and instead convicts **`field_y`** (SE 0.0498 vs sd 0.0606, 68% noise). Those disagree by 3× in opposite directions, n=8 is tiny, and the arena carry-over (§3.8) correlates consecutive evals. **Settling it costs ~20 lines offline:** replay one member's fixed weights through `EvaluateMember` N times and read the spread on all four statistics. It decides how many bins any axis can honestly support, whether `DedupCells`' survival rule and the exploiter accept test measure anything, and what `pfspTemp` should even be in.

**5.2 — Whether the archive is genuinely birth-network-grade.** Weight space says yes emphatically (§3.2). Fitness telemetry says refreshed members lose by only 6–17 goals, not the 40–57 seen on the mature runs. **Settling it:** play one archive member head-to-head against the current main offline for ~100 episodes and read the goal share directly. This is the number that decides whether the league is merely inefficient or actively feeding the policy garbage.

**5.3 — What actually causes the terminal 1-cell state.** Two mechanisms are both consistent with bfl8mbw4's Cell Count 1: (a) `DedupCells` under quantile edges that have zoomed onto a converged population, and (b) `CellIndex`'s `upper_bound` mapping everything into one bin once edges tie on near-zero BD variance (the case `LeagueConfig.h:29` acknowledges). We now know it is *not* the documented `competenceFloor` cull. It matters because (a) points at the rebin cadence and (b) points at the descriptor basis. **Settling it:** replay a mature checkpoint's stored member BDs through `CellIndex` with that checkpoint's stored `bin_edges`.

**5.4 — The candidate collision rate.** Only `Exploiter Unmapped Wins` (q>0 **and** occupied) is logged, and it freezes. The plain fraction of candidates rejected for landing in an occupied cell would say directly whether the grid is too coarse or too fine. Not instrumented.

**5.5 — The fitness scale's saturation point.** `aGoals` and `bGoals` are never logged separately (`:315-319`, `:332`), so a differential of −17 cannot be converted to a win rate and we cannot say whether −17 and −42 are meaningfully different or both mean "shut out." Relevant to every threshold in the file.

**5.6 — What `isOnGround` counts.** The field is correctly plumbed through the v3 FFI (`rocketsim_ffi/src/lib.rs:169` → `compat/RocketSimCompat.cpp:79`), but the semantics in the Rust core were not traced. If it counts wall/ceiling contact, `in_air` conflates wall-riding with genuine aerials — at a population value of 0.92, that decides what the axis even names.

**5.7 — Whether the league's ~32% of iterations does anything measurable.** There is no league win/loss panel, no member-serve counter, and no ablation anywhere in the record. The only outcome telemetry for any non-self opponent is Nexto's. The intervention's cost is well measured (~3.6% barrier + a serving tax between 3% and 11% depending on which denominator you trust, plus 2.7 GB per save); its *value* has never been measured once, in any run, at any maturity. By the project's own doctrine that is the state a lever should not be in.

**5.8 — The end-to-end wall-clock cost of serving.** The evolve barrier is solid (residue-1 median 9.007 s vs 5.247 s baseline, n≈105 each). The serving cost is not: median collection time gives +3–7% at a 32% share, while an SPS-based read gives ~11%, and the discrepancy is that `Collected Timesteps` counts in-flight rows that never enter the learn buffer (`Learner.cpp:1410`, `:2377`). **Settling it:** sum `max(0, Collection − Consumption)` by opponent class, as was done for Nexto (1,838 s of 12,291 s = 15.0%).

**5.9 — Whether 434etlix follows wsd2oclp's decay curve.** Partly answered in the last hour: the peak came at 141 members / 139 cells around 0.62B and the iteration-2000 rebin took 25 members in one step. wsd2oclp peaked at 128 / 0.563G, halved by 1.20G and quartered by 1.79G. The next two rebin events (iterations 3000 and 4000) will tell us whether this is the same curve — free to observe, no intervention needed.
---

## VERDICT — 2026-07-25: strip it

**Decision (user):** remove the league entirely. Train against a few archived past selves plus
Nexto; fix Rating in the same pass; no ablation. Deploy on the cold start the composition-critic
conformance pass already requires — the 700M-step lineage this recon was measured on is superseded.

The reasoning that carried it, from the sections above: the archive **collapses on every lineage
that learns**, and the documented cause (`competenceFloor`) turned out to be unreachable code, so
there was no parameter to tune. What it actually served were **noisy copies of the untrained birth
network** — 0.039–0.064 relative-L2 from `policy_versions/0` against 0.73–0.78 from the current
main. And the asymmetry that settled it: **its cost is well measured and its value has never been
measured once**, in any run, at any maturity. By the project's own doctrine that is not a state a
lever gets to stay in.

Two things were kept rather than deleted:

- **The log-spaced decimation** (`GigaLearnCPP/src/public/GigaLearnCPP/Util/LogSpaced.h`), salvaged
  from `LeagueArchive::DecimateSpaced`. Its rationale — that FIFO eviction recreates the very pool
  myopia a permanent set exists to cure — is what makes `Ref/Oldest Share` a fixed yardstick.
- **The anchor design's load-bearing idea**: a separate vector that nothing re-scores, re-rates or
  culls, so the exemption is structural rather than a scattering of if-checks. That is now the
  reference set. `LEAGUE_ANCHORS.md` is superseded, but this part of it was right.

What replaced it: `trainAgainstOldVersions` (uniform draw from the 32-version ring, realized 0.255
of iterations) for training diversity, and the permanent reference set for honest measurement.
Restore point: tag `pre-league-strip-20260725`.

**Findings from this recon that outlived the league** and were fixed in the same pass:

- The Nexto serve dose was a **127.773 s wall-clock sawtooth**, not a probability (§3.5). The
  opponent roll now uses a persistent `std::mt19937_64` instead of RocketSim's clock-reseeded
  `thread_local` engine.
- The Nexto goal counters **did not survive restarts** (§3.10) — `SaveStats` wrote them, the
  matching `LoadStats` reads had been deleted as collateral in `4f25b1c`. Restored.
- `Rating/1v1`'s inflation mechanism (§3.4 and the recon's §2) is now written down where it is
  read, in `SkillTrackerConfig.h` and `ExampleMain.cpp`, rather than only in this document.
