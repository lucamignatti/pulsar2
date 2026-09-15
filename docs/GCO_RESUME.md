# Pulsar GCO resume — 235.7B → variable tickskip + league, 12 nodes

*Planning doc, 2026-09-01. Status: PROPOSED — nothing executed.*
*Companion to `~/Desktop/pulsar2-gco-FULL-235.7B/{README,GCO_CONFIG}.md`, which are the
authoritative description of the asset. This doc is only about **resuming** it.*

Titan is a separate project on `private-titan`. **Nothing from titan is ported here** —
not TitanObs, not SimBa v2, not entity pooling, not Stage 6 rewards.

---

## 1. The asset

`~/Desktop/pulsar2-gco-FULL-235.7B/checkpoint/235726848000` — 22 files, **all md5-verified
against the bundle manifest**. This is a full resumable checkpoint: every model plus every
`*_OPTIM.lt`.

| | |
|---|---|
| Run | `739468ux`, 294,000 iterations, 235,726,848,000 steps |
| Trained at | commit **`d372ae7`**, branch `private` |
| Reward | `GGL_GCO=1` — goal/concede only, exactly ±150, nothing else |
| Cadence | **tickSkip 8 / actionDelay 0** (15 Hz) — *not stored in the checkpoint* |
| Discounts | `gaeGamma` 0.9969, `gaeLambda` 0.95, `goalCritic.gamma` 0.9994 |
| Optim | Muon, LR 1.5e-4 policy+critic, 2 epochs, entropyScale 0.14 |
| Obs / actions | AdvancedObsPadded **230**-dim raw, team-canonical / **90** discrete masked |
| SIL | on, coeff 0.05 (headroom-gated) |
| Seek betas | **both 0** — vdag twins are trained but do **not** inject |

Architecture (verified by dumping the tensors, not by reading config):

```
SHARED_HEAD    230 → 1280, ×3 @1280      (residual, LN, LeakyReLU)
POLICY        1280 → 768 ×3 → 90
CRITIC_TRUNK  1280 → 1536 ×3
CRITIC/CRITIC2 1536 → 1536 ×2 → 1        (twin, readout = mean)
VDAG1/VDAG2   1536 → 1536 ×2 → 1
GOAL_CRITIC   1536 → 1536 ×2 → 1
OPP_EMBED        4 → 32 → 1536
AUX_DISP      1280 → 256 → 460
GAP_EXP       (V_exp expectile twin, measurement only)
```

Absent by design: **no `REACH_*`, no `HULL_*`, no `LEAGUE.lt`.** This matters — see §5.

Measured strength on this lineage: **42–0 real-game qualifier vs Nexto** (first attempt, @99B);
85.3% head-to-head goal share vs the reward-shaped control from the same codebase (@198.8B).

---

## 2. Target

Local branch **`vts-league-push`** = `private` + one commit `cba05eb` *"League rescue +
variable tickskip (450-way joint head)"*. That single commit is the entire backport surface.

---

## 3. The four deltas, in increasing risk order

### D1 — code drift: `d372ae7` → HEAD is ~43 commits

Triaged: **6 SEMANTIC-ENV, 6 SEMANTIC-OBS, 1 SEMANTIC-LEARN (default-off), 30 INFRA.**

**The dating is the whole analysis.** Checkpoint files are stamped **2026-08-23 18:11**.
Every engine and obs commit in the range is 2026-08-25 or later — so the policy trained
*before all of them*. "Match training semantics" therefore means **off for everything**,
which is not what a naive reading of the flags suggests.

| commit | date | what | HEAD default | matches training? |
|---|---|---|---|---|
| `6898092` RocketSimV3 tuned engine import | 08-25 00:02 | wholesale engine swap | ON | ✗ **not revertible** |
| `1af6056` titan-appo physics ports | 08-26 17:18 | collision/ERP/SAT/manifolds; shipped dodge torque ON | ON | ✗ **not revertible** |
| `92fdf21` 1-tick level-takeoff contact hold | 08-26 17:29 | dodge press gate | **ON** | ✗ → `GGL_DODGE_GROUND_HOLD=0` |
| `0a8bc09` flip-reset takeoff guard | 08-26 17:13 | flip-reset hole | **ON** | ✗ → `GGL_FLIP_RESET_FIX=0` |
| `73102e0` dodge-torque-on-start-tick | 08-26 19:27 | flipped `1af6056`'s ON back to OFF | **OFF** | ✓ **leave unset** |
| `8fa49aa`→`c0c5722` obs flag noise | 08-26 | introduced *and* retired, both post-checkpoint | **OFF** | ✓ **leave unset** |
| `17e505a` touchdown exception ticks | 08-25 16:31 | 5 → 0 | verify | ⚠ **unverified** |

> **Correction, recorded because it nearly shipped.** An automated triage recommended
> `GGL_DODGE_TORQUE_START=1` on the reasoning that the policy "trained during `1af6056`'s
> deployment with torque ON". `1af6056` post-dates the checkpoint by three days. Setting
> that flag would inject a dodge force this policy has never experienced. **Leave it unset.**
> Any claim of the form "the policy trained with X" must be checked against the checkpoint's
> 2026-08-23 timestamp, not against commit ordering on the branch.

**The non-revertible part, and what it actually costs.** `6898092` + `1af6056` moved the
physics baseline wholesale; no flag undoes them. Measured on a *frozen* 528B policy with
only the engine differing (vs BonkDaddy):

| engine | goal share |
|---|---|
| pre-import | 34.9% |
| post-import, dodge torque ON (`1af6056` as shipped) | 10.7% |
| post-import, dodge torque OFF (**= HEAD default**) | 30.8% |

The widely-quoted "3× swing" is pre-import vs post-import-**with-torque-on**. HEAD's default
already recovers most of it: **34.9% → 30.8%, ≈12% relative**, not 3×.

**Recommendation: keep the corrected engine. Do not revert.** The gate-v3 / sim2real work was
validated against *real game traces* (`WAVEDASH_GATE.md`: real 42.3% vs sim 41.3–42.3%;
speed_flip divergence 1127 → 24 uu held-out). This lineage's headline result is a **42–0
real-game qualifier** — the entire point is real-match deployment, so training against the
*more accurate* physics is aligned with the objective. Reverting would optimize the policy
against a physics model we have measured to be wrong.

Use the compat block as a **P0 diagnostic, not a production setting**: run both arms briefly
to measure the real gap on *this* policy (the 30.8/34.9 numbers are from the 528B policy, not
this one), then proceed on HEAD defaults and let the ~12% adapt out.

```bash
# P0 DIAGNOSTIC ARM ONLY — measures the engine delta on this policy. Not for production.
export GGL_DODGE_GROUND_HOLD=0      # 92fdf21, post-dates the checkpoint
export GGL_FLIP_RESET_FIX=0         # 0a8bc09, post-dates the checkpoint
# GGL_DODGE_TORQUE_START — LEAVE UNSET. HEAD default (off) already matches training.
# GGL_OBS_FLAG_NOISE     — LEAVE UNSET. HEAD default (off) already matches training.
```

Still to verify: `17e505a`'s touchdown-exception tick count (5 → 0) and which value the
checkpoint trained under.

This risk is **independent of the surgery** — it applies to a plain resume with no changes at
all, and must be cleared in P0 before anything else moves.

### D2 — 12 nodes vs the 32 it was tuned at

`GCO_CONFIG.md` is explicit: *"The LR was tuned at this effective batch — at much smaller
batches, LR is the first knob to suspect."*

| | 32 nodes (as trained) | 12 nodes (naive) |
|---|---|---|
| ranks | 192 | 72 |
| `GGL_MINIBATCH` /rank | 1044 | 1044 |
| **effective batch** | **200,448** | **75,168** (2.67× smaller) |

Two ways to close it:

- **Preferred — hold the effective batch.** `GGL_MINIBATCH=2784`, `GGL_TS_PER_ITR=11136`,
  which preserves both the 200,448 effective batch and the 8 optimizer updates/iter. Costs
  2.67× more rows per rank → `GGL_NUM_GAMES` 87 → ~232. **Check this fits in 32GB V100
  memory before committing**; if it doesn't, arena count is the thing to trade.
- **Fallback — scale the LR down** by the batch ratio and record it as a deviation.

Do not silently run at 75k effective batch with the 200k-tuned LR.

### D3 — head surgery, 90 → 450

The only shape-breaking change. Convention confirmed in `TickSkipActionParser.h`:
`action = skipIdx*90 + ctrl`, softmax **joint** over all 450, action mask tiled per bucket
(so the skip choice is never itself masked).

**Construction — exactly function-preserving:**

```
for b in 0..4:
    W_new[b*90 : (b+1)*90, :] = W_old            # tile the trained control head
    b_new[b*90 : (b+1)*90]    = b_old + (C if b == b* else 0)
```

A constant added within a bucket **cancels in the within-bucket softmax**, so
`P(ctrl | b*)` is *exactly* the old distribution, while skip mass concentrates on `b*`.
At `C = 8`, `P(b*) ≈ 0.9987`. Then anneal `C → 0` to open skip exploration.

Settings, per decision to keep the current cadence:

- buckets `[1,2,4,8,16]` in **env steps**; base tickSkip stays **8**
- `b* = 0` (1 env step = 8 ticks = the current 15 Hz cadence)
- **`gaeGamma` and `goalCritic.gamma` are UNCHANGED** — base tickSkip is unchanged and the
  SMDP fold in `GAE::ComputeDecision` handles hold lengths. (Re-derivation would only be
  required if the *base* tickSkip moved.)

**`POLICY_OPTIM.lt` must be handled too.** Muon state for the final layer is `(90,768)`;
it has to be tiled the same way or reset for that layer alone. Everything else keeps
`GGL_FRESH_OPTIM=0`.

**Failure mode is loud, but only for shapes.** `Models.cpp` compares layer sizes
before/after load and `RG_ERR_CLOSE`s on mismatch — so a wrong *shape* crashes immediately.
A wrong *tile* (right shape, scrambled semantics) will load clean and play broken. Hence
the offline equivalence test in P2.

> **Caveat worth stating once.** `DYNAMIC_TS_PROBE.md` measured 8.9× decision compression —
> but on a **ts1** (120 Hz) policy. ts8 is *already* 8× compressed, so the coarsening
> direction has little measured headroom left; the probe's real prize was the *finer*
> direction (its "15% of steps need every tick"), which a base-ts8 config forecloses.
> Expect VTS here to be roughly throughput-neutral and to be judged on whether long holds
> hurt. If it reads neutral, base tickSkip 2 with `b*` = the 4-env-step bucket is the
> fallback that opens finer control — at the cost of re-deriving both gammas.

### D4 — the league

Good news: this checkpoint has **no `LEAGUE.lt`**, so there is no stale-shape hazard. (The
*ts1* lineage's league would have been poison — its `polA6` adapter is `(10,4,90)`, keyed to
the old head width.) Adapters are shape-keyed to the policy head, so the league must be
built **after** the surgery, never before.

Preconditions from `LEAGUE_LORA.md`: it was solved on a **converged** 442B policy, and it
degrades on cold Adam state. 235.7B GCO qualifies as crystallized.

Config (the solved defaults): `GGL_LEAGUE=1`, `REPEL_COEFF 0.5`, `REPEL_TARGET 0.28`,
`RANK 4`, `DIVERSE 8`, `EXPLOITERS 2`, `ADAPTER_LR 2e-5`, `FRAC 0.25`, `LEAGUE_MINIBATCH`
scaled down with rank. **Do not set `GGL_MAIN_LR` in production.**

One inherited trap: served opponents must match the main's **execution semantics**, not just
its weights — θ-commit parity was the final killer that took variant win share 0.02 → 0.46.
With VTS on, main and variants share the 450-way head, so parity holds by construction.
Verify it rather than assuming.

---

## 4. Phase plan

Each phase has a gate and a revert path. The 235.7B bundle is the restore point for all of
them; per the recovery doctrine it is **full-checkpoint-or-fresh**, never a partial.

| P | What | Gate | Revert |
|---|---|---|---|
| **P0** | Plain resume on HEAD + D1 compat block. No VTS, no league, no surgery. | boot log has **zero** unexpected `"will be reset"` lines; goal rate & episode length match the lineage within noise; offline kickoff 10/10 touches | nothing mutated |
| **P1** | Batch/LR calibration for 12 nodes (D2) | same gates, stable over ≥2 save intervals | back to P0 settings |
| **P2** | Head surgery + VTS, `C` large | **offline equivalence test first** (below); then first-iteration KL(new‖old) ≈ 0 and goal rate unchanged | reload 235.7B bundle |
| **P3** | Anneal `C → 0` | skip histogram moves off `b*` *without* goal-rate regression | freeze `C` at last good value |
| **P4** | League on, fresh folder | `LEAGUE_LORA` bars: κ ≥ 0.286, variant goal share ≥ 0.40, exploiters > 0.55 | `GGL_LEAGUE=0` |
| **P5** | *(optional, separate experiment)* `vdagSeekBeta` 0 → 0.30 | pre-register separately | set back to 0 |

**P5 is not a restore, it is a new lever.** GCO trained with both seek betas at 0, so
turning injection on is a change this lineage has never seen. It deserves its own
pre-registration, not a slot in a migration plan.

**The offline equivalence test (blocking for P2).** Before any cluster time: rebuild both
the 90-way and 450-way policies in Python, feed identical observations, and assert
`max |Δ log P(ctrl)| < 1e-5` over the control marginal. This is the only check that catches
a correctly-shaped but wrongly-tiled head, and it costs minutes.

---

## 5. Known traps, inherited (all verified empirically on this lineage)

1. **Missing-model reset is SILENT.** The loader prints one warning per absent model and
   continues with random init. With default config that is 6 models (reach×4, hull×2)
   fresh-initialized and co-training into a converged trunk. `GGL_NO_REACH=1` and
   `GGL_HULL=0` are **required**, not optional. Grep the first boot log every single time.
2. **tickSkip is not stored in the checkpoint.** A wrong decision rate loads clean and plays
   broken. This checkpoint is 15 Hz.
3. **Residual spans and the trailing activation are not recoverable from archives.** A
   non-residual rebuild has correct shapes and garbage behavior.
4. **`GoalCritic/Adv-Outcome Corr` near zero or negative is NORMAL here.** This lineage read
   −0.2..−0.5 at peak strength. Do not debug it on a resumed converged checkpoint.
5. **Mean step reward is identically 0 under GCO** (zero-sum). Read goal rate and benchmarks;
   never mean reward.
6. **mpirun `-x` forwarding.** Any new `GGL_*` must appear both as an export *and* in the
   `-x` list, or it silently never reaches the ranks. Confirm via the boot banner.

The six semantic env vars, non-negotiable:

```
GGL_GCO=1  GGL_NO_REACH=1  GGL_HULL=0
GGL_NO_VERSIONS=1  GGL_NEXTO_SERVE_FRAC=0  GGL_TRAIN_AGAINST_OLD_CHANCE=0
```

---

## 6. Code that has to be written

| # | Item | Notes |
|---|---|---|
| 1 | Checkpoint surgery tool: tile POLICY final layer 90→450, apply the `b*` bias offset, handle `POLICY_OPTIM` | Precedent exists — `RunExpandCheckpoint()` in `NextoEval.cpp` (`GGL_EXPAND_K/IN/OUT`) is the same shape of tool with a different transform |
| 2 | Offline equivalence test (Python) | Blocking for P2 |
| 3 | `C` anneal schedule | **No LR/schedule infrastructure exists.** `SetLearningRates(policyLR, criticLR)` is call-per-iteration only; freezing is lr=0 (an exact no-op under Muon) |
| 4 | `vdagSeekBeta` ramp | P5 only. Config-only field, **no env var, no ramp** — needs a small code change |
| 5 | 12-node sbatch derived from `aimos_launch.sbatch` | Keep the `pkill` preamble; update the `-x` list |

---

## 6b. EXECUTED 2026-09-01 — build fixed, surgery done, config found by smoke

Worktree `/home/luca/Projects/pulsar2-gco-resume` on `vts-league-push`, commit `7ec63f5`.
Nothing deployed; the cluster queue is still empty and the Desktop bundle is untouched.

**`vts-league-push` did not compile.** Two breaks, both because `cba05eb` is a partial
cherry-pick of titan-appo:

1. `LearnerAsync.cpp` referenced `Dist::Session::TAG_LEAGUE_ADAPTERS` at three sites, but
   **no branch has ever defined it** — `private`, `private-titan`, `titan-appo` and
   `vts-league-titan` all stop at `TAG_EXIT_ACK = 103`. The league adapter transport had
   never been compiled anywhere. **This blocked the titan run too.** Added `= 104`.
2. ExampleMain's `GGL_NEXTO_EVAL_FRAC` block arrived without its two `SkillTrackerConfig`
   fields *or* the `PolicyVersionManager` slice that reads them. Removed it — adding the
   fields alone compiles but leaves a flag that silently does nothing when set, and porting
   the reader is a 257-line unreviewed merge into the training loop.

**tickSkip had to be re-derived in source.** The branch carries ts2 (60 Hz) constants; this
checkpoint is ts8 (15 Hz), and tickSkip is not stored in a checkpoint. `GGL_TICK_SKIP` is the
wrong tool (it deliberately does not touch the gammas, and is lost at the next chain hop), so
`TRAIN_GAMMA → 0.9969`, `gaeLambda → 0.95`, `goalCritic.gamma → 0.9994`, `tickSkip → 8`.

**The surgery ran and self-verified**: `max|ΔP(ctrl)| = 4.8e-07`, `P(b*) = 0.99866` — matching
`e⁸/(4+e⁸)` exactly. Output at `staging/widened/235726848000` (450-way head, all 20 siblings
copied, `POLICY_OPTIM` omitted). Checkpoint resumes at 294,001 iterations / 2.357269e11 steps.

**Two settings the smoke tests proved are REQUIRED, neither of which was in the plan:**

| flag | why |
|---|---|
| `GGL_FRESH_OPTIM=1` | The saved Adam moments misalign against this build's param list and crash inside `Adam::step` (256 vs 1280). Reproduces with the **original** checkpoint and VTS off, so it is branch drift, not the surgery. |
| `GGL_VTS_ENVWEIGHT=0` | The advantage skip-bias pays a bonus to buckets the warm-started head holds at p≈3e-4. Rare actions with positive advantage blow up the PPO ratio and NaN the trunk within two iterations. The bias exists to fight short-skip hugging in a *from-scratch* VTS run; against a head pinned to `b*` it is backwards. |

Isolation: control (original checkpoint, VTS off, fresh optim) ran 4+ iterations clean;
widened+VTS with the default `envWeight=0.15` NaN'd on iteration 2; widened+VTS with
`envWeight=0` ran 11+ iterations clean.

> This is the D3 caveat arriving early and from an unexpected direction. The skip-bias
> tables that `DYNAMIC_TS_PROBE.md` called "crutches" are actively hostile to a warm start.
> If `C` is ever annealed toward 0, re-check whether a small `envWeight` becomes safe again —
> the hazard is the *interaction* between a peaked skip prior and a bonus on the rare tail.

## 7. Open decisions

- **D1 — resolved, pending sign-off:** resume on **HEAD defaults** (keep the corrected
  engine), with the compat block used only as a P0 diagnostic arm. Accept ≈12% relative for
  a few B steps rather than train against physics measured to be wrong.
- **D1 residual:** verify `17e505a`'s touchdown-exception tick count against the checkpoint.
- **D2:** does `GGL_NUM_GAMES≈232`/rank fit in 32GB V100, or do we take the LR reduction?
- **D3:** whether P3's outcome justifies keeping VTS at all, given the ts8 headroom caveat.
