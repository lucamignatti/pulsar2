# ENABLED FEATURE INVENTORY — 2026-07-25 (post-strip)

Every mechanism still in the live compile graph, with its **resolved live value** from
`src/ExampleMain.cpp` (not header defaults). This is the review document: read it top to bottom
and challenge anything that shouldn't be here.

Companion docs: `DEAD_CODE_AUDIT.md` (what was found), `research/reports/COMPOSITION_CRITIC.md`
(the paper the optimism stack now conforms to). Restore point for everything removed:
tag `pre-strip-20260725`.

**Session result: 7,904 lines of C++ deleted, 522 added — net −7,382.** Subsystems removed
whole: PSD/Basin-Racing, proposer/drill bank, TransferLearn, the rating latch, the Ladder's
map/banks/`V_metric`/`gap_PK`/wire, RND novelty, the impossible-control family, and
**activation steering in its entirety** (actuation, META, the derivation, `FrontierPool` and
`FrontierDrillState`, the practice/control arena split).

---

## 1. Core training loop — LIVE

| Feature | Live value | Notes |
|---|---|---|
| `pipelinedCollection` | `true` | Collect N+1 on a frozen snapshot while learning N. Now gated only on `!render`. |
| `useHalfPrecision` | `true` | bf16 for collection + GAE value preds; grad-enabled forwards stay fp32. |
| `tsPerItr` / `batchSize` | 200k | |
| `miniBatchSize` | 20k | The learn-pass activation-peak lever. Never cut — the OOM was the vdag version-clone bug, not the activation peak. |
| `gaeGamma` | 0.9969 | ~15s half-life at 15 Hz. **Re-derive if tickSkip changes.** |
| Net | trunk 3×1152, policy 3×768, critic/goal/vdag×2 5×1280, residual | ~37.96M params. Policy head is plain trunk width again (wire removed). |
| Reset mix | BallNearCar 0.30 / AirDrill 0.20 / AirPlay 0.15 / Kickoff 0.10 / Random 0.25 | **Fixed and identical in every arena** — not a curriculum, does not adapt. That is what C1 asks for. |

## 2. Optimism — now exactly the paper

`V_real` → `V_exp` → twin composition critics `V†`, actuated by **one** term.

| Component | Live value | Role |
|---|---|---|
| `V_real` | the critic | what I reliably do |
| `V_exp` | `gapSensor.enabled`, τ=0.8 | return-level expectile. **Measurement only** — publishes `Gap/*`, nothing injects from it |
| `V†₁,V†₂` | `vdagEnabled`, τ=0.75 | Bellman-level expectile twins; target uses **min over twins** (the anti-ratchet) |
| Actuation | `vdagSeekBeta = 0.15` | `Φ = +H`, `H = relu(min(V₁,V₂) − V_real)`, σ-matched, ±3σ clamp |

Both optimistic heads are **expectile**, not quantile — the asymmetric weight multiplies a
*squared* residual. The paper measured the quantile variant and rejected it (§6.4).

**Watch `Headroom/Vdag Update Magnitude`.** It read exactly 0 for the life of the run until
2026-07-25 because `SetLearningRates` never named the twins and `Model`'s ctor builds every
optimizer at `lr=0`. If it returns to 0, HEADROOM is inert and its injection is a random
projection.

## 3. Advantage injectors — down from four to two

| Injector | Beta | Anchor | Matches against |
|---|---|---|---|
| HEADROOM seek | 0.15 | `Learner.cpp:2858` | **pre**-injection extrinsic σ |
| Goal-critic blend | 0.25 | `Learner.cpp:2929` | `tAdvantages.std()` *after* HEADROOM landed |

RND (0.10) and the Ladder drive (0.05) are gone. **Open:** the two survivors still don't share a
reference — the goal critic re-reads `tAdvantages.std()` post-HEADROOM. And `GAE/Avg Advantage`
is still computed *after* the HEADROOM injection, so the panel named for raw GAE remains a
hybrid. Decide the intended composite and whether both should match the same pre-injection
reference.

## 4. Steering — GONE

Removed wholesale 2026-07-25. Not just as bloat: what remained **contradicted** the constraint
set of the paper the trainer now implements.

- **C1 (homogeneity)** — "all environment instances identical; no dedicated drill/reset
  instances." The practice/control arena split violated it.
- **C2 (no environment design)** — "no banked reset states, no success-state memories that define
  the objective retrospectively." `FrontierPool` was exactly that; §4.1 criticises the design by
  name.

Gone: `CollectSteeringConfig`, `fnSteerUpdate` (all sections), `FrontierPool` +
`FrontierDrillState`, the practice/control split and `arenaSteerRole`,
`AttemptResolutionCondition` + `resolutionTermination`, `opponentStyleChance` and the
opponent-style delta path, the `Steer/*` and `Miner/*` panels, and the never-assigned air-drill
curriculum plumbing.

**Behavioural consequence:** practice arenas were 20% of the fleet and drew `FrontierDrillState`
resets 60% of the time; they now draw the standard mix. Lands with the same cold start deployment
already required.

**Every arena is now identical** — which is C1.

The lessons live in `research/reports/` (`STEERING.md`, `STEERED_PRACTICE.md`, `FEAR_MINE.md`,
`KNOWING_DOING.md`); the code is one `git log` from `pre-strip-20260725`. Two still bind and are
cited from `ExampleMain`: the **clipping ratchet**, and **critic aliasing at episode boundaries**.

## 5. Outer loops

Rewritten 2026-07-25: the QD League was removed and its three jobs consolidated into
`PolicyVersionManager`. See `LEAGUE_RECON.md` for why, and `CLAUDE.md`'s Outer loops
section for the resulting shape.

| Feature | Live | Notes |
|---|---|---|
| Version ring | `enabled`, 32 versions @ 25M | Now the **training** opponent source as well as the Elo pool. Versions do not clone the vdag twins (~2 GB VRAM, the OOM cause). |
| Train vs past selves | `trainAgainstOldChance 0.30` | Realized **0.255** — the cascade is sequential and Nexto rolls first. Total non-self exposure 0.405, deliberately equal to the pre-strip measured 0.408. |
| `Rating/1v1` | `enabled` | **INFLATED ~6×** and kept only for continuity: new versions inherit the main's current rating and both sides move on every goal. Training against the ring makes it worse. Read `Ref/Oldest Share`. |
| Reference set | `maxReferences 8`, battery every 64 iters | Log-spaced, never trained against, never re-rated, oldest never evicted. `Ref/Oldest Share` is the non-inflating read. Cumulative counters — the slope is the signal. |
| Nexto opponent | `serveFrac 0.15`, CPU-only | Dose fixed 2026-07-25 (was a 127.773 s wall-clock sawtooth realizing 8.9%); counters now restore on boot. |
| Rating watch | measurement only | `RatingWatch/Drawdown From EMA` and `/From Peak`. **No latch** — nothing acts on it. |
| QD League | **REMOVED** | Tag `pre-league-strip-20260725`. Collapsed on every lineage that learned; served copies of the untrained birth network; value never measured. |

## 6. Reachability

`enabled`, `gateEnabled=false`, `carStateHead=true` (trained fully detached). Three heads train
every iteration. With steering's rho-gate gone and the proposer removed, **rho has no live
decision consumer** — all three heads are pure telemetry. `psi_carstate` costs a fwd+bwd+Adam
step per minibatch and 4.2 MB/checkpoint for a signal nothing reads.
**Open:** keep as instrumentation, or strip the car-state head?

## 7. Plasticity — promoted from PSD

`Plasticity/Trunk EffRank`, `/Policy EffRank`, `/Policy Dead Units`. Weights-only, every
iteration, no actuation. Kept because the residual architecture (`45a59d5`) was justified *by*
effective-rank decay and PSD owned the only implementation. Healthy reference from a fresh
smoke: trunk 927/1152, policy 88/90, 0 dead units.

## 8. Reward stack (live weights)

BallToGoal **75**, TouchAccel 10, Demo 37.5, BallProximity 4, GuardedPickupBoost 6,
AerialTouch **120**, AirIntercept **75**, ConsecutiveAirTouch 30, WallJumpToBall 30,
FlipReset 40, AirReward 0.45, OpposedSave 25, **CarEnergy 75**, TimeCost 0.01,
TeamPressure 0.15, KickoffRace 25, Goal 150.

**Open (two):** `CarEnergy 15 → 75` is recorded but unmeasured on this lineage — extrapolation
from `REWARD_SHAPING.md` puts it near 60% of per-step credit density. And six SCAFFOLD weights
totalling ~260 (vs Goal 150) name `mechanic_census` as their anneal trigger; that metric measures
none of the relevant mechanics and cannot load the live checkpoints, so "anneal later" currently
means "never".

## 9. Known-broken

**The offline toolkit cannot load a single checkpoint of this run.** `load_checkpoint.py`
hardcodes 512-wide/2-layer/non-residual shapes and `_default_root()` falls through to the frozen
`checkpoints_5.0v3`. 38 scripts route through it, including `match_play_eval.py` and
`anchor_battery.py` — the honest-progress yardsticks. A partial fix is worse than none:
`residualSpans` is never serialized, so a naive width fix rebuilds the residual net as a
skip-free MLP, replaying the 2026-07-19 h2-truncation bug class.

## 10. Safety posture — read this one

**There is no automatic guard on this run any more.**

- The rating latch was removed (your call, 2026-07-25) — `RatingWatch/*` is telemetry only.
- The impossible-arena certificate went with `gap_PK`; its acceptance criterion was defined
  against it.
- What remains: the **boot sanity probe** (3 kickoff episodes, ≥2 must have touches, only for
  checkpoints claiming rating ≥ 400) and the golden archive. Neither sees gradual update damage.

Also worth knowing: `GGL_SMOKE` with `GGL_DEVICE=cpu` is **not** GPU-free — the CUDA-linked
binary still creates a ~282 MB context at boot.

---

## Decisions waiting on you

1. **Deploy the conformance pass** — requires a fresh cold start (policy head 1152 vs the running
   lineage's 1157). Not done; the 175M-step run would be abandoned.
2. **Composite injection budget** (§3) — should both injectors match the same pre-injection σ?
3. **Reachability car-state head** (§6) — instrumentation, or strip?
4. **`CarEnergy 75`** (§8) — measure and justify, or walk back?
5. **SCAFFOLD anneal triggers** (§8) — implement `mechanic_census`, or delete the promise?
6. **Offline toolkit** (§9) — repair properly, or accept it as 5.0v3-only?
7. **Rename `steering.enabled`** (§4) — it no longer steers anything.
8. **Accept the no-guard posture** (§10), or re-introduce something narrower than the old latch?
