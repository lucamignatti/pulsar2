# ENABLED FEATURE INVENTORY — 2026-07-25 (post-strip)

Everything still in the live compile graph after the dead-code strip, with its resolved live
value. Companion to `DEAD_CODE_AUDIT.md` (what was found) — this is **what remains, and whether
you should keep it**.

Resolved against `src/ExampleMain.cpp` (THE config), not header defaults. Restore point for
everything removed: tag `pre-strip-20260725`.

**Strip result: 4,269 deletions / 1,097 insertions across the session.** Subsystems removed
whole: PSD/Basin-Racing (~1,880), proposer/drill bank (~1,220), TransferLearn (~280), dead types
and optimizers (~480), dead Report API (~45), the rating latch.

---

## 1. Core training loop — LIVE, no action

| Feature | Live value | Notes |
|---|---|---|
| `pipelinedCollection` | `true` | Collect N+1 on a frozen snapshot while learning N. Now gated only on `!render` — the proposer terms that also gated it are gone (behaviourally identical; both were already true). |
| `useHalfPrecision` | `true` | bf16 for collection + GAE value preds. Grad-enabled forwards stay fp32. |
| `tsPerItr` / `batchSize` | 200k | |
| `miniBatchSize` | 20k | The learn-pass activation-peak lever. **Not** cut this session — the OOM turned out to be the vdag version-clone bug, not the activation peak. |
| `TRAIN_GAMMA` / `gaeGamma` | 0.9969 | ~15s half-life at 15 Hz. Re-derive if tickSkip changes. |
| Net | trunk 3×1152, policy 3×768, critic/goal/vdag×2 5×1280, residual | ~38M params. |

## 2. Advantage injectors — **the thing to look at**

Four std-matched terms now chain into `tAdvantages`. **Nothing owns the total**, and the
rating latch that used to cover all four is gone.

| Injector | Beta | Anchor | Status |
|---|---|---|---|
| HEADROOM seek | 0.15 | `Learner.cpp` (~4590) | **Newly real.** vdag twins were frozen at random init for the life of the run until 2026-07-25; they now train (`Headroom/Vdag Update Magnitude` ≈ 0.3, was exactly 0). |
| Goal-critic blend | 0.25 | goalCritic | LIVE, long-horizon credit (γ=0.9994). |
| RND novelty | 0.10 | `rndOptimism` | LIVE from step 0. |
| Ladder drive | 0.05 | `gapSensor.driveBeta` | LIVE. The only one that masks terminals/truncations/impossible rows. |

**Open:** composite ≈ 0.32 σ_ext, computed assuming independence. Only HEADROOM std-matches
against *pre*-injection advantage std; the other three re-read `tAdvantages.std()` after earlier
ones have landed. `GAE/Avg Advantage` is computed *after* the HEADROOM injection, so the panel
named for raw GAE is a hybrid. Decide the intended total budget and whether all four should
match against the same pre-injection reference.

## 3. Optimism mechanisms

| Feature | Live | Notes |
|---|---|---|
| Optimistic-Critic Ladder | `enabled`, `mapEnabled`, `wireEnabled` all true | Current optimism lever. Policy head is 517-wide (512 + 5 wire). One-way migration. |
| Impossible-control family | 8 arenas | Standing falsification test. `Ladder/Imp Touches` must stay 0 for the life of the run — **protected, do not strip**. |
| RND novelty | `weight 0.1` | Advantage-space, self-annealing. |
| HEADROOM (vdag twins) | `vdagEnabled`, seek 0.15 | See §2. Now also gradient-clipped at 0.5 like every other head — they were the only unclipped block once the LR went live. |

**Open (carried from the audit):** the Ladder's acceptance bar is *unscoreable in PHASE A*.
`Ladder/Imp GapPK Spawn` publishes but its comparator `Ladder/Fear GapPK` only freezes under
`md == 1` (2v2), and this run is 1v1. Either re-key the fear-panel freeze to `md == 0` or retire
the bar in `LADDER.md` — as written, the falsification test cannot be evaluated while the drive
is live.

## 4. Steering — INERT, the remaining strip target

| Part | Live value | Status |
|---|---|---|
| `steering.enabled` | `true` | Kept ON only so derivation/census/miner telemetry runs. |
| `steering.alpha` | **0** | The actuation multiplies by zero. |
| `opponentStyleChance` | 0 | Parked by the Stage-2 protocol. |
| `resolutionTermination` | `false` | Two Elo collapses. **Keep disabled** — post-mortem in `STEERED_PRACTICE.md`. |
| `meta` | `false` | ~500 LOC dormant. |
| `frontierFearMining` / `emergenceMiner` | `true` | Feed telemetry + the frontier pool. |
| `frontierPotential` | env-gated, off | `GGL_FRONTIER_POTENTIAL`. |

**NOT stripped this session, deliberately.** The derivation is what fills `FrontierPool`, which
drives `FrontierDrillState` on 30% of arena resets — that is LIVE. Removing the actuation
(`fnApplySteering`/`SetSteering`, derivation §5/5b, the causal gate, the rho-gate snapshot)
requires splitting a ~900-line function while keeping §1-4 and §6-8, and `arenaSteerRole` is
shared with the Ladder's impossible arenas. Payoff is ~340 LOC plus **3 GPU trunk forwards and 4
barrier-zone param copies per iteration** — worth doing, but as its own careful change, not
bolted onto a large strip.

## 5. Outer loops

| Feature | Live | Notes |
|---|---|---|
| Skill tracker (Elo) | `enabled`, 32 versions | Versions no longer clone the vdag twins — that was ~2 GB of VRAM and the cause of the OOM crash loop. |
| QD League | `enabled`, `descendOpponentFrac 0.35` | Match env now clears its reward stack (~58k discarded evals/evolve removed). `LeagueConfig` extracted to its own header. |
| League anchors | **off** — no anchor dir | `checkpoints_resid_anchors` does not exist. |
| Nexto external opponent | `serveFrac 0.15`, CPU-only | Goal counters now **exclude the impossible arenas** — this is the only pool-inflation-proof yardstick and was being diluted. |
| Rating watch | measurement only | **The latch is gone** (user-directed). `RatingWatch/Drawdown From EMA` and `/From Peak` are published; nothing acts on them. Update damage is now yours to catch. |

## 6. Reachability

`enabled=true`, `gateEnabled=false`, `carStateHead=true` (trained fully detached).
Three heads train every iteration. With steering parked and the proposer removed, **rho has no
live decision consumer** — all three heads are now telemetry. `psi_carstate` in particular costs
a fwd+bwd+Adam step per minibatch and 4.2 MB/checkpoint for a signal nothing reads.
**Open:** keep as instrumentation, or strip the car-state head?

## 7. Plasticity — new

Promoted out of PSD rather than deleted, because the residual architecture (`45a59d5`) was
justified *by* effective-rank decay and PSD owned the only implementation.
`Plasticity/Trunk EffRank`, `/Policy EffRank`, `/Policy Dead Units` — weights-only, every
iteration, no actuation.

## 8. Reward stack (live weights — CLAUDE.md was stale)

BallToGoal **75**, TouchAccel 10, Demo 37.5, BallProximity 4, GuardedPickupBoost 6,
AerialTouch **120**, AirIntercept **75**, ConsecutiveAirTouch 30, WallJumpToBall 30,
FlipReset 40, AirReward 0.45, OpposedSave 25, **CarEnergy 75**, TimeCost 0.01,
TeamPressure 0.15, KickoffRace 25, Goal 150.

**Open:** `CarEnergy 15 → 75` rode uncommitted through two cold starts and is now recorded but
still unmeasured on this lineage — static extrapolation from `REWARD_SHAPING.md` puts it near
60% of per-step credit density. Six SCAFFOLD weights totalling ~260 (vs Goal 150) name
`mechanic_census` as their anneal trigger; that metric measures none of the relevant mechanics
and cannot load the live checkpoints, so "anneal later" currently means "never".

## 9. Known-broken, unaddressed

**The offline toolkit cannot load a single checkpoint of this run.**
`load_checkpoint.py` hardcodes 512-wide/2-layer/non-residual shapes and `_default_root()` falls
through to the frozen `checkpoints_5.0v3`, so 38 scripts — including `match_play_eval.py` and
`anchor_battery.py`, the honest-progress yardsticks — silently analyse the wrong lineage.
A partial fix is worse than none: `residualSpans` is never serialized, so a naive width fix
rebuilds the residual net as a skip-free MLP, replaying the 2026-07-19 h2-truncation bug class.

## 10. Also worth knowing

- **`GGL_SMOKE` with `GGL_DEVICE=cpu` is not GPU-free.** The CUDA-linked binary still creates a
  ~282 MB context at boot. Harmless at current headroom (~4 GB free) but it qualifies the
  "CPU smoke is safe" rule.
- **The test suite gates again.** It failed 6/30 on a clean tree (stale BallProx oracle, stale
  AirDrill expectations) and printed the *pass* count behind the word FAILED. Now 31/31, exit 0,
  with new coverage for the `soloFrac` solo-completion path that ships on ~8% of resets.
- **`cpp-interface` patches are safe now** — committed in the submodule *and* exported to
  `patches/cpp-interface/`, because a submodule-local commit lives on no remote.

---

## Decisions waiting on you

1. **Composite injection budget** (§2) — is ~0.32 σ intended? Should all four match the same reference?
2. **Ladder acceptance bar in PHASE A** (§3) — re-key the fear panel, or retire the criterion?
3. **Steering actuation** (§4) — strip it as its own change? ~340 LOC + real per-iteration GPU cost.
4. **Reachability car-state head** (§6) — instrumentation, or strip?
5. **CarEnergy 75** (§8) — measure and justify, or walk back?
6. **SCAFFOLD anneal triggers** (§8) — implement `mechanic_census`, or delete the "anneal later" comments?
7. **Offline toolkit** (§9) — repair properly, or accept it as 5.0v3-only?
8. **No automatic update-damage guard exists any more.** That was your call and it is recorded;
   just be aware nothing but you will catch the next `-200` Elo slide.
