# The Optimistic-Critic Ladder — Pulsar integration state (2026-07-18)

> **SUPERSEDED 2026-07-25.** The machinery described here was REMOVED from the trainer in the
> composition-critic conformance pass. Kept for provenance and for the lessons it carries — not
> as a description of the current system. What runs now: `research/reports/COMPOSITION_CRITIC.md`
> (the spec) and `docs/ENABLED_INVENTORY.md` (the live inventory).

The full spec was delivered by the user; canonical copy lives in the session
record and this file's Implementation-state section is the working map.

## Implementation state

DONE (live on 5.0v3 since 15.3B, commits fc34dd3 + 36f1e01):
- V_exp sensor: expectile twin, tau 0.8, DETACHED trunk probe, extrinsic GAE
  targets computed pre-injection (spec 2.4, Laws 3/4) = ladder rung 2.
- gap_KD drive: undiscounted closure potential, advantage-level, centered,
  std-matched beta 0.05, 3-sigma clamp, terminal-masked, latch-covered
  (spec 2.6 for the KD half). Steering actuation retired per protocol.
- Off=identical gating (Law 7). Healthy-closure telemetry (Gap/VReal vs VExp).
  40-min live read: injection ~5% adv scale, VReal rising with VExp, fear-panel
  gap halving, rating +109 to 1359.

REMAINING (build order per the spec's own validation protocol):
1. V0 unit invariants for existing pieces + segment-boundary mask audit
   (current drive masks terminals; VERIFY truncation-boundary rows too - spec
   calls out the row-shift trap explicitly).
2. V1 OFFLINE: quasimetric map (separate encoder/head nets, QRL local+spread
   losses, dual lambda) trained on archived rollouts; gate = Spearman >= 0.5
   held-out steps-to-reach incl. excluded-trajectory pairs, margin over
   Euclidean baseline. STOP if failed.
3. V2 instrument-only online: map + banks (goal/concede pre-goal windows) +
   V_metric calibration (lstsq vs extrinsic targets, EMA, clamps) + probe
   telemetry; wire and gap_PK drive OFF.
4. V3 full system: 5-input wire (policy-head zero-init surgery), combined
   Phi = -(gap_KD + gap_PK), impossible-control drill family certified on
   both axes, all Laws 1-7. Single-seed signatures per spec.
5. V4 multi-seed referendum on certified drill targets.

Laws already institutionalized here by prior incidents: 1 (separate gradient
economies - the Muon/carstate lessons), 2 (map never touches trunk), 4
(probes detached), 6 (wire+drive together), 7 (off=identical). Law 5 note:
our retired steering WAS a direction-conditioning-adjacent mechanism; the
ladder replaces it, consistent with the user's Stage-2 protocol.

## AUTHORIZATION (user, 2026-07-18): IMPLEMENT IN FULL

"This is all tested and working. go ahead and implement it in full."
- The mechanism is validated in the source codebase; the spec's V1-V4 staged
  validation is WAIVED as a gate (upstream evidence stands). Retain only this
  repo's own deployment ritual: V0 invariant unit checks (esp. the truncation
  -boundary mask audit on the existing drive), compile-check tree, CPU smoke,
  branch backup, deploy, panels, latch coverage, revert flag.
- Build in one session, in this order: map nets + own optimizer (Laws 1/2) ->
  banks + V_metric calibration -> gap_PK + combined drive Phi=-(gKD+gPK) with
  all masks incl. impossible-control rows -> 5-input wire with zero-init
  policy-head extension (517) computed fused at collection, re-derived at learn
  with collection-time bank embeddings + calibration snapshot (spec 2.5) ->
  impossible-control drill family + probe telemetry -> smoke -> deploy.
- Steering actuation stays OFF (already done). Fear drills/census/telemetry
  stay ON. Rollback anchor: checkpoints_5.0v3_branch_backup/15300219140 +
  golden archive.

## IMPLEMENTED IN FULL AND DEPLOYED (2026-07-18, commit 15eabda)

Live on 5.0v3 from ~18.88B steps (restart 20:20 EDT). What shipped, and where:

- **Quasimetric map** (`GapState` in Learner.cpp): E obs->256->256->64 GELU,
  f 64->128->32 GELU, d = relu-sum. One QRL step/iteration: L_local on
  consecutive same-agent non-boundary pairs (512), L_spread on random pairs
  (512, clamped), dual-ascent lambda (persisted). OWN Adam lr 1e-3 + OWN clip
  group 1.0 (Law 1); raw obs in, trunk untouched (Law 2). Persisted as
  GAP_MAP_E.lt / GAP_MAP_F.lt; D_CLAMP = 3 x EMA(mean episode steps),
  live-derived. Retention gauge: frozen consecutive-pair set, violation
  reported每 iteration (Ladder/Retention Viol; re-freezes per process).
- **Banks + V_metric** : goal/concede raw-obs rings (1024/side; final ~1s =
  15 rows before each +-1 goal-channel terminal, all players). Calibration:
  per-iteration fp64 lstsq (gelsd) of [g^dg, g^dc, 1] vs the critic's
  extrinsic GAE targets on 8192 rows; EMA 0.9; clamps a in [0,2*p99|y|],
  a2 in [-2*p99|y|,0], b in +-p99|y|; persisted in RUNNING_STATS. V_metric =
  V_exp until both banks >= 32 AND mapUpdates >= 50.
- **Combined drive**: Phi = -(gKD + gPK), a_int masked (terminals AND
  truncations - both nonzero TerminalType codes; AND impossible rows),
  centered OVER UNMASKED ROWS ONLY + re-masked (Law 8b correction to the
  validated code's all-rows centering), beta_eff = 0.05*sigma_ext /
  max(0.05*sigma_ext, sigma_int|m=1), clamp +-3 sigma_ext, injected into raw
  advantages before buffer handoff (no advantage normalization exists in this
  codebase - ordering law satisfied). Latch-covered (steerRatingTripped).
- **5-input wire**: policy head 512 -> 517. Zero-pad migration at load
  (Model::Load for .lt files, LeagueArchive::LoadInto for flat member
  vectors); policy optimizer (Muon) reset once at migration - accepted
  logged transient. Collection: fused in InferPolicyProbsFromModels from the
  pipelined snapshot (critic joins snapshotNames; exp/mapE/mapF deep-cloned +
  bank embeddings + calibration frozen per generation at the barrier). Learn:
  re-derived per minibatch from CURRENT heads + the COLLECTION generation's
  bank embeddings/calibration (spec 2.5 tested rule); Learn() hard-refuses if
  the handoff wasn't armed (no silent fallback). Eval/opponent/render/boot
  paths feed explicit zeros (Rating measures the raw policy, symmetric across
  the pool - documented choice). Wire columns' grad norm on the Ladder/Wire
  Col Grad panel (Law 6 Muon watch).
- **Impossible-control family**: ImpossibleInterceptState on the LAST 8
  arenas of the 1v1 block (both sites derive the rule independently;
  Learner errors on overlap with the practice slice). Certificate per
  jittered spawn: fully-airborne ballistic line into a goal (T in
  [0.75, 1.05]s, pure projectile - no bounce), every car's point-to-segment
  distance >= 1.6 x 2300 x T + 500. Rows: steer role 3 (excluded from every
  equality-keyed steering pool/census), masked from injection (Ladder/Inj
  Mean Imp must read 0), cumulative touch counter persisted (Ladder/Imp
  Touches must stay 0 for the life of the run - one touch voids the
  certificate). Spawn-row probes: Ladder/Imp {VReal,VExp,GapKD,GapPK} Spawn.
- **Telemetry** added to the curated stdout block + wandb: Ladder/Map Local
  Loss, Map Spread, Lambda, DClamp, Bank fills, Calib A/A2/B, GapPK Mean/P90,
  Fear GapPK (feasible comparator), Retention Viol, Wire Active, Wire Col
  Grad, BetaEff, Inj Mean Match/Imp, AInt DgDown/DgUp Mean (boundary-crossing
  credit geometry), Imp* spawn probes, Imp Touches.

Smokes (CPU, GGL_SMOKE + GGL_DEVICE=cpu, warmups collapsed in smoke only):
fresh run - full path live by iter ~5, no crashes; migration resume of the
live checkpoint - policy + 32 pool versions + league elites all migrated,
boot probe 3/3 kickoff touches (zero-pad behavioral exactness verified),
17 iterations clean. One found-and-fixed: league flat member vectors needed
their own migration path (view_as crash, fixed in LoadInto).

Live boot verification: MIGRATED banner on 18875158586, boot probe 3/3,
PHASE B 667/204/153, SPS ~170-200k (normal band), Gap/Loss resumed at 0.019,
drive injecting, banks full within 2 iterations, wire ACTIVE with nonzero
column grads, Imp Touches 0. gap_PK gated off until mapUpdates >= 50 (prod
warmup) - Calib/GapPK panels appear a few minutes post-boot.

WATCH LIST (days-scale):
- Ladder/Imp GapPK Spawn must fall BELOW Ladder/Fear GapPK as the map's doom
  geometry converges (it starts above - untrained map). If it never crosses,
  the falsification family is telling us optimism is self-serving.
- Ladder/Retention Viol flat = no forgetting; trending up = apply the
  reservoir contingency (spec 2.1).
- Ladder/Calib A > 0 > A2 with both off their clamps once the map matures
  (early near-collinearity pins them at the clamp, by design).
- Ladder/Wire Col Grad: bounded, not random-walking (Law 6 under Muon).
- Gap/Drive Inj Abs Mean ~5% of adv scale; Rating vs the drawdown monitor.
- Wire zeros at eval understate a wire-dependent policy symmetrically; if
  Rating drifts oddly vs head-to-head, revisit (open question #1 analog).

Rollback: checkpoints_5.0v3_branch_backup/18875158586 (pre-wire, 512-head)
+ 15300219140 + golden archive. mapEnabled=false degrades to gKD-only drive;
driveBeta=0 kills drive+wire; the 517 head itself reverts only via backup.
