# League anchor opponents — IMPLEMENTED, BUILT, SMOKE-VERIFIED (2026-07-19)

**Status: built and staged in `build/GigaLearnBot`; takes effect at the next
trainer restart.** Items 1 (permanent spaced archival, `tools/archive_anchor.sh`
+ `pulsar-anchor.timer`) and 2 (honest match-play battery,
`analysis/probes/anchor_battery.py`) are live and are the instruments this change
is judged by. The design below is as-built; see the AS-BUILT section at the end
for deviations and verification.

## The measured problem

Two independent measurements, 2026-07-19:

1. **The archive is collapsed.** Live telemetry: `League/Member Count` = **3** in
   92% of report blocks (cap 256); `League/Cell Count` = **1** of 216 in 92%.
   That is the structural floor: 1 protected cell-elite + 2 exploiter slots.
   Mechanism (code-verified): fitness = `memberGoals − mainGoals`, **re-scored
   against the CURRENT main every refresh** (`LeagueArchive.cpp` `RefreshStalest`
   → `EvaluateMember`, team B = live main). As the main improves, every fixed
   style's margin drifts to −∞, crossing `competenceFloor = −25`, and becomes
   both un-insertable (`TryInsert`) and cullable (`Cull`). `ReseedFromMain` only
   ever adds near-clones of the present main. So 35% of training iterations
   (`descendOpponentFrac`) face ~3 opponents that are all recent self.
2. **Real progress is slow.** Match-play head-to-head vs frozen old selves:
   +31 Elo over 8.6B steps (54.4% vs the 18.88B self), ≈ **+4 Elo/B**, while
   pool Rating claimed +187. Training against a self-similar pool is the
   plausible cause of the shallow slope.

`refAnchors = 3` in `LeagueConfig` (PSDConfig.h:160) is declared and **read
nowhere** — dead config, and the natural home for this.

## Design — one lever: a floor-exempt, fitness-exempt anchor set

**Source.** Anchors already exist on disk: `build/checkpoints_5.0v3_anchors/`
(item 1), full checkpoints, ~33MB each, spaced ~2B steps, never rotated.
No new archival machinery.

**Storage.** Load each anchor's `SHARED_HEAD`+`POLICY` into the SAME flat
param-vector representation league members already use, and run it through the
existing `MigrateFlatVec` at load. This is load-bearing: anchors span the
512→517 policy-head wire migration (a 15.3B anchor has a 512-wide head), and
`MigrateFlatVec` zero-pads exactly that case — the same path that fixed the
`dccba19` crossover crash. Anchors older than a deeper architecture change
(e.g. obs 109→230) cannot be migrated: **hard-reset the anchor set at such a
boundary** (document + quarantine, never hand-patch — recovery doctrine).

**The two exemptions (this is the whole design).**
- **Floor-exempt:** `Cull()` and `RefreshStalest()` never touch anchors.
- **Fitness-exempt:** anchors are **never re-scored** against the main. The
  margin metric is structurally doomed for old anchors; using it on them *at
  all* re-arms the ratchet that caused the collapse. This is the single
  load-bearing decision — a naive "add anchors but keep scoring them" build
  fails in exactly the same way within days.

**Selection / eviction.** Keep K = 16–32 anchors. When full, evict by
**spaced-decimation** (drop the anchor whose removal least widens the largest
gap in the retained step-timeline), NOT FIFO — FIFO keeps a moving recent
window, which reproduces the myopia we are fixing. Disk archival keeps
everything regardless; this bounds only the *serving* set.

**PFSP integration.** Anchors are a **separate draw**, not entries in the
`−|fitness|` softmax (they would score ~0 and never serve). Weight
recency-spaced with a per-anchor floor probability: newer anchors likeliest
(real contests), ancient ones rare but never zero (tail robustness).

**Budget: reallocation, not addition.** Split the existing
`descendOpponentFrac = 0.35`: elites **0.20** / exploiters **0.10** /
anchors **0.05**. No new arena cost. Exploiters keep the sharpest gradient
(they attack current weaknesses); anchors are cheap anti-forgetting insurance.

## Failure modes and mitigations

| Risk | Mitigation |
|---|---|
| Re-collapse via the fitness ratchet | Fitness-exemption (above). The one thing that must not be compromised. |
| Main meta-overfits a fixed opponent set (the 4.0 anchor-lead erosion: 29-15 → 13-11 over an evening) | That was a *single* fixed anchor. K≥16, spaced, and continuously growing = a moving target. Bounded further by the 0.05 cap. |
| Weak anchors become free wins / wasted arena time | 0.05 cap bounds the waste; recency-spaced weighting starves ancient free-wins. Do **NOT** drop weak anchors on a strength floor — that is the ratchet again. |
| Stale net shape (the 512→517 class) | Reuse `MigrateFlatVec`; hard-reset at deeper arch changes. |
| Anchors starve the adaptive exploiters | Separate set + separate draw: structurally cannot consume the elite/exploiter budget. |

## Pre-registration (required before build, per doctrine)

**Baseline (already captured, 2026-07-19):** `anchor_battery.py` headline —
real match-play Elo vs the oldest anchor, plus the +4 Elo/B slope over the last
8.6B steps. This is the first honest baseline this run has ever had.

**Success criterion (primary):** the anchor-battery **real-Elo slope improves**
over the equivalent post-change window vs the pre-change baseline. Judge on
Elo-per-billion-steps, not Rating.

**Success criterion (secondary, anti-overfit):** win-share against a **held-out**
anchor subset (excluded from league serving) stays in a **45–65%** band rather
than trending to 100% — that would indicate memorization of the served set.

**Diagnostic (not a criterion — gameable):** `League/Cell Count` and
`League/Member Count` sustained above the 3/1 floor.

**Automatic guard:** the existing rating-latch pattern — if enabling anchors
drops `Rating/1v1` >75 below its slow EMA, latch anchor-frac → 0 for the
process. Rating is inflated in *absolute* terms but still detects the −130-class
collapse signature it was built for; the anchor battery is the slower honest
check running alongside.

**Revert:** `anchorFrac = 0` (single flag) → archive behavior byte-identical to
today; the anchor dir is read-only input and can simply be ignored. Additive,
default-off. Branch-point checkpoint backup before enabling, per ritual.

## Touch points

- `GigaLearnCPP/src/private/GigaLearnCPP/League/LeagueArchive.{h,cpp}` —
  anchor set, load-from-dir + migrate, `Cull`/`RefreshStalest` exemptions,
  `SampleOpponent` separate draw.
- `GigaLearnCPP/src/public/GigaLearnCPP/PSDConfig.h` — `LeagueConfig`: repurpose
  the dead `refAnchors`, add `anchorFrac`, `anchorDir`.
- `GigaLearnCPP/src/public/GigaLearnCPP/Learner.cpp` (~:3330) — opponent serve
  path.
- `src/ExampleMain.cpp` (~:768-785) — live config + rationale comment.

## AS-BUILT (2026-07-19) — deviations, verification, deploy state

**Deviations from the spec above (all deliberate):**
- Anchors live in their own `std::vector<Member> anchors` on `LeagueArchive`, NOT
  as flagged entries in `members`. Every collapse mechanism (`Cull`,
  `RefreshStalest`, `TryInsert`, `cellToMember`, `DedupCells`) walks `members`
  only, so the exemptions are STRUCTURAL rather than scattered if-checks, and
  anchors stay out of the cell/diversity panels. They are also not persisted into
  the league dir — they are re-loaded from their checkpoint dirs at every boot.
- Loading reuses `Model::Load` on a `MakeClone()` of the scratch models, which
  performs the pre-wire 512→517 zero-pad itself (`allowInputExpand` survives the
  clone). No bespoke migration code. Each anchor is pre-checked for its model
  files and wrapped in try/catch — `RG_ERR_CLOSE` throws, and an anchor is an
  OPTIONAL input that must never take down the run; a bad one is skipped + logged.
- `anchorFrac` is expressed as a share of ALL iterations (0.05) and converted
  internally to `P(anchor | league serve) = anchorFrac / descendOpponentFrac`,
  so it reads the same way the budget is discussed.
- **Guard implemented, not assumed.** The league serve path was NOT rating-latch
  covered (`league && RandFloat() < frac`, no `steerRatingTripped`). Rather than
  soften the claim, `LoadPFSPOpponentModels(bool allowAnchors)` now takes the
  latch and the call site passes `!steerRatingTripped` — the latch kills the
  ANCHOR slice only; evolved members keep serving.
- `GGL_SMOKE` forces `anchorFrac = descendOpponentFrac` so a short smoke exercises
  the path deterministically. This override MUST sit after the `cfg.league.*`
  assignments — placed in the main smoke block it is silently clobbered (was, and
  produced a misleading "Anchor Serves 0").

**Verification (smoke on the exact deploy binary, `GGL_SMOKE=1 GGL_DEVICE=cpu`,
throwaway sandbox, scrubbed `run_id`):** 26 iterations, no errors;
`League anchors: serving 3 of 3 archived (log-spaced; 15300219140 .. 27925184870)`;
both pre-wire anchors logged `MIGRATED ... 512 -> 517`; `League/Anchor Serves`
climbed 0 → 9 at the forced rate. `League/Anchor Count` + `League/Anchor Serves`
added to the curated stdout block (Serves rising is the only proof the slice is
actually played; Count 0 means the anchor dir is missing and the feature is inert).

**Deploy state:** compile-clean in both trees; `build/GigaLearnBot` rebuilt with
the change; branch-point backup taken at
`build/checkpoints_5.0v3_branch_backup/28025089688` before enablement. The change
activates at the next trainer restart (the running process holds the old binary
via its inode; the wrapper's crash-restart would also pick it up).

**Watch after deploy:** `League/Anchor Serves` rising (~1 iteration in 20 at 0.05);
`League/Anchor Count` = number archived, capped at 24; `Rating/1v1` vs the latch;
and the real yardstick — re-run `anchor_battery.py` and compare the real-Elo slope
against the pre-change baseline recorded in `results/anchor_battery_history.jsonl`.
Note anchors are loaded ONCE at boot, so anchors archived later join the serving
set at the next restart (acceptable: restarts are frequent; avoids a live-reload
failure surface).

## Open question for the reviewer

Anchor spacing is currently 2B steps ≈ one anchor every ~4 hours at 150k SPS,
so the *archive* grows ~6/day (fine: 33MB each). The **serving** set is capped
at K by decimation. If you want anchors that span the run's whole history
rather than its recent weeks, K should grow or spacing should widen — worth
deciding before build.
