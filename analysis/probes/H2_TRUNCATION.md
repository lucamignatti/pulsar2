# The h2-truncation incident — offline loader bug, proof chain, corrected results (2026-07-19)

## Root cause (proven)

`load_checkpoint.py::rebuild_sequential` inferred activation modules from GAPS in
the archive's parameter indices. A parameterless module AFTER the last
parameterized index leaves no gap: the trunk `[Linear(0), LN(1), LReLU(2),
Linear(3), LN(4), LReLU(5)]` lost its final LReLU (index 5 > max param index 4).
**Every offline consumer of "h2" — policy actions in every offline rollout,
compare_checkpoints head-to-heads, trunk_h2 probes, knowing_doing, steer_test —
had been reading the PRE-ACTIVATION trunk output since the harness was written.**

Why nothing caught it: h1 (`trunk[:3]`) was complete, so the passthrough sanity
gate (h1 ball-z R² 0.97) passed honestly; policy/critic heads contain LayerNorms
that largely re-normalize the perturbed input, so offline play looked coherent;
ridge probes re-fit affine-ish distortions. The un-normalized GAP_EXP head
(plain Linear/LReLU stack) was the only h2 consumer with no defense — it read
V_exp ≈ −8 where the live trainer read V_real + 0.15, which is what exposed the
bug during the Ladder wire-reconstruction work.

## Proof chain (each step eliminated one suspect)

1. Manual matrix-multiply forward == rebuilt module (my read of the archive was
   internally faithful).
2. Trainer's own frozen `fear_panel_obs` through my pipeline still gave −6.9
   (obs port exonerated; the trainer logs 0.15–0.17 on those rows).
3. 16-way input/activation composition sweep: nothing reproduced the live panel.
4. GAP_EXP.lt differs checkpoint-to-checkpoint (~3e-2 max-abs per ~700 iters):
   the file tracks live training (persistence exonerated).
5. **GGL_SMOKE + GGL_DEVICE=cpu oracle** on a full copied checkpoint: the C++
   loader read the SAME file as a healthy net (Gap/Loss 0.041 → ~0.025,
   Gap/Fear Panel 0.139, 36 clean iterations). Trainer fully exonerated; the
   Python read was the only remaining suspect → truncation found.
6. Fix (`trailing_activations=1` for SHARED_HEAD) → my pipeline reproduces
   Gap/Fear Panel 0.125 vs the oracle's 0.139 on identical rows. Closed.

Smoke-oracle operational note: scrub `run_id` from the sandbox RUNNING_STATS or
the smoke resumes the LIVE wandb run (one ~1-min pollution window happened
before this was caught; live charts may show a few rejected/out-of-order points
around 2026-07-19 13:50).

## Corrected results (fixed loader; all offline CPU, niced; subject = 27.5B full copy)

Offline behavior is now drastically healthier, validating the fix at the
behavioral level too: mirror engagement 76% (was 31% under the bug), goals/ep
0.75 (was 0.14), kickoff first-touch 2.5–2.9s (was 7.5s; healthy boot-probe
median ≈ 3.4s), airborne-frame share 51.7% (was 11.9%).

**League gate — the "current loses to all past selves" conviction is REFUTED.**
Corrected side-swapped cross-play: current 27.5B **48%** vs the 18.88B branch
point (161–174 wired_eval; 168–183 compare), **58%** vs 15.3B (218–160). The
pre-fix 40–43% losses (and the 26.5B/27.5B 40%↔64% whiplash) were artifacts.
What remains: dead-even against a self rated ~180 Elo lower ⇒ pool-myopia Elo
inflation is real but MODEST (~tens of Elo). The league-archive collapse
(3 members / 1 cell of 216, fitness-floor ratchet) is measured from live
telemetry and still stands — the spaced-anchor fix case is downgraded from
"urgent, convicted regression" to "preventive, structural".

**Wire dependence — none detected; the rating-latch integrity worry is defused.**
Same 27.5B brain: wire-zero beat wire-fed 66:34 (275–142, both halves
consistent); wire-zero ≈ old (48%) while wire-fed lost to old (36%). The policy
is NOT under-measured by wire-zero eval — zeros are at least as good. (The
full-wire deficit is bounded by pseudo-bank fidelity: V_metric/gap_PK were
rebuilt from offline-goal banks, gap_PK read ~0 vs live mean ~0.22–0.30, so the
14pp deficit reads as reconstruction noise in columns 4–5, consistent with the
live Wire Col Grad being tiny (1e-4–4e-4) — the policy barely keys on the wire
yet.) Rating measures the wire-zeroed policy faithfully and that is currently
the policy's BEST operating mode.

**Bounce representation — the earlier "weak, train an aux predictor" finding is
REVERSED.** Corrected probes (27.7B ckpt, 51.7k airborne frames, 180 episodes,
bounce = ballistic-miss >500uu at 20.2%): median landing error (uu), bounce
slice: trunk_h1 **737**, trunk_h2 **795**, raw_obs 1073, MLP-on-obs 1298,
ballistic ~1958. R² t_land: h2 0.840 > h1 0.829 >> raw 0.681. The trunk
linearly exposes a bounce model that beats both a linear read of raw physics
and a nonlinear MLP control on raw obs. The pre-fix "h2 worse than raw obs"
was the truncated h2. **An aux future-ball-state head is NOT convicted by
representation weakness** — the representation is strong; any remaining gap is
behavioral (knowing-doing), which is the Ladder's existing lane.

Caveats on the corrected numbers: ~120 episodes per pairing with episode-cluster
variance (~±8–10pp on shares); a large blue-side scoring asymmetry exists in the
offline env (both halves, both pairings — side-swap totals remain valid but
absolute per-half tallies don't compare across colors); pseudo-banks approximate
live banks (fed from offline self-play goals, 256/side, trainer fill-rule).

## Match-play addendum (same day) — the distribution mattered

The 48%/58% cross-play above ran on the TRAINER reset mix (drill/random spawns),
which is not match play and understates the current policy. Re-measured under
viz rules (`match_play_eval.py`: kickoff-only resets, goal-only terminals, 90s
safety cap, sides swapped; ~1000 goals per pairing):

- current 27.5B vs 18.88B: **502–420 = 54.4%** (both halves positive) → real
  match-play gain ≈ **+31 Elo** where Rating claims +187 (predicted 75% share).
- current 27.5B vs 15.3B: **556–451 = 55.2%** → real ≈ **+36 Elo** where Rating
  claims +327 (predicted 87%).

Verdict: improvement is REAL and shows at match play (the earlier "actively
worse / −14 Elo" reading is retracted — a drill-mix distribution artifact), but
Rating overstates the true match-play gain ~6×: pool-relative inflation
confirmed at scale (>5σ below prediction even with episode clustering). Real
slope ≈ +4 Elo/B over the last 12.2B steps. Instrument fix (permanent spaced
anchor battery, match-play protocol) is the priority; `match_play_eval.py` is
the prototype for that battery.

## Also fixed/learned in this incident

- `compare_checkpoints.py` never called `set_obs_size` → built 109-dim obs vs
  230-dim trunks (crashed on 5.0 checkpoints); fixed with an obs-width assert.
- `load_checkpoint.py` default root now prefers `checkpoints_5.0v3` (previously
  defaulted to the frozen pre-v3 `checkpoints_5.0` — the "every reading is
  stale" trap).
- `wired_eval.py` added: faithful offline ComputeWire replication (exp/map nets
  from the checkpoint, calib/dClamp from RUNNING_STATS, banksReady=false parity
  V_metric:=V_exp — NOT zeros — and trainer-rule pseudo-banks).
- Golden `best_r*` dirs rotate like everything else: `best_r1558` was evicted
  mid-read during this work. Copy-first applies to golden too.
- All `analysis/probes/*.md` behavioral results dated before 2026-07-19 were
  measured through the truncated trunk: treat as directional, re-measure before
  load-bearing use.
