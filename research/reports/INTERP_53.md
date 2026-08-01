# INTERP-53 — The interpretability battery, rerun against the four-rung critic ladder

**Date:** 2026-08-01 (overnight batch). **Run:** `5.3-geo` (wandb `137ff497`), checkpoint
**9,750,127,919** (~9.75B steps, PHASE B engaged at 2.36B, Rating/1v1 ≈ 1403 — inflated, see
`rating-pool-inflation`). **Status: RESULT.**

This is a rerun of the historical interpretability program (Phase 0 landing probes,
knowing-doing, rho calibration, whiff-tax credit probe, critic duel, fear decomposition,
mechanic/aerial censuses, opponent-model probes) against the current run, plus a new battery
built for what did not exist before: the **four-rung optimism ladder** (`V`, `V_exp`,
`V†` twins, `V◇` — GEOMETRIC_CRITIC.md). Every offline number in this file was produced by
the tools listed in §0; every comparison to "before" cites the historical result file or
report it comes from.

**Headline findings, in order of importance:**

1. **The geometric rung (V◇) is degenerate in production, and the causes are identified
   and causally ranked.** The field is nearly constant across visited states (std 0.06 on
   a 37.29 mean) *and* across hand-built frontier states (flip-reset setup vs mastered
   ground chase differ by 0.04), at every reference snapshot since 1.05B — while
   satisfying its HJB equation (residual ~3e-5, matching the live panel). Two defects:
   **93.2% of its Σ-weighted gradient norm lives in the 8 prevAction dims** (the
   validation testbed's 30-dim obs had no prevAction block — a transfer gap), and **the
   fitted local reward r̂ is flat ≈ 0 everywhere** (std 0.031, no correlation even with
   ball-rolling-into-net states). An offline causal split (re-solving fresh fields with
   and without the vacuous dims masked from Σ) shows the **flat r̂ is binding**: the
   masked field stays flat too — no reward topology, nothing to propagate. The injected
   geometric potential at 9.75B is, to first order, scaled noise anti-correlated with
   V_real (§2).
2. **The composition rung (V†) works as designed — its ZPD structure is real, riding on a
   large uniform offset.** H = relu(min(V†₁,V†₂) − V) ranks never-completed conducts
   (air-carry 1.91, flip-reset 1.93, aerial intercept 1.88) above mastered play (1.78) and
   correctly retires to 0.77 when the ball is rolling into the opponent net. But the field
   sits on a ~+1.8 pedestal covering 99.8% of frames (§1).
3. **The V† twins have effectively collapsed** (corr 0.9994, mean |Δ| 0.012 vs H ≈ 1.8;
   live panel `Vdag Twin Spread` ~0.006 since birth). The min-in-target anti-ratchet —
   the load-bearing safety mechanism of COMPOSITION_CRITIC.md §4.3 — is disarmed in this
   architecture (shared `critic_trunk`, two private layers per head). H is *bounded* over
   9.75B steps (no runaway), so the ratchet has not fired — but the guard that was supposed
   to stop it is not functionally present (§1.2).
4. **The whiff tax is gone.** The 4.0-era credit probe measured the critic pricing
   *declining* an aerial race above pursuing it (pricing_V = −0.053 ± 0.022 z −0.11). The
   same probe, same seed and matching, now reads **+0.117 ± 0.016 (z +0.32)** — the critic
   now prices pursuit far above declining, and every rung agrees (§3.4).
5. **The reachability self-model became dramatically better calibrated**: car-head AUC
   0.70 → **0.92**, ball-head 0.74 → **0.87**, Spearman 0.36/0.48 → 0.75/0.67. "Coarse
   compass, bin only" (RHO_CALIBRATION.md) no longer describes it (§3.3).
6. **The knowing–doing gap, measured behaviorally, has inverted** — attendance of free
   landings is now *graded by knowledge* in the direction competence predicts (§3.2), and
   the trunk's landing representation is at or above the best historical read (§3.1).

---

## 0. Methods and validity

### 0.1 New tooling (all in `research/tools/`)

The 5.3 lineage broke every assumption in the old offline loader (residual BroNet stacks,
a second value-side trunk, four value heads, GAP_EXP, three geo nets). New:

- **`load_checkpoint_53.py`** — eager rebuild of all 15 model files. Construction is a
  line-for-line port of `GGL::Model::Model` (Models.cpp), including the two
  archive-invisible structural facts: trailing activations on the two trunk bodies
  (the H2_TRUNCATION bug class) and **residual skip spans** (flat module list + recorded
  (start,end) spans, added before the block's trailing activation). Structural asserts
  verify the archive's param-index pattern against the predicted module plan for every
  net; drift → hard error (this fired correctly when a stale 5.0v3 checkpoint was fed in).
- **`collect_dataset_53.py`** — trainer-parity self-play collector (tickSkip 8,
  actionDelay 0, live reset mix incl. a port of `AirPlayState`), records the full ladder
  per frame. 160k player-frames, 251 episodes, 1v1.
- **`compat53.py`** — import shim that lets the historical battery (steer_team consumers)
  run against the new loader unmodified.
- **`kickoff_health_53.py`, `critic_ladder_analysis.py`, `geo_field_probe.py`,
  `ladder_credit_probe.py`, `ref_sweep.py`, `pull_wandb_53.py`** — this file cites them
  where used.

### 0.2 Loader verification (all passed)

| check | result |
|---|---|
| structural asserts, 15/15 nets | param indices match predicted plan exactly |
| kickoff behavior probe (the offline twin of the boot sanity probe) | **12/12 touches, median 3.30s** (healthy signature ≈ 10/10, ~3.4s) |
| `Headroom/H Mean` | offline 1.81 vs live 1.99 (state-mix difference: live is ⅓ each 1v1/2v2/3v3) |
| `Headroom/H P90` | 2.29 vs 2.29 |
| `Headroom/Vdag Mean` | 1.97 vs 2.02 |
| `Geo/V Mean` | 37.29 vs 40.17 (recovered post-PHASE-B; mode-mix difference) |
| on-policy HJB residual (p50) | 4.7e-5 vs live `Geo/Residual` 2.9e-5 |
| live `Geo/H Geo Mean` 0.22 | reproduced once the **batch-affine matching** in the actuation site was ported (raw-unit gap reads ~37 — a trap for future readers) |

### 0.3 Caveats that bound every claim below

- Offline sim is pip RocketSim 2.2.1 (v2 physics) vs the trainer's vendored Rust v3 —
  the long-standing accepted difference of this toolkit.
- All offline collection is **1v1**; the live run is ⅓ 1v1 / ⅓ 2v2 / ⅓ 3v3 (PHASE B).
  The credit probe (§3.4) is 2v2 for parity with its 4.0-era original.
- Cross-run comparisons (3.1 @ ~27B, 4.0 @ ~27–30B, 5.0v3 @ 29–45B vs 5.3 @ 9.75B)
  compare **different lineages at different maturities on different architectures**; they
  are trajectory comparisons, not ablations. Where the historical number predates the
  2026-07-19 h2-truncation fix it is flagged `h2`.
- Behavioral metrics carry episode-cluster variance ~5× binomial; the 160k-frame dataset
  holds 251 episodes — enough for the field-level reads here, marginal for small slices.

---

## 1. The ladder at 9.75B (new battery)

Source: `critic_ladder_analysis.py` → `results/critic_ladder_9750127919.json`, on 160k
on-policy frames; state-sweep numbers from `geo_field_probe.py` →
`results/geo_field_probe_9750127919.json` (300 jittered states per family).

### 1.1 Ordering and calibration

The epistemic ordering V ≤ V_exp ≤ V†min holds where it should and fails at sane rates:
V_exp < V on 7.3% of frames (expectile τ=0.8 upper tail), V†min < V on 0.2%,
V†min < V_exp on 0.6%. Means: V +0.16, V_exp +0.37, V†min +1.97, V_goal +0.15 (all in
GAE-normalized return units).

**The rungs' gaps behave as three different quantities, as the papers intended:**

| field | mean | character |
|---|---|---|
| kd_gap = V_exp − V | +0.21 | small, **spatially diffuse** — flat across every feature bin (0.19–0.23), exactly the "inconsistency" signal COMPOSITION_CRITIC.md describes |
| H = relu(V†min − V) | +1.81 | large offset + real ZPD structure (§1.2) |
| h_geo (affine) | +0.29 | ≈ scaled noise ∝ −V_real (§2) |

### 1.2 The composition field: real structure on a large pedestal

On hand-built state families (`geo_field_probe.py`):

| family | V_real | V†min | **H** | note |
|---|---|---|---|---|
| C air-dribble carry | 0.18 | 2.09 | **1.91** | never-completed conduct — highest H |
| D flip-reset setup | 0.27 | 2.19 | **1.93** | never-completed conduct |
| B aerial intercept | 0.37 | 2.24 | 1.88 | drilled, partially converting |
| A ground under high ball | 0.20 | 1.99 | 1.79 | |
| F mastered ground chase | 0.17 | 1.95 | 1.78 | baseline |
| E ball rolling into opp net | **1.11** | 1.88 | **0.77** | value nearly cashed — H retires |

The ordering frontier-conducts > mastered > nearly-cashed is exactly the self-retiring
ZPD field the paper claims. Qualifications:

- **The structure (~±0.15) rides on a ~+1.8 pedestal present on 99.8% of frames.** On
  on-policy data H's largest correlate is −V_real (−0.51), then ball-vel-toward-own-goal
  (−0.31), low boost (−0.25), distance from ball (+0.14). Air vs ground means differ by
  0.04. So the *injected* seek drive is mostly "V is low here" plus a modest genuine
  frontier tilt. (Actuation note: the live mix injects the V† *level*, not H; the level
  carries the same structure — B/C/D top the level table too.)
- **The twins are not two functions.** corr(V†₁, V†₂) = 0.9994, mean |Δ| = 0.0116 (live
  panel: 0.118 at the first sample after birth, ~0.008 by ~25M steps, flat since — the
  collapse happened almost immediately). The `ExampleMain` comment calls two private layers per head
  "a floor, not a tuning choice" for exactly this reason; the floor is not holding. The
  min() is then ~the identity, i.e. the §4.4 anti-ratchet is disarmed. What we can say
  after 9.75B steps: H has stayed bounded (band 1.5–2.9 all run, no trend) — the ratchet
  has not fired even without its guard. Whether the ~1.8 pedestal *is* slow, saturated
  ratchet inflation or honest τ=0.75 optimism cannot be separated from this data alone;
  the pedestal's stability over 9B steps argues saturation either way.
- `Headroom/Vdag Update Magnitude` reads 0.22–0.38 all run — the lr=0 incident
  (2026-07-25, 6M-net lineage) is **not** present in 5.3.

### 1.3 V_exp is the most aerial-sensitive rung

Unexpected: in the ball-height sweep (family A), the rung that tracks ball_z best is
**V_exp** (corr 0.457), not V (0.09), V† (0.07), or V◇ (0.11). The upper tail of realized
returns already knows high balls sometimes pay — a nice confirmation that "what I
sometimes do" is the right description of rung 2, and a hint that the expectile twin
carries aerial-value information the mean critic discards.

## 2. The geometric rung is degenerate in production

Three independent measurements, one mechanism:

1. **Flat on-policy.** V◇ std = 0.062 on mean 37.29 across 160k frames; binned by ball
   height, car-ball distance, ball-to-goal distance, the bin means span 37.285–37.305.
   corr(V◇, V_real) = −0.11.
2. **Flat off-policy.** Across the six hand-built families of §1.2 — including states the
   run has never produced — between-family std is 0.025, *smaller than* the within-family
   std 0.033. The field cannot tell a flip-reset from a kickoff.
3. **The gradient decomposition.** E‖∇V◇‖²_Σ = 0.0142 (norm 0.119 — exactly the level
   that balances (1−γ)·37.3 against r̂ ≈ 0 in Eq. 4, i.e. the *level* of the field is set
   by the gradient norm). Split by obs group:
   prevAction **93.2%**, opponent slots 6.2%, self 0.45%, ball 0.02%, everything else <0.2%.
   The top Σ dims are prevAction throttle/steer/pitch/yaw/roll (σ 0.27–0.42 vs 0.07
   overall) — one-step action changes *are* huge "displacements" in obs space.

**Why the testbed didn't catch it:** GEOMETRIC_CRITIC.md Appendix A's obs is 30-dim —
car pose/velocity, ball pos/vel, boost, flip state. **No prevAction block.** The
production obs (AdvancedObsPadded, 230-dim) leads with an 8-dim prevAction block at
indices 9–16. The HJB residual wants ‖∇V‖_Σ large; Σ is largest, by an order of
magnitude, along dims whose "reachability" is vacuous (the agent freely rewrites its own
last action); the PINN found that solution. The consequence chain: the field needs no
physical structure to satisfy its PDE → it goes flat in the physical dims → the
affine-matched gap `relu(scale(V◇) − V)` (live `Geo/H Geo Mean` 0.22) amplifies what is
left — tiny, ~physically-meaningless variation — to V_real's scale, producing a potential
that is ≈ noise slightly ∝ −V_real (corr(h_geo, V_real) = −0.67; positive-fraction 48.6%,
what an independent-noise gap would give). Half of `Phi_mix` is this term.

**A second contributor, independent of Σ: r̂ is nearly flat.** The fitted local reward
reads mean −0.000, std 0.031, with ~zero correlation against every feature including
ball-to-goal distance (+0.008) — even on hand-built states with the ball rolling into
the opponent net (family E: r̂ = −0.010). The MSE fit over reservoir arrival states has
smoothed the sparse ±goal events and near-zero-mean dense shaping into ≈0 everywhere.
Eq. 4's magnitude story ("value flows backward along corridors from r̂") has almost no
reward topology to propagate at production scale; γ = 0.9969 amplifies whatever tiny
r̂ structure survives by 1/(1−γ) ≈ 323, which sets the field's *level* but evidently
not a usable *shape*.

**The causal split was run** (`masked_geo_resolve.py`): two fresh V◇ fields re-solved
offline against the SAME frozen r̂/Σ (same architecture, γ, residual, Adam 1e-3, 8
epochs on 120k on-policy states) — one with the production Σ-norm, one with Σ masked to
physical dims only (ball 0:9, self 51:80, opponent car 138:167). Both satisfy the PDE
(resid² ~9e-4); **both come out flat across the probe families** (between-family std:
full 0.0012, masked 0.0070; both below their within-family stds). Masking the vacuous
dims moves structure ~6× in the right direction but restores nothing usable — because
with r̂ ≈ 0 everywhere there is no reward topology for the field equation to propagate,
whatever carries the gradient norm. **The flat reward model is the binding cause; the
prevAction channel is where the degenerate solution routes, not why it exists.** (Caveat:
an 8-epoch from-scratch solve is far shallower than the live field's 24k iterations —
read the arms relative to each other, not to the live field.)

**A third, smaller defect found while explaining Σ:** `GeoReservoirAdd` (Learner.cpp)
feeds consecutive buffer rows as (s, s′) without excluding pairs that straddle episode
boundaries (only the *reward* is masked by `cont`). ~1/450 of reservoir pairs are
teleports between different episodes/arenas/players. This is why presence-flag dims
(225–229) — which cannot change within an episode — carry σ ≈ 0.27–0.31, among the
largest in the obs. It contaminates Σ everywhere, but the prevAction effect above
dominates regardless (prevAction σ is legitimately large within episodes).

**What this does and does not say about the live run.** It does *not* say the geo rung
hurt: at β = 0.04, σ-matched and clamped, half the mix being noise is a small entropy-like
perturbation (and the cold-start window, where the paper's wins live, is long past — the
4.7× early-air result is a claim about 0–3M steps; 5.3's own early trajectory is
consistent with the mechanism having been *non-degenerate* early: `Geo/V Mean` moved
36→50→28→38 through PHASE B, i.e. the field did re-solve when the environment changed).
It says the **fourth rung's information source is currently disconnected**: whatever
"what may be possible" signal exists at 9.75B, V◇ is not carrying it.

**Concrete fixes, ordered by the causal evidence** (all offline-testable with
`geo_field_probe.py` + `masked_geo_resolve.py` before touching the trainer):
1. **Give r̂ structure back** — the binding fix. The MSE fit on smoothed sparse rewards
   is the wrong estimator for a quantity whose entire useful content is its rare tail
   (goals). Candidates to test offline: fit r̂ on unnormalized reward with the goal
   events upweighted, or an expectile/quantile loss, or a separate sparse-event head.
   Success criterion (pre-register): family E (ball rolling into net) must price above
   family F in r̂, and the re-solved field must clear between > within on the probe
   families.
2. Exclude prevAction/pad/presence dims from the Σ-norm (necessary hygiene — the §2
   decomposition shows the solution routes through them — but demonstrated insufficient
   alone).
3. Skip boundary-straddling pairs in `GeoReservoirAdd` (cheap, strictly correct).
4. Re-examine the affine-matched *gap* form: when V◇'s cross-state variance is far below
   V's, the current normalization guarantees the injected dose is dominated by whichever
   field has the least real structure.

## 3. The classic battery, then vs now

### 3.1 Landing probes (Phase 0 rerun)

`train_probes.py` unchanged (92,972 labeled airborne frames, 24.7% bounce slice,
episode-grouped CV, all controls green: shuffled R² ≈ 0, h1/h2 passthrough gates 0.99/0.98).

| source | R² x/y/t | median landing err | bounce / direct |
|---|---|---|---|
| raw obs (ridge) | 0.888/0.926/0.679 | 683 uu | 1146 / 614 |
| trunk h1 | 0.954/0.971/0.810 | 438 uu | 734 / 385 |
| **trunk h2** | **0.969/0.980/0.848** | **369 uu** | **621 / 322** |
| reach_phi | 0.840/0.844/0.744 | 1221 uu | — |
| raw obs (MLP control) | 0.881/0.914/0.698 | 805 uu | — |

Against the 3.1 @ 27.7B read (h2 0.937/0.966/0.840): better on every axis, with the
trunk now beating raw observations by +0.08/+0.05/+0.17 R² and halving the median
landing error — the linear margin over obs is the *learned* representation content, and
it clears both the MLP-on-obs control and the ref_0 random-feature baseline (§4) by a
wide gap. The Phase 0 conclusion ("modest bounce-adjusted landing info") has become
"strong, including through the bounce" — bounce-slice median error 621 uu vs the raw-obs
1146 uu.

### 3.2 Knowing–doing gap — inverted from flat to knowledge-graded

`knowing_doing.py`, unchanged design (92,972 readings, 21,730 feasible free landings).
The founding 3.1 measurement (KNOWING_DOING.md, ckpt 1.46B): P(unattended | trunk knows
the landing) = 66.0% vs 71.8% when it doesn't — *flat across knowledge quartiles Q1–Q3*,
the result that classified the gap as incentive-limited, not knowledge-limited.

Now: **P(unattended | knows) = 55.5% vs 67.5%** — a 12pp knowledge gradient, monotone
across quartiles (Q1 53.9% → Q4 73.2%, with Q4 also the far/hard slice at median 961uu).
Attendance is now graded by read quality exactly as a knowledge-coupled system should be
— the state the 3.1 program only reached *after* the steering interventions
("knowledge-COUPLED, graded by read quality") is the resting state of this lineage, with
no activation machinery involved. Residual aerial passivity is still visible: of 15,166
known high balls within 2500uu, 55.7% are never attacked airborne before touchdown
(3.1's comparable read: 60.1%).

### 3.3 Rho calibration — the self-model grew up

`calibrate_rho.py` (unchanged design: 60k rows 2v2, future/jitter/cross goal menus,
ε-sweep). Then (3.1 @ 2.67B, RHO_CALIBRATION.md) vs now (5.3 @ 9.75B):

| head | AUC then | **AUC now** | Spearman then | **now** |
|---|---|---|---|---|
| car (window 20) | 0.701 | **0.912–0.919** | 0.358 | **0.752** |
| ball (window 90) | 0.737 | **0.866–0.875** | 0.481 | **0.666** |

The standing doctrine "rho is a coarse compass — bin it, never trust a single value" was
written for AUC ~0.7. At 0.87–0.92 with monotone deciles, rho is a genuinely
discriminative reachability estimate. (Architecture and lineage both changed — 384×3
residual heads vs 256×2, and reach accuracy panels live read Ball 0.61 / Car 0.83 — so
this is trajectory, not ablation. The carstate head remains weakest: live accuracy 0.24.)

### 3.4 The whiff tax is gone (ladder credit probe)

`ladder_credit_probe.py` — a faithful port of `credit_probe.py` (same seed, same 900k-row
2v2 rollout design, same matched split within req_self × t_land × margin tercile cells,
episode-cluster bootstrap): 8,414 aerial-ball readings, 4,216 best-placed feasible,
1,724 matched pairs per class.

| head | pursue − decline (raw ± se) | z | 4.0 @ 27.55B (CREDIT_PROBE.md) |
|---|---|---|---|
| **V_real** | **+0.117 ± 0.016** | **+0.32** | **−0.053 ± 0.022 (z −0.11)** |
| V_exp | +0.109 ± 0.015 | +0.30 | — |
| V†min | +0.044 ± 0.011 | +0.15 | — |
| V_goal | +0.049 ± 0.014 | +0.17 | +0.031 ± 0.013 (z +0.09) |
| V◇ | +0.005 ± 0.003 | +0.06 | — |
| H | −0.073 ± 0.017 | −0.20 | — |

The founding pathology of the credit program — the critic pricing *declining* a winnable
aerial race above pursuing it, the internal signature of PPO-taught avoidance — has not
just closed but reversed at ~7σ: every value rung now prices pursuit above declining,
the mean critic most of all. The 4.0-era "horizon disagreement" (goal critic favoring
pursuit against a reluctant dense critic, Δz +0.305) has correspondingly vanished —
the goal critic no longer needs to out-vote the dense critic (horizon Δz now −0.15,
i.e. the dense critic is the *more* pursuit-favorable head).

This result supersedes the fear-mining program's premise for this lineage: FEAR_DECOMP /
CRITIC_DUEL / FEAR_MINE were built to locate and exploit states where V was *anomalously
pessimistic about pursuing* — that population is no longer detectable at the population
level, so those two reruns were intentionally not carried over verbatim (their selector
Δz = z_G − z_V has also lost its premise: the dense critic is now the more
pursuit-favorable head).

**H reads higher on the declined branch (z −0.20)** — which is the correct semantics for
"provable-but-uncollected": a feasible race the policy declines *is* retained headroom.
Since the seek potential pays for moving up the H field, the mechanism points at exactly
the branches the policy declines. Whether that pressure is what closed the tax cannot be
established from one lineage — but the sign structure is precisely what
COMPOSITION_CRITIC.md §5 predicts.

### 3.5 Goal critic (duel + live panels vs the 42.2B audit)

On-dataset: corr(V, V_goal) = 0.802 (Spearman 0.64). The 42.2B 5.0v3 audit measured
z-corr 0.942 (on a 2v2 rollout; mine is the 1v1 mixed-reset dataset, so read the gap as
indicative, not exact) — the goal critic is **less redundant** with the dense critic in 5.3 despite
now *sharing a critic trunk* (2-layer private heads on shared features, vs an independent
raw-obs net before). Its outcome-tracking is unchanged: live `GoalCritic/Value-Outcome
Corr` 0.469 vs the audit's 0.459; Adv-Outcome 0.386.

**Where they disagree is boost- and tempo-shaped.** The goal-critic-favored decile carries
boost 60 vs 14 in the critic-favored decile, speed 1679 vs 1209, car-z 162 vs 76, ball
moving toward the opponent net vs away. The long-horizon head prices *energy* (boost +
speed + altitude) far above the dense critic — consistent with CarEnergyPotential's tempo
thesis living mostly in the dense stack's shaping while the goal head prices its actual
long-run cash-out.

### 3.6 Mechanic census

| per 100k steps | 5.0v3 @ 29.3B `h2-clean` | 5.0v3 @ 45.4B | **5.3 @ 9.75B** |
|---|---|---|---|
| wavedash-like | 165 | 147 | **143** |
| flip-cancel-ish | — | 1585 | **1500** |
| proto-dribble touches | 291 | 587 | **537** |
| grounded jump at high ball | — | 151 | **176** |

The 5.3 lineage at 9.75B has the fragment profile 5.0v3 had at ~45B — the fragment
economy that took the previous lineage its whole run to build is present at ~1/5 the
steps. (Different lineage/rewards/architecture; still, this is the pattern the
composition critic was predicted to produce: fragments first.)

### 3.7 Aerial gap census

| metric | 5.0v3 @ 29.3B | 5.0v3 @ 45.4B | **5.3 @ 9.75B** |
|---|---|---|---|
| M1 1v1 true-aerial touch share | — | 1.1% | 1.0% |
| M2 opportunity: left ground | — | 33% | 33% |
| M2 aerial-touch conversion | — | 7.3% | 5.9% |
| M2 mean boost at opportunity | 54→18 (energy-dump era) | 21 | **39** |
| M3 drill (airborne spawn) touch | 38.7% | 62% | 60% |
| M4 takeoff (grounded spawn) aerial touch | 0.67% | 14% | **4%** |
| M4 jump within 1s | — | 27% | 16% |

Read as a trajectory: at 9.75B the 5.3 bot converts airborne-spawn drills at the level
5.0v3 reached only at 45B (60% vs 62%), holds nearly twice the boost at aerial
opportunities (39 vs 21), but its ground-takeoff chain (M4) is at an intermediate stage
(4% — 6× the 29.3B figure, ~1/3 the 45.4B one). The jump→climb transition is again the
binding constraint, now with better fuel discipline.

### 3.8 Opponent-model probe

`opp_model_probe.py` unchanged (160k rows). Then (4.0 @ 27.55B `h2`) vs now:

| probe | then | **now** |
|---|---|---|
| C2 "opponent reaches ball first" AUC (h2 / raw obs) | 0.890 / 0.588 | **0.939 / 0.755** |
| A3 premeditation ("will I touch within 1s") AUC (h2 / obs) | 0.868 / — | **0.971 / 0.851** |
| C1 opp position +0.5s R² (h2, x/y/z) | 0.955 / 0.959 / 0.815 | 0.972 / 0.981 / 0.796 |

The trunk's *decision-relevant* opponent model (who wins the race) and its premeditation
signal both strengthened substantially; its literal opponent-position regression is
unchanged (and still slightly below raw obs on z, as it always was — the trunk compresses
kinematics, it doesn't out-predict its own input). The premeditation direction — the
read-write channel the whole steering program was built on — is now close to saturation
as a *readout* (0.97).

## 4. Through-training (reference-set sweep)

`ref_sweep.py`: 40k trainer-parity frames per frozen reference snapshot (the log-spaced,
never-trained-against set), behavioral + representation + geo reads per point.

| ckpt | touch/100 steps¹ | air-prox rate | mean boost | landing R² x/y/t (h2) | V◇ std | corr(V◇, ball_z) |
|---|---|---|---|---|---|---|
| ref_0 (birth) | 0.46 | 0.28% | 10.4 | 0.789/0.894/0.475² | 0.208 | −0.01 |
| 1.05B | 3.75 | 0.74% | 39.9 | 0.968/0.973/0.827 | 0.090 | −0.11 |
| 2.75B | 3.81 | 0.83% | 30.4 | 0.964/0.975/0.870 | 0.162 | +0.02 |
| 4.43B | 4.31 | 0.99% | 36.9 | 0.963/0.980/0.833 | 0.053 | −0.01 |
| 5.48B | 4.05 | 0.93% | 33.4 | 0.966/0.974/0.831 | 0.061 | −0.00 |
| 7.13B | 4.46 | 1.06% | 37.4 | 0.976/0.979/0.853 | 0.078 | +0.02 |
| 8.83B | 4.63 | 1.05% | 41.1 | 0.969/0.974/0.858 | 0.063 | +0.08 |
| 9.75B (head) | 4.28 | 1.18% | 37.7 | 0.957/0.975/0.835 | 0.061 | +0.03 |

¹ touch flags carry a uniform +~0.35 inflation (first-frame-after-reset artifact in the
collector's touched flag); trajectory shape unaffected. The sweep's kickoff columns are
invalidated by the same artifact — the valid kickoff read is `kickoff_health_53.py`
(head: 12/12, median 3.30s).
² a *random* 1152-wide trunk already yields ridge R² 0.79/0.89 on landing x/y — the
random-feature baseline that all landing-probe numbers must be read against (the trunk's
*learned* contribution is the margin over this and over raw obs, and the bounce slice).

Three reads:

- **The trunk's landing representation saturates by ~1B** and is flat to 9.75B at a level
  matching the best 3.1-era read (27.7B: 0.937/0.966/0.840 — `metrics.json`), reached
  ~25× earlier in steps.
- **Aerial engagement climbs monotonically** (air-proximity 0.28% → 1.18%) while ground
  touch rate plateaus after ~4B — the marginal behavior growth is in the air, which is
  what the scaffold weights + composition credit are for.
- **V◇ was never structured in production.** Flat (std ≤ 0.16 on |mean| 30–45) with
  ~zero ball_z correlation at *every* snapshot from 1.05B on, including the 2.75B point
  just after the PHASE B flip where `Geo/V Mean` was re-solving (28→38): the level moved,
  the spatial structure never appeared. The §2 degeneracy is a birth defect of the
  production deployment, not late decay.

### 4.1 Run-health context (live telemetry, `pull_wandb_53.py`)

- **Plasticity**: `Trunk EffRank` 836 → 822 over 9.75B (essentially flat), `Policy
  EffRank` 80 → 74, `Policy Dead Units` **0 all run** — the residual architecture is
  doing what the `45a59d5` rationale asked of it; no sign of the effective-rank decay
  that motivated it.
- **External yardstick**: cumulative Nexto goal share 54.4% (921,810 : 770,837), but the
  *marginal* share over 8B → 9.75B is **~75%** (+134k for vs +44k against) — the
  in-training-env Nexto read (not a match-play claim; see `nexto-opponent-live`) has been
  accelerating through PHASE B, where the 5.1 lineage ended at ~48-49%.
- `Ref/Oldest Share` 0.975 vs the frozen birth network — the only inflation-proof
  internal skill series; monotone all run.

## 5. What changed since the last program, in one table

| question | before | now (5.3 @ 9.75B) |
|---|---|---|
| Does the critic tax attempting? | yes — pricing_V −0.053 (whiff tax, 4.0 @ 27.6B) | reversed: pricing_V **+0.117 ± 0.016**; all rungs favor pursuit |
| Is rho usable per-goal? | no — AUC ~0.7, "bin only" | mostly yes — AUC 0.87–0.92 |
| Is optimism represented? | no dedicated machinery (steering era: activation edits) | 3 live optimism fields; V† structured ZPD ✔, V_exp aerial-sensitive ✔, V◇ degenerate ✘ |
| Is the anti-ratchet armed? | n/a (single-head era diverged) | twins collapsed (corr 0.9994) — guard nominally present, functionally ~absent; H bounded anyway |
| Goal critic redundant? | z-corr 0.942 (5.0v3 @ 42.2B) | corr 0.80 — more independent, boost/tempo-shaped disagreement |
| Fragment economy | built up over ~45B (5.0v3) | present by 9.75B |
| Knowing-doing | flat → incentive-limited (3.1) | knowledge-graded: 55.5% vs 67.5%, monotone by quartile |
| Race prediction / premeditation | AUC 0.890 / 0.868 (4.0) | AUC 0.939 / 0.971 |

## 6. Recommended follow-ups (measurement-before-machinery compliant)

1. **Geo r̂ re-estimation** (offline first): the §2 fix list, in its causal order — the
   Σ-mask experiment was already run tonight and moved structure 6× without restoring it;
   the binding lever is a reward model that can see its own sparse tail. Success
   criteria pre-registered in §2.
2. **Boundary-pair fix** in `GeoReservoirAdd` (cheap, strictly correct).
3. **Twin de-collapse probe**: measure whether staggered target-copy schedules or
   decorrelated minibatch sampling for vdag1/vdag2 restores a nonzero min-gap, offline,
   before touching the live config. If the pedestal is saturated inflation, a functional
   min should pull H's offset down without touching its ZPD structure.
4. **V_exp as an aerial-value read**: its ball-z sensitivity (0.46) suggests
   `Gap/*`-style telemetry binned by ball height would give a live leading indicator of
   aerial value acquisition for free.

---
*Toolkit provenance: every number regenerable via the scripts in §0.1;*
*JSONs under `research/results/` keyed by checkpoint 9750127919.*
