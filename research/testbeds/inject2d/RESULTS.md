# inject2d results — 2026-08-05

All numbers: 8 seeds/arm, M4 Pro, ~50–70 s/run. Compare only within a batch
(the obs gained a dim between batch 1 and 2, which reshuffles seed luck).

## Batch 1 — injection channels (3M steps, static balls)

| arm | ignited | ign step | hi q4 | rew_f |
|---|---|---|---|---|
| sil | 8/8 | 0.99M | 5.15 | 2.83 |
| sil_ent | 8/8 | 1.25M | 5.39 | 2.88 |
| entgate | 8/8 | 1.41M | 4.68 | 2.55 |
| sil_nogate | 8/8 | 1.65M | 5.00 | 2.76 |
| align | 8/8 | 2.20M | 4.12 | 2.28 |
| kstep | 6/8 | 1.20M | 3.37 | 2.03 |
| ppo | 6/8 | 2.18M | 2.96 | 1.78 |
| sparse | 4/8 | 1.25M | 3.03 | 1.86 |
| sil_pbrs | 7/8 | 1.10M | 1.45 | 1.28 |
| pbrs (prod replica) | 5/8 | 1.11M | 1.04 | 1.08 |
| servo_raw | 5/8 | — | 0.88 | 0.97 |
| intr | 2/8 | — | 0.59 | 0.84 |
| pbrs_raw | 3/8 | — | 0.58 | 0.82 |
| decay | 4/8 | — | 0.46 | 0.64 |
| temp | 0/8 | — | 0.14 | −0.03 |

Findings:
1. **The potential/advantage channel ignites fast then CAPS** (per-seed: pbrs
   ignites earlier than ppo on every shared seed, converges to ~1/3 its hi).
   Adding SIL on top does not rescue it (sil_pbrs 1.45): swap, don't add.
2. **SIL and entgate win**; they compose (sil_ent) with no interference.
   The ladder's H gate = ignition speed (sil 0.99M vs sil_nogate 1.65M).
3. **Pricing a potential-diff neutralizes it** (intr): a potential difference is
   predictable from state, so its advantage under an accurate intrinsic critic
   → 0. "Priced" and "potential-based" are mutually canceling for seek.
4. **Sampler-side modification fails under PPO IS** (temp 0/8, aprior mixed).
5. Dose servo works both directions (raw fields under-delivered 4x here);
   kstep/sparse (SNR fixes) salvage some of the channel but don't beat sil.

## Batch 2 — gap closure (max-vs-avg = relu(V_exp − V), 3M steps)

| arm | ignited | ign step | hi q4 | rew_f |
|---|---|---|---|---|
| sil | 8/8 | 1.34M | 5.54 | 3.02 |
| gap_sil (elevation) | 8/8 | 1.38M | 5.03 | 2.77 |
| gap_pre (priced potential) | 6/8 | 1.20M | 4.08 | 2.67 |
| gap_post (advantage add) | 8/8 | 1.52M | 4.02 | 2.31 |
| ppo | 6/8 | 1.50M | 3.85 | 2.27 |
| gap_rew (plain penalty) | 0/8 | — | 0.07 | **−1712** |

Findings: the plain penalty is a self-referential doom loop (penalty lowers
returns → V falls → gap widens → penalty grows). The same quantity works
through non-self-referential channels; closure-by-elevation ≫ closure-by-
avoidance; nothing beats headroom-gated SIL.

## Batch 3 — meta-transfer (4M steps, frontier shift at 2.4M)

Phase A: static balls. Phase B: high-only falling balls (timed intercept).
silobs = policy obs carries tanh(V†−V_exp), tanh(V†/5); silobs_lp adds
learning-progress gating (relu(H_old − H_now), ladder snapshots every 10 it).

| arm | A-ign (M) | shift dip | recover-80% (iters) | B air final |
|---|---|---|---|---|
| sil (scaffold only) | 1.13 | 51% kept | 71 | 10.25 |
| silobs | 1.11 | 67% kept | **9** | 10.50 (1 seed collapsed: 0.3) |
| silobs_lp | 1.44 | 63% kept | 21 | **12.55** (worst seed 8.8) |

Findings: **the meta-transfer signature is real** — self-observing policies
lose less skill at the shift and recover 3–8× faster; LP-gating trades a
slower FIRST ignition for the best and most uniform post-shift performance
(no collapsed seeds). The silobs s6 collapse is the nonstationary-self-input
hazard surfacing exactly when self-estimates go stale (at the shift).
Caveats: the fully-new conduct (high intercept) ignited in no arm within the
1.6M post-shift budget — the differential is on skill recovery/transfer, not
from-scratch second-frontier acquisition; needs a longer phase B to test that.

## Batch 4 — foresight probe (2026-08-06, `foresight.py`)

Can any rung price a NEVER-SEEN conduct class? Base: 3 phase-1-only sil runs
(2.4M). Probes: 600 phase-2 states (falling balls). Oracle: best of 18
scripted intercepts in real env physics (13% feasible). Geo trio retrained
offline from the same frozen record 3 ways (on-policy residual / collocation /
collocation + mirror symmetry; Sigma and r-hat always fit on real pairs only).

Spearman rho vs oracle, mean of 3 seeds:

| estimator | rho | note |
|---|---|---|
| **V-dagger** | **+0.287** | best, consistent (+0.27..+0.30) |
| V | +0.103 | habit baseline |
| geo (on-policy) | **-0.199** | systematically INVERTED |
| geo (colloc) | -0.146 | collocation does not fix it |
| geo (colloc+sym) | -0.151 | symmetry adds nothing |

**Mechanism of the inversion**: intercept feasibility RISES with ball height
(more fall time) while the field's learned value structure FALLS with distance
(r-hat prices proximity/touch). The new conduct inverts the old phase's value
proxy, so the field's extrapolation is anti-informative, not just uninformative.
Deeper: feasibility hinges on the BALL's fall dynamics, which the phase-1
record contains nowhere -- Sigma cannot represent reachability of a moving
target it has never seen move. No residual-sampling scheme can conjure absent
dynamics; collocation only smooths.

**Consilience**: this explains WHY the batch-3 self-obs features worked -- they
were V-dagger-based (tanh(Vdag-Vexp)), and V-dagger is the rung that carries
transferable value structure (its record contained the component skills; its
generalization over the unseen bvz input preserved ranking). The 8x
re-acquisition and the +0.29 foresight are the same phenomenon measured twice.

**Bound on the negative**: the toy's phase 2 introduced dynamics NEVER
witnessed in any context -- harsher than production, where ball flight/fall
saturates the record and a flick's component dynamics all exist scattered
(the composition premise). The transferable warning is the PROXY INVERSION:
value structure learned in one phase can rank new-phase states backwards, so
geo-derived foresight signals must never gate exploration without a
V-dagger-side cross-check.

## Batch 4b — unified critic, first attempt (2026-08-06; pre-registered FAILURE)

Consolidation candidate: one twin-field family, executed-transition expectile
TD (the composition loss) as ANCHORS + the HJB residual (constant 0.3 weight,
collocation mix) as the PROPAGATOR. `unified_r0` additionally zeroes r-hat on
off-manifold residual rows. Pre-registered: success = rho >= Vdag's +0.29.

| estimator | rho vs oracle | |
|---|---|---|
| Vdag (anchors alone) | **+0.287** | |
| unified_r0 | -0.017 | anchors + residual, no off-manifold r-hat |
| unified | -0.053 | anchors + residual |
| geo (residual alone) | -0.199 | |

Anchoring CURED the inversion (-0.20 -> ~0) but the residual DESTROYED the
anchors' foresight. Mechanism: off-manifold, (1-gamma)V = gamma*||grad V||_Sigma
ties the field's level to its local slope using EXTRAPOLATED Sigma -- garbage
coefficients over-constrain the field exactly where the TD anchors' plain
neural generalization was already correct. The ranking is monotone in
off-manifold trust placed in the equation: none (Vdag) > some (unified) > all
(geo). **Where dynamics are unwitnessed, an explicit reachability constraint
with extrapolated coefficients is WORSE than the network's own inductive
bias.** The tracks-habit check corroborates: the residual pulls the field
toward on-manifold structure (rho vs V rises to +0.36).

Reshaped consolidation (untested): the residual weight must be TRUST-GATED by
data-proximity / Sigma-confidence -- high on-manifold (corridor structure,
cold-start pressure where anchors are absent), -> 0 off-manifold (leave anchor
generalization alone). I.e. the consolidation inverts: not geo absorbing
Vdag's anchors, but Vdag absorbing a trust-gated residual term. Open question
whether the residual then adds anything the SIL-era mechanisms still need;
the honest test is a batch-1-style training arm (SIL gated by unified-H vs by
h_mix) plus this foresight probe on the trust-gated variant.

## Batches 5–7 — the unified-critic search (2026-08-06; canonical writeup:
## research/reports/EPSILON_CRITIC.md)

Iterated estimator search for one critic unifying composition + geometry.
Failed (each measured, mechanism named): QRL quasimetric (distance collapse),
kernel with V†-style expectile training (1-step edges can't calibrate a
60-step metric), kernel on observed k-step record (+0.19, correct but
dilutes). Survivor: the **ε-relaxed Bellman operator** — V†'s exact training
with the bootstrap max'd over K perturbations of the REAL next state, scale =
ε · σ_emp (empirical marginal displacement std, one global vector, nothing
pointwise to extrapolate; unseen-dynamics axes are closed at 0 and open when
fragments are witnessed).

8-seed probes: wins the record-gap family on 8/8 seeds (+0.272 vs +0.242),
small smooth-regime tax (A −0.027, C −0.019); pad-env: wins the
stitching-required family (+0.197 vs +0.173), tax on smooth (−0.032).
Training loop (sil, 8 seeds): 8/8 vs 8/8, median ignition **1.11M vs 1.25M**,
finals 5.45/2.98 vs 5.37/2.92 — no regression, faster ignition, one laggard
seed (1.8). Consolidation: ladder → V / V_exp / V‡(ε); geo trio + Φ-mix
retired; ~15-line prod change at the vdag target site, zero new params.

Instrument note: the original oracle ran families B/C with phase-2 ball
dynamics (bug, fixed in metric5.oracle2); estimator comparisons unaffected.

## What transfers to production (claims, not proofs — RocketSim testbed next)

- Replace the Φ_mix advantage injection with headroom-gated SIL + the entropy
  gate (both hot-swappable: no new params, learn-pass only).
- The self-obs + LP design is a fresh-lineage feature (obs width) with a
  deployment story: prev-step features (prevAction idiom), trunk-coupled
  heads, mode-dropout for a clamped exploit mode at eval/deploy.
- Do NOT ship: plain gap penalties (self-referential), collection-time
  temperature, priced seek potentials, adding SIL on top of the live mix.

## Batches 8-9 -- the hull operator + adversarial ledger (2026-08-07)

Canonical: research/reports/EPSILON_CRITIC.md section 7. The record-licensed
hull operator (eps-scaled witnessed displacement vectors, chart-matched
donors) prices the zero-assembly family F (0.000% in every record) at 0.430
vs V-dagger 0.365 (8/8 seeds) and family B at 0.453 open / 0.514 gated. Five
attacks run; every harm channel closed (see the ledger in the report). Seat
theorem measured: open config = best trainer (ignition 1.02M, 8/8); gated
config = best estimator (hallucination negative on all seeds) but a bad
trainer (2.16M, 7/8) -- thin-record optimism is exploration in the actuation
seat and hallucination in the estimation seat, and SIL's success-only
consolidation makes the former safe.

## Batch 10 -- the value-critic study (2026-08-07)

V is the denominator of the whole program (H = V-ddag - V, SIL weights, GAE,
LP). Seven variants + composites on the sil_hull base, 8 seeds, two new
metrics: ev (explained variance of V, calibration) and h_floor (mean
relu(V-ddag - V), hallucinated-headroom floor).

| arm | ign | ign_M | hi_q4 | rew | ev | h_floor |
|---|---|---|---|---|---|---|
| sil_hull base | 7/8 | 1.25 | 4.91 | 2.72 | .679 | .027 |
| vh_aux (displacement-NLL aux) | 8/8 | **0.74** | **6.41** | **3.49** | .491 | .000 |
| vh_mir (mirror symmetry) | 8/8 | 1.10 | 5.89 | 3.19 | .671 | **.003** |
| vh_twin (differential pair) | 8/8 | 1.13 | 5.55 | 3.00 | .662 | .019 |
| vh_td (record Bellman-consistency) | 8/8 | 0.99 | 5.43 | 2.99 | .670 | .021 |
| vh_dec (channel heads) | 8/8 | 1.16 | 5.46 | 2.97 | .665 | .016 |
| vh_q (quantile) | 8/8 | 1.08 | 5.02 | 2.73 | .661 | .035 |
| vh_final (aux+mir+twin, NO td) | 8/8 | 0.75 | 6.36 | 3.47 | .484 | .000 |

Findings: (1) representation pressure (aux) is the largest end-to-end value
lever ever measured here (-40% ignition, +30% finals, zero laggard seeds) but
DEGRADES V's calibration when wired inside the value net -- the
encoder-illusion, live; in prod it belongs on the TRUNK. (2) Mirror symmetry
is the calibration champion: H-floor 9x down at intact ev -- the compounding
(sharper V -> deflated hallucinated headroom) confirmed. (3) Twin-differential
works as designed (noise cancellation, modest). (4) The TD-consistency term
biases V low and INFLATES the H-floor -- interaction measured, dropped.
(5) Composites don't stack naively; the final config is aux+mirror+twin.

Prod design ("composite value critic"): trunk-side displacement-NLL aux head
(clean of V), twin mirrored+slot-permuted value heads with mean readout, no
TD term, plus the prod-only privileged opponent-conditioning of the value
family (asymmetric actor-critic; removes opponent-mixture variance -- the
strategic-value limiter). Parked: quantile head (gap-sensor consolidation,
architectural), channel decomposition (mild, plumbing-heavy).

## Batch 11 -- GRPO-inspired arms + the closing combo (2026-08-07)

| arm | ign | ign_M | hi_q4 | rew | ev | note |
|---|---|---|---|---|---|---|
| vh_epi (episodic baseline) | 8/8 | 1.08 | 6.17 | 3.36 | .649 | near-final perf, calibration INTACT; epi_w self-anneals 0.50->0.15 |
| vh_grp (peer-group SIL referee) | 8/8 | 1.10 | 4.76 | 2.64 | .678 | REJECTED: sil_frac 0.751 -- stale uniform-history peer returns deflate the threshold, SIL clones mediocrity |
| vh_epifinal (final + epi) | 8/8 | 0.84 | **6.43** | **3.49** | .506 | best finals + best per-seed floor (6.1) of the whole study |

Lessons: (1) THE BANK GENERALIZES WHAT IS STATIONARY -- displacements are
physics and stay true forever (hull works); returns are policy-relative and
rot (peer-group referee fails); V_exp retrained fresh each iteration already
IS the recency-correct group quantile, amortized. (2) The episodic baseline
is the memory-backed cold-start backstop: authority handed to the parametric
critic exactly as fast as it earns it (blend w = 0.5*(1-EV_ema)). (3) In the
combo the aux-suppressed EV pins epi_w at ~0.25 permanently and it STILL
helps -- but prod should key the blend on the clean twin-head EV (the
suppression is a toy-wiring artifact; prod aux lives on the trunk).

FINAL VALUE-CRITIC DESIGN (prod): trunk displacement-NLL aux + twin
mirrored/slot-permuted value heads (mean readout, |V1-V2| gauge) + episodic
baseline blend keyed to twin-head EV + privileged opponent conditioning.
No TD term. Toy evidence: ignition 1.25->0.75-0.84M, finals 4.91->6.4,
worst seed 0.1->6.1, vs the sil_hull base.
