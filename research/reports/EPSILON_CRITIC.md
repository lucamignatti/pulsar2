# The ε-Relaxed Bellman Operator: One Critic for Composition and Geometry

**Status: CANONICAL for the unified-critic design (2026-08-06; §7 hull
revision 2026-08-07 — the final mechanism is §7's record-licensed hull
operator; §2's Gaussian form is its ancestor and remains the correct
theory frame). Validated at inject2d testbed scale; production deployment
pending a fresh-lineage or hot-swap decision. Supersedes the two-rung
(V† + V_geo) achievable-value architecture of `GEOMETRIC_CRITIC.md` §5–6.**

---

## 1. The problem

The ladder's two upper rungs estimate the same thing — achievable value — from
different evidence: V† chains executed transitions (correct, record-bound);
V_geo chains local reachability statistics (manifold-free in intent). The
foresight-probe program (`research/testbeds/inject2d`, batches 4–4b) measured
the split under transfer: **compositional headroom generalizes; geometric
headroom inverts** (ρ vs oracle on a never-seen conduct: V† +0.29, Σ-HJB field
−0.20; anchoring the field cured the inversion but the residual then destroyed
the anchors' own foresight, −0.02). The failure is structural: the HJB's
coefficients (Σ(s)) are pointwise local statistics that must be *extrapolated*
off-manifold, and an equation enforced with extrapolated coefficients
over-constrains exactly where plain anchored generalization was already right.

Requirement (user-set): one critic that unifies both purposes — correct like
the composition critic, geometry-generalizing like the field was meant to be,
manifold-free, fast (no models, no search), theoretically clean, and feeding
the learning-aware stack (H, LP, SIL gate, self-obs) unchanged.

## 2. The operator

Both rungs are estimators of the discounted reachability kernel
Γ(s,a) = γ^{d(s,a)}: V† restricts d to sampled one-step edges; the HJB field
solves V = C·γ^{d_Σ} for the Riemannian distance under the Σ-metric. The
unification keeps the kernel but moves the geometry INSIDE the Bellman
operator, where every application is anchored at an executed transition:

    (T_ε V)(s) = E_(s,r,s') [ r + γ · max_{ã ∈ B_ε(s')} minTwin V(ã) ]

    B_ε(s') = { s' + ε · σ_emp ⊙ z },   σ_emp = per-dim std of executed
    one-step displacements over the record (ONE global vector)

trained exactly as V† is trained: expectile TD (τ=0.75), twin heads,
min-in-target, target nets, clamped targets — with K perturbation samples of
the *real* next state added to the bootstrap max (K=4–8), each projected back
to a physically coherent observation.

Properties, all load-bearing:

- **ε = 0 recovers the composition critic exactly.** One-parameter family,
  monotone in optimism; correctness is the pole, not a hope.
- **The geometry cannot be wrong pointwise, because it is not pointwise.**
  σ_emp is a marginal statistic of the whole record — there is no
  state-conditioned coefficient to extrapolate. This is the surviving residue
  of Σ after its measured failure mode is removed.
- **Axes open exactly when the record licenses them.** σ_emp on a dimension
  the record has never seen move (ball-fall before phase 2) is 0; the moment
  fragments of new dynamics appear anywhere — conduct never performed — the
  ellipsoid opens on that axis (measured: E4 exposure).
- **T_ε is a γ-contraction** (max over a compact perturbation set of a
  contraction), so the fixed point exists and training is as stable as V†'s;
  the fixed point is the value of the ε-relaxed MDP (dynamics granted ε·σ_emp
  of slack per decision) — a principled optimistic envelope, NOT unfalsifiable
  optimism: every target asserts "a state within ε of one I actually reached
  is worth y," refuted by the growing record the ordinary way.
- **Micro-stitching compounds.** Each bootstrap application crosses ε of
  record-gap; TD iteration composes crossings into corridor-length reach —
  the HJB's propagation mechanism with V†'s epistemology.
- **Fast.** K extra head forwards per target batch. No world model, no
  search, no input gradients, no reservoir, no σ/r̂ nets: it is CHEAPER than
  the geometric trio it replaces.

## 3. The failure map that forced it (each arm measured, 3 seeds)

| design | foresight ρ (never-seen conduct) | failure |
|---|---|---|
| Σ-HJB field (deployed design) | **−0.20** | proxy inversion + pointwise Σ extrapolation |
| + collocation / + symmetry | −0.15 | equation enforced with garbage coefficients |
| TD-anchored field (const-weight residual) | −0.02..−0.05 | residual destroys anchor generalization |
| QRL-style quasimetric + kernel readout | ≈ 0 | distance collapse (Lagrangian) + argmax winner's curse |
| kernel, V†-style expectile training | ≈ 0 flat | 1-step edges can't calibrate a 60-step metric |
| kernel + observed k-step record | +0.19 | correct but dilutes; readout noise ≥ information |
| **ε-relaxed operator (this design)** | **+0.23, wins gap families** | — |
| V† alone (control) | +0.26 | record-bound (the thing being unified) |

The monotone lesson across five estimators: off the manifold, trust placed in
explicit structure with estimated coefficients *subtracts*; trust placed in
executed anchors plus architectural slack *adds*.

## 4. Validation (inject2d; all vs scripted-maneuver oracles)

**Probe families** (600 states each): A falling-ball conduct (dynamics never
witnessed), B airborne-car states (witnessed physics, unvisited states),
C in-distribution ground states, D pad-env low-boost (value discontinuous in
boost×position — stitching mechanically required), E pad-env mixed-boost,
E4 exposure (fine-tune on random-policy phase-2 world data; conduct never
performed). Note: families B/C in the original oracle ran with falling-ball
dynamics (instrument bug, fixed in `metric5.oracle2`); estimator-vs-estimator
comparisons unaffected (same oracle both sides).

**8 seeds** (A/B/C + exposure; paired per-seed):

| family | V† | ε=1 | note |
|---|---|---|---|
| A (unwitnessed dynamics) | +0.260 | +0.233 | small tax; bound applies to all estimators |
| **B (record gaps)** | +0.242 | **+0.272** | **ε ≥ V† on 8/8 seeds** |
| C (in-distribution) | +0.192 | +0.173 | small tax, within seed spread |
| A after exposure | +0.236 | +0.222 | parity; both hold |

**Pad env, 3 seeds**: D (stitching-required) V† +0.173 / ε **+0.197**;
E (smooth) V† +0.458 / ε +0.426. Same signature: ε reallocates estimation
power to gap/discontinuity regimes at a small smooth-regime cost.

**Training loop (the real endpoint), 8 seeds, sil scaffold, identical gates:**

| | V† ladder | ε=1 ladder |
|---|---|---|
| ignited | 8/8 | 8/8 |
| median ignition | 1.25M | **1.11M** |
| final hi / reward | 5.37 / 2.92 | **5.45 / 2.98** |

No regression anywhere it is actually consumed; ignition faster (the slack
reaches conduct value slightly ahead of the record, opening the SIL gate
earlier). One ε seed converged low (1.8) — the tail to watch at scale.

## 5. What this consolidates

The ladder becomes **V, V_exp, V‡(ε)** — one achievable-value critic, one
headroom H = relu(V‡ − V), one gate field for SIL/LP, one self-obs feature
set. Retired: the geo trio (σ-net, r̂-net, field net), the reservoir, the HJB
double-backward, the Φ-mix weight, and the geo/vdag arbitration question (ε
arbitrates inside the operator, and it is data-licensed, not scheduled).

**Production integration** (when a lineage decision is made): in the V†
target computation, add K perturbed copies of the next-state obs (σ_emp =
running std of consecutive-obs deltas over the buffer — one vector, trivially
maintained), max the min-twin bootstrap over them. ~15 lines at the existing
`InferVdagMin` site; zero new parameters; checkpoint-compatible both ways;
strictly cheaper than the geo machinery it retires. ε is the one new
hyperparameter: 0 = today's critic; 1.0 = toy-validated; anneal/schedule only
with a measured reason.

## 7. The hull revision (2026-08-07): off-manifold generality + the adversarial ledger

The Gaussian σ_emp form is bounded by witnessed AXES. The user-set follow-up
goal demanded pricing of never-before-seen BEHAVIORS whose geometry the record
implies, plus adversarial validation. Iterated result (metric6–9,
gated_check):

**The record-licensed hull operator.** Perturbations are ε-scaled *witnessed
displacement vectors*, donated between states matched in a learned dynamics
chart (L1-sparsified linear projection; inspectable — it recovers
vz/on-ground/boost/velocity and prunes position, i.e. it learns the physics'
invariances from data). Generalization is bounded by CHART support, not state
support: a chart cell measured anywhere licenses slack everywhere that cell
recurs. Because donors are real displacement vectors, asymmetries are
preserved — no unwitnessed direction (e.g. +boost in air) can ever be
hallucinated. Teleport transitions are filtered from the donor bank.

**Family F — the flick analogue** (descending low-boost touch: every leg
witnessed, the assembly occurring **0.000%** in every record): hull prices it
best of every estimator tested (0.430 vs V† 0.365, ≥ on 8/8 seeds). Family B
(unvisited states, witnessed physics): hull 0.453→0.514 gated, the largest
foresight gain of the program. Correctness parity (C ≈ 0.84).

**Adversarial ledger** (each attack pre-registered, env-as-judge where
applicable):

| attack | result |
|---|---|
| A1 hidden dynamics zone (chart confound) | degrades gracefully: absorbed as wider slack, NLL parity in/out of zone; data-corrigible |
| A2 resource laundering (fuel via compounding slack) | LANDS on Gaussian slack (halluc +0.04); hull defends (−1.66); gated hull: negative on every seed (−2.85, worst seed −0.23) |
| A3 teleport pollution | closed by donor filter (free-air ball slack 0.002 vs 0.008) |
| A4 illegal transplants | 9.6% of grafts single-action-illegal with +0.047 optimistic skew; likelihood gate ELIMINATES the skew (−0.002); residual ~8% is unbiased noise |
| thin-record hallucination (weakest seed) | radius gate closes it (+1.23 → −0.23) by auto-degrading to V† where donor density is low |

**The seat theorem (measured, not assumed).** Thin-record optimism is ONE
statistical object read from two seats: in an ESTIMATOR it is hallucination;
in an ACTUATOR it is exploration pressure. Measured both ways: the gated
(radius + likelihood) configuration is the best calibrated estimator in the
program (B 0.514, hallucination negative on all seeds) but training with it
is WORSE than baseline (ignition 2.16M, 7/8 — the gate deletes the useful
exploration pressure and the chart-refresh churn adds target noise); the open
configuration is the best trainer in the program (ignition 1.02M vs sil's
1.22M, 8/8) and its hallucination is structurally cheap in the SIL
architecture — hallucinated headroom directs attempts but can never be
consolidated (SIL clones only realized R > V_exp conversions). Deployment
therefore runs ONE open-hull-trained critic for the optimism/self-obs/LP
seat — where "I think I can improve here" on a thin region is the desired
semantics, self-correcting as attempts generate data — and the gates are the
documented configuration for any consumer that needs calibrated readouts.

**Bound that survives everything** (proven against five estimator families):
physics witnessed nowhere in the record is unrankable by any mechanism under
the no-model constraint; the chart auto-opens cells as fragments appear. The
policy must stumble on new PHYSICS once (C5's job); it never again needs to
stumble on a CONDUCT.

## 6. Honest bounds

1. **Unwitnessed dynamics are unrankable by any estimator** — measured, not
   conjectured (family A: every design ≤ V†'s inductive bias). The operator's
   contribution is that its geometry auto-extends the moment fragments are
   witnessed (σ_emp axis-gating, E4) — the policy must still stumble on new
   *physics* once (C5's job), but never on the *conduct*.
2. ε=1 pays a small, consistent rank tax in smooth regimes (−0.02..−0.03).
   The training loop absorbed it; probe-grade applications shouldn't assume it
   vanishes.
3. Toy scale: 12-dim obs, 3–8 seeds, scripted oracles (imperfect), single
   agent. The RocketSim §3.3 testbed is the next validation rung, then a
   lineage. The coherent-projection step is env-specific plumbing (in prod:
   clamp + re-derive the obs builder's dependent features).
4. Perturbations are axis-independent (diagonal σ_emp): correlated-dynamics
   slack (height bought with speed) is over-covered, same as the ellipsoid
   critique of Σ — bounded by small ε, not removed.
