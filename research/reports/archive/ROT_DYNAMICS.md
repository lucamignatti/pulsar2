> **Status: SUPERSEDED — archived.** Part of the 4.0-lineage activation-steering
> campaign. Steering is numerically inert on HEAD (`steering.alpha = 0`); the
> Optimistic-Critic Ladder ([LADDER.md](../LADDER.md)) replaced it as the optimism
> mechanism. Kept as provenance for how the program was tested and why it was
> parked. **Do not cite this to justify a new change.**
>
> Offline numbers here predate the 2026-07-19 h2-truncation fix and were computed
> against **pre-activation `h2`** — see [H2_TRUNCATION.md](../H2_TRUNCATION.md).

---

# ROT_DYNAMICS — is direction rot a rotation in a stable subspace? (anti-drift study)

**Pre-registered 2026-07-15 BEFORE running.** Script `rot_dynamics.py`.

## Question (user, 2026-07-15: "how do we make this automatic so as it trains it never drifts")

Directions rot behaviorally within ~75M steps (measured +7pp → −11pp in the v1
era) — the reason everything is live-derived. This study asks about the
GEOMETRY of that rot: derive the commitment direction independently at ~8
policy versions spanning the 800M-step version window (each from its own
self-play rollout, same v2 recipe), then measure:

1. **Pairwise cos decay** vs step distance. Note the caveat up front: each
   direction lives in its own trunk's h2 basis, so cross-version cos measures
   representation drift + concept drift COMBINED — which is exactly the
   operative quantity for "could any frozen vector steer a later checkpoint".
2. **Leave-one-out subspace projection**: for each version's direction, the
   norm of its projection onto the top-k PCA subspace of the OTHER versions'
   directions.

## Registered interpretations

- Pairwise cos decays BUT LOO projection stays ≥ 0.8 at k ≤ 3 → rot is mostly
  ROTATION INSIDE A STABLE LOW-DIM SUBSPACE → a checkpointed subspace + live
  per-iteration coefficient fit becomes a legitimate lighter-weight anti-drift
  mechanism (own pre-registration before any use; the current live-derivation
  stays until then).
- LOO projection decays like the pairwise cos → GENUINE drift → live
  per-iteration derivation is the only sound mechanism; the automation question
  is CLOSED (current design confirmed as the answer), and the remaining
  automation gap is only recipe-level (handled by measurement cadence, not
  machinery, per the meta-system post-mortem).

## Results (2026-07-15; raw: `results/rot_dynamics.json`)

8 versions spanning 28.30B → 29.08B (the full ~800M version window), 152-224
matched pairs each, sigma stable 5.2-6.2.

- **Pairwise cos is HIGH and nearly distance-flat**: 0.82 within 400M steps,
  0.77 beyond — the commitment direction is geometrically near-constant across
  800M steps of training.
- **LOO subspace projection: 0.88 mean at k=1** (0.91 at k=3, min 0.72) — the
  8 independently-derived directions are essentially one common axis plus small
  wobble.

**Interpretation (sharper than either registered branch):** the geometry
barely rots. Combined with the historically measured behavioral flip
(+7pp → −11pp within ~75M, v1 era) and this week's saturation results, the
conclusion is that what drifts is not the VECTOR but its PAYOFF — the
dose-response of a nearly-fixed axis changes as the policy internalizes the
behavior and the data distribution moves. Consequences:

1. The anti-drift machinery is ALREADY correctly shaped and complete: live
   per-iteration derivation covers the (small) geometric wobble at negligible
   cost, and the causal possession gates + rating latch measure the only thing
   that actually rots (payoff), continuously, with automatic benching. Nothing
   further to automate at the vector level.
2. Recipe-level drift (which contrast/conditioning pays — e.g. this week's
   best-placed pool fix) cannot be automated without rebuilding the meta system
   (convicted 2026-07-14); it is handled by measurement cadence: rerun the
   offline census/probe battery when the live gates flatline.
3. Caveats: mature-policy window only (28.3-29.1B); cross-version cos compounds
   representation drift with concept drift, so the true concept stability is at
   least as high as measured.
