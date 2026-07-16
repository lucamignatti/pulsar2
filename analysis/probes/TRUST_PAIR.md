# TRUST_PAIR — paired commit+trust role steering (offline validation)

**Pre-registered 2026-07-16 BEFORE running** (user directive: "steer complete
teammate trust as well as commits"). Script `trust_pair.py`, newest checkpoint.

## Design

Directions (both live-derivable; offline derivation from one 2v2 rollout):
- **commit**: best-placed-conditioned WON-vs-NONE (the TEAM_GATE recipe — the
  strongest known, ~+2-4pp alone, below deploy bars alone).
- **trust**: on NON-best-placed rows of the same readings, trunk-mean contrast
  of "teammate took it" (outcome TEAMMATE) vs "nobody did" (NONE) — the
  belief axis B3 showed is linearly present (AUC 0.79).

Arms (800k rows each, same eval seed, prox-gate machinery does the role split):

| arm | best-placed player | other teammate |
|---|---|---|
| B0 | — | — |
| C | commit +0.5σ | — |
| T | — | trust +0.5σ |
| P (paired) | commit +0.5σ | trust +0.5σ |

## Metrics

teamWon and NONE on feasible readings (steer_v2 parity, cluster SEs), plus
**back-fill**: among readings where the best-placed teammate PURSUES, the other
teammate's mean canonical-y retreat (goal-side displacement over the next 1.5s)
— the rotation primitive the user is asking for.

## Frozen bars

1. **Pairing bar**: teamWon(P) − teamWon(B0) ≥ +4pp at ≥2× SE_diff AND
   P > max(C, T) point-wise (the convention needs both halves — if either
   solo arm matches P, pairing adds nothing and the simpler lever wins).
2. **Rotation bar**: back-fill(P) > back-fill(B0) by ≥2× cluster SE (the trust
   half must actually produce covering movement, not just passivity).
3. Canaries: touch/goals within −10%/−20% of B0, kickoff < 5s, NONE(P) ≤ NONE(B0).

PASS both → trainer implementation (live-derived trust vector next to the
commitment EMA, paired application via the existing per-mode steered slices +
role gate; own deploy window AFTER RC1's). FAIL → recorded; trust axis joins
the dead-candidate list and rotation waits on commitment reliability
(FEAR_MINE) + RC1.

## Results

*(to be filled; bars frozen)*
