# The Optimistic-Critic Ladder — Pulsar integration state (2026-07-18)

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
