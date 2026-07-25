# Engineering records

Implementation specs, deploy records, audits and shipping roadmaps. Split out of
`research/reports/` on 2026-07-25 — that directory is for **research**: measurements,
experiments, post-mortems, and the paper. This one is for **the build**.

The distinction that decides where a document goes: a research report answers *what is true
about the agent*; an engineering record answers *what we did to the trainer, and what to do
next*. A document with a pre-registration block and a measured result is research. A document
with an integration map, a rollout order, or a config table is engineering.

| Document | Date | Topic |
|---|---|---|
| [ENABLED_INVENTORY.md](ENABLED_INVENTORY.md) | 07-25 | What is live in the trainer, resolved values, open decisions. **Start here.** |
| [DEAD_CODE_AUDIT.md](DEAD_CODE_AUDIT.md) | 07-25 | Live-vs-inert feature surface audit, with the strip outcome |
| [REWARD_SHAPING.md](REWARD_SHAPING.md) | 07-25 | Reward-term semantics, measured gates, staged shipping plan |
| [PULSAR5.md](PULSAR5.md) | 07-16 | 5.0 cold-start design spec |
| [LADDER.md](LADDER.md) | 07-18 | Optimistic-Critic Ladder integration/deploy record |
| [FRONTIER.md](FRONTIER.md) | 07-23 | Quasimetric-potential drill curriculum, integration map |
| [EMERGENCE.md](EMERGENCE.md) | 07-16 | RND novelty / root-cause learning-fix program |
| [LEAGUE_ANCHORS.md](LEAGUE_ANCHORS.md) | 07-19 | Spaced anchor opponents — implemented and smoke-verified |

## Superseded by the composition-critic conformance pass (2026-07-25)

`LADDER.md`, `FRONTIER.md` and `EMERGENCE.md` describe optimism machinery that has since been
**removed** from the trainer: the quasimetric map, goal/concede banks, `V_metric`, `gap_PK`,
the closure drive `Φ = −(gap_KD + gap_PK)`, the 5-column policy-head wire, and RND novelty.

The live optimism stack now matches `research/reports/COMPOSITION_CRITIC.md`:
`V_real` → `V_exp` → twin composition critics `V†`, actuated by a single seek term `Φ = +H`.
These three records are retained for provenance and for the lessons they carry, **not** as a
description of the current system. `ENABLED_INVENTORY.md` is what the trainer actually runs.
