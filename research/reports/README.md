# The experimental record

**Scope.** Research only: measurements, experiments, post-mortems, and the paper. Implementation
specs, deploy records, audits and shipping roadmaps live in [`docs/`](../../docs/) — they were
split out on 2026-07-25. A report here answers *what is true about the agent*; a document in
`docs/` answers *what we did to the trainer*.

One document per study. Most open with a **pre-registration block dated before
the run** — hypothesis and success criteria fixed in advance, per the repo's
measurement-before-machinery doctrine. That is the reason this corpus is worth
keeping: it records what was predicted, not just what was found.

**Read [H2_TRUNCATION.md](H2_TRUNCATION.md) first.** It is the incident that
determines which numbers below you can trust: the offline loader dropped the
trunk's trailing LeakyReLU, so every *offline behavioral* result dated before
2026-07-19 was computed against pre-activation `h2`. In-trainer telemetry is
unaffected. Reports whose numbers are implicated carry an `h2` banner.

Status vocabulary:

| Status | Meaning |
|---|---|
| **CANONICAL** | Describes a mechanism or run that is live now. Authoritative. |
| **RESULT** | A single measurement, still current. No standing authority beyond its numbers. |
| **HISTORICAL** | The mechanism is parked or superseded, but the *lesson* still governs decisions and is cited from the C++ source. Keep for rationale, not for numbers. |
| **SUPERSEDED** | Overtaken by later work; in [`archive/`](archive/). Kept for provenance. |

## Current run and live mechanisms

| Report | Date | Run | Topic | Status |
|---|---|---|---|---|
| [COMPOSITION_CRITIC.md](COMPOSITION_CRITIC.md) | 07-25 | testbed + resid | **The paper.** Composition critic: exploration by seeking realizable headroom | **CANONICAL** |
| [H2_TRUNCATION.md](H2_TRUNCATION.md) | 07-19 | cross-run | Offline loader bug, fix + oracle verification | **CANONICAL** |
| [FEAR_MINE.md](FEAR_MINE.md) | 07-15 | 4.0 | Disagreement-mined frontier drills — *graduated to live* | **CANONICAL** `h2` |
| [GOAL_CRITIC_AUDIT.md](GOAL_CRITIC_AUDIT.md) | 07-20 | 5.0v3 | Is the goal critic driving, coasting, or hindering? | **RESULT** |
| [AERIAL_GAP.md](AERIAL_GAP.md) | 07-15→21 | 5.0v3 | Where the aerial breaks down (rolling census) | **RESULT** ⚠ |

⚠ `AERIAL_GAP.md` is a rolling document: its original 2026-07-15 sections predate
the h2 fix, the 29.3B (07-19) and 45.4B (07-21) censuses postdate it.

## Historical — parked mechanisms whose lessons still bind

The activation-steering program was **superseded by the Optimistic-Critic
Ladder**. These stay out of `archive/` because the C++ source and `CLAUDE.md`
cite them as standing rationale — particularly the critic-aliasing post-mortems,
which govern *any* future change to episode boundaries.

| Report | Date | Run | Topic | Cited from |
|---|---|---|---|---|
| [STEERED_PRACTICE.md](STEERED_PRACTICE.md) | 07-12 | 3.1 | Termination/critic-aliasing post-mortems — the two Elo collapses | `ExampleMain.cpp` ×2, `CLAUDE.md` |
| [KNOWING_DOING.md](KNOWING_DOING.md) | 07-12 | 3.1 | The founding knowing-doing gap measurement | `CLAUDE.md` |
| [STEERING.md](STEERING.md) | 07-12 | 3.1 | Commitment direction; dose window and the clipping ratchet | `CLAUDE.md` |
| [REPORT.md](REPORT.md) | 07-12 | 3.1 | Phase 0 landing probes — the original detector study | `tools/README.md` |
| [RHO_CALIBRATION.md](RHO_CALIBRATION.md) | 07-12 | 3.1 | Rho is a *coarse* compass: bin it, never trust a single value | `CLAUDE.md` |
| [CREDIT_PROBE.md](CREDIT_PROBE.md) | 07-15 | 4.0 | The whiff tax, priced in critic values | `ExampleMain.cpp` |
| [CRITIC_DUEL.md](CRITIC_DUEL.md) | 07-15 | 4.0 | Critic vs goal-critic disagreement | — |
| [FEAR_DECOMP.md](FEAR_DECOMP.md) | 07-15 | 4.0 | Where fear concentrates in state space | — |
| [MECHANICS.md](MECHANICS.md) | 07-15 | 4.0 | Why the bot lacked aerials / flicks / wavedashes | — |
| [TEAM_GATE.md](TEAM_GATE.md) | 07-15 | 4.0 | Team whiff-tax targeting | — |

All ten predate the h2 fix; treat their **numbers** as invalid and their
**arguments** as live.

## Archived

[`archive/`](archive/) holds nine superseded studies — the 4.0-lineage steering
dose/validation campaign and its planning docs. Each carries a banner naming what
replaced it. They remain readable provenance for how the steering program was
tested and why it was parked; none should be used to justify a new change.

`archive/STEERING_ROADMAP.md` in particular is explicitly flagged stale in `CLAUDE.md` —
it plans a program that no longer exists.
