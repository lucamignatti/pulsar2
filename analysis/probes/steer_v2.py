"""Shared v2-possession steering machinery for the Phase-0 offline programs
(STEERING_ROADMAP.md: 0a alpha-sweep, 0b random-direction baseline, 0c style contrasts).

Mirrors the AUTHORITATIVE in-trainer v2 derivation (Learner.cpp fnSteerUpdate), not the
v1 landing-attendance metric that derive_steering.py/steer_test.py still use:
  - readings = airborne-ball rows; ball-only landing sim; feasible if the reading's
    player can beat FEASIBLE_SPEED to the landing spot
  - outcome = FIRST TOUCH between the reading and shortly past touchdown:
    self = WON the race, opponent = LOST, neither = NONE (unclaimed)
  - direction = matched (distance x flight-time quantile bins) difference of trunk
    means, WON vs NONE (LOST excluded - punishing lost races trains hesitation)

Rollout data comes from steer_test.rollout (phys/episode/team/touched layout: rows
interleave the two players per arena step, so within an episode's row list, rows of
the reading's player are those with matching parity and +1 step = +2 rows).
"""

from pathlib import Path

import numpy as np

import RocketSim as rs
from label_landing import simulate_landing

DT = 1 / 30.0
ARM_Z = 300.0
FEASIBLE_SPEED = 1300.0
RACE_MARGIN_S = 0.5          # grace past touchdown for the race (trainer parity)
MATCH_BINS_D = 5
MATCH_BINS_T = 3
MAX_READINGS = 4000

WON, LOST, NONE = 1, 2, 0


def possession_readings(rec, max_readings: int = MAX_READINGS):
    """Label airborne readings by v2 possession outcome. Returns a dict of arrays:
    row, outcome (WON/LOST/NONE), d_now, t_land - feasible readings only."""
    phys, episode, team, touched = rec["phys"], rec["episode"], rec["team"], rec["touched"]

    ep_rows = {int(e): np.flatnonzero(episode == e) for e in np.unique(episode)}
    row_pos = np.empty(len(team), np.int64)
    for rows in ep_rows.values():
        row_pos[rows] = np.arange(len(rows))

    cand = np.flatnonzero(phys[:, 2] > ARM_Z)
    stride = max(1, len(cand) // max_readings)
    cand = cand[::stride]

    arena = rs.Arena(rs.GameMode.SOCCAR)
    margin_rows = int(round(RACE_MARGIN_S / DT)) * 2  # interleaved rows

    out = {"row": [], "outcome": [], "d_now": [], "t_land": []}
    for r in cand:
        res = simulate_landing(arena, phys[r, 0:3], phys[r, 3:6], phys[r, 6:9])
        if res is None:
            continue
        lx, ly, t_land = res
        rows = ep_rows[int(episode[r])]
        q = row_pos[r]
        q_land = q + 2 * int(round(t_land / DT))
        if q_land >= len(rows):
            continue  # censored: episode ended before touchdown
        sl = slice(20, 23) if team[r] == 1 else slice(9, 12)
        d_now = float(np.linalg.norm(phys[r, sl][:2] - np.array([lx, ly])))
        if d_now / max(t_land, 1e-6) >= FEASIBLE_SPEED:
            continue

        outcome = NONE
        parity = q % 2
        for qq in range(q + 1, min(len(rows), q_land + margin_rows + 1)):
            if touched[rows[qq]]:
                outcome = WON if (qq % 2) == parity else LOST
                break
        out["row"].append(int(r))
        out["outcome"].append(outcome)
        out["d_now"].append(d_now)
        out["t_land"].append(float(t_land))

    return {k: np.array(v) for k, v in out.items()}


def matched_pairs(readings, rng: np.random.Generator):
    """Quantile-matched WON-vs-NONE selection (trainer parity: 5 distance x 3 time bins).
    Returns (sel_won, sel_none) as indices into readings arrays."""
    keep = readings["outcome"] != LOST
    idx = np.flatnonzero(keep)
    won = readings["outcome"][idx] == WON
    d, t = readings["d_now"][idx], readings["t_land"][idx]
    d_edges = np.quantile(d, np.linspace(0, 1, MATCH_BINS_D + 1))[1:-1]
    t_edges = np.quantile(t, np.linspace(0, 1, MATCH_BINS_T + 1))[1:-1]
    bins = np.digitize(d, d_edges) * 10 + np.digitize(t, t_edges)
    sel_w, sel_n = [], []
    for b in np.unique(bins):
        w = idx[np.flatnonzero((bins == b) & won)]
        n = idx[np.flatnonzero((bins == b) & ~won)]
        m = min(len(w), len(n))
        if m == 0:
            continue
        sel_w += list(rng.choice(w, m, replace=False))
        sel_n += list(rng.choice(n, m, replace=False))
    return np.array(sel_w, dtype=int), np.array(sel_n, dtype=int)


def derive_direction(h2: np.ndarray, readings, rng: np.random.Generator):
    """v2 commitment direction: matched difference of trunk means (WON - NONE), plus
    sigma of all-row projections. Returns (v, sigma, n_per_class)."""
    sel_w, sel_n = matched_pairs(readings, rng)
    if len(sel_w) == 0:
        raise RuntimeError("no matched WON/NONE pairs")
    Hw = h2[readings["row"][sel_w]].astype(np.float32)
    Hn = h2[readings["row"][sel_n]].astype(np.float32)
    v = Hw.mean(0) - Hn.mean(0)
    v /= max(np.linalg.norm(v), 1e-8)
    # np.errstate: Apple Accelerate's sgemm raises spurious FP-exception flags on this
    # platform (numpy 2.x); the finite assert below is the real guard
    with np.errstate(all="ignore"):
        proj = h2.astype(np.float32) @ v
    assert np.isfinite(proj).all(), "non-finite trunk projections - inspect h2"
    sigma = float(np.std(proj))
    return v, sigma, len(sel_w)


def possession_metrics(rec, max_readings: int = MAX_READINGS, boot: int = 200,
                       rng: np.random.Generator | None = None):
    """The v2 sweep panel: possession outcomes on feasible readings + trainer-parity
    behavior metrics + the kickoff competence canary. poss_win_se is an EPISODE-cluster
    bootstrap (readings within an episode are near-duplicates; naive binomial SE is
    ~5x optimistic - README.md)."""
    rd = possession_readings(rec, max_readings)
    n = len(rd["row"])
    aerial = rec["touched"] & ~rec["on_ground"] & (rec["phys"][:, 2] > 400)
    won = float((rd["outcome"] == WON).mean()) if n else float("nan")
    lost = float((rd["outcome"] == LOST).mean()) if n else float("nan")

    se = float("nan")
    if n:
        rng = rng or np.random.default_rng(0)
        eps = rec["episode"][rd["row"]]
        uniq = np.unique(eps)
        by_ep = {e: rd["outcome"][eps == e] for e in uniq}
        stats = []
        for _ in range(boot):
            sample = rng.choice(uniq, len(uniq), replace=True)
            outs = np.concatenate([by_ep[e] for e in sample])
            stats.append((outs == WON).mean())
        se = float(np.std(stats))

    return {
        "poss_win": won,
        "poss_win_se": se,
        "poss_lost": lost,
        "poss_none": 1 - won - lost if n else float("nan"),
        "n_feasible_readings": int(n),
        "n_reading_episodes": int(len(np.unique(rec["episode"][rd["row"]]))) if n else 0,
        "aerial_touch_ratio": float(aerial.mean()),
        "touch_ratio": float(rec["touched"].mean()),
        "in_air_ratio": float((~rec["on_ground"]).mean()),
        "goals_per_episode": rec["goals"] / rec["episodes"],
        "kickoff_first_touch_s": float(np.median(rec["kick_first_touch"]))
        if rec["kick_first_touch"] else float("nan"),
        "n_kickoffs_touched": len(rec["kick_first_touch"]),
    }
