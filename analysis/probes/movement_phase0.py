"""Car-movement (execution) frontier Phase 0: derive a COMPLETED-vs-WHIFFED trunk
direction and run the pre-registered dose-response sweep (MOVEMENT_PHASE0.md — read
it first; the bars there were registered before this ran).

Reuses the validated harness: steer_test.rollout / SteeredPolicy (dosed self-play),
steer_v2 outcome labeling (WON/LOST/NONE, feasibility, matching recipe),
knowing_doing's approach-speed lookahead for the new PURSUED label.

Usage:
  python movement_phase0.py [--ckpt DIR] [--rows N] [--sweep-rows N] [--quick]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import RocketSim as rs
import collect_dataset as cd
from label_landing import simulate_landing
from load_checkpoint import copy_checkpoint, load_models
from steer_test import SteeredPolicy, rollout
from steer_v2 import derive_direction, possession_readings, WON, LOST, NONE

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"

SEED = 20260715
DT = 1 / 30.0
ARM_Z = 300.0
FEASIBLE_SPEED = 1300.0
ATTEND_RADIUS = 500.0
PURSUE_APPROACH_FRAC = 0.6   # net approach speed >= this * required speed = pursued
RACE_MARGIN_S = 0.5
MATCH_BINS_D = 5
MATCH_BINS_T = 3
MAX_READINGS = 12000
ALPHAS = [-1.0, -0.5, 0.0, 0.25, 0.5, 1.0]


def movement_readings(rec, max_readings: int = MAX_READINGS):
    """steer_v2.possession_readings + the PURSUED label (attend / approach lookahead).
    Feasible, uncensored airborne readings only."""
    phys, episode, team, touched = rec["phys"], rec["episode"], rec["team"], rec["touched"]

    ep_rows = {int(e): np.flatnonzero(episode == e) for e in np.unique(episode)}
    row_pos = np.empty(len(team), np.int64)
    for rows in ep_rows.values():
        row_pos[rows] = np.arange(len(rows))

    cand = np.flatnonzero(phys[:, 2] > ARM_Z)
    stride = max(1, len(cand) // max_readings)
    cand = cand[::stride]

    arena = rs.Arena(rs.GameMode.SOCCAR)
    margin_rows = int(round(RACE_MARGIN_S / DT)) * 2

    out = {k: [] for k in ("row", "outcome", "d_now", "t_land", "pursued", "approach_frac")}
    for r in cand:
        res = simulate_landing(arena, phys[r, 0:3], phys[r, 3:6], phys[r, 6:9])
        if res is None:
            continue
        lx, ly, t_land = res
        rows = ep_rows[int(episode[r])]
        q = row_pos[r]
        q_land = q + 2 * int(round(t_land / DT))
        if q_land >= len(rows):
            continue
        sl = slice(20, 23) if team[r] == 1 else slice(9, 12)
        land = np.array([lx, ly])
        d_now = float(np.linalg.norm(phys[r, sl][:2] - land))
        required = d_now / max(t_land, 1e-6)
        if required >= FEASIBLE_SPEED:
            continue

        outcome = NONE
        parity = q % 2
        for qq in range(q + 1, min(len(rows), q_land + margin_rows + 1)):
            if touched[rows[qq]]:
                outcome = WON if (qq % 2) == parity else LOST
                break

        # PURSUED: won outright, or attended the landing, or closed >= 60% of the
        # required approach speed (knowing_doing's net-displacement lookahead)
        d_land = float(np.linalg.norm(phys[rows[q_land], sl][:2] - land))
        approach = (d_now - d_land) / max(t_land, 1e-6)
        pursued = (outcome == WON) or (d_land < ATTEND_RADIUS) \
            or (approach >= PURSUE_APPROACH_FRAC * required)

        out["row"].append(int(r))
        out["outcome"].append(outcome)
        out["d_now"].append(d_now)
        out["t_land"].append(float(t_land))
        out["pursued"].append(bool(pursued))
        out["approach_frac"].append(float(approach / max(required, 1e-6)))

    return {k: np.array(v) for k, v in out.items()}


def derive_movement_direction(h2, rd, rng, v_commit=None):
    """Matched COMPLETED-vs-WHIFFED difference of trunk means among PURSUED readings.

    Quarter-scale finding (recorded in MOVEMENT_PHASE0.md results): with distance x time
    matching only, this contrast is commitment in disguise (cos vs v_commit = 0.92) -
    completion differences are mostly pursuit-intensity differences. Two refinements
    isolate execution: (a) approach-intensity joins the matching bins, so both classes
    chased equally hard; (b) the result is orthogonalized against v_commit before
    dosing, so the sweep can only move through a commitment-independent channel."""
    pursued = np.flatnonzero(rd["pursued"])
    comp = rd["outcome"][pursued] == WON
    d, t = rd["d_now"][pursued], rd["t_land"][pursued]
    a = rd["approach_frac"][pursued]
    d_edges = np.quantile(d, np.linspace(0, 1, MATCH_BINS_D + 1))[1:-1]
    t_edges = np.quantile(t, np.linspace(0, 1, MATCH_BINS_T + 1))[1:-1]
    a_edges = np.quantile(a, np.linspace(0, 1, 3 + 1))[1:-1]
    bins = (np.digitize(a, a_edges) * 100
            + np.digitize(d, d_edges) * 10 + np.digitize(t, t_edges))
    sel_c, sel_w = [], []
    for b in np.unique(bins):
        c = pursued[np.flatnonzero((bins == b) & comp)]
        w = pursued[np.flatnonzero((bins == b) & ~comp)]
        m = min(len(c), len(w))
        if m == 0:
            continue
        sel_c += list(rng.choice(c, m, replace=False))
        sel_w += list(rng.choice(w, m, replace=False))
    if not sel_c:
        raise RuntimeError("no matched COMPLETED/WHIFFED pairs")
    sel_c, sel_w = np.array(sel_c), np.array(sel_w)
    Hc = h2[rd["row"][sel_c]].astype(np.float32)
    Hw = h2[rd["row"][sel_w]].astype(np.float32)
    v = Hc.mean(0) - Hw.mean(0)
    v /= max(np.linalg.norm(v), 1e-8)
    raw_cos = float("nan")
    if v_commit is not None:
        raw_cos = float(np.dot(v, v_commit))
        v = v - np.dot(v, v_commit) * v_commit  # execution channel only
        v /= max(np.linalg.norm(v), 1e-8)
    with np.errstate(all="ignore"):
        proj = h2.astype(np.float32) @ v
    assert np.isfinite(proj).all()
    return v, float(np.std(proj)), len(sel_c), raw_cos


def movement_metrics(rec, boot: int = 200, rng=None):
    """The sweep panel: completion = P(WON | pursued) with episode-cluster SE,
    pursue-rate canary, plus the standard behavior/kickoff canaries."""
    rd = movement_readings(rec)
    n = len(rd["row"])
    pursued = rd["pursued"]
    n_pursued = int(pursued.sum()) if n else 0
    completion = float((rd["outcome"][pursued] == WON).mean()) if n_pursued else float("nan")
    pursue_rate = float(pursued.mean()) if n else float("nan")
    poss_win = float((rd["outcome"] == WON).mean()) if n else float("nan")

    se = float("nan")
    if n_pursued:
        rng = rng or np.random.default_rng(0)
        eps = rec["episode"][rd["row"][pursued]]
        outs_all = rd["outcome"][pursued]
        uniq = np.unique(eps)
        by_ep = {e: outs_all[eps == e] for e in uniq}
        stats = []
        for _ in range(boot):
            sample = rng.choice(uniq, len(uniq), replace=True)
            outs = np.concatenate([by_ep[e] for e in sample])
            stats.append((outs == WON).mean())
        se = float(np.std(stats))

    aerial = rec["touched"] & ~rec["on_ground"] & (rec["phys"][:, 2] > 400)
    return {
        "completion": completion,
        "completion_se": se,
        "pursue_rate": pursue_rate,
        "poss_win": poss_win,
        "n_feasible": int(n),
        "n_pursued": n_pursued,
        "n_episodes": int(len(np.unique(rec["episode"][rd["row"]]))) if n else 0,
        "aerial_touch_ratio": float(aerial.mean()),
        "touch_ratio": float(rec["touched"].mean()),
        "goals_per_episode": rec["goals"] / rec["episodes"],
        "kickoff_first_touch_s": float(np.median(rec["kick_first_touch"]))
        if rec["kick_first_touch"] else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--rows", type=int, default=300_000)
    ap.add_argument("--sweep-rows", type=int, default=200_000)
    ap.add_argument("--quick", action="store_true", help="1/4 rows everywhere (smoke)")
    args = ap.parse_args()
    if args.quick:
        args.rows //= 4
        args.sweep_rows //= 4

    t0 = time.time()
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    rs.init(str(HERE.parents[1] / "build" / "collision_meshes"))
    rng = np.random.default_rng(SEED)

    ckpt = args.ckpt or copy_checkpoint(HERE / "data" / "ckpt_cache_movement")
    models = load_models(ckpt)
    cd.set_obs_size(SteeredPolicy(models).obs_size)
    print(f"[{time.time()-t0:6.0f}s] checkpoint {ckpt.name} loaded")

    base_pol = SteeredPolicy(models)  # alpha 0 == plain policy, same code path
    rec = rollout(base_pol, args.rows, SEED, want_h2=True)
    rd = movement_readings(rec)
    n_pur = int(rd["pursued"].sum())
    n_comp = int((rd["outcome"][rd["pursued"]] == WON).sum())
    print(f"[{time.time()-t0:6.0f}s] base: {len(rd['row'])} feasible readings, "
          f"{n_pur} pursued ({n_comp} completed / {n_pur-n_comp} whiffed)")

    v_commit, sig_c, n_pairs_c = derive_direction(rec["h2"], possession_readings(rec), rng)
    v_move, sigma, n_pairs, raw_cos = derive_movement_direction(rec["h2"], rd, rng, v_commit)
    cos = float(np.dot(v_move, v_commit))  # ~0 by construction; raw_cos is the finding
    print(f"[{time.time()-t0:6.0f}s] v_move: {n_pairs} matched pairs, sigma {sigma:.3f}; "
          f"raw cos vs commit {raw_cos:+.3f} -> orthogonalized (applied cos {cos:+.3f}, "
          f"commit pairs {n_pairs_c})")

    table = {}
    for alpha in ALPHAS:
        pol = SteeredPolicy(models, v=v_move, alpha=alpha, scale=sigma)
        r = rollout(pol, args.sweep_rows, SEED + 1)  # paired seeds across alphas
        table[alpha] = movement_metrics(r, rng=np.random.default_rng(SEED + 2))
        m = table[alpha]
        print(f"[{time.time()-t0:6.0f}s] alpha {alpha:+.2f}: "
              f"completion {m['completion']:.3f}±{m['completion_se']:.3f} "
              f"(n_pursued {m['n_pursued']}), pursue {m['pursue_rate']:.3f}, "
              f"poss_win {m['poss_win']:.3f}, touch {m['touch_ratio']:.4f}, "
              f"kick {m['kickoff_first_touch_s']:.2f}s")

    # ---- pre-registered verdict (MOVEMENT_PHASE0.md) ----
    b, z, p5 = table[0.0], table[0.5], table[0.25]
    d_comp = z["completion"] - b["completion"]
    se_d = float(np.hypot(z["completion_se"], b["completion_se"]))
    bars = {
        "1_causal": bool(d_comp >= 0.03 and d_comp >= 2 * se_d),
        "2_dose_shape": bool(table[-0.5]["completion"] < b["completion"] < z["completion"]),
        "3_not_commitment": bool(abs(z["pursue_rate"] - b["pursue_rate"]) <= 0.03
                                 and abs(cos) < 0.6),
        "4_canaries": bool(z["kickoff_first_touch_s"] < 5.0
                           and z["touch_ratio"] >= 0.9 * b["touch_ratio"]
                           and z["goals_per_episode"] >= 0.8 * b["goals_per_episode"]
                           and z["poss_win"] >= b["poss_win"] - 2 * se_d),
    }
    verdict = all(bars.values())
    print("\n==== PRE-REGISTERED VERDICT ====")
    for k, ok in bars.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {k}")
    print(f"  => {'ENABLE in trainer' if verdict else 'DO NOT ENABLE'} "
          f"(dCompletion@+0.5 = {d_comp:+.3f} ± {se_d:.3f}, cos {cos:+.3f})")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = {
        "checkpoint": ckpt.name, "rows": args.rows, "sweep_rows": args.sweep_rows,
        "n_matched_pairs": n_pairs, "sigma": sigma, "cos_vs_commit": cos,
        "raw_cos_vs_commit": raw_cos,
        "sweep": {str(a): table[a] for a in ALPHAS},
        "bars": bars, "verdict": "enable" if verdict else "reject",
        "v_move": v_move.tolist(),
    }
    path = RESULTS_DIR / f"movement_phase0_{ckpt.name}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"[{time.time()-t0:6.0f}s] wrote {path}")


if __name__ == "__main__":
    main()
