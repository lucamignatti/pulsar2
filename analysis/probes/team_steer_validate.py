"""Team-mode steering validation (the python-first gate for taking steering to
2v2/3v3 in the trainer). Three experiments:

  census   (E1) per-mode possession-outcome census on feasible airborne readings.
           The premise of team-mode steering is a COLLECTIVE-DECLINE frontier:
           feasible balls that nobody on either team goes for. If "none" is rare
           in team modes, there is nothing for commitment steering to buy.
  sweep    (E2) per-mode dose-response: derive the mode's OWN commitment direction
           (WON vs collective-NONE, teammate-resolved races excluded) and sweep
           alpha with the live rho-band gate port. Success bar: NONE falls /
           team-won rises sign-correctly without competence collapse.
  transfer (E3) apply the 1v1-derived direction in 2v2: if it transfers, the
           trainer can share one direction across modes; if not, per-mode
           directions (and derivation pools) are justified.

Usage:
  python team_steer_validate.py census   [--rows N]
  python team_steer_validate.py sweep    --ppt 2 [--rows N]
  python team_steer_validate.py transfer --ppt 2 [--rows N]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import RocketSim as rs
from load_checkpoint import copy_checkpoint, load_models
from steer_team import (SteeredPolicyRho, derive_team_direction, rollout_team,
                        team_metrics, team_possession_readings)

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
SEED = 20260717
ALPHAS = [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0]


def fmt(m):
    return (f"selfWon {m['self_won']:.1%} mateWon {m['teammate_won']:.1%} "
            f"lost {m['lost']:.1%} NONE {m['none']:.1%} "
            f"(teamWon {m['team_won']:.1%} +-{m['team_won_se']:.1%}, n={m['n_feasible_readings']} "
            f"in {m['n_reading_episodes']} eps) | touch {m['touch_ratio']*100:.2f}% "
            f"air {m['in_air_ratio']:.0%} goals/ep {m['goals_per_episode']:.2f} "
            f"kick {m['kickoff_first_touch_s']:.2f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["census", "sweep", "transfer"])
    ap.add_argument("--ppt", type=int, default=2)
    ap.add_argument("--rows", type=int, default=120_000)
    ap.add_argument("--ckpt", type=Path, default=None)
    args = ap.parse_args()

    t0 = time.time()
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    rs.init(str(HERE.parents[1] / "build" / "collision_meshes"))
    ckpt = args.ckpt or copy_checkpoint(HERE / "data" / "ckpt_cache")
    models = load_models(ckpt)
    print(f"pinned checkpoint {ckpt.name}")
    RESULTS_DIR.mkdir(exist_ok=True)

    if args.cmd == "census":
        results = {"checkpoint": int(ckpt.name), "rows": args.rows, "modes": {}}
        for ppt in (1, 2, 3):
            rec = rollout_team(SteeredPolicyRho(models), ppt, args.rows, SEED)
            m = team_metrics(rec)
            results["modes"][f"{ppt}v{ppt}"] = m
            print(f"{ppt}v{ppt}: {fmt(m)}", flush=True)
        out = RESULTS_DIR / f"team_census_{ckpt.name}.json"

    elif args.cmd == "sweep":
        ppt = args.ppt
        base = rollout_team(SteeredPolicyRho(models), ppt, args.rows, SEED, want_h2=True)
        readings = team_possession_readings(base)
        rng = np.random.default_rng(SEED)
        v, sigma, n_pairs = derive_team_direction(base["h2"], readings, rng)
        print(f"{ppt}v{ppt} direction: {n_pairs} matched WON/NONE pairs, sigma {sigma:.2f}")
        results = {"checkpoint": int(ckpt.name), "mode": f"{ppt}v{ppt}", "rows": args.rows,
                   "n_pairs": n_pairs, "sigma": sigma, "alphas": {}}
        for a in ALPHAS:
            pol = SteeredPolicyRho(models, v, alpha=a, scale=sigma)
            rec = rollout_team(pol, ppt, args.rows, SEED + 1)
            m = team_metrics(rec)
            m["inband_frac"] = pol.last_inband_frac
            results["alphas"][str(a)] = m
            print(f"a={a:+.1f}: {fmt(m)}", flush=True)
        out = RESULTS_DIR / f"team_sweep_{ppt}v{ppt}_{ckpt.name}.json"

    else:  # transfer: 1v1-derived direction applied in NvN
        ppt = args.ppt
        base1 = rollout_team(SteeredPolicyRho(models), 1, args.rows, SEED, want_h2=True)
        r1 = team_possession_readings(base1)
        rng = np.random.default_rng(SEED)
        v1, sig1, n1 = derive_team_direction(base1["h2"], r1, rng)
        print(f"1v1 direction: {n1} pairs, sigma {sig1:.2f} -> applied in {ppt}v{ppt}")
        results = {"checkpoint": int(ckpt.name), "mode": f"{ppt}v{ppt}",
                   "direction": "1v1-derived", "rows": args.rows, "alphas": {}}
        for a in [0.0, 0.5, 1.0, 2.0]:
            pol = SteeredPolicyRho(models, v1, alpha=a, scale=sig1)
            rec = rollout_team(pol, ppt, args.rows, SEED + 1)
            m = team_metrics(rec)
            results["alphas"][str(a)] = m
            print(f"a={a:+.1f}: {fmt(m)}", flush=True)
        out = RESULTS_DIR / f"team_transfer_{ppt}v{ppt}_{ckpt.name}.json"

    results["runtime_s"] = round(time.time() - t0, 1)
    out.write_text(json.dumps(results, indent=2))
    print(f"runtime {results['runtime_s']}s; wrote {out.name}")


if __name__ == "__main__":
    main()
