"""TEAM_GATE replication round (sequential test, FINAL) — bars frozen in
TEAM_GATE.md before this ran. Two arms: R0 baseline, R2 = best-placed-
conditioned direction @ +0.5 ungated (the B2 recipe verbatim), fresh seeds."""

import json
import time
from pathlib import Path

import numpy as np
import torch

import RocketSim as rs
from load_checkpoint import copy_checkpoint, load_models
from steer_team import (NONE, WON, SteeredPolicyRho, derive_team_direction,
                        rollout_team, team_metrics)
from team_decline_probe import decline_readings
from team_gate_validate import PPT, TeamGatedPolicy, fmt

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
SEED = 20260722          # fresh derivation seed; eval seed SEED + 1 (also fresh)
ROWS_DERIVE = 700_000
ROWS_ARM = 1_200_000
N_ARENAS = 24


def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    rs.init(str(HERE.parents[1] / "build" / "collision_meshes"))
    ckpt = copy_checkpoint(HERE / "data" / "ckpt_cache")
    models = load_models(ckpt)
    print(f"pinned checkpoint {ckpt.name}", flush=True)
    rng = np.random.default_rng(SEED)

    base = rollout_team(SteeredPolicyRho(models), PPT, ROWS_DERIVE, SEED,
                        num_arenas=N_ARENAS, want_h2=True)
    rd = decline_readings(base)
    bp = rd["feas_self"] & rd["best_placed"] & np.isin(rd["outcome"], (WON, NONE))
    r_bp = {k: rd[k][bp] for k in ("row", "outcome", "d_now", "t_land")}
    v_bp, sig_bp, n_bp = derive_team_direction(base["h2"], r_bp, rng)
    print(f"bp direction: {n_bp} matched pairs, sigma {sig_bp:.2f} "
          f"({time.time()-t0:.0f}s)", flush=True)

    results = {"checkpoint": int(ckpt.name), "rows_arm": ROWS_ARM,
               "n_pairs_bp": n_bp, "sigma": float(sig_bp), "arms": {}}
    for name, v, alpha, sig in (("R0", None, 0.0, 0.0), ("R2", v_bp, 0.5, sig_bp)):
        pol = TeamGatedPolicy(models, v, alpha=alpha, scale=sig, team_gate="none")
        rec = rollout_team(pol, PPT, ROWS_ARM, SEED + 1, num_arenas=N_ARENAS)
        m = team_metrics(rec)
        results["arms"][name] = m
        print(f"{name}: {fmt(m)}", flush=True)

    a, b = results["arms"]["R2"], results["arms"]["R0"]
    d = a["team_won"] - b["team_won"]
    se = (a["team_won_se"] ** 2 + b["team_won_se"] ** 2) ** 0.5
    dn = a["none"] - b["none"]
    verdict = (d >= 0.03 and d >= 2 * se and dn <= -0.02
               and a["touch_ratio"] >= 0.9 * b["touch_ratio"]
               and a["goals_per_episode"] >= 0.8 * b["goals_per_episode"]
               and a["kickoff_first_touch_s"] < 5.0
               and a["in_air_ratio"] <= b["in_air_ratio"] + 0.15)
    results["delta_team_won"] = d
    results["se_diff"] = se
    results["delta_none"] = dn
    results["PASS"] = bool(verdict)
    print(f"delta teamWon {d:+.1%} (SE_diff {se:.1%}, {d/max(se,1e-9):.2f} sigma), "
          f"delta NONE {dn:+.1%} -> {'PASS' if verdict else 'FAIL'}", flush=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"team_gate_replicate_{ckpt.name}.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"saved {out}  ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
