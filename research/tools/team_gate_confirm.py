"""TEAM_GATE confirmation round — pre-registered arms + frozen bars in
TEAM_GATE.md ("Confirmation round"). Fresh derivation seed (recipe test, not
vector test), fresh eval seed, 800k rows/arm."""

import json
import time
from pathlib import Path

import numpy as np
import torch

import RocketSim as rs
from load_checkpoint import copy_checkpoint, load_models
from steer_team import (NONE, WON, SteeredPolicyRho, derive_team_direction,
                        rollout_team, team_metrics, team_possession_readings)
from team_decline_probe import decline_readings
from team_gate_validate import PPT, TeamGatedPolicy, fmt

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE.parent / "results"
SEED = 20260721          # fresh: derivation SEED, eval SEED + 1
ROWS_DERIVE = 700_000
ROWS_ARM = 800_000
N_ARENAS = 24


def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    rs.init(str(HERE.parents[1] / "build" / "collision_meshes"))
    ckpt = copy_checkpoint(HERE.parent / "data" / "ckpt_cache")
    models = load_models(ckpt)
    print(f"pinned checkpoint {ckpt.name}", flush=True)
    rng = np.random.default_rng(SEED)

    base = rollout_team(SteeredPolicyRho(models), PPT, ROWS_DERIVE, SEED,
                        num_arenas=N_ARENAS, want_h2=True)
    print(f"derivation rollout: {base['episodes']} episodes ({time.time()-t0:.0f}s)", flush=True)

    r_own = team_possession_readings(base)
    v_own, sig_own, n_own = derive_team_direction(base["h2"], r_own, rng)
    rd = decline_readings(base)
    bp = rd["feas_self"] & rd["best_placed"] & np.isin(rd["outcome"], (WON, NONE))
    r_bp = {k: rd[k][bp] for k in ("row", "outcome", "d_now", "t_land")}
    v_bp, sig_bp, n_bp = derive_team_direction(base["h2"], r_bp, rng)
    print(f"directions: own {n_own} pairs sigma {sig_own:.2f} | "
          f"bp {n_bp} pairs sigma {sig_bp:.2f} | cos {float(np.dot(v_bp, v_own)):+.2f}", flush=True)

    arms = [
        ("B0", None, 0.0, 0.0, "none"),
        ("B1", v_own, 0.5, sig_own, "prox"),
        ("B2", v_bp, 0.5, sig_bp, "none"),
        ("B3", v_bp, 0.5, sig_bp, "prox"),
    ]
    results = {"checkpoint": int(ckpt.name), "rows_arm": ROWS_ARM,
               "n_pairs_own": n_own, "n_pairs_bp": n_bp,
               "cos_bp_vs_own": float(np.dot(v_bp, v_own)), "arms": {}}
    for name, v, alpha, sig, tg in arms:
        pol = TeamGatedPolicy(models, v, alpha=alpha, scale=sig, team_gate=tg)
        rec = rollout_team(pol, PPT, ROWS_ARM, SEED + 1, num_arenas=N_ARENAS)
        m = team_metrics(rec)
        m["inband_frac"] = pol.last_inband_frac
        m["teamgate_frac"] = pol.last_teamgate_frac
        results["arms"][name] = m
        print(f"{name} gate={tg:5s}: {fmt(m)}", flush=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"team_gate_confirm_{ckpt.name}.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"saved {out}  ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
