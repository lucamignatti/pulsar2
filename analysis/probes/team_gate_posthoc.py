"""TEAM_GATE post-hoc supplement (EXPLORATORY, licenses nothing — see TEAM_GATE.md).

The pre-registered sweep convicted the 1v1-transfer direction as acutely dead in
2v2 at 27550023244. Two follow-up questions, answered with the same eval seed so
the arms are comparable to A0/A1:

  S1  the LIVE mechanism: 2v2-OWN derived direction (the trainer's per-mode
      WON-vs-NONE derivation, steer_vec_2v2 in RUNNING_STATS) @ +0.5, ungated.
      If this also reads ~0, live team steering is currently cosmetic.
  S2  the finding-shaped candidate: direction derived from BEST-PLACED rows only
      (WON vs NONE conditioned on 'this player was the best-placed teammate',
      labels from team_decline_probe.decline_readings) @ +0.5, prox-gated —
      matched derivation and application: the axis of 'the right player goes'.
"""

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
RESULTS_DIR = HERE / "results"
SEED = 20260720          # same base seed as the main sweep
ROWS_DERIVE = 700_000
ROWS_ARM = 400_000
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

    # one 2v2 base rollout feeds both derivations (SEED+2: never the eval seed)
    base = rollout_team(SteeredPolicyRho(models), PPT, ROWS_DERIVE, SEED + 2,
                        num_arenas=N_ARENAS, want_h2=True)
    print(f"derivation rollout: {base['episodes']} episodes ({time.time()-t0:.0f}s)", flush=True)

    # S1: the trainer's own per-mode recipe (WON vs NONE, teammate/lost excluded)
    r_own = team_possession_readings(base)
    v_own, sig_own, n_own = derive_team_direction(base["h2"], r_own, rng)
    print(f"S1 2v2-own direction: {n_own} matched pairs, sigma {sig_own:.2f}", flush=True)

    # S2: best-placed-conditioned WON vs NONE
    rd = decline_readings(base)
    bp = rd["feas_self"] & rd["best_placed"] & np.isin(rd["outcome"], (WON, NONE))
    r_bp = {k: rd[k][bp] for k in ("row", "outcome", "d_now", "t_land")}
    v_bp, sig_bp, n_bp = derive_team_direction(base["h2"], r_bp, rng)
    cos = float(np.dot(v_bp, v_own))
    print(f"S2 best-placed direction: {n_bp} matched pairs, sigma {sig_bp:.2f}, "
          f"cos(vs 2v2-own) {cos:+.2f}", flush=True)

    results = {"checkpoint": int(ckpt.name), "rows_arm": ROWS_ARM, "posthoc": True,
               "s1_pairs": n_own, "s2_pairs": n_bp, "cos_bp_vs_own": cos, "arms": {}}
    for name, v, sig, tg in (("S1", v_own, sig_own, "none"), ("S2", v_bp, sig_bp, "prox")):
        pol = TeamGatedPolicy(models, v, alpha=0.5, scale=sig, team_gate=tg)
        rec = rollout_team(pol, PPT, ROWS_ARM, SEED + 1, num_arenas=N_ARENAS)
        m = team_metrics(rec)
        m["inband_frac"] = pol.last_inband_frac
        m["teamgate_frac"] = pol.last_teamgate_frac
        results["arms"][name] = m
        print(f"{name} a=+0.5 gate={tg:5s}: {fmt(m)}", flush=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"team_gate_posthoc_{ckpt.name}.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"saved {out}  ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
