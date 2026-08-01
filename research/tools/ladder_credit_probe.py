"""LADDER CREDIT PROBE (5.3) — the whiff-tax measurement of CREDIT_PROBE.md, rerun
against the full four-rung critic ladder.

Same design as credit_probe.py (same seed, rows, matched split within
req_self x t_land x margin tercile cells, cluster bootstrap by episode): among
FEASIBLE, BEST-PLACED aerial-ball readings in 2v2 self-play, how does each value head
price PURSUING vs DECLINING at the decision point?

The 4.0-era result (ckpt 27.55B): pricing_V = -0.053+-0.022 (the critic priced
declining HIGHER - the whiff tax), pricing_G = +0.031+-0.013 (the goal critic
favored pursuit). The composition critic exists to remove exactly this avoidance
pressure - so the pre-registered question is whether vdag/v_exp price pursuit
POSITIVELY and whether V still taxes it.

Import compat53 first (python -c "import compat53, ladder_credit_probe as m; m.main()").
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

import RocketSim as rs
from load_checkpoint import load_latest          # compat53 shim -> Pulsar53Policy
from steer_team import DT, SteeredPolicyRho, rollout_team
from team_decline_probe import decline_readings, cluster_boot_diff

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE.parent / "results"
SEED = 20260723          # credit_probe parity
PPT = 2
NPL = 2 * PPT
ROWS = 900_000
N_ARENAS = 24


def matched_split(rd, sub, rng):
    pu = rd["pursued_self"][sub]
    cell = np.zeros(len(sub), np.int64)
    for key in ("req_self", "t_land", "margin"):
        v = rd[key][sub]
        edges = np.quantile(v, [1 / 3, 2 / 3])
        cell = cell * 10 + np.digitize(v, edges)
    sel_p, sel_d = [], []
    for c in np.unique(cell):
        p = sub[(cell == c) & pu]
        d = sub[(cell == c) & ~pu]
        m = min(len(p), len(d))
        if m == 0:
            continue
        sel_p += list(rng.choice(p, m, replace=False))
        sel_d += list(rng.choice(d, m, replace=False))
    return np.array(sel_p), np.array(sel_d)


def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    rs.init(str(HERE.parents[1] / "build" / "collision_meshes"))
    policy, ckpt = load_latest()
    print(f"pinned checkpoint {ckpt.name}", flush=True)
    rng = np.random.default_rng(SEED)

    rec = rollout_team(SteeredPolicyRho(policy.models), PPT, ROWS, SEED,
                       num_arenas=N_ARENAS, want_obs=True)
    rd = decline_readings(rec)
    print(f"rollout {rec['episodes']} eps; {len(rd['row'])} readings "
          f"({time.time()-t0:.0f}s)", flush=True)

    # full ladder for every row (batched, from raw obs)
    vals = {k: np.empty(len(rec["obs"]), np.float32) for k in
            ["v_real", "v_exp", "vdag_min", "v_goal", "v_geo"]}
    with torch.no_grad():
        for i in range(0, len(rec["obs"]), 32768):
            ob = torch.from_numpy(rec["obs"][i:i + 32768].astype(np.float32))
            lad = policy.ladder(ob)
            for k in vals:
                vals[k][i:i + 32768] = lad[k].numpy()
    H = np.maximum(vals["vdag_min"] - vals["v_real"], 0)
    vals["H"] = H
    z = {k: (v - v.mean()) / (v.std() + 1e-9) for k, v in vals.items()}
    print("values computed", {k: f"{v.mean():.3f}/{v.std():.3f}" for k, v in vals.items()},
          flush=True)

    sub = np.flatnonzero(rd["feas_self"] & rd["best_placed"])
    sel_p, sel_d = matched_split(rd, sub, rng)
    ep_p, ep_d = rd["episode"][sel_p], rd["episode"][sel_d]
    rows_p, rows_d = rd["row"][sel_p], rd["row"][sel_d]
    print(f"matched pairs: {len(sel_p)} pursued vs {len(sel_d)} declined "
          f"(of {len(sub)} best-placed feasible)", flush=True)

    res = {"checkpoint": int(ckpt.name), "rows": ROWS,
           "n_matched_per_class": int(len(sel_p)), "n_best_placed": int(len(sub)),
           "n_readings": int(len(rd["row"])),
           "pursue_rate_best_placed": float(rd["pursued_self"][sub].mean())}

    for nm in vals:
        dp = float(vals[nm][rows_p].mean() - vals[nm][rows_d].mean())
        se = cluster_boot_diff(vals[nm][rows_p], vals[nm][rows_d], ep_p, ep_d, seed=1)
        dz = float(z[nm][rows_p].mean() - z[nm][rows_d].mean())
        res[f"pricing_{nm}"] = {"pursue_minus_decline": dp, "se": se,
                                "pursue_minus_decline_z": dz}
        print(f"pricing {nm:8s}: pursue - decline = {dp:+.4f} +- {se:.4f} (z {dz:+.3f})",
              flush=True)

    # horizon disagreement analogues: each optimism head vs V
    for nm in ["v_exp", "vdag_min", "v_goal"]:
        dzf = z[nm] - z["v_real"]
        d = float(dzf[rows_p].mean() - dzf[rows_d].mean())
        se = cluster_boot_diff(dzf[rows_p], dzf[rows_d], ep_p, ep_d, seed=2)
        res[f"horizon_{nm}"] = {"dz_pursue_minus_decline": d, "se": se}
        print(f"horizon {nm:8s}: Dz(p)-Dz(d) = {d:+.3f} +- {se:.3f}", flush=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    p = RESULTS_DIR / f"ladder_credit_probe_{ckpt.name}.json"
    p.write_text(json.dumps(res, indent=1))
    print(f"wrote {p}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
