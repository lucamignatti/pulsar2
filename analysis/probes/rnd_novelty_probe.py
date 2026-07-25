"""RC1 offline novelty gate (EMERGENCE.md). Does RND novelty land ON mechanic-
attempt states (which RC2's advantage-surprise miner AVOIDED)? RND = a small
predictor trained toward a FROZEN random projection of trunk(h2)+onehot(action);
novelty = per-row prediction MSE. Bar: enrichment > 1.3x on >=2 of
{pre-landing, grounded-high-ball, proto-dribble}."""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import RocketSim as rs
from advanced_obs import ACTION_TABLE
from fear_decomp import copy_newest
from load_checkpoint import load_models
from steer_team import SteeredPolicyRho, rollout_team

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
SEED = 20260734
ROWS = 600_000
NPL = 2
FEAT = 512 + 90


def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    rs.init(str(HERE.parents[1] / "build" / "collision_meshes"))
    ckpt = copy_newest(HERE / "data" / "ckpt_cache")
    models = load_models(ckpt)
    print(f"pinned checkpoint {ckpt.name}", flush=True)

    rec = rollout_team(SteeredPolicyRho(models), 1, ROWS, SEED, num_arenas=24,
                       want_h2=True, want_goals=True)
    n = len(rec["slot"])
    h2 = rec["h2"].astype(np.float32)
    act = rec["action"].astype(int)
    onehot = np.zeros((n, 90), np.float32)
    onehot[np.arange(n), act] = 1.0
    X = torch.from_numpy(np.concatenate([h2, onehot], 1))

    # frozen random target + trained predictor (RND)
    g = torch.Generator().manual_seed(SEED)
    target = nn.Sequential(nn.Linear(FEAT, 256), nn.LeakyReLU(), nn.Linear(256, 128))
    pred = nn.Sequential(nn.Linear(FEAT, 256), nn.LeakyReLU(), nn.Linear(256, 256),
                         nn.LeakyReLU(), nn.Linear(256, 128))
    for p in target.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = (X - mu) / sd
    with torch.no_grad():
        T = target(Xn)
    opt = torch.optim.Adam(pred.parameters(), 1e-3)
    idx = torch.randperm(n)
    for epoch in range(3):
        for i in range(0, n, 4096):
            b = idx[i:i + 4096]
            opt.zero_grad()
            loss = ((pred(Xn[b]) - T[b]) ** 2).mean()
            loss.backward()
            opt.step()
        print(f"  RND epoch {epoch}: loss {loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)
    with torch.no_grad():
        novelty = ((pred(Xn) - T) ** 2).mean(1).numpy()

    # family masks (self block at 9:.. in phys; team-canonical not needed for these)
    phys, on_ground = rec["phys"], rec["on_ground"]
    self_z = phys[:, 11]           # self pos z is phys col 9,10,11
    self_vz = phys[:, 14]          # self vel z
    ball_z = phys[:, 2]
    dist = np.hypot(phys[:, 0] - phys[:, 9], phys[:, 1] - phys[:, 10])
    boost = phys[:, 15]
    ball_rel = np.hypot(phys[:, 0] - phys[:, 9], phys[:, 1] - phys[:, 10])
    dz_ball = ball_z - self_z

    fam = {
        "pre_landing": (~on_ground) & (self_z < 300) & (self_vz < 0),
        "grounded_high_ball": on_ground & (ball_z > 642.775) & (dist < 1200),
        "proto_dribble": rec["touched"] & (dz_ball > 100) & (dz_ball < 200),
    }
    # top-novelty decile = "where RC1's bonus would concentrate"
    thr = np.quantile(novelty, 0.9)
    top = novelty >= thr
    res = {"checkpoint": int(ckpt.name), "rows": n, "families": {}}
    print(f"top-decile novelty threshold {thr:.4f}", flush=True)
    for name, m in fam.items():
        base = m.mean()
        top_share = m[top].mean()
        enr = float(top_share / base) if base > 1e-6 else float("nan")
        res["families"][name] = {"base": float(base), "top_share": float(top_share),
                                 "enrichment": enr}
        print(f"  {name:20s}: base {base:.4f} top-decile {top_share:.4f} "
              f"enrichment {enr:.2f}x", flush=True)
    # also correlate novelty with boost (RC2 found low-boost dominates surprise)
    res["novelty_boost_corr"] = float(np.corrcoef(novelty, boost)[0, 1])
    res["novelty_ballz_corr"] = float(np.corrcoef(novelty, ball_z)[0, 1])
    passes = sum(1 for f in res["families"].values() if f["enrichment"] > 1.3)
    res["n_families_pass"] = passes
    res["PASS"] = passes >= 2
    print(f"novelty~boost corr {res['novelty_boost_corr']:+.2f}, ~ballz "
          f"{res['novelty_ballz_corr']:+.2f}; families>1.3x: {passes}/3 -> "
          f"{'PASS' if res['PASS'] else 'FAIL'}", flush=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"rnd_novelty_{ckpt.name}.json").write_text(json.dumps(res, indent=1))
    print(f"saved ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
