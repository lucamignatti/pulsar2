"""Validate the user-reported symptoms (2026-07-16) with a pre/post A/B:
FAR-DEAD (far from ball, no movement), FLOP-NEAR (near uncontested ball,
flips without touching), uncontested touch conversion. Checkpoints: pre-deploy
backup 35.0B vs current-era 39.2B (RND + energy-6 in between). Per mode."""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import RocketSim as rs
from advanced_obs import ACTION_TABLE
from load_checkpoint import load_models
from steer_team import SteeredPolicyRho, rollout_team

HERE = Path(__file__).resolve().parent
SEED = 20260737
ROWS = 400_000


def analyze(models, ppt):
    npl = 2 * ppt
    rec = rollout_team(SteeredPolicyRho(models), ppt, ROWS, SEED, num_arenas=24,
                       want_goals=True)
    n = len(rec["slot"])
    phys, touched, og = rec["phys"], rec["touched"], rec["on_ground"]
    episode, slot = rec["episode"], rec["slot"]
    jump = ACTION_TABLE[rec["action"].astype(int), 5] > 0.5
    speed = np.linalg.norm(phys[:, 12:15], axis=1)
    dist_ball = np.linalg.norm(phys[:, 0:2] - phys[:, 9:11], axis=1)

    # opponent distance to BALL via block pairing (min over other-parity rows)
    opp_dist = np.full(n, np.inf, np.float32)
    base = np.arange(n) - (np.arange(n) % npl)
    for s in range(npl):
        rows_s = base + s
        d_s = np.linalg.norm(phys[rows_s, 0:2] - phys[rows_s, 9:11], axis=1)
        mask = (s % 2) != (np.arange(n) % npl % 2)
        opp_dist = np.where(mask, np.minimum(opp_dist, d_s), opp_dist)

    far = dist_ball > 2500
    near_unc = (dist_ball < 800) & (opp_dist > 2000) & (phys[:, 2] < 400)

    # touch within 1.5s (same player) for near-uncontested rows
    ep_rows = {int(e): np.flatnonzero(episode == e) for e in np.unique(episode)}
    row_pos = np.empty(n, np.int64)
    for rows in ep_rows.values():
        row_pos[rows] = np.arange(len(rows))
    k = int(1.5 * 30)
    conv = []
    nu_idx = np.flatnonzero(near_unc)
    stride = max(1, len(nu_idx) // 4000)
    for r in nu_idx[::stride]:
        rows = ep_rows[int(episode[r])]
        q = int(row_pos[r])
        end = min(len(rows), q + npl * k)
        conv.append(bool(touched[rows[q:end:npl]].any()))
    return {
        "far_frac": float(far.mean()),
        "far_dead_frac": float((speed[far] < 300).mean()),        # not moving
        "far_mean_speed": float(speed[far].mean()),
        "nearunc_frac": float(near_unc.mean()),
        "nearunc_jump_rate": float(jump[near_unc].mean()),        # flopping proxy
        "nearunc_airborne_frac": float((~og[near_unc]).mean()),
        "nearunc_touch_1_5s": float(np.mean(conv)) if conv else None,
        "touch_ratio": float(touched.mean()),
    }


def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    rs.init(str(HERE.parents[1] / "build" / "collision_meshes"))
    out = {}
    for tag, path in (("pre_35.0B", sys.argv[1]), ("post_39.2B", sys.argv[2])):
        models = load_models(Path(path))
        for ppt in (1, 2):
            r = analyze(models, ppt)
            out[f"{tag}_{ppt}v{ppt}"] = r
            print(f"{tag} {ppt}v{ppt}: farDead {r['far_dead_frac']:.1%} "
                  f"(farSpeed {r['far_mean_speed']:.0f}) | nearUnc jump "
                  f"{r['nearunc_jump_rate']:.1%} air {r['nearunc_airborne_frac']:.1%} "
                  f"touch1.5s {r['nearunc_touch_1_5s']:.1%} | touch {r['touch_ratio']*100:.2f}% "
                  f"({time.time()-t0:.0f}s)", flush=True)
    (HERE / "results" / "behavior_symptoms.json").write_text(json.dumps(out, indent=1))
    print("saved", flush=True)


if __name__ == "__main__":
    main()
