"""EMERGENCE early-indicator check: pre-deploy backup vs current checkpoint.

For each checkpoint: mechanic-fragment rates (the RND leading indicators),
pace metrics (the energy-PBRS judge), and PLASTICITY measurements (dormant
trunk units + activation effective rank - the 'can it still learn?' answer).

CROSS-LINEAGE CAVEAT (5.0 dynamics, 2026-07-16): the per-100k-STEP counters
(takeoff_attempts, airborne_jump, proto_dribble) count decision steps, and a
5.0 step (tickSkip 8) spans 2x the sim-time of a 4.0 step (tickSkip 4 +
actionDelay 3). To compare against 4.0-era curves, divide the 5.0 counter by
2 (or double the 4.0 value). Fractions (aerial_touch_frac, *_frac) are
per-step-invariant and safe either way.
"""

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
RESULTS_DIR = HERE / "results"
SEED = 20260736
ROWS = 500_000
GOAL_Z = 642.775


def analyze(ckpt_dir: Path):
    models = load_models(ckpt_dir)
    rec = rollout_team(SteeredPolicyRho(models), 1, ROWS, SEED, num_arenas=24,
                       want_h2=True, want_goals=True)
    n = len(rec["slot"])
    phys, on_ground, touched = rec["phys"], rec["on_ground"], rec["touched"]
    jump = ACTION_TABLE[rec["action"].astype(int), 5] > 0.5
    self_v = phys[:, 12:15]
    hspeed = np.linalg.norm(self_v[:, :2], axis=1)
    speed = np.linalg.norm(self_v, axis=1)
    ball_z = phys[:, 2]
    dist_ball = np.hypot(phys[:, 0] - phys[:, 9], phys[:, 1] - phys[:, 10])
    dz_ball = ball_z - phys[:, 11]

    out = {
        "checkpoint": int(ckpt_dir.name),
        # pace (energy judge)
        "mean_speed": float(speed.mean()),
        "supersonic_frac": float((speed > 2200).mean()),
        "slow_frac": float((speed < 500).mean()),
        "jump_while_slow_frac": float((jump & (speed < 500)).mean()),  # flip-in-place proxy
        # mechanic leading indicators (per 100k steps)
        "takeoff_attempts": float((on_ground & jump & (ball_z > GOAL_Z) & (dist_ball < 1200)).sum() / n * 1e5),
        "airborne_jump": float(((~on_ground) & jump).sum() / n * 1e5),
        "proto_dribble": float((touched & (dz_ball > 100) & (dz_ball < 200)).sum() / n * 1e5),
        "aerial_touch_frac": float((touched & (ball_z > GOAL_Z)).sum() / max(touched.sum(), 1)),
    }

    # plasticity: dormant trunk units + effective rank over a 40k-row sample
    h2 = torch.from_numpy(rec["h2"][:40000].astype(np.float32))
    std = h2.std(0)
    out["h2_dormant_frac"] = float((std < 1e-3).float().mean())          # dead units
    out["h2_low_var_frac"] = float((std < 0.01 * std.mean()).float().mean())
    cov = torch.cov(h2.T)
    ev = torch.linalg.eigvalsh(cov).clamp_min(0)
    p = (ev / ev.sum()).clamp_min(1e-12)
    out["h2_effective_rank"] = float(torch.exp(-(p * p.log()).sum()))    # entropy rank
    return out


def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    rs.init(str(HERE.parents[1] / "build" / "collision_meshes"))
    results = {}
    for tag, path in (("pre_deploy", sys.argv[1]), ("current", sys.argv[2])):
        r = analyze(Path(path))
        results[tag] = r
        print(f"{tag} ({r['checkpoint']}): speed {r['mean_speed']:.0f} supersonic "
              f"{r['supersonic_frac']:.1%} slow {r['slow_frac']:.1%} "
              f"flipSlow {r['jump_while_slow_frac']*100:.2f}% | takeoffAtt "
              f"{r['takeoff_attempts']:.1f} dribble {r['proto_dribble']:.1f} "
              f"aerialTouch {r['aerial_touch_frac']:.1%} | dormant "
              f"{r['h2_dormant_frac']:.1%} effRank {r['h2_effective_rank']:.0f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "emergence_check.json").write_text(json.dumps(results, indent=1))
    print("saved", flush=True)


if __name__ == "__main__":
    main()
