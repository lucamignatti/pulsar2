"""ROT_DYNAMICS — commitment-direction geometry across policy versions.
Design + frozen interpretations: ../reports/archive/ROT_DYNAMICS.md. Versions are COPIED out of
the rotating pool before reading."""

import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

import RocketSim as rs
from load_checkpoint import load_models
from steer_team import (SteeredPolicyRho, derive_team_direction, rollout_team,
                        team_possession_readings)

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE.parent / "results"
VERSIONS_ROOT = HERE.parents[1] / "build" / "checkpoints_4.0" / "policy_versions"
CACHE = HERE.parent / "data" / "version_cache"
SEED = 20260726
ROWS = 200_000
N_PICK = 8


def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    rs.init(str(HERE.parents[1] / "build" / "collision_meshes"))
    rng = np.random.default_rng(SEED)

    avail = sorted(int(p.name) for p in VERSIONS_ROOT.iterdir() if p.name.isdigit())
    picks = [avail[i] for i in np.linspace(0, len(avail) - 1, N_PICK).astype(int)]
    CACHE.mkdir(parents=True, exist_ok=True)
    for v in picks:  # copy-first: the pool rotates
        dst = CACHE / str(v)
        if not dst.exists():
            shutil.copytree(VERSIONS_ROOT / str(v), dst)
    print(f"versions ({len(avail)} avail): {picks}", flush=True)

    dirs, metas = [], []
    for v in picks:
        models = load_models(CACHE / str(v), names=["SHARED_HEAD", "POLICY"])
        rec = rollout_team(SteeredPolicyRho(models), 1, ROWS, SEED, want_h2=True)
        rd = team_possession_readings(rec)
        try:
            vec, sig, n_pairs = derive_team_direction(rec["h2"], rd, rng)
        except RuntimeError:
            print(f"{v}: no matched pairs, skipped", flush=True)
            continue
        dirs.append(vec)
        metas.append({"version": v, "n_pairs": n_pairs, "sigma": float(sig)})
        print(f"{v}: {n_pairs} pairs sigma {sig:.2f} ({time.time()-t0:.0f}s)", flush=True)

    D = np.stack(dirs)                       # [m, 512] unit rows
    steps = np.array([m["version"] for m in metas], np.float64)
    cos = D @ D.T

    # pairwise cos vs step distance
    pairs = [(i, j, abs(steps[i] - steps[j]), float(cos[i, j]))
             for i in range(len(D)) for j in range(i + 1, len(D))]
    for dist_M in (100e6, 400e6):
        near = [c for _, _, d, c in pairs if d <= dist_M]
        far = [c for _, _, d, c in pairs if d > dist_M]
        print(f"cos within {dist_M/1e6:.0f}M: {np.mean(near):.3f} (n={len(near)}) | "
              f"beyond: {np.mean(far):.3f} (n={len(far)})", flush=True)

    # leave-one-out subspace projection at k = 1..4
    loo = {}
    for k in range(1, 5):
        fr = []
        for i in range(len(D)):
            others = np.delete(D, i, 0)
            _, _, Vt = np.linalg.svd(others, full_matrices=False)
            fr.append(float(np.linalg.norm(D[i] @ Vt[:k].T)))
        loo[k] = fr
        print(f"LOO projection k={k}: mean {np.mean(fr):.3f} min {np.min(fr):.3f}", flush=True)

    # PCA spectrum of the direction set
    _, S, _ = np.linalg.svd(D - D.mean(0), full_matrices=False)
    var = (S ** 2) / (S ** 2).sum()

    res = {"versions": metas,
           "cos_matrix": [[float(c) for c in row] for row in cos],
           "pairwise": [{"d_steps": d, "cos": c} for _, _, d, c in pairs],
           "loo_projection": {str(k): v for k, v in loo.items()},
           "pca_var_frac": [float(x) for x in var]}
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "rot_dynamics.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"saved {out}  ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
