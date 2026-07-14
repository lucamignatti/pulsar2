"""Phase-0b (STEERING_ROADMAP.md): random-direction baseline for the steering effect.

Samples N random unit directions in trunk space plus a few variants of the derived
commitment direction orthogonalized against it, and runs each through the SAME
paired-seed causal rollout as the 0a sweep at the production alpha (+0.5) and at +1.0.

Three pre-registered outcomes:
  random ~ derived   -> the effect is partly "any coherent perturbation"
                        (parameter-space-noise-in-representation-space; changes the
                        headline story, suggests cheap direction mining)
  random << derived  -> hardens the claim that the DERIVED direction is special
  some random >> derived -> gate-filtered direction search becomes a real method

Usage: python random_dirs.py [--ckpt <dir>] [--rows N] [--n-random K]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import RocketSim as rs
import collect_dataset as cd
from load_checkpoint import PulsarPolicy, copy_checkpoint, load_models
from steer_test import SteeredPolicy, rollout
from steer_v2 import derive_direction, possession_metrics, possession_readings

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"

SEED = 20260715
ALPHAS = [0.5, 1.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--rows", type=int, default=80_000)
    ap.add_argument("--n-random", type=int, default=8)
    ap.add_argument("--n-ortho", type=int, default=3)
    args = ap.parse_args()

    t0 = time.time()
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    rs.init(str(HERE.parents[1] / "build" / "collision_meshes"))

    ckpt = args.ckpt or copy_checkpoint(HERE / "data" / "ckpt_cache")
    models = load_models(ckpt)
    policy = PulsarPolicy(models)
    cd.set_obs_size(policy.obs_size)
    print(f"pinned checkpoint {ckpt.name} (obs {policy.obs_size})")

    rng = np.random.default_rng(SEED)
    base = rollout(SteeredPolicy(models), args.rows, SEED, want_h2=True)
    readings = possession_readings(base)
    v, sigma, n_pairs = derive_direction(base["h2"].astype(np.float32), readings, rng)
    print(f"derived direction: {n_pairs} pairs/class, sigma {sigma:.2f}")
    h2f = base["h2"].astype(np.float32)

    # Direction battery: derived + orthogonalized-derived + pure-random
    dirs = {"derived": v}
    for i in range(args.n_ortho):
        r = rng.standard_normal(512).astype(np.float32)
        r -= (r @ v) * v  # least-squares orthogonalization against the derived direction
        dirs[f"ortho{i}"] = r / np.linalg.norm(r)
    for i in range(args.n_random):
        r = rng.standard_normal(512).astype(np.float32)
        dirs[f"random{i}"] = r / np.linalg.norm(r)

    results = {"checkpoint": int(ckpt.name), "rows": args.rows,
               "n_pairs": n_pairs, "directions": {}}

    # Baseline (alpha 0) with the same paired seed, for the noise floor
    m0 = possession_metrics(rollout(SteeredPolicy(models), args.rows, SEED + 1))
    results["baseline"] = m0
    print(f"alpha 0 baseline: possWin {m0['poss_win']:.1%} +-{m0['poss_win_se']:.1%} "
          f"(n={m0['n_feasible_readings']})", flush=True)

    for name, d in dirs.items():
        # per-direction sigma: projections onto THIS direction (trainer parity: dose is
        # in units of the direction's own live projection std)
        with np.errstate(all="ignore"):  # spurious Accelerate FP flags (see steer_v2)
            sig = float(np.std(h2f @ d))
        results["directions"][name] = {"sigma": sig, "alphas": {}}
        for a in ALPHAS:
            pol = SteeredPolicy(models, d, alpha=a, scale=sig)
            m = possession_metrics(rollout(pol, args.rows, SEED + 1))
            results["directions"][name]["alphas"][str(a)] = m
            print(f"{name:>9} a={a:+.1f}: possWin {m['poss_win']:.1%} +-{m['poss_win_se']:.1%} "
                  f"(dvs0 {m['poss_win'] - m0['poss_win']:+.1%}), touch {m['touch_ratio']*100:.2f}%, "
                  f"goals/ep {m['goals_per_episode']:.2f}, "
                  f"kickoff {m['kickoff_first_touch_s']:.2f}s", flush=True)

    results["runtime_s"] = round(time.time() - t0, 1)
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"random_dirs_{ckpt.name}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nruntime {results['runtime_s']}s; wrote {out.name}")


if __name__ == "__main__":
    main()
