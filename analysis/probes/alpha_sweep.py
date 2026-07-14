"""Phase-0a (STEERING_ROADMAP.md): per-checkpoint alpha-sweep dose-response report.

Sweeps steering alpha over [-2, +2] with the v2 possession-outcome panel (steer_v2.py,
in-trainer derivation parity) and paired arena seeds, so the table causally separates
"can't" (steering moves nothing -> capability gap, feed curriculum) from "won't"
(steering moves it -> disposition gap, keep steering). Manual per-checkpoint ritual.

Usage:
  python alpha_sweep.py [--ckpt <dir>] [--rows N] [--wandb]
  --ckpt   checkpoint dir (default: newest under the detected checkpoint root)
  --rows   rows per rollout (default 80k base / 80k per alpha)
  --wandb  additionally log the table to the training wandb project at
           step = checkpoint timestep, so dose-response history accumulates
           next to the training panels (run id "alpha-sweep-<lineage>").
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import RocketSim as rs
import collect_dataset as cd
from load_checkpoint import PulsarPolicy, copy_checkpoint, load_models
from steer_test import SteeredPolicy, rollout
from steer_v2 import derive_direction, possession_metrics, possession_readings

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

SEED = 20260714
ALPHAS = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]  # includes the production +0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--rows", type=int, default=150_000)
    ap.add_argument("--wandb", action="store_true")
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

    # ---- derive the v2 commitment direction from a base rollout
    rng = np.random.default_rng(SEED)
    base = rollout(SteeredPolicy(models), args.rows, SEED, want_h2=True)
    readings = possession_readings(base)
    n_out = [int((readings["outcome"] == o).sum()) for o in (1, 2, 0)]
    print(f"feasible readings: {len(readings['row']):,} (WON {n_out[0]} / LOST {n_out[1]} / NONE {n_out[2]})")
    v, sigma, n_pairs = derive_direction(base["h2"].astype(np.float32), readings, rng)
    print(f"direction: {n_pairs} matched pairs/class, sigma_proj {sigma:.2f}")

    results = {"checkpoint": int(ckpt.name), "n_matched_per_class": n_pairs,
               "sigma_proj": sigma, "rows_per_alpha": args.rows, "alphas": {}}
    for a in ALPHAS:
        pol = SteeredPolicy(models, v, alpha=a, scale=sigma)
        rec = rollout(pol, args.rows, SEED + 1)  # identical seeds across alphas
        m = possession_metrics(rec, max_readings=8000)
        results["alphas"][str(a)] = m
        print(f"alpha {a:+.1f}: possWin {m['poss_win']:.1%} +-{m['poss_win_se']:.1%} "
              f"(lost {m['poss_lost']:.1%}, n={m['n_feasible_readings']} "
              f"in {m['n_reading_episodes']} eps), "
              f"aerial {m['aerial_touch_ratio']*100:.3f}%, touch {m['touch_ratio']*100:.2f}%, "
              f"air {m['in_air_ratio']:.1%}, goals/ep {m['goals_per_episode']:.2f}, "
              f"kickoff {m['kickoff_first_touch_s']:.2f}s ({m['n_kickoffs_touched']})", flush=True)

    results["runtime_s"] = round(time.time() - t0, 1)
    RESULTS_DIR.mkdir(exist_ok=True)
    out_json = RESULTS_DIR / f"alpha_sweep_{ckpt.name}.json"
    out_json.write_text(json.dumps(results, indent=2))

    keys = ["poss_win", "poss_lost", "aerial_touch_ratio", "touch_ratio",
            "in_air_ratio", "goals_per_episode", "kickoff_first_touch_s"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 7))
    xs = [float(a) for a in ALPHAS]
    for ax, k in zip(axes.flat, keys):
        ax.plot(xs, [results["alphas"][str(a)][k] for a in ALPHAS], marker="o")
        ax.axvline(0, color="k", lw=0.5)
        ax.axvline(0.5, color="g", lw=0.5, ls="--")  # production alpha
        ax.set_title(k)
        ax.set_xlabel("alpha (sigma of projection)")
        ax.grid(alpha=0.3)
    axes.flat[-1].axis("off")
    fig.suptitle(f"v2 possession dose-response, checkpoint {ckpt.name}")
    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_png = PLOTS_DIR / f"alpha_sweep_{ckpt.name}.png"
    fig.savefig(out_png, dpi=120)
    print(f"\nruntime {results['runtime_s']}s; wrote {out_json.name} + {out_png.name}")

    if args.wandb:
        import wandb
        run = wandb.init(entity="lucamignatti-personal", project="gigalearncpp",
                         id="alpha-sweep-4-0", name="alpha-sweep-4.0", resume="allow")
        log = {f"AlphaSweep/{k}@{a:+.1f}": results["alphas"][str(a)][k]
               for a in ALPHAS for k in keys}
        log["AlphaSweep/sigma_proj"] = sigma
        log["AlphaSweep/n_matched_pairs"] = n_pairs
        run.log(log, step=int(ckpt.name))
        run.finish()
        print("logged to wandb")


if __name__ == "__main__":
    main()
