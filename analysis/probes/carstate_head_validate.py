"""Car-state head: the natural meta-steering extension, offline-validated BEFORE any
trainer code (the standing rule).

The candidate third goal space is canonical CAR pos+vel - movement/positioning
capability, decoupled from the ball (the achieved stream + HER machinery already
exist in the trainer from the proposer era; only the psi head was never trained).
This script answers, conservatively, "would it work?":

  1. train psi_carstate OFFLINE against the FROZEN phi (the live run trains phi and
     psi jointly, which is strictly more flexible - so a pass here is conservative),
     using the same InfoNCE recipe (anchor = phi(trunk, action TAKEN), positives =
     own achieved car state Delta steps later, in-batch negatives, tau from config)
  2. the horizon is picked BY CALIBRATION, not by hand: train one head per candidate
     window (the self-model's existing horizon family: 20 / 45 / 90 steps) and keep
     the one with the best monotone calibration margin - a data-driven choice
  3. the winning head runs the standard M1 census + M2 causal sweep gates; verdicts
     decide whether the C++ head ships

Usage: python carstate_head_validate.py [--rows N] [--steps N] [--ckpt <dir>]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import RocketSim as rs
import frontier_goals as fg
from frontier_goals import (ClusterSteeredPolicy, bank_nn_stats, build_bank,
                            cluster_bank, derive_cluster_direction,
                            describe_cluster, mine_pairs)
from load_checkpoint import copy_checkpoint, load_models
from steer_team import rollout_team

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
SEED = 20260723
TAU = 0.02          # ReachabilityConfig::tau
# ts8 re-calibration (2026-07-17): the ts4-era family {20,45,90} is void at 15Hz.
# {10,22,45} = the real-time equivalents of ts4 {20,45,90}; 20 kept as the
# same-step-count bridge to the ts4 result; 90 = the long tail (6s).
WINDOWS = [10, 20, 22, 45, 90]
SWEEP_ALPHAS = [0.0, 0.5, 1.0]


def make_psi():
    """Same architecture as the existing psi heads: 6 -> [256,LN,LReLU]x2 -> 128."""
    return torch.nn.Sequential(
        torch.nn.Linear(6, 256), torch.nn.LayerNorm(256), torch.nn.LeakyReLU(),
        torch.nn.Linear(256, 256), torch.nn.LayerNorm(256), torch.nn.LeakyReLU(),
        torch.nn.Linear(256, 128))


def train_psi(rec, models, window, steps, rng, batch=512):
    """InfoNCE against the FROZEN phi, positives = own future achieved car state."""
    from frontier_goals import episode_index
    ep_rows, row_pos = episode_index(rec)
    npl = 2 * rec["ppt"]
    n = len(rec["slot"])
    phi = models["REACH_PHI"]

    # candidate anchor rows: enough same-player future for the window
    cand = np.array([r for r in range(n)
                     if row_pos[r] + window * npl < len(ep_rows[int(rec["episode"][r])])])
    psi = make_psi()
    opt = torch.optim.Adam(psi.parameters(), lr=3e-4)
    h2_all = rec["h2"]
    ach = rec["ach_carstate"]

    for step in range(steps):
        rows = rng.choice(cand, batch, replace=False)
        offs = rng.integers(1, window + 1, batch)
        goal_rows = np.empty(batch, np.int64)
        for i, (r, d) in enumerate(zip(rows, offs)):
            goal_rows[i] = ep_rows[int(rec["episode"][r])][row_pos[r] + d * npl]

        with torch.no_grad():
            h2 = torch.from_numpy(h2_all[rows].astype(np.float32))
            onehot = torch.nn.functional.one_hot(
                torch.from_numpy(rec["action"][rows].astype(np.int64)), 90).float()
            a = phi(torch.cat([h2, onehot], -1))
            a = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        g = psi(torch.from_numpy(ach[goal_rows]))
        g = g / g.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        logits = a @ g.t() / TAU
        loss = torch.nn.functional.cross_entropy(logits, torch.arange(batch))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 300 == 0:
            acc = (logits.argmax(1) == torch.arange(batch)).float().mean()
            print(f"    W={window} step {step}: loss {loss.item():.3f} acc {acc:.2f}", flush=True)

    psi.eval()
    for p in psi.parameters():
        p.requires_grad_(False)
    return psi


def census_for(models, rec, k, rng):
    bank = build_bank(rec, "carstate", rng)
    clusters = cluster_bank(models, "carstate", bank, k=k)
    pairs = mine_pairs(models, rec, "carstate", bank, clusters, rng)
    med = {}
    for p, nm in ((0, "below"), (1, "in_band"), (2, "above")):
        sel = pairs["band_pos"] == p
        med[nm] = float(np.median(pairs["attain"][sel])) if sel.any() else float("nan")
    return med, bank, clusters, pairs


def canaries(rec):
    return {"touch_ratio": float(rec["touched"].mean()),
            "goals_per_episode": rec["goals"] / rec["episodes"],
            "in_air_ratio": float((~rec["on_ground"]).mean()),
            "kickoff_first_touch_s": float(np.median(rec["kick_first_touch"]))
            if rec["kick_first_touch"] else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=120_000)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--ckpt", type=Path, default=None)
    args = ap.parse_args()

    t0 = time.time()
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    rs.init(str(HERE.parents[1] / "build" / "collision_meshes"))
    ckpt = args.ckpt or copy_checkpoint(HERE / "data" / "ckpt_cache")
    models = load_models(ckpt)
    rng = np.random.default_rng(SEED)
    print(f"pinned checkpoint {ckpt.name}")
    RESULTS_DIR.mkdir(exist_ok=True)

    base = rollout_team(ClusterSteeredPolicy(models), 1, args.rows, SEED,
                        want_h2=True, want_masks=True, want_goals=True)

    results = {"checkpoint": int(ckpt.name), "rows": args.rows, "windows": {}}
    best = None
    for W in WINDOWS:
        print(f"training psi_carstate, window {W} steps...", flush=True)
        psi = train_psi(base, models, W, args.steps, rng)
        models["REACH_PSI_CARSTATE"] = psi
        fg.HEADS["carstate"] = {"psi": "REACH_PSI_CARSTATE", "ach": "ach_carstate",
                                "window_steps": W}
        cal, bank, clusters, pairs = census_for(models, base, args.k, rng)
        margin = cal["above"] - cal["below"]
        monotone = cal["below"] <= cal["in_band"] <= cal["above"]
        results["windows"][str(W)] = {"calibration": cal, "monotone": bool(monotone),
                                      "margin": margin}
        print(f"  W={W}: attain below/in/above = {cal['below']:.3f}/{cal['in_band']:.3f}"
              f"/{cal['above']:.3f} monotone={monotone} margin={margin:.3f}", flush=True)
        if monotone and (best is None or margin > best["margin"]):
            best = {"W": W, "psi": psi, "margin": margin,
                    "bank": bank, "clusters": clusters, "pairs": pairs}

    if best is None:
        results["verdict"] = "FAIL: no window yields a monotone calibration"
        print(results["verdict"])
    else:
        W = best["W"]
        print(f"\nbest window {W} (margin {best['margin']:.3f}); M2 causal sweep")
        models["REACH_PSI_CARSTATE"] = best["psi"]
        fg.HEADS["carstate"]["window_steps"] = W
        pairs = best["pairs"]
        gaps = {}
        for c in range(args.k):
            sel = (pairs["band_pos"] == 1) & (pairs["cluster"] == c)
            if sel.sum() >= 200:
                a = pairs["attain"][sel]
                gaps[c] = float(np.subtract(*np.percentile(a, [75, 25])))
        cluster = max(gaps, key=gaps.get)
        v, sigma, n_pairs = derive_cluster_direction(base["h2"], pairs, cluster, rng)
        results["window"] = W
        if v is None:
            results["verdict"] = "FAIL: direction pool too thin"
            print(results["verdict"])
        else:
            rep_goal = best["bank"]["goals"][best["clusters"]["rep_goal_idx"][cluster]]
            print(f"cluster {cluster} "
                  f"({describe_cluster(base, best['bank'], best['clusters'], cluster, 'ball')}), "
                  f"{n_pairs} pairs, sigma {sigma:.2f}")
            results["cluster"] = int(cluster)
            results["alphas"] = {}
            for a in SWEEP_ALPHAS:
                pol = ClusterSteeredPolicy(models, "carstate", rep_goal, v,
                                           alpha=a, scale=sigma)
                rec = rollout_team(pol, 1, args.rows, SEED + 1,
                                   want_h2=True, want_masks=True, want_goals=True)
                p2 = mine_pairs(models, rec, "carstate", best["bank"], best["clusters"], rng)
                sel = (p2["band_pos"] == 1) & (p2["cluster"] == cluster)
                m = {"cluster_median_attain": float(np.median(p2["attain"][sel]))
                     if sel.any() else float("nan"),
                     "n": int(sel.sum()), **canaries(rec)}
                results["alphas"][str(a)] = m
                print(f"a={a:+.1f}: attain {m['cluster_median_attain']:.3f} (n={m['n']}) | "
                      f"touch {m['touch_ratio']*100:.2f}% goals/ep {m['goals_per_episode']:.2f} "
                      f"kick {m['kickoff_first_touch_s']:.2f}s", flush=True)
            a0 = results["alphas"]["0.0"]["cluster_median_attain"]
            aBest = max(results["alphas"][str(a)]["cluster_median_attain"]
                        for a in SWEEP_ALPHAS if a > 0)
            results["verdict"] = ("PASS" if aBest > a0 else "INCONCLUSIVE") \
                + f": uplift {aBest - a0:+.3f} at best positive alpha"
            print("verdict:", results["verdict"])
            torch.save(best["psi"].state_dict(),
                       RESULTS_DIR / f"psi_carstate_W{W}_{ckpt.name}.pt")

    results["runtime_s"] = round(time.time() - t0, 1)
    out = RESULTS_DIR / f"carstate_head_{ckpt.name}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"runtime {results['runtime_s']}s; wrote {out.name}")


if __name__ == "__main__":
    main()
