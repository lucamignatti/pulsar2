"""META steering validation, prior-free (the python gate before any C++).

  census  (M1) per head: the achievement-threshold calibration (consecutive-state
          cos distribution), the CALIBRATION CURVE (achievement rate below / in /
          above the rho band - the self-model is only a frontier detector if this
          is monotone), and per-emergent-cluster: prevalence, in-band n, in-band
          non-achievement (= that cluster's knowing-doing gap).
  sweep   (M2) per (head, cluster): derive the cluster direction (achieved-vs-not,
          matched on rho), steer with the cluster-goal rho band gate, paired seeds.
          Pre-registered bar: in-band achievement on the steered cluster rises at
          some alpha in {0.5, 1} without canary collapse; other clusters'
          achievement not collapsing.

No cluster names, no task labels - describe_cluster output is for the human
reading this report only.

Usage:
  python meta_frontier_validate.py census [--ppt 1] [--rows N] [--k 6]
  python meta_frontier_validate.py sweep --head car --cluster 2 [--rows N]
  python meta_frontier_validate.py sweep --head car --cluster auto   # largest gap
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import RocketSim as rs
from frontier_goals import (BAND, ClusterSteeredPolicy, HEADS, bank_nn_stats,
                            build_bank, cluster_bank, derive_cluster_direction,
                            describe_cluster, mine_pairs)
from load_checkpoint import copy_checkpoint, load_models
from steer_team import rollout_team

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
SEED = 20260722
SWEEP_ALPHAS = [0.0, 0.5, 1.0]


def base_rollout(models, ppt, rows, seed):
    return rollout_team(ClusterSteeredPolicy(models), ppt, rows, seed,
                        want_h2=True, want_masks=True, want_goals=True)


def head_analysis(models, rec, head, k, rng):
    bank = build_bank(rec, head, rng)
    clusters = cluster_bank(models, head, bank, k=k)
    pairs = mine_pairs(models, rec, head, bank, clusters, rng)
    return {"nn_dist": bank_nn_stats(bank), "bank": bank,
            "clusters": clusters, "pairs": pairs}


def calibration_curve(pairs):
    """Median attainment by band position - monotone iff the self-model's rho
    genuinely ranks reachability (the frontier-detector validity check)."""
    return {nm: {"n": int((pairs["band_pos"] == p).sum()),
                 "median_attain": float(np.median(pairs["attain"][pairs["band_pos"] == p]))
                 if (pairs["band_pos"] == p).any() else float("nan")}
            for p, nm in ((0, "below"), (1, "in_band"), (2, "above"))}


def cluster_gaps(pairs, k):
    """Per-cluster frontier profile: in-band n, median attain, and the attain IQR
    (the unresolved variance at the frontier - the scheduler's exploration prior;
    the CAUSAL signal that ranks clusters live is steered-vs-control attain)."""
    out = {}
    for c in range(k):
        sel = (pairs["band_pos"] == 1) & (pairs["cluster"] == c)
        n = int(sel.sum())
        a = pairs["attain"][sel]
        out[str(c)] = {"in_band_n": n,
                       "median_attain": float(np.median(a)) if n else float("nan"),
                       "attain_iqr": float(np.subtract(*np.percentile(a, [75, 25]))) if n else float("nan"),
                       "prevalence": float((pairs["cluster"] == c).mean())}
    return out


def canaries(rec):
    return {"touch_ratio": float(rec["touched"].mean()),
            "goals_per_episode": rec["goals"] / rec["episodes"],
            "in_air_ratio": float((~rec["on_ground"]).mean()),
            "kickoff_first_touch_s": float(np.median(rec["kick_first_touch"]))
            if rec["kick_first_touch"] else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["census", "sweep"])
    ap.add_argument("--head", type=str, default="car")
    ap.add_argument("--cluster", type=str, default="auto")
    ap.add_argument("--ppt", type=int, default=1)
    ap.add_argument("--rows", type=int, default=120_000)
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
    print(f"pinned checkpoint {ckpt.name}, mode {args.ppt}v{args.ppt}")
    RESULTS_DIR.mkdir(exist_ok=True)

    if args.cmd == "census":
        rec = base_rollout(models, args.ppt, args.rows, SEED)
        results = {"checkpoint": int(ckpt.name), "mode": f"{args.ppt}v{args.ppt}",
                   "rows": args.rows, "k": args.k, "heads": {},
                   "canaries": canaries(rec)}
        for head in HEADS:
            ha = head_analysis(models, rec, head, args.k, rng)
            cal = calibration_curve(ha["pairs"])
            gaps = cluster_gaps(ha["pairs"], args.k)
            results["heads"][head] = {
                "bank_nn_p10_p50_p90": [float(np.percentile(ha["nn_dist"], p))
                                        for p in (10, 50, 90)],
                "calibration": cal,
                "clusters": {c: {**gaps[c],
                                 "describe": describe_cluster(rec, ha["bank"], ha["clusters"], int(c), head)}
                             for c in gaps},
            }
            print(f"[{head}] median attain below/in/above band = "
                  f"{cal['below']['median_attain']:.3f}/{cal['in_band']['median_attain']:.3f}/"
                  f"{cal['above']['median_attain']:.3f}", flush=True)
            for c, g in gaps.items():
                print(f"   cluster {c}: prev {g['prevalence']:.0%} in-band n={g['in_band_n']} "
                      f"attain {g['median_attain']:.3f} iqr {g['attain_iqr']:.3f} | "
                      f"{results['heads'][head]['clusters'][c]['describe']}", flush=True)
        out = RESULTS_DIR / f"meta_census_{args.ppt}v{args.ppt}_{ckpt.name}.json"

    else:
        head = args.head
        base = base_rollout(models, args.ppt, args.rows, SEED)
        ha = head_analysis(models, base, head, args.k, rng)
        gaps = cluster_gaps(ha["pairs"], args.k)
        if args.cluster == "auto":
            viable = {c: g for c, g in gaps.items() if g["in_band_n"] >= 200}
            cluster = int(max(viable, key=lambda c: viable[c]["attain_iqr"]))
        else:
            cluster = int(args.cluster)
        v, sigma, n_pairs = derive_cluster_direction(base["h2"], ha["pairs"], cluster, rng)
        if v is None:
            print(f"[{head}] cluster {cluster}: pool too thin - not derivable")
            return
        rep_goal = ha["bank"]["goals"][ha["clusters"]["rep_goal_idx"][cluster]]
        print(f"[{head}] cluster {cluster} ({describe_cluster(base, ha['bank'], ha['clusters'], cluster, head)})")
        print(f"  attain {gaps[str(cluster)]['median_attain']:.3f} iqr {gaps[str(cluster)]['attain_iqr']:.3f}, "
              f"{n_pairs} matched pairs, sigma {sigma:.2f}")

        results = {"checkpoint": int(ckpt.name), "head": head, "cluster": cluster,
                   "mode": f"{args.ppt}v{args.ppt}", "rows": args.rows,
                   "n_pairs": n_pairs, "sigma": sigma,
                   "gap_census": gaps[str(cluster)], "alphas": {}}
        for a in SWEEP_ALPHAS:
            pol = ClusterSteeredPolicy(models, head, rep_goal, v, alpha=a, scale=sigma)
            rec = rollout_team(pol, args.ppt, args.rows, SEED + 1,
                               want_h2=True, want_masks=True, want_goals=True)
            # fresh mining on the steered rollout, SAME bank/clusters/tau (the
            # frozen reference frame; live, these refresh per iteration)
            pairs = mine_pairs(models, rec, head, ha["bank"], ha["clusters"], rng)
            sel = (pairs["band_pos"] == 1) & (pairs["cluster"] == cluster)
            other = (pairs["band_pos"] == 1) & (pairs["cluster"] != cluster)
            m = {"cluster_in_band_n": int(sel.sum()),
                 "cluster_median_attain": float(np.median(pairs["attain"][sel])) if sel.any() else float("nan"),
                 "others_median_attain": float(np.median(pairs["attain"][other])) if other.any() else float("nan"),
                 **canaries(rec)}
            results["alphas"][str(a)] = m
            print(f"a={a:+.1f}: cluster attain {m['cluster_median_attain']:.3f} "
                  f"(n={m['cluster_in_band_n']}), others {m['others_median_attain']:.3f} | "
                  f"touch {m['touch_ratio']*100:.2f}% goals/ep {m['goals_per_episode']:.2f} "
                  f"kick {m['kickoff_first_touch_s']:.2f}s", flush=True)
        out = RESULTS_DIR / f"meta_sweep_{head}_c{cluster}_{args.ppt}v{args.ppt}_{ckpt.name}.json"

    results["runtime_s"] = round(time.time() - t0, 1)
    out.write_text(json.dumps(results, indent=2))
    print(f"runtime {results['runtime_s']}s; wrote {out.name}")


if __name__ == "__main__":
    main()
