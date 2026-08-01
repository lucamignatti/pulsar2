"""Cross-critic analysis of the four-rung optimism ladder on an on-policy dataset.

Measures, per GEOMETRIC_CRITIC.md section 10 ("the rungs' disagreements carry the
information"):
  - ladder ordering + violation rates (V <= V_exp <= V_dag-min should hold in tendency)
  - cross-critic correlation structure (pearson + spearman)
  - WHERE each optimism field concentrates: headroom H (composition), h_geo (affine-
    matched geometric gap), V_exp - V (knowing-doing gap) binned by state features
  - top-vs-bottom decile state characterization for each field
  - geo internals: sigma anisotropy by obs dim group, HJB residual on-policy,
    r_hat vs realized touch/goal proximity
  - goal-critic vs critic disagreement (the critic-duel read, 5.3 edition)

Input: research/data/dataset53.npz (collect_dataset_53.py).
Output: research/results/critic_ladder_<ckpt>.json
"""

import json
import os
from pathlib import Path

import numpy as np
import torch

from load_checkpoint_53 import Pulsar53Policy, load_models, list_checkpoints

DATA = Path(__file__).resolve().parents[1] / "data"
RESULTS = Path(__file__).resolve().parents[1] / "results"

GOAL_Y = 5120.0


def summarize(x):
    x = np.asarray(x, np.float64)
    return {"mean": float(x.mean()), "std": float(x.std()),
            "p10": float(np.percentile(x, 10)), "p50": float(np.percentile(x, 50)),
            "p90": float(np.percentile(x, 90)), "p99": float(np.percentile(x, 99))}


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    d = np.load(DATA / "dataset53.npz")
    ckpt = int(d["checkpoint"])
    n = len(d["action"])
    print(f"dataset: {n:,} frames, checkpoint {ckpt:,}")

    team = d["team"].astype(int)
    phys = d["phys"]
    ball = phys[:, 0:3]
    ball_vel = phys[:, 3:6]
    self_pos = np.where(team[:, None] == 0, phys[:, 9:12], phys[:, 20:23])
    self_vel = np.where(team[:, None] == 0, phys[:, 12:15], phys[:, 23:26])
    self_boost = np.where(team == 0, phys[:, 18], phys[:, 29])
    self_ground = np.where(team == 0, phys[:, 19], phys[:, 30]).astype(bool)
    opp_pos = np.where(team[:, None] == 0, phys[:, 20:23], phys[:, 9:12])

    dist_ball = np.linalg.norm(self_pos - ball, axis=1)
    opp_dist_ball = np.linalg.norm(opp_pos - ball, axis=1)
    speed = np.linalg.norm(self_vel, axis=1)
    # team-canonical: attacked net is +y for team 0 (blue), -y for team 1
    attack_sign = np.where(team == 0, 1.0, -1.0)
    ball_to_goal = np.linalg.norm(
        np.stack([ball[:, 0], attack_sign * GOAL_Y - ball[:, 1], ball[:, 2]], 1), axis=1)
    ball_vel_to_goal = attack_sign * ball_vel[:, 1]

    lad = {k: d[f"ladder_{k}"].astype(np.float64) for k in
           ["v_real", "v_exp", "vdag1", "vdag2", "v_goal", "v_geo", "r_hat"]}
    vdag_min = np.minimum(lad["vdag1"], lad["vdag2"])
    H = np.maximum(vdag_min - lad["v_real"], 0)
    kd_gap = lad["v_exp"] - lad["v_real"]
    g, v = lad["v_geo"], lad["v_real"]
    geo_scaled = (g - g.mean()) / (g.std() + 1e-8) * v.std() + v.mean()
    h_geo = np.maximum(geo_scaled - v, 0)

    out = {"checkpoint": ckpt, "n_frames": n,
           "n_episodes": int(d["episode"].max() + 1)}

    # ---- 1. ladder ordering ----
    out["rungs"] = {k: summarize(x) for k, x in
                    {**lad, "vdag_min": vdag_min, "H": H, "kd_gap": kd_gap,
                     "h_geo_affine": h_geo}.items()}
    out["ordering"] = {
        "frac_vexp_below_vreal": float((lad["v_exp"] < lad["v_real"]).mean()),
        "frac_vdagmin_below_vreal": float((vdag_min < lad["v_real"]).mean()),
        "frac_vdagmin_below_vexp": float((vdag_min < lad["v_exp"]).mean()),
        "twin_spread_mean": float(np.abs(lad["vdag1"] - lad["vdag2"]).mean()),
        "twin_corr": float(np.corrcoef(lad["vdag1"], lad["vdag2"])[0, 1]),
        "frac_H_positive": float((H > 0).mean()),
        "frac_hgeo_positive": float((h_geo > 0).mean()),
    }

    # ---- 2. cross-critic correlations ----
    fields = {"v_real": lad["v_real"], "v_exp": lad["v_exp"], "vdag_min": vdag_min,
              "v_goal": lad["v_goal"], "v_geo": lad["v_geo"], "r_hat": lad["r_hat"],
              "H": H, "kd_gap": kd_gap, "h_geo": h_geo}
    names = list(fields)
    out["pearson"] = {f"{a}~{b}": float(np.corrcoef(fields[a], fields[b])[0, 1])
                      for i, a in enumerate(names) for b in names[i + 1:]}
    sub = np.random.default_rng(0).choice(n, min(n, 40000), replace=False)
    out["spearman"] = {f"{a}~{b}": spearman(fields[a][sub], fields[b][sub])
                       for i, a in enumerate(names) for b in names[i + 1:]}

    # ---- 3. where the optimism fields live ----
    feats = {"ball_z": ball[:, 2], "self_z": self_pos[:, 2], "dist_ball": dist_ball,
             "boost": self_boost, "speed": speed, "ball_to_goal": ball_to_goal,
             "ball_vel_to_goal": ball_vel_to_goal, "opp_dist_ball": opp_dist_ball,
             "airborne": (~self_ground).astype(np.float64)}
    bins = {"ball_z": [0, 130, 300, 600, 1000, 1500, 2044],
            "self_z": [0, 50, 300, 600, 1000, 2044],
            "dist_ball": [0, 400, 800, 1500, 3000, 6000, 13000],
            "boost": [0, 10, 33, 66, 100.1],
            "ball_to_goal": [0, 1500, 3000, 5000, 8000, 12000]}
    cond = {}
    for fname, edges in bins.items():
        f = feats[fname]
        rows = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (f >= lo) & (f < hi)
            if m.sum() < 200:
                rows.append({"bin": [lo, hi], "n": int(m.sum())})
                continue
            rows.append({"bin": [lo, hi], "n": int(m.sum()),
                         "H": float(H[m].mean()), "h_geo": float(h_geo[m].mean()),
                         "kd_gap": float(kd_gap[m].mean()),
                         "v_real": float(lad["v_real"][m].mean()),
                         "v_geo": float(lad["v_geo"][m].mean()),
                         "v_goal": float(lad["v_goal"][m].mean())})
        cond[fname] = rows
    out["binned"] = cond
    airm = ~self_ground
    out["airborne_split"] = {
        "H_air": float(H[airm].mean()), "H_ground": float(H[~airm].mean()),
        "h_geo_air": float(h_geo[airm].mean()), "h_geo_ground": float(h_geo[~airm].mean()),
        "kd_air": float(kd_gap[airm].mean()), "kd_ground": float(kd_gap[~airm].mean()),
        "frac_air": float(airm.mean())}

    # ---- 4. decile characterization ----
    def decile_profile(field):
        lo_t, hi_t = np.percentile(field, 10), np.percentile(field, 90)
        lo_m, hi_m = field <= lo_t, field >= hi_t
        return {fn: {"bottom_decile": float(fv[lo_m].mean()),
                     "top_decile": float(fv[hi_m].mean()),
                     "overall": float(fv.mean())} for fn, fv in feats.items()}
    out["decile_profiles"] = {"H": decile_profile(H), "h_geo": decile_profile(h_geo),
                              "kd_gap": decile_profile(kd_gap),
                              "v_geo": decile_profile(lad["v_geo"])}

    # ---- 5. goal-critic duel ----
    zr = (lad["v_real"] - lad["v_real"].mean()) / lad["v_real"].std()
    zg = (lad["v_goal"] - lad["v_goal"].mean()) / lad["v_goal"].std()
    dis = zg - zr
    out["goal_duel"] = {
        "corr": float(np.corrcoef(lad["v_real"], lad["v_goal"])[0, 1]),
        "spearman": spearman(lad["v_real"][sub], lad["v_goal"][sub]),
        "disagree_profile_goalcritic_higher": {
            fn: float(fv[dis >= np.percentile(dis, 90)].mean()) for fn, fv in feats.items()},
        "disagree_profile_critic_higher": {
            fn: float(fv[dis <= np.percentile(dis, 10)].mean()) for fn, fv in feats.items()},
    }

    # ---- 6. geo internals: sigma anisotropy + HJB residual (model pass, subsample) ----
    root = Path(os.environ.get("PULSAR_CKPT_ROOT", DATA / "ckpt53"))
    pol = Pulsar53Policy(load_models(list_checkpoints(root)[0]))
    obs_t = torch.from_numpy(d["obs"][sub[:8000]].astype(np.float32))
    mu, sig = pol.geo_sigma(obs_t)
    res = pol.hjb_residual(obs_t)
    sig = sig.numpy()
    # AdvancedObsPadded layout: ball pos/2300 [0:3], ball vel /2300 [3:6] (header),
    # exact indices matter less than the group contrast; we report per-dim top movers.
    dim_mean = sig.mean(0)
    top = np.argsort(-dim_mean)[:15]
    out["geo_sigma"] = {
        "mean_overall": float(sig.mean()),
        "per_dim_top15": [[int(i), float(dim_mean[i])] for i in top],
        "dim_mean_first30": [float(x) for x in dim_mean[:30]],
    }
    out["hjb_residual_onpolicy"] = summarize(res.numpy() ** 2)
    out["hjb_resid_note"] = "live Geo/Residual panel is mean squared residual on train states"

    # r_hat sanity: does it price arrival states with the ball near goal?
    out["r_hat_vs_features"] = {fn: float(np.corrcoef(lad["r_hat"], fv)[0, 1])
                                for fn, fv in feats.items()}
    out["H_vs_features"] = {fn: float(np.corrcoef(H, fv)[0, 1]) for fn, fv in feats.items()}
    out["hgeo_vs_features"] = {fn: float(np.corrcoef(h_geo, fv)[0, 1]) for fn, fv in feats.items()}
    out["vgeo_vs_features"] = {fn: float(np.corrcoef(lad["v_geo"], fv)[0, 1])
                               for fn, fv in feats.items()}

    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"critic_ladder_{ckpt}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")
    print(json.dumps(out["ordering"], indent=1))
    print(json.dumps(out["airborne_split"], indent=1))


if __name__ == "__main__":
    main()
