"""CREDIT_PROBE — whiff tax in the critics' own valuations. Design + frozen
interpretations: CREDIT_PROBE.md."""

import json
import time
from pathlib import Path

import numpy as np
import torch

import RocketSim as rs
from load_checkpoint import copy_checkpoint, load_models, rebuild_sequential
from steer_team import DT, SteeredPolicyRho, rollout_team
from team_decline_probe import decline_readings, cluster_boot_diff

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
SEED = 20260723
PPT = 2
NPL = 2 * PPT
ROWS = 900_000
N_ARENAS = 24
TRAJ_S = 3.0


def matched_split(rd, sub, rng):
    """Within (req_self x t_land x margin) tercile cells, return equal-count
    pursued / declined index arrays (indices into rd arrays)."""
    pu = rd["pursued_self"][sub]
    cell = np.zeros(len(sub), np.int64)
    for j, key in enumerate(("req_self", "t_land", "margin")):
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


def traj_values(V, rd, idx, episode_all, slot_all, ep_rows, row_pos, k_steps):
    """Mean value at reading + each of k_steps same-slot future steps (NaN-padded)."""
    out = np.full((len(idx), k_steps + 1), np.nan)
    for i, ri in enumerate(idx):
        r = rd["row"][ri]
        rows = ep_rows[int(episode_all[r])]
        q = row_pos[r]
        for k in range(k_steps + 1):
            qq = q + NPL * k
            if qq < len(rows):
                out[i, k] = V[rows[qq]]
    return out


def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    rs.init(str(HERE.parents[1] / "build" / "collision_meshes"))
    ckpt = copy_checkpoint(HERE / "data" / "ckpt_cache")
    models = load_models(ckpt)
    goal_critic = rebuild_sequential(
        torch.jit.load(str(ckpt / "GOAL_CRITIC.lt"), map_location="cpu"))
    assert goal_critic[0].in_features == models["SHARED_HEAD"][0].in_features, \
        "goal critic must run on raw obs"
    print(f"pinned checkpoint {ckpt.name} (goal critic loaded)", flush=True)
    rng = np.random.default_rng(SEED)

    rec = rollout_team(SteeredPolicyRho(models), PPT, ROWS, SEED,
                       num_arenas=N_ARENAS, want_h2=True, want_obs=True)
    rd = decline_readings(rec)
    print(f"rollout {rec['episodes']} eps; {len(rd['row'])} readings "
          f"({time.time()-t0:.0f}s)", flush=True)

    # critic values for every row (batched)
    with torch.no_grad():
        V = np.concatenate([models["CRITIC"](torch.from_numpy(
            rec["h2"][i:i + 65536].astype(np.float32))).flatten().numpy()
            for i in range(0, len(rec["h2"]), 65536)])
        G = np.concatenate([goal_critic(torch.from_numpy(
            rec["obs"][i:i + 65536].astype(np.float32))).flatten().numpy()
            for i in range(0, len(rec["obs"]), 65536)])
    zV, zG = (V - V.mean()) / V.std(), (G - G.mean()) / G.std()
    print(f"values: V mean {V.mean():.3f} std {V.std():.3f} | "
          f"G mean {G.mean():.3f} std {G.std():.3f}", flush=True)

    episode, slot = rec["episode"], rec["slot"]
    ep_rows = {int(e): np.flatnonzero(episode == e) for e in np.unique(episode)}
    row_pos = np.empty(len(slot), np.int64)
    for rows in ep_rows.values():
        row_pos[rows] = np.arange(len(rows))

    sub = np.flatnonzero(rd["feas_self"] & rd["best_placed"])
    sel_p, sel_d = matched_split(rd, sub, rng)
    ep_p, ep_d = rd["episode"][sel_p], rd["episode"][sel_d]
    rows_p, rows_d = rd["row"][sel_p], rd["row"][sel_d]
    print(f"matched pairs: {len(sel_p)} pursued vs {len(sel_d)} declined "
          f"(from {len(sub)} best-placed feasible readings)", flush=True)

    res = {"checkpoint": int(ckpt.name), "rows": ROWS,
           "n_matched_per_class": int(len(sel_p)), "n_best_placed": int(len(sub))}

    # 1) pricing at the decision
    for nm, val, z in (("V", V, zV), ("G", G, zG)):
        dp = float(val[rows_p].mean() - val[rows_d].mean())
        se = cluster_boot_diff(val[rows_p], val[rows_d], ep_p, ep_d, seed=1)
        res[f"pricing_{nm}"] = {"pursue_minus_decline": dp, "se": se,
                                "pursue_minus_decline_z": float(z[rows_p].mean() - z[rows_d].mean())}
        print(f"pricing {nm}: pursue - decline = {dp:+.4f} +- {se:.4f} "
              f"(z {res[f'pricing_{nm}']['pursue_minus_decline_z']:+.3f})", flush=True)

    # 3) horizon disagreement at the decision
    dz = zG - zV
    ddz = float(dz[rows_p].mean() - dz[rows_d].mean())
    se_dz = cluster_boot_diff(dz[rows_p], dz[rows_d], ep_p, ep_d, seed=2)
    res["horizon_disagreement"] = {"dz_pursue_minus_decline": ddz, "se": se_dz}
    print(f"horizon: Dz(pursue) - Dz(decline) = {ddz:+.3f} +- {se_dz:.3f}", flush=True)

    # 2) tax trajectory over TRAJ_S
    k_steps = int(round(TRAJ_S / DT))
    for nm, val in (("V", zV), ("G", zG)):
        tp = traj_values(val, rd, sel_p, episode, slot, ep_rows, row_pos, k_steps)
        td = traj_values(val, rd, sel_d, episode, slot, ep_rows, row_pos, k_steps)
        # delta from the reading value, averaged over the last second of the window
        tail = slice(k_steps - int(1 / DT), k_steps + 1)
        drop_p = np.nanmean(tp[:, tail], 1) - tp[:, 0]
        drop_d = np.nanmean(td[:, tail], 1) - td[:, 0]
        ok_p, ok_d = np.isfinite(drop_p), np.isfinite(drop_d)
        d = float(np.nanmean(drop_p) - np.nanmean(drop_d))
        se = cluster_boot_diff(drop_p[ok_p], drop_d[ok_d], ep_p[ok_p], ep_d[ok_d], seed=3)
        res[f"traj_{nm}"] = {
            "drop_pursue": float(np.nanmean(drop_p)), "drop_decline": float(np.nanmean(drop_d)),
            "pursue_minus_decline": d, "se": se,
            "curve_pursue": [float(x) for x in np.nanmean(tp, 0)[::9]],
            "curve_decline": [float(x) for x in np.nanmean(td, 0)[::9]]}
        print(f"traj {nm} (z, 2-3s drop): pursue {np.nanmean(drop_p):+.3f} "
              f"decline {np.nanmean(drop_d):+.3f} diff {d:+.3f} +- {se:.3f}", flush=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"credit_probe_{ckpt.name}.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"saved {out}  ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
