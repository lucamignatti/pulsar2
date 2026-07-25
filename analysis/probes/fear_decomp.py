"""FEAR_DECOMP — decompose the critic's validated fear onto interpretable state
features. Design + frozen interpretation guide: FEAR_DECOMP.md."""

import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

import RocketSim as rs
from load_checkpoint import load_models, rebuild_sequential
from steer_team import SteeredPolicyRho, rollout_team
from team_decline_probe import decline_readings

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
LIVE_ROOT = HERE.parents[1] / "build" / "checkpoints_5.0v3"
SEED = 20260728
PPT = 2
NPL = 2 * PPT
ROWS = 900_000
N_ARENAS = 24
BOOT = 300

NEEDED = ["SHARED_HEAD.lt", "POLICY.lt", "CRITIC.lt", "GOAL_CRITIC.lt",
          "REACH_PHI.lt", "REACH_PSI_BALL.lt", "REACH_PSI_CAR.lt"]


def copy_newest(cache_root: Path) -> Path:
    """Selective copy-first of the newest live checkpoint (retry next-newest once)."""
    cache_root.mkdir(parents=True, exist_ok=True)
    cands = sorted((int(p.name) for p in LIVE_ROOT.iterdir() if p.name.isdigit()),
                   reverse=True)
    for ts in cands[:2]:
        dst = cache_root / str(ts)
        try:
            dst.mkdir(exist_ok=True)
            for f in NEEDED:
                shutil.copy2(LIVE_ROOT / str(ts) / f, dst / f)
            return dst
        except OSError:
            shutil.rmtree(dst, ignore_errors=True)
    raise RuntimeError("no complete checkpoint could be copied")


def reading_features(rec, rd):
    """Interpretable, team-canonical feature matrix per reading."""
    phys, slot = rec["phys"], rec["slot"]
    n = len(rd["row"])
    feats = {}
    r = rd["row"]
    team = (r % NPL) % 2                       # slot parity = team
    flip = np.where(team == 0, 1.0, -1.0)      # canonical: attack +y

    self_pos = phys[r, 9:12]
    self_vel = phys[r, 12:15]
    ball_pos = phys[r, 0:3]
    ball_vel = phys[r, 3:6]

    feats["t_land"] = rd["t_land"]
    feats["d_landing"] = rd["d_now"]
    feats["req_self"] = rd["req_self"]
    feats["margin_tm"] = np.clip(rd["margin"], -3000, 3000)
    # min opponent required speed to the landing (contestedness): opponents are
    # the other-parity rows of the same arena-step block
    land = np.stack([rd["land_x"], rd["land_y"]], 1)
    min_opp_req = np.full(n, np.inf, np.float32)
    for k, rr in enumerate(r):
        base = int(rr) - int(slot[rr])
        for s in range(NPL):
            if s % 2 == team[k]:
                continue
            d = np.linalg.norm(phys[base + s, 9:11] - land[k])
            min_opp_req[k] = min(min_opp_req[k], d / max(rd["t_land"][k], 1e-6))
    feats["min_opp_req"] = np.clip(min_opp_req, 0, 6000)
    feats["landing_y_canon"] = rd["land_y"] * flip
    own_goal_land = np.stack([np.zeros(n), -5120 * flip], 1)
    feats["landing_dist_own_goal"] = np.linalg.norm(land - own_goal_land, axis=1)
    feats["boost"] = phys[r, 15]
    feats["self_y_canon"] = self_pos[:, 1] * flip
    own_goal = np.stack([np.zeros(n), -5120 * flip], 1)
    feats["self_dist_own_goal"] = np.linalg.norm(self_pos[:, :2] - own_goal, axis=1)
    feats["ball_y_canon"] = ball_pos[:, 1] * flip
    feats["ball_vy_canon"] = ball_vel[:, 1] * flip      # negative = toward own goal
    feats["ball_z"] = ball_pos[:, 2]
    feats["ball_speed"] = np.linalg.norm(ball_vel, axis=1)
    feats["self_speed"] = np.linalg.norm(self_vel, axis=1)
    feats["on_ground"] = rec["on_ground"][r].astype(np.float32)
    return feats


def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    rs.init(str(HERE.parents[1] / "build" / "collision_meshes"))
    ckpt = copy_newest(HERE / "data" / "ckpt_cache")
    models = load_models(ckpt)
    goal_critic = rebuild_sequential(
        torch.jit.load(str(ckpt / "GOAL_CRITIC.lt"), map_location="cpu"))
    print(f"pinned checkpoint {ckpt.name}", flush=True)
    rng = np.random.default_rng(SEED)

    rec = rollout_team(SteeredPolicyRho(models), PPT, ROWS, SEED,
                       num_arenas=N_ARENAS, want_h2=True, want_obs=True)
    rd = decline_readings(rec)
    print(f"rollout {rec['episodes']} eps; {len(rd['row'])} readings "
          f"({time.time()-t0:.0f}s)", flush=True)

    with torch.no_grad():
        V = np.concatenate([models["CRITIC"](torch.from_numpy(
            rec["h2"][i:i + 65536].astype(np.float32))).flatten().numpy()
            for i in range(0, len(rec["h2"]), 65536)])
        G = np.concatenate([goal_critic(torch.from_numpy(
            rec["obs"][i:i + 65536].astype(np.float32))).flatten().numpy()
            for i in range(0, len(rec["obs"]), 65536)])
    dz = (G - G.mean()) / G.std() - (V - V.mean()) / V.std()
    dz_r = dz[rd["row"]]

    feats = reading_features(rec, rd)
    names = list(feats.keys())
    X = np.stack([feats[k] for k in names], 1).astype(np.float64)

    dec = np.flatnonzero(rd["feas_self"] & rd["best_placed"] & rd["decline"])
    q = np.quantile(dz_r[dec], [0.25, 0.75])
    hi = dec[dz_r[dec] >= q[1]]
    lo = dec[dz_r[dec] <= q[0]]
    eps = rd["episode"]
    print(f"declined best-placed: {len(dec)} (HI {len(hi)} / LO {len(lo)})", flush=True)

    # 1) standardized mean differences HI - LO with episode bootstrap
    mu, sd = X[dec].mean(0), X[dec].std(0) + 1e-9
    Z = (X - mu) / sd
    res_feats = {}
    uniq_h, uniq_l = np.unique(eps[hi]), np.unique(eps[lo])
    for j, name in enumerate(names):
        d0 = float(Z[hi, j].mean() - Z[lo, j].mean())
        boots = []
        for _ in range(BOOT):
            sh = np.concatenate([hi[eps[hi] == e] for e in rng.choice(uniq_h, len(uniq_h))])
            sl = np.concatenate([lo[eps[lo] == e] for e in rng.choice(uniq_l, len(uniq_l))])
            boots.append(Z[sh, j].mean() - Z[sl, j].mean())
        res_feats[name] = {"smd": d0, "se": float(np.std(boots))}
    order = sorted(names, key=lambda k: -abs(res_feats[k]["smd"]))
    for name in order:
        f = res_feats[name]
        star = "*" if abs(f["smd"]) > 2 * f["se"] else " "
        print(f"  {name:>20s}: {f['smd']:+.2f} +- {f['se']:.2f} {star}", flush=True)

    # 2) ridge Dz ~ features, episode-grouped held-out R2
    from team_decline_probe import cv_ridge
    r2, _ = cv_ridge(Z[dec], dz_r[dec], eps[dec])
    print(f"ridge Dz ~ features held-out R2 = {r2:.3f}", flush=True)

    out = {"checkpoint": int(ckpt.name), "rows": ROWS, "n_declined_bp": int(len(dec)),
           "hi_lo_smd": res_feats, "ridge_r2": r2,
           "dz_hi_median": float(np.median(dz_r[hi])), "dz_lo_median": float(np.median(dz_r[lo]))}
    RESULTS_DIR.mkdir(exist_ok=True)
    p = RESULTS_DIR / f"fear_decomp_{ckpt.name}.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"saved {p}  ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
