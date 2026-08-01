"""Through-training sweep over the 5.3 reference set (log-spaced frozen snapshots,
ref_0 .. head). Per checkpoint:

  - 40k-frame self-play collection at trainer parity (collect_dataset_53 subprocess)
  - behavioral stats: touch rate, kickoff share touched, air-proximity rate, ball-z
    distribution, speed, boost
  - representation: episode-grouped-CV ridge R^2 for TEAM-CANONICAL ball-landing
    (x, y, t) from h2 - the Phase 0 landing-probe trajectory, now through training
  - geo field: v_geo mean/std on-policy, corr with ball_z (refs carry GEO_* nets)
  - kickoff health: median time-to-first-touch across kickoff episodes in the sample

Output: research/results/ref_sweep_53.json  (one entry per checkpoint)
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
RESULTS = HERE.parent / "results"
CKPT_ROOT = Path(os.environ.get("PULSAR_CKPT_ROOT", DATA / "ckpt53"))
PY = str(HERE.parent / ".venv" / "bin" / "python")

sys.path.insert(0, str(HERE))
import RocketSim as rs                     # noqa: E402
from label_landing import simulate_landing, AIRBORNE_Z   # noqa: E402


def ridge_cv_r2(X, y, groups, alphas=(1e0, 1e1, 1e2, 1e3, 1e4), folds=4):
    """Episode-grouped CV ridge R^2 per target column (per-fold target standardization)."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    r2 = np.zeros(y.shape[1])
    gkf = GroupKFold(n_splits=folds)
    pred = np.full_like(y, np.nan, dtype=np.float64)
    for tr, te in gkf.split(X, y, groups):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        ymu, ysd = y[tr].mean(0), y[tr].std(0) + 1e-8
        best = None
        # small inner split for alpha
        n_in = len(tr)
        cut = int(n_in * 0.8)
        for a in alphas:
            m = Ridge(alpha=a).fit(Xtr[:cut], (y[tr][:cut] - ymu) / ysd)
            v = ((m.predict(Xtr[cut:]) - (y[tr][cut:] - ymu) / ysd) ** 2).mean()
            if best is None or v < best[0]:
                best = (v, a)
        m = Ridge(alpha=best[1]).fit(Xtr, (y[tr] - ymu) / ysd)
        pred[te] = m.predict(Xte) * ysd + ymu
    for j in range(y.shape[1]):
        ss_res = np.nansum((pred[:, j] - y[:, j]) ** 2)
        ss_tot = np.nansum((y[:, j] - y[:, j].mean()) ** 2)
        r2[j] = 1 - ss_res / ss_tot
    return r2


def analyze(tag, ckpt_name):
    d = np.load(DATA / f"dataset53_{tag}.npz")
    n = len(d["action"])
    team = d["team"].astype(int)
    phys = d["phys"]
    ball = phys[:, 0:3]
    self_pos = np.where(team[:, None] == 0, phys[:, 9:12], phys[:, 20:23])
    self_vel = np.where(team[:, None] == 0, phys[:, 12:15], phys[:, 23:26])
    self_boost = np.where(team == 0, phys[:, 18], phys[:, 29])
    self_ground = np.where(team == 0, phys[:, 19], phys[:, 30]).astype(bool)
    dist_ball = np.linalg.norm(self_pos - ball, axis=1)
    speed = np.linalg.norm(self_vel, axis=1)

    rec = {"checkpoint": ckpt_name, "n_frames": int(n),
           "n_episodes": int(d["episode"].max() + 1),
           "touch_rate_per_100_steps": float(d["touched"].mean() * 100),
           "ball_z_gt300_frac": float((ball[:, 2] > 300).mean()),
           "air_prox_rate": float(((~self_ground) & (self_pos[:, 2] > 300)
                                   & (dist_ball < 300)).mean()),
           "airborne_frac": float((~self_ground).mean()),
           "mean_speed": float(speed.mean()),
           "mean_boost": float(self_boost.mean()),
           "mean_dist_ball": float(dist_ball.mean())}

    vg = d["ladder_v_geo"]
    if np.isfinite(vg).any():
        rec["v_geo_mean"] = float(np.nanmean(vg))
        rec["v_geo_std"] = float(np.nanstd(vg))
        rec["v_geo_corr_ball_z"] = float(np.corrcoef(vg, ball[:, 2])[0, 1])
        rec["r_hat_std"] = float(np.nanstd(d["ladder_r_hat"]))

    # landing probe: label airborne frames, ridge from h2 (team-canonical targets)
    airborne = ball[:, 2] > AIRBORNE_Z
    idx = np.flatnonzero(airborne)
    if len(idx) > 4000:
        idx = idx[:: max(1, len(idx) // 20000)]
        arena = rs.Arena(rs.GameMode.SOCCAR)
        land = np.full((len(idx), 3), np.nan, np.float32)
        cache = {}
        for i, r in enumerate(idx):
            key = (int(d["episode"][r]), round(float(phys[r, 0]), 1), round(float(phys[r, 2]), 1))
            if key not in cache:
                cache[key] = simulate_landing(arena, phys[r, 0:3], phys[r, 3:6], phys[r, 6:9])
            res = cache[key]
            if res is not None:
                land[i] = res
        ok = np.isfinite(land[:, 0])
        idx, land = idx[ok], land[ok]
        sign = np.where(team[idx] == 0, 1.0, -1.0)
        y = np.stack([land[:, 0] * sign, land[:, 1] * sign, land[:, 2]], 1)
        X = d["h2"][idx].astype(np.float32)
        r2 = ridge_cv_r2(X, y, d["episode"][idx])
        rec["landing_r2_h2"] = {"x": float(r2[0]), "y": float(r2[1]), "t": float(r2[2])}
        rec["landing_n"] = int(len(idx))

    # kickoff first-touch proxy: among kickoff episodes, did a touch occur, and when
    kick = d["is_kickoff"]
    eps = np.unique(d["episode"][kick])
    t_first = []
    touched_any = 0
    for e in eps:
        rows = np.flatnonzero((d["episode"] == e))
        tch = np.flatnonzero(d["touched"][rows])
        if len(tch):
            touched_any += 1
            t_first.append(tch[0] / 2 * (8 / 120.0))   # 2 rows per arena-step
    if len(eps):
        rec["kickoff_eps"] = int(len(eps))
        rec["kickoff_touched_frac"] = float(touched_any / len(eps))
        rec["kickoff_median_first_touch_s"] = float(np.median(t_first)) if t_first else None
    return rec


def main():
    rs.init(str(HERE.parents[1] / "build" / "collision_meshes"))
    cands = ["ref_0", "ref_1050497024", "ref_2750108463", "ref_4425008897",
             "ref_5475285704", "ref_7125489083", "ref_8825031303", "9750127919"]
    out = []
    env = dict(os.environ, PULSAR_CKPT_ROOT=str(CKPT_ROOT), PROBE_FRAMES="40000",
               OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4")
    for name in cands:
        tag = name.replace("ref_", "r")
        if not (DATA / f"dataset53_{tag}.npz").exists():
            print(f"collecting {name} ...", flush=True)
            env["DATASET_TAG"] = tag
            r = subprocess.run([PY, str(HERE / "collect_dataset_53.py"),
                                "--ckpt", str(CKPT_ROOT / name)],
                               env=env, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  FAILED: {r.stderr[-500:]}")
                continue
        print(f"analyzing {name} ...", flush=True)
        rec = analyze(tag, name)
        out.append(rec)
        print(json.dumps({k: v for k, v in rec.items() if not isinstance(v, dict)},
                         default=str), flush=True)
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / "ref_sweep_53.json").write_text(json.dumps(out, indent=1))
    print("done")


if __name__ == "__main__":
    main()
