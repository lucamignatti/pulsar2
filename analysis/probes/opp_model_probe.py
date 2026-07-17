"""INTERP_SWEEP2 Stage C — opponent-model probes (1v1, Stage-A dataset).

  C1  opponent future state: ridge h2 -> opponent canonical position at +0.5s
      and +1.0s. The obs-ridge control can learn pos + vel*t extrapolation (the
      obs carries opp pos/vel), so DELTA R2 = h2 - obs is the trunk's own
      contribution. A pure kinematic extrapolation baseline is reported too.
  C2  race prediction: "opponent reaches the ball first within 2s". The dataset
      records no touch events, so the registered 'touches first' target is
      operationalized POSITIONALLY: first player within 250uu of ball center
      (ball r=92.75 + car body). Resolved races only. AUC h2 vs obs.
  A3  premeditation (continuity probe): "self reaches the ball within 1s"
      (same positional proxy), restricted to rows currently > 500uu from the
      ball. AUC h2 vs obs.

Episode-grouped 5-fold CV, shuffled controls, per-fold target standardization.
phys layout (31): ball pos3 vel3 angvel3 | p0 pos3 vel3 (9:15) ... | p1 pos3
vel3 (20:26); rows interleave p0,p1 per arena-step; team field = player index.
Targets are TEAM-CANONICAL (x,y negated for the orange-row perspective).
"""

import json
import time
from pathlib import Path

import numpy as np

from team_decline_probe import cv_logistic, cv_ridge, _group_folds  # noqa: F401

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "dataset.npz"
RESULTS_DIR = HERE / "results"
SEED = 20260719
TOUCH_PROXY_UU = 250.0
MAX_ROWS = 60_000       # subsample cap for the sklearn fits


def cv_ridge_multi(X, Y, groups, seed=0):
    """Multi-output ridge, per-fold per-target standardization; held-out R2 per col."""
    from sklearn.linear_model import Ridge
    pred = np.full(Y.shape, np.nan)
    for tr, te in _group_folds(groups, seed=seed):
        m = Ridge(alpha=10.0)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        ymu, ysd = Y[tr].mean(0), Y[tr].std(0) + 1e-6
        m.fit((X[tr] - mu) / sd, (Y[tr] - ymu) / ysd)
        pred[te] = m.predict((X[te] - mu) / sd) * ysd + ymu
    ok = np.isfinite(pred).all(1)
    r2 = []
    for j in range(Y.shape[1]):
        ss_res = np.sum((Y[ok, j] - pred[ok, j]) ** 2)
        ss_tot = np.sum((Y[ok, j] - Y[ok, j].mean()) ** 2)
        r2.append(float(1 - ss_res / max(ss_tot, 1e-9)))
    return r2


def main():
    t0 = time.time()
    d = np.load(DATA)
    # decision-step seconds of the rollout that wrote dataset.npz (tick_skip saved
    # by collect_dataset since the 5.0 migration; older archives = tickSkip 4)
    dt = (float(d["tick_skip"]) if "tick_skip" in d.files else 4.0) / 120.0
    ckpt = int(d["checkpoint"])
    phys, team, episode = d["phys"], d["team"].astype(int), d["episode"]
    h2 = d["h2"].astype(np.float32)
    obs = d["obs"].astype(np.float32)
    n = len(team)
    print(f"dataset: {n} rows, checkpoint {ckpt}", flush=True)

    ep_rows = {int(e): np.flatnonzero(episode == e) for e in np.unique(episode)}
    row_pos = np.empty(n, np.int64)
    for rows in ep_rows.values():
        row_pos[rows] = np.arange(len(rows))
    ep_len = np.array([len(ep_rows[int(e)]) for e in episode])

    self_sl = np.where(team[:, None] == 0, np.arange(9, 12), np.arange(20, 23))
    opp_sl = np.where(team[:, None] == 0, np.arange(20, 23), np.arange(9, 12))
    # canonicalization: [+1,+1,+1] for blue-perspective rows, [-1,-1,+1] for orange
    flip = np.where(team == 0, 1.0, -1.0)
    canon = np.stack([flip, flip, np.ones(n)], axis=1)

    def future_rows(horiz_s):
        k = int(round(horiz_s / dt))
        fq = row_pos + 2 * k
        valid = fq < ep_len
        fr = np.full(n, -1, np.int64)
        for e, rows in ep_rows.items():
            m = (episode == e) & valid
            fr[m] = rows[fq[m]]
        return fr, valid

    rng = np.random.default_rng(SEED)
    keep = rng.permutation(n)[:MAX_ROWS]
    res = {"checkpoint": ckpt, "n_rows_used": int(len(keep))}

    # ---- C1: opponent future position ---------------------------------------
    for horiz in (0.5, 1.0):
        fr, valid = future_rows(horiz)
        sel = keep[valid[keep]]
        opp_fut = phys[fr[sel]][np.arange(len(sel))[:, None], opp_sl[sel]] * canon[sel]
        Xh, Xo, g = h2[sel], obs[sel], episode[sel]
        r2_h2 = cv_ridge_multi(Xh, opp_fut, g)
        r2_obs = cv_ridge_multi(Xo, opp_fut, g)
        # kinematic extrapolation baseline: opp pos + vel * t (canonical)
        opp_now = phys[sel][np.arange(len(sel))[:, None], opp_sl[sel]] * canon[sel]
        opp_vel = phys[sel][np.arange(len(sel))[:, None], opp_sl[sel] + 3] * canon[sel]
        extrap = opp_now + opp_vel * horiz
        r2_ext = [float(1 - np.sum((opp_fut[:, j] - extrap[:, j]) ** 2)
                        / max(np.sum((opp_fut[:, j] - opp_fut[:, j].mean()) ** 2), 1e-9))
                  for j in range(3)]
        res[f"C1_{horiz}s"] = {"r2_h2_xyz": r2_h2, "r2_obs_xyz": r2_obs,
                               "r2_extrap_xyz": r2_ext, "n": int(len(sel))}
        print(f"C1 +{horiz}s: h2 {['%.3f' % v for v in r2_h2]} "
              f"obs {['%.3f' % v for v in r2_obs]} extrap {['%.3f' % v for v in r2_ext]} "
              f"(n={len(sel)})", flush=True)

    # ---- touch-proxy scan for C2 / A3 ----------------------------------------
    k2 = int(round(2.0 / dt))
    k1 = int(round(1.0 / dt))
    self_first = np.full(n, -1, np.int8)   # -1 unresolved, 0 opp first, 1 self first
    self_reach1 = np.zeros(n, bool)
    d_ball_now = np.linalg.norm(
        phys[np.arange(n)[:, None], self_sl] - phys[:, 0:3], axis=1)
    for e, rows in ep_rows.items():
        ball = phys[rows[::2], 0:3]                       # per-step ball pos
        p0 = phys[rows[::2], 9:12]
        p1 = phys[rows[::2], 20:23]
        d0 = np.linalg.norm(p0 - ball, axis=1) < TOUCH_PROXY_UU
        d1 = np.linalg.norm(p1 - ball, axis=1) < TOUCH_PROXY_UU
        n_steps = len(ball)
        for r in rows:
            q = row_pos[r] // 2
            me = team[r]
            hit0 = np.flatnonzero(d0[q + 1: min(n_steps, q + 1 + k2)])
            hit1 = np.flatnonzero(d1[q + 1: min(n_steps, q + 1 + k2)])
            t_self = (hit0[0] if me == 0 else hit1[0]) if (hit0.size if me == 0 else hit1.size) else 10**9
            t_opp = (hit1[0] if me == 0 else hit0[0]) if (hit1.size if me == 0 else hit0.size) else 10**9
            if t_self < 10**9 or t_opp < 10**9:
                self_first[r] = 1 if t_self <= t_opp else 0
            self_reach1[r] = t_self < k1

    # C2: resolved races only
    resolved = keep[self_first[keep] >= 0]
    y2 = (self_first[resolved] == 0).astype(int)          # 1 = OPPONENT first
    auc2_h2, _, _ = cv_logistic(h2[resolved], y2, episode[resolved])
    auc2_obs, _, _ = cv_logistic(obs[resolved], y2, episode[resolved])
    auc2_shuf, _, _ = cv_logistic(h2[resolved], rng.permutation(y2), episode[resolved])
    res["C2"] = {"auc_h2": auc2_h2, "auc_obs": auc2_obs, "auc_shuffled": auc2_shuf,
                 "n": int(len(resolved)), "opp_first_rate": float(y2.mean())}
    print(f"C2 opp-first: AUC h2 {auc2_h2:.3f} obs {auc2_obs:.3f} shuf {auc2_shuf:.3f} "
          f"(n={len(resolved)}, rate {y2.mean():.2f})", flush=True)

    # A3: premeditation — self reaches ball within 1s, from > 500uu away
    far = keep[d_ball_now[keep] > 500]
    y3 = self_reach1[far].astype(int)
    auc3_h2, _, _ = cv_logistic(h2[far], y3, episode[far])
    auc3_obs, _, _ = cv_logistic(obs[far], y3, episode[far])
    auc3_shuf, _, _ = cv_logistic(h2[far], rng.permutation(y3), episode[far])
    res["A3"] = {"auc_h2": auc3_h2, "auc_obs": auc3_obs, "auc_shuffled": auc3_shuf,
                 "n": int(len(far)), "reach_rate": float(y3.mean())}
    print(f"A3 premeditation: AUC h2 {auc3_h2:.3f} obs {auc3_obs:.3f} shuf {auc3_shuf:.3f} "
          f"(n={len(far)}, rate {y3.mean():.2f})", flush=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"opp_model_probe_{ckpt}.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"saved {out}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
