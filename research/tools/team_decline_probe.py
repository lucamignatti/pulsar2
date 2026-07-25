"""INTERP_SWEEP2 Stage B — team collective-decline interpretability (2v2).

What does the trunk BELIEVE during the collective declines that dominate team
play (88% of feasible airborne balls unclaimed at the 12.4B census)? Three
pre-registered probes (../reports/archive/INTERP_SWEEP2.md, frozen before this ran):

  B1  self-best-placed: can h2 decode "I am the best-placed teammate for this
      ball"? Obs-ridge is the real control (the obs carries the ingredients —
      teammate pos/vel — so the question is whether the trunk COMPUTES the
      comparison).
  B2  knowledge-coupling: among readings where self is feasible AND best-placed,
      is pursue-rate graded by held-out probe-readout quality at fixed true
      margin (the knowing-doing analysis transplanted to 2v2)?
  B3  bystander belief: probe h2 for "a teammate will pursue this reading within
      1.5s", then read the trained probe out on collective-decline rows.
      HIGH belief during mutual decline -> coordination belief error;
      LOW belief -> team whiff tax; not decodable -> representation gap.

Labels (team-canonical, per-player-perspective rows):
  reading   airborne ball (z > ARM_Z), ball-only landing sim, uncensored,
            feasible (required speed < 1300) for >= 1 team player.
  pursued   MOVEMENT_PHASE0 definition: WON the race, or within 500uu of the
            landing at touchdown, or net approach >= 60% of required speed.
  tm_pursues15  same criteria for the teammate evaluated at horizon
            min(t_land, 1.5s): touch, within 600uu of ball landing, or 60%
            approach over the horizon.
  decline   feasible for >=1, NO team player pursued (full-window defs).

All probes: episode-grouped 5-fold CV, shuffled-label control, obs control.
Decodability bar (registered): held-out AUC >= 0.65, or above the obs control
where the obs already contains the ingredients.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

import RocketSim as rs
from load_checkpoint import copy_checkpoint, load_models
from steer_team import (ARM_Z, DT, FEASIBLE_SPEED, LOST, NONE, RACE_MARGIN_S,
                        TEAMMATE, WON, SteeredPolicyRho, rollout_team,
                        simulate_landing)

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE.parent / "results"
SEED = 20260718
PPT = 2
N_ROWS = int(os.environ.get("DECLINE_ROWS", 400_000))
N_ARENAS = 24
MAX_READINGS = 16_000
ATTEND_RADIUS = 500.0
PURSUE_APPROACH_FRAC = 0.6
TM_HORIZON_S = 1.5
TM_ATTEND_RADIUS = 600.0
BOOT = 200


# ---------------------------------------------------------------------------
def decline_readings(rec, max_readings=MAX_READINGS):
    """Per-player-perspective readings with self AND teammate race quantities.
    Landing sims are cached per arena-step block (all npl rows of a block share
    the ball state)."""
    ppt = rec["ppt"]
    npl = 2 * ppt
    phys, episode, slot, touched = rec["phys"], rec["episode"], rec["slot"], rec["touched"]

    ep_rows = {int(e): np.flatnonzero(episode == e) for e in np.unique(episode)}
    row_pos = np.empty(len(slot), np.int64)
    for rows in ep_rows.values():
        row_pos[rows] = np.arange(len(rows))

    cand = np.flatnonzero(phys[:, 2] > ARM_Z)
    stride = max(1, len(cand) // max_readings)
    cand = cand[::stride]

    arena = rs.Arena(rs.GameMode.SOCCAR)
    sim_cache = {}
    margin_rows = int(round(RACE_MARGIN_S / DT)) * npl

    keys = ("row", "episode", "outcome", "d_now", "t_land", "req_self", "req_tm",
            "feas_self", "feas_tm", "best_placed", "margin",
            "pursued_self", "pursued_tm", "tm_pursues15", "decline",
            "land_x", "land_y")
    out = {k: [] for k in keys}

    for r in cand:
        block = int(r) - int(slot[r])          # slots are consecutive within a block
        if block not in sim_cache:
            sim_cache[block] = simulate_landing(arena, phys[r, 0:3], phys[r, 3:6], phys[r, 6:9])
        res = sim_cache[block]
        if res is None:
            continue
        lx, ly, t_land = res
        rows = ep_rows[int(episode[r])]
        q = int(row_pos[r])
        steps = int(round(t_land / DT))
        q_land = q + npl * steps
        if q_land >= len(rows):
            continue                            # censored

        my_slot = int(slot[r])
        tm_slot = (my_slot + 2) % npl           # ppt=2: the other same-parity slot
        tm_r = int(r) + (tm_slot - my_slot)
        land = np.array([lx, ly])

        d_self = float(np.linalg.norm(phys[r, 9:11] - land))
        d_tm = float(np.linalg.norm(phys[tm_r, 9:11] - land))
        req_self = d_self / max(t_land, 1e-6)
        req_tm = d_tm / max(t_land, 1e-6)
        feas_self = req_self < FEASIBLE_SPEED
        feas_tm = req_tm < FEASIBLE_SPEED
        if not (feas_self or feas_tm):
            continue

        # outcome: first touch between reading and touchdown + margin
        outcome = NONE
        for qq in range(q + 1, min(len(rows), q_land + margin_rows + 1)):
            if touched[rows[qq]]:
                s = int(slot[rows[qq]])
                outcome = WON if s == my_slot else (TEAMMATE if s % 2 == my_slot % 2 else LOST)
                break

        # pursued, full window (self and teammate; q_land row keeps the slot)
        d_land_self = float(np.linalg.norm(phys[rows[q_land], 9:11] - land))
        appr_self = (d_self - d_land_self) / max(t_land, 1e-6)
        pursued_self = (outcome == WON) or (d_land_self < ATTEND_RADIUS) \
            or (appr_self >= PURSUE_APPROACH_FRAC * req_self)

        q_land_tm = q_land + (tm_slot - my_slot)
        d_land_tm = float(np.linalg.norm(phys[rows[q_land_tm], 9:11] - land))
        appr_tm = (d_tm - d_land_tm) / max(t_land, 1e-6)
        pursued_tm = (outcome == TEAMMATE) or (d_land_tm < ATTEND_RADIUS) \
            or (appr_tm >= PURSUE_APPROACH_FRAC * req_tm)

        # teammate pursuit at the registered 1.5s horizon
        h = min(t_land, TM_HORIZON_S)
        q_h_tm = q + npl * int(round(h / DT)) + (tm_slot - my_slot)
        tm_touch15 = any(touched[rows[qq]] for qq in range(q + 1, min(len(rows), q + npl * int(round(h / DT)) + 1))
                         if int(slot[rows[qq]]) == tm_slot)
        d_h_tm = float(np.linalg.norm(phys[rows[q_h_tm], 9:11] - land))
        appr_h_tm = (d_tm - d_h_tm) / max(h, 1e-6)
        tm_pursues15 = tm_touch15 or (d_h_tm < TM_ATTEND_RADIUS) \
            or (appr_h_tm >= PURSUE_APPROACH_FRAC * req_tm)

        feas_pool = [(req_self, True), (req_tm, False)]
        best = min((rq for rq, _ in feas_pool if rq < FEASIBLE_SPEED), default=np.inf)
        best_placed = feas_self and req_self <= best

        out["row"].append(int(r))
        out["episode"].append(int(episode[r]))
        out["outcome"].append(outcome)
        out["d_now"].append(d_self)
        out["t_land"].append(float(t_land))
        out["req_self"].append(req_self)
        out["req_tm"].append(req_tm)
        out["feas_self"].append(feas_self)
        out["feas_tm"].append(feas_tm)
        out["best_placed"].append(bool(best_placed))
        out["margin"].append(req_tm - req_self)
        out["pursued_self"].append(bool(pursued_self))
        out["pursued_tm"].append(bool(pursued_tm))
        out["tm_pursues15"].append(bool(tm_pursues15))
        out["decline"].append(bool(not pursued_self and not pursued_tm))
        out["land_x"].append(float(lx))
        out["land_y"].append(float(ly))
    return {k: np.array(v) for k, v in out.items()}


# ---------------------------------------------------------------------------
# probe helpers: episode-grouped CV, held-out predictions
# ---------------------------------------------------------------------------
def _group_folds(groups, k=5, seed=0):
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    for part in np.array_split(uniq, k):
        te = np.isin(groups, part)
        yield np.flatnonzero(~te), np.flatnonzero(te)


def cv_logistic(X, y, groups, seed=0):
    """Returns (mean held-out AUC, per-fold AUCs, pooled held-out P(y=1))."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    prob = np.full(len(y), np.nan)
    aucs = []
    for tr, te in _group_folds(groups, seed=seed):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        m = LogisticRegression(max_iter=3000, C=1.0)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        m.fit((X[tr] - mu) / sd, y[tr])
        p = m.predict_proba((X[te] - mu) / sd)[:, 1]
        prob[te] = p
        aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs)), [float(a) for a in aucs], prob


def cv_ridge(X, y, groups, seed=0):
    """Held-out R^2 with per-fold target standardization. Returns (R2, preds)."""
    from sklearn.linear_model import Ridge
    pred = np.full(len(y), np.nan)
    for tr, te in _group_folds(groups, seed=seed):
        m = Ridge(alpha=10.0)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        ymu, ysd = y[tr].mean(), y[tr].std() + 1e-6
        m.fit((X[tr] - mu) / sd, (y[tr] - ymu) / ysd)
        pred[te] = m.predict((X[te] - mu) / sd) * ysd + ymu
    ok = np.isfinite(pred)
    ss_res = np.sum((y[ok] - pred[ok]) ** 2)
    ss_tot = np.sum((y[ok] - y[ok].mean()) ** 2)
    return float(1 - ss_res / max(ss_tot, 1e-9)), pred


def cluster_boot_diff(vals_a, vals_b, eps_a, eps_b, n=BOOT, seed=0):
    """Bootstrap SE of mean(a) - mean(b), resampling episodes."""
    rng = np.random.default_rng(seed)
    ua, ub = np.unique(eps_a), np.unique(eps_b)
    diffs = []
    for _ in range(n):
        sa = rng.choice(ua, len(ua))
        sb = rng.choice(ub, len(ub))
        ma = np.concatenate([vals_a[eps_a == e] for e in sa]).mean()
        mb = np.concatenate([vals_b[eps_b == e] for e in sb]).mean()
        diffs.append(ma - mb)
    return float(np.std(diffs))


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    rs.init(str(HERE.parents[1] / "build" / "collision_meshes"))
    ckpt = copy_checkpoint(HERE.parent / "data" / "ckpt_cache")
    models = load_models(ckpt)
    print(f"pinned checkpoint {ckpt.name}", flush=True)

    rec = rollout_team(SteeredPolicyRho(models), PPT, N_ROWS, SEED,
                       num_arenas=N_ARENAS, want_h2=True, want_obs=True)
    print(f"rollout: {len(rec['episode'])} rows, {rec['episodes']} episodes, "
          f"{rec['goals']} goals  ({time.time()-t0:.0f}s)", flush=True)

    rd = decline_readings(rec)
    n = len(rd["row"])
    eps = rd["episode"]
    n_eps = len(np.unique(eps))
    decline_rate = float(rd["decline"].mean())
    print(f"readings: {n} (episodes {n_eps})  decline {decline_rate:.1%}  "
          f"pursued_self {rd['pursued_self'].mean():.1%}  "
          f"tm_pursues15 base {rd['tm_pursues15'].mean():.1%}", flush=True)

    H = rec["h2"][rd["row"]].astype(np.float32)
    O = rec["obs"][rd["row"]].astype(np.float32)
    rng = np.random.default_rng(SEED)

    res = {"checkpoint": int(ckpt.name), "rows": N_ROWS, "ppt": PPT,
           "n_readings": n, "n_episodes": n_eps, "decline_rate": decline_rate,
           "pursued_self_rate": float(rd["pursued_self"].mean()),
           "tm_pursues15_base": float(rd["tm_pursues15"].mean())}

    # ---- B1: self-best-placed ------------------------------------------------
    y = rd["best_placed"].astype(int)
    auc_h2, folds_h2, _ = cv_logistic(H, y, eps)
    auc_obs, _, _ = cv_logistic(O, y, eps)
    y_shuf = rng.permutation(y)
    auc_shuf, _, _ = cv_logistic(H, y_shuf, eps)
    # margin regression on the both-feasible subset only (an infeasible teammate
    # makes req_tm - req_self unbounded and the squared loss chases the tail)
    bf = np.flatnonzero(rd["feas_self"] & rd["feas_tm"])
    r2_h2, pm_bf = cv_ridge(H[bf], rd["margin"][bf], eps[bf])
    r2_obs, _ = cv_ridge(O[bf], rd["margin"][bf], eps[bf])
    pred_margin = np.full(n, np.nan)
    pred_margin[bf] = pm_bf
    res["B1"] = {"auc_h2": auc_h2, "auc_h2_folds": folds_h2, "auc_obs": auc_obs,
                 "auc_shuffled": auc_shuf, "margin_r2_h2": r2_h2, "margin_r2_obs": r2_obs}
    print(f"B1 best-placed: AUC h2 {auc_h2:.3f} obs {auc_obs:.3f} shuf {auc_shuf:.3f} | "
          f"margin R2 h2 {r2_h2:.3f} obs {r2_obs:.3f}", flush=True)

    # ---- B2: knowledge-coupling of decline ------------------------------------
    sub = np.flatnonzero(rd["feas_self"] & rd["best_placed"])
    pm, tm_, pu = pred_margin[sub], rd["margin"][sub], rd["pursued_self"][sub].astype(float)
    e_sub = eps[sub]
    ok = np.isfinite(pm)
    pm, tm_, pu, e_sub = pm[ok], tm_[ok], pu[ok], e_sub[ok]
    terciles = np.quantile(tm_, [1 / 3, 2 / 3])
    b2 = {"n": int(len(pm)), "terciles": []}
    hi_all, lo_all, hi_eps, lo_eps = [], [], [], []
    for lo_e, hi_e in ((-np.inf, terciles[0]), (terciles[0], terciles[1]), (terciles[1], np.inf)):
        m = (tm_ >= lo_e) & (tm_ < hi_e)
        if m.sum() < 20:
            continue
        med = np.median(pm[m])
        hi, lo = m & (pm >= med), m & (pm < med)
        b2["terciles"].append({"pursue_hi_read": float(pu[hi].mean()),
                               "pursue_lo_read": float(pu[lo].mean()),
                               "n_hi": int(hi.sum()), "n_lo": int(lo.sum())})
        hi_all.append(pu[hi]); lo_all.append(pu[lo])
        hi_eps.append(e_sub[hi]); lo_eps.append(e_sub[lo])
    if hi_all:
        hi_v, lo_v = np.concatenate(hi_all), np.concatenate(lo_all)
        he, le = np.concatenate(hi_eps), np.concatenate(lo_eps)
        d = float(hi_v.mean() - lo_v.mean())
        se = cluster_boot_diff(hi_v, lo_v, he, le)
        b2.update({"pooled_hi_minus_lo": d, "pooled_se": se,
                   "knowledge_coupled": bool(d > 2 * se)})
        print(f"B2 coupling: pursue(hi-read) - pursue(lo-read) = {d:+.3f} +- {se:.3f} "
              f"(n={len(pm)})", flush=True)
    res["B2"] = b2

    # ---- B3: bystander belief --------------------------------------------------
    y3 = rd["tm_pursues15"].astype(int)
    auc3_h2, folds3, prob3 = cv_logistic(H, y3, eps)
    auc3_obs, _, _ = cv_logistic(O, y3, eps)
    auc3_shuf, _, _ = cv_logistic(H, rng.permutation(y3), eps)
    dec = rd["decline"] & np.isfinite(prob3)
    dec_bp = dec & rd["feas_self"] & rd["best_placed"]
    base = float(y3.mean())
    b3 = {"auc_h2": auc3_h2, "auc_h2_folds": folds3, "auc_obs": auc3_obs,
          "auc_shuffled": auc3_shuf, "base_rate": base,
          "n_decline": int(dec.sum()), "n_decline_best_placed": int(dec_bp.sum()),
          "mean_prob_all": float(np.nanmean(prob3)),
          "mean_prob_decline": float(prob3[dec].mean()) if dec.any() else None,
          "mean_prob_decline_best_placed": float(prob3[dec_bp].mean()) if dec_bp.any() else None,
          "frac_decline_belief_above_base": float((prob3[dec] > base).mean()) if dec.any() else None}
    res["B3"] = b3
    print(f"B3 bystander: AUC h2 {auc3_h2:.3f} obs {auc3_obs:.3f} shuf {auc3_shuf:.3f} | "
          f"base {base:.2f} P(tm goes | decline) {b3['mean_prob_decline']} "
          f"(best-placed subset {b3['mean_prob_decline_best_placed']}, "
          f"n={b3['n_decline']}/{b3['n_decline_best_placed']})", flush=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"team_decline_probe_{ckpt.name}.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"saved {out}  ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
