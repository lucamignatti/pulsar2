"""CRITIC_DUEL — discriminate 'shaped critic is scared' vs 'goal critic is
noisy/lagging' on the frontier population. Design + frozen decision rule:
CRITIC_DUEL.md."""

import json
import time
from pathlib import Path

import numpy as np
import torch

import RocketSim as rs
from frontier_validate import reconstruct, roll_and_judge
from load_checkpoint import copy_checkpoint, load_models, rebuild_sequential
from steer_team import NONE, SteeredPolicyRho, TeamArenaEnv, rollout_team
from team_decline_probe import decline_readings, _group_folds

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
SEED = 20260727
PPT = 2
NPL = 2 * PPT
ROWS = 900_000
N_ARENAS = 24
K_TAIL = 100
REPEATS = 2
ROLL_S = 5.0
NOISE = 100.0          # small: preserve state identity (frontier_validate level)
BOOT = 300


def cv_auc_1d(x, y, groups):
    """Held-out AUC of a single scalar feature (sign-corrected on train folds)."""
    from sklearn.metrics import roc_auc_score
    aucs = []
    for tr, te in _group_folds(groups):
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
            continue
        sign = 1.0 if roc_auc_score(y[tr], x[tr]) >= 0.5 else -1.0
        aucs.append(roc_auc_score(y[te], sign * x[te]))
    return float(np.mean(aucs))


def joint_logistic_boot(zv, zg, y, eps, n_boot=BOOT, seed=0):
    """Episode-bootstrap distribution of the zG coefficient in y ~ zV + zG."""
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(seed)
    uniq = np.unique(eps)
    X = np.stack([zv, zg], 1)
    coefs = []
    for _ in range(n_boot):
        sel = np.concatenate([np.flatnonzero(eps == e) for e in rng.choice(uniq, len(uniq))])
        if len(np.unique(y[sel])) < 2:
            continue
        m = LogisticRegression(max_iter=1000).fit(X[sel], y[sel])
        coefs.append(m.coef_[0])
    coefs = np.array(coefs)
    return {"coef_zV": float(coefs[:, 0].mean()), "coef_zV_se": float(coefs[:, 0].std()),
            "coef_zG": float(coefs[:, 1].mean()), "coef_zG_se": float(coefs[:, 1].std())}


def forced_contest(pool_idx, rd, bank, models, seed):
    rng = np.random.default_rng(seed)
    env = TeamArenaEnv(0, PPT, np.random.default_rng(seed))
    pol = SteeredPolicyRho(models)
    reader = opp = none_ct = bad = 0
    for i in pool_idx:
        r = int(rd["row"][i])
        entry = bank[r // NPL]
        reader_team = int(r % NPL) % 2
        for _ in range(REPEATS):
            reconstruct(env, entry, rng, NOISE, NOISE)
            team, finite = roll_and_judge(env, pol, ROLL_S)
            if not finite:
                bad += 1
            elif team is None:
                none_ct += 1
            elif team == reader_team:
                reader += 1
            else:
                opp += 1
    res = reader + opp
    return {"n": int(len(pool_idx)) * REPEATS, "resolved": res,
            "reader_share": reader / max(res, 1), "none_rate": none_ct / max(res + none_ct, 1),
            "bad": bad}


def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    rs.init(str(HERE.parents[1] / "build" / "collision_meshes"))
    ckpt = copy_checkpoint(HERE / "data" / "ckpt_cache")
    models = load_models(ckpt)
    goal_critic = rebuild_sequential(
        torch.jit.load(str(ckpt / "GOAL_CRITIC.lt"), map_location="cpu"))
    print(f"pinned checkpoint {ckpt.name}", flush=True)

    rec = rollout_team(SteeredPolicyRho(models), PPT, ROWS, SEED,
                       num_arenas=N_ARENAS, want_h2=True, want_obs=True, want_states=True)
    rd = decline_readings(rec)
    bank = rec["state_bank"]
    print(f"rollout {rec['episodes']} eps; {len(rd['row'])} readings "
          f"({time.time()-t0:.0f}s)", flush=True)

    with torch.no_grad():
        V = np.concatenate([models["CRITIC"](torch.from_numpy(
            rec["h2"][i:i + 65536].astype(np.float32))).flatten().numpy()
            for i in range(0, len(rec["h2"]), 65536)])
        G = np.concatenate([goal_critic(torch.from_numpy(
            rec["obs"][i:i + 65536].astype(np.float32))).flatten().numpy()
            for i in range(0, len(rec["obs"]), 65536)])
    zV = (V - V.mean()) / V.std()
    zG = (G - G.mean()) / G.std()
    dz = zG - zV
    rows = rd["row"]

    res = {"checkpoint": int(ckpt.name), "rows": ROWS}

    # ---- Test 1: prediction duel on resolved races ---------------------------
    from steer_team import WON, TEAMMATE
    contested = np.flatnonzero(rd["outcome"] != NONE)
    y = np.isin(rd["outcome"][contested], (WON, TEAMMATE)).astype(int)
    eps = rd["episode"][contested]
    xv, xg = zV[rows[contested]], zG[rows[contested]]
    auc_v = cv_auc_1d(xv, y, eps)
    auc_g = cv_auc_1d(xg, y, eps)
    joint = joint_logistic_boot(xv, xg, y, eps)
    res["duel_all_contested"] = {"n": int(len(contested)), "team_won_rate": float(y.mean()),
                                 "auc_zV": auc_v, "auc_zG": auc_g, **joint}
    print(f"duel (contested, n={len(contested)}): AUC zV {auc_v:.3f} zG {auc_g:.3f} | "
          f"joint coef zV {joint['coef_zV']:+.3f}+-{joint['coef_zV_se']:.3f} "
          f"zG {joint['coef_zG']:+.3f}+-{joint['coef_zG_se']:.3f}", flush=True)

    bp = np.flatnonzero((rd["outcome"] != NONE) & rd["feas_self"] & rd["best_placed"])
    if len(bp) > 200:
        yb = np.isin(rd["outcome"][bp], (WON, TEAMMATE)).astype(int)
        jb = joint_logistic_boot(zV[rows[bp]], zG[rows[bp]], yb, rd["episode"][bp])
        res["duel_best_placed"] = {"n": int(len(bp)), **jb}
        print(f"duel (best-placed subset, n={len(bp)}): coef zG "
              f"{jb['coef_zG']:+.3f}+-{jb['coef_zG_se']:.3f}", flush=True)

    # ---- Test 2: forced-contest calibration of Dz ----------------------------
    dec = np.flatnonzero(rd["feas_self"] & rd["best_placed"] & rd["decline"])
    order = dec[np.argsort(-dz[rows[dec]])]

    def dedupe(idx_list, k):
        seen, out = set(), []
        for i in idx_list:
            b = int(rd["row"][i]) // NPL
            if b in seen:
                continue
            seen.add(b)
            out.append(i)
            if len(out) >= k:
                break
        return np.array(out)

    hi = dedupe(order, K_TAIL)
    lo = dedupe(order[::-1], K_TAIL)
    res["dz_hi_median"] = float(np.median(dz[rows[hi]]))
    res["dz_lo_median"] = float(np.median(dz[rows[lo]]))
    for name, pool in (("HI", hi), ("LO", lo)):
        res[f"forced_{name}"] = forced_contest(pool, rd, bank, models, SEED + 9)
        m = res[f"forced_{name}"]
        print(f"forced {name} (dz med {res[f'dz_{name.lower()}_median']:+.2f}): "
              f"reader-share {m['reader_share']:.1%} (resolved {m['resolved']}, "
              f"none {m['none_rate']:.1%})", flush=True)

    h, l = res["forced_HI"], res["forced_LO"]
    diff = h["reader_share"] - l["reader_share"]
    se = float(np.sqrt(h["reader_share"] * (1 - h["reader_share"]) / max(h["resolved"], 1)
                       + l["reader_share"] * (1 - l["reader_share"]) / max(l["resolved"], 1)))
    res["forced_diff"] = diff
    res["forced_se"] = se

    keep_1 = joint["coef_zG"] > 0 and joint["coef_zG"] >= 2 * joint["coef_zG_se"]
    keep_2 = diff >= 0.10 and diff >= 1.5 * se
    res["keep_bar_1_zG_adds_info"] = bool(keep_1)
    res["keep_bar_2_dz_calibrated"] = bool(keep_2)
    res["KEEP_FLAG"] = bool(keep_1 or keep_2)
    print(f"forced HI-LO: {diff:+.1%} +- {se:.1%} | keep bars: zG-info {keep_1}, "
          f"dz-calibrated {keep_2} -> {'KEEP' if res['KEEP_FLAG'] else 'REVERT'}", flush=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"critic_duel_{ckpt.name}.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"saved {out}  ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
