"""GOAL_CRITIC_AUDIT — is the goal-critic channel driving learning, coasting, or
hindering it? Design + frozen reading rule: GOAL_CRITIC_AUDIT.md.

Offline CPU. Reads a PINNED checkpoint dir (env GC_CKPT) — never the live
rotation. Reproduces the trainer's own goal-channel GAE (Learner.cpp:4679) so
the audited quantity is the one actually injected into the policy gradient.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

import RocketSim as rs
from load_checkpoint import load_models, rebuild_sequential
from steer_team import NONE, TEAMMATE, WON, SteeredPolicyRho, rollout_team
from team_decline_probe import _group_folds, decline_readings

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE.parent / "results"
SEED = 20260720

GAMMA_DENSE = 0.9969        # ExampleMain TRAIN_GAMMA (15 Hz, ~15 s half-life)
GAMMA_GOAL = 0.9994         # goalCritic.gamma (~77 s half-life)
LAM = 0.95                  # gaeLambda
BETA = 0.25                 # goalCritic.beta
ROWS_2V2 = int(os.environ.get("GC_ROWS_2V2", 400_000))
ROWS_1V1 = int(os.environ.get("GC_ROWS_1V1", 200_000))
N_ARENAS = 24
BOOT = 300
CHUNK = 65536
MAX_READINGS = int(os.environ.get("GC_MAX_READINGS", 16_000))


# --------------------------------------------------------------------------- #
def head(model, x):
    with torch.no_grad():
        return np.concatenate([
            model(torch.from_numpy(x[i:i + CHUNK].astype(np.float32))).flatten().numpy()
            for i in range(0, len(x), CHUNK)])


def player_seqs(rec):
    """Row indices per (episode, player), in time order. Rows are written
    step-major/player-minor within an episode, so slot p is rows[p::npl]."""
    npl = 2 * rec["ppt"]
    episode = rec["episode"]
    out = []
    for e in np.unique(episode):
        rows = np.flatnonzero(episode == e)
        rows = rows[: (len(rows) // npl) * npl]
        if len(rows) < npl * 4:
            continue
        for p in range(npl):
            out.append((int(e), p, rows[p::npl]))
    return out


def gae(vals, rews, terminal, gamma, lam):
    """Standard GAE over one contiguous sequence. terminal=True -> the last row
    is a true terminal (no bootstrap); False -> the sequence is truncated and the
    final row is used ONLY as the bootstrap value (it gets no advantage)."""
    n = len(vals)
    if terminal:
        nxt = np.concatenate([vals[1:], [0.0]])
        cont = np.ones(n)
        cont[-1] = 0.0
        r = rews
    else:
        n -= 1
        if n <= 0:
            return np.zeros(0)
        nxt, cont, r, vals = vals[1:], np.ones(n), rews[:n], vals[:n]
    delta = r + gamma * nxt * cont - vals
    adv = np.empty(n)
    run = 0.0
    for t in range(n - 1, -1, -1):
        run = delta[t] + gamma * lam * cont[t] * run
        adv[t] = run
    return adv


def build_advantages(rec, V, G):
    """Per-row A_goal (trainer-exact), D_G (goal drift, terminal reward removed)
    and D_V (shaped critic's local drift). Rows not covered stay NaN."""
    n = len(V)
    A_goal = np.full(n, np.nan)
    D_G = np.full(n, np.nan)
    D_V = np.full(n, np.nan)
    outcome = np.full(n, np.nan)     # realized +-1 goal sign for the row's player
    ep_of = np.full(n, -1, np.int64)
    to_end = np.full(n, -1, np.int64)   # decision steps until this player's episode ends
    goal_team = rec["ep_goal_team"]
    for e, p, rows in player_seqs(rec):
        team = p % 2
        scored = goal_team.get(e)
        is_goal = scored is not None
        rew = np.zeros(len(rows))
        if is_goal:
            sign = 1.0 if scored == team else -1.0
            rew[-1] = sign
            outcome[rows] = sign
        g, v = G[rows], V[rows]
        a = gae(g, rew, is_goal, GAMMA_GOAL, LAM)
        d = gae(g, np.zeros(len(rows)), is_goal, GAMMA_GOAL, LAM)
        w = gae(v, np.zeros(len(rows)), is_goal, GAMMA_DENSE, LAM)
        A_goal[rows[:len(a)]] = a
        D_G[rows[:len(d)]] = d
        D_V[rows[:len(w)]] = w
        ep_of[rows] = e
        to_end[rows] = np.arange(len(rows))[::-1]
    return A_goal, D_G, D_V, outcome, ep_of, to_end


def cv_auc(x, y, groups):
    from sklearn.metrics import roc_auc_score
    aucs = []
    for tr, te in _group_folds(groups):
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
            continue
        sign = 1.0 if roc_auc_score(y[tr], x[tr]) >= 0.5 else -1.0
        aucs.append(roc_auc_score(y[te], sign * x[te]))
    return float(np.mean(aucs)) if aucs else float("nan")


def auc_boot(x, y, eps, n_boot=BOOT, seed=0):
    """Episode-bootstrap SE of the plain (in-sample-direction-free) AUC."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    uniq = np.unique(eps)
    vals = []
    for _ in range(n_boot):
        sel = np.concatenate([np.flatnonzero(eps == e) for e in rng.choice(uniq, len(uniq))])
        if len(np.unique(y[sel])) < 2:
            continue
        vals.append(roc_auc_score(y[sel], x[sel]))
    return float(np.mean(vals)), float(np.std(vals))


def joint_logistic_boot(x1, x2, y, eps, n_boot=BOOT, seed=0):
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(seed)
    uniq = np.unique(eps)
    X = np.stack([(x1 - x1.mean()) / (x1.std() + 1e-9),
                  (x2 - x2.mean()) / (x2.std() + 1e-9)], 1)
    coefs = []
    for _ in range(n_boot):
        sel = np.concatenate([np.flatnonzero(eps == e) for e in rng.choice(uniq, len(uniq))])
        if len(np.unique(y[sel])) < 2:
            continue
        coefs.append(LogisticRegression(max_iter=1000).fit(X[sel], y[sel]).coef_[0])
    c = np.array(coefs)
    return {"coef_a": float(c[:, 0].mean()), "coef_a_se": float(c[:, 0].std()),
            "coef_b": float(c[:, 1].mean()), "coef_b_se": float(c[:, 1].std())}


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return float("nan")
    a, b = a[m], b[m]
    return float(np.corrcoef(a, b)[0, 1])


def interference(a_goal, seed=0):
    """T5: with betaEff std-matched, how often does the injected term overturn a
    std-matched Gaussian A_dense? Uses the MEASURED shape of A_goal."""
    rng = np.random.default_rng(seed)
    z = a_goal[np.isfinite(a_goal)]
    z = (z - z.mean()) / (z.std() + 1e-12)          # centered + unit -> inj = BETA*z*std(A_dense)
    dense = rng.standard_normal(len(z))             # std-matched Gaussian A_dense
    inj = BETA * z
    tot = dense + inj
    absz = np.abs(z)
    order = np.sort(absz)[::-1]
    top1 = order[: max(1, len(order) // 100)].sum() / absz.sum()
    return {"n": int(len(z)),
            "kurtosis": float(((z ** 4).mean()) - 3.0),
            "top1pct_abs_mass_share": float(top1),
            "sign_flip_rate": float((np.sign(tot) != np.sign(dense)).mean()),
            "inj_exceeds_dense_rate": float((np.abs(inj) > np.abs(dense)).mean()),
            "rank_corr_dense_vs_total": float(np.corrcoef(dense, tot)[0, 1])}


# --------------------------------------------------------------------------- #
def analyse(rec, models, goal_critic, ppt, res):
    tag = f"{ppt}v{ppt}"
    V = head(models["CRITIC"], rec["h2"])
    G = head(goal_critic, rec["obs"])
    A_goal, D_G, D_V, outcome, ep_of, to_end = build_advantages(rec, V, G)

    ep_len = [len(r) for _, _, r in player_seqs(rec)]
    goal_eps = set(rec["ep_goal_team"])
    all_eps = set(int(e) for e in np.unique(rec["episode"]))
    res[f"{tag}/regime"] = {
        "rows": int(len(V)), "episodes": int(len(all_eps)),
        "goal_ended_frac": float(len(goal_eps & all_eps) / max(len(all_eps), 1)),
        "median_player_seq_steps": float(np.median(ep_len)),
        "eff_window_steps_dense": 1.0 / (1 - GAMMA_DENSE * LAM),
        "eff_window_steps_goal": 1.0 / (1 - GAMMA_GOAL * LAM),
    }
    print(f"[{tag}] {len(V)} rows, {len(all_eps)} eps, "
          f"goal-ended {res[f'{tag}/regime']['goal_ended_frac']:.1%}, "
          f"median seq {np.median(ep_len):.0f} steps", flush=True)

    # ---- T1 level authority + T2 redundancy ---------------------------------
    m = np.isfinite(outcome)
    y = (outcome[m] > 0).astype(int)
    eps = ep_of[m]
    zV = (V[m] - V[m].mean()) / (V[m].std() + 1e-9)
    zG = (G[m] - G[m].mean()) / (G[m].std() + 1e-9)
    res[f"{tag}/T1_level"] = {
        "n": int(m.sum()), "win_rate": float(y.mean()),
        "corr_G_outcome": corr(G[m], outcome[m]),
        "corr_V_outcome": corr(V[m], outcome[m]),
        "auc_G": cv_auc(zG, y, eps), "auc_V": cv_auc(zV, y, eps),
    }
    res[f"{tag}/T2_redundancy"] = {
        "corr_zV_zG": corr(zV, zG),
        **joint_logistic_boot(zG, zV, y, eps, seed=SEED),
    }
    t1, t2 = res[f"{tag}/T1_level"], res[f"{tag}/T2_redundancy"]
    print(f"[{tag}] T1 level: AUC G {t1['auc_G']:.3f} V {t1['auc_V']:.3f} | "
          f"corr(G,out) {t1['corr_G_outcome']:.3f} corr(V,out) {t1['corr_V_outcome']:.3f}",
          flush=True)
    print(f"[{tag}] T2 redundancy: corr(zV,zG) {t2['corr_zV_zG']:+.3f} | joint coef "
          f"zG {t2['coef_a']:+.3f}+-{t2['coef_a_se']:.3f} zV {t2['coef_b']:+.3f}+-{t2['coef_b_se']:.3f}",
          flush=True)

    # ---- T4 agreement of the two local derivatives --------------------------
    # within- vs between-episode variance of A_goal: is it per-action credit or a
    # per-episode constant smeared over every row?
    fin = np.isfinite(A_goal)
    ep_means = {}
    for e in np.unique(ep_of[fin]):
        ep_means[int(e)] = A_goal[fin & (ep_of == e)].mean()
    centres = np.array([ep_means[int(e)] for e in ep_of[fin]])
    between = float(np.var(centres))
    total = float(np.var(A_goal[fin]))
    res[f"{tag}/T4_agreement"] = {
        "corr_Agoal_DV": corr(A_goal, D_V),
        "corr_DG_DV": corr(D_G, D_V),
        "corr_Agoal_DG": corr(A_goal, D_G),
        "std_Agoal": float(np.nanstd(A_goal)), "std_DV": float(np.nanstd(D_V)),
        "Agoal_between_episode_var_share": between / max(total, 1e-12),
    }
    t4 = res[f"{tag}/T4_agreement"]
    print(f"[{tag}] T4 agreement: corr(A_goal,D_V) {t4['corr_Agoal_DV']:+.3f} | "
          f"corr(D_G,D_V) {t4['corr_DG_DV']:+.3f}", flush=True)

    # ---- T5 interference ----------------------------------------------------
    res[f"{tag}/T5_interference"] = interference(A_goal, seed=SEED)
    t5 = res[f"{tag}/T5_interference"]
    print(f"[{tag}] T5 interference: sign-flip {t5['sign_flip_rate']:.1%}, "
          f"|inj|>|dense| {t5['inj_exceeds_dense_rate']:.1%}, kurt {t5['kurtosis']:.1f}, "
          f"top1% mass {t5['top1pct_abs_mass_share']:.1%}", flush=True)

    # ---- T3 derivative authority on resolved races (2v2 only) ---------------
    if ppt == 2:
        rd = decline_readings(rec, max_readings=MAX_READINGS)
        zVa = (V - np.nanmean(V)) / (np.nanstd(V) + 1e-9)
        zGa = (G - np.nanmean(G)) / (np.nanstd(G) + 1e-9)
        far = to_end[rd["row"]] > 40      # >2x the GAE window from the terminal:
        pops = {                          # A_goal here is PURE drift, no terminal reward
            "all_contested": rd["outcome"] != NONE,
            "frontier": (rd["outcome"] != NONE) & rd["feas_self"] & rd["best_placed"],
            "far_from_terminal": (rd["outcome"] != NONE) & far,
        }
        for pname, pmask in pops.items():
            sel = np.flatnonzero(pmask)
            if len(sel) < 200:
                continue
            rows = rd["row"][sel]
            ok = np.isfinite(A_goal[rows]) & np.isfinite(D_V[rows]) & np.isfinite(D_G[rows])
            sel, rows = sel[ok], rows[ok]
            yb = np.isin(rd["outcome"][sel], (WON, TEAMMATE)).astype(int)
            eb = rd["episode"][sel]
            ag, dv, dg = A_goal[rows], D_V[rows], D_G[rows]
            auc_a, se_a = auc_boot(ag, yb, eb, seed=SEED)
            auc_d, se_d = auc_boot(dv, yb, eb, seed=SEED + 1)
            auc_dg, se_dg = auc_boot(dg, yb, eb, seed=SEED + 5)
            # CRITIC_DUEL test-1 repeat: the LEVELS, same population, same stat
            auc_lg, se_lg = auc_boot(zGa[rows], yb, eb, seed=SEED + 3)
            auc_lv, se_lv = auc_boot(zVa[rows], yb, eb, seed=SEED + 4)
            res[f"2v2/T3_{pname}"] = {
                "n": int(len(sel)), "win_rate": float(yb.mean()),
                "cv_auc_Agoal": cv_auc(ag, yb, eb), "cv_auc_DV": cv_auc(dv, yb, eb),
                "boot_auc_Agoal": auc_a, "boot_auc_Agoal_se": se_a,
                "boot_auc_DV": auc_d, "boot_auc_DV_se": se_d,
                "boot_auc_DG": auc_dg, "boot_auc_DG_se": se_dg,
                "cv_auc_level_zG": cv_auc(zGa[rows], yb, eb),
                "cv_auc_level_zV": cv_auc(zVa[rows], yb, eb),
                "boot_auc_level_zG": auc_lg, "boot_auc_level_zG_se": se_lg,
                "boot_auc_level_zV": auc_lv, "boot_auc_level_zV_se": se_lv,
                **joint_logistic_boot(ag, dv, yb, eb, seed=SEED + 2),
            }
            t3 = res[f"2v2/T3_{pname}"]
            print(f"[2v2] T3 {pname} (n={t3['n']}): AUC A_goal {t3['boot_auc_Agoal']:.3f}"
                  f"+-{t3['boot_auc_Agoal_se']:.3f}  D_G {t3['boot_auc_DG']:.3f}"
                  f"+-{t3['boot_auc_DG_se']:.3f}  D_V {t3['boot_auc_DV']:.3f}"
                  f"+-{t3['boot_auc_DV_se']:.3f} | LEVELS zG {t3['boot_auc_level_zG']:.3f}"
                  f"+-{t3['boot_auc_level_zG_se']:.3f} zV {t3['boot_auc_level_zV']:.3f}"
                  f"+-{t3['boot_auc_level_zV_se']:.3f} | joint coef A_goal "
                  f"{t3['coef_a']:+.3f}+-{t3['coef_a_se']:.3f} D_V {t3['coef_b']:+.3f}"
                  f"+-{t3['coef_b_se']:.3f}", flush=True)


def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    rs.init(str(HERE.parents[1] / "build" / "collision_meshes"))
    ckpt = Path(os.environ["GC_CKPT"])
    models = load_models(ckpt)
    goal_critic = rebuild_sequential(
        torch.jit.load(str(ckpt / "GOAL_CRITIC.lt"), map_location="cpu"))
    print(f"pinned checkpoint {ckpt.name}", flush=True)

    res = {"checkpoint": int(ckpt.name), "gamma_dense": GAMMA_DENSE,
           "gamma_goal": GAMMA_GOAL, "lam": LAM, "beta": BETA}
    for ppt, rows in ((2, ROWS_2V2), (1, ROWS_1V1)):
        if rows <= 0:
            continue
        rec = rollout_team(SteeredPolicyRho(models), ppt, rows, SEED + ppt,
                           num_arenas=N_ARENAS, want_h2=True, want_obs=True)
        print(f"[{ppt}v{ppt}] rollout done ({time.time()-t0:.0f}s)", flush=True)
        analyse(rec, models, goal_critic, ppt, res)
        del rec

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"goal_critic_audit_{ckpt.name}.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"saved {out} ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
