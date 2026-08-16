"""Q1 of research/reports/KD_ROLLOUT.md: validate the rollout-only knowing-doing
statistic against the landing-attendance ground truth, on 7.0/AiMOS-lineage snapshots.

Per pre-registration (2026-08-16):
  c(s,a)    = max_g cos(phi(h2(s),a), psi_car(g)) / tau  over a witnessed touch-goal bank G
  rho_can   = max over VALID actions of c(s,a)
  rho_do    = sum_a pi(a|s) c(s,a)
  KD        = rho_can - rho_do
Ground truth: kd_curve.py attendance protocol (feasible / free / attended landings).
Criteria: S1 AUC(KD; feas-unatt vs infeas-unatt) >= 0.70 and >= -rho_do AUC + 0.05;
S2 AUC(KD; feas-unatt vs feas-att) >= 0.60; S3 |spearman(KD, H)| < 0.5.

Usage:
  cd research/tools && OMP_NUM_THREADS=4 nice -n 19 ../.venv/bin/python kd_rollout.py \
      --ckpt ../data/ckpt70_aimos/<ts>
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import RocketSim as rs
import collect_dataset as cd
from collect_dataset import ArenaEnv, set_obs_size
from collect_dataset_53 import LIVE_RESET_MIX, _reset_with_airplay
from calibrate_rho import car_ball_goal
from label_landing import simulate_landing
from load_checkpoint_70 import Pulsar70Policy, load_models

HERE = Path(__file__).resolve().parent
SEED = 20260816
NUM_ARENAS = 16
TARGET_FRAMES = 120_000
N_ANCHORS = 12_000
BANK_M = 64
TAU = 0.02                    # EvalRho parity (calibrate_rho.py)
DT = None                     # set from cd.TICK_SKIP at runtime
KNOW_T_MAX = 1.33             # primary slice: t_land <= 20 steps (car HER window ~0.67s)
KNOW_T_MIN = 0.13
ATTEND = 500.0
FEASIBLE_SPEED = 1300.0
BANK_RELP_MAX = 300.0 / 2300.0   # touch-config sanity: |rel_p| < 300 uu (normalized)


def collect(policy):
    torch.manual_seed(SEED)
    envs = [ArenaEnv(i, np.random.default_rng(SEED + 1000 + i)) for i in range(NUM_ARENAS)]
    n = TARGET_FRAMES
    out = {
        "h2": np.empty((n, policy.h2_width), np.float16),
        "probs": np.empty((n, 90), np.float16),
        "mask": np.empty((n, 90), np.uint8),
        "phi_taken": np.empty((n, 128), np.float16),
        "g_car": np.empty((n, 6), np.float32),
        "phys": np.empty((n, 31), np.float32),
        "headroom": np.empty(n, np.float32),
        "episode": np.empty(n, np.int32),
        "team": np.empty(n, np.int8),
        "touched": np.zeros(n, bool),
    }
    row, t0, last_touch = 0, time.time(), {}
    while row + 2 * NUM_ARENAS <= n:
        obs_l, mask_l, phys_l, gcar_l = [], [], [], []
        for env in envs:
            o, m, p = env.observe()
            obs_l.append(o); mask_l.append(m); phys_l.append(p)
            ball = env.arena.ball.get_state()
            bp, bv = np.array(ball.pos.as_tuple()), np.array(ball.vel.as_tuple())
            for k, car in enumerate(env.cars):
                st = car.get_state()
                rm = st.rot_mat
                gcar_l.append(car_ball_goal(
                    bp, bv, np.array(st.pos.as_tuple()), np.array(st.vel.as_tuple()),
                    np.array(rm.forward.as_tuple()), np.array(rm.right.as_tuple()),
                    np.array(rm.up.as_tuple())))
        obs_b = torch.from_numpy(np.concatenate(obs_l))
        mask_b = torch.from_numpy(np.concatenate(mask_l))
        _, h2 = policy.trunk_forward(obs_b)
        probs = policy.action_probs(h2, mask_b)
        actions = torch.multinomial(probs, 1, True).flatten()
        phi_t = policy.phi_embedding(h2, actions)
        with torch.no_grad():
            vt = policy.models["CRITIC_TRUNK"](h2)
            v = policy.models["CRITIC"](vt).flatten()
            if "CRITIC2" in policy.models:
                v = (v + policy.models["CRITIC2"](vt).flatten()) / 2
            vdmin = torch.minimum(policy.models["VDAG1"](vt).flatten(),
                                  policy.models["VDAG2"](vt).flatten())
            hroom = torch.relu(vdmin - v)
        for i, env in enumerate(envs):
            prev = last_touch.get(env.idx, -1)
            for k in range(2):
                j = 2 * i + k
                out["h2"][row] = h2[j].numpy()
                out["probs"][row] = probs[j].numpy()
                out["mask"][row] = mask_b[j].numpy()
                out["phi_taken"][row] = phi_t[j].numpy()
                out["g_car"][row] = gcar_l[j]
                out["phys"][row] = phys_l[i]      # one 31-dim snapshot per ARENA (both cars)
                out["headroom"][row] = hroom[j].item()
                out["episode"][row] = env.episode_id
                out["team"][row] = k
                out["touched"][row] = env.last_touch_tick != prev and prev >= 0
                row += 1
            last_touch[env.idx] = env.last_touch_tick
        for i, env in enumerate(envs):
            if env.step(actions[2 * i:2 * i + 2].tolist()):
                env.reset()
        if row % 32_000 < 2 * NUM_ARENAS:
            print(f"  {row:>7,}/{n:,}  {row/(time.time()-t0):,.0f} rows/s", flush=True)
    for k in out:
        out[k] = out[k][:row]
    print(f"collected {row:,} rows ({out['episode'].max()+1} episodes, "
          f"touch rate {out['touched'].mean():.2%}) in {time.time()-t0:.0f}s")
    return out


def touch_bank(data, rng):
    """M witnessed car-local touch configs, sampled across distinct episodes."""
    cand = np.flatnonzero(data["touched"]
                          & (np.linalg.norm(data["g_car"][:, :3], axis=1) < BANK_RELP_MAX))
    if len(cand) < BANK_M:
        raise RuntimeError(f"touch bank too small: {len(cand)} candidates")
    order = rng.permutation(cand)
    picked, seen_ep = [], set()
    for r in order:                      # prefer episode diversity, then fill
        ep = int(data["episode"][r])
        if ep not in seen_ep:
            picked.append(r); seen_ep.add(ep)
        if len(picked) == BANK_M:
            break
    for r in order:
        if len(picked) == BANK_M:
            break
        if r not in picked:
            picked.append(r)
    return data["g_car"][np.array(picked)]


def rho_reads(policy, data, anchors, bank):
    """Exact rho_can / rho_do / c_taken per anchor (all 90 actions, bank-max + bank-mean)."""
    with torch.no_grad():
        psi = policy.models["REACH_PSI_CAR"](torch.from_numpy(bank))
        psi = (psi / psi.norm(dim=-1, keepdim=True).clamp_min(1e-6)).numpy()  # (M,128)
    nA = len(anchors)
    all_a = torch.arange(90)
    res = {k: np.empty(nA, np.float32) for k in
           ["rho_can", "rho_do", "rho_can_mean", "rho_do_mean", "c_taken"]}
    B = 256
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, nA, B):
            rows = anchors[s:s + B]
            h2 = torch.from_numpy(data["h2"][rows]).float()          # (b,1280)
            b = len(rows)
            h2r = h2.repeat_interleave(90, 0)
            ar = all_a.repeat(b)
            emb = policy.phi_embedding(h2r, ar).reshape(b, 90, 128).numpy()
            sc = np.einsum("bak,mk->bam", emb, psi) / TAU            # (b,90,M)
            c_max, c_mean = sc.max(-1), sc.mean(-1)                  # (b,90)
            valid = data["mask"][rows].astype(bool)
            p = data["probs"][rows].astype(np.float32) * valid
            p /= p.sum(-1, keepdims=True).clip(1e-9)
            res["rho_can"][s:s + b] = np.where(valid, c_max, -np.inf).max(-1)
            res["rho_do"][s:s + b] = (p * np.where(valid, c_max, 0)).sum(-1)
            res["rho_can_mean"][s:s + b] = np.where(valid, c_mean, -np.inf).max(-1)
            res["rho_do_mean"][s:s + b] = (p * np.where(valid, c_mean, 0)).sum(-1)
            pt = torch.from_numpy(data["phi_taken"][rows]).float().numpy()
            sct = np.einsum("bk,mk->bm", pt, psi) / TAU
            res["c_taken"][s:s + b] = sct.max(-1)
            if s % (B * 8) == 0:
                print(f"  rho {s:>6,}/{nA:,}  ({time.time()-t0:.0f}s)", flush=True)
    return res


def attendance(data, anchors, landings):
    """kd_curve.py protocol on the interleaved 2-players-per-step layout."""
    episode, team, phys = data["episode"], data["team"], data["phys"]
    ep_rows = {int(e): np.flatnonzero(episode == e) for e in np.unique(episode)}
    row_pos = np.empty(len(episode), np.int64)
    for rows in ep_rows.values():
        row_pos[rows] = np.arange(len(rows))
    recs = {}
    for idx, r in enumerate(anchors):
        res = landings[idx]
        if res is None:
            continue
        lx, ly, t = res
        rows = ep_rows[int(episode[r])]
        q = row_pos[r]
        q_land = q + 2 * int(round(t / DT))
        if q_land >= len(rows):
            continue                                  # censored
        L = np.array([lx, ly])
        sl = slice(20, 23) if team[r] == 1 else slice(9, 12)
        d_now = np.linalg.norm(phys[rows[q], sl][:2] - L)
        feasible = d_now / max(t, 1e-6) < FEASIBLE_SPEED
        ball_td = phys[rows[q_land], 0:3]
        free = np.linalg.norm(ball_td[:2] - L) <= 300 and ball_td[2] <= 200
        went = np.linalg.norm(phys[rows[q_land], sl][:2] - L) <= ATTEND
        recs[idx] = dict(feasible=bool(feasible), free=bool(free), went=bool(went),
                         t_land=float(t), d_now=float(d_now))
    return recs


def auc(pos, neg):
    """Mann-Whitney AUC of pos > neg."""
    if len(pos) == 0 or len(neg) == 0:
        return None
    from scipy.stats import mannwhitneyu
    u = mannwhitneyu(pos, neg, alternative="two-sided").statistic
    return float(u / (len(pos) * len(neg)))


def analyze(data, anchors, rho, att, tag):
    kd = rho["rho_can"] - rho["rho_do"]
    kd_mean = rho["rho_can_mean"] - rho["rho_do_mean"]

    idxs = np.array(sorted(att))
    rec = lambda k: np.array([att[i][k] for i in idxs])
    feas, free, went, t_land, d_now = (rec(k) for k in
                                       ["feasible", "free", "went", "t_land", "d_now"])
    use = free & ((t_land >= KNOW_T_MIN) & (t_land <= KNOW_T_MAX) if tag == "primary"
                  else np.ones(len(idxs), bool))
    cells = {
        "feas_unatt": use & feas & ~went,
        "feas_att":   use & feas & went,
        "infeas_unatt": use & ~feas & ~went,
        "infeas_att": use & ~feas & went,
    }
    a = {name: idxs[m] for name, m in cells.items()}
    out = {"slice": tag, "n_cells": {k: int(len(v)) for k, v in a.items()}}

    # rho/H arrays are positional over `anchors`; att keys are positions into anchors.
    def vals(vec, cell):
        return vec[a[cell]] if len(a[cell]) else np.array([])

    signals = {
        "KD": kd, "KD_bankmean": kd_mean,
        "rho_can": rho["rho_can"], "neg_rho_do": -rho["rho_do"],
        "c_taken": rho["c_taken"],
        "H": data["headroom"][anchors],
        "neg_d_now_rate": None,   # filled below
    }
    dn = np.full(len(anchors), np.nan)
    tl = np.full(len(anchors), np.nan)
    dn[idxs], tl[idxs] = d_now, t_land
    signals["neg_d_now_rate"] = -(dn / np.maximum(tl, 1e-6))

    out["cell_means"] = {s: {c: (float(np.mean(vals(v, c))) if len(a[c]) else None)
                             for c in cells} for s, v in signals.items()}
    out["auc"] = {}
    for s, v in signals.items():
        out["auc"][s] = {
            "S1_feasunatt_vs_infeasunatt": auc(vals(v, "feas_unatt"), vals(v, "infeas_unatt")),
            "S2_feasunatt_vs_feasatt": auc(vals(v, "feas_unatt"), vals(v, "feas_att")),
        }
    return out, kd


def main():
    global DT
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frames", type=int, default=TARGET_FRAMES)
    args = ap.parse_args()
    ckpt = Path(args.ckpt)

    torch.set_num_threads(4)
    rng = np.random.default_rng(SEED)
    rs.init(str(HERE.parents[1] / "build" / "collision_meshes"))
    DT = cd.TICK_SKIP / 120.0

    policy = Pulsar70Policy(load_models(ckpt))
    set_obs_size(policy.obs_size)
    cd.RESET_MIX = LIVE_RESET_MIX
    ArenaEnv.reset = _reset_with_airplay
    print(f"checkpoint {ckpt.name} ({int(ckpt.name)/1e9:.1f}B), obs {policy.obs_size}, "
          f"tickSkip {cd.TICK_SKIP}")

    globals()["TARGET_FRAMES"] = args.frames
    data = collect(policy)
    bank = touch_bank(data, rng)
    print(f"touch bank: {len(bank)} configs, |rel_p| mean "
          f"{np.linalg.norm(bank[:, :3], axis=1).mean()*2300:.0f} uu")

    airborne = np.flatnonzero(data["phys"][:, 2] > 300)
    anchors = airborne if len(airborne) <= N_ANCHORS else \
        np.sort(rng.choice(airborne, N_ANCHORS, replace=False))
    print(f"{len(airborne):,} airborne rows, {len(anchors):,} anchors")

    arena = rs.Arena(rs.GameMode.SOCCAR)
    t0 = time.time()
    landings = [simulate_landing(arena, data["phys"][r, 0:3], data["phys"][r, 3:6],
                                 data["phys"][r, 6:9]) for r in anchors]
    print(f"landing labels: {sum(l is not None for l in landings):,} in {time.time()-t0:.0f}s")

    rho = rho_reads(policy, data, anchors, bank)
    att = attendance(data, anchors, landings)
    print(f"attendance records: {len(att):,}")

    from scipy.stats import spearmanr
    results = {"checkpoint": int(ckpt.name), "seed": SEED, "n_rows": len(data["team"]),
               "n_anchors": int(len(anchors)), "bank_m": len(bank), "tau": TAU,
               "slices": []}
    for tag in ["primary", "full"]:
        out, kd = analyze(data, anchors, rho, att, tag)
        results["slices"].append(out)
    kd = rho["rho_can"] - rho["rho_do"]
    s3 = spearmanr(kd, data["headroom"][anchors])
    results["S3_spearman_KD_H"] = float(s3.statistic)
    results["kd_summary"] = {"mean": float(kd.mean()), "p90": float(np.quantile(kd, 0.9)),
                             "rho_can_mean": float(rho["rho_can"].mean()),
                             "rho_do_mean": float(rho["rho_do"].mean())}

    outp = HERE.parent / "results" / f"kd_rollout_{ckpt.name}.json"
    outp.parent.mkdir(exist_ok=True)
    outp.write_text(json.dumps(results, indent=2))

    print(f"\n===== {ckpt.name} =====")
    print(f"S3 spearman(KD, H) = {s3.statistic:+.3f}  (pass: |.| < 0.5)")
    for sl in results["slices"]:
        print(f"-- slice {sl['slice']}  cells {sl['n_cells']}")
        for s in ["KD", "neg_rho_do", "rho_can", "H", "KD_bankmean"]:
            a1 = sl["auc"][s]["S1_feasunatt_vs_infeasunatt"]
            a2 = sl["auc"][s]["S2_feasunatt_vs_feasatt"]
            f1 = f"{a1:.3f}" if a1 is not None else "n/a"
            f2 = f"{a2:.3f}" if a2 is not None else "n/a"
            print(f"   {s:14s} S1 {f1}   S2 {f2}")
        cm = sl["cell_means"]["KD"]
        print("   KD cell means: " + "  ".join(f"{c}={cm[c]:.2f}" if cm[c] is not None
                                               else f"{c}=n/a" for c in cm))
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
