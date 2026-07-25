"""BREAKEVEN probe — is the learning signal for a genuinely beneficial mechanic
sub-noise? Prediction (skill-acquisition-tax law): at wavedash-like fragments
(measured speed GAIN on landing), the critic's own value trajectory shows a
gain indistinguishable from zero against episode noise. Control: matched
ordinary landings (same pre-landing speed tercile, no flip, no speed gain
requirement)."""

import json
import time
from pathlib import Path

import numpy as np
import torch

import RocketSim as rs
from advanced_obs import ACTION_TABLE
from fear_decomp import copy_newest
from load_checkpoint import load_models
from steer_team import SteeredPolicyRho, rollout_team
from team_decline_probe import cluster_boot_diff

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
SEED = 20260732
ROWS = 600_000
K_AFTER = 60          # 2s of same-player steps


def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    rs.init(str(HERE.parents[1] / "build" / "collision_meshes"))
    ckpt = copy_newest(HERE / "data" / "ckpt_cache")
    models = load_models(ckpt)
    print(f"pinned checkpoint {ckpt.name}", flush=True)

    rec = rollout_team(SteeredPolicyRho(models), 1, ROWS, SEED, num_arenas=24,
                       want_h2=True, want_goals=True)
    phys, episode, slot, on_ground = rec["phys"], rec["episode"], rec["slot"], rec["on_ground"]
    jump_flag = ACTION_TABLE[rec["action"].astype(int), 5] > 0.5
    with torch.no_grad():
        V = np.concatenate([models["CRITIC"](torch.from_numpy(
            rec["h2"][i:i + 65536].astype(np.float32))).flatten().numpy()
            for i in range(0, len(rec["h2"]), 65536)])
    zV = (V - V.mean()) / V.std()

    ep_rows = {int(e): np.flatnonzero(episode == e) for e in np.unique(episode)}
    wd_rows, wd_eps, wd_speed, wd_fut = [], [], [], []
    ctl_rows, ctl_eps, ctl_speed, ctl_fut = [], [], [], []
    att_rows, att_eps, att_speed, att_fut, att_ok = [], [], [], [], []
    for e, rows in ep_rows.items():
        for parity in (0, 1):
            pr = rows[parity::2]
            if len(pr) < K_AFTER + 6:
                continue
            og = on_ground[pr]
            hs = np.linalg.norm(phys[pr, 12:14], axis=1)
            jf = jump_flag[pr]
            for i in range(2, len(pr) - K_AFTER - 4):
                if og[i - 1] or not og[i]:
                    continue                      # need an air->ground landing at i
                attempted = jf[i:i + 3].any()
                is_wd = attempted and hs[min(i + 3, len(pr) - 1)] > hs[i - 1] + 100
                if attempted:
                    # ALL flip-at-landing attempts (success + failure): the
                    # break-even quantity is E[dV | attempted]
                    att_rows.append(pr[i]); att_eps.append(e)
                    att_speed.append(hs[i - 1]); att_fut.append(pr[i + K_AFTER])
                    att_ok.append(bool(is_wd))
                if is_wd:
                    wd_rows.append(pr[i]); wd_eps.append(e)
                    wd_speed.append(hs[i - 1]); wd_fut.append(pr[i + K_AFTER])
                elif not attempted:
                    ctl_rows.append(pr[i]); ctl_eps.append(e)
                    ctl_speed.append(hs[i - 1]); ctl_fut.append(pr[i + K_AFTER])

    wd_rows, wd_eps, wd_speed, wd_fut = map(np.array, (wd_rows, wd_eps, wd_speed, wd_fut))
    ctl_rows, ctl_eps, ctl_speed, ctl_fut = map(np.array, (ctl_rows, ctl_eps, ctl_speed, ctl_fut))
    print(f"landings: {len(wd_rows)} wavedash-like vs {len(ctl_rows)} controls", flush=True)

    # match controls to wavedash pre-landing speed terciles
    rng = np.random.default_rng(SEED)
    edges = np.quantile(wd_speed, [1 / 3, 2 / 3])
    sel_c = []
    for lo, hi in ((-1, edges[0]), (edges[0], edges[1]), (edges[1], 1e9)):
        w_n = int(((wd_speed >= lo) & (wd_speed < hi)).sum())
        c_pool = np.flatnonzero((ctl_speed >= lo) & (ctl_speed < hi))
        sel_c += list(rng.choice(c_pool, min(w_n * 3, len(c_pool)), replace=False))
    sel_c = np.array(sel_c)

    dv_w = zV[wd_fut] - zV[wd_rows]
    dv_c = zV[ctl_fut[sel_c]] - zV[ctl_rows[sel_c]]

    # the break-even quantity: attempt-conditioned
    att_rows_a, att_eps_a = np.array(att_rows), np.array(att_eps)
    att_fut_a, att_ok_a = np.array(att_fut), np.array(att_ok)
    dv_a = zV[att_fut_a] - zV[att_rows_a]
    d_att = float(np.nanmean(dv_a) - np.nanmean(dv_c))
    se_att = cluster_boot_diff(dv_a[np.isfinite(dv_a)], dv_c[np.isfinite(dv_c)],
                               att_eps_a[np.isfinite(dv_a)],
                               np.array(ctl_eps)[sel_c][np.isfinite(dv_c)], seed=2)
    print(f"attempts: {len(att_rows_a)} (success rate {att_ok_a.mean():.1%}) | "
          f"E[dV|attempt] - E[dV|control] = {d_att:+.3f} +- {se_att:.3f}", flush=True)
    okw, okc = np.isfinite(dv_w), np.isfinite(dv_c)
    d = float(np.nanmean(dv_w) - np.nanmean(dv_c))
    se = cluster_boot_diff(dv_w[okw], dv_c[okc], wd_eps[okw], ctl_eps[sel_c][okc], seed=1)
    res = {"checkpoint": int(ckpt.name), "n_wavedash": int(okw.sum()), "n_control": int(okc.sum()),
           "dV_wavedash": float(np.nanmean(dv_w)), "dV_control": float(np.nanmean(dv_c)),
           "diff": d, "se": se, "sub_noise": bool(abs(d) < 2 * se)}
    print(f"dV(+2s): wavedash {res['dV_wavedash']:+.3f} vs control {res['dV_control']:+.3f} "
          f"-> diff {d:+.3f} +- {se:.3f} ({'SUB-NOISE' if res['sub_noise'] else 'RESOLVED'})", flush=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"breakeven_probe_{ckpt.name}.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"saved {out}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
