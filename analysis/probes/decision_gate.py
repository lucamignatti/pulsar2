"""DECISION_GATE — premeditation-band gating vs the live rho-band gate.
Design + frozen bars: DECISION_GATE.md."""

import json
import time
from pathlib import Path

import numpy as np
import torch

import RocketSim as rs
from load_checkpoint import copy_checkpoint, load_models
from steer_team import (DT, NONE, WON, SteeredPolicyRho, derive_team_direction,
                        rollout_team, team_metrics, team_possession_readings)
from team_decline_probe import decline_readings
from team_gate_validate import TeamGatedPolicy, fmt

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
SEED = 20260724
N_ARENAS = 24
TOUCH_HORIZON_S = 1.0


def premeditation_direction(rec, rng, max_rows=120_000):
    """w = difference of h2 means, label = same-player touch within 1s.
    Live parity: the trainer has this label in combinedTraj.touched. Returns
    (w, held-out AUC of the projection)."""
    episode, slot, touched = rec["episode"], rec["slot"], rec["touched"]
    npl = len(np.unique(slot))
    ep_rows = {int(e): np.flatnonzero(episode == e) for e in np.unique(episode)}
    row_pos = np.empty(len(slot), np.int64)
    for rows in ep_rows.values():
        row_pos[rows] = np.arange(len(rows))
    k = int(round(TOUCH_HORIZON_S / DT))

    n = len(slot)
    idx = rng.permutation(n)[:max_rows]
    y = np.zeros(len(idx), bool)
    for i, r in enumerate(idx):
        rows = ep_rows[int(episode[r])]
        q = row_pos[r]
        for j in range(1, k + 1):
            qq = q + npl * j
            if qq >= len(rows):
                break
            if touched[rows[qq]]:
                y[i] = True
                break
    H = rec["h2"][idx].astype(np.float32)
    eps = episode[idx]
    uniq = np.unique(eps)
    rng.shuffle(uniq)
    tr_eps = set(uniq[: int(0.8 * len(uniq))].tolist())
    tr = np.isin(eps, list(tr_eps))
    w = H[tr & y].mean(0) - H[tr & ~y].mean(0)
    w /= max(np.linalg.norm(w), 1e-8)
    proj = H[~tr] @ w
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(y[~tr], proj)) if len(np.unique(y[~tr])) == 2 else float("nan")
    return w, auc, float(y.mean())


class DecisionGatedPolicy(TeamGatedPolicy):
    """Premeditation-band gate replaces the rho band: steer rows whose intention
    readout sits in the middle quantile band of the batch (undecided moments)."""

    def __init__(self, models, v, alpha, scale, w_premed, lo=0.2, hi=0.8):
        super().__init__(models, v, alpha, scale, team_gate="none", rho_gate=False)
        self.w = torch.as_tensor(w_premed, dtype=torch.float32)
        self.lo, self.hi = lo, hi

    @torch.no_grad()
    def act(self, obs, masks):
        h1, h2 = self.trunk_forward(obs)
        h2_pol = h2
        if self.v is not None and self.alpha != 0.0:
            n = h2.shape[0]
            if n >= 16:
                p = h2 @ self.w
                lo, hi = torch.quantile(p, self.lo), torch.quantile(p, self.hi)
                gate = ((p >= lo) & (p <= hi)).float()
            else:
                gate = torch.zeros(n)  # fail closed, live parity
            self.last_inband_frac = float(gate.mean())
            h2_pol = h2 + gate.unsqueeze(-1) * (self.alpha * self.scale) * self.v
        return h2, self.sample_actions(h2_pol, masks)


def run_mode(models, ppt, rows_arm, v, sig, w_premed, seed):
    out = {}
    arms = [
        ("D0", lambda: TeamGatedPolicy(models, None, 0.0, 0.0, "none")),
        ("Drho", lambda: TeamGatedPolicy(models, v, 0.5, sig, "none")),
        ("Ddec", lambda: DecisionGatedPolicy(models, v, 0.5, sig, w_premed)),
    ]
    for name, mk in arms:
        pol = mk()
        rec = rollout_team(pol, ppt, rows_arm, seed, num_arenas=N_ARENAS)
        m = team_metrics(rec)
        m["inband_frac"] = pol.last_inband_frac
        out[name] = m
        print(f"{ppt}v{ppt} {name}: {fmt(m)} [inband {m['inband_frac']:.2f}]", flush=True)
    return out


def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    rs.init(str(HERE.parents[1] / "build" / "collision_meshes"))
    ckpt = copy_checkpoint(HERE / "data" / "ckpt_cache")
    models = load_models(ckpt)
    print(f"pinned checkpoint {ckpt.name}", flush=True)
    rng = np.random.default_rng(SEED)
    results = {"checkpoint": int(ckpt.name), "modes": {}}

    # ---- 1v1: v2 commitment direction + premeditation from the same rollout
    base1 = rollout_team(SteeredPolicyRho(models), 1, 500_000, SEED, want_h2=True)
    r1 = team_possession_readings(base1)
    v1, sig1, n1 = derive_team_direction(base1["h2"], r1, rng)
    w1, auc1, rate1 = premeditation_direction(base1, rng)
    print(f"1v1: commit {n1} pairs sigma {sig1:.2f} | premed AUC {auc1:.3f} "
          f"(touch1s rate {rate1:.2%}) ({time.time()-t0:.0f}s)", flush=True)
    results["modes"]["1v1"] = {"n_pairs": n1, "premed_auc": auc1,
                               "arms": run_mode(models, 1, 600_000, v1, sig1, w1, SEED + 1)}

    # ---- 2v2: best-placed-conditioned direction + premeditation
    base2 = rollout_team(SteeredPolicyRho(models), 2, 700_000, SEED + 2,
                         num_arenas=N_ARENAS, want_h2=True)
    rd = decline_readings(base2)
    bp = rd["feas_self"] & rd["best_placed"] & np.isin(rd["outcome"], (WON, NONE))
    r_bp = {k: rd[k][bp] for k in ("row", "outcome", "d_now", "t_land")}
    v2, sig2, n2 = derive_team_direction(base2["h2"], r_bp, rng)
    w2, auc2, rate2 = premeditation_direction(base2, rng)
    print(f"2v2: bp {n2} pairs sigma {sig2:.2f} | premed AUC {auc2:.3f} "
          f"(touch1s rate {rate2:.2%})", flush=True)
    results["modes"]["2v2"] = {"n_pairs": n2, "premed_auc": auc2,
                               "arms": run_mode(models, 2, 800_000, v2, sig2, w2, SEED + 3)}

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"decision_gate_{ckpt.name}.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"saved {out}  ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
