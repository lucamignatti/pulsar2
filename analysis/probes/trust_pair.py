"""TRUST_PAIR — paired commit+trust role steering. Design + frozen bars:
TRUST_PAIR.md."""

import json
import time
from pathlib import Path

import numpy as np
import torch

import RocketSim as rs
from fear_decomp import copy_newest
from load_checkpoint import load_models
from steer_team import (NONE, TEAMMATE, WON, SteeredPolicyRho,
                        derive_team_direction, rollout_team, team_metrics)
from team_decline_probe import decline_readings, cluster_boot_diff
from team_gate_validate import TeamGatedPolicy, fmt

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
SEED = 20260735
PPT = 2
NPL = 2 * PPT
ROWS_DERIVE = 700_000
ROWS_ARM = 800_000
N_ARENAS = 24


class PairedPolicy(TeamGatedPolicy):
    """Role-split steering: the teammate CLOSER to the ball gets v_commit, the
    farther one gets v_trust (obs-local proximity split, live-parity)."""

    def __init__(self, models, v_commit, sig_c, v_trust, sig_t, alpha=0.5):
        super().__init__(models, v_commit, alpha, sig_c, team_gate="none")
        self.vt = None if v_trust is None else torch.as_tensor(v_trust, dtype=torch.float32)
        self.sig_t = sig_t

    @torch.no_grad()
    def act(self, obs, masks):
        h1, h2 = self.trunk_forward(obs)
        h2_pol = h2
        n = h2.shape[0]
        if n % self.npl == 0 and (self.v is not None or self.vt is not None):
            slots = torch.arange(n) % self.npl
            tm_idx = torch.arange(n) - slots + (slots + 2) % self.npl
            d = (obs[:, 51:54] - obs[:, 0:3]).norm(dim=-1)
            closer = (d < d[tm_idx]).float().unsqueeze(-1)
            delta = torch.zeros_like(h2)
            if self.v is not None:
                delta = delta + closer * (self.alpha * self.scale) * self.v
            if self.vt is not None:
                delta = delta + (1 - closer) * (self.alpha * self.sig_t) * self.vt
            h2_pol = h2 + delta
        return h2, self.sample_actions(h2_pol, masks)


def backfill_metric(rec, rd):
    """Among readings where the best-placed teammate pursues, the OTHER
    teammate's canonical goal-side displacement over the next 1.5s."""
    phys, episode, slot = rec["phys"], rec["episode"], rec["slot"]
    ep_rows = {int(e): np.flatnonzero(episode == e) for e in np.unique(episode)}
    row_pos = np.empty(len(slot), np.int64)
    for rows in ep_rows.values():
        row_pos[rows] = np.arange(len(rows))
    k = int(round(1.5 * 30))
    vals, eps = [], []
    sub = np.flatnonzero(rd["feas_self"] & rd["best_placed"] & rd["pursued_self"])
    for i in sub:
        r = int(rd["row"][i])
        rows = ep_rows[int(episode[r])]
        q = int(row_pos[r])
        tm_off = ((int(slot[r]) + 2) % NPL) - int(slot[r])
        q_end = q + NPL * k + tm_off
        if q_end >= len(rows):
            continue
        team = int(slot[r]) % 2
        flip = 1.0 if team == 0 else -1.0
        y0 = phys[rows[q + tm_off], 10] * flip
        y1 = phys[rows[q_end], 10] * flip
        vals.append(y0 - y1)   # positive = teammate moved TOWARD own goal (covered)
        eps.append(int(episode[r]))
    return np.array(vals), np.array(eps)


def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    rs.init(str(HERE.parents[1] / "build" / "collision_meshes"))
    ckpt = copy_newest(HERE / "data" / "ckpt_cache")
    models = load_models(ckpt)
    print(f"pinned checkpoint {ckpt.name}", flush=True)
    rng = np.random.default_rng(SEED)

    base = rollout_team(SteeredPolicyRho(models), PPT, ROWS_DERIVE, SEED,
                        num_arenas=N_ARENAS, want_h2=True)
    rd = decline_readings(base)
    bp = rd["feas_self"] & rd["best_placed"] & np.isin(rd["outcome"], (WON, NONE))
    r_c = {k: rd[k][bp] for k in ("row", "outcome", "d_now", "t_land")}
    v_c, sig_c, n_c = derive_team_direction(base["h2"], r_c, rng)
    # trust: NON-best-placed rows, TEAMMATE (took it) vs NONE (nobody did)
    tr = (~rd["best_placed"]) & np.isin(rd["outcome"], (TEAMMATE, NONE))
    r_t = {"row": rd["row"][tr], "outcome": np.where(rd["outcome"][tr] == TEAMMATE, WON, NONE),
           "d_now": rd["d_now"][tr], "t_land": rd["t_land"][tr]}
    v_t, sig_t, n_t = derive_team_direction(base["h2"], r_t, rng)
    print(f"commit {n_c} pairs sigma {sig_c:.2f} | trust {n_t} pairs sigma {sig_t:.2f} "
          f"| cos {float(np.dot(v_c, v_t)):+.2f} ({time.time()-t0:.0f}s)", flush=True)

    arms = [("B0", None, None), ("C", v_c, None), ("T", None, v_t), ("P", v_c, v_t)]
    res = {"checkpoint": int(ckpt.name), "n_pairs_commit": n_c, "n_pairs_trust": n_t,
           "cos_commit_trust": float(np.dot(v_c, v_t)), "arms": {}}
    for name, vc, vt in arms:
        pol = PairedPolicy(models, vc, sig_c, vt, sig_t)
        rec = rollout_team(pol, PPT, ROWS_ARM, SEED + 1, num_arenas=N_ARENAS)
        m = team_metrics(rec)
        rda = decline_readings(rec)
        bf, bfe = backfill_metric(rec, rda)
        m["backfill_mean"] = float(bf.mean()) if len(bf) else None
        m["backfill_n"] = int(len(bf))
        if len(bf):
            m["backfill_se"] = float(np.std([bf[np.isin(bfe, rng.choice(np.unique(bfe),
                len(np.unique(bfe))))].mean() for _ in range(200)]))
        res["arms"][name] = m
        print(f"{name}: {fmt(m)} | backfill {m['backfill_mean']} +- "
              f"{m.get('backfill_se')} (n={m['backfill_n']})", flush=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"trust_pair_{ckpt.name}.json").write_text(json.dumps(res, indent=1))
    print(f"saved ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
