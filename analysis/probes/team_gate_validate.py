"""TEAM_GATE offline sweep — targeted (best-placed-gated) team steering vs the
live untargeted mechanism. Design + frozen bars: TEAM_GATE.md.

Gated variants subclass SteeredPolicyRho and AND an extra per-row team gate
into the live rho-band gate:
  prox    steer row j only if j's car is strictly closer to the ball than its
          teammate's (both read from the canonical padded obs: ball@0, self@51,
          same scale 1/5000, teammates share the canonical frame).
  rhogap  steer row j only if j's contact-rho >= its teammate's (the same
          K-action rho the band gate already computes — "who wins the race"
          by the policy's own self-model).
Row pairing: the rollout batch is arena-major, npl consecutive rows per arena,
slot = row % npl, teammate slot = (slot + 2) % npl (ppt=2).
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

import RocketSim as rs
from load_checkpoint import copy_checkpoint, load_models
from steer_team import (SteeredPolicyRho, derive_team_direction, rollout_team,
                        team_metrics, team_possession_readings)

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
SEED = 20260720
PPT = 2
NPL = 2 * PPT
ROWS_DERIVE = 200_000     # 1v1 rollout for the transfer direction (E3 recipe)
ROWS_ARM = 400_000
N_ARENAS = 24
BALL_POS = slice(0, 3)
SELF_POS = slice(51, 54)


class TeamGatedPolicy(SteeredPolicyRho):
    def __init__(self, models, v, alpha, scale, team_gate, npl=NPL, **kw):
        super().__init__(models, v, alpha, scale, **kw)
        self.team_gate = team_gate      # 'none' | 'prox' | 'rhogap'
        self.npl = npl
        self.last_teamgate_frac = float("nan")

    @torch.no_grad()
    def act(self, obs, masks):
        h1, h2 = self.trunk_forward(obs)
        h2_pol = h2
        if self.v is not None and self.alpha != 0.0:
            n = h2.shape[0]
            rho = None
            gate = torch.ones(n)
            if self.rho_gate and self.phi is not None and self.psi_car is not None:
                if n >= 16:
                    maskF = masks.float().clamp_min(1e-9)
                    acts = torch.multinomial(maskF, self.rho_k, True)
                    trunk_rep = h2.repeat_interleave(self.rho_k, 0)
                    onehot = torch.nn.functional.one_hot(acts.flatten(), masks.shape[1]).float()
                    sa = self.phi(torch.cat([trunk_rep, onehot], -1))
                    sa = sa / sa.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                    rho = (sa @ self.psi0).view(n, self.rho_k).mean(-1)
                    lo, hi = torch.quantile(rho, self.rho_lo), torch.quantile(rho, self.rho_hi)
                    gate = ((rho >= lo) & (rho <= hi)).float()
                else:
                    gate = torch.zeros(n)
            self.last_inband_frac = float(gate.mean())

            if self.team_gate != "none" and n % self.npl == 0:
                slots = torch.arange(n) % self.npl
                tm_idx = torch.arange(n) - slots + (slots + 2) % self.npl
                if self.team_gate == "prox":
                    d = (obs[:, SELF_POS] - obs[:, BALL_POS]).norm(dim=-1)
                    tg = (d < d[tm_idx]).float()
                else:  # rhogap
                    tg = torch.ones(n) if rho is None else (rho >= rho[tm_idx]).float()
                self.last_teamgate_frac = float(tg.mean())
                gate = gate * tg

            h2_pol = h2 + gate.unsqueeze(-1) * (self.alpha * self.scale) * self.v
        return h2, self.sample_actions(h2_pol, masks)


def fmt(m):
    return (f"teamWon {m['team_won']:.1%} +-{m['team_won_se']:.1%} "
            f"NONE {m['none']:.1%} (n={m['n_feasible_readings']}/"
            f"{m['n_reading_episodes']}ep) | touch {m['touch_ratio']*100:.2f}% "
            f"air {m['in_air_ratio']:.0%} goals/ep {m['goals_per_episode']:.2f} "
            f"kick {m['kickoff_first_touch_s']:.2f}s")


def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    rs.init(str(HERE.parents[1] / "build" / "collision_meshes"))
    ckpt = copy_checkpoint(HERE / "data" / "ckpt_cache")
    models = load_models(ckpt)
    print(f"pinned checkpoint {ckpt.name}", flush=True)

    # transfer direction: derive from a 1v1 rollout at this checkpoint (E3 recipe)
    base1 = rollout_team(SteeredPolicyRho(models), 1, ROWS_DERIVE, SEED, want_h2=True)
    r1 = team_possession_readings(base1)
    rng = np.random.default_rng(SEED)
    v1, sig1, n1 = derive_team_direction(base1["h2"], r1, rng)
    print(f"1v1 commitment direction: {n1} matched pairs, sigma {sig1:.2f} "
          f"({time.time()-t0:.0f}s)", flush=True)

    arms = [
        ("A0", 0.0, "none"),
        ("A1", 0.5, "none"),
        ("A2", 1.0, "none"),
        ("G1", 0.5, "prox"),
        ("G2", 0.5, "rhogap"),
        ("G3", 1.0, "prox"),
    ]
    results = {"checkpoint": int(ckpt.name), "rows_arm": ROWS_ARM,
               "n_pairs": n1, "sigma": float(sig1), "arms": {}}
    for name, alpha, tg in arms:
        pol = TeamGatedPolicy(models, v1, alpha=alpha, scale=sig1, team_gate=tg)
        rec = rollout_team(pol, PPT, ROWS_ARM, SEED + 1, num_arenas=N_ARENAS)
        m = team_metrics(rec)
        m["inband_frac"] = pol.last_inband_frac
        m["teamgate_frac"] = pol.last_teamgate_frac
        results["arms"][name] = m
        print(f"{name} a={alpha:+.1f} gate={tg:6s}: {fmt(m)} "
              f"[inband {m['inband_frac']:.2f} tg {m['teamgate_frac']:.2f}]", flush=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"team_gate_{ckpt.name}.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"saved {out}  ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
