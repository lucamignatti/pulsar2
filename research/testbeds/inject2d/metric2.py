"""Gamma-critic, iteration 2: the reachability kernel trained like V-dagger.

d(s,a): twin plain nets (softplus >= 0). No architectural quasimetric, no
Lagrangian -- the composition law is a TD-style constraint enforced with the
same expectile discipline the composition critic uses:

  edge        d(s,s') = 1 on executed transitions        (calibration anchor)
  self        d(s,s)  = 0
  composition d(s,a) <= 1 + max(d1_tgt, d2_tgt)(s',a)    (triangle, Bellman form)
              expectile tau=0.3: proven bounds enforced hard, optimism soft
  floor       mean over random pairs of gamma^d matched to a small occupancy
              target (collapse prevention, calibrated not adversarial)

Twins with MAX in the composition target (conservative among optimists = the
anti-ratchet for distances). Readout: V''(s) = max(Vdag(s), top-k mean of
gamma^d(s,a) * Vdag(a)_+ over an anchor sample) -- soft, not winner's-curse max.
Same pre-registered E1-E4 as metric.py; E4 now FINE-TUNES (never retrains).
"""
import argparse, copy, glob, json, os
import numpy as np
import torch
import torch.nn as nn

from env import OBS_DIM
from ladder import mlp
from foresight import make_probes, oracle_scores
from metric import (make_air_probes, make_ground_probes, phase2_world_pairs)

GAMMA = 0.99
DCAP = 120.0
FLOOR_T = 0.05   # target mean gamma^d over random ordered pairs


class DNet(nn.Module):
    def __init__(self, obs_dim, h=128):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(2 * obs_dim, h), nn.Tanh(),
                               nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1))

    def forward(self, s, a):
        return nn.functional.softplus(self.f(torch.cat([s, a], -1))).flatten()


def expectile(pred, target, tau):
    u = target - pred
    w = torch.where(u > 0, tau, 1 - tau)
    return (w * u * u).mean()


def train_kernel(pairs_obs, pairs_nxt, iters=4000, bs=1024, seed=0,
                 nets=None, tau=0.3):
    torch.manual_seed(seed)
    n = pairs_obs.shape[0]
    if nets is None:
        d1, d2 = DNet(OBS_DIM), DNet(OBS_DIM)
    else:
        d1, d2 = nets
    t1, t2 = copy.deepcopy(d1), copy.deepcopy(d2)
    opt = torch.optim.Adam(list(d1.parameters()) + list(d2.parameters()), lr=1e-3)
    for i in range(iters):
        idx = torch.randint(0, n, (bs,))
        o, nx = pairs_obs[idx], pairs_nxt[idx]
        a = pairs_obs[torch.randint(0, n, (bs,))]
        loss = 0.0
        with torch.no_grad():
            tcomp = (1.0 + torch.maximum(t1(nx, a), t2(nx, a))).clamp(max=DCAP)
        for dn in (d1, d2):
            loss = loss + ((dn(o, nx) - 1.0) ** 2).mean()          # edge
            loss = loss + (dn(o, o) ** 2).mean()                    # self
            loss = loss + expectile(dn(o, a), tcomp, tau)           # composition
            gam = GAMMA ** dn(o, a).clamp(max=DCAP)                 # floor
            loss = loss + 10.0 * (gam.mean() - FLOOR_T) ** 2
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(list(d1.parameters()) + list(d2.parameters()), 1.0)
        opt.step()
        if i % 50 == 0:
            t1.load_state_dict(d1.state_dict()); t2.load_state_dict(d2.state_dict())
    with torch.no_grad():
        idx = torch.randint(0, n, (4096,))
        j = torch.randint(0, n, (4096,))
        dmax = lambda s, a: torch.maximum(d1(s, a), d2(s, a))
        stats = {"edge": float(dmax(pairs_obs[idx], pairs_nxt[idx]).mean()),
                 "rand": float(dmax(pairs_obs[idx], pairs_obs[j]).mean()),
                 "selfd": float(dmax(pairs_obs[idx], pairs_obs[idx]).mean())}
    return (d1, d2), stats


@torch.no_grad()
def vdd2(nets, vdag_fn, probes, anchors, topk=8, chunk=64):
    d1, d2 = nets
    va = vdag_fn(anchors).clamp(min=0)
    out = torch.empty(probes.shape[0])
    na = anchors.shape[0]
    for i0 in range(0, probes.shape[0], chunk):
        p = probes[i0:i0 + chunk]
        ps = p.unsqueeze(1).expand(-1, na, -1).reshape(-1, OBS_DIM)
        an = anchors.unsqueeze(0).expand(p.shape[0], -1, -1).reshape(-1, OBS_DIM)
        d = torch.maximum(d1(ps, an), d2(ps, an)).view(p.shape[0], na).clamp(max=DCAP)
        reach = (GAMMA ** d) * va.unsqueeze(0)
        top = reach.topk(topk, dim=1).values.mean(1)
        out[i0:i0 + chunk] = torch.maximum(top, vdag_fn(p))
    return out


def finetune_vdag(base, ro, rn, rr, iters=1500, bs=1024, seed=0, tau=0.75):
    """E4 control: CONTINUE training the base twins on the augmented pair set."""
    torch.manual_seed(seed)
    f1, f2 = mlp(OBS_DIM, 1), mlp(OBS_DIM, 1)
    f1.load_state_dict(base["vdag1"]); f2.load_state_dict(base["vdag2"])
    t1, t2 = copy.deepcopy(f1), copy.deepcopy(f2)
    opt = torch.optim.Adam(list(f1.parameters()) + list(f2.parameters()), lr=3e-4)
    n = ro.shape[0]
    for i in range(iters):
        idx = torch.randint(0, n, (bs,))
        o, nx, r = ro[idx], rn[idx], rr[idx]
        with torch.no_grad():
            y = (r + GAMMA * torch.minimum(t1(nx).flatten(), t2(nx).flatten())
                 ).clamp(-5.0, 50.0)
        u1 = y - f1(o).flatten(); u2 = y - f2(o).flatten()
        loss = ((torch.where(u1 > 0, tau, 1 - tau) * u1 * u1).mean()
                + (torch.where(u2 > 0, tau, 1 - tau) * u2 * u2).mean())
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(list(f1.parameters()) + list(f2.parameters()), 1.0)
        opt.step()
        if i % 50 == 0:
            t1.load_state_dict(f1.state_dict()); t2.load_state_dict(f2.state_dict())
    return lambda s: torch.minimum(f1(s).flatten(), f2(s).flatten())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", default="bases")
    ap.add_argument("--nprobe", type=int, default=600)
    args = ap.parse_args()
    from scipy.stats import spearmanr
    sp = lambda a, b: spearmanr(a, b).statistic

    pA, sA = make_probes(args.nprobe, seed=99)
    pB, sB = make_air_probes(args.nprobe, seed=98)
    pC, sC = make_ground_probes(args.nprobe, seed=97)
    print("oracles...")
    oA, oB, oC = oracle_scores(sA), oracle_scores(sB), oracle_scores(sC)
    w2o, w2n = phase2_world_pairs(120_000, seed=7)

    rows = []
    for f in sorted(glob.glob(os.path.join(args.bases, "base_s*.pt"))):
        base = torch.load(f, weights_only=False)
        seed = int(f.rsplit("_s", 1)[1].split(".")[0])
        vd1 = mlp(OBS_DIM, 1); vd1.load_state_dict(base["vdag1"])
        vd2 = mlp(OBS_DIM, 1); vd2.load_state_dict(base["vdag2"])
        vdag = lambda s: torch.minimum(vd1(s).flatten(), vd2(s).flatten())
        ro, rn, rr = base["res_obs"], base["res_nxt"], base["res_rew"]
        anchors = ro[torch.randint(0, ro.shape[0], (4096,))]

        nets, ms = train_kernel(ro, rn, seed=seed)
        r = {"seed": seed, **{f"m_{k}": v for k, v in ms.items()}}
        with torch.no_grad():
            for tag, probes, orc in (("A", pA, oA), ("B", pB, oB), ("C", pC, oC)):
                r[f"vdag_{tag}"] = sp(vdag(probes).numpy(), orc)
                r[f"vdd_{tag}"] = sp(vdd2(nets, vdag, probes, anchors).numpy(), orc)

        # E4: fine-tune kernel + vdag control on record + world fragments
        ro4 = torch.cat([ro, w2o]); rn4 = torch.cat([rn, w2n])
        rr4 = torch.cat([rr, torch.zeros(w2o.shape[0])])
        nets4, ms4 = train_kernel(ro4, rn4, iters=1500, seed=seed,
                                  nets=(copy.deepcopy(nets[0]), copy.deepcopy(nets[1])))
        vdag4 = finetune_vdag(base, ro4, rn4, rr4, seed=seed)
        anchors4 = ro4[torch.randint(0, ro4.shape[0], (4096,))]
        with torch.no_grad():
            r["vdd_A_exposed"] = sp(vdd2(nets4, vdag, pA, anchors4).numpy(), oA)
            r["vdag_A_exposed"] = sp(vdag4(pA).numpy(), oA)
        rows.append(r)
        print(json.dumps({k: round(float(v), 3) for k, v in r.items()}))

    print("\nmean over seeds (Spearman rho vs oracle):")
    for k in [k for k in rows[0] if k != "seed" and not k.startswith("m_")]:
        vals = [x[k] for x in rows]
        print(f"  {k:16s} {np.mean(vals):+.3f}  (seeds: {' '.join(f'{v:+.2f}' for v in vals)})")


if __name__ == "__main__":
    main()
