"""Chart-conditioned epsilon operator: slack from the LOCAL PHYSICS, not from
a global constant and not from pointwise state extrapolation.

    sigma_c(s) = g(P s),  P = L1-sparsified linear chart (inspectable),
    g = small MLP -> per-dim displacement (mu, log sigma)

trained on executed displacements with TELEPORT transitions filtered (ball
respawns would otherwise teach 'ball teleportation slack' -- a known attack).
The claim under test: sigma_c is valid EVERYWHERE because its input domain
(dynamics-relevant features: airborne, boost, velocity bands) saturates in the
first minutes of play -- generalization bounded by CHART support, not state
support. The eps-operator then composes never-assembled conducts from
measured legs.

Family F (the flick analogue): descending touches -- car above ball, falling,
low boost. Every leg witnessed; the assembly rare (rarity measured from the
record). This is 'all the geometry to imply it is possible is there'.
"""
import argparse, copy, glob, json, os
import numpy as np
import torch
import torch.nn as nn

from env import OBS_DIM
from ladder import mlp
from foresight import make_probes, obs_from_state
from metric import make_air_probes, make_ground_probes
from metric4 import train_vddag, coherent
from metric5 import oracle2

GAMMA = 0.99


class ChartSigma(nn.Module):
    def __init__(self, obs_dim, k=4, h=64):
        super().__init__()
        self.P = nn.Linear(obs_dim, k, bias=False)
        self.g = nn.Sequential(nn.Tanh(), nn.Linear(k, h), nn.Tanh(),
                               nn.Linear(h, 2 * obs_dim))

    def forward(self, s):
        out = self.g(self.P(s))
        return out[:, :s.shape[1]], out[:, s.shape[1]:].clamp(-7, 2)


def filter_teleports(ro, rn):
    """Ball respawn transitions: huge ball-dim displacement. Filter them from
    the displacement fit (they are env bookkeeping, not physics)."""
    bd = (rn[:, 9] - ro[:, 9]).abs() + (rn[:, 6] - ro[:, 6]).abs()
    thr = 5 * bd.median() + 0.05
    return bd < thr


def train_chart(ro, rn, iters=3000, bs=1024, seed=0, l1=1e-3):
    torch.manual_seed(seed)
    keep = filter_teleports(ro, rn)
    o, nx = ro[keep], rn[keep]
    n = o.shape[0]
    cs = ChartSigma(OBS_DIM)
    opt = torch.optim.Adam(cs.parameters(), lr=1e-3)
    for i in range(iters):
        idx = torch.randint(0, n, (bs,))
        mu, ls = cs(o[idx])
        d = nx[idx] - o[idx]
        nll = (ls + 0.5 * ((d - mu) / ls.exp()) ** 2).mean()
        loss = nll + l1 * cs.P.weight.abs().mean()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(cs.parameters(), 1.0)
        opt.step()
    with torch.no_grad():
        imp = cs.P.weight.abs().sum(0)  # per-obs-dim chart importance
    return cs, imp / (imp.max() + 1e-8)


def train_vddag_chart(ro, rn, rr, cs, eps, iters=4000, bs=1024, seed=0, K=8,
                      tau=0.75, nets=None, lr=1e-3):
    """eps-operator with chart-conditional slack sigma_c(s')."""
    torch.manual_seed(seed)
    n = ro.shape[0]
    if nets is None:
        f1, f2 = mlp(OBS_DIM, 1), mlp(OBS_DIM, 1)
    else:
        f1, f2 = nets
    t1, t2 = copy.deepcopy(f1), copy.deepcopy(f2)
    opt = torch.optim.Adam(list(f1.parameters()) + list(f2.parameters()), lr=lr)
    for i in range(iters):
        idx = torch.randint(0, n, (bs,))
        o, nx, r = ro[idx], rn[idx], rr[idx]
        with torch.no_grad():
            _, ls = cs(nx)
            sig = ls.exp()
            cand_v = torch.minimum(t1(nx).flatten(), t2(nx).flatten())
            for _ in range(K):
                pert = coherent(nx + eps * sig * torch.randn_like(nx))
                cand_v = torch.maximum(cand_v, torch.minimum(
                    t1(pert).flatten(), t2(pert).flatten()))
            y = (r + GAMMA * cand_v).clamp(-5.0, 50.0)
        u1 = y - f1(o).flatten(); u2 = y - f2(o).flatten()
        loss = ((torch.where(u1 > 0, tau, 1 - tau) * u1 * u1).mean()
                + (torch.where(u2 > 0, tau, 1 - tau) * u2 * u2).mean())
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(f1.parameters()) + list(f2.parameters()), 1.0)
        opt.step()
        if i % 50 == 0:
            t1.load_state_dict(f1.state_dict()); t2.load_state_dict(f2.state_dict())
    return (f1, f2)


def make_desc_probes(n, seed):
    """Family F: descending touch. Car ABOVE the ball, falling, low boost --
    feasible iff lateral drift aligns during the descent window. All legs
    (falling flight, drift, touch) witnessed; the assembly rare."""
    rng = np.random.default_rng(seed)
    bz = rng.uniform(2.5, 5.0, n)
    z = bz + rng.uniform(1.5, 4.0, n)
    x = rng.uniform(-9, 9, n)
    bx = np.clip(x + rng.uniform(-4, 4, n), -8, 8)
    vx = rng.uniform(-4, 4, n)
    vz = rng.uniform(-3.0, -0.5, n)
    boost = rng.uniform(0.0, 0.15, n)
    og = np.zeros(n, bool)
    bvz = np.zeros(n)
    t = np.full(n, 40)
    return (torch.from_numpy(obs_from_state(x, np.minimum(z, 9.5), vx, vz, boost,
                                            og, bx, bz, bvz, t)),
            dict(x=x, z=np.minimum(z, 9.5), vx=vx, vz=vz, boost=boost,
                 bx=bx, bz=bz, bvz=bvz))


def desc_rarity(traj_obs, traj_done):
    """How often does the record contain descending touches? (car above ball,
    vz<-1 at contact). Uses dist dim crossing the touch radius."""
    W, T, N, D = traj_obs.shape
    o = traj_obs.reshape(-1, D)
    near = o[:, 8] * 22.0 <= 0.95
    desc = ((o[:, 3] * 12.0 < -1.0) & (o[:, 7] < -0.15)
            & (o[:, 4] < 0.15))   # falling, WELL above ball, near-empty tank
    total = int(near.sum())
    return int((near & desc).sum()), total


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
    pF, sF = make_desc_probes(args.nprobe, seed=94)
    print("oracles (corrected, phase-explicit)...")
    oA = oracle2(sA, phase=2)
    oB = oracle2(sB, phase=1)
    oC = oracle2(sC, phase=1)
    oF = oracle2(sF, phase=1)
    print(f"  A {np.mean(oA>0):.0%}  B {np.mean(oB>0):.0%}  C {np.mean(oC>0):.0%}"
          f"  F(desc) {np.mean(oF>0):.0%}")

    rows = []
    for f in sorted(glob.glob(os.path.join(args.bases, "base_s*.pt"))):
        base = torch.load(f, weights_only=False)
        seed = int(f.rsplit("_s", 1)[1].split(".")[0])
        vd1 = mlp(OBS_DIM, 1); vd1.load_state_dict(base["vdag1"])
        vd2 = mlp(OBS_DIM, 1); vd2.load_state_dict(base["vdag2"])
        vdag = lambda s: torch.minimum(vd1(s).flatten(), vd2(s).flatten())
        ro, rn, rr = base["res_obs"], base["res_nxt"], base["res_rew"]
        nd, nt = desc_rarity(base["traj_obs"], base["traj_done"])
        r = {"seed": seed, "desc_near_frac": nd / max(nt, 1)}

        cs, imp = train_chart(ro, rn, seed=seed)
        if seed == 0:
            names = ["x","z","vx","vz","boost","og","dx","dz","dist","bz","bvz","t"]
            print("chart importance:", " ".join(f"{n}:{v:.2f}" for n, v in
                                                zip(names, imp.tolist())))
        # global-eps control and chart-eps candidate, same seeds/iters
        nets_g = train_vddag(ro, rn, rr, 1.0, seed=seed)
        nets_c = train_vddag_chart(ro, rn, rr, cs, 1.0, seed=seed)
        fg = lambda s: torch.minimum(nets_g[0](s).flatten(), nets_g[1](s).flatten())
        fc = lambda s: torch.minimum(nets_c[0](s).flatten(), nets_c[1](s).flatten())
        with torch.no_grad():
            for tag, probes, orc in (("A", pA, oA), ("B", pB, oB),
                                     ("C", pC, oC), ("F", pF, oF)):
                r[f"vdag_{tag}"] = sp(vdag(probes).numpy(), orc)
                r[f"eg_{tag}"] = sp(fg(probes).numpy(), orc)
                r[f"ec_{tag}"] = sp(fc(probes).numpy(), orc)
        rows.append(r)
        print(json.dumps({k: round(float(v), 3) for k, v in r.items()}))

    print("\nmean over seeds (Spearman rho vs oracle):")
    for k in [k for k in rows[0] if k != "seed"]:
        vals = [x[k] for x in rows]
        print(f"  {k:14s} {np.mean(vals):+.3f}  (seeds: {' '.join(f'{v:+.2f}' for v in vals)})")


if __name__ == "__main__":
    main()
