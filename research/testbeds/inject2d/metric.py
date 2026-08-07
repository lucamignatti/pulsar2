"""V-double-dagger: the reachability-kernel critic (unification candidate).

    V'' (s) = max_a  gamma^{d(s,a)} * Vdag(a)

d = learned QUASIMETRIC (decisions-to-reach), triangle inequality BY
ARCHITECTURE:  d(s,a) = sum_i w_i * relu(f_i(a) - f_i(s)),  w_i >= 0.
Each component (h(a)-h(s))_+ is a quasimetric; nonneg sums preserve it; so
distances through unvisited states are bounded by compositions of witnessed
legs -- structural generalization, no pointwise coefficients (the measured
failure mode of the Sigma-HJB rung).

Training (QRL-style Lagrangian, executed one-step pairs only):
    maximize E_random-pairs[ d(s,a) ]   (up to a cap)
    subject to d(s,s') <= 1 on executed transitions.

Readout: anchors = reservoir sample with Vdag values; V''(s) includes a=s so
V'' >= Vdag pointwise -- on-manifold correctness is structural.

Experiments (pre-registered):
  E1 falling-ball probes, zero exposure: bar = Vdag's +0.29
  E2 airborne-car probes (unseen car states, witnessed physics): expect > Vdag
  E3 in-distribution ground probes: V'' must track Vdag (correctness)
  E4 fragment exposure: retrain d on +random-policy phase-2 WORLD data (ball
     falls witnessed, conduct never performed) -> foresight must jump; Vdag
     retrained on the same extra data is the control
"""
import argparse, glob, json, os
import numpy as np
import torch
import torch.nn as nn

from env import Aerial2D, N_ACT, OBS_DIM, EP_LEN, X_LIM, Z_CEIL, VX_MAX
from ladder import mlp, expectile_loss
from foresight import (obs_from_state, make_probes, oracle_scores)

GAMMA = 0.99
DCAP = 120.0     # distance cap (decisions); gamma^120 ~ 0.30


class Quasimetric(nn.Module):
    def __init__(self, obs_dim, k=64, h=128):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(obs_dim, h), nn.Tanh(),
                               nn.Linear(h, h), nn.Tanh(), nn.Linear(h, k))
        self.logw = nn.Parameter(torch.zeros(k))

    def forward(self, s, a):
        w = nn.functional.softplus(self.logw)
        return (torch.relu(self.f(a) - self.f(s)) * w).sum(-1)


def train_metric(pairs_obs, pairs_nxt, iters=3000, bs=1024, seed=0, lam=30.0):
    torch.manual_seed(seed)
    n = pairs_obs.shape[0]
    dnet = Quasimetric(OBS_DIM)
    opt = torch.optim.Adam(dnet.parameters(), lr=1e-3)
    for i in range(iters):
        idx = torch.randint(0, n, (bs,))
        o, nx = pairs_obs[idx], pairs_nxt[idx]
        # constraint: executed transitions are one decision apart
        d_edge = dnet(o, nx)
        c_loss = (torch.relu(d_edge - 1.0) ** 2).mean()
        # spread: random ordered pairs as far as the constraints allow
        j = torch.randint(0, n, (bs,))
        d_rand = dnet(o, pairs_obs[j])
        s_loss = -d_rand.clamp(max=DCAP).mean() / DCAP
        loss = lam * c_loss + s_loss
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(dnet.parameters(), 1.0)
        opt.step()
    with torch.no_grad():
        idx = torch.randint(0, n, (4096,))
        stats = {
            "edge_mean": float(dnet(pairs_obs[idx], pairs_nxt[idx]).mean()),
            "edge_viol": float((dnet(pairs_obs[idx], pairs_nxt[idx]) > 1.5).float().mean()),
            "rand_mean": float(dnet(pairs_obs[idx],
                                    pairs_obs[torch.randint(0, n, (4096,))]).mean()),
        }
    return dnet, stats


@torch.no_grad()
def vdd(dnet, vdag_fn, probes, anchors, chunk=128):
    """V''(s) = max( Vdag(s), max_a gamma^d(s,a) Vdag(a) ) over anchor sample."""
    va = vdag_fn(anchors).clamp(min=0)          # negative anchors can't help a max
    out = torch.empty(probes.shape[0])
    for i0 in range(0, probes.shape[0], chunk):
        p = probes[i0:i0 + chunk]
        d = dnet(p.unsqueeze(1).expand(-1, anchors.shape[0], -1).reshape(-1, OBS_DIM),
                 anchors.unsqueeze(0).expand(p.shape[0], -1, -1).reshape(-1, OBS_DIM))
        d = d.view(p.shape[0], anchors.shape[0]).clamp(max=DCAP)
        reach = (GAMMA ** d) * va.unsqueeze(0)
        out[i0:i0 + chunk] = torch.maximum(reach.max(1).values, vdag_fn(p))
    return out


def make_air_probes(n, seed):
    """E2: airborne car states (off the visited jump-arc manifold), static high balls."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-X_LIM, X_LIM, n)
    z = rng.uniform(1.0, 6.0, n)
    vx = rng.uniform(-4, 4, n)
    vz = rng.uniform(-2, 8, n)
    boost = rng.uniform(0.1, 1.0, n)
    og = np.zeros(n, bool)
    bx = rng.uniform(-8, 8, n)
    bz = rng.uniform(5.0, 8.0, n)
    bvz = np.zeros(n)
    t = np.full(n, 40)
    obs = obs_from_state(x, z, vx, vz, boost, og, bx, bz, bvz, t)
    return torch.from_numpy(obs), dict(x=x, z=z, vx=vx, vz=vz, boost=boost,
                                       bx=bx, bz=bz, bvz=bvz)


def make_ground_probes(n, seed):
    """E3: in-distribution phase-1 states (correctness check)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-X_LIM, X_LIM, n)
    vx = rng.uniform(-4, 4, n)
    boost = rng.uniform(0.15, 1.0, n)
    bx = rng.uniform(-8, 8, n)
    bz = rng.uniform(0.5, 8.0, n)
    bvz = np.zeros(n)
    z = np.zeros(n); vz = np.zeros(n); og = np.ones(n, bool); t = np.full(n, 40)
    obs = obs_from_state(x, z, vx, vz, boost, og, bx, bz, bvz, t)
    return torch.from_numpy(obs), dict(x=x, z=z, vx=vx, vz=vz, boost=boost,
                                       bx=bx, bz=bz, bvz=bvz)


def phase2_world_pairs(n_steps, seed):
    """E4: random-policy phase-2 rollouts -- the WORLD witnessed (balls fall),
    the conduct never performed. Returns executed (s, s') pairs."""
    env = Aerial2D(64, seed=seed)
    env.set_phase(2)
    rng = np.random.default_rng(seed)
    obs_l, nxt_l = [], []
    o = env.obs()
    for t in range(n_steps // 64):
        m = env.action_mask()
        logits = rng.random((64, N_ACT)) * m + (m - 1) * 1e9
        a = logits.argmax(1)
        _, d, info = env.step(a)
        o2 = env.obs()
        nx = o2.copy()
        if d.any():
            nx[d] = info["final_obs"][d]
        obs_l.append(o.copy()); nxt_l.append(nx)
        o = o2
    return (torch.from_numpy(np.concatenate(obs_l)),
            torch.from_numpy(np.concatenate(nxt_l)))


def retrain_vdag(ro, rn, rr, iters=3000, bs=1024, seed=0, tau=0.75):
    """Control for E4: Vdag retrained on the SAME augmented pair set.
    Extra world pairs carry their (zero-touch) rewards."""
    import copy as _copy
    torch.manual_seed(seed)
    n = ro.shape[0]
    f1, f2 = mlp(OBS_DIM, 1), mlp(OBS_DIM, 1)
    t1, t2 = _copy.deepcopy(f1), _copy.deepcopy(f2)
    opt = torch.optim.Adam(list(f1.parameters()) + list(f2.parameters()), lr=1e-3)
    for i in range(iters):
        idx = torch.randint(0, n, (bs,))
        o, nx, r = ro[idx], rn[idx], rr[idx]
        with torch.no_grad():
            y = (r + GAMMA * torch.minimum(t1(nx).flatten(), t2(nx).flatten())
                 ).clamp(-5.0, 50.0)
        loss = (expectile_loss(f1(o).flatten(), y, tau)
                + expectile_loss(f2(o).flatten(), y, tau))
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

    # probe families + oracles (shared across seeds)
    pA, sA = make_probes(args.nprobe, seed=99)            # E1/E4: falling balls
    pB, sB = make_air_probes(args.nprobe, seed=98)        # E2: airborne car
    pC, sC = make_ground_probes(args.nprobe, seed=97)     # E3: in-distribution
    print("oracles (scripted intercepts in real physics)...")
    oA = oracle_scores(sA)
    oB = oracle_scores(sB)
    oC = oracle_scores(sC)
    print(f"  A falling-ball  feasible {np.mean(oA > 0):.0%}")
    print(f"  B airborne-car  feasible {np.mean(oB > 0):.0%}")
    print(f"  C ground        feasible {np.mean(oC > 0):.0%}")

    w2o, w2n = phase2_world_pairs(120_000, seed=7)        # E4 exposure data

    rows = []
    for f in sorted(glob.glob(os.path.join(args.bases, "base_s*.pt"))):
        base = torch.load(f, weights_only=False)
        seed = int(f.rsplit("_s", 1)[1].split(".")[0])
        vd1 = mlp(OBS_DIM, 1); vd1.load_state_dict(base["vdag1"])
        vd2 = mlp(OBS_DIM, 1); vd2.load_state_dict(base["vdag2"])
        vdag = lambda s: torch.minimum(vd1(s).flatten(), vd2(s).flatten())
        ro, rn, rr = base["res_obs"], base["res_nxt"], base["res_rew"]
        anchors = ro[torch.randint(0, ro.shape[0], (4096,))]

        dnet, ms = train_metric(ro, rn, seed=seed)
        with torch.no_grad():
            r = {"seed": seed, **{f"m_{k}": v for k, v in ms.items()}}
            for tag, probes, orc in (("A", pA, oA), ("B", pB, oB), ("C", pC, oC)):
                r[f"vdag_{tag}"] = sp(vdag(probes).numpy(), orc)
                r[f"vdd_{tag}"] = sp(vdd(dnet, vdag, probes, anchors).numpy(), orc)

        # E4: world exposure (fragments, no conduct) -- metric AND vdag control
        ro4 = torch.cat([ro, w2o]); rn4 = torch.cat([rn, w2n])
        rr4 = torch.cat([rr, torch.zeros(w2o.shape[0])])
        dnet4, ms4 = train_metric(ro4, rn4, seed=seed)
        vdag4 = retrain_vdag(ro4, rn4, rr4, seed=seed)
        anchors4 = ro4[torch.randint(0, ro4.shape[0], (4096,))]
        with torch.no_grad():
            r["vdd_A_exposed"] = sp(vdd(dnet4, vdag, pA, anchors4).numpy(), oA)
            r["vdag_A_exposed"] = sp(vdag4(pA).numpy(), oA)
            r["m4_edge_viol"] = ms4["edge_viol"]
        rows.append(r)
        print(json.dumps({k: round(float(v), 3) for k, v in r.items()}))

    print("\nmean over seeds (Spearman rho vs oracle):")
    keys = [k for k in rows[0] if k != "seed" and not k.startswith("m")]
    for k in keys:
        vals = [x[k] for x in rows]
        print(f"  {k:16s} {np.mean(vals):+.3f}  (seeds: {' '.join(f'{v:+.2f}' for v in vals)})")


if __name__ == "__main__":
    main()
