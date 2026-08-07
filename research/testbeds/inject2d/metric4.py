"""V-double-dagger, iteration 4: the epsilon-inflated Bellman operator.

    y = r + gamma * max_{a in B_eps(s')} minTwin V(a),   B_eps = real next state
    perturbed by eps * sigma_emp, sigma_emp = per-dim std of executed one-step
    displacements over the whole record (ONE GLOBAL VECTOR -- no pointwise net,
    nothing to extrapolate, cannot be wrong off-manifold because it is not a
    function of state).

One-parameter unification: eps=0 IS the composition critic; eps>0 grants the
dynamics one empirical ellipsoid of slack per decision -- the geometric rung's
honest residue. Still a gamma-contraction; optimism is falsifiable per update;
axes the record has never seen move (ball-fall in phase 1) have sigma_emp = 0,
so the geometry opens exactly where the record licenses it and nowhere else.

Same E1-E4 protocol as metric.py / metric3.py.
"""
import argparse, copy, glob, json, os
import numpy as np
import torch

from env import OBS_DIM, EP_LEN, X_LIM, Z_CEIL, VX_MAX
from ladder import mlp
from foresight import make_probes, oracle_scores, obs_from_state
from metric import make_air_probes, make_ground_probes
from metric3 import phase2_world_windows

GAMMA = 0.99


def coherent(o):
    """Recompute derived obs dims (on_ground, dx/dz-consistent dist) after jitter."""
    x = o[:, 0] * X_LIM; z = (o[:, 1] * Z_CEIL).clamp(min=0)
    vx = o[:, 2] * VX_MAX; vz = o[:, 3] * 12.0
    boost = o[:, 4].clamp(0, 1)
    bz = (o[:, 9] * Z_CEIL).clamp(min=0.2); bvz = o[:, 10] * 10.0
    dx = o[:, 6] * 20.0
    og = (z <= 1e-6).float()
    vz = vz * (1 - og)
    dz = bz - z
    dist = torch.sqrt(dx ** 2 + dz ** 2 + 1e-8)
    t = o[:, 11].clamp(0, 1)
    return torch.stack([x / X_LIM, z / Z_CEIL, vx / VX_MAX, vz / 12.0, boost, og,
                        dx / 20.0, dz / Z_CEIL, dist / 22.0, bz / Z_CEIL,
                        bvz / 10.0, t], 1)


def train_vddag(ro, rn, rr, eps, iters=4000, bs=1024, seed=0, K=8, tau=0.75,
                nets=None, lr=1e-3):
    torch.manual_seed(seed)
    n = ro.shape[0]
    sig = (rn - ro).std(0)                      # the empirical displacement scale
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
            cand = [nx]
            for _ in range(K):
                cand.append(coherent(nx + eps * sig * torch.randn_like(nx)))
            vals = torch.stack([torch.minimum(t1(c).flatten(), t2(c).flatten())
                                for c in cand])
            y = (r + GAMMA * vals.max(0).values).clamp(-5.0, 50.0)
        u1 = y - f1(o).flatten(); u2 = y - f2(o).flatten()
        loss = ((torch.where(u1 > 0, tau, 1 - tau) * u1 * u1).mean()
                + (torch.where(u2 > 0, tau, 1 - tau) * u2 * u2).mean())
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(f1.parameters()) + list(f2.parameters()), 1.0)
        opt.step()
        if i % 50 == 0:
            t1.load_state_dict(f1.state_dict()); t2.load_state_dict(f2.state_dict())
    return (f1, f2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", default="bases")
    ap.add_argument("--nprobe", type=int, default=600)
    ap.add_argument("--eps", type=float, nargs="+", default=[1.0, 2.0])
    args = ap.parse_args()
    from scipy.stats import spearmanr
    sp = lambda a, b: spearmanr(a, b).statistic

    pA, sA = make_probes(args.nprobe, seed=99)
    pB, sB = make_air_probes(args.nprobe, seed=98)
    pC, sC = make_ground_probes(args.nprobe, seed=97)
    print("oracles...")
    oA, oB, oC = oracle_scores(sA), oracle_scores(sB), oracle_scores(sC)
    w_obs, w_done = phase2_world_windows(120_000, seed=7)
    w_o = w_obs[0, :-1].reshape(-1, OBS_DIM); w_n = w_obs[0, 1:].reshape(-1, OBS_DIM)
    keep = (w_done[0, :-1].reshape(-1) < 0.5)
    w_o, w_n = w_o[keep], w_n[keep]

    rows = []
    for f in sorted(glob.glob(os.path.join(args.bases, "base_s*.pt"))):
        base = torch.load(f, weights_only=False)
        seed = int(f.rsplit("_s", 1)[1].split(".")[0])
        vd1 = mlp(OBS_DIM, 1); vd1.load_state_dict(base["vdag1"])
        vd2 = mlp(OBS_DIM, 1); vd2.load_state_dict(base["vdag2"])
        vdag = lambda s: torch.minimum(vd1(s).flatten(), vd2(s).flatten())
        ro, rn, rr = base["res_obs"], base["res_nxt"], base["res_rew"]

        r = {"seed": seed}
        with torch.no_grad():
            for tag, probes, orc in (("A", pA, oA), ("B", pB, oB), ("C", pC, oC)):
                r[f"vdag_{tag}"] = sp(vdag(probes).numpy(), orc)

        for eps in args.eps:
            nets = train_vddag(ro, rn, rr, eps, seed=seed)
            fn = lambda s: torch.minimum(nets[0](s).flatten(), nets[1](s).flatten())
            with torch.no_grad():
                for tag, probes, orc in (("A", pA, oA), ("B", pB, oB), ("C", pC, oC)):
                    r[f"e{eps:g}_{tag}"] = sp(fn(probes).numpy(), orc)
            # E4: fine-tune on record + world fragments (conduct never performed)
            ro4 = torch.cat([ro, w_o]); rn4 = torch.cat([rn, w_n])
            rr4 = torch.cat([rr, torch.zeros(w_o.shape[0])])
            nets4 = train_vddag(ro4, rn4, rr4, eps, iters=1500, seed=seed,
                                nets=(copy.deepcopy(nets[0]), copy.deepcopy(nets[1])),
                                lr=3e-4)
            fn4 = lambda s: torch.minimum(nets4[0](s).flatten(), nets4[1](s).flatten())
            with torch.no_grad():
                r[f"e{eps:g}_A_exposed"] = sp(fn4(pA).numpy(), oA)
        # vdag exposure control (eps=0 fine-tune)
        nets0 = train_vddag(torch.cat([ro, w_o]), torch.cat([rn, w_n]),
                            torch.cat([rr, torch.zeros(w_o.shape[0])]), 0.0,
                            iters=1500, seed=seed,
                            nets=(copy.deepcopy(vd1), copy.deepcopy(vd2)), lr=3e-4)
        fn0 = lambda s: torch.minimum(nets0[0](s).flatten(), nets0[1](s).flatten())
        with torch.no_grad():
            r["vdag_A_exposed"] = sp(fn0(pA).numpy(), oA)
        rows.append(r)
        print(json.dumps({k: round(float(v), 3) for k, v in r.items()}))

    print("\nmean over seeds (Spearman rho vs oracle):")
    for k in [k for k in rows[0] if k != "seed"]:
        vals = [x[k] for x in rows]
        print(f"  {k:16s} {np.mean(vals):+.3f}  (seeds: {' '.join(f'{v:+.2f}' for v in vals)})")


if __name__ == "__main__":
    main()
