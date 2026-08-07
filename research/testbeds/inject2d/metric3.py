"""Gamma-critic, iteration 3: distances get their own RECORD.

Iteration 2 failed because the metric's only real calibration was the 1-step
edge; the bootstrap's optimistic expectile then dragged every distance low
(unfalsifiable optimism, sign-flipped into distance space) and the readout
flattened. Fix: train d on OBSERVED k-step separations (s_t, s_{t+k}) from
trajectory-contiguous windows -- dense, falsifiable, multi-scale bounds, the
exact analogue of V-dagger's reward-grounded targets:

  observed   expectile( d(s_t, s_{t+k}), k, tau=0.35 )    k ~ logU[1,60]
             (optimistic LOWER envelope of path lengths actually driven;
              "maybe faster than we did it", never "shortcut never seen")
  self       d(s,s) = 0
  bootstrap  d(s,a) <= 1 + max-twin d(s',a), weight 0.2, tau=0.45
             (range extension only, no longer the main signal)
  floor      mild anti-collapse on random pairs

Twins, max-in-bootstrap-target. Readout: top-k soft reach (vdd2).
E1-E4 as before; E4 fine-tunes on phase-1 windows + random-policy phase-2
windows (world witnessed, conduct never performed).
"""
import argparse, copy, glob, json, os
import numpy as np
import torch
import torch.nn as nn

from env import Aerial2D, N_ACT, OBS_DIM
from ladder import mlp
from foresight import make_probes, oracle_scores
from metric import make_air_probes, make_ground_probes
from metric2 import DNet, expectile, vdd2, finetune_vdag

GAMMA = 0.99
DCAP = 120.0
KMAX = 60


class TrajStore:
    """(W,T,N,D) windows -> contiguous per-env timelines with done-safe k-step sampling."""
    def __init__(self, traj_obs, traj_done):
        W, T, N, D = traj_obs.shape
        self.obs = traj_obs.reshape(W * T, N, D)
        done = traj_done.reshape(W * T, N)
        self.cd = torch.cat([torch.zeros(1, N), done.cumsum(0)], 0)
        self.L, self.N = W * T, N

    def sample(self, bs, rng):
        k = torch.from_numpy(np.exp(rng.uniform(0, np.log(KMAX), bs)).astype(np.int64)).clamp(1, KMAX)
        t = torch.from_numpy(rng.integers(0, self.L - KMAX - 1, bs))
        n = torch.from_numpy(rng.integers(0, self.N, bs))
        ok = (self.cd[t + k, n] - self.cd[t, n]) == 0
        t, n, k = t[ok], n[ok], k[ok]
        return self.obs[t, n], self.obs[t + k, n], k.float()


def train_kernel3(stores, pair_obs, iters=4000, bs=1024, seed=0, nets=None,
                  tau_obs=0.35, w_boot=0.2, w_floor=2.0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if nets is None:
        d1, d2 = DNet(OBS_DIM), DNet(OBS_DIM)
    else:
        d1, d2 = nets
    t1, t2 = copy.deepcopy(d1), copy.deepcopy(d2)
    opt = torch.optim.Adam(list(d1.parameters()) + list(d2.parameters()), lr=1e-3)
    npair = pair_obs.shape[0]
    for i in range(iters):
        st = stores[i % len(stores)]
        o, ok, kk = st.sample(bs, rng)
        a = pair_obs[torch.randint(0, npair, (o.shape[0],))]
        with torch.no_grad():
            tboot = (1.0 + torch.maximum(t1(ok, a), t2(ok, a))).clamp(max=DCAP)
        loss = 0.0
        for dn in (d1, d2):
            loss = loss + expectile(dn(o, ok), kk, tau_obs)          # observed record
            loss = loss + (dn(o, o) ** 2).mean()                     # self
            loss = loss + w_boot * expectile(dn(o, a), tboot, 0.45)  # composition
            gam = GAMMA ** dn(o, a).clamp(max=DCAP)
            loss = loss + w_floor * (gam.mean() - 0.05) ** 2         # anti-collapse
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(list(d1.parameters()) + list(d2.parameters()), 1.0)
        opt.step()
        if i % 50 == 0:
            t1.load_state_dict(d1.state_dict()); t2.load_state_dict(d2.state_dict())
    with torch.no_grad():
        o, ok, kk = stores[0].sample(4096, rng)
        dmax = lambda s, a: torch.maximum(d1(s, a), d2(s, a))
        j = torch.randint(0, npair, (o.shape[0],))
        stats = {"obs_ratio": float((dmax(o, ok) / kk).median()),
                 "selfd": float(dmax(o, o).mean()),
                 "rand": float(dmax(o, pair_obs[j]).mean())}
    return (d1, d2), stats


def phase2_world_windows(n_steps, seed):
    env = Aerial2D(64, seed=seed)
    env.set_phase(2)
    rng = np.random.default_rng(seed)
    T = n_steps // 64
    obs = torch.zeros(T, 64, OBS_DIM); done = torch.zeros(T, 64)
    o = env.obs()
    for t in range(T):
        m = env.action_mask()
        logits = rng.random((64, N_ACT)) * m + (m - 1) * 1e9
        _, d, info = env.step(logits.argmax(1))
        obs[t] = torch.from_numpy(o); done[t] = torch.from_numpy(d.astype(np.float32))
        o = env.obs()
    return obs.unsqueeze(0), done.unsqueeze(0)   # (1, T, N, D)


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
    w_obs, w_done = phase2_world_windows(120_000, seed=7)
    wstore = TrajStore(w_obs, w_done)
    w_flat = w_obs.reshape(-1, OBS_DIM)

    rows = []
    for f in sorted(glob.glob(os.path.join(args.bases, "base_s*.pt"))):
        base = torch.load(f, weights_only=False)
        seed = int(f.rsplit("_s", 1)[1].split(".")[0])
        vd1 = mlp(OBS_DIM, 1); vd1.load_state_dict(base["vdag1"])
        vd2 = mlp(OBS_DIM, 1); vd2.load_state_dict(base["vdag2"])
        vdag = lambda s: torch.minimum(vd1(s).flatten(), vd2(s).flatten())
        ro, rn, rr = base["res_obs"], base["res_nxt"], base["res_rew"]
        store = TrajStore(base["traj_obs"], base["traj_done"])
        anchors = ro[torch.randint(0, ro.shape[0], (4096,))]

        nets, ms = train_kernel3([store], ro, seed=seed)
        r = {"seed": seed, **{f"m_{k}": v for k, v in ms.items()}}
        with torch.no_grad():
            for tag, probes, orc in (("A", pA, oA), ("B", pB, oB), ("C", pC, oC)):
                r[f"vdag_{tag}"] = sp(vdag(probes).numpy(), orc)
                r[f"vdd_{tag}"] = sp(vdd2(nets, vdag, probes, anchors).numpy(), orc)

        nets4, ms4 = train_kernel3([store, wstore], ro, iters=1500, seed=seed,
                                   nets=(copy.deepcopy(nets[0]), copy.deepcopy(nets[1])))
        ro4 = torch.cat([ro, w_flat[torch.randint(0, w_flat.shape[0], (40000,))]])
        anchors4 = ro4[torch.randint(0, ro4.shape[0], (4096,))]
        vdag4 = finetune_vdag(base,
                              torch.cat([ro, w_obs[0, :-1].reshape(-1, OBS_DIM)]),
                              torch.cat([rn, w_obs[0, 1:].reshape(-1, OBS_DIM)]),
                              torch.cat([rr, torch.zeros((w_obs.shape[1] - 1) * 64)]),
                              seed=seed)
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
