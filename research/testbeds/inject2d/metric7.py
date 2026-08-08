"""Adversarial suite for the chart-epsilon operator. Attacks pre-registered:

A1 CHART CONFOUND (wind zone): hidden lateral accel in x>5; the chart pruned
   x (importance 0.13). Measure: displacement NLL inside vs outside the zone
   for charts trained on wind records; does chart importance of x recover
   (fail-safe adaptation) or does sigma_c stay confidently wrong?

A2 RESOURCE LAUNDERING (the strongest a-priori attack): boost is monotone-
   drain in the air, but Gaussian slack is symmetric -- the bootstrap max
   picks +fuel every hop and compounds hallucinated tank over the chain.
   Probes: ball high, car directly under, boost clearly insufficient
   (oracle-infeasible). Metric: hallucination index = mean z-scored value
   assigned to infeasible probes (z-scored within each net's own feasible-
   probe distribution). Also tested: the HULL fix -- perturbations drawn as
   eps-scaled WITNESSED displacement vectors from nearest chart cells
   (asymmetries preserved; no +boost-in-air exists in any record).

A3 TELEPORT POLLUTION (ablation): chart trained WITHOUT the respawn filter --
   how much ball-slack does it hallucinate (ball-dim sigma in free air)?
"""
import argparse, copy, glob, json, os
import numpy as np
import torch

from env import Aerial2D, OBS_DIM
from ladder import mlp
from foresight import obs_from_state
from metric4 import train_vddag, coherent
from metric5 import oracle2
from metric6 import ChartSigma, train_chart, train_vddag_chart, filter_teleports

GAMMA = 0.99


# ---------------- A2: infeasible-fuel probes + hallucination index ----------
def make_infeasible_probes(n, seed):
    rng = np.random.default_rng(seed)
    bz = rng.uniform(6.0, 7.4, n)
    x = rng.uniform(-8, 8, n)
    bx = np.clip(x + rng.uniform(-0.6, 0.6, n), -8, 8)   # directly underneath
    boost = rng.uniform(0.15, 0.40, n)                    # clearly short of the climb
    vx = rng.uniform(-1, 1, n)
    z = np.zeros(n); vz = np.zeros(n); og = np.ones(n, bool)
    bvz = np.zeros(n); t = np.full(n, 40)
    return (torch.from_numpy(obs_from_state(x, z, vx, vz, boost, og, bx, bz, bvz, t)),
            dict(x=x, z=z, vx=vx, vz=vz, boost=boost, bx=bx, bz=bz, bvz=bvz))


def make_feasible_ref(n, seed):
    rng = np.random.default_rng(seed)
    bz = rng.uniform(6.0, 7.4, n)
    x = rng.uniform(-8, 8, n)
    bx = np.clip(x + rng.uniform(-0.6, 0.6, n), -8, 8)
    boost = rng.uniform(0.9, 1.0, n)                      # full tank: feasible
    vx = rng.uniform(-1, 1, n)
    z = np.zeros(n); vz = np.zeros(n); og = np.ones(n, bool)
    bvz = np.zeros(n); t = np.full(n, 40)
    return (torch.from_numpy(obs_from_state(x, z, vx, vz, boost, og, bx, bz, bvz, t)),
            dict(x=x, z=z, vx=vx, vz=vz, boost=boost, bx=bx, bz=bz, bvz=bvz))


def hallucination_index(fn, p_inf, p_feas):
    with torch.no_grad():
        vi = fn(p_inf).numpy(); vf = fn(p_feas).numpy()
    return float((vi.mean() - vf.mean()) / (vf.std() + 1e-8))


# ---------------- the HULL fix: witnessed-displacement perturbations --------
def train_vddag_hull(ro, rn, rr, cs, eps, iters=4000, bs=1024, seed=0, K=8,
                     tau=0.75, nets=None, lr=1e-3, bank=65536, gate=0.0, dis_th=0.0):
    """Perturbations = eps-scaled REAL displacement vectors sampled from the
    record, matched by chart cell (nearest in chart space among a random
    bank). Slack spans only what like-dynamics states have actually done."""
    torch.manual_seed(seed)
    n = ro.shape[0]
    keep = filter_teleports(ro, rn)
    bo, bd = ro[keep], (rn - ro)[keep]
    bidx = torch.randint(0, bo.shape[0], (min(bank, bo.shape[0]),))
    bank_o, bank_d = bo[bidx], bd[bidx]
    with torch.no_grad():
        bank_c = cs.P(bank_o)                              # chart coords of bank
        # self-calibrating donor radius: 2x the bank's own median NN distance.
        # Thin regions get NO donors -> the operator degrades to plain V-dagger
        # exactly where the record cannot support optimism.
        samp = bank_c[torch.randint(0, bank_c.shape[0], (2048,))]
        dself = torch.cdist(samp, bank_c)
        dself.scatter_(1, dself.argmin(1, keepdim=True), 1e9)
        r0 = 2.0 * dself.min(1).values.median()
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
            qc = cs.P(nx)
            # K nearest bank entries by chart distance (sampled subset for speed)
            sub = torch.randint(0, bank_o.shape[0], (2048,))
            dist = torch.cdist(qc, bank_c[sub])
            nnd, nnk = dist.topk(K, largest=False)         # (bs, K)
            cand_v = torch.minimum(t1(nx).flatten(), t2(nx).flatten())
            for j in range(K):
                delta = bank_d[sub[nnk[:, j]]]
                delta = torch.where((nnd[:, j] < r0).unsqueeze(1), delta,
                                    torch.zeros_like(delta))
                if gate > 0:
                    mu, ls = cs(nx)
                    z = ((delta - mu) / ls.exp()).abs().max(1).values
                    delta = torch.where((z < gate).unsqueeze(1), delta,
                                        torch.zeros_like(delta))
                pert = coherent(nx + eps * delta)
                v1p, v2p = t1(pert).flatten(), t2(pert).flatten()
                vc = torch.minimum(v1p, v2p)
                if dis_th > 0:
                    # epistemic veto: no optimism where the twins disagree
                    ok = (v1p - v2p).abs() < dis_th * (vc.abs() + 1.0)
                    vc = torch.where(ok, vc, torch.full_like(vc, -1e9))
                cand_v = torch.maximum(cand_v, vc)
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


# ---------------- A1: wind-zone chart adaptation ----------------------------
def wind_record(n_steps, seed):
    from env import N_ACT
    env = Aerial2D(64, seed=seed)
    env.enable_wind()
    rng = np.random.default_rng(seed)
    obs_l, nxt_l = [], []
    o = env.obs()
    for t in range(n_steps // 64):
        m = env.action_mask()
        logits = rng.random((64, N_ACT)) * m + (m - 1) * 1e9
        _, d, info = env.step(logits.argmax(1))
        o2 = env.obs()
        nx = o2.copy()
        if d.any():
            nx[d] = info["final_obs"][d]
        obs_l.append(o.copy()); nxt_l.append(nx)
        o = o2
    return torch.from_numpy(np.concatenate(obs_l)), torch.from_numpy(np.concatenate(nxt_l))


def zone_nll(cs, ro, rn):
    keep = filter_teleports(ro, rn)
    o, nx = ro[keep], rn[keep]
    with torch.no_grad():
        mu, ls = cs(o)
        nll = (ls + 0.5 * ((nx - o - mu) / ls.exp()) ** 2).mean(1)
    inz = o[:, 0] * 10.0 > 5.0
    return float(nll[inz].mean()), float(nll[~inz].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", default="bases")
    ap.add_argument("--nprobe", type=int, default=500)
    args = ap.parse_args()

    p_inf, s_inf = make_infeasible_probes(args.nprobe, seed=93)
    p_fea, s_fea = make_feasible_ref(args.nprobe, seed=92)
    print("oracle check on the A2 probe sets...")
    o_inf = oracle2(s_inf, phase=1)
    o_fea = oracle2(s_fea, phase=1)
    print(f"  infeasible set: oracle feasible {np.mean(o_inf > 0):.0%} (want ~0)")
    print(f"  feasible set:   oracle feasible {np.mean(o_fea > 0):.0%} (want high)")

    print("\nA1: wind-zone chart adaptation (random-policy wind record)")
    wo, wn = wind_record(150_000, seed=11)
    cs_w, imp_w = train_chart(wo, wn, seed=0)
    names = ["x","z","vx","vz","boost","og","dx","dz","dist","bz","bvz","t"]
    print("  wind-chart importance:", " ".join(f"{n}:{v:.2f}" for n, v in
                                               zip(names, imp_w.tolist())))
    nz, oz = zone_nll(cs_w, wo, wn)
    print(f"  NLL in-zone {nz:.3f} vs out-zone {oz:.3f} (parity = adapted)")

    print("\nA2 + A3 per seed:")
    rows = []
    for f in sorted(glob.glob(os.path.join(args.bases, "base_s*.pt")))[:3]:
        base = torch.load(f, weights_only=False)
        seed = int(f.rsplit("_s", 1)[1].split(".")[0])
        ro, rn, rr = base["res_obs"], base["res_nxt"], base["res_rew"]
        vd1 = mlp(OBS_DIM, 1); vd1.load_state_dict(base["vdag1"])
        vd2 = mlp(OBS_DIM, 1); vd2.load_state_dict(base["vdag2"])
        vdag = lambda s: torch.minimum(vd1(s).flatten(), vd2(s).flatten())
        cs, _ = train_chart(ro, rn, seed=seed)
        cs_dirty, _ = train_chart(ro, rn, seed=seed, l1=1e-3)  # A3 uses unfiltered:
        # retrain WITHOUT teleport filter for the ablation
        torch.manual_seed(seed)
        cs_nf = ChartSigma(OBS_DIM)
        optn = torch.optim.Adam(cs_nf.parameters(), lr=1e-3)
        n = ro.shape[0]
        for i in range(3000):
            idx = torch.randint(0, n, (1024,))
            mu, ls = cs_nf(ro[idx])
            d = rn[idx] - ro[idx]
            nll = (ls + 0.5 * ((d - mu) / ls.exp()) ** 2).mean()
            optn.zero_grad(); nll.backward(); optn.step()
        # ball-dim slack in free air (car far from ball): should be ~0
        far = ro[(ro[:, 8] * 22.0 > 5.0)][:4096]
        with torch.no_grad():
            s_f = cs(far)[1].exp()[:, 9].mean()
            s_nf = cs_nf(far)[1].exp()[:, 9].mean()
        nets_g = train_vddag(ro, rn, rr, 1.0, seed=seed)
        nets_c = train_vddag_chart(ro, rn, rr, cs, 1.0, seed=seed)
        nets_h = train_vddag_hull(ro, rn, rr, cs, 1.0, seed=seed)
        fg = lambda s: torch.minimum(nets_g[0](s).flatten(), nets_g[1](s).flatten())
        fc = lambda s: torch.minimum(nets_c[0](s).flatten(), nets_c[1](s).flatten())
        fh = lambda s: torch.minimum(nets_h[0](s).flatten(), nets_h[1](s).flatten())
        r = {"seed": seed,
             "ball_slack_filtered": float(s_f), "ball_slack_unfiltered": float(s_nf),
             "halluc_vdag": hallucination_index(vdag, p_inf, p_fea),
             "halluc_eps_global": hallucination_index(fg, p_inf, p_fea),
             "halluc_eps_chart": hallucination_index(fc, p_inf, p_fea),
             "halluc_eps_hull": hallucination_index(fh, p_inf, p_fea)}
        rows.append(r)
        print(json.dumps({k: round(float(v), 3) for k, v in r.items()}))

    print("\nmeans:")
    for k in [k for k in rows[0] if k != "seed"]:
        vals = [x[k] for x in rows]
        print(f"  {k:22s} {np.mean(vals):+.3f}")


if __name__ == "__main__":
    main()
