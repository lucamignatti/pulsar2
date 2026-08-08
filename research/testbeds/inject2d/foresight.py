"""Foresight probe: can the geometric critic price plays it has never seen?

Base: phase-1-only training (static balls). Probes: 600 phase-2 states (balls
mid-fall, bvz < 0) -- a conduct class (timed intercept) absent from the record.
Oracle: best of 18 scripted intercept maneuvers rolled in the real env physics.

Per seed, the geo trio is retrained OFFLINE from the same frozen reservoir
three ways (only where the HJB equation is ENFORCED differs; Sigma and r-hat
always fit on real executed pairs only -- they are measurements):

  onpolicy    residual on reservoir (visited-manifold) states only [current design]
  colloc      residual on 50% reservoir / 30% physically-perturbed reservoir
              (car pose jitter, ball lifted, bvz ~ U[-6,0]) / 20% domain samples
  colloc_sym  colloc + x-mirror symmetry augmentation on Sigma/r-hat/field
              (the environment's physics is mirror-invariant; the record isn't)

Scored: Spearman rho(estimate, oracle) on the probe set, per rung. V and Vdag
are the record-rung controls; rho(geo, V) is the tracks-habit check.
"""
import argparse, glob, json, os
import numpy as np
import torch

from env import (Aerial2D, OBS_DIM, EP_LEN, X_LIM, Z_CEIL, VX_MAX, GRAV,
                 BALL_Z_LO, BALL_Z_HI)
from ladder import mlp

GAMMA = 0.99
SIGMA_SCALE = 0.5


# ---------- obs construction from raw state (mirrors env.obs()) ----------
def obs_from_state(x, z, vx, vz, boost, on_ground, bx, bz, bvz, t):
    dx = bx - x; dz = bz - z
    dist = np.hypot(dx, dz)
    return np.stack([
        x / X_LIM, z / Z_CEIL, vx / VX_MAX, vz / 12.0,
        boost, on_ground.astype(np.float64),
        dx / 20.0, dz / Z_CEIL, dist / 22.0,
        bz / Z_CEIL, bvz / 10.0,
        np.minimum(t, EP_LEN) / EP_LEN,
    ], axis=1).astype(np.float32)


def mirror_obs(o):
    """x -> -x flips obs dims 0 (x), 2 (vx), 6 (dx); all else invariant."""
    m = o.clone()
    m[:, 0] = -m[:, 0]; m[:, 2] = -m[:, 2]; m[:, 6] = -m[:, 6]
    return m


# ---------- probe set: phase-2 states, never visited ----------
def make_probes(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-X_LIM, X_LIM, n)
    vx = rng.uniform(-4, 4, n)
    boost = rng.uniform(0.15, 1.0, n)
    bx = rng.uniform(-8, 8, n)
    bz = rng.uniform(5.0, 8.0, n)
    bvz = rng.uniform(-5.0, 0.0, n)
    z = np.zeros(n); vz = np.zeros(n); og = np.ones(n, bool); t = np.full(n, 40)
    obs = obs_from_state(x, z, vx, vz, boost, og, bx, bz, bvz, t)
    state = dict(x=x, z=z, vx=vx, vz=vz, boost=boost, bx=bx, bz=bz, bvz=bvz)
    return torch.from_numpy(obs), state


# ---------- oracle: best scripted intercept over macro variants ----------
def oracle_scores(state, horizon=60):
    n = len(state["x"])
    best = np.zeros(n)
    for delay in (0, 2, 4, 6, 9, 12):
        for drive_gain in (0.4, 0.8, 1.4):
            env = Aerial2D(n, seed=1)
            env.phase = 2
            for k in ("x", "z", "vx", "vz", "boost", "bx", "bz", "bvz"):
                setattr(env, k, state[k].copy())
            env.on_ground = state["z"] <= 0.0
            env.t[:] = 0
            env.prev_dist = np.hypot(env.bx - env.x, env.bz - env.z)
            alive = np.ones(n, bool)       # this ball still in the air, untouched
            touched_t = np.full(n, -1)
            for t in range(horizon):
                dx = env.bx - env.x
                a = np.zeros(n, dtype=np.int64)
                driving = t < delay
                a[driving & (dx > 0.3)] = 2
                a[driving & (dx < -0.3)] = 1
                launch = ~driving
                a[launch & env.on_ground & (np.abs(dx) < drive_gain * 3)] = 3
                a[launch & env.on_ground & (np.abs(dx) >= drive_gain * 3) & (dx > 0)] = 2
                a[launch & env.on_ground & (np.abs(dx) >= drive_gain * 3) & (dx < 0)] = 1
                air = launch & ~env.on_ground
                a[air & (dx > 0.3)] = 6
                a[air & (dx < -0.3)] = 5
                a[air & (np.abs(dx) <= 0.3)] = 4
                _, _, info = env.step(a)
                newly = info["touch"] & alive & (touched_t < 0)
                touched_t[newly] = t
                # ball grounded (respawned by env at bz<=0.5) ends THIS ball's window
                alive &= ~(info["touch"] | (env.bz != env.bz) | (env.bvz == 0.0))
            got = touched_t >= 0
            best[got] = np.maximum(best[got], GAMMA ** touched_t[got])
    return best


# ---------- offline geo retraining from a frozen record ----------
def perturb_states(obs_raw, rng):
    """Physically-perturbed copies of real states: car jitter, ball lifted,
    ball set falling. Rebuilt through obs_from_state so derived dims cohere."""
    n = obs_raw.shape[0]
    o = obs_raw.numpy()
    x = o[:, 0] * X_LIM + rng.uniform(-2, 2, n)
    z = np.clip(o[:, 1] * Z_CEIL + rng.uniform(0, 3, n) * (rng.random(n) < 0.4), 0, Z_CEIL)
    vx = np.clip(o[:, 2] * VX_MAX + rng.uniform(-1.5, 1.5, n), -VX_MAX, VX_MAX)
    vz = np.where(z > 0, rng.uniform(-5, 7, n), 0.0)
    boost = np.clip(o[:, 4] + rng.uniform(-0.2, 0.2, n), 0, 1)
    og = z <= 0.0
    bx = np.clip(x + rng.uniform(-8, 8, n), -8, 8)
    bz = rng.uniform(BALL_Z_LO, 8.0, n)
    bvz = np.where(rng.random(n) < 0.6, rng.uniform(-6, 0, n), 0.0)
    t = rng.integers(0, EP_LEN, n)
    return torch.from_numpy(obs_from_state(x, z, vx, vz, boost, og, bx, bz, bvz, t))


def domain_states(n, rng):
    x = rng.uniform(-X_LIM, X_LIM, n)
    z = np.where(rng.random(n) < 0.5, 0.0, rng.uniform(0, Z_CEIL, n))
    vx = rng.uniform(-VX_MAX, VX_MAX, n)
    vz = np.where(z > 0, rng.uniform(-8, 8, n), 0.0)
    boost = rng.uniform(0, 1, n)
    og = z <= 0.0
    bx = rng.uniform(-8, 8, n)
    bz = rng.uniform(BALL_Z_LO, 8.0, n)
    bvz = np.where(rng.random(n) < 0.5, rng.uniform(-6, 0, n), 0.0)
    t = rng.integers(0, EP_LEN, n)
    return torch.from_numpy(obs_from_state(x, z, vx, vz, boost, og, bx, bz, bvz, t))


def retrain_geo(base, scheme, iters=2500, bs=1024, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    ro, rn, rr = base["res_obs"], base["res_nxt"], base["res_rew"]
    nres = ro.shape[0]
    sym = scheme == "colloc_sym"
    if sym:
        ro = torch.cat([ro, mirror_obs(ro)])
        rn = torch.cat([rn, mirror_obs(base["res_nxt"])])
        rr = torch.cat([rr, base["res_rew"]])
        nres = ro.shape[0]
    sig = mlp(OBS_DIM, 2 * OBS_DIM); rew = mlp(OBS_DIM, 1); fld = mlp(OBS_DIM, 1)
    osig = torch.optim.Adam(sig.parameters(), lr=1e-3)
    orew = torch.optim.Adam(rew.parameters(), lr=1e-3)
    ofld = torch.optim.Adam(fld.parameters(), lr=1e-3)
    # world-facing fits: REAL (or mirrored-real) executed pairs only, all schemes
    for i in range(iters):
        idx = torch.randint(0, nres, (bs,))
        o, nx, r = ro[idx], rn[idx], rr[idx]
        out = sig(o); mu, ls = out[:, :OBS_DIM], out[:, OBS_DIM:].clamp(-7, 2)
        nll = (ls + 0.5 * ((nx - o - mu) / ls.exp()) ** 2).mean()
        osig.zero_grad(); nll.backward(); osig.step()
        rl = ((rew(nx).flatten() - r) ** 2).mean()
        orew.zero_grad(); rl.backward(); orew.step()
    # field: where the EQUATION is enforced differs by scheme
    for i in range(iters):
        if scheme == "onpolicy":
            s = ro[torch.randint(0, nres, (bs,))]
        else:
            k1, k2 = bs // 2, (3 * bs) // 10
            s_res = ro[torch.randint(0, nres, (k1,))]
            s_pert = perturb_states(ro[torch.randint(0, nres, (k2,))], rng)
            s_dom = domain_states(bs - k1 - k2, rng)
            s = torch.cat([s_res, s_pert, s_dom])
            if sym:
                half = torch.rand(s.shape[0]) < 0.5
                s[half] = mirror_obs(s[half])
        s = s.clone().requires_grad_(True)
        v = fld(s).flatten()
        (g,) = torch.autograd.grad(v.sum(), s, create_graph=True)
        with torch.no_grad():
            sg = torch.exp(sig(s)[:, OBS_DIM:].clamp(-7, 2))
            rh = rew(s).flatten()
        dual = torch.sqrt((g ** 2 * (sg * SIGMA_SCALE) ** 2).sum(1) + 1e-8)
        resid = (1 - GAMMA) * v - rh - GAMMA * dual
        loss = (resid ** 2).mean()
        ofld.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(fld.parameters(), 1.0)
        ofld.step()
    return fld


def retrain_unified(base, r0, iters=2500, bs=1024, seed=0):
    """The consolidated critic: ONE field family, two constraint sets.
    Executed-transition expectile TD (twin-min, the composition loss) anchors it
    to the record; the HJB residual propagates the ANCHORED structure along
    Sigma-corridors into unvisited states. r0=True zeroes r-hat on off-manifold
    residual rows: unvisited states draw value ONLY through corridors back to
    anchored regions -- no locally-hallucinated reward, no proximity proxy."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    ro, rn, rr = base["res_obs"], base["res_nxt"], base["res_rew"]
    nres = ro.shape[0]
    sig = mlp(OBS_DIM, 2 * OBS_DIM); rew = mlp(OBS_DIM, 1)
    f1 = mlp(OBS_DIM, 1); f2 = mlp(OBS_DIM, 1)
    import copy as _copy
    t1, t2 = _copy.deepcopy(f1), _copy.deepcopy(f2)
    osig = torch.optim.Adam(sig.parameters(), lr=1e-3)
    orew = torch.optim.Adam(rew.parameters(), lr=1e-3)
    of = torch.optim.Adam(list(f1.parameters()) + list(f2.parameters()), lr=1e-3)
    tau = 0.75
    for i in range(iters):
        idx = torch.randint(0, nres, (bs,))
        o, nx, r = ro[idx], rn[idx], rr[idx]
        out = sig(o); mu, ls = out[:, :OBS_DIM], out[:, OBS_DIM:].clamp(-7, 2)
        nll = (ls + 0.5 * ((nx - o - mu) / ls.exp()) ** 2).mean()
        osig.zero_grad(); nll.backward(); osig.step()
        rl = ((rew(nx).flatten() - r) ** 2).mean()
        orew.zero_grad(); rl.backward(); orew.step()
    for i in range(iters):
        # (1) anchors: expectile TD on executed pairs, twin-min target
        idx = torch.randint(0, nres, (bs,))
        o, nx, r = ro[idx], rn[idx], rr[idx]
        with torch.no_grad():
            y = (r + GAMMA * torch.minimum(t1(nx).flatten(), t2(nx).flatten())
                 ).clamp(-5.0, 50.0)
        td = 0.0
        for f in (f1, f2):
            u = y - f(o).flatten()
            w = torch.where(u > 0, tau, 1 - tau)
            td = td + (w * u * u).mean()
        # (2) propagator: HJB residual on the collocation mix
        k1, k2 = bs // 2, (3 * bs) // 10
        s_res = ro[torch.randint(0, nres, (k1,))]
        s_pert = perturb_states(ro[torch.randint(0, nres, (k2,))], rng)
        s_dom = domain_states(bs - k1 - k2, rng)
        s = torch.cat([s_res, s_pert, s_dom]).clone().requires_grad_(True)
        with torch.no_grad():
            sg = torch.exp(sig(s)[:, OBS_DIM:].clamp(-7, 2))
            rh = rew(s).flatten()
            if r0:
                rh = torch.cat([rh[:k1], torch.zeros(s.shape[0] - k1)])
        resid = 0.0
        for f in (f1, f2):
            v = f(s).flatten()
            (g,) = torch.autograd.grad(v.sum(), s, create_graph=True)
            dual = torch.sqrt((g ** 2 * (sg * SIGMA_SCALE) ** 2).sum(1) + 1e-8)
            resid = resid + (((1 - GAMMA) * v - rh - GAMMA * dual) ** 2).mean()
        loss = td + 0.3 * resid
        of.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(f1.parameters()) + list(f2.parameters()), 1.0)
        of.step()
        if i % 50 == 0:
            t1.load_state_dict(f1.state_dict()); t2.load_state_dict(f2.state_dict())
    return f1, f2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", default="bases")
    ap.add_argument("--nprobe", type=int, default=600)
    ap.add_argument("--schemes", default="onpolicy,colloc,colloc_sym")
    args = ap.parse_args()
    from scipy.stats import spearmanr

    probes, pstate = make_probes(args.nprobe, seed=99)
    print("computing oracle over scripted intercepts...")
    oracle = oracle_scores(pstate)
    feas = (oracle > 0).mean()
    print(f"oracle: {feas:.0%} of probes feasible, mean score {oracle.mean():.3f}")

    rows = []
    for f in sorted(glob.glob(os.path.join(args.bases, "base_s*.pt"))):
        base = torch.load(f, weights_only=False)
        seed = int(f.rsplit("_s", 1)[1].split(".")[0])
        v_net = mlp(base["in_dim"], 1); v_net.load_state_dict(base["v"])
        vd1 = mlp(OBS_DIM, 1); vd1.load_state_dict(base["vdag1"])
        vd2 = mlp(OBS_DIM, 1); vd2.load_state_dict(base["vdag2"])
        with torch.no_grad():
            v_p = v_net(probes).flatten().numpy()
            vdag_p = torch.minimum(vd1(probes).flatten(), vd2(probes).flatten()).numpy()
        out = {"seed": seed,
               "V": spearmanr(v_p, oracle).statistic,
               "Vdag": spearmanr(vdag_p, oracle).statistic}
        for scheme in args.schemes.split(","):
            if scheme.startswith("unified"):
                f1, f2 = retrain_unified(base, r0=scheme.endswith("_r0"), seed=seed)
                with torch.no_grad():
                    geo_p = torch.minimum(f1(probes).flatten(),
                                          f2(probes).flatten()).numpy()
            else:
                fld = retrain_geo(base, scheme, seed=seed)
                with torch.no_grad():
                    geo_p = fld(probes).flatten().numpy()
            out[f"geo_{scheme}"] = spearmanr(geo_p, oracle).statistic
            out[f"geo_{scheme}_vs_V"] = spearmanr(geo_p, v_p).statistic
        rows.append(out)
        print(json.dumps({k: round(float(x), 3) if isinstance(x, float) else x
                          for k, x in out.items()}))

    print("\nmean over seeds (Spearman rho vs oracle on never-visited phase-2 states):")
    keys = [k for k in rows[0] if k != "seed" and not k.endswith("_vs_V")]
    for k in keys:
        vals = [r[k] for r in rows]
        print(f"  {k:18s} {np.mean(vals):+.3f}  (seeds: {' '.join(f'{v:+.2f}' for v in vals)})")
    for k in [k for k in rows[0] if k.endswith("_vs_V")]:
        vals = [r[k] for r in rows]
        print(f"  {k:22s} {np.mean(vals):+.3f}   (tracks-habit check)")


if __name__ == "__main__":
    main()
