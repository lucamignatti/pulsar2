"""Pad study + corrected oracle.

oracle2 fixes two flaws in foresight.oracle_scores: the env phase is now
EXPLICIT (the old oracle ran phase 2 always, silently turning the 'static
ball' families B/C into falling-ball tasks), and ball lifetime is tracked via
respawn detection rather than the bvz==0 heuristic. Pad mode adds detour
macros (drive to the pad until full, then intercept).

Family D (the discriminating instrument): pad-env ground states with LOW
boost and high static balls -- value requires the pad-detour chain, so it is
DISCONTINUOUS in boost x position and long-range. The regime where stitching
must matter: if the eps-inflated operator's advantage is mechanism and not
noise, it separates here.
"""
import argparse, copy, glob, json, os
import numpy as np
import torch

from env import Aerial2D, OBS_DIM, PAD_X
from ladder import mlp
from foresight import obs_from_state
from metric4 import train_vddag

GAMMA = 0.99


def oracle2(state, phase=1, pad=False, horizon=100):
    n = len(state["x"])
    best = np.zeros(n)
    variants = [(d, g, det) for d in (0, 2, 4, 6, 9, 12) for g in (0.4, 0.8, 1.4)
                for det in ((False, True) if pad else (False,))]
    for delay, gain, detour in variants:
        env = Aerial2D(n, seed=1)
        env.phase = phase
        if pad:
            env.enable_pad()
        for k in ("x", "z", "vx", "vz", "boost", "bx", "bz", "bvz"):
            setattr(env, k, state[k].copy())
        env.on_ground = state["z"] <= 0.0
        env.t[:] = 0
        env.prev_dist = np.hypot(env.bx - env.x, env.bz - env.z)
        bx0, bz0 = env.bx.copy(), env.bz.copy()
        alive = np.ones(n, bool)
        touched_t = np.full(n, -1)
        fueled = ~np.array([detour] * n)          # detour phase done?
        for t in range(horizon):
            dx = env.bx - env.x
            a = np.zeros(n, dtype=np.int64)
            # detour phase: drive to the pad until tank full
            need_pad = ~fueled & (env.boost < 0.99)
            pdx = PAD_X - env.x
            a[need_pad & (pdx > 0.3)] = 2
            a[need_pad & (pdx < -0.3)] = 1
            fueled |= env.boost >= 0.99
            act = fueled | ~need_pad
            driving = act & (t < delay)
            a[driving & (dx > 0.3)] = 2
            a[driving & (dx < -0.3)] = 1
            launch = act & ~driving
            a[launch & env.on_ground & (np.abs(dx) < gain * 3)] = 3
            a[launch & env.on_ground & (np.abs(dx) >= gain * 3) & (dx > 0)] = 2
            a[launch & env.on_ground & (np.abs(dx) >= gain * 3) & (dx < 0)] = 1
            air = launch & ~env.on_ground
            a[air & (dx > 0.3)] = 6
            a[air & (dx < -0.3)] = 5
            a[air & (np.abs(dx) <= 0.3)] = 4
            _, _, info = env.step(a)
            newly = info["touch"] & alive & (touched_t < 0)
            touched_t[newly] = t
            respawned = (np.abs(env.bx - bx0) > 1e-9) | (np.abs(env.bz - bz0) > 1e-9)
            if phase == 2:
                # falling ball: position changes every step; respawn = upward jump
                respawned = info["touch"] | (env.bz > bz0 + 0.01)
                bz0 = env.bz.copy()
            alive &= ~respawned
            if phase == 1:
                bx0, bz0 = env.bx.copy(), env.bz.copy()
        got = touched_t >= 0
        best[got] = np.maximum(best[got], GAMMA ** touched_t[got])
    return best


def make_pad_probes(n, seed, lo_boost=True):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-10.0, 10.0, n)
    vx = rng.uniform(-4, 4, n)
    boost = rng.uniform(0.0, 0.25, n) if lo_boost else rng.uniform(0.0, 1.0, n)
    bx = rng.uniform(-8, 8, n)
    bz = rng.uniform(4.5, 7.5, n)
    bvz = np.zeros(n)
    z = np.zeros(n); vz = np.zeros(n); og = np.ones(n, bool); t = np.full(n, 30)
    return (torch.from_numpy(obs_from_state(x, z, vx, vz, boost, og, bx, bz, bvz, t)),
            dict(x=x, z=z, vx=vx, vz=vz, boost=boost, bx=bx, bz=bz, bvz=bvz))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", default="bases_pad")
    ap.add_argument("--nprobe", type=int, default=600)
    args = ap.parse_args()
    from scipy.stats import spearmanr
    sp = lambda a, b: spearmanr(a, b).statistic

    pD, sD = make_pad_probes(args.nprobe, seed=96, lo_boost=True)
    pE, sE = make_pad_probes(args.nprobe, seed=95, lo_boost=False)  # mixed boost
    print("pad oracles (with detour macros)...")
    oD = oracle2(sD, phase=1, pad=True)
    oE = oracle2(sE, phase=1, pad=True)
    print(f"  D low-boost   feasible {np.mean(oD > 0):.0%} mean {oD.mean():.3f}")
    print(f"  E mixed-boost feasible {np.mean(oE > 0):.0%} mean {oE.mean():.3f}")

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
            r["vdag_D"] = sp(vdag(pD).numpy(), oD)
            r["vdag_E"] = sp(vdag(pE).numpy(), oE)
        for eps in (1.0,):
            nets = train_vddag(ro, rn, rr, eps, seed=seed)
            fn = lambda s: torch.minimum(nets[0](s).flatten(), nets[1](s).flatten())
            with torch.no_grad():
                r[f"e{eps:g}_D"] = sp(fn(pD).numpy(), oD)
                r[f"e{eps:g}_E"] = sp(fn(pE).numpy(), oE)
        rows.append(r)
        print(json.dumps({k: round(float(v), 3) for k, v in r.items()}))

    print("\nmean over seeds (Spearman rho vs oracle, pad env):")
    for k in [k for k in rows[0] if k != "seed"]:
        vals = [x[k] for x in rows]
        print(f"  {k:10s} {np.mean(vals):+.3f}  (seeds: {' '.join(f'{v:+.2f}' for v in vals)})")


if __name__ == "__main__":
    main()
