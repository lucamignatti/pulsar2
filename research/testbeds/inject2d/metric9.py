"""A4: transplant-legality audit. The hull operator grafts witnessed
displacement vectors onto chart-nearest states. The env itself judges:
from the receiving state, can ANY single action produce a displacement
close to the transplanted one? Reports the illegal-transplant rate and
whether illegal transplants skew value-optimistic (the harm direction)."""
import glob, json
import numpy as np
import torch

from env import Aerial2D, N_ACT, OBS_DIM, X_LIM, Z_CEIL, VX_MAX
from ladder import mlp
from metric6 import train_chart, filter_teleports


def raw_from_obs(o):
    return dict(x=o[:, 0] * X_LIM, z=(o[:, 1] * Z_CEIL).clamp(min=0),
                vx=o[:, 2] * VX_MAX, vz=o[:, 3] * 12.0,
                boost=o[:, 4].clamp(0, 1), bx=o[:, 6] * 20.0 + o[:, 0] * X_LIM,
                bz=(o[:, 9] * Z_CEIL).clamp(min=0.2), bvz=o[:, 10] * 10.0)


def main():
    rows = []
    for f in sorted(glob.glob("bases/base_s*.pt"))[:3]:
        base = torch.load(f, weights_only=False)
        seed = int(f.rsplit("_s", 1)[1].split(".")[0])
        ro, rn = base["res_obs"], base["res_nxt"]
        vd1 = mlp(OBS_DIM, 1); vd1.load_state_dict(base["vdag1"])
        cs, _ = train_chart(ro, rn, seed=seed)
        keep = filter_teleports(ro, rn)
        bo, bd = ro[keep], (rn - ro)[keep]
        bidx = torch.randint(0, bo.shape[0], (16384,))
        bank_o, bank_d = bo[bidx], bd[bidx]
        with torch.no_grad():
            bank_c = cs.P(bank_o)
            n = 2048
            q = rn[torch.randint(0, rn.shape[0], (n,))]
            qc = cs.P(q)
            nnk = torch.cdist(qc, bank_c).topk(4, largest=False).indices
        illegal, dv_ill, dv_leg, gated_out = 0, [], [], 0
        for j in range(4):
            delta = bank_d[nnk[:, j]]
            with torch.no_grad():
                mu, ls = cs(q)
                z = ((delta - mu) / ls.exp()).abs().max(1).values
            keepg = z < 3.0
            gated_out += int((~keepg).sum())
            st = raw_from_obs(q)
            best = torch.full((n,), 1e9)
            for a in range(N_ACT):
                env = Aerial2D(n, seed=1)
                for k in ("x", "z", "vx", "vz", "boost", "bx", "bz", "bvz"):
                    setattr(env, k, st[k].numpy().astype(np.float64).copy())
                env.on_ground = (st["z"].numpy() <= 1e-6)
                env.t[:] = 40
                env.prev_dist = np.hypot(env.bx - env.x, env.bz - env.z)
                o0 = torch.from_numpy(env.obs())
                env.step(np.full(n, a))
                ach = torch.from_numpy(env.obs()) - o0
                # compare on the car-physics dims (0-5); ball dims judged
                # separately by the teleport filter
                err = (ach[:, :6] - delta[:, :6]).abs().max(1).values
                best = torch.minimum(best, err)
            scale = delta[:, :6].abs().max(1).values + 0.02
            bad = (best > 0.5 * scale) & keepg
            good = (best <= 0.5 * scale) & keepg
            illegal += int(bad.sum())
            with torch.no_grad():
                dv = (vd1(q + delta).flatten() - vd1(q).flatten())
            dv_ill += dv[bad].tolist(); dv_leg += dv[good].tolist()
        kept = 4 * n - gated_out
        r = {"seed": seed, "illegal_rate": illegal / max(kept, 1),
             "gated_frac": gated_out / (4 * n),
             "dv_illegal_mean": float(np.mean(dv_ill)) if dv_ill else 0.0,
             "dv_legal_mean": float(np.mean(dv_leg))}
        rows.append(r)
        print(json.dumps({k: round(float(v), 4) for k, v in r.items()}))
    print("means:", {k: round(float(np.mean([x[k] for x in rows])), 4)
                     for k in rows[0] if k != "seed"})


if __name__ == "__main__":
    main()
