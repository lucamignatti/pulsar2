"""Causal test of the prevAction-degeneracy claim (INTERP_53.md section 2).

Re-solves a FRESH V-geo field offline against the SAME frozen world-facing fits
(GEO_SIGMA, GEO_REW from the live checkpoint), same HJB residual, same gamma, same
{384,384} architecture and Adam lr 1e-3 - in two arms:

  arm "full"    Sigma-norm over all 230 obs dims  (production objective; should
                reproduce the degenerate flat field => sanity check on the offline solve)
  arm "masked"  Sigma-norm restricted to physical dims: ball pos/vel/angvel (0:9),
                self block (51:80), opponent car slot 1 (138:167). prevAction, pads,
                empty teammate/opp slots, presence flags excluded.

Then evaluates both fields on the geo_field_probe state families. Pre-registered read:
the masked arm restores between-family structure (between-family std > within-family
std, and |corr with ball_z| in family A materially above the full arm's).

Training states: the 160k on-policy dataset (the live field also trains on current
states; the frozen fits carry the reservoir's world knowledge).
"""

import json
from pathlib import Path

import numpy as np
import torch

from load_checkpoint_53 import (GEO_SIGMA_SCALE, TRAIN_GAMMA, Pulsar53Policy,
                                list_checkpoints, load_models)

SEED = 99
EPOCHS = 8
BATCH = 4096
RESULTS = Path(__file__).resolve().parents[1] / "results"
DATA = Path(__file__).resolve().parents[1] / "data"

PHYS_MASK_SLICES = [(0, 9), (51, 80), (138, 167)]


def make_net(d_in):
    return torch.nn.Sequential(
        torch.nn.Linear(d_in, 384), torch.nn.LayerNorm(384), torch.nn.LeakyReLU(),
        torch.nn.Linear(384, 384), torch.nn.LayerNorm(384), torch.nn.LeakyReLU(),
        torch.nn.Linear(384, 1))


def solve(obs, rh, sig, mask, label):
    torch.manual_seed(SEED)
    net = make_net(obs.shape[1])
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    n = len(obs)
    for ep in range(EPOCHS):
        perm = torch.randperm(n)
        tot, nb = 0.0, 0
        for i in range(0, n - BATCH + 1, BATCH):
            idx = perm[i:i + BATCH]
            gin = obs[idx].clone().requires_grad_(True)
            vg = net(gin).flatten()
            gx = torch.autograd.grad(vg.sum(), gin, create_graph=True)[0]
            gnorm = ((gx * mask) ** 2 * sig[idx] ** 2).sum(-1).clamp_min(1e-12).sqrt()
            resid = (1.0 - TRAIN_GAMMA) * vg - rh[idx] - TRAIN_GAMMA * gnorm
            loss = (resid ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += float(loss)
            nb += 1
        print(f"  [{label}] epoch {ep}: resid^2 {tot/nb:.6f}", flush=True)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def main():
    torch.set_num_threads(4)
    root = DATA / "ckpt53"
    ckpt = list_checkpoints(root)[0]
    pol = Pulsar53Policy(load_models(ckpt, ["SHARED_HEAD", "POLICY", "GEO_SIGMA",
                                            "GEO_REW", "GEO_V", "CRITIC_TRUNK", "CRITIC"]))
    d = np.load(DATA / "dataset53.npz")
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(d["obs"]), 120_000, replace=False)
    obs = torch.from_numpy(d["obs"][idx].astype(np.float32))

    with torch.no_grad():
        rh = pol.models["GEO_REW"](obs).flatten()
        so = pol.models["GEO_SIGMA"](obs)
        dd = obs.shape[1]
        sig = so[:, dd:2 * dd].clamp(-8.0, 2.0).exp() * GEO_SIGMA_SCALE

    mask_full = torch.ones(dd)
    mask_phys = torch.zeros(dd)
    for a, b in PHYS_MASK_SLICES:
        mask_phys[a:b] = 1.0

    nets = {"full": solve(obs, rh, sig, mask_full, "full"),
            "masked": solve(obs, rh, sig, mask_phys, "masked")}

    # evaluate on the probe families
    import geo_field_probe as gfp
    import RocketSim as rs
    import collect_dataset as cd
    rs.init(str(Path(__file__).resolve().parents[2] / "build" / "collision_meshes"))
    cd.set_obs_size(pol.obs_size)
    env = cd.ArenaEnv(0, np.random.default_rng(SEED + 1))
    rngf = np.random.default_rng(gfp.SEED)
    fam_obs = {}
    for fam in gfp.FAMILIES:
        rows = []
        for i in range(gfp.N_PER):
            gfp.build_family(env, rngf, fam, i)
            o, m, p = env.observe()
            rows.append(o[0])
        fam_obs[fam] = torch.from_numpy(np.stack(rows).astype(np.float32))

    out = {"checkpoint": int(ckpt.name), "epochs": EPOCHS,
           "mask_slices": PHYS_MASK_SLICES, "arms": {}}
    for arm, net in nets.items():
        fams = {}
        with torch.no_grad():
            for fam, o in fam_obs.items():
                v = net(o).flatten().numpy()
                fams[fam] = {"mean": float(v.mean()), "std": float(v.std())}
        means = [fams[f]["mean"] for f in gfp.FAMILIES]
        within = float(np.mean([fams[f]["std"] for f in gfp.FAMILIES]))
        # family A ball-z correlation
        with torch.no_grad():
            va = net(fam_obs["A_ball_height"]).flatten().numpy()
        bz = np.linspace(100, 2000, gfp.N_PER)
        out["arms"][arm] = {
            "families": fams,
            "between_family_std": float(np.std(means)),
            "within_family_std": within,
            "A_corr_ball_z": float(np.corrcoef(va, bz)[0, 1]),
        }
        print(f"arm {arm}: between-family std {np.std(means):.4f}  within {within:.4f}  "
              f"A corr(ball_z) {out['arms'][arm]['A_corr_ball_z']:+.3f}")
        for f in gfp.FAMILIES:
            print(f"   {f:20s} {fams[f]['mean']:8.3f} +- {fams[f]['std']:.3f}")

    p = RESULTS / f"masked_geo_resolve_{ckpt.name}.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
