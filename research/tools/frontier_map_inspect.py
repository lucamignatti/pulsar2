"""Interrogate the trained frontier map: is V accurate, does d recover decision
counts, and WHAT does goal selection actually aim at?  Mirrors Frontier.cpp exactly."""
import sys
from pathlib import Path
sys.path.insert(0, "/home/luca/Projects/pulsar2-3.0/research/tools")
import numpy as np, torch, json
import load_checkpoint_70 as l70

SP = Path(sys.argv[1])
torch.set_num_threads(6)
K_ASYM, K_SYM = 32, 8
BAND_LO, BAND_HI = 1.5, 6.0
GAMMA = 0.9969

d = np.load(SP / "rollout.npz")
obs = torch.from_numpy(d["obs"]); phys = d["phys"]; flip = d["flip"]
ep, stepidx, player = d["ep"], d["stepidx"], d["player"]
stats = json.load(open(SP / "RUNNING_STATS.json"))
rstd = (stats["return_stat"]["var"] / stats["return_stat"]["count"]) ** 0.5
print(f"return std {rstd:.2f}  -> a goal is worth {150/rstd:.3f} in map units\n")

# frontier nets: plain Linear/LayerNorm/ReLU stacks, no residuals, output layer
import load_checkpoint_53 as l53
l53._LADDER_ARCH = {"FRONTIER_VALUE_A": ([256, 256], True, False, True),
                    "FRONTIER_VALUE_B": ([256, 256], True, False, True),
                    "FRONTIER_QUASI":   ([256, 256], True, False, True)}
nets = {n: l53.rebuild_model(torch.jit.load(str(SP / f"{n}.lt"), map_location="cpu"), n).eval()
        for n in l53._LADDER_ARCH}


@torch.no_grad()
def V(x):
    return torch.minimum(nets["FRONTIER_VALUE_A"](x).squeeze(-1),
                         nets["FRONTIER_VALUE_B"](x).squeeze(-1))


@torch.no_grad()
def enc(x):
    return nets["FRONTIER_QUASI"](x)


def dist(a, b):
    asym = torch.relu(b[..., :K_ASYM] - a[..., :K_ASYM]).amax(-1)
    sym = (b[..., K_ASYM:K_ASYM + K_SYM] - a[..., K_ASYM:K_ASYM + K_SYM]).norm(2, -1)
    return asym + sym


with torch.no_grad():
    v = torch.cat([V(obs[i:i + 8192]) for i in range(0, len(obs), 8192)]).numpy()
    z = torch.cat([enc(obs[i:i + 8192]) for i in range(0, len(obs), 8192)])

# ---- interpretable features (team-canonical: +y is the attacking direction) -------
bpos, bvel = phys[:, 0:3], phys[:, 3:6]
c0, c1 = phys[:, 9:20], phys[:, 20:31]
own = np.where(player[:, None] == 0, c0, c1)
opp = np.where(player[:, None] == 0, c1, c0)
sgn = np.where(player == 0, 1.0, -1.0)            # ORANGE sees the field mirrored
feat = dict(
    ball_z=bpos[:, 2],
    ball_speed=np.linalg.norm(bvel, axis=1),
    ball_toward_net=sgn * bvel[:, 1],
    ball_y_attack=sgn * bpos[:, 1],
    car_z=own[:, 2],
    car_airborne=1.0 - own[:, 10],
    car_boost=own[:, 9],
    car_ball_dist=np.linalg.norm(own[:, 0:3] - bpos, axis=1),
    has_flip=flip[:, 0], had_flip_reset=flip[:, 1], air_time=flip[:, 3],
    opp_ball_dist=np.linalg.norm(opp[:, 0:3] - bpos, axis=1),
)
print(f"frames {len(obs)}  airborne {feat['car_airborne'].mean():.3f}  "
      f"flip-available {feat['has_flip'].mean():.3f}  "
      f"flip-reset frames {feat['had_flip_reset'].mean():.6f}")

# ================= 1. VALUE MAP ACCURACY =========================================
print("\n=== 1. VALUE MAP: does V predict the real discounted goal outcome? ===")
end_step = dict(zip(d["ep_end_keys"].tolist(), d["ep_end_step"].tolist()))
end_team = dict(zip(d["ep_end_keys"].tolist(), d["ep_end_team"].tolist()))
ret = np.full(len(obs), np.nan)
for i in range(len(obs)):
    e = int(ep[i])
    if e not in end_step:
        continue
    t = end_team[e]
    if t < 0:
        ret[i] = 0.0                                  # no-touch / cap: no reward at all
    else:
        k = end_step[e] - stepidx[i]
        s = 1.0 if t == player[i] else -1.0           # player index == team index
        ret[i] = s * (150.0 / rstd) * (GAMMA ** max(k, 0))
ok = np.isfinite(ret)
print(f"  labelled frames {ok.sum()} over {len(end_step)} closed episodes "
      f"({sum(1 for v_ in end_team.values() if v_>=0)} ended in a goal)")
if ok.sum() > 100:
    r = np.corrcoef(v[ok], ret[ok])[0, 1]
    print(f"  Pearson r(V, realized return) = {r:+.3f}")
    g = ok & (ret != 0)
    if g.sum() > 50:
        print(f"  on GOAL episodes only: r = {np.corrcoef(v[g], ret[g])[0,1]:+.3f}  "
              f"(n={g.sum()})")
        sc, co = g & (ret > 0), g & (ret < 0)
        print(f"  mean V | about to score {v[sc].mean():+.3f} (n={sc.sum()})  "
              f"| about to concede {v[co].mean():+.3f} (n={co.sum()})  "
              f"-> separation {v[sc].mean()-v[co].mean():+.3f}")

print("\n  V by state class (mean +- sd):")
classes = {
    "grounded, far from ball": (feat["car_airborne"] < .5) & (feat["car_ball_dist"] > 1500),
    "grounded, near ball":     (feat["car_airborne"] < .5) & (feat["car_ball_dist"] < 500),
    "airborne, low ball":      (feat["car_airborne"] > .5) & (feat["ball_z"] < 300),
    "airborne, high ball":     (feat["car_airborne"] > .5) & (feat["ball_z"] > 800),
    "airborne + flip avail":   (feat["car_airborne"] > .5) & (feat["has_flip"] > .5),
    "airborne, flip spent":    (feat["car_airborne"] > .5) & (feat["has_flip"] < .5),
    "ball in attacking third": feat["ball_y_attack"] > 2500,
    "ball in own third":       feat["ball_y_attack"] < -2500,
}
for name, msk in classes.items():
    if msk.sum() > 50:
        print(f"    {name:26s} n={msk.sum():6d}  V={v[msk].mean():+.3f} +- {v[msk].std():.3f}")

# ================= 2. QUASIMETRIC: decision counts ================================
print("\n=== 2. QUASIMETRIC: does d(s_t -> s_t+k) track k decisions? ===")
order = np.lexsort((stepidx, player, ep))
eo, so, zo = ep[order], stepidx[order], z[order]
for k in [1, 2, 4, 8, 16, 32, 64]:
    a = np.arange(len(order) - k)
    same = (eo[a] == eo[a + k]) & (so[a + k] - so[a] == k)
    a = a[same]
    if len(a) < 200:
        continue
    sub = a[np.random.default_rng(0).choice(len(a), min(4000, len(a)), replace=False)]
    fwd = dist(zo[sub], zo[sub + k]).numpy()
    bwd = dist(zo[sub + k], zo[sub]).numpy()
    print(f"  k={k:3d}  d_fwd={fwd.mean():.3f}  d_bwd={bwd.mean():.3f}  "
          f"asym ratio={bwd.mean()/max(fwd.mean(),1e-6):.2f}  n={len(sub)}")

# ================= 3. WHAT DOES IT AIM AT? =======================================
print("\n=== 3. GOAL SELECTION: what states does the map steer toward? ===")
rng = np.random.default_rng(0)
pool_i = rng.choice(len(obs), 4096, replace=False)      # candidatePool
probe_i = rng.choice(len(obs), 2048, replace=False)
zc, vc = z[pool_i], torch.from_numpy(v[pool_i])
zp, vp = z[probe_i], torch.from_numpy(v[probe_i])
D = dist(zp[:, None, :], zc[None, :, :])               # [probe, pool]
gain = vc[None, :] - vp[:, None]
inband = (D >= BAND_LO) & (D <= BAND_HI)
score = torch.where(inband, gain, torch.full_like(gain, -1e18))
best = score.argmax(1).numpy()
valid = inband.any(1).numpy()
print(f"  valid frac {valid.mean():.3f}   mean chosen dist "
      f"{D[np.arange(len(probe_i)), best].numpy()[valid].mean():.2f}   "
      f"mean gain {gain[np.arange(len(probe_i)), best].numpy()[valid].mean():+.3f}")
sel = pool_i[best[valid]]
print(f"\n  feature            pool mean    SELECTED GOALS   ratio")
for name, arr in feat.items():
    pm, sm = arr[pool_i].mean(), arr[sel].mean()
    print(f"  {name:18s} {pm:10.3f} {sm:16.3f}   {sm/pm if abs(pm)>1e-6 else float('nan'):6.2f}")
print(f"  {'V (map units)':18s} {v[pool_i].mean():10.3f} {v[sel].mean():16.3f}")
