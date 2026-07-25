"""Knowing-doing closure curve across policy_versions snapshots (lean protocol).

For each sampled version (POLICY + SHARED_HEAD + rating): 40k self-play rows, landing
labels on subsampled airborne frames, OOF ridge (h1 -> landing, 3-fold episode-grouped),
then the KD headline metrics. Versions are COPIED out first (the trainer rotates them).
Two archived full-protocol points (ckpt 1461076992 @ r688, and results/knowing_doing.json
@ ~4.18B) anchor the far past; this adds the recent-slope points in between.
"""

import json
import shutil
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import RocketSim as rs
from collect_dataset import NUM_ARENAS, TICK_SKIP, ArenaEnv
from label_landing import simulate_landing
from load_checkpoint import PulsarPolicy, load_models
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
VERSIONS = HERE.parents[1] / "build" / "checkpoints_3.1" / "policy_versions"
CACHE = HERE.parent / "data" / "version_cache"
SEED = 99
ROWS = 40_000
N_POINTS = 7
# Follows collect_dataset's dynamics. NB: the 3.1 policy_versions this script
# replays were trained at tickSkip 4 + actionDelay 3 - for faithful archaeology
# set collect_dataset back to 4/3/10 (see its migration comment).
DT = TICK_SKIP / 120.0
KNOW_ERR = 500.0
ATTEND = 500.0
FEASIBLE_SPEED = 1300.0
ALPHAS = np.logspace(-1, 4, 6)


def collect(policy):
    torch.manual_seed(SEED)
    envs = [ArenaEnv(i, np.random.default_rng(SEED + i)) for i in range(NUM_ARENAS)]
    n = ROWS
    h1s = np.empty((n, 512), np.float16)
    phys = np.empty((n, 31), np.float32)
    episode = np.empty(n, np.int32)
    team = np.empty(n, np.int8)
    on_ground = np.zeros(n, bool)
    row = 0
    while row + 2 * NUM_ARENAS <= n:
        obs_l, mask_l, meta = [], [], []
        for env in envs:
            o, m, p = env.observe()
            obs_l.append(o)
            mask_l.append(m)
            grounds = [env.cars[k].get_state().is_on_ground for k in range(2)]
            meta.append((p, grounds, env.episode_id))
        h1, h2 = policy.trunk_forward(torch.from_numpy(np.concatenate(obs_l)))
        acts = policy.sample_actions(h2, torch.from_numpy(np.concatenate(mask_l)))
        for i, env in enumerate(envs):
            p, grounds, ep = meta[i]
            for k in range(2):
                h1s[row] = h1[2 * i + k].numpy()
                phys[row] = p
                episode[row] = ep
                team[row] = k
                on_ground[row] = grounds[k]
                row += 1
        for i, env in enumerate(envs):
            if env.step(acts[2 * i:2 * i + 2].tolist()):
                env.reset()
    return h1s[:row], phys[:row], episode[:row], team[:row], on_ground[:row]


def kd_point(models):
    policy = PulsarPolicy(models)
    h1s, phys, episode, team, on_ground = collect(policy)
    n = len(team)

    arena = rs.Arena(rs.GameMode.SOCCAR)
    ep_rows = {int(e): np.flatnonzero(episode == e) for e in np.unique(episode)}
    row_pos = np.empty(n, np.int64)
    for rows in ep_rows.values():
        row_pos[rows] = np.arange(len(rows))

    # Label ALL airborne frames (the probe must train on the same basis as the full
    # protocol - training it only on feasible-free readings made 'knows' meaningless
    # and the first version of this curve pure noise)
    airborne = np.flatnonzero(phys[:, 2] > 300)
    lab_rows, lab_y = [], []
    land_of = {}
    for r in airborne:
        res = simulate_landing(arena, phys[r, 0:3], phys[r, 3:6], phys[r, 6:9])
        if res is None:
            continue
        lab_rows.append(r)
        lab_y.append(res)
        land_of[r] = res
    lab_rows = np.array(lab_rows)
    y = np.array(lab_y, np.float64)  # (x, y, t) world/team frame below
    if len(lab_rows) < 2000:
        return None

    # OOF ridge h1 -> landing on ALL labeled rows (team-canonical), 3-fold grouped
    orange = team[lab_rows] == 1
    y[orange, :2] *= -1
    X = h1s[lab_rows].astype(np.float64)
    groups = episode[lab_rows]
    pred = np.full_like(y, np.nan)
    for tr, te in GroupKFold(3).split(X, y, groups):
        xs = StandardScaler().fit(X[tr])
        ys_ = StandardScaler().fit(y[tr])
        m = RidgeCV(alphas=ALPHAS).fit(xs.transform(X[tr]), ys_.transform(y[tr]))
        pred[te] = ys_.inverse_transform(np.asarray(m.predict(xs.transform(X[te]))).reshape(len(te), -1))
    know_err_of = dict(zip(lab_rows.tolist(),
                           np.linalg.norm(pred[:, :2] - y[:, :2], axis=1)))

    # Behavior: feasible free landings among the labeled rows
    recs = []  # (know_err, went, t_land, air_before)
    for r in lab_rows:
        lx, ly, t = land_of[int(r)]
        rows = ep_rows[int(episode[r])]
        q = row_pos[r]
        q_land = q + 2 * int(round(t / DT))
        if q_land >= len(rows):
            continue
        L = np.array([lx, ly])
        sl = slice(20, 23) if team[r] == 1 else slice(9, 12)
        d_now = np.linalg.norm(phys[rows[q], sl][:2] - L)
        if d_now / max(t, 1e-6) >= FEASIBLE_SPEED:
            continue
        ball_td = phys[rows[q_land], 0:3]
        if np.linalg.norm(ball_td[:2] - L) > 300 or ball_td[2] > 200:
            continue  # contested
        went = np.linalg.norm(phys[rows[q_land], sl][:2] - L) <= ATTEND
        air_before = (~on_ground[rows[q:q_land + 1:2]]).any() if q_land > q else False
        recs.append((know_err_of[int(r)], went, t, air_before))
    if len(recs) < 400:
        return None

    know_err = np.array([x[0] for x in recs])
    went = np.array([x[1] for x in recs])
    t_land = np.array([x[2] for x in recs])
    air_before = np.array([x[3] for x in recs])

    knows = know_err < KNOW_ERR
    q = np.quantile(know_err, [0.25, 0.5, 0.75])
    quart = np.digitize(know_err, q)
    high = knows & (t_land > 0.75)
    return {
        "n_readings": int(len(recs)),
        "n_knows": int(knows.sum()),
        "p_unattend_knows": float((~went[knows]).mean()) if knows.sum() >= 100 else None,
        "p_unattend_not": float((~went[~knows]).mean()),
        "quartiles": [float((~went[quart == i]).mean()) for i in range(4)],
        "engagement": float(went.mean()),
        "aerial_passivity": float((~air_before[high]).mean()) if high.sum() >= 50 else None,
    }


def main():
    torch.set_num_threads(4)
    rs.init(str(HERE.parents[1] / "build" / "collision_meshes"))
    CACHE.mkdir(parents=True, exist_ok=True)

    versions = sorted(int(d.name) for d in VERSIONS.iterdir() if d.name.isdigit())
    step = max(1, len(versions) // N_POINTS)
    picks = versions[::step][-N_POINTS:]
    if versions[-1] not in picks:
        picks.append(versions[-1])
    print(f"{len(versions)} versions available, sampling {len(picks)}: "
          f"{picks[0]/1e9:.2f}B .. {picks[-1]/1e9:.2f}B")

    curve = []
    t0 = time.time()
    for ts in picks:
        dst = CACHE / str(ts)
        if not dst.exists():
            try:
                shutil.copytree(VERSIONS / str(ts), dst)
            except (FileNotFoundError, OSError):
                print(f"  {ts}: rotated away, skipping")
                continue
        rating = json.loads((dst / "STATS.json").read_text())["skill_ratings"].get("1v1")
        models = load_models(dst, names=["SHARED_HEAD", "POLICY"])
        point = kd_point(models)
        if point is None:
            print(f"  {ts}: too few readings, skipped")
            continue
        point.update({"timesteps": ts, "rating": rating})
        curve.append(point)
        pk = point["p_unattend_knows"]
        print(f"  {ts/1e9:.2f}B r{rating:.0f}: unattend|knows "
              f"{pk:.1%}" if pk is not None else f"  {ts/1e9:.2f}B r{rating:.0f}: knows-n too small",
              f"(n_knows {point['n_knows']}) | not {point['p_unattend_not']:.1%} "
              f"| Q1 {point['quartiles'][0]:.1%} ({time.time()-t0:.0f}s)", flush=True)

    out = HERE.parent / "results" / "kd_curve.json"
    out.write_text(json.dumps(curve, indent=2))

    # plot with the two archived anchors
    anchors = [(1.461, 688, 0.660, 0.718), (4.18, 1130, 0.526, 0.666)]  # ts_B, rating, knows, not
    fig, ax = plt.subplots(figsize=(9, 5))
    pts = [p for p in curve if p["p_unattend_knows"] is not None]
    ts = [p["timesteps"] / 1e9 for p in pts]
    ax.plot(ts, [p["p_unattend_knows"] for p in pts], "o-", color="tab:blue",
            label="P(skip | trunk reads landing well)")
    ax.plot(ts, [p["p_unattend_not"] for p in pts], "o-", color="tab:red", alpha=0.6,
            label="P(skip | reads poorly)")
    for tb, r, k, nk in anchors:
        ax.scatter([tb], [k], marker="*", s=180, color="tab:blue", zorder=5)
        ax.scatter([tb], [nk], marker="*", s=180, color="tab:red", zorder=5)
        ax.annotate(f"r{r}", (tb, k), textcoords="offset points", xytext=(6, -12))
    for p in curve[::3]:
        ax.annotate(f"r{p['rating']:.0f}", (p["timesteps"] / 1e9, p["p_unattend_knows"]),
                    textcoords="offset points", xytext=(4, 8), fontsize=8)
    ax.set_xlabel("training timesteps (billions)")
    ax.set_ylabel("P(unattended | feasible free landing)")
    ax.set_title("Knowing-doing gap closure (stars = archived full-protocol anchors)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE.parent / "results" / "plots" / "kd_curve.png", dpi=120)
    print(f"\nwrote {out} + plots/kd_curve.png  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
