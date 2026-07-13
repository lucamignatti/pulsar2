"""Offline validation of 'optimism surgery': steer a commitment direction in the trunk
during rollouts and measure whether behavior moves the intended way.

Protocol (all against ONE pinned checkpoint):
1. Collect a base self-play dataset; label ball landings (car-free sim); for each
   feasible free landing compute whether the reading's player was at the landing spot
   at touchdown (went) or not (declined). No probe needed - the knowing-doing study
   showed unattendance is flat in knowledge, so we condition only on feasibility.
2. Commitment direction v = normalize(mean h2[went] - mean h2[declined]) on frames
   MATCHED over (distance-to-landing, flight-time) bins, so v doesn't just encode
   "already close to the ball". Sanity: projection AUC went-vs-declined.
3. Roll out with h2' = h2 + alpha * sigma_proj * v fed to the POLICY HEAD only
   (trunk taps and phi untouched), alpha in ALPHAS, same arena seeds per alpha,
   both players steered. Measure: landing attendance, aerial touch ratio (trainer
   metric parity: touch & airborne & ball z>400), in-air ratio, touch rate, goals,
   and kickoff time-to-first-touch as the competence canary.

Verdict criteria: attendance and aerial-contest metrics rise with alpha while the
kickoff canary stays intact -> steering is real and directionally correct.
"""

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import RocketSim as rs
from collect_dataset import NUM_ARENAS, ArenaEnv
from label_landing import simulate_landing
from load_checkpoint import PulsarPolicy, copy_checkpoint, load_models

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

SEED = 4242
BASE_ROWS = 60_000
STEER_ROWS = 80_000
ALPHAS = [-2.0, -1.0, 0.0, 1.0, 2.0]
DT = 1 / 30.0
AIRBORNE_Z = 300.0
ATTEND_RADIUS = 500.0
FEASIBLE_SPEED = 1300.0
MATCH_BINS_D = 5
MATCH_BINS_T = 3


class SteeredPolicy(PulsarPolicy):
    """Adds alpha * scale * v to the trunk output fed to the policy head only."""

    def __init__(self, models, v=None, alpha=0.0, scale=1.0):
        super().__init__(models)
        self.v = None if v is None else torch.from_numpy(v.astype(np.float32))
        self.alpha = alpha
        self.scale = scale

    @torch.no_grad()
    def act(self, obs, masks):
        h1, h2 = self.trunk_forward(obs)
        h2_pol = h2 if (self.v is None or self.alpha == 0.0) else \
            h2 + self.alpha * self.scale * self.v
        return h2, self.sample_actions(h2_pol, masks)


def rollout(policy, n_rows, seed, want_h2=False):
    """Self-play pass recording behavior metrics + physics for landing analysis."""
    torch.manual_seed(seed)
    envs = [ArenaEnv(i, np.random.default_rng(seed + 10 + i)) for i in range(NUM_ARENAS)]

    rec = {
        "phys": np.empty((n_rows, 31), np.float32),
        "episode": np.empty(n_rows, np.int32),
        "team": np.empty(n_rows, np.int8),
        "touched": np.zeros(n_rows, bool),
        "on_ground": np.zeros(n_rows, bool),
        "kickoff": np.zeros(n_rows, bool),
    }
    if want_h2:
        rec["h2"] = np.empty((n_rows, 512), np.float16)
    goals = 0
    episodes_done = 0
    # per-env tick of last seen touch, to convert ball_hit_info to per-step events
    last_hit = [dict() for _ in envs]
    kick_first_touch = []   # seconds, kickoff episodes only
    kick_start = {}

    row = 0
    while row + 2 * NUM_ARENAS <= n_rows:
        obs_list, mask_list = [], []
        metas = []
        for ei, env in enumerate(envs):
            obs, masks, phys = env.observe()
            obs_list.append(obs)
            mask_list.append(masks)
            tick = env.arena.tick_count
            per_player = []
            for p, car in enumerate(env.cars):
                st = car.get_state()
                bhi = st.ball_hit_info
                hit_tick = bhi.tick_count_when_hit if bhi.is_valid else -1
                prev = last_hit[ei].get((env.episode_id, p), -1)
                touched = bhi.is_valid and hit_tick > prev and hit_tick > tick - 4
                if touched:
                    last_hit[ei][(env.episode_id, p)] = hit_tick
                    if env.is_kickoff and env.episode_id in kick_start:
                        kick_first_touch.append((tick - kick_start[env.episode_id]) / 120.0)
                        del kick_start[env.episode_id]
                per_player.append((touched, st.is_on_ground))
            metas.append((phys, per_player, env.is_kickoff, env.episode_id))

        obs_b = torch.from_numpy(np.concatenate(obs_list))
        mask_b = torch.from_numpy(np.concatenate(mask_list))
        h2, actions = policy.act(obs_b, mask_b)

        for i, env in enumerate(envs):
            phys, per_player, is_kick, ep = metas[i]
            for p in range(2):
                rec["phys"][row] = phys
                rec["episode"][row] = ep
                rec["team"][row] = p
                rec["touched"][row] = per_player[p][0]
                rec["on_ground"][row] = per_player[p][1]
                rec["kickoff"][row] = is_kick
                if want_h2:
                    rec["h2"][row] = h2[2 * i + p].numpy()
                row += 1

        for i, env in enumerate(envs):
            if env.step(actions[2 * i:2 * i + 2].tolist()):
                goals += int(env.goal_scored)
                episodes_done += 1
                env.reset()
                if env.is_kickoff:
                    kick_start[env.episode_id] = env.arena.tick_count

    for k in rec:
        rec[k] = rec[k][:row]
    rec["goals"] = goals
    rec["episodes"] = max(episodes_done, 1)
    rec["kick_first_touch"] = kick_first_touch
    return rec


def landing_behavior(rec, team_filter=None):
    """Label landings and compute attendance of feasible free landings (KD parity).
    team_filter: restrict readings to one team's rows (for cross-play comparisons)."""
    phys, episode, team = rec["phys"], rec["episode"], rec["team"]
    airborne = phys[:, 2] > AIRBORNE_Z
    if team_filter is not None:
        airborne &= team == team_filter

    arena = rs.Arena(rs.GameMode.SOCCAR)
    idxs = np.flatnonzero(airborne)
    # subsample for speed: every 3rd airborne frame is plenty for aggregate stats
    idxs = idxs[::3]

    ep_rows = {int(e): np.flatnonzero(episode == e) for e in np.unique(episode)}
    row_pos = np.empty(len(team), np.int64)
    for rows in ep_rows.values():
        row_pos[rows] = np.arange(len(rows))

    # Engagement accounting over ALL feasible readings, contested included - steering
    # changes which landings stay free, so P(attended | stayed free) alone drifts with
    # its own denominator. engaged = contested in flight OR present at touchdown.
    n_feas = n_contested = n_attended = 0
    for r in idxs:
        res = simulate_landing(arena, phys[r, 0:3], phys[r, 3:6], phys[r, 6:9])
        if res is None:
            continue
        lx, ly, t_land = res
        rows = ep_rows[int(episode[r])]
        q = row_pos[r]
        q_land = q + 2 * int(round(t_land / DT))
        if q_land >= len(rows):
            continue
        L = np.array([lx, ly])
        sl = slice(20, 23) if team[r] == 1 else slice(9, 12)
        d_now = np.linalg.norm(phys[rows[q], sl][:2] - L)
        if d_now / max(t_land, 1e-6) >= FEASIBLE_SPEED:
            continue
        n_feas += 1
        ball_td = phys[rows[q_land], 0:3]
        if np.linalg.norm(ball_td[:2] - L) > 300 or ball_td[2] > 200:
            n_contested += 1
        elif np.linalg.norm(phys[rows[q_land], sl][:2] - L) <= ATTEND_RADIUS:
            n_attended += 1
    return {
        "engagement": (n_contested + n_attended) / n_feas if n_feas else float("nan"),
        "attendance_of_free": n_attended / (n_feas - n_contested) if n_feas > n_contested else float("nan"),
        "contested_frac": n_contested / n_feas if n_feas else float("nan"),
        "n_feasible_readings": n_feas,
    }


def metrics_of(rec):
    aerial = rec["touched"] & ~rec["on_ground"] & (rec["phys"][:, 2] > 400)
    lb = landing_behavior(rec)
    return {
        **lb,
        "aerial_touch_ratio": float(aerial.mean()),
        "touch_ratio": float(rec["touched"].mean()),
        "in_air_ratio": float((~rec["on_ground"]).mean()),
        "goals_per_episode": rec["goals"] / rec["episodes"],
        "kickoff_first_touch_s": float(np.median(rec["kick_first_touch"]))
        if rec["kick_first_touch"] else float("nan"),
        "n_kickoffs_touched": len(rec["kick_first_touch"]),
    }


def main():
    t0 = time.time()
    torch.set_num_threads(4)
    rs.init(str(HERE.parents[1] / "build" / "collision_meshes"))

    ckpt = copy_checkpoint(HERE / "data" / "ckpt_cache")  # pin ONE checkpoint throughout
    models = load_models(ckpt)
    print(f"pinned checkpoint {ckpt.name}")

    # ---- 1: base dataset + went/declined labels
    base = rollout(SteeredPolicy(models), BASE_ROWS, SEED, want_h2=True)
    phys, episode, team = base["phys"], base["episode"], base["team"]
    airborne = phys[:, 2] > AIRBORNE_Z

    arena = rs.Arena(rs.GameMode.SOCCAR)
    ep_rows = {int(e): np.flatnonzero(episode == e) for e in np.unique(episode)}
    row_pos = np.empty(len(team), np.int64)
    for rows in ep_rows.values():
        row_pos[rows] = np.arange(len(rows))

    rows_r, went_r, dnow_r, tland_r = [], [], [], []
    for r in np.flatnonzero(airborne):
        res = simulate_landing(arena, phys[r, 0:3], phys[r, 3:6], phys[r, 6:9])
        if res is None:
            continue
        lx, ly, t_land = res
        rows = ep_rows[int(episode[r])]
        q = row_pos[r]
        q_land = q + 2 * int(round(t_land / DT))
        if q_land >= len(rows):
            continue
        L = np.array([lx, ly])
        sl = slice(20, 23) if team[r] == 1 else slice(9, 12)
        d_now = np.linalg.norm(phys[rows[q], sl][:2] - L)
        if d_now / max(t_land, 1e-6) >= FEASIBLE_SPEED:
            continue
        ball_td = phys[rows[q_land], 0:3]
        if np.linalg.norm(ball_td[:2] - L) > 300 or ball_td[2] > 200:
            continue
        rows_r.append(r)
        went_r.append(np.linalg.norm(phys[rows[q_land], sl][:2] - L) <= ATTEND_RADIUS)
        dnow_r.append(d_now)
        tland_r.append(t_land)

    rows_r = np.array(rows_r)
    went = np.array(went_r)
    d_now = np.array(dnow_r)
    t_land = np.array(tland_r)
    print(f"feasible free landings: {len(rows_r):,} ({went.mean():.1%} went)")

    # ---- 2: matched difference-of-means direction in h2
    rng = np.random.default_rng(SEED)
    d_edges = np.quantile(d_now, np.linspace(0, 1, MATCH_BINS_D + 1))[1:-1]
    t_edges = np.quantile(t_land, np.linspace(0, 1, MATCH_BINS_T + 1))[1:-1]
    bins = np.digitize(d_now, d_edges) * 10 + np.digitize(t_land, t_edges)
    sel_w, sel_d = [], []
    for b in np.unique(bins):
        w = np.flatnonzero((bins == b) & went)
        d = np.flatnonzero((bins == b) & ~went)
        m = min(len(w), len(d))
        if m == 0:
            continue
        sel_w += list(rng.choice(w, m, replace=False))
        sel_d += list(rng.choice(d, m, replace=False))
    h2b = base["h2"].astype(np.float32)
    Hw, Hd = h2b[rows_r[sel_w]], h2b[rows_r[sel_d]]
    v = Hw.mean(0) - Hd.mean(0)
    effect = float(np.linalg.norm(v) / (0.5 * (Hw.std(0).mean() + Hd.std(0).mean())))
    v /= np.linalg.norm(v)

    proj_all = h2b[rows_r] @ v
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(went[sel_w + sel_d],
                              np.concatenate([Hw @ v, Hd @ v])))
    sigma = float(np.std(h2b.reshape(-1, 512) @ v))
    print(f"direction: matched n={len(sel_w)}/class, effect size {effect:.2f}, "
          f"projection AUC {auc:.3f}, sigma_proj {sigma:.2f}")

    # ---- 3: steered rollouts
    results = {"checkpoint": int(ckpt.name), "n_matched_per_class": len(sel_w),
               "direction_auc": auc, "direction_effect": effect, "alphas": {}}
    for a in ALPHAS:
        pol = SteeredPolicy(models, v, alpha=a, scale=sigma)
        rec = rollout(pol, STEER_ROWS, SEED + 1)  # identical seeds across alphas
        m = metrics_of(rec)
        results["alphas"][str(a)] = m
        print(f"alpha {a:+.1f}: engage {m['engagement']:.1%} (contest {m['contested_frac']:.1%}, "
              f"attend-free {m['attendance_of_free']:.1%}, n={m['n_feasible_readings']}), "
              f"air {m['in_air_ratio']:.1%}, touch {m['touch_ratio']*100:.2f}%, "
              f"goals/ep {m['goals_per_episode']:.2f}, kickoff {m['kickoff_first_touch_s']:.2f}s",
              flush=True)

    results["runtime_s"] = round(time.time() - t0, 1)
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "steering.json", "w") as f:
        json.dump(results, f, indent=2)

    # ---- plot
    keys = ["engagement", "contested_frac", "in_air_ratio", "touch_ratio",
            "goals_per_episode", "kickoff_first_touch_s"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, k in zip(axes.flat, keys):
        xs = [float(a) for a in ALPHAS]
        ys = [results["alphas"][str(a)][k] for a in ALPHAS]
        ax.plot(xs, ys, marker="o")
        ax.axvline(0, color="k", lw=0.5)
        ax.set_title(k)
        ax.set_xlabel("steering alpha (sigma of projection)")
        ax.grid(alpha=0.3)
    fig.suptitle(f"Commitment-direction steering, checkpoint {ckpt.name}")
    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / "steering.png", dpi=120)
    print(f"\nruntime {results['runtime_s']}s; wrote steering.json + steering.png")


if __name__ == "__main__":
    main()
