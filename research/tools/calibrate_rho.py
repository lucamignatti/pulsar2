"""Calibration study of the reachability critic rho: is the bot's self-model of
capability trustworthy enough to steer a frontier curriculum?

rho(s -> g) = mean over K uniform VALID actions of cosine(phi(trunk(s), a), psi(g)) / tau
(EvalRho parity: uniform actions = capability "can we", not policy "would we").

The heads are trained InfoNCE + HER where positives are FUTURE ACHIEVED states within
a window (car head: car-local ball pos+vel / 2300, offsets 1..20 steps; ball head:
canonical ball pos+vel, offsets 1..90). Ground truth for calibration is therefore the
same event: does the trajectory actually pass within eps of goal g in the next W steps
of ordinary play? We collect fresh self-play, then for each anchor state score a goal
menu spanning difficulty:
  - same-episode future achieved states (reachable by construction),
  - those + Gaussian jitter in goal space (intermediate),
  - achieved states stolen from other episodes (mostly unreachable).
Reported per head: success-vs-rho calibration curve, ROC AUC, Spearman rho->success.

Note what this does and does NOT measure: HER trains "will I reach g in ordinary
play", and that is what we calibrate. The policy takes no goal input, so a
"deliberately trying for g" calibration does not exist for this architecture - that
gap is exactly what a proposer/goal-conditioning mechanism would add.
"""

import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import RocketSim as rs
from collect_dataset import NUM_ARENAS, SEED, ArenaEnv
from load_checkpoint import load_latest

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE.parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

TARGET_ROWS = int(os.environ.get("RHO_ROWS", 60_000))
N_ANCHORS = 12_000
K_ACTIONS = 16          # EvalRho's numActionSamples
TAU = 0.02
CAR_WINDOW, BALL_WINDOW = 20, 90        # carHerMaxOffset / ballHerMaxOffset
CAR_LOCAL_SCALE = 2300.0
POS_SCALE = np.array([4096.0, 6000.0, 2044.0])
VEL_SCALE = 6000.0
EPS_SWEEP = [0.075, 0.15, 0.30]         # achievement radius in normalized goal space
GOALS_PER_ANCHOR = {"future": 3, "jitter": 3, "cross": 3}
JITTER_SIGMA = [0.08, 0.2, 0.5]


def car_ball_goal(ball_pos, ball_vel, car_pos, car_vel, fwd, right, up):
    """achievedCarBall parity: car-local ball rel pos+vel / carLocalScale (inversion-invariant)."""
    rel_p, rel_v = ball_pos - car_pos, ball_vel - car_vel
    basis = np.stack([fwd, right, up], -2)  # [..., 3, 3]
    return np.concatenate([
        (basis @ rel_p[..., None])[..., 0] / CAR_LOCAL_SCALE,
        (basis @ rel_v[..., None])[..., 0] / CAR_LOCAL_SCALE,
    ], -1)


def ball_goal(ball_pos, ball_vel, is_orange):
    """achievedBall parity: canonical (orange x,y negated) normalized ball pos+vel."""
    p = ball_pos / POS_SCALE
    v = ball_vel / VEL_SCALE
    p[..., 0] *= np.where(is_orange, -1.0, 1.0)
    p[..., 1] *= np.where(is_orange, -1.0, 1.0)
    v[..., 0] *= np.where(is_orange, -1.0, 1.0)
    v[..., 1] *= np.where(is_orange, -1.0, 1.0)
    return np.concatenate([p, v], -1)


def collect(policy):
    """Self-play pass storing, per player-frame: trunk h2, action mask, both goal-space
    achieved vectors, and episode/step ids."""
    rng = np.random.default_rng(SEED + 77)
    torch.manual_seed(SEED + 77)
    envs = [ArenaEnv(i, np.random.default_rng(SEED + 500 + i)) for i in range(NUM_ARENAS)]

    n = TARGET_ROWS
    out = {
        "h2": np.empty((n, getattr(policy, "h2_width", 512)), np.float32),
        "mask": np.empty((n, 90), np.uint8),
        "goal_car": np.empty((n, 6), np.float32),
        "goal_ball": np.empty((n, 6), np.float32),
        "episode": np.empty(n, np.int32),
        "team": np.empty(n, np.int8),
    }
    row = 0
    t0 = time.time()
    while row + 2 * NUM_ARENAS <= n:
        obs_list, mask_list = [], []
        ach_car, ach_ball = [], []
        for env in envs:
            obs, masks, _ = env.observe()
            obs_list.append(obs)
            mask_list.append(masks)
            ball = env.arena.ball.get_state()
            bp = np.array(ball.pos.as_tuple())
            bv = np.array(ball.vel.as_tuple())
            for p, car in enumerate(env.cars):
                st = car.get_state()
                cp, cv = np.array(st.pos.as_tuple()), np.array(st.vel.as_tuple())
                rm = st.rot_mat
                g_car = car_ball_goal(bp, bv, cp, cv,
                                      np.array(rm.forward.as_tuple()),
                                      np.array(rm.right.as_tuple()),
                                      np.array(rm.up.as_tuple()))
                g_ball = ball_goal(bp.copy(), bv.copy(), np.array(p == 1))
                ach_car.append(g_car)
                ach_ball.append(g_ball)

        obs_b = torch.from_numpy(np.concatenate(obs_list))
        mask_b = torch.from_numpy(np.concatenate(mask_list))
        _, h2 = policy.trunk_forward(obs_b)
        actions = policy.sample_actions(h2, mask_b)

        for i, env in enumerate(envs):
            for p in range(2):
                j = 2 * i + p
                out["h2"][row] = h2[j].numpy()
                out["mask"][row] = mask_b[j].numpy()
                out["goal_car"][row] = ach_car[j]
                out["goal_ball"][row] = ach_ball[j]
                out["episode"][row] = env.episode_id
                out["team"][row] = p
                row += 1
        for i, env in enumerate(envs):
            if env.step(actions[2 * i:2 * i + 2].tolist()):
                env.reset()
    print(f"collected {row:,} rows in {time.time()-t0:.0f}s, {out['episode'].max()+1} episodes")
    for k in out:
        out[k] = out[k][:row]
    return out


def rho_batched(policy, h2, masks, goals_flat, anchor_of_goal, rng):
    """rho for each (anchor, goal) pair: mean over K uniform valid actions of cos/tau.

    phi embeddings are computed once per anchor (K rows each); each goal then costs one
    dot product against its anchor's K embeddings.
    """
    n_anchor = len(h2)
    acts = np.empty((n_anchor, K_ACTIONS), np.int64)
    for i in range(n_anchor):
        valid = np.flatnonzero(masks[i])
        acts[i] = rng.choice(valid, K_ACTIONS, replace=True)

    sa = np.empty((n_anchor, K_ACTIONS, 128), np.float32)
    B = 4096
    with torch.no_grad():
        h2_t = torch.from_numpy(h2)
        for s in range(0, n_anchor, B):
            e = min(s + B, n_anchor)
            block = h2_t[s:e].repeat_interleave(K_ACTIONS, 0)
            a = torch.from_numpy(acts[s:e].reshape(-1))
            emb = policy.phi_embedding(block, a)
            sa[s:e] = emb.reshape(e - s, K_ACTIONS, 128).numpy()

        # psi over all goals (both heads share this helper; head passed via policy attr)
        g = torch.from_numpy(goals_flat)
        psi_out = policy._psi_head(g)
        psi_out = (psi_out / psi_out.norm(dim=-1, keepdim=True).clamp_min(1e-6)).numpy()

    scores = np.einsum("gk,gak->ga", psi_out, sa[anchor_of_goal]) / TAU
    return scores.mean(-1)


def run_head(policy, data, head_name, psi, window, rng):
    """Build goal menu, ground-truth achievement, rho, calibration stats for one head."""
    goal_key = "goal_car" if head_name == "car" else "goal_ball"
    goals_all = data[goal_key]
    episode = data["episode"]
    n = len(goals_all)

    # Rows are step-major ACROSS ARENAS (arena0-blue, arena0-orange, arena1-blue, ...),
    # so "k steps later, same player" is NOT row + 2k globally. Build per-episode row
    # lists (global order = time order within an episode; positions alternate
    # blue/orange), then a player's step-k future is position + 2k in that list.
    ep_rows = {int(e): np.flatnonzero(episode == e) for e in np.unique(episode)}
    row_pos = np.empty(n, np.int64)
    for rows in ep_rows.values():
        row_pos[rows] = np.arange(len(rows))

    anchors = rng.choice(n, min(N_ANCHORS, n), replace=False)
    keep = np.array([row_pos[a] + 2 * window < len(ep_rows[int(episode[a])]) for a in anchors])
    anchors = anchors[keep]

    def future_goal(a, k):
        return goals_all[ep_rows[int(episode[a])][row_pos[a] + 2 * k]]

    menu_goal, menu_anchor, menu_kind = [], [], []
    for a in anchors:
        # future achieved (short-biased like HER: offset ~ U^2)
        for _ in range(GOALS_PER_ANCHOR["future"]):
            k = max(1, int((rng.random() ** 2) * window))
            menu_goal.append(future_goal(a, k))
            menu_anchor.append(a)
            menu_kind.append("future")
        for sig in JITTER_SIGMA:
            k = max(1, int((rng.random() ** 2) * window))
            menu_goal.append(future_goal(a, k) + rng.normal(0, sig, 6).astype(np.float32))
            menu_anchor.append(a)
            menu_kind.append("jitter")
        for _ in range(GOALS_PER_ANCHOR["cross"]):
            menu_goal.append(goals_all[rng.integers(0, n)])
            menu_anchor.append(a)
            menu_kind.append("cross")

    menu_goal = np.asarray(menu_goal, np.float32)
    menu_anchor = np.asarray(menu_anchor)
    menu_kind = np.asarray(menu_kind)

    # ground truth: min normalized 6D distance to goal over the window (same-player rows,
    # offsets 1..window like HER - offset 0 would let near-current goals self-achieve)
    future_block = {}
    for a in np.unique(menu_anchor):
        rows = ep_rows[int(episode[a])]
        q = row_pos[a]
        future_block[a] = goals_all[rows[q + 2:q + 2 * window + 1:2]]
    achieved_min = np.empty(len(menu_goal), np.float32)
    for i, (a, g) in enumerate(zip(menu_anchor, menu_goal)):
        achieved_min[i] = np.linalg.norm(future_block[a] - g, axis=1).min()

    # rho for every (anchor, goal)
    anchor_ids, anchor_inv = np.unique(menu_anchor, return_inverse=True)
    policy._psi_head = psi
    rho = rho_batched(policy, data["h2"][anchor_ids], data["mask"][anchor_ids],
                      menu_goal, anchor_inv, rng)

    # stats per epsilon
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr
    out = {"n_pairs": int(len(rho)), "n_anchors": int(len(anchors)), "window_steps": window,
           "kinds": {k: int((menu_kind == k).sum()) for k in ("future", "jitter", "cross")},
           "rho_by_kind": {k: {"mean": float(rho[menu_kind == k].mean()),
                               "std": float(rho[menu_kind == k].std())}
                           for k in ("future", "jitter", "cross")},
           "eps": {}}
    fut, cro = rho[menu_kind == "future"], rho[menu_kind == "cross"]
    print(f"  [{head_name}] rho future {fut.mean():.1f}±{fut.std():.1f} vs cross {cro.mean():.1f}±{cro.std():.1f}")
    curves = {}
    for eps in EPS_SWEEP:
        succ = achieved_min < eps
        auc = float(roc_auc_score(succ, rho)) if 0 < succ.mean() < 1 else float("nan")
        sp = float(spearmanr(rho, -achieved_min).statistic)
        # calibration curve over rho deciles
        edges = np.quantile(rho, np.linspace(0, 1, 11))
        bins = np.clip(np.digitize(rho, edges[1:-1]), 0, 9)
        curve = [float(succ[bins == b].mean()) for b in range(10)]
        out["eps"][str(eps)] = {"success_rate": float(succ.mean()), "auc": auc,
                                "spearman_rho_vs_proximity": sp, "decile_success": curve}
        curves[eps] = (edges, curve)
        print(f"  [{head_name}] eps={eps}: success {succ.mean():.1%}, AUC {auc:.3f}, "
              f"spearman {sp:.3f}")
    return out, curves, rho, achieved_min, menu_kind


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    rng = np.random.default_rng(SEED + 9)
    rs.init(str(HERE.parents[1] / "build" / "collision_meshes"))
    policy, ckpt = load_latest()
    print(f"checkpoint {ckpt.name}")
    # padded-lineage (230-obs) checkpoints need the env obs builder switched over
    import collect_dataset as _cd
    _cd.set_obs_size(policy.obs_size)

    t0 = time.time()
    data = collect(policy)

    results = {"checkpoint": int(ckpt.name), "heads": {}}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, (head, psi_name, window) in zip(axes, [
            ("car", "REACH_PSI_CAR", CAR_WINDOW),
            ("ball", "REACH_PSI_BALL", BALL_WINDOW)]):
        print(f"head: {head} (window {window} steps)")
        stats, curves, rho, dmin, kind = run_head(
            policy, data, head, policy.models[psi_name], window, rng)
        results["heads"][head] = stats

        for eps, (edges, curve) in curves.items():
            centers = 0.5 * (edges[:-1] + edges[1:])
            ax.plot(centers, curve, marker="o", ms=3, label=f"eps={eps}")
        ax.set_xlabel("rho decile center (cos/tau logits)")
        ax.set_ylabel(f"P(achieved within {window} steps)")
        ax.set_title(f"reach_{head}: success vs rho")
        ax.legend()
        ax.grid(alpha=0.3)

    results["runtime_s"] = round(time.time() - t0, 1)
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "rho_calibration.json", "w") as f:
        json.dump(results, f, indent=2)
    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / "rho_calibration.png", dpi=120)
    print(f"\nruntime {results['runtime_s']}s; wrote rho_calibration.json + plot")


if __name__ == "__main__":
    main()
