"""Knowing-doing gap: does the bot act on the landing information its trunk contains?

For each airborne reading (frame with a landing label):
  KNOWING - out-of-fold ridge prediction of the landing point from trunk h1
            (same episode-grouped folds as train_probes), error vs ground truth.
  DOING   - look ahead in the stored rollout to the touchdown time and measure
            behavior: was the ball contested in flight? was any car / the reading's
            own car at the landing spot? did the car even move toward it?

The decision-relevant contrast: among readings where attending was *feasible*
(car could physically arrive before touchdown), compare P(unattended | trunk knew)
vs P(unattended | trunk didn't know).
  - If unattended tracks bad knowledge -> representation bottleneck (Phase 1 pressure).
  - If unattended is flat in knowledge -> the bot knows and declines: incentive /
    exploration bottleneck -> optimism levers (drills, attempt-paying PBRS).

Frames are stored step-major (arena, then player), so within one episode the sorted
rows alternate blue/orange; step index = position // 2, dt = the dataset's decision
step (tick_skip/120, saved by collect_dataset since the 5.0 migration; older archives
lack the field and were all collected at tickSkip 4 = 1/30 s). All geometry here
is WORLD frame (labels are world-frame; probe predictions are de-canonicalized).

phys columns: ball pos 0:3 vel 3:6 angVel 6:9 | blue pos 9:12 vel 12:15 angVel 15:18
boost 18 onGround 19 | orange pos 20:23 vel 23:26 angVel 26:29 boost 29 onGround 30.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from train_probes import ALPHAS, SEED, cv_predict
from sklearn.linear_model import RidgeCV

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
RESULTS_DIR = HERE.parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

KNOW_ERR_UU = 500.0        # trunk "knows" if OOF landing prediction within this
ATTEND_RADIUS_UU = 500.0   # car within this of the landing point at touchdown = attended
CONTEST_DEV_UU = 300.0     # actual ball deviates this much from ball-only landing = touched in flight
FEASIBLE_SPEED = 1300.0    # uu/s straight-line budget (< non-boost max 1410) to call arrival feasible


def dataset_dt(data) -> float:
    """Decision-step seconds of the rollout that wrote dataset.npz. 5.0 datasets
    carry tick_skip (collect_dataset saves it); older archives predate the field
    and were all collected at tickSkip 4 + actionDelay 3 => 1/30 s."""
    return (float(data["tick_skip"]) if "tick_skip" in data.files else 4.0) / 120.0


def main():
    data = np.load(DATA_DIR / "dataset.npz")
    lab = np.load(DATA_DIR / "labels.npz")
    dt = dataset_dt(data)
    team, episode = data["team"], data["episode"]
    phys = data["phys"]
    valid = lab["valid"]
    labels = lab["land_xy_t"]  # world frame

    # ---- per-episode timelines (step-major storage => sorted rows alternate players)
    ep_rows = {}  # episode -> row indices of the blue player, in time order
    order = np.argsort(np.arange(len(team)))  # rows already in time order
    for e in np.unique(episode):
        rows = np.flatnonzero(episode == e)
        ep_rows[int(e)] = rows[team[rows] == 0]  # arena timeline via blue rows

    row_step = np.empty(len(team), np.int64)  # step index within its episode
    for e, rows in ep_rows.items():
        all_rows = np.flatnonzero(episode == e)
        row_step[all_rows] = np.arange(len(all_rows)) // 2

    # ---- KNOWING: OOF ridge on trunk h1, team-canonical, then back to world frame
    y_team = labels[valid].astype(np.float64).copy()
    orange = team[valid] == 1
    y_team[orange, :2] *= -1
    pred_team = cv_predict(data["h1"][valid].astype(np.float64), y_team,
                           episode[valid], lambda: RidgeCV(alphas=ALPHAS))
    pred_world = pred_team.copy()
    pred_world[orange, :2] *= -1

    know_err = np.linalg.norm(pred_world[:, :2] - labels[valid][:, :2], axis=1)

    # ---- DOING: look ahead to touchdown
    v_rows = np.flatnonzero(valid)
    n = len(v_rows)
    contested = np.zeros(n, bool)     # ball touched before (ball-only) touchdown
    censored = np.zeros(n, bool)      # episode ended before touchdown
    d_self = np.full(n, np.nan)       # self car -> landing point at touchdown (2D)
    d_any = np.full(n, np.nan)        # closer car -> landing point
    d_now = np.full(n, np.nan)        # self car -> landing point at reading time (2D)
    approach = np.full(n, np.nan)     # self net displacement toward L over lookahead (uu/s)
    air_before = np.zeros(n, bool)    # self car airborne at any lookahead step pre-touchdown

    for k, r in enumerate(v_rows):
        e = int(episode[r])
        rows_b = ep_rows[e]
        s0 = row_step[r]
        L = labels[r, :2]
        t_land = labels[r, 2]
        s_land = s0 + int(round(t_land / dt))
        if s_land >= len(rows_b):
            censored[k] = True
            continue

        is_orange = team[r] == 1
        self_pos_sl = slice(20, 23) if is_orange else slice(9, 12)
        self_gnd_col = 30 if is_orange else 19

        p_now = phys[rows_b[s0], self_pos_sl][:2]
        d_now[k] = np.linalg.norm(p_now - L)

        ball_td = phys[rows_b[s_land], 0:3]
        contested[k] = np.linalg.norm(ball_td[:2] - L) > CONTEST_DEV_UU or ball_td[2] > 200

        p_td_self = phys[rows_b[s_land], self_pos_sl][:2]
        p_td_blue = phys[rows_b[s_land], 9:12][:2]
        p_td_orng = phys[rows_b[s_land], 20:23][:2]
        d_self[k] = np.linalg.norm(p_td_self - L)
        d_any[k] = min(np.linalg.norm(p_td_blue - L), np.linalg.norm(p_td_orng - L))

        if t_land > 0:
            approach[k] = (d_now[k] - d_self[k]) / t_land
        look = phys[rows_b[s0:s_land + 1], self_gnd_col]
        air_before[k] = (look == 0).any()

    ok = ~censored
    print(f"{n:,} readings, {censored.sum():,} censored (episode ended first), "
          f"{contested[ok].mean():.1%} contested in flight")

    # ---- the gap, on clean free-landing readings where attending was feasible
    free = ok & ~contested
    t_land_all = labels[v_rows, 2]
    feasible = free & (d_now / np.maximum(t_land_all, 1e-6) < FEASIBLE_SPEED)
    knows = know_err < KNOW_ERR_UU

    def p_unattended(mask):
        m = mask & feasible
        return float((d_self[m] > ATTEND_RADIUS_UU).mean()), int(m.sum())

    p_know, n_know = p_unattended(knows)
    p_dumb, n_dumb = p_unattended(~knows)

    print(f"\nfeasible free landings: {feasible.sum():,}")
    print(f"P(self unattended | trunk KNOWS landing <{KNOW_ERR_UU:.0f}uu) = {p_know:.1%}  (n={n_know:,})")
    print(f"P(self unattended | trunk doesn't know)             = {p_dumb:.1%}  (n={n_dumb:,})")

    # graded version: unattendance by knowledge quartile, plus approach speed
    q = np.quantile(know_err[feasible], [0.25, 0.5, 0.75])
    quartile = np.digitize(know_err, q)
    by_q = []
    for i in range(4):
        m = feasible & (quartile == i)
        by_q.append({
            "know_err_max": float(([*q, know_err[feasible].max()])[i]),
            "n": int(m.sum()),
            "p_unattended": float((d_self[m] > ATTEND_RADIUS_UU).mean()),
            "median_d_self": float(np.nanmedian(d_self[m])),
            "median_approach": float(np.nanmedian(approach[m])),
        })
        print(f"  know-err Q{i+1}: p_unattended {by_q[-1]['p_unattended']:.1%}, "
              f"median d_self {by_q[-1]['median_d_self']:5.0f}uu, "
              f"approach {by_q[-1]['median_approach']:5.0f}uu/s (n={by_q[-1]['n']:,})")

    # ---- danger slice: free landing near own goal, defender absent
    own_goal_y = np.where(team[v_rows] == 1, 5120.0, -5120.0)
    danger = feasible & (np.abs(labels[v_rows, 1] - own_goal_y) < 2500) \
                      & (np.abs(labels[v_rows, 0]) < 1500)
    p_danger_unatt = float((d_self[danger] > ATTEND_RADIUS_UU).mean()) if danger.sum() else float("nan")
    print(f"\ndangerous free landings near own goal: {danger.sum():,}, "
          f"self-unattended {p_danger_unatt:.1%}")

    # ---- aerial passivity: long, high balls the bot tracked on the ground
    high = free & (t_land_all > 0.75) & knows & (d_now < 2500)
    p_ground = float((~air_before[high]).mean()) if high.sum() else float("nan")
    print(f"high known balls within 2500uu (aerial-opportunity proxy): {high.sum():,}, "
          f"never airborne before touchdown: {p_ground:.1%}")

    results = {
        "n_readings": int(n), "censored": int(censored.sum()),
        "contested_frac": float(contested[ok].mean()),
        "params": {"know_err_uu": KNOW_ERR_UU, "attend_radius_uu": ATTEND_RADIUS_UU,
                   "contest_dev_uu": CONTEST_DEV_UU, "feasible_speed": FEASIBLE_SPEED},
        "feasible_free_landings": int(feasible.sum()),
        "p_unattended_given_knows": p_know, "n_knows": n_know,
        "p_unattended_given_not_knows": p_dumb, "n_not_knows": n_dumb,
        "by_knowledge_quartile": by_q,
        "danger_slice": {"n": int(danger.sum()), "p_self_unattended": p_danger_unatt},
        "aerial_passivity": {"n": int(high.sum()), "p_never_airborne": p_ground},
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "knowing_doing.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {RESULTS_DIR / 'knowing_doing.json'}")

    # ---- plots
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    labels_q = [f"Q{i+1}" for i in range(4)]
    ax.bar(labels_q, [b["p_unattended"] for b in by_q], color="tab:blue")
    ax.set_xlabel("trunk landing-knowledge quartile (Q1 = best)")
    ax.set_ylabel("P(self car not at landing spot)")
    ax.set_title("Unattendance vs knowledge (feasible free landings)")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    m = feasible
    hb = ax.hexbin(know_err[m], d_self[m], gridsize=40, extent=(0, 2000, 0, 4000),
                   cmap="viridis", mincnt=1)
    ax.set_xlabel("probe landing error (uu) - 'knowing'")
    ax.set_ylabel("self dist to landing at touchdown (uu) - 'doing'")
    ax.set_title("Knowing vs doing, per reading")
    fig.colorbar(hb, ax=ax, label="readings")

    ax = axes[2]
    for name, mask, color in [("knows", feasible & knows, "tab:green"),
                              ("doesn't know", feasible & ~knows, "tab:red")]:
        ax.hist(d_self[mask], bins=60, range=(0, 4000), density=True,
                histtype="step", label=name, color=color)
    ax.axvline(ATTEND_RADIUS_UU, ls="--", c="k", lw=1)
    ax.set_xlabel("self dist to landing at touchdown (uu)")
    ax.set_ylabel("density")
    ax.set_title("Doing, split by knowing")
    ax.legend()

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "knowing_doing.png", dpi=120)
    print(f"wrote {PLOTS_DIR / 'knowing_doing.png'}")


if __name__ == "__main__":
    main()
