"""Quantify the 'looked better in viz' impression: baseline checkpoint vs a
steered-practice-trained (quarantined) checkpoint.

Two measurements, both offline CPU (live trainer untouched):
  1. MIRROR self-play per checkpoint: per-policy style panel (engagement with
     feasible landings, in-flight contests, aerial touches, air time, kickoffs).
  2. CROSS-PLAY head-to-head (sides swapped halfway): who actually wins, plus
     per-policy style within the same matches.

Usage: compare_checkpoints.py <baseline_ckpt_dir> <treated_ckpt_dir>
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

import RocketSim as rs
from collect_dataset import NUM_ARENAS, ArenaEnv, set_obs_size
from load_checkpoint import PulsarPolicy, load_models
from steer_test import SteeredPolicy, landing_behavior, metrics_of, rollout

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
SEED = 777
MIRROR_ROWS = 60_000
CROSS_ROWS_PER_SIDE = 40_000


class ScoringEnv(ArenaEnv):
    """ArenaEnv that records which team scored (goal callback passes team=0/1)."""

    def __init__(self, idx, rng):
        self.team_goals = [0, 0]
        super().__init__(idx, rng)

    def _on_goal(self, **kwargs):
        self.goal_scored = True
        t = kwargs.get("team", kwargs.get("scoring_team", None))
        if t is not None:
            self.team_goals[int(t)] += 1


def cross_play(models_blue, models_orange, n_rows, seed):
    """Two policies in the same arenas: blue rows acted by models_blue, orange by
    models_orange. Returns per-team behavior rec + goal tally."""
    torch.manual_seed(seed)
    pol_b = SteeredPolicy(models_blue)
    pol_o = SteeredPolicy(models_orange)
    envs = [ScoringEnv(i, np.random.default_rng(seed + 10 + i)) for i in range(NUM_ARENAS)]

    rec = {
        "phys": np.empty((n_rows, 31), np.float32),
        "episode": np.empty(n_rows, np.int32),
        "team": np.empty(n_rows, np.int8),
        "touched": np.zeros(n_rows, bool),
        "on_ground": np.zeros(n_rows, bool),
        "kickoff": np.zeros(n_rows, bool),
    }
    last_hit = [dict() for _ in envs]
    episodes = 0
    row = 0
    while row + 2 * NUM_ARENAS <= n_rows:
        obs_list, mask_list, metas = [], [], []
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
                per_player.append((touched, st.is_on_ground))
            metas.append((phys, per_player, env.episode_id))

        obs_b = torch.from_numpy(np.concatenate(obs_list))
        mask_b = torch.from_numpy(np.concatenate(mask_list))
        # row layout is [arena0-blue, arena0-orange, arena1-blue, ...]
        blue_rows = torch.arange(0, len(obs_b), 2)
        orange_rows = torch.arange(1, len(obs_b), 2)
        actions = torch.empty(len(obs_b), dtype=torch.long)
        _, h2b = pol_b.trunk_forward(obs_b[blue_rows])
        actions[blue_rows] = pol_b.sample_actions(h2b, mask_b[blue_rows])
        _, h2o = pol_o.trunk_forward(obs_b[orange_rows])
        actions[orange_rows] = pol_o.sample_actions(h2o, mask_b[orange_rows])

        for i, env in enumerate(envs):
            phys, per_player, ep = metas[i]
            for p in range(2):
                rec["phys"][row] = phys
                rec["episode"][row] = ep
                rec["team"][row] = p
                rec["touched"][row] = per_player[p][0]
                rec["on_ground"][row] = per_player[p][1]
                row += 1
        for i, env in enumerate(envs):
            if env.step(actions[2 * i:2 * i + 2].tolist()):
                episodes += 1
                env.reset()

    for k in rec:
        rec[k] = rec[k][:row]
    goals = [sum(e.team_goals[0] for e in envs), sum(e.team_goals[1] for e in envs)]
    return rec, goals, episodes


def team_stats(rec, t):
    m = rec["team"] == t
    aerial = rec["touched"] & ~rec["on_ground"] & (rec["phys"][:, 2] > 400)
    lb = landing_behavior(rec, team_filter=t)
    return {
        **{k: lb[k] for k in ("engagement", "contested_frac", "n_feasible_readings")},
        "aerial_touch_ratio": float(aerial[m].mean()),
        "touch_ratio": float(rec["touched"][m].mean()),
        "in_air_ratio": float((~rec["on_ground"][m]).mean()),
    }


def main():
    base_dir, treat_dir = Path(sys.argv[1]), Path(sys.argv[2])
    torch.set_num_threads(4)
    rs.init(str(HERE.parents[1] / "build" / "collision_meshes"))

    base = load_models(base_dir)
    treat = load_models(treat_dir)
    # 5.0 lineage is 230-dim padded obs; ArenaEnv builds 109-dim unless told (steer_test
    # /compare predate the 4.0/5.0 padded lineage and never called set_obs_size). Both
    # policies share the same arenas, so their trunk obs width must match.
    b_obs, t_obs = PulsarPolicy(base).obs_size, PulsarPolicy(treat).obs_size
    assert b_obs == t_obs, f"obs width mismatch {b_obs} vs {t_obs} - can't share arenas"
    set_obs_size(b_obs)
    print(f"baseline {base_dir.name} vs treated {treat_dir.name}  (obs {b_obs})\n")

    results = {"baseline": base_dir.name, "treated": treat_dir.name}

    # 1) mirror self-play style panels
    for name, models in [("baseline", base), ("treated", treat)]:
        rec = rollout(SteeredPolicy(models), MIRROR_ROWS, SEED)
        m = metrics_of(rec)
        results[f"mirror_{name}"] = m
        print(f"mirror {name:9s}: engage {m['engagement']:.1%} (contest {m['contested_frac']:.1%}), "
              f"aerial {m['aerial_touch_ratio']*100:.3f}%, touch {m['touch_ratio']*100:.2f}%, "
              f"air {m['in_air_ratio']:.1%}, goals/ep {m['goals_per_episode']:.2f}, "
              f"kickoff {m['kickoff_first_touch_s']:.2f}s")

    # 2) cross-play, sides swapped halfway
    tally = {"baseline": 0, "treated": 0}
    stats_acc = {"baseline": [], "treated": []}
    eps_total = 0
    for half, (mb, mo, bname, oname) in enumerate([
            (base, treat, "baseline", "treated"),
            (treat, base, "treated", "baseline")]):
        rec, goals, eps = cross_play(mb, mo, CROSS_ROWS_PER_SIDE, SEED + 50 + half)
        tally[bname] += goals[0]
        tally[oname] += goals[1]
        eps_total += eps
        stats_acc[bname].append(team_stats(rec, 0))
        stats_acc[oname].append(team_stats(rec, 1))
        print(f"cross half {half+1}: {bname}(blue) {goals[0]} - {goals[1]} {oname}(orange), "
              f"{eps} episodes")

    results["cross_goals"] = tally
    results["cross_episodes"] = eps_total
    for name in ("baseline", "treated"):
        avg = {k: float(np.mean([s[k] for s in stats_acc[name]]))
               for k in stats_acc[name][0]}
        results[f"cross_{name}"] = avg
        print(f"cross {name:9s}: engage {avg['engagement']:.1%} (contest {avg['contested_frac']:.1%}), "
              f"aerial {avg['aerial_touch_ratio']*100:.3f}%, touch {avg['touch_ratio']*100:.2f}%, "
              f"air {avg['in_air_ratio']:.1%}")

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "checkpoint_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {RESULTS_DIR / 'checkpoint_comparison.json'}")


if __name__ == "__main__":
    main()
