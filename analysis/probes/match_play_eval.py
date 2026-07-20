"""MATCH-PLAY head-to-head: kickoff-only resets, goal-only terminals - the game
the viz viewer (and real matches) actually show.

Why this exists (2026-07-19): the corrected compare_checkpoints cross-play runs
on the TRAINER-PARITY reset mix (35% ball-near-car / 20% air-drill / 15% kickoff
/ 30% random, 30s caps) - training-distribution play, not matches. The user's
"looks much better in viz" impression is about MATCH play. A policy can win the
drill mix and lose match flow or vice versa. This script settles which policy
wins the game we actually care about.

Protocol: both policies wire-zero (PulsarPolicy pads 517 heads automatically),
sides swapped halfway, kickoff after every goal (match flow), 90s no-goal safety
cap (viz has none; offline needs one - capped episodes count no goals).
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

import RocketSim as rs
from collect_dataset import NUM_ARENAS, set_obs_size
from compare_checkpoints import ScoringEnv
from load_checkpoint import PulsarPolicy, load_models
from steer_test import SteeredPolicy

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
SEED = 20260719
ROWS_PER_HALF = 100_000
CAP_STEPS = 90 * 15  # 90s at 15Hz; goal-only otherwise


class MatchEnv(ScoringEnv):
    """Kickoff-only resets, goal-only terminal (+ safety cap). Match flow."""

    def reset(self, episode_counter=[0]):
        self.arena.reset_kickoff(seed=int(self.rng.integers(0, 2**30)))
        zero = rs.CarControls()
        for car in self.cars:
            car.set_controls(zero)
        self.prev_actions = np.zeros((2, 8), dtype=np.float32)
        self.goal_scored = False
        self.steps = 0
        self.last_touch_tick = self.arena.tick_count
        self.is_kickoff = True
        self.episode_id = episode_counter[0]
        episode_counter[0] += 1

    def step(self, action_indices):
        done = super().step(action_indices)
        # override the mix terminals: goal OR safety cap only
        return self.goal_scored or self.steps >= CAP_STEPS


def cross_match(pol_b, pol_o, n_rows, seed):
    torch.manual_seed(seed)
    envs = [MatchEnv(i, np.random.default_rng(seed + 10 + i)) for i in range(NUM_ARENAS)]
    episodes = capped = 0
    rows = 0
    while rows < n_rows:
        obs_list, mask_list = [], []
        for env in envs:
            obs, masks, _ = env.observe()
            obs_list.append(obs)
            mask_list.append(masks)
        obs_b = torch.from_numpy(np.concatenate(obs_list))
        mask_b = torch.from_numpy(np.concatenate(mask_list))
        blue = torch.arange(0, len(obs_b), 2)
        orange = torch.arange(1, len(obs_b), 2)
        actions = torch.empty(len(obs_b), dtype=torch.long)
        _, ab = pol_b.act(obs_b[blue], mask_b[blue])
        _, ao = pol_o.act(obs_b[orange], mask_b[orange])
        actions[blue], actions[orange] = ab, ao
        rows += len(obs_b)
        for i, env in enumerate(envs):
            if env.step(actions[2 * i:2 * i + 2].tolist()):
                episodes += 1
                capped += int(not env.goal_scored)
                env.reset()
    goals = [sum(e.team_goals[0] for e in envs), sum(e.team_goals[1] for e in envs)]
    return goals, episodes, capped


def main():
    a_dir, b_dir = Path(sys.argv[1]), Path(sys.argv[2])
    torch.set_num_threads(4)
    rs.init(str(HERE.parents[1] / "build" / "collision_meshes"))
    A = load_models(a_dir)
    B = load_models(b_dir)
    pa, pb = PulsarPolicy(A), PulsarPolicy(B)
    assert pa.obs_size == pb.obs_size
    set_obs_size(pa.obs_size)
    print(f"MATCH PLAY: A={a_dir.name} vs B={b_dir.name} (obs {pa.obs_size}, "
          f"wire pads A={pa.wire_pad} B={pb.wire_pad} -> zeros)\n")

    ga = gb = eps = caps = 0
    for half in range(2):
        pA, pB = SteeredPolicy(A), SteeredPolicy(B)
        blue, orange = (pA, pB) if half == 0 else (pB, pA)
        goals, e, c = cross_match(blue, orange, ROWS_PER_HALF, SEED + half)
        a_g = goals[0] if half == 0 else goals[1]
        b_g = goals[1] if half == 0 else goals[0]
        ga += a_g; gb += b_g; eps += e; caps += c
        print(f"half {half+1}: A {a_g} - {b_g} B  ({e} kickoff-episodes, {c} capped no-goal)")
    tot = max(ga + gb, 1)
    print(f"\nMATCH TOTAL: A {ga} - {gb} B  -> A share {ga/tot:.0%} over {eps} episodes ({caps} capped)")
    RESULTS_DIR.mkdir(exist_ok=True)
    out = {"A": a_dir.name, "B": b_dir.name, "A_goals": ga, "B_goals": gb,
           "episodes": eps, "capped": caps, "A_share": ga / tot}
    with open(RESULTS_DIR / f"match_{a_dir.name}_vs_{b_dir.name}.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
