"""Kickoff health probe for a 5.3 checkpoint (the offline twin of the C++ boot sanity
probe): N kickoff-only episodes, report touch rate + time-to-first-touch. Healthy
lineage ≈ 10/10 touches, median ~3-4s; a garbled forward (wrong residual spans,
missing trailing activation) produces aimless play and fails this decisively."""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

import RocketSim as rs
import collect_dataset as cd
from collect_dataset import ArenaEnv, set_obs_size
from load_checkpoint_53 import Pulsar53Policy, load_models, list_checkpoints

SEED = 777


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(int(os.environ.get("PULSAR_THREADS", 4)))
    root = Path(os.environ.get("PULSAR_CKPT_ROOT",
                               Path(__file__).resolve().parents[1] / "data" / "ckpt53"))
    ckpt_dir = Path(sys.argv[sys.argv.index("--ckpt") + 1]) if "--ckpt" in sys.argv \
        else list_checkpoints(root)[0]
    n_eps = int(os.environ.get("KICKOFF_EPS", 12))

    rs.init(str(Path(__file__).resolve().parents[2] / "build" / "collision_meshes"))
    policy = Pulsar53Policy(load_models(ckpt_dir, ["SHARED_HEAD", "POLICY", "REACH_PHI"]))
    set_obs_size(policy.obs_size)
    cd.RESET_MIX = [(1.01, "kickoff")]

    env = ArenaEnv(0, np.random.default_rng(SEED))
    results = []
    for ep in range(n_eps):
        env.reset()
        base_touch_tick = env.last_touch_tick
        touched_at = None
        for step in range(int(20.0 / cd.DT)):
            obs, masks, phys = env.observe()
            h1, h2 = policy.trunk_forward(torch.from_numpy(obs))
            acts = policy.sample_actions(h2, torch.from_numpy(masks))
            done = env.step(acts.tolist())
            env.observe()  # refresh last_touch_tick from ball_hit_info
            if env.last_touch_tick > base_touch_tick and touched_at is None:
                touched_at = (step + 1) * cd.DT
            if touched_at is not None or done:
                break
        results.append(touched_at)
        print(f"  ep {ep}: {'touch @ %.2fs' % touched_at if touched_at is not None else 'NO TOUCH'}")

    hits = [r for r in results if r is not None]
    print(f"\ntouches: {len(hits)}/{n_eps}   median t {np.median(hits):.2f}s" if hits
          else f"\ntouches: 0/{n_eps}  -- POLICY IS BROKEN OR LOADER IS WRONG")


if __name__ == "__main__":
    main()
