"""Self-play rollout collector for the 5.3 (four-rung ladder) lineage.

Reuses collect_dataset.py's ArenaEnv + state-setter ports wholesale (5.0 dynamics:
tickSkip 8, actionDelay 0), with three changes:
  1. loads through load_checkpoint_53 (residual arch; ladder heads),
  2. reset mix updated to the LIVE trainer mix (ExampleMain MakeEnv, 5.x):
       BallNearCar 0.30 / AirDrill 0.20 / AirPlay 0.15 / Kickoff 0.10 / Random 0.25
     including a 1v1 port of AirPlayState (flip-reset-ready / air-carry seeding),
  3. records the full critic ladder per player-frame (v_real, v_exp, vdag1/2, v_goal,
     v_geo, r_hat) next to the activations, so cross-critic analyses don't need a
     second model pass.

Output research/data/dataset53.npz. Layout matches dataset.npz (obs/h1/h2/phi/action/
team/episode/arena/phys) with h1/h2 now 1152-wide, plus ladder_* arrays and touched.

Usage:  PULSAR_CKPT_ROOT=research/data/ckpt53 PROBE_FRAMES=120000 python collect_dataset_53.py
        [--ckpt <dir>] to point at a specific checkpoint dir (e.g. a ref_* snapshot).
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

import RocketSim as rs
import collect_dataset as cd
from collect_dataset import (ArenaEnv, BALL_RADIUS, SIDE_WALL_X, BACK_WALL_Y,
                             set_obs_size)
from load_checkpoint_53 import Pulsar53Policy, load_models, list_checkpoints

SEED = 20260801
NUM_ARENAS = 16
TARGET_FRAMES = int(os.environ.get("PROBE_FRAMES", 120_000))
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def set_air_play_state(arena, rng, carry_speed=900.0, reset_frac=0.5):
    """1v1 port of AirPlayState::ResetArena (both cars are air-players, shared ball)."""
    arena.reset_kickoff(seed=int(rng.integers(0, 2**30)))
    flip_reset = rng.random() < reset_frac
    attack_sign = 1.0 if rng.integers(0, 2) else -1.0

    bs = rs.BallState()
    if flip_reset:
        ball = np.array([rng.uniform(-2200, 2200), rng.uniform(-3000, 3000),
                         rng.uniform(1200, 1700)])
        bvel = np.array([rng.uniform(-150, 150), rng.uniform(-150, 150),
                         rng.uniform(-250, 50)])
    else:
        ball = np.array([rng.uniform(-2000, 2000),
                         attack_sign * rng.uniform(-500, 1500), rng.uniform(350, 800)])
        bvel = np.array([rng.uniform(-200, 200), attack_sign * carry_speed,
                         rng.uniform(150, 450)])
    bs.pos, bs.vel = rs.Vec(*ball), rs.Vec(*bvel)
    arena.ball.set_state(bs)

    clamp_x, clamp_y = SIDE_WALL_X - 300, BACK_WALL_Y - 300
    placed = []
    for car in arena.get_cars():
        if flip_reset:
            drop = rng.uniform(300, 520)
            pos = ball + np.array([rng.uniform(-200, 200), rng.uniform(-200, 200), -drop])
        else:
            pos = ball - np.array([0.0, attack_sign * rng.uniform(250, 500),
                                   rng.uniform(60, 200)])
        pos[0] = np.clip(pos[0], -clamp_x, clamp_x)
        pos[1] = np.clip(pos[1], -clamp_y, clamp_y)
        pos[2] = max(300.0, pos[2])
        if any(np.linalg.norm(pos - o) < 350 for o in placed):
            pos[2] += 350.0
        placed.append(pos.copy())

        to_ball = ball - pos
        tb = to_ball / max(np.linalg.norm(to_ball), 1e-9)

        cs = rs.CarState()
        cs.pos = rs.Vec(*pos)
        if flip_reset:
            # roof toward the ball: forward horizontal, up = toBall (LookAt parity)
            fwd = np.array([tb[1], -tb[0], 0.0])
            if np.linalg.norm(fwd) < 0.1:
                fwd = np.array([1.0, 0.0, 0.0])
            f = fwd / np.linalg.norm(fwd)
            u0 = tb
            r = np.cross(u0, f); r /= max(np.linalg.norm(r), 1e-9)
            u = np.cross(f, r); u /= max(np.linalg.norm(u), 1e-9)
            # RotMat::LookAt(forward, upHint): forward kept, up orthogonalized
            cs.rot_mat = rs.RotMat(rs.Vec(*f), rs.Vec(*r), rs.Vec(*u))
            cs.has_jumped = True
            cs.has_flipped = True
            cs.air_time = 0.6
            cs.air_time_since_jump = 0.6
            cs.vel = rs.Vec(*(tb * rng.uniform(300, 650)))
        else:
            f = tb
            tr = np.cross([0.0, 0.0, 1.0], f)
            u = np.cross(f, tr); u /= max(np.linalg.norm(u), 1e-9)
            r = np.cross(u, f); r /= max(np.linalg.norm(r), 1e-9)
            cs.rot_mat = rs.RotMat(rs.Vec(*f), rs.Vec(*r), rs.Vec(*u))
            cs.vel = rs.Vec(*(np.array(bvel) + f * rng.uniform(50, 200)))
        cs.boost = rng.uniform(50, 100)
        car.set_state(cs)


# Live trainer reset mix (ExampleMain MakeEnv @ 5.3): cumulative weights
LIVE_RESET_MIX = [(0.30, "near"), (0.50, "air"), (0.65, "airplay"),
                  (0.75, "kickoff"), (1.01, "random")]

_orig_reset = ArenaEnv.reset


def _reset_with_airplay(self, episode_counter=[0]):
    u = self.rng.random()
    kind = next(k for w, k in cd.RESET_MIX if u < w)
    if kind != "airplay":
        # replay the same draw deterministically through the original dispatch,
        # then unwrap so wrappers don't chain across episodes
        real_rng = self.rng
        self.rng = _StackedRng(u, real_rng)
        try:
            return _orig_reset(self, episode_counter)
        finally:
            self.rng = real_rng
    self.is_kickoff = False
    set_air_play_state(self.arena, self.rng)
    zero = rs.CarControls()
    for car in self.cars:
        car.set_controls(zero)
    self.prev_actions = np.zeros((2, 8), dtype=np.float32)
    self.goal_scored = False
    self.steps = 0
    self.last_touch_tick = self.arena.tick_count
    self.episode_id = episode_counter[0]
    episode_counter[0] += 1


class _StackedRng:
    """Feed one pre-drawn uniform back to the original reset's mix draw, then
    delegate everything else to the real rng."""

    def __init__(self, first_u, rng):
        self._first = first_u
        self._rng = rng

    def random(self):
        if self._first is not None:
            u, self._first = self._first, None
            return u
        return self._rng.random()

    def __getattr__(self, name):
        return getattr(self._rng, name)


def load_policy(ckpt_dir: Path) -> Pulsar53Policy:
    return Pulsar53Policy(load_models(ckpt_dir))


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(int(os.environ.get("PULSAR_THREADS", 4)))
    rng = np.random.default_rng(SEED)

    ckpt_dir = None
    if "--ckpt" in sys.argv:
        ckpt_dir = Path(sys.argv[sys.argv.index("--ckpt") + 1])
    else:
        root = Path(os.environ.get("PULSAR_CKPT_ROOT",
                                   Path(__file__).resolve().parents[1] / "data" / "ckpt53"))
        ckpt_dir = list_checkpoints(root)[0]

    rs.init(str(Path(__file__).resolve().parents[2] / "build" / "collision_meshes"))
    policy = load_policy(ckpt_dir)
    set_obs_size(policy.obs_size)
    cd.RESET_MIX = LIVE_RESET_MIX
    ArenaEnv.reset = _reset_with_airplay
    print(f"policy checkpoint: {ckpt_dir.name} (obs {policy.obs_size}), live reset mix + AirPlay")

    envs = [ArenaEnv(i, np.random.default_rng(SEED + 1000 + i)) for i in range(NUM_ARENAS)]

    n = TARGET_FRAMES
    width = policy.trunk.out_features
    ladder_keys = ["v_real", "v_exp", "vdag1", "vdag2", "v_goal", "v_geo", "r_hat"]
    out = {
        "obs": np.empty((n, policy.obs_size), np.float32),
        "h1": np.empty((n, width), np.float16),
        "h2": np.empty((n, width), np.float16),
        "phi": np.empty((n, 128), np.float16),
        "action": np.empty(n, np.int16),
        "team": np.empty(n, np.int8),
        "episode": np.empty(n, np.int32),
        "arena": np.empty(n, np.int16),
        "phys": np.empty((n, 31), np.float32),
        "is_kickoff": np.empty(n, bool),
        "touched": np.empty(n, bool),
    }
    for k in ladder_keys:
        out[f"ladder_{k}"] = np.empty(n, np.float32)

    row = 0
    t0 = time.time()
    steps = 0
    last_touch_ticks = {}
    while row + 2 * NUM_ARENAS <= n:
        obs_list, mask_list, phys_list = [], [], []
        for env in envs:
            o, m, p = env.observe()
            obs_list.append(o)
            mask_list.append(m)
            phys_list.append(p)

        obs_b = torch.from_numpy(np.concatenate(obs_list))
        mask_b = torch.from_numpy(np.concatenate(mask_list))
        h1, h2 = policy.trunk_forward(obs_b)
        actions = policy.sample_actions(h2, mask_b)
        phi = (policy.phi_embedding(h2, actions) if policy.phi is not None
               else torch.zeros(h2.shape[0], 128))
        with torch.no_grad():
            m = policy.models
            nanv = torch.full((h2.shape[0],), float("nan"))
            vt = m["CRITIC_TRUNK"](h2) if "CRITIC_TRUNK" in m else None
            lad = {
                "v_real": m["CRITIC"](vt).flatten() if vt is not None else nanv,
                "v_exp": m["GAP_EXP"](h2).flatten() if "GAP_EXP" in m else nanv,
                "vdag1": m["VDAG1"](vt).flatten() if vt is not None and "VDAG1" in m else nanv,
                "vdag2": m["VDAG2"](vt).flatten() if vt is not None and "VDAG2" in m else nanv,
                "v_goal": m["GOAL_CRITIC"](vt).flatten() if vt is not None and "GOAL_CRITIC" in m else nanv,
                "v_geo": m["GEO_V"](obs_b).flatten() if "GEO_V" in m else nanv,
                "r_hat": m["GEO_REW"](obs_b).flatten() if "GEO_REW" in m else nanv,
            }

        for i, env in enumerate(envs):
            prev_touch = last_touch_ticks.get(env.idx, -1)
            for p in range(2):
                r = row
                out["obs"][r] = obs_list[i][p]
                out["h1"][r] = h1[2 * i + p].numpy()
                out["h2"][r] = h2[2 * i + p].numpy()
                out["phi"][r] = phi[2 * i + p].numpy()
                out["action"][r] = actions[2 * i + p].item()
                out["team"][r] = p
                out["episode"][r] = env.episode_id
                out["arena"][r] = env.idx
                out["phys"][r] = phys_list[i]
                out["is_kickoff"][r] = env.is_kickoff
                out["touched"][r] = env.last_touch_tick != prev_touch and prev_touch >= 0
                for k in ladder_keys:
                    out[f"ladder_{k}"][r] = lad[k][2 * i + p].item()
                row += 1
            last_touch_ticks[env.idx] = env.last_touch_tick

        for i, env in enumerate(envs):
            if env.step(actions[2 * i:2 * i + 2].tolist()):
                env.reset()

        steps += 1
        if steps % 200 == 0:
            el = time.time() - t0
            print(f"  {row:>7,}/{n:,} frames  {row/el:,.0f} rows/s", flush=True)

    for k in out:
        out[k] = out[k][:row]

    ball_z = out["phys"][:, 2]
    near = float((np.abs(out["phys"][:, 0:3] - out["phys"][:, 9:12]).sum(1) < 300).mean())
    print(f"\ncollected {row:,} frames in {time.time()-t0:,.0f}s, "
          f"{out['episode'].max()+1} episodes")
    print(f"ball z>300: {(ball_z > 300).mean():.1%}   car near ball: {near:.1%}   "
          f"touched flag rate: {out['touched'].mean():.2%}")
    for k in ladder_keys:
        v = out[f"ladder_{k}"]
        print(f"  ladder {k:7s} mean {v.mean():+8.3f}  std {v.std():7.3f}  "
              f"p10 {np.percentile(v,10):+8.3f}  p90 {np.percentile(v,90):+8.3f}")
    hr = np.maximum(np.minimum(out["ladder_vdag1"], out["ladder_vdag2"]) - out["ladder_v_real"], 0)
    print(f"  headroom H mean {hr.mean():+.3f}  p90 {np.percentile(hr,90):+.3f}  "
          f"frac>0 {(hr>0).mean():.1%}")

    DATA_DIR.mkdir(exist_ok=True)
    tag = os.environ.get("DATASET_TAG", "")
    fname = f"dataset53{('_' + tag) if tag else ''}.npz"
    np.savez(DATA_DIR / fname, checkpoint=int(ckpt_dir.name.replace("ref_", "") or 0),
             seed=SEED, tick_skip=cd.TICK_SKIP, action_delay=cd.ACTION_DELAY, **out)
    print(f"saved {DATA_DIR / fname}")


if __name__ == "__main__":
    main()
