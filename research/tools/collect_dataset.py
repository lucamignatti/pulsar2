"""Roll out the frozen checkpoint policy in RocketSim (self-play, CPU) and record
per-frame activations + physics for probing.

Replicates the trainer's env loop (EnvSet.cpp), 5.0 dynamics (PULSAR5.md):
  - tickSkip 8, actionDelay 0: set new controls, then 8 ticks, then build obs (whose
    prevAction = the just-set action). 4.0 was tickSkip 4 + actionDelay 3: 3 ticks
    with the PREVIOUS controls, set new controls, 1 more tick.
  - stochastic sampling with DefaultAction masking (InferActionsFromModels parity)
  - terminals: goal scored, 20s without any ball touch (NoTouchCondition(20)), 30s cap
  - reset mix: TRAINER PARITY (ExampleMain MakeEnv): 35% BallNearCarState(600,900) /
    20% AirDrillState / 15% kickoff / 30% RandomState(randBall, randCar, air).
    The earlier kickoff+random-only mix starved ball interactions ~10x vs the trainer
    (touch ratio 0.001 vs 0.006 live), which left the v2 possession labeler with
    almost no resolved races - the drill setters are where contests concentrate.

Per player-frame we store: raw obs (109), trunk hidden 1 + 2 (512 each, f16), the
reach_phi embedding of (trunk out, sampled action) (128, f16), the sampled action,
raw physics of ball + both cars, and episode/arena ids for leakage-safe CV splits.

Differences from the training collector, accepted + noted in REPORT.md:
  - RocketSim pip bindings v2.2.1 vs in-repo v2.1.1
  - fp32 inference (trainer collects in bf16)
  - (4.0-era dynamics only, ACTION_DELAY > 0) after an arena reset the first
    ACTION_DELAY ticks run zeroed controls (the C++ EnvSet carries the pre-reset
    controls for those ticks); moot at 5.0's actionDelay 0
"""

import os
import time
from pathlib import Path

import numpy as np
import torch

import RocketSim as rs
from advanced_obs import (ACTION_TABLE, OBS_SIZE, OBS_SIZE_PADDED, build_obs,
                          build_obs_padded, build_pad_index_map, get_action_mask)
from load_checkpoint import load_latest

SEED = 1234
NUM_ARENAS = 16

# Obs width of the checkpoint being analyzed: 109 (3.1 AdvancedObs) or 230
# (4.0 AdvancedObsPadded). Set from PulsarPolicy.obs_size before building envs.
_OBS_SIZE = OBS_SIZE


def set_obs_size(n: int):
    global _OBS_SIZE
    assert n in (OBS_SIZE, OBS_SIZE_PADDED), n
    _OBS_SIZE = n


def cur_obs_size() -> int:
    return _OBS_SIZE
TARGET_FRAMES = int(os.environ.get("PROBE_FRAMES", 100_000))  # player-frames (2 per arena-step)

# 5.0 MIGRATION (2026-07-16): tickSkip 8, actionDelay 0, NoTouch 20s - keep in
# lockstep with steer_team.py's block and the trainer's ExampleMain. Collecting
# through the old 4+3 dynamics against a 5.0 checkpoint breaks obs/action/step
# parity, and every consumer of dataset.npz (label_landing / train_probes /
# knowing_doing) inherits the skew. For 4.0-era archaeology, set back to 4/3/10.
TICK_SKIP = 8
ACTION_DELAY = 0
NO_TOUCH_TERMINAL_S = 20.0
EPISODE_CAP_S = 30.0
DT = TICK_SKIP / 120.0
# Trainer-parity reset mix (ExampleMain MakeEnv): cumulative weights over
# (ball_near_car, air_drill, kickoff, random)
RESET_MIX = [(0.35, "near"), (0.55, "air"), (0.70, "kickoff"), (1.01, "random")]

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

SIDE_WALL_X, BACK_WALL_Y = 4096.0, 5120.0
BALL_RADIUS = 92.75


def _face_ball_yaw_rotmat(car_pos, ball_pos):
    to_ball = ball_pos - car_pos
    yaw = float(np.arctan2(to_ball[1], to_ball[0]))
    return rs.Angle(yaw, 0, 0).as_rot_mat()


def set_ball_near_car_state(arena, rng, min_dist=600.0, max_dist=900.0):
    """Port of the team-aware BallNearCarState's 1v1 path (both cars = contesters in
    the ring, on-ground, at rest, facing the ball; the ground-touch bootstrap state)."""
    arena.reset_kickoff(seed=int(rng.integers(0, 2**30)))

    bs = rs.BallState()
    ball = np.array([rng.uniform(-2800, 2800), rng.uniform(-3800, 3800), BALL_RADIUS])
    bs.pos = rs.Vec(*ball)
    arena.ball.set_state(bs)

    clamp_x, clamp_y = SIDE_WALL_X - 300, BACK_WALL_Y - 300
    placed = []
    for car in arena.get_cars():
        pos = None
        for _ in range(16):
            theta = rng.uniform(0, 2 * np.pi)
            dist = rng.uniform(min_dist, max_dist)
            cand = ball + np.array([np.cos(theta) * dist, np.sin(theta) * dist, 0.0])
            cand[2] = 17.0
            if abs(cand[0]) > clamp_x or abs(cand[1]) > clamp_y:
                continue
            if any(np.linalg.norm(cand - o) < 300 for o in placed):
                continue
            pos = cand
            break
        if pos is None:
            pos = np.array([np.clip(ball[0] + min_dist, -clamp_x, clamp_x),
                            np.clip(ball[1], -clamp_y, clamp_y), 17.0])
        placed.append(pos)

        cs = rs.CarState()
        cs.pos = rs.Vec(*pos)
        cs.rot_mat = _face_ball_yaw_rotmat(pos, ball)
        cs.boost = rng.uniform(0, 100)
        car.set_state(cs)


def set_air_drill_state(arena, rng):
    """Port of the team-aware AirDrillState's 1v1 path (both cars airborne, climbing
    at an overhead ball, nose on it, boost-fed; the reverse-curriculum aerial state)."""
    arena.reset_kickoff(seed=int(rng.integers(0, 2**30)))

    bs = rs.BallState()
    ball = np.array([rng.uniform(-2600, 2600), rng.uniform(-3400, 3400), rng.uniform(900, 1500)])
    bs.pos = rs.Vec(*ball)
    bs.vel = rs.Vec(rng.uniform(-200, 200), rng.uniform(-200, 200), rng.uniform(-100, 100))
    arena.ball.set_state(bs)

    clamp_x, clamp_y = SIDE_WALL_X - 300, BACK_WALL_Y - 300
    placed = []
    for car in arena.get_cars():
        pos = None
        for _ in range(16):
            theta = rng.uniform(0, 2 * np.pi)
            horiz = rng.uniform(350, 750)
            cand = ball + np.array([np.cos(theta) * horiz, np.sin(theta) * horiz, 0.0])
            cand[2] = max(250.0, ball[2] - rng.uniform(400, 900))
            if abs(cand[0]) > clamp_x or abs(cand[1]) > clamp_y:
                continue
            if any(np.linalg.norm(cand - o) < 350 for o in placed):
                continue
            pos = cand
            break
        if pos is None:
            pos = np.array([np.clip(ball[0] + 500, -clamp_x, clamp_x),
                            np.clip(ball[1], -clamp_y, clamp_y), max(250.0, ball[2] - 600)])
        placed.append(pos)

        # RotMat::LookAt(toBall, world-up) parity
        f = ball - pos
        f = f / max(np.linalg.norm(f), 1e-9)
        tr = np.cross([0.0, 0.0, 1.0], f)
        u = np.cross(f, tr)
        u = u / max(np.linalg.norm(u), 1e-9)
        r = np.cross(u, f)
        r = r / max(np.linalg.norm(r), 1e-9)

        cs = rs.CarState()
        cs.pos = rs.Vec(*pos)
        cs.rot_mat = rs.RotMat(rs.Vec(*f), rs.Vec(*r), rs.Vec(*u))
        cs.vel = rs.Vec(*(f * rng.uniform(700, 1400)))
        cs.boost = rng.uniform(45, 100)
        car.set_state(cs)


def set_random_state(arena, rng):
    """Port of RandomState(true, true, false)::ResetArena."""
    arena.reset_kickoff(seed=int(rng.integers(0, 2**30)))  # resets pads & everything

    X_MAX, Y_MAX, Z_MAX, CAR_Z_MIN = 3900, 4800, 1820, 150
    BALL_RADIUS = 92.75  # CommonValues::BALL_RADIUS (used by the C++ setter)

    def rand_vec(lo, hi):
        return rng.uniform(lo, hi, 3)

    def rand_norm_vec():
        v = rng.uniform(-1, 1, 3)
        return v / max(np.linalg.norm(v), 1e-9)

    bs = rs.BallState()
    p = rand_vec([-X_MAX, -Y_MAX, BALL_RADIUS], [X_MAX, Y_MAX, Z_MAX])
    bs.pos = rs.Vec(*p)
    bs.vel = rs.Vec(*(rand_norm_vec() * rng.uniform(0, 4000)))
    bs.ang_vel = rs.Vec(*rand_vec([-4, -4, -4], [4, 4, 4]))
    arena.ball.set_state(bs)

    for car in arena.get_cars():
        cs = rs.CarState()
        pos = rand_vec([-X_MAX, -Y_MAX, CAR_Z_MIN], [X_MAX, Y_MAX, Z_MAX])
        vel = rand_norm_vec() * rng.uniform(0, 2300)
        ang_vel = rand_norm_vec() * 5.5
        yaw = rng.uniform(-np.pi, np.pi)
        pitch = rng.uniform(-np.pi / 2, np.pi / 2)
        roll = rng.uniform(-np.pi, np.pi)

        on_ground = rng.random() > 0.5
        if on_ground:
            pos[2] = 17
            pitch = roll = 0.0
            vel[2] = 0
            ang_vel = np.zeros(3)

        cs.pos = rs.Vec(*pos)
        cs.vel = rs.Vec(*vel)
        cs.ang_vel = rs.Vec(*ang_vel)
        cs.rot_mat = rs.Angle(yaw, pitch, roll).as_rot_mat()
        cs.boost = rng.uniform(0, 100)
        car.set_state(cs)


class ArenaEnv:
    """One 1v1 arena + the bookkeeping the C++ EnvSet keeps per arena."""

    def __init__(self, idx, rng):
        self.idx = idx
        self.rng = rng
        self.arena = rs.Arena(rs.GameMode.SOCCAR)
        self.cars = [self.arena.add_car(rs.Team.BLUE), self.arena.add_car(rs.Team.ORANGE)]
        self.pad_map = build_pad_index_map(self.arena)
        self.pads = self.arena.get_boost_pads()
        self.goal_scored = False
        self.arena.set_goal_score_callback(self._on_goal, None)
        self.episode_id = -1
        self.reset()

    def _on_goal(self, **kwargs):  # bindings invoke with kwargs (arena, team, data)
        self.goal_scored = True

    def reset(self, episode_counter=[0]):
        u = self.rng.random()
        kind = next(k for w, k in RESET_MIX if u < w)
        self.is_kickoff = kind == "kickoff"
        if kind == "near":
            set_ball_near_car_state(self.arena, self.rng)
        elif kind == "air":
            set_air_drill_state(self.arena, self.rng)
        elif kind == "kickoff":
            self.arena.reset_kickoff(seed=int(self.rng.integers(0, 2**30)))
        else:
            set_random_state(self.arena, self.rng)
        zero = rs.CarControls()
        for car in self.cars:
            car.set_controls(zero)
        self.prev_actions = np.zeros((2, 8), dtype=np.float32)
        self.goal_scored = False
        self.steps = 0
        self.last_touch_tick = self.arena.tick_count
        self.episode_id = episode_counter[0]
        episode_counter[0] += 1

    def pad_states(self):
        active = np.empty(len(self.pad_map), bool)
        cooldown = np.empty(len(self.pad_map), np.float32)
        for i, j in enumerate(self.pad_map):
            st = self.pads[j].get_state()
            active[i] = st.is_active
            cooldown[i] = st.cooldown
        return active, cooldown

    def observe(self):
        """obs + action masks + physics snapshot for both players, at the current tick."""
        ball = self.arena.ball.get_state()
        states = [car.get_state() for car in self.cars]
        active, cooldown = self.pad_states()

        if _OBS_SIZE == OBS_SIZE_PADDED:  # 4.0 padded lineage: 1v1 = empty teammate slots
            obs = np.stack([
                build_obs_padded(states[0], [], [states[1]], ball, self.prev_actions[0],
                                 active, cooldown, False, self.rng),
                build_obs_padded(states[1], [], [states[0]], ball, self.prev_actions[1],
                                 active, cooldown, True, self.rng),
            ])
        else:
            obs = np.stack([
                build_obs(states[0], states[1], ball, self.prev_actions[0], active, cooldown, False),
                build_obs(states[1], states[0], ball, self.prev_actions[1], active, cooldown, True),
            ])
        masks = np.stack([get_action_mask(states[0]), get_action_mask(states[1])])

        phys = np.concatenate([
            ball.pos.as_tuple(), ball.vel.as_tuple(), ball.ang_vel.as_tuple(),
            states[0].pos.as_tuple(), states[0].vel.as_tuple(), states[0].ang_vel.as_tuple(),
            [states[0].boost, float(states[0].is_on_ground)],
            states[1].pos.as_tuple(), states[1].vel.as_tuple(), states[1].ang_vel.as_tuple(),
            [states[1].boost, float(states[1].is_on_ground)],
        ]).astype(np.float32)

        for st in states:  # track ball touches for NoTouchCondition
            bhi = st.ball_hit_info
            if bhi.is_valid:
                self.last_touch_tick = max(self.last_touch_tick, bhi.tick_count_when_hit)
        return obs, masks, phys

    def step(self, action_indices):
        """ACTION_DELAY ticks old controls -> set new controls -> the remaining
        TICK_SKIP - ACTION_DELAY ticks (5.0: delay 0 = set, then 8 ticks).
        Returns True if the episode ended."""
        if ACTION_DELAY:
            self.arena.step(ACTION_DELAY)
        for p, act_idx in enumerate(action_indices):
            elems = ACTION_TABLE[act_idx]
            ctrl = rs.CarControls()
            ctrl.throttle, ctrl.steer = float(elems[0]), float(elems[1])
            ctrl.pitch, ctrl.yaw, ctrl.roll = float(elems[2]), float(elems[3]), float(elems[4])
            ctrl.jump, ctrl.boost, ctrl.handbrake = bool(elems[5]), bool(elems[6]), bool(elems[7])
            self.cars[p].set_controls(ctrl)
            self.prev_actions[p] = elems
        self.arena.step(TICK_SKIP - ACTION_DELAY)
        self.steps += 1

        ticks_per_s = 120.0
        no_touch = (self.arena.tick_count - self.last_touch_tick) > NO_TOUCH_TERMINAL_S * ticks_per_s
        capped = self.steps >= EPISODE_CAP_S * ticks_per_s / TICK_SKIP
        return self.goal_scored or no_touch or capped


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(4)  # be nice: the GPU trainer's env workers own most cores
    rng = np.random.default_rng(SEED)

    rs.init(str(Path(__file__).resolve().parents[2] / "build" / "collision_meshes"))
    policy, ckpt_dir = load_latest()
    set_obs_size(policy.obs_size)
    print(f"policy checkpoint: {ckpt_dir.name} (obs {policy.obs_size})")

    envs = [ArenaEnv(i, np.random.default_rng(SEED + 1000 + i)) for i in range(NUM_ARENAS)]

    n = TARGET_FRAMES
    out = {
        "obs": np.empty((n, cur_obs_size()), np.float32),
        "h1": np.empty((n, 512), np.float16),
        "h2": np.empty((n, 512), np.float16),
        "phi": np.empty((n, 128), np.float16),
        "action": np.empty(n, np.int16),
        "team": np.empty(n, np.int8),
        "episode": np.empty(n, np.int32),
        "arena": np.empty(n, np.int16),
        "phys": np.empty((n, 31), np.float32),
    }

    row = 0
    t0 = time.time()
    steps = 0
    while row + 2 * NUM_ARENAS <= n:
        obs_list, mask_list, phys_list = [], [], []
        for env in envs:
            o, m, p = env.observe()
            obs_list.append(o)
            mask_list.append(m)
            phys_list.append(p)

        obs_b = torch.from_numpy(np.concatenate(obs_list))          # [2*NA, 109]
        mask_b = torch.from_numpy(np.concatenate(mask_list))
        h1, h2 = policy.trunk_forward(obs_b)
        actions = policy.sample_actions(h2, mask_b)
        phi = policy.phi_embedding(h2, actions)

        for i, env in enumerate(envs):
            for p in range(2):
                out["obs"][row] = obs_list[i][p]
                out["h1"][row] = h1[2 * i + p].numpy()
                out["h2"][row] = h2[2 * i + p].numpy()
                out["phi"][row] = phi[2 * i + p].numpy()
                out["action"][row] = actions[2 * i + p].item()
                out["team"][row] = p
                out["episode"][row] = env.episode_id
                out["arena"][row] = env.idx
                out["phys"][row] = phys_list[i]
                row += 1

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
    touch_frac = float((np.abs(out["phys"][:, 0:3] - out["phys"][:, 9:12]).sum(1) < 300).mean())
    print(f"\ncollected {row:,} frames in {time.time()-t0:,.0f}s, "
          f"{out['episode'].max()+1} episodes")
    print(f"ball z>300 (airborne, labelable): {(ball_z > 300).mean():.1%}")
    print(f"frames with car near ball (<300uu): {touch_frac:.1%}")

    DATA_DIR.mkdir(exist_ok=True)
    np.savez(DATA_DIR / "dataset.npz", checkpoint=int(ckpt_dir.name), seed=SEED,
             tick_skip=TICK_SKIP, action_delay=ACTION_DELAY, **out)
    print(f"saved {DATA_DIR / 'dataset.npz'}")


if __name__ == "__main__":
    main()
