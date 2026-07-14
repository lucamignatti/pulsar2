"""Phase-3 (frontier state setters) offline validation - E5.

The live design reconstructs reset states from OBS-VISIBLE quantities only
(pos/vel/angVel/forward/up/boost for every car + ball phys) - no ArenaSnapshot
banking - so reconstruction error is part of the intended perturbation. This
script validates that premise:

  1. mine FRONTIER readings (feasible airborne ball, collective decline) from a
     base rollout, with the true full state banked per step;
  2. reconstruct each into a fresh arena at several perturbation magnitudes;
  3. roll the policy for a few seconds and measure: playability (finite, in
     bounds), race resolution (someone touches), and contestedness (both sides
     win sometimes).

Pre-registered soft bar: 100% playable; touch rate materially above zero
(frontier resets must create PRACTICE, not dead states); a perturbation level
where outcomes vary across repeats (not exact replays). The recommended level
feeds the C++ FrontierDrillState.

Usage: python frontier_validate.py [--rows N] [--n-states M] [--ppt 1|2]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import RocketSim as rs
from load_checkpoint import copy_checkpoint, load_models
from steer_team import (NONE, SteeredPolicyRho, TeamArenaEnv, rollout_team,
                        team_possession_readings)

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
SEED = 20260719

PERTURB_LEVELS = {"exact": (0.0, 0.0), "small": (100.0, 100.0), "medium": (250.0, 250.0)}
ROLL_S = 4.0
REPEATS = 2  # per state per level (stochastic policy -> outcome variation)


def reconstruct(env, entry, rng, pos_noise, vel_noise):
    """Set the arena to a banked (ball9 + npl*17) state with optional perturbation.
    Rotation rebuilt from forward/up exactly like the C++ reconstruction would."""
    npl = env.n_players
    bs = rs.BallState()
    bpos = entry[0:3] + rng.normal(0, pos_noise, 3) * [1, 1, 0.3]
    bpos[2] = max(bpos[2], 94.0)
    bs.pos = rs.Vec(*bpos)
    bs.vel = rs.Vec(*(entry[3:6] + rng.normal(0, vel_noise, 3)))
    bs.ang_vel = rs.Vec(*entry[6:9])
    env.arena.ball.set_state(bs)

    for i, car in enumerate(env.cars):
        c = entry[9 + i * 17: 9 + (i + 1) * 17]
        pos = c[0:3] + rng.normal(0, pos_noise, 3) * [1, 1, 0.2]
        pos[2] = max(pos[2], 17.0)
        pos[0] = np.clip(pos[0], -3990, 3990)
        pos[1] = np.clip(pos[1], -5010, 5010)
        f = c[9:12] / max(np.linalg.norm(c[9:12]), 1e-9)
        u = c[12:15]
        u = u - (u @ f) * f  # re-orthogonalize the up hint against forward
        u /= max(np.linalg.norm(u), 1e-9)
        r = np.cross(u, f)
        r /= max(np.linalg.norm(r), 1e-9)
        cs = rs.CarState()
        cs.pos = rs.Vec(*pos)
        cs.vel = rs.Vec(*(c[3:6] + rng.normal(0, vel_noise, 3)))
        cs.ang_vel = rs.Vec(*c[6:9])
        cs.rot_mat = rs.RotMat(rs.Vec(*f), rs.Vec(*r), rs.Vec(*u))
        cs.boost = float(np.clip(c[15], 0, 100))
        car.set_state(cs)
    zero = rs.CarControls()
    for car in env.cars:
        car.set_controls(zero)
    env.prev_actions[:] = 0


def roll_and_judge(env, policy, seconds):
    """Roll the policy; return (first_touch_team or None, all_finite)."""
    import torch as th
    start_tick = env.arena.tick_count
    for _ in range(int(seconds * 30)):
        obs, masks, phys, states = env.observe()
        _, actions = policy.act(th.from_numpy(obs), th.from_numpy(masks))
        env.arena.step(3)
        from steer_team import ACTION_TABLE
        for p, act_idx in enumerate(actions.tolist()):
            e = ACTION_TABLE[act_idx]
            ctrl = rs.CarControls()
            ctrl.throttle, ctrl.steer = float(e[0]), float(e[1])
            ctrl.pitch, ctrl.yaw, ctrl.roll = float(e[2]), float(e[3]), float(e[4])
            ctrl.jump, ctrl.boost, ctrl.handbrake = bool(e[5]), bool(e[6]), bool(e[7])
            env.cars[p].set_controls(ctrl)
            env.prev_actions[p] = e
        env.arena.step(1)
        for p, car in enumerate(env.cars):
            st = car.get_state()
            if not np.isfinite(st.pos.as_tuple()).all():
                return None, False
            bhi = st.ball_hit_info
            if bhi.is_valid and bhi.tick_count_when_hit > start_tick:
                return p % 2, True
    return None, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=60_000)
    ap.add_argument("--n-states", type=int, default=120)
    ap.add_argument("--ppt", type=int, default=1)
    ap.add_argument("--ckpt", type=Path, default=None)
    args = ap.parse_args()

    t0 = time.time()
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    rs.init(str(HERE.parents[1] / "build" / "collision_meshes"))
    ckpt = args.ckpt or copy_checkpoint(HERE / "data" / "ckpt_cache")
    models = load_models(ckpt)
    policy = SteeredPolicyRho(models)
    print(f"pinned checkpoint {ckpt.name}, mode {args.ppt}v{args.ppt}")

    base = rollout_team(policy, args.ppt, args.rows, SEED, want_states=True)
    readings = team_possession_readings(base, max_readings=8000)
    frontier = readings["row"][readings["outcome"] == NONE]
    rng = np.random.default_rng(SEED)
    rng.shuffle(frontier)
    frontier = frontier[: args.n_states]
    npl = 2 * args.ppt
    print(f"{len(frontier)} frontier states mined "
          f"({(readings['outcome'] == NONE).mean():.0%} of feasible readings are declines)")

    env = TeamArenaEnv(0, args.ppt, np.random.default_rng(SEED + 99))
    results = {"checkpoint": int(ckpt.name), "mode": f"{args.ppt}v{args.ppt}",
               "n_states": int(len(frontier)), "levels": {}}
    for name, (pn, vn) in PERTURB_LEVELS.items():
        touch, blue_first, orange_first, playable = 0, 0, 0, 0
        n = 0
        for r in frontier:
            entry = base["state_bank"][r // npl]
            for rep in range(REPEATS):
                reconstruct(env, entry, rng, pn, vn)
                env.last_touch_tick = env.arena.tick_count
                first, finite = roll_and_judge(env, policy, ROLL_S)
                n += 1
                playable += int(finite)
                if first is not None:
                    touch += 1
                    if first == 0:
                        blue_first += 1
                    else:
                        orange_first += 1
        results["levels"][name] = {
            "playable": playable / n,
            "touch_rate": touch / n,
            "blue_first": blue_first / max(touch, 1),
            "orange_first": orange_first / max(touch, 1),
            "n_rolls": n,
        }
        print(f"{name:>7} (pos {pn:.0f}uu, vel {vn:.0f}): playable {playable/n:.0%}, "
              f"touch-in-{ROLL_S:.0f}s {touch/n:.0%}, first-touch split "
              f"{blue_first}/{orange_first}", flush=True)

    results["runtime_s"] = round(time.time() - t0, 1)
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"frontier_{args.ppt}v{args.ppt}_{ckpt.name}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"runtime {results['runtime_s']}s; wrote {out.name}")


if __name__ == "__main__":
    main()
