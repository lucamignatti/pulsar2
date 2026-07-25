"""AERIAL_GAP — where does the aerial break down: takeoff, climb, or touch?
Design + registered interpretation map: AERIAL_GAP.md."""

import json
import time
from pathlib import Path

import numpy as np
import torch

import RocketSim as rs
from fear_decomp import copy_newest
from load_checkpoint import load_models
from steer_team import (ACTION_TABLE, BACK_WALL_Y, DT, SIDE_WALL_X,
                        SteeredPolicyRho, TeamArenaEnv, rollout_team,
                        set_team_air_drill, _face_ball)
from team_decline_probe import decline_readings

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
SEED = 20260729
GOAL_Z = 642.775
N_DRILL = 300
DRILL_S = 3.0
TAKEOFF_S = 4.0

BUCKETS = [(0, 300, "ground"), (300, GOAL_Z, "jumpable"),
           (GOAL_Z, 1200, "low_aerial"), (1200, 99999, "true_aerial")]


def touch_histogram(rec):
    t = np.flatnonzero(rec["touched"])
    bz = rec["phys"][t, 2]
    n = max(len(t), 1)
    return {name: float(((bz >= lo) & (bz < hi)).mean()) for lo, hi, name in BUCKETS} | {"n_touches": int(len(t))}


def opportunity_conversion(rec, ppt):
    """Aerial opportunities: feasible readings with ball z > GOAL_Z at the reading."""
    npl = 2 * ppt
    rd = decline_readings(rec) if ppt == 2 else decline_readings(rec)  # works for ppt=1 too
    phys, episode, slot, touched = rec["phys"], rec["episode"], rec["slot"], rec["touched"]
    ep_rows = {int(e): np.flatnonzero(episode == e) for e in np.unique(episode)}
    row_pos = np.empty(len(slot), np.int64)
    for rows in ep_rows.values():
        row_pos[rows] = np.arange(len(rows))

    opp = np.flatnonzero(rd["feas_self"] & (phys[rd["row"], 2] > GOAL_Z))
    left_ground = 0
    aerial_touch = 0
    boost = []
    for i in opp:
        r = int(rd["row"][i])
        rows = ep_rows[int(episode[r])]
        q = int(row_pos[r])
        q_land = q + npl * int(round(rd["t_land"][i] / DT))
        got_air = False
        got_touch = False
        for qq in range(q, min(len(rows), q_land + 1), npl):
            rr = rows[qq]
            if phys[rr, 11] > 300:
                got_air = True
            if touched[rr] and phys[rr, 2] > GOAL_Z:
                got_touch = True
                break
        left_ground += got_air
        aerial_touch += got_touch
        boost.append(float(phys[r, 15]))
    n = max(len(opp), 1)
    return {"n_opportunities": int(len(opp)),
            "left_ground_frac": left_ground / n,
            "aerial_touch_frac": aerial_touch / n,
            "mean_boost_at_opp": float(np.mean(boost)) if boost else None}


def roll_track(env, policy, seconds, track_car=0):
    """Roll; track car `track_car`: (jumped_1s, max_car_z, touch_ball_z or None)."""
    from steer_team import ACTION_DELAY, DT, TICK_SKIP
    start = env.arena.tick_count
    jumped_1s = False
    max_z = 0.0
    touch_z = None
    steps_1s = int(1 / DT)
    for step in range(int(seconds / DT)):
        obs, masks, phys, states = env.observe()
        _, actions = policy.act(torch.from_numpy(obs), torch.from_numpy(masks))
        if ACTION_DELAY:
            env.arena.step(ACTION_DELAY)
        for p, ai in enumerate(actions.tolist()):
            e = ACTION_TABLE[ai]
            c = rs.CarControls()
            c.throttle, c.steer = float(e[0]), float(e[1])
            c.pitch, c.yaw, c.roll = float(e[2]), float(e[3]), float(e[4])
            c.jump, c.boost, c.handbrake = bool(e[5]), bool(e[6]), bool(e[7])
            env.cars[p].set_controls(c)
            env.prev_actions[p] = e
        env.arena.step(TICK_SKIP - ACTION_DELAY)
        st = env.cars[track_car].get_state()
        max_z = max(max_z, float(st.pos.z))
        if not st.is_on_ground and step < steps_1s:
            jumped_1s = True
        bhi = st.ball_hit_info
        if bhi.is_valid and bhi.tick_count_when_hit > start and touch_z is None:
            touch_z = float(env.arena.ball.get_state().pos.z)
            break
    return jumped_1s, max_z, touch_z


def set_takeoff_probe(env, rng):
    """Grounded car near a high slowly-falling ball, boost 70, opponent parked far."""
    env.arena.reset_kickoff(seed=int(rng.integers(0, 2**30)))
    ball = np.array([rng.uniform(-2200, 2200), rng.uniform(-3000, 3000),
                     rng.uniform(900, 1500)])
    bs = rs.BallState()
    bs.pos = rs.Vec(*ball)
    bs.vel = rs.Vec(rng.uniform(-100, 100), rng.uniform(-100, 100), rng.uniform(-50, 0))
    env.arena.ball.set_state(bs)

    theta = rng.uniform(0, 2 * np.pi)
    pos = ball + np.array([np.cos(theta) * rng.uniform(350, 750),
                           np.sin(theta) * rng.uniform(350, 750), 0.0])
    pos[0] = np.clip(pos[0], -SIDE_WALL_X + 400, SIDE_WALL_X - 400)
    pos[1] = np.clip(pos[1], -BACK_WALL_Y + 400, BACK_WALL_Y - 400)
    pos[2] = 17.0
    cs = rs.CarState()
    cs.pos = rs.Vec(*pos)
    cs.rot_mat = _face_ball(pos, ball)
    f = np.array([ball[0] - pos[0], ball[1] - pos[1], 0.0])
    f /= max(np.linalg.norm(f), 1e-9)
    cs.vel = rs.Vec(*(f * 800.0))
    cs.boost = 70.0
    env.cars[0].set_state(cs)

    far = rs.CarState()
    far.pos = rs.Vec(float(-np.sign(pos[0]) * 3000), float(-np.sign(pos[1]) * 4000), 17.0)
    far.rot_mat = _face_ball(np.array([far.pos.x, far.pos.y, 17.0]), ball)
    far.boost = 30.0
    env.cars[1].set_state(far)
    for car in env.cars:
        car.set_controls(rs.CarControls())
    env.prev_actions[:] = 0


def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    rs.init(str(HERE.parents[1] / "build" / "collision_meshes"))
    ckpt = copy_newest(HERE / "data" / "ckpt_cache")
    models = load_models(ckpt)
    pol = SteeredPolicyRho(models)
    print(f"pinned checkpoint {ckpt.name}", flush=True)
    rng = np.random.default_rng(SEED)
    res = {"checkpoint": int(ckpt.name)}

    # M1 + M2 per mode
    for ppt, rows in ((1, 400_000), (2, 500_000)):
        rec = rollout_team(pol, ppt, rows, SEED + ppt, num_arenas=24)
        res[f"touch_hist_{ppt}v{ppt}"] = touch_histogram(rec)
        res[f"opportunity_{ppt}v{ppt}"] = opportunity_conversion(rec, ppt)
        h, o = res[f"touch_hist_{ppt}v{ppt}"], res[f"opportunity_{ppt}v{ppt}"]
        print(f"{ppt}v{ppt}: touches ground {h['ground']:.0%} jumpable {h['jumpable']:.0%} "
              f"lowAerial {h['low_aerial']:.1%} trueAerial {h['true_aerial']:.1%} "
              f"(n={h['n_touches']}) | opps {o['n_opportunities']}: leftGround "
              f"{o['left_ground_frac']:.0%} aerialTouch {o['aerial_touch_frac']:.1%} "
              f"boost {o['mean_boost_at_opp']:.0f} ({time.time()-t0:.0f}s)", flush=True)

    # M3: drill-context completion (airborne spawn, the live drill)
    env = TeamArenaEnv(0, 1, np.random.default_rng(SEED + 50))
    done = touch = 0
    tz = []
    for _ in range(N_DRILL):
        set_team_air_drill(env.arena, rng, 1)
        for car in env.cars:
            car.set_controls(rs.CarControls())
        env.prev_actions[:] = 0
        _, mz, t = roll_track(env, pol, DRILL_S, track_car=0)
        done += 1
        if t is not None:
            touch += 1
            tz.append(t)
    res["drill_completion"] = {"n": done, "touch_frac": touch / done,
                               "touch_z_median": float(np.median(tz)) if tz else None,
                               "touch_above_goalz_frac": float(np.mean([z > GOAL_Z for z in tz])) if tz else None}
    d = res["drill_completion"]
    print(f"M3 drill (airborne spawn): touch {d['touch_frac']:.0%}, median touch z "
          f"{d['touch_z_median']}, above goal-z {d['touch_above_goalz_frac']}", flush=True)

    # M4: takeoff probe (grounded spawn, high ball)
    jump = air500 = atouch = 0
    for _ in range(N_DRILL):
        set_takeoff_probe(env, rng)
        j, mz, t = roll_track(env, pol, TAKEOFF_S, track_car=0)
        jump += j
        air500 += mz > 500
        atouch += (t is not None and t > 500)
    res["takeoff_probe"] = {"n": N_DRILL, "jump_1s_frac": jump / N_DRILL,
                            "carz500_frac": air500 / N_DRILL,
                            "aerial_touch_frac": atouch / N_DRILL}
    tk = res["takeoff_probe"]
    print(f"M4 takeoff (grounded spawn): jump1s {tk['jump_1s_frac']:.0%} "
          f"carZ>500 {tk['carz500_frac']:.0%} aerialTouch {tk['aerial_touch_frac']:.0%} "
          f"({time.time()-t0:.0f}s total)", flush=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"aerial_gap_{ckpt.name}.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
