"""ORACLE wavedash — what success rate can a PERFECT policy achieve through our
control interface (tickSkip 4, actionDelay 3, 90-action table)?

PINNED 4.0 DYNAMICS - INTENTIONALLY NOT MIGRATED to 5.0 (2026-07-16): the ts4
interface ceiling IS the question this oracle answered (arena.step(4) + 30Hz
decision constants throughout; result recorded vs the 4.0 bot's 11.2%). Asking
the same question of the 5.0 interface (tickSkip 8, actionDelay 0) is a NEW
experiment - re-derive every decision-count constant, don't just rerun this.

Car spawned falling forward (dodge available), scripted executor issues the
forward-flip action at trigger offset k decisions before predicted touchdown;
success = the same detector the census/probe used (horizontal speed gain
> 100uu/s across the landing). Sweeping k maps the timing-sensitivity curve;
the max over k is the INTERFACE CEILING. Bot's measured rate: 11.2%.
"""

import json
import time
from pathlib import Path

import numpy as np

import RocketSim as rs
from advanced_obs import ACTION_TABLE

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE.parent / "results"
SEED = 20260733
N_EP = 200
OFFSETS = [0, 1, 2, 3]        # decisions before predicted touchdown to trigger


def find_action(jump, pitch, throttle=1.0, boost=0.0):
    best, bi = 1e9, 0
    for i, e in enumerate(ACTION_TABLE):
        c = (abs(e[5] - jump) * 10 + abs(e[2] - pitch) * 3
             + abs(e[0] - throttle) + abs(e[6] - boost))
        if c < best:
            best, bi = c, i
    return bi


A_DRIVE = find_action(0, 0)                 # throttle forward
A_FLIP = find_action(1, -1)                 # jump + nose-down pitch = forward flip


def set_falling(arena, car, rng):
    bs = rs.BallState()
    bs.pos = rs.Vec(0, 4500, 93)            # ball parked far away
    arena.ball.set_state(bs)
    cs = rs.CarState()
    z0 = rng.uniform(60, 120)
    cs.pos = rs.Vec(rng.uniform(-1500, 1500), rng.uniform(-2500, 0), z0)
    fwd = rng.uniform(0, 2 * np.pi)
    f = np.array([np.cos(fwd), np.sin(fwd), 0.0])
    cs.rot_mat = rs.RotMat(rs.Vec(*f), rs.Vec(*np.cross([0, 0, 1.0], f) * -1), rs.Vec(0, 0, 1))
    cs.vel = rs.Vec(*(f * rng.uniform(600, 1100)))
    cs.boost = 30
    cs.has_jumped = True                    # falling with dodge available
    cs.has_double_jumped = False
    cs.has_flipped = False
    cs.is_on_ground = False
    car.set_state(cs)
    return z0


def apply(car, idx):
    e = ACTION_TABLE[idx]
    c = rs.CarControls()
    c.throttle, c.steer = float(e[0]), float(e[1])
    c.pitch, c.yaw, c.roll = float(e[2]), float(e[3]), float(e[4])
    c.jump, c.boost, c.handbrake = bool(e[5]), bool(e[6]), bool(e[7])
    car.set_controls(c)


def main():
    t0 = time.time()
    rs.init(str(HERE.parents[1] / "build" / "collision_meshes"))
    rng = np.random.default_rng(SEED)
    arena = rs.Arena(rs.GameMode.SOCCAR)
    car = arena.add_car(rs.Team.BLUE)

    res = {"bot_rate": 0.112, "offsets": {}}
    for k in OFFSETS:
        wins = 0
        for _ in range(N_EP):
            set_falling(arena, car, rng)
            apply(car, A_DRIVE)
            landed_speed_pre = None
            flipped = False
            success = False
            for step in range(60):          # 2s of 30Hz decisions
                st = car.get_state()
                vz = st.vel.z
                z = st.pos.z
                hs = float(np.hypot(st.vel.x, st.vel.y))
                # predicted decisions to touchdown under gravity (g = 650 down):
                # solve 17 = z + vz t - 325 t^2
                if not st.is_on_ground:
                    disc = vz * vz + 4 * 325 * (z - 17)
                    t_land = (vz + np.sqrt(max(disc, 0))) / 650.0
                    dec_left = t_land * 30
                else:
                    dec_left = 99
                if st.is_on_ground and landed_speed_pre is None:
                    landed_speed_pre = hs
                if not flipped and not st.is_on_ground and dec_left <= k + 0.5:
                    apply(car, A_FLIP)       # hold one full decision so the dodge registers
                    flipped = True
                elif flipped:
                    apply(car, A_DRIVE)      # then release jump
                arena.step(4)
                st2 = car.get_state()
                hs2 = float(np.hypot(st2.vel.x, st2.vel.y))
                if landed_speed_pre is not None and hs2 > landed_speed_pre + 100:
                    success = True
                    break
                if landed_speed_pre is not None and step > 12:
                    break
            wins += success
        res["offsets"][str(k)] = wins / N_EP
        print(f"trigger {k} decisions pre-touchdown: {wins / N_EP:.1%}", flush=True)

    res["ceiling"] = max(res["offsets"].values())
    print(f"INTERFACE CEILING: {res['ceiling']:.1%} vs bot 11.2%", flush=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "oracle_wavedash.json").write_text(json.dumps(res, indent=1))
    print(f"saved  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
