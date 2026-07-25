"""Detector #1 ground truth: where and when the airborne ball lands.

For every frame with ball z > 300uu, clone the ball (pos, vel, angVel) into a car-free
SOCCAR arena and step physics until touchdown: z <= BALL_COLLISION_RADIUS + EPS
(ball rests at z ~ 93.15; radius 91.25), cap 6 s. Labels: (x_land, y_land, t_land).
Bounces off walls/ceiling before touchdown are exactly what makes this concept
nonlinear in the obs.

Writes data/labels.npz aligned row-for-row with dataset.npz (invalid rows flagged).
"""

import time
from pathlib import Path

import numpy as np
import RocketSim as rs

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

AIRBORNE_Z = 300.0
BALL_RADIUS = 91.25          # RocketSim soccar collision radius (mutator default)
LANDING_EPS = 20.0           # first tick at/below radius+eps; ~1 tick of fall at speed
CAP_TICKS = 720              # 6 s * 120

# Landing is detected on the tick physics resolves ground contact, so the recorded
# z is ~[radius, radius+eps]; x/y error from the eps window is < 1 tick of travel.
LANDING_Z = BALL_RADIUS + LANDING_EPS


def simulate_landing(arena, pos, vel, ang_vel):
    """Returns (x, y, t_seconds) of first touchdown, or None if airborne past the cap."""
    bs = rs.BallState()
    bs.pos = rs.Vec(*pos)
    bs.vel = rs.Vec(*vel)
    bs.ang_vel = rs.Vec(*ang_vel)
    arena.ball.set_state(bs)

    for tick in range(1, CAP_TICKS + 1):
        arena.step(1)
        b = arena.ball.get_state()
        if b.pos.z <= LANDING_Z:
            return b.pos.x, b.pos.y, tick / 120.0
    return None


def main():
    rs.init(str(Path(__file__).resolve().parents[2] / "build" / "collision_meshes"))
    data = np.load(DATA_DIR / "dataset.npz")
    phys = data["phys"]
    n = len(phys)

    ball_pos, ball_vel, ball_ang = phys[:, 0:3], phys[:, 3:6], phys[:, 6:9]
    airborne = ball_pos[:, 2] > AIRBORNE_Z
    print(f"{n:,} frames, {airborne.sum():,} airborne (ball z > {AIRBORNE_Z:.0f})")

    # One car-free arena, reused; goals count as "landing" only if the ball actually
    # comes down - balls that fly into the net cross y=+-5120 first, and we keep
    # whatever touchdown the sim reports (net floor is part of the world).
    arena = rs.Arena(rs.GameMode.SOCCAR)

    labels = np.full((n, 3), np.nan, np.float32)  # x, y, t
    valid = np.zeros(n, bool)

    t0 = time.time()
    idxs = np.flatnonzero(airborne)
    for k, i in enumerate(idxs):
        res = simulate_landing(arena, ball_pos[i], ball_vel[i], ball_ang[i])
        if res is not None:
            labels[i] = res
            valid[i] = True
        if (k + 1) % 10_000 == 0:
            el = time.time() - t0
            print(f"  {k+1:,}/{len(idxs):,}  ({(k+1)/el:,.0f} sims/s)", flush=True)

    print(f"labeled {valid.sum():,}/{len(idxs):,} airborne frames "
          f"({(~valid[airborne]).sum()} never landed within 6s) in {time.time()-t0:,.0f}s")
    t = labels[valid, 2]
    print(f"t_land: median {np.median(t):.2f}s, p90 {np.percentile(t, 90):.2f}s")

    np.savez(DATA_DIR / "labels.npz", land_xy_t=labels, valid=valid,
             airborne=airborne, checkpoint=data["checkpoint"])
    print(f"saved {DATA_DIR / 'labels.npz'}")


if __name__ == "__main__":
    main()
