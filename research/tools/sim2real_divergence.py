"""One-step sim-vs-real divergence from a live RLBot match capture.

For each consecutive pair of `decision` records in the bot's debug JSONL: restore the
REAL game state at t_i into RocketSim, apply the SAME controls the bot sent, step the
same number of ticks, and compare the sim's predicted car state at t_{i+1} against what
the real game actually reported.

Buckets follow SIM2REAL_AUDIT.md, because the known effects are bucket-specific:
  - GROUNDED replays are BIASED HIGH: set_car_state cannot restore suspension state
    (rocketsim sim/car/car_extra_state.rs), so the first tick after a restore has no
    wheel contact -> no sticky force, wrong suspension compression. Do NOT read grounded
    error as a physics defect; that is the documented harness artifact.
  - AIR is the clean bucket (no suspension state to restore).
  - AIR+BOOST is the one to watch: the audit CONFIRMED RocketSim double-counts air
    throttle while boosting (+66.67 uu/s^2 on every boosted aerial). If that defect is
    live in this build, air+boost error should exceed air-coast systematically, and the
    residual should point along the car's forward axis.

CAVEAT: this uses pip RocketSim (v2 C++ bindings); the TRAINER runs the vendored
RocketSimV3 (Rust). Findings here are indicative for v3, not proof about it.
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import RocketSim as rs
rs.init("/home/luca/Projects/pulsar2-3.0/build/collision_meshes")

TICK = 1.0 / 120.0


def load(path, limit=None):
    out = []
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("type") == "decision":
            out.append(r)
            if limit and len(out) >= limit:
                break
    return out


def controls_from(tup):
    c = rs.CarControls()
    c.throttle, c.steer, c.pitch, c.yaw, c.roll = tup[0], tup[1], tup[2], tup[3], tup[4]
    c.jump, c.boost, c.handbrake = bool(tup[5]), bool(tup[6]), bool(tup[7])
    return c


def fwd_to_angles(f):
    """Forward vector -> (pitch, yaw). Roll is unrecoverable from forward alone; the
    capture only logs forward, so roll is set 0 and ROLL-SENSITIVE rows are excluded."""
    fx, fy, fz = f
    yaw = math.atan2(fy, fx)
    pitch = math.asin(max(-1.0, min(1.0, fz)))
    return pitch, yaw


def main():
    path = sys.argv[1]
    recs = load(path, limit=int(sys.argv[2]) if len(sys.argv) > 2 else None)
    print(f"loaded {len(recs)} decision records from {Path(path).name}")

    arena = rs.Arena(rs.GameMode.SOCCAR)
    car = arena.add_car(rs.Team.BLUE)

    buckets = defaultdict(list)
    fwd_resid = defaultdict(list)
    used = skipped = 0

    for a, b in zip(recs, recs[1:]):
        dt = b["t"] - a["t"]
        ticks = int(round(dt * 120))
        if ticks < 1 or ticks > 16:      # scene cuts / kickoff resets / packet gaps
            skipped += 1
            continue
        # A jump/flip in the window makes roll matter and we do not have roll -> skip.
        if a["act_tuple"][5] or a["flip"] == 0:
            skipped += 1
            continue

        cs = car.get_state()
        cs.pos = rs.Vec(*a["p"])
        cs.vel = rs.Vec(*a["v"])
        cs.ang_vel = rs.Vec(0, 0, 0)     # not captured; small over 1-2 ticks
        pitch, yaw = fwd_to_angles(a["f"])
        cs.rot_mat = rs.Angle(yaw=yaw, pitch=pitch, roll=0).as_rot_mat()
        cs.boost = a["boost"]
        cs.is_on_ground = bool(a["g"])
        car.set_state(cs)
        car.set_controls(controls_from(a["act_tuple"]))
        arena.step(ticks)

        got = car.get_state()
        pred_p = np.array([got.pos.x, got.pos.y, got.pos.z])
        pred_v = np.array([got.vel.x, got.vel.y, got.vel.z])
        real_p = np.array(b["p"], dtype=float)
        real_v = np.array(b["v"], dtype=float)

        perr = float(np.linalg.norm(pred_p - real_p))
        verr = float(np.linalg.norm(pred_v - real_v))
        if not (math.isfinite(perr) and math.isfinite(verr)) or perr > 500:
            skipped += 1
            continue

        air = not a["g"]
        boosting = bool(a["act_tuple"][6]) and a["boost"] > 1
        key = ("AIR" if air else "GROUND") + ("+BOOST" if boosting else "+coast")
        buckets[key].append((perr, verr, ticks))
        if air:
            # velocity residual projected on the car's forward axis: the signature of a
            # throttle/boost magnitude error (vs a direction error)
            f = np.array(a["f"], dtype=float)
            f /= max(np.linalg.norm(f), 1e-9)
            fwd_resid[key].append(float(np.dot(pred_v - real_v, f)))
        used += 1

    print(f"used {used} transitions, skipped {skipped}\n")
    print(f"{'bucket':<14} {'n':>6} {'med pos err':>12} {'med vel err':>12} {'p90 pos':>9}")
    for k in sorted(buckets):
        arr = np.array([(p, v) for p, v, _ in buckets[k]])
        print(f"{k:<14} {len(arr):>6} {np.median(arr[:,0]):>10.3f}uu {np.median(arr[:,1]):>10.2f}uu/s "
              f"{np.percentile(arr[:,0],90):>7.2f}uu")

    print("\nforward-axis velocity residual (sim - real), air rows:")
    print("  a systematic POSITIVE mean under +BOOST = sim accelerating too hard forward")
    for k in sorted(fwd_resid):
        a = np.array(fwd_resid[k])
        if len(a) < 20:
            continue
        se = a.std() / math.sqrt(len(a))
        print(f"  {k:<12} n={len(a):>5}  mean={a.mean():+7.2f} uu/s  median={np.median(a):+7.2f}  SE={se:.2f}")


if __name__ == "__main__":
    main()
