"""Real-game capture vs BOTH engines on the identical window set:
  - pip RocketSim (v2, the old physics family)
  - the vendored TUNED RocketSimV3 (what the trainer links), via the v3replay CLI.
Same filters as the original harness; grounded buckets still carry the documented
set_car_state suspension-restore artifact IN BOTH engines, so read AIR as truth and
GROUND as engine-relative comparison only.
"""
import json, math, subprocess, sys
from collections import defaultdict
import numpy as np

import RocketSim as rs
rs.init("/home/luca/Projects/pulsar2-3.0/build/collision_meshes")

CAP = sys.argv[1]
V3 = "/tmp/claude-1000/-home-luca-Projects-pulsar2-3-0/f156a31b-8cf8-4733-94d1-a9648c1d2c0d/scratchpad/v3replay"
MESHES = "/home/luca/Projects/pulsar2-3.0/build/collision_meshes"

dec = [r for r in (json.loads(l) for l in open(CAP) if l.strip()) if r.get("type") == "decision"]
wins = []
for a, b in zip(dec, dec[1:]):
    ticks = int(round((b["t"] - a["t"]) * 120))
    if not (1 <= ticks <= 16): continue
    if a["act_tuple"][5] or a["flip"] == 0: continue
    wins.append((a, b, ticks))
print(f"{len(wins)} windows from {len(dec)} decisions")

# ---- v3 via CLI
lines = []
for a, b, t in wins:
    c = a["act_tuple"]
    lines.append(" ".join(str(x) for x in [t, *a["p"], *a["v"], *a["f"], a["boost"], a["g"], *c]))
out = subprocess.run([V3, MESHES], input="\n".join(lines), capture_output=True, text=True)
v3pred = []
for ln in out.stdout.splitlines():
    if ln.startswith("pred "):
        v3pred.append([float(x) for x in ln.split()[1:7]])
assert len(v3pred) == len(wins), f"v3 rows {len(v3pred)} != {len(wins)}\nstderr: {out.stderr[:400]}"

# ---- v2 in-process
arena = rs.Arena(rs.GameMode.SOCCAR)
car = arena.add_car(rs.Team.BLUE)
def v2_step(a, ticks):
    cs = car.get_state()
    cs.pos = rs.Vec(*a["p"]); cs.vel = rs.Vec(*a["v"]); cs.ang_vel = rs.Vec(0, 0, 0)
    yaw = math.atan2(a["f"][1], a["f"][0]); pitch = math.asin(max(-1, min(1, a["f"][2])))
    cs.rot_mat = rs.Angle(yaw=yaw, pitch=pitch, roll=0).as_rot_mat()
    cs.boost = a["boost"]; cs.is_on_ground = bool(a["g"])
    car.set_state(cs)
    c = a["act_tuple"]
    cc = rs.CarControls()
    cc.throttle, cc.steer, cc.pitch, cc.yaw, cc.roll = c[0], c[1], c[2], c[3], c[4]
    cc.jump, cc.boost, cc.handbrake = bool(c[5]), bool(c[6]), bool(c[7])
    car.set_controls(cc)
    arena.step(ticks)
    st = car.get_state()
    return [st.pos.x, st.pos.y, st.pos.z, st.vel.x, st.vel.y, st.vel.z]

res = defaultdict(lambda: defaultdict(list))
fwd = defaultdict(lambda: defaultdict(list))
for (a, b, t), p3 in zip(wins, v3pred):
    p2 = v2_step(a, t)
    real_p = np.array(b["p"], float); real_v = np.array(b["v"], float)
    key = ("AIR" if not a["g"] else "GROUND") + ("+BOOST" if (a["act_tuple"][6] and a["boost"] > 1) else "+coast")
    fvec = np.array(a["f"], float); fvec /= max(np.linalg.norm(fvec), 1e-9)
    for eng, pred in (("v2", p2), ("v3", p3)):
        pp = np.array(pred[:3]); pv = np.array(pred[3:])
        perr = float(np.linalg.norm(pp - real_p))
        if not math.isfinite(perr) or perr > 500: continue
        res[key][eng].append((perr, float(np.linalg.norm(pv - real_v))))
        if not a["g"]:
            fwd[key][eng].append(float(np.dot(pv - real_v, fvec)))

print(f"\n{'bucket':<14} {'n':>5} | {'v2 pos':>8} {'v3 pos':>8} | {'v2 vel':>8} {'v3 vel':>8}   (medians)")
for k in sorted(res):
    r2 = np.array(res[k]["v2"]); r3 = np.array(res[k]["v3"])
    print(f"{k:<14} {len(r2):>5} | {np.median(r2[:,0]):>7.3f}u {np.median(r3[:,0]):>7.3f}u | "
          f"{np.median(r2[:,1]):>6.2f}u/s {np.median(r3[:,1]):>6.2f}u/s")

print("\nforward-axis velocity residual, air (mean +- SE) — the air-throttle defect detector:")
for k in sorted(fwd):
    a2 = np.array(fwd[k]["v2"]); a3 = np.array(fwd[k]["v3"])
    if len(a2) < 30: continue
    print(f"  {k:<12} v2: {a2.mean():+6.2f}±{a2.std()/len(a2)**.5:.2f}   "
          f"v3: {a3.mean():+6.2f}±{a3.std()/len(a3)**.5:.2f}  uu/s")
