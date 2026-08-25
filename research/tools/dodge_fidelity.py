"""Does RocketSimV3 reproduce REAL-GAME dodges — especially TURNING ones?

The earlier free-run certification (freerun_v3.py) filtered OUT every segment containing a
jump press or an unavailable flip, so it measured driving and flying only. Dodge dynamics
were never tested. This does that, and splits by yaw input during the dodge, to test the
hypothesis that dodges WITH TURNING diverge from the real game.

Per dodge: restore full pose + jump/flip state a few ticks before the dodge press, replay
the captured per-tick controls through the dodge and ~0.5 s after, and score position /
velocity / heading error against what the real game did.

Requires a capture with `u` + `av` (full pose) — RLBotClient logs these since 2026-08-25.
"""
import json, math, subprocess, sys
from collections import defaultdict
import numpy as np

CAP = sys.argv[1]
V3 = sys.argv[2] if len(sys.argv) > 2 else \
    "/tmp/claude-1000/-home-luca-Projects-pulsar2-3-0/7b31c9b3-d214-4727-b0e2-8da33eddea48/scratchpad/v3replay_dodge"
MESHES = "/home/luca/Projects/pulsar2-3.0/build/collision_meshes"
PRE, POST = 3, 60          # ticks restored before the dodge / replayed after

dec = [r for r in (json.loads(l) for l in open(CAP) if l.strip()) if r.get("type") == "decision"]
if "u" not in dec[0]:
    sys.exit("capture lacks full pose (u/av) — replay a match recorded after 2026-08-25")

# dodge events: airborne, jump pressed, already jumped (atsj>0) => this press is a DODGE
events = []
for i in range(PRE, len(dec) - POST - 1):
    d = dec[i]
    if d["g"] == 0 and d["act_tuple"][5] == 1 and d["atsj"] > 0.02 and d["flip"] == 1:
        if dec[i-1]["act_tuple"][5] == 1:       # want the rising edge of the dodge press
            continue
        # contiguous 1-tick run around it
        if all(int(round((dec[k+1]["t"] - dec[k]["t"]) * 120)) == 1
               for k in range(i - PRE, i + POST)):
            events.append(i)
print(f"{len(events)} dodge events with clean 1-tick windows")

lines = []
for i in events:
    a = dec[i - PRE]
    lines.append("R " + " ".join(str(x) for x in [
        *a["p"], *a["v"], *a["f"], *a["u"], *a["av"], a["boost"], a["g"],
        a["hj"], a["hdj"], a["hf"], 0, 0, a["atsj"]]))
    for k in range(i - PRE, i - PRE + PRE + POST):
        lines.append("C " + " ".join(str(x) for x in dec[k]["act_tuple"]))
out = subprocess.run([V3, MESHES], input="\n".join(lines), capture_output=True, text=True)
rows = [ln.split()[1:] for ln in out.stdout.splitlines() if ln.startswith("S ")]
per = PRE + POST
assert len(rows) == len(events) * per, (len(rows), len(events) * per, out.stderr[:300])

depths = [5, 10, 20, 40, 60]
res = defaultdict(lambda: defaultdict(list))
for ei, i in enumerate(events):
    d = dec[i]
    # What the bot actually varies at the dodge press is ROLL (yaw/steer are identically 0
    # across every observed dodge) — so "turning dodge" means ROLL-loaded, not yaw-loaded.
    key = "ROLLED" if abs(d["act_tuple"][4]) > 0.5 else "flat"
    for k in range(per):
        depth = k - PRE + 1
        if depth not in depths:
            continue
        pred = [float(x) for x in rows[ei * per + k]]
        real = dec[i - PRE + k + 1]
        pe = float(np.linalg.norm(np.array(pred[:3]) - np.array(real["p"])))
        ve = float(np.linalg.norm(np.array(pred[3:6]) - np.array(real["v"])))
        # heading error in degrees between predicted and real forward vectors
        pf = np.array(pred[7:10]); rf = np.array(real["f"], float)
        c = float(np.dot(pf, rf) / max(np.linalg.norm(pf) * np.linalg.norm(rf), 1e-9))
        he = math.degrees(math.acos(max(-1.0, min(1.0, c))))
        if pe < 3000:
            res[key][depth].append((pe, ve, he))

print(f"\n{'bucket':>9} {'n':>5} " + "".join(f"{('t+%d' % d):>22}" for d in depths))
print(f"{'':>15} " + "".join(f"{'pos/vel/heading':>22}" for _ in depths))
for key in ("flat", "ROLLED"):
    if not res[key]:
        continue
    n = len(res[key][depths[0]])
    cells = []
    for d in depths:
        arr = np.array(res[key][d])
        cells.append(f"{np.median(arr[:,0]):6.1f}u {np.median(arr[:,1]):5.0f}v {np.median(arr[:,2]):4.1f}°")
    print(f"{key:>9} {n:>5} " + "".join(f"{c:>22}" for c in cells))
