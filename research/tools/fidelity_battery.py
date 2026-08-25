"""Per-mechanic sim2real fidelity battery for the tuned RocketSimV3.

Covers every regime the original free-run certification EXCLUDED (its filters dropped all
jump presses, flip-unavailable rows and regime transitions, so it certified only driving
and clean flight):

  JUMP      first jump from ground (impulse + hold extension + sticky release)
  DODGE     airborne second press with directional input — split flat vs ROLLED,
            with an impulse-tick velocity-delta decomposition in the car frame
  DBLJUMP   airborne second press, no directional input
  CANCEL    flip in progress + opposite-pitch input (the wavedash core)
  LANDING   air->ground transition (suspension engagement)
  HBRAKE    grounded handbrake windows vs plain ground

Every event: restore full pose + jump/flip state PRE ticks before the event, replay the
captured per-tick controls, score pos/vel error at depths. DODGE additionally reports
sim-vs-real one-tick delta-v at the flip tick, projected on the car's forward/right/up.

Needs a full-pose capture (u+av logged, post 2026-08-25).
"""
import json, math, subprocess, sys
from collections import defaultdict
import numpy as np

CAP = sys.argv[1]
V3 = sys.argv[2] if len(sys.argv) > 2 else \
    "/tmp/claude-1000/-home-luca-Projects-pulsar2-3-0/7b31c9b3-d214-4727-b0e2-8da33eddea48/scratchpad/v3replay_dodge"
MESHES = "/home/luca/Projects/pulsar2-3.0/build/collision_meshes"
PRE = 3
POST = 40

dec = [r for r in (json.loads(l) for l in open(CAP) if l.strip()) if r.get("type") == "decision"]
if "u" not in dec[0]:
    sys.exit("need full-pose capture")

def clean(i, post=POST):
    return all(int(round((dec[k+1]["t"] - dec[k]["t"]) * 120)) == 1
               for k in range(i - PRE, min(i + post, len(dec) - 1)))

events = defaultdict(list)
for i in range(PRE, len(dec) - POST - 1):
    d, prev = dec[i], dec[i-1]
    jp = d["act_tuple"][5] == 1
    jprev = prev["act_tuple"][5] == 1
    if d["g"] == 1 and jp and not jprev and clean(i):
        events["JUMP"].append(i)
    if d["g"] == 0 and jp and not jprev and d["atsj"] > 0.02 and d["flip"] == 1 and clean(i):
        directional = abs(d["act_tuple"][2]) > 0.1 or abs(d["act_tuple"][3]) > 0.1
        if directional:
            events["DODGE_roll" if abs(d["act_tuple"][4]) > 0.5 else "DODGE_flat"].append(i)
        else:
            events["DBLJUMP"].append(i)
    # flip cancel: hasFlipped, airborne, pitch input OPPOSING current pitch rotation sign
    if d["g"] == 0 and d["hf"] == 1 and prev["hf"] == 1 and abs(d["act_tuple"][2]) > 0.5 and clean(i, 20):
        events["CANCEL"].append(i)
    # landing: airborne -> grounded transition
    if prev["g"] == 0 and d["g"] == 1 and clean(i, 20):
        events["LANDING"].append(i)
    # handbrake windows on ground
    if d["g"] == 1 and d["act_tuple"][7] == 1 and not jp and clean(i, 20):
        events["HBRAKE"].append(i)

# subsample the plentiful classes
import random
random.seed(7)
for k in ("CANCEL", "LANDING", "HBRAKE"):
    if len(events[k]) > 120:
        events[k] = sorted(random.sample(events[k], 120))

order = ["JUMP", "DODGE_flat", "DODGE_roll", "DBLJUMP", "CANCEL", "LANDING", "HBRAKE"]
print({k: len(events[k]) for k in order})

lines, meta = [], []
for kind in order:
    for i in events[kind]:
        a = dec[i - PRE]
        lines.append("R " + " ".join(str(x) for x in [
            *a["p"], *a["v"], *a["f"], *a["u"], *a["av"], a["boost"], a["g"],
            a["hj"], a["hdj"], a["hf"], 0, 0, a["atsj"]]))
        n = POST if kind.startswith(("JUMP", "DODGE", "DBL")) else 20
        for k in range(i - PRE, i - PRE + PRE + n):
            lines.append("C " + " ".join(str(x) for x in dec[k]["act_tuple"]))
        meta.append((kind, i, n))
out = subprocess.run([V3, MESHES], input="\n".join(lines), capture_output=True, text=True)
rows = [[float(x) for x in ln.split()[1:]] for ln in out.stdout.splitlines() if ln.startswith("S ")]

res = defaultdict(lambda: defaultdict(list))
dv_real, dv_sim = defaultdict(list), defaultdict(list)
ri = 0
for kind, i, n in meta:
    per = PRE + n
    seg = rows[ri:ri + per]; ri += per
    for k in range(per):
        depth = k - PRE + 1
        if depth not in (2, 5, 10, 20, 40) or depth > n:
            continue
        pred, real = seg[k], dec[i - PRE + k + 1]
        pe = float(np.linalg.norm(np.array(pred[:3]) - np.array(real["p"])))
        ve = float(np.linalg.norm(np.array(pred[3:6]) - np.array(real["v"])))
        if pe < 3000:
            res[kind][depth].append((pe, ve))
    if kind.startswith("DODGE"):
        # one-tick delta-v across the flip tick (press at i): real vs sim, car frame of i
        f = np.array(dec[i]["f"], float); u = np.array(dec[i]["u"], float)
        r = np.cross(u, f)
        # find the tick with the largest real delta-v within 3 ticks of the press
        best, bidx = 0, i
        for k in range(i, i + 4):
            dv = np.linalg.norm(np.array(dec[k+1]["v"]) - np.array(dec[k]["v"]))
            if dv > best: best, bidx = dv, k
        rv = np.array(dec[bidx+1]["v"]) - np.array(dec[bidx]["v"])
        si = bidx - (i - PRE)   # index into seg for post-step state of that tick
        if 0 < si < per:
            sv = np.array(seg[si][3:6]) - np.array(seg[si-1][3:6])
            for tag, v in (("real", rv), ("sim", sv)):
                (dv_real if tag == "real" else dv_sim)[kind].append(
                    [float(np.dot(v, f)), float(np.dot(v, r)), float(np.dot(v, u)), float(np.linalg.norm(v))])

print(f"\n{'mechanic':>11} {'n':>4} " + "".join(f"{('t+%d' % d):>16}" for d in (2, 5, 10, 20, 40)))
for kind in order:
    if not res[kind]: continue
    cells = []
    for d in (2, 5, 10, 20, 40):
        a = np.array(res[kind][d]) if res[kind][d] else None
        cells.append(f"{np.median(a[:,0]):5.1f}u {np.median(a[:,1]):5.0f}v" if a is not None and len(a) else " " * 12)
    print(f"{kind:>11} {len(res[kind][2]):>4} " + "".join(f"{c:>16}" for c in cells))

print("\nDODGE impulse-tick delta-v, car frame [fwd, right, up, |dv|] (medians):")
for kind in ("DODGE_flat", "DODGE_roll"):
    if dv_real[kind]:
        R = np.median(np.array(dv_real[kind]), axis=0)
        S = np.median(np.array(dv_sim[kind]), axis=0)
        print(f"  {kind:>10} n={len(dv_real[kind]):>3}  real [{R[0]:+7.1f} {R[1]:+7.1f} {R[2]:+7.1f} |{R[3]:6.1f}]")
        print(f"  {'':>10}       sim  [{S[0]:+7.1f} {S[1]:+7.1f} {S[2]:+7.1f} |{S[3]:6.1f}]")
