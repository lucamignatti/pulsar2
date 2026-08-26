"""Early-re-press fidelity probe. Isolates the one regime the battery's DODGE events
exclude (atsj <= 0.02): a ground jump press whose replay window contains an airborne
re-press within RE_MS. For each such event: restore full pose+jump state PRE ticks
before the ground press, replay captured controls, and score
  - pos/vel error at depths (sim vs real)
  - did REAL dodge (AirState==3 within 8 ticks of the re-press)
  - did SIM dodge (hasFlipped edge in the extended v3replay S line)
  - fire-lag difference where both fired
Run with and without GGL_LEGACY_DODGE_GATE to attribute the JUMP-battery drift.
"""
import json, math, subprocess, sys
import numpy as np

CAP = sys.argv[1]
V3 = sys.argv[2]
MESHES = "/home/luca/Projects/pulsar2-3.0/build/collision_meshes"
PRE = 3
POST = 40
RE_MS = 120

dec = [r for r in (json.loads(l) for l in open(CAP) if l.strip()) if r.get("type") == "decision"]

def clean(i, post=POST):
    return all(int(round((dec[k + 1]["t"] - dec[k]["t"]) * 120)) == 1
               for k in range(i - PRE, min(i + post, len(dec) - 1)))

events = []
for i in range(PRE, len(dec) - POST - 1):
    d, prev = dec[i], dec[i - 1]
    if d["g"] == 1 and d["act_tuple"][5] == 1 and prev["act_tuple"][5] == 0 and clean(i):
        for j in range(i + 1, i + int(RE_MS / 8.33)):
            e = dec[j]
            if e["act_tuple"][5] == 1 and dec[j - 1]["act_tuple"][5] == 0 and e["g"] == 0:
                events.append((i, j))
                break

print(f"{len(events)} early-re-press events (re-press < {RE_MS}ms after ground jump press)")

lines = []
for i, j in events:
    r0 = dec[i - PRE]
    f, u = r0["f"], r0["u"]
    av = r0.get("av", [0, 0, 0])
    st = [*r0["p"], *r0["v"], *f, *u, *av, r0["boost"], r0["g"],
          r0.get("hj", 0), r0.get("hdj", 0), r0.get("hf", 0),
          r0.get("if", 0), r0.get("ij", 0), r0.get("atsj", 0)]
    lines.append("R " + " ".join(f"{x:.6f}" for x in st))
    for k in range(i - PRE, i + POST):
        a = dec[k]["act_tuple"]
        lines.append("C " + " ".join(f"{x:.4f}" for x in a))

out = subprocess.run([V3, MESHES], input="\n".join(lines) + "\n",
                     capture_output=True, text=True).stdout.splitlines()
S = [l.split()[1:] for l in out if l.startswith("S ")]
assert len(S) == len(events) * (PRE + POST), (len(S), len(events))

DEPTHS = [2, 5, 10, 20, 40]
res = {dep: [] for dep in DEPTHS}
fire = {"real": 0, "sim": 0, "both": 0, "neither": 0, "lag": []}
for n, (i, j) in enumerate(events):
    rows = S[n * (PRE + POST):(n + 1) * (PRE + POST)]
    # depth errors measured from the ground press (row PRE-1 is state after press tick)
    for dep in DEPTHS:
        k = PRE - 1 + dep
        if i + dep >= len(dec) or k >= len(rows):
            continue
        real = dec[i + dep]
        sp = [float(x) for x in rows[k][0:3]]
        sv = [float(x) for x in rows[k][3:6]]
        pe = math.dist(sp, real["p"])
        ve = math.dist(sv, real["v"])
        res[dep].append((pe, ve))
    # real dodge fire near the re-press
    realf = any(dec[k].get("as") == 3 for k in range(j, min(j + 8, len(dec))))
    # sim dodge fire: hasFlipped column 10 edge in the same window
    simf, lagd = False, None
    for k in range(j - i, min(j - i + 8, POST)):
        row = rows[PRE - 1 + k]
        if len(row) > 10 and row[10] == "1":
            simf = True
            lagd = k - (j - i)
            break
    if realf and simf:
        fire["both"] += 1
        fire["lag"].append(lagd)
    elif realf:
        fire["real"] += 1
    elif simf:
        fire["sim"] += 1
    else:
        fire["neither"] += 1

print(f"fire: both={fire['both']} real-only={fire['real']} sim-only={fire['sim']} "
      f"neither={fire['neither']}")
if fire["lag"]:
    print(f"sim fire lag after re-press (ticks): median {np.median(fire['lag']):.1f}")
print("depth   pos_med   vel_med")
for dep in DEPTHS:
    if res[dep]:
        a = np.array(res[dep])
        print(f" t+{dep:<3} {np.median(a[:,0]):8.1f}u {np.median(a[:,1]):8.1f}v")
