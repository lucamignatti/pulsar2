"""Free-running drift: restore ONCE, roll forward with captured per-tick controls,
score error at increasing depth. Separates restore-artifact (settles after ~2-4 ticks)
from genuine physics error (persists/grows per tick).

Segments: contiguous 1-tick decision runs with no jump press and flip available
(roll unknown), single regime throughout (all-ground or all-air), length >= 30 ticks.
"""
import json, math, subprocess, sys
from collections import defaultdict
import numpy as np

CAP = sys.argv[1]
V3 = "/tmp/claude-1000/-home-luca-Projects-pulsar2-3-0/f156a31b-8cf8-4733-94d1-a9648c1d2c0d/scratchpad/v3replay"
MESHES = "/home/luca/Projects/pulsar2-3.0/build/collision_meshes"
SEG = 30  # ticks per segment

dec = [r for r in (json.loads(l) for l in open(CAP) if l.strip()) if r.get("type") == "decision"]

# contiguous runs: consecutive decisions exactly 1 tick apart, no jump, flip avail, same regime
segs = []
i = 0
while i < len(dec) - SEG:
    ok = True
    g0 = dec[i]["g"]
    for k in range(i, i + SEG):
        d = dec[k]
        if (int(round((dec[k+1]["t"] - d["t"]) * 120)) != 1 or d["act_tuple"][5]
                or d["flip"] == 0 or d["g"] != g0):
            ok = False
            break
    if ok:
        # interaction filter: ball within 400uu of the car anywhere in the segment means a
        # possible real contact/deflection the single-car replay cannot represent
        import math as _m
        for k in range(i, i + SEG + 1):
            d = dec[k]
            if _m.dist(d["p"], d["b"]) < 400: ok = False; break
    if ok:
        segs.append((i, g0))
        i += SEG          # non-overlapping
    else:
        i = max(k, i + 1)
print(f"{len(segs)} non-overlapping {SEG}-tick segments "
      f"(ground={sum(1 for _,g in segs if g)}, air={sum(1 for _,g in segs if not g)})")

# drive the v3 CLI over all segments in one process
lines = []
for i0, g0 in segs:
    a = dec[i0]
    lines.append("R " + " ".join(str(x) for x in [*a["p"], *a["v"], *a["f"], *a.get("u",[0,0,1]), *a.get("av",[0,0,0]), a["boost"], a["g"]]))
    for k in range(i0, i0 + SEG):
        lines.append("C " + " ".join(str(x) for x in dec[k]["act_tuple"]))
out = subprocess.run([V3, MESHES], input="\n".join(lines), capture_output=True, text=True)
states = [[float(x) for x in ln.split()[1:7]] for ln in out.stdout.splitlines() if ln.startswith("S ")]
assert len(states) == len(segs) * SEG, (len(states), len(segs) * SEG, out.stderr[:300])

# score drift at depths
depths = [1, 2, 4, 8, 15, 30]
perr = defaultdict(lambda: defaultdict(list))
verr = defaultdict(lambda: defaultdict(list))
vz_resid = defaultdict(lambda: defaultdict(list))   # up-axis = suspension direction
si = 0
for i0, g0 in segs:
    key = "GROUND" if g0 else "AIR"
    for k in range(SEG):
        pred = states[si + k]
        real = dec[i0 + k + 1]
        d = k + 1
        if d in depths:
            pe = float(np.linalg.norm(np.array(pred[:3]) - np.array(real["p"])))
            ve = float(np.linalg.norm(np.array(pred[3:]) - np.array(real["v"])))
            if pe < 2000:
                perr[key][d].append(pe)
                verr[key][d].append(ve)
                vz_resid[key][d].append(abs(pred[5] - real["v"][2]))
    si += SEG

print(f"\nmedian error by rollout depth (restore artifact settles; real error grows)")
print(f"{'':>8}" + "".join(f"{d:>10}t" for d in depths))
for key in ("AIR", "GROUND"):
    if not perr[key]: continue
    print(f"{key:>8} pos " + "".join(f"{np.median(perr[key][d]):>9.2f}u" for d in depths))
    print(f"{'':>8} vel " + "".join(f"{np.median(verr[key][d]):>9.2f}v" for d in depths))
    print(f"{'':>8} |vz|" + "".join(f"{np.median(vz_resid[key][d]):>9.2f}v" for d in depths))
# per-tick marginal error deep in the segment = genuine physics error rate
for key in ("AIR", "GROUND"):
    if not perr[key]: continue
    late = (np.median(verr[key][30]) - np.median(verr[key][15])) / 15
    print(f"{key}: marginal vel error ticks 15->30: {late:+.3f} uu/s per tick")
