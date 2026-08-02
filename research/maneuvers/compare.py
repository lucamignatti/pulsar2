#!/usr/bin/env python3
"""Compare a scripted maneuver run in the sim against the same script in the real game.

Both runners emit the same TSV. Segments are independent (each starts from a state set),
so a segment's error cannot be inherited from the one before it -- each is scored alone.

The real game delivers packets at its RENDER rate (60 Hz unless UncappedFramerate=True),
so its rows land on a subset of the sim's ticks. We compare only ticks present in both.

usage: compare.py sim_maneuvers.tsv real_maneuvers.tsv
"""
import sys, math, collections

def load(p):
    out = collections.defaultdict(dict)
    with open(p) as f:
        hdr = f.readline().rstrip("\n").split("\t")
        ix = {n: i for i, n in enumerate(hdr)}
        for line in f:
            r = line.rstrip("\n").split("\t")
            if len(r) < len(hdr):
                continue
            out[r[ix["seg"]]][int(r[ix["tick"]])] = [float(r[ix[c]]) for c in
                ("x","y","z","vx","vy","vz","fx","fy","fz","ux","uy","uz","avx","avy","avz")] \
                + [float(r[ix["ground"]]), float(r[ix["boost"]])]
    return out

def ang(a, b):
    d = max(-1.0, min(1.0, sum(x*y for x, y in zip(a, b))))
    return math.degrees(math.acos(d))

sim, real = load(sys.argv[1]), load(sys.argv[2])
segs = [s for s in sim if s in real]
missing = [s for s in sim if s not in real]
print(f"segments compared: {len(segs)}" + (f"   MISSING FROM REAL: {missing}" if missing else ""))
print(f"{'segment':28} {'n':>5} {'pos p50':>8} {'pos p90':>8} {'pos max':>9} "
      f"{'vel p50':>8} {'fwd deg':>8} {'up deg':>7} {'gnd mism':>8}")
worst = []
for s in sorted(segs):
    ticks = sorted(set(sim[s]) & set(real[s]))
    if not ticks:
        continue
    pe, ve, fa, ua, gm = [], [], [], [], 0
    for t in ticks:
        a, b = sim[s][t], real[s][t]
        pe.append(math.dist(a[0:3], b[0:3]))
        ve.append(math.dist(a[3:6], b[3:6]))
        fa.append(ang(a[6:9], b[6:9]))
        ua.append(ang(a[9:12], b[9:12]))
        gm += (a[15] != b[15])
    q = lambda v, p: sorted(v)[min(len(v)-1, int(len(v)*p))]
    print(f"{s:28} {len(ticks):5} {q(pe,.5):8.2f} {q(pe,.9):8.2f} {max(pe):9.1f} "
          f"{q(ve,.5):8.2f} {q(fa,.5):8.2f} {q(ua,.5):7.2f} {gm:8}")
    worst.append((q(pe,.5), s))
worst.sort(reverse=True)
print("\nworst segments by median position error:")
for e, s in worst[:6]:
    print(f"   {e:8.2f} uu  {s}")
