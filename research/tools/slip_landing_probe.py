"""Turning-wavedash probe: landings binned by SLIP ANGLE (velocity vs nose at
touchdown), real vs v3replay on identical states+controls. A turning wavedash is a
high-slip landing whose tires redirect the velocity; if sim regrips differently from
the real game, the policy learns redirects that fail in-game.

Scores, per slip bin, at t+5/10/20 after touchdown:
  - heading error: angle between real and sim GROUND-VELOCITY directions (deg)
  - speed error:   |real ground speed - sim ground speed| (uu/s)
  - handbrake share of events
"""
import json, math, os, subprocess, sys
import statistics as st

V3 = sys.argv[1]
MESH = "/home/luca/Projects/pulsar2-3.0/build/collision_meshes"
CAPS = sys.argv[2:] or [
    'rlbot-run/pulsar-gco-ts1-bot/debug.3914029.jsonl',
    'rlbot-run/pulsar-gco-ts1-bot/debug.1672672.jsonl',
    'rlbot-run/pulsar-gco-ts1-bot/debug.2095102.jsonl']
PRE, POST = 3, 24

events = []
for cap in CAPS:
    dec = [json.loads(l) for l in open(cap) if '"decision"' in l]
    for i in range(PRE, len(dec) - POST - 1):
        d, pr = dec[i], dec[i - 1]
        if not (pr["g"] == 0 and d["g"] == 1):
            continue
        if not all(int(round((dec[k + 1]["t"] - dec[k]["t"]) * 120)) == 1
                   for k in range(i - PRE, i + POST)):
            continue
        v, f = d["v"], d["f"]
        gs = math.hypot(v[0], v[1])
        if gs < 300:
            continue
        # slip angle in ground plane
        va = math.atan2(v[1], v[0])
        fa = math.atan2(f[1], f[0])
        slip = abs(math.degrees((va - fa + math.pi) % (2 * math.pi) - math.pi))
        if slip > 90:
            slip = 180 - slip  # backwards landings: measure off-axis, not reversed
        hb = any(dec[k]["act_tuple"][7] == 1 for k in range(i, i + 10))
        events.append((dec, i, slip, hb))
print(f"{len(events)} landings (ground speed > 300)")

lines = []
for dec, i, slip, hb in events:
    r0 = dec[i - PRE]
    stt = [*r0["p"], *r0["v"], *r0["f"], *r0["u"], *r0.get("av", [0, 0, 0]), r0["boost"], r0["g"],
           r0.get("hj", 0), r0.get("hdj", 0), r0.get("hf", 0), r0.get("if", 0),
           r0.get("ij", 0), r0.get("atsj", 0)]
    lines.append("R " + " ".join(f"{x:.6f}" for x in stt))
    for k in range(i - PRE, i + POST):
        lines.append("C " + " ".join(f"{x:.4f}" for x in dec[k]["act_tuple"]))
out = subprocess.run([V3, MESH], input="\n".join(lines) + "\n",
                     capture_output=True, text=True, env=os.environ).stdout.splitlines()
S = [l.split()[1:] for l in out if l.startswith("S ")]
N = PRE + POST

BINS = [(0, 8), (8, 20), (20, 45), (45, 91)]
DEPTHS = [5, 10, 20]
res = {b: {dep: [] for dep in DEPTHS} for b in BINS}
hbn = {b: [0, 0] for b in BINS}
for n, (dec, i, slip, hb) in enumerate(events):
    rows = S[n * N:(n + 1) * N]
    b = next(bb for bb in BINS if bb[0] <= slip < bb[1])
    hbn[b][0] += hb
    hbn[b][1] += 1
    for dep in DEPTHS:
        k = PRE - 1 + dep
        real = dec[i + dep]
        sv = [float(x) for x in rows[k][3:6]]
        rs = math.hypot(real["v"][0], real["v"][1])
        ss = math.hypot(sv[0], sv[1])
        if rs < 50 or ss < 50:
            continue
        ha = math.degrees(math.atan2(real["v"][1], real["v"][0])
                          - math.atan2(sv[1], sv[0]))
        ha = abs((ha + 180) % 360 - 180)
        res[b][dep].append((ha, abs(rs - ss)))
print(f"{'slip bin':>10} {'n':>5} {'hb%':>4} " +
      " ".join(f"| t+{d}: hdg_med spd_med" for d in DEPTHS))
for b in BINS:
    n = hbn[b][1]
    if not n:
        continue
    row = f"{str(b):>10} {n:>5} {100*hbn[b][0]/n:>3.0f}% "
    for dep in DEPTHS:
        xs = res[b][dep]
        if xs:
            row += f"|   {st.median([x[0] for x in xs]):6.1f} {st.median([x[1] for x in xs]):7.1f} "
        else:
            row += "|      -       - "
    print(row)
