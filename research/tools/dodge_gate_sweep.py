"""Sweep GGL_DODGE_CONTACT_EXT: replay every real dodge-eligible press edge through
v3replay and compare the sim fire-rate-by-z curve against the real game's
(661-event calibration: 0% below z~21, 50% at z~27.5, ~100% from z~31).
Usage: dodge_gate_sweep.py <v3replay> [ext values...]
"""
import json, math, os, subprocess, sys

V3 = sys.argv[1]
EXTS = [float(x) for x in sys.argv[2:]] or [-1, 0, 2, 4, 6, 8, 10]
MESH = "/home/luca/Projects/pulsar2-3.0/build/collision_meshes"
CAPS = ['rlbot-run/pulsar-gco-ts1-bot/debug.3914029.jsonl',
        'rlbot-run/pulsar-gco-ts1-bot/debug.1672672.jsonl',
        'rlbot-run/pulsar-gco-ts1-bot/debug.2095102.jsonl']
PRE, POST = 3, 6

events = []
for cap in CAPS:
    dec = [json.loads(l) for l in open(cap) if '"decision"' in l]
    for i in range(PRE, len(dec) - POST - 1):
        d, pr = dec[i], dec[i - 1]
        if d["act_tuple"][5] != 1 or pr["act_tuple"][5] != 0:
            continue
        if d.get("hj") != 1 or d.get("hf") == 1 or d.get("hdj") == 1:
            continue
        if not all(int(round((dec[k + 1]["t"] - dec[k]["t"]) * 120)) == 1
                   for k in range(i - PRE, i + POST)):
            continue
        at = d["act_tuple"]
        if abs(at[2]) + abs(at[3]) + abs(at[4]) < 0.1:
            continue  # double jumps: sim fire detection uses hasFlipped only
        fired = 0
        for k in range(i, min(i + 6, len(dec))):
            if dec[k].get("as") == 3 and dec[max(k - 1, 0)].get("as") != 3:
                fired = 1
                break
        events.append((dec, i, d["p"][2], fired))
print(f"{len(events)} directional press edges")

lines = []
for dec, i, z, fired in events:
    r0 = dec[i - PRE]
    st = [*r0["p"], *r0["v"], *r0["f"], *r0["u"], *r0.get("av", [0, 0, 0]), r0["boost"], r0["g"],
          r0.get("hj", 0), r0.get("hdj", 0), 0, 0, r0.get("ij", 0), r0.get("atsj", 0)]
    lines.append("R " + " ".join(f"{x:.6f}" for x in st))
    for k in range(i - PRE, i + POST):
        lines.append("C " + " ".join(f"{x:.4f}" for x in dec[k]["act_tuple"]))
inp = "\n".join(lines) + "\n"
N = PRE + POST

BINS = [(17, 21), (21, 25), (25, 29), (29, 33), (33, 45), (45, 900)]

def curve(fire_by_event):
    out = []
    for lo, hi in BINS:
        sel = [f for (_, _, z, _), f in zip(events, fire_by_event) if lo <= z < hi]
        out.append(100 * sum(sel) / len(sel) if sel else float("nan"))
    return out

real_curve = curve([f for (_, _, _, f) in events])
print(f"{'bins':>14}: " + " ".join(f"{lo}-{hi}" for lo, hi in BINS))
print(f"{'REAL':>14}: " + " ".join(f"{v:5.1f}" for v in real_curve))

for ext in EXTS:
    env = os.environ.copy()
    env["GGL_DODGE_CONTACT_EXT"] = str(ext)
    out = subprocess.run([V3, MESH], input=inp, capture_output=True, text=True, env=env
                         ).stdout.splitlines()
    S = [l.split()[1:] for l in out if l.startswith("S ")]
    fires = []
    for n in range(len(events)):
        rows = S[n * N:(n + 1) * N]
        f = 0
        prev = "0"
        for r in rows[PRE - 1:]:
            cur = r[10] if len(r) > 10 else "0"
            if cur == "1" and prev == "0":
                f = 1
                break
            prev = cur
        fires.append(f)
    c = curve(fires)
    # rms distance on bins with data
    d = [abs(a - b) for a, b in zip(c, real_curve) if not (math.isnan(a) or math.isnan(b))]
    agree = sum(1 for f, (_, _, _, rf) in zip(fires, events) if f == rf)
    print(f"{'ext ' + str(ext):>14}: " + " ".join(f"{v:5.1f}" for v in c)
          + f"   mean|d|={sum(d)/len(d):5.1f}  per-event agree {100*agree/len(events):.1f}%")
