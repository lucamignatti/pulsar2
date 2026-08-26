"""Calibrate the engine's takeoff wheel-contact reach against the REAL FAILURE POPULATION.

dodge_gate_sweep.py scores GGL_DODGE_CONTACT_EXT on the whole fire-rate-by-z curve, which
is dominated by easy presses where every candidate agrees. This scores it on the subset
that actually diverges in game: jump presses taken while RocketSim reports airborne but
the game's packet still reports OnGround. Measured on real ts1 captures (n=591), the real
game answers those presses with DODGE 46% / JUMP 32% / nothing 22%, while the shipped
engine dodges essentially always - the policy therefore learned a press timing that fires
1-3 ticks early in the real game (full flips instead of wavedashes).

For each ext value: restore the captured full pose+jump/flip state PRE ticks before the
press, replay the captured controls through v3replay, and record whether a dodge fired
(hasFlipped edge). Target = the real dodge share on the same events.

usage: contact_window_calib.py <v3replay> [ext values...]
"""
import json
import os
import subprocess
import sys

V3 = sys.argv[1]
EXTS = [float(x) for x in sys.argv[2:]] or [-1, 0, 2, 4, 6, 8]
MESH = "/home/luca/Projects/pulsar2-3.0/build/collision_meshes"
CAPS = [
    "rlbot-run/pulsar-gco-ts1-bot/debug.2095102.jsonl",
    "rlbot-run/pulsar-gco-ts1-bot/debug.109982.jsonl",
    "rlbot-run/pulsar-gco-ts1-bot/debug.3914029.jsonl",
]
PRE, POST = 3, 8


def collect():
    """Press edges inside the mirror=AIR / packet=ONGROUND disagreement window."""
    events = []
    for cap in CAPS:
        dec = []
        for line in open(cap):
            if '"decision"' not in line:
                continue
            d = json.loads(line)
            if "pkt_g" in d and "u" in d:
                dec.append(d)
        n = len(dec)
        for i in range(PRE, n - POST - 1):
            d, pr = dec[i], dec[i - 1]
            if d["act_tuple"][5] != 1 or pr["act_tuple"][5] != 0:
                continue
            if not (d["g"] == 0 and d["pkt_g"] == 1):
                continue
            # Gate-INDEPENDENT availability filter (same as dodge_gate_sweep): the car must
            # actually have a dodge to spend, else the press is a no-op in both worlds and
            # pollutes the denominator. Directional input only - a neutral press is a
            # double jump, which fires on different rules.
            if d.get("hj") != 1 or d.get("hf") == 1 or d.get("hdj") == 1:
                continue
            at = d["act_tuple"]
            if abs(at[2]) + abs(at[3]) + abs(at[4]) < 0.1:
                continue
            if not all(int(round((dec[k + 1]["t"] - dec[k]["t"]) * 120)) == 1
                       for k in range(i - PRE, i + POST)):
                continue
            real_dodge = any(dec[k]["as"] == 3 and dec[max(k - 1, 0)]["as"] != 3
                             for k in range(i, i + POST))
            events.append((dec, i, real_dodge))
    return events


def run(events, ext):
    """Replay every event through one v3replay process at this ext; return sim dodge share."""
    env = dict(os.environ)
    env["GGL_DODGE_CONTACT_EXT"] = str(ext)
    lines = []
    for dec, i, _ in events:
        s = dec[i - PRE]
        lines.append("R " + " ".join(f"{v:.5f}" for v in (
            *s["p"], *s["v"], *s["f"], *s["u"], *s["av"],
            s["boost"], float(s["g"]), float(s["hj"]), float(s["hdj"]),
            float(s["hf"]), float(s.get("if", 0)), float(s.get("ij", 0)), s["atsj"])))
        for k in range(i - PRE, i + POST):
            a = dec[k]["act_tuple"]
            lines.append("C " + " ".join(str(float(x)) for x in a))
    out = subprocess.run([V3, MESH], input="\n".join(lines), env=env,
                         capture_output=True, text=True).stdout.splitlines()
    per = PRE + POST
    sim = []
    for e in range(len(events)):
        rows = out[e * per:(e + 1) * per]
        fired = False
        prev = None
        for r in rows:
            # S px py pz vx vy vz g fx fy fz hasFlipped isFlipping isJumping atsj
            c = r.split()
            if len(c) < 12:
                continue
            hf = int(c[11])
            if prev is not None and hf == 1 and prev == 0:
                fired = True
            prev = hf
        sim.append(fired)
    return sim


if __name__ == "__main__":
    ev = collect()
    real = sum(1 for _, _, r in ev if r)
    print(f"{len(ev)} clean press edges in the disagreement window")
    print(f"REAL dodge share: {real}/{len(ev)} ({100 * real / max(len(ev), 1):.1f}%)  <- target\n")
    print(f"{'ext':>6}  {'sim dodge share':>16}  {'per-event agree':>16}")
    for ext in EXTS:
        sim = run(ev, ext)
        share = 100 * sum(sim) / max(len(sim), 1)
        agree = 100 * sum(1 for s, (_, _, r) in zip(sim, ev) if s == r) / max(len(sim), 1)
        print(f"{ext:>6}  {share:>15.1f}%  {agree:>15.1f}%")
