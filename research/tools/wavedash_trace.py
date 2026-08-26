"""Deep per-attempt wavedash tracer. One level below wavedash_meter: for each
ground-jump-press -> airborne-jump-press pair it reports whether the DODGE ACTUALLY
FIRED (AirState -> Dodging / has_dodged edge in real; engine isFlipping in sim JSONL),
at what window time, with what stick inputs, and the physical outcome. Separates
"policy pressed at a different time" (obs-parity problem) from "same press, different
result" (physics/gate problem).

AirState (RLBot v5): 0=OnGround 1=Jumping 2=DoubleJumping 3=Dodging 4=InAir
Sim CrossPlay JSONL carries the same fields derived from engine state (as synthesized).
"""
import json, math, sys, statistics as st
from collections import Counter


def load(path):
    out = []
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("type") == "decision":
            out.append(r)
    return out


def speed(v):
    return math.hypot(v[0], v[1])


def trace(dec):
    attempts = []
    i = 0
    while i < len(dec) - 1:
        d = dec[i]
        if d["g"] == 1 and d["act_tuple"][5] == 1 and (i == 0 or dec[i - 1]["act_tuple"][5] == 0):
            # ground jump press (rising edge)
            j = i + 1
            press = None
            while j < len(dec) and dec[j]["t"] - d["t"] < 0.6:
                e = dec[j]
                if (e["g"] == 0 and e["act_tuple"][5] == 1
                        and dec[j - 1]["act_tuple"][5] == 0 and e["p"][2] < 60):
                    press = j
                    break
                j += 1
            if press is not None:
                attempts.append((i, press))
                i = press
        i += 1

    rows = []
    for gi, pi in attempts:
        g0, e = dec[gi], dec[pi]
        # did a dodge fire within 6 ticks of the press?
        fired, fire_lag, mode = 0, None, None
        for k in range(pi, min(pi + 8, len(dec))):
            f = dec[k]
            prev = dec[max(k - 1, 0)]
            # real: AirState -> Dodging, or has_dodged edge. sim: hasFlipped edge.
            if (f.get("as") == 3
                    or (f.get("hd", 0) == 1 and prev.get("hd", 0) == 0)
                    or ("as" not in f and f.get("hf", 0) == 1 and prev.get("hf", 0) == 0)):
                fired, fire_lag = 1, f["t"] - e["t"]
                mode = "dodge"
                break
            if f.get("as") == 2 or ("as" not in f and f.get("hdj", 0) == 1 and prev.get("hdj", 0) == 0):
                fired, fire_lag = 1, f["t"] - e["t"]
                mode = "djump"
                break
        # outcome per meter criteria
        zmax, grounded, sp0, spg = 0, None, speed(e["v"]), None
        for k in range(pi, len(dec)):
            f = dec[k]
            if f["t"] - e["t"] > 0.8:
                break
            zmax = max(zmax, f["p"][2])
            if grounded is None and f["g"] == 1 and f["t"] > e["t"] + 0.05:
                grounded = f["t"] - e["t"]
                spg = speed(f["v"])
        good = zmax < 70 and grounded is not None and grounded < 0.45 and spg and spg > sp0 + 50
        at = e["act_tuple"]
        rows.append({
            "hop_ms": 1000 * (e["t"] - g0["t"]),      # press-to-press (full hop duration)
            "atsj_ms": 1000 * e.get("atsj", 0),        # window-open time at press
            "dt": e.get("dt", None),                    # raw dodge_timeout at press
            "z": e["p"][2], "vz": e["v"][2],
            "pitch": at[2], "yaw": at[3], "roll": at[4],
            "fired": fired, "mode": mode,
            "fire_lag_ms": 1000 * fire_lag if fire_lag is not None else None,
            "ok": int(good), "zmax": zmax,
            "grounded_ms": 1000 * grounded if grounded is not None else None,
            "dsp": (spg - sp0) if spg else None,
        })
    return rows


def summarize(name, rows):
    n = len(rows)
    if not n:
        print(f"{name}: no attempts")
        return
    ok = sum(r["ok"] for r in rows)
    fired = sum(r["fired"] for r in rows)
    print(f"\n=== {name}: n={n} success={100*ok/n:.1f}% dodge-fired={100*fired/n:.1f}% ===")

    def q(xs):
        xs = sorted(xs)
        if not xs:
            return "-"
        return f"p25={xs[len(xs)//4]:.0f} med={xs[len(xs)//2]:.0f} p75={xs[3*len(xs)//4]:.0f}"

    for label, sel in [("ALL", rows),
                       ("ok", [r for r in rows if r["ok"]]),
                       ("fail", [r for r in rows if not r["ok"]]),
                       ("fail+fired", [r for r in rows if not r["ok"] and r["fired"]]),
                       ("fail+nofire", [r for r in rows if not r["ok"] and not r["fired"]])]:
        if not sel:
            continue
        print(f"  {label:<12} n={len(sel):>4}  hop {q([r['hop_ms'] for r in sel])}  "
              f"atsj {q([r['atsj_ms'] for r in sel])}  z {q([r['z'] for r in sel])}  "
              f"vz {q([r['vz'] for r in sel])}")
    # press-time histogram vs fire/success
    print("  hop-duration bins (press-to-press):")
    bins = [(0, 100), (100, 200), (200, 300), (300, 450), (450, 600)]
    for lo, hi in bins:
        sel = [r for r in rows if lo <= r["hop_ms"] < hi]
        if not sel:
            continue
        f = sum(r["fired"] for r in sel)
        o = sum(r["ok"] for r in sel)
        print(f"    {lo:>3}-{hi:<3}ms n={len(sel):>4} fired={100*f/len(sel):>5.1f}% ok={100*o/len(sel):>5.1f}%")
    md = Counter(r["mode"] for r in rows if r["fired"])
    print(f"  fired modes: {dict(md)}")
    # stick input at press
    pit = Counter(round(r["pitch"], 1) for r in rows)
    print(f"  pitch at press: {dict(sorted(pit.items()))}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        rows = trace(load(p))
        summarize(p.split("/")[-1], rows)
