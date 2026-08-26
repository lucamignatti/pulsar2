"""Wavedash-quality meter v2. Works on ANY per-decision JSONL — RLBot real-game captures
(pulsar-*-bot/debug.*.jsonl) or sim self-play (GigaLearnCrossPlay GGL_XP_JSONL), which is
what makes real-vs-sim comparable on identical weights.

An attempt = ground jump press (rising edge), then an airborne jump press (rising edge)
within 0.6 s at z < 60. Success = stays low (zmax < 70), regrounds within 0.45 s, and
either gains > 50 uu/s of ground speed or holds near the 2300 cap.

v2 (2026-08-25) fixes three artifacts that made v1 report a fake 6x sim-vs-real gap
(v1 read sim 6.7% where the true number was 41.3% vs real 42.3%):
  - sp0 is taken 3 rows BEFORE the airborne press: the sim JSONL's press row already
    carries the dodge impulse in v (logging phase), so v1's same-row sp0 nulled the
    speed-gain criterion for every sim attempt.
  - attempts are defined by INPUT EDGES only (no atsj term): atsj semantics are
    gate-dependent, so v1's attempt population changed with the engine build under test.
  - speed-cap escape: a wavedash executed at 2250+ uu/s cannot gain 50; v1 counted
    every supersonic wavedash as a failure.
  - episodes are split on time resets (CrossPlay JSONL restarts t per episode; v1
    paired presses across resets, producing negative hop times).
Bin by hop duration (press-to-press) when comparing worlds — it is input-side and
engine-independent.
"""
import json
import math
import sys


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


def split_eps(dec):
    eps, cur = [], [dec[0]] if dec else []
    for a, b in zip(dec, dec[1:]):
        if b["t"] < a["t"] - 0.001:
            eps.append(cur)
            cur = []
        cur.append(b)
    eps.append(cur)
    return eps


def score(dec_all):
    def speed(v):
        return math.hypot(v[0], v[1])

    rows = []
    for dec in split_eps(dec_all):
        i = 0
        while i < len(dec) - 1:
            d = dec[i]
            if d["g"] == 1 and d["act_tuple"][5] == 1 and (i == 0 or dec[i - 1]["act_tuple"][5] == 0):
                j = i + 1
                while j < len(dec) and 0 < dec[j]["t"] - d["t"] < 0.6:
                    e = dec[j]
                    if (e["g"] == 0 and e["act_tuple"][5] == 1
                            and dec[j - 1]["act_tuple"][5] == 0 and e["p"][2] < 60):
                        sp0 = speed(dec[max(j - 3, 0)]["v"])
                        # zmax measured only UNTIL reground (v3): measuring through the
                        # full 0.8s window misclassified successful wavedashes whose
                        # follow-up (a jump, a wall) left the ground again.
                        zmax, grounded, spg = 0, None, None
                        for k in range(j, len(dec)):
                            f = dec[k]
                            if f["t"] - e["t"] > 0.8:
                                break
                            if grounded is None:
                                zmax = max(zmax, f["p"][2])
                                if f["g"] == 1 and f["t"] > e["t"] + 0.05:
                                    grounded = f["t"] - e["t"]
                                    spg = speed(f["v"])
                        ok = bool(zmax < 70 and grounded and grounded < 0.45 and spg is not None
                                  and (spg > sp0 + 50 or (spg > 2200 and spg > sp0 - 60)))
                        rows.append((1000 * (e["t"] - d["t"]), 1 if ok else 0))
                        i = j
                        break
                    j += 1
            i += 1
    return rows


if __name__ == "__main__":
    for p in sys.argv[1:]:
        d = load(p)
        rows = score(d)
        n = len(rows)
        okn = sum(o for _, o in rows)
        print(f"{p.split('/')[-1]:<34} dec={len(d):>7} n={n:>4} success={100*okn/max(n,1):5.1f}%")
        for lo, hi in [(0, 100), (100, 200), (200, 300), (300, 450), (450, 600)]:
            sel = [o for h, o in rows if lo <= h < hi]
            if sel:
                print(f"   {lo:>3}-{hi:<3}ms n={len(sel):>4} ok={100*sum(sel)/len(sel):5.1f}%")
