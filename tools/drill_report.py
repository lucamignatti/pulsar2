#!/usr/bin/env python3
"""Stage-3 drill-bank eyeball report.

Reads the JSONL dumps written to <checkpointFolder>/drill_dumps/ (one file per dump iteration,
one line per banked drill) and answers the only question gating drill replay: are these REAL
near-misses (aerial whiffs, blown saves, contested balls) or junk (kickoff chaos, dead balls,
opponent bounces the bot never had a play on)?

Usage:
  python3 tools/drill_report.py <drill_dumps_dir> [--last N]

Each row: ball_pos [x,y,z] (uu), ball_vel [x,y,z] (uu/s), goal [6] (normalized canonical),
team (0=blue,1=orange), drop (Phi-drop severity), tries, successes.
Field: x in +-4096, y in +-5120 (own net at -y for blue), z up to ~2044. Kickoff = ball at
origin at rest.
"""

import json
import math
import sys
from pathlib import Path
from collections import Counter


def load(dumps_dir, last_n=None):
    files = sorted(Path(dumps_dir).glob("itr_*.jsonl"), key=lambda p: int(p.stem.split("_")[1]))
    if last_n:
        files = files[-last_n:]
    # Dedup by drill id across dumps (the bank persists; the same drill appears in many dumps)
    by_id = {}
    for f in files:
        for line in open(f):
            r = json.loads(line)
            by_id[r["id"]] = r
    return list(by_id.values()), (files[0].stem, files[-1].stem) if files else (None, None)


def classify(r):
    x, y, z = r["ball_pos"]
    sp = math.sqrt(sum(v * v for v in r["ball_vel"]))
    at_rest = sp < 150
    near_origin = abs(x) < 500 and abs(y) < 500
    aerial = z > 400
    near_net = abs(y) > 3500
    if near_origin and at_rest:
        return "kickoff-ish (dead ball at center) -- SUSPECT"
    if at_rest and z < 120:
        return "dead ball on ground -- SUSPECT"
    if aerial and near_net:
        return "aerial near net (whiff/save) -- GOOD"
    if aerial:
        return "aerial midfield -- GOOD"
    if near_net:
        return "ground near net (save/clear) -- GOOD"
    return "ground midfield / contested"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print(__doc__)
        sys.exit(1)
    last_n = None
    if "--last" in sys.argv:
        last_n = int(sys.argv[sys.argv.index("--last") + 1])

    rows, (f0, f1) = load(args[0], last_n)
    if not rows:
        print(f"No itr_*.jsonl in {args[0]}")
        sys.exit(1)

    print(f"{len(rows)} unique drills across dumps {f0}..{f1}\n")

    cats = Counter(classify(r) for r in rows)
    print("=== drill composition ===")
    good = suspect = 0
    for cat, n in cats.most_common():
        print(f"  {n:5d} ({n/len(rows):5.1%})  {cat}")
        if "GOOD" in cat:
            good += n
        if "SUSPECT" in cat:
            suspect += n
    print(f"\n  clearly-good {good/len(rows):.0%}   suspect {suspect/len(rows):.0%}   "
          f"(rest = contested midfield, ambiguous)")

    def stats(vals):
        vals = sorted(vals)
        n = len(vals)
        return f"min {vals[0]:.0f}  p50 {vals[n//2]:.0f}  max {vals[-1]:.0f}  mean {sum(vals)/n:.0f}"

    print("\n=== distributions ===")
    print(f"  ball z (height): {stats([r['ball_pos'][2] for r in rows])}   (>400 = aerial)")
    print(f"  |ball y| (net):  {stats([abs(r['ball_pos'][1]) for r in rows])}   (>3500 = near a net)")
    print(f"  ball speed:      {stats([math.sqrt(sum(v*v for v in r['ball_vel'])) for r in rows])}")
    drops = [r["drop"] for r in rows]
    ds = sorted(drops)
    print(f"  drop severity:   min {ds[0]:.3f}  p50 {ds[len(ds)//2]:.3f}  max {ds[-1]:.3f}")
    teams = Counter(r["team"] for r in rows)
    print(f"  team split:      blue {teams.get(0,0)}  orange {teams.get(1,0)}   (want ~balanced)")

    tries = [r.get("tries", 0) for r in rows]
    if any(tries):
        print(f"  tries/drill:     {stats(tries)}   (0 everywhere = detection-only, never replayed)")

    print("\n=== verdict ===")
    if suspect / len(rows) > 0.4:
        print("  >40% suspect (dead balls / kickoff chaos). The detector is banking non-plays -")
        print("  tighten phiHighPercentile (0.85 -> 0.90) so only genuinely-reachable moments qualify,")
        print("  and re-check before enabling replay.")
    elif good / len(rows) > 0.3:
        print("  Majority real near-misses (aerial/net plays). Bank looks healthy -> drill replay at")
        print("  DrillSetter weight ~0.1 is justified. Watch Practice Step Fraction + Drill Success Rate.")
    else:
        print("  Mostly contested-midfield, few clear aerial/save states. Not junk, but not obviously")
        print("  the frontier either - eyeball a few raw rows before committing to replay.")


if __name__ == "__main__":
    main()
