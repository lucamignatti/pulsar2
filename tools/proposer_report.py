#!/usr/bin/env python3
"""Stage-1 goal-proposer validation report.

Reads the JSONL calibration dumps written to <checkpointFolder>/proposer_dumps/
and answers the questions that gate Stage 2:
  1. TRACKING  - does the proposer predict change better than a no-op ("ball stays put") predictor?
  2. ASPIRATION - is it tilted toward the good (w=1) futures, or is it just a forecast?
  3. SANITY    - clamp saturation, goal drift, reachability of proposals, goalward bias.

Usage:
  python3 tools/proposer_report.py <dumps_dir> [--plot] [--last N]

Each dump row: cur (canonical ball 6D at t), goal (proposed 6D), ach (achieved 6D at t+N),
aN (N-step advantage), w (aspiration weight), rho (reachability of goal), practice (0/1).
Positions are normalized (x/4096, y/6000, z/2044); +y is always the attacked net.
"""

import json
import math
import sys
from pathlib import Path


def load_dumps(dumps_dir, last_n=None):
    files = sorted(Path(dumps_dir).glob("itr_*.jsonl"), key=lambda p: int(p.stem.split("_")[1]))
    if last_n:
        files = files[-last_n:]
    by_itr = {}
    for f in files:
        itr = int(f.stem.split("_")[1])
        rows = [json.loads(line) for line in open(f)]
        if rows:
            by_itr[itr] = rows
    return by_itr


def dist3(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def analyze_itr(rows):
    hi = [r for r in rows if r["w"] >= 1.0]
    lo = [r for r in rows if r["w"] < 1.0]

    stats = {
        "n": len(rows),
        "weight_frac": len(hi) / len(rows),
        # 1. TRACKING: proposer error vs the no-op baseline (predict "ball stays where it is").
        # goal_ach_dist < cur_ach_dist means it's modeling dynamics, not parroting the present.
        "goal_ach_dist": mean(dist3(r["goal"], r["ach"]) for r in rows),
        "noop_ach_dist": mean(dist3(r["cur"], r["ach"]) for r in rows),
        # 2. ASPIRATION: the weighted regression should fit the w=1 rows best. If hi-error ~ lo-error,
        # the tilt did nothing and the proposer is a plain forecast (steers nothing).
        "goal_ach_dist_hi": mean(dist3(r["goal"], r["ach"]) for r in hi),
        "goal_ach_dist_lo": mean(dist3(r["goal"], r["ach"]) for r in lo),
        # Aspiration direction: proposals should look like BETTER-than-average futures.
        # Canonical +y = toward the attacked net; compare goal vs achieved vs current.
        "cur_y": mean(r["cur"][1] for r in rows),
        "goal_y": mean(r["goal"][1] for r in rows),
        "ach_y": mean(r["ach"][1] for r in rows),
        "ach_y_hi": mean(r["ach"][1] for r in hi),
        # 3. SANITY
        "rho_mean": mean(r["rho"] for r in rows),
        "clamp_frac": mean(any(abs(v) >= 1.49 for v in r["goal"]) for r in rows),
        "practice_frac": mean(r.get("practice", 0) for r in rows),
        "finite": all(all(math.isfinite(v) for v in r["goal"] + r["ach"] + r["cur"]) and
                      math.isfinite(r["aN"]) and math.isfinite(r["rho"]) for r in rows),
    }
    return stats


def verdicts(s):
    checks = [
        ("finite values", s["finite"], "all goal/ach/aN/rho finite"),
        ("weight frac ~ 0.25", abs(s["weight_frac"] - 0.25) < 0.05,
         f"{s['weight_frac']:.3f}"),
        ("TRACKING: beats no-op", s["goal_ach_dist"] < s["noop_ach_dist"],
         f"goal->ach {s['goal_ach_dist']:.3f} vs cur->ach {s['noop_ach_dist']:.3f}"),
        ("ASPIRATION: fits w=1 best", s["goal_ach_dist_hi"] < s["goal_ach_dist_lo"],
         f"hi {s['goal_ach_dist_hi']:.3f} vs lo {s['goal_ach_dist_lo']:.3f}"),
        ("ASPIRATION: goalward tilt", s["goal_y"] > s["ach_y"],
         f"goal_y {s['goal_y']:.3f} vs ach_y {s['ach_y']:.3f} (aspir. futures ach_y_hi {s['ach_y_hi']:.3f})"),
        ("SANITY: clamp not saturated", s["clamp_frac"] < 0.20,
         f"{s['clamp_frac']:.1%} of goals touch the clamp box"),
    ]
    return checks


def plot(by_itr, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not available; skipping plots)")
        return

    itrs = sorted(by_itr)
    series = {k: [] for k in ("goal_ach_dist", "noop_ach_dist", "goal_ach_dist_hi",
                              "goal_ach_dist_lo", "rho_mean", "clamp_frac")}
    for itr in itrs:
        s = analyze_itr(by_itr[itr])
        for k in series:
            series[k].append(s[k])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0][0]
    ax.plot(itrs, series["goal_ach_dist"], label="goal->achieved")
    ax.plot(itrs, series["noop_ach_dist"], label="no-op baseline (cur->achieved)", ls="--")
    ax.set_title("Tracking: prediction error vs no-op")
    ax.set_xlabel("iteration"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0][1]
    ax.plot(itrs, series["goal_ach_dist_hi"], label="error on w=1 (aspirational) rows")
    ax.plot(itrs, series["goal_ach_dist_lo"], label="error on low-w rows", ls="--")
    ax.set_title("Aspiration: weighted fit should favor w=1")
    ax.set_xlabel("iteration"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1][0]
    ax.plot(itrs, series["rho_mean"], color="tab:green")
    ax.set_title("Reachability of proposed goals (rho mean)")
    ax.set_xlabel("iteration"); ax.grid(alpha=0.3)

    # Top-down field: latest iteration, sampled rows. Blue arrow = what happened
    # (cur -> achieved), red arrow = what was proposed (cur -> goal).
    ax = axes[1][1]
    rows = by_itr[itrs[-1]][:80]
    for r in rows:
        cx, cy = r["cur"][0], r["cur"][1]
        ax.annotate("", xy=(r["ach"][0], r["ach"][1]), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle="->", color="tab:blue", alpha=0.35, lw=0.8))
        ax.annotate("", xy=(r["goal"][0], r["goal"][1]), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle="->", color="tab:red", alpha=0.35, lw=0.8))
    ax.axhline(1.0, color="k", lw=0.5); ax.axhline(-1.0, color="k", lw=0.5)
    ax.axvline(1.0, color="k", lw=0.5); ax.axvline(-1.0, color="k", lw=0.5)
    ax.plot([-0.22, 0.22], [1.0, 1.0], color="tab:orange", lw=3)  # attacked net (+y)
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)
    ax.set_title(f"Top-down, itr {itrs[-1]}: blue=happened, red=proposed (+y = attacked net)")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    print(f"\nPlots written to {out_path}")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print(__doc__)
        sys.exit(1)
    dumps_dir = args[0]
    do_plot = "--plot" in sys.argv
    last_n = None
    if "--last" in sys.argv:
        last_n = int(sys.argv[sys.argv.index("--last") + 1])

    by_itr = load_dumps(dumps_dir, last_n)
    if not by_itr:
        print(f"No itr_*.jsonl files found in {dumps_dir}")
        sys.exit(1)

    itrs = sorted(by_itr)
    print(f"Loaded {len(itrs)} dump files (itr {itrs[0]} .. {itrs[-1]})\n")

    latest = analyze_itr(by_itr[itrs[-1]])
    print(f"=== Latest iteration ({itrs[-1]}) ===")
    passed = 0
    checks = verdicts(latest)
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:32s} {detail}")
        passed += ok
    print(f"\n  {passed}/{len(checks)} gates passed."
          f" Stage 2 wants ALL of: tracking, aspiration (both), sanity.")

    # Trend over the loaded window (is it improving or plateaued?)
    if len(itrs) >= 4:
        first = analyze_itr(by_itr[itrs[0]])
        print(f"\n=== Trend (itr {itrs[0]} -> {itrs[-1]}) ===")
        for key, label in [("goal_ach_dist", "goal->achieved dist"),
                           ("rho_mean", "rho of proposals"),
                           ("clamp_frac", "clamp saturation")]:
            print(f"  {label:24s} {first[key]:+.4f} -> {latest[key]:+.4f}")

    if do_plot:
        plot(by_itr, Path(dumps_dir) / "proposer_report.png")


if __name__ == "__main__":
    main()
