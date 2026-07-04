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
        ("SANITY: clamp not saturated", s["clamp_frac"] < 0.20,
         f"{s['clamp_frac']:.1%} of goals touch the clamp box"),
        # NOTE: the old "fits w=1 best" error comparison was variance-confounded (aspirational
        # futures are intrinsically noisier, so even a perfect tilt can show higher error on
        # them) - the real aspiration check is the TILT section below, in displacement space.
    ]
    return checks


def separability(rows):
    """Can the 'fits w=1 best' gate EVER pass at this horizon?

    Compares the future-displacement (achieved - current, 6D) distributions of w=1 vs low-w
    rows via Cohen's d per dimension, bucketed by field half (defensive/offensive cur_y) so
    state-dependent effects don't cancel in the aggregate. If aspirational and ordinary
    futures are statistically identical in ball space (all |d| tiny), no amount of further
    training can separate them - the fix is the horizon/label (shorten horizonSteps, raise
    aspirationPercentile), not more updates. If some |d| is substantial, the signal exists
    and the tilt is learnable: wait for loss plateau or sharpen the weighting.
    """
    dims = ["x", "y", "z", "vx", "vy", "vz"]
    buckets = {
        "all rows": rows,
        "defensive half (cur_y < 0)": [r for r in rows if r["cur"][1] < 0],
        "offensive half (cur_y >= 0)": [r for r in rows if r["cur"][1] >= 0],
    }

    print("\n=== Label separability (is gate 4 winnable at this horizon?) ===")
    max_d_overall = 0.0
    for name, bucket in buckets.items():
        hi = [r for r in bucket if r["w"] >= 1.0]
        lo = [r for r in bucket if r["w"] < 1.0]
        if len(hi) < 30 or len(lo) < 30:
            print(f"  {name:28s} (too few rows: hi={len(hi)} lo={len(lo)})")
            continue

        ds = []
        for i in range(6):
            dh = [r["ach"][i] - r["cur"][i] for r in hi]
            dl = [r["ach"][i] - r["cur"][i] for r in lo]
            mh, ml = mean(dh), mean(dl)
            vh = mean((v - mh) ** 2 for v in dh)
            vl = mean((v - ml) ** 2 for v in dl)
            pooled = math.sqrt((vh * (len(dh) - 1) + vl * (len(dl) - 1)) /
                               max(1, len(dh) + len(dl) - 2))
            ds.append((mh - ml) / pooled if pooled > 1e-9 else 0.0)

        worst = max(range(6), key=lambda i: abs(ds[i]))
        max_d_overall = max(max_d_overall, abs(ds[worst]))
        detail = " ".join(f"{dims[i]}:{ds[i]:+.2f}" for i in range(6))
        print(f"  {name:28s} max |d| = {abs(ds[worst]):.2f} ({dims[worst]})   [{detail}]")

    if max_d_overall < 0.08:
        print("  -> VERDICT: aspirational and ordinary futures look IDENTICAL in ball space at this")
        print("     horizon. More training cannot make gate 4 pass; shorten horizonSteps and/or raise")
        print("     aspirationPercentile, then re-collect.")
    elif max_d_overall < 0.20:
        print("  -> VERDICT: weak but real signal. Tilt is learnable but slow/small - wait for loss")
        print("     plateau, and consider aspirationPercentile 0.75 -> 0.90 to sharpen the label.")
    else:
        print("  -> VERDICT: labels are clearly separable - if gate 4 still fails at loss plateau,")
        print("     the weighting isn't biting (raise aspirationPercentile / lower belowAspirationWeight).")


def rho_report(rows, squash_temp=10.0):
    """Calibrate the Stage-3 Phi-drop drill detector to the live rho distribution.

    Drills are banked when Phi = sigmoid(rho / phiSquashTemp) rises to phiHighThresh then falls
    by phiDropThresh. The shipped 0.7/0.3 assume rho spanning ~0..+10; on the real run rho is
    strongly negative, so the sigmoid sits in its flat tail and the detector never fires. This
    prints the rho percentiles and the Phi they map to, then suggests thresholds that actually
    bank drills (phiHigh near the p85 moment, phiDrop ~= a p85->p40 fall).
    """
    def sig(x):
        return 1.0 / (1.0 + math.exp(-x))

    rhos = sorted(r["rho"] for r in rows)
    if not rhos:
        return
    n = len(rhos)

    def pct(p):
        return rhos[min(n - 1, int(p / 100.0 * n))]

    print(f"\n=== Rho distribution (Stage-3 detector calibration; phiSquashTemp={squash_temp:g}) ===")
    print("  pctile    rho     Phi=sigmoid(rho/T)")
    marks = [5, 25, 40, 50, 75, 85, 95]
    phis = {}
    for p in marks:
        r = pct(p)
        phi = sig(r / squash_temp)
        phis[p] = phi
        print(f"    p{p:<3d}   {r:+7.2f}      {phi:.3f}")

    high_suggest = round(phis[85], 2)
    drop_suggest = round(phis[85] - phis[40], 2)
    print(f"  -> suggested: phiHighThresh ~= {high_suggest:.2f} (p85 moment),"
          f" phiDropThresh ~= {max(0.03, drop_suggest):.2f} (p85->p40 fall)")
    print("     If Proposer/Drill Bank Size stays ~0, lower phiHighThresh; if it fills instantly"
          " with junk, raise it / raise phiDropThresh.")


def tilt_report(rows):
    """THE aspiration check, in displacement space (immune to the group-variance confound).

    For each separable dimension (|Cohen d| >= 0.15 between w=1 and low-w future displacements),
    places the proposer's mean displacement on the [ordinary ... aspirational] axis:
        tilt = (goal_disp - lo_disp) / (hi_disp - lo_disp)
    0.0  = proposes ordinary futures; 0.25 = exactly an UNWEIGHTED forecast (hi rows are 25%
    of data); ~0.87 = matches the training gradient mass on w=1 rows; >= 0.5 = the weighting
    is biting and proposals are majority-aspirational -> PASS.
    Returns True/False, or None if no dimension is separable (see the separability section).
    """
    dims = ["x", "y", "z", "vx", "vy", "vz"]
    hi = [r for r in rows if r["w"] >= 1.0]
    lo = [r for r in rows if r["w"] < 1.0]
    if len(hi) < 30 or len(lo) < 30:
        print("\n=== Aspiration tilt === (too few rows)")
        return None

    entries = []
    for i in range(6):
        dh = [r["ach"][i] - r["cur"][i] for r in hi]
        dl = [r["ach"][i] - r["cur"][i] for r in lo]
        dg = [r["goal"][i] - r["cur"][i] for r in rows]
        mh, ml, mg = mean(dh), mean(dl), mean(dg)
        vh = mean((v - mh) ** 2 for v in dh)
        vl = mean((v - ml) ** 2 for v in dl)
        pooled = math.sqrt((vh * (len(dh) - 1) + vl * (len(dl) - 1)) /
                           max(1, len(dh) + len(dl) - 2))
        d = (mh - ml) / pooled if pooled > 1e-9 else 0.0
        if abs(d) < 0.15 or abs(mh - ml) < 1e-6:
            continue
        entries.append((abs(d), dims[i], mg, mh, ml, (mg - ml) / (mh - ml)))

    print("\n=== Aspiration tilt (pooled dumps; 0=ordinary, 0.25=unweighted forecast, 1=aspirational) ===")
    if not entries:
        print("  No separable dimension (max |d| < 0.15) - tilt is unmeasurable at this horizon;")
        print("  see the separability verdict for the fix.")
        return None

    entries.sort(reverse=True)
    for absd, name, mg, mh, ml, tilt in entries:
        print(f"  dim {name:3s} |d|={absd:.2f}   goal-disp {mg:+.4f}   aspir. {mh:+.4f}   ordinary {ml:+.4f}"
              f"   -> tilt = {tilt:+.2f}")

    top_tilt = entries[0][5]
    ok = top_tilt >= 0.5
    print(f"  [{'PASS' if ok else 'FAIL'}] TILT: proposals are majority-aspirational on the most"
          f" separable dim ({entries[0][1]}: {top_tilt:+.2f}, need >= 0.5)")
    return ok


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
    ax.set_title("Per-group error (variance-confounded; see TILT section for the real check)")
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

    # Separability + tilt over a decent sample: pool the last few dumps (~2.5k rows)
    pooled_rows = [r for itr in itrs[-5:] for r in by_itr[itr]]
    separability(pooled_rows)
    tilt_ok = tilt_report(pooled_rows)
    rho_report(pooled_rows)

    print(f"\n  Stage-2 go/no-go = TRACKING + SANITY gates above + the TILT verdict."
          f" Currently: {'GO' if (passed == len(checks) and tilt_ok) else 'NO-GO'}.")

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
