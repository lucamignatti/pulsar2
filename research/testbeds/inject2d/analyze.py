"""Aggregate the arm x seed matrix into the paper-style bucket table.

Per arm (across seeds):
  touch/air/hi per-1k-step rates in training-quarter buckets (mean of seeds)
  ignited     seeds whose final-bucket hi rate clears IGNITE_HI (well above the
              random-policy rate ~0.07/1k)
  ign_step    median steps to first sustained ignition (EMA of hi crosses the
              threshold), over ignited seeds only
  ep_rew      final-bucket mean episode reward (collapse guard)
"""
import glob, json, os, sys
import numpy as np

IGNITE_HI = 0.5  # hi touches / 1k steps, final bucket


def load(path):
    return [json.loads(l) for l in open(path)]


def ema(x, a=0.05):
    out = np.zeros(len(x)); m = 0.0
    for i, v in enumerate(x):
        m = (1 - a) * m + a * v
        out[i] = m
    return out


def main(d="runs"):
    arms = {}
    for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        base = os.path.basename(f)[:-6]
        arm, seed = base.rsplit("_s", 1)
        arms.setdefault(arm, {})[int(seed)] = load(f)

    print(f"{'arm':10s} {'touch q1..q4':>32s} {'air q1..q4':>32s} "
          f"{'hi q1..q4':>32s} {'ign':>5s} {'ign_step':>9s} {'rew_f':>7s}")
    order = []
    for arm, seeds in arms.items():
        tb, ab, hb = [], [], []
        ign, ign_steps, rew_f = 0, [], []
        for s, rows in seeds.items():
            n = len(rows)
            q = lambda k, lo, hi: np.mean([r[k] for r in rows[int(lo*n):int(hi*n)]])
            tb.append([q("touch_1k", i/4, (i+1)/4) for i in range(4)])
            ab.append([q("air_1k", i/4, (i+1)/4) for i in range(4)])
            hb.append([q("hi_1k", i/4, (i+1)/4) for i in range(4)])
            rew_f.append(q("ep_rew", 0.75, 1.0))
            hi = ema([r["hi_1k"] for r in rows])
            if hb[-1][3] >= IGNITE_HI:
                ign += 1
                cross = np.argmax(hi >= IGNITE_HI)
                ign_steps.append(rows[cross]["steps"])
        tb, ab, hb = np.mean(tb, 0), np.mean(ab, 0), np.mean(hb, 0)
        istep = f"{np.median(ign_steps)/1e6:.2f}M" if ign_steps else "-"
        fmt = lambda v: " ".join(f"{x:7.2f}" for x in v)
        print(f"{arm:10s} {fmt(tb):>32s} {fmt(ab):>32s} {fmt(hb):>32s} "
              f"{ign:>3d}/{len(seeds)} {istep:>9s} {np.mean(rew_f):7.2f}")
        order.append((arm, hb[3], ign))
    print("\nranked by final-bucket hi rate:")
    for arm, hi4, ign in sorted(order, key=lambda t: -t[1]):
        print(f"  {arm:10s} hi_q4={hi4:6.2f}  ignited={ign}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs")
