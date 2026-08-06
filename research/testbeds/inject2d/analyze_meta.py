"""Meta-transfer analysis: did the policy internalize frontier-seeking?

Phase A (static balls) ends at the shift; phase B (high falling balls -- a new
conduct, timed intercept) begins. A merely-scaffolded policy re-ignites at
scaffold speed; a policy that learned "seek high-H and engage" as an internal
behavior keyed to the self-obs should re-ignite FASTER on the second frontier.

Readouts per arm: phase-A ignition step, phase-B recovery step (smoothed air
rate recrossing REIGNITE after the shift dip), final phase-B air/hi rates.
"""
import glob, json, os, sys
import numpy as np

REIGNITE_AIR = 2.0   # air touches / 1k, phase-B recovery threshold
IGNITE_HI = 0.5


def sm(x, k=8):
    return np.convolve(x, np.ones(k) / k, "same")


def main(d="runs_meta"):
    arms = {}
    for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        base = os.path.basename(f)[:-6]
        arm, seed = base.rsplit("_s", 1)
        arms.setdefault(arm, {})[int(seed)] = [json.loads(l) for l in open(f)]

    print(f"{'arm':10s} {'A-ign(M)':>9s} {'B-rec(iters)':>13s} {'B-rec n':>8s} "
          f"{'B air_f':>8s} {'B hi_f':>7s} {'B rew_f':>8s}")
    for arm, seeds in arms.items():
        a_ign, b_rec, b_air, b_hi, b_rew = [], [], [], [], []
        for s, rows in seeds.items():
            ph = np.array([r["phase"] for r in rows])
            shift = int(np.argmax(ph == 2)) if (ph == 2).any() else len(rows)
            air = sm(np.array([r["air_1k"] for r in rows]))
            hi = sm(np.array([r["hi_1k"] for r in rows]))
            # phase A ignition (hi conduct)
            pre = hi[:shift]
            if (pre >= IGNITE_HI).any():
                a_ign.append(rows[int(np.argmax(pre >= IGNITE_HI))]["steps"] / 1e6)
            # phase B recovery (air intercept conduct)
            post = air[shift:]
            if len(post) > 10 and (post >= REIGNITE_AIR).any():
                b_rec.append(int(np.argmax(post >= REIGNITE_AIR)))
            b_air.append(np.mean([r["air_1k"] for r in rows[-len(rows)//8:]]))
            b_hi.append(np.mean([r["hi_1k"] for r in rows[-len(rows)//8:]]))
            b_rew.append(np.mean([r["ep_rew"] for r in rows[-len(rows)//8:]]))
        f1 = lambda v: f"{np.median(v):.2f}" if v else "-"
        print(f"{arm:10s} {f1(a_ign):>9s} {f1(b_rec):>13s} {len(b_rec):>5d}/8 "
              f"{np.mean(b_air):8.2f} {np.mean(b_hi):7.2f} {np.mean(b_rew):8.2f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs_meta")
