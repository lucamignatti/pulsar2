#!/usr/bin/env python3
"""Render / inspect PSD probe-round dumps for backtracking.

Every probe round writes <checkpoints>/psd_rounds/<R>/{POLICY.lt, fitness.json, ...} — a complete,
resumable state. This tool summarises those rounds so you can see how the ES search behaved and,
when a round regressed the agent, identify the last-good round to roll back to.

Usage:
  tools/psd_report.py <checkpoints_dir>                 # table of all rounds
  tools/psd_report.py <checkpoints_dir> --round 12      # detail for one round
  tools/psd_report.py <checkpoints_dir> --rollback 12   # print how to resume from round 12
"""
import argparse, json, os, sys


def load_rounds(rounds_dir):
    rounds = []
    if not os.path.isdir(rounds_dir):
        sys.exit(f"no psd_rounds dir at {rounds_dir}")
    for name in sorted(os.listdir(rounds_dir), key=lambda s: int(s) if s.isdigit() else -1):
        fj = os.path.join(rounds_dir, name, "fitness.json")
        if os.path.isfile(fj):
            with open(fj) as f:
                rounds.append(json.load(f))
    return rounds


def fmt(x, w=8, p=3):
    try:
        return f"{x:>{w}.{p}f}"
    except (TypeError, ValueError):
        return f"{str(x):>{w}}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints")
    ap.add_argument("--round", type=int, default=None)
    ap.add_argument("--rollback", type=int, default=None)
    args = ap.parse_args()

    rounds_dir = os.path.join(args.checkpoints, "psd_rounds")
    rounds = load_rounds(rounds_dir)
    if not rounds:
        sys.exit("no rounds found")

    if args.rollback is not None:
        d = os.path.join(rounds_dir, str(args.rollback))
        print(f"To roll back to round {args.rollback}:")
        print(f"  1. copy {d}/*.lt into your checkpoint's latest numbered dir (POLICY/SHARED_HEAD/CRITIC)")
        print(f"  2. set psd 'round' in RUNNING_STATS.json to {args.rollback}")
        print(f"  3. resume; DESCEND continues from the round-{args.rollback} weights.")
        print("The plain-DESCEND control run (psd.enabled=false on the same checkpoint) is the ultimate fallback.")
        return

    if args.round is not None:
        r = next((x for x in rounds if x.get("round") == args.round), None)
        if not r:
            sys.exit(f"round {args.round} not found")
        print(json.dumps(r, indent=2))
        fit = r.get("fitness", [])
        if fit:
            order = sorted(range(len(fit)), key=lambda i: -fit[i])
            print("\nprobes best->worst:", ", ".join(f"#{i}:{fit[i]:.3f}" for i in order[:8]))
        return

    print(f"{'round':>5} {'sigma':>8} {'K':>3} {'rank':>4} {'fit.mean':>9} {'fit.std':>8} {'fit.max':>8} {'stepNorm':>9} {'mode':>4}")
    for r in rounds:
        fit = r.get("fitness", [])
        mean = sum(fit) / len(fit) if fit else float("nan")
        std = (sum((v - mean) ** 2 for v in fit) / len(fit)) ** 0.5 if fit else float("nan")
        mx = max(fit) if fit else float("nan")
        print(f"{r.get('round',-1):>5} {fmt(r.get('sigma'))} {r.get('K','?'):>3} {r.get('rank','?'):>4} "
              f"{fmt(mean,9)} {fmt(std)} {fmt(mx)} {fmt(r.get('step_norm'),9)} {r.get('fitness_mode','?'):>4}")

    # Flag rounds where the ES step was large but probe spread was ~0 (ES pushing on noise).
    print("\nwatch: rounds with fit.std ~ 0 but large stepNorm = ES stepping on undifferentiated"
          " probes (widen sigma or lengthen G_probe).")


if __name__ == "__main__":
    main()
