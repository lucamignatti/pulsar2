"""Run the arm x seed matrix with bounded concurrency."""
import argparse, itertools, subprocess, sys, time

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--shift", type=float, default=0.0)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    todo = list(itertools.product(args.arms, args.seeds))
    running = []
    t0 = time.time()
    while todo or running:
        while todo and len(running) < args.jobs:
            arm, seed = todo.pop(0)
            p = subprocess.Popen(
                [sys.executable, "train.py", "--arm", arm, "--seed", str(seed),
                 "--steps", str(args.steps), "--shift", str(args.shift), "--out", args.out],
                stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            running.append((arm, seed, p))
            print(f"[{time.time()-t0:6.0f}s] start {arm} s{seed} ({len(todo)} queued)", flush=True)
        for item in running[:]:
            arm, seed, p = item
            if p.poll() is not None:
                running.remove(item)
                tag = "ok" if p.returncode == 0 else f"FAIL rc={p.returncode}"
                print(f"[{time.time()-t0:6.0f}s] done {arm} s{seed} {tag}", flush=True)
        time.sleep(2)
    print(f"all done in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
