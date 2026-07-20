"""Honest progress measurement: match-play Elo vs PERMANENT spaced anchors.

WHY (2026-07-19): Rating/1v1 is measured against a rolling 800M-step pool of
recent selves that drifts with the policy, so it reports pool-relative drift as
absolute progress - measured overstatement ~6x (claimed +187 Elo over 8.6B
steps; real match-play gain +31). Anchors never change, so this battery is the
only trustworthy progress curve on the run.

Protocol: match play (kickoff-only resets, goal-only terminals - the game viz
shows and the distribution that matters; the trainer's drill-heavy reset mix
biases AGAINST the current policy and is the wrong yardstick), sides swapped,
both policies wire-zero (eval parity - PulsarPolicy pads 517 heads with zeros).

Headline metric: real Elo vs the OLDEST anchor. Because the anchor is frozen,
this number is comparable across runs of this script - that is the progress
curve. Also reports, per anchor, the POOL Rating delta and the inflation
(pool delta - real delta).

Anchors accumulate fast (2B spacing ~ every 4h), so the battery evaluates a
bounded LOG-SPACED subset (oldest + newest + spread between); archival keeps
everything.

Usage:  anchor_battery.py [current_ckpt_dir]     (default: copy newest live)
Env:    BATTERY_ROWS (per half, default 100000), BATTERY_MAX_ANCHORS (default 6)
Writes: results/anchor_battery.json (latest) + results/anchor_battery_history.jsonl
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import torch

import RocketSim as rs
from collect_dataset import set_obs_size
from load_checkpoint import PulsarPolicy, copy_checkpoint, load_models
from match_play_eval import cross_match
from steer_test import SteeredPolicy

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS_DIR = HERE / "results"
ANCHOR_DIR = Path(os.environ.get(
    "ANCHOR_DIR", REPO / "build" / "checkpoints_5.0v3_anchors"))
ROWS_PER_HALF = int(os.environ.get("BATTERY_ROWS", 100_000))
MAX_ANCHORS = int(os.environ.get("BATTERY_MAX_ANCHORS", 6))
SEED = 20260719
NEAR_SELF_STEPS = 500_000_000  # anchors within this of current are self-controls


def elo_from_share(share: float) -> float:
    share = min(max(share, 1e-4), 1 - 1e-4)
    return 400.0 * math.log10(share / (1 - share))


def pool_rating(ckpt_dir: Path) -> float:
    try:
        s = json.loads((ckpt_dir / "RUNNING_STATS.json").read_text())
        return float(s.get("skill_ratings", {}).get("1v1", float("nan")))
    except Exception:
        return float("nan")


def pick_anchors(all_ts: list[int], k: int) -> list[int]:
    """Oldest + newest + log-spaced middle (bounded battery runtime)."""
    if len(all_ts) <= k:
        return all_ts
    picked = {all_ts[0], all_ts[-1]}
    span = len(all_ts) - 1
    for i in range(1, k - 1):
        # log spacing: denser near the recent end
        frac = 1 - math.log1p((k - 1 - i)) / math.log1p(k - 1)
        picked.add(all_ts[max(0, min(span, round(frac * span)))])
    return sorted(picked)


def main():
    torch.set_num_threads(4)
    rs.init(str(REPO / "build" / "collision_meshes"))

    if len(sys.argv) > 1:
        cur_dir = Path(sys.argv[1])
    else:
        cur_dir = copy_checkpoint(HERE / "data" / "ckpt_cache")
    cur_models = load_models(cur_dir)
    cur_pol = PulsarPolicy(cur_models)
    set_obs_size(cur_pol.obs_size)
    cur_ts = int(cur_dir.name) if cur_dir.name.isdigit() else int(
        "".join(c for c in cur_dir.name if c.isdigit()) or 0)
    cur_rating = pool_rating(cur_dir)

    all_ts = sorted(int(d.name) for d in ANCHOR_DIR.iterdir()
                    if d.is_dir() and d.name.isdigit())
    if not all_ts:
        raise SystemExit(f"no anchors in {ANCHOR_DIR} - run tools/archive_anchor.sh first")
    use_ts = pick_anchors(all_ts, MAX_ANCHORS)

    print(f"current: {cur_dir.name}  (pool Rating/1v1 = {cur_rating:.1f})")
    print(f"anchors: {len(all_ts)} archived, evaluating {len(use_ts)}: {use_ts}")
    print(f"protocol: match play, {ROWS_PER_HALF:,} rows/half, sides swapped, wire-zero\n")

    rows = []
    for ts in use_ts:
        a_dir = ANCHOR_DIR / str(ts)
        a_models = load_models(a_dir)
        a_rating = pool_rating(a_dir)
        ga = gb = eps = caps = 0
        for half in range(2):
            pA, pB = SteeredPolicy(cur_models), SteeredPolicy(a_models)
            blue, orange = (pA, pB) if half == 0 else (pB, pA)
            goals, e, c = cross_match(blue, orange, ROWS_PER_HALF, SEED + half)
            ga += goals[0] if half == 0 else goals[1]
            gb += goals[1] if half == 0 else goals[0]
            eps += e
            caps += c
        share = ga / max(ga + gb, 1)
        real = elo_from_share(share)
        pool_delta = cur_rating - a_rating
        near_self = abs(cur_ts - ts) < NEAR_SELF_STEPS
        rows.append({
            "anchor_ts": ts, "anchor_rating": a_rating,
            "goals_cur": ga, "goals_anchor": gb, "episodes": eps, "capped": caps,
            "share": share, "real_elo_delta": real,
            "pool_elo_delta": pool_delta, "inflation": pool_delta - real,
            "near_self_control": near_self,
        })
        tag = "  [near-self control, expect ~50%]" if near_self else ""
        print(f"vs {ts:<14} {ga:>4}-{gb:<4} = {share:5.1%}  "
              f"real {real:+7.1f}  pool {pool_delta:+7.1f}  "
              f"inflation {pool_delta - real:+7.1f}  ({eps} eps){tag}")

    scored = [r for r in rows if not r["near_self_control"]]
    oldest = min(rows, key=lambda r: r["anchor_ts"])
    headline = oldest["real_elo_delta"]
    mean_infl = (sum(r["inflation"] for r in scored) / len(scored)) if scored else float("nan")
    print(f"\nHEADLINE  real Elo vs oldest anchor ({oldest['anchor_ts']}): {headline:+.1f}")
    print(f"          mean inflation across scored anchors: {mean_infl:+.1f} Elo")

    out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "current_ckpt": cur_dir.name, "current_ts": cur_ts,
        "current_pool_rating": cur_rating,
        "rows_per_half": ROWS_PER_HALF,
        "headline_real_elo_vs_oldest": headline,
        "oldest_anchor_ts": oldest["anchor_ts"],
        "mean_inflation": mean_infl,
        "anchors": rows,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "anchor_battery.json").write_text(json.dumps(out, indent=2))
    with open(RESULTS_DIR / "anchor_battery_history.jsonl", "a") as f:
        f.write(json.dumps(out) + "\n")
    print(f"\nwrote {RESULTS_DIR/'anchor_battery.json'} (+ appended history)")


if __name__ == "__main__":
    main()
