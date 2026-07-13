"""Sidecar: derive the commitment direction from the newest checkpoint and write it
where the trainer hot-reloads it (LearnerConfig::steering.vectorPath).

Same extraction as steer_test.py step 1-2 (validated offline: matched went/declined
AUC 0.82-0.89 across checkpoints): self-play rollout -> label feasible free landings
by touchdown lookahead -> match went/declined on (distance, flight-time) bins ->
difference of class means in trunk-output space. Writes atomically (tmp + rename) so
the trainer can never read a torn file.

Usage:
  probe-venv/bin/python derive_steering.py --out ../../build/STEER_VEC.json
  (re-run every few hours during a steered run; each run re-derives from the newest
   checkpoint. Thread-cap + nice as in README.md.)

If the direction quality degrades (AUC < --min-auc), the file is NOT overwritten and
the exit code is 1 - a stale-but-validated direction beats a fresh-but-noisy one.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import RocketSim as rs
from collect_dataset import NUM_ARENAS, ArenaEnv
from label_landing import simulate_landing
from load_checkpoint import PulsarPolicy, copy_checkpoint, load_models
from steer_test import (ATTEND_RADIUS, DT, FEASIBLE_SPEED, MATCH_BINS_D,
                        MATCH_BINS_T, SEED, AIRBORNE_Z, SteeredPolicy,
                        landing_behavior, rollout)

HERE = Path(__file__).resolve().parent


def validate_causally(models, v, sigma, rows: int, alphas=(1.0, 2.0)):
    """The gate that matters: does steering along v actually raise engagement with
    feasible landings? Paired-seed rollouts at alpha 0 vs each candidate alpha.
    (Per-frame held-out AUC is a weak proxy - a direction's read and its causal write
    can disagree, and DID at ckpt 4163149824: AUC 0.68 read, -10.7pp write at +2.)
    Gate at the PRODUCTION alpha; larger alphas are diagnostics."""
    eng = {}
    for a in (0.0, *alphas):
        rec = rollout(SteeredPolicy(models, v, alpha=a, scale=sigma), rows, SEED + 1)
        lb = landing_behavior(rec)
        eng[a] = lb
        print(f"  validate alpha {a:+.1f}: engagement {lb['engagement']:.1%} "
              f"(n={lb['n_feasible_readings']})")
    return {a: eng[a]["engagement"] - eng[0.0]["engagement"] for a in alphas}, eng


def derive(base_rows: int, models, ckpt):
    print(f"deriving from checkpoint {ckpt.name}")

    base = rollout(SteeredPolicy(models), base_rows, SEED, want_h2=True)
    phys, episode, team = base["phys"], base["episode"], base["team"]
    airborne = phys[:, 2] > AIRBORNE_Z

    arena = rs.Arena(rs.GameMode.SOCCAR)
    ep_rows = {int(e): np.flatnonzero(episode == e) for e in np.unique(episode)}
    row_pos = np.empty(len(team), np.int64)
    for rows in ep_rows.values():
        row_pos[rows] = np.arange(len(rows))

    rows_r, went_r, dnow_r, tland_r = [], [], [], []
    for r in np.flatnonzero(airborne):
        res = simulate_landing(arena, phys[r, 0:3], phys[r, 3:6], phys[r, 6:9])
        if res is None:
            continue
        lx, ly, t_land = res
        rows = ep_rows[int(episode[r])]
        q = row_pos[r]
        q_land = q + 2 * int(round(t_land / DT))
        if q_land >= len(rows):
            continue
        L = np.array([lx, ly])
        sl = slice(20, 23) if team[r] == 1 else slice(9, 12)
        d_now = np.linalg.norm(phys[rows[q], sl][:2] - L)
        if d_now / max(t_land, 1e-6) >= FEASIBLE_SPEED:
            continue
        ball_td = phys[rows[q_land], 0:3]
        if np.linalg.norm(ball_td[:2] - L) > 300 or ball_td[2] > 200:
            continue
        rows_r.append(r)
        went_r.append(np.linalg.norm(phys[rows[q_land], sl][:2] - L) <= ATTEND_RADIUS)
        dnow_r.append(d_now)
        tland_r.append(t_land)

    rows_r, went = np.array(rows_r), np.array(went_r)
    d_now, t_land = np.array(dnow_r), np.array(tland_r)
    print(f"feasible free landings: {len(rows_r):,} ({went.mean():.1%} went)")

    rng = np.random.default_rng(SEED)
    d_edges = np.quantile(d_now, np.linspace(0, 1, MATCH_BINS_D + 1))[1:-1]
    t_edges = np.quantile(t_land, np.linspace(0, 1, MATCH_BINS_T + 1))[1:-1]
    bins = np.digitize(d_now, d_edges) * 10 + np.digitize(t_land, t_edges)
    sel_w, sel_d = [], []
    for b in np.unique(bins):
        w = np.flatnonzero((bins == b) & went)
        d = np.flatnonzero((bins == b) & ~went)
        m = min(len(w), len(d))
        if m:
            sel_w += list(rng.choice(w, m, replace=False))
            sel_d += list(rng.choice(d, m, replace=False))

    h2b = base["h2"].astype(np.float32)
    Hw, Hd = h2b[rows_r[sel_w]], h2b[rows_r[sel_d]]

    # HELD-OUT AUC: a difference-of-means in 512-d separates small samples perfectly
    # in-sample (observed: AUC 1.0 at 96/class), so the quality gate must be evaluated
    # on pairs the direction was not fit on. The shipped vector uses ALL pairs.
    from sklearn.metrics import roc_auc_score
    half_w, half_d = len(Hw) // 2, len(Hd) // 2
    v_fit = Hw[:half_w].mean(0) - Hd[:half_d].mean(0)
    v_fit /= np.linalg.norm(v_fit)
    auc = float(roc_auc_score([1] * (len(Hw) - half_w) + [0] * (len(Hd) - half_d),
                              np.concatenate([Hw[half_w:] @ v_fit, Hd[half_d:] @ v_fit])))

    v = Hw.mean(0) - Hd.mean(0)
    v /= np.linalg.norm(v)
    sigma = float(np.std(h2b @ v))
    return {
        "checkpoint": int(ckpt.name),
        "vec": [float(x) for x in v],
        "sigma": sigma,
        "auc_heldout": auc,
        "n_per_class": len(sel_w),
        "created_unix": time.time(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="STEER_VEC.json destination (trainer's steering.vectorPath)")
    ap.add_argument("--rows", type=int, default=60_000)
    ap.add_argument("--gate-alpha", type=float, default=1.0,
                    help="validate at THIS alpha - keep it equal to cfg.steering.alpha")
    ap.add_argument("--min-engagement-gain", type=float, default=0.03,
                    help="refuse to overwrite unless steering at gate-alpha raises landing "
                         "engagement by at least this (absolute) in the validation rollouts")
    ap.add_argument("--candidate", type=str, default=None,
                    help="skip derivation: validate the vector in this existing STEER_VEC.json "
                         "against the CURRENT checkpoint (stale-but-validated fallback)")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    rs.init(str(HERE.parents[1] / "build" / "collision_meshes"))

    ckpt = copy_checkpoint(HERE / "data" / "ckpt_cache")
    models = load_models(ckpt)

    if args.candidate:
        result = json.loads(Path(args.candidate).read_text())
        print(f"validating candidate vector (derived at ckpt {result['checkpoint']}) "
              f"against current ckpt {ckpt.name}")
        result["validated_at_checkpoint"] = int(ckpt.name)
    else:
        result = derive(args.rows, models, ckpt)
        print(f"direction: held-out AUC {result['auc_heldout']:.3f} (diagnostic only), "
              f"sigma {result['sigma']:.2f}, n {result['n_per_class']}/class")

    v = np.array(result["vec"], np.float32)
    gains, _ = validate_causally(models, v, result["sigma"], args.rows * 2 // 3,
                                 alphas=(args.gate_alpha, 2.0))
    result["engagement_gains"] = {str(a): float(g) for a, g in gains.items()}
    gate_gain = gains[args.gate_alpha]
    print(f"causal validation: gain at gate alpha {args.gate_alpha:+.1f} = {gate_gain:+.1%}, "
          f"at +2.0 = {gains[2.0]:+.1%}")

    if gate_gain < args.min_engagement_gain:
        print(f"gain below {args.min_engagement_gain:+.1%} - NOT writing {args.out} "
              f"(keeping the previous vector)")
        sys.exit(1)

    out = Path(args.out)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(result))
    tmp.rename(out)  # atomic on POSIX - the trainer never sees a torn file
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
