"""Phase-0c (STEERING_ROADMAP.md): style-contrast prototyping - candidate league styles
and meta-loop battery entries, derived from pure physics labels and validated causally.

Three contrasts (all labelable from rollout physics, no probes):
  challenge : on opponent possession, closed distance to the ball vs held net-side
              position (geometry-matched)
  aerial    : self-touches converted in the AIR vs on the ground; the direction is
              read 0.5s BEFORE the touch (the approach decision), geometry-matched
  depth     : net-side offset during neutral play, top vs bottom quartile,
              ball-position-matched

Success bar per direction (pre-registered): sign-correct shift of the contrast's own
behavior metric at alpha in [1, 2], WITHOUT competence collapse (touch rate and
goals/ep within noise of alpha=0). Directions that pass become steered-league styles
(phase 1) and meta-loop battery entries (phase 4). Attrition is expected and is
itself a finding - style may be less linearly encoded than commitment.

Usage: python style_contrasts.py [--ckpt <dir>] [--rows N]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import RocketSim as rs
import collect_dataset as cd
from load_checkpoint import PulsarPolicy, copy_checkpoint, load_models
from steer_test import SteeredPolicy, rollout
from steer_v2 import possession_metrics

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"

SEED = 20260716
ALPHAS = [-2.0, -1.0, 1.0, 2.0]
ROWS_PER_S = 60          # interleaved rows per second (2 players x 30Hz)
DT = 1 / 30.0


def _episode_index(rec):
    episode = rec["episode"]
    ep_rows = {int(e): np.flatnonzero(episode == e) for e in np.unique(episode)}
    row_pos = np.empty(len(episode), np.int64)
    for rows in ep_rows.values():
        row_pos[rows] = np.arange(len(rows))
    return ep_rows, row_pos


def _self_pos(rec, r):
    sl = slice(20, 23) if rec["team"][r] == 1 else slice(9, 12)
    return rec["phys"][r, sl]


def _canon_y(y, team):
    return y if team == 0 else -y


def _matched_diff(h2, rows_a, rows_b, feats, rng, bins=(5, 3)):
    """Difference of h2 means between groups a/b, quantile-matched on 1-2 features."""
    rows_a, rows_b = np.asarray(rows_a), np.asarray(rows_b)
    all_rows = np.concatenate([rows_a, rows_b])
    F = np.stack([f for f in feats], 1)  # [n_all, n_feat] aligned with all_rows
    codes = np.zeros(len(all_rows), np.int64)
    for j, nb in enumerate(bins[: F.shape[1]]):
        edges = np.quantile(F[:, j], np.linspace(0, 1, nb + 1))[1:-1]
        codes = codes * 10 + np.digitize(F[:, j], edges)
    is_a = np.zeros(len(all_rows), bool)
    is_a[: len(rows_a)] = True
    sel_a, sel_b = [], []
    for c in np.unique(codes):
        a = np.flatnonzero((codes == c) & is_a)
        b = np.flatnonzero((codes == c) & ~is_a)
        m = min(len(a), len(b))
        if m == 0:
            continue
        sel_a += list(rng.choice(a, m, replace=False))
        sel_b += list(rng.choice(b, m, replace=False))
    if not sel_a:
        return None, 0
    Ha = h2[all_rows[sel_a]].astype(np.float32)
    Hb = h2[all_rows[sel_b]].astype(np.float32)
    v = Ha.mean(0) - Hb.mean(0)
    v /= max(np.linalg.norm(v), 1e-8)
    return v, len(sel_a)


# ---------------------------------------------------------------------------
# Contrast 1: challenge-vs-shadow on opponent possession
# ---------------------------------------------------------------------------
LOOKAHEAD_S = 1.5
POSSESSION_S = 2.0
CHALLENGE_DIST = 600.0


def label_challenge(rec, stride=7):
    """Readings: rows under OPPONENT possession (their touch within the last 2s, none of
    ours since). challenge = within 1.5s the player got within 600uu of the ball or
    closed half the gap; shadow = stayed farther AND goal-side. Returns (rows, is_challenge,
    d_now, ball_y_canon) - the last two are the matching features."""
    ep_rows, row_pos = _episode_index(rec)
    touched, team, phys = rec["touched"], rec["team"], rec["phys"]
    look = int(LOOKAHEAD_S * ROWS_PER_S)
    poss = int(POSSESSION_S * ROWS_PER_S)

    rows, is_ch, d_now_f, bally_f = [], [], [], []
    for rows_ep in ep_rows.values():
        t_rows = rows_ep[touched[rows_ep]]
        if len(t_rows) == 0:
            continue
        t_pos = row_pos[t_rows]
        for q in range(0, len(rows_ep) - look, stride):
            r = rows_ep[q]
            # most recent touch before q
            k = np.searchsorted(t_pos, q) - 1
            if k < 0:
                continue
            last_q = t_pos[k]
            if q - last_q > poss:
                continue
            if (last_q % 2) == (q % 2):
                continue  # our touch -> our possession, skip
            ball = phys[r, 0:3]
            me = _self_pos(rec, r)
            d0 = float(np.linalg.norm(ball[:2] - me[:2]))
            if d0 < CHALLENGE_DIST:
                continue  # already on the ball; no decision to read
            # lookahead over MY future rows
            fut = rows_ep[q + 2: q + look: 2]
            if len(fut) < 10:
                continue
            d_fut = np.linalg.norm(phys[fut, 0:2] + 0.0 - np.stack(
                [_self_pos(rec, rr)[:2] for rr in fut]), axis=1)
            challenged = bool((d_fut.min() < CHALLENGE_DIST) or (d_fut.min() < 0.5 * d0))
            if not challenged:
                # shadow requires staying goal-side of the ball (canonical own net = -y)
                tm = team[r]
                gs = [_canon_y(_self_pos(rec, rr)[1], tm) < _canon_y(phys[rr, 1], tm)
                      for rr in fut[::3]]
                if np.mean(gs) < 0.7:
                    continue  # neither committed nor shadowing - ambiguous, drop
            rows.append(r)
            is_ch.append(challenged)
            d_now_f.append(d0)
            bally_f.append(_canon_y(ball[1], team[r]))
    return (np.array(rows), np.array(is_ch), np.array(d_now_f), np.array(bally_f))


def challenge_metric(rec):
    rows, is_ch, _, _ = label_challenge(rec, stride=11)
    return {"challenge_rate": float(is_ch.mean()) if len(rows) else float("nan"),
            "n_possession_readings": int(len(rows))}


# ---------------------------------------------------------------------------
# Contrast 2: ground-vs-aerial conversion
# ---------------------------------------------------------------------------
APPROACH_LEAD_S = 0.5
AERIAL_BALL_Z = 400.0


def label_aerial(rec):
    """Self-touch events; direction read APPROACH_LEAD_S before the touch.
    aerial = touch while airborne with ball z>400. Matching: ball z + distance at
    the approach row."""
    ep_rows, row_pos = _episode_index(rec)
    touched, on_ground, phys = rec["touched"], rec["on_ground"], rec["phys"]
    lead = int(APPROACH_LEAD_S * ROWS_PER_S)

    rows, is_air, ballz_f, d_f = [], [], [], []
    for rows_ep in ep_rows.values():
        for q in np.flatnonzero(touched[rows_ep]):
            r = rows_ep[q]
            if q - lead < 0:
                continue
            ra = rows_ep[q - lead]  # same parity (lead is even)
            aerial = (not on_ground[r]) and phys[r, 2] > AERIAL_BALL_Z
            rows.append(ra)
            is_air.append(bool(aerial))
            ballz_f.append(phys[ra, 2])
            d_f.append(float(np.linalg.norm(phys[ra, 0:2] - _self_pos(rec, ra)[:2])))
    return np.array(rows), np.array(is_air), np.array(ballz_f), np.array(d_f)


def aerial_metric(rec):
    aerial = rec["touched"] & ~rec["on_ground"] & (rec["phys"][:, 2] > AERIAL_BALL_Z)
    n_touch = int(rec["touched"].sum())
    return {"aerial_share_of_touches": float(aerial.sum() / n_touch) if n_touch else float("nan"),
            "n_touches": n_touch}


# ---------------------------------------------------------------------------
# Contrast 3: depth during neutral play
# ---------------------------------------------------------------------------
NEUTRAL_NO_TOUCH_S = 1.0
NEUTRAL_BALL_Y = 2500.0


def label_depth(rec, stride=7):
    """Neutral-play rows (no touch by anyone within 1s, ball mid-field); depth =
    canonical goal-side offset (ball_y - self_y in the attack frame). Top vs bottom
    quartile, matched on canonical ball y."""
    ep_rows, row_pos = _episode_index(rec)
    touched, team, phys = rec["touched"], rec["team"], rec["phys"]
    quiet = int(NEUTRAL_NO_TOUCH_S * ROWS_PER_S)

    rows, depth, bally = [], [], []
    for rows_ep in ep_rows.values():
        t_pos = row_pos[rows_ep[touched[rows_ep]]]
        for q in range(quiet, len(rows_ep), stride):
            k = np.searchsorted(t_pos, q) - 1
            if k >= 0 and q - t_pos[k] < quiet:
                continue
            r = rows_ep[q]
            if abs(phys[r, 1]) > NEUTRAL_BALL_Y:
                continue
            tm = team[r]
            rows.append(r)
            depth.append(_canon_y(phys[r, 1], tm) - _canon_y(_self_pos(rec, r)[1], tm))
            bally.append(_canon_y(phys[r, 1], tm))
    return np.array(rows), np.array(depth), np.array(bally)


def depth_metric(rec):
    rows, depth, _ = label_depth(rec, stride=11)
    return {"neutral_depth_mean": float(np.mean(depth)) if len(rows) else float("nan"),
            "n_neutral_readings": int(len(rows))}


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--rows", type=int, default=100_000)
    ap.add_argument("--contrasts", nargs="*", default=["challenge", "aerial", "depth"])
    args = ap.parse_args()

    t0 = time.time()
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    rs.init(str(HERE.parents[1] / "build" / "collision_meshes"))

    ckpt = args.ckpt or copy_checkpoint(HERE / "data" / "ckpt_cache")
    models = load_models(ckpt)
    policy = PulsarPolicy(models)
    cd.set_obs_size(policy.obs_size)
    print(f"pinned checkpoint {ckpt.name} (obs {policy.obs_size})")

    rng = np.random.default_rng(SEED)
    base = rollout(SteeredPolicy(models), args.rows, SEED, want_h2=True)
    h2f = base["h2"].astype(np.float32)

    # Derive each contrast's direction from the base rollout
    dirs = {}
    if "challenge" in args.contrasts:
        rows, is_ch, d0, by = label_challenge(base)
        v, n = _matched_diff(h2f, rows[is_ch], rows[~is_ch],
                             [np.concatenate([d0[is_ch], d0[~is_ch]]),
                              np.concatenate([by[is_ch], by[~is_ch]])], rng)
        print(f"challenge: {int(is_ch.sum())} challenge / {int((~is_ch).sum())} shadow "
              f"readings -> {n} matched pairs")
        if v is not None:
            dirs["challenge"] = (v, challenge_metric, "challenge_rate")
    if "aerial" in args.contrasts:
        rows, is_air, bz, d = label_aerial(base)
        if is_air.sum() >= 10:
            v, n = _matched_diff(h2f, rows[is_air], rows[~is_air],
                                 [np.concatenate([bz[is_air], bz[~is_air]]),
                                  np.concatenate([d[is_air], d[~is_air]])], rng)
            print(f"aerial: {int(is_air.sum())} air / {int((~is_air).sum())} ground "
                  f"touches -> {n} matched pairs")
            if v is not None:
                dirs["aerial"] = (v, aerial_metric, "aerial_share_of_touches")
        else:
            print(f"aerial: only {int(is_air.sum())} aerial touches - skipping (too thin)")
    if "depth" in args.contrasts:
        rows, depth, by = label_depth(base)
        qlo, qhi = np.quantile(depth, [0.25, 0.75])
        deep, high = rows[depth >= qhi], rows[depth <= qlo]
        v, n = _matched_diff(h2f, deep, high,
                             [np.concatenate([by[depth >= qhi], by[depth <= qlo]])],
                             rng, bins=(6,))
        print(f"depth: {len(deep)} deep / {len(high)} high readings -> {n} matched pairs")
        if v is not None:
            dirs["depth"] = (v, depth_metric, "neutral_depth_mean")

    # Causal validation: sweep each direction, read its own metric + canaries.
    # One shared alpha=0 baseline rollout; every contrast reads its OWN metric off it.
    results = {"checkpoint": int(ckpt.name), "rows": args.rows, "contrasts": {}}
    base_rec = rollout(SteeredPolicy(models), args.rows, SEED + 1)
    base_canaries = {k: v2 for k, v2 in possession_metrics(base_rec).items()
                     if k in ("touch_ratio", "goals_per_episode",
                              "kickoff_first_touch_s", "poss_win", "poss_win_se")}
    for name, (v, metric_fn, key) in dirs.items():
        with np.errstate(all="ignore"):  # spurious Accelerate FP flags (see steer_v2)
            sig = float(np.std(h2f @ v))
        entry = {"sigma": sig, "alphas": {}}
        m0 = {**metric_fn(base_rec), **base_canaries}
        entry["alphas"]["0.0"] = m0
        print(f"{name:>9} a=+0.0: {key} {m0.get(key):.4g}, "
              f"touch {m0['touch_ratio']*100:.2f}%, goals/ep {m0['goals_per_episode']:.2f}, "
              f"possWin {m0['poss_win']:.1%}", flush=True)
        for a in ALPHAS:
            pol = SteeredPolicy(models, v, alpha=a, scale=sig)
            rec = rollout(pol, args.rows, SEED + 1)
            m = {**metric_fn(rec), **{k: v2 for k, v2 in possession_metrics(rec).items()
                                      if k in ("touch_ratio", "goals_per_episode",
                                               "kickoff_first_touch_s", "poss_win", "poss_win_se")}}
            entry["alphas"][str(a)] = m
            print(f"{name:>9} a={a:+.1f}: {key} {m.get(key):.4g}, "
                  f"touch {m['touch_ratio']*100:.2f}%, goals/ep {m['goals_per_episode']:.2f}, "
                  f"possWin {m['poss_win']:.1%}", flush=True)
        results["contrasts"][name] = entry

    results["runtime_s"] = round(time.time() - t0, 1)
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"style_contrasts_{ckpt.name}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nruntime {results['runtime_s']}s; wrote {out.name}")


if __name__ == "__main__":
    main()
