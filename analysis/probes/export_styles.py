"""Export offline-VALIDATED style directions as the trainer's opponent-style library
(roadmap phase 1; consumed via LearnerConfig::CollectSteeringConfig::opponentStylesFile).

Only causally-validated directions ship (Phase-0 program, STEERING_PHASE0_40.md):
  shadow     challenge-vs-shadow direction at NEGATIVE dose - cuts challenge rate
             44%->24% at -2 sigma with functional canaries
  hesitant   commitment direction at NEGATIVE dose - suppresses race-winning ~2.7 sigma
  overcommit commitment direction at mild POSITIVE dose - clean canaries at +0.5..+1
Exploiter directions FAILED their pre-registered offline bar (32-pair contrast, unstable
dose) and are deliberately absent - re-derive with bigger cross-play samples first.

Usage: python export_styles.py [--ckpt <dir>] [--rows N] [--out <path>]
Writes JSON: [{name, vec[512], sigma, alpha_lo, alpha_hi}]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import RocketSim as rs
from load_checkpoint import copy_checkpoint, load_models
from steer_test import SteeredPolicy, rollout
from steer_v2 import derive_direction, possession_readings
from style_contrasts import _matched_diff, label_challenge

HERE = Path(__file__).resolve().parent
SEED = 20260720


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--rows", type=int, default=100_000)
    ap.add_argument("--out", type=Path, default=HERE.parents[1] / "steering_styles.json")
    args = ap.parse_args()

    t0 = time.time()
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    rs.init(str(HERE.parents[1] / "build" / "collision_meshes"))
    ckpt = args.ckpt or copy_checkpoint(HERE / "data" / "ckpt_cache")
    models = load_models(ckpt)
    print(f"pinned checkpoint {ckpt.name}")

    import collect_dataset as cd
    from load_checkpoint import PulsarPolicy
    cd.set_obs_size(PulsarPolicy(models).obs_size)

    base = rollout(SteeredPolicy(models), args.rows, SEED, want_h2=True)
    h2f = base["h2"].astype(np.float32)
    rng = np.random.default_rng(SEED)

    styles = []

    # Commitment direction (v2 possession contrast)
    readings = possession_readings(base)
    v_commit, sig_commit, n_pairs = derive_direction(h2f, readings, rng)
    print(f"commitment: {n_pairs} pairs, sigma {sig_commit:.2f}")
    styles.append({"name": "hesitant", "vec": v_commit.tolist(), "sigma": sig_commit,
                   "alpha_lo": -2.0, "alpha_hi": -1.0})
    styles.append({"name": "overcommit", "vec": v_commit.tolist(), "sigma": sig_commit,
                   "alpha_lo": 0.5, "alpha_hi": 1.0})

    # Challenge-vs-shadow direction (0c); "shadow" = negative dose window
    rows, is_ch, d0, by = label_challenge(base)
    v_ch, n_ch = _matched_diff(h2f, rows[is_ch], rows[~is_ch],
                               [np.concatenate([d0[is_ch], d0[~is_ch]]),
                                np.concatenate([by[is_ch], by[~is_ch]])], rng)
    if v_ch is not None:
        with np.errstate(all="ignore"):
            sig_ch = float(np.std(h2f @ v_ch))
        print(f"challenge: {n_ch} pairs, sigma {sig_ch:.2f}")
        styles.append({"name": "shadow", "vec": v_ch.tolist(), "sigma": sig_ch,
                       "alpha_lo": -2.0, "alpha_hi": -1.0})

    meta = {"checkpoint": int(ckpt.name), "rows": args.rows,
            "note": "offline-validated styles only; see analysis/probes/STEERING_PHASE0_40.md"}
    args.out.write_text(json.dumps(styles, indent=1))
    (args.out.parent / (args.out.stem + "_meta.json")).write_text(json.dumps(meta, indent=2))
    print(f"wrote {len(styles)} styles to {args.out} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
