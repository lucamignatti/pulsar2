"""MECHANIC discoverability census — which mechanic FRAGMENTS does the policy
already sample by accident? (The general-solution design question: pure
amplification vs chain-seeding per mechanic.)

Fragments detected from a large 1v1 rollout (phys + actions + on_ground):

  wavedash-like   airborne->grounded transition with a JUMP/flip action within
                  2 decisions of landing AND a horizontal speed GAIN >100uu/s
                  across the landing
  flip-cancel-ish jump action while airborne followed within 3 decisions by an
                  opposite-sign pitch action (the cancel signature)
  proto-dribble   self touch with ball 100-200uu above car center while both
                  slow (|ball vel - car vel| < 600) - hood-carry fragments
  ground-jump-at-high-ball   jump within 1s before a ball-z>642 reading
                  (takeoff ATTEMPTS in play - known ~0 conversions)

Counts are per 100k player-steps. Zero-rate fragments need chain seeding
(drills); nonzero-rate fragments are amplification territory for the
rare-event precursor replay.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

import RocketSim as rs
from advanced_obs import ACTION_TABLE
from fear_decomp import copy_newest
from load_checkpoint import load_models
from steer_team import SteeredPolicyRho, rollout_team

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
SEED = 20260731
ROWS = 600_000
NPL = 2


def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    rs.init(str(HERE.parents[1] / "build" / "collision_meshes"))
    ckpt = copy_newest(HERE / "data" / "ckpt_cache")
    models = load_models(ckpt)
    print(f"pinned checkpoint {ckpt.name}", flush=True)

    rec = rollout_team(SteeredPolicyRho(models), 1, ROWS, SEED, num_arenas=24,
                       want_goals=True)  # want_goals records per-row actions
    phys, episode, slot = rec["phys"], rec["episode"], rec["slot"]
    on_ground, touched, actions = rec["on_ground"], rec["touched"], rec["action"]
    n = len(slot)
    jump_flag = ACTION_TABLE[actions.astype(int), 5] > 0.5
    pitch_val = ACTION_TABLE[actions.astype(int), 2]

    ep_rows = {int(e): np.flatnonzero(episode == e) for e in np.unique(episode)}

    wavedash = flipcancel = dribble = takeoff_attempt = 0
    high_ball_readings = 0

    for e, rows in ep_rows.items():
        for parity in (0, 1):
            pr = rows[parity::2]        # one player's rows (self data is at 9:15 per row)
            og = on_ground[pr]
            pos = phys[pr, 9:12]
            vel = phys[pr, 12:15]
            hspeed = np.linalg.norm(vel[:, :2], axis=1)
            jf = jump_flag[pr]
            pv = pitch_val[pr]
            tch = touched[pr]
            ballz = phys[pr, 2]
            bpos = phys[pr, 0:3]
            bvel = phys[pr, 3:6]

            for i in range(2, len(pr) - 3):
                # wavedash-like: land at i (air->ground), jump within i..i+2, speed gain
                if not og[i - 1] and og[i] and jf[i:i + 3].any() \
                        and hspeed[min(i + 3, len(pr) - 1)] > hspeed[i - 1] + 100:
                    wavedash += 1
                # flip-cancel-ish: airborne jump at i, opposite pitch within 3
                if not og[i] and jf[i] and abs(pv[i]) > 0.5:
                    fut = pv[i + 1:i + 4]
                    if len(fut) and (np.sign(fut) == -np.sign(pv[i])).any() \
                            and (np.abs(fut) > 0.5).any():
                        flipcancel += 1
                # proto-dribble: touch with ball just above car, both slow-relative
                if tch[i]:
                    dz = bpos[i, 2] - pos[i, 2]
                    if 100 < dz < 200 and np.linalg.norm(bvel[i] - vel[i]) < 600:
                        dribble += 1
                # takeoff attempt: grounded jump while ball is high overhead-ish
                if og[i] and jf[i] and ballz[i] > 642 \
                        and np.linalg.norm(bpos[i, :2] - pos[i, :2]) < 1200:
                    takeoff_attempt += 1
                if ballz[i] > 642:
                    high_ball_readings += 1

    per = 100_000.0 / n  # loop covers every player-row exactly once
    res = {"checkpoint": int(ckpt.name), "rows": n,
           "high_ball_step_frac": high_ball_readings / n,
           "per_100k_steps": {
               "wavedash_like": wavedash * per,
               "flip_cancel_ish": flipcancel * per,
               "proto_dribble_touches": dribble * per,
               "grounded_jump_at_high_ball": takeoff_attempt * per},
           "raw": {"wavedash_like": wavedash, "flip_cancel_ish": flipcancel,
                   "proto_dribble_touches": dribble,
                   "grounded_jump_at_high_ball": takeoff_attempt}}
    for k, v in res["per_100k_steps"].items():
        print(f"  {k:>28s}: {v:8.1f} /100k steps (raw {res['raw'][k]})", flush=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"mechanic_census_{ckpt.name}.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"saved {out}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
