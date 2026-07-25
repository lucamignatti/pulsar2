"""FEAR census — MANUAL research tool ONLY (user rule 2026-07-15: standing
automation is C++; the production census lives IN THE TRAINER — Steer/Census *
and Steer/Fear Panel * wandb panels, Learner.cpp fnSteerUpdate section 7, with
the frozen panel persisted in RUNNING_STATS). Run this by hand only when the
extra depth is wanted (forced-contest selector calibration is the one thing
the in-trainer census cannot do cheaply).

Appends one JSON line per run to results/fear_census.jsonl:

  - decline rate / pursue rate on feasible best-placed 2v2 readings
  - scared-tail fraction (declines with Dz above the pursued median) and
    Dz medians — the reward-quality diagnostic
  - selector-calibration re-check: forced contests of HI vs LO Dz declines
    (catches the mining criterion itself aging — the metric-rot failure class)
  - LONGITUDINAL FEAR PANEL: a frozen set of high-Dz decline states (obs rows,
    created by the first census run after the FEAR_MINE deploy); each run
    re-values V and G on them under the CURRENT checkpoint, z-scored against
    the run's own rollout distribution. zV_panel rising toward 0 over
    checkpoints = the critic unlearning its fear (the deploy working).

Installed as a systemd --user timer via tools/fear_census.sh (daily). Safe to
run manually any time; flock in the wrapper prevents overlap.
"""

import datetime
import json
import time
from pathlib import Path

import numpy as np
import torch

import RocketSim as rs
from fear_decomp import copy_newest
from frontier_validate import reconstruct, roll_and_judge
from load_checkpoint import load_models, rebuild_sequential
from steer_team import SteeredPolicyRho, TeamArenaEnv, rollout_team
from team_decline_probe import decline_readings

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
PANEL_FILE = RESULTS_DIR / "fear_panel_states.npz"
CENSUS_FILE = RESULTS_DIR / "fear_census.jsonl"
PPT = 2
NPL = 2 * PPT
ROWS = 500_000
N_ARENAS = 24
K_PANEL = 150
K_CAL = 50
REPEATS = 2
ROLL_S = 5.0
CAL_NOISE = 100.0


def forced_share(pool_rows, rd, bank, models, seed):
    rng = np.random.default_rng(seed)
    env = TeamArenaEnv(0, PPT, np.random.default_rng(seed))
    pol = SteeredPolicyRho(models)
    reader = opp = 0
    for i in pool_rows:
        r = int(rd["row"][i])
        reader_team = (int(r) % NPL) % 2
        for _ in range(REPEATS):
            reconstruct(env, bank[r // NPL], rng, CAL_NOISE, CAL_NOISE)
            team, finite = roll_and_judge(env, pol, ROLL_S)
            if finite and team is not None:
                if team == reader_team:
                    reader += 1
                else:
                    opp += 1
    return reader / max(reader + opp, 1), reader + opp


def dedupe_top(order_idx, rd, k):
    seen, out = set(), []
    for i in order_idx:
        b = int(rd["row"][i]) // NPL
        if b not in seen:
            seen.add(b)
            out.append(i)
            if len(out) >= k:
                break
    return np.array(out)


def main():
    t0 = time.time()
    torch.set_num_threads(4)
    rs.init(str(HERE.parents[1] / "build" / "collision_meshes"))
    ckpt = copy_newest(HERE / "data" / "census_cache")
    seed = int(ckpt.name) % (2**31 - 1)
    torch.manual_seed(seed)
    models = load_models(ckpt)
    goal_critic = rebuild_sequential(
        torch.jit.load(str(ckpt / "GOAL_CRITIC.lt"), map_location="cpu"))
    print(f"census @ checkpoint {ckpt.name}", flush=True)

    rec = rollout_team(SteeredPolicyRho(models), PPT, ROWS, seed,
                       num_arenas=N_ARENAS, want_h2=True, want_obs=True,
                       want_states=True)
    rd = decline_readings(rec)
    bank = rec["state_bank"]

    with torch.no_grad():
        V = np.concatenate([models["CRITIC"](torch.from_numpy(
            rec["h2"][i:i + 65536].astype(np.float32))).flatten().numpy()
            for i in range(0, len(rec["h2"]), 65536)])
        G = np.concatenate([goal_critic(torch.from_numpy(
            rec["obs"][i:i + 65536].astype(np.float32))).flatten().numpy()
            for i in range(0, len(rec["obs"]), 65536)])
    vm, vs, gm, gs = V.mean(), V.std(), G.mean(), G.std()
    dz = (G - gm) / gs - (V - vm) / vs
    dz_r = dz[rd["row"]]

    bp = rd["feas_self"] & rd["best_placed"]
    med_pursued = float(np.median(dz_r[bp & rd["pursued_self"]]))
    entry = {
        "wall_time": datetime.datetime.now().isoformat(timespec="seconds"),
        "checkpoint": int(ckpt.name),
        "n_readings": int(len(rd["row"])),
        "n_episodes": int(len(np.unique(rd["episode"]))),
        "decline_rate": float(rd["decline"].mean()),
        "pursue_rate_bp": float(rd["pursued_self"][bp].mean()),
        "scared_tail_frac": float((dz_r[bp & rd["decline"]] > med_pursued).mean()),
        "median_dz_pursued": med_pursued,
        "median_dz_declined_bp": float(np.median(dz_r[bp & ~rd["pursued_self"]])),
    }

    # selector-calibration re-check
    dec = np.flatnonzero(bp & rd["decline"])
    order = dec[np.argsort(-dz_r[dec])]
    hi = dedupe_top(order, rd, K_CAL)
    lo = dedupe_top(order[::-1], rd, K_CAL)
    s_hi, n_hi = forced_share(hi, rd, bank, models, seed + 1)
    s_lo, n_lo = forced_share(lo, rd, bank, models, seed + 2)
    entry.update({"cal_hi_share": s_hi, "cal_hi_n": n_hi,
                  "cal_lo_share": s_lo, "cal_lo_n": n_lo})

    # longitudinal fear panel
    if not PANEL_FILE.exists():
        panel = dedupe_top(order, rd, K_PANEL)
        rows = rd["row"][panel]
        np.savez(PANEL_FILE, obs=rec["obs"][rows],
                 dz_at_freeze=dz[rows].astype(np.float32),
                 checkpoint=int(ckpt.name))
        entry["panel_created"] = True
        print(f"fear panel FROZEN: {len(panel)} states @ {ckpt.name}", flush=True)
    p = np.load(PANEL_FILE)
    with torch.no_grad():
        obs_p = torch.from_numpy(p["obs"].astype(np.float32))
        h2_p = models["SHARED_HEAD"](obs_p)
        vP = models["CRITIC"](h2_p).flatten().numpy()
        gP = goal_critic(obs_p).flatten().numpy()
    entry.update({
        "panel_checkpoint": int(p["checkpoint"]),
        "panel_zV": float(((vP - vm) / vs).mean()),
        "panel_zG": float(((gP - gm) / gs).mean()),
        "panel_dz": float(((gP - gm) / gs - (vP - vm) / vs).mean()),
    })

    CENSUS_FILE.parent.mkdir(exist_ok=True)
    with CENSUS_FILE.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"decline {entry['decline_rate']:.1%} | scared tail "
          f"{entry['scared_tail_frac']:.1%} | cal HI {s_hi:.1%} vs LO {s_lo:.1%} | "
          f"panel zV {entry['panel_zV']:+.3f} dz {entry['panel_dz']:+.3f} "
          f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
