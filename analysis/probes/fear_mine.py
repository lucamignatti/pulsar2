"""FEAR_MINE — disagreement-mined frontier drills. Design + frozen bars:
FEAR_MINE.md."""

import json
import time
from pathlib import Path

import numpy as np
import torch

import RocketSim as rs
from frontier_validate import reconstruct, roll_and_judge
from load_checkpoint import copy_checkpoint, load_models, rebuild_sequential
from steer_team import NONE, SteeredPolicyRho, TeamArenaEnv, rollout_team
from team_decline_probe import decline_readings

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
SEED = 20260725
PPT = 2
NPL = 2 * PPT
ROWS = 900_000
N_ARENAS = 24
K_POOL = 150
REPEATS = 2
ROLL_S = 5.0
NOISE_POS, NOISE_VEL = 250.0, 250.0


def build_pools(rd, dz_reading, rng):
    """CURRENT (live-parity) and FEAR pools as reading indices, deduped per block."""
    def dedupe(order):
        seen, out = set(), []
        for i in order:
            b = int(rd["row"][i]) // NPL
            if b in seen:
                continue
            seen.add(b)
            out.append(i)
            if len(out) >= K_POOL:
                break
        return np.array(out)

    cur_mask = rd["feas_self"] & (rd["outcome"] == NONE)
    cur_idx = np.flatnonzero(cur_mask)
    stride = max(1, len(cur_idx) // (K_POOL * 3))
    current = dedupe(cur_idx[::stride])

    fear_mask = rd["feas_self"] & rd["best_placed"] & rd["decline"]
    fear_idx = np.flatnonzero(fear_mask)
    fear = dedupe(fear_idx[np.argsort(-dz_reading[fear_idx])])
    return current, fear


def drill_test(pool_idx, rd, bank, models, seed):
    """Reconstruct each pooled state (noise 250/250), roll 5s x REPEATS."""
    rng = np.random.default_rng(seed)
    env = TeamArenaEnv(0, PPT, np.random.default_rng(seed))
    pol = SteeredPolicyRho(models)
    n_ok = n_rolls = reader_first = opp_first = unresolved = 0
    for i in pool_idx:
        r = int(rd["row"][i])
        entry = bank[r // NPL]
        reader_team = int(r % NPL) % 2
        for _ in range(REPEATS):
            reconstruct(env, entry, rng, NOISE_POS, NOISE_VEL)
            team, finite = roll_and_judge(env, pol, ROLL_S)
            n_rolls += 1
            if not finite:
                continue
            n_ok += 1
            if team is None:
                unresolved += 1
            elif team == reader_team:
                reader_first += 1
            else:
                opp_first += 1
    resolved = reader_first + opp_first
    return {
        "n_states": int(len(pool_idx)), "n_rolls": n_rolls,
        "playable_frac": n_ok / max(n_rolls, 1),
        "resolution_rate": resolved / max(n_ok, 1),
        "reader_first_share": reader_first / max(resolved, 1),
        "unresolved_rate": unresolved / max(n_ok, 1),
    }


def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    rs.init(str(HERE.parents[1] / "build" / "collision_meshes"))
    ckpt = copy_checkpoint(HERE / "data" / "ckpt_cache")
    models = load_models(ckpt)
    goal_critic = rebuild_sequential(
        torch.jit.load(str(ckpt / "GOAL_CRITIC.lt"), map_location="cpu"))
    print(f"pinned checkpoint {ckpt.name}", flush=True)
    rng = np.random.default_rng(SEED)

    rec = rollout_team(SteeredPolicyRho(models), PPT, ROWS, SEED,
                       num_arenas=N_ARENAS, want_h2=True, want_obs=True,
                       want_states=True)
    rd = decline_readings(rec)
    bank = rec["state_bank"]
    print(f"rollout {rec['episodes']} eps; {len(rd['row'])} readings; "
          f"bank {bank.shape} ({time.time()-t0:.0f}s)", flush=True)

    with torch.no_grad():
        V = np.concatenate([models["CRITIC"](torch.from_numpy(
            rec["h2"][i:i + 65536].astype(np.float32))).flatten().numpy()
            for i in range(0, len(rec["h2"]), 65536)])
        G = np.concatenate([goal_critic(torch.from_numpy(
            rec["obs"][i:i + 65536].astype(np.float32))).flatten().numpy()
            for i in range(0, len(rec["obs"]), 65536)])
    dz = (G - G.mean()) / G.std() - (V - V.mean()) / V.std()
    dz_reading = dz[rd["row"]]

    # characterization: the fear tail
    bp = rd["feas_self"] & rd["best_placed"]
    med_pursued = float(np.median(dz_reading[bp & rd["pursued_self"]]))
    med_declined = float(np.median(dz_reading[bp & ~rd["pursued_self"]]))
    tail = float((dz_reading[bp & rd["decline"]] > med_pursued).mean())
    print(f"characterization: median dz pursued {med_pursued:+.3f} vs declined "
          f"{med_declined:+.3f}; {tail:.1%} of best-placed declines sit above "
          f"the pursued median (the scared tail)", flush=True)

    current, fear = build_pools(rd, dz_reading, rng)
    med_cur = float(np.median(dz_reading[current]))
    med_fear = float(np.median(dz_reading[fear]))
    print(f"pools: CURRENT {len(current)} (median dz {med_cur:+.3f}) | "
          f"FEAR {len(fear)} (median dz {med_fear:+.3f})", flush=True)

    res = {"checkpoint": int(ckpt.name), "rows": ROWS,
           "char": {"median_dz_pursued": med_pursued, "median_dz_declined": med_declined,
                    "scared_tail_frac": tail},
           "median_dz_current": med_cur, "median_dz_fear": med_fear}
    for name, pool in (("CURRENT", current), ("FEAR", fear)):
        res[name] = drill_test(pool, rd, bank, models, SEED + 7)
        m = res[name]
        print(f"{name}: playable {m['playable_frac']:.1%} resolution "
              f"{m['resolution_rate']:.1%} reader-first {m['reader_first_share']:.1%} "
              f"({time.time()-t0:.0f}s)", flush=True)

    f, c = res["FEAR"], res["CURRENT"]
    res["bars"] = {
        "playability": f["playable_frac"] >= 0.95,
        "elicitation": f["resolution_rate"] >= c["resolution_rate"] - 0.05,
        "balance": 0.30 <= f["reader_first_share"] <= 0.70,
        "distinctness": med_fear - med_cur >= 0.5,
    }
    res["PASS"] = all(res["bars"].values())
    print(f"bars: {res['bars']} -> {'PASS' if res['PASS'] else 'FAIL'}", flush=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"fear_mine_{ckpt.name}.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"saved {out}  ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
