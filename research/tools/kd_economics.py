"""ATTEMPT_ECONOMICS.md: among feasible-free landings, does attending beat skipping
under the current game, matched for difficulty? Decides Lever A (attempt-generation)
vs Lever B (return-side) for closing the knowing-doing gap.

Usage:
  cd research/tools && OMP_NUM_THREADS=4 nice -n 19 ../.venv/bin/python kd_economics.py \
      --ckpt ../data/ckpt70_aimos/<ts>
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import RocketSim as rs
import collect_dataset as cd
from collect_dataset import ArenaEnv, set_obs_size
from collect_dataset_53 import LIVE_RESET_MIX, _reset_with_airplay
from label_landing import simulate_landing
from load_checkpoint_70 import Pulsar70Policy, load_models
from kd_rollout import attendance, FEASIBLE_SPEED, ATTEND  # protocol parity

HERE = Path(__file__).resolve().parent
SEED = 20260816
NUM_ARENAS = 16
TARGET_FRAMES = 120_000
N_ANCHORS = 20_000
DT = None
TOUCH_W = 45        # 3 s outcome window (arena steps)
SELF_W = 20         # 1.33 s whiff window
GOAL_W = 112        # 7.5 s goal window


def collect(policy):
    torch.manual_seed(SEED)
    envs = [ArenaEnv(i, np.random.default_rng(SEED + 1000 + i)) for i in range(NUM_ARENAS)]
    # capture WHICH team scored (bindings pass team kwarg to the goal callback)
    for env in envs:
        env.goal_team = None
        def on_goal(self=env, **kw):
            self.goal_scored = True
            self.goal_team = int(kw.get("team", -1)) if kw.get("team") is not None else -1
        env._on_goal = on_goal
        env.arena.set_goal_score_callback(env._on_goal, None)

    n = TARGET_FRAMES
    out = {
        "phys": np.empty((n, 31), np.float32),
        "v_real": np.empty(n, np.float32),
        "v_exp": np.empty(n, np.float32),
        "headroom": np.empty(n, np.float32),
        "episode": np.empty(n, np.int32),
        "team": np.empty(n, np.int8),
        "touched": np.zeros(n, bool),
    }
    ep_end = {}     # episode_id -> ("goal", scoring_team) | ("other", -1)
    row, t0, last_touch = 0, time.time(), {}
    while row + 2 * NUM_ARENAS <= n:
        obs_l, mask_l, phys_l = [], [], []
        for env in envs:
            o, m, p = env.observe()
            obs_l.append(o); mask_l.append(m); phys_l.append(p)
        obs_b = torch.from_numpy(np.concatenate(obs_l))
        mask_b = torch.from_numpy(np.concatenate(mask_l))
        _, h2 = policy.trunk_forward(obs_b)
        probs = policy.action_probs(h2, mask_b)
        actions = torch.multinomial(probs, 1, True).flatten()
        with torch.no_grad():
            vt = policy.models["CRITIC_TRUNK"](h2)
            v = policy.models["CRITIC"](vt).flatten()
            if "CRITIC2" in policy.models:
                v = (v + policy.models["CRITIC2"](vt).flatten()) / 2
            vexp = policy.models["GAP_EXP"](h2).flatten()
            vdmin = torch.minimum(policy.models["VDAG1"](vt).flatten(),
                                  policy.models["VDAG2"](vt).flatten())
            hroom = torch.relu(vdmin - v)
        for i, env in enumerate(envs):
            prev = last_touch.get(env.idx, -1)
            for k in range(2):
                j = 2 * i + k
                out["phys"][row] = phys_l[i]
                out["v_real"][row] = v[j].item()
                out["v_exp"][row] = vexp[j].item()
                out["headroom"][row] = hroom[j].item()
                out["episode"][row] = env.episode_id
                out["team"][row] = k
                out["touched"][row] = env.last_touch_tick != prev and prev >= 0
                row += 1
            last_touch[env.idx] = env.last_touch_tick
        for i, env in enumerate(envs):
            ep = env.episode_id
            if env.step(actions[2 * i:2 * i + 2].tolist()):
                ep_end[ep] = ("goal", env.goal_team) if env.goal_scored else ("other", -1)
                env.goal_team = None
                env.reset()
        if row % 40_000 < 2 * NUM_ARENAS:
            print(f"  {row:>7,}/{n:,}  {row/(time.time()-t0):,.0f} rows/s", flush=True)
    for k in out:
        out[k] = out[k][:row]
    print(f"collected {row:,} rows, {out['episode'].max()+1} episodes, "
          f"{sum(1 for v in ep_end.values() if v[0]=='goal')} goal-terminated")
    return out, ep_end


def economics(data, anchors, att, ep_end):
    """Per feasible-free EVENT (deduped): attend flag + outcomes after touchdown."""
    episode, team, phys = data["episode"], data["team"], data["phys"]
    ep_rows = {int(e): np.flatnonzero(episode == e) for e in np.unique(episode)}
    row_pos = np.empty(len(episode), np.int64)
    for rows in ep_rows.values():
        row_pos[rows] = np.arange(len(rows))

    events = {}     # (episode, player, q_land) -> reading with smallest t_land kept last
    for idx, rec in att.items():
        if not (rec["feasible"] and rec["free"]):
            continue
        r = anchors[idx]
        q_land = row_pos[r] + 2 * int(round(rec["t_land"] / DT))
        key = (int(episode[r]), int(team[r]), int(q_land) // 2)
        if key not in events or rec["t_land"] < events[key][1]["t_land"]:
            events[key] = (idx, rec)

    recs = []
    for (ep, k, _), (idx, rec) in events.items():
        r = anchors[idx]
        rows = ep_rows[ep]
        q_land = row_pos[r] + 2 * int(round(rec["t_land"] / DT))
        if q_land >= len(rows):
            continue
        my = slice(9, 12) if k == 0 else slice(20, 23)
        op = slice(20, 23) if k == 0 else slice(9, 12)
        sign = 1.0 if k == 0 else -1.0          # canonical: +y = my attacking direction

        # first touch attribution within TOUCH_W arena steps after touchdown
        first, whiff_self = "none", False
        end = min(q_land + 2 * TOUCH_W, len(rows) - 1)
        for q in range(q_land, end + 1, 2):
            rr = rows[q]
            if data["touched"][rr]:
                b = phys[rr, 0:3]
                dm = np.linalg.norm(phys[rr, my] - b)
                do = np.linalg.norm(phys[rr, op] - b)
                first = "self" if dm <= do else "opp"
                if first == "self" and q <= q_land + 2 * SELF_W:
                    whiff_self = True   # actually a CONVERSION marker; see below
                break
        conv_fast = first == "self" and whiff_self

        # ball progress over 3 s (or to episode end)
        qp = min(q_land + 2 * TOUCH_W, len(rows) - 2)
        prog = (phys[rows[qp], 1] - phys[rows[q_land], 1]) * sign

        # signed goal within GOAL_W of touchdown (episode must end in a goal soon after)
        goal = 0
        if ep in ep_end and ep_end[ep][0] == "goal":
            q_end = len(rows) - 1
            if q_end <= q_land + 2 * GOAL_W:
                scorer = ep_end[ep][1]          # 0=blue, 1=orange
                goal = 1 if scorer == k else -1
        recs.append(dict(
            attended=rec["went"], t_land=rec["t_land"], d_now=rec["d_now"],
            req_speed=rec["d_now"] / max(rec["t_land"], 1e-6),
            first=first, conv_fast=bool(conv_fast), prog=float(prog), goal=int(goal),
            v_real=float(data["v_real"][r]), v_exp=float(data["v_exp"][r]),
            H=float(data["headroom"][r])))
    return recs


def summarize(recs):
    A = [x for x in recs if x["attended"]]
    S = [x for x in recs if not x["attended"]]
    out = {"n_attend": len(A), "n_skip": len(S)}
    if len(A) < 20 or len(S) < 20:
        return out

    def stats(g):
        return {
            "first_self": float(np.mean([x["first"] == "self" for x in g])),
            "first_opp": float(np.mean([x["first"] == "opp" for x in g])),
            "prog": float(np.mean([x["prog"] for x in g])),
            "goal": float(np.mean([x["goal"] for x in g])),
            "v_real": float(np.mean([x["v_real"] for x in g])),
            "H": float(np.mean([x["H"] for x in g])),
        }
    out["attend"], out["skip"] = stats(A), stats(S)
    out["whiff_among_attempts"] = float(np.mean([not x["conv_fast"] for x in A]))

    # matched within required-speed quartiles
    qs = np.quantile([x["req_speed"] for x in recs], [0.25, 0.5, 0.75])
    out["matched"] = []
    d_prog, d_self, w = 0.0, 0.0, 0
    for qi in range(4):
        lo = -np.inf if qi == 0 else qs[qi - 1]
        hi = np.inf if qi == 3 else qs[qi]
        Aq = [x for x in A if lo <= x["req_speed"] < hi]
        Sq = [x for x in S if lo <= x["req_speed"] < hi]
        if len(Aq) < 15 or len(Sq) < 15:
            out["matched"].append(None)
            continue
        m = {"q": qi, "n_attend": len(Aq), "n_skip": len(Sq),
             "attend": stats(Aq), "skip": stats(Sq)}
        out["matched"].append(m)
        nw = min(len(Aq), len(Sq))
        d_prog += nw * (m["attend"]["prog"] - m["skip"]["prog"])
        d_self += nw * (m["attend"]["first_self"] - m["skip"]["first_self"])
        w += nw
    if w:
        out["pooled_matched_delta"] = {"prog": d_prog / w, "first_self": d_self / w}
    return out


def main():
    global DT
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frames", type=int, default=TARGET_FRAMES)
    args = ap.parse_args()
    ckpt = Path(args.ckpt)
    globals()["TARGET_FRAMES"] = args.frames

    torch.set_num_threads(4)
    rng = np.random.default_rng(SEED)
    rs.init(str(HERE.parents[1] / "build" / "collision_meshes"))
    DT = cd.TICK_SKIP / 120.0

    policy = Pulsar70Policy(load_models(ckpt))
    set_obs_size(policy.obs_size)
    cd.RESET_MIX = LIVE_RESET_MIX
    ArenaEnv.reset = _reset_with_airplay
    print(f"checkpoint {ckpt.name} ({int(ckpt.name)/1e9:.1f}B)")

    data, ep_end = collect(policy)
    airborne = np.flatnonzero(data["phys"][:, 2] > 300)
    anchors = airborne if len(airborne) <= N_ANCHORS else \
        np.sort(rng.choice(airborne, N_ANCHORS, replace=False))
    arena = rs.Arena(rs.GameMode.SOCCAR)
    landings = [simulate_landing(arena, data["phys"][r, 0:3], data["phys"][r, 3:6],
                                 data["phys"][r, 6:9]) for r in anchors]
    import kd_rollout
    kd_rollout.DT = DT
    att = attendance(data, anchors, landings)
    recs = economics(data, anchors, att, ep_end)
    out = summarize(recs)
    out["checkpoint"] = int(ckpt.name)
    out["n_events"] = len(recs)

    p = HERE.parent / "results" / f"kd_economics_{ckpt.name}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\n===== {ckpt.name}: {len(recs)} feasible-free EVENTS "
          f"(attend {out['n_attend']}, skip {out['n_skip']}) =====")
    if "attend" in out:
        for g in ["attend", "skip"]:
            s = out[g]
            print(f"  {g:6s}: first_self {s['first_self']:.1%}  first_opp {s['first_opp']:.1%}  "
                  f"prog {s['prog']:+7.1f}uu  goal {s['goal']:+.3f}  "
                  f"V {s['v_real']:+.2f}  H {s['H']:.2f}")
        print(f"  whiff among attempts: {out['whiff_among_attempts']:.1%}")
        if "pooled_matched_delta" in out:
            d = out["pooled_matched_delta"]
            print(f"  MATCHED delta (attend - skip): prog {d['prog']:+.1f}uu  "
                  f"first_self {d['first_self']:+.1%}")
        for m in out["matched"]:
            if m:
                print(f"    q{m['q']}: n {m['n_attend']}/{m['n_skip']}  "
                      f"prog {m['attend']['prog']:+6.1f} vs {m['skip']['prog']:+6.1f}  "
                      f"self {m['attend']['first_self']:.1%} vs {m['skip']['first_self']:.1%}")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
