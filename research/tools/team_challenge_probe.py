"""Teammate double-commit probe (2026-07-19): do same-team cars challenge each
other for the ball, and is the Optimistic-Critic Ladder paying them to?

User observation: in team modes, two same-team cars sometimes both challenge the
ball. Three candidate mechanisms, measured here:
  S1 spirit arithmetic: ZeroSum(ts=0.6) still pays the TOUCHER 0.7r vs 0.3r for
     the teammate (own*(1-ts)+teamMean*ts) - a private race-to-touch bonus on
     every touch-class reward (AerialTouch 120 scaffold = the largest).
  S2 the Ladder drive (user's hypothesis): Phi=-(gap_KD+gap_PK) is injected
     per-ROW into advantages with NO team blending (verified: tAdvantages+=inj,
     Learner.cpp) - if approaching the ball shrinks each car's OWN gap, BOTH
     challengers get paid individually; TEAM_SPIRIT never touches it. (RND is
     the same class but smaller + self-annealed.)
  S3 immaturity: PHASE B opened at ~18.9B (concurrent with the Ladder deploy -
     confounded); 1v1 habits transfer; rotation is a late skill.

Measurements:
  A) BEHAVIOR, current vs pre-team baselines (18.88B ~= zero team training,
     15.3B pre-PHASE B): double-commit rate (both teammates <1200uu of ball AND
     both closing >300uu/s), challenge-at-touch (teammate <600uu of ball when I
     touch), teammate crowding (mates <500uu apart with ball <1000uu).
     current >= baselines -> 9.3B of 35%-weight team training did not reduce
     double-commits -> an active mechanism (S1/S2) is suspected over pure S3.
  B) MECHANISM, current ckpt: per-car gap_KD = relu(V_exp - V_real) each step
     (exact offline: fixed loader + GAP_EXP). The drive pays gap REDUCTION
     (undiscounted closing delta). Report the mean per-step payment sign for the
     SECOND (farther) challenger during double-commit windows vs its payment
     elsewhere. Payment(2nd | converging) > payment(2nd | elsewhere) => the
     drive actively rewards joining an already-covered ball (S2 confirmed as a
     contributor; magnitude vs beta gives the dose).
     NOTE gap_PK is omitted (needs live banks); gap_KD is the drive's rung-2
     half and suffices for payment DIRECTION. All rollouts are wire-zero (eval
     parity).

Usage: team_challenge_probe.py <ckpt_dir> [--gkd] (repeat per checkpoint)
Writes results/team_challenge_<name>.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

import RocketSim as rs
from load_checkpoint import PulsarPolicy, load_models
from steer_team import TeamArenaEnv
from wired_eval import rebuild_gap_net

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE.parent / "results"
SEED = 20260720
PPT = 2                      # 2v2
NUM_ARENAS = 12
STEPS = 3000                 # arena-steps per arena => 36k arena-steps, 144k rows

R_COMMIT = 1200.0            # "committed": within this of ball...
V_CLOSE = 300.0              # ...and closing at least this fast (uu/s)
R_TOUCH_CHALLENGE = 600.0    # teammate this close to ball at my touch = challenged
R_CROWD_MATE = 500.0
R_CROWD_BALL = 1000.0


def main():
    ckpt = Path(sys.argv[1])
    want_gkd = "--gkd" in sys.argv
    name = ckpt.name
    torch.set_num_threads(4)
    rs.init(str(HERE.parents[1] / "build" / "collision_meshes"))

    models = load_models(ckpt)
    pol = PulsarPolicy(models)
    gap_exp = rebuild_gap_net(ckpt / "GAP_EXP.lt") if want_gkd else None

    torch.manual_seed(SEED)
    envs = [TeamArenaEnv(i, PPT, np.random.default_rng(SEED + 10 + i))
            for i in range(NUM_ARENAS)]
    nP = 2 * PPT
    teams = [[p for p in range(nP) if p % 2 == t] for t in (0, 1)]
    last_hit = [dict() for _ in envs]

    # tallies
    team_steps = 0
    dbl_commit = solo_commit = 0
    touches = 0
    touch_challenged = 0
    crowd = 0
    # gkd payment accumulators: [second-challenger converging, everything else]
    pay_sum = np.zeros(2)
    pay_n = np.zeros(2)
    gkd_dbl_sum = gkd_dbl_n = gkd_solo_sum = gkd_solo_n = 0.0

    prev = [None] * NUM_ARENAS  # per arena: (ball, pos[nP,3], gkd[nP], committed mask, second_idx per team)

    for step in range(STEPS):
        obs_list, mask_list, states_list, ball_list = [], [], [], []
        for env in envs:
            obs, masks, phys, states = env.observe()
            obs_list.append(obs)
            mask_list.append(masks)
            states_list.append(states)
            ball_list.append(np.array(phys[0][0:3]))
        obs_b = torch.from_numpy(np.concatenate(obs_list))
        mask_b = torch.from_numpy(np.concatenate(mask_list))
        with torch.no_grad():
            h1, h2 = pol.trunk_forward(obs_b)
            probs = pol.action_probs(h2, mask_b)
            actions = torch.multinomial(probs, 1, True).flatten()
            if gap_exp is not None:
                vr = pol.critic(h2).flatten()
                ve = gap_exp(h2).flatten()
                gkd_all = torch.relu(ve - vr).numpy()

        for i, env in enumerate(envs):
            states = states_list[i]
            ball = ball_list[i]
            pos = np.array([s.pos.as_tuple() for s in states])
            vel = np.array([s.vel.as_tuple() for s in states])
            tick = env.arena.tick_count
            gkd = gkd_all[i * nP:(i + 1) * nP] if gap_exp is not None else None

            # touch events (cross_play-style tick tracking)
            touched = np.zeros(nP, bool)
            for p, s in enumerate(states):
                bhi = s.ball_hit_info
                ht = bhi.tick_count_when_hit if bhi.is_valid else -1
                pv = last_hit[i].get((env.episode_id, p), -1)
                if bhi.is_valid and ht > pv and ht > tick - 2 * 8:
                    touched[p] = True
                    last_hit[i][(env.episode_id, p)] = ht

            d_ball = np.linalg.norm(pos - ball, axis=1)
            to_ball = (ball - pos)
            to_ball /= np.maximum(np.linalg.norm(to_ball, axis=1, keepdims=True), 1e-6)
            closing = (vel * to_ball).sum(1)
            committed = (d_ball < R_COMMIT) & (closing > V_CLOSE)

            second = {}
            for t, mem in enumerate(teams):
                team_steps += 1
                c = [p for p in mem if committed[p]]
                if len(c) == 2:
                    dbl_commit += 1
                    second[t] = c[int(d_ball[c[0]] < d_ball[c[1]])]  # farther car
                    if gkd is not None:
                        gkd_dbl_sum += gkd[list(mem)].mean(); gkd_dbl_n += 1
                elif len(c) == 1:
                    solo_commit += 1
                    if gkd is not None:
                        gkd_solo_sum += gkd[list(mem)].mean(); gkd_solo_n += 1
                # crowding
                a, b = mem
                if np.linalg.norm(pos[a] - pos[b]) < R_CROWD_MATE \
                        and min(d_ball[a], d_ball[b]) < R_CROWD_BALL:
                    crowd += 1
                # challenge-at-touch
                for p in mem:
                    if touched[p]:
                        touches += 1
                        mate = b if p == a else a
                        if d_ball[mate] < R_TOUCH_CHALLENGE:
                            touch_challenged += 1

            # drive payment: pay_t = gkd_t - gkd_{t+1} per car (positive = drive pays)
            if gap_exp is not None and prev[i] is not None:
                p_ball, p_pos, p_gkd, p_committed, p_second = prev[i]
                pay = p_gkd - gkd  # per car
                for t, mem in enumerate(teams):
                    for p in mem:
                        bucket = 0 if (t in p_second and p == p_second[t]) else 1
                        pay_sum[bucket] += pay[p]
                        pay_n[bucket] += 1
            prev[i] = (ball, pos, gkd, committed, second) if gap_exp is not None else None

        for i, env in enumerate(envs):
            if env.step(actions[i * nP:(i + 1) * nP].tolist()):
                env.reset()
                prev[i] = None

    out = {
        "checkpoint": name, "ppt": PPT, "arena_steps": STEPS * NUM_ARENAS,
        "team_steps": team_steps,
        "double_commit_rate": dbl_commit / team_steps,
        "solo_commit_rate": solo_commit / team_steps,
        "double_given_any_commit": dbl_commit / max(1, dbl_commit + solo_commit),
        "crowding_rate": crowd / team_steps,
        "touches": touches,
        "challenge_at_touch_rate": touch_challenged / max(1, touches),
    }
    if gap_exp is not None:
        out["gkd_mean_double_committed"] = gkd_dbl_sum / max(1, gkd_dbl_n)
        out["gkd_mean_solo_committed"] = gkd_solo_sum / max(1, gkd_solo_n)
        out["drive_pay_second_challenger"] = pay_sum[0] / max(1, pay_n[0])
        out["drive_pay_elsewhere"] = pay_sum[1] / max(1, pay_n[1])
        out["n_second_challenger_steps"] = int(pay_n[0])

    print(f"\n=== {name} (2v2, {STEPS * NUM_ARENAS:,} arena-steps) ===")
    print(f"double-commit rate:      {out['double_commit_rate']:.3%} of team-steps")
    print(f"solo-commit rate:        {out['solo_commit_rate']:.3%}")
    print(f"P(double | any commit):  {out['double_given_any_commit']:.1%}")
    print(f"crowding rate:           {out['crowding_rate']:.3%}")
    print(f"challenge-at-touch:      {out['challenge_at_touch_rate']:.1%} of {touches} touches")
    if gap_exp is not None:
        print(f"gap_KD when double-committed: {out['gkd_mean_double_committed']:.4f}  "
              f"vs solo: {out['gkd_mean_solo_committed']:.4f}")
        print(f"drive payment (gap_KD drop/step): 2nd challenger {out['drive_pay_second_challenger']:+.5f} "
              f"({out['n_second_challenger_steps']} steps) vs elsewhere {out['drive_pay_elsewhere']:+.5f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / f"team_challenge_{name}.json", "w") as f:
        json.dump(out, f, indent=2, default=float)  # np.float32 tallies


if __name__ == "__main__":
    main()
