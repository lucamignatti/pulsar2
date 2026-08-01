"""Controlled state-sweep probe of the four-rung ladder (5.3).

Motivation: on-policy, V_geo reads as nearly CONSTANT (std 0.06 on a 37.3 mean across
160k frames spanning ground play, aerials, and random states) and both optimism gaps
(H, affine h_geo) correlate strongly with -V_real. This probe asks whether any rung
carries structure the on-policy read hides, by evaluating the ladder on hand-built
state families the paper cares about:

  A  ball-height sweep      car grounded 800uu from ball, ball z 100..2000
  B  aerial intercept       car airborne under a high ball, closing (AirDrill-like)
  C  air-dribble carry      ball mid-height rising toward goal, car glued behind
                            (AirPlayState AIR_CARRY - the certified frontier conduct)
  D  flip-reset setup       car roof-to-ball under a high ball, flip spent
  E  ball-to-goal sweep     grounded drive, ball rolled toward opp goal at 5 distances
  F  mastered baseline      kickoff-like grounded chase, ball center field

Each family: N samples with jitter, evaluated for every rung + sigma-norm + HJB
residual. The report reads: which rungs rank frontier families ABOVE the mastered
baseline (optimism), which are flat, and whether V_geo's variation across families
exceeds its on-policy noise floor.

Output: research/results/geo_field_probe_<ckpt>.json
"""

import json
import os
from pathlib import Path

import numpy as np
import torch

import RocketSim as rs
import collect_dataset as cd
from collect_dataset import ArenaEnv, set_obs_size
from load_checkpoint_53 import Pulsar53Policy, load_models, list_checkpoints

SEED = 4242
N_PER = 300
RESULTS = Path(__file__).resolve().parents[1] / "results"


def look_at(f):
    f = f / max(np.linalg.norm(f), 1e-9)
    tr = np.cross([0.0, 0.0, 1.0], f)
    u = np.cross(f, tr); u /= max(np.linalg.norm(u), 1e-9)
    r = np.cross(u, f); r /= max(np.linalg.norm(r), 1e-9)
    return rs.RotMat(rs.Vec(*f), rs.Vec(*r), rs.Vec(*u))


def build_family(env, rng, family, i):
    """Set arena to one sampled state of the family; returns nothing (state is set)."""
    arena = env.arena
    arena.reset_kickoff(seed=int(rng.integers(0, 2**30)))
    cars = arena.get_cars()
    ball = np.array([rng.uniform(-1500, 1500), rng.uniform(-2000, 2000), 93.15])
    bvel = np.zeros(3)
    car_states = []

    if family == "A_ball_height":
        z = np.linspace(100, 2000, N_PER)[i]
        ball[2] = z
        bvel = np.array([0.0, 0.0, rng.uniform(-50, 50)])
        for c in cars:
            theta = rng.uniform(0, 2 * np.pi)
            pos = ball + np.array([np.cos(theta) * 800, np.sin(theta) * 800, 0])
            pos[2] = 17.0
            car_states.append(dict(pos=pos, vel=np.zeros(3), ground=True, boost=80))
    elif family == "B_aerial_intercept":
        ball[2] = rng.uniform(900, 1500)
        bvel = np.array([rng.uniform(-200, 200), rng.uniform(-200, 200), rng.uniform(-100, 100)])
        for c in cars:
            theta = rng.uniform(0, 2 * np.pi)
            pos = ball + np.array([np.cos(theta) * 500, np.sin(theta) * 500, -rng.uniform(400, 700)])
            pos[2] = max(250.0, pos[2])
            f = (ball - pos); f /= np.linalg.norm(f)
            car_states.append(dict(pos=pos, vel=f * rng.uniform(700, 1400), ground=False,
                                   boost=rng.uniform(45, 100), face=ball))
    elif family == "C_air_carry":
        sign = 1.0 if i % 2 == 0 else -1.0
        ball = np.array([rng.uniform(-2000, 2000), sign * rng.uniform(-500, 1500),
                         rng.uniform(350, 800)])
        bvel = np.array([rng.uniform(-200, 200), sign * 900, rng.uniform(150, 450)])
        for c in cars:
            pos = ball - np.array([0, sign * rng.uniform(250, 500), rng.uniform(60, 200)])
            pos[2] = max(300.0, pos[2])
            f = (ball - pos); f /= np.linalg.norm(f)
            car_states.append(dict(pos=pos, vel=bvel + f * 100, ground=False,
                                   boost=rng.uniform(50, 100), face=ball))
    elif family == "D_flip_reset":
        ball = np.array([rng.uniform(-2200, 2200), rng.uniform(-3000, 3000),
                         rng.uniform(1200, 1700)])
        bvel = np.array([rng.uniform(-150, 150), rng.uniform(-150, 150), rng.uniform(-250, 50)])
        for c in cars:
            pos = ball + np.array([rng.uniform(-200, 200), rng.uniform(-200, 200),
                                   -rng.uniform(300, 520)])
            pos[2] = max(300.0, pos[2])
            tb = (ball - pos); tb /= np.linalg.norm(tb)
            car_states.append(dict(pos=pos, vel=tb * rng.uniform(300, 650), ground=False,
                                   boost=rng.uniform(50, 100), roof=ball, spent=True))
    elif family == "E_ball_to_goal":
        dist_idx = i % 5
        dist = [4800, 3600, 2400, 1200, 400][dist_idx]
        ball = np.array([rng.uniform(-800, 800), 5120 - dist, 93.15])
        bvel = np.array([0.0, rng.uniform(500, 1200), 0.0])
        for c in cars:
            pos = ball + np.array([rng.uniform(-200, 200), -rng.uniform(300, 600), 0])
            pos[2] = 17.0
            car_states.append(dict(pos=pos, vel=np.array([0, 900, 0]), ground=True, boost=60))
    else:  # F_mastered
        ball = np.array([rng.uniform(-800, 800), rng.uniform(-800, 800), 93.15])
        bvel = np.array([rng.uniform(-300, 300), rng.uniform(-300, 300), 0.0])
        for c in cars:
            theta = rng.uniform(0, 2 * np.pi)
            pos = ball + np.array([np.cos(theta) * rng.uniform(600, 900), np.sin(theta) * 800, 0])
            pos[2] = 17.0
            car_states.append(dict(pos=pos, vel=np.zeros(3), ground=True, boost=60))

    bs = rs.BallState()
    bs.pos, bs.vel = rs.Vec(*ball), rs.Vec(*bvel)
    arena.ball.set_state(bs)
    for c, st in zip(cars, car_states):
        cs = rs.CarState()
        cs.pos = rs.Vec(*st["pos"])
        cs.vel = rs.Vec(*st["vel"])
        cs.boost = st["boost"]
        if st.get("roof") is not None:
            tb = st["roof"] - st["pos"]; tb /= np.linalg.norm(tb)
            fwd = np.array([tb[1], -tb[0], 0.0])
            if np.linalg.norm(fwd) < 0.1:
                fwd = np.array([1.0, 0, 0])
            f = fwd / np.linalg.norm(fwd)
            r = np.cross(tb, f); r /= max(np.linalg.norm(r), 1e-9)
            u = np.cross(f, r); u /= max(np.linalg.norm(u), 1e-9)
            cs.rot_mat = rs.RotMat(rs.Vec(*f), rs.Vec(*r), rs.Vec(*u))
        elif st.get("face") is not None:
            cs.rot_mat = look_at(st["face"] - st["pos"])
        else:
            to_ball = ball - st["pos"]
            yaw = float(np.arctan2(to_ball[1], to_ball[0]))
            cs.rot_mat = rs.Angle(yaw, 0, 0).as_rot_mat()
        if st.get("spent"):
            cs.has_jumped = True
            cs.has_flipped = True
            cs.air_time = 0.6
            cs.air_time_since_jump = 0.6
        c.set_state(cs)


FAMILIES = ["A_ball_height", "B_aerial_intercept", "C_air_carry",
            "D_flip_reset", "E_ball_to_goal", "F_mastered"]


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    rng = np.random.default_rng(SEED)
    root = Path(os.environ.get("PULSAR_CKPT_ROOT",
                               Path(__file__).resolve().parents[1] / "data" / "ckpt53"))
    ckpt = list_checkpoints(root)[0]
    rs.init(str(Path(__file__).resolve().parents[2] / "build" / "collision_meshes"))
    pol = Pulsar53Policy(load_models(ckpt))
    set_obs_size(pol.obs_size)
    env = ArenaEnv(0, np.random.default_rng(SEED + 1))

    out = {"checkpoint": int(ckpt.name), "n_per_family": N_PER, "families": {}}
    all_obs, fam_of = [], []
    fam_ballz = {}
    for fam in FAMILIES:
        obs_rows, ballz = [], []
        for i in range(N_PER):
            build_family(env, rng, fam, i)
            obs, masks, phys = env.observe()
            obs_rows.append(obs[0])          # blue player's view
            ballz.append(float(phys[2]))
        all_obs.append(np.stack(obs_rows))
        fam_of += [fam] * N_PER
        fam_ballz[fam] = ballz

    for fam, obs_rows in zip(FAMILIES, all_obs):
        t = torch.from_numpy(obs_rows.astype(np.float32))
        lad = pol.ladder(t)
        res = pol.hjb_residual(t)
        _, sig = pol.geo_sigma(t)
        rec = {k: {"mean": float(v.mean()), "std": float(v.std()),
                   "p90": float(np.percentile(v.numpy(), 90))}
               for k, v in lad.items()}
        rec["hjb_resid_sq"] = float((res ** 2).mean())
        rec["sigma_mean"] = float(sig.mean())
        out["families"][fam] = rec

    # cross-family field structure: does V_geo separate families at all?
    cat = torch.from_numpy(np.concatenate(all_obs).astype(np.float32))
    lad_all = pol.ladder(cat)
    vg = lad_all["v_geo"].numpy()
    out["v_geo_between_family_std"] = float(np.std(
        [out["families"][f]["v_geo"]["mean"] for f in FAMILIES]))
    out["v_geo_within_family_std"] = float(np.mean(
        [out["families"][f]["v_geo"]["std"] for f in FAMILIES]))
    # A-family: corr of each rung with ball height (the canonical frontier axis)
    a = slice(0, N_PER)
    lad_a = {k: v.numpy()[a] for k, v in lad_all.items()}
    bz = np.array(fam_ballz["A_ball_height"])
    out["A_corr_with_ball_z"] = {k: float(np.corrcoef(v, bz)[0, 1]) for k, v in lad_a.items()}

    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / f"geo_field_probe_{ckpt.name}.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"wrote {p}\n")
    hdr = ["family"] + ["v_real", "v_exp", "vdag_min", "H", "v_geo", "h_geo", "r_hat"]
    print(f"{hdr[0]:20s}" + "".join(f"{h:>10s}" for h in hdr[1:]))
    for fam in FAMILIES:
        r = out["families"][fam]
        print(f"{fam:20s}" + "".join(f"{r[k]['mean']:>10.3f}" for k in hdr[1:]))
    print("\nA-family corr with ball_z:", {k: round(v, 3) for k, v in out["A_corr_with_ball_z"].items()})
    print("v_geo between-family std:", round(out["v_geo_between_family_std"], 4),
          " within:", round(out["v_geo_within_family_std"], 4))


if __name__ == "__main__":
    main()
