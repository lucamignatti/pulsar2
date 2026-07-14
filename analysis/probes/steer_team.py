"""Team-mode (1v1/2v2/3v3) offline steering harness.

Extends the 1v1 machinery to N-player arenas with:
  - TeamArenaEnv: trainer-parity reset mix with the TEAM-AWARE setters (one
    contester/climber per team, goal-side supports) - mirrors the C++ 4.0 setters
  - rollout_team: self-play or cross-play (different policy per team), padded obs
  - SteeredPolicyRho: trunk steering with the LIVE rho-band gate port (K uniform
    valid actions -> phi/psi_car cosine -> per-batch quantile band, fail-closed)
  - team possession labels: WON (self touch first) / TEAMMATE / LOST / NONE
    (collective decline - the team-mode frontier reading)

Row layout: rows per arena-step = 2*ppt, player slot = row index within the step,
team = slot % 2 (cars are added B,O,B,O,.. like the trainer's MakeEnv).
phys per row (17): ball pos3 vel3 angVel3 | self pos3 vel3 | boost, onGround.
"""

import numpy as np
import torch

import RocketSim as rs
from advanced_obs import ACTION_TABLE, build_obs_padded, build_pad_index_map, get_action_mask
from label_landing import simulate_landing
from load_checkpoint import PulsarPolicy

TICK_SKIP = 4
ACTION_DELAY = 3
NO_TOUCH_TERMINAL_S = 10.0
EPISODE_CAP_S = 30.0
DT = 1 / 30.0

ARM_Z = 300.0
FEASIBLE_SPEED = 1300.0
RACE_MARGIN_S = 0.5
MATCH_BINS_D = 5
MATCH_BINS_T = 3

NONE, WON, LOST, TEAMMATE = 0, 1, 2, 3

SIDE_WALL_X, BACK_WALL_Y = 4096.0, 5120.0
BALL_RADIUS = 92.75

RESET_MIX = [(0.35, "near"), (0.55, "air"), (0.70, "kickoff"), (1.01, "random")]


# ---------------------------------------------------------------------------
# Team-aware setters (ports of the 4.0 C++ BallNearCarState / AirDrillState /
# RandomState; the 1v1 case degenerates to the pre-team behavior)
# ---------------------------------------------------------------------------
def _face_ball(pos, ball):
    to_ball = ball - pos
    yaw = float(np.arctan2(to_ball[1], to_ball[0]))
    return rs.Angle(yaw, 0, 0).as_rot_mat()


def _ring_place(rng, ball, r_min, r_max, placed, theta_c=None, theta_half=np.pi,
                z=17.0, clamp=(SIDE_WALL_X - 300, BACK_WALL_Y - 300), sep=300.0):
    for _ in range(16):
        theta = (theta_c or 0.0) + rng.uniform(-theta_half, theta_half) \
            if theta_c is not None else rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(r_min, r_max)
        cand = ball + np.array([np.cos(theta) * dist, np.sin(theta) * dist, 0.0])
        cand[2] = z
        if abs(cand[0]) > clamp[0] or abs(cand[1]) > clamp[1]:
            continue
        if any(np.linalg.norm(cand - o) < sep for o in placed):
            continue
        placed.append(cand)
        return cand
    cand = np.array([np.clip(ball[0] + r_min, -clamp[0], clamp[0]),
                     np.clip(ball[1], -clamp[1], clamp[1]), z])
    placed.append(cand)
    return cand


def set_team_ball_near_car(arena, rng, ppt):
    arena.reset_kickoff(seed=int(rng.integers(0, 2**30)))
    bs = rs.BallState()
    ball = np.array([rng.uniform(-2800, 2800), rng.uniform(-3800, 3800), BALL_RADIUS])
    bs.pos = rs.Vec(*ball)
    arena.ball.set_state(bs)

    cars = list(arena.get_cars())
    teams = [i % 2 for i in range(len(cars))]
    contester = {t: rng.integers(0, ppt) for t in (0, 1)}
    placed = []
    # contesters first (ring never crowded out), then goal-side supports
    order = sorted(range(len(cars)),
                   key=lambda i: 0 if (i // 2) == contester[teams[i]] else 1)
    for i in order:
        t = teams[i]
        if (i // 2) == contester[t]:
            pos = _ring_place(rng, ball, 600, 900, placed)
        else:
            own_goal_y = -BACK_WALL_Y if t == 0 else BACK_WALL_Y
            theta_c = float(np.arctan2(own_goal_y - ball[1], 0 - ball[0]))
            pos = _ring_place(rng, ball, 1800, 3200, placed, theta_c, np.pi / 2)
        cs = rs.CarState()
        cs.pos = rs.Vec(*pos)
        cs.rot_mat = _face_ball(pos, ball)
        cs.boost = rng.uniform(0, 100)
        cars[i].set_state(cs)


def set_team_air_drill(arena, rng, ppt):
    arena.reset_kickoff(seed=int(rng.integers(0, 2**30)))
    bs = rs.BallState()
    ball = np.array([rng.uniform(-2600, 2600), rng.uniform(-3400, 3400), rng.uniform(900, 1500)])
    bs.pos = rs.Vec(*ball)
    bs.vel = rs.Vec(rng.uniform(-200, 200), rng.uniform(-200, 200), rng.uniform(-100, 100))
    arena.ball.set_state(bs)

    cars = list(arena.get_cars())
    teams = [i % 2 for i in range(len(cars))]
    climber = {t: rng.integers(0, ppt) for t in (0, 1)}
    placed = []
    clamp = (SIDE_WALL_X - 300, BACK_WALL_Y - 300)
    for i, car in enumerate(cars):
        t = teams[i]
        cs = rs.CarState()
        if (i // 2) == climber[t]:
            pos = None
            for _ in range(16):
                theta = rng.uniform(0, 2 * np.pi)
                horiz = rng.uniform(350, 750)
                cand = ball + np.array([np.cos(theta) * horiz, np.sin(theta) * horiz, 0.0])
                cand[2] = max(250.0, ball[2] - rng.uniform(400, 900))
                if abs(cand[0]) > clamp[0] or abs(cand[1]) > clamp[1]:
                    continue
                if any(np.linalg.norm(cand - o) < 350 for o in placed):
                    continue
                pos = cand
                break
            if pos is None:
                pos = np.array([np.clip(ball[0] + 500, -clamp[0], clamp[0]),
                                np.clip(ball[1], -clamp[1], clamp[1]),
                                max(250.0, ball[2] - 600)])
            placed.append(pos)
            f = ball - pos
            f = f / max(np.linalg.norm(f), 1e-9)
            tr = np.cross([0.0, 0.0, 1.0], f)
            u = np.cross(f, tr)
            u /= max(np.linalg.norm(u), 1e-9)
            r = np.cross(u, f)
            r /= max(np.linalg.norm(r), 1e-9)
            cs.pos = rs.Vec(*pos)
            cs.rot_mat = rs.RotMat(rs.Vec(*f), rs.Vec(*r), rs.Vec(*u))
            cs.vel = rs.Vec(*(f * rng.uniform(700, 1400)))
        else:
            own_goal_y = -BACK_WALL_Y if t == 0 else BACK_WALL_Y
            shadow = np.array([ball[0], ball[1], 0.0])
            theta_c = float(np.arctan2(own_goal_y - shadow[1], 0 - shadow[0]))
            pos = _ring_place(rng, shadow, 1500, 3000, placed, theta_c, np.pi / 2, sep=350.0)
            cs.pos = rs.Vec(*pos)
            cs.rot_mat = _face_ball(pos, ball)
        cs.boost = rng.uniform(45, 100)
        car.set_state(cs)


def set_team_random(arena, rng):
    """RandomState(true, true, false) port (mode-agnostic)."""
    arena.reset_kickoff(seed=int(rng.integers(0, 2**30)))
    X_MAX, Y_MAX, Z_MAX, CAR_Z_MIN = 3900, 4800, 1820, 150

    def rand_norm():
        v = rng.uniform(-1, 1, 3)
        return v / max(np.linalg.norm(v), 1e-9)

    bs = rs.BallState()
    bs.pos = rs.Vec(*rng.uniform([-X_MAX, -Y_MAX, BALL_RADIUS], [X_MAX, Y_MAX, Z_MAX]))
    bs.vel = rs.Vec(*(rand_norm() * rng.uniform(0, 4000)))
    bs.ang_vel = rs.Vec(*rng.uniform(-4, 4, 3))
    arena.ball.set_state(bs)
    for car in arena.get_cars():
        cs = rs.CarState()
        pos = rng.uniform([-X_MAX, -Y_MAX, CAR_Z_MIN], [X_MAX, Y_MAX, Z_MAX])
        vel = rand_norm() * rng.uniform(0, 2300)
        ang_vel = rand_norm() * 5.5
        yaw, pitch, roll = rng.uniform(-np.pi, np.pi), rng.uniform(-np.pi / 2, np.pi / 2), rng.uniform(-np.pi, np.pi)
        if rng.random() > 0.5:
            pos[2], pitch, roll, vel[2], ang_vel = 17, 0, 0, 0, np.zeros(3)
        cs.pos, cs.vel, cs.ang_vel = rs.Vec(*pos), rs.Vec(*vel), rs.Vec(*ang_vel)
        cs.rot_mat = rs.Angle(yaw, pitch, roll).as_rot_mat()
        cs.boost = rng.uniform(0, 100)
        car.set_state(cs)


# ---------------------------------------------------------------------------
class TeamArenaEnv:
    """One NvN arena with the trainer's per-arena bookkeeping."""

    def __init__(self, idx, ppt, rng):
        self.idx, self.ppt, self.rng = idx, ppt, rng
        self.n_players = 2 * ppt
        self.arena = rs.Arena(rs.GameMode.SOCCAR)
        self.cars = []
        for _ in range(ppt):
            self.cars.append(self.arena.add_car(rs.Team.BLUE))
            self.cars.append(self.arena.add_car(rs.Team.ORANGE))
        self.pad_map = build_pad_index_map(self.arena)
        self.pads = self.arena.get_boost_pads()
        self.goal_scored = False
        self.arena.set_goal_score_callback(self._on_goal, None)
        self.episode_id = -1
        self.reset()

    def _on_goal(self, **kwargs):
        self.goal_scored = True
        # bindings pass the SCORING team; map to 0=blue, 1=orange
        team = kwargs.get("team")
        self.goal_team = 0 if team == rs.Team.BLUE else 1 if team == rs.Team.ORANGE else None

    def reset(self, episode_counter=[0]):
        u = self.rng.random()
        kind = next(k for w, k in RESET_MIX if u < w)
        self.is_kickoff = kind == "kickoff"
        if kind == "near":
            set_team_ball_near_car(self.arena, self.rng, self.ppt)
        elif kind == "air":
            set_team_air_drill(self.arena, self.rng, self.ppt)
        elif kind == "kickoff":
            self.arena.reset_kickoff(seed=int(self.rng.integers(0, 2**30)))
        else:
            set_team_random(self.arena, self.rng)
        zero = rs.CarControls()
        for car in self.cars:
            car.set_controls(zero)
        self.prev_actions = np.zeros((self.n_players, 8), dtype=np.float32)
        self.goal_scored = False
        self.goal_team = None
        self.steps = 0
        self.last_touch_tick = self.arena.tick_count
        self.episode_id = episode_counter[0]
        episode_counter[0] += 1

    def observe(self):
        ball = self.arena.ball.get_state()
        states = [car.get_state() for car in self.cars]
        active = np.empty(len(self.pad_map), bool)
        cooldown = np.empty(len(self.pad_map), np.float32)
        for i, j in enumerate(self.pad_map):
            st = self.pads[j].get_state()
            active[i], cooldown[i] = st.is_active, st.cooldown

        obs, masks, phys = [], [], []
        for p in range(self.n_players):
            team = p % 2
            mates = [states[q] for q in range(self.n_players) if q % 2 == team and q != p]
            opps = [states[q] for q in range(self.n_players) if q % 2 != team]
            obs.append(build_obs_padded(states[p], mates, opps, ball, self.prev_actions[p],
                                        active, cooldown, team == 1, self.rng))
            masks.append(get_action_mask(states[p]))
            phys.append(np.concatenate([
                ball.pos.as_tuple(), ball.vel.as_tuple(), ball.ang_vel.as_tuple(),
                states[p].pos.as_tuple(), states[p].vel.as_tuple(),
                [states[p].boost, float(states[p].is_on_ground)],
            ]).astype(np.float32))

        for st in states:
            bhi = st.ball_hit_info
            if bhi.is_valid:
                self.last_touch_tick = max(self.last_touch_tick, bhi.tick_count_when_hit)
        return np.stack(obs), np.stack(masks), phys, states

    def step(self, action_indices):
        self.arena.step(ACTION_DELAY)
        for p, act_idx in enumerate(action_indices):
            elems = ACTION_TABLE[act_idx]
            ctrl = rs.CarControls()
            ctrl.throttle, ctrl.steer = float(elems[0]), float(elems[1])
            ctrl.pitch, ctrl.yaw, ctrl.roll = float(elems[2]), float(elems[3]), float(elems[4])
            ctrl.jump, ctrl.boost, ctrl.handbrake = bool(elems[5]), bool(elems[6]), bool(elems[7])
            self.cars[p].set_controls(ctrl)
            self.prev_actions[p] = elems
        self.arena.step(TICK_SKIP - ACTION_DELAY)
        self.steps += 1
        no_touch = (self.arena.tick_count - self.last_touch_tick) > NO_TOUCH_TERMINAL_S * 120
        capped = self.steps >= EPISODE_CAP_S * 120 / TICK_SKIP
        return self.goal_scored or no_touch or capped


# ---------------------------------------------------------------------------
class SteeredPolicyRho(PulsarPolicy):
    """Trunk steering with the LIVE rho-band gate port (PPOLearner::InferActions
    parity): K uniform valid actions -> phi embedding vs psi_car(zeros) cosine,
    per-batch quantile band [lo,hi], fail-closed below 16 rows. alpha=0 or v=None
    -> plain policy."""

    def __init__(self, models, v=None, alpha=0.0, scale=1.0,
                 rho_gate=True, rho_lo=0.2, rho_hi=0.8, rho_k=8):
        super().__init__(models)
        self.v = None if v is None else torch.as_tensor(v, dtype=torch.float32)
        self.alpha, self.scale = alpha, scale
        self.rho_gate, self.rho_lo, self.rho_hi, self.rho_k = rho_gate, rho_lo, rho_hi, rho_k
        self.psi_car = models.get("REACH_PSI_CAR")
        if self.psi_car is not None:
            with torch.no_grad():
                g = self.psi_car(torch.zeros(1, 6))
                self.psi0 = (g / g.norm(dim=-1, keepdim=True).clamp_min(1e-6)).squeeze(0)
        self.last_inband_frac = float("nan")

    @torch.no_grad()
    def act(self, obs, masks):
        h1, h2 = self.trunk_forward(obs)
        h2_pol = h2
        if self.v is not None and self.alpha != 0.0:
            n = h2.shape[0]
            gate = torch.ones(n)
            if self.rho_gate and self.phi is not None and self.psi_car is not None:
                if n >= 16:
                    maskF = masks.float().clamp_min(1e-9)
                    acts = torch.multinomial(maskF, self.rho_k, True)          # [n,K]
                    trunk_rep = h2.repeat_interleave(self.rho_k, 0)
                    onehot = torch.nn.functional.one_hot(acts.flatten(), masks.shape[1]).float()
                    sa = self.phi(torch.cat([trunk_rep, onehot], -1))
                    sa = sa / sa.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                    rho = (sa @ self.psi0).view(n, self.rho_k).mean(-1)
                    lo, hi = torch.quantile(rho, self.rho_lo), torch.quantile(rho, self.rho_hi)
                    gate = ((rho >= lo) & (rho <= hi)).float()
                else:
                    gate = torch.zeros(n)  # fail closed, live parity
            self.last_inband_frac = float(gate.mean())
            h2_pol = h2 + gate.unsqueeze(-1) * (self.alpha * self.scale) * self.v
        return h2, self.sample_actions(h2_pol, masks)


def rollout_team(policies, ppt, n_rows, seed, num_arenas=12, want_h2=False,
                 want_states=False, want_masks=False, want_goals=False):
    """policies: one policy for all rows, or (blue_policy, orange_policy) for
    cross-play. Records the interleaved per-player rows. want_states banks the FULL
    physical state per arena-step (ball 9 + per-car 17: pos vel angVel forward up
    boost onGround) for reset-reconstruction studies; bank index = row // (2*ppt)."""
    torch.manual_seed(seed)
    per_team = isinstance(policies, (tuple, list))
    npl = 2 * ppt
    envs = [TeamArenaEnv(i, ppt, np.random.default_rng(seed + 10 + i)) for i in range(num_arenas)]

    rec = {
        "phys": np.empty((n_rows, 17), np.float32),
        "episode": np.empty(n_rows, np.int32),
        "slot": np.empty(n_rows, np.int8),      # player slot within the arena (team = slot%2)
        "touched": np.zeros(n_rows, bool),
        "on_ground": np.zeros(n_rows, bool),
        "kickoff": np.zeros(n_rows, bool),
    }
    if want_h2:
        rec["h2"] = np.empty((n_rows, 512), np.float16)
    if want_masks:
        rec["masks"] = np.empty((n_rows, 90), np.uint8)
    if want_goals:
        # Per-row ACHIEVED-GOAL vectors in the reachability heads' exact normalization
        # (Learner.cpp fnAppendAchieved parity), derived from the canonical obs the
        # policy consumed: ball head = canonical ball pos/vel; car head = car-local
        # ball pos/vel; carstate = canonical CAR pos/vel (the proposer-era third
        # stream - the candidate movement-frontier goal space). These are the agent's
        # own achieved-state trajectories.
        rec["ach_ball"] = np.empty((n_rows, 6), np.float32)
        rec["ach_car"] = np.empty((n_rows, 6), np.float32)
        rec["ach_carstate"] = np.empty((n_rows, 6), np.float32)
        rec["action"] = np.empty(n_rows, np.int16)
    if want_states:
        rec["state_bank"] = np.empty((n_rows // npl + 1, 9 + npl * 17), np.float32)
    goals = 0
    goals_by_team = [0, 0]
    ep_goal_team = {}  # episode id -> scoring team (0/1) for goal-ended episodes
    episodes_done = 0
    last_hit = [dict() for _ in envs]
    kick_first_touch = []
    kick_start = {}

    row = 0
    while row + npl * len(envs) <= n_rows:
        obs_list, mask_list, metas = [], [], []
        for ei, env in enumerate(envs):
            obs, masks, phys, states = env.observe()
            obs_list.append(obs)
            mask_list.append(masks)
            tick = env.arena.tick_count
            per_player = []
            for p, st in enumerate(states):
                bhi = st.ball_hit_info
                hit_tick = bhi.tick_count_when_hit if bhi.is_valid else -1
                prev = last_hit[ei].get((env.episode_id, p), -1)
                touched = bhi.is_valid and hit_tick > prev and hit_tick > tick - TICK_SKIP
                if touched:
                    last_hit[ei][(env.episode_id, p)] = hit_tick
                    if env.is_kickoff and env.episode_id in kick_start:
                        kick_first_touch.append((tick - kick_start[env.episode_id]) / 120.0)
                        del kick_start[env.episode_id]
                per_player.append((touched, st.is_on_ground))
            metas.append((phys, per_player, env.is_kickoff, env.episode_id))

        obs_b = torch.from_numpy(np.concatenate(obs_list))
        mask_b = torch.from_numpy(np.concatenate(mask_list))
        if per_team:
            slots = torch.arange(obs_b.shape[0]) % npl
            h2 = torch.empty(obs_b.shape[0], 512)
            actions = torch.empty(obs_b.shape[0], dtype=torch.long)
            for t, pol in enumerate(policies):
                sel = torch.nonzero(slots % 2 == t).flatten()
                h2_t, act_t = pol.act(obs_b[sel], mask_b[sel])
                h2[sel] = h2_t
                actions[sel] = act_t
        else:
            h2, actions = policies.act(obs_b, mask_b)

        for i, env in enumerate(envs):
            phys, per_player, is_kick, ep = metas[i]
            if want_states:
                ball = env.arena.ball.get_state()
                entry = list(ball.pos.as_tuple()) + list(ball.vel.as_tuple()) + list(ball.ang_vel.as_tuple())
                for car in env.cars:
                    st = car.get_state()
                    entry += list(st.pos.as_tuple()) + list(st.vel.as_tuple()) + list(st.ang_vel.as_tuple())
                    entry += list(st.rot_mat.forward.as_tuple()) + list(st.rot_mat.up.as_tuple())
                    entry += [st.boost, float(st.is_on_ground)]
                rec["state_bank"][row // npl] = entry
            for p in range(npl):
                rec["phys"][row] = phys[p]
                rec["episode"][row] = ep
                rec["slot"][row] = p
                rec["touched"][row] = per_player[p][0]
                rec["on_ground"][row] = per_player[p][1]
                rec["kickoff"][row] = is_kick
                if want_h2:
                    rec["h2"][row] = h2[npl * i + p].numpy()
                if want_masks:
                    rec["masks"][row] = mask_b[npl * i + p].numpy()
                if want_goals:
                    o = obs_b[npl * i + p].numpy()
                    # obs ball pos coef 1/5000, vel coef 1/2300 -> reach scales
                    rec["ach_ball"][row, 0] = o[0] * 5000.0 / 4096.0
                    rec["ach_ball"][row, 1] = o[1] * 5000.0 / 6000.0
                    rec["ach_ball"][row, 2] = o[2] * 5000.0 / 2044.0
                    rec["ach_ball"][row, 3:6] = o[3:6] * 2300.0 / 6000.0
                    # self block local ball pos (+18..20, coef 1/5000) and vel
                    # (+21..23, coef 1/2300) -> carLocalScale 2300
                    rec["ach_car"][row, 0:3] = o[51 + 18: 51 + 21] * 5000.0 / 2300.0
                    rec["ach_car"][row, 3:6] = o[51 + 21: 51 + 24]  # 2300/2300
                    # self block canonical car pos (+0..2) and vel (+9..11)
                    rec["ach_carstate"][row, 0] = o[51 + 0] * 5000.0 / 4096.0
                    rec["ach_carstate"][row, 1] = o[51 + 1] * 5000.0 / 6000.0
                    rec["ach_carstate"][row, 2] = o[51 + 2] * 5000.0 / 2044.0
                    rec["ach_carstate"][row, 3:6] = o[51 + 9: 51 + 12] * 2300.0 / 6000.0
                    rec["action"][row] = int(actions[npl * i + p])
                row += 1

        for i, env in enumerate(envs):
            if env.step(actions[npl * i: npl * (i + 1)].tolist()):
                goals += int(env.goal_scored)
                if env.goal_scored and env.goal_team is not None:
                    goals_by_team[env.goal_team] += 1
                    ep_goal_team[env.episode_id] = env.goal_team
                episodes_done += 1
                env.reset()
                if env.is_kickoff:
                    kick_start[env.episode_id] = env.arena.tick_count

    bank = rec.pop("state_bank", None)
    for k in rec:
        rec[k] = rec[k][:row]
    if bank is not None:
        rec["state_bank"] = bank[: row // npl]
    rec["goals"] = goals
    rec["goals_by_team"] = goals_by_team
    rec["ep_goal_team"] = ep_goal_team
    rec["episodes"] = max(episodes_done, 1)
    rec["kick_first_touch"] = kick_first_touch
    rec["ppt"] = ppt
    return rec


# ---------------------------------------------------------------------------
def team_possession_readings(rec, max_readings=6000):
    """v2 labels generalized to teams: first touch between reading and shortly past
    touchdown -> WON (self) / TEAMMATE / LOST (opponent) / NONE (collective decline)."""
    ppt = rec["ppt"]
    npl = 2 * ppt
    phys, episode, slot, touched = rec["phys"], rec["episode"], rec["slot"], rec["touched"]

    ep_rows = {int(e): np.flatnonzero(episode == e) for e in np.unique(episode)}
    row_pos = np.empty(len(slot), np.int64)
    for rows in ep_rows.values():
        row_pos[rows] = np.arange(len(rows))

    cand = np.flatnonzero(phys[:, 2] > ARM_Z)
    stride = max(1, len(cand) // max_readings)
    cand = cand[::stride]

    arena = rs.Arena(rs.GameMode.SOCCAR)
    margin_rows = int(round(RACE_MARGIN_S / DT)) * npl

    out = {"row": [], "outcome": [], "d_now": [], "t_land": []}
    for r in cand:
        res = simulate_landing(arena, phys[r, 0:3], phys[r, 3:6], phys[r, 6:9])
        if res is None:
            continue
        lx, ly, t_land = res
        rows = ep_rows[int(episode[r])]
        q = row_pos[r]
        q_land = q + npl * int(round(t_land / DT))
        if q_land >= len(rows):
            continue
        d_now = float(np.linalg.norm(phys[r, 9:11] - np.array([lx, ly])))
        if d_now / max(t_land, 1e-6) >= FEASIBLE_SPEED:
            continue

        my_slot, my_team = slot[r], slot[r] % 2
        outcome = NONE
        for qq in range(q + 1, min(len(rows), q_land + margin_rows + 1)):
            if touched[rows[qq]]:
                s = slot[rows[qq]]
                outcome = WON if s == my_slot else (TEAMMATE if s % 2 == my_team else LOST)
                break
        out["row"].append(int(r))
        out["outcome"].append(outcome)
        out["d_now"].append(d_now)
        out["t_land"].append(float(t_land))
    return {k: np.array(v) for k, v in out.items()}


def derive_team_direction(h2, readings, rng):
    """Matched WON-vs-NONE difference of trunk means; TEAMMATE and LOST rows are
    excluded (resolved races, not declines). Returns (v, sigma, n_pairs)."""
    keep = np.isin(readings["outcome"], (WON, NONE))
    idx = np.flatnonzero(keep)
    won = readings["outcome"][idx] == WON
    d, t = readings["d_now"][idx], readings["t_land"][idx]
    d_edges = np.quantile(d, np.linspace(0, 1, MATCH_BINS_D + 1))[1:-1]
    t_edges = np.quantile(t, np.linspace(0, 1, MATCH_BINS_T + 1))[1:-1]
    bins = np.digitize(d, d_edges) * 10 + np.digitize(t, t_edges)
    sel_w, sel_n = [], []
    for b in np.unique(bins):
        w = idx[np.flatnonzero((bins == b) & won)]
        nn = idx[np.flatnonzero((bins == b) & ~won)]
        m = min(len(w), len(nn))
        if m == 0:
            continue
        sel_w += list(rng.choice(w, m, replace=False))
        sel_n += list(rng.choice(nn, m, replace=False))
    if not sel_w:
        raise RuntimeError("no matched WON/NONE pairs")
    Hw = h2[readings["row"][np.array(sel_w)]].astype(np.float32)
    Hn = h2[readings["row"][np.array(sel_n)]].astype(np.float32)
    v = Hw.mean(0) - Hn.mean(0)
    v /= max(np.linalg.norm(v), 1e-8)
    with np.errstate(all="ignore"):  # spurious Accelerate FP flags
        proj = h2.astype(np.float32) @ v
    assert np.isfinite(proj).all()
    return v, float(np.std(proj)), len(sel_w)


def team_metrics(rec, boot=200, rng=None):
    """Per-mode panel: possession outcome rates on feasible readings (+ episode-cluster
    bootstrap SE on team-won), behavior + competence canaries."""
    rd = team_possession_readings(rec)
    n = len(rd["row"])
    aerial = rec["touched"] & ~rec["on_ground"] & (rec["phys"][:, 2] > 400)
    rates = {k: float((rd["outcome"] == v).mean()) if n else float("nan")
             for k, v in (("self_won", WON), ("teammate_won", TEAMMATE),
                          ("lost", LOST), ("none", NONE))}
    team_won = rates["self_won"] + rates["teammate_won"]

    se = float("nan")
    if n:
        rng = rng or np.random.default_rng(0)
        eps = rec["episode"][rd["row"]]
        uniq = np.unique(eps)
        by_ep = {e: rd["outcome"][eps == e] for e in uniq}
        stats = []
        for _ in range(boot):
            sample = rng.choice(uniq, len(uniq), replace=True)
            outs = np.concatenate([by_ep[e] for e in sample])
            stats.append(np.isin(outs, (WON, TEAMMATE)).mean())
        se = float(np.std(stats))

    return {
        **rates,
        "team_won": team_won,
        "team_won_se": se,
        "n_feasible_readings": int(n),
        "n_reading_episodes": int(len(np.unique(rec["episode"][rd["row"]]))) if n else 0,
        "aerial_touch_ratio": float(aerial.mean()),
        "touch_ratio": float(rec["touched"].mean()),
        "in_air_ratio": float((~rec["on_ground"]).mean()),
        "goals_per_episode": rec["goals"] / rec["episodes"],
        "kickoff_first_touch_s": float(np.median(rec["kick_first_touch"]))
        if rec["kick_first_touch"] else float("nan"),
    }
