"""Python port of the repo's C++ AdvancedObs + DefaultAction (parity-critical).

Sources:
  RLGymCPP/src/RLGymCPP/ObsBuilders/AdvancedObs.{h,cpp}   (obs layout + coefficients)
  RLGymCPP/src/RLGymCPP/ActionParsers/DefaultAction.cpp   (90-way table + masks)
  RLGymCPP/src/RLGymCPP/Gamestates/GameState.cpp          (boost pad canonical ordering)
  RLGymCPP/src/RLGymCPP/Gamestates/StateUtil.cpp          (InvertPhys)
  RLGymCPP/src/RLGymCPP/CommonValues.h                    (BOOST_LOCATIONS)

Obs layout for 1v1 (109 floats):
  [0:9]    ball pos*POS_COEF, vel*VEL_COEF, angVel*ANG_VEL_COEF   (team-inverted)
  [9:17]   prev action (throttle, steer, pitch, yaw, roll, jump, boost, handbrake)
  [17:51]  34 boost pads, active ? 1 : 1/(1+cooldown), canonical order (team-reversed)
  [51:80]  self:     pos*P, forward, up, vel*V, angVel*A, local angVel*A,
                     local(ball pos - pos)*P, local(ball vel - vel)*V,
                     boost/100, isOnGround, hasFlipOrJump, isDemoed, hasJumped
  [80:109] opponent: same 29 floats
All world-frame quantities are inverted (x,y negated) for the ORANGE player so both
teams see themselves attacking +y.
"""

import numpy as np

POS_COEF = 1 / 5000.0
VEL_COEF = 1 / 2300.0
ANG_VEL_COEF = 1 / 3.0

OBS_SIZE = 109
NUM_ACTIONS = 90

# CommonValues::BOOST_LOCATIONS - canonical pad order. The list is antisymmetric:
# entry (33 - i) is entry i with x,y negated, so the inverted view is just a reversal.
BOOST_LOCATIONS = np.array([
    [0.0, -4240.0, 70.0], [-1792.0, -4184.0, 70.0], [1792.0, -4184.0, 70.0],
    [-3072.0, -4096.0, 73.0], [3072.0, -4096.0, 73.0], [-940.0, -3308.0, 70.0],
    [940.0, -3308.0, 70.0], [0.0, -2816.0, 70.0], [-3584.0, -2484.0, 70.0],
    [3584.0, -2484.0, 70.0], [-1788.0, -2300.0, 70.0], [1788.0, -2300.0, 70.0],
    [-2048.0, -1036.0, 70.0], [0.0, -1024.0, 70.0], [2048.0, -1036.0, 70.0],
    [-3584.0, 0.0, 73.0], [-1024.0, 0.0, 70.0], [1024.0, 0.0, 70.0],
    [3584.0, 0.0, 73.0], [-2048.0, 1036.0, 70.0], [0.0, 1024.0, 70.0],
    [2048.0, 1036.0, 70.0], [-1788.0, 2300.0, 70.0], [1788.0, 2300.0, 70.0],
    [-3584.0, 2484.0, 70.0], [3584.0, 2484.0, 70.0], [0.0, 2816.0, 70.0],
    [-940.0, 3310.0, 70.0], [940.0, 3308.0, 70.0], [-3072.0, 4096.0, 73.0],
    [3072.0, 4096.0, 73.0], [-1792.0, 4184.0, 70.0], [1792.0, 4184.0, 70.0],
    [0.0, 4240.0, 70.0],
])


def build_pad_index_map(arena):
    """canonical index -> arena pad index, by 2-D position match (GameState.cpp parity)."""
    pads = arena.get_boost_pads()
    pad_pos = np.array([[*p.get_pos().as_tuple()] for p in pads])
    mapping = np.full(len(BOOST_LOCATIONS), -1, dtype=int)
    for i, loc in enumerate(BOOST_LOCATIONS):
        d2 = ((pad_pos[:, :2] - loc[:2]) ** 2).sum(1)
        j = int(d2.argmin())
        if d2[j] >= 10:
            raise RuntimeError(f"no arena pad matches canonical location {loc}")
        mapping[i] = j
    assert len(set(mapping)) == len(BOOST_LOCATIONS)
    return mapping


# ---------------------------------------------------------------------------
# DefaultAction: the 90-way discrete table, generated EXACTLY like the C++ ctor
# (order matters - the policy head indexes this table).
# Element order: throttle, steer, pitch, yaw, roll, jump, boost, handbrake.
# ---------------------------------------------------------------------------
def _build_action_table():
    actions = []
    R_B = (0.0, 1.0)
    R_F = (-1.0, 0.0, 1.0)

    for throttle in R_F:                       # ground
        for steer in R_F:
            for boost in R_B:
                for handbrake in R_B:
                    if boost == 1 and throttle != 1:
                        continue
                    actions.append([throttle, steer, 0, steer, 0, 0, boost, handbrake])
    num_ground = len(actions)

    for pitch in R_F:                          # aerial
        for yaw in R_F:
            for roll in R_F:
                for jump in R_B:
                    for boost in R_B:
                        if jump == 1 and yaw != 0:
                            continue
                        if pitch == roll == jump == 0:
                            continue
                        handbrake = float(jump == 1 and (pitch != 0 or yaw != 0 or roll != 0))
                        actions.append([boost, yaw, pitch, yaw, roll, jump, boost, handbrake])

    table = np.array(actions, dtype=np.float32)
    assert table.shape == (NUM_ACTIONS, 8), table.shape

    jump_mask = table[:, 5] == 1
    boost_mask = table[:, 6] == 1
    ground_mask = np.zeros(NUM_ACTIONS, bool)
    ground_mask[:num_ground] = True
    air_mask = ~ground_mask & ~jump_mask
    # yaw-only ground actions double as air actions (skipped in air gen as duplicates)
    for i in range(num_ground):
        t = table[i]
        if t[0] == t[6] and (t[3] != 0) == bool(t[7]):
            air_mask[i] = True
    return table, ground_mask, air_mask, jump_mask, boost_mask


ACTION_TABLE, GROUND_MASK, AIR_MASK, JUMP_MASK, BOOST_MASK = _build_action_table()


def get_action_mask(car_state) -> np.ndarray:
    """DefaultAction::GetActionMask parity. Takes a RocketSim CarState."""
    mask = GROUND_MASK.copy() if car_state.is_on_ground else AIR_MASK.copy()

    if car_state.boost == 0:
        mask &= ~BOOST_MASK

    turtled = car_state.has_world_contact and car_state.world_contact_normal.z > 0.9
    if car_state.has_flip_or_jump() or turtled:
        mask |= JUMP_MASK
    return mask


# ---------------------------------------------------------------------------
# AdvancedObs
# ---------------------------------------------------------------------------
_INV = np.array([-1.0, -1.0, 1.0])


class PhysSnap:
    """World-frame physics of one body, optionally team-inverted (InvertPhys parity)."""

    __slots__ = ("pos", "vel", "ang_vel", "forward", "right", "up")

    def __init__(self, state, inv: bool):
        self.pos = np.array(state.pos.as_tuple())
        self.vel = np.array(state.vel.as_tuple())
        self.ang_vel = np.array(state.ang_vel.as_tuple())
        rm = state.rot_mat
        self.forward = np.array(rm.forward.as_tuple())
        self.right = np.array(rm.right.as_tuple())
        self.up = np.array(rm.up.as_tuple())
        if inv:
            self.pos = self.pos * _INV
            self.vel = self.vel * _INV
            self.ang_vel = self.ang_vel * _INV
            self.forward = self.forward * _INV
            self.right = self.right * _INV
            self.up = self.up * _INV

    def local(self, v: np.ndarray) -> np.ndarray:
        """RotMat::Dot(vec): world -> local (components along forward, right, up)."""
        return np.array([v @ self.forward, v @ self.right, v @ self.up])


def _player_obs(car_state, phys: PhysSnap, ball: PhysSnap) -> list:
    out = []
    out += (phys.pos * POS_COEF).tolist()
    out += phys.forward.tolist()
    out += phys.up.tolist()
    out += (phys.vel * VEL_COEF).tolist()
    out += (phys.ang_vel * ANG_VEL_COEF).tolist()
    out += (phys.local(phys.ang_vel) * ANG_VEL_COEF).tolist()
    out += (phys.local(ball.pos - phys.pos) * POS_COEF).tolist()
    out += (phys.local(ball.vel - phys.vel) * VEL_COEF).tolist()
    out.append(car_state.boost / 100.0)
    out.append(float(car_state.is_on_ground))
    out.append(float(car_state.has_flip_or_jump()))
    out.append(float(car_state.is_demoed))
    out.append(float(car_state.has_jumped))
    return out


def build_obs(car_state, opp_state, ball_state, prev_action: np.ndarray,
              pad_active: np.ndarray, pad_cooldown: np.ndarray, is_orange: bool) -> np.ndarray:
    """AdvancedObs::BuildObs for 1v1.

    pad_active/pad_cooldown are in CANONICAL (BOOST_LOCATIONS) order, un-inverted;
    the orange-team reversal happens here.
    """
    inv = is_orange
    ball = PhysSnap(ball_state, inv)

    obs = []
    obs += (ball.pos * POS_COEF).tolist()
    obs += (ball.vel * VEL_COEF).tolist()
    obs += (ball.ang_vel * ANG_VEL_COEF).tolist()
    obs += prev_action.tolist()

    active = pad_active[::-1] if inv else pad_active
    cooldown = pad_cooldown[::-1] if inv else pad_cooldown
    obs += np.where(active, 1.0, 1.0 / (1.0 + cooldown)).tolist()

    obs += _player_obs(car_state, PhysSnap(car_state, inv), ball)
    obs += _player_obs(opp_state, PhysSnap(opp_state, inv), ball)

    o = np.asarray(obs, dtype=np.float32)
    assert o.shape == (OBS_SIZE,), o.shape
    return o


if __name__ == "__main__":
    print(f"action table: {ACTION_TABLE.shape}, ground {GROUND_MASK.sum()}, air {AIR_MASK.sum()}, "
          f"jump {JUMP_MASK.sum()}, boost {BOOST_MASK.sum()}")
    assert GROUND_MASK.sum() == 24
