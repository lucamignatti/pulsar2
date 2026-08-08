"""Aerial2D: a minimal vectorized environment that reproduces the acquisition wall.

A point-mass "car" on a 1-D ground line with a 2-D (x, z) flight phase. Ground
play (drive under the ball, jump-touch low balls) is easy and shaped toward;
touching HIGH balls requires the conduct: jump, then a sustained run of boost
actions with x-drift alignment, against a boost cost and a crash penalty. The
pieces (jumping, boosting, low touches) all occur under a young policy; the
whole chain essentially never does, and failed attempts are net-negative, so
vanilla PPO learns avoidance -- the wall.

Reward mirrors the paper's testbed philosophy: +1 any touch, small approach
shaping, costs; NO aerial-specific term anywhere. Metrics (touch/air/hi) are
logging-only.

All state is numpy arrays of shape (N,); step() is fully vectorized.
"""
import numpy as np

DT = 0.1
GRAV = 6.0
JUMP_VZ = 5.0
BOOST_AZ = 8.0          # net climb accel = BOOST_AZ - GRAV = +2
BOOST_DRAIN = 0.15      # per boost step; 1.0 tank = ~7 steps of boost
BOOST_REGEN = 0.03      # per ground step
DRIVE_AX = 8.0
DRIFT_AX = 3.0
VX_MAX = 6.0
X_LIM = 10.0
Z_CEIL = 10.0
TOUCH_R = 0.9
EP_LEN = 200

BALL_Z_LO, BALL_Z_HI = 0.5, 7.5
PAD_X, PAD_R = -7.0, 1.0   # boost pad (enable_pad mode only)
HI_Z = 5.0              # ball spawned at/above this counts as a "hi" conduct
AIR_Z = 1.5             # car above this at touch = airborne touch

R_TOUCH = 1.0
R_APPROACH = 0.05       # * (prev_dist - dist)
R_BOOST_COST = -0.015   # per boost step
R_TIME = -0.002         # per step
R_CRASH = -0.4          # landing with |vz| > 6
CRASH_VZ = 6.0

N_ACT = 7  # 0 noop, 1 left, 2 right, 3 jump, 4 boost, 5 boost+left, 6 boost+right
OBS_DIM = 12  # includes ball vz (always 0 in phase 1; live in phase 2)


class Aerial2D:
    """phase 1: static balls (the original conduct).
    phase 2 (meta-transfer frontier shift): balls spawn HIGH ONLY (z in [5,8])
    and FALL under gravity, respawning on ground contact -- a new conduct
    (timed intercept) whose obs support (ball vz) existed from step 0."""
    def __init__(self, n_envs, seed=0):
        self.n = n_envs
        self.phase = 1
        self.rng = np.random.default_rng(seed)
        self.x = np.zeros(n_envs); self.z = np.zeros(n_envs)
        self.vx = np.zeros(n_envs); self.vz = np.zeros(n_envs)
        self.boost = np.ones(n_envs)
        self.on_ground = np.ones(n_envs, bool)
        self.bx = np.zeros(n_envs); self.bz = np.zeros(n_envs)
        self.bvz = np.zeros(n_envs)
        self.t = np.zeros(n_envs, np.int64)
        self.prev_dist = np.zeros(n_envs)
        self._spawn_ball(np.ones(n_envs, bool))
        self.reset_all()

    def set_phase(self, p):
        self.phase = p
        self._spawn_ball(np.ones(self.n, bool))

    def enable_wind(self):
        """Adversarial chart-confound: hidden lateral acceleration in x > 5.
        Region-dependent dynamics whose region feature the chart may have
        pruned -- tests whether the invariance claim fails safe."""
        self.wind_on = True

    def enable_pad(self):
        """Boost-pad mode: passive regen OFF; the only refill is driving over a
        fixed ground pad. Makes value DISCONTINUOUS in boost x position (low
        boost near a high ball is bad; far from ball near the pad is good) --
        the long-chain, stitching-required regime."""
        self.pad_on = True

    def _spawn_ball(self, m):
        k = int(m.sum())
        if k == 0:
            return
        self.bx[m] = self.rng.uniform(-8, 8, k)
        if self.phase == 1:
            self.bz[m] = self.rng.uniform(BALL_Z_LO, BALL_Z_HI, k)
            self.bvz[m] = 0.0
        else:
            self.bz[m] = self.rng.uniform(5.0, 8.0, k)
            self.bvz[m] = 0.0  # released at rest, falls under gravity

    def reset_all(self):
        m = np.ones(self.n, bool)
        self._reset(m)

    def _reset(self, m):
        k = int(m.sum())
        self.x[m] = self.rng.uniform(-8, 8, k)
        self.z[m] = 0.0
        self.vx[m] = 0.0; self.vz[m] = 0.0
        self.boost[m] = 1.0
        self.on_ground[m] = True
        self.t[m] = 0
        self._spawn_ball(m)
        self.prev_dist[m] = np.hypot(self.bx[m] - self.x[m], self.bz[m] - self.z[m])

    def obs(self):
        dx = self.bx - self.x
        dz = self.bz - self.z
        dist = np.hypot(dx, dz)
        return np.stack([
            self.x / X_LIM, self.z / Z_CEIL,
            self.vx / VX_MAX, self.vz / 12.0,
            self.boost, self.on_ground.astype(np.float64),
            dx / 20.0, dz / Z_CEIL, dist / 22.0,
            self.bz / Z_CEIL, self.bvz / 10.0,
            np.minimum(self.t, EP_LEN) / EP_LEN,
        ], axis=1).astype(np.float32)

    def action_mask(self):
        """(N, N_ACT) float32; 1 = valid."""
        m = np.ones((self.n, N_ACT), np.float32)
        can_boost = (~self.on_ground) & (self.boost > 0)
        m[:, 3] = self.on_ground          # jump
        m[:, 4] = can_boost
        m[:, 5] = can_boost
        m[:, 6] = can_boost
        return m

    def step(self, a):
        n = self.n
        r = np.full(n, R_TIME)
        r_cost = np.full(n, R_TIME)   # channel: time + boost + crash costs
        r_touch = np.zeros(n)         # channel: touch payoff
        r_appr = np.zeros(n)          # channel: approach shaping
        info = {}

        left = (a == 1) | (a == 5)
        right = (a == 2) | (a == 6)
        jump = (a == 3) & self.on_ground
        # boost only converts to climb while upward momentum is kept: the conduct
        # must be chained from the jump, there is no mid-fall recovery
        boosting = ((a == 4) | (a == 5) | (a == 6)) & (~self.on_ground) \
                   & (self.boost > 0) & (self.vz > -1.0)

        # x control
        ax = np.where(self.on_ground, DRIVE_AX, DRIFT_AX)
        self.vx += (right.astype(np.float64) - left.astype(np.float64)) * ax * DT
        idle_ground = self.on_ground & ~(left | right)
        self.vx[idle_ground] *= 0.95
        np.clip(self.vx, -VX_MAX, VX_MAX, out=self.vx)

        # z control
        self.vz[jump] = JUMP_VZ
        self.on_ground[jump] = False
        air = ~self.on_ground
        az = np.where(boosting, BOOST_AZ, 0.0) - GRAV
        self.vz[air] += az[air] * DT
        self.boost[boosting] -= BOOST_DRAIN
        np.clip(self.boost, 0.0, 1.0, out=self.boost)
        r[boosting] += R_BOOST_COST
        r_cost[boosting] += R_BOOST_COST

        # integrate
        if getattr(self, "wind_on", False):
            self.vx += np.where(self.x > 5.0, 3.0 * DT, 0.0)
            np.clip(self.vx, -VX_MAX, VX_MAX, out=self.vx)
        self.x += self.vx * DT
        self.z[air] += self.vz[air] * DT
        hit_wall = np.abs(self.x) > X_LIM
        self.x = np.clip(self.x, -X_LIM, X_LIM)
        self.vx[hit_wall] = 0.0
        hit_ceil = self.z > Z_CEIL
        self.z[hit_ceil] = Z_CEIL
        self.vz[hit_ceil] = np.minimum(self.vz[hit_ceil], 0.0)

        # landing
        landed = air & (self.z <= 0.0)
        crash = landed & (self.vz < -CRASH_VZ)
        r[crash] += R_CRASH
        r_cost[crash] += R_CRASH
        self.z[landed] = 0.0
        self.vz[landed] = 0.0
        self.on_ground[landed] = True
        if getattr(self, "pad_on", False):
            hit_pad = self.on_ground & (np.abs(self.x - PAD_X) < PAD_R)
            self.boost[hit_pad] = 1.0
        else:
            regen = self.on_ground
            self.boost[regen] = np.minimum(1.0, self.boost[regen] + BOOST_REGEN)

        # ball dynamics (phase 2: falling balls, respawn on ground contact)
        if self.phase == 2:
            self.bvz -= GRAV * DT
            self.bz += self.bvz * DT
            grounded_ball = self.bz <= 0.5
            self._spawn_ball(grounded_ball)

        # touch
        dx = self.bx - self.x; dz = self.bz - self.z
        dist = np.hypot(dx, dz)
        touch = dist <= TOUCH_R
        r[touch] += R_TOUCH
        r_touch[touch] += R_TOUCH
        air_touch = touch & (self.z > AIR_Z)
        hi_touch = touch & (self.bz >= HI_Z)
        info["touch"] = touch.copy()
        info["air_touch"] = air_touch.copy()
        info["hi_touch"] = hi_touch.copy()
        info["hi_offered"] = (self.bz >= HI_Z)  # per-step exposure, for conversion rates
        self._spawn_ball(touch)

        # approach shaping (recompute dist to the possibly-respawned ball for next step's prev)
        appr = np.clip(R_APPROACH * (self.prev_dist - dist), -0.1, 0.1)
        r += appr
        r_appr += appr
        info["r_parts"] = np.stack([r_touch, r_appr, r_cost], axis=1)
        dx = self.bx - self.x; dz = self.bz - self.z
        self.prev_dist = np.hypot(dx, dz)

        # time
        self.t += 1
        done = self.t >= EP_LEN  # truncation only
        info["done"] = done.copy()
        if done.any():
            info["final_obs"] = self.obs()  # pre-reset obs, for truncation bootstrap
            self._reset(done)
        return r.astype(np.float32), done, info
