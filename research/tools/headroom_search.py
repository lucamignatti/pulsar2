"""Headroom realizability via tree/prefix search — does H point at states where a better
action sequence actually exists?

PRE-REGISTRATION (2026-09-03, written before the run)
-----------------------------------------------------
Question. The composition critic publishes H = relu(min(Vdag1, Vdag2) - V_real). The
proposed lever is "search from high-H states and imitate what the search finds". That is
only worth building if search from high-H states finds MORE realizable improvement than
search from ordinary states. Same question for the cheaper retrospective trigger, the
value dip (a large negative k-step realized advantage).

Instrument. For a banked state s (full physics snapshot, both cars, pads, prevAction):
  baseline  = mean over K fresh closed-loop policy rollouts from s of
              score = sum_t gamma^t r_t + gamma^T V_real(s_T)      (critic units)
  search    = evolutionary prefix search: a self-player action prefix (macro-actions
              held m steps, length L macros, from uniform-valid / policy-at-temperature /
              mutation generators) followed by the policy closing the loop; opponent
              always plays the policy. Best prefix by mean score over its evaluations,
              then RE-EVALUATED on K fresh rollouts (held-out; kills the winner's curse).
  gain      = mean(search re-eval) - baseline                       (critic units)
Only goals are rewarded (this lineage is GCO: +-150 terminal, nothing else), so the
score is real goal outcomes inside the horizon plus the critic's own leaf value.

Strata (each n states): H_top (H >= q95), H_mid (q45..q55), H_zero (H == 0 or bottom
20%), dip (15-step realized advantage <= q03; searched from the state BEFORE the drop),
uniform (control = the base rate at which search finds anything anywhere).

Success criteria, fixed in advance (1 critic unit ~ 1/7.5 goal at ret_std ~ 20):
  P1  H is a usable trigger  iff  mean gain(H_top) - mean gain(uniform) >= 0.5
                              AND Spearman(H, gain) over uniform >= 0.2
  P2  dip is a usable trigger iff mean gain(dip) - mean gain(uniform) >= 0.5
  Power check: if mean gain(uniform) ~ 0 AND P(goal | best) == P(goal | baseline)
      everywhere, the search is underpowered and NOTHING here is evidence about H.

Sim: pip RocketSim (v2 physics) at 5.0 dynamics (tickSkip 8 / actionDelay 0); the
checkpoint trained on the vendored v3 engine. Search and baseline run in the SAME sim from
the SAME snapshot, so the comparison is internally consistent; the absolute numbers carry
the v2/v3 gap (measured ~12% relative goal share on a frozen policy, GCO_RESUME.md §3).

Usage (from build/, needs ./collision_meshes):
  python ../research/tools/headroom_search.py --ckpt <dir> --out <json> --workers 6
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

GAMMA = 0.9969          # gaeGamma at tickSkip 8 (ExampleMain / GCO_RESUME.md)
GOAL_R = 150.0
TICK_SKIP = 8
NUM_ACTIONS = 90
RESET_MIX = [(0.35, "near"), (0.55, "air"), (0.70, "kickoff"), (1.01, "random")]


# ---------------------------------------------------------------------------
# sim wrapper: observe / step / snapshot / restore
# ---------------------------------------------------------------------------
class SimArena:
    def __init__(self, rng):
        import RocketSim as rs
        from advanced_obs import build_pad_index_map
        self.rs = rs
        self.rng = rng
        self.arena = rs.Arena(rs.GameMode.SOCCAR)
        self.cars = [self.arena.add_car(rs.Team.BLUE), self.arena.add_car(rs.Team.ORANGE)]
        self.pad_map = build_pad_index_map(self.arena)
        self.pads = self.arena.get_boost_pads()
        self.goal_team = None
        self.arena.set_goal_score_callback(self._on_goal, None)
        self.prev_actions = np.zeros((2, 8), np.float32)
        self.last_touch_tick = 0
        self.steps = 0

    def _on_goal(self, **kw):
        rs = self.rs
        t = kw.get("team")
        self.goal_team = 0 if t == rs.Team.BLUE else 1 if t == rs.Team.ORANGE else None

    # -- trainer-parity reset mix (steer_team's team-aware setters, ppt=1)
    def reset(self):
        import steer_team as stm
        u = self.rng.random()
        kind = next(k for w, k in RESET_MIX if u < w)
        if kind == "near":
            stm.set_team_ball_near_car(self.arena, self.rng, 1)
        elif kind == "air":
            stm.set_team_air_drill(self.arena, self.rng, 1)
        elif kind == "kickoff":
            self.arena.reset_kickoff(seed=int(self.rng.integers(0, 2**30)))
        else:
            stm.set_team_random(self.arena, self.rng)
        zero = self.rs.CarControls()
        for c in self.cars:
            c.set_controls(zero)
        self.prev_actions[:] = 0
        self.goal_team = None
        self.steps = 0
        self.last_touch_tick = self.arena.tick_count
        self.kind = kind

    def snapshot(self):
        pads = [(p.get_state().is_active, p.get_state().cooldown) for p in self.pads]
        return {
            "ball": self.arena.ball.get_state(),
            "cars": [c.get_state() for c in self.cars],
            "pads": pads,
            "prev": self.prev_actions.copy(),
        }

    def restore(self, snap):
        rs = self.rs
        self.arena.ball.set_state(snap["ball"])
        for c, s in zip(self.cars, snap["cars"]):
            c.set_state(s)
        for p, (act, cd) in zip(self.pads, snap["pads"]):
            ps = rs.BoostPadState()
            ps.is_active, ps.cooldown = bool(act), float(cd)
            p.set_state(ps)
        self.prev_actions[:] = snap["prev"]
        self.goal_team = None
        self.steps = 0
        self.last_touch_tick = self.arena.tick_count

    def observe(self):
        from advanced_obs import build_obs_padded, get_action_mask
        ball = self.arena.ball.get_state()
        st = [c.get_state() for c in self.cars]
        active = np.empty(len(self.pad_map), bool)
        cooldown = np.empty(len(self.pad_map), np.float32)
        for i, j in enumerate(self.pad_map):
            s = self.pads[j].get_state()
            active[i], cooldown[i] = s.is_active, s.cooldown
        obs = np.stack([
            build_obs_padded(st[p], [], [st[1 - p]], ball, self.prev_actions[p],
                             active, cooldown, p == 1, self.rng) for p in range(2)])
        masks = np.stack([get_action_mask(st[p]) for p in range(2)])
        for s in st:
            bhi = s.ball_hit_info
            if bhi.is_valid:
                self.last_touch_tick = max(self.last_touch_tick, bhi.tick_count_when_hit)
        return obs, masks, st

    def step(self, actions):
        from advanced_obs import ACTION_TABLE
        rs = self.rs
        for p, a in enumerate(actions):
            e = ACTION_TABLE[a]
            c = rs.CarControls()
            c.throttle, c.steer = float(e[0]), float(e[1])
            c.pitch, c.yaw, c.roll = float(e[2]), float(e[3]), float(e[4])
            c.jump, c.boost, c.handbrake = bool(e[5]), bool(e[6]), bool(e[7])
            self.cars[p].set_controls(c)
            self.prev_actions[p] = e
        self.arena.step(TICK_SKIP)
        self.steps += 1


# ---------------------------------------------------------------------------
# policy + value reads on GPU
# ---------------------------------------------------------------------------
class SearchPolicy:
    def __init__(self, ckpt, device):
        import torch
        import load_checkpoint_70 as l70
        names = ["SHARED_HEAD", "POLICY", "CRITIC_TRUNK", "CRITIC", "CRITIC2",
                 "VDAG1", "VDAG2", "GAP_EXP"]
        models = l70.load_models(Path(ckpt), names=names)
        for m in models.values():
            m.to(device).eval()
        self.m = models
        self.device = device
        self.torch = torch
        # opp_embed self-play ctx [isSelf, isOld, isExt, oppAgeFrac] = [1,0,0,0]
        oe = torch.jit.load(str(Path(ckpt) / "OPP_EMBED.lt"), map_location="cpu")
        p = dict(oe.named_parameters())
        with torch.no_grad():
            x = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
            h = torch.nn.functional.linear(x, p["0.weight"], p["0.bias"])
            h = torch.nn.functional.layer_norm(h, (h.shape[-1],), p["1.weight"], p["1.bias"])
            self.opp_vec = torch.nn.functional.linear(torch.relu(h), p["3.weight"], p["3.bias"]).to(device)
        stat = json.load(open(Path(ckpt) / "RUNNING_STATS.json"))["return_stat"]
        self.ret_std = float(np.sqrt(stat["var"] / max(stat["count"], 1)))
        self.goal_u = GOAL_R / self.ret_std

    def forward(self, obs_np, mask_np):
        """obs (n,230) mask (n,90) -> logits (masked), v_real, vdag_min, H  (all torch on device)"""
        torch = self.torch
        with torch.no_grad():
            obs = torch.from_numpy(obs_np).to(self.device)
            mask = torch.from_numpy(mask_np).to(self.device).bool()
            h2 = self.m["SHARED_HEAD"](obs)
            logits = self.m["POLICY"](h2)
            logits = logits.masked_fill(~mask, -1e10)
            vt = self.m["CRITIC_TRUNK"](h2)
            vtc = vt + self.opp_vec
            v = 0.5 * (self.m["CRITIC"](vtc).flatten() + self.m["CRITIC2"](vtc).flatten())
            vd = torch.minimum(self.m["VDAG1"](vt).flatten(), self.m["VDAG2"](vt).flatten())
            vexp = self.m["GAP_EXP"](h2).flatten()
            return logits, v, vd, torch.relu(vd - v), vexp

    def sample(self, logits, temp):
        torch = self.torch
        if np.isscalar(temp):
            probs = torch.softmax(logits / temp, -1)
        else:
            probs = torch.softmax(logits / temp.to(logits.device).unsqueeze(-1), -1)
        return torch.multinomial(probs, 1).flatten()


# ---------------------------------------------------------------------------
# rollout phase: bank states + per-row values + realized returns
# ---------------------------------------------------------------------------
def rollout(pol, n_arenas, n_steps, seed, no_touch_s=20.0, cap_s=30.0):
    rng = np.random.default_rng(seed)
    envs = [SimArena(np.random.default_rng(seed * 1000 + i)) for i in range(n_arenas)]
    for e in envs:
        e.reset()
    ep_id = np.arange(n_arenas)
    next_ep = n_arenas
    snaps, rows = [], []   # snaps[i] = (env idx, snapshot); rows: per (arena-step, slot)
    ep_goal = {}           # episode -> scoring team
    ep_kind = {i: envs[i].kind for i in range(n_arenas)}
    for t in range(n_steps):
        obs, masks = [], []
        for e in envs:
            o, m, _ = e.observe()
            obs.append(o); masks.append(m)
        obs = np.concatenate(obs); masks = np.concatenate(masks)
        logits, v, vd, H, vexp = pol.forward(obs, masks)
        acts = pol.sample(logits, 1.0).cpu().numpy()
        v, vd, H, vexp = v.cpu().numpy(), vd.cpu().numpy(), H.cpu().numpy(), vexp.cpu().numpy()
        for i, e in enumerate(envs):
            sid = len(snaps)
            snaps.append(e.snapshot())
            for p in range(2):
                r = 2 * i + p
                rows.append((sid, p, int(ep_id[i]), float(v[r]), float(vd[r]), float(H[r]),
                             float(vexp[r]), int(acts[r]), e.steps))
        for i, e in enumerate(envs):
            e.step(acts[2 * i: 2 * i + 2])
            done = e.goal_team is not None
            no_touch = (e.arena.tick_count - e.last_touch_tick) > no_touch_s * 120
            capped = e.steps >= cap_s * 120 / TICK_SKIP
            if done:
                ep_goal[int(ep_id[i])] = e.goal_team
            if done or no_touch or capped:
                e.reset()
                ep_id[i] = next_ep; ep_kind[next_ep] = e.kind; next_ep += 1
    R = np.array(rows, dtype=object)
    out = {
        "sid": np.array([r[0] for r in rows]), "slot": np.array([r[1] for r in rows]),
        "ep": np.array([r[2] for r in rows]), "v": np.array([r[3] for r in rows], np.float32),
        "vd": np.array([r[4] for r in rows], np.float32), "H": np.array([r[5] for r in rows], np.float32),
        "vexp": np.array([r[6] for r in rows], np.float32),
        "act": np.array([r[7] for r in rows]), "step": np.array([r[8] for r in rows]),
    }
    out["ep_goal"] = ep_goal
    out["ep_kind"] = ep_kind
    # realized discounted return per row (goal +-, truncation bootstraps V of last row of the
    # (ep, slot) segment) and the 15-step realized advantage
    n = len(rows)
    Rr = np.zeros(n, np.float32); A15 = np.full(n, np.nan, np.float32); A15_open = np.zeros(n, bool)
    ends_goal = np.zeros(n, bool)
    key = out["ep"] * 2 + out["slot"]
    order = np.lexsort((out["step"], key))
    k_o = key[order]
    starts = np.flatnonzero(np.r_[True, k_o[1:] != k_o[:-1]])
    for a, b in zip(starts, np.r_[starts[1:], n]):
        idx = order[a:b]
        e = int(out["ep"][idx[0]]); team = int(out["slot"][idx[0]])
        k = np.arange(b - a)[::-1]
        gt = ep_goal.get(e)
        if gt is not None:
            term = pol.goal_u if gt == team else -pol.goal_u
            Rr[idx] = term * GAMMA ** (k + 1); ends_goal[idx] = True
        else:
            Rr[idx] = out["v"][idx[-1]] * GAMMA ** (k + 1)
        vseg = out["v"][idx]
        L = b - a
        for j in range(L):
            if j + 15 < L:
                A15[idx[j]] = GAMMA ** 15 * vseg[j + 15] - vseg[j]; A15_open[idx[j]] = True
            elif gt is not None:
                A15[idx[j]] = term * GAMMA ** (L - j) - vseg[j]
    out["R"] = Rr; out["A15"] = A15; out["A15_open"] = A15_open; out["ends_goal"] = ends_goal
    return envs, snaps, out


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------
class Pool:
    """N persistent arenas stepped in lockstep from one restored snapshot."""

    def __init__(self, n, pol, seed):
        self.envs = [SimArena(np.random.default_rng(seed + 7000 + i)) for i in range(n)]
        self.pol = pol
        self.n = n
        self.rng = np.random.default_rng(seed + 99)

    def run(self, snap, slot, specs, horizon, macro):
        """specs: list (len<=n) of dicts {kind: 'policy'|'open'|'temp', L, actions, temp}.
        Returns scores (critic units), goal flags (+1 self goal, -1 concede, 0), executed
        prefix actions per candidate."""
        import torch
        pol = self.pol
        k = len(specs)
        envs = self.envs[:k]
        for e in envs:
            e.restore(snap)
        alive = np.ones(k, bool)
        score = np.zeros(k, np.float32)
        goal = np.zeros(k, np.int8)
        executed = [[] for _ in range(k)]
        for t in range(horizon):
            idx = np.flatnonzero(alive)
            if len(idx) == 0:
                break
            obs, masks = [], []
            for i in idx:
                o, m, _ = envs[i].observe()
                obs.append(o); masks.append(m)
            obs = np.concatenate(obs); masks = np.concatenate(masks)
            logits, v, vd, H, vexp = pol.forward(obs, masks)
            # per-row temperature: opponent rows temp 1; self rows per spec
            temp = torch.ones(2 * len(idx))
            for j, i in enumerate(idx):
                sp = specs[i]
                if sp["kind"] == "temp" and t < sp["L"] * macro:
                    temp[2 * j + slot] = sp["temp"]
            acts = pol.sample(logits, temp).cpu().numpy()
            for j, i in enumerate(idx):
                sp = specs[i]
                if sp["kind"] == "open" and t < sp["L"] * macro:
                    a = sp["actions"][t // macro]
                    if not masks[2 * j + slot][a]:
                        a = int(self.rng.choice(np.flatnonzero(masks[2 * j + slot])))
                    acts[2 * j + slot] = a
                if t % macro == 0 and t < sp.get("L", 0) * macro:
                    executed[i].append(int(acts[2 * j + slot]))
            for j, i in enumerate(idx):
                e = envs[i]
                e.step(acts[2 * j: 2 * j + 2])
                if e.goal_team is not None:
                    sgn = 1.0 if e.goal_team == slot else -1.0
                    score[i] += GAMMA ** t * sgn * pol.goal_u
                    goal[i] = int(sgn)
                    alive[i] = False
        idx = np.flatnonzero(alive)
        if len(idx):
            obs, masks = [], []
            for i in idx:
                o, m, _ = envs[i].observe()
                obs.append(o); masks.append(m)
            _, v, _, _, _ = pol.forward(np.concatenate(obs), np.concatenate(masks))
            v = v.cpu().numpy()
            for j, i in enumerate(idx):
                score[i] += GAMMA ** horizon * v[2 * j + slot]
        return score, goal, executed


def random_prefix(rng, L):
    return [int(rng.integers(0, NUM_ACTIONS)) for _ in range(L)]


def mutate(rng, actions):
    a = list(actions)
    r = rng.random()
    if r < 0.5 and len(a) > 0:
        a[int(rng.integers(0, len(a)))] = int(rng.integers(0, NUM_ACTIONS))
    elif r < 0.75:
        a.append(int(rng.integers(0, NUM_ACTIONS)))
    elif len(a) > 1:
        a.pop()
    else:
        a[0] = int(rng.integers(0, NUM_ACTIONS))
    return a


def search_state(pool, snap, slot, horizon, macro, n_cand, rounds, n_elite, k_eval, rng,
                 n_final=8, k_final=8):
    # baseline: k_eval fresh closed-loop policy rollouts
    base_specs = [{"kind": "policy", "L": 0} for _ in range(k_eval)]
    bs, bg, _ = pool.run(snap, slot, base_specs, horizon, macro)
    # round 0 candidates
    specs = []
    for L in (2, 4, 8, 12):
        for _ in range(n_cand // 8):
            specs.append({"kind": "open", "L": L, "actions": random_prefix(rng, L)})
    for temp in (2.0, 4.0):
        for L in (4, 8):
            for _ in range(n_cand // 8):
                specs.append({"kind": "temp", "L": L, "temp": temp})
    specs = specs[:n_cand]
    evals = {}  # prefix tuple -> list of scores
    def record(specs, scores, executed, goals):
        for sp, s, ex, g in zip(specs, scores, executed, goals):
            key = tuple(ex)
            evals.setdefault(key, []).append((float(s), int(g)))
    s, g, ex = pool.run(snap, slot, specs, horizon, macro)
    record(specs, s, ex, g)
    for _ in range(rounds - 1):
        ranked = sorted(evals.items(), key=lambda kv: -np.mean([x[0] for x in kv[1]]))
        elites = [list(k) for k, _ in ranked[:n_elite] if len(k) > 0]
        specs = []
        for e in elites:
            specs.append({"kind": "open", "L": len(e), "actions": e})       # re-evaluate
            for _ in range((n_cand - len(elites)) // max(len(elites), 1)):
                m = mutate(rng, e)
                specs.append({"kind": "open", "L": len(m), "actions": m})
        while len(specs) < n_cand:
            L = int(rng.choice([2, 4, 8]))
            specs.append({"kind": "open", "L": L, "actions": random_prefix(rng, L)})
        specs = specs[:n_cand]
        s, g, ex = pool.run(snap, slot, specs, horizon, macro)
        record(specs, s, ex, g)
    # the plain policy is always a candidate (its baseline evals count)
    evals[()] = [(float(x), int(y)) for x, y in zip(bs, bg)]
    # final stage: top-n_final by mean, each re-evaluated k_final fresh times; pick by that
    ranked = sorted(evals.items(), key=lambda kv: -np.mean([x[0] for x in kv[1]]))
    finalists = [list(k) for k, _ in ranked[:n_final]]
    fspecs = []
    for f in finalists:
        fspecs += [{"kind": "open", "L": len(f), "actions": f} for _ in range(k_final)]
    fs, fg, _ = pool.run(snap, slot, fspecs, horizon, macro)
    fmeans = fs.reshape(len(finalists), k_final).mean(1)
    # the policy finalist keeps its k_eval baseline evals as its estimate
    for fi, f in enumerate(finalists):
        if len(f) == 0:
            fmeans[fi] = bs.mean()
    bi = int(np.argmax(fmeans))
    best_key = finalists[bi]
    n_best_evals = len(evals[tuple(best_key)]) + (k_final if len(best_key) else 0)
    # held-out re-evaluation
    re_specs = [{"kind": "open", "L": len(best_key), "actions": best_key} for _ in range(k_eval)]
    rs_, rg, _ = pool.run(snap, slot, re_specs, horizon, macro)
    return {
        "base_mean": float(bs.mean()), "base_std": float(bs.std()),
        "base_goal": float((bg == 1).mean()), "base_concede": float((bg == -1).mean()),
        "best_mean": float(rs_.mean()), "best_std": float(rs_.std()),
        "best_goal": float((rg == 1).mean()), "best_concede": float((rg == -1).mean()),
        "best_insample": float(fmeans[bi]), "best_n_evals": n_best_evals,
        "base_q75_minus_mean": float(np.quantile(bs, 0.75) - bs.mean()),
        "n_finalists_nonpolicy": int(sum(1 for f in finalists if len(f))),
        "best_L": len(best_key), "best_prefix": best_key,
        "gain": float(rs_.mean() - bs.mean()),
        "n_evaluated": int(sum(len(v) for v in evals.values())),
    }


# ---------------------------------------------------------------------------
def select_states(out, per_stratum, rng):
    H, A15 = out["H"], out["A15"]
    n = len(H)
    strata = {}
    q95, q45, q55, q20 = np.quantile(H, [0.95, 0.45, 0.55, 0.20])
    strata["H_top"] = np.flatnonzero(H >= q95)
    q90 = np.quantile(H, 0.90)
    strata["H_top_neutral"] = np.flatnonzero((H >= q90) & (np.abs(out["v"]) < 2.0))
    strata["H_mid"] = np.flatnonzero((H >= q45) & (H <= q55))
    z = np.flatnonzero(H <= 0)
    strata["H_zero"] = z if len(z) >= per_stratum else np.flatnonzero(H <= q20)
    ok = ~np.isnan(A15) & out["A15_open"]          # 15 steps of play remain (no terminal inside)
    q03 = np.quantile(A15[ok], 0.03)
    strata["dip"] = np.flatnonzero(ok & (A15 <= q03))
    strata["uniform"] = np.arange(n)
    picks = []
    for name, idx in strata.items():
        ch = rng.choice(idx, size=min(per_stratum, len(idx)), replace=False)
        for i in ch:
            picks.append((name, int(i)))
    return picks, {"H_q95": float(q95), "H_q45": float(q45), "H_q55": float(q55),
                   "H_zero_frac": float((H <= 0).mean()), "A15_q03": float(q03)}


def worker(args, wid, seed, q):
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    os.nice(5)
    pol = SearchPolicy(args.ckpt, args.device)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    envs, snaps, out = rollout(pol, args.arenas, args.steps, seed)
    t_roll = time.time() - t0
    picks, qs = select_states(out, args.per_stratum, rng)
    pool = Pool(max(args.n_cand, args.n_final * args.k_final, args.k_eval), pol, seed)
    results = []
    t0 = time.time()
    for j, (name, i) in enumerate(picks):
        snap = snaps[out["sid"][i]]
        slot = int(out["slot"][i])
        res = search_state(pool, snap, slot, args.horizon, args.macro, args.n_cand,
                           args.rounds, args.n_elite, args.k_eval, rng,
                           n_final=args.n_final, k_final=args.k_final)
        res.update({
            "stratum": name, "worker": wid, "row": int(i), "slot": slot,
            "H": float(out["H"][i]), "v": float(out["v"][i]), "vd": float(out["vd"][i]),
            "vexp": float(out["vexp"][i]), "A15": float(out["A15"][i]),
            "R_rollout": float(out["R"][i]), "ep_step": int(out["step"][i]),
            "kind": out["ep_kind"].get(int(out["ep"][i]), "?"),
        })
        results.append(res)
        if j % 5 == 0:
            print(f"[w{wid}] {j+1}/{len(picks)} {name} H={res['H']:.2f} base={res['base_mean']:.2f} "
                  f"best={res['best_mean']:.2f} gain={res['gain']:+.2f} L={res['best_L']} "
                  f"({(time.time()-t0)/(j+1):.1f}s/state)", flush=True)
    summary = {
        "worker": wid, "rows": int(len(out["H"])), "t_roll": t_roll, "t_search": time.time() - t0,
        "quantiles": qs, "ret_std": pol.ret_std, "goal_u": pol.goal_u,
        "H_mean": float(out["H"].mean()), "H_pos_frac": float((out["H"] > 0).mean()),
        "V_mean": float(out["v"].mean()), "R_mean": float(out["R"].mean()),
        "EV": float(1 - np.var(out["R"] - out["v"]) / (np.var(out["R"]) + 1e-8)),
        "goal_rows_frac": float(out["ends_goal"].mean()),
        "n_episodes_goal": len(out["ep_goal"]),
    }
    q.put((summary, results))


def selftest(args):
    """restore parity: snapshot -> restore into another arena must reproduce a clone."""
    rng = np.random.default_rng(1)
    a = SimArena(np.random.default_rng(2)); b = SimArena(np.random.default_rng(3))
    a.reset()
    for _ in range(40):
        a.step([int(rng.integers(0, 24)), int(rng.integers(0, 24))])
    snap = a.snapshot()
    clone = a.arena.clone(False)
    b.restore(snap)
    seq = [(int(rng.integers(0, 90)), int(rng.integers(0, 90))) for _ in range(60)]
    from advanced_obs import ACTION_TABLE
    import RocketSim as rs
    for acts in seq:
        for p, act in enumerate(acts):
            e = ACTION_TABLE[act]
            c = rs.CarControls()
            c.throttle, c.steer = float(e[0]), float(e[1])
            c.pitch, c.yaw, c.roll = float(e[2]), float(e[3]), float(e[4])
            c.jump, c.boost, c.handbrake = bool(e[5]), bool(e[6]), bool(e[7])
            clone.get_cars()[p].set_controls(c)
        clone.step(TICK_SKIP)
        b.step(list(acts))
    pb = np.array(b.arena.ball.get_state().pos.as_tuple()); pc = np.array(clone.ball.get_state().pos.as_tuple())
    cb = np.array(b.cars[0].get_state().pos.as_tuple()); cc = np.array(clone.get_cars()[0].get_state().pos.as_tuple())
    print("selftest ball |restore-clone| =", np.abs(pb - pc).max(), " car0 =", np.abs(cb - cc).max())
    # restore twice must be identical
    b.restore(snap); [b.step(list(x)) for x in seq]; p1 = np.array(b.arena.ball.get_state().pos.as_tuple())
    b.restore(snap); [b.step(list(x)) for x in seq]; p2 = np.array(b.arena.ball.get_state().pos.as_tuple())
    print("selftest restore-restore =", np.abs(p1 - p2).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--arenas", type=int, default=16)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--per-stratum", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=60)   # 4 s at 15 Hz
    ap.add_argument("--macro", type=int, default=3)      # 0.2 s
    ap.add_argument("--n-cand", type=int, default=96)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--n-elite", type=int, default=12)
    ap.add_argument("--k-eval", type=int, default=24)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--n-final", type=int, default=8)
    ap.add_argument("--k-final", type=int, default=8)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest(args); return
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(args, w, args.seed + 101 * w, q)) for w in range(args.workers)]
    for p in procs:
        p.start()
    summaries, results = [], []
    for _ in procs:
        s, r = q.get()
        summaries.append(s); results += r
    for p in procs:
        p.join()
    json.dump({"args": vars(args), "summaries": summaries, "results": results}, open(args.out, "w"), indent=1)
    report(results, summaries)


def report(results, summaries):
    from scipy.stats import spearmanr
    by = {}
    for r in results:
        by.setdefault(r["stratum"], []).append(r)
    print("\n=== per stratum (critic units; goal = %.2f) ===" % summaries[0]["goal_u"])
    print(f"{'stratum':14s} {'n':>4s} {'H':>6s} {'base':>7s} {'best':>7s} {'gain':>7s} {'se':>5s} {'>0.5':>5s} "
          f"{'Pgoal b':>8s} {'Pgoal s':>8s} {'Pconc b':>8s} {'Pconc s':>8s} {'L':>4s}")
    for name in ("H_top", "H_top_neutral", "H_mid", "H_zero", "dip", "uniform"):
        rs = by.get(name, [])
        if not rs:
            continue
        g = np.array([r["gain"] for r in rs])
        print(f"{name:14s} {len(rs):4d} {np.mean([r['H'] for r in rs]):6.2f} "
              f"{np.mean([r['base_mean'] for r in rs]):7.2f} {np.mean([r['best_mean'] for r in rs]):7.2f} "
              f"{g.mean():+7.2f} {g.std()/np.sqrt(len(g)):5.2f} {(g>0.5).mean():5.2f} "
              f"{np.mean([r['base_goal'] for r in rs]):8.2f} {np.mean([r['best_goal'] for r in rs]):8.2f} "
              f"{np.mean([r['base_concede'] for r in rs]):8.2f} {np.mean([r['best_concede'] for r in rs]):8.2f} "
              f"{np.mean([r['best_L'] for r in rs]):4.1f}")
    u = by.get("uniform", [])
    if len(u) > 5:
        H = np.array([r["H"] for r in u]); g = np.array([r["gain"] for r in u])
        v = np.array([r["v"] for r in u]); b = np.array([r["base_mean"] for r in u])
        print(f"uniform: Spearman(H, gain) = {spearmanr(H, g).correlation:+.3f}  "
              f"Spearman(V_real, MC baseline) = {spearmanr(v, b).correlation:+.3f}  "
              f"mean(V_real - MC) = {np.mean(v-b):+.2f}")
    if len(u) > 5:
        sp = np.array([r["base_q75_minus_mean"] for r in u])
        print(f"uniform: Spearman(H, MC q75-mean) = {spearmanr(H, sp).correlation:+.3f}  mean H {H.mean():.2f} vs mean MC q75-mean {sp.mean():.2f}")
    allr = results
    H = np.array([r["H"] for r in allr]); g = np.array([r["gain"] for r in allr])
    print(f"all: Spearman(H, gain) = {spearmanr(H, g).correlation:+.3f}   n={len(allr)}")
    for s in summaries:
        print(f"  w{s['worker']}: rows {s['rows']} roll {s['t_roll']:.0f}s search {s['t_search']:.0f}s "
              f"H_mean {s['H_mean']:.2f} H>0 {s['H_pos_frac']:.2f} EV {s['EV']:.2f} goal_rows {s['goal_rows_frac']:.2f}")


if __name__ == "__main__":
    main()
