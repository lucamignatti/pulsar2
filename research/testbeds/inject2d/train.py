"""PPO + ladder + injection arms on Aerial2D.

Original arms: ppo, pbrs, pbrs_raw, servo_raw, entgate, temp, intr, sil,
aprior, align, kstep, sparse, decay, sil_nogate, sil_ent, sil_pbrs.

Meta-transfer arms (run with --shift to move the frontier mid-run):
  sil        scaffold only (baseline)
  silobs     sil + the policy SEES its own frontier: obs augmented with
             tanh(Vdag - Vexp) and tanh(Vdag/5), read-only ladder features
  silobs_lp  silobs + LEARNING-PROGRESS gating: SIL gate and entropy gate keyed
             to relu(H_old - H_now) (headroom that is currently FALLING),
             ladder+critic snapshots every 10 iters supply H_old

Gap-closure arms ("minimize max-vs-avg return", gap = relu(Vexp - V)):
  gap_rew    plain state penalty  r -= k*unit(gap)          [pre-GAE, priced]
  gap_pre    potential-difference form injected into REWARDS pre-GAE
             ("pbrs, not pbrs": PBRS shape, but priced by the critic and
             built from a learned, nonstationary potential)
  gap_post   same potential through the incumbent post-GAE advantage channel
  gap_sil    closure-by-elevation: SIL on above-average realizations in
             HIGH-GAP states (pull the average up to the max, never push away)
"""
import argparse, copy, json, os, time
import numpy as np
import torch
import torch.nn as nn

from env import Aerial2D, N_ACT, OBS_DIM, EP_LEN
from ladder import Ladder, mlp

GAMMA = 0.99
LAM = 0.95
CLIP = 0.2
ENT0 = 0.02
LR = 3e-4
EPOCHS = 3
MB = 4
N_ENVS = 128
HORIZON = 128
BETA = 0.04
GAMMA_INT = 0.95

AUG_ARMS = ("silobs", "silobs_lp")
SIL_ARMS = ("sil", "sil_nogate", "sil_ent", "sil_pbrs", "silobs", "silobs_lp", "gap_sil")
PBRS_ARMS = ("pbrs", "pbrs_raw", "servo_raw", "kstep", "sparse", "sil_pbrs")


def masked_dist(logits, mask):
    return torch.distributions.Categorical(logits=logits + (mask - 1.0) * 1e9)


class Agent:
    def __init__(self, arm, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.arm = arm
        self.aug = arm in AUG_ARMS
        self.in_dim = OBS_DIM + (2 if self.aug else 0)
        self.pi = mlp(self.in_dim, N_ACT)
        self.v = mlp(self.in_dim, 1)
        self.opt_pi = torch.optim.Adam(self.pi.parameters(), lr=LR)
        self.opt_v = torch.optim.Adam(self.v.parameters(), lr=LR)
        self.ladder = Ladder(OBS_DIM, N_ACT, GAMMA, {"qdag": arm == "aprior"})
        self.v_int = mlp(OBS_DIM, 1) if arm in ("intr", "align") else None
        if self.v_int is not None:
            self.opt_vint = torch.optim.Adam(self.v_int.parameters(), lr=LR)
        self.beta_eff = BETA
        self.snap = None            # LP snapshot (silobs_lp)
        self.it = 0
        self.env = Aerial2D(N_ENVS, seed=seed + 1000)

    @torch.no_grad()
    def _feats(self, o_raw):
        vd = self.ladder.vdag_min(o_raw)
        ve = self.ladder.vexp_val(o_raw)
        return torch.stack([torch.tanh(vd - ve), torch.tanh(vd / 5.0)], dim=1)

    def _aug(self, o_raw):
        if not self.aug:
            return o_raw
        return torch.cat([o_raw, self._feats(o_raw)], dim=1)

    # ---------------- collection ----------------
    @torch.no_grad()
    def collect(self):
        env = self.env
        T, N = HORIZON, N_ENVS
        obs = torch.zeros(T, N, self.in_dim)
        nxt = torch.zeros(T, N, self.in_dim)
        acts = torch.zeros(T, N, dtype=torch.long)
        logp = torch.zeros(T, N)
        rews = torch.zeros(T, N)
        dones = torch.zeros(T, N)
        masks = torch.zeros(T, N, N_ACT)
        counts = {"touch": 0, "air": 0, "hi": 0, "hi_off": 0}

        o = self._aug(torch.from_numpy(env.obs()))
        for t in range(T):
            m = torch.from_numpy(env.action_mask())
            logits = self.pi(o)
            logits_act = self.act_logits(logits, o, m)
            dist = masked_dist(logits_act, m)
            a = dist.sample()
            r, d, info = env.step(a.numpy())
            obs[t] = o; masks[t] = m; acts[t] = a
            logp[t] = dist.log_prob(a)
            rews[t] = torch.from_numpy(r)
            dones[t] = torch.from_numpy(d.astype(np.float32))
            o2 = self._aug(torch.from_numpy(env.obs()))
            if d.any():
                fo = self._aug(torch.from_numpy(info["final_obs"]))
                nxt[t] = torch.where(torch.from_numpy(d).unsqueeze(1), fo, o2)
            else:
                nxt[t] = o2
            o = o2
            counts["touch"] += int(info["touch"].sum())
            counts["air"] += int(info["air_touch"].sum())
            counts["hi"] += int(info["hi_touch"].sum())
            counts["hi_off"] += int(info["hi_offered"].sum())
        return obs, nxt, acts, logp, rews, dones, masks, counts

    def act_logits(self, logits, o, m):
        if self.arm == "temp":
            vr = self.v(o).flatten()
            _, h = self.ladder.fields(o[:, :OBS_DIM], vr)
            tmp = (1.0 + 0.5 * h.clamp(0, 3)).unsqueeze(1)
            return logits / tmp
        if self.arm == "aprior":
            vr = self.v(o).flatten()
            q = self.ladder.qdag_val(o[:, :OBS_DIM])
            adv = torch.relu(q - vr.unsqueeze(1))
            adv = adv / (adv.std() + 1e-8)
            return logits + 0.75 * adv * m
        return logits

    @torch.no_grad()
    def _h_old(self, fo_r, fo_aug):
        """Headroom as estimated by the snapshot nets (for learning progress)."""
        vd = torch.minimum(self.snap["vd1"](fo_r).flatten(), self.snap["vd2"](fo_r).flatten())
        geo = self.snap["geo"](fo_r).flatten()
        vo = self.snap["v"](fo_aug).flatten()
        eps = 1e-8
        gs = (geo - geo.mean()) / (geo.std() + eps) * (vo.std() + eps) + vo.mean()
        unit = lambda x: x / (x.std() + eps)
        return 0.5 * unit(torch.relu(gs - vo)) + 0.5 * unit(torch.relu(vd - vo))

    # ---------------- learn ----------------
    def learn(self, batch):
        obs, nxt, acts, logp0, rews, dones, masks, counts = batch
        T, N = HORIZON, N_ENVS
        fo = obs.reshape(T * N, -1); fn = nxt.reshape(T * N, -1)
        fo_r = fo[:, :OBS_DIM]; fn_r = fn[:, :OBS_DIM]
        fa = acts.reshape(-1); flp = logp0.reshape(-1)
        fd = dones.reshape(-1); fm = masks.reshape(T * N, -1)
        cont = 1.0 - fd
        stats = {}

        with torch.no_grad():
            v = self.v(fo).flatten()
            v_next = self.v(fn).flatten()

        # ---- gap-closure PRE-GAE injections (modify rewards, get priced) ----
        if self.arm in ("gap_rew", "gap_pre"):
            with torch.no_grad():
                ve_o = self.ladder.vexp_val(fo_r)
                ve_n = self.ladder.vexp_val(fn_r)
            gap_o = torch.relu(ve_o - v)
            gap_n = torch.relu(ve_n - v_next)
            sd = gap_o.std() + 1e-8
            ug, ugn = gap_o / sd, gap_n / sd
            if self.arm == "gap_rew":
                inj_r = -0.02 * ug
            else:  # gap_pre: paid when the gap FALLS along the trajectory
                inj_r = 0.05 * (ug - GAMMA * cont * ugn)
            rews = rews + inj_r.view(T, N)
            stats["gap_mean"] = float(gap_o.mean())

        fr = rews.reshape(-1)

        # GAE: bootstrap through truncation, cut the lambda-trace
        adv = torch.zeros(T, N)
        vv = v.view(T, N); vn = v_next.view(T, N)
        gae = torch.zeros(N)
        for t in reversed(range(T)):
            delta = rews[t] + GAMMA * vn[t] - vv[t]
            gae = delta + GAMMA * LAM * (1 - dones[t]) * gae
            adv[t] = gae
        returns = (adv + vv).reshape(-1)
        fadv = adv.reshape(-1)
        fadv = (fadv - fadv.mean()) / (fadv.std() + 1e-8)

        # ladder update on RAW obs, on the agent's actual reward stream
        lstats = self.ladder.update(fo_r, fn_r, fr, fd, fa, returns)
        stats.update(lstats)

        # fields in one normalization frame
        with torch.no_grad():
            v_all = self.v(torch.cat([fo, fn])).flatten()
        phi_all, h_all = self.ladder.fields(torch.cat([fo_r, fn_r]), v_all)
        phi_o, phi_n = phi_all[:T * N], phi_all[T * N:]
        h_o = h_all[:T * N]

        ent_w = torch.ones(T * N)
        sil_mask = None; sil_w = None

        # ---- learning-progress field (silobs_lp) ----
        gate_field = h_o
        if self.arm == "silobs_lp":
            if self.snap is not None:
                lp = torch.relu(self._h_old(fo_r, fo) - h_o)
                gate_field = lp
                stats["lp_mean"] = float(lp.mean())
            if self.it % 10 == 0:
                self.snap = {
                    "vd1": copy.deepcopy(self.ladder.vdag1),
                    "vd2": copy.deepcopy(self.ladder.vdag2),
                    "geo": copy.deepcopy(self.ladder.geo_v),
                    "v": copy.deepcopy(self.v),
                }
            ent_w = (1.0 + gate_field.clamp(min=0) / (gate_field.clamp(min=0).mean() + 1e-8)).clamp(1.0, 3.0)

        if self.arm == "decay":
            with torch.no_grad():
                geo = self.ladder.geo_val(torch.cat([fo_r, fn_r]))
            gm, gs = geo.mean(), geo.std() + 1e-8
            vm, vs = v_all.mean(), v_all.std() + 1e-8
            hg = torch.relu((geo - gm) / gs * vs + vm - v_all)
            g = GAMMA * cont * (hg[T * N:]) - hg[:T * N]
            g = g - g.mean()
            if not hasattr(self, "inj_scale"):
                self.inj_scale = None; self.calib = []
            if self.inj_scale is None:
                self.calib.append(float(g.std()))
                if len(self.calib) >= 20:
                    self.inj_scale = max(float(np.mean(self.calib)), 1e-3)
                si = max(float(g.std()), 0.05)
            else:
                si = self.inj_scale
            inj = (BETA / si * g).clamp(-3, 3)
            stats["dose"] = float(inj.std())
            fadv = fadv + inj

        if self.arm in PBRS_ARMS:
            if self.arm in ("pbrs_raw", "servo_raw"):
                with torch.no_grad():
                    vdag = self.ladder.vdag_min(torch.cat([fo_r, fn_r]))
                    geo = self.ladder.geo_val(torch.cat([fo_r, fn_r]))
                gm, gs = geo.mean(), geo.std() + 1e-8
                vm, vs = v_all.mean(), v_all.std() + 1e-8
                hg = torch.relu((geo - gm) / gs * vs + vm - v_all)
                phi_raw = 0.5 * hg + 0.5 * vdag
                phi_o, phi_n = phi_raw[:T * N], phi_raw[T * N:]
            if self.arm == "kstep":
                g = self.kstep_potential(phi_o.view(T, N), dones)
            else:
                g = GAMMA * cont * phi_n - phi_o
            g = g - g.mean()
            if self.arm == "sparse":
                q = torch.quantile(h_o, 0.8)
                g = torch.where(h_o >= q, g, torch.zeros_like(g))
                g = g - g.mean()
            beta = self.beta_eff if self.arm == "servo_raw" else BETA
            si = max(0.05, float(g.std()))
            inj = (beta / si * g).clamp(-3, 3)
            stats["dose"] = float(inj.std())
            if self.arm == "servo_raw":
                ratio = np.clip(BETA / max(float(inj.std()), 1e-6), 0.5, 2.0)
                self.beta_eff = float(np.clip(self.beta_eff * ratio ** 0.5, 1e-4, 1.0))
                stats["beta_eff"] = self.beta_eff
            fadv = fadv + inj

        if self.arm == "gap_post":
            with torch.no_grad():
                ve_o = self.ladder.vexp_val(fo_r)
                ve_n = self.ladder.vexp_val(fn_r)
            gap_o = torch.relu(ve_o - v); gap_n = torch.relu(ve_n - v_next)
            sd = gap_o.std() + 1e-8
            g = gap_o / sd - GAMMA * cont * (gap_n / sd)   # Phi = -unit(gap)
            g = g - g.mean()
            si = max(0.05, float(g.std()))
            inj = (BETA / si * g).clamp(-3, 3)
            stats["dose"] = float(inj.std())
            fadv = fadv + inj

        if self.arm in ("entgate", "sil_ent"):
            hn = h_o.clamp(min=0) / (h_o.clamp(min=0).mean() + 1e-8)
            ent_w = (1.0 + 1.0 * hn).clamp(1.0, 3.0)
            stats["ent_gate_mean"] = float(ent_w.mean())

        if self.arm in ("intr", "align"):
            if self.arm == "intr":
                r_int = GAMMA * cont * phi_n - phi_o
            else:
                dirs = self.ladder.geo_grad_dir(fo_r)
                dlt = fn_r - fo_r
                r_int = nn.functional.cosine_similarity(dlt, dirs, dim=1) * cont
            r_int = (r_int - r_int.mean()).view(T, N)
            with torch.no_grad():
                vi = self.v_int(fo_r).flatten().view(T, N)
                vin = self.v_int(fn_r).flatten().view(T, N)
            a_int = torch.zeros(T, N); gi = torch.zeros(N)
            for t in reversed(range(T)):
                delta = r_int[t] + GAMMA_INT * vin[t] - vi[t]
                gi = delta + GAMMA_INT * LAM * (1 - dones[t]) * gi
                a_int[t] = gi
            ret_int = (a_int + vi).reshape(-1)
            fai = a_int.reshape(-1)
            fai = (fai - fai.mean()) / (fai.std() + 1e-8)
            fadv = fadv + 0.15 * fai
            for _ in range(EPOCHS):
                for c in torch.randperm(T * N).chunk(MB):
                    lv = ((self.v_int(fo_r[c]).flatten() - ret_int[c]) ** 2).mean()
                    self.opt_vint.zero_grad(); lv.backward()
                    nn.utils.clip_grad_norm_(self.v_int.parameters(), 0.5)
                    self.opt_vint.step()

        if self.arm in SIL_ARMS:
            with torch.no_grad():
                vexp = self.ladder.vexp_val(fo_r)
            if self.arm == "gap_sil":
                gap = torch.relu(vexp - v)
                sil_mask = (returns > v) & (gap >= torch.quantile(gap, 0.7))
            else:
                sil_mask = returns > vexp
                if self.arm != "sil_nogate":
                    sil_mask = sil_mask & (gate_field >= torch.quantile(gate_field, 0.7))
            sil_w = (returns - v).clamp(0, 2.0)
            stats["sil_frac"] = float(sil_mask.float().mean())

        # PPO epochs
        kl_acc, ent_acc, nmb = 0.0, 0.0, 0
        for _ in range(EPOCHS):
            for c in torch.randperm(T * N).chunk(MB):
                dist = masked_dist(self.pi(fo[c]), fm[c])
                lp = dist.log_prob(fa[c])
                ratio = (lp - flp[c]).exp()
                a = fadv[c]
                l1 = ratio * a
                l2 = ratio.clamp(1 - CLIP, 1 + CLIP) * a
                ent = dist.entropy()
                loss_pi = -(torch.min(l1, l2) + ENT0 * ent_w[c] * ent).mean()
                if sil_mask is not None:
                    mm = sil_mask[c]
                    if mm.any():
                        loss_pi = loss_pi + 0.1 * (-lp[mm] * sil_w[c][mm]).mean()
                self.opt_pi.zero_grad(); loss_pi.backward()
                nn.utils.clip_grad_norm_(self.pi.parameters(), 0.5)
                self.opt_pi.step()
                lv = ((self.v(fo[c]).flatten() - returns[c]) ** 2).mean()
                self.opt_v.zero_grad(); lv.backward()
                nn.utils.clip_grad_norm_(self.v.parameters(), 0.5)
                self.opt_v.step()
                with torch.no_grad():
                    kl_acc += float((flp[c] - lp).mean()); ent_acc += float(ent.mean()); nmb += 1

        self.it += 1
        stats.update({
            "touch_1k": counts["touch"] / (T * N) * 1000,
            "air_1k": counts["air"] / (T * N) * 1000,
            "hi_1k": counts["hi"] / (T * N) * 1000,
            "hi_conv": counts["hi"] / max(1, counts["hi_off"]) * 1000,
            "ep_rew": float(fr.sum() / (T * N / EP_LEN)),
            "kl": kl_acc / max(nmb, 1), "entropy": ent_acc / max(nmb, 1),
            "phase": self.env.phase,
        })
        return stats

    def kstep_potential(self, phi, dones, k=8):
        T, N = phi.shape
        g = torch.zeros(T, N)
        cd = torch.cat([torch.zeros(1, N), dones.cumsum(0)], 0)
        for t in range(0, T - k, k):
            ok = (cd[t + k] - cd[t]) == 0
            g[t] = torch.where(ok, (GAMMA ** k) * phi[t + k] - phi[t], torch.zeros(N))
        return g.reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--shift", type=float, default=0.0,
                    help="fraction of run at which to move the frontier (env phase 2)")
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()
    torch.set_num_threads(2)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.arm}_s{args.seed}.jsonl")
    agent = Agent(args.arm, args.seed)
    iters = args.steps // (N_ENVS * HORIZON)
    shift_iter = int(iters * args.shift) if args.shift > 0 else -1
    t0 = time.time()
    with open(path, "w") as f:
        for it in range(iters):
            if it == shift_iter:
                agent.env.set_phase(2)
            batch = agent.collect()
            stats = agent.learn(batch)
            stats["iter"] = it
            stats["steps"] = (it + 1) * N_ENVS * HORIZON
            f.write(json.dumps({k: round(float(v), 5) if isinstance(v, (int, float)) else v
                                for k, v in stats.items()}) + "\n")
            if it % 20 == 0:
                f.flush()
                print(f"[{args.arm} s{args.seed}] it {it}/{iters} ph{stats['phase']} "
                      f"touch {stats['touch_1k']:.2f} air {stats['air_1k']:.3f} "
                      f"hi {stats['hi_1k']:.3f} ({(time.time()-t0):.0f}s)", flush=True)
    print(f"done {path} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
