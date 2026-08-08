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
VH_ARMS = ("vh_twin", "vh_q", "vh_mir", "vh_aux", "vh_dec", "vh_td", "vh_best", "vh_bestaux", "vh_final", "vh_epi", "vh_grp", "vh_epifinal")
SIL_ARMS = ("sil", "sil_eps", "sil_hull", "sil_nogate", "sil_ent", "sil_pbrs", "silobs", "silobs_lp", "gap_sil") + VH_ARMS
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
        self.vvar = arm[3:] if arm.startswith("vh_") else ""
        vout = {"q": 8, "aux": 1 + 2 * OBS_DIM, "dec": 3,
                "bestaux": 1 + 2 * OBS_DIM, "final": 1 + 2 * OBS_DIM,
                "epifinal": 1 + 2 * OBS_DIM}.get(self.vvar, 1)
        v2out = 1 + 2 * OBS_DIM if self.vvar in ("bestaux", "final", "epifinal") else 1
        self.v = mlp(self.in_dim, vout)
        self.v2 = mlp(self.in_dim, v2out) if self.vvar in ("twin", "best", "bestaux", "final", "epifinal") else None
        self.opt_pi = torch.optim.Adam(self.pi.parameters(), lr=LR)
        vparams = list(self.v.parameters()) + (list(self.v2.parameters()) if self.v2 else [])
        self.opt_v = torch.optim.Adam(vparams, lr=LR)
        self.ladder = Ladder(OBS_DIM, N_ACT, GAMMA,
            {"qdag": arm == "aprior",
             "eps": 1.0 if arm in ("sil_eps", "sil_hull") or arm in VH_ARMS else 0.0,
             "hull": arm == "sil_hull" or arm in VH_ARMS})
        self.v_int = mlp(OBS_DIM, 1) if arm in ("intr", "align") else None
        if self.v_int is not None:
            self.opt_vint = torch.optim.Adam(self.v_int.parameters(), lr=LR)
        self.beta_eff = BETA
        self.snap = None            # LP snapshot (silobs_lp)
        self.it = 0
        self.env = Aerial2D(N_ENVS, seed=seed + 1000)

    def val(self, o):
        """Scalar value readout per critic variant (differentiable)."""
        out = self.v(o)
        if self.vvar == "q":
            return out.mean(1)                      # mean of quantiles
        if self.vvar == "aux":
            return out[:, 0]
        if self.vvar == "dec":
            return out.sum(1)                       # channel heads sum to V
        if self.vvar == "twin":
            return 0.5 * (out.flatten() + self.v2(o).flatten())
        if self.vvar == "best":
            return 0.5 * (out.flatten() + self.v2(o).flatten())
        if self.vvar in ("bestaux", "final", "epifinal"):
            return 0.5 * (out[:, 0] + self.v2(o)[:, 0])
        return out.flatten()

    @staticmethod
    def mirror(o):
        m = o.clone(); m[:, 0] = -m[:, 0]; m[:, 2] = -m[:, 2]; m[:, 6] = -m[:, 6]
        return m

    def value_loss(self, o, nx, ret, ret3, d):
        """Per-variant critic loss (the thing under test)."""
        vv = self.vvar
        if vv == "q":
            # quantile (pinball) regression: 8 quantiles vs the same GAE returns
            taus = torch.arange(0.5 / 8, 1.0, 1.0 / 8)
            u = ret.unsqueeze(1) - self.v(o)
            return (torch.where(u > 0, taus, taus - 1.0) * u).mean() * 2.0
        if vv == "twin":
            # differential pair: independent nets on DISJOINT halves of each
            # minibatch -- uncorrelated sample noise, mean readout cancels it
            h = o.shape[0] // 2
            l1 = ((self.v(o[:h]).flatten() - ret[:h]) ** 2).mean()
            l2 = ((self.v2(o[h:]).flatten() - ret[h:]) ** 2).mean()
            return l1 + l2
        if vv == "mir":
            # exact env symmetry = free doubled value data
            base = ((self.v(o).flatten() - ret) ** 2).mean()
            mirr = ((self.v(self.mirror(o)).flatten() - ret) ** 2).mean()
            return 0.5 * (base + mirr)
        if vv == "aux":
            # representation pressure: displacement NLL beside the value output
            out = self.v(o)
            base = ((out[:, 0] - ret) ** 2).mean()
            mu = out[:, 1:1 + OBS_DIM]
            ls = out[:, 1 + OBS_DIM:].clamp(-7, 2)
            dlt = (nx - o)[:, :OBS_DIM]
            keep = (d < 0.5).unsqueeze(1)
            nll = ((ls + 0.5 * ((dlt - mu) / ls.exp()) ** 2) * keep).mean()
            return base + 0.1 * nll
        if vv == "dec":
            # compositional standard value: channel heads on channel returns
            return ((self.v(o) - ret3) ** 2).mean()
        if vv in ("best", "bestaux", "final", "epifinal"):
            # composites; "final" = twin x mirror x aux, NO record-consistency
            # (measured: the TD term biases V low -> inflates the H-floor)
            aux = vv in ("bestaux", "final", "epifinal")
            h = o.shape[0] // 2
            loss = 0.0
            for net, sl in ((self.v, slice(0, h)), (self.v2, slice(h, None))):
                oo, rr2 = o[sl], ret[sl]
                for inp in (oo, self.mirror(oo)):
                    out = net(inp)
                    vpred = out[:, 0] if aux else out.flatten()
                    loss = loss + 0.5 * ((vpred - rr2) ** 2).mean()
                if aux:
                    out = net(oo)
                    mu = out[:, 1:1 + OBS_DIM]
                    ls = out[:, 1 + OBS_DIM:].clamp(-7, 2)
                    dlt = (nx[sl] - oo)[:, :OBS_DIM]
                    keep = (d[sl] < 0.5).unsqueeze(1)
                    loss = loss + 0.1 * ((ls + 0.5 * ((dlt - mu) / ls.exp()) ** 2) * keep).mean()
            if vv not in ("final", "epifinal") and self.ladder.res.fill > 4096:
                ro, rn2, rr = self.ladder.res.sample(o.shape[0])
                with torch.no_grad():
                    tgt = rr + GAMMA * self.val(rn2)
                loss = loss + 0.3 * ((self.val(ro) - tgt) ** 2).mean()
            return loss
        if vv == "td":
            # record-composition: Bellman consistency over ALL past data
            base = ((self.v(o).flatten() - ret) ** 2).mean()
            if self.ladder.res.fill > 4096:
                ro, rn, rr = self.ladder.res.sample(o.shape[0])
                with torch.no_grad():
                    tgt = rr + GAMMA * self.val(rn)
                td = ((self.val(ro) - tgt) ** 2).mean()
                return base + 0.3 * td
            return base
        return ((self.v(o).flatten() - ret) ** 2).mean()

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
        parts = torch.zeros(T, N, 3)
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
            parts[t] = torch.from_numpy(info["r_parts"].astype(np.float32))
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
        return obs, nxt, acts, logp, rews, dones, masks, counts, parts

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
        obs, nxt, acts, logp0, rews, dones, masks, counts, parts = batch
        T, N = HORIZON, N_ENVS
        fo = obs.reshape(T * N, -1); fn = nxt.reshape(T * N, -1)
        fo_r = fo[:, :OBS_DIM]; fn_r = fn[:, :OBS_DIM]
        fa = acts.reshape(-1); flp = logp0.reshape(-1)
        fd = dones.reshape(-1); fm = masks.reshape(T * N, -1)
        cont = 1.0 - fd
        stats = {}

        with torch.no_grad():
            v = self.val(fo)
            v_next = self.val(fn)
        # EPISODIC BASELINE (vh_epi): kNN-over-bank value read, blended by the
        # learned critic's own inadequacy (w = 0.5*(1-EV_ema)) -- leans on memory
        # while V is young, self-anneals as V matures. State-dependent, so the
        # policy-gradient baseline stays unbiased; bootstrap bias judged empirically.
        if self.vvar in ("epi", "epifinal") and self.ladder.res.fill > 8192:
            with torch.no_grad():
                bo, br = self.ladder.res.sample_ret(2048)
                for tgt in ("fo", "fn"):
                    q = fo if tgt == "fo" else fn
                    dd = torch.cdist(q[:, :OBS_DIM], bo)
                    nd, ni = dd.topk(8, dim=1, largest=False)
                    wts = 1.0 / (nd + 1e-2)
                    epiv = (br[ni] * wts).sum(1) / wts.sum(1)
                    w = 0.5 * max(0.0, 1.0 - getattr(self, "ev_ema", 0.0))
                    if tgt == "fo":
                        v = (1 - w) * v + w * epiv
                    else:
                        v_next = (1 - w) * v_next + w * epiv
                self._epi_w = w

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

        returns3 = None
        if self.vvar == "dec":
            with torch.no_grad():
                v3 = self.v(fo).view(T, N, 3)
                v3n = self.v(fn).view(T, N, 3)
            a3 = torch.zeros(T, N, 3); g3 = torch.zeros(N, 3)
            for t in reversed(range(T)):
                d3 = parts[t] + GAMMA * v3n[t] - v3[t]
                g3 = d3 + GAMMA * LAM * (1 - dones[t]).unsqueeze(1) * g3
                a3[t] = g3
            returns3 = (a3 + v3).reshape(T * N, 3)

        # ladder update on RAW obs, on the agent's actual reward stream
        lstats = self.ladder.update(fo_r, fn_r, fr, fd, fa, returns)
        stats.update(lstats)
        with torch.no_grad():
            stats["ev"] = float(1.0 - (returns - v).var() / (returns.var() + 1e-8))
            self.ev_ema = 0.95 * getattr(self, "ev_ema", 0.0) + 0.05 * max(0.0, stats["ev"])
            if hasattr(self, "_epi_w"):
                stats["epi_w"] = self._epi_w
            sub = torch.randint(0, T * N, (4096,))
            stats["h_floor"] = float(torch.relu(
                self.ladder.vdag_min(fo_r[sub]) - v[sub]).mean())
            if self.vvar == "twin":
                stats["v_disagree"] = float(
                    (self.v(fo[sub]).flatten() - self.v2(fo[sub]).flatten()).abs().mean())

        # fields in one normalization frame
        with torch.no_grad():
            v_all = self.val(torch.cat([fo, fn]))
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
            elif self.vvar == "grp" and self.ladder.res.fill > 8192:
                # GROUP-RELATIVE conversion: beat the q75 of 16 nearest peer
                # HISTORIES (non-parametric referee, immune to V_exp drift)
                bo, br = self.ladder.res.sample_ret(2048)
                dd = torch.cdist(fo_r, bo)
                ni = dd.topk(16, dim=1, largest=False).indices
                thr = br[ni].quantile(0.75, dim=1)
                sil_mask = returns > thr
                stats["grp_thr"] = float(thr.mean())
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
                lv = self.value_loss(fo[c], fn[c], returns[c],
                                     returns3[c] if returns3 is not None else None,
                                     fd[c])
                self.opt_v.zero_grad(); lv.backward()
                nn.utils.clip_grad_norm_(
                    list(self.v.parameters())
                    + (list(self.v2.parameters()) if self.v2 else []), 0.5)
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
    ap.add_argument("--save", default="", help="save nets + reservoir here at end")
    ap.add_argument("--pad", action="store_true", help="boost-pad env variant")
    args = ap.parse_args()
    torch.set_num_threads(2)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.arm}_s{args.seed}.jsonl")
    agent = Agent(args.arm, args.seed)
    if args.pad:
        agent.env.enable_pad()
    iters = args.steps // (N_ENVS * HORIZON)
    shift_iter = int(iters * args.shift) if args.shift > 0 else -1
    t0 = time.time()
    traj_obs, traj_done = [], []   # trajectory-contiguous windows for --save
    with open(path, "w") as f:
        for it in range(iters):
            if it == shift_iter:
                agent.env.set_phase(2)
            batch = agent.collect()
            if args.save:
                traj_obs.append(batch[0][:, :, :OBS_DIM].clone())
                traj_done.append(batch[5].clone())
                if len(traj_obs) > 25:
                    traj_obs.pop(0); traj_done.pop(0)
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
    if args.save:
        L = agent.ladder
        torch.save({
            "pi": agent.pi.state_dict(), "v": agent.v.state_dict(),
            "vexp": L.vexp.state_dict(),
            "vdag1": L.vdag1.state_dict(), "vdag2": L.vdag2.state_dict(),
            "geo_sigma": L.geo_sigma.state_dict(), "geo_rew": L.geo_rew.state_dict(),
            "geo_v": L.geo_v.state_dict(),
            "res_obs": L.res.obs[:L.res.fill], "res_nxt": L.res.nxt[:L.res.fill],
            "res_rew": L.res.rew[:L.res.fill],
            "traj_obs": torch.stack(traj_obs), "traj_done": torch.stack(traj_done),
            "arm": agent.arm, "in_dim": agent.in_dim,
        }, args.save)
        print(f"saved {args.save}")
    print(f"done {path} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
