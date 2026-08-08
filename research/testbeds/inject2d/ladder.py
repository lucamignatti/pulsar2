"""The four-rung ladder, testbed scale.

V           mean critic (trained in ppo.py on GAE returns)
V_exp       return-level expectile twin (tau=0.8), measurement + SIL gate
V-dagger    twin heads, one-step expectile TD over executed transitions,
            min-in-target (the anti-ratchet), target nets
V_geo       sigma-net + r-hat-net + HJB field, uniform reservoir over all data
Q-dagger    (arm 'aprior' only) per-action composition head

All small MLPs. Everything trains on executed data only.
"""
import numpy as np
import torch
import torch.nn as nn

def mlp(i, o, h=64):
    return nn.Sequential(nn.Linear(i, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, o))


def expectile_loss(pred, target, tau):
    d = target - pred
    w = torch.where(d > 0, tau, 1 - tau)
    return (w * d * d).mean()


class Reservoir:
    """Uniform sample over all (s, s', r_arrival) ever seen. Boundary pairs are
    the CALLER's job to exclude (never feed transitions that cross a reset)."""
    def __init__(self, cap, obs_dim):
        self.cap = cap
        self.obs = torch.zeros(cap, obs_dim)
        self.nxt = torch.zeros(cap, obs_dim)
        self.rew = torch.zeros(cap)
        self.ret = torch.zeros(cap)   # realized GAE return of the departure row
        self.fill = 0
        self.seen = 0

    def add(self, obs, nxt, rew, ret=None):
        n = obs.shape[0]
        if ret is None:
            ret = torch.zeros(n)
        take = min(self.cap - self.fill, n)
        if take > 0:
            self.obs[self.fill:self.fill + take] = obs[:take]
            self.nxt[self.fill:self.fill + take] = nxt[:take]
            self.rew[self.fill:self.fill + take] = rew[:take]
            self.ret[self.fill:self.fill + take] = ret[:take]
            self.fill += take
        rest = n - take
        if rest > 0:
            keep = torch.rand(rest) < (self.cap / (self.seen + n))
            idx = keep.nonzero().flatten()
            if idx.numel() > 0:
                dst = torch.randint(0, self.cap, (idx.numel(),))
                self.obs[dst] = obs[take:][idx]
                self.nxt[dst] = nxt[take:][idx]
                self.rew[dst] = rew[take:][idx]
                self.ret[dst] = ret[take:][idx]
        self.seen += n

    def sample(self, k):
        idx = torch.randint(0, self.fill, (k,))
        return self.obs[idx], self.nxt[idx], self.rew[idx]

    def sample_ret(self, k):
        idx = torch.randint(0, self.fill, (k,))
        return self.obs[idx], self.ret[idx]


class Ladder:
    def __init__(self, obs_dim, n_act, gamma, cfg):
        self.gamma = gamma
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.vexp = mlp(obs_dim, 1)
        self.vdag1 = mlp(obs_dim, 1)
        self.vdag2 = mlp(obs_dim, 1)
        self.vdag1_t = mlp(obs_dim, 1)
        self.vdag2_t = mlp(obs_dim, 1)
        self.vdag1_t.load_state_dict(self.vdag1.state_dict())
        self.vdag2_t.load_state_dict(self.vdag2.state_dict())
        self.geo_sigma = mlp(obs_dim, 2 * obs_dim)
        self.geo_rew = mlp(obs_dim, 1)
        self.geo_v = mlp(obs_dim, 1)
        self.qdag = mlp(obs_dim, n_act) if cfg.get("qdag", False) else None
        self.res = Reservoir(cfg.get("reservoir", 100_000), obs_dim)
        params = [
            ("vexp", self.vexp), ("vdag1", self.vdag1), ("vdag2", self.vdag2),
            ("geo_sigma", self.geo_sigma), ("geo_rew", self.geo_rew), ("geo_v", self.geo_v),
        ]
        if self.qdag is not None:
            params.append(("qdag", self.qdag))
        self.opt = {k: torch.optim.Adam(m.parameters(), lr=cfg.get("lr", 3e-4)) for k, m in params}
        self.opt["geo_v"] = torch.optim.Adam(self.geo_v.parameters(), lr=cfg.get("geo_lr", 1e-3))
        self.iter = 0
        self.tau_dag = cfg.get("tau_dag", 0.75)
        self.tau_exp = cfg.get("tau_exp", 0.8)
        self.sigma_scale = cfg.get("sigma_scale", 0.5)
        self.clamp_lo, self.clamp_hi = cfg.get("dag_clamp", (-5.0, 50.0))

    # ---- inference -------------------------------------------------------
    @torch.no_grad()
    def vdag_min(self, obs):
        return torch.minimum(self.vdag1(obs).flatten(), self.vdag2(obs).flatten())

    @torch.no_grad()
    def vexp_val(self, obs):
        return self.vexp(obs).flatten()

    @torch.no_grad()
    def geo_val(self, obs):
        return self.geo_v(obs).flatten()

    @torch.no_grad()
    def qdag_val(self, obs):
        return self.qdag(obs)

    def geo_grad_dir(self, obs):
        """Sigma^2 * grad V_geo -- the field's preferred displacement direction."""
        o = obs.clone().requires_grad_(True)
        v = self.geo_v(o).sum()
        (g,) = torch.autograd.grad(v, o)
        with torch.no_grad():
            sig = torch.exp(self.geo_sigma(obs)[:, self.obs_dim:]).clamp(1e-3, 5.0)
        return g * sig ** 2 * self.sigma_scale ** 2

    # ---- training --------------------------------------------------------
    def update(self, obs, nxt, rew, done, act, returns, epochs=2, mb=4):
        """obs/nxt: (T*N, D); rew: extrinsic env reward (the ladder's currency);
        done: truncation flags; act: (T*N,) executed actions; returns: GAE returns
        (for V_exp). Boundary rows (done=1) are excluded from reservoir and from
        one-step targets' bootstrap via (1-d)."""
        g = self.gamma
        n = obs.shape[0]
        cont = 1.0 - done

        # reservoir feed: executed pairs only, never across a reset
        keep = done < 0.5
        self.res.add(obs[keep], nxt[keep], rew[keep], returns[keep])

        # V_exp on realized returns
        for _ in range(epochs):
            idx = torch.randperm(n)
            for c in idx.chunk(mb):
                loss = expectile_loss(self.vexp(obs[c]).flatten(), returns[c], self.tau_exp)
                self.opt["vexp"].zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.vexp.parameters(), 1.0)
                self.opt["vexp"].step()

        # V-dagger twins: y = r + g*(1-d)*min(tgt1, tgt2)(s')
        # With eps > 0 this is the EPSILON-INFLATED Bellman operator: the
        # bootstrap is the max of min-twin over the real next state and K
        # copies perturbed by eps * sigma_emp (the empirical marginal
        # displacement scale -- a global vector, nothing to extrapolate).
        # eps=0 recovers the plain composition critic exactly.
        eps = self.cfg.get("eps", 0.0)
        hull = self.cfg.get("hull", False)
        if hull and self.iter % 25 == 0:
            self._refresh_chart()
        with torch.no_grad():
            tv = torch.minimum(self.vdag1_t(nxt).flatten(), self.vdag2_t(nxt).flatten())
            if eps > 0 and hull and getattr(self, "chart", None) is not None:
                # HULL slack: perturb along eps-scaled WITNESSED displacement
                # vectors from nearest chart cells (asymmetries preserved --
                # no unwitnessed direction can be hallucinated)
                bo, bd = self._hull_bank
                qc = self.chart.P(nxt)
                sub = torch.randint(0, bo.shape[0], (min(1024, bo.shape[0]),))
                nnd, nnk = torch.cdist(qc, self._bank_c[sub]).topk(
                    self.cfg.get("eps_k", 4), largest=False)
                mu, ls = self.chart(nxt)
                for j in range(nnk.shape[1]):
                    delta = bd[sub[nnk[:, j]]]
                    # donor radius + likelihood gate (the adversarially-derived pair)
                    ok = (nnd[:, j] < self._hull_r0) \
                         & (((delta - mu) / ls.exp()).abs().max(1).values < 3.0)
                    delta = torch.where(ok.unsqueeze(1), delta, torch.zeros_like(delta))
                    pert = nxt + eps * delta
                    tv = torch.maximum(tv, torch.minimum(
                        self.vdag1_t(pert).flatten(), self.vdag2_t(pert).flatten()))
            elif eps > 0:
                sig = (nxt - obs).std(0)
                for _ in range(self.cfg.get("eps_k", 4)):
                    pert = nxt + eps * sig * torch.randn_like(nxt)
                    tv = torch.maximum(tv, torch.minimum(
                        self.vdag1_t(pert).flatten(), self.vdag2_t(pert).flatten()))
            y = (rew + g * cont * tv).clamp(self.clamp_lo, self.clamp_hi)
        for _ in range(epochs):
            idx = torch.randperm(n)
            for c in idx.chunk(mb):
                for k, m in (("vdag1", self.vdag1), ("vdag2", self.vdag2)):
                    loss = expectile_loss(m(obs[c]).flatten(), y[c], self.tau_dag)
                    self.opt[k].zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                    self.opt[k].step()

        # Q-dagger (optional): expectile on the executed action's cell
        if self.qdag is not None:
            for _ in range(epochs):
                idx = torch.randperm(n)
                for c in idx.chunk(mb):
                    q = self.qdag(obs[c]).gather(1, act[c].view(-1, 1)).flatten()
                    loss = expectile_loss(q, y[c], self.tau_dag)
                    self.opt["qdag"].zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(self.qdag.parameters(), 1.0)
                    self.opt["qdag"].step()

        # geo world-facing fits on the reservoir (stationary targets)
        stats = {}
        if self.res.fill > 2048:
            for _ in range(4):
                ro, rn, rr = self.res.sample(min(2048, self.res.fill))
                out = self.geo_sigma(ro)
                mu, logsig = out[:, :self.obs_dim], out[:, self.obs_dim:]
                logsig = logsig.clamp(-7, 2)
                d = rn - ro
                nll = (logsig + 0.5 * ((d - mu) / logsig.exp()) ** 2).mean()
                self.opt["geo_sigma"].zero_grad(); nll.backward()
                nn.utils.clip_grad_norm_(self.geo_sigma.parameters(), 1.0)
                self.opt["geo_sigma"].step()
                rl = ((self.geo_rew(rn).flatten() - rr) ** 2).mean()
                self.opt["geo_rew"].zero_grad(); rl.backward()
                nn.utils.clip_grad_norm_(self.geo_rew.parameters(), 1.0)
                self.opt["geo_rew"].step()
            stats["geo_nll"] = float(nll); stats["geo_rew_loss"] = float(rl)

            # HJB residual on current on-policy states, frozen world fits
            resid_acc = 0.0
            for _ in range(4):
                c = torch.randint(0, n, (min(2048, n),))
                o = obs[c].clone().requires_grad_(True)
                v = self.geo_v(o).flatten()
                (gr,) = torch.autograd.grad(v.sum(), o, create_graph=True)
                with torch.no_grad():
                    sig = torch.exp(self.geo_sigma(obs[c])[:, self.obs_dim:].clamp(-7, 2))
                    rhat = self.geo_rew(obs[c]).flatten()
                dual = torch.sqrt((gr ** 2 * (sig * self.sigma_scale) ** 2).sum(1) + 1e-8)
                resid = (1 - g) * v - rhat - g * dual
                loss = (resid ** 2).mean()
                self.opt["geo_v"].zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.geo_v.parameters(), 1.0)
                self.opt["geo_v"].step()
                resid_acc += float(loss)
            stats["hjb_resid"] = resid_acc / 4

        self.iter += 1
        if self.iter % 8 == 0:
            self.vdag1_t.load_state_dict(self.vdag1.state_dict())
            self.vdag2_t.load_state_dict(self.vdag2.state_dict())
        stats["vdag_mean"] = float(self.vdag_min(obs[:2048]).mean())
        return stats

    def _refresh_chart(self):
        """Online chart + hull bank from the reservoir (teleport-filtered)."""
        if self.res.fill < 4096:
            return
        from metric6 import ChartSigma, filter_teleports
        ro = self.res.obs[:self.res.fill]; rn = self.res.nxt[:self.res.fill]
        keep = filter_teleports(ro, rn)
        o, nx = ro[keep], rn[keep]
        if getattr(self, "chart", None) is None:
            self.chart = ChartSigma(self.obs_dim)
            self._chart_opt = torch.optim.Adam(self.chart.parameters(), lr=1e-3)
        for _ in range(300):
            idx = torch.randint(0, o.shape[0], (512,))
            mu, ls = self.chart(o[idx])
            d = nx[idx] - o[idx]
            nll = (ls + 0.5 * ((d - mu) / ls.exp()) ** 2).mean()
            loss = nll + 1e-3 * self.chart.P.weight.abs().mean()
            self._chart_opt.zero_grad(); loss.backward(); self._chart_opt.step()
        bidx = torch.randint(0, o.shape[0], (min(16384, o.shape[0]),))
        self._hull_bank = (o[bidx], (nx - o)[bidx])
        with torch.no_grad():
            self._bank_c = self.chart.P(self._hull_bank[0])
            samp = self._bank_c[torch.randint(0, self._bank_c.shape[0], (2048,))]
            dself = torch.cdist(samp, self._bank_c)
            dself.scatter_(1, dself.argmin(1, keepdim=True), 1e9)
            self._hull_r0 = 2.0 * dself.min(1).values.median()

    # ---- fields ----------------------------------------------------------
    @torch.no_grad()
    def fields(self, obs, v_real):
        """Returns (phi_mix, h_mix) exactly in the production shape:
        phi_mix = 0.5*unit(relu(affine(V_geo)-V)) + 0.5*unit(min V-dagger)   [potential]
        h_mix   = 0.5*unit(relu(geo-V)) + 0.5*unit(relu(Vdag-V))             [gate field]
        """
        eps = 1e-8
        vdag = self.vdag_min(obs)
        geo = self.geo_val(obs)
        gm, gs = geo.mean(), geo.std() + eps
        vm, vs = v_real.mean(), v_real.std() + eps
        geo_scaled = (geo - gm) / gs * vs + vm
        h_geo = torch.relu(geo_scaled - v_real)
        h_dag = torch.relu(vdag - v_real)
        unit = lambda x: x / (x.std() + eps)
        phi_mix = 0.5 * unit(h_geo) + 0.5 * unit(vdag)
        h_mix = 0.5 * unit(h_geo) + 0.5 * unit(h_dag)
        return phi_mix, h_mix
