"""Load a 5.3-lineage (residual, critic-trunk, four-rung-ladder) checkpoint into eager
PyTorch on CPU.

The 5.x resid architecture broke every assumption in load_checkpoint.py (512-wide
plain trunk): the dense nets are BroNet-style residual stacks (Models.cpp), there is a
second value-side trunk (CRITIC_TRUNK), four value heads (CRITIC / VDAG1 / VDAG2 /
GOAL_CRITIC), the V_exp expectile twin (GAP_EXP, a plain no-LayerNorm Sequential built
in Learner.cpp's GapState), and the three geometric-critic nets on raw obs
(GEO_SIGMA / GEO_REW / GEO_V, GEOMETRIC_CRITIC.md).

THE TRAP THIS FILE EXISTS TO NOT REPLAY (H2_TRUNCATION.md + the addResiduals
load-invisibility note in rlbot-run): torch::save archives carry ONLY parameterized
modules. Two structural facts are invisible in the archive and MUST come from config
knowledge:
  1. trailing activations - SHARED_HEAD and CRITIC_TRUNK are bodies ending
     [.., Linear, LN, LReLU]; the final LReLU has no params and must be appended.
  2. residual skip spans - Models.cpp records (blockStart, blockEnd) module-index
     spans at construction and Model::Forward adds the skip; nothing in the archive
     marks them. A naive nn.Sequential forward silently computes the WRONG function.

Construction is a line-for-line port of GGL::Model::Model (Models.cpp): layer i opens
a 2-layer residual block iff addResiduals && blockRemaining==0 && i>=1 && i+1<numLayers;
the block closes on its second layer's LayerNorm (span end), so Forward adds the skip
BEFORE the trailing activation (ResNet-v1: x <- Act(x + LN(W2 Act(LN(W1 x))))).

Per-model config mirrors src/ExampleMain.cpp @ 5.3 (2026-07-30 cold start) and
PPOLearner.cpp MakeModels / Learner.cpp GapState::Build. Structural asserts verify the
archive's param-index pattern matches the predicted module list exactly.

Wiring (PPOLearner.cpp):
  trunk h2   = SHARED_HEAD(obs)                       # 230 -> 1152, resid, ends in Act
  logits     = POLICY(h2)                             # 1152 -> 768x3 resid -> 90
  vt         = CRITIC_TRUNK(h2)                       # 1152 -> 1280x3 resid, ends in Act
  V_real     = CRITIC(vt);  V_dag1/2 = VDAG*(vt); V_goal = GOAL_CRITIC(vt)  # 1280x2 + out
  V_exp      = GAP_EXP(h2)                            # detached trunk read; no LN; 256x2
  V_geo      = GEO_V(obs); r_hat = GEO_REW(obs)       # raw obs, 384x2 + out, LN, no resid
  sigma      = exp(clamp(GEO_SIGMA(obs)[:, D:], -8, 2)) * geoSigmaScale(0.5); [:, :D] = drift mu
  H          = relu(min(V_dag1, V_dag2) - V_real)     # composition headroom (seek field)
  H_geo      = relu(V_geo - V_real)                   # geometric gap (pre-normalization form)
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]

GEO_SIGMA_SCALE = 0.5      # cfg.ppo.geoSigmaScale (ExampleMain default; PPOLearnerConfig.h)
TRAIN_GAMMA = 0.9969       # gaeGamma; the HJB residual uses this (PPOLearner.cpp, NOT 0.9994)


def _default_root() -> Path:
    if env := os.environ.get("PULSAR_CKPT_ROOT"):
        return Path(env)
    for cand in ["checkpoints_5.3-copy", "checkpoints_5.3"]:
        if (REPO / "build" / cand).is_dir():
            return REPO / "build" / cand
    raise RuntimeError("no 5.3 checkpoint root found; set PULSAR_CKPT_ROOT")


# name -> (layerSizes, addLayerNorm, addResiduals, addOutputLayer)
# input/output widths are read from the archive and cross-checked, not assumed.
_LADDER_ARCH = {
    "SHARED_HEAD":        ([1152, 1152, 1152], True,  True,  False),
    "POLICY":             ([768, 768, 768],    True,  True,  True),
    "CRITIC_TRUNK":       ([1280, 1280, 1280], True,  True,  False),
    "CRITIC":             ([1280, 1280],       True,  True,  True),   # resid no-op at depth 2
    "VDAG1":              ([1280, 1280],       True,  True,  True),
    "VDAG2":              ([1280, 1280],       True,  True,  True),
    "GOAL_CRITIC":        ([1280, 1280],       True,  True,  True),
    "GAP_EXP":            ([256, 256],         False, False, True),   # GapState::Build: no LN
    "GEO_SIGMA":          ([384, 384],         True,  False, True),
    "GEO_REW":            ([384, 384],         True,  False, True),
    "GEO_V":              ([384, 384],         True,  False, True),
    "REACH_PHI":          ([384, 384, 384],    True,  True,  True),
    "REACH_PSI_BALL":     ([384, 384, 384],    True,  True,  True),
    "REACH_PSI_CAR":      ([384, 384, 384],    True,  True,  True),
    "REACH_PSI_CARSTATE": ([384, 384, 384],    True,  True,  True),
}

FULL_MODEL_FILES = list(_LADDER_ARCH.keys())
POLICY_ONLY_FILES = ["SHARED_HEAD", "POLICY"]         # ref_*/version snapshots also carry GEO_*


def plan_modules(layer_sizes, add_ln, add_resid, add_out):
    """Port of GGL::Model::Model's construction loop. Returns (plan, spans) where plan is a
    list of ('linear'|'ln'|'act') module kinds in seq order and spans are residual
    (start, end) module indices - skip saved before module start, added after module end."""
    plan, spans = [], []
    n = len(layer_sizes)
    block_remaining, block_start = 0, -1
    for i in range(n):
        if add_resid and block_remaining == 0 and i >= 1 and (i + 1) < n:
            block_remaining = 2
            block_start = len(plan)
        plan.append("linear")
        if add_ln:
            plan.append("ln")
        if block_remaining > 0:
            block_remaining -= 1
            if block_remaining == 0:
                spans.append((block_start, len(plan) - 1))
        plan.append("act")
    if add_out:
        plan.append("linear")
    return plan, spans


class ResidualSeq(torch.nn.Module):
    """Flat module list + recorded residual spans, exactly like GGL::Model::Forward."""

    def __init__(self, modules, spans):
        super().__init__()
        self.seq = torch.nn.ModuleList(modules)
        self.spans = list(spans)

    def forward(self, x):
        saved = {}
        for i, mod in enumerate(self.seq):
            for s, (a, _) in enumerate(self.spans):
                if a == i:
                    saved[s] = x
            x = mod(x)
            for s, (_, b) in enumerate(self.spans):
                if b == i and s in saved:
                    x = x + saved[s]
        return x

    @property
    def in_features(self):
        return self.seq[0].in_features

    @property
    def out_features(self):
        for mod in reversed(self.seq):
            if isinstance(mod, torch.nn.Linear):
                return mod.out_features
        raise RuntimeError("no linear layer")


def rebuild_model(jit_module, name: str) -> ResidualSeq:
    layer_sizes, add_ln, add_resid, add_out = _LADDER_ARCH[name]
    plan, spans = plan_modules(layer_sizes, add_ln, add_resid, add_out)

    params = {}
    for pname, p in jit_module.named_parameters():
        idx, kind = pname.split(".")
        params.setdefault(int(idx), {})[kind] = p.detach().clone()

    # Structural verification: parameterized indices in the archive must sit exactly where
    # the plan puts Linears and LayerNorms, with matching kind (2-D vs 1-D weight).
    param_indices = sorted(params)
    expect_param_indices = [i for i, k in enumerate(plan) if k in ("linear", "ln")]
    if param_indices != expect_param_indices:
        raise RuntimeError(
            f"{name}: archive param indices {param_indices} != plan {expect_param_indices} "
            f"(plan={plan}) - config drift, DO NOT trust this rebuild")

    modules = []
    for i, kind in enumerate(plan):
        if kind == "act":
            modules.append(torch.nn.LeakyReLU())   # all 5.3 dense nets are LEAKY_RELU
            continue
        w, b = params[i]["weight"], params[i]["bias"]
        if kind == "linear":
            if w.dim() != 2:
                raise RuntimeError(f"{name}: module {i} expected Linear, got {tuple(w.shape)}")
            lin = torch.nn.Linear(w.shape[1], w.shape[0])
            lin.weight.data, lin.bias.data = w, b
            modules.append(lin)
        else:
            if w.dim() != 1:
                raise RuntimeError(f"{name}: module {i} expected LayerNorm, got {tuple(w.shape)}")
            ln = torch.nn.LayerNorm(w.shape[0])
            ln.weight.data, ln.bias.data = w, b
            modules.append(ln)

    # Width sanity vs the declared layerSizes.
    hidden_widths = [m.out_features for m in modules if isinstance(m, torch.nn.Linear)]
    body = hidden_widths[:-1] if add_out else hidden_widths
    if body != layer_sizes:
        raise RuntimeError(f"{name}: hidden widths {body} != config layerSizes {layer_sizes}")

    seq = ResidualSeq(modules, spans)
    seq.eval()
    for p in seq.parameters():
        p.requires_grad_(False)
    return seq


def list_checkpoints(root: Path):
    dirs = [d for d in root.iterdir() if d.is_dir() and d.name.isdigit()]
    return sorted(dirs, key=lambda d: int(d.name), reverse=True)


def copy_checkpoint(dest_root: Path | None = None, root: Path | None = None) -> Path:
    """Copy the newest complete checkpoint out of the (possibly live) folder."""
    root = root or _default_root()
    if dest_root is None:
        dest_root = Path(tempfile.mkdtemp(prefix="pulsar53_ckpt_"))
    dest_root.mkdir(parents=True, exist_ok=True)
    for ckpt in list_checkpoints(root):
        dest = dest_root / ckpt.name
        if dest.exists() and all((dest / f"{m}.lt").exists() for m in FULL_MODEL_FILES):
            return dest
        try:
            dest.mkdir(exist_ok=True)
            for m in FULL_MODEL_FILES:
                shutil.copy2(ckpt / f"{m}.lt", dest / f"{m}.lt")
            stats = ckpt / "RUNNING_STATS.json"
            if stats.exists():
                shutil.copy2(stats, dest / "RUNNING_STATS.json")
            return dest
        except (FileNotFoundError, OSError) as e:
            print(f"checkpoint {ckpt.name} vanished mid-copy ({e}), trying next")
            shutil.rmtree(dest, ignore_errors=True)
    raise RuntimeError(f"no complete checkpoint could be copied from {root}")


def load_models(ckpt_dir: Path, names=None) -> dict[str, ResidualSeq]:
    ckpt_dir = Path(ckpt_dir)
    if names is None:
        names = [n for n in FULL_MODEL_FILES if (ckpt_dir / f"{n}.lt").exists()]
    models = {}
    for name in names:
        jit_mod = torch.jit.load(str(ckpt_dir / f"{name}.lt"), map_location="cpu")
        models[name] = rebuild_model(jit_mod, name)
    return models


class Pulsar53Policy:
    """Frozen 5.3 policy + full critic ladder, with activation taps.

    Mirrors the old PulsarPolicy surface (trunk_forward / action_probs / sample_actions /
    phi_embedding / obs_size) so collect_dataset.py-style harnesses can drive it, and adds
    the ladder reads: v_real / v_exp / vdag / v_goal / headroom / geo.
    """

    ACTION_MIN_PROB = 1e-11
    ACTION_DISABLED_LOGIT = -1e10
    NUM_ACTIONS = 90

    def __init__(self, models: dict):
        self.models = models
        self.trunk = models["SHARED_HEAD"]
        self.policy = models["POLICY"]
        self.obs_size = self.trunk.in_features        # 230
        self.h2_width = self.trunk.out_features       # 1152 (steer_team rollout_team reads this)
        self.wire_pad = 0                             # no Ladder wire in 5.x
        # h1 tap: after the stem's activation (module idx 2: Lin, LN, Act). The stem is
        # span-free (spans start at module 3), so a plain slice is exact for block 1.
        self._trunk_stem = torch.nn.Sequential(*self.trunk.seq[:3])
        self._trunk_rest = ResidualSeq(list(self.trunk.seq[3:]),
                                       [(a - 3, b - 3) for a, b in self.trunk.spans])
        self.phi = models.get("REACH_PHI")

    # ---- policy surface (collect_dataset parity) ----
    @torch.no_grad()
    def trunk_forward(self, obs):
        h1 = self._trunk_stem(obs)
        h2 = self._trunk_rest(h1)
        return h1, h2

    @torch.no_grad()
    def action_probs(self, trunk_out, action_masks):
        logits = self.policy(trunk_out)
        logits = logits + self.ACTION_DISABLED_LOGIT * (~action_masks.bool()).float()
        return torch.softmax(logits, -1).clamp(self.ACTION_MIN_PROB, 1)

    @torch.no_grad()
    def sample_actions(self, trunk_out, action_masks):
        probs = self.action_probs(trunk_out, action_masks)
        return torch.multinomial(probs, 1, True).flatten()

    @torch.no_grad()
    def phi_embedding(self, trunk_out, actions):
        onehot = torch.nn.functional.one_hot(actions.long(), self.NUM_ACTIONS).float()
        e = self.phi(torch.cat([trunk_out, onehot], -1))
        return e / e.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    # ---- the ladder ----
    @torch.no_grad()
    def value_trunk(self, obs, h2=None):
        if h2 is None:
            _, h2 = self.trunk_forward(obs)
        return self.models["CRITIC_TRUNK"](h2)

    @torch.no_grad()
    def ladder(self, obs):
        """All value reads for a batch of raw obs. Returns dict of 1-D tensors."""
        _, h2 = self.trunk_forward(obs)
        vt = self.models["CRITIC_TRUNK"](h2)
        out = {
            "v_real": self.models["CRITIC"](vt).flatten(),
            "v_exp": self.models["GAP_EXP"](h2).flatten(),
            "vdag1": self.models["VDAG1"](vt).flatten(),
            "vdag2": self.models["VDAG2"](vt).flatten(),
            "v_goal": self.models["GOAL_CRITIC"](vt).flatten(),
            "v_geo": self.models["GEO_V"](obs).flatten(),
            "r_hat": self.models["GEO_REW"](obs).flatten(),
        }
        out["vdag_min"] = torch.minimum(out["vdag1"], out["vdag2"])
        out["headroom"] = torch.relu(out["vdag_min"] - out["v_real"])
        # LIVE PARITY (Learner.cpp geo actuation): V_geo is affine-matched to V_real's
        # batch mean/std BEFORE differencing. The raw-unit gap relu(V_geo - V_real) is
        # ~37 where the live panel reads ~0.2 - always use this form for comparisons.
        g, v = out["v_geo"], out["v_real"]
        geo_scaled = (g - g.mean()) / (g.std() + 1e-8) * v.std() + v.mean()
        out["h_geo"] = torch.relu(geo_scaled - v)
        return out

    @torch.no_grad()
    def geo_sigma(self, obs):
        """(mu, sigma): drift and the scaled per-dim displacement spread (PPOLearner units)."""
        so = self.models["GEO_SIGMA"](obs)
        d = obs.shape[1]
        mu = so[:, :d]
        sigma = so[:, d:2 * d].clamp(-8.0, 2.0).exp() * GEO_SIGMA_SCALE
        return mu, sigma

    def hjb_residual(self, obs):
        """Per-row HJB residual (Eq. 4 of GEOMETRIC_CRITIC.md), grad enabled internally."""
        gin = obs.detach().clone().requires_grad_(True)
        vg = self.models["GEO_V"](gin).flatten()
        gx = torch.autograd.grad(vg.sum(), gin, create_graph=False)[0]
        with torch.no_grad():
            _, sig = self.geo_sigma(obs)
            rh = self.models["GEO_REW"](obs).flatten()
            gnorm = (gx.pow(2) * sig.pow(2)).sum(-1).clamp_min(1e-12).sqrt()
            return (1.0 - TRAIN_GAMMA) * vg.detach() - rh - TRAIN_GAMMA * gnorm


def load_dir(ckpt_dir) -> Pulsar53Policy:
    return Pulsar53Policy(load_models(Path(ckpt_dir)))


def load_latest(cache_root: Path | None = None):
    if cache_root is None:
        cache_root = Path(__file__).resolve().parents[1] / "data" / "ckpt_cache_53"
    ckpt_dir = copy_checkpoint(cache_root)
    return load_dir(ckpt_dir), ckpt_dir


if __name__ == "__main__":
    torch.manual_seed(0)
    root = Path(os.environ.get("PULSAR_CKPT_ROOT", REPO / "research" / "data" / "ckpt53"))
    ckpts = list_checkpoints(root)
    ckpt = ckpts[0]
    print(f"checkpoint: {ckpt.name} ({int(ckpt.name):,} ts)")
    models = load_models(ckpt)
    for name, seq in models.items():
        n_params = sum(p.numel() for p in seq.parameters())
        lins = [m for m in seq.seq if isinstance(m, torch.nn.Linear)]
        arch = f"{lins[0].in_features} -> " + " -> ".join(str(l.out_features) for l in lins)
        print(f"  {name:20s} {arch:52s} {n_params:>10,} params  spans={seq.spans}")
    pol = Pulsar53Policy(models)
    obs = torch.randn(8, pol.obs_size)
    h1, h2 = pol.trunk_forward(obs)
    mask = torch.ones(8, 90, dtype=torch.uint8)
    probs = pol.action_probs(h2, mask)
    lad = pol.ladder(obs)
    mu, sig = pol.geo_sigma(obs)
    res = pol.hjb_residual(obs)
    print(f"\nsmoke: h1 {tuple(h1.shape)} h2 {tuple(h2.shape)} probs {tuple(probs.shape)} "
          f"(sum {probs.sum(-1).mean():.4f})")
    for k, v in lad.items():
        print(f"  {k:9s} mean {v.mean():+.4f}  std {v.std():.4f}")
    print(f"  sigma mean {sig.mean():.4f}  hjb resid mean^2 {res.pow(2).mean():.6f}")
