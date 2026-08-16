"""Load a 7.0-lineage (full-size ts8, composite-value) checkpoint into eager CPU torch.

Same construction machinery as load_checkpoint_53 (ResidualSeq port of GGL::Model with
the two archive-invisible facts: trailing activations and residual spans) with the 7.0
arch table: trunk 1280x3, policy 768x3, critic_trunk 1536x3, value heads 1536x2 (twin
CRITIC/CRITIC2 under valueTwinEnabled - live readout is their MEAN), reach nets 384x3.
No GEO_* and no HULL_* in this lineage's snapshots (hull_* excluded from GetPolicyModels;
geo trio retired). Sizes mirror src/ExampleMain.cpp @ 7.0b / the AiMOS fleet
(wandb 7.0b-aimos); the structural asserts in rebuild_model catch any drift.

Verified against build/checkpoints_aimos/<ts> snapshots (rsync mirror of the fleet's
~/scratch-shared/checkpoints_scale_luca).
"""

from pathlib import Path

import torch

import load_checkpoint_53 as l53
from load_checkpoint_53 import ResidualSeq, rebuild_model  # noqa: F401 (re-export)

# name -> (layerSizes, addLayerNorm, addResiduals, addOutputLayer); widths cross-checked
# against the archive by rebuild_model.
ARCH_70 = {
    "SHARED_HEAD":        ([1280, 1280, 1280], True,  True,  False),
    "POLICY":             ([768, 768, 768],    True,  True,  True),
    "CRITIC_TRUNK":       ([1536, 1536, 1536], True,  True,  False),
    "CRITIC":             ([1536, 1536],       True,  True,  True),   # resid no-op at depth 2
    "CRITIC2":            ([1536, 1536],       True,  True,  True),   # twin (valueTwinEnabled)
    "VDAG1":              ([1536, 1536],       True,  True,  True),
    "VDAG2":              ([1536, 1536],       True,  True,  True),
    "GOAL_CRITIC":        ([1536, 1536],       True,  True,  True),
    "GAP_EXP":            ([256, 256],         False, False, True),   # GapState::Build: no LN
    "REACH_PHI":          ([384, 384, 384],    True,  True,  True),
    "REACH_PSI_BALL":     ([384, 384, 384],    True,  True,  True),
    "REACH_PSI_CAR":      ([384, 384, 384],    True,  True,  True),
    "REACH_PSI_CARSTATE": ([384, 384, 384],    True,  True,  True),
}


def load_models(ckpt_dir: Path, names=None) -> dict[str, ResidualSeq]:
    ckpt_dir = Path(ckpt_dir)
    if names is None:
        names = [n for n in ARCH_70 if (ckpt_dir / f"{n}.lt").exists()]
    # rebuild_model reads the arch table via module-global lookup in l53; swap it in.
    saved = l53._LADDER_ARCH
    l53._LADDER_ARCH = ARCH_70
    try:
        models = {}
        for name in names:
            jit_mod = torch.jit.load(str(ckpt_dir / f"{name}.lt"), map_location="cpu")
            models[name] = rebuild_model(jit_mod, name)
    finally:
        l53._LADDER_ARCH = saved
    return models


class Pulsar70Policy(l53.Pulsar53Policy):
    """Frozen 7.0 policy + composite-value reads. Policy surface identical to 5.3's
    (trunk_forward / action_probs / sample_actions / phi_embedding); ladder() replaced
    by the 7.0 head set (twin critics, no geo)."""

    @torch.no_grad()
    def ladder(self, obs):
        _, h2 = self.trunk_forward(obs)
        vt = self.models["CRITIC_TRUNK"](h2)
        v1 = self.models["CRITIC"](vt).flatten()
        out = {"v_real": v1}
        if "CRITIC2" in self.models:
            v2 = self.models["CRITIC2"](vt).flatten()
            out["v_real"] = (v1 + v2) / 2          # live readout: mean of the twin heads
            out["twin_disagree"] = (v1 - v2).abs()
        for key, name in [("v_exp", "GAP_EXP"), ("v_goal", "GOAL_CRITIC")]:
            if name in self.models:
                src = h2 if name == "GAP_EXP" else vt
                out[key] = self.models[name](src).flatten()
        if "VDAG1" in self.models and "VDAG2" in self.models:
            out["vdag1"] = self.models["VDAG1"](vt).flatten()
            out["vdag2"] = self.models["VDAG2"](vt).flatten()
            out["vdag_min"] = torch.minimum(out["vdag1"], out["vdag2"])
            out["headroom"] = torch.relu(out["vdag_min"] - out["v_real"])
        return out


def load_dir(ckpt_dir) -> Pulsar70Policy:
    return Pulsar70Policy(load_models(Path(ckpt_dir)))


if __name__ == "__main__":
    import os
    import sys
    torch.manual_seed(0)
    root = Path(os.environ.get("PULSAR_CKPT_ROOT",
                               l53.REPO / "build" / "checkpoints_aimos"))
    ckpt = Path(sys.argv[1]) if len(sys.argv) > 1 else l53.list_checkpoints(root)[0]
    print(f"checkpoint: {ckpt}")
    models = load_models(ckpt)
    for name, seq in models.items():
        lins = [m for m in seq.seq if isinstance(m, torch.nn.Linear)]
        arch = f"{lins[0].in_features} -> " + " -> ".join(str(l.out_features) for l in lins)
        n_params = sum(p.numel() for p in seq.parameters())
        print(f"  {name:20s} {arch:52s} {n_params:>12,} params  spans={seq.spans}")
    pol = Pulsar70Policy(models)
    obs = torch.randn(8, pol.obs_size)
    h1, h2 = pol.trunk_forward(obs)
    mask = torch.ones(8, 90, dtype=torch.uint8)
    probs = pol.action_probs(h2, mask)
    lad = pol.ladder(obs)
    print(f"\nsmoke: obs {pol.obs_size}  h2 {tuple(h2.shape)}  probs sum "
          f"{probs.sum(-1).mean():.4f}")
    for k, v in lad.items():
        print(f"  {k:13s} mean {v.mean():+.4f}  std {v.std():.4f}")
    if pol.phi is not None:
        e = pol.phi_embedding(h2, torch.zeros(8, dtype=torch.long))
        print(f"  phi emb {tuple(e.shape)}  norm {e.norm(dim=-1).mean():.4f}")
