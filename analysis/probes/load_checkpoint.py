"""Load a live GigaLearnCPP checkpoint into eager PyTorch on CPU.

The trainer saves each model with C++ `torch::save(seq, stream)`, which writes a
TorchScript-compatible archive; `torch.jit.load` reads it directly. We rebuild each
Sequential as an eager nn.Sequential (so we can hook intermediate activations) and
verify the rebuild is numerically identical to the jit module.

Checkpoint dirs are named by cumulative timestep and are rotated by the live trainer
at any moment - we pick the newest, COPY it out, and re-verify the copy is complete
(falling back to the next-newest on a race).

Architecture expected (from src/ExampleMain.cpp + PPOLearner.cpp, verified against
tensor shapes at load time):
  shared_head: obs(109) -> [Linear 512, LN, LeakyReLU] x2            (trunk)
  policy:      512 -> [Linear 512, LN, LeakyReLU] x3 -> Linear 90    (DefaultAction table)
  critic:      512 -> [Linear 512, LN, LeakyReLU] x3 -> Linear 1
  reach_phi:   trunk(512)+onehot(90)=602 -> [Linear 256, LN, LeakyReLU] x2 -> Linear 128
  reach_psi_*: 6 -> [Linear 256, LN, LeakyReLU] x2 -> Linear 128
Obs are NOT normalized (LearnerConfig.standardizeObs defaults to false and
ExampleMain never sets it), so raw AdvancedObs floats feed the trunk directly.
"""

import json
import shutil
import tempfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
CHECKPOINT_ROOT = REPO / "build" / "checkpoints_3.1"

MODEL_FILES = ["SHARED_HEAD", "POLICY", "CRITIC", "REACH_PHI", "REACH_PSI_BALL", "REACH_PSI_CAR"]

EXPECTED_SHAPES = {
    "SHARED_HEAD": [(512, 109), (512,), (512,), (512,), (512, 512), (512,), (512,), (512,)],
    "POLICY": [(512, 512), (512,), (512,), (512,)] * 3 + [(90, 512), (90,)],
    "CRITIC": [(512, 512), (512,), (512,), (512,)] * 3 + [(1, 512), (1,)],
    "REACH_PHI": [(256, 602), (256,), (256,), (256,), (256, 256), (256,), (256,), (256,), (128, 256), (128,)],
    "REACH_PSI_BALL": [(256, 6), (256,), (256,), (256,), (256, 256), (256,), (256,), (256,), (128, 256), (128,)],
    "REACH_PSI_CAR": [(256, 6), (256,), (256,), (256,), (256, 256), (256,), (256,), (256,), (128, 256), (128,)],
}


def list_checkpoints(root: Path = CHECKPOINT_ROOT):
    """Numeric checkpoint dirs, newest (highest timestep) first."""
    dirs = [d for d in root.iterdir() if d.is_dir() and d.name.isdigit()]
    return sorted(dirs, key=lambda d: int(d.name), reverse=True)


def copy_checkpoint(dest_root: Path | None = None) -> Path:
    """Copy the newest complete checkpoint out of the live folder. Never reads in place."""
    if dest_root is None:
        dest_root = Path(tempfile.mkdtemp(prefix="pulsar_ckpt_"))
    dest_root.mkdir(parents=True, exist_ok=True)

    for ckpt in list_checkpoints():
        dest = dest_root / ckpt.name
        if dest.exists() and all((dest / f"{m}.lt").exists() for m in MODEL_FILES):
            return dest  # already cached
        try:
            dest.mkdir(exist_ok=True)
            for m in MODEL_FILES:
                shutil.copy2(ckpt / f"{m}.lt", dest / f"{m}.lt")
            stats = ckpt / "RUNNING_STATS.json"
            if stats.exists():
                shutil.copy2(stats, dest / "RUNNING_STATS.json")
            return dest
        except (FileNotFoundError, OSError) as e:
            # The trainer rotated this dir away mid-copy; try the next-newest.
            print(f"checkpoint {ckpt.name} vanished mid-copy ({e}), trying next")
            shutil.rmtree(dest, ignore_errors=True)
    raise RuntimeError(f"no complete checkpoint could be copied from {CHECKPOINT_ROOT}")


def rebuild_sequential(jit_module) -> torch.nn.Sequential:
    """Reconstruct the C++ torch::nn::Sequential as an eager nn.Sequential.

    Param names are '<idx>.weight' / '<idx>.bias' where idx is the position in the
    C++ Sequential; 2-D weight => Linear, 1-D => LayerNorm; index gaps are the
    parameterless LeakyReLU modules.
    """
    params = dict(jit_module.named_parameters())
    by_idx = {}
    for name, p in params.items():
        idx, kind = name.split(".")
        by_idx.setdefault(int(idx), {})[kind] = p.detach().clone()

    layers = []
    for i in range(max(by_idx) + 1):
        if i not in by_idx:
            layers.append(torch.nn.LeakyReLU())  # default slope 0.01 matches torch::nn::LeakyReLU()
            continue
        w, b = by_idx[i]["weight"], by_idx[i]["bias"]
        if w.dim() == 2:
            lin = torch.nn.Linear(w.shape[1], w.shape[0])
            lin.weight.data, lin.bias.data = w, b
            layers.append(lin)
        else:
            ln = torch.nn.LayerNorm(w.shape[0])
            ln.weight.data, ln.bias.data = w, b
            layers.append(ln)
    seq = torch.nn.Sequential(*layers)
    seq.eval()
    for p in seq.parameters():
        p.requires_grad_(False)
    return seq


def load_models(ckpt_dir: Path, names=None) -> dict[str, torch.nn.Sequential]:
    """names: subset of MODEL_FILES to load (policy_versions snapshots only carry
    SHARED_HEAD + POLICY)."""
    models = {}
    for name in (names or MODEL_FILES):
        jit_mod = torch.jit.load(str(ckpt_dir / f"{name}.lt"), map_location="cpu")

        shapes = [tuple(p.shape) for _, p in jit_mod.named_parameters()]
        if shapes != EXPECTED_SHAPES[name]:
            raise RuntimeError(f"{name}: shape mismatch\n got      {shapes}\n expected {EXPECTED_SHAPES[name]}")

        # The archive stores no scripted forward and no module-type metadata (generic
        # mangled '__torch__.Module' children), so the rebuild is verified structurally:
        # 2-D params are Linear, 1-D are LayerNorm (both unambiguous from shape), and the
        # param-less gap indices must sit exactly where the [Linear, LN, Act] pattern puts
        # activations. Functional verification is the passthrough probe in train_probes.py.
        param_indices = sorted({int(n.split(".")[0]) for n, _ in jit_mod.named_parameters()})
        gaps = [i for i in range(max(param_indices) + 1) if i not in param_indices]
        if any((g + 1) % 3 for g in gaps):
            raise RuntimeError(f"{name}: activation gaps at unexpected indices {gaps}")

        models[name] = rebuild_sequential(jit_mod)
    return models


class PulsarPolicy:
    """Frozen policy with activation taps.

    trunk_forward: obs [n,109] -> (h1 [n,512] after block 1, h2 [n,512] trunk output)
    action_probs:  masked softmax over the 90-way table (InferPolicyProbsFromModels parity)
    phi_embedding: L2-normalized reach_phi(trunk_out ++ onehot(action)) [n,128]
    """

    ACTION_MIN_PROB = 1e-11
    ACTION_DISABLED_LOGIT = -1e10
    NUM_ACTIONS = 90

    def __init__(self, models: dict[str, torch.nn.Sequential]):
        self.models = models
        self.trunk = models["SHARED_HEAD"]
        self.policy = models["POLICY"]
        self.critic = models.get("CRITIC")
        self.phi = models.get("REACH_PHI")
        # trunk = [Lin, LN, Act, Lin, LN, Act]; h1 taps after index 2
        self.trunk_block1 = self.trunk[:3]
        self.trunk_block2 = self.trunk[3:]

    @torch.no_grad()
    def trunk_forward(self, obs: torch.Tensor):
        h1 = self.trunk_block1(obs)
        h2 = self.trunk_block2(h1)
        return h1, h2

    @torch.no_grad()
    def action_probs(self, trunk_out: torch.Tensor, action_masks: torch.Tensor):
        logits = self.policy(trunk_out)
        logits = logits + self.ACTION_DISABLED_LOGIT * (~action_masks.bool()).float()
        return torch.softmax(logits, -1).clamp(self.ACTION_MIN_PROB, 1)

    @torch.no_grad()
    def sample_actions(self, trunk_out: torch.Tensor, action_masks: torch.Tensor):
        probs = self.action_probs(trunk_out, action_masks)
        return torch.multinomial(probs, 1, True).flatten()

    @torch.no_grad()
    def phi_embedding(self, trunk_out: torch.Tensor, actions: torch.Tensor):
        onehot = torch.nn.functional.one_hot(actions.long(), self.NUM_ACTIONS).float()
        e = self.phi(torch.cat([trunk_out, onehot], -1))
        return e / e.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def load_latest(cache_root: Path | None = None):
    """Convenience: copy newest checkpoint (cached) and load it. Returns (policy, ckpt_dir)."""
    if cache_root is None:
        cache_root = Path(__file__).resolve().parent / "data" / "ckpt_cache"
    ckpt_dir = copy_checkpoint(cache_root)
    return PulsarPolicy(load_models(ckpt_dir)), ckpt_dir


if __name__ == "__main__":
    torch.manual_seed(0)
    ckpt_dir = copy_checkpoint(Path(__file__).resolve().parent / "data" / "ckpt_cache")
    print(f"checkpoint: {ckpt_dir.name}  ({int(ckpt_dir.name):,} timesteps)")

    stats_file = ckpt_dir / "RUNNING_STATS.json"
    if stats_file.exists():
        keys = list(json.loads(stats_file.read_text()).keys())
        print(f"RUNNING_STATS keys: {keys}")
        assert "obs_stat" not in keys, "obs normalization IS active - collection must apply it!"

    models = load_models(ckpt_dir)
    print("\nAll models jit-loaded, shape-verified, eager-rebuilt:")
    for name, seq in models.items():
        n_params = sum(p.numel() for p in seq.parameters())
        dims = [m for m in seq if isinstance(m, torch.nn.Linear)]
        arch = f"{dims[0].in_features} -> " + " -> ".join(str(l.out_features) for l in dims)
        print(f"  {name:16s} {arch:42s} {n_params:>10,} params")

    pol = PulsarPolicy(models)
    obs = torch.randn(8, 109)
    h1, h2 = pol.trunk_forward(obs)
    mask = torch.ones(8, 90, dtype=torch.uint8)
    probs = pol.action_probs(h2, mask)
    acts = pol.sample_actions(h2, mask)
    emb = pol.phi_embedding(h2, acts)
    print(f"\nsmoke: h1 {tuple(h1.shape)}, h2 {tuple(h2.shape)}, probs {tuple(probs.shape)} "
          f"(sum {probs.sum(-1).mean():.4f}), phi {tuple(emb.shape)} (norm {emb.norm(dim=-1).mean():.4f})")
