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
  shared_head: obs(OBS) -> [Linear 512, LN, LeakyReLU] x2            (trunk)
  policy:      512 -> [Linear 512, LN, LeakyReLU] x3 -> Linear 90    (DefaultAction table)
  critic:      512 -> [Linear 512, LN, LeakyReLU] x3 -> Linear 1
  reach_phi:   trunk(512)+onehot(90)=602 -> [Linear 256, LN, LeakyReLU] x2 -> Linear 128
  reach_psi_*: 6 -> [Linear 256, LN, LeakyReLU] x2 -> Linear 128
OBS is lineage-dependent and AUTO-DETECTED from SHARED_HEAD's first Linear:
  109 = 3.1 lineage (AdvancedObs, 1v1) | 230 = 4.0 lineage (AdvancedObsPadded(3),
  51 header + 6x29 player slots + 5 presence flags - ball@0 / self@51 offsets are
  IDENTICAL to AdvancedObs, so landing sims and spatial readers work unchanged).
Obs are NOT normalized (LearnerConfig.standardizeObs defaults to false and
ExampleMain never sets it), so raw AdvancedObs floats feed the trunk directly.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
# Root preference: $PULSAR_CKPT_ROOT > the LIVE 5.0v3 lineage (checkpoints_5.0v3,
# the RocketSim-v3 cold start; -copy = an scp'd offline copy) > the frozen pre-v3
# 5.0 lineage > the 4.0 offline copy > the 3.1 live folder (older setups).
# NOTE: checkpoints_5.0v3 MUST lead - the v3 engine switch forked a fresh lineage,
# and reading checkpoints_5.0 by default silently analyzed the frozen <=11.75B run
# (the "every reading is stale" trap). Override with PULSAR_CKPT_ROOT for a specific
# copied-out checkpoint (e.g. a golden or branch backup).
def _default_root() -> Path:
    if env := os.environ.get("PULSAR_CKPT_ROOT"):
        return Path(env)
    for cand in ["checkpoints_5.0v3-copy", "checkpoints_5.0v3",
                 "checkpoints_5.0-copy", "checkpoints_5.0",
                 "checkpoints_4.0-copy", "checkpoints_4.0", "checkpoints_3.1"]:
        if (REPO / "build" / cand).is_dir():
            return REPO / "build" / cand
    return REPO / "build" / "checkpoints_3.1"

CHECKPOINT_ROOT = _default_root()

MODEL_FILES = ["SHARED_HEAD", "POLICY", "CRITIC", "REACH_PHI", "REACH_PSI_BALL", "REACH_PSI_CAR"]

def expected_shapes(obs_size: int, policy_in: int = 512) -> dict:
    # policy_in = 512 (pre-Ladder) or 517 (+5 self-conditioning wire columns, live 5.0v3).
    # Only the policy head's FIRST Linear widens; the trunk and critic are untouched.
    return {
        "SHARED_HEAD": [(512, obs_size), (512,), (512,), (512,), (512, 512), (512,), (512,), (512,)],
        "POLICY": [(512, policy_in), (512,), (512,), (512,)]
                  + [(512, 512), (512,), (512,), (512,)] * 2 + [(90, 512), (90,)],
        "CRITIC": [(512, 512), (512,), (512,), (512,)] * 3 + [(1, 512), (1,)],
        "REACH_PHI": [(256, 602), (256,), (256,), (256,), (256, 256), (256,), (256,), (256,), (128, 256), (128,)],
        "REACH_PSI_BALL": [(256, 6), (256,), (256,), (256,), (256, 256), (256,), (256,), (256,), (128, 256), (128,)],
        "REACH_PSI_CAR": [(256, 6), (256,), (256,), (256,), (256, 256), (256,), (256,), (256,), (128, 256), (128,)],
    }


def list_checkpoints(root: Path = None):
    """Numeric checkpoint dirs, newest (highest timestep) first."""
    root = root or CHECKPOINT_ROOT
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


def rebuild_sequential(jit_module, trailing_activations: int = 0) -> torch.nn.Sequential:
    """Reconstruct the C++ torch::nn::Sequential as an eager nn.Sequential.

    Param names are '<idx>.weight' / '<idx>.bias' where idx is the position in the
    C++ Sequential; 2-D weight => Linear, 1-D => LayerNorm; index gaps are the
    parameterless LeakyReLU modules.

    trailing_activations: parameterless modules AFTER the last parameterized index
    are invisible in the archive (nothing marks them), so the caller must say how
    many to append. THE 2026-07-19 BUG: SHARED_HEAD ends [.., Linear, LN, LReLU] -
    this function silently dropped that final LReLU, so every consumer of "h2" got
    the PRE-ACTIVATION trunk output. Heads with internal LayerNorms (policy/critic)
    largely absorbed it - the un-normalized GAP_EXP head exposed it (V_exp read -8
    where the live trainer read V_real+0.15). Verified against the C++ loader via
    a GGL_SMOKE CPU oracle on the same checkpoint. Output heads ending in a bare
    Linear need trailing_activations=0 (the default).
    """
    params = dict(jit_module.named_parameters())
    by_idx = {}
    for name, p in params.items():
        idx, kind = name.split(".")
        by_idx.setdefault(int(idx), {})[kind] = p.detach().clone()

    layers = []
    for i in range(max(by_idx) + 1 + trailing_activations):
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
    SHARED_HEAD + POLICY). The lineage's obs width is auto-detected from
    SHARED_HEAD's first Linear (109 = 3.1 AdvancedObs, 230 = 4.0 AdvancedObsPadded)."""
    names = list(names or MODEL_FILES)
    # Detect obs width first so every shape check uses the right lineage
    head = torch.jit.load(str(ckpt_dir / "SHARED_HEAD.lt"), map_location="cpu")
    obs_size = dict(head.named_parameters())["0.weight"].shape[1]
    # Policy-head input width: 512 (pre-Ladder) or 517 (+5 Ladder wire columns, live
    # 5.0v3). Auto-detect from POLICY.lt so both widths load; the offline harness feeds
    # the 5 wire values as ZEROS (eval/render/boot parity - Rating scores the wire-zeroed
    # policy, so probes and head-to-heads match how the trainer measures it).
    policy_in = 512
    pol_path = ckpt_dir / "POLICY.lt"
    if "POLICY" in names and pol_path.exists():
        pol = torch.jit.load(str(pol_path), map_location="cpu")
        policy_in = dict(pol.named_parameters())["0.weight"].shape[1]
    shapes_for = expected_shapes(obs_size, policy_in)

    models = {}
    for name in names:
        jit_mod = torch.jit.load(str(ckpt_dir / f"{name}.lt"), map_location="cpu")

        shapes = [tuple(p.shape) for _, p in jit_mod.named_parameters()]
        if shapes != shapes_for[name]:
            raise RuntimeError(f"{name}: shape mismatch\n got      {shapes}\n expected {shapes_for[name]}")

        # The archive stores no scripted forward and no module-type metadata (generic
        # mangled '__torch__.Module' children), so the rebuild is verified structurally:
        # 2-D params are Linear, 1-D are LayerNorm (both unambiguous from shape), and the
        # param-less gap indices must sit exactly where the [Linear, LN, Act] pattern puts
        # activations. Functional verification is the passthrough probe in train_probes.py.
        param_indices = sorted({int(n.split(".")[0]) for n, _ in jit_mod.named_parameters()})
        gaps = [i for i in range(max(param_indices) + 1) if i not in param_indices]
        if any((g + 1) % 3 for g in gaps):
            raise RuntimeError(f"{name}: activation gaps at unexpected indices {gaps}")

        # SHARED_HEAD is the one archive that ENDS in an activation ([Lin,LN,LReLU]x2,
        # no output layer) - the trailing LReLU is invisible to the gap scan and must
        # be appended explicitly or h2 comes out pre-activation (the 2026-07-19 bug).
        # Every head (policy/critic/reach/gap) ends in a bare output Linear: 0.
        models[name] = rebuild_sequential(jit_mod,
                                          trailing_activations=1 if name == "SHARED_HEAD" else 0)
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
        self.obs_size = self.trunk[0].in_features  # 109 = 3.1, 230 = 4.0/5.0 padded
        # Policy-head input: 512 (pre-Ladder) or 517 (+5 Ladder self-conditioning wire).
        # We feed the wire as zeros (eval/render parity), so trunk_out is right-padded
        # by wire_pad before the policy forward. Trunk taps + phi are wire-independent.
        self.policy_in = self.policy[0].in_features
        self.wire_pad = self.policy_in - 512
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
        # Ladder wire [V_real,V_exp,gap_KD,V_metric,gap_PK] is not reconstructable
        # offline (banks unpersisted) - fed as zeros exactly as eval/render/boot do.
        # This is the ONLY policy-head forward, so every caller (sample_actions,
        # SteeredPolicy.act, compare_checkpoints) inherits the padding.
        if self.wire_pad:
            trunk_out = torch.cat(
                [trunk_out, trunk_out.new_zeros(trunk_out.shape[0], self.wire_pad)], -1)
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
    obs = torch.randn(8, pol.obs_size)
    h1, h2 = pol.trunk_forward(obs)
    mask = torch.ones(8, 90, dtype=torch.uint8)
    probs = pol.action_probs(h2, mask)
    acts = pol.sample_actions(h2, mask)
    emb = pol.phi_embedding(h2, acts)
    print(f"\nsmoke: h1 {tuple(h1.shape)}, h2 {tuple(h2.shape)}, probs {tuple(probs.shape)} "
          f"(sum {probs.sum(-1).mean():.4f}), phi {tuple(emb.shape)} (norm {emb.norm(dim=-1).mean():.4f})")
