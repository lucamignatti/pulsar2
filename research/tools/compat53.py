"""Import-time shim: makes the legacy analysis battery (steer_team consumers -
mechanic_census, aerial_gap, breakeven_probe, fear_decomp, ...) run against the 5.3
residual checkpoints WITHOUT touching the historical scripts.

Usage: importing this module FIRST (before any legacy tool module) replaces the
'load_checkpoint' module in sys.modules with a 5.3-flavored facade:
  PulsarPolicy    -> Pulsar53Policy (same act/tap surface, 1152-wide h2)
  load_models     -> 5.3 residual loader (verified vs live telemetry + kickoff probe)
  load_latest     -> newest checkpoint under $PULSAR_CKPT_ROOT (default research/data/ckpt53)
  copy_checkpoint -> same
  rebuild_sequential -> RAISES: every legacy call site jit-rebuilds GOAL_CRITIC as a
      raw-obs net, and in 5.2+ the goal critic is a head on CRITIC_TRUNK - a silent
      rebuild would be the h2-truncation bug all over again. Call sites must be
      adapted to Pulsar53Policy.ladder() instead.

Also patches fear_decomp.copy_newest's LIVE_ROOT to the 5.3 copy dir (never the live
folder) when fear_decomp is imported.
"""

import os
import sys
import types
from pathlib import Path

import load_checkpoint_53 as lc53

_REPO = Path(__file__).resolve().parents[2]
_ROOT = Path(os.environ.get("PULSAR_CKPT_ROOT", _REPO / "research" / "data" / "ckpt53"))


def _no_rebuild(*a, **k):
    raise RuntimeError(
        "rebuild_sequential is unavailable under compat53: the 5.2+ goal critic reads "
        "CRITIC_TRUNK features, not raw obs. Use Pulsar53Policy.ladder() / value heads.")


def load_latest(cache_root=None):
    ckpt = lc53.list_checkpoints(_ROOT)[0]
    return lc53.Pulsar53Policy(lc53.load_models(ckpt)), ckpt


shim = types.ModuleType("load_checkpoint")
shim.PulsarPolicy = lc53.Pulsar53Policy
shim.load_models = lc53.load_models
shim.load_latest = load_latest
shim.copy_checkpoint = lambda dest_root=None: lc53.list_checkpoints(_ROOT)[0]
shim.CHECKPOINT_ROOT = _ROOT
shim.MODEL_FILES = lc53.FULL_MODEL_FILES
shim.rebuild_sequential = _no_rebuild
shim.expected_shapes = lambda *a, **k: {}
sys.modules["load_checkpoint"] = shim


def patch_fear_decomp():
    import fear_decomp
    fear_decomp.LIVE_ROOT = _ROOT
    return fear_decomp
