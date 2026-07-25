"""TEMP_PROBE — is the takeoff latent-but-suppressed or truly absent?

The policy's entropy is 0.71 nats (~2 effective actions/state). This probe
reruns the AERIAL_GAP M4 takeoff test at softmax temperatures {1, 1.5, 2, 3}:

  - climb rates RISE with temperature -> the skill is LATENT and the peaked
    policy is the jailer -> exploration levers (entropy target, hot practice
    arenas) are the unlock; drills then consolidate.
  - flat ~0 at every temperature -> the circuit does not exist -> only
    curriculum data (takeoff drill v2) can build it; entropy is a red herring.

Also reports the action-distribution entropy of the probe states at each
temperature (context for the live 0.71 figure).
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

import RocketSim as rs
from aerial_gap import roll_track, set_takeoff_probe
from fear_decomp import copy_newest
from load_checkpoint import load_models, PulsarPolicy
from steer_team import TeamArenaEnv

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE.parent / "results"
SEED = 20260730
N_EP = 200
TEMPS = [1.0, 1.5, 2.0, 3.0]


class HotPolicy(PulsarPolicy):
    def __init__(self, models, temperature=1.0):
        super().__init__(models)
        self.T = temperature

    @torch.no_grad()
    def act(self, obs, masks):
        h1, h2 = self.trunk_forward(obs)
        probs = self.action_probs(h2, masks)
        if self.T != 1.0:
            logp = probs.clamp_min(1e-11).log() / self.T
            logp = logp.masked_fill(masks == 0, -1e10)
            probs = torch.softmax(logp, -1)
        self.last_entropy = float(-(probs.clamp_min(1e-11).log() * probs).sum(-1).mean())
        return h2, torch.multinomial(probs, 1, True).flatten()


def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    rs.init(str(HERE.parents[1] / "build" / "collision_meshes"))
    ckpt = copy_newest(HERE.parent / "data" / "ckpt_cache")
    models = load_models(ckpt)
    print(f"pinned checkpoint {ckpt.name}", flush=True)
    res = {"checkpoint": int(ckpt.name), "n_ep": N_EP, "temps": {}}

    env = TeamArenaEnv(0, 1, np.random.default_rng(SEED + 5))
    for T in TEMPS:
        rng = np.random.default_rng(SEED)      # same episode set per temperature
        pol = HotPolicy(models, T)
        jump = air500 = atouch = 0
        ents = []
        for _ in range(N_EP):
            set_takeoff_probe(env, rng)
            j, mz, tz = roll_track(env, pol, 4.0, track_car=0)
            jump += j
            air500 += mz > 500
            atouch += (tz is not None and tz > 500)
            ents.append(pol.last_entropy)
        res["temps"][str(T)] = {"jump_1s": jump / N_EP, "carz500": air500 / N_EP,
                                "aerial_touch": atouch / N_EP,
                                "mean_entropy": float(np.mean(ents))}
        m = res["temps"][str(T)]
        print(f"T={T}: jump1s {m['jump_1s']:.0%} carZ>500 {m['carz500']:.0%} "
              f"aerialTouch {m['aerial_touch']:.1%} (H {m['mean_entropy']:.2f}) "
              f"({time.time()-t0:.0f}s)", flush=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"temp_probe_{ckpt.name}.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
