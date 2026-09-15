"""Roll out the gco-25 policy at trainer parity and record obs + physics, for
frontier-map inspection. Reuses collect_dataset.py's env machinery verbatim.

NOTE the offline reset mix (35/20/15/30 near/air/kickoff/random) is the 5.0 harness
mix, slightly off the live 30/20/15/10/25 (it has no AirPlay setter). Fine for asking
what the map VALUES and AIMS AT; not a parity study.
"""
import sys, os
from pathlib import Path
sys.path.insert(0, "/home/luca/Projects/pulsar2-3.0/research/tools")
import numpy as np, torch, RocketSim as rs
rs.init(str(Path("/home/luca/Projects/pulsar2-3.0/build/collision_meshes")))

import collect_dataset as cd
from advanced_obs import OBS_SIZE_PADDED
import load_checkpoint_70 as l70

SP = Path(sys.argv[1]); NARENA = int(sys.argv[2]); NSTEP = int(sys.argv[3])
cd.set_obs_size(OBS_SIZE_PADDED)
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))

m = l70.load_models(SP, names=["SHARED_HEAD", "POLICY"])
trunk, policy = m["SHARED_HEAD"].eval(), m["POLICY"].eval()


class Env(cd.ArenaEnv):
    """+ records which team scored, and per-car flip availability."""
    def _on_goal(self, **kw):
        self.goal_scored = True
        t = kw.get("team", None)
        self.scoring_team = int(t) if t is not None else -1

    def reset(self, *a, **k):
        self.scoring_team = -1
        super().reset(*a, **k)

    def flips(self):
        s = [c.get_state() for c in self.cars]
        def g(o, n):
            v = getattr(o, n)
            return float(v() if callable(v) else v)
        return np.array([[g(x, "has_flip_or_jump"), g(x, "has_flip_reset"),
                          g(x, "is_on_ground"), g(x, "air_time")] for x in s], np.float32)


envs = [Env(i, np.random.default_rng(1000 + i)) for i in range(NARENA)]

OBS, PHYS, EP, FLIP, STEPIDX = [], [], [], [], []
ep_end = {}           # episode_id -> (end_step_index_in_ep, scoring_team or -1)

with torch.no_grad():
    for t in range(NSTEP):
        obs_l, mask_l, phys_l, flip_l, ep_l, si_l = [], [], [], [], [], []
        for e in envs:
            o, mk, ph = e.observe()
            obs_l.append(o); mask_l.append(mk)
            phys_l.append(np.repeat(ph[None], 2, 0))
            flip_l.append(e.flips())
            ep_l.append(np.full(2, e.episode_id)); si_l.append(np.full(2, e.steps))
        obs = np.concatenate(obs_l, 0).astype(np.float32)
        masks = np.concatenate(mask_l, 0)
        OBS.append(obs); PHYS.append(np.concatenate(phys_l, 0))
        FLIP.append(np.concatenate(flip_l, 0)); EP.append(np.concatenate(ep_l, 0))
        STEPIDX.append(np.concatenate(si_l, 0))

        logits = policy(trunk(torch.from_numpy(obs)))
        logits = logits.masked_fill(~torch.from_numpy(masks).bool(), -1e9)
        acts = torch.distributions.Categorical(logits=logits).sample().numpy()

        for i, e in enumerate(envs):
            done = e.step(acts[2 * i:2 * i + 2])
            if done:
                ep_end[e.episode_id] = (e.steps, e.scoring_team if e.goal_scored else -1)
                e.reset()
        if (t + 1) % 250 == 0:
            print(f"  step {t+1}/{NSTEP}  episodes closed {len(ep_end)}", flush=True)

out = dict(
    obs=np.concatenate(OBS, 0), phys=np.concatenate(PHYS, 0),
    flip=np.concatenate(FLIP, 0), ep=np.concatenate(EP, 0),
    stepidx=np.concatenate(STEPIDX, 0),
    # player index within the arena pair: 0 = BLUE, 1 = ORANGE
    player=np.tile(np.tile([0, 1], NARENA), NSTEP).astype(np.int8),
    ep_end_keys=np.array(list(ep_end.keys()), np.int64),
    ep_end_step=np.array([v[0] for v in ep_end.values()], np.int64),
    ep_end_team=np.array([v[1] for v in ep_end.values()], np.int64),
)
np.savez_compressed(SP / "rollout.npz", **out)
print("frames", out["obs"].shape, "closed episodes", len(ep_end),
      "goal episodes", int((out["ep_end_team"] >= 0).sum()))
