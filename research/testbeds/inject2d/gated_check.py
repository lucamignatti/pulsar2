"""Gated hull: probe battery (B/C/F) + A2 hallucination, 4 seeds."""
import glob, json
import numpy as np
import torch
from ladder import mlp
from env import OBS_DIM
from metric import make_air_probes, make_ground_probes
from metric5 import oracle2
from metric6 import train_chart, make_desc_probes
from metric7 import (train_vddag_hull, make_infeasible_probes, make_feasible_ref,
                     hallucination_index)
from scipy.stats import spearmanr
sp = lambda a, b: spearmanr(a, b).statistic

pB, sB = make_air_probes(600, seed=98)
pC, sC = make_ground_probes(600, seed=97)
pF, sF = make_desc_probes(600, seed=94)
p_inf, s_inf = make_infeasible_probes(500, seed=93)
p_fea, s_fea = make_feasible_ref(500, seed=92)
oB = oracle2(sB, phase=1); oC = oracle2(sC, phase=1); oF = oracle2(sF, phase=1)

rows = []
for f in sorted(glob.glob("bases/base_s*.pt"))[:4]:
    base = torch.load(f, weights_only=False)
    seed = int(f.rsplit("_s", 1)[1].split(".")[0])
    ro, rn, rr = base["res_obs"], base["res_nxt"], base["res_rew"]
    cs, _ = train_chart(ro, rn, seed=seed)
    nets = train_vddag_hull(ro, rn, rr, cs, 1.0, seed=seed, gate=3.0)
    fh = lambda s: torch.minimum(nets[0](s).flatten(), nets[1](s).flatten())
    with torch.no_grad():
        r = {"seed": seed, "B": sp(fh(pB).numpy(), oB), "C": sp(fh(pC).numpy(), oC),
             "F": sp(fh(pF).numpy(), oF),
             "halluc": hallucination_index(fh, p_inf, p_fea)}
    rows.append(r)
    print(json.dumps({k: round(float(v), 3) for k, v in r.items()}), flush=True)
print("means:", {k: round(float(np.mean([x[k] for x in rows])), 3)
                 for k in rows[0] if k != "seed"})
