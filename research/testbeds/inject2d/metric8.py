"""Hull-slack battery: the A2-surviving operator on families A/B/C/F."""
import glob, json, os
import numpy as np
import torch

from env import OBS_DIM
from ladder import mlp
from foresight import make_probes
from metric import make_air_probes, make_ground_probes
from metric5 import oracle2
from metric6 import train_chart, make_desc_probes
from metric7 import train_vddag_hull


def main():
    from scipy.stats import spearmanr
    sp = lambda a, b: spearmanr(a, b).statistic
    pA, sA = make_probes(600, seed=99)
    pB, sB = make_air_probes(600, seed=98)
    pC, sC = make_ground_probes(600, seed=97)
    pF, sF = make_desc_probes(600, seed=94)
    print("oracles...")
    oA = oracle2(sA, phase=2); oB = oracle2(sB, phase=1)
    oC = oracle2(sC, phase=1); oF = oracle2(sF, phase=1)

    rows = []
    for f in sorted(glob.glob(os.path.join("bases", "base_s*.pt"))):
        base = torch.load(f, weights_only=False)
        seed = int(f.rsplit("_s", 1)[1].split(".")[0])
        ro, rn, rr = base["res_obs"], base["res_nxt"], base["res_rew"]
        cs, _ = train_chart(ro, rn, seed=seed)
        nets = train_vddag_hull(ro, rn, rr, cs, 1.0, seed=seed)
        fh = lambda s: torch.minimum(nets[0](s).flatten(), nets[1](s).flatten())
        with torch.no_grad():
            r = {"seed": seed,
                 "eh_A": sp(fh(pA).numpy(), oA), "eh_B": sp(fh(pB).numpy(), oB),
                 "eh_C": sp(fh(pC).numpy(), oC), "eh_F": sp(fh(pF).numpy(), oF)}
        rows.append(r)
        print(json.dumps({k: round(float(v), 3) for k, v in r.items()}))
    print("\nmeans:")
    for k in [k for k in rows[0] if k != "seed"]:
        vals = [x[k] for x in rows]
        print(f"  {k:6s} {np.mean(vals):+.3f}  (seeds: {' '.join(f'{v:+.2f}' for v in vals)})")


if __name__ == "__main__":
    main()
