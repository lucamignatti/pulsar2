"""Pull through-training series for the 5.3-geo run (137ff497) into
research/results/wandb_53_history.json. Uses /usr/bin/python3 (has wandb + ~/.netrc).

Trap (memory): a history query with ANY absent key returns zero rows - every key is
checked against run.summary first and queried per-family."""

import json
from pathlib import Path

import wandb

RESULTS = Path(__file__).resolve().parents[1] / "results"

FAMILIES = {
    "steps": ["Total Timesteps"],
    "rating": ["Rating/1v1", "Rating/2v2", "Rating/3v3"],
    "geo": ["Geo/V Mean", "Geo/Residual", "Geo/H Geo Mean", "Geo/H Geo P90",
            "Geo/Rew Loss", "Geo/Reservoir Fill"],
    "headroom": ["Headroom/H Mean", "Headroom/H P90", "Headroom/Vdag Mean",
                 "Headroom/Vdag Twin Spread", "Headroom/Vdag Update Magnitude",
                 "Headroom/Inj Std Ratio", "Headroom/Yv Abs"],
    "goalcritic": ["GoalCritic/Value-Outcome Corr", "GoalCritic/Adv-Outcome Corr",
                   "GoalCritic/Mean Val", "GoalCritic/Val Abs Mean"],
    "reach": ["Reach/Ball Accuracy", "Reach/Car Accuracy", "Reach/Car State Accuracy",
              "Reach/Aux Loss"],
    "nexto": [],   # filled from summary scan
    "ref": [],
    "plasticity": [],
    "core": ["Policy Entropy", "Average Episode Reward", "Value Loss", "Policy Loss"],
}


def main():
    api = wandb.Api()
    run = api.run("lucamignatti-personal/gigalearncpp/137ff497")
    skeys = set(run.summary.keys())
    for k in sorted(skeys):
        if k.startswith("Nexto/"):
            FAMILIES["nexto"].append(k)
        elif k.startswith("Ref/"):
            FAMILIES["ref"].append(k)
        elif k.startswith("Plasticity/"):
            FAMILIES["plasticity"].append(k)

    out = {"run": "137ff497", "summary_keys": sorted(skeys)}
    for fam, keys in FAMILIES.items():
        keys = [k for k in keys if k in skeys]
        if not keys:
            continue
        rows = list(run.history(keys=["Total Timesteps"] + keys, samples=400,
                                pandas=False))
        out[fam] = {"keys": keys, "rows": rows}
        print(f"{fam}: {len(rows)} rows for {keys}")

    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / "wandb_53_history.json"
    p.write_text(json.dumps(out))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
