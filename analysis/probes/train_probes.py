"""Linear probes for Detector #1 (ball landing) on the frozen Pulsar checkpoint.

Sources probed (same episode-grouped 5-fold CV for all):
  raw_obs   109-d AdvancedObs        <- THE control: what a linear readout gets for free
  trunk_h1  512-d after trunk block 1
  trunk_h2  512-d trunk output (what policy/critic/phi consume)
  reach_phi 128-d InfoNCE state-action embedding (L2-normalized)

Targets: x_land, y_land (uu), t_land (s) of the airborne ball's first touchdown.

Controls:
  - small MLP on raw obs        ("computable from obs at all" reference)
  - shuffled labels per source  (must be ~0, else leakage)
  - passthrough: current ball z decoded from trunk activations (must be ~1,
    else activation extraction is broken)

CV is grouped by EPISODE: adjacent frames are near-duplicates, so random folds would
leak. Ridge alpha is chosen per fold on the train split (RidgeCV).
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
RESULTS_DIR = HERE / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

SEED = 7
ALPHAS = np.logspace(-2, 5, 12)
TARGET_NAMES = ["x_land", "y_land", "t_land"]


def r2(y, pred):
    ss_res = ((y - pred) ** 2).sum(0)
    ss_tot = ((y - y.mean(0)) ** 2).sum(0)
    return 1 - ss_res / ss_tot


def cv_predict(X, y, groups, model_fn, n_splits=5, max_train=None):
    """Out-of-fold predictions with episode-grouped folds.

    Targets are standardized per fold (and inverse-transformed) so multi-output
    models balance x/y (~2500-3000 uu std) against t (~0.9 s std); without this the
    squared loss ignores t_land entirely. max_train subsamples training rows for
    expensive models (the MLP control).
    """
    pred = np.full_like(y, np.nan, dtype=np.float64)
    rng = np.random.default_rng(SEED)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        if max_train is not None and len(tr) > max_train:
            tr = rng.choice(tr, max_train, replace=False)
        xs = StandardScaler().fit(X[tr])
        ys = StandardScaler().fit(y[tr])
        m = model_fn()
        m.fit(xs.transform(X[tr]), ys.transform(y[tr]))
        p = np.asarray(m.predict(xs.transform(X[te]))).reshape(len(te), -1)
        pred[te] = ys.inverse_transform(p)
    assert not np.isnan(pred).any()
    return pred


def evaluate(y, pred, bounce=None):
    r2s = r2(y, pred)
    land_err = np.linalg.norm(y[:, :2] - pred[:, :2], axis=1)
    out = {
        "r2": {n: float(v) for n, v in zip(TARGET_NAMES, r2s)},
        "landing_err_uu_median": float(np.median(land_err)),
        "landing_err_uu_mean": float(land_err.mean()),
        "t_err_s_median": float(np.median(np.abs(y[:, 2] - pred[:, 2]))),
    }
    if bounce is not None:
        # the interesting slice: frames where a wall/ceiling bounce breaks ballistics
        out["landing_err_uu_median_bounce"] = float(np.median(land_err[bounce]))
        out["landing_err_uu_median_direct"] = float(np.median(land_err[~bounce]))
    return out


def ballistic_prediction(pos, vel):
    """Closed-form no-bounce, no-drag landing (solve z(t) = rest height under gravity).

    A pure-physics control: what a hand-coded formula gets without learning anything.
    Frames where this misses badly are the wall/ceiling-bounce frames - the part of the
    concept that is genuinely nonlinear in the obs.
    """
    g = 650.0
    a, b, c = -0.5 * g, vel[:, 2], pos[:, 2] - 93.0
    t = (-b - np.sqrt(np.maximum(b * b - 4 * a * c, 0))) / (2 * a)
    return np.stack([pos[:, 0] + vel[:, 0] * t, pos[:, 1] + vel[:, 1] * t, t], 1)


def main():
    rng = np.random.default_rng(SEED)
    data = np.load(DATA_DIR / "dataset.npz")
    lab = np.load(DATA_DIR / "labels.npz")
    assert int(data["checkpoint"]) == int(lab["checkpoint"])

    valid = lab["valid"]
    y = lab["land_xy_t"][valid].astype(np.float64)
    groups = data["episode"][valid]

    # CRITICAL frame alignment: AdvancedObs is team-inverted (x,y negated) for the
    # ORANGE player, so the network's entire input - and therefore its activations -
    # live in a team-canonical frame. Labels must live there too, or the x/y mapping
    # has opposite sign on half the rows and every probe degenerates (first run of
    # this script: R2 x/y ~ -0.5 on all sources, t_land unaffected - the smoking gun).
    orange = data["team"][valid] == 1
    y[orange, :2] *= -1
    print(f"{valid.sum():,} labeled airborne frames, {len(np.unique(groups))} episodes, "
          f"{orange.mean():.0%} orange (labels team-canonicalized)")

    sources = {
        "raw_obs": data["obs"][valid].astype(np.float64),
        "trunk_h1": data["h1"][valid].astype(np.float64),
        "trunk_h2": data["h2"][valid].astype(np.float64),
        "reach_phi": data["phi"][valid].astype(np.float64),
    }

    # Physics-formula control + the bounce/direct split it induces
    # (ball physics team-canonicalized to match the labels)
    phys = data["phys"][valid].astype(np.float64)
    ball_pos, ball_vel = phys[:, 0:3].copy(), phys[:, 3:6].copy()
    ball_pos[orange, :2] *= -1
    ball_vel[orange, :2] *= -1
    bal_pred = ballistic_prediction(ball_pos, ball_vel)
    bounce = np.linalg.norm(y[:, :2] - bal_pred[:, :2], axis=1) > 500
    print(f"bounce frames (ballistic misses by >500uu): {bounce.mean():.1%}")

    ridge = lambda: RidgeCV(alphas=ALPHAS)
    results = {"n_frames": int(valid.sum()), "checkpoint": int(data["checkpoint"]),
               "bounce_frac": float(bounce.mean()), "sources": {}, "controls": {}}
    preds = {}

    results["controls"]["ballistic_closed_form"] = evaluate(y, bal_pred, bounce)
    preds["ballistic"] = bal_pred

    for name, X in sources.items():
        pred = cv_predict(X, y, groups, ridge)
        preds[name] = pred
        results["sources"][name] = evaluate(y, pred, bounce)
        m = results["sources"][name]
        print(f"ridge {name:10s} R2 x/y/t = {m['r2']['x_land']:.3f}/{m['r2']['y_land']:.3f}/"
              f"{m['r2']['t_land']:.3f}   median landing err {m['landing_err_uu_median']:6.0f} uu "
              f"(bounce {m['landing_err_uu_median_bounce']:6.0f} / direct {m['landing_err_uu_median_direct']:6.0f})")

    # Control 1: shuffled labels (per source) - any R2 meaningfully above 0 is leakage
    y_shuf = y[rng.permutation(len(y))]
    for name, X in sources.items():
        pred = cv_predict(X, y_shuf, groups, ridge)
        results["controls"][f"shuffled_{name}"] = evaluate(y_shuf, pred)
        r = results["controls"][f"shuffled_{name}"]["r2"]
        print(f"shuffled {name:10s} R2 = {r['x_land']:.4f}/{r['y_land']:.4f}/{r['t_land']:.4f}")

    # Control 2: small MLP on raw obs - is the concept computable from obs at all?
    # (a reference point, not a probe: capped iterations keep it from hogging the box)
    mlp = lambda: MLPRegressor(hidden_layer_sizes=(256, 256), max_iter=300,
                               early_stopping=True, n_iter_no_change=8,
                               random_state=SEED)
    pred = cv_predict(sources["raw_obs"], y, groups, mlp, max_train=12_000)
    preds["mlp_raw_obs"] = pred
    results["controls"]["mlp_raw_obs"] = evaluate(y, pred, bounce)
    m = results["controls"]["mlp_raw_obs"]
    print(f"MLP raw_obs      R2 x/y/t = {m['r2']['x_land']:.3f}/{m['r2']['y_land']:.3f}/"
          f"{m['r2']['t_land']:.3f}   median landing err {m['landing_err_uu_median']:6.0f} uu")

    # Control 3: passthrough sanity - decode CURRENT ball z from the trunk.
    # obs[2] = ball z * (1/5000) exactly, so raw_obs must be ~1.0 by construction and
    # the trunk must be near-1.0 or activation extraction is broken.
    ball_z = (data["obs"][valid][:, 2].astype(np.float64) * 5000.0).reshape(-1, 1)
    for name in ["raw_obs", "trunk_h1", "trunk_h2"]:
        pred = cv_predict(sources[name], ball_z, groups, ridge)
        pz = float(r2(ball_z, pred)[0])
        results["controls"][f"passthrough_ball_z_{name}"] = {"r2": pz}
        print(f"passthrough ball_z from {name:10s} R2 = {pz:.4f}")
    # Extraction sanity gates on h1: h1 and h2 come from the same forward pass, so a
    # broken extraction would sink both. h2's passthrough level is itself a finding
    # (how much of the raw signal trunk layer 2 keeps linearly available) - report it,
    # don't gate on it.
    pz_trunk = results["controls"]["passthrough_ball_z_trunk_h1"]["r2"]
    if pz_trunk < 0.95:
        raise RuntimeError(f"passthrough probe R2={pz_trunk:.3f} - activation extraction broken, STOP")

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {RESULTS_DIR / 'metrics.json'}")

    make_plots(y, preds, results)


def make_plots(y, preds, results):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # R2 bar chart across sources
    names = list(results["sources"].keys()) + ["mlp_raw_obs"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, tgt in zip(axes, TARGET_NAMES):
        vals = [results["sources"].get(n, results["controls"].get(n))["r2"][tgt] for n in names]
        colors = ["tab:gray" if n == "raw_obs" else "tab:orange" if n == "mlp_raw_obs"
                  else "tab:blue" for n in names]
        ax.bar(names, vals, color=colors)
        ax.set_title(f"R² {tgt} (5-fold episode-grouped CV)")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "r2_by_source.png", dpi=120)

    # pred vs true scatter, best trunk source vs raw obs
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for row, src in enumerate(["raw_obs", "trunk_h2"]):
        for col, tgt in enumerate(TARGET_NAMES):
            ax = axes[row, col]
            s = np.random.default_rng(0).choice(len(y), min(4000, len(y)), replace=False)
            ax.scatter(y[s, col], preds[src][s, col], s=2, alpha=0.25)
            lo, hi = y[:, col].min(), y[:, col].max()
            ax.plot([lo, hi], [lo, hi], "r--", lw=1)
            ax.set_title(f"{src}: {tgt}  (R²={results['sources'][src]['r2'][tgt]:.3f})")
            ax.set_xlabel("true")
            ax.set_ylabel("pred")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "pred_vs_true.png", dpi=120)

    # landing error histogram
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for src in ["raw_obs", "trunk_h2", "mlp_raw_obs", "ballistic"]:
        p = preds[src]
        err = np.linalg.norm(y[:, :2] - p[:, :2], axis=1)
        ax.hist(err, bins=80, range=(0, 4000), histtype="step", density=True, label=src)
    ax.set_xlabel("landing position error (uu)")
    ax.set_ylabel("density")
    ax.set_title("Ball-landing probe error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "landing_error_hist.png", dpi=120)
    print(f"wrote plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
