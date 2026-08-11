"""Knowing-doing protocol across the 5.3 (four-rung ladder) datasets on disk.

Exact port of knowing_doing.py's measurement, parameterized by dataset; labels computed
inline (ball-only RocketSim landing sim, as label_landing.py). One JSON row per checkpoint
-> research/results/kd53_curve.json. Pure interpretability: no policy rollouts, no training.
"""
import json, re, sys, time
from pathlib import Path
import numpy as np
import RocketSim as rs
HERE = Path(__file__).resolve().parent
try:
    rs.init(str(HERE.parents[1] / 'build' / 'collision_meshes'))
except Exception:
    pass
sys.path.insert(0, str(HERE))
from train_probes import ALPHAS, cv_predict
from label_landing import simulate_landing, AIRBORNE_Z
from sklearn.linear_model import RidgeCV

KNOW_ERR_UU, ATTEND_RADIUS_UU, CONTEST_DEV_UU, FEASIBLE_SPEED = 500.0, 500.0, 300.0, 1300.0
DATA = HERE.parents[1] / 'research' / 'data'
OUT = HERE.parents[1] / 'research' / 'results' / 'kd53_curve.json'

def kd_point(npz_path):
    data = np.load(npz_path)
    dt = float(data['tick_skip']) / 120.0
    team, episode, phys = data['team'], data['episode'], data['phys']
    # labels
    arena = rs.Arena(rs.GameMode.SOCCAR)
    bp, bv, ba = phys[:, 0:3], phys[:, 3:6], phys[:, 6:9]
    airborne = bp[:, 2] > AIRBORNE_Z
    labels = np.full((len(phys), 3), np.nan, np.float32); valid = np.zeros(len(phys), bool)
    for i in np.flatnonzero(airborne):
        res = simulate_landing(arena, bp[i], bv[i], ba[i])
        if res is not None: labels[i] = res; valid[i] = True
    # episode timelines
    ep_rows = {}
    for e in np.unique(episode):
        rows = np.flatnonzero(episode == e)
        ep_rows[int(e)] = rows[team[rows] == 0]
    row_step = np.empty(len(team), np.int64)
    for e in np.unique(episode):
        all_rows = np.flatnonzero(episode == e)
        row_step[all_rows] = np.arange(len(all_rows)) // 2
    # knowing
    y_team = labels[valid].astype(np.float64).copy()
    orange = team[valid] == 1
    y_team[orange, :2] *= -1
    pred_team = cv_predict(data['h1'][valid].astype(np.float64), y_team,
                           episode[valid], lambda: RidgeCV(alphas=ALPHAS))
    pred_world = pred_team.copy(); pred_world[orange, :2] *= -1
    know_err = np.linalg.norm(pred_world[:, :2] - labels[valid][:, :2], axis=1)
    # doing
    v_rows = np.flatnonzero(valid); n = len(v_rows)
    contested = np.zeros(n, bool); censored = np.zeros(n, bool)
    d_self = np.full(n, np.nan); d_now = np.full(n, np.nan); air_before = np.zeros(n, bool)
    for k, r in enumerate(v_rows):
        e = int(episode[r]); rows_b = ep_rows[e]; s0 = row_step[r]
        L = labels[r, :2]; t_land = labels[r, 2]
        s_land = s0 + int(round(t_land / dt))
        if s_land >= len(rows_b): censored[k] = True; continue
        is_o = team[r] == 1
        sl = slice(20, 23) if is_o else slice(9, 12)
        gnd = 30 if is_o else 19
        d_now[k] = np.linalg.norm(phys[rows_b[s0], sl][:2] - L)
        ball_td = phys[rows_b[s_land], 0:3]
        contested[k] = np.linalg.norm(ball_td[:2] - L) > CONTEST_DEV_UU or ball_td[2] > 200
        d_self[k] = np.linalg.norm(phys[rows_b[s_land], sl][:2] - L)
        air_before[k] = (phys[rows_b[s0:s_land + 1], gnd] == 0).any()
    ok = ~censored; free = ok & ~contested
    t_land_all = labels[v_rows, 2]
    feasible = free & (d_now / np.maximum(t_land_all, 1e-6) < FEASIBLE_SPEED)
    knows = know_err < KNOW_ERR_UU
    def p_un(mask):
        m = mask & feasible
        return (float((d_self[m] > ATTEND_RADIUS_UU).mean()) if m.sum() else float('nan')), int(m.sum())
    p_know, n_know = p_un(knows); p_dumb, n_dumb = p_un(~knows)
    q = np.quantile(know_err[feasible], [0.25, 0.5, 0.75])
    quart = np.digitize(know_err, q)
    by_q = [float((d_self[feasible & (quart == i)] > ATTEND_RADIUS_UU).mean()) for i in range(4)]
    high = free & (t_land_all > 0.75) & knows & (d_now < 2500)
    p_ground = float((~air_before[high]).mean()) if high.sum() else float('nan')
    return dict(n_readings=int(n), censored=int(censored.sum()),
                feasible=int(feasible.sum()),
                p_unattend_knows=p_know, n_knows=n_know,
                p_unattend_not=p_dumb, n_not=n_dumb,
                quartiles=by_q, aerial_passivity=p_ground,
                know_frac=float(knows[feasible].mean()) if feasible.sum() else float('nan'))

rows = []
files = sorted(DATA.glob('dataset53_r*.npz')) + [DATA / 'dataset53_9750127919.npz']
for f in files:
    if 'smoke' in f.name: continue
    m = re.search(r'(?:_r|_)(\d+)\.npz', f.name)
    ts = int(m.group(1))
    t0 = time.time()
    r = kd_point(f); r['timesteps'] = ts
    rows.append(r)
    print(f'{f.name}: ts {ts/1e9:.2f}B  P(unatt|knows) {r["p_unattend_knows"]:.3f} '
          f'P(unatt|~knows) {r["p_unattend_not"]:.3f}  aerial_passivity {r["aerial_passivity"]:.3f} '
          f'({time.time()-t0:.0f}s)', flush=True)
rows.sort(key=lambda r: r['timesteps'])
json.dump(rows, open(OUT, 'w'), indent=1)
print('wrote', OUT)
