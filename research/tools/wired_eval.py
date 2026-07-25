"""Wire-on vs wire-zero head-to-head: the conclusive diagnosis for the league-gate
confound (2026-07-19).

Context: every eval path (Rating, render, compare_checkpoints) feeds the Ladder's
5 policy-head wire inputs as ZEROS, but collection always feeds REAL values
(PPOLearner.cpp InferPolicyProbsFromModels) - so a wire-dependent policy is
understated by every measurement. The league convict-gate (current loses 40-43%
to 3 spaced past selves) is confounded by exactly this.

This script replicates ComputeWire (PPOLearner.cpp:136-161) faithfully offline:
    wire = tanh([V_real, V_exp, gap_KD, V_metric, gap_PK] / wireScale)
    V_real = CRITIC(trunk_out)          (same checkpoint generation)
    V_exp  = GAP_EXP(trunk_out)         (512 -> 256 -> 256 -> 1, LeakyReLU)
    e      = GAP_MAP_F(GAP_MAP_E(raw_obs))   (230->256->256->64 GELU; 64->128->32 GELU)
    d_g/d_c = min-bank asymmetric dist  sum_j relu(e_j - b_j), clamp dClamp (fp32 CPU)
    V_metric = a*gamma^d_g + a2*gamma^d_c + b   (calib a/a2/b from RUNNING_STATS)
    banksReady=false parity: V_metric := V_exp  (gap_PK = 0) - NOT zero
    gamma = gaeGamma = TRAIN_GAMMA (0.9969), wireScale = 3,
    dClamp = max(1, 3 * max(1, ladder_dclamp_ema))       (Learner.cpp:1872-1874)

Banks are NOT persisted (they refill live), so we rebuild pseudo-banks by the
trainer's own fill rule (Learner.cpp:4906-4921): the last W = 120/tickSkip = 15
rows of every goal-terminated episode, scorer's rows -> goal bank, conceder's ->
concede bank, ring-capped at bankCapacity=256/side (ExampleMain), embedded
through the persisted map.

Experiments (sides swapped halfway, fixed seeds, ~50 eps/half):
  A subject[wire-full] vs subject[wire-zero]  - direct wire-dependence magnitude
  B subject[wire-zero] vs old 18.88B          - replicates the confounded gate
  C subject[wire-full] vs old 18.88B          - the unconfounded gate
B vs C separates "genuine forgetting" (B ~= C, both lost) from "eval artifact"
(C >> B). A is the guard-integrity number LADDER.md's watch list asks for.

Usage: wired_eval.py <full_subject_ckpt_dir> <old_ckpt_dir>
(subject dir must contain GAP_EXP/GAP_MAP_E/GAP_MAP_F/RUNNING_STATS + the 6
standard .lt; "full_<ts>" dirs from the cache qualify.)
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

import RocketSim as rs
from collect_dataset import NUM_ARENAS, set_obs_size
from compare_checkpoints import ScoringEnv
from load_checkpoint import PulsarPolicy, load_models, rebuild_sequential
from steer_test import SteeredPolicy

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE.parent / "results"
SEED = 777
ROWS_PER_HALF = 40_000
BANK_ROWS = 40_000

TRAIN_GAMMA = 0.9969        # ExampleMain TRAIN_GAMMA == gaeGamma == wire gamma
WIRE_SCALE = 3.0            # GapSensorConfig.wireScale (default, not overridden)
DCLAMP_MULT = 3.0           # GapSensorConfig.dClampMult
BANK_CAP = 256              # ExampleMain cfg.gapSensor.bankCapacity
BANK_WINDOW = 15            # 120 / tickSkip(8) decision steps ~ 1s


def rebuild_gap_net(path: Path) -> torch.nn.Sequential:
    """GAP_* nets are plain [Linear, activation]* stacks (no LayerNorm): rebuild
    Linears from params; activation gaps are LeakyReLU (GAP_EXP, Learner.cpp:117)
    or GELU (GAP_MAP_*, :137-143) - chosen by filename."""
    jit_mod = torch.jit.load(str(path), map_location="cpu")
    params = dict(jit_mod.named_parameters())
    by_idx = {}
    for name, p in params.items():
        idx, kind = name.split(".")
        by_idx.setdefault(int(idx), {})[kind] = p.detach().clone()
    act = torch.nn.GELU if "MAP" in path.name else torch.nn.LeakyReLU
    layers = []
    for i in range(max(by_idx) + 1):
        if i not in by_idx:
            layers.append(act())
            continue
        w, b = by_idx[i]["weight"], by_idx[i]["bias"]
        lin = torch.nn.Linear(w.shape[1], w.shape[0])
        lin.weight.data, lin.bias.data = w, b
        layers.append(lin)
    seq = torch.nn.Sequential(*layers)
    seq.eval()
    for p in seq.parameters():
        p.requires_grad_(False)
    return seq


def min_bank_dist(emb: torch.Tensor, bank: torch.Tensor, d_clamp: float) -> torch.Tensor:
    """d(x,y) = sum_j relu(x_j - y_j), min over bank rows, clamped (fp32 CPU parity
    with PPOLearner::MinBankDist's CPU path)."""
    d = torch.relu(emb.unsqueeze(1) - bank.unsqueeze(0)).sum(-1)  # [n, k]
    return d.min(1).values.clamp_max(d_clamp)


class WiredPolicy(PulsarPolicy):
    """PulsarPolicy whose policy head gets the REAL Ladder wire (ComputeWire parity).

    mode: "zero"   - all-zero wire (current eval behavior; the confounded baseline)
          "nobank" - banksReady=false parity: V_metric := V_exp, gap_PK = 0
          "full"   - pseudo-banks: complete 5-value wire
    """

    def __init__(self, models, gap_exp, map_e, map_f, calib, d_clamp, mode="full"):
        super().__init__(models)
        assert self.wire_pad == 5, f"subject must be 517-wide (wire_pad {self.wire_pad})"
        self.gap_exp, self.map_e, self.map_f = gap_exp, map_e, map_f
        self.a, self.a2, self.b = calib
        self.d_clamp = d_clamp
        self.mode = mode
        self.bank_g = None  # [k, 32] embeddings
        self.bank_c = None
        self.wire_log = []  # pre-tanh [V_real, V_exp, gKD, V_met, gPK] means per batch

    def embed(self, raw_obs: torch.Tensor) -> torch.Tensor:
        return self.map_f(self.map_e(raw_obs.float()))

    @torch.no_grad()
    def compute_wire(self, raw_obs, trunk_out):
        v_real = self.critic(trunk_out).flatten()
        v_exp = self.gap_exp(trunk_out).flatten()
        use_banks = self.mode == "full" and self.bank_g is not None and self.bank_c is not None
        if use_banks:
            e = self.embed(raw_obs)
            dg = min_bank_dist(e, self.bank_g, self.d_clamp)
            dc = min_bank_dist(e, self.bank_c, self.d_clamp)
            v_met = self.a * TRAIN_GAMMA ** dg + self.a2 * TRAIN_GAMMA ** dc + self.b
        else:
            v_met = v_exp
        g_kd = torch.relu(v_exp - v_real)
        g_pk = torch.relu(v_met - v_exp)
        pre = torch.stack([v_real, v_exp, g_kd, v_met, g_pk], -1)
        self.wire_log.append(pre.mean(0).numpy())
        return torch.tanh(pre / WIRE_SCALE)

    @torch.no_grad()
    def act(self, obs, masks):
        h1, h2 = self.trunk_forward(obs)
        if self.mode == "zero":
            wire = h2.new_zeros(h2.shape[0], 5)
        else:
            wire = self.compute_wire(obs, h2)
        logits = self.policy(torch.cat([h2, wire], -1))
        logits = logits + self.ACTION_DISABLED_LOGIT * (~masks.bool()).float()
        probs = torch.softmax(logits, -1).clamp(self.ACTION_MIN_PROB, 1)
        return h2, torch.multinomial(probs, 1, True).flatten()


def build_banks(policy: WiredPolicy, seed: int):
    """Trainer bank-fill parity: roll the subject (nobank wire), take the last
    BANK_WINDOW rows of each player's goal-terminated episode; scorer rows -> G,
    conceder rows -> C; keep the most recent BANK_CAP per side; embed."""
    torch.manual_seed(seed)
    envs = [ScoringEnv(i, np.random.default_rng(seed + 10 + i)) for i in range(NUM_ARENAS)]
    ep_obs = [[[], []] for _ in envs]  # per env, per player: obs rows this episode
    bank_g, bank_c = [], []

    rows = 0
    while rows < BANK_ROWS:
        obs_list, mask_list = [], []
        for env in envs:
            obs, masks, _ = env.observe()
            obs_list.append(obs)
            mask_list.append(masks)
        obs_b = torch.from_numpy(np.concatenate(obs_list))
        mask_b = torch.from_numpy(np.concatenate(mask_list))
        _, actions = policy.act(obs_b, mask_b)

        for i, env in enumerate(envs):
            for p in range(2):
                ep_obs[i][p].append(obs_list[i][p])
            rows += 2
        for i, env in enumerate(envs):
            prev_goals = list(env.team_goals)
            if env.step(actions[2 * i:2 * i + 2].tolist()):
                if env.goal_scored:
                    scorer = 0 if env.team_goals[0] > prev_goals[0] else 1
                    for p in range(2):
                        window = ep_obs[i][p][-BANK_WINDOW:]
                        (bank_g if p == scorer else bank_c).extend(window)
                ep_obs[i] = [[], []]
                env.reset()

    bank_g, bank_c = bank_g[-BANK_CAP:], bank_c[-BANK_CAP:]
    assert len(bank_g) >= 32 and len(bank_c) >= 32, \
        f"banks under min fill: G={len(bank_g)} C={len(bank_c)}"
    with torch.no_grad():
        g = policy.embed(torch.from_numpy(np.stack(bank_g)))
        c = policy.embed(torch.from_numpy(np.stack(bank_c)))
    print(f"pseudo-banks: G {len(bank_g)} rows, C {len(bank_c)} rows (window {BANK_WINDOW}, cap {BANK_CAP})")
    return g, c


def cross(pol_b, pol_o, n_rows, seed):
    """Head-to-head: blue rows by pol_b, orange by pol_o. Returns goal tally + eps."""
    torch.manual_seed(seed)
    envs = [ScoringEnv(i, np.random.default_rng(seed + 10 + i)) for i in range(NUM_ARENAS)]
    episodes = 0
    rows = 0
    while rows < n_rows:
        obs_list, mask_list = [], []
        for env in envs:
            obs, masks, _ = env.observe()
            obs_list.append(obs)
            mask_list.append(masks)
        obs_b = torch.from_numpy(np.concatenate(obs_list))
        mask_b = torch.from_numpy(np.concatenate(mask_list))
        blue = torch.arange(0, len(obs_b), 2)
        orange = torch.arange(1, len(obs_b), 2)
        actions = torch.empty(len(obs_b), dtype=torch.long)
        _, ab = pol_b.act(obs_b[blue], mask_b[blue])
        _, ao = pol_o.act(obs_b[orange], mask_b[orange])
        actions[blue], actions[orange] = ab, ao
        rows += len(obs_b)
        for i, env in enumerate(envs):
            if env.step(actions[2 * i:2 * i + 2].tolist()):
                episodes += 1
                env.reset()
    goals = [sum(e.team_goals[0] for e in envs), sum(e.team_goals[1] for e in envs)]
    return goals, episodes


def pair(name, mk_a, mk_b, results):
    """Side-swapped pairing: half 1 A=blue, half 2 A=orange. Fresh policy objects
    per half so wire logs don't blend."""
    ga = gb = eps = 0
    for half in range(2):
        pa, pb = mk_a(), mk_b()
        blue, orange = (pa, pb) if half == 0 else (pb, pa)
        goals, e = cross(blue, orange, ROWS_PER_HALF, SEED + 50 + half)
        a_goals = goals[0] if half == 0 else goals[1]
        b_goals = goals[1] if half == 0 else goals[0]
        ga += a_goals
        gb += b_goals
        eps += e
        print(f"  {name} half {half+1}: A {a_goals} - {b_goals} B ({e} eps)")
    tot = max(ga + gb, 1)
    results[name] = {"A_goals": ga, "B_goals": gb, "episodes": eps,
                     "A_share": ga / tot}
    print(f"  {name} TOTAL: A {ga} - {gb} B  -> A share {ga/tot:.0%} over {eps} eps")
    return results[name]


def main():
    subj_dir, old_dir = Path(sys.argv[1]), Path(sys.argv[2])
    torch.set_num_threads(4)
    rs.init(str(HERE.parents[1] / "build" / "collision_meshes"))

    subj_models = load_models(subj_dir)
    old_models = load_models(old_dir)
    stats = json.loads((subj_dir / "RUNNING_STATS.json").read_text())
    calib = (stats["ladder_calib_a"], stats["ladder_calib_a2"], stats["ladder_calib_b"])
    d_clamp = max(1.0, DCLAMP_MULT * max(1.0, stats["ladder_dclamp_ema"]))
    gap_exp = rebuild_gap_net(subj_dir / "GAP_EXP.lt")
    map_e = rebuild_gap_net(subj_dir / "GAP_MAP_E.lt")
    map_f = rebuild_gap_net(subj_dir / "GAP_MAP_F.lt")

    def wired(mode):
        return WiredPolicy(subj_models, gap_exp, map_e, map_f, calib, d_clamp, mode)

    set_obs_size(PulsarPolicy(subj_models).obs_size)
    print(f"subject {subj_dir.name} (wired) vs old {old_dir.name}")
    print(f"calib a={calib[0]:.3f} a2={calib[1]:.3f} b={calib[2]:.3f}  dClamp={d_clamp:.0f}\n")

    # pseudo-banks from a subject rollout (shared by every "full" policy instance)
    bank_pol = wired("nobank")
    bank_g, bank_c = build_banks(bank_pol, SEED)

    def full():
        p = wired("full")
        p.bank_g, p.bank_c = bank_g, bank_c
        return p

    # wire sanity on one batch of bank-rollout obs (pre-tanh means)
    wl = np.mean(bank_pol.wire_log, 0)
    print(f"nobank wire pre-tanh means: V_real {wl[0]:.3f}  V_exp {wl[1]:.3f}  "
          f"gap_KD {wl[2]:.3f}  (V_met==V_exp, gap_PK 0)\n")

    results = {"subject": subj_dir.name, "old": old_dir.name,
               "calib": calib, "d_clamp": d_clamp}
    print("A) subject[full-wire] vs subject[zero-wire]  (wire-dependence)")
    pair("A_full_vs_zero", full, lambda: wired("zero"), results)
    print("B) subject[zero-wire] vs old  (confounded gate, this subject)")
    pair("B_zero_vs_old", lambda: wired("zero"), lambda: SteeredPolicy(old_models), results)
    print("C) subject[full-wire] vs old  (unconfounded gate)")
    pair("C_full_vs_old", full, lambda: SteeredPolicy(old_models), results)

    fw = full()
    _ = fw.act(torch.from_numpy(np.stack([e.observe()[0][0] for e in
        [ScoringEnv(0, np.random.default_rng(1))]])), torch.ones(1, 90, dtype=torch.uint8))
    pre = fw.wire_log[-1]
    print(f"\nfull wire pre-tanh sample: V_real {pre[0]:.3f} V_exp {pre[1]:.3f} "
          f"gap_KD {pre[2]:.3f} V_met {pre[3]:.3f} gap_PK {pre[4]:.3f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "wired_eval.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
