"""Prior-free frontier mining for the META steering system.

HARD REQUIREMENT: no human priors. Everything the system steers toward is derived
from the agent itself:

  goals      sampled from the agent's OWN achieved-state bank, in the self-model's
             two goal spaces (car-local ball = contact/races; canonical ball =
             trajectories) - no human names anything a "shot" or a "clear"
  frontier   (state, goal) pairs the agent's OWN self-model rates coin-flip
             (rho quantile band, per-batch self-calibrating)
  structure  clusters in the agent's OWN psi-embedding geometry - emergent regions
             of its achieved-outcome space, refreshed as the agent changes
  outcome    CONTINUOUS ATTAINMENT, model-free and knob-free: how close (raw goal
             space) the agent's future achieved states got to the goal within the
             head's OWN HER horizon; contrasts and gates use within-population
             quantiles of attainment, so no threshold exists to hand-tune
  matching   the direction contrast is matched on rho itself - the confound IS
             "already likely to succeed" - not on hand-picked geometry
  direction  trunk(frontier goals achieved) - trunk(frontier goals not achieved),
             per cluster, EMA'd; gated by achieved-rate uplift steered-vs-control

Remaining knobs are DOSES, not task content: band quantiles [0.2,0.8], alpha,
cluster count k, and the heads' own HER horizons (car 20 steps, ball 90 - the
self-model's definitions of "reachable"). The outcome scale has NO knob: all
attainment comparisons are within-population quantiles.

This module is the python mirror of the planned C++; semantics must match.
"""

import numpy as np
import torch

# Per-head constants: the self-model's own definitions (PPOLearnerConfig +
# Learner.cpp fnAppendAchieved)
HEADS = {
    "car": {"psi": "REACH_PSI_CAR", "ach": "ach_car", "window_steps": 20},
    "ball": {"psi": "REACH_PSI_BALL", "ach": "ach_ball", "window_steps": 90},
}

BAND = (0.2, 0.8)
RHO_K = 8


@torch.no_grad()
def psi_embed(models, head, goals):
    """L2-normalized psi embeddings for [n,6] goals (numpy in, numpy out)."""
    psi = models[HEADS[head]["psi"]]
    out = []
    g = torch.as_tensor(np.asarray(goals, np.float32))
    for i in range(0, len(g), 8192):
        e = psi(g[i:i + 8192])
        out.append((e / e.norm(dim=-1, keepdim=True).clamp_min(1e-6)).numpy())
    return np.concatenate(out) if out else np.zeros((0, 128), np.float32)


@torch.no_grad()
def phi_rows(models, h2_rows, mask_rows, k=RHO_K):
    """Mean-over-K-uniform-valid-actions phi embedding per row, L2 pieces kept:
    returns [n, k, 128] normalized (live rho parity: mean of cosines, so we keep
    the k pieces and average the dot products, not the embeddings)."""
    phi = models["REACH_PHI"]
    h2 = torch.as_tensor(h2_rows, dtype=torch.float32)
    masks = torch.as_tensor(mask_rows)
    maskF = masks.float().clamp_min(1e-9)
    acts = torch.multinomial(maskF, k, True)
    trunk_rep = h2.repeat_interleave(k, 0)
    onehot = torch.nn.functional.one_hot(acts.flatten(), masks.shape[1]).float()
    sa = phi(torch.cat([trunk_rep, onehot], -1))
    sa = sa / sa.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return sa.view(h2.shape[0], k, -1).numpy()


def episode_index(rec):
    episode = rec["episode"]
    ep_rows = {int(e): np.flatnonzero(episode == e) for e in np.unique(episode)}
    row_pos = np.empty(len(episode), np.int64)
    for rows in ep_rows.values():
        row_pos[rows] = np.arange(len(rows))
    return ep_rows, row_pos


def bank_nn_stats(bank):
    """Descriptive only: nearest-neighbor distances among the agent's achieved
    goals (the outcome manifold's granularity), for the census report."""
    g = bank["goals"]
    d2 = ((g[:, None, :] - g[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    return np.sqrt(d2.min(-1))


def build_bank(rec, head, rng, size=512):
    """The agent's achieved-goal bank: uniform sample of its own outcomes."""
    n = len(rec["slot"])
    idx = rng.choice(n, min(size, n), replace=False)
    return {"goals": rec[HEADS[head]["ach"]][idx].copy(), "rows": idx,
            "episodes": rec["episode"][idx].copy()}


def cluster_bank(models, head, bank, k=6, rng=None):
    """Emergent structure: k-means in the agent's own psi geometry. The cluster
    REPRESENTATIVE is the bank goal nearest each centroid (a real achieved
    outcome, so its embedding is in-manifold)."""
    from sklearn.cluster import KMeans
    emb = psi_embed(models, head, bank["goals"])
    km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(emb)
    labels = km.labels_
    reps = np.empty(k, int)
    for c in range(k):
        sel = np.flatnonzero(labels == c)
        d = ((emb[sel] - km.cluster_centers_[c]) ** 2).sum(-1)
        reps[c] = sel[d.argmin()]
    return {"labels": labels, "rep_goal_idx": reps, "emb": emb, "k": k}


def mine_pairs(models, rec, head, bank, clusters, rng,
               max_rows=4000, goals_per_row=3, band=BAND):
    """Frontier pairs: (state row, bank goal from ANOTHER episode) with rho in the
    band; scored MODEL-FREE by ATTAINMENT = -min distance (raw goal space) between
    the goal and the same player's future achieved states within the head's own
    HER horizon. Returns arrays: row, goal_idx, cluster, rho, band_pos, attain."""
    ep_rows, row_pos = episode_index(rec)
    npl = 2 * rec["ppt"]
    W = HEADS[head]["window_steps"]

    n = len(rec["slot"])
    cand = []
    for r in range(0, n, max(1, n // max_rows)):
        rows_ep = ep_rows[int(rec["episode"][r])]
        if row_pos[r] + W * npl < len(rows_ep):
            cand.append(r)
    cand = np.array(cand, int)
    if len(cand) == 0:
        return None

    goal_emb = psi_embed(models, head, bank["goals"])
    phi = phi_rows(models, rec["h2"][cand].astype(np.float32), rec["masks"][cand])

    rows_out, goals_out = [], []
    for i, r in enumerate(cand):
        pool = np.flatnonzero(bank["episodes"] != rec["episode"][r])
        if len(pool) < goals_per_row:
            continue
        for gi in rng.choice(pool, goals_per_row, replace=False):
            rows_out.append(i)
            goals_out.append(gi)
    rows_out = np.array(rows_out, int)
    goals_out = np.array(goals_out, int)

    # rho = mean over K action pieces of cos(phi, psi(goal))
    rho = np.einsum("nkd,nd->n", phi[rows_out], goal_emb[goals_out]) / phi.shape[1]
    lo, hi = np.quantile(rho, band[0]), np.quantile(rho, band[1])
    band_pos = np.where(rho < lo, 0, np.where(rho <= hi, 1, 2))

    # Attainment (model-free, continuous): how close the future actually got
    ach_all = rec[HEADS[head]["ach"]]
    attain = np.full(len(rows_out), -np.inf, np.float32)
    for j in range(len(rows_out)):
        r = cand[rows_out[j]]
        rows_ep = ep_rows[int(rec["episode"][r])]
        q = row_pos[r]
        fut = rows_ep[q + npl: q + (W + 1) * npl: npl]
        if len(fut):
            d = np.linalg.norm(ach_all[fut] - bank["goals"][goals_out[j]], axis=-1)
            attain[j] = -d.min()

    keep = np.isfinite(attain)
    return {"row": cand[rows_out[keep]], "goal_idx": goals_out[keep],
            "cluster": clusters["labels"][goals_out[keep]],
            "rho": rho[keep], "band_pos": band_pos[keep], "attain": attain[keep]}


def derive_cluster_direction(h2_all, pairs, cluster, rng, min_pairs=100):
    """Matched top-vs-bottom ATTAINMENT-tercile difference of trunk means on
    IN-BAND pairs of one emergent cluster. Matching on rho quintiles (self-model
    units); terciles computed within each rho bin so 'attained more' is never
    confounded with 'was more reachable'."""
    sel = (pairs["band_pos"] == 1) & (pairs["cluster"] == cluster)
    idx = np.flatnonzero(sel)
    if len(idx) < 2 * min_pairs:
        return None, 0.0, 0
    rho = pairs["rho"][idx]
    att = pairs["attain"][idx]
    edges = np.quantile(rho, np.linspace(0, 1, 6)[1:-1])
    bins = np.digitize(rho, edges)
    sel_a, sel_n = [], []
    for b in np.unique(bins):
        in_bin = idx[bins == b]
        if len(in_bin) < 6:
            continue
        a_bin = pairs["attain"][in_bin]
        lo, hi = np.quantile(a_bin, [1 / 3, 2 / 3])
        top = in_bin[a_bin >= hi]
        bot = in_bin[a_bin <= lo]
        m = min(len(top), len(bot))
        if m:
            sel_a += list(rng.choice(top, m, replace=False))
            sel_n += list(rng.choice(bot, m, replace=False))
    if len(sel_a) < min_pairs:
        return None, 0.0, 0
    Ha = h2_all[pairs["row"][np.array(sel_a)]].astype(np.float32)
    Hn = h2_all[pairs["row"][np.array(sel_n)]].astype(np.float32)
    v = Ha.mean(0) - Hn.mean(0)
    v /= max(np.linalg.norm(v), 1e-8)
    with np.errstate(all="ignore"):
        proj = h2_all.astype(np.float32) @ v
    assert np.isfinite(proj).all()
    return v, float(np.std(proj)), len(sel_a)


class ClusterSteeredPolicy:
    """Steer along an emergent cluster's direction on rows whose rho toward the
    cluster's representative goal sits in the batch band. No slices, no task
    predicates - the self-model read is the whole gate."""

    def __init__(self, models, head=None, rep_goal=None, v=None, alpha=0.0,
                 scale=1.0, band=BAND, k=RHO_K):
        from load_checkpoint import PulsarPolicy
        self.pol = PulsarPolicy(models)
        self.models = models
        self.alpha, self.scale, self.band, self.k = alpha, scale, band, k
        self.v = None if v is None else torch.as_tensor(v, dtype=torch.float32)
        self.g = None
        if head is not None and rep_goal is not None:
            self.g = torch.as_tensor(psi_embed(models, head, rep_goal[None])[0])

    @torch.no_grad()
    def act(self, obs, masks):
        h1, h2 = self.pol.trunk_forward(obs)
        h2_pol = h2
        if self.v is not None and self.alpha != 0.0 and self.g is not None:
            n = h2.shape[0]
            gate = torch.zeros(n)
            if n >= 16:
                maskF = masks.float().clamp_min(1e-9)
                acts = torch.multinomial(maskF, self.k, True)
                trunk_rep = h2.repeat_interleave(self.k, 0)
                onehot = torch.nn.functional.one_hot(acts.flatten(), masks.shape[1]).float()
                phi = self.models["REACH_PHI"]
                sa = phi(torch.cat([trunk_rep, onehot], -1))
                sa = sa / sa.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                rho = (sa @ self.g).view(n, self.k).mean(-1)
                lo = torch.quantile(rho, self.band[0])
                hi = torch.quantile(rho, self.band[1])
                gate = ((rho >= lo) & (rho <= hi)).float()
            h2_pol = h2 + gate.unsqueeze(-1) * (self.alpha * self.scale) * self.v
        return h2, self.pol.sample_actions(h2_pol, masks)


def describe_cluster(rec, bank, clusters, c, head):
    """DESCRIPTIVE ONLY (for the human reading the report; the system never uses
    this): mean raw goal vector of a cluster, denormalized."""
    sel = clusters["labels"] == c
    g = bank["goals"][sel].mean(0)
    if head == "ball":
        return (f"ball pos ~({g[0]*4096:.0f},{g[1]*6000:.0f},{g[2]*2044:.0f})uu "
                f"vel ~({g[3]*6000:.0f},{g[4]*6000:.0f},{g[5]*6000:.0f})uu/s")
    return (f"car-local ball ~({g[0]*2300:.0f},{g[1]*2300:.0f},{g[2]*2300:.0f})uu "
            f"relvel ~({g[3]*2300:.0f},{g[4]*2300:.0f},{g[5]*2300:.0f})uu/s")
