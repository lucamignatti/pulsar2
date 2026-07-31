# The Optimism Ladder: Four Critics from What Happened to What May Be Possible

**Luca Mignatti**

*Preprint. Extends and completes the program of `COMPOSITION_CRITIC.md`; self-contained. Code and
experiment logs: [repository URL to be added upon release].*

---

## Abstract

Deep reinforcement learning agents cannot learn behaviors they never perform, and they cannot be safely encouraged toward behaviors whose value they have no grounds to believe in: optimism about never-executed action sequences is unfalsifiable under an on-policy data stream, and unfalsifiable optimism, trained through a bootstrap, diverges. We present a complete architecture for *graded, falsifiable optimism* in an on-policy learner: a **ladder of four value estimates** over the same state, ordered by the epistemic strength of their training targets. $V$ estimates what current behavior reliably collects (mean return). $V^{\text{exp}}$ estimates what current behavior sometimes achieves (return-level expectile). $V^\dagger$, the **composition critic**, estimates what is *provably* achievable — the upper envelope of values reachable by chaining steps the agent has executed somewhere, across episodes, arenas, and agents. The first three rungs are operators over the record of what happened, which is the source of their falsifiability and equally of their shared ceiling: a state whose value the environment makes available, but whose pieces have never been assembled anywhere in the record, is invisible to all of them. The fourth rung, the **geometric critic** $V^\diamond$, estimates what may be *possible*: it is the fixed point of a discrete Hamilton–Jacobi–Bellman equation whose only environment knowledge is a conditional *variance* statistic of one-step displacements — a quantity that never predicts where the agent will go, only how far it could, and therefore is not a model of the dynamics and does not reopen the divergence route. The mechanism is model-free in the operative sense, search-free, and costs one extra input-gradient per minibatch over three small MLPs. On an oracle-scored probe set $V^\diamond$ attains $\rho(V^*) = 0.698$ against the ordinary critic's 0.665 and the composition critic's 0.577, and is the only rung of the four whose estimate tracks what the environment *allows* better than what the policy *does*. Actuated as a permanent potential beside the composition critic's — the two normalized and summed at constant weight — the full ladder is the best arm on ball interaction in every training bucket of an aerial-discovery benchmark and on airborne interaction in three of four, reaching 4.7× the composition critic's airborne rate over the first 3M steps; a paired test shows the constant mix beats a schedule that retires $V^\diamond$, so the ladder deploys with four rungs, not three plus a bootstrap. We report the failure maps for both halves of the program — spiral, ratchet, and avoidance for the composition rungs; seven falsified designs for the geometric rung — and a methodological finding, the **dose confound**, that silently invalidated eight of our own comparisons before it was caught. The four-rung ladder is deployed from step zero of a production-scale Rocket League self-play run.

---

## 1 Introduction

Model-free policy-gradient methods learn from what they do. For a class of behaviors this is an *acquisition* limitation, not a sample-efficiency one: if a behavior requires a coordinated multi-step action sequence, and the probability of first completing that sequence under the current policy is effectively zero, the behavior contributes no reward, receives no gradient, and is never learned. More environment steps do not help; the gradient toward the behavior does not exist.

Throughout, we use **conduct** for such a target: a temporally extended behavior, as distinct from a state the agent might merely visit. Rocket League¹ supplies crisp instances at two different depths, and the difference between them organizes this paper. An *aerial* — jumping, tilting the car nose upward, boosting through the air to intercept a high ball — essentially never occurs whole under a young policy, but its *pieces* do: the policy has jumped thousands of times, has boosted while airborne in scattered fragments, has occasionally touched low balls while off the ground. An *air dribble* — carrying the ball up a wall and keeping it aloft with repeated airborne touches — poses a harder problem: every feature of its states is individually common in the record, yet the *conjunction* may appear nowhere, so there is no chain of executed steps to assemble. The first conduct is unperformed but **composable**; the second is unperformed and **uncomposed**.

¹ *Rocket League is a physics-based car-soccer game. Cars drive, jump, and carry a finite "boost" resource whose sustained thrust enables controlled flight. Distances are in Unreal units (uu); the arena ceiling sits at ≈2044 uu.*

The standard practical remedies move the problem into the environment: curricula, reset-state distributions, demonstrations, shaped conduct-specific rewards. They work, and we reproduce a spawn curriculum reaching 0.96 aerial conversion, but they require a designer to name each skill in advance and they couple the algorithm to environment instrumentation. We wanted the opposite: a purely architectural mechanism — identical environments everywhere, no resets, no probes, no world models, no search, MLP-scale cost — under which the policy itself comes to intend toward its own frontier.

Our answer is a **ladder**: four value estimates of the same state, ordered by the epistemic strength of what their training targets assert.

> | rung | estimates | grounded by |
> |---|---|---|
> | $V$ | what we normally get | expectation over executed transitions |
> | $V^{\text{exp}}$ | what we get when it goes well | upper expectile of realized returns |
> | $V^\dagger$ | what we **can** get | upper envelope over executed transitions |
> | $V^\diamond$ | what may be **possible** | the environment's local geometry |

The discipline that makes the ladder safe is an information-accounting principle established by measurement in the first half of this program:

> **The execution principle.** Information about the outcome of an action sequence that has never been executed anywhere can come only from (i) a model of the dynamics, (ii) explicit search, or (iii) an attempt. Optimism not grounded in one of these is unfalsifiable, and unfalsifiable optimism, trained through a bootstrap, diverges (§4.2).

The first three rungs obey the principle by construction: they are operators over the record. $V^\dagger$ in particular is optimistic only about *compositions* — chains of steps that each really happened — so its optimism is falsifiable piecewise, and it credits the aerial the moment the aerial's fragments exist (§5). But the principle also fixes the three rungs' shared ceiling, stated in the contrapositive: *a conduct containing a step executed nowhere is invisible to every record-operator until entropy supplies that link once.* The air dribble sits behind exactly this wall.

The fourth rung therefore asks for something different in kind, and the execution principle appears to forbid it: under constraints excluding models and search, where can information about the *never-assembled* come from? **The escape is that a conditional variance is not a model.** The one-step displacement spread $\Sigma(s)$ never predicts where the agent will go — it says only how far, in each direction of state space, single actions have ever moved it from states like this. A value function constrained to be *self-consistent against that reachability* — each state worth its local reward plus the best value its one-step reachable set contains — is an estimate of $V^*$ obtained without ever asking what any particular unexecuted action does. That constraint is a discrete Hamilton–Jacobi–Bellman equation, and solving it with a neural field costs one input-gradient per minibatch (§6).

**Contributions.**

1. **The ladder as an architecture** (§5–§7): four critics over one state, each rung's optimism grounded one level deeper — expectation, upper tail, executed composition, environment geometry — with the upper two actuated as potential-based fields, so policy invariance is exact and the whole assembly adds three small MLPs and two value heads to an unmodified PPO learner.
2. **The composition critic** $V^\dagger$: online expectile TD over executed transitions only, stabilized against the *expectile ratchet* by twin heads with minimum-in-target; its headroom field $H = \mathrm{relu}(V^\dagger - V)$ actuated with the *seek* sign (§5). Carried over from the companion program with its failure map (spiral, ratchet, avoidance) summarized in §4.
3. **The geometric critic** $V^\diamond$ (§6): a PDE-constrained value field whose only environment inputs are a fitted local reward $\hat r(s)$ and a conditional displacement spread $\Sigma(s)$; no rollouts, no search, no forward prediction. On an oracle-scored probe set it is the only rung whose value correlates more with what the environment allows than with what the policy does ($\rho(V^*) = 0.698$ vs. $\rho(V^\pi)$; every record-operator inverts this ordering), with the synthesis — not either component — carrying the signal.
4. **Permanence of the fourth rung** (§7.3): a paired experiment showing a constant mixing of the geometric and composition potentials beats a crossfade that retires $V^\diamond$ as $V^\dagger$ matures, in 6 of 8 step-bucket cells with no collapse — overturning our own earlier inference (from a dose-confounded run) that the permanent shape fails.
5. **Two methodological findings** with reach beyond this system: the **dose confound** — σ-matched potential injections at equal nominal β can differ 4× in delivered dose after clamping, which invalidated eight of our comparisons before it was measured (§8) — and the **order/magnitude split**: regression-fit value functions *rank* never-seen states correctly while saturating at their training ceiling, which is why no regression can constitute the fourth rung (§4.4).

---

## 2 Related work

**Offline RL and stitching.** Implicit Q-Learning [Kostrikov et al., 2022] introduced expectile regression on TD targets to approximate a maximum over dataset-supported actions; the resulting trajectory *stitching* is central to offline RL. The composition rung moves this estimator online — where the naive form ratchets (§4.3) — and uses it not for policy extraction but as an exploration field. Conservative methods [Kumar et al., 2020] penalize exactly the optimism the ladder is built to supply.

**Optimism and exploration.** State-novelty drives (ICM [Pathak et al., 2017]; RND [Burda et al., 2019]) reward visiting unfamiliar states; the acquisition problem is unattempted *conducts* from familiar states, which state novelty cannot see. Optimistic initialization and UCB-style bonuses inject optimism into values; §4.2 measures why the bootstrapped form diverges at scale unless grounded. Go-Explore and reset curricula [Ecoffet et al., 2021] attack acquisition via environment control, which our constraint set forbids.

**Continuous-time control and HJB methods.** The geometric rung's defining equation is the stochastic Hamilton–Jacobi–Bellman equation of continuous-time optimal control [Fleming & Soner, 2006; Doya, 2000], with the control-affine max resolved in closed form as a dual norm. Solving PDEs with neural residual losses is the physics-informed network program [Raissi et al., 2019]; value-gradient methods [Heess et al., 2015] and linearly-solvable formulations [Todorov, 2009] likewise place the value gradient at the center of control. To our knowledge the combination here is new in kind for exploration: the HJB residual is used not to *solve* a known system but to *transduce* a learned reachability statistic into an optimism field, precisely because the statistic underdetermines the dynamics (§10).

**Model-based imagination and search.** Dreamer-class agents [Hafner et al., 2023] and MuZero [Schrittwieser et al., 2020] obtain counterfactual competence from a learned model plus search or dreaming. The companion program verified both the premise (a one-step model learns flight physics from fragments) and the relocation of the problem (the imagination actor faces the same exploration barrier inside the dream); the fourth rung's escape is to keep the *reachability* content of a model while discarding its *predictive* content, which is the part that feeds divergence and the part that costs compute.

**Potential-based shaping.** Actuation throughout is classical PBRS [Ng et al., 1999]. The empirical content added here: the sign-and-field-shape analysis (closure potentials on strong peaked fields teach avoidance, §4.3), the gap-vs-level form asymmetry between the two upper rungs (§7.1), and the dose confound (§8).

**Rocket League systems.** The deployment target is a C++ self-play league trainer in the population-based lineage [Jaderberg et al., 2019; Berner et al., 2019], on the RLGym/RocketSim ecosystem. Prior aerial acquisition in this ecosystem relies on shaped event rewards and drill state-setters; to our knowledge no public system carries a graded-optimism critic ladder.

---

## 3 Preliminaries and problem setting

### 3.1 Setup and notation

Standard on-policy actor–critic training: PPO [Schulman et al., 2017] with GAE [Schulman et al., 2016] on an MDP with states $s$, actions $a$, rewards $r$, discount $\gamma$, boundary indicator $d$ (nonzero at terminals and truncations). $V$ is the ordinary critic; $A_i$ are GAE advantages. For $\tau \in (0,1)$, the $\tau$-expectile of $Y$ is $\arg\min_m \mathbb{E}[|\tau - \mathbb{1}[Y<m]|(Y-m)^2]$ [Newey & Powell, 1987]; as $\tau \to 1$ it approaches the supremum of the support.

### 3.2 Constraints

- **C1 (homogeneity):** all environment instances identical; no probe, drill, or reset instances;
- **C2 (no environment design):** no curricula, demonstrations, banked resets, or retrospective success memories;
- **C3 (no models, no search):** no learned dynamics prediction, no imagination rollouts, no tree search;
- **C4 (MLP-scale cost):** collection and deployment cost unchanged; learn-phase overhead small;
- **C5 (serendipity floor):** baseline entropy regularization is never reduced or gated. Guidance adds attention; it must never veto undirected exploration.

C3 forbids *prediction* of the consequences of particular actions. It does not forbid *summary statistics* of executed transitions — every rung of the ladder, including the fourth, is trained on executed data only. The fourth rung's statistic is chosen to be exactly the one that cannot be rolled forward (§6.1).

### 3.3 Benchmark: aerial discovery from cold starts

The offline testbed is a single-agent RocketSim task distilled from the deployment game (Appendix A). One car, one ball, 48 parallel arenas; every episode begins cold — car on the ground rolling forward with full boost, ball placed uniformly in a 3-D box mostly unreachable without flight. Reward is deliberately generic (+1 any touch, small approach and ball-toward-goal shaping, +10 score); **no aerial-specific term exists anywhere**. Indicators, all logging-only: **touch** (any ball contact per episode), **air** (airborne contact, car above 300 uu), **hi** (airborne conversion of high spawns, $z \ge 900$ uu). Vanilla PPO learns ground play and converts zero high aerials at every budget run.

---

## 4 The measured failure map

Every element of the ladder is the negation of a measured failure. We compress the companion program's map (§4.1–§4.3, full detail in that paper) and give the geometric rung's map in §4.4–§4.5, with detail in §8. All cells are 2 seeds unless noted.

### 4.1 Environment-side mechanisms work, and fix the constraints

A hand-designed spawn curriculum reaches 0.86–0.96 high-aerial conversion — the skyline. A reset-to-banked-states variant that avoids difficulty design scores 0.244 in aggregate and **0.005** on cold starts: the scaffold short-circuits the approach phase, which *is* the skill. Effective, designer-coupled, and non-transferring where the scaffold cheats: these observations fix C1–C2.

### 4.2 Unfalsifiable optimism diverges; execution grounds it and bounds it

A Q-function over the discrete action table trained with optimality backups — a max over *all* actions, untried ones valued by generalization alone — spiraled (mean $V_{\max}$ ≈ 42.6/38.3 against best-ever returns ≈ 30–36) despite distributional heads, double-Q selection, target networks, and clamps; TQC-style truncation delayed the spiral to ≈16M steps and did not prevent it. The cause is epistemic: argmax actions at most states are never executed, so their values are never corrected, and the bootstrap compounds selection bias. Converting a small fraction of arenas into ε-greedy probes that *execute* the argmax fully stabilized the same estimator — and revealed the bound: the probe's greedy chain breaks where the critic has no data, so execution-calibrated ceilings are *pessimistic on deep conducts* (the probe-assisted field read ≈0 headroom at high balls where ≥0.9 conversion is provable). Grounding by execution both rescues and limits optimism. This is the reason the ladder's rungs are graded rather than replaced by one sufficiently clever estimator.

### 4.3 The composition rung's two hazards: ratchet and avoidance

Restricting optimism to executed support removes the unfalsifiable-action channel but not the instability: single-head online expectile TD at $\tau = 0.9$ inflates headroom at easy states 0.3 → 11.8 over 20M steps with no behavioral conversion — the **expectile ratchet**, a positive feedback between the asymmetric loss and the target network's inherited inflation. Twin heads with minimum-in-target at $\tau = 0.75$ hold $V^\dagger$ bounded for 40M+ steps. Actuation has a sign hazard: a *closure* potential $\Phi = -H$ (paid to reduce the gap) is safe on the small, diffuse knowing–doing gap of $V^{\text{exp}}$ but catastrophic on the strong, peaked composition field — touch collapsed to 0.014/0.027 vs. 0.244 for PPO — because the cheapest way to reduce a peaked potential along a trajectory is to walk away from the peak. The *seek* form $\Phi = +H$ inverts the geometry. *Seek strong peaked fields; close weak diffuse ones.*

### 4.4 No regression can be the fourth rung: order extrapolates, magnitude does not

The naive route to "what may be possible" is a value regression trusted to generalize into the unseen. We measured the exact failure. With the high-value region deleted from training, an MLP fit by regression still *ranks* never-seen states correctly ($\rho \approx 0.66$) but predicts 0.098 where the truth is 0.368 — it saturates at its training ceiling. The fourth rung's whole content is "worth **more** than anything in the record," and that is the one sentence a regression cannot say. A log-additive (multiplicatively factored) model *can* extrapolate magnitude (ratio 1.06 at up to 16× its training maximum) — but only when the target genuinely factorizes, and a single-feature control shows no benefit. The conclusion is structural: the fourth rung must *derive* magnitude from something other than fitted values. The HJB route derives it from geometry and never regresses the value at all.

### 4.5 Seven falsified designs for the geometric rung

Summarized; each is a measured arm on the §3.3 benchmark (§8 for the two with standalone lessons):

| # | design | outcome |
|---|---|---|
| 1 | depth-1 optimistic backup through a latent world model | no better than the same architecture with optimism off |
| 2 | that ablation vs. the world model deleted entirely | **identical** — the gain was the encoder, not the model |
| 3 | PBRS on the raw $V^\diamond$ level | neutral vs. PPO |
| 4 | rank-normalized potential + Lipschitz penalty | catastrophic — the penalty fights the residual, which *wants* $\lVert\nabla V\rVert$ large |
| 5 | alignment-residual drive (not potential-based) | best early air of any arm, then degrades |
| 6 | ZPD-bounding the geometric gap by the proven gap (hard min) | worst of all — the clip reintroduces field roughness |
| 7 | multiplicative attention weighting on advantages | neutral alone; destroys the mix's gain; ESS 0.44–0.48 |

---

## 5 Rungs one to three: operators over the record

The first three rungs are carried from the companion program; we state them in ladder form.

**Rung 1 — $V$ (known).** The ordinary critic: mean-squared TD regression; estimates $V^\pi$.

**Rung 2 — $V^{\text{exp}}$ (when it goes well).** Expectile regression against *realized returns*: the upper tail of current behavior. Its gap to $V$ is the knowing–doing gap — what the policy sometimes achieves minus what it reliably achieves — small, spatially diffuse, and safe to close with a closure potential (§4.3).

**Rung 3 — $V^\dagger$ (can).** Twin heads $V^\dagger_1, V^\dagger_2$; one-step targets over the fresh buffer, executed transitions only:

$$y_i = r_i + \gamma(1-d_i)\,\min\!\big(V^\dagger_{1,\text{tgt}}(s_{i+1}),\,V^\dagger_{2,\text{tgt}}(s_{i+1})\big), \tag{1}$$

both heads trained by expectile regression at $\tau = 0.75$ with clamped targets. Each executed transition contributes an *inequality* — from $s$, at least $r + \gamma V^\dagger(s')$ was achievable — and high-expectile TD approximates the upper envelope consistent with all of them, chaining across trajectories wherever function approximation identifies intermediate states. The min-in-target is the anti-ratchet (§4.3). The unit of required luck falls from "a completed conduct" to "each step, somewhere, ever."

**Fields and actuation.** The headroom field $H(s) = \mathrm{relu}(\min(V^\dagger_1, V^\dagger_2)(s) - V(s))$ is zero where the policy collects what is provable and positive exactly where the record exceeds behavior — a self-retiring zone-of-proximal-development field. It is actuated by the seek potential $\Phi = +H$: $g_i = \gamma(1-d_i)H(s_{i+1}) - H(s_i)$, centered, σ-matched to a fraction β of advantage scale, clamped to $\pm 3\sigma_{\text{ext}}$. On the benchmark this moves high-aerial conversion off an exact zero (0.089/0.043 at 40M vs. 0.000 for PPO) and outperforms the probe-assisted estimator that violates C1. At production scale the composition rungs have run for multi-billion-step lineages; offline probes at 10.65B steps of the conformant deployment found $V^\dagger$ the **only** rung crediting a certified never-completed aerial sequence, at 75% of probed states — the stitching property, live, at scale.

**The ceiling.** All three rungs are operators over executed transitions. The companion paper's own contrapositive fixes the boundary: a conduct with a globally unexecuted step is invisible until entropy supplies it. What lies above the third rung is not a better operator over the record; it is a different information source.

---

## 6 Rung four: the geometric critic

### 6.1 The escape from the execution principle

The execution principle admits three sources for information about the unexecuted: a model, search, or an attempt. C3 forbids the first two as *predictive* machinery. The observation that opens the fourth rung: **the divergence route of §4.2 runs through prediction, not through statistics.** What made untried-action optimism unfalsifiable was the claim "action $a$ from state $s$ leads to good state $s'$" — a claim about a particular consequence, never checked. A conditional second moment makes no such claim. Define, over executed transitions only,

$$\Sigma(s) = \mathrm{diag\text{-}spread}\big[\,s' - s \mid s\,\big], \tag{2}$$

the per-dimension standard deviation of the one-step displacement from states like $s$, fit by maximum likelihood (a small MLP emitting per-dimension log-σ). $\Sigma$ never says where the agent will go. It says how far, along each coordinate, single decisions have ever moved it from here. It is falsified the ordinary way — by more executed data — and it cannot be rolled forward: it contains no mapping from actions to outcomes, so there is nothing to search over and no dream to get lost in. It is *reachability without prediction*.

### 6.2 The defining equation

If from state $s$ one decision can move the state anywhere in the ellipsoid $\{s + \delta : \delta^\top \Sigma(s)^{-2} \delta \le 1\}$, then the best value locally available to a smooth value field $V^\diamond$ is along its gradient, and the maximum of $\nabla V^\diamond \cdot \delta$ over the ellipsoid has a closed form — the dual norm:

$$\max_{\delta^\top \Sigma^{-2} \delta \,\le\, 1} \nabla V^\diamond(s) \cdot \delta \;=\; \big\lVert \nabla V^\diamond(s) \big\rVert_{\Sigma}\;=\;\sqrt{\textstyle\sum_j \big(\partial_j V^\diamond\big)^2\, \Sigma_j(s)^2}. \tag{3}$$

Self-consistency of the field — each state worth its local reward plus the discounted best of what one step can reach — is then a discrete-time stochastic Hamilton–Jacobi–Bellman equation [Fleming & Soner, 2006; Doya, 2000]:

$$(1 - \gamma)\, V^\diamond(s) \;=\; \hat r(s) \;+\; \gamma\, \big\lVert \nabla V^\diamond(s) \big\rVert_{\Sigma(s)}, \tag{4}$$

where $\hat r(s)$ is a small reward model fit on arrival states. $V^\diamond$ is trained by squared residual of Eq. 4 over on-policy states — a physics-informed loss [Raissi et al., 2019] — which costs exactly one input-gradient per minibatch (`create_graph=true` for the double-backward). No value regression occurs anywhere: magnitude emerges from the accumulation of $\hat r$ along geometrically-reachable paths, which is what lets the field exceed every value in the record (§4.4).

**Why this sees the air dribble.** $\Sigma$ near the wall knows that single decisions move the car and ball upward and together — from wall-drive fragments, jump fragments, airborne-touch fragments, none of them air dribbles. $\hat r$ knows ball-near-goal states pay. The field equation chains these local facts through state space: value flows backward along any corridor of positive reachability, whether or not any trajectory ever traversed the corridor. Where $V^\dagger$ chains *executed transitions*, $V^\diamond$ chains *reachability itself*.

**Two deliberate absences.** *No advection term.* The full HJB includes a drift term $\nabla V \cdot \mu(s)$ with $\mu = \mathbb{E}[s' - s \mid s]$. But $\mu$ is estimated under the *policy*, so including it re-imports the habit the rung exists to see past — measured, drift-on loses the beats-habit property at every σ-scale (0.634 vs. 0.698, §6.4). The spread stays; the direction goes. *No sliding window.* $r(s)$ and $\Sigma(s)$ are properties of the environment and are stationary; only *where we sample them* moves. Fit on the on-policy window they forget regions the policy has left — measured as the HJB residual growing 100–250× over training, flat to ≈6M steps then runaway, exactly tracking where every actuation variant stopped helping. A uniform reservoir over all data ever seen holds the residual ≈13× lower at the runaway point and is the correct estimator for a stationary target. $V^\diamond$ itself trains on *current* states (the field is queried where the policy lives); only the world-facing fits use the reservoir.

### 6.3 Algorithm and cost

**Algorithm 1. The four-rung ladder (one PPO iteration; rungs 1–3 as in §5).**

```
Nets: sigma-net Σ_ψ (obs → 2·obs: per-dim mean, log-σ; mean unused by the loss),
      reward-net r̂_ω (obs → 1), field V◇_χ (obs → 1). All small MLPs on raw obs.
Reservoir R: uniform sample over all (s, s', r) ever seen (fixed capacity).

1. Collect buffer B with π_θ (collection unchanged; no rung is consumed at collection).
2. Rungs 1–3: train V, V_exp, twin V† as in §5; compute H = relu(min V† − V).
3. Fold B into R. On minibatches of R: fit Σ_ψ by Gaussian NLL of (s' − s);
   fit r̂_ω by MSE on arrival states. (Stationary targets; reservoir prevents forgetting.)
4. On minibatches of current states s ∈ B:
      g = ∇_s V◇(s)                       (input-gradient, create_graph)
      resid = (1−γ)V◇(s) − r̂(s) − γ·sqrt(Σ_j g_j² Σ_j(s)²)
      minimize resid²; extra V◇-only passes against the frozen fits.
5. H◇ = relu(scale(V◇) − V) (affine-matched to V's scale; §7.1). Potentials:
      Φ_mix = (1−w)·unit(H◇) + w·unit(V†),  w constant (§7.3).
6. g_i = γ(1−d_i)Φ_mix(s_{i+1}) − Φ_mix(s_i); center; σ-match to β; clamp ±3σ_ext;
   A_i ← A_i + g_i. Verify delivered dose via the injection-to-advantage σ-ratio (§8).
7. Standard PPO update; entropy bonus untouched (C5).
```

**Cost (C4).** Three small MLPs and one double-backward per minibatch. Testbed throughput: 13.6k steps/s against the composition arm's 14.5k and PPO's 20.4k on the same machine — the same order as the incumbent, no rollouts, no search. Nothing is added at collection or deployment.

### 6.4 Measurement: does the field track the environment or the habit?

The fourth rung's claim is that it reads value the *environment* makes available rather than value the *policy* already collects. We score it against a ground-truth oracle on a fixed probe set spanning mastered, composable, and never-assembled regions. The oracle is receding-horizon MPC (re-planned CEM against the simulator) — chosen after measuring that open-loop CEM finds 0 of 24 aerials and would have scored the probe against the search's own weakness; the receding-horizon form converts 8/10. Correlations of each rung's estimate with the oracle's $V^*$ and with the on-policy $V^\pi$:

| rung | $\rho(V^*)$ | tracks environment over habit? |
|---|---|---|
| $V$ | 0.665 | no — $\rho(V^\pi) > \rho(V^*)$ |
| $V^{\text{exp}}$ | 0.66 | no |
| $V^\dagger$ | 0.577 | no |
| $V^\diamond$ | **0.698** | **yes — the only rung** |

Every record-operator correlates better with what the policy does than with what the environment allows — as it must, since the record *is* the policy's shadow. $V^\diamond$ inverts the ordering. Controls: with $\Sigma$ replaced by an isotropic constant the advantage vanishes (the anisotropy is the signal); with drift on, beats-habit is lost at every σ (0.634); neither $\hat r$ nor $\Sigma$ alone predicts $V^*$ ($\rho \le 0.31$) — the HJB synthesis, not either component, carries the signal. The gap form $\mathrm{relu}(V^\diamond - V)$ carries the actionable excess (excess-$z$ +0.201, best of any rung).

---

## 7 Actuation: the full ladder

### 7.1 Forms

Each upper rung is actuated in the form its own validation supports, and the asymmetry is empirical, not aesthetic. For $V^\diamond$ the **gap** works and the raw level is neutral (§4.5 #3): the level is high wherever the environment is rich, including where the policy already collects, so it cannot discriminate; the gap is self-retiring. For $V^\dagger$ the **level** works and the gap penalizes finishing (the companion seek-potential result). Both are functions of state alone, so PBRS validity is exact throughout:

$$\Phi_{\text{mix}}(s) \;=\; (1-w)\cdot \widehat{\mathrm{relu}\big(V^\diamond - V\big)} \;+\; w \cdot \widehat{V^\dagger}, \tag{5}$$

each term normalized to unit scale so $w$ is a genuine mix and not a hidden dose ramp, injected σ-matched at β with the ±3σ clamp.

### 7.2 The mix against every single-potential arm

Mean of 2 seeds, matched steps, corrected dose (§8):

**touch**

| arm | 0–3M | 3–6M | 6–9M | 9–12M |
|---|---|---|---|---|
| PPO | 0.0404 | 0.0955 | 0.2058 | 0.2477 |
| + $V^\dagger$ | 0.0362 | 0.1489 | 0.2062 | 0.2599 |
| + $V^\diamond$ alone | 0.0608 | 0.1159 | 0.2061 | 0.2507 |
| **+ ladder mix** | **0.0735** | **0.1668** | **0.2400** | **0.2624** |

**air**

| arm | 0–3M | 3–6M | 6–9M | 9–12M |
|---|---|---|---|---|
| PPO | 0.0036 | 0.0099 | 0.0471 | 0.0797 |
| + $V^\dagger$ | 0.0033 | 0.0209 | 0.0589 | 0.0937 |
| + $V^\diamond$ alone | 0.0095 | 0.0211 | 0.0489 | 0.0747 |
| **+ ladder mix** | **0.0156** | **0.0326** | **0.0659** | 0.0894 |

Best on touch in all four buckets, on air in three of four. Against the composition critic alone: **4.7× air at 0–3M**, 1.6× at 3–6M, 1.12× at 6–9M — largest exactly where acquisition is hardest, at no measured late-game cost. Neither rung alone produces this: $V^\diamond$ alone fades late (it estimates $V^*$, most informative when the policy is worst), $V^\dagger$ alone is blind early (its record is empty). The ladder's rungs cover each other's regimes.

### 7.3 The rung is permanent

The natural reading of the regime split is a crossfade — ramp the mix from geometry to composition and retire $V^\diamond$ once $V^\dagger$ matures. We deployed that reading in an earlier draft, supported by one run in which a permanent both-potentials arm collapsed. That run, we then found, had executed at 4× the intended dose (§8); its collapse was evidence about the dose, not the shape. The clean re-test pairs the two mixing rules exactly — same device, same seeds, same corrected dose, only $w(t)$ differing:

**air**, seed means, paired runs:

| rule | 0–3M | 3–6M | 6–9M | 9–12M |
|---|---|---|---|---|
| crossfade $w: 0 \to 1$ over 2–6M | **0.0122** | 0.0203 | 0.0453 | 0.0753 |
| constant $w = 0.5$ | 0.0081 | **0.0293** | **0.0586** | **0.0862** |

With touch, the constant mix takes **6 of 8 cells** and collapses nowhere; the crossfade's one win (0–3M air) is mechanical — early it is pure geometry, while the constant mix dilutes geometry with a still-empty composition rung. Two consequences. The deployed ladder has **four rungs, permanently** — not three plus a bootstrap. And our own earlier claim that $V^\diamond$ "decays into noise" late is bounded: that was measured for $V^\diamond$ alone at full weight; at half weight beside the proven-gap rung it kept paying through the full budget. The `hi` metric remains near zero for all arms at this 12M budget (the companion results show it moving from 25M on); the early-bucket air separation is the leading indicator at this scale.

---

## 8 The dose confound

The most expensive error of the program, reported for reuse. σ-matching sets the standard deviation of the *centered potential difference* — not of the injection that survives the ±3σ clamp. Two potentials at the same nominal β can deliver very different actual doses if their tail shapes differ: at equal β = 0.15, the geometric potential delivered an injection-to-advantage σ-ratio of **1.22** against the composition potential's **0.31** — a 4× overdose — and **eight comparisons were run in the over-dosed regime before the ratio was measured**, including the one that wrongly retired the permanent-rung shape (§7.3). Correcting the dose (β = 0.04 for the geometric potential) turned a flat negative into the results of §7. The rule: *read the dose from the delivered σ-ratio, never from β*; the deployment ships a dedicated panel for exactly this quantity, and it reads 0.040 at nominal β = 0.04 in the live run — the confound does not recur at scale.

Two further standalone lessons from the geometric failure map (§4.5): the **encoder illusion** — a latent world model's apparent benefit survived deleting the *model* but not deleting the *encoder*, i.e., the gain was representation learning misattributed to imagination (arms #1–2); and the **smoothness trap** — regularizing $\lVert\nabla V^\diamond\rVert$ for stability is catastrophic *by construction* here, because Eq. 4's residual is minimized by gradients as large as $\Sigma$ can pay for; the field is *supposed* to be steep where reachability is rich (arm #4).

---

## 9 Deployment

The four-rung ladder is integrated in a production-scale Rocket League self-play trainer: C++ PPO league system, 1152-wide shared trunk, ≈90-entry action table, collection in the hundreds of thousands of steps per second. Rungs 1–3 were integrated in the companion program (heads on the shared trunk; reward reconstruction from GAE outputs via the identity $r^{\text{scaled}}_i = A_i - \gamma\lambda(1-d_i)A_{i+1} - \gamma(1-d_i)V_{i+1} + V_i$; single advantage-assembly injection site). The geometric rung adds three insertion points, mirroring that recipe:

1. **Nets + loss** — three independent MLPs on raw observations (no trunk: the rung cannot co-adapt the perception path on day one), {384, 384}; the HJB block takes the input-gradient via `torch::autograd::grad(..., create_graph=true)` in the learn-pass minibatch.
2. **Reservoir feed** — at learn-prep, using the same scaled-reward reconstruction so $\hat r$ is in the critic's units; capacity 200k, vectorized fold-in.
3. **Actuation** — Eq. 5 at the single advantage-assembly site, $w = 0.5$ constant, β = 0.04.

**Telemetry.** The residual of Eq. 4 is the health gate: it sat at ~10⁻⁴ while the mechanism worked offline and grew 100–250× when the world-facing fits drifted; a rising trend means the field is no longer $V^\diamond$ and the injection has become a persistent orthogonal push. Panels: `Geo/Residual`, `Geo/Rew Loss`, `Geo/V Mean`, `Geo/Reservoir Fill` (must saturate), `Geo/Mix W`, `Geo/H Geo Mean|P90`, and the delivered-dose ratio (§8).

**Status.** A fresh lineage carrying all four rungs from step zero began 2026-07-30 (the ladder is validated as a cold-start-critical mechanism — §7.2's largest wins are the earliest buckets — so it belongs to a lineage from its first step, not to a mid-run insertion). At the time of writing the run is hours old: the residual holds at 1.5×10⁻⁴, the reservoir is saturated, the delivered dose reads exactly nominal, and the mix weight is flat at 0.5. No at-scale behavioral claims are made here; the panels above are the pre-registered health criteria by which that run will be judged.

---

## 10 Analysis

**The ladder is an epistemic hierarchy, not an ensemble.** Each rung's training target asserts a strictly weaker claim than the one below it, grounded in a strictly wider evidence class: *this happened on average* (expectation over executed transitions); *this happened in the upper tail* (same transitions, asymmetric loss); *this is constructible from what happened* (executed one-step inequalities, chained); *this is reachable given how far single steps move* (second moments of executed displacements, chained through a field equation). At no rung does the system assert anything about a particular unexecuted action — which is why all four estimators are falsifiable by more data of the ordinary kind, and why the ladder does not reopen the divergence route of §4.2. The rungs are not four opinions to be averaged; they are four different questions whose *disagreements* carry the information: $V^{\text{exp}} - V$ is inconsistency, $V^\dagger - V$ is proven-but-uncollected, $V^\diamond - V$ is possible-but-unproven.

**Why a variance is not a model.** A model answers "what happens if I do $a$?" and thereby licenses search — and search over a learned model is exactly where unfalsifiable optimism re-enters (§4.2, and the imagination barrier of the companion program). $\Sigma$ answers only "how far do single decisions move the state from here, per direction?" There is no action index, so there is nothing to argmax over, no rollout to compose, and no compounding of model error — Eq. 4 references $\Sigma$ only pointwise. The cost of this weakness is over-coverage: the ellipsoid contains displacement *combinations* no single action achieves, so $V^\diamond$ over-claims where the true reachable set is far from ellipsoidal (§11). The ladder's structure absorbs this: $V^\diamond$'s claims are never trusted alone; they are one normalized potential beside a rung whose every claim is executed-proven.

**Why the fourth rung stays.** Our initial reading — geometry for the cold start, composition for maturity — treated the rungs as substitutes on one axis. The paired result of §7.3 says they are complements: even late, when $V^\dagger$'s envelope is tight over the *visited*, $V^\diamond$ keeps paying at half weight, because the environment keeps containing corridors the record has not yet entered, and because the normalized mix means the geometric term can only redirect attention, never overwhelm proven credit. A rung that reads a different information source does not expire when another rung matures; it expires when its information source is exhausted — and the environment's geometry outlasts any finite record.

**The division of labor across a run.** Early, the record is empty: $V^\dagger$ is silent and $V^\diamond$ supplies the only non-zero optimism — measured as the 4.7× early-air advantage. Mid-run, the frontier wave of the composition field (the companion's §6.2) and the geometric gap overlap; the mix pays for entering headroom that either grounding supports. Late, $V^\dagger$ resolves fine distinctions between proven alternatives while $V^\diamond$ maintains coarse pressure toward unentered corridors. The measured no-collapse of the constant mix says these signals do not fight; the unit normalization is what keeps either from becoming a dose ramp on the other.

---

## 11 Limitations

1. **Scale of evidence for the fourth rung.** Two seeds, one testbed, 12M-step budgets; the paired permanence result is 6-of-8 cells with per-cell margins of one to two seed-spreads. The production run carrying the ladder is hours old; its pre-registered health panels (§9), not this paper, will supply at-scale evidence.
2. **Self-play strains the stationarity argument.** The reservoir is correct because $r$ and $\Sigma$ are environment properties; in live self-play the *opponent is part of the transition kernel*, so both are only quasi-stationary as the league drifts. `Geo/Residual` trending upward is the designated symptom.
3. **Discount amplification.** $1/(1-\gamma)$ is 1667 at the deployment's γ = 0.9994 against 200 in the testbed: a small $\hat r$ error becomes an 8× larger value error at scale than any tested offline.
4. **The ellipsoid over-covers and $\Sigma$ is diagonal.** Real one-step reachable sets are not ellipsoids, and coordinates move together (height is bought with speed); both approximations inflate $V^\diamond$ where they bind. The σ-scale calibration and the mix's unit normalization bound the damage; they do not remove it.
5. **Detection was validated on a young policy.** The $\rho(V^*) = 0.698$ measurement used a 4M-step policy; whether the beats-habit property degrades as the remaining headroom becomes fine-grained is untested, and the resolution argument (§10) suggests it may.
6. **Fragment coverage remains the composition rung's boundary** and serendipity (C5) its only remedy; the geometric rung relaxes but does not remove the dependence on the record — $\hat r$ and $\Sigma$ are still fit where the agent has been.
7. **`hi` at the 12M testbed budget is near zero for all arms**; the ladder's early-air separation is a leading indicator, and the companion's 40M results carry the conversion claim for rungs 1–3 only.

---

## 12 Conclusion

The acquisition wall is usually breached from the environment side. The companion program showed it can be breached architecturally, by a critic whose optimism is restricted to compositions of executed steps — and that same restriction fixed a ceiling: the record cannot price what it has never contained. This paper completes the design. The ladder's four rungs grade optimism by its grounds — expectation, upper tail, executed composition, and finally the environment's own geometry, read as a conditional variance that says how far single decisions reach without ever predicting where they land. That last distinction is the paper's load-bearing idea: reachability statistics are not models, so a value field required to be self-consistent against them — one Hamilton–Jacobi–Bellman residual, three small MLPs, one input-gradient — obtains falsifiable optimism about states no trajectory has assembled, at MLP cost, under constraints that forbid every standard tool. Measured, the field is the only rung that tracks the environment over the habit; actuated beside the composition potential at constant weight, it is the best arm wherever acquisition is hardest; tested pairwise, it earns a permanent place in the mix rather than a bootstrap's retirement. The failure maps — spiral, ratchet, avoidance, the encoder illusion, the smoothness trap, the dose confound — are offered as reusable engineering knowledge for anyone building optimism into value functions online. The four-rung ladder now trains from step zero of a production run, with its health criteria pre-registered; what we normally get, what we get when things go well, what we can provably get, and what may yet be possible are, from that step, four simultaneous opinions the policy carries about every state it meets.

---

## Acknowledgments

The experimental program (design-space search, testbed implementation, oracle construction, trainer integration, and the drafting of this manuscript) was carried out with an AI research assistant (Claude, Anthropic) operating under the author's direction; the ladder framing, the constraint set, the permanence decision, and the final mechanism selections are the author's.

## Reproducibility statement

Appendix A specifies the benchmark; Appendix B the mechanism hyperparameters for all four rungs; §9 the production insertion points, including the reward-reconstruction identity and the delivered-dose panel. The offline testbed — benchmark environment, every arm of both failure maps, the MPC oracle, and the full ladder — will be released with experiment logs at [repository URL to be added]. The retraction record (seven withdrawn claims across the program, each with the confound that produced it) is preserved in the experiment logs.

## References

- Bellemare, M. G., Dabney, W., Munos, R. (2017). A Distributional Perspective on Reinforcement Learning. *ICML*.
- Berner, C., et al. (2019). Dota 2 with Large Scale Deep Reinforcement Learning. *arXiv:1912.06680*.
- Burda, Y., Edwards, H., Storkey, A., Klimov, O. (2019). Exploration by Random Network Distillation. *ICLR*.
- Doya, K. (2000). Reinforcement Learning in Continuous Time and Space. *Neural Computation* 12(1).
- Ecoffet, A., Huizinga, J., Lehman, J., Stanley, K. O., Clune, J. (2021). First Return, Then Explore. *Nature* 590 (Go-Explore).
- Fleming, W. H., Soner, H. M. (2006). *Controlled Markov Processes and Viscosity Solutions* (2nd ed.). Springer.
- Fujimoto, S., van Hoof, H., Meger, D. (2018). Addressing Function Approximation Error in Actor-Critic Methods. *ICML* (TD3).
- Hafner, D., Pasukonis, J., Ba, J., Lillicrap, T. (2023). Mastering Diverse Domains through World Models. *arXiv:2301.04104* (DreamerV3).
- Heess, N., Wayne, G., Silver, D., Lillicrap, T., Tassa, Y., Erez, T. (2015). Learning Continuous Control Policies by Stochastic Value Gradients. *NeurIPS*.
- Jaderberg, M., et al. (2019). Human-Level Performance in 3D Multiplayer Games with Population-Based Reinforcement Learning. *Science* 364.
- Kostrikov, I., Nair, A., Levine, S. (2022). Offline Reinforcement Learning with Implicit Q-Learning. *ICLR*.
- Kumar, A., Zhou, A., Tucker, G., Levine, S. (2020). Conservative Q-Learning for Offline Reinforcement Learning. *NeurIPS*.
- Kuznetsov, A., Shvechikov, P., Grishin, A., Vetrov, D. (2020). Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics. *ICML* (TQC).
- Mignatti, L. (2026). The Composition Critic: Directed Exploration by Seeking Realizable Headroom. Companion manuscript, `COMPOSITION_CRITIC.md`.
- Newey, W. K., Powell, J. L. (1987). Asymmetric Least Squares Estimation and Testing. *Econometrica* 55(4).
- Ng, A. Y., Harada, D., Russell, S. (1999). Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping. *ICML*.
- Pathak, D., Agrawal, P., Efros, A. A., Darrell, T. (2017). Curiosity-Driven Exploration by Self-Supervised Prediction. *ICML* (ICM).
- Raissi, M., Perdikaris, P., Karniadakis, G. E. (2019). Physics-Informed Neural Networks. *J. Comput. Phys.* 378.
- RLGym. A Reinforcement Learning Environment for Rocket League. Software: rlgym.org.
- RocketSim. An Open-Source Rocket League Physics Simulation. Software: github.com/ZealanL/RocketSim.
- Schrittwieser, J., et al. (2020). Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model. *Nature* 588 (MuZero).
- Schulman, J., Moritz, P., Levine, S., Jordan, M., Abbeel, P. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation. *ICLR*.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O. (2017). Proximal Policy Optimization Algorithms. *arXiv:1707.06347*.
- Todorov, E. (2009). Efficient Computation of Optimal Actions. *PNAS* 106(28).
- van Hasselt, H. (2010). Double Q-Learning. *NeurIPS*.

---

## Appendix A: Benchmark details

**Environment.** RocketSim (faithful Rocket League physics), 48 parallel arenas, tick-skip 8 (15 Hz decisions), 120-decision episodes (≈8 s). One car, one ball.

**Observation.** 30 dimensions: car pose, velocity, orientation, boost, ground-contact and flip state; ball position and velocity; relative terms.

**Action table.** 19 curated discrete actions (drive/steer/jump/boost/pitch/yaw/roll combinations).

**Cold spawn.** Car at $(x \sim U[-300, 300],\ y = -900,\ z = 17)$ uu, velocity $(0, 900, 0)$, facing $+y$, boost 100. Ball at $(x_{\text{car}} + U[-700, 700],\ U[300, 1200],\ U[100, 1500])$ uu, at rest.

**Reward.** +1 any ball touch; approach shaping $0.02 \cdot \Delta(\text{dist})/100$; $0.002 \cdot \max(0, v_y^{\text{ball}})/\text{BALL\_MAX\_SPEED}$-scaled term for ball speed toward goal; +10 score. No aerial-specific term of any kind.

**Metrics.** *touch* (any contact), *air* (airborne contact, car above 300 uu), *hi* (airborne conversion of spawns with $z \ge 900$). Logging readouts only. Drill/probe episodes, where any arm uses them, are excluded from all metrics.

**PPO.** $\gamma = 0.995$, $\lambda = 0.95$, clip 0.2, entropy 0.02, lr $3\times10^{-4}$, 3 epochs × 4 minibatches, 6,144 steps/iteration. Policy 30 → 128 → 128 → 19.

**Oracle (§6.4).** Receding-horizon MPC: CEM over the discrete action table against the simulator, re-planned every decision; scores a fixed probe set spanning mastered, composable, and never-assembled regions. Open-loop CEM was measured inadequate (0/24 aerials) and rejected as an oracle.

## Appendix B: Mechanism hyperparameters

**Rungs 1–3 (testbed).** Twin $V^\dagger$ heads: independent MLPs 30 → 256 → 256 → 1; Adam lr $3\times10^{-4}$, grad clip 1.0; 2 epochs × 4 minibatches; $\tau = 0.75$; targets copied every 8 iterations; target clamp $[-5, 30]$.

**Rung 4 (testbed).** Three nets on raw obs: Σ-net 30 → (2·30) Gaussian NLL; $\hat r$-net 30 → 1 MSE on arrival states; $V^\diamond$ 30 → 1 HJB residual, grad clip 1.0, extra field-only passes (6) per iteration against frozen fits. Reservoir 200k, uniform over all data. σ-scale 0.5 (oracle-calibrated). No drift term, no Lipschitz penalty (both measured harmful).

**Actuation (testbed).** $\Phi_{\text{mix}}$ per Eq. 5; $w = 0.5$ constant; geometric-potential β = 0.04 (dose-corrected, §8), delivered σ-ratio verified ≈ β; injection clamp $\pm 3\sigma_{\text{ext}}$; per-batch centering.

**Deployment (lineage started 2026-07-30).** Rungs 1–3 per the companion's corrected configuration (heads on the 1152 trunk). Rung 4: three independent nets {384, 384} on raw observations, lr $10^{-3}$; reservoir 200k fed at learn-prep with GAE-reconstructed scaled rewards; γ = 0.9994 (shared with the learner); $w = 0.5$ constant; β = 0.04; panels as in §9. The geometric nets deliberately do not touch the shared trunk.
