# The Composition Critic: Intelligent Exploration by Seeking Realizable Headroom

**Luca Mignatti**

*Preprint. Code: `research/tools/` (offline testbed: `~/Projects/experiments/rltest`), trainer integration: GigaLearnCPP (`PPOLearner.cpp`, `Learner.cpp`).*

---

## Abstract

Deep reinforcement learning agents cannot learn behaviors they never perform. A multi-step *conduct* — an aerial interception in Rocket League, a maneuver requiring a coordinated sequence of actions — whose first success has near-zero probability under current behavior produces no reward, no gradient, and is never acquired, regardless of training budget. Existing remedies either move the problem into the environment (curricula, reset-state distributions, demonstrations) or rely on optimism that is never tested against reality (exploration bonuses, optimistic value estimates), which we show measurably diverges. We introduce the **composition critic** $V^\dagger$: a value head trained by asymmetric (expectile) temporal-difference regression over *executed transitions only*, whose fixed point approximates the upper envelope of the values attainable by chaining steps the agent has actually performed — across different episodes, arenas, and agents. $V^\dagger$ credits conducts never performed *as wholes* the moment their pieces exist scattered in experience: the agent "realizes" an aerial is worth attempting because it has jumped, has boosted while tilted, and has touched lower balls, in different lives. The realization is actuated by a potential-based *seek* term, $\Phi = +H$ with $H = \mathrm{relu}(V^\dagger - V_{\text{real}})$, which pays the policy for carrying play into headroom; potential-based shaping leaves the optimal policy unchanged. The mechanism is model-free and search-free, uses identical environments everywhere, adds two MLP heads at training time and nothing at deployment, and costs ~1% wall-clock in our testbed. On a RocketSim aerial-discovery benchmark it reaches 2.4× the ball-interaction rate and 6.3× the airborne-touch rate of PPO at equal wall-clock, discovers high-ball aerial conversion where PPO remains at exactly zero, and — under the constraint set (no probes, no reset states, no world models) — *exceeds* a probe-assisted ceiling-critic baseline that violates those constraints. We further deploy it inside a full-scale self-play league trainer (1152-wide trunk, ~90-action table, billions of steps), where it runs stably with the auxiliary gradients *improving* shared-representation health relative to historical baselines. We report the complete measured failure map that forced each design decision, including two failure modes we believe are underdocumented: the **expectile ratchet** (online asymmetric TD self-amplifies through its own bootstrap) and the **closure-potential pathology** (gap-closing potentials teach *avoidance* of strong, spatially peaked frontier fields).

---

## 1 Introduction

Model-free policy-gradient methods learn from what they do. This is usually framed as a sample-efficiency limitation, but for a class of behaviors it is an *acquisition* limitation: if a behavior requires a coordinated multi-step action sequence, and the probability of first completing that sequence under the current policy is effectively zero, then the behavior contributes no reward signal, receives no gradient, and is never learned. More steps do not help; the gradient toward the behavior does not exist.

Rocket League provides a crisp instance. An *aerial* — jumping, tilting the car nose upward, and boosting through the air to intercept a high ball — is among the most valuable skills in the game. Under a young policy the completed maneuver essentially never occurs by chance: in our benchmark, vanilla PPO trained for 20–40M environment steps on a task where high-ball interception is richly available converts **exactly zero** high-ball aerials, while happily mastering everything reachable by driving. The situations are not rare — high balls appear constantly. The *conduct* is what never happens. State-novelty bonuses do not address this: the states are visited; the behavior from them is not attempted.

The standard industrial answer is to move the problem into the environment: curriculum state-setters that spawn the agent mid-maneuver, reset distributions concentrated on the skill, demonstrations, or dedicated exploration workers. These work (we reproduce a spawn-curriculum reaching >0.9 aerial conversion), but they carry costs that motivated this work: they require a designer to *name* each skill and its difficulty axis; reset-based practice measurably fails to transfer when the scaffold short-circuits the part of the skill that matters (§4.1); and they entangle the learning algorithm with environment instrumentation. We wanted the opposite: a **purely architectural** mechanism — identical environments everywhere, no reset states, no probes, no world models, no search, MLP-cost — under which the *policy itself* comes to intend toward its own frontier.

Our route to this mechanism was largely a sequence of measured failures, and we consider the map of those failures a primary contribution (§4). Its summary is an information-accounting principle:

> **The execution principle.** Information about the outcome of an action sequence that has never been executed anywhere can only come from (i) a model of the dynamics, (ii) explicit search, or (iii) an attempt. Any "optimism" about such sequences that is not grounded in one of these is unfalsifiable, and unfalsifiable optimism, trained through a bootstrap, diverges — we measure this directly (§4.2).

The constraint set (no models, no search, no dedicated attempt-machinery) therefore appears to forbid the goal. The escape is that it does not forbid optimism about *compositions*: a conduct never performed as a whole, every *piece* of which has been performed somewhere. A young Rocket League policy has jumped thousands of times (entropy alone guarantees this), has boosted while airborne in scattered fragments, and has occasionally touched low balls while off the ground. The aerial is a chain of steps that all exist in the data — in different episodes. Optimism over such chains is falsifiable piecewise: every link really happened.

The **composition critic** $V^\dagger$ computes exactly this quantity, using nothing but one-step temporal-difference learning with an asymmetric loss (§5). Each executed transition $(s, a, r, s')$ is treated as evidence of an inequality — "from $s$, at least $r + \gamma V^\dagger(s')$ was achievable" — and high-expectile regression converges toward the upper envelope of these inequalities, which chains across trajectories through shared states and function-approximation generalization. This is the *stitching* property known from offline RL [Kostrikov et al., 2021], moved online and put to a different use: not conservative policy extraction, but the manufacture of a **frontier field** $H(s) = \mathrm{relu}(V^\dagger(s) - V_{\text{real}}(s))$ — "the game's own record proves more is achievable from here than you currently collect."

Actuation is a single potential-based shaping term $\Phi(s) = +H(s)$, injected into advantages with standard-deviation matching and clamping. The sign matters more than we anticipated: the *closure* form $\Phi = -H$ (used successfully elsewhere in our trainer for small diffuse gaps) is catastrophic on a strong peaked field, because the cheapest way to reduce headroom along a trajectory is to walk away from it — the policy learns to avoid the ball (§4.5).

**Contributions.**

1. **The composition critic**: an online, model-free, search-free mechanism for directed exploration toward unperformed conducts, requiring only two MLP value heads and a potential-based advantage term; identical environments everywhere; ~1% wall-clock overhead; nothing at deployment (§5).
2. **A measured failure map** of the design space under these constraints — optimality-backup ceilings spiral (with truncated-quantile control only delaying divergence); executed probes calibrate but violate homogeneity; world models learn flight physics from fragments but relocate the search problem into the dream; single-head online expectile TD exhibits a *ratchet*; closure potentials teach avoidance (§4).
3. **Benchmark results** on RocketSim aerial discovery: at equal wall-clock (overhead is ~1%), 0.61/0.55 vs 0.32 ball interaction, 0.44/0.42 vs 0.18 airborne touch, and 0.089/0.043 vs 0.000 high-ball aerial conversion against PPO; exceeding a probe-assisted baseline that uses machinery our constraints forbid (§6).
4. **A at-scale deployment** in a production self-play league trainer with a 1152-wide shared trunk, in which the critic's trunk-coupled gradients coincide with *better* shared-representation health than historical baselines, and frontier-specific behavioral metrics (touch height, aerial touch rate) rise from run start (§7).

---

## 2 Related work

**Offline RL and stitching.** Implicit Q-Learning [Kostrikov et al., 2021] introduced expectile regression on TD targets to approximate a maximum over dataset-supported actions without querying out-of-distribution actions; the resulting trajectory *stitching* is central to offline RL. We move this estimator online, where we find the naive form unstable (the expectile ratchet, §4.4), stabilize it with twin heads and minimum-target backups in the style of TD3 / clipped double-Q [Fujimoto et al., 2018; van Hasselt, 2010], and use it not for policy extraction but as an exploration field. Conservative methods such as CQL [Kumar et al., 2020] penalize exactly the optimism we require; they solve the opposite problem.

**Optimism and exploration.** Count- and prediction-error novelty (e.g., RND [Burda et al., 2018]) rewards visiting unfamiliar *states*; our problem is unattempted *conducts* from familiar states, which state novelty cannot see (we confirm: an RND drive moves nothing on our benchmark). Optimistic initialization and UCB-style bonuses inject optimism into values; our §4.2 measurements are a caution that bootstrapped optimism about untried actions, at scale and with function approximation, self-amplifies unless it is either executed (probes) or restricted to executed support (this work). Go-Explore [Ecoffet et al., 2019] and reset-based curricula attack the same acquisition problem via environment control; our §4.1 documents when the returned skill fails to transfer, and our constraint set excludes the family.

**Model-based imagination and search.** Dreamer-class agents [Hafner et al., 2020+] and MuZero [Schrittwieser et al., 2020] obtain exactly the counterfactual competence we seek, at the cost of a learned model and (for search) planning compute. We verify the premise (a one-step model trained on entropy fragments learns flight dynamics well before any aerial exists, §4.3) — and the relocation of the problem (the dream actor faces the same exploration barrier inside the model, at imagination budgets far above ours). Under a hard no-model constraint these routes are unavailable; the composition critic can be read as extracting the *chaining* benefit of a model without the model, by letting the value function do the composition.

**Potential-based shaping.** Our actuation is classical PBRS [Ng, Harada & Russell, 1999], preserving optimal policies. The empirical content is the *sign and shape* analysis: closure potentials on strong peaked fields produce avoidance (§4.5), a hazard we have not seen documented at this sharpness.

**Rocket League and self-play systems.** Our deployment target is a large self-play league trainer in the lineage of population-based systems [Jaderberg et al., 2019; Berner et al., 2019; Vinyals et al., 2019], for Rocket League specifically building on the RLGym/RocketSim ecosystem and prior public bots (Necto/Nexto). Prior aerial acquisition in this ecosystem has relied on shaped event rewards and drill state-setters; to our knowledge no public system acquires aerial pressure purely architecturally.

**Distributional critics.** We tested a quantile-distributional variant (optimism via an upper-quantile read, QR-loss [Dabney et al., 2018], truncation in the style of TQC [Kuznetsov et al., 2020]). On this benchmark it *underperformed* the scalar expectile head (§6.4); truncation also failed to rescue the optimality-backup ceiling (§4.2).

---

## 3 Problem setting

We consider standard on-policy actor–critic training (PPO [Schulman et al., 2017] with GAE [Schulman et al., 2015]) in an environment where a target *conduct* — a specific multi-step behavior — is (a) richly rewarded when performed, (b) reachable from commonly visited states, and (c) of vanishing probability under the current policy. We ask for a mechanism that produces *directed pressure* toward acquiring the conduct, subject to:

- **C1 (homogeneity):** all environment instances identical; no dedicated probe/drill/reset instances;
- **C2 (no environment design):** no curriculum state-setters, no demonstrations, no banked reset states, no success-state memories that define the objective retrospectively;
- **C3 (no models, no search):** no learned dynamics model, no imagination rollouts, no tree search at train or test time;
- **C4 (MLP-cost):** per-step cost at collection and deployment unchanged; learn-phase overhead small;
- **C5 (serendipity floor):** baseline entropy is never reduced or gated anywhere — any guidance signal *adds* attention, and must never veto undirected exploration (the mechanism's own estimates are not oracles; conducts whose every fragment is missing can only be seeded by chance, and that channel must stay open).

**Benchmark.** Our offline testbed is a single-agent RocketSim task distilled from the deployment game. One car, one ball per arena (48 parallel arenas). Every episode begins *cold*: the car on the ground rolling forward with full boost; the ball placed ahead uniformly in a 3-D box ($|x| \le 700$, $y \in [300, 1200]$ ahead of the car, $z \in [100, 1500]$ — most of the box unreachable without flight). Reward is deliberately generic — +1 for any ball touch, a small uniform approach-shaping term, a small term for ball speed toward the goal, +10 for scoring; **no aerial-specific reward, detector, or bonus exists anywhere**. Episodes run 120 decisions (~8 s at 15 Hz). We report: **touch** (fraction of episodes with any touch), **air** (fraction with an airborne touch — car off ground, above 300 uu), and **hi** (fraction of high-ball episodes, spawn $z \ge 900$, converted with an airborne touch — the target conduct; all three are evaluation-only readouts). The policy is a small MLP (30-dim observation → 128 → 128 → 19 discrete actions); all mechanisms below add heads, never change the policy.

Vanilla PPO on this task learns ground play and low-ball interception and converts **zero** high-ball aerials at every budget we ran (20M, 25M: hi = 0.000). This is the acquisition wall in its simplest reproducible form.

---

## 4 A measured map of the design space

We report the failure map first because every element of the final mechanism (§5) is the negation of a measured failure, and because we believe several of these failures are load-bearing knowledge for anyone attempting architectural exploration. All numbers are from the benchmark of §3 (2 seeds per cell unless noted; ranges are min/max over seeds).

### 4.1 Environment-side mechanisms work — and show why we exclude them

For calibration we implemented the classical remedies. A hand-designed spawn curriculum over ball height (practice concentrated at the lowest unmastered height band, advancing on sustained mastery) reaches **0.86–0.96** high-aerial conversion — the skyline. A learned variant that *discovers* the difficulty axis (ranking candidate spawns by a learned quasimetric distance-to-success) matches the hand axis (0.76 vs 0.79 on solved seeds). But: (i) both require environment control (C2); (ii) the reset-based alternatives that avoid difficulty design fail the *transfer* test — returning the agent to banked pre-success states produced an aggregate success rate of 0.244 that collapsed to **0.005** when evaluated only on cold starts: the scaffold had short-circuited the approach phase, which *is* the skill; and (iii) a retrospective success-state memory makes the frontier definition depend on which successes luck has already produced, and aims practice at *reproducing particular states* rather than at value. These observations fix constraints C1–C2 and the design goal: pressure must live in the policy's own updates.

### 4.2 Unfalsifiable optimism diverges: the optimality-backup ceiling

The direct reading of "estimate the best that can be done" is a Q-function over the discrete action table trained with optimality backups — max over *all* actions, including untried ones, valued by generalization; amortized tree search. We implemented it distributionally (16 quantiles per action, double-Q action selection, target network, reward-scale clamps). Result: the ceiling estimate **spiraled** — mean $V_{\max}$ reached 42.6 and 38.3 (two seeds) against a best-ever observed episode return of ~30–36 and *mean* returns of ~0.2; the derived drive was actively harmful, and downstream use (drill selection by apparent headroom) aimed at noise. Adding truncated-quantile targets (TQC-style: drop the top 4 of 16 target quantiles; conservative action selection) **delayed the divergence to ~16M steps but did not prevent it** (2/4 seeds re-spiraled).

The root cause is epistemic, not statistical: under an on-policy data stream, the argmax actions at most states are never *executed*, so their generalization-derived values are never corrected; the bootstrap then compounds selection bias on noise. Confirming this diagnosis, converting a small fraction of arenas into ε-greedy *probe* arenas that execute the critic's argmax (their data feeding only the critic) fully stabilized the same estimator — values grounded for the entire run, and the mechanism became productive (touch 0.44, air 0.28–0.31 at 20M; 0.53–0.57 / 0.38–0.42 / hi 0.048–0.060 at 40M). Executed probes, however, violate C1, and their calibration is itself bounded by execution depth: the probe's greedy chain breaks where the critic has no data, so the ceiling converges to "the best my current chain can demonstrate" — measurably pessimistic on deep conducts (the probe-assisted field read ~zero headroom at high balls while 0.9+ conversion was provably achievable). Probes are a fine *accelerator*; they are not the architectural answer.

### 4.3 World models learn the physics but relocate the search

A one-step dynamics model $M(s, a) \to (\Delta s, r)$ trained by supervised regression on all real transitions learns flight physics from entropy fragments alone: airborne-transition prediction error fell from 0.0084 to 0.0021, converging toward the ground-transition error (0.0013), long before any aerial success existed. The premise of imagination — the knowledge is in the fragments — is *true*. But value-from-imagination failed to ignite at our budgets: random/policy-sampled candidate rollouts cannot compose a precise 16–20-step maneuver (the realization signal appeared only where the policy was already close), and a Dreamer-style imagination actor faces the *same* exploration barrier inside the model — it must discover the maneuver in the dream, which at imagination budgets below reality-scale (ours were ~1× collected experience; Dreamer-class systems dream 10–100×) does not happen. Imagination does not dissolve the exploration problem; it relocates it into a cheaper simulator, and the relocation only pays at large imagination budgets. Excluded regardless by C3.

### 4.4 Executed-support optimism, naive form: the expectile ratchet

Restricting optimism to *executed* transitions — expectile-TD: $V^\dagger$ regressed toward $y = r + \gamma V^\dagger_{\text{tgt}}(s')$ with asymmetric weight $\tau$ — removes the unfalsifiable-action channel entirely. The naive online form ($\tau = 0.9$, single head, hard target refresh) nonetheless diverged, more slowly: $H$ at *easy* states inflated 0.3 → 11.8 over 20M with no conversion behind it. We call this the **expectile ratchet**: the asymmetric loss preferentially fits upward fluctuations of the bootstrap target, the target inherits the inflation, and the loop compounds — no untried actions required. Anchoring to executed support is *necessary but not sufficient*; the asymmetry itself must be controlled.

### 4.5 The closure-potential pathology

Our trainer's pre-existing gap machinery injects advantages from a *closure* potential $\Phi = -(\text{gap})$ — credit for reducing a knowing–doing gap — safely, because those gaps are small and spatially diffuse. Applying the same form to the headroom field ($\Phi = -H$) was catastrophic: **touch collapsed to 0.014/0.027 versus 0.244 for PPO** — the drive actively *destroyed* ball interaction. The mechanism is geometric: when the field is strong and spatially peaked (large near the ball, small elsewhere), the cheapest way to reduce $\Phi$-potential along a trajectory is to *leave the peak* — the policy is paid to walk away from its own frontier. Sign and field-shape interact: closure potentials are only safe on weak or flat fields. The seek form $\Phi = +H$ (credit for *entering* headroom) inverts the geometry and, being potential-based, still preserves optimal policies.

### 4.6 Two more measured negatives

**Headroom-modulated entropy** (per-state entropy coefficient $= \eta_0(1 + k\tilde H)$, floor preserved): at $k=1$, *hurts* (touch 0.208/0.214 vs base 0.322) — the extra policy noise at frontier states outweighs the attention benefit at this scale. **Distributional $V^\dagger$** (quantile head, optimism as a q75 read with the backup carrying the upper quantile): underperforms the scalar expectile head (touch 0.28/0.34 vs 0.41/0.37). Neither is part of the final mechanism.

---

## 5 The composition critic

The final mechanism is the conjunction of the failure map's negations. It adds, to an unmodified PPO learner:

**(1) Twin composition heads.** Two value heads $V_1^\dagger, V_2^\dagger$ (in the testbed: independent 2×256 MLPs on raw observations; at deployment: heads on the shared trunk, §7). Each iteration, one-step TD targets are formed over the freshly collected buffer, using the executed transitions only:

$$y_i \;=\; r_i \;+\; \gamma\,(1 - d_i)\, \min\!\big(V^\dagger_{1,\text{tgt}}(s_{i+1}),\, V^\dagger_{2,\text{tgt}}(s_{i+1})\big),$$

where $d_i$ is nonzero at every trajectory boundary (terminal *or* truncation), and the target heads are a periodic copy (testbed: every 8 iterations) or simply the previous iteration's heads (deployment). Both heads are trained by expectile regression at $\tau = 0.75$:

$$\mathcal{L}(V_h^\dagger) \;=\; \mathbb{E}_i\big[\, |\tau - \mathbb{1}[u_i < 0]|\; u_i^2 \,\big], \qquad u_i = y_i - V_h^\dagger(s_i),$$

with targets clamped to a fixed value-scale range. The **minimum over twins inside the target** is the anti-ratchet: upward generalization noise must appear in *both* heads at the same state to propagate, the same selection-bias suppression that clipped double-Q provides for the max-over-actions bias — here applied to the expectile's implicit max-over-dataset. With this control (and $\tau$ at IQL's own operating point rather than 0.9), $V^\dagger$ remained bounded and drift-free in every run of this paper, including billions of steps at deployment scale.

Why this object sees unperformed conducts: each executed transition contributes an *inequality* ("this step was possible here"), and high-expectile TD approximates the upper envelope consistent with all of them. Value then flows backward along any chain of inequalities — trajectory A's jump, trajectory B's airborne boost, trajectory C's low aerial touch — linked wherever the function approximator identifies their intermediate states. The unit of required luck falls from "a completed conduct" (astronomically rare) to "each step, somewhere, ever" (supplied by entropy). The honest boundary is the same statement's contrapositive: a conduct containing a step executed *nowhere* is invisible to $V^\dagger$ until baseline entropy supplies that link once — which is why C5 is a constraint and not a preference.

**(2) The headroom field.** $H(s) = \mathrm{relu}\big(\min(V_1^\dagger, V_2^\dagger)(s) - V_{\text{real}}(s)\big)$, where $V_{\text{real}}$ is the ordinary critic. $H$ is a *zone-of-proximal-development* field: zero where the policy already collects what is achievable, zero where nothing more is provably composable, positive exactly where the game's own record exceeds current behavior. It is self-retiring — as the policy learns to collect value at a state, $V_{\text{real}}$ rises and $H$ closes there; as new fragments enter experience, $V^\dagger$ rises and $H$ opens further out. No controller, no thresholds, no memory beyond the two heads.

**(3) Seek actuation.** A potential-based term with $\Phi = +H$:

$$a^{\text{int}}_i = \gamma\,(1-d_i)\,H(s_{i+1}) - H(s_i),$$

centered, scale-matched to a fraction $\beta$ of the extrinsic advantage standard deviation ($\sigma$-ratio matching with a floor), clamped to $\pm 3\sigma_{\text{ext}}$, and added to the advantages before the PPO update. $\beta$ is the mechanism's single important hyperparameter (§6.3). Baseline entropy regularization is untouched (C5).

**Cost accounting (C4).** Collection and deployment are unchanged — the policy never consumes $H$; all computation happens at learn time: two head forward/backward passes over the buffer plus one no-grad forward for targets and the field. Measured on the testbed under identical machine load: **26,313/26,222 steps/s versus 26,615 for PPO (0.99×)**. At deployment the heads are two 1152-input MLPs in a system already running several critic heads; the overhead is low single-digit percent of the learn phase.

**Deployment detail: TD targets in the critic's units.** Production learners often normalize/clip rewards inside GAE, so raw rewards are in the wrong units for a TD target meant to be compared with $V_{\text{real}}$. The scaled one-step reward can be recovered *from GAE outputs alone* via the identity $\delta_i = A_i - \gamma\lambda(1-d_i)A_{i+1}$, giving

$$r^{\text{scaled}}_i = A_i - \gamma\lambda(1-d_i)A_{i+1} - \gamma(1-d_i)V_{i+1} + V_i,$$

with pre-injection advantages $A$. This makes the integration three insertion points (heads + loss, targets at learn-prep, one injection site) and no changes to GAE or collection.

---

## 6 Offline experiments

Setup as in §3. All mechanism arms use identical PPO hyperparameters to the baseline; the only differences are the heads and the injection. Seeds: 2 per mechanism cell; the mechanism-vs-baseline comparison at 25M uses the same-wave, same-machine baseline run.

### 6.1 Main results

| Arm | Budget | touch | air | hi (aerial conversion) |
|---|---|---|---|---|
| PPO | 20M | 0.244 | 0.067 | 0.000 |
| PPO | 25M | 0.322 | 0.178 | 0.000 |
| **+ Composition critic** (β=0.05) | 25M | 0.41 / 0.37 | 0.21 / 0.19 | 0.002 / 0.002 |
| **+ Composition critic** (β=0.15) | 24M | 0.43 / 0.46 | 0.24 / 0.27 | 0.009 / 0.013 |
| **+ Composition critic** (β=0.30) | 24M | 0.53 / 0.48 | 0.36 / 0.30 | 0.033 / 0.023 |
| **+ Composition critic** (β=0.05) | 40M | 0.51 / 0.49 | 0.34 / 0.33 | 0.036 / 0.037 |
| **+ Composition critic** (β=0.15) | 40M | **0.61 / 0.55** | **0.44 / 0.42** | **0.089 / 0.043** |
| *Probe-assisted ceiling critic (violates C1; reference)* | 40M | 0.53 / 0.57 | 0.38 / 0.42 | 0.048 / 0.060 |
| *Spawn curriculum (violates C2; skyline)* | 20–25M | — | — | 0.86–0.96 |

Because the overhead is 1%, equal-steps is equal-wall-clock to within the baseline's own extrapolated gain (+0.005 touch): at iso-compute the mechanism yields **~1.6–1.9×** touch, **~1.7–2.0×** air at 24–25M (β=0.3), growing to **2.4× / 6.3×** at 40M (β=0.15, vs the 20M baseline's own trend continued), and aerial conversion strictly positive and rising against a flat zero. Under the full constraint set the composition critic also **outperforms the probe-assisted estimator** that is allowed to execute its own optimism — consistent with §4.2's finding that execution-calibrated ceilings are pessimistic on deep conducts, while composition-based ceilings keep credit flowing from fragments the probes cannot chain.

The environment-side skyline remains far ahead on the target conduct at these budgets. The claim of this paper is not that architecture beats environment design where environment design is available; it is that *when the constraint set forbids environment design*, a purely architectural mechanism moves an unmovable metric — and its trajectory at 40M (hi rising ~0.001→0.089 with no plateau) indicates acquisition in progress, not saturation.

### 6.2 The frontier wave

The headroom field's spatial evolution is directly observable by probing $H$ on a fixed grid of canonical spawn states of increasing ball height. Early in training $H$ peaks at low heights ("the record proves you could be touching these") while high balls read near zero (no fragments to chain yet). As low-ball play consolidates, $H$ collapses there and the peak migrates upward, tracking the boundary between mastered and composable-but-uncollected; airborne-touch fragments accumulate, extending $V^\dagger$'s reach; high-ball conversion begins as the wave arrives. This *frontier wave* — the intended ZPD dynamics — emerges from the two heads alone, with no controller, and is the qualitative signature by which we distinguish a healthy run from both failure modes: a ratcheting run shows $H$ growing everywhere without behavioral change; a closure-sign run shows the policy vacating the peak.

### 6.3 Dose–response

$\beta$ sweeps cleanly at 24–25M: air 0.21/0.19 (β=0.05) → 0.24/0.27 (0.15) → 0.36/0.30 (0.30); hi 0.002 → 0.009/0.013 → 0.033/0.023; with $V^\dagger$ bounded at every dose (final means 1.2–2.1). We adopted β=0.15 for the long-budget benchmark and deployment as the middle of the productive range; β=0.30 buys speed at 25M-scale budgets and showed no instability, so the range is forgiving. At β=0.15 the injected term averages ~6% of advantage scale.

### 6.4 Ablations

Each row is a measured arm (§4 for details):

| Ablation | Result | Lesson |
|---|---|---|
| max over untried actions (optimality backup) | ceiling 42 vs best-ever return 36; drive harmful | unfalsifiable optimism diverges |
| + TQC truncation | spiral delayed to ~16M, not prevented | statistics cannot fix an epistemic hole |
| + executed probes (violates C1) | stable, productive; but field pessimistic on deep conducts | execution calibrates *and* bounds |
| single head, τ=0.9 | H@easy 0.3 → 11.8, no conversion | the expectile ratchet |
| twin-min + τ=0.75 (ours) | bounded for 40M+ (and at deployment scale) | selection-bias suppression transfers to expectiles |
| closure sign Φ=−H | touch 0.014/0.027 (vs 0.244) | peaked fields + closure potentials ⇒ avoidance |
| seek sign Φ=+H (ours) | main results | PBRS, geometry-correct |
| entropy modulation (k=1) | touch 0.21 vs 0.32 | noise where you need precision |
| distributional q75 variant | touch 0.28/0.34 vs 0.41/0.37 | scalar expectile suffices; upper-quantile reads re-import inflation |
| world model + imagination | physics learned; ignition failed at ≤1× dream budget | search relocated, not removed |

### 6.5 Compute

SPS ratio 0.99 (measured under identical co-load); all overhead in the learn phase; zero additional cost at collection or deployment. At fixed wall-clock the comparison in §6.1 is therefore unchanged to within ±1% step budget.

---

## 7 Deployment at scale

We integrated the mechanism into GigaLearnCPP, a production Rocket League self-play trainer (PPO; 1152-wide shared trunk with policy/critic heads; ~90-action table; multiple auxiliary systems including a goal-horizon critic, reachability aux losses, and a population league; hundreds of thousands of steps/s). Integration followed §5's deployment recipe: twin heads mirroring the main critic head's architecture, one-iteration-frozen targets riding the experience buffer, the GAE-identity reward reconstruction, the seek term injected at the single advantage-assembly site (β=0.15, σ-matched, clamped, covered by the trainer's existing rating-drawdown latch), and telemetry panels for the field and the injection.

One deliberate departure from the testbed: at the operator's direction the heads share the trunk *with gradients flowing*, on a fresh run. The trainer's history contains a documented incident in which a fresh, chance-level auxiliary loss coupled into a *mature* trunk destroyed ~125 Elo across all modes in ~500 iterations; the working hypothesis distinguishing that incident from beneficial auxiliary training (e.g., this trainer's reachability losses) is *co-adaptation from step zero* versus mid-run grafting. The deployment provides a data point: over the first ~3.4B steps / 12k iterations of a fresh run with the composition heads coupled from step 0 —

- **Stability**: $V^\dagger$ mean bounded (0–0.7, no drift); $H$ mean 0.02–0.24 (the testbed's healthy band); injection ~6% of advantage scale; the rating latch never fired.
- **Representation health**: the trainer's canonical trunk-coupled aux-pressure metric ran at 0.25–0.29 against a documented historical baseline of 0.47–0.52 (the incident signature was 0.5 → 1.36) — i.e., *better* than baseline with the new heads coupled, consistent with the co-adaptation hypothesis and with auxiliary value prediction acting as representation learning.
- **Behavior**: skill rating climbed monotonically (EMA −2 → 1022 over the window) with no drawdown events; mean touch height rose 131–157 → 190–219 uu; aerial-touch ratio rose ~2–3× (0.0005–0.002 → 0.003–0.006). These are the two metrics the headroom drive specifically targets.

**Caveat, stated plainly:** this deployment has no paired control run; a fresh self-play run improves on many metrics regardless, and we do not attribute the rating curve to the mechanism. The defensible claims are stability at scale, negligible cost, trunk health above historical baseline with coupled gradients, and frontier-specific metrics moving in the predicted direction from run start. Attribution awaits a paired run.

---

## 8 Analysis

**Why composition escapes the impossibility argument.** The execution principle (§1) says optimism about never-executed action sequences is unfalsifiable without a model, search, or an attempt. $V^\dagger$'s optimism is about *values*, constrained to the cone generated by executed one-step inequalities: every unit of claimed headroom is a sum along a chain of steps that individually happened. Falsification is therefore piecewise and automatic — each link continues to be re-estimated from real data every iteration — and the estimator inherits the model-free trainer's speed because chaining is performed by the value function's fixed point rather than by rollouts. What is genuinely given up is the ability to see conducts with a globally-missing link; C5 (the entropy floor) is the standing channel by which such links eventually appear, at which point $V^\dagger$ propagates their value backward within iterations, not epochs.

**Why the twins are necessary online but not offline.** Offline, the dataset is fixed: expectile regression fits an envelope over a static set of inequalities. Online, the target network's inflation re-enters the regression targets, and the asymmetric loss preferentially retains it — a positive feedback absent in the offline setting. The min-over-twins breaks the loop by requiring inflation to be *reproduced independently* before it can propagate, the same decorrelation logic as clipped double-Q, transplanted from the max-over-actions to the expectile's implicit max-over-support.

**Why the sign of the potential is decisive.** For any potential $\Phi$, PBRS credit along a step is $\gamma\Phi(s') - \Phi(s)$. If $\Phi = -H$ and $H$ is large on a compact region (the frontier), trajectories are paid for *exiting* the region — the drive is a repellent exactly where attention is wanted, and the policy converges to orbiting away from its frontier (measured: ball interaction collapsed 10×). With $\Phi = +H$ the same field is an attractor; loitering is not rewarded (potential differences telescope), and *resolving* the headroom pays doubly: once through the extrinsic reward that resolution earns, once through $V_{\text{real}}$ catching up and the wave moving on. Closure-form injections remain appropriate for weak, diffuse gaps (as deployed elsewhere in the trainer); the general rule our measurements support is: *seek strong peaked fields, close weak diffuse ones.*

**Relation to the ladder.** In the deployed system's terms — a progression of critics $V_{\text{real}}$ (what I reliably do) → $V_{\text{exp}}$ (what I sometimes do; a return-level expectile) → top rung — the composition critic completes the ladder with "what my executed steps can *compose*." The return-level expectile sees only whole behaviors that occurred; the Bellman-level expectile sees their recombinations. That step — from optimism over *outcomes experienced* to optimism over *outcomes constructible* — is precisely what moves the aerial-conversion metric off zero, and it is the smallest such step our failure map found that does not diverge.

---

## 9 Limitations

1. **Fragment coverage is the hard boundary.** Conducts with a globally-unexecuted step are invisible to $V^\dagger$ until entropy supplies the link. Environment-side accelerators (probes, drills) compose cleanly with the mechanism and close this residue faster where they are acceptable; under the strict constraint set, acquisition speed for such conducts is bounded by serendipity.
2. **The environment skyline is not reached.** At 40M the spawn curriculum's 0.86–0.96 aerial conversion is far above our 0.04–0.09. The mechanism's trajectory has not plateaued, but we have not run it to convergence; the honest current claim is *acquisition where PPO has none*, not parity with environment design.
3. **Attribution at deployment scale is open** (§7 caveat): no paired control run yet.
4. **Seed counts are small** (n=2 per cell; the benchmark's variance across seeds is visible in the reported ranges) and the benchmark is a single task family in a single simulator. The deployment evidence mitigates the single-task concern but introduces its own confounds.
5. **Two hyperparameters carry real sensitivity**: $\tau$ (0.9 ratchets even with twins at single-head strength; 0.75 is stable) and $\beta$ (productive range measured 0.05–0.30). Both transferred from testbed to deployment without retuning, but we have not characterized their interaction.
6. **Non-stationarity of self-play** is only partially addressed: $V^\dagger$'s inequalities are re-estimated continuously, but an opponent-induced collapse of previously-composable value would manifest as stale headroom until the relevant transitions are revisited.

---

## 10 Conclusion

The acquisition wall — behaviors that never happen, therefore never learn — is usually breached from the environment side. We showed it can be breached from the *architecture* side, under constraints that forbid every standard tool: no curricula, no resets, no probes, no models, no search, no extra deployment cost. The route is an accounting identity taken seriously: everything an agent has ever done is evidence about what it could do, and a value function trained to hold the upper envelope of that evidence — carefully, with the asymmetry double-checked by twin estimators, and actuated with the sign that attracts rather than repels — turns scattered fragments of competence into directed pressure toward skills that have never once occurred. On our benchmark this moved high-ball aerial conversion from exactly zero to steadily climbing at 1% overhead, outperforming an execution-assisted ceiling under the same budget; at production scale it has so far been a free, stable, representation-friendly addition whose targeted behavioral metrics moved from run start. The measured failure map — spiral, ratchet, avoidance — is offered as reusable engineering knowledge for anyone building optimism into value functions online: *ground it in execution, double-check its asymmetry, and mind the sign of the potential.*

---

## Acknowledgments

The experimental program (design-space search, testbed implementation, trainer integration, and this manuscript) was conducted with an AI research assistant (Claude, Anthropic) operating under the author's direction; all design constraints, the ladder framing, the shared-trunk deployment decision, and the final mechanism selection are the author's.

## References

- Bellemare, M. G., Dabney, W., Munos, R. (2017). *A Distributional Perspective on Reinforcement Learning.* ICML.
- Berner, C., et al. (2019). *Dota 2 with Large Scale Deep Reinforcement Learning.* arXiv:1912.06680.
- Burda, Y., Edwards, H., Storkey, A., Klimov, O. (2018). *Exploration by Random Network Distillation.* arXiv:1810.12894.
- Dabney, W., Rowland, M., Bellemare, M. G., Munos, R. (2018). *Distributional Reinforcement Learning with Quantile Regression.* AAAI.
- Ecoffet, A., Huizinga, J., Lehman, J., Stanley, K. O., Clune, J. (2019). *Go-Explore: a New Approach for Hard-Exploration Problems.* arXiv:1901.10995.
- Fujimoto, S., van Hoof, H., Meger, D. (2018). *Addressing Function Approximation Error in Actor-Critic Methods.* ICML (TD3).
- Hafner, D., Lillicrap, T., Ba, J., Norouzi, M. (2020). *Dream to Control: Learning Behaviors by Latent Imagination.* ICLR; and successors (DreamerV2/V3).
- Jaderberg, M., et al. (2019). *Human-level performance in 3D multiplayer games with population-based reinforcement learning.* Science.
- Kostrikov, I., Nair, A., Levine, S. (2021). *Offline Reinforcement Learning with Implicit Q-Learning.* arXiv:2110.06169.
- Kumar, A., Zhou, A., Tucker, G., Levine, S. (2020). *Conservative Q-Learning for Offline Reinforcement Learning.* NeurIPS.
- Kuznetsov, A., Shvechikov, P., Grishin, A., Vetrov, D. (2020). *Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics.* ICML (TQC).
- Ng, A. Y., Harada, D., Russell, S. (1999). *Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping.* ICML.
- Schrittwieser, J., et al. (2020). *Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model.* Nature (MuZero).
- Schulman, J., Moritz, P., Levine, S., Jordan, M., Abbeel, P. (2015). *High-Dimensional Continuous Control Using Generalized Advantage Estimation.* arXiv:1506.02438.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.
- van Hasselt, H. (2010). *Double Q-learning.* NeurIPS; van Hasselt, H., Guez, A., Silver, D. (2016). *Deep Reinforcement Learning with Double Q-learning.* AAAI.
- Vinyals, O., et al. (2019). *Grandmaster level in StarCraft II using multi-agent reinforcement learning.* Nature (AlphaStar).
- Wang, T., Torralba, A., Isola, P., Zhang, A. (2023). *Optimal Goal-Reaching Reinforcement Learning via Quasimetric Learning.* ICML.

---

## Appendix A: Benchmark details

RocketSim (real Rocket League physics), 48 arenas, tick-skip 8 (15 Hz decisions), 120-decision episodes. Observation: 30-dim (car pose/velocity/orientation/boost/ground-contact/flip, ball position/velocity, relative terms). Action table: 19 curated discrete actions (drive/steer/jump/boost/pitch/yaw/roll combinations). Cold spawn: car at $(x \sim U[-300,300], -900, 17)$, velocity $(0, 900, 0)$, forward $+y$, boost 100; ball at $(x_{\text{car}} + U[-700,700],\ U[300,1200],\ U[100,1500])$, at rest. Reward: +1 any touch; approach shaping $0.02 \cdot \Delta(\text{dist})/100$ (uniform — no airborne multiplier); $0.002 \cdot \max(0, v_y^{\text{ball}})$; +10 goal. Metrics are per-episode indicators aggregated per logging window; *hi* restricts to episodes with spawn $z \ge 900$ and requires an airborne touch (car off ground, $z > 300$). PPO: $\gamma = 0.995$, $\lambda = 0.95$, clip 0.2, entropy 0.02, lr $3 \times 10^{-4}$, 3 epochs × 4 minibatches per iteration (6,144 steps/iteration).

## Appendix B: Mechanism hyperparameters

Testbed: heads 30→256→256→1 (independent nets), Adam lr $3\times10^{-4}$, grad-clip 1.0, 2 epochs × 4 minibatches per iteration; $\tau = 0.75$; target copy every 8 iterations; target clamp $[-5, 30]$; $\beta \in \{0.05, 0.15, 0.30\}$, $\sigma$-floor $0.05\sigma_{\text{ext}}$, clamp $\pm 3\sigma_{\text{ext}}$. Deployment: heads mirror the main critic head config on the 1152 trunk (≈4.0M params each), gradients coupled (fresh run), targets one-iteration-frozen, value-scale clamp $\pm 2 \cdot \text{p99}|y|$, $\beta = 0.15$; injection computed pre-blend at learn-prep; covered by the trainer's rating-drawdown latch.

## Appendix C: Reproduction pointers

Offline: `possible_train.py` (§4.2 arms incl. probes), `imagine_train.py` (§4.3), `stitch_train.py` (§4.4–4.5 naive form), `ladder_v2.py` (final mechanism; modes `base`/`seek`/`ent`/`seekent`; env knobs `L2_BETA`, `L2_DIST`), `aerial_env.py` (benchmark). Experimental logs: `POSSIBLE.md`, `IMAGINE.md`, `OVERNIGHT_LOG.md`. Deployment: `PPOLearnerConfig::vdagEnabled` (+ `vdagTau`, `vdagSeekBeta`); heads `vdag1/vdag2` in `PPOLearner.cpp`; targets/injection in `Learner.cpp` learn-prep; panels `Headroom/*`.
