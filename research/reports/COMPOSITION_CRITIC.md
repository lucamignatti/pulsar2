# The Composition Critic: Directed Exploration by Seeking Realizable Headroom

**Luca Mignatti**

*Preprint. Code and experiment logs: [repository URL to be added upon release].*

---

## Abstract

Deep reinforcement learning agents cannot learn behaviors they never perform. A skill requiring a coordinated multi-step action sequence, such as an aerial interception in Rocket League, yields no reward and therefore no gradient while its probability of first success remains negligible, and additional environment steps do not change this. Such skills are conventionally acquired by engineering the environment: conduct-specific shaped rewards, curricula, reset-state distributions, or demonstrations. These methods are effective, but they require a designer to specify each target skill and its difficulty axis in advance, and they couple the learning algorithm to environment instrumentation. The principal algorithmic alternative, optimism about actions the agent has not executed, is unfalsifiable under an on-policy data stream; we measure its divergence directly. We introduce the **composition critic** $V^\dagger$: a value head trained by asymmetric (expectile) temporal-difference regression over *executed transitions only*, whose fixed point approximates the upper envelope of values attainable by chaining steps the agent has actually performed, chained across different episodes, arenas, and agents. $V^\dagger$ credits behaviors never performed *as wholes* the moment their pieces exist scattered in experience, and the resulting headroom field $H = \mathrm{relu}(V^\dagger - V)$ is actuated by a potential-based *seek* term $\Phi = +H$, which pays the policy for carrying play into provable headroom while leaving optimal policies unchanged. The mechanism is model-free and search-free, uses identical environments everywhere, adds two MLP heads at training time and nothing at deployment, and costs ≈1% wall-clock. On a Rocket League aerial-discovery benchmark it reaches 2.4× the ball-interaction rate and 6.3× the airborne-touch rate of PPO at matched wall-clock, lifts high-ball aerial conversion off an exact zero, and, under a constraint set forbidding probes, resets, and world models, exceeds a probe-assisted baseline that violates those constraints. In a production-scale integration, a configuration fault left the heads frozen at random initialization for the first ≈3.4B steps; we report that run as an accidental control. It is silent on the mechanism itself, but shows that a chance-level auxiliary objective coupled to the shared trunk from step zero coincided with representation health better than historical baselines. We also report the measured failure map that forced each design decision, including two failure modes we believe are underdocumented: the **expectile ratchet** and the **closure-potential pathology**.

---

## 1 Introduction

Model-free policy-gradient methods learn from what they do. This is usually framed as a sample-efficiency limitation, but for a class of behaviors it is an *acquisition* limitation: if a behavior requires a coordinated multi-step action sequence, and the probability of first completing that sequence under the current policy is effectively zero, then the behavior contributes no reward signal, receives no gradient, and is never learned. More steps do not help; the gradient toward the behavior does not exist.

Throughout this paper we use the term **conduct** for such a target: a specific temporally extended behavior, meaning a coordinated multi-step action sequence, as distinct from a *state* the agent might merely visit. The distinction matters because the acquisition problem lives in the space of conducts, not states.

Rocket League¹ provides a crisp instance. An *aerial* (jumping, tilting the car nose upward, and boosting through the air to intercept a high ball) is among the most valuable skills in the game. Under a young policy the completed maneuver essentially never occurs by chance: in our benchmark, vanilla PPO converts **exactly zero** high-ball aerials at every budget we ran, while readily mastering everything reachable by driving. The situations are not rare; high balls appear constantly. The *conduct* is what never happens. State-novelty bonuses do not address this: the states are visited; the behavior from them is not attempted.

¹ *Rocket League is a physics-based car-soccer game. Cars drive, jump, and carry a finite "boost" resource whose sustained thrust enables controlled flight. Distances are measured in Unreal units (uu); the arena ceiling sits at ≈2044 uu.*

The standard practical answer is to move the problem into the environment: curriculum state-setters that spawn the agent mid-maneuver, reset distributions concentrated on the skill, demonstrations, or dedicated exploration workers. These work, and we reproduce a spawn curriculum reaching up to 0.96 aerial conversion, but they carry costs that motivated this work: they require a designer to *name* each skill and its difficulty axis; reset-based practice measurably fails to transfer when the scaffold short-circuits the part of the skill that matters (§4.1); and they entangle the learning algorithm with environment instrumentation. We wanted the opposite: a **purely architectural** mechanism, with identical environments everywhere, no reset states, no probes, no world models, no search, and MLP-scale cost, under which the *policy itself* comes to intend toward its own frontier.

Our route to this mechanism was largely a sequence of measured failures, and we consider the map of those failures a primary contribution (§4). Its summary is an information-accounting principle:

> **The execution principle.** Information about the outcome of an action sequence that has never been executed anywhere can only come from (i) a model of the dynamics, (ii) explicit search, or (iii) an attempt. Any "optimism" about such sequences that is not grounded in one of these is unfalsifiable, and unfalsifiable optimism, trained through a bootstrap, diverges, as we measure directly (§4.2).

The constraint set (no models, no search, no dedicated attempt-machinery) therefore appears to forbid the goal. The escape is that it does not forbid optimism about *compositions*: a conduct never performed as a whole, every *piece* of which has been performed somewhere. A young Rocket League policy has jumped thousands of times (entropy alone guarantees this), has boosted while airborne in scattered fragments, and has occasionally touched low balls while off the ground. The aerial is a chain of steps that all exist in the data, in different episodes. Optimism over such chains is falsifiable piecewise: every link really happened.

The **composition critic** $V^\dagger$ computes exactly this quantity, using nothing but one-step temporal-difference learning with an asymmetric loss (§5). Each executed transition $(s, a, r, s')$ is treated as evidence of an inequality: from $s$, at least $r + \gamma V^\dagger(s')$ was achievable. High-expectile regression converges toward the upper envelope of these inequalities, which chains across trajectories through shared states and function-approximation generalization. This is the *stitching* property known from offline RL [Kostrikov et al., 2022], moved online and put to a different use: not conservative policy extraction, but the manufacture of a **frontier field** $H(s) = \mathrm{relu}(V^\dagger(s) - V(s))$, where $V$ is the ordinary critic. The field encodes the claim that the game's own record proves more to be achievable from a state than the policy currently collects there.

Actuation is a single potential-based shaping term $\Phi(s) = +H(s)$ [Ng et al., 1999], injected into advantages with standard-deviation matching and clamping. The sign matters more than we anticipated: the *closure* form $\Phi = -H$, which is safe on small, spatially diffuse gaps (§4.5), is catastrophic on a strong peaked field, because the cheapest way to reduce headroom along a trajectory is to walk away from it; the policy learns to avoid the ball.

**Contributions.**

1. **The composition critic**: an online, model-free, search-free mechanism for directed exploration toward unperformed conducts, requiring only two MLP value heads and a potential-based advantage term; identical environments everywhere; ≈1% wall-clock overhead; nothing at deployment (§5).
2. **A measured failure map** of the design space under these constraints: optimality-backup ceilings spiral, with truncated-quantile control only delaying divergence; executed probes calibrate but violate homogeneity and are pessimistic on deep conducts; world models learn flight physics from fragments but relocate the search problem into the dream; single-head online expectile TD exhibits a *ratchet*; closure potentials teach avoidance (§4).
3. **Benchmark results** on a Rocket League aerial-discovery task: at matched wall-clock (overhead ≈1%), 0.61/0.55 vs. 0.32 ball interaction, 0.44/0.42 vs. 0.18 airborne touch, and 0.089/0.043 vs. 0.000 high-ball aerial conversion against PPO; also exceeding a probe-assisted baseline that uses machinery our constraints forbid (§6).
4. **A production-scale integration reported as an accidental control**: a configuration fault left the composition heads frozen at random initialization for the first ≈3.4B steps of a fresh deployment run while their loss gradients flowed into the 1152-wide shared trunk. We withdraw the stability and behavioral claims that run cannot support and report what survives: a chance-level auxiliary objective coupled from step zero coincided with shared-representation health better than historical baselines, an unplanned but direct test of the auxiliary-loss co-adaptation hypothesis (§7).

---

## 2 Related work

**Offline RL and stitching.** Implicit Q-Learning [Kostrikov et al., 2022] introduced expectile regression on TD targets to approximate a maximum over dataset-supported actions without querying out-of-distribution actions; the resulting trajectory *stitching* is central to offline RL. Related in-sample value learning appears in Extreme Q-Learning [Garg et al., 2023], which fits an optimistic value via Gumbel regression. We move the expectile estimator online, where we find the naive form unstable (the expectile ratchet, §4.4), stabilize it with twin heads and minimum-target backups in the style of clipped double-Q [van Hasselt, 2010; van Hasselt et al., 2016; Fujimoto et al., 2018], and use it not for policy extraction but as an exploration field. Conservative methods such as CQL [Kumar et al., 2020] penalize exactly the optimism we require; they solve the opposite problem.

**Optimism and exploration.** Count- and prediction-error novelty (e.g., ICM [Pathak et al., 2017]; RND [Burda et al., 2019]) rewards visiting unfamiliar *states*; our problem is unattempted *conducts* from familiar states, which state novelty cannot see (we confirm: an RND drive moves nothing on our benchmark). Optimistic initialization and UCB-style bonuses inject optimism into values; our §4.2 measurements are a caution that bootstrapped optimism about untried actions, at scale and with function approximation, self-amplifies unless it is either executed (probes) or restricted to executed support (this work). Go-Explore [Ecoffet et al., 2021] and reset-based curricula attack the same acquisition problem via environment control; §4.1 documents when the returned skill fails to transfer, and our constraint set excludes the family.

**Model-based imagination and search.** Dreamer-class agents [Hafner et al., 2020; Hafner et al., 2023] and MuZero [Schrittwieser et al., 2020] obtain exactly the counterfactual competence we seek, at the cost of a learned model and, for search, planning compute. We verify the premise, since a one-step model trained on entropy fragments learns flight dynamics well before any aerial exists (§4.3), and we equally verify the relocation of the problem: the imagination actor faces the same exploration barrier inside the model, at imagination budgets far above ours. Under a hard no-model constraint these routes are unavailable; the composition critic can be read as extracting the *chaining* benefit of a model without the model, by letting the value function do the composition.

**Potential-based shaping.** Our actuation is classical potential-based reward shaping (PBRS) [Ng et al., 1999], which preserves optimal policies. The empirical content here is the *sign and field-shape* analysis: closure potentials on strong peaked fields produce avoidance (§4.5), a hazard we have not seen documented at this sharpness.

**Self-play systems and Rocket League.** Our deployment target is a large self-play league trainer in the lineage of population-based systems [Jaderberg et al., 2019; Berner et al., 2019; Vinyals et al., 2019], built for Rocket League on the RLGym/RocketSim ecosystem [RLGym; RocketSim] and informed by prior public agents in that ecosystem (e.g., Necto/Nexto). Prior aerial acquisition in this ecosystem has relied on shaped event rewards and drill state-setters; to our knowledge no public system acquires aerial pressure purely architecturally.

**Distributional critics.** Distributional value learning [Bellemare et al., 2017; Dabney et al., 2018] offers an alternative route to optimism via upper-quantile reads, with truncation available to control overestimation [Kuznetsov et al., 2020]. On our benchmark a quantile-distributional variant *underperformed* the scalar expectile head (§6.4), and truncation failed to rescue the optimality-backup ceiling (§4.2).

---

## 3 Preliminaries and problem setting

### 3.1 Setup and notation

We consider standard on-policy actor–critic training, namely PPO [Schulman et al., 2017] with GAE [Schulman et al., 2016], on an MDP with states $s$, actions $a$, rewards $r$, discount $\gamma$, and episode-boundary indicator $d$ (nonzero at terminals *and* time-limit truncations). $V$ denotes the ordinary critic, an estimate of the current policy's value $V^\pi$; $A_i$ denotes GAE advantages.

**Expectiles.** For $\tau \in (0,1)$, the $\tau$-expectile of a random variable $Y$ is $\arg\min_m \mathbb{E}\big[|\tau - \mathbb{1}[Y < m]|\,(Y - m)^2\big]$ [Newey & Powell, 1987]; as $\tau \to 1$ it approaches the supremum of the support. Expectile TD regression replaces the mean-squared TD loss with this asymmetric loss, biasing the fixed point toward the upper tail of the target distribution, both over environment stochasticity and, crucially, over the behavior distribution's action choices at each state [Kostrikov et al., 2022].

### 3.2 The acquisition problem and constraints

The target setting: an environment in which a conduct is (a) richly rewarded when performed, (b) reachable from commonly visited states, and (c) of vanishing probability under the current policy. We ask for a mechanism that produces *directed pressure* toward acquiring the conduct, subject to:

- **C1 (homogeneity):** all environment instances identical; no dedicated probe, drill, or reset instances;
- **C2 (no environment design):** no curriculum state-setters, no demonstrations, no banked reset states, no success-state memories that define the objective retrospectively;
- **C3 (no models, no search):** no learned dynamics model, no imagination rollouts, no tree search at train or test time;
- **C4 (MLP-scale cost):** per-step cost at collection and deployment unchanged; learn-phase overhead small;
- **C5 (serendipity floor):** baseline entropy regularization is never reduced or gated anywhere. Any guidance signal *adds* attention and must never veto undirected exploration. The mechanism's own estimates are not oracles; conducts whose every fragment is missing can only be seeded by chance, and that channel must stay open.

### 3.3 Benchmark: aerial discovery from cold starts

Our offline testbed is a single-agent RocketSim task distilled from the deployment game (full details in Appendix A). One car, one ball per arena; 48 parallel arenas. Every episode begins *cold*: the car on the ground rolling forward with full boost; the ball placed ahead uniformly in a 3-D box ($|x| \le 700$ uu, $y \in [300, 1200]$ ahead of the car, $z \in [100, 1500]$), so that most of the box is unreachable without flight. Reward is deliberately generic: +1 for any ball touch, a small uniform approach-shaping term, a small term for ball speed toward the goal, +10 for scoring. **No aerial-specific reward, detector, or bonus exists anywhere in the system.** Episodes run 120 decisions (≈8 s at 15 Hz).

We report three per-episode indicators, aggregated per logging window (these are logging readouts only; nothing in the reward references them):

- **touch**: fraction of episodes with any ball touch;
- **air**: fraction with an *airborne* touch (car off the ground, above 300 uu);
- **hi**: fraction of high-ball episodes (spawn $z \ge 900$) converted with an airborne touch, which is the target conduct.

The policy is a small MLP (30-dim observation → 128 → 128 → 19 discrete actions); every mechanism below adds heads and never changes the policy network. Vanilla PPO on this task learns ground play and low-ball interception and converts **zero** high-ball aerials at every budget we ran (hi = 0.000 at both 20M and 25M steps). This is the acquisition wall in its simplest reproducible form.

---

## 4 A measured map of the design space

We present the design-space study before the mechanism because every element of the final mechanism (§5) is the negation of a measured failure, and because we believe several of these failures are load-bearing knowledge for anyone attempting architectural exploration. All numbers are from the benchmark of §3.3 (2 seeds per cell unless noted; paired values are per-seed results; ranges are min/max over seeds).

### 4.1 Environment-side mechanisms work, and show why we exclude them

For calibration we implemented the classical remedies. A hand-designed spawn curriculum over ball height (practice concentrated at the lowest unmastered height band, advancing on sustained mastery) reaches **0.86–0.96** high-aerial conversion, the skyline for this task. A learned variant that *discovers* the difficulty axis, ranking candidate spawns by a learned quasimetric distance-to-success [Wang et al., 2023], matches the hand-designed axis (0.76 vs. 0.79 on solved seeds). But three observations fix our constraints:

(i) both variants require environment control, violating C2; (ii) the reset-based alternative that avoids difficulty design fails the *transfer* test: returning the agent to banked pre-success states produced an aggregate success rate of 0.244 that collapsed to **0.005** when evaluated only on cold starts, because the scaffold had short-circuited the approach phase, which *is* the skill; and (iii) a retrospective success-state memory makes the frontier definition depend on which successes luck has already produced, aiming practice at *reproducing particular states* rather than at value. These observations fix C1–C2 and the design goal: the pressure must live in the policy's own updates.

### 4.2 Unfalsifiable optimism diverges: the optimality-backup ceiling

The direct reading of "estimate the best that can be done" is a Q-function over the discrete action table trained with optimality backups: a max over *all* actions, including untried ones, valued purely by generalization. This is amortized tree search. We implemented it carefully: distributional (16 quantiles per action), double-Q action selection, a target network, and reward-scale clamps. The ceiling estimate nonetheless **spiraled**: mean $V_{\max}$ reached 42.6 and 38.3 (two seeds) against a best-ever observed episode return of ≈30–36 and *mean* returns of ≈0.2. The derived exploration drive was actively harmful, and downstream uses (e.g., drill selection by apparent headroom) aimed at noise. Adding truncated-quantile targets in the style of TQC [Kuznetsov et al., 2020], dropping the top 4 of 16 target quantiles with conservative action selection, **delayed the divergence to ≈16M steps but did not prevent it** (2 of 4 seeds re-spiraled).

The root cause is epistemic, not statistical: under an on-policy data stream, the argmax actions at most states are never *executed*, so their generalization-derived values are never corrected, and the bootstrap compounds selection bias on noise. Confirming this diagnosis, converting a small fraction of arenas into ε-greedy *probe* arenas that execute the critic's argmax (their data feeding only the critic) fully stabilized the same estimator: values were grounded for the entire run, and the mechanism became productive (touch 0.44, air 0.28–0.31 at 20M; touch 0.53–0.57, air 0.38–0.42, hi 0.048–0.060 at 40M). Executed probes, however, violate C1, and their calibration is itself bounded by execution depth: the probe's greedy chain breaks where the critic has no data, so the ceiling converges to "the best my current chain can demonstrate", which is measurably pessimistic on deep conducts. The probe-assisted field read approximately zero headroom at high balls even though ≥0.9 conversion is provably achievable there (§4.1). Probes are a fine *accelerator*; they are not the architectural answer.

### 4.3 World models learn the physics but relocate the search

A one-step dynamics model $M(s, a) \to (\Delta s, r)$ trained by supervised regression on all real transitions learns flight physics from entropy fragments alone: airborne-transition prediction error fell from 0.0084 to 0.0021, converging toward the ground-transition error (0.0013), long before any aerial success existed. The premise of imagination, that the knowledge is in the fragments, is *true*. But value-from-imagination failed to ignite at our budgets: random and policy-sampled candidate rollouts cannot compose a precise 16–20-step maneuver (the realization signal appeared only where the policy was already close), and a Dreamer-style imagination actor faces the *same* exploration barrier inside the model: it must discover the maneuver in the dream. At imagination budgets below reality scale (ours were ≈1× collected experience; Dreamer-class systems dream 10–100×), this does not happen. Imagination does not dissolve the exploration problem; it relocates it into a cheaper simulator, and the relocation only pays at large imagination budgets. The route is excluded regardless by C3.

### 4.4 Executed-support optimism, naive form: the expectile ratchet

Restricting optimism to *executed* transitions, via expectile TD with $V^\dagger$ regressed toward $y = r + \gamma V^\dagger_{\text{tgt}}(s')$ under asymmetric weight $\tau$, removes the unfalsifiable-action channel entirely. The naive online form ($\tau = 0.9$, single head, hard target refresh) nonetheless diverged, more slowly: headroom at *easy* states inflated from 0.3 to 11.8 over 20M steps with no behavioral conversion behind it. We call this the **expectile ratchet**: the asymmetric loss preferentially fits upward fluctuations of the bootstrap target, the target network inherits the inflation, and the loop compounds, with no untried actions required. Anchoring optimism to executed support is *necessary but not sufficient*; the asymmetry itself must be controlled (§5.1, §8.2).

### 4.5 The closure-potential pathology

A headroom field must be converted into behavior. PBRS adds $\gamma\Phi(s') - \Phi(s)$ to the reward (equivalently, to advantages) for any potential $\Phi$ and provably preserves optimal policies [Ng et al., 1999], but two signs are available. A *closure* potential $\Phi = -G$ pays the policy for reducing a gap $G$ along its trajectories. The production trainer of §7 had long used this form safely on a different, *small and spatially diffuse* gap: the positive part of the difference between a return-level expectile head and the mean critic (a "knowing–doing" gap, meaning what the policy sometimes achieves minus what it reliably achieves; see §8.4). On such a gap, paying for closure pushes behavior toward consistency.

Transplanting the same form onto the headroom field ($\Phi = -H$) was catastrophic: **touch collapsed to 0.014/0.027 versus 0.244 for PPO**. The drive actively *destroyed* ball interaction. The mechanism is geometric: when the field is strong and spatially peaked (large near the ball, small elsewhere), the cheapest way to reduce $\Phi$-potential along a trajectory is to *leave the peak*, so the policy is paid to walk away from its own frontier. Sign and field shape interact: closure potentials are only safe on weak or flat fields. The *seek* form $\Phi = +H$ (credit for *entering* headroom) inverts the geometry and, being potential-based, still preserves optimal policies.

### 4.6 Two more measured negatives

**Headroom-modulated entropy.** Setting a per-state entropy coefficient $\eta(s) = \eta_0(1 + k\tilde H(s))$ (baseline floor preserved) at $k = 1$ *hurts*: touch 0.208/0.214 vs. 0.322 for the base, because the extra policy noise at frontier states outweighs the attention benefit at this scale. **Distributional $V^\dagger$.** A quantile head with optimism read at q75 and the backup carrying the upper quantile underperforms the scalar expectile head (touch 0.28/0.34 vs. 0.41/0.37). Neither is part of the final mechanism.

---

## 5 The composition critic

The final mechanism is the conjunction of the failure map's negations. It adds, to an otherwise unmodified PPO learner, two value heads, one derived field, and one advantage term.

### 5.1 Twin composition heads

Two value heads $V_1^\dagger, V_2^\dagger$ (testbed: independent 2×256 MLPs on raw observations; deployment: heads on the shared trunk, §7). Each iteration, one-step TD targets are formed over the freshly collected buffer, using executed transitions only:

$$y_i \;=\; r_i \;+\; \gamma\,(1 - d_i)\, \min\!\big(V^\dagger_{1,\text{tgt}}(s_{i+1}),\, V^\dagger_{2,\text{tgt}}(s_{i+1})\big), \tag{1}$$

where $d_i$ is nonzero at every trajectory boundary (both terminals and time-limit truncations cut the bootstrap), and the target heads are a periodic copy (testbed: every 8 iterations) or simply the previous iteration's heads (deployment). Both heads are trained by expectile regression at $\tau = 0.75$:

$$\mathcal{L}(V_h^\dagger) \;=\; \mathbb{E}_i\big[\, |\tau - \mathbb{1}[u_i < 0]|\; u_i^2 \,\big], \qquad u_i = y_i - V_h^\dagger(s_i), \tag{2}$$

with targets clamped to a fixed value-scale range. The **minimum over twins inside the target** is the anti-ratchet: upward generalization noise must appear in *both* heads at the same state before it can propagate. This is the same selection-bias suppression that clipped double-Q provides for the max-over-actions bias [van Hasselt, 2010; Fujimoto et al., 2018], here applied to the expectile's implicit max-over-dataset. With this control, and with $\tau$ at a moderate asymmetry in the range used by IQL rather than 0.9, $V^\dagger$ remained bounded and drift-free in every testbed run reported in this paper, through 40M+ steps. (No at-scale evidence exists yet: the deployment's heads never trained; see §7.)

**Why this object sees unperformed conducts.** Each executed transition contributes an *inequality*, namely that this step was possible here, and high-expectile TD approximates the upper envelope consistent with all of them. Value then flows backward along any chain of inequalities (trajectory A's jump, trajectory B's airborne boost, trajectory C's low aerial touch) linked wherever the function approximator identifies their intermediate states. The unit of required luck falls from "a completed conduct" (astronomically rare) to "each step, somewhere, ever" (supplied by baseline entropy). The honest boundary is the same statement's contrapositive: a conduct containing a step executed *nowhere* is invisible to $V^\dagger$ until entropy supplies that link once, which is why C5 is a constraint and not a preference.

### 5.2 The headroom field

$$H(s) \;=\; \mathrm{relu}\Big(\min\big(V_1^\dagger(s), V_2^\dagger(s)\big) - V(s)\Big), \tag{3}$$

where $V$ is the ordinary critic. $H$ is a *zone-of-proximal-development* field: zero where the policy already collects what is achievable, zero where nothing more is provably composable, positive exactly where the game's own record exceeds current behavior. It is self-retiring: as the policy learns to collect value at a state, $V$ rises and $H$ closes there; as new fragments enter experience, $V^\dagger$ rises and $H$ opens further out. No controller, no thresholds, no memory beyond the two heads.

### 5.3 Seek actuation

A potential-based term with $\Phi = +H$:

$$g_i \;=\; \gamma\,(1-d_i)\,H(s_{i+1}) \;-\; H(s_i), \tag{4}$$

centered, scale-matched to a fraction $\beta$ of the extrinsic advantage standard deviation ($\sigma$-ratio matching with a floor), clamped to $\pm 3\sigma_{\text{ext}}$, and added to the advantages before the PPO update. Rescaling by a constant preserves the potential-based form ($\Phi' = c\Phi$); per-batch centering is a constant advantage shift absorbed by PPO's advantage normalization; the clamp is a safety bound that departs from exact invariance only where it binds. $\beta$ is the mechanism's single important hyperparameter (§6.3). Baseline entropy regularization is untouched (C5).

### 5.4 Algorithm

**Algorithm 1. PPO with a composition critic (one iteration).**

```
Input: policy π_θ, critic V_φ, twin heads V†_1, V†_2 with targets Ṽ†_1, Ṽ†_2;
       expectile τ = 0.75, seek scale β, discount γ.

1. Collect the on-policy buffer B = {(s_i, a_i, r_i, d_i, s_{i+1})} with π_θ.
   (Collection is unchanged; the policy never consumes H.)
2. Compute V_φ(s_i) and GAE advantages A_i. If the learner normalizes or
   clips rewards inside GAE, recover scaled rewards via Eq. (5).
3. Form targets y_i = r_i + γ(1−d_i)·min(Ṽ†_1(s_{i+1}), Ṽ†_2(s_{i+1}));
   clamp to the value-scale range.
4. Train both heads on the expectile loss (Eq. 2) over B.
5. Compute H_i = relu(min(V†_1, V†_2)(s_i) − V_φ(s_i)) for all i (no grad).
6. Form g_i = γ(1−d_i)H_{i+1} − H_i; center; scale to β·σ(A) (with floor);
   clamp to ±3σ(A); set A_i ← A_i + g_i.
7. Run the standard PPO update of π_θ and V_φ on the modified advantages;
   the entropy bonus is unchanged (C5).
8. Refresh targets (copy every 8 iterations in the testbed; previous
   iteration's heads at deployment).
```

### 5.5 Cost accounting (C4)

Collection and deployment are unchanged, since the policy never consumes $H$; all computation happens at learn time: two head forward/backward passes over the buffer plus one no-grad forward for targets and the field. Measured on the testbed under identical machine load: **26,313/26,222 steps/s versus 26,615 for PPO (0.99×)**. At deployment the heads are two 1152-input MLPs in a system already running several critic heads; the overhead is low single-digit percent of the learn phase.

### 5.6 Integration into production learners

Production learners often normalize or clip rewards inside GAE, so raw rewards are in the wrong units for a TD target meant to be compared with the critic $V$. The scaled one-step reward can be recovered *from GAE outputs alone*. Since $A_i = \delta_i + \gamma\lambda(1-d_i)A_{i+1}$ with $\delta_i = r_i + \gamma(1-d_i)V_{i+1} - V_i$, we have $\delta_i = A_i - \gamma\lambda(1-d_i)A_{i+1}$ and therefore

$$r^{\text{scaled}}_i \;=\; A_i \;-\; \gamma\lambda(1-d_i)A_{i+1} \;-\; \gamma(1-d_i)V_{i+1} \;+\; V_i, \tag{5}$$

where the $A$ are the *pre-injection* advantages. With this identity the integration reduces to three insertion points (the heads and their loss, target formation at learn-prep, and one advantage-injection site) with no changes to GAE or to collection.

---

## 6 Offline experiments

Setup as in §3.3; full environment and hyperparameter details in Appendices A–B. All mechanism arms use PPO hyperparameters identical to the baseline; the only differences are the added heads and the injection. Seeds: 2 per mechanism cell; the mechanism-vs-baseline comparison at 25M uses a baseline run collected in the same experimental wave on the same machine.

### 6.1 Main results

| Arm | Budget | touch | air | hi (aerial conversion) |
|---|---|---|---|---|
| PPO | 20M | 0.244 | 0.067 | 0.000 |
| PPO | 25M | 0.322 | 0.178 | 0.000 |
| **+ Composition critic** (β = 0.05) | 25M | 0.41 / 0.37 | 0.21 / 0.19 | 0.002 / 0.002 |
| **+ Composition critic** (β = 0.15) | 24M | 0.43 / 0.46 | 0.24 / 0.27 | 0.009 / 0.013 |
| **+ Composition critic** (β = 0.30) | 24M | 0.53 / 0.48 | 0.36 / 0.30 | 0.033 / 0.023 |
| **+ Composition critic** (β = 0.05) | 40M | 0.51 / 0.49 | 0.34 / 0.33 | 0.036 / 0.037 |
| **+ Composition critic** (β = 0.15) | 40M | **0.61 / 0.55** | **0.44 / 0.42** | **0.089 / 0.043** |
| *Probe-assisted ceiling critic (violates C1; reference)* | 40M | 0.53 / 0.57 | 0.38 / 0.42 | 0.048 / 0.060 |
| *Spawn curriculum (violates C2; skyline)* | 20–25M | n/a | n/a | 0.86–0.96 |

Because the overhead is ≈1%, equal-steps is equal-wall-clock to within the baseline's own extrapolated gain over that margin (≈+0.005 touch). At iso-compute the mechanism yields **≈1.5–1.7×** touch and **≈1.7–2.0×** air at 24–25M (β = 0.30), growing to **≈2.4× / 6.3×** at 40M (β = 0.15, against the 20M baseline's own trend continued), with aerial conversion strictly positive and rising against a flat zero. Under the full constraint set the composition critic also **outperforms the probe-assisted estimator** that is allowed to execute its own optimism. This is consistent with §4.2's finding that execution-calibrated ceilings are pessimistic on deep conducts: composition-based ceilings keep credit flowing from fragments the probes cannot chain.

The environment-side skyline remains far ahead on the target conduct at these budgets. The claim of this paper is not that architecture beats environment design where environment design is available; it is that *when the constraint set forbids environment design*, a purely architectural mechanism moves an unmovable metric, and its trajectory at 40M (hi rising from ≈0.001 to 0.089 with no plateau) indicates acquisition in progress, not saturation.

### 6.2 The frontier wave

The headroom field's spatial evolution is directly observable by probing $H$ on a fixed grid of canonical spawn states of increasing ball height. Early in training, $H$ peaks at low heights, where the record proves the policy could already be touching the ball, while high balls read near zero (no fragments to chain yet). As low-ball play consolidates, $H$ collapses there and the peak migrates upward, tracking the boundary between mastered and composable-but-uncollected; airborne-touch fragments accumulate, extending $V^\dagger$'s reach; high-ball conversion begins as the wave arrives. This *frontier wave*, the intended zone-of-proximal-development dynamics, emerges from the two heads alone, with no controller, and is the qualitative signature by which we distinguish a healthy run from both failure modes: a ratcheting run (§4.4) shows $H$ growing everywhere without behavioral change; a closure-sign run (§4.5) shows the policy vacating the peak.

### 6.3 Dose–response

$\beta$ sweeps cleanly at 24–25M: air 0.21/0.19 (β = 0.05) → 0.24/0.27 (0.15) → 0.36/0.30 (0.30); hi 0.002 → 0.009/0.013 → 0.033/0.023; with $V^\dagger$ bounded at every dose (final means 1.2–2.1). We adopted β = 0.15 for the long-budget benchmark and for deployment as the middle of the productive range; β = 0.30 buys speed at 25M-scale budgets and showed no instability, so the range is forgiving. At β = 0.15 the injected term averages ≈6% of advantage scale.

### 6.4 Ablations

Each row is a measured arm (§4 for details):

| Ablation | Result | Lesson |
|---|---|---|
| max over untried actions (optimality backup) | ceiling ≈42 vs. best-ever return ≈36; drive harmful | unfalsifiable optimism diverges |
| + TQC truncation | spiral delayed to ≈16M, not prevented | statistics cannot fix an epistemic hole |
| + executed probes (violates C1) | stable, productive; field pessimistic on deep conducts | execution calibrates *and* bounds |
| single head, τ = 0.9 | H at easy states 0.3 → 11.8; no conversion | the expectile ratchet |
| twin-min + τ = 0.75 (ours) | bounded for 40M+ | selection-bias suppression transfers to expectiles |
| closure sign Φ = −H | touch 0.014/0.027 (vs. 0.244) | peaked fields + closure potentials ⇒ avoidance |
| seek sign Φ = +H (ours) | main results | PBRS with geometry-correct sign |
| entropy modulation (k = 1) | touch 0.21 vs. 0.32 | noise where precision is needed |
| distributional q75 variant | touch 0.28/0.34 vs. 0.41/0.37 | scalar expectile suffices; upper-quantile reads re-import inflation |
| world model + imagination | physics learned; ignition failed at ≤1× dream budget | search relocated, not removed |

### 6.5 Compute

Steps-per-second ratio 0.99, measured under identical machine co-load; all overhead in the learn phase; zero additional cost at collection or deployment. At fixed wall-clock, the comparison in §6.1 is therefore unchanged to within a ±1% step budget.

---

## 7 Deployment at scale: an accidental control

We integrated the mechanism into a production-scale Rocket League self-play trainer: a C++ PPO league system with a 1152-wide shared trunk feeding the policy head and several critic heads, a ≈90-entry discrete action table, auxiliary systems including a long-horizon goal-scoring critic and reachability-prediction auxiliary losses, a rating-based population league, and collection throughput in the hundreds of thousands of environment steps per second. Integration followed the recipe of §5.6: twin heads on the shared trunk (in this run, mirroring the main critic head's architecture at ≈4.0M parameters each), targets frozen for one iteration, reward reconstruction via the GAE identity (Eq. 5), the seek term injected at the single advantage-assembly site (β = 0.15, σ-matched, clamped), coverage by the trainer's standing safety latch (an automatic mechanism that disables experimental reward terms on a sustained league-rating drawdown), and telemetry for the field and the injection. One deliberate departure from the testbed configuration: the composition heads share the trunk *with gradients flowing*, on a fresh run. This engages a hypothesis from the trainer's operational history: in an earlier incident, coupling a freshly initialized, chance-level auxiliary loss into a *mature* trunk destroyed ≈125 Elo across all game modes within ≈500 iterations, whereas auxiliary losses present from initialization had been neutral to beneficial; the working hypothesis distinguishes *co-adaptation from step zero* from mid-run grafting.

**The fault.** A post-hoc code audit found that learning rates for the new heads had never been configured: the model constructor builds every optimizer parameter group at a learning rate of zero, and under the optimizer in use this is an exact no-op. Both composition heads therefore sat at their random initialization for the entire window reported below, covering ≈3.4B environment steps and ≈12k iterations, with the fault corrected only afterward. The consequences divide sharply. The *heads* never trained: $V^\dagger$ was a fixed random readout of the (evolving) trunk, $H = \mathrm{relu}(V^\dagger - V)$ was structured noise, and the seek term, mechanically active at β = 0.15 throughout, actuated a field with no headroom semantics. The *loss*, however, was live: the heads' expectile-TD objective joined the total loss undetached, so its gradients flowed into the shared trunk at full weight for the whole window.

**Withdrawn claims.** An earlier draft of this paper read the window as a successful deployment; every such reading is vacuous, and we withdraw it explicitly. That $V^\dagger$ remained bounded with no drift is trivially true of a frozen readout and is not evidence that the twin-min anti-ratchet works at scale. That the $H$ statistics ran inside the testbed's healthy band is coincidence, not corroboration. Rises in mean touch height and aerial-touch ratio over the window cannot be attributed to the drive, which was noise, and belong, if real, to the fresh run itself. And that the rating latch never fired is near-vacuous: its trip condition (a drawdown exceeding ≈627 Elo per 10⁹ steps within a 0.5B-step window, under its most sensitive arm) is a bar only a collapsing run would test. There is also an engineering lesson in how long the fault survived: boundedness- and scale-level telemetry cannot distinguish a healthy field from a frozen one; a semantic check, asking whether high $H$ precedes above-critic realized returns, would have caught it immediately.

**What the accidental control does show.** The trainer's standing diagnostic of auxiliary-gradient pressure on the shared trunk, the same scalar whose excursion from ≈0.5 to 1.36 accompanied the historical incident, ran at 0.25–0.29 over the window, against a documented healthy-baseline band of 0.47–0.52. Read correctly, this is *not* evidence that composition-value prediction acts as representation learning: no composition values were learned. It is evidence that gradients from a fixed-random-readout auxiliary objective, present from initialization, left trunk health better than baseline, a result closer in character to random-feature auxiliary regularization. It also bears on the co-adaptation hypothesis more directly than the planned experiment would have: the historical incident's harmful graft was precisely a chance-level auxiliary loss coupled into a mature trunk, and here a *permanently* chance-level auxiliary loss coupled from step zero produced no such signature, isolating trunk maturity, rather than the auxiliary's eventual competence, as the operative variable. This remains a single uncontrolled run, and we weight it accordingly.

**A second confound, and the corrected configuration.** Even had the heads trained, the window would not have isolated the mechanism: the run carried four concurrent advantage injections (the seek term at β = 0.15 alongside a goal-horizon-critic drive at weight 0.25, an RND novelty bonus at 0.10, and a closure-form ladder drive at 0.05; §4.5, §8.4), as well as a five-column policy-input wiring not described in §5. A subsequent conformance pass removed the additional injectors and the policy wiring and corrected the learning-rate fault, so the forthcoming re-deployment will be the first at-scale run that matches §5 as specified; its head configuration also differs from the faulted run's (Appendix B). All at-scale claims for the mechanism await that run.

---

## 8 Analysis

**Why composition escapes the impossibility argument.** The execution principle (§1) says optimism about never-executed action sequences is unfalsifiable without a model, search, or an attempt. $V^\dagger$'s optimism is about *values*, constrained to the cone generated by executed one-step inequalities: every unit of claimed headroom is a sum along a chain of steps that individually happened. Falsification is therefore piecewise and automatic, since each link continues to be re-estimated from real data every iteration, and the estimator inherits the model-free trainer's speed because chaining is performed by the value function's fixed point rather than by rollouts. What is genuinely given up is visibility into conducts with a globally missing link; C5 (the entropy floor) is the standing channel by which such links eventually appear, at which point $V^\dagger$ propagates their value backward within iterations, not epochs.

**Why the twins are necessary online but not offline.** Offline, the dataset is fixed: expectile regression fits an envelope over a static set of inequalities. Online, the target network's inflation re-enters the regression targets, and the asymmetric loss preferentially retains it, a positive feedback absent in the offline setting. The min-over-twins breaks the loop by requiring inflation to be *reproduced independently* before it can propagate: the same decorrelation logic as clipped double-Q, transplanted from the max-over-actions to the expectile's implicit max-over-support.

**Why the sign of the potential is decisive.** For any potential $\Phi$, PBRS credit along a step is $\gamma\Phi(s') - \Phi(s)$. If $\Phi = -H$ and $H$ is large on a compact region (the frontier), trajectories are paid for *exiting* the region: the drive is a repellent exactly where attention is wanted, and the policy converges to orbiting away from its own frontier (measured: ball interaction collapsed roughly tenfold, §4.5). With $\Phi = +H$ the same field is an attractor; loitering is not rewarded, because potential differences telescope, and *resolving* the headroom pays doubly: once through the extrinsic reward that resolution earns, and once through $V$ catching up so that the wave moves on. Closure-form injections remain appropriate for weak, diffuse gaps (§4.5); the general rule our measurements support is: *seek strong peaked fields; close weak diffuse ones.*

**A hierarchy of optimism.** It is useful to place $V^\dagger$ on a spectrum of value estimates ordered by the optimism of their training targets. The ordinary critic $V$ estimates the mean return of current behavior, which is what the policy *reliably* collects. A return-level expectile $V^{\text{exp}}$, obtained by expectile regression against realized (whole-return) targets, estimates what current behavior *sometimes* achieves from a state; the production trainer of §7 already carried such a head, and its small, diffuse gap to $V$ is the knowing–doing gap on which closure shaping is safe (§4.5). The composition critic completes the progression: by taking the expectile at the level of one-step Bellman targets rather than whole returns, it values *recombinations* of executed steps: outcomes constructible, not merely outcomes experienced. That last step is precisely what moves the aerial-conversion metric off zero, and it is the smallest such step our failure map found that does not diverge.

---

## 9 Limitations

1. **Fragment coverage is the hard boundary.** Conducts with a globally unexecuted step are invisible to $V^\dagger$ until entropy supplies the link. Environment-side accelerators (probes, drills) compose cleanly with the mechanism and close this residue faster where they are acceptable; under the strict constraint set, acquisition speed for such conducts is bounded by serendipity.
2. **The environment skyline is not reached.** At 40M steps, the spawn curriculum's 0.86–0.96 aerial conversion is far above our 0.04–0.09. The mechanism's trajectory has not plateaued, but we have not run it to convergence; the honest current claim is *acquisition where PPO has none*, not parity with environment design.
3. **The at-scale evidence is currently null.** A configuration fault left the composition heads untrained for the entire deployment window (§7): the treatment was never applied, so no at-scale evidence yet exists for the trained mechanism's stability or behavioral effect, and the faulted run was additionally confounded by concurrent advantage injections. The corrected, conformant re-deployment is in progress.
4. **Seed counts are small** (n = 2 per cell; cross-seed variance is visible in the reported ranges), and the benchmark is a single task family in a single simulator; until the corrected deployment reports, all evidence for the mechanism comes from this one testbed.
5. **Two hyperparameters carry real sensitivity**: $\tau$ (0.9 ratchets even beyond the single-head setting; 0.75 is stable) and $\beta$ (measured productive range 0.05–0.30). We have not characterized their interaction, and transfer to deployment scale is untested (§7).
6. **Non-stationarity of self-play** is only partially addressed: $V^\dagger$'s inequalities are re-estimated continuously, but an opponent-induced collapse of previously composable value would manifest as stale headroom until the relevant transitions are revisited.

---

## 10 Conclusion

The acquisition wall, where behaviors that never happen therefore never learn, is usually breached from the environment side. We showed it can be breached from the *architecture* side, under constraints that forbid every standard tool: no curricula, no resets, no probes, no models, no search, no extra deployment cost. The route is an accounting identity taken seriously: everything an agent has ever done is evidence about what it could do, and a value function trained to hold the upper envelope of that evidence, carefully, with the asymmetry double-checked by twin estimators and actuated with the sign that attracts rather than repels, turns scattered fragments of competence into directed pressure toward skills that have never once occurred. On our benchmark this moved high-ball aerial conversion from exactly zero to steadily climbing at ≈1% overhead, outperforming an execution-assisted ceiling under the same budget. At production scale, a configuration fault turned the first integration into an accidental control, instructive about auxiliary-gradient co-adaptation on a shared trunk and silent about the mechanism itself, leaving the at-scale test still ahead. The measured failure map of spiral, ratchet, and avoidance is offered as reusable engineering knowledge for anyone building optimism into value functions online: *ground it in execution, double-check its asymmetry, and mind the sign of the potential.*

---

## Acknowledgments

The experimental program (design-space search, testbed implementation, trainer integration, and the drafting of this manuscript) was carried out with an AI research assistant (Claude, Anthropic) operating under the author's direction; the design constraints, the hierarchy-of-optimism framing, the shared-trunk deployment decision, and the final mechanism selection are the author's.

## Reproducibility statement

Appendices A and B specify every environment and mechanism hyperparameter needed for reimplementation, and §5.6 specifies the three learner insertion points (heads and loss; target formation at learn-prep; a single advantage-injection site) together with the reward-reconstruction identity (Eq. 5) required when the host learner normalizes rewards inside GAE. The offline testbed, comprising the benchmark environment, all arms of §4 (including the probe-assisted and imagination baselines), and the final mechanism, will be released together with experiment logs at [repository URL to be added].

## References

- Bellemare, M. G., Dabney, W., Munos, R. (2017). A Distributional Perspective on Reinforcement Learning. *ICML*.
- Berner, C., et al. (2019). Dota 2 with Large Scale Deep Reinforcement Learning. *arXiv:1912.06680*.
- Burda, Y., Edwards, H., Storkey, A., Klimov, O. (2019). Exploration by Random Network Distillation. *ICLR*.
- Dabney, W., Rowland, M., Bellemare, M. G., Munos, R. (2018). Distributional Reinforcement Learning with Quantile Regression. *AAAI*.
- Ecoffet, A., Huizinga, J., Lehman, J., Stanley, K. O., Clune, J. (2021). First Return, Then Explore. *Nature* 590 (Go-Explore).
- Fujimoto, S., van Hoof, H., Meger, D. (2018). Addressing Function Approximation Error in Actor-Critic Methods. *ICML* (TD3).
- Garg, D., Hejna, J., Geist, M., Ermon, S. (2023). Extreme Q-Learning: MaxEnt RL without Entropy. *ICLR*.
- Hafner, D., Lillicrap, T., Ba, J., Norouzi, M. (2020). Dream to Control: Learning Behaviors by Latent Imagination. *ICLR*.
- Hafner, D., Pasukonis, J., Ba, J., Lillicrap, T. (2023). Mastering Diverse Domains through World Models. *arXiv:2301.04104* (DreamerV3).
- Jaderberg, M., et al. (2019). Human-Level Performance in 3D Multiplayer Games with Population-Based Reinforcement Learning. *Science* 364.
- Kostrikov, I., Nair, A., Levine, S. (2022). Offline Reinforcement Learning with Implicit Q-Learning. *ICLR*.
- Kumar, A., Zhou, A., Tucker, G., Levine, S. (2020). Conservative Q-Learning for Offline Reinforcement Learning. *NeurIPS*.
- Kuznetsov, A., Shvechikov, P., Grishin, A., Vetrov, D. (2020). Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics. *ICML* (TQC).
- Newey, W. K., Powell, J. L. (1987). Asymmetric Least Squares Estimation and Testing. *Econometrica* 55(4).
- Ng, A. Y., Harada, D., Russell, S. (1999). Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping. *ICML*.
- Pathak, D., Agrawal, P., Efros, A. A., Darrell, T. (2017). Curiosity-Driven Exploration by Self-Supervised Prediction. *ICML* (ICM).
- RLGym. A Reinforcement Learning Environment for Rocket League. Software: rlgym.org.
- RocketSim. An Open-Source Rocket League Physics Simulation. Software: github.com/ZealanL/RocketSim.
- Schrittwieser, J., et al. (2020). Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model. *Nature* 588 (MuZero).
- Schulman, J., Moritz, P., Levine, S., Jordan, M., Abbeel, P. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation. *ICLR*.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O. (2017). Proximal Policy Optimization Algorithms. *arXiv:1707.06347*.
- van Hasselt, H. (2010). Double Q-Learning. *NeurIPS*.
- van Hasselt, H., Guez, A., Silver, D. (2016). Deep Reinforcement Learning with Double Q-Learning. *AAAI*.
- Vinyals, O., et al. (2019). Grandmaster Level in StarCraft II Using Multi-Agent Reinforcement Learning. *Nature* 575 (AlphaStar).
- Wang, T., Torralba, A., Isola, P., Zhang, A. (2023). Optimal Goal-Reaching Reinforcement Learning via Quasimetric Learning. *ICML*.

---

## Appendix A: Benchmark details

**Environment.** RocketSim (faithful Rocket League physics), 48 parallel arenas, tick-skip 8 (15 Hz decisions), 120-decision episodes (≈8 s). One car, one ball.

**Observation.** 30 dimensions: car pose, velocity, orientation, boost, ground-contact and flip state; ball position and velocity; relative terms.

**Action table.** 19 curated discrete actions (drive/steer/jump/boost/pitch/yaw/roll combinations).

**Cold spawn.** Car at $(x \sim U[-300, 300],\ y = -900,\ z = 17)$ uu, velocity $(0, 900, 0)$, facing $+y$, boost 100. Ball at $(x_{\text{car}} + U[-700, 700],\ U[300, 1200],\ U[100, 1500])$ uu, at rest.

**Reward.** +1 for any ball touch; approach shaping $0.02 \cdot \Delta(\text{dist})/100$ (uniform, with no airborne multiplier); $0.002 \cdot \max(0, v_y^{\text{ball}})$ for ball speed toward the goal; +10 for scoring. No aerial-specific term of any kind.

**Metrics.** Per-episode indicators aggregated per logging window: *touch* (any ball touch), *air* (an airborne touch: car off ground, above 300 uu), *hi* (restricted to episodes with ball spawn $z \ge 900$; requires an airborne touch with car $z > 300$). All three are logging readouts only.

**PPO.** $\gamma = 0.995$, $\lambda = 0.95$, clip 0.2, entropy coefficient 0.02, learning rate $3 \times 10^{-4}$, 3 epochs × 4 minibatches per iteration, 6,144 steps per iteration. Policy network 30 → 128 → 128 → 19.

## Appendix B: Mechanism hyperparameters

**Testbed.** Heads: two independent MLPs, 30 → 256 → 256 → 1; Adam, learning rate $3 \times 10^{-4}$, gradient clip 1.0; 2 epochs × 4 minibatches per iteration. Expectile $\tau = 0.75$. Target heads copied every 8 iterations. Target clamp $[-5, 30]$. Seek scale $\beta \in \{0.05, 0.15, 0.30\}$; $\sigma$-floor $0.05\,\sigma_{\text{ext}}$; injection clamp $\pm 3\sigma_{\text{ext}}$.

**Deployment: faulted first run (§7).** Heads mirrored the main critic head's configuration on the 1152-wide shared trunk (≈4.0M parameters each), gradients coupled to the trunk from step 0 of a fresh run; head learning rates were never configured (built at zero, the fault of §7), so head parameters never updated; targets were the previous iteration's heads; $\beta = 0.15$, $\sigma$-matched and clamped as in the testbed; four concurrent advantage injectors and a five-column policy-input wiring were active (§7).

**Deployment: corrected configuration (re-deployment in progress).** Heads are 5-layer × 1280-wide MLPs on the same trunk (≈8.05M parameters each), with learning rates set; targets one-iteration frozen; value-scale clamp $\pm 2 \cdot \mathrm{p99}\,|y|$; $\beta = 0.15$, $\sigma$-matched and clamped; the seek term is the sole auxiliary advantage injection, computed at learn-prep on pre-blend advantages at the single advantage-assembly site; covered by the trainer's rating-drawdown safety latch; the field and injection are monitored via dedicated telemetry.
