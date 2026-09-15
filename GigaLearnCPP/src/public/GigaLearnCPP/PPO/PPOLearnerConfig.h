#pragma once
#include <RLGymCPP/BasicTypes/Lists.h>

#include "../Util/ModelConfig.h"

namespace RLGC {
}

namespace GGL {

	// Learned goal-reachability (InfoNCE + HER), used ONLY as:
	//  (a) an auxiliary representation loss on the shared head, and
	//  (b) the input of a multiplicative reward gate on the WeightedReward::gated components.
	// It never enters advantages, returns, or the critic (the coupling that froze prior designs).
	// Two heads share one phi(trunk(obs), action) encoder:
	//  CAR  head: "can this car reach the ball" (HER on the car-local ball)
	//  BALL head: "can the ball reach the opponent net" (HER on the canonical ball state)
	// ---------------------------------------------------------------------------------------
	// FRONTIER (research/reports/WORLD_MODEL_W163_VALUE_MAP.md, WORLD_MODEL_W161_STEPWISE_SIL.md)
	//
	// Toy result this is a port of: a policy that never performs a multi-step conduct acquires and
	// RETAINS it from purely on-policy collection, with nothing injected, when self-imitation is
	// aimed at goals chosen by a VALUE MAP built by backward value iteration. The value map is the
	// load-bearing piece; four other goal-ranking signals (novelty, learned headroom, simulated
	// return, policy advantage) all produced exactly zero acquisition across every seed.
	//
	// Two properties the toy proved are essential and are preserved here:
	//   * The backup takes a MAX over actions, not an average over what the policy did. V^pi at a
	//     state the policy never enters reports what the CURRENT policy would get there, which is
	//     low, and is therefore the wrong question. The max never needs a good policy as input.
	//   * The bootstrap target must be BOUNDED. Without it the toy's value map diverged from 0.25
	//     to 38,187 in 32 sweeps; twin nets trained on identical targets are too correlated for
	//     their minimum to act as pessimism on its own. Same failure shape as the composition
	//     critic's headroom running 0.3 -> 11.8. valueAbsMax is that bound and it is REQUIRED.
	//
	// Everything here is an instrument: it never enters advantages, returns or the critic, and the
	// trunk read is DETACHED (Law 2/4 - probes never reshape the trunk; the carStateHead incident
	// cost ~125 Rating when this was violated).
	struct FrontierConfig {
		bool enabled = false;      // Train the value map + quasimetric, publish diagnostics
		bool silEnabled = false;   // Actually add the progress-weighted imitation term (requires enabled)

		// Value map
		float gamma = 0.997f;      // Re-derive with tickSkip, like TRAIN_GAMMA
		float valueExpectile = 0.9f; // Soft max over the action distribution when actions cannot be
		                             // expanded; 0.5 would be a mean and is never what we want
		float valueAbsMax = 0;     // REQUIRED bound on |target|. 0 disables the module at boot with
		                           // a loud error rather than silently allowing divergence.
		float valueLr = 3e-4f;
		int valueSteps = 8;        // Backups per learn iteration
		int valueBatch = 4096;

		// Quasimetric reachability, d(x->y) = max_i relu(h_i(y)-h_i(x)) + ||g(y)-g(x)||
		int latentAsym = 32;
		int latentSym = 8;
		float quasiLr = 3e-4f;
		float quasiConstraintWeight = 50.0f; // Penalty on d(s,s') > 1 for OBSERVED one-step transitions
		float quasiSpreadCap = 64.0f;        // Soft cap so the max-distance term cannot diverge
		int quasiSteps = 8;
		int quasiBatch = 2048;

		// Goal selection. The band is in DECISIONS and is converted to metric units at use time
		// by multiplying the module's live one-step distance (Frontier/Quasi Local D). It was
		// fixed in metric units until 2026-09-15, which was wrong twice over: the metric's scale
		// drifts as it trains (localD 0.42 -> 0.53 in one hop, silently shrinking the band), and
		// [1.5, 6.0] units measured against a real rollout pool covered only the nearest 1% of
		// candidate pairs - the median pair sits at 35 units / 75 decisions. A band pinned to the
		// bottom percentile can only ever select states the policy is about to reach anyway,
		// which is exactly what the 591.7B inspection found it doing.
		float bandLowDecisions = 6.0f;
		float bandHighDecisions = 100.0f;

		// How the goal is chosen among the in-band candidates.
		//   VALUE  - largest value gain. Exploits the map; measured to select "ball fast toward
		//            the net" and to pick flip-reset states at 0.46x their base rate.
		//   RANDOM - uniform. The cheapest test of "pull regardless of value".
		//   RARITY - the in-band candidate FARTHEST from the current state population, i.e. the
		//            reachable place the policy least often goes. Self-limiting: once the policy
		//            goes there often it stops being rare and the pull moves on.
		//   PROGRESS - the in-band candidate where the map is still LEARNING, |V - V_lagged|.
		//            This is the map-completion signal, and it is the only novelty-shaped one
		//            that survives measurement here: count-based rarity is NEGATIVELY correlated
		//            with learning progress (-0.32 over 57.6k frames), because this policy's rare
		//            states are degenerate (ball idle in a corner) and therefore already learned.
		//            Under opponent conditioning it is also what re-fires when the opponent
		//            changes: a region settled against an old version is unlearned against a new
		//            one, so "the map goes blank there" becomes a measurable quantity.
		enum GoalSelect { GOALSEL_VALUE = 0, GOALSEL_RANDOM = 1, GOALSEL_RARITY = 2,
		                  GOALSEL_PROGRESS = 3 };
		int goalSelect = GOALSEL_VALUE;
		int progressLagIters = 100;  // refresh interval of the lagged value snapshot

		// Recent observations retained as goal candidates. At 4096 against 16,704 rows/iter/rank
		// the pool was a QUARTER of one iteration, so anything rare was evicted within seconds -
		// a 1-in-100k state was present ~4% of the time. Cross-arena transfer of rare states is
		// the whole discovery mechanism here, and it needs the pool to actually retain them.
		int candidatePool = 65536;
		int goalTtl = 330;         // Decisions a goal is held / credited over: 22s at 15Hz

		// Progress-weighted self-imitation. W161 measured the ORIGINAL endpoint-prefix rule
		// beating per-transition credit 2 seeds to 1, so the prefix rule is what is ported.
		float silSharpness = 6.0f; // weight = exp(sharpness * progress), progress in [0,1]
		float silCoeff = 0.005f;
		int silRows = 1024;        // Rows sampled into each minibatch
		int silWindows = 256;      // Candidate prefix windows scored per iteration
		int silMinWindow = 30;     // A window shorter than this (episode ended) is unusable

		// ===== THE RATCHET =====
		// A per-iteration attraction term has no memory: if the policy stops reaching a good
		// approach, the credit vanishes with it and PPO is free to discard the capability. That
		// is the failure this program already measured - PPO found EV 0.984 four times in the toy
		// and threw it away every time - and it is why the rehearsal bank, not the attraction
		// term, is what solved the toy acquisition wall.
		//
		// So goals PERSIST, and each one remembers the closest approach ever achieved toward it
		// (the RECORD) together with the prefix that achieved it. That prefix is rehearsed every
		// iteration so the policy cannot regress off it, and is replaced only when a strictly
		// better approach arrives. Entropy supplies the forward step; the mechanism only has to
		// stop the policy forgetting what it already stumbled into.

		// ===== THE GRAVITY FIELD =====
		// Goal mass is BINARY and decided by what the map believes, not by a quantile of
		// convenience: a state is a goal if the map does not KNOW it (twin disagreement high), or
		// if the map thinks it is GOOD but we have not proven it (value above median, visits
		// below median). A state the map knows and rates poorly is not a goal.
		//
		//   phi(s) = sum over goals of exp( -d(s->g) / (gravityRangeDecisions * localD) )
		//
		// Summing rather than taking the nearest is the point: a clump of unlearned states pulls
		// harder than an isolated one, and separate clumps combine. As a region is learned its
		// members stop qualifying, its mass decays, and the pull redistributes elsewhere - so the
		// field is exhausted only when there is nothing left unknown or unproven.
		//
		// Range is SCALE-FREE: sigma is a fraction of the state distribution's OWN median pairwise
		// distance, re-measured every iteration. Stating it in decisions does NOT transfer, and
		// silently mis-scales here too, because the metric's spread grows the whole time it trains
		// (Quasi Spread went 0.97 -> 39 in one run). The AirLine port proved it: the same "50
		// decisions" produced sigma far larger than that toy's entire distance distribution and the
		// field went flat at contrast 1.09; a fraction of the median took it to 3.1-3.5.
		// 0.67 is the ratio that measured contrast 1.76 on the 591.7B rollout (sigma 23.5 units
		// against a 35-unit median pair distance), which is where aerial enrichment peaked.
		float gravityFrac = 0.67f;
		float unknownQuantile = 0.90f;  // twin disagreement above this = the map does not know it
		int goalCandidates = 512;       // goals sampled from the pool per iteration (compute cap)

		// Visitation, for "good but unproven". A fixed random projection of the quasimetric latent
		// into visitBits, with decaying per-bucket counts. 2^14 = 16384 buckets against ~16.7k rows
		// per iteration per rank is ~1 sample/bucket/iteration - finer would be all zeros, coarser
		// would blur distinct manoeuvres together. decay 0.99 gives a ~100-iteration memory, the
		// same horizon as progressLagIters.
		int visitBits = 14;
		float visitDecay = 0.99f;

		int withdrawAtIteration = 0; // 0 = never withdraw; otherwise SIL is off from this iteration

		// OPPONENT CONDITIONING. The environment is RocketSim PLUS an opponent, and the opponent
		// is a changing version of ourselves, so the map is only valid relative to who is being
		// played. Conditioning makes the value map and the metric functions of (state, opponent):
		// regions mapped against an old version go stale against a new one instead of being
		// silently averaged over the whole version ring. Fed from PPOLearnerConfig::oppCtxDim
		// ([isSelf, isOld, isExt, oppAgeFrac]) - the SAME privileged context the composite value
		// critic already receives, and equally never visible to the policy.
		bool oppCond = true;
		int oppCtxDim = 0;         // set from cfg.ppo.oppCtxDim at boot; 0 = unconditioned

		PartialModelConfig value, quasi;
	};

	struct ReachabilityConfig {
		bool enabled = false;     // Train the heads (aux loss) + compute gate diagnostics
		bool gateEnabled = false; // Actually scale the gated rewards (requires enabled)

		// When the gate is OFF, the rho/gate read block (3 full-buffer model passes + CPU smoothing)
		// feeds nothing but the Reach/* diagnostic panels — so only run it every Nth iteration.
		// Ignored (every iteration) when gateEnabled=true, since rewards then depend on it.
		int diagEveryIters = 1;

		// Model
		int representationSize = 128;
		float tau = 0.02f;          // Contrastive temperature; Score = cosine(phi, psi) / tau
		float lr = 3e-4f;
		float auxLossWeight = 0.5f; // Scale of the InfoNCE losses added to the PPO backward
		int infoSubSample = 512;    // Rows per PPO minibatch used for InfoNCE (the NxN logits matrix)
		float varReg = 0.3f;
		float logsumexpPenaltyCoeff = 0.01f;
		PartialModelConfig phi, psi;

		// HER goal sampling
		int ballHerMinOffset = 1;
		int ballHerMaxOffset = 45;  // ~3s at tickSkip 8 (15Hz steps) - a REAL-TIME window; re-derive if tickSkip changes (was 90 at ts4)
		float ballHerShortBiasPower = 2;
		// Chance of targeting the most-goalward achieved state in the window instead of a
		// short-biased random one, so near-net states populate the goal space and the fixed
		// scoring query goal stays in-distribution
		float ballHerGoalwardBias = 0.5f;
		int carHerMinOffset = 1;
		int carHerMaxOffset = 10;   // ~0.67s at tickSkip 8 - REAL-TIME window feeding the steering rho-band contact
		                            // gate ("commit where the race is a coin-flip"); re-derive if tickSkip changes (was 20 at ts4)
		float carHerShortBiasPower = 2;
		// Third goal-space head: canonical CAR pos+vel ("where can my car be, moving
		// how") - the movement-capability frontier for the META steering system. The
		// head + HER machinery predate this (car-proposer era); this flag trains it
		// INDEPENDENT of the proposer. Offline (conservative frozen-phi test,
		// research/tools/carstate_head_validate.py): its calibration curve is monotone
		// with ~7x the ball head's margin at every candidate window - the sharpest
		// frontier detector of the three heads. The window below was chosen BY
		// calibration margin across {20,45,90}, not by hand - AT tickSkip 4; that
		// calibration is VOID at ts8. Deliberately HELD at 45 (2026-07-17): it is an
		// empirical choice, not a real-time design like the two windows above, so it
		// waits for a carstate_head_validate.py re-run on ts8 data instead of blind
		// halving. Actuation stays gated
		// live (meta head-validity + per-cluster causal gates). Resume-safe: a missing
		// reach_psi_carstate.lt initializes fresh (ModelSet::Load allowNotExist).
		bool carStateHead = false;
		int carStateHerMinOffset = 1;
		int carStateHerMaxOffset = 45; // ts4-calibrated (void at ts8); re-calibration pending, see comment above
		// Gradient coupling of the car-state InfoNCE into the shared state-action encoder
		// (phi -> trunk). 0 = fully detached: only psi_carstate trains, exactly the
		// offline-validated frozen-phi regime. 2026-07-14 incident: this head shipped
		// fully coupled (the equivalent of 1.0) while at chance level - its loss alone
		// (~2x every other aux term combined, Reach/Aux Loss 0.5 -> 1.0+) churned the
		// shared trunk and Rating slid ~125 across ALL modes in ~500 iterations with
		// every behavioral guard green (this path had none). Values in (0,1] gradient-
		// scale the coupling (value-preserving: loss magnitude unchanged, trunk/phi
		// gradient scaled). Raising it is a one-lever experiment that NOW HAS NO
		// automatic guard behind it (the rating latches were removed 2026-07-25) -
		// never ship it coupled while the head is fresh, and watch Rating by hand.
		float carStateCouple = 0.0f;
		// The ball head only trains on episodes where the ball exceeded this speed;
		// a dead never-touched episode would just reteach the stationary-ball manifold
		float minBallMoveSpeed = 300;

		// Goal normalization (canonical frame: +y is always the attacked net)
		float posScaleX = 4096, posScaleY = 6000, posScaleZ = 2044, velScale = 6000;
		float carLocalScale = 2300;    // Car-local ball goals are normalized uniformly by this
		float scoringGoalSpeed = 1500; // Canonical +y ball speed of the fixed scoring query goal

		// rho evaluation: rho(s -> goal) = mean over K UNIFORM VALID actions of Score.
		// Uniform (not policy-sampled) = capability ("can we", not "would we")
		int numActionSamples = 16;
		int64_t scoreChunkSize = 4096;

		// Gate: level x delta, floored, positive-part-applied:
		//  level = control^alpha * scoring^(1-alpha)
		//  D     = sigmoid(wc*deltaControl + ws*deltaScoring)   (windowed progress deltas)
		//  gRaw  = clamp(level * 2 * D, 0, 1)                   (delta=0 => exactly the level gate)
		//  gEff  = floor + (1 - floor) * gRaw
		//  mult  = 1 - beta * (1 - gEff)                        (beta=0 => bit-identical baseline)
		//  reward = total + (mult - 1) * gatedPos               (positive parts only; penalties never muted)
		float gateFloor = 0.2f;
		float gateAlpha = 0.5f;          // Control weight in the level mix
		float controlTemp = 10;          // T_c on (rhoCarUs - rhoCarOpp); rho are cosine/tau logits
		float scoringTemp = 10;          // T_s
		float scoringBias = 0;           // b_s
		float deltaControlWeight = 0.5f; // wc
		float deltaScoringWeight = 0.5f; // ws
		int deltaWindow = 8;             // W steps for the windowed deltas (~0.53s at tickSkip 8)
		int deltaSmooth = 4;             // Boxcar length for the smoothed rho reads
		int touchPredHorizon = 45;       // Steps ahead a next-touch label may be found (validity metric)

		// Anneal: beta = min(smoothstep(accEMA over [accLo,accHi]), smoothstep(agreeEMA over [aucLo,aucHi]))
		// accEMA = InfoNCE categorical accuracy (min of the two heads);
		// agreeEMA = does sign(deltaControl) predict which team touches next (measured on real touches)
		float accLo = 0.30f, accHi = 0.55f;
		float aucLo = 0.55f, aucHi = 0.65f;
		float emaDecay = 0.99f; // Per-iteration decay of both EMAs
		float betaOverride = -1; // >= 0 forces beta (smoke tests); < 0 uses the EMAs

		ReachabilityConfig() {
			// Optimizer stays ADAM for both heads: contrastive InfoNCE embeddings train
			// poorly under orthogonalized updates (phi/psi = Adam, not Muon)
			phi = {};
			phi.layerSizes = { 256, 256 };
			phi.activationType = ModelActivationType::LEAKY_RELU;
			psi = {};
			psi.layerSizes = { 256, 256 };
			psi.activationType = ModelActivationType::LEAKY_RELU;
		}
	};
	// ProposerConfig (deliberate-practice proposer + drill bank) REMOVED 2026-07-25.
	// Disabled since the 9uz761ua regression; see git log -- .../PPO/Proposer.cpp

	// Secondary GOAL-ONLY critic (multi-horizon value decomposition). The dense shaped reward needs a
	// moderate gamma or target variance swamps the critic; the true objective (goal/concede, the one
	// unfarmable signal) carries meaning at much longer horizons. This trains a SECOND critic on a
	// goal-only reward channel (+1 team scored / -1 conceded, computed straight from game outcomes —
	// independent of the reward stack) at its own, much higher gamma, and blends its advantages into
	// the policy gradient: A = A_dense + betaEff * A_goal, betaEff std-matched so beta is the FRACTION
	// of dense-advantage scale contributed (a bounded influence by construction).
	// Episodes terminate on goals, so V_goal is inherently bounded in [-1,1]: it converges toward
	// P(score) - P(concede) with mild time preference — no standardization or clipping needed.
	// The net is fully independent (raw obs in, own trunk): zero gradient interference with the
	// proven policy/critic/shared-head path. Validation: watch GoalCritic/Value-Outcome Corr — the
	// critic's prediction must correlate POSITIVELY with realized episode outcomes; a channel or
	// sign bug reads negative there within minutes on a goal-dense checkpoint.
	struct GoalCriticConfig {
		bool enabled = false;
		float gamma = 0.9997f;      // STEP-denominated "huge distance" horizon: ~77s at tickSkip 4 (30Hz) but ~154s at
		                            // tickSkip 8 (15Hz) — re-derive per tickSkip in your main (5.0 overrides to 0.9994 ≈ 77s at 15Hz)
		float beta = 0.25f;         // blended advantage fraction (std-matched); 0 = train critic, no blend
		float lr = 1.5e-4f;
		PartialModelConfig model;   // independent net, raw obs -> 1; set layerSizes in your main
	};

	// How advantage filtering picks the rows the policy trains on. See advFilterFrac below.
	enum class AdvFilterMode {
		// Top `advFilterFrac` by SIGNED advantage — the V-MPO shape, whose improvement
		// guarantee comes from a positive-only nonparametric reweighting. At frac 0.5 the
		// threshold is the buffer median, so the retained set is essentially "every positive
		// row" and the whole negative tail is discarded from the policy gradient.
		TOP_SIGNED,
		// Top `advFilterFrac` by |advantage|, i.e. drop the MIDDLE of the distribution and
		// keep both tails. Rationale: the PPO per-row gradient is |A| * grad(log pi), so
		// information content scales with MAGNITUDE and the sign only picks a direction.
		// A median cut spends the whole budget on the least informative half of the positives
		// (rows near A ~ +0 the critic already predicted) while discarding the largest-|A|
		// material in the buffer. Magnitude selection keeps the surprises in both directions
		// for the same kept count, and drops `AdvFilter/Kept Positive Frac` off 1.0, which is
		// the buffer-wide clipping-ratchet read (CLAUDE.md / STEERED_PRACTICE.md).
		// Quantile is taken on |A| rather than as explicit symmetric tails (top frac/2 +
		// bottom frac/2) because advantages here are heavily right-skewed — rare goals,
		// TouchAccel spikes — so whichever tail is genuinely fatter should claim more of the
		// budget instead of being forced into a balance the data does not have.
		MAGNITUDE,
	};

	// https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/ppo/ppo_learner.py
	struct PPOLearnerConfig {

		int64_t tsPerItr = 50'000;
		int64_t batchSize = 50'000;
		int64_t miniBatchSize = 0; // Set to 0 to just use batchSize

		// On the last batch of the iteration,
		//	if the amount of remaining experience exceeds the batch size,
		//	all remaining experience is used as a larger batch.
		// This prevents experience loss due to batch size rounding.
		// This will only happen if the amount of remaining experience is < batchSize*2.
		bool overbatching = true;

		double maxEpisodeDuration = 120; // In seconds

		// Actions with the highest probability are always chosen, instead of being more likely
		// This will make your bot play better (usually), but is horrible for learning
		// Trying to run a PPO learn iteration with deterministic mode will throw an exception
		bool deterministic = false;

		// Use half-precision models for inference
		// This is much faster on GPU, not so much for CPU
		// Dtype is GGLHalfPrecType(): BF16 on sm_80+ (5080), FP16 on V100 (sm_70).
		bool useHalfPrecision = false;

		/**
		 * Replay exact-shape CUDA graphs for the frozen self-play collection policy.
		 * Sampling, training, opponents, evaluation, MPI, and NCCL remain eager. The
		 * flag is opt-in until numerical and throughput A/Bs pass on the target GPUs.
		 **/
		bool useCudaGraphs = false;

		// AUTOCAST FOR THE LEARN PASS (2026-08-04, user-directed "more sps"; FP16 on V100 2026-08-14).
		// useHalfPrecision above covers only INFERENCE; the learn pass ran strict fp32, and
		// profiling made it the system bottleneck: the GPU sits at 92-98% and the collect
		// worker's small forwards queue behind it, so learn-pass cost sets BOTH halves of the
		// iteration (collection 5.3s / consumption 3.9s under pipelining). Weights and the
		// optimizer stay fp32 — autocast only runs the matmuls in GGLHalfPrecType() and keeps
		// reductions and loss functions in fp32. BF16 (sm_80+) needs no GradScaler; FP16 (V100)
		// scales the loss before backward and unscales grads before clip.
		//
		// SCOPE IS DELIBERATE, not blanket. Autocast covers the policy / critic / goal-critic /
		// V-dagger forwards — the 1536-wide value family that dominates the pass — and is
		// PAUSED for two paths that are numerically delicate and cheap anyway:
		//   * the geo HJB residual, which takes an input-gradient with create_graph and then
		//     backprops through it (double-backward under autocast is fragile, and the field
		//     is already conditioning-sensitive — see geoGamma).
		//   * the reachability InfoNCE, which this file's own comment keeps in fp32 because
		//     contrastive embeddings train poorly at reduced precision.
		// Backward always runs outside the autocast region, per the standard recipe.
		//
		// REVERT = this flag. Watch for non-finite losses, a Policy Entropy discontinuity, or
		// Headroom/Vdag Update Magnitude changing scale.
		//
		// Flag name is historical (BF16). On V100 this now means FP16 autocast + loss scale.
		// Revert: GGL_LEARN_AMP=0. Watch non-finite losses, Policy Entropy, Vdag Update Magnitude.
		bool learnAutocastBF16 = false;

		// ADVANTAGE FILTERING AS A TRUE ROW SUBSET (2026-08-04, user-directed).
		// advFilterFrac/advFilterMode above select rows, but they only MASK the policy loss
		// (`ppoPerRow * advKeep`) — every row still runs the full forward and backward, so the
		// filter costs full compute and the policy head is only ~10% of the pass anyway. With
		// this ON, the selected rows become an actual subset of the learn pass: the whole
		// update (policy AND all value heads) runs on them, so the learn pass shrinks with
		// advFilterFrac.
		//
		// THE PPO SURROGATE TERM IS UNCHANGED BY THIS FLAG, and that is verified algebraically:
		// excluded rows already contribute exactly zero to the clipped policy loss and the kept
		// rows are already renormalized by their own count, so mask-at-f and subset-at-f give
		// an identical policy gradient. The VALUE heads (critic, V-dagger twins, goal critic)
		// and the aux heads seeing the subset is the intended change — the thing the literature
		// is split on, which is why this is a flag and why 0.5 is the fraction to try first.
		//
		// *** CORRECTION (2026-08-04, found in audit the same night this shipped). "The policy
		// term is unchanged" is TRUE of the surrogate and FALSE of the objective, because the
		// ENTROPY BONUS is part of the policy objective and it does change population: with a
		// subset, `entRows` is computed over the kept (high-|A|) rows only, so entropy is
		// regularized over the rows that scored well rather than over the whole visited state
		// distribution — which is exactly what the loss comment in PPOLearner::Learn claims it
		// does NOT do. The direction is unfavourable: narrowing entropy pressure onto the
		// extreme-advantage tail is the mechanism arXiv:2505.22617 identifies as driving
		// entropy collapse, and 6.1 died of entropy collapse.
		// Measured after 40 iterations: median entropy 0.4989 vs 0.4929-0.4996 before, stdev
		// 0.0252 vs 0.0255-0.0277 — no detectable damage YET. (An audit claim of a 10x variance
		// jump did not survive checking; it compared mismatched windows.)
		// ALSO: `Policy Entropy`, `Mean KL Divergence` and `SB3 Clip Fraction` are now SUBSET
		// statistics and are not comparable to pre-2026-08-04 history. The guard thresholds
		// were set against the old population.
		// The clean fix is to keep the entropy term on the full row set (costs back the trunk +
		// policy forward on dropped rows, ~13% of the pass, leaving ~37% of the 50% saving).
		// Not taken unilaterally on a live run — it is a real throughput/safety trade. ***
		//
		// PRECEDENT (all on-policy self-play at scale, all ranking by |A| with both tails kept,
		// all filtering the critic too): GigaFlow arXiv:2502.03349 (ICML'25) drops ~80% on an
		// adaptive threshold for 2.3x throughput "without sacrificing sample efficiency", and
		// predicts the win lands "especially where data collection is cheaper than gradient
		// calculation" — this run exactly; Stratego arXiv:2511.07312 keeps top 25% for ~2.5x
		// wall-clock per iteration with sample efficiency going UP; Generals.io arXiv:2606.23348
		// keeps top 25% (`top_k(|adv|)` in its code) and swept 25/50/75/100 with 25% winning.
		// PufferLib 3.0 ships the soft version (segment-level priority by summed |A|).
		// AGAINST, and the reason this is not simply turned on at 25%: VSOP arXiv:2306.01460
		// filters the ACTOR only and has a lower-bound proof for doing so, and PPG
		// arXiv:2009.04416 finds the value function tolerates MORE sample reuse than the policy,
		// not less. No published work runs the controlled comparison. Also note every paper
		// above uses SEPARATE actor/critic networks; this run shares one trunk across policy,
		// critic_trunk, both V-dagger twins, goal critic, reach and geo, so a row subset shrinks
		// gradient coverage into shared parameters in a way none of them measured.
		//
		// ORDERING IS SAFE HERE: the mask is built in Learner.cpp AFTER GAE, after the
		// injections and after advantage normalization, which is what the papers that got this
		// wrong ("damaged credit assignment") violated.
		// WATCH, in order: Policy Entropy and Mean KL (reward is consistently the LAST metric to
		// break under over-filtering), then Geo/Residual — the HJB residual is now evaluated on
		// the kept rows rather than all of them. REVERT = this flag.
		bool advFilterSubset = false;

		PartialModelConfig policy, critic, sharedHead;

		// CRITIC TRUNK (2026-07-29, user-directed): a SECOND shared body stacked on top of
		// `sharedHead`, feeding every value head - critic, goal critic, and the V-dagger twins.
		// Empty layerSizes = disabled, in which case every head reads the main trunk directly and
		// the goal critic keeps its historical independent raw-obs form (the pre-2026-07-29 shape).
		//
		// WHY: the value heads were four independent 5x1280 stacks (~31.0M params, 82% of the net)
		// because vdag1/vdag2 mirror the critic's config and the goal critic had its own. Factoring
		// the first three layers out of all four is what makes them affordable: the shared features
		// are computed and STORED once per row instead of four times, and the learn-pass activation
		// peak - not the parameter count - is what sets the miniBatchSize ceiling.
		// KEEP THE HEADS AT >= 2 LAYERS. vdag1/vdag2 exist to be DIFFERENT functions: the target
		// takes min(V1,V2) as the anti-ratchet against asymmetric TD self-amplifying through its
		// own bootstrap (COMPOSITION_CRITIC.md section 4.4 measured H inflating 0.3 -> 11.8 with no
		// conversion behind it). Collapse them onto a shared body with one private layer each and
		// they become near-identical, which silently disarms that guard.
		// COST: the goal critic's gradient now reaches the main trunk (and therefore the policy).
		// That was previously impossible BY CONSTRUCTION - it was an independent net on raw obs
		// specifically so its sparse 77s-horizon +-1 channel could not reshape the proven
		// perception path. Accepted deliberately for the param/activation win; if the policy
		// regresses on a fresh lineage, this is the first thing to suspect. Reverting it alone =
		// give the goal critic back its own layerSizes on raw obs (see PPOLearner's ctor).
		PartialModelConfig criticTrunk;


		int epochs = 2;
		float policyLR = 3e-4f; // Policy learning rate
		float criticLR = 3e-4f; // Critic learning rate

		float entropyScale = 0.018f; // The scale of the normalized entropy loss
		// Whether to ignore invalid actions in the entropy calculation.
		// True means that entropy will be determined only from available actions.
		// False means that entropy for unavailable actions will be zero,
		//	meaning the entropy of the state is limited to the fraction of available actions in that state.
		bool maskEntropy = false;

		float clipRange = 0.2f;

		// ADVANTAGE FILTERING (V-MPO-style top-fraction policy update, 2026-07-29 user-directed).
		// The POLICY trains only on the rows whose post-injection advantage is in the top
		// `advFilterFrac` of the iteration; the threshold is a buffer-wide quantile computed AFTER
		// GAE and after every injector (Learner.cpp), so it selects on the same number the policy
		// loss actually consumes. 1.0 = disabled (train on all rows).
		// The CRITIC, goal critic, V-dagger twins and reachability heads keep ALL rows by design:
		// value heads must learn the honest return structure of everything that happened (the
		// 2026-07-12 phantom -V(s_end) freefall is what excluding rows from a value head costs).
		// Gradient scale is preserved — the retained rows are renormalized by the kept count, so
		// this is a selection, NOT a silent policy-LR cut.
		// KNOWN RISK, watch it: under TOP_SIGNED at 0.5 nearly every retained row has positive
		// advantage, which is the CLIPPING RATCHET asymmetry (CLAUDE.md / STEERED_PRACTICE.md)
		// applied buffer-wide — successes reinforce and punished failures are discarded, which
		// once compounded lucky overcommits into an Elo bleed while the viewer looked better.
		// MAGNITUDE mode exists to defuse exactly that; see the enum above. Watch Policy Entropy
		// (a collapse means overcommitment), SB3 Clip Fraction, and the Nexto/* goal slope (the
		// only inflation-proof yardstick). Revert = set this back to 1.
		float advFilterFrac = 1.0f;
		AdvFilterMode advFilterMode = AdvFilterMode::TOP_SIGNED;

		// Temperature of the policy's softmax distribution
		float policyTemperature = 1;

		float gaeLambda = 0.95f;
		float gaeGamma = 0.99f;
		float rewardClipRange = 10; // Clip range for normalized rewards, set 0 to disable

		bool useGuidingPolicy = false;
		std::filesystem::path guidingPolicyPath = "guiding_policy/"; // Path of the guiding policy model(s)
		float guidingStrength = 0.03f;
		// Extra policy-head inputs the GUIDING checkpoint was trained with, if different from this
		// run's. Distillation needs matching ACTIONS and OBS, not matching weights -- which is the
		// whole reason a fresh net can inherit from a checkpoint it is not weight-compatible with.
		// But the loader must still build the guiding policy at ITS width, not this run's: the
		// 422B and multi-mode GCO policies are 768x1280 (no intent features) and would fail to
		// load into a 1287-wide slot. -1 means "same as this run".
		int guidingPolicyIntentExtra = -1;

		// SECOND teacher, for team-mode rows. Pulsar 2.5 distils from two checkpoints at once:
		// the best 1v1 policy on 1v1 arenas and the best multi-mode policy on 2v2/3v3 arenas.
		// A row's mode is read from the observation's teammate PRESENCE FLAGS, so no per-row
		// plumbing is needed and the split cannot desync from the arena layout.
		bool useGuidingPolicyTeam = false;
		// Slots per team in the padded observation. Needed to locate the presence flags, which
		// are the last (2*n - 1) entries. MAX_PLAYERS_PER_TEAM is a file-local constant in
		// ExampleMain and is not exported, so it is passed rather than guessed from obsSize.
		int obsMaxPlayersPerTeam = 3;
		std::filesystem::path guidingPolicyTeamPath = "guiding_policy_team/";
		int guidingPolicyTeamIntentExtra = -1;

		FrontierConfig frontier;
		ReachabilityConfig reachability;
		GoalCriticConfig goalCritic;

		// HEADROOM — the composition critic (2026-07-24; validated offline in
		// rltest/OVERNIGHT_LOG.md: iso-compute ~2x air-touch vs vanilla PPO, SPS 0.99).
		// Twin V-dagger heads on the SHARED TRUNK (gradients flow — fresh-run
		// co-adaptation, the reachability-aux precedent; do NOT enable mid-run on a
		// mature trunk: that is the measured carstate-incident shape). Expectile-TD
		// (vdagTau) on one-iteration-frozen TD targets over executed transitions,
		// min-in-target twins (anti-ratchet). H = relu(min(V1,V2) - V_real) =
		// realizable headroom; actuation = SEEK potential Phi=+H (PBRS; the closure
		// sign measurably teaches avoidance), own std-matched beta, boundary-masked.
		// NOT guarded: the rating latch was removed 2026-07-25, so the only automatic
		// check left is the boot sanity probe, which sees crashes and not update
		// damage - watch Headroom/* by hand. ON BY DEFAULT (user directive 2026-07-24, fresh-run
		// deploys). Revert = set false here and rebuild.
		bool vdagEnabled = true;
		float vdagTau = 0.75f;
		// INTENT CLASS (research/reports/NATIVE_INTENT_PROTOCOL.md, 2026-09-11). Exogenous
		// uniform intents: every intentPeriod decisions each player draws z ~ U{0..intentDim-1};
		// the policy head reads trunk ++ onehot(z) ++ remaining/intentPeriod, and a trainable
		// [intentDim x numActions] table of action-logit biases (init N(0, intentBiasStd)) gives
		// coherent, on-policy exploration (the toy's supply lever: bias std 3 = 4x std 1). The
		// intent is NOT learned by a manager (toy: uniform intents scored as well as the learned
		// manager at execution; its entropy stayed near-maximal in training), so the likelihood
		// is plain pi(a|s,z) and the PPO ratio is exact. intentDim 0 = class off (no shape change).
		int intentDim = 0;
		int intentPeriod = 8;
		float intentBiasStd = 3.f;
		// Discriminability credit l = log q(z|s_b, y_b) - log(1/intentDim), q a small MLP on the
		// boundary obs and the interval's last obs with the prevAction block zeroed, fit each
		// iteration on that iteration's boundary rows AFTER scoring (fresh data). Broadcast to
		// the interval's rows, centred, sigma-matched at intentDiscBeta, clamped +-3 sigma,
		// added to advantages through the seek-term pathway. 0 = off.
		float intentDiscBeta = 1.f;
		float intentDiscLR = 1e-3f;   // q's Adam LR (toy: 1e-3); ADAM, not Muon (a classifier, not a trunk)
		// Dose curve measured (rltest, n=2/point, 25M): inverted-U, optimum 0.30-0.45.
		// 0.15 -> 0.30 gave touch +45% / air +93%. beta >= 1.0 is WORSE THAN BASE (the
		// +-3sigma clamp binds, clipped potential diffs stop telescoping, PBRS
		// invariance is destroyed and the term becomes reward distortion).
		float vdagSeekBeta = 0.30f;

		// THEORY (r-hat): twin optimistic REWARD models, expectile tau on arrival
		// rewards + group-L1 over input features (Occam). Supplies "what kind of state
		// pays" so V-dagger can be seeded with hypotheses at states whose payoff has
		// never been collected. PLANT: event-masked seeding rows (only where the theory
		// predicts a significant event; interior seeds measured as relay-killing
		// ballast). Set false to ablate back to composition-only.
		// DISABLED 2026-07-28 (user-directed), code intentionally LEFT IN PLACE. rhat1/rhat2
		// mirror the CRITIC's config (PPOLearner ctor), so enabling them took the 5x1280
		// critic-family head count 3 -> 5. That count times the per-row saved activations is
		// what sets the learn-pass peak: measured 11.91GB of PyTorch tensors and a hard OOM on
		// a 20000x1280 head forward, on a card with ~13.2GB free after desktop graphics. Flip
		// back on only together with a miniBatchSize cut (or narrower rhat heads - a reward
		// model with group-L1 Occam has no obvious need for critic width).
		// (The 5.3 lineage runs with this OFF; the geometry rung supplies its own r-hat.)
		bool vdagTheoryEnabled = false;
		float vdagTheoryTau = 0.5f;    // ACCURATE, not optimistic. Measured degeneracy: an
		                               // optimistic expectile and the Occam penalty share a
		                               // minimiser - a high CONSTANT satisfies both - so the
		                               // theory prunes its own features and stops
		                               // discriminating. Optimism is the field's job.
		float vdagTheoryL1 = 0.02f;    // Occam weight on r-hat input-feature columns

		// (The ARCHIVE channel — persistent field-ascent transitions replayed as weighted BC —
		// was removed entirely on the parallel line, commit 10979db; its config went with it.)

		// ===================== GEOMETRY — the 4th rung (2026-07-30) =====================
		// Validated offline in ~/Projects/experiments/possibility (see
		// research/reports/GEOMETRIC_CRITIC.md). The ladder reads:
		//   V_real    what we normally get     expectation over executed transitions
		//   V_exp     what we get when it goes well   upper expectile of the same returns
		//   V_dag     what we can get          upper envelope over EXECUTED transitions
		//   V_geo     what may be POSSIBLE     value implied by the environment's GEOMETRY
		//
		// V_geo is the fixed point of a discrete Hamilton-Jacobi-Bellman equation. Bellman over
		// the feasible one-step displacement set, max linearised (one step at 15Hz is small),
		// displacement set modelled as an ellipsoid with per-coordinate scale Sigma(s):
		//
		//     (1 - gamma) V(s)  =  r_hat(s)  +  gamma * || grad_s V(s) ||_Sigma(s)
		//
		// One MLP trained to zero that residual. NOT a world model: Sigma is a conditional
		// VARIANCE statistic, it never predicts where you go, only how far you could. No
		// rollouts, no search, no action-set expansion — one extra input-gradient per minibatch.
		//
		// Why it is not bounded by max-proven like every rung below it: the residual is a LOCAL
		// CONSISTENCY CONDITION, evaluable wherever r_hat and Sigma are defined, including states
		// nothing has ever visited. Value flows through the PDE, not through the buffer.
		//
		// Measured (200 oracle-scored probes, 2 seeds): rho(V*) 0.698 against V_real 0.665 and
		// V_dag 0.577, and it is the ONLY rung whose value tracks what the ENVIRONMENT allows
		// (rho 0.698) more than what the POLICY does (rho 0.679). Controls: sigma_scale=0
		// collapses it to 0.360 (the geometry term is load-bearing), r_hat alone scores 0.228,
		// mobility alone -0.114, and rho(V_geo, mobility) is NEGATIVE — not a mobility proxy.
		//
		// DEPLOY SHAPE: largest measured wins are the EARLIEST buckets (4.7x air over V_dag at
		// 0-3M), because V_geo estimates V* — most informative when the policy is worst. But
		// the rung is PERMANENT, not a fading bootstrap: see geoMixW below. Fresh runs only in
		// the sense that no mid-run insertion has been tested.
		bool geoEnabled = false;      // OFF by default: fresh-run mechanism, see above
		float geoLR = 1e-3f;
		// HJB gamma, DECOUPLED from gaeGamma (2026-08-03, 6.1). The residual
		//     resid = (1-g)*V - r_hat - g*|Sigma grad V|
		// is tickSkip-FRAGILE in its CONDITIONING, not its constants: (1-g) is the only
		// coefficient anchoring V's LEVEL, while the gradient-shaping gnorm path carries ~g, so
		// the level-anchoring weight is (1-g)/g — 0.0031 at the validated ts8 gamma, 0.000388 at
		// ts1's gaeGamma. Measured on 6.0: Geo/V Mean 2.9-3.0 against the ~39 dimensional
		// analysis predicts (r_hat and Sigma both scale ~1/8 at ts1, so V* is rate-invariant),
		// creeping ~0.1/20 iters — an unsolved field injected at geoMixW while every Geo/* panel
		// looked healthy, because the injector standardizes twice and discards the level.
		// Rescaling the residual (dividing by 1-g) is NOT a fix: it is a constant loss rescale,
		// and the geo nets train under Adam, which is scale-invariant — the RELATIVE term
		// weighting is what must be restored, i.e. gamma itself. Set this to the gamma the rung
		// was validated at (0.9969, ts8) regardless of tickSkip: the loss geometry is then
		// exactly the validated one, and V's level lands at ~1/8 of rate-invariant (harmless —
		// only the field's SHAPE reaches the policy). This gamma defines the FIELD only; the
		// PBRS injection of H_geo telescopes on gaeGamma downstream as it must. <0 => gaeGamma.
		float geoGamma = -1.f;
		// Shrinks the feasible-displacement ellipsoid. rho(V*) is flat at 0.697-0.698 across
		// {0.5, 1, 2}, but 0.5 is the only setting that also keeps rho(V*) > rho(V_pi).
		float geoSigmaScale = 0.5f;
		// The advection term grad V . mu is DELIBERATELY ABSENT. mu = E[ds|s] is estimated
		// under the POLICY, so including it re-imports the habit the rung exists to see past:
		// measured, drift-on loses the beats-habit property at every sigma (0.634 vs 0.698).
		//
		// RESERVOIR. r(s) and the displacement spread are properties of the ENVIRONMENT and are
		// STATIONARY; only where we sample them moves. Fitting them on the sliding on-policy
		// buffer makes them forget regions the policy has left — measured as the HJB residual
		// growing 100-250x over training, flat to ~6M then runaway, exactly tracking where every
		// actuation variant stopped helping. A uniform reservoir over all data holds it flat
		// (~13x lower at the runaway point) and is the correct fit for a stationary target.
		int geoReservoir = 200000;
		// Dose. THE most expensive error in the offline study: nominal beta equal for two
		// potentials still gave V_geo ~4x V_dag's ACTUAL injection (inj_std/adv_std 1.22 vs
		// 0.31), and eight comparisons were run over-dosed before it was measured. Watch
		// Geo/Inj Std Ratio and set this by the RATIO, not by the nominal number.
		float geoSeekBeta = 0.04f;
		// Mix: Phi = (1-w) norm(H_geo) + w norm(Phi_dag). Each potential is used in the FORM
		// its own validation supports — H_geo = relu(V_geo - V_real) (the gap; the raw level
		// measured neutral), Phi_dag = V_dag itself (the level; the gap measured to penalise
		// finishing, see the seek block in Learner.cpp). Both are functions of state alone, so
		// PBRS validity is exact throughout, and each is normalised to unit scale so w is a
		// genuine mix rather than a hidden dose ramp.
		//
		// w is CONSTANT: V_geo is a PERMANENT 4th rung, on the same footing as the other three.
		// In the paired test (same device, same seeds, only the mixing rule differing) w=0.5
		// beat a 2M->6M crossfade that retired V_geo in 6 of 8 step-bucket cells (touch 3/4,
		// air 3/4) and collapsed nowhere; the crossfade won only 0-3M air, mechanically,
		// because early it is PURE geo while the constant mix dilutes geo with a still-empty
		// V_dag rung. The one earlier failure of the permanent shape (testbed U_both) ran 4x
		// over-dosed at beta 0.15 before the dose confound was found, and is retired as
		// evidence about the shape. No schedule: a rung that expires was the assumption the
		// measurement refuted, so there is nothing here to fall back to.
		float geoMixW = 0.5f;
		PartialModelConfig geoModel;   // shared shape for the sigma / r-hat / value nets

		bool vdagEntGateEnabled = true;
		float vdagEntGateK = 3.0f;
		float vdagEntGateCap = 3.0f;   // 5.0 in the toy; kept tighter for a live league

		// ===== COMPOSITE VALUE CRITIC (research/testbeds/inject2d RESULTS.md batches 10-11) =====
		// V is the denominator of the whole optimism stack (H = Vdag - V, SIL weights,
		// GAE, LP), so its noise pollutes everything downstream. Four toy-validated
		// ingredients, individually flag-gated:
		//
		//  valueTwinEnabled  : second critic head ("critic2"); each head trains on a
		//    DISJOINT half of every minibatch (uncorrelated sample noise), readout is the
		//    mean -- which IS the noise-cancelling operation: V1 - (V1-V2)/2 = mean. The
		//    signal is common-mode, the independent noise differential. |V1-V2| is a free
		//    per-state noise gauge (Value/Twin Disagree). Trunks see BOTH halves' grads.
		//  valueMirrorEnabled: train the value heads on x-mirrored copies (exact game
		//    symmetry; slot permutation is already trained in by the obs builder's
		//    shuffleSlots). Toy: hallucinated-headroom floor 9x down at intact EV.
		//    A wrong mirror is silent corruption -- ObsMirror::Build hard-fails on any
		//    structural inconsistency, and mirrorMaxPlayersPerTeam must match the obs.
		//  auxDispEnabled    : one-step displacement (mu, log-sigma) NLL head on the
		//    SHARED trunk -- representation pressure, the largest single training lever
		//    measured in the toy (-40% ignition). Deliberately NOT inside the value head:
		//    wired there it biases V (measured, ev 0.68 -> 0.49). Small weight, own panel.
		//  epiBlendEnabled   : episodic baseline -- kNN over the reservoir's stored
		//    returns, blended into the GAE value predictions by the critic's own
		//    inadequacy (w = epiWMax * (1 - EV_ema)); memory carries the baseline while
		//    V is young and hands authority back as V matures. NOTE the stationarity
		//    rule: bank RETURNS rot as the policy improves -- this blend self-retires,
		//    which is why it is safe where the peer-referee variant (toy vh_grp) failed.
		//  oppCondEnabled    : privileged opponent conditioning -- a zero-init additive
		//    embedding of {isSelf, isOldVersion, isExternal, versionAge} into the value
		//    body. Valid by the asymmetric actor-critic argument (a baseline may use
		//    privileged inputs; it reduces variance, cannot bias the gradient). The
		//    POLICY never sees it. Zero-init = exact no-op at load -> checkpoint-safe.
		bool valueTwinEnabled = false;
		bool valueMirrorEnabled = false;
		int mirrorMaxPlayersPerTeam = 3;   // MUST match the obs builder's padding
		float valueMirrorFrac = 0.25f;     // fraction of each minibatch given a mirrored pass
		bool auxDispEnabled = false;
		float auxDispWeight = 0.05f;
		PartialModelConfig auxDispModel;   // default {256} head on the shared trunk
		bool epiBlendEnabled = false;
		int epiK = 8;                      // kNN neighbors
		int epiSub = 2048;                 // bank subsample per iteration
		float epiWMax = 0.5f;
		bool oppCondEnabled = false;
		int oppCtxDim = 4;

		// ===== HULL OPERATOR (record-licensed relaxed Bellman) =====
		// Canonical: research/reports/EPSILON_CRITIC.md section 7. The V-dagger bootstrap is
		// maxed over the real next state plus hullK candidates built by transplanting
		// eps-scaled WITNESSED one-step displacement vectors between states matched in a
		// LEARNED dynamics chart (L1-sparse projection trained by displacement NLL -- it
		// discovers the physics' invariances from data; on the toy it kept vz/contact/boost
		// and pruned position). Generalization bound moves from state support to CHART
		// support: a dynamics cell witnessed anywhere licenses slack everywhere it recurs,
		// which is what prices never-assembled conducts (toy family F: 0.000% assembly in
		// every record, priced above V-dagger on 8/8 seeds). Because donors are real
		// displacement vectors, asymmetries are preserved: no +boost-in-air can ever be
		// hallucinated. This is the OPEN (actuation-seat) configuration -- the validated
		// trainer (toy: fastest ignition of the program, 1.02M vs 1.22M) -- NOT the gated
		// estimator-grade readout; hallucinated headroom in thin regions directs attempts
		// and is structurally cheap because SIL consolidates only realized R > V_exp
		// conversions. Adversarial ledger (5 attacks, all harm channels closed) in the
		// report. eps=0 (or the flag off) recovers the plain composition critic exactly.
		bool hullEnabled = false;      // fresh-restart lever; old checkpoints fresh-init the chart
		float hullEps = 1.0f;          // slack scale on donated displacements (toy-validated)
		int hullK = 4;                 // donor candidates per bootstrap row
		int hullChartDim = 8;          // learned dynamics-chart width
		float hullChartL1 = 1e-3f;     // sparsity pressure on the chart projection
		float hullChartLR = 1e-3f;     // Adam; chart nets are OFF-trunk world-facing fits
		int hullBankSub = 2048;        // donor-bank subsample per iteration (cost lever)
		PartialModelConfig hullHeadModel;  // shape for the chart's (mu, log-sigma) head

		// ===== SELF-IMITATION (headroom-gated) =====
		// The actuation channel that replaced the potential injection (2026-08-05,
		// research/testbeds/inject2d). When a trajectory's realized return beat V_exp
		// (a genuine conversion, not routine luck) in a state the ladder flags as
		// frontier (headroom-mix >= silGateQ quantile), an extra positive-only BC term
		// -log pi(a|s) * (R - V)+ clones that behavior before the expectation gradient
		// washes it out. Success-only by construction: failures get default PPO
		// treatment, so the below-break-even avoidance loop (the acquisition wall's
		// mechanism) is attacked directly, and only executed behavior is ever imitated
		// (the execution principle is satisfied trivially). Measured (8 seeds, toy):
		// 8/8 ignition vs 5/8, ~5x conduct rate at matched budget vs the potential
		// channel, and the H gate specifically buys ignition SPEED (0.99M vs 1.65M
		// ungated). Known un-derisked hazard: imitating lucky overcommits an opponent
		// punishes (the steering-v1 ratchet) -- silCoeff is halved from the toy's 0.1
		// for that reason; watch RatingWatch/Ref shares for the signature.
		bool silEnabled = false;
		float silCoeff = 0.05f;
		float silGateQ = 0.70f;        // headroom-mix quantile above which rows may imitate
		float silWCapSigma = 2.0f;     // weight cap, in units of std(R - V)

		// ===== IMPLICIT WORLD MODEL (theorised-achievable-value field) =====
		// Twin one-step OBS-space dynamics + optimistic value iteration read out through
		// the worth theory. No rollout, no search, no buffer: one hop per update, and
		// multi-step composition happens across updates (amortised background planning).
		// Enters ONLY as an additive potential alongside Phi=Vdag - the sum of two
		// potentials is a potential, so PBRS invariance is preserved and the honest
		// critic is never corrupted.
		bool vdagWmEnabled = true;
		int vdagWmRows = 4096;        // states per iteration used for dynamics + VI
		int vdagWmActions = 6;        // candidate actions imagined per state
		float vdagWmGammaIm = 0.92f;  // OWN, shorter horizon: at the task gamma a distant
		                              // opportunity is nearly as good as a near one, so the
		                              // field goes flat and the seek gradient vanishes.
		float vdagWmKappa = 0.02f;    // per-hop cost of imagining (grounding by attrition)
		float vdagWmDisTh = 0.02f;    // twin-disagreement trust threshold
		float vdagWmLambda = 0.5f;    // weight of the imagined potential vs Vdag
		float vdagWmOccam = 0.02f;    // group-L1 on the model's INPUTS. This is what makes
		                              // off-support prediction lawful: physics that does not
		                              // depend on a feature loses that input, so the model
		                              // computes the same way in never-visited regions and
		                              // the twins agree there (measured: disagreement 300x
		                              // under threshold seven units up in unvisited air).
		std::vector<int> vdagWmLayers = { 512, 512 };

		PPOLearnerConfig() {
			policy = {};
			policy.layerSizes = { 256, 256, 256 };
			critic = {};
			critic.layerSizes = { 256, 256, 256 };
			sharedHead = {};
			sharedHead.layerSizes = { 256 };
			sharedHead.addOutputLayer = false;
			// layerSizes stays EMPTY: the critic trunk is opt-in (IsValid() == false disables it),
			// so existing configs keep the four-independent-heads shape they were tuned on.
			criticTrunk = {};
			criticTrunk.addOutputLayer = false;
		}
	};
}
