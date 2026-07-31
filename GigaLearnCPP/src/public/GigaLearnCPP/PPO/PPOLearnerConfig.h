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
		bool useHalfPrecision = false;

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
