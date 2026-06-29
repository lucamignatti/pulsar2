#pragma once
#include <RLGymCPP/BasicTypes/Lists.h>

#include "../Util/ModelConfig.h"

namespace GGL {

	struct ContrastiveGoalConfig {
		bool enabled = false;

		float lambdaStart = 0.f;
		float lambda = 0.65f;
		uint64_t lambdaAnnealSteps = 200'000'000;
		float sigmaFloor = 1e-4f;
		float sigmaMin = 1e-6f;

		// Self-gate for the contrastive advantage blend: only blend GCRL into the
		// policy gradient when meanAbsEdge >= gcrlGateRatio * baselineSpread -- i.e.
		// the taken-action edge over random actions exceeds the baseline samples'
		// own within-state spread (real action-discrimination, not sampling noise).
		// ~1.0 means "edge indistinguishable from random-action noise" (no signal);
		// >1 requires the critic to actually rank the taken action above random.
		// Watch the logged "GCRL Separation" metric to tune.
		float gcrlGateRatio = 1.3f;

		int herMinOffset = 1;
		int herMaxOffset = 90;
		float herShortBiasPower = 2.f;
		int baselineActionSamples = 16; // TRIAD-NATIVE: masked-random K (4->16, var of mean(4) is 4x noisy)

		// ── Car critic ──────────────────────────────────────────────────────────
		// A second, isolated contrastive critic. BALL-AGNOSTIC: its goal is the car's OWN future kinematic
		// state (velocity + forward + up + angular velocity, carGoalInputSize floats; see AppendCarStateGoal)
		// a SHORT horizon ahead -- i.e. car CONTROL (speedflips, wavedashes, aerial control, recoveries), the
		// action-attributable motor-skill signal. Its own short HER window, no goalward bias, no scoring mix.
		// Goals come from the per-step carStates buffer (no obs-builder offset needed).
		bool useCarCritic = false;
		// World-frame "GOALSHORT" ball-goal critic (the default contrastiveGoalLearner). It is action-INERT
		// far-field (edge ~0.04, successor-feature asymptote) and its batch-normed edge reinflates to
		// unit-std noise (~30% of the GCRL nudge). Set false to DROP it from training + coupling (CAR-only
		// governs the advantage); it stays constructed+checkpoint-loadable, just untrained/unscored.
		bool useGoalCritic = true;
		// Goal POTENTIAL shaping (positioning). Re-uses the goal critic but consumes it as a state-POTENTIAL
		// Phi(s) = action-marginalized reachability of the per-state SCORING goal, NOT as the action-edge above
		// (the edge is action-inert far-field, hence useGoalCritic coupling stays off). The shaping term
		// F = gamma*Phi(s') - Phi(s) telescopes, so it densely rewards moving toward scoreable states
		// (positioning -- the thing the egocentric CAR critic structurally can't teach) WITHOUT being
		// farmable by loitering. When on, the goal critic is TRAINED (for Phi) even if useGoalCritic is false.
		// A longer herMaxOffset is now SAFE+desirable here: Phi is a value, not the one-step action edge.
		bool useGoalPotential = false;
		float gcrlGoalPotentialScale = 0.3f; // magnitude of the (normalized) potential advantage term
		int carHerMinOffset = 1;
		int carHerMaxOffset = 20;
		float carHerShortBiasPower = 2.f;

		// ── Magnitude-blend advantage (replaces unit-renorm + the separation gate) ──
		// Per critic, per-row advantage = (taken_score - mean_baseline)/(baseline_spread
		// + sigmaFloor), clamped to +-gcrlSepClamp; NOT renormalized to unit std (that
		// rescales a ~0 signal up to noise -- the bug that froze the single critic).
		// Summed over critics, scaled by gcrlLambda, added onto NormalizeAdvantage(reward).
		// gcrlLambda ramps 0->target over gcrlLambdaWarmupSteps (short bootstrap) then holds.
		// Weak critics self-attenuate (numerator ~0); differential timing (car early /
		// goal late) emerges from real contribution -- no gate. Goal critic scores HER
		// (herGoals), not the synthetic net.
		float gcrlLambda = 0.3f;
		uint64_t gcrlLambdaWarmupSteps = 30'000'000;
		float gcrlSepClamp = 3.f;

		int representationSize = 128;
		// TRIAD-NATIVE: per-critic goal input dim (de-hardcodes MakePsiConfig(6)). GOALSHORT/world-ball=6, ANTI=8.
		int goalInputSize = 6;
		// CAR critic goal dim. The CAR critic is now BALL-AGNOSTIC: its goal is the car's OWN future kinematic
		// + mechanic state -- velocity(3) + forward(3) + up(3) + angVel(3) + air-control flags(3) = 15 --
		// built by AppendCarStateGoal. MUST match that builder's float layout (a mismatch fails the carStates
		// row-size guard and safely benches the car critic). This is car CONTROL (speedflips/wavedashes/
		// aerials), not ball control. Flags = isOnGround, HasFlipOrJump, isFlipping (the air-maneuver state).
		int carGoalInputSize = 15;
		// Hidden layers of the phi tail (action-fusion network on top of the shared_head embedding).
		// Input = shared_head output size + numActions; output = representationSize.
		std::vector<int> phiTailLayerSizes = { 256, 256 };
		// Hidden layers of the psi goal encoder (maps a goalInputSize-dim goal -> representationSize).
		// Was hardcoded at {1024,1024,1024,1024} (~3.3M params PER critic) to encode a ~6-dim goal --
		// wildly oversized vs the 256-wide actor/critic. 2x256 is still generous for a tiny input.
		std::vector<int> psiLayerSizes = { 256, 256 };
		int criticEpochs = 1;
		int64_t criticMiniBatchSize = 256;
		int64_t policyScoreBatchSize = 4096;
		int64_t infoSubSample = 512;
		float criticLR = 3e-4f;
		float tau = 0.05f; // TRIAD-NATIVE: 0.02->0.05 (0.02 saturated logits +-50, starved early gradient; unified train+Score)
		float varReg = 0.3f; // (UNUSED: replaced by VICReg below)
		float logsumexpPenaltyCoeff = 0.01f;
		// TRIAD-NATIVE VICReg anti-collapse, applied to the PRE-L2-normalization RAW phiTail/psi output
		// (the unit-variance hinge is unsatisfiable on the L2-normalized unit sphere where per-dim std is
		// structurally pinned ~1/sqrt(reprDim)). Replaces the inverse-std varReg (which pushed std unbounded).
		float vicVar = 1.0f;  // variance-hinge coeff
		float vicCov = 0.04f; // off-diagonal covariance coeff
		// ── TRIAD-NATIVE coupling controller (used by the PPOLearner advantage rewrite) ──
		// lambda integral controller (replaces the fixed warmup ramp + the outer renorm):
		// lambdaEff *= exp(gain*(ratioTarget - std(gcrl)/std(base))), clamped [max(min,warmupFloor(t)),max].
		float gcrlLambdaCtrlGain = 0.05f;
		float gcrlLambdaMin = 0.0f;
		float gcrlLambdaMax = 2.0f;
		float gcrlRatioTarget = 1.0f;   // target std(gcrl)/std(base) ~ 1:1 (the working-run signature)
		float gcrlRatioEmaDecay = 0.9f;
		float gcrlRenormStdEma = 0.7f;  // RenormToStd EMA (the inner z533fbde explosion guard, KEPT)
		// always-on smooth variance-weight w=sigmoid((crossActionSpread - sigmaMin)/scale) (replaces the dead gate)
		float gcrlVarWeightSigmaMin = 0.05f;
		float gcrlVarWeightScale = 0.05f;
		// ── TRIAD-NATIVE anti (defensive) critic: net-relative opponent-threat goal (8d), TD row-only ──
		bool useAntiCritic = false;
		int antiHerMinOffset = 1;
		int antiHerMaxOffset = 15;
		// ── TRIAD-NATIVE TD-contrastive (CDPC) for GOALSHORT/ANTI: EMA target + pi-weighted soft bootstrap ──
		bool useTDContrastive = false;
		float tdEmaDecay = 0.005f;            // EMA target tracking rate (legacy per-step; superseded by tdEmaHalfLifeIters)
		uint64_t tdRampSteps = 20'000'000;    // MC->TD target-blend ramp (in-run stability schedule)
		// ── FORK2 TD-contrastive additions (design-locked + adversarially verified) ──
		// Reachability-horizon discount for the GOALSHORT/REACH one-step bootstrap. SEPARATE from
		// gaeGamma (return horizon vs reachability horizon). H ~= 1/(1-gamma); 0.93 => ~14-step
		// strike window, matching production herMaxOffset=15. Do NOT overload gaeGamma.
		float tdContrastiveGamma = 0.93f;
		// EMA cadence as a HALF-LIFE IN ITERATIONS, applied once per Train() call (NOT per minibatch:
		// 0.005/step over ~5100 steps/iter would turn the target over ~37x/iter -> no lag). Per-iteration
		// lerp coefficient = 1 - 2^(-1/tdEmaHalfLifeIters).
		float tdEmaHalfLifeIters = 10.f;
		// EMA the TRUNK into the target branch too (target = EMA of the WHOLE encoder), not just the
		// tail+goalEncoder, so f^- is not phiTail^-(live drifting trunk) off-manifold.
		bool tdEmaTrunk = true;
		// Policy-sampled actions K for the soft-value bootstrap (importance form
		// V^- = tau*(logsumexp_{a'~pi}(f^-) - log K)). Full 126-action enumeration is ~10x the MC FLOPs;
		// K=12 gives the same operator at ~K/126 cost.
		int tdSoftValueActionSamples = 12;
		// Per-iteration collapse guard (hysteresis): if the soft-value action-distribution entropy
		// fraction (mean over valid rows of H_i/log n_valid_i) drops below tdCollapseEnterFrac, force
		// r_eff=0 (pure MC) until it recovers above tdCollapseExitFrac.
		float tdCollapseEnterFrac = 0.10f;
		float tdCollapseExitFrac = 0.20f;
		// ── FORK2 Part C: GOALSHORT scoring-goal mix ──
		// Fraction of GOALSHORT (herGoals, useCarGoals==false) training rows whose goal is drawn from the
		// SYNTHETIC scoring goal (net-mouth + velocity lead) instead of the achieved future. Construction-
		// side in relabelHERGoals. Synthetic rows are flagged (gcrlScoringMask) and EXCLUDED from the TD
		// bootstrap (MC-only) since V^-(s',g_synthetic) would be an OOD extrapolation. CAR is never mixed.
		float scoringGoalMixFrac = 0.5f;

		float targetSpeed = 1500.f;
		float targetSpeedJitter = 0.f;

		// (2A) Least-resistance scoring goal: lead the ball along its own
		// velocity before clamping into the goal mouth, so a ball angled at the
		// far post targets where it would actually cross, not the nearest point.
		// The lead is scaled by alignment-to-goal and capped, and vanishes as the
		// ball slows (degrading smoothly to the closest mouth point). See
		// AppendScoringGoal().
		float scoringGoalLeadTime = 0.6f;  // seconds of velocity lead
		float scoringGoalMaxLead = 1500.f; // cap on lead distance (uu); <=0 disables

		// (2B) Only train the contrastive critic on matches (segments between
		// resets) where the ball actually moved (peak speed >= this), so it never
		// learns the degenerate "ball never moves" manifold. Advantage scoring is
		// unaffected.
		float gcrlMinBallMoveSpeed = 300.f;

		// (2C) Fraction of HER training goals drawn from the most-goalward future
		// ball state within the offset window (the rest use short-biased horizon
		// sampling), so real near-net states populate the critic's goal space and
		// the (2A) scoring goal becomes in-distribution.
		float herGoalwardBias = 0.5f;

		// Deprecated by v1 short-biased HER sampling. Kept for source compatibility.
		int immediateMin = 1;
		int immediateMax = 4;
		int shortMin = 5;
		int shortMax = 15;
		int mediumMin = 16;
		int mediumMax = 45;
		int longMin = 46;
		int longMax = 90;

		float immediateWeight = 0.20f;
		float shortWeight = 0.35f;
		float mediumWeight = 0.30f;
		float longWeight = 0.15f;

		float posScaleX = 4096.f;
		float posScaleY = 6000.f;
		float posScaleZ = 2044.f;
		float velScale = 6000.f;
	};

	// SimBa RSNorm: running per-dim observation standardization applied as the
	// first op of the actor & critic forward (one shared normalizer). Stats are
	// count-based running mean/var (var = M2/n), updated once per rollout and
	// frozen during the K epochs; persisted with the weights. Default-off.
	// This is the only obs-normalization path (the old standardizeObs was removed).
	struct RSNormConfig {
		bool enabled = false;
		double eps = 1e-8;
		double initVar = 1.0;    // variance at init (mu = 0)
		double initCount = 1e-4; // small init count so early steps are well-defined
		double clipRange = 0.0;  // optional clamp of normalized obs; <= 0 disables (NOT SimBa)
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

		// BF16 autocast (AMP) for the PPO *update* on CUDA. Casts the dense policy/critic matmuls in the
		// forward+backward to BF16 tensor-core ops; numerically-sensitive ops (softmax/exp/log, layer_norm,
		// losses) stay fp32 via autocast's promotion lists. Master weights stay fp32 and BF16 needs no
		// GradScaler. ~up to 2x on the update matmuls + ~half the activation memory on Ampere+/Blackwell.
		// OFF by default: BF16 adds noise to logits/ratios, so validate KL/entropy/touch + non-finite
		// counts on a real run before trusting it. CUDA-only (ignored on CPU/MPS).
		bool useAMP = false;

		// Allocate the shuffled minibatch tensors in pinned (page-locked) host memory so the per-minibatch
		// .to(device, non_blocking=true) copies are real async DMA transfers instead of silently synchronous
		// (a pageable source makes non_blocking a no-op). No extra copy -- the gather writes straight into
		// the pinned buffer. Helps a data-transfer/consumption-bound loop. CUDA-only.
		bool pinBatchMemory = false;

		PartialModelConfig policy, critic, sharedHead;

		int epochs = 2;
		float policyLR = 3e-4f; // Policy learning rate
		float criticLR = 3e-4f; // Critic learning rate

		float entropyScale = 0.018f; // The scale of the normalized entropy loss

		// Entropy PIN (Lagrangian temperature). A FIXED entropyScale cannot hold an
		// entropy level, and a SOFT capped controller cannot either: a hard maxEntropyScale
		// ceiling (0.10) let entropy collapse to 0.004 once the reward/GCRL gradient
		// overpowered it (runs aooxff9n, 5gjwloq2, z533fbde). When adaptiveEntropy is set,
		// curEntropyScale (persisted in the checkpoint) is updated MULTIPLICATIVELY in
		// log-space each iteration: scale *= exp(entropyScaleAdjustRate*(targetEntropy -
		// entropy)), clamped to [minEntropyScale, maxEntropyScale]. maxEntropyScale should
		// be set HIGH (e.g. 5.0) so it is only a sanity bound, not the operating point --
		// the controller then applies whatever bonus is needed to PIN entropy at target.
		// Only active when adaptiveEntropy is true.
		bool adaptiveEntropy = false;
		float targetEntropy = 0.6f;            // desired normalized entropy in [0,1]
		float maxEntropyScale = 5.0f;          // HIGH sanity bound (uncapped in practice)
		float minEntropyScale = 1e-5f;         // floor (>0 so the multiplicative update can recover)
		float entropyScaleAdjustRate = 0.2f;   // log-space (multiplicative) gain per iteration

		// Whether to ignore invalid actions in the entropy calculation.
		// True means that entropy will be determined only from available actions.
		// False means that entropy for unavailable actions will be zero, 
		//	meaning the entropy of the state is limited to the fraction of available actions in that state.
		bool maskEntropy = false; 

		float clipRange = 0.2f;
		
		// Temperature of the policy's softmax distribution
		float policyTemperature = 1;

		float gaeLambda = 0.95f;
		float gaeGamma = 0.99f;
		float rewardClipRange = 10; // Clip range for normalized rewards, set 0 to disable

		ContrastiveGoalConfig contrastiveGoal;
		RSNormConfig rsNorm;

		bool useGuidingPolicy = false;
		std::filesystem::path guidingPolicyPath = "guiding_policy/"; // Path of the guiding policy model(s)
		float guidingStrength = 0.03f;

		PPOLearnerConfig() {
			policy = {};
			policy.layerSizes = { 256, 256, 256 };
			critic = {};
			critic.layerSizes = { 256, 256, 256 };
			sharedHead = {};
			sharedHead.layerSizes = { 256 };
			sharedHead.addOutputLayer = false;
		}
	};
}
