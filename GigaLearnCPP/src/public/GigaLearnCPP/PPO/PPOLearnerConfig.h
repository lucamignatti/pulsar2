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
		int baselineActionSamples = 4;

		// ── Car critic ──────────────────────────────────────────────────────────
		// A second, isolated contrastive critic whose goal is the car-local
		// (egocentric) ball pos+vel a SHORT horizon ahead -- local controllability,
		// the action-attributable signal the single ball-goal critic lacks. Its own
		// short, near-term HER window (no goalward bias). Needs the obs builder to
		// expose GetCarLocalBallOffset() (>=0); otherwise the car critic is skipped.
		bool useCarCritic = false;
		// Share ONE state-action encoder (phi) across the GCRL critics (goal + car), each with its own
		// small goal encoder (psi). Default off (separate critics = the A/B baseline). The goal critic
		// owns/saves/LR-steps the shared phi; the car critic references it. NOTE: the shared phi is
		// updated SEQUENTIALLY each iteration -- goal Train steps it, then car Train steps it (StepOptim
		// applies each gradient to the weights, so both land; this is alternating multi-task, not summed
		// joint gradients). Pays off as the head count grows.
		bool useSharedBase = false;
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

		// ── Potential-based shaping (the unified consumption; vs the magnitude-blend above) ──
		// When usePotentialShaping is true, GCRL enters as potential-based reward shaping
		// reward_k = gamma*Phi_k(s') - Phi_k(s) added to the reward stream BEFORE GAE, instead of
		// the advantage blend (PrepareGCRLPolicyAdvantages is skipped). Phi_k(s) = mean over
		// baselineActionSamples random actions of the head's reachability toward a FIXED shaping-goal:
		//   car  -> contact (car-local ball at origin),
		//   goal -> soft-max over scoringRangeSamples goal-mouth points (temperature potentialScoringTemp).
		// Policy-invariant (Ng'99) and farming-proof (deltas telescope). Default false = the
		// magnitude-blend advantage, kept as the A/B baseline. Defense + shared-base are follow-ups.
		bool usePotentialShaping = false;
		float potentialShapingScale = 0.3f;   // weight on the summed per-head shaping reward
		int scoringRangeSamples = 5;          // goal-mouth x-samples defining the goal head's range
		float potentialScoringTemp = 1.0f;    // soft-max temperature over the scoring range (->0 = hard max)
		// Defense head: Phi_defense(s) = -(soft-max over the agent's opponents of their goal-reachability),
		// reusing the goal head's Phi evaluated on each opponent's ego-obs, regrouped by (arena, step).
		// Agency-correct (the threat is the opponents'), policy-invariant (a potential). Only active in
		// potential-shaping mode. Costs nothing extra to compute Phi (reuses the goal head).
		bool potentialDefense = true;

		int representationSize = 128;
		int criticEpochs = 1;
		int64_t criticMiniBatchSize = 256;
		int64_t policyScoreBatchSize = 4096;
		int64_t infoSubSample = 512;
		float criticLR = 3e-4f;
		float tau = 0.02f;
		float varReg = 0.3f;
		float logsumexpPenaltyCoeff = 0.01f;

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

		PartialModelConfig policy, critic, sharedHead;

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
