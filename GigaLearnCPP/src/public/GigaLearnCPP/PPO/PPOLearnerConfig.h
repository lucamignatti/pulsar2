#pragma once
#include <RLGymCPP/BasicTypes/Lists.h>
#include <RLGymCPP/CommonValues.h>

#include "../Util/ModelConfig.h"

#include <atomic>

namespace RLGC {
	class FrontierStateBuffer;
}

namespace GGL {

	struct SORSStep {
		Vec ballPos, ballVel, playerPos, playerVel;
		Team team = Team::BLUE;
		bool touch = false;
		bool shot = false;
		bool goalFor = false;
		bool goalAgainst = false;
		bool firstToBall = false;
		bool playerOnGround = false;
		bool playerDemoed = false;
		bool gotFlipReset = false;
		bool prevValid = false;
		bool prevGotFlipReset = false;
		bool prevFlipping = false;
		bool prevOnGround = false;
		float playerUpZ = 1;
	};

	class SORSLabel {
	public:
		virtual bool IsTrigger() const { return false; }
		virtual float GetLabel(const std::vector<SORSStep>& steps, int stepIdx) const = 0;
		virtual ~SORSLabel() {}
	};

	struct WeightedSORSLabel {
		SORSLabel* label;
		float weight;

		WeightedSORSLabel(SORSLabel* label, float weight) : label(label), weight(weight) {}
		WeightedSORSLabel(SORSLabel* label, int weight) : label(label), weight(weight) {}
	};

	class AirTouchSORSLabel : public SORSLabel {
	public:
		float minBallHeight;
		AirTouchSORSLabel(float minBallHeight = 500) : minBallHeight(minBallHeight) {}
		virtual bool IsTrigger() const override { return true; }
		virtual float GetLabel(const std::vector<SORSStep>& steps, int stepIdx) const override {
			const auto& step = steps[stepIdx];
			return step.touch && !step.playerOnGround && step.ballPos.z >= minBallHeight;
		}
	};

	class FlipResetSORSLabel : public SORSLabel {
	public:
		float minBallHeight;
		FlipResetSORSLabel(float minBallHeight = 500) : minBallHeight(minBallHeight) {}
		virtual bool IsTrigger() const override { return true; }
		virtual float GetLabel(const std::vector<SORSStep>& steps, int stepIdx) const override {
			const auto& step = steps[stepIdx];
			return step.prevValid && step.gotFlipReset && !step.prevGotFlipReset && !step.playerOnGround && step.ballPos.z >= minBallHeight;
		}
	};

	class PostResetTouchSORSLabel : public SORSLabel {
	public:
		float minBallHeight;
		PostResetTouchSORSLabel(float minBallHeight = 500) : minBallHeight(minBallHeight) {}
		virtual bool IsTrigger() const override { return true; }
		virtual float GetLabel(const std::vector<SORSStep>& steps, int stepIdx) const override {
			const auto& step = steps[stepIdx];
			return step.touch && step.gotFlipReset && !step.playerOnGround && step.ballPos.z >= minBallHeight;
		}
	};

	class WavedashSORSLabel : public SORSLabel {
	public:
		virtual bool IsTrigger() const override { return true; }
		virtual float GetLabel(const std::vector<SORSStep>& steps, int stepIdx) const override {
			const auto& step = steps[stepIdx];
			return step.prevValid && step.playerOnGround && step.prevFlipping && !step.prevOnGround;
		}
	};

	class UsefulBallDeltaSORSLabel : public SORSLabel {
	public:
		int lookahead;
		float velThreshold, progressThreshold;
		UsefulBallDeltaSORSLabel(int lookahead = 30, float velThreshold = 300, float progressThreshold = 300)
			: lookahead(lookahead), velThreshold(velThreshold), progressThreshold(progressThreshold) {}
		virtual float GetLabel(const std::vector<SORSStep>& steps, int stepIdx) const override {
			const auto& eventStep = steps[stepIdx];
			Vec targetGoal = eventStep.team == Team::BLUE ? RLGC::CommonValues::ORANGE_GOAL_BACK : RLGC::CommonValues::BLUE_GOAL_BACK;
			Vec dirToGoal = (targetGoal - eventStep.ballPos).Normalized();
			float bestProgress = 0;
			int end = RS_MIN((int)steps.size(), stepIdx + lookahead + 1);
			for (int i = stepIdx; i < end; i++) {
				const auto& step = steps[i];
				bestProgress = RS_MAX(bestProgress, (step.ballPos - eventStep.ballPos).Dot(dirToGoal));
				if (step.ballVel.Dot(dirToGoal) > velThreshold || bestProgress > progressThreshold)
					return 1;
			}
			return 0;
		}
	};

	class PossessionOrFirstToBallSORSLabel : public SORSLabel {
	public:
		int lookahead;
		PossessionOrFirstToBallSORSLabel(int lookahead = 45) : lookahead(lookahead) {}
		virtual float GetLabel(const std::vector<SORSStep>& steps, int stepIdx) const override {
			int end = RS_MIN((int)steps.size(), stepIdx + lookahead + 1);
			for (int i = stepIdx; i < end; i++)
				if (steps[i].firstToBall)
					return 1;
			return 0;
		}
	};

	class LostPossessionSORSLabel : public PossessionOrFirstToBallSORSLabel {
	public:
		LostPossessionSORSLabel(int lookahead = 45) : PossessionOrFirstToBallSORSLabel(lookahead) {}
		virtual float GetLabel(const std::vector<SORSStep>& steps, int stepIdx) const override {
			return 1.0f - PossessionOrFirstToBallSORSLabel::GetLabel(steps, stepIdx);
		}
	};

	class GoodRecoverySORSLabel : public SORSLabel {
	public:
		int lookahead;
		float minUpZ;
		GoodRecoverySORSLabel(int lookahead = 30, float minUpZ = 0.5f) : lookahead(lookahead), minUpZ(minUpZ) {}
		virtual float GetLabel(const std::vector<SORSStep>& steps, int stepIdx) const override {
			int end = RS_MIN((int)steps.size(), stepIdx + lookahead + 1);
			for (int i = stepIdx; i < end; i++) {
				const auto& step = steps[i];
				if (step.playerOnGround && step.playerUpZ > minUpZ && !step.playerDemoed)
					return 1;
			}
			return 0;
		}
	};

	class BadRecoverySORSLabel : public SORSLabel {
	public:
		int lookahead;
		float maxUpZ;
		BadRecoverySORSLabel(int lookahead = 30, float maxUpZ = 0.0f) : lookahead(lookahead), maxUpZ(maxUpZ) {}
		virtual float GetLabel(const std::vector<SORSStep>& steps, int stepIdx) const override {
			int end = RS_MIN((int)steps.size(), stepIdx + lookahead + 1);
			for (int i = stepIdx; i < end; i++) {
				const auto& step = steps[i];
				if (step.playerDemoed || (step.playerOnGround && step.playerUpZ < maxUpZ))
					return 1;
			}
			return 0;
		}
	};

	class ShotCreatedSORSLabel : public SORSLabel {
	public:
		int lookahead;
		ShotCreatedSORSLabel(int lookahead = 60) : lookahead(lookahead) {}
		virtual float GetLabel(const std::vector<SORSStep>& steps, int stepIdx) const override {
			int end = RS_MIN((int)steps.size(), stepIdx + lookahead + 1);
			for (int i = stepIdx; i < end; i++)
				if (steps[i].shot)
					return 1;
			return 0;
		}
	};

	class GoalForSORSLabel : public SORSLabel {
	public:
		int lookahead;
		GoalForSORSLabel(int lookahead = 90) : lookahead(lookahead) {}
		virtual float GetLabel(const std::vector<SORSStep>& steps, int stepIdx) const override {
			int end = RS_MIN((int)steps.size(), stepIdx + lookahead + 1);
			for (int i = stepIdx; i < end; i++)
				if (steps[i].goalFor)
					return 1;
			return 0;
		}
	};

	class GoalAgainstSORSLabel : public SORSLabel {
	public:
		int lookahead;
		GoalAgainstSORSLabel(int lookahead = 90) : lookahead(lookahead) {}
		virtual float GetLabel(const std::vector<SORSStep>& steps, int stepIdx) const override {
			int end = RS_MIN((int)steps.size(), stepIdx + lookahead + 1);
			for (int i = stepIdx; i < end; i++)
				if (steps[i].goalAgainst)
					return 1;
			return 0;
		}
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

		// Step the optimizers after every minibatch instead of accumulating gradients
		// over the whole batch and stepping once. More optimizer steps per batch often
		// buys sample efficiency; watch Mean KL Divergence and lower policyLR if it
		// rises above ~0.01 sustained.
		bool stepPerMiniBatch = false;

		// Collect the next batch while the current one trains (1 update off-policy). Big speedup when env-step-bound.
		bool asyncCollection = true;

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
		bool adaptiveEntropy = false; // Tune entropyScale online to hold policy entropy near targetEntropy
		float targetEntropy = 0.75f; // Target normalized policy entropy in [0, 1]
		float adaptiveEntropyLR = 1e-3f; // Step size for the entropyScale controller
		float minEntropyScale = 0.0f; // Lower clamp for the adaptive entropy coefficient
		float maxEntropyScale = 1.0f; // Upper clamp for the adaptive entropy coefficient
		// Whether to ignore invalid actions in the entropy calculation.
		// True means that entropy will be determined only from available actions.
		// False means that entropy for unavailable actions will be zero, 
		//	meaning the entropy of the state is limited to the fraction of available actions in that state.
		bool maskEntropy = false; 

		float clipRange = 0.2f;

		// SB3-style KL early stop: if the batch-mean KL divergence exceeds this after any
		// batch, the remaining epochs/batches of the iteration are skipped. A hard speed
		// limit on policy drift (especially important with stepPerMiniBatch). 0 disables.
		float maxMeanKL = 0;
		
		// Temperature of the policy's softmax distribution
		float policyTemperature = 1;

		float gaeLambda = 0.95f;
		float gaeGamma = 0.99f;
		float rewardClipRange = 10; // Clip range for normalized rewards, set 0 to disable

		bool useGuidingPolicy = false;
		std::filesystem::path guidingPolicyPath = "guiding_policy/"; // Path of the guiding policy model(s)
		float guidingStrength = 0.03f;

		// ── Quasimetric GCRL ("game sense" channel) ──
		// Three quasimetric critics learn, contrastively (InfoNCE) over hindsight-relabeled
		// future ball goals, a directional distance from (state, action) to a goal. Their
		// combined advantage is unit-normalized and blended into the policy gradient on top
		// of the reward-driven GAE advantage. GCRL teaches positioning/anticipation; the
		// dense rewards (via GAE) teach mechanics.
		bool useGCRL = true;
		float gcrlAdvScale = 1.0f;   // Blend weight: adv = norm(GAE_adv) + gcrlAdvScale * GCRL_adv
		int64_t gcrlAdvScaleAnnealStart = 0; // Timesteps to start policy influence ramp; -1 -> current checkpoint
		int64_t gcrlAdvScaleAnnealSteps = 0; // Timesteps to ramp from 0 to gcrlAdvScale; 0 disables
		float gcrlAntiScale = 0.85f; // Weight of the pessimistic "anti" critic in the GCRL advantage
		float gcrlCarScale = 0.5f;   // Weight of the car-positioning critic in the GCRL advantage
		// Gradient surgery: instead of summing the reward advantage and the GCRL ("game sense")
		// advantage, remove the reward-advantage component that OPPOSES game-sense per sample
		// (per-sample PCGrad; the two channels are colinear per sample so projection == sign
		// reconciliation). Stops shaped reward from reinforcing actions game-sense flags as bad,
		// which should remove destabilizing updates and widen the usable LR. Watch
		// "GCRL/Surgery Conflict Fraction".
		bool gcrlSurgery = false;
		float gcrlSurgeryStrength = 1.0f; // 0 == today's pure blend; 1 == full removal of the conflicting reward component
		// Magnitude gate (per-mode-normalized std units): surgery only fires where BOTH the
		// reward and game-sense advantages exceed this AND oppose -- so it targets real
		// misalignment, not the mass of near-zero sign-flips that pin the raw conflict fraction
		// at ~0.5. 0 == blunt all-samples (old behavior); ~1.0 == clearly-substantial only.
		// Watch "GCRL/Surgery HiMag Conflict" (>0.5 == real misalignment) and "Removed Frac".
		float gcrlSurgeryMagThreshold = 1.0f;
		float gcrlTau = 0.02f;       // Embedding temperature (sharp contrast)
		// HER relabeling horizons are in STEPS. At tickSkip 4 there are ~30 steps/sec, so
		// these defaults span roughly 0.25s..1.05s of "future" — long enough to be strategic.
		int gcrlHorizon = 32;        // Max future offset (upper bound of the HER goal window)
		int gcrlMinHorizon = 8;      // Min future offset (lower bound of the HER goal window)
		bool gcrlUseVariableHER = true; // Sample goal offset uniformly in [minH, H]; else fixed H
		float gcrlInfoNCECoef = 0.5f;     // Weight of the InfoNCE loss in the combined objective
		float gcrlInfoNCEPenalty = 0.01f; // logsumexp penalty inside the InfoNCE loss
		float gcrlVarReg = 0.3f;          // Embedding variance regularization (anti-collapse)
		int gcrlInfoSubSample = 512;      // Sub-batch size for the InfoNCE contrastive matrix
		int gcrlReprDim = 128;            // Embedding (output) dim; phi and psi MUST share it
		float gcrlLR = 0;                 // GCRL critic learning rate; 0 -> use policyLR
		// Counterfactual action baseline for the GCRL advantage: K extra no-grad critic
		// forwards with within-batch shuffled actions, subtracted from Q before normalization.
		// Turns "this state is near a goal" into "this action beats a typical action here".
		// 0 disables (raw Q is used, as before).
		int gcrlBaselineSamples = 0;
		bool useGCRLRewardGate = false;   // Gate selected dense rewards with terminal-oriented GCRL progress
		float gcrlRewardGateInfluence = 1.0f; // 0 -> ungated, 1 -> full gate
		int64_t gcrlRewardGateAnnealStart = -1; // Timesteps to start gate influence ramp; -1 -> current checkpoint
		int64_t gcrlRewardGateAnnealSteps = 100'000'000; // Timesteps to ramp from 0 to gcrlRewardGateInfluence
		float gcrlRewardGateMin = 0.2f;   // Deprecated: signed gates now use tanh(delta) in [-1, 1]
		float gcrlRewardGateSharpness = 1.0f; // Signed gate sharpness for terminal progress
		float gcrlRewardGateAntiScale = 0.85f; // Own-goal danger penalty inside gate progress
		float gcrlRewardGateTargetVel = 1200.0f; // Terminal goal target ball velocity in uu/s
		int gcrlRewardGateLookahead = 15; // Steps to measure terminal-progress delta (~0.5s at tickSkip 4)
		float gcrlAerialRewardGateInfluence = 1.0f; // Separate gate influence for aerial rewards
		float gcrlAerialRewardGateStartInfluence = 0.0f; // Initial aerial gate influence before annealing
		int64_t gcrlAerialRewardGateAnnealStart = -1; // Timesteps to start aerial gate influence ramp; -1 -> current checkpoint
		int64_t gcrlAerialRewardGateAnnealSteps = 1'000'000'000; // Timesteps to ramp start influence to target influence
		float curriculumRewardScale = 1.0f; // Temporary non-aerial curriculum reward scale
		int64_t curriculumRewardAnnealStart = -1; // Timesteps to start decaying the curriculum reward; -1 -> current checkpoint
		int64_t curriculumRewardAnnealSteps = 800'000'000; // Timesteps to decay curriculum reward scale to 0
		float aerialCurriculumRewardScale = 1.0f; // Temporary aerial approach reward scale
		int64_t aerialCurriculumRewardAnnealStart = -1; // Timesteps to start decaying the curriculum reward; -1 -> current checkpoint
		int64_t aerialCurriculumRewardAnnealSteps = 800'000'000; // Timesteps to decay curriculum reward scale to 0
		// Competence gates for the timestep anneals: anneal progress only accumulates while
		// the EMA competence ratio is at/above the gate, so the scaffolding can't expire while
		// the policy still needs it (run rwkkyfej: chase/touch curriculum hit ~7% strength at
		// 2.3B ts with touch ratio still ~0.001, leaving only the time penalty). <= 0 disables
		// the gate (pure timestep anneal). curriculumAnnealTouchRatioGate reads the live-player
		// ball-touch ratio and holds BOTH the curriculum reward anneal AND the two GCRL reward
		// gate influence ramps (the gate filters rewards by critic-judged terminal progress,
		// which is noise until the critics have seen ball touches — run tkpk0780 ramped
		// influence to 1.0 on its wall clock over a touchless world and the touch-teaching
		// rewards spent ~3B steps multiplied by sigmoid(noise)). The aerial gate reads the
		// high air touch ratio (airborne touch, ball z >= 500).
		float curriculumAnnealTouchRatioGate = 0.0f;
		float aerialCurriculumAnnealAirTouchRatioGate = 0.0f;
		float competenceEMADecay = 0.99f; // Per-iteration decay of the competence EMAs (~10M ts horizon at 150k ts/itr)
		// Hidden architecture of the phi/psi towers (output is always gcrlReprDim). Configured
		// like policy/critic: layerSizes / activationType / addLayerNorm / optimType.
		PartialModelConfig gcrlCritic;

		// ── SORS mechanic densification reward model ──
		// Learns a dense reward from short, utility-weighted sparse mechanic windows.
		// The model consumes obs + continuous action components and its clipped output is
		// added to collected rewards before GAE.
		bool useSORS = false;
		float sorsRewardScale = 0.25f;
		int64_t sorsRewardScaleAnnealStart = 0; // Timesteps to start reward influence ramp; -1 -> current checkpoint
		int64_t sorsRewardScaleAnnealSteps = 0; // Timesteps to ramp from 0 to sorsRewardScale; 0 disables
		float sorsRewardClipRange = 1.0f;
		float sorsLR = 0; // SORS reward learning rate; 0 -> use policyLR
		int sorsWarmupIters = 8;
		int sorsTrainPairs = 4096;
		int sorsMaxReplayWindows = 20000;
		float sorsMinLabelDelta = 0.25f;
		int sorsWindowBefore = 45;
		int sorsWindowAfter = 30;
		std::vector<WeightedSORSLabel> sorsLabels;
		PartialModelConfig sorsReward;

		// ── Frontier reset curriculum (Feature A; requires useGCRL) ──
		// Harvests full physics states from rollouts at timesteps where the goal critic
		// contradicts itself (quasimetric self-consistency: moving a few steps along the
		// trajectory toward a goal that WAS achieved must not increase d(s, g); positive
		// increments are self-inconsistency), backtracks ~1s, and feeds the snapshots to a
		// FrontierState setter so episodes start at the edge of critic competence instead
		// of always at kickoff/random. Scoring is one batched no-grad pass on the learner
		// thread next to the existing GCRL gate pass; workers only sample a mutexed ring.
		bool useFrontierResets = false;
		// consistency = monotonicity violation of d(s_t, g_t) along the trajectory (implemented).
		// ensemble = variance over extra phi/psi heads (planned fallback if consistency proves
		// too noisy; selecting it errors at startup so a typo can't silently no-op).
		enum class FrontierUncertaintyMode { CONSISTENCY, ENSEMBLE };
		FrontierUncertaintyMode frontierUncertaintyMode = FrontierUncertaintyMode::CONSISTENCY;
		float frontierSpikePercentile = 0.90f; // self-normalized spike threshold (batch quantile, no absolute scale)
		int frontierBacktrackSteps = 30;       // reset candidate is the state ~1s before the spike (tickSkip 4)
		// Capture a physics snapshot every Nth env step. Bounds the in-flight snapshot memory:
		// at 5120 1v1 arenas / ~300-step episodes / interval 16, ~100k snapshots x ~400B ≈ 40-50MB.
		// Longer episodes or team modes scale this linearly; raise the interval if RAM matters.
		int frontierSnapshotInterval = 16;
		int frontierBufferSize = 4096;     // ring capacity; FIFO turnover every ~4-8 iters keeps it tracking the current policy
		int frontierBufferMinFill = 256;   // FrontierState falls back to its child setter below this
		float frontierResetFraction = 0.15f; // share of resets routed to FrontierState (read by user code when building CombinedState)
		// Learner-side handle to the buffer FrontierState samples from. Owned by user code
		// (created in main() before the Learner, alongside the state setters — see ExampleMain);
		// raw pointer matching the envSet userInfo convention. NULL disables capture even if
		// useFrontierResets is set.
		RLGC::FrontierStateBuffer* frontierBuffer = nullptr;

		// ── Difficulty-aware HER goal sampling (Feature B; requires useGCRL) ──
		// Instead of a uniform tau in [gcrlMinHorizon, gcrlHorizon], score K candidate
		// future-ball goals with the current goal critic and prefer goals near the middle of
		// the batch's distance distribution (too close = trivial positives, too far = noise;
		// "intermediate" is a batch percentile, so it tracks competence with no hardcoded
		// difficulty). Relabeling moves from the per-trajectory collector path to one batched
		// learner-thread pass over the combined batch — safe because futureGoals are only
		// consumed at experience tensor creation and in Learn().
		bool useDifficultyHER = false;
		int herCandidates = 8;           // K candidate offsets per anchor
		float herDifficultySigma = 0.2f; // width of the percentile preference around p=0.5
		// Probability of keeping today's uniform offset sampling per anchor. CRITICAL — DO NOT
		// remove: a purely difficulty-shaped positive distribution biases InfoNCE so the
		// critic's calibration at distance extremes degrades, corrupting the very distance
		// estimates this sampler depends on. The uniform floor breaks that feedback loop.
		float herUniformFraction = 0.4f;

		// ── Adaptive (ratcheted-quantile) targets (Feature C) ──
		// Replaces two hand-tuned thresholds with quantiles of what the policy actually
		// achieves, estimated from per-iteration touch events via slow EMA histograms.
		// Stability rules (all mandatory): each target is SEEDED at the hand value (a cold
		// or sparse buffer can never move it), holds unless >= adaptiveTargetMinSamples
		// touches arrived this iteration, rises under a slew limit, and falls only at
		// adaptiveTargetDecayPerIter (default 0 = pure ratchet) — so policy regression can't
		// lower the bar and mask itself with easy reward income. NOTE: raising the gate
		// target re-prices the gate delta; the per-mode gate EMA (~100-iter horizon) absorbs
		// it because the 2%/iter slew moves the target much slower than the EMA adapts. If
		// reward income ever oscillates in phase with target updates, halve the slew limit.
		bool useAdaptiveGateTargetVel = false;        // gcrlRewardGateTargetVel <- quantile of achieved ball speeds at contact
		float adaptiveGateTargetVelQuantile = 0.70f;
		bool useAdaptiveStrongTouchFloor = false;     // StrongTouchReward min threshold <- quantile of achieved hit forces
		float adaptiveStrongTouchFloorQuantile = 0.40f;
		float adaptiveTargetMaxSlewPerIter = 0.02f;   // max relative rise per iteration (2%)
		float adaptiveTargetDecayPerIter = 0.0f;      // max relative fall per iteration (0 = ratchet)
		int adaptiveTargetMinSamples = 32;            // hold the target if fewer touch events this iter
		// Worker-read publication point for the adaptive StrongTouch floor (uu/s). User code
		// creates the atomic, passes it to StrongTouchReward AND sets this pointer; the
		// learner seeds the ratchet from its initial value (the reward's constructor
		// threshold) and stores the ratcheted value here once per iteration (relaxed — a
		// one-iteration-stale read in workers is benign, same as the obsStat torn-read).
		std::atomic<float>* adaptiveStrongTouchFloorAtomic = nullptr;

		// ── Optionality shaping (Feature D; requires useGCRL) ──
		// Potential-based shaping on phi_opt(s) = T*logsumexp(V(g)-d(s,G)/T) - T*log|G| over
		// a stratified goal bank: pays for being in states with many cheap, valuable
		// continuations under the (frozen target) quasimetric. Candidate values are normalized
		// inside the bank and come from existing terminal GCRL scores, not a new trainable
		// model. Delivered strictly as reward-side potential shaping (gamma*phi(s')-phi(s),
		// masked to 0 across episode boundaries) pre-GAE — NOT an advantage stream, NOT
		// blended via gcrlAdvScale. Optional GCRL commit relief only softens negative
		// optionality deltas when terminal prospects improve. Scoring uses ONLY frozen
		// Polyak-target copies of the goal critic's phi/psi (a fast-moving potential breaks
		// the policy-invariance argument), with action components pinned to zero so the
		// potential is a function of state only.
		bool useOptionality = false;
		// Weight vs the normalized reward stream; a shaping voice, not a lead. Anneals to
		// optWeightFinal (not zero) on the same touch-competence gate as the main curriculum.
		// Burn-in: the quasimetric's raw scale drifts as the embedding trains, so an
		// unnormalized weight is meaningless — the first optBurnInIters iterations compute
		// and log everything but inject exactly zero, seeding the reward normalizer.
		float optWeight = 0.05f;
		float optWeightFinal = 0.01f;
		int64_t optAnnealSteps = 2'000'000'000;
		float optCommitReliefScale = 0.0f; // Fraction of negative opt reward relieved when GCRL terminal progress is positive
		float optCommitReliefSharpness = 1.0f; // Sigmoid sharpness over normalized terminal-progress delta
		float optTemp = 1.0f;            // soft-min temperature over goal-bank distances
		float optValueWeight = 0.0f;     // 0 -> reach-only optionality; >0 adds normalized V(g) to bank logits
		float optValueClip = 3.0f;       // clip normalized bank values before multiplying by optValueWeight
		bool optRefineGoals = false;     // Locally gradient-refine top real bank goals before computing phi_opt
		int optRefineTopK = 4;           // Per-state real bank candidates refined; small to bound learner cost
		int optRefineSteps = 2;          // Gradient ascent steps in 6-dim goal-row space
		int optRefineMaxStates = 32768;  // Max states refined per scoring minibatch; <=0 refines every state
		float optRefineStepSize = 0.05f; // Per-step normalized gradient length in obs-space goal coordinates
		float optRefineMaxDelta = 0.20f; // Trust radius from the original real bank goal row
		float optRefineTrustPenalty = 0.1f; // Quadratic penalty inside the refinement objective
		int optBankSize = 2048;
		float optBankRefreshFrac = 0.05f; // FIFO replacement per stratum per iteration; starved strata carry their quota over
		float optBankOffensiveFrac = 0.40f; // achieved ball goals, value-filtered by the goal critic (top half of terminal
		                                    // score — biases optionality toward commitment, the hover-pathology guard)
		float optBankDefensiveFrac = 0.30f; // ball states from own-net-adjacent / ball-behind-car configurations
		float optBankResourceFrac = 0.30f;  // ball states at big-pad pickups and supersonic+boosted moments
		float optTargetTau = 0.005f;      // Polyak coefficient for the frozen scorer nets (once per iteration)
		// Interlock: if the live gcrlAdvScale is >= 0.5 the effective optWeight is halved —
		// two strong critic-derived gradient channels share the miscalibrated-embedding
		// failure mode and must not both be loud (see Opt/Interlock Active).
		int optBurnInIters = 20;

		PPOLearnerConfig() {
			policy = {};
			policy.layerSizes = { 256, 256, 256 };
			critic = {};
			critic.layerSizes = { 256, 256, 256 };
			sharedHead = {};
			sharedHead.layerSizes = { 256 };
			sharedHead.addOutputLayer = false;
			gcrlCritic = {};
			gcrlCritic.layerSizes = { 256, 256 }; // hidden layers of phi/psi (output is gcrlReprDim)
			gcrlCritic.addLayerNorm = false;
			sorsReward = {};
			sorsReward.layerSizes = { 256, 256 };
		}
	};
}
