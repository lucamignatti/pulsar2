#pragma once
#include <RLGymCPP/BasicTypes/Lists.h>
#include <RLGymCPP/CommonValues.h>

#include "../Util/ModelConfig.h"

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
		bool useGCRLRewardGate = false;   // Gate selected dense rewards with terminal-oriented GCRL progress
		float gcrlRewardGateInfluence = 1.0f; // 0 -> ungated, 1 -> full gate
		int64_t gcrlRewardGateAnnealStart = -1; // Timesteps to start gate influence ramp; -1 -> current checkpoint
		int64_t gcrlRewardGateAnnealSteps = 100'000'000; // Timesteps to ramp from 0 to gcrlRewardGateInfluence
		float gcrlRewardGateMin = 0.2f;   // Minimum multiplier for positive gated rewards
		float gcrlRewardGateSharpness = 1.0f; // Sigmoid sharpness for terminal progress
		float gcrlRewardGateAntiScale = 0.85f; // Own-goal danger penalty inside gate progress
		float gcrlRewardGateTargetVel = 1200.0f; // Terminal goal target ball velocity in uu/s
		int gcrlRewardGateLookahead = 15; // Steps to measure terminal-progress delta (~0.5s at tickSkip 4)
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
