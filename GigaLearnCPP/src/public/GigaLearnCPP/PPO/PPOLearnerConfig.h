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

		int representationSize = 64;
		int criticEpochs = 1;
		int64_t criticMiniBatchSize = 256;
		float criticLR = 3e-4f;
		float logsumexpPenaltyCoeff = 0.1f;

		float targetSpeed = 1500.f;
		float targetSpeedJitter = 500.f;

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
