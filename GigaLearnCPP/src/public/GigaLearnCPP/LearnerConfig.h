#pragma once
#include <RLGymCPP/BasicTypes/Lists.h>
#include "PPO/PPOLearnerConfig.h"
#include "SkillTrackerConfig.h"

namespace GGL {
	enum class LearnerDeviceType {
		AUTO,
		CPU,
		GPU_CUDA,
		GPU_MPS
	};

	// https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/learner.py
	struct LearnerConfig {
		int numGames = 300;

		int tickSkip = 8;
		int actionDelay = 7;

		bool renderMode = false;
		// If renderMode, this is the scaling of time for the game
		// 1.0 = Run the game at real time
		// 2.0 = Run the game twice as fast as real time
		float renderTimeScale = 1.0f; 

		PPOLearnerConfig ppo = {};

		// Checkpoints are saved here as timestep-numbered subfolders
		//	e.g. a checkpoint at 20,000 steps will save to a subfolder called "20000"
		// Set empty to disable saving
		std::filesystem::path checkpointFolder = "checkpoints"; 

		// Save every timestep
		// Set to zero to just use timestepsPerIteration
		int64_t tsPerSave = 1'000'000;

		int64_t randomSeed = -1; // Set to -1 to use the current time
		int checkpointsToKeep = 8; // Checkpoint storage limit before old checkpoints are deleted, set to -1 to disable
		LearnerDeviceType deviceType = LearnerDeviceType::AUTO; // Auto uses MPS on macOS if available, otherwise CUDA, otherwise CPU

		// CUDA caching allocator: number of iterations between CUDACachingAllocator::emptyCache() calls.
		// The old per-iteration empty (==1) defeats the caching allocator -- it forces a device sync and
		// pushes the next allocations back through cudaMalloc every iteration. A larger interval lets the
		// pool be reused across iterations (per-iteration tensor shapes are near-stable). <=0 disables it
		// entirely (fastest, but watch peak VRAM / fragmentation on long runs). CUDA-only; ignored elsewhere.
		int64_t cudaCacheClearInterval = 50;

		// Enable TF32 matmuls on Ampere+ CUDA GPUs (near-free throughput for the dense MLP forward/backward).
		// TF32 trims the matmul mantissa to ~10 bits; benign for PPO. Set false for strict fp32 reproducibility
		// or A/B numeric comparisons against pre-TF32 runs. CUDA-only; no effect on MPS/CPU.
		bool allowTF32 = true;

		// Overlap collection with consumption (IMPALA-style depth-1 pipeline): collect rollout N+1 on a
		// background thread using a FROZEN clone of the policy + an RSNorm snapshot, while the main thread runs
		// the PPO update on rollout N. The 1-iteration staleness is corrected by PPO's clipped importance ratio
		// (safe at the small per-iter KL this trainer holds). Hides the shorter of collect/consume behind the
		// longer. OFF by default: it is a concurrency change in the most stateful path -- validate on a real run
		// (watch KL/entropy/touch/non-finite AND that Overall SPS rises) and fall back by flipping this off.
		bool overlapCollection = false;

		// Per-step scan of every collected observation for NaN/inf during collection. A debug aid for new
		// obs builders; off by default because it scans the whole obs buffer every env step. When off, a
		// NaN/inf obs will instead surface downstream (e.g. as a non-finite loss). Enable while developing
		// an obs builder.
		bool validateObs = false;

		// Observation normalization is provided by SimBa RSNorm (cfg.ppo.rsNorm);
		// the old collection-time standardizeObs path was removed.

		// Standardize the returns to help the critic (don't disable this unless you know what you're doing)
		bool standardizeReturns = true;
		int maxReturnSamples = 150;

		// Will automatically add the rewards to metrics
		bool addRewardsToMetrics = true;
		int maxRewardSamples = 50; // Maximum reward samples per step for reward metrics
		int rewardSampleRandInterval = 8; // Randomized interval range between sampling rewards (per step)

		// Send metrics to the python metrics receiver
		// The receiver can then log them to wandb or whatever
		bool sendMetrics = true;
		std::string metricsProjectName = "gigalearncpp"; // Project name for the python metrics receiver
		std::string metricsGroupName = "unnamed-runs"; // Group name for the python metrics receiver
		std::string metricsRunName = "gigalearncpp-run"; // Run name for the python metrics receiver

		bool savePolicyVersions = false;
		int64_t tsPerVersion = 25'000'000;
		int maxOldVersions = 32;

		bool trainAgainstOldVersions = false;
		float trainAgainstOldChance = 0.15f; // Chance (from 0 - 1) that an iteration will train against an old version

		SkillTrackerConfig skillTracker = {};
	};
}
