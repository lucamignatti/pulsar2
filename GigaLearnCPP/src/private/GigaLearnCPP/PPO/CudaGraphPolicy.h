#pragma once

#include <GigaLearnCPP/Framework.h>

#include <cstdint>
#include <torch/types.h>

namespace GGL {
	class Model;
	class ModelSet;

	/** Cumulative process-local policy-inference CUDA graph counters. **/
	struct PolicyCudaGraphStats {
		uint64_t captures = 0;
		uint64_t replays = 0;
		uint64_t fallbacks = 0;
		uint64_t captureFailures = 0;
		uint64_t variantLimitFallbacks = 0;
		uint64_t cacheEntries = 0;
		bool disabledAfterOom = false;
	};

	/** Results from the small numerical CUDA graph verification used by the test suite. **/
	struct PolicyCudaGraphVerification {
		bool cudaAvailable = false;
		float fp32MaxAbsError = 0;
		float halfMaxAbsError = 0;
		float refreshedHalfMaxAbsError = 0;
		float refreshedHalfChange = 0;
		float fp16MaxAbsError = 0;
		float refreshedFp16MaxAbsError = 0;
		float refreshedFp16Change = 0;
		bool actionAndLogProbMatched = false;
		bool finiteGuardMatched = false;
		bool eagerFallbackMatched = false;
		bool cacheEvicted = false;
		bool disabledAfterOom = false;
		uint64_t captures = 0;
		uint64_t replays = 0;
		uint64_t fallbacks = 0;
	};

	namespace PolicyCudaGraph {
		/**
		 * Runs the policy probability forward eagerly or through an exact-shape CUDA graph.
		 * Graph capture is deliberately limited to no-grad, unsteered inference with a
		 * deferred finite-row result; all other inputs preserve the eager behavior.
		 **/
		torch::Tensor InferPolicyProbs(
			ModelSet& models,
			torch::Tensor obs,
			torch::Tensor actionMasks,
			float temperature,
			bool halfPrec,
			torch::Tensor steerDelta,
			torch::Tensor* outRowOk,
			torch::Tensor precomputedTrunk,
			bool useCudaGraph);

		/** Evicts graphs that captured pointers to the specified model. **/
		void Release(const Model* model);

		PolicyCudaGraphStats GetStats();
		void ResetStats();
		void Clear();

		/** Runs deterministic FP32, mixed-precision, refresh, and finite-guard checks. **/
		RG_IMEXPORT PolicyCudaGraphVerification RunVerification();
	}
}
