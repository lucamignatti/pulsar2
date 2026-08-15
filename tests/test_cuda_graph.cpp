#include "TestFramework.h"

#include "../GigaLearnCPP/src/private/GigaLearnCPP/PPO/CudaGraphPolicy.h"

TEST(CudaGraph_PolicyInferenceEquivalenceAndRefresh) {
	auto result = GGL::PolicyCudaGraph::RunVerification();
	if (!result.cudaAvailable)
		return;
	std::cout << "[CUDA GRAPH] max_abs fp32=" << result.fp32MaxAbsError
		<< " native_half=" << result.halfMaxAbsError
		<< " native_refresh=" << result.refreshedHalfMaxAbsError
		<< " fp16=" << result.fp16MaxAbsError
		<< " fp16_refresh=" << result.refreshedFp16MaxAbsError
		<< " captures=" << result.captures
		<< " replays=" << result.replays
		<< " fallbacks=" << result.fallbacks << std::endl;

	CHECK(result.fp32MaxAbsError <= 1e-6f);
	CHECK(result.halfMaxAbsError <= 1e-6f);
	CHECK(result.refreshedHalfMaxAbsError <= 1e-6f);
	CHECK(result.refreshedHalfChange > 1e-5f);
	CHECK(result.fp16MaxAbsError <= 1e-6f);
	CHECK(result.refreshedFp16MaxAbsError <= 1e-6f);
	CHECK(result.refreshedFp16Change > 1e-5f);
	CHECK(result.actionAndLogProbMatched);
	CHECK(result.finiteGuardMatched);
	CHECK(result.eagerFallbackMatched);
	CHECK(result.cacheEvicted);
	CHECK(!result.disabledAfterOom);
	CHECK(result.captures >= 4);
	CHECK(result.replays >= 8);
	CHECK(result.fallbacks >= 1);
}
