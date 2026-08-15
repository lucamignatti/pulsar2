#include "CudaGraphPolicy.h"

#include "PPOLearner.h"

#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <tuple>

#include <torch/cuda.h>

#ifdef RG_CUDA_SUPPORT
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/util/Exception.h>
#include <cuda_runtime_api.h>
#endif

namespace {
	constexpr size_t MAX_GRAPH_VARIANTS_PER_POLICY = 4;
	GGL::PolicyCudaGraphStats g_graphStats = {};
	bool g_graphsDisabledAfterOom = false;

	struct PolicyForwardResult {
		torch::Tensor probs;
		torch::Tensor rowOk;
	};

	PolicyForwardResult InferPolicyProbsEager(
		GGL::ModelSet& models,
		torch::Tensor obs,
		torch::Tensor actionMasks,
		float temperature,
		bool halfPrec,
		torch::Tensor steerDelta,
		bool deferFiniteCheck,
		torch::Tensor precomputedTrunk) {
		torch::Tensor rowOk;
		auto probs = GGL::PPOLearner::InferPolicyProbsFromModels(
			models, obs, actionMasks, temperature, halfPrec,
			steerDelta, deferFiniteCheck ? &rowOk : nullptr, precomputedTrunk,
			/*useCudaGraph=*/false);
		return { probs, rowOk };
	}

#ifdef RG_CUDA_SUPPORT
	bool IsMemoryFailure(const std::exception& error) {
		const std::string message = error.what();
		return message.find("out of memory") != std::string::npos
			|| message.find("memory allocation") != std::string::npos
			|| message.find("cudaErrorMemoryAllocation") != std::string::npos
			|| message.find("CUDA_ERROR_OUT_OF_MEMORY") != std::string::npos;
	}

	struct GraphKey {
		const GGL::Model* sharedHeadModel = nullptr;
		const GGL::Model* policyModel = nullptr;
		int64_t batchSize = 0;
		int obsSize = 0;
		int numActions = 0;
		int deviceIndex = 0;
		int obsType = 0;
		int maskType = 0;
		int inferenceType = 0;
		bool halfPrec = false;
		float temperature = 1;

		bool operator<(const GraphKey& other) const {
			return std::tie(sharedHeadModel, policyModel, batchSize, obsSize, numActions,
				deviceIndex, obsType, maskType, inferenceType, halfPrec, temperature)
				< std::tie(other.sharedHeadModel, other.policyModel, other.batchSize,
					other.obsSize, other.numActions, other.deviceIndex, other.obsType,
					other.maskType, other.inferenceType, other.halfPrec, other.temperature);
		}
	};

	struct PolicyProbsGraph {
		torch::Tensor staticObs;
		torch::Tensor staticMasks;
		torch::Tensor staticOutput;
		torch::Tensor staticRowOk;
		std::optional<c10::cuda::CUDAStream> captureStream;
		at::cuda::CUDAEvent inputsReady;
		at::cuda::CUDAEvent graphDone;
		// Do not use at::cuda::CUDAGraph on CUDA 11.2 / driver 460. ATen
		// capture_begin calls cudaStreamGetCaptureInfo(..., pId=nullptr) after
		// BeginCapture; the 11.2 driver writes through that pointer and SIGSEGVs.
		cudaGraphExec_t graphExec = nullptr;
		c10::cuda::MempoolId_t mempoolId{0, 0};
		int captureDev = -1;
		bool mempoolHeld = false;
		bool captured = false;

		~PolicyProbsGraph() {
			ReleaseGraphResources();
		}

		void ReleaseGraphResources() {
			if (graphExec) {
				cudaGraphExecDestroy(graphExec);
				graphExec = nullptr;
			}
			if (mempoolHeld && captureDev >= 0) {
				c10::cuda::CUDACachingAllocator::releasePool(captureDev, mempoolId);
				mempoolHeld = false;
			}
			captured = false;
		}

		PolicyForwardResult Run(
			GGL::ModelSet& models,
			torch::Tensor obs,
			torch::Tensor actionMasks,
			float temperature,
			bool halfPrec) {
			const int deviceIndex = obs.device().index();
			c10::cuda::CUDAGuard deviceGuard(obs.device());
			if (!captureStream)
				// Collection already uses high priority to interleave its small forwards with
				// the learn pass. Keep graph replay on the same priority class.
				captureStream.emplace(c10::cuda::getStreamFromPool(true, deviceIndex));
			auto& stream = *captureStream;
			auto callerStream = c10::cuda::getCurrentCUDAStream(deviceIndex);
			// The collection stream may have just refreshed the half mirror. The capture
			// stream must observe both that refresh and the current input tensors.
			inputsReady.record(callerStream);
			inputsReady.block(stream);
			c10::cuda::CUDAStreamGuard streamGuard(stream.unwrap());
			bool captureStarted = false;
			auto endFailedCapture = [&]() {
				if (!captureStarted)
					return;
				cudaGraph_t aborted = nullptr;
				cudaStreamEndCapture(stream.stream(), &aborted);
				if (aborted)
					cudaGraphDestroy(aborted);
				captureStarted = false;
				if (mempoolHeld && captureDev >= 0)
					c10::cuda::CUDACachingAllocator::endAllocateStreamToPool(
						captureDev, stream.stream());
			};
			try {
				if (!captured) {
					staticObs = torch::empty_like(obs);
					staticMasks = torch::empty_like(actionMasks);
					staticObs.copy_(obs, /*non_blocking=*/true);
					staticMasks.copy_(actionMasks, /*non_blocking=*/true);
					for (int i = 0; i < 3; i++) {
						auto warm = InferPolicyProbsEager(
							models, staticObs, staticMasks, temperature, halfPrec,
							{}, /*deferFiniteCheck=*/true, {});
						staticOutput = warm.probs;
						staticRowOk = warm.rowOk;
					}
					// Release the warm-up outputs before capture. Their storage belongs to the
					// ordinary caching allocator, not this graph's private memory pool.
					staticOutput = torch::Tensor();
					staticRowOk = torch::Tensor();
					static std::once_flag rawCaptureLog;
					std::call_once(rawCaptureLog, []() {
						RG_LOG("CUDA graphs: raw capture (CUDA 11.2 GetCaptureInfo null pId SIGSEGV)");
					});
					captureDev = deviceIndex;
					mempoolId = at::cuda::graph_pool_handle();
					c10::cuda::CUDACachingAllocator::beginAllocateStreamToPool(
						captureDev, stream.stream(), mempoolId);
					mempoolHeld = true;
					C10_CUDA_CHECK(cudaStreamBeginCapture(
						stream.stream(), cudaStreamCaptureModeThreadLocal));
					captureStarted = true;
					cudaStreamCaptureStatus status;
					unsigned long long capId = 0;
					C10_CUDA_CHECK(cudaStreamGetCaptureInfo(stream.stream(), &status, &capId));
					auto capturedResult = InferPolicyProbsEager(
						models, staticObs, staticMasks, temperature, halfPrec,
						{}, /*deferFiniteCheck=*/true, {});
					staticOutput = capturedResult.probs;
					staticRowOk = capturedResult.rowOk;
					cudaGraph_t rawGraph = nullptr;
					C10_CUDA_CHECK(cudaStreamEndCapture(stream.stream(), &rawGraph));
					captureStarted = false;
					c10::cuda::CUDACachingAllocator::endAllocateStreamToPool(
						captureDev, stream.stream());
					C10_CUDA_CHECK(cudaGraphInstantiate(&graphExec, rawGraph, NULL, NULL, 0));
					C10_CUDA_CHECK(cudaGraphDestroy(rawGraph));
					captured = true;
					g_graphStats.captures++;
				} else {
					staticObs.copy_(obs, /*non_blocking=*/true);
					staticMasks.copy_(actionMasks, /*non_blocking=*/true);
				}
				C10_CUDA_CHECK(cudaGraphLaunch(graphExec, stream.stream()));
				int driverVersion = 0;
				C10_CUDA_CHECK(cudaDriverGetVersion(&driverVersion));
				if (driverVersion < 11040)
					C10_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
				graphDone.record(stream);
				graphDone.block(callerStream);
				g_graphStats.replays++;
				return { staticOutput, staticRowOk };
			} catch (const c10::OutOfMemoryError&) {
				endFailedCapture();
				g_graphStats.captureFailures++;
				throw;
			} catch (const c10::CUDAError& error) {
				endFailedCapture();
				g_graphStats.captureFailures++;
				// LibTorch 2.1 exposes CUDAError without a numeric CUDA error accessor.
				// Let the outer path release every graph pool, disable future captures,
				// and retry through the lower-memory eager path.
				if (IsMemoryFailure(error))
					throw;
				return {};
			} catch (const std::exception&) {
				endFailedCapture();
				g_graphStats.captureFailures++;
				return {};
			}
		}
	};

	struct PolicyGraphCache {
		std::mutex mutex;
		std::map<GraphKey, std::unique_ptr<PolicyProbsGraph>> entries;
	};

	// NOTE: Intentionally leaked so Model destructors can evict safely during static teardown.
	PolicyGraphCache& GetPolicyGraphCache() {
		static PolicyGraphCache* cache = new PolicyGraphCache();
		return *cache;
	}

	bool GraphsDisabledAfterOom() {
		auto& cache = GetPolicyGraphCache();
		std::lock_guard<std::mutex> lock(cache.mutex);
		return g_graphsDisabledAfterOom;
	}

	void DisableGraphsAfterOom() {
		auto& cache = GetPolicyGraphCache();
		std::lock_guard<std::mutex> lock(cache.mutex);
		cache.entries.clear();
		g_graphsDisabledAfterOom = true;
		g_graphStats.cacheEntries = 0;
		g_graphStats.disabledAfterOom = true;
	}

	bool CanUseGraph(
		GGL::ModelSet& models,
		const torch::Tensor& obs,
		const torch::Tensor& actionMasks,
		const torch::Tensor& steerDelta,
		const torch::Tensor& precomputedTrunk,
		const torch::Tensor* outRowOk) {

		return models["policy"]
			&& obs.defined()
			&& obs.is_cuda()
			&& obs.dim() == 2
			&& obs.size(0) > 0
			&& actionMasks.defined()
			&& actionMasks.is_cuda()
			&& actionMasks.device() == obs.device()
			&& actionMasks.dim() == 2
			&& actionMasks.size(0) == obs.size(0)
			&& !steerDelta.defined()
			&& !precomputedTrunk.defined()
			&& outRowOk
			&& !torch::GradMode::is_enabled();
	}

	PolicyForwardResult TryGraph(
		GGL::ModelSet& models,
		torch::Tensor obs,
		torch::Tensor actionMasks,
		float temperature,
		bool halfPrec) {

		if (halfPrec) {
			if (models["shared_head"])
				models["shared_head"]->RefreshHalfCache();
			models["policy"]->RefreshHalfCache();
		}

		GraphKey key = {};
		key.sharedHeadModel = models["shared_head"];
		key.policyModel = models["policy"];
		key.batchSize = obs.size(0);
		key.obsSize = (int)obs.size(1);
		key.numActions = (int)actionMasks.size(1);
		key.deviceIndex = obs.device().index();
		key.obsType = (int)obs.scalar_type();
		key.maskType = (int)actionMasks.scalar_type();
		key.inferenceType = halfPrec
			? (int)models["policy"]->seqHalf->parameters().front().scalar_type()
			: (int)torch::kFloat32;
		key.halfPrec = halfPrec;
		key.temperature = temperature;

		auto& cache = GetPolicyGraphCache();
		std::lock_guard<std::mutex> lock(cache.mutex);
		auto graphEntry = cache.entries.find(key);
		if (graphEntry == cache.entries.end()) {
			const size_t variants = std::count_if(cache.entries.begin(), cache.entries.end(), [&key](const auto& entry) {
				return entry.first.policyModel == key.policyModel;
			});
			if (variants >= MAX_GRAPH_VARIANTS_PER_POLICY) {
				g_graphStats.variantLimitFallbacks++;
				return {};
			}
			graphEntry = cache.entries.emplace(key, std::make_unique<PolicyProbsGraph>()).first;
		}

		PolicyForwardResult result;
		try {
			result = graphEntry->second->Run(models, obs, actionMasks, temperature, halfPrec);
		} catch (...) {
			cache.entries.erase(graphEntry);
			g_graphStats.cacheEntries = cache.entries.size();
			throw;
		}
		if (!result.probs.defined())
			cache.entries.erase(graphEntry);
		g_graphStats.cacheEntries = cache.entries.size();
		return result;
	}
#endif
}

torch::Tensor GGL::PolicyCudaGraph::InferPolicyProbs(
	ModelSet& models,
	torch::Tensor obs,
	torch::Tensor actionMasks,
	float temperature,
	bool halfPrec,
	torch::Tensor steerDelta,
	torch::Tensor* outRowOk,
	torch::Tensor precomputedTrunk,
	bool useCudaGraph) {

#ifdef RG_CUDA_SUPPORT
	if (useCudaGraph && !GraphsDisabledAfterOom()
		&& CanUseGraph(models, obs, actionMasks, steerDelta, precomputedTrunk, outRowOk)) {
		try {
			auto graphResult = TryGraph(models, obs, actionMasks, temperature, halfPrec);
			if (graphResult.probs.defined()) {
				*outRowOk = graphResult.rowOk;
				return graphResult.probs;
			}
		} catch (const c10::OutOfMemoryError&) {
			DisableGraphsAfterOom();
		} catch (const c10::CUDAError& error) {
			if (!IsMemoryFailure(error))
				throw;
			DisableGraphsAfterOom();
		}
	}
	if (useCudaGraph) {
		auto& cache = GetPolicyGraphCache();
		std::lock_guard<std::mutex> lock(cache.mutex);
		g_graphStats.fallbacks++;
	}
#else
	if (useCudaGraph)
		g_graphStats.fallbacks++;
#endif

	auto result = InferPolicyProbsEager(
		models, obs, actionMasks, temperature, halfPrec,
		steerDelta, outRowOk != nullptr, precomputedTrunk);
	if (outRowOk)
		*outRowOk = result.rowOk;
	return result.probs;
}

void GGL::PolicyCudaGraph::Release(const Model* model) {
#ifdef RG_CUDA_SUPPORT
	auto& cache = GetPolicyGraphCache();
	std::lock_guard<std::mutex> lock(cache.mutex);
	std::erase_if(cache.entries, [model](const auto& entry) {
		return entry.first.sharedHeadModel == model || entry.first.policyModel == model;
	});
	g_graphStats.cacheEntries = cache.entries.size();
#else
	(void)model;
#endif
}

GGL::PolicyCudaGraphStats GGL::PolicyCudaGraph::GetStats() {
#ifdef RG_CUDA_SUPPORT
	auto& cache = GetPolicyGraphCache();
	std::lock_guard<std::mutex> lock(cache.mutex);
	g_graphStats.cacheEntries = cache.entries.size();
	g_graphStats.disabledAfterOom = g_graphsDisabledAfterOom;
#endif
	return g_graphStats;
}

void GGL::PolicyCudaGraph::ResetStats() {
#ifdef RG_CUDA_SUPPORT
	auto& cache = GetPolicyGraphCache();
	std::lock_guard<std::mutex> lock(cache.mutex);
	const uint64_t entries = cache.entries.size();
	const bool disabledAfterOom = g_graphsDisabledAfterOom;
	g_graphStats = {};
	g_graphStats.cacheEntries = entries;
	g_graphStats.disabledAfterOom = disabledAfterOom;
#else
	g_graphStats = {};
#endif
}

void GGL::PolicyCudaGraph::Clear() {
#ifdef RG_CUDA_SUPPORT
	auto& cache = GetPolicyGraphCache();
	std::lock_guard<std::mutex> lock(cache.mutex);
	cache.entries.clear();
	g_graphsDisabledAfterOom = false;
	g_graphStats.cacheEntries = 0;
	g_graphStats.disabledAfterOom = false;
#else
	g_graphStats.cacheEntries = 0;
#endif
}

GGL::PolicyCudaGraphVerification GGL::PolicyCudaGraph::RunVerification() {
	PolicyCudaGraphVerification verification = {};
#ifdef RG_CUDA_SUPPORT
	if (!torch::cuda::is_available())
		return verification;
	verification.cudaAvailable = true;

	Clear();
	ResetStats();
	torch::manual_seed(12345);
	torch::NoGradGuard noGrad;
	const torch::Device device(torch::kCUDA, c10::cuda::current_device());

	PartialModelConfig sharedPartial = {};
	sharedPartial.layerSizes = { 32, 32, 32 };
	sharedPartial.addOutputLayer = false;
	sharedPartial.addResiduals = true;
	ModelConfig sharedConfig(sharedPartial);
	sharedConfig.numInputs = 24;
	sharedConfig.numOutputs = 0;

	PartialModelConfig policyPartial = {};
	policyPartial.layerSizes = { 32, 32 };
	ModelConfig policyConfig(policyPartial);
	policyConfig.numInputs = 32;
	policyConfig.numOutputs = 10;

	ModelSet models;
	models.Add(new Model("shared_head", sharedConfig, device));
	models.Add(new Model("policy", policyConfig, device));

	auto obs = torch::randn({ 37, 24 }, torch::TensorOptions().device(device).dtype(torch::kFloat32));
	auto masks = torch::rand({ 37, 10 }, torch::TensorOptions().device(device)) > 0.2;
	masks.select(1, 0).fill_(true);

	torch::Tensor eagerOk, graphOk;
	auto eagerFp32 = InferPolicyProbs(models, obs, masks, 0.85f, false, {}, &eagerOk, {}, false);
	auto graphFp32 = InferPolicyProbs(models, obs, masks, 0.85f, false, {}, &graphOk, {}, true);
	verification.fp32MaxAbsError = (eagerFp32 - graphFp32).abs().max().item<float>();
	verification.actionAndLogProbMatched = torch::equal(eagerFp32.argmax(1), graphFp32.argmax(1));
	auto chosenActions = eagerFp32.argmax(1).unsqueeze(-1);
	verification.actionAndLogProbMatched = verification.actionAndLogProbMatched
		&& (eagerFp32.log().gather(-1, chosenActions)
			- graphFp32.log().gather(-1, chosenActions)).abs().max().item<float>() <= 1e-6f;

	auto obsReplay = torch::randn_like(obs);
	auto eagerReplay = InferPolicyProbs(models, obsReplay, masks, 0.85f, false, {}, &eagerOk, {}, false);
	auto graphReplay = InferPolicyProbs(models, obsReplay, masks, 0.85f, false, {}, &graphOk, {}, true);
	verification.fp32MaxAbsError = RS_MAX(
		verification.fp32MaxAbsError,
		(eagerReplay - graphReplay).abs().max().item<float>());
	auto eagerLiveTemperature = InferPolicyProbs(models, obs, masks, 1.f, false, {}, &eagerOk, {}, false);
	auto graphLiveTemperature = InferPolicyProbs(models, obs, masks, 1.f, false, {}, &graphOk, {}, true);
	verification.fp32MaxAbsError = RS_MAX(
		verification.fp32MaxAbsError,
		(eagerLiveTemperature - graphLiveTemperature).abs().max().item<float>());

	auto eagerHalf = InferPolicyProbs(models, obs, masks, 0.85f, true, {}, &eagerOk, {}, false);
	auto graphHalf = InferPolicyProbs(models, obs, masks, 0.85f, true, {}, &graphOk, {}, true);
	verification.halfMaxAbsError = (eagerHalf - graphHalf).abs().max().item<float>();
	auto oldHalf = graphHalf.clone();

	for (Model* model : models) {
		for (auto& parameter : model->parameters())
			parameter.add_(0.025f * torch::randn_like(parameter));
		model->_seqHalfOutdated = true;
	}
	auto refreshedGraph = InferPolicyProbs(models, obs, masks, 0.85f, true, {}, &graphOk, {}, true);
	auto refreshedEager = InferPolicyProbs(models, obs, masks, 0.85f, true, {}, &eagerOk, {}, false);
	verification.refreshedHalfMaxAbsError = (refreshedGraph - refreshedEager).abs().max().item<float>();
	verification.refreshedHalfChange = (refreshedGraph - oldHalf).abs().max().item<float>();

	// Force an FP16 mirror even on Ampere+ so the local test exercises the exact dtype
	// used by the cluster's V100 ranks. The graph key includes this actual mirror dtype.
	ModelSet fp16Models;
	fp16Models.Add(new Model("shared_head", sharedConfig, device));
	fp16Models.Add(new Model("policy", policyConfig, device));
	for (Model* model : fp16Models) {
		model->RefreshHalfCache();
		model->seqHalf->to(torch::kHalf, /*non_blocking=*/true);
		model->_seqHalfOutdated = false;
	}
	torch::Tensor eagerFp16Ok, graphFp16Ok;
	auto eagerFp16 = InferPolicyProbs(fp16Models, obs, masks, 0.85f, true, {}, &eagerFp16Ok, {}, false);
	auto graphFp16 = InferPolicyProbs(fp16Models, obs, masks, 0.85f, true, {}, &graphFp16Ok, {}, true);
	verification.fp16MaxAbsError = (eagerFp16 - graphFp16).abs().max().item<float>();
	auto oldFp16 = graphFp16.clone();
	for (Model* model : fp16Models) {
		for (auto& parameter : model->parameters())
			parameter.add_(0.025f * torch::randn_like(parameter));
		model->_seqHalfOutdated = true;
	}
	auto refreshedFp16Graph = InferPolicyProbs(fp16Models, obs, masks, 0.85f, true, {}, &graphFp16Ok, {}, true);
	auto refreshedFp16Eager = InferPolicyProbs(fp16Models, obs, masks, 0.85f, true, {}, &eagerFp16Ok, {}, false);
	verification.refreshedFp16MaxAbsError =
		(refreshedFp16Graph - refreshedFp16Eager).abs().max().item<float>();
	verification.refreshedFp16Change = (refreshedFp16Graph - oldFp16).abs().max().item<float>();

	auto badObs = obs.clone();
	badObs.select(0, 4).fill_(std::numeric_limits<float>::quiet_NaN());
	auto eagerFinite = InferPolicyProbs(models, badObs, masks, 0.85f, false, {}, &eagerOk, {}, false);
	auto graphFinite = InferPolicyProbs(models, badObs, masks, 0.85f, false, {}, &graphOk, {}, true);
	verification.finiteGuardMatched = !graphOk.select(0, 4).item<bool>()
		&& torch::equal(eagerOk, graphOk)
		&& graphFinite.isfinite().all().item<bool>()
		&& (eagerFinite - graphFinite).abs().max().item<float>() <= 1e-6f;

	// A precomputed trunk is used by the learn path and is intentionally ineligible for
	// collection graphs. A graph request must preserve the production eager result.
	auto precomputedTrunk = models["shared_head"]->Forward(obs, false);
	auto eagerPrecomputed = InferPolicyProbs(
		models, obs, masks, 1.f, false, {}, &eagerOk, precomputedTrunk, false);
	auto fallbackPrecomputed = InferPolicyProbs(
		models, obs, masks, 1.f, false, {}, &graphOk, precomputedTrunk, true);
	verification.eagerFallbackMatched = torch::equal(eagerOk, graphOk)
		&& (eagerPrecomputed - fallbackPrecomputed).abs().max().item<float>() <= 1e-6f;

	auto stats = GetStats();
	verification.captures = stats.captures;
	verification.replays = stats.replays;
	verification.fallbacks = stats.fallbacks;
	verification.disabledAfterOom = stats.disabledAfterOom;

	// Drop graph-owned output aliases before the model destructors evict their captures.
	eagerFp32 = torch::Tensor(); graphFp32 = torch::Tensor();
	eagerReplay = torch::Tensor(); graphReplay = torch::Tensor();
	eagerLiveTemperature = torch::Tensor(); graphLiveTemperature = torch::Tensor();
	eagerHalf = torch::Tensor(); graphHalf = torch::Tensor(); oldHalf = torch::Tensor();
	refreshedGraph = torch::Tensor(); refreshedEager = torch::Tensor();
	eagerFinite = torch::Tensor(); graphFinite = torch::Tensor();
	precomputedTrunk = torch::Tensor(); eagerPrecomputed = torch::Tensor();
	fallbackPrecomputed = torch::Tensor();
	eagerFp16 = torch::Tensor(); graphFp16 = torch::Tensor(); oldFp16 = torch::Tensor();
	refreshedFp16Graph = torch::Tensor(); refreshedFp16Eager = torch::Tensor();
	models.Free();
	fp16Models.Free();
	verification.cacheEvicted = GetStats().cacheEntries == 0;
	Clear();
#endif
	return verification;
}
