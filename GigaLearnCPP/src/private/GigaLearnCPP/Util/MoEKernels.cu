// GGL_MOE_KERNELS — see MoEKernels.h and research/reports/MOE_SPEED.md.
// Grouped-GEMM template configuration lifted from the proven moe-bench
// (~/scratch-shared/moe-bench/src/moe_cutlass.cu on AiMOS), adapted from
// host-side problem fill to DEVICE-side fill so no routing count ever crosses
// to the host (a D2H sync would re-pay the dispatch tax this file removes).
// Compiles with nvcc 11.2 / sm_70 exactly like the bench does.

#include "MoEKernels.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifndef CUTLASS_ENABLE_F16C
#define CUTLASS_ENABLE_F16C 0
#endif

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

#include <cstdio>
#include <cstdlib>

#define GGLK_CUDA_CHECK(stmt)                                                  \
	do {                                                                       \
		cudaError_t err__ = (stmt);                                            \
		if (err__ != cudaSuccess) {                                            \
			fprintf(stderr, "MoEKernels CUDA error %s:%d: %s\n", __FILE__,     \
				__LINE__, cudaGetErrorString(err__));                          \
			abort();                                                           \
		}                                                                      \
	} while (0)

#define GGLK_CUTLASS_CHECK(status)                                             \
	do {                                                                       \
		cutlass::Status st__ = (status);                                       \
		if (st__ != cutlass::Status::kSuccess) {                               \
			fprintf(stderr, "MoEKernels CUTLASS error %s:%d: %s\n", __FILE__,  \
				__LINE__, cutlassGetStatusString(st__));                       \
			abort();                                                           \
		}                                                                      \
	} while (0)

// ============================== routing plan ==============================

// One thread per row: fp32 sigmoid over E fp16 logits, top-k selection by
// (affinity + selection bias) with gates from the RAW affinity (DSv3: the bias
// steers selection only, never the mixture weights), renormalized. Per-row
// results land in rowExp/rowGate; counts accumulate for the scan.
// E is small (128 live); a k*E scan per thread is cheap next to one dispatch.
__global__ void gglk_route_topk(
	const float* logits, const __half* selBias,
	int R, int E, int k,
	int* counts, int* rowExp, float* rowGate) {

	int r = blockIdx.x * blockDim.x + threadIdx.x;
	if (r >= R)
		return;
	const float* row = logits + (size_t)r * E;
	bool taken[512]; // E <= 512 enforced host-side
	for (int e = 0; e < E; e++)
		taken[e] = false;
	float gateSum = 0.f;
	for (int j = 0; j < k; j++) {
		int best = -1;
		float bestScore = -1e30f;
		for (int e = 0; e < E; e++) {
			if (taken[e])
				continue;
			float aff = 1.f / (1.f + __expf(-row[e]));
			float score = aff + __half2float(selBias[e]);
			if (score > bestScore) {
				bestScore = score;
				best = e;
			}
		}
		taken[best] = true;
		float aff = 1.f / (1.f + __expf(-row[best]));
		rowExp[(size_t)r * k + j] = best;
		rowGate[(size_t)r * k + j] = aff;
		gateSum += aff;
		atomicAdd(&counts[best], 1);
	}
	float inv = 1.f / fmaxf(gateSum, 1e-9f);
	for (int j = 0; j < k; j++)
		rowGate[(size_t)r * k + j] *= inv;
}

// Single-block exclusive scan counts -> offsets (E <= 512). alignUp rounds each
// expert's segment to 8 slots (learn-path wgrad alignment; see header).
__global__ void gglk_scan_offsets(const int* counts, int* offsets, int E, int alignUp) {
	__shared__ int sh[513];
	int t = threadIdx.x;
	if (t <= E)
		sh[t] = (t < E) ? counts[t] : 0;
	__syncthreads();
	if (t == 0) {
		int acc = 0;
		for (int e = 0; e <= E; e++) {
			int c = sh[e];
			if (alignUp)
				c = (c + 7) & ~7;
			offsets[e] = acc;
			acc += c;
		}
	}
}

// One thread per assignment: claim a slot in expert-sorted order via a
// per-expert cursor. Order within an expert is arbitrary (atomics), which is
// fine — the combine is a commutative sum.
__global__ void gglk_place(
	const int* rowExp, const float* rowGate, const int* offsets,
	int* cursors, int n, int k,
	int* srcRows, int* expertId, float* gates) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)
		return;
	int e = rowExp[i];
	int slot = offsets[e] + atomicAdd(&cursors[e], 1);
	srcRows[slot] = i / k;
	expertId[slot] = e;
	gates[slot] = rowGate[i];
}

// Fixed-capacity placement: slot = e*cap + cursor, overflow -> trash row E*cap.
__global__ void gglk_place_fixed(
	const int* rowExp, const float* rowGate,
	int* cursors, int n, int k, int E, int cap,
	int* srcRows, int* expertId, float* gates) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)
		return;
	int e = rowExp[i];
	int c = atomicAdd(&cursors[e], 1);
	if (c >= cap)
		return;                       // dropped: its slot keeps srcRows = -1
	int slot = e * cap + c;
	srcRows[slot] = i / k;
	expertId[slot] = e;
	gates[slot] = rowGate[i];
}

// offsets[e] = e*cap (device-side so the caller never needs a host copy).
__global__ void gglk_fixed_offsets(int* offsets, int E, int cap) {
	int e = blockIdx.x * blockDim.x + threadIdx.x;
	if (e <= E)
		offsets[e] = e * cap;
}

extern "C" void ggl_moe_route_plan_fixed_f16(
	const void* logits, const void* selBias,
	int R, int E, int k, int fixedCap,
	int* countsCursors, int* offsets,
	int* rowExpScratch, float* rowGateScratch,
	int* srcRows, int* expertId, float* gates,
	void* stream) {

	if (E > 512) {
		fprintf(stderr, "MoEKernels: E=%d exceeds the 512 routing limit\n", E);
		abort();
	}
	cudaStream_t s = (cudaStream_t)stream;
	const int n = R * k;
	const int nSlots = E * fixedCap + 1;
	int* counts = countsCursors;
	int* cursors = countsCursors + E;
	GGLK_CUDA_CHECK(cudaMemsetAsync(countsCursors, 0, sizeof(int) * 2 * (size_t)E, s));
	// Unfilled + dropped slots must read as skippable everywhere.
	GGLK_CUDA_CHECK(cudaMemsetAsync(srcRows, 0xFF, sizeof(int) * (size_t)nSlots, s));
	GGLK_CUDA_CHECK(cudaMemsetAsync(gates, 0, sizeof(float) * (size_t)nSlots, s));
	GGLK_CUDA_CHECK(cudaMemsetAsync(expertId, 0, sizeof(int) * (size_t)nSlots, s));
	gglk_route_topk<<<(R + 127) / 128, 128, 0, s>>>(
		(const float*)logits, (const __half*)selBias, R, E, k,
		counts, rowExpScratch, rowGateScratch);
	gglk_fixed_offsets<<<(E + 64) / 64, 64, 0, s>>>(offsets, E, fixedCap);
	gglk_place_fixed<<<(n + 127) / 128, 128, 0, s>>>(
		rowExpScratch, rowGateScratch, cursors, n, k, E, fixedCap,
		srcRows, expertId, gates);
}

extern "C" void ggl_moe_route_plan_f16(
	const void* logits, const void* selBias,
	int R, int E, int k, int alignSegments,
	int* countsCursors, int* offsets,
	int* rowExpScratch, float* rowGateScratch,
	int* srcRows, int* expertId, float* gates,
	void* stream) {

	if (E > 512) {
		fprintf(stderr, "MoEKernels: E=%d exceeds the 512 routing limit\n", E);
		abort();
	}
	cudaStream_t s = (cudaStream_t)stream;
	int n = R * k;
	int* counts = countsCursors;
	int* cursors = countsCursors + E;
	GGLK_CUDA_CHECK(cudaMemsetAsync(countsCursors, 0, sizeof(int) * 2 * (size_t)E, s));
	if (alignSegments) {
		// Pad slots must pre-exist as skippable: srcRows = -1 (0xFF bytes), gates = 0.
		int nBound = ((n + 7) & ~7) + 8 * E;
		GGLK_CUDA_CHECK(cudaMemsetAsync(srcRows, 0xFF, sizeof(int) * (size_t)nBound, s));
		GGLK_CUDA_CHECK(cudaMemsetAsync(gates, 0, sizeof(float) * (size_t)nBound, s));
		GGLK_CUDA_CHECK(cudaMemsetAsync(expertId, 0, sizeof(int) * (size_t)nBound, s));
	}
	gglk_route_topk<<<(R + 127) / 128, 128, 0, s>>>(
		(const float*)logits, (const __half*)selBias, R, E, k,
		counts, rowExpScratch, rowGateScratch);
	gglk_scan_offsets<<<1, E + 1, 0, s>>>(counts, offsets, E, alignSegments);
	gglk_place<<<(n + 127) / 128, 128, 0, s>>>(
		rowExpScratch, rowGateScratch, offsets, cursors, n, k,
		srcRows, expertId, gates);
}

// ============================ gather / epilogues ============================

__global__ void gglk_gather_f16(
	const __half* x, const int* srcRows, __half* packed, int n, int d) {
	int i = blockIdx.x;
	if (i >= n)
		return;
	int sr = srcRows[i];
	__half* dst = packed + (size_t)i * d;
	if (sr < 0) { // aligned-segment pad slot: zero row (wgrad sees exact zeros)
		for (int c = threadIdx.x; c < d; c += blockDim.x)
			dst[c] = __float2half(0.f);
		return;
	}
	const __half* src = x + (size_t)sr * d;
	for (int c = threadIdx.x; c < d; c += blockDim.x)
		dst[c] = src[c];
}

extern "C" void ggl_moe_gather_f16(
	const void* x, const int* srcRows, void* packed, int n, int d, void* stream) {
	if (n <= 0)
		return;
	gglk_gather_f16<<<n, 128, 0, (cudaStream_t)stream>>>(
		(const __half*)x, srcRows, (__half*)packed, n, d);
}

__global__ void gglk_bias_leaky_f16(
	__half* h, const __half* b, const int* expertId, int n, int H, float slope) {
	int i = blockIdx.x;
	if (i >= n)
		return;
	const __half* be = b + (size_t)expertId[i] * H;
	__half* row = h + (size_t)i * H;
	for (int c = threadIdx.x; c < H; c += blockDim.x) {
		float v = __half2float(row[c]) + __half2float(be[c]);
		row[c] = __float2half(v > 0.f ? v : v * slope);
	}
}

extern "C" void ggl_moe_bias_leaky_f16(
	void* h, const void* b, const int* expertId, int n, int H, float slope, void* stream) {
	if (n <= 0)
		return;
	gglk_bias_leaky_f16<<<n, 128, 0, (cudaStream_t)stream>>>(
		(__half*)h, (const __half*)b, expertId, n, H, slope);
}

extern "C" void ggl_moe_zero_f16(void* p, int64_t count, void* stream) {
	GGLK_CUDA_CHECK(cudaMemsetAsync(p, 0, sizeof(__half) * (size_t)count,
		(cudaStream_t)stream));
}

__global__ void gglk_scatter_bias_gate_f16(
	const __half* y, const __half* b, const int* expertId,
	const float* gates, const int* srcRows, __half* out, int n, int D) {
	int i = blockIdx.x;
	if (i >= n)
		return;
	int sr = srcRows[i];
	if (sr < 0)
		return;
	const __half* ye = y + (size_t)i * D;
	const __half* be = b + (size_t)expertId[i] * D;
	__half* dst = out + (size_t)sr * D;
	float g = gates[i];
	for (int c = threadIdx.x; c < D; c += blockDim.x) {
		float v = (__half2float(ye[c]) + __half2float(be[c])) * g;
		atomicAdd(&dst[c], __float2half(v)); // fp16 atomicAdd: sm_70+
	}
}

extern "C" void ggl_moe_scatter_bias_gate_f16(
	const void* y, const void* b, const int* expertId,
	const float* gates, const int* srcRows, void* out,
	int n, int D, void* stream) {
	if (n <= 0)
		return;
	gglk_scatter_bias_gate_f16<<<n, 128, 0, (cudaStream_t)stream>>>(
		(const __half*)y, (const __half*)b, expertId, gates, srcRows,
		(__half*)out, n, D);
}

// ===================== learn-path elementwise kernels =====================

__global__ void gglk_gather_scale_f16(
	const __half* dOut, const int* srcRows, const float* gates,
	__half* dY, int n, int d) {
	int i = blockIdx.x;
	if (i >= n)
		return;
	int sr = srcRows[i];
	__half* dst = dY + (size_t)i * d;
	if (sr < 0) { // pad slot: exact zeros into every downstream product
		for (int c = threadIdx.x; c < d; c += blockDim.x)
			dst[c] = __float2half(0.f);
		return;
	}
	const __half* src = dOut + (size_t)sr * d;
	float g = gates[i];
	for (int c = threadIdx.x; c < d; c += blockDim.x)
		dst[c] = __float2half(__half2float(src[c]) * g);
}

extern "C" void ggl_moe_gather_scale_f16(
	const void* dOut, const int* srcRows, const float* gates,
	void* dY, int n, int d, void* stream) {
	if (n <= 0) return;
	gglk_gather_scale_f16<<<n, 128, 0, (cudaStream_t)stream>>>(
		(const __half*)dOut, srcRows, gates, (__half*)dY, n, d);
}

__global__ void gglk_segment_sum_f16to32(
	const __half* src, const int* expertId, float* dB, int n, int d) {
	int i = blockIdx.x;
	if (i >= n)
		return;
	const __half* row = src + (size_t)i * d;
	float* dst = dB + (size_t)expertId[i] * d;
	for (int c = threadIdx.x; c < d; c += blockDim.x)
		atomicAdd(&dst[c], __half2float(row[c]));
}

extern "C" void ggl_moe_segment_sum_f16to32(
	const void* src, const int* expertId, float* dB, int n, int d, void* stream) {
	if (n <= 0) return;
	gglk_segment_sum_f16to32<<<n, 128, 0, (cudaStream_t)stream>>>(
		(const __half*)src, expertId, dB, n, d);
}

__global__ void gglk_leaky_bwd_f16(
	__half* dH, const __half* hPost, int n, int H, float slope) {
	int i = blockIdx.x;
	if (i >= n)
		return;
	__half* row = dH + (size_t)i * H;
	const __half* hp = hPost + (size_t)i * H;
	for (int c = threadIdx.x; c < H; c += blockDim.x) {
		float v = __half2float(row[c]);
		// leaky is sign-preserving, so post-activation sign == pre-activation sign
		if (!(__half2float(hp[c]) > 0.f))
			v *= slope;
		row[c] = __float2half(v);
	}
}

extern "C" void ggl_moe_leaky_bwd_f16(
	void* dH, const void* hPost, int n, int H, float slope, void* stream) {
	if (n <= 0) return;
	gglk_leaky_bwd_f16<<<n, 128, 0, (cudaStream_t)stream>>>(
		(__half*)dH, (const __half*)hPost, n, H, slope);
}

__global__ void gglk_gate_dot_f16(
	const __half* dOut, const __half* y, const __half* b,
	const int* expertId, const int* srcRows, float* gdot, int n, int d) {
	int i = blockIdx.x;
	if (i >= n)
		return;
	if (srcRows[i] < 0) {
		if (threadIdx.x == 0)
			gdot[i] = 0.f;
		return;
	}
	const __half* go = dOut + (size_t)srcRows[i] * d;
	const __half* yr = y + (size_t)i * d;
	const __half* br = b + (size_t)expertId[i] * d;
	__shared__ float acc[128];
	float part = 0.f;
	for (int c = threadIdx.x; c < d; c += blockDim.x)
		part += __half2float(go[c]) * (__half2float(yr[c]) + __half2float(br[c]));
	acc[threadIdx.x] = part;
	__syncthreads();
	for (int w = 64; w > 0; w >>= 1) {
		if (threadIdx.x < w)
			acc[threadIdx.x] += acc[threadIdx.x + w];
		__syncthreads();
	}
	if (threadIdx.x == 0)
		gdot[i] = acc[0];
}

extern "C" void ggl_moe_gate_dot_f16(
	const void* dOut, const void* y, const void* b,
	const int* expertId, const int* srcRows,
	float* gdot, int n, int d, void* stream) {
	if (n <= 0) return;
	gglk_gate_dot_f16<<<n, 128, 0, (cudaStream_t)stream>>>(
		(const __half*)dOut, (const __half*)y, (const __half*)b,
		expertId, srcRows, gdot, n, d);
}

__global__ void gglk_scatter_add_f16(
	const __half* dX, const int* srcRows, __half* dxn, int n, int d) {
	int i = blockIdx.x;
	if (i >= n)
		return;
	int sr = srcRows[i];
	if (sr < 0)
		return;
	const __half* row = dX + (size_t)i * d;
	__half* dst = dxn + (size_t)sr * d;
	for (int c = threadIdx.x; c < d; c += blockDim.x)
		atomicAdd(&dst[c], row[c]);
}

extern "C" void ggl_moe_scatter_add_f16(
	const void* dX, const int* srcRows, void* dxn, int n, int d, void* stream) {
	if (n <= 0) return;
	gglk_scatter_add_f16<<<n, 128, 0, (cudaStream_t)stream>>>(
		(const __half*)dX, srcRows, (__half*)dxn, n, d);
}

// ========================= CUTLASS grouped GEMM ==========================
// Bench-proven sm_70 configuration: fp16 tensor-core 8x8x4, small-M tile
// 32x128x32 (collect batches average m = R*k/E, tens of rows — the small tile
// is permanently correct here; the bench's large-tile variant is for m >= ~256).
// kDeviceOnly schedule: the persistent kernel walks d_problems on device, so a
// zero-count expert contributes zero tiles and the launch config is static.

using LayoutRM = cutlass::layout::RowMajor;
using Swizzle = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;
constexpr auto kSched = cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly;
using EpiF16 = cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 8, float, float>;

using GglkGemmF16 = cutlass::gemm::device::GemmGrouped<
	typename cutlass::gemm::kernel::DefaultGemmGrouped<
		cutlass::half_t, LayoutRM, cutlass::ComplexTransform::kNone, 8, cutlass::half_t,
		LayoutRM, cutlass::ComplexTransform::kNone, 8, cutlass::half_t, LayoutRM, float,
		cutlass::arch::OpClassTensorOp, cutlass::arch::Sm70,
		cutlass::gemm::GemmShape<32, 128, 32>, cutlass::gemm::GemmShape<32, 32, 32>,
		cutlass::gemm::GemmShape<8, 8, 4>, EpiF16, Swizzle, 2, kSched>::GemmKernel>;

using GglkLongIndex = typename LayoutRM::Stride::LongIndex;

namespace {
	struct GemmCtx {
		int E = 0;
		cutlass::gemm::GemmCoord* d_problems = nullptr;
		cutlass::half_t** d_ptr_A = nullptr;
		cutlass::half_t** d_ptr_B = nullptr;
		cutlass::half_t** d_ptr_C = nullptr;
		cutlass::half_t** d_ptr_D = nullptr;
		GglkLongIndex* d_ld = nullptr;   // [4E]: lda, ldb, ldc, ldd
		void* d_workspace = nullptr;
		size_t ws_cap = 0;
		int tbc = 0;
	};
}

extern "C" void* ggl_moe_ctx_create(int maxExperts) {
	GemmCtx* c = new GemmCtx();
	c->E = maxExperts;
	GGLK_CUDA_CHECK(cudaMalloc(&c->d_problems, sizeof(cutlass::gemm::GemmCoord) * (size_t)maxExperts));
	GGLK_CUDA_CHECK(cudaMalloc(&c->d_ptr_A, sizeof(void*) * (size_t)maxExperts));
	GGLK_CUDA_CHECK(cudaMalloc(&c->d_ptr_B, sizeof(void*) * (size_t)maxExperts));
	GGLK_CUDA_CHECK(cudaMalloc(&c->d_ptr_C, sizeof(void*) * (size_t)maxExperts));
	GGLK_CUDA_CHECK(cudaMalloc(&c->d_ptr_D, sizeof(void*) * (size_t)maxExperts));
	GGLK_CUDA_CHECK(cudaMalloc(&c->d_ld, sizeof(GglkLongIndex) * 4 * (size_t)maxExperts));
	// Static persistent-kernel threadblock count (no per-problem host info in
	// device-only mode; the bench uses the same fallback).
	c->tbc = GglkGemmF16::sufficient(nullptr, 0);
	if (c->tbc <= 0) {
		fprintf(stderr, "MoEKernels: GemmGrouped::sufficient() returned %d\n", c->tbc);
		abort();
	}
	return c;
}

// Fill problem descriptors + pointer tables from device offsets. One thread
// per expert.
__global__ void gglk_fill_problems_mod(
	const int* offsets, int E, int weightMod, int K, int N,
	const cutlass::half_t* A, const cutlass::half_t* B, cutlass::half_t* C,
	cutlass::gemm::GemmCoord* problems,
	cutlass::half_t** pA, cutlass::half_t** pB, cutlass::half_t** pC, cutlass::half_t** pD,
	GglkLongIndex* ld) {

	int e = blockIdx.x * blockDim.x + threadIdx.x;
	if (e >= E)
		return;
	int m = offsets[e + 1] - offsets[e];
	problems[e] = cutlass::gemm::GemmCoord(m, N, K);
	pA[e] = const_cast<cutlass::half_t*>(A) + (size_t)offsets[e] * K;
	// weightMod: EP problem lists are (rank x ownedExpert), so many problems share
	// one weight matrix — index it modulo the owned count instead of repeating it.
	int w = (weightMod > 0) ? (e % weightMod) : e;
	pB[e] = const_cast<cutlass::half_t*>(B) + (size_t)w * K * N;
	pC[e] = C + (size_t)offsets[e] * N;
	pD[e] = pC[e];
	ld[e] = K;
	ld[E + e] = N;
	ld[2 * E + e] = N;
	ld[3 * E + e] = N;
}

extern "C" void ggl_moe_grouped_gemm_f16_dev(
	void* ctx, const void* A, const void* B, void* C,
	const int* offsets, int E, int K, int N, void* stream) {

	GemmCtx* c = (GemmCtx*)ctx;
	cudaStream_t s = (cudaStream_t)stream;
	if (E > c->E) {
		fprintf(stderr, "MoEKernels: E=%d exceeds ctx capacity %d\n", E, c->E);
		abort();
	}
	gglk_fill_problems_mod<<<(E + 63) / 64, 64, 0, s>>>(
		offsets, E, /*weightMod=*/0, K, N,
		(const cutlass::half_t*)A, (const cutlass::half_t*)B, (cutlass::half_t*)C,
		c->d_problems, c->d_ptr_A, c->d_ptr_B, c->d_ptr_C, c->d_ptr_D, c->d_ld);

	typename GglkGemmF16::EpilogueOutputOp::Params epilogue(1.f, 0.f);
	typename GglkGemmF16::Arguments args(
		c->d_problems, E, c->tbc, epilogue,
		c->d_ptr_A, c->d_ptr_B, c->d_ptr_C, c->d_ptr_D,
		c->d_ld, c->d_ld + E, c->d_ld + 2 * E, c->d_ld + 3 * E,
		/*host_problem_sizes=*/nullptr);

	size_t ws = GglkGemmF16::get_workspace_size(args);
	size_t need = ws + ((size_t)1 << 20);
	if (need > c->ws_cap) {
		// grow-only: never free the old workspace (V100/11.2 sticky-error trap)
		GGLK_CUDA_CHECK(cudaMalloc(&c->d_workspace, need));
		c->ws_cap = need;
	}
	if (ws > 0)
		GGLK_CUDA_CHECK(cudaMemsetAsync(c->d_workspace, 0, ws, s));

	GglkGemmF16 gemm;
	GGLK_CUTLASS_CHECK(gemm.initialize(args, c->d_workspace, s));
	GGLK_CUTLASS_CHECK(gemm.run(s));
}

extern "C" void ggl_moe_grouped_gemm_f16_dev_mod(
	void* ctx, const void* A, const void* B, void* C,
	const int* offsets, int nProb, int weightMod, int K, int N, void* stream) {

	GemmCtx* c = (GemmCtx*)ctx;
	cudaStream_t s = (cudaStream_t)stream;
	if (nProb > c->E) {
		fprintf(stderr, "MoEKernels: nProb=%d exceeds ctx capacity %d\n", nProb, c->E);
		abort();
	}
	gglk_fill_problems_mod<<<(nProb + 63) / 64, 64, 0, s>>>(
		offsets, nProb, weightMod, K, N,
		(const cutlass::half_t*)A, (const cutlass::half_t*)B, (cutlass::half_t*)C,
		c->d_problems, c->d_ptr_A, c->d_ptr_B, c->d_ptr_C, c->d_ptr_D, c->d_ld);

	typename GglkGemmF16::EpilogueOutputOp::Params epilogue(1.f, 0.f);
	typename GglkGemmF16::Arguments args(
		c->d_problems, nProb, c->tbc, epilogue,
		c->d_ptr_A, c->d_ptr_B, c->d_ptr_C, c->d_ptr_D,
		c->d_ld, c->d_ld + nProb, c->d_ld + 2 * nProb, c->d_ld + 3 * nProb,
		/*host_problem_sizes=*/nullptr);

	size_t ws = GglkGemmF16::get_workspace_size(args);
	size_t need = ws + ((size_t)1 << 20);
	if (need > c->ws_cap) {
		// grow-only: never free the old workspace (V100/11.2 sticky-error trap)
		GGLK_CUDA_CHECK(cudaMalloc(&c->d_workspace, need));
		c->ws_cap = need;
	}
	if (ws > 0)
		GGLK_CUDA_CHECK(cudaMemsetAsync(c->d_workspace, 0, ws, s));

	GglkGemmF16 gemm;
	GGLK_CUTLASS_CHECK(gemm.initialize(args, c->d_workspace, s));
	GGLK_CUTLASS_CHECK(gemm.run(s));
}

// ======================== grouped wgrad (via transpose) ========================
// dW[e] = A_e^T @ B_e. A ColumnMajor has no legal sm_70 thread map at our tile
// shape (static asserts in pitch_linear_thread_map), so instead ONE global
// transpose T = A^T (row-major [Ka, n]) makes every expert's A^T a plain
// row-major slice: ptr = T + off_e, lda = n. Reuses the proven RowMajor GEMM.

__global__ void gglk_transpose_f16(const __half* src, __half* dst, int n, int K) {
	// src [n, K] row-major -> dst [K, n] row-major; 32x32 shared tile.
	__shared__ __half tile[32][33];
	int c0 = blockIdx.x * 32, r0 = blockIdx.y * 32;
	int c = c0 + threadIdx.x, r = r0 + threadIdx.y;
	if (r < n && c < K)
		tile[threadIdx.y][threadIdx.x] = src[(size_t)r * K + c];
	__syncthreads();
	int tc = r0 + threadIdx.x, tr = c0 + threadIdx.y;   // transposed coords
	if (tr < K && tc < n)
		dst[(size_t)tr * n + tc] = tile[threadIdx.x][threadIdx.y];
}

__global__ void gglk_fill_wgrad_problems(
	const int* offsets, int E, int Ka, int N, int n,
	const cutlass::half_t* T, const cutlass::half_t* B, cutlass::half_t* C,
	cutlass::gemm::GemmCoord* problems,
	cutlass::half_t** pA, cutlass::half_t** pB, cutlass::half_t** pC, cutlass::half_t** pD,
	GglkLongIndex* ld) {

	int e = blockIdx.x * blockDim.x + threadIdx.x;
	if (e >= E)
		return;
	int m = offsets[e + 1] - offsets[e];
	problems[e] = cutlass::gemm::GemmCoord(Ka, N, m);
	pA[e] = const_cast<cutlass::half_t*>(T) + (size_t)offsets[e];  // [Ka, m], lda = n
	pB[e] = const_cast<cutlass::half_t*>(B) + (size_t)offsets[e] * N;
	pC[e] = C + (size_t)e * Ka * N;
	pD[e] = pC[e];
	ld[e] = n;
	ld[E + e] = N;
	ld[2 * E + e] = N;
	ld[3 * E + e] = N;
}

namespace {
	struct WgradScratch {
		cutlass::half_t* T = nullptr;
		size_t cap = 0;   // elements
	};
	WgradScratch g_wgradT;
}

extern "C" void ggl_moe_grouped_wgrad_f16_dev(
	void* ctx, const void* A, const void* B, void* C,
	const int* offsets, int E, int Ka, int N, int nAssign, void* stream) {

	GemmCtx* c = (GemmCtx*)ctx;
	cudaStream_t s = (cudaStream_t)stream;
	if (E > c->E) {
		fprintf(stderr, "MoEKernels: wgrad E=%d exceeds ctx capacity %d\n", E, c->E);
		abort();
	}
	const int n = nAssign; // == offsets[E]; caller-known, no D2H sync
	if (n <= 0)
		return;
	if ((size_t)n * Ka > g_wgradT.cap) {
		GGLK_CUDA_CHECK(cudaMalloc(&g_wgradT.T, sizeof(__half) * (size_t)n * Ka * 2));
		g_wgradT.cap = (size_t)n * Ka * 2;   // grow-only (V100/11.2 trap)
	}
	dim3 tb(32, 32);
	dim3 grid((Ka + 31) / 32, (n + 31) / 32);
	gglk_transpose_f16<<<grid, tb, 0, s>>>((const __half*)A, (__half*)g_wgradT.T, n, Ka);

	gglk_fill_wgrad_problems<<<(E + 63) / 64, 64, 0, s>>>(
		offsets, E, Ka, N, n,
		(const cutlass::half_t*)g_wgradT.T, (const cutlass::half_t*)B, (cutlass::half_t*)C,
		c->d_problems, c->d_ptr_A, c->d_ptr_B, c->d_ptr_C, c->d_ptr_D, c->d_ld);

	static int tbcW = [] {
		int t = GglkGemmF16::sufficient(nullptr, 0);
		if (t <= 0) {
			fprintf(stderr, "MoEKernels: wgrad sufficient() returned %d\n", t);
			abort();
		}
		return t;
	}();

	typename GglkGemmF16::EpilogueOutputOp::Params epilogue(1.f, 0.f);
	typename GglkGemmF16::Arguments args(
		c->d_problems, E, tbcW, epilogue,
		c->d_ptr_A, c->d_ptr_B, c->d_ptr_C, c->d_ptr_D,
		c->d_ld, c->d_ld + E, c->d_ld + 2 * E, c->d_ld + 3 * E,
		/*host_problem_sizes=*/nullptr);

	size_t ws = GglkGemmF16::get_workspace_size(args);
	size_t need = ws + ((size_t)1 << 20);
	if (need > c->ws_cap) {
		GGLK_CUDA_CHECK(cudaMalloc(&c->d_workspace, need)); // grow-only
		c->ws_cap = need;
	}
	if (ws > 0)
		GGLK_CUDA_CHECK(cudaMemsetAsync(c->d_workspace, 0, ws, s));

	GglkGemmF16 gemm;
	GGLK_CUTLASS_CHECK(gemm.initialize(args, c->d_workspace, s));
	GGLK_CUTLASS_CHECK(gemm.run(s));
}

extern "C" int ggl_moe_kernels_available() {
	static int avail = [] {
		int dev = 0;
		if (cudaGetDevice(&dev) != cudaSuccess)
			return 0;
		cudaDeviceProp p;
		if (cudaGetDeviceProperties(&p, dev) != cudaSuccess)
			return 0;
		return (p.major == 7 && p.minor == 0) ? 1 : 0; // built for sm_70 exactly
	}();
	return avail;
}
