#pragma once
#include <cstdint>

// GGL_MOE_KERNELS — fused MoE collect-forward kernels for the V100 fleet
// (research/reports/MOE_SPEED.md Stage 1). Pure CUDA + CUTLASS behind a C ABI:
// this header is includable from torch-side .cpp without any CUDA headers
// (streams pass as void*). Compiled ONLY when the CMake option GGL_MOE_KERNELS
// is ON — the AiMOS cluster build. Desktop CUDA cannot even target sm_70, and
// keeps the torch baddbmm path.
//
// Everything here exists to cut LAUNCH COUNT (the ppc64le dispatch tax,
// ~0.15-0.5ms/op) and padded-GEMM waste: routing plan in 4 launches instead of
// ~15 torch ops, exact-M CUTLASS grouped GEMMs (kDeviceOnly schedule — problem
// sizes read on DEVICE, no host sync of routing counts anywhere).

extern "C" {

	// 1 when the kernels were compiled in AND the runtime device is sm_70-capable.
	int ggl_moe_kernels_available();

	// Persistent per-callsite scratch for the grouped GEMM (device problem/pointer
	// tables + CUTLASS workspace). GROW-ONLY by design: freeing grouped scratch
	// after a successful GEMM produced sticky illegal-access errors on V100 /
	// CUDA 11.2 (measured in moe-bench; see moe_cutlass_reset there).
	void* ggl_moe_ctx_create(int maxExperts);

	// Routing plan from raw router logits (FP32 [R,E] — the eager path routes on fp32
	// logits, and fp16 rounding before top-k flips boundary selections) + selection
	// bias (fp16 [E]):
	// DSv3 semantics matched to MoEBlockImpl::forward — selection by
	// sigmoid(logit)+bias, gate = sigmoid(logit) of the selected experts,
	// renormalized over the k picks. Outputs, in expert-sorted assignment order
	// (n = R*k assignments):
	//   countsCursors int32 [2E] scratch (zeroed internally)
	//   offsets       int32 [E+1] exclusive scan of per-expert counts
	//   rowScratch    int32 [n] + float [n] caller-owned per-row scratch — CALLER
	//                 owned (not a library global) so CUDA-graph capture never sees
	//                 a reallocating pointer; see rowExpScratch/rowGateScratch.
	//   srcRows       int32 [n] assignment -> source row
	//   expertId      int32 [n] assignment -> expert
	//   gates         float [n] normalized gate weight
	// 4 launches (memset, topk+count, scan, place).
	// alignSegments != 0: per-expert segment starts round up to 8 (fp16 tensor-op
	// pointer/K alignment for the LEARN-path wgrad). Pad slots get srcRows = -1 /
	// gates = 0 (pre-initialized here); callers must size the assignment buffers
	// to nBound = roundup8(R*k) + 8*E and launch the per-assignment kernels over
	// nBound — every kernel below guards srcRows < 0.
	void ggl_moe_route_plan_f16(
		const void* logits, const void* selBias,
		int R, int E, int k, int alignSegments,
		int* countsCursors, int* offsets,
		int* rowExpScratch, float* rowGateScratch,
		int* srcRows, int* expertId, float* gates,
		void* stream);

	// packed[i,:] = x[srcRows[i],:], fp16, 1 launch.
	void ggl_moe_gather_f16(
		const void* x, const int* srcRows, void* packed,
		int n, int d, void* stream);

	// Exact-M grouped GEMM, fp16 tensor-core (sm_70 8x8x4):
	// for expert e: C[offsets[e]:offsets[e+1], N] = A[...,K] @ B[e]  (B row-major [K,N]
	// per expert, i.e. the TRANSPOSED weight stack, contiguous [E,K,N]).
	// offsets is DEVICE memory; problem descriptors are filled by a device kernel
	// (1 launch) + the persistent grouped kernel (1 launch). Zero-count experts
	// contribute zero tiles. 2 launches total.
	void ggl_moe_grouped_gemm_f16_dev(
		void* ctx, const void* A, const void* B, void* C,
		const int* offsets, int E, int K, int N, void* stream);

	// EP variant: nProb problems but only `weightMod` distinct weight matrices —
	// problem i uses B[i % weightMod]. Lets an owner run every learner's tokens
	// against ONE copy of its expert stack instead of an nL-way repeat() (which
	// was ~650MB of memcpy per GEMM and made fwdbwd 3.15s at 1B).
	void ggl_moe_grouped_gemm_f16_dev_mod(
		void* ctx, const void* A, const void* B, void* C,
		const int* offsets, int nProb, int weightMod, int K, int N, void* stream);

	// h[i,:] = leaky_relu(h[i,:] + b[expertId[i],:], slope), fp16, 1 launch.
	void ggl_moe_bias_leaky_f16(
		void* h, const void* b, const int* expertId,
		int n, int H, float slope, void* stream);

	// Zero an fp16 buffer (cudaMemsetAsync — cheaper than a torch zero_ dispatch).
	void ggl_moe_zero_f16(void* p, int64_t count, void* stream);

	// out[srcRows[i],:] += (y[i,:] + b[expertId[i],:]) * gates[i]  (fp16 atomics),
	// 1 launch. out must be zeroed by the caller.
	void ggl_moe_scatter_bias_gate_f16(
		const void* y, const void* b, const int* expertId,
		const float* gates, const int* srcRows, void* out,
		int n, int D, void* stream);

	// ==================== learn path (Stage 1b, MOE_SPEED.md) ====================

	// dY[i,:] = dOut[srcRows[i],:] * gates[i], fp16 in/out, 1 launch.
	void ggl_moe_gather_scale_f16(
		const void* dOut, const int* srcRows, const float* gates,
		void* dY, int n, int d, void* stream);

	// dB[expertId[i],:] += rows of src (fp32 accumulator, caller zeroes), 1 launch.
	void ggl_moe_segment_sum_f16to32(
		const void* src, const int* expertId, float* dB,
		int n, int d, void* stream);

	// dHpre[i,:] = dH[i,:] * (Hpost[i,:] > 0 ? 1 : slope), fp16, in place on dH.
	void ggl_moe_leaky_bwd_f16(
		void* dH, const void* hPost, int n, int H, float slope, void* stream);

	// gdot[i] = sum_c dOut[srcRows[i],c] * (y[i,c] + b[expertId[i],c]), fp32 out.
	void ggl_moe_gate_dot_f16(
		const void* dOut, const void* y, const void* b,
		const int* expertId, const int* srcRows,
		float* gdot, int n, int d, void* stream);

	// dxn[srcRows[i],:] += dX[i,:] (fp16 atomics, no gate/bias), 1 launch.
	void ggl_moe_scatter_add_f16(
		const void* dX, const int* srcRows, void* dxn,
		int n, int d, void* stream);

	// Grouped wgrad: for expert e, C[e] = A_e^T @ B_e where A/B are the expert-sorted
	// assignment buffers (A [n,Ka] rows off[e]..off[e+1], read column-major = A^T) and
	// C is the [E, Ka, N] gradient stack. Per-problem K = the expert's row count, read
	// from DEVICE offsets. fp16 in, fp16 out. 2 launches (fill + persistent kernel).
	void ggl_moe_grouped_wgrad_f16_dev(
		void* ctx, const void* A, const void* B, void* C,
		const int* offsets, int E, int Ka, int N, int nAssign, void* stream);

}
