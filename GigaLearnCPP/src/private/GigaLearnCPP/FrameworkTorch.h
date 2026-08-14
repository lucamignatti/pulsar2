#pragma once
#include <GigaLearnCPP/Framework.h>
#include <RLGymCPP/BasicTypes/Lists.h>

// Include torch
#include <ATen/ATen.h>
#include <ATen/autocast_mode.h>
#include <torch/utils.h>

#define RG_NO_GRAD torch::NoGradGuard _noGradGuard

#ifdef RG_CUDA_SUPPORT
#include <cuda_runtime.h>
#endif

namespace GGL {

	// Ampere+ (sm_80) has BF16 Tensor Cores. V100 is sm_70: FP16 Tensor Cores only.
	// BF16 matmuls on Volta take a slow software path and starve the card.
	inline bool GGLCudaHasBF16() {
#ifdef RG_CUDA_SUPPORT
		int ndev = 0;
		if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev <= 0)
			return false;
		int dev = 0;
		if (cudaGetDevice(&dev) != cudaSuccess)
			dev = 0;
		cudaDeviceProp prop{};
		if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess)
			return false;
		return prop.major >= 8;
#else
		return false;
#endif
	}

	// Inference half dtype: BF16 on sm_80+, FP16 on V100, BF16 on MPS/CPU fallback.
	inline torch::ScalarType GGLHalfPrecType() {
#ifdef RG_CUDA_SUPPORT
		int ndev = 0;
		if (cudaGetDeviceCount(&ndev) == cudaSuccess && ndev > 0)
			return GGLCudaHasBF16() ? torch::kBFloat16 : torch::kHalf;
#endif
		return torch::kBFloat16;
	}

	inline const char* GGLHalfPrecName() {
		return (GGLHalfPrecType() == torch::kHalf) ? "fp16" : "bf16";
	}

} // namespace GGL

// Learn autocast: BF16 on sm_80+ (no scaler), FP16 on V100 (needs loss scaling).
#define RG_AUTOCAST_ON() { \
at::autocast::set_enabled(true); \
at::autocast::set_autocast_gpu_dtype(GGL::GGLHalfPrecType()); \
at::autocast::set_autocast_cpu_dtype(torch::kFloat); \
}

#define RG_AUTOCAST_OFF() { \
at::autocast::clear_cache(); \
at::autocast::set_enabled(false); \
}

#define RG_HALFPERC_TYPE GGL::GGLHalfPrecType()

namespace GGL {
	template <typename T>
	inline torch::Tensor DIMLIST2_TO_TENSOR(const RLGC::DimList2<T>& list) {
		return torch::tensor(list.data).reshape({ (int64_t)list.size[0], (int64_t)list.size[1] });
	}

	template <typename T>
	inline std::vector<T> TENSOR_TO_VEC(torch::Tensor tensor) {
		assert(tensor.dim() == 1);
		tensor = tensor.contiguous().cpu().detach().to(torch::CppTypeToScalarType<T>());
		T* data = tensor.data_ptr<T>();
		return std::vector<T>(data, data + tensor.size(0));
	}
}
