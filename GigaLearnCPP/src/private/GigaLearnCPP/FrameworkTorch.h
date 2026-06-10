#pragma once
#include <GigaLearnCPP/Framework.h>
#include <RLGymCPP/BasicTypes/Lists.h>

// Include torch
#include <ATen/ATen.h>
#include <ATen/autocast_mode.h>
#include <torch/utils.h>

#define RG_NO_GRAD torch::NoGradGuard _noGradGuard

#define RG_AUTOCAST_ON() { \
at::autocast::set_enabled(true); \
at::autocast::set_autocast_gpu_dtype(torch::kBFloat16); \
at::autocast::set_autocast_cpu_dtype(torch::kFloat); \
}

#define RG_AUTOCAST_OFF() { \
at::autocast::clear_cache(); \
at::autocast::set_enabled(false); \
}

#define RG_HALFPERC_TYPE torch::ScalarType::BFloat16

// Whether host->device copies may be non-blocking on this device.
// On CUDA/ROCm, pageable-memory H2D copies complete synchronously with respect to the
// source buffer before returning, so non_blocking is safe even for tensors that wrap
// externally-owned memory (e.g. env obs buffers via DIMLIST2_TO_TENSOR).
// MPS commits command buffers lazily: a non_blocking copy can execute AFTER the caller
// has overwritten the source, silently shipping garbage to the GPU (this corrupted the
// policy input and crashed multinomial with NaN probs). Block everywhere except CUDA.
#define RG_H2D_NONBLOCKING(device) ((device).is_cuda())

namespace GGL {
	template <typename T>
	inline torch::Tensor DIMLIST2_TO_TENSOR(const RLGC::DimList2<T>& list) {
		// from_blob creates a non-owning view into the DimList2 data — no copy.
		// Callers must not modify list.data until this tensor is consumed (sent to device or cloned).
		return torch::from_blob(
			const_cast<T*>(list.data.data()),
			{ (int64_t)list.size[0], (int64_t)list.size[1] },
			torch::TensorOptions().dtype(torch::CppTypeToScalarType<T>())
		);
	}

	template <typename T>
	inline std::vector<T> TENSOR_TO_VEC(torch::Tensor tensor) {
		assert(tensor.dim() == 1);
		tensor = tensor.contiguous().cpu().detach().to(torch::CppTypeToScalarType<T>());
		T* data = tensor.data_ptr<T>();
		return std::vector<T>(data, data + tensor.size(0));
	}
}