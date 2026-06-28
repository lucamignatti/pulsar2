#pragma once
#include <GigaLearnCPP/Framework.h>
#include <RLGymCPP/BasicTypes/Lists.h>

// Include torch
#include <ATen/ATen.h>
#include <ATen/autocast_mode.h>
#include <torch/utils.h>

#define RG_NO_GRAD torch::NoGradGuard _noGradGuard

// Device-typed autocast API (the old global set_enabled/set_autocast_gpu_dtype are deprecated and
// removed in newer libtorch). Enables BF16 autocast for CUDA ops only; numerically-sensitive ops
// (softmax/exp/log, layer_norm, losses) stay fp32 via autocast's own promotion lists.
#define RG_AUTOCAST_ON() { \
at::autocast::set_autocast_dtype(at::kCUDA, torch::kBFloat16); \
at::autocast::set_autocast_enabled(at::kCUDA, true); \
}

#define RG_AUTOCAST_OFF() { \
at::autocast::clear_cache(); \
at::autocast::set_autocast_enabled(at::kCUDA, false); \
}

#define RG_HALFPERC_TYPE torch::ScalarType::BFloat16

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