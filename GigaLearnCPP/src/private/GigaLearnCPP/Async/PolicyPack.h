#pragma once
#include "../FrameworkTorch.h"
#include "../Util/Models.h"

#include <cstdint>
#include <vector>

namespace GGL {

struct PolicyPackHeader {
	uint64_t policy_version = 0;
	uint64_t nbytes_payload = 0;
	uint64_t archive_seq = 0;
};
static_assert(sizeof(PolicyPackHeader) == 24, "pack header is 24 bytes");

class PolicyPack {
public:
	int64_t paramNumel = 0; // fp32 params only
	int64_t totalFloats = 0; // 6 header floats + paramNumel

	void InitFrom(ModelSet& models);
	// Staging is CUDA fp32, length totalFloats. Header occupies the first 6 floats.
	void Pack(ModelSet& models, torch::Tensor staging, uint64_t policy_version, uint64_t archive_seq);
	void Unpack(torch::Tensor staging, ModelSet& dest, PolicyPackHeader* outHdr = nullptr);

	static ModelSet ClonePolicyPair(ModelSet& src);
};

}
