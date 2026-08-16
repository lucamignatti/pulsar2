#include "PolicyPack.h"
#include <cstring>

using namespace torch;

namespace GGL {

static const char* kPackNames[] = { "shared_head", "policy" };

void PolicyPack::InitFrom(ModelSet& models) {
	paramNumel = 0;
	for (const char* n : kPackNames) {
		Model* m = models[n];
		if (!m)
			continue;
		for (auto& p : m->parameters())
			paramNumel += p.numel();
	}
	totalFloats = 6 + paramNumel;
}

ModelSet PolicyPack::ClonePolicyPair(ModelSet& src) {
	ModelSet out;
	for (const char* n : kPackNames) {
		Model* m = src[n];
		if (m)
			out.Add(m->MakeClone());
	}
	return out;
}

void PolicyPack::Pack(ModelSet& models, torch::Tensor staging, uint64_t policy_version, uint64_t archive_seq) {
	RG_NO_GRAD;
	RG_ASSERT(staging.defined());
	RG_ASSERT(staging.is_cuda());
	RG_ASSERT(staging.numel() == totalFloats);
	PolicyPackHeader hdr;
	hdr.policy_version = policy_version;
	hdr.nbytes_payload = (uint64_t)paramNumel * sizeof(float);
	hdr.archive_seq = archive_seq;
	auto hdrT = torch::from_blob(&hdr, { 6 }, torch::TensorOptions().dtype(torch::kFloat32)).clone();
	staging.narrow(0, 0, 6).copy_(hdrT.to(staging.device()), /*non_blocking=*/true);
	int64_t off = 6;
	for (const char* n : kPackNames) {
		Model* m = models[n];
		if (!m)
			continue;
		for (auto& p : m->parameters()) {
			auto flat = p.detach().to(torch::kFloat32).contiguous().view({ -1 });
			staging.narrow(0, off, flat.numel()).copy_(flat, /*non_blocking=*/true);
			off += flat.numel();
		}
	}
	RG_ASSERT(off == totalFloats);
	// Same-stream NCCL after Pack is ordered after these copies. A stream sync here
	// drained leftover Learn/Optim kernels and billed them as Weight Bcast (~116ms).
}

void PolicyPack::Unpack(torch::Tensor staging, ModelSet& dest, PolicyPackHeader* outHdr) {
	RG_NO_GRAD;
	PolicyPackHeader hdr;
	auto hcpu = staging.narrow(0, 0, 6).to(torch::kCPU, /*non_blocking=*/false);
	std::memcpy(&hdr, hcpu.data_ptr<float>(), sizeof(hdr));
	if (outHdr)
		*outHdr = hdr;
	int64_t off = 6;
	for (const char* n : kPackNames) {
		Model* m = dest[n];
		if (!m)
			continue;
		for (auto& p : m->parameters()) {
			int64_t nEl = p.numel();
			auto slice = staging.narrow(0, off, nEl).view(p.sizes());
			p.copy_(slice, /*non_blocking=*/true);
			off += nEl;
		}
		m->_seqHalfOutdated = true;
	}
}

}
