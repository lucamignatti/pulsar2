#include "ExperienceBuffer.h"

using namespace torch;

GGL::ExperienceBuffer::ExperienceBuffer(int seed, torch::Device device) :
	seed(seed), device(device), rng(seed) {

}

GGL::ExperienceTensors GGL::ExperienceBuffer::_GetSamples(torch::Tensor indices) const {
	ExperienceTensors result;
	auto* toItr = result.begin();
	auto* fromItr = data.begin();
	for (; toItr != result.end(); toItr++, fromItr++)
		if (fromItr->defined())
			*toItr = torch::index_select(*fromItr, 0, indices);
	return result;
}

GGL::ExperienceTensors GGL::ExperienceBuffer::_GetSamples(const int64_t* indices, size_t size) const {

	// TODO: Slow, use blob
	Tensor tIndices = torch::tensor(IList(indices, indices + size));
	return _GetSamples(tIndices);
}

std::vector<GGL::ExperienceTensors> GGL::ExperienceBuffer::GetAllBatchesShuffled(int64_t batchSize, bool overbatching) {

	RG_NO_GRAD;

	int64_t expSize = data.states.size(0);
	std::vector<ExperienceTensors> result;
	if (expSize <= 0)
		return result;

	auto pushRange = [&](const int64_t* cpuIdx, torch::Tensor gpuIdx, int64_t startIdx, int64_t curBatchSize) {
		if (gpuIdx.defined())
			result.push_back(_GetSamples(gpuIdx.narrow(0, startIdx, curBatchSize)));
		else
			result.push_back(_GetSamples(cpuIdx + startIdx, (size_t)curBatchSize));
	};

	if (data.IsOnCUDA()) {
		auto perm = torch::randperm(
			expSize,
			torch::TensorOptions().dtype(torch::kInt64).device(data.states.device()));
		for (int64_t startIdx = 0; startIdx + batchSize <= expSize; startIdx += batchSize) {
			int64_t curBatchSize = batchSize;
			if (startIdx + batchSize * 2 > expSize && overbatching)
				curBatchSize = expSize - startIdx;
			pushRange(nullptr, perm, startIdx, curBatchSize);
		}
		return result;
	}

	int64_t* indices = new int64_t[expSize];
	std::iota(indices, indices + expSize, 0);
	std::shuffle(indices, indices + expSize, rng);

	for (int64_t startIdx = 0; startIdx + batchSize <= expSize; startIdx += batchSize) {
		int64_t curBatchSize = batchSize;
		if (startIdx + batchSize * 2 > expSize && overbatching)
			curBatchSize = expSize - startIdx;
		pushRange(indices, {}, startIdx, curBatchSize);
	}

	delete[] indices;
	return result;
}

void GGL::ExperienceBuffer::UploadToDevice() {
	if (!device.is_cuda() || !data.states.defined())
		return;
	if (data.IsOnCUDA())
		return;

	RG_NO_GRAD;
	for (auto* t = data.begin(); t != data.end(); t++) {
		if (!t->defined() || t->numel() == 0)
			continue;
		if (t->is_cuda())
			continue;
		if (!t->is_pinned())
			*t = t->pin_memory();
		*t = t->to(device, /*non_blocking=*/true);
	}
	// Do not synchronize here: Learn's GPU shuffle / first GEMM queue on this same
	// stream and wait for the H2Ds automatically. A host sync only delays launching them.
}
