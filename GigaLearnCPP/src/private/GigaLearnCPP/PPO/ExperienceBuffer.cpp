#include "ExperienceBuffer.h"

using namespace torch;

GGL::ExperienceBuffer::ExperienceBuffer(int seed, torch::Device device) :
	seed(seed), device(device), rng(seed) {

}

GGL::ExperienceTensors GGL::ExperienceBuffer::_GetSamples(const int64_t* indices, size_t size) const {

	// Borrow the caller's int64 index buffer directly (no IList copy / int32 narrowing). index_select
	// returns fresh copies, so the borrowed blob need not outlive the returned tensors.
	Tensor tIndices = torch::from_blob(
		const_cast<int64_t*>(indices), { (int64_t)size }, torch::TensorOptions().dtype(torch::kInt64));

	ExperienceTensors result;
	auto* toItr = result.begin();
	auto* fromItr = data.begin();
	for (; toItr != result.end(); toItr++, fromItr++) {
		if (fromItr->defined())
			*toItr = torch::index_select(*fromItr, 0, tIndices);
	}

	return result;
}

std::vector<GGL::ExperienceTensors> GGL::ExperienceBuffer::GetAllBatchesShuffled(int64_t batchSize, bool overbatching) {

	RG_NO_GRAD;

	size_t expSize = data.states.size(0);

	// Make list of shuffled sample indices
	int64_t* indices = new int64_t[expSize];
	std::iota(indices, indices + expSize, 0); // Fill ascending indices
	std::shuffle(indices, indices + expSize, rng);

	// Get a sample set from each of the batches
	std::vector<ExperienceTensors> result;
	for (int64_t startIdx = 0; startIdx + batchSize <= expSize; startIdx += batchSize) {

		int curBatchSize = batchSize;
		if (startIdx + batchSize * 2 > expSize) {
			// Last batch of the iteration
			if (overbatching) {
				// Extend batch size to the end of the experience
				curBatchSize = expSize - startIdx;
			}
		}

		result.push_back(_GetSamples(indices + startIdx, curBatchSize));
	}

	delete[] indices;
	return result;
}
