#include "ExperienceBuffer.h"

using namespace torch;

GGL::ExperienceBuffer::ExperienceBuffer(int seed, torch::Device device) :
	seed(seed), device(device), rng(seed) {

}

GGL::ExperienceTensors GGL::ExperienceBuffer::_GetSamples(const int64_t* indices, size_t size) const {

	// TODO: Slow, use blob
	Tensor tIndices = torch::tensor(IList(indices, indices + size));

	ExperienceTensors result;
	auto fnSlice = [=](torch::Tensor t) -> torch::Tensor {
		return torch::index_select(t, 0, tIndices);
	};

	auto* toItr = result.begin();
	auto* fromItr = data.begin();
	for (; toItr != result.end(); toItr++, fromItr++)
		*toItr = torch::index_select(*fromItr, 0, tIndices);

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