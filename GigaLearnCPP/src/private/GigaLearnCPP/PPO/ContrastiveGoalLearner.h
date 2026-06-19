#pragma once

#include "ExperienceBuffer.h"

#include "../Util/Models.h"

namespace GGL {

	struct ContrastiveGoalStats {
		float loss = 0;
		float positiveLogitMean = 0;
		float negativeLogitMean = 0;
		float stateActionEmbeddingNorm = 0;
		float goalEmbeddingNorm = 0;
		float duplicateNegativeMaskRate = 0;
		float eligibleFutureMaskRate = 0;
		float realizedImmediate = 0;
		float realizedShort = 0;
		float realizedMedium = 0;
		float realizedLong = 0;
		int64_t anchorsUsed = 0;
		int64_t trainSamplesUsed = 0;
	};

	class ContrastiveGoalLearner {
	public:
		Model stateActionEncoder;
		Model goalEncoder;
		ContrastiveGoalConfig config;
		torch::Device device;
		int obsSize;
		int actionControlSize;

		ContrastiveGoalLearner(int obsSize, int actionControlSize, const ContrastiveGoalConfig& config, torch::Device device);

		torch::Tensor EncodeStateAction(torch::Tensor states, torch::Tensor actionControls);
		torch::Tensor EncodeGoal(torch::Tensor goals);
		torch::Tensor Score(torch::Tensor states, torch::Tensor actionControls, torch::Tensor goals);
		torch::Tensor ScoreAllActions(torch::Tensor states, torch::Tensor allActionControls, torch::Tensor goals);

		ContrastiveGoalStats Train(ExperienceTensors& data, std::default_random_engine& rng);

		void Save(std::filesystem::path folder, bool saveOptim = true);
		void Load(std::filesystem::path folder, bool allowNotExist, bool loadOptim = true);
		void SetLearningRate(float lr);
	};
}
