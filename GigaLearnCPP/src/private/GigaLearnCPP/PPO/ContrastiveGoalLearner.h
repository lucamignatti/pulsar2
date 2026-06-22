#pragma once

#include "ExperienceBuffer.h"

#include "../Util/Models.h"
#include "../Util/RSNorm.h"

namespace GGL {

	struct ContrastiveGoalStats {
		float loss = 0;
		float rowLoss = 0;
		float columnLoss = 0;
		float positiveLogitMean = 0;
		float negativeLogitMean = 0;
		float stateActionEmbeddingNorm = 0;
		float goalEmbeddingNorm = 0;
		float categoricalAccuracy = 0;
		float logsumexpMean = 0;
		float realizedImmediate = 0;
		float realizedShort = 0;
		float realizedMedium = 0;
		float realizedLong = 0;
		int64_t anchorsUsed = 0;
		int64_t trainSamplesUsed = 0;
	};

	class ContrastiveGoalLearner {
	public:
		// Stable backing storage for the encoder names (must outlive the Model that holds c_str()).
		std::string phiTailName, psiName;
		// phi tail: action-fusion MLP on top of the shared_head embedding (non-owning ptr).
		// Input = cat(shared_head(obs).detach(), one_hot(action)).
		Model phiTail;
		Model goalEncoder;
		// Non-owning. shared_head is owned and checkpointed by PPOLearner.
		Model* sharedHead = nullptr;
		// Non-owning. Applied to obs before shared_head (consistent with actor/critic).
		const RSNorm* obsNorm = nullptr;

		ContrastiveGoalConfig config;
		torch::Device device;
		int obsSize;
		int actionRepresentationSize;

		bool useCarGoals = false;
		bool applyTrainMask = true;

		// sharedHead: the PPO actor/critic trunk (owned by PPOLearner).
		// obsNorm:    optional running normalizer applied before sharedHead (may be null).
		ContrastiveGoalLearner(int obsSize, int actionRepresentationSize, const ContrastiveGoalConfig& config, torch::Device device,
			Model* sharedHead, const RSNorm* obsNorm = nullptr,
			const std::string& namePrefix = "gcrl", bool useCarGoals = false, bool applyTrainMask = true);

		// embeddings: output of shared_head(obs).detach() — [N, embeddingDim]
		torch::Tensor EncodeStateAction(torch::Tensor embeddings, torch::Tensor actionRepresentations);
		torch::Tensor EncodeGoal(torch::Tensor goals);
		torch::Tensor Score(torch::Tensor embeddings, torch::Tensor actionRepresentations, torch::Tensor goals);

		ContrastiveGoalStats Train(ExperienceTensors& data, std::default_random_engine& rng);

		void Save(std::filesystem::path folder, bool saveOptim = true);
		void Load(std::filesystem::path folder, bool allowNotExist, bool loadOptim = true);
		void SetLearningRate(float lr);
	};
}
