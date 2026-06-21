#pragma once

#include "ExperienceBuffer.h"

#include "../Util/Models.h"

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
		// Stable backing storage for the encoder names. Model keeps the const char* as-is,
		// so the string must outlive the Model -- declared BEFORE the encoders so it is
		// constructed first. (Passing a temporary's c_str() here is a use-after-free.)
		std::string phiName, psiName;
		Model stateActionEncoder;
		Model goalEncoder;
		// When non-null, the state-action encoder (phi) is SHARED with another critic; the owned
		// stateActionEncoder above is then inert. Lets useSharedBase give all GCRL heads one phi base
		// + small per-head psi. The owner (the critic that constructed the phi) saves/loads/LR-steps it.
		Model* sharedStateActionEncoder = nullptr;
		Model& Phi() { return sharedStateActionEncoder ? *sharedStateActionEncoder : stateActionEncoder; }
		ContrastiveGoalConfig config;
		torch::Device device;
		int obsSize;
		int actionRepresentationSize;

		// useCarGoals: train/score against data.carHerGoals (egocentric ball) instead of data.herGoals.
		// useBoostGoals: train/score against data.boostHerGoals (own boost level) -- the boost head.
		// applyTrainMask: when false, ignore the ball-moved gcrlTrainMask -- the car/boost critics learn
		// even on episodes where the ball never moved. namePrefix keeps the critics' encoders (and
		// checkpoint files) distinct.
		bool useCarGoals = false;
		bool useBoostGoals = false;
		bool applyTrainMask = true;

		ContrastiveGoalLearner(int obsSize, int actionRepresentationSize, const ContrastiveGoalConfig& config, torch::Device device,
			const std::string& namePrefix = "gcrl", bool useCarGoals = false, bool useBoostGoals = false, bool applyTrainMask = true, Model* sharedPhi = nullptr);

		torch::Tensor EncodeStateAction(torch::Tensor states, torch::Tensor actionRepresentations);
		torch::Tensor EncodeGoal(torch::Tensor goals);
		torch::Tensor Score(torch::Tensor states, torch::Tensor actionRepresentations, torch::Tensor goals);

		struct InfoNCEResult { torch::Tensor loss; torch::Tensor metrics; };
		// InfoNCE loss for ONE head given a precomputed (shared) state-action embedding `sa` and this
		// head's goals. Forwards only this head's small psi; does NOT include the sa (phi) variance
		// penalty (the caller adds StateActionVarPenalty once on the shared sa, so under a shared base it
		// is not triple-counted). Returns the loss to backprop and a stacked 9-element metrics tensor.
		InfoNCEResult ComputeInfoNCELoss(torch::Tensor sa, torch::Tensor goals);
		// varReg penalty on the shared state-action (phi) embedding; added once per optimizer step.
		torch::Tensor StateActionVarPenalty(torch::Tensor sa);

		ContrastiveGoalStats Train(ExperienceTensors& data, std::default_random_engine& rng);

		void Save(std::filesystem::path folder, bool saveOptim = true);
		void Load(std::filesystem::path folder, bool allowNotExist, bool loadOptim = true);
		void SetLearningRate(float lr);
	};
}
