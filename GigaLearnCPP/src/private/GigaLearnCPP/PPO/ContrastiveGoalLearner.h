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
		// FORK2 TD-contrastive (GOALSHORT only) diagnostics
		float tdBlendR = -1;               // effective MC->TD ramp r this iter (-1 if TD off for this critic)
		float tdReverted = 0;              // 1 if the collapse guard forced r=0 this iter
		float tdSoftValueEntropyFrac = 0;  // mean over valid rows of H(softmax_k f^-)/log K (soft-value peakiness; low=>collapse)
		float tdRowLoss = 0;               // mean soft-CE TD row term
		float tdEmaDrift = 0;              // ||phiTail^- - phiTail|| / ||phiTail|| (target lag observability)
		float tdValidBootstrapRows = 0;    // mean per-minibatch count of bootstrap-eligible rows
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

		// FORK2 TD-contrastive EMA targets (constructed only when useTD = useTDContrastive && !useCarGoals;
		// TD is GOALSHORT/REACH-only, CAR stays pure MC). Distinct model NAMES (suffix _tgt) so they never
		// collide with the live nets; backing strings must outlive the Model (it stores name as c_str()).
		// Targets are NOT checkpointed — lazily re-synced live->target on the first Train (resume re-lags
		// within ~tdEmaHalfLifeIters; negligible).
		bool useTD = false;
		bool tdTargetsSynced = false;
		bool tdCollapseLatched = false;   // hysteretic soft-value collapse guard (1-iter latency safety valve)
		std::string phiTailTgtName, psiTgtName, trunkTgtName;
		Model* phiTailTarget = nullptr;
		Model* goalEncoderTarget = nullptr;
		Model* sharedHeadTarget = nullptr;

		// sharedHead: the PPO actor/critic trunk (owned by PPOLearner).
		// obsNorm:    optional running normalizer applied before sharedHead (may be null).
		ContrastiveGoalLearner(int obsSize, int actionRepresentationSize, const ContrastiveGoalConfig& config, torch::Device device,
			Model* sharedHead, const RSNorm* obsNorm = nullptr,
			const std::string& namePrefix = "gcrl", bool useCarGoals = false, bool applyTrainMask = true);
		~ContrastiveGoalLearner();

		// embeddings: output of shared_head(obs).detach() — [N, embeddingDim]
		torch::Tensor EncodeStateAction(torch::Tensor embeddings, torch::Tensor actionRepresentations);
		torch::Tensor EncodeGoal(torch::Tensor goals);
		torch::Tensor Score(torch::Tensor embeddings, torch::Tensor actionRepresentations, torch::Tensor goals);

		// totalTimesteps drives the MC->TD blend ramp; nextPolicyProbs = [N, numActions] masked policy probs
		// at every buffer state (host tensor, computed by PPOLearner), index-shifted internally for the
		// one-step bootstrap. Both are ignored when useTD is false (CAR / TD-off) — pass an undefined tensor.
		ContrastiveGoalStats Train(ExperienceTensors& data, std::default_random_engine& rng,
			uint64_t totalTimesteps = 0, torch::Tensor nextPolicyProbs = {});

		void Save(std::filesystem::path folder, bool saveOptim = true);
		void Load(std::filesystem::path folder, bool allowNotExist, bool loadOptim = true);
		void SetLearningRate(float lr);
	};
}
