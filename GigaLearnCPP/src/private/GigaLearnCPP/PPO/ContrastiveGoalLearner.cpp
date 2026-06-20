#include "ContrastiveGoalLearner.h"

#include <torch/nn/modules/loss.h>

using namespace torch;

namespace GGL {

	static ModelConfig MakeEncoderConfig(int inputSize, int reprSize) {
		ModelConfig result = PartialModelConfig{};
		result.layerSizes = { 1024, 1024, 1024, 1024 };
		result.activationType = ModelActivationType::SWISH;
		result.optimType = ModelOptimType::ADAM;
		result.addLayerNorm = true;
		result.addOutputLayer = true;
		result.numInputs = inputSize;
		result.numOutputs = reprSize;
		return result;
	}

	static Tensor L2Normalize(Tensor t) {
		return t / t.norm(2, -1, true).clamp_min(1e-6f);
	}

	ContrastiveGoalLearner::ContrastiveGoalLearner(int obsSize, int actionRepresentationSize, const ContrastiveGoalConfig& config, torch::Device device) :
		stateActionEncoder("gcrl_phi", MakeEncoderConfig(obsSize + actionRepresentationSize, config.representationSize), device),
		goalEncoder("gcrl_psi", MakeEncoderConfig(6, config.representationSize), device),
		config(config), device(device), obsSize(obsSize), actionRepresentationSize(actionRepresentationSize) {
		SetLearningRate(config.criticLR);
	}

	Tensor ContrastiveGoalLearner::EncodeStateAction(Tensor states, Tensor actionRepresentations) {
		Tensor raw = stateActionEncoder.Forward(torch::cat({ states, actionRepresentations.to(kFloat32).to(states.device()) }, -1), false);
		return L2Normalize(raw);
	}

	Tensor ContrastiveGoalLearner::EncodeGoal(Tensor goals) {
		Tensor raw = goalEncoder.Forward(goals.to(kFloat32), false);
		return L2Normalize(raw);
	}

	Tensor ContrastiveGoalLearner::Score(Tensor states, Tensor actionRepresentations, Tensor goals) {
		Tensor sa = EncodeStateAction(states, actionRepresentations);
		Tensor g = EncodeGoal(goals);
		return (sa * g).sum(-1) / config.tau;
	}

	ContrastiveGoalStats ContrastiveGoalLearner::Train(ExperienceTensors& data, std::default_random_engine& rng) {
		ContrastiveGoalStats stats;

		if (
			!data.states.defined() ||
			!data.actions.defined() ||
			!data.herGoals.defined() ||
			data.states.size(0) == 0
		)
			return stats;

		int64_t n = data.states.size(0);
		if (data.actions.size(0) != n || data.herGoals.size(0) != n)
			RG_ERR_CLOSE("ContrastiveGoalLearner::Train(): tensor alignment failed");

		int64_t miniBatchSize = config.criticMiniBatchSize;
		if (miniBatchSize <= 0)
			miniBatchSize = n;
		if (config.infoSubSample > 0)
			miniBatchSize = RS_MIN(miniBatchSize, config.infoSubSample);

		float totalLoss = 0;
		float totalRowLoss = 0;
		float totalColumnLoss = 0;
		float totalPositiveLogit = 0;
		float totalNegativeLogit = 0;
		float totalSANorm = 0;
		float totalGoalNorm = 0;
		float totalAccuracy = 0;
		float totalLogsumexp = 0;
		int64_t batches = 0;

		std::vector<int64_t> indices(n);
		std::iota(indices.begin(), indices.end(), 0);

		for (int epoch = 0; epoch < config.criticEpochs; epoch++) {
			std::shuffle(indices.begin(), indices.end(), rng);

			for (int64_t start = 0; start < n; start += miniBatchSize) {
				int64_t curBatchSize = std::min<int64_t>(miniBatchSize, n - start);
				if (curBatchSize <= 1)
					continue;

				std::vector<int64_t> batchIndices(indices.begin() + start, indices.begin() + start + curBatchSize);
				stats.trainSamplesUsed += curBatchSize;

				Tensor tIndices = torch::tensor(batchIndices, TensorOptions().dtype(kLong));

				Tensor states = data.states.index_select(0, tIndices.to(data.states.device())).to(device);
				Tensor actions = data.actions.index_select(0, tIndices.to(data.actions.device())).to(device).to(kLong);
				Tensor actionRepresentations = torch::zeros(
					{ curBatchSize, actionRepresentationSize },
					TensorOptions().dtype(kFloat32).device(device)
				).scatter_(1, actions.unsqueeze(1), 1.f);
				Tensor goals = data.herGoals.index_select(0, tIndices.to(data.herGoals.device())).to(device);

				Tensor sa = EncodeStateAction(states, actionRepresentations);
				Tensor g = EncodeGoal(goals);

				Tensor logits = torch::matmul(sa, g.transpose(0, 1)) / config.tau;
				Tensor labels = torch::arange(curBatchSize, TensorOptions().dtype(kLong).device(device));
				Tensor diag = torch::eye(curBatchSize, TensorOptions().dtype(kBool).device(device));

				Tensor rowLoss = torch::nn::CrossEntropyLoss()(logits, labels);
				Tensor columnLoss = torch::nn::CrossEntropyLoss()(logits.transpose(0, 1), labels);
				Tensor logsumexpRows = torch::logsumexp(logits, 1);
				Tensor logsumexpColumns = torch::logsumexp(logits, 0);
				Tensor logsumexpPenalty = config.logsumexpPenaltyCoeff * (logsumexpRows.pow(2).mean() + logsumexpColumns.pow(2).mean());

				float stdNorm = sqrtf((float)config.representationSize);
				Tensor varPenalty = config.varReg * (
					1.0f / (sa.std(0, false).mean() * stdNorm + 1e-4f) +
					1.0f / (g.std(0, false).mean() * stdNorm + 1e-4f)
				);

				Tensor loss = rowLoss + columnLoss + logsumexpPenalty + varPenalty;

				stateActionEncoder.optim->zero_grad();
				goalEncoder.optim->zero_grad();
				loss.backward();
				stateActionEncoder.StepOptim();
				goalEncoder.StepOptim();

				Tensor positiveLogits = logits.diag();
				Tensor negativeLogits = logits.masked_select(~diag);
				Tensor rowCorrect = logits.argmax(1).eq(labels);
				Tensor columnCorrect = logits.argmax(0).eq(labels);

				totalLoss += loss.item<float>();
				totalRowLoss += rowLoss.item<float>();
				totalColumnLoss += columnLoss.item<float>();
				totalPositiveLogit += positiveLogits.mean().item<float>();
				if (negativeLogits.numel() > 0)
					totalNegativeLogit += negativeLogits.mean().item<float>();
				totalSANorm += sa.norm(2, -1).mean().item<float>();
				totalGoalNorm += g.norm(2, -1).mean().item<float>();
				totalAccuracy += 0.5f * (rowCorrect.to(kFloat).mean().item<float>() + columnCorrect.to(kFloat).mean().item<float>());
				totalLogsumexp += 0.5f * (logsumexpRows.mean().item<float>() + logsumexpColumns.mean().item<float>());
				batches++;
			}
		}

		stats.anchorsUsed = n;
		if (batches > 0) {
			stats.loss = totalLoss / batches;
			stats.rowLoss = totalRowLoss / batches;
			stats.columnLoss = totalColumnLoss / batches;
			stats.positiveLogitMean = totalPositiveLogit / batches;
			stats.negativeLogitMean = totalNegativeLogit / batches;
			stats.stateActionEmbeddingNorm = totalSANorm / batches;
			stats.goalEmbeddingNorm = totalGoalNorm / batches;
			stats.categoricalAccuracy = totalAccuracy / batches;
			stats.logsumexpMean = totalLogsumexp / batches;
		}

		return stats;
	}

	void ContrastiveGoalLearner::Save(std::filesystem::path folder, bool saveOptim) {
		stateActionEncoder.Save(folder, saveOptim);
		goalEncoder.Save(folder, saveOptim);
	}

	void ContrastiveGoalLearner::Load(std::filesystem::path folder, bool allowNotExist, bool loadOptim) {
		stateActionEncoder.Load(folder, allowNotExist, loadOptim);
		goalEncoder.Load(folder, allowNotExist, loadOptim);
	}

	void ContrastiveGoalLearner::SetLearningRate(float lr) {
		stateActionEncoder.SetOptimLR(lr);
		goalEncoder.SetOptimLR(lr);
	}
}
