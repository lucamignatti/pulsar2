#include "ContrastiveGoalLearner.h"

#include <torch/nn/modules/loss.h>
#include <unordered_map>

using namespace torch;

namespace GGL {

	struct PositiveSample {
		int64_t anchorIdx;
		int64_t positiveIdx;
		int bucket;
	};

	static ModelConfig MakeEncoderConfig(int inputSize, int reprSize) {
		ModelConfig result = PartialModelConfig{};
		result.layerSizes = { 1024, 1024, 1024, 1024, reprSize };
		result.activationType = ModelActivationType::SWISH;
		result.optimType = ModelOptimType::ADAM;
		result.addLayerNorm = true;
		result.addOutputLayer = false;
		result.numInputs = inputSize;
		result.numOutputs = reprSize;
		return result;
	}

	static bool TryPickBucketFuture(
		int anchorPos,
		int segmentSize,
		const ContrastiveGoalConfig& config,
		std::default_random_engine& rng,
		int& outFutureOffset,
		int& outBucket
	) {
		struct Bucket {
			int minOffset, maxOffset, id;
			float weight;
		};

		Bucket buckets[] = {
			{ config.immediateMin, config.immediateMax, 0, config.immediateWeight },
			{ config.shortMin, config.shortMax, 1, config.shortWeight },
			{ config.mediumMin, config.mediumMax, 2, config.mediumWeight },
			{ config.longMin, config.longMax, 3, config.longWeight },
		};

		float totalWeight = 0;
		for (Bucket& bucket : buckets) {
			if (anchorPos + bucket.minOffset < segmentSize)
				totalWeight += bucket.weight;
		}

		if (totalWeight <= 0)
			return false;

		std::uniform_real_distribution<float> bucketDist(0, totalWeight);
		float cursor = bucketDist(rng);
		Bucket* selected = &buckets[0];
		for (Bucket& bucket : buckets) {
			if (anchorPos + bucket.minOffset >= segmentSize)
				continue;

			if (cursor <= bucket.weight) {
				selected = &bucket;
				break;
			}
			cursor -= bucket.weight;
		}

		int maxOffset = std::min(selected->maxOffset, segmentSize - anchorPos - 1);
		if (maxOffset < selected->minOffset)
			return false;

		std::uniform_int_distribution<int> offsetDist(selected->minOffset, maxOffset);
		outFutureOffset = offsetDist(rng);
		outBucket = selected->id;
		return true;
	}

	static std::vector<PositiveSample> BuildPositiveSamples(ExperienceTensors& data, const ContrastiveGoalConfig& config, std::default_random_engine& rng, ContrastiveGoalStats& stats) {
		Tensor segmentIdsCPU = data.segmentIds.to(kCPU).to(kLong);
		int64_t n = segmentIdsCPU.size(0);
		auto segmentAccessor = segmentIdsCPU.accessor<int64_t, 1>();

		std::unordered_map<int64_t, std::vector<int64_t>> segmentToIndices;
		segmentToIndices.reserve(n);
		for (int64_t i = 0; i < n; i++)
			segmentToIndices[segmentAccessor[i]].push_back(i);

		std::vector<PositiveSample> result;
		result.reserve(n);

		for (auto& kv : segmentToIndices) {
			std::vector<int64_t>& indices = kv.second;
			for (int anchorPos = 0; anchorPos < (int)indices.size(); anchorPos++) {
				int futureOffset = 0, bucket = 0;
				if (!TryPickBucketFuture(anchorPos, indices.size(), config, rng, futureOffset, bucket))
					continue;

				result.push_back({ indices[anchorPos], indices[anchorPos + futureOffset], bucket });
				switch (bucket) {
				case 0: stats.realizedImmediate++; break;
				case 1: stats.realizedShort++; break;
				case 2: stats.realizedMedium++; break;
				case 3: stats.realizedLong++; break;
				}
			}
		}

		stats.anchorsUsed = result.size();
		if (stats.anchorsUsed > 0) {
			stats.realizedImmediate /= stats.anchorsUsed;
			stats.realizedShort /= stats.anchorsUsed;
			stats.realizedMedium /= stats.anchorsUsed;
			stats.realizedLong /= stats.anchorsUsed;
		}

		return result;
	}

	ContrastiveGoalLearner::ContrastiveGoalLearner(int obsSize, int actionRepresentationSize, const ContrastiveGoalConfig& config, torch::Device device) :
		stateActionEncoder("gcrl_sa", MakeEncoderConfig(obsSize + actionRepresentationSize, config.representationSize), device),
		goalEncoder("gcrl_goal", MakeEncoderConfig(6, config.representationSize), device),
		config(config), device(device), obsSize(obsSize), actionRepresentationSize(actionRepresentationSize) {
		SetLearningRate(config.criticLR);
	}

	Tensor ContrastiveGoalLearner::EncodeStateAction(Tensor states, Tensor actionRepresentations) {
		return stateActionEncoder.Forward(torch::cat({ states, actionRepresentations }, -1), false);
	}

	Tensor ContrastiveGoalLearner::EncodeGoal(Tensor goals) {
		return goalEncoder.Forward(goals, false);
	}

	Tensor ContrastiveGoalLearner::Score(Tensor states, Tensor actionRepresentations, Tensor goals) {
		Tensor sa = EncodeStateAction(states, actionRepresentations);
		Tensor g = EncodeGoal(goals);
		return -torch::norm(sa - g, 2, -1);
	}

	ContrastiveGoalStats ContrastiveGoalLearner::Train(ExperienceTensors& data, std::default_random_engine& rng) {
		ContrastiveGoalStats stats;

		if (
			!data.states.defined() ||
			!data.actions.defined() ||
			!data.achievedGoals.defined() ||
			!data.segmentIds.defined() ||
			data.segmentIds.size(0) == 0
		)
			return stats;

		std::vector<PositiveSample> samples = BuildPositiveSamples(data, config, rng, stats);
		if (samples.empty())
			return stats;

		int64_t miniBatchSize = config.criticMiniBatchSize;
		if (miniBatchSize <= 0)
			miniBatchSize = samples.size();

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

		for (int epoch = 0; epoch < config.criticEpochs; epoch++) {
			std::shuffle(samples.begin(), samples.end(), rng);

			for (int64_t start = 0; start < (int64_t)samples.size(); start += miniBatchSize) {
				int64_t curBatchSize = std::min<int64_t>(miniBatchSize, samples.size() - start);
				if (curBatchSize <= 1)
					continue;

				std::vector<int64_t> anchorIndices(curBatchSize), positiveIndices(curBatchSize);
				for (int64_t i = 0; i < curBatchSize; i++) {
					anchorIndices[i] = samples[start + i].anchorIdx;
					positiveIndices[i] = samples[start + i].positiveIdx;
				}
				stats.trainSamplesUsed += curBatchSize;

				Tensor tAnchorIndices = torch::tensor(anchorIndices, TensorOptions().dtype(kLong));
				Tensor tPositiveIndices = torch::tensor(positiveIndices, TensorOptions().dtype(kLong));

				Tensor states = data.states.index_select(0, tAnchorIndices.to(data.states.device())).to(device);
				Tensor actions = data.actions.index_select(0, tAnchorIndices.to(data.actions.device())).to(device).to(kLong);
				Tensor actionRepresentations = torch::zeros(
					{ curBatchSize, actionRepresentationSize },
					TensorOptions().dtype(kFloat32).device(device)
				).scatter_(1, actions.unsqueeze(1), 1.f);
				Tensor positiveGoals = data.achievedGoals.index_select(0, tPositiveIndices.to(data.achievedGoals.device())).to(device);

				Tensor sa = EncodeStateAction(states, actionRepresentations);
				Tensor g = EncodeGoal(positiveGoals);

				Tensor logits = -torch::cdist(sa, g, 2);
				Tensor labels = torch::arange(curBatchSize, TensorOptions().dtype(kLong).device(device));
				Tensor diag = torch::eye(curBatchSize, TensorOptions().dtype(kBool).device(device));

				Tensor rowLoss = torch::nn::CrossEntropyLoss()(logits, labels);
				Tensor columnLoss = torch::nn::CrossEntropyLoss()(logits.transpose(0, 1), labels);
				Tensor logsumexpRows = torch::logsumexp(logits, 1);
				Tensor logsumexpColumns = torch::logsumexp(logits, 0);
				Tensor logsumexpPenalty = config.logsumexpPenaltyCoeff * 0.5f * (logsumexpRows.pow(2).mean() + logsumexpColumns.pow(2).mean());
				Tensor loss = 0.5f * (rowLoss + columnLoss) + logsumexpPenalty;

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
