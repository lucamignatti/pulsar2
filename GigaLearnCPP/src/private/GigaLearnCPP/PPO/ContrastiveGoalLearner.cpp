#include "ContrastiveGoalLearner.h"

#include <torch/nn/modules/loss.h>

using namespace torch;

namespace GGL {

	static ModelConfig MakePhiTailConfig(int embeddingSize, int actionSize, int reprSize, const std::vector<int>& layerSizes) {
		ModelConfig result = PartialModelConfig{};
		result.layerSizes = layerSizes;
		result.activationType = ModelActivationType::SWISH;
		result.optimType = ModelOptimType::ADAM;
		result.addLayerNorm = true;
		result.addOutputLayer = true;
		result.numInputs = embeddingSize + actionSize;
		result.numOutputs = reprSize;
		return result;
	}

	static ModelConfig MakePsiConfig(int inputSize, int reprSize) {
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

	ContrastiveGoalLearner::ContrastiveGoalLearner(int obsSize, int actionRepresentationSize, const ContrastiveGoalConfig& config, torch::Device device,
		Model* sharedHead, const RSNorm* obsNorm,
		const std::string& namePrefix, bool useCarGoals, bool applyTrainMask) :
		phiTailName(namePrefix + "_phi_tail"),
		psiName(namePrefix + "_psi"),
		phiTail(
			phiTailName.c_str(),
			MakePhiTailConfig(sharedHead->config.numOutputs, actionRepresentationSize, config.representationSize, config.phiTailLayerSizes),
			device
		),
		goalEncoder(psiName.c_str(), MakePsiConfig(6, config.representationSize), device),
		sharedHead(sharedHead), obsNorm(obsNorm),
		config(config), device(device), obsSize(obsSize), actionRepresentationSize(actionRepresentationSize),
		useCarGoals(useCarGoals), applyTrainMask(applyTrainMask) {
		SetLearningRate(config.criticLR);
	}

	// embeddings: shared_head(obs).detach() — already normalized and truncated at trunk
	Tensor ContrastiveGoalLearner::EncodeStateAction(Tensor embeddings, Tensor actionRepresentations) {
		Tensor raw = phiTail.Forward(
			torch::cat({ embeddings, actionRepresentations.to(kFloat32).to(embeddings.device()) }, -1),
			false
		);
		return L2Normalize(raw);
	}

	Tensor ContrastiveGoalLearner::EncodeGoal(Tensor goals) {
		Tensor raw = goalEncoder.Forward(goals.to(kFloat32), false);
		return L2Normalize(raw);
	}

	Tensor ContrastiveGoalLearner::Score(Tensor embeddings, Tensor actionRepresentations, Tensor goals) {
		Tensor sa = EncodeStateAction(embeddings, actionRepresentations);
		Tensor g = EncodeGoal(goals);
		return (sa * g).sum(-1) / config.tau;
	}

	ContrastiveGoalStats ContrastiveGoalLearner::Train(ExperienceTensors& data, std::default_random_engine& rng) {
		ContrastiveGoalStats stats;

		torch::Tensor goalsAll = useCarGoals ? data.carHerGoals : data.herGoals;

		if (
			!data.states.defined() ||
			!data.actions.defined() ||
			!goalsAll.defined() ||
			data.states.size(0) == 0
		)
			return stats;

		int64_t n = data.states.size(0);
		if (data.actions.size(0) != n || goalsAll.size(0) != n)
			RG_ERR_CLOSE("ContrastiveGoalLearner::Train(): tensor alignment failed");

		int64_t miniBatchSize = config.criticMiniBatchSize;
		if (miniBatchSize <= 0)
			miniBatchSize = n;
		if (config.infoSubSample > 0)
			miniBatchSize = RS_MIN(miniBatchSize, config.infoSubSample);

		Tensor metricAccum;
		int64_t batches = 0;

		std::vector<int64_t> indices;
		if (applyTrainMask && data.gcrlTrainMask.defined() && data.gcrlTrainMask.size(0) == n) {
			Tensor maskCpu = data.gcrlTrainMask.to(kCPU).contiguous();
			const uint8_t* maskPtr = maskCpu.data_ptr<uint8_t>();
			indices.reserve(n);
			for (int64_t i = 0; i < n; i++)
				if (maskPtr[i])
					indices.push_back(i);
		} else {
			indices.resize(n);
			std::iota(indices.begin(), indices.end(), 0);
		}

		int64_t numTrainRows = (int64_t)indices.size();
		if (numTrainRows <= 1)
			return stats;

		for (int epoch = 0; epoch < config.criticEpochs; epoch++) {
			std::shuffle(indices.begin(), indices.end(), rng);

			for (int64_t start = 0; start < numTrainRows; start += miniBatchSize) {
				int64_t curBatchSize = std::min<int64_t>(miniBatchSize, numTrainRows - start);
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
				Tensor goals = goalsAll.index_select(0, tIndices.to(goalsAll.device())).to(device);

				// Run shared_head with stop-gradient: GCRL trains only the phi tail.
				if (obsNorm)
					states = obsNorm->Normalize(states);
				Tensor embeddings = sharedHead->Forward(states, false).detach();

				Tensor sa = EncodeStateAction(embeddings, actionRepresentations);
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

				phiTail.optim->zero_grad();
				goalEncoder.optim->zero_grad();
				loss.backward();
				phiTail.StepOptim();
				goalEncoder.StepOptim();

				{
					torch::NoGradGuard noGrad;
					Tensor positiveLogits = logits.diag();
					Tensor negativeLogits = logits.masked_select(~diag);
					Tensor rowCorrect = logits.argmax(1).eq(labels).to(kFloat);
					Tensor columnCorrect = logits.argmax(0).eq(labels).to(kFloat);

					Tensor batchMetrics = torch::stack({
						loss.detach(),
						rowLoss.detach(),
						columnLoss.detach(),
						positiveLogits.mean(),
						negativeLogits.mean(),
						sa.norm(2, -1).mean(),
						g.norm(2, -1).mean(),
						0.5f * (rowCorrect.mean() + columnCorrect.mean()),
						0.5f * (logsumexpRows.mean() + logsumexpColumns.mean())
					});
					metricAccum = metricAccum.defined() ? metricAccum + batchMetrics : batchMetrics;
				}
				batches++;
			}
		}

		stats.anchorsUsed = numTrainRows;
		if (batches > 0 && metricAccum.defined()) {
			auto m = (metricAccum / (float)batches).cpu();
			stats.loss = m[0].item<float>();
			stats.rowLoss = m[1].item<float>();
			stats.columnLoss = m[2].item<float>();
			stats.positiveLogitMean = m[3].item<float>();
			stats.negativeLogitMean = m[4].item<float>();
			stats.stateActionEmbeddingNorm = m[5].item<float>();
			stats.goalEmbeddingNorm = m[6].item<float>();
			stats.categoricalAccuracy = m[7].item<float>();
			stats.logsumexpMean = m[8].item<float>();
		}

		return stats;
	}

	void ContrastiveGoalLearner::Save(std::filesystem::path folder, bool saveOptim) {
		phiTail.Save(folder, saveOptim);
		goalEncoder.Save(folder, saveOptim);
	}

	void ContrastiveGoalLearner::Load(std::filesystem::path folder, bool allowNotExist, bool loadOptim) {
		phiTail.Load(folder, allowNotExist, loadOptim);
		goalEncoder.Load(folder, allowNotExist, loadOptim);
	}

	void ContrastiveGoalLearner::SetLearningRate(float lr) {
		phiTail.SetOptimLR(lr);
		goalEncoder.SetOptimLR(lr);
	}
}
