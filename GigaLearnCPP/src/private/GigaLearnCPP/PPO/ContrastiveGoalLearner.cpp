#include "ContrastiveGoalLearner.h"

#include <torch/nn/modules/loss.h>
#include <cmath>

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

	static ModelConfig MakePsiConfig(int inputSize, int reprSize, const std::vector<int>& layerSizes) {
		ModelConfig result = PartialModelConfig{};
		result.layerSizes = layerSizes;
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

	// FORK2: lerp dst toward src by coeff (coeff=1 => hard copy). Param order matches because the two
	// Models are built from the same ModelConfig. Caller guards with NoGrad.
	static void LerpModelParams(Model* dst, Model* src, float coeff) {
		auto dp = dst->parameters();
		auto sp = src->parameters();
		for (size_t i = 0; i < dp.size(); i++)
			dp[i].copy_(dp[i] * (1.0f - coeff) + sp[i] * coeff, true);
	}

	ContrastiveGoalLearner::ContrastiveGoalLearner(int obsSize, int actionRepresentationSize, const ContrastiveGoalConfig& config, torch::Device device,
		Model* sharedHead, const RSNorm* obsNorm,
		const std::string& namePrefix, bool useCarGoals, bool applyTrainMask, bool useApproachGoals) :
		phiTailName(namePrefix + "_phi_tail"),
		psiName(namePrefix + "_psi"),
		phiTail(
			phiTailName.c_str(),
			MakePhiTailConfig(sharedHead->config.numOutputs, actionRepresentationSize, config.representationSize, config.phiTailLayerSizes),
			device
		),
		goalEncoder(psiName.c_str(), MakePsiConfig(useCarGoals ? config.carGoalInputSize : config.goalInputSize, config.representationSize, config.psiLayerSizes), device),
		sharedHead(sharedHead), obsNorm(obsNorm),
		config(config), device(device), obsSize(obsSize), actionRepresentationSize(actionRepresentationSize),
		useCarGoals(useCarGoals), useApproachGoals(useApproachGoals), applyTrainMask(applyTrainMask) {
		SetLearningRate(config.criticLR);

		// FORK2: TD-contrastive is GOALSHORT/REACH-only (CAR stays pure MC). Build the EMA target nets
		// (phi tail + goal encoder, + the trunk when tdEmaTrunk) with DISTINCT names so Save never collides.
		// Params are lazily synced live->target on the first Train (handles fresh + resume identically).
		useTD = config.useTDContrastive && !useCarGoals && !useApproachGoals; // TD is GOAL-critic only
		if (useTD) {
			phiTailTgtName = namePrefix + "_phi_tail_tgt";
			psiTgtName = namePrefix + "_psi_tgt";
			trunkTgtName = namePrefix + "_trunk_tgt";
			phiTailTarget = new Model(phiTailTgtName.c_str(),
				MakePhiTailConfig(sharedHead->config.numOutputs, actionRepresentationSize, config.representationSize, config.phiTailLayerSizes),
				device);
			goalEncoderTarget = new Model(psiTgtName.c_str(),
				MakePsiConfig(useCarGoals ? config.carGoalInputSize : config.goalInputSize, config.representationSize, config.psiLayerSizes), device);
			if (config.tdEmaTrunk)
				sharedHeadTarget = new Model(trunkTgtName.c_str(), sharedHead->config, device);
		}
	}

	ContrastiveGoalLearner::~ContrastiveGoalLearner() {
		delete phiTailTarget;
		delete goalEncoderTarget;
		delete sharedHeadTarget;
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

	ContrastiveGoalStats ContrastiveGoalLearner::Train(ExperienceTensors& data, std::default_random_engine& rng,
		uint64_t totalTimesteps, torch::Tensor nextPolicyProbs) {
		ContrastiveGoalStats stats;

		torch::Tensor goalsAll = useApproachGoals ? data.approachHerGoals
			: useCarGoals ? data.carHerGoals
			: data.herGoals;

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

		// ── FORK2 TD-contrastive setup (GOALSHORT/REACH only) ──
		// TD is active iff: this critic is GOALSHORT (useTD), the policy-probs + segmentIds are provided,
		// and the ramp is past 0. The collapse latch (set last iter) forces r=0 as a 1-iter safety valve.
		const bool tdOn = useTD
			&& nextPolicyProbs.defined() && nextPolicyProbs.size(0) == n
			&& data.segmentIds.defined() && data.segmentIds.size(0) == n
			&& data.achievedGoals.defined() && data.achievedGoals.size(0) == n;
		const float rRamp = tdOn
			? std::clamp((float)totalTimesteps / (float)std::max<uint64_t>(config.tdRampSteps, 1), 0.f, 1.f)
			: 0.f;
		const float rEff = tdCollapseLatched ? 0.f : rRamp;
		const bool runTD = tdOn && rEff > 0.f;

		// Lazy one-time sync live->target (handles fresh + resume: targets start == current live params,
		// then EMA-lag). Done before the first target forward.
		if (tdOn && !tdTargetsSynced) {
			RG_NO_GRAD;
			LerpModelParams(phiTailTarget, &phiTail, 1.0f);
			LerpModelParams(goalEncoderTarget, &goalEncoder, 1.0f);
			if (config.tdEmaTrunk && sharedHeadTarget)
				LerpModelParams(sharedHeadTarget, sharedHead, 1.0f);
			tdTargetsSynced = true;
		}

		// TD stat accumulators (tensor-accumulated, .item()'d once at the end to avoid per-minibatch syncs):
		// [0]=sum_i HkFrac_i*valid_i, [1]=sum_i valid_i, [2]=sum_batches tdTerm
		torch::Tensor tdAccum;
		int64_t tdBatches = 0;

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

				// NOTE: deliberately NO BF16 autocast here. The dominant InfoNCE op is the [B,B]
				// softmax/CrossEntropy, which autocast always runs in fp32 -- so BF16 only speeds the one
				// K=128 matmul while forcing fp32 cast kernels around the dominant op. Measured NET SLOWER
				// on this path (the one real change in the 2048+BF16 run; the 2048 was a clamped no-op).
				// BF16 stays on the dense PPO update where it is a genuine win.

				// Run shared_head with stop-gradient: GCRL trains only the phi tail.
				if (obsNorm)
					states = obsNorm->Normalize(states);
				Tensor embeddings = sharedHead->Forward(states, false).detach();

				// Raw (pre-L2-norm) embeddings kept for VICReg; L2-normalized for the cosine logits.
				Tensor rawSa = phiTail.Forward(
					torch::cat({ embeddings, actionRepresentations.to(kFloat32).to(embeddings.device()) }, -1),
					false
				);
				Tensor rawG = goalEncoder.Forward(goals.to(kFloat32), false);
				Tensor sa = L2Normalize(rawSa);
				Tensor g = L2Normalize(rawG);

				Tensor logits = torch::matmul(sa, g.transpose(0, 1)) / config.tau;
				Tensor labels = torch::arange(curBatchSize, TensorOptions().dtype(kLong).device(device));
				Tensor diag = torch::eye(curBatchSize, TensorOptions().dtype(kBool).device(device));

				Tensor rowLoss = torch::nn::CrossEntropyLoss()(logits, labels);
				Tensor columnLoss = torch::nn::CrossEntropyLoss()(logits.transpose(0, 1), labels);
				Tensor logsumexpRows = torch::logsumexp(logits, 1);
				Tensor logsumexpColumns = torch::logsumexp(logits, 0);
				Tensor logsumexpPenalty = config.logsumexpPenaltyCoeff * (logsumexpRows.pow(2).mean() + logsumexpColumns.pow(2).mean());

				// TRIAD-NATIVE VICReg anti-collapse on the PRE-L2-norm RAW embeddings (the unit-variance
				// hinge is unsatisfiable on the normalized unit sphere). var = hinge(1 - per-dim std);
				// cov = off-diagonal Gram of the centered batch, normalized by reprDim.
				float invBM1 = 1.0f / (float)std::max<int64_t>(curBatchSize - 1, (int64_t)1);
				float reprD = (float)config.representationSize;
				Tensor varTerm = torch::relu(1.0f - rawSa.std(0, false)).mean()
					+ torch::relu(1.0f - rawG.std(0, false)).mean();
				Tensor cSa = rawSa - rawSa.mean(0, true);
				Tensor cG = rawG - rawG.mean(0, true);
				Tensor covSa = torch::matmul(cSa.transpose(0, 1), cSa) * invBM1;
				Tensor covG = torch::matmul(cG.transpose(0, 1), cG) * invBM1;
				Tensor covTerm =
					(covSa.pow(2).sum() - covSa.diagonal().pow(2).sum()) / reprD +
					(covG.pow(2).sum() - covG.diagonal().pow(2).sum()) / reprD;
				Tensor vicPenalty = config.vicVar * varTerm + config.vicCov * covTerm;

				Tensor mcSymmetric = rowLoss + columnLoss;
				Tensor loss;

				if (!runTD) {
					// Byte-identical legacy path (CAR, TD-off, r==0, or collapse-latched).
					loss = rowLoss + columnLoss + logsumexpPenalty + vicPenalty;
				} else {
					// ── one-step within-segment shift: s'_i = states[i+1] iff same segment ──
					Tensor nextIdx = tIndices + 1;                                   // host [B]
					Tensor inBounds = nextIdx.lt(n);
					Tensor safeNext = torch::where(inBounds, nextIdx, tIndices);     // host [B]
					Tensor segCur = data.segmentIds.index_select(0, tIndices);       // host int64 [B]
					Tensor segNext = data.segmentIds.index_select(0, safeNext);
					Tensor validHost = inBounds & segNext.eq(segCur);                // host bool [B]
					if (data.gcrlScoringMask.defined() && data.gcrlScoringMask.size(0) == n)
						// synthetic scoring-goal rows are MC-only (OOD-bootstrap guard): mask==0 => achieved
						validHost = validHost & data.gcrlScoringMask.index_select(0, tIndices).eq(0);
					Tensor bootstrapValid = validHost.to(kFloat32).to(device);       // device [B]

					Tensor sNext = data.states.index_select(0, safeNext).to(device); // [B, obsSize]
					if (obsNorm)
						sNext = obsNorm->Normalize(sNext);
					Tensor nextProbs = nextPolicyProbs.index_select(0, safeNext).to(device).clamp_min(1e-12f); // [B,A]

					// delta_i = (HER goal == own achieved next-state)  <=>  HER offset == 1
					Tensor achievedRows = data.achievedGoals.index_select(0, tIndices).to(device); // [B,6]
					Tensor delta = (goals - achievedRows).abs().sum(-1).lt(1e-5f).to(kFloat32);     // [B]

					Tensor yPos, pOffTgt, hkFrac;
					{
						RG_NO_GRAD;
						Model* trunkTgt = (config.tdEmaTrunk && sharedHeadTarget) ? sharedHeadTarget : sharedHead;
						Tensor embNext = trunkTgt->Forward(sNext, false).detach();                       // [B, embed]
						Tensor gTgt = L2Normalize(goalEncoderTarget->Forward(goals.to(kFloat32), false)); // [B, repr]

						int K = std::max(1, config.tdSoftValueActionSamples);
						Tensor sampled = torch::multinomial(nextProbs, K, true);                          // [B,K] ~ pi(.|s')
						Tensor runningBB;                                                                 // [B,B] online logsumexp
						std::vector<Tensor> innerDiag;
						innerDiag.reserve(K);
						for (int k = 0; k < K; k++) {
							Tensor aK = sampled.select(1, k);                                             // [B]
							Tensor oneH = torch::zeros({ curBatchSize, actionRepresentationSize }, TensorOptions().dtype(kFloat32).device(device))
								.scatter_(1, aK.unsqueeze(1), 1.f);
							Tensor saK = L2Normalize(phiTailTarget->Forward(torch::cat({ embNext, oneH }, -1), false)); // [B,repr]
							Tensor scoreBB = torch::matmul(saK, gTgt.transpose(0, 1)) / config.tau;       // [B,B] f^-(s',a_k,g_j)
							runningBB = runningBB.defined() ? torch::logaddexp(runningBB, scoreBB) : scoreBB;
							innerDiag.push_back(scoreBB.diagonal());                                      // [B] f^-(s',a_k,g_i)
						}
						// log E_{a'~pi} exp f^- per goal (importance form: a_k ~ pi => uniform 1/K average)
						Tensor softLogitBB = runningBB - std::log((float)K);                              // [B,B]
						Tensor tgtRowProb = torch::softmax(softLogitBB, 1);                               // [B,B] target classifier
						Tensor pDiagTgt = tgtRowProb.diagonal();                                          // [B]
						Tensor gammaEff = config.tdContrastiveGamma * bootstrapValid;                     // [B]
						yPos = (1.f - config.tdContrastiveGamma) * delta + gammaEff * pDiagTgt;           // [B] in (0,1)
						Tensor eyeOff = 1.f - torch::eye(curBatchSize, TensorOptions().dtype(kFloat32).device(device));
						Tensor offProb = tgtRowProb * eyeOff;
						pOffTgt = offProb / offProb.sum(1, true).clamp_min(1e-12f);                       // [B,B] off-diag, row-normed

						// collapse signal: entropy of the soft-value action posterior toward the OWN goal
						Tensor wK = torch::softmax(torch::stack(innerDiag, 1), 1);                        // [B,K]
						Tensor Hk = -(wK * wK.clamp_min(1e-12f).log()).sum(1);                            // [B]
						hkFrac = Hk / std::log((float)std::max(K, 2));                                    // [B] in [0,1]
					}

					// ROW-ONLY soft-CE vs the (detached) bootstrap target dist; grad flows only via LIVE logits.
					Tensor logSoftRow = torch::log_softmax(logits, 1);                                    // [B,B] live
					Tensor eyeB = torch::eye(curBatchSize, TensorOptions().dtype(kFloat32).device(device));
					Tensor tgtDist = yPos.unsqueeze(1) * eyeB + (1.f - yPos).unsqueeze(1) * pOffTgt;       // [B,B] detached
					Tensor tdRowPer = -(tgtDist * logSoftRow).sum(1);                                      // [B]
					Tensor tdTerm = (bootstrapValid * tdRowPer).sum() / bootstrapValid.sum().clamp_min(1.f);

					loss = (1.f - rEff) * mcSymmetric + rEff * tdTerm + logsumexpPenalty + vicPenalty;

					{
						RG_NO_GRAD;
						Tensor batchTd = torch::stack({
							(hkFrac * bootstrapValid).sum(),
							bootstrapValid.sum(),
							tdTerm.detach()
						});
						tdAccum = tdAccum.defined() ? tdAccum + batchTd : batchTd;
						tdBatches++;
					}
				}

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

		// ── FORK2: per-ITERATION EMA target update (NOT per minibatch: tdEmaDecay 0.005/step over
		// thousands of steps/iter would give no lag). lerp coeff = 1 - 2^(-1/halfLifeIters). ──
		if (tdOn) {
			RG_NO_GRAD;
			float emaCoeff = 1.0f - std::pow(2.0f, -1.0f / std::max(config.tdEmaHalfLifeIters, 1e-3f));
			LerpModelParams(phiTailTarget, &phiTail, emaCoeff);
			LerpModelParams(goalEncoderTarget, &goalEncoder, emaCoeff);
			if (config.tdEmaTrunk && sharedHeadTarget)
				LerpModelParams(sharedHeadTarget, sharedHead, emaCoeff);

			// drift observability: ||phiTail^- - phiTail|| / ||phiTail||
			double driftSq = 0, liveSq = 0;
			auto lp = phiTail.parameters();
			auto tp = phiTailTarget->parameters();
			for (size_t i = 0; i < lp.size(); i++) {
				driftSq += (tp[i] - lp[i]).pow(2).sum().item<double>();
				liveSq += lp[i].pow(2).sum().item<double>();
			}
			stats.tdEmaDrift = (float)(std::sqrt(driftSq) / std::max(std::sqrt(liveSq), 1e-6));
		}

		// TD aggregate stats + collapse-latch update (hysteresis; applied with 1-iter latency next call).
		stats.tdBlendR = tdOn ? rEff : -1.f;
		stats.tdReverted = (tdOn && tdCollapseLatched) ? 1.f : 0.f;
		if (tdOn && tdAccum.defined() && tdBatches > 0) {
			auto t = tdAccum.cpu();
			double entWeightedSum = t[0].item<double>();
			double validRowSum = t[1].item<double>();
			double tdLossSum = t[2].item<double>();
			float entFrac = (validRowSum > 0) ? (float)(entWeightedSum / validRowSum) : 0.f;
			stats.tdSoftValueEntropyFrac = entFrac;
			stats.tdRowLoss = (float)(tdLossSum / (double)tdBatches);
			stats.tdValidBootstrapRows = (float)(validRowSum / (double)tdBatches);
			// hysteretic latch: trip on low entropy (soft value collapsing to greedy), release above exit
			if (entFrac < config.tdCollapseEnterFrac)
				tdCollapseLatched = true;
			else if (entFrac > config.tdCollapseExitFrac)
				tdCollapseLatched = false;
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
