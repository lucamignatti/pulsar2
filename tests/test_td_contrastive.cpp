#include "TestHarness.h"

#include <GigaLearnCPP/PPO/PPOLearnerConfig.h>
#include "../GigaLearnCPP/src/private/GigaLearnCPP/PPO/ContrastiveGoalLearner.h"
#include "../GigaLearnCPP/src/private/GigaLearnCPP/Util/Models.h"

#include <torch/torch.h>
#include <cmath>

// FORK2 TD-contrastive (GOALSHORT) tests. The TD branch only runs when r>0 (totalTimesteps>=tdRampSteps),
// so these drive it directly at r=1 and pin:
//   - bootstrap-valid row count == segment-interior AND achieved-goal rows (the within-segment shift +
//     synthetic-scoring-goal exclusion), the single highest-risk piece;
//   - all stats finite (no NaN from the [B,B] soft-value accumulation / multinomial / EMA);
//   - r=0 takes the legacy MC path (TD inactive);
//   - the EMA target lags (drift > 0 after a step) and target params never receive gradient.

using namespace GGL;

static ContrastiveGoalConfig MakeTinyConfig() {
	ContrastiveGoalConfig c;
	c.enabled = true;
	c.representationSize = 8;
	c.phiTailLayerSizes = { 16 };
	c.goalInputSize = 6;
	c.tau = 0.05f;
	c.criticLR = 1e-3f;
	c.criticEpochs = 1;
	c.criticMiniBatchSize = 64;   // >= N -> a single full batch (no shuffle effect on the loss)
	c.infoSubSample = 64;
	c.logsumexpPenaltyCoeff = 0.01f;
	c.vicVar = 1.0f;
	c.vicCov = 0.04f;
	// TD
	c.useTDContrastive = true;
	c.tdContrastiveGamma = 0.9f;
	c.tdEmaHalfLifeIters = 2.0f;
	c.tdEmaTrunk = true;
	c.tdSoftValueActionSamples = 3;
	c.tdRampSteps = 10;           // totalTimesteps=100 -> r=1
	c.tdCollapseEnterFrac = -1.f; // disable the collapse latch for deterministic testing
	c.tdCollapseExitFrac = -1.f;
	return c;
}

// Two segments {1,1,1, 2,2,2}. Within-segment shift valid for rows {0,1,3,4} (rows 2 and 5 are
// segment-terminal). Row 1 is flagged synthetic -> excluded. => 3 bootstrap-valid rows: {0,3,4}.
static ExperienceTensors MakeFixture(int obsSize, int numActions) {
	torch::manual_seed(7);
	const int64_t N = 6;
	ExperienceTensors d;
	d.states = torch::randn({ N, obsSize });
	d.actions = torch::randint(0, numActions, { N }).to(torch::kInt32);
	d.achievedGoals = torch::randn({ N, 6 });
	d.herGoals = torch::randn({ N, 6 });
	// Row 0: HER goal == own achieved next-state  => delta==1 (offset 1).
	d.herGoals.index_put_({ 0 }, d.achievedGoals.index({ 0 }).clone());
	d.scoringGoals = torch::randn({ N, 6 });
	d.gcrlTrainMask = torch::ones({ N }, torch::kUInt8);
	d.gcrlScoringMask = torch::zeros({ N }, torch::kUInt8);
	d.gcrlScoringMask.index_put_({ 1 }, (uint8_t)1); // row 1 synthetic
	d.segmentIds = torch::tensor(std::vector<int64_t>{ 1, 1, 1, 2, 2, 2 });
	d.segmentSteps = torch::tensor(std::vector<int64_t>{ 0, 1, 2, 0, 1, 2 });
	d.actionMasks = torch::ones({ N, numActions }, torch::kUInt8);
	return d;
}

static bool AllFinite(const ContrastiveGoalStats& s) {
	float vals[] = { s.loss, s.rowLoss, s.columnLoss, s.categoricalAccuracy, s.logsumexpMean,
		s.tdBlendR, s.tdReverted, s.tdSoftValueEntropyFrac, s.tdRowLoss, s.tdEmaDrift, s.tdValidBootstrapRows };
	for (float v : vals)
		if (!std::isfinite(v)) return false;
	return true;
}

static void td_bootstrap_valid_count_and_finite() {
	const int obsSize = 10, numActions = 4;
	auto cfg = MakeTinyConfig();
	torch::Device dev(torch::kCPU);

	PartialModelConfig shPartial;
	shPartial.layerSizes = { 16 };
	shPartial.addOutputLayer = false;       // trunk: numOutputs becomes layerSizes.back()
	shPartial.activationType = ModelActivationType::SWISH;
	ModelConfig shConfig = shPartial;
	shConfig.numInputs = obsSize;
	shConfig.numOutputs = 0;
	Model sharedHead("shared_head", shConfig, dev);
	TCHECK(sharedHead.config.numOutputs == 16);

	ContrastiveGoalLearner learner(obsSize, numActions, cfg, dev, &sharedHead, nullptr, "gcrl", false, true);
	TCHECK(learner.useTD);                   // GOALSHORT (useCarGoals=false) + useTDContrastive
	TCHECK(learner.phiTailTarget != nullptr && learner.sharedHeadTarget != nullptr);

	auto data = MakeFixture(obsSize, numActions);
	torch::Tensor nextProbs = torch::full({ 6, numActions }, 1.0f / numActions); // uniform pi

	std::default_random_engine rng(1);
	auto stats = learner.Train(data, rng, /*totalTimesteps=*/100, nextProbs); // r=1

	TCHECK(AllFinite(stats));
	TCHECK(stats.tdBlendR > 0.99f);          // r ramped to 1
	// 3 bootstrap-valid rows: segment-interior {0,1,3,4} minus synthetic {1} = {0,3,4}.
	TCHECK(std::abs(stats.tdValidBootstrapRows - 3.0f) < 1e-3f);
	TCHECK(stats.tdEmaDrift > 0.f);          // target lagged the live update (not equal)
	// soft-value entropy fraction is a valid probability-distribution entropy in [0,1].
	TCHECK(stats.tdSoftValueEntropyFrac >= -1e-5f && stats.tdSoftValueEntropyFrac <= 1.0f + 1e-5f);
	// target nets must never receive gradient (forwarded under NoGrad).
	TCHECK(!learner.phiTailTarget->parameters()[0].grad().defined());
}

static void td_r0_takes_legacy_path() {
	const int obsSize = 10, numActions = 4;
	auto cfg = MakeTinyConfig();
	torch::Device dev(torch::kCPU);

	PartialModelConfig shPartial;
	shPartial.layerSizes = { 16 };
	shPartial.addOutputLayer = false;
	ModelConfig shConfig = shPartial;
	shConfig.numInputs = obsSize;
	shConfig.numOutputs = 0;
	Model sharedHead("shared_head", shConfig, dev);
	ContrastiveGoalLearner learner(obsSize, numActions, cfg, dev, &sharedHead, nullptr, "gcrl", false, true);

	auto data = MakeFixture(obsSize, numActions);
	torch::Tensor nextProbs = torch::full({ 6, numActions }, 1.0f / numActions);

	std::default_random_engine rng(1);
	auto stats = learner.Train(data, rng, /*totalTimesteps=*/0, nextProbs); // r=0 -> legacy MC path

	TCHECK(AllFinite(stats));
	TCHECK(stats.tdBlendR == 0.f);                 // ramp at 0
	TCHECK(stats.tdValidBootstrapRows == 0.f);     // TD branch never ran
	TCHECK(stats.loss > 0.f);                      // MC loss still computed
}

static void td_car_critic_ignores_td() {
	// useCarGoals=true => useTD must be false (CAR stays pure MC), even with useTDContrastive set.
	const int obsSize = 10, numActions = 4;
	auto cfg = MakeTinyConfig();
	torch::Device dev(torch::kCPU);
	PartialModelConfig shPartial; shPartial.layerSizes = { 16 }; shPartial.addOutputLayer = false;
	ModelConfig shConfig = shPartial; shConfig.numInputs = obsSize; shConfig.numOutputs = 0;
	Model sharedHead("shared_head", shConfig, dev);
	ContrastiveGoalLearner car(obsSize, numActions, cfg, dev, &sharedHead, nullptr, "gcrl_car", true, false);
	TCHECK(!car.useTD);
	TCHECK(car.phiTailTarget == nullptr);
}

void RunTDContrastiveTests() {
	RUN_SUITE("TD-contrastive (FORK2)", [] {
		td_bootstrap_valid_count_and_finite();
		td_r0_takes_legacy_path();
		td_car_critic_ignores_td();
	});
}
