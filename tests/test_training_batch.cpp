#include "TestHarness.h"

#include <GigaLearnCPP/PPO/TrainingBatch.h>
#include <GigaLearnCPP/PPO/ExperienceBuffer.h>

// BuildTrainingBatch tests.
//
// This is the seam the old 600-line Learner::Start() slab lacked: the advantage/buffer math
// (GAE wiring, the return-std it reads, the goal-tensor alignment check, and buffer packing)
// is now a pure function of hand-made tensors. The recorded trajectory-discard and
// return-std-coupling bugs lived in this wiring; these tests pin it.

using namespace GGL;
using RLGC::TerminalType;

static torch::Tensor F(std::initializer_list<float> v) { return torch::tensor(std::vector<float>(v)); }
static torch::Tensor I8(std::initializer_list<int8_t> v) { return torch::tensor(std::vector<int8_t>(v)); }
static torch::Tensor I32(std::initializer_list<int32_t> v) { return torch::tensor(std::vector<int32_t>(v)); }

// A 3-step terminal episode, obsSize=2, numActions=2, no clipping (clipRange<0), returnStd=1.
// These are the same numbers as test_gae's three_step_terminal_episode, so a mismatch means
// the unit mis-wired GAE's inputs (terminals / std / clip), not GAE itself.
static TrainingBatchInputs MakeBasicInputs() {
    TrainingBatchInputs in;
    in.states = torch::tensor(std::vector<float>{0,0, 1,1, 2,2}).reshape({3, 2});
    in.actions = I32({0, 1, 0});
    in.logProbs = F({-0.1f, -0.2f, -0.3f});
    in.actionMasks = torch::ones({3, 2}, torch::TensorOptions().dtype(torch::kUInt8));
    in.rewards = F({1.0f, 1.0f, 1.0f});
    in.terminals = I8({TerminalType::NOT_TERMINAL, TerminalType::NOT_TERMINAL, TerminalType::NORMAL});
    in.valPreds = F({0.5f, 0.5f, 0.5f});
    in.gaeGamma = 0.99f;
    in.gaeLambda = 0.95f;
    in.rewardClipRange = -1.0f;
    in.returnStd = 1.0f;
    return in;
}

static void AddAlignedGoals(TrainingBatchInputs& in) {
    in.gcrlEnabled = true;
    in.achievedGoals = torch::zeros({3, 6});
    in.herGoals = torch::zeros({3, 6});
    in.scoringGoals = torch::zeros({3, 6});
    in.gcrlTrainMask = torch::ones({3}, torch::TensorOptions().dtype(torch::kUInt8));
    in.segmentIds = torch::zeros({3}, torch::TensorOptions().dtype(torch::kInt64));
    in.segmentSteps = torch::zeros({3}, torch::TensorOptions().dtype(torch::kInt64));
}

static float At(const torch::Tensor& t, int i) { return t[i].item<float>(); }

// ─── tests ───────────────────────────────────────────────────────────────────

static void training_batch_advantages_match_gae() {
    auto in = MakeBasicInputs();
    ExperienceBuffer buf(0, torch::kCPU);
    auto r = BuildTrainingBatch(in, buf);

    TCHECK_NEAR(At(r.advantages, 2), 0.5f, 1e-4f);
    TCHECK_NEAR(At(r.advantages, 1), 1.46525f, 1e-4f);
    TCHECK_NEAR(At(r.advantages, 0), 2.37307f, 1e-4f);
}

static void training_batch_returns_handed_back_for_stat_update() {
    // The caller owns the WelfordStat update, so the unit must hand back the raw returns.
    auto in = MakeBasicInputs();
    ExperienceBuffer buf(0, torch::kCPU);
    auto r = BuildTrainingBatch(in, buf);

    TCHECK_NEAR(At(r.returns, 2), 1.0f, 1e-4f);
    TCHECK_NEAR(At(r.returns, 1), 1.99f, 1e-4f);      // 1 + 0.99*1
    TCHECK_NEAR(At(r.returns, 0), 2.9701f, 1e-4f);    // 1 + 0.99*1.99
}

static void training_batch_packs_buffer_fields() {
    auto in = MakeBasicInputs();
    ExperienceBuffer buf(0, torch::kCPU);
    auto r = BuildTrainingBatch(in, buf);

    // Each buffer field must receive the matching input/output (not a sibling tensor).
    TCHECK(buf.data.states.equal(in.states));
    TCHECK(buf.data.actions.equal(in.actions));
    TCHECK(buf.data.logProbs.equal(in.logProbs));
    TCHECK(buf.data.actionMasks.equal(in.actionMasks));
    TCHECK(buf.data.advantages.equal(r.advantages));
    TCHECK(buf.data.targetValues.equal(r.targetValues));

    // Goal fields stay empty when GCRL is off.
    TCHECK(!buf.data.achievedGoals.defined());
    TCHECK(!buf.data.gcrlTrainMask.defined());
}

static void training_batch_packs_goal_fields_when_enabled() {
    auto in = MakeBasicInputs();
    AddAlignedGoals(in);
    ExperienceBuffer buf(0, torch::kCPU);
    BuildTrainingBatch(in, buf);

    TCHECK(buf.data.achievedGoals.defined());
    TCHECK(buf.data.achievedGoals.equal(in.achievedGoals));
    TCHECK(buf.data.herGoals.equal(in.herGoals));
    TCHECK(buf.data.scoringGoals.equal(in.scoringGoals));
    TCHECK(buf.data.gcrlTrainMask.equal(in.gcrlTrainMask));
    TCHECK(buf.data.segmentIds.equal(in.segmentIds));
    TCHECK(buf.data.segmentSteps.equal(in.segmentSteps));
}

static void training_batch_misaligned_goals_throw() {
    // A goal tensor that doesn't line up row-for-row with the states must be caught here,
    // not silently fed to the contrastive critic downstream.
    auto in = MakeBasicInputs();
    AddAlignedGoals(in);
    in.achievedGoals = torch::zeros({2, 6});  // 2 rows vs 3 states

    ExperienceBuffer buf(0, torch::kCPU);
    TCHECK_THROW(BuildTrainingBatch(in, buf));
}

void RunTrainingBatchTests() {
    RUN_SUITE("training_batch::advantages_match_gae", training_batch_advantages_match_gae);
    RUN_SUITE("training_batch::returns_handed_back_for_stat_update", training_batch_returns_handed_back_for_stat_update);
    RUN_SUITE("training_batch::packs_buffer_fields", training_batch_packs_buffer_fields);
    RUN_SUITE("training_batch::packs_goal_fields_when_enabled", training_batch_packs_goal_fields_when_enabled);
    RUN_SUITE("training_batch::misaligned_goals_throw", training_batch_misaligned_goals_throw);
}
