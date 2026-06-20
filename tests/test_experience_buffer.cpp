#include "TestHarness.h"

#include <GigaLearnCPP/PPO/ExperienceBuffer.h>
#include <numeric>
#include <unordered_set>

// ExperienceBuffer tests.
//
// Key latent-bug areas:
//   - GetAllBatchesShuffled: covers all data exactly once (no duplicates, no drops)
//   - Overbatching boundary: when expSize < 2*batchSize the very FIRST batch
//     satisfies the "last batch" condition and grows to consume all data.
//   - _GetSamples: index selection must pick the right rows.

// ─── helpers ─────────────────────────────────────────────────────────────────

static GGL::ExperienceBuffer MakeBuffer(int n, int seed = 42) {
    GGL::ExperienceBuffer buf(seed, torch::kCPU);
    // One 1-D tensor of floats [0, 1, …, n-1] as the "states" field.
    // All other fields left undefined; _GetSamples skips undefined tensors.
    buf.data.states = torch::arange(n, torch::kFloat32);
    return buf;
}

// ─── tests ───────────────────────────────────────────────────────────────────

static void expbuf_getsamples_picks_correct_rows() {
    auto buf = MakeBuffer(10);
    int64_t idxArr[] = {0, 3, 7};
    auto samples = buf._GetSamples(idxArr, 3);

    TCHECK(samples.states.size(0) == 3);
    TCHECK_NEAR(samples.states[0].item<float>(), 0.f, 1e-6f);
    TCHECK_NEAR(samples.states[1].item<float>(), 3.f, 1e-6f);
    TCHECK_NEAR(samples.states[2].item<float>(), 7.f, 1e-6f);
}

static void expbuf_getsamples_undefined_fields_stay_undefined() {
    auto buf = MakeBuffer(5);
    // Only 'states' is defined; all others must remain undefined after sampling.
    int64_t idx[] = {0, 1};
    auto s = buf._GetSamples(idx, 2);

    TCHECK(s.states.defined());
    TCHECK(!s.actions.defined());
    TCHECK(!s.advantages.defined());
    TCHECK(!s.herGoals.defined());
}

static void expbuf_batches_cover_all_data_no_overbatching() {
    // With overbatching=false, batches cover only complete multiples of batchSize.
    // expSize=9, batchSize=3 → 3 full batches of 3, covering indices 0–8.
    auto buf = MakeBuffer(9, 7);
    auto batches = buf.GetAllBatchesShuffled(3, false);

    TCHECK(batches.size() == 3);

    std::unordered_set<int> seen;
    for (auto& b : batches) {
        TCHECK(b.states.size(0) == 3);
        for (int i = 0; i < 3; i++) {
            int v = (int)b.states[i].item<float>();
            TCHECK(v >= 0 && v < 9);
            TCHECK(seen.find(v) == seen.end());  // no duplicate
            seen.insert(v);
        }
    }
    TCHECK(seen.size() == 9);  // all 9 covered
}

static void expbuf_batches_drop_remainder_without_overbatching() {
    // expSize=10, batchSize=3, overbatching=false → 3 full batches of 3, drops 1
    auto buf = MakeBuffer(10, 7);
    auto batches = buf.GetAllBatchesShuffled(3, false);

    TCHECK(batches.size() == 3);
    int total = 0;
    for (auto& b : batches) total += b.states.size(0);
    TCHECK(total == 9);  // 10th element dropped
}

static void expbuf_overbatching_extends_last_batch() {
    // expSize=10, batchSize=3, overbatching=true
    // startIdx=0: batch=3; startIdx=3: batch=3; startIdx=6: 6+6=12>10, extend→4
    // total = 10.
    auto buf = MakeBuffer(10, 7);
    auto batches = buf.GetAllBatchesShuffled(3, true);

    int total = 0;
    std::unordered_set<int> seen;
    for (auto& b : batches) {
        total += b.states.size(0);
        for (int i = 0; i < b.states.size(0); i++) {
            int v = (int)b.states[i].item<float>();
            TCHECK(seen.find(v) == seen.end());
            seen.insert(v);
        }
    }
    TCHECK(total == 10);
    TCHECK(seen.size() == 10u);  // all 10 covered
    // Last batch is extended: should not be size 3
    TCHECK(batches.back().states.size(0) == 4);
}

static void expbuf_overbatching_first_batch_edge_case() {
    // When expSize < 2*batchSize, the very FIRST batch satisfies the "last batch"
    // condition and is extended to include all data in a single batch.
    // expSize=5, batchSize=3: 0+3*2=6>5 → curBatchSize=5; only 1 batch.
    auto buf = MakeBuffer(5, 7);
    auto batches = buf.GetAllBatchesShuffled(3, true);

    TCHECK(batches.size() == 1);
    TCHECK(batches[0].states.size(0) == 5);

    // All values present
    std::unordered_set<int> seen;
    for (int i = 0; i < 5; i++)
        seen.insert((int)batches[0].states[i].item<float>());
    TCHECK(seen.size() == 5u);
}

static void expbuf_overbatching_exact_boundary() {
    // When expSize == 2*batchSize, the condition 0+6=6>6 is FALSE:
    // we get exactly 2 equal-sized batches.
    auto buf = MakeBuffer(6, 7);
    auto batches = buf.GetAllBatchesShuffled(3, true);

    TCHECK(batches.size() == 2);
    for (auto& b : batches)
        TCHECK(b.states.size(0) == 3);
}

static void expbuf_shuffled_batches_are_actually_shuffled() {
    // With a fixed seed the same shuffle must be reproducible.
    auto buf1 = MakeBuffer(12, 99);
    auto buf2 = MakeBuffer(12, 99);
    auto b1 = buf1.GetAllBatchesShuffled(4, false);
    auto b2 = buf2.GetAllBatchesShuffled(4, false);

    TCHECK(b1.size() == b2.size());
    // Same seed → same order
    for (size_t i = 0; i < b1.size(); i++) {
        auto eq = b1[i].states.eq(b2[i].states).all().item<bool>();
        TCHECK(eq);
    }

    // Different seed → likely different order (probability 1/12! ≈ 0)
    auto buf3 = MakeBuffer(12, 11);
    auto b3 = buf3.GetAllBatchesShuffled(4, false);
    bool anyDiff = false;
    for (size_t i = 0; i < b1.size() && !anyDiff; i++) {
        auto ne = b1[i].states.ne(b3[i].states).any().item<bool>();
        if (ne) anyDiff = true;
    }
    TCHECK(anyDiff);
}

static void expbuf_gcrl_train_mask_tensor_is_preserved_in_sample() {
    // gcrlTrainMask is a uint8 tensor; sampling by index must preserve it.
    GGL::ExperienceBuffer buf(42, torch::kCPU);
    buf.data.states = torch::arange(4, torch::kFloat32);
    buf.data.gcrlTrainMask = torch::tensor(std::vector<uint8_t>{0, 1, 0, 1},
        torch::TensorOptions().dtype(torch::kUInt8));

    int64_t idx[] = {1, 3};
    auto s = buf._GetSamples(idx, 2);

    TCHECK(s.gcrlTrainMask.defined());
    TCHECK(s.gcrlTrainMask[0].item<uint8_t>() == 1u);
    TCHECK(s.gcrlTrainMask[1].item<uint8_t>() == 1u);
}

void RunExperienceBufferTests() {
    RUN_SUITE("expbuf::getsamples_picks_correct_rows", expbuf_getsamples_picks_correct_rows);
    RUN_SUITE("expbuf::getsamples_undefined_fields_stay_undefined", expbuf_getsamples_undefined_fields_stay_undefined);
    RUN_SUITE("expbuf::batches_cover_all_data_no_overbatching", expbuf_batches_cover_all_data_no_overbatching);
    RUN_SUITE("expbuf::batches_drop_remainder_without_overbatching", expbuf_batches_drop_remainder_without_overbatching);
    RUN_SUITE("expbuf::overbatching_extends_last_batch", expbuf_overbatching_extends_last_batch);
    RUN_SUITE("expbuf::overbatching_first_batch_edge_case", expbuf_overbatching_first_batch_edge_case);
    RUN_SUITE("expbuf::overbatching_exact_boundary", expbuf_overbatching_exact_boundary);
    RUN_SUITE("expbuf::shuffled_batches_are_actually_shuffled", expbuf_shuffled_batches_are_actually_shuffled);
    RUN_SUITE("expbuf::gcrl_train_mask_tensor_is_preserved_in_sample", expbuf_gcrl_train_mask_tensor_is_preserved_in_sample);
}
