#include "TestHarness.h"

#include <GigaLearnCPP/PPO/GAE.h>

// GAE (Generalised Advantage Estimation) tests.
//
// Past bugs discovered in this area:
//   - Truncated-value-pred indexing is REVERSED during the backward pass:
//     _truncValPreds[numTruncs - truncCount - 1].  Preds must be supplied in
//     forward (ascending step) order for the reversal to be correct.
//   - Return accumulation uses raw rewards (_rews) while advantage delta uses
//     normalised/clipped curReward; mixing these intentionally but subtly.
//
// These tests pin the numeric outputs so that any change to the backward-pass
// ordering or indexing breaks immediately.

using namespace RLGC;  // TerminalType::NOT_TERMINAL, NORMAL, TRUNCATED

// ─── helpers ─────────────────────────────────────────────────────────────────

static torch::Tensor T(std::initializer_list<float> vals) {
    return torch::tensor(std::vector<float>(vals));
}
static torch::Tensor Ti8(std::initializer_list<int8_t> vals) {
    return torch::tensor(std::vector<int8_t>(vals));
}

struct GAEResult {
    torch::Tensor advantages, targetValues, returns;
    float rewClipPortion;
};

static GAEResult RunGAE(
    std::initializer_list<float> rews,
    std::initializer_list<int8_t> terminals,
    std::initializer_list<float> valPreds,
    std::initializer_list<float> truncValPreds = {},
    float gamma = 0.99f, float lambda = 0.95f,
    float returnStd = 1.0f, float clipRange = -1.0f
) {
    auto tTrunc = truncValPreds.size() > 0
        ? torch::tensor(std::vector<float>(truncValPreds))
        : torch::Tensor{};

    GAEResult r;
    GGL::GAE::Compute(
        T(rews), Ti8(terminals), T(valPreds), tTrunc,
        r.advantages, r.targetValues, r.returns, r.rewClipPortion,
        gamma, lambda, returnStd, clipRange
    );
    return r;
}

static float A(const GAEResult& r, int i) { return r.advantages[i].item<float>(); }
static float Ret(const GAEResult& r, int i) { return r.returns[i].item<float>(); }
static float TV(const GAEResult& r, int i) { return r.targetValues[i].item<float>(); }

// ─── tests ───────────────────────────────────────────────────────────────────

static void gae_single_terminal_step() {
    // One step that immediately terminates.
    // A_0 = r_0 + 0 - V_0 = 1.0 - 0.5 = 0.5
    auto r = RunGAE({1.0f}, {NORMAL}, {0.5f});

    TCHECK_NEAR(A(r, 0), 0.5f, 1e-5f);
    TCHECK_NEAR(Ret(r, 0), 1.0f, 1e-5f);
    TCHECK_NEAR(TV(r, 0), 0.5f + 0.5f, 1e-5f);  // valPred + advantage
}

static void gae_three_step_terminal_episode_known_values() {
    // terminals = [0, 0, NORMAL]; valPreds has 3 entries (no val pred needed after terminal).
    // Manually computed (gamma=0.99, lambda=0.95, gl=0.9405):
    //   A[2] = 1.0 - 0.5 = 0.5
    //   A[1] = (1.0 + 0.99*0.5 - 0.5) + 0.9405*0.5 = 0.995 + 0.47025 = 1.46525
    //   A[0] = 0.995 + 0.9405*1.46525 = 0.995 + 1.37807 ≈ 2.37307
    auto r = RunGAE(
        {1.0f, 1.0f, 1.0f},
        {NOT_TERMINAL, NOT_TERMINAL, NORMAL},
        {0.5f, 0.5f, 0.5f},
        {}, 0.99f, 0.95f
    );

    TCHECK_NEAR(A(r, 2), 0.5f, 1e-4f);
    TCHECK_NEAR(A(r, 1), 1.46525f, 1e-4f);
    TCHECK_NEAR(A(r, 0), 2.37307f, 1e-4f);

    // Returns accumulate raw rewards
    TCHECK_NEAR(Ret(r, 2), 1.0f, 1e-4f);
    TCHECK_NEAR(Ret(r, 1), 1.0f + 1.0f * 0.99f, 1e-4f);   // 1.99
    TCHECK_NEAR(Ret(r, 0), 1.0f + 1.99f * 0.99f, 1e-4f);  // ≈ 2.9701

    // Target values = valPreds + advantages
    TCHECK_NEAR(TV(r, 2), 0.5f + 0.5f, 1e-4f);
}

static void gae_advantage_resets_at_terminal() {
    // Two back-to-back single-step terminal episodes.
    // The terminal boundary must prevent GAE from leaking across episodes.
    auto r = RunGAE(
        {3.0f, 7.0f},
        {NORMAL, NORMAL},
        {1.0f, 2.0f},
        {}, 0.99f, 0.95f
    );

    // A[1] = 7.0 - 2.0 = 5.0
    // A[0] = 3.0 - 1.0 = 2.0  (no carry-over from episode 1)
    TCHECK_NEAR(A(r, 1), 5.0f, 1e-5f);
    TCHECK_NEAR(A(r, 0), 2.0f, 1e-5f);
}

static void gae_single_truncated_step() {
    // One truncated step: nextValPred comes from the truncValPreds tensor.
    // A_0 = (r_0 + gamma*truncVal) - V_0 = (5.0 + 0.9*0.8) - 0.3 = 5.42
    auto r = RunGAE(
        {5.0f},
        {TRUNCATED},
        {0.3f},
        {0.8f}, 0.9f, 0.8f
    );

    TCHECK_NEAR(A(r, 0), 5.42f, 1e-5f);
    // Return: raw reward only (truncation kills the prev-return carry)
    TCHECK_NEAR(Ret(r, 0), 5.0f, 1e-5f);
}

static void gae_truncated_value_preds_consumed_in_reverse_order() {
    // Two TRUNCATED steps.  truncValPreds must be supplied in FORWARD (ascending
    // step-index) order; the GAE backward pass consumes them in REVERSED order
    // via _truncValPreds[numTruncs - truncCount - 1].
    //
    // With truncValPreds = [0.5, 0.8]:
    //   step 1 (consumed first in backward) → uses truncValPreds[1] = 0.8
    //   step 0 (consumed second)            → uses truncValPreds[0] = 0.5
    //
    // A[1] = (1.0 + 0.9*0.8) - 0.3 = 0.72 + 0.7 = 1.42
    // A[0] = (1.0 + 0.9*0.5) - 0.3 = 0.45 + 0.7 = 1.15
    auto r = RunGAE(
        {1.0f, 1.0f},
        {TRUNCATED, TRUNCATED},
        {0.3f, 0.3f},
        {0.5f, 0.8f},  // forward order: pred for step0 first, step1 last
        0.9f, 0.8f
    );

    TCHECK_NEAR(A(r, 1), 1.42f, 1e-5f);
    TCHECK_NEAR(A(r, 0), 1.15f, 1e-5f);
}

static void gae_return_std_scales_rewards_in_advantage() {
    // With returnStd=2, curReward=r/returnStd, so the advantage uses scaled rewards.
    // Return tensor still uses raw rewards.
    //
    // terminals=[NORMAL], r=2.0, V=0.0, returnStd=2.0:
    //   curReward = 2.0/2.0 = 1.0
    //   A = 1.0 - 0.0 = 1.0   (half the raw reward)
    //   Return = 2.0 (raw)
    auto r = RunGAE({2.0f}, {NORMAL}, {0.0f}, {}, 0.99f, 0.95f, 2.0f);

    TCHECK_NEAR(A(r, 0), 1.0f, 1e-5f);
    TCHECK_NEAR(Ret(r, 0), 2.0f, 1e-5f);  // return is raw
}

static void gae_reward_clip_reduces_advantage() {
    // With returnStd=1 and clipRange=0.5, a reward of 5.0 is clipped to 0.5.
    // A = 0.5 - 0.0 = 0.5 (NOT 5.0 - 0.0 = 5.0).
    // Return is also clipped (per-step raw/standardised goes through clamp).
    auto r = RunGAE({5.0f}, {NORMAL}, {0.0f}, {}, 0.99f, 0.95f, 1.0f, 0.5f);

    TCHECK_NEAR(A(r, 0), 0.5f, 1e-5f);
    // rewClipPortion = (totalRew - totalClippedRew) / totalRew = (5 - 0.5) / 5 = 0.9
    TCHECK_NEAR(r.rewClipPortion, 0.9f, 1e-5f);
}

static void gae_no_clipping_when_clip_negative() {
    // clipRange < 0 disables clipping.
    auto r = RunGAE({5.0f}, {NORMAL}, {0.0f}, {}, 0.99f, 0.95f, 1.0f, -1.0f);
    TCHECK_NEAR(A(r, 0), 5.0f, 1e-5f);
    TCHECK_NEAR(r.rewClipPortion, 0.0f, 1e-5f);
}

static void gae_truncation_count_mismatch_throws() {
    // Providing 2 truncValPreds for a sequence with only 1 TRUNCATED step
    // must produce an assertion error (truncCount != numTruncs at the end).
    TCHECK_THROW(RunGAE({1.0f}, {TRUNCATED}, {0.3f}, {0.5f, 0.8f}));
}

static void gae_missing_trunc_val_preds_for_truncated_step_throws() {
    // TRUNCATED step with an empty truncValPreds tensor: must throw because
    // the code calls RG_ERR_CLOSE("GAE encountered a truncated terminal, but has no truncated val pred").
    TCHECK_THROW(RunGAE({1.0f}, {TRUNCATED}, {0.3f}));
}

void RunGAETests() {
    RUN_SUITE("gae::single_terminal_step", gae_single_terminal_step);
    RUN_SUITE("gae::three_step_terminal_episode_known_values", gae_three_step_terminal_episode_known_values);
    RUN_SUITE("gae::advantage_resets_at_terminal", gae_advantage_resets_at_terminal);
    RUN_SUITE("gae::single_truncated_step", gae_single_truncated_step);
    RUN_SUITE("gae::truncated_value_preds_consumed_in_reverse_order", gae_truncated_value_preds_consumed_in_reverse_order);
    RUN_SUITE("gae::return_std_scales_rewards_in_advantage", gae_return_std_scales_rewards_in_advantage);
    RUN_SUITE("gae::reward_clip_reduces_advantage", gae_reward_clip_reduces_advantage);
    RUN_SUITE("gae::no_clipping_when_clip_negative", gae_no_clipping_when_clip_negative);
    RUN_SUITE("gae::truncation_count_mismatch_throws", gae_truncation_count_mismatch_throws);
    RUN_SUITE("gae::missing_trunc_val_preds_for_truncated_step_throws", gae_missing_trunc_val_preds_for_truncated_step_throws);
}
