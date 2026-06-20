#include "TestHarness.h"

#include <GigaLearnCPP/Util/WelfordStat.h>
#include <cmath>

// WelfordStat / BatchedWelfordStat: pure running-statistics classes used
// for return normalisation and obs normalisation.  Previous bugs showed up
// as silent zero-mean normalisation or NaN JSON round-trips; this suite
// pins the observable behaviour so regressions are caught immediately.

// ─────────────────────────────────────────────────────────────────────────────
// WelfordStat
// ─────────────────────────────────────────────────────────────────────────────

static void welford_empty() {
    // Empty stat must not normalise (mean=0, std=1) so dividing by it is safe.
    GGL::WelfordStat s;
    TCHECK(s.GetMean() == 0.0);
    TCHECK(s.GetSTD() == 1.0);
}

static void welford_count_one_suppresses_mean() {
    // With only 1 sample the variance estimate would be undefined (divides by 0).
    // The guard `count < 2` returns mean=0, std=1 deliberately so that
    // normalisation is a no-op until enough data arrives.
    GGL::WelfordStat s;
    s.Increment({5.0f});
    TCHECK(s.count == 1);
    TCHECK(s.GetMean() == 0.0);   // suppressed, NOT 5.0
    TCHECK(s.GetSTD() == 1.0);    // suppressed, NOT 0
}

static void welford_two_samples_known_values() {
    // samples: 2, 4  → mean=3, population-std = 1, sample-std = sqrt(2)
    GGL::WelfordStat s;
    s.Increment({2.0f, 4.0f});
    TCHECK(s.count == 2);
    TCHECK_NEAR(s.GetMean(), 3.0, 1e-9);
    // Welford uses sample std (divides by count-1 = 1), so std = sqrt(2) ≈ 1.4142
    TCHECK_NEAR(s.GetSTD(), std::sqrt(2.0), 1e-6);
}

static void welford_many_samples_matches_brute_force() {
    GGL::WelfordStat s;
    // Hand-built samples with known mean=10, variance easily computed
    std::vector<float> samples = {4.f, 7.f, 9.f, 10.f, 11.f, 13.f, 16.f};
    s.Increment(samples);

    double bf_mean = 0;
    for (float v : samples) bf_mean += v;
    bf_mean /= samples.size();

    double bf_var = 0;
    for (float v : samples) bf_var += (v - bf_mean) * (v - bf_mean);
    bf_var /= (samples.size() - 1); // sample variance

    TCHECK_NEAR(s.GetMean(), bf_mean, 1e-5);
    TCHECK_NEAR(s.GetSTD(), std::sqrt(bf_var), 1e-5);
}

static void welford_constant_samples_uses_std_floor() {
    // If all samples are identical the variance is exactly 0.
    // GetSTD() must return 1 (not 0) to prevent division-by-zero in normalisation.
    GGL::WelfordStat s;
    s.Increment({3.f, 3.f, 3.f, 3.f});
    TCHECK(s.GetSTD() == 1.0);  // floored, not 0
}

static void welford_reset_clears_state() {
    GGL::WelfordStat s;
    s.Increment({1.f, 2.f, 3.f});
    s.Reset();
    TCHECK(s.count == 0);
    TCHECK(s.GetMean() == 0.0);
    TCHECK(s.GetSTD() == 1.0);
}

static void welford_json_roundtrip() {
    GGL::WelfordStat orig;
    orig.Increment({1.f, 3.f, 5.f, 7.f, 9.f});

    nlohmann::json j = orig.ToJSON();

    GGL::WelfordStat loaded;
    loaded.ReadFromJSON(j);

    TCHECK(loaded.count == orig.count);
    TCHECK_NEAR(loaded.GetMean(), orig.GetMean(), 1e-12);
    TCHECK_NEAR(loaded.GetSTD(), orig.GetSTD(), 1e-12);
}

// ─────────────────────────────────────────────────────────────────────────────
// BatchedWelfordStat
// ─────────────────────────────────────────────────────────────────────────────

static void batched_welford_tracks_columns_independently() {
    // Column 0 gets a uniform-ish distribution; column 1 gets a single spike.
    // They must not cross-pollute each other.
    GGL::BatchedWelfordStat s(2);

    float rows[][2] = {{1.f, 100.f}, {3.f, 100.f}, {5.f, 100.f}, {7.f, 100.f}};
    for (auto& row : rows)
        s.IncrementRow(row);

    auto mean = s.GetMean();
    auto std  = s.GetSTD();

    TCHECK_NEAR(mean[0], 4.0, 1e-5);   // (1+3+5+7)/4 = 4
    TCHECK_NEAR(mean[1], 100.0, 1e-5);

    // Column 1 is constant → std floored to 1.0
    TCHECK(std[1] == 1.0);

    // Column 0 sample std of {1,3,5,7}: mean=4, Σ(x-μ)²=20, /3 → std≈2.582
    double expected_std0 = std::sqrt(20.0 / 3.0);
    TCHECK_NEAR(std[0], expected_std0, 1e-5);
}

static void batched_welford_count_one_suppresses() {
    GGL::BatchedWelfordStat s(3);
    float row[] = {1.f, 2.f, 3.f};
    s.IncrementRow(row);

    TCHECK(s.count == 1);
    // With count < 2 all elements of GetSTD() should be 1
    auto stdv = s.GetSTD();
    for (double d : stdv)
        TCHECK(d == 1.0);
}

static void batched_welford_json_roundtrip() {
    GGL::BatchedWelfordStat orig(4);
    float r1[] = {1.f, 2.f, 3.f, 4.f};
    float r2[] = {5.f, 6.f, 7.f, 8.f};
    float r3[] = {9.f, 10.f, 11.f, 12.f};
    orig.IncrementRow(r1); orig.IncrementRow(r2); orig.IncrementRow(r3);

    nlohmann::json j = orig.ToJSON();
    GGL::BatchedWelfordStat loaded(4);
    loaded.ReadFromJSON(j);

    TCHECK(loaded.count == orig.count);
    TCHECK(loaded.width == orig.width);
    for (int i = 0; i < 4; i++) {
        TCHECK_NEAR(loaded.GetMean()[i], orig.GetMean()[i], 1e-10);
        TCHECK_NEAR(loaded.GetSTD()[i], orig.GetSTD()[i], 1e-10);
    }
}

static void batched_welford_json_oldformat_compat() {
    // Old checkpoints saved the keys "mean" / "var" (singular) instead of
    // "means" / "vars" (plural).  ReadFromJSON has a fallback path for this;
    // make sure it doesn't throw and loads sensible data.
    nlohmann::json j;
    j["mean"] = {1.5, 2.5};
    j["var"]  = {0.25, 1.0};  // Welford's running variance (not sample variance)
    j["count"] = 4;

    GGL::BatchedWelfordStat s(2);
    TCHECK_NOTHROW(s.ReadFromJSON(j));
    TCHECK(s.count == 4);
    TCHECK(s.width == 2);
    // mean[0] should have been read as 1.5
    TCHECK_NEAR(s.GetMean()[0], 1.5, 1e-12);
}

// ─────────────────────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────────────────────

void RunWelfordTests() {
    RUN_SUITE("welford::empty", welford_empty);
    RUN_SUITE("welford::count_one_suppresses_mean", welford_count_one_suppresses_mean);
    RUN_SUITE("welford::two_samples_known_values", welford_two_samples_known_values);
    RUN_SUITE("welford::many_samples_matches_brute_force", welford_many_samples_matches_brute_force);
    RUN_SUITE("welford::constant_samples_uses_std_floor", welford_constant_samples_uses_std_floor);
    RUN_SUITE("welford::reset_clears_state", welford_reset_clears_state);
    RUN_SUITE("welford::json_roundtrip", welford_json_roundtrip);
    RUN_SUITE("batched_welford::tracks_columns_independently", batched_welford_tracks_columns_independently);
    RUN_SUITE("batched_welford::count_one_suppresses", batched_welford_count_one_suppresses);
    RUN_SUITE("batched_welford::json_roundtrip", batched_welford_json_roundtrip);
    RUN_SUITE("batched_welford::json_oldformat_compat", batched_welford_json_oldformat_compat);
}
