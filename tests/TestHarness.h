#pragma once

#include <cstdio>
#include <cmath>
#include <functional>
#include <string>
#include <stdexcept>

// Minimal test harness — no external dependencies.
// Tests are registered with RUN_TEST(fn), checked with CHECK/CHECK_NEAR/CHECK_THROW.

namespace TestHarness {
    inline int failures = 0;
    inline int passes = 0;
    inline int suites_failed = 0;
    inline int suites_passed = 0;

    inline bool Check(bool cond, const char* expr, const char* file, int line) {
        if (cond) {
            ++passes;
            return true;
        } else {
            ++failures;
            fprintf(stderr, "    FAIL [%s:%d]  %s\n", file, line, expr);
            return false;
        }
    }

    inline void RunSuite(const char* name, std::function<void()> fn) {
        printf("[suite] %s\n", name);
        int prevFail = failures;
        try {
            fn();
        } catch (std::exception& e) {
            fprintf(stderr, "    UNCAUGHT EXCEPTION: %s\n", e.what());
            ++failures;
        }
        if (failures == prevFail)
            ++suites_passed;
        else
            ++suites_failed;
    }

    inline int Summary() {
        printf("\n");
        printf("Tests:  %d passed, %d failed\n", passes, failures);
        printf("Suites: %d passed, %d failed\n", suites_passed, suites_failed);
        return failures > 0 ? 1 : 0;
    }
}

// Use TCHECK prefix to avoid conflicts with c10/logging_is_not_google_glog.h
// which defines its own CHECK macro pulled in by torch headers.

#define TCHECK(cond) TestHarness::Check((bool)(cond), #cond, __FILE__, __LINE__)

// Absolute difference within eps.
#define TCHECK_NEAR(a, b, eps) TestHarness::Check( \
    std::abs((double)(a) - (double)(b)) < (double)(eps), \
    #a " ~= " #b " (eps=" #eps ")", __FILE__, __LINE__)

// Relative difference within eps * |b|.
#define TCHECK_REL(a, b, eps) TestHarness::Check( \
    (std::abs((double)(b)) < 1e-30 \
        ? std::abs((double)(a) - (double)(b)) < (eps) \
        : std::abs((double)(a) - (double)(b)) / std::abs((double)(b)) < (eps)), \
    #a " ~rel= " #b " (eps=" #eps ")", __FILE__, __LINE__)

// Verify that expr throws a std::exception (or derived).
#define TCHECK_THROW(expr) do { \
    bool threw = false; \
    try { (void)(expr); } \
    catch (std::exception&) { threw = true; } \
    TestHarness::Check(threw, "throws: " #expr, __FILE__, __LINE__); \
} while(0)

// Verify that expr does NOT throw.
#define TCHECK_NOTHROW(expr) do { \
    bool threw = false; \
    try { (void)(expr); } \
    catch (std::exception& e) { threw = true; fprintf(stderr, "    unexpected throw: %s\n", e.what()); } \
    TestHarness::Check(!threw, "no throw: " #expr, __FILE__, __LINE__); \
} while(0)

#define RUN_SUITE(name, fn) TestHarness::RunSuite(name, fn)
