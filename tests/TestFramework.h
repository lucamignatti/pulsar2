#pragma once
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// TODO(titan): MPI/NCCL Session self-test currently runs inside
// GGL::Dist::Session::Init (GigaLearnCPP/distributed). Move those
// cases here as TEST()s launched with mpirun. Do not GLOB them into
// this single-process runner.

// Minimal hand-rolled test harness (the pre-reset GigaLearnTests pattern):
// TEST(name) { ... } registers a case; CHECK/CHECK_NEAR throw on failure;
// RunAllTests() runs everything and returns a process exit code.

struct TestFailure {
	std::string msg;
};

struct TestCase {
	std::string name;
	std::function<void()> fn;
};

inline std::vector<TestCase>& GetAllTests() {
	static std::vector<TestCase> tests;
	return tests;
}

struct TestRegistrar {
	TestRegistrar(std::string name, std::function<void()> fn) {
		GetAllTests().push_back({ std::move(name), std::move(fn) });
	}
};

#define TEST(name) \
	static void name(); \
	static TestRegistrar _reg_##name(#name, name); \
	static void name()

#define CHECK(cond) \
	do { \
		if (!(cond)) { \
			std::ostringstream _s; \
			_s << __FILE__ << ":" << __LINE__ << " CHECK failed: " #cond; \
			throw TestFailure{ _s.str() }; \
		} \
	} while (0)

#define CHECK_EQ(a, b) \
	do { \
		auto _a = (a); auto _b = (b); \
		if (!(_a == _b)) { \
			std::ostringstream _s; \
			_s << __FILE__ << ":" << __LINE__ << " CHECK_EQ failed: " #a " == " #b " (" << _a << " vs " << _b << ")"; \
			throw TestFailure{ _s.str() }; \
		} \
	} while (0)

#define CHECK_NEAR(a, b, eps) \
	do { \
		auto _a = (a); auto _b = (b); \
		if (!(std::abs(_a - _b) <= (eps))) { \
			std::ostringstream _s; \
			_s << __FILE__ << ":" << __LINE__ << " CHECK_NEAR failed: " #a " ~= " #b " (" << _a << " vs " << _b << ")"; \
			throw TestFailure{ _s.str() }; \
		} \
	} while (0)

inline int RunAllTests() {
	int failed = 0;
	for (auto& test : GetAllTests()) {
		try {
			test.fn();
			std::cout << "[PASS] " << test.name << std::endl;
		} catch (const TestFailure& f) {
			failed++;
			std::cout << "[FAIL] " << test.name << ": " << f.msg << std::endl;
		} catch (const std::exception& e) {
			failed++;
			std::cout << "[FAIL] " << test.name << " (exception): " << e.what() << std::endl;
		}
	}
	// The count printed is PASSES, so label it as such: "FAILED 24/30 tests" read as
	// "24 failed" and hid a 6-failure baseline for days.
	const int total = (int)GetAllTests().size();
	std::cout << (failed ? "FAILED" : "PASSED") << ": " << (total - failed) << "/" << total
	          << " tests passed";
	if (failed)
		std::cout << ", " << failed << " FAILED";
	std::cout << std::endl;
	return failed ? 1 : 0;
}
