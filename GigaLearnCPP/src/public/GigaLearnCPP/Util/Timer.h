#pragma once
#include "../Framework.h"

namespace GGL {
	struct Timer {
		std::chrono::steady_clock::time_point startTime;

		Timer() {
			Reset();
		}

		// Returns elapsed time in seconds
		// (steady_clock consistently: on libstdc++, high_resolution_clock aliases system_clock,
		// which type-mismatches the steady_clock member — libc++ happened to forgive it)
		double Elapsed() {
			auto endTime = std::chrono::steady_clock::now();
			std::chrono::duration<double> elapsed = endTime - startTime;
			return elapsed.count();
		}

		void Reset() {
			startTime = std::chrono::steady_clock::now();
		}
	};
}