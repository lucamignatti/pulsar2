#pragma once

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

namespace RLGC::CrashDebug {
	inline bool Enabled() {
		const char* value = std::getenv("GIGALEARN_CRASH_DEBUG");
		return !(value && (std::string(value) == "0" || std::string(value) == "false" || std::string(value) == "FALSE"));
	}

	inline std::mutex& Mutex() {
		static std::mutex mutex;
		return mutex;
	}

	inline std::ofstream& File() {
		static std::ofstream file([] {
			const char* path = std::getenv("GIGALEARN_CRASH_LOG");
			return path ? path : "gigalearn-crash-debug.log";
		}(), std::ios::out | std::ios::app);
		return file;
	}

	inline uint64_t NextSeq() {
		static std::atomic<uint64_t> seq = 0;
		return ++seq;
	}

	template <typename Fn>
	inline void Log(Fn&& fn) {
		if (!Enabled())
			return;

		std::ostringstream msg;
		fn(msg);

		auto now = std::chrono::steady_clock::now().time_since_epoch();
		auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();

		std::lock_guard<std::mutex> lock(Mutex());
		std::ofstream& file = File();
		if (file.good()) {
			file << "#" << NextSeq() << " t+" << ms << " tid=" << std::this_thread::get_id() << " " << msg.str() << "\n";
			file.flush();
		}
	}
}

#define RG_CRASH_LOG(expr) \
	RLGC::CrashDebug::Log([&](std::ostream& _rgCrashLogStream) { _rgCrashLogStream << expr; })
