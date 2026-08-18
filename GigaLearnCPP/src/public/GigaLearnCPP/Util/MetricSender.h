#pragma once
#include "Report.h"
#include <pybind11/pybind11.h>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace GGL {
	struct RG_IMEXPORT MetricSender {
		std::string curRunID;
		std::string projectName, groupName, runName;
		pybind11::module pyMod;

		MetricSender(std::string projectName = {}, std::string groupName = {}, std::string runName = {}, std::string runID = {});

		RG_NO_COPY(MetricSender);

		void Send(const Report& report);

		~MetricSender();

	private:
		// ASYNC SENDER. wandb.log() runs Python and, when the network path to the wandb
		// backend is degraded, blocks the calling thread for minutes per call (SDK
		// backpressure + 90s HTTP retries). On 2026-08-17 that stalled a 96-rank fleet to
		// 3 iterations per 77 minutes: rank 0 sat inside wandb.log() while every other
		// rank waited at the next lockstep collective — and no phase timer saw it,
		// because the send happened after "Overall Steps/Second" was measured.
		// Send() therefore only enqueues plain C++ data; this worker owns all Python.
		// The main thread hands off the GIL permanently after init (_gilRelease) — it
		// never runs Python again (render mode, the other Python user, has no
		// MetricSender; RenderSender guards itself with gil_scoped_acquire regardless).
		void WorkerLoop();

		std::unique_ptr<pybind11::gil_scoped_release> _gilRelease;
		std::thread _worker;
		std::mutex _mut;
		std::condition_variable _cv;
		std::deque<std::unordered_map<std::string, Report::Val>> _queue;
		bool _stop = false;
		uint64_t _dropped = 0;
		std::atomic<uint64_t> _sent{ 0 };
		std::atomic<bool> _workerDone{ false };
		// Set when import/init failed: the sender degrades to a no-op rather than
		// killing the run (a wandb init timeout took down an 8-node 1B job).
		bool _disabled = false;
	};
}
