#include "MetricSender.h"

#include "Timer.h"

namespace py = pybind11;
using namespace GGL;

GGL::MetricSender::MetricSender(std::string _projectName, std::string _groupName, std::string _runName, std::string runID) :
	projectName(_projectName), groupName(_groupName), runName(_runName) {

	RG_LOG("Initializing MetricSender...");

	// NEITHER of these is fatal any more (2026-08-18). With WANDB_MODE=online on the
	// compute nodes, wandb.init() timed out ("CommError: Run initialization has timed
	// out after 90.0 sec") and the RG_ERR_CLOSE below took down an entire 8-node 1B
	// job at boot — telemetry killing training, the exact failure the async Send path
	// was built to prevent, just moved one function earlier. On failure we degrade to
	// no metrics and train on; _disabled makes Send/dtor no-ops.
	try {
		pyMod = py::module::import("python_scripts.metric_receiver");
	} catch (std::exception& e) {
		RG_LOG("MetricSender: failed to import metrics receiver (" << e.what()
			<< ") - CONTINUING WITHOUT METRICS");
		_disabled = true;
		return;
	}

	try {
		auto returedRunID = pyMod.attr("init")(PY_EXEC_PATH, projectName, groupName, runName, runID);
		curRunID = returedRunID.cast<std::string>();
		RG_LOG(" > " << (runID.empty() ? "Starting" : "Continuing") << " run with ID : \"" << curRunID << "\"...");

	} catch (std::exception& e) {
		RG_LOG("MetricSender: failed to initialize in Python (" << e.what()
			<< ") - CONTINUING WITHOUT METRICS. If this is a wandb timeout on a compute"
			   " node, set WANDB_MODE=offline and sync the run directory afterwards.");
		_disabled = true;
		return;
	}

	// Hand the GIL to the worker for the life of the run (see header). Order matters:
	// release first, then start the worker, so the worker's first gil_scoped_acquire
	// cannot deadlock against a main thread that still holds the GIL.
	_gilRelease = std::make_unique<py::gil_scoped_release>();
	_worker = std::thread(&MetricSender::WorkerLoop, this);

	RG_LOG(" > MetricSender initalized (async).");
}

void GGL::MetricSender::Send(const Report& report) {
	if (_disabled)
		return;
	std::lock_guard<std::mutex> lk(_mut);

	// Bounded queue, drop-OLDEST on overflow: if wandb cannot keep up, training goes on
	// and the freshest reports win. Drops are logged (with power-of-two backoff) so a
	// degraded link shows up in the job log instead of silently thinning the series.
	constexpr size_t MAX_QUEUE = 8;
	if (_queue.size() >= MAX_QUEUE) {
		_queue.pop_front();
		_dropped++;
		if ((_dropped & (_dropped - 1)) == 0)
			RG_LOG("MetricSender: wandb not keeping up - dropped " << _dropped
				<< " report(s) so far (sent " << _sent.load() << ")");
	}
	_queue.push_back(report.data);
	_cv.notify_one();
}

void GGL::MetricSender::WorkerLoop() {
	while (true) {
		std::unordered_map<std::string, Report::Val> item;
		{
			std::unique_lock<std::mutex> lk(_mut);
			_cv.wait(lk, [&] { return _stop || !_queue.empty(); });
			if (_queue.empty()) {
				if (_stop)
					break;
				continue;
			}
			item = std::move(_queue.front());
			_queue.pop_front();
		}

		try {
			py::gil_scoped_acquire gil;
			py::dict reportDict = {};
			for (auto& pair : item)
				reportDict[pair.first.c_str()] = pair.second;
			pyMod.attr("add_metrics")(reportDict);
			_sent++;
		} catch (std::exception& e) {
			// Telemetry failure must never kill training. The old synchronous path
			// RG_ERR_CLOSE'd here, which turned wandb outages into dead runs.
			static std::atomic<uint64_t> errCount{ 0 };
			uint64_t n = ++errCount;
			if ((n & (n - 1)) == 0)
				RG_LOG("MetricSender: add_metrics failed (" << n << "x): " << e.what());
		}
	}

	// Drained (or abandoned) and stopping: flush the wandb run. Runs here, on the
	// worker, so the destructor's bounded wait covers a hung flush too.
	try {
		py::gil_scoped_acquire gil;
		if (pyMod && py::hasattr(pyMod, "finish"))
			pyMod.attr("finish")();
	} catch (std::exception& e) {
		RG_LOG("MetricSender: Failed to finish Python metrics run: " << e.what());
	}
	_workerDone = true;
}

GGL::MetricSender::~MetricSender() {
	if (_disabled)
		return;
	{
		std::lock_guard<std::mutex> lk(_mut);
		_stop = true;
	}
	_cv.notify_all();

	// Bounded shutdown: give the worker a window to drain + wandb.finish(), then abandon
	// it. NEVER hang the exit path on telemetry — a stuck flush would strand the job
	// until SIGKILL and cost the final checkpoint's wall time for nothing.
	Timer deadline = {};
	while (!_workerDone && deadline.Elapsed() < 20.0)
		std::this_thread::sleep_for(std::chrono::milliseconds(50));

	if (_workerDone) {
		if (_worker.joinable())
			_worker.join();
	} else {
		RG_LOG("MetricSender: worker stuck in wandb at shutdown - detaching (last metrics may be lost)");
		if (_worker.joinable())
			_worker.detach();
		// The stuck worker may hold (or be waiting on) the GIL. ~_gilRelease would
		// REACQUIRE it on this thread and hang the exit path — the exact failure this
		// bounded shutdown exists to prevent. Leak it: the interpreter is never
		// finalized and the process is exiting.
		(void)_gilRelease.release();
	}
}
