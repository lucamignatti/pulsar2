#pragma once
#include "Report.h"
#include <vector>

namespace GGL {
	// Port for the metrics side-channel. The production adapter (MetricSender) forwards metrics to
	// wandb via an embedded Python interpreter; tests use InMemoryMetricSink to capture reports with
	// no interpreter at all. Two adapters (wandb + in-memory) make this a real seam.
	struct RG_IMEXPORT MetricSink {
		virtual void Send(const Report& report) = 0;

		// Identifier for the current run (wandb run id), or empty if the sink has none.
		virtual std::string GetRunID() const { return {}; }

		virtual ~MetricSink() {}
	};

	// Test adapter: captures every report unfiltered (no wandb allowlist) so a test can assert on
	// any metric the trainer emits, and a Learner can be built and run without Python.
	struct RG_IMEXPORT InMemoryMetricSink : MetricSink {
		std::vector<Report> history;

		void Send(const Report& report) override { history.push_back(report); }

		bool Empty() const { return history.empty(); }
		const Report& Last() const { return history.back(); }
	};
}
