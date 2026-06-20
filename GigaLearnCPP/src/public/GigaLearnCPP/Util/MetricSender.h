#pragma once
#include "MetricSink.h"
#include "PythonRuntime.h"
#include <pybind11/pybind11.h>

namespace GGL {
	// wandb adapter for MetricSink: forwards (allowlisted) metrics to Python/wandb.
	struct RG_IMEXPORT MetricSender : MetricSink {
		// Owns the interpreter; declared first so it is constructed before (and destroyed after) pyMod.
		PythonRuntime _pyRuntime;

		std::string curRunID;
		std::string projectName, groupName, runName;
		pybind11::module pyMod;

		MetricSender(std::string projectName = {}, std::string groupName = {}, std::string runName = {}, std::string runID = {});

		RG_NO_COPY(MetricSender);

		void Send(const Report& report) override;
		std::string GetRunID() const override { return curRunID; }

		~MetricSender();
	};
}