#pragma once
#include "Report.h"
#include <pybind11/pybind11.h>

namespace GGL {
	struct RG_IMEXPORT MetricSender {
		std::string curRunID;
		std::string projectName, groupName, runName;
		pybind11::module pyMod;

		// Transient send failures (e.g. a wandb/network hiccup) are logged and skipped;
		// only this many failures in a row are treated as fatal.
		static constexpr int MAX_CONSECUTIVE_SEND_FAILURES = 10;
		int consecutiveSendFailures = 0;

		MetricSender(std::string projectName = {}, std::string groupName = {}, std::string runName = {}, std::string runID = {});
		
		RG_NO_COPY(MetricSender);

		void Send(const Report& report);

		~MetricSender();
	};
}