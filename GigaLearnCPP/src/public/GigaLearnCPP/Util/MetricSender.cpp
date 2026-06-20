#include "MetricSender.h"

#include "Timer.h"

namespace py = pybind11;
using namespace GGL;

static bool ShouldSendMetricToWandB(const std::string& key) {
	static const std::unordered_set<std::string> allowed = {
		"Total Timesteps",
		"Total Iterations",
		"Collected Timesteps",
		"Average Step Reward",
		"Episode Length",
		"Clipped Reward Portion",

		"Policy Entropy",
		"Mean KL Divergence",
		"Policy Loss",
		"Critic Loss",
		"SB3 Clip Fraction",
		"Policy Update Magnitude",
		"Critic Update Magnitude",

		"Overall Steps/Second",
		"Collection Steps/Second",
		"Consumption Steps/Second",
		"Collection Time",
		"Consumption Time",
		"PPO Learn Time",

		"HER Selected Offset Mean",
		"HER Selected Offset P50",
		"HER Selected Offset P90",
		"HER Valid Relabel Count",
		"HER Total Relabel Rows",

		"CRL Critic Loss",
		"GCRL Row Loss",
		"GCRL Column Loss",
		"GCRL Categorical Accuracy",
		"GCRL Taken Score Mean",
		"GCRL Baseline Score Mean",
		"GCRL Score Mean",
		"GCRL Score Std",
		"CRL Lambda Effective",
		"CRL Variance Gate",
		"A Policy Std",

		"Player/Ball Touch Ratio",
		"Player/Touch Height",
		"Game/Goal Speed"
	};

	return allowed.find(key) != allowed.end();
}

GGL::MetricSender::MetricSender(std::string _projectName, std::string _groupName, std::string _runName, std::string runID) :
	projectName(_projectName), groupName(_groupName), runName(_runName) {

	RG_LOG("Initializing MetricSender...");

	try {
		pyMod = py::module::import("python_scripts.metric_receiver");
	} catch (std::exception& e) {
		RG_ERR_CLOSE("MetricSender: Failed to import metrics receiver, exception: " << e.what());
	}

	try {
		auto returedRunID = pyMod.attr("init")(PY_EXEC_PATH, projectName, groupName, runName, runID);
		curRunID = returedRunID.cast<std::string>();
		RG_LOG(" > " << (runID.empty() ? "Starting" : "Continuing") << " run with ID : \"" << curRunID << "\"...");

	} catch (std::exception& e) {
		RG_ERR_CLOSE("MetricSender: Failed to initialize in Python, exception: " << e.what());
	}

	RG_LOG(" > MetricSender initalized.");
}

void GGL::MetricSender::Send(const Report& report) {
	py::dict reportDict = {};

	for (auto& pair : report.data)
		if (ShouldSendMetricToWandB(pair.first))
			reportDict[pair.first.c_str()] = pair.second;

	try {
		pyMod.attr("add_metrics")(reportDict);
	} catch (std::exception& e) {
		RG_ERR_CLOSE("MetricSender: Failed to add metrics, exception: " << e.what());
	}
}

GGL::MetricSender::~MetricSender() {
	
}
