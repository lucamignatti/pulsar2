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
		"Entropy Scale",
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
		"GCRL Separation",
		"GCRL Baseline Spread",
		"CRL Lambda Effective",
		"CRL Lambda Warmup Progress",
		"A Policy Std",

		"Player/Ball Touch Ratio",
		"Player/Touch Height",
		"Game/Goal Speed"
	};

	// Per-reward component metrics are logged under dynamic "Rewards/<name>" keys,
	// so they can't be listed individually above — allow the whole prefix through.
	if (key.rfind("Rewards/", 0) == 0)
		return true;

	// Skill-tracker Elo is logged under dynamic "Rating/<gamemode>" keys
	// (PolicyVersionManager). Without this it's computed but silently dropped.
	if (key.rfind("Rating/", 0) == 0)
		return true;

	// RSNorm running-obs-normalization summary stats.
	if (key.rfind("RSNorm/", 0) == 0)
		return true;

	// Per-critic GCRL diagnostics (GCRL/Goal Separation, GCRL/Car Separation, ...).
	if (key.rfind("GCRL/", 0) == 0)
		return true;

	// FORK2 striking-quality / income-vs-finishing-vs-scoring decomposition + per-touch metrics.
	if (key.rfind("Striking/", 0) == 0)
		return true;

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
