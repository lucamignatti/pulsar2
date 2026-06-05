#include "MetricSender.h"

#include "Timer.h"

#include <filesystem>
#include <vector>

#ifdef __APPLE__
#include <mach-o/dyld.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

namespace py = pybind11;
using namespace GGL;

static std::filesystem::path GetExecutableDir() {
#ifdef __APPLE__
	uint32_t size = 0;
	_NSGetExecutablePath(nullptr, &size);
	std::vector<char> buffer(size);
	if (_NSGetExecutablePath(buffer.data(), &size) == 0)
		return std::filesystem::canonical(buffer.data()).parent_path();
#elif defined(__linux__)
	std::vector<char> buffer(4096);
	ssize_t len = readlink("/proc/self/exe", buffer.data(), buffer.size() - 1);
	if (len > 0) {
		buffer[len] = '\0';
		return std::filesystem::canonical(buffer.data()).parent_path();
	}
#endif
	return std::filesystem::current_path();
}

GGL::MetricSender::MetricSender(std::string _projectName, std::string _groupName, std::string _runName, std::string runID) :
	projectName(_projectName), groupName(_groupName), runName(_runName) {

	RG_LOG("Initializing MetricSender...");

	try {
		auto sys = py::module::import("sys");
		auto sysPath = sys.attr("path");
		auto exeDir = GetExecutableDir();
		auto exeDirStr = exeDir.string();
		auto scriptsDirStr = (exeDir / "python_scripts").string();

		// Make helper modules importable regardless of the shell's current working directory.
		sysPath.attr("insert")(0, exeDirStr);
		sysPath.attr("insert")(0, scriptsDirStr);

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
		reportDict[pair.first.c_str()] = pair.second;

	try {
		pyMod.attr("add_metrics")(reportDict);
	} catch (std::exception& e) {
		RG_ERR_CLOSE("MetricSender: Failed to add metrics, exception: " << e.what());
	}
}

GGL::MetricSender::~MetricSender() {
	
}