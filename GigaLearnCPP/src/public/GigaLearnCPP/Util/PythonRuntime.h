#pragma once
#include <GigaLearnCPP/Framework.h>

namespace GGL {
	// Refcounted owner of the embedded Python interpreter.
	//
	// The pybind adapters (MetricSender, RenderSender) each hold one of these. The interpreter is
	// initialized when the first guard is constructed and finalized when the last is destroyed, so
	// the Python runtime follows the adapters: a Learner with only in-memory sinks never starts it.
	//
	// Not thread-safe: guards are created/destroyed on the main thread during Learner setup/teardown.
	struct RG_IMEXPORT PythonRuntime {
		PythonRuntime();
		~PythonRuntime();

		RG_NO_COPY(PythonRuntime);
	};
}
