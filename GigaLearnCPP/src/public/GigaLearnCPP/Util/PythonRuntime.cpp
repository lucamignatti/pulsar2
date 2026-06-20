#include "PythonRuntime.h"

#include <pybind11/embed.h>

// Single-threaded refcount (see header). Guards are acquired/released during Learner ctor/dtor.
static int g_pyRuntimeRefCount = 0;

GGL::PythonRuntime::PythonRuntime() {
	if (g_pyRuntimeRefCount++ == 0)
		pybind11::initialize_interpreter();
}

GGL::PythonRuntime::~PythonRuntime() {
	if (--g_pyRuntimeRefCount == 0)
		pybind11::finalize_interpreter();
}
