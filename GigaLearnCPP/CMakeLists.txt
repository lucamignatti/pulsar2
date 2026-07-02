cmake_minimum_required (VERSION 3.8)

project("GigaLearnCPP")

include_directories("${PROJECT_SOURCE_DIR}/src/")

# Make sure CMake finds libtorch if its in this directory
if (EXISTS "${PROJECT_SOURCE_DIR}/libtorch/")
	message("Using local libtorch folder...")
	list(APPEND CMAKE_PREFIX_PATH "libtorch")
	# Make ultra-sure we can find libtorch if its local
	set(CMAKE_PREFIX_PATH "libtorch/share/cmake/Torch")
endif()

# Help torch's cmake find nvcc on boxes where /usr/local/cuda isn't on PATH
# (enable_language(CUDA) only searches CUDACXX/PATH, not the default toolkit location)
if (NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX} AND EXISTS "/usr/local/cuda/bin/nvcc")
	set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE FILEPATH "nvcc for torch's CUDA detection")
	# Distro compilers routinely outrun nvcc's supported-host-gcc ceiling (e.g. Fedora's
	# GCC 16 vs CUDA 13.2's max of 15) — and past the version gate, nvcc's front-end
	# genuinely cannot parse the too-new libstdc++ headers. Prefer a compat g++ for
	# nvcc's probe (we compile no .cu files ourselves, so it is ONLY used for torch's
	# compiler-identification step; the project still builds with the default compiler).
	# Install one with e.g.: sudo dnf install gcc15-c++
	if (NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND NOT DEFINED ENV{CUDAHOSTCXX})
		foreach(GGL_COMPAT_GXX "/usr/bin/g++-15" "/usr/bin/g++-14" "/usr/bin/g++-13")
			if (EXISTS "${GGL_COMPAT_GXX}")
				message("Using compat host compiler for nvcc's probe: ${GGL_COMPAT_GXX}")
				set(CMAKE_CUDA_HOST_COMPILER "${GGL_COMPAT_GXX}" CACHE FILEPATH "host compiler for nvcc")
				break()
			endif()
		endforeach()
	endif()
	# Belt-and-suspenders for boxes with no compat g++ but a tolerable header mismatch.
	# A user-provided -DCMAKE_CUDA_FLAGS wins over this cache default.
	set(CMAKE_CUDA_FLAGS "-allow-unsupported-compiler" CACHE STRING "CUDA flags (default: bypass nvcc's host-gcc version ceiling)")
endif()

# Add libtorch (https://pytorch.org/cppdocs/installing.html#minimal-example)
# If no standalone libtorch is found, fall back to a pip-installed torch: the wheels
# ship the same TorchConfig.cmake (python3 -c "import torch; print(torch.utils.cmake_prefix_path)")
find_package(Torch QUIET)
if (NOT Torch_FOUND)
	execute_process(
		COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
		OUTPUT_VARIABLE GGL_PY_TORCH_CMAKE_PATH
		OUTPUT_STRIP_TRAILING_WHITESPACE
		ERROR_QUIET
		RESULT_VARIABLE GGL_PY_TORCH_RESULT
	)
	if (GGL_PY_TORCH_RESULT EQUAL 0 AND EXISTS "${GGL_PY_TORCH_CMAKE_PATH}")
		message("Torch not on CMAKE_PREFIX_PATH; using the python-installed torch: ${GGL_PY_TORCH_CMAKE_PATH}")
		list(APPEND CMAKE_PREFIX_PATH "${GGL_PY_TORCH_CMAKE_PATH}")
	endif()
endif()
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Add all headers and code files
file(GLOB_RECURSE FILES_SRC CONFIGURE_DEPENDS "src/*.cpp" "src/*.h" "src/*.hpp" "libsrc/*.cpp" "libsrc/*.h" "libsrc/*.hpp")

add_library(GigaLearnCPP SHARED ${FILES_SRC})
target_compile_definitions(GigaLearnCPP PRIVATE -DWITHIN_GGL)
target_include_directories(GigaLearnCPP PUBLIC "src/public")
target_include_directories(GigaLearnCPP PRIVATE "src/private")

# Include libtorch
target_link_libraries(GigaLearnCPP PRIVATE "${TORCH_LIBRARIES}")

# TORCH_CUDA_LIBRARIES is not reliably exported by every libtorch build (the cu128
# zips in particular), which silently compiled out the CUDA cache-empty path on the
# Linux box — gate on platform instead, like main does
if (NOT APPLE)
	message("Enabling CUDA support...")
	target_compile_definitions(GigaLearnCPP PRIVATE -DRG_CUDA_SUPPORT)
endif()

# Set C++ version to 20
set_target_properties(GigaLearnCPP PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(GigaLearnCPP PROPERTIES CXX_STANDARD 20)

if(NOT MSVC)
    # -fno-finite-math-only restores correct NaN/Inf semantics for nlohmann/json (serialises float infinity).
    target_compile_options(GigaLearnCPP PRIVATE -march=native -ffast-math -fno-finite-math-only -funroll-loops)
endif()

# Include RLGymCPP
add_subdirectory(RLGymCPP)
target_link_libraries(GigaLearnCPP PUBLIC RLGymCPP)

# Include JSON
target_include_directories(GigaLearnCPP PUBLIC "${PROJECT_SOURCE_DIR}/libsrc/json")

# Include python
find_package(Python COMPONENTS Interpreter Development)
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
target_link_libraries(GigaLearnCPP PUBLIC ${PYTHON_LIBRARIES})
message("Found Python:")
message("PYTHON_LIBRARIES: ${PYTHON_LIBRARIES}")
message("PYTHON_INCLUDE_DIRS: ${PYTHON_INCLUDE_DIRS}")
message("Python_RUNTIME_LIBRARY_DIRS: ${Python_RUNTIME_LIBRARY_DIRS}")
message("Python_EXECUTABLE: ${Python_EXECUTABLE}")
add_definitions(-DPY_EXEC_PATH="${Python_EXECUTABLE}") # Give C++ access to the executable path

# Include pybind11
add_subdirectory(pybind11)
target_link_libraries(GigaLearnCPP PUBLIC pybind11::embed)

# MSVC fails to find python DLLs even through they are in my path. Good job MSVC. Well done.
# This copies the the python DLLs to the output directory
if (MSVC)
    file(GLOB PYTHON_DLLS "${Python_RUNTIME_LIBRARY_DIRS}/*.dll")
	message("Adding Python DLLS: ${PYTHON_DLLS}")
    add_custom_command(TARGET GigaLearnCPP
                 POST_BUILD
                 COMMAND ${CMAKE_COMMAND} -E copy_if_different
                 ${PYTHON_DLLS}
                 $<TARGET_FILE_DIR:GigaLearnCPP>)
endif (MSVC)

# Make our python files copy over to our build dir
configure_file("./python_scripts/metric_receiver.py" "../python_scripts/metric_receiver.py" COPYONLY)
configure_file("./python_scripts/render_receiver.py" "../python_scripts/render_receiver.py" COPYONLY)

# MSVC sometimes won't link to the libtorch DLLs unless you do this
# This is also from https://pytorch.org/cppdocs/installing.html#minimal-example
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
	message("Adding TORCH_DLLS: ${TORCH_DLLS}")
    add_custom_command(TARGET GigaLearnCPP
                 POST_BUILD
                 COMMAND ${CMAKE_COMMAND} -E copy_if_different
                 ${TORCH_DLLS}
                 $<TARGET_FILE_DIR:GigaLearnCPP>)
endif (MSVC)