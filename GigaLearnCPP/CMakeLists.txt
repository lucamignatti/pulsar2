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

# Session backends live outside src/ so GLOB_RECURSE cannot compile both.
if (GGL_USE_MULTI_GPU)
	target_sources(GigaLearnCPP PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/distributed/SessionMpiNccl.cpp)
else()
	target_sources(GigaLearnCPP PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/distributed/SessionStub.cpp)
endif()
target_compile_definitions(GigaLearnCPP PRIVATE -DWITHIN_GGL)
target_include_directories(GigaLearnCPP PUBLIC "src/public")
target_include_directories(GigaLearnCPP PRIVATE "src/private")

# Include libtorch
target_link_libraries(GigaLearnCPP PRIVATE "${TORCH_LIBRARIES}")

# NCCL lives in the shared lib (all collectives are in GigaLearnCPP). PRIVATE so
# the static archive is not pulled a second time into GigaLearnBot.
if (GGL_USE_MULTI_GPU)
	target_compile_definitions(GigaLearnCPP PUBLIC USE_MULTI_GPU=1 USE_NCCL=1 USE_MPI=1)
	target_link_libraries(GigaLearnCPP PRIVATE nccl)
	target_link_libraries(GigaLearnCPP PUBLIC Threads::Threads)
	if (TARGET MPI::MPI_CXX)
		target_link_libraries(GigaLearnCPP PRIVATE MPI::MPI_CXX)
	else()
		target_include_directories(GigaLearnCPP PRIVATE ${MPI_CXX_INCLUDE_PATH})
		target_link_libraries(GigaLearnCPP PRIVATE ${MPI_CXX_LIBRARIES})
	endif()
else()
	target_compile_definitions(GigaLearnCPP PUBLIC USE_MULTI_GPU=0 USE_NCCL=0 USE_MPI=0)
endif()

# TORCH_CUDA_LIBRARIES is not reliably exported by every libtorch build (the cu128
# zips in particular), which silently compiled out the CUDA cache-empty path on the
# Linux box — gate on platform instead, like main does
if (NOT APPLE)
	message("Enabling CUDA support...")
	target_compile_definitions(GigaLearnCPP PRIVATE -DRG_CUDA_SUPPORT)
endif()

# GGL_MOE_KERNELS (research/reports/MOE_SPEED.md Stage 1): fused MoE routing +
# sm_70 CUTLASS grouped GEMM for the collect forward. CLUSTER BUILDS ONLY — the
# desktop's CUDA 13 cannot even target sm_70, and the desktop keeps the baddbmm
# path. Needs -DGGL_CUTLASS_INCLUDE=<cutlass>/include (on AiMOS:
# barn-shared/pytorch/third_party/cutlass/include).
option(GGL_MOE_KERNELS "Build fused MoE CUDA kernels (sm_70, cluster only)" OFF)
if (GGL_MOE_KERNELS)
	enable_language(CUDA)
	set(CMAKE_CUDA_ARCHITECTURES 70)
	set(GGL_CUTLASS_INCLUDE "" CACHE PATH "CUTLASS include directory")
	if (NOT EXISTS "${GGL_CUTLASS_INCLUDE}/cutlass/cutlass.h")
		message(FATAL_ERROR "GGL_MOE_KERNELS needs -DGGL_CUTLASS_INCLUDE pointing at a CUTLASS include dir")
	endif()
	target_sources(GigaLearnCPP PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}/src/private/GigaLearnCPP/Util/MoEKernels.cu)
	set_source_files_properties(
		${CMAKE_CURRENT_SOURCE_DIR}/src/private/GigaLearnCPP/Util/MoEKernels.cu
		PROPERTIES INCLUDE_DIRECTORIES "${GGL_CUTLASS_INCLUDE}"
		COMPILE_OPTIONS "--expt-relaxed-constexpr")
	target_compile_definitions(GigaLearnCPP PRIVATE -DGGL_MOE_KERNELS)
	set_target_properties(GigaLearnCPP PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
endif()

# Set C++ version to 20
set_target_properties(GigaLearnCPP PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(GigaLearnCPP PROPERTIES CXX_STANDARD 20)

if(NOT MSVC)
    # -fno-finite-math-only restores correct NaN/Inf semantics for nlohmann/json (serialises float infinity).
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "ppc64|powerpc")
        target_compile_options(GigaLearnCPP PRIVATE -mcpu=native -ffast-math -fno-finite-math-only -funroll-loops)
    else()
        target_compile_options(GigaLearnCPP PRIVATE -march=native -ffast-math -fno-finite-math-only -funroll-loops)
    endif()
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