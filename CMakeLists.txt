cmake_minimum_required (VERSION 3.8)

project("GigaLearnBot")

# Enable -fPIC for all targets: the static libs (RLGymCPP, RocketSim) get folded into
# the shared libGigaLearnCPP, which Linux linkers reject without position-independent
# code (macOS is PIC by default, so this only bites on Linux)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Default to Release: a typeless `cmake ..` builds -O0 with libstdc++ assertions armed,
# which is unusably slow for training AND aborts on the vendored thread pool's benign
# binary-semaphore over-release. Debug must be chosen deliberately.
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "No build type specified; defaulting to Release")
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Multi-GPU HPC path: NCCL (device collectives) + MPI (host reductions and
# NCCL unique-id broadcast). OFF by default so a normal PC configure does not
# need NCCL, CUDAToolkit, or MPI. Does not replace CMAKE_CXX_COMPILER with
# mpicxx (Spectrum MPI's mpicc is xlc; we keep gcc and only link the libs).
option(GGL_USE_MULTI_GPU "Link NCCL + MPI for multi-GPU training" OFF)
if (GGL_USE_MULTI_GPU)
	find_package(Threads REQUIRED)
	find_package(CUDAToolkit REQUIRED)
	find_package(MPI REQUIRED)
	if (NOT MPI_CXX_FOUND)
		message(FATAL_ERROR "GGL_USE_MULTI_GPU=ON requires a C++ MPI implementation")
	endif()
	if (NOT NCCL_ROOT)
		message(FATAL_ERROR "GGL_USE_MULTI_GPU=ON requires -DNCCL_ROOT=<nccl prefix>")
	endif()
	set(_ggl_nccl_lib "${NCCL_ROOT}/lib/libnccl_static.a")
	if (NOT EXISTS "${_ggl_nccl_lib}")
		message(FATAL_ERROR "NCCL static library not found: ${_ggl_nccl_lib}")
	endif()
	add_library(nccl STATIC IMPORTED GLOBAL)
	set_target_properties(nccl PROPERTIES
		IMPORTED_LOCATION "${_ggl_nccl_lib}"
		INTERFACE_INCLUDE_DIRECTORIES "${NCCL_ROOT}/include"
	)
	target_link_libraries(nccl INTERFACE CUDA::cudart Threads::Threads ${CMAKE_DL_LIBS})
	message(STATUS "NCCL: ${_ggl_nccl_lib}")
	message(STATUS "MPI: include=${MPI_CXX_INCLUDE_PATH} libs=${MPI_CXX_LIBRARIES}")
endif()

add_executable(GigaLearnBot src/ExampleMain.cpp)

# Set C++ version to 20
set_target_properties(GigaLearnBot PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(GigaLearnBot PROPERTIES CXX_STANDARD 20)

# Make sure GigaLearnCPP is going to build in the same directory as us
# Otherwise, we won't be able to import it at runtime
set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}")
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}")

# Include RLGymSim_PPO
add_subdirectory(GigaLearnCPP)
target_link_libraries(GigaLearnBot PRIVATE GigaLearnCPP)

# GGL_BUILD_CROSSPLAY: two-bot cross-play evaluator (src/CrossPlay.cpp). Runs a foreign
# GGL-family bot (its own obs builder / arch / tickSkip) against ours inside the certified
# RocketSimV3, for large-n goal-share measurement that real matches cannot supply.
option(GGL_BUILD_CROSSPLAY "Build the cross-play evaluator" OFF)
if (GGL_BUILD_CROSSPLAY)
	add_executable(GigaLearnCrossPlay src/CrossPlay.cpp)
	set_target_properties(GigaLearnCrossPlay PROPERTIES LINKER_LANGUAGE CXX CXX_STANDARD 20)
	target_link_libraries(GigaLearnCrossPlay PRIVATE GigaLearnCPP)
	if (UNIX)
		find_package(Threads REQUIRED)
		target_link_libraries(GigaLearnCrossPlay PRIVATE Threads::Threads)
	endif()
endif()
if (GGL_USE_MULTI_GPU)
	target_compile_definitions(GigaLearnBot PUBLIC USE_MULTI_GPU=1 USE_NCCL=1 USE_MPI=1)
	target_link_libraries(GigaLearnBot PRIVATE Threads::Threads)
endif()

# RLBot v5 client / viz opponent. OFF by default: this tree is a training box
# (GigaLearnBot only). Enable with -DGGL_BUILD_RLBOT=ON after `git submodule update
# --init cpp-interface` (FetchContent needs network on a fresh build dir).
option(GGL_BUILD_RLBOT "Build GigaLearnRLBot and the cpp-interface RLBot v5 client" OFF)
if (GGL_BUILD_RLBOT)
	# Official RLBot v5 C++ interface (defines the RLBotCPP target).
	add_subdirectory(cpp-interface)

	# Standalone RLBot v5 client binary: loads a trained checkpoint via InferUnit and
	# plays live matches (RLBotSim or real Rocket League). Deliberately NOT linked into
	# GigaLearnBot - the trainer and the play-against-the-bot client are separate concerns.
	add_executable(GigaLearnRLBot src/RLBotClient.cpp src/RLBotMain.cpp)
	set_target_properties(GigaLearnRLBot PROPERTIES CXX_STANDARD 20)
	target_link_libraries(GigaLearnRLBot GigaLearnCPP RLBotCPP)
endif()

# Let the VIEWER host an RLBot bot as the opponent inside its own arena (see
# src/VizRLBotServer.h). Requires GGL_BUILD_RLBOT. OFF by default so the training
# box's `cmake --build build` stays trainer-only. Turn it on for the viz tree:
#     cmake -S . -B build-viz -DGGL_BUILD_RLBOT=ON -DGGL_VIZ_RLBOT=ON
option(GGL_VIZ_RLBOT "Viewer can host an RLBot bot as the opponent" OFF)
if (GGL_VIZ_RLBOT)
	if (NOT GGL_BUILD_RLBOT)
		message(FATAL_ERROR "GGL_VIZ_RLBOT=ON requires GGL_BUILD_RLBOT=ON")
	endif()
	target_sources(GigaLearnBot PRIVATE src/VizRLBotServer.cpp)
	target_compile_definitions(GigaLearnBot PRIVATE GGL_VIZ_RLBOT)
	# RLBotCPP for the generated flatbuffer types; its own socket layer is unused,
	# since this is the SERVER side of the protocol and that library is the client.
	target_link_libraries(GigaLearnBot PRIVATE RLBotCPP)
endif()

# Offline unit tests (GigaLearnTests). OFF by default so the training box's
# `cmake --build build` stays byte-identical; enable with -DGGL_BUILD_TESTS=ON.
option(GGL_BUILD_TESTS "Build the GigaLearnTests unit test binary" OFF)
if (GGL_BUILD_TESTS)
	add_subdirectory(tests)
endif()
