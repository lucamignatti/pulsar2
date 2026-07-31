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
target_link_libraries(GigaLearnBot GigaLearnCPP)

# Official RLBot v5 C++ interface (defines the RLBotCPP target). Its CMake uses
# FetchContent to pull matching flatbuffers + RLBot/flatbuffers-schema versions,
# so building GigaLearnRLBot needs network access on a fresh build dir.
add_subdirectory(cpp-interface)

# Standalone RLBot v5 client binary: loads a trained checkpoint via InferUnit and
# plays live matches (RLBotSim or real Rocket League). Deliberately NOT linked into
# GigaLearnBot - the trainer and the play-against-the-bot client are separate concerns.
add_executable(GigaLearnRLBot src/RLBotClient.cpp src/RLBotMain.cpp)
set_target_properties(GigaLearnRLBot PROPERTIES CXX_STANDARD 20)
target_link_libraries(GigaLearnRLBot GigaLearnCPP RLBotCPP)

# Let the VIEWER host an RLBot bot as the opponent inside its own arena (see
# src/VizRLBotServer.h). OFF by default for the same reason GGL_BUILD_TESTS is: the
# training box's `cmake --build build` must stay byte-identical, and the trainer has no
# business linking a bot server. Turn it on for the viz tree:
#     cmake -S . -B build-viz -DGGL_VIZ_RLBOT=ON
option(GGL_VIZ_RLBOT "Viewer can host an RLBot bot as the opponent" OFF)
if (GGL_VIZ_RLBOT)
	target_sources(GigaLearnBot PRIVATE src/VizRLBotServer.cpp)
	target_compile_definitions(GigaLearnBot PRIVATE GGL_VIZ_RLBOT)
	# RLBotCPP for the generated flatbuffer types; its own socket layer is unused,
	# since this is the SERVER side of the protocol and that library is the client.
	target_link_libraries(GigaLearnBot RLBotCPP)
endif()

# Offline unit tests (GigaLearnTests). OFF by default so the training box's
# `cmake --build build` stays byte-identical; enable with -DGGL_BUILD_TESTS=ON.
option(GGL_BUILD_TESTS "Build the GigaLearnTests unit test binary" OFF)
if (GGL_BUILD_TESTS)
	add_subdirectory(tests)
endif()