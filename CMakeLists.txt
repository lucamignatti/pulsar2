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

file(GLOB_RECURSE FILES_SRC "src/*.cpp" "src/*.h" "src/*.hpp")
add_executable(GigaLearnBot ${FILES_SRC})

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

# Include RLBot
add_subdirectory(RLBotCPP)
target_link_libraries(GigaLearnBot RLBotCPP)