cmake_minimum_required (VERSION 3.8)

project("RLGymCPP")

# Add all headers and code files
file(GLOB_RECURSE FILES_SRC "src/*.cpp" "src/*.h")
add_library(RLGymCPP STATIC ${FILES_SRC})
target_include_directories(RLGymCPP PUBLIC "src/")

# Set C++ version to 20
set_target_properties(RLGymCPP PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(RLGymCPP PROPERTIES CXX_STANDARD 20)

# Vectorize the env/obs/reward code (the CPU collection hot path) to match the trainer. No -ffast-math:
# obs/reward floats feed the policy, so keep them bit-stable vs existing checkpoints -- -march + -unroll
# only. (Mirrors the RocketSim flags; build host ISA must not exceed the run host.)
if(NOT MSVC)
	target_compile_options(RLGymCPP PRIVATE -march=native -funroll-loops)
endif()

# Include RocketSim
add_subdirectory("RocketSim")
target_link_libraries(RLGymCPP RocketSim)

# Include thread pool library (https://github.com/DeveloperPaul123/thread-pool)
target_include_directories(RLGymCPP PUBLIC "thread_pool")