cmake_minimum_required (VERSION 3.8)

project("RocketSim")

# Add all headers and code files
file(GLOB_RECURSE FILES_ROCKETSIM "${PROJECT_SOURCE_DIR}/src/*.cpp" "${PROJECT_SOURCE_DIR}/src/*.h")

# Only include bullet headers when using MSVC, otherwise just code files
file(GLOB_RECURSE FILES_BULLET "${PROJECT_SOURCE_DIR}/libsrc/bullet3-3.24/*.cpp" "${PROJECT_SOURCE_DIR}/libsrc/bullet3-3.24/*.h")

add_library(RocketSim ${FILES_ROCKETSIM} ${FILES_BULLET})

set_target_properties(RocketSim PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(RocketSim PROPERTIES CXX_STANDARD 20)

# Match the trainer's vectorization flags. RocketSim/bullet IS the per-step physics engine and the
# dominant CPU cost of collection, but it previously inherited only -O3 -DNDEBUG (scalar vector math on
# the GCC/Clang path) -- it got none of the -march=native auto-vectorization (AVX2/FMA) the rest of the
# trainer enjoys. NOTE: deliberately NO -ffast-math here -- physics determinism (FuzzedKickoffState /
# replay) can be perturbed by float reassociation; -march + -funroll capture the win without that risk.
# -march=native bakes in the build host's ISA, so the build box must not be newer than the run box.
if(NOT MSVC)
	target_compile_options(RocketSim PRIVATE -march=native -funroll-loops)
endif()
