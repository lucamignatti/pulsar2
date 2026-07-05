cmake_minimum_required (VERSION 3.8)

project("RocketSim")

# Add all headers and code files
file(GLOB_RECURSE FILES_ROCKETSIM "${PROJECT_SOURCE_DIR}/src/*.cpp" "${PROJECT_SOURCE_DIR}/src/*.h")

# Only include bullet headers when using MSVC, otherwise just code files
file(GLOB_RECURSE FILES_BULLET "${PROJECT_SOURCE_DIR}/libsrc/bullet3-3.24/*.cpp" "${PROJECT_SOURCE_DIR}/libsrc/bullet3-3.24/*.h")

add_library(RocketSim ${FILES_ROCKETSIM} ${FILES_BULLET})

set_target_properties(RocketSim PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(RocketSim PROPERTIES CXX_STANDARD 20)

# The physics/collision inner loops are the collection hot path but were building at plain
# -O3 while the learner lib got arch-specific vectorization. Mirror GigaLearnCPP's flags
# exactly: -fno-finite-math-only keeps NaN/Inf semantics intact under -ffast-math so the
# obs finiteness guards still work.
if(NOT MSVC)
    target_compile_options(RocketSim PRIVATE -march=native -ffast-math -fno-finite-math-only -funroll-loops)
endif()