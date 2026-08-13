cmake_minimum_required (VERSION 3.8)

project("RLGymCPP")

# Add all headers and code files
file(GLOB_RECURSE FILES_SRC "src/*.cpp" "src/*.h")
add_library(RLGymCPP STATIC ${FILES_SRC})
target_include_directories(RLGymCPP PUBLIC "src/")

# Set C++ version to 20
set_target_properties(RLGymCPP PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(RLGymCPP PROPERTIES CXX_STANDARD 20)

# Env stepping / obs building is collection-hot; match GigaLearnCPP's vectorization flags
# (see the note in RocketSim/CMakeLists.txt about -fno-finite-math-only).
if(NOT MSVC)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "ppc64|powerpc")
        target_compile_options(RLGymCPP PRIVATE -mcpu=native -ffast-math -fno-finite-math-only -funroll-loops)
    else()
        target_compile_options(RLGymCPP PRIVATE -march=native -ffast-math -fno-finite-math-only -funroll-loops)
    endif()
endif()

# Engine selection: v3 (vendored Rust engine + compat facade, the default since
# the 2026-07-17 switch) vs the legacy vendored C++ v2.1.1. The v3 compat layer
# still reuses v2's math/constants headers, so the RocketSim/ tree must stay
# present either way.
option(GGL_ROCKETSIM_V3 "Use the vendored v3 Rust engine via the compat layer" ON)
if(GGL_ROCKETSIM_V3)
    add_subdirectory("RocketSimV3")
    target_link_libraries(RLGymCPP RocketSimV3)
    target_compile_definitions(RLGymCPP PUBLIC RG_ROCKETSIM_V3)
else()
    add_subdirectory("RocketSim")
    target_link_libraries(RLGymCPP RocketSim)
endif()

# Include thread pool library (https://github.com/DeveloperPaul123/thread-pool)
target_include_directories(RLGymCPP PUBLIC "thread_pool")