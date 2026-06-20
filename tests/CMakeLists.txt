cmake_minimum_required(VERSION 3.8)
project(GigaLearnTests)

file(GLOB TEST_SOURCES "*.cpp")
add_executable(GigaLearnTests ${TEST_SOURCES})

set_target_properties(GigaLearnTests PROPERTIES
    LINKER_LANGUAGE CXX
    CXX_STANDARD 20
)

if (NOT MSVC)
    target_compile_options(GigaLearnTests PRIVATE -march=native)
endif()

# Torch headers and libs aren't transitively exposed because GigaLearnCPP links
# them PRIVATE.  Re-run find_package(Torch) in this scope so the 'torch'
# imported target is created here too, giving us both the include paths and the
# linker flags without duplicating the ROCm-workaround logic.
#
# On Linux/ROCm, GigaLearnCPP's ggl_prepare_rocm_torch_dependencies() creates
# stubs for ROCm targets before its find_package(Torch).  Library-backed stubs
# (hipfft, roctracer, etc.) are IMPORTED GLOBAL and visible here.  But
# hip::amdhip64 is typically created by find_package(hip CONFIG QUIET) which
# produces a directory-scoped IMPORTED target visible only inside GigaLearnCPP.
# Torch's Caffe2Targets.cmake uses a plain-variable include guard so it
# re-runs in this directory scope and references hip::amdhip64 — which CMake
# then can't resolve at generate time.  Create a stub here if needed.
if(Torch_DIR)
    if(NOT APPLE AND NOT WIN32 AND NOT TARGET hip::amdhip64)
        add_library(hip::amdhip64 INTERFACE IMPORTED)
    endif()
    find_package(Torch QUIET)
    if(Torch_FOUND)
        target_link_libraries(GigaLearnTests "${TORCH_LIBRARIES}")
        target_include_directories(GigaLearnTests PRIVATE ${TORCH_INCLUDE_DIRS})
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
    else()
        # Fallback: derive include path from cached Torch_DIR and link the lib directly.
        get_filename_component(_GGL_TORCH_INSTALL "${Torch_DIR}/../../../" ABSOLUTE)
        target_include_directories(GigaLearnTests PRIVATE
            "${_GGL_TORCH_INSTALL}/include"
            "${_GGL_TORCH_INSTALL}/include/torch/csrc/api/include"
        )
        if(TORCH_LIBRARY)
            target_link_libraries(GigaLearnTests "${TORCH_LIBRARY}")
        endif()
    endif()
else()
    message(WARNING "GigaLearnTests: Torch_DIR not set; torch headers and libs may be missing.")
endif()

# Link against the main library (pulls in RLGymCPP, RocketSim, Python etc.)
# Must come AFTER the Torch link so the rpath ordering is correct.
target_link_libraries(GigaLearnTests GigaLearnCPP)

# Add private headers so tests can reach WelfordStat, GAE, ExperienceBuffer etc.
target_include_directories(GigaLearnTests PRIVATE
    "${CMAKE_SOURCE_DIR}/GigaLearnCPP/src/public"
    "${CMAKE_SOURCE_DIR}/GigaLearnCPP/src/private"
    "${CMAKE_SOURCE_DIR}/GigaLearnCPP/RLGymCPP/src"
    "${CMAKE_SOURCE_DIR}/GigaLearnCPP/libsrc/json"
)

# Register with CTest so `ctest` / `cmake --build . --target test` works.
enable_testing()
add_test(NAME GigaLearnTests COMMAND GigaLearnTests)
