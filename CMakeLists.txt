cmake_minimum_required (VERSION 3.8)

project("GigaLearnBot")

# Enable -fPIC for all targets (required because we mix static and shared libraries)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

file(GLOB_RECURSE FILES_SRC "src/*.cpp" "src/*.h" "src/*.hpp")
add_executable(GigaLearnBot ${FILES_SRC})

# Set C++ version to 20
set_target_properties(GigaLearnBot PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(GigaLearnBot PROPERTIES CXX_STANDARD 20)

if(NOT MSVC)
    target_compile_options(GigaLearnBot PRIVATE -march=native -ffast-math -fno-finite-math-only -funroll-loops)
endif()

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

# Unit tests
add_subdirectory(tests)
