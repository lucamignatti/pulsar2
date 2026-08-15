# GigaLearnTests — offline unit tests. Never part of the trainer build: the top level
# only adds this directory under -DGGL_BUILD_TESTS=ON, so `cmake --build build` on the
# training box is unaffected.
file(GLOB TEST_SRC CONFIGURE_DEPENDS "*.cpp" "*.h")
add_executable(GigaLearnTests ${TEST_SRC})

set_target_properties(GigaLearnTests PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(GigaLearnTests PROPERTIES CXX_STANDARD 20)

find_package(Torch REQUIRED)
target_include_directories(GigaLearnTests PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(GigaLearnTests GigaLearnCPP "${TORCH_LIBRARIES}")

# Keep reward semantics independently runnable on Windows, where unrelated internal
# GigaLearnCPP symbols are intentionally not exported from the trainer DLL.
add_executable(GigaLearnRewardTests TestMain.cpp test_team_rewards.cpp)
set_target_properties(GigaLearnRewardTests PROPERTIES LINKER_LANGUAGE CXX CXX_STANDARD 20)
target_link_libraries(GigaLearnRewardTests RLGymCPP)

# CUDA graph checks call one exported diagnostic entry point so they remain independently
# runnable on Windows, where the broader test target still references private DLL symbols.
add_executable(GigaLearnCudaGraphTests TestMain.cpp test_cuda_graph.cpp)
set_target_properties(GigaLearnCudaGraphTests PROPERTIES LINKER_LANGUAGE CXX CXX_STANDARD 20)
target_include_directories(GigaLearnCudaGraphTests PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(GigaLearnCudaGraphTests GigaLearnCPP "${TORCH_LIBRARIES}")

# Run the portable regression coverage independently of the two Windows-only test files
# that reference intentionally private ObsMirror and visualization DLL symbols.
add_executable(GigaLearnCoreTests
	TestMain.cpp
	test_advanced_obs_padded.cpp
	test_opponent_pool.cpp
	test_pbr_rewards.cpp
	test_team_envset_smoke.cpp
	test_team_rewards.cpp
	test_team_state_setters.cpp
)
set_target_properties(GigaLearnCoreTests PROPERTIES LINKER_LANGUAGE CXX CXX_STANDARD 20)
target_include_directories(GigaLearnCoreTests PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(GigaLearnCoreTests GigaLearnCPP "${TORCH_LIBRARIES}")
