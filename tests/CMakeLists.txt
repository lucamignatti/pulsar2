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
