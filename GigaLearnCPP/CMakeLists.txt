cmake_minimum_required (VERSION 3.8)

project("GigaLearnCPP")

include_directories("${PROJECT_SOURCE_DIR}/src/")

# Make sure CMake finds libtorch if its in this directory
if (EXISTS "${PROJECT_SOURCE_DIR}/libtorch/")
	message("Using local libtorch folder...")
	list(APPEND CMAKE_PREFIX_PATH "libtorch")
	# Make ultra-sure we can find libtorch if its local
	set(CMAKE_PREFIX_PATH "libtorch/share/cmake/Torch")
endif()

# ROCm PyTorch packages can export Torch targets whose link interfaces reference
# hip::amdhip64. Some pip installs do not put ROCm's CMake packages on
# CMAKE_PREFIX_PATH, so register HIP before loading Torch.
function(ggl_prepare_rocm_torch_dependencies)
	if (APPLE OR WIN32)
		return()
	endif()

	set(GGL_ROCM_PREFIXES)
	if (DEFINED ENV{ROCM_PATH})
		list(APPEND GGL_ROCM_PREFIXES "$ENV{ROCM_PATH}")
	endif()
	if (DEFINED ENV{HIP_PATH})
		list(APPEND GGL_ROCM_PREFIXES "$ENV{HIP_PATH}")
	endif()
	list(APPEND GGL_ROCM_PREFIXES "/opt/rocm" "/usr")

	foreach(GGL_ROCM_PREFIX IN LISTS GGL_ROCM_PREFIXES)
		if (EXISTS "${GGL_ROCM_PREFIX}")
			list(APPEND CMAKE_PREFIX_PATH
				"${GGL_ROCM_PREFIX}"
				"${GGL_ROCM_PREFIX}/lib/cmake"
				"${GGL_ROCM_PREFIX}/lib64/cmake"
			)
		endif()
	endforeach()
	set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}" PARENT_SCOPE)

	find_package(amdhip64 CONFIG QUIET)
	if (NOT TARGET hip::amdhip64)
		find_package(hip CONFIG QUIET)
	endif()

	if (NOT TARGET hip::amdhip64)
		find_library(GGL_AMDHIP64_LIBRARY
			NAMES amdhip64
			HINTS ${GGL_ROCM_PREFIXES}
			PATH_SUFFIXES lib lib64
		)

		if (GGL_AMDHIP64_LIBRARY)
			message("Found libamdhip64 without a CMake target, creating hip::amdhip64 fallback: ${GGL_AMDHIP64_LIBRARY}")
			add_library(hip::amdhip64 SHARED IMPORTED GLOBAL)
			set_target_properties(hip::amdhip64 PROPERTIES
				IMPORTED_LOCATION "${GGL_AMDHIP64_LIBRARY}"
			)
		endif()
	endif()
endfunction()

ggl_prepare_rocm_torch_dependencies()

# Add libtorch (https://pytorch.org/cppdocs/installing.html#minimal-example)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Add all headers and code files
file(GLOB_RECURSE FILES_SRC CONFIGURE_DEPENDS "src/*.cpp" "src/*.h" "src/*.hpp" "libsrc/*.cpp" "libsrc/.h" "libsrc/.hpp")

add_library(GigaLearnCPP SHARED ${FILES_SRC})
target_compile_definitions(GigaLearnCPP PRIVATE -DWITHIN_GGL)
target_include_directories(GigaLearnCPP PUBLIC "src/public")
target_include_directories(GigaLearnCPP PRIVATE "src/private")

# Include libtorch
target_link_libraries(GigaLearnCPP PRIVATE "${TORCH_LIBRARIES}")

if (NOT APPLE)
	message("Enabling CUDA support...")
	target_compile_definitions(GigaLearnCPP PRIVATE -DRG_CUDA_SUPPORT)
endif()

# Set C++ version to 20
set_target_properties(GigaLearnCPP PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(GigaLearnCPP PROPERTIES CXX_STANDARD 20)

if(NOT MSVC)
    # -fno-finite-math-only restores correct NaN/Inf semantics for nlohmann/json (serialises float infinity).
    target_compile_options(GigaLearnCPP PRIVATE -march=native -ffast-math -fno-finite-math-only -funroll-loops)
endif()

# Include RLGymCPP
add_subdirectory(RLGymCPP)
target_link_libraries(GigaLearnCPP PUBLIC RLGymCPP)

# Include JSON
target_include_directories(GigaLearnCPP PUBLIC "${PROJECT_SOURCE_DIR}/libsrc/json")

# Include python
find_package(Python COMPONENTS Interpreter Development)
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
target_link_libraries(GigaLearnCPP PUBLIC ${PYTHON_LIBRARIES})
message("Found Python:")
message("PYTHON_LIBRARIES: ${PYTHON_LIBRARIES}")
message("PYTHON_INCLUDE_DIRS: ${PYTHON_INCLUDE_DIRS}")
message("Python_RUNTIME_LIBRARY_DIRS: ${Python_RUNTIME_LIBRARY_DIRS}")
message("Python_EXECUTABLE: ${Python_EXECUTABLE}")
add_definitions(-DPY_EXEC_PATH="${Python_EXECUTABLE}") # Give C++ access to the executable path

# Include pybind11
add_subdirectory(pybind11)
target_link_libraries(GigaLearnCPP PUBLIC pybind11::embed)

# MSVC fails to find python DLLs even through they are in my path. Good job MSVC. Well done.
# This copies the the python DLLs to the output directory
if (MSVC)
    file(GLOB PYTHON_DLLS "${Python_RUNTIME_LIBRARY_DIRS}/*.dll")
	message("Adding Python DLLS: ${PYTHON_DLLS}")
    add_custom_command(TARGET GigaLearnCPP
                 POST_BUILD
                 COMMAND ${CMAKE_COMMAND} -E copy_if_different
                 ${PYTHON_DLLS}
                 $<TARGET_FILE_DIR:GigaLearnCPP>)
endif (MSVC)

# Make our python files copy over to our build dir
configure_file("./python_scripts/metric_receiver.py" "../python_scripts/metric_receiver.py" COPYONLY)
configure_file("./python_scripts/render_receiver.py" "../python_scripts/render_receiver.py" COPYONLY)

# MSVC sometimes won't link to the libtorch DLLs unless you do this
# This is also from https://pytorch.org/cppdocs/installing.html#minimal-example
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
	message("Adding TORCH_DLLS: ${TORCH_DLLS}")
    add_custom_command(TARGET GigaLearnCPP
                 POST_BUILD
                 COMMAND ${CMAKE_COMMAND} -E copy_if_different
                 ${TORCH_DLLS}
                 $<TARGET_FILE_DIR:GigaLearnCPP>)
endif (MSVC)
