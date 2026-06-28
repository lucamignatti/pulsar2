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
# ROCm package targets. Some pip installs do not put ROCm's CMake packages on
# CMAKE_PREFIX_PATH, so register the ROCm targets Torch may reference before
# loading Torch.
#
# ROCm/HIP target-stubbing mode. AUTO (default) stubs only when a ROCm install is
# actually present (/opt/rocm or the ROCM_PATH/HIP_PATH env); ON forces it; OFF
# disables it. The stubs exist solely to satisfy a ROCm libtorch's Torch targets.
# A CUDA build never needs them -- but if this box still has ROCm leftovers in
# /opt/rocm from a previous AMD card, AUTO would still run the (harmless but noisy)
# stubbing, so configure a clean CUDA build with -DGGL_ROCM=OFF.
set(GGL_ROCM "AUTO" CACHE STRING "ROCm/HIP libtorch dependency stubbing: AUTO, ON, or OFF")
set_property(CACHE GGL_ROCM PROPERTY STRINGS AUTO ON OFF)

function(ggl_prepare_rocm_torch_dependencies)
	if (APPLE OR WIN32)
		return()
	endif()

	if (GGL_ROCM STREQUAL "OFF")
		return()
	endif()

	# AUTO: only stub when a ROCm install is actually present, so a CUDA (or CPU-only) box is a clean no-op
	# -- no "Creating placeholder ROCm target" spam and no chance of pulling in a stray ROCm package.
	# ON: always stub (e.g. a pip ROCm wheel with no /opt/rocm and no ROCM_PATH/HIP_PATH).
	if (NOT GGL_ROCM STREQUAL "ON"
			AND NOT DEFINED ENV{ROCM_PATH}
			AND NOT DEFINED ENV{HIP_PATH}
			AND NOT EXISTS "/opt/rocm")
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

	function(ggl_register_rocm_target TARGET_NAME PACKAGE_NAME)
		find_package(${PACKAGE_NAME} CONFIG QUIET)
		if (TARGET ${TARGET_NAME})
			return()
		endif()

		foreach(GGL_ALIAS_TARGET IN LISTS ARGN)
			if (TARGET ${GGL_ALIAS_TARGET})
				add_library(${TARGET_NAME} INTERFACE IMPORTED GLOBAL)
				set_target_properties(${TARGET_NAME} PROPERTIES
					INTERFACE_LINK_LIBRARIES "${GGL_ALIAS_TARGET}"
				)
				return()
			endif()
		endforeach()

		string(REGEX REPLACE "[^A-Za-z0-9_]" "_" GGL_ROCM_TARGET_KEY "${TARGET_NAME}")
		set(GGL_ROCM_LIBRARY_VAR "GGL_ROCM_LIBRARY_${GGL_ROCM_TARGET_KEY}")
		find_library(${GGL_ROCM_LIBRARY_VAR}
			NAMES ${ARGN}
			HINTS ${GGL_ROCM_PREFIXES}
			PATH_SUFFIXES lib lib64
		)

		if (${GGL_ROCM_LIBRARY_VAR})
			message("Found ${TARGET_NAME} library without a CMake target, creating fallback: ${${GGL_ROCM_LIBRARY_VAR}}")
			add_library(${TARGET_NAME} SHARED IMPORTED GLOBAL)
			set_target_properties(${TARGET_NAME} PROPERTIES
				IMPORTED_LOCATION "${${GGL_ROCM_LIBRARY_VAR}}"
			)
		else()
			message("Creating placeholder ROCm target ${TARGET_NAME}; no library was found during configure.")
			add_library(${TARGET_NAME} INTERFACE IMPORTED GLOBAL)
		endif()
	endfunction()

	find_package(amdhip64 CONFIG QUIET)
	if (NOT TARGET hip::amdhip64)
		find_package(hip CONFIG QUIET)
	endif()
	find_package(hiprtc CONFIG QUIET)

	ggl_register_rocm_target(hip::amdhip64 amdhip64 amdhip64)
	ggl_register_rocm_target(hiprtc::hiprtc hiprtc hiprtc)
	ggl_register_rocm_target(roc::hipblas hipblas hipblas)
	ggl_register_rocm_target(roc::hipblaslt hipblaslt hipblaslt)
	ggl_register_rocm_target(roc::rocblas rocblas rocblas)
	ggl_register_rocm_target(roc::hipfft hipfft hipfft)
	ggl_register_rocm_target(roc::rocfft rocfft rocfft)
	ggl_register_rocm_target(roc::hiprand hiprand hiprand)
	ggl_register_rocm_target(roc::rocrand rocrand rocrand)
	ggl_register_rocm_target(roc::hipsparse hipsparse hipsparse)
	ggl_register_rocm_target(roc::hipsparselt hipsparselt hipsparselt)
	ggl_register_rocm_target(roc::rocsparse rocsparse rocsparse)
	ggl_register_rocm_target(roc::hipsolver hipsolver hipsolver)
	ggl_register_rocm_target(roc::rocsolver rocsolver rocsolver)
	ggl_register_rocm_target(roc::miopen MIOpen MIOpen miopen)
	ggl_register_rocm_target(MIOpen MIOpen MIOpen miopen)
	ggl_register_rocm_target(roc::rccl rccl rccl)
	ggl_register_rocm_target(hsa-runtime64::hsa-runtime64 hsa-runtime64 hsa-runtime64)
	ggl_register_rocm_target(amd_comgr::amd_comgr amd_comgr amd_comgr)
	ggl_register_rocm_target(amd_comgr amd_comgr amd_comgr)
	ggl_register_rocm_target(roc::roctracer roctracer roctracer64 roctracer)
	ggl_register_rocm_target(roctracer roctracer roctracer64 roctracer)
	ggl_register_rocm_target(roc::roctx64 roctracer roctx64)
	ggl_register_rocm_target(roctx64 roctracer roctx64)
endfunction()

ggl_prepare_rocm_torch_dependencies()

# Add libtorch (https://pytorch.org/cppdocs/installing.html#minimal-example)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Add all headers and code files
file(GLOB_RECURSE FILES_SRC CONFIGURE_DEPENDS "src/*.cpp" "src/*.h" "src/*.hpp" "libsrc/*.cpp" "libsrc/*.h" "libsrc/*.hpp")

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
