cmake_minimum_required(VERSION 3.8)

project("RocketSimV3")

# The physics engine itself is Rust (vendored, see README.md); cargo produces a
# staticlib the C++ compat layer links. rust-toolchain.toml pins the toolchain
# and makes rustup auto-install it on first build.
find_program(CARGO_EXECUTABLE cargo HINTS "$ENV{CARGO_HOME}/bin" "$ENV{HOME}/.cargo/bin")
if(NOT CARGO_EXECUTABLE)
    message(FATAL_ERROR
        "RocketSimV3 needs cargo (Rust). Install rustup once:\n"
        "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y\n"
        "then re-run cmake. (The pinned toolchain auto-installs on first build.)")
endif()

# rustc's default CPU for powerpc64le-unknown-linux-gnu is generic "ppc64le",
# not the host. On CCNI POWER9 that drops VSX; -Ctarget-cpu=native selects pwr9.
# Preserve any caller RUSTFLAGS (append, do not clobber).
set(_rsv3_rustflags "-Ctarget-cpu=native")
if(DEFINED ENV{RUSTFLAGS} AND NOT "$ENV{RUSTFLAGS}" STREQUAL "")
    set(_rsv3_rustflags "$ENV{RUSTFLAGS} ${_rsv3_rustflags}")
endif()
set(_rsv3_cargo_env "RUSTFLAGS=${_rsv3_rustflags}")
if(DEFINED ENV{RUSTUP_HOME})
    list(APPEND _rsv3_cargo_env "RUSTUP_HOME=$ENV{RUSTUP_HOME}")
endif()
if(DEFINED ENV{CARGO_HOME})
    list(APPEND _rsv3_cargo_env "CARGO_HOME=$ENV{CARGO_HOME}")
endif()

if(MSVC)
    set(RSV3_RUST_LIB_NAME "rocketsim_ffi.lib")
else()
    set(RSV3_RUST_LIB_NAME "librocketsim_ffi.a")
endif()

# Per-CMake-tree cargo output so ppc64le `build/` and x86_64 `build-npl/` can
# share this source dir without clobbering librocketsim_ffi.a. Native cargo
# writes target/release, not target/<triple>/release.
set(RSV3_CARGO_TARGET_DIR "${CMAKE_BINARY_DIR}/cargo-target")
list(APPEND _rsv3_cargo_env "CARGO_TARGET_DIR=${RSV3_CARGO_TARGET_DIR}")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(RSV3_CARGO_PROFILE "dev")
    set(RSV3_CARGO_OUT "debug")
else()
    set(RSV3_CARGO_PROFILE "release")
    set(RSV3_CARGO_OUT "release")
endif()
set(RSV3_RUST_LIB "${RSV3_CARGO_TARGET_DIR}/${RSV3_CARGO_OUT}/${RSV3_RUST_LIB_NAME}")

# Always invoke cargo (it is its own fast incremental no-op when clean).
add_custom_target(RocketSimV3Cargo
    COMMAND ${CMAKE_COMMAND} -E env ${_rsv3_cargo_env}
        ${CARGO_EXECUTABLE} build --profile ${RSV3_CARGO_PROFILE} -p rocketsim_ffi
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    BYPRODUCTS ${RSV3_RUST_LIB}
    COMMENT "cargo build rocketsim_ffi (${RSV3_CARGO_PROFILE}, native CPU, CARGO_TARGET_DIR=${RSV3_CARGO_TARGET_DIR})"
    VERBATIM
)

# Compat layer + the engine-independent v2 sources it reuses (math types and
# helpers only — no v2 engine .cpp may ever be added here, the symbols would
# collide with the facade).
add_library(RocketSimV3 STATIC
    "compat/RocketSimCompat.cpp"
    "compat/GameEventTracker.cpp"
    "../RocketSim/src/Math/Math.cpp"
    "../RocketSim/src/Math/MathTypes/MathTypes.cpp"
    "../RocketSim/src/Sim/PhysState/PhysState.cpp"
)
add_dependencies(RocketSimV3 RocketSimV3Cargo)

set_target_properties(RocketSimV3 PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(RocketSimV3 PROPERTIES CXX_STANDARD 20)

# Same collection-hot-path flags as the v2 engine build (see RocketSim/CMakeLists.txt).
if(NOT MSVC)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "ppc64|powerpc")
        target_compile_options(RocketSimV3 PRIVATE -mcpu=native -ffast-math -fno-finite-math-only -funroll-loops)
    else()
        target_compile_options(RocketSimV3 PRIVATE -march=native -ffast-math -fno-finite-math-only -funroll-loops)
    endif()
endif()

target_link_libraries(RocketSimV3 PUBLIC ${RSV3_RUST_LIB})
if(MSVC)
    # `cargo rustc -- --print native-static-libs` for the MSVC target. Cargo
    # cannot propagate these system-library requirements through a bare
    # staticlib path, so the C++ host must name them explicitly.
    target_link_libraries(RocketSimV3 PUBLIC kernel32 ntdll userenv ws2_32 dbghelp)
elseif(APPLE)
    # Rust std runtime deps on macOS
    target_link_libraries(RocketSimV3 PUBLIC "-framework CoreFoundation")
elseif(UNIX)
    target_link_libraries(RocketSimV3 PUBLIC pthread dl m)
endif()
