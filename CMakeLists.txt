cmake_minimum_required(VERSION 3.16)
project(idcol LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------- Core sources ----------
set(IDCOL_CORE
    core/idcol_kkt.cpp
    core/idcol_newton.cpp
    core/shape_core.cpp
    core/radial_bounds.cpp
)

# ---------- Executables ----------
add_executable(idcol_example
    examples/main.cpp
    ${IDCOL_CORE}
)

add_executable(idcol_ergodic
    examples/ergodic.cpp
    ${IDCOL_CORE}
)

add_executable(idcol_bench
    examples/bench.cpp
    ${IDCOL_CORE}
)

target_include_directories(idcol_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/eigen-3.4.0
)

# ---------- Include paths ----------
target_include_directories(idcol_example PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/eigen-3.4.0
)

target_include_directories(idcol_ergodic PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/eigen-3.4.0
)

# ---------- Optimization (Release only) ----------
if (MSVC)
    target_compile_options(idcol_example PRIVATE $<$<CONFIG:Release>:/O2 /DNDEBUG>)
    target_compile_options(idcol_ergodic PRIVATE $<$<CONFIG:Release>:/O2 /DNDEBUG>)

    # Enable Eigen release-only optimizations (no debug checks) on MSVC
    target_compile_definitions(idcol_example PRIVATE $<$<CONFIG:Release>:EIGEN_NO_DEBUG EIGEN_MPL2_ONLY>)
    target_compile_definitions(idcol_ergodic PRIVATE $<$<CONFIG:Release>:EIGEN_NO_DEBUG EIGEN_MPL2_ONLY>)
else()
    target_compile_options(idcol_example PRIVATE $<$<CONFIG:Release>:-O3 -DNDEBUG>)
    target_compile_options(idcol_ergodic PRIVATE $<$<CONFIG:Release>:-O3 -DNDEBUG>)

    # Enable Eigen release-only optimizations on non-MSVC toolchains
    target_compile_definitions(idcol_example PRIVATE $<$<CONFIG:Release>:EIGEN_NO_DEBUG EIGEN_MPL2_ONLY>)
    target_compile_definitions(idcol_ergodic PRIVATE $<$<CONFIG:Release>:EIGEN_NO_DEBUG EIGEN_MPL2_ONLY>)
endif()

