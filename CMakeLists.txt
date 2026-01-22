cmake_minimum_required(VERSION 3.16)
project(idcol LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------- Core sources ----------
set(IDCOL_CORE
    core/idcol_solve.cpp
    core/idcol_newton.cpp
    core/idcol_kkt.cpp
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

# ---------- Include paths ----------
target_include_directories(idcol_example PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/external/eigen
)

target_include_directories(idcol_ergodic PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/external/eigen
)

# ---------- Optimization + Eigen knobs (Release only) ----------
if (MSVC)
    target_compile_options(idcol_example PRIVATE $<$<CONFIG:Release>:/O2 /DNDEBUG>)
    target_compile_options(idcol_ergodic PRIVATE $<$<CONFIG:Release>:/O2 /DNDEBUG>)

    target_compile_definitions(idcol_example PRIVATE $<$<CONFIG:Release>:EIGEN_NO_DEBUG EIGEN_MPL2_ONLY EIGEN_DONT_PARALLELIZE>)
    target_compile_definitions(idcol_ergodic PRIVATE $<$<CONFIG:Release>:EIGEN_NO_DEBUG EIGEN_MPL2_ONLY EIGEN_DONT_PARALLELIZE>)
else()
    target_compile_options(idcol_example PRIVATE $<$<CONFIG:Release>:-O3 -DNDEBUG -march=native>)
    target_compile_options(idcol_ergodic PRIVATE $<$<CONFIG:Release>:-O3 -DNDEBUG -march=native>)

    target_compile_definitions(idcol_example PRIVATE $<$<CONFIG:Release>:EIGEN_NO_DEBUG EIGEN_MPL2_ONLY EIGEN_DONT_PARALLELIZE>)
    target_compile_definitions(idcol_ergodic PRIVATE $<$<CONFIG:Release>:EIGEN_NO_DEBUG EIGEN_MPL2_ONLY EIGEN_DONT_PARALLELIZE>)
endif()

