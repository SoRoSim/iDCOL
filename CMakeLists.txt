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
else()
    target_compile_options(idcol_example PRIVATE $<$<CONFIG:Release>:-O3 -DNDEBUG>)
    target_compile_options(idcol_ergodic PRIVATE $<$<CONFIG:Release>:-O3 -DNDEBUG>)
endif()

