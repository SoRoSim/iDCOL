cmake_minimum_required(VERSION 3.16)
project(idcol LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------- Executable ----------
add_executable(idcol_example
    examples/main.cpp
    core/idcol_kkt.cpp
    core/idcol_newton.cpp
    core/shape_core.cpp
)

# ---------- Include paths ----------
target_include_directories(idcol_example PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}            # so "core/..." works
    ${CMAKE_CURRENT_SOURCE_DIR}/eigen-3.4.0
)

# ------------------ Optimization ------------------
if (MSVC)
    target_compile_options(idcol_example PRIVATE
        $<$<CONFIG:Release>:/O2 /DNDEBUG>
    )
else()
    target_compile_options(idcol_example PRIVATE
        $<$<CONFIG:Release>:-O3 -DNDEBUG>
    )
endif()