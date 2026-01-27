cmake_minimum_required(VERSION 3.20)
project(SmokeEngine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# --------------------
# GLFW (FetchContent)
# --------------------
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG 3.3.9
)
FetchContent_MakeAvailable(glfw)

# --------------------
# OpenGL
# --------------------
find_package(OpenGL REQUIRED)

# --------------------
# ImGui (DOCKING branch via submodule)
# Make sure: external/imgui is checked out to branch "docking"
# --------------------
set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui)

add_library(imgui_lib
  ${IMGUI_DIR}/imgui.cpp
  ${IMGUI_DIR}/imgui_demo.cpp
  ${IMGUI_DIR}/imgui_draw.cpp
  ${IMGUI_DIR}/imgui_tables.cpp
  ${IMGUI_DIR}/imgui_widgets.cpp

  ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
  ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

target_include_directories(imgui_lib PUBLIC
  ${IMGUI_DIR}
  ${IMGUI_DIR}/backends
)

target_link_libraries(imgui_lib PUBLIC glfw OpenGL::GL)

# --------------------
# SmokeEngine executable
# --------------------
add_executable(SmokeEngine
  ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp

  ${CMAKE_CURRENT_SOURCE_DIR}/Sim/mac_smoke_sim_advection.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Sim/mac_smoke_sim_boundaries.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Sim/mac_smoke_sim_core.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Sim/mac_smoke_sim_forces.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Sim/mac_smoke_sim_pipe.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Sim/mac_smoke_sim_pressure.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Sim/mac_smoke_sim_sampling.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Sim/mac_grid_core.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Sim/mac_water_sim.cpp

  ${CMAKE_CURRENT_SOURCE_DIR}/Renderer/smoke_renderer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/UI/panels.cpp
)

target_include_directories(SmokeEngine PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/Sim
  ${CMAKE_CURRENT_SOURCE_DIR}/Renderer
  ${CMAKE_CURRENT_SOURCE_DIR}/UI
)

target_link_libraries(SmokeEngine PRIVATE imgui_lib glfw OpenGL::GL)

# macOS: silence OpenGL deprecation warning
if(APPLE)
  target_compile_definitions(SmokeEngine PRIVATE GL_SILENCE_DEPRECATION)
endif()
