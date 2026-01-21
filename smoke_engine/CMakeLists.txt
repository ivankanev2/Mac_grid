cmake_minimum_required(VERSION 3.20)
project(SmokeEngine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# --- GLFW ---
FetchContent_Declare(
  glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG 3.3.9
)
FetchContent_MakeAvailable(glfw)

# --- ImGui ---
FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG v1.90.9
)

set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(imgui)

# ImGui sources (core + backends)
set(IMGUI_DIR ${imgui_SOURCE_DIR})
add_library(imgui_lib
  ${IMGUI_DIR}/imgui.cpp
  ${IMGUI_DIR}/imgui_demo.cpp
  ${IMGUI_DIR}/imgui_draw.cpp
  ${IMGUI_DIR}/imgui_tables.cpp
  ${IMGUI_DIR}/imgui_widgets.cpp
  ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
  ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

target_link_libraries(imgui_lib PUBLIC glfw)

target_include_directories(imgui_lib PUBLIC
  ${IMGUI_DIR}
  ${IMGUI_DIR}/backends
)

add_executable(SmokeEngine
    src/main.cpp
    ../sim/mac_smoke_sim.cpp
)

target_include_directories(SmokeEngine PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../sim
)

target_link_libraries(SmokeEngine PRIVATE imgui_lib glfw)

# macOS: link system frameworks for OpenGL
if(APPLE)
  target_compile_definitions(SmokeEngine PRIVATE GL_SILENCE_DEPRECATION)
  target_link_libraries(SmokeEngine PRIVATE "-framework OpenGL" "-framework Cocoa" "-framework IOKit" "-framework CoreVideo")
endif()