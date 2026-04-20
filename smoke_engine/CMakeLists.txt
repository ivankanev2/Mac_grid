cmake_minimum_required(VERSION 3.20)
project(SmokeEngine CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(MSVC AND CMAKE_SIZEOF_VOID_P EQUAL 4)
  message(WARNING "Configuring a 32-bit Windows build. This will be much slower and more memory-constrained than x64. Use the x64 Native Tools prompt or a 64-bit generator.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(SMOKE_ENABLE_CUDA "Enable experimental CUDA build plumbing for the 3D water path" OFF)

if(WIN32 AND NOT SMOKE_ENABLE_CUDA)
  message(WARNING "SMOKE_ENABLE_CUDA is OFF. On NVIDIA Windows laptops, the 3D water backend will stay on CPU until you configure with -DSMOKE_ENABLE_CUDA=ON and have the CUDA toolkit installed.")
endif()
option(SMOKE_BUILD_GUI "Build the GLFW/OpenGL/ImGui visualization app" ON)
option(SMOKE_BUILD_TOOLS "Build headless simulation and validation tools" ON)
option(SMOKE_ENABLE_VERBOSE_DIAGNOSTICS "Enable very noisy per-step diagnostic logging" OFF)

if(SMOKE_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "SMOKE_ENABLE_CUDA=ON but no CUDA compiler was found. Disable SMOKE_ENABLE_CUDA on machines without nvcc.")
  endif()

  enable_language(CUDA)

  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 CACHE STRING "CUDA architectures to build for")
  endif()

  find_package(CUDAToolkit REQUIRED)
endif()

set(SMOKE_SIM_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/Sim/mac_smoke_sim_advection.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Sim/mac_smoke_sim_boundaries.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Sim/mac_smoke_sim_core.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Sim/mac_smoke_sim_forces.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Sim/mac_smoke_sim_pipe.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Sim/mac_smoke_sim_pressure.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Sim/mac_smoke_sim_sampling.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Sim/mac_grid_core.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Sim/pressure_solver.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Sim/pressure_solver3d.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Sim/mac_water_sim.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Sim/mac_water3d.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Sim/mac_smoke3d.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Sim/mac_coupled_sim.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Sim/text_mask.cpp
)

set(SMOKE_COMMON_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/Sim
  ${CMAKE_CURRENT_SOURCE_DIR}/Renderer
  ${CMAKE_CURRENT_SOURCE_DIR}/UI
  ${CMAKE_CURRENT_SOURCE_DIR}/external/stb
)

function(smoke_configure_sim_target target_name)
  target_compile_definitions(${target_name} PRIVATE
    SMOKE_ENABLE_CUDA=$<BOOL:${SMOKE_ENABLE_CUDA}>
    SMOKE_ENABLE_VERBOSE_DIAGNOSTICS=$<BOOL:${SMOKE_ENABLE_VERBOSE_DIAGNOSTICS}>
  )

  target_include_directories(${target_name} PRIVATE ${SMOKE_COMMON_INCLUDE_DIRS})

  if(MSVC)
    target_compile_options(${target_name} PRIVATE
      $<$<COMPILE_LANGUAGE:C,CXX>:/Zc:preprocessor>
    )
  endif()

  if(SMOKE_ENABLE_CUDA)
    target_sources(${target_name} PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/Sim/Water3D/water3d_cuda_backend.cu
    )

    if(MSVC)
      target_compile_options(${target_name} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
      )
    endif()

    target_link_libraries(${target_name} PRIVATE CUDA::cudart)
    set_target_properties(${target_name} PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
      CUDA_RUNTIME_LIBRARY Shared
      CUDA_SEPARABLE_COMPILATION OFF
      CUDA_RESOLVE_DEVICE_SYMBOLS OFF
    )
  endif()
endfunction()

# -----------------------------------------------------------------------------
# Core simulation library used both by SmokeEngine itself and by sibling
# projects such as pipe_fluid_engine.
# -----------------------------------------------------------------------------
add_library(smoke_sim STATIC
  ${SMOKE_SIM_SOURCES}
)
smoke_configure_sim_target(smoke_sim)
target_include_directories(smoke_sim PUBLIC ${SMOKE_COMMON_INCLUDE_DIRS})

if(SMOKE_BUILD_TOOLS)
  add_executable(Water3DBackendCompare
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/water3d_backend_compare.cpp
  )
  target_link_libraries(Water3DBackendCompare PRIVATE smoke_sim)
endif()

if(SMOKE_BUILD_GUI)
  include(FetchContent)

  # --------------------
  # GLFW (prefer system package, fall back to FetchContent)
  # --------------------
  find_package(glfw3 CONFIG QUIET)
  if(NOT TARGET glfw)
    find_package(glfw3 QUIET)
  endif()

  if(NOT TARGET glfw)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)

    set(GLFW_LOCAL_SRC "")
    foreach(_glfw_candidate
        ${CMAKE_CURRENT_SOURCE_DIR}/external/glfw
        ${CMAKE_CURRENT_SOURCE_DIR}/build/_deps/glfw-src)
      if(EXISTS ${_glfw_candidate}/CMakeLists.txt)
        set(GLFW_LOCAL_SRC ${_glfw_candidate})
        break()
      endif()
    endforeach()

    if(GLFW_LOCAL_SRC)
      message(STATUS "Using local GLFW source: ${GLFW_LOCAL_SRC}")
      add_subdirectory(${GLFW_LOCAL_SRC} ${CMAKE_BINARY_DIR}/glfw_local EXCLUDE_FROM_ALL)
    else()
      FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG 3.3.9
      )
      FetchContent_MakeAvailable(glfw)
    endif()
  endif()


  # --------------------
  # OpenGL
  # --------------------
  find_package(OpenGL REQUIRED)

  # --------------------
  # ImGui (prefer local docking branch checkout, otherwise fetch it)
  # --------------------
  set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui)
  if(NOT EXISTS ${IMGUI_DIR}/imgui.cpp)
    message(STATUS "external/imgui not found; fetching ImGui docking branch")
    FetchContent_Declare(
      imgui_ext
      GIT_REPOSITORY https://github.com/ocornut/imgui.git
      GIT_TAG docking
      GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(imgui_ext)
    set(IMGUI_DIR ${imgui_ext_SOURCE_DIR})
  endif()

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

  add_executable(SmokeEngine
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Renderer/smoke_renderer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/UI/panels.cpp
  )

  target_link_libraries(SmokeEngine PRIVATE smoke_sim imgui_lib glfw OpenGL::GL)

  if(APPLE)
    target_compile_definitions(SmokeEngine PRIVATE GL_SILENCE_DEPRECATION)
  endif()
endif()
