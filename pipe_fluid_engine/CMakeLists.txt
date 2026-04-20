cmake_minimum_required(VERSION 3.20)
project(PipeFluidEngine CXX)

# ============================================================================
# pipe_fluid_engine — couples pipe_engine (geometry + voxelizer) with
# smoke_engine (3D MAC smoke + water sims).
#
# Layout expected at build time:
#   <project_root>/pipe_engine        (sibling, header-only library "pipe_core")
#   <project_root>/smoke_engine       (sibling, contains Sim/*.cpp sources)
#   <project_root>/pipe_fluid_engine  (this folder)
#
# What this file builds:
#   pipe_fluid_core      (static lib) — scene, voxel->solid adapter, loader
#   pipe_fluid_sim_lib   (static lib) — selected smoke_engine Sim sources,
#                                        compiled here so we can link them.
#   PipeFluidEngine      (executable) — GLFW/OpenGL/ImGui viewer
# ============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(PIPEFLUID_BUILD_GUI "Build the PipeFluidEngine GUI viewer" ON)

# ---- Locate sibling projects -----------------------------------------------
set(PIPE_ENGINE_ROOT  "${CMAKE_CURRENT_SOURCE_DIR}/../pipe_engine"
    CACHE PATH "Path to pipe_engine source")
set(SMOKE_ENGINE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../smoke_engine"
    CACHE PATH "Path to smoke_engine source")

if(NOT EXISTS "${PIPE_ENGINE_ROOT}/Geometry/pipe_network.h")
    message(FATAL_ERROR "pipe_engine not found at ${PIPE_ENGINE_ROOT}")
endif()
if(NOT EXISTS "${SMOKE_ENGINE_ROOT}/Sim/mac_smoke3d.h")
    message(FATAL_ERROR "smoke_engine not found at ${SMOKE_ENGINE_ROOT}")
endif()

# ---- pipe_fluid_sim_lib: minimal slice of smoke_engine we need -------------
# We compile exactly the sim sources required for 3D smoke + water. No UI,
# no renderer, no 2D couplings.
set(_SMOKE_SOURCES
    ${SMOKE_ENGINE_ROOT}/Sim/mac_grid_core.cpp
    ${SMOKE_ENGINE_ROOT}/Sim/mac_smoke_sim_advection.cpp
    ${SMOKE_ENGINE_ROOT}/Sim/mac_smoke_sim_boundaries.cpp
    ${SMOKE_ENGINE_ROOT}/Sim/mac_smoke_sim_core.cpp
    ${SMOKE_ENGINE_ROOT}/Sim/mac_smoke_sim_forces.cpp
    ${SMOKE_ENGINE_ROOT}/Sim/mac_smoke_sim_pipe.cpp
    ${SMOKE_ENGINE_ROOT}/Sim/mac_smoke_sim_pressure.cpp
    ${SMOKE_ENGINE_ROOT}/Sim/mac_smoke_sim_sampling.cpp
    ${SMOKE_ENGINE_ROOT}/Sim/pressure_solver.cpp
    ${SMOKE_ENGINE_ROOT}/Sim/pressure_solver3d.cpp
    ${SMOKE_ENGINE_ROOT}/Sim/mac_water_sim.cpp
    ${SMOKE_ENGINE_ROOT}/Sim/mac_water3d.cpp
    ${SMOKE_ENGINE_ROOT}/Sim/mac_smoke3d.cpp
    ${SMOKE_ENGINE_ROOT}/Sim/mac_coupled_sim.cpp
    ${SMOKE_ENGINE_ROOT}/Sim/text_mask.cpp
)

add_library(pipe_fluid_sim_lib STATIC ${_SMOKE_SOURCES})
target_include_directories(pipe_fluid_sim_lib PUBLIC
    ${SMOKE_ENGINE_ROOT}
    ${SMOKE_ENGINE_ROOT}/Sim
    ${SMOKE_ENGINE_ROOT}/external/stb
)
target_compile_definitions(pipe_fluid_sim_lib PUBLIC
    SMOKE_ENABLE_CUDA=0
    SMOKE_ENABLE_VERBOSE_DIAGNOSTICS=0
)
if(MSVC)
    target_compile_options(pipe_fluid_sim_lib PRIVATE /Zc:preprocessor)
endif()

# ---- pipe_fluid_core: the integration library (no OpenGL) ------------------
add_library(pipe_fluid_core STATIC
    src/pipe_fluid_scene.cpp
    src/pipe_solid_adapter.cpp
    src/blueprint_loader.cpp
)
target_include_directories(pipe_fluid_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PIPE_ENGINE_ROOT}
    ${PIPE_ENGINE_ROOT}/Geometry
    ${PIPE_ENGINE_ROOT}/Blueprint
    ${PIPE_ENGINE_ROOT}/Voxelizer
)
target_link_libraries(pipe_fluid_core PUBLIC pipe_fluid_sim_lib)

if(APPLE)
    target_compile_definitions(pipe_fluid_core PRIVATE GL_SILENCE_DEPRECATION)
endif()

# ---- pipe_fluid_render: backend-independent volume overlay renderer --------
# Depends on OpenGL. Kept separate from pipe_fluid_core so headless tools
# can use the sim + voxelizer without pulling GL.
if(PIPEFLUID_BUILD_GUI)
    find_package(OpenGL REQUIRED)
    add_library(pipe_fluid_render STATIC
        src/volume_renderer.cpp
        src/cpu_volume_renderer.cpp
        src/gpu_volume_renderer.cpp
    )
    target_include_directories(pipe_fluid_render PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(pipe_fluid_render
        PUBLIC  pipe_fluid_core
        PRIVATE OpenGL::GL
    )
    if(APPLE)
        target_compile_definitions(pipe_fluid_render PRIVATE GL_SILENCE_DEPRECATION)
    endif()
endif()

# ============================================================================
# GUI viewer
# ============================================================================
if(PIPEFLUID_BUILD_GUI)
    include(FetchContent)

    # -- GLFW (prefer a locally built copy from either sibling engine) --------
    find_package(glfw3 CONFIG QUIET)
    if(NOT TARGET glfw)
        find_package(glfw3 QUIET)
    endif()
    if(NOT TARGET glfw)
        set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
        set(_GLFW_LOCAL "")
        foreach(_candidate
            ${SMOKE_ENGINE_ROOT}/build/_deps/glfw-src
            ${SMOKE_ENGINE_ROOT}/external/glfw
            ${PIPE_ENGINE_ROOT}/build_gui/_deps/glfw-src
            ${PIPE_ENGINE_ROOT}/external/glfw)
            if(EXISTS "${_candidate}/CMakeLists.txt")
                set(_GLFW_LOCAL "${_candidate}")
                break()
            endif()
        endforeach()
        if(_GLFW_LOCAL)
            message(STATUS "[PipeFluidEngine] Using local GLFW: ${_GLFW_LOCAL}")
            add_subdirectory(${_GLFW_LOCAL} ${CMAKE_BINARY_DIR}/glfw_local EXCLUDE_FROM_ALL)
        else()
            FetchContent_Declare(glfw
                GIT_REPOSITORY https://github.com/glfw/glfw.git
                GIT_TAG 3.3.9)
            FetchContent_MakeAvailable(glfw)
        endif()
    endif()

    find_package(OpenGL REQUIRED)

    # -- ImGui (prefer a locally fetched copy from smoke_engine) --------------
    set(IMGUI_DIR "")
    foreach(_candidate
        ${SMOKE_ENGINE_ROOT}/external/imgui
        ${SMOKE_ENGINE_ROOT}/build/_deps/imgui_ext-src
        ${PIPE_ENGINE_ROOT}/external/imgui
        ${PIPE_ENGINE_ROOT}/build_gui/_deps/imgui_ext-src)
        if(EXISTS "${_candidate}/imgui.cpp")
            set(IMGUI_DIR "${_candidate}")
            break()
        endif()
    endforeach()
    if(NOT IMGUI_DIR)
        message(STATUS "[PipeFluidEngine] Fetching ImGui docking branch")
        FetchContent_Declare(imgui_ext
            GIT_REPOSITORY https://github.com/ocornut/imgui.git
            GIT_TAG docking
            GIT_SHALLOW TRUE)
        FetchContent_MakeAvailable(imgui_ext)
        set(IMGUI_DIR ${imgui_ext_SOURCE_DIR})
    else()
        message(STATUS "[PipeFluidEngine] Using local ImGui: ${IMGUI_DIR}")
    endif()

    add_library(pipe_fluid_imgui STATIC
        ${IMGUI_DIR}/imgui.cpp
        ${IMGUI_DIR}/imgui_draw.cpp
        ${IMGUI_DIR}/imgui_tables.cpp
        ${IMGUI_DIR}/imgui_widgets.cpp
        ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
        ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
    )
    target_include_directories(pipe_fluid_imgui PUBLIC
        ${IMGUI_DIR}
        ${IMGUI_DIR}/backends
    )
    target_link_libraries(pipe_fluid_imgui PUBLIC glfw OpenGL::GL)

    add_executable(PipeFluidEngine
        viewer/main_gui.cpp
    )
    target_include_directories(PipeFluidEngine PRIVATE
        ${PIPE_ENGINE_ROOT}/Geometry
        ${PIPE_ENGINE_ROOT}/Renderer
        ${PIPE_ENGINE_ROOT}/Voxelizer
        ${PIPE_ENGINE_ROOT}/Blueprint
        ${PIPE_ENGINE_ROOT}/UI
    )
    target_link_libraries(PipeFluidEngine PRIVATE
        pipe_fluid_core
        pipe_fluid_render
        pipe_fluid_imgui
        glfw
        OpenGL::GL
    )
    if(APPLE)
        target_compile_definitions(PipeFluidEngine PRIVATE GL_SILENCE_DEPRECATION)
        target_link_libraries(PipeFluidEngine PRIVATE "-framework Cocoa" "-framework IOKit" "-framework CoreVideo")
    endif()
endif()
