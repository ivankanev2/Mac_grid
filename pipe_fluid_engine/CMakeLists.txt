cmake_minimum_required(VERSION 3.20)
project(PipeFluidEngine CXX)

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

# ---- smoke_engine as a real dependency -------------------------------------
# Keep smoke_engine's own platform/CUDA logic authoritative.
if(NOT TARGET smoke_sim)
    set(SMOKE_BUILD_GUI OFF CACHE BOOL "" FORCE)
    set(SMOKE_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    add_subdirectory(${SMOKE_ENGINE_ROOT} ${CMAKE_BINARY_DIR}/smoke_engine_build EXCLUDE_FROM_ALL)
endif()

# ---- pipe_fluid_core: the integration library (no OpenGL) ------------------
add_library(pipe_fluid_core STATIC
    src/pipe_fluid_scene.cpp
    src/pipe_boundary_field.cpp
    src/pipe_solid_adapter.cpp
    src/pipe_solver_boundary_data.cpp
    src/blueprint_loader.cpp
    src/fluid_state_loader.cpp
)

target_include_directories(pipe_fluid_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PIPE_ENGINE_ROOT}
    ${PIPE_ENGINE_ROOT}/Geometry
    ${PIPE_ENGINE_ROOT}/Blueprint
    ${PIPE_ENGINE_ROOT}/Voxelizer
)

target_link_libraries(pipe_fluid_core PUBLIC smoke_sim)

if(APPLE)
    target_compile_definitions(pipe_fluid_core PRIVATE GL_SILENCE_DEPRECATION)
endif()

# ---- pipe_fluid_render: backend-independent volume overlay renderer --------
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