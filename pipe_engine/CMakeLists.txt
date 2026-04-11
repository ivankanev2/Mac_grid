cmake_minimum_required(VERSION 3.20)
project(PipeEngine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---- Options ----------------------------------------------------------------
option(PIPE_BUILD_GUI   "Build the interactive GLFW/OpenGL/ImGui viewer" ON)
option(PIPE_BUILD_CLI   "Build the headless CLI mesh generator" ON)

# ---- Core header-only library -----------------------------------------------
add_library(pipe_core INTERFACE)
target_include_directories(pipe_core INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/Geometry
    ${CMAKE_CURRENT_SOURCE_DIR}/Blueprint
    ${CMAKE_CURRENT_SOURCE_DIR}/Voxelizer
    ${CMAKE_CURRENT_SOURCE_DIR}/Renderer
    ${CMAKE_CURRENT_SOURCE_DIR}/UI
)

# ---- CLI tool ---------------------------------------------------------------
if(PIPE_BUILD_CLI)
    add_executable(PipeEngine
        ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
    )
    target_link_libraries(PipeEngine PRIVATE pipe_core)
    if(APPLE)
        target_compile_definitions(PipeEngine PRIVATE GL_SILENCE_DEPRECATION)
    endif()
    if(MSVC)
        target_compile_options(PipeEngine PRIVATE /Zc:preprocessor)
    endif()
endif()

# ---- GUI viewer -------------------------------------------------------------
if(PIPE_BUILD_GUI)
    include(FetchContent)

    # -- GLFW -----------------------------------------------------------------
    # Try to reuse the sibling smoke_engine's already-fetched GLFW first.
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)

    set(_GLFW_LOCAL "")
    foreach(_candidate
        ${CMAKE_CURRENT_SOURCE_DIR}/../smoke_engine/build/_deps/glfw-src
        ${CMAKE_CURRENT_SOURCE_DIR}/../smoke_engine/external/glfw
        ${CMAKE_CURRENT_SOURCE_DIR}/external/glfw)
      if(EXISTS "${_candidate}/CMakeLists.txt")
        set(_GLFW_LOCAL "${_candidate}")
        break()
      endif()
    endforeach()

    if(_GLFW_LOCAL)
        message(STATUS "[PipeEngine] Using local GLFW: ${_GLFW_LOCAL}")
        add_subdirectory(${_GLFW_LOCAL} ${CMAKE_BINARY_DIR}/glfw_local EXCLUDE_FROM_ALL)
    else()
        message(STATUS "[PipeEngine] Fetching GLFW 3.3.9")
        FetchContent_Declare(glfw
            GIT_REPOSITORY https://github.com/glfw/glfw.git
            GIT_TAG        3.3.9
        )
        FetchContent_MakeAvailable(glfw)
    endif()

    # -- OpenGL ---------------------------------------------------------------
    find_package(OpenGL REQUIRED)

    # -- ImGui ----------------------------------------------------------------
    # Reuse smoke_engine's external/imgui checkout if present.
    set(_IMGUI_DIR "")
    foreach(_candidate
        ${CMAKE_CURRENT_SOURCE_DIR}/../smoke_engine/external/imgui
        ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui)
      if(EXISTS "${_candidate}/imgui.cpp")
        set(_IMGUI_DIR "${_candidate}")
        break()
      endif()
    endforeach()

    if(NOT _IMGUI_DIR)
        message(STATUS "[PipeEngine] Fetching ImGui (docking branch)")
        FetchContent_Declare(imgui_ext
            GIT_REPOSITORY https://github.com/ocornut/imgui.git
            GIT_TAG        docking
            GIT_SHALLOW    TRUE
        )
        FetchContent_MakeAvailable(imgui_ext)
        set(_IMGUI_DIR ${imgui_ext_SOURCE_DIR})
    else()
        message(STATUS "[PipeEngine] Using local ImGui: ${_IMGUI_DIR}")
    endif()

    add_library(pipe_imgui STATIC
        ${_IMGUI_DIR}/imgui.cpp
        ${_IMGUI_DIR}/imgui_demo.cpp
        ${_IMGUI_DIR}/imgui_draw.cpp
        ${_IMGUI_DIR}/imgui_tables.cpp
        ${_IMGUI_DIR}/imgui_widgets.cpp
        ${_IMGUI_DIR}/backends/imgui_impl_glfw.cpp
        ${_IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
    )
    target_include_directories(pipe_imgui PUBLIC
        ${_IMGUI_DIR}
        ${_IMGUI_DIR}/backends
    )
    target_link_libraries(pipe_imgui PUBLIC glfw OpenGL::GL)

    # -- PipeEngineGUI executable ---------------------------------------------
    add_executable(PipeEngineGUI
        ${CMAKE_CURRENT_SOURCE_DIR}/src/main_gui.cpp
    )
    target_link_libraries(PipeEngineGUI PRIVATE pipe_core pipe_imgui glfw OpenGL::GL)
    target_include_directories(PipeEngineGUI PRIVATE
        ${_IMGUI_DIR}
        ${_IMGUI_DIR}/backends
    )

    if(APPLE)
        target_compile_definitions(PipeEngineGUI PRIVATE GL_SILENCE_DEPRECATION)
        target_link_libraries(PipeEngineGUI PRIVATE "-framework Cocoa" "-framework IOKit")
    endif()
    if(MSVC)
        target_compile_options(PipeEngineGUI PRIVATE /Zc:preprocessor)
    endif()
endif()

# ---- Copy example blueprints to build dir -----------------------------------
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/examples/
     DESTINATION ${CMAKE_BINARY_DIR}/examples)
