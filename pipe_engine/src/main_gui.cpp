// ============================================================================
// Pipe Engine — Live GUI entry point
//
// Usage:
//   PipeEngineGUI                          → opens viewer with default preset
//   PipeEngineGUI <blueprint.pipe>         → loads blueprint on startup
//
// Controls:
//   Left-drag   → orbit camera
//   Right-drag  → pan
//   Scroll      → zoom
//   Ctrl+Z      → undo last segment
// ============================================================================

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#ifdef __APPLE__
#  define GL_SILENCE_DEPRECATION
#  include <OpenGL/gl3.h>
#else
#  ifndef GL_GLEXT_PROTOTYPES
#    define GL_GLEXT_PROTOTYPES
#  endif
#  include <GL/gl.h>
#  include <GL/glext.h>
#endif

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

#include "../Geometry/pipe_network.h"
#include "../Geometry/mesh_generator.h"
#include "../Geometry/obj_exporter.h"
#include "../Blueprint/blueprint_parser.h"
#include "../Renderer/camera.h"
#include "../Renderer/mesh_renderer.h"
#include "../UI/pipe_panels.h"

#include <iostream>
#include <string>
#include <cstdio>

// ---- Global state (needed in GLFW callbacks) --------------------------------
static MeshRenderer* g_renderer = nullptr;
static PipeUI*       g_ui       = nullptr;
static GLFWwindow*   g_win      = nullptr;

// ---- GLFW callbacks --------------------------------------------------------
static void errorCallback(int, const char* desc) {
    std::cerr << "[GLFW] " << desc << "\n";
}

static void scrollCallback(GLFWwindow*, double, double dy) {
    if (g_renderer && !ImGui::GetIO().WantCaptureMouse)
        g_renderer->camera.onScroll((float)dy);
}

static void mouseBtnCallback(GLFWwindow*, int button, int action, int /*mods*/) {
    if (ImGui::GetIO().WantCaptureMouse) return;
    if (!g_renderer) return;
    double mx, my;
    glfwGetCursorPos(g_win, &mx, &my);
    g_renderer->camera.onMouseButton(button, action == GLFW_PRESS, (float)mx, (float)my);
}

static void cursorPosCallback(GLFWwindow*, double x, double y) {
    if (ImGui::GetIO().WantCaptureMouse) return;
    if (g_renderer)
        g_renderer->camera.onMouseMove((float)x, (float)y);
}

static void keyCallback(GLFWwindow* win, int key, int /*scan*/, int action, int mods) {
    if (action != GLFW_PRESS) return;
    // Ctrl+Z → undo last segment
    if (key == GLFW_KEY_Z && (mods & GLFW_MOD_CONTROL)) {
        if (g_ui && !g_ui->network.segments.empty()) {
            g_ui->network.segments.pop_back();
            if (!g_ui->network.segments.empty()) {
                auto& last = *g_ui->network.segments.back();
                g_ui->network.cursor    = last.endPoint();
                g_ui->network.cursorDir = last.tangent(1.f);
            } else {
                g_ui->hasPipeStarted = false;
            }
            g_ui->meshDirty = true;
            g_ui->setStatus("Undo");
        }
    }
}

// ---- Theme (mirrors smoke engine dark style) --------------------------------
static void applyDarkTheme() {
    ImGuiStyle& s = ImGui::GetStyle();
    s.WindowRounding    = 6.f;
    s.FrameRounding     = 4.f;
    s.GrabRounding      = 4.f;
    s.ScrollbarRounding = 4.f;
    s.TabRounding       = 4.f;
    s.WindowBorderSize  = 1.f;
    s.FrameBorderSize   = 0.f;
    s.ItemSpacing       = {6, 4};
    s.FramePadding      = {6, 3};

    ImVec4* c = s.Colors;
    c[ImGuiCol_WindowBg]         = {0.12f, 0.12f, 0.14f, 1.f};
    c[ImGuiCol_ChildBg]          = {0.10f, 0.10f, 0.12f, 1.f};
    c[ImGuiCol_FrameBg]          = {0.18f, 0.18f, 0.22f, 1.f};
    c[ImGuiCol_FrameBgHovered]   = {0.24f, 0.24f, 0.30f, 1.f};
    c[ImGuiCol_FrameBgActive]    = {0.28f, 0.50f, 0.70f, 1.f};
    c[ImGuiCol_TitleBg]          = {0.09f, 0.09f, 0.11f, 1.f};
    c[ImGuiCol_TitleBgActive]    = {0.16f, 0.30f, 0.48f, 1.f};
    c[ImGuiCol_MenuBarBg]        = {0.09f, 0.09f, 0.11f, 1.f};
    c[ImGuiCol_Header]           = {0.20f, 0.40f, 0.60f, 0.55f};
    c[ImGuiCol_HeaderHovered]    = {0.26f, 0.48f, 0.70f, 0.80f};
    c[ImGuiCol_HeaderActive]     = {0.30f, 0.55f, 0.80f, 1.f};
    c[ImGuiCol_Button]           = {0.20f, 0.40f, 0.60f, 0.60f};
    c[ImGuiCol_ButtonHovered]    = {0.26f, 0.50f, 0.72f, 1.f};
    c[ImGuiCol_ButtonActive]     = {0.06f, 0.53f, 0.98f, 1.f};
    c[ImGuiCol_SliderGrab]       = {0.30f, 0.55f, 0.80f, 1.f};
    c[ImGuiCol_SliderGrabActive] = {0.06f, 0.53f, 0.98f, 1.f};
    c[ImGuiCol_Tab]              = {0.15f, 0.27f, 0.42f, 0.86f};
    c[ImGuiCol_TabHovered]       = {0.26f, 0.48f, 0.70f, 0.80f};
    c[ImGuiCol_TabActive]        = {0.20f, 0.37f, 0.57f, 1.f};
    c[ImGuiCol_CheckMark]        = {0.26f, 0.59f, 0.98f, 1.f};
    c[ImGuiCol_Text]             = {0.90f, 0.90f, 0.92f, 1.f};
    c[ImGuiCol_TextDisabled]     = {0.50f, 0.50f, 0.55f, 1.f};
    c[ImGuiCol_Separator]        = {0.28f, 0.28f, 0.34f, 1.f};
    c[ImGuiCol_ScrollbarBg]      = {0.10f, 0.10f, 0.12f, 1.f};
    c[ImGuiCol_ScrollbarGrab]    = {0.30f, 0.30f, 0.38f, 1.f};
}

// ============================================================================
int main(int argc, char* argv[]) {
    std::cout << "=== Pipe Engine GUI v0.1 ===\n";

    // Optional startup blueprint path
    std::string blueprintPath;
    if (argc >= 2) blueprintPath = argv[1];

    // ---- GLFW init ---------------------------------------------------------
    glfwSetErrorCallback(errorCallback);
    if (!glfwInit()) {
        std::cerr << "GLFW init failed\n";
        return 1;
    }

    // OpenGL 3.2 Core (same as smoke engine)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
#endif
    glfwWindowHint(GLFW_SAMPLES, 4);  // MSAA

    GLFWwindow* win = glfwCreateWindow(1400, 900, "Pipe Engine | Vizior Research", nullptr, nullptr);
    if (!win) {
        std::cerr << "GLFW window creation failed\n";
        glfwTerminate();
        return 1;
    }
    g_win = win;
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);  // vsync

    // ---- GLFW callbacks ----------------------------------------------------
    glfwSetScrollCallback(win, scrollCallback);
    glfwSetMouseButtonCallback(win, mouseBtnCallback);
    glfwSetCursorPosCallback(win, cursorPosCallback);
    glfwSetKeyCallback(win, keyCallback);

    // ---- ImGui init --------------------------------------------------------
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    // Don't enable viewports — keeps rendering inside the GLFW window
    io.IniFilename = "pipe_engine_layout.ini";

    applyDarkTheme();
    ImGui_ImplGlfw_InitForOpenGL(win, true);
    ImGui_ImplOpenGL3_Init("#version 150");

    // ---- Renderer & UI -----------------------------------------------------
    MeshRenderer renderer;
    PipeUI       ui;
    g_renderer = &renderer;
    g_ui       = &ui;

    if (!renderer.init()) {
        std::cerr << "Renderer init failed (OpenGL error)\n";
        return 1;
    }

    // Wire up the callback so each time the mesh is rebuilt it goes to GPU
    ui.onMeshRebuilt = [&](const TriMesh& mesh) {
        renderer.uploadMesh(mesh);
    };

    // Init UI (loads default blueprint and builds first mesh)
    if (!blueprintPath.empty()) {
        PipeNetwork net = BlueprintParser::parse(blueprintPath);
        if (net.numSegments() > 0) {
            ui.network = std::move(net);
            ui.hasPipeStarted = true;
            // Serialize back to text buffer
            // (simple: just write a placeholder — editing isn't affected)
            ui.meshDirty = true;
            ui.setStatus("Loaded: " + blueprintPath);
        }
    } else {
        ui.init();
    }

    // Initial GPU upload
    if (!ui.mesh.vertices.empty())
        renderer.uploadMesh(ui.mesh);

    // ---- Main loop ---------------------------------------------------------
    double prevTime = glfwGetTime();

    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();

        double now = glfwGetTime();
        float  dt  = (float)(now - prevTime);
        prevTime = now;

        ui.update(dt);

        // Begin ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Draw all UI panels
        ui.drawAll(renderer);

        // Render 3D scene into the full framebuffer first,
        // then let ImGui overlay on top.
        int fbW, fbH;
        glfwGetFramebufferSize(win, &fbW, &fbH);
        renderer.render(fbW, fbH);

        // Flush ImGui on top
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(win);

        if (ui.m_wantsQuit)
            glfwSetWindowShouldClose(win, GLFW_TRUE);
    }

    // ---- Cleanup -----------------------------------------------------------
    renderer.cleanup();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(win);
    glfwTerminate();

    std::cout << "Goodbye.\n";
    return 0;
}
