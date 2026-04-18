// ============================================================================
// pipe_fluid_engine — interactive viewer
//
// Combines pipe_engine (geometry + voxelizer) and smoke_engine (3D MAC smoke
// and water) through the PipeFluidScene integration layer.
//
// Controls:
//   Left-drag   -> orbit pipe view
//   Right-drag  -> pan
//   Scroll      -> zoom
//   Space       -> play/pause fluid
//   S           -> single step
//   R           -> rebuild (re-voxelize and reset fluids)
//
// Usage:
//   PipeFluidEngine                      -> default demo L-pipe
//   PipeFluidEngine <blueprint.pipe>     -> load blueprint on startup
// ============================================================================

#ifdef __APPLE__
#  define GL_SILENCE_DEPRECATION
#  include <OpenGL/gl3.h>
#else
#  include <GL/gl.h>
#endif

#include <GLFW/glfw3.h>

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

// pipe_engine (header-only)
#include "vec3.h"
#include "pipe_network.h"
#include "mesh_generator.h"
#include "camera.h"
#include "mesh_renderer.h"

// smoke_engine (3D sims + renderer)
#include "mac_smoke3d.h"
#include "mac_water3d.h"
#include "smoke_renderer.h"

// pipe_fluid_engine
#include "pipe_fluid/pipe_fluid_scene.h"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// ============================================================================
// Global state (needed in GLFW callbacks)
// ============================================================================
static MeshRenderer*               g_renderer      = nullptr;
static SmokeRenderer*              g_fluidRenderer = nullptr;
static pipe_fluid::PipeFluidScene* g_scene         = nullptr;
static GLFWwindow*                 g_win            = nullptr;

// Viewer state
struct ViewerState {
    bool  playing        = false;
    bool  singleStep     = false;
    float simSpeed       = 1.0f;
    int   stepsPerFrame  = 1;

    // Scene builder inputs
    float builderStartX  = 0.f, builderStartY = 0.f, builderStartZ = 0.f;
    float builderDirX    = 0.f, builderDirY = 0.f, builderDirZ = 1.f;
    float nextStraightLen = 0.5f;
    int   nextBendAxis    = 0;   // 0=+X 1=+Y 2=-X 3=-Y
    float nextBendRadius  = 0.15f;

    // Smoke injector — default on pipe centreline, 15 cm along the first run
    float sourceX = 0.00f, sourceY = 0.00f, sourceZ = 0.15f;
    float sourceR = 0.03f;
    float sourceAmount = 1.0f;
    float sourceVelX = 0.f, sourceVelY = 0.f, sourceVelZ = 1.0f;

    // Continuous emission toggles
    bool emitSmoke = false;
    bool emitWater = false;

    // Status message
    std::string status;
    double statusExpires = 0.0;

    // Fluid overlay controls (camera angles are auto-synced to the main view)
    float fluidDensity   = 3.0f;
    bool  fluidColorMode = false;

    // Blueprint path input buffer
    char blueprintPath[512] = "../examples/demo_L.pipe";

    void setStatus(const std::string& s) {
        status = s;
        statusExpires = ImGui::GetTime() + 3.0;
    }
};
static ViewerState g_ui;

// ============================================================================
// GLFW callbacks
// ============================================================================
static void errorCallback(int, const char* desc) {
    std::cerr << "[GLFW] " << desc << "\n";
}

static void scrollCallback(GLFWwindow*, double, double dy) {
    if (g_renderer && !ImGui::GetIO().WantCaptureMouse)
        g_renderer->camera.onScroll((float)dy);
}

static void mouseBtnCallback(GLFWwindow*, int button, int action, int) {
    if (ImGui::GetIO().WantCaptureMouse) return;
    if (!g_renderer) return;
    double mx, my; glfwGetCursorPos(g_win, &mx, &my);
    g_renderer->camera.onMouseButton(button, action == GLFW_PRESS, (float)mx, (float)my);
}

static void cursorPosCallback(GLFWwindow*, double x, double y) {
    if (ImGui::GetIO().WantCaptureMouse) return;
    if (g_renderer) g_renderer->camera.onMouseMove((float)x, (float)y);
}

static void keyCallback(GLFWwindow*, int key, int, int action, int) {
    if (action != GLFW_PRESS) return;
    switch (key) {
        case GLFW_KEY_SPACE: g_ui.playing = !g_ui.playing; break;
        case GLFW_KEY_S:     g_ui.singleStep = true; break;
        case GLFW_KEY_R:
            if (g_scene) { g_scene->rebuild(); g_ui.setStatus("Rebuilt scene"); }
            if (g_renderer && g_scene) g_renderer->uploadMesh(g_scene->pipeMesh());
            break;
    }
}

// ============================================================================
// ImGui dark theme (matches sibling engines)
// ============================================================================
static void applyDarkTheme() {
    ImGuiStyle& s = ImGui::GetStyle();
    s.WindowRounding    = 6.f;
    s.FrameRounding     = 4.f;
    s.GrabRounding      = 4.f;
    s.ScrollbarRounding = 4.f;
    s.TabRounding       = 4.f;
    s.WindowBorderSize  = 1.f;

    ImVec4* c = s.Colors;
    c[ImGuiCol_WindowBg]      = {0.12f, 0.12f, 0.14f, 1.f};
    c[ImGuiCol_ChildBg]       = {0.10f, 0.10f, 0.12f, 1.f};
    c[ImGuiCol_FrameBg]       = {0.18f, 0.18f, 0.22f, 1.f};
    c[ImGuiCol_Button]        = {0.20f, 0.40f, 0.60f, 0.60f};
    c[ImGuiCol_ButtonHovered] = {0.26f, 0.50f, 0.72f, 1.0f};
    c[ImGuiCol_ButtonActive]  = {0.06f, 0.53f, 0.98f, 1.0f};
    c[ImGuiCol_Header]        = {0.20f, 0.40f, 0.60f, 0.55f};
    c[ImGuiCol_HeaderHovered] = {0.26f, 0.48f, 0.70f, 0.80f};
    c[ImGuiCol_HeaderActive]  = {0.30f, 0.55f, 0.80f, 1.0f};
    c[ImGuiCol_TitleBgActive] = {0.16f, 0.30f, 0.48f, 1.0f};
    c[ImGuiCol_Text]          = {0.90f, 0.90f, 0.92f, 1.f};
    c[ImGuiCol_TextDisabled]  = {0.50f, 0.50f, 0.55f, 1.f};
}

// ============================================================================
// Demo scenes
// ============================================================================
static void buildDemoL(pipe_fluid::PipeFluidScene& scene) {
    scene.clearNetwork();
    scene.beginNetwork(Vec3{0, 0, 0}, Vec3{0, 0, 1});
    scene.addStraight(1.0f);
    scene.addBend90(Vec3{1, 0, 0}, 0.15f);
    scene.addStraight(0.8f);
    scene.rebuild();
}

static void buildDemoU(pipe_fluid::PipeFluidScene& scene) {
    scene.clearNetwork();
    scene.beginNetwork(Vec3{0, 0, 0}, Vec3{0, 0, 1});
    scene.addStraight(0.8f);
    scene.addBend90(Vec3{1, 0, 0}, 0.15f);
    scene.addStraight(0.6f);
    scene.addBend90(Vec3{0, 0, -1}, 0.15f);
    scene.addStraight(0.8f);
    scene.rebuild();
}

// ============================================================================
// ImGui panels
// ============================================================================
static void drawScenePanel(pipe_fluid::PipeFluidScene& scene) {
    ImGui::Begin("Scene");

    auto cfg = scene.config();
    pipe_fluid::PipeFluidScene::Config nextCfg = cfg;

    ImGui::SeparatorText("Voxel grid");
    ImGui::SliderFloat("cell size (m)", &nextCfg.cellSize, 0.002f, 0.05f, "%.3f");
    ImGui::SliderFloat("padding (m)",   &nextCfg.padding,  0.01f, 0.5f, "%.2f");

    ImGui::SeparatorText("Default pipe radii");
    ImGui::SliderFloat("inner R (m)", &nextCfg.defaultInnerRadius, 0.01f, 0.2f, "%.3f");
    ImGui::SliderFloat("outer R (m)", &nextCfg.defaultOuterRadius, 0.01f, 0.25f, "%.3f");
    if (nextCfg.defaultOuterRadius < nextCfg.defaultInnerRadius + 0.001f)
        nextCfg.defaultOuterRadius = nextCfg.defaultInnerRadius + 0.001f;

    ImGui::SeparatorText("Active sims");
    ImGui::Checkbox("3D Smoke (MACSmoke3D)", &nextCfg.enableSmoke);
    ImGui::Checkbox("3D Water (MACWater3D)", &nextCfg.enableWater);

    // Field-wise comparison (memcmp is unreliable due to bool padding).
    const bool configChanged =
        nextCfg.cellSize            != cfg.cellSize            ||
        nextCfg.padding             != cfg.padding             ||
        nextCfg.defaultInnerRadius  != cfg.defaultInnerRadius  ||
        nextCfg.defaultOuterRadius  != cfg.defaultOuterRadius  ||
        nextCfg.enableSmoke         != cfg.enableSmoke         ||
        nextCfg.enableWater         != cfg.enableWater;
    if (configChanged) scene.setConfig(nextCfg);

    ImGui::SeparatorText("Demo scenes");
    if (ImGui::Button("L-pipe demo"))       { buildDemoL(scene); g_renderer->uploadMesh(scene.pipeMesh()); g_ui.setStatus("Built L-pipe"); }
    ImGui::SameLine();
    if (ImGui::Button("U-bend demo"))       { buildDemoU(scene); g_renderer->uploadMesh(scene.pipeMesh()); g_ui.setStatus("Built U-bend"); }

    ImGui::SeparatorText("Blueprint");
    ImGui::InputText("path", g_ui.blueprintPath, sizeof(g_ui.blueprintPath));
    if (ImGui::Button("Load blueprint")) {
        std::string err;
        if (scene.loadBlueprint(g_ui.blueprintPath, &err)) {
            scene.rebuild();
            g_renderer->uploadMesh(scene.pipeMesh());
            g_ui.setStatus(std::string("Loaded ") + g_ui.blueprintPath);
        } else {
            g_ui.setStatus(std::string("Load failed: ") + err);
        }
    }

    ImGui::SeparatorText("Programmatic builder");
    ImGui::InputFloat3("start",     &g_ui.builderStartX);
    ImGui::InputFloat3("direction", &g_ui.builderDirX);
    if (ImGui::Button("Begin network")) {
        scene.clearNetwork();
        scene.beginNetwork(Vec3{g_ui.builderStartX, g_ui.builderStartY, g_ui.builderStartZ},
                            Vec3{g_ui.builderDirX,   g_ui.builderDirY,   g_ui.builderDirZ});
        g_ui.setStatus("Network started");
    }
    ImGui::SliderFloat("straight length (m)", &g_ui.nextStraightLen, 0.1f, 2.5f, "%.2f");
    if (ImGui::Button("+ straight")) {
        scene.addStraight(g_ui.nextStraightLen);
        g_ui.setStatus("Added straight segment");
    }
    const char* axisNames[] = {"+X", "+Y", "-X", "-Y"};
    ImGui::Combo("bend axis", &g_ui.nextBendAxis, axisNames, IM_ARRAYSIZE(axisNames));
    ImGui::SliderFloat("bend radius", &g_ui.nextBendRadius, 0.05f, 0.5f, "%.2f");
    if (ImGui::Button("+ bend90")) {
        Vec3 dir;
        switch (g_ui.nextBendAxis) {
            case 0: dir = Vec3{ 1, 0, 0}; break;
            case 1: dir = Vec3{ 0, 1, 0}; break;
            case 2: dir = Vec3{-1, 0, 0}; break;
            default: dir = Vec3{ 0,-1, 0}; break;
        }
        scene.addBend90(dir, g_ui.nextBendRadius);
        g_ui.setStatus("Added bend90");
    }

    ImGui::SeparatorText("");
    if (ImGui::Button("Rebuild (voxelize + reset fluids)")) {
        scene.rebuild();
        g_renderer->uploadMesh(scene.pipeMesh());
        g_ui.setStatus("Rebuilt scene");
    }

    ImGui::End();
}

static void drawFluidPanel(pipe_fluid::PipeFluidScene& scene) {
    ImGui::Begin("Fluid");

    if (ImGui::Button(g_ui.playing ? "Pause" : "Play")) g_ui.playing = !g_ui.playing;
    ImGui::SameLine();
    if (ImGui::Button("Step"))  g_ui.singleStep = true;
    ImGui::SameLine();
    if (ImGui::Button("Reset fluids")) { scene.resetFluids(); g_ui.setStatus("Fluids reset"); }

    ImGui::SliderFloat("sim speed",       &g_ui.simSpeed,      0.1f, 4.f,  "%.2fx");
    ImGui::SliderInt  ("steps per frame", &g_ui.stepsPerFrame, 1, 8);

    ImGui::SeparatorText("Overlay");
    ImGui::SliderFloat("fluid density",  &g_ui.fluidDensity, 0.5f, 20.f, "%.1f");
    ImGui::Checkbox("color mode",        &g_ui.fluidColorMode);

    ImGui::SeparatorText("Fluid source (world coords)");
    ImGui::InputFloat3("centre",   &g_ui.sourceX);
    ImGui::SliderFloat("radius",   &g_ui.sourceR,      0.005f, 0.2f, "%.3f");
    ImGui::SliderFloat("amount",   &g_ui.sourceAmount, 0.0f,   5.0f, "%.2f");
    ImGui::InputFloat3("velocity", &g_ui.sourceVelX);

    // Continuous emitters — active every sim step while playing
    ImGui::Checkbox("Emit smoke continuously", &g_ui.emitSmoke);
    ImGui::SameLine();
    ImGui::Checkbox("Emit water continuously", &g_ui.emitWater);

    // One-shot puff buttons
    if (ImGui::Button("Inject smoke puff")) {
        scene.addSmokeSourceSphere({g_ui.sourceX, g_ui.sourceY, g_ui.sourceZ},
                                    g_ui.sourceR, g_ui.sourceAmount,
                                    {g_ui.sourceVelX, g_ui.sourceVelY, g_ui.sourceVelZ});
        g_ui.setStatus("Injected smoke puff");
    }
    ImGui::SameLine();
    if (ImGui::Button("Inject water puff")) {
        scene.addWaterSourceSphere({g_ui.sourceX, g_ui.sourceY, g_ui.sourceZ},
                                    g_ui.sourceR,
                                    {g_ui.sourceVelX, g_ui.sourceVelY, g_ui.sourceVelZ});
        g_ui.setStatus("Injected water puff");
    }

    ImGui::End();
}

static void drawStatsPanel(pipe_fluid::PipeFluidScene& scene) {
    ImGui::Begin("Stats");
    auto s = scene.stats();
    ImGui::Text("Grid:    %d x %d x %d", s.nx, s.ny, s.nz);
    ImGui::Text("Cell:    %.4f m",       scene.cellSize());
    ImGui::Text("Solids:  %d",           s.solidCells);
    ImGui::Text("Fluid:   %d",           s.fluidCells);
    ImGui::Separator();
    ImGui::Text("Segments:    %d",       s.segmentCount);
    ImGui::Text("Pipe length: %.3f m",   s.totalPipeLength);
    ImGui::Separator();
    if (scene.smoke()) {
        auto& ss = scene.smoke()->stats();
        ImGui::Text("Smoke active cells: %d", ss.activeCells);
        ImGui::Text("Smoke max speed:    %.3f", ss.maxSpeed);
    }
    if (scene.water()) {
        auto& ws = scene.water()->stats();
        ImGui::Text("Water particles:    %d", ws.particleCount);
        ImGui::Text("Water liquid cells: %d", ws.liquidCells);
    }
    if (!g_ui.status.empty() && ImGui::GetTime() < g_ui.statusExpires) {
        ImGui::Separator();
        ImGui::TextColored({0.55f, 0.85f, 1.f, 1.f}, "%s", g_ui.status.c_str());
    }
    ImGui::End();
}

// (Fluid View panel removed — fluid is now rendered as a transparent overlay
//  directly on top of the 3D pipe scene, synced to the main orbit camera.)

// ============================================================================
// Entry point
// ============================================================================
int main(int argc, char* argv[]) {
    std::cout << "=== Pipe Fluid Engine v0.1 ===\n";

    std::string blueprintPath;
    if (argc >= 2) blueprintPath = argv[1];

    // ---- GLFW init ---------------------------------------------------------
    glfwSetErrorCallback(errorCallback);
    if (!glfwInit()) { std::cerr << "GLFW init failed\n"; return 1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
#endif
    glfwWindowHint(GLFW_SAMPLES, 4);

    GLFWwindow* win = glfwCreateWindow(1500, 950,
                                       "Pipe Fluid Engine | Vizior Research",
                                       nullptr, nullptr);
    if (!win) { std::cerr << "GLFW window failed\n"; glfwTerminate(); return 1; }
    g_win = win;
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);

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
    io.IniFilename  = "pipe_fluid_layout.ini";
    applyDarkTheme();
    ImGui_ImplGlfw_InitForOpenGL(win, true);
    ImGui_ImplOpenGL3_Init("#version 150");

    // ---- Renderer + scene --------------------------------------------------
    MeshRenderer renderer;
    if (!renderer.init()) { std::cerr << "Renderer init failed\n"; return 1; }
    g_renderer = &renderer;

    SmokeRenderer fluidRenderer(512, 512);
    g_fluidRenderer = &fluidRenderer;

    pipe_fluid::PipeFluidScene::Config cfg;
    cfg.cellSize    = 0.015f;   // 1.5 cm
    cfg.padding     = 0.10f;
    cfg.enableSmoke = true;
    cfg.enableWater = false;
    cfg.dt          = 1.0f / 60.0f;
    pipe_fluid::PipeFluidScene scene(cfg);
    g_scene = &scene;

    if (!blueprintPath.empty()) {
        std::string err;
        if (!scene.loadBlueprint(blueprintPath, &err)) {
            std::cerr << "Blueprint load failed: " << err << "\n";
            buildDemoL(scene);
        } else {
            scene.rebuild();
        }
    } else {
        buildDemoL(scene);
    }
    if (!scene.pipeMesh().vertices.empty()) renderer.uploadMesh(scene.pipeMesh());

    // ---- Main loop ---------------------------------------------------------
    double prev = glfwGetTime();

    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();
        double now = glfwGetTime();
        float  dt  = (float)(now - prev); prev = now;
        (void)dt;

        // ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        drawScenePanel(scene);
        drawFluidPanel(scene);
        drawStatsPanel(scene);

        // Step fluid
        if (g_ui.playing || g_ui.singleStep) {
            for (int i = 0; i < g_ui.stepsPerFrame; ++i) {
                // Apply continuous emitters each sub-step
                if (g_ui.emitSmoke)
                    scene.addSmokeSourceSphere(
                        {g_ui.sourceX, g_ui.sourceY, g_ui.sourceZ},
                        g_ui.sourceR, g_ui.sourceAmount,
                        {g_ui.sourceVelX, g_ui.sourceVelY, g_ui.sourceVelZ});
                if (g_ui.emitWater)
                    scene.addWaterSourceSphere(
                        {g_ui.sourceX, g_ui.sourceY, g_ui.sourceZ},
                        g_ui.sourceR,
                        {g_ui.sourceVelX, g_ui.sourceVelY, g_ui.sourceVelZ});
                scene.step(cfg.dt * g_ui.simSpeed);
            }
            g_ui.singleStep = false;
        }

        // Render 3D pipe scene
        int fbW, fbH; glfwGetFramebufferSize(win, &fbW, &fbH);
        renderer.render(fbW, fbH);

        // ---- Fluid overlay ---------------------------------------------------
        // Sync the fluid renderer's camera to the main orbit camera so the
        // volume render is from the same viewpoint as the pipe geometry.
        // Then composite the smoke/water texture on top as a transparent layer.
        if (g_fluidRenderer) {
            // Resize fluid renderer to match framebuffer
            if (g_fluidRenderer->width()  != fbW ||
                g_fluidRenderer->height() != fbH)
                g_fluidRenderer->resize(fbW, fbH);

            SmokeRenderSettings sr;
            sr.transparentBackground = true;   // key: lets pipe geometry show through
            sr.useColor   = g_ui.fluidColorMode;
            sr.alphaScale = g_ui.fluidDensity / 3.0f;   // map density slider to alphaScale

            WaterRenderSettings wr;

            const float yaw   = g_renderer->camera.yawDeg;
            const float pitch = g_renderer->camera.pitchDeg;
            const float zoom  = 1.0f;   // SmokeRenderer zoom is relative to its own box

            if (scene.smoke()) {
                auto* s = scene.smoke();
                g_fluidRenderer->updateSmokeFromVolume(
                    s->smoke, s->temp, s->solid,
                    s->nx, s->ny, s->nz,
                    yaw, pitch, zoom, g_ui.fluidDensity, sr);

                // Draw as fullscreen transparent overlay via ImGui background list
                ImGui::GetBackgroundDrawList()->AddImage(
                    static_cast<ImTextureID>(g_fluidRenderer->smokeTex()),
                    ImVec2(0, 0), ImVec2((float)fbW, (float)fbH),
                    ImVec2(0, 1), ImVec2(1, 0));   // flip Y to match OpenGL convention
            }
            if (scene.water()) {
                auto* w = scene.water();
                g_fluidRenderer->updateWaterFromVolume(
                    w->water, w->solid,
                    w->nx, w->ny, w->nz,
                    0, yaw, pitch, zoom, g_ui.fluidDensity, 0.5f, wr);

                ImGui::GetBackgroundDrawList()->AddImage(
                    static_cast<ImTextureID>(g_fluidRenderer->waterTex()),
                    ImVec2(0, 0), ImVec2((float)fbW, (float)fbH),
                    ImVec2(0, 1), ImVec2(1, 0));
            }
        }
        // ----------------------------------------------------------------------

        // ImGui on top
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(win);
    }

    renderer.cleanup();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
