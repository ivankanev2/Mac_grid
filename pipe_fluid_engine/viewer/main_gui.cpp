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
#include "voxelizer.h"
#include "camera.h"
#include "mesh_renderer.h"

// smoke_engine (3D sims)
#include "mac_smoke3d.h"
#include "mac_water3d.h"

// pipe_fluid_engine
#include "pipe_fluid/pipe_fluid_scene.h"
#include "pipe_fluid/volume_renderer.h"
#include "pipe_fluid/pipe_solver_boundary_data.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// ============================================================================
// Global state (needed in GLFW callbacks)
// ============================================================================
static MeshRenderer*                       g_renderer      = nullptr;
static pipe_fluid::VolumeOverlayRenderer*  g_volRenderer   = nullptr;
static pipe_fluid::PipeFluidScene*         g_scene         = nullptr;
static GLFWwindow*                         g_win            = nullptr;

// Viewer state
struct ViewerState {
    bool  playing        = false;
    bool  singleStep     = false;
    float simSpeed       = 1.0f;
    // Substep cap for the CFL-adaptive fluid step loop.  4 is a good
    // default for a narrow pipe with gravity active: at dx=1.5cm the
    // CFL-safe dt drops to ~7ms once particles hit ~2m/s, so 4 substeps
    // cover a full 60Hz frame with headroom.  Raise if the sim looks
    // like it's running in slow motion (accumulator is maxing out);
    // lower if the GUI stutters.
    int   stepsPerFrame  = 4;

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

    // Fluid overlay controls (camera matrices are auto-synced to the main view)
    float fluidDensity   = 3.0f;
    bool  fluidColorMode = false;
    float fluidAlpha     = 1.0f;
    int   fluidRenderScale = 1;   // 1=full-res, 2=half-res (good on weak GPU)

    // Water rendering mode.  When true, the volume renderer sphere-traces the
    // narrow-band SDF built from the FLIP particles each step; when false it
    // falls back to the old density-texture raymarch (which looks blocky at
    // pipe scale because the pipe is only 6-8 cells wide).
    bool  waterUseSdf    = true;

    // Volume renderer backend: 0=Auto, 1=CPU, 2=GPU.
    // The viewer recreates the renderer when this changes.
    int   backendChoice = 0;
    int   backendChoiceApplied = -1;
    const char* currentBackendName = "?";

    // Blueprint path input buffer
    char blueprintPath[512] = "../examples/demo_L.pipe";

    // Captured fluid state path input buffer (M2 Phase 2).
    char fluidStatePath[512] =
        "../../gaussian_splatting/fluid_capture/outputs/sim_state.bin";

    void setStatus(const std::string& s) {
        status = s;
        statusExpires = ImGui::GetTime() + 3.0;
    }
};
static ViewerState g_ui;

// ============================================================================
// Spawn-position wireframe marker
//
// The fluid source lives at absolute world coords (g_ui.sourceX/Y/Z) with
// radius g_ui.sourceR.  Without a visual anchor, positioning the source via
// the "Fluid source (world coords)" sliders is blind — the user only finds
// out they missed when water appears in the wrong place on play.  This helper
// draws a small wireframe sphere at the spawn point every frame so the user
// can see where water will emit before pressing Play.
//
// Self-contained: its own shader + VBO, lazy-initialized on first draw.
// Depth test is disabled so the marker is visible even when the spawn lies
// behind the pipe shell, and the draw happens after the volume overlay so
// nothing else writes over it.
// ============================================================================
struct SpawnMarkerDraw {
    GLuint  vao = 0, vbo = 0, prog = 0;
    GLint   locMVP = -1, locColor = -1;
    GLsizei vertexCount = 0;
    bool    ready = false;

    void initOnce() {
        if (ready) return;
        // Three orthogonal great circles of a unit sphere, as GL_LINES.
        const int   kSegs  = 48;
        const float kTwoPi = 6.2831853071795864769f;
        std::vector<float> verts;
        verts.reserve(static_cast<std::size_t>(3 * kSegs * 2 * 3));
        for (int axis = 0; axis < 3; ++axis) {
            for (int i = 0; i < kSegs; ++i) {
                const float a0 = kTwoPi *  i      / kSegs;
                const float a1 = kTwoPi * (i + 1) / kSegs;
                const float c0 = std::cos(a0), s0 = std::sin(a0);
                const float c1 = std::cos(a1), s1 = std::sin(a1);
                float p0[3] = {0,0,0}, p1[3] = {0,0,0};
                if      (axis == 0) { p0[0]=c0; p0[1]=s0; p1[0]=c1; p1[1]=s1; }
                else if (axis == 1) { p0[1]=c0; p0[2]=s0; p1[1]=c1; p1[2]=s1; }
                else                { p0[0]=c0; p0[2]=s0; p1[0]=c1; p1[2]=s1; }
                verts.insert(verts.end(), p0, p0 + 3);
                verts.insert(verts.end(), p1, p1 + 3);
            }
        }
        vertexCount = static_cast<GLsizei>(verts.size() / 3);

        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER,
                     static_cast<GLsizeiptr>(verts.size() * sizeof(float)),
                     verts.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glBindVertexArray(0);

        const char* vsSrc =
            "#version 150\n"
            "in vec3 aPos;\n"
            "uniform mat4 uMVP;\n"
            "void main() { gl_Position = uMVP * vec4(aPos, 1.0); }\n";
        const char* fsSrc =
            "#version 150\n"
            "uniform vec3 uColor;\n"
            "out vec4 fragColor;\n"
            "void main() { fragColor = vec4(uColor, 1.0); }\n";
        GLuint vs = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vs, 1, &vsSrc, nullptr);
        glCompileShader(vs);
        GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fs, 1, &fsSrc, nullptr);
        glCompileShader(fs);
        prog = glCreateProgram();
        glAttachShader(prog, vs);
        glAttachShader(prog, fs);
        glBindAttribLocation(prog, 0, "aPos");
        glLinkProgram(prog);
        glDeleteShader(vs);
        glDeleteShader(fs);
        locMVP   = glGetUniformLocation(prog, "uMVP");
        locColor = glGetUniformLocation(prog, "uColor");
        ready = true;
    }

    // Column-major 4x4 multiply: out = a * b (standard OpenGL convention).
    static void mul4(const float* a, const float* b, float* out) {
        for (int c = 0; c < 4; ++c)
            for (int r = 0; r < 4; ++r) {
                float s = 0.0f;
                for (int k = 0; k < 4; ++k) s += a[k * 4 + r] * b[c * 4 + k];
                out[c * 4 + r] = s;
            }
    }

    void draw(const float* view, const float* proj,
              float cx, float cy, float cz, float radius,
              float rCol, float gCol, float bCol) {
        initOnce();
        const float R = std::max(radius, 1.0e-4f);
        const float model[16] = {
            R, 0, 0, 0,
            0, R, 0, 0,
            0, 0, R, 0,
            cx, cy, cz, 1
        };
        float vm[16], mvp[16];
        mul4(view, model, vm);
        mul4(proj, vm,    mvp);

        const GLboolean depthWasOn = glIsEnabled(GL_DEPTH_TEST);
        glDisable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE);
        glUseProgram(prog);
        glUniformMatrix4fv(locMVP, 1, GL_FALSE, mvp);
        glUniform3f(locColor, rCol, gCol, bCol);
        glBindVertexArray(vao);
        glDrawArrays(GL_LINES, 0, vertexCount);
        glBindVertexArray(0);
        glUseProgram(0);
        glDepthMask(GL_TRUE);
        if (depthWasOn) glEnable(GL_DEPTH_TEST);
    }
};
static SpawnMarkerDraw g_spawnMarker;

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
//
// Each builder fully re-initialises the PipeFluidScene: it clears the
// existing network, adds one or more chains, then calls rebuild() so the
// voxelizer, masks, and render mesh are all in sync.  The viewer is
// expected to re-upload the pipe mesh afterwards.
//
// Conventions used throughout:
//   - Scenes live near world origin so the default orbit camera frames
//     them without the user having to pan/zoom manually.
//   - "bendR" values are chosen to be ~3x the pipe's outer radius, which
//     gives visually clean elbows at the default (5 cm inner / 6 cm outer)
//     pipe radii.
//   - Scenes that branch (T / Y / cross / manifold) use multiple calls to
//     beginChain() — each chain contributes its own pair of open pipe
//     mouths, which the voxelizer's open-ends post-pass carves out so
//     fluid can actually enter/exit the network at every free end.
// ============================================================================

// --- single-chain shapes ----------------------------------------------------

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

// Z-bend: two 90° turns in opposite senses, like a dog-leg.
static void buildDemoZ(pipe_fluid::PipeFluidScene& scene) {
    scene.clearNetwork();
    scene.beginNetwork(Vec3{0, 0, 0}, Vec3{0, 0, 1});
    scene.addStraight(0.7f);
    scene.addBend90(Vec3{1, 0, 0}, 0.15f);
    scene.addStraight(0.4f);
    scene.addBend90(Vec3{0, 0, 1}, 0.15f);
    scene.addStraight(0.7f);
    scene.rebuild();
}

// S-bend: same as Z but with gentler, slightly larger bends - good for
// showing smooth momentum redirection.
static void buildDemoS(pipe_fluid::PipeFluidScene& scene) {
    scene.clearNetwork();
    scene.beginNetwork(Vec3{0, 0, 0}, Vec3{0, 0, 1});
    scene.addStraight(0.5f);
    scene.addBend90(Vec3{0, 1, 0}, 0.20f);
    scene.addStraight(0.4f);
    scene.addBend90(Vec3{0, 0, 1}, 0.20f);
    scene.addStraight(0.5f);
    scene.addBend90(Vec3{0,-1, 0}, 0.20f);
    scene.addStraight(0.4f);
    scene.addBend90(Vec3{0, 0, 1}, 0.20f);
    scene.addStraight(0.5f);
    scene.rebuild();
}

// Zig-zag: repeated sharp turns in the same plane.
static void buildDemoZigZag(pipe_fluid::PipeFluidScene& scene) {
    scene.clearNetwork();
    scene.beginNetwork(Vec3{0, 0, 0}, Vec3{0, 0, 1});
    const float leg = 0.45f;
    const float bendR = 0.12f;
    scene.addStraight(leg);
    scene.addBend90(Vec3{ 1, 0, 0}, bendR);
    scene.addStraight(leg);
    scene.addBend90(Vec3{ 0, 0, 1}, bendR);
    scene.addStraight(leg);
    scene.addBend90(Vec3{-1, 0, 0}, bendR);
    scene.addStraight(leg);
    scene.addBend90(Vec3{ 0, 0, 1}, bendR);
    scene.addStraight(leg);
    scene.addBend90(Vec3{ 1, 0, 0}, bendR);
    scene.addStraight(leg);
    scene.rebuild();
}

// Serpentine / flat coil: long snake that folds back on itself.
// Same topology as zig-zag but with more turns and bigger radii - stresses
// the solver harder and gives lots of visible flow.
static void buildDemoSerpentine(pipe_fluid::PipeFluidScene& scene) {
    scene.clearNetwork();
    scene.beginNetwork(Vec3{0, 0, 0}, Vec3{0, 0, 1});
    const float leg   = 0.60f;
    const float shelf = 0.25f;
    const float bR    = 0.15f;
    for (int k = 0; k < 3; ++k) {
        scene.addStraight(leg);
        scene.addBend90(Vec3{1, 0, 0}, bR);
        scene.addStraight(shelf);
        scene.addBend90(Vec3{0, 0,-1}, bR);
        scene.addStraight(leg);
        scene.addBend90(Vec3{1, 0, 0}, bR);
        scene.addStraight(shelf);
        scene.addBend90(Vec3{0, 0, 1}, bR);
    }
    scene.addStraight(leg);
    scene.rebuild();
}

// 3D spiral: climb up Y while rotating around in XZ.  Four 90° turns per
// revolution, with a small straight "riser" along Y after each bend.
static void buildDemoSpiral(pipe_fluid::PipeFluidScene& scene) {
    scene.clearNetwork();
    scene.beginNetwork(Vec3{0, 0, 0}, Vec3{0, 0, 1});
    const float run  = 0.35f;   // horizontal run between bends
    const float rise = 0.18f;   // vertical riser between bends
    const float bR   = 0.12f;

    // Two turns per revolution (approximate helix using axis-aligned bends).
    const Vec3 steps[] = {
        Vec3{ 1, 0, 0}, Vec3{ 0, 0,-1},
        Vec3{-1, 0, 0}, Vec3{ 0, 0, 1}
    };
    const int turns = 8;  // 2 full revolutions
    scene.addStraight(run);
    for (int i = 0; i < turns; ++i) {
        scene.addBend90(Vec3{0, 1, 0}, bR);   // tip up to climb
        scene.addStraight(rise);
        scene.addBend90(steps[i % 4], bR);    // return to horizontal + rotate
        scene.addStraight(run);
    }
    scene.rebuild();
}

// --- branching / multi-chain scenes -----------------------------------------

// T-junction: one straight "trunk" from -X to +X, one branch coming down
// from +Y into the middle of the trunk.  Three open ends.
static void buildDemoT(pipe_fluid::PipeFluidScene& scene) {
    scene.clearNetwork();
    // Trunk (chain 1): +X horizontal
    scene.beginNetwork(Vec3{-0.7f, 0.f, 0.f}, Vec3{1, 0, 0});
    scene.addStraight(1.4f);
    // Branch (chain 2): down into trunk midpoint
    scene.beginChain(Vec3{0.f, 0.7f, 0.f}, Vec3{0, -1, 0});
    scene.addStraight(0.7f);
    scene.rebuild();
}

// Y-junction: two inlets converging to a shared outlet.
// Implemented as three chains that all START at the junction (origin) and
// run outward.  The outward tangent at the junction endpoint of every chain
// points INTO the fluid core of the other chains, so the voxelizer's
// interior-junction guard correctly skips carving at the junction and only
// carves the three external pipe mouths.
static void buildDemoY(pipe_fluid::PipeFluidScene& scene) {
    scene.clearNetwork();
    // Outlet: junction -> +X
    scene.beginNetwork(Vec3{0, 0, 0}, Vec3{1, 0, 0});
    scene.addStraight(0.70f);
    // Upper inlet: junction -> upper-left
    scene.beginChain(Vec3{0, 0, 0}, Vec3{-1, 1, 0});
    scene.addStraight(0.80f);
    // Lower inlet: junction -> lower-left
    scene.beginChain(Vec3{0, 0, 0}, Vec3{-1, -1, 0});
    scene.addStraight(0.80f);
    scene.rebuild();
}

// Cross-junction: two straight pipes intersecting at right angles.  Four
// open ends.
static void buildDemoCross(pipe_fluid::PipeFluidScene& scene) {
    scene.clearNetwork();
    // Horizontal bar
    scene.beginNetwork(Vec3{-0.7f, 0.f, 0.f}, Vec3{1, 0, 0});
    scene.addStraight(1.4f);
    // Vertical bar
    scene.beginChain(Vec3{0.f, -0.7f, 0.f}, Vec3{0, 1, 0});
    scene.addStraight(1.4f);
    scene.rebuild();
}

// Manifold: single inlet splitting into three parallel outlets via T-
// junction stubs.  Useful to see pressure/mass splitting across branches.
static void buildDemoManifold(pipe_fluid::PipeFluidScene& scene) {
    scene.clearNetwork();
    // Trunk: inlet on the left, sealed on the right by extending past the
    // last branch.  Single chain across the full width.
    scene.beginNetwork(Vec3{-0.9f, 0.f, 0.f}, Vec3{1, 0, 0});
    scene.addStraight(1.8f);
    // Three branches going down at x = -0.45, 0, +0.45
    scene.beginChain(Vec3{-0.45f, 0.f, 0.f}, Vec3{0, -1, 0});
    scene.addStraight(0.55f);
    scene.beginChain(Vec3{ 0.00f, 0.f, 0.f}, Vec3{0, -1, 0});
    scene.addStraight(0.55f);
    scene.beginChain(Vec3{ 0.45f, 0.f, 0.f}, Vec3{0, -1, 0});
    scene.addStraight(0.55f);
    scene.rebuild();
}

// ============================================================================
// ImGui panels
// ============================================================================
static void drawScenePanel(pipe_fluid::PipeFluidScene& scene) {
    ImGui::Begin("Scene");

    const auto& cfg = scene.config();
    pipe_fluid::PipeFluidScene::Config nextCfg = cfg;

    ImGui::SeparatorText("Voxel grid");
    ImGui::SliderFloat("cell size (m)", &nextCfg.grid.cellSize, 0.002f, 0.05f, "%.3f");
    ImGui::SliderFloat("padding (m)",   &nextCfg.grid.padding,  0.01f, 0.5f, "%.2f");
    ImGui::SliderFloat("gravity pad (m)", &nextCfg.grid.gravityPadding, 0.0f, 1.0f, "%.2f");

    ImGui::SeparatorText("Default pipe radii");
    ImGui::SliderFloat("inner R (m)", &nextCfg.geometry.defaultInnerRadius, 0.01f, 0.2f, "%.3f");
    ImGui::SliderFloat("outer R (m)", &nextCfg.geometry.defaultOuterRadius, 0.01f, 0.25f, "%.3f");
    if (nextCfg.geometry.defaultOuterRadius < nextCfg.geometry.defaultInnerRadius + 0.001f)
        nextCfg.geometry.defaultOuterRadius = nextCfg.geometry.defaultInnerRadius + 0.001f;

    ImGui::SeparatorText("Active sims");
    ImGui::Checkbox("3D Smoke (MACSmoke3D)", &nextCfg.sim.enableSmoke);
    ImGui::Checkbox("3D Water (MACWater3D)", &nextCfg.sim.enableWater);

    ImGui::SeparatorText("Simulation");
    ImGui::SliderFloat("base dt (s)", &nextCfg.sim.dt, 1.0f / 240.0f, 1.0f / 24.0f, "%.4f", ImGuiSliderFlags_Logarithmic);

    ImGui::SeparatorText("Water surface");
    ImGui::SliderFloat("particle radius scale", &nextCfg.waterSurface.particleRadiusScale, 0.5f, 2.5f, "%.2f");
    ImGui::SliderFloat("SDF band (cells)", &nextCfg.waterSurface.bandCells, 0.0f, 8.0f, "%.1f");

    // Field-wise comparison (memcmp is unreliable due to bool padding).
    const bool configChanged =
        nextCfg.grid.cellSize                  != cfg.grid.cellSize                  ||
        nextCfg.grid.padding                   != cfg.grid.padding                   ||
        nextCfg.grid.gravityPadding            != cfg.grid.gravityPadding            ||
        nextCfg.geometry.defaultInnerRadius    != cfg.geometry.defaultInnerRadius    ||
        nextCfg.geometry.defaultOuterRadius    != cfg.geometry.defaultOuterRadius    ||
        nextCfg.sim.enableSmoke                != cfg.sim.enableSmoke                ||
        nextCfg.sim.enableWater                != cfg.sim.enableWater                ||
        nextCfg.sim.dt                         != cfg.sim.dt                         ||
        nextCfg.waterSurface.particleRadiusScale != cfg.waterSurface.particleRadiusScale ||
        nextCfg.waterSurface.bandCells         != cfg.waterSurface.bandCells;
    if (configChanged) scene.setConfig(nextCfg);

    ImGui::SeparatorText("Demo scenes");
    // Helper lambda: run a scene builder and push the new mesh to the GPU.
    // Centralises the "build -> upload mesh -> status" trio each button needs.
    auto runScene = [&](const char* label, void (*fn)(pipe_fluid::PipeFluidScene&)) {
        if (ImGui::Button(label)) {
            fn(scene);
            g_renderer->uploadMesh(scene.pipeMesh());
            g_ui.setStatus(std::string("Built ") + label);
        }
    };

    ImGui::TextUnformatted("Simple shapes:");
    runScene("L-pipe",   buildDemoL);      ImGui::SameLine();
    runScene("U-bend",   buildDemoU);      ImGui::SameLine();
    runScene("Z-bend",   buildDemoZ);      ImGui::SameLine();
    runScene("S-curve",  buildDemoS);

    ImGui::TextUnformatted("Long / stress tests:");
    runScene("Zig-zag",    buildDemoZigZag);     ImGui::SameLine();
    runScene("Serpentine", buildDemoSerpentine); ImGui::SameLine();
    runScene("Spiral",     buildDemoSpiral);

    ImGui::TextUnformatted("Branching (multi-chain):");
    runScene("T-junction", buildDemoT);        ImGui::SameLine();
    runScene("Y-junction", buildDemoY);        ImGui::SameLine();
    runScene("Cross",      buildDemoCross);    ImGui::SameLine();
    runScene("Manifold",   buildDemoManifold);

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

    ImGui::SeparatorText("Captured fluid state");
    ImGui::InputText("fluid path", g_ui.fluidStatePath, sizeof(g_ui.fluidStatePath));
    if (ImGui::Button("Load fluid state")) {
        std::string err;
        if (scene.loadFluidState(g_ui.fluidStatePath, &err)) {
            // Captured-state scenes have no pipe mesh — clear the renderer's
            // mesh so we don't keep showing the previous blueprint's pipe.
            g_renderer->uploadMesh(scene.pipeMesh());
            g_ui.setStatus(std::string("Loaded fluid state ") + g_ui.fluidStatePath);
        } else {
            g_ui.setStatus(std::string("Fluid state load failed: ") + err);
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
    ImGui::SliderFloat("alpha scale",    &g_ui.fluidAlpha,   0.0f, 2.0f, "%.2f");
    ImGui::Checkbox("color mode",        &g_ui.fluidColorMode);
    // SDF sphere-trace vs. voxel density raymarch for water.  SDF path produces
    // a continuous liquid surface regardless of pipe voxel resolution; off
    // falls back to the older density-integration path.
    ImGui::Checkbox("water: SDF surface", &g_ui.waterUseSdf);

    // Backend selector — lets the user flip between CPU and GPU at runtime.
    // "Auto" picks based on the GL vendor string (see pickAutoBackend()).
    const char* backends[] = {"Auto", "CPU", "GPU"};
    ImGui::Combo("renderer", &g_ui.backendChoice, backends, IM_ARRAYSIZE(backends));
    ImGui::SliderInt("render scale", &g_ui.fluidRenderScale, 1, 3);
    if (g_volRenderer) {
        ImGui::TextDisabled("active: %s", g_volRenderer->backendName());
    }

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
    ImGui::Text("Openings:%d",           s.openingCells);
    ImGui::Separator();
    ImGui::Text("Chains:      %d",       s.chainCount);
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
        ImGui::Text("Water max speed:    %.3f", ws.maxSpeed);
        ImGui::Text("Water max div:      %.6f", ws.maxDivergence);
        ImGui::Text("Pre-proj max div:   %.6f", ws.preProjectionMaxDivergence);
        ImGui::Text("Post-proj max div:  %.6f", ws.postProjectionMaxDivergence);
        ImGui::Text("Water pressure iters: %d", ws.pressureIters);
        ImGui::Text("Water pressure ms:    %.3f", ws.pressureMs);
        ImGui::Text("Water step ms:        %.3f", ws.lastStepMs);
        ImGui::Text("Water backend:      %s", ws.backendName ? ws.backendName : "?");
        ImGui::Separator();
        ImGui::Text("Pressure active cells:      %d", ws.pressureActiveCellCount);
        ImGui::Text("Pressure components:        %d", ws.pressureComponentCount);
        ImGui::Text("Pressure neighbor links:    %d", ws.pressureNeighborLinkCount);
        ImGui::Text("Pressure Dirichlet faces:   %d", ws.pressureDirichletFaceCount);
        ImGui::Text("Pressure open faces:        %d", ws.pressureOpenFaceCount);
        ImGui::Text("Pressure blocked faces:     %d", ws.pressureBlockedFaceCount);
        ImGui::Text("Pressure weighted faces:    %d", ws.pressureWeightedFaceCount);
        ImGui::Separator();
        const auto& sb = scene.solverBoundary();
        ImGui::Text("Pipe face-open min:         %.6f", sb.minFaceOpen);
        ImGui::Text("Pipe faces < 0.99:          %d", sb.faceOpenCountLt099);
        ImGui::Text("Pipe faces < 0.50:          %d", sb.faceOpenCountLt050);
        ImGui::Text("Pipe faces closed:          %d", sb.faceOpenCountClosed);
        ImGui::Text("Water face-open min:        %.6f", ws.minFaceOpen);
        ImGui::Text("Water faces < 0.99:         %d", ws.faceOpenCountLt099);
        ImGui::Text("Water faces < 0.50:         %d", ws.faceOpenCountLt050);
        ImGui::Text("Water faces closed:         %d", ws.faceOpenCountClosed);
        ImGui::Separator();
        ImGui::Text("Near-closed face flux count: %d", ws.nearClosedFaceFluxCount);
        ImGui::Text("Max near-closed face flux:   %.6f", ws.maxNearClosedFaceFlux);
        // Patch A (diagnostic): mid-open face flux [0.25, 0.75), per step.
        ImGui::Text("Mid-open face flux count:    %d", ws.midOpenFaceFluxCount);
        ImGui::Text("Max mid-open face flux:      %.6f", ws.maxMidOpenFaceFlux);
        ImGui::Text("Particles near wall:         %d", ws.particlesNearWallCount);
        ImGui::Text("Particles inside wall:       %d", ws.particlesInsideWallCount);
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

    // CLI parsing.  The first non-flag positional is treated as a blueprint
    // (.pipe) path for backwards compatibility.  --fluid-state <path> takes
    // an alternative captured-fluid-state binary; if both are given,
    // --fluid-state wins.
    std::string blueprintPath;
    std::string fluidStatePath;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--fluid-state" && i + 1 < argc) {
            fluidStatePath = argv[++i];
        } else if (arg.rfind("--fluid-state=", 0) == 0) {
            fluidStatePath = arg.substr(std::string("--fluid-state=").size());
        } else if (!arg.empty() && arg[0] != '-' && blueprintPath.empty()) {
            blueprintPath = arg;
        }
    }

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

    // Volume overlay: picks backend according to g_ui.backendChoice.
    // The renderer lives in a unique_ptr so we can recreate it mid-session
    // when the user flips backends.
    std::unique_ptr<pipe_fluid::VolumeOverlayRenderer> volRenderer;
    auto rebuildVolumeRenderer = [&]() {
        if (volRenderer) { volRenderer->shutdown(); volRenderer.reset(); }
        pipe_fluid::VolumeOverlayRenderer::Backend want;
        switch (g_ui.backendChoice) {
            case 1: want = pipe_fluid::VolumeOverlayRenderer::Backend::CPU; break;
            case 2: want = pipe_fluid::VolumeOverlayRenderer::Backend::GPU; break;
            default: want = pipe_fluid::pickAutoBackend(); break;
        }
        volRenderer = pipe_fluid::makeVolumeRenderer(want);
        g_volRenderer = volRenderer.get();
        g_ui.backendChoiceApplied = g_ui.backendChoice;
        g_ui.currentBackendName = volRenderer ? volRenderer->backendName() : "none";
        std::cout << "[PipeFluidEngine] Volume renderer: "
                  << g_ui.currentBackendName << "\n";
    };
    rebuildVolumeRenderer();
    if (!volRenderer) {
        std::cerr << "Volume renderer init failed (both backends unavailable)\n";
        return 1;
    }

    pipe_fluid::PipeFluidScene::Config cfg;
    cfg.grid.cellSize    = 0.015f;   // 1.5 cm
    cfg.grid.padding     = 0.10f;
    cfg.sim.enableSmoke = true;
    cfg.sim.enableWater = false;
    cfg.sim.dt          = 1.0f / 60.0f;
    pipe_fluid::PipeFluidScene scene(cfg);
    g_scene = &scene;

    bool loadedSomething = false;
    if (!fluidStatePath.empty()) {
        std::string err;
        if (!scene.loadFluidState(fluidStatePath, &err)) {
            std::cerr << "Fluid state load failed: " << err << "\n";
        } else {
            // Sync the UI text field so the panel reflects what got loaded.
            std::strncpy(g_ui.fluidStatePath, fluidStatePath.c_str(),
                         sizeof(g_ui.fluidStatePath) - 1);
            g_ui.fluidStatePath[sizeof(g_ui.fluidStatePath) - 1] = '\0';
            loadedSomething = true;
        }
    }
    if (!loadedSomething) {
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
    }
    if (!scene.pipeMesh().vertices.empty()) renderer.uploadMesh(scene.pipeMesh());

    // ---- Main loop ---------------------------------------------------------
    double prev = glfwGetTime();

    // CFL-adaptive substepping state.  Mirrors the pattern in smoke_engine's
    // main loop (see smoke_engine/src/main.cpp:1770–1846).  Without this we
    // were stepping the fluid ONCE per render frame at a fixed dt=1/60s,
    // which (at dx=1.5cm and the velocities gravity builds up during free
    // fall) routinely exceeds CFL=1 and forces the FLIP particles to
    // advect through more than one cell per step.  That shows up as
    // "block" motion because the pressure projection can't correct an
    // advection that already overshot.  Here we accumulate real frame
    // time and substep the scene with a CFL-safe dt until the accumulator
    // drains (bounded by a step cap so the sim can't stall the GUI).
    double simAccumulator = 0.0;
    constexpr float kCFL       = 0.9f;
    constexpr float kDtMin     = 1.0f / 1000.0f;  // 1ms floor
    // "steps per frame" UI slider now controls the substep cap: if the
    // accumulator would demand more CFL-safe substeps than this, we drop
    // the remainder.  Higher = better fluid fidelity under fast flow,
    // lower = better GUI responsiveness if the sim gets expensive.

    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();
        double now = glfwGetTime();
        float  frameDt = (float)(now - prev); prev = now;
        if (frameDt > 0.1f) frameDt = 0.1f;  // clamp pauses / breakpoints

        // ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        drawScenePanel(scene);
        drawFluidPanel(scene);
        drawStatsPanel(scene);

        // Step fluid
        if (g_ui.playing || g_ui.singleStep) {
            simAccumulator += (double)frameDt * (double)g_ui.simSpeed;

            // Hard single-step: just take one dt-capped step regardless of
            // the accumulator.
            if (g_ui.singleStep) {
                simAccumulator = std::max(simAccumulator, (double)scene.config().sim.dt);
            }

            int substeps = 0;
            const auto simCfg = scene.config();
            const float cflDx = scene.cellSize() > 0.f ? scene.cellSize() : simCfg.grid.cellSize;
            const int maxSubsteps = std::max(1, g_ui.stepsPerFrame);

            while (simAccumulator > 0.0 && substeps < maxSubsteps) {
                // Figure out the fastest thing in the sim so we can pick a
                // CFL-safe dt.  Particle speed dominates velocity field speed
                // in FLIP/APIC so we query both.
                float maxSpeed = 0.f;
                if (auto* w = scene.water()) {
                    maxSpeed = std::max(maxSpeed, w->stats().maxSpeed);
                }
                if (auto* s = scene.smoke()) {
                    maxSpeed = std::max(maxSpeed, s->stats().maxSpeed);
                }

                const float dtCFL = kCFL * cflDx / (maxSpeed + 1e-6f);
                float dt = std::min(simCfg.sim.dt, dtCFL);
                dt = std::max(dt, kDtMin);
                if (dt > (float)simAccumulator) dt = (float)simAccumulator;

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

                scene.step(dt);

                simAccumulator -= (double)dt;
                ++substeps;
            }

            // If we hit the cap, drop the remainder so the sim can't lag
            // forever when it's expensive for a burst of frames.
            if (substeps == maxSubsteps) simAccumulator = 0.0;

            g_ui.singleStep = false;
        } else {
            // Paused: keep the accumulator empty so we don't fire a burst
            // of steps the moment the user hits play.
            simAccumulator = 0.0;
        }

        // Render 3D pipe scene
        int fbW, fbH; glfwGetFramebufferSize(win, &fbW, &fbH);
        renderer.render(fbW, fbH);

        // ---- Volume overlay (world-space raymarch, shared with pipe camera) --
        // Recreate renderer if the user flipped backends.
        if (g_ui.backendChoice != g_ui.backendChoiceApplied) {
            rebuildVolumeRenderer();
        }

        if (volRenderer) {
            // Build the same proj/view the pipe mesh was just drawn with.
            pipe_fluid::VolumeView vv;
            const float aspect = (float)fbW / std::max(1.f, (float)fbH);
            renderer.camera.buildProjMatrix(vv.proj, aspect);
            renderer.camera.buildViewMatrix(vv.view);
            Vec3 camPos = renderer.camera.position();
            vv.camPosX = camPos.x;
            vv.camPosY = camPos.y;
            vv.camPosZ = camPos.z;

            // World-space AABB of the voxel grid.
            const VoxelGrid& vg = scene.voxels();
            vv.originX = vg.origin.x;
            vv.originY = vg.origin.y;
            vv.originZ = vg.origin.z;
            vv.dx = vg.dx;
            vv.nx = vg.nx; vv.ny = vg.ny; vv.nz = vg.nz;
            vv.fbWidth = fbW;
            vv.fbHeight = fbH;

            pipe_fluid::VolumeSettings vs;
            vs.useColor     = g_ui.fluidColorMode;
            vs.alphaScale   = g_ui.fluidAlpha;
            vs.densityScale = g_ui.fluidDensity;
            vs.renderScale  = std::max(1, g_ui.fluidRenderScale);
            vs.stepsPerPixel = 96;

            // Smoke first, then water on top (if both enabled).
            if (scene.smoke()) {
                auto* s = scene.smoke();
                volRenderer->setVolume(s->smoke, s->temp, s->solid,
                                       s->nx, s->ny, s->nz);
                volRenderer->render(vv, vs);
            }
            if (scene.water()) {
                auto* w = scene.water();
                // Water has no temperature field; pass an empty temp.
                static const std::vector<float> kEmptyTemp;
                // IMPORTANT: pass the walls-only pipe mask here, not the
                // simulator's internal domain-border sealing.  The renderer
                // uses a nearest-neighbour solid lookup to decide where to
                // terminate rays, so only true pipe walls should be marked
                // solid for this pass.
                volRenderer->setVolume(w->water, kEmptyTemp,
                                       scene.renderSolidMask(),
                                       w->nx, w->ny, w->nz);

                // Upload the narrow-band SDF rebuilt from FLIP particles at
                // the end of every scene.step().  When useSdf is on, the
                // renderer sphere-traces this field instead of integrating
                // the density texture, producing a continuous liquid surface
                // independent of voxel resolution.  Band matches the clamp
                // used by buildLiquidSdfFromParticles() in pipe_fluid_scene
                // (3 * dx).  Passing an empty SDF is a no-op inside the
                // renderer and lets the density path run as a fallback.
                const auto& sdf = scene.waterSDF();
                const float sdfBand = scene.waterSdfBand();
                volRenderer->setWaterSdf(sdf, scene.nx(), scene.ny(),
                                         scene.nz(), sdfBand);

                pipe_fluid::VolumeSettings wsv = vs;
                wsv.useColor = false;            // blue-ish tint not yet in transfer fn
                wsv.alphaScale *= 0.6f;
                wsv.useSdf    = g_ui.waterUseSdf;
                volRenderer->render(vv, wsv);
            }
        }

        // --- Spawn-position marker ------------------------------------------
        // Drawn last in 3D (after the volume overlay) with depth test off so
        // it is visible even if the spawn lies behind the pipe shell.  Use
        // this to position the fluid source inside the pipe before pressing
        // Play; water only actually spawns on Play.
        {
            const float aspect = (float)fbW / std::max(1.f, (float)fbH);
            float proj[16], view[16];
            renderer.camera.buildProjMatrix(proj, aspect);
            renderer.camera.buildViewMatrix(view);
            g_spawnMarker.draw(view, proj,
                               g_ui.sourceX, g_ui.sourceY, g_ui.sourceZ,
                               g_ui.sourceR,
                               0.20f, 1.00f, 0.80f);  // bright cyan/teal
        }
        // ----------------------------------------------------------------------

        // ImGui on top
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(win);
    }

    if (volRenderer) { volRenderer->shutdown(); volRenderer.reset(); }
    g_volRenderer = nullptr;
    renderer.cleanup();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
