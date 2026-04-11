#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <array>
#include <vector>
#include <filesystem>
#include <string>

#include "UI/panels.h"

#ifdef __APPLE__
  #define GL_SILENCE_DEPRECATION
  #define GLFW_INCLUDE_NONE
  #include <GLFW/glfw3.h>
  #include <OpenGL/gl3.h>
#elif defined(_WIN32)
  #define WIN32_LEAN_AND_MEAN
  #define NOMINMAX
  #include <windows.h>
  #define GLFW_INCLUDE_NONE
  #include <GLFW/glfw3.h>
  #include <GL/gl.h>
#else
  #define GLFW_INCLUDE_NONE
  #include <GLFW/glfw3.h>
  #include <GL/gl.h>
#endif

#include "Sim/mac_smoke_sim.h"
#include "Sim/mac_water_sim.h"
#include "Sim/mac_water3d.h"
#include "Sim/mac_smoke3d.h"
#include "Sim/pressure_solver.h"
#include "Renderer/smoke_renderer.h"

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

#include "Sim/mac_coupled_sim.h"
#include "Sim/text_mask.h"



// window size and sim resolution
static int NX = 256;
static int NY = 256;
static const int NZ = 64;

// Persistent text mask (reused across resets).
// Always rasterized at at least 512x512 so the letters stay crisp at any grid resolution.
static std::vector<uint8_t> g_textMask;
static int g_textMaskW = 0;
static int g_textMaskH = 0;

// time variables
double lastTime = 0.0;
double accumulator = 0.0;
float simSpeed = 1.0f;          // 1.0 = real-time, 2.0 = 2x faster, etc.
int maxStepsPerFrame = 8;       // cap to avoid death-spiral

namespace {

struct IconVec2 {
    float x = 0.0f;
    float y = 0.0f;
};

struct IconColor {
    unsigned char r = 0;
    unsigned char g = 0;
    unsigned char b = 0;
    unsigned char a = 0;
};

static constexpr int kThemeDark = 0;
static constexpr int kThemeLight = 1;
static constexpr IconColor kLogoRed{232, 29, 47, 255};
static constexpr IconColor kLogoLightBg{250, 250, 249, 255};
static constexpr IconColor kLogoDarkBg{23, 27, 31, 255};

static float clamp01(float x) {
    return std::max(0.0f, std::min(1.0f, x));
}

static float distSqToSegment(float px, float py, IconVec2 a, IconVec2 b) {
    const float vx = b.x - a.x;
    const float vy = b.y - a.y;
    const float wx = px - a.x;
    const float wy = py - a.y;
    const float vv = vx * vx + vy * vy;
    const float t = (vv > 1.0e-12f) ? clamp01((wx * vx + wy * vy) / vv) : 0.0f;
    const float dx = px - (a.x + t * vx);
    const float dy = py - (a.y + t * vy);
    return dx * dx + dy * dy;
}

static float distToPolyline(float px, float py, const IconVec2* pts, int count) {
    float best = 1.0e9f;
    for (int i = 0; i + 1 < count; ++i) {
        best = std::min(best, distSqToSegment(px, py, pts[i], pts[i + 1]));
    }
    return std::sqrt(best);
}

static float coverageForDisk(float dist, float radius, float aa) {
    if (dist <= radius - aa) return 1.0f;
    if (dist >= radius + aa) return 0.0f;
    return clamp01((radius + aa - dist) / std::max(1.0e-6f, 2.0f * aa));
}

static float coverageForStroke(float dist, float halfWidth, float aa) {
    if (dist <= halfWidth - aa) return 1.0f;
    if (dist >= halfWidth + aa) return 0.0f;
    return clamp01((halfWidth + aa - dist) / std::max(1.0e-6f, 2.0f * aa));
}

static void blendOver(unsigned char& dstR,
                      unsigned char& dstG,
                      unsigned char& dstB,
                      unsigned char& dstA,
                      IconColor src,
                      float coverage)
{
    const float sa = clamp01((src.a / 255.0f) * coverage);
    const float da = dstA / 255.0f;
    const float outA = sa + da * (1.0f - sa);
    if (outA <= 1.0e-6f) {
        dstR = dstG = dstB = dstA = 0;
        return;
    }

    const float sr = src.r / 255.0f;
    const float sg = src.g / 255.0f;
    const float sb = src.b / 255.0f;
    const float dr = dstR / 255.0f;
    const float dg = dstG / 255.0f;
    const float db = dstB / 255.0f;

    const float outR = (sr * sa + dr * da * (1.0f - sa)) / outA;
    const float outG = (sg * sa + dg * da * (1.0f - sa)) / outA;
    const float outB = (sb * sa + db * da * (1.0f - sa)) / outA;

    dstR = (unsigned char)std::lround(clamp01(outR) * 255.0f);
    dstG = (unsigned char)std::lround(clamp01(outG) * 255.0f);
    dstB = (unsigned char)std::lround(clamp01(outB) * 255.0f);
    dstA = (unsigned char)std::lround(clamp01(outA) * 255.0f);
}

static std::string findExistingPath(std::initializer_list<const char*> candidates) {
    for (const char* rel : candidates) {
        if (rel && rel[0] && std::filesystem::exists(rel)) {
            return std::string(rel);
        }
    }
    return {};
}

static bool loadViziorFonts(ImGuiIO& io, float dpiScale = 1.0f) {
    io.Fonts->Clear();

    ImFontConfig cfg{};
    cfg.OversampleH = 4;
    cfg.OversampleV = 4;
    cfg.PixelSnapH = false;
    cfg.RasterizerMultiply = 1.08f;

    const float fontSize = 17.0f * std::max(1.0f, dpiScale);
    const std::string roboto = findExistingPath({
        "external/imgui/misc/fonts/Roboto-Medium.ttf",
        "../external/imgui/misc/fonts/Roboto-Medium.ttf",
        "../../external/imgui/misc/fonts/Roboto-Medium.ttf"
    });

    if (!roboto.empty()) {
        if (ImFont* font = io.Fonts->AddFontFromFileTTF(roboto.c_str(), fontSize, &cfg)) {
            io.FontDefault = font;
            return true;
        }
    }

    io.FontDefault = io.Fonts->AddFontDefault();
    return false;
}

static void rasterViziorIcon(int w, int h, int themeMode, std::vector<unsigned char>& out) {
    out.assign((size_t)w * (size_t)h * 4u, 0u);

    const IconColor circleFill = (themeMode == kThemeLight) ? kLogoLightBg : kLogoDarkBg;
    const float aa = 0.90f / (float)std::max(1, std::max(w, h));
    const float circleR = 0.43f;
    const float borderHalf = 0.014f;

    const IconVec2 serifL[3] = {{0.24f, 0.29f}, {0.36f, 0.29f}, {0.40f, 0.33f}};
    const IconVec2 serifR[3] = {{0.76f, 0.29f}, {0.64f, 0.29f}, {0.60f, 0.33f}};
    const IconVec2 outerV[3] = {{0.33f, 0.31f}, {0.50f, 0.72f}, {0.67f, 0.31f}};
    const IconVec2 innerCutV[3] = {{0.40f, 0.35f}, {0.50f, 0.61f}, {0.60f, 0.35f}};
    const IconVec2 innerCoreV[3] = {{0.45f, 0.40f}, {0.50f, 0.53f}, {0.55f, 0.40f}};

    for (int j = 0; j < h; ++j) {
        for (int i = 0; i < w; ++i) {
            const float x = (i + 0.5f) / (float)w;
            const float y = (j + 0.5f) / (float)h;
            const float dx = x - 0.5f;
            const float dy = y - 0.5f;
            const float dCircle = std::sqrt(dx * dx + dy * dy);
            unsigned char r = 0, g = 0, b = 0, a = 0;

            const float circleCov = coverageForDisk(dCircle, circleR, aa);
            blendOver(r, g, b, a, circleFill, circleCov);

            if (themeMode == kThemeLight) {
                const float ringCov = circleCov * coverageForStroke(std::fabs(dCircle - circleR), borderHalf, aa * 1.6f);
                blendOver(r, g, b, a, kLogoRed, ringCov);
            }

            const float serifDist = std::min(distToPolyline(x, y, serifL, 3), distToPolyline(x, y, serifR, 3));
            const float outerDist = std::min(distToPolyline(x, y, outerV, 3), serifDist);
            const float cutDist = distToPolyline(x, y, innerCutV, 3);
            const float coreDist = distToPolyline(x, y, innerCoreV, 3);

            blendOver(r, g, b, a, kLogoRed, circleCov * coverageForStroke(outerDist, 0.072f, aa * 1.8f));
            blendOver(r, g, b, a, circleFill, circleCov * coverageForStroke(cutDist, 0.046f, aa * 1.6f));
            blendOver(r, g, b, a, kLogoRed, circleCov * coverageForStroke(coreDist, 0.022f, aa * 1.4f));

            const size_t idx = ((size_t)j * (size_t)w + (size_t)i) * 4u;
            out[idx + 0] = r;
            out[idx + 1] = g;
            out[idx + 2] = b;
            out[idx + 3] = a;
        }
    }
}

static void setViziorWindowIcon(GLFWwindow* win, int themeMode) {
    std::vector<unsigned char> icon32;
    std::vector<unsigned char> icon64;
    rasterViziorIcon(32, 32, themeMode, icon32);
    rasterViziorIcon(64, 64, themeMode, icon64);
    GLFWimage images[2];
    images[0].width = 32;
    images[0].height = 32;
    images[0].pixels = icon32.data();
    images[1].width = 64;
    images[1].height = 64;
    images[1].pixels = icon64.data();
    glfwSetWindowIcon(win, 2, images);
}

struct StartupWindowConfig {
    int width = 1480;
    int height = 920;
    int posX = 0;
    int posY = 0;
    float uiScale = 0.80f;
    bool hasPosition = false;
};

static StartupWindowConfig computeStartupWindowConfig() {
    StartupWindowConfig cfg;
    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    if (!monitor) return cfg;

    int workX = 0, workY = 0, workW = 0, workH = 0;
    glfwGetMonitorWorkarea(monitor, &workX, &workY, &workW, &workH);
    if (workW <= 0 || workH <= 0) return cfg;

    const float fit = std::clamp(std::min((float)workW / 1480.0f, (float)workH / 920.0f), 0.72f, 1.18f);
    cfg.width = std::clamp((int)std::lround(1480.0f * fit), 960, std::max(960, workW));
    cfg.height = std::clamp((int)std::lround(920.0f * fit), 640, std::max(640, workH));
    cfg.posX = workX + std::max(0, (workW - cfg.width) / 2);
    cfg.posY = workY + std::max(0, (workH - cfg.height) / 2);
    cfg.uiScale = std::clamp(0.80f * std::sqrt(std::max(0.55f, fit)), 0.65f, 0.95f);
    cfg.hasPosition = true;
    return cfg;
}

struct VolumeRenderTargetSize {
    int width = 0;
    int height = 0;
};

static VolumeRenderTargetSize computeVolumeRenderTargetSize(GLFWwindow* win,
                                                            float logicalWidth,
                                                            float logicalHeight,
                                                            float renderScale,
                                                            int minDim = 192,
                                                            int maxDim = 1024)
{
    int fbW = 0;
    int fbH = 0;
    glfwGetFramebufferSize(win, &fbW, &fbH);

    float dpiX = 1.0f;
    float dpiY = 1.0f;
    glfwGetWindowContentScale(win, &dpiX, &dpiY);
    dpiX = std::max(1.0f, dpiX);
    dpiY = std::max(1.0f, dpiY);

    const float fallbackLogicalW = std::max(320.0f, 0.46f * (float)std::max(1, fbW) / dpiX);
    const float fallbackLogicalH = std::max(220.0f, 0.46f * (float)std::max(1, fbH) / dpiY);
    const float safeLogicalW = (logicalWidth > 1.0f) ? logicalWidth : fallbackLogicalW;
    const float safeLogicalH = (logicalHeight > 1.0f) ? logicalHeight : fallbackLogicalH;
    const float safeScale = std::clamp(renderScale, 0.35f, 2.0f);

    int targetW = (int)std::lround(safeLogicalW * dpiX * safeScale);
    int targetH = (int)std::lround(safeLogicalH * dpiY * safeScale);

    targetW = std::clamp(targetW, minDim, maxDim);
    targetH = std::clamp(targetH, minDim, maxDim);
    return VolumeRenderTargetSize{targetW, targetH};
}

} // namespace

int main()
{
    if (!glfwInit()) return 1;
    lastTime = glfwGetTime();

    // OpenGL / GLFW hints
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);

    const StartupWindowConfig startupWindow = computeStartupWindowConfig();
    GLFWwindow* win = glfwCreateWindow(startupWindow.width, startupWindow.height, "Vizior | Fluid Research Engine", nullptr, nullptr);
    if (!win) return 1;
    if (startupWindow.hasPosition) {
        glfwSetWindowPos(win, startupWindow.posX, startupWindow.posY);
    }

    UI::Settings ui;
    ui.windowWidth = startupWindow.width;
    ui.windowHeight = startupWindow.height;
    ui.uiScale = startupWindow.uiScale;
    UI::Probe probe;
    glfwGetWindowSize(win, &ui.windowWidth, &ui.windowHeight);

    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);

    // ImGui init
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = "vizior_layout.ini";

    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags &= ~ImGuiConfigFlags_ViewportsEnable;   // disable OS-windows

    float dpiX = 1.0f;
    float dpiY = 1.0f;
    glfwGetWindowContentScale(win, &dpiX, &dpiY);
    loadViziorFonts(io, std::max(dpiX, dpiY));

    UI::ApplyViziorTheme(ui.themeMode);
    ImGui::GetStyle().ScaleAllSizes(ui.uiScale);
    io.FontGlobalScale = ui.uiScale;
    setViziorWindowIcon(win, ui.themeMode);
    int   appliedTheme   = ui.themeMode;
    float appliedUiScale = ui.uiScale;

    // tweak style when viewports enabled
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
        ImGuiStyle& style = ImGui::GetStyle();
        style.WindowRounding = 0.0f;
        style.Colors[ImGuiCol_WindowBg].w = 1.0f;
    }

    ImGui_ImplGlfw_InitForOpenGL(win, true);
    ImGui_ImplOpenGL3_Init("#version 150");

    // Renderer
    SmokeRenderer renderer(NX, NY);
    SmokeRenderer water3DRenderer(NX, NY);
    SmokeRenderer smoke3DRenderer(NX, NY);
    SmokeRenderer coupledRenderer(NX, NY);

    // Simulator
    float dx = 1.0f / NX;
    float dt_initial = 0.02f;


    // One shared pressure solver instance used by BOTH smoke and water.
    // The solver is reconfigured per-solve (nx/ny/BCs/masks), but it can reuse
    // internal allocations across both sims.
    PressureSolver sharedPressureSolver;

    auto dxForGrid = [](int gx, int gy, int gz) {
        return 1.0f / (float)std::max({gx, gy, gz, 1});
    };

    MAC2D sim(NX, NY, dx, dt_initial);
    MACWater waterSim(NX, NY, dx, dt_initial);
    MACWater3D water3D(1, 1, 1, 1.0f, dt_initial);
    MACSmoke3D smoke3D(1, 1, 1, 1.0f, dt_initial);
    MACCoupledSim coupled(NX, NY, dx, dt_initial);

    // Make both sims use the same solver instance.
    sim.setSharedPressureSolver(&sharedPressureSolver);
    waterSim.setSharedPressureSolver(&sharedPressureSolver);
    coupled.setSharedPressureSolver(&sharedPressureSolver);

    // Rasterize MBZUAI text mask and apply to both simulations
    const std::string robotoPath = findExistingPath({
        "external/imgui/misc/fonts/Roboto-Medium.ttf",
        "../external/imgui/misc/fonts/Roboto-Medium.ttf",
        "../../external/imgui/misc/fonts/Roboto-Medium.ttf"
    });
    // Rasterize at the larger of (grid size, 512) so text stays sharp at any resolution.
    g_textMaskW = std::max(NX, 512);
    g_textMaskH = std::max(NY, 512);
    g_textMask = rasterizeTextMask(
        "MBZUAI",
        robotoPath.empty() ? "external/imgui/misc/fonts/Roboto-Medium.ttf" : robotoPath.c_str(),
        g_textMaskW, g_textMaskH,
        0.5f,   // center vertically
        0.15f   // text height = 15% of mask height
    );

    auto applyIntroText = [&]() {
        if (g_textMask.empty()) return;
        sim.addSolidText(g_textMask, g_textMaskW, g_textMaskH);
        waterSim.waterHeld = true;
        waterSim.addWaterTextParticles(g_textMask, g_textMaskW, g_textMaskH, 4);
        waterSim.syncSolidsFrom(sim);
    };

    auto activateWorkspace = [&](int workspace) {
        workspace = std::clamp(workspace, (int)UI::kWorkspaceSmoke2D, (int)UI::kWorkspaceCoupled);

        sim.reset();
        waterSim.reset();
        coupled.reset();
        applyIntroText();

        smoke3D.reset(1, 1, 1, 1.0f, smoke3D.dt);
        water3D.reset(1, 1, 1, 1.0f, water3D.dt);

        if (workspace == UI::kWorkspaceSmoke3D) {
            const int nx3 = std::max(1, ui.smoke3DNX);
            const int ny3 = std::max(1, ui.smoke3DNY);
            const int nz3 = std::max(1, ui.smoke3DNZ);
            smoke3D.reset(nx3, ny3, nz3, dxForGrid(nx3, ny3, nz3), smoke3D.dt);
        } else if (workspace == UI::kWorkspaceWater3D) {
            const int nx3 = std::max(1, ui.water3DNX);
            const int ny3 = std::max(1, ui.water3DNY);
            const int nz3 = std::max(1, ui.water3DNZ);
            water3D.reset(nx3, ny3, nz3, dxForGrid(nx3, ny3, nz3), water3D.dt);
        }

        accumulator = 0.0;
    };

    int activeWorkspace = std::clamp(ui.activeWorkspace, (int)UI::kWorkspaceSmoke2D, (int)UI::kWorkspaceCoupled);
    activateWorkspace(activeWorkspace);

    // Reinitialise all 2D sims at a new grid resolution.
    auto reinit2DGrid = [&]() {
        NX = std::max(32, ui.sim2DNX);
        NY = std::max(32, ui.sim2DNY);
        const float newDX = 1.0f / NX;

        sim      = MAC2D(NX, NY, newDX, dt_initial);
        waterSim = MACWater(NX, NY, newDX, dt_initial);
        coupled  = MACCoupledSim(NX, NY, newDX, dt_initial);
        sim.setSharedPressureSolver(&sharedPressureSolver);
        waterSim.setSharedPressureSolver(&sharedPressureSolver);
        coupled.setSharedPressureSolver(&sharedPressureSolver);

        renderer.resize(NX, NY);
        water3DRenderer.resize(NX, NY);
        smoke3DRenderer.resize(NX, NY);
        coupledRenderer.resize(NX, NY);

        g_textMaskW = std::max(NX, 512);
        g_textMaskH = std::max(NY, 512);
        g_textMask = rasterizeTextMask(
            "MBZUAI",
            robotoPath.empty() ? "external/imgui/misc/fonts/Roboto-Medium.ttf" : robotoPath.c_str(),
            g_textMaskW, g_textMaskH, 0.5f, 0.15f);

        // Keep the viewport filling roughly the same screen area after a resize.
        ui.viewScale = std::max(1.0f, std::min(12.0f, (256.0f * 5.0f) / float(NX)));

        activateWorkspace(activeWorkspace);
    };

    while (!glfwWindowShouldClose(win))
    {
        glfwPollEvents();

        if (appliedTheme != ui.themeMode || appliedUiScale != ui.uiScale) {
            UI::ApplyViziorTheme(ui.themeMode);
            ImGui::GetStyle().ScaleAllSizes(ui.uiScale);
            io.FontGlobalScale = ui.uiScale;
            setViziorWindowIcon(win, ui.themeMode);
            appliedTheme   = ui.themeMode;
            appliedUiScale = ui.uiScale;
        }

        const int requestedWorkspace = std::clamp(ui.activeWorkspace, (int)UI::kWorkspaceSmoke2D, (int)UI::kWorkspaceCoupled);
        if (requestedWorkspace != activeWorkspace) {
            activeWorkspace = requestedWorkspace;
            activateWorkspace(activeWorkspace);
        }

        // step sim (if playing)
        sim.smokeDissipation = ui.smokeDissipation;
        sim.tempDissipation  = ui.tempDissipation;

        waterSim.waterDissipation = ui.waterDissipation;
        waterSim.waterGravity     = ui.waterGravity;
        waterSim.velDamping       = ui.waterVelDamping;
        waterSim.openTop          = ui.waterOpenTop;

        MACWater3D::Params p3 = water3D.params;
        bool p3Changed = false;
        auto assignFloatW = [&](float& dst, float value) {
            if (dst != value) { dst = value; p3Changed = true; }
        };
        auto assignIntW = [&](int& dst, int value) {
            if (dst != value) { dst = value; p3Changed = true; }
        };
        auto assignBoolW = [&](bool& dst, bool value) {
            if (dst != value) { dst = value; p3Changed = true; }
        };

        assignFloatW(p3.waterDissipation, ui.waterDissipation);
        assignFloatW(p3.gravity, ui.waterGravity);
        assignFloatW(p3.velDamping, ui.waterVelDamping);
        assignBoolW(p3.openTop, ui.waterOpenTop);
        assignIntW(p3.pressureIters, ui.water3DPressureIters);
        assignIntW(p3.pressureSolverMode, ui.water3DPressureSolverMode);
        assignBoolW(p3.useAPIC, ui.water3DUseAPIC);
        assignFloatW(p3.flipBlend, ui.water3DFlipBlend);
        assignFloatW(p3.pressureOmega, ui.water3DPressureOmega);
        assignBoolW(p3.volumePreserveRhsMean, ui.water3DVolumePreserve);
        assignFloatW(p3.volumePreserveStrength, ui.water3DVolumePreserveStrength);
        assignIntW(p3.reseedRelaxIters, ui.water3DRelaxIters);
        assignFloatW(p3.reseedRelaxStrength, ui.water3DRelaxStrength);

        MACWater3D::BackendPreference requestedWaterBackend = MACWater3D::BackendPreference::Auto;
        switch (ui.water3DBackendMode) {
            case 1: requestedWaterBackend = MACWater3D::BackendPreference::CPU; break;
            case 2: requestedWaterBackend = MACWater3D::BackendPreference::CUDA; break;
            default: break;
        }
        if (water3D.backendPreferenceMode() != requestedWaterBackend) {
            water3D.setBackendPreference(requestedWaterBackend);
        }

        if (p3Changed) {
            water3D.setParams(p3);
        }

        MACSmoke3D::Params s3 = smoke3D.params;
        bool s3Changed = false;
        auto assignFloatS = [&](float& dst, float value) {
            if (dst != value) { dst = value; s3Changed = true; }
        };
        auto assignIntS = [&](int& dst, int value) {
            if (dst != value) { dst = value; s3Changed = true; }
        };
        auto assignBoolS = [&](bool& dst, bool value) {
            if (dst != value) { dst = value; s3Changed = true; }
        };

        assignFloatS(s3.smokeDissipation, ui.smokeDissipation);
        assignFloatS(s3.tempDissipation, ui.tempDissipation);
        assignFloatS(s3.gravity, ui.smoke3DGravity);
        assignFloatS(s3.buoyancyScale, ui.smoke3DBuoyancyScale);
        assignFloatS(s3.velDamping, ui.smoke3DVelDamping);
        assignFloatS(s3.viscosity, ui.smoke3DViscosity);
        assignFloatS(s3.smokeDiffusivity, ui.smoke3DSmokeDiffusivity);
        assignFloatS(s3.tempDiffusivity, ui.smoke3DTempDiffusivity);
        assignBoolS(s3.openTop, ui.smoke3DOpenTop);
        assignIntS(s3.pressureIters, ui.smoke3DPressureIters);
        assignIntS(s3.pressureSolverMode, ui.smoke3DPressureSolverMode);
        assignFloatS(s3.pressureOmega, ui.smoke3DPressureOmega);

        if (s3Changed) {
            smoke3D.setParams(s3);
        }

        coupled.waterDissipation = ui.waterDissipation;
        coupled.waterGravity     = ui.waterGravity;
        coupled.velDamping       = ui.waterVelDamping;
        coupled.setOpenTop(false);  // Combined view: closed top for now

        coupled.smokeDissipation = ui.smokeDissipation;
        coupled.tempDissipation  = ui.tempDissipation;
        coupled.useMacCormack    = sim.useMacCormack; // or ui toggle if you expose one

        double now = glfwGetTime();
        double frameDt = now - lastTime;
        lastTime = now;

        // avoid huge jumps (dragging window, breakpoint, etc.)
        if (frameDt > 0.1) frameDt = 0.1;

        if (ui.playing) {
            accumulator += frameDt * simSpeed;

            int steps = 0;
            while (accumulator > 0.0 && steps < maxStepsPerFrame) {
                float maxSpeed = sim.maxFaceSpeed();
                float cflDx = sim.dx;

                switch (activeWorkspace) {
                    case UI::kWorkspaceWater2D:
                        maxSpeed = std::max(waterSim.maxFaceSpeed(), waterSim.maxParticleSpeed());
                        cflDx = waterSim.dx;
                        break;
                    case UI::kWorkspaceSmoke3D:
                        maxSpeed = smoke3D.stats().maxSpeed;
                        cflDx = smoke3D.dx;
                        break;
                    case UI::kWorkspaceWater3D:
                        maxSpeed = water3D.stats().maxSpeed;
                        cflDx = water3D.dx;
                        break;
                    case UI::kWorkspaceCoupled:
                        maxSpeed = std::max(coupled.maxFaceSpeed(), coupled.maxParticleSpeed());
                        cflDx = coupled.dx;
                        break;
                    case UI::kWorkspaceSmoke2D:
                    default:
                        maxSpeed = sim.maxFaceSpeed();
                        cflDx = sim.dx;
                        break;
                }

                float dtCFL = ui.cfl * cflDx / (maxSpeed + 1e-6f);

                // CFL is a hard cap
                float dt = std::min(ui.dtMax, dtCFL);
                if (ui.dtMin <= dtCFL) {
                    dt = std::max(dt, ui.dtMin);
                }

                switch (activeWorkspace) {
                    case UI::kWorkspaceWater2D:
                        waterSim.setDt(dt);
                        waterSim.syncSolidsFrom(sim);
                        waterSim.step();
                        break;
                    case UI::kWorkspaceSmoke3D:
                        smoke3D.setDt(dt);
                        smoke3D.step();
                        break;
                    case UI::kWorkspaceWater3D:
                        water3D.setDt(dt);
                        water3D.step();
                        break;
                    case UI::kWorkspaceCoupled:
                        coupled.setDt(dt);
                        coupled.stepCoupled(ui.vortEps);
                        break;
                    case UI::kWorkspaceSmoke2D:
                    default:
                        sim.setDt(dt);
                        sim.step(ui.vortEps);
                        break;
                }

                accumulator -= dt;
                steps++;
            }

            // optional: if we hit the cap, drop the remainder so it doesn't lag forever
            if (steps == maxStepsPerFrame) accumulator = 0.0;
        }

        // upload textures (do this each frame before NewFrame so ImGui uses latest)
        SmokeRenderSettings rs;
        OverlaySettings ov;
        UI::BuildRenderSettings(ui, rs, ov);
        WaterRenderSettings wr;
        UI::BuildWaterRenderSettings(ui, wr);

        if (activeWorkspace == UI::kWorkspaceSmoke2D) {
            renderer.updateFromSim(sim, rs, ov);
        } else if (activeWorkspace == UI::kWorkspaceWater2D) {
            renderer.updateWaterFromSim(waterSim, wr);
        } else if (activeWorkspace == UI::kWorkspaceCoupled) {
            coupledRenderer.updateFromSim(coupled, rs, ov);
            coupledRenderer.updateWaterFromSim(coupled, wr);
        }

        if (activeWorkspace == UI::kWorkspaceWater3D) {
            const int viewMode = std::clamp(ui.water3DViewMode, 0, 2);
            if (viewMode == 1) {
                const int axis = std::clamp(ui.water3DSliceAxis, 0, 2);
                const int field = std::clamp(ui.water3DDebugField, 0, 3);
                const int maxSlice = (axis == 0)
                    ? std::max(0, water3D.nz - 1)
                    : (axis == 1)
                        ? std::max(0, water3D.ny - 1)
                        : std::max(0, water3D.nx - 1);
                ui.water3DSliceIndex = std::clamp(ui.water3DSliceIndex, 0, maxSlice);

                auto slice = water3D.copyDebugSlice(
                    static_cast<MACWater3D::SliceAxis>(axis),
                    ui.water3DSliceIndex,
                    static_cast<MACWater3D::DebugField>(field));
                water3DRenderer.updateWaterFromSlice(slice.values, slice.solid, slice.width, slice.height, wr);
            } else {
                const auto target = computeVolumeRenderTargetSize(
                    win,
                    ui.water3DViewportWidth,
                    ui.water3DViewportHeight,
                    ui.volumeRenderScale);
                if (water3DRenderer.width() != target.width || water3DRenderer.height() != target.height) {
                    water3DRenderer.resize(target.width, target.height);
                }
                water3DRenderer.updateWaterFromVolume(
                    water3D.water,
                    water3D.solid,
                    water3D.nx,
                    water3D.ny,
                    water3D.nz,
                    viewMode,
                    ui.water3DViewYawDeg,
                    ui.water3DViewPitchDeg,
                    ui.water3DViewZoom,
                    ui.water3DVolumeDensity,
                    ui.water3DSurfaceThreshold,
                    wr);
            }
        }

        if (activeWorkspace == UI::kWorkspaceSmoke3D) {
            const int viewMode = std::clamp(ui.smoke3DViewMode, 0, 1);
            if (viewMode == 1) {
                const int axis = std::clamp(ui.smoke3DSliceAxis, 0, 2);
                const int field = std::clamp(ui.smoke3DDebugField, 0, 4);
                const int maxSlice = (axis == 0)
                    ? std::max(0, smoke3D.nz - 1)
                    : (axis == 1)
                        ? std::max(0, smoke3D.ny - 1)
                        : std::max(0, smoke3D.nx - 1);
                ui.smoke3DSliceIndex = std::clamp(ui.smoke3DSliceIndex, 0, maxSlice);

                auto slice = smoke3D.copyDebugSlice(
                    static_cast<MACSmoke3D::SliceAxis>(axis),
                    ui.smoke3DSliceIndex,
                    static_cast<MACSmoke3D::DebugField>(field));
                smoke3DRenderer.updateSmokeFromSlice(slice.values, slice.solid, slice.width, slice.height, field, rs);
            } else {
                const auto target = computeVolumeRenderTargetSize(
                    win,
                    ui.smoke3DViewportWidth,
                    ui.smoke3DViewportHeight,
                    ui.volumeRenderScale);
                if (smoke3DRenderer.width() != target.width || smoke3DRenderer.height() != target.height) {
                    smoke3DRenderer.resize(target.width, target.height);
                }
                smoke3DRenderer.updateSmokeFromVolume(
                    smoke3D.smoke,
                    smoke3D.temp,
                    smoke3D.solid,
                    smoke3D.nx,
                    smoke3D.ny,
                    smoke3D.nz,
                    ui.smoke3DViewYawDeg,
                    ui.smoke3DViewPitchDeg,
                    ui.smoke3DViewZoom,
                    ui.smoke3DVolumeDensity,
                    rs);
            }
        }

        // start imgui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // draw UI
        int currentWindowW = 0;
        int currentWindowH = 0;
        glfwGetWindowSize(win, &currentWindowW, &currentWindowH);
        UI::Actions actions = UI::DrawAll(sim, waterSim, water3D, smoke3D, coupled,
                                          renderer, water3DRenderer, smoke3DRenderer, coupledRenderer,
                                          ui, probe, NX, NY, currentWindowW, currentWindowH);
        if (UI::ConsumeResetLayoutRequest()) {
        // nothing needed here if panels.cpp already deleted ini and rebuilt docks
        // but it's fine to keep for future, you better not delete this >:)
    }

        // handle actions
        if (actions.dropWaterTextRequested) {
            waterSim.waterHeld = false;
        }
        if (actions.resetRequested) {
            activateWorkspace(activeWorkspace);
        }
        if ((actions.applySmoke3DGridRequested || actions.resetSmoke3DRequested) && activeWorkspace == UI::kWorkspaceSmoke3D) {
            activateWorkspace(activeWorkspace);
        }
        if ((actions.applyWater3DGridRequested || actions.resetWater3DRequested) && activeWorkspace == UI::kWorkspaceWater3D) {
            activateWorkspace(activeWorkspace);
        }
        if (actions.applyWindowResolutionRequested) {
            ui.windowWidth = std::clamp(ui.windowWidth, 960, 7680);
            ui.windowHeight = std::clamp(ui.windowHeight, 540, 4320);
            glfwSetWindowSize(win, ui.windowWidth, ui.windowHeight);
        }
        if (actions.applyGrid2DRequested) {
            reinit2DGrid();
        }

        // render GL
        int w, h;
        glfwGetFramebufferSize(win, &w, &h);
        glViewport(0, 0, w, h);
        if (ui.themeMode == kThemeLight) {
            glClearColor(0.948f, 0.944f, 0.937f, 1.0f);
        } else {
            glClearColor(0.055f, 0.066f, 0.078f, 1.0f);
        }
        glClear(GL_COLOR_BUFFER_BIT);

        // render imgui
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        if (UI::ConsumeSaveLayoutRequest())
            ImGui::SaveIniSettingsToDisk(ImGui::GetIO().IniFilename);

       

        glfwSwapBuffers(win);
    }

    // shutdown
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
