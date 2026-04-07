#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <algorithm>

#include "UI/panels.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#ifdef __APPLE__
  #include <OpenGL/gl3.h>
#else
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



// window size and sim resolution
static const int NX = 96;
static const int NY = 96;
static const int NZ = 64;

// time variables
double lastTime = 0.0;
double accumulator = 0.0;
float simSpeed = 1.0f;          // 1.0 = real-time, 2.0 = 2x faster, etc.
int maxStepsPerFrame = 8;       // cap to avoid death-spiral

int main()
{
    if (!glfwInit()) return 1;
    lastTime = glfwGetTime();

    // OpenGL / GLFW hints
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);

    GLFWwindow* win = glfwCreateWindow(1100, 800, "Smoke Engine", nullptr, nullptr);
    if (!win) return 1;

    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);

    // ImGui init
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = "imgui.ini";  // i fucking hate this line of code but need it
    

    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;      // keep docking
    io.ConfigFlags &= ~ImGuiConfigFlags_ViewportsEnable;   // disable OS-windows

    ImGui::StyleColorsDark();

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

    MAC2D sim(NX, NY, dx, dt_initial);
    MACWater waterSim(NX, NY, dx, dt_initial);
    MACWater3D water3D(NX, NY, NZ, dx, dt_initial);
    MACSmoke3D smoke3D(NX, NY, NZ, dx, dt_initial);
    MACCoupledSim coupled(NX, NY, dx, dt_initial);

    // Make both sims use the same solver instance.
    sim.setSharedPressureSolver(&sharedPressureSolver);
    waterSim.setSharedPressureSolver(&sharedPressureSolver);
    coupled.setSharedPressureSolver(&sharedPressureSolver);

    // UI state
    UI::Settings ui;
    UI::Probe probe;

    while (!glfwWindowShouldClose(win))
    {
        glfwPollEvents();

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
                const float waterSpeed2D = std::max(waterSim.maxFaceSpeed(), waterSim.maxParticleSpeed());
                float maxSpeed = std::max(sim.maxFaceSpeed(), waterSpeed2D);
                float cflDx = sim.dx;

                if (ui.useWater3D) {
                    maxSpeed = std::max(maxSpeed, water3D.stats().maxSpeed);
                    cflDx = std::min(cflDx, water3D.dx);
                }
                if (ui.useSmoke3D) {
                    maxSpeed = std::max(maxSpeed, smoke3D.stats().maxSpeed);
                    cflDx = std::min(cflDx, smoke3D.dx);
                }

                float dtCFL = ui.cfl * cflDx / (maxSpeed + 1e-6f);

                // CFL is a hard cap
                float dt = std::min(ui.dtMax, dtCFL);
                if (ui.dtMin <= dtCFL) {
                    dt = std::max(dt, ui.dtMin);
                }

                sim.setDt(dt);
                sim.step(ui.vortEps);

                waterSim.setDt(dt);
                waterSim.step();

                coupled.setDt(dt);
                coupled.stepCoupled(ui.vortEps);

                if (ui.useWater3D) {
                    water3D.setDt(dt);
                    water3D.step();
                }
                if (ui.useSmoke3D) {
                    smoke3D.setDt(dt);
                    smoke3D.step();
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

        // update textures for original smoke/water sims (used by Smoke View + Water View)
        renderer.updateFromSim(sim, rs, ov);
        renderer.updateWaterFromSim(waterSim, wr);

        // update textures for the coupled sim (used by Combined View)
        coupledRenderer.updateFromSim(coupled, rs, ov);
        coupledRenderer.updateWaterFromSim(coupled, wr);

        if (ui.showWater3DView || ui.useWater3D) {
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
                const int maxDim = std::max({water3D.nx, water3D.ny, water3D.nz, 1});
                const int targetRes = std::max(192, std::min(640, maxDim * 3));
                if (water3DRenderer.width() != targetRes || water3DRenderer.height() != targetRes) {
                    water3DRenderer.resize(targetRes, targetRes);
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

        if (ui.showSmoke3DView || ui.useSmoke3D) {
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
                const int maxDim = std::max({smoke3D.nx, smoke3D.ny, smoke3D.nz, 1});
                const int targetRes = std::max(192, std::min(640, maxDim * 3));
                if (smoke3DRenderer.width() != targetRes || smoke3DRenderer.height() != targetRes) {
                    smoke3DRenderer.resize(targetRes, targetRes);
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
        UI::Actions actions = UI::DrawAll(sim, waterSim, water3D, smoke3D, coupled, renderer, water3DRenderer, smoke3DRenderer, coupledRenderer, ui, probe, NX, NY);
        if (UI::ConsumeResetLayoutRequest()) {
        // nothing needed here if panels.cpp already deleted ini and rebuilt docks
        // but it's fine to keep for future, you better not delete this >:)
    }

        // handle actions (reset)
        if (actions.resetRequested) {
            sim.reset();
            waterSim.reset();
            water3D.reset();
            smoke3D.reset();
            coupled.reset();
        }
        if (actions.applySmoke3DGridRequested) {
            const int nx3 = std::max(1, ui.smoke3DNX);
            const int ny3 = std::max(1, ui.smoke3DNY);
            const int nz3 = std::max(1, ui.smoke3DNZ);
            const int maxDim = std::max({nx3, ny3, nz3});
            const float dx3 = 1.0f / (float)maxDim;
            smoke3D.reset(nx3, ny3, nz3, dx3, smoke3D.dt);
        }
        if (actions.resetSmoke3DRequested) {
            smoke3D.reset();
        }
        if (actions.applyWater3DGridRequested) {
            const int nx3 = std::max(1, ui.water3DNX);
            const int ny3 = std::max(1, ui.water3DNY);
            const int nz3 = std::max(1, ui.water3DNZ);
            const int maxDim = std::max({nx3, ny3, nz3});
            const float dx3 = 1.0f / (float)maxDim;
            water3D.reset(nx3, ny3, nz3, dx3, water3D.dt);
        }
        if (actions.resetWater3DRequested) {
            water3D.reset();
        }

        // render GL
        int w, h;
        glfwGetFramebufferSize(win, &w, &h);
        glViewport(0, 0, w, h);
        glClearColor(0, 0, 0, 1);
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
