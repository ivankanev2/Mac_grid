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
#include "Sim/pressure_solver.h"
#include "Renderer/smoke_renderer.h"

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

#include "Sim/mac_coupled_sim.h"



// window size and sim resolution
static const int NX = 96;
static const int NY = 96;

// time variables
double lastTime = glfwGetTime();
double accumulator = 0.0;
float simSpeed = 1.0f;          // 1.0 = real-time, 2.0 = 2x faster, etc.
int maxStepsPerFrame = 8;       // cap to avoid death-spiral

int main()
{
    if (!glfwInit()) return 1;

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

        coupled.waterDissipation = ui.waterDissipation;
        coupled.waterGravity     = ui.waterGravity;
        coupled.velDamping       = ui.waterVelDamping;
        coupled.setOpenTop(ui.waterOpenTop);  // IMPORTANT: use setter so BC updates

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
                float maxSpeed = std::max({ sim.maxFaceSpeed(),
                                            waterSim.maxFaceSpeed(),
                                            waterSim.maxParticleSpeed() });
                float dtCFL = ui.cfl * sim.dx / (maxSpeed + 1e-6f);

                // CFL is a hard cap
                float dt = std::min(ui.dtMax, dtCFL);
            
                if (ui.dtMin > dtCFL) {
                    // dt stays at dtCFL
                } else {
                    dt = std::max(dt, ui.dtMin);
                }

                sim.setDt(dt);
                waterSim.setDt(dt);
                sim.step(ui.vortEps);
                waterSim.step();

                coupled.setDt(dt);
                coupled.stepCoupled(ui.vortEps);

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

        // start imgui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // draw UI
        UI::Actions actions = UI::DrawAll(sim, waterSim, coupled, renderer, coupledRenderer, ui, probe, NX, NY);
        if (UI::ConsumeResetLayoutRequest()) {
        // nothing needed here if panels.cpp already deleted ini and rebuilt docks
        // but it's fine to keep for future, you better not delete this >:)
    }

        // handle actions (reset)
        if (actions.resetRequested) {
            sim.reset();
            waterSim.reset();
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
