#include <cstdio>
#include <cmath>
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
#include "Renderer/smoke_renderer.h"

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

// window size and sim resolution
static const int NX = 96;
static const int NY = 96;

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

    // Simulator
    float dx = 1.0f / NX;
    float dt_initial = 0.02f;
    MAC2D sim(NX, NY, dx, dt_initial);

    // UI state
    UI::Settings ui;
    UI::Probe probe;

    while (!glfwWindowShouldClose(win))
    {
        glfwPollEvents();

        // step sim (if playing)
        sim.smokeDissipation = ui.smokeDissipation;
        sim.tempDissipation  = ui.tempDissipation;

        if (ui.playing) {
            for (int sub = 0; sub < 2; ++sub) {
                float maxSpeed = sim.maxFaceSpeed();
                float dt = ui.cfl * sim.dx / (maxSpeed + 1e-6f);

                if (dt > ui.dtMax) dt = ui.dtMax;
                if (dt < ui.dtMin) dt = ui.dtMin;

                sim.setDt(dt);
                sim.step(ui.vortEps);
            }
        }

        // upload textures
        SmokeRenderSettings rs;
        OverlaySettings ov;
        UI::BuildRenderSettings(ui, rs, ov);
        renderer.updateFromSim(sim, rs, ov);

        // start imgui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // draw UI
        UI::Actions actions = UI::DrawAll(sim, renderer, ui, probe, NX, NY);
        if (UI::ConsumeResetLayoutRequest()) {
        // nothing needed here if panels.cpp already deleted ini and rebuilt docks
        // but it's fine to keep for future, you better not delete this >:)
    }

        // handle actions (reset)
        if (actions.resetRequested) {
            sim.reset();
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