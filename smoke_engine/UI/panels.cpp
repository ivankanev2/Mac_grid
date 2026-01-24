#include "panels.h"

#include "imgui.h"
#include "imgui_internal.h"
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <string>
#include <filesystem>

#include "../Sim/mac_smoke_sim.h"
#include "../Renderer/smoke_renderer.h"

namespace UI {

static bool g_pipeMode = false;
static ImGuiID dock_id = 0;

static bool g_requestSaveLayout  = false;
static bool g_requestResetLayout = false;

bool ConsumeSaveLayoutRequest() {
    bool v = g_requestSaveLayout;
    g_requestSaveLayout = false;
    return v;
}

bool ConsumeResetLayoutRequest() {
    bool v = g_requestResetLayout;
    g_requestResetLayout = false;
    return v;
}



// ------------------------------------------------------------
// Dockspace root (everything can dock, and can be dragged out
// into a new native OS window when Viewports are enabled). :)
// ------------------------------------------------------------
static void BeginDockspaceRoot()
{
    ImGuiWindowFlags flags =
        ImGuiWindowFlags_NoDocking |
        ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoBringToFrontOnFocus |
        ImGuiWindowFlags_NoNavFocus |
        ImGuiWindowFlags_MenuBar;

    const ImGuiViewport* vp = ImGui::GetMainViewport();

    ImGui::SetNextWindowPos(vp->WorkPos);
    ImGui::SetNextWindowSize(vp->WorkSize);
    ImGui::SetNextWindowViewport(vp->ID);

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);

    ImGui::Begin("##EditorRoot", nullptr, flags);
    ImGui::PopStyleVar(2);

    if (g_requestResetLayout) {
        g_requestResetLayout = false;

        const char* ini = ImGui::GetIO().IniFilename;
        if (ini) std::remove(ini);          // delete saved layout
        ImGui::DockBuilderRemoveNode(dock_id); // nuke current dock node

        // force rebuild default layout on next lines this frame
    }

    if (ImGui::BeginMenuBar()) {
        if (ImGui::Button("+")) ImGui::OpenPopup("AddPanelPopup");
        ImGui::SameLine();
        ImGui::TextUnformatted("Dock/Undock: drag tabs. Viewports lets tabs become OS windows.");
        ImGui::EndMenuBar();
    }
    if (ImGui::BeginMenuBar()) {
        if (ImGui::Button("Save Layout")) {
            g_requestSaveLayout = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset Layout")) {
            g_requestResetLayout = true;
        }

        ImGui::SameLine();
        ImGui::TextUnformatted("Drag tabs to dock. Save Layout writes imgui.ini.");
        ImGui::EndMenuBar();
    }

    if (ImGui::BeginPopup("AddPanelPopup")) {
        ImGui::TextUnformatted("Next step: add/remove panels dynamically here.");
        ImGui::Separator();
        ImGui::TextDisabled("For now, your panels are:");
        ImGui::BulletText("Controls");
        ImGui::BulletText("Data / Debug");
        ImGui::BulletText("Smoke View");
        ImGui::EndPopup();
    }

    // assign to the global, do NOT redeclare locally
    dock_id = ImGui::GetID("MyDockspace");
    ImGui::DockSpace(dock_id, ImVec2(0,0), ImGuiDockNodeFlags_None);

    // --- Default layout initialization (only if there's no saved imgui.ini) ---
    static bool built_default_layout = false;
    if (!built_default_layout)
    {
        built_default_layout = true;

        // If ImGui is allowed to use an ini file, check if it exists.
        bool ini_exists = false;
        const char* ini = ImGui::GetIO().IniFilename;
        if (ini != nullptr) {
            // simple portable check without <filesystem>
            FILE* f = std::fopen(ini, "r");
            if (f) { ini_exists = true; std::fclose(f); }
        }

        if (!ini_exists)
        {
            // Remove existing node (if any), add a fresh one and split it
            ImGui::DockBuilderRemoveNode(dock_id); // clear any previous layout
            ImGui::DockBuilderAddNode(dock_id, ImGuiDockNodeFlags_DockSpace);
            ImGui::DockBuilderSetNodeSize(dock_id, vp->Size);

            // Split into left (25%) and right (75%)
            ImGuiID dock_left, dock_right;
            ImGui::DockBuilderSplitNode(dock_id, ImGuiDir_Left, 0.25f, &dock_left, &dock_right);

            // Split left into top (50%) and bottom (50%)
            ImGuiID dock_left_top, dock_left_bottom;
            ImGui::DockBuilderSplitNode(dock_left, ImGuiDir_Up, 0.5f, &dock_left_top, &dock_left_bottom);

            // Dock windows (must match ImGui::Begin titles exactly)
            ImGui::DockBuilderDockWindow("Controls",       dock_left_top);
            ImGui::DockBuilderDockWindow("Data / Debug",   dock_left_bottom);
            ImGui::DockBuilderDockWindow("Smoke View",     dock_right);

            ImGui::DockBuilderFinish(dock_id);
        }
    }

    ImGui::End();
}

// ---------- helpers ----------
static void drawArrow(ImDrawList* dl, ImVec2 a, ImVec2 b, ImU32 col) {
    dl->AddLine(a, b, col, 1.0f);

    ImVec2 d(b.x - a.x, b.y - a.y);
    float len = std::sqrt(d.x*d.x + d.y*d.y);
    if (len < 1e-5f) return;
    d.x /= len; d.y /= len;

    float head = 6.0f;
    ImVec2 left(-d.y, d.x);

    ImVec2 p1(b.x - d.x*head + left.x*head*0.5f, b.y - d.y*head + left.y*head*0.5f);
    ImVec2 p2(b.x - d.x*head - left.x*head*0.5f, b.y - d.y*head - left.y*head*0.5f);

    dl->AddLine(b, p1, col, 1.0f);
    dl->AddLine(b, p2, col, 1.0f);
}

static inline float uAtCellCenter(const MAC2D& sim, int i, int j) {
    int i0 = std::max(0, std::min(i, sim.nx - 1));
    int j0 = std::max(0, std::min(j, sim.ny - 1));
    float uL = sim.u[i0 + (sim.nx + 1) * j0];
    float uR = sim.u[(i0 + 1) + (sim.nx + 1) * j0];
    return 0.5f * (uL + uR);
}

static inline float vAtCellCenter(const MAC2D& sim, int i, int j) {
    int i0 = std::max(0, std::min(i, sim.nx - 1));
    int j0 = std::max(0, std::min(j, sim.ny - 1));
    float vB = sim.v[i0 + sim.nx * j0];
    float vT = sim.v[i0 + sim.nx * (j0 + 1)];
    return 0.5f * (vB + vT);
}

static void computeStats(const MAC2D& sim, float& outMaxDiv, float& outAvgAbsDiv, float& outMaxSpeed) {
    outMaxDiv = 0.0f;
    outAvgAbsDiv = 0.0f;
    outMaxSpeed = 0.0f;

    const int N = sim.nx * sim.ny;
    for (int j = 0; j < sim.ny; ++j) {
        for (int i = 0; i < sim.nx; ++i) {
            int idx = sim.idxP(i,j);

            float d = sim.div[idx];
            float ad = std::fabs(d);
            outMaxDiv = std::max(outMaxDiv, ad);
            outAvgAbsDiv += ad;

            float uc = uAtCellCenter(sim, i, j);
            float vc = vAtCellCenter(sim, i, j);
            float sp = std::sqrt(uc*uc + vc*vc);
            outMaxSpeed = std::max(outMaxSpeed, sp);
        }
    }
    outAvgAbsDiv /= (float)N;
}

// ---------- public ----------
void BuildRenderSettings(const Settings& ui,
                         SmokeRenderSettings& outSmoke,
                         OverlaySettings& outOverlay)
{
    outSmoke.useColor     = ui.useColorSmoke;
    outSmoke.alphaGamma   = ui.smokeAlphaGamma;
    outSmoke.alphaScale   = ui.smokeAlphaScale;
    outSmoke.tempStrength = ui.tempColorStrength;
    outSmoke.ageGray      = ui.ageGrayStrength;
    outSmoke.ageDarken    = ui.ageDarkenStrength;
    outSmoke.coreDark     = ui.coreDarkStrength;

    outOverlay.showDiv   = ui.showDivOverlay;
    outOverlay.showVort  = ui.showVortOverlay;
    outOverlay.divScale  = ui.divScale;
    outOverlay.divAlpha  = ui.divAlpha;
    outOverlay.vortScale = ui.vortScale;
    outOverlay.vortAlpha = ui.vortAlpha;
}

static Actions drawControls(MAC2D& sim, Settings& ui) {
    Actions a;

    ImGui::SetNextWindowDockID(dock_id, ImGuiCond_FirstUseEver);
    ImGui::Begin("Controls");

    ImGui::Checkbox("Play", &ui.playing);
    ImGui::Separator();

    bool ot = sim.openTop;
    if (ImGui::Checkbox("Open top (outflow)", &ot)) sim.setOpenTop(ot);

    bool open = sim.isValveOpen();
    if (ImGui::Checkbox("Valve open", &open)) sim.setValveOpen(open);
    ImGui::SameLine();
    ImGui::Text("Valve: %s", sim.isValveOpen() ? "OPEN" : "CLOSED");

    ImGui::SliderFloat("Inlet speed", &sim.inletSpeed, -3.0f, 3.0f);
    ImGui::SliderFloat("Inlet smoke", &sim.inletSmoke, 0.0f, 1.0f);
    ImGui::SliderFloat("Inlet temp",  &sim.inletTemp,  0.0f, 1.0f);

    ImGui::Separator();
    ImGui::Checkbox("Paint solid", &ui.paintSolid);
    ImGui::SameLine();
    ImGui::Checkbox("Erase", &ui.eraseSolid);

    ImGui::Checkbox("Circle", &ui.circleMode);
    ImGui::SameLine();
    ImGui::TextUnformatted("Rect");

    if (ImGui::Button("Reset Smoke")) {
        a.resetRequested = true;
    }

    ImGui::SliderFloat("Brush radius", &ui.brushRadius, 0.01f, 0.20f);
    ImGui::SliderFloat("Rect half-size", &ui.rectHalfSize, 0.01f, 0.30f);
    ImGui::SliderFloat("View scale", &ui.viewScale, 1.0f, 12.0f);

    ImGui::Separator();
    ImGui::Checkbox("Pipe mode", &g_pipeMode);
    ImGui::SliderFloat("Pipe radius", &sim.pipe.radius, 0.01f, 0.25f);
    ImGui::SliderFloat("Wall thickness", &sim.pipe.wall, 0.005f, 0.10f);

    if (ImGui::Button("Clear pipe")) {
        sim.clearPipe();
        sim.rebuildSolidsFromPipe(false);
        sim.enforceBoundaries();
    }
    ImGui::SameLine();
    if (ImGui::Button("Rebuild solids")) {
        sim.rebuildSolidsFromPipe(false);
        sim.enforceBoundaries();
    }

    ImGui::End();
    return a;
}

static void drawDebugTabs(MAC2D& sim, Settings& ui, Probe& probe) {
    ImGui::SetNextWindowDockID(dock_id, ImGuiCond_FirstUseEver);
    ImGui::Begin("Data / Debug");

    if (ImGui::BeginTabBar("DebugTabs")) {

        if (ImGui::BeginTabItem("Formulas")) {
            ImGui::TextWrapped("MAC grid (staggered): u on vertical faces, v on horizontal faces, smoke/pressure at cell centers.");
            ImGui::Separator();
            ImGui::Text("Divergence (cell center):");
            ImGui::BulletText("div(i,j) = ( u(i+1,j) - u(i,j) + v(i,j+1) - v(i,j) ) / dx");
            ImGui::Text("Pressure solve (Poisson):");
            ImGui::BulletText("Laplace(p) = div / dt (solved iteratively)");
            ImGui::Text("Projection (incompressible):");
            ImGui::BulletText("u -= dt * dp/dx,  v -= dt * dp/dy");

            ImGui::Checkbox("Use MacCormack advector", &sim.useMacCormack);

            if (ImGui::Button("Compare advectors (MacCormack vs SL)")) {
                ui.lastAdvectL2 = sim.compareAdvectors(0.995f);
                std::printf("Compare advectors L2 = %g\n", ui.lastAdvectL2);
            }
            ImGui::Text("Last advector L2: %.6e", ui.lastAdvectL2);

            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Stats")) {
            float maxDiv=0, avgAbsDiv=0, maxSpeed=0;
            computeStats(sim, maxDiv, avgAbsDiv, maxSpeed);

            ImGui::Text("Grid: %d x %d", sim.nx, sim.ny);
            ImGui::Text("dx: %.6f   dt: %.6f", sim.dx, sim.dt);
            ImGui::Separator();
            ImGui::Text("max |div|: %.6e", maxDiv);
            ImGui::Text("avg |div|: %.6e", avgAbsDiv);
            ImGui::Text("max speed: %.6f", maxSpeed);

            ImGui::Separator();
            ImGui::SliderFloat("Vorticity eps", &ui.vortEps, 0.0f, 8.0f);
            ImGui::Checkbox("Vorticity heatmap", &ui.showVortOverlay);
            ImGui::SliderFloat("Vort scale", &ui.vortScale, 0.1f, 50.0f);
            ImGui::SliderFloat("Vort alpha", &ui.vortAlpha, 0.0f, 1.0f);

            if (ImGui::Button("Impulse (test swirl)")) {
                sim.addVelocityImpulse(0.5f, 0.5f, 0.12f, 3.0f);
            }

            ImGui::Separator();
            ImGui::Checkbox("Divergence heatmap", &ui.showDivOverlay);
            ImGui::SliderFloat("Div scale", &ui.divScale, 0.1f, 50.0f);
            ImGui::SliderFloat("Div alpha", &ui.divAlpha, 0.0f, 1.0f);

            ImGui::Separator();
            ImGui::Checkbox("Velocity arrows", &ui.showVelOverlay);
            ImGui::SliderInt("Vel stride", &ui.velStride, 2, 16);
            ImGui::SliderFloat("Vel scale", &ui.velScale, 0.05f, 2.0f);

            ImGui::Separator();
            ImGui::Text("Adaptive dt (CFL)");
            ImGui::SliderFloat("CFL", &ui.cfl, 0.1f, 1.5f);
            ImGui::SliderFloat("dtMax", &ui.dtMax, 0.001f, 0.05f);
            ImGui::SliderFloat("dtMin", &ui.dtMin, 0.0001f, 0.01f);
            ImGui::Text("current dt: %.6f", sim.dt);
            ImGui::Text("max face speed: %.6f", sim.maxFaceSpeed());

            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Probe")) {
            if (!probe.has) {
                ImGui::TextUnformatted("Hover the Smoke View to inspect a cell.");
            } else {
                ImGui::Text("Cell (i,j): (%d, %d)", probe.i, probe.j);
                ImGui::Text("smoke: %.4f", probe.smoke);
                ImGui::Text("div:   %.6e", probe.div);
                ImGui::Text("u_c:   %.6f", probe.u);
                ImGui::Text("v_c:   %.6f", probe.v);
                ImGui::Text("|v|:   %.6f", probe.speed);
            }
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Smoke")) {
            ImGui::TextUnformatted("Smoke Rendering");
            ImGui::Checkbox("Color by temperature / age", &ui.useColorSmoke);
            ImGui::SliderFloat("Alpha gamma", &ui.smokeAlphaGamma, 0.10f, 2.00f);
            ImGui::SliderFloat("Alpha scale", &ui.smokeAlphaScale, 0.0f, 2.0f);

            if (ui.useColorSmoke) {
                ImGui::SliderFloat("Temp color strength", &ui.tempColorStrength, 0.0f, 1.0f);
                ImGui::SliderFloat("Age gray strength", &ui.ageGrayStrength, 0.0f, 1.0f);
                ImGui::SliderFloat("Age darken strength", &ui.ageDarkenStrength, 0.0f, 1.0f);
                ImGui::SliderFloat("Core dark strength", &ui.coreDarkStrength, 0.0f, 1.0f);
            } else {
                ImGui::TextDisabled("Color controls disabled (grayscale mode)");
            }

            ImGui::Separator();
            ImGui::TextUnformatted("Smoke Simulation");
            ImGui::SliderFloat("Smoke dissipation", &ui.smokeDissipation, 0.980f, 1.000f);
            ImGui::SliderFloat("Temp dissipation", &ui.tempDissipation, 0.900f, 1.000f);

            ImGui::EndTabItem();
        }

        ImGui::EndTabBar();
    }

    ImGui::End();
}

static void drawSmokeViewAndInteract(MAC2D& sim,
                                     SmokeRenderer& renderer,
                                     Settings& ui,
                                     Probe& probe,
                                     int NX, int NY)
{
    ImGui::SetNextWindowDockID(dock_id, ImGuiCond_FirstUseEver);
    ImGui::Begin("Smoke View");

    float scale = ui.viewScale;
    ImGui::Image((ImTextureID)(intptr_t)renderer.smokeTex(), ImVec2(NX * scale, NY * scale));

    ImVec2 p0 = ImGui::GetItemRectMin();
    ImVec2 p1 = ImGui::GetItemRectMax();
    ImDrawList* dl = ImGui::GetWindowDrawList();

    // ---- pipe polyline overlay ----
    if (sim.pipe.x.size() >= 2) {
        for (size_t k = 0; k + 1 < sim.pipe.x.size(); ++k) {
            ImVec2 a(p0.x + sim.pipe.x[k]   * (p1.x - p0.x),
                     p0.y + (1.0f - sim.pipe.y[k])   * (p1.y - p0.y));
            ImVec2 b(p0.x + sim.pipe.x[k+1] * (p1.x - p0.x),
                     p0.y + (1.0f - sim.pipe.y[k+1]) * (p1.y - p0.y));
            dl->AddLine(a, b, IM_COL32(0, 150, 255, 220), 2.0f);
        }
    }

    if (ui.showDivOverlay)
        dl->AddImage((ImTextureID)(intptr_t)renderer.divTex(), p0, p1);
    if (ui.showVortOverlay)
        dl->AddImage((ImTextureID)(intptr_t)renderer.vortTex(), p0, p1);

    // Velocity arrows overlay
    if (ui.showVelOverlay) {
        float W = (p1.x - p0.x);
        float H = (p1.y - p0.y);
        float cellW = W / sim.nx;
        float cellH = H / sim.ny;

        for (int j = 0; j < sim.ny; j += ui.velStride) {
            int simJ = (sim.ny - 1 - j);
            for (int i = 0; i < sim.nx; i += ui.velStride) {
                int idx = sim.idxP(i, simJ);
                if (sim.solid[idx]) continue;

                float uc = uAtCellCenter(sim, i, simJ);
                float vc = vAtCellCenter(sim, i, simJ);

                float cx = p0.x + (i + 0.5f) * cellW;
                float cy = p0.y + (j + 0.5f) * cellH;

                float dx =  uc * ui.velScale * cellW;
                float dy = -vc * ui.velScale * cellH;

                drawArrow(dl, ImVec2(cx, cy), ImVec2(cx + dx, cy + dy),
                          IM_COL32(0, 255, 0, 180));
            }
        }
    }

    bool hovered = ImGui::IsItemHovered();

    // PROBE (hover)
    probe.has = false;
    if (hovered) {
        ImVec2 m = ImGui::GetMousePos();
        float uu = (m.x - p0.x) / (p1.x - p0.x);
        float vv = (m.y - p0.y) / (p1.y - p0.y);

        float sx = uu;
        float sy = vv;

        if (sx >= 0 && sx <= 1 && sy >= 0 && sy <= 1) {
            probe.i = std::max(0, std::min((int)(sx * sim.nx), sim.nx - 1));
            probe.j = std::max(0, std::min((int)(sy * sim.ny), sim.ny - 1));
            int idx = sim.idxP(probe.i, probe.j);

            probe.smoke = sim.smoke[idx];
            probe.div   = sim.div[idx];
            probe.u     = uAtCellCenter(sim, probe.i, probe.j);
            probe.v     = vAtCellCenter(sim, probe.i, probe.j);
            probe.speed = std::sqrt(probe.u*probe.u + probe.v*probe.v);
            probe.has = true;
        }
    }

    // PIPE EDIT: click to add points
    if (hovered && g_pipeMode && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        ImVec2 m = ImGui::GetMousePos();
        float u = (m.x - p0.x) / (p1.x - p0.x);
        float v = (m.y - p0.y) / (p1.y - p0.y);

        float sx = u;
        float sy = 1.0f - v; // sim coordinates

        if (sx >= 0 && sx <= 1 && sy >= 0 && sy <= 1) {
            sim.pipe.x.push_back(sx);
            sim.pipe.y.push_back(sy);
            sim.rebuildSolidsFromPipe(false);
            sim.enforceBoundaries();
        }
    }

    // PAINT / ERASE solids (disabled when pipe mode is on)
    if (!g_pipeMode) {
        if (hovered && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            ImVec2 m = ImGui::GetMousePos();
            float u = (m.x - p0.x) / (p1.x - p0.x);
            float v = (m.y - p0.y) / (p1.y - p0.y);

            float sx = u;
            float sy = 1.0f - v; // sim coordinates

            if (sx >= 0 && sx <= 1 && sy >= 0 && sy <= 1) {
                if (!ui.eraseSolid) {
                    if (ui.circleMode) {
                        sim.addSolidCircle(sx, sy, ui.brushRadius);
                    } else {
                        float hs = ui.rectHalfSize;
                        for (float yy = sy-hs; yy <= sy+hs; yy += sim.dx) {
                            for (float xx = sx-hs; xx <= sx+hs; xx += sim.dx) {
                                sim.addSolidCircle(xx, yy, sim.dx*0.75f);
                            }
                        }
                    }
                } else {
                    int ci = (int)(sx * sim.nx);
                    int cj = (int)(sy * sim.ny);
                    int rad = std::max(1, (int)(ui.brushRadius / sim.dx));

                    for (int y = cj-rad; y <= cj+rad; ++y) {
                        for (int x = ci-rad; x <= ci+rad; ++x) {
                            if (x <= 0 || x >= sim.nx-1 || y <= 0 || y >= sim.ny-1) continue;
                            float dx = (x + 0.5f) / sim.nx - sx;
                            float dy = (y + 0.5f) / sim.ny - sy;
                            if (dx*dx + dy*dy <= ui.brushRadius*ui.brushRadius) {
                                sim.solid[sim.idxP(x,y)] = 0;
                            }
                        }
                    }
                }
            }
        }
    }

    ImGui::End();
}

Actions DrawAll(MAC2D& sim,
                SmokeRenderer& renderer,
                Settings& ui,
                Probe& probe,
                int NX, int NY)
{
    // Dockspace root must be drawn BEFORE your windows
    BeginDockspaceRoot();

    Actions a = drawControls(sim, ui);
    drawDebugTabs(sim, ui, probe);
    drawSmokeViewAndInteract(sim, renderer, ui, probe, NX, NY);
    return a;
}

} // namespace UI