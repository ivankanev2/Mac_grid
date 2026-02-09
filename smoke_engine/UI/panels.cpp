#include "panels.h"

#include "imgui.h"
#include "imgui_internal.h"
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <string>
#include <filesystem>

#include "../Sim/mac_smoke_sim.h"
#include "../Sim/mac_water_sim.h"
#include "../Renderer/smoke_renderer.h"
#include "../Sim/mac_coupled_sim.h"

namespace UI {

static bool g_pipeMode = false;
static ImGuiID dock_id = 0;

static bool g_requestSaveLayout  = false;
static bool g_requestResetLayout = false;

struct Ring {
    std::vector<float> v;
    int head = 0;
    bool filled = false;

    Ring(int cap=240) : v(cap, 0.0f) {}

    void push(float x) {
        v[head] = x;
        head = (head + 1) % (int)v.size();
        if (head == 0) filled = true;
    }

    int size() const { return filled ? (int)v.size() : head; }

    // Copy in chronological order into out (for plotting)
    void toChrono(std::vector<float>& out) const {
        int n = size();
        out.resize(n);
        if (!filled) {
            for (int i=0;i<n;i++) out[i] = v[i];
            return;
        }
        // oldest is head, newest is head-1
        for (int i=0;i<n;i++) out[i] = v[(head + i) % (int)v.size()];
    }
};

enum StatID {
    STAT_DT,
    STAT_MAXDIV_BEFORE,
    STAT_MAXDIV_AFTER,
    STAT_MAXSPEED_BEFORE,
    STAT_MAXSPEED_AFTER,
    STAT_PRES_ITERS,
    STAT_PRES_MS,


    STAT_OP_CHECK_PASS,       // 0/1 if op check passed
    STAT_OP_DIFF_MAX,         // max |A_mg - A_pcg|
    STAT_MG_RESIDUAL_INCR,    // 0/1 if MG residual increased during v-cycles
    STAT_RHS_MAX_PREDDIV,     // bInf * dt
    STAT_PREDDIV_INITIAL,     // initial rInf * dt (before solve)
    STAT_PREDDIV_FINAL,       // final rInf * dt (after solve)
    STAT_PRESSURE_STOP_REASON,// enum value (int)
    STAT_PRESSURE_SOLVER,     // enum value: SOLVER_MG / SOLVER_PCG

    STAT_COUNT

};

static const char* kStatNames[STAT_COUNT] = {
    "dt",
    "max|div| (before)",
    "max|div| (after)",
    "max face speed (before)",
    "max face speed (after)",
    "pressure iters",
    "pressure ms",

    "op check pass",
    "op diff max",
    "mg residual incr",
    "rhs max predDiv",
    "predDiv initial",
    "predDiv final",
    "pressure stop reason",
    "pressure solver"
};

static Ring g_hist[STAT_COUNT] = {
    Ring(360), Ring(360), Ring(360), Ring(360), Ring(360), Ring(360), Ring(360),
    Ring(360), Ring(360), Ring(360), Ring(360), Ring(360), Ring(360), Ring(360), Ring(360)
};

static int  g_selectedStat = STAT_PRES_MS;
static bool g_recordStats  = true;
static std::vector<float> g_plotScratch;


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
            ImGui::DockBuilderDockWindow("Combined View",  dock_right);

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

struct WaterStats {
    float sumWater = 0.0f;
    float maxWater = 0.0f;
    size_t particles = 0;
    int particlesInSolid = 0;
    int liquidCount = 0;
    int interiorLiquidCount = 0;
    float maxAbsDiv = 0.0f;
    float avgAbsDiv = 0.0f;
    float maxAbsDivInterior = 0.0f;
    float avgAbsDivInterior = 0.0f;
    int nearLeft = 0;
    int nearRight = 0;
    int nearBottom = 0;
    int nearTop = 0;
    float particleMassProxy = 0.0f;
};

static void computeWaterStats(const MACWater& water, WaterStats& out) {
    out = WaterStats{};
    out.particles = water.particles.size();
    out.particleMassProxy = (float)water.particles.size();
    

    // Field totals.
    double sum = 0.0;
    float maxV = 0.0f;
    const auto& wf = water.waterField();
    for (float v : wf) {
        const float vv = std::max(0.0f, v);
        sum += (double)vv;
        maxV = std::max(maxV, vv);
    }
    out.sumWater = (float)sum;
    out.maxWater = maxV;

    if (!water.particles.empty()) {
        // Border proximity checks mirror enforceParticleBounds().
        const int bt = std::max(1, water.borderThickness);
        const float minX = (bt + 0.5f) * water.dx;
        const float maxX = (water.nx - bt - 0.5f) * water.dx;
        const float minY = (bt + 0.5f) * water.dx;
        const float maxY = water.openTop
            ? (water.ny - 0.5f) * water.dx
            : (water.ny - bt - 0.5f) * water.dx;
        const float eps = 0.25f * water.dx;

        for (const auto& p : water.particles) {
            if (p.x <= minX + eps) out.nearLeft++;
            if (p.x >= maxX - eps) out.nearRight++;
            if (p.y <= minY + eps) out.nearBottom++;
            if (p.y >= maxY - eps) out.nearTop++;

            int i, j;
            water.worldToCell(p.x, p.y, i, j);
            const int id = water.idxP(i, j);
            if (id >= 0 && id < (int)water.solid.size() && water.solid[(size_t)id]) {
                out.particlesInSolid++;
            }
        }
    }

    // Divergence stats (water-only).
    auto isSolidCell = [&](int i, int j) {
        if (i < 0 || i >= water.nx || j < 0) return true;

        // For open-top water, outside above the top is AIR, not SOLID.
        if (j >= water.ny) return water.openTop ? false : true;

        return water.solid[(size_t)water.idxP(i, j)] != 0;
    };
    auto isLiquidCell = [&](int i, int j) {
        if (i < 0 || i >= water.nx || j < 0 || j >= water.ny) return false;
        const int id = water.idxP(i, j);
        return !water.solid[(size_t)id] && water.liquid[(size_t)id];
    };

    double sumAbsDiv = 0.0;
    double sumAbsDivInterior = 0.0;

    for (int j = 0; j < water.ny; ++j) {
        for (int i = 0; i < water.nx; ++i) {
            const int id = water.idxP(i, j);
            if (water.solid[(size_t)id] || !water.liquid[(size_t)id]) continue;

            float uL = water.u[(size_t)water.idxU(i, j)];
            float uR = water.u[(size_t)water.idxU(i + 1, j)];
            float vB = water.v[(size_t)water.idxV(i, j)];
            float vT = water.v[(size_t)water.idxV(i, j + 1)];

            if (isSolidCell(i - 1, j)) uL = 0.0f;
            if (isSolidCell(i + 1, j)) uR = 0.0f;
            if (isSolidCell(i, j - 1)) vB = 0.0f;
            if (isSolidCell(i, j + 1)) vT = 0.0f;

            const float div = (uR - uL + vT - vB) / water.dx;
            const float ad = std::fabs(div);

            out.liquidCount += 1;
            sumAbsDiv += (double)ad;
            out.maxAbsDiv = std::max(out.maxAbsDiv, ad);

            const bool interior =
                isLiquidCell(i - 1, j) &&
                isLiquidCell(i + 1, j) &&
                isLiquidCell(i, j - 1) &&
                isLiquidCell(i, j + 1);

            if (interior) {
                out.interiorLiquidCount += 1;
                sumAbsDivInterior += (double)ad;
                out.maxAbsDivInterior = std::max(out.maxAbsDivInterior, ad);
            }
        }
    }

    if (out.liquidCount > 0) {
        out.avgAbsDiv = (float)(sumAbsDiv / (double)out.liquidCount);
    }
    if (out.interiorLiquidCount > 0) {
        out.avgAbsDivInterior = (float)(sumAbsDivInterior / (double)out.interiorLiquidCount);
    }
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

void BuildWaterRenderSettings(const Settings& ui,
                              WaterRenderSettings& outWater)
{
    outWater.alpha = ui.waterAlpha;
}

static Actions drawControls(MAC2D& sim, MACWater& water, Settings& ui) {
    Actions a;

    ImGui::SetNextWindowDockID(dock_id, ImGuiCond_FirstUseEver);
    ImGui::Begin("Controls");

    ImGui::Checkbox("Play", &ui.playing);
    ImGui::Separator();

    bool ot = sim.getOpenTop();
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
    ImGui::TextUnformatted("Water");
    if (ImGui::Checkbox("Paint water", &ui.paintWater)) {
        if (ui.paintWater) ui.eraseSolid = false;
    }
    ImGui::SliderFloat("Water amount", &ui.waterAmount, 0.01f, 1.00f);
    ImGui::SliderFloat("Water gravity", &ui.waterGravity, -20.0f, 20.0f);
    ImGui::SliderFloat("Water dissipation", &ui.waterDissipation, 0.980f, 1.000f);
    ImGui::SliderFloat("Water damping", &ui.waterVelDamping, 0.0f, 5.0f);
    ImGui::Checkbox("Water open top", &ui.waterOpenTop);
    ImGui::Checkbox("Show water view", &ui.showWaterView);
    ImGui::Checkbox("Show water particles", &ui.showWaterParticles);
    ImGui::SliderFloat("Water alpha", &ui.waterAlpha, 0.0f, 1.0f);

    ImGui::Checkbox("Combined View", &ui.showCombinedView);
    ImGui::SliderFloat("Combined Water Alpha", &ui.combinedWaterAlpha, 0.0f, 1.0f);
    ImGui::Checkbox("Combined Particles", &ui.combinedShowParticles);

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

static void drawDebugTabs(MAC2D& sim, MACWater& water, Settings& ui, Probe& probe) {
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
            WaterStats wst;
            computeWaterStats(water, wst);

            ImGui::Text("Grid: %d x %d", sim.nx, sim.ny);
            ImGui::Text("dx: %.6f   dt: %.6f", sim.dx, sim.dt);
            ImGui::Separator();
            ImGui::Text("max |div|: %.6e", maxDiv);
            ImGui::Text("avg |div|: %.6e", avgAbsDiv);
            ImGui::Text("max speed: %.6f", maxSpeed);

            ImGui::Separator();
            ImGui::TextUnformatted("Water");
            ImGui::Text("openTop: %s   borderThickness: %d", water.openTop ? "true" : "false", water.borderThickness);
            ImGui::Text("particles: %zu   inSolid: %d", wst.particles, wst.particlesInSolid);
            ImGui::Text("sum(water): %.6f   targetMass: %.6f", wst.sumWater, water.targetMass);
            ImGui::Text("max(water): %.6f", wst.maxWater);
            ImGui::Text("liquid cells: %d   interior: %d", wst.liquidCount, wst.interiorLiquidCount);
            ImGui::Text("particle mass proxy: %.0f", wst.particleMassProxy);
            ImGui::Text("div (all) max/avg: %.3e / %.3e", wst.maxAbsDiv, wst.avgAbsDiv);
            ImGui::Text("div (int) max/avg: %.3e / %.3e", wst.maxAbsDivInterior, wst.avgAbsDivInterior);
            ImGui::Text("near borders L/R/B/T: %d / %d / %d / %d", wst.nearLeft, wst.nearRight, wst.nearBottom, wst.nearTop);

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

        if (ImGui::BeginTabItem("Profiler")) {
            const auto& st = sim.getStats();

            if (ImGui::Button(g_recordStats ? "Pause Recording" : "Resume Recording"))
                g_recordStats = !g_recordStats;

            ImGui::SameLine();
            if (ImGui::Button("Clear History")) {
                for (int i=0;i<STAT_COUNT;i++) {
                    g_hist[i] = Ring(360);
                }
            }

            ImGui::Separator();

            // Left: selectable list of stats
            ImGui::BeginChild("stat_list", ImVec2(220, 0), true);
            for (int i = 0; i < STAT_COUNT; ++i) {
                if (ImGui::Selectable(kStatNames[i], g_selectedStat == i))
                    g_selectedStat = i;
            }
            ImGui::EndChild();

            ImGui::SameLine();

            // Right: current value + history plot
            ImGui::BeginChild("stat_view", ImVec2(0, 0), true);

            auto currentValue = [&](int id)->float {
                switch (id) {
                    case STAT_DT:             return st.dt;
                    case STAT_MAXDIV_BEFORE:  return st.maxDivBefore;
                    case STAT_MAXDIV_AFTER:   return st.maxDivAfter;
                    case STAT_MAXSPEED_BEFORE:return st.maxFaceSpeedBefore;
                    case STAT_MAXSPEED_AFTER: return st.maxFaceSpeedAfter;
                    case STAT_PRES_ITERS:     return (float)st.pressureIters;
                    case STAT_PRES_MS:        return st.pressureMs;

                    case STAT_OP_CHECK_PASS:      return (float)st.opCheckPass;
                    case STAT_OP_DIFF_MAX:        return st.opDiffMax;
                    case STAT_MG_RESIDUAL_INCR:   return (float)st.mgResidualIncrease;
                    case STAT_RHS_MAX_PREDDIV:    return st.rhsMaxPredDiv;
                    case STAT_PREDDIV_INITIAL:    return st.predDivInitial;
                    case STAT_PREDDIV_FINAL:      return st.predDivFinal;
                    case STAT_PRESSURE_STOP_REASON: return (float)st.pressureStopReason;
                    case STAT_PRESSURE_SOLVER:    return (float)st.pressureSolver;

                    default: return 0.0f;
                }
            };

            float cur = currentValue(g_selectedStat);
            ImGui::Text("Selected: %s", kStatNames[g_selectedStat]);

            // human readable for two enum-like stats
            if (g_selectedStat == STAT_PRESSURE_STOP_REASON) {
                const int code = (int)cur;
                const char* names[] = {
                    "STOP_NONE", "STOP_ABS_TOL", "STOP_REL_TOL", "STOP_MAX_ITERS", "STOP_NONFINITE", "STOP_RESIDUAL_INCREASE"
                };
                const char* s = (code >= 0 && code < (int)(sizeof(names)/sizeof(names[0]))) ? names[code] : "UNKNOWN";
                ImGui::Text("Current: %d (%s)", code, s);
            } else if (g_selectedStat == STAT_PRESSURE_SOLVER) {
                const int code = (int)cur;
                const char* s = (code == 0) ? "PCG" : (code == 1) ? "MG" : "UNKNOWN";
                ImGui::Text("Current: %d (%s)", code, s);
            } else {
                ImGui::Text("Current: %.6f", cur);
            }

            ImGui::Separator();

            g_hist[g_selectedStat].toChrono(g_plotScratch);
            if (!g_plotScratch.empty()) {
                ImGui::PlotLines("History", g_plotScratch.data(), (int)g_plotScratch.size(),
                                0, nullptr, FLT_MAX, FLT_MAX, ImVec2(0, 180));
            } else {
                ImGui::TextDisabled("No data yet.");
            }

            ImGui::EndChild();

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
                                const int id = sim.idxP(x,y);
                                sim.solid[id] = 0;
                                sim.smoke[id] = 0.0f;
                                sim.temp[id]  = 0.0f;
                                sim.age[id]   = 0.0f;
                            }
                        }
                    }
                    sim.invalidatePressureMatrix();
                    sim.enforceBoundaries();
                }
            }
        }
    }

    ImGui::End();
}

static void drawWaterViewAndInteract(MACWater& water,
                                     SmokeRenderer& renderer,
                                     Settings& ui,
                                     int NX, int NY)
{
    if (!ui.showWaterView) return;

    ImGui::SetNextWindowDockID(dock_id, ImGuiCond_FirstUseEver);
    ImGui::Begin("Water View");

    float scale = ui.viewScale;
    ImGui::Image((ImTextureID)(intptr_t)renderer.waterTex(), ImVec2(NX * scale, NY * scale));

    ImVec2 p0 = ImGui::GetItemRectMin();
    ImVec2 p1 = ImGui::GetItemRectMax();

    bool hovered = ImGui::IsItemHovered();

    if (ui.showWaterParticles && !water.particles.empty()) {
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const float w = p1.x - p0.x;
        const float h = p1.y - p0.y;
        const float domainX = std::max(1e-6f, water.nx * water.dx);
        const float domainY = std::max(1e-6f, water.ny * water.dx);

        // Cap particle drawing cost by striding when the count is huge.
        const size_t maxDraw = 20000;
        const size_t n = water.particles.size();
        const size_t stride = std::max<size_t>(1, n / maxDraw);

        // Make particles clearly visible even on high-DPI screens.
        const float radiusPx = std::max(2.0f, 0.35f * ui.viewScale);
        const ImU32 col = IM_COL32(255, 245, 120, 230);

        dl->PushClipRect(p0, p1, true);

        for (size_t k = 0; k < n; k += stride) {
            const auto& p = water.particles[k];
            const float px = p.x / domainX;
            const float py = p.y / domainY;
            if (px < 0.0f || px > 1.0f || py < 0.0f || py > 1.0f) continue;

            const float sx = p0.x + px * w;
            const float sy = p1.y - py * h;
            dl->AddCircleFilled(ImVec2(sx, sy), radiusPx, col, 8);
        }

        dl->PopClipRect();
    }

    if (hovered && ui.paintWater && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        ImVec2 m = ImGui::GetMousePos();
        float u = (m.x - p0.x) / (p1.x - p0.x);
        float v = (m.y - p0.y) / (p1.y - p0.y);

        const float domainX = water.nx * water.dx;
        const float domainY = water.ny * water.dx;
        const float scale   = std::min(domainX, domainY);

        float sx = u * domainX;
        float sy = (1.0f - v) * domainY;
        float radius  = ui.brushRadius * scale;
        float rectHalf = ui.rectHalfSize * scale;

        if (sx >= 0.0f && sx <= domainX && sy >= 0.0f && sy <= domainY) {
            if (ui.circleMode) {
                water.addWaterSource(sx, sy, radius, ui.waterAmount);
            } else {
                float hs = rectHalf;
                for (float yy = sy - hs; yy <= sy + hs; yy += water.dx) {
                    for (float xx = sx - hs; xx <= sx + hs; xx += water.dx) {
                        water.addWaterSource(xx, yy, water.dx*0.75f, ui.waterAmount);
                    }
                }
            }
        }
    }

    ImGui::End();
}

// Combined view now shows the real coupled sim (smoke + water)
static void drawCombinedView(const MACCoupledSim& coupled,
                             SmokeRenderer& coupledRenderer,
                             ImGuiID dock_id,
                             Settings& ui,
                             int NX, int NY)
{
    if (!ui.showCombinedView) return;

    ImGui::SetNextWindowDockID(dock_id, ImGuiCond_FirstUseEver);
    ImGui::Begin("Combined View");

    float scale = ui.viewScale;
    ImVec2 size(NX * scale, NY * scale);

    // Draw smoke from coupled sim
    ImGui::Image((ImTextureID)(intptr_t)coupledRenderer.smokeTex(), size);

    // Remember rect for overlays
    ImVec2 p0 = ImGui::GetItemRectMin();
    ImVec2 p1 = ImGui::GetItemRectMax();

    // Overlay water from coupled sim with alpha
    ImGui::SetCursorScreenPos(p0);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImU32 tint = ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 1.0f, ui.combinedWaterAlpha));
    dl->AddImage((ImTextureID)(intptr_t)coupledRenderer.waterTex(),
                p0, p1,
                ImVec2(0.0f, 0.0f), ImVec2(1.0f, 1.0f),
                tint);

    // Optional overlay of particles from coupled sim:
    if (ui.combinedShowParticles && !coupled.particles.empty()) {
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const float w = p1.x - p0.x;
        const float h = p1.y - p0.y;
        const float domainX = std::max(1e-6f, coupled.nx * coupled.dx);
        const float domainY = std::max(1e-6f, coupled.ny * coupled.dx);

        const size_t maxDraw = 20000;
        const size_t n = coupled.particles.size();
        const size_t stride = std::max<size_t>(1, n / maxDraw);

        const float radiusPx = std::max(2.0f, 0.35f * ui.viewScale);
        const ImU32 col = IM_COL32(255, 245, 120, 230);

        dl->PushClipRect(p0, p1, true);
        for (size_t k = 0; k < n; k += stride) {
            const auto& p = coupled.particles[k];
            const float px = p.x / domainX;
            const float py = p.y / domainY;
            if (px < 0.0f || px > 1.0f || py < 0.0f || py > 1.0f) continue;

            const float sx = p0.x + px * w;
            const float sy = p1.y - py * h;
            dl->AddCircleFilled(ImVec2(sx, sy), radiusPx, col, 8);
        }
        dl->PopClipRect();
    }

    ImGui::End();
}

Actions DrawAll(MAC2D& sim,
                MACWater& water,
                MACCoupledSim& coupled,
                SmokeRenderer& renderer,
                SmokeRenderer& coupledRenderer,
                Settings& ui,
                Probe& probe,
                int NX, int NY)
{
    // Dockspace root must be drawn BEFORE your windows
    BeginDockspaceRoot();

    // --- feed stat history once per frame ---
    if (g_recordStats) {
        const auto& st = sim.getStats();
        g_hist[STAT_DT].push(st.dt);
        g_hist[STAT_MAXDIV_BEFORE].push(st.maxDivBefore);
        g_hist[STAT_MAXDIV_AFTER].push(st.maxDivAfter);
        g_hist[STAT_MAXSPEED_BEFORE].push(st.maxFaceSpeedBefore);
        g_hist[STAT_MAXSPEED_AFTER].push(st.maxFaceSpeedAfter);
        g_hist[STAT_PRES_ITERS].push((float)st.pressureIters);
        g_hist[STAT_PRES_MS].push(st.pressureMs);

        g_hist[STAT_OP_CHECK_PASS].push((float)st.opCheckPass);
        g_hist[STAT_OP_DIFF_MAX].push(st.opDiffMax);
        g_hist[STAT_MG_RESIDUAL_INCR].push((float)st.mgResidualIncrease);
        g_hist[STAT_RHS_MAX_PREDDIV].push(st.rhsMaxPredDiv);
        g_hist[STAT_PREDDIV_INITIAL].push(st.predDivInitial);
        g_hist[STAT_PREDDIV_FINAL].push(st.predDivFinal);
        g_hist[STAT_PRESSURE_STOP_REASON].push((float)st.pressureStopReason);
        g_hist[STAT_PRESSURE_SOLVER].push((float)st.pressureSolver);
    }

    Actions a = drawControls(sim, water, ui);
    drawDebugTabs(sim, water, ui, probe);
    drawSmokeViewAndInteract(sim, renderer, ui, probe, NX, NY);
    drawWaterViewAndInteract(water, renderer, ui, NX, NY);
    drawCombinedView(coupled, coupledRenderer, dock_id, ui, NX, NY);

    // keep solids consistent between smoke and water sims
    water.syncSolidsFrom(sim);
    return a;
}

} // namespace UI
