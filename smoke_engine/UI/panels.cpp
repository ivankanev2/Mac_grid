#include "panels.h"

#include "imgui.h"
#include "imgui_internal.h"
#include <cmath>
#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <string>
#include <filesystem>

#include "../Sim/mac_smoke_sim.h"
#include "../Sim/mac_water_sim.h"
#include "../Sim/mac_water3d.h"
#include "../Sim/mac_smoke3d.h"
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

static bool g_builtDefaultLayout = false;

static constexpr const char* kWinHub        = "Vizior Hub";
static constexpr const char* kWinInspector  = "Inspector";
static constexpr const char* kWinTelemetry  = "Telemetry";
static constexpr const char* kWinSmoke2D    = "Smoke 2D Viewport";
static constexpr const char* kWinWater2D    = "Water 2D Viewport";
static constexpr const char* kWinSmoke3D    = "Smoke 3D Viewport";
static constexpr const char* kWinWater3D    = "Water 3D Viewport";
static constexpr const char* kWinCombined   = "Coupled Viewport";

static constexpr int kThemeDark = 0;
static constexpr int kThemeLight = 1;
static int g_themeMode = kThemeDark;

static ImVec4 kVizBg0;
static ImVec4 kVizBg1;
static ImVec4 kVizBg2;
static ImVec4 kVizBg3;
static ImVec4 kVizBorder;
static ImVec4 kVizText;
static ImVec4 kVizTextDim;
static ImVec4 kVizAccent;
static ImVec4 kVizAccent2;
static ImVec4 kVizWhite;
static ImVec4 kVizCard;
static ImVec4 kVizChrome;

static constexpr ImVec4 kVizLogoRed      = ImVec4(232.0f / 255.0f, 29.0f / 255.0f, 47.0f / 255.0f, 1.0f);
static constexpr ImVec4 kVizLogoLightBg  = ImVec4(250.0f / 255.0f, 250.0f / 255.0f, 249.0f / 255.0f, 1.0f);
static constexpr ImVec4 kVizLogoDarkBg   = ImVec4( 23.0f / 255.0f,  27.0f / 255.0f,  31.0f / 255.0f, 1.0f);

static ImVec4 WithAlpha(ImVec4 c, float a) {
    c.w = a;
    return c;
}

static bool IsDarkTheme() {
    return g_themeMode == kThemeDark;
}

static void SetPaletteDark() {
    kVizBg0     = ImVec4(0.055f, 0.066f, 0.078f, 1.0f);
    kVizBg1     = ImVec4(0.073f, 0.086f, 0.102f, 1.0f);
    kVizBg2     = ImVec4(0.094f, 0.110f, 0.129f, 1.0f);
    kVizBg3     = ImVec4(0.128f, 0.149f, 0.176f, 1.0f);
    kVizBorder  = ImVec4(0.185f, 0.222f, 0.258f, 1.0f);
    kVizText    = ImVec4(0.960f, 0.964f, 0.968f, 1.0f);
    kVizTextDim = ImVec4(0.620f, 0.665f, 0.720f, 1.0f);
    kVizAccent  = kVizLogoRed;
    kVizAccent2 = ImVec4(1.0f, 0.320f, 0.400f, 1.0f);
    kVizWhite   = ImVec4(0.985f, 0.988f, 0.992f, 1.0f);
    kVizCard    = ImVec4(0.090f, 0.106f, 0.122f, 0.98f);
    kVizChrome  = ImVec4(0.065f, 0.074f, 0.086f, 0.94f);
}

static void SetPaletteLight() {
    kVizBg0     = ImVec4(0.948f, 0.944f, 0.937f, 1.0f);
    kVizBg1     = ImVec4(0.980f, 0.980f, 0.976f, 1.0f);
    kVizBg2     = ImVec4(0.992f, 0.992f, 0.988f, 1.0f);
    kVizBg3     = ImVec4(0.905f, 0.912f, 0.922f, 1.0f);
    kVizBorder  = ImVec4(0.825f, 0.845f, 0.870f, 1.0f);
    kVizText    = ImVec4(0.090f, 0.106f, 0.122f, 1.0f);
    kVizTextDim = ImVec4(0.420f, 0.470f, 0.525f, 1.0f);
    kVizAccent  = kVizLogoRed;
    kVizAccent2 = ImVec4(1.0f, 0.280f, 0.360f, 1.0f);
    kVizWhite   = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
    kVizCard    = ImVec4(1.0f, 1.0f, 1.0f, 0.98f);
    kVizChrome  = ImVec4(0.988f, 0.988f, 0.982f, 0.96f);
}

static ImU32 VizColorU32(const ImVec4& c, float alphaMul = 1.0f) {
    return ImGui::GetColorU32(ImVec4(c.x, c.y, c.z, c.w * alphaMul));
}

static const char* PressureSolverLabel3D(int mode) {
    switch (mode) {
        case 0: return "Multigrid";
        case 1: return "RBGS";
        case 2: return "Jacobi";
        default: return "Unknown";
    }
}

const char* ActiveWorkspaceLabel(int workspace) {
    switch (workspace) {
        case kWorkspaceSmoke2D: return "Smoke 2D";
        case kWorkspaceWater2D: return "Water 2D";
        case kWorkspaceSmoke3D: return "Smoke 3D";
        case kWorkspaceWater3D: return "Water 3D";
        case kWorkspaceCoupled: return "Coupled";
        default: return "Unknown";
    }
}

void ApplyViziorTheme(int themeMode)
{
    g_themeMode = (themeMode == kThemeLight) ? kThemeLight : kThemeDark;
    if (IsDarkTheme()) SetPaletteDark();
    else SetPaletteLight();

    ImGuiStyle& style = ImGui::GetStyle();
    style = ImGuiStyle{};
    style.WindowPadding = ImVec2(14.0f, 12.0f);
    style.FramePadding = ImVec2(12.0f, 8.0f);
    style.CellPadding = ImVec2(8.0f, 7.0f);
    style.ItemSpacing = ImVec2(10.0f, 10.0f);
    style.ItemInnerSpacing = ImVec2(7.0f, 6.0f);
    style.TouchExtraPadding = ImVec2(0.0f, 0.0f);
    style.IndentSpacing = 20.0f;
    style.ScrollbarSize = 14.0f;
    style.GrabMinSize = 12.0f;
    style.WindowRounding = 12.0f;
    style.ChildRounding = 12.0f;
    style.FrameRounding = 9.0f;
    style.PopupRounding = 10.0f;
    style.ScrollbarRounding = 10.0f;
    style.GrabRounding = 9.0f;
    style.TabRounding = 9.0f;
    style.WindowBorderSize = 1.0f;
    style.ChildBorderSize = 1.0f;
    style.PopupBorderSize = 1.0f;
    style.FrameBorderSize = 0.0f;
    style.TabBorderSize = 0.0f;
    style.DisabledAlpha = 0.70f;
    style.WindowMenuButtonPosition = ImGuiDir_None;
    style.ColorButtonPosition = ImGuiDir_Right;
    style.AntiAliasedFill = true;
    style.AntiAliasedLines = true;
    style.AntiAliasedLinesUseTex = true;
    style.CurveTessellationTol = 1.10f;
    style.CircleTessellationMaxError = 0.18f;

    ImVec4* colors = style.Colors;
    colors[ImGuiCol_Text]                 = kVizText;
    colors[ImGuiCol_TextDisabled]         = kVizTextDim;
    colors[ImGuiCol_WindowBg]             = kVizBg0;
    colors[ImGuiCol_ChildBg]              = kVizBg1;
    colors[ImGuiCol_PopupBg]              = WithAlpha(kVizBg1, 0.98f);
    colors[ImGuiCol_Border]               = WithAlpha(kVizBorder, 0.88f);
    colors[ImGuiCol_BorderShadow]         = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
    colors[ImGuiCol_FrameBg]              = IsDarkTheme() ? WithAlpha(kVizBg2, 0.92f) : WithAlpha(kVizBg3, 0.42f);
    colors[ImGuiCol_FrameBgHovered]       = WithAlpha(kVizAccent, IsDarkTheme() ? 0.16f : 0.14f);
    colors[ImGuiCol_FrameBgActive]        = WithAlpha(kVizAccent, IsDarkTheme() ? 0.24f : 0.18f);
    colors[ImGuiCol_TitleBg]              = kVizBg0;
    colors[ImGuiCol_TitleBgActive]        = kVizBg1;
    colors[ImGuiCol_TitleBgCollapsed]     = kVizBg0;
    colors[ImGuiCol_MenuBarBg]            = IsDarkTheme() ? ImVec4(0.060f, 0.070f, 0.082f, 1.0f)
                                                           : ImVec4(0.975f, 0.975f, 0.970f, 1.0f);
    colors[ImGuiCol_ScrollbarBg]          = WithAlpha(kVizBg0, 0.92f);
    colors[ImGuiCol_ScrollbarGrab]        = IsDarkTheme() ? kVizBg3 : ImVec4(0.810f, 0.830f, 0.850f, 1.0f);
    colors[ImGuiCol_ScrollbarGrabHovered] = WithAlpha(kVizAccent, 0.50f);
    colors[ImGuiCol_ScrollbarGrabActive]  = kVizAccent;
    colors[ImGuiCol_CheckMark]            = kVizAccent;
    colors[ImGuiCol_SliderGrab]           = kVizAccent;
    colors[ImGuiCol_SliderGrabActive]     = kVizAccent2;
    colors[ImGuiCol_Button]               = WithAlpha(kVizAccent, IsDarkTheme() ? 0.18f : 0.12f);
    colors[ImGuiCol_ButtonHovered]        = WithAlpha(kVizAccent, IsDarkTheme() ? 0.30f : 0.22f);
    colors[ImGuiCol_ButtonActive]         = WithAlpha(kVizAccent2, IsDarkTheme() ? 0.38f : 0.28f);
    colors[ImGuiCol_Header]               = WithAlpha(kVizAccent, IsDarkTheme() ? 0.14f : 0.12f);
    colors[ImGuiCol_HeaderHovered]        = WithAlpha(kVizAccent, IsDarkTheme() ? 0.24f : 0.18f);
    colors[ImGuiCol_HeaderActive]         = WithAlpha(kVizAccent2, IsDarkTheme() ? 0.32f : 0.22f);
    colors[ImGuiCol_Separator]            = WithAlpha(kVizBorder, 0.90f);
    colors[ImGuiCol_SeparatorHovered]     = WithAlpha(kVizAccent, 0.38f);
    colors[ImGuiCol_SeparatorActive]      = kVizAccent;
    colors[ImGuiCol_ResizeGrip]           = WithAlpha(kVizAccent, 0.14f);
    colors[ImGuiCol_ResizeGripHovered]    = WithAlpha(kVizAccent, 0.30f);
    colors[ImGuiCol_ResizeGripActive]     = WithAlpha(kVizAccent2, 0.42f);
    colors[ImGuiCol_Tab]                  = IsDarkTheme() ? kVizBg1 : WithAlpha(kVizBg3, 0.35f);
    colors[ImGuiCol_TabHovered]           = WithAlpha(kVizAccent, 0.22f);
    colors[ImGuiCol_TabSelected]          = WithAlpha(kVizAccent, IsDarkTheme() ? 0.24f : 0.18f);
    colors[ImGuiCol_TabDimmed]            = kVizBg0;
    colors[ImGuiCol_TabDimmedSelected]    = IsDarkTheme() ? kVizBg2 : kVizBg3;
    colors[ImGuiCol_DockingPreview]       = WithAlpha(kVizAccent, 0.70f);
    colors[ImGuiCol_DockingEmptyBg]       = kVizBg0;
    colors[ImGuiCol_PlotLines]            = kVizAccent;
    colors[ImGuiCol_PlotLinesHovered]     = kVizAccent2;
    colors[ImGuiCol_PlotHistogram]        = kVizAccent;
    colors[ImGuiCol_PlotHistogramHovered] = kVizAccent2;
    colors[ImGuiCol_TableHeaderBg]        = IsDarkTheme() ? kVizBg2 : kVizBg3;
    colors[ImGuiCol_TableBorderStrong]    = kVizBorder;
    colors[ImGuiCol_TableBorderLight]     = WithAlpha(kVizBorder, 0.50f);
    colors[ImGuiCol_TableRowBg]           = WithAlpha(kVizBg1, 0.18f);
    colors[ImGuiCol_TableRowBgAlt]        = WithAlpha(kVizBg2, 0.16f);
    colors[ImGuiCol_TextSelectedBg]       = WithAlpha(kVizAccent, 0.20f);
    colors[ImGuiCol_DragDropTarget]       = kVizAccent2;
    colors[ImGuiCol_NavHighlight]         = kVizAccent;
}

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

static bool DrawSegmentedToggle(const char* label, bool active) {
    const ImVec4 fg = active ? (IsDarkTheme() ? kVizWhite : kVizWhite) : kVizText;
    ImGui::PushStyleColor(ImGuiCol_Button, active ? WithAlpha(kVizAccent, IsDarkTheme() ? 0.86f : 0.82f)
                                                  : (IsDarkTheme() ? kVizBg2 : WithAlpha(kVizBg3, 0.65f)));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, active ? kVizAccent2 : WithAlpha(kVizAccent, 0.18f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, active ? kVizAccent2 : WithAlpha(kVizAccent2, 0.24f));
    ImGui::PushStyleColor(ImGuiCol_Text, fg);
    const bool clicked = ImGui::Button(label);
    ImGui::PopStyleColor(4);
    return clicked;
}

static void DrawViziorMark(ImDrawList* dl,
                           ImVec2 center,
                           float size,
                           float alphaMul = 1.0f)
{
    const float radius = size * 0.48f;
    const ImU32 circleFill = VizColorU32(IsDarkTheme() ? kVizLogoDarkBg : kVizLogoLightBg, alphaMul);
    const ImU32 circleBorder = VizColorU32(IsDarkTheme() ? WithAlpha(kVizLogoDarkBg, 0.98f)
                                                         : kVizLogoRed, alphaMul);
    const ImU32 accent = VizColorU32(kVizLogoRed, alphaMul);
    const ImU32 innerCut = VizColorU32(IsDarkTheme() ? kVizLogoDarkBg : kVizLogoLightBg, alphaMul);

    dl->AddCircleFilled(center, radius, circleFill, 112);
    if (IsDarkTheme()) {
        dl->AddCircle(center, radius, VizColorU32(WithAlpha(kVizWhite, 0.05f), alphaMul), 112, std::max(1.0f, size * 0.014f));
    } else {
        dl->AddCircle(center, radius, circleBorder, 112, std::max(1.5f, size * 0.028f));
    }

    auto map = [&](float x, float y) {
        return ImVec2(center.x + x * size, center.y + y * size);
    };
    auto stroke = [&](const ImVec2* pts, int count, ImU32 col, float thickness) {
        dl->PathClear();
        for (int i = 0; i < count; ++i) dl->PathLineTo(pts[i]);
        dl->PathStroke(col, 0, thickness);
    };

    const ImVec2 serifL[3] = { map(-0.30f, -0.18f), map(-0.18f, -0.18f), map(-0.13f, -0.13f) };
    const ImVec2 serifR[3] = { map( 0.30f, -0.18f), map( 0.18f, -0.18f), map( 0.13f, -0.13f) };
    const ImVec2 outerPath[3] = { map(-0.20f, -0.15f), map(0.00f, 0.26f), map(0.20f, -0.15f) };
    const ImVec2 innerCutPath[3] = { map(-0.12f, -0.11f), map(0.00f, 0.15f), map(0.12f, -0.11f) };
    const ImVec2 innerCorePath[3] = { map(-0.06f, -0.08f), map(0.00f, 0.04f), map(0.06f, -0.08f) };

    const float outerW = std::max(2.0f, size * 0.140f);
    const float cutW = std::max(1.0f, size * 0.090f);
    const float coreW = std::max(1.0f, size * 0.040f);

    stroke(serifL, 3, accent, std::max(1.0f, size * 0.080f));
    stroke(serifR, 3, accent, std::max(1.0f, size * 0.080f));
    stroke(outerPath, 3, accent, outerW);

    stroke(serifL, 3, innerCut, std::max(1.0f, size * 0.045f));
    stroke(serifR, 3, innerCut, std::max(1.0f, size * 0.045f));
    stroke(innerCutPath, 3, innerCut, cutW);

    stroke(innerCorePath, 3, accent, coreW);
}

static void DrawInlineTag(const char* label,
                          const ImVec4& bg,
                          const ImVec4* fgOverride = nullptr)
{
    const ImVec4 fg = fgOverride ? *fgOverride : (IsDarkTheme() ? kVizWhite : kVizText);
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    const ImVec2 textSize = ImGui::CalcTextSize(label);
    const ImVec2 size(textSize.x + 18.0f, textSize.y + 8.0f);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(pos, ImVec2(pos.x + size.x, pos.y + size.y), VizColorU32(bg), 7.0f);
    dl->AddRect(pos, ImVec2(pos.x + size.x, pos.y + size.y), VizColorU32(WithAlpha(kVizBorder, 0.65f)), 7.0f);
    dl->AddText(ImVec2(pos.x + 9.0f, pos.y + 4.0f), VizColorU32(fg), label);
    ImGui::Dummy(size);
}

static void DrawViewportHeader(const Settings& ui,
                               const char* title,
                               const char* subtitle,
                               const char* stateTag)
{
    if (!ui.showViewportHeaders) return;

    const ImVec2 p0 = ImGui::GetCursorScreenPos();
    const float width = std::max(160.0f, ImGui::GetContentRegionAvail().x);
    const float height = 42.0f;
    const ImVec2 p1(p0.x + width, p0.y + height);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(p0, p1, VizColorU32(kVizChrome), 10.0f);
    dl->AddRect(p0, p1, VizColorU32(WithAlpha(kVizBorder, 0.85f)), 10.0f);
    dl->AddRectFilled(ImVec2(p0.x, p0.y), ImVec2(p0.x + 4.0f, p1.y), VizColorU32(kVizAccent), 10.0f);
    DrawViziorMark(dl, ImVec2(p0.x + 23.0f, p0.y + height * 0.5f), 22.0f);

    dl->AddText(ImGui::GetFont(), ImGui::GetFontSize() * 1.03f,
                ImVec2(p0.x + 42.0f, p0.y + 7.0f),
                VizColorU32(kVizText), title);
    if (subtitle && subtitle[0]) {
        dl->AddText(ImGui::GetFont(), ImGui::GetFontSize() * 0.92f,
                    ImVec2(p0.x + 42.0f, p0.y + 23.0f),
                    VizColorU32(kVizTextDim), subtitle);
    }

    if (stateTag && stateTag[0]) {
        const ImVec2 tagSize = ImGui::CalcTextSize(stateTag);
        const ImVec2 tMin(p1.x - tagSize.x - 24.0f, p0.y + 9.0f);
        const ImVec2 tMax(p1.x - 10.0f, p0.y + 9.0f + tagSize.y + 6.0f);
        dl->AddRectFilled(tMin, tMax, VizColorU32(WithAlpha(kVizAccent, IsDarkTheme() ? 0.18f : 0.14f)), 7.0f);
        dl->AddRect(tMin, tMax, VizColorU32(WithAlpha(kVizAccent, 0.35f)), 7.0f);
        dl->AddText(ImVec2(tMin.x + 8.0f, tMin.y + 3.0f), VizColorU32(IsDarkTheme() ? kVizWhite : kVizText), stateTag);
    }

    ImGui::Dummy(ImVec2(width, height + 6.0f));
}

static void DrawViewportCanvasBackdrop(const ImVec2& p0, const ImVec2& size)
{
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 p1(p0.x + size.x, p0.y + size.y);
    const ImVec4 bg = IsDarkTheme() ? ImVec4(0.045f, 0.052f, 0.061f, 1.0f)
                                    : ImVec4(0.975f, 0.974f, 0.968f, 1.0f);
    dl->AddRectFilled(p0, p1, VizColorU32(bg), 12.0f);
    dl->AddRect(p0, p1, VizColorU32(WithAlpha(kVizBorder, 0.85f)), 12.0f, 0, 1.0f);
    dl->AddLine(ImVec2(p0.x + 10.0f, p0.y + 10.0f), ImVec2(p0.x + 54.0f, p0.y + 10.0f), VizColorU32(WithAlpha(kVizAccent, 0.85f)), 2.0f);
}

static void BuildDefaultDockLayout(const ImGuiViewport* vp)
{
    ImGui::DockBuilderRemoveNode(dock_id);
    ImGui::DockBuilderAddNode(dock_id, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(dock_id, vp->Size);

    ImGuiID dock_left = 0;
    ImGuiID dock_main = dock_id;
    ImGui::DockBuilderSplitNode(dock_main, ImGuiDir_Left, 0.24f, &dock_left, &dock_main);

    ImGuiID dock_hub = 0;
    ImGuiID dock_inspector_stack = 0;
    ImGui::DockBuilderSplitNode(dock_left, ImGuiDir_Up, 0.34f, &dock_hub, &dock_inspector_stack);

    ImGuiID dock_inspector = 0;
    ImGuiID dock_telemetry = 0;
    ImGui::DockBuilderSplitNode(dock_inspector_stack, ImGuiDir_Down, 0.48f, &dock_telemetry, &dock_inspector);

    ImGui::DockBuilderDockWindow(kWinHub, dock_hub);
    ImGui::DockBuilderDockWindow(kWinInspector, dock_inspector);
    ImGui::DockBuilderDockWindow(kWinTelemetry, dock_telemetry);

    ImGui::DockBuilderDockWindow(kWinSmoke2D, dock_main);
    ImGui::DockBuilderDockWindow(kWinWater2D, dock_main);
    ImGui::DockBuilderDockWindow(kWinSmoke3D, dock_main);
    ImGui::DockBuilderDockWindow(kWinWater3D, dock_main);
    ImGui::DockBuilderDockWindow(kWinCombined, dock_main);

    ImGui::DockBuilderFinish(dock_id);
}

// ------------------------------------------------------------
// Dockspace root
// ------------------------------------------------------------
static void BeginDockspaceRoot(Settings& ui)
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

    dock_id = ImGui::GetID("ViziorDockspace");

    if (g_requestResetLayout) {
        g_requestResetLayout = false;
        const char* ini = ImGui::GetIO().IniFilename;
        if (ini) std::remove(ini);
        ImGui::DockBuilderRemoveNode(dock_id);
        g_builtDefaultLayout = false;
    }

    if (ImGui::BeginMenuBar()) {
        const ImVec2 logoPos = ImGui::GetCursorScreenPos();
        DrawViziorMark(ImGui::GetWindowDrawList(), ImVec2(logoPos.x + 11.0f, logoPos.y + 10.0f), 20.0f);
        ImGui::Dummy(ImVec2(24.0f, 20.0f));
        ImGui::SameLine(0.0f, 8.0f);
        ImGui::TextUnformatted("Vizior");
        ImGui::SameLine(0.0f, 14.0f);

        if (ImGui::Button("Save Layout")) g_requestSaveLayout = true;
        ImGui::SameLine();
        if (ImGui::Button("Reset Layout")) g_requestResetLayout = true;
        ImGui::SameLine();
        if (ImGui::Button("Panels")) ImGui::OpenPopup("AddPanelPopup");
        ImGui::SameLine(0.0f, 10.0f);
        ImGui::TextDisabled("Theme");
        ImGui::SameLine(0.0f, 6.0f);
        if (DrawSegmentedToggle("Dark##TopTheme", ui.themeMode == kThemeDark)) ui.themeMode = kThemeDark;
        ImGui::SameLine(0.0f, 4.0f);
        if (DrawSegmentedToggle("Light##TopTheme", ui.themeMode == kThemeLight)) ui.themeMode = kThemeLight;

        const char* tag = IsDarkTheme() ? "Fluid Research Engine | Dark" : "Fluid Research Engine | Light";
        const float tagW = ImGui::CalcTextSize(tag).x;
        const float availW = ImGui::GetContentRegionAvail().x;
        if (availW > tagW + 8.0f) {
            ImGui::SameLine(ImGui::GetCursorPosX() + availW - tagW);
        } else {
            ImGui::SameLine();
        }
        ImGui::TextDisabled("%s", tag);
        ImGui::EndMenuBar();
    }

    if (ImGui::BeginPopup("AddPanelPopup")) {
        ImGui::TextUnformatted("Workspace panels");
        ImGui::Separator();
        ImGui::TextDisabled("Dock any tab anywhere to build your layout.");
        ImGui::BulletText("%s", kWinHub);
        ImGui::BulletText("%s", kWinInspector);
        ImGui::BulletText("%s", kWinTelemetry);
        ImGui::BulletText("%s", kWinSmoke2D);
        ImGui::BulletText("%s", kWinWater2D);
        ImGui::BulletText("%s", kWinCombined);
        ImGui::BulletText("%s", kWinSmoke3D);
        ImGui::BulletText("%s", kWinWater3D);
        ImGui::EndPopup();
    }

    ImGui::DockSpace(dock_id, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_None);

    if (!g_builtDefaultLayout) {
        g_builtDefaultLayout = true;
        bool iniExists = false;
        if (const char* ini = ImGui::GetIO().IniFilename) {
            if (FILE* f = std::fopen(ini, "r")) {
                iniExists = true;
                std::fclose(f);
            }
        }
        if (!iniExists) {
            BuildDefaultDockLayout(vp);
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

static void DrawInspectorSectionLabel(const char* title, const char* subtitle = nullptr)
{
    ImGui::PushStyleColor(ImGuiCol_Text, kVizText);
    ImGui::TextUnformatted(title);
    ImGui::PopStyleColor();
    if (subtitle && subtitle[0]) {
        ImGui::TextDisabled("%s", subtitle);
    }
    ImGui::Separator();
}

static void DrawViziorHub(MAC2D& sim,
                          MACWater& water,
                          MACWater3D& water3D,
                          MACSmoke3D& smoke3D,
                          Settings& ui,
                          Actions& actions)
{
    ImGui::SetNextWindowDockID(dock_id, ImGuiCond_FirstUseEver);
    ImGui::Begin(kWinHub);

    const ImVec2 heroMin = ImGui::GetCursorScreenPos();
    const float heroW = std::max(160.0f, ImGui::GetContentRegionAvail().x);
    const float heroH = 166.0f;
    const ImVec2 heroMax(heroMin.x + heroW, heroMin.y + heroH);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(heroMin, heroMax, VizColorU32(kVizCard), 8.0f);
    dl->AddRect(heroMin, heroMax, VizColorU32(ImVec4(kVizAccent.x, kVizAccent.y, kVizAccent.z, 0.35f)), 8.0f);
    dl->AddRectFilled(ImVec2(heroMin.x, heroMin.y), ImVec2(heroMin.x + 5.0f, heroMax.y), VizColorU32(kVizAccent2), 8.0f);

    DrawViziorMark(dl,
                   ImVec2(heroMin.x + 74.0f, heroMin.y + heroH * 0.50f),
                   110.0f);

    dl->AddText(ImGui::GetFont(), ImGui::GetFontSize() * 2.20f,
                ImVec2(heroMin.x + 146.0f, heroMin.y + 34.0f),
                VizColorU32(kVizText), "VIZIOR");
    dl->AddText(ImGui::GetFont(), ImGui::GetFontSize() * 1.00f,
                ImVec2(heroMin.x + 148.0f, heroMin.y + 74.0f),
                VizColorU32(kVizTextDim), "Fluid Research Engine");
    dl->AddText(ImGui::GetFont(), ImGui::GetFontSize() * 0.92f,
                ImVec2(heroMin.x + 148.0f, heroMin.y + 98.0f),
                VizColorU32(kVizTextDim), "Research workspace for smoke, water and 3D pressure experiments.");

    ImGui::Dummy(ImVec2(heroW, heroH + 4.0f));

    DrawInlineTag(ui.playing ? "SIM LIVE" : "SIM PAUSED",
                  ui.playing ? ImVec4(kVizAccent.x, kVizAccent.y, kVizAccent.z, 0.28f)
                             : ImVec4(kVizBg3.x, kVizBg3.y, kVizBg3.z, 1.0f));
    ImGui::SameLine(0.0f, 6.0f);
    DrawInlineTag(ActiveWorkspaceLabel(ui.activeWorkspace),
                  ImVec4(kVizAccent.x, kVizAccent.y, kVizAccent.z, 0.20f));
    if (ui.activeWorkspace == kWorkspaceWater3D) {
        ImGui::SameLine(0.0f, 6.0f);
        DrawInlineTag(water3D.stats().backendName,
                      ImVec4(kVizBg2.x, kVizBg2.y, kVizBg2.z, 1.0f));
    }

    if (ImGui::Button(ui.playing ? "Pause Simulation" : "Run Simulation")) {
        ui.playing = !ui.playing;
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset Active Workspace")) {
        actions.resetRequested = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Save Layout")) {
        g_requestSaveLayout = true;
    }

    ImGui::SameLine();
    ui.activeWorkspace = std::clamp(ui.activeWorkspace, (int)kWorkspaceSmoke2D, (int)kWorkspaceCoupled);
    ImGui::SetNextItemWidth(170.0f);
    if (ImGui::BeginCombo("##HubWorkspaceCombo", ActiveWorkspaceLabel(ui.activeWorkspace))) {
        for (int workspace = kWorkspaceSmoke2D; workspace <= kWorkspaceCoupled; ++workspace) {
            const bool selected = (ui.activeWorkspace == workspace);
            if (ImGui::Selectable(ActiveWorkspaceLabel(workspace), selected)) {
                ui.activeWorkspace = workspace;
            }
            if (selected) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }

    ImGui::Separator();
    ImGui::TextDisabled("Only the selected workspace is stepped and rendered. Switching discards the previous simulation and boots the new workspace from a clean state.");
    if (ImGui::BeginTable("HubMetrics", 2, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV)) {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextDisabled("2D workspace");
        ImGui::Text("%dx%d grid | dt %.4f", sim.nx, sim.ny, sim.dt);
        ImGui::Text("Water particles: %zu", water.particles.size());

        ImGui::TableSetColumnIndex(1);
        ImGui::TextDisabled("3D numerics");
        ImGui::Text("Smoke: %s", PressureSolverLabel3D(ui.smoke3DPressureSolverMode));
        ImGui::Text("Water: %s", PressureSolverLabel3D(ui.water3DPressureSolverMode));
        ImGui::Text("Smoke active: %d | Water particles: %d", smoke3D.stats().activeCells, water3D.stats().particleCount);
        ImGui::EndTable();
    }

    ImGui::Separator();
    ImGui::TextDisabled("Workspace note");
    ImGui::TextWrapped("Single active-workspace mode keeps hidden smoke and water solvers from burning CPU and GPU time in the background while preserving the same solver controls for the selected view.");

    ImGui::End();
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
    outSmoke.themeMode    = ui.themeMode;

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
    outWater.themeMode = ui.themeMode;
}
static Actions drawControls(MAC2D& sim,
                            MACWater& water,
                            MACWater3D& water3D,
                            MACSmoke3D& smoke3D,
                            MACCoupledSim& coupled,
                            Settings& ui) {
    Actions a;

    ImGui::SetNextWindowDockID(dock_id, ImGuiCond_FirstUseEver);
    ImGui::Begin(kWinInspector);

    if (ImGui::CollapsingHeader("Appearance", ImGuiTreeNodeFlags_DefaultOpen)) {
        DrawInspectorSectionLabel("Appearance", "Theme, branding and viewport presentation.");
        ImGui::TextDisabled("Theme");
        if (DrawSegmentedToggle("Dark", ui.themeMode == kThemeDark)) ui.themeMode = kThemeDark;
        ImGui::SameLine();
        if (DrawSegmentedToggle("Light", ui.themeMode == kThemeLight)) ui.themeMode = kThemeLight;
        ImGui::Checkbox("Viewport headers", &ui.showViewportHeaders);
        ImGui::Checkbox("Viewport hints", &ui.showViewportHints);
        ImGui::TextDisabled("Roboto is loaded at startup so text stays cleaner and less pixelated.");
    }

    if (ImGui::CollapsingHeader("Runtime", ImGuiTreeNodeFlags_DefaultOpen)) {
        DrawInspectorSectionLabel("Runtime", "Playback, global CFL stepping and primary 2D smoke controls.");
        if (ImGui::Button(ui.playing ? "Pause" : "Play")) ui.playing = !ui.playing;
        ImGui::SameLine();
        if (ImGui::Button("Reset All Sims")) a.resetRequested = true;

        bool ot = sim.getOpenTop();
        if (ImGui::Checkbox("Open top (2D smoke outflow)", &ot)) sim.setOpenTop(ot);

        bool open = sim.isValveOpen();
        if (ImGui::Checkbox("2D smoke valve open", &open)) sim.setValveOpen(open);
        ImGui::SameLine();
        ImGui::TextDisabled("%s", sim.isValveOpen() ? "OPEN" : "CLOSED");

        ImGui::Checkbox("Use MacCormack advection", &sim.useMacCormack);
        ImGui::SliderFloat("View scale", &ui.viewScale, 1.0f, 12.0f);
        ImGui::SliderFloat("CFL", &ui.cfl, 0.1f, 1.5f);
        ImGui::SliderFloat("dt max", &ui.dtMax, 0.001f, 0.05f);
        ImGui::SliderFloat("dt min", &ui.dtMin, 0.0001f, 0.01f);

        ImGui::SliderFloat("2D inlet speed", &sim.inletSpeed, -3.0f, 3.0f);
        ImGui::SliderFloat("2D inlet smoke", &sim.inletSmoke, 0.0f, 1.0f);
        ImGui::SliderFloat("2D inlet temp", &sim.inletTemp, 0.0f, 1.0f);
        ImGui::SliderFloat("Vorticity confinement", &ui.vortEps, 0.0f, 8.0f);
        ImGui::SliderFloat("Smoke dissipation", &ui.smokeDissipation, 0.980f, 1.000f);
        ImGui::SliderFloat("Temp dissipation", &ui.tempDissipation, 0.900f, 1.000f);
    }

    if (ImGui::CollapsingHeader("Workspace", ImGuiTreeNodeFlags_DefaultOpen)) {
        DrawInspectorSectionLabel("Workspace", "Only one simulation workspace stays alive at a time to avoid hidden GPU and CPU cost.");
        static const char* workspaceItems[] = { "Smoke 2D", "Water 2D", "Smoke 3D", "Water 3D", "Coupled" };
        ui.activeWorkspace = std::clamp(ui.activeWorkspace, (int)kWorkspaceSmoke2D, (int)kWorkspaceCoupled);
        ImGui::Combo("Active workspace", &ui.activeWorkspace, workspaceItems, 5);
        ImGui::TextDisabled("Switching workspaces discards the old state and boots the selected simulation from a clean reset.");
    }

    if (ImGui::CollapsingHeader("Viewport overlays", ImGuiTreeNodeFlags_DefaultOpen)) {
        DrawInspectorSectionLabel("Viewport overlays", "Diagnostic overlays for 2D debug inspection.");
        ImGui::Checkbox("Divergence heatmap", &ui.showDivOverlay);
        ImGui::SliderFloat("Div scale", &ui.divScale, 0.1f, 50.0f);
        ImGui::SliderFloat("Div alpha", &ui.divAlpha, 0.0f, 1.0f);

        ImGui::Checkbox("Velocity arrows", &ui.showVelOverlay);
        ImGui::SliderInt("Velocity stride", &ui.velStride, 2, 16);
        ImGui::SliderFloat("Velocity scale", &ui.velScale, 0.05f, 2.0f);

        ImGui::Checkbox("Vorticity heatmap", &ui.showVortOverlay);
        ImGui::SliderFloat("Vorticity scale", &ui.vortScale, 0.1f, 50.0f);
        ImGui::SliderFloat("Vorticity alpha", &ui.vortAlpha, 0.0f, 1.0f);

        ImGui::Checkbox("Color smoke by temperature/age", &ui.useColorSmoke);
        ImGui::SliderFloat("Smoke alpha gamma", &ui.smokeAlphaGamma, 0.10f, 2.00f);
        ImGui::SliderFloat("Smoke alpha scale", &ui.smokeAlphaScale, 0.0f, 2.0f);
        ImGui::SliderFloat("Water alpha", &ui.waterAlpha, 0.0f, 1.0f);

        if (ImGui::Button("Drop MBZUAI")) {
        a.dropWaterTextRequested = true;
        }
        ImGui::SameLine();
        ImGui::TextColored(
            water.waterHeld ? ImVec4(0.2f, 0.8f, 0.2f, 1.0f) : ImVec4(0.8f, 0.2f, 0.2f, 1.0f),
            water.waterHeld ? "(Held)" : "(Falling)");
    }

    if (ImGui::CollapsingHeader("Painting & solids", ImGuiTreeNodeFlags_DefaultOpen)) {
        DrawInspectorSectionLabel("Painting & solids", "Sculpt obstacles and inject fluid directly in the active viewports.");
        ImGui::Checkbox("Paint solid", &ui.paintSolid);
        ImGui::SameLine();
        ImGui::Checkbox("Erase solid", &ui.eraseSolid);
        if (ui.eraseSolid) ui.paintWater = false;

        ImGui::Checkbox("Circular brush", &ui.circleMode);
        ImGui::SliderFloat("Brush radius", &ui.brushRadius, 0.01f, 0.20f);
        ImGui::SliderFloat("Rectangle half-size", &ui.rectHalfSize, 0.01f, 0.30f);

        if (ImGui::Checkbox("Paint water source", &ui.paintWater)) {
            if (ui.paintWater) ui.eraseSolid = false;
        }
        ImGui::SliderFloat("Water source amount", &ui.waterAmount, 0.01f, 1.00f);
        ImGui::Checkbox("Pipe authoring mode", &g_pipeMode);
        ImGui::SliderFloat("Pipe radius", &sim.pipe.radius, 0.01f, 0.25f);
        ImGui::SliderFloat("Pipe wall thickness", &sim.pipe.wall, 0.005f, 0.10f);

        if (ImGui::Button("Clear 2D pipe")) {
            sim.clearPipe();
            sim.rebuildSolidsFromPipe(false);
            sim.enforceBoundaries();
        }
        ImGui::SameLine();
        if (ImGui::Button("Rebuild 2D solids")) {
            sim.rebuildSolidsFromPipe(false);
            sim.enforceBoundaries();
        }
    }

    if (ImGui::CollapsingHeader("Water 2D", ImGuiTreeNodeFlags_DefaultOpen)) {
        DrawInspectorSectionLabel("Water 2D", "Free-surface controls for the mature 2D reference solver.");
        ImGui::TextDisabled("Viewport activation is controlled by the active workspace selector.");
        ImGui::Checkbox("Show 2D water particles", &ui.showWaterParticles);
        ImGui::Checkbox("2D water open top", &ui.waterOpenTop);
        ImGui::SliderFloat("2D water gravity", &ui.waterGravity, -20.0f, 20.0f);
        ImGui::SliderFloat("2D water dissipation", &ui.waterDissipation, 0.980f, 1.000f);
        ImGui::SliderFloat("2D water damping", &ui.waterVelDamping, 0.0f, 5.0f);
        ImGui::TextDisabled("Particles: %zu | Max particle speed: %.3f", water.particles.size(), water.maxParticleSpeed());
    }

    if (ImGui::CollapsingHeader("Smoke 3D", ImGuiTreeNodeFlags_DefaultOpen)) {
        DrawInspectorSectionLabel("Smoke 3D", "Runtime, numerics, source authoring and volume/slice display.");
        ImGui::TextDisabled("Workspace status: %s", ui.activeWorkspace == kWorkspaceSmoke3D ? "active" : "inactive");
        ImGui::Checkbox("Paint 3D smoke source", &ui.paintSmoke3D);
        ImGui::TextDisabled("Backend: %s | Active cells: %d | Max speed: %.3f",
                            smoke3D.stats().backendName,
                            smoke3D.stats().activeCells,
                            smoke3D.stats().maxSpeed);
        ImGui::SliderInt("Smoke 3D nx", &ui.smoke3DNX, 16, 160);
        ImGui::SliderInt("Smoke 3D ny", &ui.smoke3DNY, 16, 160);
        ImGui::SliderInt("Smoke 3D nz", &ui.smoke3DNZ, 16, 160);
        if (ImGui::Button("Apply Smoke 3D grid")) a.applySmoke3DGridRequested = true;
        ImGui::SameLine();
        if (ImGui::Button("Reset Smoke 3D")) a.resetSmoke3DRequested = true;

        const char* solverItems[] = { "Multigrid", "RBGS", "Jacobi" };
        ui.smoke3DPressureSolverMode = std::clamp(ui.smoke3DPressureSolverMode, 0, 2);
        ImGui::Combo("Smoke 3D pressure solver", &ui.smoke3DPressureSolverMode, solverItems, 3);
        ImGui::SliderInt("Smoke 3D pressure iters", &ui.smoke3DPressureIters, 20, 1200);
        ImGui::SliderFloat("Smoke 3D pressure omega", &ui.smoke3DPressureOmega, 0.1f, 1.95f);
        ImGui::SliderFloat("Smoke 3D buoyancy", &ui.smoke3DBuoyancyScale, 0.0f, 5.0f);
        ImGui::SliderFloat("Smoke 3D gravity", &ui.smoke3DGravity, 0.0f, 20.0f);
        ImGui::SliderFloat("Smoke 3D velocity damping", &ui.smoke3DVelDamping, 0.0f, 5.0f);
        ImGui::SliderFloat("Smoke 3D viscosity", &ui.smoke3DViscosity, 0.0f, 0.01f, "%.5f");
        ImGui::SliderFloat("Smoke 3D smoke diffusivity", &ui.smoke3DSmokeDiffusivity, 0.0f, 0.01f, "%.5f");
        ImGui::SliderFloat("Smoke 3D temp diffusivity", &ui.smoke3DTempDiffusivity, 0.0f, 0.01f, "%.5f");
        ImGui::Checkbox("Smoke 3D open top", &ui.smoke3DOpenTop);
        ImGui::SliderFloat("Smoke 3D source amount", &ui.smoke3DSourceAmount, 0.0f, 1.0f);
        ImGui::SliderFloat("Smoke 3D heat amount", &ui.smoke3DHeatAmount, 0.0f, 1.0f);
        ImGui::SliderFloat("Smoke 3D source vel X", &ui.smoke3DSourceVelX, -5.0f, 5.0f);
        ImGui::SliderFloat("Smoke 3D source vel Y", &ui.smoke3DSourceVelY, -5.0f, 5.0f);
        ImGui::SliderFloat("Smoke 3D source vel Z", &ui.smoke3DSourceVelZ, -5.0f, 5.0f);

        const char* smokeViewItems[] = { "Volume", "Slice" };
        ui.smoke3DViewMode = std::clamp(ui.smoke3DViewMode, 0, 1);
        ImGui::Combo("Smoke 3D view mode", &ui.smoke3DViewMode, smokeViewItems, 2);
        if (ui.smoke3DViewMode == 1) {
            const char* axisItems[] = { "XY", "XZ", "YZ" };
            ui.smoke3DSliceAxis = std::clamp(ui.smoke3DSliceAxis, 0, 2);
            ImGui::Combo("Smoke 3D slice axis", &ui.smoke3DSliceAxis, axisItems, 3);
            const int maxSlice = (ui.smoke3DSliceAxis == 0)
                ? std::max(0, smoke3D.nz - 1)
                : (ui.smoke3DSliceAxis == 1)
                    ? std::max(0, smoke3D.ny - 1)
                    : std::max(0, smoke3D.nx - 1);
            ui.smoke3DSliceIndex = std::clamp(ui.smoke3DSliceIndex, 0, maxSlice);
            const char* fieldItems[] = { "Smoke", "Temperature", "Pressure", "Divergence", "Speed" };
            ui.smoke3DDebugField = std::clamp(ui.smoke3DDebugField, 0, 4);
            ImGui::Combo("Smoke 3D debug field", &ui.smoke3DDebugField, fieldItems, 5);
            ImGui::SliderInt("Smoke 3D slice index", &ui.smoke3DSliceIndex, 0, std::max(0, maxSlice));
        } else {
            ImGui::SliderFloat("Smoke 3D yaw", &ui.smoke3DViewYawDeg, -180.0f, 180.0f);
            ImGui::SliderFloat("Smoke 3D pitch", &ui.smoke3DViewPitchDeg, -89.0f, 89.0f);
            ImGui::SliderFloat("Smoke 3D zoom", &ui.smoke3DViewZoom, 0.35f, 3.5f);
            if (ImGui::Button("Reset Smoke 3D view")) {
                ui.smoke3DViewYawDeg = 35.0f;
                ui.smoke3DViewPitchDeg = 18.0f;
                ui.smoke3DViewZoom = 1.15f;
            }
            ImGui::SliderFloat("Smoke 3D density scale", &ui.smoke3DVolumeDensity, 0.2f, 8.0f);
            ImGui::SliderFloat("Smoke 3D source depth", &ui.smoke3DSourceDepth, 0.0f, 1.0f);
        }
    }

    if (ImGui::CollapsingHeader("Water 3D", ImGuiTreeNodeFlags_DefaultOpen)) {
        DrawInspectorSectionLabel("Water 3D", "3D liquid path, APIC controls and multigrid-driven pressure tuning.");
        ImGui::TextDisabled("Workspace status: %s", ui.activeWorkspace == kWorkspaceWater3D ? "active" : "inactive");
        ui.water3DBackendMode = std::clamp(ui.water3DBackendMode, 0, 2);
        const char* waterBackendItems[] = { "Auto", "CPU", "CUDA" };
        ImGui::Combo("Water 3D backend", &ui.water3DBackendMode, waterBackendItems, 3);
        ImGui::TextDisabled("Backend: %s | CUDA available: %s | active: %s",
                            water3D.stats().backendName,
                            water3D.isCudaAvailable() ? "yes" : "no",
                            water3D.isCudaEnabled() ? "cuda" : "cpu");
        ImGui::TextDisabled("Particles: %d | Liquid cells: %d", water3D.stats().particleCount, water3D.stats().liquidCells);

        ImGui::SliderInt("Water 3D nx", &ui.water3DNX, 16, 192);
        ImGui::SliderInt("Water 3D ny", &ui.water3DNY, 16, 192);
        ImGui::SliderInt("Water 3D nz", &ui.water3DNZ, 16, 192);
        if (ImGui::Button("Apply Water 3D grid")) a.applyWater3DGridRequested = true;
        ImGui::SameLine();
        if (ImGui::Button("Reset Water 3D")) a.resetWater3DRequested = true;

        const char* solverItems[] = { "Multigrid", "RBGS", "Jacobi" };
        ui.water3DPressureSolverMode = std::clamp(ui.water3DPressureSolverMode, 0, 2);
        ImGui::Combo("Water 3D pressure solver", &ui.water3DPressureSolverMode, solverItems, 3);
        ImGui::SliderInt("Water 3D pressure iters", &ui.water3DPressureIters, 20, 2000);
        ImGui::SliderFloat("Water 3D pressure omega", &ui.water3DPressureOmega, 0.1f, 1.95f);
        ImGui::Checkbox("Water 3D APIC", &ui.water3DUseAPIC);
        ImGui::BeginDisabled(ui.water3DUseAPIC);
        ImGui::SliderFloat("Water 3D FLIP blend", &ui.water3DFlipBlend, 0.0f, 1.0f);
        ImGui::EndDisabled();
        ImGui::Checkbox("Water 3D volume preservation", &ui.water3DVolumePreserve);
        ImGui::SliderFloat("Water 3D preserve strength", &ui.water3DVolumePreserveStrength, 0.0f, 0.25f);
        ImGui::SliderInt("Water 3D relax iters", &ui.water3DRelaxIters, 0, 8);
        ImGui::SliderFloat("Water 3D relax strength", &ui.water3DRelaxStrength, 0.0f, 1.0f);
        ImGui::SliderFloat("Water 3D source vel X", &ui.water3DSourceVelX, -5.0f, 5.0f);
        ImGui::SliderFloat("Water 3D source vel Y", &ui.water3DSourceVelY, -5.0f, 5.0f);
        ImGui::SliderFloat("Water 3D source vel Z", &ui.water3DSourceVelZ, -5.0f, 5.0f);

        const char* waterViewItems[] = { "Volume", "Slice", "Surface" };
        ui.water3DViewMode = std::clamp(ui.water3DViewMode, 0, 2);
        ImGui::Combo("Water 3D view mode", &ui.water3DViewMode, waterViewItems, 3);
        if (ui.water3DViewMode == 1) {
            const char* axisItems[] = { "XY", "XZ", "YZ" };
            ui.water3DSliceAxis = std::clamp(ui.water3DSliceAxis, 0, 2);
            ImGui::Combo("Water 3D slice axis", &ui.water3DSliceAxis, axisItems, 3);
            const int maxSlice = (ui.water3DSliceAxis == 0)
                ? std::max(0, water3D.nz - 1)
                : (ui.water3DSliceAxis == 1)
                    ? std::max(0, water3D.ny - 1)
                    : std::max(0, water3D.nx - 1);
            ui.water3DSliceIndex = std::clamp(ui.water3DSliceIndex, 0, maxSlice);
            const char* fieldItems[] = { "Water", "Pressure", "Divergence", "Speed" };
            ui.water3DDebugField = std::clamp(ui.water3DDebugField, 0, 3);
            ImGui::Combo("Water 3D debug field", &ui.water3DDebugField, fieldItems, 4);
            ImGui::SliderInt("Water 3D slice index", &ui.water3DSliceIndex, 0, std::max(0, maxSlice));
        } else {
            ImGui::SliderFloat("Water 3D yaw", &ui.water3DViewYawDeg, -180.0f, 180.0f);
            ImGui::SliderFloat("Water 3D pitch", &ui.water3DViewPitchDeg, -89.0f, 89.0f);
            ImGui::SliderFloat("Water 3D zoom", &ui.water3DViewZoom, 0.35f, 3.5f);
            if (ImGui::Button("Reset Water 3D view")) {
                ui.water3DViewYawDeg = 35.0f;
                ui.water3DViewPitchDeg = 20.0f;
                ui.water3DViewZoom = 1.15f;
            }
            ImGui::SliderFloat("Water 3D density scale", &ui.water3DVolumeDensity, 0.2f, 8.0f);
            ImGui::SliderFloat("Water 3D surface threshold", &ui.water3DSurfaceThreshold, 0.01f, 0.75f);
            ImGui::SliderFloat("Water 3D source depth", &ui.water3DSourceDepth, 0.0f, 1.0f);
        }
    }

    if (ImGui::CollapsingHeader("Coupled & pipe workspace", ImGuiTreeNodeFlags_DefaultOpen)) {
        DrawInspectorSectionLabel("Coupled & pipe workspace", "Overlay water on smoke, inspect the coupled path, and author pipe sketches.");
        ImGui::TextDisabled("Workspace status: %s", ui.activeWorkspace == kWorkspaceCoupled ? "active" : "inactive");
        ImGui::SliderFloat("Coupled water alpha", &ui.combinedWaterAlpha, 0.0f, 1.0f);
        ImGui::Checkbox("Coupled particles", &ui.combinedShowParticles);

        bool copen = coupled.isValveOpen();
        if (ImGui::Checkbox("Coupled valve open", &copen)) coupled.setValveOpen(copen);
        ImGui::SameLine();
        ImGui::TextDisabled("%s", coupled.isValveOpen() ? "OPEN" : "CLOSED");
        ImGui::SliderFloat("Coupled inlet speed", &coupled.inletSpeed, -3.0f, 3.0f);
        ImGui::SliderFloat("Coupled inlet smoke", &coupled.inletSmoke, 0.0f, 1.0f);
        ImGui::SliderFloat("Coupled inlet temp", &coupled.inletTemp, 0.0f, 1.0f);

        ImGui::Checkbox("Coupled pipe mode", &g_pipeMode);
        ImGui::SliderFloat("Coupled pipe radius", &coupled.pipe.radius, 0.01f, 0.25f);
        ImGui::SliderFloat("Coupled wall thickness", &coupled.pipe.wall, 0.005f, 0.10f);
        if (ImGui::Button("Clear coupled pipe")) {
            coupled.clearPipe();
            coupled.rebuildSolidsFromPipe(false);
            coupled.enforceBoundaries();
        }
        ImGui::SameLine();
        if (ImGui::Button("Rebuild coupled solids")) {
            coupled.rebuildSolidsFromPipe(false);
            coupled.enforceBoundaries();
        }
    }

    ImGui::End();
    return a;
}

static void drawDebugTabs(MAC2D& sim,
                          MACWater& water,
                          MACWater3D& water3D,
                          MACSmoke3D& smoke3D,
                          Settings& ui,
                          Probe& probe) {
    ImGui::SetNextWindowDockID(dock_id, ImGuiCond_FirstUseEver);
    ImGui::Begin(kWinTelemetry);

    ImGui::TextDisabled("Pressure, divergence, advection and solver diagnostics.");
    ImGui::Separator();

    if (ImGui::BeginTabBar("DebugTabs")) {

        if (ImGui::BeginTabItem("Numerics")) {
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
            ImGui::TextUnformatted("Water (2D)");
            ImGui::Text("openTop: %s   borderThickness: %d", water.openTop ? "true" : "false", water.borderThickness);
            ImGui::Text("particles: %zu   inSolid: %d", wst.particles, wst.particlesInSolid);
            ImGui::Text("sum(water): %.6f   targetMass: %.6f", wst.sumWater, water.targetMass);
            ImGui::Text("max(water): %.6f", wst.maxWater);
            ImGui::Text("liquid cells: %d   interior: %d", wst.liquidCount, wst.interiorLiquidCount);
            ImGui::Text("particle mass proxy: %.0f", wst.particleMassProxy);
            ImGui::Text("div (all) max/avg: %.3e / %.3e", wst.maxAbsDiv, wst.avgAbsDiv);
            ImGui::Text("div (int) max/avg: %.3e / %.3e", wst.maxAbsDivInterior, wst.avgAbsDivInterior);
            ImGui::Text("near borders L/R/B/T: %d / %d / %d / %d", wst.nearLeft, wst.nearRight, wst.nearBottom, wst.nearTop);

            if (ui.useWater3D || ui.showWater3DView) {
                const auto& st3 = water3D.stats();
                ImGui::Separator();
                ImGui::TextUnformatted("Water (3D)");
                ImGui::Text("grid: %d x %d x %d   dt: %.6f", water3D.nx, water3D.ny, water3D.nz, water3D.dt);
                ImGui::Text("backend: %s", st3.backendName);
                ImGui::Text("particles: %d   liquid cells: %d", st3.particleCount, st3.liquidCells);
                ImGui::Text("max speed: %.6f   max |div|: %.6e", st3.maxSpeed, st3.maxDivergence);
                ImGui::Text("step ms: %.3f", st3.lastStepMs);
                ImGui::Text("cuda enabled: %s", st3.cudaEnabled ? "true" : "false");
                ImGui::Text("feature parity with 2D: %s", water3D.hasFeatureParityWith2D() ? "true" : "false");
            }

            if (ui.useSmoke3D || ui.showSmoke3DView) {
                const auto& s3 = smoke3D.stats();
                ImGui::Separator();
                ImGui::TextUnformatted("Smoke (3D)");
                ImGui::Text("grid: %d x %d x %d   dt: %.6f", smoke3D.nx, smoke3D.ny, smoke3D.nz, smoke3D.dt);
                ImGui::Text("backend: %s", s3.backendName);
                ImGui::Text("active cells: %d", s3.activeCells);
                ImGui::Text("max speed: %.6f   max |div|: %.6e", s3.maxSpeed, s3.maxDivergence);
                ImGui::Text("step ms: %.3f", s3.lastStepMs);
                ImGui::Text("bytes: %zu", s3.bytesAllocated);
            }

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

        if (ImGui::BeginTabItem("Timeline")) {
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

        if (ImGui::BeginTabItem("Rendering")) {
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
    ImGui::Begin(kWinSmoke2D);

    DrawViewportHeader(ui, "Smoke 2D", "MAC grid | multigrid | solids", ui.playing ? "LIVE" : "PAUSED");
    const float scale = ui.viewScale;
    const ImVec2 imageSize(NX * scale, NY * scale);
    const ImVec2 imagePos = ImGui::GetCursorScreenPos();
    DrawViewportCanvasBackdrop(imagePos, imageSize);
    ImGui::Image((ImTextureID)(intptr_t)renderer.smokeTex(), imageSize);

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
            dl->AddLine(a, b, IM_COL32(225, 64, 72, 220), 2.0f);
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

            // // propagate pipe/solids to coupled sim
            // coupled.rebuildSolidsFromPipe(false);
            // coupled.enforceBoundaries();
            // coupled.invalidatePressureMatrix();
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

                    // // keep coupled sim solids & pressure config in sync
                    // coupled.syncSolidsFrom(sim);
                    // coupled.invalidatePressureMatrix();
                    // coupled.enforceBoundaries();
                }
            }
        }
    }

    if (ui.showViewportHints) {
        ImGui::TextDisabled("LMB paint solids or erase | Pipe mode uses click placement in-view.");
    }

    ImGui::End();
}


struct Water3DViewVec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

struct Water3DPanelBox {
    float hx = 0.5f;
    float hy = 0.5f;
    float hz = 0.5f;
    float camDist = 1.85f;
    float fovScale = 0.95f;
    float imageAspect = 1.0f;
};

static Water3DViewVec3 operator+(Water3DViewVec3 a, Water3DViewVec3 b) {
    return Water3DViewVec3{a.x + b.x, a.y + b.y, a.z + b.z};
}

static Water3DViewVec3 operator-(Water3DViewVec3 a, Water3DViewVec3 b) {
    return Water3DViewVec3{a.x - b.x, a.y - b.y, a.z - b.z};
}

static Water3DViewVec3 operator*(Water3DViewVec3 a, float s) {
    return Water3DViewVec3{a.x * s, a.y * s, a.z * s};
}

static float dotWater3DView(Water3DViewVec3 a, Water3DViewVec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static float lengthWater3DView(Water3DViewVec3 v) {
    return std::sqrt(dotWater3DView(v, v));
}

static Water3DViewVec3 normalizeWater3DView(Water3DViewVec3 v) {
    const float len = lengthWater3DView(v);
    if (len <= 1e-8f) return Water3DViewVec3{0.0f, 0.0f, 1.0f};
    const float invLen = 1.0f / len;
    return Water3DViewVec3{v.x * invLen, v.y * invLen, v.z * invLen};
}

static Water3DViewVec3 rotateYawPitchWater3DView(Water3DViewVec3 v, float yawRad, float pitchRad) {
    const float cy = std::cos(yawRad);
    const float sy = std::sin(yawRad);
    const float cp = std::cos(pitchRad);
    const float sp = std::sin(pitchRad);
    return Water3DViewVec3{
        cy * v.x + sy * v.z,
        cp * v.y - sp * (-sy * v.x + cy * v.z),
        sp * v.y + cp * (-sy * v.x + cy * v.z)
    };
}

static Water3DViewVec3 rotateInvYawPitchWater3DView(Water3DViewVec3 v, float yawRad, float pitchRad) {
    const float cy = std::cos(yawRad);
    const float sy = std::sin(yawRad);
    const float cp = std::cos(pitchRad);
    const float sp = std::sin(pitchRad);
    return Water3DViewVec3{
        cy * v.x + sy * sp * v.y - sy * cp * v.z,
        cp * v.y + sp * v.z,
        sy * v.x - cy * sp * v.y + cy * cp * v.z
    };
}

static Water3DPanelBox makeWater3DPanelBox(int nx, int ny, int nz, float imageAspect) {
    const int maxDim = std::max({nx, ny, nz, 1});
    Water3DPanelBox box;
    box.hx = 0.5f * (float)nx / (float)maxDim;
    box.hy = 0.5f * (float)ny / (float)maxDim;
    box.hz = 0.5f * (float)nz / (float)maxDim;
    box.imageAspect = std::max(1e-6f, imageAspect);
    return box;
}

static bool intersectWater3DPanelBox(const Water3DViewVec3& o,
                                     const Water3DViewVec3& d,
                                     const Water3DPanelBox& box,
                                     float& tminOut,
                                     float& tmaxOut) {
    float tmin = 0.0f;
    float tmax = 1.0e30f;

    auto updateAxis = [&](float oAxis, float dAxis, float halfExtent) -> bool {
        const float lo = -halfExtent;
        const float hi = halfExtent;
        if (std::fabs(dAxis) < 1e-8f) {
            return (oAxis >= lo && oAxis <= hi);
        }
        float t0 = (lo - oAxis) / dAxis;
        float t1 = (hi - oAxis) / dAxis;
        if (t0 > t1) std::swap(t0, t1);
        tmin = std::max(tmin, t0);
        tmax = std::min(tmax, t1);
        return tmax >= tmin;
    };

    if (!updateAxis(o.x, d.x, box.hx)) return false;
    if (!updateAxis(o.y, d.y, box.hy)) return false;
    if (!updateAxis(o.z, d.z, box.hz)) return false;

    tminOut = tmin;
    tmaxOut = tmax;
    return tmaxOut > tminOut;
}

static Water3DViewVec3 unitToWater3DPanelLocal(float ux, float uy, float uz, const Water3DPanelBox& box) {
    return Water3DViewVec3{
        (ux - 0.5f) * (2.0f * box.hx),
        (uy - 0.5f) * (2.0f * box.hy),
        (uz - 0.5f) * (2.0f * box.hz)
    };
}

static Water3DViewVec3 water3DPanelLocalToUnit(Water3DViewVec3 p, const Water3DPanelBox& box) {
    return Water3DViewVec3{
        (p.x + box.hx) / std::max(1e-6f, 2.0f * box.hx),
        (p.y + box.hy) / std::max(1e-6f, 2.0f * box.hy),
        (p.z + box.hz) / std::max(1e-6f, 2.0f * box.hz)
    };
}

static ImVec2 projectWater3DPanelPoint(Water3DViewVec3 local,
                                       const Water3DPanelBox& box,
                                       float yawDeg,
                                       float pitchDeg,
                                       float zoom,
                                       const ImVec2& p0,
                                       const ImVec2& p1,
                                       float* depthOut = nullptr) {
    const float yaw = yawDeg * 3.14159265358979323846f / 180.0f;
    const float pitch = pitchDeg * 3.14159265358979323846f / 180.0f;
    const float zoomClamped = std::clamp(zoom, 0.35f, 3.5f);

    const Water3DViewVec3 camP = rotateYawPitchWater3DView(local, yaw, pitch);
    const float camZ = camP.z + box.camDist / zoomClamped;
    if (depthOut) *depthOut = camZ;
    if (camZ <= 1e-4f) {
        return ImVec2(-10000.0f, -10000.0f);
    }

    const float pxAspect = camP.x * box.camDist / std::max(1e-6f, camZ * box.fovScale);
    const float py = camP.y * box.camDist / std::max(1e-6f, camZ * box.fovScale);
    const float ndcX = pxAspect / box.imageAspect;
    const float ndcY = py;

    const float sx = p0.x + (ndcX * 0.5f + 0.5f) * (p1.x - p0.x);
    const float sy = p0.y + (1.0f - (ndcY * 0.5f + 0.5f)) * (p1.y - p0.y);
    return ImVec2(sx, sy);
}

static bool makeWater3DPanelRay(const ImVec2& mouse,
                                const ImVec2& p0,
                                const ImVec2& p1,
                                const Water3DPanelBox& box,
                                float yawDeg,
                                float pitchDeg,
                                float zoom,
                                Water3DViewVec3& originOut,
                                Water3DViewVec3& dirOut) {
    const float w = std::max(1e-6f, p1.x - p0.x);
    const float h = std::max(1e-6f, p1.y - p0.y);
    const float u = std::clamp((mouse.x - p0.x) / w, 0.0f, 1.0f);
    const float v = std::clamp((mouse.y - p0.y) / h, 0.0f, 1.0f);
    const float yaw = yawDeg * 3.14159265358979323846f / 180.0f;
    const float pitch = pitchDeg * 3.14159265358979323846f / 180.0f;
    const float zoomClamped = std::clamp(zoom, 0.35f, 3.5f);

    const float px = (2.0f * u - 1.0f) * box.imageAspect;
    const float py = 1.0f - 2.0f * v;
    const Water3DViewVec3 rayOriginWorld{0.0f, 0.0f, -box.camDist / zoomClamped};
    const Water3DViewVec3 rayDirWorld = normalizeWater3DView(Water3DViewVec3{px * box.fovScale, py * box.fovScale, box.camDist});

    originOut = rotateInvYawPitchWater3DView(rayOriginWorld, yaw, pitch);
    dirOut = normalizeWater3DView(rotateInvYawPitchWater3DView(rayDirWorld, yaw, pitch));
    return true;
}

static void Handle3DNavigation(bool hovered,
                               float& yawDeg,
                               float& pitchDeg,
                               float& zoom,
                               float resetYaw,
                               float resetPitch,
                               float resetZoom)
{
    if (!hovered) return;

    ImGuiIO& io = ImGui::GetIO();
    const bool alt = io.KeyAlt;
    const bool zoomDrag = (alt && ImGui::IsMouseDown(ImGuiMouseButton_Right)) ||
                          (io.KeyCtrl && ImGui::IsMouseDown(ImGuiMouseButton_Middle));
    const bool orbitDrag = ImGui::IsMouseDown(ImGuiMouseButton_Right) ||
                           ImGui::IsMouseDown(ImGuiMouseButton_Middle) ||
                           (alt && ImGui::IsMouseDown(ImGuiMouseButton_Left));

    if (zoomDrag) {
        zoom = std::clamp(zoom * std::pow(1.014f, -io.MouseDelta.y), 0.35f, 3.5f);
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
    } else if (orbitDrag) {
        const float speed = io.KeyShift ? 0.68f : 0.36f;
        yawDeg += io.MouseDelta.x * speed;
        pitchDeg = std::clamp(pitchDeg - io.MouseDelta.y * speed, -89.0f, 89.0f);
        if (yawDeg > 180.0f) yawDeg -= 360.0f;
        if (yawDeg < -180.0f) yawDeg += 360.0f;
        ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
    }

    if (std::fabs(io.MouseWheel) > 1e-6f) {
        zoom = std::clamp(zoom * std::pow(1.12f, io.MouseWheel), 0.35f, 3.5f);
    }

    if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Middle) ||
        (alt && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))) {
        yawDeg = resetYaw;
        pitchDeg = resetPitch;
        zoom = resetZoom;
    }
}

static void drawWaterViewAndInteract(MACWater& water,
                                     SmokeRenderer& renderer,
                                     Settings& ui)
{
    if (!ui.showWaterView) return;

    ImGui::SetNextWindowDockID(dock_id, ImGuiCond_FirstUseEver);
    ImGui::Begin(kWinWater2D);

    DrawViewportHeader(ui, "Water 2D", "Particle / grid liquid", ui.paintWater ? "SOURCE" : "VIEW");
    const float scale = ui.viewScale;
    const int texW = std::max(1, renderer.width());
    const int texH = std::max(1, renderer.height());
    const ImVec2 imageSize(texW * scale, texH * scale);
    const ImVec2 imagePos = ImGui::GetCursorScreenPos();
    DrawViewportCanvasBackdrop(imagePos, imageSize);
    ImGui::Image((ImTextureID)(intptr_t)renderer.waterTex(), imageSize);

    const ImVec2 p0 = ImGui::GetItemRectMin();
    const ImVec2 p1 = ImGui::GetItemRectMax();
    const bool hovered = ImGui::IsItemHovered();

    if (ui.showWaterParticles && !water.particles.empty()) {
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const float w = p1.x - p0.x;
        const float h = p1.y - p0.y;
        const float domainX = std::max(1e-6f, water.nx * water.dx);
        const float domainY = std::max(1e-6f, water.ny * water.dx);

        const size_t maxDraw = 20000;
        const size_t n = water.particles.size();
        const size_t stride = std::max<size_t>(1, n / maxDraw);
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
        const ImVec2 m = ImGui::GetMousePos();
        const float u = (m.x - p0.x) / std::max(1e-6f, (p1.x - p0.x));
        const float v = (m.y - p0.y) / std::max(1e-6f, (p1.y - p0.y));

        const float domainX = water.nx * water.dx;
        const float domainY = water.ny * water.dx;
        const float scale2D = std::min(domainX, domainY);

        const float sx = u * domainX;
        const float sy = (1.0f - v) * domainY;
        const float radius = ui.brushRadius * scale2D;
        const float rectHalf = ui.rectHalfSize * scale2D;

        if (sx >= 0.0f && sx <= domainX && sy >= 0.0f && sy <= domainY) {
            if (ui.circleMode) {
                water.addWaterSource(sx, sy, radius, ui.waterAmount);
            } else {
                const float hs = rectHalf;
                for (float yy = sy - hs; yy <= sy + hs; yy += water.dx) {
                    for (float xx = sx - hs; xx <= sx + hs; xx += water.dx) {
                        water.addWaterSource(xx, yy, water.dx * 0.75f, ui.waterAmount);
                    }
                }
            }
        }
    }

    if (ui.showViewportHints) {
        ImGui::TextDisabled("LMB inject water source directly into the 2D liquid domain.");
    }

    ImGui::End();
}

static void drawSmoke3DViewAndInteract(MACSmoke3D& smoke3D,
                                       SmokeRenderer& renderer,
                                       Settings& ui)
{
    if (!ui.showSmoke3DView) return;

    ImGui::SetNextWindowDockID(dock_id, ImGuiCond_FirstUseEver);
    ImGui::Begin(kWinSmoke3D);

    DrawViewportHeader(ui, "Smoke 3D", PressureSolverLabel3D(ui.smoke3DPressureSolverMode), ui.useSmoke3D ? "ACTIVE" : "IDLE");
    const float scale = ui.viewScale;
    const int texW = std::max(1, renderer.width());
    const int texH = std::max(1, renderer.height());
    const ImVec2 imageSize(texW * scale, texH * scale);
    const ImVec2 imagePos = ImGui::GetCursorScreenPos();
    DrawViewportCanvasBackdrop(imagePos, imageSize);
    ImGui::Image((ImTextureID)(intptr_t)renderer.smokeTex(), imageSize);

    const ImVec2 p0 = ImGui::GetItemRectMin();
    const ImVec2 p1 = ImGui::GetItemRectMax();
    const bool hovered = ImGui::IsItemHovered();

    const int viewMode = std::clamp(ui.smoke3DViewMode, 0, 1);
    const bool sliceMode = (viewMode == 1);
    const float domainX = std::max(1e-6f, smoke3D.nx * smoke3D.dx);
    const float domainY = std::max(1e-6f, smoke3D.ny * smoke3D.dx);
    const float domainZ = std::max(1e-6f, smoke3D.nz * smoke3D.dx);
    const bool canPaintVolume = hovered && ui.paintSmoke3D && ImGui::IsMouseDown(ImGuiMouseButton_Left)
                             && !ImGui::GetIO().KeyAlt
                             && !ImGui::IsMouseDown(ImGuiMouseButton_Right)
                             && !ImGui::IsMouseDown(ImGuiMouseButton_Middle);
    const bool canPaintSlice = hovered && ui.paintSmoke3D && ImGui::IsMouseDown(ImGuiMouseButton_Left)
                            && !ImGui::GetIO().KeyAlt;

    if (!sliceMode) {
        Handle3DNavigation(hovered,
                           ui.smoke3DViewYawDeg,
                           ui.smoke3DViewPitchDeg,
                           ui.smoke3DViewZoom,
                           35.0f, 18.0f, 1.15f);

        if (ui.showViewportHints) {
            ImGui::TextDisabled("MMB/RMB/Alt+LMB orbit | wheel or Alt+RMB zoom | double-click MMB reset | LMB paint.");
        }

        const Water3DPanelBox box = makeWater3DPanelBox(
            smoke3D.nx,
            smoke3D.ny,
            smoke3D.nz,
            std::max(1e-6f, (p1.x - p0.x) / std::max(1e-6f, p1.y - p0.y)));

        ImDrawList* dl = ImGui::GetWindowDrawList();
        dl->PushClipRect(p0, p1, true);

        const Water3DViewVec3 corners[8] = {
            Water3DViewVec3{-box.hx, -box.hy, -box.hz},
            Water3DViewVec3{ box.hx, -box.hy, -box.hz},
            Water3DViewVec3{ box.hx,  box.hy, -box.hz},
            Water3DViewVec3{-box.hx,  box.hy, -box.hz},
            Water3DViewVec3{-box.hx, -box.hy,  box.hz},
            Water3DViewVec3{ box.hx, -box.hy,  box.hz},
            Water3DViewVec3{ box.hx,  box.hy,  box.hz},
            Water3DViewVec3{-box.hx,  box.hy,  box.hz}
        };
        const int edges[12][2] = {
            {0,1},{1,2},{2,3},{3,0},
            {4,5},{5,6},{6,7},{7,4},
            {0,4},{1,5},{2,6},{3,7}
        };
        ImVec2 proj[8];
        for (int i = 0; i < 8; ++i) {
            proj[i] = projectWater3DPanelPoint(corners[i], box,
                                               ui.smoke3DViewYawDeg,
                                               ui.smoke3DViewPitchDeg,
                                               ui.smoke3DViewZoom,
                                               p0, p1, nullptr);
        }
        const ImU32 edgeCol = IsDarkTheme() ? IM_COL32(240, 242, 248, 200)
                                            : IM_COL32(92, 102, 114, 182);
        for (int e = 0; e < 12; ++e) {
            dl->AddLine(proj[edges[e][0]], proj[edges[e][1]], edgeCol, 1.2f);
        }
        dl->PopClipRect();

        if (canPaintVolume) {
            Water3DViewVec3 rayOrigin{};
            Water3DViewVec3 rayDir{};
            if (makeWater3DPanelRay(ImGui::GetMousePos(), p0, p1, box,
                                    ui.smoke3DViewYawDeg,
                                    ui.smoke3DViewPitchDeg,
                                    ui.smoke3DViewZoom,
                                    rayOrigin, rayDir)) {
                float tmin = 0.0f;
                float tmax = 0.0f;
                if (intersectWater3DPanelBox(rayOrigin, rayDir, box, tmin, tmax)) {
                    const float depth01 = std::clamp(ui.smoke3DSourceDepth, 0.0f, 1.0f);
                    const float t = std::max(0.0f, tmin) + depth01 * std::max(0.0f, tmax - std::max(0.0f, tmin));
                    const Water3DViewVec3 hitLocal = rayOrigin + rayDir * t;
                    const Water3DViewVec3 hitUnit = water3DPanelLocalToUnit(hitLocal, box);

                    MACSmoke3D::Vec3 center{};
                    center.x = std::clamp(hitUnit.x, 0.0f, 1.0f) * domainX;
                    center.y = std::clamp(hitUnit.y, 0.0f, 1.0f) * domainY;
                    center.z = std::clamp(hitUnit.z, 0.0f, 1.0f) * domainZ;

                    const float sourceScale = std::min({domainX, domainY, domainZ});
                    const float radius = std::max(smoke3D.dx, ui.brushRadius * sourceScale);
                    const MACSmoke3D::Vec3 velocity{ui.smoke3DSourceVelX, ui.smoke3DSourceVelY, ui.smoke3DSourceVelZ};
                    smoke3D.addSmokeSourceSphere(center, radius, ui.smoke3DSourceAmount, velocity);
                    smoke3D.addHeatSourceSphere(center, radius, ui.smoke3DHeatAmount);
                }
            }
        }
    } else {
        if (canPaintSlice) {
            const ImVec2 m = ImGui::GetMousePos();
            const float u = (m.x - p0.x) / std::max(1e-6f, (p1.x - p0.x));
            const float v = (m.y - p0.y) / std::max(1e-6f, (p1.y - p0.y));
            const float nu = std::clamp(u, 0.0f, 1.0f);
            const float nv = std::clamp(1.0f - v, 0.0f, 1.0f);

            const int axis = std::clamp(ui.smoke3DSliceAxis, 0, 2);
            const int maxSlice = (axis == 0)
                ? std::max(0, smoke3D.nz - 1)
                : (axis == 1)
                    ? std::max(0, smoke3D.ny - 1)
                    : std::max(0, smoke3D.nx - 1);
            const int sliceIndex = std::clamp(ui.smoke3DSliceIndex, 0, maxSlice);
            const float sliceCoord = (sliceIndex + 0.5f) * smoke3D.dx;

            MACSmoke3D::Vec3 center{};
            if (axis == 0) {
                center.x = nu * domainX;
                center.y = nv * domainY;
                center.z = sliceCoord;
            } else if (axis == 1) {
                center.x = nu * domainX;
                center.y = sliceCoord;
                center.z = nv * domainZ;
            } else {
                center.x = sliceCoord;
                center.y = nu * domainY;
                center.z = nv * domainZ;
            }

            const float sourceScale = std::min({domainX, domainY, domainZ});
            const float radius = std::max(smoke3D.dx, ui.brushRadius * sourceScale);
            const MACSmoke3D::Vec3 velocity{ui.smoke3DSourceVelX, ui.smoke3DSourceVelY, ui.smoke3DSourceVelZ};
            smoke3D.addSmokeSourceSphere(center, radius, ui.smoke3DSourceAmount, velocity);
            smoke3D.addHeatSourceSphere(center, radius, ui.smoke3DHeatAmount);
        }
    }

    if (sliceMode && ui.showViewportHints) {
        ImGui::TextDisabled("Slice paint: LMB inject smoke and heat | switch axis/index in Inspector.");
    }

    ImGui::End();
}

static void drawWater3DViewAndInteract(MACWater3D& water3D,
                                       SmokeRenderer& renderer,
                                       Settings& ui)
{
    if (!ui.showWater3DView) return;

    ImGui::SetNextWindowDockID(dock_id, ImGuiCond_FirstUseEver);
    ImGui::Begin(kWinWater3D);

    DrawViewportHeader(ui, "Water 3D", PressureSolverLabel3D(ui.water3DPressureSolverMode), ui.useWater3D ? "ACTIVE" : "IDLE");
    const float scale = ui.viewScale;
    const int texW = std::max(1, renderer.width());
    const int texH = std::max(1, renderer.height());
    const ImVec2 imageSize(texW * scale, texH * scale);
    const ImVec2 imagePos = ImGui::GetCursorScreenPos();
    DrawViewportCanvasBackdrop(imagePos, imageSize);
    ImGui::Image((ImTextureID)(intptr_t)renderer.waterTex(), imageSize);

    const ImVec2 p0 = ImGui::GetItemRectMin();
    const ImVec2 p1 = ImGui::GetItemRectMax();
    const bool hovered = ImGui::IsItemHovered();

    const int viewMode = std::clamp(ui.water3DViewMode, 0, 2);
    const bool sliceMode = (viewMode == 1);
    const float domainX = std::max(1e-6f, water3D.nx * water3D.dx);
    const float domainY = std::max(1e-6f, water3D.ny * water3D.dx);
    const float domainZ = std::max(1e-6f, water3D.nz * water3D.dx);

    if (!sliceMode) {
        Handle3DNavigation(hovered,
                           ui.water3DViewYawDeg,
                           ui.water3DViewPitchDeg,
                           ui.water3DViewZoom,
                           35.0f,
                           20.0f,
                           1.15f);
    }

    const bool canPaintVolume = hovered && ui.paintWater && ImGui::IsMouseDown(ImGuiMouseButton_Left) &&
                                !ImGui::GetIO().KeyAlt &&
                                !ImGui::IsMouseDown(ImGuiMouseButton_Right) &&
                                !ImGui::IsMouseDown(ImGuiMouseButton_Middle);
    const bool canPaintSlice = hovered && ui.paintWater && ImGui::IsMouseDown(ImGuiMouseButton_Left) &&
                               !ImGui::GetIO().KeyAlt;

    if (!sliceMode) {
        const Water3DPanelBox box = makeWater3DPanelBox(
            water3D.nx,
            water3D.ny,
            water3D.nz,
            std::max(1e-6f, (p1.x - p0.x) / std::max(1e-6f, p1.y - p0.y)));

        ImDrawList* dl = ImGui::GetWindowDrawList();
        dl->PushClipRect(p0, p1, true);

        const Water3DViewVec3 corners[8] = {
            Water3DViewVec3{-box.hx, -box.hy, -box.hz},
            Water3DViewVec3{ box.hx, -box.hy, -box.hz},
            Water3DViewVec3{ box.hx,  box.hy, -box.hz},
            Water3DViewVec3{-box.hx,  box.hy, -box.hz},
            Water3DViewVec3{-box.hx, -box.hy,  box.hz},
            Water3DViewVec3{ box.hx, -box.hy,  box.hz},
            Water3DViewVec3{ box.hx,  box.hy,  box.hz},
            Water3DViewVec3{-box.hx,  box.hy,  box.hz}
        };
        const int edges[12][2] = {
            {0,1},{1,2},{2,3},{3,0},
            {4,5},{5,6},{6,7},{7,4},
            {0,4},{1,5},{2,6},{3,7}
        };
        ImVec2 proj[8];
        for (int i = 0; i < 8; ++i) {
            proj[i] = projectWater3DPanelPoint(corners[i], box,
                                               ui.water3DViewYawDeg,
                                               ui.water3DViewPitchDeg,
                                               ui.water3DViewZoom,
                                               p0, p1, nullptr);
        }
        const ImU32 edgeCol = IsDarkTheme() ? IM_COL32(242, 244, 248, 208)
                                            : IM_COL32(92, 102, 114, 185);
        for (int e = 0; e < 12; ++e) {
            dl->AddLine(proj[edges[e][0]], proj[edges[e][1]], edgeCol, 1.35f);
        }

        if (ui.showWaterParticles && !water3D.particles.empty()) {
            const size_t maxDraw = 25000;
            const size_t n = water3D.particles.size();
            const size_t stride = std::max<size_t>(1, n / maxDraw);
            for (size_t k = 0; k < n; k += stride) {
                const auto& p = water3D.particles[k];
                const float ux = p.x / domainX;
                const float uy = p.y / domainY;
                const float uz = p.z / domainZ;
                if (ux < 0.0f || ux > 1.0f || uy < 0.0f || uy > 1.0f || uz < 0.0f || uz > 1.0f) continue;

                float depth = 0.0f;
                const ImVec2 sp = projectWater3DPanelPoint(
                    unitToWater3DPanelLocal(ux, uy, uz, box),
                    box,
                    ui.water3DViewYawDeg,
                    ui.water3DViewPitchDeg,
                    ui.water3DViewZoom,
                    p0, p1,
                    &depth);
                if (depth <= 0.0f) continue;
                if (sp.x < p0.x || sp.x > p1.x || sp.y < p0.y || sp.y > p1.y) continue;

                const float radiusPx = std::clamp(3.0f / (0.35f + depth), 1.2f, 3.0f);
                const float alpha = std::clamp(255.0f * (1.2f / (0.4f + depth)), 60.0f, 210.0f);
                dl->AddCircleFilled(sp, radiusPx, IM_COL32(255, 245, 120, (int)alpha), 8);
            }
        }

        dl->PopClipRect();

        if (canPaintVolume) {
            Water3DViewVec3 rayOrigin{};
            Water3DViewVec3 rayDir{};
            if (makeWater3DPanelRay(ImGui::GetMousePos(), p0, p1, box,
                                    ui.water3DViewYawDeg,
                                    ui.water3DViewPitchDeg,
                                    ui.water3DViewZoom,
                                    rayOrigin, rayDir)) {
                float tmin = 0.0f;
                float tmax = 0.0f;
                if (intersectWater3DPanelBox(rayOrigin, rayDir, box, tmin, tmax)) {
                    const float depth01 = std::clamp(ui.water3DSourceDepth, 0.0f, 1.0f);
                    const float t = std::max(0.0f, tmin) + depth01 * std::max(0.0f, tmax - std::max(0.0f, tmin));
                    const Water3DViewVec3 hitLocal = rayOrigin + rayDir * t;
                    const Water3DViewVec3 hitUnit = water3DPanelLocalToUnit(hitLocal, box);

                    MACWater3D::Vec3 center{};
                    center.x = std::clamp(hitUnit.x, 0.0f, 1.0f) * domainX;
                    center.y = std::clamp(hitUnit.y, 0.0f, 1.0f) * domainY;
                    center.z = std::clamp(hitUnit.z, 0.0f, 1.0f) * domainZ;

                    const float sourceScale = std::min({domainX, domainY, domainZ});
                    const float radius = std::max(water3D.dx, ui.brushRadius * sourceScale);
                    const MACWater3D::Vec3 velocity{ui.water3DSourceVelX, ui.water3DSourceVelY, ui.water3DSourceVelZ};
                    water3D.addWaterSourceSphere(center, radius, velocity);
                }
            }
        }
    } else {
        if (ui.showWaterParticles && !water3D.particles.empty()) {
            ImDrawList* dl = ImGui::GetWindowDrawList();
            dl->PushClipRect(p0, p1, true);
            const float w = p1.x - p0.x;
            const float h = p1.y - p0.y;
            const int axis = std::clamp(ui.water3DSliceAxis, 0, 2);
            const int maxSlice = (axis == 0)
                ? std::max(0, water3D.nz - 1)
                : (axis == 1)
                    ? std::max(0, water3D.ny - 1)
                    : std::max(0, water3D.nx - 1);
            const int sliceIndex = std::clamp(ui.water3DSliceIndex, 0, maxSlice);
            const float sliceCoord = (sliceIndex + 0.5f) * water3D.dx;
            const float sliceHalfThickness = 0.75f * water3D.dx;
            const size_t maxDraw = 20000;
            const size_t n = water3D.particles.size();
            const size_t stride = std::max<size_t>(1, n / maxDraw);
            const float radiusPx = std::max(1.5f, 0.30f * ui.viewScale);

            for (size_t k = 0; k < n; k += stride) {
                const auto& p = water3D.particles[k];
                float px = 0.0f;
                float py = 0.0f;
                float sliceDist = 0.0f;
                if (axis == 0) {
                    px = p.x / domainX;
                    py = p.y / domainY;
                    sliceDist = std::fabs(p.z - sliceCoord);
                } else if (axis == 1) {
                    px = p.x / domainX;
                    py = p.z / domainZ;
                    sliceDist = std::fabs(p.y - sliceCoord);
                } else {
                    px = p.y / domainY;
                    py = p.z / domainZ;
                    sliceDist = std::fabs(p.x - sliceCoord);
                }
                if (sliceDist > sliceHalfThickness) continue;
                if (px < 0.0f || px > 1.0f || py < 0.0f || py > 1.0f) continue;

                const float sx = p0.x + px * w;
                const float sy = p1.y - py * h;
                const float alpha = std::clamp(255.0f * (1.0f - sliceDist / std::max(1e-6f, sliceHalfThickness)), 40.0f, 220.0f);
                dl->AddCircleFilled(ImVec2(sx, sy), radiusPx, IM_COL32(255, 245, 120, (int)alpha), 8);
            }
            dl->PopClipRect();
        }

        if (canPaintSlice) {
            const ImVec2 m = ImGui::GetMousePos();
            const float u = (m.x - p0.x) / std::max(1e-6f, (p1.x - p0.x));
            const float v = (m.y - p0.y) / std::max(1e-6f, (p1.y - p0.y));
            const float nu = std::clamp(u, 0.0f, 1.0f);
            const float nv = std::clamp(1.0f - v, 0.0f, 1.0f);

            const int axis = std::clamp(ui.water3DSliceAxis, 0, 2);
            const int maxSlice = (axis == 0)
                ? std::max(0, water3D.nz - 1)
                : (axis == 1)
                    ? std::max(0, water3D.ny - 1)
                    : std::max(0, water3D.nx - 1);
            const int sliceIndex = std::clamp(ui.water3DSliceIndex, 0, maxSlice);
            const float sliceCoord = (sliceIndex + 0.5f) * water3D.dx;

            MACWater3D::Vec3 center{};
            if (axis == 0) {
                center.x = nu * domainX;
                center.y = nv * domainY;
                center.z = sliceCoord;
            } else if (axis == 1) {
                center.x = nu * domainX;
                center.y = sliceCoord;
                center.z = nv * domainZ;
            } else {
                center.x = sliceCoord;
                center.y = nu * domainY;
                center.z = nv * domainZ;
            }

            const float sourceScale = std::min({domainX, domainY, domainZ});
            const float radius = std::max(water3D.dx, ui.brushRadius * sourceScale);
            const MACWater3D::Vec3 velocity{ui.water3DSourceVelX, ui.water3DSourceVelY, ui.water3DSourceVelZ};
            water3D.addWaterSourceSphere(center, radius, velocity);
        }
    }

    if (ui.showViewportHints) {
        if (sliceMode) {
            ImGui::TextDisabled("Slice paint: LMB inject | switch slice axis/index in Inspector.");
        } else {
            ImGui::TextDisabled("MMB/RMB/Alt+LMB orbit | wheel or Alt+RMB zoom | double-click MMB reset | LMB inject.");
        }
    }

    ImGui::End();
}

// Combined view now shows the real coupled sim (smoke + water)
static void drawCombinedView(MACCoupledSim& coupled,
                             SmokeRenderer& coupledRenderer,
                             ImGuiID dock_id,
                             Settings& ui,
                             int NX, int NY)
{
    if (!ui.showCombinedView) return;

    ImGui::SetNextWindowDockID(dock_id, ImGuiCond_FirstUseEver);
    ImGui::Begin(kWinCombined);

    DrawViewportHeader(ui, "Coupled", "Smoke + water composite workspace", g_pipeMode ? "PIPE" : "LIVE");
    const float scale = ui.viewScale;
    const ImVec2 size(NX * scale, NY * scale);
    const ImVec2 imagePos = ImGui::GetCursorScreenPos();
    DrawViewportCanvasBackdrop(imagePos, size);

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

    const bool hovered = ImGui::IsItemHovered();

    // Pipe polyline overlay
    if (coupled.pipe.x.size() >= 2) {
        for (size_t k = 0; k + 1 < coupled.pipe.x.size(); ++k) {
            ImVec2 a(p0.x + coupled.pipe.x[k]   * (p1.x - p0.x),
                    p0.y + (1.0f - coupled.pipe.y[k])   * (p1.y - p0.y));
            ImVec2 b(p0.x + coupled.pipe.x[k+1] * (p1.x - p0.x),
                    p0.y + (1.0f - coupled.pipe.y[k+1]) * (p1.y - p0.y));
            dl->AddLine(a, b, IM_COL32(225, 64, 72, 220), 2.0f);
        }
    }

    // PIPE EDIT
    if (hovered && g_pipeMode && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        ImVec2 m = ImGui::GetMousePos();
        float u = (m.x - p0.x) / (p1.x - p0.x);
        float v = (m.y - p0.y) / (p1.y - p0.y);
        float sx = u;
        float sy = 1.0f - v;
        if (sx >= 0 && sx <= 1 && sy >= 0 && sy <= 1) {
            coupled.pipe.x.push_back(sx);
            coupled.pipe.y.push_back(sy);
            coupled.rebuildSolidsFromPipe(false);
            coupled.invalidatePressureMatrix();
            coupled.enforceBoundaries();
        }
    }

    // SOLID PAINT/ERASE (Combined)
    if (!g_pipeMode) {
        if (hovered && ui.paintSolid && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            ImVec2 m = ImGui::GetMousePos();
            float u = (m.x - p0.x) / (p1.x - p0.x);
            float v = (m.y - p0.y) / (p1.y - p0.y);
            float sx = u;
            float sy = 1.0f - v;

            if (sx >= 0 && sx <= 1 && sy >= 0 && sy <= 1) {
                if (!ui.eraseSolid) {
                    if (ui.circleMode) {
                        coupled.addSolidCircle(sx, sy, ui.brushRadius);
                    } else {
                        float hs = ui.rectHalfSize;
                        for (float yy = sy-hs; yy <= sy+hs; yy += coupled.dx) {
                            for (float xx = sx-hs; xx <= sx+hs; xx += coupled.dx) {
                                coupled.addSolidCircle(xx, yy, coupled.dx*0.75f);
                            }
                        }
                    }
                } else {
                    coupled.eraseSolidCircle(sx, sy, ui.brushRadius);

                    // also clear coupled scalars in the erased region
                    int ci = (int)(sx * coupled.nx);
                    int cj = (int)(sy * coupled.ny);
                    int rad = std::max(1, (int)(ui.brushRadius / coupled.dx));
                    for (int y = cj-rad; y <= cj+rad; ++y) {
                        for (int x = ci-rad; x <= ci+rad; ++x) {
                            if (x < 0 || x >= coupled.nx || y < 0 || y >= coupled.ny) continue;
                            float dx = (x + 0.5f) / coupled.nx - sx;
                            float dy = (y + 0.5f) / coupled.ny - sy;
                            if (dx*dx + dy*dy <= ui.brushRadius*ui.brushRadius) {
                                const int id = coupled.idxP(x,y);
                                coupled.smoke[(size_t)id] = 0.0f;
                                coupled.temp [(size_t)id] = 0.0f;
                                coupled.age  [(size_t)id] = 0.0f;
                            }
                        }
                    }
                }

                coupled.enforceBoundaries();
            }
        }
    }

    // WATER PAINT (Combined)
    if (hovered && ui.paintWater && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        ImVec2 m = ImGui::GetMousePos();
        float u = (m.x - p0.x) / (p1.x - p0.x);
        float v = (m.y - p0.y) / (p1.y - p0.y);

        const float domainX = coupled.nx * coupled.dx;
        const float domainY = coupled.ny * coupled.dx;
        const float scaleW  = std::min(domainX, domainY);

        float sx = u * domainX;
        float sy = (1.0f - v) * domainY;
        float radius  = ui.brushRadius * scaleW;
        float rectHalf = ui.rectHalfSize * scaleW;

        if (sx >= 0.0f && sx <= domainX && sy >= 0.0f && sy <= domainY) {
            if (ui.circleMode) {
                coupled.addWaterSource(sx, sy, radius, ui.waterAmount);
            } else {
                float hs = rectHalf;
                for (float yy = sy - hs; yy <= sy + hs; yy += coupled.dx) {
                    for (float xx = sx - hs; xx <= sx + hs; xx += coupled.dx) {
                        coupled.addWaterSource(xx, yy, coupled.dx*0.75f, ui.waterAmount);
                    }
                }
            }
        }
    }

    if (ui.showViewportHints) {
        ImGui::TextDisabled(g_pipeMode
            ? "Pipe edit: LMB place points on the coupled viewport."
            : "LMB paint solids or water depending on active tool. Use Inspector to switch modes.");
    }

    ImGui::End();
}

Actions DrawAll(MAC2D& sim,
                MACWater& water,
                MACWater3D& water3D,
                MACSmoke3D& smoke3D,
                MACCoupledSim& coupled,
                SmokeRenderer& renderer,
                SmokeRenderer& water3DRenderer,
                SmokeRenderer& smoke3DRenderer,
                SmokeRenderer& coupledRenderer,
                Settings& ui,
                Probe& probe,
                int NX, int NY)
{
    // Dockspace root must be drawn BEFORE your windows
    BeginDockspaceRoot(ui);

    ui.activeWorkspace = std::clamp(ui.activeWorkspace, (int)kWorkspaceSmoke2D, (int)kWorkspaceCoupled);
    ui.showWaterView = (ui.activeWorkspace == kWorkspaceWater2D);
    ui.showSmoke3DView = (ui.activeWorkspace == kWorkspaceSmoke3D);
    ui.showWater3DView = (ui.activeWorkspace == kWorkspaceWater3D);
    ui.showCombinedView = (ui.activeWorkspace == kWorkspaceCoupled);
    ui.useSmoke3D = (ui.activeWorkspace == kWorkspaceSmoke3D);
    ui.useWater3D = (ui.activeWorkspace == kWorkspaceWater3D);

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

    Actions a;
    DrawViziorHub(sim, water, water3D, smoke3D, ui, a);
    {
        Actions inspectorActions = drawControls(sim, water, water3D, smoke3D, coupled, ui);
        a.resetRequested = a.resetRequested || inspectorActions.resetRequested;
        a.resetSmoke3DRequested = inspectorActions.resetSmoke3DRequested;
        a.applySmoke3DGridRequested = inspectorActions.applySmoke3DGridRequested;
        a.resetWater3DRequested = inspectorActions.resetWater3DRequested;
        a.applyWater3DGridRequested = inspectorActions.applyWater3DGridRequested;
    }

    ui.activeWorkspace = std::clamp(ui.activeWorkspace, (int)kWorkspaceSmoke2D, (int)kWorkspaceCoupled);
    ui.showWaterView = (ui.activeWorkspace == kWorkspaceWater2D);
    ui.showSmoke3DView = (ui.activeWorkspace == kWorkspaceSmoke3D);
    ui.showWater3DView = (ui.activeWorkspace == kWorkspaceWater3D);
    ui.showCombinedView = (ui.activeWorkspace == kWorkspaceCoupled);
    ui.useSmoke3D = (ui.activeWorkspace == kWorkspaceSmoke3D);
    ui.useWater3D = (ui.activeWorkspace == kWorkspaceWater3D);

    drawDebugTabs(sim, water, water3D, smoke3D, ui, probe);
    switch (std::clamp(ui.activeWorkspace, (int)kWorkspaceSmoke2D, (int)kWorkspaceCoupled)) {
        case kWorkspaceSmoke2D:
            drawSmokeViewAndInteract(sim, renderer, ui, probe, NX, NY);
            break;
        case kWorkspaceWater2D:
            drawWaterViewAndInteract(water, renderer, ui);
            break;
        case kWorkspaceSmoke3D:
            drawSmoke3DViewAndInteract(smoke3D, smoke3DRenderer, ui);
            break;
        case kWorkspaceWater3D:
            drawWater3DViewAndInteract(water3D, water3DRenderer, ui);
            break;
        case kWorkspaceCoupled:
            drawCombinedView(coupled, coupledRenderer, dock_id, ui, NX, NY);
            break;
        default:
            drawSmokeViewAndInteract(sim, renderer, ui, probe, NX, NY);
            break;
    }

    // keep solids consistent between smoke and water sims
    water.syncSolidsFrom(sim);

    return a;
}

} // namespace UI
