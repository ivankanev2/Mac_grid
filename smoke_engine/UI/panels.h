#pragma once
#include <cstdint>

struct MAC2D;
class SmokeRenderer;

struct SmokeRenderSettings;
struct OverlaySettings;

namespace UI {

struct Settings {
    // Playback
    bool  playing = true;

    // Painting solids
    bool  paintSolid = true;   // kept for future expansion (currently always painting when mouse down)
    bool  eraseSolid = false;
    bool  circleMode = true;
    float brushRadius = 0.06f;   // in sim space (0..1)
    float rectHalfSize = 0.06f;  // half-size for rectangle

    // Debug overlays
    bool  showDivOverlay  = false;
    bool  showVelOverlay  = false;
    bool  showVortOverlay = false;

    float divScale = 8.0f;
    float divAlpha = 0.75f;

    int   velStride = 6;
    float velScale  = 0.35f;

    float vortScale = 8.0f;
    float vortAlpha = 0.75f;

    // Adaptive dt (CFL)
    float dtMax = 0.02f;
    float dtMin = 0.001f;
    float cfl   = 0.9f;

    // Vorticity confinement strength
    float vortEps = 2.0f;

    // Advector debug
    float lastAdvectL2 = 0.0f;

    // Rendering look
    bool  useColorSmoke = false;
    float smokeAlphaGamma   = 0.70f;
    float smokeAlphaScale   = 1.00f;
    float tempColorStrength = 0.75f;
    float ageGrayStrength   = 0.65f;
    float ageDarkenStrength = 0.55f;
    float coreDarkStrength  = 0.75f;

    // Dissipation
    float smokeDissipation = 1.000f;
    float tempDissipation  = 0.990f;

    // Smoke view display
    float viewScale = 5.0f;

};

struct Probe {
    bool  has = false;
    int   i = 0, j = 0;
    float smoke = 0.0f;
    float div   = 0.0f;
    float u     = 0.0f;
    float v     = 0.0f;
    float speed = 0.0f;
};

struct Actions {
    bool resetRequested = false;
};

// Build renderer settings structs from UI settings
void BuildRenderSettings(const Settings& ui,
                         SmokeRenderSettings& outSmoke,
                         OverlaySettings& outOverlay);

// Draw all panels (Controls, Debug tabs, Smoke View). Also handles paint/erase & probe.
Actions DrawAll(MAC2D& sim,
                SmokeRenderer& renderer,
                Settings& ui,
                Probe& probe,
                int NX, int NY);

// -------- layout save / reset API (namespace-scope) --------
bool ConsumeSaveLayoutRequest();   // return true once when user clicked "Save Layout"
bool ConsumeResetLayoutRequest();  // return true once when user clicked "Reset Layout"

} 