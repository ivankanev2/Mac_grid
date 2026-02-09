#pragma once
#include <cstdint>

struct MAC2D;
struct MACWater;
class SmokeRenderer;

struct SmokeRenderSettings;
struct WaterRenderSettings;
struct OverlaySettings;

namespace UI {

struct Settings {
    // Playback
    bool  playing = true;

    // Painting solids
    bool  paintSolid = true;
    bool  eraseSolid = false;
    bool  circleMode = true;
    float brushRadius = 0.06f;
    float rectHalfSize = 0.06f;

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

    // Rendering look (smoke)
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

    // Water
    bool  paintWater      = false;
    float waterAmount     = 0.20f;
    float waterDissipation= 1.000f;
    float waterGravity    = -9.8f;
    float waterAlpha      = 0.85f;
    float waterVelDamping = 0.0f;
    bool  waterOpenTop    = true;
    bool  showWaterView   = true;
    bool  showWaterParticles = true;

    // Smoke view display
    float viewScale = 5.0f;

    // bool  showCombinedView = true;
    // float combinedWaterAlpha = 0.5f;   // water overlay strength
    // bool  combinedShowParticles = true;
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

void BuildRenderSettings(const Settings& ui,
                         SmokeRenderSettings& outSmoke,
                         OverlaySettings& outOverlay);

void BuildWaterRenderSettings(const Settings& ui,
                              WaterRenderSettings& outWater);

Actions DrawAll(MAC2D& smokeSim,
                MACWater& waterSim,
                SmokeRenderer& renderer,
                Settings& ui,
                Probe& probe,
                int NX, int NY);

bool ConsumeSaveLayoutRequest();
bool ConsumeResetLayoutRequest();

}
