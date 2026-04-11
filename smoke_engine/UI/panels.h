#pragma once
#include <cstdint>

struct MAC2D;
struct MACWater;
struct MACWater3D;
struct MACSmoke3D;
class SmokeRenderer;
struct MACCoupledSim;

struct SmokeRenderSettings;
struct WaterRenderSettings;
struct OverlaySettings;

namespace UI {

enum ActiveWorkspace {
    kWorkspaceSmoke2D = 0,
    kWorkspaceWater2D = 1,
    kWorkspaceSmoke3D = 2,
    kWorkspaceWater3D = 3,
    kWorkspaceCoupled = 4,
};

const char* ActiveWorkspaceLabel(int workspace);

struct Settings {
    // Playback
    bool  playing = true;
    int   activeWorkspace = kWorkspaceSmoke2D;

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

    // 2D grid resolution (Smoke 2D / Water 2D / Coupled)
    int   sim2DNX = 256;
    int   sim2DNY = 256;
    int   windowWidth = 1480;
    int   windowHeight = 920;

    // Appearance
    int   themeMode = 0;            // 0=dark, 1=light
    float uiScale   = 0.80f;        // global UI scale applied via ScaleAllSizes + FontGlobalScale
    bool  showViewportHeaders = true;
    bool  showViewportHints   = true;

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

    // View display and volume-render resolution (display only; simulation resolution stays unchanged)
    float viewScale = 1.0f;
    float volumeRenderScale = 1.0f;
    float smoke3DViewportWidth = 0.0f;
    float smoke3DViewportHeight = 0.0f;
    float water3DViewportWidth = 0.0f;
    float water3DViewportHeight = 0.0f;

    bool  showCombinedView = true;
    float combinedWaterAlpha = 0.5f;   // water overlay strength
    bool  combinedShowParticles = true;

    // 3D smoke runtime mode and view.
    bool  useSmoke3D = false;
    bool  showSmoke3DView = true;
    bool  paintSmoke3D = true;
    int   smoke3DNX = 64;
    int   smoke3DNY = 64;
    int   smoke3DNZ = 48;
    int   smoke3DViewMode = 0;     // 0=volume, 1=slice
    float smoke3DViewYawDeg = 35.0f;
    float smoke3DViewPitchDeg = 18.0f;
    float smoke3DViewZoom = 1.15f;
    float smoke3DVolumeDensity = 1.0f;
    float smoke3DSourceDepth = 0.25f;
    int   smoke3DSliceAxis = 0;    // 0=XY, 1=XZ, 2=YZ
    int   smoke3DSliceIndex = 0;
    int   smoke3DDebugField = 0;   // 0=smoke,1=temp,2=pressure,3=divergence,4=speed
    int   smoke3DPressureIters = 120;
    int   smoke3DPressureSolverMode = 0; // 0=Multigrid, 1=RBGS, 2=Jacobi
    float smoke3DPressureOmega = 1.70f;
    float smoke3DBuoyancyScale = 1.0f;
    float smoke3DGravity = 9.81f;
    float smoke3DVelDamping = 0.5f;
    float smoke3DViscosity = 1e-4f;
    float smoke3DSmokeDiffusivity = 0.0f;
    float smoke3DTempDiffusivity = 0.0f;
    bool  smoke3DOpenTop = true;
    float smoke3DSourceAmount = 0.20f;
    float smoke3DHeatAmount = 0.50f;
    float smoke3DSourceVelX = 0.0f;
    float smoke3DSourceVelY = 2.0f;
    float smoke3DSourceVelZ = 0.0f;

    // 3D water runtime mode and slice debug view.
    bool  useWater3D = false;
    bool  showWater3DView = true;
    int   water3DNX = 64;
    int   water3DNY = 64;
    int   water3DNZ = 48;
    int   water3DViewMode = 0;     // 0=volume, 1=slice, 2=surface
    float water3DViewYawDeg = 35.0f;
    float water3DViewPitchDeg = 20.0f;
    float water3DViewZoom = 1.15f;
    float water3DVolumeDensity = 1.0f;
    float water3DSurfaceThreshold = 0.12f;
    float water3DSourceDepth = 0.5f; // used in volume/surface mode [0..1]
    int   water3DSliceAxis = 0;    // 0=XY, 1=XZ, 2=YZ
    int   water3DSliceIndex = 0;
    int   water3DDebugField = 0;   // 0=water,1=pressure,2=divergence,3=speed
    int   water3DPressureIters = 200;
    int   water3DPressureSolverMode = 0; // 0=Multigrid, 1=RBGS, 2=Jacobi
    int   water3DBackendMode = 0;        // 0=Auto, 1=CPU, 2=CUDA
    bool  water3DUseAPIC = true;
    float water3DFlipBlend = 0.10f;
    float water3DPressureOmega = 1.70f;
    bool  water3DVolumePreserve = true;
    float water3DVolumePreserveStrength = 0.05f;
    int   water3DRelaxIters = 2;
    float water3DRelaxStrength = 0.45f;
    float water3DSourceVelX = 0.0f;
    float water3DSourceVelY = 0.0f;
    float water3DSourceVelZ = 0.0f;
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
    bool resetSmoke3DRequested = false;
    bool applySmoke3DGridRequested = false;
    bool resetWater3DRequested = false;
    bool applyWater3DGridRequested = false;
    bool dropWaterTextRequested = false;
    bool applyGrid2DRequested = false;
    bool applyWindowResolutionRequested = false;
};

void BuildRenderSettings(const Settings& ui,
                         SmokeRenderSettings& outSmoke,
                         OverlaySettings& outOverlay);

void BuildWaterRenderSettings(const Settings& ui,
                              WaterRenderSettings& outWater);

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
                int NX, int NY,
                int windowWidth, int windowHeight);

bool ConsumeSaveLayoutRequest();
bool ConsumeResetLayoutRequest();

void ApplyViziorTheme(int themeMode = 0);

}
