#pragma once
#include <vector>
#include <cstdint>
#include "mac_grid_core.h"

struct MACWater : public MACGridCore {
    struct Particle {
        float x = 0.0f;
        float y = 0.0f;
        float u = 0.0f;
        float v = 0.0f;
        float age = 0.0f;
    };

    std::vector<float> water, water0;
    std::vector<float> waterTarget;
    std::vector<uint8_t> liquid;
    std::vector<uint8_t> liquidPrev;
    std::vector<Particle> particles;

    MACWater(int NX, int NY, float DX, float DT);
    void reset();
    void step();

    void addWaterSource(float cx, float cy, float radius, float amount);

    void applyBoundary();
    void syncSolidsFrom(const MACGridCore& src);

    float waterDissipation = 1.0f;
    float waterGravity     = -9.8f;
    float heightPressureScale = 0.0f;
    float sourceDownwardSpeed = 2.0f;
    float sourceVelBlend   = 0.85f;
    float sourceVelHold    = 0.12f;
    float velDamping       = 1.0f;
    bool  openTop          = true;
    float flipBlend        = 0.10f;
    int   particlesPerCell = 6;
    int   pressureMaxIters = 400;
    float pressureTol      = 1e-6f;
    int   pressureRepeats  = 2;
    int   extrapIters      = 12;
    int   maskDilations    = 1;
    int   borderThickness  = 2;
    int   maxParticles     = 0; // 0 = unlimited
    float liquidThreshold  = 0.01f;
    float maxWaterPerCell  = 0.0f; // disable bottom-up reconstruction by default
    float columnDiffusion  = 0.06f;
    int   columnDiffusionIters = 2;
    float targetMass       = 0.0f;
    float waterTargetDecay = 0.995f;
    float waterTargetMax   = 0.0f; // 0 = unlimited
    int   separationIters  = 10;
    float particleRadiusScale = 0.4f;
    float separationStrength  = 0.8f;
    int   densityRelaxIters   = 1;
    int   densityRelaxInterval = 3;
    float densityRelaxMaxYFrac = 0.45f;
    int   columnRelaxIters    = 1;
    float columnRelaxSlackFrac = 0.15f;
    int   columnRelaxInterval = 2;
    float columnRelaxMaxYFrac = 0.55f;

    struct PressureDiagnostics {
        float maxAbsDivBefore = 0.0f;
        float maxAbsDivAfter  = 0.0f;
        float maxAbsDivNearSolidBefore = 0.0f;
        float maxAbsDivNearSolidAfter  = 0.0f;
        float meanAbsDivBefore = 0.0f;
        float meanAbsDivAfter  = 0.0f;
        int   liquidCells = 0;
        int   liquidCellsNearSolid = 0;
    };

    // Optional per-step pressure diagnostics (printed from the water solver only).
    bool pressureDiagnostics = false;
    int  pressureDiagInterval = 1;
    const PressureDiagnostics& lastPressureDiagnostics() const { return lastPressureDiag; }

    const std::vector<float>& waterField() const { return water; }
    float maxParticleSpeed() const;

private:
    std::vector<float> uWeight, vWeight;
    std::vector<float> uPrev, vPrev;
    std::vector<float> uDelta, vDelta;

    std::vector<float> lapDiag, lapDiagInv;
    std::vector<int> lapL, lapR, lapB, lapT;
    std::vector<float> pcg_r, pcg_z, pcg_d, pcg_q, pcg_Ap;

    int stepCounter = 0;
    PressureDiagnostics lastPressureDiag;

    void advectParticles();
    void particleToGrid();
    void gridToParticles();
    void enforceBorderSolids();
    void extrapolateVelocity();
    void buildLiquidMask();
    void projectLiquid();
    void applyHeightPressureForce();
    void rasterizeWaterField();
    void reseedParticlesFromField(const std::vector<float>& targetWater);
    void separateParticles();
    void relaxParticleDensity();
    void relaxColumnDensity();
    void removeLiquidDrift();
    void removeParticlesInSolids();
    void enforceParticleBounds();
};
