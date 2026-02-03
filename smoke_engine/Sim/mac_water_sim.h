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
    float velDamping       = 0.0f;
    float waterViscosity   = 5e-10f; // lighter viscosity to avoid "solid block"
    int   viscosityIters   = 10;
    float viscosityOmega   = 0.8f; // weighted Jacobi relaxation
    float restVelocity     = 3.0f; // cells/sec; 0 disables rest snapping
    bool  openTop          = true;
    float flipBlend        = 0.1f;
    int   particlesPerCell = 6;
    int   pressureMaxIters = 400;
    float pressureTol      = 1e-6f;
    int   pressureRepeats  = 2;
    bool  useMGPressure    = true;
    int   pressureMGVcycles = 200;
    float pressureMGTol    = 1e-4f;
    int   pressureMGPolishIters = 40;
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
    int   separationIters  = 5;
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
    bool pressureDiagnostics = true;
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

    struct MGLevel {
        int nx = 0, ny = 0;
        float invDx2 = 0.0f;
        std::vector<uint8_t> solid;
        std::vector<uint8_t> fluid;
        std::vector<int> L, R, B, T;
        std::vector<float> diagInv;
        std::vector<float> x;
        std::vector<float> b;
        std::vector<float> Ax;
        std::vector<float> r;
    };

    std::vector<MGLevel> mgLevels;
    bool mgDirty = true;
    bool mgHasDirichlet = false;
    bool mgOpenTop = true;
    int  mgMaxLevels = 6;
    int  mgPreSmooth = 2;
    int  mgPostSmooth = 2;
    int  mgCoarseSmooth = 30;
    float mgOmega = 0.8f;

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
    void applyViscosity();
    void rasterizeWaterField();
    void reseedParticlesFromField(const std::vector<float>& targetWater);
    void separateParticles();
    void relaxParticleDensity();
    void relaxColumnDensity();
    void removeLiquidDrift();
    void snapToRest(float restVel);
    void ensureWaterMG();
    void solvePressureMGWater(int vcycles, float tol);
    void mgApplyA(int lev, const std::vector<float>& x, std::vector<float>& Ax) const;
    void mgSmoothRBGS(int lev, int iters);
    void mgComputeResidual(int lev);
    void mgRestrictResidual(int fineLev);
    void mgProlongateAndAdd(int coarseLev);
    void mgVCycle(int lev);
    void mgRemoveMeanFine();
    void removeParticlesInSolids();
    void enforceParticleBounds();
};
