#pragma once

#include <cmath>  // IMPORTANT: mac_grid_core.h uses std::isfinite but doesn't include <cmath>.

#include "mac_grid_core.h"

#include <cstdint>
#include <vector>

// -----------------------------------------------------------------------------
// Refactored MAC water simulation (PIC/FLIP particles + liquid-only projection).
//
// Notes:
//  - Keep the public API relied on by UI/renderer/main.
//  - Only water code lives here; no changes required elsewhere.
//  - Pressure solve uses a simple PCG (stable, deterministic, debuggable).
// -----------------------------------------------------------------------------
struct MACWater : public MACGridCore {
    struct Particle {
        float x = 0.0f;
        float y = 0.0f;
        float u = 0.0f;
        float v = 0.0f;
        float age = 0.0f;
    };

    // --- Public simulation state (used for rendering/debug) ---
    std::vector<float>   water;     // cell-centered "water amount" for rendering (nx*ny)
    std::vector<uint8_t> liquid;    // cell-centered liquid mask used by the solver (nx*ny)
    std::vector<Particle> particles;

    // --- UI controlled parameters ---
    float waterDissipation = 1.0f;  // 1 = no dissipation, <1 removes particles over time
    float waterGravity     = -9.8f; // negative pulls down
    float velDamping       = 4.0f;  // exponential damping rate
    bool  openTop          = true;  // if true, top boundary is open (pressure ~ 0 to air)

    // --- Simulation controls ---
    int   particlesPerCell   = 0;       // visualization/initial sampling density
    float flipBlend          = 0.1f;    // 0=PIC, 1=FLIP
    int   borderThickness    = 2;       // solid border thickness (cells)
    int   maxParticles       = 0;  // safety cap

    int   maskDilations      = 1;       // expand liquid mask by N 4-neighborhood dilations
    int   extrapolationIters = 10;      // fill velocities into air for stable sampling

    int   pressureMaxIters   = 200;
    float pressureTol        = 1e-6f;   // residual infinity-norm tolerance

    // Used only for UI debug display in this project.
    float targetMass         = 0.0f;

    MACWater(int NX, int NY, float DX, float DT);

    void reset();
    void step();

    void addWaterSource(float cx, float cy, float radius, float amount);
    void applyBoundary();

    // Copy solid cells from another sim (smoke), then re-apply water borders.
    void syncSolidsFrom(const MACGridCore& src);

    const std::vector<float>& waterField() const { return water; }

    float maxParticleSpeed() const;

private:
    // External solids (from user painting in the smoke sim). We rebuild `solid`
    // each step as: solid = solidUser OR borderSolids.
    std::vector<uint8_t> solidUser;

    // Particle->grid weights
    std::vector<float> uWeight, vWeight;

    // FLIP buffers
    std::vector<float> uPrev, vPrev;
    std::vector<float> uDelta, vDelta;

    // Extrapolation masks
    std::vector<uint8_t> validU, validV;

    // PCG pressure solve buffers (cell-centered)
    std::vector<float> diagInv;
    std::vector<float> pcg_r, pcg_z, pcg_d, pcg_q, pcg_Ap;

    int stepCounter = 0;

    // --- internal helpers ---
    void rebuildSolidsFromUser();
    void removeParticlesInSolids();
    void enforceParticleBounds();

    void particleToGrid();
    void buildLiquidMask();
    void projectLiquid();
    void extrapolateVelocity();

    void gridToParticles();
    void advectParticles();
    void applyDissipation();
    void rasterizeWaterField();
};

#include "Water/water_particles.h"
#include "Water/water_rasterize.h"
