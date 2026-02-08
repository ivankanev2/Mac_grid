#pragma once

#include <cmath>  // IMPORTANT: mac_grid_core.h uses std::isfinite but doesn't include <cmath>.

#include "mac_grid_core.h"
// pressure_solver.h is already included via mac_grid_core.h

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
        // APIC affine velocity matrix (2x2)
        float c00 = 0.0f, c01 = 0.0f;
        float c10 = 0.0f, c11 = 0.0f;
    };

    // --- Public simulation state (used for rendering/debug) ---
    std::vector<float>   water;     // cell-centered "water amount" for rendering (nx*ny)
    std::vector<uint8_t> liquid;    // cell-centered liquid mask used by the solver (nx*ny)
    std::vector<Particle> particles;

    // --- UI controlled parameters ---
    float waterDissipation = 1.0f;  // 1 = no dissipation, <1 removes particles over time
    float waterGravity     = -9.8f; // negative pulls down
    float velDamping       = 0.0f;  // exponential damping rate
    bool  openTop          = true;  // if true, top boundary is open (pressure ~ 0 to air)

    // viscosity (implicit diffusion)
    float viscosity    = 5e-4f;   // [m^2/s], 0 disables
    int   diffuseIters = 20;     // 10–40 typical
    float diffuseOmega = 0.8f;   // 0.6–0.9 typical (weighted Jacobi)

    // --- Simulation controls ---
    int   particlesPerCell   = 0;       // visualization/initial sampling density
    float flipBlend          = 0.1f;    // 0=PIC, 1=FLIP
    bool  useAPIC            = true;   // enable APIC transfers (PIC w/ affine)
    int   borderThickness    = 2;       // solid border thickness (cells)
    int   maxParticles       = 0;  // safety cap

    int   maskDilations      = 0;       // # IT NEEDS REALLY GENTLE TOUCH expand liquid mask by N 4-neighborhood dilations
    int   extrapolationIters = 10;      // fill velocities into air for stable sampling

    int   pressureMaxIters   = 200;
    float pressureTol        = 1e-10f;   // residual infinity-norm tolerance

    // Used only for UI debug display in this project.
    float targetMass         = 0.0f;

    // --- Volume preservation (free-surface drift fix) ---
    // When true, we remove the mean of the pressure RHS over LIQUID cells.
    // This biases the projection so the net divergence in the liquid region is ~0,
    // which reduces slow volume gain/loss for open free-surface liquid.
    bool  volumePreserveRhsMean  = true;
    float volumePreserveStrength = 0.05f; // 0..1 (1 = full mean removal)

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

    // // Unified pressure solver (shared implementation with Smoke later)
    // PressureSolver pressureSolver;

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
    void diffuseVelocityImplicit();

    void reseedParticles();

    void relaxParticles(int iters, float strength);

};

#include "Water/water_particles.h"
#include "Water/water_rasterize.h"
