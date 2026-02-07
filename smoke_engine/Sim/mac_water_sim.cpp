#include "mac_water_sim.h"
#include "Water/water_particles.h"
#include "Water/water_rasterize.h"

#include <algorithm>
#include <cmath>

// The implementation is split into small, focused headers that are INCLUDED
// into this translation unit so CMake doesn't need to be updated (it still
// builds Sim/mac_water_sim.cpp only).
//
// (You requested no .inl files, so these are .h implementation chunks.)
#include "Water/water_solids.h"
#include "Water/water_particles.h"
#include "Water/water_grid.h"
#include "Water/water_pressure.h"
#include "Water/water_rasterize.h"

MACWater::MACWater(int NX, int NY, float DX, float DT)
    : MACGridCore(NX, NY, DX, DT) {
    reset();
}

void MACWater::reset() {
    resetCore();
    stepCounter = 0;

    const int Nc = nx * ny;

    water.assign((size_t)Nc, 0.0f);
    liquid.assign((size_t)Nc, (uint8_t)0);

    // user solids are empty at reset; border solids will be applied.
    solidUser = solid;
    rebuildSolidsFromUser();

    particles.clear();
    particles.reserve((size_t)std::max(1, particlesPerCell) * (size_t)Nc);

    // transfer buffers
    uWeight.assign(u.size(), 0.0f);
    vWeight.assign(v.size(), 0.0f);

    uPrev.assign(u.size(), 0.0f);
    vPrev.assign(v.size(), 0.0f);

    uDelta.assign(u.size(), 0.0f);
    vDelta.assign(v.size(), 0.0f);

    validU.clear();
    validV.clear();

    // PCG buffers are resized lazily in projectLiquid()

    targetMass = 0.0f;

    applyBoundary();
    rasterizeWaterField();
}

void MACWater::syncSolidsFrom(const MACGridCore& src) {
    if (src.solid.size() == solid.size()) {
        solidUser = src.solid;
    } else {
        // fallback: keep current user solids if sizes mismatch (shouldn't happen)
        solidUser = solid;
    }

    rebuildSolidsFromUser();
    removeParticlesInSolids();
    enforceParticleBounds();
}

void MACWater::step() {
    ++stepCounter;

    // Keep borders consistent with current openTop/borderThickness settings.
    rebuildSolidsFromUser();

    // If solids changed since last frame, fix particles.
    removeParticlesInSolids();
    enforceParticleBounds();

    // Nothing to simulate if there are no particles.
    if (particles.empty()) {
        std::fill(u.begin(), u.end(), 0.0f);
        std::fill(v.begin(), v.end(), 0.0f);
        applyBoundary();
        rasterizeWaterField();
        return;
    }

    // --- Particle -> grid ---
    particleToGrid();

    // --- Liquid mask (cells containing particles, optionally dilated) ---
    buildLiquidMask();

    // --- External forces ---
    // Gravity to v-faces that touch any liquid cell.
    if (waterGravity != 0.0f) {
        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const int id = idxV(i, j);

                const bool botLiquid = (j - 1 >= 0) ? (liquid[(size_t)idxP(i, j - 1)] != 0) : false;
                const bool topLiquid = (j < ny)     ? (liquid[(size_t)idxP(i, j)]     != 0) : false;

                if (botLiquid || topLiquid) {
                    v[(size_t)id] += dt * waterGravity;
                }
            }
        }
    }

    // Exponential velocity damping (stable).
    if (velDamping > 0.0f) {
        const float damp = std::exp(-velDamping * dt);
        for (float& uu : u) uu *= damp;
        for (float& vv : v) vv *= damp;
    }

    applyBoundary();
    diffuseVelocityImplicit();
    applyBoundary();

    // Save for FLIP delta (BEFORE projection).
    uPrev = u;
    vPrev = v;

    // --- Incompressibility projection in the liquid region ---
    projectLiquid();
    applyBoundary();

    // FLIP delta = projected - pre-projection
    for (size_t i = 0; i < u.size(); ++i) uDelta[i] = u[i] - uPrev[i];
    for (size_t i = 0; i < v.size(); ++i) vDelta[i] = v[i] - vPrev[i];

    // --- Extrapolate velocity into air (for stable sampling near free surface) ---
    extrapolateVelocity();
    applyBoundary();

    // --- Grid -> particle (PIC/FLIP blend) ---
    gridToParticles();

    // --- Advect particles ---
    advectParticles();
    enforceParticleBounds();
    removeParticlesInSolids();

        // IMPORTANT: liquid mask must match the *current* particle positions
    {
        const int savedDil = maskDilations;
        maskDilations = 0;        // reseed should NOT use dilated mask
        buildLiquidMask();
        maskDilations = savedDil;
    }

    relaxParticles(2, 0.5f); // 2 iters, moderate strength

    relaxParticles(2, 0.5f);

    // --- Optional dissipation (removes particles over time) ---
    applyDissipation();

    // --- Rasterize for rendering ---
    rasterizeWaterField();
}
