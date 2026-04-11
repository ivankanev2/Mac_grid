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
    pressureMGVCycles = 20;
    pressureMGCoarseIters = 200;
    pressureMGRelativeTol = 1.0e-4f;
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
    const float effectiveGravity = waterHeld ? 0.0f : waterGravity;
    if (effectiveGravity != 0.0f) {
        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const int id = idxV(i, j);

                const bool botLiquid = (j - 1 >= 0) ? (liquid[(size_t)idxP(i, j - 1)] != 0) : false;
                const bool topLiquid = (j < ny)     ? (liquid[(size_t)idxP(i, j)]     != 0) : false;

                if (botLiquid || topLiquid) {
                    v[(size_t)id] += dt * effectiveGravity;
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

    reseedParticles();

    relaxParticles(2, 0.5f);

    // --- Optional dissipation (removes particles over time) ---
    applyDissipation();

    // --- Rasterize for rendering ---
    rasterizeWaterField();
}

void MACWater::addWaterTextParticles(const std::vector<uint8_t>& textMask, int maskW, int maskH, int ppc) {
    rebuildSolidsFromUser();

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int mi = (int)((float)i / (float)nx * (float)maskW);
            int mj = (int)((float)j / (float)ny * (float)maskH);
            mi = std::min(mi, maskW - 1);
            mj = std::min(mj, maskH - 1);

            if (!textMask[(size_t)(mi + maskW * mj)]) continue;

            liquid[(size_t)idxP(i, j)] = 1;

            for (int k = 0; k < ppc; ++k) {
                Particle p;
                p.x = (i + water_internal::randRange(0.1f, 0.9f)) * dx;
                p.y = (j + water_internal::randRange(0.1f, 0.9f)) * dx;
                p.u = 0.0f;
                p.v = 0.0f;
                p.age = 0.0f;
                p.c00 = p.c01 = p.c10 = p.c11 = 0.0f;
                particles.push_back(p);
            }
        }
    }

    desiredMass = (float)particles.size();
    enforceParticleBounds();
    removeParticlesInSolids();
    rasterizeWaterField();

    std::fprintf(stderr, "[MACWater] addWaterTextParticles: %d particles spawned\n", (int)particles.size());
}

void MACWater::removeWaterTextParticles(const std::vector<uint8_t>& textMask, int maskW, int maskH) {
    auto inTextMask = [&](float x, float y) -> bool {
        int i = std::clamp((int)(x / dx), 0, nx - 1);
        int j = std::clamp((int)(y / dx), 0, ny - 1);

        int mi = (int)((float)i / (float)nx * (float)maskW);
        int mj = (int)((float)j / (float)ny * (float)maskH);
        mi = std::min(mi, maskW - 1);
        mj = std::min(mj, maskH - 1);

        return textMask[(size_t)(mi + maskW * mj)] != 0;
    };

    particles.erase(
        std::remove_if(
            particles.begin(),
            particles.end(),
            [&](const Particle& p) { return inTextMask(p.x, p.y); }),
        particles.end());

    desiredMass = (float)particles.size();

    enforceParticleBounds();
    removeParticlesInSolids();

    {
        const int savedDil = maskDilations;
        maskDilations = 0;
        buildLiquidMask();
        maskDilations = savedDil;
    }

    rasterizeWaterField();
}

void MACWater::addSolidText(const std::vector<uint8_t>& textMask, int maskW, int maskH) {
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int mi = (int)((float)i / (float)nx * (float)maskW);
            int mj = (int)((float)j / (float)ny * (float)maskH);
            mi = std::min(mi, maskW - 1);
            mj = std::min(mj, maskH - 1);

            if (textMask[(size_t)(mi + maskW * mj)]) {
                solidUser[(size_t)idxP(i, j)] = 1;
            }
        }
    }

    rebuildSolidsFromUser();
    removeParticlesInSolids();
    syncSolidsToFluidAndFaces();
    invalidatePressureMatrix();
}

void MACWater::removeSolidText(const std::vector<uint8_t>& textMask, int maskW, int maskH) {
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int mi = (int)((float)i / (float)nx * (float)maskW);
            int mj = (int)((float)j / (float)ny * (float)maskH);
            mi = std::min(mi, maskW - 1);
            mj = std::min(mj, maskH - 1);

            if (textMask[(size_t)(mi + maskW * mj)]) {
                solidUser[(size_t)idxP(i, j)] = 0;
            }
        }
    }

    rebuildSolidsFromUser();
    syncSolidsToFluidAndFaces();
    invalidatePressureMatrix();
}