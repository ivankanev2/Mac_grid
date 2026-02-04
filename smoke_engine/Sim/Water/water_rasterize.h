#pragma once
// Implementation header included by Sim/mac_water_sim.cpp
#include "water_common.h"

#include <algorithm>

inline void MACWater::rasterizeWaterField() {
    const int Nc = nx * ny;
    if ((int)water.size() != Nc) water.assign((size_t)Nc, 0.0f);

    std::fill(water.begin(), water.end(), 0.0f);

    if (particles.empty()) {
        targetMass = 0.0f;
        return;
    }

    const float invPpc = 1.0f / (float)std::max(1, particlesPerCell);

    // Deposit particle count to cell centers using bilinear weights.
    for (const Particle& p : particles) {
        const float fx = p.x / dx - 0.5f;
        const float fy = p.y / dx - 0.5f;

        int i0 = (int)std::floor(fx);
        int j0 = (int)std::floor(fy);

        i0 = water_internal::clampi(i0, 0, nx - 1);
        j0 = water_internal::clampi(j0, 0, ny - 1);

        const int i1 = std::min(i0 + 1, nx - 1);
        const int j1 = std::min(j0 + 1, ny - 1);

        const float tx = water_internal::clampf(fx - (float)i0, 0.0f, 1.0f);
        const float ty = water_internal::clampf(fy - (float)j0, 0.0f, 1.0f);

        float w00 = (1.0f - tx) * (1.0f - ty);
        float w10 = tx * (1.0f - ty);
        float w01 = (1.0f - tx) * ty;
        float w11 = tx * ty;

        // If some of the target cells are solid, renormalize the remaining weights.
        const int id00 = idxP(i0, j0);
        const int id10 = idxP(i1, j0);
        const int id01 = idxP(i0, j1);
        const int id11 = idxP(i1, j1);

        float wSum = 0.0f;
        if (!solid[(size_t)id00]) wSum += w00; else w00 = 0.0f;
        if (!solid[(size_t)id10]) wSum += w10; else w10 = 0.0f;
        if (!solid[(size_t)id01]) wSum += w01; else w01 = 0.0f;
        if (!solid[(size_t)id11]) wSum += w11; else w11 = 0.0f;

        if (wSum <= 1e-12f) continue;
        const float inv = 1.0f / wSum;
        w00 *= inv; w10 *= inv; w01 *= inv; w11 *= inv;

        water[(size_t)id00] += w00;
        water[(size_t)id10] += w10;
        water[(size_t)id01] += w01;
        water[(size_t)id11] += w11;
    }

    double sum = 0.0;
    for (int id = 0; id < Nc; ++id) {
        if (solid[(size_t)id]) {
            water[(size_t)id] = 0.0f;
            continue;
        }

        // Convert to an approximate fill fraction for rendering.
        water[(size_t)id] = water_internal::clamp01(water[(size_t)id] * invPpc);
        sum += (double)water[(size_t)id];
    }

    targetMass = (float)sum;
}

inline void MACWater::addWaterSource(float cx, float cy, float radius, float amount) {
    rebuildSolidsFromUser(); // ensure borders match the current openTop setting

    const float r = std::max(0.0f, radius);
    const float r2 = r * r;
    const float amt = std::max(0.0f, amount);

    if ((int)water.size() != nx * ny) water.assign((size_t)(nx * ny), 0.0f);

    const int iMin = water_internal::clampi((int)std::floor((cx - r) / dx - 0.5f), 0, nx - 1);
    const int iMax = water_internal::clampi((int)std::floor((cx + r) / dx - 0.5f), 0, nx - 1);
    const int jMin = water_internal::clampi((int)std::floor((cy - r) / dx - 0.5f), 0, ny - 1);
    const int jMax = water_internal::clampi((int)std::floor((cy + r) / dx - 0.5f), 0, ny - 1);

    const int spawnPerCell = std::max(1, (int)std::lround(amt * (float)std::max(1, particlesPerCell)));

    for (int j = jMin; j <= jMax; ++j) {
        for (int i = iMin; i <= iMax; ++i) {
            const float x = (i + 0.5f) * dx;
            const float y = (j + 0.5f) * dx;
            const float dx0 = x - cx;
            const float dy0 = y - cy;
            if (dx0 * dx0 + dy0 * dy0 > r2) continue;

            const int id = idxP(i, j);
            if (solid[(size_t)id]) continue;

            // Update render field immediately (useful when paused).
            water[(size_t)id] = water_internal::clamp01(water[(size_t)id] + amt);
            liquid[(size_t)id] = 1;

            // Spawn some particles in this cell.
            if (maxParticles > 0 && (int)particles.size() >= maxParticles) continue;

            const int canSpawn = (maxParticles > 0) ? std::max(0, maxParticles - (int)particles.size()) : spawnPerCell;
            const int n = std::min(spawnPerCell, canSpawn);

            for (int k = 0; k < n; ++k) {
                Particle p;
                p.x = (i + water_internal::randRange(0.05f, 0.95f)) * dx;
                p.y = (j + water_internal::randRange(0.05f, 0.95f)) * dx;
                p.u = 0.0f;
                p.v = 0.0f;
                p.age = 0.0f;
                particles.push_back(p);

                if (maxParticles > 0 && (int)particles.size() >= maxParticles) break;
            }
        }
    }

    enforceParticleBounds();
    removeParticlesInSolids();
}