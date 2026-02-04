#pragma once
// Implementation header included by Sim/mac_water_sim.cpp
#include "water_common.h"

#include <algorithm>

inline void MACWater::particleToGrid() {
    std::fill(u.begin(), u.end(), 0.0f);
    std::fill(v.begin(), v.end(), 0.0f);
    std::fill(uWeight.begin(), uWeight.end(), 0.0f);
    std::fill(vWeight.begin(), vWeight.end(), 0.0f);

    // Scatter particle velocities to staggered faces using bilinear weights.
    for (const Particle& p : particles) {
        // --- u faces (i in [0..nx], j in [0..ny-1]) ---
        {
            const float fx = p.x / dx;
            const float fy = p.y / dx - 0.5f;

            int i0 = (int)std::floor(fx);
            int j0 = (int)std::floor(fy);
            i0 = water_internal::clampi(i0, 0, nx - 1);
            j0 = water_internal::clampi(j0, 0, ny - 1);

            const int i1 = std::min(i0 + 1, nx);
            const int j1 = std::min(j0 + 1, ny - 1);

            const float tx = water_internal::clampf(fx - (float)i0, 0.0f, 1.0f);
            const float ty = water_internal::clampf(fy - (float)j0, 0.0f, 1.0f);

            const float w00 = (1.0f - tx) * (1.0f - ty);
            const float w10 = tx * (1.0f - ty);
            const float w01 = (1.0f - tx) * ty;
            const float w11 = tx * ty;

            const int id00 = idxU(i0, j0);
            const int id10 = idxU(i1, j0);
            const int id01 = idxU(i0, j1);
            const int id11 = idxU(i1, j1);

            u[(size_t)id00] += w00 * p.u; uWeight[(size_t)id00] += w00;
            u[(size_t)id10] += w10 * p.u; uWeight[(size_t)id10] += w10;
            u[(size_t)id01] += w01 * p.u; uWeight[(size_t)id01] += w01;
            u[(size_t)id11] += w11 * p.u; uWeight[(size_t)id11] += w11;
        }

        // --- v faces (i in [0..nx-1], j in [0..ny]) ---
        {
            const float fx = p.x / dx - 0.5f;
            const float fy = p.y / dx;

            int i0 = (int)std::floor(fx);
            int j0 = (int)std::floor(fy);
            i0 = water_internal::clampi(i0, 0, nx - 1);
            j0 = water_internal::clampi(j0, 0, ny - 1);

            const int i1 = std::min(i0 + 1, nx - 1);
            const int j1 = std::min(j0 + 1, ny);

            const float tx = water_internal::clampf(fx - (float)i0, 0.0f, 1.0f);
            const float ty = water_internal::clampf(fy - (float)j0, 0.0f, 1.0f);

            const float w00 = (1.0f - tx) * (1.0f - ty);
            const float w10 = tx * (1.0f - ty);
            const float w01 = (1.0f - tx) * ty;
            const float w11 = tx * ty;

            const int id00 = idxV(i0, j0);
            const int id10 = idxV(i1, j0);
            const int id01 = idxV(i0, j1);
            const int id11 = idxV(i1, j1);

            v[(size_t)id00] += w00 * p.v; vWeight[(size_t)id00] += w00;
            v[(size_t)id10] += w10 * p.v; vWeight[(size_t)id10] += w10;
            v[(size_t)id01] += w01 * p.v; vWeight[(size_t)id01] += w01;
            v[(size_t)id11] += w11 * p.v; vWeight[(size_t)id11] += w11;
        }
    }

    for (size_t i = 0; i < u.size(); ++i) {
        const float w = uWeight[i];
        u[i] = (w > 1e-6f) ? (u[i] / w) : 0.0f;
    }
    for (size_t i = 0; i < v.size(); ++i) {
        const float w = vWeight[i];
        v[i] = (w > 1e-6f) ? (v[i] / w) : 0.0f;
    }

    applyBoundary();
}

inline void MACWater::buildLiquidMask() {
    const int Nc = nx * ny;
    if ((int)liquid.size() != Nc) liquid.assign((size_t)Nc, (uint8_t)0);

    std::fill(liquid.begin(), liquid.end(), (uint8_t)0);

    for (const Particle& p : particles) {
        int i, j;
        worldToCell(p.x, p.y, i, j);
        const int id = idxP(i, j);
        if (solid[(size_t)id]) continue;
        liquid[(size_t)id] = 1;
    }

    const int dilations = std::max(0, maskDilations);
    for (int it = 0; it < dilations; ++it) {
        std::vector<uint8_t> next = liquid;
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const int id = idxP(i, j);
                if (solid[(size_t)id] || liquid[(size_t)id]) continue;

                const bool n =
                    (i > 0     && liquid[(size_t)idxP(i - 1, j)]) ||
                    (i < nx-1  && liquid[(size_t)idxP(i + 1, j)]) ||
                    (j > 0     && liquid[(size_t)idxP(i, j - 1)]) ||
                    (j < ny-1  && liquid[(size_t)idxP(i, j + 1)]);

                if (n) next[(size_t)id] = 1;
            }
        }
        // keep solids zero
        for (int id = 0; id < Nc; ++id) {
            if (solid[(size_t)id]) next[(size_t)id] = 0;
        }
        liquid.swap(next);
    }
}

inline void MACWater::extrapolateVelocity() {
    // Simple neighborhood extrapolation (iterate a few times).
    // Seed valid faces from particle weights.

    validU.assign(u.size(), (uint8_t)0);
    validV.assign(v.size(), (uint8_t)0);

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i <= nx; ++i) {
            const int id = idxU(i, j);
            const bool leftSolid  = (i - 1 >= 0) ? isSolid(i - 1, j) : true;
            const bool rightSolid = (i < nx)     ? isSolid(i, j)     : true;
            if (leftSolid || rightSolid) {
                u[(size_t)id] = 0.0f;
                continue;
            }
            if (uWeight[(size_t)id] > 1e-6f) validU[(size_t)id] = 1;
        }
    }

    for (int j = 0; j <= ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int id = idxV(i, j);
            const bool botSolid = (j - 1 >= 0) ? isSolid(i, j - 1) : true;
            const bool topSolid = (j < ny)     ? isSolid(i, j)     : (openTop ? false : true);
            if (botSolid || topSolid) {
                v[(size_t)id] = 0.0f;
                continue;
            }
            if (vWeight[(size_t)id] > 1e-6f) validV[(size_t)id] = 1;
        }
    }

    const int iters = std::max(0, extrapolationIters);
    for (int it = 0; it < iters; ++it) {
        // --- u ---
        {
            std::vector<float> nextU = u;
            std::vector<uint8_t> nextValid = validU;

            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i <= nx; ++i) {
                    const int id = idxU(i, j);
                    if (nextValid[(size_t)id]) continue;

                    const bool leftSolid  = (i - 1 >= 0) ? isSolid(i - 1, j) : true;
                    const bool rightSolid = (i < nx)     ? isSolid(i, j)     : true;
                    if (leftSolid || rightSolid) continue;

                    float sum = 0.0f;
                    int count = 0;

                    if (i > 0) {
                        int n = idxU(i - 1, j);
                        if (validU[(size_t)n]) { sum += u[(size_t)n]; ++count; }
                    }
                    if (i < nx) {
                        int n = idxU(i + 1, j);
                        if (validU[(size_t)n]) { sum += u[(size_t)n]; ++count; }
                    }
                    if (j > 0) {
                        int n = idxU(i, j - 1);
                        if (validU[(size_t)n]) { sum += u[(size_t)n]; ++count; }
                    }
                    if (j < ny - 1) {
                        int n = idxU(i, j + 1);
                        if (validU[(size_t)n]) { sum += u[(size_t)n]; ++count; }
                    }

                    if (count > 0) {
                        nextU[(size_t)id] = sum / (float)count;
                        nextValid[(size_t)id] = 1;
                    }
                }
            }

            u.swap(nextU);
            validU.swap(nextValid);
        }

        // --- v ---
        {
            std::vector<float> nextV = v;
            std::vector<uint8_t> nextValid = validV;

            for (int j = 0; j <= ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    const int id = idxV(i, j);
                    if (nextValid[(size_t)id]) continue;

                    const bool botSolid = (j - 1 >= 0) ? isSolid(i, j - 1) : true;
                    const bool topSolid = (j < ny)     ? isSolid(i, j)     : (openTop ? false : true);
                    if (botSolid || topSolid) continue;

                    float sum = 0.0f;
                    int count = 0;

                    if (i > 0) {
                        int n = idxV(i - 1, j);
                        if (validV[(size_t)n]) { sum += v[(size_t)n]; ++count; }
                    }
                    if (i < nx - 1) {
                        int n = idxV(i + 1, j);
                        if (validV[(size_t)n]) { sum += v[(size_t)n]; ++count; }
                    }
                    if (j > 0) {
                        int n = idxV(i, j - 1);
                        if (validV[(size_t)n]) { sum += v[(size_t)n]; ++count; }
                    }
                    if (j < ny) {
                        int n = idxV(i, j + 1);
                        if (validV[(size_t)n]) { sum += v[(size_t)n]; ++count; }
                    }

                    if (count > 0) {
                        nextV[(size_t)id] = sum / (float)count;
                        nextValid[(size_t)id] = 1;
                    }
                }
            }

            v.swap(nextV);
            validV.swap(nextValid);
        }
    }

    applyBoundary();
}

inline void MACWater::gridToParticles() {
    const float blend = water_internal::clampf(flipBlend, 0.0f, 1.0f);
    const float picW = 1.0f - blend;

    for (Particle& p : particles) {
        float picU, picV;
        velAt(p.x, p.y, u, v, picU, picV);

        const float du = sampleU(uDelta, p.x, p.y);
        const float dv = sampleV(vDelta, p.x, p.y);

        const float flipU = p.u + du;
        const float flipV = p.v + dv;

        p.u = picW * picU + blend * flipU;
        p.v = picW * picV + blend * flipV;
    }
}