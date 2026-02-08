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
        const bool apic = useAPIC;

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

            const float x0 = i0 * dx;
            const float x1 = i1 * dx;
            const float y0 = (j0 + 0.5f) * dx;
            const float y1 = (j1 + 0.5f) * dx;

            float u00 = p.u;
            float u10 = p.u;
            float u01 = p.u;
            float u11 = p.u;
            if (apic) {
                u00 += p.c00 * (x0 - p.x) + p.c01 * (y0 - p.y);
                u10 += p.c00 * (x1 - p.x) + p.c01 * (y0 - p.y);
                u01 += p.c00 * (x0 - p.x) + p.c01 * (y1 - p.y);
                u11 += p.c00 * (x1 - p.x) + p.c01 * (y1 - p.y);
            }

            u[(size_t)id00] += w00 * u00; uWeight[(size_t)id00] += w00;
            u[(size_t)id10] += w10 * u10; uWeight[(size_t)id10] += w10;
            u[(size_t)id01] += w01 * u01; uWeight[(size_t)id01] += w01;
            u[(size_t)id11] += w11 * u11; uWeight[(size_t)id11] += w11;
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

            const float x0 = (i0 + 0.5f) * dx;
            const float x1 = (i1 + 0.5f) * dx;
            const float y0 = j0 * dx;
            const float y1 = j1 * dx;

            float v00 = p.v;
            float v10 = p.v;
            float v01 = p.v;
            float v11 = p.v;
            if (apic) {
                v00 += p.c10 * (x0 - p.x) + p.c11 * (y0 - p.y);
                v10 += p.c10 * (x1 - p.x) + p.c11 * (y0 - p.y);
                v01 += p.c10 * (x0 - p.x) + p.c11 * (y1 - p.y);
                v11 += p.c10 * (x1 - p.x) + p.c11 * (y1 - p.y);
            }

            v[(size_t)id00] += w00 * v00; vWeight[(size_t)id00] += w00;
            v[(size_t)id10] += w10 * v10; vWeight[(size_t)id10] += w10;
            v[(size_t)id01] += w01 * v01; vWeight[(size_t)id01] += w01;
            v[(size_t)id11] += w11 * v11; vWeight[(size_t)id11] += w11;
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
        int i = (int)std::floor(p.x / dx);
        int j = (int)std::floor(p.y / dx);
        i = water_internal::clampi(i, 0, nx - 1);
        j = water_internal::clampi(j, 0, ny - 1);
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
    const bool apic = useAPIC;
    const float blend = apic ? 0.0f : water_internal::clampf(flipBlend, 0.0f, 1.0f);
    const float picW = 1.0f - blend;
    const float invDx2 = (dx > 0.0f) ? (1.0f / (dx * dx)) : 0.0f;
    const float apicScale = 3.0f * invDx2;

    for (Particle& p : particles) {
        float picU, picV;
        velAt(p.x, p.y, u, v, picU, picV);

        if (!apic && blend > 0.0f) {
            const float du = sampleU(uDelta, p.x, p.y);
            const float dv = sampleV(vDelta, p.x, p.y);
            const float flipU = p.u + du;
            const float flipV = p.v + dv;
            p.u = picW * picU + blend * flipU;
            p.v = picW * picV + blend * flipV;
        } else {
            // APIC-only (or PIC-only when blend==0)
            p.u = picU;
            p.v = picV;
        }

        if (!apic) {
            p.c00 = p.c01 = p.c10 = p.c11 = 0.0f;
            continue;
        }

        // --- APIC affine matrix from grid velocities ---
        // U component (faces at x=i*dx, y=(j+0.5)*dx)
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

            const float x0 = i0 * dx;
            const float x1 = i1 * dx;
            const float y0 = (j0 + 0.5f) * dx;
            const float y1 = (j1 + 0.5f) * dx;

            float sumUdx = 0.0f;
            float sumUdy = 0.0f;

            sumUdx += w00 * u[(size_t)idxU(i0, j0)] * (x0 - p.x);
            sumUdy += w00 * u[(size_t)idxU(i0, j0)] * (y0 - p.y);

            sumUdx += w10 * u[(size_t)idxU(i1, j0)] * (x1 - p.x);
            sumUdy += w10 * u[(size_t)idxU(i1, j0)] * (y0 - p.y);

            sumUdx += w01 * u[(size_t)idxU(i0, j1)] * (x0 - p.x);
            sumUdy += w01 * u[(size_t)idxU(i0, j1)] * (y1 - p.y);

            sumUdx += w11 * u[(size_t)idxU(i1, j1)] * (x1 - p.x);
            sumUdy += w11 * u[(size_t)idxU(i1, j1)] * (y1 - p.y);

            p.c00 = apicScale * sumUdx;
            p.c01 = apicScale * sumUdy;
        }

        // V component (faces at x=(i+0.5)*dx, y=j*dx)
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

            const float x0 = (i0 + 0.5f) * dx;
            const float x1 = (i1 + 0.5f) * dx;
            const float y0 = j0 * dx;
            const float y1 = j1 * dx;

            float sumVdx = 0.0f;
            float sumVdy = 0.0f;

            sumVdx += w00 * v[(size_t)idxV(i0, j0)] * (x0 - p.x);
            sumVdy += w00 * v[(size_t)idxV(i0, j0)] * (y0 - p.y);

            sumVdx += w10 * v[(size_t)idxV(i1, j0)] * (x1 - p.x);
            sumVdy += w10 * v[(size_t)idxV(i1, j0)] * (y0 - p.y);

            sumVdx += w01 * v[(size_t)idxV(i0, j1)] * (x0 - p.x);
            sumVdy += w01 * v[(size_t)idxV(i0, j1)] * (y1 - p.y);

            sumVdx += w11 * v[(size_t)idxV(i1, j1)] * (x1 - p.x);
            sumVdy += w11 * v[(size_t)idxV(i1, j1)] * (y1 - p.y);

            p.c10 = apicScale * sumVdx;
            p.c11 = apicScale * sumVdy;
        }
    }
}



inline void MACWater::diffuseVelocityImplicit() {
    if (viscosity <= 0.0f || diffuseIters <= 0) return;

    const float invDx2 = 1.0f / (dx * dx);
    const float alphaInvDx2 = (viscosity * dt) * invDx2; // (nu*dt)/dx^2
    if (alphaInvDx2 <= 0.0f) return;

    // Freeze RHS
    std::vector<float> bU = u;
    std::vector<float> bV = v;

    if (u0.size() != u.size()) u0.resize(u.size());
    if (v0.size() != v.size()) v0.resize(v.size());

    auto isFixedU = [&](int i, int j) {
        if (i == 0 || i == nx) return true;
        return isSolid(i - 1, j) || isSolid(i, j);
    };

    auto isFixedV = [&](int i, int j) {
        if (j == 0) return true;
        if (j == ny) return !openTop;
        return isSolid(i, j - 1) || isSolid(i, j);
    };

    auto jacobiUpdate = [&](float b, float sumN, int count) {
        return (b + alphaInvDx2 * sumN) / (1.0f + alphaInvDx2 * (float)count);
    };

    // --- diffuse U ---
    for (int it = 0; it < diffuseIters; ++it) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i <= nx; ++i) {
                int id = idxU(i, j);
                if (isFixedU(i, j)) { u0[id] = u[id]; continue; }

                float sumN = 0.0f;
                int count = 0;

                if (i - 1 >= 0)  { sumN += u[idxU(i - 1, j)]; count++; }
                if (i + 1 <= nx) { sumN += u[idxU(i + 1, j)]; count++; }
                if (j - 1 >= 0)  { sumN += u[idxU(i, j - 1)]; count++; }
                if (j + 1 < ny)  { sumN += u[idxU(i, j + 1)]; count++; }

                float xNew = jacobiUpdate(bU[id], sumN, count);
                u0[id] = (1.0f - diffuseOmega) * u[id] + diffuseOmega * xNew;
            }
        }
        u.swap(u0);
        applyBoundary();
    }

    // --- diffuse V ---
    for (int it = 0; it < diffuseIters; ++it) {
        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int id = idxV(i, j);
                if (isFixedV(i, j)) { v0[id] = v[id]; continue; }

                float sumN = 0.0f;
                int count = 0;

                if (i - 1 >= 0) { sumN += v[idxV(i - 1, j)]; count++; }
                if (i + 1 < nx) { sumN += v[idxV(i + 1, j)]; count++; }
                if (j - 1 >= 0) { sumN += v[idxV(i, j - 1)]; count++; }
                if (j + 1 <= ny) { sumN += v[idxV(i, j + 1)]; count++; }

                float xNew = jacobiUpdate(bV[id], sumN, count);
                v0[id] = (1.0f - diffuseOmega) * v[id] + diffuseOmega * xNew;
            }
        }
        v.swap(v0);
        applyBoundary();
    }
}
