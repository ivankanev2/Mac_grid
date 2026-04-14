#pragma once

#include "water3d_common.h"

#include <algorithm>
#include <cmath>

inline void MACWater3D::rasterizeWaterField() {
    const int cellCount = nx * ny * nz;
    if ((int)water.size() != cellCount) water.assign((std::size_t)cellCount, 0.0f);
    if ((int)divergence.size() != cellCount) divergence.assign((std::size_t)cellCount, 0.0f);
    if ((int)speed.size() != cellCount) speed.assign((std::size_t)cellCount, 0.0f);
    if ((int)pressureTmp.size() != cellCount) pressureTmp.assign((std::size_t)cellCount, 0.0f);
    std::fill(pressureTmp.begin(), pressureTmp.end(), 0.0f);
    std::vector<float>& mass = pressureTmp;

    for (const Particle& p : particles) {
        const float fx = p.x / dx - 0.5f;
        const float fy = p.y / dx - 0.5f;
        const float fz = p.z / dx - 0.5f;

        int i0 = water3d_internal::clampi((int)std::floor(fx), 0, nx - 1);
        int j0 = water3d_internal::clampi((int)std::floor(fy), 0, ny - 1);
        int k0 = water3d_internal::clampi((int)std::floor(fz), 0, nz - 1);

        const int i1 = std::min(i0 + 1, nx - 1);
        const int j1 = std::min(j0 + 1, ny - 1);
        const int k1 = std::min(k0 + 1, nz - 1);

        float tx = water3d_internal::clampf(fx - (float)i0, 0.0f, 1.0f);
        float ty = water3d_internal::clampf(fy - (float)j0, 0.0f, 1.0f);
        float tz = water3d_internal::clampf(fz - (float)k0, 0.0f, 1.0f);

        float w000 = (1.0f - tx) * (1.0f - ty) * (1.0f - tz);
        float w100 = tx * (1.0f - ty) * (1.0f - tz);
        float w010 = (1.0f - tx) * ty * (1.0f - tz);
        float w110 = tx * ty * (1.0f - tz);
        float w001 = (1.0f - tx) * (1.0f - ty) * tz;
        float w101 = tx * (1.0f - ty) * tz;
        float w011 = (1.0f - tx) * ty * tz;
        float w111 = tx * ty * tz;

        const int id000 = idxCell(i0, j0, k0);
        const int id100 = idxCell(i1, j0, k0);
        const int id010 = idxCell(i0, j1, k0);
        const int id110 = idxCell(i1, j1, k0);
        const int id001 = idxCell(i0, j0, k1);
        const int id101 = idxCell(i1, j0, k1);
        const int id011 = idxCell(i0, j1, k1);
        const int id111 = idxCell(i1, j1, k1);

        float wSum = 0.0f;
        if (!solid[(std::size_t)id000]) wSum += w000; else w000 = 0.0f;
        if (!solid[(std::size_t)id100]) wSum += w100; else w100 = 0.0f;
        if (!solid[(std::size_t)id010]) wSum += w010; else w010 = 0.0f;
        if (!solid[(std::size_t)id110]) wSum += w110; else w110 = 0.0f;
        if (!solid[(std::size_t)id001]) wSum += w001; else w001 = 0.0f;
        if (!solid[(std::size_t)id101]) wSum += w101; else w101 = 0.0f;
        if (!solid[(std::size_t)id011]) wSum += w011; else w011 = 0.0f;
        if (!solid[(std::size_t)id111]) wSum += w111; else w111 = 0.0f;

        if (wSum <= 1e-12f) continue;
        const float inv = 1.0f / wSum;
        mass[(std::size_t)id000] += w000 * inv;
        mass[(std::size_t)id100] += w100 * inv;
        mass[(std::size_t)id010] += w010 * inv;
        mass[(std::size_t)id110] += w110 * inv;
        mass[(std::size_t)id001] += w001 * inv;
        mass[(std::size_t)id101] += w101 * inv;
        mass[(std::size_t)id011] += w011 * inv;
        mass[(std::size_t)id111] += w111 * inv;
    }

    const float invPpc = 1.0f / (float)std::max(1, params.particlesPerCell);
    double sumMass = 0.0;

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const int id = idxCell(i, j, k);
                if (solid[(std::size_t)id]) {
                    water[(std::size_t)id] = 0.0f;
                    continue;
                }

                const float m = mass[(std::size_t)id];
                sumMass += (double)m;
                water[(std::size_t)id] = water3d_internal::clamp01(m * invPpc);
            }
        }
    }

    targetMass = (float)sumMass;
    if (desiredMass < 0.0f && targetMass > 0.0f) {
        desiredMass = targetMass;
    }

    derivedFieldsDirty = true;
}

inline void MACWater3D::ensureDerivedDebugFields() {
    if (!derivedFieldsDirty) return;

    const int cellCount = nx * ny * nz;
    if ((int)divergence.size() != cellCount) divergence.assign((std::size_t)cellCount, 0.0f);
    if ((int)speed.size() != cellCount) speed.assign((std::size_t)cellCount, 0.0f);

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const int id = idxCell(i, j, k);
                if (solid[(std::size_t)id]) {
                    divergence[(std::size_t)id] = 0.0f;
                    speed[(std::size_t)id] = 0.0f;
                    continue;
                }

                float uL = u[(std::size_t)idxU(i, j, k)];
                float uR = u[(std::size_t)idxU(i + 1, j, k)];
                float vB = v[(std::size_t)idxV(i, j, k)];
                float vT = v[(std::size_t)idxV(i, j + 1, k)];
                float wBk = w[(std::size_t)idxW(i, j, k)];
                float wFr = w[(std::size_t)idxW(i, j, k + 1)];

                if (i - 1 >= 0 && solid[(std::size_t)idxCell(i - 1, j, k)]) uL = 0.0f;
                if (i + 1 < nx && solid[(std::size_t)idxCell(i + 1, j, k)]) uR = 0.0f;
                if (j - 1 >= 0 && solid[(std::size_t)idxCell(i, j - 1, k)]) vB = 0.0f;
                if (j + 1 < ny && solid[(std::size_t)idxCell(i, j + 1, k)]) vT = 0.0f;
                if (k - 1 >= 0 && solid[(std::size_t)idxCell(i, j, k - 1)]) wBk = 0.0f;
                if (k + 1 < nz && solid[(std::size_t)idxCell(i, j, k + 1)]) wFr = 0.0f;

                divergence[(std::size_t)id] = (uR - uL + vT - vB + wFr - wBk) / dx;

                const float cx = (i + 0.5f) * dx;
                const float cy = (j + 0.5f) * dx;
                const float cz = (k + 0.5f) * dx;
                float uc, vc, wc;
                velAt(cx, cy, cz, u, v, w, uc, vc, wc);
                speed[(std::size_t)id] = std::sqrt(uc * uc + vc * vc + wc * wc);
            }
        }
    }

    derivedFieldsDirty = false;
}

inline void MACWater3D::rasterizeDebugFields() {
    rasterizeWaterField();
    ensureDerivedDebugFields();
}

inline void MACWater3D::updateStats(float stepMs) {
    lastStats.cudaEnabled = false;
    lastStats.backendReady = true;
    lastStats.nx = nx;
    lastStats.ny = ny;
    lastStats.nz = nz;
    lastStats.particleCount = (int)particles.size();
    lastStats.liquidCells = 0;
    lastStats.maxSpeed = 0.0f;
    lastStats.maxDivergence = 0.0f;
    lastStats.dt = dt;
    lastStats.lastStepMs = stepMs;
    lastStats.targetMass = targetMass;
    lastStats.desiredMass = desiredMass;
    lastStats.backendName = "CPU MAC 3D";
    lastStats.bytesAllocated =
        u.size() * sizeof(float) +
        v.size() * sizeof(float) +
        w.size() * sizeof(float) +
        pressure.size() * sizeof(float) +
        pressureTmp.size() * sizeof(float) +
        rhs.size() * sizeof(float) +
        water.size() * sizeof(float) +
        divergence.size() * sizeof(float) +
        speed.size() * sizeof(float) +
        uWeight.size() * sizeof(float) +
        vWeight.size() * sizeof(float) +
        wWeight.size() * sizeof(float) +
        uPrev.size() * sizeof(float) +
        vPrev.size() * sizeof(float) +
        wPrev.size() * sizeof(float) +
        uDelta.size() * sizeof(float) +
        vDelta.size() * sizeof(float) +
        wDelta.size() * sizeof(float) +
        uTmp.size() * sizeof(float) +
        vTmp.size() * sizeof(float) +
        wTmp.size() * sizeof(float) +
        liquid.size() * sizeof(uint8_t) +
        solid.size() * sizeof(uint8_t) +
        solidUser.size() * sizeof(uint8_t) +
        validU.size() * sizeof(uint8_t) +
        validV.size() * sizeof(uint8_t) +
        validW.size() * sizeof(uint8_t) +
        validUNext.size() * sizeof(uint8_t) +
        validVNext.size() * sizeof(uint8_t) +
        validWNext.size() * sizeof(uint8_t) +
        (extrapFrontierU.size() + extrapFrontierV.size() + extrapFrontierW.size() +
         extrapNextFrontierU.size() + extrapNextFrontierV.size() + extrapNextFrontierW.size()) * sizeof(int) +
        (uDiffusionStencil.face.size() + uDiffusionStencil.xm.size() + uDiffusionStencil.xp.size() +
         uDiffusionStencil.ym.size() + uDiffusionStencil.yp.size() + uDiffusionStencil.zm.size() +
         uDiffusionStencil.zp.size()) * sizeof(int) + uDiffusionStencil.neighborCount.size() * sizeof(uint8_t) +
        (vDiffusionStencil.face.size() + vDiffusionStencil.xm.size() + vDiffusionStencil.xp.size() +
         vDiffusionStencil.ym.size() + vDiffusionStencil.yp.size() + vDiffusionStencil.zm.size() +
         vDiffusionStencil.zp.size()) * sizeof(int) + vDiffusionStencil.neighborCount.size() * sizeof(uint8_t) +
        (wDiffusionStencil.face.size() + wDiffusionStencil.xm.size() + wDiffusionStencil.xp.size() +
         wDiffusionStencil.ym.size() + wDiffusionStencil.yp.size() + wDiffusionStencil.zm.size() +
         wDiffusionStencil.zp.size()) * sizeof(int) + wDiffusionStencil.neighborCount.size() * sizeof(uint8_t) +
        (uDiffusionScratch.r.size() + uDiffusionScratch.z.size() + uDiffusionScratch.p.size() + uDiffusionScratch.q.size() +
         vDiffusionScratch.r.size() + vDiffusionScratch.z.size() + vDiffusionScratch.p.size() + vDiffusionScratch.q.size() +
         wDiffusionScratch.r.size() + wDiffusionScratch.z.size() + wDiffusionScratch.p.size() + wDiffusionScratch.q.size()) * sizeof(float) +
        reseedCounts.size() * sizeof(int) +
        reseedOccupied.size() * sizeof(uint8_t) +
        reseedRegion.size() * sizeof(uint8_t) +
        relaxBucketCounts.size() * sizeof(int) +
        relaxBucketOffsets.size() * sizeof(int) +
        relaxBucketCursor.size() * sizeof(int) +
        relaxBucketParticles.size() * sizeof(int) +
        particles.size() * sizeof(Particle);

    for (float value : u) lastStats.maxSpeed = std::max(lastStats.maxSpeed, std::fabs(value));
    for (float value : v) lastStats.maxSpeed = std::max(lastStats.maxSpeed, std::fabs(value));
    for (float value : w) lastStats.maxSpeed = std::max(lastStats.maxSpeed, std::fabs(value));

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const int id = idxCell(i, j, k);
                if (liquid[(std::size_t)id] && !solid[(std::size_t)id]) lastStats.liquidCells++;
                if (solid[(std::size_t)id]) continue;

                float uL = u[(std::size_t)idxU(i, j, k)];
                float uR = u[(std::size_t)idxU(i + 1, j, k)];
                float vB = v[(std::size_t)idxV(i, j, k)];
                float vT = v[(std::size_t)idxV(i, j + 1, k)];
                float wBk = w[(std::size_t)idxW(i, j, k)];
                float wFr = w[(std::size_t)idxW(i, j, k + 1)];

                if (i - 1 >= 0 && solid[(std::size_t)idxCell(i - 1, j, k)]) uL = 0.0f;
                if (i + 1 < nx && solid[(std::size_t)idxCell(i + 1, j, k)]) uR = 0.0f;
                if (j - 1 >= 0 && solid[(std::size_t)idxCell(i, j - 1, k)]) vB = 0.0f;
                if (j + 1 < ny && solid[(std::size_t)idxCell(i, j + 1, k)]) vT = 0.0f;
                if (k - 1 >= 0 && solid[(std::size_t)idxCell(i, j, k - 1)]) wBk = 0.0f;
                if (k + 1 < nz && solid[(std::size_t)idxCell(i, j, k + 1)]) wFr = 0.0f;

                const float divCell = (uR - uL + vT - vB + wFr - wBk) / dx;
                lastStats.maxDivergence = std::max(lastStats.maxDivergence, std::fabs(divCell));
            }
        }
    }
}
