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

    // --- Smooth the rendered density field ---------------------------------
    // With only a few particles per cell the raw mass splat is noisy at the
    // cell grid scale, which makes the ray-marched surface look like a mass
    // of disconnected blocky strands.  A short separable blur over the
    // `water` field (rendering-only; `liquid` mask and pressure solve are
    // unaffected) rounds out the density so the renderer's isosurface and
    // alpha-accumulation paths show a continuous body of liquid.
    //
    // Previously we ran 2 passes of a 3x3x3 stencil.  That left visible
    // cell-scale lumps because each pass only reaches 1 cell outward, so
    // stochastic variance at cell scale only diffused ~2 cells total, about
    // the same size as the rasterized chunks themselves.  We now run 4
    // passes of a separable 1D triangular filter (radius 2), which is
    // ~equivalent to a 9x9x9 Gaussian-ish blur and kills the blockiness.
    {
        const int cellCountAll = nx * ny * nz;
        static thread_local std::vector<float> blurScratch;
        if ((int)blurScratch.size() != cellCountAll) {
            blurScratch.assign((std::size_t)cellCountAll, 0.0f);
        }

        // Separable 1D triangular kernel with radius 2: weights {1,2,3,2,1}.
        // Skips Solid neighbours so density can't bleed into pipe walls.
        auto blurAxis = [&](int axis) {
            std::copy(water.begin(), water.end(), blurScratch.begin());
            for (int k = 0; k < nz; ++k) {
                for (int j = 0; j < ny; ++j) {
                    for (int i = 0; i < nx; ++i) {
                        const int id = idxCell(i, j, k);
                        if (solid[(std::size_t)id]) continue;

                        float acc = 0.0f;
                        float wsum = 0.0f;
                        for (int off = -2; off <= 2; ++off) {
                            int ii = i, jj = j, kk = k;
                            if      (axis == 0) ii += off;
                            else if (axis == 1) jj += off;
                            else                kk += off;
                            if (ii < 0 || ii >= nx) continue;
                            if (jj < 0 || jj >= ny) continue;
                            if (kk < 0 || kk >= nz) continue;
                            const int nid = idxCell(ii, jj, kk);
                            if (solid[(std::size_t)nid]) continue;
                            // Triangular weights: 1,2,3,2,1.
                            const float wgt = (float)(3 - std::abs(off));
                            acc  += wgt * blurScratch[(std::size_t)nid];
                            wsum += wgt;
                        }
                        if (wsum > 0.0f) {
                            water[(std::size_t)id] = acc / wsum;
                        }
                    }
                }
            }
        };

        constexpr int BLUR_PASSES = 4;
        for (int pass = 0; pass < BLUR_PASSES; ++pass) {
            blurAxis(0);
            blurAxis(1);
            blurAxis(2);
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
    lastStats.preProjectionMaxDivergence = preProjectionMaxDivergence;
    lastStats.postProjectionMaxDivergence = postProjectionMaxDivergence;
    lastStats.pressureOpenFaceCount = pressureOpenFaceCount;
    lastStats.pressureBlockedFaceCount = pressureBlockedFaceCount;
    lastStats.pressureWeightedFaceCount = pressureWeightedFaceCount;
    lastStats.pressureActiveCellCount = pressureActiveCellCount;
    lastStats.pressureComponentCount = pressureComponentCount;
    lastStats.pressureNeighborLinkCount = pressureNeighborLinkCount;
    lastStats.pressureDirichletFaceCount = pressureDirichletFaceCount;
    lastStats.minFaceOpen = 1.0f;
    lastStats.faceOpenCountLt099 = 0;
    lastStats.faceOpenCountLt050 = 0;
    lastStats.faceOpenCountClosed = 0;
    auto accumulateFaceOpenDiagnostics = [&](const std::vector<float>& faces) {
        for (float f : faces) {
            const float clamped = std::max(0.0f, std::min(1.0f, f));
            lastStats.minFaceOpen = std::min(lastStats.minFaceOpen, clamped);
            if (clamped < 0.99f) ++lastStats.faceOpenCountLt099;
            if (clamped < 0.50f) ++lastStats.faceOpenCountLt050;
            if (clamped <= 1.0e-4f) ++lastStats.faceOpenCountClosed;
        }
    };
    accumulateFaceOpenDiagnostics(uFaceOpen);
    accumulateFaceOpenDiagnostics(vFaceOpen);
    accumulateFaceOpenDiagnostics(wFaceOpen);
    lastStats.dt = dt;
    lastStats.lastStepMs = stepMs;
    lastStats.targetMass = targetMass;
    lastStats.desiredMass = desiredMass;
    lastStats.backendName = "CPU MAC 3D";
    lastStats.nearClosedFaceFluxCount = nearClosedFaceFluxCount;
    lastStats.maxNearClosedFaceFlux = maxNearClosedFaceFlux;
    lastStats.particlesNearWallCount = particlesNearWallCount;
    lastStats.particlesInsideWallCount = particlesInsideWallCount;
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
        pressureRegion.solid.size() * sizeof(uint8_t) +
        pressureRegion.fluid.size() * sizeof(uint8_t) +
        (pressureRegion.rhs.size() + pressureRegion.pressure.size() + pressureRegion.tmp.size()) * sizeof(float) +
        pressureComponentLabel.size() * sizeof(int) +
        pressureComponentQueue.size() * sizeof(int) +
        pressureComponentCells.size() * sizeof(int) +
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

                const int uLid = idxU(i, j, k);
                const int uRid = idxU(i + 1, j, k);
                const int vBid = idxV(i, j, k);
                const int vTid = idxV(i, j + 1, k);
                const int wBid = idxW(i, j, k);
                const int wFid = idxW(i, j, k + 1);
                auto faceCoeff = [](float open) {
                    constexpr float kClosedFace = 1.0e-4f;
                    return (open <= kClosedFace) ? 0.0f : water3d_internal::clampf(open, 0.0f, 1.0f);
                };
                float uL = u[(std::size_t)uLid] * faceCoeff(uFaceOpen[(std::size_t)uLid]);
                float uR = u[(std::size_t)uRid] * faceCoeff(uFaceOpen[(std::size_t)uRid]);
                float vB = v[(std::size_t)vBid] * faceCoeff(vFaceOpen[(std::size_t)vBid]);
                float vT = v[(std::size_t)vTid] * faceCoeff(vFaceOpen[(std::size_t)vTid]);
                float wBk = w[(std::size_t)wBid] * faceCoeff(wFaceOpen[(std::size_t)wBid]);
                float wFr = w[(std::size_t)wFid] * faceCoeff(wFaceOpen[(std::size_t)wFid]);

                if (i - 1 >= 0 && solid[(std::size_t)idxCell(i - 1, j, k)]) uL = 0.0f;
                if (i + 1 < nx && solid[(std::size_t)idxCell(i + 1, j, k)]) uR = 0.0f;
                if (j - 1 >= 0 && solid[(std::size_t)idxCell(i, j - 1, k)]) vB = 0.0f;
                if (j + 1 < ny && solid[(std::size_t)idxCell(i, j + 1, k)]) vT = 0.0f;
                if (k - 1 >= 0 && solid[(std::size_t)idxCell(i, j, k - 1)]) wBk = 0.0f;
                if (k + 1 < nz && solid[(std::size_t)idxCell(i, j, k + 1)]) wFr = 0.0f;

                const float divCell = (uR - uL + vT - vB + wFr - wBk) / dx;
                lastStats.maxDivergence = std::max(lastStats.maxDivergence, std::fabs(divCell));

                const float openThresh = 0.25f;
                auto trackFace = [&](float open, float vel) {
                    if (open < openThresh && std::fabs(vel) > 1.0e-4f) {
                        ++nearClosedFaceFluxCount;
                        maxNearClosedFaceFlux = std::max(maxNearClosedFaceFlux, std::fabs(vel));
                    }
                };
                trackFace(water3d_internal::clampf(uFaceOpen[(std::size_t)uLid], 0.0f, 1.0f), u[(std::size_t)uLid]);
                trackFace(water3d_internal::clampf(uFaceOpen[(std::size_t)uRid], 0.0f, 1.0f), u[(std::size_t)uRid]);
                trackFace(water3d_internal::clampf(vFaceOpen[(std::size_t)vBid], 0.0f, 1.0f), v[(std::size_t)vBid]);
                trackFace(water3d_internal::clampf(vFaceOpen[(std::size_t)vTid], 0.0f, 1.0f), v[(std::size_t)vTid]);
                trackFace(water3d_internal::clampf(wFaceOpen[(std::size_t)wBid], 0.0f, 1.0f), w[(std::size_t)wBid]);
                trackFace(water3d_internal::clampf(wFaceOpen[(std::size_t)wFid], 0.0f, 1.0f), w[(std::size_t)wFid]);
            }
        }
    }
}
