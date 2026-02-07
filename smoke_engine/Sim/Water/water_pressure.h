#pragma once
// Implementation header included by Sim/mac_water_sim.cpp
#include "water_common.h"

#include <algorithm>


inline void MACWater::projectLiquid() {
    const int Nc = nx * ny;
    if (Nc <= 0) return;

    // If there's no liquid, nothing to project.
    bool anyLiquid = false;
    for (int id = 0; id < Nc; ++id) {
        if (!solid[(size_t)id] && liquid[(size_t)id]) { anyLiquid = true; break; }
    }
    if (!anyLiquid) return;

    // Ensure buffers are sized.
    if ((int)p.size() != Nc) p.assign((size_t)Nc, 0.0f);
    if ((int)rhs.size() != Nc) rhs.assign((size_t)Nc, 0.0f);

  


    const float invDx = 1.0f / dx;
    const float invDt = 1.0f / std::max(1e-8f, dt);



    // Build rhs and Jacobi preconditioner diagonal.
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int id = idxP(i, j);
            if (solid[(size_t)id] || !liquid[(size_t)id]) {
                rhs[(size_t)id] = 0.0f;
                // diagInv[(size_t)id] = 0.0f;
                continue;
            }

            // Divergence in this cell (flux out / dx).
            float uL = u[(size_t)idxU(i, j)];
            float uR = u[(size_t)idxU(i + 1, j)];
            float vB = v[(size_t)idxV(i, j)];
            float vT = v[(size_t)idxV(i, j + 1)];

            // Solid neighbors imply zero normal velocity.
            if (i - 1 >= 0 && solid[(size_t)idxP(i - 1, j)]) uL = 0.0f;
            if (i + 1 < nx && solid[(size_t)idxP(i + 1, j)]) uR = 0.0f;
            if (j - 1 >= 0 && solid[(size_t)idxP(i, j - 1)]) vB = 0.0f;
            if (j + 1 < ny && solid[(size_t)idxP(i, j + 1)]) vT = 0.0f;

            const float divCell = (uR - uL + vT - vB) * invDx;
            rhs[(size_t)id] = -divCell * invDt;

            
        }
    }

    // ---- Volume preservation (surface-only net-divergence compensation) ----
    // Idea: compute mean divergence over LIQUID, then add a small compensating
    // divergence term ONLY on free-surface liquid cells.
    // This avoids “biasing the whole bulk” which can destabilize / stiffen the liquid.
    if (volumePreserveRhsMean) {
        const float k = std::max(0.0f, std::min(volumePreserveStrength, 1.0f));

        // Compute mean divergence over liquid (in physical divergence units, not RHS units)
        double sumDiv = 0.0;
        int cntDiv = 0;
        float maxAbsDiv = 0.0f;

        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const int id = idxP(i, j);
                if (solid[(size_t)id] || !liquid[(size_t)id]) continue;

                float uL = u[(size_t)idxU(i, j)];
                float uR = u[(size_t)idxU(i + 1, j)];
                float vB = v[(size_t)idxV(i, j)];
                float vT = v[(size_t)idxV(i, j + 1)];

                if (i - 1 >= 0 && solid[(size_t)idxP(i - 1, j)]) uL = 0.0f;
                if (i + 1 < nx && solid[(size_t)idxP(i + 1, j)]) uR = 0.0f;
                if (j - 1 >= 0 && solid[(size_t)idxP(i, j - 1)]) vB = 0.0f;
                if (j + 1 < ny && solid[(size_t)idxP(i, j + 1)]) vT = 0.0f;

                const float divCell = (uR - uL + vT - vB) * invDx;

                sumDiv += (double)divCell;
                cntDiv++;
                maxAbsDiv = std::max(maxAbsDiv, std::fabs(divCell));
            }
        }

        if (cntDiv > 0 && maxAbsDiv > 0.0f && k > 0.0f) {
            const float meanDiv = (float)(sumDiv / (double)cntDiv);

            // We want to cancel mean divergence (net expansion/compression)
            float divBias = -k * meanDiv;

            // Safety clamp: never apply a bias bigger than some fraction of typical div
            const float clampMag = 0.25f * maxAbsDiv;
            divBias = clampf(divBias, -clampMag, clampMag);

            // Apply ONLY on surface liquid cells
            auto isSurfaceLiquid = [&](int i, int j) -> bool {
                const int id = idxP(i, j);
                if (solid[(size_t)id] || !liquid[(size_t)id]) return false;

                auto isAir = [&](int ni, int nj) -> bool {
                    if (ni < 0 || nj < 0 || ni >= nx || nj >= ny) return true; // outside = air
                    const int nid = idxP(ni, nj);
                    if (solid[(size_t)nid]) return false;
                    return liquid[(size_t)nid] == 0;
                };

                return isAir(i - 1, j) || isAir(i + 1, j) || isAir(i, j - 1) || isAir(i, j + 1);
            };

            // First pass: count surface cells
            int surfCnt = 0;
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i)
                    if (isSurfaceLiquid(i, j)) surfCnt++;

            if (surfCnt > 0) {
                // Distribute the correction so the *total* imposed divergence matches what we want,
                // instead of adding the same per-cell divergence (which can blow up "div(all)" metrics).
                //
                // We want total added divergence over surface ≈ divBias * (liquidCellCount)
                // so per-surface-cell add = divBias * (liquidCellCount / surfCnt)
                const float scaleSurf = (float)cntDiv / (float)surfCnt;
                float perCellDiv = divBias * scaleSurf;

                // clamp after scaling 
                const float perClamp = 0.25f * maxAbsDiv;  // tune: 0.1..0.5
                perCellDiv = clampf(perCellDiv, -perClamp, perClamp);

                // Apply on surface only
                for (int j = 0; j < ny; ++j) {
                    for (int i = 0; i < nx; ++i) {
                        if (!isSurfaceLiquid(i, j)) continue;
                        const int id = idxP(i, j);
                        rhs[(size_t)id] -= perCellDiv * invDt;
                    }
                }
            }

            // // If no surface cells (fully enclosed liquid), do nothing.
            (void)surfCnt;
        }
    }

    // --- Shared pressure solve (PCG) ---
    ps().configure(
        nx, ny, dx,
        /*openTopBC=*/openTop,
        solid,
        liquid,
        /*removeMeanForGauge=*/false   // IMPORTANT for free-surface water
    );

    // Warm start: do NOT zero p every frame
    // (If you still want to reset sometimes, do it outside based on user action.)
    const int maxIters = std::max(1, pressureMaxIters);

    // Your current 'pressureTol' in Water is comparing against residual directly.
    // In the shared solver we interpret tol in "predDiv space": |r|*dt <= tol.
    // So we pass your existing value as-is and dt along.
    const float tolPredDiv = std::max(0.0f, pressureTol);

    // ps().solvePCG(p, rhs, maxIters, tolPredDiv, dt);

    ps().solveMG(p, rhs, maxIters, pressureTol, dt);

    // Subtract pressure gradient from velocities.
    const float scale = dt / dx;

    // u faces
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i <= nx; ++i) {
            const int id = idxU(i, j);

            const bool leftSolid  = (i - 1 >= 0) ? isSolid(i - 1, j) : true;
            const bool rightSolid = (i < nx)     ? isSolid(i, j)     : true;
            if (leftSolid || rightSolid) {
                u[(size_t)id] = 0.0f;
                continue;
            }

            const bool leftFluid  = (i - 1 >= 0) ? (!solid[(size_t)idxP(i - 1, j)] && liquid[(size_t)idxP(i - 1, j)]) : false;
            const bool rightFluid = (i < nx)     ? (!solid[(size_t)idxP(i, j)]     && liquid[(size_t)idxP(i, j)])     : false;

            if (!leftFluid && !rightFluid) continue; // air-air face

            const float pL = leftFluid  ? p[(size_t)idxP(i - 1, j)] : 0.0f;
            const float pR = rightFluid ? p[(size_t)idxP(i, j)]     : 0.0f;
            u[(size_t)id] -= scale * (pR - pL);
        }
    }

    // v faces
    for (int j = 0; j <= ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int id = idxV(i, j);

            const bool botSolid = (j - 1 >= 0) ? isSolid(i, j - 1) : true;
            const bool topSolid = (j < ny)     ? isSolid(i, j)     : (openTop ? false : true);
            if (botSolid || topSolid) {
                v[(size_t)id] = 0.0f;
                continue;
            }

            const bool botFluid = (j - 1 >= 0) ? (!solid[(size_t)idxP(i, j - 1)] && liquid[(size_t)idxP(i, j - 1)]) : false;
            const bool topFluid = (j < ny)     ? (!solid[(size_t)idxP(i, j)]     && liquid[(size_t)idxP(i, j)])     : false;

            if (!botFluid && !topFluid) continue;

            const float pB = botFluid ? p[(size_t)idxP(i, j - 1)] : 0.0f;
            const float pT = topFluid ? p[(size_t)idxP(i, j)]     : 0.0f;
            v[(size_t)id] -= scale * (pT - pB);
        }
    }

    applyBoundary();
}