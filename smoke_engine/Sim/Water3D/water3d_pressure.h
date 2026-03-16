#pragma once

#include "water3d_common.h"

#include <algorithm>
#include <cmath>

inline void MACWater3D::projectLiquid() {
    const int cellCount = nx * ny * nz;
    if (cellCount <= 0) return;

    bool anyLiquid = false;
    for (int id = 0; id < cellCount; ++id) {
        if (!solid[(std::size_t)id] && liquid[(std::size_t)id]) {
            anyLiquid = true;
            break;
        }
    }
    if (!anyLiquid) return;

    if ((int)pressure.size() != cellCount) pressure.assign((std::size_t)cellCount, 0.0f);
    if ((int)pressureTmp.size() != cellCount) pressureTmp.assign((std::size_t)cellCount, 0.0f);
    if ((int)rhs.size() != cellCount) rhs.assign((std::size_t)cellCount, 0.0f);

    const float invDx = 1.0f / dx;
    const float invDt = 1.0f / std::max(1e-8f, dt);
    const float dx2 = dx * dx;

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const int id = idxCell(i, j, k);
                if (solid[(std::size_t)id] || !liquid[(std::size_t)id]) {
                    rhs[(std::size_t)id] = 0.0f;
                    pressure[(std::size_t)id] = 0.0f;
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

                const float divCell = (uR - uL + vT - vB + wFr - wBk) * invDx;
                rhs[(std::size_t)id] = -divCell * invDt;
            }
        }
    }

    const int maxIters = std::max(1, params.pressureIters);
    const float omega = std::max(0.0f, params.pressureOmega);

    for (int it = 0; it < maxIters; ++it) {
        float maxResidual = 0.0f;

        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    const int id = idxCell(i, j, k);
                    if (solid[(std::size_t)id] || !liquid[(std::size_t)id]) {
                        pressure[(std::size_t)id] = 0.0f;
                        continue;
                    }

                    float sum = 0.0f;
                    int diag = 0;

                    auto addNeighbor = [&](int ni, int nj, int nk, bool treatOutsideAsAir) {
                        if (ni < 0 || nj < 0 || nk < 0 || ni >= nx || nj >= ny || nk >= nz) {
                            if (treatOutsideAsAir) {
                                diag++;
                            }
                            return;
                        }

                        const int nid = idxCell(ni, nj, nk);
                        if (solid[(std::size_t)nid]) return;
                        if (liquid[(std::size_t)nid]) {
                            sum += pressure[(std::size_t)nid];
                            diag++;
                        } else {
                            diag++;
                        }
                    };

                    addNeighbor(i - 1, j, k, false);
                    addNeighbor(i + 1, j, k, false);
                    addNeighbor(i, j - 1, k, false);
                    addNeighbor(i, j + 1, k, params.openTop && (j + 1 >= ny));
                    addNeighbor(i, j, k - 1, false);
                    addNeighbor(i, j, k + 1, false);

                    if (diag <= 0) {
                        pressure[(std::size_t)id] = 0.0f;
                        continue;
                    }

                    const float target = (sum + rhs[(std::size_t)id] * dx2) / (float)diag;
                    const float updated =
                        pressure[(std::size_t)id] + omega * (target - pressure[(std::size_t)id]);
                    pressure[(std::size_t)id] = updated;

                    float residual = (float)diag * updated - sum - rhs[(std::size_t)id] * dx2;
                    residual = std::fabs(residual) / std::max(1e-8f, dx2);
                    maxResidual = std::max(maxResidual, residual);
                }
            }
        }

        if (maxResidual * dt <= params.pressureTol) {
            break;
        }
    }

    const float scale = dt / dx;

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i <= nx; ++i) {
                const int id = idxU(i, j, k);
                const bool leftSolid = (i - 1 >= 0) ? isSolidCell(i - 1, j, k) : true;
                const bool rightSolid = (i < nx) ? isSolidCell(i, j, k) : true;
                if (leftSolid || rightSolid) {
                    u[(std::size_t)id] = 0.0f;
                    continue;
                }

                const bool leftFluid = (i - 1 >= 0)
                    ? (!solid[(std::size_t)idxCell(i - 1, j, k)] && liquid[(std::size_t)idxCell(i - 1, j, k)])
                    : false;
                const bool rightFluid = (i < nx)
                    ? (!solid[(std::size_t)idxCell(i, j, k)] && liquid[(std::size_t)idxCell(i, j, k)])
                    : false;

                if (!leftFluid && !rightFluid) continue;

                const float pL = leftFluid ? pressure[(std::size_t)idxCell(i - 1, j, k)] : 0.0f;
                const float pR = rightFluid ? pressure[(std::size_t)idxCell(i, j, k)] : 0.0f;
                u[(std::size_t)id] -= scale * (pR - pL);
            }
        }
    }

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const int id = idxV(i, j, k);
                const bool botSolid = (j - 1 >= 0) ? isSolidCell(i, j - 1, k) : true;
                const bool topSolid = (j < ny) ? isSolidCell(i, j, k) : !params.openTop;
                if (botSolid || topSolid) {
                    v[(std::size_t)id] = 0.0f;
                    continue;
                }

                const bool botFluid = (j - 1 >= 0)
                    ? (!solid[(std::size_t)idxCell(i, j - 1, k)] && liquid[(std::size_t)idxCell(i, j - 1, k)])
                    : false;
                const bool topFluid = (j < ny)
                    ? (!solid[(std::size_t)idxCell(i, j, k)] && liquid[(std::size_t)idxCell(i, j, k)])
                    : false;

                if (!botFluid && !topFluid) continue;

                const float pB = botFluid ? pressure[(std::size_t)idxCell(i, j - 1, k)] : 0.0f;
                const float pT = topFluid ? pressure[(std::size_t)idxCell(i, j, k)] : 0.0f;
                v[(std::size_t)id] -= scale * (pT - pB);
            }
        }
    }

    for (int k = 0; k <= nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const int id = idxW(i, j, k);
                const bool backSolid = (k - 1 >= 0) ? isSolidCell(i, j, k - 1) : true;
                const bool frontSolid = (k < nz) ? isSolidCell(i, j, k) : true;
                if (backSolid || frontSolid) {
                    w[(std::size_t)id] = 0.0f;
                    continue;
                }

                const bool backFluid = (k - 1 >= 0)
                    ? (!solid[(std::size_t)idxCell(i, j, k - 1)] && liquid[(std::size_t)idxCell(i, j, k - 1)])
                    : false;
                const bool frontFluid = (k < nz)
                    ? (!solid[(std::size_t)idxCell(i, j, k)] && liquid[(std::size_t)idxCell(i, j, k)])
                    : false;

                if (!backFluid && !frontFluid) continue;

                const float pBk = backFluid ? pressure[(std::size_t)idxCell(i, j, k - 1)] : 0.0f;
                const float pFr = frontFluid ? pressure[(std::size_t)idxCell(i, j, k)] : 0.0f;
                w[(std::size_t)id] -= scale * (pFr - pBk);
            }
        }
    }

    applyBoundary();
}
