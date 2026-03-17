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

    float maxAbsDiv = 0.0f;
    int liquidCount = 0;

    auto isFluidCell = [&](int i, int j, int k) -> bool {
        if (i < 0 || j < 0 || k < 0 || i >= nx || j >= ny || k >= nz) return false;
        const int id = idxCell(i, j, k);
        return !solid[(std::size_t)id] && liquid[(std::size_t)id];
    };

    auto isAirNeighbor = [&](int ni, int nj, int nk) -> bool {
        if (ni < 0 || nk < 0 || ni >= nx || nk >= nz) return false;
        if (nj < 0) return false;
        if (nj >= ny) return params.openTop;
        const int nid = idxCell(ni, nj, nk);
        return !solid[(std::size_t)nid] && !liquid[(std::size_t)nid];
    };

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const int id = idxCell(i, j, k);
                if (solid[(std::size_t)id] || !liquid[(std::size_t)id]) {
                    rhs[(std::size_t)id] = 0.0f;
                    pressure[(std::size_t)id] = 0.0f;
                    continue;
                }

                if (!std::isfinite(pressure[(std::size_t)id])) {
                    pressure[(std::size_t)id] = 0.0f;
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
                maxAbsDiv = std::max(maxAbsDiv, std::fabs(divCell));
                liquidCount++;
            }
        }
    }

    if (params.volumePreserveRhsMean) {
        const float k = water3d_internal::clamp01(params.volumePreserveStrength);
        if (k > 0.0f && desiredMass > 0.0f && targetMass > 0.0f && liquidCount > 0) {
            const float relErr = (desiredMass - targetMass) / std::max(1e-6f, desiredMass);
            float divTarget = (relErr * k) / std::max(1e-6f, dt);
            const float divClamp = (maxAbsDiv > 0.0f) ? (0.25f * maxAbsDiv) : 0.0f;
            if (divClamp > 0.0f) {
                divTarget = water3d_internal::clampf(divTarget, -divClamp, divClamp);
            } else {
                divTarget = 0.0f;
            }

            if (divTarget != 0.0f) {
                const float rhsAddFace = divTarget * invDt;
                for (int k3 = 0; k3 < nz; ++k3) {
                    for (int j3 = 0; j3 < ny; ++j3) {
                        for (int i3 = 0; i3 < nx; ++i3) {
                            const int id = idxCell(i3, j3, k3);
                            if (solid[(std::size_t)id] || !liquid[(std::size_t)id]) continue;

                            int exposedFaces = 0;
                            exposedFaces += isAirNeighbor(i3 - 1, j3, k3) ? 1 : 0;
                            exposedFaces += isAirNeighbor(i3 + 1, j3, k3) ? 1 : 0;
                            exposedFaces += isAirNeighbor(i3, j3 - 1, k3) ? 1 : 0;
                            exposedFaces += isAirNeighbor(i3, j3 + 1, k3) ? 1 : 0;
                            exposedFaces += isAirNeighbor(i3, j3, k3 - 1) ? 1 : 0;
                            exposedFaces += isAirNeighbor(i3, j3, k3 + 1) ? 1 : 0;

                            if (exposedFaces > 0) {
                                rhs[(std::size_t)id] += rhsAddFace * (float)exposedFaces;
                            }
                        }
                    }
                }
            }
        }
    }

    auto neighborContribution = [&](int ni, int nj, int nk, bool openTopOutside,
                                    float& sum, int& diag) {
        if (ni < 0 || nj < 0 || nk < 0 || ni >= nx || nj >= ny || nk >= nz) {
            if (openTopOutside) {
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

    auto computeResidual = [&]() {
        float maxResidual = 0.0f;
        for (int k3 = 0; k3 < nz; ++k3) {
            for (int j3 = 0; j3 < ny; ++j3) {
                for (int i3 = 0; i3 < nx; ++i3) {
                    const int id = idxCell(i3, j3, k3);
                    if (solid[(std::size_t)id] || !liquid[(std::size_t)id]) continue;

                    float sum = 0.0f;
                    int diag = 0;
                    neighborContribution(i3 - 1, j3, k3, false, sum, diag);
                    neighborContribution(i3 + 1, j3, k3, false, sum, diag);
                    neighborContribution(i3, j3 - 1, k3, false, sum, diag);
                    neighborContribution(i3, j3 + 1, k3, params.openTop && (j3 + 1 >= ny), sum, diag);
                    neighborContribution(i3, j3, k3 - 1, false, sum, diag);
                    neighborContribution(i3, j3, k3 + 1, false, sum, diag);

                    if (diag <= 0) continue;
                    const float residual = std::fabs((float)diag * pressure[(std::size_t)id] - sum - rhs[(std::size_t)id] * dx2)
                                         / std::max(1e-8f, dx2);
                    maxResidual = std::max(maxResidual, residual);
                }
            }
        }
        return maxResidual;
    };

    const bool useJacobi = (params.pressureSolverMode == (int)PressureSolverMode::Jacobi);
    const int maxIters = std::max(1, params.pressureIters);
    const float rbgsOmega = water3d_internal::clampf(params.pressureOmega, 0.0f, 1.95f);
    const float jacobiOmega = water3d_internal::clampf(params.pressureOmega, 0.0f, 1.0f);

    if (useJacobi) {
        for (int it = 0; it < maxIters; ++it) {
            for (int k3 = 0; k3 < nz; ++k3) {
                for (int j3 = 0; j3 < ny; ++j3) {
                    for (int i3 = 0; i3 < nx; ++i3) {
                        const int id = idxCell(i3, j3, k3);
                        if (solid[(std::size_t)id] || !liquid[(std::size_t)id]) {
                            pressureTmp[(std::size_t)id] = 0.0f;
                            continue;
                        }

                        float sum = 0.0f;
                        int diag = 0;
                        neighborContribution(i3 - 1, j3, k3, false, sum, diag);
                        neighborContribution(i3 + 1, j3, k3, false, sum, diag);
                        neighborContribution(i3, j3 - 1, k3, false, sum, diag);
                        neighborContribution(i3, j3 + 1, k3, params.openTop && (j3 + 1 >= ny), sum, diag);
                        neighborContribution(i3, j3, k3 - 1, false, sum, diag);
                        neighborContribution(i3, j3, k3 + 1, false, sum, diag);

                        if (diag <= 0) {
                            pressureTmp[(std::size_t)id] = 0.0f;
                            continue;
                        }

                        const float target = (sum + rhs[(std::size_t)id] * dx2) / (float)diag;
                        pressureTmp[(std::size_t)id] =
                            pressure[(std::size_t)id] + jacobiOmega * (target - pressure[(std::size_t)id]);
                    }
                }
            }
            pressure.swap(pressureTmp);
            if (computeResidual() * dt <= params.pressureTol) break;
        }
    } else {
        for (int it = 0; it < maxIters; ++it) {
            for (int color = 0; color < 2; ++color) {
                for (int k3 = 0; k3 < nz; ++k3) {
                    for (int j3 = 0; j3 < ny; ++j3) {
                        for (int i3 = 0; i3 < nx; ++i3) {
                            if (((i3 + j3 + k3) & 1) != color) continue;
                            const int id = idxCell(i3, j3, k3);
                            if (solid[(std::size_t)id] || !liquid[(std::size_t)id]) {
                                pressure[(std::size_t)id] = 0.0f;
                                continue;
                            }

                            float sum = 0.0f;
                            int diag = 0;
                            neighborContribution(i3 - 1, j3, k3, false, sum, diag);
                            neighborContribution(i3 + 1, j3, k3, false, sum, diag);
                            neighborContribution(i3, j3 - 1, k3, false, sum, diag);
                            neighborContribution(i3, j3 + 1, k3, params.openTop && (j3 + 1 >= ny), sum, diag);
                            neighborContribution(i3, j3, k3 - 1, false, sum, diag);
                            neighborContribution(i3, j3, k3 + 1, false, sum, diag);

                            if (diag <= 0) {
                                pressure[(std::size_t)id] = 0.0f;
                                continue;
                            }

                            const float target = (sum + rhs[(std::size_t)id] * dx2) / (float)diag;
                            pressure[(std::size_t)id] += rbgsOmega * (target - pressure[(std::size_t)id]);
                        }
                    }
                }
            }
            if (computeResidual() * dt <= params.pressureTol) break;
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

                const bool leftFluid = (i - 1 >= 0) ? isFluidCell(i - 1, j, k) : false;
                const bool rightFluid = (i < nx) ? isFluidCell(i, j, k) : false;
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

                const bool botFluid = (j - 1 >= 0) ? isFluidCell(i, j - 1, k) : false;
                const bool topFluid = (j < ny) ? isFluidCell(i, j, k) : false;
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

                const bool backFluid = (k - 1 >= 0) ? isFluidCell(i, j, k - 1) : false;
                const bool frontFluid = (k < nz) ? isFluidCell(i, j, k) : false;
                if (!backFluid && !frontFluid) continue;

                const float pBk = backFluid ? pressure[(std::size_t)idxCell(i, j, k - 1)] : 0.0f;
                const float pFr = frontFluid ? pressure[(std::size_t)idxCell(i, j, k)] : 0.0f;
                w[(std::size_t)id] -= scale * (pFr - pBk);
            }
        }
    }

    applyBoundary();
}
