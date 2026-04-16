#pragma once

#include "water3d_common.h"

#include <algorithm>
#include <chrono>
#include <cmath>

inline void MACWater3D::projectLiquid() {
    using clock = std::chrono::high_resolution_clock;

    const int cellCount = nx * ny * nz;
    if (cellCount <= 0) {
        lastPressureSolveMs = 0.0f;
        lastPressureIterations = 0;
        pressureRegion.previousBoxValid = false;
        return;
    }

    if ((int)pressureTmp.size() != cellCount) pressureTmp.assign((std::size_t)cellCount, 0.0f);
    else std::fill(pressureTmp.begin(), pressureTmp.end(), 0.0f);

    if ((int)pressureComponentLabel.size() != cellCount) {
        pressureComponentLabel.assign((std::size_t)cellCount, -1);
    } else {
        std::fill(pressureComponentLabel.begin(), pressureComponentLabel.end(), -1);
    }
    pressureComponentQueue.clear();
    pressureComponentCells.clear();

    pressureRegion.previousBoxValid = false;

    const float safeDt = std::max(1.0e-8f, dt);
    const float invDx = 1.0f / dx;
    const float invDt = 1.0f / safeDt;
    const float dx2 = dx * dx;

    auto isFluidCellGlobal = [&](int i, int j, int k) -> bool {
        if (i < 0 || j < 0 || k < 0 || i >= nx || j >= ny || k >= nz) return false;
        const int id = idxCell(i, j, k);
        return !solid[(std::size_t)id] && liquid[(std::size_t)id];
    };

    auto cellDivergence = [&](int i, int j, int k) -> float {
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

        return (uR - uL + vT - vB + wFr - wBk) * invDx;
    };

    float rhsAddFace = 0.0f;
    if (params.volumePreserveRhsMean) {
        const float k = water3d_internal::clamp01(params.volumePreserveStrength);
        if (k > 0.0f && desiredMass > 0.0f && targetMass > 0.0f) {
            float globalMaxAbsDiv = 0.0f;
            int globalLiquidCount = 0;
            for (int kz = 0; kz < nz; ++kz) {
                for (int jy = 0; jy < ny; ++jy) {
                    for (int ix = 0; ix < nx; ++ix) {
                        if (!isFluidCellGlobal(ix, jy, kz)) continue;
                        globalMaxAbsDiv = std::max(globalMaxAbsDiv, std::fabs(cellDivergence(ix, jy, kz)));
                        ++globalLiquidCount;
                    }
                }
            }

            if (globalLiquidCount > 0) {
                const float relErr = (desiredMass - targetMass) / std::max(1e-6f, desiredMass);

                // Match the corrected 2D free-surface volume-preservation logic:
                // build a gentle per-step divergence target first, then convert to
                // RHS units with a single dt division. The old 3D path divided by
                // dt twice, which made the correction scale as 1/dt^2 and could
                // eventually drive very large pressure spikes once the volume error
                // had grown for a while.
                float divTarget = relErr * k;
                const float divClamp = (globalMaxAbsDiv > 0.0f) ? (0.10f * globalMaxAbsDiv) : 0.0f;
                if (divClamp > 0.0f) {
                    divTarget = water3d_internal::clampf(divTarget, -divClamp, divClamp);
                } else {
                    divTarget = 0.0f;
                }
                rhsAddFace = divTarget * invDt;
            }
        }
    }

    auto labelFluidComponent = [&](int seedId, int componentId,
                                   int& minI, int& maxI,
                                   int& minJ, int& maxJ,
                                   int& minK, int& maxK) {
        pressureComponentQueue.clear();
        pressureComponentCells.clear();
        pressureComponentQueue.push_back(seedId);
        pressureComponentLabel[(std::size_t)seedId] = componentId;

        std::size_t head = 0;
        while (head < pressureComponentQueue.size()) {
            const int id = pressureComponentQueue[head++];
            pressureComponentCells.push_back(id);

            const int plane = nx * ny;
            const int k = id / plane;
            const int rem = id - k * plane;
            const int j = rem / nx;
            const int i = rem - j * nx;

            minI = std::min(minI, i);
            maxI = std::max(maxI, i);
            minJ = std::min(minJ, j);
            maxJ = std::max(maxJ, j);
            minK = std::min(minK, k);
            maxK = std::max(maxK, k);

            auto tryPush = [&](int ni, int nj, int nk) {
                if (ni < 0 || nj < 0 || nk < 0 || ni >= nx || nj >= ny || nk >= nz) return;
                const int nid = idxCell(ni, nj, nk);
                if (solid[(std::size_t)nid] || !liquid[(std::size_t)nid]) return;
                if (pressureComponentLabel[(std::size_t)nid] >= 0) return;
                pressureComponentLabel[(std::size_t)nid] = componentId;
                pressureComponentQueue.push_back(nid);
            };

            tryPush(i - 1, j, k);
            tryPush(i + 1, j, k);
            tryPush(i, j - 1, k);
            tryPush(i, j + 1, k);
            tryPush(i, j, k - 1);
            tryPush(i, j, k + 1);
        }
    };

    const auto solverMode = static_cast<PressureSolverMode>(params.pressureSolverMode);
    float totalPressureMs = 0.0f;
    int totalPressureIterations = 0;
    int componentCount = 0;

    for (int seedId = 0; seedId < cellCount; ++seedId) {
        if (solid[(std::size_t)seedId] || !liquid[(std::size_t)seedId]) continue;
        if (pressureComponentLabel[(std::size_t)seedId] >= 0) continue;

        int minI = nx;
        int maxI = -1;
        int minJ = ny;
        int maxJ = -1;
        int minK = nz;
        int maxK = -1;
        const int componentId = componentCount++;
        labelFluidComponent(seedId, componentId, minI, maxI, minJ, maxJ, minK, maxK);
        if (pressureComponentCells.empty()) continue;

        const int i0 = std::max(0, minI - 1);
        const int i1 = std::min(nx, maxI + 2);
        const int j0 = std::max(0, minJ - 1);
        const int j1 = std::min(ny, maxJ + 2);
        const int k0 = std::max(0, minK - 1);
        const int k1 = std::min(nz, maxK + 2);

        const int sx = std::max(1, i1 - i0);
        const int sy = std::max(1, j1 - j0);
        const int sz = std::max(1, k1 - k0);
        const std::size_t subCount = (std::size_t)sx * (std::size_t)sy * (std::size_t)sz;
        const bool subOpenTop = params.openTop && (j1 >= ny);

        pressureRegion.ensureSize(subCount);
        pressureRegion.i0 = i0;
        pressureRegion.i1 = i1;
        pressureRegion.j0 = j0;
        pressureRegion.j1 = j1;
        pressureRegion.k0 = k0;
        pressureRegion.k1 = k1;

        auto subIdx = [&](int i, int j, int k) -> int {
            return i + sx * (j + sy * k);
        };

        auto isFluidCellComponent = [&](int i, int j, int k) -> bool {
            if (i < 0 || j < 0 || k < 0 || i >= nx || j >= ny || k >= nz) return false;
            const int id = idxCell(i, j, k);
            return !solid[(std::size_t)id] && pressureComponentLabel[(std::size_t)id] == componentId;
        };

        auto isAirNeighborComponent = [&](int ni, int nj, int nk) -> bool {
            if (ni < 0 || nk < 0 || ni >= nx || nk >= nz) return false;
            if (nj < 0) return false;
            if (nj >= ny) return params.openTop;
            const int nid = idxCell(ni, nj, nk);
            return !solid[(std::size_t)nid] && pressureComponentLabel[(std::size_t)nid] != componentId;
        };

        bool hasDirichletReference = false;

        for (int sk = 0; sk < sz; ++sk) {
            const int gk = k0 + sk;
            for (int sj = 0; sj < sy; ++sj) {
                const int gj = j0 + sj;
                for (int si = 0; si < sx; ++si) {
                    const int gi = i0 + si;
                    const int gid = idxCell(gi, gj, gk);
                    const int sid = subIdx(si, sj, sk);

                    const bool isSolid = solid[(std::size_t)gid] != 0;
                    const bool isFluid = !isSolid && (pressureComponentLabel[(std::size_t)gid] == componentId);
                    pressureRegion.solid[(std::size_t)sid] = isSolid ? (uint8_t)1 : (uint8_t)0;
                    pressureRegion.fluid[(std::size_t)sid] = isFluid ? (uint8_t)1 : (uint8_t)0;

                    if (!isFluid) {
                        pressureRegion.rhs[(std::size_t)sid] = 0.0f;
                        pressureRegion.pressure[(std::size_t)sid] = 0.0f;
                        continue;
                    }

                    float warm = pressure[(std::size_t)gid];
                    if (!std::isfinite(warm)) warm = 0.0f;
                    pressureRegion.pressure[(std::size_t)sid] = warm;

                    const float divCell = cellDivergence(gi, gj, gk);
                    pressureRegion.rhs[(std::size_t)sid] = -divCell * invDt;

                    if (!hasDirichletReference) {
                        hasDirichletReference =
                            isAirNeighborComponent(gi - 1, gj, gk) || isAirNeighborComponent(gi + 1, gj, gk) ||
                            isAirNeighborComponent(gi, gj - 1, gk) || isAirNeighborComponent(gi, gj + 1, gk) ||
                            isAirNeighborComponent(gi, gj, gk - 1) || isAirNeighborComponent(gi, gj, gk + 1);
                    }
                }
            }
        }

        if (rhsAddFace != 0.0f) {
            for (int sk = 0; sk < sz; ++sk) {
                const int gk = k0 + sk;
                for (int sj = 0; sj < sy; ++sj) {
                    const int gj = j0 + sj;
                    for (int si = 0; si < sx; ++si) {
                        const int gi = i0 + si;
                        const int sid = subIdx(si, sj, sk);
                        if (!pressureRegion.fluid[(std::size_t)sid]) continue;

                        int exposedFaces = 0;
                        exposedFaces += isAirNeighborComponent(gi - 1, gj, gk) ? 1 : 0;
                        exposedFaces += isAirNeighborComponent(gi + 1, gj, gk) ? 1 : 0;
                        exposedFaces += isAirNeighborComponent(gi, gj - 1, gk) ? 1 : 0;
                        exposedFaces += isAirNeighborComponent(gi, gj + 1, gk) ? 1 : 0;
                        exposedFaces += isAirNeighborComponent(gi, gj, gk - 1) ? 1 : 0;
                        exposedFaces += isAirNeighborComponent(gi, gj, gk + 1) ? 1 : 0;

                        if (exposedFaces > 0) {
                            pressureRegion.rhs[(std::size_t)sid] += rhsAddFace * (float)exposedFaces;
                        }
                    }
                }
            }
        }

        const auto solveStart = clock::now();
        int componentIters = 0;

        if (solverMode == PressureSolverMode::Multigrid) {
            pressurePoisson.configure(
                sx, sy, sz, dx,
                subOpenTop,
                pressureRegion.solid,
                pressureRegion.fluid,
                /*removeMeanForGauge=*/!hasDirichletReference);
            pressurePoisson.setMGControls(
                std::max(1, params.pressureMGCoarseIters),
                std::max(0.0f, params.pressureMGRelativeTol));
            pressurePoisson.setMGSmoother(true, water3d_internal::clampf(params.pressureMGOmega, 0.1f, 1.95f));
            pressurePoisson.solveMG(
                pressureRegion.pressure,
                pressureRegion.rhs,
                std::max(1, params.pressureMGVCycles),
                std::max(0.0f, params.pressureTol),
                dt);
            componentIters = pressurePoisson.lastIterations();
        } else {
            pressureRegion.ensureTmpSize(subCount);

            auto isFluidCellSub = [&](int i, int j, int k) -> bool {
                if (i < 0 || j < 0 || k < 0 || i >= sx || j >= sy || k >= sz) return false;
                const int sid = subIdx(i, j, k);
                return !pressureRegion.solid[(std::size_t)sid] && pressureRegion.fluid[(std::size_t)sid];
            };

            auto neighborContribution = [&](int ni, int nj, int nk, bool openTopOutside,
                                            float& sum, int& diag) {
                if (ni < 0 || nj < 0 || nk < 0 || ni >= sx || nj >= sy || nk >= sz) {
                    if (openTopOutside) {
                        diag++;
                    }
                    return;
                }

                const int sid = subIdx(ni, nj, nk);
                if (pressureRegion.solid[(std::size_t)sid]) return;
                if (pressureRegion.fluid[(std::size_t)sid]) {
                    sum += pressureRegion.pressure[(std::size_t)sid];
                }
                diag++;
            };

            auto computeResidual = [&]() -> float {
                float maxResidual = 0.0f;
                for (int k3 = 0; k3 < sz; ++k3) {
                    for (int j3 = 0; j3 < sy; ++j3) {
                        for (int i3 = 0; i3 < sx; ++i3) {
                            const int sid = subIdx(i3, j3, k3);
                            if (pressureRegion.solid[(std::size_t)sid] || !pressureRegion.fluid[(std::size_t)sid]) continue;

                            float sum = 0.0f;
                            int diag = 0;
                            neighborContribution(i3 - 1, j3, k3, false, sum, diag);
                            neighborContribution(i3 + 1, j3, k3, false, sum, diag);
                            neighborContribution(i3, j3 - 1, k3, false, sum, diag);
                            neighborContribution(i3, j3 + 1, k3, subOpenTop && (j3 + 1 >= sy), sum, diag);
                            neighborContribution(i3, j3, k3 - 1, false, sum, diag);
                            neighborContribution(i3, j3, k3 + 1, false, sum, diag);

                            if (diag <= 0) continue;
                            const float residual =
                                std::fabs((float)diag * pressureRegion.pressure[(std::size_t)sid] - sum - pressureRegion.rhs[(std::size_t)sid] * dx2)
                                / std::max(1e-8f, dx2);
                            maxResidual = std::max(maxResidual, residual);
                        }
                    }
                }
                return maxResidual;
            };

            const bool useJacobi = (solverMode == PressureSolverMode::Jacobi);
            const int maxIters = std::max(1, params.pressureIters);
            const float rbgsOmega = water3d_internal::clampf(params.pressureOmega, 0.0f, 1.95f);
            const float jacobiOmega = water3d_internal::clampf(params.pressureOmega, 0.0f, 1.0f);
            int itUsed = 0;

            if (useJacobi) {
                for (int it = 0; it < maxIters; ++it) {
                    itUsed = it + 1;
                    for (int k3 = 0; k3 < sz; ++k3) {
                        for (int j3 = 0; j3 < sy; ++j3) {
                            for (int i3 = 0; i3 < sx; ++i3) {
                                const int sid = subIdx(i3, j3, k3);
                                if (pressureRegion.solid[(std::size_t)sid] || !pressureRegion.fluid[(std::size_t)sid]) {
                                    pressureRegion.tmp[(std::size_t)sid] = 0.0f;
                                    continue;
                                }

                                float sum = 0.0f;
                                int diag = 0;
                                neighborContribution(i3 - 1, j3, k3, false, sum, diag);
                                neighborContribution(i3 + 1, j3, k3, false, sum, diag);
                                neighborContribution(i3, j3 - 1, k3, false, sum, diag);
                                neighborContribution(i3, j3 + 1, k3, subOpenTop && (j3 + 1 >= sy), sum, diag);
                                neighborContribution(i3, j3, k3 - 1, false, sum, diag);
                                neighborContribution(i3, j3, k3 + 1, false, sum, diag);

                                if (diag <= 0) {
                                    pressureRegion.tmp[(std::size_t)sid] = 0.0f;
                                    continue;
                                }

                                const float target = (sum + pressureRegion.rhs[(std::size_t)sid] * dx2) / (float)diag;
                                pressureRegion.tmp[(std::size_t)sid] =
                                    pressureRegion.pressure[(std::size_t)sid] + jacobiOmega * (target - pressureRegion.pressure[(std::size_t)sid]);
                            }
                        }
                    }
                    pressureRegion.pressure.swap(pressureRegion.tmp);
                    if (computeResidual() * dt <= params.pressureTol) break;
                }
            } else {
                for (int it = 0; it < maxIters; ++it) {
                    itUsed = it + 1;
                    for (int color = 0; color < 2; ++color) {
                        for (int k3 = 0; k3 < sz; ++k3) {
                            for (int j3 = 0; j3 < sy; ++j3) {
                                for (int i3 = 0; i3 < sx; ++i3) {
                                    if (((i3 + j3 + k3) & 1) != color) continue;
                                    const int sid = subIdx(i3, j3, k3);
                                    if (pressureRegion.solid[(std::size_t)sid] || !pressureRegion.fluid[(std::size_t)sid]) {
                                        pressureRegion.pressure[(std::size_t)sid] = 0.0f;
                                        continue;
                                    }

                                    float sum = 0.0f;
                                    int diag = 0;
                                    neighborContribution(i3 - 1, j3, k3, false, sum, diag);
                                    neighborContribution(i3 + 1, j3, k3, false, sum, diag);
                                    neighborContribution(i3, j3 - 1, k3, false, sum, diag);
                                    neighborContribution(i3, j3 + 1, k3, subOpenTop && (j3 + 1 >= sy), sum, diag);
                                    neighborContribution(i3, j3, k3 - 1, false, sum, diag);
                                    neighborContribution(i3, j3, k3 + 1, false, sum, diag);

                                    if (diag <= 0) {
                                        pressureRegion.pressure[(std::size_t)sid] = 0.0f;
                                        continue;
                                    }

                                    const float target = (sum + pressureRegion.rhs[(std::size_t)sid] * dx2) / (float)diag;
                                    pressureRegion.pressure[(std::size_t)sid] +=
                                        rbgsOmega * (target - pressureRegion.pressure[(std::size_t)sid]);
                                }
                            }
                        }
                    }
                    if (computeResidual() * dt <= params.pressureTol) break;
                }
            }

            componentIters = itUsed;
        }

        const auto solveEnd = clock::now();
        totalPressureMs += std::chrono::duration<float, std::milli>(solveEnd - solveStart).count();
        totalPressureIterations += componentIters;

        for (int sk = 0; sk < sz; ++sk) {
            const int gk = k0 + sk;
            for (int sj = 0; sj < sy; ++sj) {
                const int gj = j0 + sj;
                for (int si = 0; si < sx; ++si) {
                    const int gi = i0 + si;
                    const int gid = idxCell(gi, gj, gk);
                    const int sid = subIdx(si, sj, sk);
                    if (pressureRegion.fluid[(std::size_t)sid]) {
                        pressureTmp[(std::size_t)gid] = pressureRegion.pressure[(std::size_t)sid];
                    }
                }
            }
        }
    }

    if (componentCount == 0) {
        std::fill(pressure.begin(), pressure.end(), 0.0f);
        lastPressureSolveMs = 0.0f;
        lastPressureIterations = 0;
        pressureRegion.previousBoxValid = false;
        return;
    }

    pressure.swap(pressureTmp);
    lastPressureSolveMs = totalPressureMs;
    lastPressureIterations = totalPressureIterations;
    pressureRegion.previousBoxValid = false;

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

                const bool leftFluid = (i - 1 >= 0) ? isFluidCellGlobal(i - 1, j, k) : false;
                const bool rightFluid = (i < nx) ? isFluidCellGlobal(i, j, k) : false;
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

                const bool botFluid = (j - 1 >= 0) ? isFluidCellGlobal(i, j - 1, k) : false;
                const bool topFluid = (j < ny) ? isFluidCellGlobal(i, j, k) : false;
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

                const bool backFluid = (k - 1 >= 0) ? isFluidCellGlobal(i, j, k - 1) : false;
                const bool frontFluid = (k < nz) ? isFluidCellGlobal(i, j, k) : false;
                if (!backFluid && !frontFluid) continue;

                const float pBk = backFluid ? pressure[(std::size_t)idxCell(i, j, k - 1)] : 0.0f;
                const float pFr = frontFluid ? pressure[(std::size_t)idxCell(i, j, k)] : 0.0f;
                w[(std::size_t)id] -= scale * (pFr - pBk);
            }
        }
    }

    applyBoundary();
}
