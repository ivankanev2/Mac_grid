#pragma once
// Implementation header included by Sim/mac_water_sim.cpp
#include "water_common.h"

#include <algorithm>

// Apply A*x for the liquid-only pressure system.
// A approximates -∇² with:
//   - Neumann at solid boundaries (no contribution)
//   - Dirichlet p=0 at air/open boundaries
static inline void applyPressureA(
    const MACWater* w,
    const std::vector<float>& x,
    std::vector<float>& Ax)
{
    const int nx = w->nx;
    const int ny = w->ny;
    const float invDx2 = 1.0f / (w->dx * w->dx);

    auto isSolidCell = [&](int i, int j) -> bool {
        // Out-of-range: treat top as AIR if openTop, otherwise SOLID.
        if (i < 0 || i >= nx || j < 0 || j >= ny) {
            if (w->openTop && j == ny) return false; // open boundary (air)
            return true;
        }
        return w->solid[(size_t)w->idxP(i, j)] != 0;
    };

    auto isFluidCell = [&](int i, int j) -> bool {
        if (i < 0 || i >= nx || j < 0 || j >= ny) return false;
        const int id = w->idxP(i, j);
        return (w->solid[(size_t)id] == 0) && (w->liquid[(size_t)id] != 0);
    };

    std::fill(Ax.begin(), Ax.end(), 0.0f);

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int id = w->idxP(i, j);
            if (w->solid[(size_t)id] || !w->liquid[(size_t)id]) {
                Ax[(size_t)id] = 0.0f;
                continue;
            }

            float diag = 0.0f;
            float sumN = 0.0f;

            // Left
            if (!isSolidCell(i - 1, j)) {
                diag += invDx2;
                if (isFluidCell(i - 1, j)) sumN += invDx2 * x[(size_t)w->idxP(i - 1, j)];
            }
            // Right
            if (!isSolidCell(i + 1, j)) {
                diag += invDx2;
                if (isFluidCell(i + 1, j)) sumN += invDx2 * x[(size_t)w->idxP(i + 1, j)];
            }
            // Down
            if (!isSolidCell(i, j - 1)) {
                diag += invDx2;
                if (isFluidCell(i, j - 1)) sumN += invDx2 * x[(size_t)w->idxP(i, j - 1)];
            }
            // Up
            if (!isSolidCell(i, j + 1)) {
                diag += invDx2;
                if (isFluidCell(i, j + 1)) sumN += invDx2 * x[(size_t)w->idxP(i, j + 1)];
            }

            Ax[(size_t)id] = diag * x[(size_t)id] - sumN;
        }
    }
}

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

    diagInv.assign((size_t)Nc, 0.0f);
    pcg_r.assign((size_t)Nc, 0.0f);
    pcg_z.assign((size_t)Nc, 0.0f);
    pcg_d.assign((size_t)Nc, 0.0f);
    pcg_q.assign((size_t)Nc, 0.0f);
    pcg_Ap.assign((size_t)Nc, 0.0f);

    std::fill(p.begin(), p.end(), 0.0f);
    std::fill(rhs.begin(), rhs.end(), 0.0f);

    const float invDx = 1.0f / dx;
    const float invDx2 = invDx * invDx;
    const float invDt = 1.0f / std::max(1e-8f, dt);

    auto isSolidCell = [&](int i, int j) -> bool {
        if (i < 0 || i >= nx || j < 0 || j >= ny) {
            if (openTop && j == ny) return false; // open boundary (air)
            return true;
        }
        return solid[(size_t)idxP(i, j)] != 0;
    };

    // Build rhs and Jacobi preconditioner diagonal.
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int id = idxP(i, j);
            if (solid[(size_t)id] || !liquid[(size_t)id]) {
                rhs[(size_t)id] = 0.0f;
                diagInv[(size_t)id] = 0.0f;
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

            int nonSolid = 0;
            if (!isSolidCell(i - 1, j)) ++nonSolid;
            if (!isSolidCell(i + 1, j)) ++nonSolid;
            if (!isSolidCell(i, j - 1)) ++nonSolid;
            if (!isSolidCell(i, j + 1)) ++nonSolid;

            const float diag = (float)nonSolid * invDx2;
            diagInv[(size_t)id] = (diag > 1e-12f) ? (1.0f / diag) : 0.0f;
        }
    }

    // PCG solve
    applyPressureA(this, p, pcg_Ap);
    for (int id = 0; id < Nc; ++id) {
        if (solid[(size_t)id] || !liquid[(size_t)id]) {
            pcg_r[(size_t)id] = 0.0f;
            pcg_z[(size_t)id] = 0.0f;
            pcg_d[(size_t)id] = 0.0f;
            continue;
        }
        const float r = rhs[(size_t)id] - pcg_Ap[(size_t)id];
        pcg_r[(size_t)id] = r;
        pcg_z[(size_t)id] = diagInv[(size_t)id] * r;
        pcg_d[(size_t)id] = pcg_z[(size_t)id];
    }

    auto dotLiquid = [&](const std::vector<float>& a, const std::vector<float>& b) -> double {
        double s = 0.0;
        for (int id = 0; id < Nc; ++id) {
            if (solid[(size_t)id] || !liquid[(size_t)id]) continue;
            s += (double)a[(size_t)id] * (double)b[(size_t)id];
        }
        return s;
    };

    auto maxAbsLiquid = [&](const std::vector<float>& a) -> float {
        float m = 0.0f;
        for (int id = 0; id < Nc; ++id) {
            if (solid[(size_t)id] || !liquid[(size_t)id]) continue;
            m = std::max(m, std::fabs(a[(size_t)id]));
        }
        return m;
    };

    double deltaNew = dotLiquid(pcg_r, pcg_z);
    const float tol = std::max(0.0f, pressureTol);

    int iters = std::max(1, pressureMaxIters);
    for (int iter = 0; iter < iters; ++iter) {
        applyPressureA(this, pcg_d, pcg_q);

        const double denom = dotLiquid(pcg_d, pcg_q);
        if (!(denom > 1e-30)) break;

        const double alpha = deltaNew / denom;

        for (int id = 0; id < Nc; ++id) {
            if (solid[(size_t)id] || !liquid[(size_t)id]) continue;
            p[(size_t)id] += (float)(alpha * (double)pcg_d[(size_t)id]);
            pcg_r[(size_t)id] -= (float)(alpha * (double)pcg_q[(size_t)id]);
        }

        const float rInf = maxAbsLiquid(pcg_r);
        if (rInf <= tol) break;

        for (int id = 0; id < Nc; ++id) {
            if (solid[(size_t)id] || !liquid[(size_t)id]) continue;
            pcg_z[(size_t)id] = diagInv[(size_t)id] * pcg_r[(size_t)id];
        }

        const double deltaOld = deltaNew;
        deltaNew = dotLiquid(pcg_r, pcg_z);
        if (!(deltaNew > 0.0) || !(deltaOld > 0.0)) break;

        const double beta = deltaNew / deltaOld;
        for (int id = 0; id < Nc; ++id) {
            if (solid[(size_t)id] || !liquid[(size_t)id]) continue;
            pcg_d[(size_t)id] = pcg_z[(size_t)id] + (float)(beta * (double)pcg_d[(size_t)id]);
        }
    }

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