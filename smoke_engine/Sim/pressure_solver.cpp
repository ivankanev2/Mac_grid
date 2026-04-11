#include "pressure_solver.h"
#include <cmath>
#include <algorithm>
#include <cstdio>
#include "smoke_diagnostics.h"
#include <limits>

void PressureSolver::configure(int nx, int ny, float dx,
                               bool openTopBC,
                               const std::vector<uint8_t>& solidMask,
                               const std::vector<uint8_t>& fluidMask,
                               bool removeMeanForGauge,
                               const std::vector<float>* faceOpenU,
                               const std::vector<float>* faceOpenV)
{
    m_nx = nx; m_ny = ny;
    m_dx = dx;
    m_invDx2 = 1.0f / (dx * dx);
    m_openTopBC = openTopBC;

    m_removeMean = removeMeanForGauge;  

    const int N = nx * ny;
    m_solid = solidMask;
    m_fluid = fluidMask;

    for (int k = 0; k < N; ++k)
        if (m_solid[k]) m_fluid[k] = 0;

    // ----- Face openness (multiface) -----
    // If caller didn't provide openness, derive binary 0/1 openness from solidMask.
    m_faceOpenU.assign((size_t)(nx + 1) * (size_t)ny, 1.0f);
    m_faceOpenV.assign((size_t)nx * (size_t)(ny + 1), 1.0f);

    auto uIdx = [&](int i, int j) { return (size_t)j * (size_t)(nx + 1) + (size_t)i; }; // i in [0..nx], j in [0..ny-1]
    auto vIdx = [&](int i, int j) { return (size_t)j * (size_t)nx + (size_t)i; };       // i in [0..nx-1], j in [0..ny]

    if (faceOpenU && faceOpenU->size() == m_faceOpenU.size()) {
        m_faceOpenU = *faceOpenU;
    } else {
        // binary fallback from solids
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i <= nx; ++i) {
                bool blocked = false;
                if (i - 1 >= 0) blocked = blocked || (m_solid[(size_t)idx(i - 1, j, nx)] != 0);
                if (i < nx)     blocked = blocked || (m_solid[(size_t)idx(i,     j, nx)] != 0);
                m_faceOpenU[uIdx(i, j)] = blocked ? 0.0f : 1.0f;
            }
        }
    }

    if (faceOpenV && faceOpenV->size() == m_faceOpenV.size()) {
        m_faceOpenV = *faceOpenV;
    } else {
        // binary fallback from solids
        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                bool blocked = false;
                if (j - 1 >= 0) blocked = blocked || (m_solid[(size_t)idx(i, j - 1, nx)] != 0);
                if (j < ny)     blocked = blocked || (m_solid[(size_t)idx(i, j,     nx)] != 0);
                m_faceOpenV[vIdx(i, j)] = blocked ? 0.0f : 1.0f;
            }
        }
    }

    m_dirty = true;
    mgDirty = true;
}

void PressureSolver::ensurePCGBuffers()
{
    const int N = m_nx * m_ny;
    if ((int)m_r.size()  != N) m_r.resize(N);
    if ((int)m_z.size()  != N) m_z.resize(N);
    if ((int)m_d.size()  != N) m_d.resize(N);
    if ((int)m_q.size()  != N) m_q.resize(N);
    if ((int)m_Ap.size() != N) m_Ap.resize(N);
}

void PressureSolver::rebuildOperator()
{
    const int nx = m_nx, ny = m_ny;
    const int N  = nx * ny;

    m_L.assign((size_t)N, -1); m_R.assign((size_t)N, -1);
    m_B.assign((size_t)N, -1); m_T.assign((size_t)N, -1);

    m_wL.assign((size_t)N, 0.0f); m_wR.assign((size_t)N, 0.0f);
    m_wB.assign((size_t)N, 0.0f); m_wT.assign((size_t)N, 0.0f);

    m_diagW.assign((size_t)N, 0.0f);
    m_diagInv.assign((size_t)N, 0.0f);

    auto uIdx = [&](int i, int j) { return (size_t)j * (size_t)(nx + 1) + (size_t)i; };
    auto vIdx = [&](int i, int j) { return (size_t)j * (size_t)nx + (size_t)i; };

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int id = idx(i, j, nx);
            if (!m_fluid[(size_t)id]) continue;

            float diagW = 0.0f;

            auto addFace = [&](int ni, int nj, float wFace,
                               int& slotNbr, float& wNbr)
            {
                if (wFace <= 0.0f) return;

                // neighbor inside domain?
                if (ni >= 0 && nj >= 0 && ni < nx && nj < ny) {
                    const int nid = idx(ni, nj, nx);

                    if (m_solid[(size_t)nid]) {
                        // Neumann (solid) -> no contribution
                        return;
                    }

                    if (m_fluid[(size_t)nid]) {
                        // fluid neighbor -> offdiag + diag
                        slotNbr = nid;
                        wNbr = wFace;
                        diagW += wFace;
                    } else {
                        // air Dirichlet p=0 -> diag only
                        diagW += wFace;
                        return;
                    }
                } else {
                    // outside domain: Dirichlet p=0 if face is open
                    diagW += wFace;
                    return;
                }
            };

            // left/right use U faces
            addFace(i - 1, j, m_faceOpenU[uIdx(i,     j)], m_L[(size_t)id], m_wL[(size_t)id]);
            addFace(i + 1, j, m_faceOpenU[uIdx(i + 1, j)], m_R[(size_t)id], m_wR[(size_t)id]);

            // bottom/top use V faces
            addFace(i, j - 1, m_faceOpenV[vIdx(i, j)],     m_B[(size_t)id], m_wB[(size_t)id]);

            // top neighbor: if inside domain, regular. If outside and openTopBC, allow Dirichlet.
            if (j + 1 < ny) {
                addFace(i, j + 1, m_faceOpenV[vIdx(i, j + 1)], m_T[(size_t)id], m_wT[(size_t)id]);
            } else {
                // outside above top row
                if (m_openTopBC) {
                    addFace(i, j + 1, m_faceOpenV[vIdx(i, ny)], m_T[(size_t)id], m_wT[(size_t)id]);
                } else {
                    // closed boundary -> Neumann (no contribution)
                }
            }

            m_diagW[(size_t)id] = diagW;
            const float diag = diagW * m_invDx2;
            m_diagInv[(size_t)id] = (diag > 0.0f) ? (1.0f / diag) : 0.0f;
        }
    }

    m_dirty = false;
}

void PressureSolver::applyA(const std::vector<float>& x, std::vector<float>& Ax) const
{
    const int N = m_nx * m_ny;
    if ((int)Ax.size() != N) Ax.resize(N);

    for (int id = 0; id < N; ++id) {
        if (!m_fluid[(size_t)id]) { Ax[(size_t)id] = 0.0f; continue; }

        float sum = 0.0f;

        int n = m_L[(size_t)id]; if (n >= 0) sum += m_wL[(size_t)id] * x[(size_t)n];
        n = m_R[(size_t)id];     if (n >= 0) sum += m_wR[(size_t)id] * x[(size_t)n];
        n = m_B[(size_t)id];     if (n >= 0) sum += m_wB[(size_t)id] * x[(size_t)n];
        n = m_T[(size_t)id];     if (n >= 0) sum += m_wT[(size_t)id] * x[(size_t)n];

        const float diagW = m_diagW[(size_t)id];
        Ax[(size_t)id] = (diagW * x[(size_t)id] - sum) * m_invDx2;
    }
}

float PressureSolver::dotFluid(const std::vector<float>& a, const std::vector<float>& b) const
{
    double s = 0.0;
    const int N = m_nx * m_ny;
    for (int i = 0; i < N; ++i) {
        if (!m_fluid[i]) continue;
        s += (double)a[i] * (double)b[i];
    }
    return (float)s;
}

float PressureSolver::maxAbsFluid(const std::vector<float>& a) const
{
    float m = 0.0f;
    const int N = m_nx * m_ny;
    for (int i = 0; i < N; ++i) {
        if (!m_fluid[i]) continue;
        float v = std::fabs(a[i]);
        if (!std::isfinite(v)) return std::numeric_limits<float>::infinity();
        m = std::max(m, v);
    }
    return m;
}

void PressureSolver::removeMean(std::vector<float>& p) const
{
    if (!m_removeMean) return;
    if (m_openTopBC) return; // gauge is fixed by Dirichlet somewhere

    double sum = 0.0;
    int cnt = 0;
    const int N = m_nx * m_ny;
    for (int i = 0; i < N; ++i) {
        if (!m_fluid[i]) continue;
        sum += (double)p[i];
        cnt++;
    }
    if (cnt == 0) return;
    const float mean = (float)(sum / (double)cnt);
    for (int i = 0; i < N; ++i)
        if (m_fluid[i]) p[i] -= mean;
}

int PressureSolver::solvePCG(std::vector<float>& p,
                            const std::vector<float>& rhs,
                            int maxIters,
                            float tolPredDiv,
                            float dtForPredDiv)
{
    const int N = m_nx * m_ny;
    if (N <= 0) { m_lastIters = 0; return 0; }

    if (m_dirty) rebuildOperator();
    ensurePCGBuffers();

    if ((int)p.size()   != N) p.assign(N, 0.0f);
    if ((int)rhs.size() != N) { m_lastIters = 0; return 0; }

    const float tolRhs = tolPredDiv / std::max(1e-8f, dtForPredDiv);

    // Early out: RHS tiny in predicted-divergence units
    const float bInf = maxAbsFluid(rhs);
    if (!std::isfinite(bInf) || bInf <= tolRhs) {
        m_lastIters = 0;
        removeMean(p);
        return 0;
    }

    // r = b - A p  (warm start)
    applyA(p, m_Ap);
    for (int i = 0; i < N; ++i) m_r[i] = rhs[i] - m_Ap[i];

    const float rInf0 = std::max(maxAbsFluid(m_r), 1e-30f);

    // z = M^-1 r (Jacobi)
    for (int i = 0; i < N; ++i) m_z[i] = m_r[i] * m_diagInv[i];
    m_d = m_z;

    float deltaNew = dotFluid(m_r, m_z);
    if (!std::isfinite(deltaNew) || deltaNew <= 1e-30f) {
        m_lastIters = 0;
        removeMean(p);
        return 0;
    }

    // Keep a mild relative stop too (prevents wasted late iters)
    const float relTol = 1e-3f;

    int it_used = 0;

    for (int it = 0; it < maxIters; ++it) {
        it_used = it + 1;

        applyA(m_d, m_q);
        const float dq = dotFluid(m_d, m_q);
        if (!std::isfinite(dq) || std::fabs(dq) < 1e-30f) break;

        const float alpha = deltaNew / dq;

        for (int i = 0; i < N; ++i) {
            p[i]     += alpha * m_d[i];
            m_r[i]   -= alpha * m_q[i];
        }

        const float rInf = maxAbsFluid(m_r);
        if (!std::isfinite(rInf)) break;

        // Stop in RHS units
        if (rInf <= tolRhs) break;

        // Also stop relative to initial residual (cheap + robust)
        if (rInf <= relTol * rInf0) break;

        // z = M^-1 r
        for (int i = 0; i < N; ++i) m_z[i] = m_r[i] * m_diagInv[i];

        const float deltaOld = deltaNew;
        deltaNew = dotFluid(m_r, m_z);

        SMOKE_DIAG_PRINTF("[PCG] deltaNew=%.6g\n", deltaNew);

        if (!std::isfinite(deltaNew) || deltaNew <= 1e-30f) break;

        const float beta = deltaNew / (deltaOld + 1e-30f);
        for (int i = 0; i < N; ++i)
            m_d[i] = m_z[i] + beta * m_d[i];
    }

    removeMean(p);
    m_lastIters = it_used;
    return it_used;
}

void PressureSolver::ensureMultigrid()
{
    if (!mgDirty && mgBuiltValid &&
        mgBuiltOpenTop == m_openTopBC &&
        mgBuiltNx == m_nx && mgBuiltNy == m_ny)
        return;

    mgDirty = false;
    mgBuiltValid = true;
    mgBuiltOpenTop = m_openTopBC;
    mgBuiltNx = m_nx;
    mgBuiltNy = m_ny;

    mgLevels.clear();
    mgLevels.reserve((size_t)mgMaxLevels);

    // ----- Level 0 -----
    {
        MGLevel L0;
        L0.nx = m_nx; L0.ny = m_ny;
        L0.invDx2 = 1.0f / (m_dx * m_dx);

        const int N = m_nx * m_ny;
        L0.solid.assign((size_t)N, 0);
        L0.fluid.assign((size_t)N, 0);

        for (int j = 0; j < m_ny; ++j)
        for (int i = 0; i < m_nx; ++i) {
            const int id  = idx(i, j, m_nx);
            L0.solid[(size_t)id] = m_solid[(size_t)id] ? 1 : 0;
            L0.fluid[(size_t)id] = (m_solid[(size_t)id]==0 && m_fluid[(size_t)id]!=0) ? 1 : 0;
        }

        L0.L.assign((size_t)N, -1); L0.R.assign((size_t)N, -1);
        L0.B.assign((size_t)N, -1); L0.T.assign((size_t)N, -1);

        L0.wL.assign((size_t)N, 0.0f); L0.wR.assign((size_t)N, 0.0f);
        L0.wB.assign((size_t)N, 0.0f); L0.wT.assign((size_t)N, 0.0f);

        L0.diagW.assign((size_t)N, 0.0f);
        L0.diagInv.assign((size_t)N, 0.0f);

        L0.x.assign((size_t)N, 0.0f);
        L0.b.assign((size_t)N, 0.0f);
        L0.Ax.assign((size_t)N, 0.0f);
        L0.r.assign((size_t)N, 0.0f);

        for (int j = 0; j < m_ny; ++j) {
            for (int i = 0; i < m_nx; ++i) {
                const int id  = idx(i, j, m_nx);
                if (!L0.fluid[(size_t)id]) {
                    L0.diagW[(size_t)id] = 0.0f;
                    L0.diagInv[(size_t)id] = 0.0f;
                    continue;
                }
                float diagW = 0.0f;

                auto uIdx0 = [&](int i, int j) { return (size_t)j * (size_t)(m_nx + 1) + (size_t)i; };
                auto vIdx0 = [&](int i, int j) { return (size_t)j * (size_t)m_nx + (size_t)i; };

                auto addFace0 = [&](int ni, int nj, float wFace,
                                    int& slotNbr, float& wNbr)
                {
                    if (wFace <= 0.0f) return;

                    if (ni >= 0 && nj >= 0 && ni < m_nx && nj < m_ny) {
                        const int nid = idx(ni, nj, m_nx);
                        if (L0.solid[(size_t)nid]) return;

                        if (L0.fluid[(size_t)nid]) {
                            slotNbr = nid;
                            wNbr = wFace;
                            diagW += wFace;
                        } else {
                            diagW += wFace; // air Dirichlet
                        }
                    } else {
                        // outside domain: Dirichlet p=0 if face is open
                        diagW += wFace;
                        return;
                    }
                };

                // left/right
                addFace0(i - 1, j, m_faceOpenU[uIdx0(i,     j)], L0.L[(size_t)id], L0.wL[(size_t)id]);
                addFace0(i + 1, j, m_faceOpenU[uIdx0(i + 1, j)], L0.R[(size_t)id], L0.wR[(size_t)id]);

                // bottom
                addFace0(i, j - 1, m_faceOpenV[vIdx0(i, j)],     L0.B[(size_t)id], L0.wB[(size_t)id]);

                // top
                if (j + 1 < m_ny) {
                    addFace0(i, j + 1, m_faceOpenV[vIdx0(i, j + 1)], L0.T[(size_t)id], L0.wT[(size_t)id]);
                } else {
                    if (m_openTopBC) {
                        addFace0(i, j + 1, m_faceOpenV[vIdx0(i, m_ny)], L0.T[(size_t)id], L0.wT[(size_t)id]);
                    }
                }

                L0.diagW[(size_t)id] = diagW;
                const float diag = diagW * L0.invDx2;
                L0.diagInv[(size_t)id] = (diag > 0.0f) ? (1.0f / diag) : 0.0f;
            }
        }

        mgLevels.push_back(std::move(L0));
    }

    // ----- Coarser levels -----
    while ((int)mgLevels.size() < mgMaxLevels) {
        const MGLevel& F = mgLevels.back();
        if (F.nx <= 4 || F.ny <= 4) break;

        int cnx = F.nx / 2;
        int cny = F.ny / 2;
        if (cnx < 2 || cny < 2) break;

        MGLevel C;
        C.nx = cnx; C.ny = cny;
        C.invDx2 = F.invDx2 * 0.25f;

        const int CN = cnx * cny;
        C.solid.assign((size_t)CN, 0);
        C.fluid.assign((size_t)CN, 0);
        C.L.assign((size_t)CN, -1); C.R.assign((size_t)CN, -1);
        C.B.assign((size_t)CN, -1); C.T.assign((size_t)CN, -1);
        
        C.wL.assign((size_t)CN, 0.0f); C.wR.assign((size_t)CN, 0.0f);
        C.wB.assign((size_t)CN, 0.0f); C.wT.assign((size_t)CN, 0.0f);

        C.diagW.assign((size_t)CN, 0.0f);
        C.diagInv.assign((size_t)CN, 0.0f);

        C.x.assign((size_t)CN, 0.0f);
        C.b.assign((size_t)CN, 0.0f);
        C.Ax.assign((size_t)CN, 0.0f);
        C.r.assign((size_t)CN, 0.0f);

        // Build coarse solid/fluid from 2x2 blocks
        for (int J = 0; J < cny; ++J) {
            for (int I = 0; I < cnx; ++I) {
                int fi = 2*I, fj = 2*J;

                bool allSolid = true;
                bool allFluid = true;   // <- NEW: require full 2x2 to be fluid on coarse
                bool anyFluid = false;

                for (int dj = 0; dj < 2; ++dj) {
                    for (int di = 0; di < 2; ++di) {
                        int ii = fi + di;
                        int jj = fj + dj;

                        // Outside fine grid => treat as not-fluid (breaks allFluid)
                        if (ii < 0 || jj < 0 || ii >= F.nx || jj >= F.ny) {
                            allFluid = false;
                            continue;
                        }

                        const int fid = mgIdx(ii, jj, F.nx);

                        if (!F.solid[fid]) allSolid = false;

                        if (F.fluid[fid]) {
                            anyFluid = true;
                        } else {
                            allFluid = false;
                        }
                    }
                }

                const int cid = mgIdx(I, J, cnx);
                C.solid[cid] = allSolid ? 1 : 0;

                // IMPORTANT: only keep fluid on coarse if it's fully fluid on fine
                C.fluid[cid] = (!allSolid && allFluid) ? 1 : 0;
            }
        }

        // Stencil
        for (int J = 0; J < cny; ++J) {
            for (int I = 0; I < cnx; ++I) {
                const int cid = mgIdx(I, J, cnx);
                if (!C.fluid[(size_t)cid]) {
                    C.diagW[(size_t)cid] = 0.0f;
                    C.diagInv[(size_t)cid] = 0.0f;
                    continue;
                }

                float diagW = 0.0f;

                auto addNbr = [&](int ni, int nj, int& slot, float& wSlot) {
                    const int nid = mgIdx(ni, nj, cnx);
                    if (C.solid[(size_t)nid]) return;

                    if (C.fluid[(size_t)nid]) {
                        slot = nid;
                        wSlot = 1.0f;
                        diagW += 1.0f;
                    } else {
                        diagW += 1.0f; // air Dirichlet
                    }
                };

                if (I > 0)        addNbr(I - 1, J, C.L[(size_t)cid], C.wL[(size_t)cid]);
                if (I + 1 < cnx)  addNbr(I + 1, J, C.R[(size_t)cid], C.wR[(size_t)cid]);
                if (J > 0)        addNbr(I, J - 1, C.B[(size_t)cid], C.wB[(size_t)cid]);
                if (J + 1 < cny)  addNbr(I, J + 1, C.T[(size_t)cid], C.wT[(size_t)cid]);

                if (m_openTopBC && J == cny - 1) {
                    // top outside treated as air Dirichlet
                    diagW += 1.0f;
                }

                C.diagW[(size_t)cid] = diagW;
                const float diag = diagW * C.invDx2;
                C.diagInv[(size_t)cid] = (diag > 0.0f) ? (1.0f / diag) : 0.0f;
            }
        }

        mgLevels.push_back(std::move(C));
    }

    if (!mgLevels.empty()) {
        buildDirectCoarseSolve(mgLevels.back());
    }
}

void PressureSolver::mgApplyA(int lev, const std::vector<float>& x, std::vector<float>& Ax) const
{
    const MGLevel& L = mgLevels[(size_t)lev];
    const int N = L.nx * L.ny;
    if ((int)Ax.size() != N) Ax.resize((size_t)N);

    for (int id = 0; id < N; ++id) {
        if (!L.fluid[(size_t)id]) { Ax[(size_t)id] = 0.0f; continue; }

        float sum = 0.0f;
        int n = L.L[(size_t)id]; if (n >= 0) sum += L.wL[(size_t)id] * x[(size_t)n];
        n = L.R[(size_t)id];     if (n >= 0) sum += L.wR[(size_t)id] * x[(size_t)n];
        n = L.B[(size_t)id];     if (n >= 0) sum += L.wB[(size_t)id] * x[(size_t)n];
        n = L.T[(size_t)id];     if (n >= 0) sum += L.wT[(size_t)id] * x[(size_t)n];

        const float diagW = L.diagW[(size_t)id];
        Ax[(size_t)id] = (diagW * x[(size_t)id] - sum) * L.invDx2;
    }
}

bool PressureSolver::mgDirectSolve(int lev)
{
    MGLevel& L = mgLevels[(size_t)lev];
    if (!L.directSolveValid) return false;

    const int n = (int)L.directSolveCells.size();
    if (n <= 0) {
        std::fill(L.x.begin(), L.x.end(), 0.0f);
        return true;
    }

    if ((int)L.directSolveScratch0.size() != n) L.directSolveScratch0.assign((size_t)n, 0.0f);
    if ((int)L.directSolveScratch1.size() != n) L.directSolveScratch1.assign((size_t)n, 0.0f);

    std::fill(L.x.begin(), L.x.end(), 0.0f);

    float* const y = L.directSolveScratch0.data();
    float* const xCompact = L.directSolveScratch1.data();
    const float* const chol = L.directSolveCholesky.data();

    for (int row = 0; row < n; ++row) {
        const int cell = L.directSolveCells[(size_t)row];
        y[row] = L.b[(size_t)cell];
        xCompact[row] = 0.0f;
    }
    if (L.directSolveAnchorsGauge && n > 0) y[0] = 0.0f;

    for (int row = 0; row < n; ++row) {
        float sum = y[row];
        const size_t rowBase = (size_t)row * (size_t)n;
        for (int col = 0; col < row; ++col) {
            sum -= chol[rowBase + (size_t)col] * y[col];
        }
        const float diag = chol[rowBase + (size_t)row];
        if (!(diag > 0.0f) || !std::isfinite(diag)) return false;
        y[row] = sum / diag;
    }

    for (int row = n - 1; row >= 0; --row) {
        float sum = y[row];
        for (int col = row + 1; col < n; ++col) {
            sum -= chol[(size_t)col * (size_t)n + (size_t)row] * xCompact[col];
        }
        const float diag = chol[(size_t)row * (size_t)n + (size_t)row];
        if (!(diag > 0.0f) || !std::isfinite(diag)) return false;
        xCompact[row] = sum / diag;
    }

    if (L.directSolveAnchorsGauge && n > 0) xCompact[0] = 0.0f;

    for (int row = 0; row < n; ++row) {
        const int cell = L.directSolveCells[(size_t)row];
        L.x[(size_t)cell] = xCompact[row];
    }

    return true;
}

void PressureSolver::buildDirectCoarseSolve(MGLevel& L) const
{
    L.directSolveValid = false;
    L.directSolveAnchorsGauge = false;
    L.directSolveCells.clear();
    L.directSolveCompactIndex.clear();
    L.directSolveCholesky.clear();
    L.directSolveScratch0.clear();
    L.directSolveScratch1.clear();

    const int totalCells = L.nx * L.ny;
    if (totalCells <= 0) return;

    int n = 0;
    for (int id = 0; id < totalCells; ++id) {
        if (L.fluid[(size_t)id]) ++n;
    }
    if (n <= 0) return;

    L.directSolveCells.resize((size_t)n);
    L.directSolveCompactIndex.assign((size_t)totalCells, -1);
    int row = 0;
    for (int id = 0; id < totalCells; ++id) {
        if (!L.fluid[(size_t)id]) continue;
        L.directSolveCells[(size_t)row] = id;
        L.directSolveCompactIndex[(size_t)id] = row;
        ++row;
    }

    std::vector<float> dense((size_t)n * (size_t)n, 0.0f);
    auto addNbr = [&](int rowIndex, int nbrCell, float weight) {
        if (weight <= 0.0f || nbrCell < 0 || nbrCell >= totalCells) return;
        const int col = L.directSolveCompactIndex[(size_t)nbrCell];
        if (col < 0) return;
        dense[(size_t)rowIndex * (size_t)n + (size_t)col] -= weight * L.invDx2;
    };

    for (int rowIndex = 0; rowIndex < n; ++rowIndex) {
        const int cell = L.directSolveCells[(size_t)rowIndex];
        dense[(size_t)rowIndex * (size_t)n + (size_t)rowIndex] = L.diagW[(size_t)cell] * L.invDx2;
        addNbr(rowIndex, L.L[(size_t)cell], L.wL[(size_t)cell]);
        addNbr(rowIndex, L.R[(size_t)cell], L.wR[(size_t)cell]);
        addNbr(rowIndex, L.B[(size_t)cell], L.wB[(size_t)cell]);
        addNbr(rowIndex, L.T[(size_t)cell], L.wT[(size_t)cell]);
    }

    const bool anchorGauge = (!m_openTopBC && m_removeMean);
    if (anchorGauge && n > 0) {
        for (int j = 0; j < n; ++j) {
            dense[(size_t)j] = 0.0f;
            dense[(size_t)j * (size_t)n] = 0.0f;
        }
        dense[0] = 1.0f;
    }

    L.directSolveCholesky = dense;
    for (int rowIndex = 0; rowIndex < n; ++rowIndex) {
        const size_t rowBase = (size_t)rowIndex * (size_t)n;
        for (int col = 0; col <= rowIndex; ++col) {
            float sum = L.directSolveCholesky[rowBase + (size_t)col];
            for (int k = 0; k < col; ++k) {
                sum -= L.directSolveCholesky[rowBase + (size_t)k] *
                       L.directSolveCholesky[(size_t)col * (size_t)n + (size_t)k];
            }
            if (rowIndex == col) {
                if (!(sum > 1.0e-9f) || !std::isfinite(sum)) {
                    L.directSolveCholesky.clear();
                    return;
                }
                L.directSolveCholesky[rowBase + (size_t)col] = std::sqrt(sum);
            } else {
                const float diag = L.directSolveCholesky[(size_t)col * (size_t)n + (size_t)col];
                if (!(diag > 0.0f) || !std::isfinite(diag)) {
                    L.directSolveCholesky.clear();
                    return;
                }
                L.directSolveCholesky[rowBase + (size_t)col] = sum / diag;
            }
        }
    }

    L.directSolveScratch0.assign((size_t)n, 0.0f);
    L.directSolveScratch1.assign((size_t)n, 0.0f);
    L.directSolveAnchorsGauge = anchorGauge;
    L.directSolveValid = true;
}

void PressureSolver::mgSmoothRBGS(int lev, int iters)
{
    MGLevel& L = mgLevels[(size_t)lev];
    const int nx = L.nx;
    const int ny = L.ny;

    float omega = 1.0f;
    if (mgUseSOR) omega = std::max(1.0f, std::min(mgSORomega, 1.9f));
    const float oneMinusOmega = 1.0f - omega;
    const float invDx2 = L.invDx2;

    float* const x = L.x.data();
    const float* const b = L.b.data();
    const uint8_t* const fluid = L.fluid.data();
    const int* const left = L.L.data();
    const int* const right = L.R.data();
    const int* const bottom = L.B.data();
    const int* const top = L.T.data();
    const float* const wL = L.wL.data();
    const float* const wR = L.wR.data();
    const float* const wB = L.wB.data();
    const float* const wT = L.wT.data();
    const float* const diagInv = L.diagInv.data();

    for (int it = 0; it < iters; ++it) {
        for (int color = 0; color < 2; ++color) {
            for (int j = 0; j < ny; ++j) {
                const int i0 = (color - (j & 1)) & 1;
                for (int i = i0; i < nx; i += 2) {
                    const int id = mgIdx(i, j, nx);
                    if (!fluid[(size_t)id]) continue;

                    float sum = 0.0f;
                    int n = left[(size_t)id];   if (n >= 0) sum += wL[(size_t)id] * x[(size_t)n];
                    n = right[(size_t)id];      if (n >= 0) sum += wR[(size_t)id] * x[(size_t)n];
                    n = bottom[(size_t)id];     if (n >= 0) sum += wB[(size_t)id] * x[(size_t)n];
                    n = top[(size_t)id];        if (n >= 0) sum += wT[(size_t)id] * x[(size_t)n];

                    const float invDiag = diagInv[(size_t)id];
                    if (invDiag <= 0.0f) continue;

                    const float xGs = invDiag * (b[(size_t)id] + sum * invDx2);
                    x[(size_t)id] = oneMinusOmega * x[(size_t)id] + omega * xGs;
                }
            }
        }
    }
}

void PressureSolver::mgComputeResidual(int lev)
{
    MGLevel& L = mgLevels[(size_t)lev];
    const int N = L.nx * L.ny;
    const float invDx2 = L.invDx2;

    const float* const x = L.x.data();
    const float* const b = L.b.data();
    const uint8_t* const fluid = L.fluid.data();
    const int* const left = L.L.data();
    const int* const right = L.R.data();
    const int* const bottom = L.B.data();
    const int* const top = L.T.data();
    const float* const wL = L.wL.data();
    const float* const wR = L.wR.data();
    const float* const wB = L.wB.data();
    const float* const wT = L.wT.data();
    const float* const diagW = L.diagW.data();
    float* const r = L.r.data();

    for (int id = 0; id < N; ++id) {
        if (!fluid[(size_t)id]) { r[(size_t)id] = 0.0f; continue; }

        float sum = 0.0f;
        int n = left[(size_t)id];   if (n >= 0) sum += wL[(size_t)id] * x[(size_t)n];
        n = right[(size_t)id];      if (n >= 0) sum += wR[(size_t)id] * x[(size_t)n];
        n = bottom[(size_t)id];     if (n >= 0) sum += wB[(size_t)id] * x[(size_t)n];
        n = top[(size_t)id];        if (n >= 0) sum += wT[(size_t)id] * x[(size_t)n];

        r[(size_t)id] = b[(size_t)id] - (diagW[(size_t)id] * x[(size_t)id] - sum) * invDx2;
    }
}

void PressureSolver::mgRestrictResidual(int fineLev)
{
    MGLevel& F = mgLevels[(size_t)fineLev];
    MGLevel& C = mgLevels[(size_t)fineLev + 1];

    std::fill(C.b.begin(), C.b.end(), 0.0f);
    std::fill(C.x.begin(), C.x.end(), 0.0f);

    auto addSample = [&](int fi, int fj, float w, float& sum, float& wsum) {
        if (fi < 0 || fj < 0 || fi >= F.nx || fj >= F.ny) return;
        const int fid = mgIdx(fi, fj, F.nx);
        if (!F.fluid[(size_t)fid]) return;
        sum  += w * F.r[(size_t)fid];
        wsum += w;
    };

    for (int J = 0; J < C.ny; ++J) {
        for (int I = 0; I < C.nx; ++I) {
            const int cid = mgIdx(I, J, C.nx);
            if (!C.fluid[(size_t)cid]) { C.b[(size_t)cid] = 0.0f; continue; }

            const int fi = 2 * I;
            const int fj = 2 * J;

            float sum = 0.0f, wsum = 0.0f;

            addSample(fi,   fj,   4.0f, sum, wsum);
            addSample(fi-1, fj,   2.0f, sum, wsum);
            addSample(fi+1, fj,   2.0f, sum, wsum);
            addSample(fi,   fj-1, 2.0f, sum, wsum);
            addSample(fi,   fj+1, 2.0f, sum, wsum);
            addSample(fi-1, fj-1, 1.0f, sum, wsum);
            addSample(fi+1, fj-1, 1.0f, sum, wsum);
            addSample(fi-1, fj+1, 1.0f, sum, wsum);
            addSample(fi+1, fj+1, 1.0f, sum, wsum);

            C.b[(size_t)cid] = (wsum > 0.0f) ? (sum / wsum) : 0.0f;
        }
    }
}

void PressureSolver::mgProlongateAndAdd(int coarseLev)
{
    MGLevel& C = mgLevels[(size_t)coarseLev];
    MGLevel& F = mgLevels[(size_t)coarseLev - 1];

    auto cAt = [&](int I, int J) -> float {
        I = std::max(0, std::min(I, C.nx - 1));
        J = std::max(0, std::min(J, C.ny - 1));
        const int cid = mgIdx(I, J, C.nx);
        return C.fluid[(size_t)cid] ? C.x[(size_t)cid] : 0.0f;
    };

    for (int fj = 0; fj < F.ny; ++fj) {
        for (int fi = 0; fi < F.nx; ++fi) {
            const int fid = mgIdx(fi, fj, F.nx);
            if (!F.fluid[(size_t)fid]) continue;

            const int I = fi >> 1;
            const int J = fj >> 1;
            const int ox = fi & 1;
            const int oy = fj & 1;

            float e = 0.0f;
            if (ox == 0 && oy == 0) e = cAt(I, J);
            else if (ox == 1 && oy == 0) e = 0.5f * (cAt(I, J) + cAt(I + 1, J));
            else if (ox == 0 && oy == 1) e = 0.5f * (cAt(I, J) + cAt(I, J + 1));
            else e = 0.25f * (cAt(I, J) + cAt(I + 1, J) + cAt(I, J + 1) + cAt(I + 1, J + 1));

            F.x[(size_t)fid] += e;
        }
    }
}

void PressureSolver::mgVCycle(int lev)
{
    if (lev == (int)mgLevels.size() - 1) {
        if (!mgDirectSolve(lev)) {
            mgSmoothRBGS(lev, mgCoarseIters);
        }
        return;
    }

    mgSmoothRBGS(lev, mgPreSmooth);
    mgComputeResidual(lev);
    mgRestrictResidual(lev);

    mgVCycle(lev + 1);

    mgProlongateAndAdd(lev + 1);
    mgSmoothRBGS(lev, mgPostSmooth);
}

void PressureSolver::applyMGPrecond(const std::vector<float>& r, std::vector<float>& z)
{
    ensureMultigrid();
    if (mgLevels.empty()) { z = r; return; }

    MGLevel& F = mgLevels[0];
    const int N = F.nx * F.ny;
    if ((int)z.size() != N) z.resize((size_t)N);

    F.b = r;
    std::fill(F.x.begin(), F.x.end(), 0.0f);

    for (int k = 0; k < mgVcyclesPerApply; ++k)
        mgVCycle(0);

    z = F.x;
    if (m_removeMean && !m_openTopBC) removeMean(z);
}

void PressureSolver::solveMG(std::vector<float>& p,
                            const std::vector<float>& rhs,
                            int maxVCycles,
                            float tolPredDiv,
                            float dt)
{
    ensureMultigrid();
    if (mgLevels.empty()) {
        solvePCG(p, rhs, 80, tolPredDiv, dt);
        return;
    }

    if (m_dirty) rebuildOperator();
    ensurePCGBuffers();

    MGLevel& F = mgLevels[0];
    const int N = F.nx * F.ny;
    if ((int)p.size() != N) p.assign((size_t)N, 0.0f);
    if ((int)rhs.size() != N) {
        m_lastIters = 0;
        return;
    }

    const float tolRhs = tolPredDiv / std::max(1e-8f, dt);
    const float relTol = std::max(0.0f, mgRelativeTol);

    auto maxAbsOnFluid = [&](const std::vector<float>& values) -> float {
        float m = 0.0f;
        for (int id = 0; id < N; ++id) {
            if (!F.fluid[(size_t)id]) continue;
            const float a = std::fabs(values[(size_t)id]);
            if (!std::isfinite(a)) return std::numeric_limits<float>::infinity();
            m = std::max(m, a);
        }
        return m;
    };

    const float bInf = maxAbsOnFluid(rhs);
    if (!(bInf > 0.0f) || bInf <= tolRhs) {
        if (m_removeMean && !m_openTopBC) removeMean(p);
        m_lastIters = 0;
        return;
    }

    applyA(p, m_Ap);
    for (int id = 0; id < N; ++id) {
        m_r[(size_t)id] = rhs[(size_t)id] - m_Ap[(size_t)id];
    }

    float rInf = maxAbsOnFluid(m_r);
    if (!(rInf > 0.0f) || rInf <= tolRhs) {
        if (m_removeMean && !m_openTopBC) removeMean(p);
        m_lastIters = 0;
        return;
    }

    const float rInf0 = std::max(rInf, 1.0e-30f);

    applyMGPrecond(m_r, m_z);
    m_d = m_z;

    float deltaNew = dotFluid(m_r, m_z);
    if (!std::isfinite(deltaNew) || deltaNew <= 1.0e-30f) {
        solvePCG(p, rhs, 80, tolPredDiv, dt);
        return;
    }

    bool fallbackPCG = false;
    int itUsed = 0;
    for (int it = 0; it < std::max(1, maxVCycles); ++it) {
        itUsed = it + 1;

        applyA(m_d, m_q);
        const float dq = dotFluid(m_d, m_q);
        if (!std::isfinite(dq) || std::fabs(dq) < 1.0e-30f) {
            fallbackPCG = true;
            break;
        }

        const float alpha = deltaNew / dq;
        for (int id = 0; id < N; ++id) {
            p[(size_t)id] += alpha * m_d[(size_t)id];
            m_r[(size_t)id] -= alpha * m_q[(size_t)id];
        }

        rInf = maxAbsOnFluid(m_r);
        if (!std::isfinite(rInf)) {
            fallbackPCG = true;
            break;
        }
        if (rInf <= tolRhs) break;
        if (relTol > 0.0f && rInf <= relTol * rInf0) break;

        applyMGPrecond(m_r, m_z);
        const float deltaOld = deltaNew;
        deltaNew = dotFluid(m_r, m_z);
        if (!std::isfinite(deltaNew) || deltaNew <= 1.0e-30f) {
            fallbackPCG = true;
            break;
        }

        const float beta = deltaNew / (deltaOld + 1.0e-30f);
        for (int id = 0; id < N; ++id) {
            m_d[(size_t)id] = m_z[(size_t)id] + beta * m_d[(size_t)id];
        }
    }

    if (m_removeMean && !m_openTopBC) removeMean(p);
    m_lastIters = itUsed;

    if (fallbackPCG) {
        solvePCG(p, rhs, 80, tolPredDiv, dt);
    }
}

