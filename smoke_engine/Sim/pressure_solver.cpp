#include "pressure_solver.h"
#include <cmath>
#include <algorithm>
#include <limits>

void PressureSolver::configure(int nx, int ny, float dx,
                               bool openTopBC,
                               const std::vector<uint8_t>& solidMask,
                               const std::vector<uint8_t>& fluidMask,
                               bool removeMeanForGauge)
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

    m_L.assign(N, -1); m_R.assign(N, -1);
    m_B.assign(N, -1); m_T.assign(N, -1);
    m_count.assign(N, 0);
    m_diagInv.assign(N, 0.0f);

    auto isSolid = [&](int i, int j) { return m_solid[idx(i,j,nx)] != 0; };
    auto isFluid = [&](int i, int j) { return m_fluid[idx(i,j,nx)] != 0; };

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int id = idx(i,j,nx);
            if (!isFluid(i,j)) continue;

            int count = 0;

            auto addNbr = [&](int ni, int nj, int& slot) {
                const int nid = idx(ni,nj,nx);
                if (m_fluid[nid]) { slot = nid; count++; }
                else if (!m_solid[nid]) { count++; } // air Dirichlet (p=0)
            };

            if (i > 0)      addNbr(i-1, j, m_L[id]);
            if (i+1 < nx)   addNbr(i+1, j, m_R[id]);
            if (j > 0)      addNbr(i, j-1, m_B[id]);
            if (j+1 < ny)   addNbr(i, j+1, m_T[id]);

            if (m_openTopBC && j == ny-1) {
                // neighbor above top row acts like air (Dirichlet)
                count++;
            }

            m_count[id] = (uint8_t)count;
            const float diag = (float)count * m_invDx2;
            m_diagInv[id] = (diag > 0.0f) ? (1.0f / diag) : 0.0f;
        }
    }

    m_dirty = false;
}

void PressureSolver::applyA(const std::vector<float>& x, std::vector<float>& Ax) const
{
    const int N = m_nx * m_ny;
    if ((int)Ax.size() != N) Ax.resize(N);

    for (int id = 0; id < N; ++id) {
        if (!m_fluid[id]) { Ax[id] = 0.0f; continue; }

        float sum = 0.0f;
        int n = m_L[id]; if (n >= 0) sum += x[n];
            n = m_R[id]; if (n >= 0) sum += x[n];
            n = m_B[id]; if (n >= 0) sum += x[n];
            n = m_T[id]; if (n >= 0) sum += x[n];

        const int count = (int)m_count[id];
        Ax[id] = (count * x[id] - sum) * m_invDx2;
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

    // Early out: RHS tiny in predDiv units
    const float bInf = maxAbsFluid(rhs);
    if (!std::isfinite(bInf) || bInf * dtForPredDiv <= tolPredDiv) {
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

        // Stop in "predDiv" space
        if (rInf * dtForPredDiv <= tolPredDiv) break;

        // Also stop relative to initial residual (cheap + robust)
        if (rInf <= relTol * rInf0) break;

        // z = M^-1 r
        for (int i = 0; i < N; ++i) m_z[i] = m_r[i] * m_diagInv[i];

        const float deltaOld = deltaNew;
        deltaNew = dotFluid(m_r, m_z);
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
        L0.diagCount.assign((size_t)N, 0);
        L0.diagInv.assign((size_t)N, 0.0f);

        L0.x.assign((size_t)N, 0.0f);
        L0.b.assign((size_t)N, 0.0f);
        L0.Ax.assign((size_t)N, 0.0f);
        L0.r.assign((size_t)N, 0.0f);

        for (int j = 0; j < m_ny; ++j) {
            for (int i = 0; i < m_nx; ++i) {
                const int id  = idx(i, j, m_nx);
                if (!L0.fluid[(size_t)id]) continue;

                int count = 0;

                auto addNbr = [&](int ni, int nj, int& slot) {
                    const int nid = idx(ni, nj, m_nx);
                    if (L0.fluid[(size_t)nid]) { slot = nid; count++; }
                    else if (!L0.solid[(size_t)nid]) { count++; } // air Dirichlet
                };

                if (i > 0)        addNbr(i - 1, j, L0.L[(size_t)id]);
                if (i + 1 < m_nx) addNbr(i + 1, j, L0.R[(size_t)id]);
                if (j > 0)        addNbr(i, j - 1, L0.B[(size_t)id]);
                if (j + 1 < m_ny) addNbr(i, j + 1, L0.T[(size_t)id]);

                if (m_openTopBC && j == m_ny - 1) count++;

                L0.diagCount[(size_t)id] = (uint8_t)count;
                const float diag = (float)count * L0.invDx2;
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
        C.diagCount.assign((size_t)CN, 0);
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
                const int id = mgIdx(I,J,cnx);
                if (!C.fluid[(size_t)id]) continue;

                int count = 0;
                auto addNbrC = [&](int NI, int NJ, int& slot) {
                    const int nid = mgIdx(NI, NJ, cnx);
                    if (C.fluid[(size_t)nid]) { slot = nid; count++; }
                    else if (!C.solid[(size_t)nid]) { count++; }
                };

                if (I > 0)       addNbrC(I - 1, J, C.L[(size_t)id]);
                if (I + 1 < cnx) addNbrC(I + 1, J, C.R[(size_t)id]);
                if (J > 0)       addNbrC(I, J - 1, C.B[(size_t)id]);
                if (J + 1 < cny) addNbrC(I, J + 1, C.T[(size_t)id]);

                if (m_openTopBC && J == cny - 1) count++;

                C.diagCount[(size_t)id] = (uint8_t)count;
                const float diag = (float)count * C.invDx2;
                C.diagInv[(size_t)id] = (diag > 0.0f) ? (1.0f / diag) : 0.0f;
            }
        }

        mgLevels.push_back(std::move(C));
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
        int n = L.L[(size_t)id]; if (n >= 0) sum += x[(size_t)n];
        n = L.R[(size_t)id];     if (n >= 0) sum += x[(size_t)n];
        n = L.B[(size_t)id];     if (n >= 0) sum += x[(size_t)n];
        n = L.T[(size_t)id];     if (n >= 0) sum += x[(size_t)n];

        const int count = (int)L.diagCount[(size_t)id];
        Ax[(size_t)id] = (count * x[(size_t)id] - sum) * L.invDx2;
    }
}

void PressureSolver::mgSmoothRBGS(int lev, int iters)
{
    MGLevel& L = mgLevels[(size_t)lev];
    const int nx = L.nx, ny = L.ny;

    float omega = 1.0f;
    if (mgUseSOR) omega = std::max(1.0f, std::min(mgSORomega, 1.9f));

    for (int it = 0; it < iters; ++it) {
        for (int color = 0; color < 2; ++color) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    if (((i + j) & 1) != color) continue;
                    const int id = mgIdx(i, j, nx);
                    if (!L.fluid[(size_t)id]) continue;

                    float sum = 0.0f;
                    int n = L.L[(size_t)id]; if (n >= 0) sum += L.x[(size_t)n];
                    n = L.R[(size_t)id];     if (n >= 0) sum += L.x[(size_t)n];
                    n = L.B[(size_t)id];     if (n >= 0) sum += L.x[(size_t)n];
                    n = L.T[(size_t)id];     if (n >= 0) sum += L.x[(size_t)n];

                    const int count = (int)L.diagCount[(size_t)id];
                    if (count == 0) continue;

                    const float x_gs = (sum + L.b[(size_t)id] / L.invDx2) / (float)count;
                    L.x[(size_t)id] = (1.0f - omega) * L.x[(size_t)id] + omega * x_gs;
                }
            }
        }
    }
}

void PressureSolver::mgComputeResidual(int lev)
{
    MGLevel& L = mgLevels[(size_t)lev];
    mgApplyA(lev, L.x, L.Ax);
    const int N = L.nx * L.ny;

    for (int id = 0; id < N; ++id) {
        if (!L.fluid[(size_t)id]) { L.r[(size_t)id] = 0.0f; continue; }
        L.r[(size_t)id] = L.b[(size_t)id] - L.Ax[(size_t)id];
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
        mgSmoothRBGS(lev, mgCoarseIters);
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

    MGLevel& F = mgLevels[0];
    const int N = F.nx * F.ny;

    if ((int)p.size() != N) p.assign((size_t)N, 0.0f);
    if ((int)F.x.size() != N) F.x.assign((size_t)N, 0.0f);
    if ((int)F.b.size() != N) F.b.assign((size_t)N, 0.0f);

    // Warm start
    F.x = p;
    F.b = rhs;

    auto maxAbsB = [&]() -> float {
        float m = 0.0f;
        for (int id = 0; id < N; ++id) if (F.fluid[(size_t)id])
            m = std::max(m, std::fabs(F.b[(size_t)id]));
        return m;
    };
    auto maxAbsR = [&]() -> float {
        float m = 0.0f;
        for (int id = 0; id < N; ++id) if (F.fluid[(size_t)id])
            m = std::max(m, std::fabs(F.r[(size_t)id]));
        return m;
    };

    // early-out if RHS tiny in predDiv space
    const float bInf = maxAbsB();
    if (!(bInf > 0.0f) || bInf * dt <= tolPredDiv) {
        p = F.x;
        if (m_removeMean && !m_openTopBC) removeMean(p);
        return;
    }

    mgComputeResidual(0);
    float rInf = maxAbsR();
    if (!(rInf > 0.0f) || rInf * dt <= tolPredDiv) {
        p = F.x;
        if (m_removeMean && !m_openTopBC) removeMean(p);
        return;
    }

    bool fallbackPCG = false;

    for (int v = 0; v < maxVCycles; ++v) {
        const float prev = rInf;

        mgVCycle(0);

        if (m_removeMean && !m_openTopBC)
            removeMean(F.x);

        mgComputeResidual(0);
        rInf = maxAbsR();

        if (!std::isfinite(rInf)) { fallbackPCG = true; break; }
        if (rInf > prev * 1.2f)   { fallbackPCG = true; break; }

        if (rInf * dt <= tolPredDiv) break;
    }

    p = F.x;
    if (m_removeMean && !m_openTopBC) removeMean(p);

    if (fallbackPCG) {
        // Warm-start PCG from current MG x
        solvePCG(p, rhs, 80, tolPredDiv, dt);
    }
}