#include "pressure_solver3d.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

void PressureSolver3D::configure(int nx, int ny, int nz, float dx,
                                 bool openTopBC,
                                 const std::vector<uint8_t>& solidMask,
                                 const std::vector<uint8_t>& fluidMask,
                                 bool removeMeanForGauge,
                                 const std::vector<float>* faceOpenU,
                                 const std::vector<float>* faceOpenV,
                                 const std::vector<float>* faceOpenW)
{
    m_nx = nx;
    m_ny = ny;
    m_nz = nz;
    m_dx = dx;
    m_invDx2 = 1.0f / std::max(1e-12f, dx * dx);
    m_openTopBC = openTopBC;
    m_removeMean = removeMeanForGauge;

    const int N = std::max(0, nx * ny * nz);
    m_solid = solidMask;
    m_fluid = fluidMask;
    if ((int)m_solid.size() != N) m_solid.assign((std::size_t)N, (uint8_t)0);
    if ((int)m_fluid.size() != N) m_fluid.assign((std::size_t)N, (uint8_t)0);

    for (int id = 0; id < N; ++id) {
        if (m_solid[(std::size_t)id]) m_fluid[(std::size_t)id] = 0;
    }

    m_faceOpenU.assign((std::size_t)(nx + 1) * (std::size_t)ny * (std::size_t)nz, 1.0f);
    m_faceOpenV.assign((std::size_t)nx * (std::size_t)(ny + 1) * (std::size_t)nz, 1.0f);
    m_faceOpenW.assign((std::size_t)nx * (std::size_t)ny * (std::size_t)(nz + 1), 1.0f);

    if (faceOpenU && faceOpenU->size() == m_faceOpenU.size()) {
        m_faceOpenU = *faceOpenU;
    } else {
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i <= nx; ++i) {
                    bool blocked = false;
                    if (i - 1 >= 0) blocked = blocked || (m_solid[(std::size_t)idx(i - 1, j, k, nx, ny)] != 0);
                    if (i < nx)     blocked = blocked || (m_solid[(std::size_t)idx(i,     j, k, nx, ny)] != 0);
                    m_faceOpenU[uIdx(i, j, k, nx, ny)] = blocked ? 0.0f : 1.0f;
                }
            }
        }
    }

    if (faceOpenV && faceOpenV->size() == m_faceOpenV.size()) {
        m_faceOpenV = *faceOpenV;
    } else {
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j <= ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    bool blocked = false;
                    if (j - 1 >= 0) blocked = blocked || (m_solid[(std::size_t)idx(i, j - 1, k, nx, ny)] != 0);
                    if (j < ny)     blocked = blocked || (m_solid[(std::size_t)idx(i, j,     k, nx, ny)] != 0);
                    m_faceOpenV[vIdx(i, j, k, nx, ny)] = blocked ? 0.0f : 1.0f;
                }
            }
        }
    }

    if (faceOpenW && faceOpenW->size() == m_faceOpenW.size()) {
        m_faceOpenW = *faceOpenW;
    } else {
        for (int k = 0; k <= nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    bool blocked = false;
                    if (k - 1 >= 0) blocked = blocked || (m_solid[(std::size_t)idx(i, j, k - 1, nx, ny)] != 0);
                    if (k < nz)     blocked = blocked || (m_solid[(std::size_t)idx(i, j, k,     nx, ny)] != 0);
                    m_faceOpenW[wIdx(i, j, k, nx, ny)] = blocked ? 0.0f : 1.0f;
                }
            }
        }
    }

    m_dirty = true;
    mgDirty = true;
}

void PressureSolver3D::ensurePCGBuffers()
{
    const int N = m_nx * m_ny * m_nz;
    if ((int)m_r.size() != N) m_r.resize((std::size_t)N);
    if ((int)m_z.size() != N) m_z.resize((std::size_t)N);
    if ((int)m_d.size() != N) m_d.resize((std::size_t)N);
    if ((int)m_q.size() != N) m_q.resize((std::size_t)N);
    if ((int)m_Ap.size() != N) m_Ap.resize((std::size_t)N);
}

void PressureSolver3D::rebuildOperator()
{
    const int nx = m_nx;
    const int ny = m_ny;
    const int nz = m_nz;
    const int N = nx * ny * nz;

    m_xm.assign((std::size_t)N, -1);
    m_xp.assign((std::size_t)N, -1);
    m_ym.assign((std::size_t)N, -1);
    m_yp.assign((std::size_t)N, -1);
    m_zm.assign((std::size_t)N, -1);
    m_zp.assign((std::size_t)N, -1);

    m_wXm.assign((std::size_t)N, 0.0f);
    m_wXp.assign((std::size_t)N, 0.0f);
    m_wYm.assign((std::size_t)N, 0.0f);
    m_wYp.assign((std::size_t)N, 0.0f);
    m_wZm.assign((std::size_t)N, 0.0f);
    m_wZp.assign((std::size_t)N, 0.0f);

    m_diagW.assign((std::size_t)N, 0.0f);
    m_diagInv.assign((std::size_t)N, 0.0f);

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const int id = idx(i, j, k, nx, ny);
                if (!m_fluid[(std::size_t)id]) continue;

                float diagW = 0.0f;
                auto addFace = [&](int ni, int nj, int nk,
                                   float wFace,
                                   bool outsideIsDirichlet,
                                   int& slotNbr,
                                   float& wNbr)
                {
                    if (wFace <= 0.0f) return;

                    if (ni >= 0 && nj >= 0 && nk >= 0 && ni < nx && nj < ny && nk < nz) {
                        const int nid = idx(ni, nj, nk, nx, ny);
                        if (m_solid[(std::size_t)nid]) return;

                        if (m_fluid[(std::size_t)nid]) {
                            slotNbr = nid;
                            wNbr = wFace;
                            diagW += wFace;
                        } else {
                            diagW += wFace;
                        }
                    } else if (outsideIsDirichlet) {
                        diagW += wFace;
                    }
                };

                addFace(i - 1, j, k, m_faceOpenU[uIdx(i,     j, k, nx, ny)], false,                         m_xm[(std::size_t)id], m_wXm[(std::size_t)id]);
                addFace(i + 1, j, k, m_faceOpenU[uIdx(i + 1, j, k, nx, ny)], false,                         m_xp[(std::size_t)id], m_wXp[(std::size_t)id]);
                addFace(i, j - 1, k, m_faceOpenV[vIdx(i, j,     k, nx, ny)], false,                         m_ym[(std::size_t)id], m_wYm[(std::size_t)id]);
                addFace(i, j + 1, k, m_faceOpenV[vIdx(i, j + 1, k, nx, ny)], m_openTopBC && (j + 1 >= ny), m_yp[(std::size_t)id], m_wYp[(std::size_t)id]);
                addFace(i, j, k - 1, m_faceOpenW[wIdx(i, j, k,     nx, ny)], false,                         m_zm[(std::size_t)id], m_wZm[(std::size_t)id]);
                addFace(i, j, k + 1, m_faceOpenW[wIdx(i, j, k + 1, nx, ny)], false,                         m_zp[(std::size_t)id], m_wZp[(std::size_t)id]);

                m_diagW[(std::size_t)id] = diagW;
                const float diag = diagW * m_invDx2;
                m_diagInv[(std::size_t)id] = (diag > 0.0f) ? (1.0f / diag) : 0.0f;
            }
        }
    }

    m_dirty = false;
}

void PressureSolver3D::applyA(const std::vector<float>& x, std::vector<float>& Ax) const
{
    const int N = m_nx * m_ny * m_nz;
    if ((int)Ax.size() != N) Ax.resize((std::size_t)N);

    for (int id = 0; id < N; ++id) {
        if (!m_fluid[(std::size_t)id]) {
            Ax[(std::size_t)id] = 0.0f;
            continue;
        }

        float sum = 0.0f;
        int n = m_xm[(std::size_t)id]; if (n >= 0) sum += m_wXm[(std::size_t)id] * x[(std::size_t)n];
        n = m_xp[(std::size_t)id];     if (n >= 0) sum += m_wXp[(std::size_t)id] * x[(std::size_t)n];
        n = m_ym[(std::size_t)id];     if (n >= 0) sum += m_wYm[(std::size_t)id] * x[(std::size_t)n];
        n = m_yp[(std::size_t)id];     if (n >= 0) sum += m_wYp[(std::size_t)id] * x[(std::size_t)n];
        n = m_zm[(std::size_t)id];     if (n >= 0) sum += m_wZm[(std::size_t)id] * x[(std::size_t)n];
        n = m_zp[(std::size_t)id];     if (n >= 0) sum += m_wZp[(std::size_t)id] * x[(std::size_t)n];

        Ax[(std::size_t)id] = (m_diagW[(std::size_t)id] * x[(std::size_t)id] - sum) * m_invDx2;
    }
}

float PressureSolver3D::dotFluid(const std::vector<float>& a, const std::vector<float>& b) const
{
    double s = 0.0;
    const int N = m_nx * m_ny * m_nz;
    for (int id = 0; id < N; ++id) {
        if (!m_fluid[(std::size_t)id]) continue;
        s += (double)a[(std::size_t)id] * (double)b[(std::size_t)id];
    }
    return (float)s;
}

float PressureSolver3D::maxAbsFluid(const std::vector<float>& a) const
{
    float m = 0.0f;
    const int N = m_nx * m_ny * m_nz;
    for (int id = 0; id < N; ++id) {
        if (!m_fluid[(std::size_t)id]) continue;
        const float v = std::fabs(a[(std::size_t)id]);
        if (!std::isfinite(v)) return std::numeric_limits<float>::infinity();
        m = std::max(m, v);
    }
    return m;
}

void PressureSolver3D::removeMean(std::vector<float>& p) const
{
    if (!m_removeMean) return;

    double sum = 0.0;
    int cnt = 0;
    const int N = m_nx * m_ny * m_nz;
    for (int id = 0; id < N; ++id) {
        if (!m_fluid[(std::size_t)id]) continue;
        sum += (double)p[(std::size_t)id];
        ++cnt;
    }
    if (cnt <= 0) return;

    const float mean = (float)(sum / (double)cnt);
    for (int id = 0; id < N; ++id) {
        if (m_fluid[(std::size_t)id]) p[(std::size_t)id] -= mean;
    }
}

int PressureSolver3D::solvePCG(std::vector<float>& p,
                               const std::vector<float>& rhs,
                               int maxIters,
                               float tolPredDiv,
                               float /*dtForPredDiv*/)
{
    const int N = m_nx * m_ny * m_nz;
    if (N <= 0) {
        m_lastIters = 0;
        return 0;
    }

    if (m_dirty) rebuildOperator();
    ensurePCGBuffers();

    if ((int)p.size() != N) p.assign((std::size_t)N, 0.0f);
    if ((int)rhs.size() != N) {
        m_lastIters = 0;
        return 0;
    }

    const float bInf = maxAbsFluid(rhs);
    if (!std::isfinite(bInf) || bInf <= tolPredDiv) {
        removeMean(p);
        m_lastIters = 0;
        return 0;
    }

    applyA(p, m_Ap);
    for (int id = 0; id < N; ++id) m_r[(std::size_t)id] = rhs[(std::size_t)id] - m_Ap[(std::size_t)id];

    const float rInf0 = std::max(maxAbsFluid(m_r), 1e-30f);
    for (int id = 0; id < N; ++id) m_z[(std::size_t)id] = m_r[(std::size_t)id] * m_diagInv[(std::size_t)id];
    m_d = m_z;

    float deltaNew = dotFluid(m_r, m_z);
    if (!std::isfinite(deltaNew) || deltaNew <= 1e-30f) {
        removeMean(p);
        m_lastIters = 0;
        return 0;
    }

    const float relTol = 1e-5f;
    int itUsed = 0;

    for (int it = 0; it < std::max(1, maxIters); ++it) {
        itUsed = it + 1;

        applyA(m_d, m_q);
        const float dq = dotFluid(m_d, m_q);
        if (!std::isfinite(dq) || std::fabs(dq) < 1e-30f) break;

        const float alpha = deltaNew / dq;
        for (int id = 0; id < N; ++id) {
            p[(std::size_t)id]   += alpha * m_d[(std::size_t)id];
            m_r[(std::size_t)id] -= alpha * m_q[(std::size_t)id];
        }

        const float rInf = maxAbsFluid(m_r);
        if (!std::isfinite(rInf)) break;
        if (rInf <= tolPredDiv) break;
        if (rInf <= relTol * rInf0) break;

        for (int id = 0; id < N; ++id) m_z[(std::size_t)id] = m_r[(std::size_t)id] * m_diagInv[(std::size_t)id];

        const float deltaOld = deltaNew;
        deltaNew = dotFluid(m_r, m_z);
        if (!std::isfinite(deltaNew) || deltaNew <= 1e-30f) break;

        const float beta = deltaNew / (deltaOld + 1e-30f);
        for (int id = 0; id < N; ++id) {
            m_d[(std::size_t)id] = m_z[(std::size_t)id] + beta * m_d[(std::size_t)id];
        }
    }

    removeMean(p);
    m_lastIters = itUsed;
    return itUsed;
}

void PressureSolver3D::buildLevelStencil(MGLevel& L) const
{
    const int nx = L.nx;
    const int ny = L.ny;
    const int nz = L.nz;
    const int N = nx * ny * nz;

    L.xm.assign((std::size_t)N, -1);
    L.xp.assign((std::size_t)N, -1);
    L.ym.assign((std::size_t)N, -1);
    L.yp.assign((std::size_t)N, -1);
    L.zm.assign((std::size_t)N, -1);
    L.zp.assign((std::size_t)N, -1);

    L.wXm.assign((std::size_t)N, 0.0f);
    L.wXp.assign((std::size_t)N, 0.0f);
    L.wYm.assign((std::size_t)N, 0.0f);
    L.wYp.assign((std::size_t)N, 0.0f);
    L.wZm.assign((std::size_t)N, 0.0f);
    L.wZp.assign((std::size_t)N, 0.0f);

    L.diagW.assign((std::size_t)N, 0.0f);
    L.diagInv.assign((std::size_t)N, 0.0f);
    if ((int)L.x.size() != N)  L.x.assign((std::size_t)N, 0.0f);
    if ((int)L.b.size() != N)  L.b.assign((std::size_t)N, 0.0f);
    if ((int)L.Ax.size() != N) L.Ax.assign((std::size_t)N, 0.0f);
    if ((int)L.r.size() != N)  L.r.assign((std::size_t)N, 0.0f);

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const int id = idx(i, j, k, nx, ny);
                if (!L.fluid[(std::size_t)id]) continue;

                float diagW = 0.0f;
                auto addFace = [&](int ni, int nj, int nk,
                                   float wFace,
                                   bool outsideIsDirichlet,
                                   int& slotNbr,
                                   float& wNbr)
                {
                    if (wFace <= 0.0f) return;

                    if (ni >= 0 && nj >= 0 && nk >= 0 && ni < nx && nj < ny && nk < nz) {
                        const int nid = idx(ni, nj, nk, nx, ny);
                        if (L.solid[(std::size_t)nid]) return;

                        if (L.fluid[(std::size_t)nid]) {
                            slotNbr = nid;
                            wNbr = wFace;
                            diagW += wFace;
                        } else {
                            diagW += wFace;
                        }
                    } else if (outsideIsDirichlet) {
                        diagW += wFace;
                    }
                };

                addFace(i - 1, j, k, L.faceOpenU[uIdx(i,     j, k, nx, ny)], false,                         L.xm[(std::size_t)id], L.wXm[(std::size_t)id]);
                addFace(i + 1, j, k, L.faceOpenU[uIdx(i + 1, j, k, nx, ny)], false,                         L.xp[(std::size_t)id], L.wXp[(std::size_t)id]);
                addFace(i, j - 1, k, L.faceOpenV[vIdx(i, j,     k, nx, ny)], false,                         L.ym[(std::size_t)id], L.wYm[(std::size_t)id]);
                addFace(i, j + 1, k, L.faceOpenV[vIdx(i, j + 1, k, nx, ny)], m_openTopBC && (j + 1 >= ny), L.yp[(std::size_t)id], L.wYp[(std::size_t)id]);
                addFace(i, j, k - 1, L.faceOpenW[wIdx(i, j, k,     nx, ny)], false,                         L.zm[(std::size_t)id], L.wZm[(std::size_t)id]);
                addFace(i, j, k + 1, L.faceOpenW[wIdx(i, j, k + 1, nx, ny)], false,                         L.zp[(std::size_t)id], L.wZp[(std::size_t)id]);

                L.diagW[(std::size_t)id] = diagW;
                const float diag = diagW * L.invDx2;
                L.diagInv[(std::size_t)id] = (diag > 0.0f) ? (1.0f / diag) : 0.0f;
            }
        }
    }
}

void PressureSolver3D::ensureMultigrid()
{
    if (!mgDirty && mgBuiltValid &&
        mgBuiltNx == m_nx && mgBuiltNy == m_ny && mgBuiltNz == m_nz &&
        mgBuiltOpenTop == m_openTopBC)
    {
        return;
    }

    mgDirty = false;
    mgBuiltValid = true;
    mgBuiltNx = m_nx;
    mgBuiltNy = m_ny;
    mgBuiltNz = m_nz;
    mgBuiltOpenTop = m_openTopBC;

    mgLevels.clear();
    mgLevels.reserve((std::size_t)mgMaxLevels);

    {
        MGLevel L0;
        L0.nx = m_nx;
        L0.ny = m_ny;
        L0.nz = m_nz;
        L0.invDx2 = m_invDx2;
        L0.solid = m_solid;
        L0.fluid = m_fluid;
        L0.faceOpenU = m_faceOpenU;
        L0.faceOpenV = m_faceOpenV;
        L0.faceOpenW = m_faceOpenW;
        buildLevelStencil(L0);
        mgLevels.push_back(std::move(L0));
    }

    while ((int)mgLevels.size() < mgMaxLevels) {
        const MGLevel& F = mgLevels.back();
        if (F.nx <= 4 || F.ny <= 4 || F.nz <= 4) break;

        const int cnx = std::max(2, (F.nx + 1) / 2);
        const int cny = std::max(2, (F.ny + 1) / 2);
        const int cnz = std::max(2, (F.nz + 1) / 2);
        if (cnx == F.nx && cny == F.ny && cnz == F.nz) break;

        MGLevel C;
        C.nx = cnx;
        C.ny = cny;
        C.nz = cnz;
        C.invDx2 = F.invDx2 * 0.25f;

        const int CN = cnx * cny * cnz;
        C.solid.assign((std::size_t)CN, (uint8_t)0);
        C.fluid.assign((std::size_t)CN, (uint8_t)0);
        C.faceOpenU.assign((std::size_t)(cnx + 1) * (std::size_t)cny * (std::size_t)cnz, 0.0f);
        C.faceOpenV.assign((std::size_t)cnx * (std::size_t)(cny + 1) * (std::size_t)cnz, 0.0f);
        C.faceOpenW.assign((std::size_t)cnx * (std::size_t)cny * (std::size_t)(cnz + 1), 0.0f);

        for (int K = 0; K < cnz; ++K) {
            for (int J = 0; J < cny; ++J) {
                for (int I = 0; I < cnx; ++I) {
                    bool allSolid = true;
                    bool anyFluid = false;
                    bool anyInside = false;

                    for (int dk = 0; dk < 2; ++dk) {
                        for (int dj = 0; dj < 2; ++dj) {
                            for (int di = 0; di < 2; ++di) {
                                const int fi = 2 * I + di;
                                const int fj = 2 * J + dj;
                                const int fk = 2 * K + dk;
                                if (fi >= F.nx || fj >= F.ny || fk >= F.nz) continue;
                                anyInside = true;
                                const int fid = idx(fi, fj, fk, F.nx, F.ny);
                                if (!F.solid[(std::size_t)fid]) allSolid = false;
                                if (F.fluid[(std::size_t)fid]) anyFluid = true;
                            }
                        }
                    }

                    const int cid = idx(I, J, K, cnx, cny);
                    C.solid[(std::size_t)cid] = (anyInside && allSolid) ? (uint8_t)1 : (uint8_t)0;
                    C.fluid[(std::size_t)cid] = (!C.solid[(std::size_t)cid] && anyFluid) ? (uint8_t)1 : (uint8_t)0;
                }
            }
        }

        for (int K = 0; K < cnz; ++K) {
            for (int J = 0; J < cny; ++J) {
                for (int I = 0; I <= cnx; ++I) {
                    float sum = 0.0f;
                    int count = 0;
                    const int fi = 2 * I;
                    for (int dk = 0; dk < 2; ++dk) {
                        for (int dj = 0; dj < 2; ++dj) {
                            const int fj = 2 * J + dj;
                            const int fk = 2 * K + dk;
                            if (fi < 0 || fi > F.nx || fj < 0 || fj >= F.ny || fk < 0 || fk >= F.nz) continue;
                            sum += F.faceOpenU[uIdx(fi, fj, fk, F.nx, F.ny)];
                            ++count;
                        }
                    }
                    C.faceOpenU[uIdx(I, J, K, cnx, cny)] = (count > 0) ? (sum / (float)count) : 0.0f;
                }
            }
        }

        for (int K = 0; K < cnz; ++K) {
            for (int J = 0; J <= cny; ++J) {
                for (int I = 0; I < cnx; ++I) {
                    float sum = 0.0f;
                    int count = 0;
                    const int fj = 2 * J;
                    for (int dk = 0; dk < 2; ++dk) {
                        for (int di = 0; di < 2; ++di) {
                            const int fi = 2 * I + di;
                            const int fk = 2 * K + dk;
                            if (fi < 0 || fi >= F.nx || fj < 0 || fj > F.ny || fk < 0 || fk >= F.nz) continue;
                            sum += F.faceOpenV[vIdx(fi, fj, fk, F.nx, F.ny)];
                            ++count;
                        }
                    }
                    C.faceOpenV[vIdx(I, J, K, cnx, cny)] = (count > 0) ? (sum / (float)count) : 0.0f;
                }
            }
        }

        for (int K = 0; K <= cnz; ++K) {
            for (int J = 0; J < cny; ++J) {
                for (int I = 0; I < cnx; ++I) {
                    float sum = 0.0f;
                    int count = 0;
                    const int fk = 2 * K;
                    for (int dj = 0; dj < 2; ++dj) {
                        for (int di = 0; di < 2; ++di) {
                            const int fi = 2 * I + di;
                            const int fj = 2 * J + dj;
                            if (fi < 0 || fi >= F.nx || fj < 0 || fj >= F.ny || fk < 0 || fk > F.nz) continue;
                            sum += F.faceOpenW[wIdx(fi, fj, fk, F.nx, F.ny)];
                            ++count;
                        }
                    }
                    C.faceOpenW[wIdx(I, J, K, cnx, cny)] = (count > 0) ? (sum / (float)count) : 0.0f;
                }
            }
        }

        buildLevelStencil(C);
        mgLevels.push_back(std::move(C));
    }
}

void PressureSolver3D::mgApplyA(int lev, const std::vector<float>& x, std::vector<float>& Ax) const
{
    const MGLevel& L = mgLevels[(std::size_t)lev];
    const int N = L.nx * L.ny * L.nz;
    if ((int)Ax.size() != N) Ax.resize((std::size_t)N);

    for (int id = 0; id < N; ++id) {
        if (!L.fluid[(std::size_t)id]) {
            Ax[(std::size_t)id] = 0.0f;
            continue;
        }

        float sum = 0.0f;
        int n = L.xm[(std::size_t)id]; if (n >= 0) sum += L.wXm[(std::size_t)id] * x[(std::size_t)n];
        n = L.xp[(std::size_t)id];     if (n >= 0) sum += L.wXp[(std::size_t)id] * x[(std::size_t)n];
        n = L.ym[(std::size_t)id];     if (n >= 0) sum += L.wYm[(std::size_t)id] * x[(std::size_t)n];
        n = L.yp[(std::size_t)id];     if (n >= 0) sum += L.wYp[(std::size_t)id] * x[(std::size_t)n];
        n = L.zm[(std::size_t)id];     if (n >= 0) sum += L.wZm[(std::size_t)id] * x[(std::size_t)n];
        n = L.zp[(std::size_t)id];     if (n >= 0) sum += L.wZp[(std::size_t)id] * x[(std::size_t)n];

        Ax[(std::size_t)id] = (L.diagW[(std::size_t)id] * x[(std::size_t)id] - sum) * L.invDx2;
    }
}

void PressureSolver3D::mgSmoothRBGS(int lev, int iters)
{
    MGLevel& L = mgLevels[(std::size_t)lev];
    const int nx = L.nx;
    const int ny = L.ny;
    const int nz = L.nz;

    float omega = 1.0f;
    if (mgUseSOR) omega = std::max(1.0f, std::min(mgSORomega, 1.9f));

    for (int it = 0; it < iters; ++it) {
        for (int color = 0; color < 2; ++color) {
            for (int k = 0; k < nz; ++k) {
                for (int j = 0; j < ny; ++j) {
                    for (int i = 0; i < nx; ++i) {
                        if (((i + j + k) & 1) != color) continue;
                        const int id = idx(i, j, k, nx, ny);
                        if (!L.fluid[(std::size_t)id]) continue;

                        const float diagW = L.diagW[(std::size_t)id];
                        if (diagW <= 0.0f) continue;

                        float sum = 0.0f;
                        int n = L.xm[(std::size_t)id]; if (n >= 0) sum += L.wXm[(std::size_t)id] * L.x[(std::size_t)n];
                        n = L.xp[(std::size_t)id];     if (n >= 0) sum += L.wXp[(std::size_t)id] * L.x[(std::size_t)n];
                        n = L.ym[(std::size_t)id];     if (n >= 0) sum += L.wYm[(std::size_t)id] * L.x[(std::size_t)n];
                        n = L.yp[(std::size_t)id];     if (n >= 0) sum += L.wYp[(std::size_t)id] * L.x[(std::size_t)n];
                        n = L.zm[(std::size_t)id];     if (n >= 0) sum += L.wZm[(std::size_t)id] * L.x[(std::size_t)n];
                        n = L.zp[(std::size_t)id];     if (n >= 0) sum += L.wZp[(std::size_t)id] * L.x[(std::size_t)n];

                        const float xGs = (sum + L.b[(std::size_t)id] / L.invDx2) / diagW;
                        L.x[(std::size_t)id] = (1.0f - omega) * L.x[(std::size_t)id] + omega * xGs;
                    }
                }
            }
        }
    }
}

void PressureSolver3D::mgComputeResidual(int lev)
{
    MGLevel& L = mgLevels[(std::size_t)lev];
    mgApplyA(lev, L.x, L.Ax);
    const int N = L.nx * L.ny * L.nz;
    for (int id = 0; id < N; ++id) {
        if (!L.fluid[(std::size_t)id]) {
            L.r[(std::size_t)id] = 0.0f;
            continue;
        }
        L.r[(std::size_t)id] = L.b[(std::size_t)id] - L.Ax[(std::size_t)id];
    }
}

void PressureSolver3D::mgRestrictResidual(int fineLev)
{
    MGLevel& F = mgLevels[(std::size_t)fineLev];
    MGLevel& C = mgLevels[(std::size_t)fineLev + 1];

    std::fill(C.b.begin(), C.b.end(), 0.0f);
    std::fill(C.x.begin(), C.x.end(), 0.0f);

    static const float w1[3] = { 1.0f, 2.0f, 1.0f };

    for (int K = 0; K < C.nz; ++K) {
        for (int J = 0; J < C.ny; ++J) {
            for (int I = 0; I < C.nx; ++I) {
                const int cid = idx(I, J, K, C.nx, C.ny);
                if (!C.fluid[(std::size_t)cid]) continue;

                const int fi0 = 2 * I;
                const int fj0 = 2 * J;
                const int fk0 = 2 * K;

                float sum = 0.0f;
                float wsum = 0.0f;

                for (int dk = -1; dk <= 1; ++dk) {
                    for (int dj = -1; dj <= 1; ++dj) {
                        for (int di = -1; di <= 1; ++di) {
                            const int fi = fi0 + di;
                            const int fj = fj0 + dj;
                            const int fk = fk0 + dk;
                            if (fi < 0 || fj < 0 || fk < 0 || fi >= F.nx || fj >= F.ny || fk >= F.nz) continue;
                            const int fid = idx(fi, fj, fk, F.nx, F.ny);
                            if (!F.fluid[(std::size_t)fid]) continue;
                            const float wgt = w1[di + 1] * w1[dj + 1] * w1[dk + 1];
                            sum += wgt * F.r[(std::size_t)fid];
                            wsum += wgt;
                        }
                    }
                }

                C.b[(std::size_t)cid] = (wsum > 0.0f) ? (sum / wsum) : 0.0f;
            }
        }
    }
}

void PressureSolver3D::mgProlongateAndAdd(int coarseLev)
{
    MGLevel& C = mgLevels[(std::size_t)coarseLev];
    MGLevel& F = mgLevels[(std::size_t)coarseLev - 1];

    auto cAt = [&](int I, int J, int K) -> float {
        I = std::max(0, std::min(I, C.nx - 1));
        J = std::max(0, std::min(J, C.ny - 1));
        K = std::max(0, std::min(K, C.nz - 1));
        const int cid = idx(I, J, K, C.nx, C.ny);
        return C.fluid[(std::size_t)cid] ? C.x[(std::size_t)cid] : 0.0f;
    };

    for (int fk = 0; fk < F.nz; ++fk) {
        for (int fj = 0; fj < F.ny; ++fj) {
            for (int fi = 0; fi < F.nx; ++fi) {
                const int fid = idx(fi, fj, fk, F.nx, F.ny);
                if (!F.fluid[(std::size_t)fid]) continue;

                const int I = fi >> 1;
                const int J = fj >> 1;
                const int K = fk >> 1;
                const int ox = fi & 1;
                const int oy = fj & 1;
                const int oz = fk & 1;

                const float wx0 = (ox == 0) ? 1.0f : 0.5f;
                const float wx1 = (ox == 0) ? 0.0f : 0.5f;
                const float wy0 = (oy == 0) ? 1.0f : 0.5f;
                const float wy1 = (oy == 0) ? 0.0f : 0.5f;
                const float wz0 = (oz == 0) ? 1.0f : 0.5f;
                const float wz1 = (oz == 0) ? 0.0f : 0.5f;

                float e = 0.0f;
                e += wx0 * wy0 * wz0 * cAt(I,     J,     K    );
                e += wx1 * wy0 * wz0 * cAt(I + 1, J,     K    );
                e += wx0 * wy1 * wz0 * cAt(I,     J + 1, K    );
                e += wx1 * wy1 * wz0 * cAt(I + 1, J + 1, K    );
                e += wx0 * wy0 * wz1 * cAt(I,     J,     K + 1);
                e += wx1 * wy0 * wz1 * cAt(I + 1, J,     K + 1);
                e += wx0 * wy1 * wz1 * cAt(I,     J + 1, K + 1);
                e += wx1 * wy1 * wz1 * cAt(I + 1, J + 1, K + 1);

                F.x[(std::size_t)fid] += e;
            }
        }
    }
}

void PressureSolver3D::mgVCycle(int lev)
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

void PressureSolver3D::solveMG(std::vector<float>& p,
                               const std::vector<float>& rhs,
                               int maxVCycles,
                               float tolPredDiv,
                               float dtForPredDiv)
{
    ensureMultigrid();
    if (mgLevels.empty()) {
        solvePCG(p, rhs, 80, tolPredDiv, dtForPredDiv);
        return;
    }

    MGLevel& F = mgLevels[0];
    const int N = F.nx * F.ny * F.nz;
    if ((int)p.size() != N) p.assign((std::size_t)N, 0.0f);
    if ((int)rhs.size() != N) {
        m_lastIters = 0;
        return;
    }

    F.x = p;
    F.b = rhs;

    auto maxAbsB = [&]() -> float {
        float m = 0.0f;
        for (int id = 0; id < N; ++id) {
            if (!F.fluid[(std::size_t)id]) continue;
            m = std::max(m, std::fabs(F.b[(std::size_t)id]));
        }
        return m;
    };

    auto maxAbsR = [&]() -> float {
        float m = 0.0f;
        for (int id = 0; id < N; ++id) {
            if (!F.fluid[(std::size_t)id]) continue;
            m = std::max(m, std::fabs(F.r[(std::size_t)id]));
        }
        return m;
    };

    const float bInf = maxAbsB();
    if (!(bInf > 0.0f) || bInf <= tolPredDiv) {
        p = F.x;
        removeMean(p);
        m_lastIters = 0;
        return;
    }

    mgComputeResidual(0);
    float rInf = maxAbsR();
    if (!(rInf > 0.0f) || rInf <= tolPredDiv) {
        p = F.x;
        removeMean(p);
        m_lastIters = 0;
        return;
    }

    const float rInf0 = std::max(rInf, 1e-30f);
    const float relTol = 1e-5f;

    bool fallbackPCG = false;
    int cyclesUsed = 0;
    for (int v = 0; v < std::max(1, maxVCycles); ++v) {
        ++cyclesUsed;
        const float prev = rInf;

        mgVCycle(0);
        removeMean(F.x);
        mgComputeResidual(0);
        rInf = maxAbsR();

        if (!std::isfinite(rInf)) { fallbackPCG = true; break; }
        if (rInf > prev * 1.25f) { fallbackPCG = true; break; }
        if (rInf <= tolPredDiv) break;
        if (rInf <= relTol * rInf0) break;
    }

    p = F.x;
    removeMean(p);
    m_lastIters = cyclesUsed;

    if (fallbackPCG) {
        solvePCG(p, rhs, 120, tolPredDiv, dtForPredDiv);
    }
}
