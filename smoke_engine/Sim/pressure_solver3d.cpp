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
    m_fluidCells.clear();
    m_fluidCells.reserve((std::size_t)N);

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const int id = idx(i, j, k, nx, ny);
                if (!m_fluid[(std::size_t)id]) continue;

                m_fluidCells.push_back(id);

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

    const std::size_t fluidCount = m_fluidCells.size();
    if (fluidCount == 0u) return;

    const bool useMGStencil =
        !mgDirty && mgBuiltValid && !mgLevels.empty() &&
        mgLevels[0].nx == m_nx && mgLevels[0].ny == m_ny && mgLevels[0].nz == m_nz &&
        mgLevels[0].stencils.size() == fluidCount;

    if (useMGStencil) {
        const MGLevel& L = mgLevels[0];
        const MGCellStencil* const stencils = L.stencils.data();
        const float invDx2 = L.invDx2;
        for (std::size_t idxStencil = 0; idxStencil < fluidCount; ++idxStencil) {
            const MGCellStencil& s = stencils[idxStencil];
            const float sum =
                s.wXm * x[(std::size_t)s.xm] +
                s.wXp * x[(std::size_t)s.xp] +
                s.wYm * x[(std::size_t)s.ym] +
                s.wYp * x[(std::size_t)s.yp] +
                s.wZm * x[(std::size_t)s.zm] +
                s.wZp * x[(std::size_t)s.zp];
            Ax[(std::size_t)s.cell] = (s.diagW * x[(std::size_t)s.cell] - sum) * invDx2;
        }
        return;
    }

    for (int id : m_fluidCells) {
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
    for (int id : m_fluidCells) {
        s += (double)a[(std::size_t)id] * (double)b[(std::size_t)id];
    }
    return (float)s;
}

float PressureSolver3D::maxAbsFluid(const std::vector<float>& a) const
{
    float m = 0.0f;
    for (int id : m_fluidCells) {
        const float v = std::fabs(a[(std::size_t)id]);
        if (!std::isfinite(v)) return std::numeric_limits<float>::infinity();
        m = std::max(m, v);
    }
    return m;
}

void PressureSolver3D::removeMean(std::vector<float>& p) const
{
    if (!m_removeMean) return;
    if (m_fluidCells.empty()) return;

    double sum = 0.0;
    for (int id : m_fluidCells) {
        sum += (double)p[(std::size_t)id];
    }

    const float mean = (float)(sum / (double)m_fluidCells.size());
    for (int id : m_fluidCells) {
        p[(std::size_t)id] -= mean;
    }
}

int PressureSolver3D::solvePCG(std::vector<float>& p,
                               const std::vector<float>& rhs,
                               int maxIters,
                               float tolPredDiv,
                               float dtForPredDiv)
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

    if (m_fluidCells.empty()) {
        m_lastIters = 0;
        return 0;
    }

    const float tolRhs = tolPredDiv / std::max(1e-8f, dtForPredDiv);

    const float bInf = maxAbsFluid(rhs);
    if (!std::isfinite(bInf) || bInf <= tolRhs) {
        removeMean(p);
        m_lastIters = 0;
        return 0;
    }

    applyA(p, m_Ap);
    for (int id : m_fluidCells) {
        m_r[(std::size_t)id] = rhs[(std::size_t)id] - m_Ap[(std::size_t)id];
        m_z[(std::size_t)id] = m_r[(std::size_t)id] * m_diagInv[(std::size_t)id];
        m_d[(std::size_t)id] = m_z[(std::size_t)id];
    }

    const float rInf0 = std::max(maxAbsFluid(m_r), 1e-30f);
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
        for (int id : m_fluidCells) {
            p[(std::size_t)id] += alpha * m_d[(std::size_t)id];
            m_r[(std::size_t)id] -= alpha * m_q[(std::size_t)id];
        }

        const float rInf = maxAbsFluid(m_r);
        if (!std::isfinite(rInf)) break;
        if (rInf <= tolRhs) break;
        if (relTol > 0.0f && rInf <= relTol * rInf0) break;

        for (int id : m_fluidCells) {
            m_z[(std::size_t)id] = m_r[(std::size_t)id] * m_diagInv[(std::size_t)id];
        }

        const float deltaOld = deltaNew;
        deltaNew = dotFluid(m_r, m_z);
        if (!std::isfinite(deltaNew) || deltaNew <= 1e-30f) break;

        const float beta = deltaNew / (deltaOld + 1e-30f);
        for (int id : m_fluidCells) {
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

    L.x.assign((std::size_t)N, 0.0f);
    L.b.assign((std::size_t)N, 0.0f);
    L.r.assign((std::size_t)N, 0.0f);

    std::vector<MGCellStencil> red;
    std::vector<MGCellStencil> black;
    red.reserve((std::size_t)N / 2u + 1u);
    black.reserve((std::size_t)N / 2u + 1u);

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const int id = idx(i, j, k, nx, ny);
                if (!L.fluid[(std::size_t)id]) continue;

                MGCellStencil stencil;
                stencil.cell = id;

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

                addFace(i - 1, j, k,     L.faceOpenU[uIdx(i,     j, k, nx, ny)], false,                         stencil.xm, stencil.wXm);
                addFace(i + 1, j, k,     L.faceOpenU[uIdx(i + 1, j, k, nx, ny)], false,                         stencil.xp, stencil.wXp);
                addFace(i,     j - 1, k, L.faceOpenV[vIdx(i, j,     k, nx, ny)], false,                         stencil.ym, stencil.wYm);
                addFace(i,     j + 1, k, L.faceOpenV[vIdx(i, j + 1, k, nx, ny)], m_openTopBC && (j + 1 >= ny), stencil.yp, stencil.wYp);
                addFace(i,     j, k - 1, L.faceOpenW[wIdx(i, j, k,     nx, ny)], false,                         stencil.zm, stencil.wZm);
                addFace(i,     j, k + 1, L.faceOpenW[wIdx(i, j, k + 1, nx, ny)], false,                         stencil.zp, stencil.wZp);

                stencil.diagW = diagW;
                if (diagW > 0.0f) {
                    stencil.invDiagW = 1.0f / diagW;
                    stencil.diagInv = 1.0f / (diagW * L.invDx2);
                }

                if (((i + j + k) & 1) == 0) red.push_back(stencil);
                else                         black.push_back(stencil);
            }
        }
    }

    L.redStencilCount = red.size();
    L.stencils = std::move(red);
    L.stencils.insert(
        L.stencils.end(),
        std::make_move_iterator(black.begin()),
        std::make_move_iterator(black.end()));
}

void PressureSolver3D::buildTransfer(int fineLev, MGTransfer& transfer) const
{
    const MGLevel& F = mgLevels[(std::size_t)fineLev];
    const MGLevel& C = mgLevels[(std::size_t)fineLev + 1u];

    transfer.restrictEntries.clear();
    transfer.prolongEntries.clear();
    transfer.restrictEntries.reserve(C.stencils.size());
    transfer.prolongEntries.reserve(F.stencils.size());

    static const float w1[3] = { 1.0f, 2.0f, 1.0f };

    for (const MGCellStencil& coarseStencil : C.stencils) {
        MGRestrictEntry entry;
        entry.coarseCell = coarseStencil.cell;

        const int cid = coarseStencil.cell;
        const int I = cid % C.nx;
        const int tmpC = cid / C.nx;
        const int J = tmpC % C.ny;
        const int K = tmpC / C.ny;

        const int fi0 = 2 * I;
        const int fj0 = 2 * J;
        const int fk0 = 2 * K;

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
                    entry.ids[(std::size_t)entry.count] = fid;
                    entry.weights[(std::size_t)entry.count] = wgt;
                    ++entry.count;
                    wsum += wgt;
                }
            }
        }

        if (wsum > 0.0f) {
            const float invWsum = 1.0f / wsum;
            for (uint8_t n = 0; n < entry.count; ++n) {
                entry.weights[(std::size_t)n] *= invWsum;
            }
        }

        transfer.restrictEntries.push_back(entry);
    }

    for (const MGCellStencil& fineStencil : F.stencils) {
        MGProlongEntry entry;
        entry.fineCell = fineStencil.cell;

        const int fid = fineStencil.cell;
        const int fi = fid % F.nx;
        const int tmpF = fid / F.nx;
        const int fj = tmpF % F.ny;
        const int fk = tmpF / F.ny;

        const int I = fi >> 1;
        const int J = fj >> 1;
        const int K = fk >> 1;
        const int I1 = std::min(I + 1, C.nx - 1);
        const int J1 = std::min(J + 1, C.ny - 1);
        const int K1 = std::min(K + 1, C.nz - 1);

        const int ox = fi & 1;
        const int oy = fj & 1;
        const int oz = fk & 1;

        const float wx0 = (ox == 0) ? 1.0f : 0.5f;
        const float wx1 = (ox == 0) ? 0.0f : 0.5f;
        const float wy0 = (oy == 0) ? 1.0f : 0.5f;
        const float wy1 = (oy == 0) ? 0.0f : 0.5f;
        const float wz0 = (oz == 0) ? 1.0f : 0.5f;
        const float wz1 = (oz == 0) ? 0.0f : 0.5f;

        auto addCoarse = [&](int ci, int cj, int ck, float weight) {
            if (weight <= 0.0f) return;
            const int cidLocal = idx(ci, cj, ck, C.nx, C.ny);
            if (!C.fluid[(std::size_t)cidLocal]) return;

            for (uint8_t n = 0; n < entry.count; ++n) {
                if (entry.ids[(std::size_t)n] == cidLocal) {
                    entry.weights[(std::size_t)n] += weight;
                    return;
                }
            }

            entry.ids[(std::size_t)entry.count] = cidLocal;
            entry.weights[(std::size_t)entry.count] = weight;
            ++entry.count;
        };

        addCoarse(I,  J,  K,  wx0 * wy0 * wz0);
        addCoarse(I1, J,  K,  wx1 * wy0 * wz0);
        addCoarse(I,  J1, K,  wx0 * wy1 * wz0);
        addCoarse(I1, J1, K,  wx1 * wy1 * wz0);
        addCoarse(I,  J,  K1, wx0 * wy0 * wz1);
        addCoarse(I1, J,  K1, wx1 * wy0 * wz1);
        addCoarse(I,  J1, K1, wx0 * wy1 * wz1);
        addCoarse(I1, J1, K1, wx1 * wy1 * wz1);

        transfer.prolongEntries.push_back(entry);
    }
}

bool PressureSolver3D::mgDirectSolve(int lev)
{
    MGLevel& L = mgLevels[(std::size_t)lev];
    if (!L.directSolveValid) return false;

    const int n = (int)L.directSolveCells.size();
    if (n <= 0) {
        std::fill(L.x.begin(), L.x.end(), 0.0f);
        return true;
    }

    if ((int)L.directSolveScratch0.size() != n) L.directSolveScratch0.assign((std::size_t)n, 0.0f);
    if ((int)L.directSolveScratch1.size() != n) L.directSolveScratch1.assign((std::size_t)n, 0.0f);

    std::fill(L.x.begin(), L.x.end(), 0.0f);

    float* const y = L.directSolveScratch0.data();
    float* const xCompact = L.directSolveScratch1.data();
    const float* const chol = L.directSolveCholesky.data();

    for (int row = 0; row < n; ++row) {
        const int cell = L.directSolveCells[(std::size_t)row];
        y[row] = L.b[(std::size_t)cell];
        xCompact[row] = 0.0f;
    }
    if (L.directSolveAnchorsGauge && n > 0) y[0] = 0.0f;

    for (int row = 0; row < n; ++row) {
        float sum = y[row];
        const std::size_t rowBase = (std::size_t)row * (std::size_t)n;
        for (int col = 0; col < row; ++col) {
            sum -= chol[rowBase + (std::size_t)col] * y[col];
        }
        const float diag = chol[rowBase + (std::size_t)row];
        if (!(diag > 0.0f) || !std::isfinite(diag)) return false;
        y[row] = sum / diag;
    }

    for (int row = n - 1; row >= 0; --row) {
        float sum = y[row];
        for (int col = row + 1; col < n; ++col) {
            sum -= chol[(std::size_t)col * (std::size_t)n + (std::size_t)row] * xCompact[col];
        }
        const float diag = chol[(std::size_t)row * (std::size_t)n + (std::size_t)row];
        if (!(diag > 0.0f) || !std::isfinite(diag)) return false;
        xCompact[row] = sum / diag;
    }

    if (L.directSolveAnchorsGauge && n > 0) xCompact[0] = 0.0f;

    for (int row = 0; row < n; ++row) {
        const int cell = L.directSolveCells[(std::size_t)row];
        L.x[(std::size_t)cell] = xCompact[row];
    }

    return true;
}

void PressureSolver3D::buildDirectCoarseSolve(MGLevel& L) const
{
    L.directSolveValid = false;
    L.directSolveAnchorsGauge = false;
    L.directSolveCells.clear();
    L.directSolveCompactIndex.clear();
    L.directSolveCholesky.clear();
    L.directSolveScratch0.clear();
    L.directSolveScratch1.clear();

    const int totalCells = L.nx * L.ny * L.nz;
    const int n = (int)L.stencils.size();
    if (n <= 0 || totalCells <= 0) return;

    L.directSolveCells.resize((std::size_t)n);
    L.directSolveCompactIndex.assign((std::size_t)totalCells, -1);
    for (int row = 0; row < n; ++row) {
        const int cell = L.stencils[(std::size_t)row].cell;
        L.directSolveCells[(std::size_t)row] = cell;
        L.directSolveCompactIndex[(std::size_t)cell] = row;
    }

    std::vector<float> dense((std::size_t)n * (std::size_t)n, 0.0f);
    auto addNbr = [&](int row, int nbrCell, float weight) {
        if (weight <= 0.0f || nbrCell < 0 || nbrCell >= totalCells) return;
        const int col = L.directSolveCompactIndex[(std::size_t)nbrCell];
        if (col < 0) return;
        dense[(std::size_t)row * (std::size_t)n + (std::size_t)col] -= weight * L.invDx2;
    };

    for (int row = 0; row < n; ++row) {
        const MGCellStencil& s = L.stencils[(std::size_t)row];
        dense[(std::size_t)row * (std::size_t)n + (std::size_t)row] = s.diagW * L.invDx2;
        addNbr(row, s.xm, s.wXm);
        addNbr(row, s.xp, s.wXp);
        addNbr(row, s.ym, s.wYm);
        addNbr(row, s.yp, s.wYp);
        addNbr(row, s.zm, s.wZm);
        addNbr(row, s.zp, s.wZp);
    }

    const bool anchorGauge = (!m_openTopBC && m_removeMean);
    if (anchorGauge && n > 0) {
        for (int j = 0; j < n; ++j) {
            dense[(std::size_t)j] = 0.0f;
            dense[(std::size_t)j * (std::size_t)n] = 0.0f;
        }
        dense[0] = 1.0f;
    }

    L.directSolveCholesky = dense;
    for (int row = 0; row < n; ++row) {
        const std::size_t rowBase = (std::size_t)row * (std::size_t)n;
        for (int col = 0; col <= row; ++col) {
            float sum = L.directSolveCholesky[rowBase + (std::size_t)col];
            for (int k = 0; k < col; ++k) {
                sum -= L.directSolveCholesky[rowBase + (std::size_t)k] *
                       L.directSolveCholesky[(std::size_t)col * (std::size_t)n + (std::size_t)k];
            }
            if (row == col) {
                if (!(sum > 1.0e-9f) || !std::isfinite(sum)) {
                    L.directSolveCholesky.clear();
                    return;
                }
                L.directSolveCholesky[rowBase + (std::size_t)col] = std::sqrt(sum);
            } else {
                const float diag = L.directSolveCholesky[(std::size_t)col * (std::size_t)n + (std::size_t)col];
                if (!(diag > 0.0f) || !std::isfinite(diag)) {
                    L.directSolveCholesky.clear();
                    return;
                }
                L.directSolveCholesky[rowBase + (std::size_t)col] = sum / diag;
            }
        }
    }

    L.directSolveScratch0.assign((std::size_t)n, 0.0f);
    L.directSolveScratch1.assign((std::size_t)n, 0.0f);
    L.directSolveAnchorsGauge = anchorGauge;
    L.directSolveValid = true;
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

    mgTransfers.clear();
    if (mgLevels.size() > 1u) {
        mgTransfers.resize(mgLevels.size() - 1u);
        for (std::size_t lev = 0; lev + 1u < mgLevels.size(); ++lev) {
            buildTransfer((int)lev, mgTransfers[lev]);
        }
    }

    if (!mgLevels.empty()) {
        buildDirectCoarseSolve(mgLevels.back());
    }
}

void PressureSolver3D::mgSmoothRBGS(int lev, int iters)
{
    MGLevel& L = mgLevels[(std::size_t)lev];
    const std::size_t total = L.stencils.size();
    if (total == 0u) return;

    float omega = 1.0f;
    if (mgUseSOR) omega = std::max(1.0f, std::min(mgSORomega, 1.9f));
    const float oneMinusOmega = 1.0f - omega;

    float* const x = L.x.data();
    const float* const b = L.b.data();
    const MGCellStencil* const stencils = L.stencils.data();

    auto smoothRange = [&](std::size_t begin, std::size_t end) {
        for (std::size_t idxStencil = begin; idxStencil < end; ++idxStencil) {
            const MGCellStencil& s = stencils[idxStencil];
            if (s.diagInv <= 0.0f) continue;

            const float sum =
                s.wXm * x[(std::size_t)s.xm] +
                s.wXp * x[(std::size_t)s.xp] +
                s.wYm * x[(std::size_t)s.ym] +
                s.wYp * x[(std::size_t)s.yp] +
                s.wZm * x[(std::size_t)s.zm] +
                s.wZp * x[(std::size_t)s.zp];

            const int cell = s.cell;
            const float xGs = sum * s.invDiagW + b[(std::size_t)cell] * s.diagInv;
            x[(std::size_t)cell] = oneMinusOmega * x[(std::size_t)cell] + omega * xGs;
        }
    };

    for (int it = 0; it < iters; ++it) {
        smoothRange(0u, L.redStencilCount);
        smoothRange(L.redStencilCount, total);
    }
}

void PressureSolver3D::mgComputeResidual(int lev)
{
    MGLevel& L = mgLevels[(std::size_t)lev];
    const std::size_t total = L.stencils.size();
    if (total == 0u) return;

    const float invDx2 = L.invDx2;
    const float* const x = L.x.data();
    const float* const b = L.b.data();
    float* const r = L.r.data();
    const MGCellStencil* const stencils = L.stencils.data();

    for (std::size_t idxStencil = 0; idxStencil < total; ++idxStencil) {
        const MGCellStencil& s = stencils[idxStencil];
        const float sum =
            s.wXm * x[(std::size_t)s.xm] +
            s.wXp * x[(std::size_t)s.xp] +
            s.wYm * x[(std::size_t)s.ym] +
            s.wYp * x[(std::size_t)s.yp] +
            s.wZm * x[(std::size_t)s.zm] +
            s.wZp * x[(std::size_t)s.zp];

        const int cell = s.cell;
        r[(std::size_t)cell] = b[(std::size_t)cell] - (s.diagW * x[(std::size_t)cell] - sum) * invDx2;
    }
}

void PressureSolver3D::mgRestrictResidual(int fineLev)
{
    MGLevel& F = mgLevels[(std::size_t)fineLev];
    MGLevel& C = mgLevels[(std::size_t)fineLev + 1u];
    MGTransfer& transfer = mgTransfers[(std::size_t)fineLev];

    std::fill(C.b.begin(), C.b.end(), 0.0f);
    std::fill(C.x.begin(), C.x.end(), 0.0f);

    const float* const r = F.r.data();
    float* const coarseB = C.b.data();

    for (const MGRestrictEntry& entry : transfer.restrictEntries) {
        float sum = 0.0f;
        for (uint8_t n = 0; n < entry.count; ++n) {
            sum += entry.weights[(std::size_t)n] * r[(std::size_t)entry.ids[(std::size_t)n]];
        }
        coarseB[(std::size_t)entry.coarseCell] = sum;
    }
}

void PressureSolver3D::mgProlongateAndAdd(int coarseLev)
{
    MGLevel& C = mgLevels[(std::size_t)coarseLev];
    MGLevel& F = mgLevels[(std::size_t)coarseLev - 1u];
    MGTransfer& transfer = mgTransfers[(std::size_t)coarseLev - 1u];

    const float* const coarseX = C.x.data();
    float* const fineX = F.x.data();

    for (const MGProlongEntry& entry : transfer.prolongEntries) {
        float correction = 0.0f;
        for (uint8_t n = 0; n < entry.count; ++n) {
            correction += entry.weights[(std::size_t)n] * coarseX[(std::size_t)entry.ids[(std::size_t)n]];
        }
        fineX[(std::size_t)entry.fineCell] += correction;
    }
}

void PressureSolver3D::mgVCycle(int lev)
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

    if (m_dirty) rebuildOperator();
    ensurePCGBuffers();

    MGLevel& F = mgLevels[0];
    const int N = F.nx * F.ny * F.nz;
    if ((int)p.size() != N) p.assign((std::size_t)N, 0.0f);
    if ((int)rhs.size() != N) {
        m_lastIters = 0;
        return;
    }

    if (F.stencils.empty() || m_fluidCells.empty()) {
        std::fill(p.begin(), p.end(), 0.0f);
        m_lastIters = 0;
        return;
    }

    const float tolRhs = tolPredDiv / std::max(1e-8f, dtForPredDiv);
    const float relTol = std::max(0.0f, mgRelativeTol);

    const float bInf = maxAbsFluid(rhs);
    if (!(bInf > 0.0f) || bInf <= tolRhs) {
        removeMean(p);
        m_lastIters = 0;
        return;
    }

    applyA(p, m_Ap);
    for (int id : m_fluidCells) {
        m_r[(std::size_t)id] = rhs[(std::size_t)id] - m_Ap[(std::size_t)id];
    }

    float rInf = maxAbsFluid(m_r);
    if (!(rInf > 0.0f) || rInf <= tolRhs) {
        removeMean(p);
        m_lastIters = 0;
        return;
    }

    const float rInf0 = std::max(rInf, 1.0e-30f);

    auto applyPrecond = [&](const std::vector<float>& r, std::vector<float>& z) {
        for (int id : m_fluidCells) {
            F.b[(std::size_t)id] = r[(std::size_t)id];
            F.x[(std::size_t)id] = 0.0f;
        }
        mgVCycle(0);
        for (int id : m_fluidCells) {
            z[(std::size_t)id] = F.x[(std::size_t)id];
        }
        removeMean(z);
    };

    applyPrecond(m_r, m_z);
    for (int id : m_fluidCells) {
        m_d[(std::size_t)id] = m_z[(std::size_t)id];
    }

    float deltaNew = dotFluid(m_r, m_z);
    if (!std::isfinite(deltaNew) || deltaNew <= 1.0e-30f) {
        solvePCG(p, rhs, 120, tolPredDiv, dtForPredDiv);
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
        for (int id : m_fluidCells) {
            p[(std::size_t)id] += alpha * m_d[(std::size_t)id];
            m_r[(std::size_t)id] -= alpha * m_q[(std::size_t)id];
        }

        rInf = maxAbsFluid(m_r);
        if (!std::isfinite(rInf)) {
            fallbackPCG = true;
            break;
        }
        if (rInf <= tolRhs) break;
        if (relTol > 0.0f && rInf <= relTol * rInf0) break;

        applyPrecond(m_r, m_z);
        const float deltaOld = deltaNew;
        deltaNew = dotFluid(m_r, m_z);
        if (!std::isfinite(deltaNew) || deltaNew <= 1.0e-30f) {
            fallbackPCG = true;
            break;
        }

        const float beta = deltaNew / (deltaOld + 1.0e-30f);
        for (int id : m_fluidCells) {
            m_d[(std::size_t)id] = m_z[(std::size_t)id] + beta * m_d[(std::size_t)id];
        }
    }

    removeMean(p);
    m_lastIters = itUsed;

    if (fallbackPCG) {
        solvePCG(p, rhs, 120, tolPredDiv, dtForPredDiv);
    }
}
