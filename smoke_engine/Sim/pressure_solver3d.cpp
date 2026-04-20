#include "pressure_solver3d.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include "chunk_worker_pool.h"

namespace {
inline ChunkWorkerPool& pressureSolverWorkerPool() { return sharedChunkWorkerPool(); }
constexpr int kPressure3DParallelThreshold = 65536;
constexpr int kPressure3DMinChunk = 4096;
constexpr int kPressure3DTransferRowParallelThreshold = 128;
constexpr int kPressure3DTransferMinRows = 4;
// Dense Cholesky on the coarsest MG level is excellent for small systems, but
// for large free-surface water regions the coarsest compact mask can still have
// hundreds or thousands of unknowns. Building an O(n^2) dense matrix and doing
// an O(n^3) factorization every frame then causes catastrophic pressure spikes.
// Keep the direct solve only for genuinely small coarse systems and fall back to
// the existing coarse RBGS smoother otherwise.
constexpr int kPressure3DCoarseDirectMaxUnknowns = 256;

inline int pressure3DResolvedChunkSize(ChunkWorkerPool& pool, int count, int minChunk) {
    const int safeMinChunk = std::max(1, minChunk);
    const int totalWorkers = std::max(1, pool.maxWorkers());
    const int maxUsefulWorkers = std::min(totalWorkers, std::max(1, (count + safeMinChunk - 1) / safeMinChunk));
    const int desiredChunks = std::max(maxUsefulWorkers, maxUsefulWorkers * 2);
    return std::max(safeMinChunk, (count + desiredChunks - 1) / desiredChunks);
}

template <typename Fn> inline void parallelForPressure3D(int count, Fn&& fn) {
    if (count <= 0) return;
    ChunkWorkerPool& pool = pressureSolverWorkerPool();
    if (pool.maxWorkers() <= 1 || count < kPressure3DParallelThreshold) { fn(0, count); return; }
    pool.parallelFor(count, kPressure3DMinChunk, std::forward<Fn>(fn));
}

template <typename T, typename ChunkFn, typename CombineFn>
inline T parallelReducePressure3D(int count,
                                  int minChunk,
                                  T initValue,
                                  ChunkFn&& chunkFn,
                                  CombineFn&& combineFn,
                                  int threshold = kPressure3DParallelThreshold) {
    if (count <= 0) return initValue;

    ChunkWorkerPool& pool = pressureSolverWorkerPool();
    const int safeMinChunk = std::max(1, minChunk);
    const int chunkSize = pressure3DResolvedChunkSize(pool, count, safeMinChunk);
    const int chunkCount = std::max(1, (count + chunkSize - 1) / chunkSize);

    if (pool.maxWorkers() <= 1 || count < threshold || chunkCount <= 1) {
        return chunkFn(0, count);
    }

    std::vector<T> partials((std::size_t)chunkCount, initValue);
    pool.parallelFor(count, safeMinChunk, [&](int begin, int end) {
        const int chunkIndex = std::min(chunkCount - 1, begin / chunkSize);
        partials[(std::size_t)chunkIndex] = chunkFn(begin, end);
    });

    T result = initValue;
    for (const T& partial : partials) {
        result = combineFn(result, partial);
    }
    return result;
}

template <typename Fn> inline void parallelForPressure3DTransferRows(int rowCount, Fn&& fn) {
    if (rowCount <= 0) return;
    ChunkWorkerPool& pool = pressureSolverWorkerPool();
    if (pool.maxWorkers() <= 1 || rowCount < kPressure3DTransferRowParallelThreshold) { fn(0, rowCount); return; }
    pool.parallelFor(rowCount, kPressure3DTransferMinRows, std::forward<Fn>(fn));
}

struct Pressure3DMaxAbsReduction {
    float maxValue = 0.0f;
    bool nonFinite = false;
};

struct Pressure3DMaxAbsDotReduction {
    float maxValue = 0.0f;
    double dot = 0.0;
    bool nonFinite = false;
};

inline Pressure3DMaxAbsReduction combinePressure3DMaxAbs(const Pressure3DMaxAbsReduction& a,
                                                         const Pressure3DMaxAbsReduction& b) {
    Pressure3DMaxAbsReduction out;
    out.maxValue = std::max(a.maxValue, b.maxValue);
    out.nonFinite = a.nonFinite || b.nonFinite;
    return out;
}

inline Pressure3DMaxAbsDotReduction combinePressure3DMaxAbsDot(const Pressure3DMaxAbsDotReduction& a,
                                                               const Pressure3DMaxAbsDotReduction& b) {
    Pressure3DMaxAbsDotReduction out;
    out.maxValue = std::max(a.maxValue, b.maxValue);
    out.dot = a.dot + b.dot;
    out.nonFinite = a.nonFinite || b.nonFinite;
    return out;
}
}  // namespace

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

void PressureSolver3D::detectCompactDenseBox()
{
    m_compactDenseBoxValid = false;
    m_compactBoxNx = 0;
    m_compactBoxNy = 0;
    m_compactBoxNz = 0;
    m_compactBoxOpenTopDirichlet = false;

    const int nx = m_nx;
    const int ny = m_ny;
    const int nz = m_nz;
    const int N = nx * ny * nz;
    if (N <= 0) return;

    int fluidCount = 0;
    int minI = nx;
    int minJ = ny;
    int minK = nz;
    int maxI = -1;
    int maxJ = -1;
    int maxK = -1;

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                if (!m_fluid[(std::size_t)idx(i, j, k, nx, ny)]) continue;
                ++fluidCount;
                minI = std::min(minI, i);
                minJ = std::min(minJ, j);
                minK = std::min(minK, k);
                maxI = std::max(maxI, i);
                maxJ = std::max(maxJ, j);
                maxK = std::max(maxK, k);
            }
        }
    }

    if (fluidCount <= 0) return;

    const int i0 = minI;
    const int i1 = maxI + 1;
    const int j0 = minJ;
    const int j1 = maxJ + 1;
    const int k0 = minK;
    const int k1 = maxK + 1;
    const int extentI = i1 - i0;
    const int extentJ = j1 - j0;
    const int extentK = k1 - k0;
    const int boxVolume = extentI * extentJ * extentK;
    if (boxVolume != fluidCount) return;

    // The structured dense-box fast path is safe for genuinely volumetric blocks,
    // but one-cell-thick slabs can appear in settled free-surface water. On those
    // degenerate boxes, the specialized multigrid smoothing/transfer kernels can
    // over-amplify pressure corrections and blow up the solve. Fall back to the
    // generic masked path for any axis with only a single active cell.
    if (extentI <= 1 || extentJ <= 1 || extentK <= 1) return;

    for (int k = k0; k < k1; ++k) {
        for (int j = j0; j < j1; ++j) {
            for (int i = i0; i < i1; ++i) {
                if (!m_fluid[(std::size_t)idx(i, j, k, nx, ny)]) return;
            }
        }
    }

    auto isOne = [](float v) {
        return std::fabs(v - 1.0f) <= 1.0e-6f;
    };

    for (int k = k0; k < k1; ++k) {
        for (int j = j0; j < j1; ++j) {
            for (int i = i0 + 1; i < i1; ++i) {
                if (!isOne(m_faceOpenU[uIdx(i, j, k, nx, ny)])) return;
            }
        }
    }

    for (int k = k0; k < k1; ++k) {
        for (int j = j0 + 1; j < j1; ++j) {
            for (int i = i0; i < i1; ++i) {
                if (!isOne(m_faceOpenV[vIdx(i, j, k, nx, ny)])) return;
            }
        }
    }

    for (int k = k0 + 1; k < k1; ++k) {
        for (int j = j0; j < j1; ++j) {
            for (int i = i0; i < i1; ++i) {
                if (!isOne(m_faceOpenW[wIdx(i, j, k, nx, ny)])) return;
            }
        }
    }

    bool openTopDirichlet = false;
    if (m_openTopBC && j1 == ny) {
        openTopDirichlet = true;
        for (int k = k0; k < k1; ++k) {
            for (int i = i0; i < i1; ++i) {
                if (!isOne(m_faceOpenV[vIdx(i, j1, k, nx, ny)])) return;
            }
        }
    }

    m_compactDenseBoxValid = true;
    m_compactBoxNx = i1 - i0;
    m_compactBoxNy = j1 - j0;
    m_compactBoxNz = k1 - k0;
    m_compactBoxOpenTopDirichlet = openTopDirichlet;
}

void PressureSolver3D::ensurePCGBuffers()
{
    const int fluidCount = (int)m_fluidCells.size();
    if ((int)m_pCompact.size() != fluidCount) m_pCompact.resize((std::size_t)fluidCount);
    if ((int)m_rhsCompact.size() != fluidCount) m_rhsCompact.resize((std::size_t)fluidCount);
    if ((int)m_r.size() != fluidCount) m_r.resize((std::size_t)fluidCount);
    if ((int)m_z.size() != fluidCount) m_z.resize((std::size_t)fluidCount);
    if ((int)m_d.size() != fluidCount) m_d.resize((std::size_t)fluidCount);
    if ((int)m_q.size() != fluidCount) m_q.resize((std::size_t)fluidCount);
    if ((int)m_Ap.size() != fluidCount) m_Ap.resize((std::size_t)fluidCount);
}

void PressureSolver3D::packFluidField(const std::vector<float>& full, std::vector<float>& compact) const
{
    const int fluidCount = (int)m_fluidCells.size();
    if ((int)compact.size() != fluidCount) compact.resize((std::size_t)fluidCount);
    if (fluidCount <= 0) return;

    const int* const fluidCells = m_fluidCells.data();
    const float* const fullData = full.data();
    float* const compactData = compact.data();
    parallelForPressure3D(fluidCount, [&](int begin, int end) {
        for (int idxFluid = begin; idxFluid < end; ++idxFluid) {
            compactData[(std::size_t)idxFluid] = fullData[(std::size_t)fluidCells[idxFluid]];
        }
    });
}

void PressureSolver3D::unpackFluidField(const std::vector<float>& compact, std::vector<float>& full) const
{
    const int N = m_nx * m_ny * m_nz;
    if ((int)full.size() != N) full.assign((std::size_t)N, 0.0f);

    const int fluidCount = (int)m_fluidCells.size();
    if (fluidCount <= 0) return;

    const int* const fluidCells = m_fluidCells.data();
    const float* const compactData = compact.data();
    float* const fullData = full.data();
    parallelForPressure3D(fluidCount, [&](int begin, int end) {
        for (int idxFluid = begin; idxFluid < end; ++idxFluid) {
            fullData[(std::size_t)fluidCells[idxFluid]] = compactData[(std::size_t)idxFluid];
        }
    });
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

    const int fluidCount = (int)m_fluidCells.size();
    m_gridToCompact.assign((std::size_t)N, -1);
    for (int idxFluid = 0; idxFluid < fluidCount; ++idxFluid) {
        m_gridToCompact[(std::size_t)m_fluidCells[(std::size_t)idxFluid]] = idxFluid;
    }

    m_compactStencils.assign((std::size_t)fluidCount, CompactCellStencil{});
    m_compactDiagInv.assign((std::size_t)fluidCount, 0.0f);
    for (int idxFluid = 0; idxFluid < fluidCount; ++idxFluid) {
        const int id = m_fluidCells[(std::size_t)idxFluid];
        CompactCellStencil s;
        auto mapNeighbor = [&](int gridNeighbor, float weight, int& compactNeighbor, float& compactWeight) {
            if (gridNeighbor >= 0 && weight > 0.0f) {
                const int compact = m_gridToCompact[(std::size_t)gridNeighbor];
                compactNeighbor = (compact >= 0) ? compact : 0;
                compactWeight = weight;
            } else {
                compactNeighbor = 0;
                compactWeight = 0.0f;
            }
        };

        mapNeighbor(m_xm[(std::size_t)id], m_wXm[(std::size_t)id], s.xm, s.wXm);
        mapNeighbor(m_xp[(std::size_t)id], m_wXp[(std::size_t)id], s.xp, s.wXp);
        mapNeighbor(m_ym[(std::size_t)id], m_wYm[(std::size_t)id], s.ym, s.wYm);
        mapNeighbor(m_yp[(std::size_t)id], m_wYp[(std::size_t)id], s.yp, s.wYp);
        mapNeighbor(m_zm[(std::size_t)id], m_wZm[(std::size_t)id], s.zm, s.wZm);
        mapNeighbor(m_zp[(std::size_t)id], m_wZp[(std::size_t)id], s.zp, s.wZp);
        s.diagW = m_diagW[(std::size_t)id];
        s.diagInv = m_diagInv[(std::size_t)id];
        m_compactStencils[(std::size_t)idxFluid] = s;
        m_compactDiagInv[(std::size_t)idxFluid] = s.diagInv;
    }

    detectCompactDenseBox();
    m_dirty = false;
}

void PressureSolver3D::applyACompact(const std::vector<float>& x, std::vector<float>& Ax) const
{
    const int fluidCount = (int)m_compactStencils.size();
    if ((int)Ax.size() != fluidCount) Ax.resize((std::size_t)fluidCount);
    if (fluidCount <= 0) return;

    const float* const xData = x.data();
    float* const AxData = Ax.data();

    if (m_compactDenseBoxValid && fluidCount == m_compactBoxNx * m_compactBoxNy * m_compactBoxNz) {
        const int boxNx = m_compactBoxNx;
        const int boxNy = m_compactBoxNy;
        const int boxNz = m_compactBoxNz;
        const int sliceStride = boxNx * boxNy;
        const bool openTopDirichlet = m_compactBoxOpenTopDirichlet;
        const float invDx2 = m_invDx2;

        parallelForPressure3D(fluidCount, [&](int begin, int end) {
            for (int c = begin; c < end; ++c) {
                const int localK = c / sliceStride;
                const int rem = c - localK * sliceStride;
                const int localJ = rem / boxNx;
                const int localI = rem - localJ * boxNx;

                float sum = 0.0f;
                float diagW = 0.0f;

                if (localI > 0)              { sum += xData[(std::size_t)(c - 1)];           diagW += 1.0f; }
                if (localI + 1 < boxNx)      { sum += xData[(std::size_t)(c + 1)];           diagW += 1.0f; }
                if (localJ > 0)              { sum += xData[(std::size_t)(c - boxNx)];       diagW += 1.0f; }
                if (localJ + 1 < boxNy)      { sum += xData[(std::size_t)(c + boxNx)];       diagW += 1.0f; }
                else if (openTopDirichlet)   {                                              diagW += 1.0f; }
                if (localK > 0)              { sum += xData[(std::size_t)(c - sliceStride)]; diagW += 1.0f; }
                if (localK + 1 < boxNz)      { sum += xData[(std::size_t)(c + sliceStride)]; diagW += 1.0f; }

                AxData[(std::size_t)c] = (diagW * xData[(std::size_t)c] - sum) * invDx2;
            }
        });
        return;
    }

    const CompactCellStencil* const stencils = m_compactStencils.data();
    parallelForPressure3D(fluidCount, [&](int begin, int end) {
        for (int idxFluid = begin; idxFluid < end; ++idxFluid) {
            const CompactCellStencil& s = stencils[(std::size_t)idxFluid];
            const float sum =
                s.wXm * xData[(std::size_t)s.xm] +
                s.wXp * xData[(std::size_t)s.xp] +
                s.wYm * xData[(std::size_t)s.ym] +
                s.wYp * xData[(std::size_t)s.yp] +
                s.wZm * xData[(std::size_t)s.zm] +
                s.wZp * xData[(std::size_t)s.zp];
            AxData[(std::size_t)idxFluid] = (s.diagW * xData[(std::size_t)idxFluid] - sum) * m_invDx2;
        }
    });
}

float PressureSolver3D::dotCompact(const std::vector<float>& a, const std::vector<float>& b) const
{
    const int fluidCount = (int)m_compactStencils.size();
    if (fluidCount <= 0) return 0.0f;

    const float* const aData = a.data();
    const float* const bData = b.data();
    const double sum = parallelReducePressure3D<double>(
        fluidCount,
        kPressure3DMinChunk,
        0.0,
        [&](int begin, int end) {
            double local = 0.0;
            for (int idxFluid = begin; idxFluid < end; ++idxFluid) {
                local += (double)aData[(std::size_t)idxFluid] * (double)bData[(std::size_t)idxFluid];
            }
            return local;
        },
        [](double lhs, double rhs) { return lhs + rhs; });
    return (float)sum;
}

float PressureSolver3D::maxAbsCompact(const std::vector<float>& a) const
{
    const int fluidCount = (int)m_compactStencils.size();
    if (fluidCount <= 0) return 0.0f;

    const float* const aData = a.data();
    const Pressure3DMaxAbsReduction reduced = parallelReducePressure3D<Pressure3DMaxAbsReduction>(
        fluidCount,
        kPressure3DMinChunk,
        Pressure3DMaxAbsReduction{},
        [&](int begin, int end) {
            Pressure3DMaxAbsReduction local;
            for (int idxFluid = begin; idxFluid < end; ++idxFluid) {
                const float v = std::fabs(aData[(std::size_t)idxFluid]);
                if (!std::isfinite(v)) {
                    local.nonFinite = true;
                    break;
                }
                local.maxValue = std::max(local.maxValue, v);
            }
            return local;
        },
        combinePressure3DMaxAbs);
    return reduced.nonFinite ? std::numeric_limits<float>::infinity() : reduced.maxValue;
}

void PressureSolver3D::removeMeanCompact(std::vector<float>& p) const
{
    if (!m_removeMean) return;

    const int fluidCount = (int)m_compactStencils.size();
    if (fluidCount <= 0) return;

    float* const pData = p.data();
    const double sum = parallelReducePressure3D<double>(
        fluidCount,
        kPressure3DMinChunk,
        0.0,
        [&](int begin, int end) {
            double local = 0.0;
            for (int idxFluid = begin; idxFluid < end; ++idxFluid) {
                local += (double)pData[(std::size_t)idxFluid];
            }
            return local;
        },
        [](double lhs, double rhs) { return lhs + rhs; });

    const float mean = (float)(sum / (double)fluidCount);
    if (!std::isfinite(mean) || std::fabs(mean) <= 0.0f) return;

    parallelForPressure3D(fluidCount, [&](int begin, int end) {
        for (int idxFluid = begin; idxFluid < end; ++idxFluid) {
            pData[(std::size_t)idxFluid] -= mean;
        }
    });
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

    if ((int)p.size() != N) p.assign((std::size_t)N, 0.0f);
    if ((int)rhs.size() != N) {
        m_lastIters = 0;
        return 0;
    }

    const int fluidCount = (int)m_fluidCells.size();
    if (fluidCount <= 0) {
        m_lastIters = 0;
        return 0;
    }

    ensurePCGBuffers();
    packFluidField(p, m_pCompact);
    packFluidField(rhs, m_rhsCompact);

    const float tolRhs = tolPredDiv / std::max(1e-8f, dtForPredDiv);

    const float bInf = maxAbsCompact(m_rhsCompact);
    if (!std::isfinite(bInf) || bInf <= tolRhs) {
        removeMeanCompact(m_pCompact);
        unpackFluidField(m_pCompact, p);
        m_lastIters = 0;
        return 0;
    }

    const float* const rhsData = m_rhsCompact.data();
    float* const pData = m_pCompact.data();
    float* const rData = m_r.data();
    float* const zData = m_z.data();
    float* const dData = m_d.data();
    float* const qData = m_q.data();
    const float* const diagInvData = m_compactDiagInv.data();

    applyACompact(m_pCompact, m_Ap);
    const float* const ApData = m_Ap.data();

    const Pressure3DMaxAbsDotReduction initReduce = parallelReducePressure3D<Pressure3DMaxAbsDotReduction>(
        fluidCount,
        kPressure3DMinChunk,
        Pressure3DMaxAbsDotReduction{},
        [&](int begin, int end) {
            Pressure3DMaxAbsDotReduction local;
            for (int idxFluid = begin; idxFluid < end; ++idxFluid) {
                const float r = rhsData[(std::size_t)idxFluid] - ApData[(std::size_t)idxFluid];
                const float z = r * diagInvData[(std::size_t)idxFluid];
                rData[(std::size_t)idxFluid] = r;
                zData[(std::size_t)idxFluid] = z;
                dData[(std::size_t)idxFluid] = z;
                const float absR = std::fabs(r);
                if (!std::isfinite(absR) || !std::isfinite(z)) {
                    local.nonFinite = true;
                    break;
                }
                local.maxValue = std::max(local.maxValue, absR);
                local.dot += (double)r * (double)z;
            }
            return local;
        },
        combinePressure3DMaxAbsDot);

    const float rInf0 = std::max(initReduce.maxValue, 1e-30f);
    float deltaNew = (float)initReduce.dot;
    if (initReduce.nonFinite || !std::isfinite(deltaNew) || deltaNew <= 1e-30f) {
        removeMeanCompact(m_pCompact);
        unpackFluidField(m_pCompact, p);
        m_lastIters = 0;
        return 0;
    }

    const float relTol = 1e-5f;
    int itUsed = 0;

    for (int it = 0; it < std::max(1, maxIters); ++it) {
        itUsed = it + 1;

        applyACompact(m_d, m_q);
        const float dq = dotCompact(m_d, m_q);
        if (!std::isfinite(dq) || std::fabs(dq) < 1e-30f) break;

        const float alpha = deltaNew / dq;
        const Pressure3DMaxAbsReduction updateReduce = parallelReducePressure3D<Pressure3DMaxAbsReduction>(
            fluidCount,
            kPressure3DMinChunk,
            Pressure3DMaxAbsReduction{},
            [&](int begin, int end) {
                Pressure3DMaxAbsReduction local;
                for (int idxFluid = begin; idxFluid < end; ++idxFluid) {
                    pData[(std::size_t)idxFluid] += alpha * dData[(std::size_t)idxFluid];
                    const float r = rData[(std::size_t)idxFluid] - alpha * qData[(std::size_t)idxFluid];
                    rData[(std::size_t)idxFluid] = r;
                    const float absR = std::fabs(r);
                    if (!std::isfinite(absR)) {
                        local.nonFinite = true;
                        break;
                    }
                    local.maxValue = std::max(local.maxValue, absR);
                }
                return local;
            },
            combinePressure3DMaxAbs);

        const float rInf = updateReduce.nonFinite ? std::numeric_limits<float>::infinity() : updateReduce.maxValue;
        if (!std::isfinite(rInf)) break;
        if (rInf <= tolRhs) break;
        if (relTol > 0.0f && rInf <= relTol * rInf0) break;

        const Pressure3DMaxAbsDotReduction precondReduce = parallelReducePressure3D<Pressure3DMaxAbsDotReduction>(
            fluidCount,
            kPressure3DMinChunk,
            Pressure3DMaxAbsDotReduction{},
            [&](int begin, int end) {
                Pressure3DMaxAbsDotReduction local;
                for (int idxFluid = begin; idxFluid < end; ++idxFluid) {
                    const float z = rData[(std::size_t)idxFluid] * diagInvData[(std::size_t)idxFluid];
                    zData[(std::size_t)idxFluid] = z;
                    if (!std::isfinite(z)) {
                        local.nonFinite = true;
                        break;
                    }
                    local.dot += (double)rData[(std::size_t)idxFluid] * (double)z;
                }
                return local;
            },
            combinePressure3DMaxAbsDot);

        const float deltaOld = deltaNew;
        deltaNew = (float)precondReduce.dot;
        if (precondReduce.nonFinite || !std::isfinite(deltaNew) || deltaNew <= 1e-30f) break;

        const float beta = deltaNew / (deltaOld + 1e-30f);
        parallelForPressure3D(fluidCount, [&](int begin, int end) {
            for (int idxFluid = begin; idxFluid < end; ++idxFluid) {
                dData[(std::size_t)idxFluid] = zData[(std::size_t)idxFluid] + beta * dData[(std::size_t)idxFluid];
            }
        });
    }

    removeMeanCompact(m_pCompact);
    unpackFluidField(m_pCompact, p);
    m_lastIters = itUsed;
    return itUsed;
}


void PressureSolver3D::detectDenseBox(MGLevel& L) const
{
    L.denseBoxValid = false;
    L.boxI0 = L.boxI1 = 0;
    L.boxJ0 = L.boxJ1 = 0;
    L.boxK0 = L.boxK1 = 0;
    L.boxOpenTopDirichlet = false;

    const int nx = L.nx;
    const int ny = L.ny;
    const int nz = L.nz;
    const int N = nx * ny * nz;
    if (N <= 0) return;

    int fluidCount = 0;
    int minI = nx, minJ = ny, minK = nz;
    int maxI = -1, maxJ = -1, maxK = -1;

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                if (!L.fluid[(std::size_t)idx(i, j, k, nx, ny)]) continue;
                ++fluidCount;
                minI = std::min(minI, i); maxI = std::max(maxI, i);
                minJ = std::min(minJ, j); maxJ = std::max(maxJ, j);
                minK = std::min(minK, k); maxK = std::max(maxK, k);
            }
        }
    }

    if (fluidCount <= 0) return;

    const int i0 = minI;
    const int i1 = maxI + 1;
    const int j0 = minJ;
    const int j1 = maxJ + 1;
    const int k0 = minK;
    const int k1 = maxK + 1;
    const int boxVolume = (i1 - i0) * (j1 - j0) * (k1 - k0);
    if (boxVolume != fluidCount) return;

    for (int k = k0; k < k1; ++k) {
        for (int j = j0; j < j1; ++j) {
            for (int i = i0; i < i1; ++i) {
                if (!L.fluid[(std::size_t)idx(i, j, k, nx, ny)]) {
                    return;
                }
            }
        }
    }

    auto isOne = [](float v) {
        return std::fabs(v - 1.0f) <= 1.0e-6f;
    };

    for (int k = k0; k < k1; ++k) {
        for (int j = j0; j < j1; ++j) {
            for (int i = i0 + 1; i < i1; ++i) {
                if (!isOne(L.faceOpenU[uIdx(i, j, k, nx, ny)])) return;
            }
        }
    }

    for (int k = k0; k < k1; ++k) {
        for (int j = j0 + 1; j < j1; ++j) {
            for (int i = i0; i < i1; ++i) {
                if (!isOne(L.faceOpenV[vIdx(i, j, k, nx, ny)])) return;
            }
        }
    }

    for (int k = k0 + 1; k < k1; ++k) {
        for (int j = j0; j < j1; ++j) {
            for (int i = i0; i < i1; ++i) {
                if (!isOne(L.faceOpenW[wIdx(i, j, k, nx, ny)])) return;
            }
        }
    }

    bool openTopDirichlet = false;
    if (m_openTopBC && j1 == ny) {
        openTopDirichlet = true;
        for (int k = k0; k < k1; ++k) {
            for (int i = i0; i < i1; ++i) {
                if (!isOne(L.faceOpenV[vIdx(i, j1, k, nx, ny)])) {
                    return;
                }
            }
        }
    }

    L.denseBoxValid = true;
    L.boxI0 = i0; L.boxI1 = i1;
    L.boxJ0 = j0; L.boxJ1 = j1;
    L.boxK0 = k0; L.boxK1 = k1;
    L.boxOpenTopDirichlet = openTopDirichlet;
}


void PressureSolver3D::buildLevelStencil(MGLevel& L) const
{
    const int nx = L.nx;
    const int ny = L.ny;
    const int nz = L.nz;
    const int N = nx * ny * nz;
    detectDenseBox(L);

    L.fluidCells.clear();
    L.fluidCells.reserve((std::size_t)N);
    L.gridToCompact.assign((std::size_t)N, -1);

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const int id = idx(i, j, k, nx, ny);
                if (!L.fluid[(std::size_t)id]) continue;
                const int compact = (int)L.fluidCells.size();
                L.fluidCells.push_back(id);
                L.gridToCompact[(std::size_t)id] = compact;
            }
        }
    }

    const int fluidCount = (int)L.fluidCells.size();
    L.x.assign((std::size_t)fluidCount, 0.0f);
    L.b.assign((std::size_t)fluidCount, 0.0f);
    L.r.assign((std::size_t)fluidCount, 0.0f);

    std::vector<MGCellStencil> red;
    std::vector<MGCellStencil> black;
    red.reserve((std::size_t)fluidCount / 2u + 1u);
    black.reserve((std::size_t)fluidCount / 2u + 1u);

    for (int compact = 0; compact < fluidCount; ++compact) {
        const int id = L.fluidCells[(std::size_t)compact];
        const int i = id % nx;
        const int tmp = id / nx;
        const int j = tmp % ny;
        const int k = tmp / ny;

        MGCellStencil stencil;
        stencil.cell = compact;

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
                    const int nbrCompact = L.gridToCompact[(std::size_t)nid];
                    if (nbrCompact >= 0) {
                        slotNbr = nbrCompact;
                        wNbr = wFace;
                    }
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

    transfer.denseBoxStructured = false;
    transfer.restrictEntries.clear();
    transfer.prolongEntries.clear();

    if (F.denseBoxValid && C.denseBoxValid) {
        const int expectI0 = F.boxI0 >> 1;
        const int expectI1 = ((F.boxI1 - 1) >> 1) + 1;
        const int expectJ0 = F.boxJ0 >> 1;
        const int expectJ1 = ((F.boxJ1 - 1) >> 1) + 1;
        const int expectK0 = F.boxK0 >> 1;
        const int expectK1 = ((F.boxK1 - 1) >> 1) + 1;
        if (C.boxI0 == expectI0 && C.boxI1 == expectI1 &&
            C.boxJ0 == expectJ0 && C.boxJ1 == expectJ1 &&
            C.boxK0 == expectK0 && C.boxK1 == expectK1 &&
            C.boxOpenTopDirichlet == F.boxOpenTopDirichlet) {
            transfer.denseBoxStructured = true;
            return;
        }
    }

    transfer.restrictEntries.reserve(C.fluidCells.size());
    transfer.prolongEntries.reserve(F.fluidCells.size());

    static const float w1[3] = { 1.0f, 2.0f, 1.0f };

    const int coarseCount = (int)C.fluidCells.size();
    for (int coarseCompact = 0; coarseCompact < coarseCount; ++coarseCompact) {
        MGRestrictEntry entry;
        entry.coarseCell = coarseCompact;

        const int cid = C.fluidCells[(std::size_t)coarseCompact];
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
                    const int fineCompact = F.gridToCompact[(std::size_t)fid];
                    if (fineCompact < 0) continue;

                    const float wgt = w1[di + 1] * w1[dj + 1] * w1[dk + 1];
                    entry.ids[(std::size_t)entry.count] = fineCompact;
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

    const int fineCount = (int)F.fluidCells.size();
    for (int fineCompact = 0; fineCompact < fineCount; ++fineCompact) {
        MGProlongEntry entry;
        entry.fineCell = fineCompact;

        const int fid = F.fluidCells[(std::size_t)fineCompact];
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
            const int coarseCompactIdx = C.gridToCompact[(std::size_t)cidLocal];
            if (coarseCompactIdx < 0) return;

            for (uint8_t n = 0; n < entry.count; ++n) {
                if (entry.ids[(std::size_t)n] == coarseCompactIdx) {
                    entry.weights[(std::size_t)n] += weight;
                    return;
                }
            }

            entry.ids[(std::size_t)entry.count] = coarseCompactIdx;
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

    const int fluidCount = (int)L.fluidCells.size();
    const int n = (int)L.stencils.size();
    if (n <= 0 || fluidCount <= 0) return;
    if (n > kPressure3DCoarseDirectMaxUnknowns) return;

    L.directSolveCells.resize((std::size_t)n);
    L.directSolveCompactIndex.assign((std::size_t)fluidCount, -1);
    for (int row = 0; row < n; ++row) {
        const int cell = L.stencils[(std::size_t)row].cell;
        L.directSolveCells[(std::size_t)row] = cell;
        L.directSolveCompactIndex[(std::size_t)cell] = row;
    }

    std::vector<float> dense((std::size_t)n * (std::size_t)n, 0.0f);
    auto addNbr = [&](int row, int nbrCell, float weight) {
        if (weight <= 0.0f || nbrCell < 0 || nbrCell >= fluidCount) return;
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

    if (L.denseBoxValid) {
        const int boxNx = L.boxI1 - L.boxI0;
        const int boxNy = L.boxJ1 - L.boxJ0;
        const int boxNz = L.boxK1 - L.boxK0;
        const int sliceStride = boxNx * boxNy;
        const bool openTopDirichlet = L.boxOpenTopDirichlet;

        float* const x = L.x.data();
        const float* const b = L.b.data();

        for (int it = 0; it < iters; ++it) {
            for (int color = 0; color < 2; ++color) {
                for (int lk = 0; lk < boxNz; ++lk) {
                    const int gk = L.boxK0 + lk;
                    for (int lj = 0; lj < boxNy; ++lj) {
                        const int gj = L.boxJ0 + lj;
                        int li = 0;
                        if ((((L.boxI0 + li) + gj + gk) & 1) != color) ++li;
                        int cell = li + boxNx * (lj + boxNy * lk);
                        for (; li < boxNx; li += 2, cell += 2) {
                            float sum = 0.0f;
                            float diagW = 0.0f;

                            if (li > 0)          { sum += x[(std::size_t)(cell - 1)];          diagW += 1.0f; }
                            if (li + 1 < boxNx)  { sum += x[(std::size_t)(cell + 1)];          diagW += 1.0f; }
                            if (lj > 0)          { sum += x[(std::size_t)(cell - boxNx)];      diagW += 1.0f; }
                            if (lj + 1 < boxNy)  { sum += x[(std::size_t)(cell + boxNx)];      diagW += 1.0f; }
                            else if (openTopDirichlet) {                                    diagW += 1.0f; }
                            if (lk > 0)          { sum += x[(std::size_t)(cell - sliceStride)]; diagW += 1.0f; }
                            if (lk + 1 < boxNz)  { sum += x[(std::size_t)(cell + sliceStride)]; diagW += 1.0f; }

                            const float xGs = (diagW > 0.0f)
                                ? (sum / diagW + b[(std::size_t)cell] / (diagW * L.invDx2))
                                : 0.0f;
                            x[(std::size_t)cell] = oneMinusOmega * x[(std::size_t)cell] + omega * xGs;
                        }
                    }
                }
            }
        }
        return;
    }

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

    if (L.denseBoxValid) {
        const int boxNx = L.boxI1 - L.boxI0;
        const int boxNy = L.boxJ1 - L.boxJ0;
        const int boxNz = L.boxK1 - L.boxK0;
        const int sliceStride = boxNx * boxNy;
        const bool openTopDirichlet = L.boxOpenTopDirichlet;
        const float invDx2 = L.invDx2;

        const float* const x = L.x.data();
        const float* const b = L.b.data();
        float* const r = L.r.data();

        for (int lk = 0; lk < boxNz; ++lk) {
            for (int lj = 0; lj < boxNy; ++lj) {
                int cell = boxNx * (lj + boxNy * lk);
                for (int li = 0; li < boxNx; ++li, ++cell) {
                    float sum = 0.0f;
                    float diagW = 0.0f;

                    if (li > 0)          { sum += x[(std::size_t)(cell - 1)];          diagW += 1.0f; }
                    if (li + 1 < boxNx)  { sum += x[(std::size_t)(cell + 1)];          diagW += 1.0f; }
                    if (lj > 0)          { sum += x[(std::size_t)(cell - boxNx)];      diagW += 1.0f; }
                    if (lj + 1 < boxNy)  { sum += x[(std::size_t)(cell + boxNx)];      diagW += 1.0f; }
                    else if (openTopDirichlet) {                                    diagW += 1.0f; }
                    if (lk > 0)          { sum += x[(std::size_t)(cell - sliceStride)]; diagW += 1.0f; }
                    if (lk + 1 < boxNz)  { sum += x[(std::size_t)(cell + sliceStride)]; diagW += 1.0f; }

                    r[(std::size_t)cell] = b[(std::size_t)cell] - (diagW * x[(std::size_t)cell] - sum) * invDx2;
                }
            }
        }
        return;
    }

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

    const float* const r = F.r.data();
    float* const coarseB = C.b.data();
    float* const coarseX = C.x.data();

    if (transfer.denseBoxStructured && F.denseBoxValid && C.denseBoxValid) {
        static const float w1[3] = { 1.0f, 2.0f, 1.0f };

        const int fBoxNx = F.boxI1 - F.boxI0;
        const int fBoxNy = F.boxJ1 - F.boxJ0;
        const int fSliceStride = fBoxNx * fBoxNy;
        const int fi0 = F.boxI0;
        const int fi1 = F.boxI1;
        const int fj0 = F.boxJ0;
        const int fj1 = F.boxJ1;
        const int fk0 = F.boxK0;
        const int fk1 = F.boxK1;

        const int cBoxNx = C.boxI1 - C.boxI0;
        const int cBoxNy = C.boxJ1 - C.boxJ0;
        const int ci0 = C.boxI0;
        const int ci1 = C.boxI1;
        const int cj0 = C.boxJ0;
        const int cj1 = C.boxJ1;
        const int ck0 = C.boxK0;
        const int ck1 = C.boxK1;
        const int cjCount = cj1 - cj0;
        const int rowCount = (ck1 - ck0) * cjCount;

        parallelForPressure3DTransferRows(rowCount, [&](int rowBegin, int rowEnd) {
            for (int row = rowBegin; row < rowEnd; ++row) {
                const int localK = row / cjCount;
                const int localJ = row - localK * cjCount;
                const int K = ck0 + localK;
                const int J = cj0 + localJ;
                const int fkBase = 2 * K;
                const int fjBase = 2 * J;
                const int dkBegin = std::max(-1, fk0 - fkBase);
                const int dkEnd = std::min(1, (fk1 - 1) - fkBase);
                const int djBegin = std::max(-1, fj0 - fjBase);
                const int djEnd = std::min(1, (fj1 - 1) - fjBase);

                int coarseCell = cBoxNx * (localJ + cBoxNy * localK);
                for (int I = ci0; I < ci1; ++I, ++coarseCell) {
                    const int fiBase = 2 * I;
                    const int diBegin = std::max(-1, fi0 - fiBase);
                    const int diEnd = std::min(1, (fi1 - 1) - fiBase);

                    float sum = 0.0f;
                    float wsum = 0.0f;
                    for (int dk = dkBegin; dk <= dkEnd; ++dk) {
                        const float wk = w1[dk + 1];
                        const int fk = fkBase + dk;
                        const int localFk = fk - fk0;
                        const int kBase = localFk * fSliceStride;
                        for (int dj = djBegin; dj <= djEnd; ++dj) {
                            const float wkj = wk * w1[dj + 1];
                            const int fj = fjBase + dj;
                            const int localFj = fj - fj0;
                            int fineCell = (fiBase + diBegin - fi0) + fBoxNx * localFj + kBase;
                            for (int di = diBegin; di <= diEnd; ++di, ++fineCell) {
                                const float wgt = wkj * w1[di + 1];
                                sum += wgt * r[(std::size_t)fineCell];
                                wsum += wgt;
                            }
                        }
                    }

                    coarseB[(std::size_t)coarseCell] = (wsum > 0.0f) ? (sum / wsum) : 0.0f;
                    coarseX[(std::size_t)coarseCell] = 0.0f;
                }
            }
        });
        return;
    }

    std::fill(C.b.begin(), C.b.end(), 0.0f);
    std::fill(C.x.begin(), C.x.end(), 0.0f);

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

    if (transfer.denseBoxStructured && F.denseBoxValid && C.denseBoxValid) {
        const int cBoxNx = C.boxI1 - C.boxI0;
        const int cBoxNy = C.boxJ1 - C.boxJ0;
        const int ci0 = C.boxI0;
        const int ci1 = C.boxI1;
        const int cj0 = C.boxJ0;
        const int cj1 = C.boxJ1;
        const int ck0 = C.boxK0;
        const int ck1 = C.boxK1;

        const int fBoxNx = F.boxI1 - F.boxI0;
        const int fBoxNy = F.boxJ1 - F.boxJ0;
        const int fi0 = F.boxI0;
        const int fi1 = F.boxI1;
        const int fj0 = F.boxJ0;
        const int fj1 = F.boxJ1;
        const int fk0 = F.boxK0;
        const int fk1 = F.boxK1;
        const int fjCount = fj1 - fj0;
        const int rowCount = (fk1 - fk0) * fjCount;

        parallelForPressure3DTransferRows(rowCount, [&](int rowBegin, int rowEnd) {
            for (int row = rowBegin; row < rowEnd; ++row) {
                const int localK = row / fjCount;
                const int localJ = row - localK * fjCount;
                const int fk = fk0 + localK;
                const int fj = fj0 + localJ;

                const int K = fk >> 1;
                const int K1 = std::min(K + 1, C.nz - 1);
                int ckVals[2] = { K, K };
                float wzVals[2] = { ((fk & 1) == 0) ? 1.0f : 0.5f, 0.0f };
                int zCount = 1;
                if ((fk & 1) != 0) {
                    if (K1 != K) {
                        ckVals[1] = K1;
                        wzVals[1] = 0.5f;
                        zCount = 2;
                    } else {
                        wzVals[0] += 0.5f;
                    }
                }

                const int J = fj >> 1;
                const int J1 = std::min(J + 1, C.ny - 1);
                int cjVals[2] = { J, J };
                float wyVals[2] = { ((fj & 1) == 0) ? 1.0f : 0.5f, 0.0f };
                int yCount = 1;
                if ((fj & 1) != 0) {
                    if (J1 != J) {
                        cjVals[1] = J1;
                        wyVals[1] = 0.5f;
                        yCount = 2;
                    } else {
                        wyVals[0] += 0.5f;
                    }
                }

                for (int fi = fi0; fi < fi1; ++fi) {
                    const int I = fi >> 1;
                    const int I1 = std::min(I + 1, C.nx - 1);
                    int ciVals[2] = { I, I };
                    float wxVals[2] = { ((fi & 1) == 0) ? 1.0f : 0.5f, 0.0f };
                    int xCount = 1;
                    if ((fi & 1) != 0) {
                        if (I1 != I) {
                            ciVals[1] = I1;
                            wxVals[1] = 0.5f;
                            xCount = 2;
                        } else {
                            wxVals[0] += 0.5f;
                        }
                    }

                    float correction = 0.0f;
                    for (int zk = 0; zk < zCount; ++zk) {
                        const int ck = ckVals[zk];
                        if (ck < ck0 || ck >= ck1) continue;
                        const float wz = wzVals[zk];
                        for (int yk = 0; yk < yCount; ++yk) {
                            const int cj = cjVals[yk];
                            if (cj < cj0 || cj >= cj1) continue;
                            const float wyz = wz * wyVals[yk];
                            const int coarseRowBase = cBoxNx * ((cj - cj0) + cBoxNy * (ck - ck0));
                            for (int xk = 0; xk < xCount; ++xk) {
                                const int ci = ciVals[xk];
                                if (ci < ci0 || ci >= ci1) continue;
                                correction += (wyz * wxVals[xk]) * coarseX[(std::size_t)(coarseRowBase + (ci - ci0))];
                            }
                        }
                    }
                    const int fineCell = (fi - fi0) + fBoxNx * ((fj - fj0) + fBoxNy * (fk - fk0));
                    fineX[(std::size_t)fineCell] += correction;
                }
            }
        });
        return;
    }

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

    MGLevel& F = mgLevels[0];
    const int N = F.nx * F.ny * F.nz;
    if ((int)p.size() != N) p.assign((std::size_t)N, 0.0f);
    if ((int)rhs.size() != N) {
        m_lastIters = 0;
        return;
    }

    const int fluidCount = (int)m_fluidCells.size();
    if (F.stencils.empty() || fluidCount <= 0) {
        std::fill(p.begin(), p.end(), 0.0f);
        m_lastIters = 0;
        return;
    }

    ensurePCGBuffers();
    packFluidField(p, m_pCompact);
    packFluidField(rhs, m_rhsCompact);

    const float tolRhs = tolPredDiv / std::max(1e-8f, dtForPredDiv);
    const float relTol = std::max(0.0f, mgRelativeTol);

    const float bInf = maxAbsCompact(m_rhsCompact);
    if (!(bInf > 0.0f) || bInf <= tolRhs) {
        removeMeanCompact(m_pCompact);
        unpackFluidField(m_pCompact, p);
        m_lastIters = 0;
        return;
    }

    const float* const rhsData = m_rhsCompact.data();
    float* const pData = m_pCompact.data();
    float* const rData = m_r.data();
    float* const zData = m_z.data();
    float* const dData = m_d.data();
    float* const qData = m_q.data();

    applyACompact(m_pCompact, m_Ap);
    const float* const ApData = m_Ap.data();
    const Pressure3DMaxAbsReduction initResidual = parallelReducePressure3D<Pressure3DMaxAbsReduction>(
        fluidCount,
        kPressure3DMinChunk,
        Pressure3DMaxAbsReduction{},
        [&](int begin, int end) {
            Pressure3DMaxAbsReduction local;
            for (int idxFluid = begin; idxFluid < end; ++idxFluid) {
                const float r = rhsData[(std::size_t)idxFluid] - ApData[(std::size_t)idxFluid];
                rData[(std::size_t)idxFluid] = r;
                const float absR = std::fabs(r);
                if (!std::isfinite(absR)) {
                    local.nonFinite = true;
                    break;
                }
                local.maxValue = std::max(local.maxValue, absR);
            }
            return local;
        },
        combinePressure3DMaxAbs);

    float rInf = initResidual.nonFinite ? std::numeric_limits<float>::infinity() : initResidual.maxValue;
    if (!(rInf > 0.0f) || rInf <= tolRhs || !std::isfinite(rInf)) {
        removeMeanCompact(m_pCompact);
        unpackFluidField(m_pCompact, p);
        m_lastIters = 0;
        return;
    }

    const float rInf0 = std::max(rInf, 1.0e-30f);

    auto applyPrecond = [&](const std::vector<float>& rVec,
                            std::vector<float>& zVec,
                            std::vector<float>* dInit) -> float {
        float* const Fb = F.b.data();
        float* const Fx = F.x.data();
        const float* const rSrc = rVec.data();
        float* const zDst = zVec.data();
        float* const dDst = dInit ? dInit->data() : nullptr;

        parallelForPressure3D(fluidCount, [&](int begin, int end) {
            for (int idxFluid = begin; idxFluid < end; ++idxFluid) {
                Fb[(std::size_t)idxFluid] = rSrc[(std::size_t)idxFluid];
                Fx[(std::size_t)idxFluid] = 0.0f;
            }
        });

        mgVCycle(0);

        float mean = 0.0f;
        if (m_removeMean) {
            const double sum = parallelReducePressure3D<double>(
                fluidCount,
                kPressure3DMinChunk,
                0.0,
                [&](int begin, int end) {
                    double local = 0.0;
                    for (int idxFluid = begin; idxFluid < end; ++idxFluid) {
                        local += (double)Fx[(std::size_t)idxFluid];
                    }
                    return local;
                },
                [](double lhs, double rhs) { return lhs + rhs; });
            mean = (float)(sum / (double)fluidCount);
        }

        const Pressure3DMaxAbsDotReduction finalizeReduce = parallelReducePressure3D<Pressure3DMaxAbsDotReduction>(
            fluidCount,
            kPressure3DMinChunk,
            Pressure3DMaxAbsDotReduction{},
            [&](int begin, int end) {
                Pressure3DMaxAbsDotReduction local;
                for (int idxFluid = begin; idxFluid < end; ++idxFluid) {
                    const float z = Fx[(std::size_t)idxFluid] - mean;
                    zDst[(std::size_t)idxFluid] = z;
                    if (dDst) dDst[(std::size_t)idxFluid] = z;
                    if (!std::isfinite(z)) {
                        local.nonFinite = true;
                        break;
                    }
                    local.dot += (double)rSrc[(std::size_t)idxFluid] * (double)z;
                }
                return local;
            },
            combinePressure3DMaxAbsDot);

        if (finalizeReduce.nonFinite || !std::isfinite(finalizeReduce.dot)) {
            return std::numeric_limits<float>::quiet_NaN();
        }
        return (float)finalizeReduce.dot;
    };

    float deltaNew = applyPrecond(m_r, m_z, &m_d);
    if (!std::isfinite(deltaNew) || deltaNew <= 1.0e-30f) {
        unpackFluidField(m_pCompact, p);
        solvePCG(p, rhs, 120, tolPredDiv, dtForPredDiv);
        return;
    }

    bool fallbackPCG = false;
    int itUsed = 0;
    for (int it = 0; it < std::max(1, maxVCycles); ++it) {
        itUsed = it + 1;

        applyACompact(m_d, m_q);
        const float dq = dotCompact(m_d, m_q);
        if (!std::isfinite(dq) || std::fabs(dq) < 1.0e-30f) {
            fallbackPCG = true;
            break;
        }

        const float alpha = deltaNew / dq;
        const Pressure3DMaxAbsReduction updateReduce = parallelReducePressure3D<Pressure3DMaxAbsReduction>(
            fluidCount,
            kPressure3DMinChunk,
            Pressure3DMaxAbsReduction{},
            [&](int begin, int end) {
                Pressure3DMaxAbsReduction local;
                for (int idxFluid = begin; idxFluid < end; ++idxFluid) {
                    pData[(std::size_t)idxFluid] += alpha * dData[(std::size_t)idxFluid];
                    const float r = rData[(std::size_t)idxFluid] - alpha * qData[(std::size_t)idxFluid];
                    rData[(std::size_t)idxFluid] = r;
                    const float absR = std::fabs(r);
                    if (!std::isfinite(absR)) {
                        local.nonFinite = true;
                        break;
                    }
                    local.maxValue = std::max(local.maxValue, absR);
                }
                return local;
            },
            combinePressure3DMaxAbs);

        rInf = updateReduce.nonFinite ? std::numeric_limits<float>::infinity() : updateReduce.maxValue;
        if (!std::isfinite(rInf)) {
            fallbackPCG = true;
            break;
        }
        if (rInf <= tolRhs) break;
        if (relTol > 0.0f && rInf <= relTol * rInf0) break;

        const float deltaOld = deltaNew;
        deltaNew = applyPrecond(m_r, m_z, nullptr);
        if (!std::isfinite(deltaNew) || deltaNew <= 1.0e-30f) {
            fallbackPCG = true;
            break;
        }

        const float beta = deltaNew / (deltaOld + 1.0e-30f);
        parallelForPressure3D(fluidCount, [&](int begin, int end) {
            for (int idxFluid = begin; idxFluid < end; ++idxFluid) {
                dData[(std::size_t)idxFluid] = zData[(std::size_t)idxFluid] + beta * dData[(std::size_t)idxFluid];
            }
        });
    }

    removeMeanCompact(m_pCompact);
    unpackFluidField(m_pCompact, p);
    m_lastIters = itUsed;

    if (fallbackPCG) {
        solvePCG(p, rhs, 120, tolPredDiv, dtForPredDiv);
    }
}
