#include "mac_smoke3d.h"

#include <chrono>
#include <cmath>

#include "chunk_worker_pool.h"

#if defined(_MSC_VER)
#define SMOKE_RESTRICT __restrict
#else
#define SMOKE_RESTRICT __restrict__
#endif

namespace {

inline int clampi(int v, int lo, int hi) {
    return (v < lo) ? lo : (v > hi ? hi : v);
}

inline float clampf(float v, float lo, float hi) {
    if (!std::isfinite(v)) return lo;
    return (v < lo) ? lo : (v > hi ? hi : v);
}

inline float clamp01(float v) {
    return clampf(v, 0.0f, 1.0f);
}

inline float dissipationStep(float base, float dt) {
    const float b = clamp01(base);
    if (b >= 0.999999f) return 1.0f;
    const float dtRef = 0.02f;
    return std::pow(b, dt / std::max(1e-6f, dtRef));
}

inline ChunkWorkerPool& smoke3DWorkerPool() {
    static ChunkWorkerPool pool;
    return pool;
}

template <typename Fn>
inline void parallelForChunks(int count, int minChunk, Fn&& fn) {
    smoke3DWorkerPool().parallelFor(count, minChunk, std::forward<Fn>(fn));
}

inline int fastFloorToInt(float x) {
    const int i = (int)x;
    return i - ((x < (float)i) ? 1 : 0);
}

inline float sampleCellCenteredGrid(const float* SMOKE_RESTRICT field,
                                    int nx,
                                    int ny,
                                    int nz,
                                    float gx,
                                    float gy,
                                    float gz) {
    const float fx = gx - 0.5f;
    const float fy = gy - 0.5f;
    const float fz = gz - 0.5f;

    const int i0 = clampi(fastFloorToInt(fx), 0, nx - 1);
    const int j0 = clampi(fastFloorToInt(fy), 0, ny - 1);
    const int k0 = clampi(fastFloorToInt(fz), 0, nz - 1);

    const int i1 = std::min(i0 + 1, nx - 1);
    const int j1 = std::min(j0 + 1, ny - 1);
    const int k1 = std::min(k0 + 1, nz - 1);

    const float tx = clampf(fx - (float)i0, 0.0f, 1.0f);
    const float ty = clampf(fy - (float)j0, 0.0f, 1.0f);
    const float tz = clampf(fz - (float)k0, 0.0f, 1.0f);
    const float wx0 = 1.0f - tx;
    const float wx1 = tx;
    const float wy0 = 1.0f - ty;
    const float wy1 = ty;
    const float wz0 = 1.0f - tz;
    const float wz1 = tz;

    const int sliceStride = nx * ny;
    const int row00 = j0 * nx + k0 * sliceStride;
    const int row10 = j1 * nx + k0 * sliceStride;
    const int row01 = j0 * nx + k1 * sliceStride;
    const int row11 = j1 * nx + k1 * sliceStride;

    const float c00 = wx0 * field[(std::size_t)(row00 + i0)] + wx1 * field[(std::size_t)(row00 + i1)];
    const float c10 = wx0 * field[(std::size_t)(row10 + i0)] + wx1 * field[(std::size_t)(row10 + i1)];
    const float c01 = wx0 * field[(std::size_t)(row01 + i0)] + wx1 * field[(std::size_t)(row01 + i1)];
    const float c11 = wx0 * field[(std::size_t)(row11 + i0)] + wx1 * field[(std::size_t)(row11 + i1)];

    const float c0 = wy0 * c00 + wy1 * c10;
    const float c1 = wy0 * c01 + wy1 * c11;
    return wz0 * c0 + wz1 * c1;
}

inline void sampleCellCenteredPairGrid(const float* SMOKE_RESTRICT field0,
                                       const float* SMOKE_RESTRICT field1,
                                       int nx,
                                       int ny,
                                       int nz,
                                       float gx,
                                       float gy,
                                       float gz,
                                       float& out0,
                                       float& out1) {
    const float fx = gx - 0.5f;
    const float fy = gy - 0.5f;
    const float fz = gz - 0.5f;

    const int i0 = clampi(fastFloorToInt(fx), 0, nx - 1);
    const int j0 = clampi(fastFloorToInt(fy), 0, ny - 1);
    const int k0 = clampi(fastFloorToInt(fz), 0, nz - 1);

    const int i1 = std::min(i0 + 1, nx - 1);
    const int j1 = std::min(j0 + 1, ny - 1);
    const int k1 = std::min(k0 + 1, nz - 1);

    const float tx = clampf(fx - (float)i0, 0.0f, 1.0f);
    const float ty = clampf(fy - (float)j0, 0.0f, 1.0f);
    const float tz = clampf(fz - (float)k0, 0.0f, 1.0f);
    const float wx0 = 1.0f - tx;
    const float wx1 = tx;
    const float wy0 = 1.0f - ty;
    const float wy1 = ty;
    const float wz0 = 1.0f - tz;
    const float wz1 = tz;

    const int sliceStride = nx * ny;
    const int row00 = j0 * nx + k0 * sliceStride;
    const int row10 = j1 * nx + k0 * sliceStride;
    const int row01 = j0 * nx + k1 * sliceStride;
    const int row11 = j1 * nx + k1 * sliceStride;

    const float f0_00 = wx0 * field0[(std::size_t)(row00 + i0)] + wx1 * field0[(std::size_t)(row00 + i1)];
    const float f0_10 = wx0 * field0[(std::size_t)(row10 + i0)] + wx1 * field0[(std::size_t)(row10 + i1)];
    const float f0_01 = wx0 * field0[(std::size_t)(row01 + i0)] + wx1 * field0[(std::size_t)(row01 + i1)];
    const float f0_11 = wx0 * field0[(std::size_t)(row11 + i0)] + wx1 * field0[(std::size_t)(row11 + i1)];

    const float f1_00 = wx0 * field1[(std::size_t)(row00 + i0)] + wx1 * field1[(std::size_t)(row00 + i1)];
    const float f1_10 = wx0 * field1[(std::size_t)(row10 + i0)] + wx1 * field1[(std::size_t)(row10 + i1)];
    const float f1_01 = wx0 * field1[(std::size_t)(row01 + i0)] + wx1 * field1[(std::size_t)(row01 + i1)];
    const float f1_11 = wx0 * field1[(std::size_t)(row11 + i0)] + wx1 * field1[(std::size_t)(row11 + i1)];

    const float f0_y0 = wy0 * f0_00 + wy1 * f0_10;
    const float f0_y1 = wy0 * f0_01 + wy1 * f0_11;
    const float f1_y0 = wy0 * f1_00 + wy1 * f1_10;
    const float f1_y1 = wy0 * f1_01 + wy1 * f1_11;

    out0 = wz0 * f0_y0 + wz1 * f0_y1;
    out1 = wz0 * f1_y0 + wz1 * f1_y1;
}

inline float sampleUGrid(const float* SMOKE_RESTRICT field,
                         int nx,
                         int ny,
                         int nz,
                         float gx,
                         float gy,
                         float gz) {
    const float fx = gx;
    const float fy = gy - 0.5f;
    const float fz = gz - 0.5f;

    const int i0 = clampi(fastFloorToInt(fx), 0, nx);
    const int j0 = clampi(fastFloorToInt(fy), 0, ny - 1);
    const int k0 = clampi(fastFloorToInt(fz), 0, nz - 1);

    const int i1 = std::min(i0 + 1, nx);
    const int j1 = std::min(j0 + 1, ny - 1);
    const int k1 = std::min(k0 + 1, nz - 1);

    const float tx = clampf(fx - (float)i0, 0.0f, 1.0f);
    const float ty = clampf(fy - (float)j0, 0.0f, 1.0f);
    const float tz = clampf(fz - (float)k0, 0.0f, 1.0f);
    const float wx0 = 1.0f - tx;
    const float wx1 = tx;
    const float wy0 = 1.0f - ty;
    const float wy1 = ty;
    const float wz0 = 1.0f - tz;
    const float wz1 = tz;

    const int strideX = nx + 1;
    const int sliceStride = strideX * ny;
    const int row00 = j0 * strideX + k0 * sliceStride;
    const int row10 = j1 * strideX + k0 * sliceStride;
    const int row01 = j0 * strideX + k1 * sliceStride;
    const int row11 = j1 * strideX + k1 * sliceStride;

    const float c00 = wx0 * field[(std::size_t)(row00 + i0)] + wx1 * field[(std::size_t)(row00 + i1)];
    const float c10 = wx0 * field[(std::size_t)(row10 + i0)] + wx1 * field[(std::size_t)(row10 + i1)];
    const float c01 = wx0 * field[(std::size_t)(row01 + i0)] + wx1 * field[(std::size_t)(row01 + i1)];
    const float c11 = wx0 * field[(std::size_t)(row11 + i0)] + wx1 * field[(std::size_t)(row11 + i1)];

    const float c0 = wy0 * c00 + wy1 * c10;
    const float c1 = wy0 * c01 + wy1 * c11;
    return wz0 * c0 + wz1 * c1;
}

inline float sampleVGrid(const float* SMOKE_RESTRICT field,
                         int nx,
                         int ny,
                         int nz,
                         float gx,
                         float gy,
                         float gz) {
    const float fx = gx - 0.5f;
    const float fy = gy;
    const float fz = gz - 0.5f;

    const int i0 = clampi(fastFloorToInt(fx), 0, nx - 1);
    const int j0 = clampi(fastFloorToInt(fy), 0, ny);
    const int k0 = clampi(fastFloorToInt(fz), 0, nz - 1);

    const int i1 = std::min(i0 + 1, nx - 1);
    const int j1 = std::min(j0 + 1, ny);
    const int k1 = std::min(k0 + 1, nz - 1);

    const float tx = clampf(fx - (float)i0, 0.0f, 1.0f);
    const float ty = clampf(fy - (float)j0, 0.0f, 1.0f);
    const float tz = clampf(fz - (float)k0, 0.0f, 1.0f);
    const float wx0 = 1.0f - tx;
    const float wx1 = tx;
    const float wy0 = 1.0f - ty;
    const float wy1 = ty;
    const float wz0 = 1.0f - tz;
    const float wz1 = tz;

    const int strideX = nx;
    const int sliceStride = strideX * (ny + 1);
    const int row00 = j0 * strideX + k0 * sliceStride;
    const int row10 = j1 * strideX + k0 * sliceStride;
    const int row01 = j0 * strideX + k1 * sliceStride;
    const int row11 = j1 * strideX + k1 * sliceStride;

    const float c00 = wx0 * field[(std::size_t)(row00 + i0)] + wx1 * field[(std::size_t)(row00 + i1)];
    const float c10 = wx0 * field[(std::size_t)(row10 + i0)] + wx1 * field[(std::size_t)(row10 + i1)];
    const float c01 = wx0 * field[(std::size_t)(row01 + i0)] + wx1 * field[(std::size_t)(row01 + i1)];
    const float c11 = wx0 * field[(std::size_t)(row11 + i0)] + wx1 * field[(std::size_t)(row11 + i1)];

    const float c0 = wy0 * c00 + wy1 * c10;
    const float c1 = wy0 * c01 + wy1 * c11;
    return wz0 * c0 + wz1 * c1;
}

inline float sampleWGrid(const float* SMOKE_RESTRICT field,
                         int nx,
                         int ny,
                         int nz,
                         float gx,
                         float gy,
                         float gz) {
    const float fx = gx - 0.5f;
    const float fy = gy - 0.5f;
    const float fz = gz;

    const int i0 = clampi(fastFloorToInt(fx), 0, nx - 1);
    const int j0 = clampi(fastFloorToInt(fy), 0, ny - 1);
    const int k0 = clampi(fastFloorToInt(fz), 0, nz);

    const int i1 = std::min(i0 + 1, nx - 1);
    const int j1 = std::min(j0 + 1, ny - 1);
    const int k1 = std::min(k0 + 1, nz);

    const float tx = clampf(fx - (float)i0, 0.0f, 1.0f);
    const float ty = clampf(fy - (float)j0, 0.0f, 1.0f);
    const float tz = clampf(fz - (float)k0, 0.0f, 1.0f);
    const float wx0 = 1.0f - tx;
    const float wx1 = tx;
    const float wy0 = 1.0f - ty;
    const float wy1 = ty;
    const float wz0 = 1.0f - tz;
    const float wz1 = tz;

    const int strideX = nx;
    const int sliceStride = strideX * ny;
    const int row00 = j0 * strideX + k0 * sliceStride;
    const int row10 = j1 * strideX + k0 * sliceStride;
    const int row01 = j0 * strideX + k1 * sliceStride;
    const int row11 = j1 * strideX + k1 * sliceStride;

    const float c00 = wx0 * field[(std::size_t)(row00 + i0)] + wx1 * field[(std::size_t)(row00 + i1)];
    const float c10 = wx0 * field[(std::size_t)(row10 + i0)] + wx1 * field[(std::size_t)(row10 + i1)];
    const float c01 = wx0 * field[(std::size_t)(row01 + i0)] + wx1 * field[(std::size_t)(row01 + i1)];
    const float c11 = wx0 * field[(std::size_t)(row11 + i0)] + wx1 * field[(std::size_t)(row11 + i1)];

    const float c0 = wy0 * c00 + wy1 * c10;
    const float c1 = wy0 * c01 + wy1 * c11;
    return wz0 * c0 + wz1 * c1;
}

inline void velAtGrid(float gx,
                      float gy,
                      float gz,
                      const float* SMOKE_RESTRICT fu,
                      const float* SMOKE_RESTRICT fv,
                      const float* SMOKE_RESTRICT fw,
                      int nx,
                      int ny,
                      int nz,
                      float& outU,
                      float& outV,
                      float& outW) {
    outU = sampleUGrid(fu, nx, ny, nz, gx, gy, gz);
    outV = sampleVGrid(fv, nx, ny, nz, gx, gy, gz);
    outW = sampleWGrid(fw, nx, ny, nz, gx, gy, gz);
}

} // namespace

MACSmoke3D::MACSmoke3D(int NX, int NY, int NZ, float DX, float DT)
    : nx(NX), ny(NY), nz(NZ), dx(DX), dt(DT) {
    reset();
}

void MACSmoke3D::reset() {
    const int cellCount = std::max(1, nx * ny * nz);
    const int uCount = std::max(1, (nx + 1) * ny * nz);
    const int vCount = std::max(1, nx * (ny + 1) * nz);
    const int wCount = std::max(1, nx * ny * (nz + 1));

    u.assign((std::size_t)uCount, 0.0f);
    v.assign((std::size_t)vCount, 0.0f);
    w.assign((std::size_t)wCount, 0.0f);

    u0.assign((std::size_t)uCount, 0.0f);
    v0.assign((std::size_t)vCount, 0.0f);
    w0.assign((std::size_t)wCount, 0.0f);

    uTmp.assign((std::size_t)uCount, 0.0f);
    vTmp.assign((std::size_t)vCount, 0.0f);
    wTmp.assign((std::size_t)wCount, 0.0f);

    pressure.assign((std::size_t)cellCount, 0.0f);
    pressureTmp.assign((std::size_t)cellCount, 0.0f);
    rhs.assign((std::size_t)cellCount, 0.0f);

    smoke.assign((std::size_t)cellCount, 0.0f);
    smoke0.assign((std::size_t)cellCount, 0.0f);
    temp.assign((std::size_t)cellCount, 0.0f);
    temp0.assign((std::size_t)cellCount, 0.0f);
    cellTmp.assign((std::size_t)cellCount, 0.0f);

    divergence.assign((std::size_t)cellCount, 0.0f);
    speed.assign((std::size_t)cellCount, 0.0f);

    solid.assign((std::size_t)cellCount, (uint8_t)0);
    solidUser.assign((std::size_t)cellCount, (uint8_t)0);
    fluidMask.assign((std::size_t)cellCount, (uint8_t)0);
    fluidCellCount = 0;
    topologyDirty = true;
    pressureOperatorDirty = true;
    diffusionStencilDirty = true;
    uDiffusionStencil.clear();
    vDiffusionStencil.clear();
    wDiffusionStencil.clear();
    uDiffusionScratch.r.clear();
    uDiffusionScratch.z.clear();
    uDiffusionScratch.p.clear();
    uDiffusionScratch.q.clear();
    vDiffusionScratch.r.clear();
    vDiffusionScratch.z.clear();
    vDiffusionScratch.p.clear();
    vDiffusionScratch.q.clear();
    wDiffusionScratch.r.clear();
    wDiffusionScratch.z.clear();
    wDiffusionScratch.p.clear();
    wDiffusionScratch.q.clear();

    rebuildBorderSolids();
    applyBoundary();
    derivedFieldsDirty = true;
    updateStats(0.0f);
}

void MACSmoke3D::reset(int NX, int NY, int NZ, float DX, float DT) {
    nx = NX;
    ny = NY;
    nz = NZ;
    dx = DX;
    dt = DT;
    reset();
}

void MACSmoke3D::setParams(const Params& newParams) {
    params = newParams;
    rebuildBorderSolids();
    applyBoundary();
    derivedFieldsDirty = true;
    updateStats(0.0f);
}

void MACSmoke3D::rebuildBorderSolids() {
    const int cellCount = nx * ny * nz;
    if ((int)solidUser.size() != cellCount) {
        solidUser.assign((std::size_t)cellCount, (uint8_t)0);
    }

    solid = solidUser;

    const int maxBt = std::max(1, (std::min({nx, ny, nz}) / 2) - 1);
    const int bt = clampi(params.borderThickness, 1, maxBt);

    auto setSolid = [&](int i, int j, int k) {
        if (i < 0 || i >= nx || j < 0 || j >= ny || k < 0 || k >= nz) return;
        solid[(std::size_t)idxCell(i, j, k)] = (uint8_t)1;
        smoke[(std::size_t)idxCell(i, j, k)] = 0.0f;
        temp[(std::size_t)idxCell(i, j, k)] = 0.0f;
    };

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int t = 0; t < bt; ++t) {
                setSolid(t, j, k);
                setSolid(nx - 1 - t, j, k);
            }
        }
    }

    for (int k = 0; k < nz; ++k) {
        for (int i = 0; i < nx; ++i) {
            for (int t = 0; t < bt; ++t) {
                setSolid(i, t, k);
                if (!params.openTop) setSolid(i, ny - 1 - t, k);
            }
        }
    }

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            for (int t = 0; t < bt; ++t) {
                setSolid(i, j, t);
                setSolid(i, j, nz - 1 - t);
            }
        }
    }

    fluidCellCount = 0;
    if ((int)fluidMask.size() != cellCount) {
        fluidMask.assign((std::size_t)cellCount, (uint8_t)0);
    }
    for (int id = 0; id < cellCount; ++id) {
        fluidMask[(std::size_t)id] = solid[(std::size_t)id] ? (uint8_t)0 : (uint8_t)1;
        if (!solid[(std::size_t)id]) ++fluidCellCount;
    }

    rebuildAdvectionWorklists();

    topologyDirty = false;
    pressureOperatorDirty = true;
    diffusionStencilDirty = true;
}

void MACSmoke3D::rebuildAdvectionWorklists() {
    activeCellsByK.clear();
    activeUFacesByK.clear();
    activeVFacesByK.clear();
    activeWFacesByK.clear();

    const auto isFixedU = [&](int i, int j, int k) {
        if (i == 0 || i == nx) return true;
        return isSolidCell(i - 1, j, k) || isSolidCell(i, j, k);
    };

    const auto isFixedV = [&](int i, int j, int k) {
        if (j == 0) return true;
        if (j == ny) return !params.openTop;
        return isSolidCell(i, j - 1, k) || isSolidCell(i, j, k);
    };

    const auto isFixedW = [&](int i, int j, int k) {
        if (k == 0 || k == nz) return true;
        return isSolidCell(i, j, k - 1) || isSolidCell(i, j, k);
    };

    activeCellsByK.offsets.assign((std::size_t)nz + 1u, 0);
    activeCellsByK.entries.reserve((std::size_t)std::max(0, fluidCellCount));
    for (int k = 0; k < nz; ++k) {
        activeCellsByK.offsets[(std::size_t)k] = (int)activeCellsByK.entries.size();
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                if (!solid[(std::size_t)idxCell(i, j, k)]) {
                    activeCellsByK.entries.push_back({i, j});
                }
            }
        }
    }
    activeCellsByK.offsets[(std::size_t)nz] = (int)activeCellsByK.entries.size();

    activeUFacesByK.offsets.assign((std::size_t)nz + 1u, 0);
    activeUFacesByK.entries.reserve((std::size_t)std::max(0, (nx - 1) * ny * nz));
    for (int k = 0; k < nz; ++k) {
        activeUFacesByK.offsets[(std::size_t)k] = (int)activeUFacesByK.entries.size();
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i <= nx; ++i) {
                if (!isFixedU(i, j, k)) {
                    activeUFacesByK.entries.push_back({i, j});
                }
            }
        }
    }
    activeUFacesByK.offsets[(std::size_t)nz] = (int)activeUFacesByK.entries.size();

    activeVFacesByK.offsets.assign((std::size_t)nz + 1u, 0);
    activeVFacesByK.entries.reserve((std::size_t)std::max(0, nx * (ny - 1) * nz));
    for (int k = 0; k < nz; ++k) {
        activeVFacesByK.offsets[(std::size_t)k] = (int)activeVFacesByK.entries.size();
        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                if (!isFixedV(i, j, k)) {
                    activeVFacesByK.entries.push_back({i, j});
                }
            }
        }
    }
    activeVFacesByK.offsets[(std::size_t)nz] = (int)activeVFacesByK.entries.size();

    activeWFacesByK.offsets.assign((std::size_t)nz + 2u, 0);
    activeWFacesByK.entries.reserve((std::size_t)std::max(0, nx * ny * (nz - 1)));
    for (int k = 0; k <= nz; ++k) {
        activeWFacesByK.offsets[(std::size_t)k] = (int)activeWFacesByK.entries.size();
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                if (!isFixedW(i, j, k)) {
                    activeWFacesByK.entries.push_back({i, j});
                }
            }
        }
    }
    activeWFacesByK.offsets[(std::size_t)nz + 1u] = (int)activeWFacesByK.entries.size();
}

void MACSmoke3D::rebuildDiffusionStencils() {
    uDiffusionStencil.clear();
    vDiffusionStencil.clear();
    wDiffusionStencil.clear();

    auto appendStencil = [](DiffusionStencilSet& set,
                            int face,
                            int xm,
                            int xp,
                            int ym,
                            int yp,
                            int zm,
                            int zp,
                            uint8_t neighborCount) {
        set.face.push_back(face);
        set.xm.push_back(xm);
        set.xp.push_back(xp);
        set.ym.push_back(ym);
        set.yp.push_back(yp);
        set.zm.push_back(zm);
        set.zp.push_back(zp);
        set.neighborCount.push_back(neighborCount);
    };

    auto isFixedU = [&](int i, int j, int k) {
        if (i == 0 || i == nx) return true;
        return isSolidCell(i - 1, j, k) || isSolidCell(i, j, k);
    };

    auto isFixedV = [&](int i, int j, int k) {
        if (j == 0) return true;
        if (j == ny) return !params.openTop;
        return isSolidCell(i, j - 1, k) || isSolidCell(i, j, k);
    };

    auto isFixedW = [&](int i, int j, int k) {
        if (k == 0 || k == nz) return true;
        return isSolidCell(i, j, k - 1) || isSolidCell(i, j, k);
    };

    const std::size_t uReserve = ((std::size_t)(nx + 1) * (std::size_t)ny * (std::size_t)nz) / 2u + 1u;
    uDiffusionStencil.face.reserve(uReserve);
    uDiffusionStencil.xm.reserve(uReserve);
    uDiffusionStencil.xp.reserve(uReserve);
    uDiffusionStencil.ym.reserve(uReserve);
    uDiffusionStencil.yp.reserve(uReserve);
    uDiffusionStencil.zm.reserve(uReserve);
    uDiffusionStencil.zp.reserve(uReserve);
    uDiffusionStencil.neighborCount.reserve(uReserve);
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i <= nx; ++i) {
                if (isFixedU(i, j, k)) continue;
                uint8_t count = 0;
                int xm = -1, xp = -1, ym = -1, yp = -1, zm = -1, zp = -1;
                if (i > 0)      { ++count; if (!isFixedU(i - 1, j, k)) xm = idxU(i - 1, j, k); }
                if (i < nx)     { ++count; if (!isFixedU(i + 1, j, k)) xp = idxU(i + 1, j, k); }
                if (j > 0)      { ++count; if (!isFixedU(i, j - 1, k)) ym = idxU(i, j - 1, k); }
                if (j + 1 < ny) { ++count; if (!isFixedU(i, j + 1, k)) yp = idxU(i, j + 1, k); }
                if (k > 0)      { ++count; if (!isFixedU(i, j, k - 1)) zm = idxU(i, j, k - 1); }
                if (k + 1 < nz) { ++count; if (!isFixedU(i, j, k + 1)) zp = idxU(i, j, k + 1); }
                appendStencil(uDiffusionStencil, idxU(i, j, k), xm, xp, ym, yp, zm, zp, count);
            }
        }
    }

    const std::size_t vReserve = ((std::size_t)nx * (std::size_t)(ny + 1) * (std::size_t)nz) / 2u + 1u;
    vDiffusionStencil.face.reserve(vReserve);
    vDiffusionStencil.xm.reserve(vReserve);
    vDiffusionStencil.xp.reserve(vReserve);
    vDiffusionStencil.ym.reserve(vReserve);
    vDiffusionStencil.yp.reserve(vReserve);
    vDiffusionStencil.zm.reserve(vReserve);
    vDiffusionStencil.zp.reserve(vReserve);
    vDiffusionStencil.neighborCount.reserve(vReserve);
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                if (isFixedV(i, j, k)) continue;
                uint8_t count = 0;
                int xm = -1, xp = -1, ym = -1, yp = -1, zm = -1, zp = -1;
                if (i > 0)      { ++count; if (!isFixedV(i - 1, j, k)) xm = idxV(i - 1, j, k); }
                if (i + 1 < nx) { ++count; if (!isFixedV(i + 1, j, k)) xp = idxV(i + 1, j, k); }
                if (j > 0)      { ++count; if (!isFixedV(i, j - 1, k)) ym = idxV(i, j - 1, k); }
                if (j < ny)     { ++count; if (!isFixedV(i, j + 1, k)) yp = idxV(i, j + 1, k); }
                if (k > 0)      { ++count; if (!isFixedV(i, j, k - 1)) zm = idxV(i, j, k - 1); }
                if (k + 1 < nz) { ++count; if (!isFixedV(i, j, k + 1)) zp = idxV(i, j, k + 1); }
                appendStencil(vDiffusionStencil, idxV(i, j, k), xm, xp, ym, yp, zm, zp, count);
            }
        }
    }

    const std::size_t wReserve = ((std::size_t)nx * (std::size_t)ny * (std::size_t)(nz + 1)) / 2u + 1u;
    wDiffusionStencil.face.reserve(wReserve);
    wDiffusionStencil.xm.reserve(wReserve);
    wDiffusionStencil.xp.reserve(wReserve);
    wDiffusionStencil.ym.reserve(wReserve);
    wDiffusionStencil.yp.reserve(wReserve);
    wDiffusionStencil.zm.reserve(wReserve);
    wDiffusionStencil.zp.reserve(wReserve);
    wDiffusionStencil.neighborCount.reserve(wReserve);
    for (int k = 0; k <= nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                if (isFixedW(i, j, k)) continue;
                uint8_t count = 0;
                int xm = -1, xp = -1, ym = -1, yp = -1, zm = -1, zp = -1;
                if (i > 0)      { ++count; if (!isFixedW(i - 1, j, k)) xm = idxW(i - 1, j, k); }
                if (i + 1 < nx) { ++count; if (!isFixedW(i + 1, j, k)) xp = idxW(i + 1, j, k); }
                if (j > 0)      { ++count; if (!isFixedW(i, j - 1, k)) ym = idxW(i, j - 1, k); }
                if (j + 1 < ny) { ++count; if (!isFixedW(i, j + 1, k)) yp = idxW(i, j + 1, k); }
                if (k > 0)      { ++count; if (!isFixedW(i, j, k - 1)) zm = idxW(i, j, k - 1); }
                if (k < nz)     { ++count; if (!isFixedW(i, j, k + 1)) zp = idxW(i, j, k + 1); }
                appendStencil(wDiffusionStencil, idxW(i, j, k), xm, xp, ym, yp, zm, zp, count);
            }
        }
    }

    diffusionStencilDirty = false;
}

void MACSmoke3D::applyBoundary() {
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            u[(std::size_t)idxU(0, j, k)] = 0.0f;
            u[(std::size_t)idxU(nx, j, k)] = 0.0f;
        }
    }

    for (int k = 0; k < nz; ++k) {
        for (int i = 0; i < nx; ++i) {
            v[(std::size_t)idxV(i, 0, k)] = 0.0f;
            if (!params.openTop) {
                v[(std::size_t)idxV(i, ny, k)] = 0.0f;
            }
        }
    }

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            w[(std::size_t)idxW(i, j, 0)] = 0.0f;
            w[(std::size_t)idxW(i, j, nz)] = 0.0f;
        }
    }

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i <= nx; ++i) {
                const bool leftSolid = (i - 1 >= 0) ? isSolidCell(i - 1, j, k) : true;
                const bool rightSolid = (i < nx) ? isSolidCell(i, j, k) : true;
                if (leftSolid || rightSolid) {
                    u[(std::size_t)idxU(i, j, k)] = 0.0f;
                }
            }
        }
    }

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                if (j == 0) continue;
                if (j == ny) {
                    if (params.openTop && isSolidCell(i, ny - 1, k)) {
                        v[(std::size_t)idxV(i, j, k)] = 0.0f;
                    }
                    continue;
                }

                const bool botSolid = isSolidCell(i, j - 1, k);
                const bool topSolid = isSolidCell(i, j, k);
                if (botSolid || topSolid) {
                    v[(std::size_t)idxV(i, j, k)] = 0.0f;
                }
            }
        }
    }

    for (int k = 0; k <= nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const bool backSolid = (k - 1 >= 0) ? isSolidCell(i, j, k - 1) : true;
                const bool frontSolid = (k < nz) ? isSolidCell(i, j, k) : true;
                if (backSolid || frontSolid) {
                    w[(std::size_t)idxW(i, j, k)] = 0.0f;
                }
            }
        }
    }

    for (std::size_t id = 0; id < solid.size(); ++id) {
        if (!solid[id]) continue;
        smoke[id] = 0.0f;
        temp[id] = 0.0f;
        pressure[id] = 0.0f;
        divergence[id] = 0.0f;
        speed[id] = 0.0f;
    }
}

float MACSmoke3D::sampleCellCentered(const std::vector<float>& field, float x, float y, float z) const {
    if (field.empty() || nx <= 0 || ny <= 0 || nz <= 0) return 0.0f;

    const float invDx = 1.0f / dx;
    const float fx = x * invDx - 0.5f;
    const float fy = y * invDx - 0.5f;
    const float fz = z * invDx - 0.5f;

    const int i0 = clampi((int)std::floor(fx), 0, nx - 1);
    const int j0 = clampi((int)std::floor(fy), 0, ny - 1);
    const int k0 = clampi((int)std::floor(fz), 0, nz - 1);

    const int i1 = std::min(i0 + 1, nx - 1);
    const int j1 = std::min(j0 + 1, ny - 1);
    const int k1 = std::min(k0 + 1, nz - 1);

    const float tx = clampf(fx - (float)i0, 0.0f, 1.0f);
    const float ty = clampf(fy - (float)j0, 0.0f, 1.0f);
    const float tz = clampf(fz - (float)k0, 0.0f, 1.0f);
    const float wx0 = 1.0f - tx;
    const float wx1 = tx;
    const float wy0 = 1.0f - ty;
    const float wy1 = ty;
    const float wz0 = 1.0f - tz;
    const float wz1 = tz;

    const int sliceStride = nx * ny;
    const int row00 = j0 * nx + k0 * sliceStride;
    const int row10 = j1 * nx + k0 * sliceStride;
    const int row01 = j0 * nx + k1 * sliceStride;
    const int row11 = j1 * nx + k1 * sliceStride;

    const float c00 = wx0 * field[(std::size_t)(row00 + i0)] + wx1 * field[(std::size_t)(row00 + i1)];
    const float c10 = wx0 * field[(std::size_t)(row10 + i0)] + wx1 * field[(std::size_t)(row10 + i1)];
    const float c01 = wx0 * field[(std::size_t)(row01 + i0)] + wx1 * field[(std::size_t)(row01 + i1)];
    const float c11 = wx0 * field[(std::size_t)(row11 + i0)] + wx1 * field[(std::size_t)(row11 + i1)];

    const float c0 = wy0 * c00 + wy1 * c10;
    const float c1 = wy0 * c01 + wy1 * c11;
    return wz0 * c0 + wz1 * c1;
}

float MACSmoke3D::sampleCellCenteredOpenTop(const std::vector<float>& field, float x, float y, float z, float outsideValue) const {
    if (field.empty() || nx <= 0 || ny <= 0 || nz <= 0) return 0.0f;

    const float domainX = nx * dx;
    const float domainY = ny * dx;
    const float domainZ = nz * dx;

    x = clampf(x, 0.0f, domainX);
    z = clampf(z, 0.0f, domainZ);
    if (y < 0.0f) y = 0.0f;
    if (y > domainY) {
        if (params.openTop) return outsideValue;
        y = domainY;
    }

    return sampleCellCentered(field, x, y, z);
}

float MACSmoke3D::sampleU(const std::vector<float>& field, float x, float y, float z) const {
    if (field.empty()) return 0.0f;

    const float invDx = 1.0f / dx;
    const float fx = x * invDx;
    const float fy = y * invDx - 0.5f;
    const float fz = z * invDx - 0.5f;

    const int i0 = clampi((int)std::floor(fx), 0, nx);
    const int j0 = clampi((int)std::floor(fy), 0, ny - 1);
    const int k0 = clampi((int)std::floor(fz), 0, nz - 1);

    const int i1 = std::min(i0 + 1, nx);
    const int j1 = std::min(j0 + 1, ny - 1);
    const int k1 = std::min(k0 + 1, nz - 1);

    const float tx = clampf(fx - (float)i0, 0.0f, 1.0f);
    const float ty = clampf(fy - (float)j0, 0.0f, 1.0f);
    const float tz = clampf(fz - (float)k0, 0.0f, 1.0f);
    const float wx0 = 1.0f - tx;
    const float wx1 = tx;
    const float wy0 = 1.0f - ty;
    const float wy1 = ty;
    const float wz0 = 1.0f - tz;
    const float wz1 = tz;

    const int strideX = nx + 1;
    const int sliceStride = strideX * ny;
    const int row00 = j0 * strideX + k0 * sliceStride;
    const int row10 = j1 * strideX + k0 * sliceStride;
    const int row01 = j0 * strideX + k1 * sliceStride;
    const int row11 = j1 * strideX + k1 * sliceStride;

    const float c00 = wx0 * field[(std::size_t)(row00 + i0)] + wx1 * field[(std::size_t)(row00 + i1)];
    const float c10 = wx0 * field[(std::size_t)(row10 + i0)] + wx1 * field[(std::size_t)(row10 + i1)];
    const float c01 = wx0 * field[(std::size_t)(row01 + i0)] + wx1 * field[(std::size_t)(row01 + i1)];
    const float c11 = wx0 * field[(std::size_t)(row11 + i0)] + wx1 * field[(std::size_t)(row11 + i1)];

    const float c0 = wy0 * c00 + wy1 * c10;
    const float c1 = wy0 * c01 + wy1 * c11;
    return wz0 * c0 + wz1 * c1;
}

float MACSmoke3D::sampleV(const std::vector<float>& field, float x, float y, float z) const {
    if (field.empty()) return 0.0f;

    const float invDx = 1.0f / dx;
    const float fx = x * invDx - 0.5f;
    const float fy = y * invDx;
    const float fz = z * invDx - 0.5f;

    const int i0 = clampi((int)std::floor(fx), 0, nx - 1);
    const int j0 = clampi((int)std::floor(fy), 0, ny);
    const int k0 = clampi((int)std::floor(fz), 0, nz - 1);

    const int i1 = std::min(i0 + 1, nx - 1);
    const int j1 = std::min(j0 + 1, ny);
    const int k1 = std::min(k0 + 1, nz - 1);

    const float tx = clampf(fx - (float)i0, 0.0f, 1.0f);
    const float ty = clampf(fy - (float)j0, 0.0f, 1.0f);
    const float tz = clampf(fz - (float)k0, 0.0f, 1.0f);
    const float wx0 = 1.0f - tx;
    const float wx1 = tx;
    const float wy0 = 1.0f - ty;
    const float wy1 = ty;
    const float wz0 = 1.0f - tz;
    const float wz1 = tz;

    const int strideX = nx;
    const int sliceStride = strideX * (ny + 1);
    const int row00 = j0 * strideX + k0 * sliceStride;
    const int row10 = j1 * strideX + k0 * sliceStride;
    const int row01 = j0 * strideX + k1 * sliceStride;
    const int row11 = j1 * strideX + k1 * sliceStride;

    const float c00 = wx0 * field[(std::size_t)(row00 + i0)] + wx1 * field[(std::size_t)(row00 + i1)];
    const float c10 = wx0 * field[(std::size_t)(row10 + i0)] + wx1 * field[(std::size_t)(row10 + i1)];
    const float c01 = wx0 * field[(std::size_t)(row01 + i0)] + wx1 * field[(std::size_t)(row01 + i1)];
    const float c11 = wx0 * field[(std::size_t)(row11 + i0)] + wx1 * field[(std::size_t)(row11 + i1)];

    const float c0 = wy0 * c00 + wy1 * c10;
    const float c1 = wy0 * c01 + wy1 * c11;
    return wz0 * c0 + wz1 * c1;
}

float MACSmoke3D::sampleW(const std::vector<float>& field, float x, float y, float z) const {
    if (field.empty()) return 0.0f;

    const float invDx = 1.0f / dx;
    const float fx = x * invDx - 0.5f;
    const float fy = y * invDx - 0.5f;
    const float fz = z * invDx;

    const int i0 = clampi((int)std::floor(fx), 0, nx - 1);
    const int j0 = clampi((int)std::floor(fy), 0, ny - 1);
    const int k0 = clampi((int)std::floor(fz), 0, nz);

    const int i1 = std::min(i0 + 1, nx - 1);
    const int j1 = std::min(j0 + 1, ny - 1);
    const int k1 = std::min(k0 + 1, nz);

    const float tx = clampf(fx - (float)i0, 0.0f, 1.0f);
    const float ty = clampf(fy - (float)j0, 0.0f, 1.0f);
    const float tz = clampf(fz - (float)k0, 0.0f, 1.0f);
    const float wx0 = 1.0f - tx;
    const float wx1 = tx;
    const float wy0 = 1.0f - ty;
    const float wy1 = ty;
    const float wz0 = 1.0f - tz;
    const float wz1 = tz;

    const int strideX = nx;
    const int sliceStride = strideX * ny;
    const int row00 = j0 * strideX + k0 * sliceStride;
    const int row10 = j1 * strideX + k0 * sliceStride;
    const int row01 = j0 * strideX + k1 * sliceStride;
    const int row11 = j1 * strideX + k1 * sliceStride;

    const float c00 = wx0 * field[(std::size_t)(row00 + i0)] + wx1 * field[(std::size_t)(row00 + i1)];
    const float c10 = wx0 * field[(std::size_t)(row10 + i0)] + wx1 * field[(std::size_t)(row10 + i1)];
    const float c01 = wx0 * field[(std::size_t)(row01 + i0)] + wx1 * field[(std::size_t)(row01 + i1)];
    const float c11 = wx0 * field[(std::size_t)(row11 + i0)] + wx1 * field[(std::size_t)(row11 + i1)];

    const float c0 = wy0 * c00 + wy1 * c10;
    const float c1 = wy0 * c01 + wy1 * c11;
    return wz0 * c0 + wz1 * c1;
}

void MACSmoke3D::velAt(float x, float y, float z,
                       const std::vector<float>& fu,
                       const std::vector<float>& fv,
                       const std::vector<float>& fw,
                       float& outU, float& outV, float& outW) const {
    outU = sampleU(fu, x, y, z);
    outV = sampleV(fv, x, y, z);
    outW = sampleW(fw, x, y, z);
}

void MACSmoke3D::advectVelocity() {
    u0.swap(u);
    v0.swap(v);
    w0.swap(w);

    const float* SMOKE_RESTRICT uSrc = u0.data();
    const float* SMOKE_RESTRICT vSrc = v0.data();
    const float* SMOKE_RESTRICT wSrc = w0.data();
    float* SMOKE_RESTRICT uDst = u.data();
    float* SMOKE_RESTRICT vDst = v.data();
    float* SMOKE_RESTRICT wDst = w.data();

    const float nxf = (float)nx;
    const float nyf = (float)ny;
    const float nzf = (float)nz;
    const float dtOverDx = dt / dx;

    auto clampVelPos = [&](float& gx, float& gy, float& gz) {
        gx = clampf(gx, 0.0f, nxf);
        gy = clampf(gy, 0.0f, nyf);
        gz = clampf(gz, 0.0f, nzf);
    };

    parallelForChunks(nz, 2, [&](int kBegin, int kEnd) {
        for (int k = kBegin; k < kEnd; ++k) {
            const int rowStride = nx + 1;
            for (int entryIndex = activeUFacesByK.offsets[(std::size_t)k];
                 entryIndex < activeUFacesByK.offsets[(std::size_t)k + 1u];
                 ++entryIndex) {
                const auto& entry = activeUFacesByK.entries[(std::size_t)entryIndex];
                const int i = entry.i;
                const int j = entry.j;
                const int id = i + rowStride * (j + ny * k);

                const float gx = (float)i;
                const float gy = (float)j + 0.5f;
                const float gz = (float)k + 0.5f;

                float ux, uy, uz;
                velAtGrid(gx, gy, gz, uSrc, vSrc, wSrc, nx, ny, nz, ux, uy, uz);

                float midX = gx - 0.5f * dtOverDx * ux;
                float midY = gy - 0.5f * dtOverDx * uy;
                float midZ = gz - 0.5f * dtOverDx * uz;
                clampVelPos(midX, midY, midZ);

                float mx, my, mz;
                velAtGrid(midX, midY, midZ, uSrc, vSrc, wSrc, nx, ny, nz, mx, my, mz);

                float backX = gx - dtOverDx * mx;
                float backY = gy - dtOverDx * my;
                float backZ = gz - dtOverDx * mz;
                clampVelPos(backX, backY, backZ);

                uDst[(std::size_t)id] = sampleUGrid(uSrc, nx, ny, nz, backX, backY, backZ);
            }
        }
    });

    parallelForChunks(nz, 2, [&](int kBegin, int kEnd) {
        for (int k = kBegin; k < kEnd; ++k) {
            const int rowStride = nx;
            for (int entryIndex = activeVFacesByK.offsets[(std::size_t)k];
                 entryIndex < activeVFacesByK.offsets[(std::size_t)k + 1u];
                 ++entryIndex) {
                const auto& entry = activeVFacesByK.entries[(std::size_t)entryIndex];
                const int i = entry.i;
                const int j = entry.j;
                const int id = i + rowStride * (j + (ny + 1) * k);

                const float gx = (float)i + 0.5f;
                const float gy = (float)j;
                const float gz = (float)k + 0.5f;

                float ux, uy, uz;
                velAtGrid(gx, gy, gz, uSrc, vSrc, wSrc, nx, ny, nz, ux, uy, uz);

                float midX = gx - 0.5f * dtOverDx * ux;
                float midY = gy - 0.5f * dtOverDx * uy;
                float midZ = gz - 0.5f * dtOverDx * uz;
                clampVelPos(midX, midY, midZ);

                float mx, my, mz;
                velAtGrid(midX, midY, midZ, uSrc, vSrc, wSrc, nx, ny, nz, mx, my, mz);

                float backX = gx - dtOverDx * mx;
                float backY = gy - dtOverDx * my;
                float backZ = gz - dtOverDx * mz;
                clampVelPos(backX, backY, backZ);

                vDst[(std::size_t)id] = sampleVGrid(vSrc, nx, ny, nz, backX, backY, backZ);
            }
        }
    });

    parallelForChunks(nz + 1, 2, [&](int kBegin, int kEnd) {
        for (int k = kBegin; k < kEnd; ++k) {
            const int rowStride = nx;
            for (int entryIndex = activeWFacesByK.offsets[(std::size_t)k];
                 entryIndex < activeWFacesByK.offsets[(std::size_t)k + 1u];
                 ++entryIndex) {
                const auto& entry = activeWFacesByK.entries[(std::size_t)entryIndex];
                const int i = entry.i;
                const int j = entry.j;
                const int id = i + rowStride * (j + ny * k);

                const float gx = (float)i + 0.5f;
                const float gy = (float)j + 0.5f;
                const float gz = (float)k;

                float ux, uy, uz;
                velAtGrid(gx, gy, gz, uSrc, vSrc, wSrc, nx, ny, nz, ux, uy, uz);

                float midX = gx - 0.5f * dtOverDx * ux;
                float midY = gy - 0.5f * dtOverDx * uy;
                float midZ = gz - 0.5f * dtOverDx * uz;
                clampVelPos(midX, midY, midZ);

                float mx, my, mz;
                velAtGrid(midX, midY, midZ, uSrc, vSrc, wSrc, nx, ny, nz, mx, my, mz);

                float backX = gx - dtOverDx * mx;
                float backY = gy - dtOverDx * my;
                float backZ = gz - dtOverDx * mz;
                clampVelPos(backX, backY, backZ);

                wDst[(std::size_t)id] = sampleWGrid(wSrc, nx, ny, nz, backX, backY, backZ);
            }
        }
    });
}

void MACSmoke3D::addBuoyancy() {
    const float T0 = std::max(1e-3f, params.ambientTempK);

    parallelForChunks(nz, 2, [&](int kBegin, int kEnd) {
        for (int k = kBegin; k < kEnd; ++k) {
            const int cellSliceBase = nx * ny * k;
            const int vSliceBase = nx * (ny + 1) * k;
            for (int entryIndex = activeVFacesByK.offsets[(std::size_t)k];
                 entryIndex < activeVFacesByK.offsets[(std::size_t)k + 1u];
                 ++entryIndex) {
                const auto& entry = activeVFacesByK.entries[(std::size_t)entryIndex];
                const int i = entry.i;
                const int j = entry.j;
                const int id = vSliceBase + i + nx * j;

                float tAvg = temp[(std::size_t)(cellSliceBase + i + nx * (j - 1))];
                if (j < ny) {
                    tAvg += temp[(std::size_t)(cellSliceBase + i + nx * j)];
                    tAvg *= 0.5f;
                }
                v[(std::size_t)id] += dt * params.gravity * (tAvg / T0) * params.buoyancyScale;
            }
        }
    });

    if (params.velDamping > 0.0f) {
        const float damp = std::exp(-params.velDamping * dt);
        for (float& value : u) value *= damp;
        for (float& value : v) value *= damp;
        for (float& value : w) value *= damp;
    }
}

void MACSmoke3D::diffuseVelocityImplicit() {
    if (params.viscosity <= 0.0f || params.diffuseIters <= 0) return;

    const float alphaInvDx2 = (params.viscosity * dt) / (dx * dx);
    if (alphaInvDx2 <= 0.0f) return;

    if (diffusionStencilDirty) {
        rebuildDiffusionStencils();
    }

    auto enforceUBoundary = [&]() {
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                u[(std::size_t)idxU(0, j, k)] = 0.0f;
                u[(std::size_t)idxU(nx, j, k)] = 0.0f;
                for (int i = 1; i < nx; ++i) {
                    if (isSolidCell(i - 1, j, k) || isSolidCell(i, j, k)) {
                        u[(std::size_t)idxU(i, j, k)] = 0.0f;
                    }
                }
            }
        }
    };

    auto enforceVBoundary = [&]() {
        for (int k = 0; k < nz; ++k) {
            for (int i = 0; i < nx; ++i) {
                v[(std::size_t)idxV(i, 0, k)] = 0.0f;
                if (!params.openTop) {
                    v[(std::size_t)idxV(i, ny, k)] = 0.0f;
                }
            }

            for (int j = 1; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    if (isSolidCell(i, j - 1, k) || isSolidCell(i, j, k)) {
                        v[(std::size_t)idxV(i, j, k)] = 0.0f;
                    }
                }
            }

            if (params.openTop) {
                for (int i = 0; i < nx; ++i) {
                    if (isSolidCell(i, ny - 1, k)) {
                        v[(std::size_t)idxV(i, ny, k)] = 0.0f;
                    }
                }
            }
        }
    };

    auto enforceWBoundary = [&]() {
        for (int k = 0; k <= nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    if (k == 0 || k == nz) {
                        w[(std::size_t)idxW(i, j, k)] = 0.0f;
                        continue;
                    }
                    if (isSolidCell(i, j, k - 1) || isSolidCell(i, j, k)) {
                        w[(std::size_t)idxW(i, j, k)] = 0.0f;
                    }
                }
            }
        }
    };

    enforceUBoundary();
    enforceVBoundary();
    enforceWBoundary();

    // Freeze the RHS without per-step heap churn.
    u0 = u;
    v0 = v;
    w0 = w;

    auto solveComponent = [&](std::vector<float>& x,
                              const std::vector<float>& b,
                              const DiffusionStencilSet& stencilSet,
                              DiffusionScratchSet& scratch) {
        if (stencilSet.size() == 0u) return;

        scratch.ensureSize(x.size());
        std::vector<float>& r = scratch.r;
        std::vector<float>& z = scratch.z;
        std::vector<float>& p = scratch.p;
        std::vector<float>& q = scratch.q;

        auto applyA = [&](const std::vector<float>& in, std::vector<float>& out) {
            for (std::size_t idx = 0; idx < stencilSet.size(); ++idx) {
                float sumN = 0.0f;
                int n = stencilSet.xm[idx]; if (n >= 0) sumN += in[(std::size_t)n];
                n = stencilSet.xp[idx];     if (n >= 0) sumN += in[(std::size_t)n];
                n = stencilSet.ym[idx];     if (n >= 0) sumN += in[(std::size_t)n];
                n = stencilSet.yp[idx];     if (n >= 0) sumN += in[(std::size_t)n];
                n = stencilSet.zm[idx];     if (n >= 0) sumN += in[(std::size_t)n];
                n = stencilSet.zp[idx];     if (n >= 0) sumN += in[(std::size_t)n];
                const int face = stencilSet.face[idx];
                const float diag = 1.0f + alphaInvDx2 * (float)stencilSet.neighborCount[idx];
                out[(std::size_t)face] = diag * in[(std::size_t)face] - alphaInvDx2 * sumN;
            }
        };

        auto dotActive = [&](const std::vector<float>& a, const std::vector<float>& bvec) {
            double sum = 0.0;
            for (std::size_t idx = 0; idx < stencilSet.size(); ++idx) {
                const int face = stencilSet.face[idx];
                sum += (double)a[(std::size_t)face] * (double)bvec[(std::size_t)face];
            }
            return (float)sum;
        };

        auto maxAbsActive = [&](const std::vector<float>& a) {
            float m = 0.0f;
            for (std::size_t idx = 0; idx < stencilSet.size(); ++idx) {
                const int face = stencilSet.face[idx];
                m = std::max(m, std::fabs(a[(std::size_t)face]));
            }
            return m;
        };

        applyA(x, q);

        float bInf = 0.0f;
        float rInf = 0.0f;
        for (std::size_t idx = 0; idx < stencilSet.size(); ++idx) {
            const int face = stencilSet.face[idx];
            const float diag = 1.0f + alphaInvDx2 * (float)stencilSet.neighborCount[idx];
            r[(std::size_t)face] = b[(std::size_t)face] - q[(std::size_t)face];
            z[(std::size_t)face] = r[(std::size_t)face] / diag;
            p[(std::size_t)face] = z[(std::size_t)face];
            bInf = std::max(bInf, std::fabs(b[(std::size_t)face]));
            rInf = std::max(rInf, std::fabs(r[(std::size_t)face]));
        }

        const float absTol = std::max(1.0e-6f, 1.0e-6f * std::max(1.0f, bInf));
        const float relTol = std::max(1.0e-6f, 1.0e-4f * std::max(rInf, 1.0e-6f));
        if (rInf <= std::max(absTol, relTol)) {
            return;
        }

        float deltaNew = dotActive(r, z);
        if (!std::isfinite(deltaNew) || deltaNew <= 1.0e-20f) {
            return;
        }

        const int maxIters = std::max(1, params.diffuseIters);
        for (int it = 0; it < maxIters; ++it) {
            applyA(p, q);
            const float denom = dotActive(p, q);
            if (!std::isfinite(denom) || std::fabs(denom) < 1.0e-20f) {
                break;
            }

            const float alpha = deltaNew / denom;
            for (std::size_t idx = 0; idx < stencilSet.size(); ++idx) {
                const int face = stencilSet.face[idx];
                x[(std::size_t)face] += alpha * p[(std::size_t)face];
                r[(std::size_t)face] -= alpha * q[(std::size_t)face];
            }

            rInf = maxAbsActive(r);
            if (!std::isfinite(rInf) || rInf <= std::max(absTol, relTol)) {
                break;
            }

            for (std::size_t idx = 0; idx < stencilSet.size(); ++idx) {
                const int face = stencilSet.face[idx];
                const float diag = 1.0f + alphaInvDx2 * (float)stencilSet.neighborCount[idx];
                z[(std::size_t)face] = r[(std::size_t)face] / diag;
            }

            const float deltaOld = deltaNew;
            deltaNew = dotActive(r, z);
            if (!std::isfinite(deltaNew) || deltaNew <= 1.0e-20f) {
                break;
            }

            const float beta = deltaNew / (deltaOld + 1.0e-20f);
            for (std::size_t idx = 0; idx < stencilSet.size(); ++idx) {
                const int face = stencilSet.face[idx];
                p[(std::size_t)face] = z[(std::size_t)face] + beta * p[(std::size_t)face];
            }
        }
    };

    const std::size_t totalUnknowns = uDiffusionStencil.size() + vDiffusionStencil.size() + wDiffusionStencil.size();
    int activeComponents = 0;
    if (!uDiffusionStencil.face.empty()) ++activeComponents;
    if (!vDiffusionStencil.face.empty()) ++activeComponents;
    if (!wDiffusionStencil.face.empty()) ++activeComponents;

    const bool parallelComponents = (smoke3DWorkerPool().maxWorkers() > 1) &&
                                    (activeComponents > 1) &&
                                    (totalUnknowns >= 32768u);

    if (parallelComponents) {
        parallelForChunks(3, 1, [&](int begin, int end) {
            for (int component = begin; component < end; ++component) {
                switch (component) {
                    case 0:
                        solveComponent(u, u0, uDiffusionStencil, uDiffusionScratch);
                        enforceUBoundary();
                        break;
                    case 1:
                        solveComponent(v, v0, vDiffusionStencil, vDiffusionScratch);
                        enforceVBoundary();
                        break;
                    case 2:
                        solveComponent(w, w0, wDiffusionStencil, wDiffusionScratch);
                        enforceWBoundary();
                        break;
                    default:
                        break;
                }
            }
        });
    } else {
        solveComponent(u, u0, uDiffusionStencil, uDiffusionScratch);
        enforceUBoundary();

        solveComponent(v, v0, vDiffusionStencil, vDiffusionScratch);
        enforceVBoundary();

        solveComponent(w, w0, wDiffusionStencil, wDiffusionScratch);
        enforceWBoundary();
    }
}

void MACSmoke3D::diffuseScalarImplicit(std::vector<float>& phi,
                                       std::vector<float>& phi0,
                                       float diffusivity,
                                       float dissipation) {
    if (phi.empty()) return;
    if (phi0.size() != phi.size()) phi0.resize(phi.size());

    const float keep = dissipationStep(dissipation, dt);
    if (diffusivity <= 0.0f || params.diffuseIters <= 0) {
        if (keep != 1.0f) {
            for (std::size_t i = 0; i < phi.size(); ++i) {
                if (!solid[i]) phi[i] *= keep;
            }
        }
        return;
    }

    phi0 = phi;
    if (cellTmp.size() != phi.size()) cellTmp.resize(phi.size(), 0.0f);

    const float alpha = (diffusivity * dt) / (dx * dx);
    if (alpha <= 0.0f) return;

    for (int it = 0; it < params.diffuseIters; ++it) {
        parallelForChunks(nz, 2, [&](int kBegin, int kEnd) {
            for (int k = kBegin; k < kEnd; ++k) {
                for (int j = 0; j < ny; ++j) {
                    for (int i = 0; i < nx; ++i) {
                        const int id = idxCell(i, j, k);
                        if (solid[(std::size_t)id]) {
                            cellTmp[(std::size_t)id] = 0.0f;
                            continue;
                        }

                        float sum = 0.0f;
                        int count = 0;

                        if (i > 0 && !isSolidCell(i - 1, j, k)) { sum += phi[(std::size_t)idxCell(i - 1, j, k)]; count++; }
                        if (i + 1 < nx && !isSolidCell(i + 1, j, k)) { sum += phi[(std::size_t)idxCell(i + 1, j, k)]; count++; }
                        if (j > 0 && !isSolidCell(i, j - 1, k)) { sum += phi[(std::size_t)idxCell(i, j - 1, k)]; count++; }
                        if (j + 1 < ny && !isSolidCell(i, j + 1, k)) { sum += phi[(std::size_t)idxCell(i, j + 1, k)]; count++; }
                        if (k > 0 && !isSolidCell(i, j, k - 1)) { sum += phi[(std::size_t)idxCell(i, j, k - 1)]; count++; }
                        if (k + 1 < nz && !isSolidCell(i, j, k + 1)) { sum += phi[(std::size_t)idxCell(i, j, k + 1)]; count++; }

                        const float denom = 1.0f + alpha * (float)count;
                        const float xNew = (phi0[(std::size_t)id] + alpha * sum) / std::max(1e-6f, denom);
                        cellTmp[(std::size_t)id] =
                            (1.0f - params.diffuseOmega) * phi[(std::size_t)id] +
                            params.diffuseOmega * xNew;
                    }
                }
            }
        });
        phi.swap(cellTmp);
    }

    if (keep != 1.0f) {
        for (std::size_t i = 0; i < phi.size(); ++i) {
            if (!solid[i]) phi[i] *= keep;
        }
    }
}

void MACSmoke3D::project() {
    lastPressureSolveMs = 0.0f;
    lastPressureIterations = 0;

    const int cellCount = nx * ny * nz;
    if (cellCount <= 0) return;

    const float invDx = 1.0f / dx;
    const float invDt = 1.0f / std::max(1e-8f, dt);
    const float dx2 = dx * dx;

    bool hasDirichletReference = false;
    if (params.openTop) {
        const int topJ = ny - 1;
        for (int k = 0; k < nz && !hasDirichletReference; ++k) {
            const int sliceBase = nx * ny * k + nx * topJ;
            for (int i = 0; i < nx; ++i) {
                if (!solid[(std::size_t)(sliceBase + i)]) {
                    hasDirichletReference = true;
                    break;
                }
            }
        }
    }

    parallelForChunks(nz, 2, [&](int kBegin, int kEnd) {
        for (int k = kBegin; k < kEnd; ++k) {
            for (int entryIndex = activeCellsByK.offsets[(std::size_t)k];
                 entryIndex < activeCellsByK.offsets[(std::size_t)k + 1u];
                 ++entryIndex) {
                const auto& entry = activeCellsByK.entries[(std::size_t)entryIndex];
                const int i = entry.i;
                const int j = entry.j;
                const int id = idxCell(i, j, k);

                if (!std::isfinite(pressure[(std::size_t)id])) {
                    pressure[(std::size_t)id] = 0.0f;
                }

                const float divCell =
                    (u[(std::size_t)idxU(i + 1, j, k)] - u[(std::size_t)idxU(i, j, k)] +
                     v[(std::size_t)idxV(i, j + 1, k)] - v[(std::size_t)idxV(i, j, k)] +
                     w[(std::size_t)idxW(i, j, k + 1)] - w[(std::size_t)idxW(i, j, k)]) * invDx;
                rhs[(std::size_t)id] = -divCell * invDt;
            }
        }
    });

    const auto solverMode = static_cast<PressureSolverMode>(params.pressureSolverMode);
    const auto solveStart = std::chrono::high_resolution_clock::now();
    if (solverMode == PressureSolverMode::Multigrid) {
        if (pressureOperatorDirty) {
            pressurePoisson.configure(
                nx, ny, nz, dx,
                params.openTop,
                solid,
                fluidMask,
                /*removeMeanForGauge=*/!hasDirichletReference);
            pressureOperatorDirty = false;
        }

        pressurePoisson.setMGControls(params.pressureMGCoarseIters, params.pressureMGRelativeTol);
        pressurePoisson.setMGSmoother(true, clampf(params.pressureMGOmega, 0.1f, 1.95f));
        pressurePoisson.solveMG(
            pressure,
            rhs,
            std::max(1, params.pressureMGVCycles),
            std::max(0.0f, params.pressureTol),
            dt);
        lastPressureIterations = pressurePoisson.lastIterations();
    } else {
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
            sum += pressure[(std::size_t)nid];
            diag++;
        };

        auto computeResidual = [&]() {
            float maxResidual = 0.0f;
            for (int k3 = 0; k3 < nz; ++k3) {
                for (int entryIndex = activeCellsByK.offsets[(std::size_t)k3];
                     entryIndex < activeCellsByK.offsets[(std::size_t)k3 + 1u];
                     ++entryIndex) {
                    const auto& entry = activeCellsByK.entries[(std::size_t)entryIndex];
                    const int i3 = entry.i;
                    const int j3 = entry.j;
                    const int id = idxCell(i3, j3, k3);

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
            return maxResidual;
        };

        const bool useJacobi = (solverMode == PressureSolverMode::Jacobi);
        const int maxIters = std::max(1, params.pressureIters);
        const float rbgsOmega = clampf(params.pressureOmega, 0.0f, 1.95f);
        const float jacobiOmega = clampf(params.pressureOmega, 0.0f, 1.0f);

        int itersUsed = 0;

        if (useJacobi) {
            for (int it = 0; it < maxIters; ++it) {
                itersUsed = it + 1;
                for (int k3 = 0; k3 < nz; ++k3) {
                    for (int entryIndex = activeCellsByK.offsets[(std::size_t)k3];
                         entryIndex < activeCellsByK.offsets[(std::size_t)k3 + 1u];
                         ++entryIndex) {
                        const auto& entry = activeCellsByK.entries[(std::size_t)entryIndex];
                        const int i3 = entry.i;
                        const int j3 = entry.j;
                        const int id = idxCell(i3, j3, k3);

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
                pressure.swap(pressureTmp);
                if (computeResidual() * dt <= params.pressureTol) break;
            }
        } else {
            for (int it = 0; it < maxIters; ++it) {
                itersUsed = it + 1;
                for (int color = 0; color < 2; ++color) {
                    for (int k3 = 0; k3 < nz; ++k3) {
                        for (int entryIndex = activeCellsByK.offsets[(std::size_t)k3];
                             entryIndex < activeCellsByK.offsets[(std::size_t)k3 + 1u];
                             ++entryIndex) {
                            const auto& entry = activeCellsByK.entries[(std::size_t)entryIndex];
                            const int i3 = entry.i;
                            const int j3 = entry.j;
                            if (((i3 + j3 + k3) & 1) != color) continue;
                            const int id = idxCell(i3, j3, k3);

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
                if (computeResidual() * dt <= params.pressureTol) break;
            }
        }

        lastPressureIterations = itersUsed;
    }

    const auto solveEnd = std::chrono::high_resolution_clock::now();
    lastPressureSolveMs = std::chrono::duration<float, std::milli>(solveEnd - solveStart).count();

    const float scale = dt / dx;

    parallelForChunks(nz, 2, [&](int kBegin, int kEnd) {
        for (int k = kBegin; k < kEnd; ++k) {
            for (int entryIndex = activeUFacesByK.offsets[(std::size_t)k];
                 entryIndex < activeUFacesByK.offsets[(std::size_t)k + 1u];
                 ++entryIndex) {
                const auto& entry = activeUFacesByK.entries[(std::size_t)entryIndex];
                const int i = entry.i;
                const int j = entry.j;
                const int face = idxU(i, j, k);
                const float pL = pressure[(std::size_t)idxCell(i - 1, j, k)];
                const float pR = pressure[(std::size_t)idxCell(i, j, k)];
                u[(std::size_t)face] -= scale * (pR - pL);
            }

            for (int entryIndex = activeVFacesByK.offsets[(std::size_t)k];
                 entryIndex < activeVFacesByK.offsets[(std::size_t)k + 1u];
                 ++entryIndex) {
                const auto& entry = activeVFacesByK.entries[(std::size_t)entryIndex];
                const int i = entry.i;
                const int j = entry.j;
                const int face = idxV(i, j, k);
                if (j == ny) {
                    const float pB = pressure[(std::size_t)idxCell(i, ny - 1, k)];
                    v[(std::size_t)face] += scale * pB;
                } else {
                    const float pB = pressure[(std::size_t)idxCell(i, j - 1, k)];
                    const float pT = pressure[(std::size_t)idxCell(i, j, k)];
                    v[(std::size_t)face] -= scale * (pT - pB);
                }
            }
        }
    });

    parallelForChunks(nz + 1, 2, [&](int kBegin, int kEnd) {
        for (int k = kBegin; k < kEnd; ++k) {
            for (int entryIndex = activeWFacesByK.offsets[(std::size_t)k];
                 entryIndex < activeWFacesByK.offsets[(std::size_t)k + 1u];
                 ++entryIndex) {
                const auto& entry = activeWFacesByK.entries[(std::size_t)entryIndex];
                const int i = entry.i;
                const int j = entry.j;
                const int face = idxW(i, j, k);
                const float pBk = pressure[(std::size_t)idxCell(i, j, k - 1)];
                const float pFr = pressure[(std::size_t)idxCell(i, j, k)];
                w[(std::size_t)face] -= scale * (pFr - pBk);
            }
        }
    });
}

void MACSmoke3D::advectScalars() {
    smoke0.swap(smoke);
    temp0.swap(temp);

    const float keepSmoke = dissipationStep(params.smokeDissipation, dt);
    const float keepTemp = dissipationStep(params.tempDissipation, dt);

    const float nxf = (float)nx;
    const float nyf = (float)ny;
    const float nzf = (float)nz;
    const float sampleNyf = nyf + (params.openTop ? 1.0f : 0.0f);
    const float dtOverDx = dt / dx;

    const float* SMOKE_RESTRICT smokeSrc = smoke0.data();
    const float* SMOKE_RESTRICT tempSrc = temp0.data();
    const float* SMOKE_RESTRICT uSrc = u.data();
    const float* SMOKE_RESTRICT vSrc = v.data();
    const float* SMOKE_RESTRICT wSrc = w.data();
    float* SMOKE_RESTRICT smokeDst = smoke.data();
    float* SMOKE_RESTRICT tempDst = temp.data();

    auto clampScalarPos = [&](float& gx, float& gy, float& gz) {
        gx = clampf(gx, 0.0f, nxf);
        gy = clampf(gy, 0.0f, sampleNyf);
        gz = clampf(gz, 0.0f, nzf);
    };

    parallelForChunks(nz, 2, [&](int kBegin, int kEnd) {
        for (int k = kBegin; k < kEnd; ++k) {
            const int cellSliceBase = nx * ny * k;
            for (int entryIndex = activeCellsByK.offsets[(std::size_t)k];
                 entryIndex < activeCellsByK.offsets[(std::size_t)k + 1u];
                 ++entryIndex) {
                const auto& entry = activeCellsByK.entries[(std::size_t)entryIndex];
                const int i = entry.i;
                const int j = entry.j;
                const int id = cellSliceBase + i + nx * j;

                const float gx = (float)i + 0.5f;
                const float gy = (float)j + 0.5f;
                const float gz = (float)k + 0.5f;

                float u1, v1, w1;
                velAtGrid(gx, gy, gz, uSrc, vSrc, wSrc, nx, ny, nz, u1, v1, w1);
                float midX = gx - 0.5f * dtOverDx * u1;
                float midY = gy - 0.5f * dtOverDx * v1;
                float midZ = gz - 0.5f * dtOverDx * w1;
                clampScalarPos(midX, midY, midZ);

                float u2, v2, w2;
                velAtGrid(midX, midY, midZ, uSrc, vSrc, wSrc, nx, ny, nz, u2, v2, w2);
                float backX = gx - dtOverDx * u2;
                float backY = gy - dtOverDx * v2;
                float backZ = gz - dtOverDx * w2;
                clampScalarPos(backX, backY, backZ);

                if (backY > nyf) {
                    if (params.openTop) {
                        smokeDst[(std::size_t)id] = 0.0f;
                        tempDst[(std::size_t)id] = 0.0f;
                        continue;
                    }
                    backY = nyf;
                }

                float sampledSmoke = 0.0f;
                float sampledTemp = 0.0f;
                sampleCellCenteredPairGrid(smokeSrc, tempSrc, nx, ny, nz, backX, backY, backZ,
                                           sampledSmoke, sampledTemp);
                smokeDst[(std::size_t)id] = keepSmoke * sampledSmoke;
                tempDst[(std::size_t)id] = keepTemp * sampledTemp;
            }
        }
    });
}

void MACSmoke3D::ensureDerivedDebugFields() {
    if (!derivedFieldsDirty) return;
    rasterizeDebugFields();
}

bool MACSmoke3D::hasActiveVelocity(float eps) const {
    for (float value : u) if (std::fabs(value) > eps) return true;
    for (float value : v) if (std::fabs(value) > eps) return true;
    for (float value : w) if (std::fabs(value) > eps) return true;
    return false;
}

bool MACSmoke3D::hasActiveScalar(const std::vector<float>& field, float eps) const {
    const std::size_t count = std::min(field.size(), solid.size());
    for (std::size_t i = 0; i < count; ++i) {
        if (!solid[i] && std::fabs(field[i]) > eps) return true;
    }
    return false;
}

void MACSmoke3D::clearVelocityState() {
    std::fill(u.begin(), u.end(), 0.0f);
    std::fill(v.begin(), v.end(), 0.0f);
    std::fill(w.begin(), w.end(), 0.0f);
    std::fill(u0.begin(), u0.end(), 0.0f);
    std::fill(v0.begin(), v0.end(), 0.0f);
    std::fill(w0.begin(), w0.end(), 0.0f);
    std::fill(uTmp.begin(), uTmp.end(), 0.0f);
    std::fill(vTmp.begin(), vTmp.end(), 0.0f);
    std::fill(wTmp.begin(), wTmp.end(), 0.0f);
    std::fill(pressure.begin(), pressure.end(), 0.0f);
    std::fill(pressureTmp.begin(), pressureTmp.end(), 0.0f);
    std::fill(rhs.begin(), rhs.end(), 0.0f);
}

void MACSmoke3D::clearScalarState(std::vector<float>& field) {
    std::fill(field.begin(), field.end(), 0.0f);
}

void MACSmoke3D::clearDerivedDebugFields() {
    std::fill(divergence.begin(), divergence.end(), 0.0f);
    std::fill(speed.begin(), speed.end(), 0.0f);
    derivedFieldsDirty = false;
}

void MACSmoke3D::rasterizeDebugFields() {
    parallelForChunks(nz, 2, [&](int kBegin, int kEnd) {
        for (int k = kBegin; k < kEnd; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    const int id = idxCell(i, j, k);
                    if (solid[(std::size_t)id]) {
                        divergence[(std::size_t)id] = 0.0f;
                        speed[(std::size_t)id] = 0.0f;
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

                    divergence[(std::size_t)id] = (uR - uL + vT - vB + wFr - wBk) / dx;

                    const float cx = (i + 0.5f) * dx;
                    const float cy = (j + 0.5f) * dx;
                    const float cz = (k + 0.5f) * dx;
                    float uc, vc, wc;
                    velAt(cx, cy, cz, u, v, w, uc, vc, wc);
                    speed[(std::size_t)id] = std::sqrt(uc * uc + vc * vc + wc * wc);
                }
            }
        }
    });

    derivedFieldsDirty = false;
}

void MACSmoke3D::updateIdleStats(float stepMs) {
    lastStats.nx = nx;
    lastStats.ny = ny;
    lastStats.nz = nz;
    lastStats.activeCells = fluidCellCount;
    lastStats.maxSpeed = 0.0f;
    lastStats.maxDivergence = 0.0f;
    lastStats.dt = dt;
    lastStats.lastStepMs = stepMs;
    lastStats.pressureMs = 0.0f;
    lastStats.pressureIters = 0;
    lastStats.timings.reset();
    lastStats.timings.totalMs = stepMs;
    lastStats.backendName = "CPU Smoke 3D";
    lastStats.bytesAllocated =
        u.size() * sizeof(float) + v.size() * sizeof(float) + w.size() * sizeof(float) +
        u0.size() * sizeof(float) + v0.size() * sizeof(float) + w0.size() * sizeof(float) +
        uTmp.size() * sizeof(float) + vTmp.size() * sizeof(float) + wTmp.size() * sizeof(float) +
        pressure.size() * sizeof(float) + pressureTmp.size() * sizeof(float) + rhs.size() * sizeof(float) +
        smoke.size() * sizeof(float) + smoke0.size() * sizeof(float) +
        temp.size() * sizeof(float) + temp0.size() * sizeof(float) +
        cellTmp.size() * sizeof(float) + divergence.size() * sizeof(float) + speed.size() * sizeof(float) +
        solid.size() * sizeof(uint8_t) + solidUser.size() * sizeof(uint8_t) + fluidMask.size() * sizeof(uint8_t) +
        (uDiffusionStencil.face.size() + uDiffusionStencil.xm.size() + uDiffusionStencil.xp.size() +
         uDiffusionStencil.ym.size() + uDiffusionStencil.yp.size() + uDiffusionStencil.zm.size() +
         uDiffusionStencil.zp.size()) * sizeof(int) + uDiffusionStencil.neighborCount.size() * sizeof(uint8_t) +
        (vDiffusionStencil.face.size() + vDiffusionStencil.xm.size() + vDiffusionStencil.xp.size() +
         vDiffusionStencil.ym.size() + vDiffusionStencil.yp.size() + vDiffusionStencil.zm.size() +
         vDiffusionStencil.zp.size()) * sizeof(int) + vDiffusionStencil.neighborCount.size() * sizeof(uint8_t) +
        (wDiffusionStencil.face.size() + wDiffusionStencil.xm.size() + wDiffusionStencil.xp.size() +
         wDiffusionStencil.ym.size() + wDiffusionStencil.yp.size() + wDiffusionStencil.zm.size() +
         wDiffusionStencil.zp.size()) * sizeof(int) + wDiffusionStencil.neighborCount.size() * sizeof(uint8_t) +
        (activeCellsByK.offsets.size() + activeUFacesByK.offsets.size() + activeVFacesByK.offsets.size() +
         activeWFacesByK.offsets.size()) * sizeof(int) +
        (activeCellsByK.entries.size() + activeUFacesByK.entries.size() + activeVFacesByK.entries.size() +
         activeWFacesByK.entries.size()) * sizeof(SliceIJWorkList::Entry) +
        (uDiffusionScratch.r.size() + uDiffusionScratch.z.size() + uDiffusionScratch.p.size() + uDiffusionScratch.q.size() +
         vDiffusionScratch.r.size() + vDiffusionScratch.z.size() + vDiffusionScratch.p.size() + vDiffusionScratch.q.size() +
         wDiffusionScratch.r.size() + wDiffusionScratch.z.size() + wDiffusionScratch.p.size() + wDiffusionScratch.q.size()) * sizeof(float);
}

void MACSmoke3D::updateStats(float stepMs) {
    lastStats.nx = nx;
    lastStats.ny = ny;
    lastStats.nz = nz;
    lastStats.activeCells = 0;
    lastStats.maxSpeed = 0.0f;
    lastStats.maxDivergence = 0.0f;
    lastStats.dt = dt;
    lastStats.lastStepMs = stepMs;
    lastStats.pressureMs = 0.0f;
    lastStats.pressureIters = 0;
    lastStats.timings.reset();
    lastStats.timings.totalMs = stepMs;
    lastStats.backendName = "CPU Smoke 3D";
    lastStats.bytesAllocated =
        u.size() * sizeof(float) + v.size() * sizeof(float) + w.size() * sizeof(float) +
        u0.size() * sizeof(float) + v0.size() * sizeof(float) + w0.size() * sizeof(float) +
        uTmp.size() * sizeof(float) + vTmp.size() * sizeof(float) + wTmp.size() * sizeof(float) +
        pressure.size() * sizeof(float) + pressureTmp.size() * sizeof(float) + rhs.size() * sizeof(float) +
        smoke.size() * sizeof(float) + smoke0.size() * sizeof(float) +
        temp.size() * sizeof(float) + temp0.size() * sizeof(float) +
        cellTmp.size() * sizeof(float) + divergence.size() * sizeof(float) + speed.size() * sizeof(float) +
        solid.size() * sizeof(uint8_t) + solidUser.size() * sizeof(uint8_t) + fluidMask.size() * sizeof(uint8_t) +
        (uDiffusionStencil.face.size() + uDiffusionStencil.xm.size() + uDiffusionStencil.xp.size() +
         uDiffusionStencil.ym.size() + uDiffusionStencil.yp.size() + uDiffusionStencil.zm.size() +
         uDiffusionStencil.zp.size()) * sizeof(int) + uDiffusionStencil.neighborCount.size() * sizeof(uint8_t) +
        (vDiffusionStencil.face.size() + vDiffusionStencil.xm.size() + vDiffusionStencil.xp.size() +
         vDiffusionStencil.ym.size() + vDiffusionStencil.yp.size() + vDiffusionStencil.zm.size() +
         vDiffusionStencil.zp.size()) * sizeof(int) + vDiffusionStencil.neighborCount.size() * sizeof(uint8_t) +
        (wDiffusionStencil.face.size() + wDiffusionStencil.xm.size() + wDiffusionStencil.xp.size() +
         wDiffusionStencil.ym.size() + wDiffusionStencil.yp.size() + wDiffusionStencil.zm.size() +
         wDiffusionStencil.zp.size()) * sizeof(int) + wDiffusionStencil.neighborCount.size() * sizeof(uint8_t) +
        (activeCellsByK.offsets.size() + activeUFacesByK.offsets.size() + activeVFacesByK.offsets.size() +
         activeWFacesByK.offsets.size()) * sizeof(int) +
        (activeCellsByK.entries.size() + activeUFacesByK.entries.size() + activeVFacesByK.entries.size() +
         activeWFacesByK.entries.size()) * sizeof(SliceIJWorkList::Entry) +
        (uDiffusionScratch.r.size() + uDiffusionScratch.z.size() + uDiffusionScratch.p.size() + uDiffusionScratch.q.size() +
         vDiffusionScratch.r.size() + vDiffusionScratch.z.size() + vDiffusionScratch.p.size() + vDiffusionScratch.q.size() +
         wDiffusionScratch.r.size() + wDiffusionScratch.z.size() + wDiffusionScratch.p.size() + wDiffusionScratch.q.size()) * sizeof(float);

    for (float value : u) lastStats.maxSpeed = std::max(lastStats.maxSpeed, std::fabs(value));
    for (float value : v) lastStats.maxSpeed = std::max(lastStats.maxSpeed, std::fabs(value));
    for (float value : w) lastStats.maxSpeed = std::max(lastStats.maxSpeed, std::fabs(value));

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const int id = idxCell(i, j, k);
                if (solid[(std::size_t)id]) continue;
                lastStats.activeCells++;

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

                const float divCell = (uR - uL + vT - vB + wFr - wBk) / dx;
                lastStats.maxDivergence = std::max(lastStats.maxDivergence, std::fabs(divCell));
            }
        }
    }
}

void MACSmoke3D::step() {
    using clock = std::chrono::high_resolution_clock;
    const auto frameStart = clock::now();
    auto stageStart = frameStart;
    SimStageTimings timings;
    lastPressureSolveMs = 0.0f;
    lastPressureIterations = 0;

    auto markStage = [&](float& bucket) {
        const auto now = clock::now();
        bucket += std::chrono::duration<float, std::milli>(now - stageStart).count();
        stageStart = now;
    };

    if (topologyDirty) rebuildBorderSolids();
    applyBoundary();
    markStage(timings.setupMs);

    const bool smokeActive = hasActiveScalar(smoke, 1.0e-5f);
    const bool tempActive = hasActiveScalar(temp, 1.0e-5f);
    const bool velocityActive = hasActiveVelocity(1.0e-6f);

    if (!smokeActive) {
        clearScalarState(smoke);
        clearScalarState(smoke0);
    }
    if (!tempActive) {
        clearScalarState(temp);
        clearScalarState(temp0);
    }

    if (!smokeActive && !tempActive && !velocityActive) {
        clearVelocityState();
        clearDerivedDebugFields();

        auto statsStart = clock::now();
        updateIdleStats(0.0f);
        auto statsEnd = clock::now();
        timings.statsMs += std::chrono::duration<float, std::milli>(statsEnd - statsStart).count();

        const float stepMs = std::chrono::duration<float, std::milli>(statsEnd - frameStart).count();
        lastStats.lastStepMs = stepMs;
        lastStats.pressureMs = 0.0f;
        lastStats.pressureIters = 0;
        lastStats.timings = timings;
        lastStats.timings.totalMs = stepMs;
        return;
    }

    if (velocityActive || tempActive) {
        advectVelocity();
        applyBoundary();
        markStage(timings.advectVelocityMs);

        addBuoyancy();
        applyBoundary();
        markStage(timings.forcesMs);

        diffuseVelocityImplicit();
        markStage(timings.diffuseVelocityMs);

        project();
        markStage(timings.projectMs);
    } else {
        clearVelocityState();
        markStage(timings.setupMs);
    }

    if (smokeActive || tempActive) {
        advectScalars();
        markStage(timings.advectScalarsMs);

        diffuseScalarImplicit(smoke, smoke0, params.smokeDiffusivity, 1.0f);
        diffuseScalarImplicit(temp, temp0, params.tempDiffusivity, 1.0f);
        applyBoundary();
        markStage(timings.diffuseScalarsMs);
    } else {
        clearScalarState(smoke);
        clearScalarState(smoke0);
        clearScalarState(temp);
        clearScalarState(temp0);
        markStage(timings.setupMs);
    }

    derivedFieldsDirty = true;

    auto statsStart = clock::now();
    updateStats(0.0f);
    auto statsEnd = clock::now();
    timings.statsMs += std::chrono::duration<float, std::milli>(statsEnd - statsStart).count();

    const float stepMs = std::chrono::duration<float, std::milli>(statsEnd - frameStart).count();
    lastStats.lastStepMs = stepMs;
    lastStats.pressureMs = lastPressureSolveMs;
    lastStats.pressureIters = lastPressureIterations;
    lastStats.timings = timings;
    lastStats.timings.totalMs = stepMs;
}


void MACSmoke3D::addSmokeSourceSphere(const Vec3& center, float radius, float amount, const Vec3& velocity) {
    const float r = std::max(0.0f, radius);
    if (r <= 0.0f || amount <= 0.0f) return;

    const int minI = clampi((int)std::floor((center.x - r) / dx), 0, nx - 1);
    const int maxI = clampi((int)std::floor((center.x + r) / dx), 0, nx - 1);
    const int minJ = clampi((int)std::floor((center.y - r) / dx), 0, ny - 1);
    const int maxJ = clampi((int)std::floor((center.y + r) / dx), 0, ny - 1);
    const int minK = clampi((int)std::floor((center.z - r) / dx), 0, nz - 1);
    const int maxK = clampi((int)std::floor((center.z + r) / dx), 0, nz - 1);
    const float r2 = r * r;

    for (int k = minK; k <= maxK; ++k) {
        for (int j = minJ; j <= maxJ; ++j) {
            for (int i = minI; i <= maxI; ++i) {
                const int id = idxCell(i, j, k);
                if (solid[(std::size_t)id]) continue;

                const float cx = (i + 0.5f) * dx;
                const float cy = (j + 0.5f) * dx;
                const float cz = (k + 0.5f) * dx;
                const float d2 =
                    (cx - center.x) * (cx - center.x) +
                    (cy - center.y) * (cy - center.y) +
                    (cz - center.z) * (cz - center.z);
                if (d2 > r2) continue;

                smoke[(std::size_t)id] = clamp01(smoke[(std::size_t)id] + amount);

                u[(std::size_t)idxU(i, j, k)] = velocity.x;
                u[(std::size_t)idxU(i + 1, j, k)] = velocity.x;
                v[(std::size_t)idxV(i, j, k)] = velocity.y;
                v[(std::size_t)idxV(i, j + 1, k)] = velocity.y;
                w[(std::size_t)idxW(i, j, k)] = velocity.z;
                w[(std::size_t)idxW(i, j, k + 1)] = velocity.z;
            }
        }
    }

    applyBoundary();
    derivedFieldsDirty = true;
    updateStats(0.0f);
}

void MACSmoke3D::addHeatSourceSphere(const Vec3& center, float radius, float amount) {
    const float r = std::max(0.0f, radius);
    if (r <= 0.0f || amount <= 0.0f) return;

    const int minI = clampi((int)std::floor((center.x - r) / dx), 0, nx - 1);
    const int maxI = clampi((int)std::floor((center.x + r) / dx), 0, nx - 1);
    const int minJ = clampi((int)std::floor((center.y - r) / dx), 0, ny - 1);
    const int maxJ = clampi((int)std::floor((center.y + r) / dx), 0, ny - 1);
    const int minK = clampi((int)std::floor((center.z - r) / dx), 0, nz - 1);
    const int maxK = clampi((int)std::floor((center.z + r) / dx), 0, nz - 1);
    const float r2 = r * r;

    for (int k = minK; k <= maxK; ++k) {
        for (int j = minJ; j <= maxJ; ++j) {
            for (int i = minI; i <= maxI; ++i) {
                const int id = idxCell(i, j, k);
                if (solid[(std::size_t)id]) continue;

                const float cx = (i + 0.5f) * dx;
                const float cy = (j + 0.5f) * dx;
                const float cz = (k + 0.5f) * dx;
                const float d2 =
                    (cx - center.x) * (cx - center.x) +
                    (cy - center.y) * (cy - center.y) +
                    (cz - center.z) * (cz - center.z);
                if (d2 > r2) continue;

                temp[(std::size_t)id] += amount;
            }
        }
    }

    derivedFieldsDirty = true;
    updateStats(0.0f);
}

void MACSmoke3D::setVoxelSolids(const std::vector<uint8_t>& mask) {
    const int cellCount = nx * ny * nz;
    solidUser.assign((std::size_t)cellCount, (uint8_t)0);
    if ((int)mask.size() == cellCount) {
        solidUser = mask;
    }
    topologyDirty = true;
    rebuildBorderSolids();
    applyBoundary();
    derivedFieldsDirty = true;
    updateStats(0.0f);
}

MACSmoke3D::SliceData MACSmoke3D::copyDebugSlice(SliceAxis axis, int index, DebugField field) {
    const std::vector<float>* src = &smoke;
    switch (field) {
        case DebugField::Smoke:        src = &smoke; break;
        case DebugField::Temperature:  src = &temp; break;
        case DebugField::Pressure:     src = &pressure; break;
        case DebugField::Divergence:
            ensureDerivedDebugFields();
            src = &divergence;
            break;
        case DebugField::Speed:
            ensureDerivedDebugFields();
            src = &speed;
            break;
    }

    SliceData out;
    if (nx <= 0 || ny <= 0 || nz <= 0 || src->empty()) return out;

    if (axis == SliceAxis::XY) {
        const int kk = clampi(index, 0, nz - 1);
        out.width = nx;
        out.height = ny;
        out.values.assign((std::size_t)nx * (std::size_t)ny, 0.0f);
        out.solid.assign((std::size_t)nx * (std::size_t)ny, (uint8_t)0);

        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const int srcId = idxCell(i, j, kk);
                const int dstId = i + nx * j;
                out.values[(std::size_t)dstId] = (*src)[(std::size_t)srcId];
                out.solid[(std::size_t)dstId] = solid[(std::size_t)srcId];
            }
        }
        return out;
    }

    if (axis == SliceAxis::XZ) {
        const int jj = clampi(index, 0, ny - 1);
        out.width = nx;
        out.height = nz;
        out.values.assign((std::size_t)nx * (std::size_t)nz, 0.0f);
        out.solid.assign((std::size_t)nx * (std::size_t)nz, (uint8_t)0);

        for (int k = 0; k < nz; ++k) {
            for (int i = 0; i < nx; ++i) {
                const int srcId = idxCell(i, jj, k);
                const int dstId = i + nx * k;
                out.values[(std::size_t)dstId] = (*src)[(std::size_t)srcId];
                out.solid[(std::size_t)dstId] = solid[(std::size_t)srcId];
            }
        }
        return out;
    }

    const int ii = clampi(index, 0, nx - 1);
    out.width = ny;
    out.height = nz;
    out.values.assign((std::size_t)ny * (std::size_t)nz, 0.0f);
    out.solid.assign((std::size_t)ny * (std::size_t)nz, (uint8_t)0);

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            const int srcId = idxCell(ii, j, k);
            const int dstId = j + ny * k;
            out.values[(std::size_t)dstId] = (*src)[(std::size_t)srcId];
            out.solid[(std::size_t)dstId] = solid[(std::size_t)srcId];
        }
    }

    return out;
}
