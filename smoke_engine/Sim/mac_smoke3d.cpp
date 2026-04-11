#include "mac_smoke3d.h"

#include <chrono>
#include <cmath>
#include <thread>


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

template <typename Fn>
inline void parallelForChunks(int count, int minChunk, Fn&& fn) {
    if (count <= 0) return;

    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    int taskCount = (int)std::min<unsigned>(hw, (unsigned)std::max(1, count / std::max(1, minChunk)));
    if (taskCount <= 1) {
        fn(0, count);
        return;
    }

    const int chunk = (count + taskCount - 1) / taskCount;
    std::vector<std::thread> workers;
    workers.reserve((std::size_t)taskCount - 1u);

    for (int task = 1; task < taskCount; ++task) {
        const int begin = task * chunk;
        if (begin >= count) break;
        const int end = std::min(count, begin + chunk);
        workers.emplace_back([&, begin, end]() { fn(begin, end); });
    }

    fn(0, std::min(count, chunk));
    for (auto& worker : workers) worker.join();
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

    topologyDirty = false;
    pressureOperatorDirty = true;
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
    u0 = u;
    v0 = v;
    w0 = w;

    const float domainX = nx * dx;
    const float domainY = ny * dx;
    const float domainZ = nz * dx;

    auto clampPos = [&](float& x, float& y, float& z) {
        x = clampf(x, 0.0f, domainX);
        y = clampf(y, 0.0f, domainY);
        z = clampf(z, 0.0f, domainZ);
    };

    parallelForChunks(nz, 2, [&](int kBegin, int kEnd) {
        for (int k = kBegin; k < kEnd; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i <= nx; ++i) {
                    const int id = idxU(i, j, k);
                    const bool leftSolid = (i - 1 >= 0) ? isSolidCell(i - 1, j, k) : true;
                    const bool rightSolid = (i < nx) ? isSolidCell(i, j, k) : true;
                    if (leftSolid || rightSolid) {
                        uTmp[(std::size_t)id] = 0.0f;
                        continue;
                    }

                    float x = i * dx;
                    float y = (j + 0.5f) * dx;
                    float z = (k + 0.5f) * dx;
                    float ux, uy, uz;
                    velAt(x, y, z, u0, v0, w0, ux, uy, uz);

                    float midX = x - 0.5f * dt * ux;
                    float midY = y - 0.5f * dt * uy;
                    float midZ = z - 0.5f * dt * uz;
                    clampPos(midX, midY, midZ);

                    float mx, my, mz;
                    velAt(midX, midY, midZ, u0, v0, w0, mx, my, mz);

                    float backX = x - dt * mx;
                    float backY = y - dt * my;
                    float backZ = z - dt * mz;
                    clampPos(backX, backY, backZ);
                    uTmp[(std::size_t)id] = sampleU(u0, backX, backY, backZ);
                }
            }
        }
    });

    parallelForChunks(nz, 2, [&](int kBegin, int kEnd) {
        for (int k = kBegin; k < kEnd; ++k) {
            for (int j = 0; j <= ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    const int id = idxV(i, j, k);
                    const bool botSolid = (j - 1 >= 0) ? isSolidCell(i, j - 1, k) : true;
                    const bool topSolid = (j < ny) ? isSolidCell(i, j, k) : !params.openTop;
                    if (botSolid || topSolid) {
                        vTmp[(std::size_t)id] = 0.0f;
                        continue;
                    }

                    float x = (i + 0.5f) * dx;
                    float y = j * dx;
                    float z = (k + 0.5f) * dx;
                    float ux, uy, uz;
                    velAt(x, y, z, u0, v0, w0, ux, uy, uz);

                    float midX = x - 0.5f * dt * ux;
                    float midY = y - 0.5f * dt * uy;
                    float midZ = z - 0.5f * dt * uz;
                    clampPos(midX, midY, midZ);

                    float mx, my, mz;
                    velAt(midX, midY, midZ, u0, v0, w0, mx, my, mz);

                    float backX = x - dt * mx;
                    float backY = y - dt * my;
                    float backZ = z - dt * mz;
                    clampPos(backX, backY, backZ);
                    vTmp[(std::size_t)id] = sampleV(v0, backX, backY, backZ);
                }
            }
        }
    });

    parallelForChunks(nz + 1, 2, [&](int kBegin, int kEnd) {
        for (int k = kBegin; k < kEnd; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    const int id = idxW(i, j, k);
                    const bool backSolid = (k - 1 >= 0) ? isSolidCell(i, j, k - 1) : true;
                    const bool frontSolid = (k < nz) ? isSolidCell(i, j, k) : true;
                    if (backSolid || frontSolid) {
                        wTmp[(std::size_t)id] = 0.0f;
                        continue;
                    }

                    float x = (i + 0.5f) * dx;
                    float y = (j + 0.5f) * dx;
                    float z = k * dx;
                    float ux, uy, uz;
                    velAt(x, y, z, u0, v0, w0, ux, uy, uz);

                    float midX = x - 0.5f * dt * ux;
                    float midY = y - 0.5f * dt * uy;
                    float midZ = z - 0.5f * dt * uz;
                    clampPos(midX, midY, midZ);

                    float mx, my, mz;
                    velAt(midX, midY, midZ, u0, v0, w0, mx, my, mz);

                    float backX = x - dt * mx;
                    float backY = y - dt * my;
                    float backZ = z - dt * mz;
                    clampPos(backX, backY, backZ);
                    wTmp[(std::size_t)id] = sampleW(w0, backX, backY, backZ);
                }
            }
        }
    });

    u.swap(uTmp);
    v.swap(vTmp);
    w.swap(wTmp);
}

void MACSmoke3D::addBuoyancy() {
    const float T0 = std::max(1e-3f, params.ambientTempK);

    parallelForChunks(nz, 2, [&](int kBegin, int kEnd) {
        for (int k = kBegin; k < kEnd; ++k) {
            for (int j = 0; j <= ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    const int id = idxV(i, j, k);
                    const bool botSolid = (j - 1 >= 0) ? isSolidCell(i, j - 1, k) : true;
                    const bool topSolid = (j < ny) ? isSolidCell(i, j, k) : !params.openTop;
                    if (botSolid || topSolid) continue;

                    float tAvg = 0.0f;
                    int count = 0;
                    if (j - 1 >= 0 && !isSolidCell(i, j - 1, k)) {
                        tAvg += temp[(std::size_t)idxCell(i, j - 1, k)];
                        count++;
                    }
                    if (j < ny && !isSolidCell(i, j, k)) {
                        tAvg += temp[(std::size_t)idxCell(i, j, k)];
                        count++;
                    }
                    if (count == 0) continue;
                    tAvg /= (float)count;
                    v[(std::size_t)id] += dt * params.gravity * (tAvg / T0) * params.buoyancyScale;
                }
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

    const auto jacobiUpdate = [&](float b, float sumN, int count) {
        return (b + alphaInvDx2 * sumN) / (1.0f + alphaInvDx2 * (float)count);
    };

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

    const std::vector<float> bU = u;
    const std::vector<float> bV = v;
    const std::vector<float> bW = w;

    for (int it = 0; it < params.diffuseIters; ++it) {
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i <= nx; ++i) {
                    const int id = idxU(i, j, k);
                    if (isFixedU(i, j, k)) {
                        uTmp[(std::size_t)id] = u[(std::size_t)id];
                        continue;
                    }

                    float sumN = 0.0f;
                    int count = 0;
                    if (i > 0) { sumN += u[(std::size_t)idxU(i - 1, j, k)]; count++; }
                    if (i < nx) { sumN += u[(std::size_t)idxU(i + 1, j, k)]; count++; }
                    if (j > 0) { sumN += u[(std::size_t)idxU(i, j - 1, k)]; count++; }
                    if (j + 1 < ny) { sumN += u[(std::size_t)idxU(i, j + 1, k)]; count++; }
                    if (k > 0) { sumN += u[(std::size_t)idxU(i, j, k - 1)]; count++; }
                    if (k + 1 < nz) { sumN += u[(std::size_t)idxU(i, j, k + 1)]; count++; }

                    const float xNew = jacobiUpdate(bU[(std::size_t)id], sumN, count);
                    uTmp[(std::size_t)id] =
                        (1.0f - params.diffuseOmega) * u[(std::size_t)id] +
                        params.diffuseOmega * xNew;
                }
            }
        }
        u.swap(uTmp);
        applyBoundary();
    }

    for (int it = 0; it < params.diffuseIters; ++it) {
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j <= ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    const int id = idxV(i, j, k);
                    if (isFixedV(i, j, k)) {
                        vTmp[(std::size_t)id] = v[(std::size_t)id];
                        continue;
                    }

                    float sumN = 0.0f;
                    int count = 0;
                    if (i > 0) { sumN += v[(std::size_t)idxV(i - 1, j, k)]; count++; }
                    if (i + 1 < nx) { sumN += v[(std::size_t)idxV(i + 1, j, k)]; count++; }
                    if (j > 0) { sumN += v[(std::size_t)idxV(i, j - 1, k)]; count++; }
                    if (j < ny) { sumN += v[(std::size_t)idxV(i, j + 1, k)]; count++; }
                    if (k > 0) { sumN += v[(std::size_t)idxV(i, j, k - 1)]; count++; }
                    if (k + 1 < nz) { sumN += v[(std::size_t)idxV(i, j, k + 1)]; count++; }

                    const float xNew = jacobiUpdate(bV[(std::size_t)id], sumN, count);
                    vTmp[(std::size_t)id] =
                        (1.0f - params.diffuseOmega) * v[(std::size_t)id] +
                        params.diffuseOmega * xNew;
                }
            }
        }
        v.swap(vTmp);
        applyBoundary();
    }

    for (int it = 0; it < params.diffuseIters; ++it) {
        for (int k = 0; k <= nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    const int id = idxW(i, j, k);
                    if (isFixedW(i, j, k)) {
                        wTmp[(std::size_t)id] = w[(std::size_t)id];
                        continue;
                    }

                    float sumN = 0.0f;
                    int count = 0;
                    if (i > 0) { sumN += w[(std::size_t)idxW(i - 1, j, k)]; count++; }
                    if (i + 1 < nx) { sumN += w[(std::size_t)idxW(i + 1, j, k)]; count++; }
                    if (j > 0) { sumN += w[(std::size_t)idxW(i, j - 1, k)]; count++; }
                    if (j + 1 < ny) { sumN += w[(std::size_t)idxW(i, j + 1, k)]; count++; }
                    if (k > 0) { sumN += w[(std::size_t)idxW(i, j, k - 1)]; count++; }
                    if (k < nz) { sumN += w[(std::size_t)idxW(i, j, k + 1)]; count++; }

                    const float xNew = jacobiUpdate(bW[(std::size_t)id], sumN, count);
                    wTmp[(std::size_t)id] =
                        (1.0f - params.diffuseOmega) * w[(std::size_t)id] +
                        params.diffuseOmega * xNew;
                }
            }
        }
        w.swap(wTmp);
        applyBoundary();
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

    auto isFluidCell = [&](int i, int j, int k) -> bool {
        if (i < 0 || j < 0 || k < 0 || i >= nx || j >= ny || k >= nz) return false;
        return !solid[(std::size_t)idxCell(i, j, k)];
    };

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const int id = idxCell(i, j, k);
                if (solid[(std::size_t)id]) {
                    rhs[(std::size_t)id] = 0.0f;
                    pressure[(std::size_t)id] = 0.0f;
                    continue;
                }

                if (!std::isfinite(pressure[(std::size_t)id])) pressure[(std::size_t)id] = 0.0f;

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

                if (params.openTop && j == ny - 1) {
                    hasDirichletReference = true;
                }
            }
        }
    }

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
                for (int j3 = 0; j3 < ny; ++j3) {
                    for (int i3 = 0; i3 < nx; ++i3) {
                        const int id = idxCell(i3, j3, k3);
                        if (solid[(std::size_t)id]) continue;

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

        const bool useJacobi = (solverMode == PressureSolverMode::Jacobi);
        const int maxIters = std::max(1, params.pressureIters);
        const float rbgsOmega = clampf(params.pressureOmega, 0.0f, 1.95f);
        const float jacobiOmega = clampf(params.pressureOmega, 0.0f, 1.0f);

        int itersUsed = 0;

        if (useJacobi) {
            for (int it = 0; it < maxIters; ++it) {
                itersUsed = it + 1;
                for (int k3 = 0; k3 < nz; ++k3) {
                    for (int j3 = 0; j3 < ny; ++j3) {
                        for (int i3 = 0; i3 < nx; ++i3) {
                            const int id = idxCell(i3, j3, k3);
                            if (solid[(std::size_t)id]) {
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
                itersUsed = it + 1;
                for (int color = 0; color < 2; ++color) {
                    for (int k3 = 0; k3 < nz; ++k3) {
                        for (int j3 = 0; j3 < ny; ++j3) {
                            for (int i3 = 0; i3 < nx; ++i3) {
                                if (((i3 + j3 + k3) & 1) != color) continue;
                                const int id = idxCell(i3, j3, k3);
                                if (solid[(std::size_t)id]) {
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

        lastPressureIterations = itersUsed;
    }

    const auto solveEnd = std::chrono::high_resolution_clock::now();
    lastPressureSolveMs = std::chrono::duration<float, std::milli>(solveEnd - solveStart).count();

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

void MACSmoke3D::advectScalars() {
    smoke0 = smoke;
    temp0 = temp;

    const float keepSmoke = dissipationStep(params.smokeDissipation, dt);
    const float keepTemp = dissipationStep(params.tempDissipation, dt);

    const float domainX = nx * dx;
    const float domainY = ny * dx;
    const float domainZ = nz * dx;
    const float domainYSampleMax = domainY + (params.openTop ? dx : 0.0f);
    const float invDx = 1.0f / dx;
    const int sliceStride = nx * ny;

    parallelForChunks(nz, 2, [&](int kBegin, int kEnd) {
        for (int k = kBegin; k < kEnd; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    const int id = idxCell(i, j, k);
                    if (solid[(std::size_t)id]) {
                        smoke[(std::size_t)id] = 0.0f;
                        temp[(std::size_t)id] = 0.0f;
                        continue;
                    }

                    const float x = (i + 0.5f) * dx;
                    const float y = (j + 0.5f) * dx;
                    const float z = (k + 0.5f) * dx;

                    float u1, v1, w1;
                    velAt(x, y, z, u, v, w, u1, v1, w1);
                    const float midX = clampf(x - 0.5f * dt * u1, 0.0f, domainX);
                    const float midY = clampf(y - 0.5f * dt * v1, 0.0f, domainYSampleMax);
                    const float midZ = clampf(z - 0.5f * dt * w1, 0.0f, domainZ);

                    float u2, v2, w2;
                    velAt(midX, midY, midZ, u, v, w, u2, v2, w2);
                    const float backX = clampf(x - dt * u2, 0.0f, domainX);
                    float backY = clampf(y - dt * v2, 0.0f, domainYSampleMax);
                    const float backZ = clampf(z - dt * w2, 0.0f, domainZ);

                    if (backY > domainY) {
                        if (params.openTop) {
                            smoke[(std::size_t)id] = 0.0f;
                            temp[(std::size_t)id] = 0.0f;
                            continue;
                        }
                        backY = domainY;
                    }

                    const float fx = backX * invDx - 0.5f;
                    const float fy = backY * invDx - 0.5f;
                    const float fz = backZ * invDx - 0.5f;

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

                    const int row00 = j0 * nx + k0 * sliceStride;
                    const int row10 = j1 * nx + k0 * sliceStride;
                    const int row01 = j0 * nx + k1 * sliceStride;
                    const int row11 = j1 * nx + k1 * sliceStride;

                    const float s00 = wx0 * smoke0[(std::size_t)(row00 + i0)] + wx1 * smoke0[(std::size_t)(row00 + i1)];
                    const float s10 = wx0 * smoke0[(std::size_t)(row10 + i0)] + wx1 * smoke0[(std::size_t)(row10 + i1)];
                    const float s01 = wx0 * smoke0[(std::size_t)(row01 + i0)] + wx1 * smoke0[(std::size_t)(row01 + i1)];
                    const float s11 = wx0 * smoke0[(std::size_t)(row11 + i0)] + wx1 * smoke0[(std::size_t)(row11 + i1)];

                    const float t00 = wx0 * temp0[(std::size_t)(row00 + i0)] + wx1 * temp0[(std::size_t)(row00 + i1)];
                    const float t10 = wx0 * temp0[(std::size_t)(row10 + i0)] + wx1 * temp0[(std::size_t)(row10 + i1)];
                    const float t01 = wx0 * temp0[(std::size_t)(row01 + i0)] + wx1 * temp0[(std::size_t)(row01 + i1)];
                    const float t11 = wx0 * temp0[(std::size_t)(row11 + i0)] + wx1 * temp0[(std::size_t)(row11 + i1)];

                    const float smokeY0 = wy0 * s00 + wy1 * s10;
                    const float smokeY1 = wy0 * s01 + wy1 * s11;
                    const float tempY0 = wy0 * t00 + wy1 * t10;
                    const float tempY1 = wy0 * t01 + wy1 * t11;

                    smoke[(std::size_t)id] = keepSmoke * (wz0 * smokeY0 + wz1 * smokeY1);
                    temp[(std::size_t)id] = keepTemp * (wz0 * tempY0 + wz1 * tempY1);
                }
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
        solid.size() * sizeof(uint8_t) + solidUser.size() * sizeof(uint8_t) + fluidMask.size() * sizeof(uint8_t);
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
        solid.size() * sizeof(uint8_t) + solidUser.size() * sizeof(uint8_t) + fluidMask.size() * sizeof(uint8_t);

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
