#pragma once

#include "water3d_common.h"
#include "../chunk_worker_pool.h"

#include <algorithm>
#include <cmath>

inline ChunkWorkerPool& water3dWorkerPool() { return sharedChunkWorkerPool(); }

inline float MACWater3D::sampleCellCentered(
    const std::vector<float>& field, float x, float y, float z) const {
    if (field.empty() || nx <= 0 || ny <= 0 || nz <= 0) return 0.0f;

    const float fx = x / dx - 0.5f;
    const float fy = y / dx - 0.5f;
    const float fz = z / dx - 0.5f;

    int i0 = water3d_internal::clampi((int)std::floor(fx), 0, nx - 1);
    int j0 = water3d_internal::clampi((int)std::floor(fy), 0, ny - 1);
    int k0 = water3d_internal::clampi((int)std::floor(fz), 0, nz - 1);

    const int i1 = std::min(i0 + 1, nx - 1);
    const int j1 = std::min(j0 + 1, ny - 1);
    const int k1 = std::min(k0 + 1, nz - 1);

    const float tx = water3d_internal::clampf(fx - (float)i0, 0.0f, 1.0f);
    const float ty = water3d_internal::clampf(fy - (float)j0, 0.0f, 1.0f);
    const float tz = water3d_internal::clampf(fz - (float)k0, 0.0f, 1.0f);

    float value = 0.0f;
    for (int dk = 0; dk < 2; ++dk) {
        const int kk = (dk == 0) ? k0 : k1;
        const float wz = (dk == 0) ? (1.0f - tz) : tz;
        for (int dj = 0; dj < 2; ++dj) {
            const int jj = (dj == 0) ? j0 : j1;
            const float wy = (dj == 0) ? (1.0f - ty) : ty;
            for (int di = 0; di < 2; ++di) {
                const int ii = (di == 0) ? i0 : i1;
                const float wx = (di == 0) ? (1.0f - tx) : tx;
                value += wx * wy * wz * field[(std::size_t)idxCell(ii, jj, kk)];
            }
        }
    }
    return value;
}

inline float MACWater3D::sampleU(
    const std::vector<float>& field, float x, float y, float z) const {
    if (field.empty()) return 0.0f;

    const float fx = x / dx;
    const float fy = y / dx - 0.5f;
    const float fz = z / dx - 0.5f;

    int i0 = water3d_internal::clampi((int)std::floor(fx), 0, nx);
    int j0 = water3d_internal::clampi((int)std::floor(fy), 0, ny - 1);
    int k0 = water3d_internal::clampi((int)std::floor(fz), 0, nz - 1);

    const int i1 = std::min(i0 + 1, nx);
    const int j1 = std::min(j0 + 1, ny - 1);
    const int k1 = std::min(k0 + 1, nz - 1);

    const float tx = water3d_internal::clampf(fx - (float)i0, 0.0f, 1.0f);
    const float ty = water3d_internal::clampf(fy - (float)j0, 0.0f, 1.0f);
    const float tz = water3d_internal::clampf(fz - (float)k0, 0.0f, 1.0f);

    float value = 0.0f;
    for (int dk = 0; dk < 2; ++dk) {
        const int kk = (dk == 0) ? k0 : k1;
        const float wz = (dk == 0) ? (1.0f - tz) : tz;
        for (int dj = 0; dj < 2; ++dj) {
            const int jj = (dj == 0) ? j0 : j1;
            const float wy = (dj == 0) ? (1.0f - ty) : ty;
            for (int di = 0; di < 2; ++di) {
                const int ii = (di == 0) ? i0 : i1;
                const float wx = (di == 0) ? (1.0f - tx) : tx;
                value += wx * wy * wz * field[(std::size_t)idxU(ii, jj, kk)];
            }
        }
    }
    return value;
}

inline float MACWater3D::sampleV(
    const std::vector<float>& field, float x, float y, float z) const {
    if (field.empty()) return 0.0f;

    const float fx = x / dx - 0.5f;
    const float fy = y / dx;
    const float fz = z / dx - 0.5f;

    int i0 = water3d_internal::clampi((int)std::floor(fx), 0, nx - 1);
    int j0 = water3d_internal::clampi((int)std::floor(fy), 0, ny);
    int k0 = water3d_internal::clampi((int)std::floor(fz), 0, nz - 1);

    const int i1 = std::min(i0 + 1, nx - 1);
    const int j1 = std::min(j0 + 1, ny);
    const int k1 = std::min(k0 + 1, nz - 1);

    const float tx = water3d_internal::clampf(fx - (float)i0, 0.0f, 1.0f);
    const float ty = water3d_internal::clampf(fy - (float)j0, 0.0f, 1.0f);
    const float tz = water3d_internal::clampf(fz - (float)k0, 0.0f, 1.0f);

    float value = 0.0f;
    for (int dk = 0; dk < 2; ++dk) {
        const int kk = (dk == 0) ? k0 : k1;
        const float wz = (dk == 0) ? (1.0f - tz) : tz;
        for (int dj = 0; dj < 2; ++dj) {
            const int jj = (dj == 0) ? j0 : j1;
            const float wy = (dj == 0) ? (1.0f - ty) : ty;
            for (int di = 0; di < 2; ++di) {
                const int ii = (di == 0) ? i0 : i1;
                const float wx = (di == 0) ? (1.0f - tx) : tx;
                value += wx * wy * wz * field[(std::size_t)idxV(ii, jj, kk)];
            }
        }
    }
    return value;
}

inline float MACWater3D::sampleW(
    const std::vector<float>& field, float x, float y, float z) const {
    if (field.empty()) return 0.0f;

    const float fx = x / dx - 0.5f;
    const float fy = y / dx - 0.5f;
    const float fz = z / dx;

    int i0 = water3d_internal::clampi((int)std::floor(fx), 0, nx - 1);
    int j0 = water3d_internal::clampi((int)std::floor(fy), 0, ny - 1);
    int k0 = water3d_internal::clampi((int)std::floor(fz), 0, nz);

    const int i1 = std::min(i0 + 1, nx - 1);
    const int j1 = std::min(j0 + 1, ny - 1);
    const int k1 = std::min(k0 + 1, nz);

    const float tx = water3d_internal::clampf(fx - (float)i0, 0.0f, 1.0f);
    const float ty = water3d_internal::clampf(fy - (float)j0, 0.0f, 1.0f);
    const float tz = water3d_internal::clampf(fz - (float)k0, 0.0f, 1.0f);

    float value = 0.0f;
    for (int dk = 0; dk < 2; ++dk) {
        const int kk = (dk == 0) ? k0 : k1;
        const float wz = (dk == 0) ? (1.0f - tz) : tz;
        for (int dj = 0; dj < 2; ++dj) {
            const int jj = (dj == 0) ? j0 : j1;
            const float wy = (dj == 0) ? (1.0f - ty) : ty;
            for (int di = 0; di < 2; ++di) {
                const int ii = (di == 0) ? i0 : i1;
                const float wx = (di == 0) ? (1.0f - tx) : tx;
                value += wx * wy * wz * field[(std::size_t)idxW(ii, jj, kk)];
            }
        }
    }
    return value;
}

inline void MACWater3D::velAt(
    float x, float y, float z,
    const std::vector<float>& fu,
    const std::vector<float>& fv,
    const std::vector<float>& fw,
    float& outU, float& outV, float& outW) const {
    outU = sampleU(fu, x, y, z);
    outV = sampleV(fv, x, y, z);
    outW = sampleW(fw, x, y, z);
}

inline void MACWater3D::particleToGrid() {
    std::fill(u.begin(), u.end(), 0.0f);
    std::fill(v.begin(), v.end(), 0.0f);
    std::fill(w.begin(), w.end(), 0.0f);
    std::fill(uWeight.begin(), uWeight.end(), 0.0f);
    std::fill(vWeight.begin(), vWeight.end(), 0.0f);
    std::fill(wWeight.begin(), wWeight.end(), 0.0f);

    const bool apic = params.useAPIC;

    auto scatterFace = [&](const Particle& p,
                           float ox, float oy, float oz,
                           int sx, int sy, int sz,
                           auto idxFn,
                           std::vector<float>& dst,
                           std::vector<float>& weight,
                           float base,
                           float c0,
                           float c1,
                           float c2) {
        const float fx = p.x / dx - ox;
        const float fy = p.y / dx - oy;
        const float fz = p.z / dx - oz;

        int i0 = water3d_internal::clampi((int)std::floor(fx), 0, sx - 1);
        int j0 = water3d_internal::clampi((int)std::floor(fy), 0, sy - 1);
        int k0 = water3d_internal::clampi((int)std::floor(fz), 0, sz - 1);

        const int i1 = std::min(i0 + 1, sx - 1);
        const int j1 = std::min(j0 + 1, sy - 1);
        const int k1 = std::min(k0 + 1, sz - 1);

        const float tx = water3d_internal::clampf(fx - (float)i0, 0.0f, 1.0f);
        const float ty = water3d_internal::clampf(fy - (float)j0, 0.0f, 1.0f);
        const float tz = water3d_internal::clampf(fz - (float)k0, 0.0f, 1.0f);

        for (int dk = 0; dk < 2; ++dk) {
            const int kk = (dk == 0) ? k0 : k1;
            const float wz = (dk == 0) ? (1.0f - tz) : tz;
            const float pzFace = (kk + oz) * dx;
            for (int dj = 0; dj < 2; ++dj) {
                const int jj = (dj == 0) ? j0 : j1;
                const float wy = (dj == 0) ? (1.0f - ty) : ty;
                const float pyFace = (jj + oy) * dx;
                for (int di = 0; di < 2; ++di) {
                    const int ii = (di == 0) ? i0 : i1;
                    const float wx = (di == 0) ? (1.0f - tx) : tx;
                    const float pxFace = (ii + ox) * dx;
                    const float wght = wx * wy * wz;
                    const int id = idxFn(ii, jj, kk);

                    float value = base;
                    if (apic) {
                        value += c0 * (pxFace - p.x) + c1 * (pyFace - p.y) + c2 * (pzFace - p.z);
                    }

                    dst[(std::size_t)id] += wght * value;
                    weight[(std::size_t)id] += wght;
                }
            }
        }
    };

    for (const Particle& p : particles) {
        scatterFace(p, 0.0f, 0.5f, 0.5f, nx + 1, ny, nz,
                    [&](int i, int j, int k) { return idxU(i, j, k); },
                    u, uWeight, p.u, p.c00, p.c01, p.c02);
        scatterFace(p, 0.5f, 0.0f, 0.5f, nx, ny + 1, nz,
                    [&](int i, int j, int k) { return idxV(i, j, k); },
                    v, vWeight, p.v, p.c10, p.c11, p.c12);
        scatterFace(p, 0.5f, 0.5f, 0.0f, nx, ny, nz + 1,
                    [&](int i, int j, int k) { return idxW(i, j, k); },
                    w, wWeight, p.w, p.c20, p.c21, p.c22);
    }

    for (std::size_t i = 0; i < u.size(); ++i) {
        u[i] = (uWeight[i] > 1e-6f) ? (u[i] / uWeight[i]) : 0.0f;
    }
    for (std::size_t i = 0; i < v.size(); ++i) {
        v[i] = (vWeight[i] > 1e-6f) ? (v[i] / vWeight[i]) : 0.0f;
    }
    for (std::size_t i = 0; i < w.size(); ++i) {
        w[i] = (wWeight[i] > 1e-6f) ? (w[i] / wWeight[i]) : 0.0f;
    }

    applyBoundary();
}

inline void MACWater3D::buildLiquidMask(bool applyDilations) {
    const int cellCount = nx * ny * nz;
    if ((int)liquid.size() != cellCount) liquid.assign((std::size_t)cellCount, (uint8_t)0);
    std::fill(liquid.begin(), liquid.end(), (uint8_t)0);

    for (const Particle& p : particles) {
        int i = water3d_internal::clampi((int)std::floor(p.x / dx), 0, nx - 1);
        int j = water3d_internal::clampi((int)std::floor(p.y / dx), 0, ny - 1);
        int k = water3d_internal::clampi((int)std::floor(p.z / dx), 0, nz - 1);
        const int id = idxCell(i, j, k);
        if (solid[(std::size_t)id]) continue;
        liquid[(std::size_t)id] = (uint8_t)1;
    }

    if (!applyDilations) return;

    const int dilations = std::max(0, params.maskDilations);
    for (int it = 0; it < dilations; ++it) {
        std::vector<uint8_t> next = liquid;
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    const int id = idxCell(i, j, k);
                    if (solid[(std::size_t)id] || liquid[(std::size_t)id]) continue;

                    const bool Isnear =
                        (i > 0 && liquid[(std::size_t)idxCell(i - 1, j, k)]) ||
                        (i + 1 < nx && liquid[(std::size_t)idxCell(i + 1, j, k)]) ||
                        (j > 0 && liquid[(std::size_t)idxCell(i, j - 1, k)]) ||
                        (j + 1 < ny && liquid[(std::size_t)idxCell(i, j + 1, k)]) ||
                        (k > 0 && liquid[(std::size_t)idxCell(i, j, k - 1)]) ||
                        (k + 1 < nz && liquid[(std::size_t)idxCell(i, j, k + 1)]);
                    if (Isnear) next[(std::size_t)id] = (uint8_t)1;
                }
            }
        }
        for (int id = 0; id < cellCount; ++id) {
            if (solid[(std::size_t)id]) next[(std::size_t)id] = 0;
        }
        liquid.swap(next);
    }
}

inline void MACWater3D::rebuildDiffusionStencils() {
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

inline void MACWater3D::diffuseVelocityImplicit() {
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

    uPrev = u;
    vPrev = v;
    wPrev = w;

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

    const bool parallelComponents = (water3dWorkerPool().maxWorkers() > 1) &&
                                    (activeComponents > 1) &&
                                    (totalUnknowns >= 32768u);

    if (parallelComponents) {
        water3dWorkerPool().parallelFor(3, 1, [&](int begin, int end) {
            for (int component = begin; component < end; ++component) {
                switch (component) {
                    case 0:
                        solveComponent(u, uPrev, uDiffusionStencil, uDiffusionScratch);
                        enforceUBoundary();
                        break;
                    case 1:
                        solveComponent(v, vPrev, vDiffusionStencil, vDiffusionScratch);
                        enforceVBoundary();
                        break;
                    case 2:
                        solveComponent(w, wPrev, wDiffusionStencil, wDiffusionScratch);
                        enforceWBoundary();
                        break;
                    default:
                        break;
                }
            }
        });
    } else {
        solveComponent(u, uPrev, uDiffusionStencil, uDiffusionScratch);
        enforceUBoundary();

        solveComponent(v, vPrev, vDiffusionStencil, vDiffusionScratch);
        enforceVBoundary();

        solveComponent(w, wPrev, wDiffusionStencil, wDiffusionScratch);
        enforceWBoundary();
    }
}

inline void MACWater3D::extrapolateVelocity() {
    if (validU.size() != u.size()) validU.assign(u.size(), (uint8_t)0);
    if (validV.size() != v.size()) validV.assign(v.size(), (uint8_t)0);
    if (validW.size() != w.size()) validW.assign(w.size(), (uint8_t)0);
    if (validUNext.size() != u.size()) validUNext.assign(u.size(), (uint8_t)0);
    if (validVNext.size() != v.size()) validVNext.assign(v.size(), (uint8_t)0);
    if (validWNext.size() != w.size()) validWNext.assign(w.size(), (uint8_t)0);

    std::fill(validU.begin(), validU.end(), (uint8_t)0);
    std::fill(validV.begin(), validV.end(), (uint8_t)0);
    std::fill(validW.begin(), validW.end(), (uint8_t)0);
    std::fill(validUNext.begin(), validUNext.end(), (uint8_t)0);
    std::fill(validVNext.begin(), validVNext.end(), (uint8_t)0);
    std::fill(validWNext.begin(), validWNext.end(), (uint8_t)0);

    extrapFrontierU.clear();
    extrapFrontierV.clear();
    extrapFrontierW.clear();
    extrapNextFrontierU.clear();
    extrapNextFrontierV.clear();
    extrapNextFrontierW.clear();
    extrapFrontierU.reserve(u.size() / 8u + 1u);
    extrapFrontierV.reserve(v.size() / 8u + 1u);
    extrapFrontierW.reserve(w.size() / 8u + 1u);
    extrapNextFrontierU.reserve(u.size() / 8u + 1u);
    extrapNextFrontierV.reserve(v.size() / 8u + 1u);
    extrapNextFrontierW.reserve(w.size() / 8u + 1u);

    auto isOpenU = [&](int i, int j, int k) {
        if (i <= 0 || i >= nx) return false;
        return !isSolidCell(i - 1, j, k) && !isSolidCell(i, j, k);
    };
    auto isOpenV = [&](int i, int j, int k) {
        if (j <= 0) return false;
        if (j >= ny) return params.openTop && !isSolidCell(i, j - 1, k);
        return !isSolidCell(i, j - 1, k) && !isSolidCell(i, j, k);
    };
    auto isOpenW = [&](int i, int j, int k) {
        if (k <= 0 || k >= nz) return false;
        return !isSolidCell(i, j, k - 1) && !isSolidCell(i, j, k);
    };

    auto decodeU = [&](int id, int& i, int& j, int& k) {
        i = id % (nx + 1);
        const int t = id / (nx + 1);
        j = t % ny;
        k = t / ny;
    };
    auto decodeV = [&](int id, int& i, int& j, int& k) {
        i = id % nx;
        const int t = id / nx;
        j = t % (ny + 1);
        k = t / (ny + 1);
    };
    auto decodeW = [&](int id, int& i, int& j, int& k) {
        i = id % nx;
        const int t = id / nx;
        j = t % ny;
        k = t / ny;
    };

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i <= nx; ++i) {
                const int id = idxU(i, j, k);
                if (!isOpenU(i, j, k)) {
                    u[(std::size_t)id] = 0.0f;
                    continue;
                }
                if (uWeight[(std::size_t)id] > 1.0e-6f) {
                    validU[(std::size_t)id] = (uint8_t)1;
                    extrapFrontierU.push_back(id);
                }
            }
        }
    }

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const int id = idxV(i, j, k);
                if (!isOpenV(i, j, k)) {
                    v[(std::size_t)id] = 0.0f;
                    continue;
                }
                if (vWeight[(std::size_t)id] > 1.0e-6f) {
                    validV[(std::size_t)id] = (uint8_t)1;
                    extrapFrontierV.push_back(id);
                }
            }
        }
    }

    for (int k = 0; k <= nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const int id = idxW(i, j, k);
                if (!isOpenW(i, j, k)) {
                    w[(std::size_t)id] = 0.0f;
                    continue;
                }
                if (wWeight[(std::size_t)id] > 1.0e-6f) {
                    validW[(std::size_t)id] = (uint8_t)1;
                    extrapFrontierW.push_back(id);
                }
            }
        }
    }

    auto propagateU = [&]() {
        if (extrapFrontierU.empty()) return false;
        extrapNextFrontierU.clear();
        for (int id : extrapFrontierU) {
            int i, j, k;
            decodeU(id, i, j, k);
            const int neighbors[6][3] = {
                {i - 1, j, k}, {i + 1, j, k}, {i, j - 1, k},
                {i, j + 1, k}, {i, j, k - 1}, {i, j, k + 1}
            };
            for (const auto& n : neighbors) {
                const int ni = n[0], nj = n[1], nk = n[2];
                if (ni < 0 || ni > nx || nj < 0 || nj >= ny || nk < 0 || nk >= nz) continue;
                const int nid = idxU(ni, nj, nk);
                if (validU[(std::size_t)nid] || validUNext[(std::size_t)nid]) continue;
                if (!isOpenU(ni, nj, nk)) continue;
                validUNext[(std::size_t)nid] = (uint8_t)1;
                extrapNextFrontierU.push_back(nid);
            }
        }
        if (extrapNextFrontierU.empty()) return false;
        std::size_t write = 0;
        for (int nid : extrapNextFrontierU) {
            int i, j, k;
            decodeU(nid, i, j, k);
            float sum = 0.0f;
            int count = 0;
            if (i > 0)      { const int n = idxU(i - 1, j, k); if (validU[(std::size_t)n]) { sum += u[(std::size_t)n]; ++count; } }
            if (i < nx)     { const int n = idxU(i + 1, j, k); if (validU[(std::size_t)n]) { sum += u[(std::size_t)n]; ++count; } }
            if (j > 0)      { const int n = idxU(i, j - 1, k); if (validU[(std::size_t)n]) { sum += u[(std::size_t)n]; ++count; } }
            if (j + 1 < ny) { const int n = idxU(i, j + 1, k); if (validU[(std::size_t)n]) { sum += u[(std::size_t)n]; ++count; } }
            if (k > 0)      { const int n = idxU(i, j, k - 1); if (validU[(std::size_t)n]) { sum += u[(std::size_t)n]; ++count; } }
            if (k + 1 < nz) { const int n = idxU(i, j, k + 1); if (validU[(std::size_t)n]) { sum += u[(std::size_t)n]; ++count; } }
            validUNext[(std::size_t)nid] = (uint8_t)0;
            if (count <= 0) continue;
            u[(std::size_t)nid] = sum / (float)count;
            extrapNextFrontierU[write++] = nid;
        }
        extrapNextFrontierU.resize(write);
        for (int nextId : extrapNextFrontierU) validU[(std::size_t)nextId] = (uint8_t)1;
        extrapFrontierU.swap(extrapNextFrontierU);
        return !extrapFrontierU.empty();
    };

    auto propagateV = [&]() {
        if (extrapFrontierV.empty()) return false;
        extrapNextFrontierV.clear();
        for (int id : extrapFrontierV) {
            int i, j, k;
            decodeV(id, i, j, k);
            const int neighbors[6][3] = {
                {i - 1, j, k}, {i + 1, j, k}, {i, j - 1, k},
                {i, j + 1, k}, {i, j, k - 1}, {i, j, k + 1}
            };
            for (const auto& n : neighbors) {
                const int ni = n[0], nj = n[1], nk = n[2];
                if (ni < 0 || ni >= nx || nj < 0 || nj > ny || nk < 0 || nk >= nz) continue;
                const int nid = idxV(ni, nj, nk);
                if (validV[(std::size_t)nid] || validVNext[(std::size_t)nid]) continue;
                if (!isOpenV(ni, nj, nk)) continue;
                validVNext[(std::size_t)nid] = (uint8_t)1;
                extrapNextFrontierV.push_back(nid);
            }
        }
        if (extrapNextFrontierV.empty()) return false;
        std::size_t write = 0;
        for (int nid : extrapNextFrontierV) {
            int i, j, k;
            decodeV(nid, i, j, k);
            float sum = 0.0f;
            int count = 0;
            if (i > 0)      { const int n = idxV(i - 1, j, k); if (validV[(std::size_t)n]) { sum += v[(std::size_t)n]; ++count; } }
            if (i + 1 < nx) { const int n = idxV(i + 1, j, k); if (validV[(std::size_t)n]) { sum += v[(std::size_t)n]; ++count; } }
            if (j > 0)      { const int n = idxV(i, j - 1, k); if (validV[(std::size_t)n]) { sum += v[(std::size_t)n]; ++count; } }
            if (j < ny)     { const int n = idxV(i, j + 1, k); if (validV[(std::size_t)n]) { sum += v[(std::size_t)n]; ++count; } }
            if (k > 0)      { const int n = idxV(i, j, k - 1); if (validV[(std::size_t)n]) { sum += v[(std::size_t)n]; ++count; } }
            if (k + 1 < nz) { const int n = idxV(i, j, k + 1); if (validV[(std::size_t)n]) { sum += v[(std::size_t)n]; ++count; } }
            validVNext[(std::size_t)nid] = (uint8_t)0;
            if (count <= 0) continue;
            v[(std::size_t)nid] = sum / (float)count;
            extrapNextFrontierV[write++] = nid;
        }
        extrapNextFrontierV.resize(write);
        for (int nextId : extrapNextFrontierV) validV[(std::size_t)nextId] = (uint8_t)1;
        extrapFrontierV.swap(extrapNextFrontierV);
        return !extrapFrontierV.empty();
    };

    auto propagateW = [&]() {
        if (extrapFrontierW.empty()) return false;
        extrapNextFrontierW.clear();
        for (int id : extrapFrontierW) {
            int i, j, k;
            decodeW(id, i, j, k);
            const int neighbors[6][3] = {
                {i - 1, j, k}, {i + 1, j, k}, {i, j - 1, k},
                {i, j + 1, k}, {i, j, k - 1}, {i, j, k + 1}
            };
            for (const auto& n : neighbors) {
                const int ni = n[0], nj = n[1], nk = n[2];
                if (ni < 0 || ni >= nx || nj < 0 || nj >= ny || nk < 0 || nk > nz) continue;
                const int nid = idxW(ni, nj, nk);
                if (validW[(std::size_t)nid] || validWNext[(std::size_t)nid]) continue;
                if (!isOpenW(ni, nj, nk)) continue;
                validWNext[(std::size_t)nid] = (uint8_t)1;
                extrapNextFrontierW.push_back(nid);
            }
        }
        if (extrapNextFrontierW.empty()) return false;
        std::size_t write = 0;
        for (int nid : extrapNextFrontierW) {
            int i, j, k;
            decodeW(nid, i, j, k);
            float sum = 0.0f;
            int count = 0;
            if (i > 0)      { const int n = idxW(i - 1, j, k); if (validW[(std::size_t)n]) { sum += w[(std::size_t)n]; ++count; } }
            if (i + 1 < nx) { const int n = idxW(i + 1, j, k); if (validW[(std::size_t)n]) { sum += w[(std::size_t)n]; ++count; } }
            if (j > 0)      { const int n = idxW(i, j - 1, k); if (validW[(std::size_t)n]) { sum += w[(std::size_t)n]; ++count; } }
            if (j + 1 < ny) { const int n = idxW(i, j + 1, k); if (validW[(std::size_t)n]) { sum += w[(std::size_t)n]; ++count; } }
            if (k > 0)      { const int n = idxW(i, j, k - 1); if (validW[(std::size_t)n]) { sum += w[(std::size_t)n]; ++count; } }
            if (k < nz)     { const int n = idxW(i, j, k + 1); if (validW[(std::size_t)n]) { sum += w[(std::size_t)n]; ++count; } }
            validWNext[(std::size_t)nid] = (uint8_t)0;
            if (count <= 0) continue;
            w[(std::size_t)nid] = sum / (float)count;
            extrapNextFrontierW[write++] = nid;
        }
        extrapNextFrontierW.resize(write);
        for (int nextId : extrapNextFrontierW) validW[(std::size_t)nextId] = (uint8_t)1;
        extrapFrontierW.swap(extrapNextFrontierW);
        return !extrapFrontierW.empty();
    };

    for (int it = 0; it < std::max(0, params.extrapolationIters); ++it) {
        bool any = false;
        any = propagateU() || any;
        any = propagateV() || any;
        any = propagateW() || any;
        if (!any) break;
    }

    applyBoundary();
}

inline void MACWater3D::gridToParticles() {
    const bool apic = params.useAPIC;
    const float blend = apic ? 0.0f : water3d_internal::clamp01(params.flipBlend);
    const float picWeight = 1.0f - blend;
    const float invDx2 = (dx > 0.0f) ? (1.0f / (dx * dx)) : 0.0f;
    const float apicScale = 3.0f * invDx2;

    auto accumulateAffine = [&](const Particle& p,
                                float ox, float oy, float oz,
                                int sx, int sy, int sz,
                                auto idxFn,
                                const std::vector<float>& field,
                                float& outC0,
                                float& outC1,
                                float& outC2) {
        const float fx = p.x / dx - ox;
        const float fy = p.y / dx - oy;
        const float fz = p.z / dx - oz;

        int i0 = water3d_internal::clampi((int)std::floor(fx), 0, sx - 1);
        int j0 = water3d_internal::clampi((int)std::floor(fy), 0, sy - 1);
        int k0 = water3d_internal::clampi((int)std::floor(fz), 0, sz - 1);

        const int i1 = std::min(i0 + 1, sx - 1);
        const int j1 = std::min(j0 + 1, sy - 1);
        const int k1 = std::min(k0 + 1, sz - 1);

        const float tx = water3d_internal::clampf(fx - (float)i0, 0.0f, 1.0f);
        const float ty = water3d_internal::clampf(fy - (float)j0, 0.0f, 1.0f);
        const float tz = water3d_internal::clampf(fz - (float)k0, 0.0f, 1.0f);

        float sumDx = 0.0f;
        float sumDy = 0.0f;
        float sumDz = 0.0f;

        for (int dk = 0; dk < 2; ++dk) {
            const int kk = (dk == 0) ? k0 : k1;
            const float wz = (dk == 0) ? (1.0f - tz) : tz;
            const float pzFace = (kk + oz) * dx;
            for (int dj = 0; dj < 2; ++dj) {
                const int jj = (dj == 0) ? j0 : j1;
                const float wy = (dj == 0) ? (1.0f - ty) : ty;
                const float pyFace = (jj + oy) * dx;
                for (int di = 0; di < 2; ++di) {
                    const int ii = (di == 0) ? i0 : i1;
                    const float wx = (di == 0) ? (1.0f - tx) : tx;
                    const float pxFace = (ii + ox) * dx;
                    const float wght = wx * wy * wz;
                    const float faceVal = field[(std::size_t)idxFn(ii, jj, kk)];
                    sumDx += wght * faceVal * (pxFace - p.x);
                    sumDy += wght * faceVal * (pyFace - p.y);
                    sumDz += wght * faceVal * (pzFace - p.z);
                }
            }
        }

        outC0 = apicScale * sumDx;
        outC1 = apicScale * sumDy;
        outC2 = apicScale * sumDz;
    };

    auto processParticles = [&](int begin, int end) {
        for (int pi = begin; pi < end; ++pi) {
            Particle& p = particles[(std::size_t)pi];
            float picU, picV, picW;
            velAt(p.x, p.y, p.z, u, v, w, picU, picV, picW);

            if (!apic && blend > 0.0f) {
                const float du = sampleU(uDelta, p.x, p.y, p.z);
                const float dv = sampleV(vDelta, p.x, p.y, p.z);
                const float dw = sampleW(wDelta, p.x, p.y, p.z);
                p.u = picWeight * picU + blend * (p.u + du);
                p.v = picWeight * picV + blend * (p.v + dv);
                p.w = picWeight * picW + blend * (p.w + dw);
            } else {
                p.u = picU;
                p.v = picV;
                p.w = picW;
            }

            if (!apic) {
                p.c00 = p.c01 = p.c02 = 0.0f;
                p.c10 = p.c11 = p.c12 = 0.0f;
                p.c20 = p.c21 = p.c22 = 0.0f;
                continue;
            }

            accumulateAffine(p, 0.0f, 0.5f, 0.5f, nx + 1, ny, nz,
                             [&](int i, int j, int k) { return idxU(i, j, k); },
                             u, p.c00, p.c01, p.c02);
            accumulateAffine(p, 0.5f, 0.0f, 0.5f, nx, ny + 1, nz,
                             [&](int i, int j, int k) { return idxV(i, j, k); },
                             v, p.c10, p.c11, p.c12);
            accumulateAffine(p, 0.5f, 0.5f, 0.0f, nx, ny, nz + 1,
                             [&](int i, int j, int k) { return idxW(i, j, k); },
                             w, p.c20, p.c21, p.c22);
        }
    };

    const bool parallelParticles = (water3dWorkerPool().maxWorkers() > 1) && (particles.size() >= 4096u);
    if (parallelParticles) {
        water3dWorkerPool().parallelFor((int)particles.size(), 256, processParticles);
    } else {
        processParticles(0, (int)particles.size());
    }
}
