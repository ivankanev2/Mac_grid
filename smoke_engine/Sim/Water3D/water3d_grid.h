#pragma once

#include "water3d_common.h"

#include <algorithm>
#include <cmath>

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

                    const bool near =
                        (i > 0 && liquid[(std::size_t)idxCell(i - 1, j, k)]) ||
                        (i + 1 < nx && liquid[(std::size_t)idxCell(i + 1, j, k)]) ||
                        (j > 0 && liquid[(std::size_t)idxCell(i, j - 1, k)]) ||
                        (j + 1 < ny && liquid[(std::size_t)idxCell(i, j + 1, k)]) ||
                        (k > 0 && liquid[(std::size_t)idxCell(i, j, k - 1)]) ||
                        (k + 1 < nz && liquid[(std::size_t)idxCell(i, j, k + 1)]);
                    if (near) next[(std::size_t)id] = (uint8_t)1;
                }
            }
        }
        for (int id = 0; id < cellCount; ++id) {
            if (solid[(std::size_t)id]) next[(std::size_t)id] = 0;
        }
        liquid.swap(next);
    }
}

inline void MACWater3D::diffuseVelocityImplicit() {
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

inline void MACWater3D::extrapolateVelocity() {
    validU.assign(u.size(), (uint8_t)0);
    validV.assign(v.size(), (uint8_t)0);
    validW.assign(w.size(), (uint8_t)0);

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i <= nx; ++i) {
                const bool leftSolid = (i - 1 >= 0) ? isSolidCell(i - 1, j, k) : true;
                const bool rightSolid = (i < nx) ? isSolidCell(i, j, k) : true;
                if (leftSolid || rightSolid) {
                    u[(std::size_t)idxU(i, j, k)] = 0.0f;
                    continue;
                }
                if (uWeight[(std::size_t)idxU(i, j, k)] > 1e-6f) {
                    validU[(std::size_t)idxU(i, j, k)] = (uint8_t)1;
                }
            }
        }
    }

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const bool botSolid = (j - 1 >= 0) ? isSolidCell(i, j - 1, k) : true;
                const bool topSolid = (j < ny) ? isSolidCell(i, j, k) : !params.openTop;
                if (botSolid || topSolid) {
                    v[(std::size_t)idxV(i, j, k)] = 0.0f;
                    continue;
                }
                if (vWeight[(std::size_t)idxV(i, j, k)] > 1e-6f) {
                    validV[(std::size_t)idxV(i, j, k)] = (uint8_t)1;
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
                    continue;
                }
                if (wWeight[(std::size_t)idxW(i, j, k)] > 1e-6f) {
                    validW[(std::size_t)idxW(i, j, k)] = (uint8_t)1;
                }
            }
        }
    }

    for (int it = 0; it < std::max(0, params.extrapolationIters); ++it) {
        std::vector<float> nextU = u;
        std::vector<uint8_t> nextValidU = validU;
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i <= nx; ++i) {
                    const int id = idxU(i, j, k);
                    if (nextValidU[(std::size_t)id]) continue;
                    const bool leftSolid = (i - 1 >= 0) ? isSolidCell(i - 1, j, k) : true;
                    const bool rightSolid = (i < nx) ? isSolidCell(i, j, k) : true;
                    if (leftSolid || rightSolid) continue;

                    float sum = 0.0f;
                    int count = 0;
                    if (i > 0 && validU[(std::size_t)idxU(i - 1, j, k)]) { sum += u[(std::size_t)idxU(i - 1, j, k)]; count++; }
                    if (i < nx && validU[(std::size_t)idxU(i + 1, j, k)]) { sum += u[(std::size_t)idxU(i + 1, j, k)]; count++; }
                    if (j > 0 && validU[(std::size_t)idxU(i, j - 1, k)]) { sum += u[(std::size_t)idxU(i, j - 1, k)]; count++; }
                    if (j + 1 < ny && validU[(std::size_t)idxU(i, j + 1, k)]) { sum += u[(std::size_t)idxU(i, j + 1, k)]; count++; }
                    if (k > 0 && validU[(std::size_t)idxU(i, j, k - 1)]) { sum += u[(std::size_t)idxU(i, j, k - 1)]; count++; }
                    if (k + 1 < nz && validU[(std::size_t)idxU(i, j, k + 1)]) { sum += u[(std::size_t)idxU(i, j, k + 1)]; count++; }
                    if (count > 0) {
                        nextU[(std::size_t)id] = sum / (float)count;
                        nextValidU[(std::size_t)id] = (uint8_t)1;
                    }
                }
            }
        }
        u.swap(nextU);
        validU.swap(nextValidU);

        std::vector<float> nextV = v;
        std::vector<uint8_t> nextValidV = validV;
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j <= ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    const int id = idxV(i, j, k);
                    if (nextValidV[(std::size_t)id]) continue;
                    const bool botSolid = (j - 1 >= 0) ? isSolidCell(i, j - 1, k) : true;
                    const bool topSolid = (j < ny) ? isSolidCell(i, j, k) : !params.openTop;
                    if (botSolid || topSolid) continue;

                    float sum = 0.0f;
                    int count = 0;
                    if (i > 0 && validV[(std::size_t)idxV(i - 1, j, k)]) { sum += v[(std::size_t)idxV(i - 1, j, k)]; count++; }
                    if (i + 1 < nx && validV[(std::size_t)idxV(i + 1, j, k)]) { sum += v[(std::size_t)idxV(i + 1, j, k)]; count++; }
                    if (j > 0 && validV[(std::size_t)idxV(i, j - 1, k)]) { sum += v[(std::size_t)idxV(i, j - 1, k)]; count++; }
                    if (j < ny && validV[(std::size_t)idxV(i, j + 1, k)]) { sum += v[(std::size_t)idxV(i, j + 1, k)]; count++; }
                    if (k > 0 && validV[(std::size_t)idxV(i, j, k - 1)]) { sum += v[(std::size_t)idxV(i, j, k - 1)]; count++; }
                    if (k + 1 < nz && validV[(std::size_t)idxV(i, j, k + 1)]) { sum += v[(std::size_t)idxV(i, j, k + 1)]; count++; }
                    if (count > 0) {
                        nextV[(std::size_t)id] = sum / (float)count;
                        nextValidV[(std::size_t)id] = (uint8_t)1;
                    }
                }
            }
        }
        v.swap(nextV);
        validV.swap(nextValidV);

        std::vector<float> nextW = w;
        std::vector<uint8_t> nextValidW = validW;
        for (int k = 0; k <= nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    const int id = idxW(i, j, k);
                    if (nextValidW[(std::size_t)id]) continue;
                    const bool backSolid = (k - 1 >= 0) ? isSolidCell(i, j, k - 1) : true;
                    const bool frontSolid = (k < nz) ? isSolidCell(i, j, k) : true;
                    if (backSolid || frontSolid) continue;

                    float sum = 0.0f;
                    int count = 0;
                    if (i > 0 && validW[(std::size_t)idxW(i - 1, j, k)]) { sum += w[(std::size_t)idxW(i - 1, j, k)]; count++; }
                    if (i + 1 < nx && validW[(std::size_t)idxW(i + 1, j, k)]) { sum += w[(std::size_t)idxW(i + 1, j, k)]; count++; }
                    if (j > 0 && validW[(std::size_t)idxW(i, j - 1, k)]) { sum += w[(std::size_t)idxW(i, j - 1, k)]; count++; }
                    if (j + 1 < ny && validW[(std::size_t)idxW(i, j + 1, k)]) { sum += w[(std::size_t)idxW(i, j + 1, k)]; count++; }
                    if (k > 0 && validW[(std::size_t)idxW(i, j, k - 1)]) { sum += w[(std::size_t)idxW(i, j, k - 1)]; count++; }
                    if (k < nz && validW[(std::size_t)idxW(i, j, k + 1)]) { sum += w[(std::size_t)idxW(i, j, k + 1)]; count++; }
                    if (count > 0) {
                        nextW[(std::size_t)id] = sum / (float)count;
                        nextValidW[(std::size_t)id] = (uint8_t)1;
                    }
                }
            }
        }
        w.swap(nextW);
        validW.swap(nextValidW);
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

    for (Particle& p : particles) {
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
}
