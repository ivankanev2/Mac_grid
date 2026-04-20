#pragma once

#include "water3d_common.h"
#include "water3d_cuda_backend.h"

#include <algorithm>

inline float water3dFaceOpenClamped(const std::vector<float>& faceOpen, int idx) {
    if (idx < 0 || (std::size_t)idx >= faceOpen.size()) return 1.0f;
    return std::max(0.0f, std::min(1.0f, faceOpen[(std::size_t)idx]));
}

inline void MACWater3D::rebuildBorderSolids() {
    const int cellCount = nx * ny * nz;
    if ((int)solidUser.size() != cellCount) {
        solidUser.assign((std::size_t)cellCount, (uint8_t)0);
    }

    solid = solidUser;

    const int maxBt = std::max(1, (std::min({nx, ny, nz}) / 2) - 1);
    const int bt = water3d_internal::clampi(params.borderThickness, 1, maxBt);

    auto setSolid = [&](int i, int j, int k) {
        if (i < 0 || i >= nx || j < 0 || j >= ny || k < 0 || k >= nz) return;
        solid[(std::size_t)idxCell(i, j, k)] = (uint8_t)1;
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

    for (int id = 0; id < cellCount; ++id) {
        if (!solid[(std::size_t)id]) continue;
        if ((int)liquid.size() == cellCount) liquid[(std::size_t)id] = 0;
        if ((int)water.size() == cellCount) water[(std::size_t)id] = 0.0f;
    }

    diffusionStencilDirty = true;
    topologyDirty = false;
}

inline void MACWater3D::applyBoundary() {
    constexpr float kClosed = 1.0e-4f;

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
                const int faceId = idxU(i, j, k);
                const float open = water3dFaceOpenClamped(uFaceOpen, faceId);
                const bool leftSolid = (i - 1 >= 0) ? isSolidCell(i - 1, j, k) : true;
                const bool rightSolid = (i < nx) ? isSolidCell(i, j, k) : true;
                if (leftSolid || rightSolid || open <= kClosed) {
                    u[(std::size_t)faceId] = 0.0f;
                } else {
                    u[(std::size_t)faceId] *= open;
                }
            }
        }
    }

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const int faceId = idxV(i, j, k);
                const float open = water3dFaceOpenClamped(vFaceOpen, faceId);
                if (j == 0) continue;
                if (j == ny) {
                    if (params.openTop && isSolidCell(i, ny - 1, k)) {
                        v[(std::size_t)faceId] = 0.0f;
                    }
                    continue;
                }

                const bool botSolid = isSolidCell(i, j - 1, k);
                const bool topSolid = isSolidCell(i, j, k);
                if (botSolid || topSolid || open <= kClosed) {
                    v[(std::size_t)faceId] = 0.0f;
                } else {
                    v[(std::size_t)faceId] *= open;
                }
            }
        }
    }

    for (int k = 0; k <= nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const int faceId = idxW(i, j, k);
                const float open = water3dFaceOpenClamped(wFaceOpen, faceId);
                const bool backSolid = (k - 1 >= 0) ? isSolidCell(i, j, k - 1) : true;
                const bool frontSolid = (k < nz) ? isSolidCell(i, j, k) : true;
                if (backSolid || frontSolid || open <= kClosed) {
                    w[(std::size_t)faceId] = 0.0f;
                } else {
                    w[(std::size_t)faceId] *= open;
                }
            }
        }
    }
}

