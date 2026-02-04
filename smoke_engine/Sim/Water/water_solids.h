#pragma once
// Implementation header included by Sim/mac_water_sim.cpp
#include "water_common.h"

inline void MACWater::rebuildSolidsFromUser() {
    const int Nc = nx * ny;
    if ((int)solidUser.size() != Nc) {
        solidUser.assign((size_t)Nc, (uint8_t)0);
    }

    // Start from user solids (synced from the smoke sim)
    solid = solidUser;

    // Apply a thick solid border to keep particles/pressure solve away from the edges.
    const int maxBt = std::max(1, (std::min(nx, ny) / 2) - 1);
    const int bt = water_internal::clampi(borderThickness, 1, maxBt);

    auto setSolidCell = [&](int i, int j) {
        if (i < 0 || i >= nx || j < 0 || j >= ny) return;
        solid[(size_t)idxP(i, j)] = 1;
    };

    // Left / right
    for (int j = 0; j < ny; ++j) {
        for (int t = 0; t < bt; ++t) {
            setSolidCell(t, j);
            setSolidCell(nx - 1 - t, j);
        }
    }

    // Bottom
    for (int i = 0; i < nx; ++i) {
        for (int t = 0; t < bt; ++t) {
            setSolidCell(i, t);
        }
    }

    // Top (only if closed)
    if (!openTop) {
        for (int i = 0; i < nx; ++i) {
            for (int t = 0; t < bt; ++t) {
                setSolidCell(i, ny - 1 - t);
            }
        }
    }

    // Keep render/debug fields consistent with solids.
    if ((int)water.size() == Nc) {
        for (int id = 0; id < Nc; ++id)
            if (solid[(size_t)id]) water[(size_t)id] = 0.0f;
    }
    if ((int)liquid.size() == Nc) {
        for (int id = 0; id < Nc; ++id)
            if (solid[(size_t)id]) liquid[(size_t)id] = 0;
    }
}

inline void MACWater::applyBoundary() {
    // Outer boundary (no-through). For openTop we do NOT overwrite the top v-face
    // (it is part of the divergence constraint).

    // Left / right walls: u = 0
    for (int j = 0; j < ny; ++j) {
        u[(size_t)idxU(0, j)]  = 0.0f;
        u[(size_t)idxU(nx, j)] = 0.0f;
    }

    // Floor: v = 0
    for (int i = 0; i < nx; ++i) {
        v[(size_t)idxV(i, 0)] = 0.0f;
    }

    // Ceiling: closed => v = 0, open => leave as-is
    if (!openTop) {
        for (int i = 0; i < nx; ++i) {
            v[(size_t)idxV(i, ny)] = 0.0f;
        }
    }

    // Optional: kill tangential velocity along the floor for stability.
    for (int i = 0; i <= nx; ++i) {
        u[(size_t)idxU(i, 0)] = 0.0f;
    }

    // Internal solids: no-through for faces adjacent to any solid cell.
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i <= nx; ++i) {
            const bool leftSolid  = (i - 1 >= 0) ? isSolid(i - 1, j) : true;
            const bool rightSolid = (i < nx)     ? isSolid(i, j)     : true;
            if (leftSolid || rightSolid) u[(size_t)idxU(i, j)] = 0.0f;
        }
    }

    for (int j = 0; j <= ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            // bottom handled already
            if (j == 0) continue;

            // top: if open, allow outflow but still prevent passing through a solid cell
            if (j == ny) {
                if (openTop && isSolid(i, ny - 1)) {
                    v[(size_t)idxV(i, j)] = 0.0f;
                }
                continue;
            }

            const bool botSolid = isSolid(i, j - 1);
            const bool topSolid = isSolid(i, j);
            if (botSolid || topSolid) v[(size_t)idxV(i, j)] = 0.0f;
        }
    }
}