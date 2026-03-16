#pragma once

#include "water3d_common.h"
#include "water3d_cuda_backend.h"

#include <algorithm>
#include <cmath>

inline void MACWater3D::removeParticlesInSolids() {
    if (particles.empty()) return;

    const int maxRad = 2;

    std::size_t write = 0;
    for (std::size_t read = 0; read < particles.size(); ++read) {
        Particle p = particles[read];

        int i = water3d_internal::clampi((int)std::floor(p.x / dx), 0, nx - 1);
        int j = water3d_internal::clampi((int)std::floor(p.y / dx), 0, ny - 1);
        int k = water3d_internal::clampi((int)std::floor(p.z / dx), 0, nz - 1);

        if (!solid[(std::size_t)idxCell(i, j, k)]) {
            particles[write++] = p;
            continue;
        }

        bool found = false;
        int bestI = i;
        int bestJ = j;
        int bestK = k;
        float bestD2 = 1e30f;

        for (int rad = 1; rad <= maxRad && !found; ++rad) {
            for (int dk = -rad; dk <= rad; ++dk) {
                for (int dj = -rad; dj <= rad; ++dj) {
                    for (int di = -rad; di <= rad; ++di) {
                        const int ii = water3d_internal::clampi(i + di, 0, nx - 1);
                        const int jj = water3d_internal::clampi(j + dj, 0, ny - 1);
                        const int kk = water3d_internal::clampi(k + dk, 0, nz - 1);
                        const int id = idxCell(ii, jj, kk);
                        if (solid[(std::size_t)id]) continue;

                        const float dx0 = (float)(ii - i);
                        const float dy0 = (float)(jj - j);
                        const float dz0 = (float)(kk - k);
                        const float d2 = dx0 * dx0 + dy0 * dy0 + dz0 * dz0;
                        if (d2 < bestD2) {
                            bestD2 = d2;
                            bestI = ii;
                            bestJ = jj;
                            bestK = kk;
                            found = true;
                        }
                    }
                }
            }
        }

        if (!found) continue;

        p.x = (bestI + 0.5f) * dx;
        p.y = (bestJ + 0.5f) * dx;
        p.z = (bestK + 0.5f) * dx;
        p.u = 0.0f;
        p.v = 0.0f;
        p.w = 0.0f;

        particles[write++] = p;
    }

    particles.resize(write);
}

inline void MACWater3D::enforceParticleBounds() {
    const int maxBt = std::max(1, (std::min({nx, ny, nz}) / 2) - 1);
    const int bt = water3d_internal::clampi(params.borderThickness, 1, maxBt);

    const float minX = (bt + 0.5f) * dx;
    const float maxX = (nx - bt - 0.5f) * dx;
    const float minY = (bt + 0.5f) * dx;
    const float maxY = params.openTop ? (ny - 0.5f) * dx : (ny - bt - 0.5f) * dx;
    const float minZ = (bt + 0.5f) * dx;
    const float maxZ = (nz - bt - 0.5f) * dx;

    for (Particle& p : particles) {
        if (p.x < minX) { p.x = minX; if (p.u < 0.0f) p.u = 0.0f; }
        if (p.x > maxX) { p.x = maxX; if (p.u > 0.0f) p.u = 0.0f; }

        if (p.y < minY) { p.y = minY; if (p.v < 0.0f) p.v = 0.0f; }
        if (p.y > maxY) { p.y = maxY; if (p.v > 0.0f) p.v = 0.0f; }

        if (p.z < minZ) { p.z = minZ; if (p.w < 0.0f) p.w = 0.0f; }
        if (p.z > maxZ) { p.z = maxZ; if (p.w > 0.0f) p.w = 0.0f; }
    }
}

inline void MACWater3D::applyExternalForces() {
    if (params.gravity != 0.0f) {
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j <= ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    const bool botLiquid = (j - 1 >= 0)
                        ? (liquid[(std::size_t)idxCell(i, j - 1, k)] != 0)
                        : false;
                    const bool topLiquid = (j < ny)
                        ? (liquid[(std::size_t)idxCell(i, j, k)] != 0)
                        : false;
                    if (botLiquid || topLiquid) {
                        v[(std::size_t)idxV(i, j, k)] += dt * params.gravity;
                    }
                }
            }
        }
    }

    if (params.velDamping > 0.0f) {
        const float damp = std::exp(-params.velDamping * dt);
        for (float& value : u) value *= damp;
        for (float& value : v) value *= damp;
        for (float& value : w) value *= damp;
    }
}

inline void MACWater3D::advectParticles() {
    if (particles.empty()) return;

    const float domainX = nx * dx;
    const float domainY = ny * dx;
    const float domainZ = nz * dx;

    for (Particle& p : particles) {
        float u1, v1, w1;
        velAt(p.x, p.y, p.z, u, v, w, u1, v1, w1);

        const float midX = water3d_internal::clampf(p.x + 0.5f * dt * u1, 0.0f, domainX);
        const float midY = water3d_internal::clampf(p.y + 0.5f * dt * v1, 0.0f, domainY);
        const float midZ = water3d_internal::clampf(p.z + 0.5f * dt * w1, 0.0f, domainZ);

        float u2, v2, w2;
        velAt(midX, midY, midZ, u, v, w, u2, v2, w2);

        p.x += dt * u2;
        p.y += dt * v2;
        p.z += dt * w2;
        p.u = u2;
        p.v = v2;
        p.w = w2;
        p.age += dt;
    }
}

inline void MACWater3D::applyDissipation() {
    const float diss = water3d_internal::clamp01(params.waterDissipation);
    if (diss >= 0.999999f) return;

    const float dtRef = 0.02f;
    const float keepProb = std::pow(diss, dt / std::max(1e-6f, dtRef));

    std::size_t write = 0;
    uint32_t seed = (uint32_t)(stepCounter * 9781 + 17);
    for (std::size_t read = 0; read < particles.size(); ++read) {
        if (water3d_internal::rand01(seed) <= keepProb) {
            particles[write++] = particles[read];
        }
    }
    particles.resize(write);
}

inline void MACWater3D::reseedParticles() {
    if (params.particlesPerCell <= 0) return;

    const int cellCount = nx * ny * nz;
    std::vector<int> counts((std::size_t)cellCount, 0);
    std::vector<uint8_t> occupied((std::size_t)cellCount, (uint8_t)0);

    for (const Particle& p : particles) {
        int i = water3d_internal::clampi((int)std::floor(p.x / dx), 0, nx - 1);
        int j = water3d_internal::clampi((int)std::floor(p.y / dx), 0, ny - 1);
        int k = water3d_internal::clampi((int)std::floor(p.z / dx), 0, nz - 1);
        const int id = idxCell(i, j, k);
        if (solid[(std::size_t)id]) continue;
        counts[(std::size_t)id]++;
        occupied[(std::size_t)id] = (uint8_t)1;
    }

    std::vector<uint8_t> region = occupied;
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const int id = idxCell(i, j, k);
                if (solid[(std::size_t)id] || region[(std::size_t)id]) continue;

                const bool near =
                    (i > 0 && occupied[(std::size_t)idxCell(i - 1, j, k)]) ||
                    (i + 1 < nx && occupied[(std::size_t)idxCell(i + 1, j, k)]) ||
                    (j > 0 && occupied[(std::size_t)idxCell(i, j - 1, k)]) ||
                    (j + 1 < ny && occupied[(std::size_t)idxCell(i, j + 1, k)]) ||
                    (k > 0 && occupied[(std::size_t)idxCell(i, j, k - 1)]) ||
                    (k + 1 < nz && occupied[(std::size_t)idxCell(i, j, k + 1)]);

                if (near) region[(std::size_t)id] = (uint8_t)1;
            }
        }
    }

    const int maxNewPerStep = std::max(4096, cellCount / 8);
    int spawned = 0;
    std::vector<Particle> newParticles;
    newParticles.reserve((std::size_t)std::min(maxNewPerStep, cellCount));

    for (int k = 0; k < nz && spawned < maxNewPerStep; ++k) {
        for (int j = 0; j < ny && spawned < maxNewPerStep; ++j) {
            for (int i = 0; i < nx && spawned < maxNewPerStep; ++i) {
                const int id = idxCell(i, j, k);
                if (!region[(std::size_t)id] || solid[(std::size_t)id]) continue;

                const int have = counts[(std::size_t)id];
                if (have >= params.particlesPerCell) continue;

                const int need = params.particlesPerCell - have;
                for (int n = 0; n < need && spawned < maxNewPerStep; ++n) {
                    if (params.maxParticles > 0 &&
                        (int)(particles.size() + newParticles.size()) >= params.maxParticles) {
                        break;
                    }

                    uint32_t seed =
                        (uint32_t)(i + 92821U * j + 68917U * k + 131U * (n + 1) + 17U * stepCounter + 1U);
                    Particle p;
                    p.x = (i + 0.1f + 0.8f * water3d_internal::rand01(seed)) * dx;
                    p.y = (j + 0.1f + 0.8f * water3d_internal::rand01(seed)) * dx;
                    p.z = (k + 0.1f + 0.8f * water3d_internal::rand01(seed)) * dx;
                    velAt(p.x, p.y, p.z, u, v, w, p.u, p.v, p.w);
                    newParticles.push_back(p);
                    spawned++;
                }
            }
        }
    }

    particles.insert(particles.end(), newParticles.begin(), newParticles.end());
}
