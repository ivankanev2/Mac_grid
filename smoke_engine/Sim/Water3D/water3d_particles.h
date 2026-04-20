#pragma once

#include "water3d_common.h"
#include "water3d_cuda_backend.h"
#include "../chunk_worker_pool.h"

#include <algorithm>
#include <cmath>
#include <vector>

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
        p.c00 = p.c01 = p.c02 = 0.0f;
        p.c10 = p.c11 = p.c12 = 0.0f;
        p.c20 = p.c21 = p.c22 = 0.0f;

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
                    const bool lowerLiquid =
                        (j > 0) &&
                        liquid[(std::size_t)idxCell(i, j - 1, k)] &&
                        !solid[(std::size_t)idxCell(i, j - 1, k)];
                    const bool upperLiquid =
                        (j < ny) &&
                        liquid[(std::size_t)idxCell(i, j, k)] &&
                        !solid[(std::size_t)idxCell(i, j, k)];
                    if (!(lowerLiquid || upperLiquid)) continue;

                    if (j == 0) continue;
                    if (j == ny) {
                        if (!params.openTop || !lowerLiquid) continue;
                    } else {
                        const bool botSolid = solid[(std::size_t)idxCell(i, j - 1, k)] != 0;
                        const bool topSolid = solid[(std::size_t)idxCell(i, j, k)] != 0;
                        if (botSolid || topSolid) continue;
                    }

                    v[(std::size_t)idxV(i, j, k)] += dt * params.gravity;
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

    auto advectRange = [&](int begin, int end) {
        for (int pi = begin; pi < end; ++pi) {
            Particle& p = particles[(std::size_t)pi];
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
    };

    const bool parallelParticles = (sharedChunkWorkerPool().maxWorkers() > 1) && (particles.size() >= 4096u);
    if (parallelParticles) {
        sharedChunkWorkerPool().parallelFor((int)particles.size(), 256, advectRange);
    } else {
        advectRange(0, (int)particles.size());
    }
}

inline void MACWater3D::applyDissipation() {
    const float diss = water3d_internal::clamp01(params.waterDissipation);
    if (diss >= 0.999999f) return;

    const std::size_t before = particles.size();
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

    if (desiredMass >= 0.0f) {
        const std::size_t after = particles.size();
        const std::size_t removed = (before > after) ? (before - after) : 0;
        desiredMass = std::max(0.0f, desiredMass - (float)removed);
    }
}

inline void MACWater3D::reseedParticles() {
    if (params.particlesPerCell <= 0) return;

    const int cellCount = nx * ny * nz;
    if (cellCount <= 0) return;

    if ((int)reseedCounts.size() != cellCount) reseedCounts.assign((std::size_t)cellCount, 0);
    if ((int)reseedOccupied.size() != cellCount) reseedOccupied.assign((std::size_t)cellCount, (uint8_t)0);
    if ((int)reseedRegion.size() != cellCount) reseedRegion.assign((std::size_t)cellCount, (uint8_t)0);
    std::fill(reseedCounts.begin(), reseedCounts.end(), 0);
    std::fill(reseedOccupied.begin(), reseedOccupied.end(), (uint8_t)0);
    std::fill(reseedRegion.begin(), reseedRegion.end(), (uint8_t)0);

    for (const Particle& p : particles) {
        int i = water3d_internal::clampi((int)std::floor(p.x / dx), 0, nx - 1);
        int j = water3d_internal::clampi((int)std::floor(p.y / dx), 0, ny - 1);
        int k = water3d_internal::clampi((int)std::floor(p.z / dx), 0, nz - 1);
        const int id = idxCell(i, j, k);
        if (solid[(std::size_t)id]) continue;
        reseedCounts[(std::size_t)id]++;
        reseedOccupied[(std::size_t)id] = (uint8_t)1;
    }

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const int id = idxCell(i, j, k);
                if (solid[(std::size_t)id]) continue;
                if (reseedOccupied[(std::size_t)id]) {
                    reseedRegion[(std::size_t)id] = (uint8_t)1;
                    continue;
                }

                const bool isNear =
                    (i > 0      && reseedOccupied[(std::size_t)idxCell(i - 1, j, k)]) ||
                    (i + 1 < nx && reseedOccupied[(std::size_t)idxCell(i + 1, j, k)]) ||
                    (j > 0      && reseedOccupied[(std::size_t)idxCell(i, j - 1, k)]) ||
                    (j + 1 < ny && reseedOccupied[(std::size_t)idxCell(i, j + 1, k)]) ||
                    (k > 0      && reseedOccupied[(std::size_t)idxCell(i, j, k - 1)]) ||
                    (k + 1 < nz && reseedOccupied[(std::size_t)idxCell(i, j, k + 1)]);
                if (isNear) reseedRegion[(std::size_t)id] = (uint8_t)1;
            }
        }
    }

    const int target = std::max(1, params.particlesPerCell);
    int softMaxParticles = (params.maxParticles > 0)
        ? params.maxParticles
        : std::max((int)particles.size() + 256, target);
    if (desiredMass > 0.0f) {
        const int desiredCap = std::max(target, (int)std::ceil(desiredMass * 1.15f));
        softMaxParticles = std::min(softMaxParticles, desiredCap);
    }
    if ((int)particles.size() >= softMaxParticles) return;

    const int remainingSpawnBudget = std::max(0, softMaxParticles - (int)particles.size());
    const int baseNewCap = std::max(128, cellCount / 64);
    const int maxNewPerStep = std::min(remainingSpawnBudget, std::min(baseNewCap, 2048));
    if (maxNewPerStep <= 0) return;

    int spawned = 0;
    std::vector<Particle> newParticles;
    newParticles.reserve((std::size_t)std::min(maxNewPerStep, cellCount));

    for (int k = 0; k < nz && spawned < maxNewPerStep; ++k) {
        for (int j = 0; j < ny && spawned < maxNewPerStep; ++j) {
            for (int i = 0; i < nx && spawned < maxNewPerStep; ++i) {
                const int id = idxCell(i, j, k);
                if (!reseedRegion[(std::size_t)id] || solid[(std::size_t)id]) continue;

                const int have = reseedCounts[(std::size_t)id];
                if (have >= target) continue;

                int need = target - have;
                while (need-- > 0 && spawned < maxNewPerStep) {
                    if ((int)(particles.size() + newParticles.size()) >= softMaxParticles) {
                        break;
                    }

                    uint32_t seed = (uint32_t)(i + 92821U * j + 68917U * k + 131U * (spawned + 1) + 17U * stepCounter);
                    Particle p;
                    p.x = (i + 0.1f + 0.8f * water3d_internal::rand01(seed)) * dx;
                    p.y = (j + 0.1f + 0.8f * water3d_internal::rand01(seed)) * dx;
                    p.z = (k + 0.1f + 0.8f * water3d_internal::rand01(seed)) * dx;
                    velAt(p.x, p.y, p.z, u, v, w, p.u, p.v, p.w);
                    p.c00 = p.c01 = p.c02 = 0.0f;
                    p.c10 = p.c11 = p.c12 = 0.0f;
                    p.c20 = p.c21 = p.c22 = 0.0f;
                    p.age = 0.0f;
                    newParticles.push_back(p);
                    spawned++;
                }
            }
        }
    }

    if (!newParticles.empty()) {
        particles.insert(particles.end(), newParticles.begin(), newParticles.end());
    }
}

inline void MACWater3D::relaxParticles(int iters, float strength) {
    if (particles.empty() || iters <= 0 || strength <= 0.0f) return;

    const int cellCount = nx * ny * nz;
    if (cellCount <= 0) return;

    if ((int)relaxBucketCounts.size() != cellCount) relaxBucketCounts.assign((std::size_t)cellCount, 0);
    if ((int)relaxBucketOffsets.size() != cellCount + 1) relaxBucketOffsets.assign((std::size_t)cellCount + 1u, 0);
    if ((int)relaxBucketCursor.size() != cellCount) relaxBucketCursor.assign((std::size_t)cellCount, 0);
    if ((int)relaxBucketParticles.size() < (int)particles.size()) relaxBucketParticles.resize(particles.size());

    const float r = 0.35f * dx;
    const float r2 = r * r;

    auto bucketId = [&](float x, float y, float z) {
        int i = water3d_internal::clampi((int)std::floor(x / dx), 0, nx - 1);
        int j = water3d_internal::clampi((int)std::floor(y / dx), 0, ny - 1);
        int k = water3d_internal::clampi((int)std::floor(z / dx), 0, nz - 1);
        return idxCell(i, j, k);
    };

    auto relaxPair = [&](int a, int b) {
        float dxp = particles[(std::size_t)b].x - particles[(std::size_t)a].x;
        float dyp = particles[(std::size_t)b].y - particles[(std::size_t)a].y;
        float dzp = particles[(std::size_t)b].z - particles[(std::size_t)a].z;
        float d2 = dxp * dxp + dyp * dyp + dzp * dzp;
        if (d2 >= r2 || d2 < 1e-12f) return;

        const float d = std::sqrt(d2);
        const float push = (r - d) * strength;
        const float nxp = dxp / d;
        const float nyp = dyp / d;
        const float nzp = dzp / d;

        particles[(std::size_t)a].x -= 0.5f * push * nxp;
        particles[(std::size_t)a].y -= 0.5f * push * nyp;
        particles[(std::size_t)a].z -= 0.5f * push * nzp;
        particles[(std::size_t)b].x += 0.5f * push * nxp;
        particles[(std::size_t)b].y += 0.5f * push * nyp;
        particles[(std::size_t)b].z += 0.5f * push * nzp;
    };

    for (int it = 0; it < iters; ++it) {
        std::fill(relaxBucketCounts.begin(), relaxBucketCounts.end(), 0);
        for (int p = 0; p < (int)particles.size(); ++p) {
            const int id = bucketId(particles[(std::size_t)p].x,
                                    particles[(std::size_t)p].y,
                                    particles[(std::size_t)p].z);
            if (!solid[(std::size_t)id]) {
                ++relaxBucketCounts[(std::size_t)id];
            }
        }

        relaxBucketOffsets[0] = 0;
        for (int id = 0; id < cellCount; ++id) {
            relaxBucketOffsets[(std::size_t)id + 1u] = relaxBucketOffsets[(std::size_t)id] + relaxBucketCounts[(std::size_t)id];
        }
        std::copy(relaxBucketOffsets.begin(), relaxBucketOffsets.begin() + cellCount, relaxBucketCursor.begin());

        for (int p = 0; p < (int)particles.size(); ++p) {
            const int id = bucketId(particles[(std::size_t)p].x,
                                    particles[(std::size_t)p].y,
                                    particles[(std::size_t)p].z);
            if (solid[(std::size_t)id]) continue;
            relaxBucketParticles[(std::size_t)relaxBucketCursor[(std::size_t)id]++] = p;
        }

        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    const int id = idxCell(i, j, k);
                    if (solid[(std::size_t)id]) continue;

                    const int aBegin = relaxBucketOffsets[(std::size_t)id];
                    const int aEnd = relaxBucketOffsets[(std::size_t)id + 1u];
                    if (aBegin == aEnd) continue;

                    for (int dk = -1; dk <= 1; ++dk) {
                        for (int dj = -1; dj <= 1; ++dj) {
                            for (int di = -1; di <= 1; ++di) {
                                if (dk < 0) continue;
                                if (dk == 0 && dj < 0) continue;
                                if (dk == 0 && dj == 0 && di < 0) continue;

                                const int ii = i + di;
                                const int jj = j + dj;
                                const int kk = k + dk;
                                if (ii < 0 || jj < 0 || kk < 0 || ii >= nx || jj >= ny || kk >= nz) continue;

                                const int nid = idxCell(ii, jj, kk);
                                if (solid[(std::size_t)nid]) continue;
                                const int bBegin = relaxBucketOffsets[(std::size_t)nid];
                                const int bEnd = relaxBucketOffsets[(std::size_t)nid + 1u];
                                if (bBegin == bEnd) continue;

                                if (nid == id) {
                                    for (int ai = aBegin; ai < aEnd; ++ai) {
                                        const int a = relaxBucketParticles[(std::size_t)ai];
                                        for (int bi = ai + 1; bi < aEnd; ++bi) {
                                            const int b = relaxBucketParticles[(std::size_t)bi];
                                            relaxPair(a, b);
                                        }
                                    }
                                } else {
                                    for (int ai = aBegin; ai < aEnd; ++ai) {
                                        const int a = relaxBucketParticles[(std::size_t)ai];
                                        for (int bi = bBegin; bi < bEnd; ++bi) {
                                            const int b = relaxBucketParticles[(std::size_t)bi];
                                            relaxPair(a, b);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        enforceParticleBounds();
        removeParticlesInSolids();
    }
}
