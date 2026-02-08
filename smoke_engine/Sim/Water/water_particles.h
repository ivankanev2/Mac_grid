#pragma once
// Implementation header included by Sim/mac_water_sim.cpp
#include "water_common.h"

#include <limits>
#include <random>

inline float MACWater::maxParticleSpeed() const {
    float m2 = 0.0f;
    for (const Particle& p : particles) {
        const float s2 = p.u * p.u + p.v * p.v;
        if (!std::isfinite(s2)) return std::numeric_limits<float>::infinity();
        m2 = std::max(m2, s2);
    }
    return std::sqrt(m2);
}

inline void MACWater::removeParticlesInSolids() {
    if (particles.empty()) return;

    const int Nc = nx * ny;
    if ((int)solid.size() != Nc) return;

    const int maxRad = 3;

    size_t write = 0;
    for (size_t read = 0; read < particles.size(); ++read) {
        Particle p = particles[read];

        int i, j;
        worldToCell(p.x, p.y, i, j);
        if (!solid[(size_t)idxP(i, j)]) {
            particles[write++] = p;
            continue;
        }

        bool found = false;
        int bestI = i, bestJ = j;
        float bestD2 = 1e30f;

        for (int rad = 1; rad <= maxRad && !found; ++rad) {
            for (int dj = -rad; dj <= rad; ++dj) {
                for (int di = -rad; di <= rad; ++di) {
                    const int ii = water_internal::clampi(i + di, 0, nx - 1);
                    const int jj = water_internal::clampi(j + dj, 0, ny - 1);
                    const int id = idxP(ii, jj);
                    if (solid[(size_t)id]) continue;

                    const float dx0 = (float)(ii - i);
                    const float dy0 = (float)(jj - j);
                    const float d2 = dx0 * dx0 + dy0 * dy0;
                    if (d2 < bestD2) {
                        bestD2 = d2;
                        bestI = ii;
                        bestJ = jj;
                        found = true;
                    }
                }
            }
        }

        if (!found) {
            // Give up; drop the particle (should be rare).
            continue;
        }

        // Snap to the target cell center and reset velocity.
        p.x = (bestI + 0.5f) * dx;
        p.y = (bestJ + 0.5f) * dx;
        p.u = 0.0f;
        p.v = 0.0f;

        particles[write++] = p;
    }

    particles.resize(write);
}

inline void MACWater::enforceParticleBounds() {
    const int maxBt = std::max(1, (std::min(nx, ny) / 2) - 1);
    const int bt = water_internal::clampi(borderThickness, 1, maxBt);

    const float minX = (bt + 0.5f) * dx;
    const float maxX = (nx - bt - 0.5f) * dx;

    const float minY = (bt + 0.5f) * dx;
    const float maxYClosed = (ny - bt - 0.5f) * dx;
    const float maxYOpen   = (ny - 0.5f) * dx;

    for (Particle& p : particles) {
        if (p.x < minX) { p.x = minX; if (p.u < 0.0f) p.u = 0.0f; }
        if (p.x > maxX) { p.x = maxX; if (p.u > 0.0f) p.u = 0.0f; }

        if (p.y < minY) { p.y = minY; if (p.v < 0.0f) p.v = 0.0f; }

        if (openTop) {
            if (p.y > maxYOpen) {
                p.y = maxYOpen;
                if (p.v > 0.0f) p.v = 0.0f;
            }
        } else {
            if (p.y > maxYClosed) {
                p.y = maxYClosed;
                if (p.v > 0.0f) p.v = 0.0f;
            }
        }
    }
}

inline void MACWater::advectParticles() {
    if (particles.empty()) return;

    const float dtLocal = dt;
    if (dtLocal <= 0.0f) return;

    const float domainX = nx * dx;
    const float domainY = ny * dx;

    for (Particle& p : particles) {
        float u1, v1;
        velAt(p.x, p.y, u, v, u1, v1);

        const float midX = water_internal::clampf(p.x + 0.5f * dtLocal * u1, 0.0f, domainX);
        const float midY = water_internal::clampf(p.y + 0.5f * dtLocal * v1, 0.0f, domainY);

        float u2, v2;
        velAt(midX, midY, u, v, u2, v2);

        p.x = p.x + dtLocal * u2;
        p.y = p.y + dtLocal * v2;
        p.u = u2;
        p.v = v2;
        p.age += dtLocal;
    }
}

inline void MACWater::applyDissipation() {
    const float diss = water_internal::clamp01(waterDissipation);
    if (diss >= 0.999999f) return;

    const size_t before = particles.size();

    // Interpret dissipation similarly to smoke scalar dissipation: exponential in time.
    const float dtRef = 0.02f; // match default UI dtMax
    const float keepProb = std::pow(diss, dt / std::max(1e-6f, dtRef));

    size_t write = 0;
    for (size_t read = 0; read < particles.size(); ++read) {
        if (water_internal::rand01() <= keepProb) {
            particles[write++] = particles[read];
        }
    }
    particles.resize(write);

    // If we intentionally removed particles, update the target volume too.
    if (desiredMass >= 0.0f) {
        const size_t after = particles.size();
        const size_t removed = (before > after) ? (before - after) : 0;
        desiredMass = std::max(0.0f, desiredMass - (float)removed);
    }
}

inline uint32_t water_hash_u32(uint32_t x) {
    // simple integer hash (deterministic)
    x ^= x >> 16;
    x *= 0x7feb352d;
    x ^= x >> 15;
    x *= 0x846ca68b;
    x ^= x >> 16;
    return x;
}

inline float water_rand01(uint32_t& s) {
    s = water_hash_u32(s);
    // 24-bit mantissa -> [0,1)
    return (float)(s & 0x00FFFFFFu) / (float)0x01000000u;
}

// Keep particle coverage stable: ensure each liquid cell has >= particlesPerCell particles.
// This prevents the liquid mask from collapsing ("water shrinking") as particles clump.
inline void MACWater::reseedParticles() {
    if (particlesPerCell <= 0) return;

    const int Nc = nx * ny;
    if (Nc <= 0) return;

    // Count particles per cell + mark occupied cells from particles ONLY.
    std::vector<int> cnt((size_t)Nc, 0);
    std::vector<uint8_t> occ((size_t)Nc, 0);

    for (const auto& p : particles) {
        int i = (int)std::floor(p.x / dx);
        int j = (int)std::floor(p.y / dx);
        i = water_internal::clampi(i, 0, nx - 1);
        j = water_internal::clampi(j, 0, ny - 1);
        const int id = idxP(i, j);
        if (solid[(size_t)id]) continue;
        cnt[(size_t)id]++;
        occ[(size_t)id] = 1;
    }

    // Build a tight region: occupied cells + 1-ring neighbors (still tight, prevents nuking).
    std::vector<uint8_t> region((size_t)Nc, 0);
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int id = idxP(i, j);
            if (solid[(size_t)id]) continue;

            if (occ[(size_t)id]) { region[(size_t)id] = 1; continue; }

            const bool near =
                (i > 0     && occ[(size_t)idxP(i - 1, j)]) ||
                (i < nx-1  && occ[(size_t)idxP(i + 1, j)]) ||
                (j > 0     && occ[(size_t)idxP(i, j - 1)]) ||
                (j < ny-1  && occ[(size_t)idxP(i, j + 1)]);

            if (near) region[(size_t)id] = 1;
        }
    }

    // Deterministic RNG helpers
    auto hash_u32 = [](uint32_t x) {
        x ^= x >> 16;
        x *= 0x7feb352d;
        x ^= x >> 15;
        x *= 0x846ca68b;
        x ^= x >> 16;
        return x;
    };
    auto rand01 = [&](uint32_t& s) {
        s = hash_u32(s);
        return (float)(s & 0x00FFFFFFu) / (float)0x01000000u;
    };

    // Safety cap: never spawn insane amounts in one frame.
    const int target = particlesPerCell;
    const int maxNewPerFrame = std::max(2000, (nx * ny) / 2);

    int spawned = 0;
    std::vector<Particle> newParts;
    newParts.reserve((size_t)std::min(maxNewPerFrame, nx * ny));

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int id = idxP(i, j);
            if (!region[(size_t)id]) continue;
            if (solid[(size_t)id]) continue;

            const int have = cnt[(size_t)id];
            if (have >= target) continue;

            int need = target - have;
            while (need-- > 0) {
                if (spawned >= maxNewPerFrame) break;

                uint32_t seed = (uint32_t)(i + 73856093u * (uint32_t)j) ^ (uint32_t)(stepCounter * 19349663);

                const float rx = rand01(seed);
                const float ry = rand01(seed);

                Particle pnew;
                pnew.x = (i + rx) * dx;
                pnew.y = (j + ry) * dx;

                // sample grid vel so we don't inject energy
                float uu = 0.0f, vv = 0.0f;
                velAt(pnew.x, pnew.y, u, v, uu, vv);
                pnew.u = uu;
                pnew.v = vv;

                pnew.c00 = pnew.c01 = pnew.c10 = pnew.c11 = 0.0f;
                pnew.age = 0.0f;

                newParts.push_back(pnew);
                spawned++;
            }
            if (spawned >= maxNewPerFrame) break;
        }
        if (spawned >= maxNewPerFrame) break;
    }

    if (!newParts.empty()) {
        particles.insert(particles.end(), newParts.begin(), newParts.end());
    }
}

inline void MACWater::relaxParticles(int iters, float strength) {
    if (particles.empty() || iters <= 0) return;

    const float r = 0.35f * dx;     // interaction radius
    const float r2 = r * r;

    // simple grid hash: particles per cell index
    std::vector<std::vector<int>> buckets((size_t)(nx * ny));
    buckets.assign((size_t)(nx * ny), {});

    auto cellId = [&](float x, float y) {
        int i = (int)std::floor(x / dx);
        int j = (int)std::floor(y / dx);
        i = water_internal::clampi(i, 0, nx - 1);
        j = water_internal::clampi(j, 0, ny - 1);
        return idxP(i, j);
    };

    for (int it = 0; it < iters; ++it) {
        for (auto& b : buckets) b.clear();
        for (int k = 0; k < (int)particles.size(); ++k) {
            int id = cellId(particles[k].x, particles[k].y);
            if (!solid[(size_t)id]) buckets[(size_t)id].push_back(k);
        }

        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const int id = idxP(i, j);
                if (solid[(size_t)id]) continue;

                // check this cell + neighbors
                for (int dj = -1; dj <= 1; ++dj) {
                    for (int di = -1; di <= 1; ++di) {
                        int ii = i + di, jj = j + dj;
                        if (ii < 0 || jj < 0 || ii >= nx || jj >= ny) continue;
                        const int nid = idxP(ii, jj);
                        if (solid[(size_t)nid]) continue;

                        const auto& A = buckets[(size_t)id];
                        const auto& B = buckets[(size_t)nid];

                        for (int a : A) for (int b : B) {
                            if (a >= b) continue;

                            float dxp = particles[b].x - particles[a].x;
                            float dyp = particles[b].y - particles[a].y;
                            float d2  = dxp*dxp + dyp*dyp;
                            if (d2 >= r2 || d2 < 1e-12f) continue;

                            float d = std::sqrt(d2);
                            float push = (r - d) * strength;

                            float nxp = dxp / d;
                            float nyp = dyp / d;

                            particles[a].x -= 0.5f * push * nxp;
                            particles[a].y -= 0.5f * push * nyp;
                            particles[b].x += 0.5f * push * nxp;
                            particles[b].y += 0.5f * push * nyp;
                        }
                    }
                }
            }
        }

        enforceParticleBounds();
        removeParticlesInSolids();
    }
}