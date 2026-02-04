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
}