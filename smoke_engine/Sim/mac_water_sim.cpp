#include "mac_water_sim.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>

namespace {
static float dotVec(const std::vector<float>& a, const std::vector<float>& b) {
    double s = 0.0;
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) s += (double)a[i] * (double)b[i];
    return (float)s;
}

static float clamp01(float x) {
    return std::max(0.0f, std::min(1.0f, x));
}

static float capOrInf(float cap) {
    return cap > 0.0f ? cap : std::numeric_limits<float>::infinity();
}

static std::mt19937& globalRng() {
    static std::mt19937 rng(1337u);
    return rng;
}

static bool canSpawnMore(int maxParticles, size_t current) {
    return maxParticles <= 0 || current < (size_t)maxParticles;
}

static int maxParticleLimit(int maxParticles) {
    if (maxParticles > 0) return maxParticles;
    return std::numeric_limits<int>::max() / 4;
}

static inline int mgIdx(int i, int j, int nx) { return i + nx * j; }
}

MACWater::MACWater(int NX, int NY, float DX, float DT)
    : MACGridCore(NX, NY, DX, DT)
{
    reset();
}

void MACWater::reset() {
    resetCore();

    const size_t Nu = (size_t)(nx + 1) * (size_t)ny;
    const size_t Nv = (size_t)nx * (size_t)(ny + 1);
    const size_t Nc = (size_t)nx * (size_t)ny;

    water.assign(Nc, 0.0f);
    water0.assign(Nc, 0.0f);
    waterTarget.assign(Nc, 0.0f);
    targetMass = 0.0f;
    liquid.assign(Nc, 0);
    liquidPrev.assign(Nc, 0);
    particles.clear();

    uWeight.assign(Nu, 0.0f);
    vWeight.assign(Nv, 0.0f);
    uPrev.assign(Nu, 0.0f);
    vPrev.assign(Nv, 0.0f);
    uDelta.assign(Nu, 0.0f);
    vDelta.assign(Nv, 0.0f);

    lapDiag.assign(Nc, 0.0f);
    lapDiagInv.assign(Nc, 0.0f);
    lapL.assign(Nc, -1);
    lapR.assign(Nc, -1);
    lapB.assign(Nc, -1);
    lapT.assign(Nc, -1);
    pcg_r.assign(Nc, 0.0f);
    pcg_z.assign(Nc, 0.0f);
    pcg_d.assign(Nc, 0.0f);
    pcg_q.assign(Nc, 0.0f);
    pcg_Ap.assign(Nc, 0.0f);

    stepCounter = 0;
    mgDirty = true;
    mgHasDirichlet = false;
    mgOpenTop = openTop;
    mgLevels.clear();
    enforceBorderSolids();
}

float MACWater::maxParticleSpeed() const {
    double maxS2 = 0.0;
    for (const Particle& p : particles) {
        const double s2 = (double)p.u * (double)p.u + (double)p.v * (double)p.v;
        if (s2 > maxS2) maxS2 = s2;
    }
    return (float)std::sqrt(maxS2);
}

void MACWater::enforceBorderSolids() {
    const int maxBt = std::max(1, std::min(nx, ny) / 2 - 1);
    const int bt = std::max(1, std::min(borderThickness, maxBt));
    borderThickness = bt;

    auto markSolid = [&](int i, int j) {
        solid[idxP(i, j)] = 1;
    };

    // bottom/top thickness
    for (int i = 0; i < nx; ++i) {
        for (int t = 0; t < bt; ++t) {
            markSolid(i, t);
            if (!openTop) markSolid(i, ny - 1 - t);
        }
    }

    // left/right thickness
    for (int j = 0; j < ny; ++j) {
        for (int t = 0; t < bt; ++t) {
            markSolid(t, j);
            markSolid(nx - 1 - t, j);
        }
    }
}

void MACWater::applyBoundary() {
    // left/right: u = 0
    for (int j = 0; j < ny; j++) {
        u[idxU(0, j)]  = 0.0f;
        u[idxU(nx, j)] = 0.0f;
    }

    // floor tangential kill
    for (int i = 0; i <= nx; ++i) {
        u[idxU(i, 0)] = 0.0f;
    }

    // bottom/top v
    for (int i = 0; i < nx; i++) {
        v[idxV(i, 0)] = 0.0f;
        if (!openTop) v[idxV(i, ny)] = 0.0f;
        else          v[idxV(i, ny)] = v[idxV(i, ny - 1)];
    }

    // no-through for internal solids
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i <= nx; ++i) {
            bool leftSolid  = (i - 1 >= 0) ? isSolid(i - 1, j) : true;
            bool rightSolid = (i < nx)     ? isSolid(i, j)     : true;
            if (leftSolid || rightSolid) u[idxU(i, j)] = 0.0f;
        }
    }

    for (int j = 0; j <= ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            if (j == 0 || j == ny) continue;
            bool botSolid = isSolid(i, j - 1);
            bool topSolid = isSolid(i, j);
            if (botSolid || topSolid) v[idxV(i, j)] = 0.0f;
        }
    }
}

void MACWater::addWaterSource(float cx, float cy, float radius, float amount) {
    std::uniform_real_distribution<float> uni(-0.45f, 0.45f);
    std::uniform_real_distribution<float> uJitter(-0.25f, 0.25f);
    const float cap = capOrInf(waterTargetMax);
    const float dropDir = (waterGravity <= 0.0f) ? -1.0f : 1.0f;
    const float dropSpeed = std::max(0.0f, sourceDownwardSpeed);

    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            if (isSolid(i, j)) continue;

            float x = (i + 0.5f) * dx;
            float y = (j + 0.5f) * dx;
            float dx0 = x - cx;
            float dy0 = y - cy;

            if (dx0 * dx0 + dy0 * dy0 <= radius * radius) {
                int id = idxP(i, j);
                const float deltaMass = std::max(0.0f, amount);
                water[id] = std::min(cap, water[id] + deltaMass);
                waterTarget[id] = std::min(cap, waterTarget[id] + deltaMass);
                targetMass += deltaMass;

                int spawnCount = (int)std::lround(deltaMass * (float)particlesPerCell);
                if (deltaMass > 0.0f) spawnCount = std::max(1, spawnCount);

                for (int s = 0; s < spawnCount && canSpawnMore(maxParticles, particles.size()); ++s) {
                    Particle p;
                    p.x = x + uni(globalRng()) * dx;
                    p.y = y + uni(globalRng()) * dx;
                    // Give sources a small downward bias so water drops before spreading.
                    p.u = 0.15f * uJitter(globalRng());
                    p.v = dropDir * dropSpeed;
                    p.age = 0.0f;
                    particles.push_back(p);
                }
            }
        }
    }
}

void MACWater::syncSolidsFrom(const MACGridCore& src) {
    if (src.solid.size() != solid.size()) return;
    solid = src.solid;
    mgDirty = true;

    enforceBorderSolids();

    for (size_t k = 0; k < solid.size(); ++k) {
        if (!solid[k]) continue;
        if (k < water.size()) water[k] = 0.0f;
        if (k < waterTarget.size()) {
            targetMass -= waterTarget[k];
            waterTarget[k] = 0.0f;
        }
        if (k < liquid.size()) liquid[k] = 0;
        if (k < liquidPrev.size()) liquidPrev[k] = 0;
    }
    targetMass = std::max(0.0f, targetMass);

    removeParticlesInSolids();
}

void MACWater::advectParticles() {
    const float holdTime = std::max(0.0f, sourceVelHold);
    const float holdBlend = clamp01(sourceVelBlend);
    for (Particle& p : particles) {
        const float x0 = p.x;
        const float y0 = p.y;
        const float prevU = p.u;
        const float prevV = p.v;

        float u1, v1;
        velAt(x0, y0, u, v, u1, v1);

        float midx = x0 + 0.5f * dt * u1;
        float midy = y0 + 0.5f * dt * v1;
        midx = clampf(midx, 0.0f, nx * dx);
        midy = clampf(midy, 0.0f, ny * dx);

        float u2, v2;
        velAt(midx, midy, u, v, u2, v2);

        float blend = 0.0f;
        if (holdTime > 0.0f && holdBlend > 0.0f) {
            const float tHold = clamp01(1.0f - p.age / holdTime);
            blend = holdBlend * tHold;
        }

        const float advU = (1.0f - blend) * u2 + blend * prevU;
        const float advV = (1.0f - blend) * v2 + blend * prevV;

        p.x = x0 + dt * advU;
        p.y = y0 + dt * advV;

        p.u = advU;
        p.v = advV;
        p.age += dt;
    }
}

void MACWater::separateParticles() {
    if (particles.empty()) return;
    const int iters = std::max(0, separationIters);
    if (iters <= 0) return;

    const float radius = std::max(0.0f, particleRadiusScale) * dx;
    if (radius <= 1e-6f) return;
    const float radius2 = radius * radius;
    const float strength = std::max(0.0f, std::min(1.0f, separationStrength));
    if (strength <= 0.0f) return;

    auto resolvePair = [&](int ia, int ib) {
        Particle& a = particles[(size_t)ia];
        Particle& b = particles[(size_t)ib];
        float dx0 = b.x - a.x;
        float dy0 = b.y - a.y;
        float d2 = dx0 * dx0 + dy0 * dy0;
        if (d2 >= radius2 || d2 <= 1e-12f) return;
        float d = std::sqrt(d2);
        float overlap = radius - d;
        float corr = 0.5f * strength * overlap / (d + 1e-6f);
        float sx = dx0 * corr;
        float sy = dy0 * corr;
        a.x -= sx; a.y -= sy;
        b.x += sx; b.y += sy;
    };

    const int Nc = nx * ny;
    for (int it = 0; it < iters; ++it) {
        std::vector<std::vector<int>> buckets((size_t)Nc);

        for (int pi = 0; pi < (int)particles.size(); ++pi) {
            int i, j;
            worldToCell(particles[(size_t)pi].x, particles[(size_t)pi].y, i, j);
            int id = idxP(i, j);
            if (id >= 0 && id < Nc) buckets[(size_t)id].push_back(pi);
        }

        auto processBucketPairs = [&](const std::vector<int>& a, const std::vector<int>& b, bool same) {
            if (a.empty() || b.empty()) return;
            if (same) {
                for (size_t ii = 0; ii < a.size(); ++ii) {
                    for (size_t jj = ii + 1; jj < a.size(); ++jj) resolvePair(a[ii], a[jj]);
                }
            } else {
                for (int ia : a) for (int ib : b) resolvePair(ia, ib);
            }
        };

        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int id = idxP(i, j);
                const auto& here = buckets[(size_t)id];
                if (here.empty()) continue;

                processBucketPairs(here, here, true);

                if (i + 1 < nx) processBucketPairs(here, buckets[(size_t)idxP(i + 1, j)], false);
                if (j + 1 < ny) processBucketPairs(here, buckets[(size_t)idxP(i, j + 1)], false);
                if (i + 1 < nx && j + 1 < ny) processBucketPairs(here, buckets[(size_t)idxP(i + 1, j + 1)], false);
                if (i - 1 >= 0 && j + 1 < ny) processBucketPairs(here, buckets[(size_t)idxP(i - 1, j + 1)], false);
            }
        }

        enforceParticleBounds();
        removeParticlesInSolids();
    }
}

void MACWater::relaxParticleDensity() {
    if (particles.empty()) return;
    const int interval = std::max(1, densityRelaxInterval);
    if (interval > 1 && (stepCounter % interval) != 0) return;
    const int iters = std::max(0, densityRelaxIters);
    if (iters <= 0) return;
    const int targetPerCell = std::max(1, particlesPerCell);

    const int bt = std::max(1, borderThickness);
    const int iMin = std::min(bt, std::max(1, nx - 2));
    const int jMin = std::min(bt, std::max(1, ny - 2));
    const int iMax = std::max(iMin, nx - bt - 1);
    const int jMax = std::max(jMin, ny - bt - 1);
    const float relaxFrac = clamp01(densityRelaxMaxYFrac);
    const int relaxHeight = std::max(1, (int)std::floor(relaxFrac * (float)(jMax - jMin + 1)));
    const int jRelaxMax = std::min(jMax, jMin + relaxHeight - 1);
    if (jRelaxMax < jMin) return;

    const int Nc = nx * ny;
    std::uniform_real_distribution<float> jitter(0.2f, 0.8f);

    auto inRange = [&](int i, int j) {
        return i >= iMin && i <= iMax && j >= jMin && j <= jRelaxMax;
    };

    for (int it = 0; it < iters; ++it) {
        std::vector<int> counts((size_t)Nc, 0);
        std::vector<std::vector<int>> cellParticles((size_t)Nc);

        for (int pi = 0; pi < (int)particles.size(); ++pi) {
            int rawI, rawJ;
            worldToCell(particles[(size_t)pi].x, particles[(size_t)pi].y, rawI, rawJ);
            if (rawJ > jRelaxMax) continue;
            int i = rawI;
            int j = rawJ;
            i = std::max(iMin, std::min(iMax, i));
            j = std::max(jMin, std::min(jRelaxMax, j));
            const int id = idxP(i, j);
            if (solid[(size_t)id]) continue;
            counts[(size_t)id] += 1;
            cellParticles[(size_t)id].push_back(pi);
        }

        for (int j = jMin; j <= jRelaxMax; ++j) {
            for (int i = iMin; i <= iMax; ++i) {
                const int id = idxP(i, j);
                if (solid[(size_t)id]) continue;
                int excess = counts[(size_t)id] - targetPerCell;
                if (excess <= 0) continue;

                auto tryMoveToBestNeighbor = [&](int particleIndex) {
                    // Prefer pushing upward, then sideways.
                    const int maxRise = std::min(12, std::max(0, jRelaxMax - j));
                    if (maxRise <= 0) return false;
                    const int candI[6] = { i, i - 1, i + 1, i - 1, i + 1, i };
                    const int candJ[6] = { std::min(jRelaxMax, j + 1),
                                           std::min(jRelaxMax, j + 1),
                                           std::min(jRelaxMax, j + 1),
                                           j,
                                           j,
                                           std::max(jMin, j - 1) };

                    int bestI = -1, bestJ = -1;
                    int bestCount = std::numeric_limits<int>::max();

                    for (int c = 0; c < 6; ++c) {
                        const int ni = candI[c];
                        const int nj = candJ[c];
                        if (!inRange(ni, nj)) continue;
                        const int nid = idxP(ni, nj);
                        if (solid[(size_t)nid]) continue;
                        const int nCount = counts[(size_t)nid];
                        if (nCount < bestCount) {
                            bestCount = nCount;
                            bestI = ni;
                            bestJ = nj;
                        }
                    }

                    if (bestI < 0) return false;
                    const int nid = idxP(bestI, bestJ);
                    if (counts[(size_t)nid] >= targetPerCell && bestCount >= counts[(size_t)id]) {
                        return false;
                    }

                    Particle& p = particles[(size_t)particleIndex];
                    p.x = (bestI + jitter(globalRng())) * dx;
                    p.y = (bestJ + jitter(globalRng())) * dx;
                    counts[(size_t)id] -= 1;
                    counts[(size_t)nid] += 1;
                    return true;
                };

                auto& plist = cellParticles[(size_t)id];
                for (int k = (int)plist.size() - 1; k >= 0 && excess > 0; --k) {
                    if (tryMoveToBestNeighbor(plist[(size_t)k])) {
                        excess -= 1;
                    }
                }
            }
        }

        enforceParticleBounds();
        removeParticlesInSolids();
    }
}

void MACWater::relaxColumnDensity() {
    if (particles.empty()) return;
    const int interval = std::max(1, columnRelaxInterval);
    if (interval > 1 && (stepCounter % interval) != 0) return;
    const int iters = std::max(0, columnRelaxIters);
    if (iters <= 0) return;

    const int bt = std::max(1, borderThickness);
    const int iMin = std::min(bt, std::max(1, nx - 2));
    const int jMin = std::min(bt, std::max(1, ny - 2));
    const int iMax = std::max(iMin, nx - bt - 1);
    const int jMax = std::max(jMin, ny - bt - 1);
    if (iMax - iMin < 1 || jMax < jMin) return;

    const float relaxFrac = clamp01(columnRelaxMaxYFrac);
    const int relaxHeight = std::max(1, (int)std::floor(relaxFrac * (float)(jMax - jMin + 1)));
    const int relaxMaxJ = std::min(jMax, jMin + relaxHeight - 1);
    if (relaxMaxJ < jMin) return;

    std::uniform_real_distribution<float> jitter(0.15f, 0.85f);

    // Column validity depends only on the solid mask inside the relaxation band.
    std::vector<uint8_t> colValid((size_t)nx, (uint8_t)0);
    for (int i = iMin; i <= iMax; ++i) {
        for (int j = jMin; j <= relaxMaxJ; ++j) {
            if (!solid[(size_t)idxP(i, j)]) { colValid[(size_t)i] = 1; break; }
        }
    }

    auto findNonSolidJ = [&](int colI, int baseJ) {
        baseJ = std::max(jMin, std::min(relaxMaxJ, baseJ));
        if (!solid[(size_t)idxP(colI, baseJ)]) return baseJ;
        for (int dj = 1; dj <= (relaxMaxJ - jMin); ++dj) {
            const int up = baseJ + dj;
            if (up <= relaxMaxJ && !solid[(size_t)idxP(colI, up)]) return up;
            const int dn = baseJ - dj;
            if (dn >= jMin && !solid[(size_t)idxP(colI, dn)]) return dn;
        }
        for (int jj = jMin; jj <= relaxMaxJ; ++jj) {
            if (!solid[(size_t)idxP(colI, jj)]) return jj;
        }
        return baseJ;
    };

    for (int it = 0; it < iters; ++it) {
        std::vector<int> colCountsFull((size_t)nx, 0);
        std::vector<int> colCountsMovable((size_t)nx, 0);
        std::vector<std::vector<int>> colParticles((size_t)nx);

        int totalFull = 0;
        int validCols = 0;
        for (int i = iMin; i <= iMax; ++i) if (colValid[(size_t)i]) validCols++;
        if (validCols <= 0) return;

        for (int pi = 0; pi < (int)particles.size(); ++pi) {
            int i, j;
            worldToCell(particles[(size_t)pi].x, particles[(size_t)pi].y, i, j);
            i = std::max(iMin, std::min(iMax, i));
            j = std::max(jMin, std::min(jMax, j));
            const int id = idxP(i, j);
            if (solid[(size_t)id] || !colValid[(size_t)i]) continue;
            colCountsFull[(size_t)i] += 1;
            totalFull += 1;
            if (j <= relaxMaxJ) {
                colCountsMovable[(size_t)i] += 1;
                colParticles[(size_t)i].push_back(pi);
            }
        }

        if (totalFull <= 0) return;

        const double targetPerColD = (double)totalFull / (double)validCols;
        const int targetPerCol = std::max(1, (int)std::lround(targetPerColD));
        const float slackFrac = clamp01(columnRelaxSlackFrac);
        const int slack = std::max(2, (int)std::lround((double)targetPerCol * (double)slackFrac));

        std::vector<int> need((size_t)nx, 0);
        for (int i = iMin; i <= iMax; ++i) {
            if (!colValid[(size_t)i]) continue;
            need[(size_t)i] = targetPerCol - colCountsFull[(size_t)i];
        }

        const bool reverseI = (it & 1) != 0;
        auto coinFlip = [&]() { return (globalRng()() & 1u) != 0u; };
        const int span = iMax - iMin;

        auto pickUnderfullColumn = [&](int srcI) {
            for (int dist = 1; dist <= span; ++dist) {
                const int left = srcI - dist;
                const int right = srcI + dist;
                const bool leftOk = (left >= iMin && colValid[(size_t)left] && need[(size_t)left] > 0);
                const bool rightOk = (right <= iMax && colValid[(size_t)right] && need[(size_t)right] > 0);
                if (!leftOk && !rightOk) continue;
                if (leftOk && rightOk) return coinFlip() ? left : right;
                return leftOk ? left : right;
            }
            return -1;
        };

        for (int s = 0; s <= span; ++s) {
            const int i = reverseI ? (iMax - s) : (iMin + s);
            if (!colValid[(size_t)i]) continue;
            if (colCountsMovable[(size_t)i] <= 0) continue;

            int excess = colCountsFull[(size_t)i] - (targetPerCol + slack);
            if (excess <= 0) continue;

            auto& plist = colParticles[(size_t)i];
            for (int k = (int)plist.size() - 1; k >= 0 && excess > 0; --k) {
                const int dstI = pickUnderfullColumn(i);
                if (dstI < 0) break;

                Particle& p = particles[(size_t)plist[(size_t)k]];
                int baseI, baseJ;
                worldToCell(p.x, p.y, baseI, baseJ);
                baseJ = std::max(jMin, std::min(relaxMaxJ, baseJ));
                const int dstJ = findNonSolidJ(dstI, baseJ);

                p.x = (dstI + jitter(globalRng())) * dx;
                p.y = (dstJ + jitter(globalRng())) * dx;

                colCountsFull[(size_t)i] -= 1;
                colCountsFull[(size_t)dstI] += 1;
                colCountsMovable[(size_t)i] -= 1;
                colCountsMovable[(size_t)dstI] += 1;
                need[(size_t)i] += 1;
                need[(size_t)dstI] -= 1;
                excess -= 1;
            }
        }

        enforceParticleBounds();
        removeParticlesInSolids();
    }
}

void MACWater::particleToGrid() {
    std::fill(u.begin(), u.end(), 0.0f);
    std::fill(v.begin(), v.end(), 0.0f);
    std::fill(uWeight.begin(), uWeight.end(), 0.0f);
    std::fill(vWeight.begin(), vWeight.end(), 0.0f);

    auto clampIndex = [](int v, int lo, int hi) {
        return std::max(lo, std::min(v, hi));
    };

    for (const Particle& p : particles) {
        // scatter to u faces
        {
            float fx = p.x / dx;
            float fy = p.y / dx - 0.5f;

            int i0 = (int)std::floor(fx);
            int j0 = (int)std::floor(fy);
            i0 = clampIndex(i0, 0, nx - 1);
            j0 = clampIndex(j0, 0, ny - 1);

            int i1 = std::min(i0 + 1, nx);
            int j1 = std::min(j0 + 1, ny - 1);

            float tx = clampf(fx - (float)i0, 0.0f, 1.0f);
            float ty = clampf(fy - (float)j0, 0.0f, 1.0f);

            float w00 = (1.0f - tx) * (1.0f - ty);
            float w10 = tx * (1.0f - ty);
            float w01 = (1.0f - tx) * ty;
            float w11 = tx * ty;

            int id00 = idxU(i0, j0);
            int id10 = idxU(i1, j0);
            int id01 = idxU(i0, j1);
            int id11 = idxU(i1, j1);

            u[id00] += w00 * p.u; uWeight[id00] += w00;
            u[id10] += w10 * p.u; uWeight[id10] += w10;
            u[id01] += w01 * p.u; uWeight[id01] += w01;
            u[id11] += w11 * p.u; uWeight[id11] += w11;
        }

        // scatter to v faces
        {
            float fx = p.x / dx - 0.5f;
            float fy = p.y / dx;

            int i0 = (int)std::floor(fx);
            int j0 = (int)std::floor(fy);
            i0 = clampIndex(i0, 0, nx - 1);
            j0 = clampIndex(j0, 0, ny - 1);

            int i1 = std::min(i0 + 1, nx - 1);
            int j1 = std::min(j0 + 1, ny);

            float tx = clampf(fx - (float)i0, 0.0f, 1.0f);
            float ty = clampf(fy - (float)j0, 0.0f, 1.0f);

            float w00 = (1.0f - tx) * (1.0f - ty);
            float w10 = tx * (1.0f - ty);
            float w01 = (1.0f - tx) * ty;
            float w11 = tx * ty;

            int id00 = idxV(i0, j0);
            int id10 = idxV(i1, j0);
            int id01 = idxV(i0, j1);
            int id11 = idxV(i1, j1);

            v[id00] += w00 * p.v; vWeight[id00] += w00;
            v[id10] += w10 * p.v; vWeight[id10] += w10;
            v[id01] += w01 * p.v; vWeight[id01] += w01;
            v[id11] += w11 * p.v; vWeight[id11] += w11;
        }
    }

    for (size_t i = 0; i < u.size(); ++i) {
        if (uWeight[i] > 1e-6f) u[i] /= uWeight[i];
    }
    for (size_t i = 0; i < v.size(); ++i) {
        if (vWeight[i] > 1e-6f) v[i] /= vWeight[i];
    }

    applyBoundary();
}

void MACWater::buildLiquidMask() {
    std::fill(liquid.begin(), liquid.end(), (uint8_t)0);

    const int Nc = nx * ny;
    std::vector<int> particleCounts((size_t)Nc, 0);
    for (const Particle& p : particles) {
        int i, j;
        worldToCell(p.x, p.y, i, j);
        const int id = idxP(i, j);
        if (solid[(size_t)id]) continue;
        particleCounts[(size_t)id] += 1;
    }

    auto hasParticleNeighbor = [&](int i, int j) {
        for (int dj = -1; dj <= 1; ++dj) {
            for (int di = -1; di <= 1; ++di) {
                const int ni = i + di;
                const int nj = j + dj;
                if (ni < 0 || ni >= nx || nj < 0 || nj >= ny) continue;
                const int nid = idxP(ni, nj);
                if (particleCounts[(size_t)nid] > 0) return true;
            }
        }
        return false;
    };

    const float maskThresh = std::max(liquidThreshold, 0.02f);
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int id = idxP(i, j);
            if (solid[(size_t)id]) continue;

            if (particleCounts[(size_t)id] > 0) {
                liquid[(size_t)id] = 1;
                continue;
            }

            // Allow field-based masking only when anchored to nearby particles.
            const bool nearParticles = hasParticleNeighbor(i, j);
            if (!nearParticles) continue;

            const float prev = (id < (int)water0.size()) ? water0[(size_t)id] : 0.0f;
            if (prev > maskThresh) {
                liquid[(size_t)id] = 1;
            } else if (id < (int)liquidPrev.size() && liquidPrev[(size_t)id] && prev > 0.5f * maskThresh) {
                liquid[(size_t)id] = 1;
            }
        }
    }

    const int dilations = std::max(0, maskDilations);
    for (int it = 0; it < dilations; ++it) {
        std::vector<uint8_t> dilated = liquid;
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int id = idxP(i, j);
                if (solid[id] || liquid[id]) continue;

                bool neigh =
                    (i > 0     && liquid[idxP(i - 1, j)]) ||
                    (i < nx-1  && liquid[idxP(i + 1, j)]) ||
                    (j > 0     && liquid[idxP(i, j - 1)]) ||
                    (j < ny-1  && liquid[idxP(i, j + 1)]);

                if (neigh) dilated[id] = 1;
            }
        }
        for (size_t k = 0; k < dilated.size() && k < solid.size(); ++k) {
            if (solid[k]) dilated[k] = 0;
        }
        liquid.swap(dilated);
    }

    for (size_t k = 0; k < liquid.size() && k < solid.size(); ++k) {
        if (solid[k]) liquid[k] = 0;
    }

    bool maskChanged = (liquidPrev.size() != liquid.size());
    if (!maskChanged) {
        for (size_t k = 0; k < liquid.size(); ++k) {
            if (liquid[k] != liquidPrev[k]) { maskChanged = true; break; }
        }
    }
    if (maskChanged) mgDirty = true;
    liquidPrev = liquid;
}

void MACWater::applyHeightPressureForce() {
    const float scale = std::max(0.0f, heightPressureScale);
    const float g = std::fabs(waterGravity);
    if (scale <= 0.0f || g <= 0.0f || particles.empty()) return;

    const int bt = std::max(1, borderThickness);
    const int iMin = std::min(bt, std::max(1, nx - 2));
    const int jMin = std::min(bt, std::max(1, ny - 2));
    const int iMax = std::max(iMin, nx - bt - 1);
    const int jMax = std::max(jMin, ny - bt - 1);
    if (iMax - iMin < 1 || jMax < jMin) return;

    std::vector<double> massCol((size_t)nx, 0.0);
    std::vector<uint8_t> colValid((size_t)nx, (uint8_t)0);
    const double invPpc = 1.0 / (double)std::max(1, particlesPerCell);

    for (const Particle& p : particles) {
        int i, j;
        worldToCell(p.x, p.y, i, j);
        i = std::max(iMin, std::min(iMax, i));
        j = std::max(jMin, std::min(jMax, j));
        const int id = idxP(i, j);
        if (solid[(size_t)id]) continue;
        massCol[(size_t)i] += invPpc;
        colValid[(size_t)i] = 1;
    }

    const double maxHeightCells = (double)std::max(1, jMax - jMin + 1);
    std::vector<double> heightCells = massCol;
    for (int i = iMin; i <= iMax; ++i) {
        if (!colValid[(size_t)i]) { heightCells[(size_t)i] = 0.0; continue; }
        heightCells[(size_t)i] = std::min(heightCells[(size_t)i], maxHeightCells);
    }

    auto isLiquidCell = [&](int i, int j) {
        if (i < 0 || i >= nx || j < 0 || j >= ny) return false;
        const int id = idxP(i, j);
        return !solid[(size_t)id] && liquid[(size_t)id];
    };

    // du/dt ~= -g * d(h)/dx, where h is column height.
    for (int i = iMin + 1; i <= iMax; ++i) {
        if (!colValid[(size_t)i] && !colValid[(size_t)(i - 1)]) continue;

        const double dh = heightCells[(size_t)i] - heightCells[(size_t)(i - 1)];
        const float accel = -scale * g * (float)dh;
        if (std::fabs(accel) <= 1e-7f) continue;

        for (int j = jMin; j <= jMax; ++j) {
            const bool leftSolid = solid[(size_t)idxP(i - 1, j)] != 0;
            const bool rightSolid = solid[(size_t)idxP(i, j)] != 0;
            if (leftSolid || rightSolid) continue;

            const bool leftLiquid = isLiquidCell(i - 1, j);
            const bool rightLiquid = isLiquidCell(i, j);
            if (!leftLiquid && !rightLiquid) continue;

            u[(size_t)idxU(i, j)] += dt * accel;
        }
    }
}

void MACWater::applyViscosity() {
    if (waterViscosity <= 0.0f || viscosityIters <= 0) return;

    const float invDx2 = 1.0f / (dx * dx);
    const float alphaInvDx2 = (waterViscosity * dt) * invDx2;
    if (alphaInvDx2 <= 0.0f) return;

    const float omega = std::max(0.0f, std::min(1.0f, viscosityOmega));

    std::vector<float> bU = u;
    std::vector<float> bV = v;

    if (u0.size() != u.size()) u0.resize(u.size());
    if (v0.size() != v.size()) v0.resize(v.size());

    auto isLiquidCell = [&](int i, int j) {
        if (i < 0 || i >= nx || j < 0 || j >= ny) return false;
        const int id = idxP(i, j);
        return !solid[(size_t)id] && liquid[(size_t)id];
    };
    auto liquidAdjacentU = [&](int i, int j) {
        return isLiquidCell(i - 1, j) || isLiquidCell(i, j);
    };
    auto liquidAdjacentV = [&](int i, int j) {
        return isLiquidCell(i, j - 1) || isLiquidCell(i, j);
    };

    auto isFixedU = [&](int i, int j) {
        if (i == 0 || i == nx) return true;
        if (j == 0) return true; // floor tangential
        return isSolid(i - 1, j) || isSolid(i, j);
    };
    auto isFixedV = [&](int i, int j) {
        if (j == 0) return true;
        if (j == ny) return !openTop;
        return isSolid(i, j - 1) || isSolid(i, j);
    };

    // diffuse U
    for (int it = 0; it < viscosityIters; ++it) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i <= nx; ++i) {
                int id = idxU(i, j);
                if (isFixedU(i, j) || !liquidAdjacentU(i, j)) { u0[id] = u[id]; continue; }

                float sumN = 0.0f;
                int count = 0;
                if (i - 1 >= 0)  { sumN += u[idxU(i - 1, j)]; count++; }
                if (i + 1 <= nx) { sumN += u[idxU(i + 1, j)]; count++; }
                if (j - 1 >= 0)  { sumN += u[idxU(i, j - 1)]; count++; }
                if (j + 1 < ny)  { sumN += u[idxU(i, j + 1)]; count++; }

                float xNew = (bU[id] + alphaInvDx2 * sumN) / (1.0f + alphaInvDx2 * (float)count);
                u0[id] = (1.0f - omega) * u[id] + omega * xNew;
            }
        }
        u.swap(u0);
        applyBoundary();
    }

    // diffuse V
    for (int it = 0; it < viscosityIters; ++it) {
        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int id = idxV(i, j);
                if (isFixedV(i, j) || !liquidAdjacentV(i, j)) { v0[id] = v[id]; continue; }

                float sumN = 0.0f;
                int count = 0;
                if (i - 1 >= 0) { sumN += v[idxV(i - 1, j)]; count++; }
                if (i + 1 < nx) { sumN += v[idxV(i + 1, j)]; count++; }
                if (j - 1 >= 0)  { sumN += v[idxV(i, j - 1)]; count++; }
                if (j + 1 <= ny) { sumN += v[idxV(i, j + 1)]; count++; }

                float xNew = (bV[id] + alphaInvDx2 * sumN) / (1.0f + alphaInvDx2 * (float)count);
                v0[id] = (1.0f - omega) * v[id] + omega * xNew;
            }
        }
        v.swap(v0);
        applyBoundary();
    }
}

void MACWater::extrapolateVelocity() {
    std::vector<uint8_t> uValid(u.size(), 0);
    std::vector<uint8_t> vValid(v.size(), 0);
    for (size_t i = 0; i < u.size(); ++i) uValid[i] = (uWeight[i] > 1e-6f);
    for (size_t i = 0; i < v.size(); ++i) vValid[i] = (vWeight[i] > 1e-6f);

    auto isLiquidCell = [&](int i, int j) {
        if (i < 0 || i >= nx || j < 0 || j >= ny) return false;
        int id = idxP(i, j);
        return !solid[id] && liquid[id];
    };

    auto liquidAdjacentU = [&](int i, int j) {
        return isLiquidCell(i - 1, j) || isLiquidCell(i, j);
    };
    auto liquidAdjacentV = [&](int i, int j) {
        return isLiquidCell(i, j - 1) || isLiquidCell(i, j);
    };

    for (int it = 0; it < extrapIters; ++it) {
        std::vector<float> uNext = u;
        std::vector<float> vNext = v;
        std::vector<uint8_t> uValidNext = uValid;
        std::vector<uint8_t> vValidNext = vValid;

        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i <= nx; ++i) {
                int id = idxU(i, j);
                if (uValid[id] || !liquidAdjacentU(i, j)) continue;

                float sum = 0.0f;
                int cnt = 0;
                if (i > 0)     { int n = idxU(i - 1, j); if (uValid[n]) { sum += u[n]; cnt++; } }
                if (i < nx)    { int n = idxU(i + 1, j); if (uValid[n]) { sum += u[n]; cnt++; } }
                if (j > 0)     { int n = idxU(i, j - 1); if (uValid[n]) { sum += u[n]; cnt++; } }
                if (j < ny-1)  { int n = idxU(i, j + 1); if (uValid[n]) { sum += u[n]; cnt++; } }

                if (cnt > 0) {
                    uNext[id] = sum / (float)cnt;
                    uValidNext[id] = 1;
                }
            }
        }

        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int id = idxV(i, j);
                if (vValid[id] || !liquidAdjacentV(i, j)) continue;

                float sum = 0.0f;
                int cnt = 0;
                if (i > 0)     { int n = idxV(i - 1, j); if (vValid[n]) { sum += v[n]; cnt++; } }
                if (i < nx-1)  { int n = idxV(i + 1, j); if (vValid[n]) { sum += v[n]; cnt++; } }
                if (j > 0)     { int n = idxV(i, j - 1); if (vValid[n]) { sum += v[n]; cnt++; } }
                if (j < ny)    { int n = idxV(i, j + 1); if (vValid[n]) { sum += v[n]; cnt++; } }

                if (cnt > 0) {
                    vNext[id] = sum / (float)cnt;
                    vValidNext[id] = 1;
                }
            }
        }

        u.swap(uNext);
        v.swap(vNext);
        uValid.swap(uValidNext);
        vValid.swap(vValidNext);
    }

    applyBoundary();
}

void MACWater::projectLiquid() {
    const int N = nx * ny;
    if ((int)lapDiag.size() != N) {
        lapDiag.assign(N, 0.0f);
        lapDiagInv.assign(N, 0.0f);
        lapL.assign(N, -1);
        lapR.assign(N, -1);
        lapB.assign(N, -1);
        lapT.assign(N, -1);
        pcg_r.assign(N, 0.0f);
        pcg_z.assign(N, 0.0f);
        pcg_d.assign(N, 0.0f);
        pcg_q.assign(N, 0.0f);
        pcg_Ap.assign(N, 0.0f);
    }

    const float safeDt = std::max(dt, 1e-6f);
    auto isLiquidCell = [&](int i, int j) {
        if (i < 0 || i >= nx || j < 0 || j >= ny) return false;
        const int id = idxP(i, j);
        return !solid[id] && liquid[id];
    };

    // Domain boundaries are solid except for an open top.
    auto isSolidOrBoundary = [&](int i, int j) {
        if (i < 0 || i >= nx) return true;
        if (j < 0) return true;
        if (j >= ny) return !getOpenTop();
        return solid[(size_t)idxP(i, j)] != 0;
    };

    auto isNearSolidCell = [&](int i, int j) {
        return isSolidOrBoundary(i - 1, j) ||
               isSolidOrBoundary(i + 1, j) ||
               isSolidOrBoundary(i, j - 1) ||
               isSolidOrBoundary(i, j + 1);
    };

    double sumAbsDivBefore = 0.0;
    double sumAbsDivNearSolidBefore = 0.0;
    float maxAbsDivBefore = 0.0f;
    float maxAbsDivNearSolidBefore = 0.0f;
    int liquidCountBefore = 0;
    int nearSolidCountBefore = 0;

    // divergence and rhs only in liquid
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int id = idxP(i, j);
            if (solid[id] || !liquid[id]) {
                div[id] = 0.0f;
                rhs[id] = 0.0f;
                p[id] = 0.0f;
                continue;
            }

            float uL = u[idxU(i, j)];
            float uR = u[idxU(i + 1, j)];
            float vB = v[idxV(i, j)];
            float vT = v[idxV(i, j + 1)];

            // No-penetration at solids: normal face velocities are fixed to 0.
            if (isSolidOrBoundary(i - 1, j)) uL = 0.0f;
            if (isSolidOrBoundary(i + 1, j)) uR = 0.0f;
            if (isSolidOrBoundary(i, j - 1)) vB = 0.0f;
            if (isSolidOrBoundary(i, j + 1)) vT = 0.0f;

            const float divVal = (uR - uL + vT - vB) / dx;
            div[id] = divVal;
            rhs[id] = -divVal / safeDt;

            const float absDiv = std::fabs(divVal);
            maxAbsDivBefore = std::max(maxAbsDivBefore, absDiv);
            sumAbsDivBefore += (double)absDiv;
            liquidCountBefore += 1;

            if (isNearSolidCell(i, j)) {
                maxAbsDivNearSolidBefore = std::max(maxAbsDivNearSolidBefore, absDiv);
                sumAbsDivNearSolidBefore += (double)absDiv;
                nearSolidCountBefore += 1;
            }
        }
    }

    lastPressureDiag = PressureDiagnostics{};
    lastPressureDiag.maxAbsDivBefore = maxAbsDivBefore;
    lastPressureDiag.maxAbsDivNearSolidBefore = maxAbsDivNearSolidBefore;
    lastPressureDiag.meanAbsDivBefore =
        (liquidCountBefore > 0) ? (float)(sumAbsDivBefore / (double)liquidCountBefore) : 0.0f;
    lastPressureDiag.liquidCells = liquidCountBefore;
    lastPressureDiag.liquidCellsNearSolid = nearSolidCountBefore;

    const float invDx2 = 1.0f / (dx * dx);
    bool hasDirichletAnchor = false;
    int anchorId = -1;

    // Assemble the liquid Laplacian.
    // Air neighbors use Dirichlet p=0. Solid neighbors use Neumann (no contribution).
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int id = idxP(i, j);
            lapL[id] = lapR[id] = lapB[id] = lapT[id] = -1;
            lapDiag[id] = 0.0f;
            lapDiagInv[id] = 0.0f;

            if (!isLiquidCell(i, j)) {
                p[id] = 0.0f;
                continue;
            }

            if (anchorId < 0) anchorId = id;
            float diag = 0.0f;

            // left
            if (isSolidOrBoundary(i - 1, j)) {
                // solid boundary: Neumann -> no diagonal contribution
            } else if (isLiquidCell(i - 1, j)) {
                lapL[id] = idxP(i - 1, j);
                diag += invDx2;
            } else {
                hasDirichletAnchor = true;
                diag += invDx2;
            }

            // right
            if (isSolidOrBoundary(i + 1, j)) {
                // solid boundary: Neumann -> no diagonal contribution
            } else if (isLiquidCell(i + 1, j)) {
                lapR[id] = idxP(i + 1, j);
                diag += invDx2;
            } else {
                hasDirichletAnchor = true;
                diag += invDx2;
            }

            // bottom
            if (isSolidOrBoundary(i, j - 1)) {
                // solid boundary: Neumann -> no diagonal contribution
            } else if (isLiquidCell(i, j - 1)) {
                lapB[id] = idxP(i, j - 1);
                diag += invDx2;
            } else {
                hasDirichletAnchor = true;
                diag += invDx2;
            }

            // top
            if (isSolidOrBoundary(i, j + 1)) {
                // solid boundary: Neumann -> no diagonal contribution
            } else if (isLiquidCell(i, j + 1)) {
                lapT[id] = idxP(i, j + 1);
                diag += invDx2;
            } else {
                hasDirichletAnchor = true;
                diag += invDx2;
            }

            if (diag <= 0.0f) {
                // Isolated liquid cells (surrounded by solids) cannot be projected.
                // Force them to a benign identity row to keep the solve well-posed.
                lapDiag[id] = 1.0f;
                lapDiagInv[id] = 1.0f;
                rhs[id] = 0.0f;
                div[id] = 0.0f;
                p[id] = 0.0f;
                continue;
            }

            lapDiag[id] = diag;
            lapDiagInv[id] = 1.0f / diag;
        }
    }

    // A fully closed container with only Neumann boundaries is singular.
    // Pin a single liquid cell to zero pressure as a gauge fix.
    if (!hasDirichletAnchor && anchorId >= 0) {
        lapL[anchorId] = lapR[anchorId] = lapB[anchorId] = lapT[anchorId] = -1;
        lapDiag[anchorId] = 1.0f;
        lapDiagInv[anchorId] = 1.0f;
        rhs[anchorId] = 0.0f;
        div[anchorId] = 0.0f;
        p[anchorId] = 0.0f;
    }

    auto pcgSolve = [&](int maxIters, float tol) -> bool {
        auto applyA = [&](const std::vector<float>& x, std::vector<float>& Ax) {
            if ((int)Ax.size() != N) Ax.assign(N, 0.0f);

            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    const int id = idxP(i, j);
                    if (!isLiquidCell(i, j)) { Ax[id] = 0.0f; continue; }

                    float sum = 0.0f;
                    if (lapL[id] >= 0) sum += x[lapL[id]];
                    if (lapR[id] >= 0) sum += x[lapR[id]];
                    if (lapB[id] >= 0) sum += x[lapB[id]];
                    if (lapT[id] >= 0) sum += x[lapT[id]];

                    Ax[id] = lapDiag[id] * x[id] - invDx2 * sum;
                }
            }
        };

        applyA(p, pcg_Ap);
        for (int k = 0; k < N; ++k) pcg_r[k] = rhs[k] - pcg_Ap[k];

        const float bNorm2 = dotVec(rhs, rhs);
        if (bNorm2 < 1e-20f) return true;

        for (int k = 0; k < N; ++k) pcg_z[k] = pcg_r[k] * lapDiagInv[k];
        pcg_d = pcg_z;

        float deltaNew = dotVec(pcg_r, pcg_z);
        const float delta0 = deltaNew;
        if (deltaNew < 1e-20f) return true;

        const float tolSafe = std::max(1e-8f, tol);
        const float tol2 = tolSafe * tolSafe;
        const int maxIterSafe = std::max(10, maxIters);

        bool converged = false;
        for (int it = 0; it < maxIterSafe; ++it) {
            applyA(pcg_d, pcg_q);

            const float dq = dotVec(pcg_d, pcg_q);
            if (std::fabs(dq) < 1e-30f) break;

            const float alpha = deltaNew / dq;

            for (int k = 0; k < N; ++k) {
                p[k]     += alpha * pcg_d[k];
                pcg_r[k] -= alpha * pcg_q[k];
            }

            const float rNorm2 = dotVec(pcg_r, pcg_r);
            if (rNorm2 <= tol2 * bNorm2) { converged = true; break; }

            for (int k = 0; k < N; ++k) pcg_z[k] = pcg_r[k] * lapDiagInv[k];

            const float deltaOld = deltaNew;
            deltaNew = dotVec(pcg_r, pcg_z);
            if (deltaNew <= tol2 * delta0) { converged = true; break; }

            const float beta = deltaNew / (deltaOld + 1e-30f);
            for (int k = 0; k < N; ++k) pcg_d[k] = pcg_z[k] + beta * pcg_d[k];
        }
        return converged;
    };

    if (useMGPressure) {
        if (mgOpenTop != openTop) { mgOpenTop = openTop; mgDirty = true; }
        const float tol = (pressureMGTol > 0.0f) ? pressureMGTol : (openTop ? 5e-4f : 1e-4f);
        solvePressureMGWater(pressureMGVcycles, tol);
        const int polishIters = std::max(0, std::min(pressureMGPolishIters, pressureMaxIters));
        if (polishIters > 0) {
            const bool converged = pcgSolve(polishIters, pressureTol);
            if (!converged && polishIters < pressureMaxIters) {
                pcgSolve(pressureMaxIters, pressureTol);
            }
        }
    } else {
        pcgSolve(pressureMaxIters, pressureTol);
    }

    // apply pressure gradient with p_air = 0
    for (int j = 0; j < ny; ++j) {
        for (int i = 1; i < nx; ++i) {
            bool leftSolid  = solid[idxP(i - 1, j)] != 0;
            bool rightSolid = solid[idxP(i, j)] != 0;
            if (leftSolid || rightSolid) { u[idxU(i, j)] = 0.0f; continue; }

            bool leftLiquid  = liquid[idxP(i - 1, j)] != 0;
            bool rightLiquid = liquid[idxP(i, j)] != 0;
            if (!leftLiquid && !rightLiquid) { u[idxU(i, j)] = 0.0f; continue; }

            const float pL = leftLiquid  ? p[idxP(i - 1, j)] : 0.0f;
            const float pR = rightLiquid ? p[idxP(i, j)]     : 0.0f;
            const float gradp = (pR - pL) / dx;
            u[idxU(i, j)] -= dt * gradp;
        }
    }

    for (int j = 1; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            bool botSolid = solid[idxP(i, j - 1)] != 0;
            bool topSolid = solid[idxP(i, j)] != 0;
            if (botSolid || topSolid) { v[idxV(i, j)] = 0.0f; continue; }

            bool botLiquid = liquid[idxP(i, j - 1)] != 0;
            bool topLiquid = liquid[idxP(i, j)] != 0;
            if (!botLiquid && !topLiquid) { v[idxV(i, j)] = 0.0f; continue; }

            const float pB = botLiquid ? p[idxP(i, j - 1)] : 0.0f;
            const float pT = topLiquid ? p[idxP(i, j)]     : 0.0f;
            const float gradp = (pT - pB) / dx;
            v[idxV(i, j)] -= dt * gradp;
        }
    }

    applyBoundary();

    struct DivStats {
        float maxAbs = 0.0f;
        float maxAbsNearSolid = 0.0f;
        double sumAbs = 0.0;
        double sumAbsNearSolid = 0.0;
        int count = 0;
        int countNearSolid = 0;
    };

    auto computeDivStats = [&](bool writeDivField) {
        DivStats stats;
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const int id = idxP(i, j);
                if (!isLiquidCell(i, j)) {
                    if (writeDivField) div[id] = 0.0f;
                    continue;
                }

                float uL = u[idxU(i, j)];
                float uR = u[idxU(i + 1, j)];
                float vB = v[idxV(i, j)];
                float vT = v[idxV(i, j + 1)];

                if (isSolidOrBoundary(i - 1, j)) uL = 0.0f;
                if (isSolidOrBoundary(i + 1, j)) uR = 0.0f;
                if (isSolidOrBoundary(i, j - 1)) vB = 0.0f;
                if (isSolidOrBoundary(i, j + 1)) vT = 0.0f;

                const float divVal = (uR - uL + vT - vB) / dx;
                if (writeDivField) div[id] = divVal;

                const float absDiv = std::fabs(divVal);
                stats.maxAbs = std::max(stats.maxAbs, absDiv);
                stats.sumAbs += (double)absDiv;
                stats.count += 1;

                if (isNearSolidCell(i, j)) {
                    stats.maxAbsNearSolid = std::max(stats.maxAbsNearSolid, absDiv);
                    stats.sumAbsNearSolid += (double)absDiv;
                    stats.countNearSolid += 1;
                }
            }
        }
        return stats;
    };

    const DivStats postStats = computeDivStats(true);
    lastPressureDiag.maxAbsDivAfter = postStats.maxAbs;
    lastPressureDiag.maxAbsDivNearSolidAfter = postStats.maxAbsNearSolid;
    lastPressureDiag.meanAbsDivAfter =
        (postStats.count > 0) ? (float)(postStats.sumAbs / (double)postStats.count) : 0.0f;
    lastPressureDiag.liquidCells = postStats.count;
    lastPressureDiag.liquidCellsNearSolid = postStats.countNearSolid;

    if (pressureDiagnostics && pressureDiagInterval > 0 &&
        (stepCounter % pressureDiagInterval) == 0) {
        std::printf(
            "[water][pressure] step=%d maxDiv %.3e -> %.3e | nearSolid %.3e -> %.3e | mean %.3e -> %.3e | cells=%d near=%d\n",
            stepCounter,
            lastPressureDiag.maxAbsDivBefore,
            lastPressureDiag.maxAbsDivAfter,
            lastPressureDiag.maxAbsDivNearSolidBefore,
            lastPressureDiag.maxAbsDivNearSolidAfter,
            lastPressureDiag.meanAbsDivBefore,
            lastPressureDiag.meanAbsDivAfter,
            lastPressureDiag.liquidCells,
            lastPressureDiag.liquidCellsNearSolid);
    }
}

void MACWater::gridToParticles() {
    const float blend = clamp01(flipBlend);
    const float picW = 1.0f - blend;

    for (Particle& p : particles) {
        float picU, picV;
        velAt(p.x, p.y, u, v, picU, picV);

        float du = sampleU(uDelta, p.x, p.y);
        float dv = sampleV(vDelta, p.x, p.y);

        float flipU = p.u + du;
        float flipV = p.v + dv;

        p.u = picW * picU + blend * flipU;
        p.v = picW * picV + blend * flipV;
    }
}

void MACWater::removeLiquidDrift() {
    auto isLiquidCell = [&](int i, int j) {
        if (i < 0 || i >= nx || j < 0 || j >= ny) return false;
        const int id = idxP(i, j);
        return !solid[(size_t)id] && liquid[(size_t)id];
    };

    auto liquidAdjacentU = [&](int i, int j) {
        return isLiquidCell(i - 1, j) || isLiquidCell(i, j);
    };

    double sumU = 0.0;
    int cntU = 0;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i <= nx; ++i) {
            const int id = idxU(i, j);
            if (!liquidAdjacentU(i, j)) continue;
            const bool leftSolid  = (i - 1 >= 0) ? isSolid(i - 1, j) : true;
            const bool rightSolid = (i < nx)     ? isSolid(i, j)     : true;
            if (leftSolid || rightSolid) continue;
            sumU += (double)u[(size_t)id];
            cntU += 1;
        }
    }

    if (cntU <= 0) return;
    const float meanU = (float)(sumU / (double)cntU);
    if (std::fabs(meanU) <= 1e-7f) return;

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i <= nx; ++i) {
            const int id = idxU(i, j);
            if (!liquidAdjacentU(i, j)) continue;
            const bool leftSolid  = (i - 1 >= 0) ? isSolid(i - 1, j) : true;
            const bool rightSolid = (i < nx)     ? isSolid(i, j)     : true;
            if (leftSolid || rightSolid) continue;
            u[(size_t)id] -= meanU;
        }
    }

    // Do not remove mean vertical velocity: that cancels gravity.
}

void MACWater::snapToRest(float restVel) {
    if (restVel <= 0.0f) return;

    auto isLiquidCell = [&](int i, int j) {
        if (i < 0 || i >= nx || j < 0 || j >= ny) return false;
        const int id = idxP(i, j);
        return !solid[(size_t)id] && liquid[(size_t)id];
    };
    auto liquidAdjacentU = [&](int i, int j) {
        return isLiquidCell(i - 1, j) || isLiquidCell(i, j);
    };
    auto liquidAdjacentV = [&](int i, int j) {
        return isLiquidCell(i, j - 1) || isLiquidCell(i, j);
    };

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i <= nx; ++i) {
            if (!liquidAdjacentU(i, j)) continue;
            const int id = idxU(i, j);
            if (std::fabs(u[(size_t)id]) < restVel) u[(size_t)id] = 0.0f;
        }
    }

    for (int j = 0; j <= ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            if (!liquidAdjacentV(i, j)) continue;
            const int id = idxV(i, j);
            if (std::fabs(v[(size_t)id]) < restVel) v[(size_t)id] = 0.0f;
        }
    }

    const float restVel2 = restVel * restVel;
    for (Particle& p : particles) {
        if (p.u * p.u + p.v * p.v < restVel2) {
            p.u = 0.0f;
            p.v = 0.0f;
        }
    }

    applyBoundary();
}

void MACWater::ensureWaterMG() {
    if (!mgDirty) return;
    mgDirty = false;
    mgLevels.clear();
    mgLevels.reserve(mgMaxLevels);
    mgHasDirichlet = false;

    {
        MGLevel L0;
        L0.nx = nx; L0.ny = ny;
        L0.invDx2 = 1.0f / (dx * dx);

        const int N = nx * ny;
        L0.solid.assign(N, 0);
        L0.fluid.assign(N, 0);
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const int id = idxP(i, j);
                L0.solid[id] = solid[(size_t)id] ? 1 : 0;
                if (!L0.solid[id] && liquid[(size_t)id]) L0.fluid[id] = 1;
            }
        }

        L0.L.assign(N, -1); L0.R.assign(N, -1); L0.B.assign(N, -1); L0.T.assign(N, -1);
        L0.diagInv.assign(N, 0.0f);
        L0.x.assign(N, 0.0f);
        L0.b.assign(N, 0.0f);
        L0.Ax.assign(N, 0.0f);
        L0.r.assign(N, 0.0f);

        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const int id = idxP(i, j);
                if (!L0.fluid[id]) continue;

                int count = 0;
                auto handleNeighbor = [&](int ni, int nj, int& outIdx) {
                    if (ni < 0 || ni >= L0.nx || nj < 0) return;
                    if (nj >= L0.ny) {
                        if (openTop) { count++; mgHasDirichlet = true; }
                        return;
                    }
                    const int nid = mgIdx(ni, nj, L0.nx);
                    if (L0.solid[nid]) return;
                    if (L0.fluid[nid]) { outIdx = nid; count++; }
                    else { count++; mgHasDirichlet = true; }
                };

                handleNeighbor(i - 1, j, L0.L[id]);
                handleNeighbor(i + 1, j, L0.R[id]);
                handleNeighbor(i, j - 1, L0.B[id]);
                handleNeighbor(i, j + 1, L0.T[id]);

                if (count <= 0) {
                    L0.diagInv[id] = 1.0f;
                } else {
                    const float diag = (float)count * L0.invDx2;
                    L0.diagInv[id] = 1.0f / diag;
                }
            }
        }

        mgLevels.push_back(std::move(L0));
    }

    while ((int)mgLevels.size() < mgMaxLevels) {
        const MGLevel& F = mgLevels.back();
        if (F.nx <= 4 || F.ny <= 4) break;

        int cnx = F.nx / 2;
        int cny = F.ny / 2;
        if (cnx < 2 || cny < 2) break;

        MGLevel C;
        C.nx = cnx; C.ny = cny;
        C.invDx2 = F.invDx2 * 0.25f;

        const int CN = cnx * cny;
        C.solid.assign(CN, 0);
        C.fluid.assign(CN, 0);
        C.L.assign(CN, -1); C.R.assign(CN, -1); C.B.assign(CN, -1); C.T.assign(CN, -1);
        C.diagInv.assign(CN, 0.0f);
        C.x.assign(CN, 0.0f);
        C.b.assign(CN, 0.0f);
        C.Ax.assign(CN, 0.0f);
        C.r.assign(CN, 0.0f);

        for (int J = 0; J < cny; ++J) {
            for (int I = 0; I < cnx; ++I) {
                int fi = 2 * I;
                int fj = 2 * J;

                bool allSolid = true;
                bool anyFluid = false;
                bool anyAir = false;
                for (int dj = 0; dj < 2; ++dj) {
                    for (int di = 0; di < 2; ++di) {
                        int ii = fi + di;
                        int jj = fj + dj;
                        if (ii < F.nx && jj < F.ny) {
                            int fid = mgIdx(ii, jj, F.nx);
                            if (!F.solid[fid]) allSolid = false;
                            if (F.fluid[fid]) anyFluid = true;
                            else if (!F.solid[fid]) anyAir = true;
                        }
                    }
                }

                int cid = mgIdx(I, J, cnx);
                C.solid[cid] = allSolid ? 1 : 0;
                C.fluid[cid] = anyFluid ? 1 : 0;
                if (allSolid) C.fluid[cid] = 0;
                if (!anyFluid && anyAir) C.solid[cid] = 0;
            }
        }

        for (int J = 0; J < cny; ++J) {
            for (int I = 0; I < cnx; ++I) {
                int id = mgIdx(I, J, cnx);
                if (!C.fluid[id]) continue;

                int count = 0;
                auto handleNeighbor = [&](int ni, int nj, int& outIdx) {
                    if (ni < 0 || ni >= cnx || nj < 0) return;
                    if (nj >= cny) {
                        if (openTop) { count++; }
                        return;
                    }
                    int nid = mgIdx(ni, nj, cnx);
                    if (C.solid[nid]) return;
                    if (C.fluid[nid]) { outIdx = nid; count++; }
                    else { count++; }
                };

                handleNeighbor(I - 1, J, C.L[id]);
                handleNeighbor(I + 1, J, C.R[id]);
                handleNeighbor(I, J - 1, C.B[id]);
                handleNeighbor(I, J + 1, C.T[id]);

                if (count <= 0) {
                    C.diagInv[id] = 1.0f;
                } else {
                    const float diag = (float)count * C.invDx2;
                    C.diagInv[id] = 1.0f / diag;
                }
            }
        }

        mgLevels.push_back(std::move(C));
    }
}

void MACWater::mgApplyA(int lev, const std::vector<float>& x, std::vector<float>& Ax) const {
    const MGLevel& L = mgLevels[lev];
    const int N = L.nx * L.ny;
    if ((int)Ax.size() != N) Ax.resize(N);

    for (int id = 0; id < N; ++id) {
        if (!L.fluid[id]) { Ax[id] = 0.0f; continue; }

        float sum = 0.0f;
        int n = L.L[id]; if (n >= 0) sum += x[n];
        n = L.R[id];     if (n >= 0) sum += x[n];
        n = L.B[id];     if (n >= 0) sum += x[n];
        n = L.T[id];     if (n >= 0) sum += x[n];

        const float diag = (L.diagInv[id] > 0.0f) ? (1.0f / L.diagInv[id]) : 0.0f;
        Ax[id] = diag * x[id] - L.invDx2 * sum;
    }
}

void MACWater::mgSmoothJacobi(int lev, int iters) {
    MGLevel& L = mgLevels[lev];
    const int N = L.nx * L.ny;

    for (int it = 0; it < iters; ++it) {
        mgApplyA(lev, L.x, L.Ax);
        for (int id = 0; id < N; ++id) {
            if (!L.fluid[id]) continue;
            float r = L.b[id] - L.Ax[id];
            L.x[id] += mgOmega * (L.diagInv[id] * r);
        }
    }
}

void MACWater::mgComputeResidual(int lev) {
    MGLevel& L = mgLevels[lev];
    mgApplyA(lev, L.x, L.Ax);
    const int N = L.nx * L.ny;
    for (int id = 0; id < N; ++id) {
        if (!L.fluid[id]) { L.r[id] = 0.0f; continue; }
        L.r[id] = L.b[id] - L.Ax[id];
    }
}

void MACWater::mgRestrictResidual(int fineLev) {
    MGLevel& F = mgLevels[fineLev];
    MGLevel& C = mgLevels[fineLev + 1];

    std::fill(C.b.begin(), C.b.end(), 0.0f);
    std::fill(C.x.begin(), C.x.end(), 0.0f);

    for (int J = 0; J < C.ny; ++J) {
        for (int I = 0; I < C.nx; ++I) {
            int cid = mgIdx(I, J, C.nx);
            if (!C.fluid[cid]) { C.b[cid] = 0.0f; continue; }

            float sum = 0.0f;
            int fi = 2 * I;
            int fj = 2 * J;
            for (int dj = 0; dj < 2; ++dj) {
                for (int di = 0; di < 2; ++di) {
                    int ii = fi + di;
                    int jj = fj + dj;
                    if (ii < F.nx && jj < F.ny) {
                        int fid = mgIdx(ii, jj, F.nx);
                        if (F.fluid[fid]) sum += F.r[fid];
                    }
                }
            }
            C.b[cid] = 0.25f * sum;
        }
    }
}

void MACWater::mgProlongateAndAdd(int coarseLev) {
    MGLevel& C = mgLevels[coarseLev];
    MGLevel& F = mgLevels[coarseLev - 1];

    for (int J = 0; J < C.ny; ++J) {
        for (int I = 0; I < C.nx; ++I) {
            int cid = mgIdx(I, J, C.nx);
            float e = C.x[cid];

            int fi = 2 * I;
            int fj = 2 * J;
            for (int dj = 0; dj < 2; ++dj) {
                for (int di = 0; di < 2; ++di) {
                    int ii = fi + di;
                    int jj = fj + dj;
                    if (ii < F.nx && jj < F.ny) {
                        int fid = mgIdx(ii, jj, F.nx);
                        if (F.fluid[fid]) F.x[fid] += e;
                    }
                }
            }
        }
    }
}

void MACWater::mgVCycle(int lev) {
    if (lev == (int)mgLevels.size() - 1) {
        mgSmoothJacobi(lev, mgCoarseSmooth);
        return;
    }

    mgSmoothJacobi(lev, mgPreSmooth);
    mgComputeResidual(lev);
    mgRestrictResidual(lev);

    mgVCycle(lev + 1);

    mgProlongateAndAdd(lev + 1);
    mgSmoothJacobi(lev, mgPostSmooth);
}

void MACWater::mgRemoveMeanFine() {
    if (mgLevels.empty()) return;
    MGLevel& F = mgLevels[0];
    double sum = 0.0;
    int cnt = 0;
    const int N = F.nx * F.ny;
    for (int id = 0; id < N; ++id) {
        if (!F.fluid[id]) continue;
        sum += (double)F.x[id];
        cnt++;
    }
    if (cnt == 0) return;
    float mean = (float)(sum / (double)cnt);
    for (int id = 0; id < N; ++id) {
        if (!F.fluid[id]) continue;
        F.x[id] -= mean;
    }
}

void MACWater::solvePressureMGWater(int vcycles, float tol) {
    ensureWaterMG();
    if (mgLevels.empty()) return;

    MGLevel& F = mgLevels[0];
    const int N = F.nx * F.ny;

    auto maxAbsResidualFluid = [&]() -> float {
        float m = 0.0f;
        for (int id = 0; id < N; ++id) {
            if (!F.fluid[id]) continue;
            float a = std::fabs(F.r[id]);
            if (!std::isfinite(a)) return std::numeric_limits<float>::infinity();
            m = std::max(m, a);
        }
        return m;
    };

    auto maxAbsBFluid = [&]() -> float {
        float m = 0.0f;
        for (int id = 0; id < N; ++id) {
            if (!F.fluid[id]) continue;
            float a = std::fabs(F.b[id]);
            if (!std::isfinite(a)) return std::numeric_limits<float>::infinity();
            m = std::max(m, a);
        }
        return m;
    };

    if ((int)F.x.size() != N) F.x.assign(N, 0.0f);
    if ((int)F.b.size() != N) F.b.assign(N, 0.0f);

    for (int id = 0; id < N; ++id) {
        if (F.fluid[id]) {
            F.x[id] = p[(size_t)id];
            F.b[id] = rhs[(size_t)id];
        } else {
            F.x[id] = 0.0f;
            F.b[id] = 0.0f;
        }
    }

    const float bInf = maxAbsBFluid();
    if (bInf * dt <= tol) {
        if (!mgHasDirichlet) mgRemoveMeanFine();
        for (int id = 0; id < N; ++id) {
            p[(size_t)id] = F.fluid[id] ? F.x[id] : 0.0f;
        }
        return;
    }

    mgComputeResidual(0);
    float rInf = maxAbsResidualFluid();
    if (!std::isfinite(rInf) || rInf * dt <= tol) {
        if (!mgHasDirichlet) mgRemoveMeanFine();
        for (int id = 0; id < N; ++id) {
            p[(size_t)id] = F.fluid[id] ? F.x[id] : 0.0f;
        }
        return;
    }

    const int maxVCycles = std::max(1, vcycles);
    for (int k = 0; k < maxVCycles; ++k) {
        float prev = rInf;
        mgVCycle(0);
        mgComputeResidual(0);
        rInf = maxAbsResidualFluid();

        if (rInf > prev * 1.2f) {
            std::fill(F.x.begin(), F.x.end(), 0.0f);
            mgSmoothJacobi(0, 20);
            mgComputeResidual(0);
            break;
        }

        if (!mgHasDirichlet) mgRemoveMeanFine();

        mgComputeResidual(0);
        rInf = maxAbsResidualFluid();
        if (!std::isfinite(rInf) || rInf * dt <= tol) break;
    }

    if (!mgHasDirichlet) mgRemoveMeanFine();

    for (int id = 0; id < N; ++id) {
        p[(size_t)id] = F.fluid[id] ? F.x[id] : 0.0f;
    }
}

void MACWater::reseedParticlesFromField(const std::vector<float>& targetWater) {
    if (targetWater.size() != water.size()) return;
    if (targetMass <= liquidThreshold) return;

    const int maxAllowed = maxParticleLimit(maxParticles);
    const int desiredTotal = std::min(maxAllowed,
                                      std::max(1, (int)std::lround(targetMass * (float)particlesPerCell)));
    int missingTotal = desiredTotal - (int)particles.size();
    if (missingTotal <= 0) return;

    // Avoid pathological stalls when the requested particle count jumps a lot.
    const int maxSpawnPerStep = std::max(2048, nx * ny);
    const int spawnBudget = std::min(missingTotal, maxSpawnPerStep);
    if (spawnBudget <= 0) return;

    const int bt = std::max(1, borderThickness);
    const int iMin = std::min(bt, std::max(1, nx - 2));
    const int jMin = std::min(bt, std::max(1, ny - 2));
    const int iMax = std::max(iMin, nx - bt - 1);
    const int jMax = std::max(jMin, ny - bt - 1);

    std::vector<float> weights(water.size(), 0.0f);
    float sumW = 0.0f;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int id = idxP(i, j);
            if (solid[id]) continue;

            float w = std::max(0.0f, targetWater[id]);
            if (w < liquidThreshold && liquid[id]) w = liquidThreshold;
            if (w < liquidThreshold && id < (int)waterTarget.size()) {
                w = std::max(w, 0.25f * std::max(0.0f, waterTarget[id]));
            }

            weights[id] = w;
            sumW += w;
        }
    }

    if (sumW < 1e-6f) {
        for (int j = jMin; j <= jMax; ++j) {
            for (int i = iMin; i <= iMax; ++i) {
                int id = idxP(i, j);
                if (solid[id]) continue;
                weights[id] = 1.0f;
                sumW += 1.0f;
            }
        }
    }
    if (sumW < 1e-6f) return;

    std::vector<int> cdfIds;
    std::vector<float> cdf;
    cdfIds.reserve(weights.size());
    cdf.reserve(weights.size());
    float accum = 0.0f;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int id = idxP(i, j);
            float w = weights[id];
            if (w <= 0.0f) continue;
            accum += w;
            cdfIds.push_back(id);
            cdf.push_back(accum);
        }
    }
    if (cdf.empty()) return;

    std::uniform_real_distribution<float> pick(0.0f, accum);
    std::uniform_real_distribution<float> jitter(0.15f, 0.85f);

    for (int spawned = 0; spawned < spawnBudget && canSpawnMore(maxParticles, particles.size()); ++spawned) {
        float r = pick(globalRng());
        auto it = std::lower_bound(cdf.begin(), cdf.end(), r);
        int idx = (int)std::distance(cdf.begin(), it);
        if (idx < 0 || idx >= (int)cdfIds.size()) continue;
        int id = cdfIds[idx];
        int j = id / nx;
        int i = id - nx * j;

        Particle p;
        p.x = (i + jitter(globalRng())) * dx;
        p.y = (j + jitter(globalRng())) * dx;
        velAt(p.x, p.y, u, v, p.u, p.v);
        particles.push_back(p);
    }
}

void MACWater::rasterizeWaterField() {
    if (particles.empty()) {
        if (water0.size() == water.size()) water = water0;
        for (size_t k = 0; k < solid.size() && k < water.size(); ++k) {
            if (solid[k]) water[k] = 0.0f;
        }
        return;
    }

    std::fill(water.begin(), water.end(), 0.0f);

    std::vector<float> mass(water.size(), 0.0f);
    auto clampIndex = [](int v, int lo, int hi) {
        return std::max(lo, std::min(v, hi));
    };

    for (const Particle& p : particles) {
        // Bilinear splat to cell centers.
        float fx = p.x / dx - 0.5f;
        float fy = p.y / dx - 0.5f;

        int i0 = (int)std::floor(fx);
        int j0 = (int)std::floor(fy);
        float tx = fx - (float)i0;
        float ty = fy - (float)j0;

        i0 = clampIndex(i0, 0, nx - 1);
        j0 = clampIndex(j0, 0, ny - 1);
        int i1 = std::min(i0 + 1, nx - 1);
        int j1 = std::min(j0 + 1, ny - 1);

        tx = clampf(tx, 0.0f, 1.0f);
        ty = clampf(ty, 0.0f, 1.0f);

        const float w00 = (1.0f - tx) * (1.0f - ty);
        const float w10 = tx * (1.0f - ty);
        const float w01 = (1.0f - tx) * ty;
        const float w11 = tx * ty;

        const int id00 = idxP(i0, j0);
        const int id10 = idxP(i1, j0);
        const int id01 = idxP(i0, j1);
        const int id11 = idxP(i1, j1);

        // Renormalize to non-solid neighbors to avoid visual mass loss.
        float wSum = 0.0f;
        if (!solid[id00]) wSum += w00;
        if (!solid[id10]) wSum += w10;
        if (!solid[id01]) wSum += w01;
        if (!solid[id11]) wSum += w11;
        if (wSum <= 1e-8f) continue;

        const float inv = 1.0f / wSum;
        if (!solid[id00]) mass[(size_t)id00] += w00 * inv;
        if (!solid[id10]) mass[(size_t)id10] += w10 * inv;
        if (!solid[id01]) mass[(size_t)id01] += w01 * inv;
        if (!solid[id11]) mass[(size_t)id11] += w11 * inv;
    }

    const float denom = (float)std::max(1, particlesPerCell);
    std::vector<float> norm(water.size(), 0.0f);
    float sumNorm = 0.0f;
    for (size_t k = 0; k < water.size(); ++k) {
        if (solid[k]) continue;
        norm[k] = mass[k] / denom;
        sumNorm += norm[k];
    }

    float scale = 1.0f;
    if (targetMass > liquidThreshold && sumNorm > 1e-6f) {
        scale = targetMass / sumNorm;
    }

    const float cap = capOrInf(waterTargetMax);
    for (size_t k = 0; k < water.size(); ++k) {
        if (solid[k]) { water[k] = 0.0f; continue; }
        float val = norm[k] * scale;
        if (!std::isfinite(val)) val = 0.0f;
        water[k] = std::min(cap, val);
    }

    // Gentle, mass-conserving smoothing for rendering stability (avoids holes
    // near the pour without forcing bottom-up reconstruction).
    {
        const double diff = std::min(0.25, std::max(0.0, (double)columnDiffusion));
        const int iters = std::max(0, columnDiffusionIters);
        if (diff > 0.0 && iters > 0) {
            double totalBefore = 0.0;
            for (size_t k = 0; k < water.size(); ++k) {
                if (!solid[k]) totalBefore += (double)water[k];
            }

            auto sample = [&](int i, int j, double fallback) {
                if (i < 0 || i >= nx || j < 0 || j >= ny) return fallback;
                const int id = idxP(i, j);
                if (solid[(size_t)id]) return fallback;
                return (double)water[(size_t)id];
            };

            for (int it = 0; it < iters; ++it) {
                std::vector<float> next = water;
                for (int j = 0; j < ny; ++j) {
                    for (int i = 0; i < nx; ++i) {
                        const int id = idxP(i, j);
                        if (solid[(size_t)id]) continue;
                        const double center = (double)water[(size_t)id];

                        const double left  = sample(i - 1, j, center);
                        const double right = sample(i + 1, j, center);
                        const double down  = sample(i, j - 1, center);
                        const double up    = sample(i, j + 1, center);
                        const double avg = 0.25 * (left + right + down + up);

                        double blended = center + diff * (avg - center);
                        if (!std::isfinite(blended)) blended = center;
                        if (blended < 0.0) blended = 0.0;
                        next[(size_t)id] = (float)blended;
                    }
                }
                water.swap(next);
            }

            double totalAfter = 0.0;
            for (size_t k = 0; k < water.size(); ++k) {
                if (!solid[k]) totalAfter += (double)water[k];
            }
            if (totalAfter > 1e-12 && totalBefore > 0.0) {
                const double corr = totalBefore / totalAfter;
                for (size_t k = 0; k < water.size(); ++k) {
                    if (solid[k]) { water[k] = 0.0f; continue; }
                    water[k] = (float)((double)water[k] * corr);
                }
            }
        }
    }

    // Height reconstruction: preserve per-column mass but cap per-cell values,
    // turning overly compressed puddles into a visible water height.
    if (maxWaterPerCell > 0.0f) {
        const float cellCap = maxWaterPerCell;
        const int bt = std::max(1, borderThickness);
        const int iMin = std::min(bt, std::max(1, nx - 2));
        const int jMin = std::min(bt, std::max(1, ny - 2));
        const int iMax = std::max(iMin, nx - bt - 1);
        const int jMax = std::max(jMin, ny - bt - 1);

        // 1) Compute per-column mass and validity.
        std::vector<double> massCol((size_t)nx, 0.0);
        std::vector<uint8_t> colValid((size_t)nx, (uint8_t)0);
        double totalMass = 0.0;

        for (int i = iMin; i <= iMax; ++i) {
            double m = 0.0;
            bool valid = false;
            for (int j = jMin; j <= jMax; ++j) {
                const int id = idxP(i, j);
                if (solid[(size_t)id]) continue;
                m += (double)water[(size_t)id];
                valid = true;
            }
            massCol[(size_t)i] = m;
            colValid[(size_t)i] = valid ? (uint8_t)1 : (uint8_t)0;
            totalMass += m;
        }

        // 2) Diffuse per-column mass laterally (mass-conserving).
        std::vector<double> massColSmooth = massCol;
        const double diff = std::min(0.49, std::max(0.0, (double)columnDiffusion));
        const int diffIters = std::max(0, columnDiffusionIters);

        for (int it = 0; it < diffIters && diff > 0.0; ++it) {
            std::vector<double> next = massColSmooth;
            for (int i = iMin; i <= iMax; ++i) {
                if (!colValid[(size_t)i]) { next[(size_t)i] = 0.0; continue; }

                const double m = massColSmooth[(size_t)i];
                auto neighMass = [&](int ii) {
                    if (ii < iMin || ii > iMax) return m;
                    if (!colValid[(size_t)ii]) return m;
                    return massColSmooth[(size_t)ii];
                };

                const double left = neighMass(i - 1);
                const double right = neighMass(i + 1);
                double blended = m + diff * (left - m) + diff * (right - m);
                if (!std::isfinite(blended)) blended = 0.0;
                next[(size_t)i] = blended;
            }
            massColSmooth.swap(next);
        }

        // Renormalize to preserve total mass exactly.
        double totalSmooth = 0.0;
        for (int i = iMin; i <= iMax; ++i) totalSmooth += massColSmooth[(size_t)i];
        if (totalSmooth > 1e-12 && totalMass > 0.0) {
            const double corr = totalMass / totalSmooth;
            for (int i = iMin; i <= iMax; ++i) massColSmooth[(size_t)i] *= corr;
        }

        // 3) Clear water in the reconstruction region.
        for (int j = jMin; j <= jMax; ++j) {
            for (int i = iMin; i <= iMax; ++i) {
                const int id = idxP(i, j);
                if (!solid[(size_t)id]) water[(size_t)id] = 0.0f;
            }
        }

        // 4) Reconstruct a bottom-up height field from smoothed column mass.
        for (int i = iMin; i <= iMax; ++i) {
            double massRemaining = massColSmooth[(size_t)i];
            if (massRemaining <= 0.0 || !colValid[(size_t)i]) continue;

            int lastFilledId = -1;
            for (int j = jMin; j <= jMax && massRemaining > 0.0; ++j) {
                const int id = idxP(i, j);
                if (solid[(size_t)id]) continue;
                const float fill = (float)std::min<double>((double)cellCap, massRemaining);
                water[(size_t)id] = fill;
                massRemaining -= (double)fill;
                lastFilledId = id;
            }

            // Preserve mass even if we exceed nominal column capacity.
            if (massRemaining > 0.0 && lastFilledId >= 0) {
                water[(size_t)lastFilledId] += (float)massRemaining;
            }
        }
    }
}

void MACWater::removeParticlesInSolids() {
    const int bt = std::max(1, borderThickness);
    const int iMin = std::min(bt, std::max(1, nx - 2));
    const int jMin = std::min(bt, std::max(1, ny - 2));
    const int iMax = std::max(iMin, nx - bt - 1);
    const int jMax = std::max(jMin, ny - bt - 1);

    size_t write = 0;
    for (size_t read = 0; read < particles.size(); ++read) {
        Particle p = particles[read];
        int i, j;
        worldToCell(p.x, p.y, i, j);
        if (solid[idxP(i, j)]) {
            bool placed = false;
            const int baseI = std::max(iMin, std::min(iMax, i));
            const int baseJ = std::max(jMin, std::min(jMax, j));

            auto placeInCell = [&](int ii, int jj) {
                ii = std::max(iMin, std::min(iMax, ii));
                jj = std::max(jMin, std::min(jMax, jj));
                p.x = (ii + 0.5f) * dx;
                p.y = (jj + 0.5f) * dx;
                if (jj <= jMin) p.v = std::max(0.0f, p.v);
                placed = true;
            };

            // Push upward first (good for the bottom wall).
            int jj = baseJ;
            while (jj <= jMax && solid[idxP(baseI, jj)]) ++jj;
            if (jj <= jMax && !solid[idxP(baseI, jj)]) {
                placeInCell(baseI, jj);
            }

            // If the whole column is solid, search a small neighborhood.
            for (int rad = 1; !placed && rad <= 4; ++rad) {
                float bestDist2 = 1e9f;
                int bestI = -1, bestJ = -1;
                int i0 = std::max(iMin, baseI - rad);
                int i1 = std::min(iMax, baseI + rad);
                int j0 = std::max(jMin, baseJ - rad);
                int j1 = std::min(jMax, baseJ + rad);

                for (int jj2 = j0; jj2 <= j1; ++jj2) {
                    for (int ii2 = i0; ii2 <= i1; ++ii2) {
                        if (solid[idxP(ii2, jj2)]) continue;
                        float dx0 = (float)(ii2 - baseI);
                        float dy0 = (float)(jj2 - baseJ);
                        float d2 = dx0 * dx0 + dy0 * dy0;
                        if (d2 < bestDist2) {
                            bestDist2 = d2;
                            bestI = ii2;
                            bestJ = jj2;
                        }
                    }
                }

                if (bestI >= 0) placeInCell(bestI, bestJ);
            }

            // Last resort: place into the first non-solid cell we can find.
            if (!placed) {
                for (int jj2 = jMin; jj2 <= jMax && !placed; ++jj2) {
                    for (int ii2 = iMin; ii2 <= iMax; ++ii2) {
                        if (!solid[idxP(ii2, jj2)]) placeInCell(ii2, jj2);
                    }
                }
            }

            if (!placed) continue;
        }
        particles[write++] = p;
    }
    particles.resize(write);
}

void MACWater::enforceParticleBounds() {
    const int bt = std::max(1, borderThickness);
    float minX = (bt + 0.5f) * dx;
    float maxX = (nx - bt - 0.5f) * dx;
    float minY = (bt + 0.5f) * dx;
    float maxYClosed = (ny - bt - 0.5f) * dx;
    const float maxYOpen = (ny - 0.5f) * dx;

    if (maxX <= minX) { minX = 1.5f * dx; maxX = (nx - 1.5f) * dx; }
    if (maxYClosed <= minY) { minY = 1.5f * dx; maxYClosed = (ny - 1.5f) * dx; }

    size_t write = 0;
    for (size_t read = 0; read < particles.size(); ++read) {
        Particle p = particles[read];

        if (openTop && p.y > maxYOpen) { p.y = maxYOpen; p.v = 0.0f; }

        if (p.x < minX) { p.x = minX; p.u = 0.0f; }
        if (p.x > maxX) { p.x = maxX; p.u = 0.0f; }
        if (p.y < minY) { p.y = minY; p.v = 0.0f; }
        if (!openTop && p.y > maxYClosed) { p.y = maxYClosed; p.v = 0.0f; }

        particles[write++] = p;
    }
    particles.resize(write);
}

void MACWater::step() {
    stepCounter += 1;
    for (size_t k = 0; k < solid.size() && k < waterTarget.size(); ++k) {
        if (solid[k]) {
            targetMass -= waterTarget[k];
            waterTarget[k] = 0.0f;
        }
    }
    const float decay = std::max(0.0f, std::min(1.0f, waterTargetDecay));
    const float targetCap = capOrInf(waterTargetMax);
    for (size_t k = 0; k < waterTarget.size() && k < solid.size(); ++k) {
        if (solid[k]) continue;
        waterTarget[k] = std::min(targetCap, waterTarget[k] * decay);
    }
    targetMass = std::max(0.0f, targetMass);
    water0 = water;

    if (particles.empty()) {
        reseedParticlesFromField(water0);
        if (particles.empty()) {
            applyBoundary();
            water = water0;
            return;
        }
        applyBoundary();
    }

    applyBoundary();
    removeParticlesInSolids();
    enforceParticleBounds();

    advectParticles();
    enforceParticleBounds();
    removeParticlesInSolids();
    separateParticles();
    relaxParticleDensity();
    relaxColumnDensity();
    reseedParticlesFromField(water0);

    particleToGrid();
    buildLiquidMask();
    extrapolateVelocity();

    // gravity only near liquid faces
    auto isLiquidCell = [&](int i, int j) {
        if (i < 0 || i >= nx || j < 0 || j >= ny) return false;
        int id = idxP(i, j);
        return !solid[id] && liquid[id];
    };

    for (int j = 0; j <= ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            if (!(isLiquidCell(i, j - 1) || isLiquidCell(i, j))) continue;
            v[idxV(i, j)] += dt * waterGravity;
        }
    }

    if (velDamping > 0.0f) {
        const float factor = std::exp(-velDamping * dt);
        for (float& uu : u) uu *= factor;
        for (float& vv : v) vv *= factor;
    }

    applyHeightPressureForce();
    applyViscosity();
    applyBoundary();

    // store pre-projection grid for FLIP delta
    uPrev = u;
    vPrev = v;

    const int repeats = std::max(1, pressureRepeats);
    for (int r = 0; r < repeats; ++r) {
        projectLiquid();
        extrapolateVelocity();
        applyBoundary();
    }
    // Remove any residual drift, clamp boundaries, then do a final projection.
    removeLiquidDrift();
    applyBoundary();
    projectLiquid();
    applyBoundary();

    for (size_t i = 0; i < u.size(); ++i) uDelta[i] = u[i] - uPrev[i];
    for (size_t i = 0; i < v.size(); ++i) vDelta[i] = v[i] - vPrev[i];

    gridToParticles();
    enforceParticleBounds();
    removeParticlesInSolids();

    if (restVelocity > 0.0f) {
        const float restVel = restVelocity * dx;

        auto isLiquidCell = [&](int i, int j) {
            if (i < 0 || i >= nx || j < 0 || j >= ny) return false;
            const int id = idxP(i, j);
            return !solid[(size_t)id] && liquid[(size_t)id];
        };
        auto liquidAdjacentU = [&](int i, int j) {
            return isLiquidCell(i - 1, j) || isLiquidCell(i, j);
        };
        auto liquidAdjacentV = [&](int i, int j) {
            return isLiquidCell(i, j - 1) || isLiquidCell(i, j);
        };

        float maxFace = 0.0f;
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i <= nx; ++i) {
                if (!liquidAdjacentU(i, j)) continue;
                const float s = std::fabs(u[(size_t)idxU(i, j)]);
                if (s > maxFace) maxFace = s;
            }
        }
        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                if (!liquidAdjacentV(i, j)) continue;
                const float s = std::fabs(v[(size_t)idxV(i, j)]);
                if (s > maxFace) maxFace = s;
            }
        }

        const float maxPart = maxParticleSpeed();
        if (maxFace < restVel && maxPart < restVel) {
            snapToRest(restVel);
        } else if (maxFace < restVel) {
            // Grid is nearly at rest; damp any straggler particle velocities toward PIC.
            const float restVel2 = restVel * restVel;
            for (Particle& p : particles) {
                float picU, picV;
                velAt(p.x, p.y, u, v, picU, picV);
                const float picS2 = picU * picU + picV * picV;
                if (picS2 < restVel2) {
                    p.u = 0.0f;
                    p.v = 0.0f;
                } else {
                    p.u = picU;
                    p.v = picV;
                }
            }
        }
    }

    rasterizeWaterField();
}
