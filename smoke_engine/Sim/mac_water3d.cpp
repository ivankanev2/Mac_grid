#include "mac_water3d.h"

#include <chrono>

#include "Water3D/water3d_common.h"
#include "Water3D/water3d_solids.h"
#include "Water3D/water3d_particles.h"
#include "Water3D/water3d_grid.h"
#include "Water3D/water3d_pressure.h"
#include "Water3D/water3d_fields.h"
#include "Water3D/water3d_slices.h"
#include "Water3D/water3d_cuda_backend.h"

MACWater3D::MACWater3D(int NX, int NY, int NZ, float DX, float DT)
    : nx(NX), ny(NY), nz(NZ), dx(DX), dt(DT) {
#if SMOKE_ENABLE_CUDA
    cudaBackend = water3dCreateCudaBackend();
#endif
    reset();
}

MACWater3D::~MACWater3D() {
    if (cudaBackend != nullptr) {
        water3dDestroyCudaBackend(cudaBackend);
        cudaBackend = nullptr;
    }
}

void MACWater3D::reset() {
    const int cellCount = std::max(1, nx * ny * nz);
    const int uCount = std::max(1, (nx + 1) * ny * nz);
    const int vCount = std::max(1, nx * (ny + 1) * nz);
    const int wCount = std::max(1, nx * ny * (nz + 1));

    water.assign((std::size_t)cellCount, 0.0f);
    pressure.assign((std::size_t)cellCount, 0.0f);
    pressureTmp.assign((std::size_t)cellCount, 0.0f);
    rhs.assign((std::size_t)cellCount, 0.0f);
    divergence.assign((std::size_t)cellCount, 0.0f);
    speed.assign((std::size_t)cellCount, 0.0f);

    liquid.assign((std::size_t)cellCount, (uint8_t)0);
    solid.assign((std::size_t)cellCount, (uint8_t)0);
    solidUser.assign((std::size_t)cellCount, (uint8_t)0);

    u.assign((std::size_t)uCount, 0.0f);
    v.assign((std::size_t)vCount, 0.0f);
    w.assign((std::size_t)wCount, 0.0f);

    uWeight.assign((std::size_t)uCount, 0.0f);
    vWeight.assign((std::size_t)vCount, 0.0f);
    wWeight.assign((std::size_t)wCount, 0.0f);

    uPrev.assign((std::size_t)uCount, 0.0f);
    vPrev.assign((std::size_t)vCount, 0.0f);
    wPrev.assign((std::size_t)wCount, 0.0f);

    uDelta.assign((std::size_t)uCount, 0.0f);
    vDelta.assign((std::size_t)vCount, 0.0f);
    wDelta.assign((std::size_t)wCount, 0.0f);

    uTmp.assign((std::size_t)uCount, 0.0f);
    vTmp.assign((std::size_t)vCount, 0.0f);
    wTmp.assign((std::size_t)wCount, 0.0f);

    validU.assign((std::size_t)uCount, (uint8_t)0);
    validV.assign((std::size_t)vCount, (uint8_t)0);
    validW.assign((std::size_t)wCount, (uint8_t)0);

    particles.clear();
    stepCounter = 0;
    targetMass = 0.0f;
    desiredMass = -1.0f;

    rebuildBorderSolids();
    applyBoundary();
    buildLiquidMask();
    rasterizeDebugFields();
    updateStats(0.0f);

    if (cudaBackend != nullptr) {
        water3dCudaReset(cudaBackend, *this);
    }
}

void MACWater3D::reset(int NX, int NY, int NZ, float DX, float DT) {
    nx = NX;
    ny = NY;
    nz = NZ;
    dx = DX;
    dt = DT;
    reset();
}

void MACWater3D::setParams(const Params& newParams) {
    params = newParams;

    if (cudaBackend != nullptr) {
        water3dCudaSetParams(cudaBackend, *this);
        return;
    }

    rebuildBorderSolids();
    removeParticlesInSolids();
    enforceParticleBounds();
    buildLiquidMask();
    rasterizeDebugFields();
    updateStats(0.0f);
}

void MACWater3D::step() {
    if (cudaBackend != nullptr) {
        water3dCudaStep(cudaBackend, *this);
        return;
    }

    const auto start = std::chrono::high_resolution_clock::now();
    ++stepCounter;

    rebuildBorderSolids();
    removeParticlesInSolids();
    enforceParticleBounds();

    if (particles.empty()) {
        std::fill(u.begin(), u.end(), 0.0f);
        std::fill(v.begin(), v.end(), 0.0f);
        std::fill(w.begin(), w.end(), 0.0f);
        buildLiquidMask();
        applyBoundary();
        rasterizeDebugFields();
        const auto end = std::chrono::high_resolution_clock::now();
        const float stepMs = std::chrono::duration<float, std::milli>(end - start).count();
        updateStats(stepMs);
        return;
    }

    particleToGrid();
    buildLiquidMask();

    applyExternalForces();
    applyBoundary();

    diffuseVelocityImplicit();
    applyBoundary();

    uPrev = u;
    vPrev = v;
    wPrev = w;

    projectLiquid();
    applyBoundary();

    for (std::size_t i = 0; i < u.size(); ++i) uDelta[i] = u[i] - uPrev[i];
    for (std::size_t i = 0; i < v.size(); ++i) vDelta[i] = v[i] - vPrev[i];
    for (std::size_t i = 0; i < w.size(); ++i) wDelta[i] = w[i] - wPrev[i];

    extrapolateVelocity();
    applyBoundary();

    gridToParticles();
    advectParticles();
    enforceParticleBounds();
    removeParticlesInSolids();

    buildLiquidMask(false);
    reseedParticles();
    applyDissipation();

    buildLiquidMask();
    rasterizeDebugFields();

    const auto end = std::chrono::high_resolution_clock::now();
    const float stepMs = std::chrono::duration<float, std::milli>(end - start).count();
    updateStats(stepMs);
}

void MACWater3D::addWaterSourceSphere(const Vec3& center, float radius, const Vec3& velocity) {
    if (cudaBackend != nullptr) {
        water3dCudaAddWaterSourceSphere(cudaBackend, *this, center, radius, velocity);
        return;
    }

    rebuildBorderSolids();

    const float r = std::max(0.0f, radius);
    if (r <= 0.0f || params.particlesPerCell <= 0) return;

    const int minI = water3d_internal::clampi((int)std::floor((center.x - r) / dx), 0, nx - 1);
    const int maxI = water3d_internal::clampi((int)std::floor((center.x + r) / dx), 0, nx - 1);
    const int minJ = water3d_internal::clampi((int)std::floor((center.y - r) / dx), 0, ny - 1);
    const int maxJ = water3d_internal::clampi((int)std::floor((center.y + r) / dx), 0, ny - 1);
    const int minK = water3d_internal::clampi((int)std::floor((center.z - r) / dx), 0, nz - 1);
    const int maxK = water3d_internal::clampi((int)std::floor((center.z + r) / dx), 0, nz - 1);

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

                liquid[(std::size_t)id] = 1;

                const int spawnPerCell = std::max(1, params.particlesPerCell);
                const int canSpawn = (params.maxParticles > 0)
                    ? std::max(0, params.maxParticles - (int)particles.size())
                    : spawnPerCell;
                const int spawnCount = std::min(spawnPerCell, canSpawn);

                for (int n = 0; n < spawnCount; ++n) {
                    uint32_t seed =
                        (uint32_t)(i + 73856093U * j + 19349663U * k + 83492791U * (n + 1) + stepCounter * 131U);
                    Particle p;
                    p.x = (i + 0.1f + 0.8f * water3d_internal::rand01(seed)) * dx;
                    p.y = (j + 0.1f + 0.8f * water3d_internal::rand01(seed)) * dx;
                    p.z = (k + 0.1f + 0.8f * water3d_internal::rand01(seed)) * dx;
                    p.u = velocity.x;
                    p.v = velocity.y;
                    p.w = velocity.z;
                    particles.push_back(p);

                    if (params.maxParticles > 0 && (int)particles.size() >= params.maxParticles) {
                        break;
                    }
                }
            }
        }
    }

    enforceParticleBounds();
    removeParticlesInSolids();
    buildLiquidMask();
    rasterizeDebugFields();
    updateStats(0.0f);
}

void MACWater3D::setVoxelSolids(const std::vector<uint8_t>& mask) {
    if (cudaBackend != nullptr) {
        solidUser = mask;
        water3dCudaSetVoxelSolids(cudaBackend, *this);
        return;
    }

    const int cellCount = nx * ny * nz;
    solidUser.assign((std::size_t)cellCount, (uint8_t)0);

    if ((int)mask.size() == cellCount) {
        solidUser = mask;
    }

    rebuildBorderSolids();
    removeParticlesInSolids();
    enforceParticleBounds();
    buildLiquidMask();
    applyBoundary();
    rasterizeDebugFields();
    updateStats(0.0f);
}

MACWater3D::SliceData MACWater3D::copyDebugSlice(SliceAxis axis, int index, DebugField field) const {
    const std::vector<float>* src = &water;
    switch (field) {
        case DebugField::Water:      src = &water; break;
        case DebugField::Pressure:   src = &pressure; break;
        case DebugField::Divergence: src = &divergence; break;
        case DebugField::Speed:      src = &speed; break;
    }

    SliceData out;
    if (nx <= 0 || ny <= 0 || nz <= 0 || src->empty()) return out;

    if (axis == SliceAxis::XY) {
        const int kk = water3d_internal::clampi(index, 0, nz - 1);
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
        const int jj = water3d_internal::clampi(index, 0, ny - 1);
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

    const int ii = water3d_internal::clampi(index, 0, nx - 1);
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
