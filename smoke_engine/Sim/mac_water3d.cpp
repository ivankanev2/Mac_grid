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

MACWater3DCudaBackend* MACWater3D::activeCudaBackend() const {
    if (cudaBackend == nullptr) return nullptr;
    if (backendPreference == BackendPreference::CPU) return nullptr;
    return cudaBackend;
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
    validUNext.assign((std::size_t)uCount, (uint8_t)0);
    validVNext.assign((std::size_t)vCount, (uint8_t)0);
    validWNext.assign((std::size_t)wCount, (uint8_t)0);
    extrapFrontierU.clear();
    extrapFrontierV.clear();
    extrapFrontierW.clear();
    extrapNextFrontierU.clear();
    extrapNextFrontierV.clear();
    extrapNextFrontierW.clear();

    uDiffusionStencil.clear();
    vDiffusionStencil.clear();
    wDiffusionStencil.clear();
    uDiffusionScratch.r.clear();
    uDiffusionScratch.z.clear();
    uDiffusionScratch.p.clear();
    uDiffusionScratch.q.clear();
    vDiffusionScratch.r.clear();
    vDiffusionScratch.z.clear();
    vDiffusionScratch.p.clear();
    vDiffusionScratch.q.clear();
    wDiffusionScratch.r.clear();
    wDiffusionScratch.z.clear();
    wDiffusionScratch.p.clear();
    wDiffusionScratch.q.clear();
    reseedCounts.assign((std::size_t)cellCount, 0);
    reseedOccupied.assign((std::size_t)cellCount, (uint8_t)0);
    reseedRegion.assign((std::size_t)cellCount, (uint8_t)0);
    relaxBucketCounts.assign((std::size_t)cellCount, 0);
    relaxBucketOffsets.assign((std::size_t)cellCount + 1u, 0);
    relaxBucketCursor.assign((std::size_t)cellCount, 0);
    relaxBucketParticles.clear();
    pressureComponentLabel.assign((std::size_t)cellCount, -1);
    pressureComponentQueue.clear();
    pressureComponentCells.clear();
    diffusionStencilDirty = true;

    particles.clear();
    stepCounter = 0;
    targetMass = 0.0f;
    desiredMass = -1.0f;
    topologyDirty = true;
    pressureRegion.clear();

    rebuildBorderSolids();
    applyBoundary();
    buildLiquidMask();
    rasterizeWaterField();
    updateStats(0.0f);

    if (MACWater3DCudaBackend* backend = activeCudaBackend()) {
        water3dCudaReset(backend, *this);
        derivedFieldsDirty = false;
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
    diffusionStencilDirty = true;

    if (MACWater3DCudaBackend* backend = activeCudaBackend()) {
        water3dCudaSetParams(backend, *this);
        derivedFieldsDirty = false;
        return;
    }

    rebuildBorderSolids();
    removeParticlesInSolids();
    enforceParticleBounds();
    buildLiquidMask();
    rasterizeWaterField();
    updateStats(0.0f);
}

void MACWater3D::refreshStats(float stepMs) {
    derivedFieldsDirty = false;
    updateStats(stepMs);
    lastStats.pressureMs = 0.0f;
    lastStats.pressureIters = 0;
    lastStats.timings.reset();
    lastStats.timings.totalMs = stepMs;
}

void MACWater3D::setBackendPreference(BackendPreference newPreference) {
    backendPreference = newPreference;

    if (MACWater3DCudaBackend* backend = activeCudaBackend()) {
        water3dCudaReset(backend, *this);
    } else {
        updateStats(0.0f);
    }
}

void MACWater3D::step() {
    if (MACWater3DCudaBackend* backend = activeCudaBackend()) {
        water3dCudaStep(backend, *this);
        derivedFieldsDirty = false;
        lastStats.pressureMs = 0.0f;
        lastStats.pressureIters = 0;
        lastStats.timings.reset();
        lastStats.timings.totalMs = lastStats.lastStepMs;
        return;
    }

    using clock = std::chrono::high_resolution_clock;
    const auto frameStart = clock::now();
    auto stageStart = frameStart;
    SimStageTimings timings;
    lastPressureSolveMs = 0.0f;
    lastPressureIterations = 0;
    ++stepCounter;

    auto markStage = [&](float& bucket) {
        const auto now = clock::now();
        bucket += std::chrono::duration<float, std::milli>(now - stageStart).count();
        stageStart = now;
    };

    if (topologyDirty) rebuildBorderSolids();
    removeParticlesInSolids();
    enforceParticleBounds();
    markStage(timings.setupMs);

    if (particles.empty()) {
        std::fill(u.begin(), u.end(), 0.0f);
        std::fill(v.begin(), v.end(), 0.0f);
        std::fill(w.begin(), w.end(), 0.0f);
        buildLiquidMask();
        markStage(timings.liquidMaskMs);
        applyBoundary();
        markStage(timings.boundaryMs);
        rasterizeWaterField();
        markStage(timings.rasterizeMs);

        auto statsStart = clock::now();
        updateStats(0.0f);
        auto statsEnd = clock::now();
        timings.statsMs += std::chrono::duration<float, std::milli>(statsEnd - statsStart).count();

        const float stepMs = std::chrono::duration<float, std::milli>(statsEnd - frameStart).count();
        lastStats.lastStepMs = stepMs;
        lastStats.pressureMs = 0.0f;
        lastStats.pressureIters = 0;
        lastStats.timings = timings;
        lastStats.timings.totalMs = stepMs;
        return;
    }

    particleToGrid();
    markStage(timings.particleToGridMs);

    buildLiquidMask();
    markStage(timings.liquidMaskMs);

    applyExternalForces();
    markStage(timings.forcesMs);

    applyBoundary();
    markStage(timings.boundaryMs);

    diffuseVelocityImplicit();
    markStage(timings.diffuseVelocityMs);

    applyBoundary();
    markStage(timings.boundaryMs);

    uPrev = u;
    vPrev = v;
    wPrev = w;
    markStage(timings.setupMs);

    projectLiquid();
    markStage(timings.projectMs);

    applyBoundary();
    markStage(timings.boundaryMs);

    for (std::size_t i = 0; i < u.size(); ++i) uDelta[i] = u[i] - uPrev[i];
    for (std::size_t i = 0; i < v.size(); ++i) vDelta[i] = v[i] - vPrev[i];
    for (std::size_t i = 0; i < w.size(); ++i) wDelta[i] = w[i] - wPrev[i];
    markStage(timings.setupMs);

    extrapolateVelocity();
    markStage(timings.extrapolateMs);

    applyBoundary();
    markStage(timings.boundaryMs);

    gridToParticles();
    markStage(timings.gridToParticlesMs);

    advectParticles();
    enforceParticleBounds();
    removeParticlesInSolids();
    markStage(timings.advectParticlesMs);

    buildLiquidMask(false);
    markStage(timings.liquidMaskMs);

    reseedParticles();
    if (params.reseedRelaxIters > 0 && params.reseedRelaxStrength > 0.0f) {
        relaxParticles(params.reseedRelaxIters, params.reseedRelaxStrength);
    }
    applyDissipation();
    markStage(timings.reseedMs);

    buildLiquidMask();
    markStage(timings.liquidMaskMs);

    rasterizeWaterField();
    markStage(timings.rasterizeMs);

    auto statsStart = clock::now();
    updateStats(0.0f);
    auto statsEnd = clock::now();
    timings.statsMs += std::chrono::duration<float, std::milli>(statsEnd - statsStart).count();

    const float stepMs = std::chrono::duration<float, std::milli>(statsEnd - frameStart).count();
    lastStats.lastStepMs = stepMs;
    lastStats.pressureMs = lastPressureSolveMs;
    lastStats.pressureIters = lastPressureIterations;
    lastStats.timings = timings;
    lastStats.timings.totalMs = stepMs;
}


void MACWater3D::addWaterSourceSphere(const Vec3& center, float radius, const Vec3& velocity) {
    if (MACWater3DCudaBackend* backend = activeCudaBackend()) {
        water3dCudaAddWaterSourceSphere(backend, *this, center, radius, velocity);
        derivedFieldsDirty = false;
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
                    p.c00 = p.c01 = p.c02 = 0.0f;
                    p.c10 = p.c11 = p.c12 = 0.0f;
                    p.c20 = p.c21 = p.c22 = 0.0f;
                    particles.push_back(p);

                    if (desiredMass >= 0.0f) {
                        desiredMass += 1.0f;
                    }

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
    rasterizeWaterField();
    updateStats(0.0f);
}

void MACWater3D::setVoxelSolids(const std::vector<uint8_t>& mask) {
    if (MACWater3DCudaBackend* backend = activeCudaBackend()) {
        solidUser = mask;
        water3dCudaSetVoxelSolids(backend, *this);
        return;
    }

    const int cellCount = nx * ny * nz;
    solidUser.assign((std::size_t)cellCount, (uint8_t)0);

    if ((int)mask.size() == cellCount) {
        solidUser = mask;
    }

    topologyDirty = true;
    diffusionStencilDirty = true;
    rebuildBorderSolids();
    removeParticlesInSolids();
    enforceParticleBounds();
    buildLiquidMask();
    applyBoundary();
    rasterizeWaterField();
    updateStats(0.0f);
}

MACWater3D::SliceData MACWater3D::copyDebugSlice(SliceAxis axis, int index, DebugField field) {
    const std::vector<float>* src = &water;
    switch (field) {
        case DebugField::Water:      src = &water; break;
        case DebugField::Pressure:   src = &pressure; break;
        case DebugField::Divergence:
            ensureDerivedDebugFields();
            src = &divergence;
            break;
        case DebugField::Speed:
            ensureDerivedDebugFields();
            src = &speed;
            break;
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
