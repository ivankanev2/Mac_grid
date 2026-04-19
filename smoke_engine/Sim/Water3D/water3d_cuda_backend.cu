#if SMOKE_ENABLE_CUDA

#include "water3d_cuda_backend.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <cuda/functional>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/remove.h>
#include <thrust/transform_reduce.h>

// Runtime-owned CUDA buffers and cached simulation dimensions.
struct MACWater3DCudaBackend {
    int nx = 0;
    int ny = 0;
    int nz = 0;
    float dx = 1.0f;
    float dt = 0.02f;
    MACWater3D::Params params;

    int cellCount = 0;
    int uCount = 0;
    int vCount = 0;
    int wCount = 0;
    int particleCapacity = 0;

    float* d_u = nullptr;
    float* d_v = nullptr;
    float* d_w = nullptr;
    float* d_uWeight = nullptr;
    float* d_vWeight = nullptr;
    float* d_wWeight = nullptr;
    float* d_uPrev = nullptr;
    float* d_vPrev = nullptr;
    float* d_wPrev = nullptr;
    float* d_uDelta = nullptr;
    float* d_vDelta = nullptr;
    float* d_wDelta = nullptr;
    float* d_uTmp = nullptr;
    float* d_vTmp = nullptr;
    float* d_wTmp = nullptr;
    float* d_pressure = nullptr;
    float* d_pressureTmp = nullptr;
    float* d_rhs = nullptr;
    float* d_water = nullptr;
    float* d_divergence = nullptr;
    float* d_speed = nullptr;
    float* d_mass = nullptr;

    uint8_t* d_liquid = nullptr;
    uint8_t* d_liquidTmp = nullptr;
    uint8_t* d_solid = nullptr;
    uint8_t* d_solidUser = nullptr;
    uint8_t* d_validU = nullptr;
    uint8_t* d_validV = nullptr;
    uint8_t* d_validW = nullptr;
    uint8_t* d_validUTmp = nullptr;
    uint8_t* d_validVTmp = nullptr;
    uint8_t* d_validWTmp = nullptr;
    uint8_t* d_occ = nullptr;
    uint8_t* d_region = nullptr;

    int* d_cellCounts = nullptr;
    int* d_particleCount = nullptr;
    MACWater3D::Particle* d_particles = nullptr;

    bool lastUsedDeviceJacobi = false;
};

namespace {

#define CUDA_CHECK(expr)                                                         \
    do {                                                                         \
        cudaError_t err__ = (expr);                                              \
        if (err__ != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,   \
                         cudaGetErrorString(err__));                             \
            std::abort();                                                        \
        }                                                                        \
    } while (0)

constexpr int kThreads = 256;

template <typename T>
__global__ void fillKernel(T* data, int count, T value) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    data[idx] = value;
}

__host__ __device__ inline int idxCell3D(int i, int j, int k, int nx, int ny) {
    return i + nx * (j + ny * k);
}

__host__ __device__ inline int idxU3D(int i, int j, int k, int nx, int ny) {
    return i + (nx + 1) * (j + ny * k);
}

__host__ __device__ inline int idxV3D(int i, int j, int k, int nx, int ny) {
    return i + nx * (j + (ny + 1) * k);
}

__host__ __device__ inline int idxW3D(int i, int j, int k, int nx, int ny) {
    return i + nx * (j + ny * k);
}

__host__ __device__ inline int clampi3D(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__host__ __device__ inline float clampf3D(float v, float lo, float hi) {
    if (!::isfinite(v)) return lo;
    return v < lo ? lo : (v > hi ? hi : v);
}

__host__ __device__ inline float clamp013D(float v) {
    return clampf3D(v, 0.0f, 1.0f);
}

__host__ __device__ inline unsigned int hashU32(unsigned int x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

__host__ __device__ inline float rand01FromSeed(unsigned int& seed) {
    seed = hashU32(seed);
    return (float)(seed & 0x00FFFFFFU) / (float)0x01000000U;
}

struct DeviceParticleDead {
    __host__ __device__ bool operator()(const MACWater3D::Particle& p) const {
        return p.age < 0.0f;
    }
};

__device__ float sampleCellCenteredDevice(const float* field, float x, float y, float z,
                                          int nx, int ny, int nz, float dx) {
    const float fx = x / dx - 0.5f;
    const float fy = y / dx - 0.5f;
    const float fz = z / dx - 0.5f;

    int i0 = clampi3D((int)floorf(fx), 0, nx - 1);
    int j0 = clampi3D((int)floorf(fy), 0, ny - 1);
    int k0 = clampi3D((int)floorf(fz), 0, nz - 1);
    int i1 = min(i0 + 1, nx - 1);
    int j1 = min(j0 + 1, ny - 1);
    int k1 = min(k0 + 1, nz - 1);

    const float tx = clampf3D(fx - (float)i0, 0.0f, 1.0f);
    const float ty = clampf3D(fy - (float)j0, 0.0f, 1.0f);
    const float tz = clampf3D(fz - (float)k0, 0.0f, 1.0f);

    float result = 0.0f;
    for (int dk = 0; dk < 2; ++dk) {
        const int kk = dk == 0 ? k0 : k1;
        const float wz = dk == 0 ? (1.0f - tz) : tz;
        for (int dj = 0; dj < 2; ++dj) {
            const int jj = dj == 0 ? j0 : j1;
            const float wy = dj == 0 ? (1.0f - ty) : ty;
            for (int di = 0; di < 2; ++di) {
                const int ii = di == 0 ? i0 : i1;
                const float wx = di == 0 ? (1.0f - tx) : tx;
                result += wx * wy * wz * field[idxCell3D(ii, jj, kk, nx, ny)];
            }
        }
    }
    return result;
}

__device__ float sampleUDevice(const float* field, float x, float y, float z,
                               int nx, int ny, int nz, float dx) {
    const float fx = x / dx;
    const float fy = y / dx - 0.5f;
    const float fz = z / dx - 0.5f;

    int i0 = clampi3D((int)floorf(fx), 0, nx);
    int j0 = clampi3D((int)floorf(fy), 0, ny - 1);
    int k0 = clampi3D((int)floorf(fz), 0, nz - 1);
    int i1 = min(i0 + 1, nx);
    int j1 = min(j0 + 1, ny - 1);
    int k1 = min(k0 + 1, nz - 1);

    const float tx = clampf3D(fx - (float)i0, 0.0f, 1.0f);
    const float ty = clampf3D(fy - (float)j0, 0.0f, 1.0f);
    const float tz = clampf3D(fz - (float)k0, 0.0f, 1.0f);

    float result = 0.0f;
    for (int dk = 0; dk < 2; ++dk) {
        const int kk = dk == 0 ? k0 : k1;
        const float wz = dk == 0 ? (1.0f - tz) : tz;
        for (int dj = 0; dj < 2; ++dj) {
            const int jj = dj == 0 ? j0 : j1;
            const float wy = dj == 0 ? (1.0f - ty) : ty;
            for (int di = 0; di < 2; ++di) {
                const int ii = di == 0 ? i0 : i1;
                const float wx = di == 0 ? (1.0f - tx) : tx;
                result += wx * wy * wz * field[idxU3D(ii, jj, kk, nx, ny)];
            }
        }
    }
    return result;
}

__device__ float sampleVDevice(const float* field, float x, float y, float z,
                               int nx, int ny, int nz, float dx) {
    const float fx = x / dx - 0.5f;
    const float fy = y / dx;
    const float fz = z / dx - 0.5f;

    int i0 = clampi3D((int)floorf(fx), 0, nx - 1);
    int j0 = clampi3D((int)floorf(fy), 0, ny);
    int k0 = clampi3D((int)floorf(fz), 0, nz - 1);
    int i1 = min(i0 + 1, nx - 1);
    int j1 = min(j0 + 1, ny);
    int k1 = min(k0 + 1, nz - 1);

    const float tx = clampf3D(fx - (float)i0, 0.0f, 1.0f);
    const float ty = clampf3D(fy - (float)j0, 0.0f, 1.0f);
    const float tz = clampf3D(fz - (float)k0, 0.0f, 1.0f);

    float result = 0.0f;
    for (int dk = 0; dk < 2; ++dk) {
        const int kk = dk == 0 ? k0 : k1;
        const float wz = dk == 0 ? (1.0f - tz) : tz;
        for (int dj = 0; dj < 2; ++dj) {
            const int jj = dj == 0 ? j0 : j1;
            const float wy = dj == 0 ? (1.0f - ty) : ty;
            for (int di = 0; di < 2; ++di) {
                const int ii = di == 0 ? i0 : i1;
                const float wx = di == 0 ? (1.0f - tx) : tx;
                result += wx * wy * wz * field[idxV3D(ii, jj, kk, nx, ny)];
            }
        }
    }
    return result;
}

__device__ float sampleWDevice(const float* field, float x, float y, float z,
                               int nx, int ny, int nz, float dx) {
    const float fx = x / dx - 0.5f;
    const float fy = y / dx - 0.5f;
    const float fz = z / dx;

    int i0 = clampi3D((int)floorf(fx), 0, nx - 1);
    int j0 = clampi3D((int)floorf(fy), 0, ny - 1);
    int k0 = clampi3D((int)floorf(fz), 0, nz);
    int i1 = min(i0 + 1, nx - 1);
    int j1 = min(j0 + 1, ny - 1);
    int k1 = min(k0 + 1, nz);

    const float tx = clampf3D(fx - (float)i0, 0.0f, 1.0f);
    const float ty = clampf3D(fy - (float)j0, 0.0f, 1.0f);
    const float tz = clampf3D(fz - (float)k0, 0.0f, 1.0f);

    float result = 0.0f;
    for (int dk = 0; dk < 2; ++dk) {
        const int kk = dk == 0 ? k0 : k1;
        const float wz = dk == 0 ? (1.0f - tz) : tz;
        for (int dj = 0; dj < 2; ++dj) {
            const int jj = dj == 0 ? j0 : j1;
            const float wy = dj == 0 ? (1.0f - ty) : ty;
            for (int di = 0; di < 2; ++di) {
                const int ii = di == 0 ? i0 : i1;
                const float wx = di == 0 ? (1.0f - tx) : tx;
                result += wx * wy * wz * field[idxW3D(ii, jj, kk, nx, ny)];
            }
        }
    }
    return result;
}

__device__ void velAtDevice(float x, float y, float z,
                            const float* u, const float* v, const float* w,
                            int nx, int ny, int nz, float dx,
                            float& outU, float& outV, float& outW) {
    outU = sampleUDevice(u, x, y, z, nx, ny, nz, dx);
    outV = sampleVDevice(v, x, y, z, nx, ny, nz, dx);
    outW = sampleWDevice(w, x, y, z, nx, ny, nz, dx);
}

template <typename T>
void allocArray(T*& ptr, int count) {
    if (count <= 0) {
        ptr = nullptr;
        return;
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), (std::size_t)count * sizeof(T)));
}

template <typename T>
void freeArray(T*& ptr) {
    if (ptr != nullptr) {
        CUDA_CHECK(cudaFree(ptr));
        ptr = nullptr;
    }
}

void destroyBackendArrays(MACWater3DCudaBackend* b) {
    freeArray(b->d_u);
    freeArray(b->d_v);
    freeArray(b->d_w);
    freeArray(b->d_uWeight);
    freeArray(b->d_vWeight);
    freeArray(b->d_wWeight);
    freeArray(b->d_uPrev);
    freeArray(b->d_vPrev);
    freeArray(b->d_wPrev);
    freeArray(b->d_uDelta);
    freeArray(b->d_vDelta);
    freeArray(b->d_wDelta);
    freeArray(b->d_uTmp);
    freeArray(b->d_vTmp);
    freeArray(b->d_wTmp);
    freeArray(b->d_pressure);
    freeArray(b->d_pressureTmp);
    freeArray(b->d_rhs);
    freeArray(b->d_water);
    freeArray(b->d_divergence);
    freeArray(b->d_speed);
    freeArray(b->d_mass);
    freeArray(b->d_liquid);
    freeArray(b->d_liquidTmp);
    freeArray(b->d_solid);
    freeArray(b->d_solidUser);
    freeArray(b->d_validU);
    freeArray(b->d_validV);
    freeArray(b->d_validW);
    freeArray(b->d_validUTmp);
    freeArray(b->d_validVTmp);
    freeArray(b->d_validWTmp);
    freeArray(b->d_occ);
    freeArray(b->d_region);
    freeArray(b->d_cellCounts);
    freeArray(b->d_particleCount);
    freeArray(b->d_particles);
}

std::size_t backendBytesAllocated(const MACWater3DCudaBackend* b) {
    if (b == nullptr) return 0;

    std::size_t total = 0;
    total += (std::size_t)b->uCount * sizeof(float) * 5;
    total += (std::size_t)b->vCount * sizeof(float) * 5;
    total += (std::size_t)b->wCount * sizeof(float) * 5;
    total += (std::size_t)b->cellCount * sizeof(float) * 7;
    total += (std::size_t)b->cellCount * sizeof(uint8_t) * 6;
    total += (std::size_t)b->uCount * sizeof(uint8_t) * 2;
    total += (std::size_t)b->vCount * sizeof(uint8_t) * 2;
    total += (std::size_t)b->wCount * sizeof(uint8_t) * 2;
    total += (std::size_t)b->cellCount * sizeof(int);
    total += sizeof(int);
    total += (std::size_t)b->particleCapacity * sizeof(MACWater3D::Particle);
    return total;
}

inline void resetPressureStats(MACWater3D& sim) {
    sim.lastPressureSolveMs = 0.0f;
    sim.lastPressureIterations = 0;
}

struct AbsFloat {
    __host__ __device__ float operator()(float v) const {
        return fabsf(v);
    }
};

struct LiquidCellPredicate {
    const uint8_t* liquid = nullptr;
    const uint8_t* solid = nullptr;

    __host__ __device__ bool operator()(int idx) const {
        return liquid[idx] != 0 && solid[idx] == 0;
    }
};

float reduceAbsMaxDevice(const float* data, int count) {
    if (data == nullptr || count <= 0) return 0.0f;
    thrust::device_ptr<const float> begin = thrust::device_pointer_cast(data);
    return thrust::transform_reduce(thrust::device, begin, begin + count,
                                    AbsFloat(), 0.0f, cuda::maximum<float>());
}

int readParticleCountHost(MACWater3DCudaBackend* backend) {
    int particleCount = 0;
    if (backend != nullptr && backend->d_particleCount != nullptr) {
        CUDA_CHECK(cudaMemcpy(&particleCount, backend->d_particleCount, sizeof(int), cudaMemcpyDeviceToHost));
    }
    return particleCount;
}

int countLiquidCellsDevice(const MACWater3DCudaBackend* backend) {
    if (backend == nullptr || backend->cellCount <= 0) return 0;
    auto begin = thrust::make_counting_iterator<int>(0);
    auto end = begin + backend->cellCount;
    return (int)thrust::count_if(thrust::device, begin, end,
                                 LiquidCellPredicate{backend->d_liquid, backend->d_solid});
}

const char* cudaBackendName(const MACWater3DCudaBackend* backend) {
    if (backend == nullptr) return "CUDA MAC 3D";
    return backend->lastUsedDeviceJacobi
        ? "CUDA MAC 3D (device Jacobi)"
        : "CUDA MAC 3D (CPU pressure bridge)";
}

void applyCudaStats(MACWater3DCudaBackend* backend, MACWater3D& sim, float stepMs) {
    const int particleCount = readParticleCountHost(backend);
    const float maxU = reduceAbsMaxDevice(backend->d_u, backend->uCount);
    const float maxV = reduceAbsMaxDevice(backend->d_v, backend->vCount);
    const float maxW = reduceAbsMaxDevice(backend->d_w, backend->wCount);
    const float maxDiv = reduceAbsMaxDevice(backend->d_divergence, backend->cellCount);
    const int liquidCells = countLiquidCellsDevice(backend);

    sim.targetMass = (float)particleCount;
    if (sim.desiredMass < 0.0f && sim.targetMass > 0.0f) {
        sim.desiredMass = sim.targetMass;
    }
    sim.derivedFieldsDirty = false;

    sim.lastStats.cudaEnabled = true;
    sim.lastStats.backendReady = true;
    sim.lastStats.nx = sim.nx;
    sim.lastStats.ny = sim.ny;
    sim.lastStats.nz = sim.nz;
    sim.lastStats.particleCount = particleCount;
    sim.lastStats.liquidCells = liquidCells;
    sim.lastStats.maxSpeed = std::max(maxU, std::max(maxV, maxW));
    sim.lastStats.maxDivergence = maxDiv;
    sim.lastStats.dt = sim.dt;
    sim.lastStats.lastStepMs = stepMs;
    sim.lastStats.pressureMs = sim.lastPressureSolveMs;
    sim.lastStats.pressureIters = sim.lastPressureIterations;
    sim.lastStats.targetMass = sim.targetMass;
    sim.lastStats.desiredMass = sim.desiredMass;
    sim.lastStats.backendName = cudaBackendName(backend);
    sim.lastStats.bytesAllocated = backendBytesAllocated(backend);
    sim.lastStats.timings.reset();
    sim.lastStats.timings.totalMs = stepMs;
}

void ensureParticleCapacity(MACWater3DCudaBackend* b, int requested) {
    if (requested <= b->particleCapacity) return;

    int newCapacity = std::max(requested, std::max(1024, b->particleCapacity * 2));
    MACWater3D::Particle* d_newParticles = nullptr;
    allocArray(d_newParticles, newCapacity);

    int particleCount = 0;
    if (b->d_particleCount != nullptr) {
        CUDA_CHECK(cudaMemcpy(&particleCount, b->d_particleCount, sizeof(int), cudaMemcpyDeviceToHost));
    }
    if (b->d_particles != nullptr) {
        if (particleCount > 0) {
            CUDA_CHECK(cudaMemcpy(d_newParticles, b->d_particles,
                                  (std::size_t)particleCount * sizeof(MACWater3D::Particle),
                                  cudaMemcpyDeviceToDevice));
        }
        CUDA_CHECK(cudaFree(b->d_particles));
    }

    b->d_particles = d_newParticles;
    b->particleCapacity = newCapacity;
}

__global__ void rebuildSolidsKernel(const uint8_t* solidUser, uint8_t* solid,
                                    int nx, int ny, int nz, int bt, bool openTop) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int cellCount = nx * ny * nz;
    if (idx >= cellCount) return;

    const int k = idx / (nx * ny);
    const int rem = idx - k * nx * ny;
    const int j = rem / nx;
    const int i = rem - j * nx;

    uint8_t s = solidUser[idx];
    const bool wallX = (i < bt) || (i >= nx - bt);
    const bool wallY = (j < bt) || (!openTop && j >= ny - bt);
    const bool wallZ = (k < bt) || (k >= nz - bt);
    if (wallX || wallY || wallZ) s = 1;
    solid[idx] = s;
}

__global__ void applyBoundaryUKernel(float* u, const uint8_t* solid, int nx, int ny, int nz) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = (nx + 1) * ny * nz;
    if (idx >= count) return;

    const int k = idx / ((nx + 1) * ny);
    const int rem = idx - k * (nx + 1) * ny;
    const int j = rem / (nx + 1);
    const int i = rem - j * (nx + 1);

    const bool leftSolid = (i - 1 >= 0) ? (solid[idxCell3D(i - 1, j, k, nx, ny)] != 0) : true;
    const bool rightSolid = (i < nx) ? (solid[idxCell3D(i, j, k, nx, ny)] != 0) : true;
    if (leftSolid || rightSolid) u[idx] = 0.0f;
}

__global__ void applyBoundaryVKernel(float* v, const uint8_t* solid, int nx, int ny, int nz, bool openTop) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = nx * (ny + 1) * nz;
    if (idx >= count) return;

    const int k = idx / (nx * (ny + 1));
    const int rem = idx - k * nx * (ny + 1);
    const int j = rem / nx;
    const int i = rem - j * nx;

    if (j == 0) { v[idx] = 0.0f; return; }
    if (j == ny) {
        if (!openTop || solid[idxCell3D(i, ny - 1, k, nx, ny)] != 0) v[idx] = 0.0f;
        return;
    }

    const bool botSolid = solid[idxCell3D(i, j - 1, k, nx, ny)] != 0;
    const bool topSolid = solid[idxCell3D(i, j, k, nx, ny)] != 0;
    if (botSolid || topSolid) v[idx] = 0.0f;
}

__global__ void applyBoundaryWKernel(float* w, const uint8_t* solid, int nx, int ny, int nz) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = nx * ny * (nz + 1);
    if (idx >= count) return;

    const int k = idx / (nx * ny);
    const int rem = idx - k * nx * ny;
    const int j = rem / nx;
    const int i = rem - j * nx;

    const bool backSolid = (k - 1 >= 0) ? (solid[idxCell3D(i, j, k - 1, nx, ny)] != 0) : true;
    const bool frontSolid = (k < nz) ? (solid[idxCell3D(i, j, k, nx, ny)] != 0) : true;
    if (backSolid || frontSolid) w[idx] = 0.0f;
}

__global__ void scatterParticlesKernel(const MACWater3D::Particle* particles, int particleCount,
                                       float* u, float* v, float* w,
                                       float* uWeight, float* vWeight, float* wWeight,
                                       int nx, int ny, int nz, float dx, bool useAPIC) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particleCount) return;

    const MACWater3D::Particle p = particles[idx];

    {
        const float fx = p.x / dx;
        const float fy = p.y / dx - 0.5f;
        const float fz = p.z / dx - 0.5f;
        const int i0 = clampi3D((int)floorf(fx), 0, nx);
        const int j0 = clampi3D((int)floorf(fy), 0, ny - 1);
        const int k0 = clampi3D((int)floorf(fz), 0, nz - 1);
        const int i1 = min(i0 + 1, nx);
        const int j1 = min(j0 + 1, ny - 1);
        const int k1 = min(k0 + 1, nz - 1);
        const float tx = clampf3D(fx - (float)i0, 0.0f, 1.0f);
        const float ty = clampf3D(fy - (float)j0, 0.0f, 1.0f);
        const float tz = clampf3D(fz - (float)k0, 0.0f, 1.0f);
        const int ids[8] = {
            idxU3D(i0, j0, k0, nx, ny), idxU3D(i1, j0, k0, nx, ny),
            idxU3D(i0, j1, k0, nx, ny), idxU3D(i1, j1, k0, nx, ny),
            idxU3D(i0, j0, k1, nx, ny), idxU3D(i1, j0, k1, nx, ny),
            idxU3D(i0, j1, k1, nx, ny), idxU3D(i1, j1, k1, nx, ny)
        };
        const float ws[8] = {
            (1.0f - tx) * (1.0f - ty) * (1.0f - tz),
            tx * (1.0f - ty) * (1.0f - tz),
            (1.0f - tx) * ty * (1.0f - tz),
            tx * ty * (1.0f - tz),
            (1.0f - tx) * (1.0f - ty) * tz,
            tx * (1.0f - ty) * tz,
            (1.0f - tx) * ty * tz,
            tx * ty * tz
        };
        for (int n = 0; n < 8; ++n) {
            float value = p.u;
            if (useAPIC) {
                const int kk = (n / 4 == 0) ? k0 : k1;
                const int jj = ((n % 4) / 2 == 0) ? j0 : j1;
                const int ii = (n % 2 == 0) ? i0 : i1;
                const float pxFace = (float)ii * dx;
                const float pyFace = ((float)jj + 0.5f) * dx;
                const float pzFace = ((float)kk + 0.5f) * dx;
                value += p.c00 * (pxFace - p.x) + p.c01 * (pyFace - p.y) + p.c02 * (pzFace - p.z);
            }
            atomicAdd(u + ids[n], value * ws[n]);
            atomicAdd(uWeight + ids[n], ws[n]);
        }
    }

    {
        const float fx = p.x / dx - 0.5f;
        const float fy = p.y / dx;
        const float fz = p.z / dx - 0.5f;
        const int i0 = clampi3D((int)floorf(fx), 0, nx - 1);
        const int j0 = clampi3D((int)floorf(fy), 0, ny);
        const int k0 = clampi3D((int)floorf(fz), 0, nz - 1);
        const int i1 = min(i0 + 1, nx - 1);
        const int j1 = min(j0 + 1, ny);
        const int k1 = min(k0 + 1, nz - 1);
        const float tx = clampf3D(fx - (float)i0, 0.0f, 1.0f);
        const float ty = clampf3D(fy - (float)j0, 0.0f, 1.0f);
        const float tz = clampf3D(fz - (float)k0, 0.0f, 1.0f);
        const int ids[8] = {
            idxV3D(i0, j0, k0, nx, ny), idxV3D(i1, j0, k0, nx, ny),
            idxV3D(i0, j1, k0, nx, ny), idxV3D(i1, j1, k0, nx, ny),
            idxV3D(i0, j0, k1, nx, ny), idxV3D(i1, j0, k1, nx, ny),
            idxV3D(i0, j1, k1, nx, ny), idxV3D(i1, j1, k1, nx, ny)
        };
        const float ws[8] = {
            (1.0f - tx) * (1.0f - ty) * (1.0f - tz),
            tx * (1.0f - ty) * (1.0f - tz),
            (1.0f - tx) * ty * (1.0f - tz),
            tx * ty * (1.0f - tz),
            (1.0f - tx) * (1.0f - ty) * tz,
            tx * (1.0f - ty) * tz,
            (1.0f - tx) * ty * tz,
            tx * ty * tz
        };
        for (int n = 0; n < 8; ++n) {
            float value = p.v;
            if (useAPIC) {
                const int kk = (n / 4 == 0) ? k0 : k1;
                const int jj = ((n % 4) / 2 == 0) ? j0 : j1;
                const int ii = (n % 2 == 0) ? i0 : i1;
                const float pxFace = ((float)ii + 0.5f) * dx;
                const float pyFace = (float)jj * dx;
                const float pzFace = ((float)kk + 0.5f) * dx;
                value += p.c10 * (pxFace - p.x) + p.c11 * (pyFace - p.y) + p.c12 * (pzFace - p.z);
            }
            atomicAdd(v + ids[n], value * ws[n]);
            atomicAdd(vWeight + ids[n], ws[n]);
        }
    }

    {
        const float fx = p.x / dx - 0.5f;
        const float fy = p.y / dx - 0.5f;
        const float fz = p.z / dx;
        const int i0 = clampi3D((int)floorf(fx), 0, nx - 1);
        const int j0 = clampi3D((int)floorf(fy), 0, ny - 1);
        const int k0 = clampi3D((int)floorf(fz), 0, nz);
        const int i1 = min(i0 + 1, nx - 1);
        const int j1 = min(j0 + 1, ny - 1);
        const int k1 = min(k0 + 1, nz);
        const float tx = clampf3D(fx - (float)i0, 0.0f, 1.0f);
        const float ty = clampf3D(fy - (float)j0, 0.0f, 1.0f);
        const float tz = clampf3D(fz - (float)k0, 0.0f, 1.0f);
        const int ids[8] = {
            idxW3D(i0, j0, k0, nx, ny), idxW3D(i1, j0, k0, nx, ny),
            idxW3D(i0, j1, k0, nx, ny), idxW3D(i1, j1, k0, nx, ny),
            idxW3D(i0, j0, k1, nx, ny), idxW3D(i1, j0, k1, nx, ny),
            idxW3D(i0, j1, k1, nx, ny), idxW3D(i1, j1, k1, nx, ny)
        };
        const float ws[8] = {
            (1.0f - tx) * (1.0f - ty) * (1.0f - tz),
            tx * (1.0f - ty) * (1.0f - tz),
            (1.0f - tx) * ty * (1.0f - tz),
            tx * ty * (1.0f - tz),
            (1.0f - tx) * (1.0f - ty) * tz,
            tx * (1.0f - ty) * tz,
            (1.0f - tx) * ty * tz,
            tx * ty * tz
        };
        for (int n = 0; n < 8; ++n) {
            float value = p.w;
            if (useAPIC) {
                const int kk = (n / 4 == 0) ? k0 : k1;
                const int jj = ((n % 4) / 2 == 0) ? j0 : j1;
                const int ii = (n % 2 == 0) ? i0 : i1;
                const float pxFace = ((float)ii + 0.5f) * dx;
                const float pyFace = ((float)jj + 0.5f) * dx;
                const float pzFace = (float)kk * dx;
                value += p.c20 * (pxFace - p.x) + p.c21 * (pyFace - p.y) + p.c22 * (pzFace - p.z);
            }
            atomicAdd(w + ids[n], value * ws[n]);
            atomicAdd(wWeight + ids[n], ws[n]);
        }
    }
}

__global__ void normalizeKernel(float* values, const float* weights, int count) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    const float w = weights[idx];
    values[idx] = (w > 1e-6f) ? (values[idx] / w) : 0.0f;
}

__global__ void markLiquidFromParticlesKernel(const MACWater3D::Particle* particles, int particleCount,
                                              const uint8_t* solid, uint8_t* liquid,
                                              int nx, int ny, int nz, float dx) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particleCount) return;
    const MACWater3D::Particle p = particles[idx];
    const int i = clampi3D((int)floorf(p.x / dx), 0, nx - 1);
    const int j = clampi3D((int)floorf(p.y / dx), 0, ny - 1);
    const int k = clampi3D((int)floorf(p.z / dx), 0, nz - 1);
    const int id = idxCell3D(i, j, k, nx, ny);
    if (!solid[id]) liquid[id] = 1;
}

__global__ void dilateLiquidKernel(const uint8_t* solid, const uint8_t* liquid, uint8_t* out,
                                   int nx, int ny, int nz) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int cellCount = nx * ny * nz;
    if (idx >= cellCount) return;

    const int k = idx / (nx * ny);
    const int rem = idx - k * nx * ny;
    const int j = rem / nx;
    const int i = rem - j * nx;

    if (solid[idx]) { out[idx] = 0; return; }
    if (liquid[idx]) { out[idx] = 1; return; }

    const bool Isnear =
        (i > 0 && liquid[idxCell3D(i - 1, j, k, nx, ny)]) ||
        (i + 1 < nx && liquid[idxCell3D(i + 1, j, k, nx, ny)]) ||
        (j > 0 && liquid[idxCell3D(i, j - 1, k, nx, ny)]) ||
        (j + 1 < ny && liquid[idxCell3D(i, j + 1, k, nx, ny)]) ||
        (k > 0 && liquid[idxCell3D(i, j, k - 1, nx, ny)]) ||
        (k + 1 < nz && liquid[idxCell3D(i, j, k + 1, nx, ny)]);
    out[idx] = Isnear ? 1 : 0;
}

__global__ void applyGravityDampingKernel(float* v, int nx, int ny, int nz, float dt, float gravity,
                                          float damp) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = nx * (ny + 1) * nz;
    if (idx >= count) return;

    const int k = idx / (nx * (ny + 1));
    const int rem = idx - k * nx * (ny + 1);
    const int j = rem / nx;
    const int i = rem - j * nx;

    v[idx] += dt * gravity;
    v[idx] *= damp;
}

__global__ void applyDampingKernel(float* field, int count, float damp) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    field[idx] *= damp;
}

__global__ void copyKernel(const float* src, float* dst, int count) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    dst[idx] = src[idx];
}

__global__ void diffuseUKernel(const float* current, const float* rhs, float* next,
                               const uint8_t* solid, int nx, int ny, int nz,
                               float alphaInvDx2, float omega) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = (nx + 1) * ny * nz;
    if (idx >= count) return;

    const int k = idx / ((nx + 1) * ny);
    const int rem = idx - k * (nx + 1) * ny;
    const int j = rem / (nx + 1);
    const int i = rem - j * (nx + 1);

    const bool fixed = (i == 0 || i == nx) ||
        ((i - 1 >= 0 ? solid[idxCell3D(i - 1, j, k, nx, ny)] : 1) != 0) ||
        ((i < nx ? solid[idxCell3D(i, j, k, nx, ny)] : 1) != 0);
    if (fixed) {
        next[idx] = current[idx];
        return;
    }

    float sum = 0.0f;
    int n = 0;
    if (i > 0) { sum += current[idxU3D(i - 1, j, k, nx, ny)]; n++; }
    if (i < nx) { sum += current[idxU3D(i + 1, j, k, nx, ny)]; n++; }
    if (j > 0) { sum += current[idxU3D(i, j - 1, k, nx, ny)]; n++; }
    if (j + 1 < ny) { sum += current[idxU3D(i, j + 1, k, nx, ny)]; n++; }
    if (k > 0) { sum += current[idxU3D(i, j, k - 1, nx, ny)]; n++; }
    if (k + 1 < nz) { sum += current[idxU3D(i, j, k + 1, nx, ny)]; n++; }

    const float xNew = (rhs[idx] + alphaInvDx2 * sum) / (1.0f + alphaInvDx2 * (float)n);
    next[idx] = (1.0f - omega) * current[idx] + omega * xNew;
}

__global__ void diffuseVKernel(const float* current, const float* rhs, float* next,
                               const uint8_t* solid, int nx, int ny, int nz,
                               float alphaInvDx2, float omega, bool openTop) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = nx * (ny + 1) * nz;
    if (idx >= count) return;

    const int k = idx / (nx * (ny + 1));
    const int rem = idx - k * nx * (ny + 1);
    const int j = rem / nx;
    const int i = rem - j * nx;

    const bool fixed = (j == 0) || (j == ny && !openTop) ||
        ((j - 1 >= 0 ? solid[idxCell3D(i, j - 1, k, nx, ny)] : 1) != 0) ||
        ((j < ny ? solid[idxCell3D(i, j, k, nx, ny)] : (openTop ? 0 : 1)) != 0);
    if (fixed) {
        next[idx] = current[idx];
        return;
    }

    float sum = 0.0f;
    int n = 0;
    if (i > 0) { sum += current[idxV3D(i - 1, j, k, nx, ny)]; n++; }
    if (i + 1 < nx) { sum += current[idxV3D(i + 1, j, k, nx, ny)]; n++; }
    if (j > 0) { sum += current[idxV3D(i, j - 1, k, nx, ny)]; n++; }
    if (j < ny) { sum += current[idxV3D(i, j + 1, k, nx, ny)]; n++; }
    if (k > 0) { sum += current[idxV3D(i, j, k - 1, nx, ny)]; n++; }
    if (k + 1 < nz) { sum += current[idxV3D(i, j, k + 1, nx, ny)]; n++; }

    const float xNew = (rhs[idx] + alphaInvDx2 * sum) / (1.0f + alphaInvDx2 * (float)n);
    next[idx] = (1.0f - omega) * current[idx] + omega * xNew;
}

__global__ void diffuseWKernel(const float* current, const float* rhs, float* next,
                               const uint8_t* solid, int nx, int ny, int nz,
                               float alphaInvDx2, float omega) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = nx * ny * (nz + 1);
    if (idx >= count) return;

    const int k = idx / (nx * ny);
    const int rem = idx - k * nx * ny;
    const int j = rem / nx;
    const int i = rem - j * nx;

    const bool fixed = (k == 0 || k == nz) ||
        ((k - 1 >= 0 ? solid[idxCell3D(i, j, k - 1, nx, ny)] : 1) != 0) ||
        ((k < nz ? solid[idxCell3D(i, j, k, nx, ny)] : 1) != 0);
    if (fixed) {
        next[idx] = current[idx];
        return;
    }

    float sum = 0.0f;
    int n = 0;
    if (i > 0) { sum += current[idxW3D(i - 1, j, k, nx, ny)]; n++; }
    if (i + 1 < nx) { sum += current[idxW3D(i + 1, j, k, nx, ny)]; n++; }
    if (j > 0) { sum += current[idxW3D(i, j - 1, k, nx, ny)]; n++; }
    if (j + 1 < ny) { sum += current[idxW3D(i, j + 1, k, nx, ny)]; n++; }
    if (k > 0) { sum += current[idxW3D(i, j, k - 1, nx, ny)]; n++; }
    if (k < nz) { sum += current[idxW3D(i, j, k + 1, nx, ny)]; n++; }

    const float xNew = (rhs[idx] + alphaInvDx2 * sum) / (1.0f + alphaInvDx2 * (float)n);
    next[idx] = (1.0f - omega) * current[idx] + omega * xNew;
}

__global__ void buildPressureRhsKernel(const float* u, const float* v, const float* w,
                                       const uint8_t* solid, const uint8_t* liquid,
                                       float* rhs, int nx, int ny, int nz, float invDx, float invDt) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int cellCount = nx * ny * nz;
    if (idx >= cellCount) return;

    const int k = idx / (nx * ny);
    const int rem = idx - k * nx * ny;
    const int j = rem / nx;
    const int i = rem - j * nx;

    if (solid[idx] || !liquid[idx]) {
        rhs[idx] = 0.0f;
        return;
    }

    float uL = u[idxU3D(i, j, k, nx, ny)];
    float uR = u[idxU3D(i + 1, j, k, nx, ny)];
    float vB = v[idxV3D(i, j, k, nx, ny)];
    float vT = v[idxV3D(i, j + 1, k, nx, ny)];
    float wBk = w[idxW3D(i, j, k, nx, ny)];
    float wFr = w[idxW3D(i, j, k + 1, nx, ny)];

    if (i - 1 >= 0 && solid[idxCell3D(i - 1, j, k, nx, ny)]) uL = 0.0f;
    if (i + 1 < nx && solid[idxCell3D(i + 1, j, k, nx, ny)]) uR = 0.0f;
    if (j - 1 >= 0 && solid[idxCell3D(i, j - 1, k, nx, ny)]) vB = 0.0f;
    if (j + 1 < ny && solid[idxCell3D(i, j + 1, k, nx, ny)]) vT = 0.0f;
    if (k - 1 >= 0 && solid[idxCell3D(i, j, k - 1, nx, ny)]) wBk = 0.0f;
    if (k + 1 < nz && solid[idxCell3D(i, j, k + 1, nx, ny)]) wFr = 0.0f;

    const float div = (uR - uL + vT - vB + wFr - wBk) * invDx;
    rhs[idx] = -div * invDt;
}

__global__ void pressureJacobiKernel(const float* pressure, float* pressureOut, const float* rhs,
                                     const uint8_t* solid, const uint8_t* liquid,
                                     int nx, int ny, int nz, float dx2, float omega, bool openTop) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int cellCount = nx * ny * nz;
    if (idx >= cellCount) return;

    const int k = idx / (nx * ny);
    const int rem = idx - k * nx * ny;
    const int j = rem / nx;
    const int i = rem - j * nx;

    if (solid[idx] || !liquid[idx]) {
        pressureOut[idx] = 0.0f;
        return;
    }

    float sum = 0.0f;
    int diag = 0;

    if (i - 1 >= 0) {
        const int nid = idxCell3D(i - 1, j, k, nx, ny);
        if (!solid[nid]) {
            if (liquid[nid]) { sum += pressure[nid]; }
            diag++;
        }
    }
    if (i + 1 < nx) {
        const int nid = idxCell3D(i + 1, j, k, nx, ny);
        if (!solid[nid]) {
            if (liquid[nid]) { sum += pressure[nid]; }
            diag++;
        }
    }
    if (j - 1 >= 0) {
        const int nid = idxCell3D(i, j - 1, k, nx, ny);
        if (!solid[nid]) {
            if (liquid[nid]) { sum += pressure[nid]; }
            diag++;
        }
    }
    if (j + 1 < ny) {
        const int nid = idxCell3D(i, j + 1, k, nx, ny);
        if (!solid[nid]) {
            if (liquid[nid]) { sum += pressure[nid]; }
            diag++;
        }
    } else if (openTop) {
        diag++;
    }
    if (k - 1 >= 0) {
        const int nid = idxCell3D(i, j, k - 1, nx, ny);
        if (!solid[nid]) {
            if (liquid[nid]) { sum += pressure[nid]; }
            diag++;
        }
    }
    if (k + 1 < nz) {
        const int nid = idxCell3D(i, j, k + 1, nx, ny);
        if (!solid[nid]) {
            if (liquid[nid]) { sum += pressure[nid]; }
            diag++;
        }
    }

    if (diag <= 0) {
        pressureOut[idx] = 0.0f;
        return;
    }

    const float jacobi = (sum + rhs[idx] * dx2) / (float)diag;
    pressureOut[idx] = (1.0f - omega) * pressure[idx] + omega * jacobi;
}

__global__ void applyPressureUKernel(float* u, const float* pressure,
                                     const uint8_t* solid, const uint8_t* liquid,
                                     int nx, int ny, int nz, float scale) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = (nx + 1) * ny * nz;
    if (idx >= count) return;

    const int k = idx / ((nx + 1) * ny);
    const int rem = idx - k * (nx + 1) * ny;
    const int j = rem / (nx + 1);
    const int i = rem - j * (nx + 1);

    const bool leftSolid = (i - 1 >= 0) ? (solid[idxCell3D(i - 1, j, k, nx, ny)] != 0) : true;
    const bool rightSolid = (i < nx) ? (solid[idxCell3D(i, j, k, nx, ny)] != 0) : true;
    if (leftSolid || rightSolid) {
        u[idx] = 0.0f;
        return;
    }

    const bool leftFluid = (i - 1 >= 0) ? (!solid[idxCell3D(i - 1, j, k, nx, ny)] && liquid[idxCell3D(i - 1, j, k, nx, ny)]) : false;
    const bool rightFluid = (i < nx) ? (!solid[idxCell3D(i, j, k, nx, ny)] && liquid[idxCell3D(i, j, k, nx, ny)]) : false;
    if (!leftFluid && !rightFluid) return;

    const float pL = leftFluid ? pressure[idxCell3D(i - 1, j, k, nx, ny)] : 0.0f;
    const float pR = rightFluid ? pressure[idxCell3D(i, j, k, nx, ny)] : 0.0f;
    u[idx] -= scale * (pR - pL);
}

__global__ void applyPressureVKernel(float* v, const float* pressure,
                                     const uint8_t* solid, const uint8_t* liquid,
                                     int nx, int ny, int nz, float scale, bool openTop) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = nx * (ny + 1) * nz;
    if (idx >= count) return;

    const int k = idx / (nx * (ny + 1));
    const int rem = idx - k * nx * (ny + 1);
    const int j = rem / nx;
    const int i = rem - j * nx;

    const bool botSolid = (j - 1 >= 0) ? (solid[idxCell3D(i, j - 1, k, nx, ny)] != 0) : true;
    const bool topSolid = (j < ny) ? (solid[idxCell3D(i, j, k, nx, ny)] != 0) : !openTop;
    if (botSolid || topSolid) {
        v[idx] = 0.0f;
        return;
    }

    const bool botFluid = (j - 1 >= 0) ? (!solid[idxCell3D(i, j - 1, k, nx, ny)] && liquid[idxCell3D(i, j - 1, k, nx, ny)]) : false;
    const bool topFluid = (j < ny) ? (!solid[idxCell3D(i, j, k, nx, ny)] && liquid[idxCell3D(i, j, k, nx, ny)]) : false;
    if (!botFluid && !topFluid) return;

    const float pB = botFluid ? pressure[idxCell3D(i, j - 1, k, nx, ny)] : 0.0f;
    const float pT = topFluid ? pressure[idxCell3D(i, j, k, nx, ny)] : 0.0f;
    v[idx] -= scale * (pT - pB);
}

__global__ void applyPressureWKernel(float* w, const float* pressure,
                                     const uint8_t* solid, const uint8_t* liquid,
                                     int nx, int ny, int nz, float scale) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = nx * ny * (nz + 1);
    if (idx >= count) return;

    const int k = idx / (nx * ny);
    const int rem = idx - k * nx * ny;
    const int j = rem / nx;
    const int i = rem - j * nx;

    const bool backSolid = (k - 1 >= 0) ? (solid[idxCell3D(i, j, k - 1, nx, ny)] != 0) : true;
    const bool frontSolid = (k < nz) ? (solid[idxCell3D(i, j, k, nx, ny)] != 0) : true;
    if (backSolid || frontSolid) {
        w[idx] = 0.0f;
        return;
    }

    const bool backFluid = (k - 1 >= 0) ? (!solid[idxCell3D(i, j, k - 1, nx, ny)] && liquid[idxCell3D(i, j, k - 1, nx, ny)]) : false;
    const bool frontFluid = (k < nz) ? (!solid[idxCell3D(i, j, k, nx, ny)] && liquid[idxCell3D(i, j, k, nx, ny)]) : false;
    if (!backFluid && !frontFluid) return;

    const float pBk = backFluid ? pressure[idxCell3D(i, j, k - 1, nx, ny)] : 0.0f;
    const float pFr = frontFluid ? pressure[idxCell3D(i, j, k, nx, ny)] : 0.0f;
    w[idx] -= scale * (pFr - pBk);
}

__global__ void computeDeltaKernel(const float* current, const float* previous, float* delta, int count) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    delta[idx] = current[idx] - previous[idx];
}

__global__ void seedValidUKernel(float* u, uint8_t* valid, const float* weights, const uint8_t* solid,
                                 int nx, int ny, int nz) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = (nx + 1) * ny * nz;
    if (idx >= count) return;

    const int k = idx / ((nx + 1) * ny);
    const int rem = idx - k * (nx + 1) * ny;
    const int j = rem / (nx + 1);
    const int i = rem - j * (nx + 1);
    const bool leftSolid = (i - 1 >= 0) ? (solid[idxCell3D(i - 1, j, k, nx, ny)] != 0) : true;
    const bool rightSolid = (i < nx) ? (solid[idxCell3D(i, j, k, nx, ny)] != 0) : true;
    if (leftSolid || rightSolid) {
        u[idx] = 0.0f;
        valid[idx] = 0;
        return;
    }
    valid[idx] = weights[idx] > 1e-6f ? 1 : 0;
}

__global__ void seedValidVKernel(float* v, uint8_t* valid, const float* weights, const uint8_t* solid,
                                 int nx, int ny, int nz, bool openTop) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = nx * (ny + 1) * nz;
    if (idx >= count) return;

    const int k = idx / (nx * (ny + 1));
    const int rem = idx - k * nx * (ny + 1);
    const int j = rem / nx;
    const int i = rem - j * nx;
    const bool botSolid = (j - 1 >= 0) ? (solid[idxCell3D(i, j - 1, k, nx, ny)] != 0) : true;
    const bool topSolid = (j < ny) ? (solid[idxCell3D(i, j, k, nx, ny)] != 0) : !openTop;
    if (botSolid || topSolid) {
        v[idx] = 0.0f;
        valid[idx] = 0;
        return;
    }
    valid[idx] = weights[idx] > 1e-6f ? 1 : 0;
}

__global__ void seedValidWKernel(float* w, uint8_t* valid, const float* weights, const uint8_t* solid,
                                 int nx, int ny, int nz) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = nx * ny * (nz + 1);
    if (idx >= count) return;

    const int k = idx / (nx * ny);
    const int rem = idx - k * nx * ny;
    const int j = rem / nx;
    const int i = rem - j * nx;
    const bool backSolid = (k - 1 >= 0) ? (solid[idxCell3D(i, j, k - 1, nx, ny)] != 0) : true;
    const bool frontSolid = (k < nz) ? (solid[idxCell3D(i, j, k, nx, ny)] != 0) : true;
    if (backSolid || frontSolid) {
        w[idx] = 0.0f;
        valid[idx] = 0;
        return;
    }
    valid[idx] = weights[idx] > 1e-6f ? 1 : 0;
}

__global__ void extrapolateUKernel(const float* current, float* next,
                                   const uint8_t* valid, uint8_t* nextValid,
                                   const uint8_t* solid, int nx, int ny, int nz) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = (nx + 1) * ny * nz;
    if (idx >= count) return;

    const int k = idx / ((nx + 1) * ny);
    const int rem = idx - k * (nx + 1) * ny;
    const int j = rem / (nx + 1);
    const int i = rem - j * (nx + 1);
    if (valid[idx]) {
        next[idx] = current[idx];
        nextValid[idx] = 1;
        return;
    }

    const bool leftSolid = (i - 1 >= 0) ? (solid[idxCell3D(i - 1, j, k, nx, ny)] != 0) : true;
    const bool rightSolid = (i < nx) ? (solid[idxCell3D(i, j, k, nx, ny)] != 0) : true;
    if (leftSolid || rightSolid) {
        next[idx] = 0.0f;
        nextValid[idx] = 0;
        return;
    }

    float sum = 0.0f;
    int n = 0;
    if (i > 0 && valid[idxU3D(i - 1, j, k, nx, ny)]) { sum += current[idxU3D(i - 1, j, k, nx, ny)]; n++; }
    if (i < nx && valid[idxU3D(i + 1, j, k, nx, ny)]) { sum += current[idxU3D(i + 1, j, k, nx, ny)]; n++; }
    if (j > 0 && valid[idxU3D(i, j - 1, k, nx, ny)]) { sum += current[idxU3D(i, j - 1, k, nx, ny)]; n++; }
    if (j + 1 < ny && valid[idxU3D(i, j + 1, k, nx, ny)]) { sum += current[idxU3D(i, j + 1, k, nx, ny)]; n++; }
    if (k > 0 && valid[idxU3D(i, j, k - 1, nx, ny)]) { sum += current[idxU3D(i, j, k - 1, nx, ny)]; n++; }
    if (k + 1 < nz && valid[idxU3D(i, j, k + 1, nx, ny)]) { sum += current[idxU3D(i, j, k + 1, nx, ny)]; n++; }
    if (n > 0) {
        next[idx] = sum / (float)n;
        nextValid[idx] = 1;
    } else {
        next[idx] = current[idx];
        nextValid[idx] = 0;
    }
}

__global__ void extrapolateVKernel(const float* current, float* next,
                                   const uint8_t* valid, uint8_t* nextValid,
                                   const uint8_t* solid, int nx, int ny, int nz, bool openTop) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = nx * (ny + 1) * nz;
    if (idx >= count) return;

    const int k = idx / (nx * (ny + 1));
    const int rem = idx - k * nx * (ny + 1);
    const int j = rem / nx;
    const int i = rem - j * nx;
    if (valid[idx]) {
        next[idx] = current[idx];
        nextValid[idx] = 1;
        return;
    }

    const bool botSolid = (j - 1 >= 0) ? (solid[idxCell3D(i, j - 1, k, nx, ny)] != 0) : true;
    const bool topSolid = (j < ny) ? (solid[idxCell3D(i, j, k, nx, ny)] != 0) : !openTop;
    if (botSolid || topSolid) {
        next[idx] = 0.0f;
        nextValid[idx] = 0;
        return;
    }

    float sum = 0.0f;
    int n = 0;
    if (i > 0 && valid[idxV3D(i - 1, j, k, nx, ny)]) { sum += current[idxV3D(i - 1, j, k, nx, ny)]; n++; }
    if (i + 1 < nx && valid[idxV3D(i + 1, j, k, nx, ny)]) { sum += current[idxV3D(i + 1, j, k, nx, ny)]; n++; }
    if (j > 0 && valid[idxV3D(i, j - 1, k, nx, ny)]) { sum += current[idxV3D(i, j - 1, k, nx, ny)]; n++; }
    if (j < ny && valid[idxV3D(i, j + 1, k, nx, ny)]) { sum += current[idxV3D(i, j + 1, k, nx, ny)]; n++; }
    if (k > 0 && valid[idxV3D(i, j, k - 1, nx, ny)]) { sum += current[idxV3D(i, j, k - 1, nx, ny)]; n++; }
    if (k + 1 < nz && valid[idxV3D(i, j, k + 1, nx, ny)]) { sum += current[idxV3D(i, j, k + 1, nx, ny)]; n++; }
    if (n > 0) {
        next[idx] = sum / (float)n;
        nextValid[idx] = 1;
    } else {
        next[idx] = current[idx];
        nextValid[idx] = 0;
    }
}

__global__ void extrapolateWKernel(const float* current, float* next,
                                   const uint8_t* valid, uint8_t* nextValid,
                                   const uint8_t* solid, int nx, int ny, int nz) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = nx * ny * (nz + 1);
    if (idx >= count) return;

    const int k = idx / (nx * ny);
    const int rem = idx - k * nx * ny;
    const int j = rem / nx;
    const int i = rem - j * nx;
    if (valid[idx]) {
        next[idx] = current[idx];
        nextValid[idx] = 1;
        return;
    }

    const bool backSolid = (k - 1 >= 0) ? (solid[idxCell3D(i, j, k - 1, nx, ny)] != 0) : true;
    const bool frontSolid = (k < nz) ? (solid[idxCell3D(i, j, k, nx, ny)] != 0) : true;
    if (backSolid || frontSolid) {
        next[idx] = 0.0f;
        nextValid[idx] = 0;
        return;
    }

    float sum = 0.0f;
    int n = 0;
    if (i > 0 && valid[idxW3D(i - 1, j, k, nx, ny)]) { sum += current[idxW3D(i - 1, j, k, nx, ny)]; n++; }
    if (i + 1 < nx && valid[idxW3D(i + 1, j, k, nx, ny)]) { sum += current[idxW3D(i + 1, j, k, nx, ny)]; n++; }
    if (j > 0 && valid[idxW3D(i, j - 1, k, nx, ny)]) { sum += current[idxW3D(i, j - 1, k, nx, ny)]; n++; }
    if (j + 1 < ny && valid[idxW3D(i, j + 1, k, nx, ny)]) { sum += current[idxW3D(i, j + 1, k, nx, ny)]; n++; }
    if (k > 0 && valid[idxW3D(i, j, k - 1, nx, ny)]) { sum += current[idxW3D(i, j, k - 1, nx, ny)]; n++; }
    if (k < nz && valid[idxW3D(i, j, k + 1, nx, ny)]) { sum += current[idxW3D(i, j, k + 1, nx, ny)]; n++; }
    if (n > 0) {
        next[idx] = sum / (float)n;
        nextValid[idx] = 1;
    } else {
        next[idx] = current[idx];
        nextValid[idx] = 0;
    }
}

__global__ void gridToParticlesKernel(MACWater3D::Particle* particles, int particleCount,
                                      const float* u, const float* v, const float* w,
                                      const float* uDelta, const float* vDelta, const float* wDelta,
                                      int nx, int ny, int nz, float dx, float flipBlend, bool useAPIC) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particleCount) return;

    MACWater3D::Particle p = particles[idx];
    float picU, picV, picW;
    velAtDevice(p.x, p.y, p.z, u, v, w, nx, ny, nz, dx, picU, picV, picW);

    const float blend = useAPIC ? 0.0f : clamp013D(flipBlend);
    const float picWeight = 1.0f - blend;
    if (!useAPIC && blend > 0.0f) {
        const float du = sampleUDevice(uDelta, p.x, p.y, p.z, nx, ny, nz, dx);
        const float dv = sampleVDevice(vDelta, p.x, p.y, p.z, nx, ny, nz, dx);
        const float dw = sampleWDevice(wDelta, p.x, p.y, p.z, nx, ny, nz, dx);
        p.u = picWeight * picU + blend * (p.u + du);
        p.v = picWeight * picV + blend * (p.v + dv);
        p.w = picWeight * picW + blend * (p.w + dw);
    } else {
        p.u = picU;
        p.v = picV;
        p.w = picW;
    }

    if (!useAPIC) {
        p.c00 = p.c01 = p.c02 = 0.0f;
        p.c10 = p.c11 = p.c12 = 0.0f;
        p.c20 = p.c21 = p.c22 = 0.0f;
        particles[idx] = p;
        return;
    }

    const float invDx2 = (dx > 0.0f) ? (1.0f / (dx * dx)) : 0.0f;
    const float apicScale = 3.0f * invDx2;

    {
        const float fx = p.x / dx;
        const float fy = p.y / dx - 0.5f;
        const float fz = p.z / dx - 0.5f;
        const int i0 = clampi3D((int)floorf(fx), 0, nx);
        const int j0 = clampi3D((int)floorf(fy), 0, ny - 1);
        const int k0 = clampi3D((int)floorf(fz), 0, nz - 1);
        const int i1 = min(i0 + 1, nx);
        const int j1 = min(j0 + 1, ny - 1);
        const int k1 = min(k0 + 1, nz - 1);
        const float tx = clampf3D(fx - (float)i0, 0.0f, 1.0f);
        const float ty = clampf3D(fy - (float)j0, 0.0f, 1.0f);
        const float tz = clampf3D(fz - (float)k0, 0.0f, 1.0f);
        float sumDx = 0.0f, sumDy = 0.0f, sumDz = 0.0f;
        for (int dk = 0; dk < 2; ++dk) {
            const int kk = dk == 0 ? k0 : k1;
            const float wz = dk == 0 ? (1.0f - tz) : tz;
            const float pzFace = ((float)kk + 0.5f) * dx;
            for (int dj = 0; dj < 2; ++dj) {
                const int jj = dj == 0 ? j0 : j1;
                const float wy = dj == 0 ? (1.0f - ty) : ty;
                const float pyFace = ((float)jj + 0.5f) * dx;
                for (int di = 0; di < 2; ++di) {
                    const int ii = di == 0 ? i0 : i1;
                    const float wx = di == 0 ? (1.0f - tx) : tx;
                    const float pxFace = (float)ii * dx;
                    const float wght = wx * wy * wz;
                    const float faceVal = u[idxU3D(ii, jj, kk, nx, ny)];
                    sumDx += wght * faceVal * (pxFace - p.x);
                    sumDy += wght * faceVal * (pyFace - p.y);
                    sumDz += wght * faceVal * (pzFace - p.z);
                }
            }
        }
        p.c00 = apicScale * sumDx;
        p.c01 = apicScale * sumDy;
        p.c02 = apicScale * sumDz;
    }

    {
        const float fx = p.x / dx - 0.5f;
        const float fy = p.y / dx;
        const float fz = p.z / dx - 0.5f;
        const int i0 = clampi3D((int)floorf(fx), 0, nx - 1);
        const int j0 = clampi3D((int)floorf(fy), 0, ny);
        const int k0 = clampi3D((int)floorf(fz), 0, nz - 1);
        const int i1 = min(i0 + 1, nx - 1);
        const int j1 = min(j0 + 1, ny);
        const int k1 = min(k0 + 1, nz - 1);
        const float tx = clampf3D(fx - (float)i0, 0.0f, 1.0f);
        const float ty = clampf3D(fy - (float)j0, 0.0f, 1.0f);
        const float tz = clampf3D(fz - (float)k0, 0.0f, 1.0f);
        float sumDx = 0.0f, sumDy = 0.0f, sumDz = 0.0f;
        for (int dk = 0; dk < 2; ++dk) {
            const int kk = dk == 0 ? k0 : k1;
            const float wz = dk == 0 ? (1.0f - tz) : tz;
            const float pzFace = ((float)kk + 0.5f) * dx;
            for (int dj = 0; dj < 2; ++dj) {
                const int jj = dj == 0 ? j0 : j1;
                const float wy = dj == 0 ? (1.0f - ty) : ty;
                const float pyFace = (float)jj * dx;
                for (int di = 0; di < 2; ++di) {
                    const int ii = di == 0 ? i0 : i1;
                    const float wx = di == 0 ? (1.0f - tx) : tx;
                    const float pxFace = ((float)ii + 0.5f) * dx;
                    const float wght = wx * wy * wz;
                    const float faceVal = v[idxV3D(ii, jj, kk, nx, ny)];
                    sumDx += wght * faceVal * (pxFace - p.x);
                    sumDy += wght * faceVal * (pyFace - p.y);
                    sumDz += wght * faceVal * (pzFace - p.z);
                }
            }
        }
        p.c10 = apicScale * sumDx;
        p.c11 = apicScale * sumDy;
        p.c12 = apicScale * sumDz;
    }

    {
        const float fx = p.x / dx - 0.5f;
        const float fy = p.y / dx - 0.5f;
        const float fz = p.z / dx;
        const int i0 = clampi3D((int)floorf(fx), 0, nx - 1);
        const int j0 = clampi3D((int)floorf(fy), 0, ny - 1);
        const int k0 = clampi3D((int)floorf(fz), 0, nz);
        const int i1 = min(i0 + 1, nx - 1);
        const int j1 = min(j0 + 1, ny - 1);
        const int k1 = min(k0 + 1, nz);
        const float tx = clampf3D(fx - (float)i0, 0.0f, 1.0f);
        const float ty = clampf3D(fy - (float)j0, 0.0f, 1.0f);
        const float tz = clampf3D(fz - (float)k0, 0.0f, 1.0f);
        float sumDx = 0.0f, sumDy = 0.0f, sumDz = 0.0f;
        for (int dk = 0; dk < 2; ++dk) {
            const int kk = dk == 0 ? k0 : k1;
            const float wz = dk == 0 ? (1.0f - tz) : tz;
            const float pzFace = (float)kk * dx;
            for (int dj = 0; dj < 2; ++dj) {
                const int jj = dj == 0 ? j0 : j1;
                const float wy = dj == 0 ? (1.0f - ty) : ty;
                const float pyFace = ((float)jj + 0.5f) * dx;
                for (int di = 0; di < 2; ++di) {
                    const int ii = di == 0 ? i0 : i1;
                    const float wx = di == 0 ? (1.0f - tx) : tx;
                    const float pxFace = ((float)ii + 0.5f) * dx;
                    const float wght = wx * wy * wz;
                    const float faceVal = w[idxW3D(ii, jj, kk, nx, ny)];
                    sumDx += wght * faceVal * (pxFace - p.x);
                    sumDy += wght * faceVal * (pyFace - p.y);
                    sumDz += wght * faceVal * (pzFace - p.z);
                }
            }
        }
        p.c20 = apicScale * sumDx;
        p.c21 = apicScale * sumDy;
        p.c22 = apicScale * sumDz;
    }

    particles[idx] = p;
}

__global__ void advectParticlesKernel(MACWater3D::Particle* particles, int particleCount,
                                      const float* u, const float* v, const float* w,
                                      const uint8_t* solid, int nx, int ny, int nz, float dx, float dt) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particleCount) return;

    MACWater3D::Particle p = particles[idx];

    float u1, v1, w1;
    velAtDevice(p.x, p.y, p.z, u, v, w, nx, ny, nz, dx, u1, v1, w1);
    const float domainX = nx * dx;
    const float domainY = ny * dx;
    const float domainZ = nz * dx;
    const float midX = clampf3D(p.x + 0.5f * dt * u1, 0.0f, domainX);
    const float midY = clampf3D(p.y + 0.5f * dt * v1, 0.0f, domainY);
    const float midZ = clampf3D(p.z + 0.5f * dt * w1, 0.0f, domainZ);

    float u2, v2, w2;
    velAtDevice(midX, midY, midZ, u, v, w, nx, ny, nz, dx, u2, v2, w2);
    p.x += dt * u2;
    p.y += dt * v2;
    p.z += dt * w2;
    p.u = u2;
    p.v = v2;
    p.w = w2;
    p.age += dt;

    const int i = clampi3D((int)floorf(p.x / dx), 0, nx - 1);
    const int j = clampi3D((int)floorf(p.y / dx), 0, ny - 1);
    const int k = clampi3D((int)floorf(p.z / dx), 0, nz - 1);
    if (solid[idxCell3D(i, j, k, nx, ny)]) {
        p.age = -1.0f;
    }
    particles[idx] = p;
}

__global__ void enforceParticleBoundsKernel(MACWater3D::Particle* particles, int particleCount,
                                            int nx, int ny, int nz, float dx, int bt, bool openTop) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particleCount) return;

    MACWater3D::Particle p = particles[idx];
    const float minX = (bt + 0.5f) * dx;
    const float maxX = (nx - bt - 0.5f) * dx;
    const float minY = (bt + 0.5f) * dx;
    const float maxY = openTop ? (ny - 0.5f) * dx : (ny - bt - 0.5f) * dx;
    const float minZ = (bt + 0.5f) * dx;
    const float maxZ = (nz - bt - 0.5f) * dx;

    if (p.x < minX) { p.x = minX; if (p.u < 0.0f) p.u = 0.0f; }
    if (p.x > maxX) { p.x = maxX; if (p.u > 0.0f) p.u = 0.0f; }
    if (p.y < minY) { p.y = minY; if (p.v < 0.0f) p.v = 0.0f; }
    if (p.y > maxY) { p.y = maxY; if (p.v > 0.0f) p.v = 0.0f; }
    if (p.z < minZ) { p.z = minZ; if (p.w < 0.0f) p.w = 0.0f; }
    if (p.z > maxZ) { p.z = maxZ; if (p.w > 0.0f) p.w = 0.0f; }
    particles[idx] = p;
}

__global__ void markParticlesInSolidsKernel(MACWater3D::Particle* particles, int particleCount,
                                            const uint8_t* solid, int nx, int ny, int nz, float dx) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particleCount) return;
    MACWater3D::Particle p = particles[idx];
    const int i = clampi3D((int)floorf(p.x / dx), 0, nx - 1);
    const int j = clampi3D((int)floorf(p.y / dx), 0, ny - 1);
    const int k = clampi3D((int)floorf(p.z / dx), 0, nz - 1);
    if (solid[idxCell3D(i, j, k, nx, ny)]) {
        p.age = -1.0f;
    }
    particles[idx] = p;
}

__global__ void markDissipatedParticlesKernel(MACWater3D::Particle* particles, int particleCount,
                                              float keepProb, unsigned int seedBase) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particleCount) return;
    unsigned int seed = seedBase ^ (unsigned int)(idx * 9781U + 17U);
    if (rand01FromSeed(seed) > keepProb) {
        particles[idx].age = -1.0f;
    }
}

__global__ void countParticlesPerCellKernel(const MACWater3D::Particle* particles, int particleCount,
                                            const uint8_t* solid, int* cellCounts, uint8_t* occ,
                                            int nx, int ny, int nz, float dx) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particleCount) return;
    const MACWater3D::Particle p = particles[idx];
    const int i = clampi3D((int)floorf(p.x / dx), 0, nx - 1);
    const int j = clampi3D((int)floorf(p.y / dx), 0, ny - 1);
    const int k = clampi3D((int)floorf(p.z / dx), 0, nz - 1);
    const int cell = idxCell3D(i, j, k, nx, ny);
    if (solid[cell]) return;
    atomicAdd(cellCounts + cell, 1);
    occ[cell] = 1;
}

__global__ void buildReseedRegionKernel(const uint8_t* solid, const uint8_t* occ, uint8_t* region,
                                        int nx, int ny, int nz) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int cellCount = nx * ny * nz;
    if (idx >= cellCount) return;

    const int k = idx / (nx * ny);
    const int rem = idx - k * nx * ny;
    const int j = rem / nx;
    const int i = rem - j * nx;

    if (solid[idx]) { region[idx] = 0; return; }
    if (occ[idx]) { region[idx] = 1; return; }

    const bool Isnear =
        (i > 0 && occ[idxCell3D(i - 1, j, k, nx, ny)]) ||
        (i + 1 < nx && occ[idxCell3D(i + 1, j, k, nx, ny)]) ||
        (j > 0 && occ[idxCell3D(i, j - 1, k, nx, ny)]) ||
        (j + 1 < ny && occ[idxCell3D(i, j + 1, k, nx, ny)]) ||
        (k > 0 && occ[idxCell3D(i, j, k - 1, nx, ny)]) ||
        (k + 1 < nz && occ[idxCell3D(i, j, k + 1, nx, ny)]);
    region[idx] = Isnear ? 1 : 0;
}

__global__ void spawnReseedParticlesKernel(MACWater3D::Particle* particles, int* particleCount,
                                           int particleCapacity,
                                           const uint8_t* region, const uint8_t* solid, const int* cellCounts,
                                           const float* u, const float* v, const float* w,
                                           int nx, int ny, int nz, float dx, int ppc, int stepCounter,
                                           int maxParticleCount) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int cellCount = nx * ny * nz;
    if (idx >= cellCount) return;
    if (!region[idx] || solid[idx]) return;

    const int have = cellCounts[idx];
    if (have >= ppc) return;

    const int k = idx / (nx * ny);
    const int rem = idx - k * nx * ny;
    const int j = rem / nx;
    const int i = rem - j * nx;

    for (int n = have; n < ppc; ++n) {
        const int slot = atomicAdd(particleCount, 1);
        if (slot >= particleCapacity || slot >= maxParticleCount) {
            atomicSub(particleCount, 1);
            break;
        }

        unsigned int seed =
            (unsigned int)(i + 92821U * j + 68917U * k + 131U * (n + 1) + 17U * stepCounter + 1U);
        MACWater3D::Particle p;
        p.x = (i + 0.1f + 0.8f * rand01FromSeed(seed)) * dx;
        p.y = (j + 0.1f + 0.8f * rand01FromSeed(seed)) * dx;
        p.z = (k + 0.1f + 0.8f * rand01FromSeed(seed)) * dx;
        velAtDevice(p.x, p.y, p.z, u, v, w, nx, ny, nz, dx, p.u, p.v, p.w);
        p.age = 0.0f;
        particles[slot] = p;
    }
}

__global__ void addWaterSourceSphereKernel(MACWater3D::Particle* particles, int* particleCount,
                                           int particleCapacity,
                                           int maxParticleCount,
                                           uint8_t* liquid, const uint8_t* solid,
                                           int nx, int ny, int nz, float dx, int particlesPerCell,
                                           float cx, float cy, float cz, float radius,
                                           float velX, float velY, float velZ, int stepCounter) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int cellCount = nx * ny * nz;
    if (idx >= cellCount) return;

    const int k = idx / (nx * ny);
    const int rem = idx - k * nx * ny;
    const int j = rem / nx;
    const int i = rem - j * nx;

    if (solid[idx]) return;

    const float x = (i + 0.5f) * dx;
    const float y = (j + 0.5f) * dx;
    const float z = (k + 0.5f) * dx;
    const float d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy) + (z - cz) * (z - cz);
    if (d2 > radius * radius) return;

    liquid[idx] = 1;

    for (int n = 0; n < particlesPerCell; ++n) {
        const int slot = atomicAdd(particleCount, 1);
        if (slot >= particleCapacity || slot >= maxParticleCount) {
            atomicSub(particleCount, 1);
            break;
        }
        unsigned int seed =
            (unsigned int)(i + 73856093U * j + 19349663U * k + 83492791U * (n + 1) + stepCounter * 131U);
        MACWater3D::Particle p;
        p.x = (i + 0.1f + 0.8f * rand01FromSeed(seed)) * dx;
        p.y = (j + 0.1f + 0.8f * rand01FromSeed(seed)) * dx;
        p.z = (k + 0.1f + 0.8f * rand01FromSeed(seed)) * dx;
        p.u = velX;
        p.v = velY;
        p.w = velZ;
        p.age = 0.0f;
        particles[slot] = p;
    }
}

__global__ void scatterMassKernel(const MACWater3D::Particle* particles, int particleCount,
                                  const uint8_t* solid, float* mass,
                                  int nx, int ny, int nz, float dx) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particleCount) return;

    const MACWater3D::Particle p = particles[idx];
    const float fx = p.x / dx - 0.5f;
    const float fy = p.y / dx - 0.5f;
    const float fz = p.z / dx - 0.5f;

    int i0 = clampi3D((int)floorf(fx), 0, nx - 1);
    int j0 = clampi3D((int)floorf(fy), 0, ny - 1);
    int k0 = clampi3D((int)floorf(fz), 0, nz - 1);
    int i1 = min(i0 + 1, nx - 1);
    int j1 = min(j0 + 1, ny - 1);
    int k1 = min(k0 + 1, nz - 1);
    float tx = clampf3D(fx - (float)i0, 0.0f, 1.0f);
    float ty = clampf3D(fy - (float)j0, 0.0f, 1.0f);
    float tz = clampf3D(fz - (float)k0, 0.0f, 1.0f);

    float weights[8] = {
        (1.0f - tx) * (1.0f - ty) * (1.0f - tz),
        tx * (1.0f - ty) * (1.0f - tz),
        (1.0f - tx) * ty * (1.0f - tz),
        tx * ty * (1.0f - tz),
        (1.0f - tx) * (1.0f - ty) * tz,
        tx * (1.0f - ty) * tz,
        (1.0f - tx) * ty * tz,
        tx * ty * tz
    };
    int ids[8] = {
        idxCell3D(i0, j0, k0, nx, ny),
        idxCell3D(i1, j0, k0, nx, ny),
        idxCell3D(i0, j1, k0, nx, ny),
        idxCell3D(i1, j1, k0, nx, ny),
        idxCell3D(i0, j0, k1, nx, ny),
        idxCell3D(i1, j0, k1, nx, ny),
        idxCell3D(i0, j1, k1, nx, ny),
        idxCell3D(i1, j1, k1, nx, ny)
    };

    float sumW = 0.0f;
    for (int n = 0; n < 8; ++n) {
        if (solid[ids[n]]) weights[n] = 0.0f;
        sumW += weights[n];
    }
    if (sumW <= 1e-12f) return;
    const float inv = 1.0f / sumW;
    for (int n = 0; n < 8; ++n) {
        atomicAdd(mass + ids[n], weights[n] * inv);
    }
}

__global__ void finalizeDebugFieldsKernel(const uint8_t* solid, const float* mass,
                                          const float* u, const float* v, const float* w,
                                          float* water, float* divergence, float* speed,
                                          int nx, int ny, int nz, float dx, float invPpc) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int cellCount = nx * ny * nz;
    if (idx >= cellCount) return;

    if (solid[idx]) {
        water[idx] = 0.0f;
        divergence[idx] = 0.0f;
        speed[idx] = 0.0f;
        return;
    }

    const int k = idx / (nx * ny);
    const int rem = idx - k * nx * ny;
    const int j = rem / nx;
    const int i = rem - j * nx;

    water[idx] = clamp013D(mass[idx] * invPpc);

    float uL = u[idxU3D(i, j, k, nx, ny)];
    float uR = u[idxU3D(i + 1, j, k, nx, ny)];
    float vB = v[idxV3D(i, j, k, nx, ny)];
    float vT = v[idxV3D(i, j + 1, k, nx, ny)];
    float wBk = w[idxW3D(i, j, k, nx, ny)];
    float wFr = w[idxW3D(i, j, k + 1, nx, ny)];
    divergence[idx] = (uR - uL + vT - vB + wFr - wBk) / dx;

    const float cx = (i + 0.5f) * dx;
    const float cy = (j + 0.5f) * dx;
    const float cz = (k + 0.5f) * dx;
    float uc, vc, wc;
    velAtDevice(cx, cy, cz, u, v, w, nx, ny, nz, dx, uc, vc, wc);
    speed[idx] = sqrtf(uc * uc + vc * vc + wc * wc);
}

void compactParticles(MACWater3DCudaBackend* b) {
    int particleCount = 0;
    CUDA_CHECK(cudaMemcpy(&particleCount, b->d_particleCount, sizeof(int), cudaMemcpyDeviceToHost));
    thrust::device_ptr<MACWater3D::Particle> begin = thrust::device_pointer_cast(b->d_particles);
    thrust::device_ptr<MACWater3D::Particle> newEnd =
        thrust::remove_if(thrust::device, begin, begin + particleCount, DeviceParticleDead());
    particleCount = (int)(newEnd - begin);
    CUDA_CHECK(cudaMemcpy(b->d_particleCount, &particleCount, sizeof(int), cudaMemcpyHostToDevice));
}

void launchApplyBoundary(MACWater3DCudaBackend* b) {
    applyBoundaryUKernel<<<(b->uCount + kThreads - 1) / kThreads, kThreads>>>(b->d_u, b->d_solid, b->nx, b->ny, b->nz);
    applyBoundaryVKernel<<<(b->vCount + kThreads - 1) / kThreads, kThreads>>>(b->d_v, b->d_solid, b->nx, b->ny, b->nz, b->params.openTop);
    applyBoundaryWKernel<<<(b->wCount + kThreads - 1) / kThreads, kThreads>>>(b->d_w, b->d_solid, b->nx, b->ny, b->nz);
}

void launchBuildLiquid(MACWater3DCudaBackend* b, bool applyDilations) {
    fillKernel<uint8_t><<<(b->cellCount + kThreads - 1) / kThreads, kThreads>>>(b->d_liquid, b->cellCount, (uint8_t)0);
    int particleCount = 0;
    CUDA_CHECK(cudaMemcpy(&particleCount, b->d_particleCount, sizeof(int), cudaMemcpyDeviceToHost));
    if (particleCount > 0) {
        markLiquidFromParticlesKernel<<<(particleCount + kThreads - 1) / kThreads, kThreads>>>(
            b->d_particles, particleCount, b->d_solid, b->d_liquid, b->nx, b->ny, b->nz, b->dx);
    }
    if (!applyDilations) return;

    for (int it = 0; it < std::max(0, b->params.maskDilations); ++it) {
        dilateLiquidKernel<<<(b->cellCount + kThreads - 1) / kThreads, kThreads>>>(
            b->d_solid, b->d_liquid, b->d_liquidTmp, b->nx, b->ny, b->nz);
        std::swap(b->d_liquid, b->d_liquidTmp);
    }
}

void launchRasterize(MACWater3DCudaBackend* b) {
    fillKernel<float><<<(b->cellCount + kThreads - 1) / kThreads, kThreads>>>(b->d_mass, b->cellCount, 0.0f);
    int particleCount = 0;
    CUDA_CHECK(cudaMemcpy(&particleCount, b->d_particleCount, sizeof(int), cudaMemcpyDeviceToHost));
    if (particleCount > 0) {
        scatterMassKernel<<<(particleCount + kThreads - 1) / kThreads, kThreads>>>(
            b->d_particles, particleCount, b->d_solid, b->d_mass, b->nx, b->ny, b->nz, b->dx);
    }
    const float invPpc = 1.0f / (float)std::max(1, b->params.particlesPerCell);
    finalizeDebugFieldsKernel<<<(b->cellCount + kThreads - 1) / kThreads, kThreads>>>(
        b->d_solid, b->d_mass, b->d_u, b->d_v, b->d_w,
        b->d_water, b->d_divergence, b->d_speed,
        b->nx, b->ny, b->nz, b->dx, invPpc);
}

void ensureAllocatedForSim(MACWater3DCudaBackend* b, const MACWater3D& sim) {
    const int cellCount = std::max(1, sim.nx * sim.ny * sim.nz);
    const int uCount = std::max(1, (sim.nx + 1) * sim.ny * sim.nz);
    const int vCount = std::max(1, sim.nx * (sim.ny + 1) * sim.nz);
    const int wCount = std::max(1, sim.nx * sim.ny * (sim.nz + 1));

    const bool dimsChanged =
        b->nx != sim.nx || b->ny != sim.ny || b->nz != sim.nz ||
        b->cellCount != cellCount || b->uCount != uCount || b->vCount != vCount || b->wCount != wCount;

    if (!dimsChanged) return;

    destroyBackendArrays(b);
    b->nx = sim.nx;
    b->ny = sim.ny;
    b->nz = sim.nz;
    b->dx = sim.dx;
    b->dt = sim.dt;
    b->params = sim.params;
    b->cellCount = cellCount;
    b->uCount = uCount;
    b->vCount = vCount;
    b->wCount = wCount;
    b->particleCapacity = 0;

    allocArray(b->d_u, uCount);
    allocArray(b->d_v, vCount);
    allocArray(b->d_w, wCount);
    allocArray(b->d_uWeight, uCount);
    allocArray(b->d_vWeight, vCount);
    allocArray(b->d_wWeight, wCount);
    allocArray(b->d_uPrev, uCount);
    allocArray(b->d_vPrev, vCount);
    allocArray(b->d_wPrev, wCount);
    allocArray(b->d_uDelta, uCount);
    allocArray(b->d_vDelta, vCount);
    allocArray(b->d_wDelta, wCount);
    allocArray(b->d_uTmp, uCount);
    allocArray(b->d_vTmp, vCount);
    allocArray(b->d_wTmp, wCount);
    allocArray(b->d_pressure, cellCount);
    allocArray(b->d_pressureTmp, cellCount);
    allocArray(b->d_rhs, cellCount);
    allocArray(b->d_water, cellCount);
    allocArray(b->d_divergence, cellCount);
    allocArray(b->d_speed, cellCount);
    allocArray(b->d_mass, cellCount);
    allocArray(b->d_liquid, cellCount);
    allocArray(b->d_liquidTmp, cellCount);
    allocArray(b->d_solid, cellCount);
    allocArray(b->d_solidUser, cellCount);
    allocArray(b->d_validU, uCount);
    allocArray(b->d_validV, vCount);
    allocArray(b->d_validW, wCount);
    allocArray(b->d_validUTmp, uCount);
    allocArray(b->d_validVTmp, vCount);
    allocArray(b->d_validWTmp, wCount);
    allocArray(b->d_occ, cellCount);
    allocArray(b->d_region, cellCount);
    allocArray(b->d_cellCounts, cellCount);
    allocArray(b->d_particleCount, 1);

    const int zeroParticleCount = 0;
    CUDA_CHECK(cudaMemcpy(b->d_particleCount, &zeroParticleCount, sizeof(int), cudaMemcpyHostToDevice));

    const int initialParticleCapacity = std::max(cellCount * std::max(1, sim.params.particlesPerCell), 1024);
    ensureParticleCapacity(b, initialParticleCapacity);
}

void uploadHostStateToDevice(MACWater3DCudaBackend* b, const MACWater3D& sim) {
    b->dx = sim.dx;
    b->dt = sim.dt;
    b->params = sim.params;

    ensureAllocatedForSim(b, sim);
    ensureParticleCapacity(b, std::max((int)sim.particles.size(), 1));

    CUDA_CHECK(cudaMemcpy(b->d_u, sim.u.data(), (std::size_t)b->uCount * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b->d_v, sim.v.data(), (std::size_t)b->vCount * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b->d_w, sim.w.data(), (std::size_t)b->wCount * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b->d_pressure, sim.pressure.data(), (std::size_t)b->cellCount * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b->d_rhs, sim.rhs.data(), (std::size_t)b->cellCount * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b->d_water, sim.water.data(), (std::size_t)b->cellCount * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b->d_divergence, sim.divergence.data(), (std::size_t)b->cellCount * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b->d_speed, sim.speed.data(), (std::size_t)b->cellCount * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b->d_liquid, sim.liquid.data(), (std::size_t)b->cellCount * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b->d_solid, sim.solid.data(), (std::size_t)b->cellCount * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b->d_solidUser, sim.userSolidMask().data(), (std::size_t)b->cellCount * sizeof(uint8_t), cudaMemcpyHostToDevice));

    int particleCount = (int)sim.particles.size();
    CUDA_CHECK(cudaMemcpy(b->d_particleCount, &particleCount, sizeof(int), cudaMemcpyHostToDevice));
    if (particleCount > 0) {
        CUDA_CHECK(cudaMemcpy(b->d_particles, sim.particles.data(),
                              (std::size_t)particleCount * sizeof(MACWater3D::Particle),
                              cudaMemcpyHostToDevice));
    }
}

void downloadDeviceStateToHost(MACWater3DCudaBackend* b, MACWater3D& sim) {
    int particleCount = 0;
    CUDA_CHECK(cudaMemcpy(&particleCount, b->d_particleCount, sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(sim.u.data(), b->d_u, (std::size_t)b->uCount * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sim.v.data(), b->d_v, (std::size_t)b->vCount * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sim.w.data(), b->d_w, (std::size_t)b->wCount * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sim.pressure.data(), b->d_pressure, (std::size_t)b->cellCount * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sim.rhs.data(), b->d_rhs, (std::size_t)b->cellCount * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sim.water.data(), b->d_water, (std::size_t)b->cellCount * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sim.divergence.data(), b->d_divergence, (std::size_t)b->cellCount * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sim.speed.data(), b->d_speed, (std::size_t)b->cellCount * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sim.liquid.data(), b->d_liquid, (std::size_t)b->cellCount * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sim.solid.data(), b->d_solid, (std::size_t)b->cellCount * sizeof(uint8_t), cudaMemcpyDeviceToHost));

    sim.particles.resize((std::size_t)particleCount);
    if (particleCount > 0) {
        CUDA_CHECK(cudaMemcpy(sim.particles.data(), b->d_particles,
                              (std::size_t)particleCount * sizeof(MACWater3D::Particle),
                              cudaMemcpyDeviceToHost));
    }

    sim.targetMass = (float)particleCount;
    if (sim.desiredMass < 0.0f && sim.targetMass > 0.0f) {
        sim.desiredMass = sim.targetMass;
    }
}

void downloadVelocityStateToHost(MACWater3DCudaBackend* b, MACWater3D& sim) {
    CUDA_CHECK(cudaMemcpy(sim.u.data(), b->d_u, (std::size_t)b->uCount * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sim.v.data(), b->d_v, (std::size_t)b->vCount * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sim.w.data(), b->d_w, (std::size_t)b->wCount * sizeof(float), cudaMemcpyDeviceToHost));
}

void downloadVolumeStateToHost(MACWater3DCudaBackend* b, MACWater3D& sim) {
    CUDA_CHECK(cudaMemcpy(sim.water.data(), b->d_water, (std::size_t)b->cellCount * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sim.solid.data(), b->d_solid, (std::size_t)b->cellCount * sizeof(uint8_t), cudaMemcpyDeviceToHost));
}

void downloadPressureStateToHost(MACWater3DCudaBackend* b, MACWater3D& sim) {
    CUDA_CHECK(cudaMemcpy(sim.pressure.data(), b->d_pressure, (std::size_t)b->cellCount * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sim.rhs.data(), b->d_rhs, (std::size_t)b->cellCount * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sim.solid.data(), b->d_solid, (std::size_t)b->cellCount * sizeof(uint8_t), cudaMemcpyDeviceToHost));
}

void downloadDerivedStateToHost(MACWater3DCudaBackend* b, MACWater3D& sim) {
    CUDA_CHECK(cudaMemcpy(sim.divergence.data(), b->d_divergence, (std::size_t)b->cellCount * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sim.speed.data(), b->d_speed, (std::size_t)b->cellCount * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sim.solid.data(), b->d_solid, (std::size_t)b->cellCount * sizeof(uint8_t), cudaMemcpyDeviceToHost));
}

void downloadParticlesToHost(MACWater3DCudaBackend* b, MACWater3D& sim) {
    int particleCount = 0;
    CUDA_CHECK(cudaMemcpy(&particleCount, b->d_particleCount, sizeof(int), cudaMemcpyDeviceToHost));
    sim.particles.resize((std::size_t)particleCount);
    if (particleCount > 0) {
        CUDA_CHECK(cudaMemcpy(sim.particles.data(), b->d_particles,
                              (std::size_t)particleCount * sizeof(MACWater3D::Particle),
                              cudaMemcpyDeviceToHost));
    }

    sim.targetMass = (float)particleCount;
    if (sim.desiredMass < 0.0f && sim.targetMass > 0.0f) {
        sim.desiredMass = sim.targetMass;
    }
}

// Temporary narrow host bridge for pressure projection parity.
// Keep the CPU multigrid path intact for now, but only move the state that the
// CPU projection actually consumes and produces instead of round-tripping the
// entire simulation state in the middle of the CUDA step.
void downloadPressureBridgeInputsToHost(MACWater3DCudaBackend* b, MACWater3D& sim) {
    CUDA_CHECK(cudaMemcpy(sim.u.data(), b->d_u, (std::size_t)b->uCount * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sim.v.data(), b->d_v, (std::size_t)b->vCount * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sim.w.data(), b->d_w, (std::size_t)b->wCount * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sim.pressure.data(), b->d_pressure, (std::size_t)b->cellCount * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sim.liquid.data(), b->d_liquid, (std::size_t)b->cellCount * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sim.solid.data(), b->d_solid, (std::size_t)b->cellCount * sizeof(uint8_t), cudaMemcpyDeviceToHost));

    int particleCount = 0;
    CUDA_CHECK(cudaMemcpy(&particleCount, b->d_particleCount, sizeof(int), cudaMemcpyDeviceToHost));
    sim.targetMass = (float)particleCount;
    if (sim.desiredMass < 0.0f && sim.targetMass > 0.0f) {
        sim.desiredMass = sim.targetMass;
    }
}

void uploadPressureBridgeOutputsToDevice(MACWater3DCudaBackend* b, const MACWater3D& sim) {
    CUDA_CHECK(cudaMemcpy(b->d_u, sim.u.data(), (std::size_t)b->uCount * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b->d_v, sim.v.data(), (std::size_t)b->vCount * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b->d_w, sim.w.data(), (std::size_t)b->wCount * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b->d_pressure, sim.pressure.data(), (std::size_t)b->cellCount * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b->d_rhs, sim.rhs.data(), (std::size_t)b->cellCount * sizeof(float), cudaMemcpyHostToDevice));
}

// Temporary narrow host bridge for particle relaxation parity.
void downloadRelaxBridgeInputsToHost(MACWater3DCudaBackend* b, MACWater3D& sim) {
    int particleCount = 0;
    CUDA_CHECK(cudaMemcpy(&particleCount, b->d_particleCount, sizeof(int), cudaMemcpyDeviceToHost));
    sim.particles.resize((std::size_t)particleCount);
    if (particleCount > 0) {
        CUDA_CHECK(cudaMemcpy(sim.particles.data(), b->d_particles,
                              (std::size_t)particleCount * sizeof(MACWater3D::Particle),
                              cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaMemcpy(sim.solid.data(), b->d_solid,
                          (std::size_t)b->cellCount * sizeof(uint8_t),
                          cudaMemcpyDeviceToHost));
}

void uploadRelaxBridgeOutputsToDevice(MACWater3DCudaBackend* b, const MACWater3D& sim) {
    const int particleCount = (int)sim.particles.size();
    ensureParticleCapacity(b, std::max(particleCount, 1));
    CUDA_CHECK(cudaMemcpy(b->d_particleCount, &particleCount, sizeof(int), cudaMemcpyHostToDevice));
    if (particleCount > 0) {
        CUDA_CHECK(cudaMemcpy(b->d_particles, sim.particles.data(),
                              (std::size_t)particleCount * sizeof(MACWater3D::Particle),
                              cudaMemcpyHostToDevice));
    }
}

void runCudaPressureJacobi(MACWater3DCudaBackend* b, MACWater3D& sim) {
    const auto solveStart = std::chrono::high_resolution_clock::now();

    const int pressureIters = std::max(1, b->params.pressureIters);
    const float dxSafe = std::max(1.0e-6f, b->dx);
    const float dtSafe = std::max(1.0e-6f, b->dt);
    const float invDx = 1.0f / dxSafe;
    const float invDt = 1.0f / dtSafe;
    const float dx2 = dxSafe * dxSafe;
    const float omega = clampf3D(b->params.pressureOmega, 0.0f, 1.0f);

    fillKernel<float><<<(b->cellCount + kThreads - 1) / kThreads, kThreads>>>(b->d_rhs, b->cellCount, 0.0f);
    buildPressureRhsKernel<<<(b->cellCount + kThreads - 1) / kThreads, kThreads>>>(
        b->d_u, b->d_v, b->d_w,
        b->d_solid, b->d_liquid,
        b->d_rhs, b->nx, b->ny, b->nz, invDx, invDt);

    for (int it = 0; it < pressureIters; ++it) {
        pressureJacobiKernel<<<(b->cellCount + kThreads - 1) / kThreads, kThreads>>>(
            b->d_pressure, b->d_pressureTmp, b->d_rhs,
            b->d_solid, b->d_liquid,
            b->nx, b->ny, b->nz, dx2, omega, b->params.openTop);
        std::swap(b->d_pressure, b->d_pressureTmp);
    }

    const float scale = dtSafe / dxSafe;
    applyPressureUKernel<<<(b->uCount + kThreads - 1) / kThreads, kThreads>>>(
        b->d_u, b->d_pressure, b->d_solid, b->d_liquid,
        b->nx, b->ny, b->nz, scale);
    applyPressureVKernel<<<(b->vCount + kThreads - 1) / kThreads, kThreads>>>(
        b->d_v, b->d_pressure, b->d_solid, b->d_liquid,
        b->nx, b->ny, b->nz, scale, b->params.openTop);
    applyPressureWKernel<<<(b->wCount + kThreads - 1) / kThreads, kThreads>>>(
        b->d_w, b->d_pressure, b->d_solid, b->d_liquid,
        b->nx, b->ny, b->nz, scale);
    launchApplyBoundary(b);
    CUDA_CHECK(cudaDeviceSynchronize());

    const auto solveEnd = std::chrono::high_resolution_clock::now();
    sim.lastPressureSolveMs = std::chrono::duration<float, std::milli>(solveEnd - solveStart).count();
    sim.lastPressureIterations = pressureIters;
    b->lastUsedDeviceJacobi = true;
}

void runCudaStepInternal(MACWater3DCudaBackend* b, MACWater3D& sim) {
    b->dt = sim.dt;
    b->dx = sim.dx;
    b->params = sim.params;

    rebuildSolidsKernel<<<(b->cellCount + kThreads - 1) / kThreads, kThreads>>>(
        b->d_solidUser, b->d_solid, b->nx, b->ny, b->nz,
        std::max(1, std::min({b->nx, b->ny, b->nz}) / 2 - 1) > 0
            ? clampi3D(b->params.borderThickness, 1, std::max(1, std::min({b->nx, b->ny, b->nz}) / 2 - 1))
            : 1,
        b->params.openTop);

    int particleCount = 0;
    CUDA_CHECK(cudaMemcpy(&particleCount, b->d_particleCount, sizeof(int), cudaMemcpyDeviceToHost));
    if (particleCount > 0) {
        markParticlesInSolidsKernel<<<(particleCount + kThreads - 1) / kThreads, kThreads>>>(
            b->d_particles, particleCount, b->d_solid, b->nx, b->ny, b->nz, b->dx);
        enforceParticleBoundsKernel<<<(particleCount + kThreads - 1) / kThreads, kThreads>>>(
            b->d_particles, particleCount, b->nx, b->ny, b->nz, b->dx,
            clampi3D(b->params.borderThickness, 1, std::max(1, std::min({b->nx, b->ny, b->nz}) / 2 - 1)),
            b->params.openTop);
        compactParticles(b);
        CUDA_CHECK(cudaMemcpy(&particleCount, b->d_particleCount, sizeof(int), cudaMemcpyDeviceToHost));
    }

    if (particleCount == 0) {
        fillKernel<float><<<(b->uCount + kThreads - 1) / kThreads, kThreads>>>(b->d_u, b->uCount, 0.0f);
        fillKernel<float><<<(b->vCount + kThreads - 1) / kThreads, kThreads>>>(b->d_v, b->vCount, 0.0f);
        fillKernel<float><<<(b->wCount + kThreads - 1) / kThreads, kThreads>>>(b->d_w, b->wCount, 0.0f);
        launchBuildLiquid(b, true);
        launchApplyBoundary(b);
        launchRasterize(b);
        CUDA_CHECK(cudaDeviceSynchronize());
        return;
    }

    fillKernel<float><<<(b->uCount + kThreads - 1) / kThreads, kThreads>>>(b->d_u, b->uCount, 0.0f);
    fillKernel<float><<<(b->vCount + kThreads - 1) / kThreads, kThreads>>>(b->d_v, b->vCount, 0.0f);
    fillKernel<float><<<(b->wCount + kThreads - 1) / kThreads, kThreads>>>(b->d_w, b->wCount, 0.0f);
    fillKernel<float><<<(b->uCount + kThreads - 1) / kThreads, kThreads>>>(b->d_uWeight, b->uCount, 0.0f);
    fillKernel<float><<<(b->vCount + kThreads - 1) / kThreads, kThreads>>>(b->d_vWeight, b->vCount, 0.0f);
    fillKernel<float><<<(b->wCount + kThreads - 1) / kThreads, kThreads>>>(b->d_wWeight, b->wCount, 0.0f);
    scatterParticlesKernel<<<(particleCount + kThreads - 1) / kThreads, kThreads>>>(
        b->d_particles, particleCount,
        b->d_u, b->d_v, b->d_w,
        b->d_uWeight, b->d_vWeight, b->d_wWeight,
        b->nx, b->ny, b->nz, b->dx, b->params.useAPIC);
    normalizeKernel<<<(b->uCount + kThreads - 1) / kThreads, kThreads>>>(b->d_u, b->d_uWeight, b->uCount);
    normalizeKernel<<<(b->vCount + kThreads - 1) / kThreads, kThreads>>>(b->d_v, b->d_vWeight, b->vCount);
    normalizeKernel<<<(b->wCount + kThreads - 1) / kThreads, kThreads>>>(b->d_w, b->d_wWeight, b->wCount);
    launchApplyBoundary(b);

    launchBuildLiquid(b, true);

    const float damp = (b->params.velDamping > 0.0f) ? expf(-b->params.velDamping * b->dt) : 1.0f;
    applyGravityDampingKernel<<<(b->vCount + kThreads - 1) / kThreads, kThreads>>>(
        b->d_v, b->nx, b->ny, b->nz, b->dt, b->params.gravity, damp);
    applyDampingKernel<<<(b->uCount + kThreads - 1) / kThreads, kThreads>>>(b->d_u, b->uCount, damp);
    applyDampingKernel<<<(b->wCount + kThreads - 1) / kThreads, kThreads>>>(b->d_w, b->wCount, damp);
    launchApplyBoundary(b);

    if (b->params.viscosity > 0.0f && b->params.diffuseIters > 0) {
        const float alphaInvDx2 = (b->params.viscosity * b->dt) / (b->dx * b->dx);
        copyKernel<<<(b->uCount + kThreads - 1) / kThreads, kThreads>>>(b->d_u, b->d_uPrev, b->uCount);
        copyKernel<<<(b->vCount + kThreads - 1) / kThreads, kThreads>>>(b->d_v, b->d_vPrev, b->vCount);
        copyKernel<<<(b->wCount + kThreads - 1) / kThreads, kThreads>>>(b->d_w, b->d_wPrev, b->wCount);
        for (int it = 0; it < b->params.diffuseIters; ++it) {
            diffuseUKernel<<<(b->uCount + kThreads - 1) / kThreads, kThreads>>>(
                b->d_u, b->d_uPrev, b->d_uTmp, b->d_solid, b->nx, b->ny, b->nz, alphaInvDx2, b->params.diffuseOmega);
            std::swap(b->d_u, b->d_uTmp);
            launchApplyBoundary(b);
        }
        for (int it = 0; it < b->params.diffuseIters; ++it) {
            diffuseVKernel<<<(b->vCount + kThreads - 1) / kThreads, kThreads>>>(
                b->d_v, b->d_vPrev, b->d_vTmp, b->d_solid, b->nx, b->ny, b->nz, alphaInvDx2, b->params.diffuseOmega, b->params.openTop);
            std::swap(b->d_v, b->d_vTmp);
            launchApplyBoundary(b);
        }
        for (int it = 0; it < b->params.diffuseIters; ++it) {
            diffuseWKernel<<<(b->wCount + kThreads - 1) / kThreads, kThreads>>>(
                b->d_w, b->d_wPrev, b->d_wTmp, b->d_solid, b->nx, b->ny, b->nz, alphaInvDx2, b->params.diffuseOmega);
            std::swap(b->d_w, b->d_wTmp);
            launchApplyBoundary(b);
        }
    }

    copyKernel<<<(b->uCount + kThreads - 1) / kThreads, kThreads>>>(b->d_u, b->d_uPrev, b->uCount);
    copyKernel<<<(b->vCount + kThreads - 1) / kThreads, kThreads>>>(b->d_v, b->d_vPrev, b->vCount);
    copyKernel<<<(b->wCount + kThreads - 1) / kThreads, kThreads>>>(b->d_w, b->d_wPrev, b->wCount);

    if (static_cast<MACWater3D::PressureSolverMode>(b->params.pressureSolverMode) ==
        MACWater3D::PressureSolverMode::Jacobi) {
        runCudaPressureJacobi(b, sim);
    } else {
        CUDA_CHECK(cudaDeviceSynchronize());
        downloadPressureBridgeInputsToHost(b, sim);
        sim.projectLiquidForCudaBridge();
        uploadPressureBridgeOutputsToDevice(b, sim);
        b->lastUsedDeviceJacobi = false;
    }

    computeDeltaKernel<<<(b->uCount + kThreads - 1) / kThreads, kThreads>>>(b->d_u, b->d_uPrev, b->d_uDelta, b->uCount);
    computeDeltaKernel<<<(b->vCount + kThreads - 1) / kThreads, kThreads>>>(b->d_v, b->d_vPrev, b->d_vDelta, b->vCount);
    computeDeltaKernel<<<(b->wCount + kThreads - 1) / kThreads, kThreads>>>(b->d_w, b->d_wPrev, b->d_wDelta, b->wCount);

    seedValidUKernel<<<(b->uCount + kThreads - 1) / kThreads, kThreads>>>(b->d_u, b->d_validU, b->d_uWeight, b->d_solid, b->nx, b->ny, b->nz);
    seedValidVKernel<<<(b->vCount + kThreads - 1) / kThreads, kThreads>>>(b->d_v, b->d_validV, b->d_vWeight, b->d_solid, b->nx, b->ny, b->nz, b->params.openTop);
    seedValidWKernel<<<(b->wCount + kThreads - 1) / kThreads, kThreads>>>(b->d_w, b->d_validW, b->d_wWeight, b->d_solid, b->nx, b->ny, b->nz);
    for (int it = 0; it < std::max(0, b->params.extrapolationIters); ++it) {
        extrapolateUKernel<<<(b->uCount + kThreads - 1) / kThreads, kThreads>>>(
            b->d_u, b->d_uTmp, b->d_validU, b->d_validUTmp, b->d_solid, b->nx, b->ny, b->nz);
        std::swap(b->d_u, b->d_uTmp);
        std::swap(b->d_validU, b->d_validUTmp);

        extrapolateVKernel<<<(b->vCount + kThreads - 1) / kThreads, kThreads>>>(
            b->d_v, b->d_vTmp, b->d_validV, b->d_validVTmp, b->d_solid, b->nx, b->ny, b->nz, b->params.openTop);
        std::swap(b->d_v, b->d_vTmp);
        std::swap(b->d_validV, b->d_validVTmp);

        extrapolateWKernel<<<(b->wCount + kThreads - 1) / kThreads, kThreads>>>(
            b->d_w, b->d_wTmp, b->d_validW, b->d_validWTmp, b->d_solid, b->nx, b->ny, b->nz);
        std::swap(b->d_w, b->d_wTmp);
        std::swap(b->d_validW, b->d_validWTmp);
    }
    launchApplyBoundary(b);

    gridToParticlesKernel<<<(particleCount + kThreads - 1) / kThreads, kThreads>>>(
        b->d_particles, particleCount, b->d_u, b->d_v, b->d_w,
        b->d_uDelta, b->d_vDelta, b->d_wDelta,
        b->nx, b->ny, b->nz, b->dx, b->params.flipBlend, b->params.useAPIC);
    advectParticlesKernel<<<(particleCount + kThreads - 1) / kThreads, kThreads>>>(
        b->d_particles, particleCount, b->d_u, b->d_v, b->d_w, b->d_solid,
        b->nx, b->ny, b->nz, b->dx, b->dt);
    enforceParticleBoundsKernel<<<(particleCount + kThreads - 1) / kThreads, kThreads>>>(
        b->d_particles, particleCount, b->nx, b->ny, b->nz, b->dx,
        clampi3D(b->params.borderThickness, 1, std::max(1, std::min({b->nx, b->ny, b->nz}) / 2 - 1)),
        b->params.openTop);
    markParticlesInSolidsKernel<<<(particleCount + kThreads - 1) / kThreads, kThreads>>>(
        b->d_particles, particleCount, b->d_solid, b->nx, b->ny, b->nz, b->dx);
    compactParticles(b);

    launchBuildLiquid(b, false);

    fillKernel<int><<<(b->cellCount + kThreads - 1) / kThreads, kThreads>>>(b->d_cellCounts, b->cellCount, 0);
    fillKernel<uint8_t><<<(b->cellCount + kThreads - 1) / kThreads, kThreads>>>(b->d_occ, b->cellCount, (uint8_t)0);
    particleCount = 0;
    CUDA_CHECK(cudaMemcpy(&particleCount, b->d_particleCount, sizeof(int), cudaMemcpyDeviceToHost));
    if (particleCount > 0) {
        countParticlesPerCellKernel<<<(particleCount + kThreads - 1) / kThreads, kThreads>>>(
            b->d_particles, particleCount, b->d_solid, b->d_cellCounts, b->d_occ,
            b->nx, b->ny, b->nz, b->dx);
    }

    buildReseedRegionKernel<<<(b->cellCount + kThreads - 1) / kThreads, kThreads>>>(
        b->d_solid, b->d_occ, b->d_region, b->nx, b->ny, b->nz);

    const int target = std::max(1, b->params.particlesPerCell);
    int softMaxParticles = (b->params.maxParticles > 0)
        ? b->params.maxParticles
        : std::max(particleCount + 256, target);
    if (sim.desiredMass > 0.0f) {
        const int desiredCap = std::max(target, (int)std::ceil(sim.desiredMass * 1.15f));
        softMaxParticles = std::min(softMaxParticles, desiredCap);
    }

    const int remainingSpawnBudget = std::max(0, softMaxParticles - particleCount);
    const int baseNewCap = std::max(128, b->cellCount / 64);
    const int maxNewPerStep = std::min(remainingSpawnBudget, std::min(baseNewCap, 2048));

    if (maxNewPerStep > 0) {
        const int maxParticleCount = particleCount + maxNewPerStep;
        ensureParticleCapacity(b, maxParticleCount);

        spawnReseedParticlesKernel<<<(b->cellCount + kThreads - 1) / kThreads, kThreads>>>(
            b->d_particles, b->d_particleCount, b->particleCapacity,
            b->d_region, b->d_solid, b->d_cellCounts,
            b->d_u, b->d_v, b->d_w,
            b->nx, b->ny, b->nz, b->dx, b->params.particlesPerCell, sim.stepCounter, maxParticleCount);
    }

    if (b->params.reseedRelaxIters > 0 && b->params.reseedRelaxStrength > 0.0f) {
        CUDA_CHECK(cudaDeviceSynchronize());
        downloadRelaxBridgeInputsToHost(b, sim);
        sim.relaxParticlesForCuda(b->params.reseedRelaxIters, b->params.reseedRelaxStrength);
        uploadRelaxBridgeOutputsToDevice(b, sim);
    }

    if (b->params.waterDissipation < 0.999999f) {
        particleCount = 0;
        CUDA_CHECK(cudaMemcpy(&particleCount, b->d_particleCount, sizeof(int), cudaMemcpyDeviceToHost));
        if (particleCount > 0) {
            const int particleCountBeforeDissipation = particleCount;
            const float dtRef = 0.02f;
            const float keepProb = powf(clamp013D(b->params.waterDissipation), b->dt / std::max(1e-6f, dtRef));
            markDissipatedParticlesKernel<<<(particleCount + kThreads - 1) / kThreads, kThreads>>>(
                b->d_particles, particleCount, keepProb, (unsigned int)(sim.stepCounter * 9781 + 17));
            compactParticles(b);
            CUDA_CHECK(cudaMemcpy(&particleCount, b->d_particleCount, sizeof(int), cudaMemcpyDeviceToHost));
            if (sim.desiredMass >= 0.0f && particleCount < particleCountBeforeDissipation) {
                sim.desiredMass = std::max(0.0f, sim.desiredMass - (float)(particleCountBeforeDissipation - particleCount));
            }
        }
    }

    launchBuildLiquid(b, true);
    launchRasterize(b);
    CUDA_CHECK(cudaDeviceSynchronize());
}

}  // namespace

MACWater3DCudaBackend* water3dCreateCudaBackend() {
    int deviceCount = 0;
    const cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount <= 0) {
        return nullptr;
    }
    return new MACWater3DCudaBackend();
}

void water3dDestroyCudaBackend(MACWater3DCudaBackend* backend) {
    if (backend == nullptr) return;
    destroyBackendArrays(backend);
    delete backend;
}

void water3dCudaReset(MACWater3DCudaBackend* backend, MACWater3D& sim) {
    ensureAllocatedForSim(backend, sim);
    uploadHostStateToDevice(backend, sim);
    CUDA_CHECK(cudaDeviceSynchronize());
    resetPressureStats(sim);
    applyCudaStats(backend, sim, 0.0f);
}

void water3dCudaSetParams(MACWater3DCudaBackend* backend, MACWater3D& sim) {
    ensureAllocatedForSim(backend, sim);
    backend->dx = sim.dx;
    backend->dt = sim.dt;
    backend->params = sim.params;
    rebuildSolidsKernel<<<(backend->cellCount + kThreads - 1) / kThreads, kThreads>>>(
        backend->d_solidUser, backend->d_solid, backend->nx, backend->ny, backend->nz,
        clampi3D(backend->params.borderThickness, 1, std::max(1, std::min({backend->nx, backend->ny, backend->nz}) / 2 - 1)),
        backend->params.openTop);
    int particleCount = 0;
    CUDA_CHECK(cudaMemcpy(&particleCount, backend->d_particleCount, sizeof(int), cudaMemcpyDeviceToHost));
    if (particleCount > 0) {
        markParticlesInSolidsKernel<<<(particleCount + kThreads - 1) / kThreads, kThreads>>>(
            backend->d_particles, particleCount, backend->d_solid, backend->nx, backend->ny, backend->nz, backend->dx);
        enforceParticleBoundsKernel<<<(particleCount + kThreads - 1) / kThreads, kThreads>>>(
            backend->d_particles, particleCount, backend->nx, backend->ny, backend->nz, backend->dx,
            clampi3D(backend->params.borderThickness, 1, std::max(1, std::min({backend->nx, backend->ny, backend->nz}) / 2 - 1)),
            backend->params.openTop);
        compactParticles(backend);
    }
    launchBuildLiquid(backend, true);
    launchApplyBoundary(backend);
    launchRasterize(backend);
    CUDA_CHECK(cudaDeviceSynchronize());
    resetPressureStats(sim);
    applyCudaStats(backend, sim, 0.0f);
}

void water3dCudaSetVoxelSolids(MACWater3DCudaBackend* backend, MACWater3D& sim) {
    ensureAllocatedForSim(backend, sim);
    CUDA_CHECK(cudaMemcpy(backend->d_solidUser, sim.userSolidMask().data(),
                          (std::size_t)backend->cellCount * sizeof(uint8_t),
                          cudaMemcpyHostToDevice));
    water3dCudaSetParams(backend, sim);
}

void water3dCudaAddWaterSourceSphere(MACWater3DCudaBackend* backend, MACWater3D& sim,
                                     const MACWater3D::Vec3& center, float radius,
                                     const MACWater3D::Vec3& velocity) {
    ensureAllocatedForSim(backend, sim);
    backend->dt = sim.dt;
    backend->dx = sim.dx;
    backend->params = sim.params;
    int particleCountBefore = 0;
    CUDA_CHECK(cudaMemcpy(&particleCountBefore, backend->d_particleCount, sizeof(int), cudaMemcpyDeviceToHost));

    const int ppc = std::max(1, backend->params.particlesPerCell);
    const float dxSafe = std::max(1e-6f, backend->dx);
    const float rCells = std::max(0.0f, radius / dxSafe);
    const float sphereCellEstimate = (4.0f / 3.0f) * 3.14159265358979323846f * rCells * rCells * rCells;
    int estimatedSpawn = (int)std::ceil(std::max(1.0f, sphereCellEstimate) * (float)ppc);
    estimatedSpawn = std::min(estimatedSpawn, backend->cellCount * ppc);

    int maxParticleCount = backend->params.maxParticles > 0 ? backend->params.maxParticles : (particleCountBefore + estimatedSpawn);
    if (backend->params.maxParticles <= 0) {
        maxParticleCount = particleCountBefore + estimatedSpawn;
    }
    maxParticleCount = std::max(maxParticleCount, particleCountBefore);

    int requestedCapacity = particleCountBefore + estimatedSpawn;
    requestedCapacity = std::max(requestedCapacity, particleCountBefore + ppc);
    if (backend->params.maxParticles > 0) {
        requestedCapacity = std::min(requestedCapacity, backend->params.maxParticles);
    }
    requestedCapacity = std::max(requestedCapacity, particleCountBefore);
    ensureParticleCapacity(backend, requestedCapacity);

    addWaterSourceSphereKernel<<<(backend->cellCount + kThreads - 1) / kThreads, kThreads>>>(
        backend->d_particles, backend->d_particleCount, backend->particleCapacity, maxParticleCount,
        backend->d_liquid, backend->d_solid, backend->nx, backend->ny, backend->nz,
        backend->dx, ppc,
        center.x, center.y, center.z, radius,
        velocity.x, velocity.y, velocity.z, sim.stepCounter);

    int particleCountAfter = 0;
    CUDA_CHECK(cudaMemcpy(&particleCountAfter, backend->d_particleCount, sizeof(int), cudaMemcpyDeviceToHost));
    if (particleCountAfter > 0) {
        enforceParticleBoundsKernel<<<(particleCountAfter + kThreads - 1) / kThreads, kThreads>>>(
            backend->d_particles, particleCountAfter, backend->nx, backend->ny, backend->nz, backend->dx,
            clampi3D(backend->params.borderThickness, 1, std::max(1, std::min({backend->nx, backend->ny, backend->nz}) / 2 - 1)),
            backend->params.openTop);
        markParticlesInSolidsKernel<<<(particleCountAfter + kThreads - 1) / kThreads, kThreads>>>(
            backend->d_particles, particleCountAfter, backend->d_solid, backend->nx, backend->ny, backend->nz, backend->dx);
        compactParticles(backend);
        CUDA_CHECK(cudaMemcpy(&particleCountAfter, backend->d_particleCount, sizeof(int), cudaMemcpyDeviceToHost));
    }

    if (sim.desiredMass >= 0.0f && particleCountAfter > particleCountBefore) {
        sim.desiredMass += (float)(particleCountAfter - particleCountBefore);
    }

    launchBuildLiquid(backend, true);
    launchRasterize(backend);
    CUDA_CHECK(cudaDeviceSynchronize());
    resetPressureStats(sim);
    applyCudaStats(backend, sim, 0.0f);
}

void water3dCudaStep(MACWater3DCudaBackend* backend, MACWater3D& sim) {
    const auto start = std::chrono::high_resolution_clock::now();
    ++sim.stepCounter;
    resetPressureStats(sim);
    runCudaStepInternal(backend, sim);
    const auto end = std::chrono::high_resolution_clock::now();
    const float stepMs = std::chrono::duration<float, std::milli>(end - start).count();
    applyCudaStats(backend, sim, stepMs);
}

void water3dCudaDownloadHostAll(MACWater3DCudaBackend* backend, MACWater3D& sim) {
    downloadDeviceStateToHost(backend, sim);
}

void water3dCudaDownloadHostVolume(MACWater3DCudaBackend* backend, MACWater3D& sim) {
    downloadVolumeStateToHost(backend, sim);
}

void water3dCudaDownloadHostParticles(MACWater3DCudaBackend* backend, MACWater3D& sim) {
    downloadParticlesToHost(backend, sim);
}

void water3dCudaDownloadHostDebugField(MACWater3DCudaBackend* backend, MACWater3D& sim,
                                       MACWater3D::DebugField field) {
    switch (field) {
        case MACWater3D::DebugField::Water:
            downloadVolumeStateToHost(backend, sim);
            break;
        case MACWater3D::DebugField::Pressure:
            downloadPressureStateToHost(backend, sim);
            break;
        case MACWater3D::DebugField::Divergence:
        case MACWater3D::DebugField::Speed:
            downloadDerivedStateToHost(backend, sim);
            break;
    }
}

#endif
