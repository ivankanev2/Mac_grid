#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "pressure_solver3d.h"
#include "sim_stage_timing.h"

class MACWater3DBackend;

struct MACWater3D {
    struct Vec3 {
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
    };

    struct Particle {
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
        float u = 0.0f;
        float v = 0.0f;
        float w = 0.0f;
        float age = 0.0f;

        // APIC affine velocity matrix rows (u, v, w) in world units.
        float c00 = 0.0f, c01 = 0.0f, c02 = 0.0f;
        float c10 = 0.0f, c11 = 0.0f, c12 = 0.0f;
        float c20 = 0.0f, c21 = 0.0f, c22 = 0.0f;
    };

    enum class SliceAxis : int {
        XY = 0,
        XZ = 1,
        YZ = 2,
    };

    enum class DebugField : int {
        Water = 0,
        Pressure = 1,
        Divergence = 2,
        Speed = 3,
    };

    enum class PressureSolverMode : int {
        Multigrid = 0,
        RBGS = 1,
        Jacobi = 2,
    };

    enum class BackendPreference : int {
        Auto = 0,
        CPU = 1,
        CUDA = 2,
    };

    struct Params {
        float waterDissipation = 1.0f;
        float gravity = -9.8f;
        float velDamping = 0.0f;
        float viscosity = 5e-4f;
        float flipBlend = 0.1f;

        int particlesPerCell = 2;
        int borderThickness = 2;
        int maxParticles = 500000;

        int maskDilations = 0;
        int extrapolationIters = 10;
        int pressureIters = 200;
        int pressureMGVCycles = 200;
        int pressureMGCoarseIters = 80;
        int diffuseIters = 20;
        int pressureSolverMode = (int)PressureSolverMode::Multigrid;
        int reseedRelaxIters = 2;

        float pressureTol = 1e-10f;
        float pressureOmega = 1.7f;
        float pressureMGOmega = 1.4f;
        float pressureMGRelativeTol = 1.0e-5f;
        float diffuseOmega = 0.8f;
        float reseedRelaxStrength = 0.45f;
        float volumePreserveStrength = 0.05f;

        bool openTop = true;
        bool useAPIC = true;
        bool volumePreserveRhsMean = true;
    };

    struct SliceData {
        int width = 0;
        int height = 0;
        std::vector<float> values;
        std::vector<uint8_t> solid;
    };

    struct Stats {
        bool cudaEnabled = false;
        bool backendReady = false;
        int nx = 0;
        int ny = 0;
        int nz = 0;
        int particleCount = 0;
        int liquidCells = 0;
        float maxSpeed = 0.0f;
        float maxDivergence = 0.0f;
        float dt = 0.0f;
        float lastStepMs = 0.0f;
        float pressureMs = 0.0f;
        int pressureIters = 0;
        float targetMass = 0.0f;
        float desiredMass = -1.0f;
        std::size_t bytesAllocated = 0;
        const char* backendName = "CPU MAC 3D";
        SimStageTimings timings;
    };

    int nx = 0;
    int ny = 0;
    int nz = 0;
    float dx = 1.0f;
    float dt = 0.02f;

    Params params;
    Stats lastStats;

    PressureSolver3D pressurePoisson;

    std::vector<float> u;
    std::vector<float> v;
    std::vector<float> w;

    std::vector<float> pressure;
    std::vector<float> rhs;
    std::vector<float> water;
    std::vector<float> divergence;
    std::vector<float> speed;
    bool derivedFieldsDirty = true;

    std::vector<uint8_t> liquid;
    std::vector<uint8_t> solid;
    bool topologyDirty = true;
    float lastPressureSolveMs = 0.0f;
    int lastPressureIterations = 0;

    std::vector<Particle> particles;

    float targetMass = 0.0f;
    float desiredMass = -1.0f;

    int stepCounter = 0;
    BackendPreference backendPreference = BackendPreference::Auto;

    MACWater3D(int NX, int NY, int NZ, float DX, float DT);
    ~MACWater3D();

    void reset();
    void reset(int NX, int NY, int NZ, float DX, float DT);
    void setDt(float newDt) { dt = newDt; lastStats.dt = dt; }
    void setParams(const Params& newParams);
    void setBackendPreference(BackendPreference newPreference);
    BackendPreference backendPreferenceMode() const { return backendPreference; }
    bool isCudaAvailable() const;
    void step();

    void addWaterSourceSphere(const Vec3& center, float radius, const Vec3& velocity);
    void setVoxelSolids(const std::vector<uint8_t>& mask);

    SliceData copyDebugSlice(SliceAxis axis, int index, DebugField field);
    const Stats& stats() const { return lastStats; }
    const std::vector<uint8_t>& userSolidMask() const { return solidUser; }
    void refreshStats(float stepMs);
    void syncHostAll();
    void syncHostVolume();
    void syncHostParticles();
    void syncHostDebugField(DebugField field);

    bool isCudaEnabled() const;
    bool hasFeatureParityWith2D() const {
        // The 3D path now has its own reusable multigrid-capable pressure solve.
        // Keep this simple boolean for the UI while the remaining validation work
        // happens elsewhere.
        return true;
    }

protected:
    std::vector<uint8_t> solidUser;

    std::vector<float> uWeight;
    std::vector<float> vWeight;
    std::vector<float> wWeight;

    std::vector<float> uPrev;
    std::vector<float> vPrev;
    std::vector<float> wPrev;

    std::vector<float> uDelta;
    std::vector<float> vDelta;
    std::vector<float> wDelta;

    std::vector<float> uTmp;
    std::vector<float> vTmp;
    std::vector<float> wTmp;
    std::vector<float> pressureTmp;

    std::vector<uint8_t> validU;
    std::vector<uint8_t> validV;
    std::vector<uint8_t> validW;
    std::vector<uint8_t> validUNext;
    std::vector<uint8_t> validVNext;
    std::vector<uint8_t> validWNext;
    std::vector<int> extrapFrontierU;
    std::vector<int> extrapFrontierV;
    std::vector<int> extrapFrontierW;
    std::vector<int> extrapNextFrontierU;
    std::vector<int> extrapNextFrontierV;
    std::vector<int> extrapNextFrontierW;

    struct PressureRegionScratch {
        int i0 = 0;
        int i1 = 0;
        int j0 = 0;
        int j1 = 0;
        int k0 = 0;
        int k1 = 0;
        bool previousBoxValid = false;
        int prevI0 = 0;
        int prevI1 = 0;
        int prevJ0 = 0;
        int prevJ1 = 0;
        int prevK0 = 0;
        int prevK1 = 0;
        std::vector<uint8_t> solid;
        std::vector<uint8_t> fluid;
        std::vector<float> rhs;
        std::vector<float> pressure;
        std::vector<float> tmp;

        void ensureSize(std::size_t count) {
            if (solid.size() != count) solid.resize(count, (uint8_t)0);
            if (fluid.size() != count) fluid.resize(count, (uint8_t)0);
            if (rhs.size() != count) rhs.resize(count, 0.0f);
            if (pressure.size() != count) pressure.resize(count, 0.0f);
        }

        void ensureTmpSize(std::size_t count) {
            if (tmp.size() != count) tmp.resize(count, 0.0f);
        }

        void clear() {
            i0 = i1 = j0 = j1 = k0 = k1 = 0;
            previousBoxValid = false;
            prevI0 = prevI1 = prevJ0 = prevJ1 = prevK0 = prevK1 = 0;
            solid.clear();
            fluid.clear();
            rhs.clear();
            pressure.clear();
            tmp.clear();
        }
    };

    PressureRegionScratch pressureRegion;
    std::vector<int> pressureComponentLabel;
    std::vector<int> pressureComponentQueue;
    std::vector<int> pressureComponentCells;

    struct DiffusionStencilSet {
        std::vector<int> face;
        std::vector<int> xm;
        std::vector<int> xp;
        std::vector<int> ym;
        std::vector<int> yp;
        std::vector<int> zm;
        std::vector<int> zp;
        std::vector<uint8_t> neighborCount;

        void clear() {
            face.clear();
            xm.clear();
            xp.clear();
            ym.clear();
            yp.clear();
            zm.clear();
            zp.clear();
            neighborCount.clear();
        }

        std::size_t size() const { return face.size(); }
    };

    struct DiffusionScratchSet {
        std::vector<float> r;
        std::vector<float> z;
        std::vector<float> p;
        std::vector<float> q;

        void ensureSize(std::size_t count) {
            if (r.size() != count) r.resize(count, 0.0f);
            if (z.size() != count) z.resize(count, 0.0f);
            if (p.size() != count) p.resize(count, 0.0f);
            if (q.size() != count) q.resize(count, 0.0f);
        }
    };

    DiffusionStencilSet uDiffusionStencil;
    DiffusionStencilSet vDiffusionStencil;
    DiffusionStencilSet wDiffusionStencil;
    DiffusionScratchSet uDiffusionScratch;
    DiffusionScratchSet vDiffusionScratch;
    DiffusionScratchSet wDiffusionScratch;
    bool diffusionStencilDirty = true;

    std::vector<int> reseedCounts;
    std::vector<uint8_t> reseedOccupied;
    std::vector<uint8_t> reseedRegion;
    std::vector<int> relaxBucketCounts;
    std::vector<int> relaxBucketOffsets;
    std::vector<int> relaxBucketCursor;
    std::vector<int> relaxBucketParticles;

    inline int idxCell(int i, int j, int k) const {
        return i + nx * (j + ny * k);
    }

    inline int idxU(int i, int j, int k) const {
        return i + (nx + 1) * (j + ny * k);
    }

    inline int idxV(int i, int j, int k) const {
        return i + nx * (j + (ny + 1) * k);
    }

    inline int idxW(int i, int j, int k) const {
        return i + nx * (j + ny * k);
    }

    inline bool isSolidCell(int i, int j, int k) const {
        return solid[(std::size_t)idxCell(i, j, k)] != 0;
    }

    void rebuildBorderSolids();
    void applyBoundary();
    void removeParticlesInSolids();
    void enforceParticleBounds();

    float sampleCellCentered(const std::vector<float>& field, float x, float y, float z) const;
    float sampleU(const std::vector<float>& field, float x, float y, float z) const;
    float sampleV(const std::vector<float>& field, float x, float y, float z) const;
    float sampleW(const std::vector<float>& field, float x, float y, float z) const;
    void velAt(float x, float y, float z,
               const std::vector<float>& fu,
               const std::vector<float>& fv,
               const std::vector<float>& fw,
               float& outU, float& outV, float& outW) const;

    void particleToGrid();
    void buildLiquidMask(bool applyDilations = true);
    void rebuildDiffusionStencils();
    void applyExternalForces();
    void diffuseVelocityImplicit();
    void projectLiquid();
    void extrapolateVelocity();
    void gridToParticles();

    void advectParticles();
    void applyDissipation();
    void reseedParticles();
    void relaxParticles(int iters, float strength);

    void rasterizeWaterField();
    void rasterizeDebugFields();
    void ensureDerivedDebugFields();
    void updateStats(float stepMs);

    void setParamsCpu();
    void stepCpu();
    void addWaterSourceSphereCpu(const Vec3& center, float radius, const Vec3& velocity);
    void setVoxelSolidsCpu(const std::vector<uint8_t>& mask);
    MACWater3DBackend* activeBackend() const;

    std::unique_ptr<MACWater3DBackend> cpuBackendImpl;
    std::unique_ptr<MACWater3DBackend> cudaBackendImpl;

    bool cudaHostVelocityDirty = false;
    bool cudaHostPressureDirty = false;
    bool cudaHostVolumeDirty = false;
    bool cudaHostDerivedDirty = false;
    bool cudaHostParticlesDirty = false;

    void markCudaHostStateDirtyAll();
    void clearCudaHostStateDirtyAll();

    friend class Water3DBackendCpu;
    friend class Water3DBackendCuda;

public:
    void relaxParticlesForCuda(int iters, float strength) { relaxParticles(iters, strength); }
    void projectLiquidForCudaBridge() { projectLiquid(); }
};