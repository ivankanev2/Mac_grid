#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "pressure_solver3d.h"

struct MACWater3DCudaBackend;

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
        int diffuseIters = 20;
        int pressureSolverMode = (int)PressureSolverMode::Multigrid;
        int reseedRelaxIters = 2;

        float pressureTol = 1e-10f;
        float pressureOmega = 1.7f;
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
        float targetMass = 0.0f;
        float desiredMass = -1.0f;
        std::size_t bytesAllocated = 0;
        const char* backendName = "CPU MAC 3D";
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

    std::vector<uint8_t> liquid;
    std::vector<uint8_t> solid;

    std::vector<Particle> particles;

    float targetMass = 0.0f;
    float desiredMass = -1.0f;

    int stepCounter = 0;
    BackendPreference backendPreference = BackendPreference::Auto;
    MACWater3DCudaBackend* cudaBackend = nullptr;

    MACWater3D(int NX, int NY, int NZ, float DX, float DT);
    ~MACWater3D();

    void reset();
    void reset(int NX, int NY, int NZ, float DX, float DT);
    void setDt(float newDt) { dt = newDt; lastStats.dt = dt; }
    void setParams(const Params& newParams);
    void setBackendPreference(BackendPreference newPreference);
    BackendPreference backendPreferenceMode() const { return backendPreference; }
    bool isCudaAvailable() const { return cudaBackend != nullptr; }
    void step();

    void addWaterSourceSphere(const Vec3& center, float radius, const Vec3& velocity);
    void setVoxelSolids(const std::vector<uint8_t>& mask);

    SliceData copyDebugSlice(SliceAxis axis, int index, DebugField field) const;
    const Stats& stats() const { return lastStats; }
    const std::vector<uint8_t>& userSolidMask() const { return solidUser; }
    void refreshStats(float stepMs);

    bool isCudaEnabled() const {
        return cudaBackend != nullptr && backendPreference != BackendPreference::CPU;
    }
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
    void applyExternalForces();
    void diffuseVelocityImplicit();
    void projectLiquid();
    void extrapolateVelocity();
    void gridToParticles();

    void advectParticles();
    void applyDissipation();
    void reseedParticles();
    void relaxParticles(int iters, float strength);

    void rasterizeDebugFields();
    void updateStats(float stepMs);
    MACWater3DCudaBackend* activeCudaBackend() const;

public:
    void relaxParticlesForCuda(int iters, float strength) { relaxParticles(iters, strength); }
    void projectLiquidForCudaBridge() { projectLiquid(); }
};