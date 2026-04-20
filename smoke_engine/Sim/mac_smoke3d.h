#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "pressure_solver3d.h"
#include "sim_stage_timing.h"

struct MACSmoke3D {
    struct Vec3 {
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
    };

    enum class SliceAxis : int {
        XY = 0,
        XZ = 1,
        YZ = 2,
    };

    enum class DebugField : int {
        Smoke = 0,
        Temperature = 1,
        Pressure = 2,
        Divergence = 3,
        Speed = 4,
    };

    enum class PressureSolverMode : int {
        Multigrid = 0,
        RBGS = 1,
        Jacobi = 2,
    };

    struct Params {
        float smokeDissipation = 0.999f;
        float tempDissipation = 0.995f;
        float velDamping = 0.5f;

        float viscosity = 1e-4f;
        float smokeDiffusivity = 0.0f;
        float tempDiffusivity = 0.0f;

        float gravity = 9.81f;
        float ambientTempK = 293.15f;
        float buoyancyScale = 1.0f;

        int borderThickness = 1;
        int pressureIters = 120;
        int pressureMGVCycles = 16;
        int pressureMGCoarseIters = 80;
        int diffuseIters = 16;
        int pressureSolverMode = (int)PressureSolverMode::Multigrid;

        float pressureTol = 1e-6f;
        float pressureOmega = 1.7f;
        float pressureMGOmega = 1.4f;
        float pressureMGRelativeTol = 1.0e-5f;
        float diffuseOmega = 0.8f;

        bool openTop = true;
    };

    struct SliceData {
        int width = 0;
        int height = 0;
        std::vector<float> values;
        std::vector<uint8_t> solid;
    };

    struct Stats {
        int nx = 0;
        int ny = 0;
        int nz = 0;
        int activeCells = 0;
        float maxSpeed = 0.0f;
        float maxDivergence = 0.0f;
        float dt = 0.0f;
        float lastStepMs = 0.0f;
        float pressureMs = 0.0f;
        int pressureIters = 0;
        std::size_t bytesAllocated = 0;
        const char* backendName = "CPU Smoke 3D";
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

    std::vector<float> u0;
    std::vector<float> v0;
    std::vector<float> w0;

    std::vector<float> uTmp;
    std::vector<float> vTmp;
    std::vector<float> wTmp;

    std::vector<float> pressure;
    std::vector<float> pressureTmp;
    std::vector<float> rhs;

    std::vector<float> smoke;
    std::vector<float> smoke0;
    std::vector<float> temp;
    std::vector<float> temp0;
    std::vector<float> cellTmp;

    std::vector<float> divergence;
    std::vector<float> speed;
    bool derivedFieldsDirty = true;

    std::vector<uint8_t> solid;
    std::vector<uint8_t> solidUser;
    std::vector<uint8_t> fluidMask;
    int fluidCellCount = 0;
    bool topologyDirty = true;
    bool pressureOperatorDirty = true;
    float lastPressureSolveMs = 0.0f;
    int lastPressureIterations = 0;

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

    struct SliceIJWorkList {
        struct Entry {
            int i = 0;
            int j = 0;
        };

        std::vector<int> offsets;
        std::vector<Entry> entries;

        void clear() {
            offsets.clear();
            entries.clear();
        }

        std::size_t size() const { return entries.size(); }
    };

    DiffusionStencilSet uDiffusionStencil;
    DiffusionStencilSet vDiffusionStencil;
    DiffusionStencilSet wDiffusionStencil;
    bool diffusionStencilDirty = true;

    SliceIJWorkList activeCellsByK;
    SliceIJWorkList activeUFacesByK;
    SliceIJWorkList activeVFacesByK;
    SliceIJWorkList activeWFacesByK;

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

    DiffusionScratchSet uDiffusionScratch;
    DiffusionScratchSet vDiffusionScratch;
    DiffusionScratchSet wDiffusionScratch;

    MACSmoke3D(int NX, int NY, int NZ, float DX, float DT);

    void reset();
    void reset(int NX, int NY, int NZ, float DX, float DT);
    void setDt(float newDt) { dt = newDt; lastStats.dt = dt; }
    void setParams(const Params& newParams);
    void step();

    void addSmokeSourceSphere(const Vec3& center, float radius, float amount, const Vec3& velocity);
    void addHeatSourceSphere(const Vec3& center, float radius, float amount);
    void setVoxelSolids(const std::vector<uint8_t>& mask);

    SliceData copyDebugSlice(SliceAxis axis, int index, DebugField field);
    const Stats& stats() const { return lastStats; }

protected:
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
    void rebuildAdvectionWorklists();
    void rebuildDiffusionStencils();
    void applyBoundary();

    float sampleCellCentered(const std::vector<float>& field, float x, float y, float z) const;
    float sampleCellCenteredOpenTop(const std::vector<float>& field, float x, float y, float z, float outsideValue) const;
    float sampleU(const std::vector<float>& field, float x, float y, float z) const;
    float sampleV(const std::vector<float>& field, float x, float y, float z) const;
    float sampleW(const std::vector<float>& field, float x, float y, float z) const;
    void velAt(float x, float y, float z,
               const std::vector<float>& fu,
               const std::vector<float>& fv,
               const std::vector<float>& fw,
               float& outU, float& outV, float& outW) const;

    void advectVelocity();
    void advectScalars();
    void addBuoyancy();
    void diffuseVelocityImplicit();
    void diffuseScalarImplicit(std::vector<float>& phi,
                               std::vector<float>& phi0,
                               float diffusivity,
                               float dissipation);
    void project();
    void rasterizeDebugFields();
    void ensureDerivedDebugFields();
    bool hasActiveVelocity(float eps = 1.0e-6f) const;
    bool hasActiveScalar(const std::vector<float>& field, float eps = 1.0e-6f) const;
    void clearVelocityState();
    void clearScalarState(std::vector<float>& field);
    void clearDerivedDebugFields();
    void updateIdleStats(float stepMs);
    void updateStats(float stepMs);
};
