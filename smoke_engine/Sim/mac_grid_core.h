#pragma once
#include <vector>
#include <cstdint>
#include <algorithm>

struct MACGridCore {
    int nx, ny;
    float dx, dt;

    // Staggered velocity
    std::vector<float> u, v;
    std::vector<float> u0, v0;

    // Cell-centered fields
    std::vector<float> p;
    std::vector<float> div;
    std::vector<float> rhs;

    // Solids
    std::vector<uint8_t> solid;

    struct FrameStats {
        float dt = 0.0f;
        float maxDivBefore = 0.0f;
        float maxDivAfter  = 0.0f;
        float maxFaceSpeedBefore = 0.0f;
        float maxFaceSpeedAfter  = 0.0f;
        int   pressureIters = 0;
        float pressureMs    = 0.0f;
    };

    MACGridCore(int NX, int NY, float DX, float DT);
    void resetCore();


    // void setOpenTop(bool v) { openTopBC = v; markPressureMatrixDirty(); mgDirty = true; }

    void setOpenTopBC(bool enabled); // testing stuff
    bool getOpenTop() const { return openTopBC; }


    // inline bool isDirichletP(int i, int j) const {
    // // Pin pressure on the top row when openTop is enabled
    // // doesnt currently work well, so its temporarely disabled
    // // return openTopBC && (j == ny - 1) && !isSolid(i, j);
    // return false;
    // }

    float maxAbsDiv() const;
    float maxFaceSpeed() const;
    void setDt(float newDt) { dt = newDt; }

    const std::vector<uint8_t>& solidMask() const { return solid; }

    inline int idxP(int i,int j) const { return i + nx*j; }
    inline int idxU(int i,int j) const { return i + (nx+1)*j; }
    inline int idxV(int i,int j) const { return i + nx*j; }
    inline bool isSolid(int i,int j) const { return solid[idxP(i,j)] != 0; }

    
    static inline float clampf(float x, float a, float b) {
        if (!std::isfinite(x)) return 0.0f;
        if (x < a) return a;
        if (x > b) return b;
        return x;
    }

    void worldToCell(float x, float y, int& i, int& j) const;
    float sampleCellCentered(const std::vector<float>& f, float x, float y) const;
    float sampleU(const std::vector<float>& fu, float x, float y) const;
    float sampleV(const std::vector<float>& fv, float x, float y) const;
    void velAt(float x, float y,
               const std::vector<float>& fu,
               const std::vector<float>& fv,
               float& outUx, float& outVy) const;

    void advectVelocity();
    void computeDivergence();

    void solvePressurePCG(int maxIters = 200, float tol = 1e-6f);
    void solvePressureMG(int vcycles = 20, float tol = 1e-6f);
    void debugCheckMGvsPCGOperator();

    void setSolidCell(int i, int j, bool s);

    inline bool isDirichletP(int i, int j) const {
    // Top row is pressure Dirichlet when open
    return openTopBC && (j == ny - 1) && !isSolid(i,j);
    }

    void applyLaplacian(const std::vector<float>& x, std::vector<float>& Ax) const;
    void project();

    void advectScalarSemiLagrangian(std::vector<float>& phi,
                                    std::vector<float>& phi0,
                                    float dissipation);
    void advectScalarMacCormack(std::vector<float>& phi,
                                std::vector<float>& phi0,
                                float dissipation);

    void invalidatePressureMatrix() { markPressureMatrixDirty(); mgDirty = true; }
    const FrameStats& getStats() const { return stats; }

    

private:
    // ---- Pressure solve caches ----
    bool pressureMatrixDirty = true;
    float invDx2_cache = 0.0f;

    std::vector<int> lapL, lapR, lapB, lapT;
    std::vector<float> lapDiagInv;

    std::vector<float> pcg_r, pcg_z, pcg_d, pcg_q, pcg_Ap;

    void markPressureMatrixDirty() { pressureMatrixDirty = true; }
    void ensurePressureMatrix();
    void ensurePCGBuffers();
    void removePressureMean();

    bool openTopBC = false;

    // ---- Multigrid Preconditioner ----
    struct MGLevel {
        int nx = 0, ny = 0;
        float invDx2 = 0.0f;
        std::vector<uint8_t> solid;
        std::vector<int> L, R, B, T;
        std::vector<float> diagInv;
        std::vector<float> x;
        std::vector<float> b;
        std::vector<float> Ax;
        std::vector<float> r;
    };

    bool useMGPrecond = false;
    int  mgMaxLevels = 6;
    int  mgPreSmooth = 2;
    int  mgPostSmooth = 2;
    float mgOmega = 0.8f;
    int  mgCoarseSmooth = 30;
    int  mgVcyclesPerApply = 1;
    bool mgBuiltOpenTopBC = false;
    int  mgBuiltNx = 0;
    int  mgBuiltNy = 0;
    bool mgBuiltValid = false;

    std::vector<MGLevel> mgLevels;
    bool mgDirty = true;

    void markMGDirty() { mgDirty = true; }
    void ensureMultigrid();

    void mgApplyA(int lev, const std::vector<float>& x, std::vector<float>& Ax) const;
    void mgSmoothJacobi(int lev, int iters);
    void mgComputeResidual(int lev);
    void mgRestrictResidual(int fineLev);
    void mgProlongateAndAdd(int coarseLev);
    void mgVCycle(int lev);
    void applyMGPrecond(const std::vector<float>& r, std::vector<float>& z);

    FrameStats stats;
};
