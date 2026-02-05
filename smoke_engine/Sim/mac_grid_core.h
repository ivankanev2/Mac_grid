#pragma once
#include <vector>
#include <cstdint>
#include <algorithm>

#include "pressure_solver.h"

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
    std::vector<uint8_t> fluid;

    enum PressureSolverKind : int {
        SOLVER_PCG = 0,
        SOLVER_MG  = 1
    };

    enum PressureStopReason : int {
        STOP_NONE = 0,
        STOP_ABS_TOL,
        STOP_REL_TOL,
        STOP_MAX_ITERS,
        STOP_NONFINITE,
        STOP_RESIDUAL_INCREASE
    };

    // Pressure solver.
    // Default: each grid owns its own instance (pressureSolver).
    // Optional: smoke + water can share one solver instance by calling
    // setSharedPressureSolver(&shared) on both sims.
    PressureSolver pressureSolver;

    // If non-null, all pressure solves will use this shared solver instance instead.
    void setSharedPressureSolver(PressureSolver* s) { sharedPressureSolver = s; }
    PressureSolver& ps() { return sharedPressureSolver ? *sharedPressureSolver : pressureSolver; }
    const PressureSolver& ps() const { return sharedPressureSolver ? *sharedPressureSolver : pressureSolver; }



    struct FrameStats {
        float dt = 0.0f;
        float maxDivBefore = 0.0f;
        float maxDivAfter  = 0.0f;
        float maxFaceSpeedBefore = 0.0f;
        float maxFaceSpeedAfter  = 0.0f;
        int   pressureIters = 0;
        float pressureMs    = 0.0f;

        int   openTopBC = 0;                 // 0/1
        int   pressureSolver = SOLVER_MG;    // PCG or MG

        float rhsMaxPredDiv  = 0.0f;         // bInf * dt
        float predDivInitial = 0.0f;         // rInf0 * dt
        float predDivFinal   = 0.0f;         // rInfEnd * dt

        int   pressureStopReason = STOP_NONE;

        int   opCheckPass = 0;               // 0/1
        float opDiffMax   = 0.0f;            // max |A_mg - A_pcg|

        int   mgResidualIncrease = 0;        // 0/1

    };

    MACGridCore(int NX, int NY, float DX, float DT);
    void resetCore();



    void setOpenTopBC(bool enabled); // testing stuff
    bool getOpenTop() const { return openTopBC; }


    

    float maxAbsDiv() const;
    float maxFaceSpeed() const;
    void setDt(float newDt) { dt = newDt; }

    const std::vector<uint8_t>& solidMask() const { return solid; }

    inline int idxP(int i,int j) const { return i + nx*j; }
    inline int idxU(int i,int j) const { return i + (nx+1)*j; }
    inline int idxV(int i,int j) const { return i + nx*j; }

    inline bool isSolid(int i,int j) const { return solid[idxP(i,j)] != 0; }

    bool isFluidCell(int i, int j) const { return fluid[idxP(i,j)] != 0; }
    const std::vector<uint8_t>& fluidMask() const { return fluid; }

    // Sets the fluid mask from an external mask (same size as p). Solids are always forced to non-fluid.
    void setFluidMask(const std::vector<uint8_t>& mask);

    // Convenience: set all non-solid cells to fluid (useful for smoke default behavior).
    void setFluidMaskAllNonSolid();

    inline void setFluidCell(int i, int j, bool f) {
    fluid[idxP(i,j)] = f ? 1 : 0;
    }

    
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
    bool debugCheckMGvsPCGOperator(float eps = 1e-4f, float* outMaxDiff = nullptr);

    float divLInfFluid() const;
    float divL2Fluid() const;

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
    std::vector<uint8_t> lapCount; // diagonal stencil count (includes Dirichlet neighbors)
    std::vector<float> lapDiagInv;

    std::vector<float> pcg_r, pcg_z, pcg_d, pcg_q, pcg_Ap;

    void markPressureMatrixDirty() { pressureMatrixDirty = true; }
    void ensurePressureMatrix();
    void ensurePCGBuffers();
    void removePressureMean();

    bool openTopBC = false;

    PressureSolver* sharedPressureSolver = nullptr;

    

    // ---- Multigrid Preconditioner ----
    struct MGLevel {
        int nx = 0, ny = 0;
        float invDx2 = 0.0f;
        std::vector<uint8_t> solid;
        std::vector<uint8_t> fluid; // 1 = fluid cell, 0 = not-fluid (e.g. air); solids will be forced to 0
        std::vector<int> L, R, B, T;
        std::vector<uint8_t> diagCount; // diagonal stencil count (includes Dirichlet neighbors)
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
    // float mgOmega = 1.4f; // DO NOT TOUCH THIS PLEASE
    float mgJacobiOmega = 0.67f;   // only used by Jacobi (if you keep it)
    float mgSORomega    = 1.4f;    // used by RBGS as SOR ω (default safe)
    bool  mgUseSOR       = true;  // default OFF: RBGS uses plain GS (ω=1)
    int  mgCoarseSmooth = 120;
    int  mgVcyclesPerApply = 1;
    bool mgBuiltOpenTopBC = false;
    int  mgBuiltNx = 0;
    int  mgBuiltNy = 0;
    bool mgBuiltValid = false;

    std::vector<MGLevel> mgLevels;
    bool mgDirty = true;
    bool mgInitialized = false;

    int mgCoarseIters = 80;

    void markMGDirty() { mgDirty = true; }
    void ensureMultigrid();

    void mgApplyA(int lev, const std::vector<float>& x, std::vector<float>& Ax) const;

    // the smoothers 
    void mgSmoothJacobi(int lev, int iters);
    void mgSmoothRBGS(int lev, int iters);

    void mgComputeResidual(int lev);
    void mgRestrictResidual(int fineLev);
    void mgProlongateAndAdd(int coarseLev);
    void mgVCycle(int lev);
    void applyMGPrecond(const std::vector<float>& r, std::vector<float>& z);

    FrameStats stats;
};
