#pragma once
#include <vector>
#include <cstdint>

// Structured Poisson pressure solve on a 2D cell grid.
// Domain is defined by solidMask + fluidMask (both nx*ny, values 0/1).
//
// Operator (fluid cells only):
//   A p = (count * p - sum(fluid-neighbor p)) * invDx2
// where:
//   - fluid neighbor contributes to sum and count
//   - air neighbor (non-solid, non-fluid) is Dirichlet p=0: contributes to count only
//   - solid neighbor is Neumann: contributes nothing
// Optional openTopBC: the neighbor above top row is treated as air (Dirichlet).
class PressureSolver {
public:
    PressureSolver() = default;

    void configure(int nx, int ny, float dx,
               bool openTopBC,
               const std::vector<uint8_t>& solidMask,
               const std::vector<uint8_t>& fluidMask,
               bool removeMeanForGauge);

    // Solve A*p = rhs. p is warm-started (uses current p values).
    // Returns iterations used (0 if early-out).
    int solvePCG(std::vector<float>& p,
                 const std::vector<float>& rhs,
                 int maxIters,
                 float tolPredDiv,
                 float dtForPredDiv);

    int lastIterations() const { return m_lastIters; }

    // Full MG solve (like smoke's solvePressureMG)
    void solveMG(std::vector<float>& p,
                 const std::vector<float>& rhs,
                 int maxVCycles,
                 float tolPredDiv,
                 float dt);

private:
    static inline int idx(int i, int j, int nx) { return i + nx * j; }

    void rebuildOperator();
    void ensurePCGBuffers();
    void applyA(const std::vector<float>& x, std::vector<float>& Ax) const;

    float dotFluid(const std::vector<float>& a, const std::vector<float>& b) const;
    float maxAbsFluid(const std::vector<float>& a) const;
    void removeMean(std::vector<float>& p) const;

    int   m_nx = 0, m_ny = 0;
    float m_dx = 1.0f;
    float m_invDx2 = 1.0f;
    bool  m_openTopBC = false;

    bool m_dirty = true;

    std::vector<uint8_t> m_solid;
    std::vector<uint8_t> m_fluid;

    // Per-face openness (0..1). Default 1.0 = fully open.
    std::vector<float> faceOpenU; // size (nx+1) * ny  — u faces (vertical faces)
    std::vector<float> faceOpenV; // size nx * (ny+1)  — v faces (horizontal faces)

    // Operator cache
    std::vector<int>     m_L, m_R, m_B, m_T;
    std::vector<uint8_t> m_count;
    std::vector<float>   m_diagInv;

    // PCG buffers
    std::vector<float> m_r, m_z, m_d, m_q, m_Ap;

    int m_lastIters = 0;

    bool  m_removeMean = true;

    struct MGLevel {
        int nx = 0, ny = 0;
        float invDx2 = 0.0f;

        std::vector<uint8_t> solid;   // 1 = solid
        std::vector<uint8_t> fluid;   // 1 = in pressure domain

        std::vector<int> L, R, B, T;  // neighbor indices (-1 = none)
        std::vector<uint8_t> diagCount;
        std::vector<float> diagInv;

        std::vector<float> x, b, Ax, r;
    };

      static inline int mgIdx(int i, int j, int nx) { return i + nx * j; }

    // MG settings (match your smoke defaults-ish)
    int   mgMaxLevels = 10;
    int   mgPreSmooth = 2;
    int   mgPostSmooth = 2;
    int   mgCoarseIters = 200;
    int   mgVcyclesPerApply = 1;
    bool  mgUseSOR = true;
    float mgSORomega = 1.4f;

    bool mgDirty = true;
    bool mgBuiltValid = false;
    bool mgBuiltOpenTop = false;
    int  mgBuiltNx = 0, mgBuiltNy = 0;

    std::vector<MGLevel> mgLevels;

    void ensureMultigrid();
    void mgApplyA(int lev, const std::vector<float>& x, std::vector<float>& Ax) const;
    void mgSmoothRBGS(int lev, int iters);
    void mgComputeResidual(int lev);
    void mgRestrictResidual(int fineLev);
    void mgProlongateAndAdd(int coarseLev);
    void mgVCycle(int lev);
    void applyMGPrecond(const std::vector<float>& r, std::vector<float>& z);
};