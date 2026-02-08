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
           bool removeMeanForGauge,
           const std::vector<float>* faceOpenU = nullptr,   // (nx+1)*ny
           const std::vector<float>* faceOpenV = nullptr);  // nx*(ny+1)

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

    // Face openness (0..1). If not provided, derived from solidMask as binary 0/1.
    // U faces: (nx+1)*ny , V faces: nx*(ny+1)
    std::vector<float> m_faceOpenU;
    std::vector<float> m_faceOpenV;


    // Operator cache (multiface-ready)
    std::vector<int>   m_L, m_R, m_B, m_T;     // fluid neighbor indices (-1 = none)
    std::vector<float> m_wL, m_wR, m_wB, m_wT; // weights for fluid neighbors only (0..1)
    std::vector<float> m_diagW;                // diagonal weight sum (includes air faces)
    std::vector<float> m_diagInv;              // 1 / (diagW * invDx2)

    // PCG buffers
    std::vector<float> m_r, m_z, m_d, m_q, m_Ap;

    int m_lastIters = 0;

    bool  m_removeMean = true;

    struct MGLevel {
        int nx = 0, ny = 0;
        float invDx2 = 0.0f;

        std::vector<uint8_t> solid;   // 1 = solid
        std::vector<uint8_t> fluid;   // 1 = in pressure domain

        std::vector<int>   L, R, B, T;      // neighbor indices (-1 = none)
        std::vector<float> wL, wR, wB, wT;  // weights to fluid neighbors (0..1)
        std::vector<float> diagW;           // diagonal weight sum (includes air faces)
        std::vector<float> diagInv;         // 1 / (diagW * invDx2)

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