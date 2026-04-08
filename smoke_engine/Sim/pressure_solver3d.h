#pragma once

#include <cstdint>
#include <vector>

// Structured 3D Poisson pressure solve on a cell grid.
// Domain is defined by solidMask + fluidMask (both nx*ny*nz, values 0/1).
//
// Operator (fluid cells only):
//   A p = (diagW * p - sum_i(w_i * p_i)) * invDx2
// where each face weight w_i is the open fraction of the corresponding face.
// For binary voxel solids, these are 0/1. Air neighbors contribute only to the
// diagonal (Dirichlet p=0). Solid neighbors contribute nothing (Neumann).
//
// Only the +Y boundary can be treated as open-to-atmosphere via openTopBC.
class PressureSolver3D {
public:
    PressureSolver3D() = default;

    void configure(int nx, int ny, int nz, float dx,
                   bool openTopBC,
                   const std::vector<uint8_t>& solidMask,
                   const std::vector<uint8_t>& fluidMask,
                   bool removeMeanForGauge,
                   const std::vector<float>* faceOpenU = nullptr,   // (nx+1)*ny*nz
                   const std::vector<float>* faceOpenV = nullptr,   // nx*(ny+1)*nz
                   const std::vector<float>* faceOpenW = nullptr);  // nx*ny*(nz+1)

    // Solve A*p = rhs. p is warm-started from its current values.
    int solvePCG(std::vector<float>& p,
                 const std::vector<float>& rhs,
                 int maxIters,
                 float tolPredDiv,
                 float dtForPredDiv);

    void solveMG(std::vector<float>& p,
                 const std::vector<float>& rhs,
                 int maxVCycles,
                 float tolPredDiv,
                 float dtForPredDiv);

    int lastIterations() const { return m_lastIters; }

private:
    static inline int idx(int i, int j, int k, int nx, int ny) {
        return i + nx * (j + ny * k);
    }

    static inline std::size_t uIdx(int i, int j, int k, int nx, int ny) {
        return (std::size_t)i + (std::size_t)(nx + 1) * ((std::size_t)j + (std::size_t)ny * (std::size_t)k);
    }

    static inline std::size_t vIdx(int i, int j, int k, int nx, int ny) {
        return (std::size_t)i + (std::size_t)nx * ((std::size_t)j + (std::size_t)(ny + 1) * (std::size_t)k);
    }

    static inline std::size_t wIdx(int i, int j, int k, int nx, int ny) {
        return (std::size_t)i + (std::size_t)nx * ((std::size_t)j + (std::size_t)ny * (std::size_t)k);
    }

    void rebuildOperator();
    void ensurePCGBuffers();
    void applyA(const std::vector<float>& x, std::vector<float>& Ax) const;

    float dotFluid(const std::vector<float>& a, const std::vector<float>& b) const;
    float maxAbsFluid(const std::vector<float>& a) const;
    void removeMean(std::vector<float>& p) const;

    int m_nx = 0;
    int m_ny = 0;
    int m_nz = 0;
    float m_dx = 1.0f;
    float m_invDx2 = 1.0f;
    bool m_openTopBC = false;
    bool m_removeMean = false;
    bool m_dirty = true;

    std::vector<uint8_t> m_solid;
    std::vector<uint8_t> m_fluid;

    std::vector<float> m_faceOpenU;
    std::vector<float> m_faceOpenV;
    std::vector<float> m_faceOpenW;

    std::vector<int> m_xm, m_xp, m_ym, m_yp, m_zm, m_zp;
    std::vector<float> m_wXm, m_wXp, m_wYm, m_wYp, m_wZm, m_wZp;
    std::vector<float> m_diagW;
    std::vector<float> m_diagInv;

    std::vector<float> m_r, m_z, m_d, m_q, m_Ap;

    int m_lastIters = 0;

    struct MGLevel {
        int nx = 0;
        int ny = 0;
        int nz = 0;
        float invDx2 = 0.0f;

        std::vector<uint8_t> solid;
        std::vector<uint8_t> fluid;

        std::vector<float> faceOpenU;
        std::vector<float> faceOpenV;
        std::vector<float> faceOpenW;

        std::vector<int> xm, xp, ym, yp, zm, zp;
        std::vector<float> wXm, wXp, wYm, wYp, wZm, wZp;
        std::vector<float> diagW;
        std::vector<float> diagInv;

        std::vector<float> x, b, Ax, r;
    };

    // Multigrid settings.
    int mgMaxLevels = 10;
    int mgPreSmooth = 2;
    int mgPostSmooth = 2;
    int mgCoarseIters = 80;
    bool mgUseSOR = true;
    float mgSORomega = 1.4f;

    bool mgDirty = true;
    bool mgBuiltValid = false;
    int mgBuiltNx = 0;
    int mgBuiltNy = 0;
    int mgBuiltNz = 0;
    bool mgBuiltOpenTop = false;

    std::vector<MGLevel> mgLevels;

    void ensureMultigrid();
    void buildLevelStencil(MGLevel& level) const;
    void mgApplyA(int lev, const std::vector<float>& x, std::vector<float>& Ax) const;
    void mgSmoothRBGS(int lev, int iters);
    void mgComputeResidual(int lev);
    void mgRestrictResidual(int fineLev);
    void mgProlongateAndAdd(int coarseLev);
    void mgVCycle(int lev);
};
