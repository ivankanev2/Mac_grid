#pragma once

#include <algorithm>
#include <array>
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

    void setMGControls(int coarseIters, float relativeTol) {
        mgCoarseIters = std::max(1, coarseIters);
        mgRelativeTol = std::max(0.0f, relativeTol);
    }

    void setMGSmoother(bool useSOR, float omega) {
        mgUseSOR = useSOR;
        mgSORomega = omega;
    }

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
    void detectCompactDenseBox();
    void ensurePCGBuffers();
    void packFluidField(const std::vector<float>& full, std::vector<float>& compact) const;
    void unpackFluidField(const std::vector<float>& compact, std::vector<float>& full) const;
    void applyACompact(const std::vector<float>& x, std::vector<float>& Ax) const;

    float dotCompact(const std::vector<float>& a, const std::vector<float>& b) const;
    float maxAbsCompact(const std::vector<float>& a) const;
    void removeMeanCompact(std::vector<float>& p) const;

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
    std::vector<int> m_fluidCells;
    std::vector<int> m_gridToCompact;

    struct CompactCellStencil {
        int xm = -1;
        int xp = -1;
        int ym = -1;
        int yp = -1;
        int zm = -1;
        int zp = -1;

        float wXm = 0.0f;
        float wXp = 0.0f;
        float wYm = 0.0f;
        float wYp = 0.0f;
        float wZm = 0.0f;
        float wZp = 0.0f;

        float diagW = 0.0f;
        float diagInv = 0.0f;
    };

    bool m_compactDenseBoxValid = false;
    int m_compactBoxNx = 0;
    int m_compactBoxNy = 0;
    int m_compactBoxNz = 0;
    bool m_compactBoxOpenTopDirichlet = false;

    std::vector<CompactCellStencil> m_compactStencils;
    std::vector<float> m_compactDiagInv;
    std::vector<float> m_pCompact;
    std::vector<float> m_rhsCompact;
    std::vector<float> m_r, m_z, m_d, m_q, m_Ap;

    int m_lastIters = 0;

    struct MGCellStencil {
        int cell = 0;
        int xm = 0;
        int xp = 0;
        int ym = 0;
        int yp = 0;
        int zm = 0;
        int zp = 0;

        float wXm = 0.0f;
        float wXp = 0.0f;
        float wYm = 0.0f;
        float wYp = 0.0f;
        float wZm = 0.0f;
        float wZp = 0.0f;

        float diagW = 0.0f;
        float invDiagW = 0.0f;
        float diagInv = 0.0f;
    };

    struct MGLevel {
        int nx = 0;
        int ny = 0;
        int nz = 0;
        float invDx2 = 0.0f;

        bool denseBoxValid = false;
        int boxI0 = 0;
        int boxI1 = 0;
        int boxJ0 = 0;
        int boxJ1 = 0;
        int boxK0 = 0;
        int boxK1 = 0;
        bool boxOpenTopDirichlet = false;

        std::vector<uint8_t> solid;
        std::vector<uint8_t> fluid;

        std::vector<float> faceOpenU;
        std::vector<float> faceOpenV;
        std::vector<float> faceOpenW;

        // Full-grid <-> compact-fluid mappings for this MG level.
        std::vector<int> fluidCells;
        std::vector<int> gridToCompact;

        // Compact fluid-only stencils in color-major order:
        // [0, redStencilCount) are red cells, [redStencilCount, end) are black.
        std::vector<MGCellStencil> stencils;
        std::size_t redStencilCount = 0;

        // Compact fluid-only vectors, indexed by compact fluid id.
        std::vector<float> x;
        std::vector<float> b;
        std::vector<float> r;

        // Optional dense direct solve for the coarsest grid.
        bool directSolveValid = false;
        bool directSolveAnchorsGauge = false;
        std::vector<int> directSolveCells;
        std::vector<int> directSolveCompactIndex;
        std::vector<float> directSolveCholesky;
        std::vector<float> directSolveScratch0;
        std::vector<float> directSolveScratch1;
    };

    struct MGRestrictEntry {
        int coarseCell = 0;
        uint8_t count = 0;
        std::array<int, 27> ids{};
        std::array<float, 27> weights{};
    };

    struct MGProlongEntry {
        int fineCell = 0;
        uint8_t count = 0;
        std::array<int, 8> ids{};
        std::array<float, 8> weights{};
    };

    struct MGTransfer {
        bool denseBoxStructured = false;
        std::vector<MGRestrictEntry> restrictEntries;
        std::vector<MGProlongEntry> prolongEntries;
    };

    // Multigrid settings.
    int mgMaxLevels = 10;
    int mgPreSmooth = 2;
    int mgPostSmooth = 2;
    int mgCoarseIters = 80;
    bool mgUseSOR = true;
    float mgSORomega = 1.4f;
    float mgRelativeTol = 1.0e-4f;

    bool mgDirty = true;
    bool mgBuiltValid = false;
    int mgBuiltNx = 0;
    int mgBuiltNy = 0;
    int mgBuiltNz = 0;
    bool mgBuiltOpenTop = false;

    std::vector<MGLevel> mgLevels;
    std::vector<MGTransfer> mgTransfers;

    void detectDenseBox(MGLevel& level) const;
    void ensureMultigrid();
    void buildLevelStencil(MGLevel& level) const;
    void buildTransfer(int fineLev, MGTransfer& transfer) const;
    void buildDirectCoarseSolve(MGLevel& level) const;
    bool mgDirectSolve(int lev);
    void mgSmoothRBGS(int lev, int iters);
    void mgComputeResidual(int lev);
    void mgRestrictResidual(int fineLev);
    void mgProlongateAndAdd(int coarseLev);
    void mgVCycle(int lev);
};
