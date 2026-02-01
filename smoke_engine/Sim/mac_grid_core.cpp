#include "mac_grid_core.h"
#include <cmath>
#include <chrono>
#include <limits>

namespace {
    static float dotVec(const std::vector<float>& a, const std::vector<float>& b) {
        double s = 0.0;
        for (size_t i = 0; i < a.size(); ++i)
            s += (double)a[i] * (double)b[i];
        return (float)s;
    }
}



static inline int mgIdx(int i, int j, int nx) { return i + nx * j; }

inline bool sampleOutsideIsZero(float xq, float yq, bool openTopBC, int nx, int ny, float dx) {
    if (xq < 0.0f || xq > nx*dx) return false;      // CLOSED sides: clamp later
    if (yq < 0.0f) return false;                    // CLOSED bottom: clamp later
    if (yq > ny*dx) return openTopBC;               // top is open only if enabled
    return false;
}

MACGridCore::MACGridCore(int NX, int NY, float DX, float DT)
    : nx(NX), ny(NY), dx(DX), dt(DT)
{
    resetCore();
}

void MACGridCore::resetCore() {
    const size_t Nu = (size_t)(nx + 1) * (size_t)ny;
    const size_t Nv = (size_t)nx * (size_t)(ny + 1);
    const size_t Nc = (size_t)nx * (size_t)ny;

    u.assign(Nu, 0.0f);
    v.assign(Nv, 0.0f);
    p.assign(Nc, 0.0f);
    div.assign(Nc, 0.0f);
    rhs.assign(Nc, 0.0f);
    u0 = u;
    v0 = v;

    solid.assign(Nc, 0);

    markPressureMatrixDirty();
}

float MACGridCore::maxAbsDiv() const {
    float m = 0.0f;
    for (float d : div) {
        if (!std::isfinite(d)) return std::numeric_limits<float>::infinity();
        m = std::max(m, std::abs(d));
    }
    return m;
}

float MACGridCore::maxFaceSpeed() const {
    float m = 0.0f;
    for (float val : u) {
        if (!std::isfinite(val)) return std::numeric_limits<float>::infinity();
        m = std::max(m, std::fabs(val));
    }
    for (float val : v) {
        if (!std::isfinite(val)) return std::numeric_limits<float>::infinity();
        m = std::max(m, std::fabs(val));
    }
    return m;
}

static bool findFirstNonFinite(const char* name, const std::vector<float>& a, int& outIdx, float& outVal) {
    for (int i = 0; i < (int)a.size(); ++i) {
        if (!std::isfinite(a[i])) { outIdx = i; outVal = a[i]; return true; }
    }
    return false;
}

void MACGridCore::worldToCell(float x, float y, int &i, int &j) const {
    float fx = x / dx - 0.5f;
    float fy = y / dx - 0.5f;
    i = (int)std::floor(fx);
    j = (int)std::floor(fy);
    i = (int)clampf((float)i, 0.0f, (float)(nx - 1));
    j = (int)clampf((float)j, 0.0f, (float)(ny - 1));
}

float MACGridCore::sampleCellCentered(const std::vector<float>& f, float x, float y) const {
    float fx = x / dx - 0.5f;
    float fy = y / dx - 0.5f;

    int i0 = (int)std::floor(fx);
    int j0 = (int)std::floor(fy);
    float tx = fx - i0;
    float ty = fy - j0;

    i0 = (int)clampf((float)i0, 0.0f, (float)(nx - 1));
    j0 = (int)clampf((float)j0, 0.0f, (float)(ny - 1));
    int i1 = std::min(i0 + 1, nx - 1);
    int j1 = std::min(j0 + 1, ny - 1);

    float a = f[idxP(i0, j0)];
    float b = f[idxP(i1, j0)];
    float c = f[idxP(i0, j1)];
    float d = f[idxP(i1, j1)];

    float ab = a * (1 - tx) + b * tx;
    float cd = c * (1 - tx) + d * tx;
    return ab * (1 - ty) + cd * ty;
}

float MACGridCore::sampleU(const std::vector<float>& fu, float x, float y) const {
    float fx = x / dx;
    float fy = y / dx - 0.5f;

    int i0 = (int)std::floor(fx);
    int j0 = (int)std::floor(fy);
    float tx = fx - i0;
    float ty = fy - j0;

    i0 = (int)clampf((float)i0, 0.0f, (float)(nx));
    j0 = (int)clampf((float)j0, 0.0f, (float)(ny - 1));
    int i1 = std::min(i0 + 1, nx);
    int j1 = std::min(j0 + 1, ny - 1);

    float a = fu[idxU(i0, j0)];
    float b = fu[idxU(i1, j0)];
    float c = fu[idxU(i0, j1)];
    float d = fu[idxU(i1, j1)];

    float ab = a * (1 - tx) + b * tx;
    float cd = c * (1 - tx) + d * tx;
    return ab * (1 - ty) + cd * ty;
}

float MACGridCore::sampleV(const std::vector<float>& fv, float x, float y) const {
    float fx = x / dx - 0.5f;
    float fy = y / dx;

    int i0 = (int)std::floor(fx);
    int j0 = (int)std::floor(fy);
    float tx = fx - i0;
    float ty = fy - j0;

    i0 = (int)clampf((float)i0, 0.0f, (float)(nx - 1));
    j0 = (int)clampf((float)j0, 0.0f, (float)(ny));
    int i1 = std::min(i0 + 1, nx - 1);
    int j1 = std::min(j0 + 1, ny);

    float a = fv[idxV(i0, j0)];
    float b = fv[idxV(i1, j0)];
    float c = fv[idxV(i0, j1)];
    float d = fv[idxV(i1, j1)];

    float ab = a * (1 - tx) + b * tx;
    float cd = c * (1 - tx) + d * tx;
    return ab * (1 - ty) + cd * ty;
}

void MACGridCore::velAt(float x, float y,
                        const std::vector<float>& fu,
                        const std::vector<float>& fv,
                        float& outUx, float& outVy) const
{
    outUx = sampleU(fu, x, y);
    outVy = sampleV(fv, x, y);
}

void MACGridCore::advectVelocity() {
    u0 = u;
    v0 = v;

    for (int j = 0; j < ny; j++) {
        for (int i = 0; i <= nx; i++) {
            float x = i * dx;
            float y = (j + 0.5f) * dx;

            float ux, vy;
            velAt(x, y, u0, v0, ux, vy);

            float x0 = clampf(x - dt * ux, 0.0f, nx * dx);
            float y0 = clampf(y - dt * vy, 0.0f, ny * dx);

            u[idxU(i, j)] = sampleU(u0, x0, y0);

            bool leftSolid  = (i - 1 >= 0) ? isSolid(i - 1, j) : true;
            bool rightSolid = (i < nx)     ? isSolid(i, j)     : true;
            if (leftSolid || rightSolid) u[idxU(i, j)] = 0.0f;
        }
    }

    for (int j = 0; j <= ny; j++) {
        for (int i = 0; i < nx; i++) {
            float x = (i + 0.5f) * dx;
            float y = j * dx;

            float ux, vy;
            velAt(x, y, u0, v0, ux, vy);

            float x0 = clampf(x - dt * ux, 0.0f, nx * dx);
            float y0 = clampf(y - dt * vy, 0.0f, ny * dx);

            v[idxV(i, j)] = sampleV(v0, x0, y0);

            bool botSolid = (j - 1 >= 0) ? isSolid(i, j - 1) : true;
            bool topBlocked;
            if (j == ny && openTopBC) {
                // open boundary: don't treat outside as solid
                topBlocked = false;
            } else {
                topBlocked = (j < ny) ? isSolid(i, j) : true;
            }

            if (botSolid || topBlocked) v[idxV(i, j)] = 0.0f;
        }
    }
}

void MACGridCore::advectScalarMacCormack(std::vector<float>& phi,
                                         std::vector<float>& phi0,
                                         float dissipation)
{
    phi0 = phi;

    std::vector<float> phiFwd(phi.size(), 0.0f);
    std::vector<float> phiBack(phi.size(), 0.0f);

    auto stencilMinMax = [&](const std::vector<float>& f, float x, float y, float& outMin, float& outMax) {
        float fx = x / dx - 0.5f;
        float fy = y / dx - 0.5f;
        int i0 = (int)std::floor(fx);
        int j0 = (int)std::floor(fy);
        i0 = (int)clampf((float)i0, 0.0f, (float)(nx - 1));
        j0 = (int)clampf((float)j0, 0.0f, (float)(ny - 1));
        int i1 = std::min(i0 + 1, nx - 1);
        int j1 = std::min(j0 + 1, ny - 1);

        outMin =  std::numeric_limits<float>::infinity();
        outMax = -std::numeric_limits<float>::infinity();

        auto consider = [&](int ii, int jj) {
            if (isSolid(ii, jj)) return;
            float v = f[idxP(ii, jj)];
            outMin = std::min(outMin, v);
            outMax = std::max(outMax, v);
        };

        consider(i0, j0); consider(i1, j0);
        consider(i0, j1); consider(i1, j1);

        if (!std::isfinite(outMin)) { outMin = 0.0f; outMax = 0.0f; }
    };

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int id = idxP(i,j);
            if (isSolid(i,j)) { phiFwd[id] = 0.0f; continue; }

            float x = (i + 0.5f) * dx;
            float y = (j + 0.5f) * dx;
            float ux, vy;
            velAt(x, y, u, v, ux, vy);

            // float x0 = clampf(x - dt * ux, 0.0f, nx * dx);
            // float y0 = clampf(y - dt * vy, 0.0f, ny * dx);


            float xq = x - dt * ux;
            float yq = y - dt * vy;

            if (sampleOutsideIsZero(xq, yq, openTopBC, nx, ny, dx)) {
                phiFwd[id] = 0.0f;
                continue;
            }

            xq = clampf(xq, 0.0f, nx * dx);
            yq = clampf(yq, 0.0f, ny * dx);


            int si, sj; worldToCell(xq, yq, si, sj);
            phiFwd[id] = (!isSolid(si, sj)) ? sampleCellCentered(phi0, xq, yq) : 0.0f;
        }
    }

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int id = idxP(i,j);
            if (isSolid(i,j)) { phiBack[id] = 0.0f; continue; }

            float x = (i + 0.5f) * dx;
            float y = (j + 0.5f) * dx;
            float ux, vy;
            velAt(x, y, u, v, ux, vy);

            // float x1 = clampf(x + dt * ux, 0.0f, nx * dx);
            // float y1 = clampf(y + dt * vy, 0.0f, ny * dx);

            float xq = x + dt * ux;
            float yq = y + dt * vy;

            if (sampleOutsideIsZero(xq, yq, openTopBC, nx, ny, dx)) {
                phiBack[id] = 0.0f;
                continue;
            }

            xq = clampf(xq, 0.0f, nx * dx);
            yq = clampf(yq, 0.0f, ny * dx);

            int si, sj; worldToCell(xq, yq, si, sj);
            phiBack[id] = (!isSolid(si, sj)) ? sampleCellCentered(phiFwd, xq, yq) : 0.0f;
        }
    }

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int id = idxP(i,j);
            if (isSolid(i,j)) { phi[id] = 0.0f; continue; }

            float x = (i + 0.5f) * dx;
            float y = (j + 0.5f) * dx;
            float ux, vy;
            velAt(x, y, u, v, ux, vy);

            float xq = x - dt * ux;
            float yq = y - dt * vy;

            if (sampleOutsideIsZero(xq, yq, openTopBC, nx, ny, dx)) {
                phi[id] = 0.0f;
                continue;
            }

            xq = clampf(xq, 0.0f, nx * dx);
            yq = clampf(yq, 0.0f, ny * dx);

            float corrected = phiFwd[id] + 0.5f * (phi0[id] - phiBack[id]);

            float mn, mx;
            stencilMinMax(phi0, xq, yq, mn, mx);
            corrected = clampf(corrected, mn, mx);

            phi[id] = dissipation * corrected;
        }
    }
}

void MACGridCore::advectScalarSemiLagrangian(std::vector<float>& phi,
                                             std::vector<float>& phi0,
                                             float dissipation)
{
    phi0 = phi;
    std::vector<float> tmp(phi.size(), 0.0f);
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int id = idxP(i,j);
            if (isSolid(i,j)) { tmp[id] = 0.0f; continue; }

            float x = (i + 0.5f) * dx;
            float y = (j + 0.5f) * dx;
            float ux, vy;
            velAt(x, y, u, v, ux, vy);

            float xq = x - dt * ux;
            float yq = y - dt * vy;

            if (sampleOutsideIsZero(xq, yq, openTopBC, nx, ny, dx)) {
                tmp[id] = 0.0f;
                continue;
            }

            xq = clampf(xq, 0.0f, nx * dx);
            yq = clampf(yq, 0.0f, ny * dx);

            int si, sj;
            worldToCell(xq, yq, si, sj);
            tmp[id] = (!isSolid(si, sj)) ? sampleCellCentered(phi0, xq, yq) : 0.0f;
            tmp[id] *= dissipation;
        }
    }
    phi.swap(tmp);
}

void MACGridCore::ensurePCGBuffers() {
    const int N = nx * ny;
    if ((int)pcg_r.size()  != N) pcg_r.resize(N);
    if ((int)pcg_z.size()  != N) pcg_z.resize(N);
    if ((int)pcg_d.size()  != N) pcg_d.resize(N);
    if ((int)pcg_q.size()  != N) pcg_q.resize(N);
    if ((int)pcg_Ap.size() != N) pcg_Ap.resize(N);
}

void MACGridCore::ensurePressureMatrix() {
    if (!pressureMatrixDirty) return;
    pressureMatrixDirty = false;

    const int N = nx * ny;
    lapL.assign(N, -1);
    lapR.assign(N, -1);
    lapB.assign(N, -1);
    lapT.assign(N, -1);
    lapDiagInv.assign(N, 0.0f);

    invDx2_cache = 1.0f / (dx * dx);

    if ((int)p.size() != N) p.assign(N, 0.0f);

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int id = idxP(i, j);
            if (isSolid(i, j)) {
                lapL[id] = lapR[id] = lapB[id] = lapT[id] = -1;
                lapDiagInv[id] = 0.0f;
                continue;
            }
            

            int count = 0;
            if (i > 0 && !isSolid(i - 1, j))     { lapL[id] = idxP(i - 1, j); count++; }
            if (i + 1 < nx && !isSolid(i + 1, j)) { lapR[id] = idxP(i + 1, j); count++; }
            if (j > 0 && !isSolid(i, j - 1))     { lapB[id] = idxP(i, j - 1); count++; }
            if (j + 1 < ny && !isSolid(i, j + 1)) { lapT[id] = idxP(i, j + 1); count++; }

            if (openTopBC && j == ny - 1) {
            count++; // Dirichlet neighbor contributes to diagonal
        }

            const float diag = (float)count * invDx2_cache;
            lapDiagInv[id] = (diag > 0.0f) ? (1.0f / diag) : 0.0f;
        }
    }
}

void MACGridCore::ensureMultigrid() {
    if (!mgDirty) return;
    mgDirty = false;

    mgLevels.clear();
    mgLevels.reserve(mgMaxLevels);

    {
        MGLevel L0;
        L0.nx = nx; L0.ny = ny;
        L0.invDx2 = 1.0f / (dx * dx);

        const int N = nx * ny;
        L0.solid.assign(N, 0);

        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
                L0.solid[idxP(i,j)] = isSolid(i,j) ? 1 : 0;

        L0.L.assign(N, -1); L0.R.assign(N, -1); L0.B.assign(N, -1); L0.T.assign(N, -1);
        L0.diagInv.assign(N, 0.0f);

        L0.x.assign(N, 0.0f);
        L0.b.assign(N, 0.0f);
        L0.Ax.assign(N, 0.0f);
        L0.r.assign(N, 0.0f);

        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int id = idxP(i,j);
                if (L0.solid[id]) continue;

                int count = 0;
                if (i > 0     && !L0.solid[idxP(i-1,j)]) { L0.L[id] = idxP(i-1,j); count++; }
                if (i+1 < nx  && !L0.solid[idxP(i+1,j)]) { L0.R[id] = idxP(i+1,j); count++; }
                if (j > 0     && !L0.solid[idxP(i,j-1)]) { L0.B[id] = idxP(i,j-1); count++; }
                if (j+1 < ny  && !L0.solid[idxP(i,j+1)]) { L0.T[id] = idxP(i,j+1); count++; }

                if (openTopBC && j == ny - 1) {
                count++;
            }

                float diag = (float)count * L0.invDx2;
                L0.diagInv[id] = (diag > 0.0f) ? 1.0f / diag : 0.0f;
            }
        }

        mgLevels.push_back(std::move(L0));
    }

    while ((int)mgLevels.size() < mgMaxLevels) {
        const MGLevel& F = mgLevels.back();
        if (F.nx <= 4 || F.ny <= 4) break;

        int cnx = F.nx / 2;
        int cny = F.ny / 2;
        if (cnx < 2 || cny < 2) break;

        MGLevel C;
        C.nx = cnx; C.ny = cny;
        C.invDx2 = F.invDx2 * 0.25f;

        const int CN = cnx * cny;
        C.solid.assign(CN, 0);
        C.L.assign(CN, -1); C.R.assign(CN, -1); C.B.assign(CN, -1); C.T.assign(CN, -1);
        C.diagInv.assign(CN, 0.0f);

        C.x.assign(CN, 0.0f);
        C.b.assign(CN, 0.0f);
        C.Ax.assign(CN, 0.0f);
        C.r.assign(CN, 0.0f);

        for (int J = 0; J < cny; ++J) {
            for (int I = 0; I < cnx; ++I) {
                int fi = 2*I;
                int fj = 2*J;

                bool allSolid = true;
                for (int dj = 0; dj < 2; ++dj) {
                    for (int di = 0; di < 2; ++di) {
                        int ii = fi + di;
                        int jj = fj + dj;
                        if (ii < F.nx && jj < F.ny) {
                            if (!F.solid[mgIdx(ii,jj,F.nx)]) allSolid = false;
                        } else {
                            // out of bounds shouldn't happen with /2 sizes, but treat as solid if it does
                        }
                    }
                }

                C.solid[mgIdx(I,J,cnx)] = allSolid ? 1 : 0;
            }
}

        for (int J = 0; J < cny; ++J) {
            for (int I = 0; I < cnx; ++I) {
                int id = mgIdx(I,J,cnx);
                if (C.solid[id]) continue;

                int count = 0;
                if (I > 0      && !C.solid[mgIdx(I-1,J,cnx)]) { C.L[id] = mgIdx(I-1,J,cnx); count++; }
                if (I+1 < cnx  && !C.solid[mgIdx(I+1,J,cnx)]) { C.R[id] = mgIdx(I+1,J,cnx); count++; }
                if (J > 0      && !C.solid[mgIdx(I,J-1,cnx)]) { C.B[id] = mgIdx(I,J-1,cnx); count++; }
                if (J+1 < cny  && !C.solid[mgIdx(I,J+1,cnx)]) { C.T[id] = mgIdx(I,J+1,cnx); count++; }

                if (openTopBC && J == cny - 1) {
                count++;
            }

                float diag = (float)count * C.invDx2;
                C.diagInv[id] = (diag > 0.0f) ? 1.0f / diag : 0.0f;
            }
        }

        mgLevels.push_back(std::move(C));
    }
}

void MACGridCore::mgApplyA(int lev, const std::vector<float>& x, std::vector<float>& Ax) const {
    const MGLevel& L = mgLevels[lev];
    const int N = L.nx * L.ny;
    if ((int)Ax.size() != N) Ax.resize(N);

    for (int id = 0; id < N; ++id) {
        if (L.solid[id]) { Ax[id] = 0.0f; continue; }

        float sum = 0.0f;
        int count = 0;

        int n = L.L[id]; if (n >= 0) { sum += x[n]; count++; }
        n = L.R[id];     if (n >= 0) { sum += x[n]; count++; }
        n = L.B[id];     if (n >= 0) { sum += x[n]; count++; }
        n = L.T[id];     if (n >= 0) { sum += x[n]; count++; }

        if (openTopBC) {
            const int j = id / L.nx;
            if (j == L.ny - 1) {
                // Dirichlet p_out = 0 contributes to diagonal only (sum += 0)
                count++;
            }
        }

        Ax[id] = (count * x[id] - sum) * L.invDx2;
    }
}

void MACGridCore::mgSmoothJacobi(int lev, int iters) {
    MGLevel& L = mgLevels[lev];
    const int N = L.nx * L.ny;

    for (int it = 0; it < iters; ++it) {
        mgApplyA(lev, L.x, L.Ax);
        for (int id = 0; id < N; ++id) {
            if (L.solid[id]) continue;
            float r = L.b[id] - L.Ax[id];
            L.x[id] += mgOmega * (L.diagInv[id] * r);
        }
    }
}

void MACGridCore::mgComputeResidual(int lev) {
    MGLevel& L = mgLevels[lev];
    mgApplyA(lev, L.x, L.Ax);
    const int N = L.nx * L.ny;
    for (int id = 0; id < N; ++id) {
        if (L.solid[id]) { L.r[id] = 0.0f; continue; }
        L.r[id] = L.b[id] - L.Ax[id];
    }
}

void MACGridCore::mgRestrictResidual(int fineLev) {
    MGLevel& F = mgLevels[fineLev];
    MGLevel& C = mgLevels[fineLev + 1];

    std::fill(C.b.begin(), C.b.end(), 0.0f);
    std::fill(C.x.begin(), C.x.end(), 0.0f);

    for (int J = 0; J < C.ny; ++J) {
        for (int I = 0; I < C.nx; ++I) {
            int cid = mgIdx(I,J,C.nx);
            if (C.solid[cid]) { C.b[cid] = 0.0f; continue; }

            float sum = 0.0f;
            // float w = 0.0f;

            int fi = 2*I;
            int fj = 2*J;
            for (int dj = 0; dj < 2; ++dj) {
                for (int di = 0; di < 2; ++di) {
                    int ii = fi + di;
                    int jj = fj + dj;
                    if (ii < F.nx && jj < F.ny) {
                        int fid = mgIdx(ii,jj,F.nx);
                        if (!F.solid[fid]) {
                            sum += F.r[fid];
                            // w += 1.0f;
                        }
                    }
                }
            }
            // C.b[cid] = (w > 0.0f) ? (sum / w) : 0.0f;
            C.b[cid] = 0.25f * sum;
        }
    }
}

void MACGridCore::mgProlongateAndAdd(int coarseLev) {
    MGLevel& C = mgLevels[coarseLev];
    MGLevel& F = mgLevels[coarseLev - 1];

    for (int J = 0; J < C.ny; ++J) {
        for (int I = 0; I < C.nx; ++I) {
            int cid = mgIdx(I,J,C.nx);
            float e = C.x[cid];

            int fi = 2*I;
            int fj = 2*J;
            for (int dj = 0; dj < 2; ++dj) {
                for (int di = 0; di < 2; ++di) {
                    int ii = fi + di;
                    int jj = fj + dj;
                    if (ii < F.nx && jj < F.ny) {
                        int fid = mgIdx(ii,jj,F.nx);
                        if (!F.solid[fid]) F.x[fid] += e;
                    }
                }
            }
        }
    }
}

void MACGridCore::mgVCycle(int lev) {
    if (lev == (int)mgLevels.size() - 1) {
        mgSmoothJacobi(lev, mgCoarseSmooth);
        return;
    }

    mgSmoothJacobi(lev, mgPreSmooth);
    mgComputeResidual(lev);
    mgRestrictResidual(lev);

    mgVCycle(lev + 1);

    mgProlongateAndAdd(lev + 1);
    mgSmoothJacobi(lev, mgPostSmooth);
}

void MACGridCore::applyMGPrecond(const std::vector<float>& r, std::vector<float>& z) {
    ensureMultigrid();
    if (mgLevels.empty()) { z = r; return; }

    MGLevel& F = mgLevels[0];
    const int N = F.nx * F.ny;
    if ((int)z.size() != N) z.resize(N);

    F.b = r;
    std::fill(F.x.begin(), F.x.end(), 0.0f);

    for (int k = 0; k < mgVcyclesPerApply; ++k)
        mgVCycle(0);

    z = F.x;
}

void MACGridCore::removePressureMean() {
    double sum = 0.0;
    int cnt = 0;

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            if (isSolid(i,j)) continue;
            sum += (double)p[idxP(i,j)];
            cnt++;
        }
    }

    if (cnt == 0) return;

    float mean = (float)(sum / (double)cnt);
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            if (isSolid(i,j)) continue;
            p[idxP(i,j)] -= mean;
        }
    }
}

void MACGridCore::computeDivergence() {
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            if (isSolid(i, j)) { div[idxP(i, j)] = 0.0f; continue; }

            float uL = u[idxU(i, j)];
            float uR = u[idxU(i + 1, j)];
            float vB = v[idxV(i, j)];
            float vT = v[idxV(i, j + 1)];

            if (i > 0    && isSolid(i - 1, j)) uL = 0.0f;
            if (i < nx-1 && isSolid(i + 1, j)) uR = 0.0f;
            if (j > 0    && isSolid(i, j - 1)) vB = 0.0f;
            if (j < ny-1 && isSolid(i, j + 1)) vT = 0.0f;

            div[idxP(i, j)] = (uR - uL + vT - vB) / dx;
        }
    }
}

void MACGridCore::applyLaplacian(const std::vector<float>& x, std::vector<float>& Ax) const {
    const int N = nx * ny;
    if ((int)Ax.size() != N) Ax.resize(N);

    const float invDx2 = invDx2_cache;

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int id = idxP(i, j);
            if (isSolid(i, j)) { Ax[id] = 0.0f; continue; }

            

            float sum = 0.0f;
            int count = 0;

            int n = lapL[id]; if (n >= 0) { sum += x[n]; count++; }
            n = lapR[id];     if (n >= 0) { sum += x[n]; count++; }
            n = lapB[id];     if (n >= 0) { sum += x[n]; count++; }
            n = lapT[id];     if (n >= 0) { sum += x[n]; count++; }

            if (openTopBC && j == ny - 1) {
            // Dirichlet p_out = 0 at top boundary:
            // counts as a neighbor in the stencil, but sum += 0 so nothing to add
            count++;
        }

            Ax[id] = (count * x[id] - sum) * invDx2;
        }
    }
}

// hours wasted on this function : 5
void MACGridCore::solvePressurePCG(int maxIters, float tol) {
    std::printf("[PCG] tol=%g (openTopBC=%d)\n", tol, (int)openTopBC);

    ensurePressureMatrix();
    ensurePCGBuffers();

    const int N = nx * ny;

    auto dotFluid = [&](const std::vector<float>& a, const std::vector<float>& b) -> float {
        double s = 0.0;
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                if (isSolid(i,j)) continue;
                int id = idxP(i,j);
                s += (double)a[id] * (double)b[id];
            }
        }
        return (float)s;
    };

        auto maxAbsResidualFluid = [&]() -> float {
        float m = 0.0f;
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                if (isSolid(i,j)) continue;
                float a = std::fabs(pcg_r[idxP(i,j)]);
                if (a > m) m = a;
            }
        }
        return m;
    };

    auto maxAbsBFluid = [&]() -> float {
        float m = 0.0f;
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                if (isSolid(i,j)) continue;
                float a = std::fabs(rhs[idxP(i,j)]);
                if (a > m) m = a;
            }
        }
        return m;
    };
    

    // r = b - A p   (warm start using current p)
    applyLaplacian(p, pcg_Ap);
    for (int k = 0; k < N; ++k) pcg_r[k] = rhs[k] - pcg_Ap[k];

    // Early out: if RHS is basically zero, pressure solve is pointless
    const float bInf = maxAbsBFluid();
    if (bInf * dt <= tol) {
        stats.pressureIters = 0;
        if (!openTopBC) removePressureMean();
        return;
    }

    

    // Precondition
    const bool useMG = useMGPrecond && !openTopBC;
    if (useMG) {
        applyMGPrecond(pcg_r, pcg_z);
    } else {
        for (int k = 0; k < N; ++k) pcg_z[k] = pcg_r[k] * lapDiagInv[k];
    }
    pcg_d = pcg_z;

    float deltaNew = dotFluid(pcg_r, pcg_z);
    if (deltaNew <= 1e-30f || !std::isfinite(deltaNew)) {
        stats.pressureIters = 0;
        if (!openTopBC) removePressureMean();
        return;
    }

    const float r0Norm2 = dotFluid(pcg_r, pcg_r);
    const float r0Norm2Safe = std::max(r0Norm2, 1e-30f);

    // Optional relative tolerance (keeps accuracy; just avoids wasted late iters)
    const float relTol = 1e-3f;
    const float relTol2 = relTol * relTol;

    int it_used = 0;

    for (int it = 0; it < maxIters; ++it) {
        it_used = it + 1;

        applyLaplacian(pcg_d, pcg_q);

        const float dq = dotFluid(pcg_d, pcg_q);
        if (!std::isfinite(dq) || std::fabs(dq) < 1e-30f) break;

        const float alpha = deltaNew / dq;

        for (int k = 0; k < N; ++k) {
            p[k]     += alpha * pcg_d[k];
            pcg_r[k] -= alpha * pcg_q[k];
        }

        const float rInf = maxAbsResidualFluid();
        if ((it & 7) == 0) std::printf("[PCG] it=%d predDiv=%g\n", it, rInf * dt);

        // Primary stopping: divergence-style (matches your logs)
        if (rInf * dt <= tol) break;

        // Secondary stopping: relative L2 residual
        const float rNorm2 = dotFluid(pcg_r, pcg_r);
        if (rNorm2 <= relTol2 * r0Norm2Safe) break;

        // Precondition again
        if (useMG) {
            applyMGPrecond(pcg_r, pcg_z);
        } else {
            for (int k = 0; k < N; ++k) pcg_z[k] = pcg_r[k] * lapDiagInv[k];
        }

        const float deltaOld = deltaNew;
        deltaNew = dotFluid(pcg_r, pcg_z);
        if (!std::isfinite(deltaNew) || deltaNew <= 1e-30f) break;

        const float beta = deltaNew / (deltaOld + 1e-30f);
        for (int k = 0; k < N; ++k)
            pcg_d[k] = pcg_z[k] + beta * pcg_d[k];
    }

    stats.pressureIters = it_used;

    // For closed domains, remove gauge freedom
    if (!openTopBC) removePressureMean();
}

// the new big dick function for Multrigrid solver
void MACGridCore::solvePressureMG(int maxVCycles, float tol) {
    std::printf("[MG ] tol=%g (openTopBC=%d)\n", tol, (int)openTopBC);

    // If solids / topology changed you should already have set mgDirty,
    // but forcing a rebuild here is safe (just a bit slower).
    mgDirty = true;
    ensureMultigrid();

    if (mgLevels.empty()) {
        // Fallback if MG not built for some reason
        solvePressurePCG(80, tol);
        return;
    }

    MGLevel& F = mgLevels[0];
    const int N = F.nx * F.ny;

    auto maxAbsResidualFluid = [&]() -> float {
        float m = 0.0f;
        for (int id = 0; id < N; ++id) {
            if (F.solid[id]) continue;
            float a = std::fabs(F.r[id]);
            if (!std::isfinite(a)) return std::numeric_limits<float>::infinity();
            m = std::max(m, a);
        }
        return m;
    };

    auto maxAbsBFluid = [&]() -> float {
        float m = 0.0f;
        for (int id = 0; id < N; ++id) {
            if (F.solid[id]) continue;
            float a = std::fabs(F.b[id]);
            if (!std::isfinite(a)) return std::numeric_limits<float>::infinity();
            m = std::max(m, a);
        }
        return m;
    };

    auto removeMeanFromFineX = [&]() {
        if (openTopBC) return; // Dirichlet already fixes the gauge
        double sum = 0.0;
        int cnt = 0;
        for (int id = 0; id < N; ++id) {
            if (F.solid[id]) continue;
            sum += (double)F.x[id];
            cnt++;
        }
        if (cnt == 0) return;
        float mean = (float)(sum / (double)cnt);
        for (int id = 0; id < N; ++id) {
            if (F.solid[id]) continue;
            F.x[id] -= mean;
        }
    };

    auto allFinite = [](const std::vector<float>& a) {
        for (float v : a)
            if (!std::isfinite(v)) return false;
        return true;
    };

    if (!allFinite(p)) {
        std::printf("[MG ] WARNING: p has NaNs/Infs. Resetting warm start.\n");
        std::fill(p.begin(), p.end(), 0.0f);
    }
    if (!allFinite(rhs)) {
        std::printf("[MG ] WARNING: rhs has NaNs/Infs. Zeroing rhs.\n");
        std::fill(rhs.begin(), rhs.end(), 0.0f);
    }

    if ((int)F.x.size() != N) F.x.assign(N, 0.0f);
    if ((int)F.b.size() != N) F.b.assign(N, 0.0f);

    // Warm start + copy RHS
    F.x = p;
    F.b = rhs;

    // Early out if RHS is tiny (matches PCG logic)
    const float bInf = maxAbsBFluid();
    if (bInf * dt <= tol) {
        stats.pressureIters = 0;
        if (!openTopBC) removePressureMean();
        return;
    }

    // Initial residual
    mgComputeResidual(0);
    float rInf = maxAbsResidualFluid();
    std::printf("[MG ] predDiv_initial=%g\n", rInf * dt);

    if (!std::isfinite(rInf) || rInf * dt <= tol) {
        stats.pressureIters = 0;
        p = F.x;
        if (!openTopBC) removePressureMean();
        return;
    }

    int used = 0;
    for (int k = 0; k < maxVCycles; ++k) {
        used = k + 1;

        float prev = rInf;
        // Full V-cycle starting from level 0
        mgVCycle(0);
        mgComputeResidual(0);
        rInf = maxAbsResidualFluid();

        if (rInf > prev * 1.2f) {
            std::printf("[MG ] WARNING: residual increased (%g -> %g). Resetting x and switching to smoothing-only this frame.\n", prev, rInf);
            std::fill(F.x.begin(), F.x.end(), 0.0f);
            // do a few fine smooths to keep the sim alive
            mgSmoothJacobi(0, 20);
            mgComputeResidual(0);
            break;
        }


        // For closed domains, keep the solution in the zero-mean subspace
        removeMeanFromFineX();

        // Recompute residual on the fine level
        mgComputeResidual(0);
        rInf = maxAbsResidualFluid();

        if ((k & 3) == 0) {
            std::printf("[MG ] v=%d predDiv=%g\n", k, rInf * dt);
        }

        if (!std::isfinite(rInf) || rInf * dt <= tol)
            break;
    }

    stats.pressureIters = used;

    // Copy solution back to main pressure field
    p = F.x;

    // Gauge fix for closed domains (keeps continuity w/ PCG path)
    if (!openTopBC) removePressureMean();
}

void MACGridCore::debugCheckMGvsPCGOperator() {
    ensurePressureMatrix();
    ensureMultigrid();

    if (mgLevels.empty()) {
        std::printf("[DEBUG] No MG levels.\n");
        return;
    }

    MGLevel& F = mgLevels[0];
    const int N = nx * ny;

    std::vector<float> x(N), Ax_mg(N), Ax_pcg(N);
    // Fill x with something nontrivial but finite
    for (int k = 0; k < N; ++k) x[k] = (float)((k % 17) - 8); // small-ish pattern

    // MG operator on level 0
    mgApplyA(0, x, Ax_mg);

    // PCG operator
    applyLaplacian(x, Ax_pcg);

    float maxDiff = 0.0f;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int id = idxP(i, j);
            if (isSolid(i, j)) continue;

            float d = std::fabs(Ax_mg[id] - Ax_pcg[id]);
            if (d > maxDiff) maxDiff = d;
        }
    }

    std::printf("[DEBUG] max |A_mg - A_pcg| = %g\n", maxDiff);
}

void MACGridCore::project() {

    // the big fucking wall of checks to see what is wrong with everything
    if (!(dt > 0.0f) || !std::isfinite(dt)) {
        std::printf("[project] BAD dt=%g, skipping projection\n", dt);
        return;
    }

    int badIdx; float badVal;
    if (findFirstNonFinite("u", u, badIdx, badVal)) std::printf("[NONFINITE] u[%d]=%g\n", badIdx, badVal);
    if (findFirstNonFinite("v", v, badIdx, badVal)) std::printf("[NONFINITE] v[%d]=%g\n", badIdx, badVal);
    if (findFirstNonFinite("p", p, badIdx, badVal)) std::printf("[NONFINITE] p[%d]=%g\n", badIdx, badVal);

    computeDivergence();

    if (findFirstNonFinite("div", div, badIdx, badVal)) std::printf("[NONFINITE] div[%d]=%g\n", badIdx, badVal);
    if (!std::isfinite(maxAbsDiv()) || !std::isfinite(maxFaceSpeed())) {
    std::printf("[project] NON-FINITE STATE. Resetting u/v/p.\n");
    std::fill(u.begin(), u.end(), 0.0f);
    std::fill(v.begin(), v.end(), 0.0f);
    std::fill(p.begin(), p.end(), 0.0f);
    computeDivergence();
    return;
}

    // before project (after computeDivergence inside project already fills stats)
    // but add explicit print here (place near top of step or after computeDivergence)
    std::printf("[BEFORE] maxDiv=%g maxFace=%g\n", maxAbsDiv(), maxFaceSpeed());

    stats.dt = dt;
    stats.maxDivBefore = maxAbsDiv();
    stats.maxFaceSpeedBefore = maxFaceSpeed();

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int id = idxP(i, j);
            if (isSolid(i,j)) rhs[id] = 0.0f;
            else              rhs[id] = -div[id] / dt; // i swear to got if the sign change here breaks another thing
        }
    }

    if (!openTopBC) {
    double sum = 0.0;
    int cnt = 0;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int id = idxP(i,j);
            if (isSolid(i,j)) continue;
            sum += rhs[id];
            cnt++;
        }
    }
    if (cnt > 0) {
        float mean = (float)(sum / (double)cnt);
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int id = idxP(i,j);
                if (isSolid(i,j)) continue;
                rhs[id] -= mean;
            }
        }
    }
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    float pcgDivTol = openTopBC ? 5e-4f : 1e-4f;
//    solvePressurePCG(80, pcgDivTol); 

    float divTol = openTopBC ? 5e-4f : 1e-4f;
    solvePressureMG(20, divTol);   // start with ~10–30 V-cycles

    auto t1 = std::chrono::high_resolution_clock::now();
    stats.pressureMs = std::chrono::duration<float, std::milli>(t1 - t0).count();

    for (int j = 0; j < ny; ++j) {
        for (int i = 1; i < nx; ++i) {
            if (isSolid(i - 1, j) || isSolid(i, j)) { u[idxU(i, j)] = 0.0f; continue; }
            float gradp = (p[idxP(i, j)] - p[idxP(i - 1, j)]) / dx;
            u[idxU(i, j)] -= dt * gradp;
        }
    }

    for (int j = 1; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            if (isSolid(i, j - 1) || isSolid(i, j)) { v[idxV(i, j)] = 0.0f; continue; }
            float gradp = (p[idxP(i, j)] - p[idxP(i, j - 1)]) / dx;
            v[idxV(i, j)] -= dt * gradp;
        }
    }

    if (openTopBC) {
    const int jFace = ny;        // v-face index at the top boundary
    const int jCell = ny - 1;    // top row of cells

    for (int i = 0; i < nx; ++i) {
        // If the top cell is solid, don't allow flow through
        if (isSolid(i, jCell)) { 
            v[idxV(i, jFace)] = 0.0f; 
            continue; 
        }

        const float p_inside  = p[idxP(i, jCell)];
        const float p_outside = 0.0f;            // Dirichlet pressure outside

        // gradp = (p_out - p_in)/dx
        const float gradp = (p_outside - p_inside) / dx;
        v[idxV(i, jFace)] -= dt * gradp;
    }
}

    // Clamp face speeds to avoid extreme velocities, they dont occur anymore as of now
    const float MAX_FACE_SPEED = 200.0f;
    for (float& val : u) val = clampf(val, -MAX_FACE_SPEED, MAX_FACE_SPEED);
    for (float& val : v) val = clampf(val, -MAX_FACE_SPEED, MAX_FACE_SPEED);

    computeDivergence();
    std::printf("[AFTER ] maxDiv=%g maxFace=%g (iters=%d)\n",
                maxAbsDiv(), maxFaceSpeed(), stats.pressureIters);
    stats.maxDivAfter = maxAbsDiv();
    stats.maxFaceSpeedAfter = maxFaceSpeed();
}
