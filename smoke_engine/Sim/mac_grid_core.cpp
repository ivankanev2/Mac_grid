#include "mac_grid_core.h"
#include <cmath>
#include <chrono>
#include <limits>



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

void MACGridCore::setSolidCell(int i, int j, bool s) {
    int id = idxP(i,j);
    uint8_t ns = s ? 1 : 0;
    if (solid[id] == ns) return;
    solid[id] = ns;

    // If a cell becomes solid, it cannot be fluid.
    // If it becomes non-solid again, default it back to fluid.
    if (s) fluid[id] = 0;
    else   fluid[id] = 1;

    rebuildFaceOpennessBinaryFromSolids();

    pressureMatrixDirty = true;
}

void MACGridCore::rebuildFaceOpennessBinaryFromSolids()
{
    faceOpenU.assign((size_t)(nx + 1) * (size_t)ny, 1.0f);
    faceOpenV.assign((size_t)nx * (size_t)(ny + 1), 1.0f);

    auto uIdx = [&](int i, int j) { return (size_t)j * (size_t)(nx + 1) + (size_t)i; }; // i:0..nx, j:0..ny-1
    auto vIdx = [&](int i, int j) { return (size_t)j * (size_t)nx + (size_t)i; };       // i:0..nx-1, j:0..ny

    // U faces: blocked if either adjacent cell is solid
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i <= nx; ++i) {

            // domain walls: always closed on left/right
            if (i == 0 || i == nx) {
                faceOpenU[uIdx(i,j)] = 0.0f;
                continue;
            }

            bool blocked = false;
            blocked |= isSolid(i - 1, j);
            blocked |= isSolid(i,     j);

            faceOpenU[uIdx(i,j)] = blocked ? 0.0f : 1.0f;
        }
    }

    // V faces: blocked if either adjacent cell is solid
    for (int j = 0; j <= ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            bool blocked = false;

            // cell below the face
            if (j - 1 >= 0) blocked |= isSolid(i, j - 1);
            else            blocked |= true; // bottom boundary is closed

            // cell above the face
            if (j < ny) {
                blocked |= isSolid(i, j);
            } else {
                // j == ny => top boundary face
                if (openTopBC) {
                    // open top: outside is NOT solid; only block if the top cell itself is solid
                    blocked |= isSolid(i, ny - 1);
                } else {
                    // closed top: treat outside as solid wall
                    blocked |= true;
                }
            }

            faceOpenV[vIdx(i,j)] = blocked ? 0.0f : 1.0f;
        }
    }
}

void MACGridCore::syncSolidsToFluidAndFaces()
{
    // solids must never be fluid
    for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) {
        int id = idxP(i,j);
        fluid[id] = solid[id] ? 0 : 1;
    }

    rebuildFaceOpennessBinaryFromSolids();
    invalidatePressureMatrix(); // marks pressure matrix dirty
}

void MACGridCore::setOpenTopBC(bool enabled) {
    if (openTopBC == enabled) return;
    openTopBC = enabled;

    rebuildFaceOpennessBinaryFromSolids(); // <-- IMPORTANT: apply immediately

    markPressureMatrixDirty();
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
    fluid.assign(Nc, 1); // default: everything fluid unless solid

    for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i)
    {
        if (solid[idxP(i,j)]) fluid[idxP(i,j)] = 0;
    }


    rebuildFaceOpennessBinaryFromSolids();

    markPressureMatrixDirty();

    ensurePCGBuffers();   // allocate once, no per-solve resize later
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

float MACGridCore::divLInfFluid() const {
    float m = 0.0f;
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            if (!isFluidCell(i,j)) continue;
            m = std::max(m, std::fabs(div[idxP(i,j)]));
        }
    return m;
}

float MACGridCore::divL2Fluid() const {
    double s = 0.0;
    int cnt = 0;
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            if (!isFluidCell(i,j)) continue;
            float d = div[idxP(i,j)];
            s += (double)d * (double)d;
            cnt++;
        }
    return (cnt > 0) ? (float)std::sqrt(s / (double)cnt) : 0.0f;
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

            // multiface: just mask by openness (includes walls + solids)
            const float w = faceOpenU[(size_t)j * (size_t)(nx + 1) + (size_t)i];
            u[idxU(i, j)] *= w;
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

            const float w = faceOpenV[(size_t)j * (size_t)nx + (size_t)i];
            v[idxV(i, j)] *= w;
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
    lapCount.assign(N, 0);

    invDx2_cache = 1.0f / (dx * dx);

    if ((int)p.size() != N) p.assign(N, 0.0f);

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int id = idxP(i, j);
            if (!isFluidCell(i, j)) {
                // Not in pressure domain (air or solid) -> mark as inactive
                lapL[id] = lapR[id] = lapB[id] = lapT[id] = -1;
                lapDiagInv[id] = 0.0f;
                continue;
            }

            int count = 0;
            // Build stencil for FLUID unknowns:
            // - fluid neighbor   => connect + count++
            // - air neighbor     => Dirichlet p=0 => count++ (no connection)
            // - solid neighbor   => Neumann => nothing
            auto addNbr = [&](int ni, int nj, int& slot) {
                const int nid = idxP(ni, nj);
                if (isFluidCell(ni, nj)) { slot = nid; count++; }
                else if (!isSolid(ni, nj)) { count++; } // air Dirichlet
            };

            if (i > 0)      addNbr(i - 1, j, lapL[id]);
            if (i + 1 < nx) addNbr(i + 1, j, lapR[id]);
            if (j > 0)      addNbr(i, j - 1, lapB[id]);
            if (j + 1 < ny) addNbr(i, j + 1, lapT[id]);

            if (openTopBC && j == ny - 1) {
            count++; // Dirichlet neighbor contributes to diagonal
        }

            lapCount[id] = (uint8_t)count;
            const float diag = (float)count * invDx2_cache;
            lapDiagInv[id] = (diag > 0.0f) ? (1.0f / diag) : 0.0f;
        }
    }
}



void MACGridCore::removePressureMean() {
    double sum = 0.0;
    int cnt = 0;

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            if (!isFluidCell(i,j)) continue;
            sum += (double)p[idxP(i,j)];
            cnt++;
        }
    }

    if (cnt == 0) return;

    float mean = (float)(sum / (double)cnt);
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            if (!isFluidCell(i,j)) continue;
            p[idxP(i,j)] -= mean;
        }
    }
}

void MACGridCore::computeDivergence()
{
    auto uIdx = [&](int i, int j) { return (size_t)j * (size_t)(nx + 1) + (size_t)i; };
    auto vIdx = [&](int i, int j) { return (size_t)j * (size_t)nx + (size_t)i; };

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            if (isSolid(i,j)) { div[idxP(i,j)] = 0.0f; continue; }

            float uL = u[idxU(i,   j)];
            float uR = u[idxU(i+1, j)];
            float vB = v[idxV(i, j)];
            float vT = v[idxV(i, j+1)];

            float wUL = faceOpenU[uIdx(i,   j)];
            float wUR = faceOpenU[uIdx(i+1, j)];
            float wVB = faceOpenV[vIdx(i, j)];
            float wVT = faceOpenV[vIdx(i, j+1)];

            // IMPORTANT: top boundary flux only exists if openTopBC
            if (j + 1 == ny && !openTopBC) wVT = 0.0f;

            div[idxP(i,j)] = ((wUR*uR - wUL*uL) + (wVT*vT - wVB*vB)) / dx;
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
            if (!isFluidCell(i, j)) { Ax[id] = 0.0f; continue; }

            

            float sum = 0.0f;
            int count = (int)lapCount[id];

            int n = lapL[id]; if (n >= 0) { sum += x[n]; }
            n = lapR[id];     if (n >= 0) { sum += x[n]; }
            n = lapB[id];     if (n >= 0) { sum += x[n]; }
            n = lapT[id];     if (n >= 0) { sum += x[n]; }

            if (openTopBC && j == ny - 1) {
            // Dirichlet p_out = 0 at top boundary:
            // counts as a neighbor in the stencil, but sum += 0 so nothing to add
            count++;
        }

            Ax[id] = (count * x[id] - sum) * invDx2;
        }
    }
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
    // stats.maxDivBefore = maxAbsDiv();
    stats.maxDivBefore = divLInfFluid();
    stats.maxFaceSpeedBefore = maxFaceSpeed();

    stats.openTopBC = openTopBC ? 1 : 0;
    stats.pressureStopReason = STOP_NONE;
    stats.mgResidualIncrease = 0;
    stats.opCheckPass = 0;
    stats.opDiffMax = 0.0f;
    stats.rhsMaxPredDiv = 0.0f;
    stats.predDivInitial = 0.0f;
    stats.predDivFinal = 0.0f;

    int bad = 0;
    for (int id = 0; id < nx*ny; ++id) {
        if (solid[id] && fluid[id]) bad++;
    }
    if (bad) printf("[BUG] %d cells are solid AND fluid!\n", bad);

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int id = idxP(i, j);
            if (!isFluidCell(i,j)) rhs[id] = 0.0f;
            else              rhs[id] = -div[id] / dt; // i swear to got if the sign change here breaks another thing
        }
    }

    // rhsMaxPredDiv = max|rhs| * dt  (predicts max divergence after solve)
    float bInf = 0.0f;
    for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) {
        if (!isFluidCell(i,j)) continue;
        bInf = std::max(bInf, std::fabs(rhs[idxP(i,j)]));
    }
    stats.rhsMaxPredDiv = bInf * dt;

    if (!openTopBC) {
    double sum = 0.0;
    int cnt = 0;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int id = idxP(i,j);
            if (!isFluidCell(i,j)) continue;
            sum += rhs[id];
            cnt++;
        }
    }
    if (cnt > 0) {
        float mean = (float)(sum / (double)cnt);
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int id = idxP(i,j);
                if (!isFluidCell(i,j)) continue;
                rhs[id] -= mean;
            }
        }
    }
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    float pcgDivTol = openTopBC ? 5e-4f : 1e-4f;
//    solvePressurePCG(80, pcgDivTol); 

    float divTol = openTopBC ? 5e-4f : 1e-4f;

    ps().configure(
        nx, ny, dx,
        openTopBC,
        solid,
        fluid,
        /*removeMeanForGauge=*/!openTopBC,
        &faceOpenU,
        &faceOpenV
    );

    stats.pressureSolver = SOLVER_MG;
    ps().solveMG(p, rhs, 20, divTol, dt);


    auto t1 = std::chrono::high_resolution_clock::now();
    stats.pressureMs = std::chrono::duration<float, std::milli>(t1 - t0).count();

    if (openTopBC) {
        const int jCell = ny - 1; // top row of cells
        const int jFace = ny;     // top boundary v-face row

        float pTopMax = 0.0f;
        float vTopMax = 0.0f;
        int   pTopMaxI = -1;
        int   vTopMaxI = -1;

        for (int i = 0; i < nx; ++i) {
            if (!isSolid(i, jCell)) {
                float ap = std::fabs(p[idxP(i, jCell)]);
                if (ap > pTopMax) { pTopMax = ap; pTopMaxI = i; }
            }

            // v at the top boundary face exists for all i in [0, nx-1]
            float av = std::fabs(v[idxV(i, jFace)]);
            if (av > vTopMax) { vTopMax = av; vTopMaxI = i; }
        }

        static int dbgFrame = 0;
        if ((dbgFrame++ & 31) == 0) {
            std::printf("[TOP ] |p|_max(topCells)=%g at i=%d   |v|_max(topFace)=%g at i=%d\n",
                        pTopMax, pTopMaxI, vTopMax, vTopMaxI);
        }
    }

    auto uW = [&](int i, int j) { return faceOpenU[(size_t)j * (size_t)(nx + 1) + (size_t)i]; };

    for (int j = 0; j < ny; ++j) {
        for (int i = 1; i < nx; ++i) {

            float w = uW(i, j);

            // if either side isn’t in pressure domain, treat this face as closed for projection
            if (!isFluidCell(i - 1, j) || !isFluidCell(i, j)) w = 0.0f;

            float gradp = (p[idxP(i, j)] - p[idxP(i - 1, j)]) / dx;
            u[idxU(i, j)] -= dt * w * gradp;

            // always enforce openness (kills leftovers cleanly)
            u[idxU(i, j)] *= w;
        }
    }

    auto vW = [&](int i, int j) { return faceOpenV[(size_t)j * (size_t)nx + (size_t)i]; };

    for (int j = 1; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {

            float w = vW(i, j);
            
            if (!isFluidCell(i, j - 1) || !isFluidCell(i, j)) w = 0.0f;

            float gradp = (p[idxP(i, j)] - p[idxP(i, j - 1)]) / dx;
            v[idxV(i, j)] -= dt * w * gradp;

            v[idxV(i, j)] *= w;
        }
    }

    if (openTopBC) {
    const int jFace = ny;        // v-face index at the top boundary
    const int jCell = ny - 1;    // top row of cells

    auto vW = [&](int i, int j) { return faceOpenV[(size_t)j * (size_t)nx + (size_t)i]; };

    for (int i = 0; i < nx; ++i) {
        float w = vW(i, jFace);
        if (isSolid(i, jCell)) w = 0.0f;

        const float p_inside  = p[idxP(i, jCell)];
        const float p_outside = 0.0f;

        const float gradp = (p_outside - p_inside) / dx;
        v[idxV(i, jFace)] -= dt * w * gradp;

        v[idxV(i, jFace)] *= w;
    }
    if (openTopBC) {
        float vTopMax2 = 0.0f;
        for (int i = 0; i < nx; ++i)
            vTopMax2 = std::max(vTopMax2, std::fabs(v[idxV(i, ny)]));
        static int dbg2 = 0;
        if ((dbg2++ & 31) == 0)
            std::printf("[TOP2] |v|_max(topFace AFTER bc)=%g\n", vTopMax2);
    }
}

    // Clamp face speeds to avoid extreme velocities, they dont occur anymore as of now
    // const float MAX_FACE_SPEED = 200.0f;
    // float m = maxFaceSpeed();
    // if (m > MAX_FACE_SPEED) {
    //     float s = MAX_FACE_SPEED / m;
    //     for (float& a : u) a *= s;
    //     for (float& a : v) a *= s;
    // }

    computeDivergence();

    const float divInf = divLInfFluid();
    const float divL2  = divL2Fluid();
    std::printf("[AFTER ] divInf=%g divL2=%g maxFace=%g (iters=%d)\n",
                divInf, divL2, maxFaceSpeed(), stats.pressureIters);

    // std::printf("[AFTER ] maxDiv=%g maxFace=%g (iters=%d)\n",
    //             maxAbsDiv(), maxFaceSpeed(), stats.pressureIters);
    // stats.maxDivAfter = maxAbsDiv();
    stats.maxDivAfter  = divLInfFluid();
    stats.maxFaceSpeedAfter = maxFaceSpeed();
}

void MACGridCore::setFluidMask(const std::vector<uint8_t>& mask)
{
    // Expect same layout as p (nx*ny)
    if ((int)mask.size() != nx * ny)
        return; // keep it simple for now; we can add an assert/log later

    fluid = mask;

    // Solids must never be fluid
    for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i)
    {
        if (solid[idxP(i,j)]) fluid[idxP(i,j)] = 0;
    }
    markPressureMatrixDirty();
}

void MACGridCore::setFluidMaskAllNonSolid()
{
    fluid.assign(nx * ny, 1);
    for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i)
    {
        if (solid[idxP(i,j)]) fluid[idxP(i,j)] = 0;
    }
    markPressureMatrixDirty();
}
