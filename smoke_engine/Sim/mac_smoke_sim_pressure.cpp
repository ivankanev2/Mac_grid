#include "mac_smoke_sim.h"
#include <cmath>
#include <chrono>

// file-local helper
namespace {
    static float dotVec(const std::vector<float>& a, const std::vector<float>& b) {
        double s = 0.0;
        for (size_t i = 0; i < a.size(); ++i)
            s += (double)a[i] * (double)b[i];
        return (float)s;
    }
}

void MAC2D::ensurePCGBuffers() {
    const int N = nx * ny;
    if ((int)pcg_r.size() != N) pcg_r.resize(N);
    if ((int)pcg_z.size() != N) pcg_z.resize(N);
    if ((int)pcg_d.size() != N) pcg_d.resize(N);
    if ((int)pcg_q.size() != N) pcg_q.resize(N);
    if ((int)pcg_Ap.size() != N) pcg_Ap.resize(N);
}

void MAC2D::ensurePressureMatrix() {
    if (!pressureMatrixDirty) return;
    pressureMatrixDirty = false;

    const int N = nx * ny;
    lapL.assign(N, -1);
    lapR.assign(N, -1);
    lapB.assign(N, -1);
    lapT.assign(N, -1);
    lapDiagInv.assign(N, 0.0f);

    invDx2_cache = 1.0f / (dx * dx);

    // ensure p vector exists and has proper size for warm-starting
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
            if (i > 0 && !isSolid(i - 1, j)) { lapL[id] = idxP(i - 1, j); count++; }
            if (i + 1 < nx && !isSolid(i + 1, j)) { lapR[id] = idxP(i + 1, j); count++; }
            if (j > 0 && !isSolid(i, j - 1)) { lapB[id] = idxP(i, j - 1); count++; }
            if (j + 1 < ny && !isSolid(i, j + 1)) { lapT[id] = idxP(i, j + 1); count++; }

            const float diag = (float)count * invDx2_cache;
            lapDiagInv[id] = (diag > 0.0f) ? (1.0f / diag) : 0.0f;
        }
    }

    // If topology changed and you *want* to discard previous warm-start, clear p once:
    // std::fill(p.begin(), p.end(), 0.0f); // uncomment only if you prefer to always reset
}

// to make the pressure not go crazy over time
void MAC2D::removePressureMean()
{
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

void MAC2D::computeDivergence() {
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

void MAC2D::computeVorticity(std::vector<float>& outOmega) const {
    outOmega.assign(nx*ny, 0.0f);
    for (int j=0;j<ny;++j) for (int i=0;i<nx;++i) {
        if (isSolid(i,j)) { outOmega[idxP(i,j)] = 0.0f; continue; }
        // central differences
        int im1 = std::max(i-1,0), ip1 = std::min(i+1,nx-1);
        int jm1 = std::max(j-1,0), jp1 = std::min(j+1,ny-1);

        // dv/dx (centered)
        float vL = 0.5f*(v[idxV(i,j)] + v[idxV(i,j+1)]);
        float vR = 0.5f*(v[idxV(ip1,j)] + v[idxV(ip1,j+1)]);
        float dv_dx = (vR - vL) / dx;

        // du/dy (centered)
        float uB = 0.5f*(u[idxU(i,j)] + u[idxU(i+1,j)]);
        float uT = 0.5f*(u[idxU(i,jp1)] + u[idxU(i+1,jp1)]);
        float du_dy = (uT - uB) / dx;

        outOmega[idxP(i,j)] = dv_dx - du_dy;
    }
}

void MAC2D::applyLaplacian(const std::vector<float>& x, std::vector<float>& Ax) const {
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

            Ax[id] = (count * x[id] - sum) * invDx2;
        }
    }
}



void MAC2D::solvePressurePCG(int maxIters, float tol) {
    ensurePressureMatrix();
    ensurePCGBuffers();

    const int N = nx * ny;

    // Warm start: keep previous p as initial guess (do NOT zero every frame)
    // std::fill(p.begin(), p.end(), 0.0f);

    applyLaplacian(p, pcg_Ap);
    for (int k = 0; k < N; ++k) {
        pcg_r[k] = rhs[k] - pcg_Ap[k];
    }

    const float bNorm2 = dotVec(rhs, rhs);
    if (bNorm2 < 1e-30f) return;

    for (int k = 0; k < N; ++k) {
        pcg_z[k] = pcg_r[k] * lapDiagInv[k];
        pcg_d[k] = pcg_z[k];
    }

    float deltaNew = dotVec(pcg_r, pcg_z);
    float delta0   = deltaNew;
    if (deltaNew < 1e-30f) return;

    const float tol2 = tol * tol;

    int it_used = 0;
    for (int it = 0; it < maxIters; ++it) {
        it_used = it + 1;
        applyLaplacian(pcg_d, pcg_q);

        float dq = dotVec(pcg_d, pcg_q);
        if (std::fabs(dq) < 1e-30f) break;

        float alpha = deltaNew / dq;

        for (int k = 0; k < N; ++k) {
            p[k]     += alpha * pcg_d[k];
            pcg_r[k] -= alpha * pcg_q[k];
        }

        float rNorm2 = dotVec(pcg_r, pcg_r);
        if (rNorm2 <= tol2 * bNorm2) break;

        for (int k = 0; k < N; ++k) {
            pcg_z[k] = pcg_r[k] * lapDiagInv[k];
        }

        float deltaOld = deltaNew;
        deltaNew = dotVec(pcg_r, pcg_z);

        if (deltaNew <= tol2 * delta0) break;

        float beta = deltaNew / (deltaOld + 1e-30f);

        for (int k = 0; k < N; ++k) {
            pcg_d[k] = pcg_z[k] + beta * pcg_d[k];
        }
    }
    stats.pressureIters = it_used;
    removePressureMean();
}

void MAC2D::project() {
    // 1) divergence of current velocity field
    computeDivergence();
    stats.dt = dt;
    stats.maxDivBefore = maxAbsDiv();
    stats.maxFaceSpeedBefore = maxFaceSpeed();

    // 2) rhs for Poisson: Laplace(p) = div/dt  (solids -> 0)
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int id = idxP(i, j);
            rhs[id] = isSolid(i, j) ? 0.0f : (-div[id] / dt);
        }
    }

    // Make RHS zero-mean over fluid cells (required for Neumann Poisson)
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

    // timing the pressure solve
    auto t0 = std::chrono::high_resolution_clock::now();
    solvePressurePCG(80, 1e-4f);
    auto t1 = std::chrono::high_resolution_clock::now();
    stats.pressureMs = std::chrono::duration<float, std::milli>(t1 - t0).count();


    // 4) subtract pressure gradient from faces
    // u faces
    for (int j = 0; j < ny; ++j) {
        for (int i = 1; i < nx; ++i) {
            if (isSolid(i - 1, j) || isSolid(i, j)) { u[idxU(i, j)] = 0.0f; continue; }
            float gradp = (p[idxP(i, j)] - p[idxP(i - 1, j)]) / dx;
            u[idxU(i, j)] -= dt * gradp;
        }
    }

    // v faces
    for (int j = 1; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            if (isSolid(i, j - 1) || isSolid(i, j)) { v[idxV(i, j)] = 0.0f; continue; }
            float gradp = (p[idxP(i, j)] - p[idxP(i, j - 1)]) / dx;
            v[idxV(i, j)] -= dt * gradp;
        }
    }

    applyBoundary();

    // TEMP safety clamp (ok while debugging; remove later)
    const float MAX_FACE_SPEED = 5000.0f;
    for (float& val : u) val = clampf(val, -MAX_FACE_SPEED, MAX_FACE_SPEED);
    for (float& val : v) val = clampf(val, -MAX_FACE_SPEED, MAX_FACE_SPEED);

    // 5) measure divergence after projection
    computeDivergence();

    // to check how the divergence and speed improve
    stats.maxDivAfter = maxAbsDiv();
    stats.maxFaceSpeedAfter = maxFaceSpeed();
}