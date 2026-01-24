#include "mac_smoke_sim.h"
#include <cmath>

// file-local helper
namespace {
    static float dotVec(const std::vector<float>& a, const std::vector<float>& b) {
        double s = 0.0;
        for (size_t i = 0; i < a.size(); ++i)
            s += (double)a[i] * (double)b[i];
        return (float)s;
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
    Ax.assign(nx * ny, 0.0f);

    const float invDx2 = 1.0f / (dx * dx);

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int id = idxP(i, j);
            if (isSolid(i, j)) { Ax[id] = 0.0f; continue; }

            float center = x[id];
            float sum = 0.0f;
            int count = 0;

            // Only connect to non-solid neighbors
            if (i > 0     && !isSolid(i - 1, j)) { sum += x[idxP(i - 1, j)]; count++; }
            if (i + 1 < nx && !isSolid(i + 1, j)) { sum += x[idxP(i + 1, j)]; count++; }
            if (j > 0     && !isSolid(i, j - 1)) { sum += x[idxP(i, j - 1)]; count++; }
            if (j + 1 < ny) {
            if (!isSolid(i, j + 1)) { sum += x[idxP(i, j + 1)]; count++; }
                } else {
                    // top boundary neighbor is "outside"
                    // if top is CLOSED: treat outside as solid (do nothing extra, but keep count logic consistent)
                    // if top is OPEN: Neumann -> do NOT count an outside neighbor
                    if (!openTop) {
                        // closed: effectively a wall -> you can treat it like solid, meaning no connection
                        // (so: do nothing here)
                    }
                }

            // Discrete Laplacian: (count*center - sum) / dx^2
            Ax[id] = (count * center - sum) * invDx2;
        }
    }
}

void MAC2D::solvePressurePCG(int maxIters, float tol) {
    // b = div/dt (same as you already do)
    const int N = nx * ny;

    std::vector<float> b(N, 0.0f);
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int id = idxP(i,j);
            b[id] = isSolid(i,j) ? 0.0f : rhs[id];
    }
}

    // p initial guess = 0
    std::fill(p.begin(), p.end(), 0.0f);

    // r = b - A p  (but p=0 => r=b)
    std::vector<float> r = b;
    std::vector<float> z(N, 0.0f);
    std::vector<float> d(N, 0.0f);
    std::vector<float> q(N, 0.0f);

    // Jacobi preconditioner: z = M^{-1} r, where M is diagonal of A
    // diagonal = count/dx^2
    auto applyPrecond = [&](const std::vector<float>& rr, std::vector<float>& out) {
        out.assign(N, 0.0f);
        const float invDx2 = 1.0f / (dx * dx);
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int id = idxP(i, j);
                if (isSolid(i, j)) { out[id] = 0.0f; continue; }

                int count = 0;
                if (i > 0     && !isSolid(i - 1, j)) count++;
                if (i + 1 < nx && !isSolid(i + 1, j)) count++;
                if (j > 0     && !isSolid(i, j - 1)) count++;
                if (j + 1 < ny) {
                if (!isSolid(i, j + 1)) count++;
                } else {
                    if (!openTop) {
                        // closed top: do nothing (no neighbor)
                    } else {
                        // open top: do nothing (Neumann)
                    }
                }

                float diag = (float)count * invDx2;
                if (diag > 0.0f) out[id] = rr[id] / diag;
                else             out[id] = 0.0f;
            }
        }
    };

    applyPrecond(r, z);
    d = z;

    float deltaNew = dotVec(r, z);
    float delta0   = deltaNew;

    // If rhs is tiny, nothing to do
    if (deltaNew < 1e-30f) return;

    for (int it = 0; it < maxIters; ++it) {
        applyLaplacian(d, q);

        float dq = dotVec(d, q);
        if (std::fabs(dq) < 1e-30f) break;

        float alpha = deltaNew / dq;

        // p = p + alpha d
        // r = r - alpha q
        for (int k = 0; k < N; ++k) {
            p[k] += alpha * d[k];
            r[k] -= alpha * q[k];
        }

        // Convergence check using preconditioned residual energy
        // relative tolerance
        float rNorm2 = dotVec(r, r);
        if (rNorm2 < tol * tol) break;

        applyPrecond(r, z);

        float deltaOld = deltaNew;
        deltaNew = dotVec(r, z);

        float beta = deltaNew / (deltaOld + 1e-30f);

        // d = z + beta d
        for (int k = 0; k < N; ++k) {
            d[k] = z[k] + beta * d[k];
        }

        // optional: stricter relative check
        if (deltaNew < (tol * tol) * delta0) break;
    }
}

void MAC2D::project() {
    // 1) divergence of current velocity field
    computeDivergence();
    float maxDivBefore   = maxAbsDiv();
    float maxSpeedBefore = maxFaceSpeed();
    std::printf("[DEBUG] before project: max|div|=%g  maxFaceSpeed=%g\n",
                maxDivBefore, maxSpeedBefore);

    // 2) rhs for Poisson: Laplace(p) = div/dt  (solids -> 0)
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int id = idxP(i, j);
            rhs[id] = isSolid(i, j) ? 0.0f : (-div[id] / dt);
        }
    }

    // 3) solve A p = rhs
    solvePressurePCG(80, 1e-4f);

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
    float maxDivAfter   = maxAbsDiv();
    float maxSpeedAfter = maxFaceSpeed();
    std::printf("[DEBUG] after project:  max|div|=%g  maxFaceSpeed=%g\n",
                maxDivAfter, maxSpeedAfter);
}