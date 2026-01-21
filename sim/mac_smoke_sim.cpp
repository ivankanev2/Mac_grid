// mac_smoke_sim.cpp
#include "mac_smoke_sim.h"
#include <cmath>
#include <algorithm>
#include <iostream>

// ---------------------------
// Local helper accessors
// (idxP is already inline in the header as a public method)
// ---------------------------




// ---------------------------
// Constructor / Reset
// ---------------------------
MAC2D::MAC2D(int NX, int NY, float DX, float DT)
    : nx(NX), ny(NY), dx(DX), dt(DT)
{
    reset();
}

void MAC2D::reset() {
    const size_t Nu = (size_t)(nx + 1) * (size_t)ny;
    const size_t Nv = (size_t)nx * (size_t)(ny + 1);
    const size_t Nc = (size_t)nx * (size_t)ny;

    // If anything is not the expected size, do a full re-init (safe path)
    if (u.size() != Nu || v.size() != Nv || p.size() != Nc || smoke.size() != Nc ||
        u0.size() != Nu || v0.size() != Nv || smoke0.size() != Nc ||
        div.size() != Nc || rhs.size() != Nc || solid.size() != Nc) {

        u.assign(Nu, 0.0f);
        v.assign(Nv, 0.0f);
        p.assign(Nc, 0.0f);
        smoke.assign(Nc, 0.0f);

        u0 = u; v0 = v; smoke0 = smoke;
        div.assign(Nc, 0.0f);
        rhs.assign(Nc, 0.0f);

        solid.assign(Nc, 0);

        // walls
        for (int i = 0; i < nx; ++i) { solid[idxP(i, 0)] = 1; solid[idxP(i, ny - 1)] = 1; }
        for (int j = 0; j < ny; ++j) { solid[idxP(0, j)] = 1; solid[idxP(nx - 1, j)] = 1; }

        return;
    }

    // Fast path: keep solid blocks, reset dynamic fields
    std::fill(u.begin(), u.end(), 0.0f);
    std::fill(v.begin(), v.end(), 0.0f);
    std::fill(p.begin(), p.end(), 0.0f);
    std::fill(smoke.begin(), smoke.end(), 0.0f);
    std::fill(u0.begin(), u0.end(), 0.0f);
    std::fill(v0.begin(), v0.end(), 0.0f);
    std::fill(smoke0.begin(), smoke0.end(), 0.0f);
    std::fill(div.begin(), div.end(), 0.0f);
    std::fill(rhs.begin(), rhs.end(), 0.0f);

    // Re-enforce walls as solid (does NOT delete user solids)
    for (int i = 0; i < nx; ++i) { solid[idxP(i, 0)] = 1; solid[idxP(i, ny - 1)] = 1; }
    for (int j = 0; j < ny; ++j) { solid[idxP(0, j)] = 1; solid[idxP(nx - 1, j)] = 1; }

    // Optional: make sure no smoke lives inside solids
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
            if (solid[idxP(i, j)]) smoke[idxP(i, j)] = 0.0f;
}

// ---------------------------
// Utility
// ---------------------------
float MAC2D::maxAbsDiv() const {
    float m = 0.0f;
    for (float d : div) m = std::max(m, std::abs(d));
    return m;
}

void MAC2D::worldToCell(float x, float y, int &i, int &j) const {
    // cell centers at (i+0.5)*dx
    float fx = x / dx - 0.5f;
    float fy = y / dx - 0.5f;
    i = (int)std::floor(fx);
    j = (int)std::floor(fy);
    i = (int)clampf((float)i, 0.0f, (float)(nx - 1));
    j = (int)clampf((float)j, 0.0f, (float)(ny - 1));
}

// ---------------------------
// Sampling (bilinear)
// ---------------------------
float MAC2D::sampleCellCentered(const std::vector<float>& f, float x, float y) const {
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

float MAC2D::sampleU(const std::vector<float>& fu, float x, float y) const {
    // u at (i*dx, (j+0.5)*dx)
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

float MAC2D::sampleV(const std::vector<float>& fv, float x, float y) const {
    // v at ((i+0.5)*dx, j*dx)
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

void MAC2D::velAt(float x, float y,
                  const std::vector<float>& fu,
                  const std::vector<float>& fv,
                  float& outUx, float& outVy) const
{
    outUx = sampleU(fu, x, y);
    outVy = sampleV(fv, x, y);
}

// ---------------------------
// Physics steps
// ---------------------------
void MAC2D::addForces(float buoyancy, float gravity) {
    // add vertical force to v faces using smoke as buoyancy
    for (int j = 0; j <= ny; j++) {
        for (int i = 0; i < nx; i++) {
            float x = (i + 0.5f) * dx;
            float y = (j) * dx;

            float s = sampleCellCentered(smoke, x, y);
            v[idxV(i, j)] += dt * (buoyancy * s + gravity);
        }
    }
}

void MAC2D::applyBoundary() {
    // outer boundary no-through
    for (int j = 0; j < ny; j++) {
        u[idxU(0, j)] = 0.0f;
        u[idxU(nx, j)] = 0.0f;
    }
    for (int i = 0; i < nx; i++) {
        v[idxV(i, 0)] = 0.0f;
        v[idxV(i, ny)] = 0.0f;
    }

    // no-through for internal solids:
    // u faces between (i-1,j) and (i,j)
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i <= nx; ++i) {
            bool leftSolid  = (i - 1 >= 0) ? isSolid(i - 1, j) : true;
            bool rightSolid = (i < nx)     ? isSolid(i, j)     : true;
            if (leftSolid || rightSolid) u[idxU(i, j)] = 0.0f;
        }
    }

    // v faces between (i,j-1) and (i,j)
    for (int j = 0; j <= ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            bool botSolid = (j - 1 >= 0) ? isSolid(i, j - 1) : true;
            bool topSolid = (j < ny)     ? isSolid(i, j)     : true;
            if (botSolid || topSolid) v[idxV(i, j)] = 0.0f;
        }
    }
}

void MAC2D::advectVelocity() {
    u0 = u;
    v0 = v;

    // Advect u
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

    // Advect v
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
            bool topSolid = (j < ny)     ? isSolid(i, j)     : true;
            if (botSolid || topSolid) v[idxV(i, j)] = 0.0f;
        }
    }

    applyBoundary();
}

// Vorticity confinement
void MAC2D::addVorticityConfinement(float eps) {
    if (eps <= 0.0f) return;

    const int Nc = nx * ny;
    std::vector<float> omega(Nc, 0.0f);
    std::vector<float> mag(Nc, 0.0f);
    std::vector<float> fx(Nc, 0.0f);
    std::vector<float> fy(Nc, 0.0f);

    auto idx = [&](int i, int j) { return idxP(i, j); };

    // --- 1) vorticity at cell centers: ω = dv/dx - du/dy
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            if (isSolid(i, j)) { omega[idx(i,j)] = 0.0f; continue; }

            // dv/dx at center using v averaged to center rows
            // v is at (i+0.5, j) and (i+0.5, j+1)
            float vL = 0.5f * (v[idxV(i, j)]     + v[idxV(i, j + 1)]);
            float vR = 0.5f * (v[idxV(std::min(i + 1, nx - 1), j)] +
                               v[idxV(std::min(i + 1, nx - 1), j + 1)]);

            // du/dy at center using u averaged to center columns
            // u is at (i, j+0.5) and (i+1, j+0.5)
            float uB = 0.5f * (u[idxU(i, j)]     + u[idxU(i + 1, j)]);
            float uT = 0.5f * (u[idxU(i, std::min(j + 1, ny - 1))] +
                               u[idxU(i + 1, std::min(j + 1, ny - 1))]);

            float dv_dx = (vR - vL) / dx;
            float du_dy = (uT - uB) / dx;

            float w = dv_dx - du_dy;
            omega[idx(i,j)] = w;
            mag[idx(i,j)] = std::fabs(w);
        }
    }

    // --- 2) N = ∇|ω| / |∇|ω||, force = eps * dx * (N × ω)
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            if (isSolid(i, j)) continue;

            int im1 = std::max(i - 1, 0);
            int ip1 = std::min(i + 1, nx - 1);
            int jm1 = std::max(j - 1, 0);
            int jp1 = std::min(j + 1, ny - 1);

            float dmx = (mag[idx(ip1, j)] - mag[idx(im1, j)]) / (2.0f * dx);
            float dmy = (mag[idx(i, jp1)] - mag[idx(i, jm1)]) / (2.0f * dx);

            float len = std::sqrt(dmx*dmx + dmy*dmy) + 1e-6f;
            float Nx = dmx / len;
            float Ny = dmy / len;

            float w = omega[idx(i,j)];

            // 2D: N x (0,0,w) = (Ny*w, -Nx*w, 0)
            fx[idx(i,j)] = eps * dx * (Ny * w);
            fy[idx(i,j)] = eps * dx * (-Nx * w);
        }
    }

    // --- 3) Apply to faces (average adjacent cell forces onto the face)
    // u faces: between cell (i-1,j) and (i,j)
    for (int j = 0; j < ny; ++j) {
        for (int i = 1; i < nx; ++i) {
            bool solidL = isSolid(i - 1, j);
            bool solidR = isSolid(i, j);
            if (solidL || solidR) continue;

            float f = 0.5f * (fx[idx(i - 1, j)] + fx[idx(i, j)]);
            u[idxU(i, j)] += dt * f;
        }
    }

    // v faces: between cell (i,j-1) and (i,j)
    for (int j = 1; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            bool solidB = isSolid(i, j - 1);
            bool solidT = isSolid(i, j);
            if (solidB || solidT) continue;

            float f = 0.5f * (fy[idx(i, j - 1)] + fy[idx(i, j)]);
            v[idxV(i, j)] += dt * f;
        }
    }
}

void MAC2D::addVelocityImpulse(float cx, float cy, float radius, float strength) {
    // Add a circular rotational impulse: u and v get +/- tangential components
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int id = idxP(i,j);
            if (isSolid(i,j)) continue;
            float x = (i + 0.5f) * dx;
            float y = (j + 0.5f) * dx;
            float rx = x - cx;
            float ry = y - cy;
            float r2 = rx*rx + ry*ry;
            if (r2 <= radius*radius) {
                float r = std::sqrt(r2) + 1e-9f;
                // tangential unit vector = (-ry/r, rx/r)
                float tux = -ry / r;
                float tvy =  rx / r;
                float w = (1.0f - (r / radius)) * strength; // falloff
                // add to face arrays by nearest faces (simple splat)
                int iu = (int)clampf((float)i, 0.0f, (float)nx); // used for u indexing safety below
                // add to surrounding faces (u and v grids)
                // add u on the two vertical faces around this cell
                u[idxU(i, j)] += 0.5f * w * tux;
                u[idxU(i+1 <= nx ? i+1 : nx, j)] += 0.5f * w * tux;
                // add v on the two horizontal faces
                v[idxV(i, j)] += 0.5f * w * tvy;
                v[idxV(i, j+1 <= ny ? j+1 : ny)] += 0.5f * w * tvy;
            }
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

void MAC2D::solvePressure(int iters) {
    std::fill(p.begin(), p.end(), 0.0f);

    for (int k = 0; k < nx * ny; k++) rhs[k] = div[k] / dt;

    for (int it = 0; it < iters; it++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                if (isSolid(i, j)) { p[idxP(i, j)] = 0.0f; continue; }

                float sum = 0.0f;
                int count = 0;

                if (i > 0     && !isSolid(i - 1, j)) { sum += p[idxP(i - 1, j)]; count++; }
                if (i + 1 < nx && !isSolid(i + 1, j)) { sum += p[idxP(i + 1, j)]; count++; }
                if (j > 0     && !isSolid(i, j - 1)) { sum += p[idxP(i, j - 1)]; count++; }
                if (j + 1 < ny && !isSolid(i, j + 1)) { sum += p[idxP(i, j + 1)]; count++; }

                if (count == 0) { p[idxP(i, j)] = 0.0f; continue; }

                float b = rhs[idxP(i, j)];
                p[idxP(i, j)] = (sum - b * dx * dx) / (float)count;
            }
        }
    }
}

void MAC2D::project() {
    computeDivergence();
    solvePressure(80);

    // u faces
    for (int j = 0; j < ny; j++) {
        for (int i = 1; i < nx; i++) {
            if (isSolid(i - 1, j) || isSolid(i, j)) { u[idxU(i, j)] = 0.0f; continue; }
            float gradp = (p[idxP(i, j)] - p[idxP(i - 1, j)]) / dx;
            u[idxU(i, j)] -= dt * gradp;
        }
    }

    // v faces
    for (int j = 1; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            if (isSolid(i, j - 1) || isSolid(i, j)) { v[idxV(i, j)] = 0.0f; continue; }
            float gradp = (p[idxP(i, j)] - p[idxP(i, j - 1)]) / dx;
            v[idxV(i, j)] -= dt * gradp;
        }
    }

    applyBoundary();
}

#include <limits> // add at top of mac_smoke_sim.cpp for numeric_limits

void MAC2D::advectSmoke(float dissipation /*= 0.995f*/) {
    // Source field at time n
    smoke0 = smoke;

    // Temporary buffers
    std::vector<float> smokeFwd(smoke.size(), 0.0f);
    std::vector<float> smokeBack(smoke.size(), 0.0f);

    // Helper: compute min/max of the *bilinear stencil* used by sampleCellCentered at (x,y)
    auto stencilMinMax = [&](const std::vector<float>& f, float x, float y, float& outMin, float& outMax) {
        // Same index math as sampleCellCentered
        float fx = x / dx - 0.5f;
        float fy = y / dx - 0.5f;

        int i0 = (int)std::floor(fx);
        int j0 = (int)std::floor(fy);

        // clamp to valid cells
        i0 = (int)clampf((float)i0, 0.0f, (float)(nx - 1));
        j0 = (int)clampf((float)j0, 0.0f, (float)(ny - 1));
        int i1 = std::min(i0 + 1, nx - 1);
        int j1 = std::min(j0 + 1, ny - 1);

        outMin =  std::numeric_limits<float>::infinity();
        outMax = -std::numeric_limits<float>::infinity();

        auto consider = [&](int i, int j) {
            if (isSolid(i, j)) return; // ignore solid samples
            float v = f[idxP(i, j)];
            outMin = std::min(outMin, v);
            outMax = std::max(outMax, v);
        };

        consider(i0, j0);
        consider(i1, j0);
        consider(i0, j1);
        consider(i1, j1);

        // If all stencil cells were solid, just clamp to 0
        if (!std::isfinite(outMin)) { outMin = 0.0f; outMax = 0.0f; }
    };

    // -------------------------
    // 1) Forward semi-Lagrangian: smoke0 -> smokeFwd
    // -------------------------
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int id = idxP(i, j);
            if (isSolid(i, j)) { smokeFwd[id] = 0.0f; continue; }

            float x = (i + 0.5f) * dx;
            float y = (j + 0.5f) * dx;

            float ux, vy;
            velAt(x, y, u, v, ux, vy);

            // Backtrace
            float x0 = clampf(x - dt * ux, 0.0f, nx * dx);
            float y0 = clampf(y - dt * vy, 0.0f, ny * dx);

            int si, sj;
            worldToCell(x0, y0, si, sj);

            smokeFwd[id] = (!isSolid(si, sj)) ? sampleCellCentered(smoke0, x0, y0) : 0.0f;
        }
    }

    // -------------------------
    // 2) Backward semi-Lagrangian: smokeFwd -> smokeBack (using -dt)
    // -------------------------
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int id = idxP(i, j);
            if (isSolid(i, j)) { smokeBack[id] = 0.0f; continue; }

            float x = (i + 0.5f) * dx;
            float y = (j + 0.5f) * dx;

            float ux, vy;
            velAt(x, y, u, v, ux, vy);

            // "Backwards" pass: trace with -dt => x1 = x + dt*u
            float x1 = clampf(x + dt * ux, 0.0f, nx * dx);
            float y1 = clampf(y + dt * vy, 0.0f, ny * dx);

            int si, sj;
            worldToCell(x1, y1, si, sj);

            smokeBack[id] = (!isSolid(si, sj)) ? sampleCellCentered(smokeFwd, x1, y1) : 0.0f;
        }
    }

    // -------------------------
    // 3) MacCormack correction + clamp to stencil min/max
    // -------------------------
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int id = idxP(i, j);
            if (isSolid(i, j)) { smoke[id] = 0.0f; continue; }

            float x = (i + 0.5f) * dx;
            float y = (j + 0.5f) * dx;

            float ux, vy;
            velAt(x, y, u, v, ux, vy);

            // Same forward backtrace point (so we clamp to the same donor stencil)
            float x0 = clampf(x - dt * ux, 0.0f, nx * dx);
            float y0 = clampf(y - dt * vy, 0.0f, ny * dx);

            // correction
            float phiF = smokeFwd[id];
            float phiB = smokeBack[id];
            float corrected = phiF + 0.5f * (smoke0[id] - phiB);

            // clamp to min/max of donor stencil (prevents overshoot/ringing)
            float mn, mx;
            stencilMinMax(smoke0, x0, y0, mn, mx);
            corrected = clampf(corrected, mn, mx);

            // dissipation
            smoke[id] = dissipation * corrected;
        }
    }
}

float MAC2D::maxFaceSpeed() const {
    float m = 0.0f;

    // u faces: (nx+1)*ny
    for (float val : u) m = std::max(m, std::fabs(val));

    // v faces: nx*(ny+1)
    for (float val : v) m = std::max(m, std::fabs(val));

    return m;
}

// ---------------------------
// Public interaction API
// ---------------------------
void MAC2D::addSmokeSource(float cx, float cy, float radius, float amount) {
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            if (isSolid(i, j)) continue;

            float x = (i + 0.5f) * dx;
            float y = (j + 0.5f) * dx;
            float dx0 = x - cx;
            float dy0 = y - cy;
            if (dx0 * dx0 + dy0 * dy0 <= radius * radius) {
                smoke[idxP(i, j)] = std::min(1.0f, smoke[idxP(i, j)] + amount);
            }
        }
    }
}

void MAC2D::addSolidCircle(float cx, float cy, float r) {
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            float x = (i + 0.5f) * dx;
            float y = (j + 0.5f) * dx;
            float dx0 = x - cx;
            float dy0 = y - cy;
            if (dx0 * dx0 + dy0 * dy0 <= r * r) {
                solid[idxP(i, j)] = 1;
                smoke[idxP(i, j)] = 0.0f;
            }
        }
    }
}

// ---------------------------
// Main step
// ---------------------------
void MAC2D::step(float vortEps) {
    // smoke source (you can remove this once UI places sources)
    addSmokeSource(0.5f * nx * dx, 0.2f * ny * dx, 0.10f * nx * dx, 0.08f);

    addForces(/*buoyancy=*/3.5f, /*gravity=*/0.0f);
    applyBoundary();

    advectVelocity();
    applyBoundary();

    addVorticityConfinement(vortEps);
    applyBoundary();

    project();

    // debug
    computeDivergence();
    std::cout << "max |div| after project: " << maxAbsDiv() << "\n";

    advectSmoke(/*dissipation=*/0.995f);
}