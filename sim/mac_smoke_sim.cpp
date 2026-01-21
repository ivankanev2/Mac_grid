// mac_smoke_sim.cpp
#include "mac_smoke_sim.h"
#include <cmath>
#include <algorithm>
#include <iostream>

// ---------------------------
// Local helper accessors
// (idxP is already inline in the header as a public method)
// ---------------------------
static inline float clampf(float x, float a, float b) {
    return std::max(a, std::min(b, x));
}



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
    i = std::clamp(i, 0, nx - 1);
    j = std::clamp(j, 0, ny - 1);
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

    i0 = std::clamp(i0, 0, nx - 1);
    j0 = std::clamp(j0, 0, ny - 1);
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

    i0 = std::clamp(i0, 0, nx);
    j0 = std::clamp(j0, 0, ny - 1);
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

    i0 = std::clamp(i0, 0, nx - 1);
    j0 = std::clamp(j0, 0, ny);
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

void MAC2D::advectSmoke(float dissipation) {
    smoke0 = smoke;

    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            if (isSolid(i, j)) { smoke[idxP(i, j)] = 0.0f; continue; }

            float x = (i + 0.5f) * dx;
            float y = (j + 0.5f) * dx;

            float ux, vy;
            velAt(x, y, u, v, ux, vy);

            float x0 = clampf(x - dt * ux, 0.0f, nx * dx);
            float y0 = clampf(y - dt * vy, 0.0f, ny * dx);

            int si, sj;
            worldToCell(x0, y0, si, sj);

            float s = (!isSolid(si, sj)) ? sampleCellCentered(smoke0, x0, y0) : 0.0f;
            smoke[idxP(i, j)] = dissipation * s;
        }
    }
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
void MAC2D::step() {
    // smoke source (you can remove this once UI places sources)
    addSmokeSource(0.5f * nx * dx, 0.2f * ny * dx, 0.10f * nx * dx, 0.08f);

    addForces(/*buoyancy=*/3.5f, /*gravity=*/0.0f);
    applyBoundary();

    advectVelocity();
    applyBoundary();

    project();

    // debug
    computeDivergence();
    std::cout << "max |div| after project: " << maxAbsDiv() << "\n";

    advectSmoke(/*dissipation=*/0.995f);
}