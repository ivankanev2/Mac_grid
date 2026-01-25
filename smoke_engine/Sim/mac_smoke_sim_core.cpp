// mac_smoke_sim.cpp
#include "mac_smoke_sim.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <limits> 


//TODO: check whether our formulas and anything actually works

// ---------------------------
// Constructor / Reset
// ---------------------------
MAC2D::MAC2D(int NX, int NY, float DX, float DT)
    : nx(NX), ny(NY), dx(DX), dt(DT)
{
    recomputeValveIndices();
    reset();
}

void MAC2D::reset() {

    recomputeValveIndices();

    const size_t Nu = (size_t)(nx + 1) * (size_t)ny;
    const size_t Nv = (size_t)nx * (size_t)(ny + 1);
    const size_t Nc = (size_t)nx * (size_t)ny;

    // If anything is not the expected size, do a full re-init (safe path)
    if (u.size() != Nu || v.size() != Nv || p.size() != Nc || smoke.size() != Nc ||
        u0.size() != Nu || v0.size() != Nv || smoke0.size() != Nc ||
        div.size() != Nc || rhs.size() != Nc || solid.size() != Nc ||
        temp.size() != Nc || temp0.size() != Nc || age.size() != Nc || age0.size() != Nc)
    {
        u.assign(Nu, 0.0f);
        v.assign(Nv, 0.0f);
        p.assign(Nc, 0.0f);
        smoke.assign(Nc, 0.0f);

        u0 = u;
        v0 = v;
        smoke0 = smoke;

        div.assign(Nc, 0.0f);
        rhs.assign(Nc, 0.0f);

        solid.assign(Nc, 0);

        temp.assign(Nc, 0.0f);
        temp0 = temp;

        age.assign(Nc, 0.0f);
        age0 = age;

        // Outer walls
        for (int i = 0; i < nx; ++i) {
            solid[idxP(i, 0)]       = 1;
            solid[idxP(i, ny - 1)]  = openTop ? 0 : 1; // open top => not solid
        }
        for (int j = 0; j < ny; ++j) {
            solid[idxP(0, j)]       = 1;
            solid[idxP(nx - 1, j)]  = 1;
        }

        // carve valve opening in the bottom row
        for (int i = valveI0; i <= valveI1; ++i) {
            solid[idxP(i, 0)] = 0;
        }

        // build pipe solids if any
        rebuildSolidsFromPipe(false);

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

    std::fill(temp.begin(), temp.end(), 0.0f);
    std::fill(temp0.begin(), temp0.end(), 0.0f);

    std::fill(age.begin(), age.end(), 0.0f);
    std::fill(age0.begin(), age0.end(), 0.0f);

    // Re-enforce outer walls as solid (does NOT delete user solids)
    for (int i = 0; i < nx; ++i) {
        solid[idxP(i, 0)]      = 1;
        solid[idxP(i, ny - 1)] = openTop ? 0 : 1;
    }
    for (int j = 0; j < ny; ++j) {
        solid[idxP(0, j)]      = 1;
        solid[idxP(nx - 1, j)] = 1;
    }

    // carve valve opening (MUST be in both reset paths)
    for (int i = valveI0; i <= valveI1; ++i) {
        solid[idxP(i, 0)] = 0;
    }

    // rebuild pipe solids on top of the base walls
    rebuildSolidsFromPipe(false);

    // Optional: make sure no smoke lives inside solids
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            if (solid[idxP(i, j)]) smoke[idxP(i, j)] = 0.0f;
        }
    }

    markPressureMatrixDirty();
}

float MAC2D::maxFaceSpeed() const {
    float m = 0.0f;

    // u faces: (nx+1)*ny
    for (float val : u) m = std::max(m, std::fabs(val));

    // v faces: nx*(ny+1)
    for (float val : v) m = std::max(m, std::fabs(val));

    return m;
}

void MAC2D::addSmokeSource(float cx, float cy, float radius, float amount) {
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            if (solid[idxP(i, j)]) continue;

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
                temp[idxP(i, j)]  = 0.0f;
                age[idxP(i, j)]   = 0.0f;
            }
        }
    }

    markPressureMatrixDirty();
}

void MAC2D::step(float vortEps) {
    // 0) Inlet BC FIRST so the solver sees it (and doesn't "inject divergence" after projection)
    applyValveBC();
    applyBoundary();

    // 1) Forces
    addForces(1.5f, 0.0f);
    applyBoundary();

    // 2) Advect velocity
    advectVelocity();
    applyBoundary();

    // 3) Vorticity confinement
    addVorticityConfinement(vortEps);

    // 4) Pressure projection (make velocity divergence-free)
    project();
    applyBoundary();

    // 5) Advect scalars ONCE
    advectScalar(temp,  temp0,  tempDissipation);
    advectScalar(smoke, smoke0, smokeDissipation);
    advectScalar(age,   age0,   1.0f);

    // 6) Top outflow “sink” for scalars
    if (openTop) {
        int j = ny - 1;
        for (int i = 0; i < nx; ++i) {
            if (solid[idxP(i, j)]) continue;
            smoke[idxP(i, j)] *= 0.0f;        // hard delete at top row
            temp[idxP(i, j)]  = ambientTemp;
            age[idxP(i, j)]   = 0.0f;
        }
    }

    // 7) Bottom valve sink (if flow reverses out)
    applyValveSink();

    // 8) Temperature evolution (cooling/diffusion) + age update
    coolAndDiffuseTemperature();
    updateAge(dt);

    // 9) Re-apply BCs at end (keeps valve u-tangential killed and prevents "leak")
    applyBoundary();
}