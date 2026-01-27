#include "mac_smoke_sim.h"
#include <cmath>
#include <algorithm>

MAC2D::MAC2D(int NX, int NY, float DX, float DT)
    : MACGridCore(NX, NY, DX, DT)
{
    recomputeValveIndices();
    reset();
}

void MAC2D::reset() {
    recomputeValveIndices();

    resetCore();

    const size_t Nc = (size_t)nx * (size_t)ny;
    smoke.assign(Nc, 0.0f);
    smoke0.assign(Nc, 0.0f);

    temp.assign(Nc, 0.0f);
    temp0.assign(Nc, 0.0f);

    age.assign(Nc, 0.0f);
    age0.assign(Nc, 0.0f);

    // Outer walls
    for (int i = 0; i < nx; ++i) {
        solid[idxP(i, 0)]       = 1;
        solid[idxP(i, ny - 1)]  = openTop ? 0 : 1;
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

    // clear smoke in solids
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            if (solid[idxP(i, j)]) smoke[idxP(i, j)] = 0.0f;
        }
    }

    invalidatePressureMatrix();
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

    invalidatePressureMatrix();
}

void MAC2D::step(float vortEps) {
    applyValveBC();
    applyBoundary();

    addForces(1.5f, 0.0f);
    applyBoundary();

    diffuseVelocityImplicit(); // implicit viscosity please god make it work

    advectVelocity();
    applyBoundary();

    addVorticityConfinement(vortEps);

    project();
    applyBoundary();

    advectScalar(temp,  temp0,  tempDissipation);
    advectScalar(smoke, smoke0, smokeDissipation);
    advectScalar(age,   age0,   1.0f);

    diffuseScalarImplicit(smoke, smoke0, smokeDiffusivity, 0.0f);

    if (openTop) {
        int j = ny - 1;
        for (int i = 0; i < nx; ++i) {
            if (solid[idxP(i, j)]) continue;
            smoke[idxP(i, j)] *= 0.0f;
            temp[idxP(i, j)]  = ambientTemp;
            age[idxP(i, j)]   = 0.0f;
        }
    }

    applyValveSink();

    coolAndDiffuseTemperature();
    updateAge(dt);

    applyBoundary();
}
