#include "mac_smoke_sim.h"
#include <cmath>
#include <algorithm>

void MAC2D::applyScalarOutflowTop(std::vector<float>& phi, float outsideValue, int layers)
{
    if (!getOpenTop()) return;
    layers = std::max(1, std::min(layers, ny));

    for (int j = ny - layers; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int id = idxP(i, j);
            if (solid[id]) continue;

            // outward face velocity above this cell (v on the face at j+1)
            float vout = v[idxV(i, j + 1)];
            if (vout <= 0.0f) continue;

            // blend factor ~ Courant number across the top boundary
            float a = clampf(dt * vout / dx, 0.0f, 1.0f);
            phi[id] = (1.0f - a) * phi[id] + a * outsideValue;
        }
    }
}

inline void clampFaceSpeeds(std::vector<float>& u, std::vector<float>& v, float maxAbs) {
    for (float &x : u) if (std::fabs(x) > maxAbs) x = (x > 0 ? maxAbs : -maxAbs);
    for (float &y : v) if (std::fabs(y) > maxAbs) y = (y > 0 ? maxAbs : -maxAbs);
}

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
        solid[idxP(i, 0)]      = 1;   // bottom wall cells (except valve gets carved)
        solid[idxP(i, ny - 1)] = 0;   // IMPORTANT: top row is never solid; BC handles open/closed
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

    syncSolidsToFluidAndFaces();


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
    printf("[step] openTop=%d\n", (int)getOpenTop());
    // Velocity BC only (don’t inject scalars here)
    applyValveVelocityBC();
    applyBoundary();

    // Velocity sim
    advectVelocity();
    applyBoundary();

    addForces(1.5f, 0.0f);
    applyBoundary();
    clampFaceSpeeds(u, v, 50.0f);

    diffuseVelocityImplicit();
    applyBoundary();

    addVorticityConfinement(vortEps);

    // setOpenTop(openTop);
    project();

    computeDivergence();
    std::printf("[POST-PROJ] maxDiv=%g\n", maxAbsDiv());

    applyBoundary();

    computeDivergence();

    const float maxDivAfterBC  = maxAbsDiv();
    const float maxFaceAfterBC = maxFaceSpeed();

    setPostBCStats(maxDivAfterBC, maxFaceAfterBC);

    std::printf("[POST-BC2 ] maxDiv=%g\n", maxAbsDiv());

    computeDivergence();
    printf("[POST-BC] maxDiv=%g maxFace=%g\n", maxAbsDiv(), maxFaceSpeed());



    // Scalars
    advectScalar(temp,  temp0,  tempDissipation);
    advectScalar(smoke, smoke0, smokeDissipation);
    advectScalar(age,   age0,   1.0f);

    addValveScalars();
    diffuseScalarImplicit(smoke, smoke0, smokeDiffusivity, 0.0f);
    clampFaceSpeeds(u, v, 50.0f);
    coolAndDiffuseTemperature();
    updateAge(dt);


    // top outflow (now it won’t fight your injection timing)
    if (getOpenTop()) {
        applyScalarOutflowTop(smoke, 0.0f,        4);
        applyScalarOutflowTop(temp,  0.0f, 4);
        applyScalarOutflowTop(age,   0.0f,        4);
    }

    applyValveSink();
    applyBoundary();

    float maxT = -1e9f, minT = 1e9f;
    for (float T : temp) { maxT = std::max(maxT, T); minT = std::min(minT, T); }
    float maxFace = 0.0f;
    for (float val : u) maxFace = std::max(maxFace, fabsf(val));
    for (float val : v) maxFace = std::max(maxFace, fabsf(val));

    if (!inletTempIsAbsoluteK) {
        // temps are stored as delta relative to ambient; display delta
        printf("[diag] Tdelta_min=%.3f Tdelta_max=%.3f maxFace=%.3f\n",
               minT, maxT, maxFace);
    } else {
        // temps stored as Kelvin; subtract ambient for delta
        printf("[diag] T_min=%.2f T_max=%.2f dT_max=%.2f maxFace=%.3f\n",
               minT, maxT, (maxT - ambientTempK), maxFace);
    }
}
