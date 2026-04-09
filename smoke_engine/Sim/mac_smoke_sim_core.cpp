#include "mac_smoke_sim.h"
#include <cmath>
#include <algorithm>
#include <cstdio>
#include "smoke_diagnostics.h"

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

void MAC2D::addSolidText(const std::vector<uint8_t>& textMask, int maskW, int maskH) {
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int mi = (int)((float)i / (float)nx * (float)maskW);
            int mj = (int)((float)j / (float)ny * (float)maskH);
            mi = std::min(mi, maskW - 1);
            mj = std::min(mj, maskH - 1);

            if (textMask[(size_t)(mi + maskW * mj)]) {
                solid[idxP(i, j)] = 1;
                smoke[idxP(i, j)] = 0.0f;
                temp[idxP(i, j)]  = 0.0f;
                age[idxP(i, j)]   = 0.0f;
            }
        }
    }

    syncSolidsToFluidAndFaces();
    invalidatePressureMatrix();
}


void MAC2D::removeSolidText(const std::vector<uint8_t>& textMask, int maskW, int maskH) {
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int mi = (int)((float)i / (float)nx * (float)maskW);
            int mj = (int)((float)j / (float)ny * (float)maskH);
            mi = std::min(mi, maskW - 1);
            mj = std::min(mj, maskH - 1);

            if (textMask[(size_t)(mi + maskW * mj)]) {
                solid[idxP(i, j)] = 0;
            }
        }
    }

    syncSolidsToFluidAndFaces();
    invalidatePressureMatrix();
}



bool MAC2D::hasDynamicContent(float velEps, float scalarEps) const {
    if (valveOpen) return true;
    for (float value : u) if (std::fabs(value) > velEps) return true;
    for (float value : v) if (std::fabs(value) > velEps) return true;
    const std::size_t cellCount = smoke.size();
    for (std::size_t idx = 0; idx < cellCount; ++idx) {
        if (solid[idx]) continue;
        if (std::fabs(smoke[idx]) > scalarEps) return true;
        if (std::fabs(temp[idx]) > scalarEps) return true;
        if (std::fabs(age[idx]) > scalarEps) return true;
    }
    return false;
}

void MAC2D::clearDynamicState() {
    std::fill(u.begin(), u.end(), 0.0f);
    std::fill(v.begin(), v.end(), 0.0f);
    std::fill(p.begin(), p.end(), 0.0f);
    std::fill(div.begin(), div.end(), 0.0f);
    std::fill(rhs.begin(), rhs.end(), 0.0f);
    std::fill(smoke.begin(), smoke.end(), 0.0f);
    std::fill(temp.begin(), temp.end(), 0.0f);
    std::fill(age.begin(), age.end(), 0.0f);
    setPostBCStats(0.0f, 0.0f);
}

void MAC2D::step(float vortEps) {
    SMOKE_DIAG_PRINTF("[step] openTop=%d\n", (int)getOpenTop());
    // Velocity BC only (don’t inject scalars here)
    applyValveVelocityBC();
    applyBoundary();

    if (!hasDynamicContent()) {
        clearDynamicState();
        applyBoundary();
        return;
    }

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
    SMOKE_DIAG_PRINTF("[POST-PROJ] maxDiv=%g\n", maxAbsDiv());

    applyBoundary();

    computeDivergence();

    const float maxDivAfterBC  = maxAbsDiv();
    const float maxFaceAfterBC = maxFaceSpeed();

    setPostBCStats(maxDivAfterBC, maxFaceAfterBC);

    SMOKE_DIAG_PRINTF("[POST-BC2 ] maxDiv=%g\n", maxAbsDiv());

    computeDivergence();
    SMOKE_DIAG_PRINTF("[POST-BC] maxDiv=%g maxFace=%g\n", maxAbsDiv(), maxFaceSpeed());



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
        SMOKE_DIAG_PRINTF("[diag] Tdelta_min=%.3f Tdelta_max=%.3f maxFace=%.3f\n",
               minT, maxT, maxFace);
    } else {
        // temps stored as Kelvin; subtract ambient for delta
        SMOKE_DIAG_PRINTF("[diag] T_min=%.2f T_max=%.2f dT_max=%.2f maxFace=%.3f\n",
               minT, maxT, (maxT - ambientTempK), maxFace);
    }
}
