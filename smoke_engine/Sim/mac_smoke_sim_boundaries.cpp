#include "mac_smoke_sim.h"
#include <cstdio>
#include <cmath>
#include <algorithm> 

void MAC2D::recomputeValveIndices() {
    int w = std::max(2, (int)std::round(0.10f * nx)); // 10% of width, at least 2 cells
    int c = nx / 2;
    valveI0 = c - w/2;
    valveI1 = valveI0 + w - 1;

    // keep inside the domain (avoid the side-wall solid cells)
    valveI0 = std::max(1, std::min(valveI0, nx - 2));
    valveI1 = std::max(1, std::min(valveI1, nx - 2));

}

void MAC2D::applyBoundary() {
    // outer boundary no-through (and no-slip tangential on the floor)

    // Left / right walls: u = 0
    for (int j = 0; j < ny; j++) {
        u[idxU(0, j)]  = 0.0f;
        u[idxU(nx, j)] = 0.0f;
    }

    

    // Floor / ceiling: v
    for (int i = 0; i < nx; i++) {
        // bottom: closed except valve
        if (!(valveOpen && inValve(i))) v[idxV(i, 0)] = 0.0f;

        // top: closed unless openTop (zero-gradient outflow)
        // top: open => zero normal gradient on v  (outflow style)
        if (getOpenTop()) {
            v[idxV(i, ny)] = v[idxV(i, ny - 1)];
        } else {
            v[idxV(i, ny)] = 0.0f;
        }
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
            // bottom handled already
            if (j == 0) continue;

            // top handled already (keep outflow if openTop)
            if (j == ny) continue;

            bool botSolid = isSolid(i, j - 1);
            bool topSolid = isSolid(i, j);
            if (botSolid || topSolid) v[idxV(i, j)] = 0.0f;
        }
    }
}

void MAC2D::applyValveVelocityBC() {
    if (!valveOpen) return;
    // no forced jet
    for (int i = valveI0; i <= valveI1; ++i)
        v[idxV(i, 0)] = 0.0f;
}

void MAC2D::addValveScalars() {
    if (!valveOpen) return;
    int j = 1;

    // Smoke: keep your rate-based injection
    const float add = inletSmoke * dt;
    const float maxD = 1.0f; // clamp max density but also makes the smoke brighter

    // Temperature: inject delta-K (relative to ambient)
    const float addT = inletTempDeltaK; // e.g. 20 K hotter than ambient

    for (int i = valveI0; i <= valveI1; ++i) {
        int id = idxP(i, j);
        if (isSolid(i, j)) continue;

        smoke[id] = std::min(maxD, smoke[id] + add);

        temp[id] = std::max(temp[id], addT);

        age[id] = 0.0f;
    }
}

void MAC2D::applyValveSink() {
    if (!valveOpen) return;

    // If flow is going OUT through the bottom (v < 0), delete scalars in that band.
    // This is a pragmatic “open boundary” for scalars.
    int j = 0;
    for (int i = valveI0; i <= valveI1; ++i) {
        float vb = v[idxV(i, 0)];
        if (vb < 0.0f) {
            int id = idxP(i, j);
            smoke[id] = 0.0f;
            temp[id] = 0.0f;
            age[id]   = 0.0f;
        }
    }
}

void MAC2D::setOpenTop(bool on)
{
    printf("[setOpenTop] requested=%d current=%d\n", (int)on, (int)getOpenTop());
    if (getOpenTop() == on) return;

    setOpenTopBC(on);

    const int jTop = ny - 1;

    if (on) {
        // OPEN: clear the top wall cells (keep corners optional)
        for (int i = 1; i < nx - 1; ++i) {
            solid[idxP(i, jTop)] = 0;
        }
    } else {
        // CLOSED: restore the top wall cells
        for (int i = 1; i < nx - 1; ++i) {
            solid[idxP(i, jTop)] = 1;
        }
    }

    // Make fluid consistent + rebuild face openness + invalidate operators
    syncSolidsToFluidAndFaces();   // this calls rebuildFaceOpenness + invalidatePressureMatrix()

    int solidTop = 0;
    for (int i = 0; i < nx; ++i) solidTop += (solid[idxP(i, ny-1)] != 0);
    printf("[setOpenTop] solidTop=%d / %d\n", solidTop, nx);

    // IMPORTANT: make sure the projection domain + operator reflect the new BC immediately
    setFluidMaskAllNonSolid();      // smoke sim should treat all non-solid as fluid
    invalidatePressureMatrix();     // rebuild Laplacian + MG if needed

    // Optional but makes it visually “snap” open right away:
    if (on) {
        for (int i = 0; i < nx; ++i)
            v[idxV(i, ny)] = v[idxV(i, ny - 1)];
    } else {
        for (int i = 0; i < nx; ++i)
            v[idxV(i, ny)] = 0.0f;
    }

    enforceBoundaries();

    printf("[setOpenTop] applied. now=%d\n", (int)getOpenTop());
}

void MAC2D::enforceBoundaries() {
    applyBoundary();
}
