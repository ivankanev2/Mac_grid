#include "mac_smoke_sim.h"
#include <cstdio>

void MAC2D::recomputeValveIndices() {
    int w = std::max(2, (int)std::round(0.10f * nx)); // 10% of width, at least 2 cells
    int c = nx / 2;
    valveI0 = c - w/2;
    valveI1 = valveI0 + w - 1;

    // keep inside the domain (avoid the side-wall solid cells)
    valveI0 = std::max(1, std::min(valveI0, nx - 2));
    valveI1 = std::max(1, std::min(valveI1, nx - 2));

    std::printf("[VALVE] recompute: valveI0=%d valveI1=%d (nx=%d)\n", valveI0, valveI1, nx);
}

void MAC2D::applyBoundary() {
    // outer boundary no-through
    for (int j = 0; j < ny; j++) {
        u[idxU(0, j)] = 0.0f;
        u[idxU(nx, j)] = 0.0f;
    }
    for (int i = 0; i < nx; i++) {
    if (!(valveOpen && inValve(i))) v[idxV(i, 0)] = 0.0f;
    if (!openTop) v[idxV(i, ny)] = 0.0f;
    else          v[idxV(i, ny)] = v[idxV(i, ny-1)]; // zero-gradient outflow
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
            if (j == 0) {
                if (!(valveOpen && inValve(i))) v[idxV(i, 0)] = 0.0f;
                continue;
            }

            // TOP boundary: if openTop, do NOT zero v(i,ny) here (or you'll kill outflow)
            if (j == ny) {
                if (!openTop) v[idxV(i, ny)] = 0.0f;
                continue;
            }

            bool botSolid = isSolid(i, j - 1);
            bool topSolid = isSolid(i, j);
            if (botSolid || topSolid) v[idxV(i, j)] = 0.0f;
        }
    }
}

void MAC2D::applyValveBC() {
    if (!valveOpen) return;

    if (valveOpen) {
    std::printf("[VALVE] open, inletSpeed=%g inletSmoke=%g inletTemp=%g\n",
                inletSpeed, inletSmoke, inletTemp);
    }

    // impose upward inflow through bottom boundary faces
    for (int i = valveI0; i <= valveI1; ++i) {
        v[idxV(i, 0)] = inletSpeed; // +up into domain
    }

    // set scalars just inside the domain (row 1)
    int j = 1;
    for (int i = valveI0; i <= valveI1; ++i) {
        if (isSolid(i, j)) continue;
        int id = idxP(i, j);
        smoke[id] = inletSmoke;
        temp[id]  = inletTemp;
        age[id]   = 0.0f;
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
            temp[id]  = ambientTemp;
            age[id]   = 0.0f;
        }
    }
}

void MAC2D::setOpenTop(bool on) {
    openTop = on;
    for (int i = 0; i < nx; ++i) {
        solid[idxP(i, ny - 1)] = openTop ? 0 : 1;
    }
}

void MAC2D::enforceBoundaries() {
    applyBoundary();
}