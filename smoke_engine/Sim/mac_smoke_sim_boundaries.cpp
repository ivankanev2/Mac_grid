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

    // // Floor: kill tangential component u(i,0) to prevent "valve sideways leak"
    // its fixed, keeping it just in case
    // for (int i = 0; i <= nx; ++i) {
    //     u[idxU(i, 0)] = 0.0f;
    // }

    // Floor / ceiling: v
    for (int i = 0; i < nx; i++) {
        // bottom: closed except valve
        if (!(valveOpen && inValve(i))) v[idxV(i, 0)] = 0.0f;

        // top: closed unless openTop (zero-gradient outflow)
        if (!openTop) {
        v[idxV(i, ny)] = 0.0f;   // closed top
        } else {
            // openTopBC: do nothing here (projection already set v(i,ny))
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

// void MAC2D::applyValveBC() {
//     if (!valveOpen) return;

//     // 1) impose upward inflow through bottom boundary faces
//     for (int i = valveI0; i <= valveI1; ++i) {
//         v[idxV(i, 0)] = inletSpeed; // +up into domain
//     }

//     // 2) Kill tangential velocity along the opening to prevent sideways "leak"
//     //
//     // u lives on vertical faces: i in [0..nx], so the valve opening affects faces
//     // from i=valveI0 .. valveI1+1 (inclusive).
//     //
//     // Clamp into valid u-face index range.
//     int u0 = std::max(0, valveI0);
//     int u1 = std::min(nx, valveI1 + 1);

//     for (int i = u0; i <= u1; ++i) {
//         u[idxU(i, 0)] = 0.0f;
//     }

//     // Optional but often helps a LOT:
//     // also kill tangential velocity one cell above the inlet lip (prevents "jet attaching")
//     // Comment this out if you dislike how "engineered" it feels.
//     for (int i = u0; i <= u1; ++i) {
//         if (ny > 1) u[idxU(i, 1)] = 0.0f;
//     }

//     // 3) Inject smoke/temp into first fluid row above the boundary (j=1)
//     int j = 1;
//     if (j < ny) {
//         for (int i = valveI0; i <= valveI1; ++i) {
//             if (!isSolid(i, j)) {
//                 smoke[idxP(i, j)] = inletSmoke;
//                 temp[idxP(i, j)]  = inletTemp;
//                 age[idxP(i, j)]   = 0.0f;   // important: otherwise age can “teleport old smoke” into new inflow
//             }
//         }
//     }
// }

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

void MAC2D::setOpenTop(bool on) {
    openTop = on;
    MACGridCore::setOpenTop(on); // forgot to add this earlier ops
    for (int i = 0; i < nx; ++i) {
        solid[idxP(i, ny - 1)] = openTop ? 0 : 1;
    }

    invalidatePressureMatrix();
}

void MAC2D::enforceBoundaries() {
    applyBoundary();
}
