#include "mac_smoke_sim.h"
#include <cmath>
#include <algorithm>

// helpers for the pipe polyline
float MAC2D::distPointToSegment(float px,float py, float ax,float ay, float bx,float by) {
    float abx = bx - ax, aby = by - ay;
    float apx = px - ax, apy = py - ay;
    float denom = abx*abx + aby*aby;
    float t = (denom > 1e-12f) ? (apx*abx + apy*aby) / denom : 0.0f;
    t = clampf(t, 0.0f, 1.0f);
    float cx = ax + t*abx;
    float cy = ay + t*aby;
    float dx = px - cx, dy = py - cy;
    return std::sqrt(dx*dx + dy*dy);
}

float MAC2D::distPointToPolyline(float px,float py) const {
    if (pipe.x.size() < 2) return 1e9f;
    float best = 1e9f;
    for (size_t k = 0; k + 1 < pipe.x.size(); ++k) {
        best = std::min(best,
            distPointToSegment(px, py, pipe.x[k], pipe.y[k], pipe.x[k+1], pipe.y[k+1]));
    }
    return best;
}

void MAC2D::clearPipe() {
    pipe.x.clear();
    pipe.y.clear();

    invalidatePressureMatrix();
}

void MAC2D::rebuildSolidsFromPipe(bool clearInterior) {
    // 1) reset solids to empty
    std::fill(solid.begin(), solid.end(), 0);

    // 2) re-apply outer walls
    for (int i = 0; i < nx; ++i) {
        solid[idxP(i, 0)] = 1;
        solid[idxP(i, ny - 1)] = openTop ? 0 : 1;
    }
    for (int j = 0; j < ny; ++j) { solid[idxP(0, j)] = 1; solid[idxP(nx - 1, j)] = 1; }

    // 3) keep valve opening carved in bottom wall
    for (int i = valveI0; i <= valveI1; ++i) solid[idxP(i, 0)] = 0;

    // 4) no pipe? done
    if (!pipe.enabled || pipe.x.size() < 2) return;

    // 5) build pipe walls from polyline distance field
    float R  = pipe.radius;
    float T  = pipe.wall;
    float Rin = R;
    float Rout = R + T;

    for (int j = 1; j < ny-1; ++j) {
        for (int i = 1; i < nx-1; ++i) {
            float cx = (i + 0.5f) / (float)nx; // normalized
            float cy = (j + 0.5f) / (float)ny;

            float d = distPointToPolyline(cx, cy);

            // interior of pipe
            if (d <= Rin) {
                if (clearInterior) {
                    smoke[idxP(i,j)] = 0.0f;
                }
                // keep it empty (fluid)
                continue;
            }

            // wall band
            if (d <= Rout) {
                solid[idxP(i,j)] = 1;
                smoke[idxP(i,j)] = 0.0f;
                temp[idxP(i,j)]  = 0.0f;
                age[idxP(i,j)]   = 0.0f;
            }
        }
    }

    invalidatePressureMatrix();
}
