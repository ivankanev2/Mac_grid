#include "mac_smoke_sim.h"
#include <algorithm>

void MAC2D::computeVorticity(std::vector<float>& outOmega) const {
    outOmega.assign(nx*ny, 0.0f);
    for (int j=0;j<ny;++j) for (int i=0;i<nx;++i) {
        if (isSolid(i,j)) { outOmega[idxP(i,j)] = 0.0f; continue; }

        int ip1 = std::min(i+1,nx-1);
        int jp1 = std::min(j+1,ny-1);

        float vL = 0.5f*(v[idxV(i,j)] + v[idxV(i,j+1)]);
        float vR = 0.5f*(v[idxV(ip1,j)] + v[idxV(ip1,j+1)]);
        float dv_dx = (vR - vL) / dx;

        float uB = 0.5f*(u[idxU(i,j)] + u[idxU(i+1,j)]);
        float uT = 0.5f*(u[idxU(i,jp1)] + u[idxU(i+1,jp1)]);
        float du_dy = (uT - uB) / dx;

        outOmega[idxP(i,j)] = dv_dx - du_dy;
    }
}
