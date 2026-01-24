#include "mac_smoke_sim.h"
#include <cmath>
#include <limits>

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

void MAC2D::advectScalar(std::vector<float>& phi,
                         std::vector<float>& phi0,
                         float dissipation)
{
    if (useMacCormack) advectScalarMacCormack(phi, phi0, dissipation);
    else               advectScalarSemiLagrangian(phi, phi0, dissipation);
}

void MAC2D::advectScalarMacCormack(std::vector<float>& phi,
                                   std::vector<float>& phi0,
                                   float dissipation)
{
    // store source
    phi0 = phi;

    std::vector<float> phiFwd(phi.size(), 0.0f);
    std::vector<float> phiBack(phi.size(), 0.0f);

    auto stencilMinMax = [&](const std::vector<float>& f, float x, float y, float& outMin, float& outMax) {
        float fx = x / dx - 0.5f;
        float fy = y / dx - 0.5f;
        int i0 = (int)std::floor(fx);
        int j0 = (int)std::floor(fy);
        i0 = (int)clampf((float)i0, 0.0f, (float)(nx - 1));
        j0 = (int)clampf((float)j0, 0.0f, (float)(ny - 1));
        int i1 = std::min(i0 + 1, nx - 1);
        int j1 = std::min(j0 + 1, ny - 1);

        outMin =  std::numeric_limits<float>::infinity();
        outMax = -std::numeric_limits<float>::infinity();

        auto consider = [&](int ii, int jj) {
            if (isSolid(ii, jj)) return;
            float v = f[idxP(ii, jj)];
            outMin = std::min(outMin, v);
            outMax = std::max(outMax, v);
        };

        consider(i0, j0); consider(i1, j0);
        consider(i0, j1); consider(i1, j1);

        if (!std::isfinite(outMin)) { outMin = 0.0f; outMax = 0.0f; }
    };

    // forward pass
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int id = idxP(i,j);
            if (isSolid(i,j)) { phiFwd[id] = 0.0f; continue; }

            float x = (i + 0.5f) * dx;
            float y = (j + 0.5f) * dx;
            float ux, vy;
            velAt(x, y, u, v, ux, vy);

            float x0 = clampf(x - dt * ux, 0.0f, nx * dx);
            float y0 = clampf(y - dt * vy, 0.0f, ny * dx);

            int si, sj; worldToCell(x0, y0, si, sj);
            phiFwd[id] = (!isSolid(si, sj)) ? sampleCellCentered(phi0, x0, y0) : 0.0f;
        }
    }

    // backward pass
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int id = idxP(i,j);
            if (isSolid(i,j)) { phiBack[id] = 0.0f; continue; }

            float x = (i + 0.5f) * dx;
            float y = (j + 0.5f) * dx;
            float ux, vy;
            velAt(x, y, u, v, ux, vy);

            float x1 = clampf(x + dt * ux, 0.0f, nx * dx);
            float y1 = clampf(y + dt * vy, 0.0f, ny * dx);

            int si, sj; worldToCell(x1, y1, si, sj);
            phiBack[id] = (!isSolid(si, sj)) ? sampleCellCentered(phiFwd, x1, y1) : 0.0f;
        }
    }

    // correction + clamp
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int id = idxP(i,j);
            if (isSolid(i,j)) { phi[id] = 0.0f; continue; }

            float x = (i + 0.5f) * dx;
            float y = (j + 0.5f) * dx;
            float ux, vy;
            velAt(x, y, u, v, ux, vy);

            float x0 = clampf(x - dt * ux, 0.0f, nx * dx);
            float y0 = clampf(y - dt * vy, 0.0f, ny * dx);

            float corrected = phiFwd[id] + 0.5f * (phi0[id] - phiBack[id]);

            float mn, mx;
            stencilMinMax(phi0, x0, y0, mn, mx);
            corrected = clampf(corrected, mn, mx);

            phi[id] = dissipation * corrected;
        }
    }
}

// used to compare MacCormack vs Semi-Lagrangian
void MAC2D::advectScalarSemiLagrangian(std::vector<float>& phi,
                                       std::vector<float>& phi0,
                                       float dissipation)
{
    phi0 = phi;
    std::vector<float> tmp(phi.size(), 0.0f);
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int id = idxP(i,j);
            if (isSolid(i,j)) { tmp[id] = 0.0f; continue; }

            float x = (i + 0.5f) * dx;
            float y = (j + 0.5f) * dx;
            float ux, vy;
            velAt(x, y, u, v, ux, vy);

            float x0 = clampf(x - dt * ux, 0.0f, nx * dx);
            float y0 = clampf(y - dt * vy, 0.0f, ny * dx);

            int si, sj;
            worldToCell(x0, y0, si, sj);
            tmp[id] = (!isSolid(si, sj)) ? sampleCellCentered(phi0, x0, y0) : 0.0f;
            tmp[id] *= dissipation;
        }
    }
    phi.swap(tmp);
}

// L2 error (sqrt(sum((a-b)^2)))
float MAC2D::smokeL2Diff(const std::vector<float>& a, const std::vector<float>& b) const {
    double s = 0.0;
    size_t N = a.size();
    for (size_t k = 0; k < N; ++k) {
        double d = (double)a[k] - (double)b[k];
        s += d * d;
    }
    return (float)std::sqrt(s / (double)N);
}

float MAC2D::compareAdvectors(float dissipation)
{


    // Copy current smoke and the velocity fields used by the advector (phi0 is overwritten by advectors)
    std::vector<float> smoke_before = smoke;
    std::vector<float> phi0_copy    = smoke0; // advector functions expect phi0 param but overwrite it, so preserve shape

    // 1) MacCormack result
    std::vector<float> mac = smoke_before;
    std::vector<float> mac_phi0 = phi0_copy;

    // advectScalarMacCormack is defined in this translation unit; call with copies
    advectScalarMacCormack(mac, mac_phi0, dissipation);

    // 2) Semi-Lagrangian result
    std::vector<float> sl = smoke_before;
    std::vector<float> sl_phi0 = phi0_copy;
    advectScalarSemiLagrangian(sl, sl_phi0, dissipation);

    // Compute L2 difference
    float l2 = smokeL2Diff(mac, sl);
    return l2;
}

void MAC2D::advectSmoke(float dissipation /*= 0.995f*/) {
    if (useMacCormack) {
        std::vector<float> before = smoke;
        advectScalarMacCormack(smoke, smoke0, dissipation);

        // optional debug compile-time output
        #ifdef SIM_DEBUG
        std::vector<float> sl = before;
        advectScalarSemiLagrangian(sl, smoke0, dissipation);
        float l2 = smokeL2Diff(smoke, sl);
        std::cout << "advect: L2(MacCormack vs SL) = " << l2 << "\n";
        #endif
    } else {
        advectScalarSemiLagrangian(smoke, smoke0, dissipation);
    }
}