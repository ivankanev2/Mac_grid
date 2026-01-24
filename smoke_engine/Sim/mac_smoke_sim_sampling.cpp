#include "mac_smoke_sim.h"
#include <cmath>

float MAC2D::maxAbsDiv() const {
    float m = 0.0f;
    for (float d : div) m = std::max(m, std::abs(d));
    return m;
}

void MAC2D::worldToCell(float x, float y, int &i, int &j) const {
    // cell centers at (i+0.5)*dx
    float fx = x / dx - 0.5f;
    float fy = y / dx - 0.5f;
    i = (int)std::floor(fx);
    j = (int)std::floor(fy);
    i = (int)clampf((float)i, 0.0f, (float)(nx - 1));
    j = (int)clampf((float)j, 0.0f, (float)(ny - 1));
}

float MAC2D::sampleCellCentered(const std::vector<float>& f, float x, float y) const {
    float fx = x / dx - 0.5f;
    float fy = y / dx - 0.5f;

    int i0 = (int)std::floor(fx);
    int j0 = (int)std::floor(fy);
    float tx = fx - i0;
    float ty = fy - j0;

    i0 = (int)clampf((float)i0, 0.0f, (float)(nx - 1));
    j0 = (int)clampf((float)j0, 0.0f, (float)(ny - 1));
    int i1 = std::min(i0 + 1, nx - 1);
    int j1 = std::min(j0 + 1, ny - 1);

    float a = f[idxP(i0, j0)];
    float b = f[idxP(i1, j0)];
    float c = f[idxP(i0, j1)];
    float d = f[idxP(i1, j1)];

    float ab = a * (1 - tx) + b * tx;
    float cd = c * (1 - tx) + d * tx;
    return ab * (1 - ty) + cd * ty;
}

float MAC2D::sampleU(const std::vector<float>& fu, float x, float y) const {
    // u at (i*dx, (j+0.5)*dx)
    float fx = x / dx;
    float fy = y / dx - 0.5f;

    int i0 = (int)std::floor(fx);
    int j0 = (int)std::floor(fy);
    float tx = fx - i0;
    float ty = fy - j0;

    i0 = (int)clampf((float)i0, 0.0f, (float)(nx));
    j0 = (int)clampf((float)j0, 0.0f, (float)(ny - 1));
    int i1 = std::min(i0 + 1, nx);
    int j1 = std::min(j0 + 1, ny - 1);

    float a = fu[idxU(i0, j0)];
    float b = fu[idxU(i1, j0)];
    float c = fu[idxU(i0, j1)];
    float d = fu[idxU(i1, j1)];

    float ab = a * (1 - tx) + b * tx;
    float cd = c * (1 - tx) + d * tx;
    return ab * (1 - ty) + cd * ty;
}

float MAC2D::sampleV(const std::vector<float>& fv, float x, float y) const {
    // v at ((i+0.5)*dx, j*dx)
    float fx = x / dx - 0.5f;
    float fy = y / dx;

    int i0 = (int)std::floor(fx);
    int j0 = (int)std::floor(fy);
    float tx = fx - i0;
    float ty = fy - j0;

    i0 = (int)clampf((float)i0, 0.0f, (float)(nx - 1));
    j0 = (int)clampf((float)j0, 0.0f, (float)(ny));
    int i1 = std::min(i0 + 1, nx - 1);
    int j1 = std::min(j0 + 1, ny);

    float a = fv[idxV(i0, j0)];
    float b = fv[idxV(i1, j0)];
    float c = fv[idxV(i0, j1)];
    float d = fv[idxV(i1, j1)];

    float ab = a * (1 - tx) + b * tx;
    float cd = c * (1 - tx) + d * tx;
    return ab * (1 - ty) + cd * ty;
}

void MAC2D::velAt(float x, float y,
                  const std::vector<float>& fu,
                  const std::vector<float>& fv,
                  float& outUx, float& outVy) const
{
    outUx = sampleU(fu, x, y);
    outVy = sampleV(fv, x, y);
}