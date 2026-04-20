#include "pipe_fluid/pipe_solver_boundary_data.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace pipe_fluid {
namespace {

inline int clampi(int v, int lo, int hi) {
    return std::max(lo, std::min(hi, v));
}

inline float sampleWallSdf(const PipeBoundaryField& field, int i, int j, int k) {
    i = clampi(i, 0, field.nx - 1);
    j = clampi(j, 0, field.ny - 1);
    k = clampi(k, 0, field.nz - 1);
    return field.wallSdf[static_cast<std::size_t>(field.idx(i, j, k))];
}

inline void computeWallNormal(const PipeBoundaryField& field,
                              int i, int j, int k,
                              float& nx, float& ny, float& nz) {
    const float dx = std::max(field.dx, 1e-6f);
    const float sx0 = sampleWallSdf(field, i - 1, j, k);
    const float sx1 = sampleWallSdf(field, i + 1, j, k);
    const float sy0 = sampleWallSdf(field, i, j - 1, k);
    const float sy1 = sampleWallSdf(field, i, j + 1, k);
    const float sz0 = sampleWallSdf(field, i, j, k - 1);
    const float sz1 = sampleWallSdf(field, i, j, k + 1);

    nx = (sx1 - sx0) / (2.0f * dx);
    ny = (sy1 - sy0) / (2.0f * dx);
    nz = (sz1 - sz0) / (2.0f * dx);

    const float len = std::sqrt(nx * nx + ny * ny + nz * nz);
    if (len > 1e-8f) {
        nx /= len;
        ny /= len;
        nz /= len;
    } else {
        nx = 0.0f;
        ny = 0.0f;
        nz = 0.0f;
    }
}

} // namespace

PipeSolverBoundaryData buildSolverBoundaryData(const PipeBoundaryField& field) {
    PipeSolverBoundaryData out;
    out.nx = field.nx;
    out.ny = field.ny;
    out.nz = field.nz;
    out.dx = field.dx;
    out.terminals = field.terminals;

    if (!field.valid()) {
        return out;
    }

    const std::size_t n = static_cast<std::size_t>(field.nx) * static_cast<std::size_t>(field.ny) * static_cast<std::size_t>(field.nz);
    out.solidMask.assign(n, uint8_t(0));
    out.waterSolidMask.assign(n, uint8_t(0));
    out.openingMask.assign(n, uint8_t(0));
    out.wallNx.assign(n, 0.0f);
    out.wallNy.assign(n, 0.0f);
    out.wallNz.assign(n, 0.0f);

    out.uOpen = field.uOpen;
    out.vOpen = field.vOpen;
    out.wOpen = field.wOpen;

    for (int k = 0; k < field.nz; ++k) {
        for (int j = 0; j < field.ny; ++j) {
            for (int i = 0; i < field.nx; ++i) {
                const std::size_t idx = static_cast<std::size_t>(field.idx(i, j, k));
                const PipeBoundaryCell c = field.cells[idx];
                const bool isWall = (c == PipeBoundaryCell::Wall);
                const bool isOpening = (c == PipeBoundaryCell::Opening);

                out.solidMask[idx] = isWall ? uint8_t(1) : uint8_t(0);
                out.waterSolidMask[idx] = isWall ? uint8_t(1) : uint8_t(0);
                out.openingMask[idx] = isOpening ? uint8_t(1) : uint8_t(0);

                computeWallNormal(field, i, j, k,
                                  out.wallNx[idx],
                                  out.wallNy[idx],
                                  out.wallNz[idx]);
            }
        }
    }

    return out;
}

} // namespace pipe_fluid