#include "pipe_fluid/pipe_boundary_field.h"

#include "pipe_network.h"
#include "voxelizer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace pipe_fluid {
namespace {

struct NeighborStep {
    int di, dj, dk;
    float cost;
};

constexpr float kInf = 1.0e30f;

inline bool inBounds(int i, int j, int k, int nx, int ny, int nz) {
    return i >= 0 && i < nx && j >= 0 && j < ny && k >= 0 && k < nz;
}

std::vector<NeighborStep> makeChamferMask(bool forward) {
    std::vector<NeighborStep> steps;
    steps.reserve(13);
    for (int dk = -1; dk <= 1; ++dk) {
        for (int dj = -1; dj <= 1; ++dj) {
            for (int di = -1; di <= 1; ++di) {
                if (di == 0 && dj == 0 && dk == 0) continue;
                const bool earlier =
                    (dk < 0) || (dk == 0 && dj < 0) || (dk == 0 && dj == 0 && di < 0);
                if (earlier != forward) continue;
                const float len = std::sqrt(float(di*di + dj*dj + dk*dk));
                steps.push_back({di, dj, dk, len});
            }
        }
    }
    return steps;
}

void chamferSweep(std::vector<float>& dist, int nx, int ny, int nz, float dx) {
    const auto forward = makeChamferMask(true);
    const auto backward = makeChamferMask(false);

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const int idx = i + nx * (j + ny * k);
                float best = dist[idx];
                for (const auto& s : forward) {
                    const int ni = i + s.di;
                    const int nj = j + s.dj;
                    const int nk = k + s.dk;
                    if (!inBounds(ni, nj, nk, nx, ny, nz)) continue;
                    const int nidx = ni + nx * (nj + ny * nk);
                    best = std::min(best, dist[nidx] + s.cost * dx);
                }
                dist[idx] = best;
            }
        }
    }

    for (int k = nz - 1; k >= 0; --k) {
        for (int j = ny - 1; j >= 0; --j) {
            for (int i = nx - 1; i >= 0; --i) {
                const int idx = i + nx * (j + ny * k);
                float best = dist[idx];
                for (const auto& s : backward) {
                    const int ni = i + s.di;
                    const int nj = j + s.dj;
                    const int nk = k + s.dk;
                    if (!inBounds(ni, nj, nk, nx, ny, nz)) continue;
                    const int nidx = ni + nx * (nj + ny * nk);
                    best = std::min(best, dist[nidx] + s.cost * dx);
                }
                dist[idx] = best;
            }
        }
    }
}

bool cellIsWall(PipeBoundaryCell c) {
    return c == PipeBoundaryCell::Wall;
}

bool hasOppositeNeighbor(const std::vector<PipeBoundaryCell>& cells,
                         int nx, int ny, int nz,
                         int i, int j, int k,
                         bool wantWall) {
    const int idx = i + nx * (j + ny * k);
    const bool self = cellIsWall(cells[idx]);
    if (self != wantWall) return false;
    for (int dk = -1; dk <= 1; ++dk) {
        for (int dj = -1; dj <= 1; ++dj) {
            for (int di = -1; di <= 1; ++di) {
                if (di == 0 && dj == 0 && dk == 0) continue;
                const int ni = i + di;
                const int nj = j + dj;
                const int nk = k + dk;
                if (!inBounds(ni, nj, nk, nx, ny, nz)) continue;
                const int nidx = ni + nx * (nj + ny * nk);
                if (cellIsWall(cells[nidx]) != wantWall) return true;
            }
        }
    }
    return false;
}

std::vector<float> buildSignedWallDistance(const std::vector<PipeBoundaryCell>& cells,
                                           int nx, int ny, int nz, float dx) {
    const std::size_t total =
        static_cast<std::size_t>(nx) *
        static_cast<std::size_t>(ny) *
        static_cast<std::size_t>(nz);

    std::vector<float> inside(total, kInf);
    std::vector<float> outside(total, kInf);

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const std::size_t idx = static_cast<std::size_t>(i) +
                    static_cast<std::size_t>(nx) * (static_cast<std::size_t>(j) +
                    static_cast<std::size_t>(ny) * static_cast<std::size_t>(k));
                const bool isWall = cellIsWall(cells[idx]);
                if (hasOppositeNeighbor(cells, nx, ny, nz, i, j, k, isWall)) {
                    if (isWall) inside[idx] = 0.5f * dx;
                    else        outside[idx] = 0.5f * dx;
                }
            }
        }
    }

    chamferSweep(inside, nx, ny, nz, dx);
    chamferSweep(outside, nx, ny, nz, dx);

    std::vector<float> sdf(total, 0.0f);
    for (std::size_t idx = 0; idx < total; ++idx) {
        const bool isWall = cellIsWall(cells[idx]);
        sdf[idx] = isWall ? -inside[idx] : outside[idx];
        if (!std::isfinite(sdf[idx])) sdf[idx] = isWall ? -0.5f * dx : 0.5f * dx;
    }
    return sdf;
}

} // namespace

PipeBoundaryField buildPipeBoundaryField(const PipeNetwork& network,
                                         const VoxelGrid& voxels) {
    PipeBoundaryField field;
    field.nx = voxels.nx;
    field.ny = voxels.ny;
    field.nz = voxels.nz;
    field.dx = voxels.dx;
    field.origin = voxels.origin;

    const std::size_t total =
        static_cast<std::size_t>(voxels.nx) *
        static_cast<std::size_t>(voxels.ny) *
        static_cast<std::size_t>(voxels.nz);
    field.cells.resize(total, PipeBoundaryCell::Exterior);
    field.wallMask.assign(total, 0u);

    for (std::size_t i = 0; i < total; ++i) {
        PipeBoundaryCell c = PipeBoundaryCell::Exterior;
        switch (voxels.cells[i]) {
            case VoxelType::Air:     c = PipeBoundaryCell::Exterior; break;
            case VoxelType::Fluid:   c = PipeBoundaryCell::Interior; break;
            case VoxelType::Solid:   c = PipeBoundaryCell::Wall;     break;
            case VoxelType::Opening: c = PipeBoundaryCell::Opening;  break;
        }
        field.cells[i] = c;
        field.wallMask[i] = (c == PipeBoundaryCell::Wall) ? 1u : 0u;
    }

    field.wallSdf = buildSignedWallDistance(field.cells, field.nx, field.ny, field.nz, field.dx);

    for (const auto& end : network.openEnds()) {
        field.terminals.push_back(PipeBoundaryTerminal{
            end.position,
            end.outwardTangent,
            end.innerRadius,
            end.outerRadius,
        });
    }

    return field;
}

} // namespace pipe_fluid
