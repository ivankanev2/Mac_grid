#pragma once

#include "vec3.h"

#include <cstdint>
#include <vector>

struct PipeNetwork;
struct VoxelGrid;

namespace pipe_fluid {

enum class PipeBoundaryCell : uint8_t {
    Exterior = 0,
    Interior = 1,
    Wall     = 2,
    Opening  = 3,
};

struct PipeBoundaryTerminal {
    Vec3  position;
    Vec3  outwardTangent;
    float innerRadius = 0.0f;
    float outerRadius = 0.0f;
};

// Canonical grid-aligned pipe boundary representation.
//
// This is the single boundary object that pipe_fluid_engine uses to derive:
//   - simulator solid masks
//   - renderer wall masks
//   - pipe-wall signed distance values for debugging / future cut-cell work
//   - open-end terminal metadata
//
// For now the field is built from the voxelizer output, but the rest of the
// integration layer no longer consumes the VoxelGrid directly when deciding
// what is solid or passable. That keeps the "source of truth" localized to one
// object and makes it straightforward to replace the construction step later
// with an analytic SDF or face-fraction pipeline.
struct PipeBoundaryField {
    int nx = 0;
    int ny = 0;
    int nz = 0;
    float dx = 0.01f;
    Vec3 origin{0.f, 0.f, 0.f};

    std::vector<PipeBoundaryCell> cells;   // authoritative semantic labels
    std::vector<float> wallSdf;            // metres; negative in wall cells
    std::vector<uint8_t> wallMask;         // 1 = wall, 0 = passable
    std::vector<PipeBoundaryTerminal> terminals;

    int idx(int i, int j, int k) const { return i + nx * (j + ny * k); }

    bool valid() const {
        const std::size_t n =
            static_cast<std::size_t>(nx) *
            static_cast<std::size_t>(ny) *
            static_cast<std::size_t>(nz);
        return nx > 0 && ny > 0 && nz > 0 &&
               cells.size() == n && wallSdf.size() == n && wallMask.size() == n;
    }

    Vec3 cellCenter(int i, int j, int k) const {
        return origin + Vec3{(i + 0.5f) * dx, (j + 0.5f) * dx, (k + 0.5f) * dx};
    }

    float wallSdfAt(int i, int j, int k) const {
        return wallSdf[idx(i, j, k)];
    }

    PipeBoundaryCell cellAt(int i, int j, int k) const {
        return cells[idx(i, j, k)];
    }
};

PipeBoundaryField buildPipeBoundaryField(const PipeNetwork& network,
                                         const VoxelGrid& voxels);

} // namespace pipe_fluid
