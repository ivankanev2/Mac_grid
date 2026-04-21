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

struct PipeBoundaryField {
    int nx = 0;
    int ny = 0;
    int nz = 0;
    float dx = 0.01f;
    Vec3 origin{0.f, 0.f, 0.f};

    // Canonical per-cell classification.
    std::vector<PipeBoundaryCell> cells;
    // Signed distance to the pipe wall region, in world metres.
    // Negative inside wall material, positive elsewhere (both Interior AND
    // Exterior).  This is the field fed to solidFractionOnEdge / face-open
    // computation so that Exterior↔Exterior faces remain fully open and the
    // MAC pressure solver can run normally outside the pipe shell.
    std::vector<float> wallSdf;
    // Patch F: second signed-distance field used exclusively by the
    // particle confinement code.  Negative inside Wall AND Exterior cells
    // (only Interior / Opening positive), so grad(phi) points consistently
    // from outside-pipe toward inside-pipe on both sides of the wall.
    // Using a separate field here keeps the face-open logic unaffected.
    std::vector<float> interiorSdf;
    // Binary wall occupancy kept for current renderer / legacy solver paths.
    std::vector<uint8_t> wallMask;

    // Face-open fractions derived from wallSdf. These are the first
    // subcell / face-aware boundary quantities we can hand to future
    // embedded-boundary or cut-cell solvers.
    //   1.0 = fully open face
    //   0.0 = fully blocked by wall
    std::vector<float> uOpen; // (nx+1) * ny * nz, x-faces
    std::vector<float> vOpen; // nx * (ny+1) * nz, y-faces
    std::vector<float> wOpen; // nx * ny * (nz+1), z-faces

    std::vector<PipeBoundaryTerminal> terminals;

    int idx(int i, int j, int k) const { return i + nx * (j + ny * k); }
    int uIdx(int i, int j, int k) const { return i + (nx + 1) * (j + ny * k); }
    int vIdx(int i, int j, int k) const { return i + nx * (j + (ny + 1) * k); }
    int wIdx(int i, int j, int k) const { return i + nx * (j + ny * k); }
    bool valid() const {
        const std::size_t n = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz);
        const std::size_t nu = static_cast<std::size_t>(nx + 1) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz);
        const std::size_t nv = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny + 1) * static_cast<std::size_t>(nz);
        const std::size_t nw = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz + 1);
        return nx > 0 && ny > 0 && nz > 0 &&
               cells.size() == n && wallSdf.size() == n && interiorSdf.size() == n &&
               wallMask.size() == n &&
               uOpen.size() == nu && vOpen.size() == nv && wOpen.size() == nw;
    }
    Vec3 cellCenter(int i, int j, int k) const {
        return origin + Vec3{(i + 0.5f) * dx, (j + 0.5f) * dx, (k + 0.5f) * dx};
    }
};

PipeBoundaryField buildPipeBoundaryField(const PipeNetwork& network,
                                         const VoxelGrid& voxels);

} // namespace pipe_fluid
