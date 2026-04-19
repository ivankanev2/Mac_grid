#include "pipe_fluid/pipe_solid_adapter.h"

#include "voxelizer.h"       // pipe_engine/Voxelizer
#include "mac_smoke3d.h"     // smoke_engine/Sim
#include "mac_water3d.h"     // smoke_engine/Sim

#include <cstddef>

namespace pipe_fluid {

void voxelGridToSolidMask(const VoxelGrid& vg, std::vector<uint8_t>& out) {
    const std::size_t n =
        static_cast<std::size_t>(vg.nx) *
        static_cast<std::size_t>(vg.ny) *
        static_cast<std::size_t>(vg.nz);
    out.assign(n, 0);
    // VoxelType enum: Air=0, Fluid=1, Solid=2, Opening=3.
    // Only the pipe-wall material (Solid) becomes a sim wall for smoke.
    // Air (outside the pipe) and Opening (pipe mouths carved by the open-
    // ends post-pass) are both left as non-solid so the pressure-projection
    // solver can route smoke into and out of the pipe ends.
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = (vg.cells[i] == VoxelType::Solid) ? 1u : 0u;
    }
}

void voxelGridToWaterSolidMask(const VoxelGrid& vg, std::vector<uint8_t>& out) {
    const std::size_t n =
        static_cast<std::size_t>(vg.nx) *
        static_cast<std::size_t>(vg.ny) *
        static_cast<std::size_t>(vg.nz);
    out.assign(n, 0);
    // Water-specific mask.  Passable cells (0) are:
    //   - VoxelType::Fluid   : the pipe interior
    //   - VoxelType::Opening : the exit channel carved past each open end
    // Blocked cells (1) are:
    //   - VoxelType::Solid   : pipe wall
    //   - VoxelType::Air     : outside the pipe.  Marking Air as solid
    //     SEALS the pad so FLIP particles that slip through voxelizer
    //     gaps at bends can't free-fall through the pad onto the grid
    //     floor under gravity.  Because Opening is a distinct type, the
    //     water still has a legal way to leave the pipe through the mouth.
    for (std::size_t i = 0; i < n; ++i) {
        const VoxelType c = vg.cells[i];
        const bool passable = (c == VoxelType::Fluid) || (c == VoxelType::Opening);
        out[i] = passable ? 0u : 1u;
    }
}

void applySolidsToSmoke(MACSmoke3D& smoke, const std::vector<uint8_t>& mask) {
    smoke.setVoxelSolids(mask);
}

void applySolidsToWater(MACWater3D& water, const std::vector<uint8_t>& mask) {
    water.setVoxelSolids(mask);
}

} // namespace pipe_fluid
