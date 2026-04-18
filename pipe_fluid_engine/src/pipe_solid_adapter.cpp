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
    // VoxelType enum: Air=0, Fluid=1, Solid=2.
    // Only the pipe-wall material (Solid) becomes a sim wall.  The Air region
    // outside the pipe is left as non-solid so the pressure-projection solver
    // can route fluid into and out of the pipe ends (open-boundary behaviour).
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = (vg.cells[i] == VoxelType::Solid) ? 1u : 0u;
    }
}

void applySolidsToSmoke(MACSmoke3D& smoke, const std::vector<uint8_t>& mask) {
    smoke.setVoxelSolids(mask);
}

void applySolidsToWater(MACWater3D& water, const std::vector<uint8_t>& mask) {
    water.setVoxelSolids(mask);
}

} // namespace pipe_fluid
