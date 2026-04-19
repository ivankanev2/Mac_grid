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
    // Water-specific mask.  Only the pipe MATERIAL blocks the fluid:
    //   Passable (0): VoxelType::Fluid   — the pipe interior
    //                 VoxelType::Opening — exit/fall-through cells
    //                 VoxelType::Air     — everything outside the pipe
    //   Blocked  (1): VoxelType::Solid   — the pipe wall
    //
    // Previously Air was blocked to SEAL the domain so FLIP particles
    // couldn't leak through voxelizer gaps at bends and free-fall onto
    // the grid floor.  The side-effect was that once water passed the
    // pipe mouth it hit an invisible "Air-is-solid" wall and could only
    // continue along the narrow Opening fall-through cylinder carved by
    // the voxelizer — producing a sausage-shaped column of stuck water
    // past the exit instead of a gravity-driven splash.
    //
    // By making Air passable, water that exits the pipe enters open air
    // and falls under gravity to the grid floor (which is still sealed
    // by rebuildBorderSolids() so the domain stays closed).  Any minor
    // leakage at bend wall voxelization simply joins the splash pool
    // below the pipe and is visually consistent with real plumbing.
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
