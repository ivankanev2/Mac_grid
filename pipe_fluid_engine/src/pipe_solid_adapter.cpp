#include "pipe_fluid/pipe_solid_adapter.h"

#include "pipe_fluid/pipe_boundary_field.h"

#include "mac_smoke3d.h"
#include "mac_water3d.h"

#include <cstddef>

namespace pipe_fluid {

void pipeBoundaryFieldToSolidMask(const PipeBoundaryField& field,
                                  std::vector<uint8_t>& out) {
    const std::size_t n =
        static_cast<std::size_t>(field.nx) *
        static_cast<std::size_t>(field.ny) *
        static_cast<std::size_t>(field.nz);
    out.assign(n, 0u);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = (field.wallMask[i] != 0u) ? 1u : 0u;
    }
}

void pipeBoundaryFieldToWaterSolidMask(const PipeBoundaryField& field,
                                       std::vector<uint8_t>& out) {
    const std::size_t n =
        static_cast<std::size_t>(field.nx) *
        static_cast<std::size_t>(field.ny) *
        static_cast<std::size_t>(field.nz);
    out.assign(n, 0u);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = (field.wallMask[i] != 0u) ? 1u : 0u;
    }
}

void applySolidsToSmoke(MACSmoke3D& smoke, const std::vector<uint8_t>& mask) {
    smoke.setVoxelSolids(mask);
}

void applySolidsToWater(MACWater3D& water, const std::vector<uint8_t>& mask) {
    water.setVoxelSolids(mask);
}

} // namespace pipe_fluid
