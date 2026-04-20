#pragma once
// ============================================================================
// pipe_solid_adapter
//
// Translates the canonical PipeBoundaryField into the uint8_t (0=fluid,
// !=0=solid) mask expected by MACSmoke3D::setVoxelSolids and
// MACWater3D::setVoxelSolids.
// ============================================================================

#include <cstdint>
#include <vector>

struct MACSmoke3D;
struct MACWater3D;

namespace pipe_fluid {

struct PipeBoundaryField;

void pipeBoundaryFieldToSolidMask(const PipeBoundaryField& field,
                                  std::vector<uint8_t>& out);
void pipeBoundaryFieldToWaterSolidMask(const PipeBoundaryField& field,
                                       std::vector<uint8_t>& out);
void applySolidsToSmoke(MACSmoke3D& smoke, const std::vector<uint8_t>& mask);
void applySolidsToWater(MACWater3D& water, const std::vector<uint8_t>& mask);

} // namespace pipe_fluid
