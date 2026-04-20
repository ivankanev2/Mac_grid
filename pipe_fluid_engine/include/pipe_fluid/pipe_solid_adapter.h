#pragma once
// ============================================================================
// pipe_solid_adapter
//
// Translates the canonical PipeBoundaryField into the uint8_t (0=fluid,
// !=0=solid) masks expected by MACSmoke3D::setVoxelSolids and
// MACWater3D::setVoxelSolids.
//
// Axis convention matches on both sides: idx = i + nx*(j + ny*k), so no
// resampling or transposition is needed as long as the fluid grid was
// constructed with the same (nx, ny, nz) dimensions as the PipeBoundaryField.
// ============================================================================

#include <cstdint>
#include <vector>

struct MACSmoke3D;
struct MACWater3D;

namespace pipe_fluid {

struct PipeBoundaryField;

// Convert the canonical pipe boundary field into a simulator-ready uint8_t
// solid mask. Only wall cells become solid; interior, opening, and exterior
// cells remain passable.
void pipeBoundaryFieldToSolidMask(const PipeBoundaryField& field,
                                  std::vector<uint8_t>& out);

// Water-oriented mask derived from the canonical boundary field.
//
// Passable (0): Interior, Opening, Exterior
// Blocked  (1): Wall
//
// This is still a walls-only mask. MACWater3D later seals the outermost domain
// border internally via rebuildBorderSolids().
void pipeBoundaryFieldToWaterSolidMask(const PipeBoundaryField& field,
                                       std::vector<uint8_t>& out);

// Push the mask into each fluid simulator.
// The caller is responsible for ensuring each sim has matching (nx,ny,nz).
void applySolidsToSmoke(MACSmoke3D& smoke, const std::vector<uint8_t>& mask);
void applySolidsToWater(MACWater3D& water, const std::vector<uint8_t>& mask);

} // namespace pipe_fluid
