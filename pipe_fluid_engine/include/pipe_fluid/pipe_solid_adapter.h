#pragma once
// ============================================================================
// pipe_solid_adapter
//
// Translates a pipe_engine VoxelGrid (Air=0, Fluid=1, Solid=2) into the
// uint8_t (0=fluid, !=0=solid) mask expected by MACSmoke3D::setVoxelSolids
// and MACWater3D::setVoxelSolids.
//
// Axis convention matches on both sides: idx = i + nx*(j + ny*k), so no
// resampling or transposition is needed as long as the fluid grid was
// constructed with the same (nx, ny, nz) dimensions as the VoxelGrid.
// ============================================================================

#include <cstdint>
#include <vector>

struct VoxelGrid;        // Voxelizer/voxelizer.h
struct MACSmoke3D;       // Sim/mac_smoke3d.h
struct MACWater3D;       // Sim/mac_water3d.h

namespace pipe_fluid {

// Convert a pipe VoxelGrid into a simulator-ready uint8_t solid mask.
// `out` is resized to nx*ny*nz. VoxelType::Solid -> 1, everything else -> 0.
//
// This "wall-only" mask is appropriate for SMOKE, where the open Air cells
// outside the pipe act as a sink so smoke can vent past the pipe ends
// without piling up at the grid boundary.
void voxelGridToSolidMask(const VoxelGrid& vg, std::vector<uint8_t>& out);

// Water-oriented mask used by the current pipe-fluid integration.
//
// Passable (0): VoxelType::Fluid, VoxelType::Opening, VoxelType::Air
// Blocked  (1): VoxelType::Solid
//
// In other words, this is also a walls-only mask.  Water is allowed to exit
// into Opening/Air cells, while MACWater3D later seals the outermost domain
// border internally via rebuildBorderSolids().
void voxelGridToWaterSolidMask(const VoxelGrid& vg, std::vector<uint8_t>& out);

// Push the mask into each fluid simulator.
// The caller is responsible for ensuring each sim has matching (nx,ny,nz).
void applySolidsToSmoke(MACSmoke3D& smoke, const std::vector<uint8_t>& mask);
void applySolidsToWater(MACWater3D& water, const std::vector<uint8_t>& mask);

} // namespace pipe_fluid
