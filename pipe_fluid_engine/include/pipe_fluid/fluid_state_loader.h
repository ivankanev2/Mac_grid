#pragma once
// ============================================================================
// fluid_state_loader.h
//
// Reader for the binary "captured fluid state" format produced by the Python
// pipeline at gaussian_splatting/fluid_capture/bridge_to_sim.py.  The format is
// the C++ mirror of pipe_fluid/fluid_capture/pipeline/state_writer.py:
//
//   [uint32  magic       = 0x46535431  // 'FST1' little-endian]
//   [uint32  version     = 1]
//   [uint32  nx]
//   [uint32  ny]
//   [uint32  nz]
//   [float32 dx]
//   [float32 origin_x]
//   [float32 origin_y]
//   [float32 origin_z]
//   [uint32  n_particles]
//   [for i in 0..n_particles:
//       float32 px, py, pz, vx, vy, vz   // 24 bytes per particle
//   ]
//
// All values are little-endian; the host is assumed to be little-endian.
// ============================================================================

#include <cstdint>
#include <string>
#include <vector>

namespace pipe_fluid {

struct CapturedFluidState {
    // Grid parameters in the simulator's world frame.
    int   nx = 0;
    int   ny = 0;
    int   nz = 0;
    float dx = 0.0f;
    float originX = 0.0f;
    float originY = 0.0f;
    float originZ = 0.0f;

    // Captured FLIP particles, world-frame metres and metres-per-second.
    // Aligned: row i of `positions` corresponds to row i of `velocities`.
    std::vector<float> positions;   // size = 3 * nParticles, layout (x,y,z)
    std::vector<float> velocities;  // size = 3 * nParticles, layout (vx,vy,vz)
    int                nParticles = 0;

    [[nodiscard]] bool valid() const noexcept {
        return nx > 0 && ny > 0 && nz > 0 && dx > 0.0f
            && nParticles >= 0
            && positions.size() == static_cast<std::size_t>(3 * nParticles)
            && velocities.size() == static_cast<std::size_t>(3 * nParticles);
    }
};

struct LoadFluidStateResult {
    bool ok = false;
    std::string error;
};

// Load a fluid state from disk into ``out``.  On success, returns ok=true and
// ``out`` contains the parsed grid + particles.  On failure, returns ok=false
// with a human-readable error string and ``out`` is left in a partially-filled
// (but valid()-rejecting) state.
LoadFluidStateResult loadFluidStateFile(const std::string& path,
                                        CapturedFluidState& out);

} // namespace pipe_fluid
