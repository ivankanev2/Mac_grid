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

// ---- Time series support (Phase C2) -----------------------------------------
//
// A captured-fluid time series is a folder of sim_state_NNNN.bin files (one
// per video frame) plus a manifest.json with metadata.  Produced by
// gaussian_splatting/dynamic_capture/extract_fluid_state.py.
//
// Each frame is one full CapturedFluidState.  All frames in a series share
// the same grid parameters (nx, ny, nz, dx, origin) — only the particle
// content varies between frames.  The simulator's replay-mode loader picks
// up the grid from frame 0 and then advances through frames over time,
// overwriting water particles on each frame boundary.
struct CapturedFluidStateSeries {
    std::vector<CapturedFluidState> frames;
    float capturedFps = 25.0f;     // playback rate in captured frames per sim second
    std::string manifestPath;      // resolved path to manifest.json (informational)

    [[nodiscard]] bool valid() const noexcept {
        if (frames.empty() || capturedFps <= 0.0f) return false;
        for (const auto& f : frames) if (!f.valid()) return false;
        return true;
    }

    [[nodiscard]] int nFrames() const noexcept {
        return static_cast<int>(frames.size());
    }

    [[nodiscard]] float capturedDt() const noexcept {
        return capturedFps > 0.0f ? 1.0f / capturedFps : 0.04f;
    }

    [[nodiscard]] float totalDuration() const noexcept {
        return capturedDt() * static_cast<float>(nFrames());
    }
};

// Load all sim_state_*.bin files from a folder into ``out``, sorted lexically
// by filename (which matches frame order because the writer pads with zeros).
// If a manifest.json is present in the folder it is consulted for capturedFps.
LoadFluidStateResult loadFluidStateSeriesFolder(
    const std::string& folder_path,
    CapturedFluidStateSeries& out);

} // namespace pipe_fluid
