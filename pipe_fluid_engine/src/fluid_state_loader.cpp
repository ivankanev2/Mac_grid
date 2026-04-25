#include "pipe_fluid/fluid_state_loader.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>

namespace pipe_fluid {
namespace {

constexpr std::uint32_t kFluidStateMagic   = 0x46535431u;  // 'FST1' little-endian
constexpr std::uint32_t kFluidStateVersion = 1u;

template <typename T>
bool readScalar(std::istream& in, T& out) {
    in.read(reinterpret_cast<char*>(&out), sizeof(T));
    return static_cast<bool>(in) && in.gcount() == static_cast<std::streamsize>(sizeof(T));
}

} // namespace

LoadFluidStateResult loadFluidStateFile(const std::string& path,
                                        CapturedFluidState& out) {
    LoadFluidStateResult res;
    out = CapturedFluidState{};

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        res.ok = false;
        res.error = "Cannot open file: " + path;
        return res;
    }

    std::uint32_t magic = 0, version = 0;
    std::uint32_t nx_u = 0, ny_u = 0, nz_u = 0;
    float dx = 0.0f, ox = 0.0f, oy = 0.0f, oz = 0.0f;
    std::uint32_t n_particles_u = 0;

    if (!readScalar(in, magic) || magic != kFluidStateMagic) {
        std::ostringstream s;
        s << "Bad magic 0x" << std::hex << magic
          << " (expected 0x" << kFluidStateMagic << ") in " << path;
        res.error = s.str();
        return res;
    }
    if (!readScalar(in, version) || version != kFluidStateVersion) {
        std::ostringstream s;
        s << "Unsupported fluid_state version " << version
          << " (this build understands only v" << kFluidStateVersion << ")";
        res.error = s.str();
        return res;
    }

    if (!readScalar(in, nx_u) || !readScalar(in, ny_u) || !readScalar(in, nz_u)) {
        res.error = "Truncated header (grid dimensions) in " + path;
        return res;
    }
    if (!readScalar(in, dx)) {
        res.error = "Truncated header (dx) in " + path;
        return res;
    }
    if (!readScalar(in, ox) || !readScalar(in, oy) || !readScalar(in, oz)) {
        res.error = "Truncated header (origin) in " + path;
        return res;
    }
    if (!readScalar(in, n_particles_u)) {
        res.error = "Truncated header (n_particles) in " + path;
        return res;
    }

    if (nx_u == 0 || ny_u == 0 || nz_u == 0) {
        res.error = "Invalid grid dimensions (zero side) in " + path;
        return res;
    }
    if (!(dx > 0.0f)) {
        res.error = "Invalid voxel size dx in " + path;
        return res;
    }

    const std::size_t n = static_cast<std::size_t>(n_particles_u);
    out.nx = static_cast<int>(nx_u);
    out.ny = static_cast<int>(ny_u);
    out.nz = static_cast<int>(nz_u);
    out.dx = dx;
    out.originX = ox;
    out.originY = oy;
    out.originZ = oz;
    out.nParticles = static_cast<int>(n);

    // Particle blob: N * 6 floats interleaved (px py pz vx vy vz).  Read it as
    // one contiguous chunk and split into positions / velocities.
    out.positions.resize(3 * n);
    out.velocities.resize(3 * n);
    if (n > 0) {
        std::vector<float> blob(6 * n);
        const std::streamsize blobBytes =
            static_cast<std::streamsize>(blob.size() * sizeof(float));
        in.read(reinterpret_cast<char*>(blob.data()), blobBytes);
        if (!in || in.gcount() != blobBytes) {
            std::ostringstream s;
            s << "Truncated particle body: expected " << blobBytes
              << " bytes for " << n << " particles, got " << in.gcount();
            res.error = s.str();
            return res;
        }
        for (std::size_t i = 0; i < n; ++i) {
            out.positions[3 * i + 0] = blob[6 * i + 0];
            out.positions[3 * i + 1] = blob[6 * i + 1];
            out.positions[3 * i + 2] = blob[6 * i + 2];
            out.velocities[3 * i + 0] = blob[6 * i + 3];
            out.velocities[3 * i + 1] = blob[6 * i + 4];
            out.velocities[3 * i + 2] = blob[6 * i + 5];
        }
    }

    if (!out.valid()) {
        res.error = "Internal post-load validation failed for " + path;
        return res;
    }

    res.ok = true;
    return res;
}

} // namespace pipe_fluid
