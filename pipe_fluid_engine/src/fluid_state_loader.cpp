#include "pipe_fluid/fluid_state_loader.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
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

// ---- Time series loader (Phase C2) -----------------------------------------

namespace {

// Tiny manifest.json parser — we only need ``capturedFps`` if present.  We
// avoid pulling a full JSON dependency by scanning for a couple of specific
// keys (``"fps"``, ``"captured_fps"``).  Any other content in the manifest is
// ignored; absent keys fall back to the default 25 fps.
float parseCapturedFpsFromManifest(const std::string& path) {
    std::ifstream in(path);
    if (!in) return -1.0f;
    std::stringstream buf;
    buf << in.rdbuf();
    std::string content = buf.str();

    auto findFloatAfter = [&](const std::string& key) -> float {
        std::size_t pos = content.find(std::string("\"") + key + "\"");
        if (pos == std::string::npos) return -1.0f;
        pos = content.find(':', pos);
        if (pos == std::string::npos) return -1.0f;
        ++pos;
        while (pos < content.size() && std::isspace(static_cast<unsigned char>(content[pos]))) ++pos;
        std::size_t end = pos;
        while (end < content.size() &&
               (std::isdigit(static_cast<unsigned char>(content[end])) ||
                content[end] == '.' || content[end] == '-' || content[end] == '+' ||
                content[end] == 'e' || content[end] == 'E')) {
            ++end;
        }
        if (end == pos) return -1.0f;
        try {
            return std::stof(content.substr(pos, end - pos));
        } catch (...) {
            return -1.0f;
        }
    };

    // Keys we recognise, in priority order.
    for (const auto& key : {std::string("captured_fps"),
                            std::string("capturedFps"),
                            std::string("fps")}) {
        const float v = findFloatAfter(key);
        if (v > 0.0f && std::isfinite(v)) return v;
    }
    return -1.0f;
}

bool isSimStateFilename(const std::string& name) {
    // Match "sim_state_<digits>.bin"
    constexpr const char prefix[] = "sim_state_";
    constexpr const char suffix[] = ".bin";
    constexpr std::size_t prefLen = sizeof(prefix) - 1;
    constexpr std::size_t sufLen = sizeof(suffix) - 1;
    if (name.size() <= prefLen + sufLen) return false;
    if (name.compare(0, prefLen, prefix) != 0) return false;
    if (name.compare(name.size() - sufLen, sufLen, suffix) != 0) return false;
    for (std::size_t i = prefLen; i < name.size() - sufLen; ++i) {
        if (!std::isdigit(static_cast<unsigned char>(name[i]))) return false;
    }
    return true;
}

} // namespace

LoadFluidStateResult loadFluidStateSeriesFolder(
    const std::string& folder_path,
    CapturedFluidStateSeries& out) {
    LoadFluidStateResult res;
    out = CapturedFluidStateSeries{};

    namespace fs = std::filesystem;
    fs::path root(folder_path);
    if (!fs::exists(root)) {
        res.error = "Folder not found: " + folder_path;
        return res;
    }
    if (!fs::is_directory(root)) {
        res.error = "Not a folder: " + folder_path;
        return res;
    }

    // Enumerate matching files, sorted lexically (which matches frame order
    // because extract_fluid_state.py pads with zeros).
    std::vector<fs::path> simFiles;
    for (const auto& entry : fs::directory_iterator(root)) {
        if (!entry.is_regular_file()) continue;
        const auto name = entry.path().filename().string();
        if (isSimStateFilename(name)) {
            simFiles.push_back(entry.path());
        }
    }
    if (simFiles.empty()) {
        res.error = "No sim_state_*.bin files found in " + folder_path;
        return res;
    }
    std::sort(simFiles.begin(), simFiles.end(),
              [](const fs::path& a, const fs::path& b) {
                  return a.filename().string() < b.filename().string();
              });

    // Load every file.
    out.frames.reserve(simFiles.size());
    for (const auto& p : simFiles) {
        CapturedFluidState f;
        auto sub = loadFluidStateFile(p.string(), f);
        if (!sub.ok) {
            std::ostringstream s;
            s << "While reading " << p.filename().string() << ": " << sub.error;
            res.error = s.str();
            out = CapturedFluidStateSeries{};
            return res;
        }
        out.frames.push_back(std::move(f));
    }

    // Manifest (optional).
    fs::path manifest = root / "manifest.json";
    if (fs::exists(manifest) && fs::is_regular_file(manifest)) {
        out.manifestPath = manifest.string();
        const float fpsFromManifest = parseCapturedFpsFromManifest(manifest.string());
        if (fpsFromManifest > 0.0f) {
            out.capturedFps = fpsFromManifest;
        }
    }
    if (out.capturedFps <= 0.0f) out.capturedFps = 25.0f;

    if (!out.valid()) {
        res.error = "Loaded series failed validity check (mismatched grid?)";
        out = CapturedFluidStateSeries{};
        return res;
    }
    res.ok = true;
    return res;
}

// ---- Bottle solid loader (Phase C v2) ---------------------------------------

namespace {
constexpr std::uint32_t kBottleSolidMagic   = 0x42534C31u;  // 'BSL1'
constexpr std::uint32_t kBottleSolidVersion = 1u;
} // namespace

LoadFluidStateResult loadBottleSolidFile(const std::string& path,
                                         CapturedBottleSolid& out) {
    LoadFluidStateResult res;
    out = CapturedBottleSolid{};

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        res.error = "Cannot open bottle solid file: " + path;
        return res;
    }

    std::uint32_t magic = 0, version = 0;
    std::uint32_t nx_u = 0, ny_u = 0, nz_u = 0;
    float dx = 0.0f, ox = 0.0f, oy = 0.0f, oz = 0.0f;
    std::uint32_t nSolid_u = 0;

    if (!readScalar(in, magic) || magic != kBottleSolidMagic) {
        std::ostringstream s;
        s << "Bad bottle solid magic 0x" << std::hex << magic
          << " (expected 0x" << kBottleSolidMagic << ") in " << path;
        res.error = s.str();
        return res;
    }
    if (!readScalar(in, version) || version != kBottleSolidVersion) {
        std::ostringstream s;
        s << "Unsupported bottle_solid version " << version
          << " (this build understands only v" << kBottleSolidVersion << ")";
        res.error = s.str();
        return res;
    }
    if (!readScalar(in, nx_u) || !readScalar(in, ny_u) || !readScalar(in, nz_u)) {
        res.error = "Truncated bottle header (grid dims) in " + path;
        return res;
    }
    if (!readScalar(in, dx)) {
        res.error = "Truncated bottle header (dx) in " + path;
        return res;
    }
    if (!readScalar(in, ox) || !readScalar(in, oy) || !readScalar(in, oz)) {
        res.error = "Truncated bottle header (origin) in " + path;
        return res;
    }
    if (!readScalar(in, nSolid_u)) {
        res.error = "Truncated bottle header (n_solid) in " + path;
        return res;
    }

    if (nx_u == 0 || ny_u == 0 || nz_u == 0) {
        res.error = "Invalid bottle grid dimensions (zero side) in " + path;
        return res;
    }
    if (!(dx > 0.0f)) {
        res.error = "Invalid bottle voxel size dx in " + path;
        return res;
    }

    out.nx = static_cast<int>(nx_u);
    out.ny = static_cast<int>(ny_u);
    out.nz = static_cast<int>(nz_u);
    out.dx = dx;
    out.originX = ox;
    out.originY = oy;
    out.originZ = oz;
    out.nSolidCells = static_cast<int>(nSolid_u);

    const std::size_t total = static_cast<std::size_t>(nx_u) *
                              static_cast<std::size_t>(ny_u) *
                              static_cast<std::size_t>(nz_u);
    out.mask.resize(total);
    if (total > 0) {
        const std::streamsize bytes = static_cast<std::streamsize>(total);
        in.read(reinterpret_cast<char*>(out.mask.data()), bytes);
        if (!in || in.gcount() != bytes) {
            std::ostringstream s;
            s << "Truncated bottle mask body: expected " << bytes
              << " bytes for " << nx_u << "x" << ny_u << "x" << nz_u
              << " grid, got " << in.gcount();
            res.error = s.str();
            return res;
        }
    }

    if (!out.valid()) {
        res.error = "Internal post-load validation failed for " + path;
        return res;
    }
    res.ok = true;
    return res;
}

// ---- Column emitter trajectory loader (Tier B) -----------------------------

namespace {
constexpr std::uint32_t kColumnEmitterMagic   = 0x43454D31u;  // 'CEM1'
constexpr std::uint32_t kColumnEmitterVersion = 1u;
} // namespace

LoadFluidStateResult loadColumnEmitterFile(const std::string& path,
                                           CapturedColumnEmitter& out) {
    LoadFluidStateResult res;
    out = CapturedColumnEmitter{};

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        res.error = "Cannot open column emitter file: " + path;
        return res;
    }

    std::uint32_t magic = 0, version = 0, nFrames = 0;
    float capturedFps = 0.0f;

    if (!readScalar(in, magic) || magic != kColumnEmitterMagic) {
        std::ostringstream s;
        s << "Bad column emitter magic 0x" << std::hex << magic
          << " (expected 0x" << kColumnEmitterMagic << ") in " << path;
        res.error = s.str();
        return res;
    }
    if (!readScalar(in, version) || version != kColumnEmitterVersion) {
        std::ostringstream s;
        s << "Unsupported column_emitter version " << version
          << " (this build understands only v" << kColumnEmitterVersion << ")";
        res.error = s.str();
        return res;
    }
    if (!readScalar(in, nFrames)) {
        res.error = "Truncated header (n_frames) in " + path;
        return res;
    }
    if (!readScalar(in, capturedFps) || !(capturedFps > 0.0f)) {
        res.error = "Invalid captured_fps in " + path;
        return res;
    }

    out.capturedFps = capturedFps;
    out.frames.resize(nFrames);

    // Per-frame body: 1 byte active + 3 bytes pad + 8 floats = 36 bytes.
    constexpr std::streamsize kPerFrame = 1 + 3 + 8 * 4;
    for (std::uint32_t i = 0; i < nFrames; ++i) {
        std::uint8_t active = 0;
        char pad[3];
        float vals[8] = {};

        if (!readScalar(in, active)) {
            std::ostringstream s;
            s << "Truncated frame " << i << " (active byte) in " << path;
            res.error = s.str();
            return res;
        }
        in.read(pad, 3);
        if (!in || in.gcount() != 3) {
            std::ostringstream s;
            s << "Truncated frame " << i << " (padding) in " << path;
            res.error = s.str();
            return res;
        }
        in.read(reinterpret_cast<char*>(vals), sizeof(vals));
        if (!in || in.gcount() != static_cast<std::streamsize>(sizeof(vals))) {
            std::ostringstream s;
            s << "Truncated frame " << i << " (body, expected "
              << kPerFrame - 4 << " bytes after active+pad) in " << path;
            res.error = s.str();
            return res;
        }

        auto& f = out.frames[i];
        f.active = (active != 0);
        f.posX = vals[0]; f.posY = vals[1]; f.posZ = vals[2];
        f.velX = vals[3]; f.velY = vals[4]; f.velZ = vals[5];
        f.radius = vals[6];
        f.amount = vals[7];
    }

    if (!out.valid()) {
        res.error = "Internal post-load validation failed for " + path;
        return res;
    }
    res.ok = true;
    return res;
}

} // namespace pipe_fluid
