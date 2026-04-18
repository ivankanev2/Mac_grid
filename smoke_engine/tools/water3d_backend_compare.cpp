#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "Sim/mac_water3d.h"

namespace {

struct CompareStats {
    double rms = 0.0;
    double linf = 0.0;
    double meanAbs = 0.0;
};

CompareStats compareField(const std::vector<float>& a, const std::vector<float>& b) {
    CompareStats out;
    if (a.size() != b.size() || a.empty()) return out;

    long double sumSq = 0.0;
    long double sumAbs = 0.0;
    double linf = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        const double ad = std::fabs(d);
        sumSq += static_cast<long double>(d) * static_cast<long double>(d);
        sumAbs += ad;
        linf = std::max(linf, ad);
    }

    out.rms = std::sqrt(static_cast<double>(sumSq / static_cast<long double>(a.size())));
    out.meanAbs = static_cast<double>(sumAbs / static_cast<long double>(a.size()));
    out.linf = linf;
    return out;
}

bool parseIntArg(char** begin, char** end, const char* name, int& out) {
    for (char** it = begin; it != end; ++it) {
        if (std::strcmp(*it, name) == 0 && (it + 1) != end) {
            out = std::max(1, std::atoi(*(it + 1)));
            return true;
        }
    }
    return false;
}

bool parseFloatArg(char** begin, char** end, const char* name, float& out) {
    for (char** it = begin; it != end; ++it) {
        if (std::strcmp(*it, name) == 0 && (it + 1) != end) {
            out = std::max(1.0e-6f, std::strtof(*(it + 1), nullptr));
            return true;
        }
    }
    return false;
}

void printUsage(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " [--nx N] [--ny N] [--nz N] [--steps N] [--dt DT] [--source-radius R]\n"
        << "\n"
        << "Runs the 3D water solver twice: once forced to CPU and once preferring CUDA.\n"
        << "If CUDA is unavailable or the project was built without SMOKE_ENABLE_CUDA, the tool\n"
        << "still runs the CPU reference path and reports that CUDA validation was skipped.\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && (std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0)) {
        printUsage(argv[0]);
        return 0;
    }

    int nx = 48;
    int ny = 48;
    int nz = 36;
    int steps = 20;
    float dt = 0.01f;
    float sourceRadius = 0.14f;

    parseIntArg(argv + 1, argv + argc, "--nx", nx);
    parseIntArg(argv + 1, argv + argc, "--ny", ny);
    parseIntArg(argv + 1, argv + argc, "--nz", nz);
    parseIntArg(argv + 1, argv + argc, "--steps", steps);
    parseFloatArg(argv + 1, argv + argc, "--dt", dt);
    parseFloatArg(argv + 1, argv + argc, "--source-radius", sourceRadius);

    const int maxDim = std::max({nx, ny, nz, 1});
    const float dx = 1.0f / static_cast<float>(maxDim);

    auto configure = [&](MACWater3D& sim) {
        MACWater3D::Params p = sim.params;
        p.particlesPerCell = 2;
        p.maxParticles = std::max(200000, nx * ny * nz * 4);
        p.pressureIters = 120;
        p.diffuseIters = 12;
        p.extrapolationIters = 8;
        p.maskDilations = 1;
        p.useAPIC = true;
        p.flipBlend = 0.05f;
        p.volumePreserveRhsMean = true;
        p.volumePreserveStrength = 0.05f;
        p.reseedRelaxIters = 2;
        p.reseedRelaxStrength = 0.45f;
        sim.setParams(p);
    };

    MACWater3D cpu(nx, ny, nz, dx, dt);
    cpu.setBackendPreference(MACWater3D::BackendPreference::CPU);
    configure(cpu);

    MACWater3D gpu(nx, ny, nz, dx, dt);
    gpu.setBackendPreference(MACWater3D::BackendPreference::CUDA);
    configure(gpu);

    const MACWater3D::Vec3 sourceCenter{0.5f, 0.32f, 0.5f};
    const MACWater3D::Vec3 sourceVelocity{0.0f, 0.0f, 0.0f};

    cpu.addWaterSourceSphere(sourceCenter, sourceRadius, sourceVelocity);
    gpu.addWaterSourceSphere(sourceCenter, sourceRadius, sourceVelocity);

    for (int step = 0; step < steps; ++step) {
        if (step < steps / 2) {
            const float jetY = 0.16f + 0.03f * static_cast<float>(step % 3);
            cpu.addWaterSourceSphere({0.5f, jetY, 0.5f}, sourceRadius * 0.35f, {0.0f, 0.4f, 0.0f});
            gpu.addWaterSourceSphere({0.5f, jetY, 0.5f}, sourceRadius * 0.35f, {0.0f, 0.4f, 0.0f});
        }

        cpu.step();
        gpu.step();
    }

    const bool cudaAvailable = gpu.isCudaAvailable();
    const bool cudaActive = gpu.isCudaEnabled();

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "grid=" << nx << "x" << ny << "x" << nz << " steps=" << steps << " dt=" << dt << "\n";
    std::cout << "cpu_backend=\"" << cpu.stats().backendName << "\"\n";
    std::cout << "gpu_backend=\"" << gpu.stats().backendName << "\"\n";
    std::cout << "cuda_available=" << (cudaAvailable ? "true" : "false") << "\n";
    std::cout << "cuda_active=" << (cudaActive ? "true" : "false") << "\n";
    std::cout << "cpu_step_ms=" << cpu.stats().lastStepMs << "\n";
    std::cout << "gpu_step_ms=" << gpu.stats().lastStepMs << "\n";
    std::cout << "cpu_pressure_ms=" << cpu.stats().pressureMs << " cpu_pressure_iters=" << cpu.stats().pressureIters << "\n";
    std::cout << "gpu_pressure_ms=" << gpu.stats().pressureMs << " gpu_pressure_iters=" << gpu.stats().pressureIters << "\n";
    std::cout << "cpu_particles=" << cpu.stats().particleCount << " gpu_particles=" << gpu.stats().particleCount << "\n";
    std::cout << "cpu_liquid_cells=" << cpu.stats().liquidCells << " gpu_liquid_cells=" << gpu.stats().liquidCells << "\n";

    if (!cudaActive) {
        std::cout << "note=\"CUDA path is unavailable in this build or on this machine; CPU baseline completed.\"\n";
        return 0;
    }

    const CompareStats waterCmp = compareField(cpu.water, gpu.water);
    const CompareStats pressureCmp = compareField(cpu.pressure, gpu.pressure);
    const CompareStats divergenceCmp = compareField(cpu.divergence, gpu.divergence);
    const CompareStats speedCmp = compareField(cpu.speed, gpu.speed);

    std::cout << "water_rms=" << waterCmp.rms << " water_linf=" << waterCmp.linf << " water_mean_abs=" << waterCmp.meanAbs << "\n";
    std::cout << "pressure_rms=" << pressureCmp.rms << " pressure_linf=" << pressureCmp.linf << " pressure_mean_abs=" << pressureCmp.meanAbs << "\n";
    std::cout << "divergence_rms=" << divergenceCmp.rms << " divergence_linf=" << divergenceCmp.linf << " divergence_mean_abs=" << divergenceCmp.meanAbs << "\n";
    std::cout << "speed_rms=" << speedCmp.rms << " speed_linf=" << speedCmp.linf << " speed_mean_abs=" << speedCmp.meanAbs << "\n";

    return 0;
}
