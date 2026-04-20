#pragma once

#include "mac_water3d.h"

namespace smoke {

struct PipeWater3DConfig {
    int particlesPerCell = 8;
    float flipBlend = 0.95f;
    int borderThickness = 1;
    bool openTop = false;
    bool useAPIC = true;
    int pressureSolverMode = (int)MACWater3D::PressureSolverMode::Multigrid;
    int pressureItersMin = 200;
    int pressureMGVCyclesMin = 50;
    int pressureMGCoarseItersMin = 40;
    int reseedRelaxItersMin = 2;
    float reseedRelaxStrength = 0.45f;
    bool volumePreserveRhsMean = true;
    float volumePreserveStrength = 0.05f;
    MACWater3D::BackendPreference backendPreference = MACWater3D::BackendPreference::Auto;
};

inline PipeWater3DConfig defaultPipeWater3DConfig() {
    return PipeWater3DConfig{};
}

inline MACWater3D::Params makePipeWater3DParams(const MACWater3D::Params& base,
                                                const PipeWater3DConfig& cfg = defaultPipeWater3DConfig()) {
    MACWater3D::Params params = base;
    params.particlesPerCell = (params.particlesPerCell < cfg.particlesPerCell)
        ? cfg.particlesPerCell : params.particlesPerCell;
    params.flipBlend = cfg.flipBlend;
    params.borderThickness = cfg.borderThickness;
    params.openTop = cfg.openTop;
    params.useAPIC = cfg.useAPIC;
    params.pressureSolverMode = cfg.pressureSolverMode;
    params.pressureIters = (params.pressureIters < cfg.pressureItersMin)
        ? cfg.pressureItersMin : params.pressureIters;
    params.pressureMGVCycles = (params.pressureMGVCycles < cfg.pressureMGVCyclesMin)
        ? cfg.pressureMGVCyclesMin : params.pressureMGVCycles;
    params.pressureMGCoarseIters = (params.pressureMGCoarseIters < cfg.pressureMGCoarseItersMin)
        ? cfg.pressureMGCoarseItersMin : params.pressureMGCoarseIters;
    params.reseedRelaxIters = (params.reseedRelaxIters < cfg.reseedRelaxItersMin)
        ? cfg.reseedRelaxItersMin : params.reseedRelaxIters;
    params.reseedRelaxStrength = cfg.reseedRelaxStrength;
    params.volumePreserveRhsMean = cfg.volumePreserveRhsMean;
    params.volumePreserveStrength = cfg.volumePreserveStrength;
    return params;
}

inline void applyPipeWater3DConfig(MACWater3D& sim,
                                   const PipeWater3DConfig& cfg = defaultPipeWater3DConfig()) {
    sim.setParams(makePipeWater3DParams(sim.params, cfg));
    sim.setBackendPreference(cfg.backendPreference);
}

} // namespace smoke
