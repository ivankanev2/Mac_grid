// ============================================================================
// volume_renderer.cpp — factory that constructs either the CPU or GPU backend.
//
// Each backend is a self-contained translation unit with its own `make...`
// entry point. This file just picks between them (with safe fallback) so
// consumers never include the backend headers directly.
// ============================================================================

#include "pipe_fluid/volume_renderer.h"

namespace pipe_fluid {

// Forward declarations — defined in cpu_volume_renderer.cpp and
// gpu_volume_renderer.cpp respectively.
std::unique_ptr<VolumeOverlayRenderer> makeCpuVolumeRenderer();
std::unique_ptr<VolumeOverlayRenderer> makeGpuVolumeRenderer();

std::unique_ptr<VolumeOverlayRenderer>
makeVolumeRenderer(VolumeOverlayRenderer::Backend requested) {
    if (requested == VolumeOverlayRenderer::Backend::GPU) {
        auto gpu = makeGpuVolumeRenderer();
        if (gpu && gpu->init()) return gpu;
        // GPU init failed (e.g. no GL_TEXTURE_3D / R16F support). Fall back.
    }
    auto cpu = makeCpuVolumeRenderer();
    if (cpu && cpu->init()) return cpu;
    return nullptr;
}

} // namespace pipe_fluid
