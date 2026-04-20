#pragma once
// ============================================================================
// VolumeOverlayRenderer — backend-independent smoke/water overlay.
//
// This replaces the old full-screen-blit overlay in main_gui.cpp, which
// rendered the smoke in its own local unit-box and was therefore spatially
// disconnected from the pipe mesh. Both implementations here ray-march the
// voxel grid's real world-space AABB using the same projection/view matrices
// as the pipe mesh, so the smoke and pipe share one world.
//
// Two backends:
//   - CPU (cpu_volume_renderer.cpp): multi-threaded raymarch into an RGBA
//     texture; then a fullscreen textured quad is blitted with alpha blend.
//     Preferred on Apple Silicon, integrated GPUs, and any case where
//     GL_TEXTURE_3D support is shaky.
//
//   - GPU (gpu_volume_renderer.cpp): 3D texture upload + fragment-shader
//     raymarch on a fullscreen quad. Preferred on discrete GPUs.
//
// Both backends:
//   * accept the volume in simulator order (`i + nx*(j + ny*k)`);
//   * accept the world-space origin (voxels.origin) and the cell size
//     (voxels.dx) so the AABB is expressed in metres;
//   * accept the full OrbitCamera state (proj, view, camPos) via VolumeView;
//   * composite as pre-multiplied alpha over the currently-bound framebuffer.
//
// Depth-buffer-based occlusion (smoke correctly hidden behind pipe walls)
// is not yet implemented; it requires an FBO-first refactor of the main
// render loop. The visual consequence is that smoke can "show through" the
// pipe when the pipe is between camera and smoke. The sim-space placement
// is fully correct regardless.
// ============================================================================

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace pipe_fluid {

// ---- Per-frame view description --------------------------------------------
struct VolumeView {
    // World-space AABB of the voxel grid.
    //   min corner = (originX, originY, originZ)
    //   max corner = origin + (nx*dx, ny*dx, nz*dx)
    float originX = 0.f, originY = 0.f, originZ = 0.f;
    float dx      = 0.01f;
    int   nx = 0, ny = 0, nz = 0;

    // Camera matrices (column-major, OpenGL convention).
    // These are exactly what pipe_engine's OrbitCamera produces.
    float proj[16] = {};
    float view[16] = {};
    float camPosX = 0.f, camPosY = 0.f, camPosZ = 0.f;

    // Target framebuffer size in pixels.
    int fbWidth  = 0;
    int fbHeight = 0;
};

// ---- Appearance controls ----------------------------------------------------
struct VolumeSettings {
    bool  useColor     = false;   // monochrome vs. temperature-tinted
    float alphaScale   = 1.0f;    // global opacity multiplier
    float densityScale = 3.0f;    // sigma multiplier in the optical model
    float tempStrength = 0.75f;   // how much temp tints the color
    float coreDark     = 0.75f;   // darkens dense cores in color mode

    // When true, the water path sphere-traces the SDF uploaded via
    // setWaterSdf() instead of volume-integrating the density texture.
    // Only has effect for backends that implement setWaterSdf().
    bool  useSdf       = true;

    // CPU-only knobs (GPU uses fixed values tuned to match CPU output).
    int   stepsPerPixel = 96;     // upper bound on ray-march steps
    int   renderScale   = 1;      // 1=full-res, 2=half-res (interactive Mac)
};

// ---- Abstract interface -----------------------------------------------------
class VolumeOverlayRenderer {
public:
    enum class Backend { CPU, GPU };

    virtual ~VolumeOverlayRenderer() = default;

    // Initialize GPU objects (textures, shaders, VAOs). Must be called with
    // a current OpenGL context. Returns false on failure.
    virtual bool init() = 0;

    // Upload the density + temperature + solid fields. Called whenever the
    // sim has advanced. Dimensions must match VolumeView::nx/ny/nz.
    virtual void setVolume(const std::vector<float>&   density,
                           const std::vector<float>&   temp,
                           const std::vector<uint8_t>& solid,
                           int nx, int ny, int nz) = 0;

    // Upload a narrow-band SDF (values in world metres) to be sphere-traced
    // as the water surface.  When VolumeSettings::useSdf is true and the
    // renderer supports SDF, the water path sphere-traces this field instead
    // of volume-integrating the density texture.  `band` is the positive
    // clamp value used when building the SDF — cells saturated to +band
    // are treated as "far from water" by the tracer so it can take big
    // empty steps.  Pass an empty vector to disable SDF mode.
    //
    // Default implementation is a no-op so existing backends keep working
    // until they opt in.
    virtual void setWaterSdf(const std::vector<float>& sdf,
                             int nx, int ny, int nz, float band) {
        (void)sdf; (void)nx; (void)ny; (void)nz; (void)band;
    }

    // Render into the currently-bound default framebuffer. Expects the pipe
    // mesh to already be drawn for the frame. Uses alpha blend; does not
    // clear.
    virtual void render(const VolumeView& v, const VolumeSettings& s) = 0;

    // Release all GL resources. Safe to call multiple times.
    virtual void shutdown() = 0;

    virtual Backend     backend()     const = 0;
    virtual const char* backendName() const = 0;
};

// ---- Factory ----------------------------------------------------------------
// Produces a backend. If the requested backend can't be constructed (e.g.
// GPU on a machine lacking GL_TEXTURE_3D) the factory falls back to CPU.
std::unique_ptr<VolumeOverlayRenderer>
makeVolumeRenderer(VolumeOverlayRenderer::Backend requested);

// Probes the current GL context and returns a sensible default backend:
//   - CPU on Apple Silicon / Intel iGPU / llvmpipe / software renderers
//   - GPU on discrete NVIDIA/AMD/Intel Arc
// Must be called with a current GL context (queries glGetString).
VolumeOverlayRenderer::Backend pickAutoBackend();

// Human-readable backend name, for UI dropdowns.
inline const char* backendName(VolumeOverlayRenderer::Backend b) {
    return (b == VolumeOverlayRenderer::Backend::GPU) ? "GPU" : "CPU";
}

} // namespace pipe_fluid
