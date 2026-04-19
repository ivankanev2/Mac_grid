#pragma once
// ============================================================================
// PipeFluidScene
//
// The top-level integration point between pipe_engine and smoke_engine.
//
// A PipeFluidScene owns:
//   - A PipeNetwork   (geometry)
//   - A VoxelGrid     (voxelized pipe walls, built on rebuild())
//   - A TriMesh       (pipe surface mesh, for rendering)
//   - An optional MACSmoke3D and MACWater3D, both sharing the same (nx,ny,nz)
//     solid mask derived from the VoxelGrid.
//
// Typical usage (programmatic):
//
//     pipe_fluid::PipeFluidScene::Config cfg;
//     cfg.cellSize = 0.01f;      // 1 cm voxels
//     cfg.padding  = 0.10f;      // 10 cm padding around bbox
//     cfg.enableSmoke = true;
//     cfg.enableWater = false;
//     pipe_fluid::PipeFluidScene scene(cfg);
//
//     scene.beginNetwork({0,0,0}, {0,0,1});
//     scene.addStraight(1.0f);
//     scene.addBend90({1,0,0});
//     scene.addStraight(0.8f);
//     scene.rebuild();
//
//     for (int i = 0; i < 300; ++i) scene.step();
//
// Typical usage (blueprint):
//
//     pipe_fluid::PipeFluidScene scene(cfg);
//     scene.loadBlueprint("pipe_fluid_engine/examples/demo_L.pipe");
//     scene.rebuild();
// ============================================================================

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Forward-declare to keep includes light for consumers.
struct PipeNetwork;           // pipe_engine/Geometry/pipe_network.h
struct VoxelGrid;             // pipe_engine/Voxelizer/voxelizer.h
struct TriMesh;               // pipe_engine/Geometry/mesh_generator.h
struct Vec3;                  // pipe_engine/Geometry/vec3.h
struct MACSmoke3D;            // smoke_engine/Sim/mac_smoke3d.h
struct MACWater3D;            // smoke_engine/Sim/mac_water3d.h

namespace pipe_fluid {

class PipeFluidScene {
public:
    struct Config {
        // Voxelizer parameters — these drive both the solid mask and the
        // fluid grid dimensions. Keep them in sync: the fluid sims are
        // recreated with the VoxelGrid's resulting (nx,ny,nz,dx).
        float cellSize = 0.01f;   // metres per voxel
        float padding  = 0.10f;   // metres around the pipe bbox

        bool enableSmoke = true;
        bool enableWater = false;

        // Simulation timestep. If <= 0, each sim uses its own default.
        float dt = 1.0f / 60.0f;

        // Default pipe radii applied to the network builder.
        float defaultInnerRadius = 0.05f;
        float defaultOuterRadius = 0.06f;
    };

    // Construct with a config. Use `PipeFluidScene(PipeFluidScene::Config{})`
    // to get all defaults.
    explicit PipeFluidScene(const Config& cfg);
    ~PipeFluidScene();

    // Non-copyable, movable.
    PipeFluidScene(const PipeFluidScene&) = delete;
    PipeFluidScene& operator=(const PipeFluidScene&) = delete;
    PipeFluidScene(PipeFluidScene&&) noexcept;
    PipeFluidScene& operator=(PipeFluidScene&&) noexcept;

    // ---- Builder API (wraps PipeNetwork) -----------------------------------
    //
    // A scene can contain multiple independent chains.  The first chain is
    // started with `beginNetwork`; additional branches (for T / Y / cross /
    // manifold topologies) are started with `beginChain` at a new world
    // point.  Each chain contributes its OWN pair of open endpoints, so the
    // voxelizer carves a pipe mouth at each chain's start and end.
    void beginNetwork(const Vec3& start, const Vec3& dir);
    void beginChain  (const Vec3& start, const Vec3& dir);
    void addStraight(float length);
    void addBend90(const Vec3& newDir, float bendRadius = 0.15f);
    void clearNetwork();

    // ---- Blueprint IO ------------------------------------------------------
    // Returns true on success. On failure, errorOut (if non-null) holds
    // a human-readable message.
    bool loadBlueprint(const std::string& path, std::string* errorOut = nullptr);

    // ---- Core pipeline -----------------------------------------------------
    // Re-voxelizes the network, rebuilds the solid mask, regenerates the
    // render mesh, and recreates the fluid simulators at the resulting
    // (nx,ny,nz,dx). Call after any geometry change.
    void rebuild();

    // Step the enabled fluid simulators. Pass dt<=0 to use config.dt.
    void step(float dt = -1.0f);

    // Clear fluid state (smoke density, water particles) without touching
    // geometry. Useful for the "reset" button in the viewer.
    void resetFluids();

    // ---- Smoke / water source helpers -------------------------------------
    // Inject a sphere of smoke/water into the fluid grid. Coordinates are
    // in world space (metres), matching the VoxelGrid origin.
    void addSmokeSourceSphere(const Vec3& centre, float radius,
                              float amount, const Vec3& velocity);
    void addWaterSourceSphere(const Vec3& centre, float radius,
                              const Vec3& velocity);

    // ---- Accessors ---------------------------------------------------------
    const PipeNetwork& network() const;
    PipeNetwork&       network();
    const VoxelGrid&   voxels() const;
    const TriMesh&     pipeMesh() const;

    MACSmoke3D*        smoke();              // nullptr if not enabled
    MACWater3D*        water();              // nullptr if not enabled

    // The "walls-only" solid mask (Solid->1, everything else ->0).  This is
    // the mask to pass to the volume RENDERER so raymarch hard-cutoffs only
    // trigger on actual pipe walls.  The water sim internally uses a sealed
    // mask that additionally blocks Air cells to keep FLIP particles inside
    // the pipe; that sealed mask is NOT suitable for rendering because the
    // per-cell step function produces blocky cutoffs at every Air cell a ray
    // crosses, even when the water volume is continuous.
    const std::vector<uint8_t>& renderSolidMask() const;

    // Narrow-band signed distance field built from the FLIP particles, sized
    // nx*ny*nz in simulator order.  Values are in WORLD UNITS (metres):
    //   negative = inside the water body (distance to nearest surface)
    //   positive = outside the water body (distance to nearest particle)
    //   clamped  = 3 * cellSize above/below the surface.
    // Use this for SDF sphere-tracing in the volume renderer; it eliminates
    // the per-voxel density quantisation artefacts that make the voxel path
    // look blocky at pipe scale.  Rebuilt once at the end of every step().
    const std::vector<float>& waterSDF() const;

    const Config&      config() const;
    void               setConfig(const Config& c);  // takes effect on next rebuild()

    int nx() const; int ny() const; int nz() const;
    float cellSize() const;

    // Diagnostics
    struct Stats {
        int nx = 0, ny = 0, nz = 0;
        int solidCells   = 0;
        int fluidCells   = 0;
        int openingCells = 0;      // carved by voxelizer's open-ends pass
        float totalPipeLength = 0.f;
        int segmentCount = 0;
        int chainCount   = 0;      // number of non-empty chains
    };
    Stats stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

} // namespace pipe_fluid
