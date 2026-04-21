#include "pipe_fluid/pipe_fluid_scene.h"
#include "pipe_fluid/pipe_boundary_field.h"
#include "pipe_fluid/pipe_solid_adapter.h"
#include "pipe_fluid/pipe_solver_boundary_data.h"
#include "pipe_fluid/blueprint_loader.h"

#include "vec3.h"               // pipe_engine/Geometry
#include "pipe_network.h"       // pipe_engine/Geometry
#include "mesh_generator.h"     // pipe_engine/Geometry
#include "voxelizer.h"          // pipe_engine/Voxelizer

#include "mac_smoke3d.h"        // smoke_engine/Sim
#include "mac_water3d.h"        // smoke_engine/Sim

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>

namespace pipe_fluid {

namespace {
// Converts a pipe_engine Vec3 into a sim's nested Vec3 (MACSmoke3D::Vec3 or
// MACWater3D::Vec3). Both are plain {x,y,z} structs, so a templated helper
// keeps the call sites clean.
template <typename SimVec3>
inline SimVec3 toSimVec3(const Vec3& v) { return SimVec3{ v.x, v.y, v.z }; }

// Persistent RNG used to break the perfectly-laminar "plug flow" look of the
// smoke source.  Seeded once per process so the result is deterministic per
// run but still has per-frame variation.
inline std::mt19937& smokeJitterRng() {
    static std::mt19937 rng(0xC0FFEEu);
    return rng;
}

// Build two unit axes perpendicular to `dir`.  Used to scatter jittered
// smoke bursts across the injection disc rather than along the axis of flow.
inline void buildOrthoBasis(const Vec3& dir, Vec3& u, Vec3& v) {
    Vec3 d = dir;
    float len = std::sqrt(d.x*d.x + d.y*d.y + d.z*d.z);
    if (len < 1e-6f) {
        u = Vec3{1.f, 0.f, 0.f};
        v = Vec3{0.f, 1.f, 0.f};
        return;
    }
    d.x /= len; d.y /= len; d.z /= len;

    // Pick any reference axis not parallel to d.
    Vec3 ref = (std::fabs(d.y) < 0.9f) ? Vec3{0.f,1.f,0.f} : Vec3{1.f,0.f,0.f};

    // u = normalize(d x ref)
    u = Vec3{
        d.y*ref.z - d.z*ref.y,
        d.z*ref.x - d.x*ref.z,
        d.x*ref.y - d.y*ref.x
    };
    float ul = std::sqrt(u.x*u.x + u.y*u.y + u.z*u.z);
    if (ul < 1e-6f) {
        u = Vec3{1.f, 0.f, 0.f};
    } else {
        u.x /= ul; u.y /= ul; u.z /= ul;
    }

    // v = d x u
    v = Vec3{
        d.y*u.z - d.z*u.y,
        d.z*u.x - d.x*u.z,
        d.x*u.y - d.y*u.x
    };
}


inline void applyFaceFractions(std::vector<float>& faceVel,
                               const std::vector<float>& faceOpen,
                               float closedThreshold = 1.0e-4f) {
    if (faceVel.size() != faceOpen.size()) return;
    for (std::size_t idx = 0; idx < faceVel.size(); ++idx) {
        const float open = std::clamp(faceOpen[idx], 0.0f, 1.0f);
        faceVel[idx] *= open;
        if (open <= closedThreshold) {
            faceVel[idx] = 0.0f;
        }
    }
}

inline void applySolverBoundaryToSmoke(MACSmoke3D& smoke,
                                       const PipeSolverBoundaryData& boundary) {
    if (!boundary.valid()) return;
    if ((int)boundary.uOpen.size() != (smoke.nx + 1) * smoke.ny * smoke.nz) return;
    if ((int)boundary.vOpen.size() != smoke.nx * (smoke.ny + 1) * smoke.nz) return;
    if ((int)boundary.wOpen.size() != smoke.nx * smoke.ny * (smoke.nz + 1)) return;

    applyFaceFractions(smoke.u, boundary.uOpen);
    applyFaceFractions(smoke.v, boundary.vOpen);
    applyFaceFractions(smoke.w, boundary.wOpen);
    smoke.derivedFieldsDirty = true;
}

inline void applySolverBoundaryToWater(MACWater3D& water,
                                       const PipeSolverBoundaryData& boundary) {
    if (!boundary.valid()) return;
    if ((int)boundary.uOpen.size() != (water.nx + 1) * water.ny * water.nz) return;
    if ((int)boundary.vOpen.size() != water.nx * (water.ny + 1) * water.nz) return;
    if ((int)boundary.wOpen.size() != water.nx * water.ny * (water.nz + 1)) return;

    // Always push the face-openness data into the solver.
    water.setFaceOpenFractions(boundary.uOpen, boundary.vOpen, boundary.wOpen);

    if (water.isCudaEnabled()) {
        // CUDA keeps its own device-side velocity state, so skip host-side
        // velocity edits here, but DO keep the solver's face-openness arrays updated.
        return;
    }

    applyFaceFractions(water.u, boundary.uOpen);
    applyFaceFractions(water.v, boundary.vOpen);
    applyFaceFractions(water.w, boundary.wOpen);
    water.derivedFieldsDirty = true;
}

inline float sampleCellCenteredFieldTrilinear(const std::vector<float>& field,
                                              int nx, int ny, int nz,
                                              float dx,
                                              float x, float y, float z) {
    if (field.empty() || nx <= 0 || ny <= 0 || nz <= 0 || dx <= 0.0f) {
        return 0.0f;
    }

    const auto clampf = [](float v, float lo, float hi) {
        return std::max(lo, std::min(hi, v));
    };
    const auto idx = [nx, ny](int i, int j, int k) {
        return static_cast<std::size_t>(i + nx * (j + ny * k));
    };

    const float gx = clampf(x / dx - 0.5f, 0.0f, std::max(0.0f, float(nx - 1)));
    const float gy = clampf(y / dx - 0.5f, 0.0f, std::max(0.0f, float(ny - 1)));
    const float gz = clampf(z / dx - 0.5f, 0.0f, std::max(0.0f, float(nz - 1)));

    const int i0 = std::max(0, std::min(nx - 1, int(std::floor(gx))));
    const int j0 = std::max(0, std::min(ny - 1, int(std::floor(gy))));
    const int k0 = std::max(0, std::min(nz - 1, int(std::floor(gz))));
    const int i1 = std::min(nx - 1, i0 + 1);
    const int j1 = std::min(ny - 1, j0 + 1);
    const int k1 = std::min(nz - 1, k0 + 1);

    const float tx = gx - float(i0);
    const float ty = gy - float(j0);
    const float tz = gz - float(k0);

    const float c000 = field[idx(i0, j0, k0)];
    const float c100 = field[idx(i1, j0, k0)];
    const float c010 = field[idx(i0, j1, k0)];
    const float c110 = field[idx(i1, j1, k0)];
    const float c001 = field[idx(i0, j0, k1)];
    const float c101 = field[idx(i1, j0, k1)];
    const float c011 = field[idx(i0, j1, k1)];
    const float c111 = field[idx(i1, j1, k1)];

    const float c00 = c000 + tx * (c100 - c000);
    const float c10 = c010 + tx * (c110 - c010);
    const float c01 = c001 + tx * (c101 - c001);
    const float c11 = c011 + tx * (c111 - c011);
    const float c0 = c00 + ty * (c10 - c00);
    const float c1 = c01 + ty * (c11 - c01);
    return c0 + tz * (c1 - c0);
}

inline void sampleWallSdfAndNormal(const PipeBoundaryField& boundary,
                                   float x, float y, float z,
                                   float& phi,
                                   float& nx, float& ny, float& nz) {
    phi = sampleCellCenteredFieldTrilinear(boundary.wallSdf,
                                           boundary.nx, boundary.ny, boundary.nz,
                                           boundary.dx,
                                           x, y, z);

    const float eps = std::max(0.25f * boundary.dx, 1.0e-5f);
    const float px0 = sampleCellCenteredFieldTrilinear(boundary.wallSdf,
                                                       boundary.nx, boundary.ny, boundary.nz,
                                                       boundary.dx,
                                                       x - eps, y, z);
    const float px1 = sampleCellCenteredFieldTrilinear(boundary.wallSdf,
                                                       boundary.nx, boundary.ny, boundary.nz,
                                                       boundary.dx,
                                                       x + eps, y, z);
    const float py0 = sampleCellCenteredFieldTrilinear(boundary.wallSdf,
                                                       boundary.nx, boundary.ny, boundary.nz,
                                                       boundary.dx,
                                                       x, y - eps, z);
    const float py1 = sampleCellCenteredFieldTrilinear(boundary.wallSdf,
                                                       boundary.nx, boundary.ny, boundary.nz,
                                                       boundary.dx,
                                                       x, y + eps, z);
    const float pz0 = sampleCellCenteredFieldTrilinear(boundary.wallSdf,
                                                       boundary.nx, boundary.ny, boundary.nz,
                                                       boundary.dx,
                                                       x, y, z - eps);
    const float pz1 = sampleCellCenteredFieldTrilinear(boundary.wallSdf,
                                                       boundary.nx, boundary.ny, boundary.nz,
                                                       boundary.dx,
                                                       x, y, z + eps);

    nx = (px1 - px0) / (2.0f * eps);
    ny = (py1 - py0) / (2.0f * eps);
    nz = (pz1 - pz0) / (2.0f * eps);
    const float len = std::sqrt(nx * nx + ny * ny + nz * nz);
    if (len > 1.0e-8f) {
        nx /= len;
        ny /= len;
        nz /= len;
    } else {
        nx = ny = nz = 0.0f;
    }
}

inline void confineWaterParticlesToPipe(MACWater3D& water,
                                        const PipeBoundaryField& boundary) {
    if (water.isCudaEnabled()) {
        // Without a dedicated host->device particle upload path, keep the CUDA
        // backend unchanged. This confinement pass currently targets the CPU path.
        return;
    }
    if (!boundary.valid() || water.particles.empty()) return;
    if (boundary.nx != water.nx || boundary.ny != water.ny || boundary.nz != water.nz) return;
    if (std::fabs(boundary.dx - water.dx) > 1.0e-6f) return;

    const float dx = water.dx;
    const float clearance = 0.35f * dx;
    const float maxPush = 1.5f * dx;
    const float velDampTangent = 0.98f;

    const float xMax = std::max(0.0f, water.nx * dx - 1.0e-4f * dx);
    const float yMax = std::max(0.0f, water.ny * dx - 1.0e-4f * dx);
    const float zMax = std::max(0.0f, water.nz * dx - 1.0e-4f * dx);

    for (auto& p : water.particles) {
        float phi = 0.0f;
        float nx = 0.0f, ny = 0.0f, nz = 0.0f;
        sampleWallSdfAndNormal(boundary, p.x, p.y, p.z, phi, nx, ny, nz);

        // Patch E: skip confinement when the particle is already outside the
        // pipe.  With the signed-SDF from Patch D, grad(phi) at an Exterior
        // cell points *toward* the pipe interior, so the along-+grad push
        // below would yank outside-spawned particles onto (and, when the
        // per-step push exceeds the 1-cell wall shell, *through*) the pipe
        // walls.  Outside-pipe motion is handled correctly by gravity plus
        // the face-blocked MAC pressure solve — no confinement needed.
        {
            const int ci = (int)std::floor(p.x / dx);
            const int cj = (int)std::floor(p.y / dx);
            const int ck = (int)std::floor(p.z / dx);
            if (ci >= 0 && ci < boundary.nx &&
                cj >= 0 && cj < boundary.ny &&
                ck >= 0 && ck < boundary.nz) {
                if (boundary.cells[boundary.idx(ci, cj, ck)] ==
                    PipeBoundaryCell::Exterior) {
                    continue;
                }
            } else {
                // Outside the boundary grid entirely — nothing to confine to.
                continue;
            }
        }

        if (phi >= clearance) continue;

        const float nLen2 = nx * nx + ny * ny + nz * nz;
        if (nLen2 < 1.0e-10f) continue;

        const float push = std::min(clearance - phi + 1.0e-4f * dx, maxPush);
        p.x += nx * push;
        p.y += ny * push;
        p.z += nz * push;

        p.x = std::max(0.0f, std::min(xMax, p.x));
        p.y = std::max(0.0f, std::min(yMax, p.y));
        p.z = std::max(0.0f, std::min(zMax, p.z));

        const float vn = p.u * nx + p.v * ny + p.w * nz;
        if (vn < 0.0f) {
            p.u -= vn * nx;
            p.v -= vn * ny;
            p.w -= vn * nz;
        }

        const float vtX = p.u - (p.u * nx + p.v * ny + p.w * nz) * nx;
        const float vtY = p.v - (p.u * nx + p.v * ny + p.w * nz) * ny;
        const float vtZ = p.w - (p.u * nx + p.v * ny + p.w * nz) * nz;
        const float vn2 = p.u * nx + p.v * ny + p.w * nz;
        p.u = vn2 * nx + velDampTangent * vtX;
        p.v = vn2 * ny + velDampTangent * vtY;
        p.w = vn2 * nz + velDampTangent * vtZ;

        p.c00 = p.c01 = p.c02 = 0.0f;
        p.c10 = p.c11 = p.c12 = 0.0f;
        p.c20 = p.c21 = p.c22 = 0.0f;
    }

    water.derivedFieldsDirty = true;
}
}

struct PipeFluidScene::Impl {
    Config cfg;

    PipeNetwork   network;
    VoxelGrid     voxels;
    TriMesh       mesh;
    PipeBoundaryField boundary;
    PipeSolverBoundaryData solverBoundary;

    // Both fluids use a "wall-only" mask: only the pipe MATERIAL (Solid)
    // blocks the sim.  Fluid, Opening and Air are all passable.  The open
    // Air pad around the pipe acts as an open sink for smoke and a free-
    // fall basin for water exiting the pipe mouth.  The grid borders are
    // re-sealed each step by MACWater3D::rebuildBorderSolids(), so the
    // domain stays closed even though the interior Air is permeable.
    std::vector<uint8_t> smokeMask;   // nx*ny*nz: Solid->1, Air/Fluid->0
    std::vector<uint8_t> waterMask;   // nx*ny*nz: Solid->1, Air/Fluid/Opening->0

    std::unique_ptr<MACSmoke3D> smoke;
    std::unique_ptr<MACWater3D> water;

    // Narrow-band SDF built from water->particles after each step().
    // nx*ny*nz floats in world units (metres).  Negative inside the water
    // body, positive outside, clamped to +/- bandWidth.
    std::vector<float> waterSdf;
    float              waterSdfBand = 0.0f;   // positive clamp value used

    bool geometryDirty = true;        // rebuild() will refresh everything
};

// ---- ctor / dtor / move -----------------------------------------------------

PipeFluidScene::PipeFluidScene(const Config& cfg) : p_(std::make_unique<Impl>()) {
    p_->cfg = cfg;
    p_->network.defaultInnerRadius = cfg.geometry.defaultInnerRadius;
    p_->network.defaultOuterRadius = cfg.geometry.defaultOuterRadius;
}

PipeFluidScene::~PipeFluidScene() = default;
PipeFluidScene::PipeFluidScene(PipeFluidScene&&) noexcept = default;
PipeFluidScene& PipeFluidScene::operator=(PipeFluidScene&&) noexcept = default;

// ---- Builder ----------------------------------------------------------------

void PipeFluidScene::beginNetwork(const Vec3& start, const Vec3& dir) {
    p_->network.begin(start, dir);
    p_->geometryDirty = true;
}

void PipeFluidScene::beginChain(const Vec3& start, const Vec3& dir) {
    // Alias for beginNetwork: starts a NEW chain at (start, dir) without
    // clearing previous chains.  Use this to build T / Y / cross junctions
    // or any topology with multiple inlets/outlets.
    p_->network.begin(start, dir);
    p_->geometryDirty = true;
}

void PipeFluidScene::addStraight(float length) {
    p_->network.addStraight(length);
    p_->geometryDirty = true;
}

void PipeFluidScene::addBend90(const Vec3& newDir, float bendRadius) {
    p_->network.addBend90(newDir, bendRadius);
    p_->geometryDirty = true;
}

void PipeFluidScene::clearNetwork() {
    // Drops segments AND chain partition so subsequent begin() calls start
    // cleanly.  Without this, the voxelizer's openEnds() method would still
    // see the previous chains' endpoints and try to carve them.
    p_->network.clear();
    p_->network.defaultInnerRadius = p_->cfg.geometry.defaultInnerRadius;
    p_->network.defaultOuterRadius = p_->cfg.geometry.defaultOuterRadius;
    p_->geometryDirty = true;
}

bool PipeFluidScene::loadBlueprint(const std::string& path, std::string* errorOut) {
    PipeNetwork loaded;
    auto r = loadBlueprintFile(path, loaded);
    if (!r.ok) {
        if (errorOut) *errorOut = r.error;
        return false;
    }
    p_->network = std::move(loaded);
    // keep default radii defined by config, but BlueprintParser may overwrite
    p_->geometryDirty = true;
    return true;
}

// ---- Core rebuild -----------------------------------------------------------

void PipeFluidScene::rebuild() {
    // 1. Voxelize the pipe network into a VoxelGrid sized by the pipe bbox.
    PipeVoxelizer voxer;
    voxer.cellSize       = p_->cfg.grid.cellSize;
    voxer.padding        = p_->cfg.grid.padding;
    voxer.gravityPadding = p_->cfg.grid.gravityPadding;
    p_->voxels = voxer.voxelize(p_->network);

    // 2. Build the canonical boundary field, then derive solver-facing
    //    boundary data and legacy simulator masks from it.
    p_->boundary = buildPipeBoundaryField(p_->network, p_->voxels);
    p_->solverBoundary = buildSolverBoundaryData(p_->boundary);
    p_->smokeMask = p_->solverBoundary.solidMask;
    p_->waterMask = p_->solverBoundary.waterSolidMask;

    // 3. Regenerate the render mesh.
    MeshGenerator mg;
    p_->mesh = mg.generatePipeMesh(p_->network);

    // 4. Re-dimension the fluid simulators to match the voxel grid exactly.
    //    The sims accept setVoxelSolids with size nx*ny*nz using the same
    //    idx = i + nx*(j + ny*k) convention as the VoxelGrid.
    const int NX = p_->voxels.nx;
    const int NY = p_->voxels.ny;
    const int NZ = p_->voxels.nz;
    const float DX = p_->voxels.dx;
    const float DT = (p_->cfg.sim.dt > 0.f) ? p_->cfg.sim.dt : 1.0f / 60.0f;

    if (p_->cfg.sim.enableSmoke) {
        if (!p_->smoke) {
            p_->smoke = std::make_unique<MACSmoke3D>(NX, NY, NZ, DX, DT);
        } else {
            p_->smoke->reset(NX, NY, NZ, DX, DT);
        }
        applySolidsToSmoke(*p_->smoke, p_->smokeMask);
        p_->smoke->setFaceOpenFractions(p_->solverBoundary.uOpen,
                                        p_->solverBoundary.vOpen,
                                        p_->solverBoundary.wOpen);
        applySolverBoundaryToSmoke(*p_->smoke, p_->solverBoundary);
    } else {
        p_->smoke.reset();
    }

    if (p_->cfg.sim.enableWater) {
        if (!p_->water) {
            p_->water = std::make_unique<MACWater3D>(NX, NY, NZ, DX, DT);
        } else {
            p_->water->reset(NX, NY, NZ, DX, DT);
        }
        // Higher particles-per-cell → smoother rasterised density field.
        // The default (2) yields a ~50% stochastic variation at cell scale
        // between neighbouring cells, which is what shows up in the renderer
        // as "blocky strands".  With 8 particles/cell the variance drops by
        // ~2x and the continuous water body is resolved much better.
        //
        // flipBlend controls the FLIP/PIC mix.  The default (0.1) is very
        // PIC-heavy, which damps all sub-grid motion away and makes the
        // water look like a viscous gel: particles move as one coherent
        // blob rather than a turbulent fluid.  0.95 keeps ~95% of the per-
        // particle velocity delta each step (FLIP-dominant), so splashes,
        // swirls and free-fall dispersion all show up — much closer to what
        // real water looks like flowing through a pipe.
        //
        // borderThickness=1 (not the default 2) is the minimum the sim
        // allows.  It matters because MACWater3D::rebuildBorderSolids()
        // always seals the outermost `borderThickness` cells on all six
        // faces, OVERWRITING our user waterMask.  The pipe voxelizer
        // carves fall-through Opening columns that may extend right to
        // the grid floor; with borderThickness=2 the bottom 2 layers of
        // those columns are re-sealed every step, which is why water
        // reaches the end of the exit channel and stalls.  At
        // borderThickness=1 and padding=0.10m the exit channel and
        // almost all of the fall column stay passable.
        //
        // openTop=false is critical.  The default is `true`, which opens
        // the grid's +Y face and leaves gravity with nowhere to send
        // water in a pipe whose outlet faces sideways or downward.
        // Closing the top lets the pressure solve see a properly-bounded
        // liquid region and the Opening cells at the pipe mouth become
        // the only real free surface, which is what we want.
        //
        // useAPIC=true + pressureMGVCycles/pressureMGCoarseIters/
        // reseedRelaxIters/reseedRelaxStrength/volumePreserveStrength
        // are the exact settings smoke_engine pushes in from its UI
        // when running Water3D.  Without them we inherit MACWater3D's
        // bare defaults (useAPIC=true already, but the relaxation /
        // volume-preservation work that keeps particles evenly spaced
        // would otherwise run at the raw defaults, which show up as
        // clumping in a narrow pipe).
        auto wParams = p_->water->params;
        wParams.particlesPerCell       = std::max(wParams.particlesPerCell, 8);
        wParams.flipBlend              = 0.95f;
        wParams.borderThickness        = 1;
        wParams.openTop                = false;
        wParams.useAPIC                = true;
        wParams.pressureSolverMode     = (int)MACWater3D::PressureSolverMode::Multigrid;
        wParams.pressureIters          = std::max(wParams.pressureIters, 200);
        wParams.pressureMGVCycles      = std::max(wParams.pressureMGVCycles, 50);
        wParams.pressureMGCoarseIters  = std::max(wParams.pressureMGCoarseIters, 40);
        wParams.reseedRelaxIters       = std::max(wParams.reseedRelaxIters, 2);
        wParams.reseedRelaxStrength    = 0.45f;
        wParams.volumePreserveRhsMean  = true;
        wParams.volumePreserveStrength = 0.05f;
        p_->water->setParams(wParams);
        applySolidsToWater(*p_->water, p_->waterMask);
        applySolverBoundaryToWater(*p_->water, p_->solverBoundary);

        // Pre-size the SDF to the new grid dimensions, initialised to
        // "far from water" (+band) so a first-frame render before any
        // step() returns a fully-transparent water pass.
        const float initBand = 3.0f * p_->voxels.dx;
        p_->waterSdf.assign(
            (std::size_t)NX * (std::size_t)NY * (std::size_t)NZ, initBand);
        p_->waterSdfBand = initBand;
    } else {
        p_->water.reset();
        p_->waterSdf.clear();
        p_->waterSdfBand = 0.0f;
    }

    p_->geometryDirty = false;
}

// ---- Step loop --------------------------------------------------------------

namespace {
// Build a narrow-band SDF from a FLIP particle cloud on a uniform MAC grid.
//
// Method: union of spheres.  Each particle contributes a sphere of radius
// `particleR` centred on its position; the per-cell SDF value is the
// minimum over all contributing particles of
//     (distance from cell centre to particle) - particleR
// which is the standard blob-union SDF used in particle renderers (think
// metaballs, Zhu-Bridson's simple variant).  A narrow-band clamp of +/-
// `band` keeps the SDF meaningful only near the surface — cells far from
// any water particle saturate to +band rather than infinity, so the
// renderer can still sphere-trace large empty steps without NaNs.
//
// Complexity: O(N_particles * R_cells^3) where R_cells is the per-particle
// bounding box in cells.  With particleR ~= 1.1*dx the per-particle footprint
// is at most 3x3x3 cells = 27 updates — cheap even for 500k particles.
void buildLiquidSdfFromParticles(const MACWater3D& water,
                                 float particleRadiusScale,
                                 float bandCells,
                                 std::vector<float>& out,
                                 float& bandOut) {
    const int nx = water.nx;
    const int ny = water.ny;
    const int nz = water.nz;
    const float dx = water.dx;
    if (nx <= 0 || ny <= 0 || nz <= 0 || dx <= 0.f) {
        out.clear();
        bandOut = 0.f;
        return;
    }

    // Slightly-larger-than-cell metaball radius.  Too small -> pinholes
    // between adjacent particles; too large -> water looks like it fills
    // the pipe whether or not it does.  1.1*dx gives ~10% overlap between
    // neighbouring cells of particles, which fuses them smoothly.
    const float particleR = std::max(0.1f, particleRadiusScale) * dx;
    // Narrow band: enough to give the sphere-tracer a meaningful step size
    // several cells outside the surface without consuming memory on a
    // full-grid SDF.
    const float band      = std::max(0.0f, bandCells) * dx;
    bandOut = band;

    const std::size_t total = (std::size_t)nx * (std::size_t)ny * (std::size_t)nz;
    out.assign(total, band);

    // Per-particle footprint radius: sphere-of-influence extends particleR
    // out from the particle centre; we widen it by one cell so boundary
    // cells of the sphere are still updated.
    const float influence = particleR + dx;
    const float invDx     = 1.f / dx;

    auto clampi = [](int x, int lo, int hi) {
        return (x < lo) ? lo : (x > hi) ? hi : x;
    };

    for (const auto& p : water.particles) {
        const float px = p.x;
        const float py = p.y;
        const float pz = p.z;

        const int i0 = clampi((int)std::floor((px - influence) * invDx), 0, nx - 1);
        const int i1 = clampi((int)std::floor((px + influence) * invDx), 0, nx - 1);
        const int j0 = clampi((int)std::floor((py - influence) * invDx), 0, ny - 1);
        const int j1 = clampi((int)std::floor((py + influence) * invDx), 0, ny - 1);
        const int k0 = clampi((int)std::floor((pz - influence) * invDx), 0, nz - 1);
        const int k1 = clampi((int)std::floor((pz + influence) * invDx), 0, nz - 1);

        for (int k = k0; k <= k1; ++k) {
            const float cz = (k + 0.5f) * dx;
            const float dz = cz - pz;
            for (int j = j0; j <= j1; ++j) {
                const float cy = (j + 0.5f) * dx;
                const float dy = cy - py;
                for (int i = i0; i <= i1; ++i) {
                    const float cx = (i + 0.5f) * dx;
                    const float dxw = cx - px;
                    const float dist = std::sqrt(dxw*dxw + dy*dy + dz*dz);
                    const float val  = dist - particleR;

                    const std::size_t idx =
                        (std::size_t)i +
                        (std::size_t)nx * ((std::size_t)j + (std::size_t)ny * (std::size_t)k);
                    if (val < out[idx]) out[idx] = val;
                }
            }
        }
    }

    // Clamp below at -band too, so the gradient stays well-conditioned in
    // deep fluid regions (otherwise very-interior cells can hit -2*dx or
    // more, which the renderer never needs).
    const float minBand = -band;
    for (std::size_t i = 0; i < total; ++i) {
        if (out[i] < minBand) out[i] = minBand;
    }
}
} // anonymous namespace

void PipeFluidScene::step(float dtOverride) {
    if (p_->geometryDirty) rebuild();
    const float dt = (dtOverride > 0.f) ? dtOverride : p_->cfg.sim.dt;
    if (p_->smoke) {
        if (dt > 0.f) p_->smoke->setDt(dt);
        p_->smoke->step();
        applySolverBoundaryToSmoke(*p_->smoke, p_->solverBoundary);
    }
    if (p_->water) {
        if (dt > 0.f) p_->water->setDt(dt);
        p_->water->step();
        applySolverBoundaryToWater(*p_->water, p_->solverBoundary);
        confineWaterParticlesToPipe(*p_->water, p_->boundary);
        // Rebuild the narrow-band SDF from the FLIP particles so the volume
        // renderer can sphere-trace a smooth liquid surface instead of
        // volume-integrating a per-cell density field.  This is the bridge
        // that eliminates the blocky-cube look: SDF gives a continuous
        // isosurface independent of the cell resolution.
        buildLiquidSdfFromParticles(*p_->water,
                                   p_->cfg.waterSurface.particleRadiusScale,
                                   p_->cfg.waterSurface.bandCells,
                                   p_->waterSdf, p_->waterSdfBand);
    }
}

void PipeFluidScene::resetFluids() {
    // Preserve geometry; just clear fluid state and re-push the per-sim mask.
    if (p_->smoke) {
        p_->smoke->reset();
        applySolidsToSmoke(*p_->smoke, p_->smokeMask);
        p_->smoke->setFaceOpenFractions(p_->solverBoundary.uOpen,
                                        p_->solverBoundary.vOpen,
                                        p_->solverBoundary.wOpen);
        applySolverBoundaryToSmoke(*p_->smoke, p_->solverBoundary);
    }
    if (p_->water) {
        p_->water->reset();
        applySolidsToWater(*p_->water, p_->waterMask);
        applySolverBoundaryToWater(*p_->water, p_->solverBoundary);
        p_->waterSdf.assign(p_->waterSdf.size(), p_->waterSdfBand > 0.0f ? p_->waterSdfBand : 0.0f);
    }
}

// ---- Source helpers ---------------------------------------------------------

void PipeFluidScene::addSmokeSourceSphere(const Vec3& centre, float radius,
                                          float amount, const Vec3& velocity) {
    if (!p_->smoke) return;
    // The smoke sim's coordinate origin is the voxel grid's world origin.
    // Convert world-space coords to sim-space before passing them in.
    const Vec3 simC = centre - p_->voxels.origin;

    // Smoke injected with a single uniform velocity over a sphere comes out as
    // a perfectly symmetric "plug" - the solver has no asymmetry to amplify
    // into eddies, so the smoke slides down the pipe like a solid core.
    // Here we split the injection into a handful of smaller, slightly
    // jittered sub-bursts scattered across the disc perpendicular to the
    // flow direction.  The solver sees a non-uniform source every step and
    // the resulting shear / pressure differences produce realistic
    // vortex-shedding and mixing inside the pipe.
    Vec3 u, v;
    buildOrthoBasis(velocity, u, v);

    auto& rng = smokeJitterRng();
    // Offsets in the plane perpendicular to `velocity`, as a fraction of
    // `radius`.  [-0.6 .. +0.6] keeps every sub-burst inside the parent
    // sphere (their own radii are 0.4*radius, see below).
    std::uniform_real_distribution<float> off(-0.6f, 0.6f);
    // Per-burst speed scaling. 0.8 .. 1.2 = +/-20% speed variation.
    std::uniform_real_distribution<float> spdScale(0.8f, 1.2f);
    // Small off-axis tilt (as a fraction of |velocity|).
    std::uniform_real_distribution<float> tiltScale(-0.25f, 0.25f);

    constexpr int N = 4;                  // number of sub-bursts
    const float   subR = 0.4f * radius;   // each sub-burst radius
    const float   subA = amount / float(N);

    // Tilts are scaled by |velocity| so the off-axis kick stays proportional
    // to the flow speed no matter what the caller passes in.
    const float vmag = std::sqrt(velocity.x*velocity.x +
                                 velocity.y*velocity.y +
                                 velocity.z*velocity.z);

    for (int i = 0; i < N; ++i) {
        const float ou = off(rng) * radius;
        const float ov = off(rng) * radius;
        const Vec3  c  = simC + Vec3{ u.x*ou + v.x*ov,
                                      u.y*ou + v.y*ov,
                                      u.z*ou + v.z*ov };
        const float s   = spdScale(rng);
        const float tu  = tiltScale(rng) * vmag;
        const float tv  = tiltScale(rng) * vmag;
        const Vec3  vel{
            velocity.x * s + u.x*tu + v.x*tv,
            velocity.y * s + u.y*tu + v.y*tv,
            velocity.z * s + u.z*tu + v.z*tv,
        };

        p_->smoke->addSmokeSourceSphere(
            toSimVec3<MACSmoke3D::Vec3>(c),
            subR, subA,
            toSimVec3<MACSmoke3D::Vec3>(vel));
    }
    applySolverBoundaryToSmoke(*p_->smoke, p_->solverBoundary);
}

void PipeFluidScene::addWaterSourceSphere(const Vec3& centre, float radius,
                                          const Vec3& velocity) {
    if (!p_->water) return;
    const Vec3 simC = centre - p_->voxels.origin;

    // Same rationale as addSmokeSourceSphere: a single uniform-velocity
    // sphere injected into a narrow pipe produces perfectly symmetric
    // plug flow — every particle gets exactly the same velocity, the
    // pressure solve sees no asymmetry to amplify, and the fluid slides
    // down the pipe as a coherent block.  Splitting the injection into
    // N smaller jittered sub-bursts scattered across the disc
    // perpendicular to `velocity` gives the solver a non-uniform source
    // every step, which is what produces real fluid-like shear, vortex
    // shedding, and free-fall dispersion at the pipe mouth.
    Vec3 u, v;
    buildOrthoBasis(velocity, u, v);

    auto& rng = smokeJitterRng();
    std::uniform_real_distribution<float> off(-0.6f, 0.6f);
    std::uniform_real_distribution<float> spdScale(0.8f, 1.2f);
    std::uniform_real_distribution<float> tiltScale(-0.25f, 0.25f);

    constexpr int N = 4;                  // number of sub-bursts
    const float   subR = 0.5f * radius;   // each sub-burst radius

    const float vmag = std::sqrt(velocity.x*velocity.x +
                                 velocity.y*velocity.y +
                                 velocity.z*velocity.z);

    for (int i = 0; i < N; ++i) {
        const float ou = off(rng) * radius;
        const float ov = off(rng) * radius;
        const Vec3  c  = simC + Vec3{ u.x*ou + v.x*ov,
                                      u.y*ou + v.y*ov,
                                      u.z*ou + v.z*ov };
        const float s   = spdScale(rng);
        const float tu  = tiltScale(rng) * vmag;
        const float tv  = tiltScale(rng) * vmag;
        const Vec3  vel{
            velocity.x * s + u.x*tu + v.x*tv,
            velocity.y * s + u.y*tu + v.y*tv,
            velocity.z * s + u.z*tu + v.z*tv,
        };

        p_->water->addWaterSourceSphere(
            toSimVec3<MACWater3D::Vec3>(c),
            subR,
            toSimVec3<MACWater3D::Vec3>(vel));
    }
    applySolverBoundaryToWater(*p_->water, p_->solverBoundary);
    confineWaterParticlesToPipe(*p_->water, p_->boundary);
}

// ---- Accessors --------------------------------------------------------------

const PipeNetwork& PipeFluidScene::network() const { return p_->network; }
PipeNetwork&       PipeFluidScene::network()       { p_->geometryDirty = true; return p_->network; }
const VoxelGrid&   PipeFluidScene::voxels()  const { return p_->voxels; }
const TriMesh&     PipeFluidScene::pipeMesh() const { return p_->mesh; }

MACSmoke3D* PipeFluidScene::smoke() { return p_->smoke.get(); }
const MACSmoke3D* PipeFluidScene::smoke() const { return p_->smoke.get(); }
MACWater3D* PipeFluidScene::water() { return p_->water.get(); }
const MACWater3D* PipeFluidScene::water() const { return p_->water.get(); }

const PipeBoundaryField& PipeFluidScene::boundaryField() const {
    return p_->boundary;
}

const PipeSolverBoundaryData& PipeFluidScene::solverBoundary() const {
    return p_->solverBoundary;
}

const std::vector<float>& PipeFluidScene::waterSDF() const {
    return p_->waterSdf;
}

float PipeFluidScene::waterSdfBand() const {
    return p_->waterSdfBand;
}

const std::vector<float>& PipeFluidScene::pipeWallSDF() const {
    return p_->boundary.wallSdf;
}

const std::vector<uint8_t>& PipeFluidScene::renderSolidMask() const {
    // Always use the walls-only mask for rendering.  The water solver seals
    // only the outermost simulation-domain border internally; the renderer
    // should not treat that temporary border sealing as real pipe geometry.
    return p_->boundary.wallMask;
}

const PipeFluidScene::Config& PipeFluidScene::config() const { return p_->cfg; }
void PipeFluidScene::setConfig(const Config& c) {
    p_->cfg = c;
    p_->network.defaultInnerRadius = c.geometry.defaultInnerRadius;
    p_->network.defaultOuterRadius = c.geometry.defaultOuterRadius;
    p_->geometryDirty = true;
}

int   PipeFluidScene::nx() const { return p_->voxels.nx; }
int   PipeFluidScene::ny() const { return p_->voxels.ny; }
int   PipeFluidScene::nz() const { return p_->voxels.nz; }
float PipeFluidScene::cellSize() const { return p_->voxels.dx; }

PipeFluidScene::Stats PipeFluidScene::stats() const {
    Stats s;
    s.nx = p_->voxels.nx; s.ny = p_->voxels.ny; s.nz = p_->voxels.nz;
    for (auto c : p_->voxels.cells) {
        if      (c == VoxelType::Solid)   ++s.solidCells;
        else if (c == VoxelType::Fluid)   ++s.fluidCells;
        else if (c == VoxelType::Opening) ++s.openingCells;
    }
    s.totalPipeLength = p_->network.totalLength();
    s.segmentCount = static_cast<int>(p_->network.numSegments());
    s.chainCount   = static_cast<int>(p_->network.numChains());
    return s;
}

} // namespace pipe_fluid