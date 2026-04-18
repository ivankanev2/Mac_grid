#include "pipe_fluid/pipe_fluid_scene.h"
#include "pipe_fluid/pipe_solid_adapter.h"
#include "pipe_fluid/blueprint_loader.h"

#include "vec3.h"               // pipe_engine/Geometry
#include "pipe_network.h"       // pipe_engine/Geometry
#include "mesh_generator.h"     // pipe_engine/Geometry
#include "voxelizer.h"          // pipe_engine/Voxelizer

#include "mac_smoke3d.h"        // smoke_engine/Sim
#include "mac_water3d.h"        // smoke_engine/Sim

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace pipe_fluid {

namespace {
// Converts a pipe_engine Vec3 into a sim's nested Vec3 (MACSmoke3D::Vec3 or
// MACWater3D::Vec3). Both are plain {x,y,z} structs, so a templated helper
// keeps the call sites clean.
template <typename SimVec3>
inline SimVec3 toSimVec3(const Vec3& v) { return SimVec3{ v.x, v.y, v.z }; }
}

struct PipeFluidScene::Impl {
    Config cfg;

    PipeNetwork network;
    VoxelGrid   voxels;
    TriMesh     mesh;
    std::vector<uint8_t> solidMask;   // nx*ny*nz, 0=fluid, 1=solid

    std::unique_ptr<MACSmoke3D> smoke;
    std::unique_ptr<MACWater3D> water;

    bool geometryDirty = true;        // rebuild() will refresh everything
};

// ---- ctor / dtor / move -----------------------------------------------------

PipeFluidScene::PipeFluidScene(const Config& cfg) : p_(std::make_unique<Impl>()) {
    p_->cfg = cfg;
    p_->network.defaultInnerRadius = cfg.defaultInnerRadius;
    p_->network.defaultOuterRadius = cfg.defaultOuterRadius;
}

PipeFluidScene::~PipeFluidScene() = default;
PipeFluidScene::PipeFluidScene(PipeFluidScene&&) noexcept = default;
PipeFluidScene& PipeFluidScene::operator=(PipeFluidScene&&) noexcept = default;

// ---- Builder ----------------------------------------------------------------

void PipeFluidScene::beginNetwork(const Vec3& start, const Vec3& dir) {
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
    p_->network.segments.clear();
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
    voxer.cellSize = p_->cfg.cellSize;
    voxer.padding  = p_->cfg.padding;
    p_->voxels = voxer.voxelize(p_->network);

    // 2. Build the uint8_t solid mask in simulator order.
    voxelGridToSolidMask(p_->voxels, p_->solidMask);

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
    const float DT = (p_->cfg.dt > 0.f) ? p_->cfg.dt : 1.0f / 60.0f;

    if (p_->cfg.enableSmoke) {
        if (!p_->smoke) {
            p_->smoke = std::make_unique<MACSmoke3D>(NX, NY, NZ, DX, DT);
        } else {
            p_->smoke->reset(NX, NY, NZ, DX, DT);
        }
        applySolidsToSmoke(*p_->smoke, p_->solidMask);
    } else {
        p_->smoke.reset();
    }

    if (p_->cfg.enableWater) {
        if (!p_->water) {
            p_->water = std::make_unique<MACWater3D>(NX, NY, NZ, DX, DT);
        } else {
            p_->water->reset(NX, NY, NZ, DX, DT);
        }
        applySolidsToWater(*p_->water, p_->solidMask);
    } else {
        p_->water.reset();
    }

    p_->geometryDirty = false;
}

// ---- Step loop --------------------------------------------------------------

void PipeFluidScene::step(float dtOverride) {
    if (p_->geometryDirty) rebuild();
    const float dt = (dtOverride > 0.f) ? dtOverride : p_->cfg.dt;
    if (p_->smoke) {
        if (dt > 0.f) p_->smoke->setDt(dt);
        p_->smoke->step();
    }
    if (p_->water) {
        if (dt > 0.f) p_->water->setDt(dt);
        p_->water->step();
    }
}

void PipeFluidScene::resetFluids() {
    // Preserve geometry; just clear fluid state and re-push the solid mask.
    if (p_->smoke) {
        p_->smoke->reset();
        applySolidsToSmoke(*p_->smoke, p_->solidMask);
    }
    if (p_->water) {
        p_->water->reset();
        applySolidsToWater(*p_->water, p_->solidMask);
    }
}

// ---- Source helpers ---------------------------------------------------------

void PipeFluidScene::addSmokeSourceSphere(const Vec3& centre, float radius,
                                          float amount, const Vec3& velocity) {
    if (!p_->smoke) return;
    // The smoke sim's coordinate origin is the voxel grid's world origin.
    // Convert world-space coords to sim-space before passing them in.
    const Vec3 simC = centre - p_->voxels.origin;
    p_->smoke->addSmokeSourceSphere(
        toSimVec3<MACSmoke3D::Vec3>(simC),
        radius, amount,
        toSimVec3<MACSmoke3D::Vec3>(velocity));
}

void PipeFluidScene::addWaterSourceSphere(const Vec3& centre, float radius,
                                          const Vec3& velocity) {
    if (!p_->water) return;
    const Vec3 simC = centre - p_->voxels.origin;
    p_->water->addWaterSourceSphere(
        toSimVec3<MACWater3D::Vec3>(simC),
        radius,
        toSimVec3<MACWater3D::Vec3>(velocity));
}

// ---- Accessors --------------------------------------------------------------

const PipeNetwork& PipeFluidScene::network() const { return p_->network; }
PipeNetwork&       PipeFluidScene::network()       { p_->geometryDirty = true; return p_->network; }
const VoxelGrid&   PipeFluidScene::voxels()  const { return p_->voxels; }
const TriMesh&     PipeFluidScene::pipeMesh() const { return p_->mesh; }

MACSmoke3D* PipeFluidScene::smoke() { return p_->smoke.get(); }
MACWater3D* PipeFluidScene::water() { return p_->water.get(); }

const PipeFluidScene::Config& PipeFluidScene::config() const { return p_->cfg; }
void PipeFluidScene::setConfig(const Config& c) {
    p_->cfg = c;
    p_->network.defaultInnerRadius = c.defaultInnerRadius;
    p_->network.defaultOuterRadius = c.defaultOuterRadius;
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
        if (c == VoxelType::Solid) ++s.solidCells;
        else if (c == VoxelType::Fluid) ++s.fluidCells;
    }
    s.totalPipeLength = p_->network.totalLength();
    s.segmentCount = static_cast<int>(p_->network.numSegments());
    return s;
}

} // namespace pipe_fluid
