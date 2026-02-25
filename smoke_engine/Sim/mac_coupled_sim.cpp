#include "mac_coupled_sim.h"
#include "mac_smoke_sim.h" // for MAC2D definition (syncSolidsFrom needs smokeSim.solid)
#include <algorithm>
#include <cmath>

static inline float clamp01(float x) { return x < 0 ? 0 : (x > 1 ? 1 : x); }

static inline int clampi(int x, int a, int b) { return x < a ? a : (x > b ? b : x); }

static inline float smokeAt(const MACCoupledSim& s, int i, int j)
{
    i = std::max(0, std::min(s.nx - 1, i));
    j = std::max(0, std::min(s.ny - 1, j));
    return s.smoke[(size_t)s.idxP(i, j)];
}

int MACCoupledSim::borderBt() const
{
    // Match MACWater::rebuildSolidsFromUser() logic (Water/water_solids.h)
    const int maxBt = std::max(1, (std::min(nx, ny) / 2) - 1);
    return clampi(borderThickness, 1, maxBt);
}

void MACCoupledSim::recomputeValveIndices()
{
    // Use a stable “middle band” like your smoke sim convention
    // (You can tweak width; this works well visually.)
    const int w = std::max(2, nx / 6);
    const int c = nx / 2;
    valve_i0 = clampi(c - w / 2, 0, nx);
    valve_i1 = clampi(valve_i0 + w, 0, nx);
    if (valve_i1 < valve_i0) std::swap(valve_i0, valve_i1);
}

void MACCoupledSim::carveTopVentSolids()
{
    // The water sim builds a thick solid border at the top when openTop == false.
    // We want a *permanent* vent hole through that border in the valve band.
    const int bt = borderBt();
    for (int t = 0; t < bt; ++t) {
        const int j = ny - 1 - t;
        if (j < 0) continue;
        for (int i = valve_i0; i < valve_i1; ++i) {
            solid[(size_t)idxP(i, j)] = 0;
        }
    }
}

int MACCoupledSim::valveSourceJ() const
{
    // Inject scalars just *below* the carved border thickness.
    // Example: bt=2 => inject at ny-1-2.
    const int j = ny - 1 - borderBt();
    return clampi(j, 0, ny - 1);
}

void MACCoupledSim::reset()
{
    MACWater::reset();
    resizeSmokeFields();

    // Rebuild solids + keep permanent vent carved
    rebuildSolidsFromUser();
    carveTopVentSolids();

    invalidatePressureMatrix();
    enforceBoundaries();
}

// ---- ctor ----
MACCoupledSim::MACCoupledSim(int NX, int NY, float DX, float DT)
: MACWater(NX, NY, DX, DT)
{
    resizeSmokeFields();
    recomputeValveIndices();
    // Make core BC reflect openTop
    setOpenTopBC(openTop);
    // Ensure initial solids mask is built
    rebuildSolidsFromUser();
    carveTopVentSolids();
    enforceBoundaries();
}

// ---- sizing ----
void MACCoupledSim::resizeSmokeFields()
{
    const size_t Nc = (size_t)nx * (size_t)ny;
    smoke.assign(Nc, 0.0f);
    temp .assign(Nc, 0.0f);
    age  .assign(Nc, 0.0f);
    smoke0.assign(Nc, 0.0f);
    temp0 .assign(Nc, 0.0f);
    age0  .assign(Nc, 0.0f);
}

// ---- UI API ----
void MACCoupledSim::invalidatePressureMatrix()
{
    // this is MACGridCore::invalidatePressureMatrix()
    MACGridCore::invalidatePressureMatrix();
}

void MACCoupledSim::setOpenTop(bool v)
{
    openTop = v;        // MACWater toggle
    setOpenTopBC(v);    // MACGridCore toggle used by the solver
}

void MACCoupledSim::setValveOpen(bool v)
{
    valveOpen = v;
}

void MACCoupledSim::clearPipe()
{
    pipe.x.clear();
    pipe.y.clear();
}

void MACCoupledSim::applyBoundary()
{
    // IMPORTANT: water step rebuilds border solids; carve the valve opening every time
    carveTopValveOpening();

    // base water BC (no-slip solids + domain)
    MACWater::applyBoundary();

    // overwrite valve inflow on top v-face if open
    applyValveVelocityBC();

    // clear scalars in solids
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int id = idxP(i, j);
            if (solid[(size_t)id]) {
                smoke[(size_t)id] = 0.0f;
                temp [(size_t)id] = 0.0f;
                age  [(size_t)id] = 0.0f;
            }
        }
    }

    // inject scalars (top band) if valve is open
    applyValveScalars();
}

void MACCoupledSim::addSolidCircle(float cx, float cy, float r)
{
    // cx,cy,r are in normalized sim-space [0..1] like MAC2D painting.
    if (solidUser.empty()) return;

    cx = std::max(0.0f, std::min(1.0f, cx));
    cy = std::max(0.0f, std::min(1.0f, cy));
    r  = std::max(0.0f, r);

    const float r2 = r * r;

    // Bounding box in cell indices (safe clamp)
    int i0 = (int)std::floor((cx - r) * nx);
    int i1 = (int)std::floor((cx + r) * nx);
    int j0 = (int)std::floor((cy - r) * ny);
    int j1 = (int)std::floor((cy + r) * ny);

    i0 = std::max(0, std::min(nx - 1, i0));
    i1 = std::max(0, std::min(nx - 1, i1));
    j0 = std::max(0, std::min(ny - 1, j0));
    j1 = std::max(0, std::min(ny - 1, j1));

    for (int j = j0; j <= j1; ++j) {
        for (int i = i0; i <= i1; ++i) {
            const float x = (i + 0.5f) / (float)nx;
            const float y = (j + 0.5f) / (float)ny;
            const float dx = x - cx;
            const float dy = y - cy;
            if (dx*dx + dy*dy <= r2) {
                solidUser[(size_t)idxP(i, j)] = 1;
            }
        }
    }

    rebuildSolidsFromUser();
    invalidatePressureMatrix();
    enforceBoundaries();
}

void MACCoupledSim::eraseSolidCircle(float cx, float cy, float r)
{
    if (solidUser.empty()) return;

    cx = std::max(0.0f, std::min(1.0f, cx));
    cy = std::max(0.0f, std::min(1.0f, cy));
    r  = std::max(0.0f, r);

    const float r2 = r * r;

    int i0 = (int)std::floor((cx - r) * nx);
    int i1 = (int)std::floor((cx + r) * nx);
    int j0 = (int)std::floor((cy - r) * ny);
    int j1 = (int)std::floor((cy + r) * ny);

    i0 = std::max(0, std::min(nx - 1, i0));
    i1 = std::max(0, std::min(nx - 1, i1));
    j0 = std::max(0, std::min(ny - 1, j0));
    j1 = std::max(0, std::min(ny - 1, j1));

    for (int j = j0; j <= j1; ++j) {
        for (int i = i0; i <= i1; ++i) {
            const float x = (i + 0.5f) / (float)nx;
            const float y = (j + 0.5f) / (float)ny;
            const float dx = x - cx;
            const float dy = y - cy;
            if (dx*dx + dy*dy <= r2) {
                solidUser[(size_t)idxP(i, j)] = 0;
            }
        }
    }

    rebuildSolidsFromUser();
    invalidatePressureMatrix();
    enforceBoundaries();
}



// ---- valve helpers ----
void MACCoupledSim::applyValveVelocityBC()
{
    if (!valveOpen) return;

    // push DOWN into the domain
    for (int i = valve_i0; i < valve_i1; ++i) {
        if (solid[(size_t)idxP(i, ny - 1)]) continue;

        // boundary face
        v[(size_t)idxV(i, ny)] = -inletSpeed;

        // IMPORTANT: interior face between (ny-2) and (ny-1)
        if (ny - 1 >= 1) {
            v[(size_t)idxV(i, ny - 1)] = -inletSpeed;
        }
    }
}

void MACCoupledSim::applyValveScalars()
{
    if (!valveOpen) return;

    const int j = valveSourceJ();

    for (int i = valve_i0; i < valve_i1; ++i) {
        const int id = idxP(i, j);
        if (solid[(size_t)id]) continue;

        smoke[(size_t)id] = inletSmoke;
        temp [(size_t)id] = inletTemp;
        age  [(size_t)id] = 0.0f;
    }
}

void MACCoupledSim::carveTopValveOpening()
{
    if (!valveOpen) return;

    // Same style as smoke sim: a centered band
    int i0 = (int)(0.45f * nx);
    int i1 = (int)(0.55f * nx);
    i0 = std::max(0, std::min(i0, nx - 1));
    i1 = std::max(0, std::min(i1, nx - 1));
    if (i1 < i0) std::swap(i0, i1);

    // Water uses borderThickness walls; carve through that thickness
    const int bt = std::max(1, borderThickness);
    const int jStart = std::max(0, ny - bt);
    const int jEnd   = ny - 1;

    for (int j = jStart; j <= jEnd; ++j) {
        for (int i = i0; i <= i1; ++i) {
            const int id = idxP(i, j);
            solid[(size_t)id] = 0;
            if ((size_t)id < solidUser.size())
                solidUser[(size_t)id] = 0;
        }
    }
}

// ---- boundaries wrapper ----
void MACCoupledSim::enforceBoundaries()
{
    // 1) Keep the vent carved through solid border every time solids may rebuild
    carveTopVentSolids();

    // 2) Preserve top-vent v faces so MACWater::applyBoundary() doesn't clamp them to 0
    //    (since openTop is false, MACWater will zero the entire top row otherwise).
    std::vector<float> savedTopV;
    savedTopV.reserve((size_t)(valve_i1 - valve_i0));
    for (int i = valve_i0; i < valve_i1; ++i) {
        savedTopV.push_back(v[(size_t)idxV(i, ny)]);
    }

    // Base water BC (no-slip solids + domain)
    MACWater::applyBoundary();

    // Restore the vent faces (vent exists even when valve closed)
    {
        size_t k = 0;
        for (int i = valve_i0; i < valve_i1; ++i, ++k) {
            v[(size_t)idxV(i, ny)] = savedTopV[k];
        }
    }

    // 3) If valve open, impose inflow velocity and inject scalars
    applyValveVelocityBC();

    // clear scalars in solids
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int id = idxP(i, j);
            if (solid[(size_t)id]) {
                smoke[(size_t)id] = 0.0f;
                temp [(size_t)id] = 0.0f;
                age  [(size_t)id] = 0.0f;
            }
        }
    }

    applyValveScalars();
}

// ---- pipe distance helper ----
float MACCoupledSim::distPtSegSq(float px, float py, float ax, float ay, float bx, float by)
{
    const float abx = bx - ax;
    const float aby = by - ay;
    const float apx = px - ax;
    const float apy = py - ay;
    const float ab2 = abx*abx + aby*aby;

    float t = 0.0f;
    if (ab2 > 1e-12f) t = (apx*abx + apy*aby) / ab2;
    t = std::max(0.0f, std::min(1.0f, t));

    const float cx = ax + t * abx;
    const float cy = ay + t * aby;
    const float dx = px - cx;
    const float dy = py - cy;
    return dx*dx + dy*dy;
}

// ---- solids from pipe (matches UI call name) ----
void MACCoupledSim::rebuildSolidsFromPipe(bool keepBoundaries)
{
    // Rebuild into solidUser (same pattern as MACWater uses)
    std::fill(solidUser.begin(), solidUser.end(), 0);

    if (pipe.x.size() >= 2) {
        const float inner = pipe.radius;
        const float outer = pipe.radius + pipe.wall;
        const float inner2 = inner * inner;
        const float outer2 = outer * outer;

        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const float x = (i + 0.5f) / (float)nx;
                const float y = (j + 0.5f) / (float)ny;

                float best = 1e30f;
                for (size_t k = 0; k + 1 < pipe.x.size(); ++k) {
                    best = std::min(best, distPtSegSq(x, y, pipe.x[k], pipe.y[k], pipe.x[k+1], pipe.y[k+1]));
                }

                // solid ring between inner and outer
                if (best <= outer2 && best >= inner2) {
                    solidUser[(size_t)idxP(i, j)] = 1;
                }
            }
        }
    }

    if (!keepBoundaries) {
        // keep outer border solid (like your smoke sim expects)
        for (int i = 0; i < nx; ++i) {
            solidUser[(size_t)idxP(i, 0)] = 1;
            solidUser[(size_t)idxP(i, ny-1)] = openTop ? 0 : 1;
        }
        for (int j = 0; j < ny; ++j) {
            solidUser[(size_t)idxP(0, j)] = 1;
            solidUser[(size_t)idxP(nx-1, j)] = 1;
        }
    }

    // rebuild solid[] from solidUser
    rebuildSolidsFromUser();

    carveTopVentSolids();
    recomputeValveIndices(); 

    // clear coupled scalars in solids
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int id = idxP(i, j);
            if (solid[(size_t)id]) {
                smoke[(size_t)id] = 0.0f;
                temp [(size_t)id] = 0.0f;
                age  [(size_t)id] = 0.0f;
                water[(size_t)id] = 0.0f;
                liquid[(size_t)id] = 0;
            }
        }
    }

    invalidatePressureMatrix();
    enforceBoundaries();
}

// ---- sync solids painted in smoke sim ----
void MACCoupledSim::syncSolidsFrom(const MAC2D& smokeSim)
{
    // assumes same nx/ny
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int id = idxP(i, j);
            solidUser[(size_t)id] = smokeSim.solid[(size_t)id] ? 1 : 0;
        }
    }
    rebuildSolidsFromUser();
    carveTopVentSolids();
    recomputeValveIndices();
    invalidatePressureMatrix();
    enforceBoundaries();
}

// ---- scalar advection helper (uses MACGridCore methods) ----
static inline void advectScalar(MACCoupledSim& s,
                                std::vector<float>& phi,
                                std::vector<float>& phi0,
                                float dissipation)
{
    if (s.useMacCormack) s.advectScalarMacCormack(phi, phi0, dissipation);
    else                 s.advectScalarSemiLagrangian(phi, phi0, dissipation);
}

void MACCoupledSim::buildInvRhoFaceWeights()
{
    auto cellRho = [&](int i, int j) -> float {
        if (i < 0 || i >= nx || j < 0 || j >= ny) return rhoAir;
        const int id = idxP(i, j);
        if (solid[(size_t)id]) return rhoAir;
        return liquid[(size_t)id] ? rhoWater : rhoAir;
    };
    auto invRhoFace = [&](float rhoA, float rhoB) -> float {
        // 1/rho at face: harmonic mean of rho => 2/(rhoA+rhoB)
        return 2.0f / (rhoA + rhoB);
    };

    // U faces: between (i-1,j) and (i,j)
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i <= nx; ++i) {
            float w = 0.0f;
            const int iL = i - 1;
            const int iR = i;
            if (iL >= 0 && iR < nx) {
                const int idL = idxP(iL, j);
                const int idR = idxP(iR, j);
                if (!solid[(size_t)idL] && !solid[(size_t)idR]) {
                    w = invRhoFace(cellRho(iL, j), cellRho(iR, j));
                }
            }
            faceOpenU[(size_t)idxU(i, j)] = w;
        }
    }

    // V faces: between (i,j-1) and (i,j)
    for (int j = 0; j <= ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            float w = 0.0f;
            const int jB = j - 1;
            const int jT = j;
            if (jB >= 0 && jT < ny) {
                const int idB = idxP(i, jB);
                const int idT = idxP(i, jT);
                if (!solid[(size_t)idB] && !solid[(size_t)idT]) {
                    w = invRhoFace(cellRho(i, jB), cellRho(i, jT));
                }
            }
            faceOpenV[(size_t)idxV(i, j)] = w;
        }
    }
}

// ---- main coupled step ----
void MACCoupledSim::stepCoupled(float /*vortEps*/)
{
    // NOTE: MACWater::step() does a liquid-only projection and re-writes u/v from
    // particles every frame, which kills air motion. For coupled smoke+water we
    // instead keep air velocities alive and run one shared multiphase projection.

    // Always treat the coupled domain as closed-top except the explicit valve band.
    setOpenTop(false);

    rebuildSolidsFromUser();
    carveTopVentSolids();
    recomputeValveIndices();
    removeParticlesInSolids();
    enforceParticleBounds();

    // Preserve air velocity before water transfer.
    std::vector<float> uAir = u;
    std::vector<float> vAir = v;

    // Water transfer -> grid.
    if (!particles.empty()) {
        particleToGrid();
        buildLiquidMask();

        // If no particle contribution on a face, keep the air velocity.
        for (size_t k = 0; k < u.size(); ++k)
            if (uWeight[k] <= 1e-6f) u[k] = uAir[k];
        for (size_t k = 0; k < v.size(); ++k)
            if (vWeight[k] <= 1e-6f) v[k] = vAir[k];
    } else {
        std::fill(liquid.begin(), liquid.end(), 0);
    }

    // Gravity only acts on water (liquid-adjacent v faces).
    if (waterGravity != 0.0f) {
        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const bool botL = (j - 1 >= 0) ? (liquid[(size_t)idxP(i, j - 1)] != 0) : false;
                const bool topL = (j < ny)     ? (liquid[(size_t)idxP(i, j)]     != 0) : false;
                if (botL || topL) v[(size_t)idxV(i, j)] += dt * waterGravity;
            }
        }
    }

    if (velDamping > 0.0f) {
        const float damp = std::exp(-velDamping * dt);
        for (float& uu : u) uu *= damp;
        for (float& vv : v) vv *= damp;
    }
    
    // --- heavy smoke: accelerate air DOWN where smoke exists (but not inside liquid) ---
    const float smokeFall = 6.0f; // tweak: 2..20
    for (int j = 1; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {

            // face between cells (i,j-1) and (i,j)
            const int cB = idxP(i, j - 1);
            const int cT = idxP(i, j);

            if (solid[(size_t)cB] || solid[(size_t)cT]) continue;

            // don't force inside water
            if (liquid[(size_t)cB] || liquid[(size_t)cT]) continue;

            float sAvg = 0.5f * (smokeAt(*this, i, j - 1) + smokeAt(*this, i, j));
            if (sAvg <= 0.0f) continue;

            v[(size_t)idxV(i, j)] += dt * (-smokeFall) * sAvg;
        }
    }

    // Boundary conditions (solids + domain) + valve velocity BC.
    enforceBoundaries();

    // ---------------- shared multiphase projection ----------------
    setFluidMaskAllNonSolid();
    syncSolidsToFluidAndFaces();
    // buildInvRhoFaceWeights();
    invalidatePressureMatrix();

    uPrev = u;
    vPrev = v;
    project();

    for (size_t k = 0; k < u.size(); ++k) uDelta[k] = u[k] - uPrev[k];
    for (size_t k = 0; k < v.size(); ++k) vDelta[k] = v[k] - vPrev[k];

    // ---------------- water particle update ----------------
    if (!particles.empty()) {
        extrapolateVelocity();
        enforceBoundaries();

        gridToParticles();
        advectParticles();
        enforceParticleBounds();
        removeParticlesInSolids();

        {
            const int savedDil = maskDilations;
            maskDilations = 0;
            buildLiquidMask();
            maskDilations = savedDil;
        }
        reseedParticles();
        relaxParticles(2, 0.5f);
        applyDissipation();
    }

    rasterizeWaterField();

    // ---------------- smoke scalars ----------------
    applyValveScalars();
    advectScalar(*this, temp,  temp0,  tempDissipation);
    advectScalar(*this, smoke, smoke0, smokeDissipation);
    advectScalar(*this, age,   age0,   1.0f);
    for (float& a : age) a += dt;

    enforceBoundaries();
}