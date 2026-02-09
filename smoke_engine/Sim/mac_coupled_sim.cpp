#include "mac_coupled_sim.h"
#include "mac_smoke_sim.h" // for MAC2D definition (syncSolidsFrom needs smokeSim.solid)
#include <algorithm>
#include <cmath>

static inline float clamp01(float x) { return x < 0 ? 0 : (x > 1 ? 1 : x); }

// ---- ctor ----
MACCoupledSim::MACCoupledSim(int NX, int NY, float DX, float DT)
: MACWater(NX, NY, DX, DT)
{
    resizeSmokeFields();
    // Make core BC reflect openTop
    setOpenTopBC(openTop);
    // Ensure initial solids mask is built
    rebuildSolidsFromUser();
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

// ---- valve helpers ----
void MACCoupledSim::applyValveVelocityBC()
{
    if (!valveOpen) return;

    // mirror your smoke sim convention: valve on left boundary, middle band
    const int iFace = 0;
    const int j0 = ny / 4;
    const int j1 = 3 * ny / 4;

    for (int j = j0; j < j1; ++j) {
        const int cell = idxP(0, j);
        if (solid[(size_t)cell]) continue;
        u[(size_t)idxU(iFace, j)] = inletSpeed;
    }
}

void MACCoupledSim::applyValveScalars()
{
    if (!valveOpen) return;

    // inject into leftmost column cells
    const int i = 0;
    const int j0 = ny / 4;
    const int j1 = 3 * ny / 4;

    for (int j = j0; j < j1; ++j) {
        const int id = idxP(i, j);
        if (solid[(size_t)id]) continue;
        smoke[(size_t)id] = inletSmoke;
        temp [(size_t)id] = inletTemp;
        age  [(size_t)id] = 0.0f;
    }
}

// ---- boundaries wrapper ----
void MACCoupledSim::enforceBoundaries()
{
    // base water BC (no-slip solids + domain)
    MACWater::applyBoundary();

    // overwrite valve inflow on u if open
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

    // inject scalars if valve is open
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

// ---- main coupled step ----
void MACCoupledSim::stepCoupled(float /*vortEps*/)
{
    // Run the water sim step (particles + pressure + velocity)
    // Uses dt already set from main via coupled.setDt(dt).
    MACWater::step();

    // Then advect smoke-like scalars using the resulting grid velocity
    advectScalar(*this, temp,  temp0,  tempDissipation);
    advectScalar(*this, smoke, smoke0, smokeDissipation);
    advectScalar(*this, age,   age0,   1.0f);

    // age increments
    for (float& a : age) a += dt;

    // enforce coupled-specific boundary/scalar rules
    enforceBoundaries();
}