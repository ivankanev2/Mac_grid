
#include "mac_coupled_sim.h"
#include <algorithm>
#include <cmath>

static inline float clamp01(float x) { return x < 0 ? 0 : (x > 1 ? 1 : x); }

MACCoupledSim::MACCoupledSim(int NX, int NY, float DX, float DT)
: MACWater(NX, NY, DX, DT)
{
    reset();
}

void MACCoupledSim::reset()
{
    MACWater::reset();

    const size_t Nc = (size_t)nx * (size_t)ny;
    smoke.assign(Nc, 0.0f); smoke0.assign(Nc, 0.0f);
    temp .assign(Nc, 0.0f); temp0 .assign(Nc, 0.0f);
    age  .assign(Nc, 0.0f); age0  .assign(Nc, 0.0f);

    // Ensure pressure domain matches coupled intent
    setOpenTopBC(openTop);
    setFluidMaskAllNonSolid();
    syncSolidsToFluidAndFaces();
    invalidatePressureMatrix();
}

static inline void advectScalar(MACCoupledSim& s,
                                std::vector<float>& phi,
                                std::vector<float>& phi0,
                                float dissipation)
{
    if (s.useMacCormack) s.advectScalarMacCormack(phi, phi0, dissipation);
    else                 s.advectScalarSemiLagrangian(phi, phi0, dissipation);
}

static inline void addBuoyancyAirOnly(MACCoupledSim& s)
{
    const float g  = s.gravity_g;
    const float T0 = std::max(1e-3f, s.ambientTempK);

    for (int j = 0; j <= s.ny; ++j) {
        for (int i = 0; i < s.nx; ++i) {
            // Face center of v
            float x = (i + 0.5f) * s.dx;
            float y = (j) * s.dx;

            // Sample temp (your smoke temp is currently “delta-K” in MAC2D)
            float dT = s.sampleCellCentered(s.temp, x, y);
            float ab = g * (dT / T0) * s.buoyancyScale;

            // Optional: suppress buoyancy inside liquid (simple)
            // Use nearest cell below face as a cheap mask
            int ci = std::max(0, std::min(i, s.nx - 1));
            int cj = std::max(0, std::min(j - 1, s.ny - 1));
            bool inLiquid = (cj >= 0 && cj < s.ny) ? (s.liquid[(size_t)s.idxP(ci, cj)] != 0) : false;
            if (!inLiquid) {
                s.v[(size_t)s.idxV(i, j)] += s.dt * ab;
            }
        }
    }
}

void MACCoupledSim::stepCoupled(float /*vortEps*/)
{
    // --- 0) solids / BC ---
    rebuildSolidsFromUser();
    removeParticlesInSolids();
    enforceParticleBounds();

    // Keep MACGridCore open-top consistent with water’s openTop toggle
    setOpenTopBC(openTop);

    // --- 1) preserve “air” velocity where no water particles contribute ---
    std::vector<float> uAir = u;
    std::vector<float> vAir = v;

    // --- 2) water momentum -> grid ---
    if (!particles.empty()) {
        particleToGrid();      // fills u/v and uWeight/vWeight, then applyBoundary()
        buildLiquidMask();

        // Blend: if face has no particle weight, keep previous air velocity
        for (size_t k = 0; k < u.size(); ++k)
            if (uWeight[k] <= 1e-6f) u[k] = uAir[k];
        for (size_t k = 0; k < v.size(); ++k)
            if (vWeight[k] <= 1e-6f) v[k] = vAir[k];
    } else {
        // no water: just air sim
        buildLiquidMask(); // keep consistent (all zero)
    }

    applyBoundary();

    // --- 3) forces on shared velocity ---
    // Water gravity (exactly like MACWater::step does, but keep it here)
    if (waterGravity != 0.0f) {
        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const int id = idxV(i, j);
                const bool botL = (j - 1 >= 0) ? (liquid[(size_t)idxP(i, j - 1)] != 0) : false;
                const bool topL = (j < ny)     ? (liquid[(size_t)idxP(i, j)]     != 0) : false;
                if (botL || topL) v[(size_t)id] += dt * waterGravity;
            }
        }
    }

    // Smoke buoyancy (air-only masking)
    addBuoyancyAirOnly(*this);

    // Velocity damping (reuse water’s knob)
    if (velDamping > 0.0f) {
        const float damp = std::exp(-velDamping * dt);
        for (float& uu : u) uu *= damp;
        for (float& vv : v) vv *= damp;
    }

    applyBoundary();

    // Optional viscosity: start enabled if you want (you already have stable implicit diffusion)
    diffuseVelocityImplicit();
    applyBoundary();

    // --- 4) one pressure solve (air + liquid) ---
    // IMPORTANT: coupled domain = all non-solid
    setFluidMaskAllNonSolid();
    syncSolidsToFluidAndFaces();
    invalidatePressureMatrix(); // safe for now; later you can make it smarter
    // Save pre-projection for FLIP
    uPrev = u;
    vPrev = v;

    project();
    applyBoundary();

    // FLIP delta
    for (size_t k = 0; k < u.size(); ++k) uDelta[k] = u[k] - uPrev[k];
    for (size_t k = 0; k < v.size(); ++k) vDelta[k] = v[k] - vPrev[k];

    // --- 5) water update ---
    if (!particles.empty()) {
        extrapolateVelocity();
        applyBoundary();

        gridToParticles();
        advectParticles();
        enforceParticleBounds();
        removeParticlesInSolids();

        // keep your reseed/relax pipeline
        {
            const int savedDil = maskDilations;
            maskDilations = 0;
            buildLiquidMask();
            maskDilations = savedDil;
        }
        reseedParticles();
        relaxParticles(2, 0.5f);

        applyDissipation();
        rasterizeWaterField();
    } else {
        rasterizeWaterField();
    }

    // --- 6) smoke scalars (advected by the same projected velocity) ---
    advectScalar(*this, temp,  temp0,  tempDissipation);
    advectScalar(*this, smoke, smoke0, smokeDissipation);
    advectScalar(*this, age,   age0,   1.0f);

    // simple cooling + age increment
    if (tempCoolRate > 0.0f) {
        float k = std::max(0.0f, 1.0f - tempCoolRate * dt);
        for (float& T : temp) T *= k;
    }
    for (float& a : age) a += dt;

    // clear smoke in solids
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
            if (isSolid(i,j)) {
                const int id = idxP(i,j);
                smoke[(size_t)id] = 0.0f;
                temp [(size_t)id] = 0.0f;
                age  [(size_t)id] = 0.0f;
            }
}