#pragma once
#include "mac_water_sim.h"
#include <vector>
#include <algorithm>
#include <cmath>

// forward declare (we only need it for syncSolidsFrom signature)
struct MAC2D;

// Coupled sim: MACWater (particles + liquid projection) + smoke-like scalars
struct MACCoupledSim : public MACWater
{
    // smoke-like fields (cell centered)
    std::vector<float> smoke, temp, age;
    std::vector<float> smoke0, temp0, age0; // advect buffers

    // ----- Inlet / valve controls (match MAC2D UI) -----
    bool  valveOpen  = false;
    float inletSpeed = 1.0f;
    float inletSmoke = 1.0f;
    float inletTemp  = 0.0f;

    // ---- Smoke advection controls (to match MAC2D UI) ----
    bool  useMacCormack    = true;   // default same as your smoke sim
    float smokeDissipation = 0.995f;
    float tempDissipation  = 0.995f;

    // --- Multiphase coupling knobs ---
    // Used by the coupled pressure projection as cell densities.
    // (rhoWater >> rhoAir gives much nicer smoke/water interaction.)
    float rhoAir   = 1.0f;
    float rhoWater = 1000.0f;

    // ----- Pipe editing (match MAC2D UI) -----
    struct Pipe {
        float radius = 0.08f; // inner radius
        float wall   = 0.03f; // wall thickness
        std::vector<float> x;
        std::vector<float> y;
    } pipe;

    MACCoupledSim(int NX, int NY, float DX, float DT);

    // UI expected
    void setValveOpen(bool v);
    bool isValveOpen() const { return valveOpen; }

    void setOpenTop(bool v);        // updates both MACWater::openTop and MACGridCore BC
    bool getOpenTop() const { return openTop; }

    void clearPipe();

    // UI calls these names
    void rebuildSolidsFromPipe(bool keepBoundaries);
    void enforceBoundaries();               // wraps MACWater boundary + valve/scalars
    void invalidatePressureMatrix();        // wrap core function for UI

    void applyBoundary() override;

    void reset();

    void addSolidCircle(float cx, float cy, float r);
    void eraseSolidCircle(float cx, float cy, float r);

    void syncSolidsFrom(const MAC2D& smokeSim);

    // main coupled step (called by main.cpp as stepCoupled(ui.vortEps))
    void stepCoupled(float vortEps);

    // ---- SmokeRenderer expects the same API as MAC2D ----
    const std::vector<float>& density() const     { return smoke; }
    const std::vector<float>& temperature() const { return temp;  }
    const std::vector<float>& ageField() const    { return age;   }

    // Renderer uses this to know where solids are.
    // MACWater stores solids in `solid` as uint8_t, same size nx*ny.
    const std::vector<uint8_t>& solidMask() const { return solid; }

private:

     // --- Top-vent valve geometry (like smoke sim has a fixed vent band) ---
    int valve_i0 = 0;   // inclusive
    int valve_i1 = 0;   // exclusive

    void recomputeValveIndices();
    int  borderBt() const;          // match MACWater border thickness clamp
    bool inValveBand(int i) const { return (i >= valve_i0 && i < valve_i1); }

    void carveTopVentSolids();      // NEW: keep a permanent opening through the border solids
    int  valveSourceJ() const;      // NEW: where to inject scalars (just under border)

    void resizeSmokeFields();
    void applyValveScalars();
    void applyValveVelocityBC();

    void carveTopValveOpening();

    void buildInvRhoFaceWeights();

    // pipe distance helper
    static float distPtSegSq(float px, float py, float ax, float ay, float bx, float by);
};