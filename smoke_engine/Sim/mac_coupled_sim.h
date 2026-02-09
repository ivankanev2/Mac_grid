#pragma once
#include "mac_water_sim.h"   // we will derive from MACWater to reuse water guts
#include <vector>

struct MACCoupledSim : public MACWater
{
    // --- smoke scalars ---
    std::vector<float> smoke, smoke0;
    std::vector<float> temp,  temp0;
    std::vector<float> age,   age0;

    // --- smoke params (match MAC2D’s defaults / UI wiring later) ---
    bool  useMacCormack    = true;
    float smokeDissipation = 0.999f;
    float tempDissipation  = 0.995f;
    float tempCoolRate     = 0.005f;

    // buoyancy (copy the physical form you already use)
    float gravity_g       = 9.81f;
    float ambientTempK    = 293.15f;
    float buoyancyScale   = 1.0f;

    MACCoupledSim(int NX, int NY, float DX, float DT);

    void reset();
    void stepCoupled(float vortEps /*can be ignored in v0*/);

    // renderer expects these names on MAC2D; we provide same accessors
    const std::vector<float>& density()     const { return smoke; }
    const std::vector<float>& temperature() const { return temp;  }
    const std::vector<float>& ageField()    const { return age;   }
};