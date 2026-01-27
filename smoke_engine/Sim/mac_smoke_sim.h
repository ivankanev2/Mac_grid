#pragma once
#include <vector>
#include <cstdint>
#include <algorithm>
#include "mac_grid_core.h"

struct MAC2D : public MACGridCore {
    std::vector<float> smoke, smoke0;
    std::vector<float> temp, temp0;
    std::vector<float> age,  age0;

    MAC2D(int NX, int NY, float DX, float DT);
    void reset();
    void step(float vortEps);

    void addSolidCircle(float cx, float cy, float r);
    void addSmokeSource(float cx, float cy, float radius, float amount);
    void addHeatSource(float cx, float cy, float radius, float amount);
    void updateAge(float dtLocal);
    void addVorticityConfinement(float eps);
    void addVelocityImpulse(float cx, float cy, float radius, float strength);
    void computeVorticity(std::vector<float>& outOmega) const;

    void advectScalar(std::vector<float>& phi,
                      std::vector<float>& phi0,
                      float dissipation);

    float smokeL2Diff(const std::vector<float>& a, const std::vector<float>& b) const;
    float compareAdvectors(float dissipation);

    bool useMacCormack = true;
    bool openTop = true;

    void setOpenTop(bool on);

    float smokeDissipation = 0.999f;
    float tempDissipation  = 0.995f;

    float velDamping       = 0.5f;

    float ambientTemp      = 0.0f;
    float tempCoolRate     = 0.3f;
    float tempDiffusivity  = 0.0f;


    // this is the wall of doom, may god help us
    // implicit diffusion (engineering realism + stability) ----
    // please dido test these values to see how the smoke changes
    // Units (if dx is meters, dt is seconds):
    //  - viscosity [m^2/s] (air ~ 1.5e-5)
    //  - diffusivity [m^2/s]
    float viscosity        = 0.0f;     // start 0, then try 1e-4 .. 5e-3 depending on dx
    float smokeDiffusivity = 0.0f;     // start 0, then try 1e-5 .. 1e-3

    int   diffuseIters     = 20;       // 10–40 typical
    float diffuseOmega     = 0.8f;     // 0.6–0.9 typical (weighted Jacobi)
    
    const std::vector<float>& density() const { return smoke; }
    const std::vector<float>& temperature() const { return temp; }
    const std::vector<float>& ageField() const { return age; }
    const std::vector<float>& divergence() const { return div; }
    const std::vector<uint8_t>& solidMask() const { return solid; }

    // valve
    void setValveOpen(bool open) { valveOpen = open; }
    void recomputeValveIndices();
    bool isValveOpen() const { return valveOpen; }
    void setValveSpan(int i0, int i1) { valveI0 = std::max(0, i0); valveI1 = std::min(nx-1, i1); }

    float inletSpeed = 1.0f;
    float inletSmoke = 1.0f;
    float inletTemp  = 1.0f;

    struct PipePolyline {
        std::vector<float> x;
        std::vector<float> y;
        float radius = 0.08f;
        float wall   = 0.02f;
        bool  closed = false;
        bool  enabled = true;
    };

    PipePolyline pipe;

    void rebuildSolidsFromPipe(bool clearInterior = false);
    void clearPipe();
    void enforceBoundaries();

private:
    void addForces(float buoyancy, float gravity);
    void applyBoundary();
    void coolAndDiffuseTemperature();

    void diffuseVelocityImplicit();
    void diffuseScalarImplicit(std::vector<float>& phi,
                            std::vector<float>& tmp,
                            float diffusivity,
                            float solidVal);


    bool valveOpen = false;
    int valveI0 = 0;
    int valveI1 = 0;

    inline bool inValve(int i) const { return valveOpen && (i >= valveI0 && i <= valveI1); }

    void applyValveBC();
    void applyValveSink();

    static float distPointToSegment(float px,float py, float ax,float ay, float bx,float by);
    float distPointToPolyline(float px,float py) const;
};
