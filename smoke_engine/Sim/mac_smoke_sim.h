#pragma once
#include <vector>
#include <cstdint>
#include <algorithm> 

struct MAC2D {
    int nx, ny;
    float dx, dt;

    std::vector<float> u, v, p, smoke;
    std::vector<float> u0, v0, smoke0;
    std::vector<float> div, rhs;
    std::vector<uint8_t> solid;

    std::vector<float> temp, temp0;
    std::vector<float> age,  age0;
    
    const std::vector<float>& temperature() const { return temp; }
    const std::vector<float>& ageField()   const { return age;  }

    // Renderer-friendly read-only accessors
    const std::vector<float>& density() const { return smoke; }
    const std::vector<float>& divergence() const { return div; }
    const std::vector<uint8_t>& solidMask() const { return solid; }

    MAC2D(int NX, int NY, float DX, float DT);
    void step(float vortEps);
    void reset();

    void addSolidCircle(float cx, float cy, float r);
    void addSmokeSource(float cx, float cy, float radius, float amount);
    void addHeatSource(float cx, float cy, float radius, float amount);
    void updateAge(float dtLocal);
    void addVorticityConfinement(float eps);
    void addVelocityImpulse(float cx, float cy, float radius, float strength);
    void computeVorticity(std::vector<float>& outOmega) const;

    void advectScalarMacCormack(std::vector<float>& phi,
                            std::vector<float>& phi0,
                            float dissipation);

    bool useMacCormack = true; 

    bool openTop = true; // outflow at the top boundary

    void setOpenTop(bool on);

    void advectScalar(std::vector<float>& phi,
                  std::vector<float>& phi0,
                  float dissipation);

    void advectScalarSemiLagrangian(std::vector<float>& phi,
                                std::vector<float>& phi0,
                                float dissipation);
    float smokeL2Diff(const std::vector<float>& a, const std::vector<float>& b) const;
    float compareAdvectors(float dissipation);

    float maxAbsDiv() const;
    float maxFaceSpeed() const;
    void setDt(float newDt) { dt = newDt; }

    float smokeDissipation = 0.999f;
    float tempDissipation  = 0.995f;

    // NEW: temperature physics
    float ambientTemp      = 0.0f;   // reference temperature
    float tempCoolRate     = 0.3f;   // 1/seconds, tweak in UI later
    float tempDiffusivity  = 0.0f;   // e.g. 0.1f if you want diffusion

    inline int idxP(int i,int j) const { return i + nx*j; }


    // valve shit
    void setValveOpen(bool open) { valveOpen = open; }
    void recomputeValveIndices();
    bool isValveOpen() const { return valveOpen; }

     // valve span on the bottom boundary (inclusive i0, inclusive i1)
    void setValveSpan(int i0, int i1) { valveI0 = std::max(0, i0); valveI1 = std::min(nx-1, i1); }

    // inlet parameters (pipe feed)
    float inletSpeed = 1.0f;      // +up into domain (tweak)
    float inletSmoke = 1.0f;      // target smoke density at inlet
    float inletTemp  = 1.0f;      // target temp at inlet (0..1)


    struct PipePolyline {
    std::vector<float> x; // normalized 0..1
    std::vector<float> y; // normalized 0..1
    float radius = 0.08f;        // inner radius (normalized)
    float wall   = 0.02f;        // wall thickness (normalized)
    bool  closed = false;        // optional (not needed now)
    bool  enabled = true;
};

PipePolyline pipe;

// call after editing pipe points or radius/wall
void rebuildSolidsFromPipe(bool clearInterior = false);
void clearPipe();

void enforceBoundaries();


private:
        // it is very important every time you use idxU in the future to use it as nx+1
    inline int idxU(int i,int j) const { return i + (nx+1)*j; }
    inline int idxV(int i,int j) const { return i + nx*j; }
    inline bool isSolid(int i,int j) const { return solid[idxP(i,j)] != 0; }

    static inline float clampf(float x,float a,float b) {
        return std::max(a, std::min(b, x));
    }

    void worldToCell(float x, float y, int& i, int& j) const;
    float sampleCellCentered(const std::vector<float>& f, float x, float y) const;
    float sampleU(const std::vector<float>& fu, float x, float y) const;
    float sampleV(const std::vector<float>& fv, float x, float y) const;
    void velAt(float x, float y,
               const std::vector<float>& fu,
               const std::vector<float>& fv,
               float& outUx, float& outVy) const;
    void addForces(float buoyancy, float gravity);
    void applyBoundary();
    void advectVelocity();
    void computeDivergence();
    void solvePressurePCG(int maxIters = 200, float tol = 1e-6f);
    void applyLaplacian(const std::vector<float>& x, std::vector<float>& Ax) const;
    void project();
    void advectSmoke(float dissipation);
    void coolAndDiffuseTemperature();

    // valve shit but private
    bool valveOpen = false;
    int valveI0 = 0;
    int valveI1 = 0;

    inline bool inValve(int i) const { return valveOpen && (i >= valveI0 && i <= valveI1); }

    void applyValveBC();       // new (not so new anymore)
    void applyValveSink();     // new (for outflow “leaving”)

    static float distPointToSegment(float px,float py, float ax,float ay, float bx,float by);
    float distPointToPolyline(float px,float py) const;
};