// mac_smoke_sim.h
#pragma once
#include <vector>
#include <cstdint>

struct MAC2D {
    int nx, ny;
    float dx, dt;

    std::vector<float> u, v, p, smoke;
    std::vector<float> u0, v0, smoke0;
    std::vector<float> div, rhs;
    std::vector<uint8_t> solid;
    MAC2D(int NX, int NY, float DX, float DT);
    void step();
    void reset();

    void addSolidCircle(float cx, float cy, float r);
    void addSmokeSource(float cx, float cy, float radius, float amount);

    float maxAbsDiv() const;

    inline int idxP(int i,int j) const { return i + nx*j; }
    // allow renderer to read density (smoke)
    const std::vector<float>& density() const { return smoke; }

private:
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
    void solvePressure(int iters);
    void project();
    void advectSmoke(float dissipation);
};