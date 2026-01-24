#pragma once
#include <vector>
#include <cstdint>

struct MAC2D;

struct SmokeRenderSettings {
    bool useColor = false;

    float alphaGamma   = 0.70f;
    float alphaScale   = 1.00f;
    float tempStrength = 0.75f;
    float ageGray      = 0.65f;
    float ageDarken    = 0.55f;
    float coreDark     = 0.75f;
};

struct OverlaySettings {
    bool showDiv  = false;
    bool showVort = false;
    float divScale  = 8.0f;
    float divAlpha  = 0.75f;
    float vortScale = 8.0f;
    float vortAlpha = 0.75f;
};

class SmokeRenderer {
public:
    SmokeRenderer(int w, int h);
    ~SmokeRenderer();

    void resize(int w, int h); // optional
    // update texture content from sim (reads sim through its const API)
    void updateFromSim(const MAC2D& sim,
                       const SmokeRenderSettings& smoke,
                       const OverlaySettings& ov);

    unsigned int smokeTex() const { return m_smokeTex; }
    unsigned int divTex()   const { return m_divTex; }
    unsigned int vortTex()  const { return m_vortTex; }

private:
    int m_w = 0, m_h = 0;
    unsigned int m_smokeTex = 0;
    unsigned int m_divTex = 0;
    unsigned int m_vortTex = 0;

    // helpers
    unsigned int makeTexture(int w, int h);
    void uploadSmokeRGBA(const std::vector<float>& smoke,
                         const std::vector<float>& temp,
                         const std::vector<float>& age,
                         const std::vector<uint8_t>& solid,
                         const SmokeRenderSettings& s);
    void uploadDivOverlay(const std::vector<float>& div,
                          const std::vector<uint8_t>& solid,
                          float scale, float alpha);
    void uploadVortOverlay(const std::vector<float>& omega,
                           const std::vector<uint8_t>& solid,
                           float scale, float alpha);
};