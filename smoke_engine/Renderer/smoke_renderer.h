#pragma once
#include <vector>
#include <cstdint>

struct MAC2D;
struct MACWater;
struct MACSmoke3D;

struct SmokeRenderSettings {
    bool useColor = false;
    // When true the texture background is fully transparent (alpha=0) and
    // only smoke pixels carry alpha, so the image can be composited as an
    // overlay on top of another render (e.g. a 3D pipe mesh).
    bool transparentBackground = false;

    float alphaGamma   = 0.70f;
    float alphaScale   = 1.00f;
    float tempStrength = 0.75f;
    float ageGray      = 0.65f;
    float ageDarken    = 0.55f;
    float coreDark     = 0.75f;
    int   themeMode    = 0;
};

struct WaterRenderSettings {
    float alpha = 0.85f;
    int   themeMode = 0;
};

struct OverlaySettings {
    bool showDiv  = false;
    bool showVort = false;
    float divScale  = 8.0f;
    float divAlpha  = 0.75f;
    float vortScale = 8.0f;
    float vortAlpha = 0.75f;
};

struct MACCoupledSim;

class SmokeRenderer {
public:
    SmokeRenderer(int w, int h);
    ~SmokeRenderer();

    void resize(int w, int h);

    void updateFromSim(const MAC2D& sim,
                       const SmokeRenderSettings& smoke,
                       const OverlaySettings& ov);

    void updateWaterFromSim(const MACWater& sim,
                            const WaterRenderSettings& water);
    
    void updateFromSim(const MACCoupledSim& sim,
                       const SmokeRenderSettings& smoke,
                       const OverlaySettings& ov);

    void updateWaterFromSim(const MACCoupledSim& sim,
                            const WaterRenderSettings& water);

    void updateSmokeFromSlice(const std::vector<float>& values,
                              const std::vector<uint8_t>& solid,
                              int width,
                              int height,
                              int fieldMode,
                              const SmokeRenderSettings& smoke);

    void updateSmokeFromVolume(const std::vector<float>& smokeValues,
                               const std::vector<float>& tempValues,
                               const std::vector<uint8_t>& solid,
                               int nx,
                               int ny,
                               int nz,
                               float yawDeg,
                               float pitchDeg,
                               float zoom,
                               float densityScale,
                               const SmokeRenderSettings& smoke);

    void updateWaterFromSlice(const std::vector<float>& values,
                              const std::vector<uint8_t>& solid,
                              int width,
                              int height,
                              const WaterRenderSettings& water);

    void updateWaterFromVolume(const std::vector<float>& values,
                               const std::vector<uint8_t>& solid,
                               int nx,
                               int ny,
                               int nz,
                               int viewMode,
                               float yawDeg,
                               float pitchDeg,
                               float zoom,
                               float densityScale,
                               float surfaceThreshold,
                               const WaterRenderSettings& water);

    unsigned int smokeTex() const { return m_smokeTex; }
    unsigned int divTex()   const { return m_divTex; }
    unsigned int vortTex()  const { return m_vortTex; }
    unsigned int waterTex() const { return m_waterTex; }
    int width() const { return m_w; }
    int height() const { return m_h; }

private:
    int m_w = 0, m_h = 0;
    unsigned int m_smokeTex = 0;
    unsigned int m_divTex = 0;
    unsigned int m_vortTex = 0;
    unsigned int m_waterTex = 0;
    // Reused temporary RGBA buffer for CPU image generation paths.
    std::vector<uint8_t> m_rgbaScratch;

    unsigned int makeTexture(int w, int h);
    void uploadSmokeRGBA(const std::vector<float>& smoke,
                         const std::vector<float>& temp,
                         const std::vector<float>& age,
                         const std::vector<uint8_t>& solid,
                         const SmokeRenderSettings& s);
    void uploadWaterRGBA(const std::vector<float>& water,
                         const std::vector<uint8_t>& solid,
                         const WaterRenderSettings& w);
    void uploadDivOverlay(const std::vector<float>& div,
                          const std::vector<uint8_t>& solid,
                          float scale, float alpha);
    void uploadVortOverlay(const std::vector<float>& omega,
                           const std::vector<uint8_t>& solid,
                           float scale, float alpha);
};
