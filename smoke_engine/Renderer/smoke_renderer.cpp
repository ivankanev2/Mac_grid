#include "smoke_renderer.h"
#include "Sim/mac_smoke_sim.h"
#include <algorithm>  
#include <cmath>     

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#ifdef __APPLE__
  #define GL_SILENCE_DEPRECATION
  #include <OpenGL/gl3.h>
#else
  #include <GL/gl.h>
#endif

#include <vector>
#include <cstdint>

// ---------------- helpers ----------------
unsigned int SmokeRenderer::makeTexture(int w, int h) {
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    std::vector<uint8_t> blank(w * h * 4, 0);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, blank.data());
    return (unsigned int)tex;
}

SmokeRenderer::SmokeRenderer(int w, int h) : m_w(w), m_h(h) {
    m_smokeTex = makeTexture(w, h);
    m_divTex = makeTexture(w, h);
    m_vortTex = makeTexture(w, h);
}
SmokeRenderer::~SmokeRenderer() {
    if (m_smokeTex) glDeleteTextures(1, (GLuint*)&m_smokeTex);
    if (m_divTex)   glDeleteTextures(1, (GLuint*)&m_divTex);
    if (m_vortTex)  glDeleteTextures(1, (GLuint*)&m_vortTex);
}
void SmokeRenderer::resize(int w, int h) {
    if (w == m_w && h == m_h) return;
    m_w = w; m_h = h;
    if (m_smokeTex) glDeleteTextures(1, (GLuint*)&m_smokeTex);
    if (m_divTex)   glDeleteTextures(1, (GLuint*)&m_divTex);
    if (m_vortTex)  glDeleteTextures(1, (GLuint*)&m_vortTex);
    m_smokeTex = makeTexture(w,h);
    m_divTex   = makeTexture(w,h);
    m_vortTex  = makeTexture(w,h);
}

void SmokeRenderer::uploadSmokeRGBA(const std::vector<float>& smoke,
                                   const std::vector<float>& temp,
                                   const std::vector<float>& age,
                                   const std::vector<uint8_t>& solid,
                                   const SmokeRenderSettings& s)
{
    int w = m_w, h = m_h;
    std::vector<uint8_t> img(w * h * 4, 0);

    auto clamp01 = [](float x) {
        if (x < 0.0f) return 0.0f;
        if (x > 1.0f) return 1.0f;
        return x;
    };

    for (int j = 0; j < h; ++j) {
        int srcJ = (h - 1 - j);
        for (int i = 0; i < w; ++i) {
            int srcIdx = i + w * srcJ;
            int dstIdx = i + w * j;

            if (solid[srcIdx]) {
                img[dstIdx*4 + 0] = 40;
                img[dstIdx*4 + 1] = 90;
                img[dstIdx*4 + 2] = 200;
                img[dstIdx*4 + 3] = 255;
                continue;
            }

            float d = clamp01(smoke[srcIdx]);
            float alpha = std::pow(d, s.alphaGamma) * s.alphaScale;
            alpha = clamp01(alpha);

            float r, g, b;
            if (!s.useColor) {
                float gray = std::pow(d, 0.6f);
                r = g = b = gray;
            } else {
                float t = clamp01(temp[srcIdx] * s.tempStrength);
                float a = clamp01(age[srcIdx]);

                r = (1.0f - t) * 0.20f + t * 1.00f;
                g = (1.0f - t) * 0.25f + t * 0.55f;
                b = (1.0f - t) * 0.30f + t * 0.10f;

                float brightness = (1.0f - s.coreDark) * 1.0f +
                    s.coreDark * (0.25f + 0.75f * (1.0f - std::pow(d, 0.5f)));

                float baseGray = (r + g + b) / 3.0f;
                float ageMix = clamp01(s.ageGray * a);
                float darken = 1.0f - clamp01(s.ageDarken * a);

                r = (1.0f - ageMix) * r + ageMix * baseGray;
                g = (1.0f - ageMix) * g + ageMix * baseGray;
                b = (1.0f - ageMix) * b + ageMix * baseGray;

                brightness *= darken;
                r *= brightness; g *= brightness; b *= brightness;
            }

            r = clamp01(r); g = clamp01(g); b = clamp01(b);

            img[dstIdx*4 + 0] = (uint8_t)std::lround(r * 255.0f);
            img[dstIdx*4 + 1] = (uint8_t)std::lround(g * 255.0f);
            img[dstIdx*4 + 2] = (uint8_t)std::lround(b * 255.0f);
            img[dstIdx*4 + 3] = (uint8_t)std::lround(alpha * 255.0f);
        }
    }

    glBindTexture(GL_TEXTURE_2D, m_smokeTex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, img.data());
}

void SmokeRenderer::uploadDivOverlay(const std::vector<float>& div,
                                    const std::vector<uint8_t>& solid,
                                    float scale, float alpha)
{
    int w = m_w, h = m_h;
    std::vector<uint8_t> img(w * h * 4, 0);

    auto clamp01 = [](float x) {
        if (x < 0.0f) return 0.0f;
        if (x > 1.0f) return 1.0f;
        return x;
    };

    for (int j = 0; j < h; ++j) {
        int srcJ = (h - 1 - j);
        for (int i = 0; i < w; ++i) {
            int srcIdx = i + w * srcJ;
            int dstIdx = i + w * j;

            if (solid[srcIdx]) { img[dstIdx*4 + 3] = 0; continue; }

            float d = div[srcIdx] * scale;
            d = std::max(-1.0f, std::min(1.0f, d));

            float m = std::fabs(d);
            uint8_t A = (uint8_t)std::lround(clamp01(m * alpha) * 255.0f);
            uint8_t R = (d > 0.0f) ? (uint8_t)std::lround(m * 255.0f) : 0;
            uint8_t B = (d < 0.0f) ? (uint8_t)std::lround(m * 255.0f) : 0;

            img[dstIdx*4 + 0] = R;
            img[dstIdx*4 + 1] = 0;
            img[dstIdx*4 + 2] = B;
            img[dstIdx*4 + 3] = A;
        }
    }

    glBindTexture(GL_TEXTURE_2D, m_divTex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, img.data());
}

void SmokeRenderer::uploadVortOverlay(const std::vector<float>& omega,
                                     const std::vector<uint8_t>& solid,
                                     float scale, float alpha)
{
    int w = m_w, h = m_h;
    std::vector<uint8_t> img(w * h * 4, 0);

    auto clamp01 = [](float x) {
        if (x < 0.0f) return 0.0f;
        if (x > 1.0f) return 1.0f;
        return x;
    };

    for (int j = 0; j < h; ++j) {
        int srcJ = (h - 1 - j);
        for (int i = 0; i < w; ++i) {
            int srcIdx = i + w * srcJ;
            int dstIdx = i + w * j;

            if (solid[srcIdx]) { img[dstIdx*4 + 3] = 0; continue; }

            float v = omega[srcIdx] * scale;
            v = std::max(-1.0f, std::min(1.0f, v));

            float m = std::fabs(v);
            uint8_t A = (uint8_t)std::lround(clamp01(m * alpha) * 255.0f);
            uint8_t R = (v > 0.0f) ? (uint8_t)std::lround(m * 255.0f) : 0;
            uint8_t B = (v < 0.0f) ? (uint8_t)std::lround(m * 255.0f) : 0;

            img[dstIdx*4 + 0] = R;
            img[dstIdx*4 + 1] = 0;
            img[dstIdx*4 + 2] = B;
            img[dstIdx*4 + 3] = A;
        }
    }

    glBindTexture(GL_TEXTURE_2D, m_vortTex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, img.data());
}

// --- public updateFromSim ---
void SmokeRenderer::updateFromSim(const MAC2D& sim,
                       const SmokeRenderSettings& smoke,
                       const OverlaySettings& ov)
{
    // update smoke texture
    uploadSmokeRGBA(sim.density(), sim.temperature(), sim.ageField(), sim.solidMask(), smoke);

    // div overlay
    if (ov.showDiv) {
        uploadDivOverlay(sim.divergence(), sim.solidMask(), ov.divScale, ov.divAlpha);
    }

    // vort overlay
    if (ov.showVort) {
        std::vector<float> vort(sim.nx * sim.ny);
        sim.computeVorticity(vort);
        uploadVortOverlay(vort, sim.solidMask(), ov.vortScale, ov.vortAlpha);
    }
}