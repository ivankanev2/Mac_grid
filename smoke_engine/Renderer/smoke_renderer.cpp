#include "smoke_renderer.h"
#include "Sim/mac_smoke_sim.h"
#include "Sim/mac_water_sim.h"
#include "Sim/mac_coupled_sim.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <thread>

#ifdef __APPLE__
  #define GL_SILENCE_DEPRECATION
  #define GLFW_INCLUDE_NONE
  #include <GLFW/glfw3.h>
  #include <OpenGL/gl3.h>
#elif defined(_WIN32)
  #define WIN32_LEAN_AND_MEAN
  #define NOMINMAX
  #include <windows.h>
  #define GLFW_INCLUDE_NONE
  #include <GLFW/glfw3.h>
  #include <GL/gl.h>
#else
  #define GLFW_INCLUDE_NONE
  #include <GLFW/glfw3.h>
  #include <GL/gl.h>
#endif

#ifndef GL_CLAMP_TO_EDGE
#define GL_CLAMP_TO_EDGE 0x812F
#endif

#include <vector>
#include <cstdint>

namespace {

struct Vec3f {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

struct WaterViewBox {
    float hx = 0.5f;
    float hy = 0.5f;
    float hz = 0.5f;
    float camDist = 1.85f;
    float fovScale = 0.95f;
    float imageAspect = 1.0f;
};

static Vec3f operator+(Vec3f a, Vec3f b) { return Vec3f{a.x + b.x, a.y + b.y, a.z + b.z}; }
static Vec3f operator-(Vec3f a, Vec3f b) { return Vec3f{a.x - b.x, a.y - b.y, a.z - b.z}; }
static Vec3f operator*(Vec3f a, float s) { return Vec3f{a.x * s, a.y * s, a.z * s}; }
static Vec3f operator*(float s, Vec3f a) { return a * s; }

static float dot3(Vec3f a, Vec3f b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static float length3(Vec3f v) {
    return std::sqrt(dot3(v, v));
}

static Vec3f normalize3(Vec3f v) {
    const float len2 = dot3(v, v);
    if (len2 <= 1e-20f) return Vec3f{0.0f, 0.0f, 1.0f};
    const float inv = 1.0f / std::sqrt(len2);
    return Vec3f{v.x * inv, v.y * inv, v.z * inv};
}

static Vec3f rotateYawPitch(Vec3f v, float yawRad, float pitchRad) {
    const float cy = std::cos(yawRad);
    const float sy = std::sin(yawRad);
    const float cp = std::cos(pitchRad);
    const float sp = std::sin(pitchRad);
    return Vec3f{
        cy * v.x + sy * v.z,
        cp * v.y - sp * (-sy * v.x + cy * v.z),
        sp * v.y + cp * (-sy * v.x + cy * v.z)
    };
}

static Vec3f rotateInvYawPitch(Vec3f v, float yawRad, float pitchRad) {
    const float cy = std::cos(yawRad);
    const float sy = std::sin(yawRad);
    const float cp = std::cos(pitchRad);
    const float sp = std::sin(pitchRad);
    return Vec3f{
        cy * v.x + sy * sp * v.y - sy * cp * v.z,
        cp * v.y + sp * v.z,
        sy * v.x - cy * sp * v.y + cy * cp * v.z
    };
}

static WaterViewBox makeWaterViewBox(int nx, int ny, int nz, float imageAspect) {
    const int maxDim = std::max({nx, ny, nz, 1});
    WaterViewBox box;
    box.hx = 0.5f * (float)nx / (float)maxDim;
    box.hy = 0.5f * (float)ny / (float)maxDim;
    box.hz = 0.5f * (float)nz / (float)maxDim;
    box.imageAspect = std::max(1e-6f, imageAspect);
    return box;
}

static bool intersectBox(const Vec3f& o, const Vec3f& d, const WaterViewBox& box, float& tminOut, float& tmaxOut) {
    float tmin = 0.0f;
    float tmax = std::numeric_limits<float>::infinity();

    auto updateAxis = [&](float oAxis, float dAxis, float halfExtent) -> bool {
        const float lo = -halfExtent;
        const float hi = halfExtent;
        if (std::fabs(dAxis) < 1e-8f) {
            return (oAxis >= lo && oAxis <= hi);
        }
        float t0 = (lo - oAxis) / dAxis;
        float t1 = (hi - oAxis) / dAxis;
        if (t0 > t1) std::swap(t0, t1);
        tmin = std::max(tmin, t0);
        tmax = std::min(tmax, t1);
        return tmax >= tmin;
    };

    if (!updateAxis(o.x, d.x, box.hx)) return false;
    if (!updateAxis(o.y, d.y, box.hy)) return false;
    if (!updateAxis(o.z, d.z, box.hz)) return false;

    tminOut = tmin;
    tmaxOut = tmax;
    return tmaxOut > tminOut;
}

static Vec3f localToUnit(const Vec3f& p, const WaterViewBox& box) {
    return Vec3f{
        (p.x + box.hx) / std::max(1e-6f, 2.0f * box.hx),
        (p.y + box.hy) / std::max(1e-6f, 2.0f * box.hy),
        (p.z + box.hz) / std::max(1e-6f, 2.0f * box.hz)
    };
}

static float sampleTrilinear(const std::vector<float>& volume, int nx, int ny, int nz, float x, float y, float z) {
    if (volume.empty() || nx <= 0 || ny <= 0 || nz <= 0) return 0.0f;

    x = std::clamp(x, 0.0f, 1.0f);
    y = std::clamp(y, 0.0f, 1.0f);
    z = std::clamp(z, 0.0f, 1.0f);

    const float fx = x * (float)(nx - 1);
    const float fy = y * (float)(ny - 1);
    const float fz = z * (float)(nz - 1);

    const int i0 = std::clamp((int)std::floor(fx), 0, nx - 1);
    const int j0 = std::clamp((int)std::floor(fy), 0, ny - 1);
    const int k0 = std::clamp((int)std::floor(fz), 0, nz - 1);
    const int i1 = std::min(i0 + 1, nx - 1);
    const int j1 = std::min(j0 + 1, ny - 1);
    const int k1 = std::min(k0 + 1, nz - 1);

    const float tx = std::clamp(fx - (float)i0, 0.0f, 1.0f);
    const float ty = std::clamp(fy - (float)j0, 0.0f, 1.0f);
    const float tz = std::clamp(fz - (float)k0, 0.0f, 1.0f);

    auto at = [&](int i, int j, int k) -> float {
        return volume[(std::size_t)i + (std::size_t)nx * ((std::size_t)j + (std::size_t)ny * (std::size_t)k)];
    };

    const float c000 = at(i0, j0, k0);
    const float c100 = at(i1, j0, k0);
    const float c010 = at(i0, j1, k0);
    const float c110 = at(i1, j1, k0);
    const float c001 = at(i0, j0, k1);
    const float c101 = at(i1, j0, k1);
    const float c011 = at(i0, j1, k1);
    const float c111 = at(i1, j1, k1);

    const float c00 = c000 * (1.0f - tx) + c100 * tx;
    const float c10 = c010 * (1.0f - tx) + c110 * tx;
    const float c01 = c001 * (1.0f - tx) + c101 * tx;
    const float c11 = c011 * (1.0f - tx) + c111 * tx;
    const float c0 = c00 * (1.0f - ty) + c10 * ty;
    const float c1 = c01 * (1.0f - ty) + c11 * ty;
    return c0 * (1.0f - tz) + c1 * tz;
}

static uint8_t sampleSolidNearest(const std::vector<uint8_t>& solid, int nx, int ny, int nz, float x, float y, float z) {
    if (solid.empty() || nx <= 0 || ny <= 0 || nz <= 0) return (uint8_t)0;
    const int i = std::clamp((int)std::round(std::clamp(x, 0.0f, 1.0f) * (float)(nx - 1)), 0, nx - 1);
    const int j = std::clamp((int)std::round(std::clamp(y, 0.0f, 1.0f) * (float)(ny - 1)), 0, ny - 1);
    const int k = std::clamp((int)std::round(std::clamp(z, 0.0f, 1.0f) * (float)(nz - 1)), 0, nz - 1);
    return solid[(std::size_t)i + (std::size_t)nx * ((std::size_t)j + (std::size_t)ny * (std::size_t)k)];
}

static Vec3f sampleGradient(const std::vector<float>& volume, int nx, int ny, int nz, Vec3f q) {
    const float epsX = 1.0f / (float)std::max(2, nx - 1);
    const float epsY = 1.0f / (float)std::max(2, ny - 1);
    const float epsZ = 1.0f / (float)std::max(2, nz - 1);
    const float gx = sampleTrilinear(volume, nx, ny, nz, q.x + epsX, q.y, q.z)
                   - sampleTrilinear(volume, nx, ny, nz, q.x - epsX, q.y, q.z);
    const float gy = sampleTrilinear(volume, nx, ny, nz, q.x, q.y + epsY, q.z)
                   - sampleTrilinear(volume, nx, ny, nz, q.x, q.y - epsY, q.z);
    const float gz = sampleTrilinear(volume, nx, ny, nz, q.x, q.y, q.z + epsZ)
                   - sampleTrilinear(volume, nx, ny, nz, q.x, q.y, q.z - epsZ);
    return Vec3f{gx, gy, gz};
}

static bool isDarkTheme(int themeMode) {
    return themeMode == 0;
}

static void solidThemeColor(int themeMode, uint8_t& r, uint8_t& g, uint8_t& b) {
    if (isDarkTheme(themeMode)) {
        r = 44; g = 49; b = 55;
    } else {
        r = 214; g = 218; b = 224;
    }
}

static Vec3f themedSmokeBgA(int themeMode) {
    return isDarkTheme(themeMode) ? Vec3f{0.025f, 0.030f, 0.040f}
                                  : Vec3f{0.982f, 0.980f, 0.974f};
}

static Vec3f themedSmokeBgB(int themeMode) {
    return isDarkTheme(themeMode) ? Vec3f{0.060f, 0.070f, 0.085f}
                                  : Vec3f{0.918f, 0.925f, 0.938f};
}

static Vec3f themedWaterBgA(int themeMode) {
    return isDarkTheme(themeMode) ? Vec3f{0.045f, 0.055f, 0.070f}
                                  : Vec3f{0.986f, 0.984f, 0.978f};
}

static Vec3f themedWaterBgB(int themeMode) {
    return isDarkTheme(themeMode) ? Vec3f{0.080f, 0.095f, 0.120f}
                                  : Vec3f{0.910f, 0.922f, 0.940f};
}

template <typename Fn>
static void parallelForRows(int rowCount, Fn&& fn) {
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    const int minRowsPerTask = 32;
    const int maxTasks = std::max(1, std::min<int>((int)hw, rowCount / minRowsPerTask));
    if (maxTasks <= 1 || rowCount <= minRowsPerTask) {
        fn(0, rowCount);
        return;
    }

    std::vector<std::thread> workers;
    workers.reserve((size_t)maxTasks - 1);
    const int chunk = (rowCount + maxTasks - 1) / maxTasks;
    for (int task = 1; task < maxTasks; ++task) {
        const int begin = task * chunk;
        if (begin >= rowCount) break;
        const int end = std::min(rowCount, begin + chunk);
        workers.emplace_back([&, begin, end]() { fn(begin, end); });
    }

    fn(0, std::min(rowCount, chunk));
    for (auto& worker : workers) worker.join();
}

struct VolumeActiveBounds {
    bool hasData = false;
    int i0 = 0;
    int j0 = 0;
    int k0 = 0;
    int i1 = 0;
    int j1 = 0;
    int k1 = 0;
};

static VolumeActiveBounds findActiveBounds(const std::vector<float>& values,
                                           const std::vector<uint8_t>& solid,
                                           int nx,
                                           int ny,
                                           int nz,
                                           float threshold)
{
    VolumeActiveBounds bounds;
    int minI = nx, minJ = ny, minK = nz;
    int maxI = -1, maxJ = -1, maxK = -1;
    const std::size_t cellCount = (std::size_t)nx * (std::size_t)ny * (std::size_t)nz;
    for (std::size_t idx = 0; idx < cellCount; ++idx) {
        if (solid[idx]) continue;
        if (values[idx] <= threshold) continue;
        const int k = (int)(idx / ((std::size_t)nx * (std::size_t)ny));
        const std::size_t rem = idx - (std::size_t)k * (std::size_t)nx * (std::size_t)ny;
        const int j = (int)(rem / (std::size_t)nx);
        const int i = (int)(rem - (std::size_t)j * (std::size_t)nx);
        minI = std::min(minI, i); minJ = std::min(minJ, j); minK = std::min(minK, k);
        maxI = std::max(maxI, i); maxJ = std::max(maxJ, j); maxK = std::max(maxK, k);
    }

    if (maxI < minI || maxJ < minJ || maxK < minK) {
        return bounds;
    }

    bounds.hasData = true;
    bounds.i0 = std::max(0, minI - 1);
    bounds.j0 = std::max(0, minJ - 1);
    bounds.k0 = std::max(0, minK - 1);
    bounds.i1 = std::min(nx - 1, maxI + 1);
    bounds.j1 = std::min(ny - 1, maxJ + 1);
    bounds.k1 = std::min(nz - 1, maxK + 1);
    return bounds;
}

static bool intersectAabb(const Vec3f& o,
                          const Vec3f& d,
                          const Vec3f& bmin,
                          const Vec3f& bmax,
                          float& tminOut,
                          float& tmaxOut)
{
    float tmin = 0.0f;
    float tmax = std::numeric_limits<float>::infinity();
    auto updateAxis = [&](float oAxis, float dAxis, float lo, float hi) -> bool {
        if (std::fabs(dAxis) < 1e-8f) {
            return (oAxis >= lo && oAxis <= hi);
        }
        float t0 = (lo - oAxis) / dAxis;
        float t1 = (hi - oAxis) / dAxis;
        if (t0 > t1) std::swap(t0, t1);
        tmin = std::max(tmin, t0);
        tmax = std::min(tmax, t1);
        return tmax >= tmin;
    };

    if (!updateAxis(o.x, d.x, bmin.x, bmax.x)) return false;
    if (!updateAxis(o.y, d.y, bmin.y, bmax.y)) return false;
    if (!updateAxis(o.z, d.z, bmin.z, bmax.z)) return false;

    tminOut = tmin;
    tmaxOut = tmax;
    return tmaxOut > tminOut;
}

static Vec3f activeBoundsMinLocal(const VolumeActiveBounds& bounds, const WaterViewBox& box, int nx, int ny, int nz) {
    return Vec3f{
        -box.hx + (2.0f * box.hx) * ((float)bounds.i0 / (float)std::max(1, nx)),
        -box.hy + (2.0f * box.hy) * ((float)bounds.j0 / (float)std::max(1, ny)),
        -box.hz + (2.0f * box.hz) * ((float)bounds.k0 / (float)std::max(1, nz))
    };
}

static Vec3f activeBoundsMaxLocal(const VolumeActiveBounds& bounds, const WaterViewBox& box, int nx, int ny, int nz) {
    return Vec3f{
        -box.hx + (2.0f * box.hx) * ((float)(bounds.i1 + 1) / (float)std::max(1, nx)),
        -box.hy + (2.0f * box.hy) * ((float)(bounds.j1 + 1) / (float)std::max(1, ny)),
        -box.hz + (2.0f * box.hz) * ((float)(bounds.k1 + 1) / (float)std::max(1, nz))
    };
}

static void fillVerticalGradientBackground(std::vector<uint8_t>& img,
                                           int w,
                                           int h,
                                           Vec3f bgA,
                                           Vec3f bgB)
{
    img.assign((std::size_t)w * (std::size_t)h * 4u, 0u);
    auto clamp01f = [](float x) {
        return std::clamp(x, 0.0f, 1.0f);
    };
    parallelForRows(h, [&](int rowBegin, int rowEnd) {
        for (int j = rowBegin; j < rowEnd; ++j) {
            const float t = (float)j / (float)std::max(1, h - 1);
            const Vec3f color{
                bgA.x * (1.0f - t) + bgB.x * t,
                bgA.y * (1.0f - t) + bgB.y * t,
                bgA.z * (1.0f - t) + bgB.z * t
            };
            const uint8_t r = (uint8_t)std::lround(clamp01f(color.x) * 255.0f);
            const uint8_t g = (uint8_t)std::lround(clamp01f(color.y) * 255.0f);
            const uint8_t b = (uint8_t)std::lround(clamp01f(color.z) * 255.0f);
            for (int i = 0; i < w; ++i) {
                const std::size_t dst = ((std::size_t)j * (std::size_t)w + (std::size_t)i) * 4u;
                img[dst + 0] = r;
                img[dst + 1] = g;
                img[dst + 2] = b;
                img[dst + 3] = 255;
            }
        }
    });
}

}

unsigned int SmokeRenderer::makeTexture(int w, int h) {
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
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
    m_waterTex = makeTexture(w, h);
}
SmokeRenderer::~SmokeRenderer() {
    if (m_smokeTex) glDeleteTextures(1, (GLuint*)&m_smokeTex);
    if (m_divTex)   glDeleteTextures(1, (GLuint*)&m_divTex);
    if (m_vortTex)  glDeleteTextures(1, (GLuint*)&m_vortTex);
    if (m_waterTex) glDeleteTextures(1, (GLuint*)&m_waterTex);
}
void SmokeRenderer::resize(int w, int h) {
    if (w == m_w && h == m_h) return;
    m_w = w; m_h = h;
    if (m_smokeTex) glDeleteTextures(1, (GLuint*)&m_smokeTex);
    if (m_divTex)   glDeleteTextures(1, (GLuint*)&m_divTex);
    if (m_vortTex)  glDeleteTextures(1, (GLuint*)&m_vortTex);
    if (m_waterTex) glDeleteTextures(1, (GLuint*)&m_waterTex);
    m_smokeTex = makeTexture(w,h);
    m_divTex   = makeTexture(w,h);
    m_vortTex  = makeTexture(w,h);
    m_waterTex = makeTexture(w,h);
}

void SmokeRenderer::uploadSmokeRGBA(const std::vector<float>& smoke,
                                   const std::vector<float>& temp,
                                   const std::vector<float>& age,
                                   const std::vector<uint8_t>& solid,
                                   const SmokeRenderSettings& s)
{
    int w = m_w, h = m_h;
    m_rgbaScratch.assign((std::size_t)w * (std::size_t)h * 4u, 0u);
    auto& img = m_rgbaScratch;

    auto clamp01 = [](float x) {
        if (x < 0.0f) return 0.0f;
        if (x > 1.0f) return 1.0f;
        return x;
    };

    parallelForRows(h, [&](int rowBegin, int rowEnd) {
        for (int j = rowBegin; j < rowEnd; ++j) {
            const int srcJ = (h - 1 - j);
            for (int i = 0; i < w; ++i) {
                const int srcIdx = i + w * srcJ;
                const int dstIdx = i + w * j;

                if (solid[srcIdx]) {
                    uint8_t sr = 0, sg = 0, sb = 0;
                    solidThemeColor(s.themeMode, sr, sg, sb);
                    img[(std::size_t)dstIdx * 4 + 0] = sr;
                    img[(std::size_t)dstIdx * 4 + 1] = sg;
                    img[(std::size_t)dstIdx * 4 + 2] = sb;
                    img[(std::size_t)dstIdx * 4 + 3] = 255;
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
                    r *= brightness;
                    g *= brightness;
                    b *= brightness;
                }

                r = clamp01(r);
                g = clamp01(g);
                b = clamp01(b);

                img[(std::size_t)dstIdx * 4 + 0] = (uint8_t)std::lround(r * 255.0f);
                img[(std::size_t)dstIdx * 4 + 1] = (uint8_t)std::lround(g * 255.0f);
                img[(std::size_t)dstIdx * 4 + 2] = (uint8_t)std::lround(b * 255.0f);
                img[(std::size_t)dstIdx * 4 + 3] = (uint8_t)std::lround(alpha * 255.0f);
            }
        }
    });

    glBindTexture(GL_TEXTURE_2D, m_smokeTex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, img.data());
}

void SmokeRenderer::uploadWaterRGBA(const std::vector<float>& water,
                                    const std::vector<uint8_t>& solid,
                                    const WaterRenderSettings& wset)
{
    int w = m_w, h = m_h;
    m_rgbaScratch.assign((std::size_t)w * (std::size_t)h * 4u, 0u);
    auto& img = m_rgbaScratch;

    auto clamp01 = [](float x) {
        if (x < 0.0f) return 0.0f;
        if (x > 1.0f) return 1.0f;
        return x;
    };

    parallelForRows(h, [&](int rowBegin, int rowEnd) {
        for (int j = rowBegin; j < rowEnd; ++j) {
            const int srcJ = (h - 1 - j);
            for (int i = 0; i < w; ++i) {
                const int srcIdx = i + w * srcJ;
                const int dstIdx = i + w * j;

                if (solid[srcIdx]) {
                    uint8_t sr = 0, sg = 0, sb = 0;
                    solidThemeColor(wset.themeMode, sr, sg, sb);
                    img[(std::size_t)dstIdx * 4 + 0] = sr;
                    img[(std::size_t)dstIdx * 4 + 1] = sg;
                    img[(std::size_t)dstIdx * 4 + 2] = sb;
                    img[(std::size_t)dstIdx * 4 + 3] = 255;
                    continue;
                }

                const float raw = std::max(0.0f, water[srcIdx]);
                const float d = 1.0f - std::exp(-raw);
                const float a = clamp01(d * wset.alpha);

                float r, g, b;
                if (isDarkTheme(wset.themeMode)) {
                    r = 0.08f + 0.08f * d;
                    g = 0.16f + 0.18f * d;
                    b = 0.30f + 0.28f * d;
                } else {
                    r = 0.18f + 0.08f * d;
                    g = 0.30f + 0.16f * d;
                    b = 0.50f + 0.22f * d;
                }

                img[(std::size_t)dstIdx * 4 + 0] = (uint8_t)std::lround(r * 255.0f);
                img[(std::size_t)dstIdx * 4 + 1] = (uint8_t)std::lround(g * 255.0f);
                img[(std::size_t)dstIdx * 4 + 2] = (uint8_t)std::lround(b * 255.0f);
                img[(std::size_t)dstIdx * 4 + 3] = (uint8_t)std::lround(a * 255.0f);
            }
        }
    });

    glBindTexture(GL_TEXTURE_2D, m_waterTex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, img.data());
}

void SmokeRenderer::uploadDivOverlay(const std::vector<float>& div,
                                    const std::vector<uint8_t>& solid,
                                    float scale, float alpha)
{
    int w = m_w, h = m_h;
    m_rgbaScratch.assign((std::size_t)w * (std::size_t)h * 4u, 0u);
    auto& img = m_rgbaScratch;

    auto clamp01 = [](float x) {
        if (x < 0.0f) return 0.0f;
        if (x > 1.0f) return 1.0f;
        return x;
    };

    parallelForRows(h, [&](int rowBegin, int rowEnd) {
        for (int j = rowBegin; j < rowEnd; ++j) {
            const int srcJ = (h - 1 - j);
            for (int i = 0; i < w; ++i) {
                const int srcIdx = i + w * srcJ;
                const int dstIdx = i + w * j;

                if (solid[srcIdx]) {
                    img[(std::size_t)dstIdx * 4 + 0] = 0;
                    img[(std::size_t)dstIdx * 4 + 1] = 0;
                    img[(std::size_t)dstIdx * 4 + 2] = 0;
                    img[(std::size_t)dstIdx * 4 + 3] = 0;
                    continue;
                }

                float d = div[srcIdx] * scale;
                d = std::max(-1.0f, std::min(1.0f, d));

                const float m = std::fabs(d);
                const uint8_t A = (uint8_t)std::lround(clamp01(m * alpha) * 255.0f);
                const uint8_t R = (d > 0.0f) ? (uint8_t)std::lround(m * 255.0f) : 0;
                const uint8_t B = (d < 0.0f) ? (uint8_t)std::lround(m * 255.0f) : 0;

                img[(std::size_t)dstIdx * 4 + 0] = R;
                img[(std::size_t)dstIdx * 4 + 1] = 0;
                img[(std::size_t)dstIdx * 4 + 2] = B;
                img[(std::size_t)dstIdx * 4 + 3] = A;
            }
        }
    });

    glBindTexture(GL_TEXTURE_2D, m_divTex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, img.data());
}

void SmokeRenderer::uploadVortOverlay(const std::vector<float>& omega,
                                     const std::vector<uint8_t>& solid,
                                     float scale, float alpha)
{
    int w = m_w, h = m_h;
    m_rgbaScratch.assign((std::size_t)w * (std::size_t)h * 4u, 0u);
    auto& img = m_rgbaScratch;

    auto clamp01 = [](float x) {
        if (x < 0.0f) return 0.0f;
        if (x > 1.0f) return 1.0f;
        return x;
    };

    parallelForRows(h, [&](int rowBegin, int rowEnd) {
        for (int j = rowBegin; j < rowEnd; ++j) {
            const int srcJ = (h - 1 - j);
            for (int i = 0; i < w; ++i) {
                const int srcIdx = i + w * srcJ;
                const int dstIdx = i + w * j;

                if (solid[srcIdx]) {
                    img[(std::size_t)dstIdx * 4 + 0] = 0;
                    img[(std::size_t)dstIdx * 4 + 1] = 0;
                    img[(std::size_t)dstIdx * 4 + 2] = 0;
                    img[(std::size_t)dstIdx * 4 + 3] = 0;
                    continue;
                }

                float v = omega[srcIdx] * scale;
                v = std::max(-1.0f, std::min(1.0f, v));

                const float m = std::fabs(v);
                const uint8_t A = (uint8_t)std::lround(clamp01(m * alpha) * 255.0f);
                const uint8_t R = (v > 0.0f) ? (uint8_t)std::lround(m * 255.0f) : 0;
                const uint8_t B = (v < 0.0f) ? (uint8_t)std::lround(m * 255.0f) : 0;

                img[(std::size_t)dstIdx * 4 + 0] = R;
                img[(std::size_t)dstIdx * 4 + 1] = 0;
                img[(std::size_t)dstIdx * 4 + 2] = B;
                img[(std::size_t)dstIdx * 4 + 3] = A;
            }
        }
    });

    glBindTexture(GL_TEXTURE_2D, m_vortTex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, img.data());
}

void SmokeRenderer::updateFromSim(const MAC2D& sim,
                       const SmokeRenderSettings& smoke,
                       const OverlaySettings& ov)
{
    uploadSmokeRGBA(sim.density(), sim.temperature(), sim.ageField(), sim.solidMask(), smoke);

    if (ov.showDiv) {
        uploadDivOverlay(sim.divergence(), sim.solidMask(), ov.divScale, ov.divAlpha);
    }

    if (ov.showVort) {
        std::vector<float> vort(sim.nx * sim.ny);
        sim.computeVorticity(vort);
        uploadVortOverlay(vort, sim.solidMask(), ov.vortScale, ov.vortAlpha);
    }
}

void SmokeRenderer::updateWaterFromSim(const MACWater& sim,
                                       const WaterRenderSettings& water)
{
    uploadWaterRGBA(sim.waterField(), sim.solidMask(), water);
}

void SmokeRenderer::updateFromSim(const MACCoupledSim& sim,
                                  const SmokeRenderSettings& smoke,
                                  const OverlaySettings& ov)
{
    uploadSmokeRGBA(sim.density(), sim.temperature(), sim.ageField(), sim.solidMask(), smoke);

    if (ov.showDiv) {
        uploadDivOverlay(sim.divergence(), sim.solidMask(), ov.divScale, ov.divAlpha);
    }

    if (ov.showVort) {
        // simplest: don’t show vort yet, or compute it later
        std::vector<float> vort(sim.nx * sim.ny, 0.0f);
        uploadVortOverlay(vort, sim.solidMask(), ov.vortScale, ov.vortAlpha);
    }
}

void SmokeRenderer::updateWaterFromSim(const MACCoupledSim& sim,
                                       const WaterRenderSettings& water)
{
    uploadWaterRGBA(sim.waterField(), sim.solidMask(), water);
}


void SmokeRenderer::updateSmokeFromSlice(const std::vector<float>& values,
                                         const std::vector<uint8_t>& solid,
                                         int width,
                                         int height,
                                         int fieldMode,
                                         const SmokeRenderSettings& smoke)
{
    if (width <= 0 || height <= 0) return;
    if ((int)values.size() != width * height || (int)solid.size() != width * height) return;

    if (width != m_w || height != m_h) {
        resize(width, height);
    }

    const bool signedField = (fieldMode == 2 || fieldMode == 3);
    float maxVal = 0.0f;
    float maxAbs = 0.0f;
    for (int idx = 0; idx < width * height; ++idx) {
        if (solid[(std::size_t)idx]) continue;
        const float v = values[(std::size_t)idx];
        maxVal = std::max(maxVal, v);
        maxAbs = std::max(maxAbs, std::fabs(v));
    }
    maxVal = std::max(maxVal, 1e-6f);
    maxAbs = std::max(maxAbs, 1e-6f);

    m_rgbaScratch.assign((std::size_t)m_w * (std::size_t)m_h * 4u, 0u);
    auto& img = m_rgbaScratch;

    auto clamp01 = [](float x) {
        if (x < 0.0f) return 0.0f;
        if (x > 1.0f) return 1.0f;
        return x;
    };

    parallelForRows(m_h, [&](int rowBegin, int rowEnd) {
        for (int j = rowBegin; j < rowEnd; ++j) {
            const int srcJ = (m_h - 1 - j);
            for (int i = 0; i < m_w; ++i) {
                const int srcIdx = i + m_w * srcJ;
                const int dstIdx = i + m_w * j;

                if (solid[(std::size_t)srcIdx]) {
                    uint8_t sr = 0, sg = 0, sb = 0;
                    solidThemeColor(smoke.themeMode, sr, sg, sb);
                    img[(std::size_t)dstIdx * 4 + 0] = sr;
                    img[(std::size_t)dstIdx * 4 + 1] = sg;
                    img[(std::size_t)dstIdx * 4 + 2] = sb;
                    img[(std::size_t)dstIdx * 4 + 3] = 255;
                    continue;
                }

                const float raw = values[(std::size_t)srcIdx];
                float r = 0.0f;
                float g = 0.0f;
                float b = 0.0f;
                float a = 0.0f;

                if (signedField) {
                    const float mag = clamp01(std::fabs(raw) / maxAbs);
                    a = std::pow(mag, 0.65f) * 0.95f;
                    if (raw >= 0.0f) {
                        r = 0.18f + 0.82f * mag;
                        g = 0.05f + 0.32f * mag;
                        b = 0.04f + 0.14f * mag;
                    } else {
                        r = 0.05f + 0.14f * mag;
                        g = 0.16f + 0.40f * mag;
                        b = 0.24f + 0.76f * mag;
                    }
                } else if (fieldMode == 1) {
                    const float t = clamp01(std::max(0.0f, raw) / maxVal);
                    a = std::pow(t, 0.60f) * std::max(0.2f, smoke.alphaScale);
                    r = 0.20f + 0.80f * t;
                    g = 0.06f + 0.62f * t;
                    b = 0.03f + 0.22f * t;
                } else if (fieldMode == 4) {
                    const float s = clamp01(std::max(0.0f, raw) / maxVal);
                    a = std::pow(s, 0.60f) * 0.95f;
                    r = 0.08f + 0.18f * s;
                    g = 0.18f + 0.58f * s;
                    b = 0.30f + 0.70f * s;
                } else {
                    const float d = clamp01(std::max(0.0f, raw) / std::max(1.0f, maxVal));
                    a = std::pow(d, smoke.alphaGamma) * smoke.alphaScale;
                    if (!smoke.useColor) {
                        const float gray = std::pow(d, 0.6f);
                        r = g = b = gray;
                    } else {
                        const float gray = 0.12f + 0.88f * std::pow(d, 0.55f);
                        r = gray;
                        g = gray;
                        b = gray;
                    }
                }

                img[(std::size_t)dstIdx * 4 + 0] = (uint8_t)std::lround(clamp01(r) * 255.0f);
                img[(std::size_t)dstIdx * 4 + 1] = (uint8_t)std::lround(clamp01(g) * 255.0f);
                img[(std::size_t)dstIdx * 4 + 2] = (uint8_t)std::lround(clamp01(b) * 255.0f);
                img[(std::size_t)dstIdx * 4 + 3] = (uint8_t)std::lround(clamp01(a) * 255.0f);
            }
        }
    });

    glBindTexture(GL_TEXTURE_2D, m_smokeTex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_w, m_h, GL_RGBA, GL_UNSIGNED_BYTE, img.data());
}

void SmokeRenderer::updateSmokeFromVolume(const std::vector<float>& smokeValues,
                                          const std::vector<float>& tempValues,
                                          const std::vector<uint8_t>& solid,
                                          int nx,
                                          int ny,
                                          int nz,
                                          float yawDeg,
                                          float pitchDeg,
                                          float zoom,
                                          float densityScale,
                                          const SmokeRenderSettings& smoke)
{
    if (nx <= 1 || ny <= 1 || nz <= 1) return;
    const std::size_t cellCount = (std::size_t)nx * (std::size_t)ny * (std::size_t)nz;
    if (smokeValues.size() != cellCount || solid.size() != cellCount) return;
    if (!tempValues.empty() && tempValues.size() != cellCount) return;
    if (m_w <= 0 || m_h <= 0) return;

    const float yaw = yawDeg * 3.14159265358979323846f / 180.0f;
    const float pitch = pitchDeg * 3.14159265358979323846f / 180.0f;
    const float zoomClamped = std::clamp(zoom, 0.35f, 3.5f);
    const float sigmaScale = std::max(0.05f, densityScale);
    const int maxDim = std::max({nx, ny, nz, 1});
    const WaterViewBox box = makeWaterViewBox(nx, ny, nz, (float)m_w / (float)std::max(1, m_h));
    const float step = 0.60f / (float)std::max(24, maxDim);
    const Vec3f bgA = themedSmokeBgA(smoke.themeMode);
    const Vec3f bgB = themedSmokeBgB(smoke.themeMode);

    const VolumeActiveBounds activeBounds = findActiveBounds(smokeValues, solid, nx, ny, nz, 1.0e-4f);
    m_rgbaScratch.assign((std::size_t)m_w * (std::size_t)m_h * 4u, 0u);
    auto& img = m_rgbaScratch;

    if (!activeBounds.hasData) {
        if (!smoke.transparentBackground)
            fillVerticalGradientBackground(img, m_w, m_h, bgA, bgB);
        // else: leave img all-zeros (fully transparent)
        glBindTexture(GL_TEXTURE_2D, m_smokeTex);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_w, m_h, GL_RGBA, GL_UNSIGNED_BYTE, img.data());
        return;
    }

    const Vec3f activeMin = activeBoundsMinLocal(activeBounds, box, nx, ny, nz);
    const Vec3f activeMax = activeBoundsMaxLocal(activeBounds, box, nx, ny, nz);

    auto clamp01 = [](float x) {
        if (x < 0.0f) return 0.0f;
        if (x > 1.0f) return 1.0f;
        return x;
    };

    parallelForRows(m_h, [&](int rowBegin, int rowEnd) {
        for (int j = rowBegin; j < rowEnd; ++j) {
            const float py = 1.0f - 2.0f * ((j + 0.5f) / (float)m_h);
            for (int i = 0; i < m_w; ++i) {
                const float px = (2.0f * ((i + 0.5f) / (float)m_w) - 1.0f) * box.imageAspect;

                const Vec3f rayOriginWorld{0.0f, 0.0f, -box.camDist / zoomClamped};
                const Vec3f rayDirWorld = normalize3(Vec3f{px * box.fovScale, py * box.fovScale, box.camDist});
                const Vec3f o = rotateInvYawPitch(rayOriginWorld, yaw, pitch);
                const Vec3f d = normalize3(rotateInvYawPitch(rayDirWorld, yaw, pitch));

                const float tBg = (float)j / (float)std::max(1, m_h - 1);
                Vec3f color{
                    bgA.x * (1.0f - tBg) + bgB.x * tBg,
                    bgA.y * (1.0f - tBg) + bgB.y * tBg,
                    bgA.z * (1.0f - tBg) + bgB.z * tBg
                };

                float tmin = 0.0f;
                float tmax = 0.0f;
                float pixelAccumA = 0.0f;   // hoisted for transparent-background alpha
                if (intersectAabb(o, d, activeMin, activeMax, tmin, tmax)) {
                    float t = std::max(0.0f, tmin);
                    float accumR = 0.0f;
                    float accumG = 0.0f;
                    float accumB = 0.0f;
                    float accumA = 0.0f;

                    while (t < tmax && accumA < 0.995f) {
                        const Vec3f q = localToUnit(o + d * t, box);
                        const float density = std::max(0.0f, sampleTrilinear(smokeValues, nx, ny, nz, q.x, q.y, q.z));
                        const uint8_t solidCell = sampleSolidNearest(solid, nx, ny, nz, q.x, q.y, q.z);
                        if (!solidCell && density > 1e-4f) {
                            const float temp = tempValues.empty()
                                ? 0.0f
                                : std::max(0.0f, sampleTrilinear(tempValues, nx, ny, nz, q.x, q.y, q.z));

                            const float sigma = density * sigmaScale * 3.0f;
                            const float aStep = (1.0f - std::exp(-sigma * step * 3.0f)) * clamp01(smoke.alphaScale);

                            float cr = 0.0f;
                            float cg = 0.0f;
                            float cb = 0.0f;
                            if (!smoke.useColor) {
                                const float gray = 0.22f + 0.68f * std::pow(clamp01(density), 0.60f);
                                cr = gray;
                                cg = gray;
                                cb = gray;
                            } else {
                                const float tCol = clamp01(std::pow(temp, 0.55f));
                                const float gray = 0.10f + 0.90f * std::pow(clamp01(density), 0.55f);
                                cr = gray + smoke.tempStrength * tCol * 0.55f;
                                cg = gray + smoke.tempStrength * tCol * 0.12f;
                                cb = gray * (1.0f - 0.35f * tCol);
                                const float ageDark = smoke.coreDark * clamp01(density * density);
                                const float core = 1.0f - ageDark;
                                cr *= 0.35f + 0.65f * core;
                                cg *= 0.35f + 0.65f * core;
                                cb *= 0.35f + 0.65f * core;
                            }

                            const float depthFade = 0.90f - 0.25f * std::clamp((t - tmin) / std::max(1e-6f, tmax - tmin), 0.0f, 1.0f);
                            cr *= depthFade;
                            cg *= depthFade;
                            cb *= depthFade;

                            const float oneMinusA = 1.0f - accumA;
                            accumR += oneMinusA * aStep * cr;
                            accumG += oneMinusA * aStep * cg;
                            accumB += oneMinusA * aStep * cb;
                            accumA += oneMinusA * aStep;
                        }
                        t += step;
                    }

                    if (accumA > 0.0f) {
                        color.x = color.x * (1.0f - accumA) + accumR;
                        color.y = color.y * (1.0f - accumA) + accumG;
                        color.z = color.z * (1.0f - accumA) + accumB;
                    }
                    pixelAccumA = accumA;
                }

                const int dst = (j * m_w + i) * 4;
                img[(std::size_t)dst + 0] = (uint8_t)std::lround(clamp01(color.x) * 255.0f);
                img[(std::size_t)dst + 1] = (uint8_t)std::lround(clamp01(color.y) * 255.0f);
                img[(std::size_t)dst + 2] = (uint8_t)std::lround(clamp01(color.z) * 255.0f);
                // In transparent-background mode the alpha encodes smoke
                // density so this image can be composited over another render.
                img[(std::size_t)dst + 3] = smoke.transparentBackground
                    ? (uint8_t)std::lround(clamp01(pixelAccumA * 2.0f) * 255.0f)
                    : (uint8_t)255;
            }
        }
    });

    glBindTexture(GL_TEXTURE_2D, m_smokeTex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_w, m_h, GL_RGBA, GL_UNSIGNED_BYTE, img.data());
}

void SmokeRenderer::updateWaterFromSlice(const std::vector<float>& values,
                                         const std::vector<uint8_t>& solid,
                                         int width,
                                         int height,
                                         const WaterRenderSettings& water)
{
    if (width <= 0 || height <= 0) return;
    const std::size_t cellCount = (std::size_t)width * (std::size_t)height;
    if (values.size() != cellCount || solid.size() != cellCount) return;

    if (width != m_w || height != m_h) {
        resize(width, height);
    }

    uploadWaterRGBA(values, solid, water);
}

void SmokeRenderer::updateWaterFromVolume(const std::vector<float>& values,
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
                                          const WaterRenderSettings& water) {
    if (nx <= 1 || ny <= 1 || nz <= 1) return;
    const std::size_t cellCount = (std::size_t)nx * (std::size_t)ny * (std::size_t)nz;
    if (values.size() != cellCount || solid.size() != cellCount) return;
    if (m_w <= 0 || m_h <= 0) return;

    const int mode = std::clamp(viewMode, 0, 2);
    const bool surfaceMode = (mode == 2);
    const float yaw = yawDeg * 3.14159265358979323846f / 180.0f;
    const float pitch = pitchDeg * 3.14159265358979323846f / 180.0f;
    const float zoomClamped = std::clamp(zoom, 0.35f, 3.5f);
    const float sigmaScale = std::max(0.05f, densityScale);
    const float iso = std::max(0.01f, surfaceThreshold);
    const int maxDim = std::max({nx, ny, nz, 1});
    const WaterViewBox box = makeWaterViewBox(nx, ny, nz, (float)m_w / (float)std::max(1, m_h));
    const float step = 0.55f / (float)std::max(24, maxDim);
    const Vec3f lightDir = normalize3(Vec3f{-0.45f, 0.72f, 0.53f});
    const Vec3f viewLight = normalize3(Vec3f{0.0f, 0.0f, 1.0f});
    const Vec3f bgA = themedWaterBgA(water.themeMode);
    const Vec3f bgB = themedWaterBgB(water.themeMode);

    const VolumeActiveBounds activeBounds = findActiveBounds(values, solid, nx, ny, nz, 1.0e-4f);
    m_rgbaScratch.assign((std::size_t)m_w * (std::size_t)m_h * 4u, 0u);
    auto& img = m_rgbaScratch;

    if (!activeBounds.hasData) {
        fillVerticalGradientBackground(img, m_w, m_h, bgA, bgB);
        glBindTexture(GL_TEXTURE_2D, m_waterTex);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_w, m_h, GL_RGBA, GL_UNSIGNED_BYTE, img.data());
        return;
    }

    const Vec3f activeMin = activeBoundsMinLocal(activeBounds, box, nx, ny, nz);
    const Vec3f activeMax = activeBoundsMaxLocal(activeBounds, box, nx, ny, nz);

    parallelForRows(m_h, [&](int rowBegin, int rowEnd) {
        for (int j = rowBegin; j < rowEnd; ++j) {
            const float py = 1.0f - 2.0f * ((j + 0.5f) / (float)m_h);
            for (int i = 0; i < m_w; ++i) {
                const float px = (2.0f * ((i + 0.5f) / (float)m_w) - 1.0f) * box.imageAspect;

                const Vec3f rayOriginWorld{0.0f, 0.0f, -box.camDist / zoomClamped};
                const Vec3f rayDirWorld = normalize3(Vec3f{px * box.fovScale, py * box.fovScale, box.camDist});
                const Vec3f o = rotateInvYawPitch(rayOriginWorld, yaw, pitch);
                const Vec3f d = normalize3(rotateInvYawPitch(rayDirWorld, yaw, pitch));

                const float tBg = (float)j / (float)std::max(1, m_h - 1);
                Vec3f color{
                    bgA.x * (1.0f - tBg) + bgB.x * tBg,
                    bgA.y * (1.0f - tBg) + bgB.y * tBg,
                    bgA.z * (1.0f - tBg) + bgB.z * tBg
                };

                float tmin = 0.0f;
                float tmax = 0.0f;
                if (intersectAabb(o, d, activeMin, activeMax, tmin, tmax)) {
                    float t = std::max(0.0f, tmin);
                    if (surfaceMode) {
                        float prevT = t;
                        bool prevValid = false;
                        bool hit = false;
                        Vec3f hitQ{};

                        while (t < tmax) {
                            const Vec3f q = localToUnit(o + d * t, box);
                            const float raw = std::max(0.0f, sampleTrilinear(values, nx, ny, nz, q.x, q.y, q.z));
                            const uint8_t solidCell = sampleSolidNearest(solid, nx, ny, nz, q.x, q.y, q.z);
                            if (!solidCell && raw >= iso && prevValid) {
                                float a = prevT;
                                float b = t;
                                for (int refine = 0; refine < 5; ++refine) {
                                    const float mid = 0.5f * (a + b);
                                    const Vec3f qm = localToUnit(o + d * mid, box);
                                    const float vm = std::max(0.0f, sampleTrilinear(values, nx, ny, nz, qm.x, qm.y, qm.z));
                                    if (vm >= iso) {
                                        b = mid;
                                    } else {
                                        a = mid;
                                    }
                                }
                                hitQ = localToUnit(o + d * b, box);
                                hit = true;
                                break;
                            }
                            prevT = t;
                            prevValid = !solidCell;
                            t += step;
                        }

                        if (hit) {
                            Vec3f grad = sampleGradient(values, nx, ny, nz, hitQ);
                            grad.x /= std::max(1e-6f, box.hx * 2.0f);
                            grad.y /= std::max(1e-6f, box.hy * 2.0f);
                            grad.z /= std::max(1e-6f, box.hz * 2.0f);
                            Vec3f normal = normalize3(grad);
                            if (dot3(normal, d) > 0.0f) normal = normal * -1.0f;

                            const float ndl = std::clamp(dot3(normal, lightDir), 0.0f, 1.0f);
                            const float rim = std::pow(std::clamp(1.0f - std::fabs(dot3(normal, d)), 0.0f, 1.0f), 2.0f);
                            const float spec = std::pow(std::clamp(dot3(normal, viewLight), 0.0f, 1.0f), 18.0f);

                            if (isDarkTheme(water.themeMode)) {
                                color.x = 0.08f + 0.16f * ndl + 0.12f * rim + 0.22f * spec;
                                color.y = 0.26f + 0.32f * ndl + 0.12f * rim + 0.18f * spec;
                                color.z = 0.52f + 0.36f * ndl + 0.14f * rim + 0.20f * spec;
                            } else {
                                color.x = 0.36f + 0.16f * ndl + 0.08f * rim + 0.18f * spec;
                                color.y = 0.52f + 0.20f * ndl + 0.08f * rim + 0.16f * spec;
                                color.z = 0.72f + 0.18f * ndl + 0.10f * rim + 0.18f * spec;
                            }
                        }
                    } else {
                        float accumR = 0.0f;
                        float accumG = 0.0f;
                        float accumB = 0.0f;
                        float accumA = 0.0f;
                        float firstSurfaceT = -1.0f;

                        while (t < tmax && accumA < 0.995f) {
                            const Vec3f q = localToUnit(o + d * t, box);
                            const float raw = std::max(0.0f, sampleTrilinear(values, nx, ny, nz, q.x, q.y, q.z));
                            const uint8_t solidCell = sampleSolidNearest(solid, nx, ny, nz, q.x, q.y, q.z);
                            if (!solidCell && raw > 1e-4f) {
                                if (firstSurfaceT < 0.0f && raw >= iso) {
                                    firstSurfaceT = t;
                                }
                                const float sigma = raw * sigmaScale * 3.2f;
                                const float aStep = (1.0f - std::exp(-sigma * step * 3.0f)) * water.alpha;

                                Vec3f grad = sampleGradient(values, nx, ny, nz, q);
                                grad.x /= std::max(1e-6f, box.hx * 2.0f);
                                grad.y /= std::max(1e-6f, box.hy * 2.0f);
                                grad.z /= std::max(1e-6f, box.hz * 2.0f);
                                Vec3f normal = normalize3(grad);
                                if (dot3(normal, d) > 0.0f) normal = normal * -1.0f;

                                const float ndl = std::clamp(dot3(normal, lightDir), 0.0f, 1.0f);
                                const float rim = std::pow(std::clamp(1.0f - std::fabs(dot3(normal, d)), 0.0f, 1.0f), 2.0f);
                                const float depthFade = (firstSurfaceT < 0.0f)
                                    ? 1.0f
                                    : (0.92f - 0.20f * std::clamp((t - firstSurfaceT) / std::max(1e-6f, tmax - firstSurfaceT), 0.0f, 1.0f));

                                float cr, cg, cb;
                                if (isDarkTheme(water.themeMode)) {
                                    cr = (0.07f + 0.10f * ndl + 0.08f * rim) * depthFade;
                                    cg = (0.22f + 0.24f * ndl + 0.10f * rim) * depthFade;
                                    cb = (0.44f + 0.30f * ndl + 0.12f * rim) * depthFade;
                                } else {
                                    cr = (0.34f + 0.10f * ndl + 0.06f * rim) * depthFade;
                                    cg = (0.50f + 0.12f * ndl + 0.06f * rim) * depthFade;
                                    cb = (0.70f + 0.12f * ndl + 0.08f * rim) * depthFade;
                                }

                                const float oneMinusA = 1.0f - accumA;
                                accumR += oneMinusA * aStep * cr;
                                accumG += oneMinusA * aStep * cg;
                                accumB += oneMinusA * aStep * cb;
                                accumA += oneMinusA * aStep;
                            }
                            t += step;
                        }

                        if (accumA > 0.0f) {
                            color.x = color.x * (1.0f - accumA) + accumR;
                            color.y = color.y * (1.0f - accumA) + accumG;
                            color.z = color.z * (1.0f - accumA) + accumB;
                        }
                    }
                }

                const int dst = (j * m_w + i) * 4;
                img[(std::size_t)dst + 0] = (uint8_t)std::lround(std::clamp(color.x, 0.0f, 1.0f) * 255.0f);
                img[(std::size_t)dst + 1] = (uint8_t)std::lround(std::clamp(color.y, 0.0f, 1.0f) * 255.0f);
                img[(std::size_t)dst + 2] = (uint8_t)std::lround(std::clamp(color.z, 0.0f, 1.0f) * 255.0f);
                img[(std::size_t)dst + 3] = 255;
            }
        }
    });

    glBindTexture(GL_TEXTURE_2D, m_waterTex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_w, m_h, GL_RGBA, GL_UNSIGNED_BYTE, img.data());
}
