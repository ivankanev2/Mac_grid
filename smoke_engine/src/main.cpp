#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <array>
#include <vector>
#include <filesystem>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <cstring>
#include <functional>

#include "UI/panels.h"

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

#include "Sim/mac_smoke_sim.h"
#include "Sim/mac_water_sim.h"
#include "Sim/mac_water3d.h"
#include "Sim/mac_smoke3d.h"
#include "Sim/pressure_solver.h"
#include "Renderer/smoke_renderer.h"

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

#include "Sim/mac_coupled_sim.h"
#include "Sim/text_mask.h"



// window size and sim resolution
static int NX = 256;
static int NY = 256;
static const int NZ = 64;

// Persistent text mask (reused across resets).
// Always rasterized at at least 512x512 so the letters stay crisp at any grid resolution.
static std::vector<uint8_t> g_textMask;
static int g_textMaskW = 0;
static int g_textMaskH = 0;

// time variables
double lastTime = 0.0;
double accumulator = 0.0;
float simSpeed = 1.0f;          // 1.0 = real-time, 2.0 = 2x faster, etc.
int maxStepsPerFrame = 8;       // cap to avoid death-spiral

namespace {

struct IconVec2 {
    float x = 0.0f;
    float y = 0.0f;
};

struct IconColor {
    unsigned char r = 0;
    unsigned char g = 0;
    unsigned char b = 0;
    unsigned char a = 0;
};

static constexpr int kThemeDark = 0;
static constexpr int kThemeLight = 1;
static constexpr IconColor kLogoRed{232, 29, 47, 255};
static constexpr IconColor kLogoLightBg{250, 250, 249, 255};
static constexpr IconColor kLogoDarkBg{23, 27, 31, 255};

static float clamp01(float x) {
    return std::max(0.0f, std::min(1.0f, x));
}

static float distSqToSegment(float px, float py, IconVec2 a, IconVec2 b) {
    const float vx = b.x - a.x;
    const float vy = b.y - a.y;
    const float wx = px - a.x;
    const float wy = py - a.y;
    const float vv = vx * vx + vy * vy;
    const float t = (vv > 1.0e-12f) ? clamp01((wx * vx + wy * vy) / vv) : 0.0f;
    const float dx = px - (a.x + t * vx);
    const float dy = py - (a.y + t * vy);
    return dx * dx + dy * dy;
}

static float distToPolyline(float px, float py, const IconVec2* pts, int count) {
    float best = 1.0e9f;
    for (int i = 0; i + 1 < count; ++i) {
        best = std::min(best, distSqToSegment(px, py, pts[i], pts[i + 1]));
    }
    return std::sqrt(best);
}

static float coverageForDisk(float dist, float radius, float aa) {
    if (dist <= radius - aa) return 1.0f;
    if (dist >= radius + aa) return 0.0f;
    return clamp01((radius + aa - dist) / std::max(1.0e-6f, 2.0f * aa));
}

static float coverageForStroke(float dist, float halfWidth, float aa) {
    if (dist <= halfWidth - aa) return 1.0f;
    if (dist >= halfWidth + aa) return 0.0f;
    return clamp01((halfWidth + aa - dist) / std::max(1.0e-6f, 2.0f * aa));
}

static void blendOver(unsigned char& dstR,
                      unsigned char& dstG,
                      unsigned char& dstB,
                      unsigned char& dstA,
                      IconColor src,
                      float coverage)
{
    const float sa = clamp01((src.a / 255.0f) * coverage);
    const float da = dstA / 255.0f;
    const float outA = sa + da * (1.0f - sa);
    if (outA <= 1.0e-6f) {
        dstR = dstG = dstB = dstA = 0;
        return;
    }

    const float sr = src.r / 255.0f;
    const float sg = src.g / 255.0f;
    const float sb = src.b / 255.0f;
    const float dr = dstR / 255.0f;
    const float dg = dstG / 255.0f;
    const float db = dstB / 255.0f;

    const float outR = (sr * sa + dr * da * (1.0f - sa)) / outA;
    const float outG = (sg * sa + dg * da * (1.0f - sa)) / outA;
    const float outB = (sb * sa + db * da * (1.0f - sa)) / outA;

    dstR = (unsigned char)std::lround(clamp01(outR) * 255.0f);
    dstG = (unsigned char)std::lround(clamp01(outG) * 255.0f);
    dstB = (unsigned char)std::lround(clamp01(outB) * 255.0f);
    dstA = (unsigned char)std::lround(clamp01(outA) * 255.0f);
}

static std::string findExistingPath(std::initializer_list<const char*> candidates) {
    for (const char* rel : candidates) {
        if (rel && rel[0] && std::filesystem::exists(rel)) {
            return std::string(rel);
        }
    }
    return {};
}

static bool loadViziorFonts(ImGuiIO& io, float dpiScale = 1.0f) {
    io.Fonts->Clear();

    ImFontConfig cfg{};
    cfg.OversampleH = 4;
    cfg.OversampleV = 4;
    cfg.PixelSnapH = false;
    cfg.RasterizerMultiply = 1.08f;

    const float fontSize = 17.0f * std::max(1.0f, dpiScale);
    const std::string roboto = findExistingPath({
        "external/imgui/misc/fonts/Roboto-Medium.ttf",
        "../external/imgui/misc/fonts/Roboto-Medium.ttf",
        "../../external/imgui/misc/fonts/Roboto-Medium.ttf"
    });

    if (!roboto.empty()) {
        if (ImFont* font = io.Fonts->AddFontFromFileTTF(roboto.c_str(), fontSize, &cfg)) {
            io.FontDefault = font;
            return true;
        }
    }

    io.FontDefault = io.Fonts->AddFontDefault();
    return false;
}

static void rasterViziorIcon(int w, int h, int themeMode, std::vector<unsigned char>& out) {
    out.assign((size_t)w * (size_t)h * 4u, 0u);

    const IconColor circleFill = (themeMode == kThemeLight) ? kLogoLightBg : kLogoDarkBg;
    const float aa = 0.90f / (float)std::max(1, std::max(w, h));
    const float circleR = 0.43f;
    const float borderHalf = 0.014f;

    const IconVec2 serifL[3] = {{0.24f, 0.29f}, {0.36f, 0.29f}, {0.40f, 0.33f}};
    const IconVec2 serifR[3] = {{0.76f, 0.29f}, {0.64f, 0.29f}, {0.60f, 0.33f}};
    const IconVec2 outerV[3] = {{0.33f, 0.31f}, {0.50f, 0.72f}, {0.67f, 0.31f}};
    const IconVec2 innerCutV[3] = {{0.40f, 0.35f}, {0.50f, 0.61f}, {0.60f, 0.35f}};
    const IconVec2 innerCoreV[3] = {{0.45f, 0.40f}, {0.50f, 0.53f}, {0.55f, 0.40f}};

    for (int j = 0; j < h; ++j) {
        for (int i = 0; i < w; ++i) {
            const float x = (i + 0.5f) / (float)w;
            const float y = (j + 0.5f) / (float)h;
            const float dx = x - 0.5f;
            const float dy = y - 0.5f;
            const float dCircle = std::sqrt(dx * dx + dy * dy);
            unsigned char r = 0, g = 0, b = 0, a = 0;

            const float circleCov = coverageForDisk(dCircle, circleR, aa);
            blendOver(r, g, b, a, circleFill, circleCov);

            if (themeMode == kThemeLight) {
                const float ringCov = circleCov * coverageForStroke(std::fabs(dCircle - circleR), borderHalf, aa * 1.6f);
                blendOver(r, g, b, a, kLogoRed, ringCov);
            }

            const float serifDist = std::min(distToPolyline(x, y, serifL, 3), distToPolyline(x, y, serifR, 3));
            const float outerDist = std::min(distToPolyline(x, y, outerV, 3), serifDist);
            const float cutDist = distToPolyline(x, y, innerCutV, 3);
            const float coreDist = distToPolyline(x, y, innerCoreV, 3);

            blendOver(r, g, b, a, kLogoRed, circleCov * coverageForStroke(outerDist, 0.072f, aa * 1.8f));
            blendOver(r, g, b, a, circleFill, circleCov * coverageForStroke(cutDist, 0.046f, aa * 1.6f));
            blendOver(r, g, b, a, kLogoRed, circleCov * coverageForStroke(coreDist, 0.022f, aa * 1.4f));

            const size_t idx = ((size_t)j * (size_t)w + (size_t)i) * 4u;
            out[idx + 0] = r;
            out[idx + 1] = g;
            out[idx + 2] = b;
            out[idx + 3] = a;
        }
    }
}

static void setViziorWindowIcon(GLFWwindow* win, int themeMode) {
    std::vector<unsigned char> icon32;
    std::vector<unsigned char> icon64;
    rasterViziorIcon(32, 32, themeMode, icon32);
    rasterViziorIcon(64, 64, themeMode, icon64);
    GLFWimage images[2];
    images[0].width = 32;
    images[0].height = 32;
    images[0].pixels = icon32.data();
    images[1].width = 64;
    images[1].height = 64;
    images[1].pixels = icon64.data();
    glfwSetWindowIcon(win, 2, images);
}

struct StartupWindowConfig {
    int width = 1480;
    int height = 920;
    int posX = 0;
    int posY = 0;
    float uiScale = 0.80f;
    bool hasPosition = false;
};

static StartupWindowConfig computeStartupWindowConfig() {
    StartupWindowConfig cfg;
    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    if (!monitor) return cfg;

    int workX = 0, workY = 0, workW = 0, workH = 0;
    glfwGetMonitorWorkarea(monitor, &workX, &workY, &workW, &workH);
    if (workW <= 0 || workH <= 0) return cfg;

    const float fit = std::clamp(std::min((float)workW / 1480.0f, (float)workH / 920.0f), 0.72f, 1.18f);
    cfg.width = std::clamp((int)std::lround(1480.0f * fit), 960, std::max(960, workW));
    cfg.height = std::clamp((int)std::lround(920.0f * fit), 640, std::max(640, workH));
    cfg.posX = workX + std::max(0, (workW - cfg.width) / 2);
    cfg.posY = workY + std::max(0, (workH - cfg.height) / 2);
    cfg.uiScale = std::clamp(0.80f * std::sqrt(std::max(0.55f, fit)), 0.65f, 0.95f);
    cfg.hasPosition = true;
    return cfg;
}

struct VolumeRenderTargetSize {
    int width = 0;
    int height = 0;
};

static bool nearlyEqual(float a, float b, float eps = 1.0e-4f) {
    return std::fabs(a - b) <= eps;
}

struct VolumeRenderCacheState {
    bool valid = false;
    double lastRenderTime = 0.0;
    unsigned long long lastSimVersion = 0;
    int lastWidth = 0;
    int lastHeight = 0;
    int lastViewMode = -1;
    int lastSliceAxis = -1;
    int lastSliceIndex = -1;
    int lastDebugField = -1;
    float lastYaw = 0.0f;
    float lastPitch = 0.0f;
    float lastZoom = 0.0f;
    float lastDensity = 0.0f;
    float lastSurfaceThreshold = 0.0f;
};

static VolumeRenderTargetSize computeVolumeRenderTargetSize(GLFWwindow* win,
                                                            float logicalWidth,
                                                            float logicalHeight,
                                                            float renderScale,
                                                            int minDim = 192,
                                                            int maxDim = 1024)
{
    int fbW = 0;
    int fbH = 0;
    glfwGetFramebufferSize(win, &fbW, &fbH);

    float dpiX = 1.0f;
    float dpiY = 1.0f;
    glfwGetWindowContentScale(win, &dpiX, &dpiY);
    dpiX = std::max(1.0f, dpiX);
    dpiY = std::max(1.0f, dpiY);

    const float fallbackLogicalW = std::max(320.0f, 0.46f * (float)std::max(1, fbW) / dpiX);
    const float fallbackLogicalH = std::max(220.0f, 0.46f * (float)std::max(1, fbH) / dpiY);
    const float safeLogicalW = (logicalWidth > 1.0f) ? logicalWidth : fallbackLogicalW;
    const float safeLogicalH = (logicalHeight > 1.0f) ? logicalHeight : fallbackLogicalH;
    const float safeScale = std::clamp(renderScale, 0.35f, 2.0f);

    int targetW = (int)std::lround(safeLogicalW * dpiX * safeScale);
    int targetH = (int)std::lround(safeLogicalH * dpiY * safeScale);

    targetW = std::clamp(targetW, minDim, maxDim);
    targetH = std::clamp(targetH, minDim, maxDim);
    return VolumeRenderTargetSize{targetW, targetH};
}


struct OfflineFitRect {
    int x = 0;
    int y = 0;
    int width = 0;
    int height = 0;
};

struct OfflineFrameImage {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> rgba;
};

static std::array<uint8_t, 4> themeCanvasColor(int themeMode) {
    if (themeMode == kThemeLight) {
        return {242u, 241u, 239u, 255u};
    }
    return {14u, 17u, 20u, 255u};
}

static std::vector<uint8_t> makeSolidCanvasRGBA(int width,
                                                int height,
                                                uint8_t r,
                                                uint8_t g,
                                                uint8_t b,
                                                uint8_t a = 255)
{
    std::vector<uint8_t> out((std::size_t)std::max(0, width) * (std::size_t)std::max(0, height) * 4u, 0u);
    for (std::size_t i = 0; i + 3 < out.size(); i += 4u) {
        out[i + 0] = r;
        out[i + 1] = g;
        out[i + 2] = b;
        out[i + 3] = a;
    }
    return out;
}

static void alphaCompositeOverOpaque(std::vector<uint8_t>& dst,
                                     const std::vector<uint8_t>& src,
                                     float globalAlpha = 1.0f)
{
    if (dst.size() != src.size()) return;
    const float alphaScale = std::clamp(globalAlpha, 0.0f, 1.0f);
    const std::size_t count = dst.size() / 4u;
    for (std::size_t px = 0; px < count; ++px) {
        const std::size_t idx = px * 4u;
        const float sa = (src[idx + 3] / 255.0f) * alphaScale;
        if (sa <= 1.0e-6f) continue;
        const float inv = 1.0f - sa;
        dst[idx + 0] = (uint8_t)std::lround(std::clamp(src[idx + 0] * sa + dst[idx + 0] * inv, 0.0f, 255.0f));
        dst[idx + 1] = (uint8_t)std::lround(std::clamp(src[idx + 1] * sa + dst[idx + 1] * inv, 0.0f, 255.0f));
        dst[idx + 2] = (uint8_t)std::lround(std::clamp(src[idx + 2] * sa + dst[idx + 2] * inv, 0.0f, 255.0f));
        dst[idx + 3] = 255;
    }
}

static bool readTextureRGBA(unsigned int texture,
                            int width,
                            int height,
                            std::vector<uint8_t>& out)
{
    if (texture == 0u || width <= 0 || height <= 0) return false;
    out.resize((std::size_t)width * (std::size_t)height * 4u);
    glBindTexture(GL_TEXTURE_2D, texture);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, out.data());
    return glGetError() == GL_NO_ERROR;
}

static OfflineFitRect computeAspectFitRect(int srcWidth,
                                           int srcHeight,
                                           int dstWidth,
                                           int dstHeight)
{
    OfflineFitRect rect;
    if (srcWidth <= 0 || srcHeight <= 0 || dstWidth <= 0 || dstHeight <= 0) return rect;
    const float scale = std::min((float)dstWidth / (float)srcWidth,
                                 (float)dstHeight / (float)srcHeight);
    rect.width = std::max(1, (int)std::lround((float)srcWidth * scale));
    rect.height = std::max(1, (int)std::lround((float)srcHeight * scale));
    rect.x = std::max(0, (dstWidth - rect.width) / 2);
    rect.y = std::max(0, (dstHeight - rect.height) / 2);
    return rect;
}

static std::vector<uint8_t> scaleOpaqueRGBAWithLetterbox(const std::vector<uint8_t>& src,
                                                         int srcWidth,
                                                         int srcHeight,
                                                         int dstWidth,
                                                         int dstHeight,
                                                         uint8_t bgR,
                                                         uint8_t bgG,
                                                         uint8_t bgB,
                                                         OfflineFitRect* outRect = nullptr)
{
    std::vector<uint8_t> dst = makeSolidCanvasRGBA(dstWidth, dstHeight, bgR, bgG, bgB, 255);
    if (srcWidth <= 0 || srcHeight <= 0 || src.size() != (std::size_t)srcWidth * (std::size_t)srcHeight * 4u) {
        if (outRect) *outRect = {};
        return dst;
    }

    const OfflineFitRect rect = computeAspectFitRect(srcWidth, srcHeight, dstWidth, dstHeight);
    if (outRect) *outRect = rect;
    if (rect.width <= 0 || rect.height <= 0) return dst;

    auto sample = [&](float x, float y, int channel) -> float {
        x = std::clamp(x, 0.0f, (float)std::max(0, srcWidth - 1));
        y = std::clamp(y, 0.0f, (float)std::max(0, srcHeight - 1));
        const int x0 = (int)std::floor(x);
        const int y0 = (int)std::floor(y);
        const int x1 = std::min(x0 + 1, srcWidth - 1);
        const int y1 = std::min(y0 + 1, srcHeight - 1);
        const float tx = x - (float)x0;
        const float ty = y - (float)y0;
        auto at = [&](int sx, int sy) -> float {
            return (float)src[((std::size_t)sy * (std::size_t)srcWidth + (std::size_t)sx) * 4u + (std::size_t)channel];
        };
        const float c00 = at(x0, y0);
        const float c10 = at(x1, y0);
        const float c01 = at(x0, y1);
        const float c11 = at(x1, y1);
        const float c0 = c00 * (1.0f - tx) + c10 * tx;
        const float c1 = c01 * (1.0f - tx) + c11 * tx;
        return c0 * (1.0f - ty) + c1 * ty;
    };

    for (int y = 0; y < rect.height; ++y) {
        const float srcY = ((y + 0.5f) / (float)rect.height) * (float)srcHeight - 0.5f;
        for (int x = 0; x < rect.width; ++x) {
            const float srcX = ((x + 0.5f) / (float)rect.width) * (float)srcWidth - 0.5f;
            const std::size_t dstIdx = ((std::size_t)(rect.y + y) * (std::size_t)dstWidth + (std::size_t)(rect.x + x)) * 4u;
            dst[dstIdx + 0] = (uint8_t)std::lround(sample(srcX, srcY, 0));
            dst[dstIdx + 1] = (uint8_t)std::lround(sample(srcX, srcY, 1));
            dst[dstIdx + 2] = (uint8_t)std::lround(sample(srcX, srcY, 2));
            dst[dstIdx + 3] = 255;
        }
    }
    return dst;
}

static void drawFilledCircleOnCanvas(std::vector<uint8_t>& canvas,
                                     int canvasWidth,
                                     int canvasHeight,
                                     float cx,
                                     float cy,
                                     float radius,
                                     uint8_t r,
                                     uint8_t g,
                                     uint8_t b,
                                     uint8_t a)
{
    if (canvasWidth <= 0 || canvasHeight <= 0 || radius <= 0.0f) return;
    const int x0 = std::max(0, (int)std::floor(cx - radius - 1.0f));
    const int x1 = std::min(canvasWidth - 1, (int)std::ceil(cx + radius + 1.0f));
    const int y0 = std::max(0, (int)std::floor(cy - radius - 1.0f));
    const int y1 = std::min(canvasHeight - 1, (int)std::ceil(cy + radius + 1.0f));
    const float alpha = a / 255.0f;
    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            const float dx = (x + 0.5f) - cx;
            const float dy = (y + 0.5f) - cy;
            const float dist2 = dx * dx + dy * dy;
            if (dist2 > radius * radius) continue;
            const std::size_t idx = ((std::size_t)y * (std::size_t)canvasWidth + (std::size_t)x) * 4u;
            const float inv = 1.0f - alpha;
            canvas[idx + 0] = (uint8_t)std::lround(std::clamp(r * alpha + canvas[idx + 0] * inv, 0.0f, 255.0f));
            canvas[idx + 1] = (uint8_t)std::lround(std::clamp(g * alpha + canvas[idx + 1] * inv, 0.0f, 255.0f));
            canvas[idx + 2] = (uint8_t)std::lround(std::clamp(b * alpha + canvas[idx + 2] * inv, 0.0f, 255.0f));
            canvas[idx + 3] = 255;
        }
    }
}

template <typename ParticleContainer>
static void drawParticlesOnCanvas(std::vector<uint8_t>& canvas,
                                  int canvasWidth,
                                  int canvasHeight,
                                  const OfflineFitRect& rect,
                                  const ParticleContainer& particles,
                                  float domainWidth,
                                  float domainHeight)
{
    if (particles.empty() || rect.width <= 0 || rect.height <= 0) return;
    const float safeDomainX = std::max(1.0e-6f, domainWidth);
    const float safeDomainY = std::max(1.0e-6f, domainHeight);
    const float radius = std::max(2.0f, 0.0035f * (float)std::max(rect.width, rect.height));
    for (const auto& p : particles) {
        const float px = p.x / safeDomainX;
        const float py = p.y / safeDomainY;
        if (px < 0.0f || px > 1.0f || py < 0.0f || py > 1.0f) continue;
        const float sx = rect.x + px * (float)rect.width;
        const float sy = rect.y + (1.0f - py) * (float)rect.height;
        drawFilledCircleOnCanvas(canvas, canvasWidth, canvasHeight, sx, sy, radius, 255, 245, 120, 230);
    }
}

static bool writeTgaImage(const std::filesystem::path& path,
                          int width,
                          int height,
                          const std::vector<uint8_t>& rgba)
{
    if (width <= 0 || height <= 0) return false;
    if (rgba.size() != (std::size_t)width * (std::size_t)height * 4u) return false;

    std::ofstream out(path, std::ios::binary);
    if (!out) return false;

    uint8_t header[18] = {};
    header[2] = 2; // uncompressed true-color
    header[12] = (uint8_t)(width & 0xFF);
    header[13] = (uint8_t)((width >> 8) & 0xFF);
    header[14] = (uint8_t)(height & 0xFF);
    header[15] = (uint8_t)((height >> 8) & 0xFF);
    header[16] = 32;
    header[17] = 0x28; // 8-bit alpha + top-left origin
    out.write(reinterpret_cast<const char*>(header), sizeof(header));

    std::vector<uint8_t> bgra(rgba.size(), 0u);
    for (std::size_t i = 0; i + 3 < rgba.size(); i += 4u) {
        bgra[i + 0] = rgba[i + 2];
        bgra[i + 1] = rgba[i + 1];
        bgra[i + 2] = rgba[i + 0];
        bgra[i + 3] = rgba[i + 3];
    }
    out.write(reinterpret_cast<const char*>(bgra.data()), (std::streamsize)bgra.size());
    return (bool)out;
}

static std::string shellQuote(const std::filesystem::path& path) {
    std::string s = path.generic_string();
    std::string out;
    out.reserve(s.size() + 8u);
    out.push_back('"');
    for (char c : s) {
        if (c == '"') out += "\\\"";
        else out.push_back(c);
    }
    out.push_back('"');
    return out;
}

static bool captureWorkspaceFrame(int workspace,
                                  const UI::Settings& renderUi,
                                  MAC2D& sim,
                                  MACWater& waterSim,
                                  MACSmoke3D& smoke3D,
                                  MACWater3D& water3D,
                                  MACCoupledSim& coupled,
                                  SmokeRenderer& bakeRenderer,
                                  int outputWidth,
                                  int outputHeight,
                                  OfflineFrameImage& outImage,
                                  std::string& outError)
{
    outError.clear();
    outImage = {};
    const auto bg = themeCanvasColor(renderUi.themeMode);

    SmokeRenderSettings smokeRender;
    OverlaySettings overlay;
    UI::BuildRenderSettings(renderUi, smokeRender, overlay);
    WaterRenderSettings waterRender;
    UI::BuildWaterRenderSettings(renderUi, waterRender);

    auto finalize2DLike = [&](const std::vector<uint8_t>& rgba,
                              int srcWidth,
                              int srcHeight,
                              const std::function<void(std::vector<uint8_t>&, int, int, const OfflineFitRect&)>& decorate) {
        if (rgba.size() != (std::size_t)srcWidth * (std::size_t)srcHeight * 4u) {
            outError = "Texture readback size mismatch.";
            return false;
        }
        OfflineFitRect fit;
        outImage.width = outputWidth;
        outImage.height = outputHeight;
        outImage.rgba = scaleOpaqueRGBAWithLetterbox(rgba, srcWidth, srcHeight, outputWidth, outputHeight,
                                                     bg[0], bg[1], bg[2], &fit);
        decorate(outImage.rgba, outputWidth, outputHeight, fit);
        return true;
    };

    switch (workspace) {
        case UI::kWorkspaceSmoke2D: {
            bakeRenderer.resize(sim.nx, sim.ny);
            bakeRenderer.updateFromSim(sim, smokeRender, overlay);
            std::vector<uint8_t> smokeRGBA;
            if (!readTextureRGBA(bakeRenderer.smokeTex(), bakeRenderer.width(), bakeRenderer.height(), smokeRGBA)) {
                outError = "Failed to read Smoke 2D texture.";
                return false;
            }
            std::vector<uint8_t> opaque = makeSolidCanvasRGBA(sim.nx, sim.ny, bg[0], bg[1], bg[2], 255);
            alphaCompositeOverOpaque(opaque, smokeRGBA, 1.0f);
            if (overlay.showDiv) {
                std::vector<uint8_t> divRGBA;
                if (readTextureRGBA(bakeRenderer.divTex(), bakeRenderer.width(), bakeRenderer.height(), divRGBA)) {
                    alphaCompositeOverOpaque(opaque, divRGBA, 1.0f);
                }
            }
            if (overlay.showVort) {
                std::vector<uint8_t> vortRGBA;
                if (readTextureRGBA(bakeRenderer.vortTex(), bakeRenderer.width(), bakeRenderer.height(), vortRGBA)) {
                    alphaCompositeOverOpaque(opaque, vortRGBA, 1.0f);
                }
            }
            return finalize2DLike(opaque, sim.nx, sim.ny,
                                  [](std::vector<uint8_t>&, int, int, const OfflineFitRect&) {});
        }
        case UI::kWorkspaceWater2D: {
            bakeRenderer.resize(waterSim.nx, waterSim.ny);
            bakeRenderer.updateWaterFromSim(waterSim, waterRender);
            std::vector<uint8_t> waterRGBA;
            if (!readTextureRGBA(bakeRenderer.waterTex(), bakeRenderer.width(), bakeRenderer.height(), waterRGBA)) {
                outError = "Failed to read Water 2D texture.";
                return false;
            }
            std::vector<uint8_t> opaque = makeSolidCanvasRGBA(waterSim.nx, waterSim.ny, bg[0], bg[1], bg[2], 255);
            alphaCompositeOverOpaque(opaque, waterRGBA, 1.0f);
            return finalize2DLike(opaque, waterSim.nx, waterSim.ny,
                                  [&](std::vector<uint8_t>& canvas, int canvasW, int canvasH, const OfflineFitRect& fit) {
                                      if (renderUi.offlineBakeIncludeParticles && renderUi.showWaterParticles) {
                                          drawParticlesOnCanvas(canvas, canvasW, canvasH, fit,
                                                                waterSim.particles,
                                                                waterSim.nx * waterSim.dx,
                                                                waterSim.ny * waterSim.dx);
                                      }
                                  });
        }
        case UI::kWorkspaceSmoke3D: {
            const int viewMode = std::clamp(renderUi.smoke3DViewMode, 0, 1);
            if (viewMode == 1) {
                const int axis = std::clamp(renderUi.smoke3DSliceAxis, 0, 2);
                const int maxSlice = (axis == 0)
                    ? std::max(0, smoke3D.nz - 1)
                    : (axis == 1)
                        ? std::max(0, smoke3D.ny - 1)
                        : std::max(0, smoke3D.nx - 1);
                const int sliceIndex = std::clamp(renderUi.smoke3DSliceIndex, 0, maxSlice);
                const int field = std::clamp(renderUi.smoke3DDebugField, 0, 4);
                auto slice = smoke3D.copyDebugSlice(
                    static_cast<MACSmoke3D::SliceAxis>(axis),
                    sliceIndex,
                    static_cast<MACSmoke3D::DebugField>(field));
                bakeRenderer.updateSmokeFromSlice(slice.values, slice.solid, slice.width, slice.height, field, smokeRender);
                std::vector<uint8_t> smokeRGBA;
                if (!readTextureRGBA(bakeRenderer.smokeTex(), bakeRenderer.width(), bakeRenderer.height(), smokeRGBA)) {
                    outError = "Failed to read Smoke 3D slice texture.";
                    return false;
                }
                std::vector<uint8_t> opaque = makeSolidCanvasRGBA(slice.width, slice.height, bg[0], bg[1], bg[2], 255);
                alphaCompositeOverOpaque(opaque, smokeRGBA, 1.0f);
                return finalize2DLike(opaque, slice.width, slice.height,
                                      [](std::vector<uint8_t>&, int, int, const OfflineFitRect&) {});
            }
            bakeRenderer.resize(outputWidth, outputHeight);
            bakeRenderer.updateSmokeFromVolume(smoke3D.smoke, smoke3D.temp, smoke3D.solid,
                                               smoke3D.nx, smoke3D.ny, smoke3D.nz,
                                               renderUi.smoke3DViewYawDeg,
                                               renderUi.smoke3DViewPitchDeg,
                                               renderUi.smoke3DViewZoom,
                                               renderUi.smoke3DVolumeDensity,
                                               smokeRender);
            outImage.width = bakeRenderer.width();
            outImage.height = bakeRenderer.height();
            if (!readTextureRGBA(bakeRenderer.smokeTex(), outImage.width, outImage.height, outImage.rgba)) {
                outError = "Failed to read Smoke 3D volume texture.";
                return false;
            }
            return true;
        }
        case UI::kWorkspaceWater3D: {
            const int viewMode = std::clamp(renderUi.water3DViewMode, 0, 2);
            if (viewMode == 1) {
                const int axis = std::clamp(renderUi.water3DSliceAxis, 0, 2);
                const int maxSlice = (axis == 0)
                    ? std::max(0, water3D.nz - 1)
                    : (axis == 1)
                        ? std::max(0, water3D.ny - 1)
                        : std::max(0, water3D.nx - 1);
                const int sliceIndex = std::clamp(renderUi.water3DSliceIndex, 0, maxSlice);
                const int field = std::clamp(renderUi.water3DDebugField, 0, 3);
                auto slice = water3D.copyDebugSlice(
                    static_cast<MACWater3D::SliceAxis>(axis),
                    sliceIndex,
                    static_cast<MACWater3D::DebugField>(field));
                bakeRenderer.updateWaterFromSlice(slice.values, slice.solid, slice.width, slice.height, waterRender);
                std::vector<uint8_t> waterRGBA;
                if (!readTextureRGBA(bakeRenderer.waterTex(), bakeRenderer.width(), bakeRenderer.height(), waterRGBA)) {
                    outError = "Failed to read Water 3D slice texture.";
                    return false;
                }
                std::vector<uint8_t> opaque = makeSolidCanvasRGBA(slice.width, slice.height, bg[0], bg[1], bg[2], 255);
                alphaCompositeOverOpaque(opaque, waterRGBA, 1.0f);
                return finalize2DLike(opaque, slice.width, slice.height,
                                      [](std::vector<uint8_t>&, int, int, const OfflineFitRect&) {});
            }
            bakeRenderer.resize(outputWidth, outputHeight);
            bakeRenderer.updateWaterFromVolume(water3D.water, water3D.solid,
                                               water3D.nx, water3D.ny, water3D.nz,
                                               viewMode,
                                               renderUi.water3DViewYawDeg,
                                               renderUi.water3DViewPitchDeg,
                                               renderUi.water3DViewZoom,
                                               renderUi.water3DVolumeDensity,
                                               renderUi.water3DSurfaceThreshold,
                                               waterRender);
            outImage.width = bakeRenderer.width();
            outImage.height = bakeRenderer.height();
            if (!readTextureRGBA(bakeRenderer.waterTex(), outImage.width, outImage.height, outImage.rgba)) {
                outError = "Failed to read Water 3D texture.";
                return false;
            }
            return true;
        }
        case UI::kWorkspaceCoupled: {
            bakeRenderer.resize(coupled.nx, coupled.ny);
            bakeRenderer.updateFromSim(coupled, smokeRender, overlay);
            bakeRenderer.updateWaterFromSim(coupled, waterRender);
            std::vector<uint8_t> smokeRGBA;
            std::vector<uint8_t> waterRGBA;
            if (!readTextureRGBA(bakeRenderer.smokeTex(), bakeRenderer.width(), bakeRenderer.height(), smokeRGBA)) {
                outError = "Failed to read coupled smoke texture.";
                return false;
            }
            if (!readTextureRGBA(bakeRenderer.waterTex(), bakeRenderer.width(), bakeRenderer.height(), waterRGBA)) {
                outError = "Failed to read coupled water texture.";
                return false;
            }
            std::vector<uint8_t> opaque = makeSolidCanvasRGBA(coupled.nx, coupled.ny, bg[0], bg[1], bg[2], 255);
            alphaCompositeOverOpaque(opaque, smokeRGBA, 1.0f);
            alphaCompositeOverOpaque(opaque, waterRGBA, std::clamp(renderUi.combinedWaterAlpha, 0.0f, 1.0f));
            return finalize2DLike(opaque, coupled.nx, coupled.ny,
                                  [&](std::vector<uint8_t>& canvas, int canvasW, int canvasH, const OfflineFitRect& fit) {
                                      if (renderUi.offlineBakeIncludeParticles && renderUi.combinedShowParticles) {
                                          drawParticlesOnCanvas(canvas, canvasW, canvasH, fit,
                                                                coupled.particles,
                                                                coupled.nx * coupled.dx,
                                                                coupled.ny * coupled.dx);
                                      }
                                  });
        }
        default:
            break;
    }

    outError = "Unsupported workspace.";
    return false;
}

} // namespace

int main()
{
    if (!glfwInit()) return 1;
    lastTime = glfwGetTime();

    // OpenGL / GLFW hints
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);

    const StartupWindowConfig startupWindow = computeStartupWindowConfig();
    GLFWwindow* win = glfwCreateWindow(startupWindow.width, startupWindow.height, "Vizior | Fluid Research Engine", nullptr, nullptr);
    if (!win) return 1;
    if (startupWindow.hasPosition) {
        glfwSetWindowPos(win, startupWindow.posX, startupWindow.posY);
    }

    UI::Settings ui;
    ui.windowWidth = startupWindow.width;
    ui.windowHeight = startupWindow.height;
    ui.uiScale = startupWindow.uiScale;
    UI::Probe probe;
    glfwGetWindowSize(win, &ui.windowWidth, &ui.windowHeight);

    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);

    // ImGui init
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = "vizior_layout.ini";

    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags &= ~ImGuiConfigFlags_ViewportsEnable;   // disable OS-windows

    float dpiX = 1.0f;
    float dpiY = 1.0f;
    glfwGetWindowContentScale(win, &dpiX, &dpiY);
    loadViziorFonts(io, std::max(dpiX, dpiY));

    UI::ApplyViziorTheme(ui.themeMode);
    ImGui::GetStyle().ScaleAllSizes(ui.uiScale);
    io.FontGlobalScale = ui.uiScale;
    setViziorWindowIcon(win, ui.themeMode);
    int   appliedTheme   = ui.themeMode;
    float appliedUiScale = ui.uiScale;

    // tweak style when viewports enabled
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
        ImGuiStyle& style = ImGui::GetStyle();
        style.WindowRounding = 0.0f;
        style.Colors[ImGuiCol_WindowBg].w = 1.0f;
    }

    ImGui_ImplGlfw_InitForOpenGL(win, true);
    ImGui_ImplOpenGL3_Init("#version 150");

    // Renderer
    SmokeRenderer renderer(NX, NY);
    SmokeRenderer water3DRenderer(NX, NY);
    SmokeRenderer smoke3DRenderer(NX, NY);
    SmokeRenderer coupledRenderer(NX, NY);

    VolumeRenderCacheState smoke3DRenderCache;
    VolumeRenderCacheState water3DRenderCache;
    unsigned long long smoke3DSimVersion = 1;
    unsigned long long water3DSimVersion = 1;
    auto invalidate3DRenderCaches = [&]() {
        smoke3DRenderCache = {};
        water3DRenderCache = {};
    };

    // Simulator
    float dx = 1.0f / NX;
    float dt_initial = 0.02f;


    // One shared pressure solver instance used by BOTH smoke and water.
    // The solver is reconfigured per-solve (nx/ny/BCs/masks), but it can reuse
    // internal allocations across both sims.
    PressureSolver sharedPressureSolver;

    auto dxForGrid = [](int gx, int gy, int gz) {
        return 1.0f / (float)std::max({gx, gy, gz, 1});
    };

    MAC2D sim(NX, NY, dx, dt_initial);
    MACWater waterSim(NX, NY, dx, dt_initial);
    MACWater3D water3D(1, 1, 1, 1.0f, dt_initial);
    MACSmoke3D smoke3D(1, 1, 1, 1.0f, dt_initial);
    MACCoupledSim coupled(NX, NY, dx, dt_initial);

    // Make both sims use the same solver instance.
    sim.setSharedPressureSolver(&sharedPressureSolver);
    waterSim.setSharedPressureSolver(&sharedPressureSolver);
    coupled.setSharedPressureSolver(&sharedPressureSolver);

    // Rasterize MBZUAI text mask and apply to both simulations
    const std::string robotoPath = findExistingPath({
        "external/imgui/misc/fonts/Roboto-Medium.ttf",
        "../external/imgui/misc/fonts/Roboto-Medium.ttf",
        "../../external/imgui/misc/fonts/Roboto-Medium.ttf"
    });
    // Rasterize at the larger of (grid size, 512) so text stays sharp at any resolution.
    g_textMaskW = std::max(NX, 512);
    g_textMaskH = std::max(NY, 512);
    g_textMask = rasterizeTextMask(
        "MBZUAI",
        robotoPath.empty() ? "external/imgui/misc/fonts/Roboto-Medium.ttf" : robotoPath.c_str(),
        g_textMaskW, g_textMaskH,
        0.5f,   // center vertically
        0.15f   // text height = 15% of mask height
    );

    auto applyIntroText = [&]() {
        if (g_textMask.empty()) return;
        sim.addSolidText(g_textMask, g_textMaskW, g_textMaskH);
        waterSim.waterHeld = true;
        waterSim.addWaterTextParticles(g_textMask, g_textMaskW, g_textMaskH, 4);
        waterSim.syncSolidsFrom(sim);
    };

    auto activateWorkspace = [&](int workspace) {
        workspace = std::clamp(workspace, (int)UI::kWorkspaceSmoke2D, (int)UI::kWorkspaceCoupled);

        sim.reset();
        waterSim.reset();
        coupled.reset();
        applyIntroText();

        smoke3D.reset(1, 1, 1, 1.0f, smoke3D.dt);
        water3D.reset(1, 1, 1, 1.0f, water3D.dt);

        if (workspace == UI::kWorkspaceSmoke3D) {
            const int nx3 = std::max(1, ui.smoke3DNX);
            const int ny3 = std::max(1, ui.smoke3DNY);
            const int nz3 = std::max(1, ui.smoke3DNZ);
            smoke3D.reset(nx3, ny3, nz3, dxForGrid(nx3, ny3, nz3), smoke3D.dt);
        } else if (workspace == UI::kWorkspaceWater3D) {
            const int nx3 = std::max(1, ui.water3DNX);
            const int ny3 = std::max(1, ui.water3DNY);
            const int nz3 = std::max(1, ui.water3DNZ);
            water3D.reset(nx3, ny3, nz3, dxForGrid(nx3, ny3, nz3), water3D.dt);
        }

        ++smoke3DSimVersion;
        ++water3DSimVersion;
        invalidate3DRenderCaches();
        accumulator = 0.0;
    };

    int activeWorkspace = std::clamp(ui.activeWorkspace, (int)UI::kWorkspaceSmoke2D, (int)UI::kWorkspaceCoupled);
    activateWorkspace(activeWorkspace);

    // Reinitialise all 2D sims at a new grid resolution.
    auto reinit2DGrid = [&]() {
        NX = std::max(32, ui.sim2DNX);
        NY = std::max(32, ui.sim2DNY);
        const float newDX = 1.0f / NX;

        sim      = MAC2D(NX, NY, newDX, dt_initial);
        waterSim = MACWater(NX, NY, newDX, dt_initial);
        coupled  = MACCoupledSim(NX, NY, newDX, dt_initial);
        sim.setSharedPressureSolver(&sharedPressureSolver);
        waterSim.setSharedPressureSolver(&sharedPressureSolver);
        coupled.setSharedPressureSolver(&sharedPressureSolver);

        renderer.resize(NX, NY);
        water3DRenderer.resize(NX, NY);
        smoke3DRenderer.resize(NX, NY);
        coupledRenderer.resize(NX, NY);
        invalidate3DRenderCaches();

        g_textMaskW = std::max(NX, 512);
        g_textMaskH = std::max(NY, 512);
        g_textMask = rasterizeTextMask(
            "MBZUAI",
            robotoPath.empty() ? "external/imgui/misc/fonts/Roboto-Medium.ttf" : robotoPath.c_str(),
            g_textMaskW, g_textMaskH, 0.5f, 0.15f);

        // Preserve the current zoom. Lower-resolution 2D grids now draw smaller
        // instead of being auto-stretched to the old on-screen footprint.
        ui.viewScale = std::clamp(ui.viewScale, 0.5f, 4.0f);

        activateWorkspace(activeWorkspace);
    };

    SmokeRenderer offlineBakeRenderer(1, 1);
    struct OfflineBakeRuntime {
        bool running = false;
        bool cancelRequested = false;
        int nextFrame = 0;
        int totalFrames = 0;
        int workspace = UI::kWorkspaceSmoke2D;
        int outputWidth = 0;
        int outputHeight = 0;
        int videoFPS = 30;
        int simStepsPerFrame = 1;
        float fixedDt = 1.0f / 120.0f;
        bool encodeVideo = true;
        bool keepImageSequence = true;
        std::filesystem::path jobDir;
        std::filesystem::path framesDir;
        std::filesystem::path videoPath;
        std::string framePrefix;
        UI::Settings snapshot{};
        double startTime = 0.0;
    };
    OfflineBakeRuntime offlineBake;

    auto setOfflineStatus = [&](const std::string& message) {
        std::snprintf(ui.offlineBakeStatus, sizeof(ui.offlineBakeStatus), "%s", message.c_str());
    };
    setOfflineStatus("Idle");

    auto writeOfflineBakeManifest = [&](const OfflineBakeRuntime& job) {
        try {
            std::ofstream manifest(job.jobDir / "job.txt", std::ios::out | std::ios::trunc);
            if (!manifest) return;
            const std::time_t nowTs = std::time(nullptr);
            manifest << "Vizior Offline Bake\n";
            manifest << "Workspace: " << UI::ActiveWorkspaceLabel(job.workspace) << "\n";
            manifest << "Frames: " << job.totalFrames << "\n";
            manifest << "Output resolution: " << job.outputWidth << " x " << job.outputHeight << "\n";
            manifest << "Video FPS: " << job.videoFPS << "\n";
            manifest << "Sim steps per frame: " << job.simStepsPerFrame << "\n";
            manifest << "Fixed sim dt: " << job.fixedDt << "\n";
            manifest << "Frame prefix: " << job.framePrefix << "\n";
            manifest << "Encode video: " << (job.encodeVideo ? "yes" : "no") << "\n";
            manifest << "Keep image sequence: " << (job.keepImageSequence ? "yes" : "no") << "\n";
            manifest << "Created: " << std::asctime(std::localtime(&nowTs));
        } catch (...) {
        }
    };

    auto finishOfflineBake = [&](const std::string& status) {
        offlineBake.running = false;
        offlineBake.cancelRequested = false;
        ui.offlineBakeRunning = false;
        ui.offlineBakeCurrentFrame = std::min(ui.offlineBakeFrameCount, offlineBake.nextFrame);
        ++smoke3DSimVersion;
        ++water3DSimVersion;
        invalidate3DRenderCaches();
        setOfflineStatus(status);
    };

    auto startOfflineBake = [&]() {
        if (offlineBake.running) return;

        const std::string outputDir = std::strlen(ui.offlineBakeOutputDir) > 0
            ? std::string(ui.offlineBakeOutputDir)
            : std::string("offline_bakes/latest");
        const std::string framePrefix = std::strlen(ui.offlineBakeFramePrefix) > 0
            ? std::string(ui.offlineBakeFramePrefix)
            : std::string("frame");
        std::string videoFile = std::strlen(ui.offlineBakeVideoFile) > 0
            ? std::string(ui.offlineBakeVideoFile)
            : std::string("preview.mp4");
        if (std::filesystem::path(videoFile).extension().empty()) {
            videoFile += ".mp4";
        }

        OfflineBakeRuntime job;
        job.workspace = activeWorkspace;
        job.totalFrames = std::max(1, ui.offlineBakeFrameCount);
        job.outputWidth = std::max(64, ui.offlineBakeWidth);
        job.outputHeight = std::max(64, ui.offlineBakeHeight);
        if (ui.offlineBakeAutoEncodeVideo) {
            if ((job.outputWidth & 1) != 0) ++job.outputWidth;
            if ((job.outputHeight & 1) != 0) ++job.outputHeight;
        }
        job.videoFPS = std::max(1, ui.offlineBakeVideoFPS);
        job.simStepsPerFrame = std::max(1, ui.offlineBakeSimStepsPerFrame);
        job.fixedDt = std::clamp(ui.offlineBakeFixedDt, 1.0e-4f, 0.1f);
        job.encodeVideo = ui.offlineBakeAutoEncodeVideo;
        job.keepImageSequence = ui.offlineBakeKeepImageSequence;
        job.framePrefix = framePrefix;
        job.snapshot = ui;
        job.snapshot.activeWorkspace = activeWorkspace;
        job.jobDir = std::filesystem::path(outputDir);
        job.framesDir = job.jobDir / "frames";
        job.videoPath = job.jobDir / videoFile;
        job.startTime = glfwGetTime();

        try {
            std::filesystem::create_directories(job.jobDir);
            if (std::filesystem::exists(job.framesDir)) {
                std::filesystem::remove_all(job.framesDir);
            }
            std::filesystem::create_directories(job.framesDir);
            std::error_code ec;
            std::filesystem::remove(job.videoPath, ec);
        } catch (const std::exception& e) {
            setOfflineStatus(std::string("Bake setup failed: ") + e.what());
            return;
        }

        offlineBake = std::move(job);
        offlineBake.running = true;
        ui.playing = false;
        ui.offlineBakeRunning = true;
        ui.offlineBakeCurrentFrame = 0;
        writeOfflineBakeManifest(offlineBake);

        std::ostringstream oss;
        oss << "Bake ready: " << UI::ActiveWorkspaceLabel(offlineBake.workspace)
            << " -> " << offlineBake.jobDir.generic_string();
        setOfflineStatus(oss.str());
    };

    auto stepOfflineWorkspace = [&](int workspace, const UI::Settings& cfg, float dt) {
        switch (workspace) {
            case UI::kWorkspaceWater2D:
                waterSim.setDt(dt);
                waterSim.syncSolidsFrom(sim);
                waterSim.step();
                break;
            case UI::kWorkspaceSmoke3D:
                smoke3D.setDt(dt);
                smoke3D.step();
                break;
            case UI::kWorkspaceWater3D:
                water3D.setDt(dt);
                water3D.step();
                break;
            case UI::kWorkspaceCoupled:
                coupled.setDt(dt);
                coupled.stepCoupled(cfg.vortEps);
                break;
            case UI::kWorkspaceSmoke2D:
            default:
                sim.setDt(dt);
                sim.step(cfg.vortEps);
                break;
        }
    };

    auto processOfflineBakeChunk = [&]() {
        if (!offlineBake.running) return;
        constexpr int kMaxFramesPerTick = 4;
        const double chunkStart = glfwGetTime();
        int framesDone = 0;

        while (offlineBake.running && !offlineBake.cancelRequested && offlineBake.nextFrame < offlineBake.totalFrames) {
            OfflineFrameImage frame;
            std::string frameError;
            if (!captureWorkspaceFrame(offlineBake.workspace,
                                       offlineBake.snapshot,
                                       sim,
                                       waterSim,
                                       smoke3D,
                                       water3D,
                                       coupled,
                                       offlineBakeRenderer,
                                       offlineBake.outputWidth,
                                       offlineBake.outputHeight,
                                       frame,
                                       frameError)) {
                finishOfflineBake(std::string("Bake failed: ") + frameError);
                return;
            }

            char frameName[256];
            std::snprintf(frameName, sizeof(frameName), "%s_%06d.tga", offlineBake.framePrefix.c_str(), offlineBake.nextFrame);
            const std::filesystem::path framePath = offlineBake.framesDir / frameName;
            if (!writeTgaImage(framePath, frame.width, frame.height, frame.rgba)) {
                finishOfflineBake(std::string("Bake failed: could not write ") + framePath.generic_string());
                return;
            }

            ++offlineBake.nextFrame;
            ui.offlineBakeCurrentFrame = std::min(ui.offlineBakeFrameCount, offlineBake.nextFrame);

            if (offlineBake.nextFrame < offlineBake.totalFrames) {
                for (int step = 0; step < offlineBake.simStepsPerFrame; ++step) {
                    stepOfflineWorkspace(offlineBake.workspace, offlineBake.snapshot, offlineBake.fixedDt);
                }
            }

            ++framesDone;
            if (framesDone >= kMaxFramesPerTick || (glfwGetTime() - chunkStart) > 0.20) {
                break;
            }
        }

        if (!offlineBake.running) return;

        if (offlineBake.cancelRequested) {
            std::ostringstream oss;
            oss << "Bake cancelled after " << offlineBake.nextFrame << " frame(s). Images kept in "
                << offlineBake.framesDir.generic_string();
            finishOfflineBake(oss.str());
            return;
        }

        if (offlineBake.nextFrame >= offlineBake.totalFrames) {
            std::ostringstream status;
            status << "Bake finished: " << offlineBake.framesDir.generic_string();
            if (offlineBake.encodeVideo) {
                const std::filesystem::path inputPattern = offlineBake.framesDir /
                    (offlineBake.framePrefix + std::string("_%06d.tga"));
                const std::string cmd =
                    std::string("ffmpeg -y -framerate ") + std::to_string(offlineBake.videoFPS) +
                    std::string(" -i ") + shellQuote(inputPattern) +
                    std::string(" -c:v libx264 -pix_fmt yuv420p ") + shellQuote(offlineBake.videoPath);
                setOfflineStatus(std::string("Encoding video: ") + offlineBake.videoPath.generic_string());
                const int rc = std::system(cmd.c_str());
                if (rc == 0) {
                    if (!offlineBake.keepImageSequence) {
                        std::error_code ec;
                        std::filesystem::remove_all(offlineBake.framesDir, ec);
                    }
                    status.str(std::string());
                    status.clear();
                    status << "Bake finished: " << offlineBake.videoPath.generic_string();
                    if (offlineBake.keepImageSequence) {
                        status << " (frames in " << offlineBake.framesDir.generic_string() << ")";
                    }
                } else {
                    status.str(std::string());
                    status.clear();
                    status << "Video encode failed. Frames kept in " << offlineBake.framesDir.generic_string();
                }
            }
            finishOfflineBake(status.str());
            return;
        }

        std::ostringstream progress;
        progress << "Baking frame " << offlineBake.nextFrame << " / " << offlineBake.totalFrames
                 << " -> " << offlineBake.framesDir.generic_string();
        setOfflineStatus(progress.str());
    };

    while (!glfwWindowShouldClose(win))
    {
        glfwPollEvents();

        if (appliedTheme != ui.themeMode || appliedUiScale != ui.uiScale) {
            UI::ApplyViziorTheme(ui.themeMode);
            ImGui::GetStyle().ScaleAllSizes(ui.uiScale);
            io.FontGlobalScale = ui.uiScale;
            setViziorWindowIcon(win, ui.themeMode);
            appliedTheme   = ui.themeMode;
            appliedUiScale = ui.uiScale;
        }

        if (!offlineBake.running) {
            const int requestedWorkspace = std::clamp(ui.activeWorkspace, (int)UI::kWorkspaceSmoke2D, (int)UI::kWorkspaceCoupled);
            if (requestedWorkspace != activeWorkspace) {
                activeWorkspace = requestedWorkspace;
                activateWorkspace(activeWorkspace);
            }

            // step sim (if playing)
        sim.smokeDissipation = ui.smokeDissipation;
        sim.tempDissipation  = ui.tempDissipation;

        waterSim.waterDissipation = ui.waterDissipation;
        waterSim.waterGravity     = ui.waterGravity;
        waterSim.velDamping       = ui.waterVelDamping;
        waterSim.openTop          = ui.waterOpenTop;

        MACWater3D::Params p3 = water3D.params;
        bool p3Changed = false;
        auto assignFloatW = [&](float& dst, float value) {
            if (dst != value) { dst = value; p3Changed = true; }
        };
        auto assignIntW = [&](int& dst, int value) {
            if (dst != value) { dst = value; p3Changed = true; }
        };
        auto assignBoolW = [&](bool& dst, bool value) {
            if (dst != value) { dst = value; p3Changed = true; }
        };

        assignFloatW(p3.waterDissipation, ui.waterDissipation);
        assignFloatW(p3.gravity, ui.waterGravity);
        assignFloatW(p3.velDamping, ui.waterVelDamping);
        assignBoolW(p3.openTop, ui.waterOpenTop);
        assignIntW(p3.pressureIters, ui.water3DPressureIters);
        assignIntW(p3.pressureSolverMode, ui.water3DPressureSolverMode);
        assignBoolW(p3.useAPIC, ui.water3DUseAPIC);
        assignFloatW(p3.flipBlend, ui.water3DFlipBlend);
        assignFloatW(p3.pressureOmega, ui.water3DPressureOmega);
        assignFloatW(p3.pressureMGOmega, ui.water3DPressureMGOmega);
        assignIntW(p3.pressureMGVCycles, ui.water3DPressureMGVCycles);
        assignIntW(p3.pressureMGCoarseIters, ui.water3DPressureMGCoarseIters);
        assignBoolW(p3.volumePreserveRhsMean, ui.water3DVolumePreserve);
        assignFloatW(p3.volumePreserveStrength, ui.water3DVolumePreserveStrength);
        assignIntW(p3.reseedRelaxIters, ui.water3DRelaxIters);
        assignFloatW(p3.reseedRelaxStrength, ui.water3DRelaxStrength);

        MACWater3D::BackendPreference requestedWaterBackend = MACWater3D::BackendPreference::Auto;
        switch (ui.water3DBackendMode) {
            case 1: requestedWaterBackend = MACWater3D::BackendPreference::CPU; break;
            case 2: requestedWaterBackend = MACWater3D::BackendPreference::CUDA; break;
            default: break;
        }
        if (water3D.backendPreferenceMode() != requestedWaterBackend) {
            water3D.setBackendPreference(requestedWaterBackend);
            ++water3DSimVersion;
            water3DRenderCache = {};
        }

        if (p3Changed) {
            water3D.setParams(p3);
            ++water3DSimVersion;
            water3DRenderCache = {};
        }

        MACSmoke3D::Params s3 = smoke3D.params;
        bool s3Changed = false;
        auto assignFloatS = [&](float& dst, float value) {
            if (dst != value) { dst = value; s3Changed = true; }
        };
        auto assignIntS = [&](int& dst, int value) {
            if (dst != value) { dst = value; s3Changed = true; }
        };
        auto assignBoolS = [&](bool& dst, bool value) {
            if (dst != value) { dst = value; s3Changed = true; }
        };

        assignFloatS(s3.smokeDissipation, ui.smokeDissipation);
        assignFloatS(s3.tempDissipation, ui.tempDissipation);
        assignFloatS(s3.gravity, ui.smoke3DGravity);
        assignFloatS(s3.buoyancyScale, ui.smoke3DBuoyancyScale);
        assignFloatS(s3.velDamping, ui.smoke3DVelDamping);
        assignFloatS(s3.viscosity, ui.smoke3DViscosity);
        assignFloatS(s3.smokeDiffusivity, ui.smoke3DSmokeDiffusivity);
        assignFloatS(s3.tempDiffusivity, ui.smoke3DTempDiffusivity);
        assignBoolS(s3.openTop, ui.smoke3DOpenTop);
        assignIntS(s3.pressureIters, ui.smoke3DPressureIters);
        assignIntS(s3.pressureSolverMode, ui.smoke3DPressureSolverMode);
        assignFloatS(s3.pressureOmega, ui.smoke3DPressureOmega);
        assignFloatS(s3.pressureMGOmega, ui.smoke3DPressureMGOmega);
        assignIntS(s3.pressureMGVCycles, ui.smoke3DPressureMGVCycles);
        assignIntS(s3.pressureMGCoarseIters, ui.smoke3DPressureMGCoarseIters);

        if (s3Changed) {
            smoke3D.setParams(s3);
            ++smoke3DSimVersion;
            smoke3DRenderCache = {};
        }

        coupled.waterDissipation = ui.waterDissipation;
        coupled.waterGravity     = ui.waterGravity;
        coupled.velDamping       = ui.waterVelDamping;
        coupled.setOpenTop(false);  // Combined view: closed top for now

        sim.pressureMGVCycles = ui.smoke2DPressureMGVCycles;
        sim.pressureMGCoarseIters = ui.smoke2DPressureMGCoarseIters;
        coupled.pressureMGVCycles = ui.smoke2DPressureMGVCycles;
        coupled.pressureMGCoarseIters = ui.smoke2DPressureMGCoarseIters;

        coupled.smokeDissipation = ui.smokeDissipation;
        coupled.tempDissipation  = ui.tempDissipation;
        coupled.useMacCormack    = sim.useMacCormack; // or ui toggle if you expose one

        double now = glfwGetTime();
        double frameDt = now - lastTime;
        lastTime = now;

        // avoid huge jumps (dragging window, breakpoint, etc.)
        if (frameDt > 0.1) frameDt = 0.1;

        if (ui.playing) {
            accumulator += frameDt * simSpeed;

            int steps = 0;
            while (accumulator > 0.0 && steps < maxStepsPerFrame) {
                float maxSpeed = sim.maxFaceSpeed();
                float cflDx = sim.dx;

                switch (activeWorkspace) {
                    case UI::kWorkspaceWater2D:
                        maxSpeed = std::max(waterSim.maxFaceSpeed(), waterSim.maxParticleSpeed());
                        cflDx = waterSim.dx;
                        break;
                    case UI::kWorkspaceSmoke3D:
                        maxSpeed = smoke3D.stats().maxSpeed;
                        cflDx = smoke3D.dx;
                        break;
                    case UI::kWorkspaceWater3D:
                        maxSpeed = water3D.stats().maxSpeed;
                        cflDx = water3D.dx;
                        break;
                    case UI::kWorkspaceCoupled:
                        maxSpeed = std::max(coupled.maxFaceSpeed(), coupled.maxParticleSpeed());
                        cflDx = coupled.dx;
                        break;
                    case UI::kWorkspaceSmoke2D:
                    default:
                        maxSpeed = sim.maxFaceSpeed();
                        cflDx = sim.dx;
                        break;
                }

                float dtCFL = ui.cfl * cflDx / (maxSpeed + 1e-6f);

                // CFL is a hard cap
                float dt = std::min(ui.dtMax, dtCFL);
                if (ui.dtMin <= dtCFL) {
                    dt = std::max(dt, ui.dtMin);
                }

                switch (activeWorkspace) {
                    case UI::kWorkspaceWater2D:
                        waterSim.setDt(dt);
                        waterSim.syncSolidsFrom(sim);
                        waterSim.step();
                        break;
                    case UI::kWorkspaceSmoke3D:
                        smoke3D.setDt(dt);
                        smoke3D.step();
                        ++smoke3DSimVersion;
                        break;
                    case UI::kWorkspaceWater3D:
                        water3D.setDt(dt);
                        water3D.step();
                        ++water3DSimVersion;
                        break;
                    case UI::kWorkspaceCoupled:
                        coupled.setDt(dt);
                        coupled.stepCoupled(ui.vortEps);
                        break;
                    case UI::kWorkspaceSmoke2D:
                    default:
                        sim.setDt(dt);
                        sim.step(ui.vortEps);
                        break;
                }

                accumulator -= dt;
                steps++;
            }

            // optional: if we hit the cap, drop the remainder so it doesn't lag forever
            if (steps == maxStepsPerFrame) accumulator = 0.0;
        }

        // upload textures (do this each frame before NewFrame so ImGui uses latest)
        SmokeRenderSettings rs;
        OverlaySettings ov;
        UI::BuildRenderSettings(ui, rs, ov);
        WaterRenderSettings wr;
        UI::BuildWaterRenderSettings(ui, wr);

        if (activeWorkspace == UI::kWorkspaceSmoke2D) {
            renderer.updateFromSim(sim, rs, ov);
        } else if (activeWorkspace == UI::kWorkspaceWater2D) {
            renderer.updateWaterFromSim(waterSim, wr);
        } else if (activeWorkspace == UI::kWorkspaceCoupled) {
            coupledRenderer.updateFromSim(coupled, rs, ov);
            coupledRenderer.updateWaterFromSim(coupled, wr);
        }

        if (activeWorkspace == UI::kWorkspaceWater3D) {
            const int viewMode = std::clamp(ui.water3DViewMode, 0, 2);
            if (viewMode == 1) {
                const int axis = std::clamp(ui.water3DSliceAxis, 0, 2);
                const int field = std::clamp(ui.water3DDebugField, 0, 3);
                const int maxSlice = (axis == 0)
                    ? std::max(0, water3D.nz - 1)
                    : (axis == 1)
                        ? std::max(0, water3D.ny - 1)
                        : std::max(0, water3D.nx - 1);
                ui.water3DSliceIndex = std::clamp(ui.water3DSliceIndex, 0, maxSlice);

                auto slice = water3D.copyDebugSlice(
                    static_cast<MACWater3D::SliceAxis>(axis),
                    ui.water3DSliceIndex,
                    static_cast<MACWater3D::DebugField>(field));
                water3DRenderer.updateWaterFromSlice(slice.values, slice.solid, slice.width, slice.height, wr);
                water3DRenderCache = {};
            } else {
                const auto target = computeVolumeRenderTargetSize(
                    win,
                    ui.water3DViewportWidth,
                    ui.water3DViewportHeight,
                    ui.volumeRenderScale);
                const bool sizeChanged =
                    water3DRenderer.width() != target.width || water3DRenderer.height() != target.height;
                if (sizeChanged) {
                    water3DRenderer.resize(target.width, target.height);
                }

                const bool viewChanged =
                    !water3DRenderCache.valid ||
                    water3DRenderCache.lastWidth != target.width ||
                    water3DRenderCache.lastHeight != target.height ||
                    water3DRenderCache.lastViewMode != viewMode ||
                    !nearlyEqual(water3DRenderCache.lastYaw, ui.water3DViewYawDeg) ||
                    !nearlyEqual(water3DRenderCache.lastPitch, ui.water3DViewPitchDeg) ||
                    !nearlyEqual(water3DRenderCache.lastZoom, ui.water3DViewZoom) ||
                    !nearlyEqual(water3DRenderCache.lastDensity, ui.water3DVolumeDensity) ||
                    !nearlyEqual(water3DRenderCache.lastSurfaceThreshold, ui.water3DSurfaceThreshold);
                const bool simChanged =
                    !water3DRenderCache.valid || water3DRenderCache.lastSimVersion != water3DSimVersion;

                bool needUpdate = false;
                if (!water3DRenderCache.valid || sizeChanged || viewChanged) {
                    needUpdate = true;
                } else if (!ui.playing) {
                    needUpdate = true;
                } else if (!ui.water3DThrottleRendering) {
                    needUpdate = simChanged;
                } else if (simChanged) {
                    const double interval = 1.0 / std::max(1.0f, ui.water3DRenderFPS);
                    needUpdate = (now - water3DRenderCache.lastRenderTime) >= interval;
                }

                if (needUpdate) {
                    water3DRenderer.updateWaterFromVolume(
                        water3D.water,
                        water3D.solid,
                        water3D.nx,
                        water3D.ny,
                        water3D.nz,
                        viewMode,
                        ui.water3DViewYawDeg,
                        ui.water3DViewPitchDeg,
                        ui.water3DViewZoom,
                        ui.water3DVolumeDensity,
                        ui.water3DSurfaceThreshold,
                        wr);
                    water3DRenderCache.valid = true;
                    water3DRenderCache.lastRenderTime = now;
                    water3DRenderCache.lastSimVersion = water3DSimVersion;
                    water3DRenderCache.lastWidth = target.width;
                    water3DRenderCache.lastHeight = target.height;
                    water3DRenderCache.lastViewMode = viewMode;
                    water3DRenderCache.lastYaw = ui.water3DViewYawDeg;
                    water3DRenderCache.lastPitch = ui.water3DViewPitchDeg;
                    water3DRenderCache.lastZoom = ui.water3DViewZoom;
                    water3DRenderCache.lastDensity = ui.water3DVolumeDensity;
                    water3DRenderCache.lastSurfaceThreshold = ui.water3DSurfaceThreshold;
                }
            }
        }

        if (activeWorkspace == UI::kWorkspaceSmoke3D) {
            const int viewMode = std::clamp(ui.smoke3DViewMode, 0, 1);
            if (viewMode == 1) {
                const int axis = std::clamp(ui.smoke3DSliceAxis, 0, 2);
                const int field = std::clamp(ui.smoke3DDebugField, 0, 4);
                const int maxSlice = (axis == 0)
                    ? std::max(0, smoke3D.nz - 1)
                    : (axis == 1)
                        ? std::max(0, smoke3D.ny - 1)
                        : std::max(0, smoke3D.nx - 1);
                ui.smoke3DSliceIndex = std::clamp(ui.smoke3DSliceIndex, 0, maxSlice);

                auto slice = smoke3D.copyDebugSlice(
                    static_cast<MACSmoke3D::SliceAxis>(axis),
                    ui.smoke3DSliceIndex,
                    static_cast<MACSmoke3D::DebugField>(field));
                smoke3DRenderer.updateSmokeFromSlice(slice.values, slice.solid, slice.width, slice.height, field, rs);
                smoke3DRenderCache = {};
            } else {
                const auto target = computeVolumeRenderTargetSize(
                    win,
                    ui.smoke3DViewportWidth,
                    ui.smoke3DViewportHeight,
                    ui.volumeRenderScale);
                const bool sizeChanged =
                    smoke3DRenderer.width() != target.width || smoke3DRenderer.height() != target.height;
                if (sizeChanged) {
                    smoke3DRenderer.resize(target.width, target.height);
                }

                const bool viewChanged =
                    !smoke3DRenderCache.valid ||
                    smoke3DRenderCache.lastWidth != target.width ||
                    smoke3DRenderCache.lastHeight != target.height ||
                    smoke3DRenderCache.lastViewMode != viewMode ||
                    !nearlyEqual(smoke3DRenderCache.lastYaw, ui.smoke3DViewYawDeg) ||
                    !nearlyEqual(smoke3DRenderCache.lastPitch, ui.smoke3DViewPitchDeg) ||
                    !nearlyEqual(smoke3DRenderCache.lastZoom, ui.smoke3DViewZoom) ||
                    !nearlyEqual(smoke3DRenderCache.lastDensity, ui.smoke3DVolumeDensity);
                const bool simChanged =
                    !smoke3DRenderCache.valid || smoke3DRenderCache.lastSimVersion != smoke3DSimVersion;

                bool needUpdate = false;
                if (!smoke3DRenderCache.valid || sizeChanged || viewChanged) {
                    needUpdate = true;
                } else if (!ui.playing) {
                    needUpdate = true;
                } else if (!ui.smoke3DThrottleRendering) {
                    needUpdate = simChanged;
                } else if (simChanged) {
                    const double interval = 1.0 / std::max(1.0f, ui.smoke3DRenderFPS);
                    needUpdate = (now - smoke3DRenderCache.lastRenderTime) >= interval;
                }

                if (needUpdate) {
                    smoke3DRenderer.updateSmokeFromVolume(
                        smoke3D.smoke,
                        smoke3D.temp,
                        smoke3D.solid,
                        smoke3D.nx,
                        smoke3D.ny,
                        smoke3D.nz,
                        ui.smoke3DViewYawDeg,
                        ui.smoke3DViewPitchDeg,
                        ui.smoke3DViewZoom,
                        ui.smoke3DVolumeDensity,
                        rs);
                    smoke3DRenderCache.valid = true;
                    smoke3DRenderCache.lastRenderTime = now;
                    smoke3DRenderCache.lastSimVersion = smoke3DSimVersion;
                    smoke3DRenderCache.lastWidth = target.width;
                    smoke3DRenderCache.lastHeight = target.height;
                    smoke3DRenderCache.lastViewMode = viewMode;
                    smoke3DRenderCache.lastYaw = ui.smoke3DViewYawDeg;
                    smoke3DRenderCache.lastPitch = ui.smoke3DViewPitchDeg;
                    smoke3DRenderCache.lastZoom = ui.smoke3DViewZoom;
                    smoke3DRenderCache.lastDensity = ui.smoke3DVolumeDensity;
                }
            }
        }

        } else {
            processOfflineBakeChunk();
        }

        // start imgui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // draw UI
        int currentWindowW = 0;
        int currentWindowH = 0;
        glfwGetWindowSize(win, &currentWindowW, &currentWindowH);
        UI::Actions actions = UI::DrawAll(sim, waterSim, water3D, smoke3D, coupled,
                                          renderer, water3DRenderer, smoke3DRenderer, coupledRenderer,
                                          ui, probe, NX, NY, currentWindowW, currentWindowH);
        if (UI::ConsumeResetLayoutRequest()) {
        // nothing needed here if panels.cpp already deleted ini and rebuilt docks
        // but it's fine to keep for future, you better not delete this >:)
    }

        // handle actions
        if (actions.startOfflineBakeRequested && !offlineBake.running) {
            startOfflineBake();
        }
        if (actions.cancelOfflineBakeRequested && offlineBake.running) {
            offlineBake.cancelRequested = true;
            setOfflineStatus("Cancelling bake after the current frame batch...");
        }
        if (!offlineBake.running) {
            if (actions.dropWaterTextRequested) {
                waterSim.waterHeld = false;
            }
            if (actions.resetRequested) {
                activateWorkspace(activeWorkspace);
            }
            if ((actions.applySmoke3DGridRequested || actions.resetSmoke3DRequested) && activeWorkspace == UI::kWorkspaceSmoke3D) {
                activateWorkspace(activeWorkspace);
            }
            if ((actions.applyWater3DGridRequested || actions.resetWater3DRequested) && activeWorkspace == UI::kWorkspaceWater3D) {
                activateWorkspace(activeWorkspace);
            }
            if (actions.applyWindowResolutionRequested) {
                ui.windowWidth = std::clamp(ui.windowWidth, 960, 7680);
                ui.windowHeight = std::clamp(ui.windowHeight, 540, 4320);
                glfwSetWindowSize(win, ui.windowWidth, ui.windowHeight);
            }
            if (actions.applyGrid2DRequested) {
                reinit2DGrid();
            }
        }

        // render GL
        int w, h;
        glfwGetFramebufferSize(win, &w, &h);
        glViewport(0, 0, w, h);
        if (ui.themeMode == kThemeLight) {
            glClearColor(0.948f, 0.944f, 0.937f, 1.0f);
        } else {
            glClearColor(0.055f, 0.066f, 0.078f, 1.0f);
        }
        glClear(GL_COLOR_BUFFER_BIT);

        // render imgui
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        if (UI::ConsumeSaveLayoutRequest())
            ImGui::SaveIniSettingsToDisk(ImGui::GetIO().IniFilename);

       

        glfwSwapBuffers(win);
    }

    // shutdown
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
