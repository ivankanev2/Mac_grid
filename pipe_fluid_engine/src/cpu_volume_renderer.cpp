// ============================================================================
// cpu_volume_renderer.cpp — multi-threaded CPU ray-march of a voxel volume.
//
// The image is produced at (fbW/renderScale, fbH/renderScale) and uploaded
// into a GL_RGBA8 texture. That texture is then drawn as a fullscreen quad
// with pre-multiplied alpha blending so it composites over the already-
// rendered pipe mesh in the right pixels.
//
// Ray setup: for each output pixel we invert (proj * view) to recover a
// world-space ray. We then intersect that ray with the voxel AABB
// [origin, origin + (nx*dx, ny*dx, nz*dx)] and march from tmin to tmax,
// sampling the density and temperature fields trilinearly.
// ============================================================================

#include "pipe_fluid/volume_renderer.h"

#ifdef __APPLE__
#  define GL_SILENCE_DEPRECATION
#  include <OpenGL/gl3.h>
#else
#  include <GL/gl.h>
#endif

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace pipe_fluid {
namespace {

// ---- Minimal vector/matrix helpers ----------------------------------------
struct V3 { float x, y, z; };

static V3 v3(float x, float y, float z) { return {x, y, z}; }
static V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
static V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
static V3 operator*(V3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
static float dot(V3 a, V3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
static float length(V3 a) { return std::sqrt(dot(a, a)); }
static V3 normalize(V3 a) {
    float L = length(a);
    return (L > 1e-20f) ? a * (1.0f / L) : v3(0, 0, 1);
}

// Column-major 4x4 matrix multiplication.
static void mat4Mul(const float* A, const float* B, float* out) {
    float r[16];
    for (int c = 0; c < 4; ++c) {
        for (int rw = 0; rw < 4; ++rw) {
            r[c*4 + rw] = A[0*4 + rw] * B[c*4 + 0]
                        + A[1*4 + rw] * B[c*4 + 1]
                        + A[2*4 + rw] * B[c*4 + 2]
                        + A[3*4 + rw] * B[c*4 + 3];
        }
    }
    std::memcpy(out, r, sizeof(r));
}

// Column-major 4x4 inverse. Returns true on success.
static bool mat4Inverse(const float* m, float* inv) {
    float a[16];
    a[0]  =  m[5]*m[10]*m[15] - m[5]*m[11]*m[14] - m[9]*m[6]*m[15]
           + m[9]*m[7]*m[14] + m[13]*m[6]*m[11] - m[13]*m[7]*m[10];
    a[4]  = -m[4]*m[10]*m[15] + m[4]*m[11]*m[14] + m[8]*m[6]*m[15]
           - m[8]*m[7]*m[14] - m[12]*m[6]*m[11] + m[12]*m[7]*m[10];
    a[8]  =  m[4]*m[9]*m[15] - m[4]*m[11]*m[13] - m[8]*m[5]*m[15]
           + m[8]*m[7]*m[13] + m[12]*m[5]*m[11] - m[12]*m[7]*m[9];
    a[12] = -m[4]*m[9]*m[14] + m[4]*m[10]*m[13] + m[8]*m[5]*m[14]
           - m[8]*m[6]*m[13] - m[12]*m[5]*m[10] + m[12]*m[6]*m[9];
    a[1]  = -m[1]*m[10]*m[15] + m[1]*m[11]*m[14] + m[9]*m[2]*m[15]
           - m[9]*m[3]*m[14] - m[13]*m[2]*m[11] + m[13]*m[3]*m[10];
    a[5]  =  m[0]*m[10]*m[15] - m[0]*m[11]*m[14] - m[8]*m[2]*m[15]
           + m[8]*m[3]*m[14] + m[12]*m[2]*m[11] - m[12]*m[3]*m[10];
    a[9]  = -m[0]*m[9]*m[15] + m[0]*m[11]*m[13] + m[8]*m[1]*m[15]
           - m[8]*m[3]*m[13] - m[12]*m[1]*m[11] + m[12]*m[3]*m[9];
    a[13] =  m[0]*m[9]*m[14] - m[0]*m[10]*m[13] - m[8]*m[1]*m[14]
           + m[8]*m[2]*m[13] + m[12]*m[1]*m[10] - m[12]*m[2]*m[9];
    a[2]  =  m[1]*m[6]*m[15] - m[1]*m[7]*m[14] - m[5]*m[2]*m[15]
           + m[5]*m[3]*m[14] + m[13]*m[2]*m[7] - m[13]*m[3]*m[6];
    a[6]  = -m[0]*m[6]*m[15] + m[0]*m[7]*m[14] + m[4]*m[2]*m[15]
           - m[4]*m[3]*m[14] - m[12]*m[2]*m[7] + m[12]*m[3]*m[6];
    a[10] =  m[0]*m[5]*m[15] - m[0]*m[7]*m[13] - m[4]*m[1]*m[15]
           + m[4]*m[3]*m[13] + m[12]*m[1]*m[7] - m[12]*m[3]*m[5];
    a[14] = -m[0]*m[5]*m[14] + m[0]*m[6]*m[13] + m[4]*m[1]*m[14]
           - m[4]*m[2]*m[13] - m[12]*m[1]*m[6] + m[12]*m[2]*m[5];
    a[3]  = -m[1]*m[6]*m[11] + m[1]*m[7]*m[10] + m[5]*m[2]*m[11]
           - m[5]*m[3]*m[10] - m[9]*m[2]*m[7]  + m[9]*m[3]*m[6];
    a[7]  =  m[0]*m[6]*m[11] - m[0]*m[7]*m[10] - m[4]*m[2]*m[11]
           + m[4]*m[3]*m[10] + m[8]*m[2]*m[7]  - m[8]*m[3]*m[6];
    a[11] = -m[0]*m[5]*m[11] + m[0]*m[7]*m[9]  + m[4]*m[1]*m[11]
           - m[4]*m[3]*m[9]  - m[8]*m[1]*m[7]  + m[8]*m[3]*m[5];
    a[15] =  m[0]*m[5]*m[10] - m[0]*m[6]*m[9]  - m[4]*m[1]*m[10]
           + m[4]*m[2]*m[9]  + m[8]*m[1]*m[6]  - m[8]*m[2]*m[5];

    float det = m[0]*a[0] + m[1]*a[4] + m[2]*a[8] + m[3]*a[12];
    if (std::fabs(det) < 1e-20f) return false;
    const float invDet = 1.0f / det;
    for (int i = 0; i < 16; ++i) inv[i] = a[i] * invDet;
    return true;
}

// Transform a (x,y,z,w) point by a col-major 4x4 matrix, returning xyz/w.
static V3 mat4MulPoint(const float* m, float x, float y, float z, float w) {
    const float rx = m[0]*x + m[4]*y + m[8]*z  + m[12]*w;
    const float ry = m[1]*x + m[5]*y + m[9]*z  + m[13]*w;
    const float rz = m[2]*x + m[6]*y + m[10]*z + m[14]*w;
    const float rw = m[3]*x + m[7]*y + m[11]*z + m[15]*w;
    const float inv = (std::fabs(rw) > 1e-20f) ? 1.0f / rw : 0.0f;
    return v3(rx * inv, ry * inv, rz * inv);
}

// Slab method for ray vs AABB intersection. Returns true if the ray hits
// and writes tmin, tmax (may be negative if origin is inside).
static bool intersectAABB(V3 o, V3 d,
                          V3 bmin, V3 bmax,
                          float& tmin, float& tmax) {
    float tLo = -1e30f, tHi = 1e30f;
    for (int i = 0; i < 3; ++i) {
        const float oi = (&o.x)[i];
        const float di = (&d.x)[i];
        const float lo = (&bmin.x)[i];
        const float hi = (&bmax.x)[i];
        if (std::fabs(di) < 1e-12f) {
            if (oi < lo || oi > hi) return false;
        } else {
            float t0 = (lo - oi) / di;
            float t1 = (hi - oi) / di;
            if (t0 > t1) std::swap(t0, t1);
            tLo = std::max(tLo, t0);
            tHi = std::min(tHi, t1);
            if (tHi < tLo) return false;
        }
    }
    tmin = tLo;
    tmax = tHi;
    return tmax > 0.0f;
}

static float clamp01(float x) {
    return x < 0.f ? 0.f : (x > 1.f ? 1.f : x);
}

// Trilinear sample of a cell-centered field at normalized [0,1]^3 coords.
static float sampleTrilinear(const std::vector<float>& vol,
                             int nx, int ny, int nz,
                             float ux, float uy, float uz) {
    ux = clamp01(ux); uy = clamp01(uy); uz = clamp01(uz);
    const float fx = ux * (float)(nx - 1);
    const float fy = uy * (float)(ny - 1);
    const float fz = uz * (float)(nz - 1);
    const int i0 = std::clamp((int)std::floor(fx), 0, nx - 1);
    const int j0 = std::clamp((int)std::floor(fy), 0, ny - 1);
    const int k0 = std::clamp((int)std::floor(fz), 0, nz - 1);
    const int i1 = std::min(i0 + 1, nx - 1);
    const int j1 = std::min(j0 + 1, ny - 1);
    const int k1 = std::min(k0 + 1, nz - 1);
    const float tx = fx - (float)i0;
    const float ty = fy - (float)j0;
    const float tz = fz - (float)k0;
    auto at = [&](int i, int j, int k) -> float {
        return vol[(size_t)i + (size_t)nx * ((size_t)j + (size_t)ny * (size_t)k)];
    };
    const float c00 = at(i0,j0,k0)*(1-tx) + at(i1,j0,k0)*tx;
    const float c10 = at(i0,j1,k0)*(1-tx) + at(i1,j1,k0)*tx;
    const float c01 = at(i0,j0,k1)*(1-tx) + at(i1,j0,k1)*tx;
    const float c11 = at(i0,j1,k1)*(1-tx) + at(i1,j1,k1)*tx;
    const float c0  = c00*(1-ty) + c10*ty;
    const float c1  = c01*(1-ty) + c11*ty;
    return c0*(1-tz) + c1*tz;
}

static uint8_t sampleSolidNearest(const std::vector<uint8_t>& solid,
                                  int nx, int ny, int nz,
                                  float ux, float uy, float uz) {
    if (solid.empty()) return 0;
    const int i = std::clamp((int)std::round(clamp01(ux) * (float)(nx - 1)), 0, nx - 1);
    const int j = std::clamp((int)std::round(clamp01(uy) * (float)(ny - 1)), 0, ny - 1);
    const int k = std::clamp((int)std::round(clamp01(uz) * (float)(nz - 1)), 0, nz - 1);
    return solid[(size_t)i + (size_t)nx * ((size_t)j + (size_t)ny * (size_t)k)];
}

// ---- Fullscreen-quad helpers ----------------------------------------------
static const char* QUAD_VERT = R"GLSL(
#version 150 core
in vec2 aPos;
out vec2 vUV;
void main() {
    vUV = aPos * 0.5 + 0.5;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
)GLSL";

static const char* QUAD_FRAG = R"GLSL(
#version 150 core
in  vec2 vUV;
uniform sampler2D uTex;
out vec4 fragColor;
void main() {
    vec4 c = texture(uTex, vUV);
    // Texture already holds pre-multiplied RGB*A; use straight-alpha blending.
    fragColor = c;
}
)GLSL";

static GLuint compileShader(GLenum type, const char* src) {
    GLuint sh = glCreateShader(type);
    glShaderSource(sh, 1, &src, nullptr);
    glCompileShader(sh);
    GLint ok = GL_FALSE;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) { glDeleteShader(sh); return 0; }
    return sh;
}

static GLuint compileProgram(const char* vs, const char* fs) {
    GLuint v = compileShader(GL_VERTEX_SHADER, vs);
    GLuint f = compileShader(GL_FRAGMENT_SHADER, fs);
    if (!v || !f) { if (v) glDeleteShader(v); if (f) glDeleteShader(f); return 0; }
    GLuint p = glCreateProgram();
    glAttachShader(p, v);
    glAttachShader(p, f);
    glBindAttribLocation(p, 0, "aPos");
    glLinkProgram(p);
    GLint ok = GL_FALSE;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    glDeleteShader(v); glDeleteShader(f);
    if (!ok) { glDeleteProgram(p); return 0; }
    return p;
}

// ============================================================================
// CPU backend implementation
// ============================================================================
class CpuVolumeRenderer : public VolumeOverlayRenderer {
public:
    bool init() override {
        m_prog = compileProgram(QUAD_VERT, QUAD_FRAG);
        if (!m_prog) return false;

        const float quad[8] = { -1,-1,  1,-1, -1, 1,  1, 1 };
        glGenVertexArrays(1, &m_vao);
        glGenBuffers(1, &m_vbo);
        glBindVertexArray(m_vao);
        glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), nullptr);
        glBindVertexArray(0);

        glGenTextures(1, &m_tex);
        glBindTexture(GL_TEXTURE_2D, m_tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glBindTexture(GL_TEXTURE_2D, 0);

        return true;
    }

    void setVolume(const std::vector<float>& density,
                   const std::vector<float>& temp,
                   const std::vector<uint8_t>& solid,
                   int nx, int ny, int nz) override {
        m_density = &density;
        m_temp    = &temp;
        m_solid   = &solid;
        m_nx = nx; m_ny = ny; m_nz = nz;
    }

    void render(const VolumeView& V, const VolumeSettings& S) override {
        if (!m_density || m_nx <= 1 || m_ny <= 1 || m_nz <= 1) return;
        if (V.fbWidth <= 0 || V.fbHeight <= 0) return;

        const int scale = std::max(1, S.renderScale);
        const int W = std::max(16, V.fbWidth  / scale);
        const int H = std::max(16, V.fbHeight / scale);

        if ((int)m_rgba.size() != W * H * 4) m_rgba.resize((size_t)W * H * 4);
        std::fill(m_rgba.begin(), m_rgba.end(), (uint8_t)0);

        // Build inverse(proj * view) once.
        float PV[16], invPV[16];
        mat4Mul(V.proj, V.view, PV);
        if (!mat4Inverse(PV, invPV)) return;

        const V3 camPos = v3(V.camPosX, V.camPosY, V.camPosZ);
        const V3 bmin   = v3(V.originX,
                             V.originY,
                             V.originZ);
        const V3 bmax   = v3(V.originX + V.nx * V.dx,
                             V.originY + V.ny * V.dx,
                             V.originZ + V.nz * V.dx);
        const V3 bSize  = bmax - bmin;
        if (bSize.x <= 0 || bSize.y <= 0 || bSize.z <= 0) return;

        // March step = ~half a voxel. Capped by VolumeSettings::stepsPerPixel
        // to bound the worst case when rays skim the AABB diagonally.
        const float step = 0.5f * V.dx;
        const int   maxSteps = std::max(16, S.stepsPerPixel);
        const float sigmaScale = std::max(0.05f, S.densityScale);
        const bool  useColor = S.useColor;
        const float alphaScale = clamp01(S.alphaScale);
        const float tempStrength = S.tempStrength;
        const float coreDark = S.coreDark;

        const std::vector<float>&   density = *m_density;
        const std::vector<float>&   temp    = *m_temp;
        const std::vector<uint8_t>& solid   = *m_solid;
        const int nx = m_nx, ny = m_ny, nz = m_nz;

        // Row-parallel dispatch.
        const unsigned threads = std::max(1u, std::thread::hardware_concurrency());
        std::vector<std::thread> pool;
        pool.reserve(threads);
        std::atomic<int> nextRow{0};
        const int rowsPerChunk = std::max(1, H / (int)(threads * 4));

        auto worker = [&]() {
            while (true) {
                const int rowBegin = nextRow.fetch_add(rowsPerChunk);
                if (rowBegin >= H) break;
                const int rowEnd = std::min(H, rowBegin + rowsPerChunk);

                for (int j = rowBegin; j < rowEnd; ++j) {
                    // GL texture convention: data[0] is the BOTTOM-LEFT pixel,
                    // so image-row j=0 must correspond to NDC y = -1 (bottom).
                    // Previously this mapped j=0 to NDC y = +1, which caused the
                    // overlay to be uploaded upside-down and made the smoke
                    // appear mirrored vertically about the pipe as the camera
                    // pitched/yawed (i.e. "rotates with camera").
                    const float py = 2.0f * ((float)j + 0.5f) / (float)H - 1.0f;
                    for (int i = 0; i < W; ++i) {
                        const float px = 2.0f * ((float)i + 0.5f) / (float)W - 1.0f;

                        // Reconstruct world-space ray by unprojecting NDC.
                        const V3 pNear = mat4MulPoint(invPV, px, py, -1.0f, 1.0f);
                        const V3 pFar  = mat4MulPoint(invPV, px, py,  1.0f, 1.0f);
                        V3 rayDir = normalize(pFar - pNear);
                        V3 rayOri = camPos;

                        float tEnter = 0.f, tExit = 0.f;
                        if (!intersectAABB(rayOri, rayDir, bmin, bmax, tEnter, tExit)) {
                            // Fully transparent
                            continue;
                        }

                        float t = std::max(0.0f, tEnter);
                        float accumR = 0.f, accumG = 0.f, accumB = 0.f, accumA = 0.f;
                        int   steps = 0;
                        while (t < tExit && accumA < 0.995f && steps < maxSteps) {
                            V3 p = rayOri + rayDir * t;
                            const float ux = (p.x - bmin.x) / bSize.x;
                            const float uy = (p.y - bmin.y) / bSize.y;
                            const float uz = (p.z - bmin.z) / bSize.z;

                            const uint8_t isSolid = sampleSolidNearest(solid, nx, ny, nz, ux, uy, uz);
                            if (!isSolid) {
                                const float d = std::max(0.f, sampleTrilinear(density, nx, ny, nz, ux, uy, uz));
                                if (d > 1e-4f) {
                                    const float T = temp.empty()
                                        ? 0.f
                                        : std::max(0.f, sampleTrilinear(temp, nx, ny, nz, ux, uy, uz));

                                    const float sigma = d * sigmaScale * 3.0f;
                                    const float aStep = (1.0f - std::exp(-sigma * step * 3.0f)) * alphaScale;

                                    float cr, cg, cb;
                                    if (!useColor) {
                                        const float gray = 0.22f + 0.68f * std::pow(clamp01(d), 0.60f);
                                        cr = cg = cb = gray;
                                    } else {
                                        const float tCol = clamp01(std::pow(T, 0.55f));
                                        const float gray = 0.10f + 0.90f * std::pow(clamp01(d), 0.55f);
                                        cr = gray + tempStrength * tCol * 0.55f;
                                        cg = gray + tempStrength * tCol * 0.12f;
                                        cb = gray * (1.0f - 0.35f * tCol);
                                        const float ageDark = coreDark * clamp01(d * d);
                                        const float core = 1.0f - ageDark;
                                        cr *= 0.35f + 0.65f * core;
                                        cg *= 0.35f + 0.65f * core;
                                        cb *= 0.35f + 0.65f * core;
                                    }

                                    const float oneMinusA = 1.0f - accumA;
                                    // Pre-multiplied alpha accumulation.
                                    accumR += oneMinusA * aStep * cr;
                                    accumG += oneMinusA * aStep * cg;
                                    accumB += oneMinusA * aStep * cb;
                                    accumA += oneMinusA * aStep;
                                }
                            }
                            t += step;
                            ++steps;
                        }

                        if (accumA > 0.f) {
                            const int dst = (j * W + i) * 4;
                            m_rgba[(size_t)dst + 0] = (uint8_t)std::lround(clamp01(accumR) * 255.0f);
                            m_rgba[(size_t)dst + 1] = (uint8_t)std::lround(clamp01(accumG) * 255.0f);
                            m_rgba[(size_t)dst + 2] = (uint8_t)std::lround(clamp01(accumB) * 255.0f);
                            m_rgba[(size_t)dst + 3] = (uint8_t)std::lround(clamp01(accumA) * 255.0f);
                        }
                    }
                }
            }
        };

        for (unsigned t = 0; t < threads; ++t) pool.emplace_back(worker);
        for (auto& th : pool) th.join();

        // Upload and composite.
        glBindTexture(GL_TEXTURE_2D, m_tex);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        if (W != m_texW || H != m_texH) {
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, m_rgba.data());
            m_texW = W; m_texH = H;
        } else {
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, m_rgba.data());
        }

        drawCompositeQuad(V.fbWidth, V.fbHeight);
    }

    void shutdown() override {
        if (m_tex) { glDeleteTextures(1, &m_tex); m_tex = 0; }
        if (m_vbo) { glDeleteBuffers(1, &m_vbo); m_vbo = 0; }
        if (m_vao) { glDeleteVertexArrays(1, &m_vao); m_vao = 0; }
        if (m_prog) { glDeleteProgram(m_prog); m_prog = 0; }
    }

    Backend     backend()     const override { return Backend::CPU; }
    const char* backendName() const override { return "CPU"; }

private:
    void drawCompositeQuad(int vpW, int vpH) {
        glViewport(0, 0, vpW, vpH);

        GLboolean depthWas = glIsEnabled(GL_DEPTH_TEST);
        GLboolean blendWas = glIsEnabled(GL_BLEND);
        GLint srcRGB=0, dstRGB=0, srcA=0, dstA=0;
        glGetIntegerv(GL_BLEND_SRC_RGB, &srcRGB);
        glGetIntegerv(GL_BLEND_DST_RGB, &dstRGB);
        glGetIntegerv(GL_BLEND_SRC_ALPHA, &srcA);
        glGetIntegerv(GL_BLEND_DST_ALPHA, &dstA);

        glDisable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        // Texture holds pre-multiplied color; straight alpha over.
        glBlendFuncSeparate(GL_ONE, GL_ONE_MINUS_SRC_ALPHA,
                            GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

        glUseProgram(m_prog);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, m_tex);
        GLint loc = glGetUniformLocation(m_prog, "uTex");
        if (loc >= 0) glUniform1i(loc, 0);

        glBindVertexArray(m_vao);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glBindVertexArray(0);
        glUseProgram(0);

        // Restore GL state.
        glBlendFuncSeparate(srcRGB, dstRGB, srcA, dstA);
        if (!blendWas) glDisable(GL_BLEND);
        if (depthWas) glEnable(GL_DEPTH_TEST);
    }

    // Cached volume pointers (owned by PipeFluidScene; not copied).
    const std::vector<float>*   m_density = nullptr;
    const std::vector<float>*   m_temp    = nullptr;
    const std::vector<uint8_t>* m_solid   = nullptr;
    int m_nx = 0, m_ny = 0, m_nz = 0;

    // GL objects.
    GLuint m_prog = 0;
    GLuint m_vao = 0, m_vbo = 0;
    GLuint m_tex = 0;
    int    m_texW = 0, m_texH = 0;
    std::vector<uint8_t> m_rgba;
};

} // namespace

// ---- Factory hook used by volume_renderer.cpp (separate file) --------------
std::unique_ptr<VolumeOverlayRenderer> makeCpuVolumeRenderer() {
    return std::unique_ptr<VolumeOverlayRenderer>(new CpuVolumeRenderer());
}

} // namespace pipe_fluid
