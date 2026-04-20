// ============================================================================
// gpu_volume_renderer.cpp — GPU ray-march of the smoke/water volume.
//
// Strategy:
//   - Upload density, temperature, and solid-mask as three GL_TEXTURE_3D
//     objects. Density and temp use GL_R16F; solid uses GL_R8 (0 or 1).
//   - Draw a fullscreen triangle-strip with depth-test disabled and straight
//     alpha blending enabled. The fragment shader reconstructs a world-space
//     ray for each pixel by unprojecting (invProj * invView * clip), clips
//     it against the voxel AABB, and accumulates front-to-back.
//   - The shader uses the same transfer function as the CPU backend so the
//     two produce visually identical output (modulo float precision).
//
// Requirements: OpenGL 3.2 Core or later (provided by pipe_engine's context).
//               `glTexImage3D` is core since 3.0. GL_R16F is core since 3.0.
// ============================================================================

#include "pipe_fluid/volume_renderer.h"

#include "Renderer/gl_loader.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace pipe_fluid {
namespace {

// ---- Shaders ---------------------------------------------------------------
// The fragment shader does the actual raymarch. Keeping it self-contained
// means a weak CPU can still spawn the shader fine; the GPU will chew
// through it.

static const char* VOL_VERT = R"GLSL(
#version 150 core
in vec2 aPos;
out vec2 vNdc;
void main() {
    vNdc = aPos;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
)GLSL";

static const char* VOL_FRAG = R"GLSL(
#version 150 core

in  vec2 vNdc;
out vec4 fragColor;

// Inverse of proj * view, so we can take NDC -> world.
uniform mat4  uInvPV;
uniform vec3  uCamPos;
uniform vec3  uBoxMin;      // world-space AABB min = voxels.origin
uniform vec3  uBoxMax;      // world-space AABB max = origin + (nx,ny,nz)*dx
uniform vec3  uCellSize;    // (dx, dx, dx)  — kept as vec3 for symmetry

uniform sampler3D uDensity;
uniform sampler3D uTemp;
uniform sampler3D uSolid;
// Narrow-band signed distance field for the water surface.  Values are in
// WORLD UNITS (metres).  Negative inside water, positive outside, clamped
// to +/- uSdfBand.  When uUseSdf == 1 and we're in the water path, the
// shader sphere-traces this field and shades the isosurface at sdf==0
// instead of volume-integrating the density.
uniform sampler3D uWaterSdf;
uniform float     uSdfBand;
uniform int       uUseSdf;

uniform float uAlphaScale;
uniform float uDensityScale;
uniform float uTempStrength;
uniform float uCoreDark;
uniform int   uUseColor;    // 0 = monochrome, 1 = temperature-tinted

// Step size in metres. Set from CPU to ~0.5 * dx.
uniform float uStep;
uniform int   uMaxSteps;

bool intersectAABB(vec3 o, vec3 d, vec3 bmin, vec3 bmax,
                   out float tmin, out float tmax) {
    vec3 invD = 1.0 / d;
    vec3 t0 = (bmin - o) * invD;
    vec3 t1 = (bmax - o) * invD;
    vec3 tl = min(t0, t1);
    vec3 th = max(t0, t1);
    tmin = max(max(tl.x, tl.y), tl.z);
    tmax = min(min(th.x, th.y), th.z);
    return tmax > max(tmin, 0.0);
}

// ---------------------------------------------------------------------------
// SDF helpers
// ---------------------------------------------------------------------------
// Sample the water SDF at a world-space point, returning signed distance
// in metres.  Outside the AABB the SDF is clamped to +uSdfBand so the
// sphere-tracer exits cleanly.
float sampleWaterSdf(vec3 uv) {
    uv = clamp(uv, vec3(0.0), vec3(1.0));
    return texture(uWaterSdf, uv).r;
}

// Central-difference gradient of the SDF at (uv, worldPos).  Returns a
// unit-length normal pointing AWAY from water (the SDF gradient points
// from inside to outside).  Step is one texel in each axis, scaled to
// world units by (bSize / texSize).
vec3 waterSdfNormal(vec3 uv, vec3 texel) {
    float dxp = sampleWaterSdf(uv + vec3(texel.x, 0.0, 0.0));
    float dxm = sampleWaterSdf(uv - vec3(texel.x, 0.0, 0.0));
    float dyp = sampleWaterSdf(uv + vec3(0.0, texel.y, 0.0));
    float dym = sampleWaterSdf(uv - vec3(0.0, texel.y, 0.0));
    float dzp = sampleWaterSdf(uv + vec3(0.0, 0.0, texel.z));
    float dzm = sampleWaterSdf(uv - vec3(0.0, 0.0, texel.z));
    vec3 g = vec3(dxp - dxm, dyp - dym, dzp - dzm);
    float gl = length(g);
    return (gl > 1e-6) ? (g / gl) : vec3(0.0, 1.0, 0.0);
}

void main() {
    // Unproject two NDC points to recover a world ray.
    vec4 pNear4 = uInvPV * vec4(vNdc, -1.0, 1.0);
    vec4 pFar4  = uInvPV * vec4(vNdc,  1.0, 1.0);
    vec3 pNear = pNear4.xyz / pNear4.w;
    vec3 pFar  = pFar4.xyz  / pFar4.w;

    vec3 rayOri = uCamPos;
    vec3 rayDir = normalize(pFar - pNear);

    float tEnter, tExit;
    if (!intersectAABB(rayOri, rayDir, uBoxMin, uBoxMax, tEnter, tExit)) {
        discard;
    }

    vec3 bSize = uBoxMax - uBoxMin;
    float t = max(0.0, tEnter);
    vec4 accum = vec4(0.0);
    int steps = 0;
    // firstSurfaceT: t at which the ray first hit a non-trivial density
    // sample.  Used by the water path for a depth-fade term.  Negative =
    // not yet hit.  Mirrors smoke_engine/Renderer/smoke_renderer.cpp:1131.
    float firstSurfaceT = -1.0;
    vec3  lightDir = normalize(vec3(-0.45, 0.72, 0.53));
    vec3  viewLight = normalize(vec3(0.0, 0.0, 1.0));

    // =================================================================
    // WATER SDF SPHERE-TRACE PATH
    // =================================================================
    // When the user requested the water path (uUseColor == 0) and the
    // caller uploaded an SDF (uUseSdf == 1), sphere-trace the SDF and
    // shade the surface analytically.  This replaces the per-voxel
    // density ray-march, which is what produces the "blocky"/"cell-
    // structured" look at pipe scale: density sampling visualises
    // per-cell occupancy, while SDF sphere-tracing visualises the
    // continuous union-of-spheres isosurface of the underlying FLIP
    // particles.  The resulting liquid surface is independent of the
    // simulator's cell size and looks like a real water meniscus.
    if (uUseColor == 0 && uUseSdf == 1) {
        // Precompute the minimum world-space step so we never stall in
        // degenerate SDF regions (e.g. saturated narrow-band).
        float dx = uCellSize.x;
        float stepMin = 0.35 * dx;
        vec3  texel = 1.0 / vec3(textureSize(uWaterSdf, 0));

        float tt = tEnter + 1e-5;
        const int   SDF_MAX_STEPS = 128;
        const float HIT_EPS       = 1e-4;  // metres

        bool  hit     = false;
        vec3  hitPos  = vec3(0.0);
        vec3  hitUv   = vec3(0.0);

        for (int i = 0; i < SDF_MAX_STEPS; ++i) {
            if (tt >= tExit) break;
            vec3 p  = rayOri + rayDir * tt;
            vec3 uv = (p - uBoxMin) / bSize;
            float d = sampleWaterSdf(uv);

            if (d < HIT_EPS) {
                // Crossed the zero isosurface.  Refine with one bisection
                // step between the previous safe t and here.
                float tPrev = max(tt - stepMin, tEnter);
                vec3  pMid  = rayOri + rayDir * (0.5 * (tPrev + tt));
                vec3  uvMid = (pMid - uBoxMin) / bSize;
                float dMid  = sampleWaterSdf(uvMid);
                if (dMid < HIT_EPS) { p = pMid; uv = uvMid; }
                hit = true;
                hitPos = p;
                hitUv  = uv;
                firstSurfaceT = tt;
                break;
            }

            // Sphere-trace step.  Clamp to the narrow-band saturation
            // so saturated "far from water" cells don't cause giant
            // single-step jumps past thin fluid features.
            float safe = max(d, stepMin);
            safe = min(safe, uSdfBand);
            tt += safe;
        }

        if (!hit) discard;

        // ---- Surface shading -----------------------------------------
        vec3 N = waterSdfNormal(hitUv, texel);
        // SDF gradient points AWAY from water (outward); make sure we
        // face the camera for lighting.
        if (dot(N, rayDir) > 0.0) N = -N;

        float ndl  = clamp(dot(N, lightDir), 0.0, 1.0);
        float rim  = pow(clamp(1.0 - abs(dot(N, rayDir)), 0.0, 1.0), 2.2);

        // Blinn-Phong specular using a halfway vector between light
        // and view.  Water needs a tight, bright highlight to read as
        // a liquid surface.
        vec3  halfVec = normalize(lightDir + viewLight);
        float spec    = pow(clamp(dot(N, halfVec), 0.0, 1.0), 64.0);

        // Schlick's Fresnel approximation for water (n1=1, n2=1.33 ->
        // F0 ≈ 0.02).  Boost at grazing angles so the liquid silhouette
        // brightens naturally instead of fading to the background.
        float cosTheta = clamp(1.0 - abs(dot(N, rayDir)), 0.0, 1.0);
        float F0       = 0.02;
        float fresnel  = F0 + (1.0 - F0) * pow(cosTheta, 5.0);

        // Deep-water body colour: a muted blue with a slight green
        // shift to suggest volume absorption without needing a second
        // ray-march through the body.
        vec3 deepColor = vec3(0.04, 0.18, 0.44);
        // Bright surface tint: the thin-film / foam look at crests.
        vec3 sheen     = vec3(0.65, 0.82, 0.95);

        vec3 col = deepColor * (0.55 + 0.45 * ndl)
                 + sheen    * (0.35 * rim + 0.65 * spec) * fresnel
                 + vec3(0.08, 0.12, 0.18) * ndl;

        fragColor = vec4(col, 0.95);
        return;
    }

    // =================================================================
    // LEGACY DENSITY VOLUME-INTEGRATION PATH (smoke, or water fallback
    // when SDF isn't available).
    // =================================================================
    while (t < tExit && accum.a < 0.995 && steps < uMaxSteps) {
        vec3 p = rayOri + rayDir * t;
        vec3 u = (p - uBoxMin) / bSize;

        // The smoke path (uUseColor==1) hard-gates on the nearest-
        // neighbour solid texture; that's safe for smoke because wall
        // silhouettes are a small fraction of the rendered volume.
        //
        // The water path (uUseColor==0), however, was showing cubic
        // wall silhouettes at voxel resolution — the "blocky chunks"
        // visible in the rendered water.  The simulator already zeroes
        // water density in solid cells (MACWater3D::rebuildBorderSolids
        // clears water[id]=0 for every solid cell), so trilinear
        // sampling of uDensity alone produces a smooth ramp from fluid
        // density to 0 across the wall boundary.  Skip the solid gate
        // for water and rely on the density field's own falloff.
        bool inWaterPath = (uUseColor == 0);
        float isSolid = inWaterPath ? 0.0 : texture(uSolid, u).r;
        if (isSolid < 0.5) {
            float d = max(0.0, texture(uDensity, u).r);
            if (d > 1e-4) {
                float T = max(0.0, texture(uTemp, u).r);
                // Sigma multiplier 3.2 for water (matches smoke_engine
                // smoke_renderer.cpp:1119) vs 3.0 for smoke.
                float sigmaMul = inWaterPath ? 3.2 : 3.0;
                float sigma = d * uDensityScale * sigmaMul;
                float aStep = (1.0 - exp(-sigma * uStep * 3.0)) * clamp(uAlphaScale, 0.0, 1.0);

                vec3 col;
                if (uUseColor == 0) {
                    // Water path: gradient-based (N·L + rim + specular +
                    // depth-fade) shading.  Direct port of smoke_engine's
                    // water renderer (smoke_renderer.cpp lines 1122-1144).
                    if (firstSurfaceT < 0.0) firstSurfaceT = t;

                    vec3 texel = 1.0 / vec3(textureSize(uDensity, 0));
                    float gx = texture(uDensity, u + vec3(texel.x, 0.0, 0.0)).r
                             - texture(uDensity, u - vec3(texel.x, 0.0, 0.0)).r;
                    float gy = texture(uDensity, u + vec3(0.0, texel.y, 0.0)).r
                             - texture(uDensity, u - vec3(0.0, texel.y, 0.0)).r;
                    float gz = texture(uDensity, u + vec3(0.0, 0.0, texel.z)).r
                             - texture(uDensity, u - vec3(0.0, 0.0, texel.z)).r;
                    vec3 grad = vec3(gx, gy, gz);
                    float gl = length(grad);
                    vec3 normal = (gl > 1e-6) ? (grad / gl) : vec3(0.0, 0.0, 1.0);
                    if (dot(normal, rayDir) > 0.0) normal = -normal;

                    float ndl  = clamp(dot(normal, lightDir), 0.0, 1.0);
                    float rim  = pow(clamp(1.0 - abs(dot(normal, rayDir)), 0.0, 1.0), 2.0);
                    float spec = pow(clamp(dot(normal, viewLight), 0.0, 1.0), 18.0);

                    // Depth-fade across the thickness of the water body.
                    float denom = max(1e-6, tExit - firstSurfaceT);
                    float depthFade = 0.92 - 0.20 * clamp((t - firstSurfaceT) / denom, 0.0, 1.0);

                    // Dark-theme water palette (smoke_renderer.cpp:1137-1139).
                    col = vec3(
                        (0.07 + 0.10 * ndl + 0.08 * rim) * depthFade + 0.22 * spec,
                        (0.22 + 0.24 * ndl + 0.10 * rim) * depthFade + 0.18 * spec,
                        (0.44 + 0.30 * ndl + 0.12 * rim) * depthFade + 0.20 * spec
                    );
                } else {
                    float tCol = clamp(pow(T, 0.55), 0.0, 1.0);
                    float gray = 0.10 + 0.90 * pow(clamp(d, 0.0, 1.0), 0.55);
                    col = vec3(gray + uTempStrength * tCol * 0.55,
                               gray + uTempStrength * tCol * 0.12,
                               gray * (1.0 - 0.35 * tCol));
                    float ageDark = uCoreDark * clamp(d * d, 0.0, 1.0);
                    float core = 1.0 - ageDark;
                    col *= 0.35 + 0.65 * core;
                }

                float oneMinusA = 1.0 - accum.a;
                accum.rgb += oneMinusA * aStep * col;
                accum.a   += oneMinusA * aStep;
            }
        }
        t += uStep;
        ++steps;
    }

    if (accum.a <= 0.0) discard;
    fragColor = accum;  // pre-multiplied RGB*A
}
)GLSL";

// ---- Helpers --------------------------------------------------------------
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

// Column-major mat4 multiply + inverse (copied shape matches CPU path).
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

// Feature probe: does this GL support GL_TEXTURE_3D + GL_R16F? Called at
// init() and used to let the factory fall back to CPU transparently.
static bool gpuBackendSupported() {
    GLint max3D = 0;
    glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &max3D);
    if (max3D < 128) return false;

    // Minimal sanity: try to create a tiny R16F 3D texture.
    GLuint probe = 0;
    glGenTextures(1, &probe);
    glBindTexture(GL_TEXTURE_3D, probe);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_R16F, 4, 4, 4, 0, GL_RED, GL_FLOAT, nullptr);
    const GLenum err = glGetError();
    glBindTexture(GL_TEXTURE_3D, 0);
    glDeleteTextures(1, &probe);
    return err == GL_NO_ERROR;
}

// ============================================================================
// GPU backend
// ============================================================================
class GpuVolumeRenderer : public VolumeOverlayRenderer {
public:
    bool init() override {
        if (!pipe_gl::ensureLoaded()) return false;
        if (!gpuBackendSupported()) return false;

        m_prog = compileProgram(VOL_VERT, VOL_FRAG);
        if (!m_prog) return false;

        // Cache uniform locations up front.
        m_locInvPV        = glGetUniformLocation(m_prog, "uInvPV");
        m_locCamPos       = glGetUniformLocation(m_prog, "uCamPos");
        m_locBoxMin       = glGetUniformLocation(m_prog, "uBoxMin");
        m_locBoxMax       = glGetUniformLocation(m_prog, "uBoxMax");
        m_locCellSize     = glGetUniformLocation(m_prog, "uCellSize");
        m_locDensity      = glGetUniformLocation(m_prog, "uDensity");
        m_locTemp         = glGetUniformLocation(m_prog, "uTemp");
        m_locSolid        = glGetUniformLocation(m_prog, "uSolid");
        m_locWaterSdf     = glGetUniformLocation(m_prog, "uWaterSdf");
        m_locSdfBand      = glGetUniformLocation(m_prog, "uSdfBand");
        m_locUseSdf       = glGetUniformLocation(m_prog, "uUseSdf");
        m_locAlphaScale   = glGetUniformLocation(m_prog, "uAlphaScale");
        m_locDensityScale = glGetUniformLocation(m_prog, "uDensityScale");
        m_locTempStrength = glGetUniformLocation(m_prog, "uTempStrength");
        m_locCoreDark     = glGetUniformLocation(m_prog, "uCoreDark");
        m_locUseColor     = glGetUniformLocation(m_prog, "uUseColor");
        m_locStep         = glGetUniformLocation(m_prog, "uStep");
        m_locMaxSteps     = glGetUniformLocation(m_prog, "uMaxSteps");

        // Fullscreen triangle strip.
        const float quad[8] = { -1,-1,  1,-1, -1, 1,  1, 1 };
        glGenVertexArrays(1, &m_vao);
        glGenBuffers(1, &m_vbo);
        glBindVertexArray(m_vao);
        glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), nullptr);
        glBindVertexArray(0);

        auto makeVolTex = [](GLuint& tex, GLint internal, GLenum format, GLenum type) {
            glGenTextures(1, &tex);
            glBindTexture(GL_TEXTURE_3D, tex);
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
            // Seed with a 1x1x1 texture so samplers are complete even if
            // setVolume() hasn't been called yet.
            const float zero[1] = {0.0f};
            (void)internal; (void)format; (void)type;
            glTexImage3D(GL_TEXTURE_3D, 0, internal, 1, 1, 1, 0, format, type, zero);
            glBindTexture(GL_TEXTURE_3D, 0);
        };
        makeVolTex(m_texDensity, GL_R16F, GL_RED, GL_FLOAT);
        makeVolTex(m_texTemp,    GL_R16F, GL_RED, GL_FLOAT);
        // Solid: use R8 normalized (0 or 255 → 0.0 or 1.0) so we can sample
        // it through a plain sampler3D in the shader.
        makeVolTex(m_texSolid,   GL_R8,  GL_RED, GL_UNSIGNED_BYTE);
        // Water SDF: signed float in metres, narrow-band.  Use R16F — the
        // typical |value| is bounded by 3*dx (a few centimetres) which is
        // well within R16F's representable range.  Seed with the "far"
        // narrow-band value so ray-marches before the first setWaterSdf()
        // upload cleanly miss.
        makeVolTex(m_texWaterSdf, GL_R16F, GL_RED, GL_FLOAT);

        return true;
    }

    void setVolume(const std::vector<float>& density,
                   const std::vector<float>& temp,
                   const std::vector<uint8_t>& solid,
                   int nx, int ny, int nz) override {
        if (nx <= 0 || ny <= 0 || nz <= 0) return;
        const size_t expected = (size_t)nx * ny * nz;
        if (density.size() != expected) return;

        const bool reshape = (nx != m_nx || ny != m_ny || nz != m_nz);
        m_nx = nx; m_ny = ny; m_nz = nz;

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

        // Density
        glBindTexture(GL_TEXTURE_3D, m_texDensity);
        if (reshape) {
            glTexImage3D(GL_TEXTURE_3D, 0, GL_R16F, nx, ny, nz, 0,
                         GL_RED, GL_FLOAT, density.data());
        } else {
            glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, nx, ny, nz,
                            GL_RED, GL_FLOAT, density.data());
        }

        // Temperature (may be empty if caller only has smoke with no temp)
        glBindTexture(GL_TEXTURE_3D, m_texTemp);
        if (temp.size() == expected) {
            if (reshape) {
                glTexImage3D(GL_TEXTURE_3D, 0, GL_R16F, nx, ny, nz, 0,
                             GL_RED, GL_FLOAT, temp.data());
            } else {
                glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, nx, ny, nz,
                                GL_RED, GL_FLOAT, temp.data());
            }
        } else if (reshape) {
            // Zero-fill temperature if the sim doesn't provide one.
            std::vector<float> zeros(expected, 0.0f);
            glTexImage3D(GL_TEXTURE_3D, 0, GL_R16F, nx, ny, nz, 0,
                         GL_RED, GL_FLOAT, zeros.data());
        }

        // Solid mask
        glBindTexture(GL_TEXTURE_3D, m_texSolid);
        if (solid.size() == expected) {
            if (reshape) {
                glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, nx, ny, nz, 0,
                             GL_RED, GL_UNSIGNED_BYTE, solid.data());
            } else {
                glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, nx, ny, nz,
                                GL_RED, GL_UNSIGNED_BYTE, solid.data());
            }
        } else if (reshape) {
            std::vector<uint8_t> zeros(expected, 0u);
            glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, nx, ny, nz, 0,
                         GL_RED, GL_UNSIGNED_BYTE, zeros.data());
        }

        glBindTexture(GL_TEXTURE_3D, 0);
    }

    void setWaterSdf(const std::vector<float>& sdf,
                     int nx, int ny, int nz, float band) override {
        if (nx <= 0 || ny <= 0 || nz <= 0) return;
        const size_t expected = (size_t)nx * ny * nz;

        m_sdfBand = band;
        m_sdfValid = (sdf.size() == expected && band > 0.0f);

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glBindTexture(GL_TEXTURE_3D, m_texWaterSdf);

        const bool reshape = (nx != m_sdfNx || ny != m_sdfNy || nz != m_sdfNz);
        m_sdfNx = nx; m_sdfNy = ny; m_sdfNz = nz;

        if (m_sdfValid) {
            if (reshape) {
                glTexImage3D(GL_TEXTURE_3D, 0, GL_R16F, nx, ny, nz, 0,
                             GL_RED, GL_FLOAT, sdf.data());
            } else {
                glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, nx, ny, nz,
                                GL_RED, GL_FLOAT, sdf.data());
            }
        } else if (reshape) {
            // Seed to "far from water" so a render before any meaningful
            // upload cleanly misses the SDF isosurface.
            const float bandFill = (band > 0.0f) ? band : 1.0f;
            std::vector<float> fill(expected, bandFill);
            glTexImage3D(GL_TEXTURE_3D, 0, GL_R16F, nx, ny, nz, 0,
                         GL_RED, GL_FLOAT, fill.data());
        }

        glBindTexture(GL_TEXTURE_3D, 0);
    }

    void render(const VolumeView& V, const VolumeSettings& S) override {
        if (m_nx <= 1 || m_ny <= 1 || m_nz <= 1) return;
        if (V.fbWidth <= 0 || V.fbHeight <= 0) return;

        float PV[16], invPV[16];
        mat4Mul(V.proj, V.view, PV);
        if (!mat4Inverse(PV, invPV)) return;

        // Save & set GL state for the compositing pass.
        GLboolean depthWas = glIsEnabled(GL_DEPTH_TEST);
        GLboolean blendWas = glIsEnabled(GL_BLEND);
        GLint srcRGB=0, dstRGB=0, srcA=0, dstA=0;
        glGetIntegerv(GL_BLEND_SRC_RGB, &srcRGB);
        glGetIntegerv(GL_BLEND_DST_RGB, &dstRGB);
        glGetIntegerv(GL_BLEND_SRC_ALPHA, &srcA);
        glGetIntegerv(GL_BLEND_DST_ALPHA, &dstA);

        glDisable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFuncSeparate(GL_ONE, GL_ONE_MINUS_SRC_ALPHA,
                            GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
        glViewport(0, 0, V.fbWidth, V.fbHeight);

        glUseProgram(m_prog);
        glUniformMatrix4fv(m_locInvPV, 1, GL_FALSE, invPV);
        glUniform3f(m_locCamPos, V.camPosX, V.camPosY, V.camPosZ);
        glUniform3f(m_locBoxMin, V.originX, V.originY, V.originZ);
        glUniform3f(m_locBoxMax,
                    V.originX + V.nx * V.dx,
                    V.originY + V.ny * V.dx,
                    V.originZ + V.nz * V.dx);
        glUniform3f(m_locCellSize, V.dx, V.dx, V.dx);
        glUniform1f(m_locAlphaScale,   S.alphaScale);
        glUniform1f(m_locDensityScale, S.densityScale);
        glUniform1f(m_locTempStrength, S.tempStrength);
        glUniform1f(m_locCoreDark,     S.coreDark);
        glUniform1i(m_locUseColor,     S.useColor ? 1 : 0);
        // Step sized to ~half a voxel in world metres; same convention as CPU.
        glUniform1f(m_locStep,     0.5f * V.dx);
        glUniform1i(m_locMaxSteps, std::max(16, S.stepsPerPixel) * 4);

        // SDF uniforms.  Enable the sphere-tracer when the caller has
        // provided a valid SDF AND the settings request it AND we're in
        // the water path (useColor==false).  Otherwise fall back to the
        // legacy density volume-integration path.
        const bool sdfEnabled = S.useSdf && m_sdfValid && !S.useColor;
        glUniform1i(m_locUseSdf,   sdfEnabled ? 1 : 0);
        glUniform1f(m_locSdfBand,  m_sdfBand > 0.f ? m_sdfBand : (3.f * V.dx));

        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_3D, m_texDensity);
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_3D, m_texTemp);
        glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_3D, m_texSolid);
        glActiveTexture(GL_TEXTURE3); glBindTexture(GL_TEXTURE_3D, m_texWaterSdf);
        glUniform1i(m_locDensity,  0);
        glUniform1i(m_locTemp,     1);
        glUniform1i(m_locSolid,    2);
        glUniform1i(m_locWaterSdf, 3);

        glBindVertexArray(m_vao);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glBindVertexArray(0);
        glUseProgram(0);

        // Restore state.
        glBlendFuncSeparate(srcRGB, dstRGB, srcA, dstA);
        if (!blendWas) glDisable(GL_BLEND);
        if (depthWas) glEnable(GL_DEPTH_TEST);
    }

    void shutdown() override {
        if (m_texDensity)  { glDeleteTextures(1, &m_texDensity);  m_texDensity = 0; }
        if (m_texTemp)     { glDeleteTextures(1, &m_texTemp);     m_texTemp = 0; }
        if (m_texSolid)    { glDeleteTextures(1, &m_texSolid);    m_texSolid = 0; }
        if (m_texWaterSdf) { glDeleteTextures(1, &m_texWaterSdf); m_texWaterSdf = 0; }
        if (m_vbo)  { glDeleteBuffers(1, &m_vbo); m_vbo = 0; }
        if (m_vao)  { glDeleteVertexArrays(1, &m_vao); m_vao = 0; }
        if (m_prog) { glDeleteProgram(m_prog); m_prog = 0; }
    }

    Backend     backend()     const override { return Backend::GPU; }
    const char* backendName() const override { return "GPU"; }

private:
    GLuint m_prog = 0;
    GLuint m_vao = 0, m_vbo = 0;
    GLuint m_texDensity = 0, m_texTemp = 0, m_texSolid = 0;
    GLuint m_texWaterSdf = 0;
    int m_nx = 0, m_ny = 0, m_nz = 0;
    int m_sdfNx = 0, m_sdfNy = 0, m_sdfNz = 0;
    float m_sdfBand = 0.0f;
    bool  m_sdfValid = false;

    GLint m_locInvPV = -1, m_locCamPos = -1, m_locBoxMin = -1, m_locBoxMax = -1;
    GLint m_locCellSize = -1;
    GLint m_locDensity = -1, m_locTemp = -1, m_locSolid = -1;
    GLint m_locWaterSdf = -1, m_locSdfBand = -1, m_locUseSdf = -1;
    GLint m_locAlphaScale = -1, m_locDensityScale = -1, m_locTempStrength = -1;
    GLint m_locCoreDark = -1, m_locUseColor = -1;
    GLint m_locStep = -1, m_locMaxSteps = -1;
};

} // namespace

// ---- Factory hook (paired with cpu_volume_renderer.cpp) --------------------
std::unique_ptr<VolumeOverlayRenderer> makeGpuVolumeRenderer() {
    return std::unique_ptr<VolumeOverlayRenderer>(new GpuVolumeRenderer());
}

// ---- Auto-detect helper (shared public factory entry) ----------------------
VolumeOverlayRenderer::Backend pickAutoBackend() {
    // Heuristic: Apple always falls back to CPU for consistency with
    // OpenGL 4.1 Core's conservative 3D texture / R16F support on older
    // integrated GPUs. Users can still force GPU via the UI selector.
#ifdef __APPLE__
    return VolumeOverlayRenderer::Backend::CPU;
#else
    const char* renderer = (const char*)glGetString(GL_RENDERER);
    if (renderer) {
        std::string s = renderer;
        for (auto& c : s) c = (char)std::tolower((unsigned char)c);
        if (s.find("llvmpipe") != std::string::npos ||
            s.find("swiftshader") != std::string::npos ||
            s.find("software") != std::string::npos) {
            return VolumeOverlayRenderer::Backend::CPU;
        }
    }
    return VolumeOverlayRenderer::Backend::GPU;
#endif
}

} // namespace pipe_fluid
