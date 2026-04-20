#pragma once
#include "../Geometry/vec3.h"
#include <cmath>
#include <algorithm>
#include "../Geometry/math_constants.h"

// ============================================================================
// OrbitCamera: arcball-style camera that orbits a target point.
//
// Controls (matched to how the smoke engine feels):
//   Left-drag   → orbit (yaw / pitch)
//   Right-drag  → pan target
//   Scroll      → zoom
// ============================================================================

struct OrbitCamera {
    Vec3  target  = {0, 0, 0};  // look-at point
    float yawDeg  = 35.f;
    float pitchDeg = 20.f;
    float distance = 3.f;       // metres from target

    float fovYDeg = 45.f;
    float nearZ   = 0.01f;
    float farZ    = 100.f;

    // State for drag
    bool  draggingOrbit = false;
    bool  draggingPan   = false;
    float lastX = 0, lastY = 0;

    // ---- Derived quantities ------------------------------------------------

    Vec3 position() const {
        float yaw   = yawDeg * pipe_math::kPiF / 180.f;
        float pitch = pitchDeg * pipe_math::kPiF / 180.f;
        float r = distance;
        return {
            target.x + r * std::cos(pitch) * std::sin(yaw),
            target.y + r * std::sin(pitch),
            target.z + r * std::cos(pitch) * std::cos(yaw)
        };
    }

    Vec3 forward() const {
        return (target - position()).normalized();
    }

    Vec3 right() const {
        Vec3 f = forward();
        Vec3 up{0, 1, 0};
        return f.cross(up).normalized();
    }

    Vec3 up() const {
        return right().cross(forward()).normalized();
    }

    // ---- Build 4x4 view matrix (column-major, OpenGL convention) -----------
    // Fills 16 floats, col-major: m[col*4 + row]
    void buildViewMatrix(float* m) const {
        Vec3 pos = position();
        Vec3 f   = forward().normalized();
        Vec3 r   = right().normalized();
        Vec3 u   = r.cross(f).normalized();

        // Standard look-at matrix
        m[0]  =  r.x; m[4]  =  r.y; m[8]  =  r.z; m[12] = -r.dot(pos);
        m[1]  =  u.x; m[5]  =  u.y; m[9]  =  u.z; m[13] = -u.dot(pos);
        m[2]  = -f.x; m[6]  = -f.y; m[10] = -f.z; m[14] =  f.dot(pos);
        m[3]  =  0;   m[7]  =  0;   m[11] =  0;   m[15] = 1;
    }

    // Build 4x4 projection matrix (col-major)
    void buildProjMatrix(float* m, float aspect) const {
        float f = 1.f / std::tan(fovYDeg * pipe_math::kPiF / 360.f);
        float nf = 1.f / (nearZ - farZ);

        m[0]  = f / aspect; m[4]  = 0; m[8]  = 0;                     m[12] = 0;
        m[1]  = 0;          m[5]  = f; m[9]  = 0;                     m[13] = 0;
        m[2]  = 0;          m[6]  = 0; m[10] = (farZ + nearZ) * nf;   m[14] = 2*farZ*nearZ*nf;
        m[3]  = 0;          m[7]  = 0; m[11] = -1;                    m[15] = 0;
    }

    // Normal matrix = transpose of inverse of upper-left 3x3 of view
    // For orthonormal view matrix this is just the upper-left 3x3 itself.
    void buildNormalMatrix(float* m3) const {
        float view[16];
        buildViewMatrix(view);
        // Upper-left 3x3 (col-major 4x4 → col-major 3x3)
        m3[0] = view[0]; m3[3] = view[4]; m3[6] = view[8];
        m3[1] = view[1]; m3[4] = view[5]; m3[7] = view[9];
        m3[2] = view[2]; m3[5] = view[6]; m3[8] = view[10];
    }

    // ---- Mouse handlers ----------------------------------------------------

    void onMouseButton(int button, bool pressed, float x, float y) {
        if (button == 0) {  // left
            draggingOrbit = pressed;
            lastX = x; lastY = y;
        }
        if (button == 1) {  // right
            draggingPan = pressed;
            lastX = x; lastY = y;
        }
    }

    void onMouseMove(float x, float y) {
        float dx = x - lastX;
        float dy = y - lastY;
        lastX = x;
        lastY = y;

        if (draggingOrbit) {
            yawDeg   -= dx * 0.4f;
            pitchDeg += dy * 0.4f;
            pitchDeg = std::clamp(pitchDeg, -89.f, 89.f);
        }

        if (draggingPan) {
            float panSpeed = distance * 0.0012f;
            Vec3 r = right();
            Vec3 u = up();
            target -= r * (dx * panSpeed);
            target += u * (dy * panSpeed);
        }
    }

    void onScroll(float delta) {
        distance *= std::exp(-delta * 0.1f);
        distance = std::clamp(distance, 0.1f, 50.f);
    }

    // Auto-fit camera to show a bounding sphere of given radius at given centre.
    void fitToBounds(const Vec3& centre, float radius) {
        target   = centre;
        distance = radius * 2.5f;
        pitchDeg = 20.f;
        yawDeg   = 35.f;
    }
};
