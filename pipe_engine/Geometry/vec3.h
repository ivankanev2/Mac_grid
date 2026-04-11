#pragma once
#include <cmath>
#include <algorithm>

// Minimal 3D vector type for the pipe engine.
// Keeps the engine self-contained without pulling in GLM or Eigen.

struct Vec3 {
    float x = 0.f, y = 0.f, z = 0.f;

    Vec3() = default;
    constexpr Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    Vec3 operator+(const Vec3& b) const { return {x+b.x, y+b.y, z+b.z}; }
    Vec3 operator-(const Vec3& b) const { return {x-b.x, y-b.y, z-b.z}; }
    Vec3 operator*(float s)       const { return {x*s, y*s, z*s}; }
    Vec3 operator/(float s)       const { float inv = 1.f/s; return {x*inv, y*inv, z*inv}; }
    Vec3& operator+=(const Vec3& b) { x+=b.x; y+=b.y; z+=b.z; return *this; }
    Vec3& operator-=(const Vec3& b) { x-=b.x; y-=b.y; z-=b.z; return *this; }
    Vec3& operator*=(float s)       { x*=s; y*=s; z*=s; return *this; }

    float dot(const Vec3& b)   const { return x*b.x + y*b.y + z*b.z; }
    Vec3  cross(const Vec3& b) const {
        return {y*b.z - z*b.y, z*b.x - x*b.z, x*b.y - y*b.x};
    }
    float length()    const { return std::sqrt(x*x + y*y + z*z); }
    float lengthSq()  const { return x*x + y*y + z*z; }
    Vec3  normalized() const {
        float len = length();
        return (len > 1e-12f) ? (*this / len) : Vec3{0,0,0};
    }

    static Vec3 lerp(const Vec3& a, const Vec3& b, float t) {
        return a + (b - a) * t;
    }
};

inline Vec3 operator*(float s, const Vec3& v) { return v * s; }

// Construct an orthonormal frame from a forward direction.
// Returns (tangent, normal, binormal).
inline void buildFrame(const Vec3& forward, Vec3& outNormal, Vec3& outBinormal) {
    Vec3 t = forward.normalized();
    // pick an arbitrary axis not parallel to t
    Vec3 up = (std::abs(t.y) < 0.99f) ? Vec3{0,1,0} : Vec3{1,0,0};
    outBinormal = t.cross(up).normalized();
    outNormal   = outBinormal.cross(t).normalized();
}
