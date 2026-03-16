#pragma once

#include <cmath>
#include <cstdint>

namespace water3d_internal {

inline int clampi(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

inline float clampf(float v, float lo, float hi) {
    if (!std::isfinite(v)) return lo;
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

inline float clamp01(float v) {
    return clampf(v, 0.0f, 1.0f);
}

inline uint32_t hash_u32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

inline float rand01(uint32_t& state) {
    state = hash_u32(state);
    return (float)(state & 0x00FFFFFFU) / (float)0x01000000U;
}

}  // namespace water3d_internal
