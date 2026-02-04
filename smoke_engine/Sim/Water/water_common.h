#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>

namespace water_internal {

inline int clampi(int v, int lo, int hi) {
    return std::max(lo, std::min(v, hi));
}

inline float clamp01(float x) {
    if (!std::isfinite(x)) return 0.0f;
    if (x < 0.0f) return 0.0f;
    if (x > 1.0f) return 1.0f;
    return x;
}

inline float clampf(float x, float lo, float hi) {
    if (!std::isfinite(x)) return lo;
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

// Deterministic RNG helpers (internal linkage per TU).
inline std::mt19937& rng() {
    static std::mt19937 gen(1337u);
    return gen;
}

inline float rand01() {
    static std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    return dist(rng());
}

inline float randRange(float lo, float hi) {
    return lo + (hi - lo) * rand01();
}

} 