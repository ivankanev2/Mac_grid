#pragma once
#include <vector>
#include <cstdint>
#include <string>

// Rasterizes a text string into a binary mask at the given grid resolution.
// The mask is row-major with (0,0) at bottom-left (matching the simulation grid).
// Returns a vector<uint8_t> of size gridW * gridH where 1 = inside text, 0 = outside.
//
// fontPath       : path to a .ttf font file
// textScale      : fraction of gridH used for text height (e.g., 0.15 = 15% of grid height)
// verticalCenter : 0.0-1.0, vertical position of text center (0.5 = middle)
// letterSpacing  : extra spacing between letters, as a fraction of text height (0.0 = default font spacing)
std::vector<uint8_t> rasterizeTextMask(
    const std::string& text,
    const std::string& fontPath,
    int gridW,
    int gridH,
    float verticalCenter = 0.5f,
    float textScale = 0.15f,
    float letterSpacing = 0.0f
);

// Takes a filled mask and returns only the outline (border pixels).
// thickness: how many pixels deep the outline extends inward.
std::vector<uint8_t> outlineMask(
    const std::vector<uint8_t>& filled,
    int w,
    int h,
    int thickness = 2
);