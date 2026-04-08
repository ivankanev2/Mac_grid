
#define STB_TRUETYPE_IMPLEMENTATION
#include "stb_truetype.h"

#include "text_mask.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

std::vector<uint8_t> rasterizeTextMask(
    const std::string& text,
    const std::string& fontPath,
    int gridW,
    int gridH,
    float verticalCenter,
    float textScale,
    float letterSpacing)
{
    std::vector<uint8_t> mask((size_t)gridW * (size_t)gridH, 0);

    if (text.empty() || gridW <= 0 || gridH <= 0) return mask;

    // Load font file
    FILE* f = fopen(fontPath.c_str(), "rb");
    if (!f) {
        std::fprintf(stderr, "[text_mask] Failed to open font: %s\n", fontPath.c_str());
        return mask;
    }

    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    std::vector<unsigned char> fontData((size_t)fsize);
    fread(fontData.data(), 1, (size_t)fsize, f);
    fclose(f);

    // Init font
    stbtt_fontinfo font;
    if (!stbtt_InitFont(&font, fontData.data(), stbtt_GetFontOffsetForIndex(fontData.data(), 0))) {
        std::fprintf(stderr, "[text_mask] Failed to init font\n");
        return mask;
    }

    // Desired pixel height for the text
    float pixelHeight = textScale * (float)gridH;
    float scale = stbtt_ScaleForPixelHeight(&font, pixelHeight);

    // Extra spacing between letters in pixels
    float extraSpacing = letterSpacing * pixelHeight;

    // Get font vertical metrics
    int ascent, descent, lineGap;
    stbtt_GetFontVMetrics(&font, &ascent, &descent, &lineGap);
    float scaledAscent = (float)ascent * scale;
    float scaledDescent = (float)descent * scale;
    float totalHeight = scaledAscent - scaledDescent;

    // First pass: compute total text width (including extra spacing)
    float totalWidth = 0.0f;
    for (size_t c = 0; c < text.size(); ++c) {
        int advanceWidth, leftSideBearing;
        stbtt_GetCodepointHMetrics(&font, text[c], &advanceWidth, &leftSideBearing);
        totalWidth += (float)advanceWidth * scale;

        // Add kerning + extra spacing if not last character
        if (c + 1 < text.size()) {
            int kern = stbtt_GetCodepointKernAdvance(&font, text[c], text[c + 1]);
            totalWidth += (float)kern * scale + extraSpacing;
        }
    }

    // Compute offsets to center text in the grid
    float xOffset = ((float)gridW - totalWidth) * 0.5f;
    float yCenter = verticalCenter * (float)gridH;
    float yOffset = yCenter - totalHeight * 0.5f;

    // Render into a temporary grayscale buffer (top-down, then we'll flip)
    std::vector<unsigned char> grayscale((size_t)gridW * (size_t)gridH, 0);

    // Baseline in screen coords (top-down) = gridH - (yOffset + scaledAscent)
    float screenBaselineY = (float)gridH - (yOffset + scaledAscent);

    float cursorX = xOffset;

    for (size_t c = 0; c < text.size(); ++c) {
        int x0, y0, x1, y1;
        stbtt_GetCodepointBitmapBox(&font, text[c], scale, scale, &x0, &y0, &x1, &y1);

        int glyphW = x1 - x0;
        int glyphH = y1 - y0;

        if (glyphW > 0 && glyphH > 0) {
            std::vector<unsigned char> glyphBuf((size_t)glyphW * (size_t)glyphH, 0);
            stbtt_MakeCodepointBitmap(&font, glyphBuf.data(), glyphW, glyphH, glyphW, scale, scale, text[c]);

            int destX = (int)(cursorX + 0.5f) + x0;
            int destY = (int)(screenBaselineY + 0.5f) + y0;

            for (int gy = 0; gy < glyphH; ++gy) {
                for (int gx = 0; gx < glyphW; ++gx) {
                    int px = destX + gx;
                    int py = destY + gy;
                    if (px < 0 || px >= gridW || py < 0 || py >= gridH) continue;

                    unsigned char val = glyphBuf[(size_t)(gy * glyphW + gx)];
                    unsigned char& dst = grayscale[(size_t)(py * gridW + px)];
                    if (val > dst) dst = val;
                }
            }
        }

        // Advance cursor
        int advanceWidth, leftSideBearing;
        stbtt_GetCodepointHMetrics(&font, text[c], &advanceWidth, &leftSideBearing);
        cursorX += (float)advanceWidth * scale;

        if (c + 1 < text.size()) {
            int kern = stbtt_GetCodepointKernAdvance(&font, text[c], text[c + 1]);
            cursorX += (float)kern * scale + extraSpacing;
        }
    }

    // Convert grayscale (top-down) to binary mask (bottom-up, matching sim grid)
    for (int j = 0; j < gridH; ++j) {
        for (int i = 0; i < gridW; ++i) {
            int screenRow = gridH - 1 - j;
            unsigned char val = grayscale[(size_t)(screenRow * gridW + i)];
            mask[(size_t)(j * gridW + i)] = (val >= 128) ? 1 : 0;
        }
    }

    int count = 0;
    for (auto v : mask) count += v;
    std::fprintf(stderr, "[text_mask] Rasterized '%s' at %dx%d, textHeight=%.0f px, spacing=%.1f px, %d cells set\n",
                 text.c_str(), gridW, gridH, pixelHeight, extraSpacing, count);

    return mask;
}

std::vector<uint8_t> outlineMask(
    const std::vector<uint8_t>& filled,
    int w,
    int h,
    int thickness)
{
    if ((int)filled.size() != w * h || w <= 0 || h <= 0) return filled;

    // Erode the filled mask by `thickness` iterations (remove border pixels each pass)
    std::vector<uint8_t> eroded = filled;
    std::vector<uint8_t> tmp(eroded.size());

    for (int iter = 0; iter < thickness; ++iter) {
        tmp = eroded;
        for (int j = 0; j < h; ++j) {
            for (int i = 0; i < w; ++i) {
                int id = i + w * j;
                if (!eroded[(size_t)id]) continue;

                // If any 4-neighbor is unset (or out of bounds), erode this pixel
                bool border = false;
                if (i == 0     || !eroded[(size_t)(id - 1)])     border = true;
                if (i == w - 1 || !eroded[(size_t)(id + 1)])     border = true;
                if (j == 0     || !eroded[(size_t)(id - w)])     border = true;
                if (j == h - 1 || !eroded[(size_t)(id + w)])     border = true;

                if (border) tmp[(size_t)id] = 0;
            }
        }
        eroded = tmp;
    }

    // Outline = filled AND NOT eroded
    std::vector<uint8_t> outline(filled.size(), 0);
    for (size_t i = 0; i < filled.size(); ++i) {
        outline[i] = (filled[i] && !eroded[i]) ? 1 : 0;
    }

    int count = 0;
    for (auto v : outline) count += v;
    std::fprintf(stderr, "[text_mask] Outline: thickness=%d, %d cells set\n", thickness, count);

    return outline;
}