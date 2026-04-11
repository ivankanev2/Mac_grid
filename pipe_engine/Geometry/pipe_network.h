#pragma once
#include "pipe_segment.h"
#include <memory>
#include <vector>
#include <stdexcept>
#include <iostream>

// ============================================================================
// PipeNetwork: an ordered chain of connected pipe segments.
//
// Segments are stored in order.  Each segment's endPoint() is expected to
// match the next segment's startPoint() (within tolerance).  The builder
// helpers (addStraight, addBend) guarantee this automatically.
// ============================================================================

struct PipeNetwork {
    std::vector<std::unique_ptr<PipeSegment>> segments;
    std::string name = "pipe_network";

    // Default pipe radii for newly added segments
    float defaultInnerRadius = 0.05f;   // 5 cm
    float defaultOuterRadius = 0.06f;   // 6 cm  (1 cm wall)

    // ---- Builder API -------------------------------------------------------

    // Start the network at a given position heading in `direction`.
    Vec3 cursor;         // current end-point
    Vec3 cursorDir;      // current forward direction

    void begin(const Vec3& startPos, const Vec3& startDir) {
        cursor    = startPos;
        cursorDir = startDir.normalized();
    }

    // Add a straight segment of the given length (metres) continuing in the
    // current direction.
    void addStraight(float length) {
        Vec3 endPt = cursor + cursorDir * length;
        auto seg = std::make_unique<StraightSegment>(cursor, endPt,
                                                     defaultInnerRadius,
                                                     defaultOuterRadius);
        segments.push_back(std::move(seg));
        cursor = endPt;
        // direction unchanged
    }

    // Add a bend that redirects the pipe towards `newDir`.
    // `bendRadius` controls the curvature (larger = gentler turn).
    void addBend(const Vec3& newDir, float bendRadius) {
        Vec3 nd = newDir.normalized();
        auto seg = std::make_unique<BendSegment>(
            BendSegment::makeElbow(cursor, cursorDir, nd,
                                   bendRadius,
                                   defaultInnerRadius,
                                   defaultOuterRadius));
        cursor    = seg->endPoint();
        cursorDir = nd;
        segments.push_back(std::move(seg));
    }

    // Add a 90-degree bend toward the given axis direction.
    void addBend90(const Vec3& newDir, float bendRadius = 0.15f) {
        addBend(newDir, bendRadius);
    }

    // ---- Query API ---------------------------------------------------------

    size_t numSegments() const { return segments.size(); }

    float totalLength() const {
        float L = 0.f;
        for (auto& s : segments) L += s->arcLength();
        return L;
    }

    // Global parameter T in [0, numSegments].
    // Integer part selects segment, fractional part is local t.
    Vec3 sampleCentreLine(float T) const {
        int idx = std::min((int)T, (int)segments.size()-1);
        float localT = T - (float)idx;
        return segments[idx]->centreLine(std::clamp(localT, 0.f, 1.f));
    }

    // Validate connectivity (endpoints match within tolerance).
    bool validate(float tol = 1e-3f) const {
        for (size_t i = 1; i < segments.size(); ++i) {
            Vec3 prev = segments[i-1]->endPoint();
            Vec3 curr = segments[i]->startPoint();
            float gap = (prev - curr).length();
            if (gap > tol) {
                std::cerr << "[PipeNetwork] Gap of " << gap
                          << " between segment " << (i-1) << " and " << i << "\n";
                return false;
            }
        }
        return true;
    }

    void printSummary() const {
        std::cout << "PipeNetwork \"" << name << "\": "
                  << segments.size() << " segments, total length "
                  << totalLength() << " m\n";
        for (size_t i = 0; i < segments.size(); ++i) {
            auto& s = *segments[i];
            const char* typeName = (s.type == SegmentType::Straight) ? "Straight" : "Bend";
            Vec3 sp = s.startPoint(), ep = s.endPoint();
            std::cout << "  [" << i << "] " << typeName
                      << "  len=" << s.arcLength()
                      << "  from=(" << sp.x << "," << sp.y << "," << sp.z << ")"
                      << "  to=("   << ep.x << "," << ep.y << "," << ep.z << ")\n";
        }
    }
};
