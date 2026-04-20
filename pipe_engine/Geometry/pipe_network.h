#pragma once
#include "pipe_segment.h"
#include <memory>
#include <utility>
#include <vector>
#include <stdexcept>
#include <iostream>

// ============================================================================
// PipeNetwork: one or more connected chains of pipe segments.
//
// A "chain" is an ordered run of segments where segment i's endPoint matches
// segment i+1's startPoint.  begin() starts a new chain at the specified
// world point; addStraight / addBend extend the currently-active chain.
//
// Multiple chains can exist in the same network (to model T / Y / cross
// junctions, parallel manifolds, etc.).  The voxelizer rasterizes all
// chains into the same VoxelGrid, so where two chains meet they simply
// merge at the fluid level.  Each chain exposes two open endpoints (its
// global start and its global end), which the voxelizer's open-ends
// post-pass uses to carve pipe mouths.
// ============================================================================

struct PipeNetwork {
    std::vector<std::unique_ptr<PipeSegment>> segments;
    std::string name = "pipe_network";

    // Default pipe radii for newly added segments
    float defaultInnerRadius = 0.05f;   // 5 cm
    float defaultOuterRadius = 0.06f;   // 6 cm  (1 cm wall)

    // Partition of `segments` into chains.  Each entry is [first, last]
    // (inclusive) indices into `segments`.  An empty chain (first > last)
    // is allowed transiently between begin() and the first addStraight/
    // addBend call, but is filtered out by openEnds().
    struct ChainRange { int first; int last; };
    std::vector<ChainRange> chainRanges;

    // ---- Builder API -------------------------------------------------------

    // Start the network at a given position heading in `direction`.
    Vec3 cursor;         // current end-point of the active chain
    Vec3 cursorDir;      // current forward direction of the active chain

    // Start a brand-new chain at `startPos` heading in `startDir`.
    // Existing chains are preserved - this is how the caller builds
    // branching topologies (T, Y, cross, manifold).  To start a fresh
    // network, call `segments.clear()` + `chainRanges.clear()` first
    // (the PipeFluidScene::clearNetwork helper does this).
    void begin(const Vec3& startPos, const Vec3& startDir) {
        cursor    = startPos;
        cursorDir = startDir.normalized();
        // Open a new, currently-empty chain.  Its `last` is first-1 until
        // a segment is appended.
        const int nextIdx = (int)segments.size();
        chainRanges.push_back({ nextIdx, nextIdx - 1 });
    }

    // Add a straight segment of the given length (metres) continuing in the
    // current direction.
    void addStraight(float length) {
        ensureActiveChain();
        Vec3 endPt = cursor + cursorDir * length;
        auto seg = std::make_unique<StraightSegment>(cursor, endPt,
                                                     defaultInnerRadius,
                                                     defaultOuterRadius);
        segments.push_back(std::move(seg));
        chainRanges.back().last = (int)segments.size() - 1;
        cursor = endPt;
        // direction unchanged
    }

    // Add a bend that redirects the pipe towards `newDir`.
    // `bendRadius` controls the curvature (larger = gentler turn).
    void addBend(const Vec3& newDir, float bendRadius) {
        ensureActiveChain();
        Vec3 nd = newDir.normalized();
        auto seg = std::make_unique<BendSegment>(
            BendSegment::makeElbow(cursor, cursorDir, nd,
                                   bendRadius,
                                   defaultInnerRadius,
                                   defaultOuterRadius));
        cursor    = seg->endPoint();
        cursorDir = nd;
        segments.push_back(std::move(seg));
        chainRanges.back().last = (int)segments.size() - 1;
    }

    // Add a 90-degree bend toward the given axis direction.
    void addBend90(const Vec3& newDir, float bendRadius = 0.15f) {
        addBend(newDir, bendRadius);
    }

    // ---- Open endpoints ----------------------------------------------------

    // An open endpoint of the network: a pipe mouth where fluid can physically
    // enter or exit.  Each chain contributes two open ends (its start and its
    // end), with `outwardTangent` pointing away from the pipe interior.
    struct OpenEnd {
        Vec3  position;
        Vec3  outwardTangent;   // unit vector pointing OUT of the pipe
        float innerRadius;
        float outerRadius;
    };

    std::vector<OpenEnd> openEnds() const {
        std::vector<OpenEnd> out;
        for (const auto& r : chainRanges) {
            if (r.first > r.last) continue;                   // empty chain
            if (r.first < 0 || r.last >= (int)segments.size()) continue;
            const auto& first = *segments[r.first];
            // Outward at the start of a chain = -tangent(0) (points backwards).
            out.push_back(OpenEnd{
                first.startPoint(),
                Vec3{-first.tangent(0.f).x, -first.tangent(0.f).y, -first.tangent(0.f).z},
                first.innerRadius,
                first.outerRadius});
            const auto& last = *segments[r.last];
            // Outward at the end of a chain = +tangent(1) (points forwards).
            out.push_back(OpenEnd{
                last.endPoint(),
                last.tangent(1.f),
                last.innerRadius,
                last.outerRadius});
        }
        return out;
    }

private:
    // Make sure there is an active chain to append to.  If begin() was never
    // called, implicitly start a chain at the current cursor so that legacy
    // callers that just add segments still work.
    void ensureActiveChain() {
        if (chainRanges.empty()) {
            const int nextIdx = (int)segments.size();
            chainRanges.push_back({ nextIdx, nextIdx - 1 });
        }
    }

public:

    // Reset the network to an empty state.  Clears both segments and the
    // chain partition so a subsequent begin() starts a clean chain list.
    void clear() {
        segments.clear();
        chainRanges.clear();
        cursor    = Vec3{0,0,0};
        cursorDir = Vec3{0,0,1};
    }

    // ---- Query API ---------------------------------------------------------

    size_t numSegments() const { return segments.size(); }
    size_t numChains()   const {
        // Only count chains that actually contain at least one segment.
        size_t n = 0;
        for (const auto& r : chainRanges) if (r.first <= r.last) ++n;
        return n;
    }

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
    //
    // Only WITHIN a chain are consecutive segments expected to share
    // endpoints.  Segments at chain boundaries belong to different chains
    // (e.g. a branching T-junction), so we skip those pairs instead of
    // flagging them as gaps.
    bool validate(float tol = 1e-3f) const {
        // If chainRanges is empty but segments exist (legacy), treat the
        // whole thing as one chain.
        if (chainRanges.empty()) {
            for (size_t i = 1; i < segments.size(); ++i) {
                Vec3 prev = segments[i-1]->endPoint();
                Vec3 curr = segments[i]->startPoint();
                if ((prev - curr).length() > tol) return false;
            }
            return true;
        }
        for (const auto& r : chainRanges) {
            for (int i = r.first + 1; i <= r.last; ++i) {
                Vec3 prev = segments[i-1]->endPoint();
                Vec3 curr = segments[i]->startPoint();
                float gap = (prev - curr).length();
                if (gap > tol) {
                    std::cerr << "[PipeNetwork] Gap of " << gap
                              << " between segment " << (i-1) << " and " << i
                              << " (within a chain)\n";
                    return false;
                }
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
