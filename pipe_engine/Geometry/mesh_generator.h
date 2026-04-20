#pragma once
#include "pipe_network.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include "math_constants.h"

// ============================================================================
// TriMesh: simple indexed triangle mesh suitable for OBJ/STL export and
//          eventual voxelization.
// ============================================================================

struct TriMesh {
    struct Vertex {
        Vec3 pos;
        Vec3 normal;
        float u = 0, v = 0;   // texture coords (u = around circumference, v = along pipe)
    };
    struct Triangle {
        uint32_t a, b, c;
    };

    std::vector<Vertex>   vertices;
    std::vector<Triangle> triangles;

    void clear() { vertices.clear(); triangles.clear(); }

    void reserveRing(int rings, int slices) {
        vertices.reserve(rings * slices);
        triangles.reserve((rings - 1) * slices * 2);
    }

    // Merge another mesh into this one (for combining inner/outer walls + caps).
    void append(const TriMesh& other) {
        uint32_t offset = (uint32_t)vertices.size();
        vertices.insert(vertices.end(), other.vertices.begin(), other.vertices.end());
        for (auto& tri : other.triangles) {
            triangles.push_back({tri.a + offset, tri.b + offset, tri.c + offset});
        }
    }
};

// ============================================================================
// MeshGenerator: sweeps a circular cross-section along a PipeSegment's
//                centre-line to produce a triangle mesh.
// ============================================================================

class MeshGenerator {
public:
    int ringSlices = 32;       // vertices around circumference
    int lengthSamples = 0;     // samples along length (0 = auto from arc length)
    float samplesPerMetre = 40.f;  // used when lengthSamples == 0

    // Generate a tube mesh for one segment at a given radius.
    // flipNormals = true → normals point inward (for inner wall of hollow pipe).
    TriMesh generateTube(const PipeSegment& seg, float radius, bool flipNormals = false) const {
        int rings = lengthSamples;
        if (rings <= 0) {
            rings = std::max(4, (int)(seg.arcLength() * samplesPerMetre));
        }

        TriMesh mesh;
        mesh.reserveRing(rings, ringSlices);

        // Generate vertex rings
        for (int ri = 0; ri < rings; ++ri) {
            float t = (float)ri / (float)(rings - 1);
            Vec3 pos = seg.centreLine(t);
            Vec3 tang = seg.tangent(t);

            Vec3 normal, binormal;
            buildFrame(tang, normal, binormal);

            float vCoord = t;
            for (int si = 0; si < ringSlices; ++si) {
                float theta = 2.f * pipe_math::kPiF * (float)si / (float)ringSlices;
                float cosT = std::cos(theta), sinT = std::sin(theta);

                Vec3 offset = normal * (cosT * radius) + binormal * (sinT * radius);
                Vec3 vertPos = pos + offset;
                Vec3 vertNormal = offset.normalized();
                if (flipNormals) vertNormal = vertNormal * -1.f;

                float uCoord = (float)si / (float)ringSlices;
                mesh.vertices.push_back({vertPos, vertNormal, uCoord, vCoord});
            }
        }

        // Generate triangle indices (quad strips between adjacent rings)
        for (int ri = 0; ri < rings - 1; ++ri) {
            for (int si = 0; si < ringSlices; ++si) {
                uint32_t curr = ri * ringSlices + si;
                uint32_t next = ri * ringSlices + ((si + 1) % ringSlices);
                uint32_t currNext = (ri + 1) * ringSlices + si;
                uint32_t nextNext = (ri + 1) * ringSlices + ((si + 1) % ringSlices);

                if (flipNormals) {
                    mesh.triangles.push_back({curr, currNext, next});
                    mesh.triangles.push_back({next, currNext, nextNext});
                } else {
                    mesh.triangles.push_back({curr, next, currNext});
                    mesh.triangles.push_back({next, nextNext, currNext});
                }
            }
        }

        return mesh;
    }

    // Generate a ring cap (annular disc) at one end of the pipe.
    // `centre`, `tangent` define the plane; inner/outerRadius define the annulus.
    // `facingForward` controls the winding direction.
    TriMesh generateAnnularCap(const Vec3& centre, const Vec3& tang,
                               float innerR, float outerR,
                               bool facingForward) const {
        TriMesh mesh;
        Vec3 normal, binormal;
        buildFrame(tang, normal, binormal);

        // Two rings of vertices: inner and outer
        for (int ring = 0; ring < 2; ++ring) {
            float r = (ring == 0) ? innerR : outerR;
            for (int si = 0; si < ringSlices; ++si) {
                float theta = 2.f * pipe_math::kPiF * (float)si / (float)ringSlices;
                float cosT = std::cos(theta), sinT = std::sin(theta);
                Vec3 offset = normal * (cosT * r) + binormal * (sinT * r);
                Vec3 n = facingForward ? tang : tang * -1.f;
                mesh.vertices.push_back({centre + offset, n, 0, 0});
            }
        }

        // Triangulate the annulus
        for (int si = 0; si < ringSlices; ++si) {
            uint32_t inner0 = si;
            uint32_t inner1 = (si + 1) % ringSlices;
            uint32_t outer0 = ringSlices + si;
            uint32_t outer1 = ringSlices + (si + 1) % ringSlices;

            if (facingForward) {
                mesh.triangles.push_back({inner0, inner1, outer0});
                mesh.triangles.push_back({inner1, outer1, outer0});
            } else {
                mesh.triangles.push_back({inner0, outer0, inner1});
                mesh.triangles.push_back({inner1, outer0, outer1});
            }
        }
        return mesh;
    }

    // Generate a complete pipe mesh for an entire network:
    // outer wall + inner wall + end caps.
    TriMesh generatePipeMesh(const PipeNetwork& network) const {
        TriMesh result;

        for (size_t i = 0; i < network.segments.size(); ++i) {
            const auto& seg = *network.segments[i];

            // Outer wall (normals out)
            TriMesh outer = generateTube(seg, seg.outerRadius, false);
            result.append(outer);

            // Inner wall (normals in, for hollow pipe)
            TriMesh inner = generateTube(seg, seg.innerRadius, true);
            result.append(inner);

            // Start cap (only for the first segment)
            if (i == 0) {
                TriMesh cap = generateAnnularCap(
                    seg.startPoint(), seg.tangent(0.f),
                    seg.innerRadius, seg.outerRadius, false);
                result.append(cap);
            }

            // End cap (only for the last segment)
            if (i == network.segments.size() - 1) {
                TriMesh cap = generateAnnularCap(
                    seg.endPoint(), seg.tangent(1.f),
                    seg.innerRadius, seg.outerRadius, true);
                result.append(cap);
            }
        }

        return result;
    }
};