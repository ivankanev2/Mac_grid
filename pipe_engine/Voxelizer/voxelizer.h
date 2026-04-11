#pragma once
#include "../Geometry/pipe_network.h"
#include "../Geometry/vec3.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

// ============================================================================
// PipeVoxelizer: converts a PipeNetwork into a 3D voxel grid where each cell
//                is marked as SOLID (pipe wall), FLUID (interior), or AIR.
//
// This is the bridge between the pipe geometry and the MAC-grid fluid solver.
// The voxelizer samples the analytical distance field of each pipe segment
// rather than rasterizing the triangle mesh, giving smooth boundaries.
// ============================================================================

enum class VoxelType : uint8_t { Air = 0, Fluid = 1, Solid = 2 };

struct VoxelGrid {
    int nx, ny, nz;
    float dx;              // cell size (metres)
    Vec3  origin;          // world-space position of cell (0,0,0)
    std::vector<VoxelType> cells;

    VoxelGrid() : nx(0), ny(0), nz(0), dx(0.01f) {}

    VoxelGrid(int nx, int ny, int nz, float dx, const Vec3& origin)
        : nx(nx), ny(ny), nz(nz), dx(dx), origin(origin),
          cells(nx * ny * nz, VoxelType::Air) {}

    int idx(int i, int j, int k) const { return i + nx * (j + ny * k); }

    VoxelType& at(int i, int j, int k) { return cells[idx(i, j, k)]; }
    VoxelType  at(int i, int j, int k) const { return cells[idx(i, j, k)]; }

    Vec3 cellCentre(int i, int j, int k) const {
        return origin + Vec3{(i + 0.5f) * dx, (j + 0.5f) * dx, (k + 0.5f) * dx};
    }

    void printStats() const {
        int nAir = 0, nFluid = 0, nSolid = 0;
        for (auto c : cells) {
            if (c == VoxelType::Air)   ++nAir;
            if (c == VoxelType::Fluid) ++nFluid;
            if (c == VoxelType::Solid) ++nSolid;
        }
        std::cout << "[VoxelGrid] " << nx << "x" << ny << "x" << nz
                  << " dx=" << dx << "m"
                  << "  Air=" << nAir << " Fluid=" << nFluid << " Solid=" << nSolid << "\n";
    }
};

class PipeVoxelizer {
public:
    float padding = 0.1f;   // metres of padding around the pipe bounding box
    float cellSize = 0.01f; // metres per voxel

    // Compute the minimum distance from a world-space point to the centre-line
    // of a segment, by sampling at `nSamples` points along the segment.
    static float distToCentreLine(const Vec3& p, const PipeSegment& seg, int nSamples = 64) {
        float bestDist = 1e9f;
        for (int i = 0; i <= nSamples; ++i) {
            float t = (float)i / (float)nSamples;
            Vec3 c = seg.centreLine(t);
            float d = (p - c).length();
            bestDist = std::min(bestDist, d);
        }
        return bestDist;
    }

    VoxelGrid voxelize(const PipeNetwork& network) const {
        // 1. Compute bounding box of the network
        Vec3 bmin{1e9f, 1e9f, 1e9f}, bmax{-1e9f, -1e9f, -1e9f};
        for (auto& seg : network.segments) {
            float maxR = seg->outerRadius;
            int nSamples = std::max(8, (int)(seg->arcLength() / cellSize));
            for (int i = 0; i <= nSamples; ++i) {
                float t = (float)i / (float)nSamples;
                Vec3 p = seg->centreLine(t);
                bmin.x = std::min(bmin.x, p.x - maxR);
                bmin.y = std::min(bmin.y, p.y - maxR);
                bmin.z = std::min(bmin.z, p.z - maxR);
                bmax.x = std::max(bmax.x, p.x + maxR);
                bmax.y = std::max(bmax.y, p.y + maxR);
                bmax.z = std::max(bmax.z, p.z + maxR);
            }
        }

        // Add padding
        bmin -= Vec3{padding, padding, padding};
        bmax += Vec3{padding, padding, padding};

        int nx = std::max(1, (int)std::ceil((bmax.x - bmin.x) / cellSize));
        int ny = std::max(1, (int)std::ceil((bmax.y - bmin.y) / cellSize));
        int nz = std::max(1, (int)std::ceil((bmax.z - bmin.z) / cellSize));

        VoxelGrid grid(nx, ny, nz, cellSize, bmin);

        // 2. Classify each voxel
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    Vec3 p = grid.cellCentre(i, j, k);

                    float minDist = 1e9f;
                    float innerR = 0, outerR = 0;

                    // Find closest segment
                    for (auto& seg : network.segments) {
                        float d = distToCentreLine(p, *seg);
                        if (d < minDist) {
                            minDist = d;
                            innerR = seg->innerRadius;
                            outerR = seg->outerRadius;
                        }
                    }

                    if (minDist <= innerR) {
                        grid.at(i, j, k) = VoxelType::Fluid;
                    } else if (minDist <= outerR) {
                        grid.at(i, j, k) = VoxelType::Solid;
                    }
                    // else Air (default)
                }
            }
        }

        return grid;
    }
};
