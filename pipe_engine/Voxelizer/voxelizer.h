#pragma once
#include "../Geometry/pipe_network.h"
#include "../Geometry/vec3.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cstdint>

// ============================================================================
// PipeVoxelizer: converts a PipeNetwork into a 3D voxel grid where each cell
//                is marked as SOLID (pipe wall), FLUID (interior), or AIR.
//
// This is the bridge between the pipe geometry and the MAC-grid fluid solver.
// The voxelizer samples the analytical distance field of each pipe segment
// rather than rasterizing the triangle mesh, giving smooth boundaries.
// ============================================================================

// VoxelType
//   Air     : empty space outside the pipe.  Smoke sees this as "vent" when
//             the smoke mask keeps it non-solid; water sees it as solid
//             (sealed) so FLIP particles can't free-fall through voxelizer
//             gaps onto the grid floor.
//   Fluid   : pipe interior - both smoke and water live here.
//   Solid   : pipe wall - never passable.
//   Opening : carved out past an OPEN endpoint by the open-ends post-pass.
//             Treated as passable by BOTH sims (smoke and water), so fluids
//             can physically flow out of the pipe mouth instead of hitting
//             a spherical cap.  This is only produced at open ends, never
//             at interior junctions.
enum class VoxelType : uint8_t { Air = 0, Fluid = 1, Solid = 2, Opening = 3 };

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
        int nAir = 0, nFluid = 0, nSolid = 0, nOpen = 0;
        for (auto c : cells) {
            if (c == VoxelType::Air)     ++nAir;
            if (c == VoxelType::Fluid)   ++nFluid;
            if (c == VoxelType::Solid)   ++nSolid;
            if (c == VoxelType::Opening) ++nOpen;
        }
        std::cout << "[VoxelGrid] " << nx << "x" << ny << "x" << nz
                  << " dx=" << dx << "m"
                  << "  Air=" << nAir << " Fluid=" << nFluid
                  << " Solid=" << nSolid << " Opening=" << nOpen << "\n";
    }
};

class PipeVoxelizer {
public:
    float padding = 0.1f;   // metres of padding around the pipe bounding box
    float cellSize = 0.01f; // metres per voxel

    // Extra padding added specifically to the -gravity side of the pipe
    // bounding box.  This creates a "splash basin" of open air below the
    // pipe exit where water can fall and pool under gravity instead of
    // hitting the grid floor almost immediately.  The value is added on
    // TOP of `padding` in the gravity-down direction only, so the grid
    // doesn't bloat horizontally or upward — only downward.  Default 0
    // preserves the tight bbox behaviour used by pipe_engine; consumers
    // that want a fall basin (e.g. PipeFluidScene) set this to ~0.4 m.
    float gravityPadding = 0.0f;

    // When true, voxelize() runs a post-pass that carves out the spherical
    // wall "caps" that otherwise form at every open endpoint of the network.
    // The cap is replaced with VoxelType::Opening inside a cylindrical exit
    // channel, and with VoxelType::Air in the outer shell region past the
    // endpoint.  This makes pipe mouths behave as open boundaries for both
    // fluids, so smoke can vent and water particles can exit the pipe.
    bool openEnds = true;

    // When true, each open endpoint additionally carves a vertical "fall-
    // through" column of Opening cells downward from the exit channel.
    // Without this, the water mask (which seals the Air pad outside the
    // pipe for safety at bends) traps fluid at the pipe mouth: particles
    // reach the Opening cylinder, then bump into a sealed Air cell and
    // cannot fall under gravity.  The fall column is a simple axis-aligned
    // cylinder in -Y from the exit down to the grid floor, so water can
    // exit the pipe and drop naturally.  Only AIR cells become OPENING
    // here (never Solid, so we don't chew through another pipe's wall).
    bool fallThrough = true;

    // Gravity direction used for the fall column carve.  Must point in
    // the direction water accelerates under gravity.  Defaults to -Y,
    // matching MACWater3D::Params::gravity which is -9.8 along +Y's
    // negative.
    Vec3 gravityDir = Vec3{0.f, -1.f, 0.f};

    // When true, any Air cell that has a Fluid neighbour (26-connected)
    // is promoted to Solid after the base classification.  This closes
    // single-cell gaps in the wall that the distance-field classifier
    // leaves at bends and segment junctions, so water with a permeable
    // Air mask can't spray out of the pipe body.  Cheap: one grid pass.
    bool sealWalls = true;

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

        // Add padding.  Uniform `padding` on all six faces, plus an extra
        // `gravityPadding` shift on the face pointing DOWN the gravity
        // vector so water exiting the pipe has an open basin to fall into
        // rather than slamming into the sealed grid floor after a few cm.
        bmin -= Vec3{padding, padding, padding};
        bmax += Vec3{padding, padding, padding};
        if (gravityPadding > 0.0f) {
            // gravityDir is a unit vector pointing in the acceleration
            // direction (default {0,-1,0}).  Positive components move the
            // MAX face out; negative components move the MIN face out.
            // Each component just scales the extra pad along its axis.
            if (gravityDir.x < 0.0f) bmin.x += gravityDir.x * gravityPadding;
            else if (gravityDir.x > 0.0f) bmax.x += gravityDir.x * gravityPadding;
            if (gravityDir.y < 0.0f) bmin.y += gravityDir.y * gravityPadding;
            else if (gravityDir.y > 0.0f) bmax.y += gravityDir.y * gravityPadding;
            if (gravityDir.z < 0.0f) bmin.z += gravityDir.z * gravityPadding;
            else if (gravityDir.z > 0.0f) bmax.z += gravityDir.z * gravityPadding;
        }

        int nx = std::max(1, (int)std::ceil((bmax.x - bmin.x) / cellSize));
        int ny = std::max(1, (int)std::ceil((bmax.y - bmin.y) / cellSize));
        int nz = std::max(1, (int)std::ceil((bmax.z - bmin.z) / cellSize));

        VoxelGrid grid(nx, ny, nz, cellSize, bmin);

        // 2. Classify each voxel.
        //
        // Wall-rasterization reliability: the distance-to-centre-line test
        // uses a cell's centre, but a cell physically spans sqrt(3)/2*dx
        // from centre to far corner.  If the nominal wall thickness
        // (outerR - innerR) is close to dx — which is the default case
        // here (wall=1cm, dx=1cm) — the one-cell-thick Solid shell can
        // have sub-cell holes wherever the wall surface happens to pass
        // between two neighbouring cell centres on the outer side.  With
        // a permeable-Air water mask, each such hole becomes a visible
        // spray of water escaping the pipe body.
        //
        // Fix: extend the Solid classification to `outerR + wallBuffer`
        // (default wallBuffer = 1*dx).  That turns the wall from a
        // nominal 1-cell shell into a nominal 2-cell shell, eliminating
        // 1-cell aliasing gaps across the entire pipe body.  The buffer
        // cells were Air before, so nothing is lost — they just become
        // part of the sealed wall.  The value matches the cap-shell
        // tolerance `outerR + dx` used by carveOpenEnds(), so the mouth
        // carve still opens the full thickened dome correctly.
        const float wallBuffer = 1.0f * cellSize;
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
                    } else if (minDist <= outerR + wallBuffer) {
                        grid.at(i, j, k) = VoxelType::Solid;
                    }
                    // else Air (default)
                }
            }
        }

        // 2b. Wall-seal pass.  The centre-line distance classifier can leave
        //     single-cell gaps in the wall where a cell sits marginally
        //     outside outerR but is sandwiched between Fluid (inside) and
        //     Air (outside) — typical spots are the outside of bends and
        //     segment junctions where two centre-line arcs join.  With the
        //     water mask now treating Air as passable (so gravity-driven
        //     water can fall past the pipe mouth), those gaps become leaks
        //     that spray fluid through the pipe body.
        //
        //     This pass walks every Air cell and promotes it to Solid if
        //     ANY of its 26 neighbours (6 face + 12 edge + 8 corner) is a
        //     Fluid cell.  26-connectivity is used rather than 6 because a
        //     corner-shared Fluid-Air pair forms a diagonal gap that FLIP
        //     particles can tunnel through during advection.  Scanning one
        //     extra ring ensures the wall is watertight.
        if (sealWalls) {
            sealWallGaps(grid);
        }

        // 3. Open-ends post-pass.  The centre-line distance field extends
        //    isotropically past the endpoints of each leaf segment, which
        //    builds a spherical half-dome of Solid cells that seals the
        //    pipe mouth.  Here we carve that dome out and punch a short
        //    cylindrical exit channel so that both fluids can escape.
        if (openEnds) {
            carveOpenEnds(grid, network);
        }

        return grid;
    }

private:
    // Promote any Air cell that touches a Fluid cell (26-connected) to
    // Solid.  This closes single-cell gaps the centre-line distance
    // classifier leaves in the wall at bends and segment junctions, so
    // a permeable-Air water mask doesn't leak water through the pipe
    // body.  O(nx*ny*nz) with 26 neighbour checks; copies the cells
    // array once to avoid the update racing with the scan.
    void sealWallGaps(VoxelGrid& grid) const {
        const int nx = grid.nx, ny = grid.ny, nz = grid.nz;
        const std::vector<VoxelType> src = grid.cells;  // snapshot
        int nSealed = 0;
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    const int idx = i + nx * (j + ny * k);
                    if (src[idx] != VoxelType::Air) continue;
                    bool touchesFluid = false;
                    for (int dk = -1; dk <= 1 && !touchesFluid; ++dk) {
                        const int kk = k + dk;
                        if (kk < 0 || kk >= nz) continue;
                        for (int dj = -1; dj <= 1 && !touchesFluid; ++dj) {
                            const int jj = j + dj;
                            if (jj < 0 || jj >= ny) continue;
                            for (int di = -1; di <= 1 && !touchesFluid; ++di) {
                                if (di == 0 && dj == 0 && dk == 0) continue;
                                const int ii = i + di;
                                if (ii < 0 || ii >= nx) continue;
                                if (src[ii + nx * (jj + ny * kk)] == VoxelType::Fluid) {
                                    touchesFluid = true;
                                }
                            }
                        }
                    }
                    if (touchesFluid) {
                        grid.cells[idx] = VoxelType::Solid;
                        ++nSealed;
                    }
                }
            }
        }
        if (nSealed > 0) {
            std::cout << "[VoxelGrid] sealWallGaps: promoted "
                      << nSealed << " Air->Solid cells\n";
        }
    }

    // Walk each open endpoint (see PipeNetwork::openEnds()) and, in a small
    // neighbourhood around it, convert spurious wall caps into passable
    // cells.  Three disjoint regions are processed per endpoint:
    //   - Cap shell : sqrt(axial^2 + radial^2) <= outerR + dx AND axial > 0
    //                 Any Solid here becomes Air (remove dome).
    //   - Exit tube : axial in (0 .. channelLen] AND radial <= innerR + dx
    //                 Any Solid/Air here becomes Opening, so both sims have
    //                 a well-defined way to flow out of the pipe mouth.
    //   - Fall col  : vertical cylinder in gravityDir from the exit downward
    //                 to the grid floor.  Only Air becomes Opening here.
    //                 This lets water particles physically fall out of the
    //                 pipe mouth under gravity instead of piling up against
    //                 the sealed Air pad that surrounds the network.
    //                 Carved iff `fallThrough` is true.
    void carveOpenEnds(VoxelGrid& grid, const PipeNetwork& network) const {
        const auto ends = network.openEnds();
        if (ends.empty()) return;

        const float dx = grid.dx;
        const float halfDx = 0.5f * dx;

        for (const auto& oe : ends) {
            const Vec3  E       = oe.position;
            const Vec3  T       = oe.outwardTangent;
            const float innerR  = oe.innerRadius;
            const float outerR  = oe.outerRadius;

            // Interior-junction guard.  For branching networks (T / Y /
            // cross), a chain's endpoint may land inside the fluid core of
            // another chain — e.g. a branch of a T-junction "plugs into"
            // the trunk.  Carving past such an endpoint would chew through
            // the other pipe's wall, so if the endpoint voxel is already
            // Fluid (i.e. we're INSIDE another pipe) we skip it.
            int ei = (int)std::floor((E.x - grid.origin.x) / dx);
            int ej = (int)std::floor((E.y - grid.origin.y) / dx);
            int ek = (int)std::floor((E.z - grid.origin.z) / dx);
            if (ei >= 0 && ej >= 0 && ek >= 0 &&
                ei < grid.nx && ej < grid.ny && ek < grid.nz) {
                if (grid.at(ei, ej, ek) == VoxelType::Fluid) continue;
            }

            // Exit-channel length: at least 3 cells, or enough to clear the
            // voxelizer's spherical cap, whichever is longer.  Water
            // particles need this headroom to actually leave the pipe
            // before hitting the sealed Air region.
            const float channelLen = std::max(3.f * dx, 2.f * innerR);
            const float capR       = outerR + dx;       // cap shell extent

            // World-space AABB around this endpoint tight enough to skip
            // voxels we'll definitely never touch.
            const float reach = std::max(capR, channelLen) + dx;
            Vec3 bmin{ E.x - reach, E.y - reach, E.z - reach };
            Vec3 bmax{ E.x + reach, E.y + reach, E.z + reach };

            int i0 = std::max(0, (int)std::floor((bmin.x - grid.origin.x) / dx));
            int j0 = std::max(0, (int)std::floor((bmin.y - grid.origin.y) / dx));
            int k0 = std::max(0, (int)std::floor((bmin.z - grid.origin.z) / dx));
            int i1 = std::min(grid.nx - 1, (int)std::ceil((bmax.x - grid.origin.x) / dx));
            int j1 = std::min(grid.ny - 1, (int)std::ceil((bmax.y - grid.origin.y) / dx));
            int k1 = std::min(grid.nz - 1, (int)std::ceil((bmax.z - grid.origin.z) / dx));

            for (int k = k0; k <= k1; ++k)
            for (int j = j0; j <= j1; ++j)
            for (int i = i0; i <= i1; ++i) {
                Vec3 p  = grid.cellCentre(i, j, k);
                Vec3 v  = p - E;
                float a = v.dot(T);                // axial (along outward)
                if (a <= 0.f) continue;            // behind the mouth
                Vec3  perp   = v - T * a;
                float radial = perp.length();

                VoxelType& cell = grid.at(i, j, k);

                // Exit channel: short cylinder of passable cells that both
                // sims treat as an open boundary.
                if (a <= channelLen && radial <= innerR + halfDx) {
                    if (cell == VoxelType::Solid || cell == VoxelType::Air) {
                        cell = VoxelType::Opening;
                    }
                    continue;
                }

                // Cap shell: the spherical dome the voxelizer builds around
                // the endpoint due to isotropic centre-line distance.  Any
                // Solid inside this half-ball is spurious.
                float sph = std::sqrt(a * a + radial * radial);
                if (sph <= capR) {
                    if (cell == VoxelType::Solid) {
                        cell = VoxelType::Air;
                    }
                }
            }

            // --- Fall-through column --------------------------------------
            // Extend an Opening column in the gravity direction from the
            // end of the exit channel down to the grid floor, so water
            // particles can exit the pipe mouth and keep falling under
            // gravity.  We never carve through Solid here (that would
            // eat through another pipe's wall); only Air becomes Opening.
            if (!fallThrough) continue;

            // Skip if the gravity dir is degenerate.
            Vec3 g = gravityDir;
            float gl = std::sqrt(g.x*g.x + g.y*g.y + g.z*g.z);
            if (gl < 1e-6f) continue;
            g.x /= gl; g.y /= gl; g.z /= gl;

            // Skip mouths pointing strongly AGAINST gravity (e.g. a pipe
            // that opens upward).  Fluid there can't fall out; instead it
            // would just splash back into the pipe, and carving a column
            // below such a mouth risks intersecting the pipe itself.
            float topScore = T.x*(-g.x) + T.y*(-g.y) + T.z*(-g.z);
            if (topScore > 0.85f) continue;

            // Column axis: the vertical line through the end of the exit
            // channel, heading in the gravity direction.  We parameterise
            // with s >= 0 where s = 0 is the start of the column (sits
            // just past the Opening cylinder's outer lip).
            Vec3 colStart = E + T * channelLen;

            // How far down can we possibly go before leaving the grid?
            // Compute the max s along `g` that stays inside the grid
            // bounding box.
            float sMax = 1e9f;
            auto limitAlongAxis = [&](float axisDir, float pos,
                                      float lo, float hi) {
                if (std::fabs(axisDir) < 1e-6f) return 1e9f;
                // s where pos + axisDir*s = lo or hi
                float sLo = (lo - pos) / axisDir;
                float sHi = (hi - pos) / axisDir;
                // pick the positive one; the other is behind us
                float s = std::max(sLo, sHi);
                return (s > 0.f) ? s : 1e9f;
            };
            sMax = std::min(sMax, limitAlongAxis(g.x, colStart.x,
                                                 grid.origin.x,
                                                 grid.origin.x + grid.nx * dx));
            sMax = std::min(sMax, limitAlongAxis(g.y, colStart.y,
                                                 grid.origin.y,
                                                 grid.origin.y + grid.ny * dx));
            sMax = std::min(sMax, limitAlongAxis(g.z, colStart.z,
                                                 grid.origin.z,
                                                 grid.origin.z + grid.nz * dx));
            if (sMax <= 0.f || sMax > 1e8f) continue;

            // Column radius: WIDENED splash basin.  Previously this matched
            // the exit-channel interior radius (innerR + halfDx), which gave
            // water exiting the pipe mouth only a pipe-width drop chute to
            // fall into — it could not spread laterally, so it pooled as a
            // coherent pressurised blob instead of splashing.  A splash
            // basin of ~3x the outer pipe radius (and at least innerR + 4
            // cells for thin pipes) gives water enough lateral freedom to
            // behave like a real falling jet.  The carve below still refuses
            // to overwrite Solid, so adjacent pipes stay safe in branching
            // networks.
            const float colR  = std::max(3.0f * outerR, innerR + 4.0f * dx);
            const float colR2 = colR * colR;

            // AABB around the whole column for a tight loop.
            Vec3 ce = colStart + g * sMax;
            Vec3 cbmin{
                std::min(colStart.x, ce.x) - colR - dx,
                std::min(colStart.y, ce.y) - colR - dx,
                std::min(colStart.z, ce.z) - colR - dx
            };
            Vec3 cbmax{
                std::max(colStart.x, ce.x) + colR + dx,
                std::max(colStart.y, ce.y) + colR + dx,
                std::max(colStart.z, ce.z) + colR + dx
            };

            int ci0 = std::max(0, (int)std::floor((cbmin.x - grid.origin.x) / dx));
            int cj0 = std::max(0, (int)std::floor((cbmin.y - grid.origin.y) / dx));
            int ck0 = std::max(0, (int)std::floor((cbmin.z - grid.origin.z) / dx));
            int ci1 = std::min(grid.nx - 1, (int)std::ceil((cbmax.x - grid.origin.x) / dx));
            int cj1 = std::min(grid.ny - 1, (int)std::ceil((cbmax.y - grid.origin.y) / dx));
            int ck1 = std::min(grid.nz - 1, (int)std::ceil((cbmax.z - grid.origin.z) / dx));

            for (int k = ck0; k <= ck1; ++k)
            for (int j = cj0; j <= cj1; ++j)
            for (int i = ci0; i <= ci1; ++i) {
                Vec3 p = grid.cellCentre(i, j, k);
                Vec3 rel = p - colStart;
                // Project onto the column axis.
                float s = rel.x*g.x + rel.y*g.y + rel.z*g.z;
                if (s < -halfDx || s > sMax + halfDx) continue;
                // Distance squared to the column axis.
                Vec3 perpCol = rel - Vec3{g.x*s, g.y*s, g.z*s};
                float r2 = perpCol.x*perpCol.x
                         + perpCol.y*perpCol.y
                         + perpCol.z*perpCol.z;
                if (r2 > colR2) continue;

                VoxelType& cell = grid.at(i, j, k);
                // Only carve Air.  Never cut through Solid (other pipes)
                // or overwrite Fluid (the pipe interior).
                if (cell == VoxelType::Air) {
                    cell = VoxelType::Opening;
                }
            }
        }
    }
};
