#include "pipe_fluid/pipe_boundary_field.h"

#include "pipe_network.h"
#include "pipe_segment.h"
#include "voxelizer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace pipe_fluid {
namespace {

constexpr float kInf = 1.0e30f;
constexpr float kEps = 1.0e-6f;

inline bool inBounds(int i, int j, int k, int nx, int ny, int nz) {
    return i >= 0 && i < nx && j >= 0 && j < ny && k >= 0 && k < nz;
}

struct NeighborStep { int di, dj, dk; float cost; };

std::vector<NeighborStep> makeChamferMask(bool forward) {
    std::vector<NeighborStep> steps;
    for (int dk = -1; dk <= 1; ++dk) for (int dj = -1; dj <= 1; ++dj) for (int di = -1; di <= 1; ++di) {
        if (di == 0 && dj == 0 && dk == 0) continue;
        const bool earlier = (dk < 0) || (dk == 0 && dj < 0) || (dk == 0 && dj == 0 && di < 0);
        if (earlier != forward) continue;
        steps.push_back({di, dj, dk, std::sqrt(float(di*di + dj*dj + dk*dk))});
    }
    return steps;
}

void chamferSweep(std::vector<float>& dist, int nx, int ny, int nz, float dx) {
    const auto forward = makeChamferMask(true);
    const auto backward = makeChamferMask(false);
    for (int k = 0; k < nz; ++k) for (int j = 0; j < ny; ++j) for (int i = 0; i < nx; ++i) {
        const int idx = i + nx * (j + ny * k);
        float best = dist[idx];
        for (const auto& s : forward) {
            const int ni = i + s.di, nj = j + s.dj, nk = k + s.dk;
            if (!inBounds(ni, nj, nk, nx, ny, nz)) continue;
            best = std::min(best, dist[ni + nx * (nj + ny * nk)] + s.cost * dx);
        }
        dist[idx] = best;
    }
    for (int k = nz - 1; k >= 0; --k) for (int j = ny - 1; j >= 0; --j) for (int i = nx - 1; i >= 0; --i) {
        const int idx = i + nx * (j + ny * k);
        float best = dist[idx];
        for (const auto& s : backward) {
            const int ni = i + s.di, nj = j + s.dj, nk = k + s.dk;
            if (!inBounds(ni, nj, nk, nx, ny, nz)) continue;
            best = std::min(best, dist[ni + nx * (nj + ny * nk)] + s.cost * dx);
        }
        dist[idx] = best;
    }
}

bool cellIsWall(PipeBoundaryCell c) { return c == PipeBoundaryCell::Wall; }

bool hasOppositeNeighbor(const std::vector<PipeBoundaryCell>& cells, int nx, int ny, int nz, int i, int j, int k, bool wantWall) {
    const int idx = i + nx * (j + ny * k);
    if (cellIsWall(cells[idx]) != wantWall) return false;
    for (int dk = -1; dk <= 1; ++dk) for (int dj = -1; dj <= 1; ++dj) for (int di = -1; di <= 1; ++di) {
        if (di == 0 && dj == 0 && dk == 0) continue;
        const int ni = i + di, nj = j + dj, nk = k + dk;
        if (!inBounds(ni, nj, nk, nx, ny, nz)) continue;
        if (cellIsWall(cells[ni + nx * (nj + ny * nk)]) != wantWall) return true;
    }
    return false;
}

std::vector<float> buildSignedWallDistance(const std::vector<PipeBoundaryCell>& cells, int nx, int ny, int nz, float dx) {
    const std::size_t total = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz);
    std::vector<float> inside(total, kInf), outside(total, kInf);
    for (int k = 0; k < nz; ++k) for (int j = 0; j < ny; ++j) for (int i = 0; i < nx; ++i) {
        const std::size_t idx = static_cast<std::size_t>(i) + static_cast<std::size_t>(nx) * (static_cast<std::size_t>(j) + static_cast<std::size_t>(ny) * static_cast<std::size_t>(k));
        const bool isWall = cellIsWall(cells[idx]);
        if (hasOppositeNeighbor(cells, nx, ny, nz, i, j, k, isWall)) {
            if (isWall) inside[idx] = 0.5f * dx; else outside[idx] = 0.5f * dx;
        }
    }
    chamferSweep(inside, nx, ny, nz, dx);
    chamferSweep(outside, nx, ny, nz, dx);
    std::vector<float> sdf(total, 0.0f);
    for (std::size_t idx = 0; idx < total; ++idx) {
        // Patch F: this field is the "wall-only" signed distance used by the
        // face-open / cut-cell machinery.  Only Wall cells are negative; both
        // Interior and Exterior (plus Opening) carry positive phi so the MAC
        // solver sees exterior↔exterior faces as fully open and can run real
        // pressure projection outside the pipe shell.  Confinement uses a
        // separate "interior-signed" distance field (see
        // buildInteriorSignedDistance below) that keeps the Patch D property
        // of having grad(phi) point toward the pipe interior on both sides
        // of the wall.
        const bool isWall = cellIsWall(cells[idx]);
        sdf[idx] = isWall ? -inside[idx] : outside[idx];
        if (!std::isfinite(sdf[idx])) sdf[idx] = isWall ? -0.5f * dx : 0.5f * dx;
    }
    return sdf;
}

// Patch F: second SDF signed the Patch-D way — both Wall and Exterior cells
// are negative, only Interior/Opening are positive.  Used exclusively by the
// particle confinement pass so that grad(phi) always points from outside-the-
// pipe toward inside-the-pipe.  Keeping this separate from wallSdf means
// solidFractionOnEdge (which interprets negative phi as "solid") does not see
// Exterior cells as solid and therefore does not close Exterior↔Exterior faces.
std::vector<float> buildInteriorSignedDistance(const std::vector<PipeBoundaryCell>& cells, int nx, int ny, int nz, float dx) {
    const std::size_t total = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz);
    std::vector<float> inside(total, kInf), outside(total, kInf);
    for (int k = 0; k < nz; ++k) for (int j = 0; j < ny; ++j) for (int i = 0; i < nx; ++i) {
        const std::size_t idx = static_cast<std::size_t>(i) + static_cast<std::size_t>(nx) * (static_cast<std::size_t>(j) + static_cast<std::size_t>(ny) * static_cast<std::size_t>(k));
        const bool isWall = cellIsWall(cells[idx]);
        if (hasOppositeNeighbor(cells, nx, ny, nz, i, j, k, isWall)) {
            if (isWall) inside[idx] = 0.5f * dx; else outside[idx] = 0.5f * dx;
        }
    }
    chamferSweep(inside, nx, ny, nz, dx);
    chamferSweep(outside, nx, ny, nz, dx);
    std::vector<float> sdf(total, 0.0f);
    for (std::size_t idx = 0; idx < total; ++idx) {
        const PipeBoundaryCell c = cells[idx];
        const bool isWall     = (c == PipeBoundaryCell::Wall);
        const bool isExterior = (c == PipeBoundaryCell::Exterior);
        if (isWall)          sdf[idx] = -inside[idx];
        else if (isExterior) sdf[idx] = -outside[idx];
        else                 sdf[idx] =  outside[idx];
        if (!std::isfinite(sdf[idx])) sdf[idx] = (isWall || isExterior) ? -0.5f * dx : 0.5f * dx;
    }
    return sdf;
}

struct NearestPipeSample {
    float radialDistance = kInf;
    float innerRadius = 0.0f;
    float outerRadius = 0.0f;
};

NearestPipeSample sampleStraight(const StraightSegment& seg, const Vec3& p) {
    const float len = std::max(seg.length, kEps);
    const float proj = std::clamp((p - seg.origin).dot(seg.direction), 0.0f, len);
    const Vec3 q = seg.origin + seg.direction * proj;
    return {(p - q).length(), seg.innerRadius, seg.outerRadius};
}

NearestPipeSample sampleBend(const BendSegment& seg, const Vec3& p) {
    Vec3 rel = p - seg.centre;
    Vec3 plane = rel - seg.normal * rel.dot(seg.normal);
    if (plane.lengthSq() <= kEps * kEps) plane = seg.startDir; else plane = plane.normalized();
    const float sinTheta = seg.normal.dot(seg.startDir.cross(plane));
    const float cosTheta = std::clamp(seg.startDir.dot(plane), -1.0f, 1.0f);
    float theta = std::atan2(sinTheta, cosTheta);
    theta = (seg.angleRad >= 0.0f) ? std::clamp(theta, 0.0f, seg.angleRad) : std::clamp(theta, seg.angleRad, 0.0f);
    const float cosT = std::cos(theta), sinT = std::sin(theta);
    const Vec3 rotated = seg.startDir * cosT + seg.normal.cross(seg.startDir) * sinT + seg.normal * (seg.normal.dot(seg.startDir) * (1.0f - cosT));
    const Vec3 q = seg.centre + rotated * seg.bendRadius;
    return {(p - q).length(), seg.innerRadius, seg.outerRadius};
}

NearestPipeSample sampleNearestPipe(const PipeNetwork& network, const Vec3& p) {
    NearestPipeSample best;
    for (const auto& segPtr : network.segments) {
        if (!segPtr) continue;
        NearestPipeSample cand = (segPtr->type == SegmentType::Straight)
            ? sampleStraight(static_cast<const StraightSegment&>(*segPtr), p)
            : sampleBend(static_cast<const BendSegment&>(*segPtr), p);
        if (cand.radialDistance < best.radialDistance) best = cand;
    }
    return best;
}

bool terminalConnectsIntoPipe(const PipeNetwork& network, const PipeNetwork::OpenEnd& terminal, float dx) {
    const Vec3 probe = terminal.position + terminal.outwardTangent * (0.75f * dx);
    const NearestPipeSample near = sampleNearestPipe(network, probe);
    return std::isfinite(near.radialDistance) && near.radialDistance <= near.outerRadius + 0.5f * dx;
}

void applyTerminalSemantics(const PipeNetwork& network, const VoxelGrid& voxels, PipeBoundaryField& field) {
    const float dx = voxels.dx;
    const float halfDx = 0.5f * dx;
    for (const auto& end : network.openEnds()) {
        if (terminalConnectsIntoPipe(network, end, dx)) continue;

        const Vec3 E = end.position;
        const Vec3 T = end.outwardTangent.normalized();
        const float innerR = end.innerRadius;

        // Keep the opening region tightly localized to the actual terminal mouth.
        // The previous logic carved a long cylindrical channel plus a spherical
        // cap using the outer radius, which could erode wall cells around the
        // pipe lip and create side-adjacent "open" regions near bends/outlets.
        const float mouthLen = std::max(1.5f * dx, std::min(2.5f * dx, innerR + 0.5f * dx));
        const float capHalfThickness = 0.75f * dx;
        const float openingR = std::max(0.0f, innerR - 0.25f * dx);
        const float capOpenR = innerR + 0.5f * dx;
        const float reach = std::max(capOpenR, mouthLen) + dx;

        const int i0 = std::max(0, (int)std::floor((E.x - reach - voxels.origin.x) / dx));
        const int j0 = std::max(0, (int)std::floor((E.y - reach - voxels.origin.y) / dx));
        const int k0 = std::max(0, (int)std::floor((E.z - reach - voxels.origin.z) / dx));
        const int i1 = std::min(voxels.nx - 1, (int)std::ceil((E.x + reach - voxels.origin.x) / dx));
        const int j1 = std::min(voxels.ny - 1, (int)std::ceil((E.y + reach - voxels.origin.y) / dx));
        const int k1 = std::min(voxels.nz - 1, (int)std::ceil((E.z + reach - voxels.origin.z) / dx));

        for (int k = k0; k <= k1; ++k) for (int j = j0; j <= j1; ++j) for (int i = i0; i <= i1; ++i) {
            const int idx = field.idx(i, j, k);
            const Vec3 p = field.cellCenter(i, j, k);
            const Vec3 v = p - E;
            const float a = v.dot(T);
            const float radial = (v - T * a).length();

            // Only clear the terminal cap itself, and only within the bore.
            // This prevents removing wall cells around the outside lip.
            if (std::fabs(a) <= capHalfThickness && radial <= capOpenR) {
                if (field.cells[idx] == PipeBoundaryCell::Wall) {
                    field.cells[idx] = PipeBoundaryCell::Exterior;
                    field.wallMask[idx] = 0u;
                }
            }

            // Mark a short region just outside the true mouth as Opening, but do
            // not let the opening radius extend into the wall shell.
            if (a > 0.0f && a <= mouthLen && radial <= openingR) {
                field.cells[idx] = PipeBoundaryCell::Opening;
                field.wallMask[idx] = 0u;
            }
        }

        field.terminals.push_back({end.position, end.outwardTangent, end.innerRadius, end.outerRadius});
    }
}

float clamp01(float x) {
    return std::max(0.0f, std::min(1.0f, x));
}

float solidFractionOnEdge(float phiA, float phiB) {
    const bool aSolid = phiA < 0.0f;
    const bool bSolid = phiB < 0.0f;
    if (aSolid && bSolid) return 1.0f;
    if (!aSolid && !bSolid) return 0.0f;
    const float a = std::fabs(phiA);
    const float b = std::fabs(phiB);
    const float denom = a + b;
    if (denom <= kEps) return 0.5f;
    return aSolid ? clamp01(a / denom) : clamp01(b / denom);
}

float boundaryFaceOpenFraction(float phiA, float phiB) {
    return 1.0f - solidFractionOnEdge(phiA, phiB);
}

void buildFaceOpenFractions(PipeBoundaryField& field) {
    const int nx = field.nx;
    const int ny = field.ny;
    const int nz = field.nz;

    field.uOpen.assign(static_cast<std::size_t>(nx + 1) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz), 1.0f);
    field.vOpen.assign(static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny + 1) * static_cast<std::size_t>(nz), 1.0f);
    field.wOpen.assign(static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz + 1), 1.0f);

    // Patch G: gate interior face-opens by the cell classification.  The
    // chamfer SDF (wallSdf) is built from the cell labels, not from the true
    // pipe geometry, so its sub-voxel magnitudes carry no real geometric
    // information — and using them via solidFractionOnEdge produces ~50 %
    // partial-open faces between Interior and Wall cells, which is exactly
    // the pressure-projection leak path that lets water sheet out through
    // the pipe walls.  A face is open iff neither adjacent cell is Wall.
    // This keeps every Interior↔Interior, Exterior↔Exterior, and
    // Interior↔Opening face fully open (the MAC solver runs normally inside
    // and outside the pipe), while Wall-adjacent faces are hard-closed.
    for (int k = 0; k < nz; ++k) for (int j = 0; j < ny; ++j) for (int i = 0; i <= nx; ++i) {
        float open = 1.0f;
        if (i == 0) {
            open = (field.cells[field.idx(0, j, k)] == PipeBoundaryCell::Wall) ? 0.0f : 1.0f;
        } else if (i == nx) {
            open = (field.cells[field.idx(nx - 1, j, k)] == PipeBoundaryCell::Wall) ? 0.0f : 1.0f;
        } else {
            const bool aWall = field.cells[field.idx(i - 1, j, k)] == PipeBoundaryCell::Wall;
            const bool bWall = field.cells[field.idx(i,     j, k)] == PipeBoundaryCell::Wall;
            open = (aWall || bWall) ? 0.0f : 1.0f;
        }
        field.uOpen[field.uIdx(i, j, k)] = open;
    }

    for (int k = 0; k < nz; ++k) for (int j = 0; j <= ny; ++j) for (int i = 0; i < nx; ++i) {
        float open = 1.0f;
        if (j == 0) {
            open = (field.cells[field.idx(i, 0, k)] == PipeBoundaryCell::Wall) ? 0.0f : 1.0f;
        } else if (j == ny) {
            open = (field.cells[field.idx(i, ny - 1, k)] == PipeBoundaryCell::Wall) ? 0.0f : 1.0f;
        } else {
            const bool aWall = field.cells[field.idx(i, j - 1, k)] == PipeBoundaryCell::Wall;
            const bool bWall = field.cells[field.idx(i, j,     k)] == PipeBoundaryCell::Wall;
            open = (aWall || bWall) ? 0.0f : 1.0f;
        }
        field.vOpen[field.vIdx(i, j, k)] = open;
    }

    for (int k = 0; k <= nz; ++k) for (int j = 0; j < ny; ++j) for (int i = 0; i < nx; ++i) {
        float open = 1.0f;
        if (k == 0) {
            open = (field.cells[field.idx(i, j, 0)] == PipeBoundaryCell::Wall) ? 0.0f : 1.0f;
        } else if (k == nz) {
            open = (field.cells[field.idx(i, j, nz - 1)] == PipeBoundaryCell::Wall) ? 0.0f : 1.0f;
        } else {
            const bool aWall = field.cells[field.idx(i, j, k - 1)] == PipeBoundaryCell::Wall;
            const bool bWall = field.cells[field.idx(i, j, k    )] == PipeBoundaryCell::Wall;
            open = (aWall || bWall) ? 0.0f : 1.0f;
        }
        field.wOpen[field.wIdx(i, j, k)] = open;
    }
}

} // namespace

PipeBoundaryField buildPipeBoundaryField(const PipeNetwork& network, const VoxelGrid& voxels) {
    PipeBoundaryField field;
    field.nx = voxels.nx;
    field.ny = voxels.ny;
    field.nz = voxels.nz;
    field.dx = voxels.dx;
    field.origin = voxels.origin;
    const std::size_t total = static_cast<std::size_t>(voxels.nx) * static_cast<std::size_t>(voxels.ny) * static_cast<std::size_t>(voxels.nz);
    field.cells.resize(total, PipeBoundaryCell::Exterior);
    field.wallMask.assign(total, 0u);
    const float halfDx = 0.5f * voxels.dx;
    for (int k = 0; k < voxels.nz; ++k) for (int j = 0; j < voxels.ny; ++j) for (int i = 0; i < voxels.nx; ++i) {
        const int idx = field.idx(i, j, k);
        const NearestPipeSample near = sampleNearestPipe(network, field.cellCenter(i, j, k));
        PipeBoundaryCell c = PipeBoundaryCell::Exterior;
        if (std::isfinite(near.radialDistance)) {
            if (near.radialDistance <= near.innerRadius + halfDx) c = PipeBoundaryCell::Interior;
            else if (near.radialDistance <= near.outerRadius + halfDx) c = PipeBoundaryCell::Wall;
        }
        field.cells[idx] = c;
        field.wallMask[idx] = (c == PipeBoundaryCell::Wall) ? 1u : 0u;
    }
    applyTerminalSemantics(network, voxels, field);
    field.wallSdf     = buildSignedWallDistance(field.cells, field.nx, field.ny, field.nz, field.dx);
    field.interiorSdf = buildInteriorSignedDistance(field.cells, field.nx, field.ny, field.nz, field.dx);
    buildFaceOpenFractions(field);
    return field;
}

} // namespace pipe_fluid
