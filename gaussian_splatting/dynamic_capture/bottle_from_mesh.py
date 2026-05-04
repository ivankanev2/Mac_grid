#!/usr/bin/env python3
"""bottle_from_mesh.py — voxelise a 3D mesh into a bottle_solid.bin.

Use case
--------
The 4DGS reconstruction of refractive containers (glass bottles, jars) is
noisy and chunky.  An image-to-3D model (TripoSR / InstantMesh / Hunyuan3D /
Stable3D) trained on Objaverse produces a much cleaner mesh for static
objects.  This script takes such a mesh and voxelises it into the same
``bottle_solid.bin`` format that ``extract_fluid_state_v2.py`` writes, so
the simulator can use it as a drop-in replacement for the captured-Gaussian
bottle.

Pipeline
--------
1. Pick a frame from the source video showing the bottle clearly.
   ``ffmpeg -i videoplayback-3 -vf "select=eq(n\\,0)" -vframes 1 frame0.png``
2. Run TripoSR (or another image-to-3D model) on that frame.
   Output: ``bottle_mesh.obj`` (or .glb / .ply).
3. Run this script to voxelise the mesh into ``bottle_solid.bin``,
   using the captured-states_v2 manifest for the simulator grid params.
4. Replace the old bottle_solid.bin with the new one and run the viewer.

Frame coordinates
-----------------
Image-to-3D models output meshes in their own coordinate convention
(usually y-up, centred at origin, scaled to ~1 unit).  We:
  - Optionally rotate (y-up → z-up via ``--rotate-x-90``, default ON
    because most pre-trained models output y-up).
  - Scale the mesh so its z-extent matches ``--target-height``
    (default: pulled from the manifest's ``bottle_height_target``).
  - Translate so the mesh's bottom sits at world z=0 and its xy centre
    sits at (0, 0) — the same world frame ``extract_fluid_state_v2``
    produces.

Voxelisation
------------
Uses ``trimesh.voxel.creation.voxelize_subdivide`` (or
``mesh.voxelized``) to rasterise the mesh's surface into a 3D boolean
grid at the simulator's cell size ``dx`` (read from the manifest).
Surface cells become solid; interior cells stay air (so the bottle has
a hollow cavity for fluid to pool in).

Optionally fill any internal holes (in case the model produces a
non-watertight mesh) via ``--fill-holes``.

Output
------
``bottle_solid.bin`` in the same binary format as the existing extractor,
so the simulator path is unchanged: load, OR-merge into the water solid
mask, run.

Usage
-----
    python bottle_from_mesh.py \\
        --mesh        bottle_mesh.obj \\
        --manifest    captured_states_v2/manifest.json \\
        --output      captured_states_v2/bottle_solid.bin
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np


# Binary format constants — must match
# pipe_fluid_engine/include/pipe_fluid/fluid_state_loader.h.
_BSL_MAGIC   = 0x42534C31   # 'B','S','L','1' little-endian
_BSL_VERSION = 1


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Voxelise an image-to-3D mesh into a bottle_solid.bin.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mesh", required=True, type=Path,
                   help="Input mesh file (.obj, .glb, .ply, .stl).  Output of "
                        "TripoSR / InstantMesh / Hunyuan3D / similar.")
    p.add_argument("--manifest", required=True, type=Path,
                   help="captured_states_v2/manifest.json — gives us the "
                        "simulator's grid params (nx, ny, nz, dx, origin) "
                        "and the target bottle height.")
    p.add_argument("--output", required=True, type=Path,
                   help="Output bottle_solid.bin path.")
    p.add_argument("--target-height", type=float, default=None,
                   help="Override the bottle's world-frame height (m).  "
                        "Default: bottle_height_target from manifest.")
    p.add_argument("--rotate-x-90", default=True, action="store_true",
                   help="Rotate the mesh -90 deg around X (y-up -> z-up).  "
                        "Default ON because most image-to-3D models output "
                        "y-up.  Pass --no-rotate-x-90 to skip.")
    p.add_argument("--no-rotate-x-90", dest="rotate_x_90",
                   action="store_false",
                   help="Skip the y-up -> z-up rotation.  Use when the input "
                        "mesh is already z-up.")
    p.add_argument("--fill-holes", action="store_true", default=False,
                   help="Fill internal holes in the voxelised mesh.  Useful "
                        "if the image-to-3D model outputs a non-watertight "
                        "shell.  Off by default; enable if the bottle "
                        "appears leaky in the viewer.")
    p.add_argument("--seal-base", action="store_true", default=True,
                   help="Stamp the bottle's xy footprint as solid at z=0 "
                        "(same as extract_fluid_state_v2 does).  Default ON "
                        "because mesh bottoms are sometimes thin and leak.")
    p.add_argument("--no-seal-base", dest="seal_base", action="store_false",
                   help="Skip the base-seal stamp.")
    p.add_argument("--carve-top-cells", type=int, default=0,
                   help="Drop the top N z-slices of the bottle mask AFTER "
                        "voxelisation.  Use this to remove the hallucinated "
                        "cap that image-to-3D models put on cropped objects "
                        "(they assume the input is a complete sealed thing).  "
                        "0 = keep mesh as-is.  Try 2-3 if the bottle's mouth "
                        "appears closed in the viewer.")
    return p.parse_args(argv)


# -----------------------------------------------------------------------------
# Manifest loader
# -----------------------------------------------------------------------------
def _load_manifest(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# -----------------------------------------------------------------------------
# Mesh loading + transformation
# -----------------------------------------------------------------------------
def _load_and_transform_mesh(path: Path,
                              target_height: float,
                              rotate_x_90: bool):
    try:
        import trimesh
    except ImportError as e:
        print("ERROR: trimesh not installed.  Install with:", file=sys.stderr)
        print("    pip install trimesh", file=sys.stderr)
        raise

    print(f"  Loading mesh: {path}")
    mesh = trimesh.load(str(path), force="mesh")
    if not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
        raise ValueError(f"mesh has no vertices: {path}")
    print(f"  vertices: {len(mesh.vertices):>8d}  "
          f"faces: {len(mesh.faces):>8d}")

    # Step 1: rotate y-up -> z-up if needed.
    # +pi/2 around +X maps +y -> +z (right-hand rule), so the bottle's
    # natural top stays at the top of the new frame (correct orientation).
    # An earlier version used -pi/2 here, which inverted the bottle.
    if rotate_x_90:
        rot = trimesh.transformations.rotation_matrix(
            +np.pi / 2.0, [1.0, 0.0, 0.0])
        mesh.apply_transform(rot)

    # Step 2: scale so z-extent matches target_height.
    bbox_min, bbox_max = mesh.bounds
    z_extent = float(bbox_max[2] - bbox_min[2])
    if z_extent <= 1e-9:
        raise ValueError("mesh has zero z-extent after rotation; "
                         "did you forget --rotate-x-90?")
    scale = float(target_height) / z_extent
    mesh.apply_scale(scale)
    print(f"  scaled by {scale:.4f}  (target z-extent = {target_height:.3f} m)")

    # Step 3: translate so bottom at z=0 and xy centred at origin.
    bbox_min, bbox_max = mesh.bounds
    centre_xy = 0.5 * (bbox_min[:2] + bbox_max[:2])
    translation = np.array([-centre_xy[0], -centre_xy[1], -bbox_min[2]],
                           dtype=np.float64)
    mesh.apply_translation(translation)
    bbox_min, bbox_max = mesh.bounds
    print(f"  bbox after transform: "
          f"[{bbox_min[0]:+.3f}, {bbox_min[1]:+.3f}, {bbox_min[2]:+.3f}]"
          f" -> "
          f"[{bbox_max[0]:+.3f}, {bbox_max[1]:+.3f}, {bbox_max[2]:+.3f}]")
    return mesh


# -----------------------------------------------------------------------------
# Voxelisation
# -----------------------------------------------------------------------------
def _voxelise_mesh_into_grid(mesh,
                              nx: int, ny: int, nz: int,
                              dx: float,
                              origin: np.ndarray,
                              fill_holes: bool):
    """Rasterise the mesh's surface into our (nx, ny, nz) grid at cell-centre
    sampling.  Returns a bool array of shape (nx, ny, nz)."""
    import trimesh

    # trimesh.voxelized samples at pitch=dx and returns a VoxelGrid where
    # cells crossed by the mesh surface are True.  Its origin sits at the
    # mesh's bbox min minus pitch/2 (or similar, depending on version).
    print(f"  Voxelising at pitch={dx}...")
    vg = mesh.voxelized(pitch=dx)
    surface = np.asarray(vg.matrix, dtype=bool)
    # trimesh 3.x had vg.origin; trimesh 4.x removed it.  Fall back to bounds[0]
    # (min corner of the voxel grid in world space) for either version.
    if hasattr(vg, "origin"):
        surface_origin = np.asarray(vg.origin, dtype=np.float64)
    else:
        surface_origin = np.asarray(vg.bounds[0], dtype=np.float64)
    print(f"  surface voxels: {int(surface.sum())} cells in "
          f"{surface.shape}")

    # Map surface voxels into our simulator grid.  For each cell of OUR
    # grid, look up whether the corresponding world position falls inside
    # a "True" surface voxel.
    out = np.zeros((nx, ny, nz), dtype=bool)
    sx, sy, sz = surface.shape

    # Build cell-centre world coords as 3 1D arrays — vectorise the lookup.
    ii = np.arange(nx, dtype=np.float64)
    jj = np.arange(ny, dtype=np.float64)
    kk = np.arange(nz, dtype=np.float64)
    wx = origin[0] + (ii + 0.5) * dx
    wy = origin[1] + (jj + 0.5) * dx
    wz = origin[2] + (kk + 0.5) * dx

    # For each axis, compute which surface-voxel index that world coord maps to.
    mi = np.floor((wx - surface_origin[0]) / dx).astype(np.int64)
    mj = np.floor((wy - surface_origin[1]) / dx).astype(np.int64)
    mk = np.floor((wz - surface_origin[2]) / dx).astype(np.int64)

    # Clip out-of-range indices.
    in_x = (mi >= 0) & (mi < sx)
    in_y = (mj >= 0) & (mj < sy)
    in_z = (mk >= 0) & (mk < sz)

    # Triple loop is fine here because nx*ny*nz is small (~30k) and inner
    # body is a single index lookup.
    for i in range(nx):
        if not in_x[i]:
            continue
        for j in range(ny):
            if not in_y[j]:
                continue
            for k in range(nz):
                if not in_z[k]:
                    continue
                if surface[mi[i], mj[j], mk[k]]:
                    out[i, j, k] = True

    n_after_voxelise = int(out.sum())
    print(f"  surface cells in our grid: {n_after_voxelise}")

    if fill_holes:
        try:
            from scipy.ndimage import binary_fill_holes
            print("  filling internal holes (3D)...")
            out = np.asarray(binary_fill_holes(out), dtype=bool)
            print(f"  cells after fill_holes: {int(out.sum())}")
        except ImportError:
            print("  WARNING: scipy unavailable; --fill-holes ignored.",
                  file=sys.stderr)

    return out


# -----------------------------------------------------------------------------
# Base seal — same operation extract_fluid_state_v2 applies
# -----------------------------------------------------------------------------
def _stamp_base_seal(bottle_mask: np.ndarray, dx: float,
                     origin: np.ndarray) -> np.ndarray:
    """Find the bottle's xy bbox and stamp an elliptical disk at z=0
    (k = floor((-origin.z) / dx) approximately) covering the full xy
    cross-section.  Same operation as extract_fluid_state_v2's base seal —
    insurance against a non-watertight mesh bottom."""
    if not bottle_mask.any():
        return bottle_mask
    nx, ny, nz = bottle_mask.shape
    bottle_idx = np.argwhere(bottle_mask)
    x_lo = int(bottle_idx[:, 0].min())
    x_hi = int(bottle_idx[:, 0].max())
    y_lo = int(bottle_idx[:, 1].min())
    y_hi = int(bottle_idx[:, 1].max())
    x_c = 0.5 * (x_lo + x_hi)
    y_c = 0.5 * (y_lo + y_hi)
    x_r = max(1.0, 0.5 * (x_hi - x_lo))
    y_r = max(1.0, 0.5 * (y_hi - y_lo))

    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    base_footprint = (((ii - x_c) / x_r) ** 2
                      + ((jj - y_c) / y_r) ** 2) <= 1.0

    k_floor = int(bottle_idx[:, 2].min())
    n_added = int((base_footprint & ~bottle_mask[:, :, k_floor]).sum())
    bottle_mask[:, :, k_floor] |= base_footprint
    print(f"  base seal: stamped k={k_floor}, +{n_added} new solid cells")
    return bottle_mask


# -----------------------------------------------------------------------------
# Binary writer — same format as extract_fluid_state_v2's _write_bottle_solid_bin
# -----------------------------------------------------------------------------
def _write_bottle_solid_bin(path: Path, mask: np.ndarray, dx: float,
                            origin: np.ndarray) -> None:
    nx, ny, nz = mask.shape
    n_solid = int(mask.sum())
    flat = np.ascontiguousarray(mask.transpose(2, 1, 0)).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack(
            "<IIIII fff fI",
            _BSL_MAGIC, _BSL_VERSION,
            int(nx), int(ny), int(nz),
            float(dx),
            float(origin[0]), float(origin[1]), float(origin[2]),
            n_solid,
        ))
        f.write(flat.tobytes(order="C"))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main(argv=None) -> int:
    args = _parse_args(argv)

    print("=" * 72)
    print("  bottle_from_mesh.py — image-to-3D mesh -> bottle_solid.bin")
    print("=" * 72)

    manifest = _load_manifest(args.manifest)
    grid = manifest["grid"]
    nx = int(grid["nx"]); ny = int(grid["ny"]); nz = int(grid["nz"])
    dx = float(manifest["dx"])
    origin = np.array(
        [float(grid["origin_x"]),
         float(grid["origin_y"]),
         float(grid["origin_z"])],
        dtype=np.float64,
    )
    target_height = (
        float(args.target_height) if args.target_height is not None
        else float(manifest.get("bottle_height_target", 0.15))
    )

    print(f"  Mesh:           {args.mesh}")
    print(f"  Manifest:       {args.manifest}")
    print(f"  Output:         {args.output}")
    print(f"  Grid:           {nx} x {ny} x {nz}  dx={dx}")
    print(f"  Origin:         "
          f"[{origin[0]:+.4f}, {origin[1]:+.4f}, {origin[2]:+.4f}]")
    print(f"  Target height:  {target_height} m")
    print(f"  Rotate y->z:    {args.rotate_x_90}")
    print(f"  Fill holes:     {args.fill_holes}")
    print(f"  Seal base:      {args.seal_base}")
    print(f"  Carve top:      {args.carve_top_cells} cells")
    print("-" * 72)

    print("[1/3] Loading + transforming mesh ...")
    mesh = _load_and_transform_mesh(
        args.mesh, target_height, args.rotate_x_90)

    print("[2/3] Voxelising mesh into simulator grid ...")
    bottle_mask = _voxelise_mesh_into_grid(
        mesh, nx, ny, nz, dx, origin, fill_holes=args.fill_holes)

    if args.seal_base:
        bottle_mask = _stamp_base_seal(bottle_mask, dx, origin)

    if args.carve_top_cells > 0:
        # Find the highest z that has any solid cells, then zero out the
        # top N slices.  This removes the hallucinated cap.
        occupied_z = np.where(bottle_mask.any(axis=(0, 1)))[0]
        if len(occupied_z) > 0:
            k_top = int(occupied_z.max())
            k_carve_lo = k_top - args.carve_top_cells + 1
            n_carved = int(bottle_mask[:, :, k_carve_lo:k_top + 1].sum())
            bottle_mask[:, :, k_carve_lo:k_top + 1] = False
            print(f"  carved top {args.carve_top_cells} z-slices "
                  f"(k={k_carve_lo}..{k_top}): -{n_carved} cells "
                  f"(removes hallucinated cap)")

    n_solid = int(bottle_mask.sum())
    print(f"  bottle solid cells (final): {n_solid} "
          f"({100.0 * n_solid / max(1, nx * ny * nz):.2f}% of grid)")

    print("[3/3] Writing bottle_solid.bin ...")
    _write_bottle_solid_bin(args.output, bottle_mask, dx, origin)
    print(f"  wrote {args.output} ({args.output.stat().st_size} bytes)")

    print("-" * 72)
    print("  Done.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
