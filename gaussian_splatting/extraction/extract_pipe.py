#!/usr/bin/env python3
"""Extract a clean, parametric straight pipe from a noisy Gaussian-splat mesh.

Pipeline:

    load → (interactive or bbox) crop → voxel downsample → statistical
    outlier removal → straight-cylinder fit (PCA + iterative refine) →
    centre-line endpoints → .pipe blueprint + cleaned mesh + inlier PLY

Example
-------
    python extract_pipe.py \\
        --input  /path/to/fuse_post.ply \\
        --output-dir outputs/

Scripted (no viewer):

    python extract_pipe.py \\
        --input  /path/to/fuse_post.ply \\
        --output-dir outputs/ \\
        --bbox   "-0.5,-0.5,0,0.5,0.5,2.0" \\
        --no-interactive
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d

from pipeline import (
    blueprint_writer,
    centerline as centerline_mod,
    crop_interactive,
    cylinder_fit,
    denoise,
    io_ply,
    mesh_refine,
)


def _parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract a straight pipe from a noisy GS reconstruction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, type=Path,
                   help="Input .ply (mesh or point cloud).")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Directory for pipe.pipe / pipe_clean.ply / pipe_inliers.ply.")
    p.add_argument("--name", default="extracted_pipe",
                   help="Blueprint `name` field.")
    p.add_argument("--wall-thickness", type=float, default=0.01,
                   help="Inner radius = outer_radius − wall_thickness (metres).")
    p.add_argument("--voxel-size", type=float, default=0.005,
                   help="Voxel downsample size (metres). 0 disables.")
    p.add_argument("--outlier-nb", type=int, default=20,
                   help="Statistical outlier removal neighbourhood size.")
    p.add_argument("--outlier-std", type=float, default=2.0,
                   help="Statistical outlier removal std-ratio threshold.")
    p.add_argument("--radial-tol", type=float, default=2.0,
                   help="Cylinder-fit refinement radial tolerance (× MAD).")
    p.add_argument("--trim-frac", type=float, default=0.02,
                   help="End-cap axial trim fraction (each side).")
    p.add_argument("--radial-margin", type=float, default=0.02,
                   help="Mesh-crop radial margin around the fit radius (m).")
    p.add_argument("--axial-margin", type=float, default=0.02,
                   help="Mesh-crop axial margin past each endpoint (m).")
    p.add_argument("--smooth-iters", type=int, default=1,
                   help="Laplacian smoothing passes on the cleaned mesh.")
    p.add_argument("--bbox", default=None,
                   help='Non-interactive crop: "x0,y0,z0,x1,y1,z1".')
    p.add_argument("--no-interactive", action="store_true",
                   help="Skip the interactive viewer; requires --bbox.")
    return p.parse_args(argv)


def _print_header(args: argparse.Namespace) -> None:
    print("=" * 72)
    print("  Pipe extraction (v1: single straight pipe)")
    print("=" * 72)
    print(f"  Input:      {args.input}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Voxel:      {args.voxel_size}  m")
    print(f"  Outlier:    k={args.outlier_nb}  std_ratio={args.outlier_std}")
    print(f"  Refine tol: {args.radial_tol} × MAD")
    print(f"  Trim frac:  {args.trim_frac}")
    print(f"  Wall:       {args.wall_thickness}  m")
    print("-" * 72)


def main(argv: Optional[list] = None) -> int:
    args = _parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _print_header(args)

    # 1. Load.
    print("[1/7] Loading PLY ...")
    geom = io_ply.load_mesh_or_points(args.input)
    has_mesh = geom.mesh is not None
    n_pts = len(geom.points.points)
    n_tris = len(geom.mesh.triangles) if has_mesh else 0
    print(f"      vertices={n_pts}  triangles={n_tris}")

    # 2. Crop.
    if args.no_interactive:
        if args.bbox is None:
            print("ERROR: --no-interactive requires --bbox", file=sys.stderr)
            return 2
        bbox_min, bbox_max = crop_interactive.parse_bbox_string(args.bbox)
        print(
            f"[2/7] Cropping (bbox): min={bbox_min.tolist()}  max={bbox_max.tolist()}"
        )
        crop = crop_interactive.crop_by_bbox(geom.points, bbox_min, bbox_max)
    else:
        print("[2/7] Cropping (interactive) ...")
        crop = crop_interactive.crop_interactive(geom.points)

    if len(crop.points.points) < 200:
        print(
            f"ERROR: cropped cloud has only {len(crop.points.points)} points — "
            "crop is probably too tight.",
            file=sys.stderr,
        )
        return 3

    # 3. Denoise.
    print("[3/7] Denoising ...")
    down = denoise.voxel_downsample(crop.points, voxel_size=args.voxel_size)
    clean = denoise.statistical_outlier_removal(
        down,
        nb_neighbors=args.outlier_nb,
        std_ratio=args.outlier_std,
    )
    print(
        f"      voxel={len(down.points)}  after outlier removal={len(clean.points)}"
    )

    # 4. Cylinder fit.
    print("[4/7] Fitting straight cylinder ...")
    pts = np.asarray(clean.points)
    fit = cylinder_fit.fit_straight_cylinder(
        pts,
        radial_tol=args.radial_tol,
    )
    print(
        f"      radius={fit.radius:.4f} m  "
        f"inliers={fit.diagnostics['inliers']}/{fit.diagnostics['input_points']}  "
        f"iters={fit.diagnostics['iters_run']}"
    )
    print(
        f"      axis_dir=[{fit.axis_dir[0]:+.4f}, "
        f"{fit.axis_dir[1]:+.4f}, {fit.axis_dir[2]:+.4f}]"
    )

    # 5. Centre-line endpoints.
    print("[5/7] Deriving centre-line ...")
    cl = centerline_mod.derive_centerline(
        pts[fit.inlier_mask],
        fit.axis_point,
        fit.axis_dir,
        trim_frac=args.trim_frac,
    )
    print(
        f"      length={cl.length:.4f} m  "
        f"start={np.round(cl.start, 4).tolist()}  "
        f"end={np.round(cl.end, 4).tolist()}"
    )

    # 6. Emit .pipe blueprint.
    print("[6/7] Writing .pipe blueprint ...")
    header = (
        f"Auto-generated by gaussian_splatting/extraction/extract_pipe.py\n"
        f"source: {args.input}\n"
        f"fit.radius={fit.radius:.6f} m  "
        f"inliers={fit.diagnostics['inliers']}  "
        f"wall_thickness={args.wall_thickness:.6f} m"
    )
    pipe_path = blueprint_writer.write_straight_pipe(
        args.output_dir / "pipe.pipe",
        centerline=cl,
        outer_radius=fit.radius,
        wall_thickness=args.wall_thickness,
        name=args.name,
        header_comment=header,
    )
    print(f"      {pipe_path}")

    # 7. Cleaned mesh + inlier PLY (visual sanity-check companion).
    print("[7/7] Writing cleaned mesh + inliers ...")
    inlier_pcd = mesh_refine.inliers_to_point_cloud(
        pts,
        fit.inlier_mask,
        colors=np.asarray(clean.colors) if clean.has_colors() else None,
    )
    inlier_path = args.output_dir / "pipe_inliers.ply"
    io_ply.save_point_cloud(inlier_path, inlier_pcd)
    print(f"      {inlier_path}")

    if has_mesh:
        cleaned = mesh_refine.crop_mesh_by_cylinder(
            geom.mesh,
            fit=fit,
            centerline=cl,
            radial_margin=args.radial_margin,
            axial_margin=args.axial_margin,
            smooth_iters=args.smooth_iters,
        )
        clean_path = args.output_dir / "pipe_clean.ply"
        io_ply.save_mesh(clean_path, cleaned)
        print(f"      {clean_path}  ({len(cleaned.triangles)} triangles)")
    else:
        print("      (input was point-cloud-only — skipping pipe_clean.ply)")

    print("-" * 72)
    print("  Done.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
