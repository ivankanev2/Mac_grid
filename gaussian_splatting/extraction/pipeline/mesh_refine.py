"""Mesh refinement: crop the original high-res mesh by the fitted cylinder.

The fit gives us an axis + radius + length.  We build a thick cylindrical
shell mask (radial margin outside + axial margin past each cap) and keep
only the vertices that fall inside it, plus the triangles incident to
those vertices.  This preserves the original tessellation quality on the
pipe surface while dropping the floor / splash clutter.

Optionally we run a gentle Laplacian smoothing pass to soften the
remaining crust.  Default is one pass — it's subtle and reversible by
setting ``smooth_iters=0``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import open3d as o3d

from .centerline import Centerline
from .cylinder_fit import CylinderFit


def crop_mesh_by_cylinder(
    mesh: o3d.geometry.TriangleMesh,
    fit: CylinderFit,
    centerline: Centerline,
    radial_margin: float = 0.02,
    axial_margin: float = 0.02,
    smooth_iters: int = 1,
) -> o3d.geometry.TriangleMesh:
    """Keep triangles on the pipe surface; drop everything else.

    Parameters
    ----------
    mesh : o3d.geometry.TriangleMesh
        The ORIGINAL high-res mesh (fuse_post.ply), not the downsampled
        point cloud used for fitting.
    fit, centerline : see :mod:`cylinder_fit`, :mod:`centerline`
    radial_margin : float
        How much thicker than the fit radius the kept shell is, on both
        inside and outside.  Needs to cover the real wall thickness plus
        a little slack for tessellation noise.
    axial_margin : float
        Extension past each endpoint along the axis.  Small — the
        blueprint length uses the trimmed extents anyway.
    smooth_iters : int
        Laplacian smoothing passes to run after cropping.  0 disables.
    """
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    if len(verts) == 0 or len(tris) == 0:
        raise ValueError("Input mesh is empty")

    axis_dir = centerline.direction / (np.linalg.norm(centerline.direction) + 1e-12)
    start = centerline.start
    rel = verts - start
    axial = rel @ axis_dir

    orth = rel - np.outer(axial, axis_dir)
    radial = np.linalg.norm(orth, axis=1)

    r_min = max(0.0, fit.radius - radial_margin)
    r_max = fit.radius + radial_margin

    in_radial = (radial >= r_min) & (radial <= r_max)
    in_axial = (axial >= -axial_margin) & (axial <= centerline.length + axial_margin)
    vertex_keep = in_radial & in_axial

    # Keep a triangle only if all three of its vertices survive — this
    # avoids creating unbounded edges at the crop boundary.
    tri_keep = vertex_keep[tris[:, 0]] & vertex_keep[tris[:, 1]] & vertex_keep[tris[:, 2]]

    new_tris_full = tris[tri_keep]
    if len(new_tris_full) == 0:
        raise RuntimeError(
            "Mesh crop removed all triangles — check radial/axial margins"
        )

    # Remap kept vertices to a compact index space.
    used_vidx = np.unique(new_tris_full.ravel())
    remap = -np.ones(len(verts), dtype=np.int64)
    remap[used_vidx] = np.arange(len(used_vidx))
    new_tris = remap[new_tris_full]
    new_verts = verts[used_vidx]

    cleaned = o3d.geometry.TriangleMesh()
    cleaned.vertices = o3d.utility.Vector3dVector(new_verts)
    cleaned.triangles = o3d.utility.Vector3iVector(new_tris)

    if mesh.has_vertex_colors():
        cols = np.asarray(mesh.vertex_colors)[used_vidx]
        cleaned.vertex_colors = o3d.utility.Vector3dVector(cols)

    cleaned.remove_duplicated_vertices()
    cleaned.remove_duplicated_triangles()
    cleaned.remove_degenerate_triangles()
    cleaned.remove_unreferenced_vertices()

    if smooth_iters > 0:
        cleaned = cleaned.filter_smooth_laplacian(
            number_of_iterations=int(smooth_iters)
        )

    cleaned.compute_vertex_normals()
    return cleaned


def inliers_to_point_cloud(
    points: np.ndarray,
    inlier_mask: np.ndarray,
    colors: Optional[np.ndarray] = None,
) -> o3d.geometry.PointCloud:
    """Build a point cloud of just the cylinder-fit inliers — handy for
    visually sanity-checking the fit before committing to the blueprint.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[inlier_mask])
    if colors is not None and len(colors) == len(points):
        pcd.colors = o3d.utility.Vector3dVector(colors[inlier_mask])
    return pcd
